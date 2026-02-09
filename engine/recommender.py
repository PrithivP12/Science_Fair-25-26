from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from engine.recommender_config import CONFIG, WEIGHTS
from engine.run_state import (
    Mutation,
    canonical_mutation_key,
    detect_st_crossing,
    parse_mutation_list,
    parse_pdb_structure,
    run_key,
    structure_hash_from_file,
    THREE_TO_ONE,
    normalize_resn,
)
from engine.vqe_n5_edge import AA_PROPERTIES
from engine.mutator import apply_mutations_to_pdb, atom_record_hash, load_coords_array

# Conservative substitution libraries
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "Y", "W"}
POLAR = {"S", "T", "N", "Q", "H"}
CHARGED = {"K", "R", "D", "E"}
SMALL = {"A", "G", "S"}
AROMATIC = {"F", "Y", "W", "H"}


def _format_eta(seconds: float) -> str:
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def _resn_to_aa1(resn: str) -> str:
    token = (resn or "").strip().upper()
    if len(token) == 1:
        return token
    return THREE_TO_ONE.get(normalize_resn(token), token[:1] if token else "X")


def _parse_exclusions(do_not_mutate: List[str]) -> Tuple[set, set]:
    """
    Parse exclusions from forms like "50" or "A:75".
    Returns (chain_specific_set, global_resseq_set).
    """
    chain_specific = set()
    global_resseq = set()
    for raw in do_not_mutate or []:
        token = (raw or "").strip()
        if not token:
            continue
        if ":" in token:
            chain, res = token.split(":", 1)
            chain = chain.strip()
            res = res.strip()
            if chain and res:
                chain_specific.add((chain, res))
        else:
            global_resseq.add(token)
    return chain_specific, global_resseq


def _site_sort_key(site_key: Tuple[str, str, str]) -> Tuple[str, int, str]:
    chain, resi, icode = site_key
    try:
        resnum = int(str(resi).strip())
    except Exception:
        resnum = 10**9
    return (chain, resnum, icode)


def _select_cofactor_center(
    residue_coords: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]],
    cofactor_sites: Dict[Tuple[str, str, str], Dict[str, object]],
    radius: float,
) -> np.ndarray | None:
    """
    Pick one representative cofactor site in multi-cofactor structures.
    Preference: highest nearby residue count, then deterministic site order.
    """
    if not cofactor_sites:
        return None
    best_center = None
    best_score = -1
    best_key = None
    for site_key, site in cofactor_sites.items():
        center_raw = site.get("n5") if site.get("n5") is not None else site.get("centroid")
        if center_raw is None:
            continue
        center = np.array(center_raw, dtype=float)
        score = 0
        for coords in residue_coords.values():
            pts = np.array(coords, dtype=float)
            if pts.size == 0:
                continue
            if float(np.min(np.linalg.norm(pts - center, axis=1))) <= radius:
                score += 1
        if (score > best_score) or (score == best_score and (best_key is None or _site_sort_key(site_key) < _site_sort_key(best_key))):
            best_score = score
            best_center = center
            best_key = site_key
    if best_center is not None:
        return best_center
    first_key = sorted(cofactor_sites.keys(), key=_site_sort_key)[0]
    site = cofactor_sites[first_key]
    center_raw = site.get("n5") if site.get("n5") is not None else site.get("centroid")
    if center_raw is None:
        return None
    return np.array(center_raw, dtype=float)


def _candidate_mutation_score(wt_aa: str, mut_aa: str, distance: float, buried: float) -> float:
    """
    Fast ranking heuristic to prioritize candidates before expensive quantum calls.
    """
    wt_props = AA_PROPERTIES.get(wt_aa, {"e_neg": 0.0, "steric_index": 0.7})
    mut_props = AA_PROPERTIES.get(mut_aa, {"e_neg": 0.0, "steric_index": 0.7})
    delta_eneg = float(mut_props.get("e_neg", 0.0)) - float(wt_props.get("e_neg", 0.0))
    delta_steric = float(mut_props.get("steric_index", 0.7)) - float(wt_props.get("steric_index", 0.7))
    aromatic_bonus = 0.45 if mut_aa in AROMATIC else 0.0
    polar_bonus = 0.18 if mut_aa in POLAR else 0.0
    charge_bonus = 0.12 if mut_aa in {"K", "R"} else 0.0
    proximity_bonus = max(0.0, 1.0 - min(distance, 12.0) / 12.0)
    burial_penalty = 0.10 * max(0.0, buried - 8.0)
    steric_penalty = 0.20 * abs(delta_steric)
    return 1.35 * delta_eneg + aromatic_bonus + polar_bonus + charge_bonus + 0.90 * proximity_bonus - steric_penalty - burial_penalty


@dataclass
class Recommendation:
    mutations: List[Mutation]
    total_score: float
    pred_brightness: float
    delta_brightness: float
    synergy: float
    clash_risk: float
    stability_risk: float
    confidence: float
    notes: str
    run_key: str
    heuristic: bool = False
    feature_hash: str | None = None
    st_gap: float | None = None


def residue_positions_within_radius(pdb_path: str, cofactor_resns=None, radius: float = 8.0) -> Tuple[Dict[Tuple[str, str, str], float], bool]:
    if cofactor_resns is None:
        cofactor_resns = {"FMN", "FAD", "RIB", "RBF"}
    residue_coords: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = {}
    cofactor_sites: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resn = line[17:20].strip().upper()
            chain = (line[21] or "?").strip() or "?"
            resi = line[22:26].strip()
            icode = line[26].strip() or " "
            atom = line[12:16].strip().upper()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coord = (x, y, z)
            if resn in cofactor_resns:
                site = cofactor_sites.setdefault((chain, resi, icode), {"coords": [], "n5": None, "centroid": None})
                site["coords"].append(coord)
                if atom == "N5":
                    site["n5"] = coord
            else:
                residue_coords.setdefault((chain, resi, icode), []).append(coord)
    for site in cofactor_sites.values():
        coords = site.get("coords", [])
        site["centroid"] = tuple(np.mean(np.array(coords), axis=0)) if coords else None

    flavin_found = len(cofactor_sites) > 0
    if not flavin_found:
        return {}, False
    flavin_centroid = _select_cofactor_center(residue_coords, cofactor_sites, radius=radius)
    if flavin_centroid is None:
        return {}, False
    dist_map = {}
    for key, pts in residue_coords.items():
        dists = np.linalg.norm(np.array(pts) - flavin_centroid, axis=1)
        dist_map[key] = float(np.min(dists))
    in_radius = {k: v for k, v in dist_map.items() if v <= radius}
    if in_radius:
        return in_radius, True
    # If no residues are within the requested radius, use nearest residues so the
    # recommender can still return ranked suggestions instead of an empty result.
    nearest = dict(sorted(dist_map.items(), key=lambda item: item[1])[:24])
    return nearest, True


def burial_metric(residue_coords: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]], target_key) -> float:
    coords = residue_coords.get(target_key, [])
    if not coords or not isinstance(coords, list) or (coords and not isinstance(coords[0], tuple)):
        return 0.0
    target_pts = np.array(coords)
    if target_pts.size == 0:
        return 0.0
    centroid = target_pts.mean(axis=0)
    neighbor_count = 0
    for key, pts in residue_coords.items():
        if key == target_key or not pts or not isinstance(pts[0], tuple):
            continue
        pts_arr = np.array(pts)
        dists = np.linalg.norm(pts_arr - centroid, axis=1)
        if np.min(dists) < 6.0:
            neighbor_count += 1
    return neighbor_count


def allowed_substitutions(resn: str, buried_score: float) -> List[str]:
    resn = resn.upper()
    choices: List[str] = []
    if buried_score > 6:
        choices = list(HYDROPHOBIC | {"H", "Y"})
    else:
        choices = list(HYDROPHOBIC | POLAR | {"K", "R"})
    if resn == "P":
        choices = [r for r in choices if r != "P"]
    return sorted(set(choices) - {resn})


def conservative_candidate_pool(pdb_path: str, do_not_mutate: List[str], radius: float, second_shell: bool) -> List[Mutation]:
    residue_index, _ = parse_pdb_structure(pdb_path)
    residue_coords: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = {}
    # build coords map to validate presence
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                chain = (line[21] or "?").strip() or "?"
                resi = line[22:26].strip()
                icode = line[26].strip() or " "
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                residue_coords.setdefault((chain, resi, icode), []).append((x, y, z))
    except Exception:
        residue_coords = {}
    dist_primary, flavin_found = residue_positions_within_radius(pdb_path, radius=radius)
    dist_secondary, _ = residue_positions_within_radius(pdb_path, radius=CONFIG.radius_secondary) if second_shell else ({}, True)
    if not flavin_found:
        return []
    excl_chain, excl_global = _parse_exclusions(do_not_mutate)
    max_subs_env = os.environ.get("MAX_MUTATIONS_PER_SITE", "").strip()
    try:
        max_subs_per_site = int(max_subs_env) if max_subs_env else 3
    except ValueError:
        max_subs_per_site = 3
    if max_subs_per_site <= 0:
        max_subs_per_site = 3
    candidate_scores: Dict[str, Tuple[float, Mutation]] = {}

    def register_site_mutations(key: Tuple[str, str, str], dist: float, shell_penalty: float = 0.0) -> None:
        chain, resi, icode = key
        rec = residue_index.get(key) or residue_index.get((chain, resi, " "))
        if not rec:
            return
        if key not in residue_coords:
            return
        resn_full = rec.resn.upper()
        resn = _resn_to_aa1(resn_full)
        if (chain, resi) in excl_chain or resi in excl_global:
            return
        if rec.resn == "CYS" and dist < 4.0:
            return
        buried = burial_metric(residue_coords, key)
        scored: List[Tuple[float, Mutation]] = []
        for mut_aa in allowed_substitutions(resn, buried):
            mutation = Mutation(chain=chain, resseq=resi, icode=icode, wt=resn, mut=mut_aa)
            score = _candidate_mutation_score(resn, mut_aa, dist, buried) - shell_penalty
            scored.append((score, mutation))
        for score, mutation in sorted(scored, key=lambda item: (-item[0], item[1].canonical_token()))[:max_subs_per_site]:
            token = mutation.canonical_token()
            prev = candidate_scores.get(token)
            if prev is None or score > prev[0]:
                candidate_scores[token] = (score, mutation)

    for key, dist in dist_primary.items():
        register_site_mutations(key, dist, shell_penalty=0.0)
    if second_shell:
        for key, dist in dist_secondary.items():
            if key in dist_primary or dist > CONFIG.radius_secondary:
                continue
            register_site_mutations(key, dist, shell_penalty=0.35)

    ranked = sorted(candidate_scores.values(), key=lambda item: (-item[0], item[1].canonical_token()))
    return [mutation for _, mutation in ranked]


def brightness_gain_heuristic(mutation: Mutation, baseline_brightness: float) -> float:
    # Simple heuristic: electrostatic and steric adjustment
    gain = 2.0
    if mutation.mut in {"K", "R"}:
        gain += 1.0
    if mutation.mut in {"F", "Y", "W"}:
        gain += 0.5
    return baseline_brightness + gain


def clash_risk(mutation: Mutation, buried: float) -> float:
    risk = max(0.0, buried - 6.0)
    if mutation.mut in {"W", "Y"}:
        risk += 1.0
    return risk


def stability_risk(mutation: Mutation, buried: float) -> float:
    risk = 0.0
    if buried > 8 and mutation.mut in {"K", "R", "D", "E"}:
        risk += 2.0
    if mutation.wt == "G" and mutation.mut not in SMALL:
        risk += 1.5
    return risk


def _extract_brightness(row: pd.Series, wt_brightness: float | None) -> Tuple[float, float, str | None]:
    """Extract brightness_pred and delta from a row with unit checks.

    Returns (brightness, delta, note_flag) where note_flag marks any derivation.
    """
    brightness_sources = ["pred_brightness", "brightness_pred"]
    brightness_raw = None
    note = None
    for src in brightness_sources:
        if src in row and pd.notna(row[src]):
            try:
                brightness_raw = float(row[src])
                break
            except Exception:
                continue
    # fallback: derive from st_gap if explicit brightness missing
    if brightness_raw is None and "st_gap" in row and pd.notna(row["st_gap"]):
        try:
            gap_ev = float(row["st_gap"])
            if gap_ev > 5.0:
                raise ValueError("UNIT_ST_GAP")
            brightness_raw = max(0.0, (gap_ev / 0.35) * 150.0)
            note = "derived_from_st_gap"
        except Exception:
            brightness_raw = None
    # last-resort fallback: use emission proxy (pred_final) as relative brightness.
    if brightness_raw is None and "pred_final" in row and pd.notna(row["pred_final"]):
        try:
            em_proxy = float(row["pred_final"])
            brightness_raw = -em_proxy if em_proxy < 0 else em_proxy
            note = "emission_proxy"
        except Exception:
            brightness_raw = None
    if brightness_raw is None:
        raise ValueError("BRIGHTNESS_MISSING")
    if brightness_raw < 0:
        # negative brightness is a hard unit bug
        raise ValueError("UNIT_MIXUP_BUG")
    delta = brightness_raw - (wt_brightness if wt_brightness is not None else 0.0)
    return brightness_raw, delta, note


def score_mutation(mutation: Mutation, baseline_brightness: float, buried: float, confidence: float, base_run_key: str, structure_hash: str, cofactor_choice: str) -> Recommendation:
    pred_bright = brightness_gain_heuristic(mutation, baseline_brightness)
    delta = pred_bright - baseline_brightness
    c_risk = clash_risk(mutation, buried)
    s_risk = stability_risk(mutation, buried)
    total = delta - WEIGHTS.clash_lambda * c_risk - WEIGHTS.stability_mu * s_risk + WEIGHTS.confidence_nu * confidence
    run_key_val = run_key(structure_hash, cofactor_choice, [mutation])
    notes = "Conservative substitution; heuristic gain applied."
    return Recommendation(
        mutations=[mutation],
        total_score=total,
        pred_brightness=pred_bright,
        delta_brightness=delta,
        synergy=0.0,
        clash_risk=c_risk,
        stability_risk=s_risk,
        confidence=confidence,
        notes=notes,
        run_key=run_key_val,
        heuristic=True,
    )


def beam_search(
    baseline_brightness: float,
    candidates: List[Recommendation],
    max_size: int,
    beam_width: int,
    confidence: float,
    structure_hash: str,
    cofactor_choice: str,
) -> Dict[int, List[Recommendation]]:
    beams: Dict[int, List[Recommendation]] = {1: sorted(candidates, key=lambda r: (-r.total_score, r.run_key))[:beam_width]}
    for size in range(2, max_size + 1):
        new_states: List[Recommendation] = []
        for parent in beams[size - 1]:
            used_positions = {(m.chain, m.resseq, m.icode) for m in parent.mutations}
            for cand in candidates:
                pos = (cand.mutations[0].chain, cand.mutations[0].resseq, cand.mutations[0].icode)
                if pos in used_positions:
                    continue
                combo_mutations = sorted(parent.mutations + cand.mutations)
                combo_key = canonical_mutation_key(combo_mutations)
                synergy = (parent.delta_brightness + cand.delta_brightness) * 0.1
                pred_bright = baseline_brightness + sum(r.delta_brightness for r in [parent, cand]) + synergy
                delta = pred_bright - baseline_brightness
                clash = parent.clash_risk + cand.clash_risk
                stability = parent.stability_risk + cand.stability_risk
                total = delta - WEIGHTS.clash_lambda * clash - WEIGHTS.stability_mu * stability + WEIGHTS.confidence_nu * confidence + WEIGHTS.synergy_xi * synergy
                rk = run_key(structure_hash, cofactor_choice, combo_mutations)
                new_states.append(
                    Recommendation(
                        mutations=combo_mutations,
                        total_score=total,
                        pred_brightness=pred_bright,
                        delta_brightness=delta,
                        synergy=synergy,
                        clash_risk=clash,
                        stability_risk=stability,
                        confidence=confidence,
                        notes="Synergy heuristic applied.",
                        run_key=rk,
                        heuristic=True,
                    )
                )
        beams[size] = sorted(new_states, key=lambda r: (-r.total_score, r.run_key))[:beam_width]
    return beams


def predict_for_mutations(
    pdb_path: str,
    cofactor_choice: str,
    mutation_tokens: List[str],
    wt_feature_hash: str | None,
    wt_brightness: float | None,
    confidence: float,
    wt_run_key: str | None = None,
    wt_st_gap: float | None = None,
) -> Recommendation | None:
    """Run the full prediction pipeline for a given mutation set and return a Recommendation."""
    # build mutated PDB
    mutated_path = pdb_path
    mut_changed = False
    if mutation_tokens:
        mutated_path, mut_changed = apply_mutations_to_pdb(pdb_path, mutation_tokens)
    engine_mutation_list = "" if mut_changed else ",".join(mutation_tokens)
    env = os.environ.copy()
    env["MUTATION_LIST"] = engine_mutation_list
    # vqe_n5_edge reads COFACTOR_SELECTED; keep legacy key for compatibility.
    env["COFACTOR_SELECTED"] = cofactor_choice
    env["COFACTOR_CONFIGURATION"] = cofactor_choice
    env["PDB_LABEL"] = Path(pdb_path).stem
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "vqe_n5_edge.py"), "--pdb", mutated_path]
    timeout_env = (env.get("RECOMMENDER_VQE_TIMEOUT_SEC", "") or env.get("VQE_TIMEOUT_SEC", "")).strip()
    try:
        timeout_val = float(timeout_env) if timeout_env else 12.0
        if timeout_val <= 0:
            timeout_val = None  # explicit 0 disables timeout
    except ValueError:
        timeout_val = 12.0
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_val, env=env)
    except subprocess.TimeoutExpired:
        print("VQE_TIMEOUT", file=sys.stderr)
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    manifest_run_key = None
    saved_profile_id = None
    for line in proc.stdout.splitlines():
        if line.startswith("RUN_MANIFEST:"):
            try:
                payload = json.loads(line.split("RUN_MANIFEST:", 1)[1].strip())
                manifest_run_key = payload.get("run_key")
            except Exception:
                manifest_run_key = None
        if line.startswith("SUCCESS: Saved "):
            try:
                saved_profile_id = line.split("SUCCESS: Saved ", 1)[1].split(" to ", 1)[0].strip()
            except Exception:
                saved_profile_id = None
    profile_path = Path(os.getcwd()) / "artifacts" / "qc_n5_gpr" / "Final_Quantum_Profiles.csv"
    if not profile_path.exists():
        return None
    try:
        df = pd.read_csv(profile_path, on_bad_lines="skip", engine="python")
    except Exception:
        return None
    rec_df = pd.DataFrame()
    # Prefer exact run-key match from current subprocess.
    if manifest_run_key and "run_key" in df.columns:
        rec_df = df[df["run_key"].astype(str) == str(manifest_run_key)]
    # Fallback: explicit profile id emitted by engine single-run output.
    if rec_df.empty and saved_profile_id and "pdb_id" in df.columns:
        rec_df = df[df["pdb_id"].astype(str) == saved_profile_id]
    # Fallback: mutation + cofactor filter.
    if rec_df.empty and "mutation_list" in df.columns:
        rec_df = df[df["mutation_list"].fillna("").astype(str) == engine_mutation_list]
        if "cofactor_type_used" in rec_df.columns and cofactor_choice:
            rec_df = rec_df[rec_df["cofactor_type_used"].astype(str).str.upper() == cofactor_choice.upper()]
    if rec_df.empty:
        rec_df = df.tail(1)
    row = rec_df.tail(1).iloc[0]
    # canonical brightness extraction: prefer explicit brightness fields; fall back to pred_final (emission proxy)
    pred_bright, delta, note_flag = _extract_brightness(row, wt_brightness)
    clash = float(row.get("clash_penalty", 0.0))
    stability = float(row.get("triad_score", 0.0))
    scs = float(row.get("scs", confidence))
    n5_spin = float(row.get("n5_spin_density", float("nan")))
    st_gap = float(row.get("st_gap", float("nan")))
    feature_hash = row.get("feature_hash", "") or f"runhash:{env['MUTATION_LIST']}"
    # Build Mutation objects from tokens
    muts, _ = parse_mutation_list(",".join(mutation_tokens))
    if mutation_tokens and not mut_changed:
        raise ValueError("mutated_pdb_identical_to_wt_for_targets")
    # Volume delta check (steric index change)
    vol_delta = 0.0
    wt_ref = 0.0
    hybridization_change = False
    for m in muts:
        wt_idx = AA_PROPERTIES.get(m.wt, {}).get("steric_index", 0.0)
        mut_idx = AA_PROPERTIES.get(m.mut, {}).get("steric_index", 0.0)
        vol_delta += (mut_idx - wt_idx)
        wt_ref = max(wt_ref, abs(wt_idx))
        if (m.wt, m.mut) in {("P", "H"), ("P", "Y"), ("I", "H"), ("I", "Y")}:
            hybridization_change = True
    # append st_gap with 6 decimals into feature hash
    feature_hash = f"{feature_hash}|st:{st_gap:.6f}"
    if mutation_tokens and wt_feature_hash and feature_hash == wt_feature_hash:
        if hybridization_change or abs(vol_delta) >= 0.2 * (wt_ref + 1e-3):
            raise ValueError("FEATURE_HASH_COLLISION_WT")
        print(f"WARNING_MUTATION_NO_EFFECT for {mutation_tokens}")
    # Hash collision audit using wavefunction signature if available
    wf_sig = row.get("wavefunction_signature", None)
    if wf_sig is not None:
        feature_hash = feature_hash + f"|wf:{wf_sig}"
    st_gap_val = float(row.get("st_gap", float("nan")))
    # st_gap delta vs WT triggers new hash metadata
    if wt_st_gap is not None and pd.notna(st_gap_val) and abs(st_gap_val - wt_st_gap) > 0.005:
        feature_hash = f"{feature_hash}|stshift"
    coords_arr = load_coords_array(mutated_path)
    metadata = f"{env.get('MUTATION_LIST','')}"
    rk = run_key(coords_arr, cofactor_choice, muts, metadata=metadata)
    if wt_run_key and rk == wt_run_key and wt_st_gap is not None and pd.notna(st_gap_val) and abs(st_gap_val - wt_st_gap) > 0.0:
        print("DATA_INTEGRITY_FAILURE")
        raise ValueError("STALE_STRUCTURE_ERROR")
    total = delta - WEIGHTS.clash_lambda * clash - WEIGHTS.stability_mu * stability + WEIGHTS.confidence_nu * scs
    notes = "Computed via full pipeline"
    if note_flag == "emission_proxy":
        notes += "; emission-derived brightness proxy"
    return Recommendation(
        mutations=muts,
        total_score=total,
        pred_brightness=pred_bright,
        delta_brightness=delta,
        synergy=0.0,
        clash_risk=clash,
        stability_risk=stability,
        confidence=scs,
        notes=notes,
        run_key=rk,
        heuristic=False,
        feature_hash=feature_hash,
        st_gap=st_gap_val,
    )


def recommend(
    pdb_path: str,
    cofactor_choice: str,
    baseline_brightness: float | None,
    confidence: float = 50.0,
    radius: float = CONFIG.radius_primary,
    max_combo_size: int = CONFIG.max_combo_size,
    beam_width: int = CONFIG.beam_width,
    top_n: int = CONFIG.top_n,
    do_not_mutate: List[str] | None = None,
) -> Dict[str, List[Recommendation]]:
    do_not_mutate = do_not_mutate or []
    budget_env = os.environ.get("RECOMMENDER_TIME_BUDGET_SEC", "").strip()
    if budget_env:
        try:
            time_budget_sec = float(budget_env)
            if time_budget_sec <= 0:
                time_budget_sec = None
        except ValueError:
            time_budget_sec = None
    else:
        # Unlimited by default unless user explicitly sets RECOMMENDER_TIME_BUDGET_SEC.
        time_budget_sec = None
    t0 = time.monotonic()

    def budget_exhausted() -> bool:
        return time_budget_sec is not None and (time.monotonic() - t0) >= time_budget_sec

    residue_index, _ = parse_pdb_structure(pdb_path)
    structure_hash = structure_hash_from_file(pdb_path)
    cand_mutations = conservative_candidate_pool(pdb_path, do_not_mutate, radius, CONFIG.allow_secondary_shell)
    max_single_env = os.environ.get("MAX_SINGLE_MUTATIONS", "").strip()
    try:
        max_single = int(max_single_env) if max_single_env else 0
    except ValueError:
        max_single = 0
    if max_single > 0 and len(cand_mutations) > max_single:
        cand_mutations = cand_mutations[:max_single]
        print(f"INFO: Truncated candidate mutations to top {max_single} for tractable recommendation runtime.")
    elif max_single <= 0:
        print(f"INFO: Unlimited mode enabled for singles (evaluating all {len(cand_mutations)} candidates).")
    if not cand_mutations:
        return {"singles": [], "doubles": [], "triples": [], "quads": []}
    structure_keys = set(residue_index.keys())

    def exists_in_structure(mut: Mutation) -> bool:
        rec = residue_index.get((mut.chain, mut.resseq, mut.icode)) or residue_index.get((mut.chain, mut.resseq, " "))
        if rec is None:
            return False
        wt_from_structure = _resn_to_aa1(rec.resn)
        return wt_from_structure == mut.wt and (
            (mut.chain, mut.resseq, mut.icode) in structure_keys or (mut.chain, mut.resseq, " ") in structure_keys
        )

    cand_mutations = [m for m in cand_mutations if exists_in_structure(m)]
    if max_single <= 0:
        print(f"INFO: Singles after structure validation: {len(cand_mutations)}")

    est_env = os.environ.get("RECOMMENDER_SEC_PER_EVAL", "").strip()
    try:
        est_sec_per_eval = float(est_env) if est_env else 8.0
        if est_sec_per_eval <= 0:
            est_sec_per_eval = 8.0
    except ValueError:
        est_sec_per_eval = 8.0
    planned_singles = len(cand_mutations)
    planned_doubles = beam_width if (max_combo_size >= 2 and planned_singles > 1) else 0
    planned_triples = beam_width if (max_combo_size >= 3 and planned_singles > 2) else 0
    planned_total = 1 + planned_singles + planned_doubles + planned_triples  # + WT baseline
    print(
        f"INFO: Recommendation ETA ~{_format_eta(planned_total * est_sec_per_eval)} "
        f"(~{planned_total} evaluations @ {est_sec_per_eval:.1f}s each)."
    )

    # Baseline (WT) run to obtain brightness and feature hash
    try:
        wt_result = predict_for_mutations(pdb_path, cofactor_choice, [], None, None, confidence)
    except Exception:
        wt_result = None
    wt_brightness = wt_result.pred_brightness if wt_result else baseline_brightness
    wt_feature_hash = wt_result.feature_hash if wt_result else None
    wt_run_key = wt_result.run_key if wt_result else None
    wt_st_gap = wt_result.st_gap if wt_result else None
    wt_feature_vector = {}

    # Ensure downstream scoring has a numeric baseline even if WT run failed.
    if wt_brightness is None:
        wt_brightness = baseline_brightness if baseline_brightness is not None else 0.0
    baseline_for_combos = baseline_brightness if baseline_brightness is not None else wt_brightness

    def heuristic_results(reason: str) -> Dict[str, List[Recommendation]]:
        heur_pool = [
            score_mutation(
                mutation=m,
                baseline_brightness=baseline_for_combos,
                buried=0.0,
                confidence=confidence,
                base_run_key=wt_run_key or "",
                structure_hash=structure_hash,
                cofactor_choice=cofactor_choice,
            )
            for m in cand_mutations
        ]
        for rec in heur_pool:
            rec.notes = f"{rec.notes} ({reason})"
        heur_singles = sorted(heur_pool, key=lambda r: (-r.total_score, r.run_key))[:top_n]
        heur_beams = beam_search(
            baseline_brightness=baseline_for_combos,
            candidates=heur_singles,
            max_size=max_combo_size,
            beam_width=beam_width,
            confidence=confidence,
            structure_hash=structure_hash,
            cofactor_choice=cofactor_choice,
        )
        return {
            "singles": heur_singles,
            "doubles": sorted(heur_beams.get(2, []), key=lambda r: (-r.total_score, r.run_key))[:top_n],
            "triples": sorted(heur_beams.get(3, []), key=lambda r: (-r.total_score, r.run_key))[:top_n],
            "quads": [],
        }

    evaluated: List[Recommendation] = []
    cache: Dict[str, Recommendation] = {}
    singles_start = time.monotonic()
    for i, mut in enumerate(cand_mutations, start=1):
        if budget_exhausted():
            print("INFO: Recommendation time budget reached during single-mutation evaluation.")
            break
        # run_key determined after mutation generation; cache later
        try:
            res = predict_for_mutations(
                pdb_path,
                cofactor_choice,
                [mut.canonical_token()],
                wt_feature_hash,
                wt_brightness,
                confidence,
                wt_run_key=wt_run_key,
                wt_st_gap=wt_st_gap,
            )
        except ValueError:
            res = None
        if res:
            # hash collision guard: identical hash but nonzero delta
            if wt_feature_hash and res.feature_hash == wt_feature_hash and abs(res.delta_brightness) > 0:
                print("HASH_COLLISION_BUG")
                res.feature_hash = res.feature_hash + "_collision"
            rk = res.run_key
            if rk in cache:
                evaluated.append(cache[rk])
            else:
                cache[rk] = res
                evaluated.append(res)
        if i % 5 == 0 or i == len(cand_mutations):
            elapsed = max(1e-6, time.monotonic() - singles_start)
            rate = i / elapsed
            remaining = max(0, len(cand_mutations) - i)
            eta = remaining / max(rate, 1e-6)
            print(f"INFO: Singles progress {i}/{len(cand_mutations)}; ETA ~{_format_eta(eta)}")

    if not evaluated:
        return heuristic_results("no-computed-singles")

    singles = sorted(evaluated, key=lambda r: (-r.total_score, r.run_key))[:top_n]
    beams = beam_search(
        baseline_brightness=baseline_for_combos,
        candidates=singles,
        max_size=max_combo_size,
        beam_width=beam_width,
        confidence=confidence,
        structure_hash=structure_hash,
        cofactor_choice=cofactor_choice,
    )
    doubles_eval: List[Recommendation] = []
    doubles_beam = beams.get(2, [])
    doubles_start = time.monotonic()
    for i, rec in enumerate(doubles_beam, start=1):
        if budget_exhausted():
            print("INFO: Recommendation time budget reached during double-mutation evaluation.")
            break
        try:
            res = predict_for_mutations(pdb_path, cofactor_choice, [m.canonical_token() for m in rec.mutations], wt_feature_hash, wt_brightness, confidence)
        except ValueError:
            res = None
        if res:
            doubles_eval.append(res)
        if i % 3 == 0 or i == len(doubles_beam):
            elapsed = max(1e-6, time.monotonic() - doubles_start)
            rate = i / elapsed
            remaining = max(0, len(doubles_beam) - i)
            eta = remaining / max(rate, 1e-6)
            print(f"INFO: Doubles progress {i}/{len(doubles_beam)}; ETA ~{_format_eta(eta)}")
    triples_eval: List[Recommendation] = []
    triples_beam = beams.get(3, [])
    triples_start = time.monotonic()
    for i, rec in enumerate(triples_beam, start=1):
        if budget_exhausted():
            print("INFO: Recommendation time budget reached during triple-mutation evaluation.")
            break
        try:
            res = predict_for_mutations(pdb_path, cofactor_choice, [m.canonical_token() for m in rec.mutations], wt_feature_hash, wt_brightness, confidence)
        except ValueError:
            res = None
        if res:
            triples_eval.append(res)
        if i % 3 == 0 or i == len(triples_beam):
            elapsed = max(1e-6, time.monotonic() - triples_start)
            rate = i / elapsed
            remaining = max(0, len(triples_beam) - i)
            eta = remaining / max(rate, 1e-6)
            print(f"INFO: Triples progress {i}/{len(triples_beam)}; ETA ~{_format_eta(eta)}")
    computed_all = singles + doubles_eval + triples_eval
    # Validation gate: ensure variation in feature_hash and brightness
    if len(computed_all) >= 2:
        hashes = {r.feature_hash for r in computed_all if r.feature_hash}
        brightness_vals = [r.pred_brightness for r in computed_all if pd.notna(r.pred_brightness)]
        unique_brightness = {round(b, 6) for b in brightness_vals}
        if len(unique_brightness) <= 1 and len(hashes) <= 1:
            if all(abs(r.delta_brightness or 0.0) < 1e-9 for r in computed_all):
                return heuristic_results("computed-flat-response")
    return {
        "singles": singles,
        "doubles": sorted(doubles_eval, key=lambda r: (-r.total_score, r.run_key))[:top_n],
        "triples": sorted(triples_eval, key=lambda r: (-r.total_score, r.run_key))[:top_n],
        "quads": [],
    }


def recommendation_to_row(rec: Recommendation) -> Dict[str, object]:
    muts = [m.canonical_token() for m in rec.mutations]
    return {
        "Mutations": ";".join(muts),
        "TotalScore": rec.total_score,
        "PredBrightness": rec.pred_brightness,
        "DeltaBrightness": rec.delta_brightness,
        "Synergy": rec.synergy,
        "ClashRisk": rec.clash_risk,
        "StabilityRisk": rec.stability_risk,
        "Confidence": rec.confidence,
        "Notes": rec.notes,
        "run_key": rec.run_key,
        "heuristic": rec.heuristic,
        "feature_hash": rec.feature_hash or "",
    }
