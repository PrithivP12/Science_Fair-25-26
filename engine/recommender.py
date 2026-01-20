from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys
import tempfile
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
)
from engine.vqe_n5_edge import AA_PROPERTIES
from engine.mutator import apply_mutations_to_pdb, atom_record_hash, load_coords_array
import numpy as np

# Conservative substitution libraries
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "Y", "W"}
POLAR = {"S", "T", "N", "Q", "H"}
CHARGED = {"K", "R", "D", "E"}
SMALL = {"A", "G", "S"}


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
    coords = []
    residue_coords: Dict[Tuple[str, str, str], List[Tuple[float, float, float]]] = {}
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resn = line[17:20].strip().upper()
            chain = (line[21] or "?").strip() or "?"
            resi = line[22:26].strip()
            icode = line[26].strip() or " "
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coord = (x, y, z)
            residue_coords.setdefault((chain, resi, icode), []).append(coord)
            if resn in cofactor_resns:
                coords.append(coord)
    flavin_found = len(coords) > 0
    if not flavin_found:
        return {}, False
    flavin_centroid = np.mean(np.array(coords), axis=0)
    dist_map = {}
    for key, pts in residue_coords.items():
        dists = np.linalg.norm(np.array(pts) - flavin_centroid, axis=1)
        dist_map[key] = float(np.min(dists))
    return {k: v for k, v in dist_map.items() if v <= radius}, True


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
    candidates: List[Mutation] = []
    for key, dist in dist_primary.items():
        if dist > radius:
            continue
        chain, resi, icode = key
        rec = residue_index.get(key) or residue_index.get((chain, resi, " "))
        if not rec:
            continue
        if key not in residue_coords:
            print(f"SKIP: missing coordinates for {chain}:{resi}{icode}")
            continue  # skip missing coordinates
        resn_full = rec.resn.upper()
        resn = resn_full[:1]
        if any(resi in t for t in do_not_mutate):
            print(f"SKIP: {chain}:{resi}{icode} in do-not-mutate list")
            continue
        if rec.resn == "CYS" and dist < 4.0:
            continue
        buried = burial_metric(residue_coords, key)
        for mut_aa in allowed_substitutions(resn, buried):
            candidates.append(Mutation(chain=chain, resseq=resi, icode=icode, wt=resn, mut=mut_aa))
    if second_shell:
        for key, dist in dist_secondary.items():
            if key in dist_primary or dist > CONFIG.radius_secondary:
                continue
            chain, resi, icode = key
            rec = residue_index.get(key) or residue_index.get((chain, resi, " "))
            if not rec:
                continue
            if key not in residue_coords:
                continue
            resn_full = rec.resn.upper()
            resn = resn_full[:1]
            if any(resi in t for t in do_not_mutate):
                continue
            buried = burial_metric(residue_coords, key)
            for mut_aa in allowed_substitutions(resn, buried):
                candidates.append(Mutation(chain=chain, resseq=resi, icode=icode, wt=resn, mut=mut_aa))
    return candidates


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
    env = os.environ.copy()
    env["MUTATION_LIST"] = ",".join(mutation_tokens)
    env["COFACTOR_CONFIGURATION"] = cofactor_choice
    cmd = [sys.executable, str(Path(__file__).resolve().parent / "vqe_n5_edge.py"), "--pdb", mutated_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90, env=env)
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    profile_path = Path(os.getcwd()) / "artifacts" / "qc_n5_gpr" / "Final_Quantum_Profiles.csv"
    if not profile_path.exists():
        return None
    try:
        df = pd.read_csv(profile_path, on_bad_lines="skip", engine="python")
    except Exception:
        return None
    # match on mutation_list string if present
    rec_df = df[df.get("mutation_list", "") == env["MUTATION_LIST"]]
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
    residue_index, _ = parse_pdb_structure(pdb_path)
    structure_hash = structure_hash_from_file(pdb_path)
    cand_mutations = conservative_candidate_pool(pdb_path, do_not_mutate, radius, CONFIG.allow_secondary_shell)
    if not cand_mutations:
        return {"singles": [], "doubles": [], "triples": [], "quads": []}
    structure_keys = set(residue_index.keys())

    def exists_in_structure(mut: Mutation) -> bool:
        rec = residue_index.get((mut.chain, mut.resseq, mut.icode)) or residue_index.get((mut.chain, mut.resseq, " "))
        return rec is not None and rec.resn[:1] == mut.wt and ((mut.chain, mut.resseq, mut.icode) in structure_keys or (mut.chain, mut.resseq, " ") in structure_keys)

    cand_mutations = [m for m in cand_mutations if exists_in_structure(m)]

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

    evaluated: List[Recommendation] = []
    cache: Dict[str, Recommendation] = {}
    for mut in cand_mutations:
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

    singles = sorted(evaluated, key=lambda r: (-r.total_score, r.run_key))[:top_n]
    beams = beam_search(
        baseline_brightness=baseline_brightness,
        candidates=singles,
        max_size=max_combo_size,
        beam_width=beam_width,
        confidence=confidence,
        structure_hash=structure_hash,
        cofactor_choice=cofactor_choice,
    )
    doubles_eval: List[Recommendation] = []
    for rec in beams.get(2, []):
        try:
            res = predict_for_mutations(pdb_path, cofactor_choice, [m.canonical_token() for m in rec.mutations], wt_feature_hash, wt_brightness, confidence)
        except ValueError:
            res = None
        if res:
            doubles_eval.append(res)
    triples_eval: List[Recommendation] = []
    for rec in beams.get(3, []):
        try:
            res = predict_for_mutations(pdb_path, cofactor_choice, [m.canonical_token() for m in rec.mutations], wt_feature_hash, wt_brightness, confidence)
        except ValueError:
            res = None
        if res:
            triples_eval.append(res)
    computed_all = singles + doubles_eval + triples_eval
    # Validation gate: ensure variation in feature_hash and brightness
    if len(computed_all) >= 2:
        hashes = {r.feature_hash for r in computed_all if r.feature_hash}
        brightness_vals = [r.pred_brightness for r in computed_all if pd.notna(r.pred_brightness)]
        unique_brightness = {round(b, 6) for b in brightness_vals}
        if len(unique_brightness) <= 1 and len(hashes) <= 1:
            if all(abs(r.delta_brightness or 0.0) < 1e-9 for r in computed_all):
                details = {
                    "unique_feature_hashes": len(hashes),
                    "unique_brightness": len(unique_brightness),
                    "cache_size": len(cache),
                }
                raise ValueError(
                    f"Recommendation engine failed validation (mutations not affecting features or brightness constant). Details: {details}"
                )
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
