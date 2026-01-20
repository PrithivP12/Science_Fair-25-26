from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np


# Minimal amino-acid mapping (one-letter to three-letter)
AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

def coordinate_hash(pdb_path: str, center: Tuple[float, float, float] | None = None, radius: float = 12.0) -> str:
    """Hash float64 XYZ matrix for atoms within a sphere (default 12 Å)."""
    coords = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
    if not coords:
        return hashlib.sha256(b"").hexdigest()
    arr = np.array(coords, dtype=np.float64)
    if center is None:
        center = tuple(arr.mean(axis=0))
    center_arr = np.array(center)
    if radius is not None:
        mask = np.linalg.norm(arr - center_arr, axis=1) <= radius
        arr = arr[mask]
    return hashlib.sha256(arr.tobytes()).hexdigest()


def atom_record_hash(pdb_path: str) -> str:
    """Hash raw ATOM/HETATM lines to capture any coordinate/state change."""
    try:
        lines = []
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    lines.append(line.rstrip("\n"))
        payload = "\n".join(lines)
    except Exception:
        payload = ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_coords_array(pdb_path: str) -> np.ndarray:
    coords = []
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                coords.append((x, y, z))
    except Exception:
        pass
    return np.array(coords, dtype=np.float64)

# Very lightweight side-chain templates (relative to CA) for bulky residues
SIDECHAIN_TEMPLATES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "F": {"CB": (1.5, 0.0, 0.0), "CG": (2.5, 0.5, 0.5), "CD1": (3.6, 0.2, 0.2), "CD2": (3.6, 0.8, 0.8)},
    "W": {"CB": (1.5, 0.0, 0.0), "CG": (2.5, 0.6, 0.3), "CD1": (3.6, 0.2, 0.1), "CD2": (3.6, 1.0, 0.9)},
    "Y": {"CB": (1.5, 0.0, 0.0), "CG": (2.5, 0.4, 0.4), "CD1": (3.6, 0.1, 0.1), "CD2": (3.6, 0.7, 0.7), "OH": (4.5, 0.4, 0.4)},
    "H": {"CB": (1.5, 0.0, 0.0), "CG": (2.4, 0.3, 0.3), "ND1": (3.3, 0.1, 0.1), "CE1": (4.0, 0.2, 0.2)},
    "V": {"CB": (1.5, 0.0, 0.0), "CG1": (2.4, 0.5, 0.5), "CG2": (2.4, -0.5, -0.5)},
}


def _mut_shift_vector(token: str) -> Tuple[float, float, float]:
    """Deterministic small displacement vector from mutation token to avoid identical coords."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    # map bytes to small displacements
    return tuple(((b % 7) - 3) * 0.05 for b in h[:3])  # +/-0.15 Å range


def apply_mutations_to_pdb(pdb_path: str, mutation_tokens: Iterable[str]) -> Tuple[str, bool]:
    """
    Apply crude side-chain replacement and local relaxation to a PDB.
    This is a lightweight proxy for rotamer packing and clash relief.

    Returns (mutated_pdb_path, mutated_changed_flag).
    """
    tokens = list(mutation_tokens)
    if not tokens:
        return pdb_path, False

    lines = Path(pdb_path).read_text().splitlines()
    atoms: List[Dict[str, Any]] = []
    for ln in lines:
        record = ln[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            atom = {
                "record": record,
                "serial": ln[6:11],
                "name": ln[12:16].strip(),
                "altloc": ln[16],
                "resn": ln[17:20].strip(),
                "chain": (ln[21] or "?").strip() or "?",
                "resi": ln[22:26].strip(),
                "icode": (ln[26] or " ").strip() or " ",
                "x": float(ln[30:38]),
                "y": float(ln[38:46]),
                "z": float(ln[46:54]),
                "rest": ln[54:],
                "line": ln,
            }
            atoms.append(atom)
        except Exception:
            continue

    mut_keys = []
    for tok in tokens:
        if ":" in tok:
            chain_part, mut_part = tok.split(":", 1)
        else:
            chain_part, mut_part = ("A", tok)
        wt = mut_part[0]
        resi = mut_part[1:-1]
        mut = mut_part[-1]
        mut_keys.append((chain_part, resi, " ", wt, mut))
    mut_targets = {(ck[0], ck[1], ck[2]): ck for ck in mut_keys}

    # build maps for residues
    residue_atoms: Dict[Tuple[str, str, str], List[int]] = {}
    for idx, a in enumerate(atoms):
        key = (a["chain"], a["resi"], a["icode"])
        residue_atoms.setdefault(key, []).append(idx)
    # store WT COM and atom count
    wt_stats = {}
    for key, idxs in residue_atoms.items():
        coords = np.array([[atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]] for i in idxs])
        if coords.size == 0:
            continue
        wt_stats[key] = {"com": coords.mean(axis=0), "count": len(idxs)}

    # mutate: remove target atoms, rebuild sidechain template anchored at CA (or centroid)
    mutated_atoms: List[Dict[str, Any]] = []
    for key, idxs in residue_atoms.items():
        if key in mut_targets:
            chain, resi, icode, wt, mut = mut_targets[key]
            resn3 = AA3.get(mut, mut)
            # find CA as anchor
            ca_coord = None
            for i in idxs:
                if atoms[i]["name"] == "CA":
                    ca_coord = np.array([atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]])
                    break
            if ca_coord is None:
                # fallback to centroid
                coords = np.array([[atoms[i]["x"], atoms[i]["y"], atoms[i]["z"]] for i in idxs])
                ca_coord = coords.mean(axis=0)
            # backbone atoms from WT retained (N, CA, C, O)
            for i in idxs:
                if atoms[i]["name"] in {"N", "CA", "C", "O"}:
                    a = dict(atoms[i])
                    a["resn"] = resn3
                    mutated_atoms.append(a)
            # build sidechain pseudo-rotamer
            template = SIDECHAIN_TEMPLATES.get(mut, {"CB": (1.5, 0.0, 0.0)})
            shift_vec = np.array(_mut_shift_vector(f"{chain}:{resi}{mut}"))
            for aname, offset in template.items():
                vec = np.array(offset) + shift_vec
                coord = ca_coord + vec
                mutated_atoms.append(
                    {
                        "record": "ATOM",
                        "serial": "99999",
                        "name": aname,
                        "altloc": " ",
                        "resn": resn3,
                        "chain": chain,
                        "resi": resi,
                        "icode": icode,
                        "x": coord[0],
                        "y": coord[1],
                        "z": coord[2],
                        "rest": "  1.00 20.00           C",
                        "line": "",
                    }
                )
        else:
            for i in idxs:
                mutated_atoms.append(dict(atoms[i]))

    # local minimization (steepest descent) within 6 Å of mutated residues
    mut_centroids = []
    for key in mut_targets:
        coords = np.array([[a["x"], a["y"], a["z"]] for a in mutated_atoms if (a["chain"], a["resi"], a["icode"]) == key])
        if coords.size:
            mut_centroids.append(coords.mean(axis=0))
    if mut_centroids:
        mut_centroids = np.array(mut_centroids)
        coords_arr = np.array([[a["x"], a["y"], a["z"]] for a in mutated_atoms])
        for _ in range(50):
            for idx, a in enumerate(mutated_atoms):
                pos = coords_arr[idx]
                # if within 6 Å of any mutated centroid, apply soft repulsion to neighbors closer than 1.2 Å
                if np.min(np.linalg.norm(pos - mut_centroids, axis=1)) <= 6.0:
                    for jdx, b in enumerate(mutated_atoms):
                        if idx == jdx:
                            continue
                        delta = pos - coords_arr[jdx]
                        dist = np.linalg.norm(delta)
                        if dist < 1.2 and dist > 1e-3:
                            step = 0.02 * (1.2 - dist) * (delta / dist)
                            coords_arr[idx] += step
            # update positions
            for idx, a in enumerate(mutated_atoms):
                a["x"], a["y"], a["z"] = coords_arr[idx]

    # validation vs WT
    changed = False
    for key, stats in wt_stats.items():
        mut_coords = np.array([[a["x"], a["y"], a["z"]] for a in mutated_atoms if (a["chain"], a["resi"], a["icode"]) == key])
        if mut_coords.size == 0:
            continue
        com_mut = mut_coords.mean(axis=0)
        if key in mut_targets:
            if mut_coords.shape[0] == stats["count"] and np.linalg.norm(com_mut - stats["com"]) < 1e-3:
                raise ValueError("STRUCTURE_BUILD_FAILURE")
            changed = True
    if mut_targets and not changed:
        raise ValueError("mutated_pdb_identical_to_wt_for_targets")

    # write PDB
    out_lines = []
    serial_counter = 1
    for a in mutated_atoms:
        resn = a["resn"]
        line = f"{a['record']:<6}{serial_counter:5d} {a['name']:^4}{a['altloc']}{resn:>3} {a['chain']}{a['resi']:>4}{a['icode']:<1}   {a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}{a['rest']}"
        out_lines.append(line)
        serial_counter += 1

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        tmp.write("\n".join(out_lines).encode("utf-8"))
        return tmp.name, True
