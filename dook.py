#!/usr/bin/env python3
"""
dook.py — Flavoprotein redox-potential feature builder (per cofactor instance)

What it does:
- Reads an EM/validation CSV with at least: pdb_id, uniprot_id (optional), Em (optional), pH (optional)
- For each PDB ID:
  - Finds/loads the structure file from --pdb-dir (supports .ent/.pdb/.cif, gz, nested dirs)
  - If missing locally, downloads via Biopython PDBList
  - Detects FAD/FMNs in the structure and emits ONE feature row per cofactor instance
  - If your label CSV has multiple rows per pdb_id (e.g., different pH/Em), it will emit one feature row per label row

Key differences vs your broken run:
- Robust structure-file discovery + gz auto-unzip
- Supports MMCIF (.cif) via Bio.PDB.MMCIFParser
- Accepts BOTH flag styles:
    --em OR --validation
    --workers OR --n-jobs
- Does NOT require 'in_jcim' column (it’s optional). If missing, filled with NaN.
"""

import os
import re
import math
import glob
import gzip
import shutil
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import requests

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBList import PDBList

# -----------------------------
# Configuration / constants
# -----------------------------
FLOPPY_NANS = np.nan

COFACTORS = {"FAD", "FMN"}
RING_ATOMS = ["N1", "O2", "N3", "O4", "N5"]  # isoalloxazine atoms (common naming)
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
POLAR_ELEMENTS = {"N", "O"}
HEAVY_ELEMENTS = {"C", "N", "O", "S", "P", "SE"}

GROUPS = {
    "pos": {"ARG", "LYS", "HIS"},
    "neg": {"ASP", "GLU"},
    "polar": {"SER", "THR", "ASN", "GLN", "CYS"},
    "arom": {"PHE", "TYR", "TRP"},
    "hydro": {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY"},
}
AA_STANDARD = set().union(*GROUPS.values())

# Rough AA property table (fast, no external deps).
AA_PROPS = {
    # volume, hydrophobicity, pI, P(helix), P(sheet), flexibility, nO, nN, nS, hbond_cap, steric
    "ALA": ( 88.6,  1.8, 6.0, 1.42, 0.83, 0.36, 0, 0, 0, 0, 1.0),
    "ARG": (173.4, -4.5,10.8, 0.98, 0.93, 0.53, 0, 3, 0, 4, 2.3),
    "ASN": (114.1, -3.5, 5.4, 0.67, 0.89, 0.46, 1, 1, 0, 2, 1.6),
    "ASP": (111.1, -3.5, 2.8, 1.01, 0.54, 0.51, 2, 0, 0, 2, 1.4),
    "CYS": (108.5,  2.5, 5.1, 0.70, 1.19, 0.35, 0, 0, 1, 1, 1.2),
    "GLN": (143.8, -3.5, 5.7, 1.11, 1.10, 0.49, 1, 1, 0, 2, 1.9),
    "GLU": (138.4, -3.5, 3.2, 1.51, 0.37, 0.50, 2, 0, 0, 2, 1.8),
    "GLY": ( 60.1, -0.4, 6.0, 0.57, 0.75, 0.54, 0, 0, 0, 0, 0.8),
    "HIS": (153.2, -3.2, 7.6, 1.00, 0.87, 0.32, 0, 2, 0, 2, 1.8),
    "ILE": (166.7,  4.5, 6.0, 1.08, 1.60, 0.46, 0, 0, 0, 0, 2.0),
    "LEU": (166.7,  3.8, 6.0, 1.21, 1.30, 0.37, 0, 0, 0, 0, 2.0),
    "LYS": (168.6, -3.9, 9.7, 1.14, 0.74, 0.47, 0, 1, 0, 2, 2.0),
    "MET": (162.9,  1.9, 5.7, 1.45, 1.05, 0.30, 0, 0, 1, 0, 2.1),
    "PHE": (189.9,  2.8, 5.5, 1.13, 1.38, 0.31, 0, 0, 0, 0, 2.3),
    "PRO": (112.7, -1.6, 6.3, 0.57, 0.55, 0.51, 0, 0, 0, 0, 1.4),
    "SER": ( 89.0, -0.8, 5.7, 0.77, 0.75, 0.51, 1, 0, 0, 1, 1.0),
    "THR": (116.1, -0.7, 5.6, 0.83, 1.19, 0.44, 1, 0, 0, 1, 1.2),
    "TRP": (227.8, -0.9, 5.9, 1.08, 1.37, 0.31, 0, 1, 0, 1, 2.7),
    "TYR": (193.6, -1.3, 5.7, 0.69, 1.47, 0.42, 1, 0, 0, 1, 2.4),
    "VAL": (140.0,  4.2, 6.0, 1.06, 1.70, 0.39, 0, 0, 0, 0, 1.8),
}

# Rough residue charges at ~neutral pH (fast proxy; refine later with PROPKA if you want)
RES_Q = {"ASP": -1.0, "GLU": -1.0, "ARG": +1.0, "LYS": +1.0, "HIS": +0.1}


# -----------------------------
# Logging
# -----------------------------
def log(msg: str):
    print(f"[dook] {msg}", flush=True)


# -----------------------------
# Tool detection (optional deps)
# -----------------------------
def have_cmd(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def have_freesasa_py() -> bool:
    try:
        import freesasa  # noqa
        return True
    except Exception:
        return False


# -----------------------------
# Helpers
# -----------------------------
def safe_get_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def is_water_residue(res) -> bool:
    return res.get_resname() in {"HOH", "WAT", "H2O"}

def is_standard_aa(res) -> bool:
    return res.get_resname() in AA_STANDARD

def atom_distance(a1, a2) -> float:
    v = a1.get_coord() - a2.get_coord()
    return float(np.linalg.norm(v))

def residue_id_tuple(res) -> Tuple[Any, Any, Any]:
    return res.get_id()

def parse_resolution_from_pdb_text(pdb_path: str) -> float:
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK   2 RESOLUTION."):
                    m = re.search(r"RESOLUTION\.\s+([0-9]+\.[0-9]+)\s+ANGSTROMS", line)
                    if m:
                        return float(m.group(1))
    except Exception:
        pass
    return FLOPPY_NANS

def parse_method_from_pdb_text(pdb_path: str) -> str:
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("EXPDTA"):
                    return line.strip().replace("EXPDTA", "").strip()
    except Exception:
        pass
    return ""

def sum_props_for_resnames(resnames: List[str]) -> Dict[str, float]:
    out = {
        "Volume": 0.0,
        "Hydrophobicity": 0.0,
        "IsoelectricPoint": 0.0,
        "P_helix": 0.0,
        "P_sheet": 0.0,
        "Flexibility": 0.0,
        "nO_side": 0.0,
        "nN_side": 0.0,
        "nS_side": 0.0,
        "HBondCap": 0.0,
        "Steric": 0.0,
    }
    n = 0
    for rn in resnames:
        if rn in AA_PROPS:
            vol, hyd, pi, ph, ps, flex, nO, nN, nS, hb, steric = AA_PROPS[rn]
            out["Volume"] += vol
            out["Hydrophobicity"] += hyd
            out["IsoelectricPoint"] += pi
            out["P_helix"] += ph
            out["P_sheet"] += ps
            out["Flexibility"] += flex
            out["nO_side"] += nO
            out["nN_side"] += nN
            out["nS_side"] += nS
            out["HBondCap"] += hb
            out["Steric"] += steric
            n += 1
    out["NumAA"] = float(n)
    return out

def region_residue_list_from_atoms(atoms: List[Any]) -> List[Any]:
    seen = set()
    residues = []
    for a in atoms:
        r = a.get_parent()
        key = (r.get_parent().get_id(), residue_id_tuple(r), r.get_resname())
        if key not in seen:
            seen.add(key)
            residues.append(r)
    return residues

def charge_sums_from_atoms(target_atom, atoms: List[Any]) -> Dict[str, float]:
    s_q_over_r = 0.0
    s_q_over_r2 = 0.0
    min_pos = None
    min_neg = None
    eps = 1e-6
    for a in atoms:
        r = a.get_parent()
        rn = r.get_resname()
        q = RES_Q.get(rn, 0.0)
        if q == 0.0:
            continue
        d = atom_distance(target_atom, a) + eps
        s_q_over_r += q / d
        s_q_over_r2 += q / (d * d)
        if q > 0:
            min_pos = d if (min_pos is None or d < min_pos) else min_pos
        if q < 0:
            min_neg = d if (min_neg is None or d < min_neg) else min_neg
    return {
        "coulomb_q_over_r": s_q_over_r,
        "coulomb_q_over_r2": s_q_over_r2,
        "min_dist_pos_charge": float(min_pos) if min_pos is not None else 0.0,
        "min_dist_neg_charge": float(min_neg) if min_neg is not None else 0.0,
    }

def compute_group_counts(resnames: List[str]) -> Dict[str, int]:
    out = {f"count_{g}": 0 for g in GROUPS.keys()}
    for rn in resnames:
        for g, s in GROUPS.items():
            if rn in s:
                out[f"count_{g}"] += 1
    out["count_apolar"] = out["count_hydro"]
    out["count_aromatic"] = out["count_arom"]
    out["formal_charge"] = out["count_pos"] - out["count_neg"]
    return out

def compute_atom_counts(atoms: List[Any]) -> Dict[str, int]:
    elems = [(a.element or "").strip().upper() for a in atoms]
    return {
        "count_N": int(sum(1 for e in elems if e == "N")),
        "count_O": int(sum(1 for e in elems if e == "O")),
        "count_C": int(sum(1 for e in elems if e == "C")),
        "count_S": int(sum(1 for e in elems if e == "S")),
        "count_NO": int(sum(1 for e in elems if e in {"N", "O"})),
        "heavy_atom_count": int(sum(1 for e in elems if e in HEAVY_ELEMENTS)),
    }

def compute_contact_stats(target_atom, atoms: List[Any]) -> Dict[str, float]:
    if not atoms:
        return {
            "dist_min": 0.0,
            "dist_mean": 0.0,
            "weighted_contacts_e_minus_d": 0.0,
            "sum_1_over_d2": 0.0,
            "polar_contacts_3A": 0.0,
            "polar_contacts_35A": 0.0,
            "polar_contacts_4A": 0.0,
        }
    ds = [atom_distance(target_atom, a) for a in atoms]
    eps = 1e-6
    elems = [(a.element or "").strip().upper() for a in atoms]
    polar_ds = [d for d, e in zip(ds, elems) if e in POLAR_ELEMENTS]
    return {
        "dist_min": float(min(ds)),
        "dist_mean": float(np.mean(ds)),
        "weighted_contacts_e_minus_d": float(sum(math.exp(-d) for d in ds)),
        "sum_1_over_d2": float(sum(1.0 / ((d + eps) ** 2) for d in ds)),
        "polar_contacts_3A": float(sum(1 for d in polar_ds if d <= 3.0)),
        "polar_contacts_35A": float(sum(1 for d in polar_ds if d <= 3.5)),
        "polar_contacts_4A": float(sum(1 for d in polar_ds if d <= 4.0)),
    }

def compute_backbone_ratio(atoms: List[Any]) -> float:
    if not atoms:
        return 0.0
    return float(sum(1 for a in atoms if a.get_name().strip() in BACKBONE_ATOMS) / len(atoms))

def compute_mean_bfactor(atoms: List[Any]) -> float:
    if not atoms:
        return 0.0
    return float(np.mean([a.get_bfactor() for a in atoms]))


# -----------------------------
# Optional: DSSP
# -----------------------------
def compute_dssp_secondary_structure(structure, pdb_path: str) -> Optional[Dict[Tuple[str, Tuple[Any, Any, Any]], str]]:
    try:
        from Bio.PDB.DSSP import DSSP
        model = next(structure.get_models())
        dssp = DSSP(model, pdb_path, dssp="mkdssp")  # requires mkdssp installed
        ss_map = {}
        for key in dssp.keys():
            chain_id, res_id = key[0], key[1]
            ss = dssp[key][2]
            ss_map[(chain_id, res_id)] = ss
        return ss_map
    except Exception:
        return None

def ss_fractions(residues: List[Any], ss_map: Optional[Dict[Tuple[str, Tuple[Any, Any, Any]], str]]) -> Dict[str, float]:
    if ss_map is None or not residues:
        return {"frac_helix": FLOPPY_NANS, "frac_sheet": FLOPPY_NANS, "frac_loop": FLOPPY_NANS}
    helix = 0
    sheet = 0
    loop = 0
    n = 0
    for r in residues:
        chain_id = r.get_parent().get_id()
        code = ss_map.get((chain_id, residue_id_tuple(r)), " ")
        if code in {"H", "G", "I"}:
            helix += 1
        elif code in {"E", "B"}:
            sheet += 1
        else:
            loop += 1
        n += 1
    if n == 0:
        return {"frac_helix": FLOPPY_NANS, "frac_sheet": FLOPPY_NANS, "frac_loop": FLOPPY_NANS}
    return {"frac_helix": helix / n, "frac_sheet": sheet / n, "frac_loop": loop / n}


# -----------------------------
# Optional: FreeSASA
# -----------------------------
def compute_sasa_per_residue(pdb_path: str) -> Optional[Dict[Tuple[str, Tuple[Any, Any, Any]], float]]:
    if not have_freesasa_py():
        return None
    try:
        import freesasa
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        areas = {}
        for i in range(structure.nAtoms()):
            chain = structure.chainLabel(i)
            resnum = structure.residueNumber(i)
            icode = structure.residueInsertionCode(i) or " "
            res_id = (" ", int(resnum), icode if icode != "" else " ")
            areas[(chain, res_id)] = areas.get((chain, res_id), 0.0) + result.atomArea(i)
        return areas
    except Exception:
        return None

def region_sasa(residues: List[Any], sasa_map: Optional[Dict[Tuple[str, Tuple[Any, Any, Any]], float]]) -> Dict[str, float]:
    if sasa_map is None:
        return {"sasa_sum": FLOPPY_NANS, "sasa_mean": FLOPPY_NANS}
    if not residues:
        return {"sasa_sum": 0.0, "sasa_mean": 0.0}
    vals = []
    for r in residues:
        chain = r.get_parent().get_id()
        vals.append(sasa_map.get((chain, residue_id_tuple(r)), 0.0))
    return {"sasa_sum": float(np.sum(vals)), "sasa_mean": float(np.mean(vals)) if vals else 0.0}


# -----------------------------
# Core extraction per PDB
# -----------------------------
@dataclass
class CofactorInstance:
    chain_id: str
    resname: str
    res_id: Tuple[Any, Any, Any]
    ring_atoms_present: List[str]

def find_cofactor_instances(model) -> List[CofactorInstance]:
    out = []
    for chain in model:
        for res in chain:
            if res.get_resname() not in COFACTORS:
                continue
            present = [a for a in RING_ATOMS if a in res]
            if not present:
                continue
            out.append(
                CofactorInstance(
                    chain_id=chain.get_id(),
                    resname=res.get_resname(),
                    res_id=residue_id_tuple(res),
                    ring_atoms_present=present,
                )
            )
    return out

def get_residue_by_id(model, chain_id: str, res_id: Tuple[Any, Any, Any]):
    chain = model[chain_id]
    return chain[res_id]

def build_neighbor_search(model) -> NeighborSearch:
    atoms = [a for a in model.get_atoms()]
    return NeighborSearch(atoms)

def protein_residue_list(model) -> List[Any]:
    residues = []
    for r in model.get_residues():
        if is_water_residue(r):
            continue
        if is_standard_aa(r):
            residues.append(r)
    return residues

def atoms_within_sphere(ns: NeighborSearch, center_xyz: np.ndarray, radius: float) -> List[Any]:
    return ns.search(center_xyz, radius, level="A")

def filter_atoms_to_protein(atoms: List[Any]) -> List[Any]:
    out = []
    for a in atoms:
        r = a.get_parent()
        if is_water_residue(r):
            continue
        if not is_standard_aa(r):
            continue
        out.append(a)
    return out

def ring_barycenter(res) -> Optional[np.ndarray]:
    coords = []
    for a in RING_ATOMS:
        if a in res:
            coords.append(res[a].get_coord())
    if not coords:
        return None
    return np.mean(np.array(coords), axis=0)

def ring_union_atoms(ns: NeighborSearch, res, r2: float) -> List[Any]:
    atoms = []
    for a in RING_ATOMS:
        if a not in res:
            continue
        hits = atoms_within_sphere(ns, res[a].get_coord(), r2)
        atoms.extend(hits)
    return atoms

def nearest_residue_to_atom(target_atom, atoms: List[Any]) -> Optional[Any]:
    best = None
    best_d = None
    for a in atoms:
        r = a.get_parent()
        if not is_standard_aa(r):
            continue
        d = atom_distance(target_atom, a)
        if best_d is None or d < best_d:
            best_d = d
            best = r
    return best

def seq_neighbors(chain, res_id: Tuple[Any, Any, Any], k: int = 1) -> List[Any]:
    het, resseq, icode = res_id
    out = []
    for delta in range(-k, k + 1):
        if delta == 0:
            continue
        rid = (het, int(resseq) + delta, icode)
        if rid in chain:
            r = chain[rid]
            if is_standard_aa(r):
                out.append(r)
    return out

def region_features(prefix: str, residues: List[Any], atoms: List[Any], target_atom_for_charge=None) -> Dict[str, float]:
    resnames = [r.get_resname() for r in residues]
    out: Dict[str, float] = {}

    # AA type counts
    aa_counts = {f"{prefix}{aa}": 0.0 for aa in sorted(AA_STANDARD)}
    for rn in resnames:
        if rn in AA_STANDARD:
            aa_counts[f"{prefix}{rn}"] += 1.0
    out.update(aa_counts)

    # Group counts + charge
    gc = compute_group_counts(resnames)
    out.update({f"{prefix}{k}": float(v) for k, v in gc.items()})

    # Atom counts
    ac = compute_atom_counts(atoms)
    out.update({f"{prefix}{k}": float(v) for k, v in ac.items()})

    # Summed AA props
    props = sum_props_for_resnames(resnames)
    for k, v in props.items():
        out[f"{prefix}{k}"] = float(v)

    # Derived combos
    vol = out.get(f"{prefix}Volume", 0.0)
    ph = out.get(f"{prefix}P_helix", 0.0)
    ps = out.get(f"{prefix}P_sheet", 0.0)
    hyd = out.get(f"{prefix}Hydrophobicity", 0.0)
    flex = out.get(f"{prefix}Flexibility", 0.0)
    ster = out.get(f"{prefix}Steric", 0.0)
    iso = out.get(f"{prefix}IsoelectricPoint", 0.0)

    out[f"{prefix}P_helix_plus_sheet"] = ph + ps
    out[f"{prefix}Hydrophobicity_x_Flex"] = hyd * flex
    out[f"{prefix}Steric_x_Flex"] = ster * flex
    out[f"{prefix}Vol_x_IsoelectricPoint"] = vol * iso
    out[f"{prefix}Vol_over_P_helix"] = vol / ph if ph != 0 else 0.0
    out[f"{prefix}Vol_over_P_sheet"] = vol / ps if ps != 0 else 0.0
    out[f"{prefix}Vol_over_P_helix_plus_sheet"] = vol / (ph + ps) if (ph + ps) != 0 else 0.0

    if target_atom_for_charge is not None:
        out.update({f"{prefix}{k}": float(v) for k, v in charge_sums_from_atoms(target_atom_for_charge, atoms).items()})

    out[f"{prefix}backbone_ratio"] = compute_backbone_ratio(atoms)
    out[f"{prefix}mean_b_factor"] = compute_mean_bfactor(atoms)

    return out

def shell_features_for_ring_atoms(ns: NeighborSearch, res, shells: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for atom_name in RING_ATOMS:
        if atom_name not in res:
            continue
        target = res[atom_name]
        for r in shells:
            hits = atoms_within_sphere(ns, target.get_coord(), r)
            hits = filter_atoms_to_protein(hits)

            stats = compute_contact_stats(target, hits)
            out.update({f"{atom_name}_r{r}_{k}": float(v) for k, v in stats.items()})

            res_list = region_residue_list_from_atoms(hits)
            resnames = [rr.get_resname() for rr in res_list]
            gc = compute_group_counts(resnames)
            out.update({f"{atom_name}_r{r}_{k}": float(v) for k, v in gc.items()})

            ac = compute_atom_counts(hits)
            out.update({f"{atom_name}_r{r}_{k}": float(v) for k, v in ac.items()})

            out.update({f"{atom_name}_r{r}_{k}": float(v) for k, v in charge_sums_from_atoms(target, hits).items()})

            out[f"{atom_name}_r{r}_backbone_ratio"] = compute_backbone_ratio(hits)
            out[f"{atom_name}_r{r}_mean_b_factor"] = compute_mean_bfactor(hits)
    return out


def read_structure_any(pdb_id: str, path: str):
    """
    Parse .pdb/.ent with PDBParser, .cif with MMCIFParser.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".cif", ".mmcif"]:
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(pdb_id, path)
    else:
        parser = PDBParser(QUIET=True)
        return parser.get_structure(pdb_id, path)


def extract_features_for_pdb(
    pdb_path: str,
    pdb_id: str,
    label_rows: pd.DataFrame,
    r1: float,
    r2: float,
    shells: List[float],
) -> List[Dict[str, Any]]:
    try:
        structure = read_structure_any(pdb_id, pdb_path)
    except Exception:
        return []

    try:
        model = next(structure.get_models())
    except StopIteration:
        return []

    ns = build_neighbor_search(model)

    ss_map = compute_dssp_secondary_structure(structure, pdb_path) if have_cmd("mkdssp") else None
    sasa_map = compute_sasa_per_residue(pdb_path)  # None if not installed

    # Global protein region
    prot_res = protein_residue_list(model)
    prot_atoms = [a for a in model.get_atoms() if is_standard_aa(a.get_parent())]
    prot_feats = region_features("Protein_", prot_res, prot_atoms, target_atom_for_charge=None)
    prot_feats.update(ss_fractions(prot_res, ss_map))
    prot_feats.update({f"Protein_{k}": v for k, v in region_sasa(prot_res, sasa_map).items()})

    # PDB quality (only for PDB-like text files; CIF often won't have these lines)
    resolution = parse_resolution_from_pdb_text(pdb_path) if pdb_path.lower().endswith((".pdb", ".ent")) else FLOPPY_NANS
    method = parse_method_from_pdb_text(pdb_path) if pdb_path.lower().endswith((".pdb", ".ent")) else ""

    # Find cofactor instances
    cofs = find_cofactor_instances(model)
    if not cofs:
        return []

    out_rows: List[Dict[str, Any]] = []

    # IMPORTANT: label_rows already filtered for this PDB ID by caller
    for cof in cofs:
        cof_res = get_residue_by_id(model, cof.chain_id, cof.res_id)
        bary = ring_barycenter(cof_res)
        if bary is None:
            continue

        # Bar region: sphere around barycenter
        bar_atoms = filter_atoms_to_protein(atoms_within_sphere(ns, bary, r1))
        bar_residues = region_residue_list_from_atoms(bar_atoms)

        # Ring union: union of spheres around ring atoms
        ring_atoms = filter_atoms_to_protein(ring_union_atoms(ns, cof_res, r2))
        ring_residues = region_residue_list_from_atoms(ring_atoms)

        # N5 features: nearest residue to N5, plus +/-1 sequence neighbors
        n5_atom = cof_res["N5"] if "N5" in cof_res else None
        n5_nearest_res = None
        around_n5_residues: List[Any] = []
        if n5_atom is not None:
            local_atoms = filter_atoms_to_protein(atoms_within_sphere(ns, n5_atom.get_coord(), 8.0))
            n5_nearest_res = nearest_residue_to_atom(n5_atom, local_atoms)
            if n5_nearest_res is not None:
                around_n5_residues = [n5_nearest_res]
                chain = n5_nearest_res.get_parent()
                around_n5_residues.extend(seq_neighbors(chain, residue_id_tuple(n5_nearest_res), k=1))

        base = {
            "pdb_id": pdb_id.upper(),
            "structure_path": pdb_path,
            "cofactor": cof.resname,
            "cof_chain": cof.chain_id,
            "cof_resseq": int(cof.res_id[1]),
            "cof_icode": str(cof.res_id[2]).strip() if cof.res_id[2] else "",
            "ring_atoms_present": ",".join(cof.ring_atoms_present),
            "r1_bar": float(r1),
            "r2_ring": float(r2),
            "pdb_resolution_A": float(resolution) if not (isinstance(resolution, float) and np.isnan(resolution)) else FLOPPY_NANS,
            "pdb_method": method,
            "has_dssp": bool(ss_map is not None),
            "has_freesasa": bool(sasa_map is not None),
        }
        base.update(prot_feats)

        # Region features (Bar / Ring) with charge proxies relative to N5 if present
        bar_feats = region_features("Bar_", bar_residues, bar_atoms, target_atom_for_charge=n5_atom)
        ring_feats = region_features("Ring_", ring_residues, ring_atoms, target_atom_for_charge=n5_atom)

        # DSSP + SASA for regions
        bar_feats.update({f"Bar_{k}": v for k, v in ss_fractions(bar_residues, ss_map).items()})
        ring_feats.update({f"Ring_{k}": v for k, v in ss_fractions(ring_residues, ss_map).items()})
        bar_feats.update({f"Bar_{k}": v for k, v in region_sasa(bar_residues, sasa_map).items()})
        ring_feats.update({f"Ring_{k}": v for k, v in region_sasa(ring_residues, sasa_map).items()})

        base.update(bar_feats)
        base.update(ring_feats)

        # Shell features around ring atoms
        base.update(shell_features_for_ring_atoms(ns, cof_res, shells))

        # N5_nearest and Around_N5
        if n5_nearest_res is not None:
            n5_rn = n5_nearest_res.get_resname()
            base["N5_nearest_resname"] = n5_rn
            p = sum_props_for_resnames([n5_rn])
            for k, v in p.items():
                base[f"N5_nearest_{k}"] = float(v)
        else:
            base["N5_nearest_resname"] = ""

        if around_n5_residues:
            rns = [r.get_resname() for r in around_n5_residues]
            p = sum_props_for_resnames(rns)
            for k, v in p.items():
                base[f"Around_N5_{k}"] = float(v)
            gc = compute_group_counts(rns)
            for k, v in gc.items():
                base[f"Around_N5_{k}"] = float(v)

        # Attach labels: one row per label row for this PDB
        for _, lab in label_rows.iterrows():
            row = dict(base)
            row["uniprot_id"] = lab.get("uniprot_id", "")
            # optional cols
            if "Em" in lab.index:
                row["Em"] = lab["Em"]
            if "pH" in lab.index:
                row["pH"] = lab["pH"]
            if "in_jcim" in lab.index:
                row["in_jcim"] = lab["in_jcim"]
            out_rows.append(row)

    return out_rows


# -----------------------------
# PDB retrieval / discovery (FIXED)
# -----------------------------
def ensure_pdb_file(pdb_id: str, pdb_dir: str) -> Optional[str]:
    """
    Find an existing structure file for pdb_id inside pdb_dir, or download it.

    Supports:
      - pdbXXXX.ent
      - pdbXXXX.ent.gz
      - XXXX.pdb / XXXX.ent / XXXX.cif / XXXX.pdb.gz / XXXX.cif.gz (any case)
      - nested directories under pdb_dir

    Returns a path to a readable *unzipped* file, or None.
    """
    os.makedirs(pdb_dir, exist_ok=True)
    pid = pdb_id.strip().lower()

    def _maybe_unzip(path: str) -> Optional[str]:
        if not path.endswith(".gz"):
            return path
        out_path = path[:-3]
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
        try:
            with gzip.open(path, "rb") as f_in, open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            return out_path
        except Exception:
            return None

    def _find_existing() -> Optional[str]:
        # Fast path common names in pdb_dir root
        candidates = [
            os.path.join(pdb_dir, f"pdb{pid}.ent"),
            os.path.join(pdb_dir, f"pdb{pid}.ent.gz"),
            os.path.join(pdb_dir, f"{pid}.pdb"),
            os.path.join(pdb_dir, f"{pid}.ent"),
            os.path.join(pdb_dir, f"{pid}.cif"),
            os.path.join(pdb_dir, f"{pid}.pdb.gz"),
            os.path.join(pdb_dir, f"{pid}.ent.gz"),
            os.path.join(pdb_dir, f"{pid}.cif.gz"),
            os.path.join(pdb_dir, f"{pid.upper()}.pdb"),
            os.path.join(pdb_dir, f"{pid.upper()}.ent"),
            os.path.join(pdb_dir, f"{pid.upper()}.cif"),
            os.path.join(pdb_dir, f"{pid.upper()}.pdb.gz"),
            os.path.join(pdb_dir, f"{pid.upper()}.ent.gz"),
            os.path.join(pdb_dir, f"{pid.upper()}.cif.gz"),
        ]

        for c in candidates:
            if os.path.exists(c) and os.path.getsize(c) > 0:
                return _maybe_unzip(c)

        # Recursive search under pdb_dir (supports nested layouts)
        patterns = [
            f"**/*{pid}*.pdb",
            f"**/*{pid}*.ent",
            f"**/*{pid}*.cif",
            f"**/*{pid}*.pdb.gz",
            f"**/*{pid}*.ent.gz",
            f"**/*{pid}*.cif.gz",
            f"**/*{pid.upper()}*.pdb",
            f"**/*{pid.upper()}*.ent",
            f"**/*{pid.upper()}*.cif",
            f"**/*{pid.upper()}*.pdb.gz",
            f"**/*{pid.upper()}*.ent.gz",
            f"**/*{pid.upper()}*.cif.gz",
        ]
        for pat in patterns:
            for c in glob.glob(os.path.join(pdb_dir, pat), recursive=True):
                if os.path.exists(c) and os.path.getsize(c) > 0:
                    p = _maybe_unzip(c)
                    if p is not None and os.path.exists(p) and os.path.getsize(p) > 0:
                        return p
        return None

    # 1) Try to find anything already on disk
    existing = _find_existing()
    if existing is not None:
        return existing

    # 2) Download if not found locally (Biopython PDBList)
    try:
        pdbl = PDBList(obsolete=False)
        # Prefer PDB format; you can swap to "mmCif" if you want CIF.
        pdbl.retrieve_pdb_file(pid, pdir=pdb_dir, file_format="pdb")
    except Exception:
        pass

    existing = _find_existing()
    if existing is not None:
        return existing

    # 3) Direct HTTPS download from RCSB (avoids FTP issues)
    pid_up = pid.upper()
    urls = [
        (f"https://files.rcsb.org/download/{pid_up}.pdb", os.path.join(pdb_dir, f"{pid_up}.pdb")),
        (f"https://files.rcsb.org/download/{pid_up}.cif", os.path.join(pdb_dir, f"{pid_up}.cif")),
    ]
    for url, out_path in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200 or not r.content:
                continue
            with open(out_path, "wb") as f:
                f.write(r.content)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        except Exception:
            continue

    return None


# -----------------------------
# CSV normalization
# -----------------------------
def normalize_em_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            if name in df.columns:
                return name
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    rename = {}
    pdb_col = pick("pdb_id", "pdb", "pdbid")
    if pdb_col:
        rename[pdb_col] = "pdb_id"
    uniprot_col = pick("uniprot_id", "uniprot", "uniprotid")
    if uniprot_col:
        rename[uniprot_col] = "uniprot_id"
    em_col = pick("Em", "em", "e_m")
    if em_col:
        rename[em_col] = "Em"
    ph_col = pick("pH", "ph")
    if ph_col:
        rename[ph_col] = "pH"
    jcim_col = pick("in_jcim", "jcim", "inJCIM")
    if jcim_col:
        rename[jcim_col] = "in_jcim"

    df = df.rename(columns=rename)

    # If missing optional columns, add them so downstream code is happy
    if "uniprot_id" not in df.columns:
        df["uniprot_id"] = ""
    if "Em" not in df.columns:
        df["Em"] = np.nan
    if "pH" not in df.columns:
        df["pH"] = np.nan
    if "in_jcim" not in df.columns:
        df["in_jcim"] = np.nan

    return df


# -----------------------------
# Parallel runner
# -----------------------------
def process_one_pdb(args_tuple):
    pdb_id, group_df, pdb_dir, r1, r2, shells = args_tuple
    pdb_path = ensure_pdb_file(pdb_id, pdb_dir)
    if pdb_path is None:
        return [], pdb_id
    rows = extract_features_for_pdb(pdb_path, pdb_id, group_df, r1, r2, shells)
    return rows, None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # Accept both --em and --validation as aliases (your earlier runs used --validation)
    ap.add_argument("--em", required=False, default="/home/ubuntu/OOK/em.csv",
                    help="CSV with labels (pdb_id required; uniprot_id/Em/pH optional)")
    ap.add_argument("--validation", required=False, default=None,
                    help="Alias for --em (older command style)")
    ap.add_argument("--out", required=False, default="/home/ubuntu/OOK/ml_ready_features.csv",
                    help="Output CSV path")
    ap.add_argument("--pdb-dir", required=False, default="/home/ubuntu/OOK/pdbs",
                    help="Directory containing/caching structure files")
    ap.add_argument("--r1", type=float, default=12.0, help="Bar radius around ring barycenter (A)")
    ap.add_argument("--r2", type=float, default=6.0, help="Ring radius around ring atoms (A)")
    ap.add_argument("--shells", default="3,6,9,12", help="Comma-separated shell radii (A)")

    # Accept both --workers and --n-jobs as aliases (your earlier runs used --n-jobs)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--n-jobs", type=int, default=None, help="Alias for --workers")
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit number of unique PDB IDs (0 = no limit)")
    ap.add_argument("--no-parallel", action="store_true", help="Disable multiprocessing (debug)")
    args = ap.parse_args()

    if args.validation:
        args.em = args.validation
    if args.n_jobs is not None:
        args.workers = int(args.n_jobs)

    shells = [float(x.strip()) for x in args.shells.split(",") if x.strip()]

    if not os.path.exists(args.em):
        raise SystemExit(f"Label CSV not found: {args.em}")

    df = pd.read_csv(args.em)
    df = normalize_em_columns(df)

    if "pdb_id" not in df.columns:
        raise SystemExit("Label CSV must contain a pdb_id column (or a column that can be normalized to pdb_id).")

    # Normalize types
    df["pdb_id"] = df["pdb_id"].astype(str).str.upper().str.strip()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["Em"] = pd.to_numeric(df["Em"], errors="coerce")
    df["pH"] = pd.to_numeric(df["pH"], errors="coerce")

    # Normalize in_jcim if present (otherwise stays NaN)
    truthy = {"true", "t", "yes", "y", "1"}
    falsy = {"false", "f", "no", "n", "0"}
    def coerce_jcim(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in truthy:
            return True
        if s in falsy:
            return False
        return np.nan
    df["in_jcim"] = df["in_jcim"].apply(coerce_jcim)

    # Drop empty pdb IDs
    df = df[df["pdb_id"].astype(str).str.len() > 0].copy()

    pdb_ids = sorted(df["pdb_id"].unique().tolist())
    if args.limit and args.limit > 0:
        pdb_ids = pdb_ids[: args.limit]

    log(f"validation rows: {len(df)}")
    log(f"unique pdb_ids: {len(pdb_ids)}")
    log(f"freesasa installed: {have_freesasa_py()}")
    log(f"mkdssp available: {have_cmd('mkdssp')}")
    log(f"pdb-dir: {args.pdb_dir}")
    log(f"workers: {args.workers} (no-parallel={args.no_parallel})")

    jobs = []
    for pid in pdb_ids:
        g = df[df["pdb_id"] == pid].copy()
        jobs.append((pid, g, args.pdb_dir, args.r1, args.r2, shells))

    all_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    if args.no_parallel or args.workers <= 1:
        for i, (pid, g, pdb_dir, r1, r2, sh) in enumerate(jobs, start=1):
            pdb_path = ensure_pdb_file(pid, pdb_dir)
            if pdb_path is None:
                missing.append(pid)
                log(f"{i}/{len(jobs)} {pid}: MISSING structure file")
                continue
            rows = extract_features_for_pdb(pdb_path, pid, g, r1, r2, sh)
            all_rows.extend(rows)
            if i % 10 == 0 or i == len(jobs):
                log(f"done {i}/{len(jobs)} PDBs | rows so far: {len(all_rows)}")
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (rows, miss) in enumerate(ex.map(process_one_pdb, jobs), start=1):
                if miss is not None:
                    missing.append(miss)
                    log(f"{i}/{len(jobs)} {miss}: MISSING structure file")
                else:
                    if rows:
                        all_rows.extend(rows)
                if i % 10 == 0 or i == len(jobs):
                    log(f"done {i}/{len(jobs)} PDBs | rows so far: {len(all_rows)}")

    log(f"total feature rows (per cofactor instance, per label row): {len(all_rows)}")
    if missing:
        log(f"missing pdb files: {len(missing)} -> {missing[:10]}{'...' if len(missing) > 10 else ''}")

    if not all_rows:
        raise ValueError("No feature rows produced. Check that your structures exist and contain FAD/FMNs.")

    out_df = pd.DataFrame(all_rows)

    # Put identifiers + labels first
    front = [c for c in ["uniprot_id", "pdb_id", "Em", "pH", "in_jcim", "cofactor", "cof_chain", "cof_resseq", "cof_icode"] if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + sorted(rest)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    log(f"Wrote {len(out_df)} rows, {len(out_df.columns)} columns -> {args.out}")


if __name__ == "__main__":
    main()
