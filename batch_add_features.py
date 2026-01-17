#!/usr/bin/env python3
"""
Batch augment features.json files with additional metadata/confidence/regime
features for redox-potential prediction and aggregate them into a CSV.

CLI:
    python batch_add_features.py --root ./fad --out_csv ./fad/all_features.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

# Optional RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    RDKit_AVAILABLE = False

LOG = logging.getLogger("batch_add_features")
TOOL_CACHE: Dict[str, bool] = {}


# ---------------------------- Data structures ---------------------------- #
@dataclass
class AtomRecord:
    name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coords: np.ndarray
    element: str
    het: bool
    altloc: str


# ---------------------------- Utility helpers ---------------------------- #
def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(ch)


def atomic_write_json(path: Path, data: dict) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return []


def has_header_markers(lines: List[str]) -> bool:
    header_tokens = ("HEADER", "EXPDTA", "REMARK")
    return any(line.lstrip().startswith(header_tokens) for line in lines)


def parse_resolution(lines: List[str]) -> Optional[float]:
    resolution = None
    remark_pattern = re.compile(r"REMARK\s+2\s+RESOLUTION\.*\s+([\d\.]+)\s+ANGSTROM", re.IGNORECASE)
    for line in lines:
        m = remark_pattern.search(line)
        if m:
            try:
                resolution = float(m.group(1))
                break
            except ValueError:
                continue
    return resolution


def detect_bfactors(lines: List[str]) -> bool:
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            segment = line[60:66]
            if segment.strip() != "":
                return True
    return False


def parse_pdb_atoms(path: Path) -> List[AtomRecord]:
    """Parse PDB and return AtomRecords (keeps het/water atoms)."""
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("pdb", str(path))
    except Exception as exc:  # pragma: no cover - robust to malformed files
        LOG.warning("Failed to parse PDB %s: %s", path, exc)
        return []

    atoms: List[AtomRecord] = []
    for atom in structure.get_atoms():
        res = atom.get_parent()
        hetflag, resseq, icode = res.id
        het = hetflag != " "
        resname = res.get_resname().strip()
        chain = res.get_parent().id if res.get_parent() else ""
        element = (atom.element or "").upper().strip()
        altloc = atom.get_altloc().strip() if hasattr(atom, "get_altloc") else ""
        atoms.append(
            AtomRecord(
                name=atom.get_name().strip(),
                resname=resname,
                chain=chain,
                resseq=int(resseq) if resseq is not None else 0,
                icode=str(icode).strip(),
                coords=atom.get_coord().astype(float),
                element=element,
                het=het,
                altloc=altloc,
            )
        )
    return atoms


def split_atoms(atoms: Iterable[AtomRecord]) -> Tuple[List[AtomRecord], List[AtomRecord], List[AtomRecord]]:
    protein: List[AtomRecord] = []
    ligand: List[AtomRecord] = []
    waters: List[AtomRecord] = []
    for atom in atoms:
        if atom.het:
            if atom.resname in {"HOH", "WAT"}:
                waters.append(atom)
            else:
                ligand.append(atom)
        else:
            protein.append(atom)
    return protein, ligand, waters


def ligand_residue_groups(ligand_atoms: List[AtomRecord]) -> Dict[Tuple[str, str, int, str], List[AtomRecord]]:
    groups: Dict[Tuple[str, str, int, str], List[AtomRecord]] = {}
    for atom in ligand_atoms:
        key = (atom.resname, atom.chain, atom.resseq, atom.icode)
        groups.setdefault(key, []).append(atom)
    return groups


def largest_ligand_group(ligand_atoms: List[AtomRecord]) -> List[AtomRecord]:
    if not ligand_atoms:
        return []
    groups = ligand_residue_groups(ligand_atoms)
    return max(groups.values(), key=len)


def ring_centroid_from_iso_atoms(ligand_atoms: List[AtomRecord]) -> Optional[np.ndarray]:
    iso_names = {
        "N1",
        "N3",
        "N5",
        "O2",
        "O4",
        "C2",
        "C4",
        "C4A",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C8M",
        "C6M",
        "C7M",
    }
    coords = [a.coords for a in ligand_atoms if a.name.upper() in iso_names]
    if len(coords) >= 4:
        return np.mean(coords, axis=0)
    return None


def ring_centroid_rdkit(sdf_path: Path) -> Optional[np.ndarray]:
    if not RDKit_AVAILABLE or not sdf_path.exists():
        return None
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mol = supplier[0] if supplier else None
        if mol is None:
            return None
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        if not rings:
            return None
        largest_ring = max(rings, key=len)
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(idx) for idx in largest_ring], dtype=float)
        return coords.mean(axis=0)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("RDKit ring centroid failed for %s: %s", sdf_path, exc)
        return None


def ring_centroid_dense_subset(ligand_atoms: List[AtomRecord]) -> Optional[np.ndarray]:
    if len(ligand_atoms) < 3:
        return None
    coords = np.array([a.coords for a in ligand_atoms if a.element != "H"])
    if len(coords) < 3:
        return None
    # Dense subset heuristic: atoms with smallest mean distance to k nearest neighbors
    tree = cKDTree(coords)
    k = min(6, len(coords) - 1)
    dists, _ = tree.query(coords, k=k + 1)
    scores = dists[:, 1:].mean(axis=1)  # skip self
    n_subset = max(3, int(0.4 * len(coords)))
    subset_idx = np.argsort(scores)[:n_subset]
    return coords[subset_idx].mean(axis=0)


def compute_ring_centroid(
    ligand_atoms: List[AtomRecord], sdf_path: Path, methods_meta: Dict[str, str], errors: List[str]
) -> Optional[np.ndarray]:
    centroid = ring_centroid_from_iso_atoms(ligand_atoms)
    if centroid is not None:
        methods_meta["ring_centroid_method"] = "isoalloxazine_atom_names"
        return centroid
    centroid = ring_centroid_rdkit(sdf_path)
    if centroid is not None:
        methods_meta["ring_centroid_method"] = "rdkit_largest_ring"
        return centroid
    centroid = ring_centroid_dense_subset(ligand_atoms)
    if centroid is not None:
        methods_meta["ring_centroid_method"] = "dense_subset_proxy"
        return centroid
    errors.append("ring_centroid_unavailable")
    methods_meta["ring_centroid_method"] = "unavailable"
    return None


def kd_tree(coords: List[np.ndarray]) -> Optional[cKDTree]:
    if not coords:
        return None
    return cKDTree(np.array(coords))


def is_tool_available(name: str) -> bool:
    if name in TOOL_CACHE:
        return TOOL_CACHE[name]
    TOOL_CACHE[name] = shutil.which(name) is not None
    return TOOL_CACHE[name]


def run_subprocess_capture(cmd: List[str], timeout: int = 60, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def detect_structure_source(
    protein_lines: List[str], complex_lines: List[str], has_fad: bool
) -> str:
    protein_header = has_header_markers(protein_lines)
    complex_header = has_header_markers(complex_lines)
    vina_like = any("VINA" in line.upper() for line in complex_lines if line.startswith("REMARK"))
    if (vina_like or not complex_header) and complex_lines:
        return "docked"
    if has_fad and protein_header:
        return "holo_pdb"
    return "unknown"


def cofactor_from_complex(ligand_atoms: List[AtomRecord]) -> str:
    resnames = {a.resname.upper() for a in ligand_atoms}
    if "FAD" in resnames:
        return "FAD"
    if "FMN" in resnames or "FMN " in resnames:
        return "FMN"
    if resnames:
        return "other"
    return "none"


def parse_sdf_atoms(sdf_path: Path) -> List[str]:
    if not sdf_path.exists():
        return []
    if RDKit_AVAILABLE:
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
            mol = supplier[0] if supplier else None
            if mol:
                return [atom.GetSymbol() for atom in mol.GetAtoms()]
        except Exception:  # pragma: no cover
            pass
    # fallback minimal parser (V2000/3000 counts line)
    lines = read_lines(sdf_path)
    atom_lines = []
    started = False
    atom_count = 0
    for idx, line in enumerate(lines):
        if idx == 3:
            try:
                atom_count = int(line[0:3])
            except Exception:
                pass
            started = True
            continue
        if started and atom_count > 0:
            atom_lines = lines[4 : 4 + atom_count]
            break
    symbols: List[str] = []
    for l in atom_lines:
        if len(l) >= 34:
            symbols.append(l[31:34].strip())
    return symbols


def predict_cofactor_from_sdf(sdf_path: Path, methods: Dict[str, str]) -> Tuple[str, float, Dict[str, int]]:
    symbols = parse_sdf_atoms(sdf_path)
    counts = {"total_atoms": len(symbols), "P": symbols.count("P"), "N": symbols.count("N"), "O": symbols.count("O")}
    pred = "unknown"
    if counts["total_atoms"] >= 45 and counts["P"] >= 2 and counts["N"] >= 5:
        pred = "FAD_like"
    elif counts["total_atoms"] >= 22 and counts["P"] >= 1:
        pred = "FAD_like"
    elif counts["total_atoms"] >= 18:
        pred = "FMN_like"
    size_score = min(1.0, counts["total_atoms"] / 50.0) if counts["total_atoms"] else 0.0
    conf = 0.0
    if pred == "FAD_like":
        conf = min(1.0, 0.4 + 0.4 * size_score + 0.1 * counts["P"])
    elif pred == "FMN_like":
        conf = min(1.0, 0.3 + 0.5 * size_score)
    else:
        conf = 0.1 * size_score
    methods["cofactor_pred_method"] = "rdkit" if RDKit_AVAILABLE else "heuristic_counts"
    return pred, conf, counts


def centroid_of_atoms(atoms: List[AtomRecord]) -> Optional[np.ndarray]:
    if not atoms:
        return None
    return np.mean([a.coords for a in atoms], axis=0)


def find_tail_centroids(
    ligand_atoms: List[AtomRecord], ring_centroid: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not ligand_atoms:
        return None, None, "no_ligand"
    # Adenine centroid: atoms farthest from flavin ring
    coords = np.array([a.coords for a in ligand_atoms if a.element != "H"])
    if ring_centroid is None or len(coords) == 0:
        return None, None, "missing_ring_centroid"
    distances = np.linalg.norm(coords - ring_centroid, axis=1)
    if len(distances) == 0:
        return None, None, "missing_ring_centroid"
    far_fraction = max(3, int(0.2 * len(coords)))
    adenine_coords = coords[np.argsort(distances)[-far_fraction:]]
    adenine_centroid = adenine_coords.mean(axis=0) if len(adenine_coords) else None

    # Pyrophosphate centroid: use phosphorus atoms if present, else farthest oxygens
    p_atoms = [a.coords for a in ligand_atoms if a.element == "P"]
    if p_atoms:
        pyrophos_centroid = np.mean(p_atoms, axis=0)
        method = "phosphorus_atoms"
    else:
        oxy_atoms = [a.coords for a in ligand_atoms if a.element == "O"]
        if oxy_atoms:
            oxy_coords = np.array(oxy_atoms)
            distances_o = np.linalg.norm(oxy_coords - ring_centroid, axis=1)
            select = oxy_coords[np.argsort(distances_o)[-far_fraction:]] if len(oxy_coords) >= far_fraction else oxy_coords
            pyrophos_centroid = select.mean(axis=0)
            method = "oxygen_far_proxy"
        else:
            pyrophos_centroid = None
            method = "unavailable"
    return adenine_centroid, pyrophos_centroid, method


def count_contacts(tree: Optional[cKDTree], point: np.ndarray, cutoff: float) -> int:
    if tree is None:
        return 0
    idxs = tree.query_ball_point(point, r=cutoff)
    return len(idxs)


def ligand_protein_clashes(
    protein_tree: Optional[cKDTree], ligand_coords: np.ndarray
) -> Tuple[int, Optional[float]]:
    if protein_tree is None or ligand_coords.size == 0:
        return 0, None
    clash_count = 0
    min_dist = None
    for coord in ligand_coords:
        d, _ = protein_tree.query(coord, k=1)
        if min_dist is None or d < min_dist:
            min_dist = float(d)
        if d <= 2.0:
            clash_count += 1
    return clash_count, min_dist


def ligand_burial_proxy(
    protein_tree: Optional[cKDTree], protein_coords: List[np.ndarray], ligand_coords: np.ndarray, methods: Dict[str, str]
) -> Optional[float]:
    if ligand_coords.size == 0:
        return None
    if protein_tree is None or not protein_coords:
        methods["burial_method"] = "proxy_no_protein"
        return None
    neighborhood = []
    for coord in ligand_coords:
        neighbors = protein_tree.query_ball_point(coord, r=6.0)
        neighborhood.append(len(neighbors))
    if not neighborhood:
        methods["burial_method"] = "proxy_no_neighbors"
        return None
    # Normalize neighbor density to [0,1] using a soft cap of 60 neighbors
    density = np.clip(np.array(neighborhood) / 60.0, 0.0, 1.0)
    methods["burial_method"] = "neighbor_density_proxy"
    return float(density.mean())


def pocket_depth(
    protein_coords: List[np.ndarray], ring_centroid: Optional[np.ndarray]
) -> Optional[float]:
    if ring_centroid is None or not protein_coords:
        return None
    tree = cKDTree(np.array(protein_coords))
    # Surface-like atoms: those with few neighbors within 6 Å
    neighbors = tree.query_ball_point(protein_coords, r=6.0)
    surface_mask = [len(nbrs) <= 15 for nbrs in neighbors]
    if not any(surface_mask):
        return None
    surface_coords = np.array(protein_coords)[surface_mask]
    dists = np.linalg.norm(surface_coords - ring_centroid, axis=1)
    if dists.size == 0:
        return None
    return float(dists.min())


def parse_vina_log(log_path: Path) -> Dict[str, Optional[float]]:
    result = {
        "vina_best_score": None,
        "vina_second_score": None,
        "vina_gap": None,
        "vina_exhaustiveness": None,
        "vina_num_modes": None,
    }
    if not log_path.exists():
        return result
    lines = read_lines(log_path)
    scores = []
    for line in lines:
        m = re.search(r"REMARK\s+VINA\s+RESULT:\s*([-\\d\\.]+)", line)
        if m:
            try:
                scores.append(float(m.group(1)))
            except ValueError:
                continue
        m = re.search(r"^\s*\d+\s+([-\\d\\.]+)", line.strip())
        if "-----+------------+----------+----------" in line:
            # skip table header
            continue
        if scores and len(scores) >= 2:
            break
    if scores:
        scores_sorted = sorted(scores)
        result["vina_best_score"] = scores_sorted[0]
        if len(scores_sorted) > 1:
            result["vina_second_score"] = scores_sorted[1]
            result["vina_gap"] = scores_sorted[1] - scores_sorted[0]
    for line in lines:
        if "exhaustiveness" in line.lower():
            nums = re.findall(r"\\d+", line)
            if nums:
                result["vina_exhaustiveness"] = float(nums[0])
        if "num_modes" in line.lower():
            nums = re.findall(r"\\d+", line)
            if nums:
                result["vina_num_modes"] = float(nums[0])
    return result


def hbond_features(
    site_name: str,
    site_coord: np.ndarray,
    protein_atoms: List[AtomRecord],
    methods: Dict[str, str],
) -> Dict[str, Optional[float]]:
    donor_elements = {"N", "S"}
    acceptor_elements = {"O"}
    max_dist = 3.6
    donor_count = 0
    acceptor_count = 0
    nearest = None
    for atom in protein_atoms:
        if atom.element not in donor_elements and atom.element not in acceptor_elements:
            continue
        dist = np.linalg.norm(atom.coords - site_coord)
        if dist <= max_dist:
            if atom.element in donor_elements:
                donor_count += 1
            if atom.element in acceptor_elements:
                acceptor_count += 1
            if nearest is None or dist < nearest:
                nearest = dist
    methods["hbonds_method"] = "heavy_atom_proxy"
    return {
        f"hb_{site_name}_count": donor_count + acceptor_count,
        f"hb_{site_name}_nearest_dist": float(nearest) if nearest is not None else None,
        f"hb_{site_name}_nearest_angle": None,
        f"hb_{site_name}_donor_count": donor_count,
        f"hb_{site_name}_acceptor_count": acceptor_count,
    }


def residue_charge(resname: str, his_charge: float = 0.1) -> float:
    res = resname.upper()
    if res in {"ASP", "GLU"}:
        return -1.0
    if res in {"LYS", "ARG"}:
        return 1.0
    if res == "HIS":
        return his_charge
    return 0.0


def residue_centroids(protein_atoms: List[AtomRecord]) -> Dict[Tuple[str, str, int, str], np.ndarray]:
    residues: Dict[Tuple[str, str, int, str], List[np.ndarray]] = {}
    for atom in protein_atoms:
        key = (atom.resname, atom.chain, atom.resseq, atom.icode)
        residues.setdefault(key, []).append(atom.coords)
    centroids: Dict[Tuple[str, str, int, str], np.ndarray] = {}
    for key, coords in residues.items():
        centroids[key] = np.mean(coords, axis=0)
    return centroids


def pocket_charge_features(
    protein_atoms: List[AtomRecord],
    ring_centroid: Optional[np.ndarray],
    pH: Optional[float],
    propka_charges: Optional[Dict[Tuple[str, str, int, str], float]] = None,
) -> Dict[str, Optional[float]]:
    if ring_centroid is None or not protein_atoms:
        return {
            "pocket_histidine_count_6A": None,
            "pocket_acidic_count_6A": None,
            "pocket_basic_count_6A": None,
            "net_charge_4A": None,
            "net_charge_6A": None,
            "net_charge_8A": None,
            "net_charge_10A": None,
            "net_charge_pH_4A": None,
            "net_charge_pH_6A": None,
            "net_charge_pH_8A": None,
            "net_charge_pH_10A": None,
        }
    residues = residue_centroids(protein_atoms)
    radii = [4.0, 6.0, 8.0, 10.0]
    his_charge_default = 0.1
    his_charge_ph = his_charge_default
    if pH is not None:
        # Henderson–Hasselbalch for His (pKa ~6.0)
        his_charge_ph = 1.0 / (1.0 + 10 ** (pH - 6.0))
    net_charge = {r: 0.0 for r in radii}
    net_charge_ph = {r: 0.0 for r in radii}
    net_charge_propka = {r: None for r in radii}
    histidine_count = 0
    acidic_count = 0
    basic_count = 0
    for (resname, chain, resseq, icode), centroid in residues.items():
        dist = np.linalg.norm(centroid - ring_centroid)
        charge_default = residue_charge(resname, his_charge_default)
        charge_ph = residue_charge(resname, his_charge_ph)
        if resname.upper() == "HIS" and dist <= 6.0:
            histidine_count += 1
        if resname.upper() in {"ASP", "GLU"} and dist <= 6.0:
            acidic_count += 1
        if resname.upper() in {"LYS", "ARG", "HIS"} and dist <= 6.0:
            basic_count += 1
        for r in radii:
            if dist <= r:
                net_charge[r] += charge_default
                net_charge_ph[r] += charge_ph
                if propka_charges is not None and (resname, chain, resseq, icode) in propka_charges:
                    if net_charge_propka[r] is None:
                        net_charge_propka[r] = 0.0
                    net_charge_propka[r] += propka_charges[(resname, chain, resseq, icode)]
    return {
        "pocket_histidine_count_6A": histidine_count,
        "pocket_acidic_count_6A": acidic_count,
        "pocket_basic_count_6A": basic_count,
        "net_charge_4A": net_charge[4.0],
        "net_charge_6A": net_charge[6.0],
        "net_charge_8A": net_charge[8.0],
        "net_charge_10A": net_charge[10.0],
        "net_charge_pH_4A": net_charge_ph[4.0],
        "net_charge_pH_6A": net_charge_ph[6.0],
        "net_charge_pH_8A": net_charge_ph[8.0],
        "net_charge_pH_10A": net_charge_ph[10.0],
        "net_charge_propka_4A": net_charge_propka[4.0],
        "net_charge_propka_6A": net_charge_propka[6.0],
        "net_charge_propka_8A": net_charge_propka[8.0],
        "net_charge_propka_10A": net_charge_propka[10.0],
    }


def update_summary(summary_path: Path, features_extra: Dict[str, object]) -> None:
    lines = read_lines(summary_path)
    section_header = "## Additional Features (features_extra)"
    table_lines = [section_header, "", "| Key | Value |", "|-----|-------|"]
    for key, value in sorted(features_extra.items()):
        if isinstance(value, float):
            val_str = f"{value:.4f}"
        else:
            val_str = json.dumps(value)
        table_lines.append(f"| {key} | {val_str} |")
    new_section = "\n".join(table_lines)
    if not lines:
        summary_path.write_text(new_section + "\n", encoding="utf-8")
        return
    joined = "\n".join(lines)
    if section_header in joined:
        start = joined.index(section_header)
        before = joined[:start].rstrip()
        after_part = joined[start:]
        # Try to keep any content after this section by finding the next header
        next_header_idx = after_part.find("\n## ", len(section_header))
        tail = after_part[next_header_idx + 1 :] if next_header_idx != -1 else ""
        updated_parts = [before, new_section]
        if tail.strip():
            updated_parts.append(tail.strip())
        updated = "\n\n".join(part for part in updated_parts if part) + "\n"
    else:
        updated = joined.rstrip() + "\n\n" + new_section + "\n"
    summary_path.write_text(updated, encoding="utf-8")


# ---------------------------- Hydrogen / PROPKA helpers ---------------------------- #
def add_hydrogens_with_pdb2pqr_or_reduce(
    protein_pdb: Path,
    tmp_dir: Path,
    gating_decisions: Dict[str, Dict[str, object]],
    errors: List[str],
    tool_versions: Dict[str, str],
) -> Tuple[Optional[Path], Optional[str], Dict[str, object]]:
    """Add hydrogens using pdb2pqr if available, else reduce. Returns (path, method, info)."""
    info: Dict[str, object] = {}
    hydrogenated = None
    method = None
    if not protein_pdb.exists():
        errors.append("protein_pdb_missing_for_h_add")
        gating_decisions["hydrogen_addition"] = {"passed": False, "reason": "protein_pdb_missing"}
        return None, None, info
    heavy_atoms = [a for a in parse_pdb_atoms(protein_pdb) if a.element != "H"]
    heavy_count = len(heavy_atoms)
    existing_h = len([a for a in parse_pdb_atoms(protein_pdb) if a.element == "H"])

    # Prefer pdb2pqr
    if is_tool_available("pdb2pqr"):
        tool_versions.setdefault("pdb2pqr", run_subprocess_capture(["pdb2pqr", "--version"], timeout=5)[1].strip())
        hydrogenated = tmp_dir / "protein_h.pdb"
        cmd = [
            "pdb2pqr",
            "--ff=PARSE",
            "--nodebump",
            "--keep-chain",
            "--drop-water",
            "--pdb-output",
            str(hydrogenated),
            str(protein_pdb),
        ]
        code, out, err = run_subprocess_capture(cmd, timeout=120)
        info["pdb2pqr_stdout"] = out[-4000:]
        info["pdb2pqr_stderr"] = err[-4000:]
        method = "pdb2pqr"
        if code != 0 or not hydrogenated.exists():
            errors.append("pdb2pqr_failed")
            gating_decisions["hydrogen_addition"] = {"passed": False, "reason": "pdb2pqr_failed"}
            return None, None, info
    elif is_tool_available("reduce"):
        tool_versions.setdefault("reduce", run_subprocess_capture(["reduce", "-h"], timeout=5)[1].strip())
        hydrogenated = tmp_dir / "protein_h.pdb"
        with open(hydrogenated, "w", encoding="utf-8") as h:
            code = subprocess.call(["reduce", "-BUILD", str(protein_pdb)], stdout=h, stderr=subprocess.DEVNULL)
        method = "reduce"
        if code != 0 or not hydrogenated.exists():
            errors.append("reduce_failed")
            gating_decisions["hydrogen_addition"] = {"passed": False, "reason": "reduce_failed"}
            return None, None, info
    else:
        gating_decisions["hydrogen_addition"] = {"passed": False, "reason": "no_hydrogen_tool_available"}
        errors.append("no_hydrogen_tool_available")
        return None, None, info

    atoms_h = parse_pdb_atoms(hydrogenated)
    h_count = len([a for a in atoms_h if a.element == "H"])
    added = h_count - existing_h
    if heavy_count == 0:
        gating_decisions["hydrogen_addition"] = {"passed": False, "reason": "no_heavy_atoms"}
        errors.append("H_add_failed_or_unreasonable")
        return None, None, info
    # Gate 2 sanity: 5%-70% of heavy atoms
    lower = 0.05 * heavy_count
    upper = 0.7 * heavy_count
    if added < lower or added > upper:
        gating_decisions["hydrogen_addition"] = {
            "passed": False,
            "reason": f"H_added_out_of_bounds({added} vs heavy {heavy_count})",
        }
        errors.append("H_add_failed_or_unreasonable")
        return None, None, info
    gating_decisions["hydrogen_addition"] = {"passed": True, "reason": ""}
    return hydrogenated, method, info


def parse_propka_output(pka_path: Path) -> Dict[Tuple[str, str, int, str], float]:
    """Parse PROPKA .pka output; returns mapping of residue to pKa."""
    pkas: Dict[Tuple[str, str, int, str], float] = {}
    if not pka_path.exists():
        return pkas
    lines = read_lines(pka_path)
    pattern = re.compile(r"^\s*([A-Z]{3})\s+([A-Za-z0-9])\s+(-?\d+)\s+([-0-9\\.]+)")
    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        resname, chain, resseq, pka = m.groups()
        try:
            pkas[(resname.upper(), chain, int(resseq), "")] = float(pka)
        except ValueError:
            continue
    return pkas


def propka_charge(resname: str, pka: float, pH: float) -> float:
    res = resname.upper()
    if res in {"ASP", "GLU", "CYS", "TYR"}:
        return -1.0 / (1.0 + 10 ** (pka - pH))
    if res in {"LYS", "ARG"}:
        return 1.0 / (1.0 + 10 ** (pH - pka))
    if res == "HIS":
        return 1.0 / (1.0 + 10 ** (pH - pka))
    return 0.0


def compute_hbond_angles_with_hydrogens(
    site_coord: np.ndarray, atoms: List[AtomRecord]
) -> Optional[float]:
    """Return nearest D-H-A angle (deg) for a site using explicit hydrogens."""
    hydrogen_atoms = [a for a in atoms if a.element == "H"]
    heavy_atoms = [a for a in atoms if a.element != "H"]
    if not hydrogen_atoms or not heavy_atoms:
        return None
    heavy_by_res: Dict[Tuple[str, str, int, str], List[AtomRecord]] = {}
    for a in heavy_atoms:
        key = (a.resname, a.chain, a.resseq, a.icode)
        heavy_by_res.setdefault(key, []).append(a)
    best_angle = None
    for h in hydrogen_atoms:
        key = (h.resname, h.chain, h.resseq, h.icode)
        donors = heavy_by_res.get(key, [])
        if not donors:
            continue
        donor = min(donors, key=lambda d: np.linalg.norm(d.coords - h.coords))
        if np.linalg.norm(donor.coords - h.coords) > 1.2:
            continue
        da_dist = np.linalg.norm(donor.coords - site_coord)
        if da_dist > 3.5:
            continue
        v1 = donor.coords - h.coords
        v2 = site_coord - h.coords
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            continue
        cosang = np.dot(v1, v2) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))
        if angle >= 120.0:
            if best_angle is None or da_dist < best_angle[0]:
                best_angle = (da_dist, angle)
    return best_angle[1] if best_angle else None


def apply_quality_gates_and_write(
    feat_path: Path, summary_path: Path, data: dict, features_extra: Dict[str, object], meta: Dict[str, object]
) -> None:
    """Persist updated data and summary after gates have been applied."""
    data["features_extra"] = features_extra
    data["features_extra_meta"] = meta
    atomic_write_json(feat_path, data)
    update_summary(summary_path, features_extra)


# ---------------------------- Core processing ---------------------------- #
def process_folder(folder: Path, args: argparse.Namespace) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Return updated features dict and row for CSV."""
    errors: List[str] = []
    methods: Dict[str, str] = {}
    features_extra: Dict[str, object] = {}
    gating_decisions: Dict[str, Dict[str, object]] = {}
    tool_versions: Dict[str, str] = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="feat_tmp_", dir=folder))
    try:

        feat_path = folder / "features.json"
        data = json.loads(feat_path.read_text(encoding="utf-8"))
        existing_features = data.get("features", {})
        status = data.get("status", "UNKNOWN")

        protein_pdb = folder / "protein_input.pdb"
        complex_pdb = folder / args.complex_name
        ligand_sdf = folder / "ligand_input.sdf"
        debug_log = folder / "debug.log"
        
        # Optional: skip if --only_if_redock_qc_pass and redock_qc_pass.txt doesn't exist
        if args.only_if_redock_qc_pass and not (folder / "redock_qc_pass.txt").exists():
            LOG.info("Skipping %s: redock_qc_pass.txt not found", folder.name)
            row = {"id": folder.name, "status": "SKIPPED", "reason": "redock_qc_pass_not_found"}
            return data, row

        protein_lines = read_lines(protein_pdb)
        complex_lines = read_lines(complex_pdb)

        pdb_header = has_header_markers(protein_lines)
        pdb_resolution = parse_resolution(protein_lines)
        bfactors_present = detect_bfactors(protein_lines)

        protein_atoms_all = parse_pdb_atoms(protein_pdb) if protein_pdb.exists() else []
        complex_atoms_all = parse_pdb_atoms(complex_pdb) if complex_pdb.exists() else []
        protein_atoms, ligand_atoms, waters = split_atoms(complex_atoms_all)
        ligand_atoms_main = largest_ligand_group(ligand_atoms)

        protein_atoms_count = len([a for a in protein_atoms_all if not a.het])
        complex_atoms_count = len(complex_atoms_all)
        ligand_atoms_count = len(ligand_atoms)

        has_waters = bool(waters)
        chain_count = len({a.chain for a in protein_atoms_all if not a.het})

        cofactor_complex = cofactor_from_complex(ligand_atoms)
        has_fad_like = cofactor_complex in {"FAD", "FMN"}

        ring_cent = compute_ring_centroid(ligand_atoms_main, ligand_sdf, methods, errors)

        altloc_fraction = None
        if ring_cent is not None and protein_atoms_all:
            atom_coords = [a.coords for a in protein_atoms_all if not a.het]
            atom_altlocs = [a.altloc for a in protein_atoms_all if not a.het]
            dists = np.linalg.norm(np.array(atom_coords) - ring_cent, axis=1)
            within = dists <= 8.0
            if within.any():
                altloc_non_blank = sum(1 for idx, flag in enumerate(within) if flag and atom_altlocs[idx].strip() != "")
                altloc_fraction = altloc_non_blank / within.sum()

        structure_source = detect_structure_source(protein_lines, complex_lines, has_fad_like)
        features_extra["structure_source"] = structure_source
        features_extra["pdb_has_header"] = pdb_header
        features_extra["pdb_resolution_A"] = pdb_resolution
        features_extra["protein_atoms_count"] = protein_atoms_count
        features_extra["complex_atoms_count"] = complex_atoms_count
        features_extra["ligand_atoms_count_in_complex"] = ligand_atoms_count
        features_extra["has_waters"] = has_waters
        features_extra["has_bfactors"] = bfactors_present
        features_extra["chain_count"] = chain_count
        features_extra["altloc_fraction_pocket"] = altloc_fraction

        cofactor_pred, pred_conf, sdf_counts = predict_cofactor_from_sdf(ligand_sdf, methods)
        cofactor_confidence = 1.0 if cofactor_complex in {"FAD", "FMN"} else pred_conf
        features_extra["cofactor_detected_from_complex"] = cofactor_complex
        features_extra["cofactor_pred_from_sdf"] = cofactor_pred
        features_extra["cofactor_confidence_score"] = cofactor_confidence

        # Tail contacts (if FAD-like)
        fad_tail_adenine = None
        fad_tail_pyro = None
        if cofactor_complex == "FAD" or cofactor_pred == "FAD_like":
            adenine_cent, pyro_cent, tail_method = find_tail_centroids(ligand_atoms_main, ring_cent)
            methods["tail_centroid_method"] = tail_method
            protein_tree = kd_tree([a.coords for a in protein_atoms if a.element != "H"])
            if adenine_cent is not None:
                fad_tail_adenine = count_contacts(protein_tree, adenine_cent, cutoff=4.0)
            if pyro_cent is not None:
                fad_tail_pyro = count_contacts(protein_tree, pyro_cent, cutoff=4.0)
        features_extra["fad_tail_contacts_adenine"] = fad_tail_adenine
        features_extra["fad_tail_contacts_pyrophosphate"] = fad_tail_pyro

        ligand_coords = np.array([a.coords for a in ligand_atoms_main if a.element != "H"])
        protein_coords = [a.coords for a in protein_atoms if a.element != "H"]
        protein_tree = kd_tree(protein_coords)

        clash_count, min_dist = ligand_protein_clashes(protein_tree, ligand_coords)
        features_extra["ligand_protein_clash_count"] = clash_count
        features_extra["ligand_protein_min_distance"] = min_dist

        # Gate 1: pose plausibility
        pose_ok = True
        if structure_source == "docked" and (
            (min_dist is not None and min_dist < 1.2) or (clash_count is not None and clash_count > 10)
        ):
            pose_ok = False
            errors.append("pose_unreliable_for_H_protonation")
        gating_decisions["pose_gate"] = {
            "passed": pose_ok,
            "reason": "" if pose_ok else "pose_unreliable_for_H_protonation",
        }

        features_extra["ligand_burial_fraction"] = ligand_burial_proxy(
            protein_tree, protein_coords, ligand_coords, methods
        )
        features_extra["pocket_depth_proxy"] = pocket_depth(protein_coords, ring_cent)

        vina_data = parse_vina_log(debug_log)
        features_extra.update(vina_data)

        # Redox placeholders
        features_extra["redox_couple"] = "unknown"
        features_extra["n_electrons"] = "unknown"
        features_extra["pH"] = None
        features_extra["temperature_C"] = None
        features_extra["ionic_strength_mM"] = None
        features_extra["buffer"] = None
        features_extra["reference_electrode"] = None
        features_extra["measurement_method"] = None

        # Adenosine pocket cues
        adenine_centroid, pyro_centroid, _ = find_tail_centroids(ligand_atoms_main, ring_cent)
        if cofactor_complex == "FAD" or cofactor_pred == "FAD_like":
            # second pocket volume proxy: grid points free within 6 Å sphere around adenine centroid
            if adenine_centroid is not None:
                grid_range = np.arange(-6, 6.5, 2.0)
                free_points = 0
                if protein_tree is not None:
                    for dx in grid_range:
                        for dy in grid_range:
                            for dz in grid_range:
                                pt = adenine_centroid + np.array([dx, dy, dz])
                                if np.linalg.norm(pt - adenine_centroid) <= 6.0:
                                    if not protein_tree.query_ball_point(pt, r=1.8):
                                        free_points += 1
                features_extra["second_pocket_volume_proxy"] = free_points
            else:
                features_extra["second_pocket_volume_proxy"] = None
            if pyro_centroid is not None:
                lys_arg_his = [a for a in protein_atoms if a.resname.upper() in {"LYS", "ARG", "HIS"}]
                coords_pos = np.array([a.coords for a in lys_arg_his]) if lys_arg_his else np.empty((0, 3))
                if coords_pos.size > 0:
                    tree_pos = cKDTree(coords_pos)
                    hits = tree_pos.query_ball_point(pyro_centroid, r=6.0)
                    features_extra["phosphate_positive_patch"] = len(hits)
                else:
                    features_extra["phosphate_positive_patch"] = 0
            else:
                features_extra["phosphate_positive_patch"] = None
            if adenine_centroid is not None:
                arom = [a for a in protein_atoms if a.resname.upper() in {"PHE", "TYR", "TRP"}]
                coords_arom = np.array([a.coords for a in arom]) if arom else np.empty((0, 3))
                if coords_arom.size > 0:
                    tree_arom = cKDTree(coords_arom)
                    hits = tree_arom.query_ball_point(adenine_centroid, r=6.0)
                    features_extra["adenine_stacking_opportunities"] = len(hits)
                else:
                    features_extra["adenine_stacking_opportunities"] = 0
            else:
                features_extra["adenine_stacking_opportunities"] = None
        else:
            features_extra["second_pocket_volume_proxy"] = None
            features_extra["phosphate_positive_patch"] = None
            features_extra["adenine_stacking_opportunities"] = None

        # H-bond features for flavin atom sites
        site_names = {"N5": None, "O4": None, "O2": None, "N1": None}
        for atom in ligand_atoms_main:
            if atom.name.upper() in site_names:
                site_names[atom.name.upper()] = atom.coords
        # Default method is heavy-atom proxy; may be overwritten if explicit hydrogens used.
        features_extra["hbonds_method"] = "heavy_atom_proxy"
        for site, coord in site_names.items():
            if coord is not None:
                features_extra.update(hbond_features(site, coord, protein_atoms, methods))
            else:
                # leave as None to indicate unavailable
                features_extra[f"hb_{site}_count"] = None
                features_extra[f"hb_{site}_nearest_dist"] = None
                features_extra[f"hb_{site}_nearest_angle"] = None
                features_extra[f"hb_{site}_donor_count"] = None
                features_extra[f"hb_{site}_acceptor_count"] = None
                errors.append(f"missing_site_{site}")

        # Protonation / pH proxy (default)
        features_extra["protonation_method"] = "none"
        features_extra["assumed_pH_used"] = False
        features_extra["his_charge_model"] = "fixed_0.1"
        features_extra.update(pocket_charge_features(protein_atoms, ring_cent, features_extra.get("pH")))

        # Optional: explicit hydrogen angles
        hydrogenated_atoms: List[AtomRecord] = []
        hydrogen_method = None
        if args.enable_hydrogen_hbonds and pose_ok:
            h_pdb, hydrogen_method, h_info = add_hydrogens_with_pdb2pqr_or_reduce(
                protein_pdb, tmp_dir, gating_decisions, errors, tool_versions
            )
            methods.update(h_info)
            if h_pdb is not None and gating_decisions.get("hydrogen_addition", {}).get("passed"):
                hydrogenated_atoms = parse_pdb_atoms(h_pdb)
            else:
                errors.append("H_add_failed_or_unreasonable")
        elif args.enable_hydrogen_hbonds and not pose_ok:
            gating_decisions["hydrogen_addition"] = {
                "passed": False,
                "reason": "pose_unreliable_for_H_protonation",
            }

        if hydrogenated_atoms:
            for site, coord in site_names.items():
                if coord is None:
                    continue
                angle = compute_hbond_angles_with_hydrogens(coord, hydrogenated_atoms)
                if angle is not None:
                    features_extra[f"hb_{site}_nearest_angle"] = angle
            features_extra["hbonds_method"] = "hydrogen_explicit"
        else:
            # keep heavy-atom proxy angles as is (likely None)
            pass

        # Optional: PROPKA protonation-aware charges
        if args.enable_propka_charges and pose_ok and ring_cent is not None:
            propka_ok = False
            pkas: Dict[Tuple[str, str, int, str], float] = {}
            assumed_pH_used = False
            target_pH = features_extra.get("pH")
            if target_pH is None:
                target_pH = args.assumed_ph
                assumed_pH_used = True
            propka_input = protein_pdb
            if hydrogen_method is None and args.enable_hydrogen_hbonds and pose_ok:
                # attempt hydrogens for better PROPKA if not already
                h_pdb, hydrogen_method2, _ = add_hydrogens_with_pdb2pqr_or_reduce(
                    protein_pdb, tmp_dir, gating_decisions, errors, tool_versions
                )
                if h_pdb is not None and gating_decisions.get("hydrogen_addition", {}).get("passed"):
                    propka_input = h_pdb
            if is_tool_available("propka"):
                tool_versions.setdefault("propka", run_subprocess_capture(["propka", "--version"], timeout=5)[1].strip())
                run_input = tmp_dir / "propka_input.pdb"
                shutil.copyfile(propka_input, run_input)
                code, out, err = run_subprocess_capture(["propka", run_input.name], timeout=120, cwd=tmp_dir)
                methods["propka_stdout"] = out[-4000:]
                methods["propka_stderr"] = err[-4000:]
                pka_file = tmp_dir / "propka_input.pka"
                if pka_file.exists():
                    pkas = parse_propka_output(pka_file)
                if code != 0 or not pkas:
                    errors.append("propka_failed")
                    gating_decisions["propka"] = {"passed": False, "reason": "propka_failed"}
                else:
                    gating_decisions["propka"] = {"passed": True, "reason": ""}
            else:
                gating_decisions["propka"] = {"passed": False, "reason": "propka_not_available"}
                errors.append("propka_insufficient_coverage")

            propka_charges: Dict[Tuple[str, str, int, str], float] = {}
            coverage = 0.0
            if pkas:
                titratable = {k: v for k, v in pkas.items() if k[0] in {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR"}}
                residues = residue_centroids(protein_atoms)
                in_range = {k: v for k, v in residues.items() if np.linalg.norm(v - ring_cent) <= 10.0}
                titratable_in_range = [k for k in in_range if k[0] in {"ASP", "GLU", "HIS", "LYS", "ARG", "CYS", "TYR"}]
                if titratable_in_range:
                    covered = [k for k in titratable_in_range if k in titratable]
                    coverage = len(covered) / len(titratable_in_range)
                if coverage >= 0.6:
                    for reskey, pka in titratable.items():
                        propka_charges[reskey] = propka_charge(reskey[0], pka, target_pH)
                    propka_ok = True
                else:
                    errors.append("propka_insufficient_coverage")
                    gating_decisions["propka_coverage"] = {
                        "passed": False,
                        "reason": f"coverage={coverage:.2f}",
                    }
            if propka_ok:
                gating_decisions["propka_coverage"] = {"passed": True, "reason": f"coverage={coverage:.2f}"}
                features_extra["protonation_method"] = "pdb2pqr+propka" if hydrogen_method else "propka_only"
                features_extra["assumed_pH_used"] = assumed_pH_used
                features_extra["his_charge_model"] = "propka_adjusted" if not assumed_pH_used else "HH_assumed_pH7"
                charges = pocket_charge_features(protein_atoms, ring_cent, target_pH, propka_charges)
                # Only update propka-derived fields
                for k in ("net_charge_propka_4A", "net_charge_propka_6A", "net_charge_propka_8A", "net_charge_propka_10A"):
                    features_extra[k] = charges.get(k)
            else:
                # leave propka fields null
                pass
        else:
            if args.enable_propka_charges and not pose_ok:
                errors.append("pose_unreliable_for_H_protonation")

        meta = {
            "methods": methods,
            "version": "v1.0",
            "errors": errors,
            "gating_decisions": gating_decisions,
            "tool_versions": tool_versions,
        }

        # Persist updates
        apply_quality_gates_and_write(feat_path, folder / "summary.md", data, features_extra, meta)

        # Build CSV row
        row: Dict[str, object] = {
            "id": folder.name,
            "status": status,
            "structure_source": structure_source,
            "cofactor_detected_from_complex": cofactor_complex,
            "cofactor_confidence_score": cofactor_confidence,
        }
        for k, v in existing_features.items():
            row[k] = v
        for k, v in features_extra.items():
            row[k] = v
        return data, row
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def discover_feature_folders(root: Path) -> List[Path]:
    return [p for p in root.iterdir() if p.is_dir() and (p / "features.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch add extra features for redox generalization.")
    parser.add_argument("--root", required=True, help="Root directory containing per-protein folders.")
    parser.add_argument("--out_csv", required=True, help="Path to write aggregated CSV.")
    parser.add_argument(
        "--enable_hydrogen_hbonds",
        action="store_true",
        help="Add hydrogens (pdb2pqr/reduce) to compute H-bond angles when quality gates pass.",
    )
    parser.add_argument(
        "--enable_propka_charges",
        action="store_true",
        help="Run PROPKA to refine pocket charge proxies when quality gates pass.",
    )
    parser.add_argument(
        "--assumed_ph",
        type=float,
        default=7.0,
        help="Assumed pH if not present in features; used for PROPKA gating.",
    )
    parser.add_argument(
        "--complex_name",
        type=str,
        default="complex.pdb",
        help="Name of complex PDB file to use (default: complex.pdb).",
    )
    parser.add_argument(
        "--only_if_redock_qc_pass",
        action="store_true",
        help="Only process folders that have redock_qc_pass.txt (from redocking pipeline).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    log_path = root / "batch_add_features.log"
    setup_logging(log_path)

    folders = discover_feature_folders(root)
    LOG.info("Discovered %d folders with features.json", len(folders))
    rows: List[Dict[str, object]] = []
    for idx, folder in enumerate(folders, 1):
        try:
            _, row = process_folder(folder, args)
            rows.append(row)
        except Exception as exc:
            LOG.exception("Failed to process %s: %s", folder, exc)
            rows.append({"id": folder.name, "status": "FAIL", "error": str(exc)})
        if idx % 500 == 0:
            LOG.info("Processed %d/%d folders", idx, len(folders))

    if rows:
        # Collect all columns
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        columns = ["id", "status", "structure_source", "cofactor_detected_from_complex", "cofactor_confidence_score"]
        other_cols = sorted(all_keys - set(columns))
        columns.extend(other_cols)
        df = pd.DataFrame(rows, columns=columns)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        LOG.info("Wrote CSV to %s with %d rows and %d columns", out_csv, len(df), len(df.columns))
    else:
        LOG.warning("No rows produced; nothing written")


if __name__ == "__main__":
    main()
