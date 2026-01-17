#!/usr/bin/env python3
"""
Batch redocking and minimization pipeline for FAD-like ligands with GNINA rescoring support.

CLI:
    python tools/redock_and_minimize.py --root /Users/prithivponnusamy/Downloads/FAD --jobs 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

# Optional RDKit import
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

# Optional OpenMM import
try:
    import openmm
    import openmm.app
    import openmm.unit
    OpenMM_AVAILABLE = True
except Exception:
    OpenMM_AVAILABLE = False

LOG = logging.getLogger("redock")
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


@dataclass
class ModeResult:
    """Results for a single docking mode."""
    mode_index: int
    docking_affinity: Optional[float] = None
    gnina_cnnscore: Optional[float] = None
    gnina_cnnaffinity: Optional[float] = None
    min_distance: Optional[float] = None
    clash_count: Optional[int] = None
    qc_pass: bool = False
    minimized_pdb: Optional[Path] = None


@dataclass
class RedockResult:
    id: str
    status: str
    selected_engine: Optional[str]
    selected_score_vina: Optional[float]
    selected_score_gnina_cnnscore: Optional[float]
    selected_score_gnina_cnnaffinity: Optional[float]
    selected_by: Optional[str]
    min_distance: Optional[float]
    clash_count: Optional[int]
    qc_pass: bool
    reason: str
    runtime_sec: float
    modes: List[ModeResult] = field(default_factory=list)


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


def is_tool_available(name: str) -> bool:
    if name in TOOL_CACHE:
        return TOOL_CACHE[name]
    TOOL_CACHE[name] = shutil.which(name) is not None
    return TOOL_CACHE[name]


def run_cmd(cmd: List[str], timeout: int, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run command with timeout, return (returncode, stdout, stderr)."""
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


def parse_pdb_atoms(path: Path) -> List[AtomRecord]:
    """Parse PDB and return AtomRecords."""
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        structure = parser.get_structure("pdb", str(path))
    except Exception as exc:
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
            )
        )
    return atoms


def split_atoms(atoms: List[AtomRecord]) -> Tuple[List[AtomRecord], List[AtomRecord]]:
    """Split into protein and ligand atoms."""
    protein: List[AtomRecord] = []
    ligand: List[AtomRecord] = []
    for atom in atoms:
        if atom.het and atom.resname not in {"HOH", "WAT"}:
            ligand.append(atom)
        elif not atom.het:
            protein.append(atom)
    return protein, ligand


def largest_ligand_group(ligand_atoms: List[AtomRecord]) -> List[AtomRecord]:
    """Get largest ligand group by residue."""
    if not ligand_atoms:
        return []
    groups: Dict[Tuple[str, str, int, str], List[AtomRecord]] = {}
    for atom in ligand_atoms:
        key = (atom.resname, atom.chain, atom.resseq, atom.icode)
        groups.setdefault(key, []).append(atom)
    return max(groups.values(), key=len)


def compute_ring_centroid(ligand_atoms: List[AtomRecord]) -> Optional[np.ndarray]:
    """Compute ring centroid from FAD-like atoms or dense subset."""
    iso_names = {
        "N1", "N3", "N5", "O2", "O4", "C2", "C4", "C4A", "C5", "C6", "C7", "C8", "C9", "C10",
        "C8M", "C6M", "C7M"
    }
    coords = [a.coords for a in ligand_atoms if a.name.upper() in iso_names]
    if len(coords) >= 4:
        return np.mean(coords, axis=0)
    # Fallback: dense subset
    if len(ligand_atoms) < 3:
        return None
    coords = np.array([a.coords for a in ligand_atoms if a.element != "H"])
    if len(coords) < 3:
        return None
    tree = cKDTree(coords)
    k = min(6, len(coords) - 1)
    dists, _ = tree.query(coords, k=k + 1)
    scores = dists[:, 1:].mean(axis=1)
    n_subset = max(3, int(0.4 * len(coords)))
    subset_idx = np.argsort(scores)[:n_subset]
    return coords[subset_idx].mean(axis=0)


def ligand_protein_clashes(
    protein_tree: Optional[cKDTree], ligand_coords: np.ndarray
) -> Tuple[int, Optional[float]]:
    """Compute clash count and min distance."""
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


# ---------------------------- Receptor preparation ---------------------------- #
def prepare_receptor(protein_pdb: Path, receptor_pdbqt: Path, tmp_dir: Path) -> Tuple[bool, str]:
    """
    Prepare receptor PDBQT using prepare_receptor4.py (preferred) or Meeko (fallback).
    Returns (success, method_name).
    """
    # Try prepare_receptor4.py first (MGLTools - preferred)
    if is_tool_available("prepare_receptor4.py"):
        code, out, err = run_cmd(
            ["prepare_receptor4.py", "-r", str(protein_pdb), "-o", str(receptor_pdbqt)],
            timeout=120
        )
        if code == 0 and receptor_pdbqt.exists():
            return True, "mgltools"
        # If it fails, continue to Meeko fallback
        LOG.warning("prepare_receptor4.py failed, trying Meeko fallback: %s", err[-200:])
    
    # Fallback to Meeko
    if is_tool_available("mk_prepare_receptor.py"):
        # First try without --allow_bad_res
        code, out, err = run_cmd(
            ["mk_prepare_receptor.py", "--read_pdb", str(protein_pdb), "-p", "--write_pdbqt", str(receptor_pdbqt)],
            timeout=120
        )
        if code == 0 and receptor_pdbqt.exists():
            return True, "meeko"
        
        # If that failed, try with --allow_bad_res for unusual residues (e.g., FAD)
        LOG.warning("Meeko failed without --allow_bad_res, retrying with flag: %s", err[-200:])
        code, out, err = run_cmd(
            ["mk_prepare_receptor.py", "--read_pdb", str(protein_pdb), "-p", "--allow_bad_res", "--write_pdbqt", str(receptor_pdbqt)],
            timeout=120
        )
        if code == 0 and receptor_pdbqt.exists():
            return True, "meeko"
        
        # Capture error details
        error_msg = err[-500:] if err else out[-500:] if out else "unknown_error"
        return False, f"meeko_receptor_prep_failed: {error_msg}"
    
    return False, "no_receptor_prep_tool"


# ---------------------------- Ligand preparation ---------------------------- #
def prepare_ligand_sdf_to_pdbqt(ligand_sdf: Path, ligand_pdbqt: Path, tmp_dir: Path) -> Tuple[bool, str]:
    """Prepare ligand PDBQT from SDF."""
    if is_tool_available("obabel"):
        code, out, err = run_cmd(
            ["obabel", str(ligand_sdf), "-O", str(ligand_pdbqt), "-xh"],
            timeout=60
        )
        if code == 0 and ligand_pdbqt.exists():
            return True, "obabel"
        return False, f"obabel failed: {err}"
    
    # Fallback: RDKit -> PDB -> obabel
    if RDKit_AVAILABLE:
        try:
            supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
            mol = supplier[0] if supplier else None
            if mol is None:
                return False, "rdkit_no_mol"
            # Generate 3D coordinates if needed
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            # Write to PDB
            pdb_tmp = tmp_dir / "ligand_tmp.pdb"
            writer = Chem.PDBWriter(str(pdb_tmp))
            writer.write(mol)
            writer.close()
            # Convert to PDBQT
            if is_tool_available("obabel"):
                code, out, err = run_cmd(
                    ["obabel", str(pdb_tmp), "-O", str(ligand_pdbqt), "-xh"],
                    timeout=60
                )
                if code == 0 and ligand_pdbqt.exists():
                    return True, "rdkit+obabel"
        except Exception as exc:
            return False, f"rdkit_error: {exc}"
    
    return False, "no_ligand_prep_tool"


# ---------------------------- Docking box definition ---------------------------- #
def define_docking_box(
    folder: Path, box_size: float, box_size_fallback: float
) -> Tuple[Optional[np.ndarray], float, str]:
    """Define docking box center and size. Returns (center, size, method)."""
    complex_pdb = folder / "complex.pdb"
    
    # Priority 1: Extract from existing complex.pdb
    if complex_pdb.exists():
        atoms = parse_pdb_atoms(complex_pdb)
        protein_atoms, ligand_atoms = split_atoms(atoms)
        ligand_main = largest_ligand_group(ligand_atoms)
        if ligand_main:
            centroid = compute_ring_centroid(ligand_main)
            if centroid is not None:
                return centroid, box_size, "complex_ring_centroid"
    
    # Priority 2: Check features.json for pocket info
    features_json = folder / "features.json"
    if features_json.exists():
        try:
            data = json.loads(features_json.read_text())
            # Could check for pocket center if stored
            pass
        except Exception:
            pass
    
    # Priority 3: Fallback to protein center of mass
    protein_pdb = folder / "protein_input.pdb"
    if protein_pdb.exists():
        atoms = parse_pdb_atoms(protein_pdb)
        protein_atoms = [a for a in atoms if not a.het]
        if protein_atoms:
            coords = np.array([a.coords for a in protein_atoms])
            center = coords.mean(axis=0)
            return center, box_size_fallback, "protein_com_fallback"
    
    return None, box_size_fallback, "unavailable"


# ---------------------------- Docking ---------------------------- #
def run_docking(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_pdbqt: Path,
    box_center: np.ndarray,
    box_size: float,
    exhaustiveness: int,
    num_modes: int,
    tmp_dir: Path,
    dock_engine: str = "auto",
) -> Tuple[bool, List[Tuple[int, float]], str]:
    """
    Run docking and extract all modes with scores.
    Returns (success, list of (mode_index, affinity), tool_used).
    """
    # Determine engine
    if dock_engine == "auto":
        if is_tool_available("smina"):
            engine = "smina"
        elif is_tool_available("vina"):
            engine = "vina"
        elif is_tool_available("gnina"):
            engine = "gnina"
        else:
            return False, [], "no_docking_tool"
    else:
        engine = dock_engine
        if not is_tool_available(engine):
            return False, [], f"{engine}_not_available"
    
    # Run smina
    if engine == "smina":
        cmd = [
            "smina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--out", str(output_pdbqt),
            "--center_x", str(box_center[0]),
            "--center_y", str(box_center[1]),
            "--center_z", str(box_center[2]),
            "--size_x", str(box_size),
            "--size_y", str(box_size),
            "--size_z", str(box_size),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
        ]
        code, out, err = run_cmd(cmd, timeout=300, cwd=tmp_dir)  # 5 min timeout for docking
        if code == 0 and output_pdbqt.exists():
            # Parse all mode scores from output
            scores = []
            # Try to parse from PDBQT file REMARK lines first
            try:
                pdbqt_lines = output_pdbqt.read_text().splitlines()
                for line in pdbqt_lines:
                    if "REMARK" in line and ("Affinity" in line or "score" in line.lower()):
                        # Pattern: "REMARK   1  Affinity: -7.5 kcal/mol"
                        match = re.search(r"Affinity[:\s]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                        if match:
                            scores.append(float(match.group(1)))
            except Exception:
                pass
            
            # Also try parsing from stdout/stderr
            if not scores:
                for line in out.splitlines() + err.splitlines():
                    if "Affinity" in line or "score" in line.lower():
                        try:
                            match = re.search(r"Affinity[:\s]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                            if match:
                                scores.append(float(match.group(1)))
                        except (ValueError, AttributeError):
                            pass
            
            # Return scores with mode indices
            return True, [(i, s) for i, s in enumerate(scores) if s is not None], "smina"
        return False, [], f"smina_failed: {err[-500:]}"
    
    # Run vina
    elif engine == "vina":
        config_file = tmp_dir / "vina_config.txt"
        with open(config_file, "w") as f:
            f.write(f"receptor = {receptor_pdbqt}\n")
            f.write(f"ligand = {ligand_pdbqt}\n")
            f.write(f"out = {output_pdbqt}\n")
            f.write(f"center_x = {box_center[0]}\n")
            f.write(f"center_y = {box_center[1]}\n")
            f.write(f"center_z = {box_center[2]}\n")
            f.write(f"size_x = {box_size}\n")
            f.write(f"size_y = {box_size}\n")
            f.write(f"size_z = {box_size}\n")
            f.write(f"exhaustiveness = {exhaustiveness}\n")
            f.write(f"num_modes = {num_modes}\n")
        
        cmd = ["vina", "--config", str(config_file)]
        code, out, err = run_cmd(cmd, timeout=300, cwd=tmp_dir)  # 5 min timeout for docking
        if code == 0 and output_pdbqt.exists():
            # Parse scores from REMARK lines in output file
            scores = []
            try:
                pdbqt_lines = output_pdbqt.read_text().splitlines()
                for line in pdbqt_lines:
                    if "REMARK VINA RESULT" in line:
                        try:
                            # Pattern: "REMARK VINA RESULT:   1    -7.5      0.000      0.000"
                            parts = line.split()
                            # Find the affinity value (usually after mode number)
                            for i, p in enumerate(parts):
                                try:
                                    val = float(p)
                                    # Check if previous part is a number (mode index)
                                    if i > 0 and parts[i-1].isdigit():
                                        mode_idx = int(parts[i-1]) - 1  # Convert to 0-indexed
                                        scores.append((mode_idx, val))
                                        break
                                except ValueError:
                                    pass
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass
            
            # Also try stdout
            if not scores:
                for line in out.splitlines():
                    if "REMARK VINA RESULT" in line:
                        try:
                            parts = line.split()
                            score = float(parts[-1])
                            mode_idx = len(scores)  # Assume sequential
                            scores.append((mode_idx, score))
                        except (ValueError, IndexError):
                            pass
            
            return True, scores, "vina"
        return False, [], f"vina_failed: {err[-500:]}"
    
    # Run gnina
    elif engine == "gnina":
        cmd = [
            "gnina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--out", str(output_pdbqt),
            "--center_x", str(box_center[0]),
            "--center_y", str(box_center[1]),
            "--center_z", str(box_center[2]),
            "--size_x", str(box_size),
            "--size_y", str(box_size),
            "--size_z", str(box_size),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
        ]
        code, out, err = run_cmd(cmd, timeout=300, cwd=tmp_dir)  # 5 min timeout for docking
        if code == 0 and output_pdbqt.exists():
            # Parse gnina output for affinities and CNN scores
            scores = []
            try:
                pdbqt_lines = output_pdbqt.read_text().splitlines()
                for line in pdbqt_lines:
                    if "REMARK" in line:
                        # Look for affinity
                        if "Affinity" in line:
                            try:
                                match = re.search(r"Affinity[:\s]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                                if match:
                                    score = float(match.group(1))
                                    # Try to find mode index
                                    mode_match = re.search(r"(\d+)\s+[-+]?\d+\.?\d*", line)
                                    if mode_match:
                                        mode_idx = int(mode_match.group(1)) - 1  # 0-indexed
                                    else:
                                        mode_idx = len(scores)
                                    scores.append((mode_idx, score))
                            except (ValueError, AttributeError, IndexError):
                                pass
            except Exception:
                pass
            
            # Also try stdout/stderr
            if not scores:
                for line in out.splitlines() + err.splitlines():
                    if "Affinity" in line:
                        try:
                            match = re.search(r"Affinity[:\s]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                            if match:
                                score = float(match.group(1))
                                mode_idx = len(scores)
                                scores.append((mode_idx, score))
                        except (ValueError, AttributeError):
                            pass
            
            return True, scores, "gnina"
        return False, [], f"gnina_failed: {err[-500:]}"
    
    return False, [], "no_docking_tool"


def extract_modes_from_pdbqt(pdbqt_path: Path, num_modes: int, tmp_dir: Path) -> List[Path]:
    """Extract individual mode PDBQT files from multi-mode output."""
    mode_files = []
    try:
        lines = pdbqt_path.read_text().splitlines()
        current_mode = -1
        current_lines = []
        
        for line in lines:
            if line.startswith("MODEL"):
                # Save previous mode
                if current_mode >= 0 and current_lines:
                    mode_file = tmp_dir / f"mode_{current_mode}.pdbqt"
                    mode_file.write_text("\n".join(current_lines) + "\n")
                    mode_files.append(mode_file)
                # Start new mode
                try:
                    current_mode = int(line.split()[1])
                except (ValueError, IndexError):
                    current_mode += 1
                current_lines = [line]
            elif line.startswith("ENDMDL"):
                current_lines.append(line)
            elif current_mode >= 0:
                current_lines.append(line)
        
        # Save last mode
        if current_mode >= 0 and current_lines:
            mode_file = tmp_dir / f"mode_{current_mode}.pdbqt"
            mode_file.write_text("\n".join(current_lines) + "\n")
            mode_files.append(mode_file)
        
        # If no MODEL/ENDMDL found, treat entire file as mode 0
        if not mode_files and lines:
            mode_file = tmp_dir / "mode_0.pdbqt"
            mode_file.write_text("\n".join(lines) + "\n")
            mode_files.append(mode_file)
        
    except Exception as exc:
        LOG.warning("Failed to extract modes from %s: %s", pdbqt_path, exc)
    
    return mode_files[:num_modes]  # Limit to requested number


# ---------------------------- GNINA rescoring ---------------------------- #
def run_gnina_rescore(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    tmp_dir: Path,
    score_mode: str = "cnnscore",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run GNINA in scoring-only mode.
    Returns (cnnscore, cnnaffinity) or (None, None) if failed.
    """
    if not is_tool_available("gnina"):
        return None, None
    
    try:
        # GNINA scoring requires PDBQT input
        cmd = [
            "gnina",
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(ligand_pdbqt),
            "--score_only",
        ]
        code, out, err = run_cmd(cmd, timeout=60, cwd=tmp_dir)
        
        if code != 0:
            return None, None
        
        # Parse output for CNN scores
        cnnscore = None
        cnnaffinity = None
        
        for line in out.splitlines() + err.splitlines():
            # Look for patterns like:
            # "CNNscore: 0.85"
            # "CNNaffinity: -8.2"
            # "CNNscore = 0.85"
            if "CNNscore" in line or "CNN score" in line:
                try:
                    # Try various patterns
                    match = re.search(r"CNNscore[:\s=]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                    if match:
                        cnnscore = float(match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            if "CNNaffinity" in line or "CNN affinity" in line:
                try:
                    match = re.search(r"CNNaffinity[:\s=]+([-+]?\d+\.?\d*)", line, re.IGNORECASE)
                    if match:
                        cnnaffinity = float(match.group(1))
                except (ValueError, AttributeError):
                    pass
        
        return cnnscore, cnnaffinity
    
    except Exception as exc:
        LOG.warning("GNINA rescoring failed: %s", exc)
        return None, None


# ---------------------------- Post-docking minimization ---------------------------- #
def minimize_complex(
    complex_pdb: Path, output_pdb: Path, tmp_dir: Path
) -> Tuple[bool, str]:
    """Minimize complex using OpenMM. Returns (success, method_used)."""
    if not OpenMM_AVAILABLE:
        return False, "openmm_not_available"
    
    try:
        # Parse complex
        atoms = parse_pdb_atoms(complex_pdb)
        protein_atoms, ligand_atoms = split_atoms(atoms)
        ligand_main = largest_ligand_group(ligand_atoms)
        
        if not protein_atoms or not ligand_main:
            return False, "missing_protein_or_ligand"
        
        # Write a clean PDB for OpenMM
        temp_pdb = tmp_dir / "temp_complex.pdb"
        with open(temp_pdb, "w") as f:
            atom_idx = 1
            for atom in protein_atoms:
                f.write(
                    f"ATOM  {atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                    f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                )
                atom_idx += 1
            for atom in ligand_main:
                f.write(
                    f"HETATM{atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                    f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                )
                atom_idx += 1
        
        # Load with OpenMM
        pdb = openmm.app.PDBFile(str(temp_pdb))
        
        # Try to use a simple force field
        try:
            # Use implicit solvent or vacuum
            forcefield = openmm.app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
            system = forcefield.createSystem(
                pdb.topology,
                constraints=openmm.app.HBonds,
                nonbondedMethod=openmm.app.NoCutoff,
            )
        except Exception:
            # Fallback: simple harmonic restraints on protein only
            system = openmm.System()
            for i in range(len(protein_atoms)):
                system.addParticle(15.999)  # Approximate heavy atom mass
            
            # Add harmonic position restraints to protein atoms (soft, to allow relaxation)
            force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", 1.0)  # Soft restraint
            force.addPerParticleParameter("x0")
            force.addPerParticleParameter("y0")
            force.addPerParticleParameter("z0")
            
            positions = pdb.positions
            for i in range(len(protein_atoms)):
                pos = positions[i]
                force.addParticle(i, [pos.x, pos.y, pos.z])
            system.addForce(force)
        
        # Create integrator and simulation
        integrator = openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin,
            1 / openmm.unit.picosecond,
            0.002 * openmm.unit.picoseconds
        )
        simulation = openmm.app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        # Minimize
        simulation.minimizeEnergy(maxIterations=500)
        
        # Get minimized positions
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        
        # Write output PDB
        with open(output_pdb, "w") as f:
            atom_idx = 1
            # Write minimized protein atoms
            for i, atom in enumerate(protein_atoms):
                pos = positions[i] if i < len(positions) else atom.coords
                if hasattr(pos, 'value_in_unit'):
                    x, y, z = pos.value_in_unit(openmm.unit.angstrom)
                else:
                    x, y, z = atom.coords
                f.write(
                    f"ATOM  {atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                    f"{atom.resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                )
                atom_idx += 1
            # Write ligand (keep original positions - fixed during minimization)
            for atom in ligand_main:
                f.write(
                    f"HETATM{atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                    f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                )
                atom_idx += 1
        
        return True, "openmm_protein_minimized"
    
    except Exception as exc:
        return False, f"openmm_error: {str(exc)[:200]}"


# ---------------------------- PDBQT to PDB conversion ---------------------------- #
def pdbqt_to_pdb(pdbqt_path: Path, pdb_path: Path) -> bool:
    """Convert PDBQT to PDB."""
    if is_tool_available("obabel"):
        code, out, err = run_cmd(
            ["obabel", str(pdbqt_path), "-O", str(pdb_path), "-h"],
            timeout=30
        )
        return code == 0 and pdb_path.exists()
    
    # Manual parsing fallback
    try:
        lines = pdbqt_path.read_text().splitlines()
        pdb_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                # PDBQT format: remove charge and atom type columns
                if len(line) >= 66:
                    pdb_line = line[:66] + "\n"
                    pdb_lines.append(pdb_line)
        pdb_path.write_text("".join(pdb_lines))
        return True
    except Exception:
        return False


# ---------------------------- QC evaluation ---------------------------- #
def evaluate_qc(complex_pdb: Path) -> Tuple[Optional[float], Optional[int], bool]:
    """Evaluate QC metrics. Returns (min_distance, clash_count, qc_pass)."""
    atoms = parse_pdb_atoms(complex_pdb)
    protein_atoms, ligand_atoms = split_atoms(atoms)
    ligand_main = largest_ligand_group(ligand_atoms)
    
    if not protein_atoms or not ligand_main:
        return None, None, False
    
    protein_coords = np.array([a.coords for a in protein_atoms if a.element != "H"])
    ligand_coords = np.array([a.coords for a in ligand_main if a.element != "H"])
    
    if protein_coords.size == 0 or ligand_coords.size == 0:
        return None, None, False
    
    protein_tree = cKDTree(protein_coords)
    clash_count, min_dist = ligand_protein_clashes(protein_tree, ligand_coords)
    
    qc_pass = (min_dist is not None and min_dist >= 1.2) and (clash_count is not None and clash_count <= 10)
    
    return min_dist, clash_count, qc_pass


# ---------------------------- Mode selection ---------------------------- #
def select_best_mode(
    modes: List[ModeResult],
    selection_policy: str,
) -> Tuple[Optional[ModeResult], str]:
    """
    Select best mode based on policy.
    Returns (selected_mode, selection_method).
    """
    if not modes:
        return None, "no_modes"
    
    # Separate QC-pass and QC-fail modes
    qc_pass_modes = [m for m in modes if m.qc_pass]
    qc_fail_modes = [m for m in modes if not m.qc_pass]
    
    # If we have QC-pass modes, choose from them
    if qc_pass_modes:
        candidates = qc_pass_modes
        
        if selection_policy == "vina_only":
            # Pick lowest (most negative) docking affinity
            best = min(candidates, key=lambda m: m.docking_affinity if m.docking_affinity is not None else float('inf'))
            return best, "qc+vina"
        
        elif selection_policy == "gnina_prefer":
            # Prefer gnina score if available
            candidates_with_gnina = [m for m in candidates if m.gnina_cnnscore is not None]
            if candidates_with_gnina:
                # Highest cnnscore is best
                best = max(candidates_with_gnina, key=lambda m: m.gnina_cnnscore if m.gnina_cnnscore is not None else float('-inf'))
                return best, "qc+gnina"
            else:
                # Fallback to docking affinity
                best = min(candidates, key=lambda m: m.docking_affinity if m.docking_affinity is not None else float('inf'))
                return best, "qc+vina"
        
        elif selection_policy == "best_of_both":
            # Check if gnina scores are consistent (>=90% present)
            gnina_present = sum(1 for m in candidates if m.gnina_cnnscore is not None)
            gnina_consistent = (gnina_present / len(candidates)) >= 0.9 if candidates else False
            
            if gnina_consistent:
                # Use gnina scores
                best = max(candidates, key=lambda m: m.gnina_cnnscore if m.gnina_cnnscore is not None else float('-inf'))
                return best, "qc+gnina"
            else:
                # Use docking affinity
                best = min(candidates, key=lambda m: m.docking_affinity if m.docking_affinity is not None else float('inf'))
                return best, "qc+vina"
        
        else:  # auto or unknown
            # Default: try gnina, fallback to vina
            candidates_with_gnina = [m for m in candidates if m.gnina_cnnscore is not None]
            if candidates_with_gnina:
                best = max(candidates_with_gnina, key=lambda m: m.gnina_cnnscore if m.gnina_cnnscore is not None else float('-inf'))
                return best, "qc+gnina"
            else:
                best = min(candidates, key=lambda m: m.docking_affinity if m.docking_affinity is not None else float('inf'))
                return best, "qc+vina"
    
    # No QC-pass modes: pick least bad
    if qc_fail_modes:
        # Maximize min_distance, then minimize clash_count, then use score as tiebreaker
        best = max(
            qc_fail_modes,
            key=lambda m: (
                m.min_distance if m.min_distance is not None else -1.0,
                -(m.clash_count if m.clash_count is not None else 999),
                -(m.docking_affinity if m.docking_affinity is not None else float('inf'))
            )
        )
        return best, "leastbad"
    
    # Fallback: just return first mode
    return modes[0], "fallback"


# ---------------------------- Main processing ---------------------------- #
def process_folder(
    folder: Path,
    args: argparse.Namespace,
) -> RedockResult:
    """Process a single folder. Returns RedockResult."""
    start_time = time.time()
    folder_id = folder.name
    
    # Check inputs
    protein_pdb = folder / "protein_input.pdb"
    ligand_sdf = folder / "ligand_input.sdf"
    
    if not protein_pdb.exists():
        runtime = time.time() - start_time
        LOG.info("Finished %s: status=missing_protein, reason=protein_input.pdb missing, runtime=%.1fs", 
                 folder_id, runtime)
        return RedockResult(
            id=folder_id, status="missing_protein", selected_engine=None,
            selected_score_vina=None, selected_score_gnina_cnnscore=None,
            selected_score_gnina_cnnaffinity=None, selected_by=None,
            min_distance=None, clash_count=None, qc_pass=False,
            reason="protein_input.pdb missing", runtime_sec=runtime
        )
    
    if not ligand_sdf.exists():
        runtime = time.time() - start_time
        LOG.info("Finished %s: status=missing_ligand, reason=ligand_input.sdf missing, runtime=%.1fs", 
                 folder_id, runtime)
        return RedockResult(
            id=folder_id, status="missing_ligand", selected_engine=None,
            selected_score_vina=None, selected_score_gnina_cnnscore=None,
            selected_score_gnina_cnnaffinity=None, selected_by=None,
            min_distance=None, clash_count=None, qc_pass=False,
            reason="ligand_input.sdf missing", runtime_sec=runtime
        )
    
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"redock_{folder_id}_", dir=folder))
    
    try:
        # Step A: Prepare receptor
        receptor_pdbqt = tmp_dir / "receptor.pdbqt"
        success, receptor_prep_method = prepare_receptor(protein_pdb, receptor_pdbqt, tmp_dir)
        if not success:
            runtime = time.time() - start_time
            LOG.info("Finished %s: status=receptor_prep_failed, reason=%s, runtime=%.1fs", 
                     folder_id, receptor_prep_method[:100], runtime)
            return RedockResult(
                id=folder_id, status="receptor_prep_failed", selected_engine=None,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason=receptor_prep_method, runtime_sec=runtime
            )
        
        # Step B: Prepare ligand
        ligand_pdbqt = tmp_dir / "ligand.pdbqt"
        if (folder / "ligand_named.pdbqt").exists():
            shutil.copy(folder / "ligand_named.pdbqt", ligand_pdbqt)
            ligand_method = "reused"
        else:
            success, ligand_method = prepare_ligand_sdf_to_pdbqt(ligand_sdf, ligand_pdbqt, tmp_dir)
            if not success:
                runtime = time.time() - start_time
                LOG.info("Finished %s: status=ligand_prep_failed, reason=%s, runtime=%.1fs", 
                         folder_id, ligand_method[:100], runtime)
                return RedockResult(
                    id=folder_id, status="ligand_prep_failed", selected_engine=None,
                    selected_score_vina=None, selected_score_gnina_cnnscore=None,
                    selected_score_gnina_cnnaffinity=None, selected_by=None,
                    min_distance=None, clash_count=None, qc_pass=False,
                    reason=ligand_method, runtime_sec=runtime
                )
        
        # Step C: Define docking box
        box_center, box_size, box_method = define_docking_box(
            folder, args.box_size, args.box_size_fallback
        )
        if box_center is None:
            runtime = time.time() - start_time
            LOG.info("Finished %s: status=box_definition_failed, reason=%s, runtime=%.1fs", 
                     folder_id, box_method, runtime)
            return RedockResult(
                id=folder_id, status="box_definition_failed", selected_engine=None,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason=box_method, runtime_sec=runtime
            )
        
        # Step D: Run docking (generate multiple modes)
        docked_pdbqt = tmp_dir / "docked.pdbqt"
        success, mode_scores, docking_engine = run_docking(
            receptor_pdbqt, ligand_pdbqt, docked_pdbqt,
            box_center, box_size, args.exhaustiveness, args.num_modes, tmp_dir,
            dock_engine=args.dock_engine
        )
        if not success:
            runtime = time.time() - start_time
            LOG.info("Finished %s: status=docking_failed, engine=%s, reason=%s, runtime=%.1fs", 
                     folder_id, docking_engine, docking_engine[:100], runtime)
            return RedockResult(
                id=folder_id, status="docking_failed", selected_engine=docking_engine,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason=docking_engine, runtime_sec=runtime
            )
        
        # Extract individual modes
        mode_pdbqt_files = extract_modes_from_pdbqt(docked_pdbqt, args.num_modes, tmp_dir)
        if not mode_pdbqt_files:
            return RedockResult(
                id=folder_id, status="mode_extraction_failed", selected_engine=docking_engine,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason="failed_to_extract_modes", runtime_sec=time.time() - start_time
            )
        
        # Process each mode
        protein_atoms = parse_pdb_atoms(protein_pdb)
        modes: List[ModeResult] = []
        
        for mode_idx, mode_pdbqt in enumerate(mode_pdbqt_files):
            mode_result = ModeResult(mode_index=mode_idx)
            
            # Get docking affinity for this mode
            mode_affinity = None
            for idx, aff in mode_scores:
                if idx == mode_idx:
                    mode_affinity = aff
                    break
            mode_result.docking_affinity = mode_affinity
            
            # Convert to PDB
            mode_pdb = tmp_dir / f"mode_{mode_idx}.pdb"
            if not pdbqt_to_pdb(mode_pdbqt, mode_pdb):
                continue
            
            # Merge with protein
            mode_complex_raw = tmp_dir / f"mode_{mode_idx}_raw.pdb"
            docked_atoms = parse_pdb_atoms(mode_pdb)
            with open(mode_complex_raw, "w") as f:
                atom_idx = 1
                for atom in protein_atoms:
                    f.write(
                        f"ATOM  {atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                        f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                    )
                    atom_idx += 1
                for atom in docked_atoms:
                    f.write(
                        f"HETATM{atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                        f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                    )
                    atom_idx += 1
            
            # Quick prefilter: skip very bad poses before minimization
            min_dist_pre, clash_pre, _ = evaluate_qc(mode_complex_raw)
            if min_dist_pre is not None and min_dist_pre < 0.7 and clash_pre is not None and clash_pre > 100:
                # Skip this mode
                continue
            
            # Minimize
            mode_complex_min = tmp_dir / f"mode_{mode_idx}_min.pdb"
            success, _ = minimize_complex(mode_complex_raw, mode_complex_min, tmp_dir)
            if not success:
                shutil.copy(mode_complex_raw, mode_complex_min)
            
            # QC evaluation
            min_dist, clash_count, qc_pass = evaluate_qc(mode_complex_min)
            mode_result.min_distance = min_dist
            mode_result.clash_count = clash_count
            mode_result.qc_pass = qc_pass
            mode_result.minimized_pdb = mode_complex_min
            
            # GNINA rescoring (if enabled and pose is reasonable)
            if args.use_gnina and (qc_pass or (min_dist is not None and min_dist >= 0.7)):
                # Convert minimized complex back to PDBQT for rescoring
                # We need ligand-only PDBQT
                mode_ligand_pdbqt = tmp_dir / f"mode_{mode_idx}_ligand.pdbqt"
                if is_tool_available("obabel"):
                    # Extract ligand from minimized complex
                    mode_ligand_pdb = tmp_dir / f"mode_{mode_idx}_ligand.pdb"
                    ligand_atoms_min = parse_pdb_atoms(mode_complex_min)
                    _, ligand_atoms_min_list = split_atoms(ligand_atoms_min)
                    ligand_main_min = largest_ligand_group(ligand_atoms_min_list)
                    
                    if ligand_main_min:
                        with open(mode_ligand_pdb, "w") as f:
                            atom_idx = 1
                            for atom in ligand_main_min:
                                f.write(
                                    f"HETATM{atom_idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain:>1s} "
                                    f"{atom.resseq:4d}    {atom.coords[0]:8.3f}{atom.coords[1]:8.3f}{atom.coords[2]:8.3f}  1.00  0.00          {atom.element:>2s}\n"
                                )
                                atom_idx += 1
                        
                        # Convert to PDBQT
                        code, _, _ = run_cmd(
                            ["obabel", str(mode_ligand_pdb), "-O", str(mode_ligand_pdbqt), "-xh"],
                            timeout=30
                        )
                        if code == 0 and mode_ligand_pdbqt.exists():
                            cnnscore, cnnaffinity = run_gnina_rescore(
                                receptor_pdbqt, mode_ligand_pdbqt, tmp_dir, args.gnina_score_mode
                            )
                            mode_result.gnina_cnnscore = cnnscore
                            mode_result.gnina_cnnaffinity = cnnaffinity
            
            modes.append(mode_result)
        
        if not modes:
            return RedockResult(
                id=folder_id, status="no_valid_modes", selected_engine=docking_engine,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason="no_valid_modes_after_processing", runtime_sec=time.time() - start_time
            )
        
        # Step E: Select best mode
        selected_mode, selection_method = select_best_mode(modes, args.selection_policy)
        
        if selected_mode is None:
            return RedockResult(
                id=folder_id, status="mode_selection_failed", selected_engine=docking_engine,
                selected_score_vina=None, selected_score_gnina_cnnscore=None,
                selected_score_gnina_cnnaffinity=None, selected_by=None,
                min_distance=None, clash_count=None, qc_pass=False,
                reason="mode_selection_failed", runtime_sec=time.time() - start_time,
                modes=modes
            )
        
        # Copy selected mode to final output
        complex_final = folder / "complex_redocked.pdb"
        if selected_mode.minimized_pdb and selected_mode.minimized_pdb.exists():
            shutil.copy(selected_mode.minimized_pdb, complex_final)
        else:
            return RedockResult(
                id=folder_id, status="output_copy_failed", selected_engine=docking_engine,
                selected_score_vina=selected_mode.docking_affinity,
                selected_score_gnina_cnnscore=selected_mode.gnina_cnnscore,
                selected_score_gnina_cnnaffinity=selected_mode.gnina_cnnaffinity,
                selected_by=selection_method, min_distance=selected_mode.min_distance,
                clash_count=selected_mode.clash_count, qc_pass=selected_mode.qc_pass,
                reason="failed_to_copy_output", runtime_sec=time.time() - start_time,
                modes=modes
            )
        
        # Write metadata
        meta = {
            "receptor_prep_method": receptor_prep_method,
            "selected_engine": docking_engine,
            "selected_mode_index": selected_mode.mode_index,
            "selected_by": selection_method,
            "box_center": box_center.tolist() if box_center is not None else None,
            "box_size": box_size,
            "box_method": box_method,
            "selected_mode": {
                "docking_affinity": selected_mode.docking_affinity,
                "gnina_cnnscore": selected_mode.gnina_cnnscore,
                "gnina_cnnaffinity": selected_mode.gnina_cnnaffinity,
                "min_distance": selected_mode.min_distance,
                "clash_count": selected_mode.clash_count,
                "qc_pass": selected_mode.qc_pass,
            },
            "all_modes": [
                {
                    "mode_index": m.mode_index,
                    "docking_affinity": m.docking_affinity,
                    "gnina_cnnscore": m.gnina_cnnscore,
                    "gnina_cnnaffinity": m.gnina_cnnaffinity,
                    "min_distance": m.min_distance,
                    "clash_count": m.clash_count,
                    "qc_pass": m.qc_pass,
                }
                for m in modes
            ],
        }
        (folder / "redock_meta.json").write_text(json.dumps(meta, indent=2))
        
        # Write QC pass marker if passed
        if selected_mode.qc_pass:
            (folder / "redock_qc_pass.txt").write_text("QC_PASS\n")
        
        status = "success" if selected_mode.qc_pass else "redock_failed_qc"
        reason = "" if selected_mode.qc_pass else f"min_dist={selected_mode.min_distance:.2f}, clashes={selected_mode.clash_count}"
        
        runtime = time.time() - start_time
        qc_status = "PASS" if selected_mode.qc_pass else "FAIL"
        LOG.info("Finished %s: status=%s, qc=%s, engine=%s, runtime=%.1fs", 
                 folder_id, status, qc_status, docking_engine, runtime)
        
        return RedockResult(
            id=folder_id, status=status, selected_engine=docking_engine,
            selected_score_vina=selected_mode.docking_affinity,
            selected_score_gnina_cnnscore=selected_mode.gnina_cnnscore,
            selected_score_gnina_cnnaffinity=selected_mode.gnina_cnnaffinity,
            selected_by=selection_method, min_distance=selected_mode.min_distance,
            clash_count=selected_mode.clash_count, qc_pass=selected_mode.qc_pass,
            reason=reason, runtime_sec=runtime, modes=modes
        )
    
    except Exception as exc:
        LOG.exception("Error processing %s: %s", folder_id, exc)
        return RedockResult(
            id=folder_id, status="error", selected_engine=None,
            selected_score_vina=None, selected_score_gnina_cnnscore=None,
            selected_score_gnina_cnnaffinity=None, selected_by=None,
            min_distance=None, clash_count=None, qc_pass=False,
            reason=str(exc), runtime_sec=time.time() - start_time
        )
    
    finally:
        # Cleanup temp dir
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def discover_folders(root: Path) -> List[Path]:
    """Discover folders with protein_input.pdb and ligand_input.sdf."""
    folders = []
    for path in root.iterdir():
        if path.is_dir():
            if (path / "protein_input.pdb").exists() and (path / "ligand_input.sdf").exists():
                folders.append(path)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch redocking and minimization pipeline with GNINA support.")
    parser.add_argument("--root", required=True, help="Root directory containing folders.")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs.")
    parser.add_argument("--box_size", type=float, default=24.0, help="Docking box size (Å).")
    parser.add_argument("--box_size_fallback", type=float, default=32.0, help="Fallback box size (Å).")
    parser.add_argument("--exhaustiveness", type=int, default=32, help="Docking exhaustiveness.")
    parser.add_argument("--num_modes", type=int, default=20, help="Number of docking modes.")
    parser.add_argument(
        "--use_gnina",
        type=lambda x: (str(x).lower() == 'true') if isinstance(x, str) else bool(x),
        default=None,  # Will be set based on availability
        help="Use GNINA for rescoring (default: True if gnina available, else False)."
    )
    parser.add_argument(
        "--gnina_score_mode",
        type=str,
        default="cnnscore",
        choices=["cnnscore", "cnnaffinity"],
        help="GNINA scoring mode (default: cnnscore)."
    )
    parser.add_argument(
        "--dock_engine",
        type=str,
        default="auto",
        choices=["auto", "smina", "vina", "gnina"],
        help="Docking engine (default: auto)."
    )
    parser.add_argument(
        "--selection_policy",
        type=str,
        default="best_of_both",
        choices=["auto", "vina_only", "gnina_prefer", "best_of_both"],
        help="Mode selection policy (default: best_of_both)."
    )
    args = parser.parse_args()
    
    # Set use_gnina default based on availability
    if args.use_gnina is None:
        args.use_gnina = is_tool_available("gnina")
    
    root = Path(args.root).resolve()
    log_path = root / "redock.log"
    report_path = root / "redock_report.csv"
    
    setup_logging(log_path)
    
    # Detect tools
    LOG.info("Tool availability:")
    LOG.info("  smina: %s", is_tool_available("smina"))
    LOG.info("  vina: %s", is_tool_available("vina"))
    LOG.info("  gnina: %s", is_tool_available("gnina"))
    LOG.info("  obabel: %s", is_tool_available("obabel"))
    LOG.info("  prepare_receptor4.py: %s", is_tool_available("prepare_receptor4.py"))
    LOG.info("  mk_prepare_receptor.py (Meeko): %s", is_tool_available("mk_prepare_receptor.py"))
    LOG.info("  OpenMM: %s", OpenMM_AVAILABLE)
    LOG.info("  RDKit: %s", RDKit_AVAILABLE)
    LOG.info("  use_gnina: %s", args.use_gnina)
    
    folders = discover_folders(root)
    LOG.info("Discovered %d folders to process", len(folders))
    
    # Process folders
    results: List[RedockResult] = []
    
    if args.jobs > 1:
        # For multiprocessing, process in parallel
        # Note: Individual folder completion is logged inside process_folder()
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.starmap(process_folder, [(f, args) for f in folders])
    else:
        # Single-threaded: process sequentially
        # Note: Individual folder completion is logged inside process_folder()
        for folder in folders:
            result = process_folder(folder, args)
            results.append(result)
    
    # Write report
    import pandas as pd
    df = pd.DataFrame([
        {
            "id": r.id,
            "status": r.status,
            "selected_engine": r.selected_engine,
            "selected_score_vina": r.selected_score_vina,
            "selected_score_gnina_cnnscore": r.selected_score_gnina_cnnscore,
            "selected_score_gnina_cnnaffinity": r.selected_score_gnina_cnnaffinity,
            "selected_by": r.selected_by,
            "min_distance": r.min_distance,
            "clash_count": r.clash_count,
            "qc_pass": r.qc_pass,
            "reason": r.reason,
            "runtime_sec": r.runtime_sec,
        }
        for r in results
    ])
    df.to_csv(report_path, index=False)
    
    # Summary
    qc_passed = sum(1 for r in results if r.qc_pass)
    LOG.info("Summary:")
    LOG.info("  Total processed: %d", len(results))
    LOG.info("  QC passed: %d", qc_passed)
    LOG.info("  QC failed: %d", len(results) - qc_passed)
    LOG.info("  Report written to: %s", report_path)


if __name__ == "__main__":
    main()
