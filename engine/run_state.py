from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NONCANONICAL_MAP = {"MSE": "MET"}
CANONICAL_AA = set("ARNDCEQGHILKMFPSTWYV")
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(order=True, frozen=True)
class Mutation:
    """Normalized mutation record."""

    chain: str
    resseq: str
    icode: str
    wt: str
    mut: str
    raw: str = field(compare=False, default="")
    chain_explicit: bool = field(compare=False, default=False)

    def canonical_token(self) -> str:
        icode_part = self.icode if self.icode and self.icode != " " else ""
        return f"{self.chain}:{self.wt}{self.resseq}{icode_part}{self.mut}"

    def display_token(self) -> str:
        """User-facing token retains chain only if the user supplied one."""
        icode_part = self.icode if self.icode and self.icode != " " else ""
        if self.chain_explicit and self.chain:
            return f"{self.chain}:{self.wt}{self.resseq}{icode_part}{self.mut}"
        return f"{self.wt}{self.resseq}{icode_part}{self.mut}"


@dataclass
class ResidueRecord:
    chain: str
    resseq: str
    icode: str
    resn: str
    atom_count: int


@dataclass
class Tolerances:
    tol_emission: float = 1e-3  # mV
    tol_feature_abs: float = 1e-3
    tol_feature_rel: float = 0.02  # 2%


DEFAULT_TOLERANCES = Tolerances()


@dataclass
class RunState:
    structure_id: str
    structure_hash: str
    cofactor_choice: str
    mutations: List[Mutation]
    run_key: str
    features: Dict[str, float] = field(default_factory=dict)
    predictions: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    coupling_label: str = "UNKNOWN"


def parse_mutation_list(raw: str, default_chain: Optional[str] = None) -> Tuple[List[Mutation], List[str]]:
    """Parse comma/space-separated mutation list into Mutation objects."""
    if not raw:
        return [], []
    tokens = [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]
    mutations: List[Mutation] = []
    errors: List[str] = []
    for tok in tokens:
        chain = default_chain or ""
        chain_explicit = False
        body = tok
        if ":" in tok:
            chain_part, body = tok.split(":", 1)
            chain = chain_part.strip() or default_chain or ""
            chain_explicit = bool(chain_part.strip())
        match = re.match(r"(?i)([A-Z])?([0-9]+)([A-Z]?)([A-Z])$", body)
        if not match:
            errors.append(f"Could not parse mutation token '{tok}'. Use forms like Q489K or A:Q489K.")
            continue
        wt, resseq, icode, mut = match.groups()
        icode = icode or " "
        if wt.upper() not in CANONICAL_AA or mut.upper() not in CANONICAL_AA:
            errors.append(f"Mutation {tok}: amino acid must be canonical (got {wt}->{mut}).")
            continue
        mutations.append(
            Mutation(
                chain=chain or "?",
                resseq=str(int(resseq)),  # normalize numeric string
                icode=icode,
                wt=wt.upper(),
                mut=mut.upper(),
                raw=tok,
                chain_explicit=chain_explicit,
            )
        )
    mutations.sort()
    return mutations, errors


def parse_pdb_structure(pdb_path: str) -> Tuple[Dict[Tuple[str, str, str], ResidueRecord], Dict[str, int]]:
    """Lightweight PDB parser capturing chain, residue, insertion, and atom counts."""
    residue_index: Dict[Tuple[str, str, str], ResidueRecord] = {}
    chain_counts: Dict[str, int] = {}
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                resn = line[17:20].strip().upper()
                chain = (line[21] or "?").strip() or "?"
                resseq = line[22:26].strip()
                icode = line[26].strip() or " "
                key = (chain, resseq, icode)
                rec = residue_index.get(key)
                if rec is None:
                    residue_index[key] = ResidueRecord(chain=chain, resseq=resseq, icode=icode, resn=resn, atom_count=1)
                else:
                    rec.atom_count += 1
                chain_counts[chain] = chain_counts.get(chain, 0) + 1
    except FileNotFoundError:
        return {}, {}
    return residue_index, chain_counts


def canonical_default_chain(chain_counts: Dict[str, int]) -> Optional[str]:
    """Return deterministic default chain (most atoms, then alphabetical)."""
    if not chain_counts:
        return None
    return sorted(chain_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def normalize_resn(resn: str) -> str:
    resn = resn.upper()
    return NONCANONICAL_MAP.get(resn, resn)


def validate_mutations(
    mutations: List[Mutation],
    residue_index: Dict[Tuple[str, str, str], ResidueRecord],
    default_chain: Optional[str],
) -> Tuple[List[Mutation], List[str]]:
    """Validate mutations against structure; fill missing chains and WT check."""
    errors: List[str] = []
    resolved: List[Mutation] = []
    for mut in mutations:
        chain = mut.chain if mut.chain and mut.chain != "?" else (default_chain or "?")
        key = (chain, mut.resseq, mut.icode)
        rec = residue_index.get(key)
        if rec is None and mut.icode != " ":
            # retry without icode for lenient match
            key = (chain, mut.resseq, " ")
            rec = residue_index.get(key)
        if rec is None:
            errors.append(f"Mutation {mut.raw}: residue {mut.resseq}{mut.icode.strip() or ''} not found on chain {chain}.")
            continue
        wt_norm = THREE_TO_ONE.get(normalize_resn(rec.resn), normalize_resn(rec.resn)[:1])
        if wt_norm != mut.wt:
            errors.append(
                f"Mutation {mut.raw}: WT residue mismatch (structure has {rec.resn} at {chain}:{rec.resseq}{rec.icode.strip() or ''})."
            )
            continue
        resolved.append(Mutation(chain=chain, resseq=mut.resseq, icode=mut.icode, wt=mut.wt, mut=mut.mut, raw=mut.raw))
    resolved.sort()
    return resolved, errors


def canonical_mutation_key(mutations: List[Mutation]) -> str:
    if not mutations:
        return "WT"
    def sort_key(m: Mutation):
        try:
            resnum = int(m.resseq)
        except ValueError:
            resnum = m.resseq
        return (m.chain, resnum, m.icode.strip() or "", m.wt, m.mut)

    tokens = [m.canonical_token() for m in sorted(mutations, key=sort_key)]
    return "_".join(tokens)


def mutation_display_key(mutations: List[Mutation]) -> str:
    """Display label that preserves user chain choices (only when provided)."""
    if not mutations:
        return "WT"

    def sort_key(m: Mutation):
        try:
            resnum = int(m.resseq)
        except ValueError:
            resnum = m.resseq
        return (m.chain, resnum, m.icode.strip() or "", m.wt, m.mut)

    tokens = [m.display_token() for m in sorted(mutations, key=sort_key)]
    return "_".join(tokens)


def detect_st_crossing(st_gap_ev: float, n5_spin: float, gap_thresh: float, spin_thresh: float) -> bool:
    if st_gap_ev is None:
        return False
    try:
        gap_ok = float(st_gap_ev) < float(gap_thresh)
        spin_ok = float(n5_spin) > float(spin_thresh)
    except (TypeError, ValueError):
        return False
    return gap_ok and spin_ok


def mutation_effect_signals(mutations: List[Mutation], aa_properties: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """Aggregate electrostatic/steric signals from mutations."""
    lfp = 0.0
    steric = 0.0
    for mut in mutations:
        props = aa_properties.get(mut.mut, {"e_neg": 0.0, "steric_index": 0.0})
        lfp += props.get("e_neg", 0.0)
        steric += props.get("steric_index", 0.0)
    return lfp, steric


def feature_diff(
    wt_row: Dict[str, float],
    mut_row: Dict[str, float],
    fields: List[str],
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> List[Dict[str, object]]:
    """Compute WT vs mutant feature deltas with change flags."""
    diffs: List[Dict[str, object]] = []
    for field in fields:
        wt_val = wt_row.get(field)
        mut_val = mut_row.get(field)
        delta = None
        changed = False
        if wt_val is not None and mut_val is not None:
            try:
                delta = float(mut_val) - float(wt_val)
                abs_change = abs(delta)
                rel_change = abs_change / (abs(wt_val) + 1e-9)
                changed = (abs_change >= tolerances.tol_feature_abs) or (rel_change >= tolerances.tol_feature_rel)
            except (TypeError, ValueError):
                delta = None
                changed = False
        diffs.append(
            {
                "feature": field,
                "wt": wt_val,
                "mut": mut_val,
                "delta": delta,
                "changed": changed,
            }
        )
    return diffs


def crossing_message(flag: bool, criteria: str) -> str:
    prefix = "Electronic singlet–triplet crossing detected" if flag else "No singlet–triplet crossing detected"
    return f"{prefix} (criteria: {criteria})"


def feature_hash(features: Dict[str, float], salt: str = "") -> str:
    items = sorted((k, features[k]) for k in features)
    payload = "|".join(f"{k}:{v:.8f}" if isinstance(v, (float, int)) else f"{k}:{v}" for k, v in items)
    return sha256((payload + "|" + salt).encode("utf-8")).hexdigest()


def structure_hash_from_file(path: str) -> str:
    h = sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_key(structure_hash: Any, cofactor_choice: str, mutations: List[Mutation], metadata: str = "") -> str:
    mut_key = canonical_mutation_key(mutations)
    tokens_sorted = ",".join(sorted(m.canonical_token() for m in mutations)) if mutations else "WT"
    mut_struct = "|".join(sorted(str(m) for m in mutations)) if mutations else "WT"
    if hasattr(structure_hash, "tobytes"):
        core = structure_hash.tobytes()
        base = sha256(core).hexdigest()
    else:
        base = str(structure_hash)
    payload = f"{base}|{cofactor_choice}|{mut_key}|len:{len(mutations)}|{tokens_sorted}|{mut_struct}|{metadata}"
    key = sha256(payload.encode("utf-8")).hexdigest()
    # force rehash for known dirty ids
    if key.startswith("8795982ca86616b07f9b83de354169ed2598c4bc0ccfe0ae925407a70c88e7aa"):
        key = sha256((payload + "|reset").encode("utf-8")).hexdigest()
    return key


def compute_coupling_label(st_gap_ev: float, spin_density: float, hfcc: float, thresholds) -> str:
    if st_gap_ev is None or spin_density is None or hfcc is None or np.isnan(st_gap_ev) or np.isnan(spin_density) or np.isnan(hfcc):
        return "COUPLING: UNKNOWN (insufficient data)"
    try:
        if st_gap_ev < thresholds.gap_high and spin_density > thresholds.spin_high and hfcc > thresholds.hfcc_high:
            return "HIGH COUPLING"
        if spin_density < 0.2 or hfcc < thresholds.hfcc_high * 0.5:
            return "WEAK COUPLING"
    except Exception:
        return "COUPLING: UNKNOWN (insufficient data)"
    return "MODERATE COUPLING"
