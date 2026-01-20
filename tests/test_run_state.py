import math
from pathlib import Path

import numpy as np

from engine import vqe_n5_edge as engine
from engine.config import CONFIG
from engine.run_state import (
    DEFAULT_TOLERANCES,
    canonical_mutation_key,
    crossing_message,
    detect_st_crossing,
    feature_hash,
    mutation_display_key,
    mutation_effect_signals,
    parse_mutation_list,
    parse_pdb_structure,
    run_key,
    structure_hash_from_file,
    validate_mutations,
)


DATA_DIR = Path(__file__).parent / "data"


def test_parse_mutation_list_handles_chain_and_insertion():
    muts, errs = parse_mutation_list("Q489K, A:V392A A:Q10AK", default_chain="A")
    assert not errs
    assert any(m.icode.strip() == "A" for m in muts)
    key = canonical_mutation_key(muts)
    assert "A:Q10AK" in key
    assert key.startswith("A:")
    display_key = mutation_display_key(muts)
    # display key should include chain only when explicitly provided
    assert "A:Q10AK" in display_key
    assert "Q489K" in display_key


def test_parse_mutation_list_rejects_noncanonical():
    muts, errs = parse_mutation_list("Z324R", default_chain="A")
    assert errs
    assert not muts


def test_validate_mutations_matches_structure():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    residue_index, chain_counts = parse_pdb_structure(str(pdb_path))
    muts, _ = parse_mutation_list("A1V, A:G10AA", default_chain="A")
    default_chain = "A"
    resolved, errors = validate_mutations(muts, residue_index, default_chain)
    assert not errors
    assert len(resolved) == 2
    assert all(m.chain == "A" for m in resolved)


def test_validate_trp_to_phe_matches_three_letter():
    residue_index = {("A", "1", " "): type("Rec", (), {"resn": "TRP", "chain": "A", "resseq": "1", "icode": " ", "atom_count": 5})()}
    muts, _ = parse_mutation_list("W1F", default_chain="A")
    resolved, errors = validate_mutations(muts, residue_index, "A")
    assert not errors, errors


def test_mutation_adjustment_changes_features():
    base_features = {
        "Around_N5_IsoelectricPoint": 0.2,
        "Around_N5_HBondCap": 0.1,
        "Around_N5_Flexibility": 0.3,
    }
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    muts, _ = parse_mutation_list("A1V", default_chain="A")
    lfp_mut, steric_mut = mutation_effect_signals(muts, engine.AA_PROPERTIES)
    lfp_pdb, steric_pdb = engine.compute_lfp_from_pdb(str(pdb_path))
    adjusted = base_features.copy()
    adjusted["Around_N5_IsoelectricPoint"] += lfp_pdb + lfp_mut
    adjusted["Around_N5_HBondCap"] += 0.5 * (steric_pdb + steric_mut)
    adjusted["Around_N5_Flexibility"] *= (1.0 + 0.02 * len(muts))
    changed = [
        abs(adjusted[k] - base_features[k]) >= DEFAULT_TOLERANCES.tol_feature_abs
        for k in base_features
    ]
    assert any(changed), "At least one feature should change when mutation applied."


def test_mutation_display_omits_inferred_chain():
    muts, errs = parse_mutation_list("D393R", default_chain="A")
    assert not errs
    key = canonical_mutation_key(muts)
    disp = mutation_display_key(muts)
    assert key.startswith("A:")
    # display should not inject chain when user omitted it
    assert disp == "D393R"


def test_crossing_message_consistency():
    crit = "gap<0.01 and spin>0.4"
    msg_true = crossing_message(True, crit)
    msg_false = crossing_message(False, crit)
    assert "No singlet" not in msg_true
    assert "detected" in msg_true
    assert "No singlet" in msg_false or "No" in msg_false
    assert msg_true != msg_false
    assert detect_st_crossing(0.005, 0.5, 0.01, 0.4) is True
    assert detect_st_crossing(0.02, 0.5, 0.01, 0.4) is False


def test_run_key_changes_with_mutations(tmp_path):
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    shash = structure_hash_from_file(str(pdb_path))
    muts_a, _ = parse_mutation_list("A1V", default_chain="A")
    muts_b, _ = parse_mutation_list("A1A", default_chain="A")
    key_a = run_key(shash, "AUTO", muts_a)
    key_b = run_key(shash, "AUTO", muts_b)
    assert key_a != key_b


def test_run_key_changes_with_cofactor(tmp_path):
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    shash = structure_hash_from_file(str(pdb_path))
    muts, _ = parse_mutation_list("A1V", default_chain="A")
    key_fad = run_key(shash, "FAD", muts)
    key_fmn = run_key(shash, "FMN", muts)
    assert key_fad != key_fmn


def test_feature_hash_changes_with_feature_delta():
    base = {"iso": 0.1, "hb": 0.2}
    mutated = {"iso": 0.1, "hb": 0.21}
    assert feature_hash(base) != feature_hash(mutated)
