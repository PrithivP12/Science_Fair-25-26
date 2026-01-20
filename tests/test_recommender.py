from engine.recommender import recommend, recommendation_to_row
from engine.recommender_config import CONFIG
from engine.run_state import (
    parse_mutation_list,
    canonical_mutation_key,
    structure_hash_from_file,
    run_key,
    parse_pdb_structure,
)
from engine import recommender

from pathlib import Path
import pytest


DATA_DIR = Path(__file__).parent / "data"


def test_canonical_run_key_changes_for_combos():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    shash = structure_hash_from_file(str(pdb_path))
    muts_a, _ = parse_mutation_list("A1V", default_chain="A")
    muts_b, _ = parse_mutation_list("A2K", default_chain="A")
    key_a = run_key(shash, "AUTO", muts_a)
    key_b = run_key(shash, "AUTO", muts_b)
    assert key_a != key_b


def test_recommender_returns_singles_and_doubles():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    def fake_predict(pdb_path, cofactor_choice, mutation_tokens, wt_feature_hash, wt_brightness, confidence, wt_run_key=None, wt_st_gap=None):
        token = ",".join(mutation_tokens)
        idx = abs(hash(token)) % 7 + 1 if token else 0
        muts, _ = parse_mutation_list(",".join(mutation_tokens) if mutation_tokens else "", default_chain="A")
        return recommender.Recommendation(
            mutations=muts,
            total_score=5.0 + idx,
            pred_brightness=100.0 + idx,
            delta_brightness=idx,
            synergy=0.0,
            clash_risk=0.0,
            stability_risk=0.0,
            confidence=confidence,
            notes="fake",
            run_key=f"rk{idx}",
            heuristic=False,
            feature_hash=f"fh{token}_{idx}",
            st_gap=0.1 * idx,
        )
    mp = pytest.MonkeyPatch()
    mp.setattr(recommender, "predict_for_mutations", fake_predict)
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=2,
        top_n=3,
        beam_width=3,
    )
    assert len(recs["singles"]) >= 3
    # ensure validator passed (no exceptions) even if doubles may be empty
    if recs["doubles"]:
        for rec in recs["doubles"]:
            positions = {(m.chain, m.resseq) for m in rec.mutations}
            assert len(positions) == len(rec.mutations)
    mp.undo()


def test_recommendation_to_row_conversion():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    mp = pytest.MonkeyPatch()
    mp.setattr(recommender, "predict_for_mutations", lambda *args, **kwargs: recommender.Recommendation(
        mutations=[recommender.Mutation(chain="A", resseq="1", icode=" ", wt="A", mut="V")],
        total_score=10.0,
        pred_brightness=101.0,
        delta_brightness=1.0,
        synergy=0.0,
        clash_risk=0.0,
        stability_risk=0.0,
        confidence=60.0,
        notes="fake",
        run_key="rk1",
        heuristic=False,
        feature_hash="fh1",
    ))
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=2,
        top_n=1,
        beam_width=2,
    )
    mp.undo()
    row = recommendation_to_row(recs["singles"][0])
    assert "Mutations" in row and "TotalScore" in row and "run_key" in row


def test_recommended_residues_exist_in_structure():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    residue_index, _ = parse_pdb_structure(str(pdb_path))
    max_by_chain = {}
    for (chain, resi, icode), rec in residue_index.items():
        try:
            resi_int = int(resi)
        except ValueError:
            continue
        max_by_chain[chain] = max(max_by_chain.get(chain, 0), resi_int)
    # build deterministic candidate list from actual structure keys
    candidates = []
    for (chain, resi, icode), rec in list(residue_index.items())[:3]:
        candidates.append(recommender.Mutation(chain=chain, resseq=resi, icode=icode, wt=rec.resn[:1], mut="V"))
    token_map = {m.canonical_token(): m for m in candidates}
    mp = pytest.MonkeyPatch()
    mp.setattr(recommender, "conservative_candidate_pool", lambda *args, **kwargs: candidates)
    mp.setattr(
        recommender,
        "predict_for_mutations",
        lambda pdb_path, cofactor_choice, mutation_tokens, wt_feature_hash, wt_brightness, confidence, wt_run_key=None, wt_st_gap=None: recommender.Recommendation(
            mutations=[token_map.get(mutation_tokens[0], candidates[0])] if mutation_tokens else [],
            total_score=10.0,
            pred_brightness=101.0 + len(mutation_tokens[0]) if mutation_tokens else 100.0,
            delta_brightness=1.0,
            synergy=0.0,
            clash_risk=0.0,
            stability_risk=0.0,
            confidence=confidence,
            notes="fake",
            run_key=f"rk{mutation_tokens[0] if mutation_tokens else 'wt'}",
            heuristic=False,
            feature_hash=f"fh{mutation_tokens[0] if mutation_tokens else 'wt'}",
            st_gap=0.1,
        ),
    )
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=2,
        top_n=3,
        beam_width=3,
    )
    mp.undo()
    keys = set(residue_index.keys())
    for rec in recs["singles"]:
        for m in rec.mutations:
            assert (m.chain, m.resseq, m.icode) in keys or (m.chain, m.resseq, " ") in keys
            try:
                resi_int = int(m.resseq)
                assert resi_int <= max_by_chain.get(m.chain, resi_int)
            except ValueError:
                pass
            rec_struct = residue_index.get((m.chain, m.resseq, m.icode)) or residue_index.get((m.chain, m.resseq, " "))
            if rec_struct:
                assert rec_struct.resn[:1] == m.wt


def test_run_key_includes_mutations_and_is_order_invariant():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    shash = structure_hash_from_file(str(pdb_path))
    muts_a, _ = parse_mutation_list("A1V,C2K", default_chain="A")
    muts_b, _ = parse_mutation_list("C2K,A1V", default_chain="A")
    muts_c, _ = parse_mutation_list("A1V,C3K", default_chain="A")
    key_ab = run_key(shash, "AUTO", muts_a)
    key_ba = run_key(shash, "AUTO", muts_b)
    key_ac = run_key(shash, "AUTO", muts_c)
    assert key_ab == key_ba
    assert key_ab != key_ac


def test_recommender_blocks_when_no_flavin_detected(monkeypatch, tmp_path):
    pdb_path = tmp_path / "noflavin.pdb"
    pdb_path.write_text("ATOM      1  N   ALA A   1      11.104  13.207   8.560  1.00 20.00           N\n")
    monkeypatch.setattr(recommender, "residue_positions_within_radius", lambda *args, **kwargs: ({}, False))
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=2,
        top_n=3,
        beam_width=3,
    )
    assert all(len(recs[k]) == 0 for k in ("singles", "doubles", "triples"))


def test_recommender_rejects_nonexistent_residues():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    fake_candidates = [
        recommender.Mutation(chain="A", resseq="9999", icode=" ", wt="A", mut="V"),
        recommender.Mutation(chain="A", resseq="1", icode=" ", wt="A", mut="V"),
    ]
    mp = pytest.MonkeyPatch()
    mp.setattr(recommender, "conservative_candidate_pool", lambda *args, **kwargs: fake_candidates)
    mp.setattr(recommender, "predict_for_mutations", lambda *args, **kwargs: recommender.Recommendation(
        mutations=[fake_candidates[-1]],
        total_score=1.0,
        pred_brightness=101.0,
        delta_brightness=1.0,
        synergy=0.0,
        clash_risk=0.0,
        stability_risk=0.0,
        confidence=60.0,
        notes="fake",
        run_key="rk",
        heuristic=False,
        feature_hash="fh",
    ))
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=1,
        top_n=2,
        beam_width=2,
    )
    mp.undo()
    # only the valid residue should remain
    assert all(m.resseq != "9999" for r in recs["singles"] for m in r.mutations)


def test_unit_mixup_detection_brightness_vs_emission():
    import pandas as pd

    row = pd.Series({"gpr_pred": -207.0})
    with pytest.raises(ValueError):
        recommender._extract_brightness(row, 0.0)


def test_recommender_returns_nonconstant_brightness_on_fixture(monkeypatch):
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    residue_index, _ = parse_pdb_structure(str(pdb_path))
    muts = []
    for (chain, resi, icode), rec in list(residue_index.items())[:3]:
        muts.append(recommender.Mutation(chain=chain, resseq=resi, icode=icode, wt=rec.resn[:1], mut="V"))
    token_map = {m.canonical_token(): m for m in muts}

    def fake_predict(pdb_path, cofactor_choice, mutation_tokens, wt_feature_hash, wt_brightness, confidence, wt_run_key=None, wt_st_gap=None):
        if not mutation_tokens:
            return recommender.Recommendation(
                mutations=[],
                total_score=0.0,
                pred_brightness=100.0,
                delta_brightness=0.0,
                synergy=0.0,
                clash_risk=0.0,
                stability_risk=0.0,
                confidence=confidence,
                notes="wt",
                run_key="wt",
                heuristic=False,
                feature_hash="fh_wt",
                st_gap=0.0,
            )
        token = mutation_tokens[0]
        idx = abs(hash(token)) % 11 + 1
        return recommender.Recommendation(
            mutations=[token_map.get(token, muts[0])],
            total_score=10.0 + idx,
            pred_brightness=100.0 + idx,
            delta_brightness=idx,
            synergy=0.0,
            clash_risk=0.0,
            stability_risk=0.0,
            confidence=confidence,
            notes="fake",
            run_key=f"rk{idx}",
            heuristic=False,
            feature_hash=f"fh{token}",
            st_gap=0.1 * idx,
        )

    monkeypatch.setattr(recommender, "conservative_candidate_pool", lambda *args, **kwargs: muts)
    monkeypatch.setattr(recommender, "predict_for_mutations", fake_predict)
    recs = recommend(
        pdb_path=str(pdb_path),
        cofactor_choice="AUTO",
        baseline_brightness=10.0,
        confidence=60.0,
        max_combo_size=2,
        top_n=3,
        beam_width=3,
    )
    singles = recs["singles"]
    assert len(singles) >= 1
    brightness_vals = {r.pred_brightness for r in singles}
    assert len(brightness_vals) > 1
