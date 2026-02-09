import numpy as np
import pandas as pd
import pytest

from engine.vqe_n5_edge import (
    compute_env_features,
    extract_uniprot_from_pdb,
    predict_gpr,
    train_gpr,
)


def _pdb_line(record, serial, atom, resn, chain, resi, x, y, z, b=20.0, element="C"):
    return (
        f"{record:<6}{serial:>5d} {atom:<4} {resn:>3} {chain:1}{resi:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{b:>6.2f}          {element:>2}\n"
    )


def test_compute_env_features_uses_three_letter_residue_mapping(tmp_path):
    pdb_path = tmp_path / "mini.pdb"
    pdb_text = "".join(
        [
            _pdb_line("HETATM", 1, "N1", "FMN", "A", 1, 0.0, 0.0, 0.0, element="N"),
            _pdb_line("ATOM", 2, "NZ", "LYS", "A", 2, 1.0, 0.0, 0.0, element="N"),
            _pdb_line("ATOM", 3, "OD1", "ASP", "A", 3, 0.0, 1.0, 0.0, element="O"),
        ]
    )
    pdb_path.write_text(pdb_text)

    iso, hb, flex, found = compute_env_features(str(pdb_path), radius=6.0)

    assert found is True
    # LYS (K) + ASP (D): (10.5 + 3.1) / 1.5
    assert iso == pytest.approx(9.066666666666666, rel=1e-6)
    # Both residues are H-bond capable -> 2 / 2
    assert hb == pytest.approx(1.0, rel=1e-6)
    assert flex == pytest.approx(1.08, rel=1e-6)


def test_train_gpr_and_predict_gpr_not_constant_for_distinct_inputs():
    rows = []
    for iso in (12.0, 14.0, 16.0, 18.0):
        for hb in (1.0, 3.0):
            for flex in (1.2, 1.4):
                em = -300.0 + 5.0 * (iso - 15.0) + 20.0 * (hb - 2.0) - 50.0 * (flex - 1.3)
                rows.append(
                    {
                        "Around_N5_IsoelectricPoint": iso,
                        "Around_N5_HBondCap": hb,
                        "Around_N5_Flexibility": flex,
                        "Em": em,
                    }
                )
    df = pd.DataFrame(rows)
    gpr = train_gpr(df)
    preds = predict_gpr(
        gpr,
        np.array(
            [
                [18.0, 3.0, 1.4],
                [12.0, 1.0, 1.2],
            ]
        ),
        return_std=False,
    )
    assert abs(float(preds[0]) - float(preds[1])) > 1.0


def test_extract_uniprot_from_dbref(tmp_path):
    pdb_path = tmp_path / "uid.pdb"
    pdb_path.write_text(
        "DBREF  2E82 A    1   347  UNP    P14920   OXDA_HUMAN       1    347\n"
        + _pdb_line("HETATM", 1, "N5", "FAD", "A", 351, 0.0, 0.0, 0.0, element="N")
    )
    assert extract_uniprot_from_pdb(str(pdb_path)) == "P14920"


def test_compute_env_features_multicofactor_uses_local_site(tmp_path):
    pdb_path = tmp_path / "multi.pdb"
    pdb_text = "".join(
        [
            _pdb_line("HETATM", 1, "N5", "FAD", "A", 101, 0.0, 0.0, 0.0, element="N"),
            _pdb_line("HETATM", 2, "C4A", "FAD", "A", 101, 0.2, 0.0, 0.0),
            _pdb_line("HETATM", 3, "N5", "FAD", "B", 201, 100.0, 100.0, 100.0, element="N"),
            _pdb_line("HETATM", 4, "C4A", "FAD", "B", 201, 100.2, 100.0, 100.0),
            _pdb_line("ATOM", 5, "NZ", "LYS", "A", 5, 1.0, 0.0, 0.0, element="N"),
            _pdb_line("ATOM", 6, "OD1", "ASP", "A", 6, 0.0, 1.0, 0.0, element="O"),
        ]
    )
    pdb_path.write_text(pdb_text)
    iso, hb, flex, found = compute_env_features(str(pdb_path), radius=6.0)
    assert found is True
    assert iso > 8.0
    assert hb > 0.0
    assert flex >= 1.05
