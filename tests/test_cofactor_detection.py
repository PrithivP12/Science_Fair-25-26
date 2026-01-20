from pathlib import Path

from engine.vqe_n5_edge import (
    detect_cofactor_from_pdb,
    cofactor_presence,
    ALLOWED_COFAC,
)


def _write_pdb(path: Path, resname: str):
    content = f"""HETATM    1  N1  {resname} A 501      10.000  10.000  10.000  1.00 20.00           N
HETATM    2  C2  {resname} A 501      11.000  10.000  10.000  1.00 20.00           C
HETATM    3  O2  {resname} A 501      12.000  10.000  10.000  1.00 20.00           O
"""
    path.write_text(content)


def test_detect_cofactor_fad(tmp_path):
    pdb_path = tmp_path / "fad.pdb"
    _write_pdb(pdb_path, "FAD")
    meta = detect_cofactor_from_pdb(str(pdb_path))
    assert meta is not None
    assert meta["type"] == "FAD"
    assert meta["resname"] == "FAD"
    assert meta["chain"] == "A"
    assert meta["resseq"] == "501"


def test_detect_cofactor_fmn(tmp_path):
    pdb_path = tmp_path / "fmn.pdb"
    _write_pdb(pdb_path, "FMN")
    meta = detect_cofactor_from_pdb(str(pdb_path))
    assert meta is not None
    assert meta["type"] == "FMN"


def test_cofactor_presence_matches_selection(tmp_path):
    pdb_path = tmp_path / "mix.pdb"
    _write_pdb(pdb_path, "FMN")
    present, meta = cofactor_presence(str(pdb_path), "FMN")
    assert present is True
    assert meta["type"] == "FMN"
    present2, meta2 = cofactor_presence(str(pdb_path), "FAD")
    assert present2 is False
    assert meta2 is None


def test_cofactor_presence_absent(tmp_path):
    pdb_path = tmp_path / "none.pdb"
    pdb_path.write_text("ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n")
    present, meta = cofactor_presence(str(pdb_path), "FMN")
    assert present is False
    assert meta is None


def test_no_auto_in_allowed():
    assert "AUTO" not in ALLOWED_COFAC
