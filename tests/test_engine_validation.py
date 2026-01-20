import pytest
import subprocess
import sys
from pathlib import Path
import pandas as pd
import tempfile

BASE = Path(__file__).resolve().parent.parent
ENGINE = BASE / "engine" / "vqe_n5_edge.py"
DATA_DIR = Path(__file__).parent / "data"


def run_engine(pdb_name: str, mutation_list: str, cofactor: str = "FAD", force: bool = False):
    pdb_path = DATA_DIR / pdb_name
    # build minimal dataset with required columns
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_ds:
        pd.DataFrame(
            {
                "Em": [0.1],
                "Around_N5_IsoelectricPoint": [0.1],
                "Around_N5_HBondCap": [0.1],
                "Around_N5_Flexibility": [0.1],
            }
        ).to_csv(tmp_ds.name, index=False)
        data_path = Path(tmp_ds.name)
    env = {
        "MUTATION_LIST": mutation_list,
        "COFACTOR_SELECTED": cofactor,
        "COFACTOR_FORCE": "1" if force else "0",
        "UNIPROT_ID": "TEST",
    }
    result = subprocess.run(
        [sys.executable, str(ENGINE), "--pdb", str(pdb_path), "--data", str(data_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result


def test_invalid_mutation_fails():
    res = run_engine("sample_fmn.pdb", "Z324R", cofactor="FAD", force=True)
    assert res.returncode != 0


def test_missing_cofactor_fails_without_force():
    # sample_fmn contains FMN, request FAD without force -> fail
    res = run_engine("sample_fmn.pdb", "A1V", cofactor="FAD", force=False)
    assert res.returncode != 0


def test_force_allows_missing_cofactor():
    res = run_engine("sample_fmn.pdb", "A1V", cofactor="FAD", force=True)
    # Force may still fail if dataset invalid; ensure error is informative
    if res.returncode != 0:
        combined = (res.stdout + res.stderr)
        assert "Dataset not found" in combined or "Training dataset invalid" in combined
