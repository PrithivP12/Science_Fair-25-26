from pathlib import Path

import pandas as pd

from engine.run_state import (
    DEFAULT_TOLERANCES,
    canonical_mutation_key,
    feature_diff,
    run_key,
    structure_hash_from_file,
)


DATA_DIR = Path(__file__).parent / "data"


def test_run_key_and_feature_changes():
    pdb_path = DATA_DIR / "sample_fmn.pdb"
    shash = structure_hash_from_file(str(pdb_path))
    run_key_wt = run_key(shash, "AUTO", [])
    run_key_mut = run_key(shash, "AUTO", [])
    assert run_key_wt == run_key_mut  # identical when no mutations
    muts_df = pd.DataFrame(
        [
            {"feature": "Around_N5_IsoelectricPoint", "wt": 0.1, "mut": 0.2},
            {"feature": "Around_N5_HBondCap", "wt": 0.1, "mut": 0.1},
        ]
    )
    diff = feature_diff(
        muts_df.set_index("feature")["wt"].to_dict(),
        muts_df.set_index("feature")["mut"].to_dict(),
        list(muts_df["feature"].values),
        DEFAULT_TOLERANCES,
    )
    changed = [d for d in diff if d["changed"]]
    assert changed, "At least one feature should change for mutant."
