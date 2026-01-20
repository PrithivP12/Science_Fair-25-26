import pandas as pd
from pathlib import Path
from engine.vqe_n5_edge import atomic_write_df
import os


def test_atomic_write_two_writes(tmp_path):
    path = tmp_path / "test.csv"
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "b": [4]})
    atomic_write_df(df1, path)
    atomic_write_df(df2, path)
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 1


def test_atomic_lock_prevents_corruption(tmp_path):
    path = tmp_path / "test.csv"
    df = pd.DataFrame({"x": [1, 2, 3]})
    atomic_write_df(df, path)
    # Ensure file readable immediately
    loaded = pd.read_csv(path)
    assert loaded.equals(df)
