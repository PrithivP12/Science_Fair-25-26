from __future__ import annotations

from typing import Tuple
import pandas as pd
import os


TARGET_COL = "Em"


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception:
            if not os.path.exists(path):
                fallback_csv = os.path.splitext(path)[0] + ".csv"
                if os.path.exists(fallback_csv):
                    return pd.read_csv(fallback_csv, low_memory=False)
            # fallback to CSV if parquet engine unavailable or file is CSV with parquet name
            return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def get_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    return df[TARGET_COL]


def filter_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    return df[df[TARGET_COL].notna()].copy()
