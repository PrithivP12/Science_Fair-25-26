from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
import pandas as pd

TARGET_COL = "Em"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)
    lower_map = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> Optional[str]:
        for name in names:
            if name in df.columns:
                return name
            low = name.lower()
            if low in lower_map:
                return lower_map[low]
        return None

    col_map = {}
    em_col = pick("Em", "em", "e_m")
    if em_col:
        col_map[em_col] = "Em"
    ph_col = pick("pH", "ph")
    if ph_col:
        col_map[ph_col] = "pH"
    temp_col = pick("temperature_C", "temp_c", "temperature")
    if temp_col:
        col_map[temp_col] = "temperature_C"
    uni_col = pick("uniprot_id", "uniprot")
    if uni_col:
        col_map[uni_col] = "uniprot_id"
    pdb_col = pick("pdb_id", "pdb")
    if pdb_col:
        col_map[pdb_col] = "pdb_id"

    return df.rename(columns=col_map)


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(r"^\s*$", np.nan, regex=True), errors="coerce")


def add_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    if "pH" in df.columns:
        df["pH"] = _coerce_float(df["pH"])
        df["has_pH"] = df["pH"].notna().astype(int)
        df["pH_centered"] = df["pH"] - 7.0
        df["pH_sq"] = df["pH_centered"] ** 2
    else:
        df["has_pH"] = 0

    if "temperature_C" in df.columns:
        df["temperature_C"] = _coerce_float(df["temperature_C"])
        df["has_temp"] = df["temperature_C"].notna().astype(int)
        df["temperature_centered"] = df["temperature_C"] - 25.0
    else:
        df["has_temp"] = 0

    return df


def maybe_drop_temperature(df: pd.DataFrame, min_frac: float = 0.2) -> Tuple[pd.DataFrame, bool]:
    if "temperature_C" not in df.columns:
        return df, False
    frac = df["temperature_C"].notna().mean()
    if frac < min_frac:
        df = df.drop(columns=["temperature_C", "temperature_centered"], errors="ignore")
        return df, True
    return df, False


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "uniprot_id" not in df.columns:
        return df

    df = df.copy()
    if "pH" in df.columns:
        df["pH_rounded"] = df["pH"].round(1)
    else:
        df["pH_rounded"] = np.nan
    if "temperature_C" in df.columns:
        df["temperature_rounded"] = df["temperature_C"].round(1)
    else:
        df["temperature_rounded"] = np.nan
    if "cofactor" not in df.columns:
        df["cofactor"] = "missing"

    key_cols = ["uniprot_id", "pH_rounded", "temperature_rounded", "cofactor"]
    key_df = df[key_cols].copy()
    key_df = key_df.fillna("missing")
    df["_group_key"] = key_df.astype(str).agg("|".join, axis=1)

    if df["_group_key"].duplicated().any():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in {"pH_rounded", "temperature_rounded"}]
        cat_cols = [c for c in df.columns if c not in numeric_cols and c != "_group_key"]

        agg: Dict[str, Any] = {}
        for c in numeric_cols:
            agg[c] = "mean"
        for c in cat_cols:
            agg[c] = "first"

        grouped = df.groupby("_group_key", as_index=False)
        aggregated = grouped.agg(agg)
        stats = grouped["Em"].agg(Em_mean="mean", Em_std="std", n_measurements="count").reset_index(drop=True)
        aggregated = aggregated.reset_index(drop=True)
        aggregated = pd.concat([aggregated, stats], axis=1)
        aggregated["Em_std"] = aggregated["Em_std"].fillna(0.0)
        aggregated = aggregated.drop(columns=["_group_key"], errors="ignore")
        aggregated["Em"] = aggregated["Em_mean"]
        return aggregated

    df = df.drop(columns=["_group_key"], errors="ignore")
    df["Em_mean"] = df["Em"]
    df["n_measurements"] = 1
    df["Em_std"] = 0.0
    return df


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = normalize_columns(df)
    if TARGET_COL not in df.columns:
        raise ValueError("Em column not found after normalization")

    df[TARGET_COL] = _coerce_float(df[TARGET_COL])
    n_rows_before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    n_rows_after_target = len(df)

    df = add_condition_features(df)
    df, temp_dropped = maybe_drop_temperature(df)

    df = aggregate_duplicates(df)
    df["sample_weight"] = 1.0 / (1.0 + df["Em_std"].fillna(0.0))

    meta = {
        "rows_before": n_rows_before,
        "rows_after_target": n_rows_after_target,
        "rows_after_dedup": len(df),
        "temperature_dropped": float(temp_dropped),
        "duplicates_merged": max(n_rows_after_target - len(df), 0),
        "missing_rates": {
            col: float(df[col].isna().mean()) for col in ["Em", "pH", "temperature_C"] if col in df.columns
        },
    }
    return df, meta


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backwards-compatible alias for preprocess_dataframe.
    """
    return preprocess_dataframe(df)
