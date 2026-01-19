from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


TARGET_COL = "Em"
LEAKAGE_COLS = {
    "uniprot_id",
    "pdb_id",
    "structure_path",
    "cof_chain",
    "cof_resseq",
    "cof_icode",
    "Em_mean",
    "Em_std",
    "n_measurements",
    "sample_weight",
}

CATEGORICAL_COLS = {
    "cofactor",
    "pdb_method",
    "has_dssp",
    "has_freesasa",
    "ring_atoms_present",
    "in_jcim",
    "N5_nearest_resname",
}

IDENTIFIER_HINTS = ("id", "path", "file")


@dataclass
class FeatureSpec:
    feature_cols: List[str]
    categorical_cols: List[str]
    numeric_cols: List[str]
    dropped_all_nan: List[str]


def _warn_leakage(cols: List[str]) -> None:
    suspected = [c for c in cols if any(hint in c.lower() for hint in IDENTIFIER_HINTS)]
    for col in suspected:
        if col not in LEAKAGE_COLS and col != TARGET_COL:
            logging.warning("Possible leakage column kept: %s", col)


def drop_all_nan_columns(df: pd.DataFrame, candidate_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    dropped = [c for c in candidate_cols if df[c].isna().all()]
    if dropped:
        logging.warning("Dropping all-NaN columns: %s", dropped)
    return df.drop(columns=dropped), dropped


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    patterns = ("sasa", "b_factor", "polar_contacts")
    cols = [c for c in df.columns if any(p in c.lower() for p in patterns)]
    for c in cols:
        miss_col = f"{c}_missing"
        if miss_col in df.columns:
            continue
        df[miss_col] = df[c].isna().astype(int)
    return df


def build_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    cols = [c for c in df.columns if c != TARGET_COL and c not in LEAKAGE_COLS]
    _warn_leakage(cols)

    df_clean, dropped = drop_all_nan_columns(df, cols)
    cols = [c for c in cols if c not in dropped]

    inferred_cats = [
        c for c in cols if is_object_dtype(df_clean[c]) or is_bool_dtype(df_clean[c]) or is_categorical_dtype(df_clean[c])
    ]
    categorical_cols = sorted(set([c for c in cols if c in CATEGORICAL_COLS] + inferred_cats))
    numeric_cols = [c for c in cols if c not in categorical_cols]

    if not cols:
        raise ValueError("No feature columns available after dropping leakage/target columns")

    return FeatureSpec(
        feature_cols=cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        dropped_all_nan=dropped,
    )


def align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    aligned = df.copy()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned


def make_linear_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if spec.numeric_cols:
        transformers.append(("num", numeric_pipeline, spec.numeric_cols))
    if spec.categorical_cols:
        transformers.append(("cat", categorical_pipeline, spec.categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_tree_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if spec.numeric_cols:
        transformers.append(("num", numeric_pipeline, spec.numeric_cols))
    if spec.categorical_cols:
        transformers.append(("cat", categorical_pipeline, spec.categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_catboost_frame(
    df: pd.DataFrame,
    spec: FeatureSpec,
    medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    X = df[spec.feature_cols].copy()

    # Numeric coercion and median impute
    for col in spec.numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if medians is None:
        medians = {col: float(np.nanmedian(X[col].values)) for col in spec.numeric_cols}
    for col, med in medians.items():
        X[col] = X[col].fillna(med)

    # Categorical as string with missing token
    for col in spec.categorical_cols:
        X[col] = X[col].astype(str).fillna("missing")
        X[col] = X[col].replace("nan", "missing")

    return X, medians
