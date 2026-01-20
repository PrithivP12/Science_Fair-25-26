#!/usr/bin/env python3
"""
Small-data redox potential training script focused on generalization.

Features:
- Drops leakage/meta columns.
- Builds numeric/categorical preprocessors.
- Feature reduction to top-k drivers via SelectFromModel.
- Low-variance model choice: GPR (Matern) or Ridge.
- GroupKFold if a group column is provided; otherwise RepeatedKFold.
- Optional log1p transform for skewed positive targets.
- Exports selected features and top drivers for downstream (e.g., quantum inputs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectFromModel


FORBIDDEN = {
    "uniprot_id",
    "pdb_id",
    "structure_path",
    "cof_chain",
    "cof_resseq",
    "cof_icode",
    "method",
    "source",
    "measurement_note",
    "group_std",
    "group_n",
    "sample_weight",
}

ALLOWED_CATEGORICALS = {
    "cofactor",
    "pdb_method",
    "ring_atoms_present",
    "has_dssp",
    "has_freesasa",
    "in_jcim",
    "N5_nearest_resname",
}


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    feature_cols = [c for c in df.columns if c != "Em" and c not in FORBIDDEN]
    cat_cols = sorted([c for c in feature_cols if c in ALLOWED_CATEGORICALS or df[c].dtype == "object"])
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre, num_cols, cat_cols


def make_selector(n_keep: int = 10) -> SelectFromModel:
    # L1 for sparsity on standardized inputs
    lasso = Lasso(alpha=0.001, max_iter=5000)
    return SelectFromModel(lasso, max_features=n_keep, threshold=-np.inf)


def make_model(model_type: str, n_restarts: int, random_state: int):
    if model_type == "gpr":
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-4, 1e2)
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
        )
    if model_type == "ridge":
        return Ridge(alpha=5.0, random_state=random_state)
    raise ValueError("model_type must be 'gpr' or 'ridge'")


def maybe_log_transform(y: np.ndarray) -> FunctionTransformer:
    if np.all(y > 0) and pd.Series(y).skew() > 1.0:
        return FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
    return FunctionTransformer(lambda x: x, validate=False)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.data, low_memory=False)

    # Condition features
    if "pH" in df.columns:
        df["has_pH"] = df["pH"].notna().astype(int)
        df["pH_centered"] = df["pH"] - 7.0
        df["pH_sq"] = df["pH_centered"] ** 2
    if "temperature_C" in df.columns:
        df["has_temp"] = df["temperature_C"].notna().astype(int)

    df = df[df["Em"].notna()].copy()

    # Drop all-NaN columns to avoid imputer warnings
    all_nan = [c for c in df.columns if c != "Em" and df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)

    pre, num_cols, cat_cols = build_preprocessor(df)
    selector = make_selector(n_keep=args.top_k)
    model = make_model(args.model, args.n_restarts, args.seed)
    target_tx = maybe_log_transform(df["Em"].values)

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("selector", selector),
            ("model", model),
        ]
    )

    X = df[[c for c in df.columns if c != "Em"]]
    y = df["Em"].values
    groups = df[args.group] if args.group and args.group in df.columns else None

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=args.seed, stratify=None
    )

    if g_train is not None:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(X_train, y_train, groups=g_train)
    else:
        splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=args.seed)
        splits = splitter.split(X_train, y_train)

    cv_metrics = []
    for tr_idx, va_idx in splits:
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        cv_metrics.append(eval_metrics(y_va, preds))

    cv_mae = float(np.mean([m["mae"] for m in cv_metrics]))
    cv_rmse = float(np.mean([m["rmse"] for m in cv_metrics]))
    cv_r2 = float(np.mean([m["r2"] for m in cv_metrics]))

    pipe.fit(X_train, y_train)
    test_pred = pipe.predict(X_test)
    test_metrics = eval_metrics(y_test, test_pred)

    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    selected_mask = pipe.named_steps["selector"].get_support()
    selected_features = feature_names[selected_mask]

    importance = None
    coeffs = None
    top_drivers = []
    if args.model == "ridge":
        coeffs = dict(zip(selected_features, pipe.named_steps["model"].coef_))
        top_drivers = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)[: min(8, len(coeffs))]
    else:
        perm = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=10,
            random_state=args.seed,
            scoring="neg_mean_absolute_error",
        )
        importance = sorted(
            [(f, float(s)) for f, s in zip(feature_names, perm.importances_mean)],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[: min(8, len(feature_names))]
        top_drivers = importance

    results = {
        "cv": {"mae": cv_mae, "rmse": cv_rmse, "r2": cv_r2},
        "test": test_metrics,
        "selected_features": selected_features.tolist(),
        "top_drivers": top_drivers,
        "model": args.model,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with Em target")
    parser.add_argument("--group", default=None, help="Optional group/family column for GroupKFold")
    parser.add_argument("--model", choices=["gpr", "ridge"], default="gpr")
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--n-restarts", type=int, default=2, dest="n_restarts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts_qc")
    args = parser.parse_args()
    main(args)
