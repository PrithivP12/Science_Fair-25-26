from __future__ import annotations

from typing import Any, Dict
import logging
import os

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_spec, align_features, prepare_catboost_frame


def run_prediction(model_path: str, input_path: str, output_path: str) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    bundle: Dict[str, Any] = joblib.load(model_path)
    model_type = bundle.get("model_type")
    feature_cols = bundle.get("feature_cols")
    cat_cols = bundle.get("categorical_cols", [])
    num_cols = bundle.get("numeric_cols", [])
    dropped_all_nan = bundle.get("dropped_all_nan", [])

    df = pd.read_csv(input_path, low_memory=False)
    if dropped_all_nan:
        df = df.drop(columns=[c for c in dropped_all_nan if c in df.columns])
    df = align_features(df, feature_cols)

    X = df[feature_cols]

    if model_type == "catboost":
        spec = build_feature_spec(X.assign(Em=0))
        spec.categorical_cols = cat_cols
        spec.numeric_cols = num_cols
        X_cb, _ = prepare_catboost_frame(X, spec, medians=bundle.get("numeric_medians"))
        preds = bundle["model"].predict(X_cb)
    elif model_type == "xgb":
        pre = bundle.get("preprocessor")
        if pre is None:
            raise ValueError("Missing preprocessor for XGBoost model")
        preds = bundle["model"].predict(pre.transform(X))
    elif model_type == "ensemble":
        cat_bundle = bundle.get("catboost_model")
        xgb_bundle = bundle.get("xgb_model")
        if cat_bundle is None or xgb_bundle is None:
            raise ValueError("Missing base models for ensemble")
        spec = build_feature_spec(X.assign(Em=0))
        spec.categorical_cols = cat_bundle.get("categorical_cols", [])
        spec.numeric_cols = cat_bundle.get("numeric_cols", [])
        X_cb, _ = prepare_catboost_frame(X, spec, medians=cat_bundle.get("numeric_medians"))
        cat_pred = cat_bundle["model"].predict(X_cb)
        pre = xgb_bundle.get("preprocessor")
        xgb_pred = xgb_bundle["model"].predict(pre.transform(X))
        preds = 0.5 * (cat_pred + xgb_pred)
    else:
        preds = bundle["model"].predict(X)

    out = df.copy()
    out["Em_pred"] = preds

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    logging.info("Wrote predictions -> %s", output_path)
    return output_path
