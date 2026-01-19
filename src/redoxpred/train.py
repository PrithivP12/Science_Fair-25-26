from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import logging
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import Config, load_config
from .data import load_dataset
from .evaluate import regression_metrics, cofactor_metrics, tolerance_metrics, plot_diagnostics
from .features import (
    FeatureSpec,
    add_missing_indicators,
    build_feature_spec,
    make_linear_preprocessor,
    make_tree_preprocessor,
    prepare_catboost_frame,
)
from .split import GROUP_COL, group_kfold_indices, train_val_test_split
from .utils import ensure_dir, save_json, set_seed
from .explain import plot_feature_importance, plot_shap_summary


def _get_catboost():
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor
    except Exception as exc:
        raise RuntimeError(
            "CatBoost is required but not installed. Install with: pip install catboost"
        ) from exc


def _get_lightgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except Exception:
        return None


def _get_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except Exception:
        return None


def _make_dirs(artifacts_dir: str) -> Dict[str, str]:
    paths = {
        "models": os.path.join(artifacts_dir, "models"),
        "reports": os.path.join(artifacts_dir, "reports"),
        "figures": os.path.join(artifacts_dir, "figures"),
        "predictions": os.path.join(artifacts_dir, "predictions"),
    }
    for p in paths.values():
        ensure_dir(p)
    return paths


def _eval_split(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(tolerance_metrics(y_true, y_pred))
    tmp = df.copy()
    tmp["Em_true"] = y_true
    tmp["Em_pred"] = y_pred
    cof = cofactor_metrics(tmp, "Em_true", "Em_pred")
    return {"overall": metrics, "by_cofactor": cof}


def _train_dummy(spec: FeatureSpec, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    preprocessor = make_linear_preprocessor(spec)
    model = DummyRegressor(strategy="mean")
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def _train_ridge(spec: FeatureSpec, X_train: pd.DataFrame, y_train: np.ndarray, alpha: float) -> Pipeline:
    preprocessor = make_linear_preprocessor(spec)
    model = Ridge(alpha=alpha, random_state=0)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def _train_catboost(
    spec: FeatureSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Any, Dict[str, float]]:
    CatBoostRegressor = _get_catboost()
    X_train_cb, medians = prepare_catboost_frame(X_train, spec)
    X_val_cb, _ = prepare_catboost_frame(X_val, spec, medians=medians)

    cat_features = [X_train_cb.columns.get_loc(c) for c in spec.categorical_cols]

    params = dict(params)
    early_stop = params.pop("early_stopping_rounds", 50)
    model = CatBoostRegressor(**params)
    fit_params = {
        "eval_set": (X_val_cb, y_val),
        "use_best_model": True,
        "verbose": False,
        "early_stopping_rounds": early_stop,
        "cat_features": cat_features,
    }
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight
    model.fit(
        X_train_cb,
        y_train,
        **fit_params,
    )
    return model, medians


def _train_xgboost(
    spec: FeatureSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any],
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Any, Any]:
    XGBRegressor = _get_xgboost()
    if XGBRegressor is None:
        raise RuntimeError("XGBoost is not installed")
    preprocessor = make_tree_preprocessor(spec)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    model = XGBRegressor(**params)
    fit_params = {
        "eval_set": [(X_val_t, y_val)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight
    model.fit(X_train_t, y_train, **fit_params)
    return model, preprocessor


def _tune_catboost(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: List[Tuple[List[int], List[int]]],
    base_params: Dict[str, Any],
    trials: int,
    seed: int,
    sample_weight: Optional[pd.Series],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []
    best_params = dict(base_params)
    best_mae = None

    for t in range(trials):
        params = dict(base_params)
        params.update(
            {
                "depth": int(rng.integers(4, 9)),
                "learning_rate": float(rng.uniform(0.05, 0.2)),
                "l2_leaf_reg": float(rng.uniform(1.0, 5.0)),
                "iterations": int(rng.integers(120, 240)),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "early_stopping_rounds": 20,
            }
        )
        maes = []
        for train_idx, val_idx in folds:
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y[val_idx]
            sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
            model, med = _train_catboost(spec, X_tr, y_tr, X_va, y_va, params, sample_weight=sw)
            X_va_cb, _ = prepare_catboost_frame(X_va, spec, medians=med)
            preds = model.predict(X_va_cb)
            maes.append(regression_metrics(y_va, preds)["mae"])
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        results.append({"model": "catboost", "mean_mae": mean_mae, "std_mae": std_mae, "params": params})
        if best_mae is None or mean_mae < best_mae:
            best_mae = mean_mae
            best_params = params
    return best_params, results


def _tune_xgb(
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: List[Tuple[List[int], List[int]]],
    base_params: Dict[str, Any],
    trials: int,
    seed: int,
    sample_weight: Optional[pd.Series],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []
    best_params = dict(base_params)
    best_mae = None

    for t in range(trials):
        params = dict(base_params)
        params.update(
            {
                "max_depth": int(rng.integers(3, 7)),
                "learning_rate": float(rng.uniform(0.05, 0.2)),
                "min_child_weight": float(rng.uniform(1.0, 5.0)),
                "subsample": float(rng.uniform(0.7, 1.0)),
                "colsample_bytree": float(rng.uniform(0.7, 1.0)),
                "reg_lambda": float(rng.uniform(0.5, 4.0)),
                "n_estimators": int(rng.integers(120, 240)),
            }
        )
        maes = []
        for train_idx, val_idx in folds:
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y[val_idx]
            sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
            model, pre = _train_xgboost(spec, X_tr, y_tr, X_va, y_va, params, sample_weight=sw)
            preds = model.predict(pre.transform(X_va))
            maes.append(regression_metrics(y_va, preds)["mae"])
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes))
        results.append({"model": "xgb", "mean_mae": mean_mae, "std_mae": std_mae, "params": params})
        if best_mae is None or mean_mae < best_mae:
            best_mae = mean_mae
            best_params = params
    return best_params, results


def _aggregate_cv_metrics(cv_metrics: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, data in cv_metrics.items():
        folds = data.get("folds", [])
        if not folds:
            continue
        maes = [f["metrics"]["mae"] for f in folds if "metrics" in f]
        rmses = [f["metrics"]["rmse"] for f in folds if "metrics" in f]
        r2s = [f["metrics"]["r2"] for f in folds if "metrics" in f]
        out[name] = {
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "mae_std": float(np.std(maes)) if maes else float("nan"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "rmse_std": float(np.std(rmses)) if rmses else float("nan"),
            "r2": float(np.mean(r2s)) if r2s else float("nan"),
            "r2_std": float(np.std(r2s)) if r2s else float("nan"),
        }
    return out


def _select_best_model(
    metrics_by_model: Dict[str, Dict[str, Any]],
    cv_metrics: Dict[str, Any],
    selection_metric: str,
    selection_split: str,
) -> Tuple[str, str]:
    metric_key = selection_metric.lower()
    if metric_key not in {"mae", "rmse", "r2"}:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    prefer_split = selection_split.lower()
    cv_agg = _aggregate_cv_metrics(cv_metrics)

    def score_for_model(name: str) -> Optional[float]:
        if prefer_split == "cv":
            if name in cv_agg and not np.isnan(cv_agg[name][metric_key]):
                return cv_agg[name][metric_key]
        if "val" in metrics_by_model.get(name, {}):
            return metrics_by_model[name]["val"]["overall"][metric_key]
        if name in cv_agg and not np.isnan(cv_agg[name][metric_key]):
            return cv_agg[name][metric_key]
        if "train" in metrics_by_model.get(name, {}):
            return metrics_by_model[name]["train"]["overall"][metric_key]
        return None

    best_model = None
    best_score = None
    for name in metrics_by_model.keys():
        score = score_for_model(name)
        if score is None or np.isnan(score):
            continue
        if metric_key in {"mae", "rmse"}:
            better = best_score is None or score < best_score
        else:
            better = best_score is None or score > best_score
        if better:
            best_score = score
            best_model = name

    if best_model is None:
        raise ValueError("No model metrics available to select best model")
    return best_model, metric_key


def _render_metrics_table(
    metrics_by_model: Dict[str, Dict[str, Any]],
    cv_agg: Dict[str, Dict[str, float]],
    split_name: str,
    include_std: bool,
) -> List[str]:
    lines = []
    for name in metrics_by_model.keys():
        if split_name == "cv":
            vals = cv_agg.get(name)
            if not vals:
                continue
            if include_std:
                lines.append(
                    f"- {name}: MAE={vals['mae']:.3f}±{vals['mae_std']:.3f}, "
                    f"RMSE={vals['rmse']:.3f}±{vals['rmse_std']:.3f}, "
                    f"R2={vals['r2']:.3f}±{vals['r2_std']:.3f}"
                )
            else:
                lines.append(
                    f"- {name}: MAE={vals['mae']:.3f}, RMSE={vals['rmse']:.3f}, R2={vals['r2']:.3f}"
                )
        else:
            if split_name not in metrics_by_model.get(name, {}):
                continue
            overall = metrics_by_model[name][split_name]["overall"]
            if split_name == "test":
                lines.append(
                    f"- {name}: MAE={overall['mae']:.3f}, RMSE={overall['rmse']:.3f}, "
                    f"R2={overall['r2']:.3f}, within_25={overall['within_25']:.3f}, "
                    f"within_50={overall['within_50']:.3f}"
                )
            else:
                lines.append(
                    f"- {name}: MAE={overall['mae']:.3f}, RMSE={overall['rmse']:.3f}, R2={overall['r2']:.3f}"
                )
    return lines


def _predict_model(
    name: str,
    models: Dict[str, Any],
    spec: FeatureSpec,
    X_df: pd.DataFrame,
    cat_medians: Optional[Dict[str, float]],
) -> np.ndarray:
    if name == "catboost":
        X_cb, _ = prepare_catboost_frame(X_df, spec, medians=cat_medians)
        return models["catboost"].predict(X_cb)
    if name == "xgb":
        pre = models["xgb"]["preprocess"]
        model = models["xgb"]["model"]
        return model.predict(pre.transform(X_df))
    if name == "ensemble":
        cat_pred = _predict_model("catboost", models, spec, X_df, cat_medians)
        xgb_pred = _predict_model("xgb", models, spec, X_df, cat_medians)
        return 0.5 * (cat_pred + xgb_pred)
    return models[name].predict(X_df)


def _compute_oof_preds(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: List[Tuple[List[int], List[int]]],
    spec: FeatureSpec,
    cat_params: Dict[str, Any],
    xgb_params: Dict[str, Any],
    sample_weight: Optional[pd.Series],
) -> Optional[np.ndarray]:
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    for train_idx, val_idx in folds:
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y[val_idx]
        sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
        if name == "catboost":
            model, med = _train_catboost(spec, X_tr, y_tr, X_va, y_va, cat_params, sample_weight=sw)
            X_va_cb, _ = prepare_catboost_frame(X_va, spec, medians=med)
            preds = model.predict(X_va_cb)
        elif name == "xgb":
            model, pre = _train_xgboost(spec, X_tr, y_tr, X_va, y_va, xgb_params, sample_weight=sw)
            preds = model.predict(pre.transform(X_va))
        elif name == "ensemble":
            cat_model, med = _train_catboost(spec, X_tr, y_tr, X_va, y_va, cat_params, sample_weight=sw)
            cat_preds = cat_model.predict(prepare_catboost_frame(X_va, spec, medians=med)[0])
            xgb_model, pre = _train_xgboost(spec, X_tr, y_tr, X_va, y_va, xgb_params, sample_weight=sw)
            xgb_preds = xgb_model.predict(pre.transform(X_va))
            preds = 0.5 * (cat_preds + xgb_preds)
        else:
            return None
        oof[val_idx] = preds
    return oof


def run_training(config_path: str, tune_override: Optional[bool] = None, models_override: Optional[List[str]] = None) -> Dict[str, Any]:
    config = load_config(config_path)
    if tune_override is not None:
        config = Config(
            data_path=config.data_path,
            artifacts_dir=config.artifacts_dir,
            random_seed=config.random_seed,
            test_size=config.test_size,
            val_size=config.val_size,
            n_splits=config.n_splits,
            selection_metric=config.selection_metric,
            selection_split=config.selection_split,
            tune=tune_override,
            tune_trials=config.tune_trials,
            models=config.models,
            ensemble=config.ensemble,
            catboost_params=config.catboost_params,
            ridge_params=config.ridge_params,
            xgb_params=config.xgb_params,
        )
    if models_override is not None:
        config = Config(
            data_path=config.data_path,
            artifacts_dir=config.artifacts_dir,
            random_seed=config.random_seed,
            test_size=config.test_size,
            val_size=config.val_size,
            n_splits=config.n_splits,
            selection_metric=config.selection_metric,
            selection_split=config.selection_split,
            tune=config.tune,
            tune_trials=config.tune_trials,
            models=models_override,
            ensemble=config.ensemble,
            catboost_params=config.catboost_params,
            ridge_params=config.ridge_params,
            xgb_params=config.xgb_params,
        )
    set_seed(config.random_seed)
    logging.info("Loading data: %s", config.data_path)

    df = load_dataset(config.data_path)
    df = df.copy()
    if "Em" not in df.columns:
        raise ValueError("Target column 'Em' missing from dataset")
    if df["Em"].isna().any():
        before = len(df)
        df = df[df["Em"].notna()].copy()
        logging.warning("Dropped %d rows with missing Em", before - len(df))
    clean_meta: Dict[str, Any] = {}
    df = add_missing_indicators(df)

    if GROUP_COL not in df.columns:
        raise ValueError(f"Required group column '{GROUP_COL}' missing from dataset")

    spec = build_feature_spec(df)
    if spec.dropped_all_nan:
        df = df.drop(columns=spec.dropped_all_nan)
    if not spec.feature_cols:
        raise ValueError("No usable features found")

    splits = train_val_test_split(
        df,
        group_col=GROUP_COL,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.random_seed,
    )
    cv_folds = group_kfold_indices(df, GROUP_COL, config.n_splits)

    paths = _make_dirs(config.artifacts_dir)
    save_json(os.path.join(paths["reports"], "splits.json"), splits)

    X = df[spec.feature_cols]
    y = df["Em"].values
    sample_weight = df.get("sample_weight")

    def subset(idx: List[int]) -> pd.DataFrame:
        return X.iloc[idx].copy()

    X_train = subset(splits["train"])
    X_val = subset(splits["val"])
    X_test = subset(splits["test"])

    y_train = y[splits["train"]]
    y_val = y[splits["val"]]
    y_test = y[splits["test"]]

    model_enabled = {m.lower() for m in config.models}

    metrics_by_model: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Any] = {}
    bundles: Dict[str, Dict[str, Any]] = {}
    models_succeeded: List[str] = []
    models_failed: Dict[str, str] = {}
    cat_medians = None
    X_train_cb = None
    X_val_cb = None
    X_test_cb = None

    # Dummy baseline
    if "dummy" in model_enabled:
        try:
            dummy = _train_dummy(spec, X_train, y_train)
            models["dummy"] = dummy
            metrics_by_model["dummy"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, dummy.predict(X_train)),
                "val": _eval_split(df.iloc[splits["val"]], y_val, dummy.predict(X_val)),
                "test": _eval_split(df.iloc[splits["test"]], y_test, dummy.predict(X_test)),
            }
            bundles["dummy"] = {
                "model_type": "dummy",
                "model": dummy,
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "dropped_all_nan": spec.dropped_all_nan,
            }
            models_succeeded.append("dummy")
        except Exception as exc:
            logging.error("Model dummy failed: %s", exc)
            models_failed["dummy"] = str(exc)

    # Ridge
    best_alpha = None
    if "ridge" in model_enabled:
        try:
            alpha_grid = config.ridge_params.get("alpha_grid", [0.1, 1.0, 10.0, 100.0])
            best_mae = None
            best_ridge = None
            for alpha in alpha_grid:
                ridge = _train_ridge(spec, X_train, y_train, alpha)
                preds = ridge.predict(X_val)
                mae = regression_metrics(y_val, preds)["mae"]
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_ridge = ridge
            ridge = best_ridge
            if ridge is None:
                raise ValueError("Ridge training did not produce a model")
            models["ridge"] = ridge
            metrics_by_model["ridge"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, ridge.predict(X_train)),
                "val": _eval_split(df.iloc[splits["val"]], y_val, ridge.predict(X_val)),
                "test": _eval_split(df.iloc[splits["test"]], y_test, ridge.predict(X_test)),
                "best_alpha": best_alpha,
            }
            bundles["ridge"] = {
                "model_type": "ridge",
                "model": ridge,
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "dropped_all_nan": spec.dropped_all_nan,
            }
            models_succeeded.append("ridge")
        except Exception as exc:
            logging.error("Model ridge failed: %s", exc)
            models_failed["ridge"] = str(exc)

    # CatBoost
    cat_params = {
        "depth": config.catboost_params.get("depth", 8),
        "learning_rate": config.catboost_params.get("learning_rate", 0.05),
        "iterations": config.catboost_params.get("iterations", 500),
        "l2_leaf_reg": config.catboost_params.get("l2_leaf_reg", 3.0),
        "loss_function": config.catboost_params.get("loss_function", "MAE"),
        "eval_metric": config.catboost_params.get("eval_metric", "MAE"),
        "random_seed": config.random_seed,
        "early_stopping_rounds": config.catboost_params.get("early_stopping_rounds", 50),
        "subsample": config.catboost_params.get("subsample", 0.8),
        "bootstrap_type": config.catboost_params.get("bootstrap_type", "Bernoulli"),
        "task_type": config.catboost_params.get("task_type", "GPU"),
        "devices": config.catboost_params.get("devices", "0"),
    }
    xgb_params = {
        "max_depth": config.xgb_params.get("max_depth", 6),
        "learning_rate": config.xgb_params.get("learning_rate", 0.05),
        "n_estimators": config.xgb_params.get("n_estimators", 400),
        "min_child_weight": config.xgb_params.get("min_child_weight", 1.0),
        "subsample": config.xgb_params.get("subsample", 0.8),
        "colsample_bytree": config.xgb_params.get("colsample_bytree", 0.8),
        "reg_lambda": config.xgb_params.get("reg_lambda", 1.0),
        "objective": config.xgb_params.get("objective", "reg:absoluteerror"),
        "tree_method": config.xgb_params.get("tree_method", "gpu_hist"),
        "predictor": config.xgb_params.get("predictor", "gpu_predictor"),
        "eval_metric": config.xgb_params.get("eval_metric", "rmse"),
        "random_state": config.random_seed,
    }
    tuning_results: List[Dict[str, Any]] = []
    if config.tune:
        if "catboost" in model_enabled:
            tuned, results = _tune_catboost(
                spec,
                X,
                y,
                cv_folds,
                cat_params,
                config.tune_trials,
                config.random_seed,
                sample_weight,
            )
            cat_params.update(tuned)
            tuning_results.extend(results)
        if "xgb" in model_enabled and _get_xgboost() is not None:
            tuned, results = _tune_xgb(
                spec,
                X,
                y,
                cv_folds,
                xgb_params,
                config.tune_trials,
                config.random_seed,
                sample_weight,
            )
            xgb_params.update(tuned)
            tuning_results.extend(results)
    if "catboost" in model_enabled:
        try:
            sw_train = sample_weight.iloc[splits["train"]].values if sample_weight is not None else None
            cat_model, medians = _train_catboost(
                spec,
                X_train,
                y_train,
                X_val,
                y_val,
                cat_params,
                sample_weight=sw_train,
            )
            models["catboost"] = cat_model
            cat_medians = medians
            X_train_cb, _ = prepare_catboost_frame(X_train, spec, medians=medians)
            X_val_cb, _ = prepare_catboost_frame(X_val, spec, medians=medians)
            X_test_cb, _ = prepare_catboost_frame(X_test, spec, medians=medians)
            metrics_by_model["catboost"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, cat_model.predict(X_train_cb)),
                "val": _eval_split(df.iloc[splits["val"]], y_val, cat_model.predict(X_val_cb)),
                "test": _eval_split(df.iloc[splits["test"]], y_test, cat_model.predict(X_test_cb)),
            }
            bundles["catboost"] = {
                "model_type": "catboost",
                "model": cat_model,
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "numeric_medians": medians,
                "dropped_all_nan": spec.dropped_all_nan,
            }
            models_succeeded.append("catboost")
        except Exception as exc:
            logging.error("Model catboost failed: %s", exc)
            models_failed["catboost"] = str(exc)

    # LightGBM optional
    LGBMRegressor = _get_lightgbm()
    if LGBMRegressor is not None:
        try:
            lgbm_model = LGBMRegressor(random_state=config.random_seed)
            lgbm_pipe = Pipeline(steps=[("preprocess", make_linear_preprocessor(spec)), ("model", lgbm_model)])
            lgbm_pipe.fit(X_train, y_train)
            models["lightgbm"] = lgbm_pipe
            metrics_by_model["lightgbm"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, lgbm_pipe.predict(X_train)),
                "val": _eval_split(df.iloc[splits["val"]], y_val, lgbm_pipe.predict(X_val)),
                "test": _eval_split(df.iloc[splits["test"]], y_test, lgbm_pipe.predict(X_test)),
            }
            bundles["lightgbm"] = {
                "model_type": "lightgbm",
                "model": lgbm_pipe,
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "dropped_all_nan": spec.dropped_all_nan,
            }
            models_succeeded.append("lightgbm")
        except Exception as exc:
            logging.error("Model lightgbm failed: %s", exc)
            models_failed["lightgbm"] = str(exc)

    # XGBoost
    if "xgb" in model_enabled:
        try:
            sw_train = sample_weight.iloc[splits["train"]].values if sample_weight is not None else None
            xgb_model, xgb_pre = _train_xgboost(spec, X_train, y_train, X_val, y_val, xgb_params, sample_weight=sw_train)
            models["xgb"] = {"model": xgb_model, "preprocess": xgb_pre}
            X_train_t = xgb_pre.transform(X_train)
            X_val_t = xgb_pre.transform(X_val)
            X_test_t = xgb_pre.transform(X_test)
            metrics_by_model["xgb"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, xgb_model.predict(X_train_t)),
                "val": _eval_split(df.iloc[splits["val"]], y_val, xgb_model.predict(X_val_t)),
                "test": _eval_split(df.iloc[splits["test"]], y_test, xgb_model.predict(X_test_t)),
            }
            bundles["xgb"] = {
                "model_type": "xgb",
                "model": xgb_model,
                "preprocessor": xgb_pre,
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "dropped_all_nan": spec.dropped_all_nan,
            }
            models_succeeded.append("xgb")
        except Exception as exc:
            logging.error("Model xgb failed: %s", exc)
            models_failed["xgb"] = str(exc)

    # Ensemble (CatBoost + XGBoost)
    if config.ensemble and "ensemble" in model_enabled and "catboost" in models_succeeded and "xgb" in models_succeeded:
        try:
            xgb_pre = models["xgb"]["preprocess"]
            xgb_model = models["xgb"]["model"]
            xgb_train = xgb_model.predict(xgb_pre.transform(X_train))
            xgb_val = xgb_model.predict(xgb_pre.transform(X_val))
            xgb_test = xgb_model.predict(xgb_pre.transform(X_test))
            cat_train = models["catboost"].predict(X_train_cb)
            cat_val = models["catboost"].predict(X_val_cb)
            cat_test = models["catboost"].predict(X_test_cb)

            ens_train = 0.5 * (xgb_train + cat_train)
            ens_val = 0.5 * (xgb_val + cat_val)
            ens_test = 0.5 * (xgb_test + cat_test)

            metrics_by_model["ensemble"] = {
                "train": _eval_split(df.iloc[splits["train"]], y_train, ens_train),
                "val": _eval_split(df.iloc[splits["val"]], y_val, ens_val),
                "test": _eval_split(df.iloc[splits["test"]], y_test, ens_test),
            }
            bundles["ensemble"] = {
                "model_type": "ensemble",
                "feature_cols": spec.feature_cols,
                "categorical_cols": spec.categorical_cols,
                "numeric_cols": spec.numeric_cols,
                "dropped_all_nan": spec.dropped_all_nan,
                "catboost_model": bundles["catboost"],
                "xgb_model": bundles["xgb"],
            }
            models_succeeded.append("ensemble")
        except Exception as exc:
            logging.error("Model ensemble failed: %s", exc)
            models_failed["ensemble"] = str(exc)

    # CV metrics
    cv_folds = group_kfold_indices(df, GROUP_COL, config.n_splits)
    cv_metrics: Dict[str, Any] = {}
    for name in models_succeeded:
        cv_metrics[name] = {"folds": []}
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds, start=1):
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y[val_idx]

        # Dummy
        if "dummy" in models_succeeded:
            dummy_cv = _train_dummy(spec, X_tr, y_tr)
            cv_metrics["dummy"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, dummy_cv.predict(X_va))}
            )

        # Ridge
        if "ridge" in models_succeeded:
            ridge_cv = _train_ridge(spec, X_tr, y_tr, best_alpha or 1.0)
            cv_metrics["ridge"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, ridge_cv.predict(X_va))}
            )

        # CatBoost
        if "catboost" in models_succeeded:
            sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
            cat_cv, med_cv = _train_catboost(spec, X_tr, y_tr, X_va, y_va, cat_params, sample_weight=sw)
            X_va_cb, _ = prepare_catboost_frame(X_va, spec, medians=med_cv)
            cv_metrics["catboost"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, cat_cv.predict(X_va_cb))}
            )

        if "xgb" in models_succeeded:
            sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
            xgb_cv, xgb_pre = _train_xgboost(spec, X_tr, y_tr, X_va, y_va, xgb_params, sample_weight=sw)
            cv_metrics["xgb"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, xgb_cv.predict(xgb_pre.transform(X_va)))}
            )

        if "ensemble" in models_succeeded:
            sw = sample_weight.iloc[train_idx].values if sample_weight is not None else None
            cat_cv, med_cv = _train_catboost(spec, X_tr, y_tr, X_va, y_va, cat_params, sample_weight=sw)
            xgb_cv, xgb_pre = _train_xgboost(spec, X_tr, y_tr, X_va, y_va, xgb_params, sample_weight=sw)
            cat_pred = cat_cv.predict(prepare_catboost_frame(X_va, spec, medians=med_cv)[0])
            xgb_pred = xgb_cv.predict(xgb_pre.transform(X_va))
            ens_pred = 0.5 * (cat_pred + xgb_pred)
            cv_metrics["ensemble"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, ens_pred)}
            )

        if "lightgbm" in models_succeeded:
            lgbm_cv = Pipeline(steps=[("preprocess", make_linear_preprocessor(spec)), ("model", LGBMRegressor(random_state=config.random_seed))])
            lgbm_cv.fit(X_tr, y_tr)
            cv_metrics["lightgbm"]["folds"].append(
                {"fold": fold_idx, "metrics": regression_metrics(y_va, lgbm_cv.predict(X_va))}
            )

    # Save models
    for name, bundle in bundles.items():
        model_path = os.path.join(paths["models"], f"{name}.pkl")
        joblib.dump(bundle, model_path)

    best_model_name, metric_key = _select_best_model(
        metrics_by_model,
        cv_metrics,
        config.selection_metric,
        config.selection_split,
    )
    if config.selection_split.lower() == "val" and metric_key == "mae":
        val_maes = {
            name: metrics_by_model[name]["val"]["overall"]["mae"]
            for name in models_succeeded
            if "val" in metrics_by_model.get(name, {})
        }
        if val_maes:
            min_name = min(val_maes, key=val_maes.get)
            if min_name != best_model_name:
                raise RuntimeError(
                    f"Selection mismatch: expected {min_name} to minimize val MAE, got {best_model_name}"
                )
    best_bundle = bundles[best_model_name]
    best_model_path = os.path.join(paths["models"], "best_model.pkl")
    joblib.dump(best_bundle, best_model_path)

    cv_agg = _aggregate_cv_metrics(cv_metrics)
    metrics_payload = {
        "models": metrics_by_model,
        "cv": cv_metrics,
        "cv_aggregate": cv_agg,
        "selected_model_name": best_model_name,
        "selection_metric": metric_key,
        "selection_split": config.selection_split.lower(),
        "models_succeeded": models_succeeded,
        "models_failed": models_failed,
        "cleaning_meta": clean_meta,
    }
    save_json(os.path.join(paths["reports"], "metrics.json"), metrics_payload)
    if tuning_results:
        tr_df = pd.DataFrame(tuning_results)
        tr_df.to_csv(os.path.join(paths["reports"], "tuning_results.csv"), index=False)

    # Diagnostics + residuals
    best_test_pred = _predict_model(best_model_name, models, spec, X_test, cat_medians)
    plot_diagnostics(
        y_test,
        best_test_pred,
        df.iloc[splits["test"]]["pH"] if "pH" in df.columns else None,
        paths["figures"],
    )
    abs_err = np.abs(best_test_pred - y_test)
    plt.figure(figsize=(5, 4))
    plt.hist(y, bins=30)
    plt.xlabel("Em (mV)")
    plt.ylabel("Count")
    plt.title("Em distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["figures"], "em_hist.png"), dpi=150)
    plt.close()

    if "pH" in df.columns:
        plt.figure(figsize=(5, 4))
        plt.hist(df["pH"].dropna(), bins=20)
        plt.xlabel("pH")
        plt.ylabel("Count")
        plt.title("pH distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(paths["figures"], "ph_hist.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(5, 4))
    plt.hist(abs_err, bins=30)
    plt.xlabel("Absolute error (mV)")
    plt.ylabel("Count")
    plt.title("Absolute error distribution (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths["figures"], "abs_error_hist.png"), dpi=150)
    plt.close()

    oof = _compute_oof_preds(
        best_model_name,
        X,
        y,
        cv_folds,
        spec,
        cat_params,
        xgb_params,
        sample_weight,
    )
    if oof is not None:
        df_resid = df.copy()
        df_resid["pred"] = oof
        df_resid["abs_error"] = np.abs(df_resid["pred"] - df_resid["Em"])
        cols = ["uniprot_id", "pdb_id", "Em", "pH", "temperature_C", "cofactor", "pred", "abs_error"]
        keep_cols = [c for c in cols if c in df_resid.columns]
        top_resid = df_resid.sort_values("abs_error", ascending=False).head(30)
        top_resid[keep_cols].to_csv(os.path.join(paths["reports"], "top_residuals.csv"), index=False)

    # Permutation importance on test set
    try:
        class _Wrapper:
            def __init__(self, name: str):
                self.name = name
            def fit(self, X_in, y_in=None):
                return self
            def predict(self, X_in):
                X_df = pd.DataFrame(X_in, columns=spec.feature_cols)
                return _predict_model(self.name, models, spec, X_df, cat_medians)

        wrapper = _Wrapper(best_model_name)
        perm = permutation_importance(
            wrapper,
            X_test,
            y_test,
            n_repeats=5,
            random_state=config.random_seed,
            scoring="neg_mean_absolute_error",
        )
        sorted_idx = np.argsort(perm.importances_mean)[::-1][:30]
        plt.figure(figsize=(8, 10))
        plt.barh(
            [spec.feature_cols[i] for i in sorted_idx][::-1],
            perm.importances_mean[sorted_idx][::-1],
        )
        plt.xlabel("Permutation Importance (neg MAE)")
        plt.title("Top Permutation Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(paths["figures"], "permutation_importance.png"), dpi=150)
        plt.close()
    except Exception as exc:
        logging.warning("Permutation importance skipped: %s", exc)

    # Explainability for CatBoost
    fig_importance = os.path.join(paths["figures"], "feature_importance.png")
    fig_shap = os.path.join(paths["figures"], "shap_summary.png")
    shap_note = None
    if "catboost" in models_succeeded:
        try:
            plot_feature_importance(models["catboost"], spec.feature_cols, fig_importance)
            X_full_cb, _ = prepare_catboost_frame(X, spec, medians=bundles["catboost"]["numeric_medians"])
            shap_note = plot_shap_summary(models["catboost"], X_full_cb, fig_shap)
        except Exception as exc:
            shap_note = f"explainability skipped: {exc}"

    # Summary markdown
    summary_path = os.path.join(paths["reports"], "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Redox Potential Prediction Summary\n\n")
        f.write(f"Best model: **{best_model_name}**\n\n")
        f.write(f"Selection metric: **{metric_key.upper()}** (split: **{config.selection_split.lower()}**)\n\n")
        f.write("## Split sizes\n")
        f.write(f"- Train: {len(splits['train'])}\n")
        f.write(f"- Val: {len(splits['val'])}\n")
        f.write(f"- Test: {len(splits['test'])}\n\n")
        selection_split = config.selection_split.lower()
        f.write("## Selection metrics\n")
        if selection_split == "cv":
            for line in _render_metrics_table(metrics_by_model, cv_agg, "cv", include_std=True):
                f.write(f"{line}\n")
        else:
            for line in _render_metrics_table(metrics_by_model, cv_agg, selection_split, include_std=False):
                f.write(f"{line}\n")
        f.write("\n## Test metrics (not used for selection)\n")
        for line in _render_metrics_table(metrics_by_model, cv_agg, "test", include_std=False):
            f.write(f"{line}\n")
        f.write("\n## Artifacts\n")
        f.write(f"- Models: {paths['models']}\n")
        f.write(f"- Reports: {paths['reports']}\n")
        f.write(f"- Figures: {paths['figures']}\n")
        if shap_note:
            f.write(f"\nNote: SHAP summary skipped ({shap_note}).\n")

    logging.info("Models evaluated successfully: %s", models_succeeded)
    logging.info("Models failed: %s", list(models_failed.keys()))

    return {
        "best_model": best_model_name,
        "best_model_path": best_model_path,
        "metrics": metrics_payload,
        "summary_path": summary_path,
    }
