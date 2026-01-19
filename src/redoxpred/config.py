from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import os
import yaml


@dataclass(frozen=True)
class Config:
    data_path: str
    artifacts_dir: str
    random_seed: int
    test_size: float
    val_size: float
    n_splits: int
    selection_metric: str
    selection_split: str
    tune: bool
    tune_trials: int
    models: list
    ensemble: bool
    catboost_params: Dict[str, Any]
    ridge_params: Dict[str, Any]
    xgb_params: Dict[str, Any]


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_path = raw.get("data", {}).get("path", "data/redox_dataset.csv")
    artifacts_dir = raw.get("artifacts", {}).get("dir", "artifacts")

    random_seed = int(raw.get("random_seed", 42))
    test_size = float(raw.get("test_size", 0.2))
    val_size = float(raw.get("val_size", 0.1))
    n_splits = int(raw.get("n_splits", 5))
    selection_metric = str(raw.get("selection_metric", "MAE"))
    selection_split = str(raw.get("selection_split", "val"))
    tune = bool(raw.get("tuning", False))
    tune_trials = int(raw.get("tune_trials", 25))
    models = raw.get("models", ["dummy", "ridge", "catboost"])
    ensemble = bool(raw.get("ensemble", False))

    catboost_params = raw.get("catboost", {})
    ridge_params = raw.get("ridge", {})
    xgb_params = raw.get("xgboost", {})

    return Config(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        random_seed=random_seed,
        test_size=test_size,
        val_size=val_size,
        n_splits=n_splits,
        selection_metric=selection_metric,
        selection_split=selection_split,
        tune=tune,
        tune_trials=tune_trials,
        models=models,
        ensemble=ensemble,
        catboost_params=catboost_params,
        ridge_params=ridge_params,
        xgb_params=xgb_params,
    )
