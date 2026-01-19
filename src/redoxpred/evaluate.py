from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "median_ae": med_ae}


def tolerance_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_err = np.abs(y_true - y_pred)
    return {
        "within_25": float(np.mean(abs_err <= 25.0)),
        "within_50": float(np.mean(abs_err <= 50.0)),
    }


def cofactor_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    cofactor_col: str = "cofactor",
) -> Optional[Dict[str, Dict[str, float]]]:
    if cofactor_col not in df.columns:
        return None

    out: Dict[str, Dict[str, float]] = {}
    for cof, grp in df.groupby(cofactor_col):
        if grp.empty:
            continue
        out[str(cof)] = regression_metrics(grp[y_true_col], grp[y_pred_col])
    return out


def plot_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pH: Optional[pd.Series],
    out_dir: str,
) -> None:
    residuals = y_pred - y_true

    # Pred vs True
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=12)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1)
    plt.xlabel("True Em (mV)")
    plt.ylabel("Pred Em (mV)")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pred_vs_true.png", dpi=150)
    plt.close()

    # Residual vs Em
    plt.figure(figsize=(5, 4))
    plt.scatter(y_true, residuals, alpha=0.6, s=12)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("True Em (mV)")
    plt.ylabel("Residual (pred - true)")
    plt.title("Residuals vs Em")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residuals_vs_em.png", dpi=150)
    plt.close()

    # Residual vs pH
    if pH is not None:
        plt.figure(figsize=(5, 4))
        plt.scatter(pH, residuals, alpha=0.6, s=12)
        plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("pH")
        plt.ylabel("Residual (pred - true)")
        plt.title("Residuals vs pH")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/residuals_vs_pH.png", dpi=150)
        plt.close()
