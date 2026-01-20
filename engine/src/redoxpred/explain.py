from __future__ import annotations

from typing import Dict, List, Optional
import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(
    model,
    feature_names: List[str],
    out_path: str,
    top_n: int = 30,
) -> None:
    try:
        importances = model.get_feature_importance()
    except Exception:
        logging.warning("Model does not support feature importance")
        return

    if len(importances) != len(feature_names):
        logging.warning("Feature importance length mismatch; skipping plot")
        return

    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = [importances[i] for i in idx]

    plt.figure(figsize=(8, 10))
    plt.barh(list(reversed(names)), list(reversed(vals)))
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap_summary(
    model,
    X,
    out_path: str,
    max_display: int = 30,
) -> Optional[str]:
    try:
        import shap
    except Exception:
        return "shap not installed"

    try:
        shap_values = model.get_feature_importance(type="ShapValues", data=X)
        shap_vals = shap_values[:, :-1]
    except Exception:
        return "failed to compute shap values"

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shap.summary_plot(shap_vals, X, show=False, max_display=max_display)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return None
    except Exception:
        return "failed to render shap plot"
