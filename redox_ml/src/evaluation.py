"""
Evaluation metrics and error analysis for redox potential prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Confidence intervals (bootstrap)
    n_bootstrap = 1000
    mae_samples = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        mae_samples.append(np.mean(np.abs(y_true[indices] - y_pred[indices])))
    
    mae_ci = np.percentile(mae_samples, [2.5, 97.5])
    
    return {
        'mae_mv': mae,
        'rmse_mv': rmse,
        'r2': r2,
        'mae_ci_lower': mae_ci[0],
        'mae_ci_upper': mae_ci[1],
    }


def error_analysis(y_true: np.ndarray, y_pred: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """Analyze errors by metadata groups."""
    errors = np.abs(y_true - y_pred)
    
    analysis = []
    
    # By cofactor type
    if 'cofactor_detected_from_complex' in metadata.columns:
        for cofactor in metadata['cofactor_detected_from_complex'].unique():
            mask = metadata['cofactor_detected_from_complex'] == cofactor
            if mask.sum() > 0:
                analysis.append({
                    'group': 'cofactor',
                    'value': cofactor,
                    'n_samples': mask.sum(),
                    'mean_error': errors[mask].mean(),
                    'median_error': np.median(errors[mask]),
                })
    
    # By burial depth
    if 'ligand_burial_fraction' in metadata.columns:
        for bin_edge in [0.2, 0.4, 0.6, 0.8, 1.0]:
            mask = metadata['ligand_burial_fraction'] < bin_edge
            if mask.sum() > 0:
                analysis.append({
                    'group': 'burial',
                    'value': f'<{bin_edge}',
                    'n_samples': mask.sum(),
                    'mean_error': errors[mask].mean(),
                    'median_error': np.median(errors[mask]),
                })
    
    # By structure source
    if 'structure_source' in metadata.columns:
        for source in metadata['structure_source'].unique():
            mask = metadata['structure_source'] == source
            if mask.sum() > 0:
                analysis.append({
                    'group': 'source',
                    'value': source,
                    'n_samples': mask.sum(),
                    'mean_error': errors[mask].mean(),
                    'median_error': np.median(errors[mask]),
                })
    
    return pd.DataFrame(analysis)


def sanity_check(model, X, y, metadata: pd.DataFrame) -> Dict:
    """Check for model artifacts and data leakage."""
    warnings = []
    
    # Check if model is learning from structure_source
    if 'structure_source' in metadata.columns:
        source_importance = model.feature_importances_[X.columns.str.contains('structure_source')]
        if source_importance.sum() > 0.1:
            warnings.append("High importance on structure_source - possible artifact")
    
    # Check correlation with clash_count (should be low if model is good)
    if 'ligand_protein_clash_count' in metadata.columns:
        corr = np.corrcoef(metadata['ligand_protein_clash_count'], y)[0, 1]
        if abs(corr) > 0.5:
            warnings.append(f"High correlation with clash_count ({corr:.2f}) - check for leakage")
    
    # Check if predictions are too uniform
    pred_std = np.std(model.predict(X))
    if pred_std < 10:
        warnings.append(f"Predictions too uniform (std={pred_std:.1f} mV) - model may not be learning")
    
    return {
        'warnings': warnings,
        'passed': len(warnings) == 0,
    }

