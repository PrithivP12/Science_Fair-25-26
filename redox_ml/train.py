#!/usr/bin/env python3
"""
Train redox potential prediction model.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Optional

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import FeaturePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(features_path: str, labels_path: Optional[str] = None) -> tuple:
    """Load features and labels."""
    logger.info(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    if labels_path:
        logger.info(f"Loading labels from {labels_path}...")
        labels_df = pd.read_csv(labels_path)
        # Merge on 'id' column
        df = df.merge(labels_df, on='id', how='inner')
        y = df['em_mv'].values  # TODO: adjust column name
        df = df.drop(columns=['em_mv'])
    else:
        logger.warning("No labels provided - will train representation learning model")
        y = None
    
    return df, y


def train_xgboost(X_train, y_train, X_val, y_val, params: dict = None):
    """Train XGBoost model."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }
    
    logger.info("Training XGBoost model...")
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100,
    )
    
    return model


def evaluate_model(model, X, y, name: str = "Test"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    logger.info(f"{name} Metrics:")
    logger.info(f"  MAE: {mae:.2f} mV")
    logger.info(f"  RMSE: {rmse:.2f} mV")
    logger.info(f"  R²: {r2:.3f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_true': y,
        'y_pred': y_pred,
    }


def plot_predictions(y_true, y_pred, output_path: Path):
    """Plot predictions vs true values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Em (mV)')
    plt.ylabel('Predicted Em (mV)')
    plt.title('Predictions vs True Values')
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved prediction plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train redox potential model")
    parser.add_argument("--data", required=True, help="Preprocessed features CSV")
    parser.add_argument("--labels", help="Labels CSV (id, em_mv columns)")
    parser.add_argument("--output", required=True, help="Output model path (.pkl)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    # Load data
    df, y = load_data(args.data, args.labels)
    
    if y is None:
        logger.error("Labels required for supervised training!")
        logger.info("For unsupervised/representation learning, use train_representation.py")
        return
    
    # Separate features from metadata
    feature_cols = [c for c in df.columns if c not in ['id', 'status']]
    X = df[feature_cols]
    
    # Preprocess
    logger.info("Preprocessing features...")
    preprocessor = FeaturePreprocessor(missing_strategy='indicator', scaler_type='robust')
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=args.test_size, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size/(1-args.test_size), random_state=args.random_state
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Plot
    plot_path = Path(args.output).parent / "predictions.png"
    plot_predictions(test_metrics['y_true'], test_metrics['y_pred'], plot_path)
    
    # Feature importance
    importance = model.feature_importances_
    feature_names = X_processed.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    
    importance_path = Path(args.output).parent / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Top 10 features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model and preprocessor
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': list(feature_names),
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()

