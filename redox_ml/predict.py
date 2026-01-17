#!/usr/bin/env python3
"""
Predict redox potentials using trained model.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Predict redox potentials")
    parser.add_argument("--model", required=True, help="Trained model path (.pkl)")
    parser.add_argument("--data", required=True, help="Preprocessed features CSV")
    parser.add_argument("--output", required=True, help="Output predictions CSV")
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Separate features
    feature_cols = [c for c in df.columns if c not in ['id', 'status']]
    X = df[feature_cols]
    
    # Preprocess
    logger.info("Preprocessing...")
    X_processed = preprocessor.transform(X)
    
    # Predict
    logger.info("Making predictions...")
    predictions = model.predict(X_processed)
    
    # Create output
    output_df = pd.DataFrame({
        'id': df['id'].values,
        'predicted_em_mv': predictions,
    })
    
    # Add uncertainty if available (quantile regression)
    if hasattr(model, 'predict_quantiles'):
        quantiles = model.predict_quantiles(X_processed, quantiles=[0.05, 0.95])
        output_df['predicted_em_mv_lower'] = quantiles[:, 0]
        output_df['predicted_em_mv_upper'] = quantiles[:, 1]
        output_df['predicted_em_mv_uncertainty'] = (quantiles[:, 1] - quantiles[:, 0]) / 2
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} predictions to {args.output}")
    logger.info(f"Prediction range: {predictions.min():.1f} to {predictions.max():.1f} mV")


if __name__ == "__main__":
    main()

