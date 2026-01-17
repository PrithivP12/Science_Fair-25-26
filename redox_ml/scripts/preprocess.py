#!/usr/bin/env python3
"""
Preprocess features: apply QC filters and handle missing values.
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import apply_qc_filters, FeaturePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess features")
    parser.add_argument("--input", required=True, help="Input CSV with features")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--qc_output", help="Output CSV for rejected samples")
    parser.add_argument("--missing_strategy", default="indicator", 
                       choices=["drop", "mean", "median", "indicator", "tree_friendly"],
                       help="Strategy for handling missing values")
    parser.add_argument("--scaler", default="robust", choices=["standard", "robust", "none"],
                       help="Scaling method")
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Apply QC filters
    logger.info("Applying QC filters...")
    passed_df, rejected_df = apply_qc_filters(df, log_rejections=True)
    
    # Save rejected samples if requested
    if args.qc_output and len(rejected_df) > 0:
        Path(args.qc_output).parent.mkdir(parents=True, exist_ok=True)
        rejected_df.to_csv(args.qc_output, index=False)
        logger.info(f"Saved {len(rejected_df)} rejected samples to {args.qc_output}")
    
    # Fit preprocessor (on passed samples only)
    logger.info("Fitting preprocessor...")
    preprocessor = FeaturePreprocessor(
        missing_strategy=args.missing_strategy,
        scaler_type=None if args.scaler == "none" else args.scaler,
    )
    
    # Separate features from metadata
    feature_cols = [c for c in passed_df.columns 
                   if c not in ['id', 'status', 'cofactor_detected_from_complex']]
    X = passed_df[feature_cols]
    
    preprocessor.fit(X)
    X_processed = preprocessor.transform(X)
    
    # Combine with metadata
    metadata_cols = ['id', 'status']
    metadata_cols = [c for c in metadata_cols if c in passed_df.columns]
    output_df = pd.concat([
        passed_df[metadata_cols].reset_index(drop=True),
        X_processed.reset_index(drop=True)
    ], axis=1)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df)} preprocessed samples to {args.output}")
    logger.info(f"Final feature count: {len(X_processed.columns)}")


if __name__ == "__main__":
    main()

