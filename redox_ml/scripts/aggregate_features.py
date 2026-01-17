#!/usr/bin/env python3
"""
Aggregate features from per-protein JSON/CSV files into a single dataframe.
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features_from_json(folder: Path) -> Dict:
    """Load features from features.json file."""
    feat_path = folder / "features.json"
    if not feat_path.exists():
        return None
    
    try:
        data = json.loads(feat_path.read_text())
        # Extract features_extra if present
        features = data.get('features', {})
        features_extra = data.get('features_extra', {})
        
        # Combine
        combined = {**features, **features_extra}
        combined['id'] = folder.name
        combined['status'] = data.get('status', 'UNKNOWN')
        
        return combined
    except Exception as e:
        logger.warning(f"Failed to load {feat_path}: {e}")
        return None


def load_features_from_csv(folder: Path) -> Dict:
    """Load features from CSV file (if exists)."""
    # Check for CSV files in parent directory (from batch_add_features.py output)
    return None


def load_from_master_csv(csv_path: Path) -> pd.DataFrame:
    """Load features from master CSV (output of batch_add_features.py)."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from master CSV")
        return df
    except Exception as e:
        logger.warning(f"Failed to load {csv_path}: {e}")
        return None


def aggregate_features(input_dir: Path, output_path: Path, master_csv: Path = None):
    """Aggregate all feature files into one CSV."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Option 1: Load from master CSV (from batch_add_features.py)
    if master_csv:
        master_csv = Path(master_csv)
        df = load_from_master_csv(master_csv)
        if df is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} rows from master CSV to {output_path}")
            return
    
    # Option 2: Scan per-folder JSON files
    logger.info(f"Scanning {input_dir} for feature files...")
    
    rows = []
    folders_processed = 0
    folders_failed = 0
    
    # Scan all subdirectories
    for folder in input_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Try JSON first
        features = load_features_from_json(folder)
        if features is None:
            # Try CSV
            features = load_features_from_csv(folder)
        
        if features:
            rows.append(features)
            folders_processed += 1
        else:
            folders_failed += 1
        
        if folders_processed % 1000 == 0:
            logger.info(f"Processed {folders_processed} folders...")
    
    logger.info(f"Loaded features from {folders_processed} folders ({folders_failed} failed)")
    
    if not rows:
        logger.error("No features loaded!")
        return
    
    # Create dataframe
    df = pd.DataFrame(rows)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows with {len(df.columns)} columns to {output_path}")
    
    # Print summary
    logger.info(f"\nColumn summary:")
    logger.info(f"  Total columns: {len(df.columns)}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Missing values: {df.isna().sum().sum()}")
    logger.info(f"\nFirst few columns: {list(df.columns[:10])}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate features from per-protein files")
    parser.add_argument("--input_dir", help="Root directory with protein folders")
    parser.add_argument("--master_csv", help="Master CSV file (from batch_add_features.py)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    
    if not args.input_dir and not args.master_csv:
        parser.error("Must provide either --input_dir or --master_csv")
    
    aggregate_features(args.input_dir, args.output, args.master_csv)


if __name__ == "__main__":
    main()

