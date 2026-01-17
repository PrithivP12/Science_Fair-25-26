#!/bin/bash
# Example usage script for redox ML pipeline

# Step 1: Aggregate features
echo "Step 1: Aggregating features..."
python scripts/aggregate_features.py \
  --input_dir /Users/prithivponnusamy/Downloads/FAD \
  --output data/features_all.csv

# Step 2: QC report
echo "Step 2: Generating QC report..."
python scripts/qc_report.py \
  --input data/features_all.csv \
  --output reports/qc_report.html

# Step 3: Preprocess
echo "Step 3: Preprocessing features..."
python scripts/preprocess.py \
  --input data/features_all.csv \
  --output data/features_clean.csv \
  --qc_output data/features_rejected.csv

echo "Done! Check data/features_clean.csv for preprocessed features."
echo "If you have labels, run: python train.py --data data/features_clean.csv --labels labels.csv --output models/model.pkl"
