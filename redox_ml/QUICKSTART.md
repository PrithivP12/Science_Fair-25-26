# Quick Start Guide

## Step 1: Aggregate Features

If your features are in per-protein JSON files (from `batch_add_features.py`):

```bash
python scripts/aggregate_features.py \
  --input_dir /Users/prithivponnusamy/Downloads/FAD \
  --output data/features_all.csv
```

This scans all subdirectories for `features.json` files and combines them.

## Step 2: Generate QC Report

```bash
python scripts/qc_report.py \
  --input data/features_all.csv \
  --output reports/qc_report.html
```

Open `reports/qc_report.html` in a browser to see what gets filtered.

## Step 3: Preprocess Features

```bash
python scripts/preprocess.py \
  --input data/features_all.csv \
  --output data/features_clean.csv \
  --qc_output data/features_rejected.csv \
  --missing_strategy indicator \
  --scaler robust
```

This:
- Applies QC filters (removes clashed/bad structures)
- Handles missing values
- Scales features
- Saves rejected samples for inspection

## Step 4A: Train Model (if you have labels)

Create a labels CSV with columns: `id`, `em_mv`

```bash
python train.py \
  --data data/features_clean.csv \
  --labels data/labels.csv \
  --output models/xgboost_model.pkl
```

## Step 4B: Representation Learning (if NO labels)

```bash
python scripts/train_representation.py \
  --data data/features_clean.csv \
  --output models/representation_model.pkl \
  --method pca \
  --encoding_dim 64
```

## Step 5: Make Predictions

```bash
python predict.py \
  --model models/xgboost_model.pkl \
  --data data/features_clean.csv \
  --output predictions.csv
```

## Expected Outputs

- `data/features_all.csv` - All aggregated features
- `data/features_clean.csv` - QC-filtered, preprocessed features
- `data/features_rejected.csv` - Samples that failed QC
- `reports/qc_report.html` - QC analysis report
- `models/xgboost_model.pkl` - Trained model
- `models/feature_importance.csv` - Feature importance rankings
- `models/predictions.png` - Prediction vs true scatter plot
- `predictions.csv` - Final predictions

## Customization

### Adjust QC Thresholds

Edit `src/schema.py`:
```python
class QCRules:
    MIN_DISTANCE_THRESHOLD = 1.2  # Change this
    MAX_CLASH_COUNT = 10  # Change this
    MIN_COFACTOR_CONFIDENCE = 0.3  # Change this
```

### Change Missing Value Strategy

Options: `drop`, `mean`, `median`, `indicator`, `tree_friendly`

`indicator` adds binary columns for missing values (good for tree models).
`tree_friendly` only adds indicators for features with >50% missing.

### Adjust Model Parameters

Edit `train.py` or pass custom parameters:
```python
params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.01,
    # ... etc
}
```

## Troubleshooting

**"No features loaded"**
- Check that `features.json` files exist in subdirectories
- Verify JSON structure matches expected format

**"All samples rejected by QC"**
- Check `reports/qc_report.html` for rejection reasons
- Adjust thresholds in `src/schema.py` if too strict

**"Feature mismatch during prediction"**
- Ensure using same preprocessing as training
- Check that feature columns match

