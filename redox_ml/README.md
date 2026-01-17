# Redox Potential Prediction Pipeline for Flavoproteins

## Overview
ML pipeline for predicting redox potentials (Em) of flavoproteins using structure-derived features.

## Project Structure
```
redox_ml/
├── data/              # Raw and processed data
├── src/               # Core modules
├── scripts/           # Utility scripts
├── models/            # Trained models
├── reports/           # QC reports and analysis
└── README.md
```

## Setup

```bash
# Install dependencies
pip install pandas scikit-learn xgboost numpy scipy matplotlib seaborn

# Or with conda
conda install pandas scikit-learn xgboost numpy scipy matplotlib seaborn
```

## Quick Start

### 1. Prepare data
```bash
# Aggregate features from per-protein JSON/CSV files
python scripts/aggregate_features.py --input_dir /path/to/folders --output data/features_all.csv
```

### 2. Run QC and preprocessing
```bash
python scripts/qc_report.py --input data/features_all.csv --output reports/qc_report.html
python scripts/preprocess.py --input data/features_all.csv --output data/features_clean.csv
```

### 3. Train model (if labels available)
```bash
python train.py --data data/features_clean.csv --labels data/labels.csv --output models/xgboost_model.pkl
```

### 4. Predict (if no labels, use pretrained or representation learning)
```bash
python predict.py --model models/xgboost_model.pkl --data data/features_clean.csv --output predictions.csv
```

## Data Schema

See `src/schema.py` for complete feature definitions and QC rules.

## QC Rules

Samples are rejected if:
- `ligand_protein_min_distance < 1.2 Å`
- `ligand_protein_clash_count > 10`
- `cofactor_confidence_score < 0.3`
- Missing critical features (ESP at N5/O4, net_charge_6A)

## License
MIT

