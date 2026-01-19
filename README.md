# Redox Potential Prediction Pipeline

This project trains models to predict flavin redox potential (Em, mV) from structure-derived features and experimental conditions.

## Workflow (two-stage: local CPU then Colab GPU)

1) Local CPU preprocess

```bash
python3 scripts/run_preprocess.py --in data/redox_dataset.csv --out data/redox_dataset_preprocessed.parquet --report artifacts/reports/preprocess_report.md
```

This cleans types, drops missing Em, adds condition features (pH_centered/pH_sq, has_pH/has_temp), deduplicates repeated measurements, and writes a markdown report. If parquet isn’t available, it falls back to `data/redox_dataset_preprocessed.csv`.

2) Colab GPU train/eval

Upload/clone the repo (with the preprocessed file) into Colab, select a GPU runtime, then run:

```bash
cd /content/OOK    # or your repo path in Colab
bash scripts/colab_train.sh
```

This installs dependencies and trains CatBoost/XGBoost on GPU using `configs/default.yaml`, writing reports/models under `artifacts/`.

3) Predict locally (uses the trained bundle)

```bash
python3 scripts/run_predict.py --model artifacts/models/best_model.pkl --input data/redox_dataset_preprocessed.parquet --output artifacts/predictions/redox_preds.csv
```

## Project Layout

```
src/redoxpred/
  config.py
  data.py
  features.py
  split.py
  train.py
  evaluate.py
  explain.py
  predict.py
  utils.py
scripts/
  run_train.py
  run_predict.py
configs/
  default.yaml
artifacts/
  models/
  reports/
  figures/
  predictions/
```

## Notes

- Group-based splitting uses `uniprot_id` to avoid leakage.
- Identifier/path columns are excluded from training features.
- CatBoost handles categorical features directly; linear models use one-hot encoding.
- SHAP plots are generated if `shap` is installed.
