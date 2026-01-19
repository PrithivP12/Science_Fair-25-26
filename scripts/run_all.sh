#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_preprocess.py --in data/redox_dataset.csv --out data/redox_dataset_preprocessed.csv --report artifacts/reports/preprocess_report.md
python3 scripts/run_train.py --config configs/default.yaml
