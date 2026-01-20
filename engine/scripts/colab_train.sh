#!/usr/bin/env bash
set -euo pipefail

echo "GPU devices:"
nvidia-smi -L || true

pip install -r requirements.txt
python3 scripts/run_train.py --config configs/default.yaml
