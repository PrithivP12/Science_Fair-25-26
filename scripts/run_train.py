#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redoxpred.train import run_training
from redoxpred.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    ap.add_argument("--models", default=None, help="Comma-separated model list (e.g., catboost,xgb,ensemble)")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    models = args.models.split(",") if args.models else None
    result = run_training(args.config, tune_override=args.tune, models_override=models)
    logging.info("Best model: %s", result["best_model"])
    logging.info("Summary: %s", result["summary_path"])


if __name__ == "__main__":
    main()
