#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from redoxpred.predict import run_prediction
from redoxpred.utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model bundle (.pkl)")
    ap.add_argument("--input", required=True, help="Input CSV with features")
    ap.add_argument("--output", default="artifacts/predictions/preds.csv", help="Output CSV")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    run_prediction(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
