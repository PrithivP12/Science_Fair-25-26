#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from redoxpred.preprocess import preprocess_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU preprocessing for redox dataset")
    ap.add_argument("--in", dest="input_path", default="data/redox_dataset.csv", help="Input CSV path")
    ap.add_argument(
        "--out",
        dest="output_path",
        default="data/redox_dataset_preprocessed.parquet",
        help="Output path (parquet preferred; falls back to CSV if parquet engine missing)",
    )
    ap.add_argument(
        "--report",
        dest="report_path",
        default="artifacts/reports/preprocess_report.md",
        help="Report path (markdown)",
    )
    args = ap.parse_args()

    saved_path = preprocess_file(args.input_path, args.output_path, args.report_path)
    print(f"Preprocessed data written to: {saved_path}")
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
