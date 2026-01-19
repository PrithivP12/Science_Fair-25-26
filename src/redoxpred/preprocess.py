from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .clean import preprocess_dataframe
from .utils import ensure_dir


def _write_report(meta: Dict[str, Any], output_path: Path) -> None:
    ensure_dir(str(output_path.parent))
    missing_lines = []
    for col, rate in meta.get("missing_rates", {}).items():
        missing_lines.append(f"- {col}: {rate*100:.2f}% missing")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Preprocessing Report\n\n")
        f.write(f"- Rows (raw): {meta.get('rows_before', 'n/a')}\n")
        f.write(f"- Rows after dropping missing Em: {meta.get('rows_after_target', 'n/a')}\n")
        f.write(f"- Rows after deduplication: {meta.get('rows_after_dedup', 'n/a')}\n")
        f.write(f"- Duplicates merged: {meta.get('duplicates_merged', 0)}\n")
        f.write(f"- Temperature dropped: {bool(meta.get('temperature_dropped'))}\n")
        if missing_lines:
            f.write("\nMissingness:\n")
            for line in missing_lines:
                f.write(f"{line}\n")


def preprocess_file(input_path: str, output_path: str, report_path: str) -> str:
    df_raw = pd.read_csv(input_path, low_memory=False)
    processed, meta = preprocess_dataframe(df_raw)

    out_path = Path(output_path)
    ensure_dir(str(out_path.parent))
    saved_path: Path
    if out_path.suffix.lower() == ".parquet":
        try:
            processed.to_parquet(out_path, index=False)
            saved_path = out_path
        except Exception:
            # parquet engine missing; fallback to CSV with same stem
            fallback = out_path.with_suffix(".csv")
            processed.to_csv(fallback, index=False)
            saved_path = fallback
    else:
        processed.to_csv(out_path, index=False)
        saved_path = out_path

    _write_report(meta, Path(report_path))
    return str(saved_path)
