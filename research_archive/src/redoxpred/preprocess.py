from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from .utils import ensure_dir


TARGET_COL = "Em"


@dataclass
class PreprocessReport:
    rows_before: int
    rows_after: int
    exact_duplicates_dropped: int
    frac_missing_ph_before: float
    frac_missing_ph_after: float
    frac_missing_temp_before: float
    frac_missing_temp_after: float
    widest_proteins: pd.DataFrame
    group_stats: Dict[str, float]


def _coerce_float(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series.replace(r"^\s*$", np.nan, regex=True), errors="coerce")


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, PreprocessReport]:
    rows_before = len(df)

    # Normalize and coerce numeric
    df = df.copy()
    df[TARGET_COL] = _coerce_float(df.get(TARGET_COL))
    df["pH"] = _coerce_float(df.get("pH"))
    df["temperature_C"] = _coerce_float(df.get("temperature_C"))

    frac_missing_ph_before = float(df["pH"].isna().mean() if "pH" in df.columns else 1.0)
    frac_missing_temp_before = float(df["temperature_C"].isna().mean() if "temperature_C" in df.columns else 1.0)

    # Drop rows without target
    df = df[df[TARGET_COL].notna()].copy()

    # Drop exact duplicates only
    dedup_subset = ["uniprot_id", "pdb_id", "Em", "pH", "temperature_C"]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_subset)
    exact_duplicates_dropped = before_dedup - len(df)

    # Flags and condition features
    df["has_pH"] = df["pH"].notna().astype(int)
    df["has_temp"] = df["temperature_C"].notna().astype(int)
    df["pH_centered"] = df["pH"] - 7.0
    df["pH_sq"] = df["pH_centered"] ** 2

    frac_missing_ph_after = float(df["pH"].isna().mean())
    frac_missing_temp_after = float(df["temperature_C"].isna().mean())

    group_stats = {}

    # Proteins with widest Em range across conditions (from original df)
    ranges = df.groupby("uniprot_id")[TARGET_COL].agg(["min", "max"])
    ranges["Em_range"] = ranges["max"] - ranges["min"]
    widest = ranges.sort_values("Em_range", ascending=False).head(20).reset_index()

    report = PreprocessReport(
        rows_before=rows_before,
        rows_after=len(df),
        exact_duplicates_dropped=exact_duplicates_dropped,
        frac_missing_ph_before=frac_missing_ph_before,
        frac_missing_ph_after=frac_missing_ph_after,
        frac_missing_temp_before=frac_missing_temp_before,
        frac_missing_temp_after=frac_missing_temp_after,
        widest_proteins=widest,
        group_stats=group_stats,
    )
    return df, report


def _write_report(report: PreprocessReport, path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Preprocess Report\n\n")
        f.write(f"- Rows before: {report.rows_before}\n")
        f.write(f"- Rows after dedup: {report.rows_after}\n")
        f.write(f"- Exact duplicates dropped: {report.exact_duplicates_dropped}\n")
        f.write(
            f"- Missing pH: before {report.frac_missing_ph_before*100:.2f}% | after {report.frac_missing_ph_after*100:.2f}%\n"
        )
        f.write(
            f"- Missing temperature: before {report.frac_missing_temp_before*100:.2f}% | after {report.frac_missing_temp_after*100:.2f}%\n"
        )
        if report.group_stats:
            f.write(
                f"- group_n stats: min={report.group_stats.get('group_n_min')}, median={report.group_stats.get('group_n_median')}, "
                f"mean={report.group_stats.get('group_n_mean')}, max={report.group_stats.get('group_n_max')}\n"
            )
        f.write("\n## Proteins with widest Em range\n")
        f.write("uniprot_id,min,max,Em_range\n")
        for idx, row in report.widest_proteins.iterrows():
            uid = row["uniprot_id"] if "uniprot_id" in row else idx
            f.write(f"{uid},{row['min']},{row['max']},{row['Em_range']}\n")


def preprocess_file(input_path: str, output_path: str, report_path: str) -> str:
    df_raw = pd.read_csv(input_path, low_memory=False)
    processed, rep = preprocess_dataframe(df_raw)

    out_path = Path(output_path)
    ensure_dir(str(out_path.parent))
    processed.to_csv(out_path, index=False)

    _write_report(rep, Path(report_path))
    return str(out_path)
