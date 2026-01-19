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
    num_groups: int
    n_meas_stats: Dict[str, float]
    frac_missing_ph_before: float
    frac_missing_ph_after: float
    frac_missing_temp_before: float
    frac_missing_temp_after: float
    widest_proteins: pd.DataFrame


def _coerce_float(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series.replace(r"^\s*$", np.nan, regex=True), errors="coerce")


def _condition_key(row: pd.Series) -> Tuple[Any, ...]:
    return (
        row.get("uniprot_id", "NA"),
        row.get("cofactor", "missing"),
        row.get("pH_rounded", "NA"),
        row.get("temperature_rounded", "NA"),
    )


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

    # Flags and condition features
    df["has_pH"] = df["pH"].notna().astype(int)
    df["has_temp"] = df["temperature_C"].notna().astype(int)
    df["pH_centered"] = df["pH"] - 7.0
    df["pH_sq"] = df["pH_centered"] ** 2
    df["temperature_centered"] = df["temperature_C"] - 25.0

    # Rounding bins
    df["pH_rounded"] = df["pH"].apply(lambda x: round(x * 2) / 2 if pd.notna(x) else np.nan)
    df["temperature_rounded"] = df["temperature_C"].apply(lambda x: round(x / 5) * 5 if pd.notna(x) else np.nan)

    # Condition key and dedup
    df["cofactor"] = df.get("cofactor", "missing").fillna("missing")
    df["cond_key"] = df.apply(_condition_key, axis=1)

    grouped = df.groupby("cond_key")
    agg_df = grouped.agg(
        {
            TARGET_COL: ["mean", "std", "count"],
            "pH": "first",
            "temperature_C": "first",
            "pH_centered": "first",
            "pH_sq": "first",
            "temperature_centered": "first",
            "has_pH": "first",
            "has_temp": "first",
            "pH_rounded": "first",
            "temperature_rounded": "first",
            "cofactor": "first",
            "uniprot_id": "first",
            "pdb_id": "first",
        }
    ).reset_index(drop=True)
    agg_df.columns = [
        "Em_mean",
        "Em_std",
        "n_measurements",
        "pH",
        "temperature_C",
        "pH_centered",
        "pH_sq",
        "temperature_centered",
        "has_pH",
        "has_temp",
        "pH_rounded",
        "temperature_rounded",
        "cofactor",
        "uniprot_id",
        "pdb_id",
    ]
    agg_df["Em"] = agg_df["Em_mean"]
    agg_df["Em_std"] = agg_df["Em_std"].fillna(0.0)

    frac_missing_ph_after = float(agg_df["pH"].isna().mean())
    frac_missing_temp_after = float(agg_df["temperature_C"].isna().mean())

    # n_measurements distribution stats
    n_meas = agg_df["n_measurements"]
    n_meas_stats = {
        "min": float(n_meas.min()),
        "max": float(n_meas.max()),
        "mean": float(n_meas.mean()),
        "median": float(n_meas.median()),
    }

    # Proteins with widest Em range across conditions (from original df)
    ranges = df.groupby("uniprot_id")[TARGET_COL].agg(["min", "max"])
    ranges["Em_range"] = ranges["max"] - ranges["min"]
    widest = ranges.sort_values("Em_range", ascending=False).head(20).reset_index()

    report = PreprocessReport(
        rows_before=rows_before,
        rows_after=len(agg_df),
        num_groups=len(agg_df),
        n_meas_stats=n_meas_stats,
        frac_missing_ph_before=frac_missing_ph_before,
        frac_missing_ph_after=frac_missing_ph_after,
        frac_missing_temp_before=frac_missing_temp_before,
        frac_missing_temp_after=frac_missing_temp_after,
        widest_proteins=widest,
    )
    return agg_df, report


def _write_report(report: PreprocessReport, path: Path) -> None:
    ensure_dir(str(path.parent))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Preprocess Report\n\n")
        f.write(f"- Rows before: {report.rows_before}\n")
        f.write(f"- Rows after dedup: {report.rows_after}\n")
        f.write(f"- Deduplicated condition groups: {report.num_groups}\n")
        f.write(
            f"- Missing pH: before {report.frac_missing_ph_before*100:.2f}% | after {report.frac_missing_ph_after*100:.2f}%\n"
        )
        f.write(
            f"- Missing temperature: before {report.frac_missing_temp_before*100:.2f}% | after {report.frac_missing_temp_after*100:.2f}%\n"
        )
        f.write(
            f"- n_measurements stats: min={report.n_meas_stats['min']}, median={report.n_meas_stats['median']}, "
            f"mean={report.n_meas_stats['mean']:.2f}, max={report.n_meas_stats['max']}\n"
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
