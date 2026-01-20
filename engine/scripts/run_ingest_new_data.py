#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

NEW_TABLE = """UniProt_ID,Redox_Potential (mV),pH,Temp (°C),Method,Source
A0A075BSX9,+131 mV (FMN cofactor),7.0,25,Experimental,UniProt annotation (PubMed:19762326)
A0A075TRK9,+70 mV (FAD mutant),7.0,25,Experimental,UniProt annotation (PubMed:19159186)
A0A248QE08,-277 mV,7.0,25,Experimental,UniProt annotation (literature)
A0A499UB99,-63 mV,8.0,25,Experimental,UniProt annotation (literature)
P00371,-63 mV,8.0,25,Experimental,UniProt annotation (Pig kidney DAAO)
P00363,-241 mV / -412 mV (2-step),7.5,25,Experimental,UniProt annotation (E. coli FrdA)
O64743,-285 mV,7.0,25,Experimental,UniProt annotation (SoxR)
P0CC17,-270 mV,7.0,20,Experimental,UniProt annotation (NrfH heme)
Q8U195,-270 mV,7.4,25,Experimental,UniProt annotation (Thioredoxin)
Q9BRQ8,-479 mV (heme, high pH),8.0,30,Experimental,UniProt annotation (SoxAX)
Q9BRQ8,-430 mV (heme, low pH),6.0,30,Experimental,UniProt annotation (SoxAX with Cu)
Q9DCX8,-108 mV,7.0,25,Experimental,UniProt annotation (Cytochrome b5 reductase 4)
P0ABE5,+20 mV,7.4,25,Experimental,Purification & Properties of cytochrome b561
"""


def parse_em_values(em_str: str) -> List[float]:
    # Extract signed numbers
    nums = re.findall(r"[-+]?\d+\.?\d*", em_str)
    return [float(n) for n in nums]


def infer_cofactor(text: str) -> str:
    t = text.lower()
    if "fmn" in t:
        return "FMN"
    if "fad" in t:
        return "FAD"
    if "heme" in t:
        return "heme"
    return ""


def main() -> None:
    base_path = ROOT / "data" / "redox_dataset.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    df_base = pd.read_csv(base_path, low_memory=False)

    # Parse new table
    lines = NEW_TABLE.strip().splitlines()
    header = lines[0]
    cleaned_rows = []
    for line in lines[1:]:
        cleaned = re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).replace(",", ";") + ")", line)
        cleaned_rows.append(cleaned)
    reader = csv.DictReader(io.StringIO("\n".join([header] + cleaned_rows)))
    new_rows: List[Dict[str, Any]] = []
    for row in reader:
        em_vals = parse_em_values(row["Redox_Potential (mV)"])
        if not em_vals:
            continue
        note = row["Redox_Potential (mV)"]
        cofactor = infer_cofactor(note)
        for em in em_vals:
            new_rows.append(
                {
                    "uniprot_id": row["UniProt_ID"],
                    "pdb_id": "",
                    "Em": em,
                    "pH": float(row["pH"]) if row["pH"] else None,
                    "temperature_C": float(row["Temp (°C)"]) if row["Temp (°C)"] else None,
                    "cofactor": cofactor if cofactor else pd.NA,
                    "in_jcim": False,
                    "method": row["Method"],
                    "source": row["Source"],
                    "measurement_note": note,
                }
            )

    df_new = pd.DataFrame(new_rows)

    # Duplicate check
    dups = []
    keep_rows = []
    for _, r in df_new.iterrows():
        mask = (df_base["uniprot_id"] == r["uniprot_id"])
        if "Em" in df_base:
            mask &= (df_base["Em"] - r["Em"]).abs() <= 5
        if "pH" in df_base:
            mask &= (df_base["pH"] - r["pH"]).abs() <= 0.1
        if "temperature_C" in df_base:
            mask &= (df_base["temperature_C"] - r["temperature_C"]).abs() <= 2
        dup_matches = df_base[mask]
        if not dup_matches.empty:
            first = dup_matches.iloc[0]
            dups.append(
                {
                    "uniprot_id": r["uniprot_id"],
                    "Em_new": r["Em"],
                    "Em_existing": first["Em"],
                    "pH": r["pH"],
                    "temperature_C": r["temperature_C"],
                    "reason": "close match to existing row",
                }
            )
        else:
            keep_rows.append(r)

    df_keep = pd.DataFrame(keep_rows)
    merged = pd.concat([df_base, df_keep], ignore_index=True)

    out_path = ROOT / "data" / "redox_dataset_merged.csv"
    merged.to_csv(out_path, index=False)

    dup_path = ROOT / "artifacts" / "reports" / "new_data_duplicates.csv"
    dup_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dups).to_csv(dup_path, index=False)

    ingest_report = ROOT / "artifacts" / "reports" / "new_data_ingest.md"
    ingest_report.parent.mkdir(parents=True, exist_ok=True)
    with ingest_report.open("w", encoding="utf-8") as f:
        f.write("# New Data Ingest\n\n")
        f.write(f"- Rows provided: {len(df_new)}\n")
        f.write(f"- Parsed successfully: {len(df_new)}\n")
        f.write(f"- Duplicates skipped: {len(dups)} (see {dup_path.name})\n")
        f.write(f"- Added rows: {len(df_keep)}\n\n")
        f.write("## Examples (up to 5)\n")
        f.write(df_keep.head(5).to_csv(index=False))

    print(f"Merged dataset written to {out_path}")
    print(f"Ingest report: {ingest_report}")
    print(f"Duplicates logged to: {dup_path}")


if __name__ == "__main__":
    main()
