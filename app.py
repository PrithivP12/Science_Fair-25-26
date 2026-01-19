import io
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import py3Dmol
import time

ARTIFACT_DIR = Path("artifacts/qc_n5_gpr")
BULK_RESULTS = ARTIFACT_DIR / "bulk_quantum_results.csv"
QUANTUM_PROFILES = ARTIFACT_DIR / "Final_Quantum_Profiles.csv"
SCORECARD = ARTIFACT_DIR / "Final_Project_Scorecard.json"
FAILURE_10 = {"1CF3","1IJH","4MJW","1UMK","1NG4","1VAO","2GMJ","1E0Y","5K9B","1HUV"}
CRITICAL_ST_GAP = 0.01

@st.cache_data
def load_data(_reload_token: int = 0):
    bulk = pd.read_csv(BULK_RESULTS) if BULK_RESULTS.exists() else pd.DataFrame()
    qprof = pd.read_csv(QUANTUM_PROFILES) if QUANTUM_PROFILES.exists() else pd.DataFrame()
    if not qprof.empty and "pdb_id" in qprof:
        qprof["PDB_ID"] = qprof["pdb_id"].astype(str).str.upper()
        qprof = qprof.drop_duplicates(subset=["PDB_ID"], keep="last")
    if not bulk.empty and "pdb_id" in bulk:
        bulk["PDB_ID"] = bulk["pdb_id"].astype(str).str.upper()
    scorecard = {}
    if SCORECARD.exists():
        with open(SCORECARD, "r", encoding="utf-8") as f:
            scorecard = json.load(f)
    return bulk, qprof, scorecard

def get_entry(pdb_id, bulk, qprof):
    if bulk.empty or qprof.empty:
        return None, None
    pid = pdb_id.upper()
    bulk_row = bulk[bulk["PDB_ID"] == pid].head(1) if "PDB_ID" in bulk else pd.DataFrame()
    q_row = qprof[qprof["PDB_ID"] == pid].head(1) if "PDB_ID" in qprof else pd.DataFrame()
    if bulk_row.empty or q_row.empty:
        return None, None
    return bulk_row.iloc[0], q_row.iloc[0]

def render_structure(pdb_id, pdb_text):
    st.subheader("3D Structure")
    view = py3Dmol.view(width=600, height=400)
    if pdb_text:
        view.addModel(pdb_text, "pdb")
    else:
        view.addModel(f"fetch {pdb_id}", "pdb")
    view.setStyle({"cartoon": {"color": "white"}})
    view.addStyle({"resn": ["FMN", "FAD"]}, {"stick": {"color": "yellow"}})
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=420)

def main():
    st.title("Quantum Needle: Phase 6 Hybrid GPR–VQE Dashboard")
    reload_token = st.session_state.get("reload_token", 0)
    bulk, qprof, scorecard = load_data(reload_token)

    if qprof.empty:
        available_ids = []
    else:
        col = "PDB_ID" if "PDB_ID" in qprof else "pdb_id"
    available_ids = sorted(set(qprof[col].astype(str).str.upper()))
    default_id = available_ids[0] if available_ids else ""

    pdb_input = st.text_input("Enter PDB ID (e.g., 1CF3):", value=default_id).strip()
    if available_ids:
        pdb_select = st.selectbox("Or pick a PDB ID from current Phase 6 artifacts:", [""] + available_ids)
        if pdb_select:
            pdb_input = pdb_select
    pdb_file = st.file_uploader("...or upload a .pdb file", type=["pdb"])
    pdb_text, pdb_id = None, None
    if pdb_file is not None:
        pdb_text = pdb_file.getvalue().decode("utf-8")
        for line in pdb_text.splitlines():
            if line.startswith("HEADER") and len(line) >= 66:
                pdb_id = line[62:66].strip()
                break
        if not pdb_id:
            pdb_id = pdb_file.name.replace(".pdb", "")
        st.info("New Protein Detected: Initializing 16-Qubit VQE Engine...")
    elif pdb_input:
        pdb_id = pdb_input

    # Auto-analysis for uploaded PDB not in artifacts
    if pdb_file is not None and pdb_id and (qprof.empty or pdb_id.upper() not in set(qprof["pdb_id"].astype(str).str.upper())):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            tmp.write(pdb_file.getvalue())
            tmp_path = tmp.name
        with st.spinner("Quantum simulation in progress... calculating Spin Density and ST-Gaps."):
            proc = subprocess.run(
                [sys.executable, "scripts/vqe_n5_edge.py", "--pdb", tmp_path],
                capture_output=True,
                text=True,
            )
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
            st.error(f"VQE simulation failed: {err_msg}")
            return
        st.cache_data.clear()
        st.session_state["reload_token"] = st.session_state.get("reload_token", 0) + 1
        # retry load a couple of times to catch fresh writes
        for _ in range(3):
            bulk, qprof, scorecard = load_data(st.session_state["reload_token"])
            available_ids = sorted(set(qprof["PDB_ID"].astype(str).str.upper())) if not qprof.empty else []
            if pdb_id.upper() in set(available_ids):
                break
            time.sleep(2)

    if not pdb_input and "FAD" in available_ids:
        pdb_id = "FAD"
    if pdb_id:
        st.markdown(f"### Selected PDB: `{pdb_id}`")
        bulk_row, q_row = get_entry(pdb_id, bulk, qprof)
        if bulk_row is None or q_row is None:
            st.warning("Calculation pending for this molecule. Please wait for the VQE engine to finish.")
        else:
            # PNAS needle check for this PDB (e.g., FAD)
            st_gap = float(q_row.get("st_gap", float("nan")))
            n5_spin = float(q_row.get("n5_spin_density", float("nan")))
            precision = 1.0 / (st_gap + 1e-6) if pd.notna(st_gap) else float("nan")
            if pdb_id.upper() == "FAD_BASELINE":
                st.info("FAD baseline loaded.")
            st.subheader("Quantum Profile")
            st.json({
                "ST_Gap (mV)": float(q_row.get("st_gap", float("nan"))),
                "HOMO_LUMO_Gap (mV)": float(q_row.get("homo_lumo_gap", float("nan"))),
                "N5_Spin_Density": float(q_row.get("n5_spin_density", float("nan"))),
                "N10_Spin_Density": float(q_row.get("n10_spin_density", float("nan"))),
                "Polarizability": float(q_row.get("quantum_polarizability", float("nan"))),
                "Magnetic_Sensitivity": float(q_row.get("magnetic_sensitivity_index", float("nan"))),
                "Quantum_Precision": precision,
            })

            st.subheader("Magnetoreception Gauge")
            candidate = q_row.get("Magnetoreception_Candidate", False)
            st.metric(
                "Heading Precision Index",
                f"{(1.0 / (st_gap + 1e-6)) * n5_spin:,.1f}" if pd.notna(st_gap) else "N/A",
                delta="CRITICAL" if st_gap < CRITICAL_ST_GAP else "OK",
            )
            if pd.notna(st_gap) and pd.notna(n5_spin) and (st_gap < CRITICAL_ST_GAP) and (n5_spin > 0.4):
                st.success("🎯 CRITICAL DISCOVERY: HIGH-PRECISION QUANTUM NEEDLE DETECTED.\n\nThis protein's electronic signature matches the Hore et al. (PNAS 2016) criteria for microsecond spin coherence and 5-degree heading precision.")
            elif candidate:
                st.warning("Magnetoreception Candidate (PNAS 1600341113 criteria)")
            else:
                st.info("Enzymatic Profile: Stable, but lacking magnetic 'spike' sensitivity.")
            if pdb_id.upper() in FAILURE_10:
                st.warning("Failure-10 outlier: electronically unstable")

            st.subheader("Prediction")
            st.json({
                "Hybrid_Pred_Em (mV)": float(bulk_row.get("pred_final", float("nan"))),
                "GPR_Pred_Em (mV)": float(bulk_row.get("gpr_pred", float("nan"))),
                "Abs_Error_GPR": float(bulk_row.get("abs_err_gpr", float("nan"))),
                "Abs_Error_Hybrid": float(bulk_row.get("abs_err_final", float("nan"))),
            })

            try:
                render_structure(pdb_id, pdb_text)
            except Exception as exc:
                st.warning(f"3D view unavailable: {exc}")

    st.sidebar.header("Project Scorecard")
    if scorecard:
        st.sidebar.json(scorecard)
        if scorecard.get("global_mae_n139", 0) == 0:
            st.sidebar.error("DATABASE CORRUPTED. RUN FULL DATASET RECOVERY.")
    else:
        st.sidebar.info("Run scripts/vqe_n5_edge.py to generate artifacts.")
    if st.sidebar.button("Refresh Data", key="refresh_data"):
        st.cache_data.clear()
        bulk, qprof, scorecard = load_data(st.session_state.get("reload_token", 0))
    debug_path = QUANTUM_PROFILES
    if debug_path.exists():
        try:
            tail = pd.read_csv(debug_path).tail(5)
            st.sidebar.markdown("**Final_Quantum_Profiles tail**")
            st.sidebar.dataframe(tail)
            if tail.empty:
                st.error("Database is empty. Please run the full Phase 6 dataset script first.")
        except Exception as exc:
            st.sidebar.write(f"Debug read failed: {exc}")
    else:
        st.error("Database is empty. Please run the full Phase 6 dataset script first.")
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        bulk, qprof, scorecard = load_data(st.session_state.get("reload_token", 0))
    debug_path = QUANTUM_PROFILES
    if debug_path.exists():
        st.sidebar.markdown("**Final_Quantum_Profiles tail**")
        try:
            tail = pd.read_csv(debug_path).tail(5)
            st.sidebar.dataframe(tail)
        except Exception as exc:
            st.sidebar.write(f"Debug read failed: {exc}")
    top5_path = ARTIFACT_DIR / "Top5_HBond_Stability.csv"
    if top5_path.exists():
        st.sidebar.markdown("**Top 5 H-Bond Stability**")
        st.sidebar.dataframe(pd.read_csv(top5_path))

if __name__ == "__main__":
    main()
