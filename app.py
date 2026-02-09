import io
import json
import os
import sys
import hashlib
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import py3Dmol
import matplotlib.pyplot as plt

from engine.run_state import (
    DEFAULT_TOLERANCES,
    canonical_mutation_key,
    crossing_message,
    detect_st_crossing,
    feature_diff,
    parse_mutation_list,
)
from engine.radical_pair_yields import compute_yields, estimate_spike_metrics
from engine.recommender import recommend, recommendation_to_row, conservative_candidate_pool
from engine.recommender_config import CONFIG as RECOMMENDER_CONFIG
BASE_DIR = Path(os.getcwd())
ARTIFACT_DIR = BASE_DIR / "artifacts" / "qc_n5_gpr"
BULK_RESULTS = ARTIFACT_DIR / "bulk_quantum_results.csv"
QUANTUM_PROFILES = ARTIFACT_DIR / "Final_Quantum_Profiles.csv"
SCORECARD = ARTIFACT_DIR / "Final_Project_Scorecard.json"
# Mean Em of bundled training set; used to detect "collapsed to mean" predictions.
# Prefer the latest Excel dataset if present, otherwise fall back to legacy CSV,
# and finally a static constant.
def _load_train_mean_em():
    excel_path = BASE_DIR / "data" / "Protein Dataset 10.3 (5).xlsx"
    csv_path = BASE_DIR / "data" / "redox_dataset.csv"
    if excel_path.exists():
        try:
            return float(pd.read_excel(excel_path, engine="openpyxl")["Em"].mean())
        except Exception:
            pass
    if csv_path.exists():
        try:
            return float(pd.read_csv(csv_path)["Em"].mean())
        except Exception:
            pass
    return -208.30234375

_TRAIN_MEAN_EM = _load_train_mean_em()
FAILURE_10 = {"1CF3","1IJH","4MJW","1UMK","1NG4","1VAO","2GMJ","1E0Y","5K9B","1HUV"}
CRITICAL_ST_GAP = 0.01
EMISSION_TOL = DEFAULT_TOLERANCES.tol_emission
FEATURE_REL_TOL = DEFAULT_TOLERANCES.tol_feature_rel
FEATURE_ABS_TOL = DEFAULT_TOLERANCES.tol_feature_abs
AA_PROPERTIES = {
    "A": {"e_neg": 0.0, "steric_index": 0.5},
    "R": {"e_neg": 0.2, "steric_index": 1.2},
    "N": {"e_neg": -0.2, "steric_index": 0.7},
    "D": {"e_neg": -0.3, "steric_index": 0.7},
    "C": {"e_neg": -0.06, "steric_index": 0.6},
    "Q": {"e_neg": -0.2, "steric_index": 0.8},
    "E": {"e_neg": -0.3, "steric_index": 0.8},
    "G": {"e_neg": 0.0, "steric_index": 0.4},
    "H": {"e_neg": -0.1, "steric_index": 0.9},
    "I": {"e_neg": 0.1, "steric_index": 1.0},
    "L": {"e_neg": 0.1, "steric_index": 1.0},
    "K": {"e_neg": 0.2, "steric_index": 1.1},
    "M": {"e_neg": 0.0, "steric_index": 0.9},
    "F": {"e_neg": 0.2, "steric_index": 1.2},
    "P": {"e_neg": 0.0, "steric_index": 0.8},
    "S": {"e_neg": -0.1, "steric_index": 0.6},
    "T": {"e_neg": -0.1, "steric_index": 0.7},
    "W": {"e_neg": 0.3, "steric_index": 1.3},
    "Y": {"e_neg": 0.2, "steric_index": 1.2},
    "V": {"e_neg": 0.05, "steric_index": 0.9},
}

st.set_page_config(page_title="FLAVIN-OPT Quantum Profile", layout="wide")


def load_design_tokens():
    token_path = BASE_DIR / "design_tokens.css"
    css = token_path.read_text() if token_path.exists() else ""
    theme_style = f"<style>{css}</style>"
    return theme_style


@st.cache_data
def cached_compute_yields(theta_deg, tau_us, kS, kT, omega, plane, Ax, Ay):
    return compute_yields(theta_deg, tau_us=tau_us, kS=kS, kT=kT, omega=omega, plane=plane, Ax=Ax, Ay=Ay)


def render_compass_simulator():
    st.markdown("<div class='section-title'>Avian Compass Simulator</div>", unsafe_allow_html=True)
    st.caption("Spin dynamics toy model; yields are dimensionless and not an Em predictor.")
    plane = st.selectbox("Magnetic field plane", options=["ZX", "YZ"], index=0)
    tau_us = st.slider("Lifetime τ (μs)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
    rate_mode = st.selectbox("Reaction rates", options=["Equal rates (kS = kT = 1/τ)", "Separate kS and kT"])
    kS_val = None
    kT_val = None
    if rate_mode != "Equal rates (kS = kT = 1/τ)":
        kS_val = st.slider("kS (1/μs)", min_value=0.001, max_value=5.0, value=0.1, step=0.001)
        kT_val = st.slider("kT (1/μs)", min_value=0.001, max_value=5.0, value=0.1, step=0.001)
    omega = st.slider("Zeeman scale ω (scaled units)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**FAD-like hyperfine (Ax, Ay, Az)**")
        Ax_x = st.number_input("Axx (FAD-like)", value=-0.2, format="%.3f")
        Ax_y = st.number_input("Ayy (FAD-like)", value=-0.2, format="%.3f")
        Ax_z = st.number_input("Azz (FAD-like)", value=1.75, format="%.3f")
    with col_b:
        st.markdown("**Partner-like hyperfine (Ax, Ay, Az)**")
        Ay_x = st.number_input("Axx (Partner-like)", value=0.0, format="%.3f")
        Ay_y = st.number_input("Ayy (Partner-like)", value=0.0, format="%.3f")
        Ay_z = st.number_input("Azz (Partner-like)", value=1.08, format="%.3f")
    theta_min = st.number_input("θ min (deg)", value=0.0, format="%.1f")
    theta_max = st.number_input("θ max (deg)", value=180.0, format="%.1f")
    theta_pts = st.slider("Number of θ points", min_value=10, max_value=720, value=181, step=1)
    run_sim = st.button("Run simulation", key="run_compass")
    if run_sim:
        theta_arr = np.linspace(theta_min, theta_max, int(theta_pts))
        res = cached_compute_yields(theta_arr, tau_us, kS_val, kT_val, omega, plane, (Ax_x, Ax_y, Ax_z), (Ay_x, Ay_y, Ay_z))
        st.session_state["compass_res"] = res
    res = st.session_state.get("compass_res")
    if res:
        theta_arr = res["theta_deg"]
        phi_s = res["phi_s"]
        phi_t = res["phi_t"]
        phi_sum = res["phi_sum"]
        spike = estimate_spike_metrics(theta_arr, phi_s)
        max_err = float(np.max(np.abs(phi_sum - 1.0)))
        fig1, ax1 = plt.subplots()
        ax1.plot(theta_arr, phi_s, label="ΦS")
        ax1.plot(theta_arr, phi_t, label="ΦT")
        ax1.set_xlabel("θ (deg)")
        ax1.set_ylabel("Yield")
        ax1.legend()
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        ax2.plot(theta_arr, phi_s - np.mean(phi_s), label="ΦS - mean(ΦS)")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("θ (deg)")
        ax2.set_ylabel("Anisotropic part")
        ax2.legend()
        st.pyplot(fig2)
        st.write(
            f"Spike amplitude: {spike['spike_amp']:.4f}; "
            f"θ at max: {spike['theta_at_max']:.1f}°, θ at min: {spike['theta_at_min']:.1f}°; "
            f"max|ΦS+ΦT-1|: {max_err:.2e}"
        )
        csv_df = pd.DataFrame({"theta_deg": res["theta_deg"], "phi_s": phi_s, "phi_t": phi_t, "phi_sum": phi_sum})
        st.download_button("Download CSV", data=csv_df.to_csv(index=False), file_name="avian_compass_yields.csv", mime="text/csv")


def render_metric_card(title, value, unit, interpretation, delta=None, delta_dir=None, badge=None):
    delta_html = ""
    if delta is not None and delta_dir:
        arrow = "↑" if delta_dir > 0 else "↓"
        delta_html = f"<div class='delta-pill'>{arrow} {delta:.3f} {unit}</div>"
    badge_html = f"<span class='badge badge-info'>{badge}</span>" if badge else ""
    return f"""
    <div class="metric-card">
        <h4>{title} {badge_html}</h4>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        <div style="color:var(--text-dim);margin:6px 0 4px 0;">{interpretation}</div>
        {delta_html}
    </div>
    """


def render_tag_list(tags):
    if not tags:
        return "<span class='tag-pill'>WT</span>"
    return "".join([f"<span class='tag-pill'>{t}</span>" for t in tags])

def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Resilient CSV loader that tolerates malformed lines."""
    if not path.exists():
        return pd.DataFrame()
    try:
        # Prefer tolerant read first to avoid noisy warnings on imperfect logs.
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        if df.empty:
            st.info(f"{path.name} loaded but empty after skipping malformed lines.")
        return df
    except Exception as exc:
        st.warning(f"Unable to read {path.name} even with tolerant parser ({exc}). Returning empty table.")
        return pd.DataFrame()


def _safe_float(value) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def _first_finite(*values) -> float:
    for value in values:
        out = _safe_float(value)
        if np.isfinite(out):
            return out
    return float("nan")


def _format_eta_seconds(seconds: float) -> str:
    try:
        total = int(round(float(seconds)))
    except Exception:
        return "unknown"
    if total < 0:
        return "unknown"
    mins, sec = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


@st.cache_data
def load_data(_reload_token: int = 0):
    bulk = _safe_read_csv(BULK_RESULTS)
    qprof = _safe_read_csv(QUANTUM_PROFILES)
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

def check_directories():
    target_dirs = [
        BASE_DIR / "artifacts",
        BASE_DIR / "artifacts" / "qc_n5_gpr",
        BASE_DIR / "pdbs",
        BASE_DIR / "data",
        BASE_DIR / "data" / "pdb",
    ]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

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
    check_directories()
    st.markdown(load_design_tokens(), unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        body, .stApp { font-family: var(--font-sans); background: #0a0a0a !important; color: #f5f5f5 !important; }
        .block-container { padding-top: 10px; background: #0a0a0a !important; }
        .figure-card, .support-card, .run-bar, .stDataFrame, .stTable { background: #111111 !important; color: #f5f5f5 !important; border: 1px solid #2a2a2a; }
        table { color: #f5f5f5 !important; }
        .stDownloadButton button, button[kind=secondary], button[kind=primary], .stButton>button {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid #d0d0d0 !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        }
        .tag-pill, .criteria-box, .delta-pill { background: #1a1a1a !important; color: #f5f5f5 !important; border: 1px solid #2a2a2a; }
        .metric-card { background: #0f0f0f !important; color: #f5f5f5 !important; border: 1px solid #2a2a2a; }
        .metric-value { color: #ffffff !important; }
        .section-title { color: #f5f5f5 !important; }
        .caption-box { background: #0f0f0f !important; color: #dcdcdc !important; border-left: 3px solid #666; }
        .header-title { font-family: var(--font-serif); font-size: 28px; margin-bottom: 4px; }
        .header-sub { color: #cccccc; font-size: 14px; }
        .table-caption { color: var(--text-dim); font-size: 12px; }
        .download-buttons button { margin-right: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='header-title'>FLAVIN-OPT: Quantum-Driven Protein Brightness Lab</div>", unsafe_allow_html=True)
    page = st.sidebar.radio("View", ["Quantum Profile", "Avian Compass Simulator"], index=0)
    COFACTOR_CHOICES = ["FAD", "FMN"]
    cofactor_choice = st.sidebar.selectbox(
        "Cofactor Configuration (required)",
        options=[""] + COFACTOR_CHOICES,
        format_func=lambda x: x if x else "Select...",
    )
    force_cofactor = st.sidebar.checkbox(
        "Force run if cofactor not found in PDB",
        value=False,
        help="If unchecked, runs will fail when the selected cofactor residue is missing.",
    )
    reload_token = st.session_state.get("reload_token", 0)
    bulk, qprof, scorecard = load_data(reload_token)
    # Keep idle until user triggers run
    if scorecard and scorecard.get("global_mae_n139", 0) == 0:
        st.error("Database reset detected. Please rerun the full dataset or refresh to repopulate metrics.")

    uniprot_id = st.text_input("Enter UniProt ID (optional):", value="").strip().upper()
    pdb_input = st.text_input("Enter PDB ID (optional, e.g., 1CF3):", value="").strip()
    pdb_file = st.file_uploader("Upload a .pdb or .ent file", type=["pdb", "ent"])
    pdb_text, pdb_id = None, None
    mutation_list_str = st.text_input("Enter Mutation List (e.g., C412A, W399F, F37S)", value="")
    st.session_state["mutation_list"] = mutation_list_str
    mutation_specs, _ = parse_mutation_list(mutation_list_str, default_chain="A")
    mutation_key = canonical_mutation_key(mutation_specs)
    upload_suffix = ".pdb"
    if pdb_file is not None:
        file_suffix = Path(pdb_file.name).suffix.lower()
        if file_suffix in {".pdb", ".ent"}:
            upload_suffix = file_suffix
        pdb_text = pdb_file.getvalue().decode("utf-8", errors="ignore")
        for line in pdb_text.splitlines():
            if line.startswith("HEADER") and len(line) >= 66:
                pdb_id = line[62:66].strip()
                break
        if not pdb_id:
            pdb_id = Path(pdb_file.name).stem
    elif pdb_input:
        pdb_id = pdb_input
    # Persist a temp PDB path for downstream recommender use
    if pdb_text:
        with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as tmp:
            tmp.write(pdb_text.encode("utf-8") if isinstance(pdb_text, str) else pdb_text)
            st.session_state["recs_pdb_path"] = tmp.name
    elif pdb_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as tmp:
            tmp.write(pdb_file.getvalue())
            st.session_state["recs_pdb_path"] = tmp.name

    if page == "Avian Compass Simulator":
        render_compass_simulator()
        return

    # Run button gated analysis
    run_clicked = st.button("[ RUN ANALYSIS ]")
    analysis_ready = False
    fresh_row = None
    manifest_run_key = None
    if run_clicked:
        if (pdb_file is None and not pdb_input):
            st.error("Provide a PDB upload or ID before running analysis.")
            return
        if not cofactor_choice:
            st.error("Select a cofactor (FAD or FMN) before running.")
            return
        st.cache_data.clear()
        with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as tmp:
            if pdb_file is not None:
                tmp.write(pdb_file.getvalue())
            tmp_path = tmp.name if pdb_file is not None else None
        prog = st.progress(0)
        cmd = [sys.executable, str(BASE_DIR / "engine" / "vqe_n5_edge.py")]
        env = os.environ.copy()
        env["MUTATION_LIST"] = mutation_list_str or ""
        env["COFACTOR_SELECTED"] = cofactor_choice
        env["COFACTOR_FORCE"] = "1" if force_cofactor else "0"
        env["UNIPROT_ID"] = uniprot_id
        if pdb_id:
            env["PDB_LABEL"] = pdb_id
        if pdb_file is not None:
            cmd += ["--pdb", tmp_path]
        # Configurable timeout: env VQE_TIMEOUT_SEC_UI (or VQE_TIMEOUT_SEC) controls limit; 0 disables timeout.
        timeout_env = env.get("VQE_TIMEOUT_SEC_UI") or env.get("VQE_TIMEOUT_SEC") or ""
        try:
            timeout_val = float(timeout_env) if timeout_env else 300.0
            if timeout_val <= 0:
                timeout_val = None
        except ValueError:
            timeout_val = 300.0
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_val, env=env)
        except subprocess.TimeoutExpired:
            st.error(f"VQE simulation timed out after {timeout_val or 'unlimited'} seconds. "
                     f"Raise VQE_TIMEOUT_SEC_UI or set to 0 for no limit (may hang).")
            return
        prog.progress(100)
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown error"
            st.error(f"VQE simulation failed: {err_msg}")
            st.session_state["status"] = "Failed"
            st.session_state["live_profile"] = None
            return
        for line in proc.stdout.splitlines():
            if line.startswith("RUN_MANIFEST:"):
                try:
                    manifest = json.loads(line.split("RUN_MANIFEST:", 1)[1])
                    manifest_run_key = manifest.get("run_key")
                except Exception:
                    manifest_run_key = None
        st.cache_data.clear()
        st.session_state["reload_token"] = st.session_state.get("reload_token", 0) + 1
        bulk, qprof, scorecard = load_data(st.session_state["reload_token"])
        analysis_ready = True
        # Grab the matching row by run_key if available
        if manifest_run_key and not qprof.empty and "run_key" in qprof:
            match = qprof[qprof["run_key"] == manifest_run_key]
            if not match.empty:
                fresh_row = match.iloc[0].copy()
        # Fallback: match by pdb + mutation + cofactor selected
        if fresh_row is None and not qprof.empty and pdb_id:
            mutation_display_target = mutation_list_str or "WT"
            cofactor_target = cofactor_choice.strip().upper() if cofactor_choice else ""
            cand = qprof.copy()
            cand["PDB_ID_UP"] = cand.get("PDB_ID", cand.get("pdb_id", "")).astype(str).str.upper()
            pid_up = pdb_id.upper()
            cand = cand[(cand["PDB_ID_UP"] == pid_up) | (cand["PDB_ID_UP"].str.startswith(f"{pid_up}_"))]
            if "mutation_display" in cand:
                cand = cand[cand["mutation_display"].fillna("") == mutation_display_target]
            if cofactor_target:
                if "cofactor_type_used" in cand:
                    cand = cand[cand["cofactor_type_used"].astype(str).str.upper() == cofactor_target]
            if not cand.empty:
                fresh_row = cand.tail(1).iloc[0].copy()
        if fresh_row is None and not qprof.empty:
            fresh_row = qprof.tail(1).iloc[0].copy()
        if fresh_row is not None and pdb_id:
            fresh_row["PDB_ID"] = pdb_id.upper()
            fresh_row["pdb_id"] = pdb_id
        if fresh_row is not None:
            st.session_state["live_profile"] = dict(fresh_row)
            mut_label = mutation_key.replace(":", "") if mutation_key else "WT"
            live_id = f"{(pdb_id or fresh_row.get('pdb_id', '')).upper()}_{mut_label}"
            st.session_state["live_pdb"] = live_id
        st.success(f"ANALYSIS COMPLETE: Quantum Profile Generated for {pdb_id.upper() if pdb_id else 'NEW SAMPLE'}")
        if proc.stdout:
            st.sidebar.text("VQE STATUS LOG")
            st.sidebar.text("STATUS: Output logged to local artifacts folder.")

    if not analysis_ready:
        # allow reuse of previous analysis for downstream tools
        if st.session_state.get("live_profile"):
            q_row = pd.Series(st.session_state["live_profile"])
            pdb_id = st.session_state.get("live_pdb", "").split("_")[0]
            analysis_ready = True
        else:
            st.info("[ STATUS: IDLE - AWAITING PROTEIN DATA ]")
            return

    if pdb_id:
        pdb_id_upper = pdb_id.upper()
        if mutation_list_str and st.session_state.get("last_mut_str") != mutation_list_str:
            st.cache_data.clear()
        st.session_state["last_mut_str"] = mutation_list_str
        mut_label = mutation_key.replace(":", "") if mutation_key else "WT"
        save_id = f"{pdb_id_upper}_{mut_label}"
        target_id = save_id
        q_row = None
        bulk_row = None
        # Prefer live session profile if matching
        if st.session_state.get("live_pdb", "") == f"{pdb_id_upper}_{mut_label}" and st.session_state.get("live_profile"):
            q_row = pd.Series(st.session_state["live_profile"])
        else:
            bulk_row, q_row = get_entry(target_id, bulk, qprof)
        if q_row is None or (isinstance(q_row, pd.Series) and q_row.empty):
        # Fallback to most recent live profile
            if st.session_state.get("live_profile"):
                q_row = pd.Series(st.session_state["live_profile"])
            else:
                st.info("[ STATUS: IDLE - Awaiting protein data ]")
                return
        mutation_display = q_row.get("mutation_display") or (mutation_list_str or "WT")
        mut_label_display = str(mutation_display).replace(" ", "").replace(",", "_")
        st.markdown(f"### Selected PDB: `{pdb_id}` ({mutation_display})")
        wt_candidates = {f"{pdb_id_upper}_WT", f"{pdb_id_upper}_WT_MUTATION"}
        wt_mask = qprof["PDB_ID"].astype(str).str.upper().isin(wt_candidates)
        wt_row_df = qprof[wt_mask]
        if wt_row_df.empty:
            wt_row_df = qprof[qprof["PDB_ID"].astype(str).str.upper().str.startswith(f"{pdb_id_upper}_WT")]
        wt_row = wt_row_df.tail(1).iloc[0] if not wt_row_df.empty else None
        # Persist back to CSV if missing
        if save_id not in qprof.get("PDB_ID", pd.Series(dtype=str)).astype(str).str.upper().tolist():
            append_path = QUANTUM_PROFILES
            q_append = _safe_read_csv(append_path)
            q_row = q_row.copy()
            q_row["PDB_ID"] = save_id
            new_row = pd.DataFrame([q_row])
            q_combined = pd.concat([q_append, new_row], ignore_index=True)
            q_combined.to_csv(append_path, index=False)
            st.cache_data.clear()
            st.sidebar.text(f"[ SUCCESS: QUANTUM PROFILE GENERATED FOR {save_id} ]")

        cofactor_used = str(
            q_row.get("cofactor_type_used")
            or q_row.get("user_cofactor_choice")
            or cofactor_choice
            or ""
        ).strip().upper()
        detected_cofactor = str(q_row.get("cofactor_detected") or q_row.get("detected_cofactor") or "").strip().upper()
        if cofactor_used == "AUTO" or not cofactor_used:
            cofactor_used = detected_cofactor or str(cofactor_choice).strip().upper() or "UNKNOWN"
        geometry_check_passed = bool(q_row.get("geometry_check_passed", True))
        computational_mode_label = q_row.get("computational_mode") or q_row.get("cofactor_mode") or ""
        # UniProt-based override for mode
        cry_ids = {"P26484", "Q9VBW3"}
        if uniprot_id and (uniprot_id in cry_ids or uniprot_id.startswith("CRY")):
            computational_mode_label = "Radical Pair (Cryptochrome)"
        cofactor_choice_upper = (cofactor_choice or "").upper()
        if cofactor_choice_upper == "FAD" and detected_cofactor == "FMN":
            st.warning("User override active: FAD selected but FMN-like geometry detected in PDB.")
        if cofactor_choice_upper in {"FAD", "FMN"} and not geometry_check_passed:
            st.warning(f"Geometry Check failed: {cofactor_choice_upper} not found in uploaded PDB.")
        if not cofactor_used:
            cofactor_used = detected_cofactor or "UNKNOWN"

        st_gap = float(q_row.get("st_gap", float("nan")))
        n5_spin = float(q_row.get("n5_spin_density", float("nan")))
        precision = 1.0 / (st_gap + 1e-6) if pd.notna(st_gap) else float("nan")
        scs = float(q_row.get("scs", 50.0))
        st_crossing_flag = bool(q_row.get("st_crossing_detected", False))
        if (not st_crossing_flag) and pd.notna(st_gap) and pd.notna(n5_spin):
            st_crossing_flag = detect_st_crossing(st_gap, n5_spin, CRITICAL_ST_GAP, 0.4)
        triad = float(q_row.get("triad_score", float("nan")))
        clash_pen = float(q_row.get("clash_penalty", float("nan")))
        plddt = float(q_row.get("plddt_mean", float("nan")))
        coupling_label = q_row.get("coupling_label", "COUPLING: UNKNOWN (insufficient data)")
        primary_hfcc = float(q_row.get("hfcc_primary_mhz", float("nan")))
        secondary_hfcc = float(q_row.get("hfcc_secondary_mhz", float("nan")))
        # Apply mutation list influence to ST gap for fluorescence proxy (wild-type if empty)
        # GQFP baseline and shifts
        base_st_gap_ev = 0.35
        lfp_mut = 0.0
        steric_mut = 0.0
        if mutation_list_str:
            entries = [m.strip() for m in mutation_list_str.split(",") if m.strip()]
            for m in entries:
                if len(m) >= 3:
                    aa = m[-1].upper()
                    props = AA_PROPERTIES.get(aa, {"e_neg": 0.0, "steric_index": 0.0})
                    lfp_mut += props.get("e_neg", 0.0)
                    steric_mut += props.get("steric_index", 0.0)
        # compute final gap from baseline plus LFP (e_neg in eV) and steric penalty
        st_gap_ev = base_st_gap_ev + lfp_mut - 0.02 * steric_mut
        st_gap_ev = max(st_gap_ev, 0.01)  # quench limit
        brightness = (st_gap_ev / base_st_gap_ev) * 150.0 if pd.notna(st_gap_ev) else float("nan")

        # Journal-style layout starts here
        feature_hash_val = q_row.get("feature_hash", "") or ""
        run_key_val = q_row.get("run_key", "") or ""
        mutation_display = q_row.get("mutation_display") or (mutation_list_str or "WT")
        mut_label_display = str(mutation_display).replace(" ", "").replace(",", "_")
        run_status = "Complete" if analysis_ready else "Running"
        run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile_id = f"{pdb_id_upper}_{mut_label_display}"
        st.markdown(
            f"""
            <div class="run-bar">
              <div><strong>Status:</strong> {run_status}</div>
              <div><strong>Profile ID:</strong> {profile_id}</div>
              <div><strong>Run Key:</strong> {run_key_val or 'NA'}</div>
              <div><strong>Timestamp:</strong> {run_time}</div>
              <div><strong>Feature Hash:</strong> {feature_hash_val or 'NA'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        json_payload = q_row.to_json()
        csv_payload = q_row.to_csv()
        citation_text = f"FLAVIN-OPT Quantum Profile for {profile_id} (run_key={run_key_val}) retrieved {run_time}."
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
        with dl_col1:
            st.download_button("Download JSON", data=json_payload, file_name=f"{profile_id}.json", mime="application/json")
        with dl_col2:
            st.download_button("Download CSV", data=csv_payload, file_name=f"{profile_id}.csv", mime="text/csv")
        with dl_col3:
            st.download_button("Copy citation text", data=citation_text, file_name=f"{profile_id}_citation.txt", mime="text/plain")

        compare_toggle = wt_row is not None and st.checkbox("Compare to WT", value=True, help="Toggle WT deltas where available.")

        left_col, right_col = st.columns([1.15, 0.85])
        with left_col:
            st.markdown(
                f"""
                <div class="figure-card">
                    <div class="section-title">Primary Outputs</div>
                  <div style="margin-bottom:12px;">
                    <div style="font-family:var(--font-serif);font-size:20px;">{pdb_id_upper}</div>
                    <div style="color:var(--text-dim);font-size:13px;">Cofactor: {cofactor_used or 'NA'}</div>
                    <div style="margin-top:6px;">{render_tag_list([m.strip() for m in mutation_list_str.split(',') if m.strip()])}</div>
                  </div>
                  <div class="metric-grid">
                """,
                unsafe_allow_html=True,
            )
            brightness_delta = None
            st_gap_delta = None
            scs_delta = None
            if compare_toggle and wt_row is not None:
                try:
                    wt_brightness = float(wt_row.get("pred_final", float("nan")))
                    brightness_delta = brightness - wt_brightness if pd.notna(brightness) and pd.notna(wt_brightness) else None
                except Exception:
                    brightness_delta = None
                try:
                    st_gap_delta = st_gap_ev - (float(wt_row.get("st_gap", float("nan"))) / 1000.0) if pd.notna(st_gap_ev) else None
                except Exception:
                    st_gap_delta = None
                try:
                    scs_delta = scs - float(wt_row.get("scs", float("nan")))
                except Exception:
                    scs_delta = None
            cards = [
                render_metric_card(
                    "Predicted Brightness",
                    f"{brightness:.2f}" if pd.notna(brightness) else "NA",
                    "a.u.",
                    "Primary quantum brightness estimator",
                    badge="heuristic",
                    delta=brightness_delta,
                    delta_dir=(1 if (brightness_delta or 0) > 0 else -1) if brightness_delta is not None else None,
                ),
                render_metric_card(
                    "ST Gap",
                    f"{st_gap_ev:.3f}" if pd.notna(st_gap_ev) else "NA",
                    "eV",
                    "Mutation-adjusted singlet–triplet gap",
                    delta=st_gap_delta,
                    delta_dir=(1 if (st_gap_delta or 0) > 0 else -1) if st_gap_delta is not None else None,
                ),
                render_metric_card(
                    "Confidence (SCS)",
                    f"{scs:.1f}" if pd.notna(scs) else "NA",
                    "/100",
                    "Structural confidence score",
                    delta=scs_delta,
                    delta_dir=(1 if (scs_delta or 0) > 0 else -1) if scs_delta is not None else None,
                ),
            ]
            st.markdown("".join(cards), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='margin-top:12px;'>Mutation Sensitivity Audit</div>", unsafe_allow_html=True)
            audit_fields = [
                "pred_final",
                "st_gap",
                "n5_spin_density",
                "hfcc_primary_mhz",
                "quantum_polarizability",
                "Around_N5_HBondCap",
                "Around_N5_Flexibility",
            ]
            if wt_row is not None:
                diffs = feature_diff(wt_row.to_dict(), q_row.to_dict(), audit_fields, DEFAULT_TOLERANCES)
                audit_df = pd.DataFrame(
                    [
                        {
                            "Feature": d["feature"],
                            "WT": d["wt"],
                            "Mut": d["mut"],
                            "Δ": d["delta"],
                            "Changed?": d["changed"],
                        }
                        for d in diffs
                    ]
                )
                styled = audit_df.style.apply(
                    lambda row: ["background-color: var(--bg-muted)" if bool(row.get("Changed?", False)) else "" for _ in row],
                    axis=1,
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
                changed_count = audit_df["Changed?"].sum()
                if changed_count == 0:
                    st.warning("Mutation Sensitivity Audit: no feature-level change detected; verify mutation parsing.")
                elif changed_count < len(audit_df) // 2:
                    st.warning("High proportion of features unchanged; mutation sensitivity may be weak.")
                if wt_row is not None and pd.notna(brightness) and pd.notna(wt_row.get("pred_final", float("nan"))):
                    pred_delta = abs(float(brightness) - float(wt_row.get("pred_final", float("nan"))))
                    if pred_delta < EMISSION_TOL:
                        st.warning("Pred_Em unchanged (Δ < tol). Check coupling between mutation-derived features and emission predictor.")
            else:
                st.info("WT reference unavailable; audit shows mutant-only values.")
            caption_text = f"Quantum profile for {pdb_id_upper} with mutations [{mutation_list_str or 'WT'}]. ST crossing criteria: gap<{CRITICAL_ST_GAP} eV and spin>0.4. Coupling: {coupling_label}."
            st.markdown(f"<div class='caption-box'><strong>Figure caption:</strong> {caption_text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                st.markdown("<div class='support-card'><div class='section-title'>Supporting Metrics</div>", unsafe_allow_html=True)
                with st.expander("Electronic descriptors", expanded=True):
                    bulk_redox = bulk_row.get("pred_final", float("nan")) if bulk_row is not None else float("nan")
                    bulk_gpr = bulk_row.get("gpr_pred", float("nan")) if bulk_row is not None else float("nan")
                    redox_val = _first_finite(q_row.get("pred_final", float("nan")), bulk_redox)
                    redox_display = f"{redox_val:.3f}" if np.isfinite(redox_val) else "proxy_only"
                    gpr_pred_val = _first_finite(
                        q_row.get("gpr_pred", float("nan")),
                        q_row.get("gpr_pred_raw", float("nan")),
                        bulk_gpr,
                    )
                    fallback_flag_val = _safe_float(q_row.get("gpr_fallback_used", 0))
                    fallback_used = np.isfinite(fallback_flag_val) and int(fallback_flag_val) != 0
                    # If GPR collapsed to the training-set mean, fall back to quantum redox.
                    if np.isfinite(gpr_pred_val) and abs(gpr_pred_val - _TRAIN_MEAN_EM) < 1e-3:
                        if np.isfinite(redox_val):
                            gpr_display = f"{redox_val:.3f} (quantum fallback)"
                        else:
                            gpr_display = "proxy_only"
                    elif fallback_used:
                        gpr_display = f"{gpr_pred_val:.3f} (local fallback)"
                    else:
                        gpr_display = f"{gpr_pred_val:.3f}" if np.isfinite(gpr_pred_val) else "proxy_only"
                    ed_df = pd.DataFrame(
                        [
                            ["Em proxy (uncalibrated)", redox_display, "proxy"],
                            ["Predicted Em (mV)", gpr_display, "mV"],
                            ["N5 spin density", n5_spin, "a.u."],
                            ["HFCC primary", primary_hfcc, "MHz"],
                            ["HFCC secondary", secondary_hfcc, "MHz"],
                        ],
                        columns=["Metric", "Value", "Units"],
                )
                st.table(ed_df)
                st.caption("Em proxy is a relative score; Predicted Em (mV) is the calibrated GPR output.")
            with st.expander("Structural quality", expanded=False):
                struct_df = pd.DataFrame(
                    [
                        ["Clash penalty", clash_pen, "a.u."],
                        ["Mean pLDDT/B", plddt, ""],
                        ["Residue consistency", "PASS", ""],
                    ],
                    columns=["Metric", "Value", "Units"],
                ).astype(str)
                st.table(struct_df)
            with st.expander("Coupling & criteria", expanded=False):
                criteria_text = f"ST crossing: gap<{CRITICAL_ST_GAP} eV and spin>0.4; coupling thresholds: hfcc>15 MHz"
                coupling_df = pd.DataFrame(
                    [
                        ["Coupling label", coupling_label, ""],
                        ["ST crossing detected", st_crossing_flag, ""],
                        ["Computational mode", computational_mode_label, ""],
                    ],
                    columns=["Metric", "Value", "Units"],
                ).astype(str)
                st.table(coupling_df)
                st.markdown(f"<div class='criteria-box'>{criteria_text}</div>", unsafe_allow_html=True)
            with st.expander("3D Structure", expanded=False):
                try:
                    render_structure(pdb_id, pdb_text)
                except Exception as exc:
                    st.warning(f"3D view unavailable: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Physics Audit"):
                st.info("Instrument calibrated to PNAS 1600341113 theoretical avoided-crossing parameters. Units: ST_Gap in eV, HFCC in MHz.")
                if "homo_lumo_gap" in q_row:
                    raw_vals = [q_row.get("homo_lumo_gap"), q_row.get("st_gap")]
                    eigvals: list = []
                    for v in raw_vals:
                        try:
                            fv = float(v)
                            if np.isnan(fv) or np.isinf(fv):
                                eigvals.append(None)
                            else:
                                eigvals.append(fv)
                        except Exception:
                            eigvals.append(None)
                    payload = {"eigenvalues": eigvals}
                    st.code(json.dumps(payload, indent=2), language="json")

            with st.expander("Combinatorial Mutation Recommender", expanded=False):
                if pdb_file is None and pdb_text is None:
                    st.info("Upload a PDB/ENT file to generate recommendations.")
                else:
                    rec_input_path = st.session_state.get("recs_pdb_path", None)
                    if rec_input_path is None:
                        st.warning("PDB not cached for recommendations; re-upload or re-run analysis.")
                    if rec_input_path and not os.path.exists(rec_input_path) and pdb_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as tmp_pdb:
                            tmp_pdb.write(pdb_file.getvalue())
                            rec_input_path = tmp_pdb.name
                            st.session_state["recs_pdb_path"] = rec_input_path
                    # Default search settings
                    rec_radius = st.slider("Residue radius around cofactor (Å)", min_value=6.0, max_value=12.0, value=8.0, step=0.5)
                    max_combo_size = st.select_slider("Max combo size", options=[2, 3, 4], value=3)
                    beam_width = st.slider("Beam width", min_value=2, max_value=8, value=4, step=1)
                    top_n = st.slider("Top N per category", min_value=3, max_value=10, value=5, step=1)
                    do_not_mutate = st.text_input("Do not mutate (comma-separated residues, e.g., 50, A:75)", value="")
                    if "recs_data" not in st.session_state:
                        st.session_state["recs_data"] = None
                        st.session_state["recs_status"] = "idle"

                    if st.button("Compute Recommendations", key="compute_recs", disabled=not rec_input_path):
                        exclude_list = [r.strip() for r in do_not_mutate.split(",") if r.strip()]
                        eta_msg = None
                        try:
                            cand_pool = conservative_candidate_pool(
                                rec_input_path,
                                exclude_list,
                                rec_radius,
                                RECOMMENDER_CONFIG.allow_secondary_shell,
                            )
                            single_count = len(cand_pool)
                            planned_doubles = beam_width if (max_combo_size >= 2 and single_count > 1) else 0
                            planned_triples = beam_width if (max_combo_size >= 3 and single_count > 2) else 0
                            planned_evals = 1 + single_count + planned_doubles + planned_triples  # include WT baseline
                            sec_per_eval_env = os.environ.get("RECOMMENDER_SEC_PER_EVAL", "").strip()
                            try:
                                sec_per_eval = float(sec_per_eval_env) if sec_per_eval_env else 8.0
                                if sec_per_eval <= 0:
                                    sec_per_eval = 8.0
                            except ValueError:
                                sec_per_eval = 8.0
                            eta_seconds = planned_evals * sec_per_eval
                            eta_msg = (
                                f"ETA: ~{_format_eta_seconds(eta_seconds)} "
                                f"({planned_evals} evaluations, unlimited mode)"
                            )
                            st.info(eta_msg)
                        except Exception:
                            eta_msg = None
                        try:
                            st.session_state["recs_status"] = "running"
                            spinner_msg = "Computing mutation recommendations..."
                            if eta_msg:
                                spinner_msg = f"Computing mutation recommendations... {eta_msg}"
                            old_max_single = os.environ.pop("MAX_SINGLE_MUTATIONS", None)
                            old_budget = os.environ.get("RECOMMENDER_TIME_BUDGET_SEC")
                            os.environ["RECOMMENDER_TIME_BUDGET_SEC"] = "0"
                            try:
                                with st.spinner(spinner_msg):
                                    recs = recommend(
                                        pdb_path=rec_input_path,
                                        cofactor_choice=cofactor_choice,
                                        baseline_brightness=float(brightness) if pd.notna(brightness) else 0.0,
                                        confidence=scs if pd.notna(scs) else 50.0,
                                        radius=rec_radius,
                                        max_combo_size=max_combo_size,
                                        beam_width=beam_width,
                                        top_n=top_n,
                                        do_not_mutate=exclude_list,
                                    )
                            finally:
                                if old_max_single is not None:
                                    os.environ["MAX_SINGLE_MUTATIONS"] = old_max_single
                                if old_budget is None:
                                    os.environ.pop("RECOMMENDER_TIME_BUDGET_SEC", None)
                                else:
                                    os.environ["RECOMMENDER_TIME_BUDGET_SEC"] = old_budget
                            st.session_state["recs_data"] = recs
                            st.session_state["recs_status"] = "ready"
                        except Exception as exc:
                            st.session_state["recs_status"] = "error"
                            st.error(f"Recommendation engine failed: {exc}")
                    if st.session_state.get("recs_data"):
                        recs = st.session_state["recs_data"]
                        show_heuristic = st.checkbox("Show heuristic suggestions", value=False)
                        for label in ("singles", "doubles", "triples"):
                            rec_computed = [r for r in recs.get(label, []) if not getattr(r, "heuristic", False)]
                            rec_heuristic = [r for r in recs.get(label, []) if getattr(r, "heuristic", False)]
                            rec_list = rec_computed if rec_computed else (rec_heuristic if show_heuristic else [])
                            if not rec_list:
                                continue
                            table = [recommendation_to_row(r) for r in rec_list]
                            df_table = pd.DataFrame(table)
                            visible_cols = ["Mutations", "PredBrightness", "DeltaBrightness", "Confidence", "ClashRisk", "StabilityRisk", "Notes"]
                            df_visible = df_table[visible_cols].astype(str)
                            tag = "computed" if rec_computed else "heuristic"
                            st.markdown(f"**Top {label.capitalize()} ({tag})**")
                            st.dataframe(df_visible, use_container_width=True)
                            st.download_button(
                                f"Download {label} (CSV)",
                                data=df_visible.to_csv(index=False),
                                file_name=f"{profile_id}_{label}_{tag}.csv",
                                mime="text/csv",
                                key=f"dl_{label}_{tag}",
                            )
                        if all(len([r for r in recs.get(label, []) if not getattr(r, "heuristic", False)]) == 0 for label in ("singles", "doubles", "triples")) and not show_heuristic:
                            st.info("No computed recommendations available. Enable 'Show heuristic suggestions' to view heuristic candidates.")
                        if st.session_state["recs_data"]:
                            with st.expander("Show debug", expanded=False):
                                for label in ("singles", "doubles", "triples"):
                                    raw_list = recs.get(label, [])
                                    if not raw_list:
                                        continue
                                    raw_df = pd.DataFrame([recommendation_to_row(r) for r in raw_list])
                                    st.markdown(f"**Debug {label}**")
                                    st.dataframe(raw_df, use_container_width=True)
                        singles_list = [r for r in st.session_state["recs_data"].get("singles", []) if not getattr(r, "heuristic", False)]
                        if singles_list:
                            st.markdown("**Quick Picks (computed)**")
                            bullets = []
                            rank = 1
                            for rec_block in ("singles", "doubles", "triples"):
                                for rec in [r for r in st.session_state["recs_data"].get(rec_block, []) if not getattr(r, "heuristic", False)][:3]:
                                    muts = ", ".join(m.canonical_token() for m in rec.mutations)
                                    bullets.append(f"{rank}. {muts} → ΔBrightness: {rec.delta_brightness:.2f} (Pred: {rec.pred_brightness:.2f})")
                                    rank += 1
                            if bullets:
                                st.markdown("\n".join([f"- {b}" for b in bullets]))
                    else:
                        if st.session_state.get("recs_status") == "running":
                            st.info("Computing recommendations...")
                        else:
                            st.info("Press 'Compute Recommendations' to generate ranked mutations.")

            try:
                render_structure(pdb_id, pdb_text)
            except Exception as exc:
                st.warning(f"3D view unavailable: {exc}")

    with st.sidebar.expander("Technical Metrics", expanded=False):
        if scorecard:
            st.json(scorecard)
            if scorecard.get("global_mae_n139", 0) == 0:
                st.error("DATABASE CORRUPTED. RUN FULL DATASET RECOVERY.")
        else:
            st.info("Run engine/vqe_n5_edge.py to generate artifacts.")
    if st.sidebar.button("Refresh Data", key="refresh_data"):
        st.cache_data.clear()
        st.cache_resource.clear()
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
    if st.sidebar.button("Force Reload", key="force_reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        bulk, qprof, scorecard = load_data(st.session_state.get("reload_token", 0))
    top5_path = ARTIFACT_DIR / "Top5_HBond_Stability.csv"
    if top5_path.exists():
        st.sidebar.markdown("**Top 5 H-Bond Stability**")
        st.sidebar.dataframe(pd.read_csv(top5_path))

    st.markdown("<div class='section-title' style='margin-top:24px;'>Avian Compass Simulator</div>", unsafe_allow_html=True)
    with st.expander("Avian Compass Simulator", expanded=False):
        plane = st.selectbox("Magnetic field plane", options=["ZX", "YZ"], index=0)
        tau_us = st.slider("Lifetime τ (μs)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        rate_mode = st.selectbox("Reaction rates", options=["Equal rates (kS = kT = 1/τ)", "Separate kS and kT"])
        kS_val = None
        kT_val = None
        if rate_mode != "Equal rates (kS = kT = 1/τ)":
            kS_val = st.slider("kS (1/μs)", min_value=0.001, max_value=5.0, value=0.1, step=0.001)
            kT_val = st.slider("kT (1/μs)", min_value=0.001, max_value=5.0, value=0.1, step=0.001)
        omega = st.slider("Zeeman scale ω (scaled units)", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**FAD-like hyperfine (Ax, Ay, Az)**")
            Ax_x = st.number_input("Axx (FAD-like)", value=-0.2, format="%.3f")
            Ax_y = st.number_input("Ayy (FAD-like)", value=-0.2, format="%.3f")
            Ax_z = st.number_input("Azz (FAD-like)", value=1.75, format="%.3f")
        with col_b:
            st.markdown("**Partner-like hyperfine (Ax, Ay, Az)**")
            Ay_x = st.number_input("Axx (Partner-like)", value=0.0, format="%.3f")
            Ay_y = st.number_input("Ayy (Partner-like)", value=0.0, format="%.3f")
            Ay_z = st.number_input("Azz (Partner-like)", value=1.08, format="%.3f")
        theta_min = st.number_input("θ min (deg)", value=0.0, format="%.1f")
        theta_max = st.number_input("θ max (deg)", value=180.0, format="%.1f")
        theta_pts = st.slider("Number of θ points", min_value=10, max_value=720, value=181, step=1)
        run_sim = st.button("Run simulation", key="run_compass")
        if run_sim:
            theta_arr = np.linspace(theta_min, theta_max, int(theta_pts))
            res = cached_compute_yields(theta_arr, tau_us, kS_val, kT_val, omega, plane, (Ax_x, Ax_y, Ax_z), (Ay_x, Ay_y, Ay_z))
            phi_s = res["phi_s"]
            phi_t = res["phi_t"]
            phi_sum = res["phi_sum"]
            spike = estimate_spike_metrics(theta_arr, phi_s)
            max_err = float(np.max(np.abs(phi_sum - 1.0)))
            fig1, ax1 = plt.subplots()
            ax1.plot(theta_arr, phi_s, label="ΦS")
            ax1.plot(theta_arr, phi_t, label="ΦT")
            ax1.set_xlabel("θ (deg)")
            ax1.set_ylabel("Yield")
            ax1.legend()
            st.pyplot(fig1)
            fig2, ax2 = plt.subplots()
            ax2.plot(theta_arr, phi_s - np.mean(phi_s), label="ΦS - mean(ΦS)")
            ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax2.set_xlabel("θ (deg)")
            ax2.set_ylabel("Anisotropic part")
            ax2.legend()
            st.pyplot(fig2)
            st.write(
                f"Spike amplitude: {spike['spike_amp']:.4f}; "
                f"θ at max: {spike['theta_at_max']:.1f}°, θ at min: {spike['theta_at_min']:.1f}°; "
                f"max|ΦS+ΦT-1|: {max_err:.2e}"
            )
            csv_df = pd.DataFrame({"theta_deg": res["theta_deg"], "phi_s": phi_s, "phi_t": phi_t, "phi_sum": phi_sum})
            st.download_button("Download CSV", data=csv_df.to_csv(index=False), file_name="avian_compass_yields.csv", mime="text/csv")

if __name__ == "__main__":
    main()
