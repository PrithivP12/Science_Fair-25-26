# OOK: Quantum-Driven Protein Engineering & Brightness Lab

A hybrid quantum-classical pipeline utilizing 16-qubit VQE and Generalized Quantum Field Perturbation (GQFP) to predict protein fluorescence and engineer hyper-bright variants.

## Quick Start
1. git clone https://github.com/PrithivP12/Science_Fair-25-26.git
2. pip install -r requirements.txt
3. python3 -m streamlit run app.py
4. (Optional) Run the hybrid GPR+VQE pipeline on the bundled dataset: `python3 engine/vqe_n5_edge.py --data data/redox_dataset.csv`

## Troubleshooting
- If VQE returns 0.0, ensure your PDB file contains a valid HETATM record for FMN/FAD.

## UI / Reporting Notes
- Design tokens live in `design_tokens.css` (colors, typography, spacing). Add new tokens there, then reference via CSS variables in `app.py`.
- To add a new metric to the dashboard, populate it in `engine/vqe_n5_edge.py` outputs and surface it in the Primary/Supporting sections in `app.py`.
- Mark metrics as fixed/heuristic by adding a badge in the metric card text (e.g., “(fixed reference)” or “(heuristic)” next to the label).
- WT vs mutant comparisons are driven by the canonical mutation key; ensure the WT profile is present to enable deltas. The “Compare to WT” toggle controls delta visibility.
- Redox proxy (formerly Hybrid_Pred_Em) is shown as an uncalibrated score; calibrated midpoint potential comes from GPR_Pred_Em (mV).
- The Avian Compass Simulator is a toy radical-pair model (spin dynamics only) and does not predict redox potentials.
