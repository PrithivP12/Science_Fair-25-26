# Science_Fair-25-26

Hybrid quantum-classical inference and mutation ranking for flavin systems (FAD/FMN), with Streamlit visualization and CSV artifact generation.

## Scope
- Single-structure inference from `.pdb` or `.ent` input.
- Batch processing from tabular datasets (CSV/Excel).
- Cofactor-aware feature extraction around flavin environments.
- Midpoint potential (`Em`) prediction with GPR-based calibration and quantum-derived descriptors.
- Mutation recommendation (singles/doubles/triples) with computed and heuristic fallback modes.

## Repository Layout
- `app.py`: Streamlit interface, run orchestration, reporting panels, recommendation UI.
- `engine/vqe_n5_edge.py`: core inference pipeline, feature extraction, GPR/quantum integration, artifact writes.
- `engine/recommender.py`: candidate generation, scoring, beam search, per-mutation pipeline execution.
- `engine/mutator.py`: structure mutation utilities used by recommender runs.
- `tests/`: unit tests for inference and recommendation behavior.
- `artifacts/`: generated outputs (ignored by Git).

## Environment
- Python 3.9+ recommended.
- macOS/Linux shell environment tested.

## Setup
```bash
git clone https://github.com/PrithivP12/Science_Fair-25-26.git
cd Science_Fair-25-26
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution
### Streamlit
```bash
python3 -m streamlit run app.py
```

### Single-file CLI inference
```bash
COFACTOR_SELECTED=FAD python3 engine/vqe_n5_edge.py --pdb /absolute/path/to/input.pdb
```

### Dataset/batch run
```bash
python3 engine/vqe_n5_edge.py --data data/redox_dataset.csv
```

## Input/Inference Notes
- Supported upload formats: `.pdb`, `.ent`.
- Cofactor mode: `FAD` or `FMN` (explicit selection in UI; CLI auto-detection if present).
- Multi-cofactor structures: representative local flavin site is selected deterministically before neighborhood feature extraction.
- Single-file prediction path supports UniProt DBREF extraction for prior anchoring when available.

## Recommender Runtime Controls
- `MAX_SINGLE_MUTATIONS`: optional hard cap for evaluated single mutations. Unset/`0` => evaluate all candidates.
- `RECOMMENDER_TIME_BUDGET_SEC`: optional global time budget. Unset/`0` => unlimited.
- `RECOMMENDER_SEC_PER_EVAL`: ETA scaling factor for UI/log estimate (default `8.0` seconds).
- `RECOMMENDER_VQE_TIMEOUT_SEC`: per-evaluation subprocess timeout (seconds).

## Testing
```bash
PYTHONPATH=. .venv/bin/pytest -q
```

## Output Files
- Main profile output: `artifacts/qc_n5_gpr/Final_Quantum_Profiles.csv` (local runtime artifact).
- Batch summaries and plots: `artifacts/qc_n5_gpr/*`.
- Prediction dataset append target: `data/prediction_redox_dataset.csv`.

## Troubleshooting
- `Cofactor ... not found`: verify `HETATM` records include `FAD`/`FMN`, or enable forced cofactor mode.
- Constant/mean-like Em outputs: verify flavin atoms are present and input structure corresponds to the intended chain/site.
- Stale UI values after changes: restart Streamlit process and rerun analysis.
