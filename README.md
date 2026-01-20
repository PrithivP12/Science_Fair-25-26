# OOK: Quantum-Driven Protein Engineering & Brightness Lab

A hybrid quantum-classical pipeline utilizing 16-qubit VQE and Generalized Quantum Field Perturbation (GQFP) to predict protein fluorescence and engineer hyper-bright variants.

## Quick Start
1. git clone https://github.com/PrithivP12/Science_Fair-25-26.git
2. pip install -r requirements.txt
3. streamlit run app.py
4. (Optional) Run the hybrid GPR+VQE pipeline on the bundled dataset: `python3 engine/vqe_n5_edge.py --data data/redox_dataset.csv`

## Troubleshooting
- If VQE returns 0.0, ensure your PDB file contains a valid HETATM record for FMN/FAD.
