# OOK: Quantum-Driven Protein Brightness Lab

OOK is a Quantum-Classical Hybrid simulator for predicting protein fluorescence. It uses a 16-qubit VQE engine to solve the electronic Hamiltonian of Flavin-based proteins.

## Quick Start
1. git clone https://github.com/PrithivP12/Science_Fair-25-26.git
2. cd Science_Fair-25-26
3. pip install -r requirements.txt
4. streamlit run app.py

## Troubleshooting
- If VQE returns 0.0, ensure your PDB file contains a valid HETATM record for FMN/FAD.
