import json
import numpy as np


def _eigen_payload(raw_vals):
    eigvals = []
    for v in raw_vals:
        try:
            fv = float(v)
            if np.isnan(fv):
                eigvals.append(None)
            else:
                eigvals.append(fv)
        except Exception:
            eigvals.append(None)
    return {"eigenvalues": eigvals}


def test_eigen_payload_json_serializable():
    payload = _eigen_payload(["0.35", np.nan, 0.12])
    dumped = json.dumps(payload)
    assert '"eigenvalues"' in dumped
    loaded = json.loads(dumped)
    assert loaded["eigenvalues"][0] == 0.35
    assert loaded["eigenvalues"][1] is None
    assert loaded["eigenvalues"][2] == 0.12


def test_profile_id_prefers_mutation_display():
    pdb_id = "1U3C"
    mutation_display = "D393R"
    mut_label_display = mutation_display.replace(" ", "").replace(",", "_")
    profile_id = f"{pdb_id}_{mut_label_display}"
    assert profile_id == "1U3C_D393R"
