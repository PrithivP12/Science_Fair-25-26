import json
import numpy as np


def build_payload(raw_vals):
    eigvals = []
    for v in raw_vals:
        try:
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                eigvals.append(None)
            else:
                eigvals.append(fv)
        except Exception:
            eigvals.append(None)
    return {"eigenvalues": eigvals}


def test_eigen_payload_is_json_serializable():
    payload = build_payload([0.05927, "0.05927", np.nan, np.inf, "bad"])
    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["eigenvalues"][0] == 0.05927
    assert loaded["eigenvalues"][1] == 0.05927
    assert loaded["eigenvalues"][2] is None
    assert loaded["eigenvalues"][3] is None
    assert loaded["eigenvalues"][4] is None
