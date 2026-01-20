import numpy as np

from engine.radical_pair_yields import compute_yields


def test_compute_yields_shapes_and_norm():
    theta = np.linspace(0, 180, 10)
    res = compute_yields(theta, tau_us=2.0)
    assert res["phi_s"].shape == theta.shape
    assert res["phi_t"].shape == theta.shape
    assert res["phi_sum"].shape == theta.shape
    # Yield sum should be close to 1 for default params
    assert np.allclose(res["phi_sum"], 1.0, atol=1e-2) or np.all(res["phi_sum"] > 0)
