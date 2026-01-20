import numpy as np
from scipy.integrate import solve_ivp

# Pauli matrices with 1/2 factor (spin-1/2 operators)
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_n(*ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _electron_projectors():
    up = np.array([[1], [0]], dtype=complex)
    dn = np.array([[0], [1]], dtype=complex)
    s_state = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)
    ps_e = s_state @ s_state.conj().T
    pt_e = np.eye(4, dtype=complex) - ps_e
    return ps_e, pt_e


PS_E, PT_E = _electron_projectors()


def build_operators_two_e_two_n():
    # order: e1, e2, n1, n2
    sx1 = kron_n(Sx, I2, I2, I2)
    sy1 = kron_n(Sy, I2, I2, I2)
    sz1 = kron_n(Sz, I2, I2, I2)

    sx2 = kron_n(I2, Sx, I2, I2)
    sy2 = kron_n(I2, Sy, I2, I2)
    sz2 = kron_n(I2, Sz, I2, I2)

    ix1 = kron_n(I2, I2, Sx, I2)
    iy1 = kron_n(I2, I2, Sy, I2)
    iz1 = kron_n(I2, I2, Sz, I2)

    ix2 = kron_n(I2, I2, I2, Sx)
    iy2 = kron_n(I2, I2, I2, Sy)
    iz2 = kron_n(I2, I2, I2, Sz)

    ps = kron_n(PS_E, np.eye(4, dtype=complex))
    pt = kron_n(PT_E, np.eye(4, dtype=complex))

    return {
        "S1": (sx1, sy1, sz1),
        "S2": (sx2, sy2, sz2),
        "I1": (ix1, iy1, iz1),
        "I2": (ix2, iy2, iz2),
        "PS": ps,
        "PT": pt,
        "dim": ps.shape[0],
    }


OPS = build_operators_two_e_two_n()


def _hamiltonian(plane, theta, omega, Ax, Ay):
    sx1, sy1, sz1 = OPS["S1"]
    sx2, sy2, sz2 = OPS["S2"]
    ix1, iy1, iz1 = OPS["I1"]
    ix2, iy2, iz2 = OPS["I2"]

    if plane.upper() == "YZ":
        bhat = np.array([0.0, np.sin(theta), np.cos(theta)])
    else:  # ZX default
        bhat = np.array([np.sin(theta), 0.0, np.cos(theta)])

    # hyperfine (diagonal tensors)
    Hhf = (
        Ax[0] * sx1 @ ix1 + Ax[1] * sy1 @ iy1 + Ax[2] * sz1 @ iz1 +
        Ay[0] * sx2 @ ix2 + Ay[1] * sy2 @ iy2 + Ay[2] * sz2 @ iz2
    )

    # zeeman
    Hzee = omega * (bhat[0] * (sx1 + sx2) + bhat[1] * (sy1 + sy2) + bhat[2] * (sz1 + sz2))

    return Hhf + Hzee


def compute_yields(theta_deg, tau_us=10.0, kS=None, kT=None, omega=0.1, plane="ZX", Ax=(-0.2, -0.2, 1.75), Ay=(0.0, 0.0, 1.08)):
    theta_rad = np.deg2rad(theta_deg)
    if kS is None or kT is None:
        k = 1.0 / max(tau_us, 1e-6)
        kS = k if kS is None else kS
        kT = k if kT is None else kT

    ps = OPS["PS"]
    pt = OPS["PT"]
    dim = OPS["dim"]
    rho0 = ps / np.trace(ps)  # electron singlet
    rho0 = rho0 / np.trace(rho0)
    # include maximally mixed nuclei (already in ps tensor with identity)
    rho0_vec = rho0.flatten()

    results_phi_s = []
    results_phi_t = []
    for th in theta_rad:
        H = _hamiltonian(plane, th, omega, Ax, Ay)
        comm = -1j * (np.kron(np.eye(dim), H) - np.kron(H.T, np.eye(dim)))
        AS = -0.5 * kS * (np.kron(np.eye(dim), ps) + np.kron(ps.T, np.eye(dim)))
        AT = -0.5 * kT * (np.kron(np.eye(dim), pt) + np.kron(pt.T, np.eye(dim)))

        L = comm + AS + AT

        rho_vec = rho0_vec.copy()
        ys = 0.0
        yt = 0.0
        t_end = max(10.0 / max(kS, kT, 1e-6), tau_us * 2)
        n_steps = 400
        dt = t_end / n_steps
        for _ in range(n_steps):
            # RK4
            k1 = L @ rho_vec
            k2 = L @ (rho_vec + 0.5 * dt * k1)
            k3 = L @ (rho_vec + 0.5 * dt * k2)
            k4 = L @ (rho_vec + dt * k3)
            rho_vec = rho_vec + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho_mat = rho_vec.reshape((dim, dim))
            ys += dt * kS * np.trace(ps @ rho_mat).real
            yt += dt * kT * np.trace(pt @ rho_mat).real
        results_phi_s.append(ys)
        results_phi_t.append(yt)

    results_phi_s = np.array(results_phi_s)
    results_phi_t = np.array(results_phi_t)
    return {
        "theta_deg": np.array(theta_deg, dtype=float),
        "phi_s": results_phi_s,
        "phi_t": results_phi_t,
        "phi_sum": results_phi_s + results_phi_t,
    }


def estimate_spike_metrics(theta_deg, phi_s):
    theta_deg = np.array(theta_deg, dtype=float)
    phi_s = np.array(phi_s, dtype=float)
    idx_max = int(np.argmax(phi_s))
    idx_min = int(np.argmin(phi_s))
    spike_amp = float(phi_s[idx_max] - phi_s[idx_min])
    return {
        "spike_amp": spike_amp,
        "theta_at_max": float(theta_deg[idx_max]) if theta_deg.size else float("nan"),
        "theta_at_min": float(theta_deg[idx_min]) if theta_deg.size else float("nan"),
    }
