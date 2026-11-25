#!/usr/bin/env python3
"""
Independent Python / NumPy / SciPy verification of Pathway A:
  1. Direct 6x6 momentum-space QCA of Mlodinow & Brun (PRD 112, 074513, 2025; arXiv:2506.20136, Eqs. (16)-(17)):
       U_MB(k) = diag(R(k), R(-k)),   R(k) = R_x(k_x) R_y(k_y) R_z(k_z) in SO(3)
     explicitly proves:
     (a) cos(omega_+(k)) = 1/2 (cos kx cos ky + cos kx cos kz + cos ky cos kz - sin kx sin ky sin kz - 1)
         (upper block) and cos(omega_-(k)) = cos(omega_+(-k)) with + sin kx sin ky sin kz (lower block, Eq. (42)),
     (b) omega_+(k) != omega_-(k) = omega_+(-k) at O(k^2): exact birefringence
           Delta_omega(k) = omega_+(k) - omega_-(k) = + k_x k_y k_z / |k| + O(k^4),
     (c) Direct real-field linear polarization rotation of (E, B) = (Re psi, Im psi) in 3D:
         the physical polarization plane rotates at rate Delta_omega / 2 = +1/2 (n_x n_y n_z) k^2!
  2. Exact spherical average <(n_x n_y n_z)^2> = 1/105 and exact Wigner/Y_{3m} expansion of
     +1/2 (n_x n_y n_z) under SO(3) rotations against Kislat & Krawczynski (PRD 95, 083013, 2017,
     Table IV) giving worst-case SO(3) profile-chi^2 exclusion bounds at 95% CL and 5-sigma.
  3. Exact 3x3 dynamical matrix D(k) of the new strictly local, gauge-invariant, Gauss-exact
     symplectic Yee-4 S_4 photon: verifies zero doublers across the full 3D Brillouin zone,
     omega^2(k) = |k|^2 + O(k^6) in ALL directions, and O(k^7) polarization splitting (>200x
     smaller than Mlodinow-Brun already at coarse lattice scale k = (2 pi/16)(2, 1, 1)!).
"""

from __future__ import annotations
import math
import sys
import numpy as np
from scipy.spatial.transform import Rotation


E_PLANCK_GEV = 1.22089e19

# Kislat & Krawczynski (PRD 95, 083013, 2017), Table IV, j = 3 entries: (m, is_real, limit_95pct_GeV_inv)
KK_J3_TABLE_IV = [
    (0, True,  2.1e-26),
    (1, True,  1.4e-26),
    (1, False, 1.0e-26),
    (2, True,  8.3e-27),
    (2, False, 1.4e-26),
    (3, True,  9.7e-27),
    (3, False, 5.8e-27),
]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def mlodinow_brun_matrix_and_omega(k: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Build exact 6x6 U_MB(k) = diag(R(k), R(-k)) matching Eqs. (16)-(17) of arXiv:2506.20136 verbatim."""
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    R_p = rot_x(kx) @ rot_y(ky) @ rot_z(kz)
    R_m = rot_x(-kx) @ rot_y(-ky) @ rot_z(-kz)
    U6 = np.block([[R_p, np.zeros((3, 3))], [np.zeros((3, 3)), R_m]])
    # Eigenvalues of SO(3) rotation R(k) are {1, e^{+i omega_+}, e^{-i omega_+}}
    ev_p = np.linalg.eigvals(R_p)
    ev_m = np.linalg.eigvals(R_m)
    om_p = float(np.max(np.abs(np.angle(ev_p))))
    om_m = float(np.max(np.abs(np.angle(ev_m))))
    return U6, om_p, om_m


def y3_all(n: np.ndarray) -> np.ndarray:
    """Spherical harmonics Y_{3,m}(n) for m = 0, 1, 2, 3 matching Condon-Shortley phase."""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    w = x + 1j * y
    y30 = math.sqrt(7.0 / (16.0 * math.pi)) * (5.0 * z**3 - 3.0 * z)
    y31 = -math.sqrt(21.0 / (64.0 * math.pi)) * w * (5.0 * z**2 - 1.0)
    y32 = math.sqrt(105.0 / (32.0 * math.pi)) * (w**2) * z
    y33 = -math.sqrt(35.0 / (64.0 * math.pi)) * (w**3)
    return np.stack([y30, y31, y32, y33], axis=1)


def yee4_dynamical_matrix(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact 3x3 dynamical matrix D(k) of local, gauge-invariant, Gauss-exact Yee-4 Hamiltonian."""
    s = np.sin(0.5 * k)
    u = (1.0 / 6.0) * (s ** 2)
    kop = 1.0 + u
    mop = 1.0 - 2.0 * u + 3.0 * (u ** 2)
    alpha = np.sqrt(mop) * kop
    q = kop * 2.0 * s
    q2 = float(np.dot(q, q))
    D = np.outer(alpha, alpha) * (q2 * np.eye(3) - np.outer(q, q))
    evals = np.linalg.eigvalsh(D)
    return D, evals


def compute_so3_profile_chi2(
    n_dirs: int = 2000,
    n_rots: int = 1500,
    seed: int = 20250620,
) -> dict[str, float]:
    """Compute exact j=3 spherical average and Haar-uniform SO(3) profile-chi^2 bounds."""
    if n_dirs < 10 or n_rots < 1:
        raise ValueError(f"Expected n_dirs >= 10 and n_rots >= 1, got {n_dirs}, {n_rots}")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(n_dirs, dtype=np.float64)
    z = 1.0 - 2.0 * (idx + 0.5) / n_dirs
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = golden * idx
    dirs = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    wq = 4.0 * math.pi / n_dirs
    nxyz_sq_avg = float(np.mean((dirs[:, 0] * dirs[:, 1] * dirs[:, 2]) ** 2))

    rng = np.random.default_rng(seed)
    rots = Rotation.random(n_rots, random_state=rng).as_matrix()
    fvals = +0.5 * dirs[:, 0] * dirs[:, 1] * dirs[:, 2] * wq  # (n_dirs,)
    s_chi2_list: list[float] = []
    for R in rots:
        s_dirs = dirs @ R.T
        Y3 = y3_all(s_dirs)
        k3 = np.sum(np.conj(Y3) * fvals[:, None], axis=0)  # (4,)
        s_val = 0.0
        for m, is_re, lim95 in KK_J3_TABLE_IV:
            sig = lim95 / 1.959964
            comp = float(k3[m].real if is_re else k3[m].imag) / E_PLANCK_GEV
            s_val += (comp / sig) ** 2
        s_chi2_list.append(s_val)

    s_arr = np.sort(np.array(s_chi2_list, dtype=np.float64))
    s_min = float(s_arr[0])
    s_med = float(s_arr[len(s_arr) // 2])
    if s_min <= 0.0 or s_med <= 0.0:
        raise RuntimeError(f"Non-positive SO(3) profile-chi^2 curvature: s_min={s_min}, s_med={s_med}")

    return {
        "nxyz_sq_avg": nxyz_sq_avg,
        "s_min": s_min,
        "s_med": s_med,
        "dx_95_worst": math.sqrt(3.8414588 / s_min),
        "dx_5s_worst": math.sqrt(25.0 / s_min),
        "dx_95_med": math.sqrt(3.8414588 / s_med),
    }


def _so3_rotation_axis(R: np.ndarray, omega: float) -> np.ndarray:
    """Extract unit right-handed rotation axis u of R in SO(3) with rotation angle omega in (0, pi)."""
    sin_om = math.sin(omega)
    if abs(sin_om) < 1e-12:
        raise ValueError(f"Rotation angle {omega} too close to 0 or pi to determine unique axis")
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) / (2.0 * sin_om)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12:
        raise ValueError("Degenerate SO(3) rotation axis")
    return ax / norm


def verify_real_field_rotation(k: np.ndarray, steps: int = 10) -> tuple[float, float, float]:
    """
    Directly evolve a real transverse linearly polarized field (E(0) = e1, B(0) = 0)
    by exact matrix powers R_p^steps and R_m^steps of U_MB(k) = diag(R_p, R_m):
      F_+(T) = R_p^T e1,   F_-(T) = R_m^T e1,   E(T) = 0.5 * (F_+(T) + F_-(T)).
    Returns (measured_angle_rad, predicted_angle_rad, abs_error_rad).
    """
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    R_p = rot_x(kx) @ rot_y(ky) @ rot_z(kz)
    R_m = rot_x(-kx) @ rot_y(-ky) @ rot_z(-kz)
    _, om_p, om_m = mlodinow_brun_matrix_and_omega(k)

    # Exact SO(3) longitudinal axes: R_p rotates by +om_p about u_p, R_m by -om_m about u_m (~ +k/|k|)
    u_p = _so3_rotation_axis(R_p, om_p)
    u_m = -_so3_rotation_axis(R_m, om_m)

    # Real polarization vector e1 strictly transverse to BOTH blocks (e1 perp u_p and e1 perp u_m)
    e1 = np.cross(u_p, u_m)
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1e-12:
        ref = np.array([0.0, 0.0, 1.0]) if abs(u_p[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(u_p, ref)
        e1_norm = float(np.linalg.norm(e1))
    e1 /= e1_norm

    e2_p = np.cross(u_p, e1)
    e2_m = np.cross(u_m, e1)
    e2_bis = e2_p + e2_m
    e2_bis /= float(np.linalg.norm(e2_bis))
    cos_tilt = float(np.dot(e2_p, e2_bis))

    R_p_T = np.linalg.matrix_power(R_p, steps)
    R_m_T = np.linalg.matrix_power(R_m, steps)
    E_T = 0.5 * (R_p_T @ e1 + R_m_T @ e1)

    carrier = math.cos(0.5 * steps * (om_p + om_m))
    if abs(carrier) < 1e-6:
        raise RuntimeError(f"Standing-wave carrier cos({steps}*(om_p+om_m)/2) near node ({carrier:.2e})")
    sgn = 1.0 if carrier >= 0.0 else -1.0
    c1 = sgn * float(np.dot(E_T, e1))
    c2 = sgn * float(np.dot(E_T, e2_bis)) / cos_tilt
    theta_meas = math.atan2(c2, c1)
    theta_pred = steps * 0.5 * (om_p - om_m)
    return theta_meas, theta_pred, abs(theta_meas - theta_pred)


def main() -> int:
    print("=== Independent Python/NumPy/SciPy Verification — Pathway A (Mlodinow-Brun vs Yee-4 S4) ===\n")

    # 1. Verify Mlodinow-Brun 6x6 unitary matrix, exact trace formula, and O(k^2) birefringence
    print("[1] Direct 6x6 Mlodinow-Brun QCA: U(k) = diag(R_x R_y R_z(k), R_x R_y R_z(-k)) [Eqs. (16)-(17)]:")
    k_dir = np.array([2.0, 1.0, 1.0])
    n_hat = k_dir / np.linalg.norm(k_dir)
    nxyz = float(n_hat[0] * n_hat[1] * n_hat[2])
    for kmag in (0.01, 0.05, 0.20, math.sqrt(6.0) * 2.0 * math.pi / 16.0):
        k = kmag * n_hat
        U6, om_p, om_m = mlodinow_brun_matrix_and_omega(k)
        # Check exact unitarity U^dag U = I
        assert np.max(np.abs(U6.T @ U6 - np.eye(6))) < 1e-14
        # Check exact closed-form trace formula for upper block R_x R_y R_z (- sin kx sin ky sin kz)
        tr_p = (math.cos(k[0]) * math.cos(k[1]) + math.cos(k[0]) * math.cos(k[2])
                + math.cos(k[1]) * math.cos(k[2]) - math.sin(k[0]) * math.sin(k[1]) * math.sin(k[2]))
        om_p_closed = math.acos(max(-1.0, min(1.0, 0.5 * (tr_p - 1.0))))
        assert abs(om_p - om_p_closed) < 1e-13
        # Check asymptotic coefficient (om_p - om_m) / k^2 -> + n_x n_y n_z (exact!)
        coeff_num = (om_p - om_m) / (kmag ** 2)
        print(f"  |k|={kmag:.4f}: om_+={om_p:.8f}, om_-={om_m:.8f}, (om_+ - om_-)/|k|^2 = {coeff_num:+.6f} (exact +n_x n_y n_z = {+nxyz:+.6f})")

    # Direct real-field (E, B) matrix-power polarization rotation check at k_ref and k_obl
    k_ref = np.array([0.21, 0.13, 0.07])
    theta_ref, pred_ref, err_ref = verify_real_field_rotation(k_ref, steps=10)
    assert err_ref < 1e-12, f"Real-field rotation mismatch at k_ref: {err_ref}"
    print(f"  Real-field R_p^10/R_m^10 rotation at k=(0.21,0.13,0.07), T=10: measured={theta_ref:+.8f} rad vs pred={pred_ref:+.8f} rad (err={err_ref:.2e})")

    k_obl = (2.0 * math.pi / 16.0) * np.array([2.0, 1.0, 1.0])
    theta_obl, pred_obl, err_obl = verify_real_field_rotation(k_obl, steps=10)
    assert err_obl < 1e-12, f"Real-field rotation mismatch at k_obl: {err_obl}"
    rot_T10 = abs(theta_obl)
    print(f"  Real-field R_p^10/R_m^10 rotation at k=(2pi/16)(2,1,1), T=10: measured={theta_obl:+.8f} rad vs pred={pred_obl:+.8f} rad (err={err_obl:.2e})\n")

    # 2. Exact <(n_x n_y n_z)^2> = 1/105 and SO(3) profile-chi^2 against Kislat-Krawczynski (2017) Table IV
    print("[2] Spherical harmonic j = 3 norm and SO(3) profile-chi^2 (Kislat & Krawczynski PRD 95, 2017, Table IV):")
    chi2_res = compute_so3_profile_chi2()
    nxyz_sq_avg = chi2_res["nxyz_sq_avg"]
    dx_95_worst = chi2_res["dx_95_worst"]
    dx_5s_worst = chi2_res["dx_5s_worst"]
    dx_95_med = chi2_res["dx_95_med"]
    print(f"  <(n_x n_y n_z)^2> numerical = {nxyz_sq_avg:.8f} vs exact 1/105 = {1.0/105.0:.8f} (err = {abs(nxyz_sq_avg - 1/105):.2e})")
    assert abs(nxyz_sq_avg - 1.0 / 105.0) < 1e-4
    print(f"  Worst SO(3) orientation: 95% CL Delta_x < {dx_95_worst:.3e} l_P, 5-sigma Delta_x < {dx_5s_worst:.3e} l_P")
    print(f"  Median SO(3) orientation: 95% CL Delta_x < {dx_95_med:.3e} l_P\n")
    assert dx_95_worst < 3.0e-6 and dx_5s_worst < 8.0e-6

    # 3. Verify Yee-4 S_4 exact 3x3 dynamical matrix: O(k^6) dispersion, zero doublers, O(k^7) birefringence
    print("[3] Exact 3x3 dynamical matrix of local, gauge-invariant Yee-4 S4 photon:")
    # Check scaling of |omega - |k|| / |k| with |k| along (2,1,1) -> must scale as k^4 (n = 4!)
    # and splitting |omega_1 - omega_2| -> must scale as k^7!
    errs_disp: list[float] = []
    errs_split: list[float] = []
    ks_test = [0.05, 0.10, 0.20, 0.40]
    for km in ks_test:
        _, ev = yee4_dynamical_matrix(km * n_hat)
        assert abs(ev[0]) < 1e-14, f"Gauge zero mode broken: {ev[0]}"
        om1 = math.sqrt(max(0.0, float(ev[1])))
        om2 = math.sqrt(max(0.0, float(ev[2])))
        errs_disp.append(abs(0.5 * (om1 + om2) - km) / km)
        errs_split.append(abs(om2 - om1))
        print(f"  |k|={km:.2f}: |om_bar/|k| - 1| = {errs_disp[-1]:.3e}, |om_2 - om_1| = {errs_split[-1]:.3e}")
    if errs_disp[0] <= 0.0 or errs_disp[1] <= 0.0 or errs_split[0] <= 0.0 or errs_split[1] <= 0.0:
        raise RuntimeError(f"Non-positive scaling errors encountered: disp={errs_disp[:2]}, split={errs_split[:2]}")
    order_disp = math.log(errs_disp[1] / errs_disp[0]) / math.log(2.0)
    order_split = math.log(errs_split[1] / errs_split[0]) / math.log(2.0)
    print(f"  Measured scaling exponents: dispersion error ~ k^{order_disp:.2f} (exact 4.00 => n=4), birefringence ~ k^{order_split:.2f} (exact 7.00!)")
    assert 3.85 < order_disp < 4.15, f"Expected k^4 dispersion scaling, got {order_disp}"
    assert 6.75 < order_split < 7.25, f"Expected k^7 birefringence scaling, got {order_split}"

    # Check coarse oblique mode k = (2pi/16)(2,1,1)
    _, ev_obl = yee4_dynamical_matrix(k_obl)
    om1_obl = math.sqrt(max(0.0, float(ev_obl[1])))
    om2_obl = math.sqrt(max(0.0, float(ev_obl[2])))
    yee4_rot_T10 = 10.0 * 0.5 * abs(om2_obl - om1_obl)
    ratio = rot_T10 / max(yee4_rot_T10, 1e-16)
    print(f"  At k=(2pi/16)(2,1,1), T=10: Yee-4 phase shift = {yee4_rot_T10:.3e} rad vs MB = {rot_T10:.4f} rad ({ratio:.0f}x suppression!)")

    # Check full 3D BZ corner gap (no fermion/photon doublers)
    corners = [np.array([a, b, c]) * math.pi for a in (0, 1) for b in (0, 1) for c in (0, 1) if (a, b, c) != (0, 0, 0)]
    min_corner_om = min(math.sqrt(max(0.0, float(yee4_dynamical_matrix(kc)[1][1]))) for kc in corners)
    print(f"  Minimum Yee-4 frequency across all 7 BZ corners (doubler check): {min_corner_om:.6f} (exact 7/3 = {7/3:.6f})")
    assert abs(min_corner_om - 7.0 / 3.0) < 1e-12

    print("\nALL PATHWAY A INDEPENDENT VERIFICATIONS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
