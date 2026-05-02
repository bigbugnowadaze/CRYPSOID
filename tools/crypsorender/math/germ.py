"""Phoxoidal germ fitting and action evaluation (Tier 2).

Germ basis (5 coefficients, Pearcey-class):

    H(s, t) = k1 * s^2
            + k2 * t^2
            + chi   * (s^3 - 3 s t^2)
            + omega * (3 s^2 t - t^3)
            + zeta  * (s^4 + t^4)
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ---------- germ basis ----------

def germ_basis(s, t):
    s2 = s * s; t2 = t * t
    return np.stack([
        s2, t2,
        s * (s2 - 3.0 * t2),
        t * (3.0 * s2 - t2),
        s2 * s2 + t2 * t2,
    ], axis=-1)


def germ_eval(theta, s, t):
    return (germ_basis(s, t) * theta).sum(axis=-1)


def germ_grad(theta, s, t):
    k1, k2, chi, omg, zeta = (theta[..., i] for i in range(5))
    dHs = 2 * k1 * s + 3 * chi * (s * s - t * t) + 6 * omg * s * t + 4 * zeta * (s ** 3)
    dHt = 2 * k2 * t - 6 * chi * s * t + 3 * omg * (s * s - t * t) + 4 * zeta * (t ** 3)
    return dHs, dHt


def germ_hess(theta, s, t):
    k1, k2, chi, omg, zeta = (theta[..., i] for i in range(5))
    Hss = 2 * k1 + 6 * chi * s + 6 * omg * t + 12 * zeta * (s * s)
    Htt = 2 * k2 - 6 * chi * s - 6 * omg * t + 12 * zeta * (t * t)
    Hst = -6 * chi * t + 6 * omg * s
    return Hss, Htt, Hst


# ---------- Newton solver ----------

def closest_point_on_germ(theta, sigma, u, n_iter=3, damping=0.85, lambda_support=100.0):
    """Vectorized 2D Newton minimizer for the phoxoidal action.

    Minimizes:
        F(s,t) = (a-s)^2/sa^2 + (b-t)^2/sb^2 + (n-H(s,t))^2/sn^2
                 + lambda_support * (s^2 + t^2)^2

    The support gate (last term) prevents the chart from extending infinitely.
    With lambda_support -> infinity, the phoxoid reduces to the standard 3D
    anisotropic Gaussian.  Default lambda=100 is "near-Gaussian".

    Returns (s_star, t_star, action) -- each shape (M,) float32.
    """
    a = u[:, 0]; b = u[:, 1]; n = u[:, 2]
    sa = sigma[:, 0]; sb = sigma[:, 1]; sn = sigma[:, 2]
    sa2 = sa * sa; sb2 = sb * sb; sn2 = sn * sn
    lam = float(lambda_support)
    s = a.copy().astype(np.float64)
    t = b.copy().astype(np.float64)
    for _ in range(n_iter):
        H = germ_eval(theta, s, t)
        Hs, Ht = germ_grad(theta, s, t)
        Hss, Htt, Hst = germ_hess(theta, s, t)
        diff_n = n - H
        r2 = s * s + t * t
        gs = -2.0 * (a - s) / sa2 - 2.0 * diff_n / sn2 * Hs + 4.0 * lam * s * r2
        gt = -2.0 * (b - t) / sb2 - 2.0 * diff_n / sn2 * Ht + 4.0 * lam * t * r2
        Jss = 2.0 / sa2 + 2.0 / sn2 * (Hs * Hs - diff_n * Hss) + 4.0 * lam * (3.0 * s * s + t * t) + 1e-6
        Jtt = 2.0 / sb2 + 2.0 / sn2 * (Ht * Ht - diff_n * Htt) + 4.0 * lam * (3.0 * t * t + s * s) + 1e-6
        Jst = 2.0 / sn2 * (Hs * Ht - diff_n * Hst) + 8.0 * lam * s * t
        det = Jss * Jtt - Jst * Jst
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        delta_s = (Jtt * gs - Jst * gt) / det
        delta_t = (Jss * gt - Jst * gs) / det
        s = s - damping * delta_s
        t = t - damping * delta_t
    H_final = germ_eval(theta, s, t)
    r2_final = s * s + t * t
    action = (((a - s) ** 2) / sa2 + ((b - t) ** 2) / sb2
              + ((n - H_final) ** 2) / sn2
              + lam * r2_final * r2_final)
    return s.astype(np.float32), t.astype(np.float32), action.astype(np.float32)


def phoxoidal_action_at_u(theta, sigma, u, n_iter=3, lambda_support=100.0):
    _, _, action = closest_point_on_germ(theta, sigma, u, n_iter=n_iter, lambda_support=lambda_support)
    return action


# ---------- 5-coef germ fitter ----------

def fit_synthetic_germs_5(xyz, quats, scales, tier=None, k=16,
                          max_kappa=25.0, max_chi_omega=50.0, max_zeta=100.0,
                          normalize_to_sigma_units=True):
    """Returns (n, 5) float32 germs (k1, k2, chi, omega, zeta).  Tier C kept zero."""
    from .quat import quat_to_rot_matrix
    n = xyz.shape[0]
    germs = np.zeros((n, 5), dtype=np.float32)
    if tier is None:
        active = np.ones(n, dtype=bool)
    else:
        active = tier != 2
    if not active.any() or cKDTree is None:
        return germs
    print(f"  fitting 5-coef germs for {int(active.sum())}/{n} splats ...", flush=True)
    tree = cKDTree(xyz)
    active_idx = np.where(active)[0]
    rot_all = quat_to_rot_matrix(quats)
    sigma_world = np.exp(scales).astype(np.float64)

    chunk = 4096
    for start in range(0, len(active_idx), chunk):
        end = min(start + chunk, len(active_idx))
        ids = active_idx[start:end]
        kk = min(k + 1, n)
        _, neigh = tree.query(xyz[ids], k=kk)
        if neigh.ndim == 1:
            neigh = neigh[:, None]
        neigh = neigh[:, 1:]
        R_loc = rot_all[ids]
        offsets = xyz[neigh] - xyz[ids][:, None, :]
        local = np.einsum("nij,nkj->nki", R_loc.transpose(0, 2, 1), offsets)
        s = local[..., 0]; t = local[..., 1]; nrm = local[..., 2]
        if normalize_to_sigma_units:
            sa = sigma_world[ids, 0:1]
            sb = sigma_world[ids, 1:2]
            sn = sigma_world[ids, 2:3]
            s = s / np.maximum(sa, 1e-9)
            t = t / np.maximum(sb, 1e-9)
            nrm = nrm / np.maximum(sn, 1e-9)
        s2, t2 = s * s, t * t
        M = np.stack([
            s2, t2,
            s * (s2 - 3.0 * t2),
            t * (3.0 * s2 - t2),
            s2 * s2 + t2 * t2,
        ], axis=-1).astype(np.float64)
        b = nrm.astype(np.float64)
        MtM = np.einsum("nki,nkj->nij", M, M)
        Mtb = np.einsum("nki,nk->ni", M, b)
        ridge = 1e-6 * np.eye(5)
        MtM = MtM + ridge[None, :, :]
        try:
            # np.linalg.solve broadcasts (chunk, 5, 5) @ (chunk, 5, 1) cleanly;
            # passing (chunk, 5) for b can be misparsed in some numpy versions.
            theta = np.linalg.solve(MtM, Mtb[..., None])[..., 0]
        except np.linalg.LinAlgError:
            theta = np.zeros((end - start, 5))
        theta[:, 0] = np.clip(theta[:, 0], -max_kappa, max_kappa)
        theta[:, 1] = np.clip(theta[:, 1], -max_kappa, max_kappa)
        theta[:, 2] = np.clip(theta[:, 2], -max_chi_omega, max_chi_omega)
        theta[:, 3] = np.clip(theta[:, 3], -max_chi_omega, max_chi_omega)
        theta[:, 4] = np.clip(theta[:, 4], -max_zeta, max_zeta)
        germs[ids] = theta.astype(np.float32)
    return germs


def fit_synthetic_germs(*args, **kwargs):
    """DEPRECATED Tier 1 alias.  Returns (n, 2) by truncating Tier 2 fit."""
    full = fit_synthetic_germs_5(*args, **kwargs)
    return full[:, :2].copy()


# ---------- Tier 1 screen-space approximation (kept for fallback) ----------

def phoxoidal_density_screen(centers_2d, cov_2d_inv, germ, px_flat, py_flat, splat_idx):
    dx = px_flat - centers_2d[splat_idx, 0]
    dy = py_flat - centers_2d[splat_idx, 1]
    cinv = cov_2d_inv[splat_idx]
    mahal_sq = (cinv[0, 0] * dx * dx + 2.0 * cinv[0, 1] * dx * dy + cinv[1, 1] * dy * dy)
    mahal_sq = np.maximum(mahal_sq, 0.0)
    g = germ[splat_idx]
    kappa_mag = abs(float(g[0])) + abs(float(g[1]))
    LAMBDA = 0.5
    action = 0.5 * mahal_sq * (1.0 + LAMBDA * kappa_mag * mahal_sq)
    return np.exp(-np.minimum(np.maximum(action, 0.0), 20.0))


# ---------- Tier 2 faithful screen-space density (full 5-coef germ) ----------

def phoxoidal_density_germ_full(centers_2d, cov_2d, cov_2d_inv, germ, sigma_n_screen,
                                px_flat, py_flat, splat_idx):
    """Tier 2 faithful-ish evaluator using the full 5-coef germ in eigenframe coords."""
    g = germ[splat_idx]
    if not np.any(g != 0):
        dx = px_flat - centers_2d[splat_idx, 0]
        dy = py_flat - centers_2d[splat_idx, 1]
        cinv = cov_2d_inv[splat_idx]
        m = cinv[0, 0] * dx * dx + 2.0 * cinv[0, 1] * dx * dy + cinv[1, 1] * dy * dy
        return np.exp(-0.5 * np.maximum(m, 0.0))
    dx = px_flat - centers_2d[splat_idx, 0]
    dy = py_flat - centers_2d[splat_idx, 1]
    cinv = cov_2d_inv[splat_idx]
    mahal_sq = cinv[0, 0] * dx * dx + 2.0 * cinv[0, 1] * dx * dy + cinv[1, 1] * dy * dy
    mahal_sq = np.maximum(mahal_sq, 0.0)
    cov = cov_2d[splat_idx]
    a, b, d = cov[0, 0], cov[0, 1], cov[1, 1]
    trace = a + d
    det = a * d - b * b
    disc = np.sqrt(max(trace * trace - 4 * det, 0.0))
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)
    if abs(b) > 1e-9:
        v1 = np.array([lam1 - d, b], dtype=np.float64)
    else:
        v1 = np.array([1.0, 0.0], dtype=np.float64) if a >= d else np.array([0.0, 1.0], dtype=np.float64)
    v1 /= (np.linalg.norm(v1) + 1e-12)
    v2 = np.array([-v1[1], v1[0]], dtype=np.float64)
    p1 = dx * v1[0] + dy * v1[1]
    p2 = dx * v2[0] + dy * v2[1]
    s_loc = p1 / (np.sqrt(max(lam1, 1e-12)) + 1e-12)
    t_loc = p2 / (np.sqrt(max(lam2, 1e-12)) + 1e-12)
    H = germ_eval(g[None, :].astype(np.float32),
                  s_loc.astype(np.float32),
                  t_loc.astype(np.float32))
    H = np.asarray(H).flatten()
    action = mahal_sq + H * H
    action = np.minimum(np.maximum(action, 0.0), 40.0)
    return np.exp(-0.5 * action)
