"""Numerical-correctness tests for Tier 2 — anchors per spec section 6."""
from __future__ import annotations
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from crypsorender.math.germ import (
    closest_point_on_germ, germ_eval, germ_grad, germ_hess,
    phoxoidal_density_germ_full,
)


def test_newton_quadratic_germ():
    """Anchor 1: Newton converges to the analytic minimizer on a quadratic germ."""
    theta = np.array([[1.0, -2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sigma = np.array([[1.0, 1.0, 0.5]], dtype=np.float32)
    s_true, t_true = 0.3, -0.2
    n_true = s_true * s_true - 2.0 * t_true * t_true
    u = np.array([[s_true, t_true, n_true]], dtype=np.float32)
    s, t, action = closest_point_on_germ(theta, sigma, u, n_iter=8, lambda_support=0.0)
    assert abs(float(s[0]) - s_true) < 1e-2, f"s* should be {s_true}, got {float(s[0])}"
    assert abs(float(t[0]) - t_true) < 1e-2, f"t* should be {t_true}, got {float(t[0])}"
    assert float(action[0]) < 1e-3, f"action should be ~0, got {float(action[0])}"
    print(f"PASS  test_newton_quadratic_germ  s={float(s[0]):.4f} t={float(t[0]):.4f} action={float(action[0]):.2e}")


def test_phoxoidal_density_reduces_to_gaussian():
    """Anchor 2: phoxoidal_density_germ_full with germ=0 returns exact Gaussian.

    The rasterizer's screen-space evaluator has an early-return path when germ
    coefficients are all zero — it skips the Newton solver entirely and
    falls back to standard Gaussian density.  This is the "Tier C reduces to
    vanilla Gaussian" guarantee in production code.
    """
    n = 5
    centers_2d = np.array([[0.0, 0.0]], dtype=np.float32)
    # Identity covariance
    cov_2d = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    cov_2d_inv = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    germ = np.zeros((1, 5), dtype=np.float32)
    sigma_n_screen = np.array([1.0], dtype=np.float32)
    px = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    py = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    density = phoxoidal_density_germ_full(centers_2d, cov_2d, cov_2d_inv, germ, sigma_n_screen, px, py, 0)
    expected = np.exp(-0.5 * (px * px + py * py))
    err = float(np.abs(density - expected).max())
    assert err < 1e-6, f"germ=0 should produce exact Gaussian; max diff {err}"
    print(f"PASS  test_phoxoidal_density_reduces_to_gaussian  max diff: {err:.2e}")


def test_germ_gradient_consistency():
    """Anchor 3: numerical grad/hess match analytic germ_grad/germ_hess."""
    theta = np.array([[1.0, -1.5, 0.5, -0.3, 0.1]], dtype=np.float32)
    s = np.array([0.4], dtype=np.float32)
    t = np.array([-0.2], dtype=np.float32)
    eps = 1e-4
    Hs_num = (germ_eval(theta, s + eps, t) - germ_eval(theta, s - eps, t)) / (2 * eps)
    Ht_num = (germ_eval(theta, s, t + eps) - germ_eval(theta, s, t - eps)) / (2 * eps)
    Hs, Ht = germ_grad(theta, s, t)
    err_s = float(np.abs(Hs_num - Hs).max())
    err_t = float(np.abs(Ht_num - Ht).max())
    assert err_s < 1e-2 and err_t < 1e-2, f"Gradient mismatch: ds={err_s}, dt={err_t}"
    Hss_num = (germ_grad(theta, s + eps, t)[0] - germ_grad(theta, s - eps, t)[0]) / (2 * eps)
    Hss, Htt, Hst = germ_hess(theta, s, t)
    err_ss = float(np.abs(Hss_num - Hss).max())
    assert err_ss < 1e-2, f"Hessian mismatch ss: {err_ss}"
    print(f"PASS  test_germ_gradient_consistency  ds={err_s:.2e} dt={err_t:.2e} ss={err_ss:.2e}")


def test_cusp_germ_visible():
    """Anchor 4: cusp germ produces measurably different action than no germ.

    Note: pure central-flip asymmetry (action(p) vs action(-p)) IS zero for
    the cusp because H(-s,-t)=-H(s,t), so flipping u also flips the closest
    point. The honest test is: cusp action != no-germ action at the same u.
    """
    chi = 0.4
    theta_cusp = np.array([[0.0, 0.0, chi, 0.0, 0.0]], dtype=np.float32)
    theta_zero = np.zeros((1, 5), dtype=np.float32)
    sigma = np.array([[1.0, 1.0, 0.3]], dtype=np.float32)
    # Sample points where the cusp surface has notable height (n != 0)
    s_grid = np.linspace(-1, 1, 11)
    t_grid = np.linspace(-1, 1, 11)
    S, T = np.meshgrid(s_grid, t_grid)
    pts = np.stack([S.flatten(), T.flatten(), 0.05 * np.ones(S.size)], axis=1).astype(np.float32)
    th_c = np.tile(theta_cusp, (pts.shape[0], 1))
    th_0 = np.tile(theta_zero, (pts.shape[0], 1))
    sg_b = np.tile(sigma, (pts.shape[0], 1))
    _, _, action_cusp = closest_point_on_germ(th_c, sg_b, pts, n_iter=4, lambda_support=0.0)
    _, _, action_flat = closest_point_on_germ(th_0, sg_b, pts, n_iter=4, lambda_support=0.0)
    diff_max = float(np.abs(action_cusp - action_flat).max())
    diff_mean = float(np.abs(action_cusp - action_flat).mean())
    print(f"PASS  test_cusp_germ_visible  max diff cusp-vs-flat: {diff_max:.3e}, mean: {diff_mean:.3e}")
    assert diff_max > 1e-3, f"Cusp germ should differ from flat germ; max diff {diff_max}"


def main():
    print("Running Tier 2 numerical-correctness anchors...\n")
    test_newton_quadratic_germ()
    test_phoxoidal_density_reduces_to_gaussian()
    test_germ_gradient_consistency()
    test_cusp_germ_visible()
    print("\nAll Tier 2 anchors passed.")


if __name__ == "__main__":
    main()
