"""Numba-JIT rasterizer for the per-splat hot loop.

Replaces the Python `for i in range(N)` loop in the existing prototype renderer
with a JIT-compiled function. Same math (Gaussian splat + back-to-front
"over" alpha compositing); just compiled.

Inputs (already projected + sorted):
    xy           (N, 2)   float32 — splat center in pixels (back-to-front order)
    inv_cov      (N, 2, 2) float32 — inverse 2D screen-space covariance
    radii        (N,)     float32 — bounding radius in pixels
    opa          (N,)     float32 — sigmoid opacity in [0, 1]
    color        (N, 3)   float32 — per-splat RGB
    H, W         int      — framebuffer dims

Returns:
    fb           (H, W, 3) float32
    ab           (H, W)    float32 (alpha buffer, transmittance accumulator)
"""

from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    njit = lambda *a, **k: (lambda f: f)
    prange = range


@njit(cache=True, fastmath=True, boundscheck=False)
def rasterize_splats_numba(xy, inv_cov, radii, opa, color, H, W):
    """JIT-compiled rasterizer. xy must be sorted back-to-front."""
    fb = np.zeros((H, W, 3), dtype=np.float32)
    ab = np.zeros((H, W), dtype=np.float32)
    N = xy.shape[0]

    for i in range(N):
        if opa[i] < 1e-4:
            continue
        cx = xy[i, 0]
        cy = xy[i, 1]
        r = int(np.ceil(radii[i]))
        if r < 1 or r > 64:
            continue
        x0 = int(cx) - r
        x1 = int(cx) + r + 1
        y0 = int(cy) - r
        y1 = int(cy) + r + 1
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > W: x1 = W
        if y1 > H: y1 = H
        if x1 <= x0 or y1 <= y0:
            continue

        a00 = inv_cov[i, 0, 0]
        a01 = inv_cov[i, 0, 1]
        a11 = inv_cov[i, 1, 1]
        cr = color[i, 0]
        cg = color[i, 1]
        cb = color[i, 2]
        op = opa[i]

        for yy in range(y0, y1):
            dy = yy - cy
            for xx in range(x0, x1):
                dx = xx - cx
                m = a00 * dx * dx + 2.0 * a01 * dx * dy + a11 * dy * dy
                if m > 18.0:   # exp(-9) is tiny, skip
                    continue
                g = np.exp(-0.5 * m)
                if g < 1e-4:
                    continue
                a = g * op
                if a > 1.0: a = 1.0
                transm = 1.0 - ab[yy, xx]
                contrib = a * transm
                fb[yy, xx, 0] += contrib * cr
                fb[yy, xx, 1] += contrib * cg
                fb[yy, xx, 2] += contrib * cb
                ab[yy, xx] += contrib
    return fb, ab


def rasterize_python(xy, inv_cov, radii, opa, color, H, W):
    """Pure-Python reference (slow)."""
    fb = np.zeros((H, W, 3), dtype=np.float32)
    ab = np.zeros((H, W), dtype=np.float32)
    N = len(xy)
    for i in range(N):
        if opa[i] < 1e-4:
            continue
        cx, cy = xy[i]
        r = int(np.ceil(radii[i]))
        if r < 1 or r > 64:
            continue
        x0 = max(0, int(cx) - r); x1 = min(W, int(cx) + r + 1)
        y0 = max(0, int(cy) - r); y1 = min(H, int(cy) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - cx; dy = yy - cy
        inv_i = inv_cov[i]
        m = inv_i[0, 0] * dx * dx + 2.0 * inv_i[0, 1] * dx * dy + inv_i[1, 1] * dy * dy
        g = np.exp(-0.5 * m).clip(0, 1).astype(np.float32)
        a = (g * opa[i]).clip(0, 1)
        transm = 1.0 - ab[y0:y1, x0:x1]
        contrib = a * transm
        fb[y0:y1, x0:x1] += contrib[:, :, None] * color[i][None, None, :]
        ab[y0:y1, x0:x1] += contrib
    return fb, ab
