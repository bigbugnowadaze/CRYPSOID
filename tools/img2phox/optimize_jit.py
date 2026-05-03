"""F.8+ Numba-JIT'd hot loops for the dense optimizer.

Two functions get JIT-compiled:
  - render_blobs_jit: per-blob splat rasterizer with alpha-over compositing
  - aggregate_signal_jit: per-blob residual aggregation across all cameras

Both run with @njit(parallel=True). On a 17k-blob, 6-camera scene we expect
~50-100x speedup over the pure-Python optimize.py / optimize_dense.py loops.

The function signatures match the corresponding pure-Python equivalents so
optimize_dense.py can swap them in transparently.
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
def render_blobs_jit(xyz, scales_max_lin, opacity, sh_dc,
                      cam_R, cam_t, focal, cx, cy, H, W):
    """JIT splat rasterizer.  Returns (H, W, 3) float32 framebuffer + (H, W) alpha.

    Args:
        xyz: (N, 3) world positions
        scales_max_lin: (N,) per-blob max sigma in LINEAR units (already exp'd)
        opacity: (N,) clamped to [0, 1]
        sh_dc: (N, 3) RGB color in [0, 1]
        cam_R: (3, 3) world-to-cam rotation
        cam_t: (3,)   world-to-cam translation
        focal, cx, cy: intrinsics
        H, W: framebuffer dims
    """
    fb = np.zeros((H, W, 3), dtype=np.float32)
    ab = np.zeros((H, W), dtype=np.float32)
    N = xyz.shape[0]

    # Project all blobs to camera space
    cx_arr = np.empty(N, dtype=np.float32)
    cy_arr = np.empty(N, dtype=np.float32)
    cz_arr = np.empty(N, dtype=np.float32)
    for i in range(N):
        cx_arr[i] = (cam_R[0, 0] * xyz[i, 0] + cam_R[0, 1] * xyz[i, 1] +
                      cam_R[0, 2] * xyz[i, 2] + cam_t[0])
        cy_arr[i] = (cam_R[1, 0] * xyz[i, 0] + cam_R[1, 1] * xyz[i, 1] +
                      cam_R[1, 2] * xyz[i, 2] + cam_t[1])
        cz_arr[i] = (cam_R[2, 0] * xyz[i, 0] + cam_R[2, 1] * xyz[i, 1] +
                      cam_R[2, 2] * xyz[i, 2] + cam_t[2])

    # Sort indices by depth (back-to-front)
    order = np.argsort(-cz_arr)

    for k in range(N):
        i = order[k]
        z = cz_arr[i]
        if z <= 0.05:
            continue
        px = (cx_arr[i] / z) * focal + cx
        py = (cy_arr[i] / z) * focal + cy
        rad_px = (focal * scales_max_lin[i] / z)
        if rad_px < 0.5:
            rad_px = 0.5
        elif rad_px > 50.0:
            rad_px = 50.0
        r = int(np.ceil(rad_px))
        x0 = int(px) - r;  x1 = int(px) + r + 1
        y0 = int(py) - r;  y1 = int(py) + r + 1
        if x0 < 0: x0 = 0
        if y0 < 0: y0 = 0
        if x1 > W: x1 = W
        if y1 > H: y1 = H
        if x1 <= x0 or y1 <= y0:
            continue
        sigma2_inv = 2.0 / (rad_px * rad_px + 1e-6)
        cr = sh_dc[i, 0]; cg = sh_dc[i, 1]; cb = sh_dc[i, 2]
        op = opacity[i]
        for yy in range(y0, y1):
            dy = yy - py
            for xx in range(x0, x1):
                dx = xx - px
                m = (dx*dx + dy*dy) * sigma2_inv
                if m > 18.0:
                    continue
                g = np.exp(-0.5 * m)
                if g < 1e-4:
                    continue
                a = op * g
                if a > 1.0: a = 1.0
                contrib = a * (1.0 - ab[yy, xx])
                fb[yy, xx, 0] += contrib * cr
                fb[yy, xx, 1] += contrib * cg
                fb[yy, xx, 2] += contrib * cb
                ab[yy, xx] += contrib
    return fb, ab


@njit(cache=True, fastmath=True, boundscheck=False)
def aggregate_residual_signal_jit(xyz, cam_R, cam_t, focal, cx, cy, H, W,
                                    residual):
    """For each blob, sample the residual at its projected pixel.

    Returns (color_corr (N,3), opa_grad (N,), grad_mag (N,), seen_mask (N,)).
    """
    N = xyz.shape[0]
    color_corr = np.zeros((N, 3), dtype=np.float32)
    opa_grad   = np.zeros(N, dtype=np.float32)
    grad_mag   = np.zeros(N, dtype=np.float32)
    seen       = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        cx_w = (cam_R[0, 0] * xyz[i, 0] + cam_R[0, 1] * xyz[i, 1] +
                 cam_R[0, 2] * xyz[i, 2] + cam_t[0])
        cy_w = (cam_R[1, 0] * xyz[i, 0] + cam_R[1, 1] * xyz[i, 1] +
                 cam_R[1, 2] * xyz[i, 2] + cam_t[1])
        cz_w = (cam_R[2, 0] * xyz[i, 0] + cam_R[2, 1] * xyz[i, 1] +
                 cam_R[2, 2] * xyz[i, 2] + cam_t[2])
        if cz_w <= 0.05:
            continue
        px_f = (cx_w / cz_w) * focal + cx
        py_f = (cy_w / cz_w) * focal + cy
        if px_f < 0 or px_f >= W or py_f < 0 or py_f >= H:
            continue
        pxi = int(px_f); pyi = int(py_f)
        rx = residual[pyi, pxi, 0]
        ry = residual[pyi, pxi, 1]
        rz = residual[pyi, pxi, 2]
        color_corr[i, 0] = rx
        color_corr[i, 1] = ry
        color_corr[i, 2] = rz
        opa_grad[i] = (rx + ry + rz) * (1.0 / 3.0)
        grad_mag[i] = np.sqrt(rx*rx + ry*ry + rz*rz)
        seen[i] = 1
    return color_corr, opa_grad, grad_mag, seen


# ---------- Wrappers that look like the existing optimize.py / optimize_dense.py API ----------

def render_blobs_to_photo_jit(blobs, cameras, cam_idx):
    """Drop-in for optimize.render_blobs_to_photo, JIT'd."""
    intr = cameras.intrinsics
    extr = cameras.extrinsics[cam_idx]
    scales_max_lin = np.exp(blobs.scales.max(axis=1)).astype(np.float32)
    opa = np.clip(blobs.opacity, 0, 1).astype(np.float32)
    sh_dc = np.clip(blobs.sh_dc, 0, 1).astype(np.float32)
    fb, ab = render_blobs_jit(
        blobs.xyz.astype(np.float32),
        scales_max_lin, opa, sh_dc,
        extr.R.astype(np.float32), extr.t.astype(np.float32),
        float(intr.focal_x), float(intr.cx), float(intr.cy),
        int(intr.height), int(intr.width),
    )
    return fb.clip(0, 1)


def aggregate_signal_jit(blobs, cameras, residuals):
    """Drop-in for optimize_dense._per_blob_residual_signal, JIT'd.

    residuals: list of N (H, W, 3) float32 residual maps (one per camera).
    """
    intr = cameras.intrinsics
    N = len(blobs)
    color_corr_total = np.zeros((N, 3), dtype=np.float32)
    opa_grad_total   = np.zeros(N, dtype=np.float32)
    grad_mag_total   = np.zeros(N, dtype=np.float32)
    seen_count       = np.zeros(N, dtype=np.int32)
    xyz = blobs.xyz.astype(np.float32)
    for ci, extr in enumerate(cameras.extrinsics):
        cc, og, gm, sm = aggregate_residual_signal_jit(
            xyz, extr.R.astype(np.float32), extr.t.astype(np.float32),
            float(intr.focal_x), float(intr.cx), float(intr.cy),
            int(intr.height), int(intr.width),
            residuals[ci].astype(np.float32),
        )
        color_corr_total += cc
        opa_grad_total   += og
        grad_mag_total   += gm
        seen_count       += sm.astype(np.int32)
    return {
        'color_correction':   color_corr_total,
        'opacity_gradient':   opa_grad_total,
        'gradient_magnitude': grad_mag_total,
        'coverage_count':     seen_count,
    }
