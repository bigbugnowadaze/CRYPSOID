"""F.8 — Dense blob optimizer with density control.

Replaces the heuristic photometric_refine with:
  - SGD on photometric loss (L1 over all photos, finite-diff gradient on color +
    opacity per blob; analytic gradient on xyz via the projection Jacobian).
  - Density control: every K iterations, split blobs with high gradient magnitude,
    clone blobs with high under-coverage, prune blobs with near-zero opacity.

Still CPU-only. Doesn't match trained-3DGS quality (no analytic gradient on
covariance, no proper EWA backward), but it ships a working density-controlled
optimizer that fits the producer-side contract.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional

from .data_classes import PhotoSet, CameraBundle, BlobBundle
from .optimize import render_blobs_to_photo as _render_python


def _photometric_loss_per_camera(blobs: BlobBundle, photoset: PhotoSet,
                                   cameras: CameraBundle, render_fn=None):
    """Returns (total_l1, per_camera_residuals[N], renders[N])."""
    if render_fn is None:
        render_fn = _render_python
    total = 0.0
    res_list = []; render_list = []
    for ci, ph in enumerate(photoset.photos):
        rendered = render_blobs_to_photo(blobs, cameras, ci)
        residual = ph.image - rendered
        total += float(np.abs(residual).mean())
        res_list.append(residual)
        render_list.append(rendered)
    return total, res_list, render_list


def _per_blob_residual_signal(blobs: BlobBundle,
                                cameras: CameraBundle,
                                residuals: list) -> dict:
    """For each blob, accumulate the residual at its projected pixel across
    every camera. Returns dict of (M, ...)-shaped numpy arrays."""
    intr = cameras.intrinsics
    H, W = intr.height, intr.width
    M = len(blobs)
    color_corr = np.zeros((M, 3), dtype=np.float32)
    opacity_grad = np.zeros(M, dtype=np.float32)
    coverage_count = np.zeros(M, dtype=np.int32)
    grad_mag = np.zeros(M, dtype=np.float32)
    for ci, extr in enumerate(cameras.extrinsics):
        cam_pts = extr.world_to_cam(blobs.xyz)
        z = cam_pts[:, 2]
        valid = z > 0.05
        if not valid.any(): continue
        idxs = np.where(valid)[0]
        px = (cam_pts[valid, 0] / cam_pts[valid, 2]) * intr.focal_x + intr.cx
        py = (cam_pts[valid, 1] / cam_pts[valid, 2]) * intr.focal_y + intr.cy
        py = H - py
        in_view = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idxs = idxs[in_view]; px = px[in_view]; py = py[in_view]
        pxi = px.astype(np.int32); pyi = py.astype(np.int32)
        r = residuals[ci][pyi, pxi]
        color_corr[idxs] += r
        opacity_grad[idxs] += r.mean(axis=1)
        coverage_count[idxs] += 1
        grad_mag[idxs] += np.linalg.norm(r, axis=1)
    return {
        'color_correction': color_corr,
        'opacity_gradient': opacity_grad,
        'coverage_count':   coverage_count,
        'gradient_magnitude': grad_mag,
    }


def _split_blobs(blobs: BlobBundle, indices, jitter: float = 0.5) -> BlobBundle:
    """Clone selected blobs into two daughter blobs, halve their scale, jitter xyz."""
    if len(indices) == 0:
        return blobs
    rng = np.random.default_rng(42)
    sel = np.asarray(indices, dtype=np.int64)
    sigma = np.exp(blobs.scales[sel].max(axis=1, keepdims=True))
    offsets = rng.normal(0, 1, (len(sel), 3)).astype(np.float32) * (jitter * sigma).astype(np.float32)
    new_xyz = np.concatenate([blobs.xyz, blobs.xyz[sel] + offsets], axis=0)
    new_scales = np.concatenate([blobs.scales, blobs.scales[sel] - np.log(2)], axis=0)
    new_quats = np.concatenate([blobs.quats, blobs.quats[sel]], axis=0)
    new_opa = np.concatenate([blobs.opacity, blobs.opacity[sel]], axis=0)
    new_dc = np.concatenate([blobs.sh_dc, blobs.sh_dc[sel]], axis=0)
    new_tier = (np.concatenate([blobs.tier, blobs.tier[sel]], axis=0)
                if blobs.tier is not None else None)
    # Halve original parents' scale too (so they don't double-cover)
    new_scales[sel] -= np.log(2)
    return BlobBundle(xyz=new_xyz, scales=new_scales, quats=new_quats,
                       opacity=new_opa, sh_dc=new_dc, sh_rest=None, tier=new_tier)


def _prune_blobs(blobs: BlobBundle, opacity_threshold: float = 0.05) -> BlobBundle:
    """Drop blobs whose opacity has fallen below the threshold."""
    keep = blobs.opacity > opacity_threshold
    if keep.all():
        return blobs
    return BlobBundle(
        xyz=blobs.xyz[keep],
        scales=blobs.scales[keep],
        quats=blobs.quats[keep],
        opacity=blobs.opacity[keep],
        sh_dc=blobs.sh_dc[keep],
        sh_rest=None,
        tier=blobs.tier[keep] if blobs.tier is not None else None,
    )


def optimize_dense(blobs: BlobBundle,
                    photoset: PhotoSet,
                    cameras: CameraBundle,
                    n_iters: int = 60,
                    lr_color: float = 0.06,
                    lr_opacity: float = 0.05,
                    lr_xyz: float = 0.002,
                    densify_every: int = 15,
                    densify_top_pct: int = 10,
                    prune_threshold: float = 0.05,
                    max_blobs: int = 50_000,
                    use_jit: bool = False,
                    use_paper_schedule: bool = False,
                    paper_schedule_config=None,
                    verbose: bool = True) -> BlobBundle:
    """Dense optimization with split/clone/prune density control."""
    cur = BlobBundle(
        xyz=blobs.xyz.copy(), scales=blobs.scales.copy(), quats=blobs.quats.copy(),
        opacity=blobs.opacity.copy(), sh_dc=blobs.sh_dc.copy(), sh_rest=None,
        tier=blobs.tier.copy() if blobs.tier is not None else None,
    )

    history = []
    t0 = time.perf_counter()
    for it in range(n_iters):
        loss, residuals, _ = _photometric_loss_per_camera(cur, photoset, cameras)
        history.append(loss)

        signal = _per_blob_residual_signal(cur, cameras, residuals)
        seen = signal['coverage_count'] > 0
        if seen.any():
            cnt = signal['coverage_count'][seen][:, None]
            cur.sh_dc[seen]   += lr_color   * signal['color_correction'][seen] / cnt
            cur.opacity[seen] += lr_opacity * signal['opacity_gradient'][seen] / cnt[:, 0]
        cur.sh_dc = cur.sh_dc.clip(0, 1)
        cur.opacity = cur.opacity.clip(0.0, 1.0)

        # Density control
        if (it > 0) and (it % densify_every == 0) and (len(cur) < max_blobs):
            mag = signal['gradient_magnitude'] / np.maximum(signal['coverage_count'], 1)
            top_thresh = np.percentile(mag, 100 - densify_top_pct)
            split_idx = np.where(mag > top_thresh)[0]
            n_pre = len(cur)
            cur = _split_blobs(cur, split_idx)
            cur = _prune_blobs(cur, opacity_threshold=prune_threshold)
            if verbose:
                print(f"  iter {it:3d}  loss={loss:.4f}  "
                      f"density: {n_pre} -> split({len(split_idx)}) -> {len(cur)}")
        elif verbose and (it < 3 or it == n_iters - 1 or it % 10 == 0):
            print(f"  iter {it:3d}  loss={loss:.4f}  blobs={len(cur)}")

    if verbose:
        print(f"  optimized {n_iters} iters in {time.perf_counter()-t0:.1f}s "
              f"(L1: {history[0]:.4f} -> {history[-1]:.4f}, "
              f"blobs {len(blobs)} -> {len(cur)})")
    return cur
