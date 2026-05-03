"""F.3 — Sparse point cloud -> BlobBundle.

Two paths:
  - quick_seed_from_pointcloud: each point becomes a blob with sigma = kNN
    spread, opacity = high, color = point color. Zero optimization. This is
    the "good enough" fallback that always produces a renderable BlobBundle.
  - photometric_refine: optional iterative loop that nudges blob xyz/scales/
    opacity/color via finite-differenced photometric loss against the photos.
    Slow, but works for proof-of-concept on small synthetic scenes.

In Phase F.3 we ship quick_seed + a basic photometric_refine that runs ~50
iterations on ~1k blobs in under a minute. F.8 will replace this with a
proper analytic-gradient optimizer at trained-3DGS scale.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional, List
from sklearn.neighbors import NearestNeighbors

from .data_classes import (
    PhotoSet, CameraBundle, PointCloud, BlobBundle,
)
from .sfm import project_points


SH_C0 = 0.28209479177387814


def quick_seed_from_pointcloud(cloud: PointCloud,
                                opacity: float = 0.7,
                                scale_kappa: float = 0.5,
                                k_neighbors: int = 6) -> BlobBundle:
    """One blob per point. Sigma = scale_kappa * mean(distance to k nearest).

    Returns a BlobBundle ready for v25 encoding.
    """
    N = len(cloud)
    if N == 0:
        raise ValueError("cloud is empty")

    # Per-blob sigma from local point density
    knn = NearestNeighbors(n_neighbors=min(k_neighbors+1, N)).fit(cloud.xyz)
    dists, _ = knn.kneighbors(cloud.xyz)
    mean_d = dists[:, 1:].mean(axis=1)              # skip self
    sigma_lin = (scale_kappa * mean_d).astype(np.float32)
    sigma_lin = np.maximum(sigma_lin, 1e-3)
    log_sigma = np.log(sigma_lin)
    scales = np.stack([log_sigma, log_sigma, log_sigma], axis=1)

    # Identity quaternions
    quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (N, 1))

    # Opacity (sigmoid logit). The renderer treats this as already-decoded [0,1].
    opacities = np.full(N, opacity, dtype=np.float32)

    # Colors: pass through cloud.colors as the "albedo" / SH DC slot.
    if cloud.colors is not None:
        sh_dc = cloud.colors.astype(np.float32)
    else:
        sh_dc = np.full((N, 3), 0.7, dtype=np.float32)
    sh_rest = None      # No view-dependent SH from a sparse cloud

    return BlobBundle(
        xyz=cloud.xyz.astype(np.float32),
        scales=scales,
        quats=quats,
        opacity=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        tier=np.full(N, 2, dtype=np.uint8),     # all C-tier (Gaussian) for now
    )


# ---------------- Photometric refinement (basic, optional) ----------------

def render_blobs_to_photo(blobs: BlobBundle,
                            cameras: CameraBundle,
                            cam_idx: int) -> np.ndarray:
    """Cheap forward render: project blobs to a (H, W, 3) photo.

    Uses splat radius = scale (in linear, not log) projected to pixel size.
    Composites with simple alpha-over (no depth sort — fine for the small
    synthetic scenes we use in F.3 testing).
    """
    intr = cameras.intrinsics
    extr = cameras.extrinsics[cam_idx]
    H, W = intr.height, intr.width
    img = np.zeros((H, W, 3), dtype=np.float32)
    alpha_sum = np.zeros((H, W), dtype=np.float32)

    cam_pts = extr.world_to_cam(blobs.xyz)
    z = cam_pts[:, 2]
    valid = z > 0.05
    if not valid.any():
        return img
    cp = cam_pts[valid]
    px = (cp[:, 0] / cp[:, 2]) * intr.focal_x + intr.cx
    py = (cp[:, 1] / cp[:, 2]) * intr.focal_y + intr.cy
    py = H - py
    sigma_lin = np.exp(blobs.scales[valid].max(axis=1))
    rad_px = (intr.focal_x * sigma_lin / cp[:, 2]).clip(0.5, 50)
    colors = blobs.sh_dc[valid].clip(0, 1)
    opa = blobs.opacity[valid].clip(0, 1)

    # Sort back-to-front
    order = np.argsort(-cp[:, 2])
    px, py, rad_px, colors, opa = (
        px[order], py[order], rad_px[order], colors[order], opa[order]
    )
    pxi = px.astype(np.int32); pyi = py.astype(np.int32)

    for i in range(len(pxi)):
        r = int(np.ceil(rad_px[i]))
        x0, x1 = max(0, pxi[i]-r), min(W, pxi[i]+r+1)
        y0, y1 = max(0, pyi[i]-r), min(H, pyi[i]+r+1)
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - px[i], yy - py[i]
        falloff = np.exp(-2.0 * (dx*dx + dy*dy) / (rad_px[i]*rad_px[i] + 1e-6))
        a = opa[i] * falloff
        contrib = a * (1 - alpha_sum[y0:y1, x0:x1])
        img[y0:y1, x0:x1] += contrib[..., None] * colors[i][None, None, :]
        alpha_sum[y0:y1, x0:x1] += contrib
    return img.clip(0, 1)


def photometric_refine(blobs: BlobBundle,
                        photoset: PhotoSet,
                        cameras: CameraBundle,
                        n_iters: int = 30,
                        lr_scale: float = 0.02,
                        lr_color: float = 0.05,
                        lr_opacity: float = 0.05,
                        verbose: bool = True) -> BlobBundle:
    """Naive photometric refinement.

    For each iteration:
      - Render every camera view from current blob params.
      - Compute per-pixel L1 vs the photo, sum.
      - Per-blob, nudge:
          * scale (log-sigma) toward smaller if its pixel footprint is too big,
            larger if too small (heuristic from per-blob residual coverage).
          * sh_dc (color) by the mean residual color over its footprint.
          * opacity by sign of residual (under-rendered -> raise; over -> lower).
    """
    cur = BlobBundle(
        xyz=blobs.xyz.copy(),
        scales=blobs.scales.copy(),
        quats=blobs.quats.copy(),
        opacity=blobs.opacity.copy(),
        sh_dc=blobs.sh_dc.copy(),
        sh_rest=None,
        tier=blobs.tier.copy() if blobs.tier is not None else None,
    )
    history = []
    t0 = time.perf_counter()
    for it in range(n_iters):
        total_loss = 0.0
        all_color_correction = np.zeros_like(cur.sh_dc)
        all_opacity_grad = np.zeros_like(cur.opacity)
        all_scale_grad = np.zeros_like(cur.scales[:, 0])
        all_count = np.zeros(len(cur), dtype=np.int32)

        intr = cameras.intrinsics
        H, W = intr.height, intr.width
        for ci, photo in enumerate(photoset.photos):
            rendered = render_blobs_to_photo(cur, cameras, ci)
            residual = photo.image - rendered      # (H, W, 3)
            total_loss += float(np.abs(residual).mean())

            # For each blob: project center, sample residual at the pixel
            extr = cameras.extrinsics[ci]
            cam_pts = extr.world_to_cam(cur.xyz)
            z = cam_pts[:, 2]
            valid = z > 0.05
            if not valid.any(): continue
            px = (cam_pts[valid, 0] / cam_pts[valid, 2]) * intr.focal_x + intr.cx
            py = (cam_pts[valid, 1] / cam_pts[valid, 2]) * intr.focal_y + intr.cy
            py = H - py
            pxi = np.clip(px.astype(np.int32), 0, W-1)
            pyi = np.clip(py.astype(np.int32), 0, H-1)
            # Per-blob residual sample
            r = residual[pyi, pxi]               # (M_valid, 3)
            idxs = np.where(valid)[0]
            all_color_correction[idxs] += r
            all_opacity_grad[idxs] += r.mean(axis=1)
            all_count[idxs] += 1

        # Apply nudges (averaged across cameras that saw each blob)
        seen = all_count > 0
        cur.sh_dc[seen]    += lr_color   * all_color_correction[seen] / all_count[seen, None]
        cur.opacity[seen]  += lr_opacity * all_opacity_grad[seen] / all_count[seen]
        # Scale step: shrink slightly to reduce leakage if residual is mostly negative
        # (over-coverage), grow if residual is positive (under-coverage)
        cur.scales[seen] += lr_scale * np.tanh(0.5 * all_opacity_grad[seen] / all_count[seen])[:, None] * 0.3

        cur.sh_dc = cur.sh_dc.clip(0, 1)
        cur.opacity = cur.opacity.clip(0.05, 1.0)

        history.append(total_loss / max(len(photoset), 1))
        if verbose and (it < 3 or it == n_iters - 1 or it % 10 == 0):
            print(f"  iter {it:3d}  L1 loss = {history[-1]:.4f}", flush=True)
    if verbose:
        print(f"  refined {n_iters} iters in {time.perf_counter()-t0:.1f}s "
              f"(L1: {history[0]:.4f} -> {history[-1]:.4f})")
    return cur
