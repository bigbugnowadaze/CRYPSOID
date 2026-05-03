"""F.8++ Adaptive density control following Kerbl et al. 2023, §5.2.

Schedule (defaults match the paper; all configurable):
  - densify_from_iter:        500    (warm up before adapting)
  - densify_until_iter:    15_000    (stop densifying entirely after this)
  - densify_interval:         100    (try densify every K iters)
  - opacity_reset_interval: 3_000    (reset all opacities to 0.01 every K iters)
  - prune_interval:           100    (prune low-opacity blobs every K iters)
  - prune_opacity_threshold:  0.005  (drop blobs below this)
  - grad_threshold_init:      2e-4   (per-pixel-screen-space gradient magnitude)
  - grad_threshold_halve_at: (1000, 5000, 10_000)  (halve grad threshold at these iters)
  - split_size_threshold:     0.01   (in scene units; > => split, < => clone)
  - max_blobs:                500_000

Two flavors of densification:
  - SPLIT: blob is large (max scale > split_size_threshold). Replace with two
    daughters of half the scale at jittered positions sampled from the parent's
    own ellipsoid. Stops the parent from over-covering its area.
  - CLONE: blob is small but has high gradient. Duplicate it nearby. Builds
    coverage in under-represented detail.

Pruning:
  - Drop any blob with sigmoid(opacity) < prune_opacity_threshold.
  - Drop blobs whose max scale exceeds 0.1 * scene_radius (catches degenerate
    huge blobs that grow during optimization).

Opacity reset (every 3000 iters):
  - Set all opacities to a small value (0.01). The optimizer will then have to
    re-grow opacity for blobs that actually contribute. Blobs that *don't*
    re-grow get pruned within a few hundred iters. This is the paper's trick
    for periodically clearing accumulated dead weight.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DensityScheduleConfig:
    densify_from_iter:         int = 500
    densify_until_iter:        int = 15_000
    densify_interval:          int = 100
    opacity_reset_interval:    int = 3_000
    prune_interval:            int = 100
    prune_opacity_threshold:   float = 0.005
    grad_threshold_init:       float = 2e-4
    grad_threshold_halve_at:   tuple = (1000, 5000, 10_000)
    split_size_threshold:      float = 0.01      # in scene units
    max_blobs:                 int = 500_000

    # When pruning oversized blobs: drop if max-scale > kappa * scene_radius
    big_blob_kappa:            float = 0.1


@dataclass
class DensityScheduleState:
    """Tracks accumulated per-blob gradients across iterations.
    Reset whenever density control fires."""
    grad_accum:    np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    coverage_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))


def current_grad_threshold(it: int, cfg: DensityScheduleConfig) -> float:
    """The grad threshold halves at each milestone. So at iter 0..999 it's the
    init value; 1000..4999 it's init/2; 5000..9999 init/4; 10000+ init/8."""
    g = cfg.grad_threshold_init
    for milestone in cfg.grad_threshold_halve_at:
        if it >= milestone:
            g *= 0.5
    return g


def reset_opacity(blobs, value: float = 0.01):
    """Reset all blob opacities to `value`. Returns blobs unchanged in shape."""
    blobs.opacity[:] = value
    return blobs


def density_step(blobs, state: DensityScheduleState,
                  per_blob_signal: dict,
                  scene_radius: float,
                  iter_idx: int,
                  cfg: DensityScheduleConfig):
    """Run one density-control step at iteration `iter_idx`.

    Args:
        blobs: BlobBundle (mutated in place + can be replaced by larger BlobBundle)
        state: persistent state for grad accumulation
        per_blob_signal: dict from optimize_jit.aggregate_signal_jit, keys
            'gradient_magnitude' (N,) and 'coverage_count' (N,)
        scene_radius: rough size of the scene (used for big-blob threshold)
        iter_idx: current iteration index
        cfg: schedule config

    Returns: (blobs, state) — possibly with different N.
    """
    # Init / resize accumulator if blobs grew
    N = len(blobs)
    if state.grad_accum.shape[0] != N:
        # Pad with zeros (new blobs start at zero accumulated grad)
        if N > state.grad_accum.shape[0]:
            new_grad = np.zeros(N, dtype=np.float32)
            new_grad[:state.grad_accum.shape[0]] = state.grad_accum
            state.grad_accum = new_grad
            new_cov = np.zeros(N, dtype=np.int32)
            new_cov[:state.coverage_accum.shape[0]] = state.coverage_accum
            state.coverage_accum = new_cov
        else:
            state.grad_accum = state.grad_accum[:N]
            state.coverage_accum = state.coverage_accum[:N]

    # Accumulate gradients each iter (lightweight)
    state.grad_accum += per_blob_signal['gradient_magnitude'].astype(np.float32)
    state.coverage_accum += per_blob_signal['coverage_count'].astype(np.int32)

    out_blobs = blobs

    # ---- Periodic pruning ----
    if iter_idx > 0 and (iter_idx % cfg.prune_interval == 0):
        opa_mask = out_blobs.opacity > cfg.prune_opacity_threshold
        big_size = np.exp(out_blobs.scales).max(axis=1)
        size_mask = big_size < (cfg.big_blob_kappa * scene_radius)
        keep = opa_mask & size_mask
        if not keep.all():
            n_pruned = int((~keep).sum())
            out_blobs = _select_blobs(out_blobs, keep)
            state.grad_accum = state.grad_accum[keep]
            state.coverage_accum = state.coverage_accum[keep]

    # ---- Periodic densification (split / clone) ----
    in_window = (iter_idx >= cfg.densify_from_iter and
                  iter_idx < cfg.densify_until_iter)
    fires = (iter_idx > 0 and iter_idx % cfg.densify_interval == 0)
    can_grow = len(out_blobs) < cfg.max_blobs
    if in_window and fires and can_grow:
        threshold = current_grad_threshold(iter_idx, cfg)
        # Average accumulated grad / coverage = mean grad-per-observation
        cov = np.maximum(state.coverage_accum, 1).astype(np.float32)
        avg_grad = state.grad_accum / cov
        big = np.exp(out_blobs.scales).max(axis=1)

        densify_mask = avg_grad > threshold
        is_big = big > cfg.split_size_threshold

        split_idx = np.where(densify_mask & is_big)[0]
        clone_idx = np.where(densify_mask & ~is_big)[0]

        if len(split_idx) > 0 or len(clone_idx) > 0:
            out_blobs = _split_clone(out_blobs, split_idx, clone_idx, jitter=0.5)
            # Reset accumulators (they include daughters which start fresh)
            state.grad_accum = np.zeros(len(out_blobs), dtype=np.float32)
            state.coverage_accum = np.zeros(len(out_blobs), dtype=np.int32)

    # ---- Periodic opacity reset (paper's "let the optimizer re-grow what matters") ----
    if iter_idx > 0 and (iter_idx % cfg.opacity_reset_interval == 0):
        out_blobs.opacity[:] = 0.01

    return out_blobs, state


# ---------- Internal helpers ----------

def _select_blobs(blobs, mask):
    """Keep blobs where mask is True. Returns a NEW BlobBundle."""
    from .data_classes import BlobBundle
    return BlobBundle(
        xyz=blobs.xyz[mask].copy(),
        scales=blobs.scales[mask].copy(),
        quats=blobs.quats[mask].copy(),
        opacity=blobs.opacity[mask].copy(),
        sh_dc=blobs.sh_dc[mask].copy(),
        sh_rest=blobs.sh_rest[mask].copy() if blobs.sh_rest is not None else None,
        tier=(blobs.tier[mask].copy() if (blobs.tier is not None and len(blobs.tier) == len(mask)) else None),
    )


def _split_clone(blobs, split_idx, clone_idx, jitter: float = 0.5):
    """Create new blobs from split + clone selections. Returns a new BlobBundle."""
    from .data_classes import BlobBundle
    rng = np.random.default_rng(42)
    pieces = [(blobs.xyz, blobs.scales, blobs.quats, blobs.opacity, blobs.sh_dc)]
    # SPLIT: replace each parent with two daughters of half the scale, jittered
    if len(split_idx) > 0:
        sigma = np.exp(blobs.scales[split_idx].max(axis=1, keepdims=True))
        for _ in range(2):
            offsets = rng.normal(0, jitter, (len(split_idx), 3)).astype(np.float32) * sigma.astype(np.float32)
            pieces.append((
                blobs.xyz[split_idx] + offsets,
                blobs.scales[split_idx] - np.log(2),
                blobs.quats[split_idx].copy(),
                blobs.opacity[split_idx].copy(),
                blobs.sh_dc[split_idx].copy(),
            ))
        # Original parents get pruned out below via mask
    # CLONE: duplicate each at the same position (will diverge during optimization)
    if len(clone_idx) > 0:
        pieces.append((
            blobs.xyz[clone_idx].copy(),
            blobs.scales[clone_idx].copy(),
            blobs.quats[clone_idx].copy(),
            blobs.opacity[clone_idx].copy(),
            blobs.sh_dc[clone_idx].copy(),
        ))
    new_xyz    = np.concatenate([p[0] for p in pieces], axis=0)
    new_scales = np.concatenate([p[1] for p in pieces], axis=0)
    new_quats  = np.concatenate([p[2] for p in pieces], axis=0)
    new_opa    = np.concatenate([p[3] for p in pieces], axis=0)
    new_dc     = np.concatenate([p[4] for p in pieces], axis=0)
    # Build mask: keep all originals EXCEPT split parents (they're replaced by daughters)
    keep = np.ones(len(blobs), dtype=bool)
    keep[split_idx] = False
    extra_n = sum(p[0].shape[0] for p in pieces[1:])
    final_keep = np.concatenate([keep, np.ones(extra_n, dtype=bool)])
    return BlobBundle(
        xyz=new_xyz[final_keep],
        scales=new_scales[final_keep],
        quats=new_quats[final_keep],
        opacity=new_opa[final_keep],
        sh_dc=new_dc[final_keep],
        sh_rest=None,
        tier=(blobs.tier.copy() if blobs.tier is not None else None),
    )
