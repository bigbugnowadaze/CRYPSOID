# Phase F.12 — Phoxel results (v01)

**Date**: 2026-05-02
**Dataset**: Tanks & Temples Family, 8 evenly-spaced views @ 320×180

## Headline

| Pipeline                                          | Mean PSNR (8 views) |
|---------------------------------------------------|---------------------|
| F.10/F.11 splat-from-scratch (position-frozen)    | ~5 dB               |
| **F.12 Phoxel** (64³ voxel grid, 80 iters)        | **18.50 dB**        |

That's **+13.5 dB** — a ~22× reduction in MSE — on the same 8-photo dataset
with the same SfM front-end. Logarithmic scale: each 3 dB doubles the SNR.

## What changed

The F.10/F.11 stack froze blob positions/scales/quats and only optimized
color + opacity. After ~75 iters the loss plateaued because:
  - density-control kept *adding* blobs at the (noisy) MVS positions
  - the optimizer couldn't *move* them onto the actual surface
  - extra blobs at wrong positions hurt as much as they helped → flat 5 dB

Phoxel inverts the parameterization. The voxel grid IS the parameter:
  - 64³ = 262 144 cells, each holding (density, RGB)
  - density and color flow via analytic gradients along the photometric
    residual (no positions to "freeze")
  - RMSProp adapts per-cell lr to handle wildly varying gradient magnitude
  - at the end we extract one isotropic blob per occupied cell into a
    standard `.3dphox` file (no format change needed)

## Numerics

  - SfM (LUD-global): 8/8 cams registered, 95 sparse 3D points
  - AABB extent: 0.67 (auto-fit from cam positions + sparse cloud)
  - Train resolution: 0.5× (160×90), 40 ray samples, RMSProp
  - lr_density 2.0, lr_color 0.3
  - 80 iters in 39 s of wall time on CPU (Numba JIT, parallel rays)
  - Final blobs: 66 289 (cells with density > 0.5)
  - .3dphox size: 377 KB

Per-camera PSNRs:  18.1  19.1  19.2  17.2  18.7  19.6  18.2  17.8

## Why this matters (claimed novelty)

To our knowledge no published CPU implementation of a Plenoxel-class
volumetric reconstruction → splat-format pipeline exists. Plenoxels itself
(Fridovich-Keil et al., CVPR 2022) is GPU-only. Our differentiators:

  1. **First CPU Plenoxel-class** reconstruction (no torch/CUDA, just Numba)
  2. **Voxel grid as intermediate, splats as output** — preserves CRYPSOID
     `.3dphox` compatibility, every downstream tool just works
  3. **Phoxoidal extraction roadmap** (F.12.2): derive (κ₁, κ₂) per blob
     from the local Hessian of the density field — surface-aligned germs,
     not axis-aligned voxel cubes. This is the CRYPSOID-distinctive piece.

## Files

  - `outputs/family_phoxel_v02.3dphox` — 66 289 blobs, 377 KB
  - `outputs/renders/phoxel_v01/SHOWCASE_PHOXEL_v01_8panel.png` — 8-view
    contact sheet (GT | Phoxel render | |diff|×4)
  - `tools/img2phox/phoxel.py` — voxel grid + JIT forward/backward + RMSProp
  - `tools/img2phox/cli_phoxel.py` — end-to-end driver

## Next steps

  - F.12.1: push to 96³ + 200 iters → estimated 22-25 dB
  - F.12.2: phoxoidal Hessian-based germ extraction (the novelty payoff)
  - F.12.3: octree subdivision (spend cells on surface, not interior)
