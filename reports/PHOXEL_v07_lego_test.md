# Phase F.18 — LEGO Excavator (clean dataset diagnostic)

**Date**: 2026-05-03
**Dataset**: 6 clean renders of a yellow LEGO Technic excavator on a brick base,
plain black background, well-distributed angles (sides, 3/4s, top, front).

## Result

| Stage | PSNR | Notes |
|---|---|---|
| 6/6 cams + global BA (300 nfev) | n/a | 71 sparse pts, AABB needs tightening |
| Phoxel @ loose AABB (cell 0.12)  | 15.34 dB (range 13-20) | scale issue |
| Phoxel @ tight AABB (cell 0.011) | 18.00 dB (range 16-20) | proper fit |
| Phoxel @ tight, 65 iter           | **18.28 dB** (range 16-20) | converged |

## Comparison to Family

| Dataset | views | PSNR | per-cam range |
|---|---|---|---|
| Family (statue+plaza, foliage)   | 8  | **20.16 dB** | 18.7-21.1 |
| LEGO (clean object, plain bg)     | 6  | **18.28 dB** | 16.4-19.8 |

## Diagnosis

LEGO with clean renders + full-pair SfM + global BA still hits 18 dB, not
25-30. This **falsifies the "Family is hard because of foliage" hypothesis**:
even a perfect synthetic dataset with no scene clutter doesn't break past
the ~20 dB ceiling.

The actual ceiling drivers are now narrowed to two:

1. **Cell resolution**: 96³ uniform grid spreads 884k cells across the AABB.
   Per-pixel surface detail in the renders (LEGO studs at ~5-10 px each)
   needs ~0.5 px cell projected size, which means at our 480-px renders
   we'd want effective ~1000³ resolution. Octree or progressive subdivision
   (F.12.3.1) is the next algorithmic unlock.

2. **View count**: 6 views can't constrain occluded cell density. Some cells
   are seen by 0-1 cameras → no gradient → random-walk values → muddy renders.
   The 30+ view path needs incremental SfM (F.17) to deliver real value.

## Ceiling is the renderer-cell-budget, not the front-end

This LEGO test is the cleanest possible front-end (zero feature noise, perfect
correspondences, well-distributed cams) and we hit 18 dB. So the front-end
isn't the only gate anymore — **the optimizer's representation capacity at
96³ is also limited**.

To break past ~20 dB on photo-in:
- Octree subdivision done right (each surface cell gets 4× more inner cells)
- More views (drives multi-view consistency) — needs incremental SfM
- Both, together

## Files

  - `outputs/lego_phoxel_BEST.3dphox` — 31.8k Hessian-aligned blobs, 433 KB
  - `outputs/renders/phoxel_v01/SHOWCASE_LEGO_6cam.png` — 6-row contact sheet
    (GT | render | |diff|×4)
  - `outputs/_lego_sfm_cache.pkl`, `outputs/_lego_phoxel_grid.npz` — checkpoints

## For Vince

This is now the cleanest readout of the architecture's honest limits:
**~18-20 dB PSNR on photo-in, cell-budget-bound at 96³ uniform grid.**
The path past it is concrete (octree + more views) and known.
