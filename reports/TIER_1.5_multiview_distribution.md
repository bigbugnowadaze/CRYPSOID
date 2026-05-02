# Tier 1.5 item 5 — Multi-view distribution metrics

**Why this matters.** A single side-view PSNR number could be lucky. The honest version: render at many viewpoints, report the *distribution* (mean / median / worst) across views.

## Setup

- 8 cameras orbiting the Audi at yaw = 0°, 45°, 90°, …, 315°, pitch = +8°.
- Render resolution: 256×256, max-points 30,000 subsample (sandbox-time-bounded).
- Object-mask metrics (background pixels excluded, per item 4).
- Reference: PLY rendered through the same harness/camera/subsample.

## Results

| Comparison | PSNR mean | PSNR median | PSNR worst | SSIM mean | SSIM worst |
|---|---:|---:|---:|---:|---:|
| **v28 EXACT archive vs PLY** | 55.28 dB | 55.32 dB | **54.44 dB** | 0.99993 | 0.99991 |
| **v28 VQ render vs PLY**     | 37.83 dB | 38.01 dB | **36.85 dB** | 0.99646 | 0.99581 |

PSNR spread (worst → best) is small for both — the side-view number is representative across the orbit, not lucky:

- v28 EXACT archive: spread = 1.42 dB
- v28 VQ render:     spread = 2.22 dB

## Per-camera detail

| Camera | yaw | archive PSNR | archive SSIM | VQ-render PSNR | VQ-render SSIM |
|---|---:|---:|---:|---:|---:|
| az00_el+8 | 0° | 54.44 dB | 0.99991 | 38.12 dB | 0.99581 |
| az01_el+8 | 45° | 55.78 dB | 0.99993 | 38.12 dB | 0.99658 |
| az02_el+8 | 90° | 55.65 dB | 0.99993 | 36.88 dB | 0.99598 |
| az03_el+8 | 135° | 54.96 dB | 0.99992 | 38.09 dB | 0.99694 |
| az04_el+8 | 180° | 55.86 dB | 0.99994 | 39.06 dB | 0.99737 |
| az05_el+8 | 225° | 54.80 dB | 0.99991 | 37.62 dB | 0.99642 |
| az06_el+8 | 270° | 54.99 dB | 0.99993 | 36.85 dB | 0.99586 |
| az07_el+8 | 315° | 55.72 dB | 0.99993 | 37.93 dB | 0.99672 |

## Reading the numbers

- **Worst-case archive PSNR is 54.44 dB.** That's the honest "ceiling-floor" number: at every camera angle tested, the v28 EXACT archive reconstructs the PLY to within ~55 dB / SSIM 0.9999 of the reference. The single-view 56.33 dB headline (in `manifest_T1.json`) was not lucky — it sits ~1 dB above the worst across the orbit.
- **Worst-case VQ render PSNR is 36.85 dB.** ~17 dB lower than the archive, consistent with the VQ codebook losing detail in the SH coefficients. SSIM stays above 0.995 across all views.
- **PSNR spread is tight** (~1.4 dB for archive, ~2.2 dB for VQ render) — there's no single bad angle.

## Caveats

- 8 cameras at one elevation isn't a full hemisphere. Adding +/-30° elevations would tell us if top-down or under-views are systematically worse.
- 30k subsample is ~4% of the full 763k splat density. Full-density renders may shift absolute PSNR but the relative spread should be similar.
- All renders use Gaussian mode for both PLY and CRYPSOID — so the comparison is fair (same math path on both sides). The phoxoidal-path PSNR is a separate measurement (Tier 2 work, see `T2_audi_faithful_512.png`).

## Visual

![multiview chart](TIER_1.5_multiview_chart.png)

## Reproduce

```bash
python3 tools/multiview_cameras.py --n-azimuth 8 --elevations 8.0 \
    --distance 1.4 --fov 45 --out /tmp/mv_cams_8.json
python3 tools/tier2_multiview.py \
    --cameras /tmp/mv_cams_8.json \
    --out renders/crypsorender_v01/multiview \
    --size 256 --max-points 30000
# (run repeatedly until all 24 PNGs exist; each call resumes by skipping done frames)
```
