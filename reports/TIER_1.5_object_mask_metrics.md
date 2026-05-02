# Tier 1.5 item 4 — Object-mask metrics

**Why this matters.** SSIM rewards structural agreement, including agreement on background pixels. When most of the frame is constant-black background and both renders trivially agree there, full-frame SSIM is artificially inflated. The honest number is to **mask out the background and recompute over object pixels only.** Same logic for PSNR — averaging squared error over all the near-zero background pixels drags total MSE down and inflates the reported dB.

Reference: `renders/crypsorender_v01/ply_200k_side.png` (512×512). Object mask = pixels with luma > 0.02. Coverage: **38.8%** of the frame is object.

## Comparisons (camera-aligned)

| Pair | Full-frame PSNR / SSIM | Object-only PSNR / SSIM | SSIM inflation from background | PSNR change when masked |
|---|---|---|---:|---:|
| PLY vs v28 EXACT archive | 56.33 dB / 0.99957 | 52.23 dB / 0.99936 | +0.00021 | -4.10 dB |
| PLY vs v28 VQ render | 38.76 dB / 0.99226 | 34.67 dB / 0.98862 | +0.00364 | -4.09 dB |

## Reading the numbers

- **The object covers only 38.8% of the frame.** The other 61.2% is background where both renders trivially agree on "black." That background is what was inflating the original SSIM.
- **SSIM background inflation** column tells you how much SSIM was being given for free by the background. A value of +0.0xxx means the original full-frame SSIM was overstating quality by that many points.
- **PSNR usually drops when restricted to object pixels.** Removing the easy near-zero-error background raises mean MSE and lowers dB. The masked PSNR is the *real* reconstruction quality on the parts of the image that matter.
- The masked metrics are the honest comparison to use against any baseline.

## Where to use these

- README headline numbers should reference object-only values, not full-frame.
- `manifest_T1.json` PSNR/SSIM section should be updated to add the object-only row alongside the full-frame row.
- Future PhoxBench Tier 2+ runs should compute masked metrics by default (auto-derive mask from reference brightness with the same 0.02 threshold).

## Reproduce

```bash
# This script lives at tools/compute_object_mask_metrics.py (to be created)
python3 -c "$(cat reports/TIER_1.5_object_mask_metrics.md | sed -n '/```python/,/```/p')"
```
