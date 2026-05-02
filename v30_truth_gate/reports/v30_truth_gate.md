# CRYPSOID v0.30 Render Truth Gate

**Generated:** v30_truth_gate

## Summary

This gate compares the original Audi PLY against the v0.28 decoded container using a CPU DC/opacity point preview renderer.
All panels use the same camera perspective (yaw 35°, pitch 18°, distance 2.4, FOV 42°).

## Image Metrics (v28 vs original)

| Metric | Value |
|---|---|
| MSE | 0.000003 |
| MAE | 0.000138 |
| PSNR (dB) | 54.57 |
| SSIM | 0.9999 |

## Decoding Timing (seconds)

| Step | Time |
|---|---|
| Read container | 0.2128 |
| Decompress chunks | 0.0639 |
| Decode XYZ (u24) | 0.0453 |
| Decode DC/RGB/opacity (u8) | 0.0093 |
| Decode tier labels (u8) | 0.0000 |
| **Total** | **0.3314** |

## Rendering Timing (seconds)

| Task | Time |
|---|---|
| Render original | 0.2823 |
| Render v28 | 0.2480 |
| Render tier view | 0.5881 |

## Tier Distribution

v28 splat assignments by tier (from tier_labels_u8 chunk):

| Tier | Count | Percent |
|---|---:|---|
| A (native) | 94,006 | 12.3% |
| B (native) | 144,271 | 18.9% |
| C (native) | 525,523 | 68.8% |

## Attribute Parity (v25 → v28 passthrough)

v0.28 passes these five chunks through unchanged from v25. All must be byte-identical.

| Chunk | Byte-identical | Differing bytes |
|---|---|---|
| tier_labels_u8 | True | 0 |
| xyz_u24_fixed | True | 0 |
| dc_rgb_opacity_u8 | True | 0 |
| scale_f16 | True | 0 |
| quat_i16_norm4 | True | 0 |

## Visual Comparison

![Contact sheet](../renders/v30_contact_sheet.png)

**Top-left:** Original PLY rendered as DC + opacity dots (763,800 splats).
**Top-right:** v0.28 decoded container rendered the same way (763,800 splats).
**Bottom-left:** Per-pixel absolute-difference heatmap (blue = 0, red = 54.57).
**Bottom-right:** Per-splat tier visualization (Red=Tier A, Green=Tier B, Blue=Tier C).

## Truth Note

This is a CPU DC/opacity point preview renderer. It uses only the diffuse (DC) color and opacity channels; SH bands 1-3 are not exercised. It draws screen-space dots, not anisotropic Gaussians. It is meant to catch gross geometry/color errors before the full viewer path. It is NOT final visual truth. SSIM computed with skimage's structural_similarity (window size 11 by default) on 0-1 normalized RGB.

---

**Files:**
- Contact sheet: renders/v30_contact_sheet.png
- Error heatmap: renders/v30_error_heatmap.png
- Tier view: renders/v30_tier_view.png
- Full metrics: reports/v30_truth_gate.json
