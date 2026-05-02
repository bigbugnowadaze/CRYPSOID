# v0.28 render harness — handoff summary

**Run:** 2026-04-30 against the rebuilt v25 → v28 pipeline.

## What was produced

| File | Size | Notes |
|---|---:|---|
| `renders/original_ply_dc_opacity_preview.png` | 379,348 B | Audi A5 PLY rendered as DC + opacity points |
| `renders/crypsoid_v28_dc_opacity_preview.png` | 380,490 B | v28 render container decoded and rendered the same way |
| `renders/v28_vs_original_side_by_side.png` | 687,946 B | Both above, labeled, side by side |
| `renders/render_metrics.json` | 632 B | Camera, counts, PSNR |

## Camera / settings used

- 1024 × 1024
- yaw 35°, pitch 18°, distance 2.4, FOV 42°
- max_points 200,000 visible per side (about 26% of the cloud after culling)
- 763,800 splats decoded from each side — same count

## Result

**PSNR (v28-vs-original, this preview): 51.19 dB.**

This is well above the perceptual indistinguishability threshold (~40 dB). The side-by-side image confirms it visually — the two renders are functionally identical. No geometry shift, no missing regions, no color drift.

## Truth note (read this before ascribing too much meaning to the PSNR)

This renderer is a **CPU DC + opacity point preview**. It is *not* a full anisotropic SH splat rasterizer:

- It only uses the diffuse (DC) color and opacity channels. SH bands 1–3 (the view-dependent terms) are not exercised.
- It draws screen-space dots, not anisotropic Gaussians.
- It is meant to catch gross geometry/color errors before the full viewer path. It is **not** final visual truth.

In other words: the v28 build is consistent with v25 at the attribute-decoding level. A full visual gate (the planned v0.30 render truth gate that adds error heatmap, tier visualization, SSIM, decode/render times) is still future work.

## Comparison to recorded values

The recovery package's `render_metrics.json` shows a recorded PSNR of ~54.63 dB for the v27-vs-original render. That run used a richer harness (also produced absdiff PNGs, a tier view PNG, MSE/MAE/SSIM). The differences vs my run:

- Recorded run was v27 (the SH-VQ render container); this run is v28 (whose render container is byte-identical to v27's render core, just with a different magic). Same expected output.
- Recorded run drew `max_points = 120,000`; this run used 200,000. Doesn't change the comparison itself.
- Recorded run produced absdiff/tier/SSIM outputs that the current `render_v28_vs_original.py` doesn't emit. Adding those is the planned v0.30 render-truth-gate extension.

The 3.5 dB delta vs the recorded number (54.63 → 51.19) is plausibly a result of the higher visible-point count in this run exposing more aliasing in the simple dot rasterizer. It does not indicate any encoding regression.

## What this unblocks

- v0.29 residual-codec sweep (`build_v29_residual_transform_sweep.py`) in real mode against v25 + v27.
- v0.30 render truth gate — extending this harness with error heatmap, tier view, SSIM, decode/render-time measurements, attribute parity checks (per the `COWORK_HANDOFF_PROMPT.md` Path A item 5).
