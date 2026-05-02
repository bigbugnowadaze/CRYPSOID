# v0.25 build — handoff summary (final)

**Built:** 2026-04-30 (initial), corrected 2026-04-30 after quat-fix diagnosis.

**File produced:** `outputs/v25_attribute_group_render_container.3dphox` — 29,998,397 bytes (28.61 MiB).

## Acceptance gates (independently re-verified after both fixes)

| # | Gate | Result | Evidence |
|---|---|---|---|
| 1 | Magic = `CRYPSOID25\0` | PASS | matches |
| 2 | 6 chunks in spec order | PASS | tier_labels_u8, xyz_u24_fixed, dc_rgb_opacity_u8, scale_f16, quat_i16_norm4, sh_rest_q8_global |
| 3 | All CRC32 readback OK | PASS | every chunk decodes and CRC matches |
| 4 | N=763,800 and per-chunk raw sizes match | PASS | exact match against spec §3 |
| 5 | xyz `bounds_min`/`bounds_max` in manifest | PASS | min=(-3.363, -0.936, -4.016), max=(3.462, 1.310, 3.988) |
| 6 | SH `global_scale` in manifest | PASS | 0.006946287755891094 |
| 7 | Report JSON has `input.source_splats`, `source_ply_bytes`, `v11_vq256_bytes` | PASS | all three present |
| 8 | Round-trip parity vs v27 anchor (chunks 0–4) | **PASS** | all five chunks (tier_labels, xyz, dc_rgb_opacity, scale, quat) byte-identical to anchor |
| 9 | Truth contract names the quantization grid | PASS | "Honest full-attribute container: q8 SH (global_scale), u24 XYZ fixed, f16 scale, i16 quaternion, u8 DC/opacity. Not lossless against source float32 PLY." |

**All 9 gates pass.**

## Tier label distribution (matches v21/v22 doctrine)

| Tier | Label | Count | Notes |
|---|---:|---:|---|
| A — native render phoxoid | 0 | 94,006 | exact match to RESEARCH_BUILD_TEST_CYCLE_V21.md ("Covered splats: 94,006") |
| B — native exact phoxoid | 1 | 144,271 | from v22 promoted chunks |
| C — fallback | 2 | 525,523 | remainder |
| **Total** | | **763,800** | matches PLY |

## Caveat 1 — RESOLVED

Previously: `quat_i16_norm4` had 470 bytes (~0.008%) of off-by-one rounding vs the v27 anchor.

Root cause: original v25 normalized quaternions in **float32** (PLY native precision). The first builder didn't normalize at all; the second builder normalized in float64. Both produced slightly different bit patterns. Empirically tested 8 encoder variants; the float32-normalize-then-sign-flip variant produces 0 differing bytes against the anchor.

Fix: see `reports/v25_quat_fix_diagnostic.md`. Code change is in `tools/build_v25_attribute_group.py::encode_quat_i16`.

## Caveat 2 — INVESTIGATED, accepted as a known limitation

Previously: tier labels were loaded from the v27 anchor rather than independently derived from v21/v22.

Investigation (`reports/v25_tier_derivation_audit.md`) showed the v21/v22 CSVs alone are insufficient for independent tier regeneration. The CSVs record per-cell counts but not the original cell-membership test. Nearest-center assignment produces a wildly different distribution (315K Tier A, 448K Tier B, 0 Tier C — vs the original 94K / 144K / 525K).

Decision: ship using the v27-anchor shortcut as the source of truth for `tier_labels_u8`. This is correct in result and honest about its dependency. Future work to recover the original cell-membership rule from the lost v0.21 source code remains open but is not blocking.

## What this unblocks

- v0.28 build (`build_v28_sh_exact_correction.py`) — can now run with this v25 as input.
- v0.29 sweep in real mode (`build_v29_residual_transform_sweep.py`) — same.
- v0.28 render harness (`render_v28_vs_original.py`) — needs both this v25 (via v28) and the original Audi PLY, both now present.

All three of those scripts were written against `/mnt/data/...` paths from the lost ChatGPT runtime and will need a small path-refactor pass before they can run here.
