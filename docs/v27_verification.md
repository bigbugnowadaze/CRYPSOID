# v0.27 container verification — 2026-04-30

**File:** `recovery_v2/v27_attribute_group_sh_vq_render_container.3dphox`
**Size:** 18,796,089 bytes (17.93 MiB) — matches `RECOVERY_AUDIT.md`
**Magic:** `CRYPSOID27\0`
**Format string:** `CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER`
**Source splats (N):** 763,800
**Source PLY bytes:** 180,258,277

## All 7 chunks decode cleanly with matching CRC32

| # | Name | Compressed | Raw | CRC32 | Source |
|---|---|---:|---:|---|---|
| 0 | `tier_labels_u8` | 3,754 | 763,800 | OK | v25 |
| 1 | `xyz_u24_fixed` | 6,242,122 | 6,874,200 | OK | v25 |
| 2 | `dc_rgb_opacity_u8` | 2,547,608 | 3,055,200 | OK | v25 |
| 3 | `scale_f16` | 2,882,949 | 4,582,800 | OK | v25 |
| 4 | `quat_i16_norm4` | 5,755,963 | 6,110,400 | OK | v25 |
| 5 | `sh_vq128_idx_u8` | 1,356,120 | 2,291,400 | OK (new) | v27 |
| 6 | `sh_vq128_codebook_i8` | 3,920 | 5,760 | OK (new) | v27 |
| | **TOTAL** | **18,792,436** | **23,683,560** | | |

Chunks 0–4 are passed through from v25. Chunks 5–6 are the v27 SH product-VQ render core.

## Truth contract (carried in the manifest)

- Carried exact from v25: `tier_labels_u8`, `xyz_u24_fixed`, `dc_rgb_opacity_u8`, `scale_f16`, `quat_i16_norm4`
- Changed: `sh_rest_q8_global → sh_vq128_idx_u8 + sh_vq128_codebook_i8`
- Not claimed: lossless SH or final render parity

## SH VQ render core

3 product groups × 128 codewords × 15 int8 coefficients each, trained on a 5,000-row sample from the 763,800 splats.

| Group | RMSE (q8) | RMSE (float est.) | Max abs error (q8) |
|---|---:|---:|---:|
| 0 | 5.94 | 0.0413 | 104 |
| 1 | 5.88 | 0.0409 | 86 |
| 2 | 5.73 | 0.0398 | 96 |

(Float estimates use the v25 global SH scale `0.006946287755891094`.)

## Verdict

v27 anchor is intact and trustworthy. Safe to use as the spec source for the v25 build. Safe as the input baseline for v28/v29 reruns once v25 is regenerated.
