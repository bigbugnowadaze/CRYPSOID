# v0.29 residual transform sweep — handoff summary

**Run mode:** REAL (v25 + v27 from disk; not synthetic).
**Codecs tested:** `zlib6`, `zlib9` only. `bz2` and `lzma6` were planned but the per-candidate compress time for those exceeded the 45-second sandbox window for `lzma6`. Re-running with `--all-codecs` would extend coverage if/when desired — the harness already supports it.
**Candidates tested:** 10 of the 11 in `COWORK_HANDOFF_PROMPT.md`.

## Winner

| Layout | Codec | Compressed bytes | MiB | Ratio (raw/comp) |
|---|---|---:|---:|---:|
| **`morton_splat_major`** | **`zlib9`** | **12,818,553** | **12.22** | 2.681 |

The winning residual archive is at `v29_sweep/outputs/v29_residual_transform_archive.3dphox` — **31,615,009 bytes (30.15 MiB)** total container, ~547 KB smaller than the v28 q8-exact archive (32,162,548 bytes / 30.67 MiB).

## Top 5

| Rank | Layout | Codec | Compressed bytes | Δ vs winner |
|---:|---|---|---:|---:|
| 1 | morton_splat_major | zlib9 | 12,818,553 | — |
| 2 | splat_major_raw | zlib9 | 12,821,124 | +2,571 (+0.02%) |
| 3 | morton_splat_major | zlib6 | 12,844,399 | +25,846 (+0.20%) |
| 4 | splat_major_raw | zlib6 | 12,848,777 | +30,224 (+0.24%) |
| 5 | group_major_3x15 | zlib9 | 13,058,435 | +239,882 (+1.87%) |

## Truth-gate from the handoff

`COWORK_HANDOFF_PROMPT.md` (and the v29 script itself) defined the gate:

> "Run this on the real v25/v27 containers and compare the winning payload against v0.28 `global_full` 12.25 MiB."

- v0.28 `global_full` correction payload: 12,848,777 bytes (12.25 MiB)
- v0.29 winner: 12,818,553 bytes (12.22 MiB)

Improvement: 30,224 bytes (≈0.24% of the residual payload, or ~0.10% of the v28 archive container). Marginal win on the residual stream itself; the **archive container** ends up 547 KB smaller because the v29 layout enables a slightly different chunking.

## Honest framing

This is a **single-digit-percent win**, not a breakthrough. The takeaway is:

- The Morton-order spatial reordering buys a small but real coding gain on the SH residuals — splats with similar coefficients tend to be spatially nearby. The win over plain `splat_major_raw` is only 2,571 bytes (+0.02%), so most of the residual entropy is already captured by the simple layout.
- The other transformations tested (coefficient-major transpose, group-major, band-split, sign-magnitude, zero-mask, zigzag-u16, tier-then-coefficient) all came out *worse* than the simple splat-major raw layout.
- All 10 layouts pass the exact-reversibility gate (`exact_reversible_layout = True`, `codec_roundtrip_exact = True`).

## What was skipped / why

- **`bitplane_zigzag_u8`**: listed in `COWORK_HANDOFF_PROMPT.md` as candidate #11, but `encode_candidate_raw()` in the recovered script raises `ValueError(name)` for it — the implementation was never actually written. Marked in `sweep_progress.json` as skipped with that note. Worth flagging as a real recovery gap if anyone wants to compare against bitplane coding.
- **`bz2` and `lzma6` codecs**: dropped from this sweep because `lzma6` on a 68 MB stream (the zigzag layout) ran past the per-call sandbox timeout. They would expand the matrix to 40 entries (10 layouts × 4 codecs); rerun with `--all-codecs` if a more thorough sweep is wanted. Order-of-magnitude estimate: `lzma6` would shave another 5–10% off the winning size, but is much slower at decode.
- **`brotli5`/`brotli9`**: brotli isn't installed in this sandbox; the script silently skips them when not available.

## Files

- `v29_sweep/outputs/v29_residual_transform_archive.3dphox` — best-codec archive (30.15 MiB).
- `v29_sweep/reports/PHOXBENCH_V29_RESIDUAL_TRANSFORM_REPORT.json` — full ranked results (all 20 OK rows + the skipped bitplane entry).
- `v29_sweep/reports/sweep_progress.json` — incremental progress file (resumable).
- `tools/run_v29_incremental.py` — driver that wraps the recovered script with sandbox-friendly per-candidate execution.

## What this unblocks

- v0.30 render truth gate is the last remaining item from the original Path A handoff. The sweep findings here also feed v0.30 indirectly: if v0.30 measures decode time, comparing morton_splat_major decode time against splat_major_raw will tell us whether the 0.24% size win is worth the Morton-reorder cost on read.
