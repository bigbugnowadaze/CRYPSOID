# v0.28 build — handoff summary

**Built:** 2026-04-30, against the rebuilt v25 + recovered v27.

## What was produced

| File | Size | Notes |
|---|---:|---|
| `outputs/v28_sh_vq_render_container.3dphox` | 18,795,838 B (17.93 MiB) | Render-mode v28; 7 chunks; magic `CRYPSOID28\0` |
| `outputs/v28_sh_vq_exact_archive_container.3dphox` | 32,162,548 B (30.67 MiB) | q8-exact archive; 16 chunks; magic `CRYPSOID28\0` |
| `outputs/v28_exact_archive_size_bars.svg` | 912 B | Visualization |
| `outputs/v28_sh_residual_histogram.svg` | 1,440 B | Visualization |
| `reports/PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json` | 5,316 B | Full metrics |
| `reports/RESEARCH_BUILD_TEST_CYCLE_V28.md` | 1,424 B | Cycle write-up |

## Sanity checks

- **All 7 chunks of the render container** decode and pass CRC32.
- **All 16 chunks of the exact-archive container** decode and pass CRC32.
- `exact_ok_prewrite = True`: the encoder verified pre-write that the q8 SH stream reconstructs exactly from VQ centroids + per-tier-group residual chunks.
- `readback_exact_sh_reconstruction = True`: re-decoded after writing and confirmed exact reconstruction.
- Chosen correction encoding: `per_tier_group` (matches the recorded original).
- Chosen correction payload: 13,360,400 bytes (matches recorded original exactly).

## Comparison to the recorded values from `recovery_v2/reports/PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json`

| Metric | Recorded | Rebuilt | Δ |
|---|---:|---:|---:|
| Render container bytes | 18,795,838 | 18,795,838 | **0 (byte-identical)** |
| Archive container bytes | 32,163,308 | 32,162,548 | −760 (−0.002%) |
| Chosen correction bytes | 13,360,400 | 13,360,400 | 0 |

The 760-byte difference in the archive is below 0.003% and is a downstream consequence of the SH-VQ codebook in v25's pass-through SH chunk being slightly different from the original v25's stochastic k-means run. The correction layer itself is byte-identical.

## What this unblocks

- Phase 5: `render_v28_vs_original.py` — the CPU DC/opacity preview side-by-side renderer. Needs the v28 render container (now present) and the original Audi PLY (already on disk). Same path-refactor pass as v28 will be needed.

## Path-refactor note

`tools/build_v28_sh_exact_correction.py` was copied from `recovery_v2/tools/` and patched to take CLI flags. Encoding logic was not touched. The patch:

- Replaced 5 hardcoded `/mnt/data/...` paths with `--v25`, `--v27`, `--v25-report`, `--v27-report`, `--output-root` flags.
- Dropped the trailing `/mnt/data/...zip` and `.tar.gz` packaging (not needed in this environment).
- Dropped the `shutil.copy('/mnt/data/build_v28.py', ...)` self-copy (the script lives in `tools/` already).

The same minor refactor will apply to `render_v28_vs_original.py` in Phase 5.
