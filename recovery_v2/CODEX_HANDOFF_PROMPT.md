# Codex Handoff Prompt — CRYPSOID continuation (v2)

Paste everything below this line into Codex after attaching this package and the Audi A5 PLY zip.

---

## Context

You are continuing the CRYPSOID `.3dphox` Gaussian-splat compression project after a ChatGPT chat-loss recovery. Read `RECOVERY_AUDIT.md` first; it covers what was recovered. The previous audit told the next agent to rebuild missing scripts from spec — that is no longer accurate. The build and render scripts have been recovered. Your job is to extend the codebase, not rebuild it.

## Inputs

In this package:

- `v27_attribute_group_sh_vq_render_container.3dphox` — 17.93 MiB, the latest verified render container (763,800 splats, all 7 chunks CRC-verified)
- `tools/build_v27_fast.py` — wrote the above. Reads v25. Documents v25's chunk layout in code (see lines that call `dec25(...)`).
- `tools/build_v28_sh_exact_correction.py` — full v0.28 build. Reads v25 and v27, writes v28 render core + q8-exact archive container. Documents the v28 magic (`b"CRYPSOID28\0"`) and per-tier-group residual chunk layout.
- `tools/build_v29_residual_transform_sweep.py` — full v0.29 residual sweep. 11 candidate layouts × 4 codecs (zlib/bz2/lzma/optionally brotli). Has both real-input and synthetic modes. Magic: `b"CRYPSOID29\0"`.
- `tools/render_v28_vs_original.py` — CPU DC/opacity preview renderer; produces side-by-side contact sheet and metrics JSON. Uses camera (yaw=35, pitch=18, distance=2.4, fov=42, size=1024).
- `reports/PHOXBENCH_V27_SH_DEBT_REPORT.json` — v27 metrics
- `reports/PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json` — v28 metrics, including all four correction encoding variants tested and confirmed q8-exact reconstruction

Provided separately:

- Audi A5 source PLY zip (~172 MB). Inside: `scene.ply` (binary little-endian PLY, 763,800 vertices, 59 float32 properties: x y z scale_0..2 f_dc_0..2 opacity rot_0..3 f_rest_0..44).

Not in this package, but the build scripts read them:

- `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/outputs/v25_attribute_group_render_container.3dphox`
- `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json`

## Path note — do this first

All recovered scripts hardcode `/mnt/data/...` paths. Either:

(a) Mount inputs at those paths in your environment, or
(b) Refactor each script to accept `--input-root` / `--output-root` flags. About 10 lines of edits per script. This is option I'd recommend if you have multiple environments.

The dependency graph between scripts:

```
Audi PLY  ─►  (NEED: v25 build script)  ─►  v25.3dphox + v25 report
                                              │
                              ┌───────────────┼─────────────────────────┐
                              ▼                                          ▼
                       build_v27_fast.py                    build_v28_sh_exact_correction.py
                              │                                          │
                              ▼                                          ▼
                       v27.3dphox                             v28_render.3dphox + v28_archive.3dphox
                              │                                          │
                              └────────────┬─────────────────────────────┘
                                           ▼
                              build_v29_residual_transform_sweep.py
                                           │
                                           ▼
                                v29_archive.3dphox + sweep results JSON

  Audi PLY + v27 (or v28) ─► render_v28_vs_original.py ─► contact sheet + metrics JSON
```

## Tasks, in priority order

### Task 1 — Write `tools/build_v25_attribute_group.py`

The v25 build script was not recovered. Write it. Reverse-engineer v25's chunk layout from `tools/build_v27_fast.py` (which decodes v25 chunks) and from the v25 chunk references in `tools/build_v28_sh_exact_correction.py`.

**v25 must produce a `.3dphox` with:**
- Magic: `b"CRYPSOID25\0"` (11 bytes)
- 8-byte little-endian uint64 manifest length, then JSON manifest (UTF-8), then concatenated zlib-compressed chunk payloads
- The manifest must include an `input` block matching what `build_v27_fast.py` reads: `source_splats` (763800), `source_ply_bytes` (180258277), `v11_vq256_bytes` (19123179)
- Chunks (in this order):
  - `tier_labels_u8` — uint8[763800]. Tier classification per splat. v25 inherited this from earlier cycles; for a clean regeneration, classify all splats as tier 2 (exact splat-stream debt) by default unless you implement tier A/B logic. The v27 report shows the actual distribution: tier 0 = 94,006, tier 1 = 144,271, tier 2 = 525,523. If you don't implement the phoxoid clustering that produced those tiers, label that explicitly in the manifest and proceed with all-tier-2.
  - `xyz_u24_fixed` — uint8[763800, 9]. Per-axis 24-bit fixed-point quantization. Manifest must carry `bounds_min` and `bounds_max` (3-element float lists). See `decode_u24_xyz` in `tools/render_v28_vs_original.py` for the dequantize formula; invert it for the encoder.
  - `dc_rgb_opacity_u8` — uint8[763800, 4]. Apply `clip(f_dc * 0.28209479177387814 + 0.5, 0, 1) * 255` for RGB and `sigmoid(opacity) * 255` for opacity. (`C0 = 0.28209479177387814` is the SH-band-0 normalization constant; see `render_v28_vs_original.py`.)
  - `scale_f16` — float16[763800, 3]. Cast scale_0..2 to float16.
  - `quat_i16_norm4` — int16[763800, 4]. Multiply normalized rotations by 32767 and round.
  - `sh_rest_q8_global` — int8[763800, 45]. Quantize f_rest_0..44 to int8 with a single global scale; the manifest entry must include `global_scale` (the v27 build expects this; default in that code is `0.006946287755891094`). Use the same scale to be compatible with the recovered codebase.
- Each chunk dict must include: `name`, `offset`, `raw_bytes`, `compressed_bytes`, `crc32_raw`, `dtype`, `shape`, plus `bounds_min`/`bounds_max` for `xyz_u24_fixed` and `global_scale` for `sh_rest_q8_global`.
- Write `reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json` with the schema `{"input": {"source_splats": ..., "source_ply_bytes": ..., "v11_vq256_bytes": ...}, ...}` so `build_v27_fast.py` can read it.
- Verify CRC readback for all chunks before exiting.

**Acceptance:** Running `build_v27_fast.py` after your v25 must produce a v27 container byte-identical (or chunk-CRC-identical) to `v27_attribute_group_sh_vq_render_container.3dphox` already in this package. If not byte-identical, the chunk CRCs of the non-SH chunks (tier_labels, xyz, dc, scale, quat) must match.

### Task 2 — Refactor scripts for portable paths

Add `--input-root` and `--output-root` argparse args to all four build scripts and the renderer. Default to `/mnt/data` for backward compatibility. This makes the codebase usable outside the original ChatGPT runtime.

### Task 3 — Run the existing pipeline end-to-end

Once Task 1 is done:

```bash
python tools/build_v25_attribute_group.py --audi-ply <path>
python tools/build_v27_fast.py
python tools/build_v28_sh_exact_correction.py
python tools/render_v28_vs_original.py --original <audi> --v28 <v28 render container>
python tools/build_v29_residual_transform_sweep.py  # real mode now that v25 exists
```

Verify:
- v27 byte-identical to the recovered v27 (or chunk CRCs match)
- v28 render and archive sizes match `PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json`
- v28 archive readback achieves exact q8 SH reconstruction
- v29 sweep emits a results table comparing all 11 layouts × 4 codecs

### Task 4 — Build v0.30 render truth gate

Extend `tools/render_v28_vs_original.py` into `tools/render_v30_truth_gate.py`. Additions:

- Output `error_heatmap.png` (per-pixel absolute difference, contrast-stretched)
- Output `tier_view.png` (v27/v28 splats colored by `tier_labels_u8`: tier 0 red, tier 1 green, tier 2 blue)
- Add SSIM (use `scikit-image`)
- Add decode-time and render-time measurements to `render_metrics.json`
- Add attribute parity checks: for each chunk, compare decoded v27/v28 to v25 (where applicable), report match/mismatch
- Self-label the renderer in JSON output as a "CPU DC/opacity preview, not anisotropic SH-aware"

**Acceptance gates for v0.30:**
- All chunk CRC32s match for v27 and v28 containers
- Splat counts are 763,800 across original / v27 / v28
- PSNR within ±2 dB of 54.63 (the existing reference) for v27 vs original
- SSIM > 0.99
- Renderer string in JSON honestly labels itself as a sanity preview

### Hard rules — do not relax

- Do not invent new compression formats or new primitive types in this work. The v0.30 task is a measuring instrument; it is not a result.
- Do not promote SARC/phoxoid replacement as the primary render path.
- Do not claim wins without listing carried attribute groups.
- Splatpack/native exact remains master/fallback/parity.
- The CPU preview is not visual truth. Browser/WebGPU parity is Phase 3 in the documented plan; do not skip ahead.

## Dependencies

```
numpy
Pillow
scikit-image  # SSIM
scikit-learn  # MiniBatchKMeans, used by build_v27_fast.py
```

Optional: `brotli` for the v29 sweep's brotli codec (the script handles its absence gracefully).

No GPU / torch / cuda required.

## After Task 4

Stop and check in with Bug. The next phase per `docs/CRYPSOID_V29_PHASE_PLAN.md` Phase 2 is using the v0.30 truth gate to gate residual codec promotions in a real v0.29 sweep. Don't start that without confirming with Bug.
