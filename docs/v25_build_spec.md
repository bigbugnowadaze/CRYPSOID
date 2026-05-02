# v0.25 Build Script — One-Page Spec

**Status:** DRAFT v2 — needs Bug's sign-off before any code is written.
**Audience:** the coding agent who will implement `tools/build_v25_attribute_group.py`.
**Source of truth:** reverse-engineered from `tools/build_v27_fast.py` and `tools/build_v28_sh_exact_correction.py`, both of which read v25 and tell us exactly what v25 must contain. Tier-labels logic now anchored on `inputs/v21_v22_artifacts/build_v22.py` and the v21/v22 CSVs.

**What we now have on disk (as of 2026-04-30):**
- `inputs/audi/Audi A5 Sportback.zip` → `scene.ply` (180,258,277 bytes, 763,800 vertices, standard 3DGS layout: x/y/z, scale_0..2, f_dc_0..2, opacity, rot_0..3, f_rest_0..44).
- `inputs/v21_v22_artifacts/v21_context_container_chunks.csv` (247 accepted v21 chunks at grid=32).
- `inputs/v21_v22_artifacts/v22_native_exact_promoted_chunks.csv` (483 v22 promoted chunks at grid=32, with a `tier` column = `native_exact_phoxoid`).
- `inputs/v21_v22_artifacts/build_v22.py` and `build_v22_native_burndown.py` — show exactly how Tier A/B/C are defined.
- `inputs/v21_v22_artifacts/build_v25_attribute_groups.py` — **stub only (113 bytes)**, not the real source. The actual v25 builder ran in the lost ChatGPT sandbox and was never saved out. Treat this file as a pointer to `PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json`, not as code.

---

## 1. What this script does

Take the original Audi A5 source PLY (~172 MB, ~763,800 splats) and produce two artifacts:

1. `outputs/v25_attribute_group_render_container.3dphox` — the honest full-attribute container that v27/v28/v29 all consume as input.
2. `reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json` — the companion metadata report v27/v28 read for `source_splats`, `source_ply_bytes`, and `v11_vq256_bytes`.

Nothing else. v25 is *not* the place to do VQ, residual sweeps, or render comparisons — those are v27/v28/v29.

---

## 2. Container file format (must match exactly)

Header, in order, big-endian-friendly little-endian integers:

| Bytes | Meaning |
|---|---|
| 11 | ASCII magic `CRYPSOID25\0` |
| 8 | `<Q` little-endian uint64 = manifest JSON byte length |
| N | Manifest JSON (UTF-8) |
| rest | Concatenated zlib-compressed chunk payloads in manifest order |

Manifest JSON top-level keys:

```
format         : "CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V25"
cycle          : "v0.25"
source_splats  : N (int, e.g. 763800)
source_ply_bytes : original PLY size in bytes (int)
chunks         : [chunk-entry, chunk-entry, ...]   # 6 entries, in order below
input          : { source_splats, source_ply_bytes, v11_vq256_bytes }   # mirrors top-level for downstream readers
truth_contract : { ... }   # plain-English statement of what's exact and what's not
```

Each chunk entry is:

```
{
  "name": "<chunk name>",
  "offset": <bytes from start of payload region, not from start of file>,
  "raw_bytes": <uncompressed length>,
  "compressed_bytes": <zlib-compressed length>,
  "crc32_raw": <CRC32 of UNCOMPRESSED bytes, as uint32>,
  "dtype": "<numpy dtype name>",
  "shape": [<dims>],
  "semantic": "<one-line description>"
  // plus any per-chunk extras listed below
}
```

Compression: `zlib.compress(raw, 6)` — same as v27.

---

## 3. The six chunks (must be in this exact order)

| # | Name | dtype | shape | raw bytes for N=763,800 | Per-chunk manifest extras |
|---|---|---|---|---:|---|
| 0 | `tier_labels_u8` | uint8 | [N] | 763,800 | none |
| 1 | `xyz_u24_fixed` | uint8 | [N, 9] | 6,874,200 | `bounds_min: [x, y, z]`, `bounds_max: [x, y, z]` (floats from PLY axis-aligned bounding box) |
| 2 | `dc_rgb_opacity_u8` | uint8 | [N, 4] | 3,055,200 | none |
| 3 | `scale_f16` | float16 | [N, 3] | 4,582,800 | none |
| 4 | `quat_i16_norm4` | int16 | [N, 4] | 6,110,400 | none |
| 5 | `sh_rest_q8_global` | int8 | [N, 45] | 34,371,000 | `global_scale: <float>` (the scalar that converts q8 back to float, default seen in Audi cycle: `0.006946287755891094`) |

Raw byte counts above MUST match what a v27 readback shows (already verified — see `recovery_v2/v27_attribute_group_sh_vq_render_container.3dphox`).

### How each chunk is encoded

**`tier_labels_u8`** — one byte per splat tagging which phoxoid tier this splat belongs to (A native render, B native exact, C splat-like, etc., per the v23 doctrine). **OPEN QUESTION — see §6.**

**`xyz_u24_fixed`** — for each splat, three 24-bit unsigned little-endian fixed-point integers packed as 9 bytes total (no padding). Encoding for one axis:
```
u24 = round( (xyz_axis - bounds_min[axis]) / (bounds_max[axis] - bounds_min[axis]) * (2**24 - 1) )
```
Decode is the inverse (see `decode_u24_xyz` in `build_v29_residual_transform_sweep.py`).

**`dc_rgb_opacity_u8`** — four bytes per splat: `[R8, G8, B8, opacity_u8]`. DC term of the splat color in 0..255 plus an 8-bit opacity. Encoding from the float values in the PLY:
```
R8 = clip( round(255 * sigmoid_or_linear(dc_r)), 0, 255 )    # confirm with PLY convention
opacity_u8 = clip( round(255 * sigmoid(opacity_logit)), 0, 255 )
```
**OPEN QUESTION — see §6** about whether DC RGB needs a sigmoid step or is already linear in the PLY.

**`scale_f16`** — three IEEE-754 binary16 values per splat, in PLY order. No quantization beyond float16.

**`quat_i16_norm4`** — four int16 values per splat representing the rotation quaternion, scaled so the unit quaternion `(1,0,0,0)` would encode as `(32767, 0, 0, 0)`. Encoding:
```
q_i16 = clip( round(q_float * 32767), -32768, 32767 )
```
The four-component form (no shortest-three trick).

**`sh_rest_q8_global`** — 45 int8 values per splat: the spherical-harmonic rest coefficients (degree 1–3, 15 coefficients × 3 RGB channels), quantized with a single global scale. Encoding:
```
sh_q8 = clip( round(sh_float / global_scale), -128, 127 ).astype(int8)
```
Pick `global_scale` so that the 99th-percentile absolute coefficient lands near 127. The Audi cycle landed on `0.006946287755891094`. Acceptance: rerunning v28's exact-archive readback against the new v25 must reconstruct the q8 SH stream exactly.

---

## 4. The companion report `PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json`

Minimum required structure (keys that v27/v28 actually read):

```
{
  "cycle": "v0.25",
  "input": {
    "source_splats": <int>,             // = N
    "source_ply_bytes": <int>,           // size of the source PLY zip or PLY file in bytes
    "v11_vq256_bytes": <int>             // size of the historical v11 VQ256 baseline container,
                                         // used only for size-comparison bars in v27/v28 reports.
                                         // If the v11 binary is unavailable, document the source
                                         // for this number (e.g. recovered from version_timeline.csv)
                                         // instead of guessing.
  },
  "outputs": { "container": "<path>" },
  "chunks": [ ... mirror of manifest chunk entries with sizes ... ],
  "truth_contract": "Honest q8 SH full-attribute container; no VQ, no correction chunks. Lossless against the q8/u24/f16/i16 quantization grid above; not lossless against the float32 PLY."
}
```

---

## 5. Reference values from the verified v27 container (do not guess these)

Use these to sanity-check the new v25 build:

```
N (source_splats):    763,800
source_ply_bytes:     180,258,277
xyz bounds:           in v25 chunk[1] manifest entry — preserved exactly into v27 chunk[1]
sh global_scale:      0.006946287755891094  (preserved in v25 chunk[5] manifest entry)
```

The build script must produce a v25 whose chunks 0–4 (everything except the SH stream) round-trip into v27 byte-identically when `build_v27_fast.py` is rerun against it.

---

## 6. Tier-labels derivation (resolved — for the coding agent)

`tier_labels_u8` is now derivable from the recovered v21/v22 CSVs. The doctrine, copied from `inputs/v21_v22_artifacts/build_v22.py`:

- **Tier A** (label = `0`) — splats whose grid-32 cell appears in `v21_context_container_chunks.csv` (247 accepted "native render phoxoid" cells, ~94,006 splats).
- **Tier B** (label = `1`) — splats whose grid-32 cell appears in `v22_native_exact_promoted_chunks.csv` *and* not in v21 (483 promoted "native exact phoxoid" cells).
- **Tier C** (label = `2`) — every remaining splat (the "fallback / VQ" tier).

To map each splat to its `cell_key` the coding agent must reproduce the v21 grid-32 spatial decomposition. The v21/v22 CSVs include `center_x/y/z` and the eigenvector spread per cell, but not the explicit `(splat_xyz) → cell_key` formula. Two acceptable approaches; pick whichever the coding agent can demonstrate against the v21 cell counts:

1. **Reverse-engineer from cell centers.** Compute the PLY's axis-aligned bounding box, divide each axis into 32 buckets, and define `cell_key = ix * 32*32 + iy * 32 + iz` (or whichever permutation reproduces the v21 cells). Acceptance: every cell_key in v21 must contain at least one splat, and the per-cell `count` columns must match the CSV within rounding.
2. **Nearest-center assignment.** For each splat, find the nearest v21 or v22 `center_x/y/z` and copy that cell's tier. Slower but doesn't require guessing the index permutation.

Bug should not have to choose between these — that's an implementation detail. The acceptance gate (per-cell count match against v21 CSV) tells the coding agent whether they got it right.

## 6b. Smaller open questions still worth confirming

1. **DC RGB encoding convention.** The Audi PLY stores `f_dc_0..2` as raw floats. Whether `dc_rgb_opacity_u8` applies the standard 3DGS sigmoid (`SH_C0 * f_dc + 0.5`, then clip to 0..255) or a linear scaling. Default assumption for the coding agent: **standard 3DGS sigmoid**, since the `f_dc_*` floats in the PLY are pre-sigmoid logits in the Inria/PlayCanvas convention. Same for `opacity` (sigmoid-then-quantize). Coding agent should verify against the v27 readback values.
2. **`v11_vq256_bytes`.** Default to `19,123,179` — that's the value the recovered `build_v22.py` uses as a constant. No need to find the v11 binary just for this number.
3. **Global SH scale.** Accept the recovered Audi value `0.006946287755891094` and write it into the chunk manifest. Do **not** recompute, since changing it would invalidate the v27 anchor we just verified.

---

## 7. Acceptance criteria (the coding agent must demonstrate all of these)

1. `outputs/v25_attribute_group_render_container.3dphox` exists, magic = `CRYPSOID25\0`.
2. Manifest decodes; all six chunks present, in the order in §3, with the exact names in §3.
3. CRC32 readback over every chunk passes.
4. `N = 763,800` and the per-chunk raw byte counts in §3 match exactly.
5. `xyz_u24_fixed` chunk entry contains `bounds_min` and `bounds_max`.
6. `sh_rest_q8_global` chunk entry contains `global_scale`.
7. `reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json` exists with the keys in §4.
8. **Round-trip gate:** `tools/build_v27_fast.py` runs to completion against the new v25 and produces a v27 container byte-identical to the verified `recovery_v2/v27_attribute_group_sh_vq_render_container.3dphox` for chunks 0–4 (the five non-SH chunks). Chunks 5–6 will differ in payload bytes (VQ is stochastic) but must still pass CRC.
9. **Truth gate:** the script does not claim float-lossless reconstruction. The truth contract string explicitly lists the quantization grid (q8 SH, u24 XYZ, f16 scale, i16 quat).

---

## 8. What the script must NOT do

- No SARC, no phoxoid replacement, no render parity claim.
- No VQ or residual processing — that is v27 and v28's job.
- No silent fallbacks that swallow errors. If the PLY is the wrong size or shape, fail loudly.
- No hardcoded `/mnt/data/...` paths — accept `--input-ply <path>` and `--output-root <path>`.

---

## 9. Path conventions

```
--input-ply       path to the Audi A5 PLY (or zip containing one PLY)
--output-root     directory to write outputs/ and reports/ subfolders into
```

Default output layout:
```
<output-root>/outputs/v25_attribute_group_render_container.3dphox
<output-root>/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json
<output-root>/tools/build_v25_attribute_group.py   # script copies itself here
```
