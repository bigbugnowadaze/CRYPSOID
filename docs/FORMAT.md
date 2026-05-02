# `.3dphox` — file format reference

Canonical description of the CRYPSOID `.3dphox` file format. If you want to
write a third-party `.3dphox` reader or writer, this is the spec.

## High-level structure

A `.3dphox` file is a single binary blob:

```
+----------------------------------------+  byte offset
| Magic (11 bytes ASCII + NUL)           |  0
+----------------------------------------+
| Manifest length (uint64 LE, 8 bytes)   |  11
+----------------------------------------+
| Manifest JSON (UTF-8, N bytes)         |  19
+----------------------------------------+
| Chunk payload region (concatenated     |  19 + N
|   zlib-compressed chunks)              |
+----------------------------------------+
```

All multi-byte numeric fields are little-endian.

### Magic

11 bytes (10 ASCII characters + NUL terminator). Identifies the format version:

| Magic | Format family |
|---|---|
| `CRYPSOID25\0` | v0.25 attribute-group container (full-attribute, q8 SH) |
| `CRYPSOID27\0` | v0.27 SH-VQ render container (curvature-aware SH) |
| `CRYPSOID28\0` | v0.28 — render OR EXACT archive (disambiguated by manifest `format`) |
| `CRYPSOID30\0` | reserved for v0.30 (not yet shipped) |
| `CRYPSOID40\0` | reserved for v0.40 (native germ chunks) |

### Manifest

Top-level JSON object with at minimum:

```jsonc
{
  "format": "CRYPSOID_3DPHOX_<descriptive_name>",   // see table below
  "cycle":  "v0.28",
  "source_splats": 763800,                          // canonical N
  "source_ply_bytes": 180258277,                    // for round-trip metadata
  "chunks": [                                       // ordered chunk index
    {
      "name": "tier_labels_u8",
      "offset": 0,
      "compressed_bytes": 3754,
      "raw_bytes": 763800,
      "crc32_raw": 1709369810,
      "dtype": "uint8",
      "shape": [763800],
      "semantic": "tier labels: 0=A native render, 1=B native exact, 2=C fallback"
      // ... per-chunk extras (e.g. bounds_min/max for xyz, global_scale for SH)
    },
    /* ... more chunks ... */
  ],
  "truth_contract": "Honest full-attribute container: ... explains what's lossy"
  // plus format-specific extras (sh_vq summary, correction_encoding, etc.)
}
```

### Chunk payload region

Each chunk is **zlib-compressed** (`zlib.compress(raw, level=6)` is the
typical writer setting; level isn't part of the format — readers just
`zlib.decompress`).

Chunks are concatenated in the order they appear in the `manifest.chunks`
array. `offset` and `compressed_bytes` in the manifest reference the chunk
payload region's local coordinate (so chunk 0 is at offset 0 *within the
payload region*, not within the file).

To read chunk `i`:
```
chunk_bytes = blob[manifest.chunks[i].offset : manifest.chunks[i].offset + manifest.chunks[i].compressed_bytes]
raw         = zlib.decompress(chunk_bytes)
assert crc32(raw) == manifest.chunks[i].crc32_raw      # integrity check
```

## Format families

| Magic | Format string | Required chunks | Optional chunks | Used for |
|---|---|---|---|---|
| `CRYPSOID25\0` | `CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V25` | tier_labels_u8, xyz_u24_fixed, dc_rgb_opacity_u8, scale_f16, quat_i16_norm4, sh_rest_q8_global | — | "honest" container, q8 SH |
| `CRYPSOID27\0` | `CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER` | tier_labels_u8, xyz_u24_fixed, dc_rgb_opacity_u8, scale_f16, quat_i16_norm4, sh_vq128_idx_u8, sh_vq128_codebook_i8 | — | smaller, lossy SH (VQ approx) |
| `CRYPSOID28\0` (render) | `CRYPSOID_3DPHOX_V28_SH_VQ_RENDER_CORE` | same as v27 | — | identical layout to v27, distinct magic |
| `CRYPSOID28\0` (archive) | `CRYPSOID_3DPHOX_V28_SH_VQ_EXACT_ARCHIVE` | v27 chunks + 9 sh_exact_residual_t{0..2}_g{0..2}_int8 | — | bit-exact reconstruction of v25 q8 SH |

A reader should dispatch on (magic, manifest.format) to decide which path
to take.

## Per-chunk encoding

### `tier_labels_u8` — (n,) uint8

One byte per splat. Values: `0` = Tier A (native render phoxoid), `1` = Tier B
(native exact phoxoid), `2` = Tier C (Gaussian fallback).

Tier semantics come from the v0.21–v0.23 doctrine — see `docs/thesis_digest.md`
§5 and `inputs/v21_v22_artifacts/`.

### `xyz_u24_fixed` — (n, 3) packed u24 little-endian

Three 24-bit unsigned integers per splat (9 bytes per splat). Manifest entry
must include `bounds_min` and `bounds_max` (each a 3-element float array).

Encoding:
```
u24 = round((xyz_axis - bounds_min[axis]) / (bounds_max[axis] - bounds_min[axis]) * (2**24 - 1))
```

Decoding:
```
xyz_axis = (u24 / (2**24 - 1)) * (bounds_max[axis] - bounds_min[axis]) + bounds_min[axis]
```

### `dc_rgb_opacity_u8` — (n, 4) uint8

Four bytes per splat: `[R8, G8, B8, opacity8]`.

The values are post-sigmoid in [0, 255]. To convert to/from raw 3DGS PLY
floats (`f_dc_*`, `opacity`):

```
SH_C0 = 0.28209479177387814
R8 = clip(round((SH_C0 * f_dc_0 + 0.5) * 255), 0, 255)
opacity8 = clip(round(sigmoid(opacity_logit) * 255), 0, 255)
```

### `scale_f16` — (n, 3) IEEE-754 binary16

Three half-floats per splat (6 bytes). Values are log-space scales (3DGS
convention) — recover linear sigma with `exp(scale_f16)`.

### `quat_i16_norm4` — (n, 4) int16

Four signed 16-bit integers per splat (8 bytes). The unit quaternion has
been pre-normalized in **float32** (precision matters — see
`reports/v25_quat_fix_diagnostic.md`) and scaled by 32767, with a sign-flip
canonicalization rule:

```
q_i16 = clip(round(q_normalized_float32 * 32767), -32768, 32767)
if q_i16[0] < 0: q_i16 *= -1            # sign-flip
```

### `sh_rest_q8_global` — (n, 45) int8 (v25 only)

45 signed-int8 SH coefficients per splat (degrees 1–3, 15 per channel,
channel-major). Manifest entry must include `global_scale` (a single float).

```
sh_q8 = clip(round(sh_float / global_scale), -128, 127)
sh_float = sh_q8 * global_scale
```

For the Audi reference asset: `global_scale = 0.006946287755891094`.

### `sh_vq128_idx_u8` + `sh_vq128_codebook_i8` (v27, v28)

Replaces `sh_rest_q8_global` with a product VQ.

- `sh_vq128_idx_u8` — (n, 3) uint8: one codebook index per splat per group.
- `sh_vq128_codebook_i8` — (3, 128, 15) int8: 3 product groups, 128 codewords each, 15 coefficients per codeword.

To reconstruct the 45-coef q8 stream:
```
for splat i, group g in [0,1,2]:
    label = sh_vq128_idx_u8[i, g]
    sh_q8[i, g*15 : (g+1)*15] = sh_vq128_codebook_i8[g, label, :]
sh_float[i, :] = sh_q8[i, :] * global_scale         # global_scale = 0.006946...
```

Per-coefficient RMSE vs the original q8 stream is ~5 (in q8 units) ≈ 0.04 in
float space. This is intentionally lossy (the v27/v28 render container
exists for size; it shaves the largest single chunk in v25).

### `sh_exact_residual_t{T}_g{G}_int8` (v28 EXACT archive only)

9 chunks total, T ∈ {0,1,2}, G ∈ {0,1,2}. Each chunk is shape
`(tier_count[T], 15)` int8.

These add a per-tier-group residual to the VQ centroid to recover the
original v25 q8 SH stream BYTE-FOR-BYTE:

```
# After reconstructing sh_q8 from VQ as above:
for t in [0, 1, 2]:
    tier_indices = where(tier_labels == t)         # in original order
    for g in [0, 1, 2]:
        res = chunks[f'sh_exact_residual_t{t}_g{g}_int8']  # shape (len(tier_indices), 15)
        for r, splat_idx in enumerate(tier_indices):
            sh_q8[splat_idx, g*15:(g+1)*15] += res[r]
sh_q8 = clip(sh_q8, -128, 127)                     # int8 round-trip safety
sh_float = sh_q8 * global_scale
```

The "EXACT" promise: this reconstruction is byte-identical to the
v25-stored `sh_rest_q8_global` chunk for every element (verified — see
`reports/TIER_1.5_compression_baselines.md` §1.5.1).

## Reference implementations

- **Python reader** (~250 LoC): `tools/crypsorender/io/phox_loader.py`
- **JavaScript reader** (~200 LoC, browser-side): `viewer/phox_decoder.js`
- **Python writers**:
  - v25 (from PLY): `tools/build_v25_attribute_group.py`
  - v27 / v28 (from v25): `recovery_v2/tools/build_v27_fast.py`, `tools/build_v28_sh_exact_correction.py`

## Honest property summary

| Property | Claim | Verified? |
|---|---|---|
| Container integrity | Every chunk's `crc32_raw` matches `crc32(zlib.decompress(payload))` | yes (every reader checks) |
| v25 → v28-archive bit-exactness | Round-trip preserves the v25 q8 grid byte-for-byte | yes (`reports/TIER_1.5_compression_baselines.md` §1.5.1) |
| PLY → v25 quantization is deterministic | Re-quantizing PLY produces v25's stored bytes | yes (same report) |
| v25 q8 SH is the float-loss boundary | One-time PLY → v25 quantization is the only lossy step in the chain | yes — `sh_rest_q8_global = clip(round(f_rest / global_scale), -128, 127)` is the irreducible lossy step |
| Compression vs zstd-12 PLY | EXACT archive ~1.4× smaller; VQ render ~2.4× smaller | yes (`reports/TIER_1.5_compression_baselines.md`) |
| Compression vs SOTA splat compressors | NOT competitive yet (CRYPSOID 197–337 bpg vs SOG 50–80, HAC 30–60) | yes (`reports/TIER_1.5_bits_per_gaussian.md`) |

## Versioning policy

- **Magic** changes when the chunk-layout vocabulary changes incompatibly.
  Adding new chunk types within a magic family (e.g. `CRYPSOID28` adding
  the 9 EXACT residual chunks) is allowed and is disambiguated by the
  manifest `format` string.
- **Manifest `format`** changes whenever any new optional chunk type is
  introduced. Old readers that don't understand new chunks should still
  load the file by ignoring chunks they don't recognize, as long as the
  required chunks are present.
- **Per-chunk dtype/shape** is part of the chunk's contract; never change
  in place — define a new chunk name (e.g. `xyz_u24_fixed_v2`) instead.

## v0.4 (planned, not yet implemented)

Add native germ chunks so renderers don't have to compute them at load:

- `germ_5coef_f16` — (n_tier_AB, 5) float16. Per-(Tier A or B)-splat
  Pearcey germ coefficients (κ₁, κ₂, χ, ω, ζ) in sigma-normalized units.
- `germ_index_u32` — (n_tier_AB,) uint32. Splat index this germ row applies to.

`CRYPSOID40\0` magic; format string `CRYPSOID_3DPHOX_V40_FAITHFUL_PHOXOID`.
Chunks 0–6 same as v25/v27. New chunks 7+ are the germ ones.

This changes the on-disk size by ~5 floats × 240k splats × 2 bytes ≈ 2.4 MiB,
plus a 32-bit per-splat index ≈ 1 MiB, total ~3.5 MiB added. Saves ~3 sec of
load-time germ fitting per render.
