# Fix #1 Diagnostic: Quaternion Normalization (`quat_i16_norm4`)

## Final result

**Byte-identical to v27 anchor's `quat_i16_norm4` chunk** (0 differing bytes out of 6,110,400).

## What the original v25 actually did

Empirically reverse-engineered by testing all plausible encoder variants against the anchor:

1. Read `rot_0..3` from PLY as **float32** (PLY native precision).
2. Normalize to unit length, **in float32** (`q / np.linalg.norm(q, axis=1, keepdims=True)`).
3. Quantize: `np.clip(np.round(q * 32767), -32768, 32767).astype(np.int16)`.
4. Sign-canonicalize: if `q[0] < 0` after quantization, negate the whole row.

## Hypothesis test results (763,800 quaternions)

| Encoder variant | Matching rows (of 763,800) | Differing bytes (of 6,110,400) |
|---|---:|---:|
| Raw float32 × 32767, no normalize, no flip | 475,490 | 2,295,777 |
| Normalize fp32 × 32767, no flip | 475,810 | 2,295,457 |
| Normalize fp64 × 32767, no flip | 475,238 | 2,296,029 |
| Raw fp64 × 32767, no flip | 475,198 | 2,296,069 |
| Raw fp32 × 32767, sign-flip | 763,330 | 470 |
| **Normalize fp32 × 32767, sign-flip** | **763,800** | **0** |
| Normalize fp64 × 32767, sign-flip | 762,919 | 881 |
| Raw fp64 × 32767, sign-flip | 762,849 | 951 |

The float64 normalize variant the previous coding agent shipped is the second-best "sign-flip" variant but is *worse* than the simpler "raw float32 + sign-flip" path (881 vs 470 bytes diff). The actual winner is float32 normalize plus sign-flip.

## Why the precision matters

When `np.linalg.norm` is computed in float64 vs float32, the result differs by a few ULPs for most quaternions. After multiplying by 32767 and rounding, those ULP differences land on opposite sides of 0.5 for ~881 components, producing off-by-one quantization. The original v25 builder evidently never upcast to float64 — it kept everything in float32 — so honoring its precision exactly requires us to do the same.

## Code change applied

`tools/build_v25_attribute_group.py`, function `encode_quat_i16`:

```python
def encode_quat_i16(verts, N):
    quat_f32 = np.stack([verts[f'rot_{j}'].astype(np.float32) for j in range(4)], axis=1)
    norms = np.linalg.norm(quat_f32, axis=1, keepdims=True)
    quat_f32 = quat_f32 / norms
    quat_i16 = np.clip(np.round(quat_f32 * 32767), -32768, 32767).astype(np.int16)
    flip = quat_i16[:, 0] < 0
    quat_i16[flip] = -quat_i16[flip]
    return quat_i16.tobytes()
```

## Verification

Decompressed `quat_i16_norm4` raw bytes from the rebuilt v25 are byte-equal to the v27 anchor's `quat_i16_norm4` raw bytes (verified 2026-04-30). Gate 8 of the spec now passes for all five v25-passthrough chunks.
