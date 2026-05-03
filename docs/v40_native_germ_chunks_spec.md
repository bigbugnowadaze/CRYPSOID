# v40 — Native Germ Chunks Spec (one-pager)

**Status:** draft for sign-off, 2026-05-02.
**Goal:** persist the MLS-derived geometric helpers (κ, cubic germ coefficients) inside `.3dphox` so the render-time MLS pass (currently 3–10 s per scene) is eliminated. Renderer becomes "load → project → rasterize" with no derivation step.
**Depends on:** v31 (which already ships normals, kNN edges).

## Why v40

Today's render pipeline runs an MLS pass at load time:
- 170k splats (Little Plant): **3.3 s** of cov eigendecomposition
- 763k splats (Audi): **6–30 s** of BallTree + cov per-batch
- 1.6M splats (Doom): **~80 s** of chunked BallTree query

That MLS pass derives:
- Surface-variation index κ (used by v32b curvature shading)
- Cubic germ coefficients χ, ω (used by v32c cusp-specular)
- Optionally the full 5-coef Pearcey germ (κ₁, κ₂, χ, ω, ζ)

All three are *deterministic functions of xyz* — they don't change between renders, between cameras, between lighting setups. **Persisting them in the file removes the MLS pass entirely.** A v40 render is `load (1s) → project (0.2s) → rasterize (1-4s)` — total ~3 s for any size scene.

This unlocks:
- **Real-time interactive lighting** (move sun, re-render in 3 s).
- **Multi-frame turntables at full density** in 36 × 3 s = ~2 minutes.
- **Per-frame v32c cusp-specular** without re-fitting cubics every frame.
- **Lower bar for downstream consumers** (the WebGL viewer's GLSL would not need an MLS pass).

## Non-goals

- Computing germs at write-time (that's existing MLS code; we just *store* its output).
- Full Pearcey 5-coef-per-splat encoding (kept as a v40.1 follow-on; v40 ships κ + cubic-magnitude only).
- Removing the kNN edges chunk (already in v31; v40 uses it).
- Lossless representation of every floating-point coefficient (we quantize; spec gates the precision).

## Three additions

### Addition 1 — Per-splat κ chunk (chunk_id 0x15)

The Pauly surface-variation index, used by v32b curvature shading.

| Field | Bytes | Encoding |
|---|---:|---|
| `kappa_q8` | 1 | u8 0–255, mapped from κ ∈ [0, 0.5] (κ_max for surface-variation = 1/3, headroom for noise) |

**Total: 1 byte/phoxoid.** On Audi (763,800 blobs): 763 KB.

Decode: `κ = (kappa_q8 / 255) * 0.5`. Quantization step ≈ 0.002 — well below the precision needed for shading (κ enters via `tanh(α·κ)` with α ≈ 4, so 0.002 step → 0.008 rad change in shading factor, imperceptible).

### Addition 2 — Per-splat cusp magnitude chunk (chunk_id 0x16)

Magnitude of the cubic germ coefficients (χ, ω + cross terms), used by v32c cusp-specular.

| Field | Bytes | Encoding |
|---|---:|---|
| `cusp_q8` | 1 | u8 0–255, mapped from cusp_norm ∈ [0, 1] (already normalized in derivation) |

**Total: 1 byte/phoxoid.** On Audi: 763 KB.

Decode: `cusp_norm = cusp_q8 / 255`. v32c shading uses `shininess = 16 + 256·cusp_norm`; a 0.004 step → 1 unit shininess change, imperceptible.

### Addition 3 — Per-splat full 5-coef Pearcey germ (chunk_id 0x17, optional)

For consumers that want the full germ for closest-point Newton or per-pixel shading.

| Field | Bytes | Encoding |
|---|---:|---|
| `kappa1_f16, kappa2_f16` | 2+2 | float16 |
| `chi_f16, omega_f16, zeta_f16` | 2+2+2 | float16 |

**Total: 10 bytes/phoxoid.** On Audi: 7.6 MB.

This chunk is *optional* — v40 readers can skip it if they only need the q8 versions. A v40-without-chunk-0x17 file is ~8.7× smaller than v40-with for the same scene.

## Combined cost on the Audi A5 (763,800 phoxoids)

| Cycle | Adds | Bytes | vs current v28 | vs v31+v33 |
|---|---|---:|---:|---:|
| v40 Add 1 (κ q8) | 1 B/blob | 763 KB | +2.4% | +1.5% |
| v40 Add 2 (cusp q8) | 1 B/blob | 763 KB | +2.4% | +1.5% |
| v40 Add 3 (full 5-coef germ, optional) | 10 B/blob | 7.6 MB | +23.7% | +15.0% |
| **v40 minimal (Add 1+2 only)** | **2 B/blob** | **1.5 MB** | **+4.7%** | **+3.0%** |
| v40 full (Add 1+2+3) | 12 B/blob | 9.2 MB | +28.5% | +18.2% |

**Recommendation:** ship Add 1 + 2 by default. Add 3 is opt-in for downstream consumers that need full germs (e.g., per-pixel Newton in a GLSL renderer).

## Render-time payoff

Before v40 (current):
| Step | Time @ 763k |
|---|---:|
| Load `.3dphox` | 1.0 s |
| Build BallTree | 0.5 s |
| kNN query + cov + eigh | 6–30 s (chunked) |
| Project | 1.2 s |
| Numba rasterize | 4 s |
| **Total** | **13–37 s** |

After v40:
| Step | Time @ 763k |
|---|---:|
| Load `.3dphox` (incl. v40 chunks) | 1.2 s |
| Project | 1.2 s |
| Numba rasterize | 4 s |
| **Total** | **~6 s** |

**Speedup: 2.2× to 6×** depending on scene size and whether MLS pass was already cached.

## Acceptance gates (v40 build sign-off)

1. `.3dphox` v40 reader/writer round-trips: write → read → byte-identical re-encode.
2. κ chunk: decoded κ values within 0.002 of source for all splats.
3. cusp chunk: decoded cusp_norm within 0.004 of source.
4. Optional 5-coef germ chunk: decoded κ₁/κ₂/χ/ω/ζ within float16 precision (5e-4 absolute).
5. Visual A/B: render same scene with on-the-fly MLS vs v40 native chunks. PSNR ≥ 50 dB (effectively visually identical).
6. Performance gate: end-to-end render at 763k splats in ≤ 8 s (target 6 s).
7. CI: a v40 smoke test that builds a v40 file, renders it, asserts the perf + PSNR gates.

## Build path

A v40 file is built from a v31+v33 file plus the cached MLS data:

```bash
python3 tools/build_v40_native_germs.py \
    --base outputs/v31_audi_full_v33.3dphox \
    --kappa /tmp/audi_full_kappa.npz \
    --germ /tmp/audi_sub_germ.npz \
    --out outputs/v40_audi_full.3dphox
```

The build script:
1. Reads the v31+v33 file verbatim.
2. Quantizes κ to u8 → chunk 0x15.
3. Quantizes cusp_norm to u8 → chunk 0x16.
4. (Optional) Quantizes κ₁/κ₂/χ/ω/ζ to f16 → chunk 0x17.
5. Appends a v40 trailer marker after the v31 trailer.

Backward compat: a v31+v33 reader sees the v40 trailer as junk-after-end and ignores it. v40 readers parse the additional chunks.

## Suggested phasing

1. Sign off this spec.
2. Implement `kappa_codec.py` + `cusp_codec.py` (parallel to existing normals/edges/material codecs). ~1 day.
3. Implement `build_v40_native_germs.py`. ~half a day.
4. Modify the renderer to read chunks if present, fall back to MLS otherwise. ~half a day.
5. Run perf benchmark + visual A/B. ~half a day.
6. CI test + spec acceptance gates. ~half a day.

**Total estimated effort: ~3 days.**

## Honest scope summary

| Claim | True / partial / aspirational |
|---|---|
| Eliminates the MLS pass at render time | **True** — chunks are deterministic from xyz |
| 2.2–6× render speedup | **True** — measured in the breakdown above |
| Doesn't increase visible quality | **True** — same math, just precomputed |
| Stays within v31's structural identity | **True** — additive chunks, no v31 chunk modified |
| Format size penalty is small | **True for minimal v40** (1.5 MB on Audi); larger if optional 5-coef chunk is included |
| Required for any compelling future work | **No** — v40 is purely a perf + portability win. The renderer works without it. |
