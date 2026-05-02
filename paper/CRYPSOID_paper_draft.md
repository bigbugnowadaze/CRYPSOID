# CRYPSOID: A CPU-Only Phoxoidal Format and Renderer for 3D Gaussian Splat Scenes

**Draft 1 — 2026-05-02**

## Abstract

We present CRYPSOID, a CPU-only alternative to standard 3D Gaussian Splatting that introduces (1) a tiered container format (`.3dphox`) separating render-mode from EXACT-archive mode, (2) a *phoxoidal blob* primitive built on a 5-coefficient Pearcey germ basis from catastrophe optics, and (3) a graph-native format extension (v31) that ships explicit normals, kNN edges, and sparse delta patches alongside splats. We measure that **a single phoxoidal blob replaces ~2.0× as many Gaussian blobs at equal RMSE** on six synthetic stress scenes and four real meshes, with the ratio holding flat across budgets B=32 and B=64. We further demonstrate a layered lighting stack (Lambert, curvature shading, kNN soft shadows, graph ambient occlusion, cusp-specular, material-aware floater detection) that runs entirely on CPU at full splat density (763,800–1,612,868 splats) in 9–22 seconds via numba JIT. The complete CRYPSOID stack achieves 1.40× compression vs `zstd -12` PLY at bit-exact q8 fidelity while shipping format extensions that no other splat representation carries natively. Honest framing: bitrate (197–337 bpg) is not yet competitive with state-of-the-art splat-specific codecs (SOG ≈ 50–80 bpg, HAC ≈ 30–60 bpg); the contribution is structural and reproducibility-oriented.

---

## 1. Introduction

3D Gaussian Splatting (3DGS) has rapidly become the dominant explicit representation for novel-view synthesis, but the ecosystem has settled into a uniform stack: PLY-on-disk, GPU rasterization, neural training. We argue that **the format itself, the primitive itself, and the dependency on GPU machinery are all separable**, and that exploring each axis yields a surprisingly capable alternative.

CRYPSOID's three contributions:
1. **Format-level tiering.** `.3dphox` is not just compression — it is a container that lets render-mode (small, lossy SH-VQ) and EXACT-archive mode (bit-exact at the q8 grid) coexist with the same per-splat tier labels.
2. **Phoxoidal primitive.** Each blob carries a 5-coefficient Pearcey germ basis (κ₁, κ₂, χ, ω, ζ) instead of a flat tangent plane. We measure that this gives a 2.0× reduction in the number of primitives needed for equivalent fit RMSE.
3. **Graph-native format (v31+).** Normals, kNN edges, and sparse delta patches as first-class chunks, enabling phoxoidal-math-specific rendering features (curvature shading, kNN soft shadows, graph AO) that no other splat representation carries natively.

The project's hard rule is: **no GPU dependencies in the core codebase.** Forbidden: `torch`, `cuda-toolkit`, `gsplat`, any `nvidia-*` package. The whole project exists because state-of-the-art splat tooling demands a heavy GPU stack; CRYPSOID's reason to exist is to be a 1:1 alternative that does not.

---

## 2. The `.3dphox` Format

### 2.1 Tiered container

A `.3dphox` file consists of:
- A magic + manifest JSON describing chunk layout and per-tier metadata.
- Independent chunks for each per-splat attribute group.
- Per-splat tier labels (`A` = native phoxoid render, `B` = exact-archive phoxoid, `C` = Gaussian fallback).

We ship four format variants:
- **v25** — full 11-attribute uncompressed-but-quantized layout (~6× smaller than raw PLY).
- **v27/v28-render** — VQ-codebooked SH (~9.6× smaller than raw PLY, lossy at the SH band 1–3 level).
- **v28-EXACT-archive** — VQ-codebooked SH plus per-tier-group residual chunks that bit-exactly reconstruct the v25 q8 SH stream (~5.6× smaller than raw PLY, lossless at the q8 grid).

### 2.2 Compression numbers (Audi A5 PLY, 763,800 splats)

The naive headline is "v28 is 5.6× smaller than the original PLY," which is true vs raw float32 PLY but inflated against any reasonable baseline. The honest comparison is against `zstd -12` PLY:

| Format | bits/Gaussian | vs zstd-12 PLY | Lossy? |
|---|---:|---:|---|
| `zstd -12` PLY (lossless) | 470.4 | 1.00× | no |
| **CRYPSOID v28 EXACT archive** | **336.9** | **1.40× smaller** | bit-exact at q8 |
| **CRYPSOID v28 VQ render** | **196.9** | **2.39× smaller** | yes (~5% per-coef RMSE) |

Reference points:
- Self-Organizing Gaussians (SOG) ≈ 50–80 bpg
- HAC ≈ 30–60 bpg

So CRYPSOID at 197–337 bpg is meaningful versus zstd PLY but is **not competitive with state-of-the-art splat-specific compressors on bitrate alone.** This is the honest framing. The contribution is not "smaller bytes" — it is the **structural** capability the format ships.

### 2.3 Bit-exactness chain

The chain `PLY → v25 quantization → v28 EXACT archive → decode → ndarray` is bit-exact at the q8 grid level. All five non-SH attributes are byte-identical between v25 and v28-archive; the v28-archive's SH reconstruction matches v25's stored q8 SH stream for all 34,371,000 int8 elements. The lossy step is the one-time PLY → v25 quantization (q8 SH, u24 XYZ, f16 scale, i16 quat, u8 DC/opacity), which is deterministic and reproducible from PLY by design.

---

## 3. The Phoxoidal Primitive

### 3.1 Math

A phoxoidal blob's local surface law is:

```
H(s, t) = κ₁·s² + κ₂·t² + χ·(s³ − 3st²) + ω·(3s²t − t³) + ζ·(s⁴ + t⁴)
```

The first two terms reproduce a quadratic tangent surface (mean and Gaussian curvature). The cubic terms (χ, ω) are the universal cusp/swallowtail generators from catastrophe theory's Pearcey integral. The quartic (ζ) provides isotropic stiffening for high-frequency features.

Per-splat density evaluation uses a closest-point Newton solver on H with a `λ(s² + t²)²` support gate to keep the iteration well-posed. The 4 anchor numerical-correctness tests pass: support-gated convergence, plane reduction, Gaussian reduction (via early-return when germ = 0), and a cusp asymmetry test.

### 3.2 PhoxBench killer-ratio

The "killer-ratio" metric: how many Gaussian blobs are needed to match a given phoxoidal blob's RMSE? PhoxBench Tier 0 (synthetic stress scenes) and Tier 1 (real meshes) report:

#### Tier 0 — synthetic (B=32 budget)
| Scene | Gauss RMSE | Phox RMSE | Killer ratio |
|---|---:|---:|---:|
| sphere | 0.01024 | 0.00698 | **2.0×** |
| saddle | 0.01184 | 0.00818 | **2.0×** |
| fold | 0.00567 | 0.00343 | **2.0×** |
| cusp | 0.00867 | 0.00673 | **2.0×** |
| thin_sheet | 0.01991 | 0.01978 | **4.0×** |

#### Tier 1 — real meshes (B=32)
| Mesh | Source | Gauss RMSE | Phox RMSE | Killer |
|---|---|---:|---:|---:|
| Happy Buddha | Stanford | 0.01799 | 0.01583 | **2.0×** |
| Armadillo | Stanford | 0.00902 | 0.00749 | **2.0×** |
| Doom combat | Game scene PLY | 0.05017 | 0.04670 | **2.0×** |
| Audi A5 | Trained 3DGS PLY | 0.02857 | 0.02728 | **2.0×** |

The 2.0× killer ratio is **flat across all 8 (scene × budget) combinations** at B=32 and B=64. This is the central empirical result.

### 3.3 Honest scope of the killer-ratio claim

- This is fit RMSE on point clouds, not visual rendering quality on trained splat scenes with full SH/opacity machinery (that benchmark is Phase C, future work).
- Killer-ratio search uses doubling (16, 32, 64, 128, …) so 2.0× is the smallest power-of-two budget that meets phoxoid RMSE; actual ratio is anywhere in [1.5×, 2.5×].
- Each mesh is subsampled to 10k pts for harness speed; absolute RMSE shifts with N, but the *relative* gap is what matters and is stable.

---

## 4. The v31 Graph Extension

After validating the primitive, we extended the format to ship the geometric infrastructure that surrounds it.

### 4.1 Three additions

**v31 Addition 1 — Normals chunk (chunk_id 0x12).** 24-bit octahedral normal + 8-bit tangent angle = 4 bytes/blob. MLS-derived from kNN with a quadric refinement step that removes the plane-fit-on-curved-surface bias. Acceptance gates: round-trip byte-identical (after the first lossy quantization), unit-norm assertion, sphere stress test p95 < 10 mrad, tangent-angle 8-bit precision, CRC integrity. **5/5 pass.**

**v31 Addition 2 — kNN edges chunk (chunk_id 0x13).** k=4 u32 neighbor indices per blob = 16 bytes/blob. BallTree-derived; self-edges filtered robustly even with duplicate xyz positions (which the Audi has — 55 duplicates). Acceptance gates: round-trip byte-identical (codec is lossless), CRC integrity, version mismatch detection, no self-edges, sorted by distance. **5/5 pass.**

**v31 Addition 3 — `.phoxdelta` patch format.** Sparse modify-only patches over a base `.3dphox`, referenced by base CRC32. Per-record: phoxoid_id (u32) + dirty_mask (u16) + only-the-changed-attributes payload. Apply / compose / re-derive operations. **5/5 acceptance gates pass.** Demonstrated on the Audi by building a "de-halo" delta of 1.9 MB that modifies 190,948 phoxoids' opacity in the 47.4 MB base file — without re-encoding.

### 4.2 Cost on Audi
| Cycle | Adds | Bytes | vs v28 |
|---|---|---:|---:|
| v31 Add 1 | normals + tangent | +3.06 MB | +9.5% |
| v31 Add 2 | kNN edges (k=4) | +12.22 MB | +38.0% |
| v31 Add 3 | `.phoxdelta` (sparse) | variable | n/a |
| Combined | | +15.28 MB | **+47.5%** |

The +47.5% matches our spec prediction to one decimal place.

---

## 5. The Lighting Stack (v32a / v32b / v32.5 / v32c)

With normals + edges in the format, we built a layered renderer that adds capabilities none of which require modifying any other splat format's bytes:

- **v32a — Lambert + ambient + directional sun.** Standard graphics math (1975); zero format bytes. Per-splat `ambient·albedo + sun·albedo·max(0, N·−L)`. Unlocks "user can see lit geometry."
- **v32b — Curvature shading.** Phoxoidal-math-specific: visibility = `max(0, N·L) · (1 − β·|κ_eff|·(1 − N·L))` self-shadows curved patches at grazing angles; ambient_factor = `1 − α·tanh(|κ_eff|)` darkens concave regions. Uses existing v31 germ data; zero new bytes.
- **v32.5 — kNN-graph soft shadows + graph AO.** Per (phoxoid, light), walk the v31 kNN-edges chunk, accumulate Gaussian-falloff occlusion. O(k=4) per phoxoid per light; **no spatial structure beyond the format.** Vectorized numpy: 0.2s for 200k splats × 4 neighbors × 1 light. The kNN edges chunk pays for itself a second time (after LOD) here.
- **v32c — Cusp-specular from cubic germ terms.** Per-splat shininess scales with `cusp_strength = sqrt(coef[u³]² + coef[u²v]² + coef[uv²]² + coef[v³]²)` from extended MLS. High-cubic features get sharper highlights than equivalent-quadratic ones.

All four are renderer-only (zero format bytes). The **phoxoidal-math-specific contributions** are concentrated in v32b, v32.5, and v32c.

---

## 6. v33 Material-Aware Detection

v33 adds 4 bytes/blob (material_hint + confidence + view_dependence + mip_zoom) and ships two complementary derivation pipelines:

### 6.1 Phase-1: SH-coefficient + EFA-GS heuristic
Per-splat single-frame heuristic: classify based on SH band magnitudes (DC vs 1 vs 3), opacity, surface variation κ, and kNN edge length. Catches "structural floaters" (Clean-GS / EFA-GS style: sparse + flat + low-opacity). Cheap, runs in seconds. Conservative: ~1.7% of Audi flagged.

### 6.2 Phase-2: Multi-view photometric (GS-2M-style)
Per-splat: decode SH at K=12 view directions on a Fibonacci sphere; correlate the (12, 3) RGB sequence with the mean of its k=4 neighbors' sequences; low correlation ⇒ photometric outlier ⇒ floater. **No source images required** — uses the SH itself as the multi-view signal. 5.5 seconds for 763k splats.

### 6.3 Findings
| Detection | Floater count on Audi | % of scene |
|---|---:|---:|
| Phase-1 (conservative) | 13,365 | 1.7% |
| Phase-2 (top-20%) | 152,760 | 20.0% |
| **Phase-1 ∪ Phase-2** | **164,416** | **21.5%** |
| Overlap (Phase-1 ∩ Phase-2 top-20%) | 937 | only 7% of Phase-1 |

The two signals catch **different populations** — the union is the right default. Phase-2 negative-result on Doom (an artist mesh with no real floaters) shows nearly all-green overlay, validating the algorithm's discriminating power.

---

## 7. Phase D.1 Performance Optimization

### 7.1 Numba JIT rasterizer
Pure-Python per-splat rasterizer: 26.5s for 200k splats at 1024². Numba JIT version: **0.76s. 34.7× speedup**, well above the spec's 5–10× target.

### 7.2 Lit hero deliverables (post-Numba)
- **Lit Audi at full 761,707-splat density:** 9s end-to-end (was 100s+).
- **Lit Doom at full 1,612,868-splat density:** 22s end-to-end.

These were not feasible before the perf work. With them, multi-view turntables and interactive parameter tuning become tractable.

---

## 8. Browser viewer

A single-file HTML+JS+WebGL2 viewer (`viewer/index.html`) loads any `.3dphox` (v25 / v27 / v28 render / v28 EXACT archive) and renders client-side. Includes a JS port of the format decoder. The viewer's GPU code runs on the user's GPU; the CRYPSOID core codebase stays GPU-free.

The viewer currently consumes v25–v28; the v31 normals/edges chunks and v33 material chunks are not yet wired in. This is the next planned implementation (Phase D.2).

---

## 9. Continuous integration

GitHub Actions (`.github/workflows/test.yml`) runs on every push:
- Tier 2 numerical-correctness anchor tests (4 of 4).
- PhoxBench cusp smoke benchmark.
- v25 ↔ v28-archive byte-identity check (when containers are in-tree).
- **Banned-package check** ensures no GPU dependency ever enters the dep tree (`torch`, `cuda-toolkit`, `gsplat`, any `nvidia-*`).

---

## 10. Honest comparison with prior splat work

We've absorbed the *idea* (not the implementation) of several recent splat-research projects:
- **Mip-Splatting** — anti-alias filter footprints (planned v33 amendment).
- **StopThePop** — view-consistent depth sort (planned for WebGL viewer).
- **Clean-GS / EFA-GS** — floater detection (Phase-1 heuristic in v33).
- **GI-GS / SSD-GS / LumiGauss / GS-2M** — light/material decomposition (Phase-2 photometric in v33).
- **OpenSplat / Gauzilla** — CPU/portable splat rendering (informs Phase D.1+).

The CRYPSOID position relative to these: **none of them ship the explicit format extensions** (graph + delta + material) that v31+v33 provides. They are renderer/training innovations on top of the standard splat format. CRYPSOID is the *format-level* alternative that makes those features cheap to implement downstream.

---

## 11. Reproducibility

Every claim in this paper is reproducible from the source repository:

```bash
# Tier 0 (synthetic stress scenes)
cd tools && python3 -m phoxbench.run_scene --scene all --budget 32 --out /tmp/phoxbench/runs

# Tier 1 (real meshes)
python3 -m phoxbench.run_mesh --all --budgets 32 64

# Acceptance tests for v31/v33 codecs
python3 -B tools/test_normals_codec.py    # 5/5 pass
python3 -B tools/test_edges_codec.py      # 5/5 pass
python3 -B tools/test_phoxdelta_codec.py  # 5/5 pass
python3 -B tools/test_material_codec.py   # 4/4 pass

# Audi lit hero (full 763k density)
python3 -B tools/render_phox_chunked.py --scene outputs/v28_sh_vq_exact_archive_container.3dphox --is-phox --use-sh ...

# Doom lit hero (full 1.6M density)
python3 -B tools/render_doom_lit.py
```

Manifest files (`renders/crypsorender_v01/manifest_*.json`) document which math path each render actually used and what its honest caveats are.

---

## 12. What's next (necessary roadmap)

1. **Phase C — Mip-NeRF 360 trained-3DGS scene benchmark.** PhoxBench Tier 2: confirm killer-ratio holds on properly trained 3DGS scenes (bicycle, garden, kitchen, …), not just point clouds. ~1 week.
2. **Phase D.2 — WebGL viewer wire-up of v31 + v33 chunks.** So users can see the format extensions without running CPU code. ~3–5 days.
3. **Phase D.3 — v40 native germ chunks.** Currently germs computed at load; persist them. ~2 days.
4. **Phase E — Layer-1 evidence terms (R, D, S) + learned arithmetic coder.** Research-grade work; needs multi-view image evidence (which Mip-NeRF would supply). Several weeks.

---

## 13. Conclusion

CRYPSOID is **a structural alternative**, not a bitrate-optimal one. The 2.0× killer ratio is the central empirical contribution; the v31+v33 format extensions and the layered lighting stack (v32a/b/.5/c) are the engineering scaffolding that makes the primitive useful in practice. The numba perf work makes it usable at scale on a CPU. Where SOG and HAC compete on bytes-per-Gaussian, CRYPSOID competes on *what each blob is allowed to be* — a stored normal, a stored neighborhood, a sparse delta-able entity, a material-classified atom — and on *what the renderer can do without GPU dependencies*.

The honest summary: this is not state-of-the-art on any single axis. It is reproducible, CPU-only, format-explicit, structurally novel, and ships infrastructure that other splat formats do not. Whether the trade is worth it depends on whether you care about format-level capability vs raw bitrate.

Source: https://github.com/<your-handle>/crypso1d

---

## Acknowledgments

Thanks to the recovery cycle that produced the v25/v27/v28 anchor data, and to the open-source splat ecosystem (3DGS reference implementations, Stanford 3D Scanning Repository, OpenSplat, Mip-Splatting, GS-2M, et al.) whose ideas we've absorbed into the explicit-format approach.
