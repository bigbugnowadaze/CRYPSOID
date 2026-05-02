# Thesis digest — what the renderer must understand

Source: `recovery_v2/THESIS.txt` (2,059 lines). Read end-to-end 2026-04-30.

This digest pulls only the parts that constrain the renderer. It is intentionally short — re-read the thesis for the surrounding rationale.

---

## 1. The one-line claim

> A Gaussian splat is the exponential of quadratic distance from a center. A phoxoidal blob is the exponential of a *generated action field* — the soft action of a local surface/caustic germ. Gaussians become the boring constant-metric case of phoxoids.

That's the whole thesis in one sentence. Everything else is mechanism.

## 2. Two layers of math, do not conflate them

The thesis introduces TWO distinct generalizations of the Gaussian. The renderer needs both, but they sit at different layers of the stack.

### Layer 1 — The phoxoidal gauge `Φ` (replaces the exponent)

`density(u) = α · exp(-Φ(u))`, with

```
Φ = shape_cost + evidence_cost + boundary_cost + visibility_cost + gluing_cost
   = uᵀG(u,E)u      (deformable body, evidence-adaptive metric)
   + λ_r · R(x,v)²  (render residual: does this point predict observed color?)
   + λ_b · D(x,E)   (discontinuity barrier: edges, depth jumps, normal flips)
   + λ_s · S(x,𝒩)   (neighbor-consistency: does this blob agree with overlap?)
```

- `G` is NOT the constant covariance Σ⁻¹. It's a metric tensor that depends on the local evidence `E`.
- `R, D, S` are scalar penalty fields. They make the blob shrink at evidence breaks and expand through coherent regions.
- For a renderer this means: per splat, per query point, evaluate up to four scalar penalties and add them. Gaussian splats only evaluate the first term, with `G = Σ⁻¹` constant.

### Layer 2 — The "phoxponential caustic-chart action" (replaces the *shape* of the exponent)

Even after Layer 1 is added, the exponent in a Gaussian-style splat still measures distance from a *point*. Layer 2 says: the blob isn't centered on a point — it's softly emitted from a tiny **generator surface chart**.

In local frame coordinates `u = (a, b, n)` (two tangent + one normal):

```
Pho(u) = exp( -A_θ(u) )
A_θ(u) = -τ · log ∫∫ exp(-F_θ(u, s, t) / τ) ds dt        (a softmin over chart coords)
F_θ(u, s, t) = (a-s)²/σ_a²
             + (b-t)²/σ_b²
             + (n - H_θ(s, t))²/σ_n²
             + V_θ(s, t)                                  (chart support potential)
```

The new piece is `H_θ(s, t)` — the **germ**, a small local surface generator. Three increasingly rich choices:

| Germ | `H_θ(s,t) =` | Produces |
|---|---|---|
| Flat | `0` | Equivalent to a Gaussian disk in the tangent plane |
| Curved | `κ₁ s² + κ₂ t²` | A curved patch (saddle if `κ₁ κ₂ < 0`) |
| Phoxoidal | `κ₁ s² + κ₂ t² + χ(s³ - 3st²) + ω(3s²t - t³) + ζ(s⁴ + t⁴)` | Folds, cusps, commas, leaves, ribbons, lenses |

The cubic terms `s³ - 3st²` and `3s²t - t³` are the real and imaginary parts of `(s+it)³`, i.e. **cusp** generators (Pearcey caustic family). The quartic `s⁴ + t⁴` is the swallow-tail unfolding term. This is catastrophe-optics math, not generic polynomial fitting.

### How the two layers compose

Full phoxoidal density:
```
P(x, v) = α(v) · exp( -Φ(x, v; E, 𝒩) ) · A(x, v; E)
```
where the exponent `Φ` *can* incorporate the caustic-chart action `A_θ` from Layer 2 in place of the simple body term `uᵀGu`. In practice the renderer can choose its complexity:

| Profile | Body term | Evidence terms | Speed cost vs Gaussian |
|---|---|---|---|
| Gaussian splat | `uᵀΣ⁻¹u` | none | 1× (baseline) |
| Surfel-Gaussian | flat germ via Layer 2 | none | ~1.2× |
| Curved phoxoid | quadratic germ | none | ~1.5× |
| Cusp phoxoid | full polynomial germ | none | ~3× |
| Full phoxoid | full germ + Layer 1 evidence | all four | depends on cost-of-`E` |

## 3. The strict-generalization rule (load-bearing for the renderer)

A phoxoidal blob **reduces exactly to a Gaussian splat** when:
- The germ is flat: `H_θ ≡ 0`.
- The metric is constant: `G = Σ⁻¹`.
- All evidence terms are off: `λ_r = λ_b = λ_s = 0`.
- No visibility polarity, no neighbor compatibility maps.

So:
```
Φ_i(x) = ½ (x - c_i)ᵀ Σ_i⁻¹ (x - c_i)       ← recovers Gaussian
```

This is critical for the renderer: **it can render Gaussian-only data correctly with the phoxoidal pipeline by setting all the new terms to identity.** No special "Gaussian compatibility" code path needed — they're the same math at the limit.

## 4. The blob descriptor (what each splat carries)

```
Gaussian splat:
  { center, covariance, opacity, color/SH }

Phoxoidal blob:
  { center,
    frame,                                  (3×3 rotation, local tangent + normal)
    germ { κ₁, κ₂, χ, ω, ζ, support_pot },  (Layer 2 generator)
    softness { σ_a, σ_b, σ_n, τ },          (Layer 2 widths + temperature)
    appearance,                             (color/SH/material)
    opacity,
    visibility,                             (front/back bias, view validity cone)
    neighbor_glue                           (overlap IDs + transition maps)
  }
```

Note: **our current v25/v28 Audi data does NOT carry germ or evidence fields** — it's standard 3DGS (xyz, scale, quat, dc, opacity, sh_rest). Every splat in our Audi data today is in the "Gaussian splat / Tier C fallback" representation. To exercise the phoxoidal math on real data we'd need to run a converter that fits germs to local neighborhoods (this is what `recovery_v2/tools/phoxoid_convert.py` does at the v0 level).

## 5. Tier semantics (the bridge from thesis to .3dphox)

The v0.21–v0.23 doctrine in CRYPSOID assigns each splat one of three tiers (`tier_labels_u8` in v25/v27/v28). Mapped to thesis terms:

| Tier | Label | Render path | Splat representation |
|---|---:|---|---|
| A — native render phoxoid | 0 | Layer 1 + Layer 2 (full phoxoid) | Phoxoidal blob with evidence terms |
| B — native exact phoxoid | 1 | Layer 2 + exact-residual correction | Phoxoidal blob + lossless correction chunk |
| C — fallback | 2 | Standard Gaussian (degenerate phoxoid) | Vanilla anisotropic Gaussian |

In our Audi data: 94,006 / 144,271 / 525,523 splats. So 12.3% of splats *would* render through the full phoxoidal path if we had the germ data; the rest fall back to Gaussian. We don't have germ data yet — every splat is currently rendered as Gaussian. The renderer needs the *plumbing* for tier-aware paths even though all three paths reduce to Gaussian on this dataset until a phoxoid converter has been run.

## 6. The killer metric (what to optimize for)

From the thesis, repeated multiple times:

> **How many Gaussian splats does one phoxoidal blob replace at equal visual/geometric error?**

Not "is RMS slightly lower on Bunny." The whole project lives or dies by this question. The renderer's job is to make this measurable.

## 7. The development methodology — PhoxBench

The thesis explicitly demands a benchmark harness, not one-off tests:

- **Tier 0** — Synthetic truth scenes (plane, sphere, saddle, fold, cusp, thin sheet, crossing sheets, occlusion edge, noise patch, hair strands, rippled surface, material boundary). This is where invention happens fastest because ground truth is exact.
- **Tier 1** — Canonical meshes (Bunny, Dragon, Armadillo, Buddha, Lucy). Sanity layer.
- **Tier 2** — Trained 3DGS .ply scenes. The real bridge to splat work — and **this is exactly what our Audi A5 data is**.
- **Tier 3** — Real reconstruction datasets (Mip-NeRF360, Tanks & Temples, DTU, ScanNet, Replica).

Every run produces:
```
runs/<date>_<scene>_<kernel>_<budget>_<seed>/
  input_preview.png
  gaussian_baseline.png
  phoxoid_output.png
  side_by_side.png
  error_heatmap.png
  primitive_overlay.png
  metrics.json
  report.md
  output.phox.json
  baseline.gauss.json
```

Pareto scorecard across geometry / rendering / efficiency / representation metrics. Failures get added back to the synthetic suite as permanent test scenes.

## 8. What this means for our renderer (concrete)

1. **Build a CPU rasterizer that handles Gaussian splats correctly first** (the Tier C fallback path). Required for the Audi side-by-side and turntable Bug asked for, and is the degenerate case of the full phoxoid renderer anyway.
2. **Architect it from day 1 as tier-aware.** A `splat_kind` dispatch in the rasterization inner loop, even if all three tier paths currently call the same Gaussian code. When phoxoidal germ data shows up, we drop in the new math without rewriting the depth-sort, tile-binning, alpha-compositing, or projection plumbing.
3. **Implement the phoxoidal Layer 2 (germ-based density) next**, with `H_θ` as a parameter object so we can A/B test germ families (flat / curved / cusp / fold / quartic).
4. **Layer 1 evidence terms come last.** They depend on having an evidence stack `E` per scene (RGB residual vs. ground truth, edge maps, depth maps, etc.). For our Audi data we don't have that yet; if we want to demonstrate Layer 1 we need synthetic scenes from the Tier 0 list above.
5. **No torch, no CUDA, no gsplat. Ever.** This is in `feedback_no_gpu_deps.md` already.

## 9. What we have on disk that aligns

- `recovery_v2/tools/phoxoid_convert.py` — v0 PLY-to-`.phox.json` converter. Does PCA-tangent baseline + chart-germ fit per cluster. Smoke-test quality, but it IS a working germ fitter we can study/adapt for an Audi-PLY phoxoid conversion run.
- v21/v22 CSVs (in `inputs/v21_v22_artifacts/`) — these are the spatial-cell decomposition the original v25 used to assign tier labels. They contain `center_x/y/z`, eigenvalues, and per-cell appearance stats — i.e. the seeds of the local frame and base softness for each potential phoxoidal blob.
- v28 q8-exact archive — the Tier B "native exact phoxoid + correction" container. The correction chunks here are what would be added to the germ-decoded SH to recover bit-exact original values.

## 10. Open questions the renderer design must resolve

1. How is the local frame `R_i` defined for each splat in our current Audi data? (Standard 3DGS quaternion → 3×3 rotation, OK.)
2. Do we render the full softmin integral or use a closest-point approximation on the chart `(s*, t*)`? Closest-point is much faster; for `H = κ₁s² + κ₂t²` it's solvable in 1–2 Newton iterations.
3. How does sorting by depth interact with the germ? A vanilla Gaussian sorts by `z(center)`; a curved phoxoid spans a depth interval. Probably we sort by the deepest point on the splat's bounding shape, but this needs a small experiment.
4. How do neighbor-glue terms (`S(x, 𝒩)`) actually get computed in the rasterizer's inner loop? Naively they're `O(neighbors)` per pixel per splat — way too expensive. Likely needs a precomputed neighbor table and only the splat-pair compatibility precomputed once.
5. What's the visualization for "tier" in the rendered image? Color-tint each splat by tier? A separate tier-overlay panel? Both?
