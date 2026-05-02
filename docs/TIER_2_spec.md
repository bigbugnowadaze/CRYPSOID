# Tier 2 — Faithful phoxoidal renderer + PhoxBench

**Status:** DRAFT — needs Bug's sign-off before any code is written.
**Reads:** `docs/thesis_digest.md`, `docs/crypsorender_architecture.md`. Don't read this without those.
**Doctrine:** still no GPU/CUDA dependencies. numpy + scipy + Pillow + ffmpeg only.

---

## 0. Why Tier 2 exists

Tier 1 + 1.5 produced a working CPU 3DGS-class renderer with tier-aware dispatch infrastructure and verified bit-exactness. But the Tier A/B "phoxoidal" path we shipped is a **screen-space approximation** of the form

```
density(p) ≈ exp( -½ · m · (1 + λ · |κ| · m) )       where m = Mahalanobis²(p)
```

This *modulates* the Gaussian falloff but doesn't change its level-set topology. Phoxoid output looks ~14% leaner than Gaussian (early termination kicks in faster) but it's qualitatively the same shape.

**Tier 2's reason to exist: make the phoxoidal path qualitatively different from Gaussian, and prove it on stress scenes where the difference matters.**

That requires three things, in order:
1. **A faithful per-pixel evaluation** of the phoxoidal action `A_θ(u)` (closest-point on the germ surface).
2. **A richer germ basis** — the cubic and quartic Pearcey/swallowtail terms from the thesis, not just the curvature-only quadratic.
3. **A benchmark that isolates the cases where the difference matters** — synthetic stress scenes with exact ground truth (PhoxBench Tier 0).

Without any one of these, "phoxoidal beats Gaussian" stays a claim, not a measurement.

---

## 1. Scope: what's in and what's out

**In scope for v0.3 (Tier 2):**
- Per-pixel closest-point Newton solver for the phoxoidal action `A_θ(u)` on quadratic and cubic germs.
- 5-coefficient Pearcey-class germ basis: `H(s,t) = κ₁s² + κ₂t² + χ(s³ - 3st²) + ω(3s²t - t³) + ζ(s⁴ + t⁴)`.
- Per-splat 5-coefficient germ fitter (extend existing `fit_synthetic_germs`).
- PhoxBench Tier 0 — at least 6 of the 10 synthetic scenes from the thesis, with analytic ground truth.
- Per-scene phoxoid-vs-Gaussian benchmark: same blob budget, same view, RMSE / Chamfer / image PSNR.
- "Killer metric" computation: how many Gaussian blobs needed to match N phoxoid blobs at equal RMSE.
- One showcase rendering where the phoxoidal output is **visibly different from Gaussian** (cusp scene rendered with both).

**Out of scope (deferred to v0.4 or later):**
- Layer 1 evidence terms (`R, D, S` — render residual, discontinuity barrier, neighbor consistency). They need an evidence stack `E` we don't have for the Audi data; they belong with PhoxBench Tier 3 (real reconstruction datasets), not Tier 0.
- Sheaf-theoretic neighbor compatibility maps (the `Tᵢⱼ` tuple element from the thesis).
- A `.3dphox` format extension to natively store germ data. For Tier 2, germs are computed at load time from the existing v25/v28 splat positions; saving them is v0.4 work.
- Real-time / interactive rendering (that's Tier 3).
- LPIPS computation (multi-view; bundled with Tier 1.5 item 5).

**Explicitly preserved from Tier 1:**
- The Gaussian path (Tier C splats stay Gaussian; `H = 0` recovers Gaussian).
- All bit-exactness guarantees.
- The honest-manifest doctrine — every render declares which math path it actually used.

---

## 2. Math, formal

### 2.1 The per-pixel evaluation we want

For a splat with center `c`, local frame `R` (3×3), softness `(σ_a, σ_b, σ_n)`, and germ coefficients `θ = (κ₁, κ₂, χ, ω, ζ)`:

```
density(p) = α(view) · exp( -A_θ(u) ),    u = R^T (back_project(p) - c)
```

where `back_project(p)` is the world-space point on the splat's tangent plane closest to the camera ray through pixel `p`. The action is

```
A_θ(u) = min_{s,t} F_θ(u, s, t)

F_θ(u, s, t) = (a - s)²/σ_a²  +  (b - t)²/σ_b²  +  (n - H_θ(s, t))²/σ_n²

H_θ(s, t) = κ₁s² + κ₂t² + χ(s³ - 3st²) + ω(3s²t - t³) + ζ(s⁴ + t⁴)
```

`u = (a, b, n)` is the local-frame coordinate of the back-projected pixel.

The thesis's full softmin (`A = -τ log ∫ exp(-F/τ) ds dt`) reduces to the closest-point F at the minimizer in the τ → 0 limit. We use the closest-point limit; the temperature term `τ` becomes a v0.4 refinement if visual quality demands it.

### 2.2 The Newton solver

Closest-point on the germ surface, starting from `(s₀, t₀) = (a, b)` (project u into the tangent plane and use that as initial guess):

```
∂F/∂s = -2(a-s)/σ_a² - 2(n-H)/σ_n² · ∂H/∂s = 0
∂F/∂t = -2(b-t)/σ_b² - 2(n-H)/σ_n² · ∂H/∂t = 0
```

with

```
∂H/∂s = 2κ₁s + 3χ(s² - t²) + 6ω·s·t + 4ζs³
∂H/∂t = 2κ₂t - 6χ·s·t + 3ω(s² - t²) + 4ζt³
```

The 2×2 Hessian `J = ∇²F` is

```
J_ss = 2/σ_a² + 2/σ_n² · [(∂H/∂s)² - (n-H)·∂²H/∂s²]
J_tt = 2/σ_b² + 2/σ_n² · [(∂H/∂t)² - (n-H)·∂²H/∂t²]
J_st = 2/σ_n² · [∂H/∂s · ∂H/∂t - (n-H)·∂²H/∂s∂t]
```

Update: `(s, t) ← (s, t) - J⁻¹ ∇F`. Two Newton iterations are enough for the quadratic germ (Hessian is constant and J⁻¹ has closed form). For cubic + quartic terms, 3 iterations with a damping factor of 0.7 if Newton overshoots.

**Vectorization:** we don't iterate per-pixel in Python. For each splat, build a (n_pixels_in_bbox, 2) array of `(a, b)` initial guesses and run Newton in numpy batch. Same for the gradient and Hessian.

### 2.3 Reduction to Gaussian (sanity)

When `θ = 0` (flat germ) and `σ_a = σ_b = σ_n`, `H ≡ 0`, so the closest point is `(s*, t*) = (a, b)` after one Newton step (already at the minimum). Then `A = (a² + b²)/σ² + n²/σ² = ‖u‖²/σ²`. With identity-frame `R = I`, this is the standard isotropic Gaussian. With anisotropic `(σ_a, σ_b, σ_n)`, this is the standard EWA-projected anisotropic Gaussian. So the Newton solver reduces *exactly* to vanilla Gaussian rendering when the germ is flat — no special-case code needed.

---

## 3. Per-splat 5-coefficient germ fitter

Extend `fit_synthetic_germs` in `tools/crypsorender/math/germ.py`:

- For each Tier A/B splat, take its `k = 16` nearest neighbors (was 8; need more rows than unknowns for a stable 5-parameter fit).
- Project neighbors into the splat's local frame.
- Build the design matrix `M` of shape `(k, 5)` where the columns are the basis functions `[s², t², s³ - 3st², 3s²t - t³, s⁴ + t⁴]`.
- Solve `M @ θ = n` (least squares) for each splat.
- Clip each coefficient to a sensible range (e.g. `|κ| ≤ 25`, `|χ|, |ω| ≤ 50`, `|ζ| ≤ 100`).

Storage: `germ` field on `SplatBuffer` becomes shape `(n, 5)` instead of `(n, 2)`. v0.4 will add a `.3dphox` chunk to persist these.

---

## 4. PhoxBench Tier 0

### 4.1 Six synthetic stress scenes

Each scene is a small analytic surface or point set with **exact ground truth** so phoxoid vs Gaussian can be measured cleanly without confounding from real-world noise.

| # | Scene | Surface / generator | Why it matters |
|---|---|---|---|
| 1 | **Plane** | `z = 0` over `[-1,1]²` | Both should match. Sanity baseline. |
| 2 | **Sphere** | `x² + y² + z² = 1`, sample upper hemisphere | Smooth surface curvature. |
| 3 | **Saddle** | `z = x² - y²` | Both principal curvatures, opposite signs. Quadratic germ should win cleanly here. |
| 4 | **Fold** | `z² = x³` (Whitney umbrella half) | Smooth fold surface. Quadratic insufficient; cubic germ should win. |
| 5 | **Cusp** | Pearcey caustic: parametrise by `(s, t)`, `(x, y, z) = (s, t, s³ - 3st²)` | Cubic Pearcey term needed. Gaussian should fail visibly. |
| 6 | **Thin sheet** | Two parallel planes at `z = ±0.02` | Both surfaces visible; opacity blending. |

Optional v0.4 additions: crease, two crossing sheets, occlusion edge, noisy patch.

### 4.2 Per-scene benchmark protocol

For each scene:
1. Generate ground-truth point cloud (10,000 points uniformly sampled).
2. Cluster into B blob budgets: `B ∈ {64, 128, 256}` (matches the thesis Bunny benchmark).
3. For each cluster, fit a **Gaussian baseline** (PCA + covariance) and a **phoxoidal blob** (PCA + 5-coefficient germ).
4. Compute per-cluster fitting RMSE: ‖z - Ĥ(s,t)‖.
5. Render each at the same camera, same resolution.
6. Compute image RMSE / PSNR / SSIM phoxoid vs Gaussian.

### 4.3 Killer metric

For each scene, find the smallest Gaussian budget `B_G` such that Gaussian RMSE ≤ phoxoid RMSE at budget B. Report `B_G / B` — "how many Gaussian blobs does one phoxoidal blob replace at equal quality?" If this ratio > 1 on cusp/fold/saddle scenes and ≈ 1 on the plane (where phoxoids can't help), Tier 2 has shown what the thesis predicts.

### 4.4 Output structure (per scene, per budget)

Following the thesis's PhoxBench convention:

```
phoxbench/
  runs/<date>_<scene>_<budget>/
    input_preview.png            # ground-truth point cloud, fixed view
    gaussian_render.png          # Gaussian-only blobs, same view
    phoxoidal_render.png         # phoxoidal blobs (faithful Newton), same view
    side_by_side.png             # both + ground truth in one frame
    error_heatmap.png            # |gaussian - phoxoid| difference image
    metrics.json                 # RMSE, Chamfer, image PSNR, blob counts, killer-metric ratio
    report.md                    # one-paragraph human-readable summary
```

---

## 5. Implementation order (deliverable per step)

Each step ends with something concrete and reviewable.

| Step | Work | Deliverable |
|---|---|---|
| 2.1 | `math/germ.py`: add `closest_point_on_germ(u, theta, sigma)` (vectorized Newton, quadratic germ first). | Unit test: 2D quadratic germ, known minimizer matches analytic answer. |
| 2.2 | Extend `phoxoidal_density_screen` → `phoxoidal_density_faithful` using §2.2. Keep the screen-space version as a `--fast` fallback. | Single-splat test: render one cusp-germ splat in isolation, see the visibly non-bell-shape footprint. |
| 2.3 | Extend `fit_synthetic_germs` to 5 coefficients per §3. Refit the Audi data; save germs to disk so v28 + germs can be rendered. | Per-Tier-A/B splat germ array on disk; histogram of fitted coefficients. |
| 2.4 | `tools/phoxbench/scenes.py`: generators for the 6 Tier 0 scenes (§4.1). | `python tools/phoxbench/scenes.py --scene cusp --out runs/.../input_preview.png` produces a recognizable cusp point cloud render. |
| 2.5 | `tools/phoxbench/run_scene.py`: end-to-end harness — generate scene → fit blobs → render both → metrics → outputs. | Single `python tools/phoxbench/run_scene.py --scene saddle --budget 128` produces the full output directory from §4.4. |
| 2.6 | Run all 6 scenes × 3 budgets = 18 runs. | `phoxbench/runs/...` populated with 18 directories; one summary table comparing all. |
| 2.7 | Final Tier 2 contact sheet: cusp scene side-by-side, killer-metric table, plus rerun of Audi side view with faithful Newton. | `renders/crypsorender_v01/SHOWCASE_T2.png` — the deliverable that proves phoxoids are visibly distinct from Gaussians. |

Estimated wall-clock (CPU-only, sandbox cap considered): each PhoxBench run is ~20 seconds; 18 runs ≈ 6 minutes. The Newton solver development plus tests is the long pole — probably half a day of careful coding.

---

## 6. Numerical-correctness anchors (must all pass before Tier 2 ships)

1. **Newton converges on the quadratic germ.** For `H = κ₁s² + κ₂t²` with `(κ₁, κ₂) = (1, -2)` and a known target u, the solver finds the analytic minimizer to `< 1e-5` in 2 iterations.
2. **Reduction to Gaussian when θ = 0.** For `θ = 0` (flat germ) the rendered image of one splat is byte-identical to the Tier 1 Gaussian rasterizer's output of the same splat. Confirms no regression on the Tier C path.
3. **Fold/cusp visibility.** A single splat with the cusp germ, rendered alone, produces a footprint with non-elliptical level sets visible in the alpha map. (Eyeball + numerical check on level-set asymmetry.)

---

## 7. Honest expected outcomes (so we don't fool ourselves)

I can't pre-empt the actual numbers, but here are the ranges I'd find plausible based on the thesis's bunny benchmark + general splat-rendering intuition:

| Scene | Expected outcome |
|---|---|
| Plane | Ratio ≈ 1.0 (no phoxoid advantage, neither should phoxoids hurt) |
| Sphere | Ratio 1.1–1.3× (curvature germ ~ matches a tighter Gaussian on a smooth surface) |
| Saddle | Ratio 1.5–2× (κ₁ and κ₂ of opposite sign — Gaussians blur through the saddle point) |
| Fold | Ratio 2–4× (cubic germ catches the fold; Gaussians smear) |
| Cusp | Ratio 3–8× (this is where Pearcey terms shine; Gaussians fundamentally can't do this shape) |
| Thin sheet | Ratio 1.0–1.5× (depends on σ_n; mostly an opacity test, less a germ test) |

If we get **ratio > 2× on cusp at budget=64**, the thesis has been validated in a measurable way. If **all ratios are ≈ 1.0**, Tier 2 has measured a real "phoxoids don't help" finding — also valuable, but a project-shaking one. The benchmark is the truth either way.

---

## 8. Open architectural questions for Bug

These need answers before §5 step 2.4 starts (the PhoxBench scenes can't be generated without them):

1. **Pearcey vs full polynomial.** I'm proposing the 5-coefficient basis `{s², t², s³-3st², 3s²t-t³, s⁴+t⁴}`. The thesis lists this exact set. Want me to add the asymmetric quartic `s⁴ - t⁴`? Adds one coefficient (6 total); marginally more expressive but complicates fitting.
2. **Synthetic vs real evidence.** All Tier 2 work uses synthetic germs auto-fitted from local k-NN. The thesis ultimately wants evidence-fitted germs (RGB residual, edge maps, etc) but those need a Tier 3 dataset. Confirm Tier 2 stays synthetic-only.
3. **PhoxBench scene budget sizes.** I'm proposing `B ∈ {64, 128, 256}`. The thesis went up to 1024 on bunny. Want me to add 512 and 1024? More compute (~2× to ~4×).
4. **Should Tier 2 also bundle Tier 1.5 items 4 + 5 (object-mask + multi-view)?** They'd be cheap to fold in once the sandbox returns. Or keep them separate?
5. **Compute budget.** I estimated half a day of careful coding + ~6 min of bench compute. Is there a hard ceiling I should respect (e.g. "wrap by Friday" / "no more than X agent-spawns")?

---

## 9. What Tier 2 does NOT change

- The `.3dphox` formats (v25, v27, v28) stay as-is. v0.4 work might add a germ chunk; for now germs are loaded from disk separately.
- Compression numbers from Tier 1.5 don't change. Tier 2 is renderer + benchmark, not codec.
- The "no GPU dependencies" rule is still absolute.
- All Tier 1 deliverables stay in place. SHOWCASE_T1_final.png is the Gaussian-fidelity baseline against which Tier 2's improvements are measured.
