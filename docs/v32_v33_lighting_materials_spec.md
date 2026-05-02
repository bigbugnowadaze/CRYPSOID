# v32 + v33 — Lighting & Materials Spec (one-pager)

**Status:** draft for sign-off, 2026-05-01.
**Goal:** make `.3dphox` scenes *relightable* — the renderer can move a sun, change time of day, toggle materials — instead of being frozen with whatever lighting the SH coefficients baked in.
**Depends on:** v31 (explicit normals on every phoxoid). v32a is the unlock the moment v31 ships normals.

## Why pair v32 with v33

Lighting without materials is half a story. A normal vector lets you compute `N · L` for Lambertian shading, but if every phoxoid's SH already bakes in *both* surface color and lighting, applying a new light just doubles up. To do real relighting you have to *separate* what the SH currently tangles together. v33 adds the minimum format slot for that separation. Shipping them as one signed-off spec avoids a wasted cycle where v32 lighting looks wrong because v33 hasn't landed.

The phoxoid math contributes specifically to v32**b** (curvature-aware lighting) and to v33's specular-peak prediction. v32**a** is standard graphics math; v32b is the part that justifies "this is a CRYPSOID spec" rather than "Lambert bolted onto a splat renderer."

## Non-goals

- Full PBR (Cook-Torrance, Disney BRDF, GGX) — that's v34+.
- Multi-light shadows via ray-marching — v32.5 (separate spec, depends on v31's kNN edges).
- Image-based lighting / HDRI environment maps — v34+.
- Transparency / refraction / caustics (TransparentGS-class) — v40+, where the Pearcey-class germ math becomes the natural fit.
- Temporal `.phoxdelta` (volumetric video) — v34, separate spec.

---

## v32a — Standard lighting (Lambert + ambient + sun)

### What's added
**Nothing in the `.3dphox` format.** Lights are scene-external, configured at render time. Pure renderer change.

### Renderer config
A new `lights` block in render config (JSON or CLI args):

```
lights:
  ambient:        rgb=[0.10, 0.10, 0.12]
  directional[0]: dir=[0.3, -1.0, 0.4], rgb=[1.0, 0.95, 0.85], intensity=1.0
  directional[1]: dir=[-0.5, -0.2, 0.7], rgb=[0.4, 0.5, 0.6],  intensity=0.3
```

(Multiple directional lights allowed; point/spot lights deferred to v32.5.)

### Math (standard)
For each phoxoid with normal `N` and SH-decoded base color `C(view_dir)`:

```
shaded = ambient.rgb · C
       + Σ_lights  light.rgb · C · max(0, N · L)
```

Where `L` is the light direction. SH base color `C` is evaluated in the view direction (current behavior); v33 will optionally replace `C` with a separated albedo.

### Honest scope
- This works on **any** oriented surface element. Surfels, splats, polygons. No phoxoidal-math-specific contribution.
- It's the prerequisite that makes the user *see* lit geometry. Without it, every visual is "frozen lighting from the training views."
- Visually noticeable change: re-render the Audi with sun moving from morning to noon to evening. SH-baked color modulates by `N · L`; metal panels brighten on the sun-facing side, darken on the shadow side. Not photoreal, but legible as "lit 3D."

---

## v32b — Curvature-aware lighting (the phoxoidal contribution)

### What's added
**Nothing in the format.** Uses existing v31 normals + v31 germ chunks. Renderer evaluates a few extra terms per phoxoid.

### The three uses of the germ

#### 1. Self-shadowing from curvature
Standard Lambert assumes the surface is a flat tangent plane at the phoxoid. A real phoxoid has curvature (κ₁, κ₂) — at grazing angles the patch's *own bending* partially occludes itself. Replace plain `max(0, N · L)` with:

```
visibility = max(0, N · L) · (1 − β · |κ_eff| · (1 − N · L))
```

Where `κ_eff = sqrt(κ₁² + κ₂²)` (a single scalar curvature magnitude per phoxoid) and `β` is a tunable scalar (default ≈ 0.5). Effect: at normal incidence (`N · L ≈ 1`) the curvature term vanishes and shading is unchanged. At grazing angles (`N · L ≈ 0`) high-curvature patches darken further than flat ones — correctly modeling that they're tipping into self-shadow.

This is **mathematically motivated** by the germ being a real local quadric, not a heuristic darkening hack. Surfels can't do this; they have one normal and call it done.

#### 2. Curvature-modulated ambient occlusion
Highly-curved patches (high `|κ_eff|`) sit in a self-occlusion well — they should receive less ambient light. Add an analytic per-phoxoid ambient factor:

```
ambient_factor = 1.0 − α · tanh(|κ_eff|)
```

`α` default ≈ 0.3. Applied to the ambient term only:

```
shaded_ambient = ambient.rgb · C · ambient_factor
```

One parameter, no extra math. Gives concave regions a natural darkening without requiring screen-space ambient occlusion.

#### 3. Cusp-specular (deferred to v32c)
The cubic terms (χ, ω) and quartic (ζ) in the germ encode cusp/swallowtail behavior. A specular highlight is mathematically a sharp peak at a specific reflection angle. The germ knows analytically where the surface bends, so the *exact* reflection direction at a sub-blob position is known, not approximated.

Concretely: for a viewer direction `V` and light `L`, standard Phong specular is `(max(0, R · V))^shininess` where `R = reflect(L, N)`. With the germ, we can compute `R(s, t)` as a function of position within the phoxoid's local (s, t) frame — the cubic terms give the local "twist" that classical Phong assumes is zero.

This is real novel math. **Defer to v32c** because:
- Sub-pixel reflection-direction integration is non-trivial to implement correctly.
- Visual benefit is subtle compared to (1) and (2).
- (1) and (2) are 2-line shader changes; (3) needs careful numerics.

If we ship v32b with (1) + (2) we already get the "phoxoidal-specific lighting" win and we earn the v32c cusp-specular as a follow-on once the simpler curvature contributions are validated.

### Acceptance gates (v32b)
1. Render the same scene with `--curvature-shading=off` (= v32a) and `--curvature-shading=on` (= v32b). Visual A/B: curvature-shading darkens grazing-angle and concave regions but never brightens.
2. Sphere stress test: a single phoxoid with κ_eff = 1.0 should darken at least 20% at grazing incidence vs flat-shading.
3. Plane stress test: a phoxoid with κ_eff = 0 should produce *identical* pixels in v32a and v32b (curvature term vanishes).
4. PhoxBench Tier 0 regression: re-run synthetic scenes with v32b shading on. Fit RMSE should be unchanged (lighting is a renderer concern, not a fitter concern).

---

## v33 — Material hint + albedo/lighting separation

### What's stored
For each of N phoxoids:

| Field | Bytes | Encoding |
|---|---:|---|
| `material_hint`         | 1 | u8 enum (see below) |
| `confidence`            | 1 | u8, 0–255 |
| `view_dependence_score` | 1 | u8, 0–255 — magnitude of SH bands 1-3 relative to band 0 |

**Total: 3 bytes/phoxoid.** On the Audi (763,800 blobs): 2.19 MB. (Amendment 2026-05-01 — added `view_dependence_score` per ChatGPT relightable-GS analysis. Useful as quality signal + material classifier input + render-time blend weight.)

### `material_hint` enum
```
0 = unknown
1 = diffuse / mostly Lambertian
2 = glossy / specular
3 = mirror-like
4 = transparent / refractive
5 = emissive
6 = floater (background junk; Clean-GS / EFA-GS style)
```

These are *coarse classes* — enough to drive renderer behavior; not committing to a specific BRDF. Class 6 (floater) was added on 2026-05-02 after the v31-implementation pass made the floater problem concrete (the full-density Audi render had its body washed out by halo splats — `SHOWCASE_HIGHEST_MAX.png`).

### `confidence` byte
0 = "I don't know what this is" (treat as enum=0, fall back to SH).
255 = "high confidence in the assigned class."
Renderer uses confidence as a blend weight: `output = lerp(SH_baked, material_aware, confidence/255)`.

### How material_hint is derived

#### Phase 1 — heuristic (ships with v33)
From per-phoxoid SH coefficient distribution:

| Pattern | Class assigned |
|---|---|
| Mostly DC band (band 0), low energy in bands 1–3 | 1 (diffuse) |
| High energy in bands 1–3, smooth angular variation | 2 (glossy) |
| Sharp angular peaks in band-3 SH | 3 (mirror-like) |
| Negative ambient + bright SH peaks | 5 (emissive) |
| DC near zero + high opacity gradient | 4 (transparent) — heuristic, low confidence |
| Low surface-variation + isolated (long kNN edges) + low opacity | **6 (floater)** — EFA-GS-style |
| Anything else | 0 (unknown) |

**EFA-GS-style floater detection (Phase 1.5, added 2026-05-02):** The Clean-GS / EFA-GS class of papers identifies "floater" splats — background junk that doesn't represent real surface — by *low-frequency-first residual diagnosis*. CRYPSOID's cheap analog (composable from data we already have):

1. Long kNN edges (mean of edges chunk's distances > p90 of all edges) → splat sits in a sparse region.
2. Low surface-variation index (κ ≈ 0) → no real curvature signal; suggests it's not a coherent surface element.
3. Low opacity (sigmoid logit < threshold) → already partially fading.

Combined: classify as `floater` with confidence proportional to how strongly all three conditions agree. Renderer can then either render with reduced opacity OR skip entirely (LOD prune). The `.phoxdelta` demo (`SHOWCASE_v31delta_compare.png`) showed the visual win of fading the floor halo — a derived `material_hint=floater` field would automate that decision instead of needing an explicit y-coordinate threshold.

### Field 4 — `mip_zoom` (Mip-Splatting style, added 2026-05-02)

A 1-byte per-phoxoid "max-zoom frequency" field that tells the renderer at what camera distance the splat begins to alias (Mip-Splatting). Below the zoom threshold, the renderer applies a 2D pre-filter to the splat to prevent the "small splats become aliased dots" failure mode.

| Field | Bytes | Encoding |
|---|---:|---|
| `mip_zoom` | 1 | u8 — log₂(max-frequency in screen pixels), clamped to [0, 255] |

Adds 1 byte/blob (=763 KB on Audi). Brings total v33 cost to **4 bytes/blob (~+9.5% on Audi)**, still much cheaper than v31's +47%.

### FeatureGS classification (optional, 2026-05-02)

Use eigenvalues `(λ_0, λ_1, λ_2)` of the local covariance (already computed in MLS pass) to classify the splat's local structure:

| Classification | Eigenvalue ratio | Notes |
|---|---|---|
| linear (edge / wire) | λ_2 ≫ λ_1 ≈ λ_0 | one dominant direction |
| planar (surface) | λ_2 ≈ λ_1 ≫ λ_0 | two dominant; one small (the normal) |
| scattered (volume / floater) | λ_2 ≈ λ_1 ≈ λ_0 | no dominant direction |

This is *additive to v32b's κ* — same eigenvalues, ternary classification instead of scalar. Doesn't need a new field; can be folded into `material_hint` as values 7/8/9 if useful, or computed at render time from the existing curvature signal.

Confidence is calibrated from the *strength* of the matching pattern (how cleanly the SH fits the template). Cheap to compute; runs in seconds across all 763k Audi splats.

#### Phase 2 — multi-view photometric variation (deferred)
The GS-2M approach: if the original training views are available, compute per-phoxoid photometric variance across views. High variance under SH-predicted appearance → glossy. This is *better* but requires going back to source images. For the Audi PLY (xyz+SH only, no source images), Phase 1 is what we get.

#### Phase 3 — learned classifier (out of scope for v33)
A small classifier trained on labeled splat data. Not building it now — but the format slot is in place so it can be populated later.

### Albedo / lighting separation

When `confidence > 0`, the renderer can optionally split the SH into approximate (albedo, lighting_response):

```
estimated_albedo  = SH.dc                              # band 0 only
estimated_lighting = SH(view_dir) − SH.dc              # bands 1-3 contribution
material_response  = BRDF(material_hint, N, V, L)      # depends on hint

shaded = ambient · estimated_albedo · ambient_factor
       + Σ_lights  light.rgb · (
            estimated_albedo · max(0, N · L) ·  visibility    # diffuse (Lambert + curvature)
          + material_response                                  # glossy / mirror / etc.
        )
       + α · estimated_lighting                                # leftover SH that we can't separate cleanly
```

Where `α` is a "trust the SH" weight (high when `confidence` is low, low when high). This is approximate — we're separating a tangle that was never explicitly separated — but it's *good enough* to enable:

- **Time-of-day relighting:** move the sun, the diffuse term recomputes with the new direction; the leftover SH provides residual ambient-like fill.
- **Material override:** force a class (e.g. "treat all phoxoids in this region as metallic") for stylized renders.
- **Selective re-coloring:** change albedo while keeping lighting.

### Chunk layout
New chunk type in `.3dphox`: `material_hints` (chunk_id `0x14`).

| Field | Bytes |
|---|---:|
| version | 1 (currently `0x01`) |
| reserved | 1 (zero) |
| count | 4 (must equal N) |
| payload | N × 2 |
| CRC32 | 4 |

### Acceptance gates (v33)
1. `material_hints` chunk round-trips: write → read → byte-identical.
2. Heuristic derivation on a v31+v33 build of Audi produces a sensible distribution (e.g. *not* 100% one class; expect 60–80% diffuse, 10–20% glossy, small fractions of others).
3. Renderer toggle `--material-mode={off,baked,hints}`:
   - `off` = SH directly (current behavior, baseline).
   - `baked` = SH + v32a Lambert (lighting bolt-on, no separation).
   - `hints` = albedo/lighting separation per material_hint, blended by confidence.
4. Visual A/B between the three modes shows graceful degradation, not catastrophic breakage.
5. CI: build a synthetic scene with known materials (matte sphere, mirror sphere, glass sphere). Heuristic must classify ≥3 of 5 classes correctly across the scene.

---

## Combined cost on the Audi A5 (763,800 phoxoids)

| Addition | Per-blob bytes | Total bytes | vs current v28 archive |
|---|---:|---:|---:|
| v32a (lighting) | 0 | 0 | 0% — renderer-only |
| v32b (curvature shading) | 0 | 0 | 0% — uses v31 germ |
| v33 (material_hint + confidence) | 2 | 1,527,600 | +4.7% |
| **v32 + v33 combined** | **2** | **~1.5 MB** | **+4.7%** |

(For comparison, v31 alone was ~+47%. v32+v33 is a much cheaper expansion because lighting and material *hints* are tiny; lighting itself stores nothing.)

## Suggested phasing

1. Sign off v31 first (normals are the prerequisite).
2. Implement v31 Addition 1 (normals chunk + writer).
3. Ship **v32a lighting** — visible win in one cycle, lets users move a sun.
4. Ship **v32b curvature shading** as a flag; A/B vs v32a to prove the math contribution.
5. Ship **v33 material_hints chunk + heuristic derivation**; renderer adds the toggle.
6. Optional v32c **cusp-specular** if v32b validates and the appetite is there.
7. Defer materials *Phase 2* (multi-view photometric variation), v34 temporal `.phoxdelta`, and v40 transparency/caustics — separate specs.

Each step is a separate phased reviewable artifact, per project convention.

## Honest scope summary

| Claim | True / partial / aspirational |
|---|---|
| v32a gives "lit 3D" — sun moves, surfaces respond | **True**, standard math, works the day v31 normals land. |
| v32b is phoxoidal-math-specific | **True** for self-shadowing + AO; deferred for cusp-specular. |
| v33 enables real relighting | **Partial** — approximate, because SH was never explicitly material-separated. Good enough for "time of day," not enough for full PBR. |
| v33 fields can be populated correctly | **Partial** — heuristic ships with v33; multi-view photometric (the better way) needs source images we don't have for the Audi. Format slot is ready for the future. |
| This is the path to caustics / glass / transparency | **Aspirational** — those are v40+. v33's `material_hint` enum at least reserves the slot. |
