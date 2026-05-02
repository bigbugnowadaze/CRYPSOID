# v32.5 — kNN-graph soft shadows (one-pager)

**Status:** draft for sign-off, 2026-05-01.
**Goal:** add shadows to lit phoxoid scenes — without ray tracing, without shadow maps, without spatial structures other than the v31 kNN graph already in the format.
**Depends on:** v31 (normals + kNN edges chunk) and v32a (lights). Sits *between* v32a and v33 in the spec sequence.

## Why this spec

Lambertian lighting (v32a) gets you "the surface knows where the sun is." It does not get you "this wall casts a shadow on that floor." Standard shadow techniques need extra machinery:

| Technique | What it needs | Fits CRYPSOID? |
|---|---|---|
| Shadow rays (ray tracing) | BVH or grid; per-pixel ray cast | Heavy; needs spatial structure we don't store |
| Shadow maps | Depth render from light's POV; reprojection | Aliasing; multiple passes; ill-defined for soft splats |
| Screen-space shadows | March in screen space | Limited to visible geometry; misses off-screen blockers |
| **kNN-graph soft shadows** | v31 kNN edges (already there) | **Native fit — graph query is O(k=4) per phoxoid** |

The v31 kNN edges chunk was originally pitched as the bridge to "geometric attention" and LOD. Shadows are the second use case for it. Each phoxoid asks its k=4 neighbors: "do any of you sit between me and the light?" If yes → partial shadow. The graph IS the shadow query structure.

**This is genuinely phoxoidal-specific.** Surfels and standard splats don't carry an explicit neighbor graph; they'd have to build one at render time or fall back to ray tracing. With v31 the graph is part of the format — shadows become a 4-neighbor walk, no spatial structure required.

## Non-goals

- Hard, sharp shadows. Soft by construction (a feature for fuzzy splat scenes, but a limit for stylized rendering).
- Long-range shadows (a distant building shadowing a nearby surface). Local-only by default; longer-range can be added later via multi-hop walks or recursive `k=4`-of-`k=4`.
- Cast shadows from non-phoxoid geometry (UI overlays, helper meshes). Out of scope.
- GPU shader implementation — Python/numpy first, viewer port later.

## Algorithm

For each phoxoid `P` with normal `N`, kNN neighbors `{N₁, N₂, N₃, N₄}`, and each light direction `L`:

```
shadow_factor(P, L) = 1.0
for each neighbor Nᵢ:
    d   = Nᵢ.position − P.position           # offset vector
    t   = dot(d, L)                          # signed distance along light direction
    if t <= 0: continue                      # neighbor is behind P relative to light
    perp = d − t·L                           # perpendicular component
    rᵢ  = max(Nᵢ.scale)                      # neighbor's effective support radius
    
    # Gaussian falloff in perpendicular distance, weighted by neighbor opacity
    occlusion_i = exp(− |perp|² / (2·rᵢ²)) · Nᵢ.opacity
    
    shadow_factor *= (1.0 − occlusion_i)

shadow_factor = clamp(shadow_factor, 0.0, 1.0)
```

Compose with v32a/v32b shading:
```
shaded_with_shadow = shadow_factor · diffuse_term + ambient_term
                                                     ↑
                              (ambient is not shadowed by the sun)
```

### Why this works
- **Soft by construction.** Gaussian falloff in `|perp|` produces a smooth shadow boundary; opacity weighting handles thin surfaces gracefully.
- **Cheap.** ~12 flops per (phoxoid, neighbor, light) pair. Audi at 763k blobs × 4 neighbors × 1 sun light = ~36M flops, sub-second on CPU.
- **No spatial structure needed.** The kNN edges are already in the format; we just walk them.
- **Naturally tier-aware.** A Tier-A phoxoid's neighbors might include Tier-C blobs and vice versa — the algorithm doesn't care, it shadows across tier boundaries correctly.

### What it doesn't capture
- **Long-range occlusion.** A neighbor in `{N₁..N₄}` is by definition spatially close. A wall 5 m away can't shadow a phoxoid through this scheme. (Mitigation: optional 2-hop walk; expensive.)
- **Self-shadowing of the same phoxoid.** v32b's curvature self-shadowing handles this; v32.5 handles *between-phoxoid* shadows.
- **Direct sun disk vs ambient.** The `shadow_factor` only attenuates the directional term. Ambient is treated as unblocked (a simplification — true ambient occlusion is what v32b's curvature AO covers locally and what graph-AO would cover globally).

## Optional: graph-based ambient occlusion (v32.5 sub-feature)

The same neighborhood walk gives ambient occlusion almost for free. Instead of testing against one light direction, test against the *upper hemisphere*: count how much of the local sky is blocked by neighbors.

```
ambient_occlusion(P, N) = sum over neighbors Nᵢ:
    d = Nᵢ.position − P.position
    t = dot(d, N)                            # how "above" P is the neighbor (along normal)
    if t <= 0: continue                      # neighbor is below the surface
    
    aoᵢ = exp(− |d|² / (2·R²)) · Nᵢ.opacity
            ↑
        R = ambient occlusion radius (config; default 2× median scale)
    
    ao_total += aoᵢ

ao_factor = exp(−γ · ao_total)               # γ default = 1.0
```

Compose:
```
shaded_ambient_with_ao = ambient.rgb · estimated_albedo · ao_factor
```

This is *complementary* to v32b's curvature AO:
- v32b curvature AO: per-phoxoid, based on its own bending. "Concave patches darken locally."
- v32.5 graph AO: between-phoxoid, based on actual blockers. "Phoxoids in densely-packed regions darken because real blockers are nearby."

Both compose multiplicatively: `final_ambient_factor = curvature_ao · graph_ao`.

## Format additions

**None.** v32.5 uses v31 normals + v31 kNN edges + v31 scales/opacity (already in v28). Pure renderer change.

## Render-time configuration

Renderer flags:
```
--shadows={off, knn, knn+ao}
    off    : v32a only (no shadows)
    knn    : per-light shadow_factor from kNN walk
    knn+ao : also apply graph-based ambient occlusion

--shadow-strength=1.0     # multiplier on occlusion sum (0 = disable, 2 = darker shadows)
--ao-radius=auto          # ambient occlusion radius; "auto" = 2 × median scale
--ao-gamma=1.0            # ambient occlusion falloff
```

## Acceptance gates (v32.5)

1. **Plane test.** A flat plane of phoxoids with one neighbor "wall" of phoxoids: shadow visible on the side opposite the light, soft falloff at the boundary.
2. **No-neighbor test.** A single isolated phoxoid (kNN points to far-away blobs): `shadow_factor ≈ 1.0` (no shadow, since no neighbor lies between it and the light).
3. **Symmetry test.** Light from +X vs −X on a symmetric scene produces mirror-image shadows.
4. **Performance gate.** Audi at 763k blobs, 1 directional light: shadow pass adds < 200 ms on CPU at 512² render.
5. **Composition test.** v32a only, v32a+v32b, v32a+v32b+v32.5 all render without artifacts; each adds visible darkening progressively in concave / occluded regions.
6. **CI.** Synthetic scene (phoxoid sphere atop a phoxoid plane, single sun light): shadow under the sphere is visible; ground beyond the sphere's projected footprint is unshadowed.

## Cost on the Audi A5 (763,800 phoxoids)

| Pass | Per phoxoid | Per scene per light | Audi @ 512² (1 light) |
|---|---:|---:|---:|
| v32.5 shadow walk | ~50 flops | 38 M flops | ~80 ms |
| v32.5 graph AO | ~50 flops | 38 M flops | ~80 ms |
| Combined | ~100 flops | 76 M flops | **~160 ms** |

(Numbers are upper-bound estimates; actual depends on numpy vectorization. Practical target: shadow pass < 200 ms at 512², well below the existing render time.)

**Format bytes added: 0.** The whole spec is a renderer feature.

## Honest scope summary

| Claim | True / partial / aspirational |
|---|---|
| v32.5 gives soft shadows on lit scenes | **True**, the day v31 kNN edges + v32a lights both exist. |
| It uses phoxoidal-graph structure that no other splat format ships natively | **True** — the kNN edges chunk is the v31-specific addition that makes this O(k) instead of O(n). |
| Shadows are accurate at long range | **Partial** — local-only by default; multi-hop walks possible but expensive. |
| Graph-based AO replaces SSAO and ray-traced AO | **Partial** — qualitatively similar; quantitatively coarser. Composes cleanly with v32b curvature AO so total shading reads correctly. |
| This is the path to fully ray-traced shadows / global illumination | **Aspirational** — those are v40+, where Pearcey-class germ caustics also live. |

## Suggested phasing within v32.5

1. Sign off v31 and v32+v33 first (this spec depends on both).
2. Implement bare directional-light shadow pass (no AO) — single sun, get a shadow on the ground.
3. Add multi-light support — sum shadow contributions per light.
4. Add graph AO as a separate flag (opt-in).
5. Wire flags into the WebGL viewer (port the kNN walk to GLSL fragment shader).

Each step is a separate phased reviewable artifact, per project convention.
