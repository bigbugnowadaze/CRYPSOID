# F.28 — Inverse rendering: albedo recovery on the trained-3DGS Audi

**Date:** 2026-05-03

## TL;DR

Strip the baked-in capture-time lighting from the v40 Audi's SH-DC so the
photoreal renderer can re-light cleanly without compounding the original
shading. Implementation is an **inverse-Lambert** — fit a single dominant
light direction L from the data, fit ambient/diffuse split (A, D) by
binning brightness vs `n·L`, then divide each splat's `sh_dc` by its
modeled illumination.

Result: underside splats brighten ~17%, up-facing stay roughly the same,
brightness becomes much flatter across normal directions. The renderer
then adds its own three-point lights on a "neutral" base.

## Method

For each splat, the trained-3DGS DC term is the view-averaged colour:

```
sh_dc ≈ albedo · I(n)
where I(n) = A + D · max(0, n · L)
```

Recovery:

```
albedo = sh_dc / max(I(n), epsilon)
```

Two estimation steps, both deterministic and cheap:

1. **Dominant light L** — Fibonacci-sample 18 candidate directions, score
   each by Pearson correlation between `max(0, n·L)` and `brightness`,
   keep the best. On the Audi this gives `L ≈ (+0.085, +0.972, −0.218)` —
   essentially +Y with a tiny back-tilt, matching how the scan was lit.
2. **Ambient/diffuse split (A, D)** — bin splats by `n·L`, take mean
   brightness per bin, fit `brightness = A + D · cos_t` by linear
   least-squares. On the Audi: `A=0.806, D=0.194` (renormalised so A+D=1).
   The relatively small D matches the modest 25% per-bin brightness
   spread we saw — the scan lighting was diffuse, not a hard sun.

After `albedo = sh_dc / max(A + D·max(0,n·L), 0.20)` and clipping to [0,1],
the per-bin brightness equalises:

| `n·Y` bin | Original brightness | Recovered brightness |
|---|---:|---:|
| [0.0, 0.2] | 0.566 | **0.663**  (+17%) |
| [0.2, 0.4] | 0.560 | 0.640  (+14%) |
| [0.5, 0.7] | 0.575 | 0.623  (+8%) |
| [0.8, 1.0] | 0.693 | 0.703  (+1%) |

Up-facing splats (where `n·Y` is large) had baked-in diffuse light, so
recovery moves them less. Underside splats had only ambient, so they
brighten substantially.

## Files shipped

| File | What |
|---|---|
| `tools/build_audi_relit.py` | Builds `v40_audi_full_relit.3dphox` |
| `tools/render_audi_relit.py` | Renders the relit file via the photoreal stack |
| `tools/build_audi_relit_panel.py` | A/B panel: original v2 vs relit |
| `outputs/v40_audi_full_relit.3dphox` | 763,800 splats, all v31/v40 trailers preserved verbatim, only `dc_rgb_opacity_u8` rewritten |
| `outputs/v40_audi_full_relit.meta.json` | Estimated L, A, D, scores |
| `renders/.../SHOWCASE_AUDI_RELIT.png` | 1k relit render |
| `renders/.../SHOWCASE_AUDI_RELIT_2k.png` | 2k version |
| `renders/.../SHOWCASE_AUDI_RELIT_AB.png` | A/B comparison panel |

## What it shows

Compare `SHOWCASE_AUDI_PHOTOREAL_v2_2k.png` (baked-in) against
`SHOWCASE_AUDI_RELIT_2k.png` (recovered):

- The car body silhouette is **more visible** in the relit version — the
  underside no longer reads as solid black under the dark studio grade.
- Side panels and hood retain their car-coloured tone instead of being
  crushed by the shadow term.
- Rim-lit highlights (back-of-car) still appear where the rim light hits,
  but they're now true rim lights, not "rim light on top of capture
  light".

The floor-slab + scan halo are unchanged because they have no preferred
normal direction; their albedo recovery is essentially a small
per-channel rescale.

## Honest caveats

1. **Single-light model** — the real capture probably had multiple lights
   (windows + ambient room). One-direction inverse-Lambert is the simplest
   robust model; it absorbs the rest into A.
2. **Lambertian assumption** — chrome trim and glass panels aren't
   Lambertian. The recovery slightly under-corrects them. A per-material
   pass would help (use the `material_hint` chunk to apply a different
   formula to mirror/glass splats).
3. **Forced-up normals** — `derive_normals_mls` flips all normals toward
   +Y, so true downward-facing surfaces have wrong normals. Their
   recovery is the geometric average of "up" and "down" — not ideal but
   harmless because they're below the camera anyway.
4. **No specular separation** — the residual `sh_rest` (bands 1–3) was
   left untouched. The view-dependent specular highlights are still where
   the capture put them. Stripping those would need a per-band fit.

These are addressable with more sophisticated models (NeRD, NeRFactor,
IRON), but they cost research-paper effort. This Lambert-only first pass
captures the dominant baked-in DC term, which is most of the visible
"double-lit" problem on the Audi.

## What it closes

The relit `.3dphox` is now a **drop-in replacement** for the original v40
in any of the existing renders (`render_audi_photoreal_v2`,
`render_audi_3way`, `render_audi_hero_v3`). Same geometry, same v31/v40
aux chunks, just cleaner DC colour. The "input data, not pipeline"
findings from F.26/F.27 stand: with cleaner input, the renderer produces
cleaner output.

For the scan-Audi specifically, this is the cleanest re-light we have
without going to Blender or doing inverse-rendering research-paper work.
