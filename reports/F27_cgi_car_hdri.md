# F.27 — CGI v2: stylized car + HDRI environment

**Date:** 2026-05-03

## What it is

A companion to F.26's studio scene. Same idea — clean procedural input
fed through the existing photoreal renderer — but this time with:

- A **stylized car** built from primitive shape samples (rounded boxes for
  the lower body and cabin, cylinders for 4 wheels, spheres for chrome
  wheel hubs and headlights, plus a small tinted-glass band around the
  cabin top)
- An **HDRI environment** (the synthesized studio HDR from
  `outputs/test_smoke_hdr.npy` made for the F.26-era HDRI smoke test)
  driving both ambient light and glossy/mirror reflections — instead of
  the procedural sky+ground from F.26
- The HDRI sampled directly for the **render background**, so the sky
  behind the car comes from the same env map the chrome reflects

## What shipped

| File | What |
|---|---|
| `tools/build_cgi_car_phox.py` | Procedural SDF-style car → `cgi_car_v1.3dphox` |
| `tools/render_cgi_car_hdri.py` | Renders the car through the photoreal stack with HDRI env |
| `tools/build_cgi_3way_panel.py` | 3-panel: scan-Audi / CGI-studio / CGI-car |
| `outputs/cgi_car_v1.3dphox` | 359k splats; v25 + v31 (normals/edges/material_hints) + v40 (kappa/cusp) trailers |
| `outputs/cgi_car_v1.pbr.npz` | Ground-truth per-splat (albedo, metallic, roughness, F0, kd) |
| `outputs/cgi_car_v1.scene.json` | Scene manifest |
| `renders/crypsorender_v01/SHOWCASE_CGI_CAR_HDRI.png` | 1k car render |
| `renders/crypsorender_v01/SHOWCASE_CGI_CAR_HDRI_2k.png` | 2k version |
| `renders/crypsorender_v01/SHOWCASE_CGI_3WAY.png` | 3-panel deliverable |
| `renders/crypsorender_v01/SHOWCASE_CGI_3WAY_thumb.png` | 1920px thumb |

## Scene composition

| Part | Splats | Material |
|---|---:|---|
| ground plane | 70k | matte gray |
| lower body (rounded box) | 80k | red painted, mildly metallic |
| cabin (rounded box) | 50k | red painted |
| cabin window strip | 35k | tinted dark glass |
| 4 wheels (cylinders) | 80k total | matte black rubber |
| 8 wheel-hub disks | 32k total | chrome (metallic 0.92) |
| 2 headlights (spheres) | 12k total | warm white |
| **Total** | **~359k** | |

Built with golden-spiral / face-uniform / cylindrical analytic samplers
(reusing the helpers from F.26's `build_cgi_studio_phox.py`). Per-splat
normals are exact, not MLS-estimated. PBR is ground-truth from the build,
not recovered from SH bands.

## Renderer integration

Same `apply_photoreal_lighting` stack as F.26 / Audi v2, with one swap:

```python
from crypsorender.math.environment import HDRIEnvironment
env = HDRIEnvironment(HDR_SRC, intensity=1.0)
shaded_hdr = apply_photoreal_lighting(..., environment=env, ...)
```

Where the F.26 studio uses the procedural `StudioEnvironment` (sky-up /
ground-down sweep), this F.27 render uses an actual HDR equirect map. The
chrome wheel hubs reflect the studio's actual brightness gradient instead
of a procedural one. Bg pixels also sample the HDRI so the world behind
the car is consistent with what the chrome sees.

## What it shows

The `SHOWCASE_CGI_3WAY.png` panel: same renderer, three .3dphox sources,
three distinct aesthetics:

1. **Trained-3DGS scan** — fuzzy splat-cloud Audi from the v40 photoreal
   render (the scan-aesthetic).
2. **Procedural CGI studio (F.26)** — chrome sphere + matte ball + plastic
   cube on a clean ground, smooth procedural sky.
3. **Procedural CGI car + HDRI (F.27)** — red car with chrome hubs, lit by
   an HDR studio environment.

The aesthetic is a property of the input data, not the pipeline. F.26
proved that with one clean source. F.27 extends the proof: drop in an HDR
environment and a stylized object, get a "hot-wheels-on-a-cyclorama"
render — same code, different inputs.

## Honest caveats

- The HDR is the **synthesized smoke-test** map, not a downloaded studio
  HDR. It's mostly a sky-ground gradient with mild banding. A real HDRI
  Haven asset would give richer reflections in the chrome.
- The car is intentionally **stylized / low-poly**. Adding hood-line
  cutouts, side-mirror fillets, or wheel-spoke geometry is a content
  question — none of it requires renderer changes.
- The splat-softness at silhouettes is still there (intrinsic to 3D
  Gaussian rendering). The CGI-source pathway proves the renderer
  responds *correctly* to clean PBR + HDRI; it doesn't make Gaussian
  splats stop being Gaussian.

## What it closes

This is the F.26 sequel that was promised in `AUDI_HERO_attempt.md` →
`F26_CGI_source_demo.md`:

> Adding any of these is now a content question, not a renderer question.
> [...] With a richer HDR (like the studio HDRIs that ship in Bar 1's
> HDRI loader smoke test), chrome would show actual studio reflections.

We did exactly that. Chrome reflects the studio. The pipeline holds.
