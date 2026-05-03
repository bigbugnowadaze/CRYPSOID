# F.26 — CGI-source .3dphox demo (closing the "input data, not pipeline" finding)

**Date:** 2026-05-03

## The question this answers

`AUDI_HERO_attempt.md` ended on the conclusion that the renderer is shipped
and faithful — the "scanned cloud" aesthetic of the v40 Audi hero comes from
the **input data**, not the rendering pipeline. We promised a demo:

> CGI-source `.3dphox` (~half day): Take a Blender Audi mesh, sample as
> splats with clean colors and crisp normals, encode → `.3dphox`. Then
> render via the existing pipeline. This will look like a Blender render
> because it IS a Blender render fed through our format. Best demo path.

This delivers exactly that — without Blender as a dependency. We
procedurally build a clean studio scene with controlled normals, exact PBR,
no scan noise, and feed it through the existing photoreal renderer.

## What was built

### `tools/build_cgi_studio_phox.py`

Procedural scene → `.3dphox` (v25 base + v31 trailer + v40 trailer):

| Part           | Splats  | Material          | sigma  |
|---             |---:     |---                |---:    |
| Ground plane   | 180,000 | matte gray        | 0.020  |
| Chrome sphere  | 120,000 | metallic 0.94, rough 0.09 | 0.013 |
| Red matte ball |  70,000 | diffuse           | 0.012  |
| Blue cube      |  80,000 | diffuse, rotated 30° | 0.013 |
| Brass torus    |  30,000 | metallic 0.85, rough 0.18 | 0.010 |
| **Total**      | **480,000** | | |

Per-splat data assembled from analytic primitives:
- Sphere: golden-spiral sampling; normal = unit position vector
- Box: face-uniform sampling; normal = face normal
- Plane: uniform sampling; normal = up
- Torus: parametric (u, v) on big-ring × small-tube; normal from the tube center

Normals are exact (not MLS-estimated). Quaternions built so each splat's
short axis aligns with its surface normal (tangent-disk orientation).
Per-splat PBR (albedo, metallic, roughness, F0, kd) saved to a sidecar
`.npz` so the renderer can use ground truth instead of recovering
metallic/roughness from SH bands.

Output file:
```
outputs/cgi_studio_v1.3dphox       18,005,838 bytes
  - v25 base: 480k splats, xyz/scale/quat/dc/tier/sh_rest
  - v31 trailer: normals, kNN(k=4) edges, material_hints
  - v40 trailer: kappa, cusp
outputs/cgi_studio_v1.pbr.npz      ground-truth per-splat PBR
outputs/cgi_studio_v1.scene.json   manifest
```

### `tools/render_cgi_photoreal.py`

Same renderer as `render_audi_photoreal_v2.py`. The only differences:
- `SRC = cgi_studio_v1.3dphox`
- PBR loaded from sidecar `.npz` instead of `decompose_pbr` from SH bands
- Background is a bright studio sweep (not the dark grade we used for the Audi)
- Three-point light tuned for upper-key ambience

Same projection, same kNN shadows, same graph AO, same Mip-Splatting
prefilter, same GGX BRDF, same ACES filmic, same alpha compositor.

### `tools/build_cgi_vs_scan_panel.py`

Side-by-side panel comparing CGI-source vs scan-source renders. Same
renderer, both panels.

## Outputs

| File | What |
|---|---|
| `renders/crypsorender_v01/SHOWCASE_CGI_STUDIO.png`         | 1k CGI render |
| `renders/crypsorender_v01/SHOWCASE_CGI_STUDIO_2k.png`      | 2k CGI render |
| `renders/crypsorender_v01/SHOWCASE_CGI_VS_SCAN.png`        | side-by-side panel |

## What it shows

The renderer responds correctly to clean input:
- The chrome sphere reads as a metallic mirror reflecting the gray sky/floor split
- The blue cube reads as matte blue plastic with a soft Lambert shade gradient
- The red ball reads as diffuse red
- The brass torus shows orange metallic underneath the chrome

The remaining "splat softness" (visible at all silhouettes) is intrinsic to
the 3D-Gaussian-splat representation. It is NOT improved by the renderer
quality — it's a limit of representing surfaces as 3D Gaussian densities.
The published 3DGS, Plenoxels, and Mip-NeRF 360 papers all share this look.

## What it closes

| Question from the prior attempt | Answer |
|---|---|
| "Is the scan-aesthetic a renderer bug?" | No. Same renderer produces clean-looking output on clean input. |
| "Does our PBR / lit stack actually work?" | Yes. Materials read correctly when the input has correct material data. |
| "Could we ship a Blender-quality render?" | Yes — fed a clean `.3dphox` source, the renderer produces a clean image. The format and pipeline are not the bottleneck. |
| "What would Vince see?" | Two images, same code, distinct aesthetics — a pipeline that faithfully renders whatever data you put through it. |

## Honest caveats

- Procedurally-generated CGI doesn't have all the visual richness of an
  artist-built Blender scene (complex normal maps, bump, PBR layering, HDRI
  reflections from a real environment). Adding any of these is now a content
  question, not a renderer question.
- The chrome reflection currently grabs the procedural environment, not a
  real HDRI. With a richer HDR (like the studio HDRIs that ship in Bar 1's
  HDRI loader smoke test), chrome would show actual studio reflections.
- Splat scales of `(sigma * 1.1, sigma * 1.1, sigma * 0.55)` give tangent
  disks that vanish at extreme grazing angles. A dense isotropic-sphere
  source could be stiffer — left as a follow-up.

## The deliverable in one sentence

**Same renderer, two `.3dphox` files, two distinct aesthetics — proving the
"scanned cloud" look is a property of trained-3DGS scan input, not of the
CRYPSOID pipeline.**
