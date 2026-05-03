# Audi hero render attempts — what we learned

**Date**: 2026-05-03

## TL;DR

Built `tools/render_audi_hero_v3.py` to crop floor/base-plate splats and render the Audi alone. Several iterations:

| Attempt | Crop | Result |
|---|---|---|
| v3a — Y > -0.04 | drop below base plate | floor still dominates (it's at Y in [0, 0.13]) |
| v3b — drop band Y in [0.005, 0.14] | drop just the slab | underside snowdrift now visible |
| v3c — Y > 0.14 only | keep cabin only | car cabin shows as thin cloud, no body |
| v3d — band reject + side cam | drop slab, low pitch | upper "snow mound" + lower body — split silhouette |

## Why no attempt produces "Blender showroom" quality

The v40 `.3dphox` is a **3D-Gaussian-splat training output** of a real Audi scan, not a clean CGI mesh. Specifically:

1. **Splats are fuzzy by construction.** They're 3D Gaussian densities, not sharp polygon surfaces. Even at full lit-stack, you get the "scanned cloud" look that's the published-3DGS aesthetic. Same look in Plenoxels, Mip-NeRF 360, original 3DGS — all of them look like this.

2. **The base plate dominates the splat budget.** ~360k of 763k splats are in the thin Y in [0, 0.13] slab. Another ~290k are in undercarriage scan noise below. Only ~110k are the actual cabin geometry. We're rendering 14% car, 86% scan-floor.

3. **SH-DC has lighting baked in.** The training process bakes the capture-time lighting into the colors. Adding more lighting on top compounds — that's why v1 photoreal was washed out and v2 has to use very low key intensities.

## What it would take to get a real "Blender-style" Audi render

In priority order:

1. **CGI-source .3dphox** (~half day): Take a Blender Audi mesh, sample as splats with clean colors and crisp normals, encode → `.3dphox`. Then render via the existing pipeline. This will look like a Blender render because it IS a Blender render fed through our format. Best demo path.

2. **Inverse rendering / albedo recovery** (~research-paper effort): Separate albedo from baked-in lighting in the trained SH-DC. Several published methods (NeRD, NeRFactor, IRON) — none simple.

3. **Manual splat cleanup** (~1 day): Mask out the floor + scan-noise splats in 3D editor, save a cleaned `.3dphox`. Tedious but tractable.

## What we DID demonstrate clearly

- The CRYPSOID renderer **faithfully renders any** `.3dphox`. It doesn't add or hide content.
- The full lit stack (Lambert + curvature + AO + GGX + ACES) all works on real data.
- The 763k-splat full Audi renders in ~20s on CPU with parallel + Numba JIT.
- The image-in pipeline (Phase F) produces `.3dphox` files that load identically to PLY-derived `.3dphox` in this same renderer.

## For Vince

**The renderer is shipped and faithful.** The question of "showroom photo quality" is a question about the **input data**, not the rendering pipeline. With a clean CGI `.3dphox` source we'd get clean CGI renders. With trained-3DGS scan input we get scan-aesthetic output — but it's the same renderer either way.

## Files

- `tools/render_audi_hero_v3.py` — cropped-source hero render script (4 variants tried)
- `renders/crypsorender_v01/SHOWCASE_AUDI_PHOTOREAL_v2.png` — best dark-studio render of full v40 data
- `renders/crypsorender_v01/SHOWCASE_AUDI_HERO.png` — band-rejected (current state)
- `renders/crypsorender_v01/SHOWCASE_AUDI_3WAY.png` — already-shipped 3-panel showing original PLY vs .3dphox vs lit MAX
