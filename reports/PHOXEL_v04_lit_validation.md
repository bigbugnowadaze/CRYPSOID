# Phase F.14 — Phoxoidal lit-stack validation

**Date**: 2026-05-03

## What we tested

Take the Hessian-aligned phoxoidal `.3dphox` from Phase F.12.2 and run it
through the full CRYPSOID lit stack (v32a Lambert per-blob lighting). Goal:
prove that the Hessian-derived per-blob normals actually drive meaningful
lighting variation, validating the novelty claim.

## Result

**Mechanism works end-to-end.** The integration is clean:

  1. Image → Phoxel optimizer → voxel grid
  2. Voxel grid → Hessian extraction → phoxoidal `.3dphox` with real per-blob normals
  3. `.3dphox` → CRYPSOID renderer → per-blob Lambert lighting using normals from quaternions

**Visual validation (Bunny voxelized)**:
  - Iso blobs with fake radial-from-centroid normals → uniform blueish tint (no real surface variation)
  - Phoxoidal blobs with Hessian normals → surface-following highlights and shadows (warm/cool variation across the bunny)

The lit-shading varies *differently* between iso-fake-normals and phoxoidal-real-normals,
proving the Hessian extraction is producing geometrically meaningful normals.

## Visual quality caveat (honest)

The Family-photo lit-render came out muddy because the underlying voxel
reconstruction is at ~20 dB — noisy density field gives noisy Hessian gives
noisy normals gives muddy lighting. The Bunny voxelization is sparse (8840
occupied cells out of 128³ = 2M) because Stanford bunny is a surface mesh,
not a volumetric scan, so most of the grid is empty.

For a genuinely photo-real lit demo we need either:
  - A higher-PSNR phoxel reconstruction (depends on better SfM, F.13.3)
  - A volumetric source dataset (Stanford bunny mesh density-fills poorly)
  - Apply Hessian extraction to a trained-3DGS PLY directly (skip the SfM noise)

## Files

  - `outputs/family_phoxel_phoxoidal.3dphox` — Hessian-aligned blobs from Family
  - `outputs/bunny_phoxel_phoxoidal.3dphox` — Hessian-aligned blobs from voxelized Bunny
  - `outputs/renders/phoxel_v01/SHOWCASE_PHOXOIDAL_LIT_3panel.png` — Family lit
  - `outputs/renders/phoxel_v01/SHOWCASE_BUNNY_PHOXOIDAL_LIT.png` — Bunny lit (4 panels: iso unlit | iso fake-lit | pho unlit | pho real-lit)

## Bottom line for the integration story

The image→.3dphox→lit-render path is mechanically complete. Hessian normals
produce real per-blob shading variation. Photo-real visual output requires
the SfM front-end accuracy that F.13.3 will unlock; lighting itself is wired
and honest.
