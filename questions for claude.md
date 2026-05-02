The non-AI versions to absorb
AI / NeRF glue	Non-AI equivalent	CRYPSOID absorption
Hypernetwork that emits scene-specific weights	Scene compiler that emits scene-specific phoxoid parameters from geometry, normals, color, visibility, and neighborhood stats	One-pass .ply/.images → .3dphox compiler
Diffusion/SDS guidance	Energy minimization: photometric reprojection, silhouette consistency, normal/depth consistency, curvature smoothness, visibility penalties	No “imagining”; just force phoxoids to satisfy measured views
Meta-learning / few-shot adaptation	Template libraries + priors: wall/floor/edge/foliage/object surface priors, not neural weights	Scene-class phoxoid presets and solve schedules
Continual NeRF / CL-NeRF updates	SLAM-style incremental fusion: TSDF, surfels, pose graph updates, delta chunks	Append/update phoxoids without retraining the scene
LoRA for NeRFs	Low-rank residual fields: PCA/tensor/spline/RBF deltas over explicit parameters	Tiny .3dphox.patch files
Neural radiance field	Explicit radiance field: sparse voxels, surfels, splats, spherical harmonics, texture atlases	Radiance lives on/around phoxoids, not inside an MLP
Neural completion	Nearest-neighbor + symmetry + local surface continuation	Deterministic phoxoid extrapolation

This is backed by the direction of the field already: Plenoxels showed that radiance fields can be represented as sparse voxel grids with spherical harmonics and optimized without neural components, while 3D Gaussian Splatting showed how explicit anisotropic primitives can render radiance fields in real time without a NeRF MLP at render time.

The strongest CRYPSOID version

CRYPSOID should become an explicit adaptive scene compiler:

images / video / ply / splats
        ↓
camera poses + point cloud + normals + depth estimates
        ↓
local phoxoid extraction
        ↓
nearest-neighbor phoxoid graph
        ↓
radiance / material / normal / residual fields
        ↓
compressed .3dphox scene
        ↓
renderer + update patches

COLMAP/OpenMVS-like reconstruction gives you the traditional non-AI photogrammetry base: camera poses, sparse/dense point clouds, and multi-view geometry. COLMAP is specifically an end-to-end SfM/MVS reconstruction pipeline for ordered or unordered image collections.

Then CRYPSOID’s actual invention happens after that: do not keep the point cloud, mesh, voxel grid, or Gaussian splats as the final truth. Convert them into phoxoids.

Where normal maps fit

Normal maps are not cosmetic here. They are one of the cleanest non-AI bridges into phoxoidal math.

A normal map gives you local orientation change. For CRYPSOID, that means each phoxoid can carry:

position
normal
tangent frame
anisotropic support radius
curvature estimate
surface confidence
radiance coefficients
residual displacement / normal correction

So instead of saying:

“This is a textured mesh with a normal map.”

CRYPSOID says:

“This is a field of local oriented phoxoids whose tangent-frame deviations are encoded as compact residuals.”

That keeps you inside the phoxoidal ontology.

Normal maps become microgeometry residuals. Height maps become surface offset residuals. Displacement maps become explicit phoxoid deformation fields. Texture becomes radiance attached to local phoxoid frames.

Where nearest neighbor fits

Nearest neighbor is not a dumb fallback. It is the first non-AI version of “scene understanding.”

A k-nearest-neighbor graph lets each phoxoid know:

who its local neighbors are
which surface patch it belongs to
which direction continuity flows
where density increases or thins
where edges, cracks, corners, and occlusions occur
what can be extrapolated safely
what must remain uncertain

That gives you a graph:

phoxoid_i → {neighbor phoxoids}
          → local tangent agreement
          → normal deviation
          → color/radiance similarity
          → curvature continuity
          → visibility overlap
          → semantic/region label, optional

This is the non-AI analog of attention. Not transformer attention, but geometric attention.

QSplat is relevant here because it already used multiresolution point/sphere hierarchies for progressive point rendering and LOD selection; CRYPSOID can absorb that idea, but replace “sphere hierarchy” with phoxoid hierarchy.

The biggest non-AI tools to absorb
1. TSDF / SDF fusion

TSDF fusion is one of the most important pieces. KinectFusion used a truncated signed distance field to fuse depth measurements into a dense real-time surface model, separating free space, uncertain measurement zones, and unknown areas.

CRYPSOID version:

TSDF voxel → local signed-distance samples
          → phoxoid surface support
          → confidence shell
          → uncertainty boundary

You do not keep TSDF as final. You use it to infer where phoxoids should exist.

2. Surfels

Surfels are already close to phoxoids: oriented surface elements with position, normal, color, radius, confidence. ElasticFusion and later surfel systems show how dense maps can be built and updated online using surfel-based mapping.

CRYPSOID version:

surfel = primitive ancestor
phoxoid = richer surfel with anisotropy, curvature, residuals, radiance, neighborhood law

So yes: surfels should be absorbed hard.

3. Moving Least Squares surfaces

MLS is useful because it fits local surface patches from points using polynomial approximation. That is almost exactly what you need for turning noisy neighbors into stable local phoxoid geometry. Point Set Surfaces use moving least squares to approximate surfaces from point data through local polynomial fits.

CRYPSOID version:

take kNN neighborhood
fit local MLS patch
derive tangent frame + curvature
encode as phoxoid local surface law

This is phoxoidal math expansion without drifting.

4. Poisson reconstruction

Poisson reconstruction is useful as a temporary global consistency solver. Screened Poisson reconstruction creates watertight surfaces from oriented point sets and explicitly incorporates point constraints.

CRYPSOID version:

oriented points → Poisson surface sanity check
Poisson surface → normal/curvature/global continuity hints
hints → phoxoid correction field

Again: do not keep the mesh as final. Use it to discipline the phoxoids.

5. Explicit radiance fields

Plenoxels are very important philosophically because they prove: “radiance field” does not have to mean “neural network.” Plenoxels use a sparse 3D grid with spherical harmonics and no neural components, optimized much faster than classic NeRF.

CRYPSOID version:

spherical harmonics / view-dependent color
        ↓
stored per phoxoid or per phoxoid cluster

So a phoxoid can have:

base color
view-dependent SH coefficients
roughness/specular estimate
normal residual
opacity / density
visibility confidence

This gives you NeRF-like appearance without NeRF.

The AI glue you mentioned, translated into CRYPSOID

Hypernetwork NeRF papers generate scene-specific weights and encodings; HyP-NeRF, for example, uses hypernetworks to estimate NeRF weights and multi-resolution hash encodings.

CRYPSOID should steal the behavior, not the network:

hypernetwork behavior:
"one pass emits a scene-specific representation"

CRYPSOID version:
"one deterministic compiler pass emits a scene-specific phoxoid field"

Latent-NeRF/SDS uses diffusion priors to push a 3D representation toward plausible images or shapes; Latent-NeRF adapts score distillation into latent diffusion space and adds shape guidance.

CRYPSOID should translate that into:

AI SDS:
"make it look plausible"

non-AI CRYPSOID:
"make it satisfy measured multi-view evidence"

CLNeRF-style work handles scenes that change over time by adding continual learning and update mechanisms; CLNeRF uses generative replay, Instant-NGP architecture, and trainable appearance/geometry embeddings for scene changes.

CRYPSOID version:

new frames arrive
        ↓
detect changed regions
        ↓
update affected phoxoid clusters
        ↓
write .3dphox delta patch
        ↓
do not retrain entire scene

LoRA-style neural-field updates are also directly translatable. Recent work on low-rank adaptation of neural fields frames LoRA as an efficient way to encode small changes to neural fields; CRYPSOID can do the same thing explicitly as low-rank changes over phoxoid parameters.

CRYPSOID version:

base.phox
patch_001.phoxdelta
patch_002.phoxdelta

Where each patch is not neural weights, but small low-rank updates to:

position
normal
scale
curvature
radiance
opacity
neighbor weights
visibility
The actual “glue dissolving” roadmap
Phase 1 — Use everything as scaffolding

Allow COLMAP, depth maps, normal maps, segmentation, Gaussian splats, NeRFs, AI depth, SAM, whatever.

But every output must be forced into this structure:

phoxoid = {
  xyz,
  normal,
  tangent_frame,
  anisotropic_scale,
  curvature,
  color/radiance,
  opacity/density,
  residual_normal,
  residual_displacement,
  neighbor_edges,
  confidence,
  provenance
}
Phase 2 — Replace neural outputs with explicit solvers

Replace:

AI depth → MVS/depth fusion where possible
AI normal → local PCA / MLS / Poisson-derived normals
AI completion → NN graph continuation + symmetry + surface priors
AI radiance → SH coefficients / texture projection / residual atlas
AI updates → SLAM delta fusion
Phase 3 — Collapse all “helpers” into phoxoidal operators

At this point, the tools stop being separate systems.

Not:

CRYPSOID + normal maps + nearest neighbor + surfels + Poisson + Plenoxels

But:

CRYPSOID uses:
- phoxoid normal residuals
- phoxoid neighbor graph
- phoxoid surface relaxation
- phoxoid radiance coefficients
- phoxoid delta patches

That is the glue dissolving.

The key insight

Normal maps and nearest neighbor are not side features. They are probably the correct bridge from splats/point clouds into phoxoidal math.

A Gaussian splat says:

here is an oriented fuzzy ellipsoid of density/radiance

A phoxoid should say:

here is a local scene atom whose shape, surface behavior, radiance, normal deviation, neighborhood relation, uncertainty, and update law are all explicit

That is more than a splat. Less opaque than a NeRF. More compressible than a mesh plus textures. More adaptable than a static point cloud.

So yes: absorb non-AI versions of all of it — TSDF, surfels, MLS, Poisson, Plenoxels, spherical harmonics, nearest-neighbor graphs, normal maps, residual maps, LOD hierarchies, low-rank deltas, SLAM fusion