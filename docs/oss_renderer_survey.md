# OSS splat-renderer survey — what to absorb, what to leave behind

Goal: pick the parts of existing 3DGS renderers that map cleanly onto a CPU-friendly, tier-aware, phoxoidal-capable CRYPSOID renderer. Not a feature comparison. Reverse-engineering of the underlying *algorithmic shape* — what each renderer does in its inner loop and which decisions are GPU-bound vs portable.

Read paired with `docs/thesis_digest.md`.

---

## 1. Reference designs surveyed

| Project | Stack | Sort | Tile binning | Per-pixel inner loop | Notes |
|---|---|---|---|---|---|
| **Inria reference** (`graphdeco-inria/gaussian-splatting`) | CUDA + custom kernels | GPU radix on `(tile_id, depth)` | **Yes**, 16×16 | Front-to-back alpha blend with early termination | The canonical algorithm; everything below is a reimplementation. |
| **antimatter15/splat** | WebGL 1.0, ~350 lines | CPU radix sort in a Web Worker (async) | **No** — relies on GPU fragment pipeline | Vertex shader emits a quad per Gaussian; fragment shader computes Mahalanobis density and alpha-blends through standard depth-equal blending | Tiny and elegant. No tile binning; each splat draws to every fragment in its quad. |
| **mkkellogg/GaussianSplats3D** | Three.js | CPU radix sort in Web Worker | No (same WebGL-fragment-pipeline approach) | Same as antimatter15 + SH eval up to degree 3 | More polished; chunked loading for large scenes. |
| **SuperSplat (PlayCanvas)** | PlayCanvas WebGL/WebGPU | Engine-managed | Yes on WebGPU path | PlayCanvas shader pipeline | Mostly an editor; renderer is PlayCanvas's `gsplat` material. |
| **gsplat (nerfstudio)** | CUDA + PyTorch | GPU | Yes, 16×16 | Same as Inria reference | The de facto Python API; pure GPU bound. **Disallowed for our project.** |

## 2. The Inria algorithm in one diagram

```
For each Gaussian g_i in scene:
  1. Project center c_i to NDC, get screen-space (x, y, z_view).
  2. Build 3D covariance Σ_3D = R · diag(scales²) · R^T.
  3. Project: Σ_2D = J · W · Σ_3D · W^T · J^T   (J = projection Jacobian, W = world-to-camera).
  4. Bound: take eigenvalues of Σ_2D, take sqrt → semi-axes; tile-bbox = center ± 3·max_axis.
  5. Emit one (tile_id, sort_key=z_view, gaussian_idx) tuple per overlapped tile.

Sort all tuples by (tile_id, sort_key).         // GPU radix sort

For each tile T (16×16 pixels), in parallel:
  6. Walk the sorted list for tile T (front to back).
  7. For each Gaussian g_i in the list:
       For each pixel p in T:
         d = p - center_2D(g_i)
         power = -0.5 · d^T · Σ_2D⁻¹ · d
         alpha = opacity_i · exp(power)
         If alpha < ε: skip
         color = SH_eval(c_i_sh, view_dir(p))
         dst_color += transmittance · alpha · color
         transmittance *= (1 - alpha)
         If transmittance < ε: break out of g_i loop      // early termination
```

The sentence to commit to memory: **everything is per-tile, sorted front-to-back, with early termination once a pixel saturates.** Without tile binning the cost is `O(N_splats × bbox_pixels)` per frame; with it, the cost is `O(N_splats_per_tile × tile_pixels)` and most tiles only see a few hundred splats.

## 3. What we will absorb

### From the Inria reference
1. **Tile binning at 16×16.** The right answer for CPU too. Without it, our 763,800-splat scene runs ~30× slower than necessary because most splats touch the same pixels.
2. **Per-tile depth sort.** A single global sort once per frame is fine; we don't need the GPU's parallel radix. NumPy's `argsort` is enough.
3. **Front-to-back alpha compositing with early termination.** Saturated tiles stop iterating splats. This is the single biggest perf win for dense scenes.
4. **Project 3D → 2D covariance once per splat, cache it.** `Σ_2D = J W Σ_3D W^T J^T` is a small matrix multiply chain; do it once, store the 2D inverse.

### From antimatter15/splat
1. **Tiny, dependency-free design.** Their renderer is ~350 lines of JS + GLSL. The CPU equivalent should also stay under ~600 lines. Numpy + math + that's it.
2. **CPU sort is fine.** The whole world has been doing CPU sort in a worker thread; we'll do it in the main thread on the CPU because the rest of the render is CPU too. No threading lift needed for static frames.
3. **Per-splat quad in a "draw" sense, but virtual on CPU.** We don't emit a quad to a GPU; we compute the *bounding box of pixels* the splat affects and rasterize directly into a tile-sized scratch buffer.

### From mkkellogg/GaussianSplats3D
1. **SH evaluation up to degree 3.** Standard 3DGS uses 16 coefficients per channel (1 DC + 15 rest). Our v25 carries exactly this. Use the standard real-valued spherical-harmonic basis evaluation; precompute the basis vector for a viewing direction once per pixel-block (tiles share view direction approximately for non-fisheye cameras).
2. **Chunked loading.** Not relevant to a static-frame renderer but relevant if the turntable streams.

### Note on antimatter15's simplification (verified from their main.js)
antimatter15/splat **does NOT do per-fragment SH evaluation** — it bakes the DC color per-vertex and the fragment shader just multiplies by the Gaussian falloff. Their full fragment shader is 5 lines:

```glsl
void main () {
    float A = -dot(vPosition, vPosition);  // -Mahalanobis² in eigenframe
    if (A < -4.0) discard;                 // outside 2-sigma cull
    float B = exp(A) * vColor.a;           // alpha = α · exp(-|p|²)
    fragColor = vec4(B * vColor.rgb, B);   // pre-multiplied alpha
}
```

Their vertex shader does the standard EWA projection:

```glsl
mat3 J = mat3(  // perspective Jacobian
    focal.x/cam.z, 0,             -focal.x*cam.x/cam.z²,
    0,             -focal.y/cam.z, focal.y*cam.y/cam.z²,
    0, 0, 0);
mat3 T = transpose(view) * J;
mat3 cov2d = transpose(T) * Vrk * T;       // 2D covariance
float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
float radius = length(vec2((cov2d[0][0]-cov2d[1][1])/2.0, cov2d[0][1]));
float lambda1 = mid + radius, lambda2 = mid - radius;  // eigenvalues
// quad emitted: vCenter ± position·majorAxis ± position·minorAxis
```

This is the correct EWA-splatting math for our renderer to absorb, and we can do **better than antimatter15** by evaluating SH per-pixel-block (or per-tile) instead of pre-baking flat color.

## 4. What we will deliberately NOT do

1. **No GPU shaders.** The whole project exists to avoid this. Per the no-GPU-deps rule.
2. **No vertex/fragment-pipeline emulation.** Antimatter15's "emit a quad and let the fragment shader do the math" doesn't translate to CPU — without a hardware rasterizer we'd be paying per-pixel cost without any tile-binning benefit. We rasterize directly into NumPy arrays.
3. **No JIT compiler unless we have to.** Numba and Cython are CPU-only and theoretically compatible with the no-GPU rule, but they add a build dependency. Stay in pure NumPy for v0.1; if perf is unacceptable we revisit.
4. **No real-time targets.** All the prior art is built around 60+ FPS. We don't need that. Static-frame rendering and a turntable that takes minutes to encode are fine. **Optimizing for "novel and correct" beats optimizing for FPS** — that's a CRYPSOID-specific decision that drops out of the project's reason to exist.

## 5. What we will do *differently* (and that's the novel part)

### 5a. Tier-aware splat dispatch in the inner loop

None of the surveyed renderers have a per-splat `kind` field. They all assume every primitive is the same Gaussian. CRYPSOID's `tier_labels_u8` partitions the 763,800 Audi splats into:
- 94,006 Tier A (12.3%) — meant to render through the full phoxoidal path (germ + Layer 1 evidence).
- 144,271 Tier B (18.9%) — phoxoidal + exact-residual correction.
- 525,523 Tier C (68.8%) — Gaussian fallback.

In v0.1 all three paths call the Gaussian rasterizer because we don't have germ data yet. But we wire the dispatch from day one. Pseudocode:

```python
def rasterize_splat(splat_data, tile_pixels, view_dir):
    if splat_data.tier == TIER_C:
        return rasterize_gaussian(splat_data, tile_pixels, view_dir)
    if splat_data.tier == TIER_B:
        return rasterize_phoxoid_with_correction(splat_data, tile_pixels, view_dir)
    if splat_data.tier == TIER_A:
        return rasterize_phoxoid_full(splat_data, tile_pixels, view_dir)
```

Every Gaussian path is the limit of the phoxoidal path with `H=0` and evidence terms off. So `rasterize_phoxoid_full` reduces to `rasterize_gaussian` when its germ and evidence are zero — which they are in our current data.

### 5b. Phoxoidal density evaluation, when germ data is present

Where the Gaussian inner loop computes `power = -0.5 · d^T Σ_2D⁻¹ d`, the phoxoidal inner loop evaluates `power = -A_θ(u)` where `u = (a, b, n)` is the local-frame coordinate and `A_θ` is the caustic-chart action from `thesis_digest.md` §2 Layer 2.

Approximation strategy: instead of integrating the softmin over `(s, t)` per pixel, find the closest point `(s*, t*)` on the germ surface to `u`. For the simplest curved germ `H(s,t) = κ₁s² + κ₂t²` this is a 1–2 Newton iteration starting from the projection of `u` onto the tangent plane. Then `A_θ ≈ F_θ(u, s*, t*)`. This makes the cost per splat-pixel maybe 3–5× a Gaussian — order of magnitude, still real-time-ish on CPU for a static frame.

### 5c. Render-mode flag in the metrics output

Each render outputs JSON metadata tagged with which tier path was actually exercised. So a static-frame run on Audi will report:
```
{
  "tier_dispatch_counts": {"A_via_gaussian_fallback": 94006,
                           "B_via_gaussian_fallback": 144271,
                           "C_native_gaussian": 525523,
                           "A_phoxoidal_full": 0,
                           "B_phoxoidal_corrected": 0},
  ...
}
```

This is the truth gate the thesis demands — **the renderer always declares which math path it actually used**, so we can never accidentally claim phoxoidal results on Gaussian-only data.

### 5d. PhoxBench-style output structure from day one

Every render run writes the directory tree the thesis prescribes:
```
runs/<date>_<scene>_<config>/
  input_preview.png
  gaussian_baseline.png      (the Tier C rasterizer applied to every splat)
  phoxoidal_output.png       (whatever path the data actually allows)
  side_by_side.png
  error_heatmap.png
  primitive_overlay.png
  metrics.json
  report.md
```

Even on v0.1 where `gaussian_baseline.png == phoxoidal_output.png` because we have no germ data — the structure is fixed. When germ data lands, the side-by-side becomes an honest A/B test instead of a comparative renderer rewrite.

## 6. Code-shape guess for the v0.1 implementation (preview, not commitment)

About six modules, ~600 lines total in pure NumPy:

```
tools/crypsorender/
  __init__.py
  camera.py          # world-to-camera, projection Jacobian, view direction per pixel
  project.py         # 3D Gaussian → 2D covariance + screen-space center; uses camera.py
  tile.py            # tile binning: which tiles each splat overlaps; precomputed per frame
  rasterize.py       # the inner loop: per-tile sorted splat list → pixel buffer
                     #   front-to-back, alpha-blend, early termination
                     #   dispatches by splat_kind (A/B/C)
  sh.py              # SH basis evaluation, degrees 0-3, real basis
  decode.py          # read .3dphox / .ply, decode chunks, expose splat batch in numpy
  cli.py             # entrypoint: --scene <ply|.3dphox> --camera <...> --out <png>
```

Plus a thin wrapper `render_audi.py` that calls into the above with the Audi-specific paths.

The architecture-design doc (next task) makes this concrete. This is just a sketch of the absorption pattern.

---

## Sources

- [Inria 3D Gaussian Splatting paper / repo](https://github.com/graphdeco-inria/gaussian-splatting)
- [antimatter15/splat — WebGL 3D Gaussian Splat Viewer](https://github.com/antimatter15/splat)
- [mkkellogg/GaussianSplats3D — Three.js library](https://github.com/mkkellogg/GaussianSplats3D)
- [PlayCanvas SuperSplat](https://github.com/playcanvas/supersplat)
- [gsplat documentation — rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html)
- [Differentiable Tile Rasterization explanation](https://oboe.com/learn/advanced-3d-gaussian-splatting-and-real-time-rendering-vutk9k/differentiable-tile-rasterization-1ere60z)
- [Understanding 3D Gaussian Splatting (Loges Siva)](https://logessiva.medium.com/understanding-and-exploring-3d-gaussian-splatting-a-comprehensive-overview-b4004f28ef1c)
- [LearnOpenCV — 3D Gaussian Splatting](https://learnopencv.com/3d-gaussian-splatting/)
