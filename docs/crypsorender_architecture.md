# Crypsorender — architecture (v0.x)

**Status:** DRAFT — needs Bug's sign-off before implementation begins.
**Reads:** `docs/thesis_digest.md`, `docs/oss_renderer_survey.md`. Don't read this without those.
**Doctrine:** no GPU/CUDA dependencies (`feedback_no_gpu_deps.md` rule). Numpy + scipy + skimage + cv2 + Pillow + ffmpeg only.

---

## 0. What this renderer is

A **CPU-only, tier-aware, phoxoidal-capable** Gaussian-splat-compatible rasterizer for `.3dphox` and standard `.ply` 3DGS files. Output: PNG frames, MP4 turntables, JSON metrics.

**Three rules of the project, in priority order:**
1. **Honest.** Every output declares which math path it actually used. Never claim phoxoidal results on Gaussian-only data.
2. **Novel.** Tier-aware dispatch and the phoxoidal Φ-gauge are first-class, not bolted on.
3. **Correct.** Match the canonical EWA-splatting algorithm exactly for the Gaussian (Tier C) path. The Tier C result is the verified baseline against which phoxoidal claims are measured.

Performance is the **fourth** priority. We accept slow.

---

## 1. The split: what's CPU-feasible, what's deliberate scope

**In scope for v0.1 (the side-by-side static deliverable):**
- Read `.ply` and `.3dphox` (v25/v27/v28 render container).
- World→camera→NDC→pixel projection per the EWA-splatting formula.
- 3D→2D covariance projection (the J·W·Σ_3D·W^T·J^T chain).
- Per-frame depth sort.
- 16×16 tile binning.
- Per-tile front-to-back alpha compositing with early termination.
- Per-splat SH evaluation (degree 0–3, real basis). **Better than antimatter15's vertex-baked color.**
- Tier-aware dispatch.
- **Synthetic germ auto-fitter** (`H(s,t) = κ₁s² + κ₂t²`) for Tier A/B splats when no germ data is present.
- **Phoxoidal Tier A/B rasterizer paths** using the closest-point-on-germ approximation (1–2 Newton iterations).
- **TWO renders per scene** (per Bug's decision): Gaussian-only "truth" render + synthetic-germ "preview" render. Both with `honesty_caveat` in the manifest. Contact sheet: original PLY | Gaussian-only v28 | synthetic-germ v28.
- Static image only — no turntable in v0.1.

**In scope for v0.2 (real phoxoidal data + turntable):**
- Read germ chunks from `.3dphox` v40+ when they exist, populating `SplatBuffer.germ`.
- Tier B exact-residual correction lookup from v28's archive container.
- Turntable MP4 via ffmpeg.

**Deliberately out of scope:**
- Real-time rendering. We're targeting "minutes per frame" tolerance, not 60 FPS.
- Adaptive level-of-detail.
- Anti-aliasing beyond what the Gaussian footprint already provides.
- Training/optimization (this is a renderer, not a trainer).
- Multi-camera/stereo, VR, panoramic.
- Layer 1 evidence terms from the thesis (`R, D, S` penalties). Out of v0.2 scope because we don't have an evidence stack `E` for the Audi data. Synthetic stress scenes (Tier 0 of PhoxBench) would be the first place to exercise these.
- Network/HTTP rendering server. Local CLI only.

---

## 2. Module structure

```
tools/crypsorender/
  __init__.py            # public API
  io/
    ply_loader.py        # standard 3DGS .ply (763,800 splats × 62 floats)
    phox_loader.py       # .3dphox v25/v27/v28 reader; decodes chunks → numpy
    splat_buffer.py      # the canonical in-memory splat representation
  math/
    quat.py              # quaternion → rotation matrix
    sh.py                # SH basis, degrees 0–3, real basis; evaluates view direction
    ewa.py               # 3D→2D covariance projection (the J·W·Σ chain)
    germ.py              # phoxoidal germ H_θ(s,t) and closest-point solver  [v0.2]
  pipeline/
    camera.py            # world↔camera↔NDC↔pixel transforms; turntable cam path
    project.py           # batch-project all splats; emit per-splat (center_2d, cov_2d, depth)
    cull.py              # frustum + small-screen-area cull
    sort.py              # per-frame depth sort (numpy argsort, descending z_view)
    tile.py              # 16×16 tile binning; emit per-tile splat lists
    rasterize.py         # the inner loop: per-tile sorted list → tile pixel buffer
                         #   tier-dispatched: A/B/C → phoxoid_full/phoxoid_corr/gaussian
                         #   front-to-back alpha-composite, early termination
    composite.py         # assemble tile buffers into the final framebuffer
  output/
    png.py               # framebuffer → PNG via Pillow
    metrics.py           # PSNR / SSIM / MSE / MAE; tier dispatch counts; timing
    contact_sheet.py     # multi-panel composite (input | output | heatmap | tier view)
    turntable.py         # generate per-frame cameras; orchestrate frame renders;
                         #   feed frames to ffmpeg for MP4 assembly
  cli.py                 # entrypoint:
                         #   crypsorender render --scene <ply|3dphox> --camera <yaml>
                         #                       --out <png> --tile 16 --max-points 0
                         #   crypsorender turntable --scene <...> --out <mp4>
                         #                          --frames 150 --fps 30
                         #   crypsorender compare --a <ply> --b <3dphox> --camera <yaml>
                         #                        --out <contact_sheet.png>
```

Total: ~13 source files. Target line counts:

| Module | LoC budget |
|---|---:|
| io/* | 250 |
| math/* | 200 |
| pipeline/* | 600 |
| output/* | 250 |
| cli.py | 100 |
| **Total** | **~1400 LoC** |

That's about 2× antimatter15/splat's whole codebase, which is right — we're doing more (CPU rasterization, tier dispatch, phoxoidal hooks) but staying lean.

---

## 3. The canonical splat-buffer schema

All loaders produce this in-memory shape. Renderer is agnostic to source format.

```python
@dataclass
class SplatBuffer:
    n: int                       # number of splats
    xyz: np.ndarray              # (n, 3) float32 — world position
    scales: np.ndarray           # (n, 3) float32 — log-space scales (3DGS convention)
    quats: np.ndarray            # (n, 4) float32 — unit quaternion (wxyz)
    opacities: np.ndarray        # (n,)   float32 — sigmoid logit
    sh_dc: np.ndarray            # (n, 3) float32 — degree-0 SH coefficients (RGB)
    sh_rest: np.ndarray | None   # (n, 45) float32 — degrees 1–3 SH coefficients;
                                 #                    None = degree-0-only (DC) rendering
    tier: np.ndarray             # (n,)   uint8 — 0=A, 1=B, 2=C
    germ: GermBuffer | None      # phoxoidal germ data, None if vanilla 3DGS
    correction: CorrBuffer | None # Tier B exact-residual correction, None for v0.1
```

Standard 3DGS `.ply` loader produces `tier = all 2 (Tier C)`, `germ = None`, `correction = None`.
`.3dphox` v25 loader produces real `tier`, `germ = None` (no germ chunks yet), `correction = None`.
`.3dphox` v28 archive loader produces real `tier`, `germ = None`, `correction = the residual chunks`.

Future `.3dphox` v40+ with germ chunks: `germ` populated, exercising Tier A/B paths.

---

## 4. The render loop in pseudocode

```python
def render_frame(scene: SplatBuffer, camera: Camera, opts: RenderOpts) -> Frame:
    # 1. Project & cull (vectorized over all splats).
    centers_3d_cam = camera.world_to_cam(scene.xyz)               # (n, 3)
    depths = -centers_3d_cam[:, 2]                                # camera looks down -z
    visible = depths > camera.near
    cov_3d = build_3d_cov(scene.scales[visible], scene.quats[visible])   # (m, 3, 3)
    centers_2d, cov_2d = ewa_project(camera, centers_3d_cam[visible], cov_3d)
    radii = sqrt(eigvals_2x2(cov_2d).max(axis=1)) * 3              # 3-sigma footprint
    on_screen = bbox_intersects_screen(centers_2d, radii, camera.size)
    cov_2d_inv = invert_2x2(cov_2d)

    # 2. SH eval (per-splat for now — could be per-pixel-block later).
    view_dirs = normalize(scene.xyz - camera.position)             # (m, 3)
    rgb = sh_eval(scene.sh_dc, scene.sh_rest, view_dirs, view_clip=True)  # (m, 3)

    # 3. Sort splats by depth, descending (back-to-front for tile fill).
    order = argsort(-depths[on_screen])

    # 4. Tile binning. Emit (tile_id, splat_idx) tuples for each splat × overlapped tile.
    tile_ids, splat_idx = bin_to_tiles(centers_2d, radii, camera.size, tile_size=16)

    # 5. Per-tile rasterize.
    framebuffer = zeros((camera.size_y, camera.size_x, 3), dtype=float32)
    transmittance = ones((camera.size_y, camera.size_x), dtype=float32)
    for tile in unique(tile_ids):
        splats_in_tile = splat_idx[tile_ids == tile]                # already depth-sorted
        for s in splats_in_tile:
            kind = scene.tier[s]
            if kind == TIER_C or scene.germ is None:
                rasterize_gaussian_into_tile(s, framebuffer, transmittance, ...)
            elif kind == TIER_B:
                rasterize_phoxoid_corrected_into_tile(s, ...)       # v0.2
            elif kind == TIER_A:
                rasterize_phoxoid_full_into_tile(s, ...)            # v0.2
            if (transmittance[tile_pixels] < epsilon).all():
                break                                                # early termination

    return Frame(rgb=framebuffer, alpha=1-transmittance, metrics={...})
```

**Honest tier_dispatch_counts in the metrics:** count which path each splat actually took. If `scene.germ is None`, Tier A/B splats go through `rasterize_gaussian_into_tile` and the metrics report `A_via_gaussian_fallback: <count>`. This is the truth-gate the project doctrine demands.

---

## 5. Three numerical-correctness anchors

Before claiming the renderer works, all three must pass:

1. **EWA projection round-trip.** For an axis-aligned splat at the origin, project + back-project should recover the input scales/rotation to single-precision tolerance. Tests in `test_ewa.py`.
2. **Single-splat baseline match.** Render one splat alone with our pipeline. Compare against a reference 1-splat result computed from the published 3DGS paper's formulas (independent code path inside the test). Pixel RMSE must be ≤ 1e-4 in floats.
3. **Audi DC-only sanity.** Render the Audi v28 with `sh_rest = None` (degree-0 only, like the v0.30 truth gate did). Compare against `v30_truth_gate/renders/v28_dc_opacity.png`. Should match within a small tolerance because v0.30 used the same DC values, just rendered with a worse (dot-only) rasterizer. Anything wildly different here is a bug in projection or compositing.

When all three pass, we add the Audi-with-SH render as the v0.1 deliverable artifact.

---

## 6. The PhoxBench output contract

Every `crypsorender render` call writes:

```
<out_dir>/
  frame.png                   # the actual render
  metrics.json                # quantitative
  manifest.json               # what was rendered, by what code
```

`manifest.json` includes:
```json
{
  "renderer_version": "0.1.0",
  "scene_path": "...",
  "scene_format": "ply" | "3dphox_v25" | "3dphox_v28_render" | ...,
  "camera": {...},
  "tier_dispatch_counts": {
    "A_phoxoidal_full": int,
    "A_via_gaussian_fallback": int,
    "B_phoxoidal_corrected": int,
    "B_via_gaussian_fallback": int,
    "C_native_gaussian": int
  },
  "sh_degree_used": 0 | 1 | 2 | 3,
  "code_paths_exercised": ["gaussian_inner_loop", "..."],
  "honesty_caveat": "This render used the Gaussian fallback for all tiers because no germ data was present in the source." | null
}
```

This makes it impossible to publish a misleading render — every output documents what math actually ran.

---

## 7. Implementation order (v0.1)

Each step ends with a runnable artifact. Bug-friendly review at each.

1. **`io/ply_loader.py` + `io/splat_buffer.py`** — load Audi PLY into the canonical buffer. Test: print first 5 splats' values and confirm they match raw PLY contents.
2. **`math/quat.py` + `math/sh.py`** — quaternion rotations and SH basis. Test: known rotations recover identity; SH at view direction `(0,0,1)` matches paper's tabulated values.
3. **`math/ewa.py` + `pipeline/camera.py` + `pipeline/project.py`** — projection chain. Test: projection-correctness anchor 1.
4. **`pipeline/sort.py` + `pipeline/cull.py`** — sort and cull. Test: 1-splat scene round-trips correctly; far/near clipping behaves.
5. **`pipeline/tile.py`** — tile binning. Test: a splat at a known position lands in the predicted tiles.
6. **`pipeline/rasterize.py`** — the inner loop. Test: anchor 2 (single splat).
7. **`pipeline/composite.py` + `output/png.py`** — full image. Test: anchor 3 (Audi DC-only matches v0.30).
8. **`output/metrics.py` + `output/contact_sheet.py`** — quantitative + visual deliverable.
9. **`io/phox_loader.py`** — same as PLY but reads `.3dphox`. Re-render and confirm bit-identical to PLY-rendered version (because v25 chunks 0–4 are byte-identical to PLY-derived equivalents per our v25 verification).
10. **`cli.py` + `tools/crypsorender/__init__.py`** — wire it together. The v0.1 deliverable is one CLI invocation that produces the side-by-side Bug asked for.

Each step is independently testable. If we get to step 7 and the Audi-DC-only render doesn't match the v0.30 baseline, we stop and find the bug — we don't proceed to fancier features on a broken foundation.

---

## 8. v0.2 add-on order (after v0.1 is stable)

1. **`math/germ.py`** — `H(s,t)` polynomial evaluation + closest-point solver.
2. **Extend `io/phox_loader.py`** — recognize germ chunks if present, populate `SplatBuffer.germ`. (This requires germ chunks to actually exist in some `.3dphox`. Since they don't yet for Audi, also add a `--synthetic-germ` flag that fits a curvature germ to each splat's local neighborhood for prototyping.)
3. **Add tier-A/B paths to `pipeline/rasterize.py`** — phoxoidal density, the closest-point approximation for the action `A_θ(u) ≈ F_θ(u, s*, t*)`.
4. **`output/turntable.py`** — circular camera path, incremental frame render (sandbox-friendly chunked execution like the v29 driver), ffmpeg assembly.
5. **`pipeline/rasterize.py` Tier B** — read the v28 q8-exact correction chunks and apply them as a residual to the SH evaluation.

---

## 9. Architecture decisions (signed off by Bug, 2026-04-30)

1. **Germ-data default:** when `.3dphox` doesn't carry germ chunks (which is the case for our current Audi v25/v28), **render BOTH**:
   - One Gaussian-only render (Tier A/B/C all → Gaussian path) — the truth render, with `honesty_caveat` set.
   - One synthetic-germ render — fit a quadratic curvature germ `H(s,t) = κ₁s² + κ₂t²` per Tier A/B splat from its local neighborhood (use the splat's existing scales as starting eigenvalues), then run the phoxoidal Tier A/B paths. `honesty_caveat` says "synthetic germ auto-fitted by the renderer; not from data."
   - Both renders go in the same run directory. Contact sheet shows them side-by-side along with the original PLY render.
   - **Implication for v0.1:** the synthetic germ fitter and the phoxoidal Tier A/B rasterizer paths must be in v0.1, not deferred to v0.2. v0.2 reduces to "do the same thing with germ chunks read from the file instead of auto-fitted."
2. **Turntable:** **out of scope for v0.1.** Static images only. Get the side-by-side fidelity right first; turntable becomes v0.2 priority once we trust the renderer.
3. **Output directory layout:** PhoxBench convention. `renders/<timestamp>_<scene>_<config>/{frame.png, metrics.json, manifest.json, ...}`. Consistent with the thesis spec.
4. **SH evaluation:** per-splat (view direction from camera to splat center). Standard 3DGS simplification. Per-pixel is left as a possible v0.3 upgrade if needed.
5. **Pre-multiplied alpha.** Standard 3DGS convention.
6. **Sigmoid decoding of opacity and DC.** Standard 3DGS convention. `R8 = clip(round((SH_C0 * f_dc + 0.5) * 255), 0, 255)` for DC, `op = sigmoid(opacity_logit)` for opacity. Already verified in the v25 build to match the v27 anchor byte-for-byte.
7. **Resolution defaults:** 1024×1024, `--max-points 0` (render all 763,800 splats). For previews users can override `--max-points` to a smaller value.
