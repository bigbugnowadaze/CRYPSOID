# CRYPSOID — roadmap, blueprint, phases

**Last updated:** 2026-05-02

This document is the canonical "where is the project, where is it going" doc.
Read [`reports/PROJECT_STATE.md`](../reports/PROJECT_STATE.md) for the running
status; read this for the bigger picture and the milestones ahead.

## North star

A **CPU-only 1:1 alternative to standard 3D Gaussian Splatting** with:
1. A novel format (`.3dphox`) that separates render-mode from EXACT-archive mode at the chunk level.
2. A novel rendering primitive (phoxoidal blob with 5-coefficient Pearcey germ) that demonstrably replaces N Gaussians at equal quality.
3. No GPU dependencies anywhere in the core codebase. End-user viewers may use GPU, but the encoder/decoder/benchmark do not.

The killer metric: **how many Gaussians does ONE phoxoidal blob replace at equal RMSE?** Currently measured at **2.0× across both synthetic stress scenes AND real meshes** (Tier 0 + Tier 1).

---

## Where we are now (2026-05-02)

| Area | Status | Verified |
|---|---|---|
| `.3dphox` format | v25 / v27 / v28-render / v28-EXACT-archive all shipped | ✓ bit-exact at q8 grid |
| CPU renderer (`crypsorender`) | ~1,600 LoC, EWA + tile binning + alpha compositing + SH eval + tier dispatch + faithful Newton | ✓ 4 anchor tests pass |
| PhoxBench Tier 0 | 6 synthetic stress scenes (plane, sphere, saddle, fold, cusp, thin sheet) | ✓ 2× killer ratio on every curved scene |
| PhoxBench Tier 1 | Real meshes — Happy Buddha + Audi PLY done, Dragon + Armadillo pending | ✓ 2× killer ratio holds |
| Compression baselines | gzip + zstd-12 measured; xz/zst-19+/npz pending | ✓ honest 1.40×-2.39× vs zstd-12 PLY |
| Browser viewer | Single-file HTML+JS+WebGL2, drag-and-drop `.3dphox` | shipped, GLSL phoxoidal density not yet wired |
| GitHub repo | README, .gitignore, CONTRIBUTING, FORMAT spec, CI workflow all written | awaiting Bug's `git push` |
| Audi turntable | 36-frame MP4, oriented correctly | shipped |

**Net read:** the core architecture and central thesis claim are validated. The remaining work is breadth (more datasets) + depth (richer phoxoid math + native format support + WebGL phoxoid path) + polish (paper, packaging, real-time perf).

---

## Phase ladder

The project sits on a tier-numbered ladder. Each tier is a **promotable hypothesis** — done means "the next tier's prerequisites exist," not "no more work in this area."

```
Tier 0 — synthetic stress scenes        DONE (2.0× killer)
   │
Tier 1 — canonical real meshes          IN PROGRESS (Happy + Audi: 2.0×; Dragon + Armadillo pending)
   │
Tier 2 — trained 3DGS scenes (rendering quality, not just geometric fit)
   │
Tier 3 — production-quality real-time renderer
   │
Tier 4 — research extensions: evidence terms, neighbor compatibility, learned coders
```

---

## Phase A — Synthetic validation (DONE)

**Goal:** measure phoxoid-vs-Gaussian on scenes designed to make the difference visible.

What landed:
- 5-coefficient Pearcey-class germ basis (κ₁, κ₂, χ, ω, ζ).
- Vectorized Newton solver with support gate (`λ(s²+t²)²`).
- 6 synthetic stress scenes with analytic ground truth.
- Killer-ratio binary search (16, 32, 64, 128, ...).
- Numerical-correctness anchor tests (4 of 4 PASS).

Result: **2.0× killer ratio on every curved scene**. Validated.

**Deliverable:** [`SHOWCASE_T2.png`](../renders/crypsorender_v01/SHOWCASE_T2.png), [`reports/TIER_2_results.md`](../reports/TIER_2_results.md).

---

## Phase B — Real-mesh validation (IN PROGRESS)

**Goal:** confirm the phoxoid advantage holds when the test isn't designed for it.

What's shipped:
- `phoxbench/scenes_mesh.py` — Stanford ASCII PLY + binary PLY loader.
- `phoxbench/run_mesh.py` — Tier 1 harness that reuses Tier 0's fit + render + killer-ratio code.
- Happy Buddha (`happyStandRight_0`, 78k pts subsampled to 10k) → 2.0× killer.
- Audi A5 PLY (xyz cloud, 763k pts subsampled to 10k) → 2.0× killer.

What's pending (Bug's downloads are mid-flight):
- Stanford Dragon (~150 MB tar.gz, expanding to ~5 MB per scan)
- Stanford Armadillo (~30 MB tar.gz)
- Multiple scan angles per mesh — the Stanford set ships ~10–20 scans per object; we'd benchmark each.

**Time to finish Phase B once downloads complete: ~10 minutes.**

**Honest expected outcome:** likely 2.0× across Dragon and Armadillo too. The pattern holds: phoxoidal blobs structurally replace Gaussians at curved regions; the killer ratio doubling-search clamps the result to a power-of-2.

If we get a result substantially different from 2× on Dragon (which has very fine surface detail), it would suggest the 5-coef basis runs out of expressiveness at very high frequency — interesting, would justify a richer germ (Phase C.1 work).

---

## Phase C — Trained 3DGS scenes / image-quality benchmark (PLANNED)

**Goal:** move from "phoxoids fit point clouds 2× better" to "phoxoids RENDER 3DGS scenes at higher PSNR/SSIM at the same primitive count."

This is the bridge from PhoxBench's geometric-fit metric to actual visual quality.

### C.1 — Convert a trained 3DGS scene to phoxoidal blobs

Need a trained 3DGS scene other than Audi. Standard test scenes used by 3DGS research:
- **Mip-NeRF 360** (bicycle, garden, kitchen, room, treehill, stump, flowers, counter, bonsai) — ~150-300 MB each
- **Tanks and Temples** (Truck, Train, Caterpillar, Family, Ignatius, ...) — varies
- **Deep Blending** (drjohnson, playroom)

These are typically distributed as `.ply` after a 3DGS training run completes. We'd:
1. Take a 3DGS PLY in.
2. Cluster splats spatially.
3. Per cluster, fit a phoxoidal blob with 5-coef germ (vs. keeping all splats as Gaussians).
4. Render BOTH at the same camera; compute PSNR/SSIM at multiple views.
5. Find the smallest phoxoid budget that matches Gaussian image quality at each view.

**Open question:** does "fitting fewer phoxoidal blobs" actually preserve visual quality, or does the splat-cluster approximation lose too much per-splat detail (color, opacity, view-dependent SH)?

### C.2 — Multi-view PSNR/SSIM with proper foreground masking

Tier 1.5 left two items blocked on sandbox state:
- **Item 4 (object-mask metrics):** apply alpha mask to PSNR/SSIM so the constant-black background doesn't inflate the numbers.
- **Item 5 (multi-view distribution):** render 32 cameras around the scene, report mean / median / worst PSNR / SSIM (and LPIPS if we add a CPU implementation).

Both are written but not run. They'd run today if executed.

**Phase C estimated effort: 1–2 weeks.**

---

## Phase D — Production-quality renderer (PLANNED)

**Goal:** make the renderer fast enough to be used by other people on real scenes.

### D.1 — CPU optimization

Current state: ~1 second per 100k splats per 384×384 frame in pure NumPy. That's ~0.5 fps on the Audi at full density.

Path forward:
- **Tile-batched rasterization** — vectorize per-tile splat lists so all pixels in a tile are processed together. Should give 5–10× speedup.
- **Numba JIT** for the inner Mahalanobis loop (CPU-only, no GPU dep). Another 2–5×.
- **Float16 framebuffer + reduced precision math** where sensible.

Target: 10 fps at 512² for 200k splats. Probably attainable.

### D.2 — Browser viewer maturity

Current viewer (`viewer/index.html`):
- Loads any `.3dphox`. ✓
- Three render modes (full, DC-only, tier overlay). ✓
- Mouse orbit + wheel zoom. ✓
- Additive blending (correct per-pixel sum, not order-correct compositing).

Not yet:
- **GLSL fragment-shader phoxoidal density** — port the closest-point Newton solver to a fragment shader. Would let phoxoids render natively in-browser.
- **Web Worker depth sort** (`viewer/sort_worker.js` is written but not wired). Proper "over" compositing.
- **Loading progress UI** for big files.
- **URL-based scene loading** (deep-link to a `.3dphox` on the web).

### D.3 — Native germ chunks in `.3dphox` (v0.4 format)

Currently germs are computed at load time from splat positions. Adds ~3 seconds of fitting per render. v0.4 spec already in [`docs/FORMAT.md`](FORMAT.md):
- `germ_5coef_f16` chunk: (n_tier_AB, 5) float16 germs.
- `germ_index_u32` chunk: (n_tier_AB,) uint32 splat indices.
- New magic `CRYPSOID40\0`, format string `CRYPSOID_3DPHOX_V40_FAITHFUL_PHOXOID`.

**Phase D estimated effort: 2-4 weeks.**

---

## Phase D.4 — Format extensions: graph + lighting + materials + temporal (NEW, drafted 2026-05-01)

A second axis of work has opened up alongside the renderer perf work. Three signed-off-or-pending one-pager specs now sit in `docs/` covering structural extensions to `.3dphox`. The strategic case is in `questions for claude.md` (absorb non-AI ancestors of NeRF/hypernet/SDS/CL-NeRF/LoRA-NeRF: TSDF, surfels, MLS, Poisson, Plenoxels, kNN graphs, normal maps).

### Spec sequence

| Spec | Doc | Adds | Cost on Audi | Phoxoidal-math? | Status |
|---|---|---|---:|---|---|
| **v31** | `docs/v31_graph_extension_spec.md` | normals + kNN edges + `.phoxdelta` patches | +47% | format-neutral | drafted, awaiting sign-off |
| **v32a** | `docs/v32_v33_lighting_materials_spec.md` | Lambert + ambient + directional sun | 0 (renderer) | standard | drafted |
| **v32b** | (same) | curvature self-shadow + curvature AO using germ | 0 (renderer) | **yes — uses germ** | drafted |
| **v32c** | deferred | cusp-specular from cubic germ terms | 0 (renderer) | yes | future |
| **v32.5** | `docs/v32_5_shadows_spec.md` | kNN-graph soft shadows + graph-AO | 0 (renderer) | **yes — uses kNN edges** | drafted |
| **v33** | `docs/v32_v33_lighting_materials_spec.md` | material_hint enum + confidence + albedo separation | +4.7% | scaffolding | drafted |
| **v34** | future | temporal `.phoxdelta` (time_range / births / deaths) | sparse | format work | future |
| **v40+** | future | transparency / refraction / Pearcey caustics | TBD | **strongly phoxoidal** | aspirational |

### Dependency graph

```
v31 (normals + edges + delta)
 ├─→ v32a (lights)            ──→ v32b (curvature shading)   ──→ v32c (cusp-specular)
 │                              \
 │                               └→ v32.5 (kNN shadows + graph-AO)
 ├─→ v33 (material_hint)        ──→ v34 (temporal delta) ──→ v40+ (caustics, glass)
 └─→ Phase D.2 (WebGL viewer)
```

The structural novelty isn't any single spec — it's the *combination* (explicit graph + germ + sparse deltas) that nothing else in the splat ecosystem ships natively. v32a, v33, v34 alone are standard-ish; v32b, v32.5, and v40+ are where the phoxoidal math actually contributes.

### Why pair v32 with v33

Lighting without materials is half a story. SH currently bakes lighting + material + view-angle into 45 floats per blob; relighting requires separating them. Pairing avoids a wasted cycle where v32 lighting looks wrong because v33 hasn't separated SH yet.

### Phasing once v31 lands

1. v32a (1–2 days) — visible "lit 3D" win.
2. v32b (1–2 days) — phoxoidal-math-specific darkening; A/B vs v32a proves the contribution.
3. v32.5 shadows (3–5 days) — proper between-phoxoid shadows + graph AO.
4. v33 material hints (3–5 days) — heuristic derivation + relightable rendering toggle.
5. v32c cusp-specular (optional follow-on, ~1 week) — only if v32b validates and there's appetite.

**Phase D.4 estimated effort: ~3 weeks total once v31 ships.**

---

## Phase E — Research extensions (LATER)

Things from the thesis that aren't load-bearing yet:

### E.1 — Layer 1 evidence terms

The thesis defines four cost terms inside Φ (the phoxoidal gauge):
- `R(x, v)` — render residual (does this point predict observed appearance?)
- `D(x, E)` — discontinuity barrier (edges, depth jumps, normals)
- `S(x, 𝒩)` — neighbor consistency
- (plus the body term `uᵀG(u,E)u` already in our screen-space evaluator)

Currently we ship only the body term. The other three need an *evidence stack* `E` (RGB residual maps, edge maps, etc.) which the synthetic Audi doesn't have. To do this properly we'd need either:
- Real reconstruction datasets with multi-view image evidence.
- Or synthetic scenes with known per-pixel evidence (extension of PhoxBench Tier 0).

### E.2 — Sheaf-theoretic neighbor compatibility maps

The thesis's `Tᵢⱼ` transition maps. Per-pair-of-overlapping-blobs compatibility data. Honestly, this is the most speculative thesis claim — it's well-founded mathematically but unclear how much it helps render quality in practice. Worth prototyping after Phase C tells us how much room is left.

### E.3 — Learned arithmetic coder over residuals

The v28 EXACT archive's correction residuals are 41.5% of the archive size. A learned arithmetic coder (CABAC-style or simpler context-adaptive) would likely shave 20–40% off that. This is the path that closes the gap with HAC/SOG (CRYPSOID currently sits at 197–337 bpg; HAC/SOG at 30–80 bpg).

**Phase E estimated effort: months. Research-grade work.**

---

## Phase F — Polish / paper / packaging (CONCURRENT)

Things that should happen alongside the technical work, not after:

1. **GitHub push** — done in design, needs your `git push` to land. [`PUSH_TO_GITHUB.md`](../PUSH_TO_GITHUB.md).
2. **A LICENSE choice** — MIT recommended for max adoption.
3. **PyPI package** — `pip install crypsorender` shouldn't be hard once the layout settles. CPU-only deps so it's clean.
4. **A short paper** — the FORMAT spec + the 2× killer-ratio result + the architectural framing IS a paper. ~6-8 pages.
5. **A demo video** — the turntable MP4 we just rendered, plus the WebGL viewer in action, plus a quick walkthrough of the killer-ratio table. Useful for any external pitch.
6. **CI matrix** — extend the existing GitHub Actions to cover Python 3.10/3.11/3.12 and macOS/Windows/Linux. ~30 min.

---

## Time estimates (what's in front of us)

| Phase | Item | Estimate |
|---|---|---:|
| **B** | Finish Tier 1 (Dragon + Armadillo + multi-scan averaging) | **<1 day** once downloads finish |
| 1.5 | Cleanup items 4 + 5 (object-mask + 32-view distribution) | half a day |
| C.1 | Convert one trained 3DGS scene to phoxoid + render comparison | ~1 week |
| C.2 | Multi-view PSNR/SSIM/LPIPS pipeline | done as part of 1.5 |
| D.1 | CPU optimization (tile-batched rasterization) | ~1 week |
| D.2 | WebGL fragment-shader phoxoidal density | ~1 week |
| D.3 | v0.4 native germ chunks in `.3dphox` | ~2 days |
| **D.4** | v31 graph extension (normals + edges + .phoxdelta) | ~1 week |
| **D.4** | v32a + v32b lighting (standard + curvature) | ~3 days |
| **D.4** | v32.5 kNN shadows + graph AO | ~5 days |
| **D.4** | v33 material hints + relightable rendering | ~5 days |
| F.1-3 | GitHub push + LICENSE + PyPI | ~1 day |
| F.4 | Short paper draft | ~1 week |
| E.1 | Evidence terms in Φ | several weeks (research) |
| E.3 | Learned residual coder | several weeks (research) |

**Realistic 30-day plan:** finish Phase B, run Phase 1.5 cleanup, knock out one Phase C trained-scene benchmark, ship Phase D.1 perf, push Phase F.1-3. That puts the project at "real-time-ish CPU renderer + ~2× killer ratio on Mip-NeRF 360 + on PyPI + on github with CI" by end of month.

**Realistic 90-day plan:** add D.2 + D.3 + F.4. That gets us "interactive WebGL viewer for `.3dphox` files + native germ format + first paper draft."

**Realistic 6-month plan:** start Phase E. Research-grade work.

---

## What I (the agent) need from you (Bug)

In rough priority:

1. **Wait for Dragon + Armadillo downloads to finish**, then say "go" and I run the rest of Phase B in ~10 minutes.
2. **Push to github** per [`PUSH_TO_GITHUB.md`](../PUSH_TO_GITHUB.md). One-time, ~5 minutes of your time.
3. **Pick a LICENSE** (recommend MIT).
4. **For Phase C:** point me at a trained 3DGS scene other than the Audi. Mip-NeRF 360 scenes are public; downloading is easiest from the [3DGS scenes page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) or any of the github mirrors. ~150-300 MB each.
5. **For Phase D.2 (browser viewer):** decide if you want a deeper browser experience (e.g., turn the SHOWCASE_T2.png into an interactive page where users can A/B Gaussian vs phoxoid) or if a static viewer is fine for now.
6. **For Phase F.4 (paper):** decide if you want a short paper / blog post written. If yes, I draft it from the existing reports.

---

## What I'll do without further input

- Wait for the rest of the Tier 1 set to download, then run them.
- Generate the SHOWCASE_T2 update with Tier 1 numbers added.
- Keep the codebase in a clean push-ready state.

Anything else needs your call.
