# F.30 — Blender → `.3dphox` bridge spec

**Status:** SPEC — phase 1 of three (F.30.1)
**Owner:** CRYPSOID content pipeline
**Date:** 2026-05-03

## 1. Why this exists

After F.26 (procedural CGI studio) and F.27 (procedural CGI car + HDRI),
the renderer is proven to produce clean output from clean input. But the
input was written by hand in Python — anyone wanting to drive the renderer
with their own content has to learn our procedural API.

This bridge closes that gap. Anyone with a 3D scene exports it from Blender
(or any DCC) as a single OBJ + MTL file pair and gets a `.3dphox` they can
render or load in the WebGL viewer.

The "input data, not pipeline" thesis from F.26 then becomes a workflow:

> Blender scene → File → Export → Wavefront (.obj)
>   → `python3 tools/build_blender_phox.py scene.obj`
>   → `outputs/scene.3dphox`
>   → drag onto `viewer/index.html` or render via `tools/render_*.py`

## 2. Input formats

### Primary: Wavefront OBJ + MTL

- **Why OBJ first**: it's the lowest common denominator across every 3D
  tool. Blender exports it natively (no add-on needed). It's text-based
  and trivial to parse from scratch — no third-party dependency required
  to keep with the project's "no GPU deps" spirit. Materials live in a
  sibling `.mtl` file referenced by `mtllib`.
- **OBJ subset we'll support**:
  - `v x y z`        — vertex positions
  - `vn nx ny nz`    — vertex normals (optional; we compute face normals if absent)
  - `vt u v`         — UV coords (parsed but only used for documentation in v1)
  - `f a b c [d ...]`  — face (triangle, quad, or n-gon; we triangulate fans)
  - `f a/t/n` syntax — full vertex/uv/normal indices
  - `usemtl name`    — assign current material to subsequent faces
  - `mtllib file.mtl` — load material library
  - `g name` / `o name` / `s ...` — group / object / smoothing — parsed but
    grouping doesn't affect splat output
- **MTL subset**:
  - `newmtl name`
  - `Kd r g b`       — diffuse colour → albedo
  - `Ka r g b`       — ambient (ignored; renderer adds its own ambient)
  - `Ks r g b`       — specular colour (used to bias metallic-ness if Pm absent)
  - `Ke r g b`       — emissive (parsed; treated as MATERIAL_HINT_EMISSIVE if non-zero)
  - `Ns shininess`   — Phong exponent → roughness via mapping below
  - `Pm`             — metalness (PBR extension — preferred over Ks heuristic)
  - `Pr`             — roughness (PBR extension — preferred over Ns mapping)
  - `d` or `Tr`      — opacity (1 - Tr)
  - `illum 0..10`    — illumination model — 3 / 5 / 7 hint at metallic, 9 at glass

Anything not listed is silently ignored.

### Future: glTF / glb

Out of scope for v1. glTF would give us PBR metallic-roughness natively
(no MTL ambiguity) and texture maps. Worth doing, but OBJ ships first.

### Future: textures (`map_Kd` etc.)

Out of scope for v1. v1 uses solid per-material colours. Textured input
would require us to either bake the texture into per-vertex colour at
sample time (easy, if we sample in UV space too) or store a texture +
UV per splat (format extension). v2 candidate.

## 3. Surface sampling

Each face contributes splats proportional to its area. For a target total
N splats:

```
total_area = sum of face areas
for face f:
    n_face = max(1, round(N * area(f) / total_area))
    sample n_face points uniformly on f
```

For triangles, uniform sampling uses the standard barycentric trick:

```
u, v = random in [0, 1)
if u + v > 1: u, v = 1 - u, 1 - v
p = v0 * (1 - u - v) + v1 * u + v2 * v
```

Quads and n-gons are triangulated via fan triangulation from the first
vertex (`v0, v_i, v_{i+1}`) before sampling.

**Per-splat normal** = face normal `(v1 - v0) × (v2 - v0)`, normalised. We
deliberately don't interpolate vertex normals — sharp edges look better as
hard splat boundaries than as smooth gradient transitions for our renderer's
splat-disk geometry.

**Per-splat sigma** scales with sqrt(area / n_face) so dense small faces get
small splats and sparse large faces get bigger splats. Default rule:

```
sigma = clip(sqrt(area / n_face) * 0.7, 0.003, 0.05)
```

This matches the procedural-car build's hand-tuned sigma values.

## 4. MTL → PBR mapping

Renderer expects (albedo, metallic, roughness, F0, kd) per splat. From MTL:

| MTL field | PBR field | Mapping |
|---|---|---|
| `Kd` | `albedo` | direct copy, RGB |
| `Pm` | `metallic` | direct, clamped to [0, 0.95] |
| `Pr` | `roughness` | direct, clamped to [0.05, 0.95] |
| `Ns` | `roughness` (if no `Pr`) | `roughness = 1 - clip(log10(max(Ns, 1)) / 3, 0, 0.95)` (Ns=1000 → 0.05, Ns=1 → 0.95) |
| `Ks` brightness | `metallic` (if no `Pm`) | `metallic = clip(mean(Ks), 0, 0.95)` |
| `illum` ∈ {3, 5, 7} | bias `metallic += 0.4` if Ks dominant | mirror-ish models |
| `Ke` non-zero | `MATERIAL_HINT_EMISSIVE` | also adds `Ke` to base colour |
| `d` | `opacity` | direct |

Then derived:

- `F0 = 0.04 * (1 - metallic) + albedo * metallic` (standard PBR)
- `kd = 1 - metallic`

If a face has no `usemtl` directive, default material is `(albedo=0.7 grey, metallic=0.05, roughness=0.6)`.

## 5. File-by-file delta

| File | Phase | Change |
|---|---|---|
| `docs/blender_bridge_spec.md` | F.30.1 | new (this file) |
| `tools/img2phox/obj_loader.py` | F.30.2 | new — parse OBJ + MTL → simple data classes |
| `tools/build_blender_phox.py` | F.30.2 | new — sample faces, encode .3dphox |
| `tools/render_blender_demo.py` | F.30.3 | new — render imported scene via photoreal stack |
| `tools/build_blender_3way_panel.py` | F.30.3 | new — 3-panel: procedural / OBJ-imported / scan |
| `reports/F30_blender_bridge.md` | F.30.3 | new — results + caveats + next steps |

## 6. Integration with existing code

The output `.3dphox` is byte-identical in format to F.26/F.27 output:
v25 base + v31 trailer (normals, edges, material_hints) + v40 trailer
(kappa, cusp). The existing renderers consume it without modification.

We re-use:
- `tools/build_cgi_studio_phox.py`'s `normals_to_quats`, area-weighted sigma
- `crypsorender.io.normals_codec.write_normals_chunk`
- `crypsorender.io.edges_codec.write_edges_chunk` + `derive_knn_edges`
- `crypsorender.io.material_codec.write_material_chunk` + `derive_mip_zoom`
- `crypsorender.io.germ_codec.write_kappa_chunk` + `write_cusp_chunk`
- `img2phox.encode.encode_blobbundle_to_3dphox` for the v25 base

## 7. Acceptance gates

| # | Test | What it proves |
|---|---|---|
| 1 | `obj_loader.py` parses Stanford bunny .obj (in `inputs/`) without error, returns sane vertex/face counts | parser works on real OBJ |
| 2 | Round-trip: build .3dphox from a procedural cube OBJ, load via `load_3dphox`, verify N matches expected face-area-weighted sample count ±5% | sampler is calibrated |
| 3 | Build .3dphox from `inputs/bunny.obj`, render via existing photoreal stack, get a recognisable bunny shape | end-to-end works on real input |
| 4 | Per-material PBR test: build OBJ with two materials (matte red + chrome metal), verify renderer shows the chrome reflection on the chrome region but not the red region | PBR mapping is correct |

Gate 1 is a unit test. Gate 2 is a smoke test. Gates 3 + 4 are render
deliverables in F.30.3.

## 8. Time estimate

- F.30.1 (this spec): done
- F.30.2 (loader + build script + Gates 1-2): ~3 hours
- F.30.3 (Gates 3-4 + report): ~1 hour

Total: roughly half a day of focused work.

## 9. Out of scope (future)

- Texture maps (v2 candidate — store per-splat UV + texture, sample at render)
- glTF / glb support (v2 candidate — gives us native PBR + textures)
- Per-scene background HDRI selection from MTL hints
- Light export (Blender lamp → directional light in render config)
- Animation frames (would feed into the v34 .phoxseq spec)

## 10. References

- Wavefront OBJ format: spec hosted at FileFormat.info.
- MTL extensions for PBR: NextGenPBR proposal (2018) — `Pm`, `Pr`, `Ps`, `Pc`.
- Blender's OBJ exporter: defaults to writing both `Pm` and `Pr` if the
  Principled BSDF shader is used, which is the default in Blender 4.x.
