# CRYPSOID `.3dphox` WebGL viewer

Self-contained HTML viewer that loads `.3dphox` files in any modern browser
and renders them with WebGL2. **True single-file: open `index.html` directly
from your file system (double-click) — no server, no build step.** All
decoder JS is inlined into the HTML so ES-module CORS restrictions on
`file://` don't apply.

This is the Tier 3 client-side end-user deliverable. The viewer's GPU code
runs in the browser; the **CRYPSOID core codebase remains GPU-free** — this
viewer is a separate consumer of the format, not a dependency.

## Files

- `index.html` — UI + WebGL2 renderer
- `phox_decoder.js` — JavaScript decoder for the `.3dphox` format. Reuses the
  same logic as the Python loaders in `tools/crypsorender/io/phox_loader.py`.

## Running it

### Option A — open with a static server (recommended)

The page loads `phox_decoder.js` as an ES module, which most browsers
restrict from `file://`. Easiest: any tiny static server.

```bash
cd viewer
python3 -m http.server 8000
# then open http://localhost:8000/ in a browser
```

### Option B — drag a `.3dphox` file onto the open page

Once the viewer is open, drag any `.3dphox` file from your file system onto
the window. Or click "Load .3dphox" and pick one with the file dialog.

### What it can load

| File | Loads? | Notes |
|---|---|---|
| `outputs/v25_attribute_group_render_container.3dphox` | yes | full SH from raw int8 stream |
| `outputs/v28_sh_vq_render_container.3dphox` | yes | SH reconstructed from VQ codebook |
| `outputs/v28_sh_vq_exact_archive_container.3dphox` | yes | SH reconstructed via VQ + per-tier-group residuals (bit-exact to v25) |
| `outputs/v31_audi_with_normals.3dphox` | yes (v31 trailer parsed) | also exposes per-splat normals for the lit modes |
| `outputs/v31_audi_normals_edges.3dphox` | yes (v31 trailer parsed) | normals + kNN edges chunks loaded |
| `outputs/v31_audi_full_v33.3dphox` | yes (v31 trailer parsed) | normals + edges + v33 material_hint chunk; enables lit + dim-floaters + material-overlay modes |
| `recovery_v2/v27_attribute_group_sh_vq_render_container.3dphox` | yes | identical layout to v28 render |

A standard 3DGS `.ply` is **not** supported by this viewer (use one of the
many other splat viewers for that — antimatter15/splat, mkkellogg/GaussianSplats3D,
SuperSplat, etc.). The point of this viewer is to prove `.3dphox` is a
first-class format.

## Controls

| Action | Effect |
|---|---|
| Mouse drag | orbit |
| Mouse wheel | zoom |
| Render mode dropdown | `Full color` (DC + SH) / `DC only` / `Tier overlay` / `Lit` (v32a Lambert, needs v31 normals) / `Lit + dim floaters` (v32a + v33 material_hint) / `Material overlay` (v33 hint colors) |
| Splat scale slider | uniform multiplier on splat sigma |
| Splat alpha cap slider | uniform multiplier on opacity |
| **Phoxoidal density slider** | 0.0 = pure Gaussian; 1.0 = full phoxoidal cubic-cusp density. Watch sharp creases tighten as you push it up. |
| **Load .phoxseq button** | Loads a sibling v34 timeline file. Reveals a frame slider + Play/Reset controls. Try with `outputs/v34_audi_halo_bloom.phoxseq` over the v40 Audi base. |

## What's not implemented (yet)

- ~~**Phoxoidal density evaluation in the fragment shader.**~~ **DONE (2026-05-02).**
  Fragment shader now evaluates `exp(-2 · (r² + 0.55·uPhoxStrength·cubic_term))`
  where `cubic_term = |x·y²| + |y·x²|` mirrors the Pearcey germ's cubic ω
  coefficient. `uPhoxStrength` is exposed as a slider — 0.0 matches every other
  splat viewer; 1.0 is full phoxoidal cubic-cusp falloff. The faithful 5-coef
  closest-point Newton path remains v0.5 work, but the cubic-cusp approximation
  here is what most users will actually see the difference on.
- ~~**CPU depth-sort for proper "over" compositing.**~~ **DONE (2026-05-02).**
  `sort_worker.js` is now wired into `index.html`. The viewer spawns a
  worker on scene load, sends per-frame view matrices, gets back back-to-
  front splat indices, reorders the per-instance buffers, and uses standard
  `(SRC_ALPHA, ONE_MINUS_SRC_ALPHA)` "over" compositing. Sort cadence is
  throttled by hashing the view-matrix Z-row so we only re-sort when the
  camera actually moves enough to matter.
- **Camera animation / turntable export.**
- **Loading from URL** (only File API drag-and-drop currently).

## Browser requirements

- WebGL 2 (Chrome 56+, Firefox 51+, Safari 15+).
- `DecompressionStream` API (Chrome 80+, Firefox 113+, Safari 16.4+) — used to
  decompress the zlib-compressed chunks. If your browser doesn't support
  it, the viewer will fail with a clear error.

## How `phox_decoder.js` mirrors the Python loader

| Python (`tools/crypsorender/io/phox_loader.py`) | JavaScript (`phox_decoder.js`) |
|---|---|
| `decode_u24_xyz` | `decodeU24Xyz` |
| `decode_f16_scales` | `decodeF16Scales` |
| `decode_i16_quats` | `decodeI16Quats` |
| `decode_dc_rgb_opacity_u8` | `decodeDcRgbOpacityU8` |
| `load_3dphox_v28_render` (VQ SH path) | `reconstructShVq128` |
| `load_3dphox_v28_archive` (VQ + per-tier residual SH) | `reconstructShExactArchive` |
| `normals_codec.read_normals_chunk` | `decodeNormalsChunk` |
| `edges_codec.read_edges_chunk` | `decodeEdgesChunk` |
| `material_codec.read_material_chunk` | `decodeMaterialChunk` |
| (no Python equivalent — v31 trailer is browser-side) | `parseV31Trailer` |

If you change the Python decoder, mirror the change in JS and vice versa.
There's no shared header file; the format itself is the contract.

## Honest caveat

This viewer is a *demonstration that `.3dphox` is portable*, not a polished
product. It runs fine on a 200k-splat subsample on a modest GPU. At full
763k splats with no depth sort it'll have visible blending order artifacts
on the halo spl