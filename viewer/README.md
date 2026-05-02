# CRYPSOID `.3dphox` WebGL viewer

Self-contained HTML viewer that loads `.3dphox` files in any modern browser
and renders them with WebGL2. **Single-file, no build step, no server
needed for local-file usage that uses the browser's File API; for
fetch-loading you need a tiny static server.**

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
| Render mode dropdown | `Full color` (DC + SH) / `DC only` / `Tier overlay` |
| Splat scale slider | uniform multiplier on splat sigma |
| Splat alpha cap slider | uniform multiplier on opacity |

## What's not implemented (yet)

- **Phoxoidal density evaluation in the fragment shader.** The viewer
  currently uses standard Gaussian density `exp(-2|p|²)`. To exercise the
  faithful 5-coef phoxoidal path here would need the closest-point Newton
  solver in GLSL — doable as v0.5 work.
- **CPU depth-sort for proper "over" compositing.** The blending mode used
  is `(ONE_MINUS_DST_ALPHA, ONE)` — order-independent additive accumulation.
  This is correct for the visualization use case (you can see the model)
  but not perfectly correct for transparent splat compositing. A
  `sort_worker.js` is provided in this folder (~50 lines, counting-radix on
  16-bit quantized depth keys) but is **not yet wired into `index.html`**.
  Wiring it requires adding an instance-index buffer and switching to
  indexed instancing in the draw call. v0.5 work.
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

If you change the Python decoder, mirror the change in JS and vice versa.
There's no shared header file; the format itself is the contract.

## Honest caveat

This viewer is a *demonstration that `.3dphox` is portable*, not a polished
product. It runs fine on a 200k-splat subsample on a modest GPU. At full
763k splats with no depth sort it'll have visible blending order artifacts
on the halo splats — fixable in a v0.5 pass that adds a Web Worker sort.
