# Phase F — Image → `.3dphox` Compiler (one-pager)

**Status:** spec drafted + scaffolding + synthetic round-trip working, 2026-05-02.
**Goal:** turn a folder of photos of a scene into a `.3dphox` file the existing CRYPSOID renderer can load.
**Depends on:** existing `.3dphox` writer (v25 → v40), CRYPSOID renderer (Bar 2). No deps on lit-stack — image→phox is pure geometry.

## Why

So far CRYPSOID is consumer-side: take an existing trained 3DGS PLY, compress it. The producer side (training the splats from photos) has been someone else's problem. Phase F closes that loop. After this, a user can:

```
$ tools/img2phox/cli.py photos/ --output scene.3dphox
$ python3 -m crypsorender --scene scene.3dphox --camera 35,18 --output render.png
```

Without ever touching COLMAP, gsplat, or any training pipeline.

## Honest scope statement (read this first)

A useful image→splats pipeline is a **3-to-6 month engineering project at minimum**. COLMAP is years of structure-from-motion code. 3DGS training is months of optimizer engineering. We're CPU-only, which makes us slower than CUDA-trained references by 100× on the optimization step.

What we will deliver in Phase F:
- **F.0:** This spec (this doc).
- **F.1:** Architectural scaffolding: package layout, data classes, API contract.
- **F.2:** Working SfM stage on **synthetic textured scenes** (proves the math). Real-photo SfM is downstream of this.
- **F.3:** Working blob optimizer on synthetic data. ~10k blobs, scenes that fit in RAM. Real-photo scaling is downstream.
- **F.4:** End-to-end synthetic demo: render synthetic scene → "photos" → recover .3dphox → re-render → side-by-side.

What Phase F does **not** deliver, reserved for Phase F.5+:
- Real-photo SfM (needs robust feature matching across exposure changes, lens distortion, scale variance — real-world hardness)
- Multi-view stereo densification (depth maps + fusion)
- Dense optimization at trained-3DGS scale (763k+ blobs)
- Distortion / lens calibration inference
- Background segmentation
- Loop closure / global bundle adjustment

We will emit `.3dphox` files **on synthetic data** in Phase F. Real photos work after F.5+.

## Architecture (5 stages)

```
photos/                                            (input: directory of images)
   |
   v   F.1 load_photos.py
PhotoSet { images[], exif[] }
   |
   v   F.2 sfm.py       (Structure-from-Motion)
CameraBundle { intrinsics, extrinsics[N] }   +   PointCloud (sparse, ~1k pts)
   |
   v   F.3 mvs.py       (deferred to F.5; uses sparse cloud directly for now)
PointCloud (denser, ~100k pts)
   |
   v   F.3 optimize.py  (CPU blob optimizer)
BlobBundle { xyz, scales, quats, opacity, sh_dc, sh_rest }
   |
   v   F.4 encode.py
scene.3dphox  (v25 → v28-archive → optionally v31/v40)
```

Each stage has a clean input/output contract. Stages can be swapped out (e.g. plug in COLMAP for the SfM stage if Bug ever wants to). The whole pipeline is a thin wrapper around the contract.

## Data classes

```python
@dataclass
class Photo:
    path: Path
    image: np.ndarray   # (H, W, 3) uint8 or float32
    exif: dict          # focal length hints, etc.

@dataclass
class CameraIntrinsics:
    focal_x: float
    focal_y: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = ()   # k1, k2, p1, p2 (Brown-Conrady), defer to F.5

@dataclass
class CameraExtrinsics:
    R: np.ndarray   # (3, 3) world-to-camera rotation
    t: np.ndarray   # (3,)   world-to-camera translation

@dataclass
class CameraBundle:
    intrinsics: CameraIntrinsics      # currently shared across all cameras
    extrinsics: list[CameraExtrinsics]

@dataclass
class PointCloud:
    xyz: np.ndarray              # (M, 3)
    colors: np.ndarray | None    # (M, 3) RGB
    visibility: list[set[int]]   # per-point set of camera indices that saw it

@dataclass
class BlobBundle:
    xyz: np.ndarray              # (N, 3)
    scales: np.ndarray           # (N, 3) log-sigma
    quats: np.ndarray            # (N, 4) wxyz
    opacity: np.ndarray          # (N,) sigmoid logit
    sh_dc: np.ndarray            # (N, 3)
    sh_rest: np.ndarray | None   # (N, 45) bands 1..3
```

## Algorithm choices (F.0 baseline; can be swapped)

### F.2 SfM — incremental (Snavely-style)
1. Detect features in every photo (ORB or SIFT — ORB for speed, no patent issues).
2. Match pairwise; keep matches with epipolar geometry consistency (RANSAC F-matrix).
3. Bootstrap: pick a 2-photo pair with the most matches, triangulate.
4. Add cameras one at a time, register via PnP (RANSAC), triangulate new points.
5. Bundle adjustment every K cameras (sparse Levenberg-Marquardt, scipy).

For F.2 synthetic test we skip step 1 (we know the features — they're the rendered cube vertices) and only validate steps 3-5.

### F.3 Blob optimizer — per-photo photometric loss
1. Initialize blobs at the sparse point cloud (xyz from PointCloud).
2. Initialize scales = mean nearest-neighbor distance, quats = identity, opacity = 0.5, sh_dc = point's color.
3. For each photo: project blobs → forward render → compute L1 photometric loss vs photo.
4. Backprop the loss to blob params via numerical gradient (finite-diff initially; analytic later).
5. Repeat with adaptive density control (split big blobs, prune low-opacity blobs).

For the synthetic test in F.3 we run this for 50-200 iterations on a ~10k-blob scene, ~1-2 minutes on CPU.

### F.4 Encode — straight to v25 archive
The trained BlobBundle has all the fields the v25 attribute-group container expects. Run it through `build_v25_attribute_group.py` style encoding. v28-archive (VQ + residuals), v31 (normals + edges), v40 (germ chunks) are all optional decorators on top.

## Acceptance gates (Phase F)

1. **F.1:** `tools/img2phox/` package importable; PhotoSet loads N images cleanly.
2. **F.2:** Synthetic scene with 8 known camera poses → recovered poses within 5° rotation, 5% translation of ground truth.
3. **F.3:** Synthetic scene → reconstructed BlobBundle, reprojected photometric L1 < 10% of photo dynamic range across the held-out test view.
4. **F.4:** End-to-end run: photos folder → .3dphox → re-render through CRYPSOID renderer → PSNR vs original synthetic ground truth ≥ 25 dB.
5. **CI:** F.1 + F.2 + F.3 + F.4 all pass on the synthetic test in under 3 minutes total.

## What this enables (and what it doesn't)

**Enables:**
- Synthetic-scene roundtrip benchmarks for the renderer (ground-truth-driven quality metrics)
- Validation that the full producer-side math works in pure CPU
- A natural foundation for real-photo work in F.5+

**Doesn't enable yet:**
- "Drop a folder of iPhone photos in, get a usable .3dphox out." That's F.5+ work — needs robust feature matching, distortion calibration, exposure normalization, dense MVS.
- Competitive quality vs gsplat-trained references. We're CPU-only and starting with a much simpler optimizer.

## Files added in Phase F

```
docs/img2phox_spec.md                          — this doc
tools/img2phox/__init__.py
tools/img2phox/data_classes.py                 — Photo / CameraBundle / PointCloud / BlobBundle
tools/img2phox/load_photos.py                  — PhotoSet loader (PIL)
tools/img2phox/sfm.py                          — incremental SfM (synthetic mode for F.2)
tools/img2phox/optimize.py                     — CPU blob optimizer (numerical gradient)
tools/img2phox/encode.py                       — BlobBundle → .3dphox v25 archive
tools/img2phox/cli.py                          — top-level driver
tools/img2phox/synth_scene.py                  — synthetic textured-cube renderer for F.2/F.3 tests
tools/test_img2phox_round_trip.py              — F.2/F.3/F.4 acceptance test
```

## Phasing

| Phase | Effort | Status |
|---|---:|---|
| F.0 spec | done | this doc |
| F.1 scaffolding | 1 day | done in this session |
| F.2 synthetic SfM round-trip | ~1 day | done in this session |
| F.3 synthetic blob optimizer | ~2 days | done in this session (basic) |
| F.4 end-to-end synthetic demo | ~half day | done in this session |
| F.5 real-photo SfM (ORB + RANSAC + BA) | ~3-4 weeks | future |
| F.6 dense MVS | ~3-4 weeks | future |
| F.7 distortion + EXIF + exposure normalization | ~1-2 weeks | future |
| F.8 dense optimizer at trained-3DGS scale | ~6-8 weeks | future |

**Cumulative real-photo workable pipeline: 3-4 months from F.5 start.** Honest.

## Why this is the right next chapter (and why it's also dangerous)

CRYPSOID's competitive position has been "consumer-side only" — we accept gsplat's output as input. Phase F removes that dependency. Strategically that's huge: it means the project owns its own data pipeline.

But it's also a place where the project could lose focus. Image → 3D reconstruction is a massive research field; we don't need to compete on quality with NeRFStudio or COLMAP+gsplat. We need to ship a CPU-only "good enough for prototyping" path so the format is self-contained. The synthetic-scene baseline in F.2-F.4 is the right scope. Anything beyond that is future-Bug's call.
