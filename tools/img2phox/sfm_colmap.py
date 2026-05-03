"""F.21 — COLMAP backend for SfM.

Drop-in alternative to run_sfm_chunked.py's pure-Python SfM. Uses pycolmap
(CPU-only build, installed via pip) for incremental SfM with proper bundle
adjustment. Produces the same _sfm_cache.pkl shape so downstream phoxel
code is unchanged.

Why: our spanning-tree global SfM caps at 20 dB on real photos because
gauge ambiguities accumulate across the rotation chain. Incremental SfM
(add cams one-at-a-time via PnP against growing model) avoids this.

Pipeline:
  1. Copy/symlink images into a temp COLMAP workdir
  2. extract_features (SIFT, CPU)
  3. match_exhaustive (SIFT)
  4. incremental_mapping (the actual SfM + BA)
  5. Convert COLMAP Reconstruction → our CameraBundle + PointCloud + obs

Usage:
  python3 -m img2phox.sfm_colmap --photos inputs/lego_excavator \\
      --max-photos 13 --max-dim 480 \\
      --out outputs/_lego_colmap_cache.pkl
"""
from __future__ import annotations
import sys, os, shutil, pickle, time, argparse, tempfile
sys.path.insert(0, 'tools')

import numpy as np
from pathlib import Path

try:
    import pycolmap
except ImportError:
    raise ImportError("pycolmap not installed. Run: pip install pycolmap")

from img2phox.load_photos import load_photoset
from img2phox.preprocess import preprocess_photoset
from img2phox.data_classes import (PhotoSet, CameraExtrinsics, CameraBundle,
                                       CameraIntrinsics, PointCloud, Photo)


def _quat_to_rotmat(q):
    """COLMAP quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def run_colmap_sfm(image_dir: Path, work_dir: Path = None,
                     max_photos: int = None, verbose: bool = True):
    """Run COLMAP incremental SfM on images in `image_dir`.

    Returns: dict with 'photoset', 'rec_cams', 'sparse', 'observations'
             matching the run_sfm_chunked.py output shape.
    """
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix='colmap_'))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    # Stage images into a single subdir for COLMAP
    img_subdir = work_dir / 'images'
    img_subdir.mkdir(exist_ok=True)
    src_imgs = sorted([p for p in image_dir.iterdir()
                         if p.suffix.lower() in ('.jpg', '.jpeg', '.png',
                                                   '.tiff', '.tif', '.bmp', '.webp')])
    if max_photos and len(src_imgs) > max_photos:
        idxs = np.linspace(0, len(src_imgs)-1, max_photos).astype(int)
        src_imgs = [src_imgs[i] for i in idxs]
    for p in src_imgs:
        dst = img_subdir / p.name
        if not dst.exists():
            shutil.copy(p, dst)
    if verbose:
        print(f'  staged {len(src_imgs)} images at {img_subdir}', flush=True)

    db_path = work_dir / 'database.db'
    sparse_dir = work_dir / 'sparse'
    sparse_dir.mkdir(exist_ok=True)

    # COLMAP step 1: extract features
    t0 = time.perf_counter()
    if not db_path.exists():
        pycolmap.extract_features(database_path=db_path, image_path=img_subdir,
                                     extraction_options=pycolmap.FeatureExtractionOptions())
        if verbose: print(f'  COLMAP feature extraction: {time.perf_counter()-t0:.1f}s', flush=True)

    # COLMAP step 2: matching (sequential for video-style >50 frames; exhaustive otherwise)
    t1 = time.perf_counter()
    if len(src_imgs) > 40:
        pycolmap.match_sequential(database_path=db_path)
        if verbose: print(f'  COLMAP sequential matching ({len(src_imgs)} imgs): '
                            f'{time.perf_counter()-t1:.1f}s', flush=True)
    else:
        pycolmap.match_exhaustive(database_path=db_path)
        if verbose: print(f'  COLMAP exhaustive matching ({len(src_imgs)} imgs): '
                            f'{time.perf_counter()-t1:.1f}s', flush=True)

    # COLMAP step 3: incremental mapping
    t2 = time.perf_counter()
    map_opts = pycolmap.IncrementalPipelineOptions()
    map_opts.min_num_matches = 15
    map_opts.multiple_models = False
    reconstructions = pycolmap.incremental_mapping(
        database_path=db_path, image_path=img_subdir,
        output_path=sparse_dir, options=map_opts)
    if verbose: print(f'  COLMAP incremental mapping: {time.perf_counter()-t2:.1f}s', flush=True)

    # Pick the largest reconstruction
    if len(reconstructions) == 0:
        raise RuntimeError("COLMAP produced no reconstructions")
    rec_idx = max(reconstructions.keys(), key=lambda k: reconstructions[k].num_images())
    rec = reconstructions[rec_idx]
    if verbose:
        print(f'  picked reconstruction: {rec.num_images()} cams, '
              f'{rec.num_points3D()} 3D points', flush=True)

    return _convert_reconstruction_to_cache(rec, src_imgs, img_subdir, verbose=verbose)


def _convert_reconstruction_to_cache(rec, src_imgs, img_subdir, verbose=True):
    """Convert pycolmap.Reconstruction to our CameraBundle + PointCloud + obs."""
    # Build PhotoSet from the same staged images (ensures filename order matches)
    from img2phox.load_photos import load_photo
    photos = []
    for p in src_imgs:
        photos.append(load_photo(p, max_dim=None))
    photoset_full = PhotoSet(photos=photos)

    # Map COLMAP image names -> our photo index
    name_to_idx = {p.path.name: i for i, p in enumerate(photoset_full.photos)}

    # Cameras (intrinsics) — COLMAP can give per-image cameras; we take the first
    cam_ids = list(rec.cameras.keys())
    cam0 = rec.cameras[cam_ids[0]]
    fx, fy, cx, cy = _extract_intrinsics(cam0)
    intr = CameraIntrinsics(focal_x=fx, focal_y=fy, cx=cx, cy=cy,
                              width=cam0.width, height=cam0.height)
    if verbose:
        print(f'  intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, '
              f'{cam0.width}x{cam0.height}', flush=True)

    # Extrinsics per registered image
    valid_indices = []
    extrinsics = []
    image_id_to_local_idx = {}  # COLMAP image ID -> our list index
    for img_id, img in rec.images.items():
        if img.name not in name_to_idx: continue
        local_idx = name_to_idx[img.name]
        if not img.has_pose: continue
        cfw = img.cam_from_world()
        qx, qy, qz, qw = cfw.rotation.quat
        R = _quat_to_rotmat([qw, qx, qy, qz])
        t = np.array(cfw.translation, dtype=np.float64)
        extrinsics.append((local_idx, CameraExtrinsics(R=R.astype(np.float32),
                                                          t=t.astype(np.float32))))
        image_id_to_local_idx[img_id] = local_idx

    extrinsics.sort(key=lambda x: x[0])
    valid_indices = [li for li, _ in extrinsics]
    rec_cams = CameraBundle(intrinsics=intr,
                              extrinsics=[e for _, e in extrinsics])
    valid_photoset = PhotoSet(photos=[photoset_full.photos[li] for li in valid_indices])
    if verbose:
        print(f'  registered: {len(valid_indices)}/{len(src_imgs)} cams', flush=True)

    # 3D points + observations_per_cam
    pts3d_list = []; colors_list = []
    point_id_to_local = {}
    observations = [[] for _ in valid_indices]
    local_to_obs_idx = {li: i for i, li in enumerate(valid_indices)}
    image_id_to_obs_idx = {iid: local_to_obs_idx[li] for iid, li in image_id_to_local_idx.items()
                            if li in local_to_obs_idx}
    for pt_id, pt in rec.points3D.items():
        new_id = len(pts3d_list)
        pts3d_list.append(pt.xyz)
        colors_list.append(pt.color / 255.0 if pt.color.dtype.kind == 'u' else pt.color)
        point_id_to_local[pt_id] = new_id
        # walk track
        for elem in pt.track.elements:
            iid = elem.image_id
            if iid not in image_id_to_obs_idx: continue
            obs_idx = image_id_to_obs_idx[iid]
            point2D = rec.images[iid].points2D[elem.point2D_idx]
            u, v = float(point2D.xy[0]), float(point2D.xy[1])
            observations[obs_idx].append((new_id, u, v))
    pts = np.asarray(pts3d_list, dtype=np.float32) if pts3d_list else np.zeros((0,3), dtype=np.float32)
    colors = np.asarray(colors_list, dtype=np.float32) if colors_list else np.zeros((0,3), dtype=np.float32)
    sparse = PointCloud(xyz=pts, colors=colors)
    if verbose:
        print(f'  -> {len(pts)} 3D points, {sum(len(o) for o in observations)} observations',
              flush=True)

    return {
        'photoset': valid_photoset,
        'rec_cams': rec_cams,
        'sparse': sparse,
        'observations': observations,
        'valid_idx': valid_indices,
        'n_views': len(src_imgs),
        'backend': 'colmap',
    }


def _extract_intrinsics(cam):
    """Extract (fx, fy, cx, cy) from a pycolmap.Camera. Handles
    SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV models."""
    params = list(cam.params)
    model_name = str(cam.model)
    if 'SIMPLE_PINHOLE' in model_name or 'SIMPLE_RADIAL' in model_name:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:  # PINHOLE / RADIAL / OPENCV all start with fx, fy, cx, cy
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return float(fx), float(fy), float(cx), float(cy)


def _quat_normalize_to_wxyz(q):
    """Detect quaternion convention and return (w, x, y, z)."""
    # pycolmap.Rotation3d.quat returns (x, y, z, w) in some versions.
    # Heuristic: w is typically closer to +1 for small rotations than x,y,z.
    # If q[3] has the largest absolute value, it's likely w.
    q = np.asarray(q, dtype=np.float64)
    if abs(q[3]) >= abs(q[0]) and abs(q[3]) >= abs(q[1]) and abs(q[3]) >= abs(q[2]):
        # xyzw convention
        return q[3], q[0], q[1], q[2]
    elif abs(q[0]) >= abs(q[1]) and abs(q[0]) >= abs(q[2]) and abs(q[0]) >= abs(q[3]):
        # wxyz convention
        return q[0], q[1], q[2], q[3]
    else:
        # fallback: assume xyzw (pycolmap default)
        return q[3], q[0], q[1], q[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--photos', type=str, required=True)
    p.add_argument('--max-photos', type=int, default=None)
    p.add_argument('--max-dim', type=int, default=480)
    p.add_argument('--work-dir', type=str, default=None,
                    help='COLMAP working directory (default: temp)')
    p.add_argument('--out', type=str, required=True,
                    help='output _sfm_cache.pkl')
    args = p.parse_args()

    t0 = time.perf_counter()
    cache = run_colmap_sfm(Path(args.photos),
                              work_dir=Path(args.work_dir) if args.work_dir else None,
                              max_photos=args.max_photos,
                              verbose=True)
    print(f'COLMAP SfM total: {time.perf_counter()-t0:.1f}s', flush=True)

    with open(args.out + '.tmp', 'wb') as f:
        pickle.dump(cache, f)
    with open(args.out + '.tmp', 'rb') as fi, open(args.out, 'wb') as fo:
        fo.write(fi.read())
    print(f'  -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
