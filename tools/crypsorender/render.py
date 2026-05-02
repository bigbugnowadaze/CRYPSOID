"""Main render orchestrator."""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .io.phox_loader import load_3dphox
from .io.ply_loader import load_ply
from .io.splat_buffer import SplatBuffer
from .math.sh import eval_sh_color
from .output.contact_sheet import make_contact_sheet_3panel
from .output.metrics import compute_metrics, save_metrics
from .output.png import save_png
from .pipeline.camera import Camera, CameraParams
from .pipeline.project import project_splats
from .pipeline.rasterize import composite
from .pipeline.sort import sort_splats_by_depth
from .pipeline.tile import bin_splats_to_tiles


def render_frame(
    scene: SplatBuffer,
    camera: Camera,
    use_sh: bool = True,
    max_points: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Render a single frame.

    Args:
        scene: SplatBuffer with splat data
        camera: Camera object
        use_sh: if True, evaluate SH; if False, use DC only
        max_points: if > 0, subsample to this many splats

    Returns:
        (framebuffer, alpha, timing_dict)
    """
    timing = {}

    # Optionally subsample splats
    if max_points > 0 and max_points < scene.n:
        rng = np.random.default_rng(2030)
        idx = rng.choice(scene.n, size=max_points, replace=False)
        subset = SplatBuffer(
            n=max_points,
            xyz=scene.xyz[idx],
            scales=scene.scales[idx],
            quats=scene.quats[idx],
            opacities=scene.opacities[idx],
            sh_dc=scene.sh_dc[idx],
            sh_rest=scene.sh_rest[idx] if scene.sh_rest is not None else None,
            tier=scene.tier[idx] if scene.tier is not None else None,
        )
        scene = subset

    # Project splats
    t0 = time.perf_counter()
    centers_2d, cov_2d_inv, radii, depths, keep, splat_indices = project_splats(
        scene, camera
    )
    timing["project_s"] = time.perf_counter() - t0

    m = centers_2d.shape[0]
    if m == 0:
        return np.zeros((camera.size, camera.size, 3), dtype=np.float32), np.zeros(
            (camera.size, camera.size), dtype=np.float32
        ), timing

    # Evaluate SH color for all visible splats
    t0 = time.perf_counter()
    original_indices = np.where(keep)[0]
    view_dirs = scene.xyz[original_indices] - camera.eye
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)

    if use_sh and scene.sh_rest is not None:
        sh_rest_vis = scene.sh_rest[original_indices]
    else:
        sh_rest_vis = None

    sh_dc_vis = scene.sh_dc[original_indices]
    rgb_vals = eval_sh_color(sh_dc_vis, sh_rest_vis, view_dirs, view_clip=True)
    timing["sh_eval_s"] = time.perf_counter() - t0

    # Opacity
    opacity_vals = scene.opacities[original_indices]

    # Tile binning
    t0 = time.perf_counter()
    tile_lists, tile_counts, n_tiles = bin_splats_to_tiles(
        centers_2d, radii, camera.size, tile_size=16
    )
    timing["tile_binning_s"] = time.perf_counter() - t0

    # Composite
    t0 = time.perf_counter()
    framebuffer, alpha_accum = composite(
        scene,
        camera,
        centers_2d,
        cov_2d_inv,
        radii,
        depths,
        original_indices,
        tile_lists,
        tile_counts,
        n_tiles,
        rgb_vals,
        opacity_vals,
    )
    timing["composite_s"] = time.perf_counter() - t0

    return framebuffer, alpha_accum, timing


def render_and_save(
    scene_path: Path,
    is_phox: bool,
    out_dir: Path,
    camera_params: CameraParams,
    use_sh: bool = True,
    max_points: int = 0,
) -> dict:
    """Load scene, render, and save outputs.

    Args:
        scene_path: path to .ply or .3dphox file
        is_phox: if True, load as .3dphox; else load as .ply
        out_dir: output directory
        camera_params: camera parameters
        use_sh: if True, evaluate SH
        max_points: subsample parameter

    Returns:
        dict with render results and metadata
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load scene
    t0 = time.perf_counter()
    if is_phox:
        scene = load_3dphox(scene_path)
    else:
        scene = load_ply(scene_path)
    load_time = time.perf_counter() - t0

    # Setup camera
    camera = Camera(scene.xyz, camera_params)

    # Render
    t0 = time.perf_counter()
    framebuffer, alpha_accum, timing = render_frame(
        scene, camera, use_sh=use_sh, max_points=max_points
    )
    render_time = time.perf_counter() - t0
    timing["total_render_s"] = render_time

    # Convert to uint8 and save PNG
    framebuffer_uint8 = np.clip(framebuffer * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(framebuffer_uint8, "RGB")
    img_path = out_dir / "frame.png"
    img.save(img_path)

    # Tier dispatch counts
    if scene.tier is not None:
        tier_counts = {
            "A": int((scene.tier == 0).sum()),
            "B": int((scene.tier == 1).sum()),
            "C": int((scene.tier == 2).sum()),
        }
    else:
        tier_counts = {}

    return {
        "image_path": str(img_path),
        "framebuffer": framebuffer_uint8,
        "load_time_s": load_time,
        "render_time_s": render_time,
        "timing": timing,
        "scene_format": scene.scene_format,
        "splat_count": scene.n,
        "tier_counts": tier_counts,
        "use_sh": use_sh,
    }
