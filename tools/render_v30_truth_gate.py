#!/usr/bin/env python3
"""
CRYPSOID v0.30 render truth gate — extended v28 vs original comparison with error heatmap,
tier visualization, SSIM, decode-time and render-time measurements, attribute parity checks.

Purpose:
- Render the original Audi PLY and the v28 decoded container from the same camera.
- Compute per-pixel MSE/MAE/PSNR/SSIM metrics.
- Visualize absolute-difference heatmap and tier assignments.
- Measure decode and render times for each major step.
- Check attribute parity (byte-by-byte) for the five passthrough chunks from v25.
- Produce a multi-panel contact sheet, standalone renders, and a comprehensive metrics JSON.

Truth note:
This is a CPU DC/opacity preview renderer, not a full CUDA/PlayCanvas/gsplat renderer.
It is meant to catch gross geometry/color parity errors before the real viewer path.
"""
from __future__ import annotations

import argparse
import binascii
import json
import math
import struct
import time
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_ply_header(f) -> Tuple[dict, int]:
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header ended unexpectedly")
        header_lines.append(line.decode("ascii", errors="replace").strip())
        if header_lines[-1] == "end_header":
            break
    header_len = f.tell()
    fmt = None
    vertex_count = None
    props: List[str] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element "):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            props.append(parts[-1])
    if fmt != "binary_little_endian":
        raise ValueError(f"Expected binary_little_endian PLY, got {fmt}")
    if vertex_count is None:
        raise ValueError("Could not find vertex count in PLY")
    return {"header": header_lines, "vertex_count": vertex_count, "properties": props}, header_len


def load_original_ply_or_zip(path: Path, member: str = "scene.ply") -> Dict[str, np.ndarray]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            chosen = member if member in z.namelist() else next(n for n in z.namelist() if n.lower().endswith(".ply"))
            data = z.read(chosen)
        import io
        f = io.BytesIO(data)
    else:
        f = path.open("rb")
    with f:
        meta, header_len = parse_ply_header(f)
        n = meta["vertex_count"]
        props = meta["properties"]
        raw = f.read(n * len(props) * 4)
        if len(raw) != n * len(props) * 4:
            raise ValueError(f"Expected {n * len(props) * 4} bytes of vertex data, got {len(raw)}")
        arr = np.frombuffer(raw, dtype="<f4").reshape(n, len(props))
    col = {name: i for i, name in enumerate(props)}
    xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float32)
    fdc = arr[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]].astype(np.float32)
    rgb = np.clip(fdc * C0 + 0.5, 0, 1)
    opacity = sigmoid(arr[:, col["opacity"]].astype(np.float32))
    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "count": n, "source": str(path)}


def read_container(path: Path):
    t0 = time.perf_counter()
    with path.open("rb") as f:
        magic = f.read(11)
        manifest_len = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(manifest_len))
        blob = f.read()
    t_read = time.perf_counter() - t0
    chunks = {c["name"]: c for c in manifest["chunks"]}

    def comp(name: str) -> bytes:
        c = chunks[name]
        return blob[c["offset"]:c["offset"] + c["compressed_bytes"]]

    def dec(name: str) -> bytes:
        return zlib.decompress(comp(name))

    return magic, manifest, blob, chunks, comp, dec, t_read


def decode_u24_xyz(raw: bytes, n: int, bounds_min: List[float], bounds_max: List[float]) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 9)
    q = np.empty((n, 3), dtype=np.uint32)
    for j in range(3):
        q[:, j] = (
            a[:, 3*j].astype(np.uint32)
            | (a[:, 3*j+1].astype(np.uint32) << 8)
            | (a[:, 3*j+2].astype(np.uint32) << 16)
        )
    mn = np.asarray(bounds_min, dtype=np.float32)
    mx = np.asarray(bounds_max, dtype=np.float32)
    return (q.astype(np.float32) / float((1 << 24) - 1)) * (mx - mn) + mn


def load_crypsoid_v28(path: Path) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Load v28 container and return data + decode timing."""
    magic, man, blob, chunks, comp, dec, t_read = read_container(path)

    t0_decomp = time.perf_counter()
    xyz_chunk = chunks["xyz_u24_fixed"]
    n = int(xyz_chunk["shape"][0])

    # Decompress all needed chunks
    xyz_raw = dec("xyz_u24_fixed")
    dc_raw = dec("dc_rgb_opacity_u8")
    tier_raw = dec("tier_labels_u8")

    t_decomp = time.perf_counter() - t0_decomp

    # Decode XYZ
    t0_xyz = time.perf_counter()
    xyz = decode_u24_xyz(xyz_raw, n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])
    t_xyz = time.perf_counter() - t0_xyz

    # Decode DC/opacity
    t0_dc = time.perf_counter()
    dc = np.frombuffer(dc_raw, dtype=np.uint8).reshape(n, 4)
    rgb = dc[:, :3].astype(np.float32) / 255.0
    opacity = dc[:, 3].astype(np.float32) / 255.0
    t_dc = time.perf_counter() - t0_dc

    # Decode tier labels
    t0_tier = time.perf_counter()
    tier_labels = np.frombuffer(tier_raw, dtype=np.uint8)
    t_tier = time.perf_counter() - t0_tier

    timing = {
        "read_container_s": t_read,
        "decompress_chunks_s": t_decomp,
        "decode_xyz_u24_s": t_xyz,
        "decode_dc_rgb_opacity_u8_s": t_dc,
        "decode_tier_labels_u8_s": t_tier,
    }

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "tier_labels": tier_labels,
        "count": n,
        "source": str(path),
        "magic": magic.decode(errors="replace"),
    }, timing


def camera_project(xyz: np.ndarray, yaw_deg: float, pitch_deg: float, distance: float, fov_deg: float, size: int):
    center = xyz.mean(axis=0)
    pts = xyz - center
    radius = np.linalg.norm(pts, axis=1).max()
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    eye_dir = np.array([
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ], dtype=np.float32)
    eye = center + eye_dir * radius * distance
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    world_up = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    rel = xyz - eye
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    keep = z > 1e-5
    focal = 0.5 * size / math.tan(math.radians(fov_deg) / 2)
    px = (x[keep] / z[keep]) * focal + size / 2
    py = -(y[keep] / z[keep]) * focal + size / 2
    return px, py, z[keep], keep


def render_points(scene: Dict[str, np.ndarray], size: int = 1024, max_points: int = 0,
                  yaw: float = 35, pitch: float = 18, distance: float = 2.4, fov: float = 42,
                  radius_px: int = 1) -> Image.Image:
    xyz = scene["xyz"]
    rgb = scene["rgb"]
    opacity = scene["opacity"]
    n = xyz.shape[0]
    if max_points and max_points < n:
        rng = np.random.default_rng(2030)
        idx = rng.choice(n, size=max_points, replace=False)
        xyz = xyz[idx]; rgb = rgb[idx]; opacity = opacity[idx]
    px, py, depth, keep = camera_project(xyz, yaw, pitch, distance, fov, size)
    rgb = rgb[keep]
    opacity = opacity[keep]
    xi = np.rint(px).astype(np.int32)
    yi = np.rint(py).astype(np.int32)
    inside = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi = xi[inside]; yi = yi[inside]; depth = depth[inside]; rgb = rgb[inside]; opacity = opacity[inside]
    order = np.argsort(depth)[::-1]
    xi = xi[order]; yi = yi[order]; rgb = rgb[order]; opacity = opacity[order]
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    alpha = np.zeros((size, size), dtype=np.float32)
    offsets = [(0, 0)] if radius_px <= 0 else [(dx, dy) for dy in range(-radius_px, radius_px+1) for dx in range(-radius_px, radius_px+1) if dx*dx+dy*dy <= radius_px*radius_px]
    for dx, dy in offsets:
        xx = xi + dx; yy = yi + dy
        m = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        a = np.clip(opacity[m] * 0.55, 0, 1)
        yy2 = yy[m]; xx2 = xx[m]
        old_a = alpha[yy2, xx2]
        new_a = a + old_a * (1 - a)
        canvas[yy2, xx2] = (rgb[m] * a[:, None] + canvas[yy2, xx2] * old_a[:, None] * (1 - a[:, None])) / np.maximum(new_a[:, None], 1e-6)
        alpha[yy2, xx2] = new_a
    out = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def render_tier_view(scene: Dict[str, np.ndarray], size: int = 1024, max_points: int = 0,
                     yaw: float = 35, pitch: float = 18, distance: float = 2.4, fov: float = 42,
                     radius_px: int = 1, tier_colors: Dict[int, Tuple[int, int, int]] = None) -> Image.Image:
    """Render splats colored by tier assignment."""
    if tier_colors is None:
        tier_colors = {0: (255, 64, 64), 1: (64, 255, 64), 2: (64, 64, 255)}

    xyz = scene["xyz"]
    tier_labels = scene["tier_labels"]
    n = xyz.shape[0]
    if max_points and max_points < n:
        rng = np.random.default_rng(2030)
        idx = rng.choice(n, size=max_points, replace=False)
        xyz = xyz[idx]; tier_labels = tier_labels[idx]

    px, py, depth, keep = camera_project(xyz, yaw, pitch, distance, fov, size)
    tier_labels = tier_labels[keep]
    xi = np.rint(px).astype(np.int32)
    yi = np.rint(py).astype(np.int32)
    inside = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi = xi[inside]; yi = yi[inside]; depth = depth[inside]; tier_labels = tier_labels[inside]
    order = np.argsort(depth)[::-1]
    xi = xi[order]; yi = yi[order]; tier_labels = tier_labels[order]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.float32)
    offsets = [(0, 0)] if radius_px <= 0 else [(dx, dy) for dy in range(-radius_px, radius_px+1) for dx in range(-radius_px, radius_px+1) if dx*dx+dy*dy <= radius_px*radius_px]
    for dx, dy in offsets:
        xx = xi + dx; yy = yi + dy
        m = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        tier = tier_labels[m]
        rgb = np.array([tier_colors.get(int(t), (128, 128, 128)) for t in tier], dtype=np.uint8)
        a = np.full(m.sum(), 0.55, dtype=np.float32)
        yy2 = yy[m]; xx2 = xx[m]
        old_a = alpha[yy2, xx2]
        new_a = a + old_a * (1 - a)
        old_rgb = canvas[yy2, xx2].astype(np.float32)
        canvas[yy2, xx2] = np.clip(
            (rgb.astype(np.float32) * a[:, None] + old_rgb * old_a[:, None] * (1 - a[:, None])) / np.maximum(new_a[:, None], 1e-6),
            0, 255
        ).astype(np.uint8)
        alpha[yy2, xx2] = new_a
    return Image.fromarray(canvas, "RGB")


def compute_error_heatmap(img_a: np.ndarray, img_b: np.ndarray, colormap: str = "viridis") -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel absolute difference and return heatmap (grayscale) and colored version."""
    diff = np.abs(img_a.astype(np.float32) - img_b.astype(np.float32))
    gray_diff = np.mean(diff, axis=2)  # Average across RGB

    # Normalize to 0-1 for colormap
    max_diff = np.max(gray_diff)
    if max_diff > 0:
        normalized = gray_diff / max_diff
    else:
        normalized = gray_diff

    # Apply simple heatmap: blue (low) -> red (high)
    heatmap = np.zeros((*gray_diff.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red channel
    heatmap[:, :, 2] = ((1 - normalized) * 255).astype(np.uint8)  # Blue channel

    return gray_diff, heatmap


def add_label(img: Image.Image, label: str) -> Image.Image:
    pad = 44
    out = Image.new("RGB", (img.width, img.height + pad), (18, 18, 18))
    out.paste(img, (0, pad))
    d = ImageDraw.Draw(out)
    d.text((12, 12), label, fill=(245, 245, 245))
    return out


def add_colorbar(img: np.ndarray, max_diff: float, width: int = 30, height: int = 200) -> Image.Image:
    """Add a colorbar legend to the bottom of a heatmap image."""
    img_height, img_width = img.shape[:2]
    bar_h = 20

    # Create colorbar
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = 1.0 - (y / height)  # Top = max, bottom = min
        colorbar[y, :, 0] = int(val * 255)  # Red
        colorbar[y, :, 2] = int((1 - val) * 255)  # Blue

    # Extend image vertically to fit colorbar + labels
    extended = np.zeros((img_height + height + 50, img_width, 3), dtype=np.uint8)
    extended[:img_height, :] = img
    extended[img_height:img_height+height, img_width-width-10:img_width-10] = colorbar

    # Create PIL image and add text labels
    out = Image.fromarray(extended.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(out)
    y_max = img_height
    y_min = img_height + height
    d.text((img_width - width - 5, y_max - 10), f"{max_diff:.1f}", fill=(255, 255, 255))
    d.text((img_width - width - 5, y_min + 5), "0", fill=(255, 255, 255))

    return out


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, PSNR, and SSIM."""
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0

    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))

    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(-10.0 * math.log10(mse))

    ssim_val = None
    if SSIM_AVAILABLE:
        try:
            ssim_val = float(ssim(a, b, channel_axis=2, data_range=1.0))
        except Exception as e:
            print(f"SSIM computation failed: {e}")

    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim_val}


def check_attribute_parity(v25_path: Path, v28_path: Path) -> Dict[str, Dict]:
    """Check byte-by-byte parity of passthrough chunks from v25 to v28."""
    _, _, _, _, comp25, dec25, _ = read_container(v25_path)
    _, _, _, _, comp28, dec28, _ = read_container(v28_path)

    passthrough_chunks = [
        "tier_labels_u8",
        "xyz_u24_fixed",
        "dc_rgb_opacity_u8",
        "scale_f16",
        "quat_i16_norm4",
    ]

    parity = {}
    for chunk_name in passthrough_chunks:
        try:
            raw25 = dec25(chunk_name)
            raw28 = dec28(chunk_name)
            equal = (raw25 == raw28)
            diff_bytes = 0 if equal else int(np.sum(np.frombuffer(raw25, dtype=np.uint8) != np.frombuffer(raw28, dtype=np.uint8)))
            parity[chunk_name] = {
                "equal": bool(equal),
                "differing_bytes": diff_bytes,
            }
        except Exception as e:
            parity[chunk_name] = {
                "equal": False,
                "differing_bytes": -1,
                "error": str(e),
            }

    return parity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/inputs/audi/Audi A5 Sportback.zip"))
    ap.add_argument("--v28", type=Path, default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/v28_sh_vq_render_container.3dphox"))
    ap.add_argument("--v25", type=Path, default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/v25_attribute_group_render_container.3dphox"))
    ap.add_argument("--out", type=Path, default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/v30_truth_gate"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--yaw", type=float, default=35)
    ap.add_argument("--pitch", type=float, default=18)
    ap.add_argument("--distance", type=float, default=2.4)
    ap.add_argument("--fov", type=float, default=42)
    args = ap.parse_args()

    # Create output directories
    renders_dir = args.out / "renders"
    reports_dir = args.out / "reports"
    renders_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original PLY...")
    t0_orig = time.perf_counter()
    original = load_original_ply_or_zip(args.original)
    t_orig_load = time.perf_counter() - t0_orig

    print("Loading and decoding v28 container...")
    t0_v28 = time.perf_counter()
    v28, decode_timing = load_crypsoid_v28(args.v28)
    t_v28_load = time.perf_counter() - t0_v28

    print("Rendering original...")
    t0_render_orig = time.perf_counter()
    img_orig = render_points(original, size=args.size, max_points=args.max_points, yaw=args.yaw, pitch=args.pitch, distance=args.distance, fov=args.fov)
    t_render_orig = time.perf_counter() - t0_render_orig

    print("Rendering v28...")
    t0_render_v28 = time.perf_counter()
    img_v28 = render_points(v28, size=args.size, max_points=args.max_points, yaw=args.yaw, pitch=args.pitch, distance=args.distance, fov=args.fov)
    t_render_v28 = time.perf_counter() - t0_render_v28

    print("Rendering tier view...")
    t0_render_tier = time.perf_counter()
    tier_colors = {0: (255, 64, 64), 1: (64, 255, 64), 2: (64, 64, 255)}
    img_tier = render_tier_view(v28, size=args.size, max_points=args.max_points, yaw=args.yaw, pitch=args.pitch, distance=args.distance, fov=args.fov, tier_colors=tier_colors)
    t_render_tier = time.perf_counter() - t0_render_tier

    print("Computing error heatmap...")
    orig_arr = np.asarray(img_orig)
    v28_arr = np.asarray(img_v28)
    gray_diff, heatmap_colored = compute_error_heatmap(orig_arr, v28_arr)
    max_diff = float(np.max(gray_diff))

    print("Computing image metrics...")
    metrics = compute_metrics(orig_arr, v28_arr)

    print("Counting tier assignments...")
    tier_labels = v28["tier_labels"]
    tier_counts = {0: int((tier_labels == 0).sum()), 1: int((tier_labels == 1).sum()), 2: int((tier_labels == 2).sum())}

    print("Checking attribute parity...")
    parity = check_attribute_parity(args.v25, args.v28)

    # Save individual renders
    img_orig.save(renders_dir / "original_ply_dc_opacity.png")
    img_v28.save(renders_dir / "v28_dc_opacity.png")
    img_tier.save(renders_dir / "v30_tier_view.png")

    # Save heatmap with colorbar
    heatmap_img = Image.fromarray(heatmap_colored.astype(np.uint8), "RGB")
    heatmap_with_bar = add_colorbar(heatmap_colored, max_diff)
    heatmap_with_bar.save(renders_dir / "v30_error_heatmap.png")

    # Create contact sheet with 4 labeled panels
    panel_width = args.size
    panel_height = args.size + 44
    contact_width = panel_width * 2
    contact_height = panel_height * 2
    contact = Image.new("RGB", (contact_width, contact_height), (0, 0, 0))

    # Top-left: original
    contact.paste(add_label(img_orig, "Original PLY DC/opacity preview"), (0, 0))
    # Top-right: v28
    contact.paste(add_label(img_v28, "v28 decoded DC/opacity preview"), (panel_width, 0))
    # Bottom-left: heatmap
    heatmap_labeled = add_label(heatmap_img, f"Per-pixel absolute difference | max: {max_diff:.2f}")
    contact.paste(heatmap_labeled, (0, panel_height))
    # Bottom-right: tier view
    contact.paste(add_label(img_tier, "Per-splat tier visualization (R=A, G=B, B=C)"), (panel_width, panel_height))

    contact.save(renders_dir / "v30_contact_sheet.png")

    # Build timing report
    total_decode = sum(decode_timing.values())
    total_render = t_render_orig + t_render_v28 + t_render_tier + (time.perf_counter() - t0_render_orig - t_render_orig - t_render_v28 - t_render_tier)

    # Build metrics report
    report = {
        "image_metrics": metrics,
        "decode_times_seconds": decode_timing,
        "render_times_seconds": {
            "render_original_s": t_render_orig,
            "render_v28_s": t_render_v28,
            "render_tier_view_s": t_render_tier,
        },
        "attribute_parity": parity,
        "tier_counts": tier_counts,
        "tier_color_legend": {
            "0_tier_a_native_render": "#FF4040",
            "1_tier_b_native_render": "#40FF40",
            "2_tier_c_native_render": "#4040FF",
        },
        "truth_note": (
            "This is a CPU DC/opacity point preview renderer. It uses only the diffuse (DC) color "
            "and opacity channels; SH bands 1-3 are not exercised. It draws screen-space dots, not anisotropic Gaussians. "
            "It is meant to catch gross geometry/color errors before the full viewer path. "
            "It is NOT final visual truth. SSIM computed with skimage's structural_similarity (window size 11 by default) on 0-1 normalized RGB."
        ),
        "camera": {
            "yaw": args.yaw,
            "pitch": args.pitch,
            "distance": args.distance,
            "fov": args.fov,
        },
        "render_settings": {
            "size": args.size,
            "max_points": args.max_points,
        },
        "counts": {
            "original": int(original["count"]),
            "v28": int(v28["count"]),
        },
        "outputs": {
            "contact_sheet": str(renders_dir / "v30_contact_sheet.png"),
            "error_heatmap": str(renders_dir / "v30_error_heatmap.png"),
            "tier_view": str(renders_dir / "v30_tier_view.png"),
        },
    }

    # Write JSON report
    report_json_path = reports_dir / "v30_truth_gate.json"
    report_json_path.write_text(json.dumps(report, indent=2))

    # Write Markdown report
    ssim_str = f"{report['image_metrics']['ssim']:.4f}" if report["image_metrics"]["ssim"] is not None else "N/A"
    md_report = f"""# CRYPSOID v0.30 Render Truth Gate

**Generated:** {Path(renders_dir).parent.name}

## Summary

This gate compares the original Audi PLY against the v0.28 decoded container using a CPU DC/opacity point preview renderer.
All panels use the same camera perspective (yaw {args.yaw}°, pitch {args.pitch}°, distance {args.distance}, FOV {args.fov}°).

## Image Metrics (v28 vs original)

| Metric | Value |
|---|---|
| MSE | {report["image_metrics"]["mse"]:.6f} |
| MAE | {report["image_metrics"]["mae"]:.6f} |
| PSNR (dB) | {report["image_metrics"]["psnr_db"]:.2f} |
| SSIM | {ssim_str} |

## Decoding Timing (seconds)

| Step | Time |
|---|---|
| Read container | {decode_timing.get("read_container_s", 0):.4f} |
| Decompress chunks | {decode_timing.get("decompress_chunks_s", 0):.4f} |
| Decode XYZ (u24) | {decode_timing.get("decode_xyz_u24_s", 0):.4f} |
| Decode DC/RGB/opacity (u8) | {decode_timing.get("decode_dc_rgb_opacity_u8_s", 0):.4f} |
| Decode tier labels (u8) | {decode_timing.get("decode_tier_labels_u8_s", 0):.4f} |
| **Total** | **{total_decode:.4f}** |

## Rendering Timing (seconds)

| Task | Time |
|---|---|
| Render original | {t_render_orig:.4f} |
| Render v28 | {t_render_v28:.4f} |
| Render tier view | {t_render_tier:.4f} |

## Tier Distribution

v28 splat assignments by tier (from tier_labels_u8 chunk):

| Tier | Count | Percent |
|---|---:|---|
| A (native) | {tier_counts[0]:,} | {100.0*tier_counts[0]/(tier_counts[0]+tier_counts[1]+tier_counts[2]):.1f}% |
| B (native) | {tier_counts[1]:,} | {100.0*tier_counts[1]/(tier_counts[0]+tier_counts[1]+tier_counts[2]):.1f}% |
| C (native) | {tier_counts[2]:,} | {100.0*tier_counts[2]/(tier_counts[0]+tier_counts[1]+tier_counts[2]):.1f}% |

## Attribute Parity (v25 → v28 passthrough)

v0.28 passes these five chunks through unchanged from v25. All must be byte-identical.

| Chunk | Byte-identical | Differing bytes |
|---|---|---|
| tier_labels_u8 | {str(parity.get("tier_labels_u8", {}).get("equal", False))} | {parity.get("tier_labels_u8", {}).get("differing_bytes", -1)} |
| xyz_u24_fixed | {str(parity.get("xyz_u24_fixed", {}).get("equal", False))} | {parity.get("xyz_u24_fixed", {}).get("differing_bytes", -1)} |
| dc_rgb_opacity_u8 | {str(parity.get("dc_rgb_opacity_u8", {}).get("equal", False))} | {parity.get("dc_rgb_opacity_u8", {}).get("differing_bytes", -1)} |
| scale_f16 | {str(parity.get("scale_f16", {}).get("equal", False))} | {parity.get("scale_f16", {}).get("differing_bytes", -1)} |
| quat_i16_norm4 | {str(parity.get("quat_i16_norm4", {}).get("equal", False))} | {parity.get("quat_i16_norm4", {}).get("differing_bytes", -1)} |

## Visual Comparison

![Contact sheet](../renders/v30_contact_sheet.png)

**Top-left:** Original PLY rendered as DC + opacity dots.
**Top-right:** v0.28 decoded container rendered the same way.
**Bottom-left:** Per-pixel absolute-difference heatmap (blue = 0, red = {max_diff:.2f}).
**Bottom-right:** Per-splat tier visualization (Red=Tier A, Green=Tier B, Blue=Tier C).

## Truth Note

{report["truth_note"]}

---

**Files:**
- Contact sheet: `renders/v30_contact_sheet.png`
- Error heatmap: `renders/v30_error_heatmap.png`
- Tier view: `renders/v30_tier_view.png`
- Full metrics: `reports/v30_truth_gate.json`
"""

    md_report_path = reports_dir / "v30_truth_gate.md"
    md_report_path.write_text(md_report)

    print(f"\nReport written to {report_json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
