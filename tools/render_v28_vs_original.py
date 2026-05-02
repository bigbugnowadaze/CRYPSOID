#!/usr/bin/env python3
"""
CRYPSOID v0.28 vs original Gaussian-splat PLY quick render harness.

Purpose:
- Render the original Audi PLY/ZIP and the CRYPSOID v0.28 .3dphox container from the same camera.
- Produce a side-by-side contact sheet and simple image metrics.

Truth note:
This is a CPU DC/opacity preview renderer, not a full CUDA/PlayCanvas/gsplat renderer.
It is meant to catch gross geometry/color parity errors before the real viewer path.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
        # CRYPSOID Audi source used 59 float32 properties = 236 bytes/record.
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
    with path.open("rb") as f:
        magic = f.read(11)
        manifest_len = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(manifest_len))
        blob = f.read()
    chunks = {c["name"]: c for c in manifest["chunks"]}

    def comp(name: str) -> bytes:
        c = chunks[name]
        return blob[c["offset"]:c["offset"] + c["compressed_bytes"]]

    def dec(name: str) -> bytes:
        return zlib.decompress(comp(name))

    return magic, manifest, blob, chunks, comp, dec


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


def load_crypsoid_v28(path: Path) -> Dict[str, np.ndarray]:
    magic, man, blob, chunks, comp, dec = read_container(path)
    # Source splat count may live either in manifest root or chunk shape.
    xyz_chunk = chunks["xyz_u24_fixed"]
    n = int(xyz_chunk["shape"][0])
    xyz = decode_u24_xyz(dec("xyz_u24_fixed"), n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])
    dc = np.frombuffer(dec("dc_rgb_opacity_u8"), dtype=np.uint8).reshape(n, 4)
    rgb = dc[:, :3].astype(np.float32) / 255.0
    opacity = dc[:, 3].astype(np.float32) / 255.0
    return {"xyz": xyz, "rgb": rgb, "opacity": opacity, "count": n, "source": str(path), "magic": magic.decode(errors="replace")}


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
        # deterministic spatially broad sample
        rng = np.random.default_rng(28029)
        idx = rng.choice(n, size=max_points, replace=False)
        xyz = xyz[idx]; rgb = rgb[idx]; opacity = opacity[idx]
    px, py, depth, keep = camera_project(xyz, yaw, pitch, distance, fov, size)
    rgb = rgb[keep]
    opacity = opacity[keep]
    xi = np.rint(px).astype(np.int32)
    yi = np.rint(py).astype(np.int32)
    inside = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
    xi = xi[inside]; yi = yi[inside]; depth = depth[inside]; rgb = rgb[inside]; opacity = opacity[inside]
    # far to near alpha blend
    order = np.argsort(depth)[::-1]
    xi = xi[order]; yi = yi[order]; rgb = rgb[order]; opacity = opacity[order]
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    alpha = np.zeros((size, size), dtype=np.float32)
    # point stamp offsets
    offsets = [(0, 0)] if radius_px <= 0 else [(dx, dy) for dy in range(-radius_px, radius_px+1) for dx in range(-radius_px, radius_px+1) if dx*dx+dy*dy <= radius_px*radius_px]
    for dx, dy in offsets:
        xx = xi + dx; yy = yi + dy
        m = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        a = np.clip(opacity[m] * 0.55, 0, 1)
        yy2 = yy[m]; xx2 = xx[m]
        old_a = alpha[yy2, xx2]
        new_a = a + old_a * (1 - a)
        # premult-ish blend
        canvas[yy2, xx2] = (rgb[m] * a[:, None] + canvas[yy2, xx2] * old_a[:, None] * (1 - a[:, None])) / np.maximum(new_a[:, None], 1e-6)
        alpha[yy2, xx2] = new_a
    out = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def add_label(img: Image.Image, label: str) -> Image.Image:
    pad = 44
    out = Image.new("RGB", (img.width, img.height + pad), (18, 18, 18))
    out.paste(img, (0, pad))
    d = ImageDraw.Draw(out)
    d.text((12, 12), label, fill=(245, 245, 245))
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, default=Path("/mnt/data/Audi A5 Sportback.zip"))
    ap.add_argument("--v28", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_28/outputs/v28_sh_vq_render_container.3dphox"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/data/crypsoid_v28_render_check"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max-points", type=int, default=0, help="0 = all visible points; set 100000 for faster preview")
    ap.add_argument("--yaw", type=float, default=35)
    ap.add_argument("--pitch", type=float, default=18)
    ap.add_argument("--distance", type=float, default=2.4)
    ap.add_argument("--fov", type=float, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    original = load_original_ply_or_zip(args.original)
    crypsoid = load_crypsoid_v28(args.v28)
    img_a = render_points(original, size=args.size, max_points=args.max_points, yaw=args.yaw, pitch=args.pitch, distance=args.distance, fov=args.fov)
    img_b = render_points(crypsoid, size=args.size, max_points=args.max_points, yaw=args.yaw, pitch=args.pitch, distance=args.distance, fov=args.fov)
    p_a = args.out / "original_ply_dc_opacity_preview.png"
    p_b = args.out / "crypsoid_v28_dc_opacity_preview.png"
    img_a.save(p_a)
    img_b.save(p_b)
    sheet = Image.new("RGB", (args.size*2, args.size + 44), (0, 0, 0))
    sheet.paste(add_label(img_a, f"Original PLY/ZIP DC opacity preview | n={original['count']:,}"), (0, 0))
    sheet.paste(add_label(img_b, f"CRYPSOID v0.28 decoded container DC opacity preview | n={crypsoid['count']:,}"), (args.size, 0))
    p_sheet = args.out / "v28_vs_original_side_by_side.png"
    sheet.save(p_sheet)
    metrics = {
        "renderer": "CPU DC/opacity point preview; not full anisotropic SH rasterizer",
        "original": str(args.original),
        "v28": str(args.v28),
        "size": args.size,
        "max_points": args.max_points,
        "camera": {"yaw": args.yaw, "pitch": args.pitch, "distance": args.distance, "fov": args.fov},
        "counts": {"original": int(original["count"]), "v28": int(crypsoid["count"])},
        "render_psnr_db_same_preview": psnr(np.asarray(img_a), np.asarray(img_b)),
        "outputs": {"original": str(p_a), "v28": str(p_b), "side_by_side": str(p_sheet)},
    }
    (args.out / "render_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
