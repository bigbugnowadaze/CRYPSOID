"""PhoxBench Tier 1 — load real meshes / point clouds as scenes.

Handles:
  - Stanford ASCII PLY (cyberware scans: Bunny, Dragon, Armadillo, Happy Buddha)
  - Stanford binary PLY (3DGS-style float32)
  - Anything with `element vertex N` + `property float [xyz]` triple

Returns the same `Scene` dataclass as `scenes.py` so `run_scene.py` can
operate on it unchanged.
"""

from __future__ import annotations
import io, sys, zipfile
from pathlib import Path
import numpy as np

from .scenes import Scene, _color_by_height


def _parse_ply_header_text(text):
    """Parse PLY header from a text blob. Returns (format, vertex_count, properties_per_vertex_in_order, header_byte_length)."""
    lines = []
    for line in text.split("\n"):
        lines.append(line)
        if line.strip() == "end_header":
            break
    fmt = None
    vertex_count = None
    props = []
    in_vertex_section = False
    for line in lines:
        s = line.strip()
        if s.startswith("format "):
            fmt = s.split()[1]
        elif s.startswith("element "):
            parts = s.split()
            in_vertex_section = parts[1] == "vertex"
            if in_vertex_section:
                vertex_count = int(parts[2])
            else:
                in_vertex_section = False
        elif in_vertex_section and s.startswith("property "):
            props.append(s.split())
    header_str = "\n".join(lines) + "\n"
    return fmt, vertex_count, props, len(header_str.encode("utf-8"))


def load_ply_pointcloud(path, max_points=None):
    """Load any PLY (ASCII or binary little-endian) and return (n, 3) float32 xyz."""
    path = Path(path)
    raw = path.read_bytes()
    end_marker = b"end_header\n"
    end = raw.find(end_marker)
    if end < 0:
        raise ValueError(f"{path}: no 'end_header' marker")
    header = raw[:end + len(end_marker)].decode("utf-8", errors="replace")
    body = raw[end + len(end_marker):]

    fmt, n, props, _ = _parse_ply_header_text(header)
    # Find x/y/z indices
    prop_names = [p[-1] for p in props]
    if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
        raise ValueError(f"{path}: PLY missing x/y/z properties")
    xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")

    if fmt == "ascii":
        # Each vertex is one whitespace-separated line; columns map to props.
        pts = np.empty((n, 3), dtype=np.float32)
        text = body.decode("utf-8", errors="replace")
        line_iter = (ln for ln in text.split("\n") if ln.strip())
        for i in range(n):
            try:
                vals = next(line_iter).split()
            except StopIteration:
                pts = pts[:i]
                break
            try:
                pts[i, 0] = float(vals[xi])
                pts[i, 1] = float(vals[yi])
                pts[i, 2] = float(vals[zi])
            except (IndexError, ValueError):
                pts[i] = 0.0
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if "little" in fmt else ">"
        # Build numpy dtype
        type_map = {
            "char": "i1", "uchar": "u1",
            "short": "i2", "ushort": "u2",
            "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
        }
        fields = []
        for p in props:
            t = p[1]
            if t == "list":
                # variable-length list field; need to manually parse... unsupported here
                raise NotImplementedError(f"{path}: PLY 'property list' not supported in this loader")
            np_t = type_map.get(t, "f4")
            fields.append((p[-1], endian + np_t))
        dt = np.dtype(fields)
        verts = np.frombuffer(body, dtype=dt, count=n)
        pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    else:
        raise ValueError(f"{path}: unknown format {fmt}")

    # Filter NaNs / infinities
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]

    # Optional cap for big meshes
    if max_points and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def normalize_pointcloud(pts):
    """Center + scale so the cloud fits in a unit ball.  Returns (pts_normalized, center, scale)."""
    center = pts.mean(axis=0)
    centered = pts - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-9:
        scale = 1.0
    return (centered / scale).astype(np.float32), center, scale


def make_mesh_scene(name, ply_path, max_points=10000, normalize=True):
    """Wrap a PLY pointcloud as a phoxbench Scene."""
    pts = load_ply_pointcloud(ply_path, max_points=None)
    if normalize:
        pts, _, _ = normalize_pointcloud(pts)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    colors = _color_by_height(pts[:, 1])  # height-color along Y
    return Scene(name, pts, colors,
                 f"PLY mesh {Path(ply_path).name} ({pts.shape[0]} pts after subsample)")


# CLI smoke test
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--max-points", type=int, default=10000)
    args = ap.parse_args()
    pts = load_ply_pointcloud(args.ply)
    print(f"loaded {pts.shape[0]} pts; bbox = {pts.min(0).tolist()} -> {pts.max(0).tolist()}")
