"""Load standard 3DGS .ply files into SplatBuffer."""

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .splat_buffer import SplatBuffer


def parse_ply_header(f) -> Tuple[dict, int]:
    """Parse PLY header and return metadata + header byte length."""
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


def load_ply_or_zip(path: Path, member: str = "scene.ply") -> np.ndarray:
    """Load a .ply or .zip containing a .ply. Return raw float32 array (n, num_props)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            chosen = member if member in z.namelist() else next(
                (n for n in z.namelist() if n.lower().endswith(".ply")), None
            )
            if chosen is None:
                raise ValueError(f"No .ply file found in {path}")
            data = z.read(chosen)
        f = io.BytesIO(data)
    else:
        f = path.open("rb")

    with f:
        meta, header_len = parse_ply_header(f)
        n = meta["vertex_count"]
        props = meta["properties"]
        raw = f.read(n * len(props) * 4)
        if len(raw) != n * len(props) * 4:
            raise ValueError(f"Expected {n * len(props) * 4} bytes, got {len(raw)}")
        arr = np.frombuffer(raw, dtype="<f4").reshape(n, len(props))
    return arr, meta["properties"]


def load_ply(path: Path) -> SplatBuffer:
    """Load a standard 3DGS .ply file into a SplatBuffer.

    Handles both .ply and .zip-wrapped .ply.
    Assumes standard 3DGS format with float32 properties.
    """
    arr, props = load_ply_or_zip(path)
    n = arr.shape[0]

    # Build column index
    col = {name: i for i, name in enumerate(props)}

    # Extract required fields
    xyz = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float32)
    scales = arr[:, [col["scale_0"], col["scale_1"], col["scale_2"]]].astype(np.float32)

    # Quaternion: rot_0, rot_1, rot_2, rot_3 in wxyz order
    quats = arr[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]].astype(np.float32)

    # Opacity and DC color (unify with v28 loader's representation)
    SH_C0 = 0.28209479177387814
    opacities = 1.0 / (1.0 + np.exp(-arr[:, col["opacity"]].astype(np.float32)))
    f_dc = arr[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]].astype(np.float32)
    sh_dc = (SH_C0 * f_dc + 0.5).astype(np.float32)

    # SH rest (degrees 1-3): 16 * 3 = 48 coefficients per splat
    sh_rest_cols = [col[f"f_rest_{i}"] for i in range(48) if f"f_rest_{i}" in col]
    if sh_rest_cols:
        sh_rest = arr[:, sh_rest_cols].astype(np.float32)
        if sh_rest.shape[1] < 45:
            # Pad with zeros if less than 45 coefficients
            padded = np.zeros((n, 45), dtype=np.float32)
            padded[:, :sh_rest.shape[1]] = sh_rest
            sh_rest = padded
    else:
        sh_rest = None

    # Standard PLY has no tier labels, so all are Tier C (Gaussian fallback)
    tier = np.full((n,), 2, dtype=np.uint8)

    return SplatBuffer(
        n=n,
        xyz=xyz,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        tier=tier,
        germ=None,
        correction=None,
        source=str(path),
        scene_format="ply",
    )
