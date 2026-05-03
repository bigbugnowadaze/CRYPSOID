"""F.30.2 — Wavefront OBJ + MTL parser.

Pure Python, no third-party dependency. Subset per docs/blender_bridge_spec.md
section 2 (OBJ subset + MTL subset).

Output dataclasses are intentionally simple:

    Material:  name, albedo, metallic, roughness, opacity, emissive, hint
    OBJScene:  vertices(N,3), normals(N,3) or None, faces (list of Face)
    Face:      indices (list of int), material_name (or None for default)

The face indices are 0-based vertex indices (the OBJ format uses 1-based;
the parser converts). UV / normal indices are parsed but currently dropped
since v1 doesn't use them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import math


# ---------- Material ----------

DEFAULT_MAT_NAME = '__default__'


@dataclass
class Material:
    name: str = DEFAULT_MAT_NAME
    albedo: Tuple[float, float, float]   = (0.70, 0.70, 0.70)
    metallic: float                       = 0.05
    roughness: float                      = 0.60
    opacity: float                        = 1.00
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hint: str                             = 'diffuse'   # 'diffuse', 'glossy',
                                                          # 'mirror', 'emissive'


def parse_mtl(path: Path) -> Dict[str, Material]:
    """Parse a .mtl file. Returns {material_name: Material}."""
    mats: Dict[str, Material] = {}
    cur: Optional[Material] = None
    have_pm = False
    have_pr = False
    Ks_brightness = 0.0
    illum = 2

    def flush(m, have_pm, have_pr, Ks_brightness, illum):
        if m is None: return
        # Heuristics if PBR fields absent
        if not have_pm:
            base_metal = min(0.95, max(0.0, Ks_brightness))
            if illum in (3, 5, 7):
                base_metal = min(0.95, base_metal + 0.4)
            m.metallic = base_metal
        if any(c > 1e-4 for c in m.emissive):
            m.hint = 'emissive'
        elif m.metallic > 0.6:
            m.hint = 'mirror'
        elif m.metallic > 0.2:
            m.hint = 'glossy'
        else:
            m.hint = 'diffuse'

    with path.open('r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tag = parts[0].lower()

            if tag == 'newmtl':
                # finalise previous
                flush(cur, have_pm, have_pr, Ks_brightness, illum)
                if cur is not None:
                    mats[cur.name] = cur
                cur = Material(name=parts[1])
                have_pm = False
                have_pr = False
                Ks_brightness = 0.0
                illum = 2
            elif cur is None:
                continue
            elif tag == 'kd' and len(parts) >= 4:
                cur.albedo = tuple(float(p) for p in parts[1:4])
            elif tag == 'ke' and len(parts) >= 4:
                cur.emissive = tuple(float(p) for p in parts[1:4])
            elif tag == 'ks' and len(parts) >= 4:
                Ks_brightness = sum(float(p) for p in parts[1:4]) / 3.0
            elif tag == 'pm' and len(parts) >= 2:
                cur.metallic = max(0.0, min(0.95, float(parts[1])))
                have_pm = True
            elif tag == 'pr' and len(parts) >= 2:
                cur.roughness = max(0.05, min(0.95, float(parts[1])))
                have_pr = True
            elif tag == 'ns' and len(parts) >= 2 and not have_pr:
                Ns = max(1.0, float(parts[1]))
                cur.roughness = max(0.05, min(0.95,
                    1.0 - max(0.0, min(0.95, math.log10(Ns) / 3.0))))
            elif tag == 'd' and len(parts) >= 2:
                cur.opacity = max(0.0, min(1.0, float(parts[1])))
            elif tag == 'tr' and len(parts) >= 2:
                cur.opacity = max(0.0, min(1.0, 1.0 - float(parts[1])))
            elif tag == 'illum' and len(parts) >= 2:
                try:
                    illum = int(parts[1])
                except ValueError:
                    pass

    flush(cur, have_pm, have_pr, Ks_brightness, illum)
    if cur is not None:
        mats[cur.name] = cur
    return mats


# ---------- OBJ ----------

@dataclass
class Face:
    indices: List[int]                    # 0-based vertex indices
    material_name: Optional[str] = None    # None → default material


@dataclass
class OBJScene:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    normals:  List[Tuple[float, float, float]] = field(default_factory=list)
    faces:    List[Face] = field(default_factory=list)
    materials: Dict[str, Material] = field(default_factory=dict)
    mtl_paths: List[Path] = field(default_factory=list)

    def __len__(self): return len(self.faces)


def parse_obj(path: Path) -> OBJScene:
    """Parse an .obj file. mtllib references are resolved relative to the OBJ."""
    path = Path(path)
    scene = OBJScene()
    cur_mat: Optional[str] = None

    with path.open('r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tag = parts[0].lower()

            if tag == 'v' and len(parts) >= 4:
                scene.vertices.append((float(parts[1]), float(parts[2]),
                                          float(parts[3])))
            elif tag == 'vn' and len(parts) >= 4:
                scene.normals.append((float(parts[1]), float(parts[2]),
                                         float(parts[3])))
            elif tag == 'vt':
                pass   # UV ignored in v1
            elif tag == 'f' and len(parts) >= 4:
                # Each vertex spec is 'a' or 'a/t' or 'a/t/n' or 'a//n'
                idxs = []
                for p in parts[1:]:
                    v_str = p.split('/')[0]
                    if not v_str:
                        continue
                    v_idx = int(v_str)
                    # OBJ allows negative (relative) indices
                    if v_idx < 0:
                        v_idx = len(scene.vertices) + 1 + v_idx
                    idxs.append(v_idx - 1)        # 1-based → 0-based
                if len(idxs) >= 3:
                    scene.faces.append(Face(indices=idxs, material_name=cur_mat))
            elif tag == 'usemtl' and len(parts) >= 2:
                cur_mat = parts[1]
            elif tag == 'mtllib' and len(parts) >= 2:
                mtl_name = ' '.join(parts[1:])
                mtl_path = (path.parent / mtl_name).resolve()
                if mtl_path.exists():
                    scene.materials.update(parse_mtl(mtl_path))
                    scene.mtl_paths.append(mtl_path)
            elif tag in ('g', 'o', 's'):
                pass    # parsed, ignored
            # Anything else silently ignored.

    # Always provide a default material so faces without usemtl have something
    if DEFAULT_MAT_NAME not in scene.materials:
        scene.materials[DEFAULT_MAT_NAME] = Material()
    return scene


# ---------- Triangulation + sampling helpers ----------

def triangulate_face(indices: List[int]) -> List[Tuple[int, int, int]]:
    """Fan triangulate an n-gon into triangles. n >= 3."""
    return [(indices[0], indices[i], indices[i + 1])
            for i in range(1, len(indices) - 1)]


def triangle_area(v0, v1, v2) -> float:
    """Area of the triangle (v0, v1, v2)."""
    ax, ay, az = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    bx, by, bz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def triangle_normal(v0, v1, v2) -> Tuple[float, float, float]:
    """Right-handed face normal of (v0, v1, v2), normalised."""
    ax, ay, az = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    bx, by, bz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    L = math.sqrt(cx * cx + cy * cy + cz * cz)
    if L < 1e-12:
        return (0.0, 1.0, 0.0)
    return (cx / L, cy / L, cz / L)
