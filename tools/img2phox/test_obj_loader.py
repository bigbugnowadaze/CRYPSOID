"""F.30.2 — Acceptance Gates 1+2 for the OBJ loader + sampler.

Gate 1: parse a procedural cube + sphere OBJ + MTL without error.
Gate 2: round-trip the .3dphox (build → load) and check splat count
        matches face-area-weighted target ±5%, plus PBR fields propagated.
"""
from __future__ import annotations
import sys, math, tempfile, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from img2phox.obj_loader import (
    parse_obj, parse_mtl, triangulate_face, triangle_area, triangle_normal,
)


def write_test_fixture(dirpath: Path) -> Path:
    """Procedural OBJ + MTL: a unit cube (red diffuse) + an icosahedron (chrome)."""
    obj = dirpath / 'fixture.obj'
    mtl = dirpath / 'fixture.mtl'

    # MTL
    mtl.write_text(
        'newmtl matte_red\n'
        'Kd 0.85 0.10 0.10\n'
        'Pm 0.05\n'
        'Pr 0.55\n'
        'd 1.0\n'
        '\n'
        'newmtl chrome\n'
        'Kd 0.92 0.93 0.95\n'
        'Pm 0.92\n'
        'Pr 0.10\n'
        'd 1.0\n'
    )

    # Cube vertices (centred at origin, side 1)
    cube = [
        (-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5),
        ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5),
        (-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5),
    ]
    # Cube quads (CCW from outside)
    cube_quads = [
        (1, 2, 3, 4),    # -Z
        (5, 8, 7, 6),    # +Z
        (1, 5, 6, 2),    # -Y
        (4, 3, 7, 8),    # +Y
        (1, 4, 8, 5),    # -X
        (2, 6, 7, 3),    # +X
    ]
    # Icosahedron at (1.5, 0, 0), radius 0.4
    phi = (1 + math.sqrt(5)) / 2
    icoR = 0.4
    icoC = (1.5, 0.0, 0.0)
    ico_v = [
        (-1,  phi,  0), ( 1,  phi,  0), (-1, -phi,  0), ( 1, -phi,  0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi,  0, -1), ( phi,  0,  1), (-phi,  0, -1), (-phi,  0,  1),
    ]
    norm = math.sqrt(1 + phi * phi)
    ico_v = [(x / norm * icoR + icoC[0],
               y / norm * icoR + icoC[1],
               z / norm * icoR + icoC[2]) for (x, y, z) in ico_v]
    # 20 icosahedron faces (1-based after the 8 cube verts)
    base = 8
    ico_f = [
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2),
    ]
    ico_f = [(a + base, b + base, c + base) for (a, b, c) in ico_f]

    lines = ['mtllib fixture.mtl', '']
    for v in cube:
        lines.append(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
    for v in ico_v:
        lines.append(f'v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
    lines.append('')
    lines.append('usemtl matte_red')
    for q in cube_quads:
        lines.append('f ' + ' '.join(str(i) for i in q))
    lines.append('')
    lines.append('usemtl chrome')
    for tri in ico_f:
        lines.append('f ' + ' '.join(str(i) for i in tri))
    obj.write_text('\n'.join(lines) + '\n')
    return obj


def gate_1(obj_path: Path) -> bool:
    print('GATE 1 — parser sanity:')
    scene = parse_obj(obj_path)
    print(f'  {len(scene.vertices)} vertices, {len(scene.faces)} faces, '
          f'{len(scene.materials)} materials')
    expected_v = 8 + 12       # 8 cube + 12 icosahedron
    expected_f = 6 + 20       # 6 cube quads + 20 ico tris
    expected_m = 2 + 1        # matte_red + chrome + __default__
    ok = (len(scene.vertices) == expected_v and
          len(scene.faces) == expected_f and
          len(scene.materials) == expected_m)
    if 'matte_red' in scene.materials:
        m = scene.materials['matte_red']
        print(f'  matte_red: albedo={m.albedo}  metallic={m.metallic:.2f}  '
              f'roughness={m.roughness:.2f}  hint={m.hint}')
        ok = ok and m.metallic < 0.2 and abs(m.albedo[0] - 0.85) < 0.01
    if 'chrome' in scene.materials:
        m = scene.materials['chrome']
        print(f'  chrome:    albedo={m.albedo}  metallic={m.metallic:.2f}  '
              f'roughness={m.roughness:.2f}  hint={m.hint}')
        ok = ok and m.metallic > 0.6 and m.hint in ('mirror', 'glossy')
    print(f'  GATE 1: {"PASS" if ok else "FAIL"}')
    return ok


def gate_2(obj_path: Path) -> bool:
    print('\nGATE 2 — face-area sampling round-trip:')
    from build_blender_phox import sample_obj_scene
    samp = sample_obj_scene(obj_path, n_total=50_000, seed=42)
    n_actual = samp['xyz'].shape[0]
    print(f'  sampled {n_actual:,} splats (target 50,000)')
    err = abs(n_actual - 50_000) / 50_000
    print(f'  count error = {err*100:.2f}%')

    # Per-material histogram
    mat_names = samp['mat_names']
    for mi, name in enumerate(mat_names):
        cnt = int((samp['mat_id'] == mi).sum())
        if cnt == 0:
            continue
        alb = samp['albedo'][samp['mat_id'] == mi][0]
        print(f'  material "{name}": {cnt:,} splats, '
              f'albedo[0]={alb}  metallic[0]={samp["metallic"][samp["mat_id"]==mi][0]:.2f}')

    ok_count = err < 0.05    # within 5%
    # The cube has area 6 * 1 = 6; icosahedron has area ~ 20 * sqrt(3)/4 * (edge)^2
    # with edge = icoR * 2/phi ≈ 0.494 → area ≈ 20 * 0.433 * 0.244 ≈ 2.1
    # So cube should get ~6/(6+2.1) ≈ 74% of splats.
    # Detect red by metallic level (chrome is metallic 0.92, red is 0.05)
    n_red    = int((samp['metallic'] < 0.5).sum())
    n_chrome = n_actual - n_red
    frac_red = n_red / n_actual
    print(f'  split: red {n_red:,} ({frac_red*100:.1f}%) / '
          f'chrome {n_chrome:,} ({100 - frac_red*100:.1f}%)')
    ok_split = 0.70 < frac_red < 0.90    # ~80% expected, allow ±10%
    ok = ok_count and ok_split
    print(f'  GATE 2: {"PASS" if ok else "FAIL"}  '
          f'(count_within_5pct={ok_count}  area_split_correct={ok_split})')
    return ok


def main():
    print('=' * 70)
    print('F.30.2 — OBJ loader + sampler acceptance gates')
    print('=' * 70)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        obj_path = write_test_fixture(td_p)
        print(f'  fixture at {obj_path}\n')
        ok1 = gate_1(obj_path)
        ok2 = gate_2(obj_path)
    print()
    print('=' * 70)
    if ok1 and ok2:
        print('ALL GATES PASS')
        return 0
    else:
        print(f'FAIL — gate1={ok1}  gate2={ok2}')
     