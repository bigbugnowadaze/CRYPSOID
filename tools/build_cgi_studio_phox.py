"""F.26 — Build a CLEAN procedural CGI source .3dphox.

Closes AUDI_HERO_attempt.md's "input data, not pipeline" finding by feeding
the same renderer a hand-built scene with:

    - sharp surfaces (densely sampled spheres, cube, plane, torus)
    - clean per-splat normals (analytic, not estimated from a noisy point cloud)
    - controlled per-splat PBR (chrome / brass / red plastic / blue plastic / matte gray)
    - no scan noise, no floor-slab, no halo

Outputs:
    outputs/cgi_studio_v1.3dphox          — v25 base + v31 trailer (normals, edges,
                                              material_hints) + v40 trailer (kappa, cusp)
    outputs/cgi_studio_v1.pbr.npz         — ground-truth per-splat (albedo, metallic,
                                              roughness, F0, kd) for the renderer
    outputs/cgi_studio_v1.scene.json      — small manifest for documentation

Scene layout (in front of camera, ground at y=0):
    Ground plane     4x4 m   light gray, matte
    Chrome sphere    r=0.40  center, very metallic, low roughness
    Red plastic ball r=0.30  back-right, diffuse red
    Blue plastic box 0.6 cube  back-left, diffuse blue, rotated 30°
    Brass torus      R=0.55, r=0.06  ringed around center sphere base
"""
from __future__ import annotations
import sys, json, struct, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from img2phox.encode import encode_blobbundle_to_3dphox, MAGIC as V25_MAGIC
from img2phox.data_classes import BlobBundle
from crypsorender.io.normals_codec import write_normals_chunk, NORMALS_CHUNK_ID
from crypsorender.io.edges_codec   import write_edges_chunk, derive_knn_edges, EDGES_CHUNK_ID
from crypsorender.io.material_codec import (
    write_material_chunk, MATERIAL_CHUNK_ID, derive_mip_zoom,
    MATERIAL_HINT_DIFFUSE, MATERIAL_HINT_GLOSSY, MATERIAL_HINT_MIRROR,
)
from crypsorender.io.germ_codec import (
    write_kappa_chunk, write_cusp_chunk, KAPPA_CHUNK_ID, CUSP_CHUNK_ID,
)


OUT_PHOX  = ROOT / 'outputs' / 'cgi_studio_v1.3dphox'
OUT_PBR   = ROOT / 'outputs' / 'cgi_studio_v1.pbr.npz'
OUT_SCENE = ROOT / 'outputs' / 'cgi_studio_v1.scene.json'

V31_MAGIC = b'CRYPSOID31\x00'
V40_MAGIC = b'CRYPSOID40\x00'


# ---------- Sampling helpers ----------

def sample_sphere(center, radius, n, rng):
    """Uniform sample on a sphere via golden spiral. Returns (xyz, normals)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    pts = np.stack([x, y, z], axis=1)
    # tiny random jitter so duplicate-x splats don't bunch
    pts += rng.normal(0, 0.002, pts.shape)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    normals = pts.copy()
    pts = pts * radius + np.asarray(center)
    return pts.astype(np.float32), normals.astype(np.float32)


def sample_box(center, size, n, rng, rot_deg=0.0):
    """Sample uniformly on the surface of an axis-aligned box rotated about Y.

    size: (sx, sy, sz). Returns (xyz, normals).
    """
    sx, sy, sz = size
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    p = areas / areas.sum()
    face = rng.choice(6, size=n, p=p)
    pts = np.zeros((n, 3), dtype=np.float64)
    nrm = np.zeros((n, 3), dtype=np.float64)
    u = rng.uniform(-0.5, 0.5, n)
    v = rng.uniform(-0.5, 0.5, n)
    # 0:+x, 1:-x, 2:+y, 3:-y, 4:+z, 5:-z
    for f, (axis, sign) in enumerate(
        [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]
    ):
        m = face == f
        if not m.any(): continue
        if axis == 0:
            pts[m, 0] = sign * sx / 2
            pts[m, 1] = u[m] * sy
            pts[m, 2] = v[m] * sz
            nrm[m] = (sign, 0, 0)
        elif axis == 1:
            pts[m, 0] = u[m] * sx
            pts[m, 1] = sign * sy / 2
            pts[m, 2] = v[m] * sz
            nrm[m] = (0, sign, 0)
        else:
            pts[m, 0] = u[m] * sx
            pts[m, 1] = v[m] * sy
            pts[m, 2] = sign * sz / 2
            nrm[m] = (0, 0, sign)
    if rot_deg != 0.0:
        a = np.deg2rad(rot_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pts = pts @ R.T
        nrm = nrm @ R.T
    pts += np.asarray(center)
    return pts.astype(np.float32), nrm.astype(np.float32)


def sample_torus(center, R, r, n, rng):
    """Sample a torus around the Y axis. R = major, r = minor."""
    u = rng.uniform(0, 2 * np.pi, n)        # around the big ring
    v = rng.uniform(0, 2 * np.pi, n)        # around the tube
    cu, su = np.cos(u), np.sin(u)
    cv, sv = np.cos(v), np.sin(v)
    x = (R + r * cv) * cu
    y = r * sv
    z = (R + r * cv) * su
    pts = np.stack([x, y, z], axis=1) + np.asarray(center)
    # outward normal at (u,v) is (cv*cu, sv, cv*su)
    nrm = np.stack([cv * cu, sv, cv * su], axis=1)
    return pts.astype(np.float32), nrm.astype(np.float32)


def sample_plane(center, sx, sz, n, rng):
    """Sample a horizontal plane (normal +Y), centered at `center`."""
    u = rng.uniform(-sx / 2, sx / 2, n)
    v = rng.uniform(-sz / 2, sz / 2, n)
    pts = np.stack([u + center[0],
                    np.full(n, center[1], dtype=np.float64),
                    v + center[2]], axis=1)
    nrm = np.tile([0, 1, 0], (n, 1)).astype(np.float64)
    return pts.astype(np.float32), nrm.astype(np.float32)


# ---------- Normal -> quaternion that flattens splat to its tangent plane ----------

def normals_to_quats(normals: np.ndarray) -> np.ndarray:
    """Build a unit quaternion (wxyz) that rotates the local frame so the
    splat's small axis (z in local) lies along `normal`."""
    n = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
    z = np.array([0, 0, 1.0])
    cos_t = n @ z
    axis = np.cross(np.broadcast_to(z, n.shape), n)
    axis_n = np.linalg.norm(axis, axis=1, keepdims=True).clip(1e-9)
    axis = axis / axis_n
    half = np.arccos(np.clip(cos_t, -1, 1)) * 0.5
    sin_h = np.sin(half)
    cos_h = np.cos(half)
    q = np.zeros((n.shape[0], 4), dtype=np.float32)
    q[:, 0] = cos_h
    q[:, 1] = axis[:, 0] * sin_h
    q[:, 2] = axis[:, 1] * sin_h
    q[:, 3] = axis[:, 2] * sin_h
    # Edge case: when normal == z exactly, axis is degenerate; quat is identity.
    flat = np.abs(cos_t - 1.0) < 1e-6
    q[flat] = (1, 0, 0, 0)
    flip = np.abs(cos_t + 1.0) < 1e-6
    q[flip] = (0, 1, 0, 0)
    return q


# ---------- Build the full scene ----------

def build_scene():
    rng = np.random.default_rng(42)
    parts = []

    # Ground plane: 3x3 light gray (denser, bigger splats so it reads as solid)
    p, n = sample_plane(center=(0, 0, 0), sx=3, sz=3, n=180_000, rng=rng)
    parts.append(dict(name='ground',
                      xyz=p, normals=n,
                      albedo=np.tile([0.62, 0.62, 0.65], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.02, dtype=np.float32),
                      roughness=np.full(len(p), 0.85, dtype=np.float32),
                      sigma=0.020))

    # Chrome sphere center stage (denser sampling + bigger splats)
    p, n = sample_sphere(center=(0, 0.40, 0), radius=0.40, n=120_000, rng=rng)
    parts.append(dict(name='chrome_sphere',
                      xyz=p, normals=n,
                      albedo=np.tile([0.95, 0.96, 0.98], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.94, dtype=np.float32),
                      roughness=np.full(len(p), 0.09, dtype=np.float32),
                      sigma=0.013))

    # Red matte ball back-right
    p, n = sample_sphere(center=(0.55, 0.30, -0.55), radius=0.30, n=70_000, rng=rng)
    parts.append(dict(name='red_ball',
                      xyz=p, normals=n,
                      albedo=np.tile([0.82, 0.12, 0.10], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.02, dtype=np.float32),
                      roughness=np.full(len(p), 0.55, dtype=np.float32),
                      sigma=0.012))

    # Blue plastic cube back-left, rotated 30
    p, n = sample_box(center=(-0.60, 0.30, -0.55), size=(0.60, 0.60, 0.60),
                      n=80_000, rng=rng, rot_deg=30.0)
    parts.append(dict(name='blue_cube',
                      xyz=p, normals=n,
                      albedo=np.tile([0.15, 0.25, 0.82], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.05, dtype=np.float32),
                      roughness=np.full(len(p), 0.30, dtype=np.float32),
                      sigma=0.013))

    # Brass torus ringed around chrome sphere base
    p, n = sample_torus(center=(0, 0.05, 0), R=0.62, r=0.05, n=30_000, rng=rng)
    parts.append(dict(name='brass_torus',
                      xyz=p, normals=n,
                      albedo=np.tile([0.92, 0.70, 0.22], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.85, dtype=np.float32),
                      roughness=np.full(len(p), 0.18, dtype=np.float32),
                      sigma=0.010))

    return parts


def main():
    t0 = time.perf_counter()
    print('=' * 72)
    print('F.26 — CGI studio scene → CRYPSOID .3dphox')
    print('=' * 72)

    parts = build_scene()
    counts = {p['name']: len(p['xyz']) for p in parts}
    n_total = sum(counts.values())
    print(f'\n[1/6] Procedural scene built: {n_total:,} splats total')
    for k, v in counts.items():
        print(f'      {k:14s}: {v:,}')

    # Concatenate all parts into one big buffer
    xyz       = np.concatenate([p['xyz']       for p in parts], axis=0)
    normals   = np.concatenate([p['normals']   for p in parts], axis=0)
    albedo    = np.concatenate([p['albedo']    for p in parts], axis=0)
    metallic  = np.concatenate([p['metallic']  for p in parts], axis=0)
    roughness = np.concatenate([p['roughness'] for p in parts], axis=0)
    sigma_lin = np.concatenate([np.full(len(p['xyz']), p['sigma'], dtype=np.float32)
                                  for p in parts], axis=0)

    # Build per-splat anisotropic scales: in-plane = sigma; through-plane = sigma*0.5
    # Keep mostly disk-like but not so thin that grazing-angle views vanish.
    scales_lin = np.stack([sigma_lin * 1.1, sigma_lin * 1.1, sigma_lin * 0.55], axis=1)
    scales_log = np.log(scales_lin).astype(np.float32)

    quats   = normals_to_quats(normals)
    opacity = np.full(n_total, 0.97, dtype=np.float32)

    # Material hint from per-splat metallic level
    mhint = np.where(metallic > 0.6,
                     MATERIAL_HINT_MIRROR,
                     np.where(metallic > 0.2, MATERIAL_HINT_GLOSSY, MATERIAL_HINT_DIFFUSE)
                     ).astype(np.uint8)
    mconf = np.full(n_total, 230, dtype=np.uint8)
    mvdep = (metallic * 255).astype(np.uint8)
    mip   = derive_mip_zoom(scales_lin)

    # Write PBR sidecar so the render script doesn't have to recover it from SH
    np.savez(OUT_PBR,
             albedo=albedo,
             metallic=metallic,
             roughness=roughness,
             F0=(0.04 * (1 - metallic)[:, None] + albedo * metallic[:, None]).astype(np.float32),
             kd=(1.0 - metallic).astype(np.float32),
             part_id=np.concatenate([np.full(len(p['xyz']), i, dtype=np.uint8)
                                      for i, p in enumerate(parts)], axis=0),
             part_names=np.array([p['name'] for p in parts]))
    print(f'\n[2/6] Wrote PBR sidecar: {OUT_PBR.name} '
          f'({OUT_PBR.stat().st_size:,} bytes)')

    # Bundle for the v25 encoder
    blobs = BlobBundle(
        xyz=xyz, scales=scales_log, quats=quats,
        opacity=opacity, sh_dc=albedo, sh_rest=None,
        tier=np.full(n_total, 2, dtype=np.uint8),
    )
    print(f'\n[3/6] Encoding v25 base container ...')
    n_bytes = encode_blobbundle_to_3dphox(blobs, OUT_PHOX)
    base_size = OUT_PHOX.stat().st_size
    print(f'      v25 base = {base_size:,} bytes')

    # Build v31 trailer: normals + edges + material_hints
    print(f'\n[4/6] Building v31 trailer (normals + edges + material_hints) ...')
    # Octahedral encode of the analytic normals (write_normals_chunk handles it)
    # tangent_angles unused (no anisotropy on tangent plane), use 0
    tangent_angles = np.zeros(n_total, dtype=np.float32)
    normals_chunk_bytes = write_normals_chunk(normals, tangent_angles)

    print(f'      computing kNN(k=4) edges ...')
    t_knn = time.perf_counter()
    edges = derive_knn_edges(xyz, k=4)
    edges_chunk_bytes = write_edges_chunk(edges)
    print(f'      kNN done in {time.perf_counter() - t_knn:.1f}s, '
          f'{len(edges_chunk_bytes):,} bytes')

    material_chunk_bytes = write_material_chunk(mhint, mconf, mvdep, mip)

    v31_chunks_meta = []
    v31_payload = b''
    cursor = 0
    for cid, name, body, extra in [
        (NORMALS_CHUNK_ID,  'normals_oct24_tangent8', normals_chunk_bytes,  {}),
        (EDGES_CHUNK_ID,    'knn_edges',              edges_chunk_bytes,    {'k': 4}),
        (MATERIAL_CHUNK_ID, 'material_hints',         material_chunk_bytes, {}),
    ]:
        meta = {'chunk_id': cid, 'name': name,
                 'offset_in_trailer': cursor, 'size_bytes': len(body)}
        meta.update(extra)
        v31_chunks_meta.append(meta)
        v31_payload += body
        cursor += len(body)

    v31_manifest = {
        'format': 'CRYPSOID_3DPHOX_V31_TRAILER',
        'source': 'F.26 cgi_studio_v1',
        'n_phoxoids': n_total,
        'chunks': v31_chunks_meta,
    }
    v31_mjson = json.dumps(v31_manifest, indent=2).encode('utf-8')
    v31_trailer = V31_MAGIC + struct.pack('<Q', len(v31_mjson)) + v31_mjson + v31_payload
    print(f'      v31 trailer = {len(v31_trailer):,} bytes')

    # Build v40 trailer: kappa + cusp (clean surfaces => low kappa, low cusp)
    print(f'\n[5/6] Building v40 trailer (kappa + cusp) ...')
    # Clean surfaces have near-zero curvature; chrome edges get a tiny boost.
    kappa = np.full(n_total, 0.04, dtype=np.float32)
    cusp_norm = np.full(n_total, 0.0, dtype=np.float32)
    # Brass torus tube has higher curvature
    offset = 0
    for p in parts:
        end = offset + len(p['xyz'])
        if p['name'] == 'brass_torus':
            kappa[offset:end] = 0.18
        elif p['name'] in ('chrome_sphere', 'red_ball'):
            kappa[offset:end] = 0.08
        offset = end

    kappa_chunk_bytes = write_kappa_chunk(kappa)
    cusp_chunk_bytes  = write_cusp_chunk(cusp_norm)
    v40_chunks_meta = []
    v40_payload = b''
    cursor = 0
    for cid, name, body in [
        (KAPPA_CHUNK_ID, 'kappa_q8', kappa_chunk_bytes),
        (CUSP_CHUNK_ID,  'cusp_q8',  cusp_chunk_bytes),
    ]:
        meta = {'chunk_id': cid, 'name': name,
                 'offset_in_trailer': cursor, 'size_bytes': len(body)}
        v40_chunks_meta.append(meta)
        v40_payload += body
        cursor += len(body)
    v40_manifest = {
        'format': 'CRYPSOID_3DPHOX_V40_TRAILER',
        'source': 'F.26 cgi_studio_v1',
        'n_phoxoids': n_total,
        'chunks': v40_chunks_meta,
    }
    v40_mjson = json.dumps(v40_manifest, indent=2).encode('utf-8')
    v40_trailer = V40_MAGIC + struct.pack('<Q', len(v40_mjson)) + v40_mjson + v40_payload
    print(f'      v40 trailer = {len(v40_trailer):,} bytes')

    # Assemble final file
    base_bytes = OUT_PHOX.read_bytes()
    full = base_bytes + v31_trailer + v40_trailer
    OUT_PHOX.write_bytes(full)
    final_size = OUT_PHOX.stat().st_size
    print(f'\n[6/6] Final .3dphox = {final_size:,} bytes  ({final_size/1024:.1f} KB)')

    # Round-trip verification
    print('\nVerification:')
    from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox
    sb = load_3dphox(OUT_PHOX)
    aux = load_aux_from_3dphox(OUT_PHOX)
    assert sb.n == n_total, f'splat count {sb.n} != {n_total}'
    assert 'normals'      in aux, 'normals missing from v31'
    assert 'edges'        in aux, 'edges missing from v31'
    assert 'material_mip' in aux, 'material_hints missing from v31'
    assert 'kappa'        in aux, 'kappa missing from v40'
    assert 'cusp_norm'    in aux, 'cusp missing from v40'
    print(f'  OK  splat count = {sb.n:,}')
    print(f'  OK  v31 chunks: normals, edges, material_hints')
    print(f'  OK  v40 chunks: kappa, cusp')
    print(f'  OK  normals mean |N| = '
          f'{np.linalg.norm(aux["normals"], axis=1).mean():.4f}  (should ~ 1.0)')

    # Write scene manifest
    scene = {
        'source': 'F.26 procedural CGI studio scene',
        'n_total': n_total,
        'parts': [{'name': p['name'], 'count': len(p['xyz']),
                    'sigma_world': float(p['sigma'])}
                   for p in parts],
        'files': {
            'phox': OUT_PHOX.name,
            'pbr_sidecar': OUT_PBR.name,
        },
    }
    OUT_SCENE.write_text(json.dumps(scene, indent=2))
    print(f'  OK  manifest -> {OUT_SCENE.name}')

    print(f'\nDONE in {time.perf_counter() - t0:.1f}s')
    print(f'  -> {OUT_PHOX}')
    print(f'  -> {OUT_PBR}')
    print(f'  -> {OUT_SCENE}')


if __name__ == '__main__':
    main()
