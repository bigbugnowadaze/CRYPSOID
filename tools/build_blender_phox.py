"""F.30.2 — OBJ + MTL → .3dphox builder.

Per docs/blender_bridge_spec.md.

Usage:
    python3 tools/build_blender_phox.py path/to/scene.obj
        [--out outputs/name.3dphox]
        [--n 250000]                  # target splat count
        [--seed 0]

Reads the OBJ + (referenced) MTL, samples each face proportional to its
area, picks per-face material, encodes a v25 + v31 + v40 .3dphox file.

Output is byte-format-compatible with all existing CRYPSOID renderers
and the WebGL viewer.
"""
from __future__ import annotations
import sys, json, struct, time, argparse
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from img2phox.obj_loader import (
    parse_obj, triangulate_face, triangle_area, triangle_normal,
    DEFAULT_MAT_NAME,
)
from img2phox.encode import encode_blobbundle_to_3dphox
from img2phox.data_classes import BlobBundle
from build_cgi_studio_phox import normals_to_quats
from crypsorender.io.normals_codec import write_normals_chunk, NORMALS_CHUNK_ID
from crypsorender.io.edges_codec import (
    write_edges_chunk, derive_knn_edges, EDGES_CHUNK_ID,
)
from crypsorender.io.material_codec import (
    write_material_chunk, MATERIAL_CHUNK_ID, derive_mip_zoom,
    MATERIAL_HINT_DIFFUSE, MATERIAL_HINT_GLOSSY, MATERIAL_HINT_MIRROR,
    MATERIAL_HINT_EMISSIVE,
)
from crypsorender.io.germ_codec import (
    write_kappa_chunk, write_cusp_chunk, KAPPA_CHUNK_ID, CUSP_CHUNK_ID,
)


V31_MAGIC = b'CRYPSOID31\x00'
V40_MAGIC = b'CRYPSOID40\x00'


HINT_TO_INT = {
    'diffuse':  MATERIAL_HINT_DIFFUSE,
    'glossy':   MATERIAL_HINT_GLOSSY,
    'mirror':   MATERIAL_HINT_MIRROR,
    'emissive': MATERIAL_HINT_EMISSIVE,
}


def sample_obj_scene(obj_path: Path, n_total: int, seed: int = 0):
    """Read OBJ, fan-triangulate, area-sample, return numpy arrays.

    Returns dict with: xyz, normals, albedo, metallic, roughness, opacity,
                        sigma, mat_id, mat_names (list).
    """
    rng = np.random.default_rng(seed)
    scene = parse_obj(obj_path)
    if not scene.vertices:
        raise SystemExit(f'OBJ has no vertices: {obj_path}')
    if not scene.faces:
        raise SystemExit(f'OBJ has no faces: {obj_path}')

    V = np.asarray(scene.vertices, dtype=np.float64)
    print(f'  parsed: {V.shape[0]:,} vertices, {len(scene.faces):,} faces, '
          f'{len(scene.materials)} materials', flush=True)

    # Triangulate; collect (tri_indices, mat_name, area, normal)
    tris = []
    for face in scene.faces:
        if len(face.indices) < 3:
            continue
        for ti in triangulate_face(face.indices):
            v0 = V[ti[0]]; v1 = V[ti[1]]; v2 = V[ti[2]]
            a = triangle_area(v0, v1, v2)
            if a <= 0:
                continue
            n = triangle_normal(v0, v1, v2)
            tris.append((ti, face.material_name or DEFAULT_MAT_NAME, a, n))

    if not tris:
        raise SystemExit('No triangles after triangulation')
    n_tris = len(tris)
    areas = np.array([t[2] for t in tris], dtype=np.float64)
    total_area = float(areas.sum())
    print(f'  triangulated: {n_tris:,} triangles, total area={total_area:.4f}',
          flush=True)

    # Allocate samples per triangle
    raw_n = n_total * areas / total_area
    n_per = np.maximum(1, np.round(raw_n).astype(int))
    actual_total = int(n_per.sum())
    print(f'  sampling: target {n_total:,}, actual {actual_total:,}',
          flush=True)

    # Sort material names so mat_id is deterministic
    mat_name_order = sorted(scene.materials.keys())
    mat_name_to_id = {name: i for i, name in enumerate(mat_name_order)}

    # Pre-build per-material PBR arrays
    mat_albedo    = np.array([scene.materials[n].albedo
                                 for n in mat_name_order], dtype=np.float32)
    mat_metallic  = np.array([scene.materials[n].metallic
                                 for n in mat_name_order], dtype=np.float32)
    mat_roughness = np.array([scene.materials[n].roughness
                                 for n in mat_name_order], dtype=np.float32)
    mat_opacity   = np.array([scene.materials[n].opacity
                                 for n in mat_name_order], dtype=np.float32)
    mat_emissive  = np.array([scene.materials[n].emissive
                                 for n in mat_name_order], dtype=np.float32)
    mat_hint      = np.array([HINT_TO_INT.get(scene.materials[n].hint,
                                                 MATERIAL_HINT_DIFFUSE)
                                 for n in mat_name_order], dtype=np.uint8)

    # Sample points
    xyz_chunks = []; nrm_chunks = []; mat_id_chunks = []; sigma_chunks = []

    for (ti, mat_name, area, n_face) in zip(
            [t[0] for t in tris],
            [t[1] for t in tris],
            areas,
            n_per,
        ):
        v0 = V[ti[0]]; v1 = V[ti[1]]; v2 = V[ti[2]]
        # Uniform on triangle via barycentrics
        u = rng.random(n_face)
        v = rng.random(n_face)
        flip = u + v > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        w = 1.0 - u - v
        pts = (w[:, None] * v0[None, :] + u[:, None] * v1[None, :]
                + v[:, None] * v2[None, :])
        nrm = triangle_normal(v0, v1, v2)
        xyz_chunks.append(pts)
        nrm_chunks.append(np.tile(nrm, (n_face, 1)))
        mat_id_chunks.append(np.full(n_face,
            mat_name_to_id.get(mat_name,
                                  mat_name_to_id.get(DEFAULT_MAT_NAME, 0)),
            dtype=np.uint8))
        sigma = max(0.003, min(0.05, 0.7 * float((area / max(n_face, 1)) ** 0.5)))
        sigma_chunks.append(np.full(n_face, sigma, dtype=np.float32))

    xyz       = np.concatenate(xyz_chunks, axis=0).astype(np.float32)
    normals   = np.concatenate(nrm_chunks, axis=0).astype(np.float32)
    mat_ids   = np.concatenate(mat_id_chunks, axis=0)
    sigma_lin = np.concatenate(sigma_chunks, axis=0)

    # Per-splat PBR by index lookup
    albedo    = mat_albedo[mat_ids]
    metallic  = mat_metallic[mat_ids]
    roughness = mat_roughness[mat_ids]
    opacity   = mat_opacity[mat_ids]
    emissive  = mat_emissive[mat_ids]
    # If emissive non-zero, blend it into albedo as the visible colour
    em_mask = (emissive > 1e-4).any(axis=1)
    albedo = np.where(em_mask[:, None],
                       np.clip(albedo + emissive, 0.0, 1.0),
                       albedo).astype(np.float32)
    hint_per = mat_hint[mat_ids]

    return {
        'xyz': xyz, 'normals': normals,
        'albedo': albedo, 'metallic': metallic, 'roughness': roughness,
        'opacity': opacity, 'sigma': sigma_lin,
        'mat_id': mat_ids, 'mat_names': mat_name_order,
        'hint': hint_per,
    }


def encode_to_3dphox(samp: dict, out_path: Path, source_label: str):
    n_total = samp['xyz'].shape[0]

    # Anisotropic disk-like splats (matching F.26/F.27 convention)
    sigma_lin = samp['sigma']
    scales_lin = np.stack([sigma_lin * 1.1, sigma_lin * 1.1, sigma_lin * 0.55],
                            axis=1)
    scales_log = np.log(scales_lin).astype(np.float32)
    quats = normals_to_quats(samp['normals'])

    # Material chunk
    mhint = samp['hint']
    mconf = np.full(n_total, 230, dtype=np.uint8)
    mvdep = (samp['metallic'] * 255).astype(np.uint8)
    mip   = derive_mip_zoom(scales_lin)

    # PBR sidecar (so render scripts can use ground-truth)
    pbr_path = out_path.with_suffix('.pbr.npz')
    F0 = (0.04 * (1 - samp['metallic'])[:, None]
          + samp['albedo'] * samp['metallic'][:, None]).astype(np.float32)
    kd = (1.0 - samp['metallic']).astype(np.float32)
    np.savez(pbr_path,
              albedo=samp['albedo'], metallic=samp['metallic'],
              roughness=samp['roughness'], F0=F0, kd=kd,
              part_id=samp['mat_id'],
              part_names=np.array(samp['mat_names']))
    print(f'  wrote PBR sidecar: {pbr_path.name}', flush=True)

    blobs = BlobBundle(xyz=samp['xyz'], scales=scales_log, quats=quats,
                        opacity=samp['opacity'], sh_dc=samp['albedo'],
                        sh_rest=None,
                        tier=np.full(n_total, 2, dtype=np.uint8))
    encode_blobbundle_to_3dphox(blobs, out_path)
    base_bytes = out_path.read_bytes()
    print(f'  v25 base = {len(base_bytes):,} bytes', flush=True)

    # v31 trailer
    tangent_angles = np.zeros(n_total, dtype=np.float32)
    normals_chunk  = write_normals_chunk(samp['normals'], tangent_angles)
    print(f'  computing kNN(k=4) ...', flush=True)
    edges = derive_knn_edges(samp['xyz'], k=4)
    edges_chunk = write_edges_chunk(edges)
    material_chunk = write_material_chunk(mhint, mconf, mvdep, mip)

    v31_meta = []; v31_payload = b''; cursor = 0
    for cid, name, body, extra in [
        (NORMALS_CHUNK_ID,  'normals_oct24_tangent8', normals_chunk,  {}),
        (EDGES_CHUNK_ID,    'knn_edges',              edges_chunk,    {'k': 4}),
        (MATERIAL_CHUNK_ID, 'material_hints',         material_chunk, {}),
    ]:
        meta = {'chunk_id': cid, 'name': name,
                 'offset_in_trailer': cursor, 'size_bytes': len(body)}
        meta.update(extra)
        v31_meta.append(meta)
        v31_payload += body; cursor += len(body)
    v31_manifest = {'format': 'CRYPSOID_3DPHOX_V31_TRAILER',
                     'source': f'F.30 {source_label}',
                     'n_phoxoids': n_total, 'chunks': v31_meta}
    v31_mjson = json.dumps(v31_manifest, indent=2).encode('utf-8')
    v31_trailer = (V31_MAGIC + struct.pack('<Q', len(v31_mjson))
                    + v31_mjson + v31_payload)
    print(f'  v31 trailer = {len(v31_trailer):,} bytes', flush=True)

    # v40 trailer (light: low kappa default)
    kappa = np.full(n_total, 0.04, dtype=np.float32)
    cusp_norm = np.full(n_total, 0.0, dtype=np.float32)
    kappa_chunk = write_kappa_chunk(kappa)
    cusp_chunk  = write_cusp_chunk(cusp_norm)
    v40_meta = []; v40_payload = b''; cursor = 0
    for cid, name, body in [
        (KAPPA_CHUNK_ID, 'kappa_q8', kappa_chunk),
        (CUSP_CHUNK_ID,  'cusp_q8',  cusp_chunk),
    ]:
        meta = {'chunk_id': cid, 'name': name,
                 'offset_in_trailer': cursor, 'size_bytes': len(body)}
        v40_meta.append(meta)
        v40_payload += body; cursor += len(body)
    v40_manifest = {'format': 'CRYPSOID_3DPHOX_V40_TRAILER',
                     'source': f'F.30 {source_label}',
                     'n_phoxoids': n_total, 'chunks': v40_meta}
    v40_mjson = json.dumps(v40_manifest, indent=2).encode('utf-8')
    v40_trailer = (V40_MAGIC + struct.pack('<Q', len(v40_mjson))
                    + v40_mjson + v40_payload)
    print(f'  v40 trailer = {len(v40_trailer):,} bytes', flush=True)

    out_path.write_bytes(base_bytes + v31_trailer + v40_trailer)
    final_size = out_path.stat().st_size
    print(f'  final .3dphox = {final_size:,} bytes ({final_size/1024:.1f} KB)',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('obj', type=str, help='Path to .obj input')
    ap.add_argument('--out', type=str, default=None,
                     help='output .3dphox (defaults to outputs/<name>.3dphox)')
    ap.add_argument('--n', type=int, default=250_000,
                     help='target total splat count')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    obj_path = Path(args.obj)
    if not obj_path.exists():
        raise SystemExit(f'OBJ not found: {obj_path}')
    if args.out is None:
        out_path = ROOT / 'outputs' / (obj_path.stem + '.3dphox')
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print(f'F.30 — Blender bridge: {obj_path.name} → {out_path.name}')
    print('=' * 72)

    t0 = time.perf_counter()
    print(f'\n[1/3] Sampling OBJ surface ...')
    samp = sample_obj_scene(obj_path, n_total=args.n, seed=args.seed)
    print(f'      {samp["xyz"].shape[0]:,} splats sampled')

    print(f'\n[2/3] Encoding .3dphox + sidecar PBR ...')
    encode_to_3dphox(samp, out_path, source_label=obj_path.stem)

    print(f'\n[3/3] Round-trip verification ...')
    from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox
    sb = load_3dphox(out_path)
    aux = load_aux_from_3dphox(out_path)
    assert sb.n == samp['xyz'].shape[0]
    for k in ['normals', 'edges', 'material_mip', 'kappa', 'cusp_norm']:
        assert k in aux, f'missing aux chunk: {k}'
    print(f'      OK  N={sb.n:,}, all chunks present, '
          f'|N|mean={np.linalg.norm(aux["normals"], axis=1).mean():.4f}')

    print(f'\nDONE in {time.perf_counter() - t0:.1f}s')
    print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
