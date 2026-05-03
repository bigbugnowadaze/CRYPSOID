"""F.27 — Stylized CGI car .3dphox (companion to F.26 studio scene).

Procedurally builds a stylized "hot-wheels" car directly as splats with
analytic normals + clean PBR. Encoded to .3dphox for the existing renderer.

Composition:
    Lower body  rounded box, painted red, slightly metallic
    Cabin       rounded box, painted red, with tinted "windows"
    4 Wheels    cylinders, matte black rubber + chrome hub
    Headlights  2 emissive-white spheres at the front
    Ground      shallow plane underneath, matte light gray

Output:
    outputs/cgi_car_v1.3dphox       — v25 + v31 + v40 trailer (~500k splats)
    outputs/cgi_car_v1.pbr.npz      — ground-truth PBR sidecar
    outputs/cgi_car_v1.scene.json   — manifest

Reuses the proven sampling helpers from build_cgi_studio_phox (re-imported).
"""
from __future__ import annotations
import sys, json, struct, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

# Reuse helpers from F.26
from build_cgi_studio_phox import (
    sample_sphere, sample_box, sample_plane, normals_to_quats,
    V31_MAGIC, V40_MAGIC,
)
from img2phox.encode import encode_blobbundle_to_3dphox
from img2phox.data_classes import BlobBundle
from crypsorender.io.normals_codec import write_normals_chunk, NORMALS_CHUNK_ID
from crypsorender.io.edges_codec import write_edges_chunk, derive_knn_edges, EDGES_CHUNK_ID
from crypsorender.io.material_codec import (
    write_material_chunk, MATERIAL_CHUNK_ID, derive_mip_zoom,
    MATERIAL_HINT_DIFFUSE, MATERIAL_HINT_GLOSSY, MATERIAL_HINT_MIRROR,
    MATERIAL_HINT_EMISSIVE,
)
from crypsorender.io.germ_codec import (
    write_kappa_chunk, write_cusp_chunk, KAPPA_CHUNK_ID, CUSP_CHUNK_ID,
)


OUT_PHOX  = ROOT / 'outputs' / 'cgi_car_v1.3dphox'
OUT_PBR   = ROOT / 'outputs' / 'cgi_car_v1.pbr.npz'
OUT_SCENE = ROOT / 'outputs' / 'cgi_car_v1.scene.json'


# ---------- Cylinder sampler (new) ----------

def sample_cylinder(center, radius, half_len, n, rng, axis='z'):
    """Sample uniformly on the surface of a cylinder.

    Side surface = 2π·r·(2·half_len), caps = 2·π·r². Allocate samples
    proportionally so coverage is uniform.
    """
    A_side = 2 * np.pi * radius * (2 * half_len)
    A_cap  = np.pi * radius * radius
    A_tot  = A_side + 2 * A_cap
    n_side = max(2, int(round(n * A_side / A_tot)))
    n_cap  = max(2, (n - n_side) // 2)
    # Side
    th = rng.uniform(0, 2 * np.pi, n_side)
    h  = rng.uniform(-half_len, half_len, n_side)
    cs, ss = np.cos(th), np.sin(th)
    # Caps
    th_p = rng.uniform(0, 2 * np.pi, n_cap)
    r_p  = np.sqrt(rng.uniform(0, 1, n_cap)) * radius
    csp, ssp = np.cos(th_p), np.sin(th_p)
    th_n = rng.uniform(0, 2 * np.pi, n_cap)
    r_n  = np.sqrt(rng.uniform(0, 1, n_cap)) * radius
    csn, ssn = np.cos(th_n), np.sin(th_n)

    if axis == 'z':
        side_pts = np.stack([radius * cs, radius * ss, h], axis=1)
        side_nrm = np.stack([cs, ss, np.zeros_like(cs)], axis=1)
        cap_p_pts = np.stack([r_p * csp, r_p * ssp,
                                np.full(n_cap, +half_len)], axis=1)
        cap_p_nrm = np.tile([0, 0, +1], (n_cap, 1)).astype(np.float64)
        cap_n_pts = np.stack([r_n * csn, r_n * ssn,
                                np.full(n_cap, -half_len)], axis=1)
        cap_n_nrm = np.tile([0, 0, -1], (n_cap, 1)).astype(np.float64)
    elif axis == 'y':
        side_pts = np.stack([radius * cs, h, radius * ss], axis=1)
        side_nrm = np.stack([cs, np.zeros_like(cs), ss], axis=1)
        cap_p_pts = np.stack([r_p * csp,  np.full(n_cap, +half_len),
                                r_p * ssp], axis=1)
        cap_p_nrm = np.tile([0, +1, 0], (n_cap, 1)).astype(np.float64)
        cap_n_pts = np.stack([r_n * csn,  np.full(n_cap, -half_len),
                                r_n * ssn], axis=1)
        cap_n_nrm = np.tile([0, -1, 0], (n_cap, 1)).astype(np.float64)
    else:  # 'x'
        side_pts = np.stack([h, radius * cs, radius * ss], axis=1)
        side_nrm = np.stack([np.zeros_like(cs), cs, ss], axis=1)
        cap_p_pts = np.stack([np.full(n_cap, +half_len), r_p * csp, r_p * ssp], axis=1)
        cap_p_nrm = np.tile([+1, 0, 0], (n_cap, 1)).astype(np.float64)
        cap_n_pts = np.stack([np.full(n_cap, -half_len), r_n * csn, r_n * ssn], axis=1)
        cap_n_nrm = np.tile([-1, 0, 0], (n_cap, 1)).astype(np.float64)

    pts = np.concatenate([side_pts, cap_p_pts, cap_n_pts], axis=0) + np.asarray(center)
    nrm = np.concatenate([side_nrm, cap_p_nrm, cap_n_nrm], axis=0)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    return pts.astype(np.float32), nrm.astype(np.float32)


# ---------- Build the car ----------

def build_car():
    rng = np.random.default_rng(11)
    parts = []

    # Ground plane (small footprint underneath the car)
    p, n = sample_plane(center=(0, 0, 0), sx=3.5, sz=2.5, n=70_000, rng=rng)
    parts.append(dict(name='ground',
                      xyz=p, normals=n,
                      albedo=np.tile([0.32, 0.33, 0.36], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.04, dtype=np.float32),
                      roughness=np.full(len(p), 0.78, dtype=np.float32),
                      sigma=0.018))

    # Lower body (main hull): wide flat-ish rounded slab
    # Centered at y=0.30, span x=±0.9, y=±0.18, z=±0.42
    p, n = sample_box(center=(0, 0.30, 0), size=(1.80, 0.36, 0.84),
                      n=80_000, rng=rng, rot_deg=0)
    parts.append(dict(name='lower_body',
                      xyz=p, normals=n,
                      albedo=np.tile([0.78, 0.10, 0.10], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.45, dtype=np.float32),
                      roughness=np.full(len(p), 0.18, dtype=np.float32),
                      sigma=0.013))

    # Cabin (smaller rounded box on top, set back slightly)
    p, n = sample_box(center=(-0.05, 0.62, 0), size=(0.85, 0.30, 0.78),
                      n=50_000, rng=rng, rot_deg=0)
    parts.append(dict(name='cabin',
                      xyz=p, normals=n,
                      albedo=np.tile([0.78, 0.10, 0.10], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.45, dtype=np.float32),
                      roughness=np.full(len(p), 0.20, dtype=np.float32),
                      sigma=0.012))

    # Window strip wrapping the cabin (tinted glass-ish)
    # Implemented as a slightly larger thin box around the cabin top
    p, n = sample_box(center=(-0.05, 0.65, 0), size=(0.78, 0.20, 0.81),
                      n=35_000, rng=rng, rot_deg=0)
    parts.append(dict(name='windows',
                      xyz=p, normals=n,
                      albedo=np.tile([0.10, 0.13, 0.18], (len(p), 1)).astype(np.float32),
                      metallic=np.full(len(p), 0.20, dtype=np.float32),
                      roughness=np.full(len(p), 0.10, dtype=np.float32),
                      sigma=0.011))

    # 4 Wheels (cylinders along z axis)
    wheel_r = 0.20
    wheel_w = 0.08
    wheel_y = 0.20  # bottom of body at y=0.12, wheel center at 0.20 lifts car
    for wx, wz, name in [
        (+0.55, +0.36, 'wheel_FR'),
        (+0.55, -0.36, 'wheel_FL'),
        (-0.55, +0.36, 'wheel_BR'),
        (-0.55, -0.36, 'wheel_BL'),
    ]:
        p, n = sample_cylinder(center=(wx, wheel_y, wz),
                                 radius=wheel_r, half_len=wheel_w,
                                 n=20_000, rng=rng, axis='z')
        parts.append(dict(name=name,
                          xyz=p, normals=n,
                          albedo=np.tile([0.07, 0.07, 0.08], (len(p), 1)).astype(np.float32),
                          metallic=np.full(len(p), 0.05, dtype=np.float32),
                          roughness=np.full(len(p), 0.78, dtype=np.float32),
                          sigma=0.010))
        # Chrome hub: a smaller silver disk on each wheel face
        for sign, hub_name in [(+1, name + '_hubR'), (-1, name + '_hubL')]:
            p2, n2 = sample_cylinder(center=(wx, wheel_y, wz + sign * (wheel_w + 0.005)),
                                       radius=wheel_r * 0.45, half_len=0.015,
                                       n=4_000, rng=rng, axis='z')
            parts.append(dict(name=hub_name,
                              xyz=p2, normals=n2,
                              albedo=np.tile([0.92, 0.93, 0.95], (len(p2), 1)).astype(np.float32),
                              metallic=np.full(len(p2), 0.92, dtype=np.float32),
                              roughness=np.full(len(p2), 0.10, dtype=np.float32),
                              sigma=0.008))

    # Headlights (2 emissive spheres at the front)
    for hx, hz, name in [(+0.88, +0.28, 'headlight_R'),
                           (+0.88, -0.28, 'headlight_L')]:
        p, n = sample_sphere(center=(hx, 0.30, hz), radius=0.07,
                              n=6_000, rng=rng)
        parts.append(dict(name=name,
                          xyz=p, normals=n,
                          albedo=np.tile([0.98, 0.96, 0.85], (len(p), 1)).astype(np.float32),
                          metallic=np.full(len(p), 0.05, dtype=np.float32),
                          roughness=np.full(len(p), 0.20, dtype=np.float32),
                          sigma=0.008))

    return parts


def main():
    t0 = time.perf_counter()
    print('=' * 72)
    print('F.27 — CGI car (HDRI-ready) → CRYPSOID .3dphox')
    print('=' * 72)

    parts = build_car()
    counts = {p['name']: len(p['xyz']) for p in parts}
    n_total = sum(counts.values())
    print(f'\n[1/6] Procedural car built: {n_total:,} splats')
    for k, v in counts.items():
        print(f'      {k:18s}: {v:,}')

    xyz       = np.concatenate([p['xyz']       for p in parts], axis=0)
    normals   = np.concatenate([p['normals']   for p in parts], axis=0)
    albedo    = np.concatenate([p['albedo']    for p in parts], axis=0)
    metallic  = np.concatenate([p['metallic']  for p in parts], axis=0)
    roughness = np.concatenate([p['roughness'] for p in parts], axis=0)
    sigma_lin = np.concatenate([np.full(len(p['xyz']), p['sigma'], dtype=np.float32)
                                  for p in parts], axis=0)
    scales_lin = np.stack([sigma_lin * 1.1, sigma_lin * 1.1, sigma_lin * 0.55], axis=1)
    scales_log = np.log(scales_lin).astype(np.float32)
    quats   = normals_to_quats(normals)
    opacity = np.full(n_total, 0.97, dtype=np.float32)

    # Material hints
    mhint = np.where(metallic > 0.6, MATERIAL_HINT_MIRROR,
                     np.where(metallic > 0.2, MATERIAL_HINT_GLOSSY,
                                MATERIAL_HINT_DIFFUSE)).astype(np.uint8)
    mconf = np.full(n_total, 230, dtype=np.uint8)
    mvdep = (metallic * 255).astype(np.uint8)
    mip   = derive_mip_zoom(scales_lin)

    np.savez(OUT_PBR,
             albedo=albedo, metallic=metallic, roughness=roughness,
             F0=(0.04 * (1 - metallic)[:, None] + albedo * metallic[:, None]).astype(np.float32),
             kd=(1.0 - metallic).astype(np.float32),
             part_id=np.concatenate([np.full(len(p['xyz']), i, dtype=np.uint8)
                                      for i, p in enumerate(parts)], axis=0),
             part_names=np.array([p['name'] for p in parts]))
    print(f'\n[2/6] PBR sidecar -> {OUT_PBR.name} ({OUT_PBR.stat().st_size:,} bytes)')

    blobs = BlobBundle(xyz=xyz, scales=scales_log, quats=quats,
                        opacity=opacity, sh_dc=albedo, sh_rest=None,
                        tier=np.full(n_total, 2, dtype=np.uint8))
    print(f'\n[3/6] Encoding v25 base ...')
    encode_blobbundle_to_3dphox(blobs, OUT_PHOX)
    base_bytes = OUT_PHOX.read_bytes()
    print(f'      v25 base = {len(base_bytes):,} bytes')

    print(f'\n[4/6] Building v31 trailer ...')
    tangent_angles = np.zeros(n_total, dtype=np.float32)
    normals_chunk = write_normals_chunk(normals, tangent_angles)
    print(f'      kNN(k=4) ...')
    edges = derive_knn_edges(xyz, k=4)
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
                     'source': 'F.27 cgi_car_v1', 'n_phoxoids': n_total,
                     'chunks': v31_meta}
    v31_mjson = json.dumps(v31_manifest, indent=2).encode('utf-8')
    v31_trailer = V31_MAGIC + struct.pack('<Q', len(v31_mjson)) + v31_mjson + v31_payload
    print(f'      v31 trailer = {len(v31_trailer):,} bytes')

    print(f'\n[5/6] Building v40 trailer ...')
    kappa = np.full(n_total, 0.04, dtype=np.float32)
    cusp_norm = np.full(n_total, 0.0, dtype=np.float32)
    offset = 0
    for p in parts:
        end = offset + len(p['xyz'])
        if 'wheel' in p['name'] and 'hub' not in p['name']:
            kappa[offset:end] = 0.20
        elif p['name'].startswith('headlight'):
            kappa[offset:end] = 0.18
        elif 'hub' in p['name']:
            kappa[offset:end] = 0.18
        offset = end
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
                     'source': 'F.27 cgi_car_v1', 'n_phoxoids': n_total,
                     'chunks': v40_meta}
    v40_mjson = json.dumps(v40_manifest, indent=2).encode('utf-8')
    v40_trailer = V40_MAGIC + struct.pack('<Q', len(v40_mjson)) + v40_mjson + v40_payload
    print(f'      v40 trailer = {len(v40_trailer):,} bytes')

    full = base_bytes + v31_trailer + v40_trailer
    OUT_PHOX.write_bytes(full)
    print(f'\n[6/6] Final .3dphox = {OUT_PHOX.stat().st_size:,} bytes  '
          f'({OUT_PHOX.stat().st_size/1024:.1f} KB)')

    # Round-trip verify
    print('\nVerification:')
    from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox
    sb = load_3dphox(OUT_PHOX)
    aux = load_aux_from_3dphox(OUT_PHOX)
    assert sb.n == n_total
    assert all(k in aux for k in ['normals', 'edges', 'material_mip',
                                     'kappa', 'cusp_norm'])
    print(f'  OK  N={sb.n:,}, all chunks present, '
          f'|N|mean={np.linalg.norm(aux["normals"], axis=1).mean():.4f}')

    OUT_SCENE.write_text(json.dumps({
        'source': 'F.27 procedural CGI car',
        'n_total': n_total,
        'parts': [{'name': p['name'], 'count': len(p['xyz']),
                    'sigma_world': float(p['sigma'])} for p in parts],
        'files': {'phox': OUT_PHOX.name,
                   'pbr_sidecar': OUT_PBR.name},
    }, indent=2))
    print(f'  OK  manifest -> {OUT_SCENE.name}')
    print(f'\nDONE in {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
  