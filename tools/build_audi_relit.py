"""F.28 — Inverse-Lambert albedo recovery on the trained-3DGS Audi.

Goal: strip the baked-in capture-time lighting from the v40 Audi's SH-DC
so the renderer can re-light it cleanly. Drops the "double-lit" look the
existing photoreal Audi has.

The trained 3DGS color is C(view) = sh_dc + sh_rest · Y(view). The DC
term is the view-averaged colour. For a Lambertian surface lit by a
single dominant directional light L plus ambient A:

    sh_dc ≈ albedo · (A + D · max(0, n · L))

We estimate L analytically from a per-normal-bin brightness fit (or use
+Y if data shows that's the dominant direction), then recover:

    albedo = sh_dc / max(A + D · max(0, n · L), epsilon)

Result is a new .3dphox with the same geometry / normals / edges but with
recovered albedo in the sh_dc slot. The render script using this file
through the photoreal stack will apply lighting *on* the recovered albedo
without compounding the baked-in light.

Usage:
    python3 tools/build_audi_relit.py
"""
from __future__ import annotations
import sys, json, struct, time, shutil
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox

SRC = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT = ROOT / 'outputs' / 'v40_audi_full_relit.3dphox'
META = ROOT / 'outputs' / 'v40_audi_full_relit.meta.json'


def estimate_dominant_light(normals, brightness, n_bins=18):
    """Find the direction L that maximises the correlation between
    max(0, n·L) and brightness. Coarse search on a Fibonacci sphere.
    """
    # Sample candidate light directions on the upper-hemisphere mostly
    i = np.arange(n_bins, dtype=np.float64) + 0.5
    phi = np.arccos(1 - i / n_bins)
    theta = np.pi * (1 + 5 ** 0.5) * i
    L_cands = np.stack([np.cos(theta) * np.sin(phi),
                          np.cos(phi),
                          np.sin(theta) * np.sin(phi)], axis=1)
    # subsample splats for speed
    rng = np.random.default_rng(0)
    n = normals.shape[0]
    idx = rng.choice(n, size=min(50_000, n), replace=False)
    N = normals[idx]
    B = brightness[idx]
    B_centered = B - B.mean()
    best_L, best_score = None, -np.inf
    for L in L_cands:
        Lf = L / np.linalg.norm(L)
        cos_t = np.maximum(0.0, N @ Lf)
        ct_centered = cos_t - cos_t.mean()
        denom = np.sqrt((ct_centered ** 2).sum() * (B_centered ** 2).sum() + 1e-12)
        if denom < 1e-9:
            continue
        score = (ct_centered * B_centered).sum() / denom
        if score > best_score:
            best_score, best_L = float(score), Lf
    return best_L, best_score


def fit_ambient_diffuse(normals, brightness, L, n_bins=12):
    """Given L, fit (A, D) so that A + D·max(0, n·L) explains brightness mean
    in each normal·L bin. Linear least squares.
    """
    cos_t = np.maximum(0.0, normals @ L)
    edges = np.linspace(0, 1, n_bins + 1)
    bins_x, bins_y = [], []
    for i in range(n_bins):
        mask = (cos_t >= edges[i]) & (cos_t < edges[i + 1])
        if mask.sum() < 100:
            continue
        bins_x.append(0.5 * (edges[i] + edges[i + 1]))
        bins_y.append(brightness[mask].mean())
    X = np.array(bins_x); Y = np.array(bins_y)
    # Y = A + D * X  →  least squares
    A_mat = np.stack([np.ones_like(X), X], axis=1)
    sol, *_ = np.linalg.lstsq(A_mat, Y, rcond=None)
    A, D = float(sol[0]), float(sol[1])
    # Clamp to reasonable physical values
    A = max(A, 0.05)
    D = max(D, 0.05)
    # Renormalise so A + D = 1 (preserves overall mean brightness)
    s = A + D
    return A / s, D / s, X, Y


def main():
    t0 = time.perf_counter()
    print(f'[1/5] Loading {SRC.name} ...')
    sb = load_3dphox(SRC)
    aux = load_aux_from_3dphox(SRC)
    normals = aux['normals'].astype(np.float32)
    sh_dc = sb.sh_dc.astype(np.float32)
    n = sb.n
    brightness = sh_dc.mean(axis=1)
    print(f'  N={n:,} splats, brightness mean={brightness.mean():.3f}, '
          f'p10={np.percentile(brightness, 10):.3f}, p90={np.percentile(brightness, 90):.3f}')

    print(f'\n[2/5] Estimating dominant light direction (Fibonacci search) ...')
    L, score = estimate_dominant_light(normals, brightness)
    print(f'  L = ({L[0]:+.3f}, {L[1]:+.3f}, {L[2]:+.3f})  correlation = {score:.3f}')

    print(f'\n[3/5] Fitting ambient + diffuse split ...')
    A, D, bx, by = fit_ambient_diffuse(normals, brightness, L)
    print(f'  A = {A:.3f}, D = {D:.3f}  (renormalised so A+D=1)')
    print(f'  bin samples (cos_t -> brightness):')
    for x, y in zip(bx, by):
        print(f'    cos_t={x:.2f}  brightness={y:.3f}')

    print(f'\n[4/5] Recovering albedo ...')
    cos_t = np.maximum(0.0, normals @ L)
    illum = A + D * cos_t                                   # (N,)
    illum = np.maximum(illum, 0.20)                         # floor to avoid blowup
    # Apply per-channel; sh_dc is already albedo-form in [0,1]
    albedo = (sh_dc / illum[:, None]).astype(np.float32)
    # Clamp to [0, 1] — no negative or hyperbright albedos
    albedo = np.clip(albedo, 0.0, 1.0)

    bright_new = albedo.mean(axis=1)
    print(f'  albedo brightness: mean={bright_new.mean():.3f}, '
          f'p10={np.percentile(bright_new, 10):.3f}, p90={np.percentile(bright_new, 90):.3f}')
    # Verify equalisation
    for ylo in [0, 0.2, 0.5, 0.8]:
        yhi = ylo + 0.2
        mask = (normals[:, 1] >= ylo) & (normals[:, 1] < yhi)
        if mask.sum() > 0:
            print(f'    ny in [{ylo:.1f}, {yhi:.1f}]: '
                  f'orig {brightness[mask].mean():.3f} -> recovered {bright_new[mask].mean():.3f}')

    print(f'\n[5/5] Splicing recovered albedo back into the .3dphox file ...')
    # We re-encode the sh_dc / opacity bytes only and splice into the existing
    # file, preserving the v25 layout AND the v31/v40 trailers verbatim.
    raw = SRC.read_bytes()
    # parse v25 manifest
    magic = raw[:11]
    if not magic.startswith(b'CRYPSOID25'):
        # phox_loader handles other magics but for splicing we need v25 layout.
        # Walk forward: 11 magic + 8 manifest_len + manifest + chunks
        pass
    mlen = struct.unpack('<Q', raw[11:19])[0]
    manifest = json.loads(raw[19:19 + mlen])
    chunks = manifest['chunks']
    chunks_offset_in_file = 19 + mlen
    # Find dc_rgb_opacity_u8
    dc_meta = next(c for c in chunks if c['name'] == 'dc_rgb_opacity_u8')
    # The chunk is zlib-compressed. We need to write a new compressed payload
    # of EXACTLY the same compressed size to splice in place — that's not
    # always achievable with zlib. So instead: re-emit the WHOLE container.
    # Simplest: copy original bytes, but rewrite dc + recompute manifest.
    import zlib

    n_total = sb.n
    arr = np.zeros((n_total, 4), dtype=np.uint8)
    arr[:, :3] = (albedo.clip(0, 1) * 255).astype(np.uint8)
    arr[:, 3]  = (sb.opacities.clip(0, 1) * 255).astype(np.uint8) \
                 if sb.opacities.dtype != np.uint8 else sb.opacities
    new_dc_raw = arr.tobytes()
    new_dc_comp = zlib.compress(new_dc_raw, level=6)
    new_dc_crc  = zlib.crc32(new_dc_raw) & 0xFFFFFFFF
    print(f'  new dc chunk: raw={len(new_dc_raw):,} comp={len(new_dc_comp):,} '
          f'(was comp={dc_meta["compressed_bytes"]:,})')

    # Rebuild the v25 base bytes (chunks ordered as in the manifest)
    new_chunks_meta = []
    new_chunks_payload = b''
    cursor = 0
    for c in chunks:
        if c['name'] == 'dc_rgb_opacity_u8':
            comp = new_dc_comp
            crc = new_dc_crc
        else:
            # Pull the compressed bytes from the original file at the
            # absolute offset = chunks_offset_in_file + c['offset']
            start = chunks_offset_in_file + c['offset']
            comp = raw[start:start + c['compressed_bytes']]
            crc = c.get('crc32', None)   # not all manifests carry crc32
        meta = dict(c)
        meta['offset'] = cursor
        meta['compressed_bytes'] = len(comp)
        if crc is not None:
            meta['crc32'] = crc
        new_chunks_meta.append(meta)
        new_chunks_payload += comp
        cursor += len(comp)
    new_manifest = dict(manifest)
    new_manifest['chunks'] = new_chunks_meta
    new_manifest_json = json.dumps(new_manifest, separators=(',', ':')).encode('utf-8')
    new_v25 = magic + struct.pack('<Q', len(new_manifest_json)) + new_manifest_json + new_chunks_payload

    # Find where v31 trailer started in the original (= end of original v25 base)
    V31_MAGIC = b'CRYPSOID31\x00'
    V40_MAGIC = b'CRYPSOID40\x00'
    v31_pos = raw.rfind(V31_MAGIC)
    if v31_pos < 0:
        # No v31; write only v25
        OUT.write_bytes(new_v25)
    else:
        v40_pos = raw.find(V40_MAGIC, v31_pos + len(V31_MAGIC))
        if v40_pos < 0:
            v31_trailer_bytes = raw[v31_pos:]
            v40_trailer_bytes = b''
        else:
            v31_trailer_bytes = raw[v31_pos:v40_pos]
            v40_trailer_bytes = raw[v40_pos:]
        OUT.write_bytes(new_v25 + v31_trailer_bytes + v40_trailer_bytes)
    print(f'  wrote {OUT.name}  ({OUT.stat().st_size:,} bytes; '
          f'orig was {len(raw):,})')

    # Verify round-trip
    print('\nVerification:')
    sb2 = load_3dphox(OUT)
    aux2 = load_aux_from_3dphox(OUT)
    assert sb2.n == n, f'splat count mismatch {sb2.n} vs {n}'
    sample_diff = np.abs(sb2.sh_dc.mean(axis=1) - bright_new).mean()
    print(f'  N preserved: {sb2.n:,}')
    print(f'  trailers preserved: '
          f'normals={"normals" in aux2}  edges={"edges" in aux2}  '
          f'kappa={"kappa" in aux2}  cusp_norm={"cusp_norm" in aux2}')
    print(f'  recovered-albedo round-trip mean abs diff: {sample_diff:.5f}')

    # Save metadata for reproducibility / for the renderer
    META.write_text(json.dumps({
        'src': str(SRC.name),
        'method': 'inverse-Lambert with Fibonacci-search dominant L + LSQ A,D',
        'L': L.tolist(),
        'L_correlation_score': score,
        'A': A,
        'D': D,
        'illum_floor': 0.20,
        'orig_brightness_mean': float(brightness.mean()),
        'recovered_brightness_mean': float(bright_new.mean()),
    }, indent=2))
    print(f'  meta -> {META.name}')

    print(f'\nDONE in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
