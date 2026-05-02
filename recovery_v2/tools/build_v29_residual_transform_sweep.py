#!/usr/bin/env python3
"""
CRYPSOID v0.29: render-gated SH correction debt burndown.

What this does:
1. If real v25/v27 containers are present, reconstructs the q8 SH residual stream:
       original_q8_SH - VQ_render_core_SH
   Then sweeps exact, reversible residual layouts/codecs and optionally writes a v29 archive container.
2. If real containers are missing, runs the same exactness/codec harness on a synthetic residual field.
   Synthetic results validate the machinery only; they are not Audi compression claims.

No GPU required. No ML dependency required.
"""
from __future__ import annotations

import argparse
import bz2
import json
import lzma
import math
import os
import struct
import time
import zlib
import binascii
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import brotli  # type: ignore
except Exception:  # pragma: no cover
    brotli = None

MAGIC29 = b"CRYPSOID29\0"  # 11 bytes

# -----------------------------
# Container IO
# -----------------------------

def read_container(path: Path):
    with path.open("rb") as f:
        magic = f.read(11)
        ml = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(ml))
        blob = f.read()
    chunks = {c["name"]: c for c in manifest["chunks"]}

    def comp(name: str) -> bytes:
        c = chunks[name]
        return blob[c["offset"]:c["offset"] + c["compressed_bytes"]]

    def dec(name: str) -> bytes:
        return zlib.decompress(comp(name))

    return magic, manifest, blob, chunks, comp, dec


def write_container(path: Path, manifest: dict, payloads: List[bytes]) -> int:
    chunks = manifest["chunks"]
    off = 0
    for c, p in zip(chunks, payloads):
        c["offset"] = off
        c["compressed_bytes"] = len(p)
        off += len(p)
    encoded = json.dumps(manifest, indent=2).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC29)
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for p in payloads:
            f.write(p)
    return path.stat().st_size

# -----------------------------
# Compression codecs
# -----------------------------

def codec_compress(name: str, raw: bytes) -> bytes:
    if name == "zlib6":
        return zlib.compress(raw, 6)
    if name == "zlib9":
        return zlib.compress(raw, 9)
    if name == "bz2":
        return bz2.compress(raw, compresslevel=9)
    if name == "lzma6":
        return lzma.compress(raw, preset=6)
    if name == "brotli5":
        if brotli is None:
            raise RuntimeError("brotli not installed")
        return brotli.compress(raw, quality=5)
    if name == "brotli9":
        if brotli is None:
            raise RuntimeError("brotli not installed")
        return brotli.compress(raw, quality=9)
    raise ValueError(name)


def codec_decompress(name: str, payload: bytes) -> bytes:
    if name.startswith("zlib"):
        return zlib.decompress(payload)
    if name == "bz2":
        return bz2.decompress(payload)
    if name == "lzma6":
        return lzma.decompress(payload)
    if name.startswith("brotli"):
        if brotli is None:
            raise RuntimeError("brotli not installed")
        return brotli.decompress(payload)
    raise ValueError(name)

# -----------------------------
# Residual state loading
# -----------------------------

@dataclass
class ResidualState:
    name: str
    residual: np.ndarray        # int8/int16, shape [N,45]
    sh_q8: np.ndarray           # int8, shape [N,45]
    approx_q8: np.ndarray       # int8, shape [N,45]
    xyz: np.ndarray             # float32, shape [N,3]
    tiers: np.ndarray           # uint8, shape [N]
    source: dict
    base_manifest: Optional[dict] = None
    base_payloads: Optional[List[bytes]] = None


def decode_u24_xyz(raw: bytes, n: int, bounds_min: List[float], bounds_max: List[float]) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 9)
    q = np.empty((n, 3), dtype=np.uint32)
    for j in range(3):
        q[:, j] = (
            a[:, 3*j].astype(np.uint32)
            | (a[:, 3*j + 1].astype(np.uint32) << 8)
            | (a[:, 3*j + 2].astype(np.uint32) << 16)
        )
    mn = np.asarray(bounds_min, dtype=np.float32)
    mx = np.asarray(bounds_max, dtype=np.float32)
    return (q.astype(np.float32) / float((1 << 24) - 1)) * (mx - mn) + mn


def load_real_state(v25: Path, v27: Path, v25_report: Path, v27_report: Path) -> ResidualState:
    m25, man25, blob25, ch25, comp25, dec25 = read_container(v25)
    m27, man27, blob27, ch27, comp27, dec27 = read_container(v27)
    rep25 = json.loads(v25_report.read_text())
    rep27 = json.loads(v27_report.read_text())
    n = int(rep25["input"]["source_splats"])

    sh_q8 = np.frombuffer(dec25("sh_rest_q8_global"), dtype=np.int8).reshape(n, 45).copy()
    labels = np.frombuffer(dec27("sh_vq128_idx_u8"), dtype=np.uint8).reshape(n, 3).copy()
    codebooks = np.frombuffer(dec27("sh_vq128_codebook_i8"), dtype=np.int8).reshape(3, 128, 15).copy()
    tiers = np.frombuffer(dec25("tier_labels_u8"), dtype=np.uint8).copy()

    approx = np.empty_like(sh_q8)
    for g in range(3):
        approx[:, g*15:(g+1)*15] = codebooks[g][labels[:, g]]

    residual_i16 = sh_q8.astype(np.int16) - approx.astype(np.int16)
    if int(residual_i16.min()) >= -128 and int(residual_i16.max()) <= 127:
        residual = residual_i16.astype(np.int8)
    else:
        residual = residual_i16

    xyz_chunk = ch25.get("xyz_u24_fixed", ch27.get("xyz_u24_fixed"))
    dec_xyz = dec25 if "xyz_u24_fixed" in ch25 else dec27
    xyz = decode_u24_xyz(dec_xyz("xyz_u24_fixed"), n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])

    base_payloads = [comp27(c["name"]) for c in man27["chunks"]]
    return ResidualState(
        name="Audi_A5_real_v25_v27",
        residual=residual,
        sh_q8=sh_q8,
        approx_q8=approx,
        xyz=xyz,
        tiers=tiers,
        source={
            "mode": "real",
            "source_splats": n,
            "source_ply_bytes": rep25["input"].get("source_ply_bytes"),
            "v11_vq256_bytes": rep25["input"].get("v11_vq256_bytes"),
            "v25_full_attribute_bytes": v25.stat().st_size,
            "v27_render_bytes": v27.stat().st_size,
            "v27_reported_render_bytes": rep27.get("v27", {}).get("render_container_bytes"),
        },
        base_manifest=man27,
        base_payloads=base_payloads,
    )


def make_synthetic_state(n: int, seed: int = 29029) -> ResidualState:
    rng = np.random.default_rng(seed)
    # Vehicle-like-ish elongated point distribution with correlated residuals.
    x = rng.normal(0, 2.2, size=n)
    y = rng.normal(0, 0.45, size=n)
    z = rng.normal(0, 0.85, size=n)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    # Tiers mimic panel/body/detail/silhouette-ish buckets.
    radial = np.sqrt((x / 2.8) ** 2 + (z / 1.2) ** 2)
    tiers = np.digitize(radial + 0.25 * rng.random(n), [0.55, 0.9, 1.25]).astype(np.uint8)

    base = rng.laplace(0, 3.0, size=(n, 45))
    # spatial/context drift: each tier has its own bias vector, so layouts/context can sometimes matter.
    tier_bias = rng.integers(-3, 4, size=(4, 45))
    group_bias = np.repeat(np.array([[0]*15, [1]*15, [-1]*15]), 1, axis=0).reshape(45)
    residual = np.rint(base + tier_bias[np.clip(tiers, 0, 3)] + group_bias).clip(-80, 96).astype(np.int8)
    # Approx and source q8 just need to obey exact reconstruction.
    approx = rng.integers(-80, 81, size=(n, 45), dtype=np.int16).astype(np.int8)
    sh_q8 = (approx.astype(np.int16) + residual.astype(np.int16)).clip(-128, 127).astype(np.int8)
    residual = (sh_q8.astype(np.int16) - approx.astype(np.int16)).astype(np.int8)
    return ResidualState(
        name="synthetic_v29_residual_field",
        residual=residual,
        sh_q8=sh_q8,
        approx_q8=approx,
        xyz=xyz,
        tiers=tiers,
        source={"mode": "synthetic", "source_splats": n, "truth_note": "Synthetic smoke test only; not an Audi compression claim."},
    )

# -----------------------------
# Layout transforms
# -----------------------------

def quantize_u10(x: np.ndarray) -> np.ndarray:
    mn = x.min(axis=0)
    mx = x.max(axis=0)
    denom = np.maximum(mx - mn, 1e-9)
    q = np.floor((x - mn) / denom * 1023.0).clip(0, 1023).astype(np.uint32)
    return q


def part1by2(n: np.ndarray) -> np.ndarray:
    n = n & 0x3ff
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n


def morton_order(xyz: np.ndarray) -> np.ndarray:
    q = quantize_u10(xyz)
    code = part1by2(q[:, 0]) | (part1by2(q[:, 1]) << 1) | (part1by2(q[:, 2]) << 2)
    return np.argsort(code, kind="mergesort")


def inv_perm(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size, dtype=order.dtype)
    return inv


def zigzag_i16_to_u16(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.int16)
    return ((y << 1) ^ (y >> 15)).astype(np.uint16)


def zigzag_u16_to_i16(u: np.ndarray) -> np.ndarray:
    v = u.astype(np.uint16)
    return ((v >> 1).astype(np.int16) ^ -(v & 1).astype(np.int16)).astype(np.int16)


def bitplanes_u8(a: np.ndarray) -> bytes:
    flat = a.reshape(-1).astype(np.uint8)
    planes = []
    for b in range(7, -1, -1):
        planes.append(np.packbits(((flat >> b) & 1).astype(np.uint8)).tobytes())
    return b"".join(planes)


def encode_candidate_raw(name: str, state: ResidualState) -> Tuple[bytes, dict, Callable[[bytes, dict, ResidualState], np.ndarray]]:
    r = state.residual
    n, c = r.shape

    def dec_splat(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"]).copy()

    if name == "splat_major_raw":
        return np.ascontiguousarray(r).tobytes(), {"layout": name, "dtype": str(r.dtype), "shape": [n, c]}, dec_splat

    if name == "coefficient_major_transpose":
        raw = np.ascontiguousarray(r.T).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(c, n).T.copy()
        return raw, {"layout": name, "dtype": str(r.dtype), "shape": [c, n]}, dec

    if name == "group_major_3x15":
        parts = [np.ascontiguousarray(r[:, g*15:(g+1)*15]).tobytes() for g in range(3)]
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            out = np.empty((n, 45), dtype=np.dtype(meta["dtype"]))
            item = out.dtype.itemsize
            off = 0
            for g in range(3):
                length = n * 15 * item
                out[:, g*15:(g+1)*15] = np.frombuffer(raw[off:off+length], dtype=out.dtype).reshape(n, 15)
                off += length
            return out
        return b"".join(parts), {"layout": name, "dtype": str(r.dtype), "shape": [n, c]}, dec

    if name == "band_split_low_mid_high":
        # For each of the 3 groups, split coefficient positions into low/mid/high bands.
        band_slices = [(0, 3), (3, 8), (8, 15)]
        parts = []
        widths = []
        for g in range(3):
            for a, b in band_slices:
                arr = np.ascontiguousarray(r[:, g*15+a:g*15+b])
                parts.append(arr.tobytes())
                widths.append(b-a)
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            dtype = np.dtype(meta["dtype"])
            out = np.empty((n, 45), dtype=dtype)
            off = 0
            k = 0
            for g in range(3):
                for a, b in band_slices:
                    width = b-a
                    length = n * width * dtype.itemsize
                    out[:, g*15+a:g*15+b] = np.frombuffer(raw[off:off+length], dtype=dtype).reshape(n, width)
                    off += length; k += 1
            return out
        return b"".join(parts), {"layout": name, "dtype": str(r.dtype), "shape": [n, c], "bands": band_slices}, dec

    if name == "morton_splat_major":
        order = morton_order(state.xyz)
        raw = np.ascontiguousarray(r[order]).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            order = morton_order(st.xyz)
            arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(n, c)
            out = np.empty_like(arr)
            out[order] = arr
            return out
        return raw, {"layout": name, "dtype": str(r.dtype), "shape": [n, c], "order": "morton_xyz_u10"}, dec

    if name == "morton_delta_i16":
        order = morton_order(state.xyz)
        arr = r[order].astype(np.int16)
        diff = np.empty_like(arr, dtype=np.int16)
        diff[0] = arr[0]
        diff[1:] = arr[1:] - arr[:-1]
        raw = np.ascontiguousarray(diff).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            order = morton_order(st.xyz)
            diff = np.frombuffer(raw, dtype=np.int16).reshape(n, c).copy()
            arr = np.cumsum(diff, axis=0).astype(st.residual.dtype)
            out = np.empty_like(arr)
            out[order] = arr
            return out
        return raw, {"layout": name, "dtype": "int16", "shape": [n, c], "order": "morton_xyz_u10", "predictor": "previous_morton_row"}, dec

    if name == "tier_then_coefficient_major":
        order = np.lexsort((morton_order(state.xyz), state.tiers))
        arr = r[order].T
        raw = np.ascontiguousarray(arr).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            order = np.lexsort((morton_order(st.xyz), st.tiers))
            arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(c, n).T
            out = np.empty_like(arr)
            out[order] = arr
            return out
        return raw, {"layout": name, "dtype": str(r.dtype), "shape": [c, n], "order": "tier_then_morton", "semantic": "context split without separate streams"}, dec

    if name == "zigzag_splat_major_u16":
        zz = zigzag_i16_to_u16(r.astype(np.int16))
        raw = np.ascontiguousarray(zz).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            zz = np.frombuffer(raw, dtype=np.uint16).reshape(n, c)
            return zigzag_u16_to_i16(zz).astype(st.residual.dtype)
        return raw, {"layout": name, "dtype": "uint16", "shape": [n, c], "map": "zigzag_i16_to_u16"}, dec

    if name == "sign_magnitude_planes":
        arr = r.astype(np.int16)
        sign = (arr < 0).astype(np.uint8)
        mag = np.abs(arr).clip(0, 127).astype(np.uint8)
        raw = np.packbits(sign.reshape(-1)).tobytes() + np.ascontiguousarray(mag).tobytes()
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            total = n * c
            sign_bytes = (total + 7) // 8
            sign_bits = np.unpackbits(np.frombuffer(raw[:sign_bytes], dtype=np.uint8))[:total].reshape(n, c)
            mag = np.frombuffer(raw[sign_bytes:], dtype=np.uint8).reshape(n, c).astype(np.int16)
            out = np.where(sign_bits.astype(bool), -mag, mag).astype(st.residual.dtype)
            return out
        return raw, {"layout": name, "dtype": "packed_sign_plus_u8_mag", "shape": [n, c]}, dec

    if name == "zero_mask_values":
        flat = r.reshape(-1)
        nz = flat != 0
        mask = np.packbits(nz.astype(np.uint8)).tobytes()
        vals = np.ascontiguousarray(flat[nz]).tobytes()
        raw = mask + vals
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            total = n*c
            mask_len = (total + 7) // 8
            nz = np.unpackbits(np.frombuffer(raw[:mask_len], dtype=np.uint8))[:total].astype(bool)
            out = np.zeros(total, dtype=np.dtype(meta["dtype"]))
            out[nz] = np.frombuffer(raw[mask_len:], dtype=out.dtype)
            return out.reshape(n, c)
        return raw, {"layout": name, "dtype": str(r.dtype), "shape": [n, c], "mask": "packbits_nonzero"}, dec

    if name == "bitplane_zigzag_u8_if_safe":
        if int(r.min()) < -128 or int(r.max()) > 127:
            raise RuntimeError("bitplane_zigzag_u8_if_safe only supports int8-range residuals")
        zz = ((r.astype(np.int16) << 1) ^ (r.astype(np.int16) >> 7)).clip(0, 255).astype(np.uint8)
        raw = bitplanes_u8(zz)
        def dec(raw: bytes, meta: dict, st: ResidualState) -> np.ndarray:
            total = n*c
            plane_len = (total + 7) // 8
            flat = np.zeros(total, dtype=np.uint8)
            off = 0
            for b in range(7, -1, -1):
                bits = np.unpackbits(np.frombuffer(raw[off:off+plane_len], dtype=np.uint8))[:total]
                flat |= (bits.astype(np.uint8) << b)
                off += plane_len
            u = flat.reshape(n, c).astype(np.uint16)
            out = ((u >> 1).astype(np.int16) ^ -(u & 1).astype(np.int16)).astype(np.int16)
            return out.astype(st.residual.dtype)
        return raw, {"layout": name, "dtype": "u8_bitplanes", "shape": [n, c], "source_map": "zigzag_i8_to_u8"}, dec

    raise ValueError(name)


def sweep(state: ResidualState, codecs: List[str], candidate_names: List[str]) -> Tuple[List[dict], Dict[str, Callable]]:
    results = []
    decoders = {}
    for cname in candidate_names:
        try:
            raw, meta, decoder = encode_candidate_raw(cname, state)
            exact_layout = bool(np.array_equal(decoder(raw, meta, state), state.residual))
            decoders[cname] = decoder
        except Exception as e:
            results.append({"candidate": cname, "codec": None, "error": str(e), "ok": False})
            continue
        for codec in codecs:
            if codec.startswith("brotli") and brotli is None:
                continue
            t0 = time.perf_counter()
            try:
                payload = codec_compress(codec, raw)
                elapsed = time.perf_counter() - t0
                rt = codec_decompress(codec, payload)
                exact = bool(exact_layout and rt == raw)
                results.append({
                    "candidate": cname,
                    "codec": codec,
                    "compressed_bytes": len(payload),
                    "raw_bytes": len(raw),
                    "ratio_raw_to_compressed": len(raw) / max(len(payload), 1),
                    "seconds": elapsed,
                    "exact_reversible_layout": exact_layout,
                    "codec_roundtrip_exact": exact,
                    "meta": meta,
                    "ok": exact,
                })
            except Exception as e:
                results.append({"candidate": cname, "codec": codec, "error": str(e), "ok": False})
    results.sort(key=lambda d: d.get("compressed_bytes", 10**18))
    return results, decoders


def write_best_archive_if_real(state: ResidualState, best: dict, out_path: Path) -> Optional[int]:
    if state.base_manifest is None or state.base_payloads is None:
        return None
    raw, meta, decoder = encode_candidate_raw(best["candidate"], state)
    payload = codec_compress(best["codec"], raw)
    decoded = decoder(codec_decompress(best["codec"], payload), meta, state)
    recon = (state.approx_q8.astype(np.int16) + decoded.astype(np.int16)).clip(-128, 127).astype(np.int8)
    exact = bool(np.array_equal(recon, state.sh_q8))
    if not exact:
        raise RuntimeError("Refusing to write archive: winning residual did not reconstruct q8 SH exactly")
    base_chunks = [dict(c) for c in state.base_manifest["chunks"]]
    corr_chunk = {
        "name": "v29_best_sh_exact_residual_payload",
        "semantic": "Exact q8 SH residual correction: original_q8 = VQ_render_core + decoded_residual",
        "candidate": best["candidate"],
        "codec": best["codec"],
        "raw_bytes": len(raw),
        "compressed_bytes": len(payload),
        "crc32_raw": binascii.crc32(raw) & 0xffffffff,
        "meta": meta,
    }
    manifest = {
        "format": "CRYPSOID_3DPHOX_V29_RENDER_GATED_RESIDUAL_DEBT_BURNDOWN",
        "cycle": "v0.29",
        "source": state.source,
        "chunks": base_chunks + [corr_chunk],
        "correction_contract": {
            "archive_exact_q8_sh": exact,
            "residual_dtype": str(state.residual.dtype),
            "residual_min": int(state.residual.min()),
            "residual_max": int(state.residual.max()),
            "winner": {k: v for k, v in best.items() if k != "meta"},
            "truth_note": "Render core remains VQ approximate; archive mode carries the exact correction payload.",
        },
    }
    payloads = list(state.base_payloads) + [payload]
    return write_container(out_path, manifest, payloads)


def make_svg_bar(results: List[dict], out: Path, baseline_name: str = "splat_major_raw"):
    good = [r for r in results if r.get("ok") and r.get("compressed_bytes") is not None]
    top = good[:12]
    if not top:
        out.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='800' height='80'><text x='20' y='40'>No successful candidates</text></svg>")
        return
    maxb = max(r["compressed_bytes"] for r in top)
    h = 54 + 34 * len(top)
    lines = [f"<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='{h}'><rect width='100%' height='100%' fill='white'/><style>text{{font-family:monospace;font-size:13px}}</style><text x='20' y='25'>CRYPSOID v0.29 residual transform sweep — top exact candidates</text>"]
    y = 54
    for r in top:
        w = 640 * r["compressed_bytes"] / maxb
        label = f"{r['candidate']} / {r['codec']}"
        lines.append(f"<text x='20' y='{y}'>{label}</text><rect x='430' y='{y-15}' width='{w:.1f}' height='18' fill='#222'/><text x='{440+w:.1f}' y='{y}'>{r['compressed_bytes']/1024/1024:.3f} MiB</text>")
        y += 34
    lines.append("</svg>")
    out.write_text("\n".join(lines))


def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v25", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/outputs/v25_attribute_group_render_container.3dphox"))
    ap.add_argument("--v27", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27/outputs/v27_attribute_group_sh_vq_render_container.3dphox"))
    ap.add_argument("--v25-report", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json"))
    ap.add_argument("--v27-report", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27/reports/PHOXBENCH_V27_SH_DEBT_REPORT.json"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_29"))
    ap.add_argument("--synthetic-n", type=int, default=80000)
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--fast", action="store_true", help="Use zlib codecs only; useful on phone/Termux.")
    args = ap.parse_args()

    out = args.out
    reports = out / "reports"; outputs = out / "outputs"; tools = out / "tools"
    for d in [reports, outputs, tools]:
        d.mkdir(parents=True, exist_ok=True)

    real_ready = all(p.exists() for p in [args.v25, args.v27, args.v25_report, args.v27_report]) and not args.force_synthetic
    missing = [str(p) for p in [args.v25, args.v27, args.v25_report, args.v27_report] if not p.exists()]
    if real_ready:
        state = load_real_state(args.v25, args.v27, args.v25_report, args.v27_report)
    else:
        state = make_synthetic_state(args.synthetic_n)

    codecs = ["zlib6", "zlib9"] if args.fast else ["zlib6", "zlib9", "bz2", "lzma6"]
    if brotli is not None and not args.fast:
        codecs += ["brotli5", "brotli9"]

    candidate_names = [
        "splat_major_raw",
        "coefficient_major_transpose",
        "group_major_3x15",
        "band_split_low_mid_high",
        "morton_splat_major",
        "morton_delta_i16",
        "tier_then_coefficient_major",
        "zigzag_splat_major_u16",
        "sign_magnitude_planes",
        "zero_mask_values",
        "bitplane_zigzag_u8_if_safe",
    ]

    t0 = time.perf_counter()
    results, decoders = sweep(state, codecs, candidate_names)
    elapsed = time.perf_counter() - t0
    good = [r for r in results if r.get("ok") and r.get("compressed_bytes") is not None]
    best = good[0] if good else None

    archive_size = None
    archive_path = outputs / "v29_best_exact_archive_container.3dphox"
    if best and real_ready:
        archive_size = write_best_archive_if_real(state, best, archive_path)

    baseline = next((r for r in good if r["candidate"] == "splat_major_raw" and r["codec"] == "zlib6"), None)
    improvement_vs_baseline = None
    if best and baseline:
        improvement_vs_baseline = 100.0 * (1.0 - best["compressed_bytes"] / baseline["compressed_bytes"])

    report = {
        "cycle": "v0.29",
        "title": "Render-gated residual transform sweep / exact SH correction debt burndown",
        "status": "real Audi run" if real_ready else "synthetic smoke run because required real containers were not present",
        "truth_warning": None if real_ready else "Synthetic numbers validate the transform harness only; they are not compression claims for Audi or CRYPSOID quality.",
        "missing_real_inputs": missing,
        "source": state.source,
        "residual_stats": {
            "shape": list(state.residual.shape),
            "dtype": str(state.residual.dtype),
            "min": int(state.residual.min()),
            "max": int(state.residual.max()),
            "zero_fraction": float(np.mean(state.residual == 0)),
            "mean_abs": float(np.mean(np.abs(state.residual.astype(np.int16)))),
            "rmse": float(np.sqrt(np.mean(state.residual.astype(np.float64) ** 2))),
        },
        "codecs": codecs,
        "candidate_count": len(candidate_names),
        "result_count": len(results),
        "elapsed_seconds": elapsed,
        "best": best,
        "baseline_zlib6_splat_major": baseline,
        "best_improvement_vs_baseline_percent": improvement_vs_baseline,
        "archive_output": str(archive_path) if archive_size else None,
        "archive_size_bytes": archive_size,
        "all_results": results,
        "next": {
            "gate_1": "Run this on the real v25/v27 containers and compare the winning payload against v0.28 global_full 12.25 MiB.",
            "gate_2": "Only promote a context split if it beats the global stream; v0.28 proved semantic tiering can lose.",
            "gate_3": "Then render original/v28/v29 from the same camera path and attach PSNR/SSIM/contact sheet before any further primitive work.",
        },
    }
    save_json(reports / "PHOXBENCH_V29_RESIDUAL_TRANSFORM_SWEEP_REPORT.json", report)
    make_svg_bar(results, outputs / "v29_residual_transform_sweep_top.svg")

    # Markdown report
    lines = []
    lines.append("# CRYPSOID v0.29 — Render-Gated Residual Debt Burndown")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    if real_ready:
        lines.append("Real v25/v27 containers were present. v0.29 swept exact residual transform layouts and wrote the best exact archive candidate.")
    else:
        lines.append("The actual v25/v27 binary containers were not present in this workspace, so this run executed the v0.29 harness on a synthetic residual field only.")
        lines.append("")
        lines.append("This is **not** an Audi compression result. It is a correctness test for the next real pass.")
        lines.append("")
        lines.append("Missing real inputs:")
        for m in missing:
            lines.append(f"- `{m}`")
    lines.append("")
    lines.append("## Best result from this run")
    lines.append("")
    if best:
        lines.append(f"- Winner: `{best['candidate']}` with `{best['codec']}`")
        lines.append(f"- Payload: `{best['compressed_bytes']:,}` bytes / `{best['compressed_bytes']/1024/1024:.3f}` MiB")
        if improvement_vs_baseline is not None:
            lines.append(f"- Improvement vs splat-major zlib6 baseline: `{improvement_vs_baseline:.2f}%`")
    else:
        lines.append("No valid candidate produced an exact round trip.")
    lines.append("")
    lines.append("## Why this phase matters")
    lines.append("")
    lines.append("v0.28 showed that exact SH correction debt, not geometry storage, is now the bottleneck. v0.29 therefore tests byte-level residual orderings before inventing more semantic structure.")
    lines.append("")
    lines.append("## Promotion rule")
    lines.append("")
    lines.append("A transform only advances if it reconstructs q8 SH exactly and beats the v0.28 global correction payload. Context-aware/semantic splits are rejected when they compress worse than the global stream.")
    lines.append("")
    lines.append("## Next phase")
    lines.append("")
    lines.append("v0.30 should add a real render gate: original PLY, v28 render core, and v29 exact/archive decode from the same camera path, then compare contact sheets and PSNR/SSIM before any native phoxoid/SARC work resumes.")
    (reports / "RESEARCH_BUILD_TEST_CYCLE_V29.md").write_text("\n".join(lines))

    print(json.dumps({
        "real_ready": real_ready,
        "missing": missing,
        "best": best,
        "report": str(reports / "PHOXBENCH_V29_RESIDUAL_TRANSFORM_SWEEP_REPORT.json"),
        "archive": str(archive_path) if archive_size else None,
        "svg": str(outputs / "v29_residual_transform_sweep_top.svg"),
    }, indent=2))

if __name__ == "__main__":
    main()
