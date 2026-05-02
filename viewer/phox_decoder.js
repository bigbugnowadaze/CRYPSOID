// CRYPSOID .3dphox decoder for the browser.
//
// Reads any of:  v25 (CRYPSOID25), v27 (CRYPSOID27), v28 render (CRYPSOID28
// non-EXACT format), v28 EXACT archive (CRYPSOID28 EXACT format).
//
// All chunks are zlib-compressed; we decompress with the browser's native
// DecompressionStream API (Chrome 80+, Firefox 113+, Safari 16.4+).
// No third-party dependencies.

const SH_C0 = 0.28209479177387814;
const GLOBAL_SCALE = 0.006946287755891094;

// ---------- low-level decoders ----------

async function inflate(u8) {
    // u8 is Uint8Array of zlib-compressed bytes
    const ds = new DecompressionStream("deflate");
    const blob = new Blob([u8]);
    const decompressedStream = blob.stream().pipeThrough(ds);
    const buf = await new Response(decompressedStream).arrayBuffer();
    return new Uint8Array(buf);
}

function readU64LE(view, off) {
    const lo = view.getUint32(off, true);
    const hi = view.getUint32(off + 4, true);
    return hi * 0x100000000 + lo;
}

// ---------- container parsing ----------

export async function parsePhox(arrayBuffer) {
    const u8 = new Uint8Array(arrayBuffer);
    const dv = new DataView(arrayBuffer);

    // First 11 bytes: magic
    const magic = new TextDecoder("latin1").decode(u8.slice(0, 11));
    if (!magic.startsWith("CRYPSOID")) throw new Error("not a CRYPSOID file: " + magic);
    const version = magic.replace(/[^\d]/g, "");

    // Next 8 bytes: little-endian uint64 manifest length
    const mlen = readU64LE(dv, 11);
    const manifestBytes = u8.slice(19, 19 + mlen);
    const manifest = JSON.parse(new TextDecoder("utf-8").decode(manifestBytes));

    const blob = u8.subarray(19 + mlen);
    const chunks = {};
    for (const c of manifest.chunks) chunks[c.name] = c;

    async function decompressChunk(name) {
        const c = chunks[name];
        if (!c) return null;
        const compressed = blob.subarray(c.offset, c.offset + c.compressed_bytes);
        return await inflate(compressed);
    }

    return { magic, version, manifest, chunks, decompressChunk };
}

// ---------- per-attribute decoders ----------

export function decodeU24Xyz(raw, n, boundsMin, boundsMax) {
    // raw is Uint8Array of length n*9 (3 u24 LE values per splat)
    const out = new Float32Array(n * 3);
    const span = [
        boundsMax[0] - boundsMin[0],
        boundsMax[1] - boundsMin[1],
        boundsMax[2] - boundsMin[2],
    ];
    const u24max = (1 << 24) - 1;
    let r = 0, w = 0;
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = raw[r] | (raw[r + 1] << 8) | (raw[r + 2] << 16);
            out[w + a] = (v / u24max) * span[a] + boundsMin[a];
            r += 3;
        }
        w += 3;
    }
    return out;
}

export function decodeF16Scales(raw, n) {
    // raw is Uint8Array of length n*6 (3 IEEE-754 float16 per splat)
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) out[i] = float16ToFloat32(view.getUint16(i * 2, true));
    return out;
}

export function decodeI16Quats(raw, n) {
    const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n * 4; i++) out[i] = view.getInt16(i * 2, true) / 32767;
    return out;
}

export function decodeDcRgbOpacityU8(raw, n) {
    // raw is n*4 bytes: R8, G8, B8, opacity8
    const dc = new Float32Array(n * 3);
    const op = new Float32Array(n);
    let r = 0;
    for (let i = 0; i < n; i++) {
        dc[i * 3] = raw[r] / 255;
        dc[i * 3 + 1] = raw[r + 1] / 255;
        dc[i * 3 + 2] = raw[r + 2] / 255;
        op[i] = raw[r + 3] / 255;
        r += 4;
    }
    return { dc, opacity: op };
}

export function decodeTierLabels(raw, n) {
    return raw.subarray(0, n);   // already uint8
}

// ---------- SH reconstruction ----------

export function reconstructShVq128(idxRaw, codebookRaw, n, globalScale = GLOBAL_SCALE) {
    // idxRaw: n*3 uint8 (3 product-VQ groups)
    // codebookRaw: 3*128*15 int8 (3 groups x 128 codewords x 15 coefficients)
    const cb = new Int8Array(codebookRaw.buffer, codebookRaw.byteOffset, codebookRaw.byteLength);
    const idx = new Uint8Array(idxRaw.buffer, idxRaw.byteOffset, idxRaw.byteLength);
    const out = new Float32Array(n * 45);
    for (let i = 0; i < n; i++) {
        for (let g = 0; g < 3; g++) {
            const label = idx[i * 3 + g];
            const cbBase = g * 128 * 15 + label * 15;
            const outBase = i * 45 + g * 15;
            for (let k = 0; k < 15; k++) out[outBase + k] = cb[cbBase + k] * globalScale;
        }
    }
    return out;
}

export async function reconstructShExactArchive(decompressChunk, chunks, n, tierLabels) {
    // Same as reconstructShVq128, then ADD per-tier-group residual chunks.
    const idx = await decompressChunk("sh_vq128_idx_u8");
    const cb = await decompressChunk("sh_vq128_codebook_i8");
    const cbSigned = new Int8Array(cb.buffer, cb.byteOffset, cb.byteLength);
    const idxU8 = new Uint8Array(idx.buffer, idx.byteOffset, idx.byteLength);

    // Reconstruct as int16 first so we can add residuals in int space without clipping early.
    const sh = new Int16Array(n * 45);
    for (let i = 0; i < n; i++) {
        for (let g = 0; g < 3; g++) {
            const label = idxU8[i * 3 + g];
            const cbBase = g * 128 * 15 + label * 15;
            const outBase = i * 45 + g * 15;
            for (let k = 0; k < 15; k++) sh[outBase + k] = cbSigned[cbBase + k];
        }
    }

    // Per-tier indices, in their original order
    const tierIndices = [[], [], []];
    for (let i = 0; i < n; i++) tierIndices[tierLabels[i]].push(i);

    for (let t = 0; t < 3; t++) {
        for (let g = 0; g < 3; g++) {
            const name = `sh_exact_residual_t${t}_g${g}_int8`;
            if (!chunks[name]) continue;
            const raw = await decompressChunk(name);
            const res = new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength);
            const expected = tierIndices[t].length * 15;
            if (res.length !== expected)
                throw new Error(`${name}: expected ${expected} bytes, got ${res.length}`);
            for (let r = 0; r < tierIndices[t].length; r++) {
                const splatIdx = tierIndices[t][r];
                const outBase = splatIdx * 45 + g * 15;
                const inBase = r * 15;
                for (let k = 0; k < 15; k++) {
                    let v = sh[outBase + k] + res[inBase + k];
                    if (v < -128) v = -128;
                    if (v > 127) v = 127;
                    sh[outBase + k] = v;
                }
            }
        }
    }
    // Convert int8-clipped values to float using global scale
    const out = new Float32Array(n * 45);
    for (let i = 0; i < n * 45; i++) out[i] = sh[i] * GLOBAL_SCALE;
    return out;
}

// ---------- entry point ----------

export async function loadPhoxAll(arrayBuffer) {
    const phox = await parsePhox(arrayBuffer);
    const n = phox.manifest.source_splats || phox.chunks.xyz_u24_fixed.shape[0];

    const xyzChunk = phox.chunks.xyz_u24_fixed;
    const xyzRaw = await phox.decompressChunk("xyz_u24_fixed");
    const xyz = decodeU24Xyz(xyzRaw, n, xyzChunk.bounds_min, xyzChunk.bounds_max);

    const scales = decodeF16Scales(await phox.decompressChunk("scale_f16"), n);
    const quats = decodeI16Quats(await phox.decompressChunk("quat_i16_norm4"), n);
    const dco = decodeDcRgbOpacityU8(await phox.decompressChunk("dc_rgb_opacity_u8"), n);
    const tier = decodeTierLabels(await phox.decompressChunk("tier_labels_u8"), n);

    let shRest = null;
    const isExact = (phox.manifest.format || "").includes("EXACT_ARCHIVE");
    if (isExact && phox.chunks.sh_vq128_idx_u8 && phox.chunks.sh_vq128_codebook_i8) {
        shRest = await reconstructShExactArchive(phox.decompressChunk, phox.chunks, n, tier);
    } else if (phox.chunks.sh_vq128_idx_u8 && phox.chunks.sh_vq128_codebook_i8) {
        const idx = await phox.decompressChunk("sh_vq128_idx_u8");
        const cb = await phox.decompressChunk("sh_vq128_codebook_i8");
        shRest = reconstructShVq128(idx, cb, n);
    } else if (phox.chunks.sh_rest_q8_global) {
        // v25 format: raw int8 SH stream
        const raw = await phox.decompressChunk("sh_rest_q8_global");
        const i8 = new Int8Array(raw.buffer, raw.byteOffset, raw.byteLength);
        shRest = new Float32Array(n * 45);
        for (let i = 0; i < n * 45; i++) shRest[i] = i8[i] * GLOBAL_SCALE;
    }

    return {
        n,
        magic: phox.magic, format: phox.manifest.format,
        xyz, scales, quats, dc: dco.dc, opacity: dco.opacity,
        sh_rest: shRest, tier,
        bounds_min: xyzChunk.bounds_min, bounds_max: xyzChunk.bounds_max,
        is_exact: isExact,
    };
}

// ---------- helpers ----------

function float16ToFloat32(h) {
    const sign = (h & 0x8000) >> 15;
    const exp  = (h & 0x7c00) >> 10;
    const frac =  h & 0x03ff;
    if (exp === 0) return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
    if (exp === 0x1f) return frac ? NaN : (sign ? -Infinity : Infinity);
    return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}


// ---------- v31 trailer parsing ----------
// A v31-versioned .3dphox is: [v28 archive bytes] + [CRYPSOID31\0] + [u64 manifest_len] + [JSON manifest] + [chunks]
// Backward-compatible: v28 readers stop at the v28 region; v31 readers detect the trailer marker.

const V31_MAGIC = "CRYPSOID31\x00";

export function parseV31Trailer(arrayBuffer) {
    const u8 = new Uint8Array(arrayBuffer);
    const dv = new DataView(arrayBuffer);
    // Search backwards for the trailer marker (last 30 MB at most — typically near end of file)
    const sentinel = new TextEncoder().encode(V31_MAGIC);
    let pos = -1;
    // scan from later half, byte-by-byte (small enough since file is tens of MB)
    const scanFrom = Math.max(0, u8.length - 50 * 1024 * 1024);
    outer: for (let i = u8.length - sentinel.length; i >= scanFrom; i--) {
        for (let j = 0; j < sentinel.length; j++) {
            if (u8[i + j] !== sentinel[j]) continue outer;
        }
        pos = i; break;
    }
    if (pos < 0) return null;   // no v31 trailer
    let p = pos + sentinel.length;
    const mlen = readU64LE(dv, p); p += 8;
    const manifestBytes = u8.slice(p, p + mlen);
    const manifest = JSON.parse(new TextDecoder("utf-8").decode(manifestBytes));
    p += mlen;
    const chunkRegion = u8.slice(p);
    const chunks = {};
    for (const c of manifest.chunks) {
        const start = c.offset_in_trailer;
        const end = start + c.size_bytes;
        chunks[c.name] = chunkRegion.slice(start, end);
    }
    return { magic: V31_MAGIC, manifest, chunks };
}


// ---------- v31 chunk decoders ----------

// Octahedral 24-bit + 8-bit tangent → unit normal
export function decodeNormalsChunk(chunkBytes) {
    // Header: 1 B version + 1 B reserved + 4 B count + N*4 payload + 4 B CRC
    const dv = new DataView(chunkBytes.buffer, chunkBytes.byteOffset, chunkBytes.byteLength);
    const version = chunkBytes[0];
    if (version !== 0x01) throw new Error("normals chunk version " + version);
    const n = dv.getUint32(2, true);
    const expected = 6 + n*4 + 4;
    if (chunkBytes.length !== expected) throw new Error("normals chunk len " + chunkBytes.length + " != " + expected);
    const payload = chunkBytes.subarray(6, 6 + n*4);
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const b0 = payload[i*4 + 0];
        const b1 = payload[i*4 + 1];
        const b2 = payload[i*4 + 2];
        // unpack 12-bit qx + 12-bit qy
        const qx = ((b1 & 0x0F) << 8) | b0;
        const qy = (b2 << 4) | ((b1 >> 4) & 0x0F);
        let x = (qx / 4095.0) * 2.0 - 1.0;
        let y = (qy / 4095.0) * 2.0 - 1.0;
        let z = 1.0 - Math.abs(x) - Math.abs(y);
        if (z < 0.0) {
            const xS = x >= 0 ? 1 : -1;
            const yS = y >= 0 ? 1 : -1;
            const xT = (1.0 - Math.abs(y)) * xS;
            const yT = (1.0 - Math.abs(x)) * yS;
            x = xT; y = yT;
        }
        const len = Math.sqrt(x*x + y*y + z*z);
        out[i*3 + 0] = x / len;
        out[i*3 + 1] = y / len;
        out[i*3 + 2] = z / len;
    }
    return out;
}

// 4 bytes/blob: hint + confidence + view_dep + mip_zoom
export function decodeMaterialChunk(chunkBytes) {
    const dv = new DataView(chunkBytes.buffer, chunkBytes.byteOffset, chunkBytes.byteLength);
    const version = chunkBytes[0];
    if (version !== 0x01) throw new Error("material chunk version " + version);
    const fields = chunkBytes[1];
    if (fields !== 0x04) throw new Error("expected 4 fields, got " + fields);
    const n = dv.getUint32(2, true);
    const expected = 6 + n*4 + 4;
    if (chunkBytes.length !== expected) throw new Error("material chunk len " + chunkBytes.length + " != " + expected);
    const payload = chunkBytes.subarray(6, 6 + n*4);
    const hint = new Uint8Array(n);
    const confidence = new Uint8Array(n);
    const viewDep = new Uint8Array(n);
    const mipZoom = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        hint[i]       = payload[i*4 + 0];
        confidence[i] = payload[i*4 + 1];
        viewDep[i]    = payload[i*4 + 2];
        mipZoom[i]    = payload[i*4 + 3];
    }
    return { hint, confidence, viewDep, mipZoom };
}

// kNN edges: 16 bytes/blob (4 × u32)
export function decodeEdgesChunk(chunkBytes) {
    const dv = new DataView(chunkBytes.buffer, chunkBytes.byteOffset, chunkBytes.byteLength);
    const version = chunkBytes[0];
    if (version !== 0x01) throw new Error("edges chunk version " + version);
    const k = chunkBytes[1];
    const n = dv.getUint32(2, true);
    const expected = 6 + n*k*4 + 4;
    if (chunkBytes.length !== expected) throw new Error("edges chunk len " + chunkBytes.length + " != " + expected);
    const payload = chunkBytes.subarray(6, 6 + n*k*4);
    return { neighbors: new Uint32Array(payload.buffer, payload.byteOffset, n*k), k };
}
