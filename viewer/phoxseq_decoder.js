// v34 .phoxseq decoder for the WebGL viewer.
// Mirrors the Python phoxseq_codec.py format, but only the parts a viewer needs:
//   - parseV34PhoxSeq(arrayBuffer) -> { baseCrc, baseN, frames[], fpsMilli, timeStartMs, timeEndMs }
//   - applyFrameToBuffers(scene, frame) — applies a frame's phoxdelta to per-splat arrays in place
//
// Format reference: docs/v34_phoxseq_spec.md.

const SEQ_MAGIC_BYTES = [0x50, 0x48, 0x4f, 0x58, 0x53, 0x45, 0x51, 0x00]; // "PHOXSEQ\0"
const PHOXDELTA_MAGIC_BYTES = [0x50, 0x48, 0x4f, 0x58, 0x44, 0x4c, 0x54, 0x00]; // "PHOXDLT\0"

const FLAG_COMPRESSED = 0x01;
const HEADER_SIZE = 40;
const FRAME_INDEX_SIZE = 16;

function magicMatches(view, offset, magic) {
  for (let i = 0; i < magic.length; i++) {
    if (view.getUint8(offset + i) !== magic[i]) return false;
  }
  return true;
}

async function inflateZlib(bytes) {
  // Use the browser's DecompressionStream API.
  const blob = new Blob([bytes]);
  const ds = new DecompressionStream('deflate');
  const decompressed = blob.stream().pipeThrough(ds);
  const buf = await new Response(decompressed).arrayBuffer();
  return new Uint8Array(buf);
}

// ---------- Phoxdelta inner parser (per frame) ----------

const BIT_LAYOUT = [
  // [bit, name, dtype, n_elements per record]
  [0, 'xyz',     'float32', 3],
  [1, 'scale',   'float32', 3],
  [2, 'quat',    'float32', 4],
  [3, 'opacity', 'float32', 1],
  [4, 'f_dc',    'float32', 3],
  [5, 'f_rest',  'float32', 45],
  [6, 'tier',    'uint8',   1],
  [7, 'germ',    'float32', 5],
  [8, 'normal',  'uint8',   4],
];
const BIT_NEIGHBORS = 9;

function dtypeBytes(dt) {
  switch (dt) {
    case 'float32': return 4;
    case 'uint8':   return 1;
    case 'uint32':  return 4;
    default: throw new Error(`unknown dtype ${dt}`);
  }
}

function decodePhoxdelta(bytes, kNeighbors = 4) {
  if (bytes.length < 24) throw new Error(`phoxdelta too short: ${bytes.length}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < 8; i++)
    if (view.getUint8(i) !== PHOXDELTA_MAGIC_BYTES[i])
      throw new Error('phoxdelta bad magic');
  const version = view.getUint8(8);
  if (version !== 0x01) throw new Error(`phoxdelta version ${version}`);
  const baseCrc = view.getUint32(12, true);
  const baseN = view.getUint32(16, true);
  const M = view.getUint32(20, true);

  let p = 24;
  const ids = new Uint32Array(M);
  const masks = new Uint16Array(M);
  const perAttrRecords = {};

  for (let i = 0; i < M; i++) {
    ids[i] = view.getUint32(p, true); p += 4;
    masks[i] = view.getUint16(p, true); p += 2;
    const m = masks[i];
    for (const [bit, name, dt, n] of BIT_LAYOUT) {
      if (m & (1 << bit)) {
        const sz = dtypeBytes(dt) * n;
        let arr;
        if (dt === 'float32') arr = new Float32Array(bytes.buffer.slice(bytes.byteOffset + p, bytes.byteOffset + p + sz));
        else if (dt === 'uint8') arr = new Uint8Array(bytes.buffer.slice(bytes.byteOffset + p, bytes.byteOffset + p + sz));
        else if (dt === 'uint32') arr = new Uint32Array(bytes.buffer.slice(bytes.byteOffset + p, bytes.byteOffset + p + sz));
        if (!perAttrRecords[name]) perAttrRecords[name] = { idxs: [], values: [] };
        perAttrRecords[name].idxs.push(i);
        perAttrRecords[name].values.push(arr);
        p += sz;
      }
    }
    if (m & (1 << BIT_NEIGHBORS)) {
      const sz = kNeighbors * 4;
      const nb = new Uint32Array(bytes.buffer.slice(bytes.byteOffset + p, bytes.byteOffset + p + sz));
      if (!perAttrRecords['neighbors']) perAttrRecords['neighbors'] = { idxs: [], values: [] };
      perAttrRecords['neighbors'].idxs.push(i);
      perAttrRecords['neighbors'].values.push(nb);
      p += sz;
    }
  }

  return { baseCrc, baseN, M, phoxoidIds: ids, dirtyMask: masks, attrs: perAttrRecords, kNeighbors };
}

// ---------- Sequence parser ----------

export async function parseV34PhoxSeq(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (view.byteLength < HEADER_SIZE) throw new Error(`phoxseq too short: ${view.byteLength}`);
  if (!magicMatches(view, 0, SEQ_MAGIC_BYTES)) throw new Error('phoxseq bad magic');
  const version = view.getUint8(8);
  if (version !== 0x01) throw new Error(`phoxseq version ${version}`);
  const baseCrc      = view.getUint32(12, true);
  const baseN        = view.getUint32(16, true);
  const frameCount   = view.getUint32(20, true);
  const fpsMilli     = view.getUint32(24, true);
  const timeStartMs  = view.getInt32(28, true);
  const timeEndMs    = view.getInt32(32, true);

  let p = HEADER_SIZE;
  const frames = [];
  for (let i = 0; i < frameCount; i++) {
    const t      = view.getInt32(p, true);
    const flags  = view.getUint16(p + 4, true);
    const offset = view.getUint32(p + 8, true);
    const size   = view.getUint32(p + 12, true);
    let payload = new Uint8Array(arrayBuffer, offset, size);
    if (flags & FLAG_COMPRESSED) {
      payload = await inflateZlib(payload);
    }
    const delta = decodePhoxdelta(payload);
    frames.push({ timeOffsetMs: t, flags, delta });
    p += FRAME_INDEX_SIZE;
  }
  return { baseCrc, baseN, frameCount, fpsMilli, timeStartMs, timeEndMs, frames };
}

// ---------- Apply a frame's phoxdelta to in-memory per-splat buffers ----------
// scene is a {xyz, opacities, scales, quats, sh_dc, ...} object.
// Returns nothing — mutates the scene buffers in place. Save originals first if you
// want to scrub backwards.

export function applyFrameToScene(scene, frame, originals = null) {
  const { phoxoidIds, attrs } = frame.delta;

  const fieldMap = {
    xyz:     'xyz',
    scale:   'scales',
    quat:    'quats',
    opacity: 'opacity',     // viewer's scene uses singular 'opacity'
    f_dc:    'dc',          // viewer uses 'dc' for DC
    f_rest:  'sh_rest',
    tier:    'tier',
  };

  for (const [attrName, fieldName] of Object.entries(fieldMap)) {
    if (!attrs[attrName]) continue;
    const target = scene[fieldName];
    if (!target) continue;
    const idxs = attrs[attrName].idxs;
    const vals = attrs[attrName].values;
    for (let i = 0; i < idxs.length; i++) {
      const pid = phoxoidIds[idxs[i]];
      const v = vals[i];
      // Determine stride per record from buffer length (depends on field)
      const stride = v.length;
      const dest = target;
      for (let k = 0; k < stride; k++) {
        dest[pid * stride + k] = v[k];
      }
    }
  }
}

// Apply ALL frames with timeOffsetMs <= t_ms cumulatively. Resets state to
// baseScene each call by copying originals — NOT efficient for scrubbing
// backwards through long sequences, but correct.

export function applyFramesUpToTime(scene, originals, seq, tMs) {
  // Restore originals first
  for (const k of Object.keys(originals)) {
    if (scene[k] && originals[k]) scene[k].set(originals[k]);
  }
  for (const f of seq.frames) {
    if (f.timeOffsetMs > tMs) break;
    applyFrameToScene(scene, f);
  }
}
