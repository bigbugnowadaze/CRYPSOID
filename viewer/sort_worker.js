// Background depth-sort worker for the .3dphox viewer.
//
// Receives:  { type: "init", xyz: Float32Array }       — splat positions, set once
//            { type: "sort", view: Float32Array }       — 4x4 view matrix, sort to it
// Replies:   { type: "order", order: Uint32Array }      — splat indices, back-to-front
//
// Uses a counting-radix-style 16-bit sort on quantized depth keys for speed.

let positions = null;
let n = 0;
let depths = null;
let order = null;

self.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "init") {
        positions = msg.xyz;
        n = positions.length / 3;
        depths = new Float32Array(n);
        order = new Uint32Array(n);
        for (let i = 0; i < n; i++) order[i] = i;
        return;
    }
    if (msg.type === "sort") {
        if (!positions) return;
        const v = msg.view;  // 4x4 column-major (OpenGL convention)
        // We only need the third row of view (camera-Z direction in world).
        // For column-major: v[0+2*4]=v[8], v[1+2*4]=v[9], v[2+2*4]=v[10], v[3+2*4]=v[14] (translation)
        const r2x = v[2], r2y = v[6], r2z = v[10], r2w = v[14];
        // Compute depth_z = r2x * x + r2y * y + r2z * z + r2w (negative-z = forward in view space)
        let zmin = +Infinity, zmax = -Infinity;
        for (let i = 0; i < n; i++) {
            const d = r2x * positions[i*3] + r2y * positions[i*3+1] + r2z * positions[i*3+2] + r2w;
            depths[i] = d;
            if (d < zmin) zmin = d;
            if (d > zmax) zmax = d;
        }
        // Quantize depths to 16-bit keys (back-to-front order = ascending depth in view space, MORE NEGATIVE = closer)
        // We want farthest splats first (larger d in our view convention -> closer? — depends on view.)
        // Actually for additive blending the order doesn't matter for the value, but for proper "over" compositing
        // we want closest first (front-to-back). We'll sort ASCENDING by d (most negative = farthest in standard
        // RH OpenGL view; flip if needed by the renderer).
        const span = zmax - zmin || 1;
        const counts = new Uint32Array(65537);
        const keys = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
            const k = Math.min(65535, Math.max(0, ((depths[i] - zmin) / span * 65535) | 0));
            keys[i] = k;
            counts[k+1]++;
        }
        for (let i = 1; i < 65537; i++) counts[i] += counts[i-1];
        const sorted = new Uint32Array(n);
        for (let i = 0; i < n; i++) {
            const k = keys[i];
            sorted[counts[k]++] = i;
        }
        // Post the result. Transfer ownership for zero-copy.
        self.postMessage({ type: "order", order: sorted }, [sorted.buffer]);
    }
};
