import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { setFlagsFromString } from 'node:v8';
import { runInNewContext } from 'node:vm';

// showcase-car-render.mjs's addCar on a stand-in device (no GPU): the bytes it uploads
// for a weighted-skin visual, and what it keeps on the CPU once they are uploaded. A
// generated car's skin is hundreds of MB; a copy kept alive past its upload is a phone
// tab's memory (engine/docs/measurements/lab-load-memory.md).
setFlagsFromString('--expose-gc');
const gc = runInNewContext('gc');

Object.assign(globalThis, {
  GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
  GPUTextureUsage: { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  GPUMapMode: { READ: 1, WRITE: 2 },
});
const { createShowcaseCarRenderer } = await import('../showcase-car-render.mjs');

// Anything the renderer asks of a device: buffers keep their label; writeBuffer keeps a
// weak reference to what it was given and a copy of its bytes.
function fakeDevice() {
  const buffers = [];
  const any = () => new Proxy(function () {}, { get: (_, k) => (k === 'then' ? undefined : any()), apply: () => any() });
  const writes = [];
  const queue = {
    writeBuffer(buffer, offset, data, dataOffset = 0, size) {
      const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
      const bytes = new Uint8Array(src.buffer, src.byteOffset + dataOffset * (src.BYTES_PER_ELEMENT ?? 1), size !== undefined ? size * (src.BYTES_PER_ELEMENT ?? 1) : src.byteLength - dataOffset * (src.BYTES_PER_ELEMENT ?? 1));
      buffer.image.set(bytes, offset);
      writes.push({ label: buffer.label, offset, size: bytes.byteLength, source: new WeakRef(ArrayBuffer.isView(data) ? data.buffer : data) });
    },
    writeTexture() {},
    submit() {},
  };
  const device = new Proxy({}, {
    get: (_, k) => {
      if (k === 'queue') return queue;
      if (k === 'createBuffer') return (d) => { const b = { label: d.label, size: d.size, usage: d.usage, image: new Uint8Array(d.size), destroy() {} }; buffers.push(b); return b; };
      if (k === 'then') return undefined;
      return any();
    },
  });
  // A buffer's contents once every write has landed (the first with that label).
  const image = (label) => buffers.find((b) => b.label === label).image;
  return { device, writes, image };
}

// A small weighted-skin visual in the wasm's format (showcase.rs visual_js): three nodes,
// one triangle, each vertex on all three nodes with weights on the rskin grids.
function skinnedVisual() {
  const n = 3;
  return {
    nodeCount: 3, vertexCount: n,
    normalTris: new Uint32Array([0, 1, 2]), nodeTriStart: new Uint32Array([0, 1, 2, 3]), nodeTriList: new Uint32Array([0, 0, 0]),
    nodeBeamStart: new Uint32Array([0, 0, 0, 0]), nodeBeamList: new Uint32Array(0), beamRest0: new Float32Array(0),
    nodeRestNormal: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]), nodeRest: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]), normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array(2 * n), colors: new Float32Array([0.6, 0.04, 0.03, 0.6, 0.04, 0.03, 0.6, 0.04, 0.03]), ao: new Float32Array([1, 1, 1]),
    material: new Uint8Array(n), indices: new Uint32Array([0, 1, 2]),
    bindNodes: new Uint32Array([0, 1, 2, 1, 2, 0, 2, 0, 1]), bindCoords: new Float32Array(3 * n), bindNormal: new Float32Array(3 * n), bindRest: new Float32Array([1, 1, 1, 1, 1, 1]),
    smooth: new Uint8Array(n), rigid: new Uint8Array(n),
    wtStart: new Uint32Array([0, 3, 6, 9]),
    wtNode: new Uint32Array([0, 1, 2, 0, 1, 2, 2, 1, 0]),
    wtW: new Float32Array([0.5, 0.25, -0.125, 0.25, 0.5, 0.0625, 0.25, 0.25, 1, 0.125, 0.75, 0, 0.375, 0.125, -0.5, 0.5, 0.125, 0.25, 0.25, 0, 0, 0.25, 1.5, 0.5, 0.5, -1, 0.25]),
    wtOff: new Float32Array([0.001, 0, 0, 0, 0.002, 0, 0, 0, 0.003]), wtNrm: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  };
}

const sha = (b) => createHash('sha256').update(b).digest('hex').slice(0, 16);

// Pinned from the renderer before the memory work (5b5a527f): what it uploads must not change.
test('a skinned car\'s uploads are the same bytes (binds, rest pose, attributes, indices)', () => {
  const { device, image } = fakeDevice();
  const r = createShowcaseCarRenderer(device, { format: 'bgra8unorm' });
  r.addCar(skinnedVisual());
  const got = Object.fromEntries(['showcase-binds', 'showcase-out', 'showcase-attrs', 'showcase-index', 'showcase-topology']
    .map((l) => [l, sha(image(l))]));
  assert.deepEqual(got, {
    'showcase-binds': '559f728b0c7b6e43',
    'showcase-out': 'ea471a5319892872',
    'showcase-attrs': '8fc4cb57bf764fe8',
    'showcase-index': 'ad5dc1478de06a4c',
    'showcase-topology': '79f11e5544822b94',
  });
});

// The weights as the sidecar stores them (showcase.rs visual_js when every one is on
// its grid): u16 nodes, i16 steps of wtGrid's (A, dA) steps.
const asSteps = (v) => {
  const grid = new Float32Array([2 ** -13, 2 ** -9]);
  const q = new Int16Array(v.wtW.length);
  for (let k = 0; k < q.length; k++) q[k] = v.wtW[k] / grid[k % 3 === 0 ? 0 : 1];
  return { ...v, wtNode: undefined, wtW: undefined, wtNode16: Uint16Array.from(v.wtNode), wtQ: q, wtGrid: grid };
};
const bindsOf = (visual) => {
  const { device, image } = fakeDevice();
  createShowcaseCarRenderer(device, { format: 'bgra8unorm' }).addCar(visual);
  return image('showcase-binds');
};

// Every weighted vertex's entries as SKIN_WGSL's wEntry reads them from a bind buffer
// (its CPU mirror, both layouts): per vertex (node, A, dA/dT, dA/dB as f32 bits) in
// the order the shader sums them. The step layout's decode is the shader's integer
// one: f32(s)'s bits with the exponent lowered by k.
function entries(bytes, n) {
  const u = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const f32 = new Float32Array(1), b32 = new Uint32Array(f32.buffer);
  const stepBits = (s16, k) => { const s = (s16 << 16) >> 16; if (s === 0) return 0; f32[0] = s; return (b32[0] - (k << 23)) >>> 0; };
  const out = [];
  for (let i = 0; i < n; i++) {
    const flags = u[16 * i + 15];
    if (!(flags & (1 << 25))) continue;
    const count = u[16 * i], at = u[16 * i + 3];
    for (let e = 0; e < count; e++) {
      if (flags & (1 << 27)) {
        const x = u[2 * (at + e)], y = u[2 * (at + e) + 1];
        out.push([i, x & 0xffff, stepBits(x >>> 16, 13), stepBits(y & 0xffff, 9), stepBits(y >>> 16, 9)]);
      } else {
        const w = 16 * (at + (e >> 2)) + 4 * (e & 3);
        out.push([i, u[w], u[w + 1], u[w + 2], u[w + 3]]);
      }
    }
  }
  return out;
}

// The step layout: 8 bytes an entry instead of 16 (and no 4-entry slots' padding), and
// every entry the shader reads is the same f32 bits as the f32 layout's.
test('a skin\'s weights as grid steps upload 8-byte entries that decode to the same f32 bits', () => {
  const v = skinnedVisual();
  const f = bindsOf(v), q = bindsOf(asSteps(v));
  assert.equal(sha(f), '559f728b0c7b6e43');
  assert.equal(sha(q), '2b78ff9751a374b8');
  assert.ok(q.byteLength < f.byteLength, `${q.byteLength} < ${f.byteLength}`);
  assert.deepEqual(entries(q, 3), entries(f, 3));
  assert.equal(entries(f, 3).length, 9);
});

// A grid other than the one SKIN_WGSL decodes: the f32 layout (never rounded).
test('weights on another grid keep the f32 layout', () => {
  const v = asSteps(skinnedVisual());
  v.wtGrid = new Float32Array([2 ** -12, 2 ** -9]);
  for (let k = 0; k < v.wtQ.length; k += 3) v.wtQ[k] = v.wtQ[k] >> 1;
  assert.equal(sha(bindsOf(v)), '559f728b0c7b6e43');
});

test('addCar keeps no CPU copy of what it uploaded', async () => {
  const { device, writes } = fakeDevice();
  const r = createShowcaseCarRenderer(device, { format: 'bgra8unorm' });
  const id = r.addCar(skinnedVisual());
  for (let k = 0; k < 3; k++) { await new Promise((res) => setTimeout(res)); gc(); }
  const kept = writes.filter((w) => /^showcase-(binds|out|attrs|index|floor-index|topology)/.test(w.label) && w.source.deref()).map((w) => w.label);
  assert.deepEqual(kept, []);
  assert.ok(r.removeCar && id !== undefined);
});

// Nor of the visual itself: the page may drop it once uploaded (a generated car's is
// hundreds of MB) and build it again when it needs it.
test('addCar keeps nothing of the visual it was given', async () => {
  const { device } = fakeDevice();
  const r = createShowcaseCarRenderer(device, { format: 'bgra8unorm' });
  let v = skinnedVisual();
  const refs = Object.entries(v).filter(([, x]) => ArrayBuffer.isView(x)).map(([k, x]) => [k, new WeakRef(x)]);
  const whole = new WeakRef(v);
  r.addCar(v);
  v = null;
  for (let k = 0; k < 3; k++) { await new Promise((res) => setTimeout(res)); gc(); }
  assert.equal(whole.deref(), undefined);
  assert.deepEqual(refs.filter(([, w]) => w.deref()).map(([k]) => k), []);
});

// skinnedVisual's three vertices tiled `copies` times (every third vertex with nine
// entries: three 4-entry slots), for a bind buffer of tens of MB.
function tiled(copies) {
  const v = skinnedVisual(), n = 3 * copies;
  const rep = (a, per) => { const o = new a.constructor(a.length * copies); for (let c = 0; c < copies; c++) o.set(a, c * a.length); return o; };
  const out = { ...v, vertexCount: n };
  for (const k of ['positions', 'normals', 'uvs', 'colors', 'ao', 'material', 'bindNodes', 'bindCoords', 'bindNormal', 'bindRest', 'smooth', 'rigid', 'wtOff', 'wtNrm']) out[k] = rep(v[k]);
  out.indices = new Uint32Array(n); for (let i = 0; i < n; i++) out.indices[i] = i;
  const start = [0], node = [], w = [];
  for (let i = 0; i < n; i++) {
    const r = i % 3 === 0 ? 3 : 1;
    for (let k = 0; k < r; k++) for (let e = 0; e < 3; e++) { const j = 3 * (i % 3) + e; node.push(v.wtNode[j]); w.push(v.wtW[3 * j], v.wtW[3 * j + 1], v.wtW[3 * j + 2]); }
    start.push(node.length);
  }
  Object.assign(out, { wtStart: Uint32Array.from(start), wtNode: Uint32Array.from(node), wtW: Float32Array.from(w) });
  return out;
}

// Pinned from the renderer that packed the whole bind buffer at once (cf30b0b9).
test('a large skin\'s bindings upload in bounded pieces, the same bytes', () => {
  const { device, writes, image } = fakeDevice();
  createShowcaseCarRenderer(device, { format: 'bgra8unorm' }).addCar(tiled(100_000));
  const binds = writes.filter((w) => w.label === 'showcase-binds');
  const total = binds.reduce((s, w) => s + w.size, 0);
  assert.equal(total, image('showcase-binds').byteLength);
  assert.ok(total > 30 * 2 ** 20, `${total} bytes`);
  assert.equal(sha(image('showcase-binds')), '084f4adc2b15eaac');
  assert.ok(Math.max(...binds.map((w) => w.size)) <= 8 * 2 ** 20, 'each piece at most 8 MB');
});

// The step layout staged in pieces (entries packed end to end cross the pieces'
// edges, a record shared by two vertices among them): the same entries as the f32
// layout's, and each piece at most 8 MB.
test('a large skin\'s step-layout bindings upload in bounded pieces, the same entries', () => {
  const v = tiled(200_000), n = v.vertexCount;
  const { device, writes, image } = fakeDevice();
  createShowcaseCarRenderer(device, { format: 'bgra8unorm' }).addCar(asSteps(v));
  const binds = writes.filter((w) => w.label === 'showcase-binds');
  assert.ok(binds.length > 3, `${binds.length} pieces`);
  assert.ok(Math.max(...binds.map((w) => w.size)) <= 8 * 2 ** 20, 'each piece at most 8 MB');
  const q = entries(image('showcase-binds'), n), f = entries(bindsOf(v), n);
  assert.equal(q.length, f.length);
  for (let k = 0; k < f.length; k++) if (q[k].some((x, c) => x !== f[k][c])) assert.fail(`entry ${k}: ${q[k]} vs ${f[k]}`);
});
