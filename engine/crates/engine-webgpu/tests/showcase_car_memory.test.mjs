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
function asSteps(v, grid) {
  const q = new Int16Array(v.wtW.length);
  for (let k = 0; k < q.length; k++) q[k] = v.wtW[k] / grid[k % 3 === 0 ? 0 : 1];
  return { ...v, wtNode: undefined, wtW: undefined, wtNode16: Uint16Array.from(v.wtNode), wtQ: q, wtGrid: grid };
}
const bindImage = (visual) => {
  const { device, image } = fakeDevice();
  createShowcaseCarRenderer(device, { format: 'bgra8unorm' }).addCar(visual);
  return image('showcase-binds');
};
// A bind image's vertices as (flags without the packed bit, record words 4..14, entries
// [node, A, dA/dT, dA/dB] as f32 values), read as SKIN_WGSL reads either form:
// weightedVertex's 4 entries a 64-byte slot, weightedPacked's 8 (2 words an entry, the
// steps times Q_A = 2^-13 and Q_D = 2^-9).
function decodeBinds(image, n) {
  const u = new Uint32Array(image.buffer, image.byteOffset, image.byteLength / 4), f = new Float32Array(u.buffer, u.byteOffset, u.length);
  const i16 = (x) => (x << 16) >> 16;
  const verts = [];
  let slots = 0;
  for (let i = 0; i < n; i++) {
    const o = 16 * i, count = u[o], first = u[o + 3], packed = (u[o + 15] >>> 27) & 1;
    const entries = [];
    for (let e = 0; e < count; e++) {
      if (packed) {
        const q = 16 * first + 2 * e;
        entries.push([u[q] & 0xffff, Math.fround(i16(u[q] >>> 16) * 2 ** -13), Math.fround(i16(u[q + 1] & 0xffff) * 2 ** -9), Math.fround(i16(u[q + 1] >>> 16) * 2 ** -9)]);
      } else {
        const q = 16 * first + 4 * e;
        entries.push([u[q], f[q + 1], f[q + 2], f[q + 3]]);
      }
    }
    slots += Math.ceil(count / (packed ? 8 : 4));
    verts.push({ packed, flags: u[o + 15] & ~(1 << 27), slot: first, words: [...u.subarray(o + 4, o + 15)], entries });
  }
  return { verts, slots, records: u.length / 16 };
}

// Grid steps on the shader's grid upload packed (8 entries a slot, not 4): the same
// vertices and the same f32 weights as the f32 form, from half the entry bytes.
test('a skin\'s weights as grid steps upload packed, the same values as f32 weights', () => {
  const v = tiled(50), n = v.vertexCount;
  const plain = decodeBinds(bindImage(v), n);
  const packed = decodeBinds(bindImage(asSteps(v, new Float32Array([2 ** -13, 2 ** -9]))), n);
  assert.ok(plain.verts.every((x) => !x.packed) && packed.verts.every((x) => x.packed));
  const values = (d) => d.verts.map(({ flags, words, entries }) => ({ flags, words, entries }));
  assert.deepEqual(values(packed), values(plain));
  // Every third vertex has nine entries: three slots of 4, two of 8; the others one.
  assert.equal(plain.slots, 50 * (3 + 1 + 1));
  assert.equal(packed.slots, 50 * (2 + 1 + 1));
  assert.equal(packed.records, n + packed.slots);
  assert.equal(sha(bindImage(skinnedVisual())), '559f728b0c7b6e43');
});

// Steps of another grid than the shader's stay in the f32 form (the same bytes as f32 weights).
test('a skin\'s weights on another grid upload as f32 weights', () => {
  const v = skinnedVisual();
  assert.equal(sha(bindImage(asSteps(v, new Float32Array([2 ** -12, 2 ** -9])))), sha(bindImage({ ...v, wtW: v.wtW.map((x, k) => (k % 3 === 0 ? Math.trunc(x / 2 ** -12) * 2 ** -12 : x)) })));
  assert.ok(decodeBinds(bindImage(asSteps(v, new Float32Array([2 ** -12, 2 ** -9]))), 3).verts.every((x) => !x.packed));
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
