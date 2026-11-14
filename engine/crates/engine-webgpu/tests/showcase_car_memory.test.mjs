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
  const any = () => new Proxy(function () {}, { get: (_, k) => (k === 'then' ? undefined : any()), apply: () => any() });
  const writes = [];
  const queue = {
    writeBuffer(buffer, offset, data, dataOffset = 0, size) {
      const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
      const bytes = new Uint8Array(src.buffer, src.byteOffset + dataOffset * (src.BYTES_PER_ELEMENT ?? 1), size !== undefined ? size * (src.BYTES_PER_ELEMENT ?? 1) : src.byteLength - dataOffset * (src.BYTES_PER_ELEMENT ?? 1));
      writes.push({ label: buffer.label, offset, bytes: bytes.slice(), source: new WeakRef(ArrayBuffer.isView(data) ? data.buffer : data) });
    },
    writeTexture() {},
    submit() {},
  };
  const device = new Proxy({}, {
    get: (_, k) => {
      if (k === 'queue') return queue;
      if (k === 'createBuffer') return (d) => ({ label: d.label, size: d.size, usage: d.usage, destroy() {} });
      if (k === 'then') return undefined;
      return any();
    },
  });
  return { device, writes };
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
  const { device, writes } = fakeDevice();
  const r = createShowcaseCarRenderer(device, { format: 'bgra8unorm' });
  r.addCar(skinnedVisual());
  const got = Object.fromEntries(['showcase-binds', 'showcase-out', 'showcase-attrs', 'showcase-index', 'showcase-topology']
    .map((l) => [l, sha(writes.find((w) => w.label === l).bytes)]));
  assert.deepEqual(got, {
    'showcase-binds': '559f728b0c7b6e43',
    'showcase-out': 'ea471a5319892872',
    'showcase-attrs': '8fc4cb57bf764fe8',
    'showcase-index': 'ad5dc1478de06a4c',
    'showcase-topology': '79f11e5544822b94',
  });
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
