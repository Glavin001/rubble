// What the showcase renderer uploads for a converted vehicle (its skinned visual from
// the World wasm, packed by showcase-car-render.mjs addCar), hashed per buffer label,
// without a GPU (a stand-in device; node). Run it on two builds (a wasm and a renderer
// each) to check that a change leaves the drawn car's data bit for bit the same.
//
//   node engine/tools/vehicle-lab/skin-upload-identity.mjs \
//     [--mod ~/Development/rubble-artifacts/sedan-live/mods/sedan_new] [--pc sedan/sedan.pc] \
//     [--wasm engine/crates/engine-webgpu/generated/world-wasm] [--renderer path/showcase-car-render.mjs]
//
// --wasm: a folder with engine_world_wasm.js and engine_world_wasm_bg.wasm (a build).
// Prints one JSON object: { label: sha256 of every write to it, in order }.
import { register } from 'node:module';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../crates/engine-webgpu');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const MOD = opt('--mod', join(homedir(), 'Development/rubble-artifacts/sedan-live/mods/sedan_new'));
const PC = opt('--pc', 'sedan/sedan.pc');
const WASM = resolve(opt('--wasm', join(ROOT, 'generated/world-wasm')));
const RENDERER = resolve(opt('--renderer', join(ROOT, 'showcase-car-render.mjs')));

// The wasm's GPU-wait import (resolved by the pages' import map).
register('data:text/javascript,' + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  if (spec === 'rubble-gpu-wait') return { url: ${JSON.stringify(pathToFileURL(join(ROOT, 'fracture/gpu-wait.mjs')).href)}, shortCircuit: true };
  return next(spec, ctx);
}`));
Object.assign(globalThis, {
  GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
  GPUTextureUsage: { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 },
  GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  GPUMapMode: { READ: 1, WRITE: 2 },
});

const wasm = await import(pathToFileURL(join(WASM, 'engine_world_wasm.js')).href);
wasm.initSync({ module: readFileSync(join(WASM, 'engine_world_wasm_bg.wasm')) });

// The vehicle folder as beamng-mod.mjs loadMod reads it (its files from disk).
const list = JSON.parse(readFileSync(join(MOD, 'files.json'), 'utf8'));
const text = (f) => [f, readFileSync(join(MOD, f), 'utf8')];
const files = list.filter((f) => f.endsWith('.jbeam')).map(text);
const materials = list.filter((f) => f.endsWith('.materials.json')).map(text);
const glb = new Uint8Array(readFileSync(join(MOD, list.find((f) => f.endsWith('.glb') && f.includes('/')))));
const skin = new Uint8Array(readFileSync(join(MOD, list.find((f) => f.endsWith('.rskin')))));
const spec = { jbeam: files, pc: readFileSync(join(MOD, PC), 'utf8'), native: true, wheels: 'raycast', paint: [0.6, 0.04, 0.03], materials };
const r = wasm.jbeamVisualSkinned(spec, [], glb, skin);
const visual = r.visual;

// Anything the renderer asks of a device; each buffer's contents once written (however
// many writes it took), hashed by label.
const any = () => new Proxy(function () {}, { get: (_, k) => (k === 'then' ? undefined : any()), apply: () => any() });
const buffers = [];
const queue = {
  writeBuffer(buffer, offset, data, dataOffset = 0, size) {
    const el = ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT : 1;
    const src = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
    buffer.image.set(src.subarray(dataOffset * el, size !== undefined ? (dataOffset + size) * el : undefined), offset);
  },
  writeTexture() {}, submit() {},
};
const device = new Proxy({}, { get: (_, k) => (k === 'queue' ? queue : k === 'createBuffer' ? (d) => { const b = { label: d.label, size: d.size, image: new Uint8Array(d.size), destroy() {} }; buffers.push(b); return b; } : k === 'then' ? undefined : any()) });
const { createShowcaseCarRenderer } = await import(pathToFileURL(RENDERER).href);
const cars = createShowcaseCarRenderer(device, { format: 'bgra8unorm', depthFormat: 'depth32float', sampleCount: 4 });
// As the lab adds its first car: raycast wheels' records stand in (4 words a wheel).
cars.addCar(visual, { nodeOffset: 0, beamOffset: 0, vehicle: 0, lod: visual.lod ?? null, wheelRecords: new Uint32Array(16), retireAt: 1, labelAt: 2 });
const out = { vertices: visual.vertexCount, weights: visual.wtQ ? 'grid steps (wtQ)' : 'f32 (wtW)' };
for (const b of buffers.filter((b) => b.label?.startsWith('showcase-') && !/frame|view|fx|contact|mirror/.test(b.label)).sort((x, y) => (x.label < y.label ? -1 : 1))) {
  out[b.label] = `${b.size} bytes ${createHash('sha256').update(b.image).digest('hex').slice(0, 24)}`;
}
console.log(JSON.stringify(out, null, 1));
