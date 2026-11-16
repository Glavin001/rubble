import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';

// beamng-mod.mjs without the wasm: rubble-world.mjs (the wasm's wrapper, built, absent
// in CI) stands in as a module whose visual builders report what they were given.
register('data:text/javascript,' + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  if (spec.endsWith('/rubble-world.mjs') || spec === './rubble-world.mjs') return { url: 'data:text/javascript,' + encodeURIComponent(
    'export const jbeamVisual = () => ({ visual: { kind: "collada" } });' +
    'export const jbeamVisualSkinned = (spec, meshes, glb, skin) => ({ visual: { kind: "skinned", glb: glb.length, skin: skin.length } });' +
    'export const decodeTexture = () => null;'), shortCircuit: true };
  return next(spec, ctx);
}`));
const { loadMod, modVisual, modModels, releaseModels, closedSolids } = await import('../beamng-mod.mjs');

// A converted vehicle folder served by a stand-in fetch, counting each file's fetches.
const FILES = {
  'files.json': JSON.stringify(['car/car.jbeam', 'car/car.pc', 'car/car.glb', 'car/car.rskin']),
  'car/car.jbeam': '{}', 'car/car.pc': '{}',
  'car/car.glb': new Uint8Array([1, 2, 3, 4, 5]), 'car/car.rskin': new Uint8Array([9, 8, 7]),
};
const fetches = {};
globalThis.fetch = async (url) => {
  const f = String(url).replace(/^\.\/m\//, '');
  fetches[f] = (fetches[f] ?? 0) + 1;
  const body = FILES[f];
  if (body === undefined) return new Response('', { status: 404, statusText: 'Not Found' });
  return new Response(body, { status: 200, headers: { 'Content-Length': String(typeof body === 'string' ? body.length : body.byteLength) } });
};

test('a skinned vehicle\'s model files can be released after its visual is built, and fetched again', async () => {
  const mod = await loadMod('./m');
  assert.equal(mod.modelBytes, 8);
  assert.deepEqual(modVisual(mod).visual, { kind: 'skinned', glb: 5, skin: 3, lod: [], closedPanels: false });
  releaseModels(mod);
  assert.equal(mod.glb, null);
  assert.equal(mod.skin, null);
  // Released, it is never drawn from its Collada fallback instead: a clear error.
  assert.throws(() => modVisual(mod), /model files were released/);
  await modModels(mod);
  assert.deepEqual([...mod.glb], [1, 2, 3, 4, 5]);
  assert.deepEqual([...mod.skin], [9, 8, 7]);
  assert.deepEqual(modVisual(mod).visual, { kind: 'skinned', glb: 5, skin: 3, lod: [], closedPanels: false });
  assert.equal(fetches['car/car.rskin'], 2);
  // Held, nothing is fetched again.
  await modModels(mod);
  assert.equal(fetches['car/car.rskin'], 2);
});

// A glTF binary with this JSON chunk (and no buffer).
const glbOf = (json) => {
  const text = new TextEncoder().encode(JSON.stringify(json));
  const pad = (4 - (text.length % 4)) % 4;
  const out = new Uint8Array(20 + text.length + pad).fill(0x20);
  const view = new DataView(out.buffer);
  view.setUint32(0, 0x46546c67, true); view.setUint32(4, 2, true); view.setUint32(8, out.length, true);
  view.setUint32(12, text.length + pad, true); view.setUint32(16, 0x4e4f534a, true);
  out.set(text, 20);
  return out;
};

test('a model whose parts carry the generator\'s surface tags is built of closed solids', () => {
  const sheet = { meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 } }] }] };
  const solid = { meshes: [sheet.meshes[0], { primitives: [{ attributes: { POSITION: 2, _SURFACE: 3 } }] }] };
  assert.equal(closedSolids(glbOf(solid)), true);
  assert.equal(closedSolids(glbOf(sheet)), false);
  // No model, or bytes that are not a glTF binary: sheets (the renderer's default).
  assert.equal(closedSolids(null), false);
  assert.equal(closedSolids(new Uint8Array([1, 2, 3, 4, 5])), false);
  assert.equal(closedSolids(new Uint8Array(64)), false);
  // Read through a view into a larger buffer (a file's bytes at an offset).
  const bytes = glbOf(solid), wide = new Uint8Array(bytes.length + 7);
  wide.set(bytes, 7);
  assert.equal(closedSolids(wide.subarray(7)), true);
});
