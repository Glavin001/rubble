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
    'export const jbeamVisualSkinnedHeld = (spec, meshes, glb, skin) => ({ visual: { kind: "held", glb: glb.bytes(), skin: skin.bytes() } });' +
    'export class HeldBytes { constructor(c) { this.parts = []; this.freed = false; HeldBytes.made.push(this); } push(b) { this.parts.push(Uint8Array.from(b)); } get length() { return this.parts.reduce((s, p) => s + p.length, 0); } bytes() { return [...this.parts.flatMap((p) => [...p])]; } free() { this.freed = true; } };' +
    'HeldBytes.made = [];' +
    'export const decodeTexture = () => null;'), shortCircuit: true };
  return next(spec, ctx);
}`));
const { loadMod, modVisual, modModels, releaseModels } = await import('../beamng-mod.mjs');
const { HeldBytes } = await import('../rubble-world.mjs');

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
  assert.deepEqual(modVisual(mod).visual, { kind: 'skinned', glb: 5, skin: 3, lod: [] });
  releaseModels(mod);
  assert.equal(mod.glb, null);
  assert.equal(mod.skin, null);
  // Released, it is never drawn from its Collada fallback instead: a clear error.
  assert.throws(() => modVisual(mod), /model files were released/);
  await modModels(mod);
  assert.deepEqual([...mod.glb], [1, 2, 3, 4, 5]);
  assert.deepEqual([...mod.skin], [9, 8, 7]);
  assert.deepEqual(modVisual(mod).visual, { kind: 'skinned', glb: 5, skin: 3, lod: [] });
  assert.equal(fetches['car/car.rskin'], 2);
  // Held, nothing is fetched again.
  await modModels(mod);
  assert.equal(fetches['car/car.rskin'], 2);
});

// held: the model files streamed into the wasm heap (HeldBytes, a stand-in here) as
// they download, read there, and freed when released; fetched (streamed) again.
test('a skinned vehicle\'s model files can be held in the wasm heap, released and fetched again', async () => {
  const mod = await loadMod('./m', { held: true });
  assert.equal(mod.modelBytes, 8);
  assert.ok(mod.glb instanceof HeldBytes && mod.skin instanceof HeldBytes);
  assert.deepEqual(modVisual(mod).visual, { kind: 'held', glb: [1, 2, 3, 4, 5], skin: [9, 8, 7], lod: [] });
  const [glb, skin] = [mod.glb, mod.skin];
  releaseModels(mod);
  assert.ok(glb.freed && skin.freed);
  assert.equal(mod.skin, null);
  assert.throws(() => modVisual(mod), /model files were released/);
  await modModels(mod);
  assert.deepEqual(modVisual(mod).visual.skin, [9, 8, 7]);
});

test('a held file cut short is fetched once more, then refused naming both sizes', async () => {
  const real = globalThis.fetch;
  let n = 0;
  globalThis.fetch = async (url) => {
    if (!String(url).endsWith('.rskin')) return real(url);
    n++;
    return new Response(new Uint8Array([1, 2]), { status: 200, headers: { 'Content-Length': '3' } });
  };
  try {
    await assert.rejects(loadMod('./m', { held: true }), /car\.rskin: truncated, 2 of 3 bytes \(after a retry\)/);
    assert.equal(n, 2);
    assert.ok(HeldBytes.made.filter((h) => h.length === 2).every((h) => h.freed));
  } finally { globalThis.fetch = real; }
});
