import test from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';

// The vehicle lab's guard before it loads a glTF-skinned vehicle: its footprint from
// the model files' sizes, checked against what the device says it has. rubble-world.mjs
// (the wasm's wrapper, built, absent in CI) stands in.
register('data:text/javascript,' + encodeURIComponent(`
export async function resolve(spec, ctx, next) {
  if (spec.endsWith('/rubble-world.mjs') || spec === './rubble-world.mjs') return { url: 'data:text/javascript,' + encodeURIComponent(
    'export const jbeamVisual = () => null; export const jbeamVisualSkinned = () => null; export const decodeTexture = () => null;'), shortCircuit: true };
  return next(spec, ctx);
}`));
const { modFootprint, footprintCheck, modModelSizes } = await import('../beamng-mod.mjs');

const MB = 2 ** 20, GB = 2 ** 30;
// The generated sedan's files (sedan.glb, sedan.rskin).
const GEN = { glb: 16_268_632, skin: 123_844_942 };

test('a bigger model costs more, in the tab and on the GPU', () => {
  const small = modFootprint({ glb: 1 * MB, skin: 2 * MB }), big = modFootprint(GEN);
  assert.ok(big.tab > small.tab && big.gpu > small.gpu && big.largestBuffer > small.largestBuffer);
  // The skin's bind buffer is the largest single buffer: about twice the sidecar.
  assert.ok(big.largestBuffer > 1.9 * GEN.skin && big.largestBuffer < 2.3 * GEN.skin);
});

test('a device that reports too little memory is refused, with the numbers', () => {
  const fp = modFootprint(GEN);
  const r = footprintCheck(fp, { deviceMemory: 1, limits: null, name: 'Generated sedan' });
  assert.equal(r.fits, false);
  assert.match(r.message, /Generated sedan/);
  assert.match(r.message, /1 GB/);
  assert.match(r.message, new RegExp(`${Math.round(fp.tab / MB)} MB`));
  assert.equal(footprintCheck(fp, { deviceMemory: 8, limits: null, name: 'x' }).fits, true);
});

test('a device that does not say is not refused for memory; its GPU limits still apply', () => {
  const fp = modFootprint(GEN);
  assert.equal(footprintCheck(fp, { deviceMemory: undefined, limits: null, name: 'x' }).fits, true);
  const small = { maxBufferSize: 256 * MB, maxStorageBufferBindingSize: 128 * MB };
  const r = footprintCheck(fp, { deviceMemory: undefined, limits: small, name: 'Generated sedan' });
  assert.equal(r.fits, false);
  assert.match(r.message, /maxStorageBufferBindingSize/);
  assert.equal(footprintCheck(fp, { deviceMemory: undefined, limits: { maxBufferSize: GB, maxStorageBufferBindingSize: GB }, name: 'x' }).fits, true);
});

test('the model files\' sizes come from their headers, before downloading them', async () => {
  const seen = [];
  globalThis.fetch = async (url, init) => {
    seen.push([String(url), init?.method ?? 'GET']);
    if (String(url).endsWith('files.json')) return new Response(JSON.stringify(['car/car.jbeam', 'car/car.glb', 'car/car.rskin']));
    return new Response(null, { status: 200, headers: { 'Content-Length': String(url).endsWith('.glb') ? '500' : '7000' } });
  };
  assert.deepEqual(await modModelSizes('./m'), { glb: 500, skin: 7000 });
  assert.deepEqual(seen.filter(([u]) => !u.endsWith('files.json')).map(([, m]) => m), ['HEAD', 'HEAD']);
});
