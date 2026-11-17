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
const { modFootprint, footprintCheck, modModelSizes, modModelNames } = await import('../beamng-mod.mjs');

const MB = 2 ** 20, GB = 2 ** 30;
// The generated sedan's files (sedan.glb, sedan.rskin).
const GEN = { glb: 16_268_632, skin: 123_844_942 };

test('a bigger model costs more, in the tab and on the GPU', () => {
  const small = modFootprint({ glb: 1 * MB, skin: 2 * MB }), big = modFootprint(GEN);
  assert.ok(big.tab > small.tab && big.gpu > small.gpu && big.largestBuffer > small.largestBuffer);
  // The skin's bind buffer is the largest single buffer: 157,440,000 bytes for the
  // sedan (its entries as grid steps, 8 a slot; showcase-car-render.mjs), a little over.
  assert.ok(big.largestBuffer > 157_440_000 && big.largestBuffer < 1.05 * 157_440_000);
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
  assert.deepEqual(await modModelSizes('./m'), { glb: 500, skin: 7000, detail: 'full', hasLite: false });
  assert.deepEqual(seen.filter(([u]) => !u.endsWith('files.json')).map(([, m]) => m), ['HEAD', 'HEAD']);
  // A folder without a lighter render gives its art for either detail.
  assert.deepEqual(await modModelSizes('./m', 'lite'), { glb: 500, skin: 7000, detail: 'full', hasLite: false });
});

test('a generated car\'s lighter render is picked only when asked for, by its own names', async () => {
  const list = ['car/car.jbeam', 'car/car.glb', 'car/car.rskin', 'car/car.lite.glb', 'car/car.lite.rskin'];
  assert.deepEqual(modModelNames(list), { glbName: 'car/car.glb', skinName: 'car/car.rskin', detail: 'full', hasLite: true });
  assert.deepEqual(modModelNames(list, 'lite'), { glbName: 'car/car.lite.glb', skinName: 'car/car.lite.rskin', detail: 'lite', hasLite: true });
  // (Listed lighter first: the art is still the full one.)
  assert.equal(modModelNames([...list].reverse()).skinName, 'car/car.rskin');
  globalThis.fetch = async (url) => (String(url).endsWith('files.json') ? new Response(JSON.stringify(list))
    : new Response(null, { status: 200, headers: { 'Content-Length': String(url).includes('.lite.') ? (String(url).endsWith('.glb') ? '9' : '60') : (String(url).endsWith('.glb') ? '16' : '124') } }));
  assert.deepEqual(await modModelSizes('./m', 'lite'), { glb: 9, skin: 60, detail: 'lite', hasLite: true });
  assert.deepEqual(await modModelSizes('./m'), { glb: 16, skin: 124, detail: 'full', hasLite: true });
});
