// Whether two showcase renderers skin a converted vehicle to the same bits on the GPU:
// headless Chrome builds the car's visual once (the World wasm, as the vehicle lab does),
// then each renderer (an old build's showcase-car-render.mjs and this one) uploads it
// and runs its compute passes on the same deformed node positions, each level of detail
// in turn; every level's `showcase-out` (position, normal, tear, damage, plastic strain
// and scrape per vertex) is read back and compared byte for byte.
//
//   python3 engine/scripts/with-gpu-lock.py --limit 600 -- env RUBBLE_GPU_LOCK=0 \
//     node engine/tools/vehicle-lab/skin-gpu-identity.mjs --base http://127.0.0.1:8863/web-dist \
//       --old http://127.0.0.1:8863/web-dist-before/showcase-car-render.mjs \
//       [--mod ./__mods/sedan_new] [--pc sedan/sedan.pc]
//
// --base: a static server over a built site (engine/web-dist: the world wasm and this
// renderer); --old: the renderer to compare with (any URL the page may import). Serves
// nothing; the page itself is served from memory (CDP Fetch) under --base.
// Prints per level the vertex count, how many vertices moved off the rest pose and
// whether the bytes match; exits 1 on any difference.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const BASE = opt('--base', 'http://127.0.0.1:8863');
const OLD = opt('--old', null);
const MOD = opt('--mod', './__mods/sedan_new');
const PC = opt('--pc', 'sedan/sedan.pc');
const LIMIT_S = Number(opt('--limit', '280'));
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
if (!OLD) { console.error('--old <url of the renderer to compare with> is required'); process.exit(2); }
setTimeout(() => { console.error(`time limit (${LIMIT_S} s)`); process.exit(9); }, LIMIT_S * 1000).unref();

const PAGE = `<!doctype html><meta charset="utf-8">
<script type="importmap">{ "imports": { "rubble-gpu-wait": "./fracture/gpu-wait.mjs" } }</script>
<script type="module">
import { loadWasm } from './rubble-world.mjs';
import { loadMod, modVisual } from './beamng-mod.mjs';
import { createShowcaseCarRenderer as createNew } from './showcase-car-render.mjs';
try {
  const q = new URLSearchParams(location.search);
  const { createShowcaseCarRenderer: createOld } = await import(q.get('old'));
  await loadWasm();
  const mod = await loadMod(q.get('mod'), { pc: q.get('pc') });
  const visual = modVisual(mod, { native: true, raycast: true, paint: [0.6, 0.04, 0.03] }).visual;
  mod.glb = mod.skin = null;
  const adapter = await navigator.gpu.requestAdapter();
  const requiredLimits = {};
  for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage', 'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxBindingsPerBindGroup', 'maxUniformBufferBindingSize']) requiredLimits[k] = adapter.limits[k];
  const device = await adapter.requestDevice({ requiredLimits });
  const errors = [];
  device.addEventListener('uncapturederror', (e) => errors.push(String(e.error?.message ?? e.error)));
  // The node positions: the rest pose bent, twisted and rippled (every term of the
  // skin moves), the same for both renderers.
  const nc = visual.nodeCount, rest = visual.nodeRest;
  const nodes = new Float32Array(12 * nc);
  for (let i = 0; i < nc; i++) {
    const [x, y, z] = [rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]];
    const t = 0.08 * z, c = Math.cos(t), s = Math.sin(t);
    nodes.set([c * x - s * y + 0.02 * Math.sin(7 * z + 3 * y), s * x + c * y + 0.015 * z * z, z + 0.01 * Math.cos(11 * x), 1], 12 * i);
  }
  const mk = (data, usage) => { const b = device.createBuffer({ size: Math.max(256, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
  const nodesBuf = mk(nodes, GPUBufferUsage.STORAGE);
  const miscBuf = mk(new Uint32Array(1 << 16), GPUBufferUsage.STORAGE);
  const nb = visual.beamRest0?.length ?? 0;
  const beams = new Float32Array(8 * Math.max(nb, 1));
  for (let j = 0; j < nb; j++) beams[8 * j] = visual.beamRest0[j];
  const beamsBuf = mk(beams, GPUBufferUsage.STORAGE);
  const levels = [visual, ...(visual.lod ?? [])];
  const run = async (create) => {
    const made = [];
    const cb = device.createBuffer.bind(device);
    device.createBuffer = (d) => { const b = cb(d); made.push([d.label, b]); return b; };
    const r = create(device, { format: 'bgra8unorm', depthFormat: 'depth32float', sampleCount: 4 });
    r.setSolver(nodesBuf, miscBuf, beamsBuf);
    const id = r.addCar(visual, { nodeOffset: 0, beamOffset: 0, vehicle: 0, lod: visual.lod ?? null, wheelRecords: new Uint32Array(16), retireAt: 1, labelAt: 2 });
    delete device.createBuffer;
    const view = made.find(([l]) => l === 'showcase-view')[1];
    // The level of detail follows the eye's distance (view.w > 0): near (level 0) to
    // far (the last), every level picked at some distance.
    for (const d of [0, ...Array.from({ length: 48 }, (_, k) => 10 ** (k / 8 - 1))]) {
      device.queue.writeBuffer(view, 0, new Float32Array([d, 0, 0, d ? 1000 : 0]));
      const enc = device.createCommandEncoder();
      r.update(enc, { dt: 0, floorY: -10 });
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const outs = [];
    for (const [l, b] of made) {
      if (l !== 'showcase-out') continue;
      const rb = device.createBuffer({ size: b.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(b, 0, rb, 0, b.size);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      outs.push(new Uint8Array(rb.getMappedRange()).slice());
      rb.destroy();
    }
    r.removeCar(id); r.destroy?.();
    return outs;
  };
  const a = await run(createOld), b = await run(createNew);
  const report = levels.map((lv, k) => {
    const x = a[k], y = b[k], fx = new Float32Array(y.buffer), n = lv.vertexCount;
    let diff = 0, moved = 0;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) diff++;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) if (fx[8 * i + c] !== lv.positions[3 * i + c]) { moved++; break; }
    }
    return { level: k, vertices: n, weighted: lv.wtStart ? lv.wtStart[n] - lv.wtStart[0] : 0, steps: !!lv.wtQ, bytes: y.length, moved, differingBytes: diff, same: diff === 0 && x.length === y.length };
  });
  window.__result = { report, errors, outsOld: a.length, outsNew: b.length };
} catch (e) { window.__result = { error: String(e?.stack ?? e) }; }
</script>`;

const profile = mkdtempSync(join(tmpdir(), 'skin-gpu-identity-chrome-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--enable-unsafe-webgpu',
  '--window-size=1280,720', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
process.on('exit', cleanup);

let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map();
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Fetch.requestPaused') {
    send('Fetch.fulfillRequest', { requestId: m.params.requestId, responseCode: 200, responseHeaders: [{ name: 'Content-Type', value: 'text/html' }], body: Buffer.from(PAGE).toString('base64') });
  }
  if (m.method === 'Runtime.exceptionThrown') console.error('page: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Fetch.enable', { patterns: [{ urlPattern: '*__skin-gpu-identity.html*' }] });
await send('Page.navigate', { url: `${BASE}/__skin-gpu-identity.html?old=${encodeURIComponent(OLD)}&mod=${encodeURIComponent(MOD)}&pc=${encodeURIComponent(PC)}` });
let result = null;
for (const t0 = Date.now(); !result && Date.now() - t0 < (LIMIT_S - 10) * 1000;) {
  await new Promise((r) => setTimeout(r, 500));
  result = await evaluate('window.__result ? JSON.stringify(window.__result) : null').then((s) => (s ? JSON.parse(s) : null)).catch(() => null);
}
if (!result) { console.error('no result'); process.exit(1); }
console.log(JSON.stringify(result, null, 1));
process.exit(result.error || result.errors?.length || !result.report.every((r) => r.same) ? 1 : 0);
