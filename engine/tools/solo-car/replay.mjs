#!/usr/bin/env node
// The solo core kernel against today's frame kernel in Chrome's WebGPU (perf doc §8.43):
// both from one saved state (a bundle of `solo_core_check --dump DIR`: every binding's
// bytes, the frame's parameters, the car's table, both modules' WGSL), each a dispatch a
// frame; every word of the state buffers compared after FRAMES frames, and each kernel's
// GPU time a frame (timestamp queries), alternating runs so neither has the warmer GPU.
//
//   node --experimental-websocket tools/solo-car/replay.mjs BUNDLE_DIR [FRAMES] [RUNS]
//
// Run it under scripts/with-gpu-lock.py. Needs Chrome (CHROME to override).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, normalize, resolve } from 'node:path';

const DIR = resolve(process.argv[2] || '');
const FRAMES = Number(process.argv[3] || 60);
const RUNS = Number(process.argv[4] || 3);
if (!existsSync(join(DIR, 'manifest.json'))) {
  console.error(`no ${DIR}/manifest.json: run solo_core_check --dump first`);
  process.exit(2);
}
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const PAGE = `<!doctype html><meta charset="utf-8"><title>solo replay</title><script type="module">
const bin = async (n) => new Uint8Array(await (await fetch('/' + n)).arrayBuffer());
window.__run = async (frames, runs) => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const L = adapter.limits;
  const feats = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({ requiredFeatures: feats, requiredLimits: {
    maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize,
  } });
  const errors = [];
  device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
  const init = [];
  for (let b = 0; b <= 8; b++) init.push(await bin('b' + b + '.bin'));
  const table = await bin('table.bin');
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const pad = (n) => Math.max(16, Math.ceil(n / 16) * 16);
  const make = (bytes, usage) => {
    const buf = device.createBuffer({ size: pad(bytes.length), usage });
    device.queue.writeBuffer(buf, 0, bytes);
    return buf;
  };
  const params = make(init[0], GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const tbl = make(table, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const entry = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
  const l0 = device.createBindGroupLayout({ entries: [entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'read-only-storage'),
    ...[3, 4, 5, 6, 7, 8].map((b) => entry(b, 'storage'))] });
  const l1 = device.createBindGroupLayout({ entries: [entry(0, 'read-only-storage')] });
  const pipe = async (file, entryPoint, layouts) => {
    const code = await (await fetch('/' + file)).text();
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error').map((m) => m.lineNum + ': ' + m.message);
    if (errs.length) throw new Error(file + ': ' + errs.slice(0, 3).join(' | '));
    return device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module, entryPoint } });
  };
  const kernels = {
    frame: { pipe: await pipe('frame.wgsl', 'explicit_beams_frame', [l0]) },
    solo: { pipe: await pipe('solo.wgsl', 'explicit_beams_solo_core', [l0, l1]) },
  };
  // A fresh copy of the state for a run.
  const state = () => {
    const bufs = [params];
    for (let b = 1; b <= 8; b++) bufs.push(make(init[b], b <= 2 ? GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST : S));
    return bufs;
  };
  const groups = (k, bufs) => {
    const g0 = device.createBindGroup({ layout: l0, entries: bufs.map((buffer, binding) => ({ binding, resource: binding === 0 ? { buffer, size: 512 } : { buffer } })) });
    const out = [g0];
    if (k === 'solo') out.push(device.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: tbl } }] }));
    return out;
  };
  const qs = feats.length ? device.createQuerySet({ type: 'timestamp', count: 2 * frames }) : null;
  const resolveBuf = qs ? device.createBuffer({ size: 16 * frames, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const read = async (buf) => {
    const st = device.createBuffer({ size: buf.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, buf.size);
    device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(st.getMappedRange().slice(0));
    st.unmap();
    return out;
  };
  const run = async (k, timed) => {
    const bufs = state();
    const gs = groups(k, bufs);
    for (let f = 0; f < frames; f++) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(timed && qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * f, endOfPassWriteIndex: 2 * f + 1 } } : {});
      pass.setPipeline(kernels[k].pipe);
      gs.forEach((g, i) => pass.setBindGroup(i, g));
      pass.dispatchWorkgroups(1);
      pass.end();
      device.queue.submit([enc.finish()]);
      // (One frame at a time, as the World waits on its step: no frames queued behind.)
      await device.queue.onSubmittedWorkDone();
    }
    let ms = [];
    if (timed && qs) {
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(qs, 0, 2 * frames, resolveBuf, 0);
      device.queue.submit([enc.finish()]);
      const st = device.createBuffer({ size: 16 * frames, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e2 = device.createCommandEncoder();
      e2.copyBufferToBuffer(resolveBuf, 0, st, 0, 16 * frames);
      device.queue.submit([e2.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(st.getMappedRange().slice(0));
      st.unmap();
      for (let f = 0; f < frames; f++) ms.push(Number(t[2 * f + 1] - t[2 * f]) / 1e6);
    }
    return { bufs, ms };
  };
  // Bit for bit: both kernels FRAMES frames from the state, every word of bindings 3-8.
  const a = await run('frame', false);
  const b = await run('solo', false);
  const names = ['', '', '', 'nodes', 'beams', 'acc', 'groups', 'misc', 'next'];
  const diffs = [];
  for (let k = 3; k <= 8; k++) {
    const x = await read(a.bufs[k]);
    const y = await read(b.bufs[k]);
    let n = 0, first = -1;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) { n++; if (first < 0) first = i; }
    if (n) diffs.push(names[k] + ': ' + n + ' words differ, first at ' + first + ' (' + x[first] + ' vs ' + y[first] + ')');
  }
  const times = { frame: [], solo: [] };
  for (let r = 0; r < runs; r++) for (const k of ['frame', 'solo']) times[k].push(...(await run(k, true)).ms);
  return { adapter: (adapter.info && adapter.info.description) || '', wg: L.maxComputeWorkgroupStorageSize, inv: L.maxComputeInvocationsPerWorkgroup, diffs, times, errors };
};
window.__ready = true;
</script>`;
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p === '/' || p === '/index.html') { res.writeHead(200, { 'content-type': 'text/html' }).end(PAGE); return; }
  const path = normalize(join(DIR, p));
  if (!path.startsWith(DIR) || !existsSync(path)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': 'application/octet-stream' }).end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'solo-replay-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--enable-features=WebGPUTimestampQuery', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} server.close(); };
process.on('exit', cleanup);
for (const sig of ['SIGALRM', 'SIGTERM', 'SIGINT']) process.on(sig, () => process.exit(3));
let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map();
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 900_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.navigate', { url: `${base}/` });
for (let i = 0; i < 100; i++) {
  if (await evaluate('window.__ready === true').catch(() => false)) break;
  await new Promise((r) => setTimeout(r, 100));
}
const r = await evaluate(`window.__run(${FRAMES}, ${RUNS})`);
const stat = (v) => { const s = [...v].sort((a, b) => a - b); const q = (f) => s[Math.round((s.length - 1) * f)]; return `median ${q(0.5).toFixed(3)} ms, p95 ${q(0.95).toFixed(3)}, worst ${s[s.length - 1].toFixed(3)} (${s.length} frames)`; };
console.log(`Chrome WebGPU ${r.adapter}: workgroup storage ${r.wg} B, invocations ${r.inv}`);
console.log(r.diffs.length ? `DIFFERS after ${FRAMES} frames: ${r.diffs.join('; ')}` : `bit for bit: every word of nodes, beams, acc, groups, misc, next after ${FRAMES} frames`);
for (const k of ['frame', 'solo']) if (r.times[k].length) console.log(`${k}: GPU ${stat(r.times[k])}`);
if (r.errors.length) console.log('device errors:', r.errors.slice(0, 5));
process.exit(r.diffs.length || r.errors.length ? 1 : 0);
