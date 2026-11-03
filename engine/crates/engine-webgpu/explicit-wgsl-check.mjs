#!/usr/bin/env node
// Compile every shipped explicit-beams WGSL module in Chrome's WebGPU (Tint): the
// shader module (its compilation messages, uniformity included) and a compute
// pipeline (layout "auto"), on a device with the adapter's limits as the page
// requests them. Native wgpu (naga) accepts barriers under non-uniform control flow
// that Chrome rejects, so run this after kernel changes.
//
//   cargo run -p engine-webgpu --release --example explicit_wgsl_dump
//   node --experimental-websocket explicit-wgsl-check.mjs
//
// Needs Chrome (CHROME env to override). Exits non-zero when a module fails.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const DIR = join(ROOT, 'generated', 'explicit-wgsl');
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const PAGE = `<!doctype html><meta charset="utf-8"><title>explicit wgsl</title><script type="module">
window.__run = async () => {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize,
  } });
  const errors = [];
  device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
  const list = await (await fetch('/manifest.json')).json();
  const out = [];
  for (const m of list) {
    const code = await (await fetch('/' + m.name + '.wgsl')).text();
    device.pushErrorScope('validation');
    const mod = device.createShaderModule({ code, label: m.name });
    const info = await mod.getCompilationInfo();
    const msgs = info.messages.filter((x) => x.type === 'error').map((x) => x.lineNum + ':' + x.linePos + ' ' + x.message);
    let pipe = null;
    try {
      await device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: m.entry }, label: m.name });
    } catch (e) { pipe = String(e.message || e); }
    const scope = await device.popErrorScope();
    out.push({ name: m.name, errors: msgs, pipeline: pipe, scope: scope ? scope.message : null });
  }
  return { adapter: (adapter.info && adapter.info.description) || '', limits: { wg: L.maxComputeWorkgroupStorageSize, inv: L.maxComputeInvocationsPerWorkgroup }, out, errors };
};
window.__ready = true;
</script>`;
if (!existsSync(join(DIR, 'manifest.json'))) {
  console.error(`no ${DIR}/manifest.json: run the explicit_wgsl_dump example first`);
  process.exit(2);
}
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p === '/' || p === '/index.html') { res.writeHead(200, { 'content-type': 'text/html' }).end(PAGE); return; }
  const path = normalize(join(DIR, p));
  if (!path.startsWith(DIR) || !existsSync(path)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': p.endsWith('.json') ? 'application/json' : 'text/plain' }).end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'explicit-wgsl-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} server.close(); };
process.on('exit', cleanup);
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
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.navigate', { url: `${base}/` });
for (let i = 0; i < 100; i++) {
  if (await evaluate('window.__ready === true').catch(() => false)) break;
  await new Promise((r) => setTimeout(r, 100));
}
const r = await evaluate('window.__run()');
console.log(`Chrome WebGPU: ${r.adapter}; workgroup storage ${r.limits.wg} B, invocations ${r.limits.inv}`);
let bad = 0;
for (const m of r.out) {
  const fail = m.errors.length || m.pipeline || m.scope;
  if (fail) bad++;
  console.log(`${fail ? 'FAIL' : 'ok  '} ${m.name}${m.errors.length ? ' ' + m.errors.slice(0, 3).join(' | ') : ''}${m.pipeline ? ' pipeline: ' + m.pipeline : ''}${m.scope ? ' scope: ' + m.scope : ''}`);
}
if (r.errors.length) { bad++; console.log('device errors:', r.errors.slice(0, 5)); }
console.log(bad ? `${bad} failed` : `all ${r.out.length} modules compile in Chrome`);
process.exit(bad ? 1 : 0);
