// The vehicle lab's memory while it loads a car and once it has settled, from headless
// Chrome: per load phase (the page's progress text) the JS heap, the wasm linear memory,
// the GPU buffers and textures the page created (summed at createBuffer/createTexture,
// less destroy(), by label), and the renderer and GPU processes' physical footprint
// (macOS `footprint`, what iOS's jetsam limits too) sampled through the load for peaks.
// Settled: after --settle seconds of running and a forced GC, with the page's own
// breakdown of what it holds (window.__mem(), vehicle-lab.html) when it has one.
//
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- env RUBBLE_GPU_LOCK=0 \
//     node engine/tools/vehicle-lab/lab-memory.mjs [--base http://127.0.0.1:8863] \
//       [--query 'mode=mountain&car=gen'] [--settle 8] [--json out.json] [--deviceMemory GB]
//
// --deviceMemory: the page sees this navigator.deviceMemory (the lab's footprint guard).
//
// Serves nothing: --base is a running static server over engine/web-dist.
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const BASE = opt('--base', 'http://127.0.0.1:8863');
const QUERY = opt('--query', 'mode=mountain&car=gen');
const SETTLE = Number(opt('--settle', '8'));
const LIMIT_S = Number(opt('--limit', '280'));
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
setTimeout(() => { console.error(`time limit (${LIMIT_S} s)`); process.exit(9); }, LIMIT_S * 1000).unref();

// The page's own allocations, from before its first script.
const INSTRUMENT = `(() => {
  const M = window.__memProbe = { buf: 0, tex: 0, peakBuf: 0, peakTex: 0, byLabel: {}, mems: [], phases: [] };
  const sizeOf = new WeakMap();
  const texBytes = (d) => {
    const s = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1];
    const [w, h, l] = [s[0], s[1] ?? 1, s[2] ?? 1];
    const bpp = /32float|rgba16/.test(d.format) ? (/rgba32/.test(d.format) ? 16 : /rgba16/.test(d.format) ? 8 : 4) : /depth24plus-stencil8|depth32float-stencil8/.test(d.format) ? 5 : /r8|stencil8/.test(d.format) && !/rgba8|rg8/.test(d.format) ? 1 : /rg8/.test(d.format) ? 2 : 4;
    let total = 0;
    for (let m = 0, mw = w, mh = h; m < (d.mipLevelCount ?? 1); m++, mw = Math.max(1, mw >> 1), mh = Math.max(1, mh >> 1)) total += mw * mh * l * bpp;
    return total * (d.sampleCount ?? 1);
  };
  const add = (kind, label, n) => {
    M[kind] += n; M['peak' + kind[0].toUpperCase() + kind.slice(1)] = Math.max(M['peak' + kind[0].toUpperCase() + kind.slice(1)], M[kind]);
    const k = kind + ':' + (label || '(none)');
    M.byLabel[k] = (M.byLabel[k] || 0) + n;
  };
  if (window.GPUDevice) {
    const cb = GPUDevice.prototype.createBuffer, ct = GPUDevice.prototype.createTexture;
    GPUDevice.prototype.createBuffer = function (d) { const b = cb.call(this, d); sizeOf.set(b, ['buf', d.label, d.size]); add('buf', d.label, d.size); return b; };
    GPUDevice.prototype.createTexture = function (d) { const t = ct.call(this, d); const n = texBytes(d); sizeOf.set(t, ['tex', d.label, n]); add('tex', d.label, n); return t; };
    for (const C of [GPUBuffer, GPUTexture]) {
      const ds = C.prototype.destroy;
      C.prototype.destroy = function () { const s = sizeOf.get(this); if (s) { add(s[0], s[1], -s[2]); sizeOf.delete(this); } return ds.call(this); };
    }
  }
  const keep = (r) => { const m = (r.instance ?? r).exports?.memory; if (m) M.mems.push(m); return r; };
  const is = WebAssembly.instantiateStreaming, i = WebAssembly.instantiate;
  if (is) WebAssembly.instantiateStreaming = (...a) => is.apply(WebAssembly, a).then(keep);
  WebAssembly.instantiate = (...a) => i.apply(WebAssembly, a).then(keep);
  M.snap = (text) => ({ t: performance.now(), text, heap: performance.memory.usedJSHeapSize, heapTotal: performance.memory.totalJSHeapSize,
    wasm: M.mems.reduce((s, m) => s + m.buffer.byteLength, 0), buf: M.buf, tex: M.tex });
  addEventListener('DOMContentLoaded', () => {
    const busy = document.getElementById('busy'), err = document.getElementById('err');
    let last = '';
    new MutationObserver(() => { const t = (busy?.style.display !== 'none' && busy?.textContent) || ''; if (t !== last) { M.phases.push(M.snap(last + ' → done')); last = t; } }).observe(busy, { childList: true, characterData: true, subtree: true, attributes: true });
    new MutationObserver(() => { if (err.textContent.trim()) M.phases.push(M.snap('error: ' + err.textContent.slice(0, 200))); }).observe(err, { childList: true, characterData: true, subtree: true });
  });
})();`;

const profile = mkdtempSync(join(tmpdir(), 'lab-memory-chrome-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--enable-unsafe-webgpu',
  '--enable-precise-memory-info', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'], { stdio: 'ignore' });
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
const errors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 120_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.enable');
await send('Page.addScriptToEvaluateOnNewDocument', { source: INSTRUMENT });
if (opt('--deviceMemory')) await send('Page.addScriptToEvaluateOnNewDocument', { source: `Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => ${Number(opt('--deviceMemory'))} });` });

// Chrome's processes under this profile: the renderer(s) and the GPU process.
const procs = () => execFileSync('ps', ['-A', '-o', 'pid=,command='], { encoding: 'utf8' }).split('\n')
  .filter((l) => l.includes(profile) && /--type=(renderer|gpu-process)/.test(l) && !l.includes('--extension-process'))
  .map((l) => { const [pid, ...c] = l.trim().split(/\s+/); return { pid: Number(pid), type: /--type=renderer/.test(c.join(' ')) ? 'renderer' : 'gpu' }; });
const footprint = (pid) => { try { const o = execFileSync('footprint', [String(pid)], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }); const m = /Footprint:\s*([\d.]+)\s*(KB|MB|GB|B)/.exec(o); return m ? Number(m[1]) * { B: 1, KB: 1024, MB: 2 ** 20, GB: 2 ** 30 }[m[2]] : 0; } catch { return 0; } };
const peak = { renderer: 0, gpu: 0 };
const sampleFootprint = () => {
  const now = { renderer: 0, gpu: 0 };
  for (const p of procs()) now[p.type] = Math.max(now[p.type], footprint(p.pid));
  peak.renderer = Math.max(peak.renderer, now.renderer); peak.gpu = Math.max(peak.gpu, now.gpu);
  return now;
};
let sampling = true;
const sampler = (async () => { while (sampling) { sampleFootprint(); await new Promise((r) => setTimeout(r, 150)); } })();

const url = `${BASE}/vehicle-lab.html?${QUERY}`;
const t0 = Date.now();
await send('Page.navigate', { url });
let ready = false;
for (;;) {
  await new Promise((r) => setTimeout(r, 300));
  const s = await evaluate('JSON.stringify({ ready: window.__ready === true, failed: window.__failed || "" })').then(JSON.parse).catch(() => ({}));
  if (s.failed) { console.error('page failed: ' + s.failed); console.error('banner: ' + await evaluate('document.getElementById("fatal").style.display + " | " + document.getElementById("fatal").textContent')); break; }
  if (s.ready) { ready = true; break; }
  if (Date.now() - t0 > (LIMIT_S - 30) * 1000) { console.error('did not start'); break; }
}
const loadS = (Date.now() - t0) / 1000;
const atReady = { ...(await evaluate('JSON.stringify(window.__memProbe.snap("ready"))').then(JSON.parse)), fp: sampleFootprint() };
const peakAtReady = { ...peak };
await new Promise((r) => setTimeout(r, SETTLE * 1000));
await send('HeapProfiler.enable');
await send('HeapProfiler.collectGarbage');
await new Promise((r) => setTimeout(r, 500));
const settled = { ...(await evaluate('JSON.stringify(window.__memProbe.snap("settled"))').then(JSON.parse)), fp: sampleFootprint() };
sampling = false;
await sampler;
const probe = await evaluate('JSON.stringify({ phases: window.__memProbe.phases, peakBuf: window.__memProbe.peakBuf, peakTex: window.__memProbe.peakTex, byLabel: window.__memProbe.byLabel })').then(JSON.parse);
const owners = await evaluate('JSON.stringify(window.__mem ? window.__mem() : null)').then(JSON.parse).catch(() => null);

const MB = (b) => (b / 2 ** 20).toFixed(1);
console.log(`${url}: ${ready ? 'ready' : 'NOT ready'} in ${loadS.toFixed(1)} s`);
console.log('phase (snapshot when the step ended)                          JS heap  wasm  GPU buf  GPU tex  (MB)');
for (const p of probe.phases) console.log(`  ${p.text.slice(0, 60).padEnd(60)} ${MB(p.heap).padStart(7)} ${MB(p.wasm).padStart(6)} ${MB(p.buf).padStart(7)} ${MB(p.tex).padStart(7)}`);
for (const [k, s] of [['ready', atReady], ['settled (after GC)', settled]]) console.log(`  ${k.padEnd(60)} ${MB(s.heap).padStart(7)} ${MB(s.wasm).padStart(6)} ${MB(s.buf).padStart(7)} ${MB(s.tex).padStart(7)}   footprint renderer ${MB(s.fp.renderer)} gpu ${MB(s.fp.gpu)}`);
console.log(`peak footprint: renderer ${MB(peak.renderer)} MB (by ready ${MB(peakAtReady.renderer)}), GPU process ${MB(peak.gpu)} MB; peak GPU buffers ${MB(probe.peakBuf)} MB, textures ${MB(probe.peakTex)} MB; peak JS heap at a phase end ${MB(Math.max(...probe.phases.map((p) => p.heap), atReady.heap))} MB`);
const labels = Object.entries(probe.byLabel).filter(([, v]) => v > 1 << 20).sort((a, b) => b[1] - a[1]);
console.log('GPU allocations held at the end, over 1 MB:');
for (const [k, v] of labels) console.log(`  ${k.padEnd(48)} ${MB(v).padStart(8)} MB`);
if (owners) { console.log('the page\'s own breakdown (window.__mem):'); for (const [k, v] of Object.entries(owners)) console.log(`  ${k.padEnd(48)} ${typeof v === 'number' ? MB(v).padStart(8) + ' MB' : JSON.stringify(v)}`); }
if (errors.length) console.log('page exceptions:\n' + errors.slice(0, 5).join('\n'));
if (opt('--json')) writeFileSync(opt('--json'), JSON.stringify({ url, ready, loadS, phases: probe.phases, atReady, settled, peak, peakAtReady, peakBuf: probe.peakBuf, peakTex: probe.peakTex, byLabel: probe.byLabel, owners }, null, 1));
process.exit(ready ? 0 : 1);
