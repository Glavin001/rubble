// Where every millisecond of a browser frame goes: headless Chrome (the system's, real
// WebGPU on this machine's GPU) loads a page, runs its own real-time loop, and records
//   - the page's visibility and rAF rate (asserted: a hidden or throttled page runs rAF at
//     ~1 Hz and every number would be wrong),
//   - per frame: the rAF callbacks' JS time, the World's host spans from Rust
//     (engine_webgpu::host_profile: world.step and its parts, sim.encode_dispatches,
//     world.poll; a span suspended by a JSPI GPU wait is kept apart), the page's named
//     spans (page.draw) — profiler.mjs,
//   - GPU timestamps per step and per phase (World gpuTiming, from the page's query),
//   - a Chrome trace (CDP Tracing: gpu, dawn, v8, User Timing, toplevel) with a table of
//     where each process's threads spent their time,
//   - optionally the kernel replay (every GPU dispatch of one step timed alone),
// and writes `summary.json`, `table.txt` and `trace.json` to the run's out directory.
//
// Run it through tools/webgpu-profile/run.sh (the GPU lock, the time limit, the repo's Node):
//   run.sh --page vehicle-lab.html --query 'mode=mountain&cars=8&terrain=ledges&timing=step' --name lab8
//   run.sh --page brick-ring-drive.html --query 'car=sedan&timing=phases' --replay 15 --name ring
// Options: --secs N (measured, default 10), --warm N (seconds before, default 5), --trace S
// (seconds of trace, default 3; 0: none), --replay REPS (0: none), --labels (debug groups),
// --size WxH, --out DIR (default <repo>/out/webgpu-profile/<name>), --startLimit S.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(ROOT, '../../..');
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const PAGE = opt('--page', 'vehicle-lab.html');
const QUERY = opt('--query', '');
const NAME = opt('--name', PAGE.replace(/\.html$/, ''));
const OUT = resolve(opt('--out', join(REPO, 'out', 'webgpu-profile', NAME)));
const SECS = Number(opt('--secs', '10'));
const WARM = Number(opt('--warm', '5'));
const TRACE = Number(opt('--trace', '3'));
const REPLAY = Number(opt('--replay', '0'));
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const LIMIT_MIN = Number(opt('--limit', '10'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png', '.dds': 'application/octet-stream' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

// Timestamps unquantised (developer features; Dawn's timestamp_quantization off) and the
// unsafe APIs (the experimental timestamp features, to report what exists). No flag here
// changes the work the page does.
const profile = mkdtempSync(join(tmpdir(), 'rubble-profile-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
  '--enable-dawn-features=allow_unsafe_apis', '--disable-dawn-features=timestamp_quantization',
  `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
process.on('exit', cleanup);

let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const browserWs = (await (await fetch(`http://127.0.0.1:${debugPort}/json/version`)).json()).webSocketDebuggerUrl;
const connect = async (url) => {
  const ws = new WebSocket(url);
  await new Promise((r) => ws.addEventListener('open', r));
  let nextId = 0; const waiting = new Map(); const listeners = [];
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
    for (const l of listeners) l(m);
  });
  const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
  return { send, on: (f) => listeners.push(f) };
};
const page = await connect(targets[0].webSocketDebuggerUrl);
const browser = await connect(browserWs);
const pageErrors = [];
page.on((m) => {
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const evaluate = async (expression) => {
  const r = await page.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 300_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await page.send('Runtime.enable');
await page.send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const url = `${base}/${PAGE}${QUERY ? `?${QUERY}` : ''}`;
console.log(`profile ${NAME}: ${url}`);
await page.send('Page.navigate', { url });
const t0 = Date.now();
for (;;) {
  await sleep(250);
  const err = await evaluate('window.__failed ? String(window.__failed) : (document.getElementById("err")?.textContent || "")').catch(() => '');
  if (err) throw new Error(`page error: ${err}\n${pageErrors.join('\n')}`);
  if (await evaluate('window.__ready === true && !!window.__rubbleProfiler').catch(() => false)) break;
  if (Date.now() - t0 > Number(opt('--startLimit', '180')) * 1000) throw new Error(`the page did not start\n${pageErrors.join('\n')}`);
}
const readyS = (Date.now() - t0) / 1000;

// The adapter: what this browser's WebGPU offers for timing.
const adapter = await evaluate(`(async () => { const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const i = a.info || {}; return { vendor: i.vendor, architecture: i.architecture, description: i.description,
  features: [...a.features].sort(), userAgent: navigator.userAgent, jspi: 'Suspending' in WebAssembly }; })()`);

// Visibility and the rAF rate, before anything is measured.
const raf = await evaluate(`new Promise((done) => { let n = 0; const t0 = performance.now();
  const f = () => { n++; if (performance.now() - t0 < 1000) requestAnimationFrame(f); else done({ hz: n * 1000 / (performance.now() - t0), visibility: document.visibilityState }); };
  requestAnimationFrame(f); })`);
console.log(`  ready in ${readyS.toFixed(1)} s; ${raf.visibility}, rAF ${raf.hz.toFixed(0)} Hz; ${adapter.description || adapter.vendor}`);
if (raf.visibility !== 'visible' || raf.hz < 30) throw new Error(`the page is ${raf.visibility} at ${raf.hz.toFixed(1)} Hz rAF: timing would be throttled`);

if (args.includes('--labels')) await evaluate('window.__rubbleProfiler.setDebugLabels(true)');
await sleep(WARM * 1000);
await evaluate(`window.__rubbleProfiler.start(${TRACE > 0})`);

// The trace: the last TRACE seconds of the measured window.
let traceFile = null;
const traceTable = [];
const measureMs = SECS * 1000;
if (TRACE > 0) await sleep(Math.max(0, measureMs - TRACE * 1000));
else await sleep(measureMs);
if (TRACE > 0) {
  const cats = ['gpu', 'dawn', 'v8', 'blink.user_timing', 'toplevel', 'devtools.timeline', 'disabled-by-default-gpu.dawn', 'disabled-by-default-devtools.timeline', 'gpu.dawn', 'webgpu', 'renderer.scheduler', 'viz'];
  await browser.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { includedCategories: cats, recordMode: 'recordContinuously' } });
  await sleep(TRACE * 1000);
  const done = new Promise((r) => browser.on((m) => { if (m.method === 'Tracing.tracingComplete') r(m.params); }));
  await browser.send('Tracing.end');
  const { stream } = await done;
  let text = '';
  for (;;) {
    const r = await browser.send('IO.read', { handle: stream, size: 1 << 20 });
    text += r.result.base64Encoded ? Buffer.from(r.result.data, 'base64').toString() : r.result.data;
    if (r.result.eof) break;
  }
  await browser.send('IO.close', { handle: stream });
  traceFile = join(OUT, 'trace.json');
  writeFileSync(traceFile, text);
  traceTable.push(...analyzeTrace(JSON.parse(text)));
}
const summary = await evaluate('window.__rubbleProfiler.summary()');
await evaluate('window.__rubbleProfiler.stop()');
let replay = null;
if (REPLAY > 0) {
  replay = await evaluate(`window.__rubbleProfiler.replay(${REPLAY})`);
  if (replay?.error) console.log(`  kernel replay: ${replay.error}`);
}
const pageState = await evaluate('JSON.stringify(window.__state?.() ?? null)').then((s) => JSON.parse(s)).catch(() => null);

const result = {
  name: NAME, page: PAGE, query: QUERY, at: new Date().toISOString(), readyS, raf, adapter,
  flags: { labels: args.includes('--labels'), warmS: WARM, secs: SECS, traceS: TRACE },
  summary, trace: traceTable, replay, pageErrors: pageErrors.slice(0, 20), pageState,
};
writeFileSync(join(OUT, 'summary.json'), JSON.stringify(result, null, 1));
const table = textTable(result);
writeFileSync(join(OUT, 'table.txt'), table);
console.log(table);
console.log(`  wrote ${OUT}/summary.json, table.txt${traceFile ? ', trace.json' : ''}`);
process.exit(0);

/** Per process and thread, the trace's complete events by self-inclusive time (top 12 each). */
function analyzeTrace(trace) {
  const ev = Array.isArray(trace) ? trace : trace.traceEvents || [];
  const pname = new Map(), tname = new Map();
  for (const e of ev) if (e.ph === 'M') {
    if (e.name === 'process_name') pname.set(e.pid, e.args?.name);
    if (e.name === 'thread_name') tname.set(`${e.pid}:${e.tid}`, e.args?.name);
  }
  const ts = ev.filter((e) => e.ts).map((e) => e.ts);
  const spanUs = ts.length ? Math.max(...ts) - Math.min(...ts) : 1;
  const per = new Map(); const open = new Map();
  const add = (e, dur) => {
    const key = `${pname.get(e.pid) || e.pid} / ${tname.get(`${e.pid}:${e.tid}`) || e.tid}`;
    if (!per.has(key)) per.set(key, new Map());
    const m = per.get(key); const k = `${e.name}`;
    const v = m.get(k) || { ms: 0, n: 0, cat: e.cat }; v.ms += dur / 1000; v.n++; m.set(k, v);
  };
  for (const e of ev) {
    if (e.ph === 'X' && e.dur != null) add(e, e.dur);
    else if (e.ph === 'B') { const k = `${e.pid}:${e.tid}`; if (!open.has(k)) open.set(k, []); open.get(k).push(e); }
    else if (e.ph === 'E') { const s = open.get(`${e.pid}:${e.tid}`)?.pop(); if (s) add(s, e.ts - s.ts); }
  }
  const rows = [];
  for (const [thread, m] of per) {
    const top = [...m].sort((a, b) => b[1].ms - a[1].ms).slice(0, 12);
    if (!top.length || top[0][1].ms < 1) continue;
    rows.push({ thread, events: top.map(([name, v]) => ({ name, cat: v.cat, msPerSecond: v.ms / (spanUs / 1e6), count: v.n })) });
  }
  return rows.sort((a, b) => b.events[0].msPerSecond - a.events[0].msPerSecond).slice(0, 10);
}

function textTable(r) {
  const s = r.summary; const f = (x) => (x ?? 0).toFixed(3);
  const L = [];
  L.push(`${r.name}: ${r.page}?${r.query}`);
  L.push(`  ${r.adapter.description || r.adapter.vendor} · ${r.raf.visibility} · rAF ${s.rafHz.toFixed(1)} Hz · ${s.frames} frames in ${s.seconds.toFixed(1)} s · ${s.stepsTimed} GPU-timed steps`);
  L.push(`  timestamp features: ${r.adapter.features.filter((x) => x.includes('timestamp')).join(', ') || 'none'}`);
  L.push('  per display frame (ms)            median      p90      max');
  const row = (k, v) => L.push(`    ${k.padEnd(32)} ${f(v.median).padStart(7)} ${f(v.p90).padStart(8)} ${f(v.max).padStart(8)}`);
  row('frame interval (dt)', s.frameDt);
  row('JS in rAF callbacks', s.js);
  for (const [k, v] of Object.entries(s.host).sort((a, b) => b[1].median - a[1].median)) row(`  ${k}`, v);
  for (const [k, v] of Object.entries(s.page)) row(`  ${k}`, v);
  row('  other JS', s.jsOther);
  if (s.suspendedMs.max > 0) row('suspended in GPU waits (JSPI)', s.suspendedMs);
  L.push('  GPU per step (ms)');
  row('step (timestamps)', s.gpuStep);
  for (const [k, v] of Object.entries(s.gpuSegments).sort((a, b) => b[1].median - a[1].median).slice(0, 20)) row(`  ${k}`, v);
  if (r.trace.length) {
    L.push('  trace: busiest threads (ms per second of trace)');
    for (const t of r.trace.slice(0, 6)) {
      L.push(`    ${t.thread}`);
      for (const e of t.events.slice(0, 6)) L.push(`      ${e.name.slice(0, 48).padEnd(48)} ${e.msPerSecond.toFixed(2).padStart(8)}  (${e.count})`);
    }
  }
  if (r.replay?.table) L.push(r.replay.table);
  return L.join('\n') + '\n';
}
