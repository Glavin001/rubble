#!/usr/bin/env node
// End-to-end check of the soft-body demo's explicit solver (softbody.html?solver=explicit)
// in Chrome WebGPU: for each test vehicle, the idle scene (settles, then the launch
// key sets it rolling), the 100 km/h wall (it reaches the wall and crumples), and the
// solver button (rebuilds the scene on the soft solver and back), and a few seconds
// of the page's own live loop (launch key, GPU time in the HUD). Fails on any page
// or GPU error, and when a vehicle does not move or deform. Screenshots per run.
//
//   cargo run -p engine-webgpu --release --bin wgsl_export      # WGSL + vehicle wasm
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket explicit-check.mjs [vehicle ...] [--out DIR]
//
// Vehicles: sedan (the procedural sedan, `.truck`), hires (the high-resolution test
// car, generated JBeam), hires_truck (its committed `.truck`, the acceptance file),
// and with CONTENT=<RigsOfRods/content folder> agora_s and agora_l
// (agora/95bbUID-agoras.truck, -agoral.truck; not in this repository). The page
// fetches `.truck` files through ?truck=, served here under /__vehicles/ and
// /__content/. Needs Chrome (CHROME env to override). Exits non-zero on a failure.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const VEHICLES = normalize(join(ROOT, '../../docs/explicit-beams-acceptance/vehicles'));
const CONTENT = process.env.CONTENT ? normalize(process.env.CONTENT) : null;
const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(ROOT, 'recordings', 'explicit');
const only = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });

// Each vehicle: its query (?car= or ?truck=), and whether it can run here.
const ALL = {
  sedan: { query: 'car=sedan' },
  hires: { query: 'car=hires' },
  hires_truck: { query: 'truck=/__vehicles/rubble_hires_jbeam.truck' },
  agora_s: { query: 'truck=/__content/agora/95bbUID-agoras.truck', needs: CONTENT && join(CONTENT, 'agora/95bbUID-agoras.truck') },
  agora_l: { query: 'truck=/__content/agora/95bbUID-agoral.truck', needs: CONTENT && join(CONTENT, 'agora/95bbUID-agoral.truck') },
};
const names = only.length ? only : Object.keys(ALL);

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const serveFrom = (base, rel) => {
  const path = normalize(join(base, rel));
  return path.startsWith(base) && existsSync(path) && !statSync(path).isDirectory() ? path : null;
};
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = p.startsWith('/__vehicles/') ? serveFrom(VEHICLES, p.slice(12))
    : p.startsWith('/__content/') ? CONTENT && serveFrom(CONTENT, p.slice(11))
      : serveFrom(ROOT, p);
  if (!path) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'explicit-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
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
const pageErrors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 300_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });

const shot = async (name) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  const png = join(OUT, `${name}.png`);
  writeFileSync(png, Buffer.from(s.result.data, 'base64'));
  return png;
};
// Wait for the page (after a navigation) to be ready on `solver`; throws on a page error.
const ready = async (solver) => {
  for (let i = 0; i < 300; i++) {
    await new Promise((r) => setTimeout(r, 200));
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}`);
    const s = await evaluate('window.__ready === true ? window.__solver : null').catch(() => null);
    if (s === solver) return;
  }
  throw new Error(`the page did not start (${solver})`);
};
const open = async (query, solver) => {
  await send('Page.navigate', { url: `${base}/softbody.html?${query}` });
  await ready(solver);
};
const pageError = async () => (await evaluate('document.getElementById("err")?.textContent || ""')) || pageErrors[0] || '';
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const finite = (r) => r.player && [...r.player.p, ...r.player.v].every(Number.isFinite);

// Runs: [name, fn(vehicle) → { ok, msg }].
const RUNS = [
  // At rest it settles where it was spawned, nothing breaks; the launch key sets it
  // rolling at 100 km/h.
  ['idle', async (v) => {
    await open(`solver=explicit&scene=yard&${v.query}`, 'explicit');
    const lay = await evaluate('window.__explicit.layout()');
    const a = await evaluate('window.__step(2000)'); // 1 s
    const d = a.damage[0];
    await evaluate('window.__explicit.launch()');
    const b = await evaluate('window.__step(1000)'); // 0.5 s
    const moved = dist(a.player.p, b.player.p);
    const kmh = Math.hypot(...b.player.v) * 3.6;
    await evaluate(`window.__setView({ orbit: { yaw: 2.2, pitch: 0.3, dist: 10 } })`);
    const ok = finite(a) && finite(b) && Math.hypot(...a.player.v) < 1 && d.broken < 0.01 * d.beams && moved > 5;
    return { ok, msg: `${lay.vehicles[0].name.split(' (')[0]}: ${lay.nodes} nodes, ${lay.beams} beams${lay.wide ? ' (wide)' : ''}; at rest ${(Math.hypot(...a.player.v) * 100).toFixed(1)} cm/s, ${d.broken} broken, ${d.yielded} bent; launched: ${moved.toFixed(1)} m in 0.5 s, ${kmh.toFixed(0)} km/h` };
  }],
  // 100 km/h into the wall (about 15 m ahead): it gets there and crumples.
  ['wall', async (v) => {
    await open(`solver=explicit&scene=wall&${v.query}`, 'explicit');
    const a = await evaluate('window.__step(40)');
    // 1.5 s, the furthest it got along +Z (it may rebound off the wall).
    let b = a, reach = 0;
    for (let i = 0; i < 30; i++) {
      b = await evaluate('window.__step(100)');
      reach = Math.max(reach, b.player.p[2] - a.player.p[2]);
    }
    const d = b.damage[0];
    const gap = (await evaluate('window.__explicit.layout()')).boxes[0][2] - 0.5;
    const gpu = await evaluate('window.__explicit.gpuMs()');
    // The screenshot in the beam view (bent and broken beams coloured).
    await evaluate('window.__setView({ beams: true })');
    await evaluate('window.__step(1)');
    const ok = finite(b) && reach > 10 && d.yielded + d.broken > 0.02 * d.beams;
    return { ok, msg: `reached ${reach.toFixed(1)} m (wall face at z ${gap.toFixed(1)}), ${d.yielded} bent, ${d.broken} broken of ${d.beams}, ${(Math.hypot(...b.player.v) * 3.6).toFixed(0)} km/h after${gpu != null ? `, GPU ${gpu.toFixed(2)} ms/frame` : ''}` };
  }],
  // Live (the page's own frame loop at wall-clock 2 kHz): the launch key, frames
  // stepping, GPU time per frame in the HUD.
  ['live', async (v) => {
    await open(`solver=explicit&scene=yard&${v.query}`, 'explicit');
    await evaluate(`dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyL' }))`);
    const live = await evaluate('window.__live(3000)');
    const steps = await evaluate('window.__explicit.sim().steps()');
    const hud = await evaluate(`document.getElementById('xstats').innerText.replace(/\\n/g, ' · ')`);
    const r = await evaluate('window.__step(1)');
    const kmh = Math.hypot(...r.player.v) * 3.6;
    const ok = steps > 1000 && /GPU [≈0-9]/.test(hud) && kmh > 20;
    return { ok, msg: `${live.fps.toFixed(0)} fps, ${(live.realTime * 100).toFixed(0)}% real time, ${steps} steps, launched to ${kmh.toFixed(0)} km/h; HUD: ${hud}` };
  }],
  // The button rebuilds the scene on the soft solver (same vehicle, same spawn), and back.
  ['toggle', async (v) => {
    await open(`solver=explicit&scene=wall&${v.query}`, 'explicit');
    await evaluate('window.__step(200)');
    await evaluate(`document.getElementById('solverBtn').click()`);
    await ready('soft');
    const url = await evaluate('location.search');
    const s = await evaluate('window.__step(240)'); // 1 s at 240 Hz
    const soft = s.damage?.length ?? 0;
    await shot(`${v.name}_toggle_soft`);
    await evaluate(`document.getElementById('solverBtn').click()`);
    await ready('explicit');
    const back = await evaluate('location.search');
    const x = await evaluate('window.__step(200)');
    const ok = !url.includes('solver=explicit') && back.includes('solver=explicit') && soft > 0 && finite(x);
    return { ok, msg: `soft ${url} (${soft} soft bodies, ${s.damage?.[0]?.yielded ?? 0} bent after 1 s) → explicit ${back}` };
  }],
];

let failed = 0;
const report = [];
for (const name of names) {
  const v = ALL[name];
  if (!v) throw new Error(`unknown vehicle ${name}; have ${Object.keys(ALL).join(', ')}`);
  v.name = name;
  if (v.needs !== undefined && !(v.needs && existsSync(v.needs))) {
    console.log(`- ${name}: skipped (set CONTENT to the RigsOfRods/content folder)`);
    report.push({ name, skipped: true });
    continue;
  }
  for (const [run, fn] of RUNS) {
    pageErrors.length = 0;
    const t0 = performance.now();
    let res;
    try {
      res = await fn(v);
    } catch (e) {
      res = { ok: false, msg: String(e.message || e) };
    }
    const err = await pageError().catch(() => '');
    if (err) res = { ok: false, msg: `${res.msg} · page error: ${err}` };
    const png = await shot(`${name}_${run}`).catch(() => null);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    console.log(`${res.ok ? '✓' : '✗'} ${name} ${run}: ${res.msg} (${secs} s) → ${png}`);
    report.push({ name, run, ...res, seconds: Number(secs), screenshot: png });
    if (!res.ok) failed++;
  }
}
writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
ws.close();
server.close();
process.exit(failed ? 1 : 0);
