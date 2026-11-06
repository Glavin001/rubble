#!/usr/bin/env node
// End-to-end check of the crash-test lab (softbody.html ?solver=explicit with the
// lab on, crash-lab.mjs and crash-lab-render.mjs) in Chrome WebGPU: each preset
// (wall 56, pole 29 at 75°, head-on 2 × 50, drop 15 m) is stepped through its
// impact in small frames; the impact is detected (not before the vehicle can have
// reached it), the time readout counts from it, slow motion and stepping take the
// steps asked, the recorded run scrubs back to before the impact (a negative
// readout), and the colour, skin and camera modes render. A grab run pulls a
// resting car's front node up and tears it. Fails on any page or GPU error.
// Screenshots of each preset at moments matching the reference shots, clean (the
// panels hidden) and one with the panels.
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket crash-lab-check.mjs [run ...] [--car sedan|e36|...] [--out DIR] [--stock]
//
// --stock: a user's Chrome (a visible window, no --enable-unsafe-webgpu or other
// flags; check-chrome.mjs), where a windowed GPU watchdog applies. Every run also
// fails a large compute pipeline compiled synchronously without an async warm-up.
//
// Runs: wall56, pole29, headon50, drop15, tbone214, tbone80, overlap64, swipe50, grab, modes. --car showcase: Rubble's authored
// sedan with its skinned body. --car e36 (and viper,
// mustang, agora) uses the local Rigs of Rods cars in engine/local-vehicles
// (gitignored; skipped when absent). Needs Chrome (CHROME env to override).
import { frameContent } from './frame-content.mjs';
import { chromeArgs, SYNC_COMPILE_PROBE } from './check-chrome.mjs';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const LOCAL = normalize(join(ROOT, '../../local-vehicles'));
// Community BeamNG vehicles (with their .dae flexbodies), gitignored and local
// only: RUBBLE_MODS, else the checkout's inspiration/beamng-mods.
const MODS = normalize(process.env.RUBBLE_MODS || join(ROOT, '../../../inspiration/beamng-mods'));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', join(ROOT, 'recordings', 'crash-lab'));
const CAR = opt('--car', 'sedan');
// --size WxH: the window (screenshots) size; 1280x720 by default.
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
// --stock: a user's Chrome (a visible window, no WebGPU flags; check-chrome.mjs).
const STOCK = args.includes('--stock');
const only = args.filter((a, i) => !a.startsWith('--') && (!args[i - 1]?.startsWith('--') || args[i - 1] === '--stock'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
const CARS = { sedan: 'car=sedan', showcase: 'car=showcase', showcase1: 'car=showcase&style=sedan1',
  // Community BeamNG cars from a local conversion cache (community_cache: one LOD,
  // textures at 1K; engine-webgpu/__mods, never committed), on hub-only wheels.
  etkk: 'mod=/__mods/etkk_tex&pc=etkk/kr6.pc', ccf: 'mod=/__mods/ccf_tex&pc=ccf/f4_sport_M.pc',
  vertex: 'mod=/__mods/vertex_tex&pc=gavril_vertex/gavril_vertex.pc', hires: 'car=hires', e36: 'truck=/__local/E36Sedan.truck', viper: 'truck=/__local/ViperGTS.car',
  mustang: 'truck=/__local/1982-Ford-Mustang-GT-351W.truck', agora: 'truck=/__local/95bbUID-agoras.truck',
  // Imported BeamNG mods drawn with their flexbodies (?mod=: the lab reads the
  // folder's files.json, which this server lists).
  supercar: 'mod=/__mods/monster_supercar/vehicles/supercar&jbeamMode=native', tutorial: 'mod=/__mods/morzbng_tutorial/vehicles/tutorial&pc=none' }; // (its .pc is not valid JSON: the slot defaults)
if (!CARS[CAR]) throw new Error(`--car: one of ${Object.keys(CARS).join(', ')}`);
const carFile = CARS[CAR].startsWith('truck=') ? join(LOCAL, CARS[CAR].split('/__local/')[1])
  : CARS[CAR].startsWith('mod=') ? join(MODS, CARS[CAR].split('/__mods/')[1].split('&')[0]) : null;
if (carFile && !existsSync(carFile)) { console.log(`- ${CAR}: ${carFile} not found, skipped`); process.exit(0); }

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const serveFrom = (base, rel) => {
  const path = normalize(join(base, rel));
  return path.startsWith(base) && existsSync(path) && !statSync(path).isDirectory() ? path : null;
};
const listFiles = (dir, rel = '') => readdirSync(join(dir, rel), { withFileTypes: true }).flatMap((e) =>
  e.isDirectory() ? listFiles(dir, join(rel, e.name)) : [join(rel, e.name)]);
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p.startsWith('/__mods/') && p.endsWith('/files.json')) {
    const dir = normalize(join(MODS, p.slice(8, -'/files.json'.length)));
    if (!dir.startsWith(MODS) || !existsSync(dir)) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(listFiles(dir)));
    return;
  }
  if (p.startsWith('/__mods/')) {
    const path = serveFrom(MODS, p.slice(8));
    if (!path) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'content-type': 'application/octet-stream', 'cache-control': 'no-store' });
    res.end(readFileSync(path));
    return;
  }
  const path = p.startsWith('/__local/') ? serveFrom(LOCAL, p.slice(9)) : serveFrom(ROOT, p);
  if (!path) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'crash-lab-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: STOCK, port: debugPort, profile, width: W, height: H }), { stdio: 'ignore' });
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
  if (m.method === 'Log.entryAdded' && /webgpu|wgsl|validat/i.test(m.params.entry.text) && m.params.entry.level !== 'verbose') pageErrors.push(m.params.entry.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 300_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Log.enable');
await send('Page.enable');
// Every run: no large compute pipeline compiled synchronously (check-chrome.mjs).
await send('Page.addScriptToEvaluateOnNewDocument', { source: SYNC_COMPILE_PROBE });
if (STOCK) console.log('stock Chrome: a visible window, no WebGPU flags');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

// A screenshot; an empty frame (the studio without the car and scenery: under
// EMPTY of it differs from the backdrop, frame-content.mjs) is recorded in
// `empty` and fails its run.
const EMPTY = 0.03;
const empty = [];
const shot = async (name) => {
  const s = await send('Page.captureScreenshot', { format: 'png' });
  const png = join(OUT, `${name}.png`);
  const buf = Buffer.from(s.result.data, 'base64');
  writeFileSync(png, buf);
  const c = frameContent(buf);
  if (c < EMPTY) empty.push(`${name} (${(c * 100).toFixed(1)} % content)`);
  return png;
};
// Wait for the page (after a navigation) to be ready on `solver`; throws on a page error.
const ready = async (solver) => {
  // Up to 3 min: a cold shader cache compiles the solver's kernels first.
  for (let i = 0; i < 900; i++) {
    await new Promise((r) => setTimeout(r, 200));
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}`);
    const s = await evaluate('window.__ready === true ? window.__solver : null').catch(() => null);
    if (s === solver) return;
  }
  throw new Error(`the page did not start (${solver})`);
};

const open = async (query) => {
  await send('Page.navigate', { url: `${base}/softbody.html?${query}&${CARS[CAR]}` });
  await ready('explicit');
  if (!(await evaluate('!!window.__lab'))) throw new Error('the lab is not on');
};
const lab = (expr) => evaluate(`(async () => { const L = window.__lab; return ${expr}; })()`);
const pageError = async () => (await evaluate('document.getElementById("err")?.textContent || ""')) || pageErrors[0] || '';
const clean = (on) => lab(`(L.lab.setClean(${on}), true)`);
const READOUT = /^[+−]?\d+ ms \| (\d+x SLOWER|REAL TIME|PAUSED|REPLAY) · FROM THE SIMULATION$/;

// Step a preset until its impact (frames of `chunk` steps, at most `limit` steps).
const toImpact = async (chunk, limit) => {
  let s = await lab('L.state()');
  while (s.impactStep == null && s.step < limit) s = await lab(`L.advance(${chunk * 10}, ${chunk})`);
  return s;
};
// Step to `ms` after the impact (in frames of `chunk`).
const toAfter = async (s, ms, chunk) => {
  const target = s.impactStep + ms * 2;
  const n = target - s.step;
  return n > 0 ? lab(`L.advance(${n}, ${chunk})`) : s;
};
// Replay the recorded frame nearest `ms` from the impact (negative: before it).
const replayAt = async (ms) => lab(`(() => { const s = L.lab.st; const want = s.impactStep + ${ms} * 2; let best = 0;
  s.ring.forEach((r, i) => { if (Math.abs(r.step - want) < Math.abs(s.ring[best].step - want)) best = i; }); L.lab.enterReplay(best); return L.draw().then(() => L.state()); })()`);

// A preset: [query, earliest plausible impact (s), camera, shots ms after impact].
const PRESET_RUNS = {
  wall56: ['preset=wall56&slow=20', 0.3, 'three', [-5, 23, 127]],
  pole29: ['preset=pole29&slow=20', 0.08, 'threeFront', [-5, 23, 127]],
  headon50: ['preset=headon50&slow=20', 0.2, 'side', [-5, 23, 127]],
  drop15: ['preset=drop15&slow=20', 1.4, 'side', [-5, 23, 127]],
  tbone214: ['preset=tbone214&slow=20', 0.2, 'three', [-5, 23, 127]],
  tbone80: ['preset=tbone80&slow=20', 0.2, 'three', [-5, 23, 127]],
  overlap64: ['preset=overlap64&slow=20', 0.2, 'threeFront', [-5, 23, 127]],
  swipe50: ['preset=swipe50&slow=20', 0.1, 'threeFront', [-5, 23, 127]],
  // FMVSS 214's moving barrier (vehicle 1, drawn by the triangle path) and the
  // door pull (the rig node is the car's last node, its cables beams).
  mdb214: ['preset=mdb214&slow=20', 0.05, 'three', [-5, 23, 127]],
  doorpull: ['preset=doorpull&slow=20', 0.0, 'threeFront', [-5, 250, 700]],
};
const RUNS = Object.fromEntries(Object.entries(PRESET_RUNS).map(([name, [query, earliest, cam, shots]]) => [name, async () => {
  await open(query);
  const s0 = await lab('L.state()');
  const s = await toImpact(4, 2 * 2000 * 3);
  if (s.impactStep == null) return { ok: false, msg: `no impact in ${s.step} steps` };
  const at = s.impactStep / 2000;
  let msgs = [`impact at ${(at * 1000).toFixed(0)} ms (step ${s.impactStep})`, `pills ${s.pills}`, `counts ${s.counts}`];
  let ok = at >= earliest && READOUT.test(s.readout) && /NODES \/ [\d,]+ BEAMS/.test(s.counts) && s0.slow === 20;
  if (!READOUT.test(s.readout)) msgs.push(`readout "${s.readout}"`);
  await lab(`(L.lab.camera('${cam}'), true)`);
  await clean(true);
  let cur = s;
  for (const ms of shots.filter((x) => x > 0)) {
    cur = await toAfter(cur, ms, 2);
    // Two drawn frames (the camera eases toward the preset).
    for (let i = 0; i < 40; i++) await lab('L.draw()');
    const png = await shot(`${CAR}_${name}_+${ms}ms`);
    msgs.push(`+${ms} ms: ${cur.readout.split(' | ')[0]}, ${cur.bent} bent, ${cur.broken} broken → ${png}`);
  }
  ok = ok && (cur.bent + cur.broken > 0);
  // Scrub back to before the impact: a negative readout.
  for (const ms of shots.filter((x) => x < 0)) {
    const r = await replayAt(ms);
    const png = await shot(`${CAR}_${name}_${ms}ms`);
    const neg = r.readout.startsWith('−') && r.readout.includes('REPLAY');
    ok = ok && neg && r.replay >= 0;
    msgs.push(`replay ${ms} ms: ${r.readout.split(' | ')[0]} → ${png}`);
  }
  await lab('(L.lab.enterReplay(1e9), L.lab.playPause(), L.lab.setPaused(true), true)');
  await clean(false);
  await lab('L.draw()');
  const png = await shot(`${CAR}_${name}_panels`);
  msgs.push(`panels → ${png}`);
  return { ok, msg: msgs.join('; ') };
}]));

// Grab: a resting car's frontmost node pulled 2.5 m up and 1.5 m forward hard
// enough to tear: the node moves, beams break; release stops it.
RUNS.grab = async () => {
  await open('solver=explicit&xscene=idle&slow=1');
  await lab('L.advance(1000, 34)');
  const node = await lab(`(() => { const p = L.lab.st.lastState.pos; let best = 0; for (let i = 0; i < p.length / 3; i++) if (p[3 * i + 2] > p[3 * best + 2]) best = i; return best; })()`);
  const start = await lab(`Array.from(L.lab.st.lastState.pos.slice(${node} * 3, ${node} * 3 + 3))`);
  const target = [start[0], start[1] + 2.5, start[2] + 1.5];
  await lab(`(L.lab.st.grabK = 1e6, L.lab.grabAt(${node}, ${JSON.stringify(target)}))`);
  await lab(`(L.lab.camera('three'), true)`);
  let s = await lab('L.advance(600, 17)');
  await clean(true);
  for (let i = 0; i < 40; i++) await lab('L.draw()');
  const png1 = await shot(`${CAR}_grab_pull`);
  s = await lab('L.advance(1400, 17)');
  const now = await lab(`Array.from(L.lab.st.lastState.pos.slice(${node} * 3, ${node} * 3 + 3))`);
  const moved = Math.hypot(now[0] - start[0], now[1] - start[1], now[2] - start[2]);
  for (let i = 0; i < 10; i++) await lab('L.draw()');
  const png2 = await shot(`${CAR}_grab_torn`);
  await lab('(L.lab.release(), true)');
  const after = await lab('L.advance(400, 17)');
  await clean(false);
  const ok = moved > 1.0 && s.broken > 0 && !after.grab && s.grab;
  return { ok, msg: `node ${node} moved ${moved.toFixed(2)} m, ${s.broken} broken, ${s.bent} bent; released: ${!after.grab} → ${png1}, ${png2}` };
};

// Modes: pause, single frame and single physics step; colour and skin modes and
// cameras render without errors.
RUNS.modes = async () => {
  await open('preset=wall56&slow=20&paused=1');
  const a = await lab('L.state()');
  await lab('(L.lab.physicsStep(), true)');
  const b = await lab('L.frames(3)');
  await lab('(L.lab.frameStep(), true)');
  const c = await lab('L.frames(3)');
  const msgs = [`paused ${a.paused}; physics step +${b.step - a.step}; frame step +${c.step - b.step}`];
  let ok = a.paused && b.step - a.step === 1 && c.step - b.step === 2;
  // Into the crash, then each mode.
  await lab('L.advance(1400, 34)');
  const t = await toImpact(4, 6000);
  await toAfter(t, 60, 4);
  await clean(true);
  for (const [color, skin, cam] of [['part', 'off', 'side'], ['damage', 'solid', 'three'], ['strain', 'off', 'top'], ['strain', 'ghost', 'follow']]) {
    await lab(`(L.lab.st && (L.lab.camera('${cam}')), L.lab.renderer?.set, true)`);
    await evaluate(`(() => { const r = document.getElementById('labColor'); r.value = '${color}'; r.onchange({ target: r }); const k = document.getElementById('labSkin'); k.value = '${skin}'; k.onchange({ target: k }); })()`);
    for (let i = 0; i < 40; i++) await lab('L.draw()');
    const png = await shot(`${CAR}_mode_${color}_${skin}_${cam}`);
    msgs.push(`${color}/${skin}/${cam} → ${png}`);
  }
  await clean(false);
  return { ok, msg: msgs.join('; ') };
};

const names = only.length ? only : Object.keys(RUNS);
let failed = 0;
const report = [];
for (const name of names) {
  if (!RUNS[name]) throw new Error(`unknown run ${name}; have ${Object.keys(RUNS).join(', ')}`);
  pageErrors.length = 0;
  empty.length = 0;
  const t0 = performance.now();
  let res;
  try { res = await RUNS[name](); } catch (e) { res = { ok: false, msg: String(e.message || e) }; }
  const err = await pageError().catch(() => '');
  if (err) res = { ok: false, msg: `${res.msg} · page error: ${err}` };
  const sync = await evaluate('(window.__syncCompiles || []).join(", ")').catch(() => '');
  if (sync) res = { ok: false, msg: `${res.msg} · compiled synchronously (a windowed Chrome's GPU watchdog can kill a long compile; warm it with createComputePipelineAsync first): ${sync}` };
  if (empty.length) res = { ok: false, msg: `${res.msg} · empty frames: ${empty.join(', ')}` };
  if (!res.ok && pageErrors.length) res.msg += ` · console: ${pageErrors.slice(0, 3).map((e) => String(e).split('\n')[0]).join(' | ')}`;
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`${res.ok ? '✓' : '✗'} ${CAR} ${name}: ${res.msg} (${secs} s)`);
  report.push({ car: CAR, name, ...res, seconds: Number(secs) });
  if (!res.ok) failed++;
}
writeFileSync(join(OUT, `report-${CAR}.json`), JSON.stringify(report, null, 2));
ws.close();
server.close();
process.exit(failed ? 1 : 0);
