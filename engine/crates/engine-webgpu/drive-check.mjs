#!/usr/bin/env node
// End-to-end check of driving on the explicit solver (softbody.html ?solver=explicit,
// raycast wheels: engine_softbody::explicit_drive, slang/kernels/explicit_beams_drive.slang)
// in Chrome WebGPU. Per drive scene and car: the page loads, the car settles on its
// four wheels, speed rises with throttle (W), the car turns with steer (A/D), the
// frames are not empty, and driving into the yard's wall crashes it (beams bend or
// break while the car decelerates). Scripted input goes through the page's own path
// (window.__explicit.drive: the same setDriveInput the keys use). Fails on any page
// or GPU error.
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket drive-check.mjs [run ...] [--car sedan|showcase|showcase1|hires|...] [--out DIR] [--stock]
//
// --stock: a user's Chrome (a visible window, no --enable-unsafe-webgpu or other
// flags; check-chrome.mjs), where a windowed GPU watchdog applies. Every run also
// fails a large compute pipeline compiled synchronously without an async warm-up.
//
// Runs: yard (settle, throttle, steer, brake), crash (full throttle into the wall),
// drive (?drive=1 in an ordinary explicit scene: the wall crash scene, driven).
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
const OUT = opt('--out', join(ROOT, 'recordings', 'drive'));
const CAR = opt('--car', 'sedan');
// --size WxH: the window (screenshots) size; 1280x720 by default.
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
// --stock: a user's Chrome (a visible window, no WebGPU flags; check-chrome.mjs).
const STOCK = args.includes('--stock');
const only = args.filter((a, i) => !a.startsWith('--') && (!args[i - 1]?.startsWith('--') || args[i - 1] === '--stock'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
const CARS = { sedan: 'car=sedan', showcase: 'car=showcase', showcase1: 'car=showcase&style=sedan1', hires: 'car=hires', e36: 'truck=/__local/E36Sedan.truck', viper: 'truck=/__local/ViperGTS.car',
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

const profile = mkdtempSync(join(tmpdir(), 'drive-check-chrome-'));
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
  await send('Page.navigate', { url: `${base}/softbody.html?${query}&lab=1&${CARS[CAR]}` });
  await ready('explicit');
  if (!(await evaluate('!!window.__lab'))) throw new Error('the lab is not on');
};
const lab = (expr) => evaluate(`(async () => { const L = window.__lab; const X = window.__explicit; return ${expr}; })()`);
const pageError = async () => (await evaluate('document.getElementById("err")?.textContent || ""')) || pageErrors[0] || '';
const clean = (on) => lab(`(L.lab.setClean(${on}), true)`);
// The player's drive state: { speed, heading (rad, about +Y from +Z), centre, contacts }.
const player = async () => lab(`X.readDrive().then((d) => { const v = d.vehicles[0]; return { speed: v.speed, heading: Math.atan2(v.forward[0], v.forward[2]),
  centre: v.centre, contacts: v.wheels.filter((w) => w.contact).length, wheels: v.wheels.length, cut: v.tractionCut, step: d.step }; })`);
// Hold `input` for `s` seconds of simulation (frames of 34 steps), then draw.
const drive = async (input, s) => {
  await lab(`(X.drive(${JSON.stringify(input)}), true)`);
  const st = await lab(`L.advance(${Math.round(s * 2000)}, 34)`);
  return st;
};
const unwrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

const RUNS = {};
RUNS.yard = async () => {
  await open('solver=explicit&xscene=yard');
  await clean(true);
  await drive({ forward: 0, steer: 0 }, 1.5);
  const rest = await player();
  const msgs = [`rest: ${rest.contacts}/${rest.wheels} wheels down, ${rest.speed.toFixed(3)} m/s`];
  const wheelsDrawn = await lab('X.wheelBodies()');
  msgs.push(`${wheelsDrawn} wheels drawn`);
  let ok = rest.contacts === rest.wheels && Math.abs(rest.speed) < 0.2 && wheelsDrawn >= rest.wheels;
  const z0 = rest.centre[2];
  await drive({ forward: 1 }, 2.5);
  const fast = await player();
  msgs.push(`throttle 2.5 s: ${(fast.speed * 3.6).toFixed(1)} km/h, ${(fast.centre[2] - z0).toFixed(1)} m on`);
  ok = ok && fast.speed > 4 && fast.centre[2] - z0 > 3;
  for (let i = 0; i < 30; i++) await lab('L.draw()');
  msgs.push(`→ ${await shot(`${CAR}_yard_throttle`)}`);
  // Right first (toward −X, the open side), then left back: the parked cars
  // stand at x = 14 (z 20 and 30), left of the path, where a left turn from
  // ~40 km/h ended in them (the turn read +5° once the car was pinned there).
  // +Y yaw from +Z toward +X is a left turn.
  const xz = (p) => `(${p.centre[0].toFixed(1)}, ${p.centre[2].toFixed(1)})`;
  await drive({ forward: 0.4, steer: 1 }, 2.0);
  const right = await player();
  const dr = unwrap(right.heading - fast.heading);
  msgs.push(`steer right 2 s: heading ${(dr * 180 / Math.PI).toFixed(0)}°, ${(right.speed * 3.6).toFixed(1)} km/h, at ${xz(right)}`);
  ok = ok && dr < -0.3;
  await drive({ forward: 0.4, steer: -1 }, 2.0);
  const turned = await player();
  const dh = unwrap(turned.heading - right.heading);
  msgs.push(`steer left 2 s: heading ${(dh * 180 / Math.PI).toFixed(0)}°, at ${xz(turned)}`);
  ok = ok && dh > 0.3;
  for (let i = 0; i < 30; i++) await lab('L.draw()');
  msgs.push(`→ ${await shot(`${CAR}_yard_turn`)}`);
  await drive({ forward: -1 }, 2.5);
  const stop = await player();
  msgs.push(`S (brake, then reverse) 2.5 s: ${(stop.speed * 3.6).toFixed(1)} km/h`);
  ok = ok && stop.speed < 1.0;
  await lab('(X.drive(null), true)');
  await clean(false);
  for (let i = 0; i < 10; i++) await lab('L.draw()');
  msgs.push(`panels → ${await shot(`${CAR}_yard_panels`)}`);
  return { ok, msg: msgs.join('; ') };
};

// Steer toward (x, z) from the player's state (+ steer is right).
const steerTo = (p, x, z) => {
  const want = Math.atan2(x - p.centre[0], z - p.centre[2]);
  return Math.max(-1, Math.min(1, -2.5 * unwrap(want - p.heading)));
};
RUNS.crash = async () => {
  // Right of the poles (z = 34, x ≤ 0) and on, full throttle, into the yard's wall (z = 60).
  await open('solver=explicit&xscene=yard');
  await clean(true);
  await drive({ forward: 0 }, 1.0);
  let s = await lab('L.state()');
  let p = await player();
  let top = 0, t = 0, before = s;
  while (t < 14) {
    const target = p.centre[2] < 26 ? [5, 30] : [5, 70];
    s = await drive({ forward: 1, steer: steerTo(p, ...target) }, 0.1);
    p = await player(); t += 0.1;
    if (p.speed > top) { top = p.speed; before = s; }
    if (top > 8 && p.speed < 0.5 * top) break;
  }
  s = await drive({ forward: 0 }, 0.5);
  p = await player();
  for (let i = 0; i < 30; i++) await lab('L.draw()');
  const png = await shot(`${CAR}_yard_crash`);
  const ok = top > 10 && p.speed < top * 0.5 && p.centre[2] > 50 && (s.bent + s.broken) > (before.bent + before.broken);
  return { ok, msg: `${(top * 3.6).toFixed(0)} km/h into the wall (z ${p.centre[2].toFixed(1)}) after ${t.toFixed(1)} s; after: ${(p.speed * 3.6).toFixed(1)} km/h, bent ${before.bent}→${s.bent}, broken ${before.broken}→${s.broken} → ${png}` };
};

RUNS.drive = async () => {
  // ?drive=1 on an ordinary explicit scene (at rest): its car on raycast wheels, driven.
  await open('solver=explicit&xscene=idle&drive=1&slow=1');
  await clean(true);
  await drive({ forward: 0 }, 1.0);
  const a = await player();
  await drive({ forward: 1 }, 2.0);
  const b = await player();
  for (let i = 0; i < 30; i++) await lab('L.draw()');
  const png = await shot(`${CAR}_wall_driven`);
  const ok = a.contacts === a.wheels && b.speed > 3;
  return { ok, msg: `${a.contacts}/${a.wheels} wheels down; throttle 2 s: ${(b.speed * 3.6).toFixed(1)} km/h → ${png}` };
};

RUNS.traffic = async () => {
  // The traffic derby: the cars drive themselves (ExplicitSim.autopilot); the plain
  // view (no lab). Live (real time) for 8 s.
  await send('Page.navigate', { url: `${base}/softbody.html?solver=explicit&xscene=traffic&cars=6&${CARS[CAR]}` });
  await ready('explicit');
  const read = () => evaluate('window.__explicit.readDrive().then((d) => d.vehicles.map((v) => ({ speed: v.speed, c: v.centre, down: v.wheels.filter((w) => w.contact).length })))');
  const a = await read();
  await evaluate('window.__live(8000)');
  const b = await read();
  const moved = b.map((v, i) => Math.hypot(v.c[0] - a[i].c[0], v.c[2] - a[i].c[2]));
  const png = await shot(`${CAR}_traffic`);
  const driving = moved.filter((m) => m > 5).length;
  const ok = b.length === 6 && driving >= 4;
  return { ok, msg: `${b.length} cars; moved ${moved.map((m) => m.toFixed(0)).join(', ')} m in 8 s; speeds ${b.map((v) => (v.speed * 3.6).toFixed(0)).join(', ')} km/h → ${png}` };
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
