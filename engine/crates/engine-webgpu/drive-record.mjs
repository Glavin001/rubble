// Records driving on the explicit solver (softbody.html ?solver=explicit&xscene=yard:
// raycast wheels on the soft-body car) as a video: the car pulls away, weaves left and
// right, then drives into the side of a parked car at speed; real time while driving,
// 10x slow motion through the crash, then real time again. The driver is scripted
// here (steering toward waypoints from the page's own drive read-back) and its input
// goes through the page's input path (window.__explicit.drive). Frames are captured
// from headless Chrome and encoded with ffmpeg (60 fps, H.264).
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket drive-record.mjs [--car sedan|showcase|showcase1|hires] [--size 1280x720] [--out file.mp4]
//
// Needs Chrome and ffmpeg.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const LOCAL = normalize(join(ROOT, '../../local-vehicles'));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT_FILE = opt('--out', join(ROOT, 'recordings', 'drive', 'drive.mp4'));
const OUT = dirname(OUT_FILE);
const CAR = opt('--car', 'showcase');
// --size WxH: the window (screenshots) size; 1280x720 by default.
const [W, H] = opt('--size', '1920x1080').split('x').map(Number);
const only = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
const CARS = { sedan: 'car=sedan', showcase: 'car=showcase', showcase1: 'car=showcase&style=sedan1', hires: 'car=hires', e36: 'truck=/__local/E36Sedan.truck', viper: 'truck=/__local/ViperGTS.car',
  mustang: 'truck=/__local/1982-Ford-Mustang-GT-351W.truck', agora: 'truck=/__local/95bbUID-agoras.truck' };
if (!CARS[CAR]) throw new Error(`--car: one of ${Object.keys(CARS).join(', ')}`);
const carFile = CARS[CAR].startsWith('truck=') ? join(LOCAL, CARS[CAR].split('/__local/')[1]) : null;
if (carFile && !existsSync(carFile)) { console.log(`- ${CAR}: ${carFile} not found, skipped`); process.exit(0); }

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const serveFrom = (base, rel) => {
  const path = normalize(join(base, rel));
  return path.startsWith(base) && existsSync(path) && !statSync(path).isDirectory() ? path : null;
};
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = p.startsWith('/__local/') ? serveFrom(LOCAL, p.slice(9)) : serveFrom(ROOT, p);
  if (!path) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'drive-record-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
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
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });


import { spawnSync } from 'node:child_process';
const frameDir = mkdtempSync(join(tmpdir(), 'drive-record-frames-'));
let frameNo = 0;
const grab = async () => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
  writeFileSync(join(frameDir, `f${String(frameNo++).padStart(6, '0')}.jpg`), Buffer.from(s.result.data, 'base64'));
};
// Up to 5 min: the first load compiles the solver's kernels (explicitSimAsync),
// past a minute on a loaded machine.
const ready = async () => {
  for (let i = 0; i < 1500; i++) {
    await new Promise((r) => setTimeout(r, 200));
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}`);
    if (await evaluate('window.__ready === true && !!window.__lab').catch(() => false)) return;
  }
  throw new Error('the page did not start');
};
const open = async (query) => { await send('Page.navigate', { url: `${base}/softbody.html?solver=explicit&lab=1&${query}&${CARS[CAR]}` }); await ready(); };
const lab = (expr) => evaluate(`(async () => { const L = window.__lab; return ${expr}; })()`);
const HZ = 2000, FPS = 60;
const setSlow = (k) => lab(`(L.lab.st.auto = false, L.lab.st.paused = false, L.lab.st.slow = ${k}, true)`);
const X = (expr) => evaluate(`(async () => { const L = window.__lab; const X = window.__explicit; return ${expr}; })()`);
const me = () => X(`X.readDrive().then((d) => { const v = d.vehicles[0]; return { speed: v.speed, fwd: v.forward, c: v.centre }; })`);
// The scripted driver: steer toward (tx, tz) at `throttle`.
const steerTo = (p, tx, tz) => {
  const want = Math.atan2(tx - p.c[0], tz - p.c[2]), have = Math.atan2(p.fwd[0], p.fwd[2]);
  const err = Math.atan2(Math.sin(want - have), Math.cos(want - have));
  return Math.max(-1, Math.min(1, -2.5 * err)); // + steer is right; + heading error is left
};
let acc = 0, simT = 0;
// `seconds` of simulation at `slow`x, one captured frame per 1/60 s of video; the
// driver chooses each frame's input from the last read-back.
const play = async (seconds, slow, driver) => {
  await setSlow(slow);
  const frames = Math.round(seconds * slow * FPS);
  for (let f = 0; f < frames; f++) {
    if (driver) { const p = await me(); await X(`(X.drive(${JSON.stringify(driver(p, simT))}), true)`); }
    acc += HZ / FPS / slow; const k = Math.floor(acc); acc -= k;
    simT += k / HZ;
    if (k > 0) await lab(`L.advance(${k}, ${k})`); else await lab('L.draw()');
    await grab();
  }
};
await open(`xscene=yard&paused=1`);
await lab('(L.lab.setClean(true), true)');
await X('(X.drive({ forward: 0 }), true)');
await lab('L.advance(2000, 40)');  // settle on the wheels (not recorded)
for (let i = 0; i < 30; i++) await lab('L.draw()');
// Pull away weaving between waypoints, then into the side of the car parked
// across x = 14, z = 20.
const route = [[-3, 6], [2, 12], [14, 20]];
let leg = 0;
const driver = (p) => {
  while (leg < route.length - 1 && p.c[2] > route[leg][1] - 2) leg++;
  return { forward: 1, steer: steerTo(p, route[leg][0], route[leg][1]) };
};
let impact = null;
await setSlow(1);
for (let n = 0; n < 60 * 12 && impact == null; n++) {
  await play(1 / 60, 1, driver);
  const s = await lab('L.state()');
  if (s.impactStep != null) impact = s;
}
if (impact == null) throw new Error('no impact');
console.log(`impact at ${(simT).toFixed(2)} s`);
// The three-quarter view for the crash (both cars in frame).
await lab('(L.lab.camera(\'three\', 0.9), true)');
await play(0.4, 10, () => ({ forward: 0 }));   // 10x slow motion through the crash
await play(1.5, 1, () => ({ forward: 0 }));    // real time after
const hold = async (seconds) => { for (let f = 0; f < seconds * FPS; f++) { await lab('L.draw()'); await grab(); } };
await hold(0.8);
const after = await lab('L.state()');
console.log(`after: ${after.bent} bent, ${after.broken} broken`);
const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
  '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT_FILE], { stdio: 'inherit' });
rmSync(frameDir, { recursive: true, force: true });
if (enc.status !== 0) throw new Error('ffmpeg failed');
console.log(`→ ${OUT_FILE} (${(frameNo / FPS).toFixed(1)} s)`);
process.exit(0);
