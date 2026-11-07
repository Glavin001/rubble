// Records the crash-test lab (softbody.html ?solver=explicit with the lab on) as a video:
// each preset runs frame-locked (the page's own physics and renderer, stepped from here),
// real time up to just before the impact, exactly 20x slow motion through it, then real
// time again; the wall crash is shown a second time in the beam-strain view. Frames are
// captured from headless Chrome and encoded with ffmpeg (60 fps, H.264).
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket crash-lab-record.mjs [--car showcase] [--size 1920x1080] [--out file.mp4] [preset ...]
//
// Presets: wall56, pole29, tbone80, drop15 (by default), headon50, tbone214,
// overlap64, swipe50, mdb214, doorpull; name:strain for the beam-strain view. Needs Chrome and ffmpeg.
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
const OUT_FILE = opt('--out', join(ROOT, 'recordings', 'crash-lab', 'crash-lab.mp4'));
const OUT = dirname(OUT_FILE);
const CAR = opt('--car', 'showcase');
// --size WxH: the window (screenshots) size; 1280x720 by default.
const [W, H] = opt('--size', '1920x1080').split('x').map(Number);
const only = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
const CARS = { sedan: 'car=sedan', showcase: 'car=showcase', showcase1: 'car=showcase&style=sedan1',
  // Community BeamNG cars from a local conversion cache (community_cache: one LOD,
  // textures at 1K; engine-webgpu/__mods, never committed), on raycast wheels.
  etkk: 'mod=/__mods/etkk_tex&pc=etkk/kr6.pc&wheels=raycast', ccf: 'mod=/__mods/ccf_tex&pc=ccf/f4_sport_M.pc&wheels=raycast',
  vertex: 'mod=/__mods/vertex_tex&pc=gavril_vertex/gavril_vertex.pc&wheels=raycast', hires: 'car=hires', e36: 'truck=/__local/E36Sedan.truck', viper: 'truck=/__local/ViperGTS.car',
  mustang: 'truck=/__local/1982-Ford-Mustang-GT-351W.truck', agora: 'truck=/__local/95bbUID-agoras.truck',
  // Rubble's generated sedan (gen2_export writes it to __mods/gen2), on raycast wheels as the lab's ETK.
  gen2: 'mod=/__mods/gen2&pc=gen2/gen2.pc&wheels=raycast' };
// The comparison manifest's cars (vehicle-compare-cars.json, shared with the
// CPU harness `vehicle_compare`): a JBeam car under __mods/ in JBeam mode, the
// physics the harness runs it on.
for (const c of JSON.parse(readFileSync(join(ROOT, 'vehicle-compare-cars.json'), 'utf8')).cars) {
  CARS[c.name] ??= c.page ?? `mod=/__mods/${c.mod}&pc=${c.vehicle ?? c.mod}/${c.pc}&jbeamMode=native`;
}
if (CARS[CAR]?.startsWith("mod=") && !existsSync(join(ROOT, CARS[CAR].split('&')[0].slice(5)))) { console.log(`- ${CAR}: not in __mods/, skipped`); process.exit(0); }
// --label TEXT: a caption drawn over every frame (the side-by-side recorder's car names).
const LABEL = opt('--label', '');
if (!CARS[CAR]) throw new Error(`--car: one of ${Object.keys(CARS).join(', ')}`);
const carFile = CARS[CAR].startsWith('truck=') ? join(LOCAL, CARS[CAR].split('/__local/')[1])
  : CARS[CAR].startsWith('mod=/__mods/') ? join(ROOT, '__mods', CARS[CAR].slice('mod=/__mods/'.length).split('&')[0]) : null;
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

// CRASH_LAB_PROFILE: a kept Chrome profile, so its shader cache serves later runs.
const keepProfile = process.env.CRASH_LAB_PROFILE;
if (keepProfile) mkdirSync(keepProfile, { recursive: true });
const profile = keepProfile || mkdtempSync(join(tmpdir(), 'crash-lab-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} if (!keepProfile) { try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} } };
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
const frameDir = mkdtempSync(join(tmpdir(), 'crash-lab-frames-'));
let frameNo = 0;
const grab = async () => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality: 92 });
  writeFileSync(join(frameDir, `f${String(frameNo++).padStart(6, '0')}.jpg`), Buffer.from(s.result.data, 'base64'));
};
const ready = async () => {
  // (Up to 15 min: Chrome takes about 8 min to compile the frame module since the surface pass merge; CRASH_LAB_PROFILE keeps its cache.)
  for (let i = 0; i < 4500; i++) {
    await new Promise((r) => setTimeout(r, 200));
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}`);
    if (await evaluate('window.__ready === true && !!window.__lab').catch(() => false)) {
      if (process.env.CRASH_LAB_PROGRESS) console.log(`  ready after ${(i / 5).toFixed(0)} s`);
      return;
    }
    // CRASH_LAB_PROGRESS: the page's status line every 30 s while it loads.
    if (process.env.CRASH_LAB_PROGRESS && i % 150 === 149) {
      const st = await evaluate('document.getElementById("status")?.textContent || ""').catch((e) => `(${e.message})`);
      console.log(`  loading ${((i + 1) / 5).toFixed(0)} s: ${st.slice(0, 160)}`);
    }
  }
  throw new Error('the page did not start');
};
// --query a=b&c=d: appended to every page's query (bind=legacy, alt=0: a
// visual A/B on identical physics).
const EXTRA = opt('--query', '');
const open = async (query) => { await send('Page.navigate', { url: `${base}/softbody.html?solver=explicit&${query}&${CARS[CAR]}${EXTRA ? '&' + EXTRA : ''}` }); await ready(); };
const lab = (expr) => evaluate(`(async () => { const L = window.__lab; return ${expr}; })()`);
const HZ = 2000, FPS = 60;
// The lab's readout follows its own slow-motion setting; keep it truthful.
// (Pages open paused so nothing runs before the recording; the readout should not say so.)
const setSlow = (k) => lab(`(L.lab.st.auto = false, L.lab.st.paused = false, L.lab.st.slow = ${k}, true)`);
// Advance `seconds` of simulation at `slow`x, one captured frame per 1/60 s of video.
let acc = 0;
const play = async (seconds, slow) => {
  await setSlow(slow);
  const frames = Math.round(seconds * slow * FPS);
  for (let f = 0; f < frames; f++) {
    acc += HZ / FPS / slow; const k = Math.floor(acc); acc -= k;
    if (k > 0) await lab(`L.advance(${k}, ${k})`); else await lab('L.draw()');
    await grab();
  }
};
const hold = async (seconds) => { for (let f = 0; f < seconds * FPS; f++) { await lab('L.draw()'); await grab(); } };
const PRESETS = {
  // zoom: the camera's distance against the lab preset's (closer, so the
  // damage reads).
  wall56: { cam: 'three', zoom: 0.55, yaw: 1.2 }, pole29: { cam: 'threeFront', zoom: 0.6 }, headon50: { cam: 'side', zoom: 0.8 }, drop15: { cam: 'side', zoom: 0.7 },
  tbone214: { cam: 'three', zoom: 0.7 }, tbone80: { cam: 'three', zoom: 0.4 }, overlap64: { cam: 'threeFront', zoom: 0.7 }, swipe50: { cam: 'threeFront', zoom: 0.7 },
  mdb214: { cam: 'three' }, doorpull: { cam: 'threeFront', slow: 2.0 },
};
const runs = only.length ? only : ['wall56', 'pole29', 'tbone80', 'drop15', 'wall56:strain'];
for (const run of runs) {
  const [name, view] = run.split(':');
  const p = PRESETS[name];
  if (!p) throw new Error(`unknown preset ${name}`);
  // Pass 1: the impact step.
  await open(`preset=${name}&paused=1`);
  let s = await lab('L.state()');
  while (s.impactStep == null && s.step < 6 * HZ) s = await lab('L.advance(200, 20)');
  if (s.impactStep == null) throw new Error(`${name}: no impact`);
  // --impact MS: the impact time given (the side-by-side recorder aligns every car to the first one's; a car whose detection fires early, at spawn, is placed the same).
  const impact = opt('--impact', null) != null ? Math.round(Number(opt('--impact')) * HZ / 1000) : s.impactStep;
  // Pass 2: the recording.
  // The strain pass: the beam skeleton and nodes coloured by strain over a ghost body.
  await open(`preset=${name}&paused=1${view === 'strain' ? '&skin3=ghost&skeleton=1&nodes=1&color=strain' : ''}`);
  await lab(`(L.lab.camera('${p.cam}', ${p.zoom ?? 1}, ${p.yaw ?? null}), true)`);
  await lab('(L.lab.setClean(true), true)');
  if (LABEL) await evaluate(`(() => { const d = document.createElement('div'); d.textContent = ${JSON.stringify(LABEL)}; d.style.cssText = 'position:fixed;left:24px;bottom:24px;z-index:99999;font:600 28px system-ui,sans-serif;color:#fff;background:rgba(0,0,0,.55);padding:6px 14px;border-radius:6px;pointer-events:none'; document.body.appendChild(d); return true; })()`);
  const pre = impact / HZ;
  const lead = Math.min(1.0, pre - 0.03);
  if (pre - 0.03 - lead > 0) await lab(`L.advance(${Math.round((pre - 0.03 - lead) * HZ)}, 40)`);
  for (let i = 0; i < 30; i++) await lab('L.draw()'); // the camera settles
  acc = 0;
  await play(lead, 1);          // real time up to 30 ms before the impact
  if (p.slow) {
    await play(p.slow, 1);      // a slow event (the door pull): real time throughout
  } else {
    await play(0.33, 20);       // 20x slow motion through the crash (6.6 s of video)
    await play(1.2, 1);         // real time after
  }
  await hold(0.8);
  console.log(`${run}: impact at ${(impact / HZ * 1000).toFixed(0)} ms, ${frameNo} frames so far`);
}
const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
  '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT_FILE], { stdio: 'inherit' });
rmSync(frameDir, { recursive: true, force: true });
if (enc.status !== 0) throw new Error('ffmpeg failed');
console.log(`→ ${OUT_FILE} (${(frameNo / FPS).toFixed(1)} s)`);
process.exit(0);
