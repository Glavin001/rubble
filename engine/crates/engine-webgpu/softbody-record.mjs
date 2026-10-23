#!/usr/bin/env node
// Records the soft-body crash tests to MP4: headless Chrome steps softbody.html
// frame-locked through its __step hook (no wall-clock dependence), a scripted orbit
// camera follows each crash, and every frame is a CDP screenshot encoded by ffmpeg.
//
//   cargo run -p engine-webgpu --release --bin wgsl_export   # WGSL + wasm first
//   node --experimental-websocket softbody-record.mjs [take ...] [--out DIR]
//   RUBBLE_ROR_DIR=… RUBBLE_JBEAM_DIR=… node --experimental-websocket softbody-record.mjs bus_wall semi_car legocar_wall
//
// Each take: the crash in real time (4 steps of 240 Hz per 60 fps frame), the impact
// again in 4× slow motion (1 step per frame), an orbit around the wreck, and the beam
// view coloured by plastic strain. Needs Chrome and ffmpeg (CHROME / FFMPEG env).
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(ROOT, 'recordings', 'softbody');
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

// Camera target per take (crash point), framing and impact timing (steps at 240 Hz).
// Run-up: metres of approach before impact; impact window in steps at 240 Hz.
const TAKES = {
  headon: { query: 'scene=headon&runup=8', title: 'Head-on: two node–beam cars at 29 km/h each', target: [0.15, 0.7, 0], dist: 10, yaw: 1.25, impact: [100, 330], settle: 560 },
  wall: { sweep: 1.6, query: 'scene=wall&runup=8', title: 'Rigid wall at 54 km/h', target: [0, 0.8, -1], dist: 10, yaw: 2.2, impact: [120, 340], settle: 560 },
  tbone: { query: 'scene=tbone&runup=4', title: 'Side impact: 1,500 kg block at 40 km/h', target: [1, 0.8, 0], dist: 9.5, yaw: 0.5, impact: [120, 360], settle: 560 },
  pole: { sweep: 2.4, query: 'scene=pole&runup=6', title: '20 cm pole at 40 km/h', target: [0, 0.8, 0], dist: 9, yaw: 2.2, impact: [120, 360], settle: 560 },
  // Harder tests (scene parameters; target null = follow the car).
  wall100: { sweep: 1.6, query: 'scene=wall&speed=100&runup=8', title: 'Rigid wall at 100 km/h', target: [0, 0.8, 0], dist: 10, yaw: 2.2, impact: [50, 300], settle: 560 },
  overlap: { sweep: 1.6, query: 'scene=wall&speed=64&overlap=0.25&runup=8', title: 'Small overlap (25% of the width) at 64 km/h', target: [0, 0.8, 0], dist: 10, yaw: 2.4, impact: [90, 340], settle: 600 },
  tbone_front: { query: 'scene=tbone&speed=60&at=1.6&runup=4', title: 'Side impact at the front wheel: 1,500 kg at 60 km/h', target: [0.5, 0.8, 1], dist: 9.5, yaw: 0.5, impact: [110, 350], settle: 600 },
  tbone80: { query: 'scene=tbone&speed=80&mass=2000&runup=4', title: 'Side impact: 2,000 kg at 80 km/h', target: null, dist: 10, yaw: 0.5, impact: [100, 340], settle: 640 },
  sidepole: { query: 'scene=sidepole&speed=60&runup=4', title: 'Side pole at the door, 60 km/h', target: [-1.5, 0.8, 0], dist: 9, yaw: 0.4, impact: [40, 280], settle: 560 },
  headon_offset: { query: 'scene=headon&speed=80&overlap=0.5&runup=8', title: 'Head-on, 2 × 80 km/h, 50% offset', target: [0.4, 0.8, 0], dist: 12, yaw: 1.3, impact: [20, 260], settle: 560 },
  rollover: { query: 'scene=rollover&speed=70&runup=4', title: 'Sliding sideways into a curb at 70 km/h', target: null, dist: 11, yaw: 0.3, impact: [40, 400], settle: 840 },
  ramp: { query: 'scene=ramp&speed=90', title: 'One-side ramp at 90 km/h', target: null, dist: 13, yaw: 1.9, impact: [40, 520], settle: 960 },
  limbo: { query: 'scene=limbo&speed=80&runup=6', title: 'Limbo bar at 80 km/h: the roof shears off', target: [0, 1, 4], dist: 11, yaw: 2.0, sweep: 1.4, impact: [60, 330], settle: 600 },
  sandwich: { query: 'scene=sandwich&speed=60&runup=6', title: 'Sandwich: 18 t truck at 60 km/h into a car behind a parked 15 t truck', target: [0, 1, 0], dist: 18, yaw: 1.3, impact: [60, 360], settle: 640 },
  crusher: { query: 'scene=crusher', title: 'Crusher: 10 t block dropped 4 m onto the roof', target: [0, 1, 0], dist: 10, yaw: 1.0, impact: [180, 400], settle: 640 },
  // Camera views on subjects (softbody.html cam modes): crash phases ride the view.
  sandwich_cab: { query: 'scene=sandwich&speed=60&runup=6', cam: { mode: 'mount', from: '18 t truck', target: 'car0' }, title: 'Sandwich, from the 18 t truck\'s cab', target: [0, 1, 0], dist: 18, yaw: 1.3, impact: [60, 360], settle: 640 },
  sandwich_side: { query: 'scene=sandwich&speed=60&runup=6', cam: { mode: 'right', target: 'car0' }, title: 'Sandwich, the car\'s starboard side', target: [0, 1, 0], dist: 18, yaw: 1.3, impact: [60, 360], settle: 640 },
  crusher_side: { query: 'scene=crusher', cam: { mode: 'right', target: 'car0' }, title: 'Crusher: drop, then the winch lifts it to show the wreck', target: [0, 1, 0], dist: 10, yaw: 1.0, impact: [180, 420], settle: 1800 },
  limbo_driver: { query: 'scene=limbo&speed=80&runup=6', cam: { mode: 'driver', target: 'car0' }, title: 'Limbo bar from the driver\'s seat', target: [0, 1, 4], dist: 11, yaw: 2.0, sweep: 1.4, impact: [60, 330], settle: 600 },
  traffic20: { query: 'scene=traffic&cars=20', title: 'Traffic derby: 20 self-driving soft cars', target: [0, 0, 0], dist: 70, yaw: 0.6, impact: [0, 0], settle: 2400, live: true },
  drop: { query: 'scene=drop&height=12&pitch=50', title: 'Nose-first drop from 12 m', target: [0, 1.2, 1], dist: 11, yaw: 1.6, impact: [330, 570], settle: 760 },
};
// Imported vehicles (local content, never committed): RUBBLE_ROR_DIR = Rigs of Rods
// content-master/, RUBBLE_JBEAM_DIR = a vehicles/ dir with agenty_legocar/ and common/.
// The import scene's target is at z = 12; the vehicle starts `gap` m short of it.
const IMPORTED = {
  bus_wall: { ror: 'agora/95bbUID-agoras.truck', target: 'wall', speed: 11, gap: 8, title: 'Rigs of Rods Agora bus (imported .truck) into a wall at 40 km/h', look: [0, 1.5, 6], dist: 22, yaw: 2.5, sweep: 1.6, impact: [150, 420] },
  semi_car: { ror: 'dafsemi/b6b0UID-semi.truck', target: 'car', speed: 11, gap: 8, title: 'Rigs of Rods DAF semi (imported .truck) into a parked car at 40 km/h', look: null, dist: 15, yaw: 1.0, impact: [150, 420] },
  legocar_wall_raw: { jbeam: 'agenty_legocar', query: 'threshold=1', target: 'wall', speed: 14, gap: 8, title: 'BeamNG legocar (imported JBeam, file thresholds) into a wall at 50 km/h', look: [0, 1, 9], dist: 12, yaw: 2.3, sweep: 1.6, impact: [110, 380] },
  legocar_wall: { jbeam: 'agenty_legocar', target: 'wall', speed: 14, gap: 8, title: 'BeamNG legocar (imported JBeam) into a wall at 50 km/h', look: [0, 1, 9], dist: 12, yaw: 2.3, sweep: 1.6, impact: [110, 380] },
};
for (const [k, v] of Object.entries(IMPORTED)) {
  TAKES[k] = { query: `scene=import&target=${v.target}${v.query ? '&' + v.query : ''}`, title: v.title, target: v.look, dist: v.dist, yaw: v.yaw, sweep: v.sweep, impact: v.impact, settle: v.impact[1] + 240, imported: v };
}
const walk = (dir, out = []) => { for (const f of readdirSync(dir)) { const p = join(dir, f); if (statSync(p).isDirectory()) walk(p, out); else out.push(p); } return out; };
const importExpr = (v) => {
  if (v.ror) {
    const dir = process.env.RUBBLE_ROR_DIR;
    if (!dir) throw new Error('set RUBBLE_ROR_DIR to the Rigs of Rods content-master/ dir');
    return `m.loadTruck(${JSON.stringify(readFileSync(join(dir, v.ror), 'utf8'))})`;
  }
  const dir = process.env.RUBBLE_JBEAM_DIR;
  if (!dir) throw new Error('set RUBBLE_JBEAM_DIR to a vehicles/ dir with agenty_legocar/ and common/');
  const files = [...walk(join(dir, v.jbeam)), ...walk(join(dir, 'common'))].filter((f) => f.endsWith('.jbeam'))
    .map((f) => [f.slice(dir.length + 1), readFileSync(f, 'utf8')]);
  const pc = readFileSync(join(dir, v.jbeam, 'start.pc'), 'utf8');
  return `m.loadJbeam(${JSON.stringify(files)}, ${JSON.stringify(pc)})`;
};
const names = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
const DEFAULT_TAKES = ['headon', 'wall', 'tbone', 'pole'];
const takes = names.length ? names : DEFAULT_TAKES;
mkdirSync(OUT, { recursive: true });

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'softbody-rec-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'], { stdio: 'ignore' });
process.on('exit', () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} });
let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
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
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });

for (const name of takes) {
  const t = TAKES[name];
  if (!t) throw new Error(`unknown take ${name}; have ${Object.keys(TAKES).join(', ')}`);
  const frames = mkdtempSync(join(tmpdir(), `softbody-${name}-`));
  let n = 0;
  const shot = async () => {
    const r = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
    writeFileSync(join(frames, `f${String(n++).padStart(5, '0')}.jpg`), Buffer.from(r.result.data, 'base64'));
  };
  // One video frame: steps, camera, caption, screenshot.
  const frame = async (steps, view, caption) => {
    await evaluate(`window.__setView(${JSON.stringify(view)}); window.__setCaption(${JSON.stringify(caption)})`);
    await evaluate(`window.__step(${steps}).then(() => 0)`);
    await shot();
  };
  let info = null;
  const reload = async () => {
    await send('Page.navigate', { url: `${base}/softbody.html?${t.query}` });
    for (let i = 0; i < 300; i++) {
      await new Promise((r) => setTimeout(r, 200));
      if (await evaluate('window.__ready === true').catch(() => false)) {
        if (t.imported) {
          const v = t.imported;
          info = await evaluate(`(async () => { const m = await import('./soft.mjs'); return window.__crashImported(${importExpr(v)}, ${JSON.stringify({ speed: v.speed, gap: v.gap, label: name })}); })()`);
        }
        return;
      }
    }
    const status = await evaluate(`document.getElementById('status')?.textContent + ' ' + (document.getElementById('err')?.textContent || '')`).catch(() => '');
    throw new Error(`${name}: demo did not start: ${status}`);
  };
  const orbit = (yaw, dist = t.dist, pitch = 0.32) => t.cam && !wreck ? { skin: 'flex', beams: false, cam: t.cam } : { skin: 'flex', beams: false, cam: { mode: 'orbit' }, orbit: { yaw, pitch, dist, target: t.target } };
  let wreck = false;
  const t0 = performance.now();
  // 1. Real time.
  await reload();
  let yaw = t.yaw;
  for (let s = 0; s < t.settle; s += 4) await frame(4, orbit((yaw += 0.004)), `${t.title} — real time`);
  // 2. The impact again, 4× slow motion, closer.
  await reload();
  yaw = t.yaw + 0.3;
  await evaluate(`window.__step(${t.impact[0]}).then(() => 0)`);
  for (let s = t.impact[0]; s < t.impact[1]; s += 1) await frame(1, orbit((yaw += 0.003), t.dist * 0.7, 0.25), `${t.title} — 4× slow motion`);
  // 3. Orbit the wreck, then the beam view (plastic strain).
  wreck = true;
  await evaluate(`window.__step(${Math.max(0, t.settle - t.impact[1])}).then(() => 0)`);
  // Full turn, or a swing across the open side where a wall would block the view.
  const sweep = t.sweep ?? Math.PI * 2;
  const swing = (i, n) => t.yaw + (t.sweep ? sweep * (Math.sin((i / n) * Math.PI * 2 - Math.PI / 2) * 0.5) : 0.3 + (i / n) * sweep);
  for (let i = 0; i < 150; i++) await frame(0, orbit(swing(i, 150), t.dist * 0.75, 0.35), 'The wreck: permanent (plastic) deformation');
  for (let i = 0; i < 120; i++) await frame(0, { ...orbit(swing(i, 120), t.dist * 0.75, 0.3), beams: true }, 'Beams: yellow → red = plastic strain, green = intact');
  const mp4 = join(OUT, `crash_${name}.mp4`);
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-framerate', '60', '-i', join(frames, 'f%05d.jpg'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('ffmpeg failed');
  spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-i', join(frames, 'f%05d.jpg'), '-vf', "select='not(mod(n\\,45))',scale=400:-1,tile=4x4", '-frames:v', '1', join(OUT, `crash_${name}_sheet.jpg`)]);
  rmSync(frames, { recursive: true, force: true });
  if (info) console.log(`${name}: ${info.nodes} nodes, ${info.beams} beams, ${info.mass.toFixed(0)} kg, ${info.length.toFixed(1)} m long`);
  console.log(`${name}: ${n} frames (${(n / 60).toFixed(1)} s) in ${((performance.now() - t0) / 1000).toFixed(0)} s → ${mp4}`);
}
ws.close();
server.close();
process.exit(0);
