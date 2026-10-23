#!/usr/bin/env node
// Records the soft-body crash tests to MP4: headless Chrome steps softbody.html
// frame-locked through its __step hook (no wall-clock dependence), a scripted orbit
// camera follows each crash, and every frame is a CDP screenshot encoded by ffmpeg.
//
//   cargo run -p engine-webgpu --release --bin wgsl_export   # WGSL + wasm first
//   node --experimental-websocket softbody-record.mjs [take ...] [--out DIR]
//
// Each take: the crash in real time (4 steps of 240 Hz per 60 fps frame), the impact
// again in 4× slow motion (1 step per frame), an orbit around the wreck, and the beam
// view coloured by plastic strain. Needs Chrome and ffmpeg (CHROME / FFMPEG env).
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
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
};
const names = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
const takes = names.length ? names : Object.keys(TAKES);
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
  const reload = async () => {
    await send('Page.navigate', { url: `${base}/softbody.html?${t.query}` });
    for (let i = 0; i < 300; i++) {
      await new Promise((r) => setTimeout(r, 200));
      if (await evaluate('window.__ready === true').catch(() => false)) return;
    }
    throw new Error(`${name}: demo did not start`);
  };
  const orbit = (yaw, dist = t.dist, pitch = 0.32) => ({ skin: 'flex', beams: false, orbit: { yaw, pitch, dist, target: t.target } });
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
  console.log(`${name}: ${n} frames (${(n / 60).toFixed(1)} s) in ${((performance.now() - t0) / 1000).toFixed(0)} s → ${mp4}`);
}
ws.close();
server.close();
process.exit(0);
