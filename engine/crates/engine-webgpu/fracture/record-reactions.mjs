#!/usr/bin/env node
// Record a reference rig and an authored shot (packs/, destruction-reactions.md) from the
// Fracture Lab's stress-solver mode to a PNG sequence and an MP4. Headless Chrome opens
// fracture/?pack=…&shot=…, the page is stepped frame-locked (__FRACTURE_DEMO__.stressAdvance:
// one 60 Hz physics frame per video frame, no wall-clock dependence) and every frame is a
// CDP screenshot; ffmpeg encodes it.
//
//   bash engine/scripts/build-web.sh          # WGSL, the fracture and stress-web wasm
//   node record-reactions.mjs crush-wall:1.4x [reference-building-crush:8x ...]
//        [--out DIR] [--seconds 5] [--bonds threshold|stress] [--orbit dx,dy] [--zoom d]
//        [--size 1280x720] [--png]   (keep the PNG frames next to the MP4)
//
// Output defaults to $TMPDIR/rubble-reactions (large media is never committed). Needs
// Chrome with WebGPU and JSPI (CHROME env) and ffmpeg (FFMPEG env).
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const flags = {}, takes = [];
for (let a = process.argv.slice(2), i = 0; i < a.length; i++) {
  if (a[i] === '--png') flags.png = true;
  else if (a[i].startsWith('--')) flags[a[i].slice(2)] = a[++i];
  else takes.push(a[i]);
}
if (!takes.length) takes.push('crush-wall:1.4x');
const OUT = flags.out ?? join(tmpdir(), 'rubble-reactions');
const SECONDS = Number(flags.seconds ?? 5);
const BONDS = flags.bonds ?? 'threshold';
const [ORBIT_X, ORBIT_Y] = (flags.orbit ?? '-160,-40').split(',').map(Number);
const ZOOM = Number(flags.zoom ?? 0);
const [W, H] = (flags.size ?? '1280x720').split('x').map(Number);
const KEEP_PNG = !!flags.png;
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
mkdirSync(OUT, { recursive: true });

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm', '.glb': 'model/gltf-binary' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  let path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, 'index.html');
  if (!path.startsWith(ROOT) || !existsSync(path)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'reactions-rec-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
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
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

for (const take of takes) {
  const [pack, shot] = take.split(':');
  const name = `${pack}_${shot ?? 'gravity'}`.replace(/[^\w.-]/g, '_');
  const query = new URLSearchParams({ pack, bonds: BONDS, ...(shot && shot !== 'gravity' ? { shot } : {}) });
  const t0 = performance.now();
  await send('Page.navigate', { url: `${base}/fracture/?${query}` });
  let ready = false;
  for (let i = 0; i < 600 && !ready; i++) {
    await new Promise((r) => setTimeout(r, 200));
    ready = await evaluate('window.__FRACTURE_DEMO__?.stressReady === true').catch(() => false);
  }
  if (!ready) throw new Error(`${take}: the stress mode did not load: ${await evaluate('document.getElementById("status")?.textContent').catch(() => '?')}`);
  // Collapse the panel so the view fills the frame, then frame the camera.
  await evaluate(`document.body.classList.add('collapsed'); window.__FRACTURE_DEMO__.stressOrbit(${ORBIT_X}, ${ORBIT_Y}); ${ZOOM ? `window.__FRACTURE_DEMO__.stressZoom(${ZOOM});` : ''} 0`);
  const frames = KEEP_PNG ? join(OUT, name) : mkdtempSync(join(tmpdir(), `reactions-${name}-`));
  mkdirSync(frames, { recursive: true });
  const count = Math.round(SECONDS * 60);
  let st = null;
  for (let n = 0; n < count; n++) {
    st = await evaluate('window.__FRACTURE_DEMO__.stressAdvance(1000 / 60)');
    const r = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(frames, `f${String(n).padStart(5, '0')}.png`), Buffer.from(r.result.data, 'base64'));
  }
  const status = await evaluate('window.__FRACTURE_DEMO__.status');
  if (/error|failed|unavailable/i.test(status)) throw new Error(`${take}: ${status}`);
  const mp4 = join(OUT, `${name}.mp4`);
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-framerate', '60', '-i', join(frames, 'f%05d.png'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('ffmpeg failed');
  spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-i', join(frames, 'f%05d.png'), '-vf', `select='not(mod(n\\,${Math.max(1, Math.floor(count / 16))}))',scale=480:-1,tile=4x4`, '-frames:v', '1', join(OUT, `${name}_sheet.jpg`)]);
  if (!KEEP_PNG) rmSync(frames, { recursive: true, force: true });
  console.log(`${take}: ${count} frames, ${st.frames} physics frames, ${st.brokenBonds} bonds broken, ${st.bodies ?? st.fragments} bodies, ${st.shots} shots in ${((performance.now() - t0) / 1000).toFixed(0)} s → ${mp4}${KEEP_PNG ? ` (+ ${frames}/)` : ''}`);
}
ws.close();
server.close();
process.exit(0);
