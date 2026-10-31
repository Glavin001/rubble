// Record Broadside (broadside.html) as a film: serves this directory, launches headless
// Chrome (throwaway profile, WebGPU on), and renders the take frame by frame through the
// page's window.__film, each frame 1/FPS of clock time on (the take's slow motion
// included) however long the physics takes, then encodes recordings/broadside.mp4 and a
// contact sheet with ffmpeg.
//
//   node record-broadside.mjs [--out DIR] [--fps 30] [--max-frames N] [--level 0|1|2]
//
// Needs the WGSL export and both wasm modules built (scripts/build-web.sh, or its
// wgsl_export, fracture-wasm and world-wasm steps). CHROME / FFMPEG override the binaries.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
const OUT = opt('--out', join(ROOT, 'recordings'));
const FPS = Number(opt('--fps', 30));
const MAX_FRAMES = Number(opt('--max-frames', 3600));
const LEVEL = opt('--level', '2');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const FRAMES = mkdtempSync(join(tmpdir(), 'broadside-frames-'));
mkdirSync(OUT, { recursive: true });

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm', '.glb': 'model/gltf-binary' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'broadside-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} rmSync(profile, { recursive: true, force: true }); rmSync(FRAMES, { recursive: true, force: true }); };
process.on('exit', cleanup);

let targets = [];
for (let i = 0; i < 600 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map();
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('[page]', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('[page]', m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, timeout: 3_600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: `${base}/broadside.html?film=1&level=${LEVEL}` });
let ready = false;
for (let i = 0; i < 600 && !ready; i++) {
  await new Promise((r) => setTimeout(r, 200));
  ready = await evaluate('!!window.__filmReady');
  const err = await evaluate('document.getElementById("status")?.classList.contains("error") ? document.getElementById("status").textContent : ""');
  if (err) throw new Error(`broadside: ${err}`);
}
if (!ready) throw new Error('broadside: the page did not get ready');
await evaluate('window.__film.start()', true);

const t0 = performance.now();
let n = 0;
for (; n < MAX_FRAMES; n++) {
  const r = await evaluate(`window.__film.frame(${FPS})`, true);
  writeFileSync(join(FRAMES, `f${String(n).padStart(5, '0')}.png`), Buffer.from(r.png.split(',')[1], 'base64'));
  if (n % FPS === 0) console.log(`frame ${n}: t ${r.stats.simSeconds.toFixed(2)} s, ${r.stats.shots} shots, ${r.stats.brokenBonds} bonds broken (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
  if (r.done) { n++; break; }
}
const ffmpeg = (a) => {
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', ...a], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${a.join(' ')}`);
};
const frames = join(FRAMES, 'f%05d.png');
ffmpeg(['-framerate', String(FPS), '-i', frames, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', join(OUT, `broadside-l${LEVEL}.mp4`)]);
ffmpeg(['-i', frames, '-vf', `select='not(mod(n\\,${FPS * 3}))',scale=480:-1,tile=4x4`, '-frames:v', '1', join(OUT, `broadside-l${LEVEL}_sheet.jpg`)]);
console.log(`broadside: ${n} frames (${(n / FPS).toFixed(1)} s of film) rendered in ${((performance.now() - t0) / 1000).toFixed(0)} s → ${join(OUT, `broadside-l${LEVEL}.mp4`)}`);
ws.close();
server.close();
process.exit(0);
