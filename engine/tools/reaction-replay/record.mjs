#!/usr/bin/env node
// Record a reaction replay (index.html playing a reaction_replay JSON) to an MP4 and a
// contact sheet: headless Chrome steps the player frame-locked (window.__REPLAY__.show),
// every frame is a CDP screenshot, ffmpeg encodes them at 60 fps.
//
//   cargo run --release -p engine-world --example reaction_replay -- slab-columns --out /tmp/x.json
//   node engine/tools/reaction-replay/record.mjs /tmp/x.json [--out DIR] [--name NAME]
//        [--size 1280x720] [--eye x,y,z --target x,y,z] [--every 1] [--png]
//
// Large media is never committed: output defaults to $TMPDIR/rubble-reactions. Needs Chrome
// (CHROME env), network for the pinned three.js CDN build, and ffmpeg (FFMPEG env).
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const flags = {}, inputs = [];
for (let a = process.argv.slice(2), i = 0; i < a.length; i++) {
  if (a[i] === '--png') flags.png = true;
  else if (a[i].startsWith('--')) flags[a[i].slice(2)] = a[++i];
  else inputs.push(a[i]);
}
if (!inputs.length) throw new Error('usage: record.mjs <replay.json> [--out DIR]');
const OUT = flags.out ?? join(tmpdir(), 'rubble-reactions');
const [W, H] = (flags.size ?? '1280x720').split('x').map(Number);
const EVERY = Number(flags.every ?? 1);
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
mkdirSync(OUT, { recursive: true });

const files = new Map(inputs.map((p, i) => [`/replay${i}.json`, resolve(p)]));
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = url.pathname === '/' ? join(HERE, 'index.html') : files.get(url.pathname);
  if (!path || !existsSync(path)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': path.endsWith('.html') ? 'text/html' : 'application/json', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'replay-rec-'));
const port = 9800 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check', '--use-angle=metal',
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
const pageErrors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

for (const [route, path] of files) {
  const name = flags.name && files.size === 1 ? flags.name : basename(path, '.json');
  const query = new URLSearchParams({ replay: route, frame: '0', ...(flags.eye ? { eye: flags.eye } : {}), ...(flags.target ? { target: flags.target } : {}) });
  await send('Page.navigate', { url: `${base}/?${query}` });
  let n = 0;
  for (let i = 0; i < 300 && !n; i++) {
    await new Promise((r) => setTimeout(r, 200));
    n = await evaluate('window.__REPLAY__?.frames ?? 0').catch(() => 0);
  }
  if (!n) throw new Error(`${path}: the player did not load: ${pageErrors.join('; ') || 'no page error'}`);
  const frames = flags.png ? join(OUT, name) : mkdtempSync(join(tmpdir(), `replay-${name}-`));
  mkdirSync(frames, { recursive: true });
  let k = 0;
  for (let f = 0; f < n; f += EVERY, k++) {
    await evaluate(`window.__REPLAY__.show(${f})`);
    const r = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(frames, `f${String(k).padStart(5, '0')}.png`), Buffer.from(r.result.data, 'base64'));
  }
  const mp4 = join(OUT, `${name}.mp4`);
  const fps = Math.round(1 / (await evaluate('window.__REPLAY__.dt')) / EVERY);
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-framerate', String(fps), '-i', join(frames, 'f%05d.png'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error('ffmpeg failed');
  spawnSync(FFMPEG, ['-loglevel', 'error', '-y', '-i', join(frames, 'f%05d.png'), '-vf', `select='not(mod(n\\,${Math.max(1, Math.floor(k / 16))}))',scale=480:-1,tile=4x4`, '-frames:v', '1', join(OUT, `${name}_sheet.jpg`)]);
  if (!flags.png) rmSync(frames, { recursive: true, force: true });
  console.log(`${name}: ${k} frames → ${mp4} (+ ${name}_sheet.jpg)`);
}
ws.close();
server.close();
process.exit(0);
