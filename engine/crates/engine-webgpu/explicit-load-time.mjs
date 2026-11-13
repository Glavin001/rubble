#!/usr/bin/env node
// The explicit solver page's load time in Chrome: softbody.html?solver=explicit (the
// yard scene by default) from navigation to its first frame (the loading card's
// "[loading] done in"), with the card's stages and each GPU kernel's compile as the
// card lists them. Run twice with the same PROFILE for a warm (cached) load.
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node --experimental-websocket explicit-load-time.mjs [query]
//
// PROFILE=<dir> keeps Chrome's profile (its shader caches) between runs; without it a
// fresh profile each run. The driver's Metal cache outlives a profile: a kernel's
// first load on a machine is the cold one (explicit-wgsl-check.mjs EXPLICIT_WGSL_COLD
// times a cold compile on demand). Needs Chrome (CHROME to override).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync, statSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const QUERY = process.argv[2] || 'solver=explicit&scene=yard';
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const keep = process.env.PROFILE;
if (keep) mkdirSync(keep, { recursive: true });
const profile = keep || mkdtempSync(join(tmpdir(), 'explicit-load-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => {
  try { chrome.kill('SIGKILL'); } catch {}
  if (!keep) try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {}
  server.close();
};
process.on('exit', cleanup);
for (const sig of ['SIGALRM', 'SIGTERM', 'SIGINT']) process.on(sig, () => { console.log(`${sig}: stopping`); process.exit(3); });
let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map();
const t0 = Date.now();
let done = null;
const errors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') {
    const text = m.params.args.map((a) => a.value ?? a.description).join(' ');
    if (text.startsWith('[loading]')) console.log(`${((Date.now() - t0) / 1000).toFixed(1).padStart(6)} s  ${text}`);
    if (text.startsWith('[loading] done in')) done = text;
    if (m.params.type === 'error') errors.push(text);
  }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
await send('Runtime.enable');
await send('Page.navigate', { url: `${base}/softbody.html?${QUERY}` });
while (!done && !errors.length && Date.now() - t0 < 1_200_000) await new Promise((r) => setTimeout(r, 200));
if (errors.length) { console.log('page errors:', errors.slice(0, 3)); process.exit(1); }
console.log(done ? `first frame after ${((Date.now() - t0) / 1000).toFixed(1)} s (${QUERY})` : 'no first frame in 20 min');
process.exit(done ? 0 : 1);
