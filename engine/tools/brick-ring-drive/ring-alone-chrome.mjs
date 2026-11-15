// The ring alone in headless Chrome: demo.html ?scene=ring110k (&sleep=on by default),
// wall ms per World step from the page's own __wallNow (steps taken as fast as the
// readbacks allow), after the ring has settled. Run it under engine/scripts/with-gpu-lock.py.
//
//   node ring-alone-chrome.mjs --root <engine-webgpu dir> [--query 'scene=ring110k&sleep=on']
//     [--settle 8] [--rounds 3] [--steps 120]
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync, statSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const ROOT = resolve(opt('--root', '.'));
const QUERY = opt('--query', 'scene=ring110k&sleep=on');
const SETTLE = Number(opt('--settle', '8'));
const ROUNDS = Number(opt('--rounds', '3'));
const STEPS = Number(opt('--steps', '120'));
setTimeout(() => { console.error('time limit (8 min)'); process.exit(9); }, 8 * 60_000).unref();
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const profile = mkdtempSync(join(tmpdir(), 'ring-alone-chrome-'));
const port = 9800 + Math.floor(Math.random() * 150);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'], { stdio: 'ignore' });
process.on('exit', () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true }); } catch {} });

let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0; const waiting = new Map();
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const k = ++id; waiting.set(k, r); ws.send(JSON.stringify({ id: k, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 120_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/demo.html?${QUERY}` });
const t0 = Date.now();
for (;;) {
  await new Promise((r) => setTimeout(r, 250));
  const failed = await evaluate('window.__failed || ""').catch(() => '');
  if (failed) throw new Error(`page error: ${failed}`);
  if (await evaluate('window.__ready === true').catch(() => false)) break;
  if (Date.now() - t0 > 180_000) throw new Error('the page did not start');
}
await new Promise((r) => setTimeout(r, SETTLE * 1000));
for (let k = 0; k < ROUNDS; k++) {
  const w = await evaluate(`window.__wallNow(${STEPS}, 'off')`);
  console.log(`ring alone (${QUERY}) round ${k + 1}: ${w.wallMs.toFixed(2)} ms/step wall, ${(1000 / w.wallMs).toFixed(0)} steps/s possible, awake ${w.awake}`);
}
process.exit(0);
