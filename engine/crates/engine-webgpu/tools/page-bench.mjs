// Step time of a page in headless Chrome (WebGPU): serves engine-webgpu/, opens each
// page, waits for window.__ready, then times window.__step(n) after a warm-up. One page
// at a time, one Chrome: nothing else should be on the GPU while it runs.
//
//   node engine/crates/engine-webgpu/tools/page-bench.mjs 'terrain.html?n=2000' ...
//   STEPS=300 WARM=60 node ... --json out.json
//
// SHOT=<dir> saves a screenshot after SHOT_MS of the page's own loop instead.
// CHECK=1 runs window.__check() (or the expression EXPR) instead and prints its result.
//
// Prints, per page, ms per step (wall time of frame-locked automation steps, which
// includes the host's encode and readback overhead) and the page's own __bench()
// result when it has one.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const jsonIdx = args.indexOf('--json');
const JSON_OUT = jsonIdx >= 0 ? args[jsonIdx + 1] : null;
const pages = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--json');
const STEPS = Number(process.env.STEPS || 300);
const WARM = Number(process.env.WARM || 60);
const REPEAT = Number(process.env.REPEAT || 3);
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

const TYPES = { '.glb': 'model/gltf-binary', '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm', '.bin': 'application/octet-stream' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'bench-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true }); } catch {} server.close(); };
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
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('[page]', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('[console]', m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 900_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || 'evaluate failed');
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.enable');

const results = [];
for (const page of pages) {
  await send('Page.navigate', { url: `${base}/${page}` });
  let ready = false;
  for (let i = 0; i < 1800 && !ready; i++) {
    const s = await evaluate('JSON.stringify({ready: !!window.__ready, failed: window.__failed || null})').catch(() => null);
    const st = s ? JSON.parse(s) : {};
    if (st.failed) throw new Error(`${page}: ${st.failed}`);
    ready = !!st.ready;
    if (!ready) await new Promise((r) => setTimeout(r, 100));
  }
  if (!ready) throw new Error(`${page}: never ready`);
  if (process.env.SHOT) {
    // A few frames of the page's own loop, then a screenshot.
    await new Promise((r) => setTimeout(r, Number(process.env.SHOT_MS || 3000)));
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    const file = join(process.env.SHOT, page.replace(/[^a-z0-9]+/gi, '_') + '.png');
    writeFileSync(file, Buffer.from(shot.result.data, 'base64'));
    console.log(`${page} screenshot: ${file}`);
    continue;
  }
  if (process.env.CHECK) {
    const c = await evaluate(`(async () => JSON.stringify(await (${process.env.EXPR || 'window.__check()'})))()`);
    console.log(`${page} __check: ${c}`);
    results.push({ page, check: JSON.parse(c) });
    continue;
  }
  const r = JSON.parse(await evaluate(`(async () => {
    await window.__step(${WARM});
    const runs = [];
    for (let k = 0; k < ${REPEAT}; k++) {
      const t = performance.now();
      await window.__step(${STEPS});
      runs.push((performance.now() - t) / ${STEPS});
    }
    const bench = window.__bench ? await window.__bench() : null;
    return JSON.stringify({ runs, bench });
  })()`));
  const best = Math.min(...r.runs);
  results.push({ page, steps: STEPS, msPerStep: best, runs: r.runs, bench: r.bench });
  console.log(`${page.padEnd(40)} ${best.toFixed(3)} ms/step  (runs ${r.runs.map((x) => x.toFixed(3)).join(', ')})${r.bench ? '  ' + JSON.stringify(r.bench) : ''}`);
}
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));
process.exit(0);
