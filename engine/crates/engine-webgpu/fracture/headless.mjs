#!/usr/bin/env node
// Physics-only fracture benchmark in headless Chrome (WebGPU on the real GPU,
// no canvas or renderer). Serves engine-webgpu/, opens fracture/headless.html
// and prints its JSON result. No npm dependencies (Node 22: fetch, WebSocket).
//
//   node headless.mjs [scene=grid] [warm=420] [measure=120] [profile=1]
//                     [trace=0] [shot=row|none] [budget='{"deterministic":true}']
//   CHROME=/path/to/chrome node headless.mjs ...   (default: Google Chrome)
//   CHROME_ARGS='--enable-dawn-features=disable_robustness' adds browser flags
import http from 'node:http';
import {readFile, mkdtemp, rm} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const types = {'.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.wgsl': 'text/plain', '.glb': 'model/gltf-binary', '.json': 'application/json'};
const params = new URLSearchParams(process.argv.slice(2).map((arg) => arg.includes('=') ? arg : `scene=${arg}`).join('&'));
const chrome = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const timeoutMs = Number(process.env.TIMEOUT_MS ?? 600000);

const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    const body = await readFile(file);
    res.writeHead(200, {'content-type': types[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store'}).end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const page = `http://127.0.0.1:${server.address().port}/fracture/headless.html?${params}`;

const profile = await mkdtemp(path.join(tmpdir(), 'fracture-headless-'));
const browser = spawn(chrome, ['--headless=new', '--enable-unsafe-webgpu', '--remote-debugging-port=0',
  `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
  ...(process.env.CHROME_ARGS ?? '').split(' ').filter(Boolean), 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const finish = async (code) => {
  const exited = browser.exitCode !== null ? Promise.resolve() : new Promise((ok) => browser.once('exit', ok));
  browser.kill(); server.close(); await exited;
  await rm(profile, {recursive: true, force: true, maxRetries: 5}).catch(() => {});
  process.exit(code);
};
setTimeout(() => { console.error(`timed out after ${timeoutMs} ms`); finish(2); }, timeoutMs).unref();

try {
  // Chrome prints its DevTools endpoint on stderr.
  const endpoint = await new Promise((ok, fail) => {
    let text = '';
    browser.stderr.on('data', (d) => { text += d; const m = text.match(/DevTools listening on (ws:\/\/\S+)/); if (m) ok(m[1]); });
    browser.on('exit', (c) => fail(new Error(`Chrome exited (${c}): ${text.slice(-400)}`)));
  });
  const port = new URL(endpoint).port;
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(page)}`, {method: 'PUT'})).json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok) => ws.addEventListener('open', ok, {once: true}));
  let id = 0;
  const call = (method, args) => new Promise((ok) => {
    const mine = ++id;
    const on = (e) => { const m = JSON.parse(e.data); if (m.id === mine) { ws.removeEventListener('message', on); ok(m.result); } };
    ws.addEventListener('message', on);
    ws.send(JSON.stringify({id: mine, method, params: args}));
  });
  for (;;) {
    const r = await call('Runtime.evaluate', {expression: 'window.__result ? JSON.stringify(window.__result) : ""', returnByValue: true});
    const value = r?.result?.value;
    if (value) {
      const result = JSON.parse(value);
      console.log(JSON.stringify(result, null, 2));
      await finish(result.error || result.gpuError ? 1 : 0);
    }
    await new Promise((ok) => setTimeout(ok, 500));
  }
} catch (e) {
  console.error(e);
  await finish(1);
}
