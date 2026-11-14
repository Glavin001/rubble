#!/usr/bin/env node
// Headless Chrome half of the contact-group dispatch/barrier costs
// (engine/docs/measurements/contact-groups-design.md): serves this directory on a free
// 127.0.0.1 port, opens dispatch.html in headless Chrome (WebGPU on the real GPU,
// timestamps unquantised by the developer features), prints its lines, and stops both.
// Run under the GPU lock:
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- node engine/tools/contact-groups/dispatch.mjs
import http from 'node:http';
import {readFile, mkdtemp, rm} from 'node:fs/promises';
import {spawn} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const types = {'.html': 'text/html', '.mjs': 'text/javascript', '.wgsl': 'text/plain'};
const chrome = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    res.writeHead(200, {'content-type': types[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store'}).end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const page = `http://127.0.0.1:${server.address().port}/dispatch.html`;
const outDir = path.resolve(root, '../../out');
const profile = await mkdtemp(path.join(outDir, 'chrome-dispatch-'));
const browser = spawn(chrome, ['--headless=new', '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
  ...(process.env.CHROME_ARGS ?? '').split(' ').filter(Boolean), 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe']});
const finish = async (code) => {
  const exited = browser.exitCode !== null ? Promise.resolve() : new Promise((ok) => browser.once('exit', ok));
  browser.kill(); server.close(); await exited;
  await rm(profile, {recursive: true, force: true, maxRetries: 5}).catch(() => {});
  process.exit(code);
};
setTimeout(() => { console.error('timed out'); finish(2); }, 240000).unref();
try {
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
  const call = (method, params) => new Promise((ok) => {
    const mine = ++id;
    const on = (e) => { const m = JSON.parse(e.data); if (m.id === mine) { ws.removeEventListener('message', on); ok(m.result); } };
    ws.addEventListener('message', on);
    ws.send(JSON.stringify({id: mine, method, params}));
  });
  for (;;) {
    const r = await call('Runtime.evaluate', {expression: 'window.__result ? JSON.stringify(window.__result) : ""', returnByValue: true});
    if (r?.result?.value) {
      const result = JSON.parse(r.result.value);
      for (const l of result.lines) console.log(l);
      if (result.error) console.error(result.error);
      await finish(result.error ? 1 : 0);
    }
    await new Promise((ok) => setTimeout(ok, 500));
  }
} catch (e) {
  console.error(e);
  await finish(1);
}
