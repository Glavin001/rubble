#!/usr/bin/env node
// The crash town in Chrome WebGPU (headless, 1280x720): for each car count, opens
// explicit-world.html?n=N&timing=phases from a built site (engine/scripts/build-web.sh's
// OUT), waits for the page to run, then samples the HUD (World steps/s, the GPU time per
// World step by segment) for a few seconds. Fails on a page error.
//
//   node --experimental-websocket crash-town-chrome.mjs <site dir> [n ...] [--secs S] [--warm W]
//
// Run it under bounded.sh (the GPU lock and a kill timer). CHROME overrides the binary.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, normalize } from 'node:path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? Number(args[i + 1]) : d; };
const SITE = normalize(args[0]);
const counts = args.slice(1).filter((a, i, all) => !a.startsWith('--') && !(all[i - 1] || '').startsWith('--')).map(Number);
const SECS = opt('--secs', 6);
const WARM = opt('--warm', 6);
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm', '.css': 'text/css' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(SITE, p));
  if (!path.startsWith(SITE) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'crash-town-chrome-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
process.on('exit', cleanup);

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
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true });
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let failed = false;
for (const n of counts) {
  pageErrors.length = 0;
  await send('Page.navigate', { url: `${base}/explicit-world.html?n=${n}&timing=phases` });
  // Ready: the HUD shows World steps.
  let ready = false;
  for (let i = 0; i < 600 && !ready; i++) {
    await sleep(200);
    const t = await evaluate('document.getElementById("hud")?.innerText || ""').catch(() => '');
    ready = /World steps\s+[1-9]/.test(t);
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err || pageErrors.length) { console.log(`n=${n} PAGE ERROR: ${err || pageErrors[0]}`); failed = true; break; }
  }
  if (!ready) { console.log(`n=${n}: the page did not start; HUD: ${await evaluate('document.getElementById("hud")?.innerText || document.body.innerText.slice(0, 400)')}`); failed = true; continue; }
  await sleep(WARM * 1000);
  const samples = [];
  for (let i = 0; i < SECS; i++) {
    await sleep(1000);
    samples.push(await evaluate('document.getElementById("hud")?.innerText || ""'));
  }
  if (pageErrors.length) { console.log(`n=${n} PAGE ERROR: ${pageErrors[0]}`); failed = true; }
  // Each sample: the HUD's label and value lines paired, the timing ones kept.
  console.log(`=== n=${n}`);
  for (const s of samples) {
    const l = s.split('\n');
    const keep = [];
    for (let i = 0; i + 1 < l.length; i++) {
      if (/^(GPU \/ step|real time|explicit |cpl_|primal|bonds broken)/.test(l[i])) keep.push(`${l[i]}: ${l[i + 1]}`);
    }
    console.log(`  ${keep.join(' | ')}`);
  }
}
ws.close();
server.close();
cleanup();
process.exit(failed ? 1 : 0);
