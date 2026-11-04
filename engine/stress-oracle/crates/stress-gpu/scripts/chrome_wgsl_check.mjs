#!/usr/bin/env node
// Every generated stress-gpu WGSL (shaders/generated/*.wgsl, as stress-web hands it to the
// browser) compiled by Chrome's WebGPU (Tint). Native wgpu (naga) accepts shaders Chrome
// rejects, chiefly barriers under non-uniform control flow, and a rejected module fails its
// pipeline and every submission that uses it. For a module Chrome rejects over a barrier,
// each call of a function holding the barrier is dropped in turn, naming the call sites
// whose removal lets the module compile.
//
//   node --experimental-websocket scripts/chrome_wgsl_check.mjs [name ...]
//
// Needs Chrome (CHROME env to override). Exits non-zero when a module fails to compile.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'shaders', 'generated');
const only = process.argv.slice(2);
const names = readdirSync(DIR).filter((f) => f.endsWith('.wgsl')).map((f) => f.slice(0, -5)).filter((n) => !only.length || only.includes(n)).sort();
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

const server = createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://x').pathname.slice(1));
  const path = join(DIR, name);
  if (!name || name.includes('/') || !existsSync(path)) { res.writeHead(200, { 'content-type': 'text/html' }).end('<!doctype html>'); return; }
  res.writeHead(200, { 'content-type': 'text/plain', 'cache-control': 'no-store' }).end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'wgsl-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--no-first-run', '--no-default-browser-check', 'about:blank'], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} server.close(); };
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
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Page.navigate', { url: `${base}/` });
await new Promise((r) => setTimeout(r, 500));

// In the page: compile a module's text; on a barrier error, drop each call of each function
// that holds a barrier (one call at a time) and keep the calls whose removal compiles.
const check = async (names) => evaluate(`(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const limits = {};
  for (const k of ['maxStorageBuffersPerShaderStage', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX']) limits[k] = adapter.limits[k];
  const device = await adapter.requestDevice({ requiredLimits: limits });
  const errors = async (code) => (await device.createShaderModule({ code }).getCompilationInfo()).messages.filter((m) => m.type === 'error');
  const out = [];
  for (const name of ${JSON.stringify(names)}) {
    const code = await (await fetch('/' + name + '.wgsl')).text();
    const errs = await errors(code);
    if (!errs.length) { out.push({ name, ok: true }); continue; }
    const r = { name, ok: false, errors: errs.slice(0, 3).map((e) => 'line ' + e.lineNum + ': ' + e.message + ' | ' + (code.split('\\n')[e.lineNum - 1] || '').trim().slice(0, 120)) };
    if (errs.some((e) => /uniform control flow/.test(e.message))) {
      const lines = code.split('\\n');
      const holders = [];
      for (let i = 0, fn = null; i < lines.length; i++) {
        const f = lines[i].match(/^fn (\\w+)\\(/);
        if (f) fn = f[1];
        if (fn && /Barrier\\(/.test(lines[i]) && !holders.includes(fn)) holders.push(fn);
      }
      r.culprits = [];
      for (const fn of holders) {
        for (let i = 0; i < lines.length; i++) {
          if (!new RegExp('\\\\b' + fn + '\\\\(\\\\)').test(lines[i]) || lines[i].startsWith('fn ')) continue;
          const trial = lines.slice();
          trial[i] = trial[i].replace(new RegExp('\\\\b' + fn + '\\\\(\\\\);'), '');
          const left = await errors(trial.join('\\n'));
          if (!left.some((e) => /uniform control flow/.test(e.message))) r.culprits.push('line ' + (i + 1) + ': ' + fn + ' (' + lines[i].trim() + ')');
        }
      }
    }
    out.push(r);
  }
  return out;
})()`);

const results = await check(names);
let failed = 0;
for (const r of results) {
  if (r.ok) { console.log(`ok    ${r.name}`); continue; }
  failed++;
  console.log(`FAIL  ${r.name}`);
  for (const e of r.errors) console.log(`        ${e}`);
  for (const c of r.culprits || []) console.log(`        non-uniform call: ${c}`);
}
console.log(`${results.length - failed} of ${results.length} modules compile in Chrome`);
cleanup();
process.exit(failed ? 1 : 0);
