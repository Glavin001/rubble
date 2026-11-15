#!/usr/bin/env node
// Opens one page in headless Chrome WebGPU, waits, evaluates an expression and
// saves a screenshot: a quick look at any demo URL without touching a window
// the user is in.
//
//   engine/scripts/with-gpu-lock.py --limit 180 -- \
//     node page-shot.mjs '<url>' [--until '<js expression, truthy when ready>'] [--wait ms] \
//       [--eval '<js expression>'] [--out file.png] [--size 1280x720]
//
// `--until` is polled every 250 ms (60 s at most), then `--wait` passes, then
// `--eval`'s JSON is printed with the page's console errors, and the
// screenshot is written. Needs Chrome (CHROME env to override).
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromeArgs } from './check-chrome.mjs';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const url = args[0];
if (!url || url.startsWith('--')) { console.error('usage: page-shot.mjs <url> [--until expr] [--wait ms] [--eval expr] [--out png] [--size WxH]'); process.exit(2); }
const [width, height] = opt('--size', '1280x720').split('x').map(Number);
const CHROME = process.env.CHROME || (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'page-shot-'));
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width, height }), { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 0;
try {
  let target;
  for (let i = 0; i < 80 && !target; i++) {
    await sleep(250);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page'); } catch { /* not up yet */ }
  }
  if (!target) throw new Error('Chrome did not start');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map(), errors = [];
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    if (d.method === 'Runtime.exceptionThrown') errors.push(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
    if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error') errors.push(d.params.args.map((a) => a.value ?? a.description).join(' '));
  };
  const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    return r.result?.result?.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Page.navigate', { url });
  const until = opt('--until');
  if (until) {
    let ok = false;
    for (let i = 0; i < 240 && !ok; i++) { await sleep(250); try { ok = !!(await evaluate(`!!(${until})`)); } catch { /* loading */ } }
    if (!ok) { console.error(`--until never held: ${until}`); code = 1; }
  }
  await sleep(Number(opt('--wait', 0)));
  const ev = opt('--eval');
  if (ev) console.log(JSON.stringify(await evaluate(`(${ev})`)));
  if (errors.length) console.log(`console errors: ${JSON.stringify(errors.slice(0, 8))}`);
  const out = opt('--out');
  if (out) {
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
    console.log(`wrote ${out}`);
  }
  ws.close();
} catch (e) {
  console.error(String(e));
  code = 1;
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
process.exit(code);
