#!/usr/bin/env node
// The loading card (load-progress.mjs) driven with a scripted load in headless
// Chrome, no GPU (--disable-gpu): the kernel list's states, the estimate from the
// stored times, the Cancel button, and the device-lost state with its Retry. Then
// the stop panel (stability-stop.mjs) with a scripted health check.
//
//   node --experimental-websocket load-progress-check.mjs [shots-dir]
//
// With a directory, a screenshot of each state is written there. Exits non-zero
// when a check fails. Needs Chrome (CHROME env to override).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const SHOTS = process.argv[2];
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const PAGE = `<!doctype html><meta charset="utf-8"><title>loading card</title>
<body style="margin:0;height:100vh;background:#223"><script type="module">
import { createLoading, isDeviceLost } from './load-progress.mjs';
import { createStabilityStop } from './stability-stop.mjs';
// A scripted VehicleHealth: healthy until __health.message is set.
window.__health = { message: null, checks: 0, damages: 0, stopped() { return undefined; }, check() { this.checks++; return Promise.resolve(this.message ?? undefined); }, damage() { this.damages++; }, launched() {} };
window.__stop = createStabilityStop(window.__health, { params: { xscene: 'sidepole', speed: 186 }, onStop: () => { window.__onStop = (window.__onStop ?? 0) + 1; } });
window.__make = (withCancel) => {
  document.getElementById('loading')?.remove();
  document.getElementById('fatal')?.remove();
  window.__cancelled = 0;
  window.__retried = 0;
  window.__loading = createLoading({ title: 'Loading Rubble', cancel: withCancel ? { label: 'Cancel: load without surface contact', run: () => { window.__cancelled++; } } : null });
  return true;
};
window.__lost = isDeviceLost;
window.__text = () => (document.getElementById('loading') || document.getElementById('fatal'))?.innerText ?? '';
window.__ready = true;
</script>`;
const server = createServer((req, res) => {
  const p = new URL(req.url, 'http://x').pathname;
  if (p === '/') { res.writeHead(200, { 'content-type': 'text/html' }).end(PAGE); return; }
  if (p === '/stability-stop.mjs') { res.writeHead(200, { 'content-type': 'text/javascript' }).end(readFileSync(join(ROOT, 'stability-stop.mjs'))); return; }
  if (p === '/load-progress.mjs') { res.writeHead(200, { 'content-type': 'text/javascript' }).end(readFileSync(join(ROOT, 'load-progress.mjs'))); return; }
  res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), 'load-progress-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', '--disable-gpu', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--window-size=900,700', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} server.close(); };
process.on('exit', cleanup);
for (const sig of ['SIGALRM', 'SIGTERM', 'SIGINT']) process.on(sig, () => process.exit(3));
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
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result.result.value;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const shot = async (name) => {
  if (!SHOTS) return;
  const r = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(SHOTS, `${name}.png`), Buffer.from(r.result.data, 'base64'));
};
let failed = 0;
const check = (what, ok, text) => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`);
  if (!ok) { failed++; if (text !== undefined) console.log(text); }
};

await send('Page.enable');
await send('Page.navigate', { url: base + '/' });
for (let i = 0; i < 100 && !(await evaluate('window.__ready === true').catch(() => false)); i++) await sleep(50);
await evaluate('localStorage.clear()');

const KERNELS = JSON.stringify([
  { name: 'explicit_beams_contact', key: 'aaaa', bytes: 740743, deferred: false },
  { name: 'explicit_beams_solo', key: 'bbbb', bytes: 596726, deferred: false },
  { name: 'explicit_beams_vv_group', key: 'cccc', bytes: 24357, deferred: false },
  { name: 'explicit_beams_seg', key: 'dddd', bytes: 366687, deferred: true },
]);
const say = (line) => evaluate(`window.__loading.progress(${JSON.stringify(line)})`);

// 1. A first load: no record, so no estimate; every waited kernel compiling with a clock.
await evaluate('window.__make(true)');
await evaluate(`window.__loading.stage('Reading the scene')`);
await say('spawn');
await say(`kernels ${KERNELS}`);
await sleep(350);
let t = await evaluate('window.__text()');
check('first load: every kernel queued, smallest first', /explicit_beams_vv_group\s+queued · first time here\s+explicit_beams_solo\s+queued[^\n]*\s+explicit_beams_contact\s+queued/.test(t), t);
await say('compiling explicit_beams_vv_group');
await sleep(350);
t = await evaluate('window.__text()');
check('first load: the one compiling has a running clock', /explicit_beams_vv_group\s+compiling 0\.[2-9] s · first time here/.test(t), t);
check('first load: says which kernel of how many, by name', /Compiling GPU kernel 1 of 3: explicit_beams_vv_group/.test(t), t);
check('first load: no estimate is claimed, and the cache is explained', /No estimate/.test(t) && /next load of this version skips the wait/.test(t), t);
check('first load: a deferred kernel is queued, not counted', /explicit_beams_seg\s+queued: compiles after the first frame/.test(t), t);
check('no "about a minute"', !/about a minute/i.test(t), t);
check('cancel button shown while compiling', /Cancel: load without surface contact/.test(t), t);
const clockA = t.match(/total ([\d.]+) s/)?.[1];
await sleep(300);
t = await evaluate('window.__text()');
check('the clock keeps moving', t.match(/total ([\d.]+) s/)?.[1] !== clockA, t);
await say('compiled explicit_beams_vv_group 200');
await say('compiling explicit_beams_solo');
await sleep(120);
t = await evaluate('window.__text()');
check('a finished kernel shows its time, the next one compiles', /explicit_beams_vv_group\s+done 0\.2 s/.test(t) && /kernel 2 of 3: explicit_beams_solo/.test(t) && /explicit_beams_contact\s+queued/.test(t), t);
await shot('1-first-load');
await evaluate(`document.querySelector('#loading .rb-actions button').click()`);
check('cancel runs the page\'s fallback', (await evaluate('window.__cancelled')) === 1);
await say('compiled explicit_beams_solo 61000');
await say('compiling explicit_beams_contact');
await say('compiled explicit_beams_contact 95000');
await sleep(120);
t = await evaluate('window.__text()');
check('all compiled: the card says what it is doing next', /GPU kernels compiled \(3\)/.test(t) && !/Cancel/.test(t), t);
await evaluate(`window.__loading.stage('Building the renderers')`);
const times = await evaluate('JSON.stringify(window.__loading.done())');
check('done() returns stages and kernels', JSON.parse(times).kernels.length === 4 && JSON.parse(times).stages.length >= 3, times);
check('the card is gone once loaded', (await evaluate('window.__text()')) === '');

// 2. The second load of the same kernels: an estimate from the stored times.
await evaluate('window.__make(false)');
await say(`kernels ${KERNELS}`);
await say('compiling explicit_beams_vv_group');
await sleep(150);
t = await evaluate('window.__text()');
check('second load: each kernel shows last time', /explicit_beams_contact\s+queued · last time 1 min 35 s/.test(t) && /explicit_beams_vv_group\s+compiling [\d.]+ s · last time 0\.2 s/.test(t), t);
check('second load: an estimate from this browser\'s record (the sum: one at a time)', /About 2 min 3\d s left, going by this browser's last load/.test(t), t);
check('second load: no cancel when the page gave none', !/Cancel/.test(t), t);
await shot('2-second-load');

// 3. A changed kernel: its record belongs to another version, so no estimate.
await evaluate('window.__make(true)');
await say(`kernels ${KERNELS.replace('aaaa', 'eeee')}`);
await say('compiling explicit_beams_vv_group');
await sleep(150);
t = await evaluate('window.__text()');
check('changed kernel: named as changed, with the previous version\'s time', /explicit_beams_contact\s+queued · changed; the previous version took 1 min 35 s/.test(t), t);
check('changed kernel: no estimate claimed', /No estimate/.test(t), t);

// 4. The device is lost mid-compile: an error state with Retry, not a stuck bar.
await say('compiled explicit_beams_vv_group 250');
await say('compiling explicit_beams_solo');
await say('failed explicit_beams_solo 78000 A valid external Instance reference no longer exists.');
await evaluate(`window.__loading.fail({ detail: 'The browser lost the GPU device while compiling the solver\\'s kernels (1 of 3 done after 78 s): device lost (Unknown): A valid external Instance reference no longer exists.', retry: () => { window.__retried++; } })`);
await sleep(300);
t = await evaluate('window.__text()');
check('device lost: titled as such', /The browser lost the GPU device/.test(t.split('\n')[0]), t);
check('device lost: kernel states kept', /explicit_beams_vv_group\s+done 0\.3 s/.test(t) && /explicit_beams_solo\s+failed at 78\.0 s/.test(t) && /explicit_beams_contact\s+not started/.test(t), t);
check('device lost: Retry and the fallback offered', /Retry/.test(t) && /Cancel: load without surface contact/.test(t), t);
const frozen = t;
await sleep(300);
check('device lost: the clocks stopped', (await evaluate('window.__text()')) === frozen);
check('device lost: the bar stopped animating', (await evaluate(`getComputedStyle(document.querySelector('#fatal .rb-bar i')).animationName`)) === 'none');
await shot('3-device-lost');
await evaluate(`[...document.querySelectorAll('#fatal button')].find((b) => b.textContent === 'Retry').click()`);
check('Retry runs', (await evaluate('window.__retried')) === 1);
check('isDeviceLost', (await evaluate(`window.__lost('A valid external Instance reference no longer exists') && !window.__lost('shader error')`)) === true);

// 5. The stop panel: healthy checks show nothing; a message stops for good.
await evaluate(`document.getElementById('fatal')?.remove()`);
await evaluate('window.__stop.check(0.5); window.__stop.damage(0, 1, 2, 3)');
await sleep(50);
check('healthy: no panel, not stopped', (await evaluate(`window.__stop.stopped === null && !document.getElementById('stopped')`)) === true);
const MSG = 'ENGINE ERROR: instability detected at t = 1.050 s: the vehicles\' kinetic energy rose from 1,935 kJ to 32,460 kJ, past 8,854 kJ. This is a simulation bug, not a limit: the simulation was stopped so the broken state does not overload the GPU. Please report it with the scene link and the numbers below.\nvehicle 0: finite true, speed 12.0 km/h';
await evaluate(`window.__health.message = ${JSON.stringify(MSG)}; window.__stop.check(1.05)`);
await sleep(100);
t = await evaluate(`document.getElementById('stopped')?.innerText ?? ''`);
check('unstable: an engine error panel, loud', /^Engine error: the simulation became unstable and was stopped/.test(t) && /This is a bug in the simulation, not a limit/.test(t), t);
check('unstable: the evidence, the scene link and the parameters are in the report', /kinetic energy rose from 1,935 kJ to 32,460 kJ/.test(t) && /scene: http:\/\/127\.0\.0\.1/.test(t) && /"xscene":"sidepole"/.test(t), t);
check('unstable: Restart and Copy offered', /Restart \(reload the scene\)/.test(t) && /Copy the report/.test(t), t);
check('unstable: stopped for good, the page told once', (await evaluate(`window.__stop.stopped === ${JSON.stringify(MSG)} && window.__stopped === window.__stop.stopped && window.__onStop === 1`)) === true);
const before = await evaluate('window.__health.checks + window.__health.damages');
await evaluate('window.__stop.check(2); window.__stop.check(3); window.__stop.damage(0, 1, 2, 3)');
check('unstable: no further checks once stopped', (await evaluate('window.__health.checks + window.__health.damages')) === before);
await shot('4-engine-error');

console.log(failed ? `${failed} check(s) failed` : 'all checks passed');
process.exit(failed ? 1 : 0);
