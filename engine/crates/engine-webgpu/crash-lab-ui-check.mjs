#!/usr/bin/env node
// The crash lab's layout (softbody.html?solver=explicit, crash-lab-ui.mjs) in headless
// Chrome at several window sizes: a screenshot of each state (as loaded, every panel
// collapsed, every panel open, focus mode, the results open once the run is done, a
// keyboard-only pass) and a check that no two panels overlap, none leaves the window
// and none clips its text sideways. Layout only: the physics is the page's, stepped
// frame-locked from here (window.__lab.advance) so the vehicles are mid-crash.
//
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- \
//     node crash-lab-ui-check.mjs [--case S05|S02|wall56|<query>] [--tag after] \
//       [--sizes 800x640,1280x720,1920x1080] [--out DIR]
//
// Output: DIR (default engine/out/lab-ui/<tag>/): <case>-<W>x<H>-<state>.png and
// <case>.json (each state's panel rectangles and problems). Exit code 1 on a page error,
// a console error or a layout problem (overlap, off-screen, clipped). A page without the
// layout module (an older commit: --tag before) gets its default and clean states only.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromeArgs } from './check-chrome.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const TAG = opt('--tag', 'after');
const OUT = opt('--out', normalize(join(ROOT, '../../out/lab-ui', TAG)));
mkdirSync(OUT, { recursive: true });
const CASES = { S05: 'crashRef=S05&run=0&contact=discrete', S02: 'crashRef=S02&run=0&contact=discrete', wall56: 'preset=wall56' };
const CASE = opt('--case', 'S05');
const QUERY = CASES[CASE] ?? CASE;
const NAME = CASES[CASE] ? CASE : 'case';
const SIZES = opt('--sizes', '800x640,1280x720,1920x1080').split(',').map((s) => s.split('x').map(Number));
const CHROME = process.env.CHROME || (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const p = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (!p.startsWith(ROOT) || !existsSync(p) || statSync(p).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(p));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
// The shader cache is kept between runs (a fresh origin each run: no saved panel states).
const profile = process.env.LAB_UI_PROFILE || normalize(join(ROOT, '../../out/lab-ui/chrome-profile'));
mkdirSync(profile, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width: 1920, height: 1080 }), { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errors = [];
const problems = [];
const record = {};
let code = 0;

// The panels and overlays whose rectangles are checked (old page and new).
const RECTS = `(() => {
  const sel = ['#info', '#hud', '#labPanel', 'body > #crashRef', '#labPills', '#labTime', '#labCounts', '#labLegend', '.lab-dock'];
  const out = [];
  for (const s of sel) for (const e of document.querySelectorAll(s)) {
    const cs = getComputedStyle(e);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const r = e.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    // Clipped sideways: a child wider than the box without a scroller.
    const clipped = e.scrollWidth > e.clientWidth + 1 && cs.overflowX !== 'auto' && cs.overflowX !== 'scroll';
    out.push({ id: e.id || e.dataset.dock || s, x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height), clipped, text: e.innerText.slice(0, 60).replace(/\\n/g, ' | ') });
  }
  return { rects: out, w: innerWidth, h: innerHeight, focus: document.activeElement?.id || document.activeElement?.tagName };
})()`;
// Containers whose children legitimately overlap them (the HUD inside a dock column).
const nests = (a, b) => a.x <= b.x && a.y <= b.y && a.x + a.w >= b.x + b.w && a.y + a.h >= b.y + b.h;
function check(state, r) {
  const list = r.rects;
  for (const a of list) {
    if (a.x < 0 || a.y < 0 || a.x + a.w > r.w || a.y + a.h > r.h) problems.push(`${state}: ${a.id} leaves the window (${a.x},${a.y} ${a.w}x${a.h} in ${r.w}x${r.h})`);
    if (a.clipped) problems.push(`${state}: ${a.id} clips its content sideways`);
  }
  for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
    const a = list[i], b = list[j];
    if (nests(a, b) || nests(b, a)) continue;
    const ox = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x), oy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
    if (ox > 0 && oy > 0) problems.push(`${state}: ${a.id} overlaps ${b.id} by ${ox}x${oy}`);
  }
}

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
  const pending = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    if (d.method === 'Runtime.exceptionThrown') errors.push(d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text);
    if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error') errors.push(d.params.args.map((a) => a.value ?? a.description).join(' '));
  };
  const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  const L = (expr) => evaluate(`(async () => { const L = window.__lab; return ${expr}; })()`);
  const size = async (w, h) => { await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false }); await sleep(150); await L('L.draw()'); await L('L.draw()'); };
  // A key as the keyboard sends it (code, key, Shift).
  const KEYS = { KeyH: ['h', 72], KeyS: ['s', 83], KeyV: ['v', 86], KeyR: ['r', 82], KeyP: ['p', 80], Escape: ['Escape', 27], Tab: ['Tab', 9], Enter: ['Enter', 13], Space: [' ', 32] };
  const key = async (code, shift = false) => {
    const [k, vk] = KEYS[code];
    const key = shift && k.length === 1 ? k.toUpperCase() : k;
    const base = { code, key, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk, modifiers: shift ? 8 : 0 };
    await send('Input.dispatchKeyEvent', { type: 'rawKeyDown', ...base });
    if (key.length === 1 || code === 'Enter') await send('Input.dispatchKeyEvent', { type: 'char', ...base, text: code === 'Enter' ? '\r' : key });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', ...base });
    await sleep(60);
  };
  const shot = async (w, h, state) => {
    await L('L.draw()');
    const r = await evaluate(RECTS);
    const name = `${NAME}-${w}x${h}-${state}`;
    record[name] = r;
    check(name, r);
    const s = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, `${name}.png`), Buffer.from(s.result.data, 'base64'));
    console.log(`  ${name}.png${r.rects.length ? '' : ' (no panels)'}`);
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  const url = `${base}/softbody.html?solver=explicit&paused=1&${QUERY}`;
  console.log(url);
  await send('Page.navigate', { url });
  const t0 = Date.now();
  for (;;) {
    await sleep(500);
    const st = await evaluate('({ ready: window.__ready === true && !!window.__lab, failed: window.__failed ?? null, err: document.getElementById("err")?.textContent || "" })').catch(() => ({}));
    if (st.failed || st.err) throw new Error(`page error: ${st.failed || st.err}`);
    if (st.ready) break;
    if (Date.now() - t0 > 200_000) throw new Error('the page did not start in 200 s');
  }
  console.log(`ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  const ref = await evaluate('window.__crashRef ? window.__crashRef.info() : null');
  const ui = await evaluate('!!window.__labUi');
  // Mid-crash: 60 ms after first contact (a reference crash's), else 0.6 s in.
  const mid = ref ? Math.round(ref.impactStep + 120) : 1200;
  let at = 0;
  const advanceTo = async (step) => { while (at < step) { const k = Math.min(step - at, 64); at = (await L(`L.advance(${k}, ${k})`)).step; } };
  await advanceTo(mid);
  for (const [w, h] of SIZES) {
    await size(w, h);
    await shot(w, h, 'default');
    if (ui) {
      // Every panel collapsed, then open, from their shortcuts (Shift+S/V/R/P).
      const open = await evaluate('window.__labUi.state()');
      for (const [p, k] of [['scene', 'KeyS'], ['view', 'KeyV'], ['results', 'KeyR'], ['playback', 'KeyP']]) if (open[p]) await key(k, true);
      await shot(w, h, 'collapsed');
      for (const [p, k] of [['scene', 'KeyS'], ['view', 'KeyV'], ['results', 'KeyR'], ['playback', 'KeyP']]) if (!(await evaluate('window.__labUi.state()'))[p]) await key(k, true);
      await shot(w, h, 'expanded');
      await evaluate('window.__labUi.restore(' + JSON.stringify(open) + ')');
    }
    // Focus (clean) mode from its key, and back.
    await key('KeyH');
    await shot(w, h, 'focus');
    await key(ui ? 'Escape' : 'KeyH');
  }
  if (ui) {
    // Keyboard only at the owner's width: Tab to the Results handle, Enter opens it.
    const [w, h] = SIZES[0];
    await size(w, h);
    await evaluate('document.activeElement?.blur(), document.body.focus(), true');
    let found = false;
    for (let i = 0; i < 60 && !found; i++) { await key('Tab'); found = await evaluate('document.activeElement?.dataset?.dockToggle === "results"'); }
    if (!found) problems.push('keyboard: Tab never reached the Results handle');
    await shot(w, h, 'keyboard-focus');
    const before = (await evaluate('window.__labUi.state()')).results;
    await key('Enter');
    const after = (await evaluate('window.__labUi.state()')).results;
    if (before === after) problems.push('keyboard: Enter on the Results handle did not toggle it');
    await shot(w, h, 'keyboard-enter');
    await evaluate('window.__labUi.open("results", false), true');
  }
  if (ref) {
    // The whole run: the report, with the results open.
    await advanceTo(ref.totalSteps);
    for (let i = 0; i < 100 && !(await evaluate('!!window.__crashRef.report()')); i++) await sleep(100);
    if (ui) await evaluate('window.__labUi.open("results", true), true');
    for (const [w, h] of SIZES) { await size(w, h); await shot(w, h, 'done'); }
  }
  writeFileSync(join(OUT, `${NAME}.json`), JSON.stringify({ url, problems, errors, record }, null, 1));
  if (errors.length) { console.log(`console errors: ${JSON.stringify(errors.slice(0, 8))}`); code = 1; }
  if (problems.length) { console.log(`layout problems:\n  ${problems.join('\n  ')}`); if (TAG !== 'before') code = 1; }
  else console.log('layout: no overlaps, nothing off-screen or clipped');
  console.log(`→ ${OUT}`);
  ws.close();
} catch (e) {
  console.error(String(e?.stack ?? e));
  code = 1;
} finally {
  chrome.kill();
  server.close();
  await sleep(300);
}
process.exit(code);
