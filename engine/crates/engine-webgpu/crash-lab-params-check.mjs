#!/usr/bin/env node
// The crash lab's scene configurations (crash-lab-scene.mjs) in headless Chrome: the panel
// shell while the engine loads, the scene picker grouped by family, and a scene's
// parameters as controls placing it again at run time (no reload): a reference crash
// (S02: its speed, then its car), a lab preset (wall56: its speed) and a family opened
// directly (pole). Screenshots at each size; fails on a page or console error, a reload, or
// a change that does not re-place the scene.
//
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- \
//     node crash-lab-params-check.mjs [--sizes 800x640,1280x720] [--out DIR] [--only loading,S02,wall56,pole]
//
// Output: DIR (default engine/out/families/): <case>-<W>x<H>-<state>.png, check.json.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromeArgs } from './check-chrome.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', normalize(join(ROOT, '../../out/families')));
mkdirSync(OUT, { recursive: true });
const SIZES = opt('--sizes', '800x640,1280x720').split(',').map((s) => s.split('x').map(Number));
const ONLY = opt('--only', 'loading,S02,wall56,pole').split(',');
const CHROME = process.env.CHROME || (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
// `slow`: every request held this long (ms) while set: the loading state stays on screen.
let slow = 0;
const server = createServer((req, res) => {
  const p = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (!p.startsWith(ROOT) || !existsSync(p) || statSync(p).isDirectory()) { res.writeHead(404).end(); return; }
  const send = () => { res.writeHead(200, { 'content-type': TYPES[extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' }); res.end(readFileSync(p)); };
  // The page and its modules come at once; the wasm and the rest wait (the loading stage).
  if (slow && !/\.(html|mjs)$/.test(p)) setTimeout(send, slow); else send();
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = process.env.LAB_UI_PROFILE || normalize(join(ROOT, '../../out/lab-ui/chrome-profile'));
mkdirSync(profile, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width: 1920, height: 1080 }), { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errors = [], problems = [], record = {};
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
  const size = (w, h) => send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
  const png = async (name, clip = null) => {
    const s = await send('Page.captureScreenshot', { format: 'png', ...(clip ? { clip: { ...clip, scale: 1 } } : {}) });
    writeFileSync(join(OUT, `${name}.png`), Buffer.from(s.result.data, 'base64'));
    console.log(`  ${name}.png`);
  };
  const draw = () => evaluate('window.__lab?.draw?.() ?? null');
  const open = async (query) => {
    const url = `${base}/crash-lab.html?${query}`;
    console.log(url);
    await send('Page.navigate', { url });
    const t0 = Date.now();
    for (;;) {
      await sleep(400);
      const st = await evaluate('({ ready: window.__ready === true && !!window.__lab, failed: window.__failed ?? null, err: document.getElementById("err")?.textContent || "" })').catch(() => ({}));
      if (st.failed || st.err) throw new Error(`page error: ${st.failed || st.err}`);
      if (st.ready) break;
      if (Date.now() - t0 > 200_000) throw new Error('the page did not start in 200 s');
    }
    // A mark that survives only while the page is not reloaded.
    await evaluate('window.__noReload = 1');
    return url;
  };
  const advance = (n) => evaluate(`window.__lab.advance(${n}, 16)`);
  // The Scene panel's rectangle (its clip), open.
  const sceneClip = () => evaluate(`(() => { window.__labUi.open('scene', true); const r = document.getElementById('labScene').getBoundingClientRect(); return { x: Math.max(0, r.left - 4), y: Math.max(0, r.top - 4), width: Math.min(innerWidth, r.width + 8), height: Math.min(innerHeight - r.top, r.height + 8) }; })()`);
  // A control's new value, as the user commits it (a number field's change; a select's).
  const setParam = (key, value) => evaluate(`(() => { const r = document.querySelector('.lab-params .pr[data-k="${key}"]'); if (!r) return 'no control ${key}'; const c = r.querySelector('input[type=number]') ?? r.querySelector('select') ?? r.querySelector('input'); c.value = ${JSON.stringify(String(value))}; c.dispatchEvent(new Event('change')); return 'ok'; })()`);
  const settle = async (what) => {
    for (let i = 0; i < 400; i++) {
      await sleep(150);
      const busy = await evaluate(`(document.querySelector('.lab-params .note')?.textContent ?? '').startsWith('placing')`);
      if (!busy) return;
    }
    problems.push(`${what}: still placing after 60 s`);
  };
  const state = () => evaluate(`(() => { const i = window.__explicit.sim().crashInfo(); const L = window.__explicit.layout(); return { noReload: window.__noReload === 1, url: location.search, modified: !document.querySelector('.lab-params .mod')?.hidden, note: document.querySelector('.lab-params .note')?.textContent ?? '', bodies: i?.bodies ?? null, boxes: L.boxes, vehicles: L.vehicles.map((v) => v.name), step: window.__explicit.sim().steps(), head: document.getElementById('crHead')?.innerText ?? null, pills: document.getElementById('labPills').innerText }; })()`);
  await send('Runtime.enable');
  await send('Page.enable');

  if (ONLY.includes('loading')) {
    // The loading state: the panels' shell before the engine is up (requests held 4 s).
    for (const [w, h] of SIZES) {
      await size(w, h);
      slow = 4000;
      await send('Page.navigate', { url: `${base}/crash-lab.html?crashRef=S02&run=0&contact=edge` });
      await sleep(2500);
      const s = await evaluate(`({ lab: document.body.classList.contains('lab-on'), info: document.getElementById('info') ? getComputedStyle(document.getElementById('info')).display : 'none', card: !!document.getElementById('loading'), scene: document.getElementById('labSceneBody')?.innerText ?? null })`).catch((e) => ({ e: String(e) }));
      record[`loading-${w}x${h}`] = s;
      if (!s.lab || s.info !== 'none') problems.push(`loading ${w}x${h}: the lab's shell is not up while loading (${JSON.stringify(s)})`);
      await png(`loading-${w}x${h}`);
      // Later: the progress card over the 3D area, beside the panels (not over one).
      for (let i = 0; i < 120 && !(await evaluate(`!!document.getElementById('loading')`).catch(() => false)); i++) await sleep(250);
      await sleep(300);
      const card = await evaluate(`(() => { const c = document.getElementById('loading')?.getBoundingClientRect(); const docks = [...document.querySelectorAll('.lab-dock')].map((d) => d.getBoundingClientRect()).filter((r) => r.width > 0); return { card: c && { x: c.left, y: c.top, w: c.width, h: c.height }, overlaps: c ? docks.filter((r) => Math.min(c.right, r.right) > Math.max(c.left, r.left) && Math.min(c.bottom, r.bottom) > Math.max(c.top, r.top)).length : -1 }; })()`).catch((e) => ({ e: String(e) }));
      record[`loading-card-${w}x${h}`] = card;
      if (card.overlaps) problems.push(`loading ${w}x${h}: the progress card overlaps ${card.overlaps} panel(s) (${JSON.stringify(card)})`);
      await png(`loading-card-${w}x${h}`);
      slow = 0;
      await open('crashRef=S02&run=0&contact=edge');
      await draw();
      await png(`loaded-${w}x${h}`);
    }
  }

  if (ONLY.includes('S02')) {
    // A reference crash: its parameters; the speed, then the car, changed.
    await size(1280, 720);
    await open('crashRef=S02&run=0&contact=edge');
    const s0 = await state();
    record.S02_start = s0;
    await advance(160);
    for (const [w, h] of SIZES) { await size(w, h); await draw(); await png(`S02-${w}x${h}-scene`, await sceneClip()); await png(`S02-${w}x${h}-full`); }
    // The picker, drawn open (a native list is not in a screenshot: shown as a list box).
    await size(1280, 720);
    // Its option tree (groups and entries, as the select lists them) drawn as an overlay.
    await evaluate(`(() => { const d = document.createElement('div'); d.id = 'pickerShot'; d.style.cssText = 'position:fixed;inset:0;z-index:99;background:#0b0e14;color:#dfe6f3;font:12px/1.35 system-ui,sans-serif;padding:14px 18px;column-count:3;column-gap:22px'; d.innerHTML = '<b style="font-size:14px">Scene picker (the select\\'s option tree, grouped by scenario family)</b>' + [...document.getElementById('labPreset').querySelectorAll('optgroup')].map((g) => '<div style="break-inside:avoid;margin-top:10px"><div style="color:#9fb0cf;font-weight:700;text-transform:uppercase;letter-spacing:.06em;font-size:11px">' + g.label + '</div>' + [...g.children].map((o) => '<div style="padding-left:8px' + (o.selected ? ';background:#2b4f8a' : '') + '">' + o.textContent + '</div>').join('') + '</div>').join(''); document.body.appendChild(d); return true; })()`);
    await png('picker-1280x720');
    record.picker = await evaluate(`[...document.getElementById('labPreset').querySelectorAll('optgroup')].map((g) => ({ group: g.label, options: [...g.children].map((o) => o.value + ' | ' + o.textContent) }))`);
    await evaluate(`document.getElementById('pickerShot').remove(), true`);
    console.log(await setParam('speed', 70));
    await settle('S02 speed');
    const s1 = await state();
    record.S02_speed70 = s1;
    if (!s1.noReload) problems.push('S02 speed: the page reloaded');
    if (!s1.modified || !(s1.bodies?.[0]?.speed > 19 && s1.bodies?.[0]?.speed < 20)) problems.push(`S02 speed: not re-placed at 70 km/h (${JSON.stringify(s1.bodies)})`);
    if (!/Modified from S02/.test(s1.head ?? '')) problems.push('S02 speed: the results panel does not say the reference criteria are not applied');
    await advance(240);
    await draw();
    await png('S02-speed70-1280x720-full');
    await png('S02-speed70-1280x720-scene', await sceneClip());
    console.log(await setParam('car', 'sedan'));
    await settle('S02 car');
    const s2 = await state();
    record.S02_sedan = s2;
    if (s2.bodies?.[0]?.design !== 'sedan') problems.push(`S02 car: not re-spawned as the sedan (${JSON.stringify(s2.bodies)})`);
    await advance(260);
    await draw();
    await png('S02-sedan70-1280x720-full');
    await png('S02-sedan70-800x640-scene', await (async () => { await size(800, 640); await draw(); return sceneClip(); })());
    await size(1280, 720);
    // Reset: the preset's values and criteria again.
    await evaluate(`document.querySelector('.lab-params .reset').click(), true`);
    await settle('S02 reset');
    const s3 = await state();
    record.S02_reset = s3;
    if (s3.modified || s3.bodies?.[0]?.design !== 'suv_4x4') problems.push(`S02 reset: not the preset again (${JSON.stringify(s3.bodies)})`);
  }

  if (ONLY.includes('wall56')) {
    // A lab preset on the page's car: its speed changed at run time.
    await size(1280, 720);
    await open('preset=wall56');
    const w0 = await state();
    record.wall56_start = w0;
    await advance(200);
    await draw();
    await png('wall56-1280x720-scene', await sceneClip());
    console.log(await setParam('overlap', 0.4));
    await settle('wall56 overlap');
    const w1 = await state();
    record.wall56_overlap = w1;
    if (!w1.noReload) problems.push('wall56: the page reloaded');
    if (JSON.stringify(w1.boxes) === JSON.stringify(w0.boxes)) problems.push('wall56 overlap: the wall did not move');
    if (!w1.modified) problems.push('wall56: not marked modified');
    await advance(900);
    await draw();
    for (const [w, h] of SIZES) { await size(w, h); await draw(); await png(`wall56-overlap40-${w}x${h}-full`); }
  }

  if (ONLY.includes('pole')) {
    // A family opened directly (the generated sedan into a pole), its offset changed.
    await size(1280, 720);
    await open('crashFamily=pole');
    record.pole_start = await state();
    await advance(150);
    await draw();
    await png('family-pole-1280x720-full');
    console.log(await setParam('offset', 0.5));
    await settle('pole offset');
    const p1 = await state();
    record.pole_offset = p1;
    if (JSON.stringify(p1.boxes) === JSON.stringify(record.pole_start.boxes)) problems.push('pole offset: the pole did not move');
    await advance(500);
    for (const [w, h] of SIZES) { await size(w, h); await draw(); await png(`family-pole-offset-${w}x${h}-full`); await png(`family-pole-${w}x${h}-scene`, await sceneClip()); }
  }

  if (ONLY.includes('soft')) {
    // The soft (AVBD) page and the explicit plain view (the arena) still load as before: no
    // lab frame, their own panels.
    for (const q of ['softbody.html?scene=wall', 'crash-lab.html?xscene=arena&n=16']) {
      await size(1280, 720);
      await send('Page.navigate', { url: `${base}/${q}` });
      let st = {};
      for (let i = 0; i < 300 && !st.ready; i++) { await sleep(400); st = await evaluate(`({ ready: window.__ready === true, failed: window.__failed ?? null, lab: document.body.classList.contains('lab-on'), info: getComputedStyle(document.getElementById('info')).display })`).catch(() => ({})); if (st.failed) break; }
      record[`plain ${q}`] = st;
      if (!st.ready || st.failed || st.lab || st.info === 'none') problems.push(`${q}: ${JSON.stringify(st)}`);
      await png(`plain-${q.replace(/[^a-z0-9]+/gi, '_')}-1280x720`);
    }
  }

  writeFileSync(join(OUT, 'check.json'), JSON.stringify({ problems, errors, record }, null, 1));
  if (errors.length) { console.log(`console errors: ${JSON.stringify(errors.slice(0, 8))}`); code = 1; }
  if (problems.length) { console.log(`problems:\n  ${problems.join('\n  ')}`); code = 1; } else console.log('ok: every change re-placed the scene without a reload');
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
