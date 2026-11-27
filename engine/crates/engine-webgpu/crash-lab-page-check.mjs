// The crash lab's page split (crash-lab.html from softbody.html, 2026-11-27) in headless
// Chrome: the old explicit URLs redirect to crash-lab.html with their parameters (any order,
// the fragment kept, `solver` dropped) and run; crash-lab.html's first paint is the lab's
// layout (a still while it loads and one after, at each size); the plain views (the arena,
// ?lab=0) get their own panels; the soft page (softbody.html) still runs its scenes
// (the default yard, driven, and the traffic derby) with no console errors.
//
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- node crash-lab-page-check.mjs [--out DIR] [--only redirect,paint,plain,soft]
//
// Output: DIR (default engine/out/lab-page/): <case>.png and check.json. Exit 1 on a problem
// or a console error.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromeArgs } from './check-chrome.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', normalize(join(ROOT, '../../out/lab-page')));
const ONLY = opt('--only', 'redirect,paint,plain,soft').split(',');
mkdirSync(OUT, { recursive: true });
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
const profile = mkdtempSync(join(tmpdir(), 'lab-page-chrome-'));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width: 1280, height: 720 }), { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errors = [], network = [], problems = [], record = {};
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
  let id = 0, label = '';
  const pending = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    if (d.method === 'Runtime.exceptionThrown') errors.push(`${label}: ${d.params.exceptionDetails.exception?.description ?? d.params.exceptionDetails.text}`);
    if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error') errors.push(`${label}: ${d.params.args.map((a) => a.value ?? a.description).join(' ')}`);
    // Failed requests (a local-only optional file such as __mods/index.json, the favicon) are
    // listed apart; other browser log errors count as console errors.
    if (d.method === 'Log.entryAdded' && d.params.entry.level === 'error') (d.params.entry.source === 'network' ? network : errors).push(`${label}: ${d.params.entry.text} ${d.params.entry.url ?? ''}`);
  };
  const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  const size = (w, h) => send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });
  const png = async (name) => {
    const s = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, `${name}.png`), Buffer.from(s.result.data, 'base64'));
    console.log(`  ${name}.png`);
  };
  // Navigate and wait for the page's __ready (or its failure); its location and solver.
  const open = async (path, name) => {
    label = name;
    await send('Page.navigate', { url: `${base}/${path}` });
    const t0 = Date.now();
    for (;;) {
      await sleep(300);
      const st = await evaluate('({ ready: window.__ready === true, failed: window.__failed ?? null, err: document.getElementById("err")?.textContent || "" })').catch(() => ({}));
      if (st.failed || st.err) { problems.push(`${name}: page error ${st.failed || st.err}`); break; }
      if (st.ready) break;
      if (Date.now() - t0 > 120_000) { problems.push(`${name}: not ready in 120 s`); break; }
    }
    return evaluate(`({ path: location.pathname, search: location.search, hash: location.hash, solver: window.__solver ?? null, lab: document.body.classList.contains('lab-on'), info: !!document.getElementById('info') && getComputedStyle(document.getElementById('info')).display !== 'none' })`);
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Log.enable');

  if (ONLY.includes('redirect')) {
    // Old URL → expected search on crash-lab.html (the parameters in their order, `solver` dropped).
    const cases = [
      ['softbody.html?solver=explicit&crashRef=S05&run=0&contact=discrete', '?crashRef=S05&run=0&contact=discrete', ''],
      ['softbody.html?solver=explicit&preset=wall56&clean=1', '?preset=wall56&clean=1', ''],
      ['softbody.html?clean=1&preset=pole29&solver=explicit&paused=1#peak', '?clean=1&preset=pole29&paused=1', '#peak'],
      ['softbody.html?crashRef=S02', '?crashRef=S02', ''],
    ];
    await size(1280, 720);
    for (const [from, search, hash] of cases) {
      const name = `redirect ${from}`;
      const st = await open(from, name);
      // The lab's own URL edits (cam3 for a reference crash) come after the redirect: compare
      // the parameters the redirect carried.
      const got = new URLSearchParams(st.search), want = new URLSearchParams(search);
      const kept = [...want].every(([k, v]) => got.get(k) === v) && !got.has('solver');
      const steps = await evaluate('window.__lab ? window.__lab.advance(40).then((s) => s.step) : window.__step(40).then((r) => r.steps)').catch((e) => String(e));
      record[name] = { ...st, steps };
      if (st.path !== '/crash-lab.html' || !kept || st.hash !== hash || st.solver !== 'explicit' || !(steps >= 40)) problems.push(`${name}: ${JSON.stringify(record[name])}`);
      await png(`redirect-${from.replace(/^softbody\.html\?/, '').replace(/[^a-z0-9]+/gi, '_')}`);
    }
  }

  if (ONLY.includes('paint')) {
    // The first paint: the lab's frame while the engine loads, then the lab.
    for (const [w, h] of [[800, 640], [1280, 720]]) {
      await size(w, h);
      label = `paint ${w}x${h}`;
      await send('Page.navigate', { url: `${base}/crash-lab.html?preset=wall56` });
      let early = null;
      for (let i = 0; i < 60 && !early; i++) {
        await sleep(50);
        early = await evaluate(`document.readyState !== 'loading' && document.body ? ({ lab: document.body.classList.contains('lab-on'), info: !!document.getElementById('info'), hud: !!document.getElementById('hud'), status: getComputedStyle(document.getElementById('status')).display, ready: window.__ready === true }) : null`).catch(() => null);
      }
      record[`paint-loading-${w}x${h}`] = early;
      if (!early || !early.lab || early.info || early.hud || early.status !== 'none') problems.push(`paint ${w}x${h} while loading: ${JSON.stringify(early)}`);
      await png(`paint-loading-${w}x${h}`);
      const st = await open(`crash-lab.html?preset=wall56`, label);
      await evaluate('window.__lab.advance(400).then(() => true)');
      record[`paint-loaded-${w}x${h}`] = st;
      if (!st.lab || st.info) problems.push(`paint ${w}x${h} loaded: ${JSON.stringify(st)}`);
      await png(`paint-loaded-${w}x${h}`);
    }
  }

  if (ONLY.includes('plain')) {
    await size(1280, 720);
    for (const q of ['xscene=arena&n=16', 'xscene=wall&lab=0']) {
      const st = await open(`crash-lab.html?${q}`, `plain ${q}`);
      const r = await evaluate('window.__step(200)').catch((e) => ({ e: String(e) }));
      record[`plain ${q}`] = { ...st, steps: r.steps };
      if (st.lab || !st.info || st.solver !== 'explicit' || !(r.steps >= 200)) problems.push(`plain ${q}: ${JSON.stringify(record[`plain ${q}`])}`);
      await png(`plain-${q.replace(/[^a-z0-9]+/gi, '_')}`);
    }
  }

  if (ONLY.includes('soft')) {
    // The soft solver's page: its default scene (the yard, driven) and the traffic derby.
    await size(1280, 720);
    for (const [q, name] of [['', 'default'], ['scene=traffic&cars=10', 'traffic']]) {
      const st = await open(`softbody.html${q ? `?${q}` : ''}`, `soft ${name}`);
      const a = await evaluate('window.__step(60)').catch((e) => ({ e: String(e) }));
      const b = await evaluate(`window.__step(240, { forward: 1, steer: 0.3, brake: 0, handbrake: 0 })`).catch((e) => ({ e: String(e) }));
      const moved = a.player && b.player ? Math.hypot(...[0, 1, 2].map((i) => b.player.p[i] - a.player.p[i])) : 0;
      const link = await evaluate(`document.getElementById('solverBtn')?.getAttribute('href')`);
      record[`soft ${name}`] = { ...st, steps: b.steps, moved, softBodies: b.damage?.length ?? 0, link };
      if (st.path !== '/softbody.html' || st.solver !== 'soft' || !(b.steps > a.steps) || !(moved > 0.5) || !link?.startsWith('crash-lab.html')) problems.push(`soft ${name}: ${JSON.stringify(record[`soft ${name}`])}`);
      await png(`soft-${name}`);
    }
  }

  writeFileSync(join(OUT, 'check.json'), JSON.stringify({ problems, errors, network, record }, null, 1));
  if (network.length) console.log(`failed requests (not counted): ${[...new Set(network.map((x) => x.replace(/^[^:]*: /, '')))].join('; ')}`);
  console.log(JSON.stringify(record, null, 1));
  if (errors.length) { console.log(`console errors: ${JSON.stringify(errors.slice(0, 8))}`); code = 1; } else console.log('console errors: none');
  if (problems.length) { console.log(`problems:\n  ${problems.join('\n  ')}`); code = 1; } else console.log('ok');
  console.log(`→ ${OUT}`);
  ws.close();
} catch (e) {
  console.error(String(e?.stack ?? e));
  code = 1;
} finally {
  chrome.kill();
  server.close();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* in use */ }
}
process.exit(code);
