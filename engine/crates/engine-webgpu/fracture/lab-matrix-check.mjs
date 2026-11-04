#!/usr/bin/env node
// The fracture lab's matrix in Chrome WebGPU (engine/docs/lab-world-plan.md, Checks): every
// scene × bond model × action on the built site, through the page's own controls. Each cell:
//   - no WebGPU validation or page error;
//   - frames advance;
//   - the action fires (its shots counted);
//   - a breakable bond model breaks something (under at least one of the scene's actions: the
//     default ball against a strong scene may break nothing, which a cell notes);
//   - the HUD's GPU time per physics step is present and plausible (where the device has
//     timestamp queries).
// A combination that cannot work must fail with its reason (the page's status); the check
// lists every such cell. A screenshot per cell goes to --out.
//
//   bash engine/scripts/build-web.sh
//   node --experimental-websocket lab-matrix-check.mjs [--scenes a,b] [--bonds a,b] [--actions a,b] [--out DIR]
//
// Scenes: the Scene menu's keys (default cuboid,arch,house,cuboidGrid,crush-wall,glass-tower,
// vc-towers; vc-towers needs the vibe-city packs: engine/scenes/vibe-city, bun run export.ts).
// Bonds: none,d6,threshold,load_path,impact_local,blast. Actions: click,row,volley,topple,meteor
// (each only on the scenes that have it). Needs Chrome (CHROME env).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
const ROOT = opt('--site', join(HERE, '..', '..', '..', 'web-dist'));
const OUT = opt('--out', join(HERE, '..', 'recordings', 'lab-matrix'));
const list = (name, fallback) => opt(name, fallback).split(',').filter(Boolean);
const SCENES = list('--scenes', 'cuboid,arch,house,cuboidGrid,crush-wall,glass-tower,vc-towers');
const BONDS = list('--bonds', 'none,d6,threshold,load_path,impact_local,blast');
const ACTIONS = list('--actions', 'click,row,volley,topple,meteor,pack');
const BREAKABLE = new Set(['d6', 'threshold', 'load_path', 'impact_local', 'blast']);
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
if (!existsSync(join(ROOT, 'fracture', 'index.html'))) { console.error(`no built site at ${ROOT}: run engine/scripts/build-web.sh`); process.exit(2); }
mkdirSync(OUT, { recursive: true });

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css', '.glb': 'model/gltf-binary', '.wgsl': 'text/plain' };
const server = createServer((req, res) => {
  let path = normalize(join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, 'index.html');
  if (!path.startsWith(ROOT) || !existsSync(path)) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' }).end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const [W, H] = [1280, 720];
const profile = mkdtempSync(join(tmpdir(), 'lab-matrix-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'], { stdio: 'ignore' });
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
let messages = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') messages.push(`${m.params.type}: ${m.params.args.map((a) => a.value ?? a.description ?? '').join(' ')}`);
  if (m.method === 'Log.entryAdded') messages.push(`${m.params.entry.level}: ${m.params.entry.text}`);
  if (m.method === 'Runtime.exceptionThrown') messages.push(`exception: ${m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text}`);
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Log.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Page.navigate', { url: `${base}/fracture/` });
await evaluate(`(async () => { for (let i = 0; i < 600 && !window.__FRACTURE_DEMO__; i++) await new Promise((r) => setTimeout(r, 100)); })()`);

// Select a scene and bond model as the controls do; wait for the load to finish (ready, or
// the status showing an error).
const load = (scene, bonds) => evaluate(`(async () => {
  const set = (id, v) => { const s = document.getElementById(id); s.value = v; s.dispatchEvent(new Event('change', { bubbles: true })); };
  const d = window.__FRACTURE_DEMO__, status = document.getElementById('status');
  if (!document.getElementById('scene').querySelector('option[value="${scene}"]')) return { missing: true };
  status.classList.remove('error');
  if (document.getElementById('scene').value !== ${JSON.stringify(scene)}) { set('scene', ${JSON.stringify(scene)}); await new Promise((r) => setTimeout(r, 300)); }
  if (document.getElementById('bond-mode').value !== ${JSON.stringify(bonds)}) set('bond-mode', ${JSON.stringify(bonds)});
  await new Promise((r) => setTimeout(r, 300));
  for (let i = 0; i < 1800 && !d.stressReady && !status.classList.contains('error'); i++) await new Promise((r) => setTimeout(r, 100));
  const release = document.getElementById('release');
  if (d.stressReady && release.textContent !== 'Pause') release.click();
  return { ready: d.stressReady && !status.classList.contains('error'), error: status.classList.contains('error') ? status.textContent : null, status: d.status,
    actions: { row: !document.getElementById('fire').hidden, volley: !document.getElementById('volley').hidden, topple: !document.getElementById('topple').hidden, meteor: !document.getElementById('meteor').hidden, pack: !document.getElementById('pack-shot-row').hidden, click: true } };
})()`);
const advance = (frames) => evaluate(`(async () => { const d = window.__FRACTURE_DEMO__; let st; for (let i = 0; i < ${frames}; i++) { st = await d.stressAdvance(1000 / 60); await new Promise((r) => requestAnimationFrame(r)); } return st; })()`);
const clickCentre = async () => {
  const r = await evaluate(`(() => { const c = document.getElementById('view'); const b = c.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
  for (const type of ['mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type, x: r[0], y: r[1], button: 'left', clickCount: 1 });
};
const BUTTON = { row: 'fire', volley: 'volley', topple: 'topple', meteor: 'meteor', pack: 'fire-pack-shot' };
// Scenes with nothing at the middle of the view (the arch's opening, the street between the
// towers, the ground beside the glass tower): a click there aims at the highest piece instead.
const AIM_TOP = new Set(['arch', 'glass-tower', 'vc-towers', 'vc-meteor']);
const webgpuProblem = (m) => /Invalid|uniform control flow|validation|GPUValidationError|device lost|exception/i.test(m);

const rows = [], errors = [];
let failed = 0, cells = 0;
for (const scene of SCENES) {
  for (const bonds of BONDS) {
    messages = [];
    const loaded = await load(scene, bonds);
    if (loaded.missing) { console.log(`skip  ${scene}: not in the Scene menu (not built here)`); break; }
    if (!loaded.ready) {
      // A combination that cannot work: it must say why.
      const reason = loaded.error ?? `did not load: ${loaded.status}`;
      errors.push(`${scene} × ${bonds}: ${reason}`);
      console.log(`error ${scene} × ${bonds}: ${reason}`);
      if (process.env.VERBOSE) {
        for (const m of messages.slice(0, 3)) console.log(`          | ${m.slice(0, 600)}`);
        console.log('          device:', JSON.stringify(await evaluate(`(async () => { const { labDevice } = await import('./lab-world.mjs'); const dev = await labDevice(); const d = window.__FRACTURE_DEMO__; return { sim: !!d.simulation, same: d.simulation ? d.simulation.device === dev : null, page: d.device === dev, lost: await Promise.race([dev.lost.then((i) => i.message), new Promise((r) => setTimeout(() => r('not lost'), 50))]) }; })()`)));
      }
      if (!loaded.error) failed++;
      continue;
    }
    let brokeAny = false;
    for (const action of ACTIONS.filter((a) => loaded.actions[a])) {
      cells++;
      const problems = [];
      messages = [];
      // Each action from the scene as placed (the bond models reset; the stress models reload).
      if (action !== ACTIONS.find((a) => loaded.actions[a])) {
        await evaluate(`window.__FRACTURE_DEMO__.reset()`);
        await evaluate(`(async () => { const d = window.__FRACTURE_DEMO__; for (let i = 0; i < 1800 && !d.stressReady; i++) await new Promise((r) => setTimeout(r, 100)); const b = document.getElementById('release'); if (b.textContent !== 'Pause') b.click(); })()`);
      }
      let rest, st, frames;
      try {
      rest = await advance(30);
      // The analytic strike holes a wall only above its ballistic limit: its ball at 8× speed.
      await evaluate(`(() => { const s = document.getElementById('ball-speed'); s.value = '${bonds === 'impact_local' ? 3 : 0}'; s.dispatchEvent(new Event('input', { bubbles: true })); })()`);
      // A click at the middle of the view, through the page's own input; where nothing is
      // there, the highest piece from in front, through the page's shot path.
      if (action === 'click' && AIM_TOP.has(scene)) await evaluate(`(async () => { const d = window.__FRACTURE_DEMO__, s = d.scene; const top = s.bodies.slice(1, s.projectileStart).reduce((a, b) => (b.p[1] > a.p[1] ? b : a)).p; d.shoot({ origin: [top[0], top[1], top[2] + 3 * Math.max(1, s.extent)], direction: [0, 0, -1] }); })()`);
      else if (action === 'click') await clickCentre();
      else await evaluate(`document.getElementById(${JSON.stringify(BUTTON[action])}).click()`);
      frames = action === 'volley' || action === 'topple' ? 600 : 240;
      st = await advance(frames);
      } catch (e) {
        // The page failed the step (a clear error: its status says why).
        const why = String(e.message ?? e).split('\n')[0];
        console.log(`FAIL  ${scene} × ${bonds} × ${action}: ${why}`);
        rows.push({ scene, bonds, action, ok: false, problems: [why] });
        failed++;
        await evaluate(`window.__FRACTURE_DEMO__.stressRelease()`).catch(() => {});
        continue;
      }
      if (!(st.frames >= rest.frames + frames * 0.9)) problems.push(`stalled: ${st.frames} frames, ${rest.frames} before`);
      if (!(st.shots >= 1)) problems.push('the action fired no shot');
      brokeAny ||= st.brokenBonds > 0;
      const note = BREAKABLE.has(bonds) && !(st.brokenBonds > 0) ? ' (broke nothing)' : '';
      if (st.gpuStepMs != null && !(st.gpuStepMs > 0 && st.gpuStepMs < 2000)) problems.push(`implausible GPU time per step: ${st.gpuStepMs}`);
      const bad = messages.filter(webgpuProblem);
      if (bad.length) problems.push(`${bad.length} WebGPU/page errors, first: ${bad[0].slice(0, 400)}`);
      const png = await send('Page.captureScreenshot', { format: 'png' });
      writeFileSync(join(OUT, `${scene}-${bonds}-${action}.png`), Buffer.from(png.result.data, 'base64'));
      const line = `${scene} × ${bonds} × ${action}: ${st.shots} shots · ${st.brokenBonds} of ${st.bonds} bonds broken${note} · ${st.bodies} bodies · ${st.frames} frames · ${st.gpuStepMs?.toFixed?.(2) ?? '—'} ms GPU/step`;
      console.log(`${problems.length ? 'FAIL' : 'ok  '}  ${line}`);
      for (const p of problems) console.log(`        ${p}`);
      if (problems.length && process.env.VERBOSE) for (const m of messages) console.log(`          | ${m.slice(0, 900)}`);
      rows.push({ scene, bonds, action, ok: !problems.length, ...st, problems });
      if (problems.length) failed++;
    }
    if (BREAKABLE.has(bonds) && rows.some((r) => r.scene === scene && r.bonds === bonds && r.frames !== undefined) && !brokeAny) {
      console.log(`FAIL  ${scene} × ${bonds}: no action broke anything`);
      rows.push({ scene, bonds, action: '(any)', ok: false, problems: ['no action broke anything'] });
      failed++;
    }
  }
}
writeFileSync(join(OUT, 'matrix.json'), JSON.stringify({ rows, errors }, null, 1));
console.log(`\n${cells - rows.filter((r) => !r.ok).length} of ${cells} cells pass; ${errors.length} combinations are errors:`);
for (const e of errors) console.log(`  ${e}`);
console.log(`screenshots and matrix.json in ${OUT}`);
cleanup();
process.exit(failed ? 1 : 0);
