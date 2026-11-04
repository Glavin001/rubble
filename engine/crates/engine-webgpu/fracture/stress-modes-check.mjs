#!/usr/bin/env node
// The fracture lab's stress models in Chrome WebGPU, end to end (load_path, impact_local and
// blast on World; stress, blast_parity, on its own engine): the built site
// (scripts/build-web.sh, engine/web-dist) served locally, each mode loaded on a scene,
// stepped at rest, then shot through the page's own click path and stepped on. A mode
// fails on a WebGPU validation message (a pipeline Chrome rejects stalls the frame
// silently), a frame counter that stops, a solver error, or a shot that breaks nothing.
// A screenshot per mode goes to --out.
//
//   bash engine/scripts/build-web.sh
//   node --experimental-websocket stress-modes-check.mjs [mode ...] [--scene cuboid] [--out DIR]
//
// Modes: stress load_path impact_local blast (default: all). Needs Chrome (CHROME env).
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..', '..', '..', 'web-dist');
const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
const OUT = opt('--out', join(HERE, '..', 'recordings', 'stress-modes'));
const SCENE = opt('--scene', 'cuboid');
const MODES = args.filter((a, i) => !a.startsWith('--') && !['--out', '--scene'].includes(args[i - 1]));
const modes = MODES.length ? MODES : ['stress', 'load_path', 'impact_local', 'blast'];
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
const profile = mkdtempSync(join(tmpdir(), 'stress-modes-chrome-'));
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
// Every console message and exception since the last `drain`.
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

// Select a scene and mode as the controls do, then wait for the stress world.
const load = (mode) => evaluate(`(async () => {
  const set = (id, v) => { const s = document.getElementById(id); s.value = v; s.dispatchEvent(new Event('change', { bubbles: true })); };
  const d = window.__FRACTURE_DEMO__;
  if (document.getElementById('scene').value !== ${JSON.stringify(SCENE)}) { set('scene', ${JSON.stringify(SCENE)}); await new Promise((r) => setTimeout(r, 300)); }
  set('bond-mode', ${JSON.stringify(mode)});
  await new Promise((r) => setTimeout(r, 300));
  for (let i = 0; i < 600 && !d.stressReady; i++) await new Promise((r) => setTimeout(r, 100));
  // Start the simulation as the Start button does (a stress world loads paused).
  const release = document.getElementById('release');
  if (d.stressReady && release.textContent !== 'Pause') release.click();
  return { ready: d.stressReady, status: d.status };
})()`);
// One display frame at a time, yielding to the browser between them as its own loop does (the
// fracture world's readbacks resolve there).
const advance = (frames) => evaluate(`(async () => { const d = window.__FRACTURE_DEMO__; let st; for (let i = 0; i < ${frames}; i++) { st = await d.stressAdvance(1000 / 60); await new Promise((r) => requestAnimationFrame(r)); } return st; })()`);
// The page's click path: a press and release at the canvas centre.
const clickCentre = async () => {
  // The canvas in view: World's (#view), or blast_parity's own (#stress-view).
  const r = await evaluate(`(() => { const c = [document.getElementById('view'), document.getElementById('stress-view')].find((v) => !v.hidden); const b = c.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()`);
  for (const type of ['mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type, x: r[0], y: r[1], button: 'left', clickCount: 1 });
};
const webgpuProblem = (m) => /Invalid|uniform control flow|validation|GPUValidationError|device lost|exception/i.test(m);

let failed = 0;
for (const mode of modes) {
  messages = [];
  const problems = [];
  const loaded = await load(mode);
  if (!loaded.ready) problems.push(`did not load: ${loaded.status}`);
  const rest = loaded.ready ? await advance(120) : null;
  if (rest) {
    if (rest.errors) problems.push(`${rest.errors} solver errors at rest`);
    if (!(rest.frames >= 100)) problems.push(`stalled at rest: ${rest.frames} frames after 120`);
    await clickCentre();
    const shot = await advance(240);
    if (shot.errors) problems.push(`${shot.errors} solver errors`);
    if (!(shot.frames >= rest.frames + 200)) problems.push(`stalled: ${shot.frames} frames, ${rest.frames} before the shot`);
    if (!(shot.shots >= 1)) problems.push('no shot fired');
    if (!(shot.brokenBonds > 0)) problems.push('the shot broke nothing');
    const bad = messages.filter(webgpuProblem);
    if (bad.length) problems.push(`${bad.length} WebGPU/page errors, first: ${bad[0].slice(0, 600)}`);
    if (bad.length && process.env.VERBOSE) for (const m of messages) console.log(`          | ${m.slice(0, 400)}`);
    const shotPng = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, `${SCENE}-${mode}.png`), Buffer.from(shotPng.result.data, 'base64'));
    console.log(`${problems.length ? 'FAIL' : 'ok  '}  ${mode}: at rest ${rest.brokenBonds} broken; after the shot ${shot.brokenBonds} of ${shot.bonds} bonds broken, ${shot.bodies} bodies, ${shot.frames} frames`);
    // Throughput: frames requested back to back (no display wait), wall time per frame that
    // ran; at most a few frames queue ahead of the GPU, so this is the GPU's time per frame
    // when it is the bottleneck. Beside it, the timestamps' figure.
    const through = await evaluate(`(async () => { const d = window.__FRACTURE_DEMO__; const f0 = (await d.stressAdvance(0)).frames; const t0 = performance.now(); let st; for (let i = 0; i < 120; i++) { st = await d.stressAdvance(1000 / 60); await new Promise((r) => setTimeout(r, 0)); } return { ms: (performance.now() - t0) / Math.max(1, st.frames - f0), frames: st.frames - f0, gpu: st.gpuStepMs, draw: st.gpuDrawMs } })()`);
    console.log(`        throughput: ${through.ms.toFixed(2)} ms wall per frame over ${through.frames} frames; timestamps: ${through.gpu?.toFixed(3)} ms physics, ${through.draw?.toFixed(3)} ms draw`);
    // The HUD's timing readout (GPU physics and draw time, display rate), after the page's
    // own display loop has run a few seconds (the HUD counts that loop's frames).
    const hud = await evaluate(`(async () => { window.__FRACTURE_DEMO__.stressRelease(); await new Promise((r) => setTimeout(r, 3500)); return ['tps','step-ms','fps','frame-ms','sim-speed','timing'].map((id) => document.getElementById(id).textContent).join(' | ') })()`);
    console.log(`        HUD: ${hud}`);
  } else {
    console.log(`FAIL  ${mode}`);
  }
  for (const p of problems) console.log(`        ${p}`);
  if (problems.length) failed++;
}
console.log(`${modes.length - failed} of ${modes.length} modes pass; screenshots in ${OUT}`);
cleanup();
process.exit(failed ? 1 : 0);
