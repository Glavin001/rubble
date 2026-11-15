// Records and probes the vehicle lab (vehicle-lab.html) from headless Chrome.
//
// Record (frame-locked, `?record=1`: the page steps the World itself, one video frame of
// 1/60 s at a time; slow motion as the scenario's `slow` events say): a clip and a stats
// JSON per run, stills at a few times of each.
//
//   node vehicle-lab-record.mjs [--out dir] [--query 'car=sedan'] name[:query] …
//     name: a scenario of vehicle-lab-scripts/ (shoot-box, compactor-top, mountain, …); or
//     with a query, the lab's own state for --secs seconds (`crash-offset:mode=crash&kmh=56`).
//   node vehicle-lab-record.mjs --probe [--secs 10] name[:query] …
//     the page's own real-time loop: World steps/s, FPS and the GPU split every second,
//     then the lab's table; console errors fail the run.
//   node vehicle-lab-record.mjs --restarts name:query
//     times the in-place restart (a slider change to running again) and an in-place
//     rebuild, and checks the page did not reload.
//
//   node vehicle-lab-record.mjs --qa [--secs 45] [--stills 4,15,30] [--beams-at 30] name:query …
//     QA of a page (local or deployed): time to ready, the adapter (fails on a software
//     one), a line a second, PNG stills at those simulated times (the beams view once),
//     then the perf report, the lab's table and stats (penetration) as files; console
//     errors and a failure banner fail the run.
//   --base URL: a deployed page instead of the local server (--auth URL first, e.g. a
//   Vercel share link, which sets the session cookie); --profile DIR keeps Chrome's
//   profile (its GPU caches) between runs; CHROME_FLAGS adds Chrome flags.
// Needs Chrome and ffmpeg (record), the World wasm (CONTRIBUTING.md). Run it as the
// machine's one GPU job: engine/scripts/with-gpu-lock.py --limit 900 -- node …
// A hard time limit (--limit, minutes) bounds every run.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync, realpathSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', join(homedir(), 'Development', 'rubble-artifacts', 'vehicle-lab'));
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const PROBE = args.includes('--probe'), RESTARTS = args.includes('--restarts'), QA = args.includes('--qa');
const EXTRA = opt('--query', '');
const FLAGS = ['--probe', '--restarts', '--segments', '--qa'];
const runs = args.filter((a, i) => !a.startsWith('--') && (!args[i - 1]?.startsWith('--') || FLAGS.includes(args[i - 1])));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(join(OUT, 'stills'), { recursive: true });
const LIMIT_MIN = Number(opt('--limit', '15'));
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();

// A static server over this directory (the vehicle folders in __mods may be symlinks).
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  if (req.method === 'HEAD') { res.end(); return; }
  res.end(readFileSync(realpathSync(path)));
});
const BASE = opt('--base', null);
if (!BASE) await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = BASE ? BASE.replace(/\/$/, '') : `http://127.0.0.1:${server.address().port}`;

const keepProfile = opt('--profile', null);
if (keepProfile) mkdirSync(keepProfile, { recursive: true });
const profile = keepProfile || mkdtempSync(join(tmpdir(), 'vehicle-lab-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  ...(process.env.CHROME_FLAGS || '').split(' ').filter(Boolean), 'about:blank',
], { stdio: process.env.CHROME_LOG ? ['ignore', 'inherit', 'inherit'] : 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} if (!keepProfile) try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
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
const pageErrors = [];
let navigations = 0;
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Page.frameNavigated' && !m.params.frame.parentId) navigations++;
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 300_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
if (opt('--auth', null)) {
  // A share link (Vercel Authentication): its redirect sets the session cookie.
  await send('Page.navigate', { url: opt('--auth') });
  await new Promise((r) => setTimeout(r, 4000));
}

const open = async (query) => {
  await send('Page.navigate', { url: `${base}/vehicle-lab.html?${query}${EXTRA ? '&' + EXTRA : ''}` });
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const err = await evaluate('window.__failed || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}\n${pageErrors.join('\n')}`);
    if (await evaluate('window.__ready === true').catch(() => false)) break;
    if (Date.now() - t0 > Number(opt('--startLimit', '240')) * 1000) throw new Error(`the page did not start: ${await evaluate('document.getElementById("status")?.textContent').catch(() => '?')}\n${pageErrors.join('\n')}`);
  }
  return (Date.now() - t0) / 1000;
};
const grabTo = async (file, quality = 90) => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality });
  writeFileSync(file, Buffer.from(s.result.data, 'base64'));
};
// name[:query]: without a query the name is a scenario.
const parse = (run) => { const [name, ...q] = run.split(':'); const query = q.join(':'); return { name, query: query || `script=${name}`, scripted: !query || /(^|&)script=/.test(query) }; };
const failOnErrors = (name) => { if (pageErrors.length) { console.error(`${name}: console errors:\n${pageErrors.slice(0, 10).join('\n')}`); process.exit(1); } };

if (RESTARTS) {
  for (const run of runs.length ? runs : ['crash:mode=crash']) {
    const { name, query } = parse(run);
    const built = await open(query);
    const nav0 = navigations;
    console.log(`${name}: first load ${built.toFixed(1)} s; ${JSON.stringify(await evaluate('(({ firstLoadMs, carLoadMs, carBytes }) => ({ firstLoadMs: Math.round(firstLoadMs), carLoadMs: Math.round(carLoadMs), carBytes }))(window.__state())'))}`);
    await new Promise((r) => setTimeout(r, 1500));
    const times = [];
    for (const kmh of [30, 64, 40, 56, 80]) {
      await evaluate(`(async () => { await window.__set({ mode: 'crash', kmh: ${kmh}, overlap: ${(0.2 + kmh / 200).toFixed(2)} }); await window.__restart(); })()`);
      await new Promise((r) => setTimeout(r, 1200));
      const s = await evaluate('(({ restartMs, againMs }) => ({ restartMs, againMs }))(window.__state())');
      const st = await evaluate('(({ targetKmh, arrivalKmh, car }) => ({ targetKmh, arrivalKmh, broken: car.beams.broken }))(window.__stats())');
      times.push(s);
      console.log(`  crash at ${kmh} km/h: restart ${s.restartMs.toFixed(1)} ms, running again after ${s.againMs.toFixed(0)} ms; arrived at ${st.arrivalKmh?.toFixed(1)} km/h, ${st.broken} beams broken`);
    }
    const t0 = Date.now();
    await evaluate(`(async () => { await window.__set({ surface: !document.getElementById('surface').checked }); await window.__build(); })()`);
    console.log(`  in-place rebuild (surface contact toggled): ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    await evaluate(`(async () => { await window.__set({ mode: 'mountain' }); await window.__restart(); })()`);
    await new Promise((r) => setTimeout(r, 1500));
    console.log(`  flat → mountain in place: ${JSON.stringify(await evaluate('(({ restartMs, againMs }) => ({ restartMs, againMs }))(window.__state())'))}; fallen ${(await evaluate('window.__stats().mountain.fallen')).toFixed(1)} m`);
    console.log(`  page loads since the first: ${navigations - nav0} (must be 0)`);
    if (navigations !== nav0) process.exit(1);
    failOnErrors(name);
  }
  process.exit(0);
}

if (QA) {
  const grabPng = async (file) => {
    const s = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(file, Buffer.from(s.result.data, 'base64'));
    console.log(`  still ${file}`);
  };
  const waitSim = async (t, limitS) => {
    const t0 = Date.now();
    for (;;) {
      const now = await evaluate('window.__state().t').catch(() => 0);
      if (now >= t) return now;
      if (Date.now() - t0 > limitS * 1000) return now;
      await new Promise((r) => setTimeout(r, 200));
    }
  };
  let failedRuns = 0;
  // The adapter first (fails fast on a software one): on the site's root page.
  await send('Page.navigate', { url: `${base}/` });
  await new Promise((r) => setTimeout(r, 3000));
  const adapter0 = await evaluate(`(async () => { if (!navigator.gpu) return null; const a = await navigator.gpu.requestAdapter(); const i = a?.info || {}; return a && { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: !!a.isFallbackAdapter }; })()`).catch((e) => ({ error: e.message }));
  console.log(`WebGPU adapter: ${JSON.stringify(adapter0)}`);
  if (!adapter0 || adapter0.error || adapter0.fallback || /swiftshader|llvmpipe|lavapipe/i.test(`${adapter0.vendor} ${adapter0.description} ${adapter0.device} ${adapter0.architecture}`)) {
    console.log('FAILED: no hardware WebGPU adapter'); process.exit(3);
  }
  for (const run of runs) {
    const { name, query } = parse(run);
    pageErrors.length = 0;
    const tLoad = Date.now();
    let ready;
    try { ready = await open(query); } catch (e) { console.log(`${name}: FAILED to start after ${((Date.now() - tLoad) / 1000).toFixed(1)} s: ${e.message}`); failedRuns++; continue; }
    const adapter = await evaluate(`(async () => { const a = await navigator.gpu.requestAdapter(); const i = a?.info || {}; return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description, fallback: !!a?.isFallbackAdapter }; })()`);
    console.log(`${name}: ready in ${ready.toFixed(1)} s; adapter ${JSON.stringify(adapter)}`);
    if (!adapter || adapter.fallback || /swiftshader|llvmpipe|lavapipe/i.test(`${adapter.vendor} ${adapter.description} ${adapter.device} ${adapter.architecture}`)) {
      console.log(`${name}: FAILED: software or no WebGPU adapter`); process.exit(3);
    }
    const secs = Number(opt('--secs', '45'));
    const stills = (opt('--stills', '') || '').split(',').filter(Boolean).map(Number).sort((a, b) => a - b);
    const beamsAt = opt('--beams-at', null) == null ? null : Number(opt('--beams-at'));
    const shots = [...stills.map((t) => ({ t, beams: false })), ...(beamsAt != null ? [{ t: beamsAt, beams: true }] : [])].sort((a, b) => a.t - b.t);
    const wall0 = Date.now();
    let lastLog = 0;
    for (const shot of shots) {
      const at = await waitSim(shot.t, secs * 4);
      if (shot.beams) { await evaluate(`(async () => { await window.__set({ beams: true }); document.getElementById('beams').checked = true; document.getElementById('beams').dispatchEvent(new Event('change')); })()`); await new Promise((r) => setTimeout(r, 600)); }
      await grabPng(join(OUT, 'stills', `${name}-t${String(shot.t).padStart(2, '0')}${shot.beams ? '-beams' : ''}.png`));
      if (shot.beams) await evaluate(`(async () => { document.getElementById('beams').checked = false; document.getElementById('beams').dispatchEvent(new Event('change')); })()`);
      console.log(`    (sim ${Number(at).toFixed(1)} s)`);
    }
    // The rest of the run, a line a second.
    while ((Date.now() - wall0) / 1000 < secs) {
      await new Promise((r) => setTimeout(r, 1000));
      const s = Math.round((Date.now() - wall0) / 1000);
      if (s - lastLog >= 5) {
        lastLog = s;
        try { console.log(`  ${s}s ${await evaluate(`JSON.stringify((({ t, simRate, fps, frameMs, refused, gpu }) => ({ t: +t.toFixed(2), simRate: +simRate.toFixed(1), fps: +fps.toFixed(0), frameP50: +frameMs.p50.toFixed(1), refused, gpu: gpu && gpu.total.length ? { total: +gpu.split.total.toFixed(2), rigid: +gpu.split.rigid.toFixed(2), vehicle: +gpu.split.vehicle.toFixed(2) } : null }))(window.__state()))`)}`); } catch (e) { console.log(`  ${s}s (no reading: ${String(e.message).slice(0, 80)})`); }
      }
    }
    await grabPng(join(OUT, 'stills', `${name}-end.png`));
    const state = await evaluate('window.__state()');
    const stats = await evaluate('window.__stats()');
    const pen = stats?.car?.penetration;
    writeFileSync(join(OUT, `${name}.perf.txt`), await evaluate('window.__perfReport()'));
    writeFileSync(join(OUT, `${name}.table.txt`), await evaluate('window.__table()'));
    writeFileSync(join(OUT, `${name}.stats.json`), JSON.stringify({ name, query, readySeconds: ready, adapter, state, stats }, null, 1));
    const banner = await evaluate('document.getElementById("err")?.textContent || ""');
    const failed = await evaluate('window.__failed || ""');
    console.log(`${name}: sim ${state.t.toFixed(1)} s in ${((Date.now() - wall0) / 1000).toFixed(0)} s (sim rate ${state.simRate.toFixed(1)} steps/s, fps ${state.fps.toFixed(0)}); nodes inside rigid bodies ${pen?.nodesInside}; banner "${banner}"; failed "${failed}"; console errors ${pageErrors.length}`);
    for (const e of pageErrors.slice(0, 10)) console.log(`  console error: ${String(e).slice(0, 300)}`);
    if (pageErrors.length || failed || (pen && pen.nodesInside > 0)) failedRuns++;
  }
  process.exit(failedRuns ? 1 : 0);
}

if (PROBE) {
  for (const run of runs.length ? runs : ['shoot-sphere']) {
    const { name, query } = parse(run);
    const built = await open(`${query}&timing=phases`);
    console.log(`${name}: ready in ${built.toFixed(1)} s`);
    if (opt('--eval', null)) console.log(await evaluate(readFileSync(opt('--eval'), 'utf8')));
    for (let s = 1; s <= Number(opt('--secs', '10')); s++) {
      await new Promise((r) => setTimeout(r, 1000));
      try { console.log(`  ${s}s ${await evaluate(`JSON.stringify((({ t, simRate, fps, bodies, readsSkipped, gpu, frameMs }) => ({ t: +t.toFixed(2), simRate: +simRate.toFixed(1), fps: +fps.toFixed(0), frameP50: +frameMs.p50.toFixed(1), frameP95: +frameMs.p95.toFixed(1), bodies, readsSkipped,
        gpu: gpu && gpu.total.length ? { total: +gpu.split.total.toFixed(2), rigid: +gpu.split.rigid.toFixed(2), vehicle: +gpu.split.vehicle.toFixed(2) } : null }))(window.__state()))`)}`); } catch (e) { console.log(`  ${s}s (no reading: ${String(e.message).slice(0, 80)})`); }
      if (args.includes('--segments') && s % 5 === 0) console.log(`     segments ${await evaluate(`JSON.stringify(Object.fromEntries(Object.entries(window.__gpu().segments).map(([k, v]) => [k, +v.toFixed(2)]).filter(([, v]) => v >= 0.05)))`)}`);
    }
    await grabTo(join(OUT, 'stills', `probe-${name}.jpg`), 85);
    if (opt('--eval', null)) console.log(await evaluate('(window.__abLog || []).join("\\n")'));
    console.log(await evaluate('window.__table()'));
    failOnErrors(name);
  }
  process.exit(0);
}

const FPS = 60;
for (const run of runs.length ? runs : ['shoot-box']) {
  const { name, query, scripted } = parse(run);
  mkdirSync(join(OUT, 'frames'), { recursive: true });
  const frameDir = mkdtempSync(join(OUT, 'frames', `${name}-`));
  let frameNo = 0;
  const rec = (expr) => evaluate(`(async () => { const R = window.__rec; return ${expr}; })()`);
  await open(`${query}&record=1&clean=1`);
  if (opt('--eval', null)) await evaluate(readFileSync(opt('--eval'), 'utf8'));
  const secs = Number(opt('--secs', '0'));
  // Stills: a few video times through the clip (--stills '1.2,2.5,…' video seconds).
  const stillAt = (opt('--stills', '') || '').split(',').filter(Boolean).map(Number);
  let st = null, still = 0;
  const maxFrames = Number(opt('--maxSecs', '150')) * FPS;
  while (frameNo < maxFrames) {
    st = await rec('R.frame(60)');
    if (st.failed) throw new Error(`${name}: ${st.failed}`);
    await grabTo(join(frameDir, `f${String(frameNo++).padStart(6, '0')}.jpg`));
    const video = frameNo / FPS;
    if (stillAt.length ? (still < stillAt.length && video >= stillAt[still]) : frameNo % (3 * FPS) === 0) { await grabTo(join(OUT, 'stills', `${name}-${String(still++).padStart(2, '0')}.jpg`), 85); }
    if (secs ? st.t >= secs : scripted && st.done) break;
  }
  await grabTo(join(OUT, 'stills', `${name}-end.jpg`), 85);
  const stats = await evaluate('window.__stats()');
  writeFileSync(join(OUT, `${name}.stats.json`), JSON.stringify({ name, query, videoSeconds: frameNo / FPS, simulatedSeconds: st.t, stats }, null, 1));
  writeFileSync(join(OUT, `${name}.table.txt`), await evaluate('window.__table()'));
  const file = join(OUT, `vehicle-lab-${name}.mp4`);
  const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: 'inherit' });
  rmSync(frameDir, { recursive: true, force: true });
  if (enc.status !== 0) throw new Error('ffmpeg failed');
  console.log(`${name}: ${(frameNo / FPS).toFixed(1)} s of video for ${st.t.toFixed(1)} s simulated → ${file}`);
  failOnErrors(name);
}
process.exit(0);
