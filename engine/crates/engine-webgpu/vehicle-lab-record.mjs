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
//   node vehicle-lab-record.mjs --probe --ab MASK [--abFrames 20] [--abSteps 34] [--abEvery 1]
//       [--abUntil 40] [--abOut dir] name:query
//     Chrome's bit-for-bit A/B of perf flag MASK on the page's own run (the flag's
//     `vehicle_lab --ab-switch perf=MASK` in the browser; WebWorld.explicitAb): every
//     --abEvery simulated seconds the World is held, the vehicles run --abFrames frames of
//     --abSteps steps from the state now with the flag on, then from the same state with it
//     off, every state word compared, the state put back and the World let go; until
//     --abUntil s. One line a sample; the first state the paths differ from is saved as a
//     native fixture (--abOut, default engine/out/chrome-ab/cars<N>_t<T>.fx) to replay with
//     `tools/explicit-2khz/fast-check.sh perf=MASK FILE`. Exit 1 when a sample differs.
//   node vehicle-lab-record.mjs --restarts name:query
//     times the in-place restart (a slider change to running again) and an in-place
//     rebuild, and checks the page did not reload.
//
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
const PROBE = args.includes('--probe'), RESTARTS = args.includes('--restarts');
const EXTRA = opt('--query', '');
const FLAGS = ['--probe', '--restarts', '--segments', '--locked', '--uncapped'];
const runs = args.filter((a, i) => !a.startsWith('--') && (!args[i - 1]?.startsWith('--') || FLAGS.includes(args[i - 1])));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(join(OUT, 'stills'), { recursive: true });
const LIMIT_MIN = Number(opt('--limit', '15'));
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();

// A static server over this directory (the vehicle folders in __mods may be symlinks).
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png' };
const AB = opt('--ab', null);
// (--abSaveAll TAG: the state saved at every sample, as cars<N>_t<T>TAG.fx; --ab 0: no
// A/B, the samples only: two page runs' states compared natively, fast-check.sh's --compare.)
const SAVE_ALL = args.includes('--abSaveAll'), TAG = SAVE_ALL ? opt('--abSaveAll', '') : '';
const AB_OUT = opt('--abOut', join(ROOT, '..', '..', 'out', 'chrome-ab'));
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  // (--ab: the page's saved state, POSTed to /__ab/<name>.fx.)
  if (AB != null && req.method === 'POST' && /^\/__ab\/[\w.-]+\.fx$/.test(p)) {
    const parts = [];
    req.on('data', (c) => parts.push(c));
    req.on('end', () => { mkdirSync(AB_OUT, { recursive: true }); writeFileSync(join(AB_OUT, p.slice(5)), Buffer.concat(parts)); res.writeHead(200).end(); });
    return;
  }
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  if (req.method === 'HEAD') { res.end(); return; }
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'vehicle-lab-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
  // (--uncapped: frames as fast as Chrome makes them, not headless's 60 a second: a display
  // faster than the World, as a 120 Hz one is.)
  ...(args.includes('--uncapped') ? ['--disable-frame-rate-limit', '--disable-gpu-vsync'] : []), 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} };
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

if (PROBE && AB != null) {
  const [frames, steps, every, until] = [opt('--abFrames', '20'), opt('--abSteps', '34'), opt('--abEvery', '1'), opt('--abUntil', '40')].map(Number);
  let bad = 0;
  for (const run of runs) {
    const parsed = parse(run);
    const name = parsed.name;
    // (Bit-exact results, the engine-wide setting off by default since 2026-11-19: the A/B
    // is bit for bit with it on, unless the query asks otherwise.)
    const query = /(^|&)bitExact=/.test(parsed.query) ? parsed.query : `${parsed.query}&bitExact=1`;
    // (--locked: the page frame-locked, `record=1`: each step's readback awaited before the
    // next, so the World's run is the same whatever the GPU's speed; otherwise the page's
    // own real-time loop, whose readbacks land a speed-dependent number of steps later.)
    const locked = args.includes('--locked');
    console.log(`${name}: ready in ${(await open(locked ? `${query}&record=1` : query)).toFixed(1)} s${locked ? ', frame-locked' : ''}; A/B of perf flag ${AB}, ${frames} frames of ${steps} steps every ${every} s to ${until} s`);
    let saved = null, samples = 0, differ = 0;
    for (let k = 1; k * every <= until + 1e-9; k++) {
      // (The page holds itself before its first step at or past the sample's time: two
      // runs are sampled at the same step.)
      if (locked) {
        while (await evaluate('window.__lab.time()') < k * every - 1e-9) {
          const st = await evaluate('window.__rec.frame(60)');
          if (st.failed) throw new Error(`${name}: ${st.failed}`);
        }
      } else await evaluate(`window.__hold(false, ${k * every})`);
      let lastT = -1, movedAt = Date.now(), stopped = null;
      while (!locked && !(await evaluate('window.__held()'))) {
        await new Promise((r) => setTimeout(r, 50));
        const t = await evaluate('window.__lab.time()');
        if (t !== lastT) { lastT = t; movedAt = Date.now(); }
        if (Date.now() - movedAt > 30_000) { stopped = t; break; }
      }
      if (stopped !== null) {
        console.log(`  the page stopped at t ${stopped.toFixed(2)}: ${await evaluate('JSON.stringify({ failed: window.__failed || null, lost: window.__lost || null, err: document.getElementById("err")?.textContent })')}`);
        bad = 1;
        break;
      }
      const r = await evaluate(`(async () => {
        {
          const t = window.__lab.time();
          const diff = ${Number(AB)} ? await window.__world.explicitAb(${Number(AB)}, ${frames}, ${steps}) : [];
          let file = null;
          if ((diff.length && ${saved === null}) || ${SAVE_ALL}) {
            file = 'cars' + window.__lab.vehicleCount() + '_t' + t.toFixed(2) + ${JSON.stringify(TAG)} + '.fx';
            await fetch('/__ab/' + file, { method: 'POST', body: await window.__lab.fixture(window.__world) });
          }
          return { t, diff, file, errors: window.__world.explicitErrors() };
        }
      })()`);
      samples++;
      if (r.file && r.diff.length && saved === null) saved = r.file;
      if (r.diff.length) differ++;
      console.log(`  t ${r.t.toFixed(2)}: ${r.diff.length ? `DIFFER (${r.diff.length} runs)` : 'bit for bit'}${r.file ? `; state saved: ${join(AB_OUT, r.file)}` : ''}${r.errors.length ? `; errors ${r.errors.join('; ')}` : ''}`);
      for (const l of r.diff.slice(0, 8)) console.log(`     ${l}`);
    }
    if (!locked) await evaluate('window.__hold(false)');
    const pose = await evaluate('Array.from(window.__lab.pose())');
    console.log(`${name}: ${samples} samples, ${differ} differ${saved ? `; first differing state ${join(AB_OUT, saved)}` : ''}; first car at (${pose.slice(0, 3).map((x) => x.toFixed(2)).join(', ')})`);
    if (differ) bad = 1;
    failOnErrors(name);
  }
  process.exit(bad);
}

if (PROBE) {
  for (const run of runs.length ? runs : ['shoot-sphere']) {
    const { name, query } = parse(run);
    const built = await open(`${query}&timing=phases`);
    console.log(`${name}: ready in ${built.toFixed(1)} s`);
    if (opt('--eval', null)) console.log(await evaluate(readFileSync(opt('--eval'), 'utf8')));
    for (let s = 1; s <= Number(opt('--secs', '10')); s++) {
      await new Promise((r) => setTimeout(r, 1000));
      try { console.log(`  ${s}s ${await evaluate(`JSON.stringify((({ t, simRate, fps, bodies, readsSkipped, camAge, gpu, frameMs }) => ({ t: +t.toFixed(2), simRate: +simRate.toFixed(1), fps: +fps.toFixed(0), frameP50: +frameMs.p50.toFixed(1), frameP95: +frameMs.p95.toFixed(1), bodies, readsSkipped,
        camAge: camAge && { p50: +camAge.p50.toFixed(3), max: +camAge.max.toFixed(3) },
        gpu: gpu && gpu.total.length ? { total: +gpu.split.total.toFixed(2), rigid: +gpu.split.rigid.toFixed(2), vehicle: +gpu.split.vehicle.toFixed(2) } : null }))(window.__state()))`)}`); } catch (e) { console.log(`  ${s}s (no reading: ${String(e.message).slice(0, 80)})`); }
      if (args.includes('--segments') && s % 5 === 0) console.log(`     segments ${await evaluate(`JSON.stringify(Object.fromEntries(Object.entries(window.__gpu().segments).map(([k, v]) => [k, +v.toFixed(2)]).filter(([, v]) => v >= 0.05)))`)}`);
    }
    await grabTo(join(OUT, 'stills', `probe-${name}.jpg`), 85);
    if (opt('--eval', null)) console.log(await evaluate('(window.__abLog || []).join("\\n")'));
    console.log(`  gpu run (t >= 0.5 s): ${await evaluate('JSON.stringify(window.__gpuRun(0.5), (k, v) => typeof v === "number" ? +v.toFixed(3) : v)')}`);
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
  // (The derby's round too, when the run is one: Lab::derby_json; with --eval
  // tools/vehicle-lab/derby-probe.js, its log a second.)
  const derby = await evaluate('window.__derby?.() ? { ...window.__derby(), log: window.__abLog ?? null } : null');
  writeFileSync(join(OUT, `${name}.stats.json`), JSON.stringify({ name, query, videoSeconds: frameNo / FPS, simulatedSeconds: st.t, stats, ...(derby ? { derby } : {}) }, null, 1));
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
