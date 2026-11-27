// Records a crash reference scenario (engine_softbody::crash_reference, docs/crash-reference.md)
// in the web crash lab: softbody.html?solver=explicit&crashRef=ID, frame-locked in headless
// Chrome (the page's own physics, renderer and report, stepped from here). Saves a still at
// each of the scenario's moments (ms after first contact, from the registry) from its own
// camera, the criteria panel at the end, the run's CrashReport as judged by the page (Rust,
// crash_ref.rs) and, with --video, an H.264 film of the whole run (one frame per sample,
// 4 ms of simulation: 15x slow motion at 60 fps).
//
//   python3 engine/scripts/with-gpu-lock.py --limit 300 -- \
//     node crash-ref-record.mjs --crashRef S02 [--cam side] [--contact edge] [--run 0] \
//       [--size 1280x720] [--video] [--all-cams] [--query 'skin3=ghost&skeleton=1'] [--eval EXPR] [--out DIR]
//
// Output: DIR (default engine/out/crash-reference/<id>-<contact>-run<k>/): <moment>-<cam>.png
// (--all-cams: every camera's), panel.png, report.json, scenario.json (the registry entry:
// its cameras and moments, for engine/tools/crash-reference/sheet.py) and <id>.mp4. Serves
// this folder (its __mods/<design>_new: the site build's generated designs, or a link to
// `vehicle_gen_rig --out` folders). The
// Chrome profile is kept in engine/out/crash-reference/chrome-profile (CRASH_REF_PROFILE)
// so its shader cache serves later runs. Exit code 1 on a page error or console error.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync, rmSync } from 'node:fs';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromeArgs } from './check-chrome.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const OUT_ROOT = normalize(join(ROOT, '../../out/crash-reference'));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const ID = opt('--crashRef');
if (!ID) { console.error('usage: crash-ref-record.mjs --crashRef ID [--cam NAME] [--contact edge] [--run 0] [--size WxH] [--video] [--all-cams] [--out DIR]'); process.exit(2); }
const CONTACT = opt('--contact', 'edge');
const RUN = Number(opt('--run', 0));
const CAM = opt('--cam', null);
const VIDEO = args.includes('--video');
const ALL_CAMS = args.includes('--all-cams');
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const OUT = opt('--out', join(OUT_ROOT, `${ID}-${CONTACT}-run${RUN}`));
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

const profile = process.env.CRASH_REF_PROFILE || join(OUT_ROOT, 'chrome-profile');
mkdirSync(profile, { recursive: true });
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, chromeArgs({ stock: false, port, profile, width: W, height: H }), { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 0;
const errors = [];
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
  const shot = async (file) => {
    const s = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, file), Buffer.from(s.result.data, 'base64'));
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  const q = new URLSearchParams({ solver: 'explicit', crashRef: ID, run: String(RUN), contact: CONTACT, paused: '1' });
  if (CAM) q.set('cam', CAM);
  // --query a=b&c=d: more of the page's options (skin3=ghost&skeleton=1&color=strain: the beams).
  for (const [k, v] of new URLSearchParams(opt('--query', ''))) q.set(k, v);
  const url = `${base}/softbody.html?${q}`;
  console.log(url);
  await send('Page.navigate', { url });
  const t0 = Date.now();
  for (;;) {
    await sleep(500);
    const st = await evaluate('({ ready: window.__ready === true && !!window.__lab && !!window.__crashRef, failed: window.__failed ?? null, err: document.getElementById("err")?.textContent || "" })').catch(() => ({}));
    if (st.failed || st.err) throw new Error(`page error: ${st.failed || st.err}`);
    if (st.ready) break;
    if (Date.now() - t0 > 240_000) throw new Error('the page did not start in 240 s (kernel compiles: run again, the profile keeps the cache)');
  }
  console.log(`ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  // --eval EXPR: a page expression's JSON, printed once the page is ready (inspection).
  if (opt('--eval')) console.log(JSON.stringify(await evaluate(`(${opt('--eval')})`)));
  const info = await evaluate('window.__crashRef.info()');
  const scenario = await evaluate('window.__crashRef.scenario');
  const cam = CAM && scenario.cameras.some((c) => c.name === CAM) ? CAM : scenario.cameras[0].name;
  const L = (expr) => evaluate(`(async () => { const L = window.__lab; return ${expr}; })()`);
  await L(`(L.lab.camera(${JSON.stringify(`ref:${cam}`)}), L.lab.setClean(true), true)`);
  // The moments' steps (ms after first contact at 2 kHz), in order.
  const stepOf = (ms) => Math.max(0, Math.round(info.impactStep + ms * 2));
  const moments = scenario.moments.map((m) => ({ ...m, step: stepOf(m.after_impact_ms) })).sort((a, b) => a.step - b.step);
  const frames = [];
  let at = 0;
  const advanceTo = async (step) => {
    while (at < step) {
      // (A video frame at each sample: chunks end on the sample steps.)
      const k = Math.min(step - at, VIDEO ? info.sampleSteps - (at % info.sampleSteps) : 64);
      const s = await L(`L.advance(${k}, ${k})`);
      at = s.step;
      if (VIDEO && at % info.sampleSteps === 0) {
        const f = await send('Page.captureScreenshot', { format: 'jpeg', quality: 90 });
        frames.push(Buffer.from(f.result.data, 'base64'));
      }
    }
  };
  // --all-cams: a still from each of the scenario's cameras at each moment (the video stays
  // on the primary camera: the view goes back to it after the stills).
  const stillCams = ALL_CAMS ? scenario.cameras.map((c) => c.name) : [cam];
  writeFileSync(join(OUT, 'scenario.json'), JSON.stringify(scenario, null, 1));
  for (const m of moments) {
    await advanceTo(m.step);
    // (The page eases its camera toward a new view, about half the way a draw: 16 draws
    // after a switch put it there to 1e-4 of the move.)
    for (const c of stillCams) {
      if (c !== cam) await L(`(L.lab.camera(${JSON.stringify(`ref:${c}`)}), true)`);
      for (let i = 0; i < (c !== cam ? 16 : 3); i++) await L('L.draw()');
      const file = `${m.name}-${c.replace(/\s+/g, '_')}.png`;
      await shot(file);
      console.log(`  ${m.name} (${m.after_impact_ms} ms after contact, step ${m.step}): ${file}`);
    }
    if (stillCams.some((c) => c !== cam)) {
      await L(`(L.lab.camera(${JSON.stringify(`ref:${cam}`)}), true)`);
      for (let i = 0; i < 16; i++) await L('L.draw()');
    }
  }
  await advanceTo(info.totalSteps);
  for (let i = 0; i < 600; i++) {
    const p = await evaluate('window.__crashRef.progress()');
    if (p.error) throw new Error(`crash reference run: ${p.error}`);
    if (p.pushed >= p.samples && await evaluate('!!window.__crashRef.report()')) break;
    await sleep(100);
  }
  const report = await evaluate('window.__crashRef.report()');
  if (!report) throw new Error('no report: the run did not complete its samples');
  writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 1));
  // panel.png: the criteria, the Results panel open (it starts collapsed to its summary).
  await L('(L.lab.setClean(false), L.lab.ui?.open("results", true), true)');
  for (let i = 0; i < 3; i++) await L('L.draw()');
  await shot('panel.png');
  if (VIDEO && frames.length) {
    const dir = join(OUT, 'frames');
    mkdirSync(dir, { recursive: true });
    frames.forEach((b, i) => writeFileSync(join(dir, `f${String(i).padStart(5, '0')}.jpg`), b));
    const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', '60', '-i', join(dir, 'f%05d.jpg'), '-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', join(OUT, `${ID}.mp4`)], { stdio: 'inherit' });
    rmSync(dir, { recursive: true, force: true });
    if (enc.status !== 0) throw new Error('ffmpeg failed');
    console.log(`  ${ID}.mp4 (${frames.length} frames)`);
  }
  const fails = report.results.filter((r) => r.asserted && !r.pass).map((r) => r.name);
  console.log(`report: ${Object.keys(report.metrics).length} metrics, ${report.results.length} criteria, asserted failures: ${fails.length ? fails.join('; ') : 'none'}`);
  console.log(`→ ${OUT}`);
  ws.close();
} catch (e) {
  console.error(String(e?.stack ?? e));
  code = 1;
} finally {
  if (errors.length) { console.log(`console errors (${errors.length}): ${JSON.stringify(errors.slice(0, 8))}`); code = 1; }
  else console.log('console errors: none');
  chrome.kill();
  await sleep(300);
  server.close();
}
process.exit(code);
