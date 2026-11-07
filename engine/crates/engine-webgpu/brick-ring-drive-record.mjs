// Records and probes the brick ring drive (brick-ring-drive.html) from headless Chrome.
//
// Record (frame-locked, `?record=1`: the page steps the World itself, one video frame of
// 1/60 s at a time, the scripted run `?auto=1`): the run-up from the chase camera, the
// strike from the wall's side, an orbit and the chase out of the hole; then the strike
// again ×4 slower from the wall's side and from outside the ring (replays: the scene is built
// again and driven by the same script to the strike), encoded with ffmpeg (H.264).
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node brick-ring-drive-record.mjs [--out dir] [--stills dir] [--query 'car=etk'] [name]
//   node brick-ring-drive-record.mjs --probe [--secs 14] [--query 'car=sedan&timing=phases']
//
// --probe: the page's own real-time loop (its FramePacer, ?auto=1) for --secs seconds,
// its readout every second (World steps/s, FPS, GPU ms split, awake bodies, broken beams).
// Runs: name[:query] (default `etk:car=etk`, `sedan:car=sedan`). Needs Chrome and ffmpeg.
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
const OUT = opt('--out', join(homedir(), 'Development', 'rubble-artifacts', 'brick-ring-drive'));
const STILLS = opt('--stills', null);
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const PROBE = args.includes('--probe');
const EXTRA = opt('--query', '');
const runs = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
if (STILLS) mkdirSync(STILLS, { recursive: true });
const LIMIT_MIN = Number(opt('--limit', '15'));
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();

// A static server over this directory (the vehicle folders in __mods may be a symlink).
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png', '.dds': 'application/octet-stream' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'brick-ring-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
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
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) pageErrors.push(m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 300_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

const open = async (query) => {
  await send('Page.navigate', { url: `${base}/brick-ring-drive.html?${query}${EXTRA ? '&' + EXTRA : ''}` });
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const err = await evaluate('window.__failed || document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}\n${pageErrors.join('\n')}`);
    if (await evaluate('window.__ready === true').catch(() => false)) break;
    if (Date.now() - t0 > Number(opt('--startLimit', '180')) * 1000) throw new Error(`the page did not start: ${await evaluate('document.getElementById("status")?.textContent').catch(() => '?')}\n${pageErrors.join('\n')}`);
  }
  return (Date.now() - t0) / 1000;
};
const grabTo = async (file, quality = 90) => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality });
  writeFileSync(file, Buffer.from(s.result.data, 'base64'));
};
const parse = (run) => { const [name, ...q] = run.split(':'); return { name, query: q.join(':') || `car=${name}` }; };

if (PROBE) {
  for (const run of runs.length ? runs : ['sedan']) {
    const { name, query } = parse(run);
    const built = await open(`${query}&auto=1&clean=1`);
    console.log(`${name}: ready in ${built.toFixed(1)} s`);
    if (opt('--eval', null)) console.log(await evaluate(readFileSync(opt('--eval'), 'utf8')));
    for (let s = 1; s <= Number(opt('--secs', '14')); s++) {
      await new Promise((r) => setTimeout(r, 1000));
      const st = await evaluate(`JSON.stringify((({beamsBroken, awake, simRate, fps, t, gpu, cars}) => ({ t: +t.toFixed(2), simRate: +simRate.toFixed(1), fps: +fps.toFixed(0), awake, beamsBroken,
        kmh: +(3.6 * Math.hypot(cars[0].v[0], cars[0].v[2])).toFixed(0), z: +cars[0].c[2].toFixed(1),
        gpu: gpu && gpu.total.length ? { total: +gpu.split.total.toFixed(2), rigid: +gpu.split.rigid.toFixed(2), vehicle: +gpu.split.vehicle.toFixed(2) } : null }))(window.__state()))`);
      console.log(`  ${s}s ${st}`);
      if (STILLS && s % 3 === 0) await grabTo(join(STILLS, `probe-${name}-${s}.jpg`), 80);
    }
    if (opt('--segments', null)) console.log(JSON.stringify(await evaluate('window.__gpu().segments')));
    if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
  }
  process.exit(0);
}

const FPS = 60;
const videos = [];
for (const run of runs.length ? runs : ['etk']) {
  const { name, query } = parse(run);
  // Frames beside the videos (a restart wipes the temporary directory).
  mkdirSync(join(OUT, 'frames'), { recursive: true });
  const frameDir = mkdtempSync(join(OUT, 'frames', `${name}-`));
  let frameNo = 0;
  const grab = () => grabTo(join(frameDir, `f${String(frameNo++).padStart(6, '0')}.jpg`));
  const rec = (expr) => evaluate(`(async () => { const R = window.__rec; return ${expr}; })()`);
  // Pass 1: real time, the director: chase on the run-up, the wall side through the
  // strike, an orbit, then the chase out of the hole.
  await open(`${query}&record=1`);
  await rec('R.set({ cam: "director" })');
  let st = null;
  for (let f = 0; f < 4; f++) await rec('R.frame(60, 0)');
  const secs = Number(opt('--secs', '16'));
  let hit = null, k = 0;
  const stills = [0.25, 0.7, 1.5, 3.5];
  while (!st || st.t < Math.min(secs, (hit ?? secs) + 9)) {
    st = await rec('R.frame(60, 1)'); await grab();
    hit ??= st.impactAt;
    if (STILLS && hit != null && k < stills.length && st.t >= hit + stills[k]) await grabTo(join(STILLS, `${name}-${k++}.jpg`), 78);
  }
  if (hit == null) throw new Error(`${name}: no strike seen in ${st.t.toFixed(1)} s`);
  const realFrames = frameNo;
  // Passes 2 and 3: the strike ×4 slower from the wall's side, then from outside the
  // ring where the car comes out.
  for (const [r, camName] of ['side', 'outside'].entries()) {
    await open(`${query}&record=1&slowmo=1&impactAt=${hit.toFixed(4)}`);
    await rec(`R.set({ cam: "${camName}" })`);
    // The entry from inside (the car is in the wall a second later), then the
    // break-out from outside.
    const from = Math.max(0, hit - (r ? 0.15 : 0.6));
    while (true) { st = await rec('R.frame(60, 1)'); if (st.t >= from - 1e-6) break; }
    const end = hit + (r ? 2.4 : 0.9);
    let shot = 0;
    while (st.t < end) {
      st = await rec('R.frame(60)');
      await grab();
      if (STILLS && shot < 2 && st.t >= hit + (r ? [0.45, 0.9] : [0.05, 0.3])[shot]) await grabTo(join(STILLS, `${name}-${camName}-${shot++}.jpg`), 78);
    }
  }
  const file = join(OUT, `brick-ring-drive-${name}.mp4`);
  const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: 'inherit' });
  rmSync(frameDir, { recursive: true, force: true });
  if (enc.status !== 0) throw new Error('ffmpeg failed');
  console.log(`${name}: ${(realFrames / FPS).toFixed(1)} s real time + ${((frameNo - realFrames) / FPS).toFixed(1)} s replays (strike at ${hit.toFixed(3)} s) → ${file}`);
  videos.push({ name, file, real: realFrames / FPS, total: frameNo / FPS, impact: hit });
  if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
}
const index = join(OUT, 'videos.json');
const all = existsSync(index) ? JSON.parse(readFileSync(index, 'utf8')).filter((v) => !videos.some((w) => w.name === v.name)) : [];
writeFileSync(index, JSON.stringify([...all, ...videos], null, 1));
process.exit(0);
