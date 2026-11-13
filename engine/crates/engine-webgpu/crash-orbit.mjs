// The cars after the brick ring: snapshots, crush measurements and orbit videos.
//
//   node crash-orbit.mjs capture etk|sedan   the scripted drive (brick-ring-drive.html
//       ?record=1, frame-locked) at rest for 1.5 s → <car>-rest.json; again through the
//       wall until the car is still or 3 s past the break-out → <car>-crash.json, with
//       a still from the drive's camera beside it (<car>-crash-drive.jpg).
//   node crash-orbit.mjs measure etk|sedan   no browser: the crush of <car>-crash.json
//       against <car>-rest.json in the car's own frame, written into the crash snapshot.
//   node crash-orbit.mjs record name snap[,snap] [--secs 12]   crash-orbit.html,
//       frame-locked, one orbit at 60 fps → name.mp4 (H.264), stills at four angles.
//   node crash-orbit.mjs stills name snap[,snap] [--views front34,side,rear34,topfront]
//       [--to dir]   crash-orbit.html at set views about the car's heading, framed tight.
//
// Runs: etk, sedan (?drag=0), etk2 (two ETKs: <run>-0-…, <run>-1-…). capture --after s:
// the crash snapshot s seconds after each car is out of the wall or at rest.
//
// Snapshots and stills go to --out (default <repo>/out/crash-orbit, gitignored) and are
// copied with the videos to ~/Development/rubble-artifacts/crash-orbit. ETK snapshots
// hold node positions of BeamNG community content: never commit them. Needs Chrome and
// ffmpeg; run it under engine/scripts/with-gpu-lock.py (one GPU job on the machine).
// A hard time limit (--limit, minutes) bounds every run.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync, realpathSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { measureCrush } from './crash-measure.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const pos = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--'));
const [MODE, ...REST] = pos;
const OUT = resolve(opt('--out', join(ROOT, '..', '..', '..', 'out', 'crash-orbit')));
const ART = opt('--artifacts', join(homedir(), 'Development', 'rubble-artifacts', 'crash-orbit'));
const STILLS = opt('--stills', null);
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const FPS = 60;
mkdirSync(OUT, { recursive: true });
mkdirSync(ART, { recursive: true });
if (STILLS) mkdirSync(STILLS, { recursive: true });
const LIMIT_MIN = Number(opt('--limit', '15'));
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();
const QUERY = { etk: 'car=etk', etk2: 'car=etk&n=2', sedan: 'car=sedan&drag=0' };
const keep = (file) => { copyFileSync(file, join(ART, basename(file))); };

// ---------------------------------------------------------------- measure (no browser)
// The crush (crash-measure.mjs): <name>-crash.json against <name>-rest.json in the rest
// car's frame (x right, y up, z forward: the car drives at the wall along +z at rest).
function measure(name) {
  const rest = JSON.parse(readFileSync(join(OUT, `${name}-rest.json`), 'utf8'));
  const crashFile = join(OUT, `${name}-crash.json`);
  const crash = JSON.parse(readFileSync(crashFile, 'utf8'));
  const metrics = measureCrush(rest.positions, crash.positions);
  crash.metrics = metrics;
  writeFileSync(crashFile, JSON.stringify(crash));
  keep(crashFile);
  console.log(`${name}: ${JSON.stringify(metrics, null, 1)}`);
}
if (MODE === 'measure') { for (const c of REST.length ? REST : ['etk', 'sedan']) measure(c); process.exit(0); }

// ---------------------------------------------------------------- the browser
// A static server over this directory (the vehicle folders in __mods may be a
// symlink), the snapshots under /__snap/.
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png', '.dds': 'application/octet-stream' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const [base, rel] = p.startsWith('/__snap/') ? [OUT, p.slice(8)] : [ROOT, p];
  const path = normalize(join(base, rel));
  if (!path.startsWith(base) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const profile = mkdtempSync(join(tmpdir(), 'crash-orbit-chrome-'));
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
const open = async (page, query) => {
  await send('Page.navigate', { url: `${base}/${page}?${query}` });
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const err = await evaluate('window.__failed || document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}\n${pageErrors.join('\n')}`);
    if (await evaluate('window.__ready === true').catch(() => false)) break;
    if (pageErrors.some((e) => /SyntaxError|TypeError|ReferenceError/.test(e))) throw new Error(`page error:\n${pageErrors.join('\n')}`);
    if (Date.now() - t0 > 180_000) throw new Error(`the page did not start\n${pageErrors.join('\n')}`);
  }
  return (Date.now() - t0) / 1000;
};
const grabTo = async (file, quality = 90) => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality });
  writeFileSync(file, Buffer.from(s.result.data, 'base64'));
};
const save = (name, obj) => { const f = join(OUT, name); writeFileSync(f, JSON.stringify(obj)); keep(f); return f; };

if (MODE === 'capture') {
  // --after s: the crash snapshot this long after each car is out of the wall or at rest
  // (whichever comes first; by default 3 s after the break-out or 0.5 s at rest).
  const AFTER = opt('--after', null);
  const outFor = AFTER == null ? 3 : Number(AFTER), restFor = AFTER == null ? 0 : Number(AFTER);
  for (const run of REST.length ? REST : ['etk', 'sedan']) {
    const q = QUERY[run];
    if (!q) throw new Error(`no run ${run} (${Object.keys(QUERY).join(', ')})`);
    const n = /(^|&)n=2/.test(q) ? 2 : 1;
    // One car: <run>-rest.json, <run>-crash.json; two: <run>-<i>-… for each.
    const name = (i) => (n > 1 ? `${run}-${i}` : run);
    const rec = (expr) => evaluate(`(async () => { const R = window.__rec; return ${expr}; })()`);
    // At rest: no input for 1.5 s (the car settles on its wheels).
    await open('brick-ring-drive.html', `${q}&record=1`);
    await evaluate('window.__drive({ forward: 0, steer: 0, brake: 1, handbrake: 1 })');
    let st = null;
    for (let f = 0; f < 90; f++) st = await rec('R.frame(60, 1)');
    for (let i = 0; i < n; i++) {
      const rest = await evaluate(`window.__snapshot(${i})`);
      rest.impactKmh = null; rest.fwd = st.cars[i].fwd;
      console.log(`${name(i)} rest: t ${rest.t.toFixed(2)} s, ${rest.beamsBroken} of ${rest.beamCount} beams broken → ${save(`${name(i)}-rest.json`, rest)}`);
    }
    // The crash: the scripted run at the wall.
    await open('brick-ring-drive.html', `${q}&record=1`);
    const layout = await evaluate('({ inner: window.__layout.inner })');
    await rec('R.set({ cam: "director" })');
    const OUT_AT = layout.inner + 5.2 + 2.3; // the car's rear past the wall's outer face
    const cars = Array.from({ length: n }, () => ({ speeds: [], impactAt: null, impactKmh: null, breakout: null, restAt: null, still: 0, maxKmh: 0, done: false }));
    for (;;) {
      st = await rec('R.frame(60, 1)');
      for (const [i, k] of cars.entries()) {
        const c = st.cars[i], kmh = 3.6 * Math.abs(c.v[0] * c.fwd[0] + c.v[2] * c.fwd[2]);
        k.speeds.push([st.t, kmh]); k.maxKmh = Math.max(k.maxKmh, kmh);
        // The strike: the page's (car 0), else this car's front within 0.3 m of the
        // wall's inner face; the speed on the frame at that time (seen a frame later).
        const toWall = layout.inner - (Math.hypot(c.c[0], c.c[2]) + 2.3);
        if (k.impactAt == null && (i === 0 ? st.impactAt != null : toWall < 0.3)) {
          k.impactAt = i === 0 ? st.impactAt : st.t - 1 / 60;
          k.impactKmh = k.speeds.reduce((b, s) => (Math.abs(s[0] - k.impactAt) < Math.abs(b[0] - k.impactAt) ? s : b))[1];
          console.log(`${name(i)}: strike at ${k.impactAt.toFixed(3)} s, ${k.impactKmh.toFixed(1)} km/h`);
        }
        if (k.impactAt != null && k.breakout == null && Math.hypot(c.c[0], c.c[2]) > OUT_AT) { k.breakout = st.t; console.log(`${name(i)}: out of the wall at ${st.t.toFixed(2)} s, ${kmh.toFixed(0)} km/h`); }
        k.still = Math.hypot(...c.v) < 0.3 ? k.still + 1 : 0;
        if (k.impactAt != null && k.restAt == null && k.still >= 30) { k.restAt = st.t - 0.5; console.log(`${name(i)}: at rest from ${k.restAt.toFixed(2)} s`); }
        if (k.impactAt != null && ((k.restAt != null && st.t >= k.restAt + 0.5 + restFor) || (k.breakout != null && st.t >= k.breakout + outFor))) k.done = true;
      }
      if (cars.every((k) => k.done)) break;
      if (st.t > 40) throw new Error(`${run}: no rest by 40 s (${JSON.stringify(cars.map(({ speeds, ...k }) => k))})`);
    }
    // The drive's own view of the car as it stands (beside it, outside the ring), to
    // hold the orbit's rendering against.
    await rec('R.set({ cam: "exit" })');
    for (let f = 0; f < 2; f++) await rec('R.frame(60, 0)');
    await grabTo(join(OUT, `${run}-crash-drive.jpg`), 88); keep(join(OUT, `${run}-crash-drive.jpg`));
    for (const [i, k] of cars.entries()) {
      const crash = await evaluate(`window.__snapshot(${i})`);
      Object.assign(crash, { impactAt: k.impactAt, impactKmh: k.impactKmh, breakout: k.breakout, restAt: k.restAt, maxKmh: k.maxKmh, fwd: st.cars[i].fwd, speeds: k.speeds.filter((_, j) => j % 6 === 0).map(([t, v]) => [+t.toFixed(3), +v.toFixed(2)]) });
      console.log(`${name(i)} crash: t ${crash.t.toFixed(2)} s (strike ${k.impactAt.toFixed(2)} s at ${k.impactKmh.toFixed(1)} km/h, top ${k.maxKmh.toFixed(0)} km/h), ${crash.beamsBroken} of ${crash.beamCount} beams broken → ${save(`${name(i)}-crash.json`, crash)}`);
    }
    if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
    pageErrors.length = 0;
  }
  process.exit(0);
}

if (MODE === 'record') {
  const [name, snaps] = REST;
  const secs = Number(opt('--secs', '12'));
  const extra = opt('--query', '');
  const built = await open('crash-orbit.html', `record=1&secs=${secs}&snap=${snaps.split(',').map((s) => `/__snap/${s}`).join(',')}${extra ? '&' + extra : ''}`);
  console.log(`${name}: ready in ${built.toFixed(1)} s`);
  mkdirSync(join(OUT, 'frames'), { recursive: true });
  const frameDir = mkdtempSync(join(OUT, 'frames', `${name}-`));
  const frames = Math.round(secs * FPS);
  const shots = [0, 0.25, 0.5, 0.75].map((f) => Math.round(f * frames));
  for (let k = 0; k < frames; k++) {
    await evaluate(`window.__rec.frame(${k / FPS})`);
    const f = join(frameDir, `f${String(k).padStart(6, '0')}.jpg`);
    await grabTo(f);
    if (shots.includes(k)) { const s = join(OUT, `${name}-${shots.indexOf(k)}.jpg`); copyFileSync(f, s); if (STILLS) copyFileSync(f, join(STILLS, `${name}-${shots.indexOf(k)}.jpg`)); }
  }
  const file = join(ART, `${name}.mp4`);
  const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: 'inherit' });
  if (enc.status !== 0) throw new Error('ffmpeg failed');
  rmSync(frameDir, { recursive: true, force: true });
  console.log(`${name}: ${frames} frames (${secs} s) → ${file}`);
  if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
  process.exit(0);
}
if (MODE === 'stills') {
  // Stills of crash-orbit.html at set views (front34, side, rear34, topfront) →
  // <--to or ART>/<name>-<view>.jpg.
  const [name, snaps] = REST;
  const views = opt('--views', 'front34,side,rear34,topfront').split(',');
  const to = opt('--to', ART);
  mkdirSync(to, { recursive: true });
  const extra = opt('--query', '');
  await open('crash-orbit.html', `record=1&snap=${snaps.split(',').map((s) => `/__snap/${s}`).join(',')}${extra ? '&' + extra : ''}`);
  for (const v of views) {
    // Drawn twice: the before / after tags follow the first draw.
    for (let k = 0; k < 2; k++) await evaluate(`window.__rec.view(${JSON.stringify(v)}, ${Number(opt('--fill', '0.9'))})`);
    const f = join(to, `${name}-${v}.jpg`);
    await grabTo(f, 92);
    console.log(`${name} ${v} → ${f}`);
  }
  if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
  process.exit(0);
}
console.error('usage: node crash-orbit.mjs capture|measure|record|stills …');
process.exit(2);
