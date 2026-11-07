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
const QUERY = { etk: 'car=etk', sedan: 'car=sedan&drag=0' };
const keep = (file) => { copyFileSync(file, join(ART, basename(file))); };

// ---------------------------------------------------------------- measure (no browser)
// Rigid alignment (Kabsch, Horn's quaternion form) of the crashed nodes onto the rest
// nodes on the car's rear two fifths, iterated with the nodes that moved more than
// 3 cm left out, then each node's displacement in the rest car's frame: x right, y up,
// z forward (the car drives at the wall along +z at rest).
function kabsch(P, Q, w) {
  // Rotation R and translation t with R·P + t ≈ Q (weighted rows).
  let sw = 0; const cp = [0, 0, 0], cq = [0, 0, 0];
  for (let i = 0; i < P.length; i++) if (w[i]) { sw += 1; for (let j = 0; j < 3; j++) { cp[j] += P[i][j]; cq[j] += Q[i][j]; } }
  for (let j = 0; j < 3; j++) { cp[j] /= sw; cq[j] /= sw; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < P.length; i++) if (w[i]) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += (P[i][a] - cp[a]) * (Q[i][b] - cq[b]);
  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz], [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]];
  // The largest eigenvector by power iteration on N + cI.
  let q = [1, 0, 0, 0];
  for (let it = 0; it < 500; it++) {
    const n = [0, 1, 2, 3].map((r) => N[r].reduce((s, v, c) => s + v * q[c], 0) + 1e3 * q[r]);
    const l = Math.hypot(...n); q = n.map((v) => v / l);
  }
  const [w0, x, y, z] = q;
  const R = [[w0 * w0 + x * x - y * y - z * z, 2 * (x * y - w0 * z), 2 * (x * z + w0 * y)], [2 * (x * y + w0 * z), w0 * w0 - x * x + y * y - z * z, 2 * (y * z - w0 * x)], [2 * (x * z - w0 * y), 2 * (y * z + w0 * x), w0 * w0 - x * x - y * y + z * z]];
  const t = [0, 1, 2].map((a) => cq[a] - (R[a][0] * cp[0] + R[a][1] * cp[1] + R[a][2] * cp[2]));
  return (p) => [0, 1, 2].map((a) => R[a][0] * p[0] + R[a][1] * p[1] + R[a][2] * p[2] + t[a]);
}
function measure(car) {
  const rest = JSON.parse(readFileSync(join(OUT, `${car}-rest.json`), 'utf8'));
  const crashFile = join(OUT, `${car}-crash.json`);
  const crash = JSON.parse(readFileSync(crashFile, 'utf8'));
  const R = rest.positions, C = crash.positions, n = R.length;
  if (C.length !== n) throw new Error('snapshots of different cars');
  // The rest car's frame: z its +z heading (forward), centred on its bounding box.
  let zlo = Infinity, zhi = -Infinity, xlo = Infinity, xhi = -Infinity;
  for (const p of R) { zlo = Math.min(zlo, p[2]); zhi = Math.max(zhi, p[2]); xlo = Math.min(xlo, p[0]); xhi = Math.max(xhi, p[0]); }
  const len = zhi - zlo, cx = 0.5 * (xlo + xhi);
  let w = R.map((p) => (p[2] - zlo < 0.4 * len ? 1 : 0));
  let align = null;
  for (let it = 0; it < 6; it++) {
    align = kabsch(C, R, w);
    const res = C.map((p, i) => { const a = align(p); return Math.hypot(a[0] - R[i][0], a[1] - R[i][1], a[2] - R[i][2]); });
    const nw = R.map((p, i) => (p[2] - zlo < 0.4 * len && res[i] < 0.03 ? 1 : 0));
    if (nw.reduce((a, b) => a + b, 0) < 12) break;
    w = nw;
  }
  const A = C.map(align);
  const used = w.reduce((a, b) => a + b, 0);
  const rms = Math.sqrt(A.reduce((s, a, i) => s + (w[i] ? (a[0] - R[i][0]) ** 2 + (a[1] - R[i][1]) ** 2 + (a[2] - R[i][2]) ** 2 : 0), 0) / used);
  const disp = A.map((a, i) => [a[0] - R[i][0], a[1] - R[i][1], a[2] - R[i][2]]);
  const mag = disp.map((d) => Math.hypot(...d));
  // Front crush: the frontmost rest nodes (within 10 cm of the nose), how far back
  // they went (mean and the most), and the car's overall length lost at the front.
  const front = R.map((p, i) => i).filter((i) => zhi - R[i][2] < 0.1);
  const back = front.map((i) => -disp[i][2]);
  let azhi = -Infinity; for (const a of A) azhi = Math.max(azhi, a[2]);
  let mi = 0; for (let i = 1; i < n; i++) if (mag[i] > mag[mi]) mi = i;
  const where = (p) => {
    const f = (p[2] - zlo) / len;
    const along = f > 0.8 ? 'front' : f > 0.6 ? 'front quarter' : f > 0.4 ? 'middle' : f > 0.2 ? 'rear quarter' : 'rear';
    const side = Math.abs(p[0] - cx) < 0.25 ? 'centre' : p[0] > cx ? '+x side' : '−x side';
    return `${along}, ${side}`;
  };
  const over = (d) => mag.filter((m) => m > d).length;
  // Displacement by fifth of the car's length (rest position): mean and max.
  const bands = [0, 1, 2, 3, 4].map((b) => { const ids = R.map((p, i) => i).filter((i) => Math.min(4, Math.floor(5 * (R[i][2] - zlo) / len)) === b); return { from: +(b / 5).toFixed(1), nodes: ids.length, mean: +(ids.reduce((s, i) => s + mag[i], 0) / ids.length).toFixed(4), max: +Math.max(...ids.map((i) => mag[i])).toFixed(4) }; });
  const metrics = {
    restLength: +len.toFixed(4),
    alignNodes: used, alignRms: +rms.toFixed(4),
    frontNodes: front.length,
    frontCrush: +(back.reduce((a, b) => a + b, 0) / back.length).toFixed(4),
    frontCrushMax: +Math.max(...back).toFixed(4),
    lengthLoss: +(zhi - azhi).toFixed(4),
    maxDisp: +mag[mi].toFixed(4), maxNode: mi, maxAt: R[mi].map((x) => +x.toFixed(3)), maxDir: disp[mi].map((x) => +x.toFixed(3)), maxWhere: where(R[mi]),
    nodesMoved: { over5cm: over(0.05), over10cm: over(0.1), over20cm: over(0.2), of: n },
    bands,
  };
  crash.metrics = metrics;
  writeFileSync(crashFile, JSON.stringify(crash));
  keep(crashFile);
  console.log(`${car}: ${JSON.stringify(metrics, null, 1)}`);
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
  for (const car of REST.length ? REST : ['etk', 'sedan']) {
    const q = QUERY[car];
    if (!q) throw new Error(`no car ${car}`);
    const rec = (expr) => evaluate(`(async () => { const R = window.__rec; return ${expr}; })()`);
    // At rest: no input for 1.5 s (the car settles on its wheels).
    await open('brick-ring-drive.html', `${q}&record=1`);
    await evaluate('window.__drive({ forward: 0, steer: 0, brake: 1, handbrake: 1 })');
    let st = null;
    for (let f = 0; f < 90; f++) st = await rec('R.frame(60, 1)');
    const rest = await evaluate('window.__snapshot(0)');
    rest.impactKmh = null;
    console.log(`${car} rest: t ${rest.t.toFixed(2)} s, ${rest.beamsBroken} of ${rest.beamCount} beams broken → ${save(`${car}-rest.json`, rest)}`);
    // The crash: the scripted run at the wall.
    await open('brick-ring-drive.html', `${q}&record=1`);
    const layout = await evaluate('({ inner: window.__layout.inner })');
    await rec('R.set({ cam: "director" })');
    const speeds = [];
    let impactKmh = null, breakout = null, still = 0, maxKmh = 0;
    const OUT_AT = layout.inner + 5.2 + 2.3; // the car's rear past the wall's outer face
    for (;;) {
      st = await rec('R.frame(60, 1)');
      const c = st.cars[0], kmh = 3.6 * Math.abs(c.v[0] * c.fwd[0] + c.v[2] * c.fwd[2]);
      speeds.push([st.t, kmh]); maxKmh = Math.max(maxKmh, kmh);
      if (impactKmh == null && st.impactAt != null) {
        // The speed on the frame at the strike's time (detected a frame later).
        impactKmh = speeds.reduce((b, s) => (Math.abs(s[0] - st.impactAt) < Math.abs(b[0] - st.impactAt) ? s : b))[1];
        console.log(`${car}: strike at ${st.impactAt.toFixed(3)} s, ${impactKmh.toFixed(1)} km/h`);
      }
      if (impactKmh != null && breakout == null && Math.hypot(c.c[0], c.c[2]) > OUT_AT) { breakout = st.t; console.log(`${car}: out of the wall at ${st.t.toFixed(2)} s, ${kmh.toFixed(0)} km/h`); }
      still = Math.hypot(...c.v) < 0.3 ? still + 1 : 0;
      if (impactKmh != null && (still >= 30 || (breakout != null && st.t > breakout + 3))) break;
      if (st.t > 40) throw new Error(`${car}: no rest by 40 s (strike ${impactKmh}, break-out ${breakout})`);
    }
    // The drive's own view of the car as it stands (beside it, outside the ring), to
    // hold the orbit's rendering against.
    await rec('R.set({ cam: "exit" })');
    for (let f = 0; f < 2; f++) await rec('R.frame(60, 0)');
    await grabTo(join(OUT, `${car}-crash-drive.jpg`), 88); keep(join(OUT, `${car}-crash-drive.jpg`));
    const crash = await evaluate('window.__snapshot(0)');
    Object.assign(crash, { impactKmh, breakout, maxKmh, speeds: speeds.filter((_, k) => k % 6 === 0).map(([t, v]) => [+t.toFixed(3), +v.toFixed(2)]) });
    console.log(`${car} crash: t ${crash.t.toFixed(2)} s (strike ${crash.impactAt.toFixed(2)} s at ${impactKmh.toFixed(1)} km/h, top ${maxKmh.toFixed(0)} km/h), ${crash.beamsBroken} of ${crash.beamCount} beams broken → ${save(`${car}-crash.json`, crash)}`);
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
console.error('usage: node crash-orbit.mjs capture|measure|record …');
process.exit(2);
