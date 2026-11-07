// Records the one-World showcase (one-world.html) from headless Chrome: each preset
// frame-locked (`?record=1`: the page steps the World itself, one video frame of
// 1/60 s at a time), first in real time from a chase camera through the crash and
// after it, then the strike again from the impact camera ×4 slower (a replay: the
// scene is built again and run to the strike), encoded with ffmpeg (H.264).
//
//   cargo run --release -p engine-webgpu --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node one-world-record.mjs [--size 1280x720] [--out dir] [--stills dir] [preset ...]
//   node one-world-record.mjs --probe house      # real time for 12 s: the page's readout
//
// Presets: house, crates, twoway, convoy (convoy:4: four cars). --montage: also a
// 60 s cut of every preset's video. Needs Chrome and ffmpeg.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', join(ROOT, 'recordings', 'one-world'));
const STILLS = opt('--stills', null);
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const PROBE = args.includes('--probe');
const MONTAGE = args.includes('--montage');
const EXTRA = opt('--query', '');
const only = args.filter((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--') || (args[i - 1] === '--probe' && !a.startsWith('--')));
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
// A hard limit on the whole run (GPU runs stay bounded).
const LIMIT_MIN = Number(opt('--limit', '15'));
setTimeout(() => { console.error(`time limit (${LIMIT_MIN} min)`); process.exit(9); }, LIMIT_MIN * 60_000).unref();

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const keepProfile = process.env.ONE_WORLD_PROFILE;
if (keepProfile) mkdirSync(keepProfile, { recursive: true });
const profile = keepProfile || mkdtempSync(join(tmpdir(), 'one-world-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} if (!keepProfile) { try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} } };
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
  await send('Page.navigate', { url: `${base}/one-world.html?${query}${EXTRA ? '&' + EXTRA : ''}` });
  const t0 = Date.now();
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const err = await evaluate('window.__failed || document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) throw new Error(`page error: ${err}\n${pageErrors.join('\n')}`);
    if (await evaluate('window.__ready === true').catch(() => false)) break;
    if (Date.now() - t0 > Number(opt('--startLimit', '150')) * 1000) throw new Error(`the page did not start: ${await evaluate('document.getElementById("status")?.textContent').catch(() => '?')}\n${pageErrors.join('\n')}`);
  }
  return (Date.now() - t0) / 1000;
};
const grabTo = async (file, quality = 90) => {
  const s = await send('Page.captureScreenshot', { format: 'jpeg', quality });
  writeFileSync(file, Buffer.from(s.result.data, 'base64'));
};

const parse = (run) => { const [preset, n] = run.split(':'); return { preset, n: Number(n) || 2, name: n ? `${preset}${n}` : preset }; };

if (PROBE) {
  // Real time (the page's own loop and pacer) for 12 s: its readout every second.
  for (const run of only.length ? only : ['house']) {
    const { preset, n, name } = parse(run);
    const built = await open(`preset=${preset}&n=${n}&timing=step&clean=1`);
    console.log(`${name}: ready in ${built.toFixed(1)} s`);
    // --eval FILE: a script run in the page once it is ready (diagnostics); its value printed.
    if (opt('--eval', null)) console.log(await evaluate(readFileSync(opt('--eval'), 'utf8')));
    for (let s = 1; s <= Number(opt("--secs", "12")); s++) {
      await new Promise((r) => setTimeout(r, 1000));
      const st = await evaluate('JSON.stringify((({broken, beamsBroken, bodies, simRate, fps, t, gpu}) => ({broken, beamsBroken, bodies, simRate: +simRate.toFixed(1), fps: +fps.toFixed(0), t: +t.toFixed(2), gpu: gpu && gpu.total.length ? +[...gpu.total].sort((a, b) => a - b)[gpu.total.length >> 1].toFixed(2) : null}))(window.__state()))');
      console.log(`  ${s}s ${st}`);
      if (STILLS && s % 3 === 0) await grabTo(join(STILLS, `probe-${name}-${s}.jpg`), 80);
    }
    if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
  }
  process.exit(0);
}

const FPS = 60;
const videos = [];
for (const run of only.length ? only : ['house', 'crates', 'twoway', 'convoy']) {
  const { preset, n, name } = parse(run);
  const frameDir = mkdtempSync(join(tmpdir(), 'one-world-frames-'));
  let frameNo = 0;
  const grab = () => grabTo(join(frameDir, `f${String(frameNo++).padStart(6, '0')}.jpg`));
  const rec = (expr) => evaluate(`(async () => { const R = window.__rec; return ${expr}; })()`);
  // Pass 1: real time, the director's chase → impact → orbit, no slow motion.
  await open(`preset=${preset}&n=${n}&record=1`);
  const layout = await evaluate('window.__layout');
  const T = layout.impactTime;
  await rec('R.set({ cam: "director", slow: false })');
  let st = null;
  for (let f = 0; f < 6; f++) await rec('R.frame(60, 0)'); // the camera settles, nothing moves
  // Until 6 s past the impact the page saw (or 6 s past the predicted one).
  let hit = null;
  while (!st || st.t < (hit ?? T) + (preset === 'twoway' ? 5.5 : 6.5)) {
    st = await rec('R.frame(60, 1)'); await grab();
    hit ??= await evaluate('window.__state().impactAt');
  }
  hit ??= T;
  const realFrames = frameNo;
  if (STILLS) await grabTo(join(STILLS, `${name}-after.jpg`), 82);
  // Pass 2 and 3: the replay, the strike ×4 slow from the impact camera, then again
  // from a second angle.
  const REPLAYS = { house: ['side', 'top'], crates: ['side', 'top'], twoway: ['side', 'orbit'], convoy: ['side', 'top'] }[preset];
  let k = 0;
  for (const [r, camName] of REPLAYS.entries()) {
    await open(`preset=${preset}&n=${n}&record=1&impactAt=${hit.toFixed(4)}`);
    await rec(`R.set({ cam: "${camName}", slow: true })`);
    const from = Math.max(0, hit - (r ? 0.3 : preset === 'twoway' ? 0.5 : 0.8));
    while (true) { st = await rec('R.frame(60, 1)'); if (st.t >= from - 1e-6) break; }
    const stillAt = r ? [] : preset === 'twoway' ? [hit + 0.05, hit + 0.6, hit + 1.0] : [hit + 0.1, hit + 0.35, hit + 0.8];
    const end = hit + (r ? 1.5 : preset === 'twoway' ? 2.6 : 2.2);
    while (st.t < end) {
      st = await rec('R.frame(60)');
      await grab();
      if (STILLS && k < stillAt.length && st.t >= stillAt[k]) await grabTo(join(STILLS, `${name}-impact-${k++}.jpg`), 82);
    }
  }
  const file = join(OUT, `one-world-${name}.mp4`);
  const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-framerate', String(FPS), '-i', join(frameDir, 'f%06d.jpg'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: 'inherit' });
  rmSync(frameDir, { recursive: true, force: true });
  if (enc.status !== 0) throw new Error('ffmpeg failed');
  console.log(`${name}: ${(realFrames / FPS).toFixed(1)} s real time + ${((frameNo - realFrames) / FPS).toFixed(1)} s replay → ${file}`);
  videos.push({ name, file, real: realFrames / FPS, total: frameNo / FPS, impact: hit, replayFrom: realFrames / FPS });
  if (pageErrors.length) console.log('page messages:\n' + pageErrors.slice(0, 10).join('\n'));
}
writeFileSync(join(OUT, 'videos.json'), JSON.stringify(videos, null, 1));
process.exit(0);
