#!/usr/bin/env node
// Screenshots of the showcase car page (showcase-car.html) in headless Chrome WebGPU:
// at rest from three sides, and frozen after a wall, a centre pole, a side pole and
// a roll-over drop. Fails on a page or GPU error.
//
//   cargo run -p engine-webgpu --release --no-default-features --bin wgsl_export
//   wasm-pack build crates/engine-world-wasm --release --target web \
//     --out-dir ../engine-webgpu/generated/world-wasm --no-typescript
//   node showcase-car-check.mjs [--res medium] [--out DIR] [--size 1920x1080] [--only a,b]
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', join(ROOT, 'recordings', 'showcase-car'));
const RES = opt('--res', 'medium');
const [W, H] = opt('--size', '1280x720').split('x').map(Number);
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const path = normalize(join(ROOT, p));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'showcase-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', `--window-size=${W},${H}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
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
const pageErrors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') pageErrors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });

const RUNS = [
  // Rest pose (no steps): the design itself.
  ['rest_three', 'scene=idle&freeze=0&cam=three'],
  ['rest_front', 'scene=idle&freeze=0&cam=front'],
  ['rest_rear', 'scene=idle&freeze=0&cam=rear'],
  ['rest_side', 'scene=idle&freeze=0&cam=side&dist=6.5'],
  ['rest_close', 'scene=idle&freeze=0&cam=three&dist=3.4'],
  ['rest_low', 'scene=idle&freeze=0&cam=0.85,0.1,0.75&dist=5.4&look=0,-0.1,0'],
  ['rest_rear_low', 'scene=idle&freeze=0&cam=-0.7,0.12,-0.8&dist=5.4&look=0,-0.1,0'],
  ['rest_back', 'scene=idle&freeze=0&cam=0.02,0.18,-1&dist=4.6&look=0,0,0'],
  ['rest_nose', 'scene=idle&freeze=0&cam=0.02,0.12,1&dist=4.6&look=0,-0.05,0'],
  ['rest_tail', 'scene=idle&freeze=0&cam=-0.45,0.2,-1&dist=1.7&look=-0.6,0.3,-2.2'],
  ['rest_head', 'scene=idle&freeze=0&cam=0.5,0.15,1&dist=1.7&look=0.6,0.15,2.2'],
  ['rest_screen', 'scene=idle&freeze=0&cam=0.55,0.5,1&dist=2.2&look=0.2,0.45,1.0'],
  ['rest_fplate', 'scene=idle&freeze=0&cam=0.3,0.05,1&dist=1.6&look=0.35,-0.25,2.3'],
  ['rest_interior', 'scene=idle&freeze=0&cam=1,0.3,0.1&dist=3.6&look=0,0.25,-0.1'],
  ['rest_dash', 'scene=idle&freeze=0&cam=0.95,0.3,-0.35&dist=1.5&look=0.25,0.3,0.6'],
  ['rest_wheel', 'scene=idle&freeze=0&cam=1,0.1,0.25&dist=1.9&look=0.3,-0.25,1.3'],
  // Surfacing close-ups: the C-pillar into the deck and shoulder, the flank's
  // highlights from low and level.
  ['rest_cpillar', 'scene=idle&freeze=0&cam=0.75,0.55,-0.35&dist=2.3&look=0.6,0.45,-1.0'],
  ['rest_flank', 'scene=idle&freeze=0&cam=1,0.02,0.35&dist=4.2&look=0.4,-0.1,0'],
  ['intact_three', 'scene=idle&freeze=1.5&cam=three'],
  ['intact_front', 'scene=idle&freeze=1.5&cam=front'],
  ['intact_rear', 'scene=idle&freeze=1.5&cam=rear'],
  ['intact_side', 'scene=idle&freeze=1.5&cam=side&dist=6.5'],
  ['wall_56', 'scene=wall&speed=56&runUp=0.4&freeze=1.2&cam=three'],
  ['pole_50', 'scene=pole&speed=50&runUp=0.4&freeze=1.4&cam=three'],
  ['sidepole_32', 'scene=sidepole&speed=32&runUp=0.3&freeze=1.2&cam=front&dist=6'],
  ['drop_15', 'scene=drop&height=15&freeze=3.5&cam=three'],
  ['wall_120', 'scene=wall&speed=120&runUp=0.6&freeze=1.0&cam=three&dist=7'],
  ['interior', 'scene=idle&freeze=1.0&cam=left&dist=3.2'],
  ['perf', 'scene=idle&freeze=0&cam=three&perf=1'],
  ['perf_far', 'scene=idle&freeze=0&cam=three&perf=1&dist=30'],
  ['perf_skin', 'scene=idle&freeze=0&cam=three&perf=skin'],
  ['perf_draw', 'scene=idle&freeze=0&cam=three&perf=draw'],
  ['perf_noskin', 'scene=idle&freeze=0&cam=three&perf=1&noskin=1'],
  ['perf_n10', 'scene=idle&freeze=0&cam=three&perf=1&n=10&dist=16'],
  ['perf_n10_far', 'scene=idle&freeze=0&cam=three&perf=1&n=10&dist=80'],
  ['perf_n10_lod0', 'scene=idle&freeze=0&cam=three&perf=1&n=10&dist=16&lod=0'],
  ['perf_pass', 'scene=idle&freeze=0&cam=three&perf=pass'],
];
let failed = 0;
const only = opt('--only', null);
for (const [name, q] of RUNS.filter(([n]) => !only || only.split(',').includes(n))) {
  await send('Page.navigate', { url: `${base}/showcase-car.html?res=${RES}&${q}` });
  let st = null;
  for (let i = 0; i < 600; i++) {
    await new Promise((r) => setTimeout(r, 100));
    st = await evaluate('window.__state');
    if (st?.error || (st?.frozen && st.ready)) break;
  }
  await new Promise((r) => setTimeout(r, 400));
  const s = await send('Page.captureScreenshot', { format: 'png' });
  const png = join(OUT, `${RES}_${name}.png`);
  writeFileSync(png, Buffer.from(s.result.data, 'base64'));
  const err = st?.error || pageErrors.shift();
  if (err) failed++;
  console.log(`${name}: ${err ? 'ERROR ' + err : 'ok'}  t ${st?.time?.toFixed?.(2)} s  GPU ${st?.gpuMs?.toFixed?.(2)} ms/frame${st?.renderMs ? ` car render ${st.renderMs.toFixed(3)} ms [${st.features}]` : ''}  ${st?.damage ? `broken ${st.damage.broken} bent ${st.damage.bent}` : ''}  → ${png}`);
}
process.exit(failed ? 1 : 0);
