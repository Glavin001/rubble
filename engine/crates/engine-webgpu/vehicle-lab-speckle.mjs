// The vehicle lab's crushed car, the same simulated state drawn several ways: stills
// that show which part of the drawing puts dark pixels into the paint.
//
//   engine/scripts/with-gpu-lock.py --limit 900 -- \
//     node vehicle-lab-speckle.mjs [--out dir] [--secs 60 | --secs '20,40,60'] [--query 'mode=mountain&car=gen&cars=1'] \
//       [--views 'yaw,pitch,dist;…'] [--variants a,b,…] [--size 1920x1080] [--realtime] [--root dir] [--take files]
//
// The page runs frame-locked (`?record=1`) to --secs simulated seconds, is paused, and
// then every variant is drawn from every view with the World untouched: the variants are
// other builds of the car's draw shader (showcase-car-render.mjs, label `showcase-draw`),
// made beside the page's own by a hook on createShaderModule / createRenderPipeline and
// swapped in at setPipeline. `page` is the page's own shader, unchanged.
//
// Serves this directory, falling back to engine/web-dist for what the build makes
// (generated/, __mods/): run engine/scripts/build-web.sh first. Needs Chrome.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const args0 = process.argv.slice(2);
// --root <dir>: another checkout's engine-webgpu directory (its page, wasm and __mods).
const ROOT = args0.includes('--root') ? realpathSync(args0[args0.indexOf('--root') + 1]) : HERE;
const DIST = join(ROOT, '..', '..', 'web-dist');
// --take 'a.mjs,b.mjs': with --root, these files from this directory instead (this
// checkout's renderer on the other checkout's page and wasm).
const TAKE = args0.includes('--take') ? args0[args0.indexOf('--take') + 1].split(',') : [];
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const OUT = opt('--out', join(HERE, '..', '..', '..', 'out', 'speckle'));
const [W, H] = opt('--size', '1920x1080').split('x').map(Number);
// --secs '20,40,60': stills at each of these simulated times of the one run.
const TIMES = opt('--secs', '60').split(',').map(Number);
const QUERY = opt('--query', 'mode=mountain&car=gen&cars=1');
const VIEWS = opt('--views', '0.9,0.45,6;2.5,0.7,5;4.0,0.25,6;5.4,1.1,5').split(';').map((v) => v.split(',').map(Number));
// --realtime: the page's own real-time loop (no `?record=1`), as a visitor runs it.
const REALTIME = args.includes('--realtime');
const CHROME = process.env.CHROME || (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });
setTimeout(() => { console.error('time limit'); process.exit(9); }, Number(opt('--limit', '14')) * 60_000).unref();

// Variants: [name, edits of the draw shader (find, replace)…, pipeline primitive overrides].
// Every `find` must be in the shader (a moved line fails the run, not the comparison).
// The debug colours are mixed over the shaded one, so every binding stays in use.
const VARIANTS = {
  page: { edits: [] },
  // The shader before closed solids' back faces were left out (the dark patches).
  before: { edits: [['if (i.closed != 0u && i.torn < CLOSED_TEAR) { discard; }', '']] },
  // No scratches (paint worn to primer and steel where the car slid).
  noScratch: { edits: [['let scr = i.wear.y;', 'let scr = 0.0;']] },
  // No dark inside for a body panel's back face.
  noInside: { edits: [['if (!front && i.mat == 0u && dot(n, v) < 0.0) {', 'if (false) {']] },
  // Back faces not drawn at all.
  cullBack: { edits: [], primitive: { cullMode: 'back' } },
  // What is drawn where: front faces green, back faces red, brighter for paint (material 0).
  faces: { edits: [['  return shade(i, front);\n}', '  let k = select(0.35, 1.0, i.mat == 0u);\n  return vec4f(mix(shade(i, front).rgb, select(vec3f(k, 0.0, 0.0), vec3f(0.0, k, 0.0), front), 0.999), 1.0);\n}']] },
  // The material id as a colour.
  material: { edits: [['  return shade(i, front);\n}', '  let m = f32(i.mat);\n  return vec4f(mix(shade(i, front).rgb, vec3f(fract(m * 0.37), fract(m * 0.61 + 0.2), fract(m * 0.83 + 0.4)), 0.999), 1.0);\n}']] },
  // The tear field with nothing cut away: blue whole, through green at the cut (0.5), to
  // red torn; white where the page's shader discards.
  tear: { edits: [['if (i.torn > 0.5) { discard; }', ''], ['  return shade(i, front);\n}', '  let t = clamp(i.torn, 0.0, 1.0);\n  let heat = select(vec3f(2.0 * t, 2.0 * t, 1.0 - 2.0 * t), vec3f(1.0, 1.0, 1.0), t > 0.5);\n  return vec4f(mix(shade(i, front).rgb, heat, 0.999), 1.0);\n}']] },
  // Nothing cut away at a tear (the page's shading otherwise).
  noCut: { edits: [['if (i.torn > 0.5) { discard; }', '']] },
  // The wear fields: red the tear, green plastic strain (×10), blue paint worn (µm / 400).
  wear: { edits: [['  return shade(i, front);\n}', '  return vec4f(mix(shade(i, front).rgb, vec3f(i.torn, min(i.wear.x * 10.0, 1.0), min(i.wear.y / 400.0, 1.0)), 0.999), 1.0);\n}']] },
};
const NAMES = opt('--variants', Object.keys(VARIANTS).join(',')).split(',');
for (const n of NAMES) if (!VARIANTS[n]) { console.error(`no variant ${n}: ${Object.keys(VARIANTS).join(', ')}`); process.exit(2); }

const HOOK = `(() => {
  if (!self.GPUDevice) return;
  const VARIANTS = ${JSON.stringify(Object.fromEntries(NAMES.map((n) => [n, VARIANTS[n]])))};
  window.__variant = 'page'; window.__variantErrors = [];
  // The page's pipeline has an automatic layout, whose bind groups fit no other
  // pipeline: each variant gets its own copy of every bind group made for the page's.
  const code = new WeakMap(), swaps = new WeakMap(), layoutOf = new WeakMap(), groups = new WeakMap();
  const sm = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (d) { const m = sm.call(this, d); if (d.label === 'showcase-draw') code.set(m, d.code); return m; };
  const rp = GPUDevice.prototype.createRenderPipeline;
  GPUDevice.prototype.createRenderPipeline = function (d) {
    const p = rp.call(this, d);
    const src = d.label === 'showcase-draw' && code.get(d.fragment?.module);
    if (!src) return p;
    const made = {};
    for (const [name, v] of Object.entries(VARIANTS)) {
      if (name === 'page') continue;
      let c = src;
      for (const [find, to] of v.edits) { if (!c.includes(find)) window.__variantErrors.push(name + ': not in the shader: ' + find); c = c.replace(find, to); }
      const module = sm.call(this, { label: 'speckle-' + name, code: c });
      made[name] = rp.call(this, { ...d, label: 'speckle-' + name, vertex: { ...d.vertex, module }, fragment: { ...d.fragment, module }, primitive: { ...d.primitive, ...(v.primitive || {}) } });
    }
    swaps.set(p, made);
    return p;
  };
  const gl = GPURenderPipeline.prototype.getBindGroupLayout;
  GPURenderPipeline.prototype.getBindGroupLayout = function (k) { const l = gl.call(this, k); if (swaps.has(this)) layoutOf.set(l, { pipe: this, k }); return l; };
  const bg = GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup = function (d) {
    const g = bg.call(this, d);
    const from = layoutOf.get(d.layout);
    if (from) groups.set(g, Object.fromEntries(Object.entries(swaps.get(from.pipe)).map(([name, p]) => [name, bg.call(this, { ...d, layout: gl.call(p, from.k) })])));
    return g;
  };
  const sp = GPURenderPassEncoder.prototype.setPipeline;
  GPURenderPassEncoder.prototype.setPipeline = function (p) { return sp.call(this, swaps.get(p)?.[window.__variant] ?? p); };
  const sb = GPURenderPassEncoder.prototype.setBindGroup;
  GPURenderPassEncoder.prototype.setBindGroup = function (k, g, ...rest) { return sb.call(this, k, groups.get(g)?.[window.__variant] ?? g, ...rest); };
})()`;

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png' };
const server = createServer((req, res) => {
  const p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const roots = TAKE.includes(p.slice(1)) ? [HERE] : [ROOT, DIST];
  const path = roots.map((r) => normalize(join(r, p))).find((f, i) => f.startsWith(roots[i]) && existsSync(f) && !statSync(f).isDirectory());
  if (!path) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  if (req.method === 'HEAD') { res.end(); return; }
  res.end(readFileSync(realpathSync(path)));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'vehicle-lab-speckle-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', `--window-size=${W},${H}`,
  '--no-first-run', '--no-default-browser-check', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'], { stdio: 'ignore' });
process.on('exit', () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {} });

let targets = [];
for (let i = 0; i < 100 && !targets.length; i++) {
  try { targets = (await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
  if (!targets.length) await new Promise((r) => setTimeout(r, 100));
}
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nextId = 0;
const waiting = new Map(), pageErrors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
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
await send('Page.addScriptToEvaluateOnNewDocument', { source: HOOK });
await send('Page.navigate', { url: `${base}/vehicle-lab.html?${QUERY}${REALTIME ? '' : '&record=1'}&clean=1` });
for (const t0 = Date.now(); ;) {
  await new Promise((r) => setTimeout(r, 250));
  const err = await evaluate('window.__failed || ""').catch(() => '');
  if (err) { console.error(`page error: ${err}\n${pageErrors.join('\n')}`); process.exit(1); }
  if (await evaluate('window.__ready === true').catch(() => false)) break;
  if (Date.now() - t0 > 240_000) { console.error(`the page did not start\n${pageErrors.join('\n')}`); process.exit(1); }
}
const hookErrors = await evaluate('window.__variantErrors');
if (hookErrors.length) { console.error(hookErrors.join('\n')); process.exit(1); }

// Run: several frames a call (one round trip each would dominate).
let t = 0;
const runTo = async (secs) => {
  const wall0 = Date.now();
  while (REALTIME && t < secs) {
    await new Promise((r) => setTimeout(r, 250));
    if (await evaluate('window.__failed || ""')) { console.error(`failed: ${await evaluate('window.__failed')}`); process.exit(1); }
    t = await evaluate('window.__state().t');
    if (Date.now() - wall0 > (secs + 30) * 4000) { console.error(`too slow: ${t} s simulated`); process.exit(1); }
  }
  while (t < secs) {
    const st = await evaluate(`(async () => { let s; for (let i = 0; i < 30; i++) { s = await window.__rec.frame(60); if (s.failed || s.t >= ${secs}) break; } return s; })()`);
    if (st.failed) { console.error(`failed at ${st.t} s: ${st.failed}`); process.exit(1); }
    t = st.t;
  }
  console.log(`${t.toFixed(2)} s simulated (${((Date.now() - wall0) / 1000).toFixed(0)} s of wall time for this stretch)`);
};
const pause = () => evaluate(`document.getElementById('pause').click()`);

// A view: the orbit camera at (yaw, pitch, distance), set through the page's own drag
// and wheel handlers (which also stop the recorder's slow orbit).
const canvasEvent = (type, init) => evaluate(`document.querySelector('canvas').dispatchEvent(new ${type === 'wheel' ? 'WheelEvent' : 'PointerEvent'}('${type}', ${JSON.stringify({ bubbles: true, cancelable: true, pointerId: 1, button: 0, ...init })}))`);
let yaw = REALTIME ? 0.9 : null, pitch = 0.3, dist = 8.5;
const setView = async ([y, p, d]) => {
  await canvasEvent('pointerdown', { clientX: 400, clientY: 400 });
  if (yaw === null) {
    // The first drag stops the slow orbit at an unknown yaw: pitch to the floor of its
    // range so the next move's pitch is known; yaw is then relative to this one.
    await canvasEvent('pointermove', { clientX: 400, clientY: 400 - 2000 });
    await canvasEvent('pointerup', { clientX: 400, clientY: -1600 });
    await canvasEvent('pointerdown', { clientX: 400, clientY: 400 });
    yaw = 0; pitch = -0.2;
  }
  const dx = Math.round((yaw - y) / 0.005), dy = Math.round((p - pitch) / 0.005);
  await canvasEvent('pointermove', { clientX: 400 + dx, clientY: 400 + dy });
  await canvasEvent('pointerup', { clientX: 400 + dx, clientY: 400 + dy });
  yaw -= dx * 0.005; pitch += dy * 0.005;
  await canvasEvent('wheel', { deltaY: Math.log(d / dist) / 0.001 });
  dist = d;
};
const grab = async (file) => {
  // Two frames: the camera and the level of detail settle on the new view.
  if (REALTIME) await new Promise((r) => setTimeout(r, 400));
  else await evaluate('(async () => { await window.__rec.frame(60); await window.__rec.frame(60); })()');
  const s = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(file, Buffer.from(s.result.data, 'base64'));
};
for (const secs of TIMES) {
  await runTo(secs);
  await pause();
  const tag = TIMES.length > 1 ? `t${secs}-` : '';
  for (let v = 0; v < VIEWS.length; v++) {
    await setView(VIEWS[v]);
    for (const name of NAMES) {
      await evaluate(`window.__variant = ${JSON.stringify(name)}`);
      await grab(join(OUT, `${tag}view${v}-${name}.png`));
    }
    await evaluate(`window.__variant = 'page'`);
  }
  await pause();
}
const stats = await evaluate('window.__stats()');
writeFileSync(join(OUT, 'stats.json'), JSON.stringify({ query: QUERY, simulatedSeconds: t, views: VIEWS, variants: NAMES, stats }, null, 1));
console.log(`wrote ${TIMES.length * VIEWS.length * NAMES.length} stills to ${OUT}`);
if (pageErrors.length) { console.error(`console errors:\n${pageErrors.slice(0, 10).join('\n')}`); process.exit(1); }
process.exit(0);
