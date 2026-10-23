// End-to-end check of motion.html in headless Chrome (WebGPU): serves this
// directory, opens each scene, runs its __check() script frame-locked and
// asserts what the native tests assert (tests/ccd_tunnel.rs, character.rs,
// queries.rs), then saves a screenshot per scene.
//
//   cargo run -p engine-webgpu --release --bin wgsl_export
//   node engine/crates/engine-webgpu/motion-check.mjs [--out dir] [scene ...]
//
// Needs Chrome (CHROME=… to override). Exits non-zero on any failed check.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(ROOT, 'recordings');
const only = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
mkdirSync(OUT, { recursive: true });
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');

const TYPES = { '.glb': 'model/gltf-binary', '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'motion-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} try { rmSync(profile, { recursive: true, force: true }); } catch {} server.close(); };
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
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') console.error('[page]', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('[console]', m.params.args.map((a) => a.value ?? a.description).join(' '));
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || 'evaluate failed');
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });

async function open(query) {
  await send('Page.navigate', { url: `${base}/motion.html?${query}` });
  for (let i = 0; i < 1200; i++) {
    const s = await evaluate('JSON.stringify({ready: !!window.__ready, failed: window.__failed || null})');
    const st = s ? JSON.parse(s) : {};
    if (st.failed) throw new Error(`page failed: ${st.failed}`);
    if (st.ready) return;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`page did not load: ${query}`);
}
async function shot(name) {
  const r = await send('Page.captureScreenshot', { format: 'png' });
  const file = join(OUT, `${name}.png`);
  writeFileSync(file, Buffer.from(r.result.data, 'base64'));
  return file;
}

const failures = [];
const expect = (ok, msg) => { if (!ok) failures.push(msg); console.log(`${ok ? '  ok ' : '  FAIL'} ${msg}`); };

const CHECKS = {
  async ccd() {
    for (const speed of [60, 140, 250]) {
      await open(`scene=ccd&speed=${speed}&autofire=0`);
      const r = await evaluate('window.__check().then((r) => JSON.stringify(r))', true).then(JSON.parse);
      console.log(`ccd ${speed} m/s:`);
      for (const L of r.lanes) {
        console.log(`  ${L.name.padEnd(20)} off: ${L.control.padEnd(16)} on: ${L.ccd} (z ${L.ccdZ.toFixed(3)})`);
        expect(L.ccd !== 'TUNNELLED' && L.ccd !== 'flying', `${speed} m/s ${L.name}: CCD ball did not tunnel (${L.ccd})`);
      }
      console.log(`  debris through the slab: off ${r.showerThrough.control}/49, on ${r.showerThrough.ccd}/49; overflow ${r.counters.overflow}`);
      expect(r.showerThrough.ccd === 0, `${speed} m/s: no CCD debris through the slab (${r.showerThrough.ccd})`);
      expect(r.counters.overflow === 0, `${speed} m/s: no overflow`);
      if (speed === 250) {
        const tunnelled = r.lanes.filter((L) => L.control === 'TUNNELLED').length;
        expect(tunnelled >= 3, `250 m/s: the control balls tunnel (${tunnelled} of ${r.lanes.length} lanes)`);
        console.log(`  screenshot ${await shot('motion_ccd_250')}`);
      }
    }
  },
  async character() {
    await open('scene=character&autofire=0');
    const r = await evaluate('window.__check().then((r) => JSON.stringify(r))', true).then(JSON.parse);
    console.log(`character: lane ${r.atLane.map((x) => x.toFixed(2))}, top ${r.top.position.map((x) => x.toFixed(2))} ${r.top.ground}`);
    console.log(`  ground states: ${r.grounds.join(' ')}`);
    console.log(`  events: ${r.events.join(' | ')}`);
    expect(r.top.position[0] > 6.5 && Math.abs(r.top.position[1] - (1.8 + 0.92)) < 0.05, `climbed the six 0.3 m steps (${r.top.position})`);
    const steep = r.grounds.filter((g) => g === 'steep').length;
    expect(steep === 0, `no steep-ground flicker on the stairs (${steep})`);
    expect(r.events.some((e) => e.includes('landed')), 'jumped and landed (event)');
    expect(r.counters.overflow === 0, 'no overflow');
    console.log(`  screenshot ${await shot('motion_character')}`);
  },
  async bullets() {
    await open('scene=bullets&autofire=0');
    const r = await evaluate('window.__check().then((r) => JSON.stringify(r))', true).then(JSON.parse);
    const hits = r.hitLines;
    console.log(`bullets: ${hits.length} hits (${hits.slice(-3).join(' | ')}); projectiles ${JSON.stringify(r.projectiles.map((p) => ({ x: p.position.map((v) => +v.toFixed(2)), alive: p.alive, hits: p.hits })))}`);
    expect(r.projectiles.length === 3 && r.projectiles.every((p) => p.hits >= 5), `every bullet crossed the five walls (${r.projectiles.map((p) => p.hits)})`);
    expect(hits.length >= 15, `hit events reported (${hits.length})`);
    const pushed = r.targets.filter((t) => t && Math.hypot(...t.v) > 0.01).length;
    expect(pushed >= 1, `bullets pushed the targets (${pushed})`);
    console.log(`  screenshot ${await shot('motion_bullets')}`);
  },
  // demo.html's own schedule with CCD smash balls (default) and without (?ccd=0):
  // both run the wall smash to step 180 with no GPU error or overflow.
  async demo() {
    const runs = {};
    for (const ccd of ['1', '0']) {
      await send('Page.navigate', { url: `${base}/demo.html?scene=wall&stopAt=180&ccd=${ccd}` });
      let status = '', err = '';
      for (let i = 0; i < 1800; i++) {
        const st = JSON.parse(await evaluate(`JSON.stringify({ s: document.getElementById('status')?.textContent || '', e: document.getElementById('err')?.textContent || '' })`) || '{}');
        status = st.s || ''; err = st.e || '';
        if (err || status.includes('stopped at step')) break;
        await new Promise((r) => setTimeout(r, 100));
      }
      const pos = err ? null : await evaluate('window.__bodyPositions().then((p) => JSON.stringify(p))', true).then(JSON.parse);
      const n = pos ? pos.positions.length / 3 : 0;
      const ball = pos ? pos.positions.slice(3 * (n - 1), 3 * n) : null;
      runs[ccd] = { status, err, ball };
      console.log(`demo wall, ccd=${ccd}: ${err ? `ERROR ${err.slice(0, 300)}` : status}`);
      if (ball) console.log(`  ball at (${ball.map((x) => x.toFixed(2)).join(', ')})`);
      expect(!err && status.includes('stopped at step 180') && status.includes('ov 0x0'), `demo wall ccd=${ccd}: runs to step 180 without error or overflow`);
      if (ccd === '1') console.log(`  screenshot ${await shot('demo_wall_ccd')}`);
    }
    // The 30 m/s ball smashes through the wall either way (it is too slow to tunnel).
    if (runs['1'].ball && runs['0'].ball) expect(runs['1'].ball[2] > 5 && runs['0'].ball[2] > 5, `demo wall: the ball smashes through with and without CCD (ball z ${runs['1'].ball[2].toFixed(2)} vs ${runs['0'].ball[2].toFixed(2)})`);
  },
  // demo.html's ring regression (tests/ring_hold.rs in the browser): CCD off (the
  // check's default, its native baseline) and on (?ccd=1, the smash balls CCD).
  async ring() {
    for (const q of ['', '&ccd=1']) {
      await send('Page.navigate', { url: `${base}/demo.html?scene=ring110k&check=ring${q}` });
      let status = '', err = '';
      for (let i = 0; i < 6000; i++) {
        const st = JSON.parse(await evaluate(`JSON.stringify({ s: document.getElementById('status')?.textContent || '', e: document.getElementById('err')?.textContent || '' })`) || '{}');
        status = st.s || ''; err = st.e || '';
        if (err || status.startsWith('PASS')) break;
        await new Promise((r) => setTimeout(r, 100));
      }
      console.log(`ring regression${q || ' (ccd off)'}: ${err ? `ERROR ${err.slice(0, 400)}` : status}`);
      expect(!err && status.startsWith('PASS'), `ring regression${q} passes`);
    }
  },
  // vehicle.html (world-host.mjs): CCD cannonballs fired into the destruction scene.
  async vehicle() {
    await send('Page.navigate', { url: `${base}/vehicle.html?scene=destruction` });
    for (let i = 0; i < 1800 && !(await evaluate('!!window.__step || !!document.getElementById("err")?.textContent')); i++)
      await new Promise((r) => setTimeout(r, 100));
    let r = await evaluate('window.__step(60).then((r) => JSON.stringify(r.counters))', true).then(JSON.parse);
    const key = async (code) => { for (const type of ['keyDown', 'keyUp']) await send('Input.dispatchKeyEvent', { type, code, key: 'b', windowsVirtualKeyCode: 66 }); };
    for (let k = 0; k < 4; k++) { await key('KeyB'); r = await evaluate('window.__step(30).then((r) => JSON.stringify(r.counters))', true).then(JSON.parse); }
    const err = await evaluate('document.getElementById("err")?.textContent || ""');
    const ccd = await evaluate('JSON.stringify(window.__world?.motion || null)');
    console.log(`vehicle destruction, 4 cannonballs: counters ${JSON.stringify(r)}, motion ${ccd}${err ? `, ERROR ${err.slice(0, 300)}` : ''}`);
    expect(!err && r.overflow === 0, 'vehicle: CCD cannonballs fired with no GPU error or overflow');
    expect(ccd && JSON.parse(ccd).ccd === true, 'vehicle: the scene schedules the CCD passes');
    console.log(`  screenshot ${await shot('vehicle_ccd_cannon')}`);
  },
  // Phase-through acceptance in the browser lab (tests/fracture_tower_ccd.rs): the
  // vibe-city fracturedTower pack (bun run export.ts in engine/scenes/vibe-city),
  // CCD balls at 16× and 64× the shot speed from 18 directions; none may cross a
  // still piece.
  async tower() {
    await send('Page.navigate', { url: `${base}/fracture/index.html` });
    const ready = async () => {
      for (let i = 0; i < 3000; i++) {
        if (await evaluate('!!window.__FRACTURE_DEMO__?.ready')) return;
        await new Promise((r) => setTimeout(r, 100));
      }
      throw new Error('fracture lab did not load');
    };
    await ready();
    for (const pack of ['fracturedTower', 'reinforcedTower']) {
      await evaluate(`window.__FRACTURE_DEMO__.load("vc:${pack}")`, true);
      await ready();
      const control = await evaluate(`import('./regression.mjs').then((m) => m.runPhaseThroughRegression(undefined, { scales: [64], ccd: false, limit: 4 })).then((r) => JSON.stringify(r))`, true).then(JSON.parse);
      console.log(`lab ${pack} control (CCD off, 64×, 4 shots): ${control.phaseThroughs} phase-throughs`);
      expect(control.phaseThroughs > 0, `lab ${pack} control: without CCD the detector finds phase-throughs`);
      const r = await evaluate(`import('./regression.mjs').then((m) => { const lines = []; return m.runPhaseThroughRegression(undefined, { log: (l) => lines.push(l) }).then((r) => JSON.stringify({ ...r, lines })); })`, true).then(JSON.parse);
      for (const l of r.lines) if (!l.includes(': 0 phase-throughs')) console.log(`  ${l}`);
      console.log(`lab ${pack}: ${r.shots} CCD shots, ${r.failed.length} failed`);
      expect(r.passed, `lab ${pack}: no CCD ball phases through a still piece (${r.shots} shots)`);
      await evaluate('window.__FRACTURE_DEMO__.stepFrames(20)', true);
      console.log(`  screenshot ${await shot(`motion_lab_${pack}_ccd`)}`);
    }
  },
  // The fracture lab's own step plan (engine-fracture plan.rs) with CCD balls:
  // needs the lab's wasm (wasm-pack build engine/crates/engine-fracture-wasm …).
  async lab() {
    await send('Page.navigate', { url: `${base}/fracture/index.html` });
    const ready = async () => {
      for (let i = 0; i < 1200; i++) {
        if (await evaluate('!!window.__FRACTURE_DEMO__?.ready')) return;
        await new Promise((r) => setTimeout(r, 100));
      }
      throw new Error('fracture lab did not load');
    };
    await ready();
    await evaluate('window.__FRACTURE_DEMO__.load("arch")', true);
    await ready();
    const r = await evaluate(`import('./regression.mjs').then((m) => m.runCcdShotRegression()).then((r) => JSON.stringify(r))`, true).then(JSON.parse);
    console.log(`lab arch, 300 m/s ball: control reached z ${r.control.minZ.toFixed(2)} and moved the arch ${r.control.moved.toFixed(3)} m; CCD reached z ${r.ccd.minZ.toFixed(2)}, moved ${r.ccd.moved.toFixed(3)} m (arch from z ${r.back.toFixed(2)})`);
    expect(r.passed, 'lab: the CCD ball knocks the crown out, the plain ball tunnels');
    await evaluate('window.__FRACTURE_DEMO__.stepFrames(30)', true);
    console.log(`  screenshot ${await shot('motion_lab_ccd_arch')}`);
  },
};

for (const name of only.length ? only : Object.keys(CHECKS)) {
  try { await CHECKS[name](); } catch (e) { failures.push(`${name}: ${e.message}`); console.error(e); }
}
ws.close();
console.log(failures.length ? `\n${failures.length} FAILED:\n${failures.join('\n')}` : '\nall motion checks passed');
process.exit(failures.length ? 1 : 0);
