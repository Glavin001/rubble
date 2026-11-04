#!/usr/bin/env node
// The cross-host check (engine/docs/lab-world-plan.md, Checks): one scenario, a pack's 2×2
// tower block under a bond model with the block's volley, 480 steps at 60 Hz, run three ways
// over the same Rust (`rubble::lab`):
//   - native Rust (engine-world tests/lab_api.rs `cross_host_tower_block_reference`);
//   - C, through rubble.h (engine-ffi examples/lab.c);
//   - the web page (the built site in headless Chrome, its own controls).
// All three must fire the same shots and report the same bodies, bonds broken and crushed
// chunks within the tolerance native runs hold to each other (native wgpu worlds are not
// bit-reproducible run to run: engine-world tests/stress_models.rs). Prints a table.
//
//   bash engine/scripts/build-web.sh
//   node --experimental-websocket lab-cross-host.mjs [--bonds blast_gpu] [--pack fracturedTower.json] [--steps 480]
import { execFileSync, spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE = join(HERE, '..', '..', '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };
const BONDS = opt('--bonds', 'blast_gpu');
const STEPS = Number(opt('--steps', '480'));
const PACK = opt('--pack', join(HERE, '..', 'generated', 'vibe-city', 'fracturedTower.json'));
const SITE = opt('--site', join(ENGINE, 'web-dist'));
// Bodies and bonds broken may differ by this fraction between hosts (stress_models.rs holds
// World to the lab within a third; runs of one host differ less).
const TOLERANCE = Number(opt('--tolerance', '0.25'));
if (!existsSync(PACK)) { console.error(`no pack ${PACK}: engine/scenes/vibe-city, bun run export.ts fracturedTower`); process.exit(2); }
const parse = (line) => Object.fromEntries(line.trim().split(/\s+/).filter((kv) => kv.includes('=')).map((kv) => { const [k, v] = kv.split('='); return [k, isNaN(Number(v)) ? v : Number(v)]; }));
const results = {};

// Native Rust.
{
  const out = execFileSync('cargo', ['test', '--release', '-p', 'engine-world', '--test', 'lab_api', 'cross_host', '--', '--nocapture'],
    { cwd: ENGINE, encoding: 'utf8', env: { ...process.env, LAB_CROSS_HOST_PACK: PACK, LAB_CROSS_HOST_BONDS: BONDS, LAB_CROSS_HOST_STEPS: String(STEPS) }, stdio: ['ignore', 'pipe', 'inherit'] });
  const line = out.split('\n').find((l) => l.startsWith('LAB_STATS'));
  if (!line) throw new Error(`native: no LAB_STATS line\n${out}`);
  results.native = parse(line);
}

// C, through rubble.h.
{
  execFileSync('cargo', ['build', '--release', '-p', 'engine-ffi'], { cwd: ENGINE, stdio: ['ignore', 'ignore', 'inherit'] });
  const target = process.env.CARGO_TARGET_DIR || join(ENGINE, 'target');
  const dir = mkdtempSync(join(tmpdir(), 'lab-c-'));
  const exe = join(dir, 'lab');
  execFileSync('cc', ['-std=c11', '-I', join(ENGINE, 'crates/engine-ffi/include'), join(ENGINE, 'crates/engine-ffi/examples/lab.c'),
    '-L', join(target, 'release'), '-lrubble_ffi', `-Wl,-rpath,${join(target, 'release')}`, '-o', exe], { stdio: 'inherit' });
  const out = execFileSync(exe, [PACK, BONDS, String(STEPS)], { encoding: 'utf8' });
  rmSync(dir, { recursive: true, force: true });
  results.c = parse(out);
}

// The web page.
{
  if (!existsSync(join(SITE, 'fracture', 'index.html'))) throw new Error(`no built site at ${SITE}: run engine/scripts/build-web.sh`);
  const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css', '.glb': 'model/gltf-binary', '.wgsl': 'text/plain' };
  const server = createServer((req, res) => {
    let path = normalize(join(SITE, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
    if (existsSync(path) && statSync(path).isDirectory()) path = join(path, 'index.html');
    if (!path.startsWith(SITE) || !existsSync(path)) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' }).end(readFileSync(path));
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const CHROME = process.env.CHROME || (process.platform === 'darwin' ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
  const profile = mkdtempSync(join(tmpdir(), 'lab-cross-chrome-'));
  const port = 9300 + Math.floor(Math.random() * 500);
  const chrome = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', '--window-size=1280,720', 'about:blank'], { stdio: 'ignore' });
  try {
    let targets = [];
    for (let i = 0; i < 100 && !targets.length; i++) {
      try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).filter((t) => t.type === 'page'); } catch {}
      if (!targets.length) await new Promise((r) => setTimeout(r, 100));
    }
    const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
    await new Promise((r) => ws.addEventListener('open', r));
    let id = 0; const waiting = new Map();
    ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
    const send = (method, params = {}) => new Promise((r) => { const k = ++id; waiting.set(k, r); ws.send(JSON.stringify({ id: k, method, params })); });
    const evaluate = async (expression) => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 900_000 }); if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text); return r.result?.result?.value; };
    await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/fracture/?scene=vc-towers&bonds=${BONDS === 'blast_gpu' ? 'blast' : BONDS}&hz=60` });
    const st = await evaluate(`(async () => {
      for (let i = 0; i < 600 && !window.__FRACTURE_DEMO__; i++) await new Promise((r) => setTimeout(r, 100));
      const d = window.__FRACTURE_DEMO__, status = document.getElementById('status');
      for (let i = 0; i < 1800 && !d.stressReady && !status.classList.contains('error'); i++) await new Promise((r) => setTimeout(r, 100));
      if (!d.stressReady) throw new Error(status.textContent);
      document.getElementById('ball-ccd').checked = true;
      document.getElementById('volley').click();
      let st; for (let i = 0; i < ${STEPS}; i++) st = await d.stressAdvance(1000 / 60);
      await d.simulation.flush();
      return d.simulation.stats();
    })()`);
    results.web = { steps: st.steps, shots: st.shots, bodies: st.bodies, broken: st.bonds - st.bondsIntact, crushed: st.crushed, bonds: st.bonds, chunks: st.chunks };
  } finally {
    try { chrome.kill('SIGKILL'); } catch {}
    try { rmSync(profile, { recursive: true, force: true, maxRetries: 3 }); } catch {}
    server.close();
  }
}

const keys = ['steps', 'shots', 'chunks', 'bonds', 'bodies', 'broken', 'crushed'];
console.log(`${BONDS} · ${PACK.split('/').pop()} tower block · volley · ${STEPS} steps`);
console.log(['host', ...keys].map((k) => k.padStart(8)).join(''));
for (const [host, r] of Object.entries(results)) console.log([host, ...keys.map((k) => r[k])].map((v) => String(v).padStart(8)).join(''));
const problems = [];
const ref = results.native;
for (const [host, r] of Object.entries(results)) {
  for (const k of ['steps', 'shots', 'chunks', 'bonds']) if (r[k] !== ref[k]) problems.push(`${host} ${k} ${r[k]}, native ${ref[k]}`);
  for (const k of ['bodies', 'broken', 'crushed']) {
    const a = r[k], b = ref[k];
    if (Math.abs(a - b) > TOLERANCE * Math.max(a, b, 1)) problems.push(`${host} ${k} ${a}, native ${b} (beyond ${TOLERANCE * 100}%)`);
  }
}
for (const p of problems) console.log(`FAIL ${p}`);
console.log(problems.length ? 'the hosts differ' : `the three hosts agree within ${TOLERANCE * 100}%`);
process.exit(problems.length ? 1 : 0);
