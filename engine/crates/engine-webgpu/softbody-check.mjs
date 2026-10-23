#!/usr/bin/env node
// End-to-end check of the soft-body demo in Chrome WebGPU: loads softbody.html for
// each scene, steps it through the page's `__step` hook, checks the outcome
// (driving, crushing, breaking, two-way contact) and saves a screenshot per scene.
//
//   cargo run -p engine-webgpu --release --bin wgsl_export      # WGSL + wasm first
//   node --experimental-websocket softbody-check.mjs [scene ...] [--out DIR]
//   SOFT_QUERY='iters=5&sweeps=2' node --experimental-websocket softbody-check.mjs  # other solver settings
//
// Needs Chrome (CHROME env to override). Exits non-zero when a check fails.
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(ROOT, 'recordings', 'softbody');
const only = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
mkdirSync(OUT, { recursive: true });

const sum = (d, k) => d.reduce((n, x) => n + x[k], 0);
// Each check: query, steps (240 Hz), optional input, camera, and an assertion on the result.
const CHECKS = {
  drive: {
    query: 'scene=yard',
    run: async (page) => {
      const p0 = (await page.step(60, { forward: 0, steer: 0 })).player.p;
      const r = await page.step(360, { forward: 1, steer: 0 }); // 1.5 s: clear of the poles and crates
      const moved = Math.hypot(r.player.p[0] - p0[0], r.player.p[2] - p0[2]);
      const speed = Math.hypot(...r.player.v) * 3.6;
      const r2 = await page.step(300, { forward: 0.5, steer: 1 });
      const turned = Math.abs(r2.player.p[0] - r.player.p[0]);
      await page.view({ chase: true });
      return { ok: moved > 5 && speed > 20 && turned > 2 && r2.damage[0].broken === 0,
        msg: `drove ${moved.toFixed(1)} m to ${speed.toFixed(0)} km/h, turned ${turned.toFixed(1)} m sideways, player beams bent ${r2.damage[0].yielded} broken ${r2.damage[0].broken}` };
    },
  },
  wall: {
    query: 'scene=wall',
    run: async (page) => {
      const r = await page.step(360);
      const d = r.damage[0];
      return { ok: d.yielded > 80, msg: `${d.yielded} beams bent, ${d.broken} broken, ${r.breaks} break events` };
    },
  },
  // A front-corner T-bone pulls the bumper sideways: its mounts tear (one detacher group).
  tear: {
    query: 'scene=tbone&speed=60&at=1.6',
    run: async (page) => {
      const r = await page.step(420);
      const d = r.damage[0];
      return { ok: d.broken >= 10, msg: `${d.yielded} beams bent, ${d.broken} broken, ${r.breaks} break events` };
    },
  },
  // Traffic: self-driving cars spread out and collide (the AI drives in rAF time,
  // so this lets the page run in real time for a few seconds).
  traffic: {
    query: 'scene=traffic&cars=20',
    run: async (page) => {
      await page.eval(`new Promise((r) => setTimeout(r, 8000))`);
      const r = await page.step(1);
      const moved = await page.eval(`window.__particles(0).then((p) => p.length)`);
      const bent = r.damage.reduce((n, d) => n + d.yielded, 0);
      return { ok: moved > 0 && r.damage.length === 20, msg: `20 cars, ${bent} beams bent across them after 8 s` };
    },
  },
  // The other crash-test scenes run without blowing up and deform.
  // Traffic scale in real time (opt-in: node softbody-check.mjs traffic_scale):
  // frame rate and the share of real time the simulation keeps.
  ...Object.fromEntries([10, 20, 50, 100].map((n) => [`traffic_scale_${n}`, {
    query: `scene=traffic&cars=${n}`,
    optIn: 'TRAFFIC_SCALE',
    run: async (page) => {
      await page.step(240); // warm up, cars spread out (the AI drives in __step too)
      const ms = await page.eval(`(async () => { const t = performance.now(); await window.__step(480); return (performance.now() - t) / 480; })()`);
      return { ok: ms > 0, msg: `${n} cars: ${ms.toFixed(2)} ms per 240 Hz step (GPU-bound real-time capacity ${(100 * (1000 / 240) / ms).toFixed(0)}%, no rendering)` };
    },
  }])),
  // Frame budget (opt-in: BUDGET=1): physics ms per step at the fine and coarse
  // rates, rendering ms per frame, in the yard after the cars have settled.
  ...Object.fromEntries([['budget_240x10', 'scene=yard&hz=240&iters=10'], ['budget_60x8', 'scene=yard&hz=60&iters=8'], ['budget_yard_drive', 'scene=yard&drive=1'],
    ['budget_traffic20', 'scene=traffic&cars=20'], ['budget_traffic50', 'scene=traffic&cars=50'], ['budget_traffic100', 'scene=traffic&cars=100']].map(([k, q]) => [k, {
    query: process.env.BUDGET_SPLIT ? q + '&gputime' : q,
    optIn: 'BUDGET',
    run: async (page) => {
      // drive=1: the player car drives (throttle, a gentle turn) through the run.
      const input = q.includes('drive=1') ? '{ forward: 0.7, steer: 0.25 }' : 'null';
      await page.eval(`window.__step(240, ${input})`);
      const step = await page.eval(`(async () => { const t = performance.now(); await window.__step(240, ${input}); return (performance.now() - t) / 240; })()`);
      const render = await page.eval(`window.__renderBench(60)`);
      const info = await page.eval(`JSON.stringify({ counts: Array.from(window.__world.lastCounts), h: window.__world.dispatchHistogram() })`);
      if (process.env.BUDGET_VERBOSE) console.log(q, info);
      const live = await page.eval(`window.__live(4000)`);
      // CPU cost of encoding steps (JS + wasm plan + WebGPU encoding), no GPU wait.
      const cpu = await page.eval(`(() => { const w = window.__world; const d = window.__device; const enc = d.createCommandEncoder(); const t = performance.now(); for (let i = 0; i < 64; i++) w.encodeStep(enc); const ms = (performance.now() - t) / 64; const t2 = performance.now(); d.queue.submit([enc.finish()]); return [ms, performance.now() - t2]; })()`);
      if (process.env.BUDGET_SPLIT) {
        const split = await page.eval(`(async () => {
          const w = window.__world, out = {};
          const t = async (k) => { await window.__step(8); const t0 = performance.now(); await window.__step(64); out[k] = ((performance.now() - t0) / 64).toFixed(2); };
          await t('base');
          const it = w.params.iterations; w.params.iterations = 1; await t('iters1'); w.params.iterations = it;
          const acc = {};
          for (let r = 0; r < 8; r++) { const p = await w.profileStep(); for (const k in p) acc[k] = (acc[k] || 0) + p[k] / 8; }
          const tot = Object.values(acc).reduce((a, b) => a + b, 0);
          out.gpu = tot.toFixed(2);
          out.top = Object.entries(acc).sort((a, b) => b[1] - a[1]).slice(0, 14).map(([k, v]) => k + ' ' + v.toFixed(3)).join(', ');
          return JSON.stringify(out);
        })()`);
        console.log(q, 'split', split);
      }
      const modes = await page.eval(`window.__world.readSoftModes().then((m) => Array.from(m).join(''))`);
      console.log(q, 'modes', modes);
      console.log(q, 'cpu encode ms/step', cpu[0].toFixed(3), 'finish+submit ms/64', cpu[1].toFixed(2));
      return { ok: true, msg: `${q}: physics ${step.toFixed(2)} ms/step, render ${render.toFixed(2)} ms/frame; live ${live.fps.toFixed(0)} fps, ${(live.realTime * 100).toFixed(0)}% real time` };
    },
  }])),
  // The limbo bar shears the roof off: a piece-detached event names it.
  limbo: {
    query: 'scene=limbo&speed=80',
    run: async (page) => {
      await page.step(600);
      await page.step(60); // one more damage read for the last events
      const torn = await page.eval(`window.__tornOff.join(', ')`);
      return { ok: /roof/.test(torn), msg: `torn off: ${torn}` };
    },
  },
  ...Object.fromEntries(['sidepole&speed=60', 'rollover&speed=60', 'ramp&speed=90', 'drop&height=12&pitch=50', 'headon&speed=100&overlap=0.5', 'sandwich&speed=60', 'crusher'].map((q) => [q.split('&')[0] + (q.startsWith('headon') ? '_offset' : ''), {
    query: `scene=${q}`,
    run: async (page) => {
      const r = await page.step(600);
      const d = r.damage[0];
      const finite = await page.eval(`window.__particles().then((ps) => ps.every((p) => p.every(Number.isFinite)))`);
      return { ok: d.yielded > 5 && finite, msg: `${q.replace(/&/g, ' ')}: ${d.yielded} beams bent, ${d.broken} broken${finite ? '' : ', NON-FINITE positions'}` };
    },
  }])),
  pole: {
    query: 'scene=pole',
    run: async (page) => {
      const r = await page.step(360);
      const d = r.damage[0];
      return { ok: d.yielded > 10, msg: `${d.yielded} beams bent, ${d.broken} broken` };
    },
  },
  tbone: {
    query: 'scene=tbone',
    run: async (page) => {
      const r = await page.step(420);
      const d = r.damage[0];
      return { ok: d.yielded > 10, msg: `${d.yielded} beams bent, ${d.broken} broken by the side ram` };
    },
  },
  headon: {
    query: 'scene=headon',
    run: async (page) => {
      const r = await page.step(360);
      const [a, b] = r.damage;
      const pa = await page.eval('window.__particles(0)');
      const pb = await page.eval('window.__particles(1)');
      const cz = (ps) => ps.reduce((n, p) => n + p[2], 0) / ps.length;
      const gap = cz(pb) - cz(pa);
      return { ok: a.yielded > 10 && b.yielded > 10 && gap > 2, msg: `car A ${a.yielded} bent, car B ${b.yielded} bent, centres ${gap.toFixed(2)} m apart (no pass-through)` };
    },
  },
  beams: {
    query: 'scene=wall&beams=1',
    run: async (page) => {
      const r = await page.step(360);
      return { ok: sum(r.damage, 'yielded') > 40, msg: `beam stress view after the wall hit (${sum(r.damage, 'yielded')} bent)` };
    },
  },
  crates: {
    query: 'scene=crates',
    run: async (page) => {
      const r = await page.step(480);
      const awake = r.counters.awake;
      return { ok: true, msg: `${r.damage.length} lattice crates dropped, ${sum(r.damage, 'yielded')} beams bent, ${awake} bodies awake` };
    },
  },
  probe: {
    query: 'scene=headon',
    optIn: 'SOFT_PROBE',
    run: async (page) => {
      const { readFileSync } = await import('node:fs');
      const body = readFileSync(process.env.SOFT_PROBE, 'utf8');
      return await (new Function('page', `return (async () => { ${body} })()`))(page);
    },
  },
  // Close-ups of the head-on crush: flexbody skin, collision skin, beam view.
  headon_zoom: {
    query: 'scene=headon',
    run: async (page) => {
      await page.step(360);
      const views = [['flex', { skin: 'flex', beams: false, orbit: { yaw: 1.2, pitch: 0.25, dist: 3.2, target: [0.15, 0.7, 0] } }],
        ['flex_top', { skin: 'flex', beams: false, orbit: { yaw: 0.0, pitch: 1.2, dist: 3.5, target: [0.15, 0.7, 0] } }],
        ['collision', { skin: 'collision', beams: false, orbit: { yaw: 1.2, pitch: 0.25, dist: 3.2, target: [0.15, 0.7, 0] } }],
        ['beams', { skin: 'flex', beams: true, orbit: { yaw: 1.2, pitch: 0.25, dist: 3.2, target: [0.15, 0.7, 0] } }]];
      for (const [name, v] of views) {
        await page.view(v);
        await page.step(1);
        await page.shot(`headon_zoom_${name}`);
      }
      return { ok: true, msg: `close-ups: ${views.map((v) => v[0]).join(', ')}` };
    },
  },
  // Chrome step cost: the yard (3 cars, 3 crates: 387 particles, 2,190 beams) at 240 Hz × 10 iterations.
  perf: {
    query: 'scene=yard',
    run: async (page) => {
      await page.step(120, { forward: 0.5, steer: 0.2 });
      const ms = await page.eval(`(async () => { const t = performance.now(); await window.__step(480); return (performance.now() - t) / 480; })()`);
      return { ok: ms > 0, msg: `${ms.toFixed(2)} ms per 240 Hz step in Chrome WebGPU (${(ms * 4).toFixed(1)} ms per 60 fps frame of 4 steps)` };
    },
  },
  // Opt-in: a real BeamNG vehicle from local files (not redistributable):
  // RUBBLE_JBEAM_DIR=<vehicles dir with agenty_legocar/ and common/>.
  legocar: {
    query: 'scene=yard',
    optIn: 'RUBBLE_JBEAM_DIR',
    run: async (page) => {
      const { readdirSync } = await import('node:fs');
      const dir = process.env.RUBBLE_JBEAM_DIR;
      const files = [];
      const walk = (d) => { for (const e of readdirSync(d, { withFileTypes: true })) { const p = join(d, e.name); if (e.isDirectory()) walk(p); else if (e.name.endsWith('.jbeam')) files.push([p, readFileSync(p, 'utf8')]); } };
      walk(join(dir, 'agenty_legocar'));
      walk(join(dir, 'common'));
      const pc = readFileSync(join(dir, 'agenty_legocar', 'start.pc'), 'utf8');
      const h = await page.eval(`(async () => { const m = await import('./soft.mjs'); const d = m.loadJbeam(${JSON.stringify(files)}, ${JSON.stringify(pc)}); return window.__spawnImported(d, 'legocar'); })()`);
      const r = await page.step(480);
      const d = r.damage[h];
      await page.view({ orbit: { yaw: 0.4, pitch: 0.3, dist: 9, target: [0, 0.8, 8] } });
      await page.step(1);
      return { ok: !!d && d.broken < 20, msg: `legocar spawned as soft body ${h}: ${d?.beams} beams, ${d?.yielded} bent, ${d?.broken} broken after 2 s` };
    },
  },
};
const names = only.length ? only : Object.keys(CHECKS).filter((k) => !CHECKS[k].optIn || process.env[CHECKS[k].optIn]);

const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'softbody-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', ...(process.env.BUDGET_SPLIT ? ['--enable-webgpu-developer-features'] : []), '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
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
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 600_000 });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
  return r.result?.result?.value;
};
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });

const page = {
  step: (n, input = null) => evaluate(`window.__step(${n}, ${JSON.stringify(input)})`),
  view: (o) => evaluate(`window.__setView(${JSON.stringify(o)})`),
  eval: (e) => evaluate(e),
  shot: async (name) => {
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    writeFileSync(join(OUT, `${name}.png`), Buffer.from(shot.result.data, 'base64'));
  },
};
let failed = 0;
const report = [];
for (const name of names) {
  const c = CHECKS[name];
  if (!c) throw new Error(`unknown check ${name}; have ${Object.keys(CHECKS).join(', ')}`);
  pageErrors.length = 0;
  // SOFT_QUERY appends to every scene's query (e.g. SOFT_QUERY='iters=5&sweeps=2').
  const extra = process.env.SOFT_QUERY ? `&${process.env.SOFT_QUERY}` : '';
  await send('Page.navigate', { url: `${base}/softbody.html?${c.query}${extra}` });
  let ready = false;
  for (let i = 0; i < 300 && !ready; i++) {
    await new Promise((r) => setTimeout(r, 200));
    ready = await evaluate('window.__ready === true').catch(() => false);
    const err = await evaluate('document.getElementById("err")?.textContent || ""').catch(() => '');
    if (err) { ready = false; console.log(`✗ ${name}: page error ${err}`); break; }
  }
  if (!ready) { failed++; report.push({ name, ok: false, msg: 'did not start' }); continue; }
  const t0 = performance.now();
  let res;
  try {
    res = await c.run(page);
  } catch (e) {
    res = { ok: false, msg: String(e.message || e) };
  }
  const err = await evaluate('document.getElementById("err")?.textContent || ""');
  if (err || pageErrors.length) res = { ok: false, msg: `${res.msg} · page error: ${err || pageErrors[0]}` };
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const png = join(OUT, `${name}.png`);
  writeFileSync(png, Buffer.from(shot.result.data, 'base64'));
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`${res.ok ? '✓' : '✗'} ${name}: ${res.msg} (${secs} s) → ${png}`);
  report.push({ name, ...res, seconds: Number(secs), screenshot: png });
  if (!res.ok) failed++;
}
writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 2));
ws.close();
server.close();
process.exit(failed ? 1 : 0);
