#!/usr/bin/env node
// Records the vehicle demo takes to MP4 (play-testing and showcase videos).
//
//   node --experimental-websocket record-vehicle-demos.mjs [take ...] [--out DIR] [--reel]
//
// Serves this directory, launches headless Chrome (throwaway profile, WebGPU on), and
// runs each take through vehicle.html's frame-locked `__record` hook: one physics step
// per 60 fps frame, independent of wall-clock speed. Frames are PUT back to this
// server and encoded with ffmpeg into recordings/<take>.mp4 plus a <take>_sheet.jpg
// contact sheet (one frame per second) for quick review. --reel concatenates the
// takes into recordings/vehicles_reel.mp4. Needs Chrome and ffmpeg on PATH (or
// CHROME / FFMPEG env), and `cargo run -p engine-webgpu --bin wgsl_export` first.
import { createServer } from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));

// Each take: query string for vehicle.html and a __record script (see vehicle.html).
// Segment fields: n (steps), input, cruise (hold m/s), act, cam, caption.
const loop = (k, f) => Array.from({ length: k }, (_, i) => f(i));
const TAKES = {
  sedan_ramps: {
    query: 'preset=sedan&spawn=0,0,0',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Sedan: raycast wheels solved as rows inside AVBD', cam: 'chase' },
      { n: 120, input: { forward: 1, steer: 0 }, caption: 'Full throttle: auto gearbox, open diff' },
      { n: 150, cruise: 16, input: { steer: 0 }, caption: 'Through the bollards and over the ramps' },
      { n: 90, cruise: 16, input: { steer: 0 }, caption: 'Down the far ramp', cam: { orbit: [-1.2, 0.25, 11], spin: 0.01 } },
      { n: 40, cruise: 16, input: { steer: 0 }, caption: 'Handbrake turn', cam: 'chase' },
      { n: 70, input: { forward: 0, steer: -1, handbrake: 1 }, caption: 'Handbrake turn: rear wheels lock and slide' },
      { n: 60, input: { forward: 1, steer: -0.3 }, caption: 'Power out' },
      { n: 120, cruise: 12, input: { steer: 0.35 }, caption: 'Steady right turn: anti-roll bars, Ackermann steering' },
      { n: 100, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake to a stop' },
    ],
  },
  motorcycle_slalom: {
    query: 'preset=motorcycle&spawn=0,-8,3.14159',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Motorcycle: the lean controller holds it upright at rest', cam: 'chase' },
      { n: 150, cruise: 12, input: { steer: 0 }, caption: 'Accelerate to 43 km/h (Jolt MotorcycleTest drivetrain)' },
      ...loop(6, (k) => ({ n: 70, cruise: 12, input: { steer: k % 2 ? 1 : -1 }, caption: 'Slalom: leans into each turn, steering limited by max lean' })),
      { n: 240, cruise: 10, input: { steer: -1 }, caption: 'Sweeping left turn', cam: { orbit: [0, 0.18, 5.5], spin: 0.012 } },
      { n: 90, input: { forward: 0.8, steer: 0 }, caption: 'Straighten up and accelerate', cam: 'chase' },
      { n: 150, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Full brake to a stop' },
    ],
  },
  tank_wall: {
    query: 'preset=tank&spawn=-15,42,0',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Tank: two 7-wheel tracks, each one shared spin', cam: { orbit: [2.4, 0.35, 16], spin: 0.004 } },
      { n: 130, input: { forward: 0, steer: -1 }, caption: 'Pivot turn: tracks counter-rotate (no throttle, below pivot speed)' },
      { n: 130, input: { forward: 0, steer: 1 }, caption: 'Pivot back the other way' },
      { n: 40, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Settle' },
      { n: 200, cruise: 6, input: { steer: 0 }, caption: 'Drive into the brick wall (box friction tracks, Jolt drivetrain)', cam: 'chase' },
      { n: 150, cruise: 6, input: { steer: 0 }, caption: 'Breaching the wall', cam: { orbit: [1.9, 0.3, 18], spin: 0.006 } },
      { n: 180, cruise: 8, input: { steer: 0.7 }, caption: 'Steer by track ratio: steering brake on the inner track, torque to the outer', cam: 'chase' },
      { n: 90, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  monster_stack: {
    query: 'preset=monster_truck&spawn=-10,22,0',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Monster truck: sphere-cast wheels, 4WD with limited-slip diffs', cam: 'chase' },
      { n: 150, cruise: 11, input: { steer: 0 }, caption: 'Smash through the box stack' },
      { n: 120, cruise: 11, input: { steer: 0 }, caption: 'Boxes are ordinary AVBD bodies; the truck is one chassis body', cam: { orbit: [2.6, 0.3, 16], spin: 0.008 } },
      { n: 150, cruise: 10, input: { steer: 1 }, caption: 'Hard right turn', cam: 'chase' },
      { n: 210, cruise: 12, input: { steer: 0.15 }, caption: 'Back toward the ramps' },
      { n: 100, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  kart_ice: {
    query: 'preset=kart&spawn=-30,-6,3.14159',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Kart: direct drive, arcade assists', cam: 'chase' },
      { n: 100, cruise: 12, input: { steer: 0 }, caption: 'Onto the ice patch (μ 0.08)' },
      { n: 120, cruise: 12, input: { steer: -1 }, caption: 'Steer on ice: the tyres saturate and slide', cam: { orbit: [0.6, 0.45, 12], spin: 0.006 } },
      { n: 60, input: { forward: 1, steer: 1, handbrake: 1 }, caption: 'Handbrake spin' },
      { n: 120, input: { forward: 1, steer: 0.4 }, caption: 'Back onto asphalt: grip returns', cam: 'chase' },
      { n: 80, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  destruction: {
    query: 'scene=destruction&spawn=0,-15,3.14159',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Destructible sedan: 12 parts on one chassis body', cam: 'chase' },
      { n: 90, cruise: 9, input: { steer: 0 }, caption: 'Drive' },
      { n: 60, act: 'part', cruise: 9, input: { steer: 0 }, caption: 'Sever a part: it becomes a free rigid body with the chassis velocity' },
      { n: 60, act: 'part', cruise: 9, input: { steer: 0 }, caption: 'Sever another part' },
      { n: 90, act: 'wheel:0', cruise: 9, input: { steer: 0 }, caption: 'Sever the front-left wheel: it keeps its spin and rolls away', cam: { orbit: [2.2, 0.3, 12], spin: 0.01 } },
      { n: 120, cruise: 7, input: { steer: 0 }, caption: 'Three wheels left: suspension load redistributes' },
      { n: 90, act: 'wheel:3', cruise: 7, input: { steer: 0 }, caption: 'Sever a rear wheel (drive lost on that side)', cam: 'chase' },
      { n: 120, input: { forward: 0.8, steer: 0.2 }, caption: 'Limping on two wheels' },
      ...loop(4, () => ({ n: 20, act: 'part', input: { forward: 0.5, steer: 0 }, caption: 'Strip the body panels' })),
      { n: 120, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Wreck', cam: { orbit: [0.5, 0.35, 12], spin: 0.008 } },
    ],
  },
  traffic: {
    query: 'scene=traffic&spawn=0,0,0',
    script: [
      { n: 480, input: { forward: 0.4, steer: 0.2 }, caption: 'Traffic: 49 vehicles + 400 boxes, all on the GPU', cam: { orbit: [0.3, 0.55, 70], spin: 0.002 } },
      { n: 240, input: { forward: 0.6, steer: -0.3 }, caption: 'Chase cam in the crowd', cam: 'chase' },
    ],
  },
  // Two-way ground coupling labs (each mirrors a test in tests/vehicle.rs).
  slab_ice: {
    query: 'scene=slab',
    script: [
      { n: 60, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: sedan on a 2.5 t slab resting on frictionless skids', cam: { orbit: [1.57, 0.35, 22] } },
      { n: 200, input: { forward: 0.5, steer: 0 }, caption: 'Throttle: the tyres push the slab backward (momentum is conserved)' },
      { n: 120, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake: the tyre force reverses and both come to rest together' },
      { n: 60, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Slab displaced; the static cones show how far', cam: { orbit: [1.2, 0.5, 30] } },
    ],
  },
  seesaw: {
    query: 'scene=seesaw',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: a 250 kg seesaw plank on a pivot', cam: { orbit: [1.57, 0.25, 20] } },
      { n: 330, cruise: 3.5, input: { steer: 0 }, caption: 'The wheel loads act on the plank: it tips once the car passes the pivot' },
      { n: 120, cruise: 3.5, input: { steer: 0 }, caption: 'Down the other side', cam: { orbit: [1.9, 0.3, 18] } },
      { n: 80, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake' },
    ],
  },
  carry: {
    query: 'scene=carry',
    script: [
      { n: 90, input: { forward: 0, steer: 0 }, caption: 'Two-way coupling: a parked kart riding on a monster truck', cam: { orbit: [2.4, 0.35, 14] } },
      { n: 180, input: { forward: 0.15, steer: 0 }, caption: 'The kart’s tyres load the truck’s roof; friction carries it along', cam: 'chase' },
      { n: 240, input: { forward: 0.15, steer: 0.35 }, caption: 'Turning: the kart’s lateral tyre forces hold it on', cam: { orbit: [2.4, 0.35, 16], spin: 0.006 } },
      { n: 120, input: { forward: 0, steer: 0, brake: 0.4 }, caption: 'Gentle brake' },
    ],
  },
  sidewall: {
    query: 'scene=sidewall',
    script: [
      { n: 40, input: { forward: 0, steer: 0 }, caption: 'Wheel shells: crates beside the tyres, off the suspension ray lines', cam: { orbit: [3.5, 0.45, 12] } },
      { n: 300, cruise: 6, input: { steer: 0 }, caption: 'The tyre outlines shove the crates aside (wheels are physical, not just rays)' },
      { n: 90, input: { forward: 0, steer: 0, brake: 1 }, caption: 'Brake', cam: { orbit: [3.0, 0.5, 18] } },
    ],
  },
};

const args = process.argv.slice(2);
const outIdx = args.indexOf('--out');
const OUT = outIdx >= 0 ? args[outIdx + 1] : join(ROOT, 'recordings');
const reel = args.includes('--reel');
const names = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--out');
const takes = names.length ? names : Object.keys(TAKES);
for (const t of takes) if (!TAKES[t]) throw new Error(`unknown take ${t}; have ${Object.keys(TAKES).join(', ')}`);
const FFMPEG = process.env.FFMPEG || 'ffmpeg';
const CHROME = process.env.CHROME || (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome');
const FRAMES = mkdtempSync(join(tmpdir(), 'vehicle-frames-'));
mkdirSync(OUT, { recursive: true });

// Static files + PUT /rec/<take>/<frame>.jpg.
const TYPES = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wgsl': 'text/plain' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  const m = req.method === 'PUT' && url.pathname.match(/^\/rec\/([\w-]+)\/([\w.-]+)$/);
  if (m) {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      mkdirSync(join(FRAMES, m[1]), { recursive: true });
      writeFileSync(join(FRAMES, m[1], m[2]), Buffer.concat(chunks));
      res.writeHead(204).end();
    });
    return;
  }
  const path = normalize(join(ROOT, decodeURIComponent(url.pathname)));
  if (!path.startsWith(ROOT) || !existsSync(path) || statSync(path).isDirectory()) { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'content-type': TYPES[extname(path)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(path));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const profile = mkdtempSync(join(tmpdir(), 'vehicle-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${profile}`,
  '--enable-unsafe-webgpu', '--window-size=1280,720', '--no-first-run', '--no-default-browser-check',
  '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank',
], { stdio: 'ignore' });
const cleanup = () => { try { chrome.kill('SIGKILL'); } catch {} rmSync(profile, { recursive: true, force: true }); rmSync(FRAMES, { recursive: true, force: true }); };
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
});
const send = (method, params = {}) => new Promise((r) => { const id = ++nextId; waiting.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression, awaitPromise = false) => (await send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true, timeout: 3_600_000 })).result?.result?.value;
await send('Runtime.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });

const ffmpeg = (a) => {
  const r = spawnSync(FFMPEG, ['-loglevel', 'error', '-y', ...a], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`ffmpeg failed: ${a.join(' ')}`);
};
for (const name of takes) {
  const take = TAKES[name];
  await send('Page.navigate', { url: `${base}/vehicle.html?${take.query}` });
  let ready = false;
  for (let i = 0; i < 300 && !ready; i++) {
    await new Promise((r) => setTimeout(r, 200));
    ready = await evaluate('typeof window.__record === "function" && !!window.__vehicleState()');
  }
  const err = await evaluate('document.getElementById("err")?.textContent || ""');
  if (!ready || err) throw new Error(`${name}: demo did not start ${err}`);
  const r = await evaluate(`window.__record(${JSON.stringify(take.script)}, { name: ${JSON.stringify(name)} })
    .then((r) => ({ frames: r.frames, seconds: r.seconds }))`, true);
  const frames = join(FRAMES, name, 'f%05d.jpg');
  ffmpeg(['-framerate', '60', '-i', frames, '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', join(OUT, `${name}.mp4`)]);
  ffmpeg(['-i', frames, '-vf', "select='not(mod(n\\,60))',scale=480:-1,tile=4x4", '-frames:v', '1', join(OUT, `${name}_sheet.jpg`)]);
  console.log(`${name}: ${r.frames} frames (${(r.frames / 60).toFixed(1)} s video) in ${r.seconds.toFixed(1)} s → ${join(OUT, `${name}.mp4`)}`);
}
if (reel) {
  const list = join(FRAMES, 'reel.txt');
  writeFileSync(list, takes.map((t) => `file '${join(OUT, `${t}.mp4`)}'`).join('\n'));
  ffmpeg(['-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', join(OUT, 'vehicles_reel.mp4')]);
  console.log(`reel → ${join(OUT, 'vehicles_reel.mp4')}`);
}
ws.close();
server.close();
process.exit(0);
