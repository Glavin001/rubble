// The web World's physics in Node on Dawn (Chrome's WebGPU implementation, the npm
// `webgpu` package): the vehicle lab without its page. Loads the release World wasm
// (kernels embedded), builds the lab the page builds, steps it to a moment, then times
// frames and runs the explicit vehicles' A/B of a perf flag (`world.explicitAb`: the
// same frames with the flag on and off from one state, every state word compared).
//
//   BUNDLE=<dir> CAR=<generated sedan dir> node --experimental-wasm-jspi \
//     --import ./loader.mjs world_ab.mjs [--cars N] [--scene mountain|flat]
//     [--until SECONDS] [--ab MASK] [--frames N] [--steps N] [--time-frames N]
//
// BUNDLE holds the web build's files at their places under crates/engine-webgpu:
// rubble-world.mjs, world-loop.mjs, fracture/gpu-wait.mjs, generated/world-wasm/.
// Prints one `RESULT {json}` line.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { create, globals } from 'webgpu';

const arg = (name, def) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : def;
};
const cars = Number(arg('--cars', 8));
const scene = arg('--scene', 'mountain');
const until = Number(arg('--until', 2));
const mask = Number(arg('--ab', 0));
const frames = Number(arg('--frames', 20));
const steps = Number(arg('--steps', 34));
const timeFrames = Number(arg('--time-frames', 60));
const B = process.env.BUNDLE;
const CAR = process.env.CAR;
if (!B || !CAR) throw new Error('BUNDLE and CAR must be set');

// Dawn as the browser's WebGPU: wgpu's wasm backend checks for a Window (or worker) global,
// then reads navigator.gpu.
Object.assign(globalThis, globals);
globalThis.Window ??= class Window {};
const gpu = create([]);
if (globalThis.navigator) Object.defineProperty(globalThis.navigator, 'gpu', { value: gpu, configurable: true });
else globalThis.navigator = { gpu };

const out = { t: {} };
const t0 = performance.now();
const lap = (k) => { out.t[k] = Math.round(performance.now() - t0); };
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
// (GPUAdapterInfo's fields are getters: copied by name, a spread gives {}.)
out.adapter = adapter
  ? Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map((k) => [k, adapter.info?.[k]]))
  : null;

const W = join(B, 'generated/world-wasm');
const glue = await import(pathToFileURL(join(W, 'engine_world_wasm.js')).href);
await glue.default({ module_or_path: readFileSync(join(W, 'engine_world_wasm_bg.wasm')) });
lap('wasm');
console.log(`[+${((performance.now() - t0) / 1000).toFixed(1)} s] wasm loaded; adapter ${JSON.stringify(out.adapter)}`);
const rw = await import(pathToFileURL(join(B, 'rubble-world.mjs')).href);

// The generated sedan as the page loads it (beamng-mod.mjs loadMod): every .jbeam's text
// and the .pc file.
const list = JSON.parse(readFileSync(join(CAR, 'files.json'), 'utf8'));
const files = list.filter((f) => f.endsWith('.jbeam')).map((f) => [f, readFileSync(join(CAR, f), 'utf8')]);
const pcName = list.find((f) => f.endsWith('.pc'));
const pc = pcName ? readFileSync(join(CAR, pcName), 'utf8') : undefined;

const spec = { vehicle: { jbeam: files, pc }, scene, cars, bitExact: true,
  ...(scene === 'mountain' ? { mountain: { terrain: arg('--terrain', 'ledges') } } : {}) };
// Every progress line the lab reports (each kernel's compile, as the page's loading card
// shows them), with the time since the harness began.
const stamp = () => `[+${((performance.now() - t0) / 1000).toFixed(1)} s]`;
// Kernels the lab started compiling and has not finished (deferred ones compile while it
// steps): timing waits until none are left, so a compile never lands in a frame's time.
const pending = new Set();
out.compileMs = {};
const progress = (...m) => {
  const text = m.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ');
  console.log(stamp(), 'lab:', text);
  const c = /^compiling (\S+)/.exec(text);
  if (c) pending.add(c[1]);
  const d = /^compiled (\S+) (\d+)/.exec(text);
  if (d) { pending.delete(d[1]); out.compileMs[d[1]] = Number(d[2]); }
};
const { world, lab } = await rw.createVehicleLab(spec, progress);
console.log(stamp(), 'lab built');
lap('lab');
out.vehicles = lab.vehicleCount?.();

const before = () => lab.beforeStep(world);
const stepTo = async (t) => {
  while (lab.time() < t - 1e-9) await rw.stepFrames(world, 1, () => {}, before);
};
console.log(stamp(), `stepping to t ${until} s`);
await stepTo(until);
await rw.drain(world, () => {});
lap('stepped');
if (pending.size) console.log(stamp(), `waiting for ${[...pending].join(', ')} to finish compiling`);
const waitStart = performance.now();
while (pending.size && performance.now() - waitStart < 900_000) await rw.stepFrames(world, 1, () => {}, before);
await rw.drain(world, () => {});
lap('kernels');
out.simTime = lab.time();
out.compileTotalMs = Object.values(out.compileMs).reduce((a, b) => a + b, 0);
console.log(stamp(), `timing ${timeFrames} frames from t ${out.simTime.toFixed(2)} s`);

// Frame time: `timeFrames` World steps (60 Hz frames) from here as fast as readbacks
// allow, as the page's automation steps (wall, ms a frame).
const f0 = performance.now();
await rw.stepFrames(world, timeFrames, () => {}, before);
await rw.drain(world, () => {});
out.msPerFrame = (performance.now() - f0) / timeFrames;
lap('timed');

if (mask) {
  const a0 = performance.now();
  const diff = await world.explicitAb(mask, frames, steps);
  out.ab = { mask, frames, steps, bitForBit: diff.length === 0, differences: diff.slice(0, 8), ms: Math.round(performance.now() - a0) };
  lap('ab');
}
out.errors = world.explicitErrors?.() ?? [];
console.log('RESULT ' + JSON.stringify(out));
process.exit(0);
