// The pages' physics: rubble's Rust World (engine-world-wasm) on the browser's WebGPU,
// the same host native and Unreal run. This file only loads it and keeps per-frame
// bookkeeping the pages share; all physics logic is in Rust.
//
// Pages need an import map for the wasm's GPU-wait import:
//   <script type="importmap">{ "imports": { "rubble-gpu-wait": "./fracture/gpu-wait.mjs" } }</script>
import init, {
  WebWorld, solverFlags, softCalibratedSettings, terrainSettings, terrainPredictImpact, terrainPredictExplosion, terrainLayout,
} from './generated/world-wasm/engine_world_wasm.js';

export { WebWorld, softCalibratedSettings };
let ready = null;

/** The wasm's world-less functions (terrain settings, predictions, buffer layout), once loaded. */
export async function worldWasm() {
  ready ??= init();
  await ready;
  return { terrainSettings, terrainPredictImpact, terrainPredictExplosion, terrainLayout };
}

/** A World on its own WebGPU device (`WorldDef` fields, camelCase). Render on `world.device()`. */
export async function createWorld(def = {}) {
  ready ??= init();
  await ready;
  const world = await WebWorld.create(def);
  world.flags = solverFlags();
  return world;
}

/** Floats per body move in `WebFrame.moves()` (engine-world-wasm frame.rs). */
export const MOVE_STRIDE = 16;
export const MOVE_ASLEEP = 1, MOVE_WOKE = 2, MOVE_FIRST = 4;

/** The latest origin pose and sleep state of every body slot, kept from frames. */
export class Poses {
  constructor(capacity) {
    this.pose = new Float32Array(capacity * 7); // position xyz, rotation xyzw
    this.asleep = new Uint8Array(capacity);
  }
  apply(frame) {
    const m = frame.moves();
    for (let o = 0; o < m.length; o += MOVE_STRIDE) {
      const s = m[o], flags = m[o + 1];
      this.pose.set(m.subarray(o + 2, o + 9), s * 7);
      if (flags & MOVE_ASLEEP) this.asleep[s] = 1;
      if (flags & MOVE_WOKE) this.asleep[s] = 0;
    }
    return m.length / MOVE_STRIDE;
  }
  position(slot) { return Array.from(this.pose.subarray(slot * 7, slot * 7 + 3)); }
  rotation(slot) { return Array.from(this.pose.subarray(slot * 7 + 3, slot * 7 + 7)); }
}

// One macrotask: lets the browser run GPU map callbacks.
const channel = new MessageChannel();
let waiting = [];
channel.port1.onmessage = () => { const w = waiting; waiting = []; for (const r of w) r(); };
export const macrotask = () => new Promise((r) => { waiting.push(r); if (waiting.length === 1) channel.port2.postMessage(0); });

/** Polls the world, hands the frame to `onFrame` and frees it; tracks the last step seen. */
export function pollFrame(world, onFrame) {
  const f = world.poll();
  if (!f.empty) world.lastPolledStep = f.lastStep;
  onFrame(f);
  f.free();
}

/**
 * `n` steps as fast as readbacks allow (automation, checks), handing each poll's
 * frame to `onFrame`; resolves once every step's results have arrived.
 * `beforeStep()` runs right before each step is taken (per-step inputs).
 */
export async function stepFrames(world, n, onFrame, beforeStep = null) {
  for (let i = 0; i < n;) {
    if (world.canStep()) {
      beforeStep?.();
      world.step();
      i++;
    } else await macrotask();
    pollFrame(world, onFrame);
  }
  await drain(world, onFrame);
}

/** Resolves once the frames of every step taken so far have been polled. */
export async function drain(world, onFrame) {
  const last = world.stepCount() - 1;
  while ((world.lastPolledStep ?? -1) < last) {
    await macrotask();
    pollFrame(world, onFrame);
  }
}
