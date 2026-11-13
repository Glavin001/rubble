// The pages' physics: rubble's Rust World (engine-world-wasm) on the browser's WebGPU,
// the same host native and Unreal run. This file only loads it and keeps per-frame
// bookkeeping the pages share; all physics logic is in Rust.
//
// Pages need an import map for the wasm's GPU-wait import:
//   <script type="importmap">{ "imports": { "rubble-gpu-wait": "./fracture/gpu-wait.mjs" } }</script>
import init, {
  WebWorld, WebCity, solverFlags, softCalibratedSettings, terrainSettings, terrainPredictImpact, terrainPredictExplosion, terrainLayout,
  hiresCarJbeam, explicitSedanTruck, showcaseCar, jbeamVisual, jbeamVisualSkinned, decodeTexture, vehicleTownLayout, FramePacer,
  showcaseLayout, brickRingLayout, createVehicleLab as createVehicleLabWasm, vehicleLabMaterials,
} from './generated/world-wasm/engine_world_wasm.js';

// hiresCarJbeam: the high-resolution test car (engine_softbody::hires) as JBeam text, once loaded.
// explicitSedanTruck(tyres): the explicit solver's procedural sedan as .truck text.
// showcaseCar(params): an authored vehicle (engine_softbody::authoring): { truck, counts, visual }.
// jbeamVisual(spec, meshes): an imported JBeam vehicle's flexbodies and props (engine_softbody::flexmesh).
// decodeTexture(bytes, size): a PNG or DDS as one mipped RGBA8 layer (engine_softbody::flextex).
// jbeamVisualSkinned(spec, meshes, glb, skin): the same for a vehicle whose
// flexbodies are a glTF render model with weighted skins (flexskin, `.rskin`).
// FramePacer(rate, gpuBudget): fixed steps per displayed frame (engine-world-wasm frame_pacer.rs).
export { WebWorld, WebCity, softCalibratedSettings, hiresCarJbeam, explicitSedanTruck, showcaseCar, jbeamVisual, jbeamVisualSkinned, decodeTexture, FramePacer };
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

/**
 * The large-maps city (`rubble::city`, as the native benchmark builds it): `opts`
 * `{ houses, map, sleep, pool }`, `page` the WorldDef fields a page may set (`gpuTiming`,
 * `killBelow`, `readbackFrames`, ...). Place the houses with `world.cityBegin(city)` then
 * `world.cityPlace(city, n)` over frames; `world.cityRebuild(city, page)` starts over.
 */
export async function createCity(opts, page = {}) {
  ready ??= init();
  await ready;
  const city = new WebCity(opts);
  const world = await WebWorld.createCity(city, page);
  world.flags = solverFlags();
  return { city, world };
}

/**
 * The explicit vehicles' crash town (`rubble::explicit_scene::crash_town`, as the native
 * benchmark builds it): `spec` `{ truck, count, houses, speed, runUp, drag }` (`truck`
 * .truck text; absent, the procedural sedan with tyres). Returns the world and
 * `{ houses, ground, centres }`.
 */
export async function createVehicleTown(spec, progress) {
  ready ??= init();
  await ready;
  const world = await WebWorld.createVehicleTown({ ...spec, truck: spec.truck ?? explicitSedanTruck(true) }, progress);
  world.flags = solverFlags();
  return { world, layout: vehicleTownLayout(spec.houses ?? 0, spec.count ?? 4) };
}

/**
 * The one-World showcase (`rubble::showcase_scene`): `spec` `{ vehicles: [{ truck } |
 * { jbeam: [[name, text]], pc }] (cycled over the preset's cars), preset: 'house' |
 * 'crates' | 'twoway' | 'convoy', count, speed, runUp, drag }` (or `truck`: one .truck
 * vehicle). Returns the world and its layout (`showcaseLayout`). `progress` (optional,
 * here and in the other create functions): a loading card's `loading.progress`
 * (load-progress.mjs), which hears the load's stages and each GPU kernel's compile.
 */
export async function createShowcase(spec, progress) {
  ready ??= init();
  await ready;
  const { truck, vehicles, ...rest } = spec;
  const layout = showcaseLayout(rest);
  const world = await WebWorld.createShowcase(spec, progress);
  world.flags = solverFlags();
  return { world, layout };
}

/**
 * The brick ring drive (`rubble::brick_ring_drive`): the 110k brick ring with drivable
 * soft-body cars on raycast wheels in one World on its own device (spec as
 * `WebWorld.createBrickRingDrive`), and its layout (`brickRingLayout`).
 */
export async function createBrickRingDrive(spec, progress) {
  ready ??= init();
  await ready;
  const { vehicles, ...rest } = spec;
  const layout = brickRingLayout(rest);
  const world = await WebWorld.createBrickRingDrive(spec, progress);
  world.flags = solverFlags();
  return { world, layout };
}

/**
 * The vehicle test lab (`rubble::vehicle_lab`, vehicle-lab.html): one soft-body car on a
 * flat arena or the infinite mountain in one World on its own device, and the lab's
 * state beside it (spec as the wasm's `createVehicleLab`).
 */
export async function createVehicleLab(spec, progress) {
  ready ??= init();
  await ready;
  const [world, lab] = await createVehicleLabWasm(spec, progress);
  world.flags = solverFlags();
  return { world, lab, materials: vehicleLabMaterials() };
}

/** The wasm, once loaded (pages that call its free functions before a world). */
export async function loadWasm() {
  ready ??= init();
  await ready;
}

/**
 * A compact snapshot for `world.restoreState` (`World::measure_state`): each GPU
 * buffer copied up to its last non-zero word, the same bytes on restore. A scene
 * saved before it runs needs a fraction of `world.saveState()`'s GPU memory, which
 * copies every buffer whole (phones run out). Take it before stepping again.
 */
export async function saveState(world) {
  const measure = world.measureState();
  try {
    return world.saveStateMeasured(measure, await measure.extents());
  } finally {
    measure.free();
  }
}

/** An error for a page's error line: its message, then its stack (Safari's stack
 * leaves the message out). */
export function errorText(e) {
  if (!e || !e.stack) return String(e);
  const stack = String(e.stack);
  return e.message && !stack.includes(e.message) ? `${e.name || 'Error'}: ${e.message}\n${stack}` : stack;
}

/** Bytes of the wasm heap (its linear memory), once loaded. */
export async function wasmHeapBytes() {
  return (await ready)?.memory?.buffer.byteLength ?? 0;
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

// The frame helpers every page shares (no wasm: unit tests import them alone).
export { macrotask, pollFrame, stepFrames, drain } from './world-loop.mjs';
