// Vehicle constants and descriptors for the WebGPU pages: engine-vehicle compiled to wasm
// (crates/engine-vehicle-wasm, built into generated/vehicle_wasm by wgsl_export). The pages
// run vehicles on World (engine-world-wasm); this module gives them the layout constants
// their renderers read, the scene builders that only describe launches, and the step plan
// the terrain test checks. Descriptors come from generated/vehicle_presets.json
// (serde-expanded Rust presets).
import init, * as wasm from './generated/vehicle_wasm/engine_vehicle_wasm.js';

const wasmUrl = new URL('./generated/vehicle_wasm/engine_vehicle_wasm_bg.wasm', import.meta.url);
// Node (tests) cannot fetch file: URLs; hand it the bytes instead.
await init({ module_or_path: wasmUrl.protocol === 'file:' ? (await import('node:fs')).readFileSync(wasmUrl) : wasmUrl });

/** Layout constants (slang/physics/vehicle_layout.slang, via engine-schema). */
export const L = JSON.parse(wasm.layout_constants());
export const { BODY_VEHICLE, JOINT_FLAG_TYRE, JOINT_BYTES, WHEEL_RIM_MATERIAL } = L;
/** engine_exec::rigid_plan (tests), D6 joint records, and the scene builders. */
export const { RigidStep, ball_joint_words: ballJointWords, Gauntlet, MeteorRange, MeteorRun } = wasm;
/** Character / projectile pool and continuous-collision constants (engine-schema). */
export const M = JSON.parse(wasm.motion_constants());
/** The whole wasm module (soft.mjs builds on the same instance). */
export { wasm };

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
export const bitsf = (u) => { u32b[0] = u >>> 0; return f32b[0]; };

/** Pool sizes a descriptor needs. */
export const requirements = (desc) => JSON.parse(wasm.requirements(JSON.stringify(desc)));
/** Design-space centre of mass (where addVehicle puts the chassis body's origin). */
export const centerOfMass = (desc) => Array.from(wasm.center_of_mass(JSON.stringify(desc)));

export const DEFAULT_VEHICLE_CAPACITY = {
  maxVehicles: 64, maxWheels: 512, maxCurvePoints: 4096, maxDiffs: 256, maxAntiRolls: 256, maxParts: 512, maxRays: 1024,
};
export const capWords = (c) => Uint32Array.of(c.maxVehicles, c.maxWheels, c.maxCurvePoints, c.maxDiffs, c.maxAntiRolls, c.maxParts, c.maxRays);

/** Region bases (float4 slots) inside gVehicleData for a capacity. */
export function vehicleLayout(cap) {
  const h = new wasm.WasmVehicleHost(capWords({ ...DEFAULT_VEHICLE_CAPACITY, ...cap }), 0);
  const l = JSON.parse(h.layout());
  h.free();
  return l;
}

