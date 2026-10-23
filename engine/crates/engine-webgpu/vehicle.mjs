// Vehicle controller host for the WebGPU demo: a thin wrapper over engine-vehicle's
// packer and VehicleHost compiled to wasm (crates/engine-vehicle-wasm, built into
// generated/vehicle_wasm by wgsl_export). Rust returns every GPU record write as an
// op stream; `applyOps` replays it into the backend. Descriptors come from
// generated/vehicle_presets.json (serde-expanded Rust presets).
import init, * as wasm from './generated/vehicle_wasm/engine_vehicle_wasm.js';

const wasmUrl = new URL('./generated/vehicle_wasm/engine_vehicle_wasm_bg.wasm', import.meta.url);
// Node (tests) cannot fetch file: URLs; hand it the bytes instead.
await init({ module_or_path: wasmUrl.protocol === 'file:' ? (await import('node:fs')).readFileSync(wasmUrl) : wasmUrl });

/** Layout constants (slang/physics/vehicle_layout.slang, via engine-schema). */
export const L = JSON.parse(wasm.layout_constants());
export const { BODY_VEHICLE, JOINT_FLAG_TYRE, JOINT_BYTES, WHEEL_RIM_MATERIAL } = L;
/** engine_exec::rigid_plan for browser hosts, and D6 joint records (world-host.mjs). */
export const { RigidStep, ball_joint_words: ballJointWords } = wasm;
/** Character / projectile pool and continuous-collision constants (engine-schema). */
export const M = JSON.parse(wasm.motion_constants());
/** A character record (Uint32Array, 16 slots) from a CharacterDesc JSON. */
export const characterRecord = (descJson, innerBody, x, y, z) => wasm.character_record(descJson, innerBody, x, y, z);
export const characterDefault = () => JSON.parse(wasm.character_default());
/** The whole wasm module (soft.mjs builds on the same instance). */
export { wasm };
const OP = JSON.parse(wasm.op_codes());

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
export const fbits = (v) => { f32b[0] = v; return u32b[0]; };
export const bitsf = (u) => { u32b[0] = u >>> 0; return f32b[0]; };

/** Pool sizes a descriptor needs. */
export const requirements = (desc) => JSON.parse(wasm.requirements(JSON.stringify(desc)));

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

/** Replay a Rust op stream (`[op, arg, len, words…]`) into `backend`. */
export function applyOps(backend, ops) {
  for (let i = 0; i < ops.length;) {
    const op = ops[i], arg = ops[i + 1], n = ops[i + 2];
    const w = ops.subarray(i + 3, i + 3 + n);
    i += 3 + n;
    switch (op) {
      case OP.OP_APPEND_BODY: backend.appendBodies([w]); break;
      case OP.OP_APPEND_COLLIDER: backend.appendColliders([w]); break;
      case OP.OP_APPEND_JOINT: backend.appendJoints([w]); break;
      case OP.OP_WRITE_BODY: backend.writeBody(arg, w); break;
      case OP.OP_WRITE_COLLIDER: backend.writeCollider(arg, w); break;
      case OP.OP_WRITE_SLOTS: backend.writeSlots(arg, w); break;
      case OP.OP_WRITE_BYTES: backend.writeBytes(arg, w); break;
      case OP.OP_WRITE_SOFT_SLOTS: backend.writeSoftSlots(arg, w); break;
      default: throw new Error(`unknown vehicle op ${op}`);
    }
  }
}

// `read(slot, count)` for Rust from a Uint32Array view of gVehicleData.
const reader = (words) => (slot, count) => words.subarray(slot * 4, (slot + count) * 4);

/**
 * engine_vehicle::VehicleHost in wasm. `backend` provides bodyCount()/colliderCount()/
 * jointCount(), appendBodies/appendColliders/appendJoints(words[]), writeBody(index, words),
 * writeCollider(index, words), writeSlots(slot, Uint32Array) and writeBytes(byteOffset,
 * Uint32Array) into gVehicleData. `entries[h]` mirrors the Rust bookkeeping
 * ({desc, slots, wheelSlots, chassis, firstPartBody}) for the UI.
 */
export class VehicleHost {
  constructor(cap, aabbTail) {
    this.cap = { ...DEFAULT_VEHICLE_CAPACITY, ...cap };
    this.inner = new wasm.WasmVehicleHost(capWords(this.cap), aabbTail);
    this.layout = JSON.parse(this.inner.layout());
    this.entries = [];
  }

  get vehicleCount() { return this.inner.vehicle_count; }
  get wheelCount() { return this.inner.wheel_count; }
  get dirty() { return this.inner.dirty; }
  set dirty(v) { this.inner.dirty = v; }
  // Wheel probes scan every collider instead of walking the broadphase grid (oracle).
  get fullScanQueries() { return this.inner.full_scan_queries; }
  set fullScanQueries(v) { this.inner.full_scan_queries = v; }

  refreshEntry(h) { this.entries[h] = JSON.parse(this.inner.entry(h)); }

  addVehicle(backend, desc, position, rotation = [0, 0, 0, 1]) {
    const h = this.inner.vehicle_count;
    const counts = Uint32Array.of(backend.bodyCount(), backend.colliderCount(), backend.jointCount());
    applyOps(backend, this.inner.add_vehicle(counts, JSON.stringify(desc), position, rotation));
    this.refreshEntry(h);
    return h;
  }

  setInput(backend, h, { forward = 0, steer = 0, brake = 0, handbrake = 0, gear = null } = {}) {
    applyOps(backend, this.inner.set_input(h, forward, steer, brake, handbrake, gear));
  }

  setFunctionalState(backend, h, { wheelMask = 0xFFFFFFFF, driveMask = 0xFFFFFFFF, steerMask = 0xFFFFFFFF, brakeMask = 0xFFFFFFFF, engineConnected = true } = {}) {
    applyOps(backend, this.inner.set_functional_state(h, wheelMask >>> 0, driveMask >>> 0, steerMask >>> 0, brakeMask >>> 0, engineConnected));
  }

  detachWheel(backend, h, w) { applyOps(backend, this.inner.detach_wheel(h, w)); }
  disableWheel(backend, h, w) { applyOps(backend, this.inner.disable_wheel(h, w)); }
  detachPart(backend, h, p) { applyOps(backend, this.inner.detach_part(h, p)); }
  disablePart(backend, h, p) { applyOps(backend, this.inner.disable_part(h, p)); }
  wake(backend, h) { applyOps(backend, this.inner.wake(h)); }

  /** Attach `wheel` (design space) into a spare slot; `words` views gVehicleData. */
  addWheel(backend, h, wheel, words) {
    const idx = this.entries[h].desc.wheels.length;
    applyOps(backend, this.inner.add_wheel(h, JSON.stringify(wheel), reader(words)));
    this.refreshEntry(h);
    return idx;
  }

  /** Replace a wheel's tuning in place; `words` views gVehicleData. */
  setWheelParams(backend, h, w, wheel, words) {
    applyOps(backend, this.inner.set_wheel_params(h, w, JSON.stringify(wheel), reader(words)));
    this.refreshEntry(h);
  }

  chassisBody(h) { return this.inner.chassis_body(h); }
  wheelBody(h, w) { return this.inner.wheel_body(h, w); }
  partBody(h, p) { return this.inner.part_body(h, p); }

  /** Decode outputs from a Uint32Array view of gVehicleData (whole buffer or prefix). */
  readState(words, h) { return JSON.parse(this.inner.read_state(h, reader(words))); }

  /** Bookkeeping copy for a scene reset (restore the GPU records alongside). */
  save() { return { inner: this.inner.save(), entries: structuredClone(this.entries) }; }
  restore(snap) {
    this.inner.restore(snap.inner);
    this.entries = structuredClone(snap.entries);
  }
}
