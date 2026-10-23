// Soft-body (node–beam) vehicles for the WebGPU host: engine-softbody's SoftHost
// and Rigs of Rods `.truck` importer compiled to wasm (crates/engine-vehicle-wasm,
// built into generated/vehicle_wasm by wgsl_export). Like vehicle.mjs, every call
// that touches GPU records returns an op stream replayed into the world backend.
import { wasm, applyOps } from './vehicle.mjs';

export const SOFT = JSON.parse(wasm.soft_constants());

/** Soft arena capacity (binding 32). Smaller than the native default for the browser. */
export const DEFAULT_SOFT_CAPACITY = {
  maxBodies: 32, maxBeams: 24576, maxParticles: 4096, maxColors: 64, maxGroups: 1024, maxTriangles: 12288, maxObstacles: 64,
};
export const softCapWords = (c) => Uint32Array.of(c.maxBodies, c.maxBeams, c.maxParticles, c.maxColors, c.maxGroups, c.maxTriangles, c.maxObstacles);

/** `.truck` text → SoftVehicleDesc (masses distributed, warnings listed). */
export const loadTruck = (src) => JSON.parse(wasm.soft_load_truck(src));
/** SoftSettings with the threshold scale calibrated for a step rate (soft_calibrate). */
export const calibratedSettings = (hz) => JSON.parse(wasm.soft_calibrated_settings(hz));
/** JBeam files ([[name, source], …]) and an optional `.pc` → SoftVehicleDesc. */
export const loadJbeam = (files, pc = undefined) => JSON.parse(wasm.soft_load_jbeam(JSON.stringify(files), pc));
/** The procedural test hatchback as `.truck` text (thresholds in N, 0 = default). */
export const proceduralCar = (deform = 0, strength = 0) => wasm.soft_procedural_car(deform, strength);
/** A braced lattice crate (n³ nodes) as `.truck` text. */
export const latticeCrate = (n, step, mass, spring, deform, strength) => wasm.soft_box_truck(n, step, mass, spring, deform, strength);

export class SoftHost {
  constructor(cap) {
    this.cap = { ...DEFAULT_SOFT_CAPACITY, ...cap };
    this.inner = new wasm.WasmSoftHost(softCapWords(this.cap));
    this.layout = JSON.parse(this.inner.layout());
    this.entries = [];
  }

  get bodyCount() { return this.inner.body_count; }
  /** Plan inputs (WasmSoftHost::counts). */
  counts() { return this.inner.counts(); }

  /**
   * Spawn a soft vehicle at `position`/`rotation`; with wheels its rigid core joins
   * `vehicles` (the world's VehicleHost). `opts` = engine_softbody::SpawnOptions.
   */
  add(backend, vehicles, desc, position, rotation = [0, 0, 0, 1], opts = {}) {
    const h = this.inner.body_count;
    const counts = Uint32Array.of(backend.bodyCount(), backend.colliderCount(), backend.jointCount());
    applyOps(backend, this.inner.add(counts, JSON.stringify(desc), position, rotation, JSON.stringify(opts), vehicles.inner));
    this.entries[h] = JSON.parse(this.inner.entry(h));
    if (this.entries[h].core) vehicles.refreshEntry(this.entries[h].core.vehicle);
    return h;
  }

  addObstacle(backend, collider) { applyOps(backend, this.inner.add_obstacle(collider)); }
  setSettings(backend, settings) { applyOps(backend, this.inner.set_settings(JSON.stringify({ ...this.settings(), ...settings }))); }
  settings() { return JSON.parse(this.inner.settings()); }
  setSteer(backend, h, steer) { applyOps(backend, this.inner.set_steer(h, steer)); }

  save() { return { inner: this.inner.save(), entries: structuredClone(this.entries) }; }
  restore(snap) { this.inner.restore(snap.inner); this.entries = structuredClone(snap.entries); }
}
