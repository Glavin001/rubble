// Soft-body (node–beam) vehicle descriptors for the WebGPU pages: engine-softbody's
// importers (Rigs of Rods `.truck`, JBeam) and procedural vehicles compiled to wasm
// (crates/engine-vehicle-wasm, built into generated/vehicle_wasm by wgsl_export). The
// pages spawn them on World (engine-world-wasm `addSoftVehicle`).
import { wasm } from './vehicle.mjs';

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
/** vibe-land's dune buggy, hybrid: SoftVehicleDesc (engine_softbody::buggy); mount limits × `mountScale`. */
export const duneBuggySoft = (mountScale = 1) => JSON.parse(wasm.soft_dune_buggy(mountScale));
/** Render mesh of the procedural hatchback (engine_softbody::visual): parts,
 * materials and node bindings; see `soft_procedural_car_visual` for the words. */
export const proceduralCarVisual = () => wasm.soft_procedural_car_visual();
/** The procedural hatchback as a SoftVehicleDesc with named parts (body,
 * bonnet, boot, door_fl, door_rl, door_fr, door_rr, bumper_front, bumper_rear, engine). */
export const proceduralCarDesc = () => JSON.parse(wasm.soft_procedural_car_desc());
/** A braced lattice crate (n³ nodes) as `.truck` text. */
export const latticeCrate = (n, step, mass, spring, deform, strength) => wasm.soft_box_truck(n, step, mass, spring, deform, strength);
