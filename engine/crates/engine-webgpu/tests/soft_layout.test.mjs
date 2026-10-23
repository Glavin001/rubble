// soft.mjs wraps engine-softbody compiled to wasm (crates/engine-vehicle-wasm/src/soft.rs):
// the importers, the SoftHost op stream the browser replays, and the step plan's soft
// passes. GPU behaviour is tested natively (crates/engine-webgpu/tests/softbody.rs) and
// in Chrome (softbody-check.mjs).
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { VehicleHost, RigidStep, capWords, DEFAULT_VEHICLE_CAPACITY } from '../vehicle.mjs';
import { SoftHost, SOFT, loadTruck, loadJbeam, proceduralCar, latticeCrate, calibratedSettings, softCapWords, DEFAULT_SOFT_CAPACITY } from '../soft.mjs';

test('body flag and event code match soft_step.slang', () => {
  const src = readFileSync(new URL('../../../slang/physics/common.slang', import.meta.url), 'utf8');
  assert.match(src, new RegExp(`BODY_PARTICLE = ${SOFT.BODY_PARTICLE}u`));
  const kern = readFileSync(new URL('../../../slang/kernels/soft_step.slang', import.meta.url), 'utf8');
  assert.match(kern, new RegExp(`EVENT_BEAM_BREAK = ${SOFT.EVENT_BEAM_BREAK}u`));
  assert.match(kern, new RegExp(`BEAM_BROKEN = ${SOFT.BEAM_BROKEN}u`));
  assert.match(kern, new RegExp(`BEAM_YIELDED = ${SOFT.BEAM_YIELDED}u`));
});

test('importers: procedural .truck and a JBeam part tree', () => {
  const car = loadTruck(proceduralCar());
  assert.equal(car.wheels.length, 4);
  assert.ok(car.nodes.length > 90 && car.beams.length > 500);
  // Infinite thresholds survive the JSON boundary (as ±f32::MAX).
  const crate = loadTruck(latticeCrate(3, 0.4, 100, 1e6, 1e5, 1e6));
  assert.equal(crate.nodes.length, 27);
  const jbeam = `{ "toy": { "slotType": "main",
      "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 5}, ["a", 0, 0, 0], ["b", 1, 0, 0], ["c", 0, 1, 0]],
      "beams": [["id1:", "id2:"], {"beamStrength": "FLT_MAX"}, ["a", "b"], ["b", "c"], ["c", "a"]],
      "triangles": [["id1:", "id2:", "id3:"], ["a", "b", "c"]] } }`;
  const d = loadJbeam([['toy.jbeam', jbeam]]);
  assert.equal(d.nodes.length, 3);
  assert.equal(d.beams.length, 3);
  assert.ok(d.beams[0].strength > 3e38, 'FLT_MAX strength');
  assert.deepEqual(d.nodes[2].position, [0, 0, -1], 'JBeam +Y (rear) becomes Rubble −Z');
  assert.ok(calibratedSettings(240).threshold_scale < 1 && calibratedSettings(960).threshold_scale === 1);
});

/** In-memory backend shaped like world-host's (bodies, colliders, joints, both pools). */
function memBackend(vehicleSlots, softSlots) {
  const b = {
    bodies: [], colliders: [], joints: [], data: new Uint32Array(vehicleSlots * 4), soft: new Uint32Array(softSlots * 4),
    bodyCount: () => b.bodies.length, colliderCount: () => b.colliders.length, jointCount: () => b.joints.length,
    appendBodies: (l) => l.forEach((w) => b.bodies.push(w.slice())),
    appendColliders: (l) => l.forEach((w) => b.colliders.push(w.slice())),
    appendJoints: (l) => l.forEach((w) => b.joints.push(w.slice())),
    writeBody: (i, w) => { b.bodies[i] = w.slice(); },
    writeCollider: (i, w) => { b.colliders[i] = w.slice(); },
    writeSlots: (slot, w) => b.data.set(w, slot * 4),
    writeBytes: (off, w) => b.data.set(w, off / 4),
    writeSoftSlots: (slot, w) => b.soft.set(w, slot * 4),
  };
  return b;
}

test('a soft car spawns particles, colliders, beams, a rigid core and its joints', () => {
  const vehicles = new VehicleHost({}, 4000);
  const soft = new SoftHost({});
  const be = memBackend(vehicles.layout.totalSlots, soft.layout.totalSlots);
  be.bodies.push(new Uint32Array(28)); // ground
  const car = loadTruck(proceduralCar());
  const h = soft.add(be, vehicles, car, [0, 0.05, 0]);
  const e = soft.entries[h];
  assert.equal(e.firstBody, 1);
  assert.equal(e.particles, car.nodes.length);
  assert.equal(e.beams, car.beams.length);
  assert.ok(e.core, 'wheels give a rigid core');
  assert.equal(vehicles.vehicleCount, 1);
  // Particles carry BODY_PARTICLE and zero inverse inertia.
  const f = new Float32Array(be.bodies[1].buffer);
  assert.ok(be.bodies[1][11] & SOFT.BODY_PARTICLE);
  assert.deepEqual([f[8], f[9], f[10]], [0, 0, 0]);
  // Node colliders share a negative group with the core's colliders.
  const group = (c) => (c[2] | 0) >> 16;
  assert.ok(group(be.colliders[0]) < 0);
  assert.ok(be.colliders.every((c) => group(c) === group(be.colliders[0])));
  // Ball joints from the core to the eight axle nodes.
  assert.equal(be.joints.length - vehicles.wheelCount, 8);
  // Header: counts and the beam records' endpoints.
  const hdr = be.soft.subarray(0, 4);
  assert.deepEqual(Array.from(hdr.subarray(0, 2)), [1, car.beams.length]);
  const beamBase = soft.layout.beamBase * 4;
  assert.equal(be.soft[beamBase], 1 + car.beams[0].a);
  assert.equal(be.soft[beamBase + 1], 1 + car.beams[0].b);
});

test('the step plan lists every soft kernel and sizes binding 32', () => {
  const cap = Uint32Array.of(4096, 8192, 65536, 64, 131072, 32, 64, 64);
  const plan = new RigidStep(cap, capWords(DEFAULT_VEHICLE_CAPACITY), softCapWords(DEFAULT_SOFT_CAPACITY));
  const kernels = JSON.parse(plan.kernels());
  for (const k of ['soft_wake', 'soft_begin', 'soft_bounds', 'soft_collide_nodes', 'soft_collide_tris', 'soft_primal', 'soft_primal_fused', 'soft_chebyshev', 'soft_dual', 'soft_plastic', 'soft_break_groups', 'soft_sleep']) {
    assert.ok(kernels.includes(k), `${k} missing from the plan's kernels`);
  }
  const sizes = plan.buffer_sizes();
  assert.equal(sizes.length, 33);
  assert.equal(sizes[32], new SoftHost({}).layout.bytes);
  // Without a soft arena the plan emits no soft passes.
  const bare = new RigidStep(cap, capWords(DEFAULT_VEHICLE_CAPACITY));
  const cmds = bare.encode(Uint32Array.of(10, 10, 0, 0, 0));
  const names = JSON.parse(bare.kernels());
  for (let i = 0; i < cmds.length; i += 6) if (cmds[i] !== 0) assert.ok(!names[cmds[i + 1]].startsWith('soft_'));
});
