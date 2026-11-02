// soft.mjs wraps engine-softbody compiled to wasm (crates/engine-vehicle-wasm/src/soft.rs):
// the importers and the step plan's soft passes (the pages spawn soft vehicles on World;
// the JS SoftHost that replayed the spawn's op stream went with the JS host). GPU
// behaviour is tested natively (crates/engine-webgpu/tests/softbody.rs) and
// in Chrome (softbody-check.mjs).
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { RigidStep, capWords, DEFAULT_VEHICLE_CAPACITY, wasm } from '../vehicle.mjs';
import { SOFT, loadTruck, loadJbeam, proceduralCar, latticeCrate, calibratedSettings, softCapWords, DEFAULT_SOFT_CAPACITY } from '../soft.mjs';

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
  assert.ok(calibratedSettings(120).threshold_scale > 1 && calibratedSettings(240).threshold_scale === 1 && calibratedSettings(960).threshold_scale === 1);
});

test('the step plan lists every soft kernel and sizes binding 32', () => {
  const cap = Uint32Array.of(4096, 8192, 65536, 64, 131072, 32, 64, 64);
  const plan = new RigidStep(cap, capWords(DEFAULT_VEHICLE_CAPACITY), softCapWords(DEFAULT_SOFT_CAPACITY));
  const kernels = JSON.parse(plan.kernels());
  for (const k of ['soft_wake', 'soft_begin', 'soft_bounds', 'soft_collide_nodes', 'soft_collide_tris', 'soft_tri_planes', 'soft_pairs', 'soft_primal_bodies', 'soft_dual', 'soft_plastic', 'soft_break_groups', 'soft_sleep']) {
    assert.ok(kernels.includes(k), `${k} missing from the plan's kernels`);
  }
  const sizes = plan.buffer_sizes();
  assert.equal(sizes.length, 34);
  const soft = new wasm.WasmSoftHost(softCapWords(DEFAULT_SOFT_CAPACITY));
  assert.equal(sizes[32], JSON.parse(soft.layout()).bytes);
  soft.free();
  // Without a soft arena the plan emits no soft passes.
  const bare = new RigidStep(cap, capWords(DEFAULT_VEHICLE_CAPACITY));
  const cmds = bare.encode(Uint32Array.of(10, 10, 0, 0, 0));
  const names = JSON.parse(bare.kernels());
  for (let i = 0; i < cmds.length; i += 6) if (cmds[i] !== 0) assert.ok(!names[cmds[i + 1]].startsWith('soft_'));
});
