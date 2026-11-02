// vehicle.mjs is a thin wrapper over engine-vehicle compiled to wasm: the layout it
// exports must match the Slang layout. The pages run vehicles on World (engine-world-wasm),
// whose packing is engine-vehicle's VehicleHost, tested natively (crates/engine-vehicle-wasm,
// engine-world); the JS VehicleHost that replayed its op stream went with the JS host.
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { L, bitsf, RigidStep, ballJointWords, capWords, DEFAULT_VEHICLE_CAPACITY } from '../vehicle.mjs';

const slangPath = new URL('../../../slang/physics/vehicle_layout.slang', import.meta.url);

test('layout constants match slang/physics/vehicle_layout.slang', () => {
  const src = readFileSync(slangPath, 'utf8');
  const re = /static const uint (\w+) = (0x[0-9A-Fa-f]+|\d+)u;/g;
  let n = 0;
  for (const m of src.matchAll(re)) {
    const v = Number(m[2]);
    assert.ok(m[1] in L, `${m[1]} missing from vehicle.mjs`);
    assert.equal(L[m[1]], v, `${m[1]}: JS ${L[m[1]]} vs Slang ${v}`);
    n++;
  }
  assert.ok(n > 100, `parsed ${n} constants`);
});

test('the browser step replays rigid_plan with the vehicle passes', () => {
  assert.equal(L.JOINT_BYTES, 288);
  const plan = new RigidStep(Uint32Array.of(4096, 8192, 65536, 64, 131072, 32, 64, 64), capWords(DEFAULT_VEHICLE_CAPACITY));
  const kernels = JSON.parse(plan.kernels());
  for (const k of ['vehicle_input', 'wheel_query', 'vehicle_link', 'vehicle_prepare', 'vehicle_iterate', 'vehicle_post', 'warmstart_joints', 'solve_dual'])
    assert.ok(kernels.includes(k), `${k} in the plan`);
  assert.ok(!kernels.includes('sleep_island_gate_joints'), 'joint edges gate sleep inside sleep_island_gate');
  const sizes = Array.from(plan.buffer_sizes());
  assert.equal(sizes.length, 34, 'bindings 0..31, the soft arena at 32, the character / query / projectile pool at 33');
  assert.ok(sizes[31] > 16, 'vehicle pool at binding 31');
  assert.ok(sizes[33] > 16, 'character / query / projectile pool at binding 33');
  assert.equal(sizes[5], (64 + DEFAULT_VEHICLE_CAPACITY.maxWheels) * 288, 'user + tyre joints');
  const counts = Uint32Array.of(10, 10, 4, 1, 4);
  const host = new Uint32Array(plan.host_upload(counts).buffer.slice(0, 256));
  assert.equal(host[30] & 2, 2, 'vehicle sever grace (HOST_S(30) bit 1)');
  const cmds = plan.encode(counts);
  assert.equal(cmds.length % 6, 0);
  const names = [];
  for (let i = 0; i < cmds.length; i += 6) if (cmds[i] !== 0) names.push(kernels[cmds[i + 1]]);
  const at = (k) => names.indexOf(k);
  assert.ok(at('broadphase_query') < at('wheel_query') && at('wheel_query') < at('vehicle_link'));
  assert.ok(at('predict_bodies') < at('vehicle_prepare') && names.lastIndexOf('vehicle_iterate') > names.lastIndexOf('solve_dual'));
  assert.equal(plan.steps(), 1);
  plan.free();
});

test('D6 ball bolts are built in Rust', () => {
  const w = ballJointWords(3, 4, Float32Array.of(1, 0, 0), Float32Array.of(-1, 0, 0), 2.5e5, 0, 1e8);
  assert.equal(w.length * 4, L.JOINT_BYTES);
  assert.deepEqual([w[0], w[1]], [3, 4]);
  assert.equal(bitsf(w[4]), 1, 'anchor A');
});
