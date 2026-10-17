// vehicle.mjs is a thin wrapper over engine-vehicle compiled to wasm: the layout it
// exports must match the Slang layout, and the op stream it replays must allocate
// what the Rust packer asks for. Bit-exactness of the replay against VehicleHost is
// tested natively in crates/engine-vehicle-wasm.
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { L, VehicleHost, requirements, bitsf, RigidStep, ballJointWords, capWords, DEFAULT_VEHICLE_CAPACITY } from '../vehicle.mjs';

const slangPath = new URL('../../../slang/physics/vehicle_layout.slang', import.meta.url);
const presetsPath = new URL('../generated/vehicle_presets.json', import.meta.url);

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

/** In-memory backend shaped like world-host's. */
function memBackend(totalSlots) {
  const b = {
    bodies: [], colliders: [], joints: [], data: new Uint32Array(totalSlots * 4),
    bodyCount: () => b.bodies.length, colliderCount: () => b.colliders.length, jointCount: () => b.joints.length,
    appendBodies: (l) => l.forEach((w) => b.bodies.push(w.slice())),
    appendColliders: (l) => l.forEach((w) => b.colliders.push(w.slice())),
    appendJoints: (l) => l.forEach((w) => b.joints.push(w.slice())),
    writeBody: (i, w) => { b.bodies[i] = w.slice(); },
    writeCollider: (i, w) => { b.colliders[i] = w.slice(); },
    writeSlots: (slot, w) => b.data.set(w, slot * 4),
    writeBytes: (off, w) => b.data.set(w, off / 4),
  };
  return b;
}

test('every preset packs through the wasm host', () => {
  const presets = JSON.parse(readFileSync(presetsPath, 'utf8'));
  const host = new VehicleHost({}, 4000);
  const be = memBackend(host.layout.totalSlots);
  be.bodies.push(new Uint32Array(28)); // a ground body first: non-zero bases
  for (const desc of presets) {
    const req = requirements(desc);
    const [b0, c0, j0] = [be.bodyCount(), be.colliderCount(), be.jointCount()];
    const h = host.addVehicle(be, desc, [1, 2, 3], [0, 0.149438, 0, 0.988771]);
    assert.equal(be.bodyCount() - b0, req.bodies, `${desc.name}: bodies`);
    assert.equal(be.colliderCount() - c0, req.colliders, `${desc.name}: colliders`);
    assert.equal(be.jointCount() - j0, req.wheelSlots, `${desc.name}: tyre joints`);
    assert.ok(be.bodies.every((w) => w.length === 28) && be.colliders.every((w) => w.length === 16));
    assert.ok(be.joints.every((w) => w.length * 4 === L.JOINT_BYTES && w[2] === L.JOINT_FLAG_TYRE), 'tyre joints are D6 records (engine_schema::Joint) flagged JOINT_FLAG_TYRE');
    const e = host.entries[h];
    assert.equal(e.chassis, b0);
    assert.equal(e.wheelSlots, req.wheelSlots);
    assert.equal(e.desc.name, desc.name);
    assert.equal(be.bodies[b0][11] & L.BODY_VEHICLE, L.BODY_VEHICLE, `${desc.name}: chassis flag`);
    const vrec = (host.layout.vehicleBase + h * L.VEH_STRIDE) * 4;
    assert.equal(be.data[vrec + L.V_CFG * 4], b0, `${desc.name}: V_CFG chassis`);
    assert.equal(be.data[vrec + L.V_CFG * 4 + 2], req.wheelSlots, `${desc.name}: V_CFG wheel count`);
    // Fresh records decode: wheels at full droop, no contact.
    const st = host.readState(be.data, h);
    assert.equal(st.wheels.length, desc.wheels.length);
    st.wheels.forEach((w, i) => {
      assert.ok(Math.abs(w.radius - desc.wheels[i].radius) < 1e-6, `${desc.name}: wheel ${i} radius`);
      assert.equal(w.inContact, false);
    });
  }
  assert.equal(host.vehicleCount, presets.length);
  assert.deepEqual(Array.from(be.data.subarray(0, 4)), [64, 512, presets.length, host.wheelCount], 'header');
});

test('controls, snapshots and spare wheels go through Rust', () => {
  const presets = JSON.parse(readFileSync(presetsPath, 'utf8'));
  const desc = presets.find((p) => p.spare_wheels > 0);
  assert.ok(desc, 'a preset with spare wheel slots');
  const host = new VehicleHost({}, 100);
  const be = memBackend(host.layout.totalSlots);
  const h = host.addVehicle(be, desc, [0, 1, 0]);
  const in0 = (host.layout.vehicleBase + h * L.VEH_STRIDE + L.V_IN0) * 4;
  host.setInput(be, h, { forward: 0.5, steer: -0.25, gear: 2 });
  assert.equal(bitsf(be.data[in0]), 0.5);
  assert.equal(bitsf(be.data[in0 + 1]), -0.25);
  assert.equal(be.data[in0 + 4], 2);
  host.dirty = false;
  host.setInput(be, h, { forward: 0.5, steer: -0.25, gear: 2 });
  assert.equal(host.dirty, false, 'unchanged input writes nothing');
  host.setInput(be, h, {});
  assert.equal(be.data[in0 + 4], L.GEAR_AUTO);
  const snap = host.save();
  host.detachWheel(be, h, 1);
  assert.equal(be.data[in0 + 6], 2);
  const before = host.entries[h].desc.wheels.length;
  const idx = host.addWheel(be, h, { ...desc.wheels[0], position: [0.4, 0.2, 1.0] }, be.data);
  assert.equal(idx, before);
  assert.equal(host.entries[h].desc.wheels.length, before + 1);
  assert.deepEqual(host.entries[h].desc.wheels[idx].position, [0.4, 0.2, 1.0].map(Math.fround));
  const rec = (host.layout.wheelBase + (host.entries[h].slots.wheelBase + idx) * L.WHEEL_STRIDE) * 4;
  assert.equal(be.data[rec + L.W_C0 * 4 + 2], L.WD_PENDING_ADD);
  host.restore(snap);
  assert.equal(host.entries[h].desc.wheels.length, before);
  assert.equal(host.dirty, true);
});

test('the browser step replays rigid_plan with the vehicle passes', () => {
  assert.equal(L.JOINT_BYTES, 288);
  const plan = new RigidStep(Uint32Array.of(4096, 8192, 65536, 64, 131072, 32, 64, 64), capWords(DEFAULT_VEHICLE_CAPACITY));
  const kernels = JSON.parse(plan.kernels());
  for (const k of ['vehicle_input', 'wheel_query', 'vehicle_link', 'vehicle_prepare', 'vehicle_iterate', 'vehicle_post', 'warmstart_joints', 'solve_dual'])
    assert.ok(kernels.includes(k), `${k} in the plan`);
  assert.ok(!kernels.includes('sleep_island_gate_joints'), 'joint edges gate sleep inside sleep_island_gate');
  const sizes = Array.from(plan.buffer_sizes());
  assert.equal(sizes.length, 32);
  assert.ok(sizes[31] > 16, 'vehicle pool at binding 31');
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
