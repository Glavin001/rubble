// Minimal WebGPU host for the shared Slang→WGSL step: it replays the commands of
// engine_exec::rigid_plan (wasm RigidStep), the graph WgpuSim runs, on the canonical
// contact path, including the vehicle passes. Used by vehicle.html; demo.html keeps
// its own benchmark host.
import { loadShaderModules, pipelineBuilder } from './shader-loader.mjs';
import { VehicleHost, DEFAULT_VEHICLE_CAPACITY, vehicleLayout, capWords, fbits, L, RigidStep, ballJointWords } from './vehicle.mjs';
import { SoftHost, DEFAULT_SOFT_CAPACITY, softCapWords } from './soft.mjs';

// ---- record encoding (BodyState 112 B, Collider 64 B) for scene objects ----
export function bodyWords(b) {
  const u = new Uint32Array(28);
  const put = (i, v) => { u[i] = fbits(v); };
  b.position.forEach((x, i) => put(i, x)); put(3, b.invMass);
  b.orientation.forEach((x, i) => put(4 + i, x));
  b.invInertia.forEach((x, i) => put(8 + i, x)); u[11] = b.flags >>> 0;
  b.linearVel.forEach((x, i) => put(12 + i, x)); put(15, b.sleepTimer);
  b.angularVel.forEach((x, i) => put(16 + i, x)); u[19] = b.generation >>> 0;
  b.initialPos.forEach((x, i) => put(20 + i, x)); u[23] = b.initialPadBits >>> 0;
  b.initialRot.forEach((x, i) => put(24 + i, x));
  return u;
}

export function colliderWords(c) {
  const u = new Uint32Array(16);
  u[0] = c.body; u[1] = c.shape; u[2] = c.material; u[3] = c.generation;
  c.localPos.forEach((x, i) => { u[4 + i] = fbits(x); }); u[7] = fbits(c.density);
  c.localRot.forEach((x, i) => { u[8 + i] = fbits(x); });
  c.params.forEach((x, i) => { u[12 + i] = fbits(x); });
  return u;
}

export const PASS_ALIGN = 256;
const nextPow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };

// Joint records are engine_schema::Joint (D6, 288 B), built in Rust (ball_joint_words).
export const JOINT_BYTES = L.JOINT_BYTES;
// Scratch atomics (binding 21): the EVENT_BREAK ring starts at word 96 (engine-schema
// EVENT_RING_BASE): cursor, then [kind, joint, force bits, torque bits] records.
export const EVENT_RING_BASE = 96, EVENT_RING_CAPACITY = 1024, EVENT_BREAK = 3, EVENT_BEAM_BREAK = 5;

export function sceneCapacity(bodies, colliders) {
  const b = nextPow2(Math.max(bodies, 4096));
  return {
    maxBodies: b,
    maxColliders: nextPow2(Math.max(colliders, b * 2, 8192)),
    maxContacts: Math.min(nextPow2(Math.max(bodies * 16, 65536)), 1 << 21),
    maxJoints: 64,
    maxCandidates: Math.min(nextPow2(Math.max(bodies * 32, 131072)), 1 << 21),
    maxColors: 32,
    maxCommands: 64,
    maxHull: 64,
  };
}

function bindingsOf(wgsl) {
  const out = [];
  for (const line of wgsl.split('\n')) {
    const i = line.indexOf('@binding(');
    if (i < 0) continue;
    const n = Number(line.slice(i + 9, line.indexOf(')', i)));
    if (out.some((b) => b.index === n)) continue;
    out.push({ index: n, uniform: line.includes('var<uniform') });
  }
  return out.sort((a, b) => a.index - b.index);
}

export const DEFAULT_PARAMS = {
  dt: 1 / 60, gravityY: -10, iterations: 10, flags: 0, alpha: 0.99, gamma: 0.999, betaLin: 10000, betaAng: 100,
  kStart: 1000, kMaxN: 1e10, kMaxT: 1e6, contactOffset: 0.01, sleepLin: 0.15, sleepAng: 0.15, sleepTime: 0.5,
  residualExit: 1e-4, trustLin: 0.35, trustAng: 0.5, pivotFloor: 1e-6, gridCellSize: 2,
};

export async function createWorld(device, { cap: userCap, vehicleCap = DEFAULT_VEHICLE_CAPACITY, softCap = null, wgslBase = './generated/wgsl/', status = () => {} }) {
  // engine_exec::rigid_plan in wasm: buffer sizes, host uploads and the step's
  // commands, the same graph WgpuSim and MetalSim replay (canonical contacts).
  const vcap = { ...DEFAULT_VEHICLE_CAPACITY, ...vehicleCap };
  // Soft bodies (binding 32) only when asked for: softCap = {} takes the defaults.
  const scap = softCap ? { ...DEFAULT_SOFT_CAPACITY, ...softCap } : null;
  const plan = new RigidStep(
    Uint32Array.of(userCap.maxBodies, userCap.maxColliders, userCap.maxContacts, userCap.maxJoints,
      userCap.maxCandidates, userCap.maxColors, userCap.maxCommands, userCap.maxHull),
    capWords(vcap), scap ? softCapWords(scap) : undefined);
  // One tyre joint per vehicle wheel slot rides on top of the user joints.
  const cap = { ...userCap, maxJoints: userCap.maxJoints + vcap.maxWheels };
  const vlayout = vehicleLayout(vcap);
  const aabbTail = plan.aabb_tail();
  const sizes = Array.from(plan.buffer_sizes());
  const kernels = JSON.parse(plan.kernels());

  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  sizes.forEach((s, i) => { if (s > limit) throw new Error(`buffer ${i} needs ${s} bytes (> ${limit})`); });
  const usage = (i) => GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    | ((i === 6 || i === 22) ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE)
    | (i === 23 ? GPUBufferUsage.INDIRECT : 0);
  const bufs = sizes.map((size, i) => device.createBuffer({ label: 'b' + i, size, usage: usage(i) }));
  device.queue.writeBuffer(bufs[22], 0, plan.pass_table());

  const modules = await loadShaderModules(device, kernels.map((name) => ({ name, file: name })),
    { urlOf: (file) => `${wgslBase}${file}.wgsl`, status });
  const builder = pipelineBuilder(device, kernels.length, status);
  const pipes = [];
  for (const [i, name] of kernels.entries()) {
    const { src, module } = modules[i];
    const binds = bindingsOf(src);
    const layout = device.createBindGroupLayout({
      label: name,
      entries: binds.map((b) => ({
        binding: b.index, visibility: GPUShaderStage.COMPUTE,
        buffer: b.uniform ? { type: 'uniform', hasDynamicOffset: b.index === 22, minBindingSize: b.index === 22 ? 16 : undefined } : { type: 'storage' },
      })),
    });
    const pipeline = await builder.create(name, { label: name, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: name } });
    const bind = device.createBindGroup({
      label: name, layout,
      entries: binds.map((b) => ({ binding: b.index, resource: b.index === 22 ? { buffer: bufs[22], size: 16 } : { buffer: bufs[b.index] } })),
    });
    pipes.push({ pipeline, bind, dynamic: binds.some((b) => b.index === 22) });
  }
  await builder.finish();

  const params = { ...DEFAULT_PARAMS };
  const jointWords32 = JOINT_BYTES / 4;
  const world = {
    device, bufs, cap, params, vehicleCap: vcap, vlayout, aabbTail, plan,
    bodyCount: 0, colliderCount: 0, jointCount: 0, steps: 0,
    last: { contact: 0, candidate: 0, awake: 0, color: 0, manifold: 0, overflow: 0 },
    // CPU mirrors so a scene can be restored (R key) bit for bit.
    bodyWords: new Uint32Array(cap.maxBodies * 28),
    colliderWords: new Uint32Array(cap.maxColliders * 16),
    jointWords: new Uint32Array(cap.maxJoints * jointWords32),
    colliderInfo: [], // {body, radius, dynamic}
    vehicles: new VehicleHost(vcap, aabbTail),
    vehicleShadow: new Uint32Array(vlayout.totalSlots * 4),
    soft: scap ? new SoftHost(scap) : null,
    softShadow: null,
    hasFilters: false,
    eventsSeen: 0,
    beamEventsSeen: 0,
  };
  if (world.soft) world.softShadow = new Uint32Array(world.soft.layout.totalSlots * 4);

  const writeBodyRaw = (index, words) => {
    world.bodyWords.set(words, index * 28);
    device.queue.writeBuffer(bufs[0], index * 112, words);
    device.queue.writeBuffer(bufs[0], (cap.maxBodies + index) * 112, words);
  };
  const writeColliderRaw = (index, words) => {
    world.colliderWords.set(words, index * 16);
    device.queue.writeBuffer(bufs[2], index * 64, words);
  };
  world.backend = {
    bodyCount: () => world.bodyCount,
    colliderCount: () => world.colliderCount,
    jointCount: () => world.jointCount,
    appendJoints: (list) => list.forEach((words) => {
      if (world.jointCount >= cap.maxJoints) throw new Error('joint capacity exhausted');
      if (words.length !== jointWords32) throw new Error(`joint record is ${words.length * 4} B, not ${JOINT_BYTES}`);
      world.jointWords.set(words, world.jointCount * jointWords32);
      device.queue.writeBuffer(bufs[5], world.jointCount * JOINT_BYTES, words);
      world.jointCount++;
    }),
    appendBodies: (list) => list.forEach((w) => world.addBodyWords(w)),
    appendColliders: (list) => list.forEach((w) => world.addColliderWords(w)),
    writeBody: (i, w) => writeBodyRaw(i, w),
    writeCollider: (i, w) => writeColliderRaw(i, w),
    writeSlots: (slot, u32) => { world.vehicleShadow.set(u32, slot * 4); device.queue.writeBuffer(bufs[31], slot * 16, u32); },
    writeBytes: (offset, u32) => { world.vehicleShadow.set(u32, offset / 4); device.queue.writeBuffer(bufs[31], offset, u32); },
    writeSoftSlots: (slot, u32) => { world.softShadow.set(u32, slot * 4); device.queue.writeBuffer(bufs[32], slot * 16, u32); },
  };

  world.addBodyWords = (words) => {
    if (world.bodyCount >= cap.maxBodies) throw new Error('body capacity exhausted');
    const i = world.bodyCount++;
    writeBodyRaw(i, words);
    return i;
  };
  world.addColliderWords = (words) => {
    if (world.colliderCount >= cap.maxColliders) throw new Error('collider capacity exhausted');
    const i = world.colliderCount++;
    writeColliderRaw(i, words);
    // Category/mask (generation) or group (material bits 16..31): enable the GPU filter test.
    if (words[3] !== 0 || (words[2] >>> 16) !== 0) world.hasFilters = true;
    const f = new Float32Array(words.buffer, words.byteOffset, 16);
    const body = words[0];
    const inv = new Float32Array(world.bodyWords.buffer, body * 112, 28);
    world.colliderInfo[i] = { body, radius: f[15] > 0 ? f[15] : Math.hypot(f[12], f[13], f[14]), dynamic: inv[3] > 0, plane: words[1] === 4 };
    return i;
  };
  // Convenience: {p, q, invMass, invI, flags, v, w}.
  world.addBody = (b) => world.addBodyWords(bodyWords({
    position: b.p, invMass: b.invMass ?? 0, orientation: b.q || [0, 0, 0, 1], invInertia: b.invI || [0, 0, 0],
    flags: b.flags ?? 0, linearVel: b.v || [0, 0, 0], sleepTimer: 0, angularVel: b.w || [0, 0, 0], generation: 0,
    initialPos: b.p, initialPadBits: 0, initialRot: b.q || [0, 0, 0, 1],
  }));
  // D6 ball joint {bodyA, bodyB, anchorA, anchorB, breakForce, breakTorque, penalty}
  // (engine_schema::JointDesc::ball; break limits ≤ 0 = unbreakable).
  world.addBallJoint = (j) => {
    const id = world.jointCount;
    world.backend.appendJoints([ballJointWords(j.bodyA, j.bodyB, Float32Array.from(j.anchorA || [0, 0, 0]),
      Float32Array.from(j.anchorB || [0, 0, 0]), j.breakForce ?? 0, j.breakTorque ?? 0, j.penalty ?? 0)]);
    return id;
  };
  // Joints that broke since the last call: [{joint, force, torque}] (async readback
  // of the EVENT_BREAK ring in the scratch atomics).
  const ringBytes = (4 + 4 * EVENT_RING_CAPACITY) * 4;
  const ringRead = device.createBuffer({ size: ringBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let ringBusy = false;
  world.pollJointBreaks = async () => {
    if (ringBusy) return [];
    ringBusy = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufs[21], EVENT_RING_BASE * 4, ringRead, 0, ringBytes);
    device.queue.submit([enc.finish()]);
    await ringRead.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(ringRead.getMappedRange().slice(0));
    ringRead.unmap();
    ringBusy = false;
    const out = [];
    const cursor = w[0];
    let n = Math.max(world.eventsSeen, cursor - EVENT_RING_CAPACITY);
    const fl = new Float32Array(w.buffer);
    for (; n < cursor; n++) {
      const r = 4 + 4 * (n % EVENT_RING_CAPACITY);
      if (w[r] === EVENT_BREAK) out.push({ joint: w[r + 1], force: fl[r + 2], torque: fl[r + 3] });
    }
    world.eventsSeen = Math.max(world.eventsSeen, cursor);
    return out;
  };
  // Soft-body beams that broke since the last call: [{beam, force, softBody}]
  // (EVENT_BEAM_BREAK records of the same ring).
  world.pollBeamBreaks = async () => {
    if (ringBusy) return [];
    ringBusy = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufs[21], EVENT_RING_BASE * 4, ringRead, 0, ringBytes);
    device.queue.submit([enc.finish()]);
    await ringRead.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(ringRead.getMappedRange().slice(0));
    ringRead.unmap();
    ringBusy = false;
    const out = [];
    const cursor = w[0];
    const fl = new Float32Array(w.buffer);
    for (let n = Math.max(world.beamEventsSeen, cursor - EVENT_RING_CAPACITY); n < cursor; n++) {
      const r = 4 + 4 * (n % EVENT_RING_CAPACITY);
      if (w[r] === EVENT_BEAM_BREAK) out.push({ beam: w[r + 1], force: fl[r + 2], softBody: w[r + 3] });
    }
    world.beamEventsLost = Math.max(0, cursor - world.beamEventsSeen - EVENT_RING_CAPACITY);
    world.beamEventsSeen = Math.max(world.beamEventsSeen, cursor);
    return out;
  };
  // GPU → CPU copy of `bytes` from `buffer` at `offset` (Uint32Array).
  const readWords = async (buffer, offset, bytes) => {
    const rb = device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    if (bytes) enc.copyBufferToBuffer(buffer, offset, rb, 0, bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(rb.getMappedRange(0, bytes).slice(0));
    rb.unmap(); rb.destroy();
    return w;
  };
  // Soft body `h`'s beam records (16 words each: engine_softbody::model::Beam).
  world.readSoftBeams = async (h) => {
    const e = world.soft.entries[h];
    return readWords(bufs[32], (world.soft.layout.beamBase + 4 * e.firstBeam) * 16, e.beams * 64);
  };
  // Soft body `h`'s particle positions (Float32Array xyz).
  world.readSoftParticles = async (h) => {
    const e = world.soft.entries[h];
    const f = new Float32Array((await readWords(bufs[0], e.firstBody * 112, e.particles * 112)).buffer);
    const out = new Float32Array(e.particles * 3);
    for (let i = 0; i < e.particles; i++) out.set(f.subarray(i * 28, i * 28 + 3), i * 3);
    return out;
  };
  // Damage of soft body `h` (SoftHost.damage).
  world.readSoftDamage = async (h) => world.soft.damage(h, await world.readSoftBeams(h), await world.readSoftParticles(h));
  // Destruction events since the last call: beam breaks and the pieces they cut
  // loose ([{softBody, nodes, parts, mass}]). After an event-ring overflow every
  // soft body is resynchronised from a beam readback. Use this or pollBeamBreaks,
  // not both (they share the ring cursor).
  world.pollSoftEvents = async () => {
    const breaks = await world.pollBeamBreaks();
    const lost = world.beamEventsLost || 0;
    let pieces;
    if (lost > 0) {
      pieces = [];
      for (let h = 0; h < world.soft.entries.length; h++) pieces.push(...world.soft.syncBroken(h, await world.readSoftBeams(h)));
    } else {
      pieces = world.soft.noteBreaks(breaks);
    }
    return { breaks, pieces, lost };
  };
  // Soft vehicles (engine_softbody::SoftHost): `desc` from soft.mjs loadTruck.
  world.addSoftVehicle = (desc, p, q = [0, 0, 0, 1], opts = {}) => world.soft.add(world.backend, world.vehicles, desc, p, q, opts);
  world.addSoftObstacle = (collider) => world.soft.addObstacle(world.backend, collider);
  world.setSoftSettings = (s) => world.soft.setSettings(world.backend, s);
  world.addCollider = (c) => world.addColliderWords(colliderWords({
    body: c.body, shape: c.shape, material: c.material ?? 0, generation: 0, localPos: c.localPos || [0, 0, 0],
    density: 0, localRot: c.localRot || [0, 0, 0, 1], params: c.params,
  }));

  const mats = new Float32Array(64 * 4);
  world.materials = mats;
  world.setMaterial = (i, friction, contactOffset = 0.01) => {
    mats.set([friction, 0, contactOffset, 0], i * 4);
    device.queue.writeBuffer(bufs[3], 0, mats);
  };
  for (let i = 0; i < 64; i++) mats.set([0.6, 0, 0.01, 0], i * 4);
  device.queue.writeBuffer(bufs[3], 0, mats);

  // Grid cell = 2·max small radius over dynamics (rigid_plan::grid_cell): r > 2 m
  // bodies (vehicle chassis, smash balls) use the large-body list instead.
  world.refreshCell = () => {
    let r = 0.25;
    for (const c of world.colliderInfo) if (c && c.dynamic && !c.plane && c.radius <= 2.0) r = Math.max(r, c.radius);
    params.gridCellSize = Math.max(0.5, 2 * r * 1.0001);
  };

  // ---- per-step replay of the rigid_plan commands ----
  const PFIELDS = ['dt', 'gravityY', 'iterations', 'flags', 'alpha', 'gamma', 'betaLin', 'betaAng', 'kStart', 'kMaxN',
    'kMaxT', 'contactOffset', 'sleepLin', 'sleepAng', 'sleepTime', 'residualExit', 'trustLin', 'trustAng', 'pivotFloor', 'gridCellSize'];
  const paramBytes = new ArrayBuffer(80);
  const pv = new DataView(paramBytes);
  const upload = device.createBuffer({ size: 256 + 256 + 256, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const colorZero = new Uint32Array(128);
  const groups = (threads) => { const n = Math.ceil(Math.max(threads, 1) / 64); return [Math.min(n, 32768), Math.ceil(n / 32768), 1]; };
  const bindPipe = (p, k, passIdx) => {
    p.setPipeline(k.pipeline);
    if (k.dynamic) p.setBindGroup(0, k.bind, [passIdx * PASS_ALIGN]); else p.setBindGroup(0, k.bind);
  };

  world.encodeStep = (enc) => {
    world.refreshCell();
    PFIELDS.forEach((k, i) => (i === 2 || i === 3) ? pv.setUint32(i * 4, params[k], true) : pv.setFloat32(i * 4, params[k], true));
    plan.set_params(new Uint8Array(paramBytes));
    const soft = world.soft && world.soft.bodyCount > 0 ? world.soft.counts() : new Uint32Array(10);
    const counts = Uint32Array.of(world.bodyCount, world.colliderCount, world.jointCount,
      world.vehicles.vehicleCount, world.vehicles.wheelCount, ...soft, world.hasFilters ? 1 : 0,
      world.soft && world.soft.inner.dual() ? 1 : 0, world.soft ? world.soft.inner.sweeps() : 1);
    // Host words (binding 21), host uniform (binding 6), counter block (copy source).
    const host = plan.host_upload(counts);
    const uniformBytes = host.length - 256 - 80;
    device.queue.writeBuffer(upload, 0, host.subarray(0, 256));
    device.queue.writeBuffer(upload, 256, host.subarray(256, 256 + uniformBytes));
    device.queue.writeBuffer(upload, 512, host.subarray(256 + uniformBytes));
    enc.copyBufferToBuffer(upload, 0, bufs[21], 0, 256);
    enc.copyBufferToBuffer(upload, 256, bufs[6], 0, uniformBytes);
    device.queue.writeBuffer(bufs[17], 0, colorZero);
    const cmds = plan.encode(counts);
    for (let i = 0; i < cmds.length; i += 6) {
      if (cmds[i] !== 0) continue;
      const src = cmds[i + 1] === 0xFFFFFFFF ? upload : bufs[cmds[i + 1]];
      const srcOff = cmds[i + 1] === 0xFFFFFFFF ? 512 + cmds[i + 2] : cmds[i + 2];
      enc.copyBufferToBuffer(src, srcOff, bufs[cmds[i + 3]], cmds[i + 4], cmds[i + 5]);
    }
    const p = enc.beginComputePass();
    for (let i = 0; i < cmds.length; i += 6) {
      const op = cmds[i];
      if (op === 1) {
        bindPipe(p, pipes[cmds[i + 1]], cmds[i + 3]);
        p.dispatchWorkgroups(...groups(cmds[i + 2]));
      } else if (op === 2) {
        bindPipe(p, pipes[cmds[i + 1]], cmds[i + 3]);
        p.dispatchWorkgroupsIndirect(bufs[23], cmds[i + 2] * 12);
      }
    }
    p.end();
    world.steps = plan.steps();
    world.vehicles.dirty = false;
  };
  // Counters read back by the page (binding 7 words) size the next steps.
  world.setLastCounters = (words) => {
    plan.set_last(words);
    world.last.contact = words[2];
  };

  // Snapshot of the uploaded scene (bodies, colliders, joints, vehicle records) for reset.
  world.snapshot = () => ({
    bodyCount: world.bodyCount, colliderCount: world.colliderCount,
    bodies: world.bodyWords.slice(0, world.bodyCount * 28),
    jointCount: world.jointCount,
    joints: world.jointWords.slice(0, world.jointCount * jointWords32),
    colliders: world.colliderWords.slice(0, world.colliderCount * 16),
    vehicle: world.vehicleShadow.slice(),
    host: world.vehicles.save(),
    soft: world.soft ? world.softShadow.slice() : null,
    softHost: world.soft ? world.soft.save() : null,
    hasFilters: world.hasFilters,
  });
  world.restore = (snap) => {
    world.bodyCount = snap.bodyCount;
    world.colliderCount = snap.colliderCount;
    world.bodyWords.set(snap.bodies);
    world.colliderWords.set(snap.colliders);
    device.queue.writeBuffer(bufs[0], 0, snap.bodies);
    device.queue.writeBuffer(bufs[0], cap.maxBodies * 112, snap.bodies);
    device.queue.writeBuffer(bufs[2], 0, snap.colliders);
    // Joints come back unbroken with their initial multipliers.
    world.jointCount = snap.jointCount;
    world.jointWords.set(snap.joints);
    if (snap.joints.length) device.queue.writeBuffer(bufs[5], 0, snap.joints);
    world.vehicleShadow.set(snap.vehicle);
    device.queue.writeBuffer(bufs[31], 0, snap.vehicle);
    world.vehicles.restore(snap.host);
    if (snap.soft) {
      world.softShadow.set(snap.soft);
      device.queue.writeBuffer(bufs[32], 0, snap.soft);
      world.soft.restore(snap.softHost);
    }
    world.hasFilters = snap.hasFilters;
    plan.reset();
    world.steps = 0;
    world.last = { contact: 0, candidate: 0, awake: 0, color: 0, manifold: 0, overflow: 0 };
  };

  return world;
}
