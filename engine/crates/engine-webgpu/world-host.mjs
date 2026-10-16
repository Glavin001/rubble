// Minimal WebGPU host for the shared Slang→WGSL step, mirroring WgpuSim::step
// (crates/engine-webgpu/src/gpu.rs) on the canonical contact path, including the
// vehicle passes. Used by vehicle.html; demo.html keeps its own benchmark host.
import { VehicleHost, DEFAULT_VEHICLE_CAPACITY, vehicleLayout, fbits } from './vehicle.mjs';

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
const BIND_COUNT = 30;
const nextPow2 = (n) => { let p = 1; while (p < n) p <<= 1; return p; };

export const KERNELS = [
  'build_aabbs', 'broadphase_clear', 'broadphase_insert', 'broadphase_query', 'write_work_args',
  'update_refs', 'narrowphase', 'contact_hash_clear', 'contact_hash_insert', 'persist_contacts',
  'build_awake_list', 'csr_clear', 'csr_degree_contacts', 'csr_degree_joints', 'csr_prefix', 'csr_prefix_blocks',
  'csr_prefix_block_sums', 'csr_prefix_add', 'csr_scatter_contacts', 'csr_scatter_joints', 'csr_sort', 'init_colors',
  'joint_hash_clear', 'joint_hash_insert', 'warmstart_joints',
  'color_bodies', 'count_color_tiles', 'prefix_color_tiles_parallel', 'scatter_color_tiles',
  'write_color_args', 'predict_bodies', 'solve_primal_color', 'solve_dual', 'extract_velocity_sleep',
  'write_sleep_args', 'sleep_island_gate', 'sleep_island_gate_joints', 'sleep_island_commit',
  'vehicle_input', 'wheel_query', 'vehicle_link', 'vehicle_prepare', 'vehicle_iterate', 'vehicle_post', 'raycast_batch',
];

// Joint record (engine-schema `Joint`, 160 B): ids, anchors, axes, C0, break force,
// λ[6], k[6], angular C0, break torque. JOINT_FLAG_BROKEN is set by solve_dual when it snaps.
export const JOINT_WORDS = 40;
export const JOINT_BALL = 0, JOINT_FIXED = 1, JOINT_HINGE = 2;
export const JOINT_FLAG_BROKEN = 0x80000000;
// gEvents: 64k joint pair hash, then the event ring (engine-schema EVENT_RING_*).
export const EVENT_RING_BASE = 65536, EVENT_RING_CAPACITY = 1024, EVENT_BREAK = 3;
const EVENTS_WORDS = EVENT_RING_BASE + 4 + 4 * EVENT_RING_CAPACITY;
const LARGE_STATIC_CAP = 1024;
export function jointWords(j) {
  const u = new Uint32Array(JOINT_WORDS);
  const f = new Float32Array(u.buffer);
  u[0] = j.bodyA; u[1] = j.bodyB; u[2] = j.jointType ?? JOINT_BALL; u[3] = j.flags || 0;
  f.set(j.anchorA || [0, 0, 0], 4); f[7] = j.stiffness ?? 1e10;
  f.set(j.anchorB || [0, 0, 0], 8); f[11] = j.damping ?? 0;
  f.set(j.axisA || [0, 1, 0], 12); f[15] = j.limitMin ?? 0;
  f.set(j.axisB || [0, 1, 0], 16); f[19] = j.limitMax ?? 0;
  f[23] = j.breakForce ?? 0;
  f.fill(j.k ?? 1e4, 30, 36);
  f[39] = j.breakTorque ?? 0;
  return u;
}

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

export async function createWorld(device, { cap: userCap, vehicleCap = DEFAULT_VEHICLE_CAPACITY, wgslBase = './generated/wgsl/', status = () => {} }) {
  // One tyre joint per vehicle wheel slot rides on top of the user joints.
  const cap = { ...userCap, maxJoints: userCap.maxJoints + vehicleCap.maxWheels };
  const constraintIdWords = nextPow2(Math.max(32768, 2 * (cap.maxContacts + cap.maxJoints)));
  const csrBase = Math.max(constraintIdWords, 131072);
  const vlayout = vehicleLayout(vehicleCap);
  const aabbTail = cap.maxBodies + cap.maxColliders;

  const sizes = new Array(BIND_COUNT).fill(256);
  sizes[0] = cap.maxBodies * 2 * 112;
  sizes[1] = cap.maxBodies * 112;
  sizes[2] = cap.maxColliders * 64;
  sizes[3] = 64 * 16;
  sizes[4] = cap.maxContacts * 2 * 128;
  sizes[5] = cap.maxJoints * 256;
  sizes[8] = cap.maxCommands * 64;
  // Collider AABBs, then a 4-entry block per vehicle and per wheel slot (ground blocks).
  // Then the oversized-static list (engine-schema LARGE_STATIC_CAP, host scalar 29).
  const largeStaticBase = cap.maxBodies + cap.maxColliders + 4 * (vehicleCap.maxVehicles + vehicleCap.maxWheels);
  sizes[9] = (largeStaticBase + LARGE_STATIC_CAP) * 32;
  sizes[10] = cap.maxCandidates * 16;
  sizes[11] = cap.maxBodies * 4;
  sizes[12] = cap.maxBodies * 4;
  sizes[13] = (cap.maxBodies + 1) * 4;
  sizes[14] = (csrBase + cap.maxBodies + 1) * 4;
  sizes[15] = cap.maxBodies * (cap.maxColors + 2) * 4;
  sizes[16] = cap.maxColors * Math.max(1, Math.ceil(cap.maxBodies / 64)) * 4;
  sizes[17] = cap.maxColors * 4;
  sizes[18] = cap.maxHull * 16;
  sizes[19] = EVENTS_WORDS * 4;
  sizes[20] = cap.maxBodies * 4;
  sizes[21] = 64 * 4;
  sizes[22] = 96 * PASS_ALIGN;
  sizes[23] = Math.max(cap.maxColors, 73) * 3 * 4;
  sizes[24] = cap.maxColliders * 80;
  sizes[25] = cap.maxCandidates * 32;
  sizes[26] = cap.maxBodies * 32;
  sizes[27] = 64;
  sizes[28] = 32;
  sizes[29] = vlayout.totalSlots * 16;
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  sizes.forEach((s, i) => { if (s > limit) throw new Error(`buffer ${i} needs ${s} bytes (> ${limit})`); });
  const usage = (i) => GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    | ((i === 6 || i === 22) ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE)
    | (i === 23 ? GPUBufferUsage.INDIRECT : 0);
  const bufs = sizes.map((size, i) => device.createBuffer({ label: 'b' + i, size, usage: usage(i) }));

  const pass = new Uint8Array(96 * PASS_ALIGN);
  const pv = new DataView(pass.buffer);
  for (let i = 0; i < 96; i++) {
    let v = i;
    if (i >= 48 && i < 68) { const k = i - 48; v = (1 << k) | ((k & 1) << 31); } else if (i >= 68) v = 0;
    pv.setUint32(i * PASS_ALIGN, v >>> 0, true);
  }
  device.queue.writeBuffer(bufs[22], 0, pass);

  const pipes = {};
  for (const name of KERNELS) {
    status(`compiling ${name}`);
    const src = await (await fetch(`${wgslBase}${name}.wgsl`, { cache: 'no-store' })).text();
    const module = device.createShaderModule({ label: name, code: src });
    if (module.getCompilationInfo) {
      const info = await module.getCompilationInfo();
      const err = info.messages.find((m) => m.type === 'error');
      if (err) throw new Error(`${name}: ${err.message}`);
    }
    const binds = bindingsOf(src);
    const layout = device.createBindGroupLayout({
      label: name,
      entries: binds.map((b) => ({
        binding: b.index, visibility: GPUShaderStage.COMPUTE,
        buffer: b.uniform ? { type: 'uniform', hasDynamicOffset: b.index === 22, minBindingSize: b.index === 22 ? 16 : undefined } : { type: 'storage' },
      })),
    });
    device.pushErrorScope('validation');
    const pipeline = device.createComputePipeline({ label: name, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: name } });
    const e = await device.popErrorScope();
    if (e) throw new Error(`${name}: ${e.message}`);
    const bind = device.createBindGroup({
      label: name, layout,
      entries: binds.map((b) => ({ binding: b.index, resource: b.index === 22 ? { buffer: bufs[22], size: 16 } : { buffer: bufs[b.index] } })),
    });
    pipes[name] = { pipeline, bind, dynamic: binds.some((b) => b.index === 22) };
    await new Promise((r) => setTimeout(r, 0));
  }

  const params = { ...DEFAULT_PARAMS };
  const world = {
    device, bufs, cap, params, vehicleCap, vlayout, aabbTail, csrBase,
    bodyCount: 0, colliderCount: 0, steps: 0, parity: 0,
    last: { contact: 0, candidate: 0, awake: 0, color: 0, manifold: 0, overflow: 0 },
    // CPU mirrors so a scene can be restored (R key) bit for bit.
    bodyWords: new Uint32Array(cap.maxBodies * 28),
    colliderWords: new Uint32Array(cap.maxColliders * 16),
    colliderInfo: [], // {body, radius, dynamic}
    vehicles: new VehicleHost(vehicleCap, aabbTail),
    vehicleShadow: new Uint32Array(vlayout.totalSlots * 4),
  };

  const writeBodyRaw = (index, words) => {
    world.bodyWords.set(words, index * 28);
    device.queue.writeBuffer(bufs[0], index * 112, words);
    device.queue.writeBuffer(bufs[0], (cap.maxBodies + index) * 112, words);
  };
  const writeColliderRaw = (index, words) => {
    world.colliderWords.set(words, index * 16);
    device.queue.writeBuffer(bufs[2], index * 64, words);
  };
  world.jointCount = 0;
  world.userJointCount = 0;
  world.jointWords = new Uint32Array(cap.maxJoints * JOINT_WORDS);
  world.eventsSeen = 0;
  world.backend = {
    bodyCount: () => world.bodyCount,
    colliderCount: () => world.colliderCount,
    jointCount: () => world.jointCount,
    // Shorter records (older tyre joint words) are zero-padded to the 160 B stride.
    appendJoints: (list) => list.forEach((words) => {
      if (world.jointCount >= cap.maxJoints) throw new Error('joint capacity exhausted');
      const full = new Uint32Array(JOINT_WORDS);
      full.set(words.subarray(0, JOINT_WORDS));
      world.jointWords.set(full, world.jointCount * JOINT_WORDS);
      device.queue.writeBuffer(bufs[5], world.jointCount * JOINT_WORDS * 4, full);
      if (full[2] !== 7) world.userJointCount++;
      world.jointCount++;
      world.jointHashClearPending = true;
    }),
    appendBodies: (list) => list.forEach((w) => world.addBodyWords(w)),
    appendColliders: (list) => list.forEach((w) => world.addColliderWords(w)),
    writeBody: (i, w) => writeBodyRaw(i, w),
    writeCollider: (i, w) => writeColliderRaw(i, w),
    writeSlots: (slot, u32) => { world.vehicleShadow.set(u32, slot * 4); device.queue.writeBuffer(bufs[29], slot * 16, u32); },
    writeBytes: (offset, u32) => { world.vehicleShadow.set(u32, offset / 4); device.queue.writeBuffer(bufs[29], offset, u32); },
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
    const f = new Float32Array(words.buffer, words.byteOffset, 16);
    const body = words[0];
    const inv = new Float32Array(world.bodyWords.buffer, body * 112, 28);
    const flags = world.bodyWords[body * 28 + 11];
    world.colliderInfo[i] = { body, radius: f[15] > 0 ? f[15] : Math.hypot(f[12], f[13], f[14]), dynamic: inv[3] > 0 && (flags & 1) !== 0, plane: words[1] === 4 };
    return i;
  };
  // Convenience: {p, q, invMass, invI, flags, v, w}.
  world.addBody = (b) => world.addBodyWords(bodyWords({
    position: b.p, invMass: b.invMass ?? 0, orientation: b.q || [0, 0, 0, 1], invInertia: b.invI || [0, 0, 0],
    flags: b.flags ?? 0, linearVel: b.v || [0, 0, 0], sleepTimer: 0, angularVel: b.w || [0, 0, 0], generation: 0,
    initialPos: b.p, initialPadBits: 0, initialRot: b.q || [0, 0, 0, 1],
  }));
  // {bodyA, bodyB, jointType, anchorA, anchorB, axisA, axisB, breakForce, breakTorque}.
  world.addJoint = (j) => { const id = world.jointCount; world.backend.appendJoints([jointWords(j)]); return id; };
  // Joints that broke since the last call: [{joint, force, torque}] (async readback).
  const ringBytes = (EVENTS_WORDS - EVENT_RING_BASE) * 4;
  const ringRead = device.createBuffer({ size: ringBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let ringBusy = false;
  world.pollJointBreaks = async () => {
    if (ringBusy) return [];
    ringBusy = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufs[19], EVENT_RING_BASE * 4, ringRead, 0, ringBytes);
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

  // Grid cell = 2·max small radius over dynamics (WgpuSim::refresh_cell): r > 2.5 m
  // bodies (vehicle chassis, smash balls) use the large-body list instead.
  world.refreshCell = () => {
    let r = 0.25;
    for (const c of world.colliderInfo) if (c && c.dynamic && !c.plane && c.radius <= 2.5) r = Math.max(r, c.radius);
    params.gridCellSize = Math.max(0.5, 2 * r * 1.0001);
  };

  // ---- per-step host uniforms (WgpuSim::write_host_uniforms) ----
  const hostUpload = device.createBuffer({ size: 256 + 256 + 64 + cap.maxColors * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const hostCpu = new ArrayBuffer(256 + 256 + 64 + cap.maxColors * 4);
  const scratch = new Uint32Array(hostCpu, 0, 64);
  const hostParams = new DataView(hostCpu, 256, 208);
  const counters = new Uint32Array(hostCpu, 512, 16);
  const colorZero = new Uint32Array(hostCpu, 576, cap.maxColors);
  const warmHashSlots = (prev) => Math.min(nextPow2(Math.max(prev * 4, 1024)), csrBase, 1 << 21);
  const gridHashWords = () => Math.min(nextPow2(Math.max(2 * world.colliderCount, 131072)), csrBase);
  const PFIELDS = ['dt', 'gravityY', 'iterations', 'flags', 'alpha', 'gamma', 'betaLin', 'betaAng', 'kStart', 'kMaxN',
    'kMaxT', 'contactOffset', 'sleepLin', 'sleepAng', 'sleepTime', 'residualExit', 'trustLin', 'trustAng', 'pivotFloor', 'gridCellSize'];

  function writeHost(enc) {
    const prevN = Math.min(Math.max(world.last.contact, 256), cap.maxContacts);
    const s = scratch;
    s.fill(0);
    s[1] = cap.maxBodies; s[2] = cap.maxCandidates; s[3] = cap.maxContacts;
    s[7] = world.steps === 0 ? 0 : Math.min(world.last.contact, cap.maxContacts);
    s[8] = world.parity === 0 ? cap.maxContacts : 0;
    s[9] = warmHashSlots(prevN);
    s[10] = cap.maxCandidates;
    s[11] = world.parity === 0 ? 0 : cap.maxContacts;
    let log = 0;
    while ((1 << log) < Math.max(world.bodyCount, 1) && log < 20) log++;
    s[12] = log === 0 ? 0 : (log - 1) & 1;
    s[16] = gridHashWords();
    s[17] = world.steps;
    s[24] = csrBase;
    s[26] = 1;
    s[29] = largeStaticBase;
    s[4] = Math.min(cap.maxColors, 32);
    PFIELDS.forEach((k, i) => (i === 2 || i === 3) ? hostParams.setUint32(i * 4, params[k], true) : hostParams.setFloat32(i * 4, params[k], true));
    for (let i = 0; i < 32; i++) hostParams.setUint32(80 + i * 4, s[i], true);
    counters.fill(0);
    counters[0] = world.bodyCount;
    counters[1] = world.colliderCount;
    counters[3] = world.jointCount;
    colorZero.fill(0);
    device.queue.writeBuffer(hostUpload, 0, hostCpu);
    enc.copyBufferToBuffer(hostUpload, 0, bufs[21], 0, 256);
    enc.copyBufferToBuffer(hostUpload, 256, bufs[6], 0, 208);
    // Previous GPU contact count (pipelined frames see stale CPU counters).
    if (world.steps > 0) enc.copyBufferToBuffer(bufs[7], 8, bufs[6], 80 + 7 * 4, 4);
    enc.copyBufferToBuffer(hostUpload, 512, bufs[7], 0, 64);
    enc.copyBufferToBuffer(hostUpload, 576, bufs[17], 0, cap.maxColors * 4);
  }

  const groups = (threads) => { const n = Math.ceil(Math.max(threads, 1) / 64); return [Math.min(n, 32768), Math.ceil(n / 32768), 1]; };
  const disp = (p, name, threads, passIdx = 0) => {
    const k = pipes[name];
    p.setPipeline(k.pipeline);
    if (k.dynamic) p.setBindGroup(0, k.bind, [passIdx * PASS_ALIGN]); else p.setBindGroup(0, k.bind);
    p.dispatchWorkgroups(...groups(threads));
  };
  const dispWork = (p, name, slot, passIdx = 0) => {
    const k = pipes[name];
    p.setPipeline(k.pipeline);
    if (k.dynamic) p.setBindGroup(0, k.bind, [passIdx * PASS_ALIGN]); else p.setBindGroup(0, k.bind);
    p.dispatchWorkgroupsIndirect(bufs[23], slot * 12);
  };

  world.encodeStep = (enc) => {
    world.refreshCell();
    const nB = Math.max(world.bodyCount, 1);
    const nC = Math.max(world.colliderCount, 1);
    const nV = world.vehicles.vehicleCount;
    const nW = world.vehicles.wheelCount;
    const prevN = Math.min(Math.max(world.last.contact, 256), cap.maxContacts);
    const hashN = warmHashSlots(prevN);
    writeHost(enc);
    const p = enc.beginComputePass();
    if (nV > 0) disp(p, 'vehicle_input', nV);
    disp(p, 'build_aabbs', nC);
    // Joint pair filter, rebuilt each step so a broken joint's bodies collide again.
    // Tyre joints skip it: with no user joints it only needs one clear.
    if (world.userJointCount > 0 || world.jointHashClearPending) {
      disp(p, 'joint_hash_clear', 65536);
      if (world.userJointCount > 0) disp(p, 'joint_hash_insert', world.jointCount);
      world.jointHashClearPending = false;
    }
    disp(p, 'broadphase_clear', gridHashWords());
    disp(p, 'broadphase_insert', nC);
    disp(p, 'broadphase_query', nC);
    if (nV > 0) {
      disp(p, 'wheel_query', nW * 64); // grid walk: before the contact hash reuses the heads
      disp(p, 'vehicle_link', nW); // wheels on dynamic bodies join the constraint graph
    }
    disp(p, 'write_work_args', 64);
    const reuse = (params.flags & 1) !== 0;
    if (reuse) {
      disp(p, 'update_refs', nB);
      disp(p, 'contact_hash_clear', hashN);
      dispWork(p, 'contact_hash_insert', 68);
    }
    dispWork(p, 'narrowphase', 64);
    disp(p, 'write_work_args', 64);
    if (!reuse) {
      disp(p, 'contact_hash_clear', hashN);
      dispWork(p, 'contact_hash_insert', 68);
    }
    dispWork(p, 'persist_contacts', 65);
    if (world.jointCount > 0) disp(p, 'warmstart_joints', world.jointCount);
    disp(p, 'build_awake_list', nB);
    disp(p, 'write_work_args', 64);
    disp(p, 'csr_clear', nB + 1);
    dispWork(p, 'csr_degree_contacts', 66);
    if (world.jointCount > 0) disp(p, 'csr_degree_joints', world.jointCount);
    if (world.bodyCount <= 512) disp(p, 'csr_prefix', 1);
    else {
      disp(p, 'csr_prefix_blocks', Math.ceil(nB / 4));
      disp(p, 'csr_prefix_block_sums', 64);
      disp(p, 'csr_prefix_add', nB);
    }
    dispWork(p, 'csr_scatter_contacts', 66);
    if (world.jointCount > 0) disp(p, 'csr_scatter_joints', world.jointCount);
    dispWork(p, 'csr_sort', 67);
    dispWork(p, 'init_colors', 67);
    disp(p, 'write_work_args', 64);
    for (let round = 0; round < 32; round++) dispWork(p, 'color_bodies', 69, round % 16);
    disp(p, 'count_color_tiles', nB);
    disp(p, 'prefix_color_tiles_parallel', Math.min(cap.maxColors, 32) * 64);
    disp(p, 'scatter_color_tiles', nB);
    disp(p, 'write_color_args', 64);
    disp(p, 'predict_bodies', nB);
    if (nV > 0) disp(p, 'vehicle_prepare', nV);
    const colors = Math.min(cap.maxColors, 32);
    for (let iter = 0; iter < params.iterations; iter++) {
      const primal = pipes.solve_primal_color;
      for (let c = 0; c < colors; c++) {
        p.setPipeline(primal.pipeline);
        p.setBindGroup(0, primal.bind, [c * PASS_ALIGN]);
        p.dispatchWorkgroupsIndirect(bufs[23], c * 12);
      }
      dispWork(p, 'solve_dual', 65, iter + 1 === params.iterations ? 1 : 0);
      if (nV > 0) disp(p, 'vehicle_iterate', nV);
    }
    disp(p, 'extract_velocity_sleep', nB);
    if (nV > 0) disp(p, 'vehicle_post', nV);
    disp(p, 'write_sleep_args', 64);
    for (let i = 0; i < 8; i++) {
      dispWork(p, 'sleep_island_gate', 66);
      if (world.jointCount > 0) disp(p, 'sleep_island_gate_joints', world.jointCount);
    }
    dispWork(p, 'sleep_island_commit', 67);
    p.end();
    world.parity ^= 1;
    world.steps++;
    world.vehicles.dirty = false;
  };

  // Snapshot of the uploaded scene (bodies, colliders, vehicle records) for reset.
  world.snapshot = () => ({
    bodyCount: world.bodyCount, colliderCount: world.colliderCount,
    bodies: world.bodyWords.slice(0, world.bodyCount * 28),
    jointCount: world.jointCount, userJointCount: world.userJointCount,
    joints: world.jointWords.slice(0, world.jointCount * JOINT_WORDS),
    colliders: world.colliderWords.slice(0, world.colliderCount * 16),
    vehicle: world.vehicleShadow.slice(),
    host: world.vehicles.save(),
  });
  world.restore = (snap) => {
    world.bodyCount = snap.bodyCount;
    world.colliderCount = snap.colliderCount;
    world.bodyWords.set(snap.bodies);
    world.colliderWords.set(snap.colliders);
    device.queue.writeBuffer(bufs[0], 0, snap.bodies);
    device.queue.writeBuffer(bufs[0], cap.maxBodies * 112, snap.bodies);
    device.queue.writeBuffer(bufs[2], 0, snap.colliders);
    // Joints come back unbroken with fresh multipliers.
    world.jointCount = snap.jointCount;
    world.userJointCount = snap.userJointCount;
    world.jointWords.set(snap.joints);
    if (snap.joints.length) device.queue.writeBuffer(bufs[5], 0, snap.joints);
    world.jointHashClearPending = true;
    world.vehicleShadow.set(snap.vehicle);
    device.queue.writeBuffer(bufs[29], 0, snap.vehicle);
    world.vehicles.restore(snap.host);
    world.steps = 0;
    world.parity = 0;
    world.last = { contact: 0, candidate: 0, awake: 0, color: 0, manifold: 0, overflow: 0 };
  };

  return world;
}
