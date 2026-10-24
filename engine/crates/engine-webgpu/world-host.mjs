// Minimal WebGPU host for the shared Slang→WGSL step: it replays the commands of
// engine_exec::rigid_plan (wasm RigidStep), the graph WgpuSim runs, on the canonical
// contact path, including the vehicle passes. Used by vehicle.html; demo.html keeps
// its own benchmark host.
import { loadShaderModules, pipelineBuilder } from './shader-loader.mjs';
import { VehicleHost, DEFAULT_VEHICLE_CAPACITY, vehicleLayout, capWords, fbits, L, RigidStep, ballJointWords, M, characterRecord } from './vehicle.mjs';
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
export const EVENT_RING_BASE = 96, EVENT_RING_CAPACITY = 1024, EVENT_BREAK = 3, EVENT_BEAM_BREAK = 5, EVENT_MOUNT_BREAK = 6;

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

/** Character / query / projectile pool sizes (binding 33). */
export const DEFAULT_MOTION_CAPACITY = { maxCharacters: 64, maxQueries: 256, maxProjectiles: 256 };
export function motionLayout(c) {
  const characterBase = M.CH_SLOTS;
  const queryBase = characterBase + c.maxCharacters * M.CHAR_STRIDE;
  const projectileBase = queryBase + c.maxQueries * M.QUERY_STRIDE;
  const slots = projectileBase + c.maxProjectiles * M.PROJ_STRIDE;
  return { characterBase, queryBase, projectileBase, slots, bytes: slots * 16 };
}

export async function createWorld(device, { cap: userCap, vehicleCap = DEFAULT_VEHICLE_CAPACITY, motionCap = DEFAULT_MOTION_CAPACITY, softCap = null, wgslBase = './generated/wgsl/', status = () => {} }) {
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
  const mcap = { ...DEFAULT_MOTION_CAPACITY, ...motionCap };
  const mlayout = motionLayout(mcap);
  sizes[33] = Math.max(sizes[33] || 0, mlayout.bytes);
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
    // Continuous collision, kinematics, impulses, characters, projectiles.
    motionCap: mcap, mlayout,
    motion: { ccd: false, kinematic: false },
    impulses: [],
    characters: [], // {body, collider}
    projectileCount: 0,
    queryCount: 0,
    motionShadow: new Uint32Array(mlayout.slots * 4),
    motionEventsSeen: 0,
    beamEventsSeen: 0,
    mountEventsSeen: 0,
  };
  if (world.soft) world.softShadow = new Uint32Array(world.soft.layout.totalSlots * 4);

  const noteFlags = (words) => {
    const flags = words[11];
    if (flags & M.BODY_CCD) world.motion.ccd = true;
    if ((flags & 3) === M.BODY_KINEMATIC) world.motion.kinematic = true;
  };
  const writeBodyRaw = (index, words) => {
    noteFlags(words);
    world.bodyWords.set(words, index * 28);
    device.queue.writeBuffer(bufs[0], index * 112, words);
    device.queue.writeBuffer(bufs[0], (cap.maxBodies + index) * 112, words);
  };
  const writeColliderRaw = (index, words) => {
    world.colliderVersion = (world.colliderVersion || 0) + 1; // renderers' drawable lists
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
    // Heightfield terrain: its pairs take the convex stages (RigidStep counts[18] bit 3).
    if (words[1] === 5) world.hasHeightField = true;
    const f = new Float32Array(words.buffer, words.byteOffset, 16);
    const body = words[0];
    const inv = new Float32Array(world.bodyWords.buffer, body * 112, 28);
    world.colliderInfo[i] = { body, radius: f[15] > 0 ? f[15] : Math.hypot(f[12], f[13], f[14]), dynamic: inv[3] > 0, plane: words[1] === 4 };
    return i;
  };
  // Hull arena records (binding 18, uint4 each), e.g. from height_field_arena
  // (vehicle wasm): written from record `first`, so edits upload only what changed.
  world.setHullRecords = (u32, first = 0) => {
    const bytes = (first * 4 + u32.length) * 4;
    if (bytes > bufs[18].size) throw new Error(`hull arena needs ${bytes} B (> ${bufs[18].size}): raise cap.maxHull`);
    device.queue.writeBuffer(bufs[18], first * 16, u32);
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
  // Vehicle part / wheel mounts that broke since the last call (EVENT_MOUNT_BREAK
  // records): [{vehicle, index, wheel, force, torque}].
  world.pollMountBreaks = async () => {
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
    for (let n = Math.max(world.mountEventsSeen, cursor - EVENT_RING_CAPACITY); n < cursor; n++) {
      const r = 4 + 4 * (n % EVENT_RING_CAPACITY);
      if (w[r] === EVENT_MOUNT_BREAK) {
        const id = w[r + 1];
        out.push({ vehicle: id >>> 8, index: id & 0x7f, wheel: (id & 0x80) !== 0, force: fl[r + 2], torque: fl[r + 3] });
      }
    }
    world.mountEventsSeen = Math.max(world.mountEventsSeen, cursor);
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
  // Rigid mode of every soft body (1 = riding its core rigidly, 0 = soft):
  // SoftSettings.rigid_mode; the record's first word, BODY_STRIDE = 6 slots.
  world.readSoftModes = async () => {
    const n = world.soft.entries.length;
    const w = await readWords(bufs[32], world.soft.layout.bodyBase * 16, n * 6 * 16);
    return Array.from({ length: n }, (_, s) => w[(s * 6 + 3) * 4]);
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

  // ---- characters, queries, projectiles (binding 33), impulses (binding 21) ----
  const writeMotion = (slot, words) => {
    world.motionShadow.set(words, slot * 4);
    device.queue.writeBuffer(bufs[33], slot * 16, words);
  };
  world.writeMotionHeader = () => {
    const h = new Uint32Array(16);
    h.set([mcap.maxCharacters, world.characters.length, mcap.maxQueries, world.queryCount], 0);
    h.set([mlayout.characterBase, mlayout.queryBase, mlayout.projectileBase, mcap.maxProjectiles], 4);
    h.set([world.projectileCount, 0, 0, 0], 8);
    writeMotion(0, h);
  };
  world.writeMotionHeader();
  /** Capsule character (engine_schema::character::CharacterDesc, snake_case);
   * its inner kinematic capsule is a body of its own. Returns its index. */
  world.addCharacter = (desc, p) => {
    const d = { ...desc };
    const r = d.radius ?? 0.3, hh = d.half_height ?? 0.6;
    if (world.characters.length >= mcap.maxCharacters) throw new Error('character capacity exhausted');
    const body = world.addBody({ p, flags: M.BODY_KINEMATIC | 8 });
    const collider = world.addCollider({ body, shape: 1, material: d.material ?? 0, params: [r, hh, 0, r + hh] });
    const index = world.characters.length;
    world.characters.push({ body, collider });
    writeMotion(mlayout.characterBase + index * M.CHAR_STRIDE, characterRecord(JSON.stringify(d), body, p[0], p[1], p[2]));
    world.writeMotionHeader();
    return index;
  };
  /** Desired horizontal velocity (world m/s) and the jump sequence (bump to jump). */
  world.setCharacterInput = (index, v, jumpSeq = 0) => {
    const w = new Uint32Array(4);
    w.set([fbits(v[0]), fbits(v[1] ?? 0), fbits(v[2]), jumpSeq >>> 0]);
    writeMotion(mlayout.characterBase + index * M.CHAR_STRIDE + M.C_IN, w);
  };
  /** Bullet-class projectile (segment casts before each step). */
  world.spawnProjectile = ({ p, v, radius = 0, mass = 0.01, cost = 1e5, drag = 0, maxHits = 8, excludeBody = M.NO_INDEX, mask = 0, impulses = true }) => {
    if (world.projectileCount >= mcap.maxProjectiles) world.projectileCount = 0; // ring: reuse the oldest slot
    const i = world.projectileCount++;
    const e = 0.5 * mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const w = new Uint32Array(M.PROJ_STRIDE * 4);
    w.set([fbits(p[0]), fbits(p[1]), fbits(p[2]), fbits(radius)], 0);
    w.set([fbits(v[0]), fbits(v[1]), fbits(v[2]), fbits(mass)], 4);
    w.set([1 | (impulses ? 2 : 0), mask, excludeBody, 0], 8);
    w.set([fbits(e), fbits(cost), fbits(drag), fbits(maxHits)], 12);
    w.set([0, 0, 0, M.NO_INDEX], 16);
    writeMotion(mlayout.projectileBase + i * M.PROJ_STRIDE, w);
    world.writeMotionHeader();
    return i;
  };
  /** Impulse J (N·s) at world point p, applied on the GPU at the next step's start. */
  world.applyImpulse = (body, j, p) => world.impulses.push([body, M.IMPULSE_AT_POINT, ...j, ...p]);
  /** A kinematic body's velocity from the next step on (GPU record, no body upload). */
  world.setKinematicVelocity = (body, lin, ang = [0, 0, 0]) => world.impulses.push([body, M.IMPULSE_SET_VELOCITY, ...lin, ...ang]);
  const uploadImpulses = () => {
    const n = world.impulses.length;
    if (!n) return false;
    if (n > M.IMPULSE_CAP) throw new Error(`more than ${M.IMPULSE_CAP} impulse records in one step`);
    const w = new Uint32Array(n * M.IMPULSE_WORDS);
    const f = new Float32Array(w.buffer);
    world.impulses.forEach((r, k) => {
      w[k * 8] = r[0]; w[k * 8 + 1] = (r[1] & 255) | (k << 8);
      f.set(r.slice(2), k * 8 + 2);
    });
    device.queue.writeBuffer(bufs[21], M.IMPULSE_BASE * 4, w);
    device.queue.writeBuffer(bufs[21], M.IMPULSE_COUNT * 4, Uint32Array.of(n));
    device.queue.writeBuffer(bufs[21], M.IMPULSE_FIRST * 4, Uint32Array.of(n));
    world.impulses = [];
    return true;
  };
  // Async read of the characters, projectiles and the event ring (one map).
  const motionReadBytes = mlayout.slots * 16;
  const ringWords = 4 + 4 * M.EVENT_RING_CAPACITY;
  const motionRead = device.createBuffer({ size: motionReadBytes + ringWords * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let motionBusy = false;
  world.readMotion = async () => {
    if (motionBusy) return null;
    motionBusy = true;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bufs[33], 0, motionRead, 0, motionReadBytes);
    enc.copyBufferToBuffer(bufs[21], M.EVENT_RING_BASE * 4, motionRead, motionReadBytes, ringWords * 4);
    device.queue.submit([enc.finish()]);
    await motionRead.mapAsync(GPUMapMode.READ);
    const all = new Uint32Array(motionRead.getMappedRange().slice(0));
    motionRead.unmap();
    motionBusy = false;
    const f = new Float32Array(all.buffer);
    const v3 = (q) => [f[q], f[q + 1], f[q + 2]];
    const characters = world.characters.map((c, i) => {
      const r = (mlayout.characterBase + i * M.CHAR_STRIDE) * 4;
      const gb = all[r + M.C_GND * 4 + 3];
      return {
        position: v3(r + M.C_POS * 4), velocity: v3(r + M.C_VEL * 4), ground: ['ground', 'steep', 'air'][all[r + M.C_VEL * 4 + 3]] ?? 'air',
        groundNormal: v3(r + M.C_GND * 4), groundBody: gb === M.NO_INDEX ? null : gb, flags: all[r + M.C_OUT * 4 + 3],
        planes: all[r + M.C_OUT * 4 + 2], push: v3(r + M.C_PUSH * 4),
      };
    });
    const projectiles = [];
    for (let i = 0; i < world.projectileCount; i++) {
      const r = (mlayout.projectileBase + i * M.PROJ_STRIDE) * 4;
      projectiles.push({ position: v3(r), velocity: v3(r + 4), alive: (all[r + 8] & 1) !== 0, hits: all[r + 11], energy: f[r + 12] });
    }
    const ring = all.subarray(motionReadBytes / 4);
    const cursor = ring[0];
    const events = [];
    const rf = new Float32Array(ring.buffer, ring.byteOffset, ring.length);
    for (let n = Math.max(world.motionEventsSeen, cursor - M.EVENT_RING_CAPACITY); n < cursor; n++) {
      const r = 4 + 4 * (n % M.EVENT_RING_CAPACITY);
      const t = ring[r];
      if (t >= M.EVENT_CHAR_LANDED && t <= M.EVENT_PROJECTILE_STOPPED)
        events.push({ type: { 5: 'landed', 6: 'supportLost', 7: 'crushed', 8: 'projectileHit', 9: 'projectileStopped' }[t], id: ring[r + 1], a: ring[r + 2], b: rf[r + 3] });
    }
    world.motionEventsSeen = Math.max(world.motionEventsSeen, cursor);
    return { characters, projectiles, events };
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

  // Dispatches of the last encoded step by kernel name (profiling).
  world.dispatchHistogram = () => {
    const h = {};
    const c = world.lastCmds || [];
    for (let i = 0; i < c.length; i += 6) if (c[i] === 1 || c[i] === 2) { const k = kernels[c[i + 1]]; h[k] = (h[k] || 0) + 1; }
    return h;
  };
  world.encodeStep = (enc) => {
    world.refreshCell();
    PFIELDS.forEach((k, i) => (i === 2 || i === 3) ? pv.setUint32(i * 4, params[k], true) : pv.setFloat32(i * 4, params[k], true));
    plan.set_params(new Uint8Array(paramBytes));
    const impulses = uploadImpulses();
    const motionBits = (world.motion.ccd ? 1 : 0) | (world.motion.kinematic ? 2 : 0) | (impulses ? 4 : 0)
      | (world.hasHeightField ? 8 : 0);
    const soft = world.soft && world.soft.bodyCount > 0 ? world.soft.counts() : new Uint32Array(10);
    // [bodies, colliders, joints, vehicles, wheels, soft x10, filters, soft dual, sweeps,
    //  motion bits, projectiles, characters, queries] (RigidStep::inputs).
    const counts = Uint32Array.of(world.bodyCount, world.colliderCount, world.jointCount,
      world.vehicles.vehicleCount, world.vehicles.wheelCount, ...soft, world.hasFilters ? 1 : 0,
      world.soft && world.soft.inner.dual() ? 1 : 0, world.soft ? world.soft.inner.sweeps() : 1,
      motionBits, world.projectileCount, world.characters.length, world.queryCount);
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
    world.lastCounts = counts;
    world.lastCmds = cmds;
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
  // Profiling (device with 'timestamp-query'): encode and run one step with every
  // dispatch in its own timed pass; resolves to GPU ms per kernel name. The step
  // itself is a normal step (same commands, same order).
  world.profileStep = async () => {
    world.encodeStep(device.createCommandEncoder()); // plan this step's commands
    const cmds = world.lastCmds;
    const n = cmds.length / 6;
    const qs = device.createQuerySet({ type: 'timestamp', count: 2 * n });
    const res = device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    // The host words and uniform the planning encodeStep staged in `upload`.
    const host = plan.host_upload(world.lastCounts);
    enc.copyBufferToBuffer(upload, 0, bufs[21], 0, 256);
    enc.copyBufferToBuffer(upload, 256, bufs[6], 0, host.length - 256 - 80);
    for (let i = 0; i < cmds.length; i += 6) {
      if (cmds[i] !== 0) continue;
      const src = cmds[i + 1] === 0xFFFFFFFF ? upload : bufs[cmds[i + 1]];
      const srcOff = cmds[i + 1] === 0xFFFFFFFF ? 512 + cmds[i + 2] : cmds[i + 2];
      enc.copyBufferToBuffer(src, srcOff, bufs[cmds[i + 3]], cmds[i + 4], cmds[i + 5]);
    }
    const timed = [];
    for (let i = 0; i < cmds.length; i += 6) {
      const op = cmds[i];
      if (op !== 1 && op !== 2) continue;
      const k = timed.length;
      timed.push(kernels[cmds[i + 1]]);
      const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
      bindPipe(p, pipes[cmds[i + 1]], cmds[i + 3]);
      if (op === 1) p.dispatchWorkgroups(...groups(cmds[i + 2])); else p.dispatchWorkgroupsIndirect(bufs[23], cmds[i + 2] * 12);
      p.end();
    }
    enc.resolveQuerySet(qs, 0, 2 * timed.length, res, 0);
    enc.copyBufferToBuffer(res, 0, rb, 0, 16 * timed.length);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rb.getMappedRange().slice(0));
    rb.unmap();
    const out = {};
    timed.forEach((k, j) => { out[k] = (out[k] || 0) + Number(t[2 * j + 1] - t[2 * j]) / 1e6; });
    qs.destroy(); res.destroy(); rb.destroy();
    return out;
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
    motionPool: world.motionShadow.slice(),
    characters: world.characters.slice(),
    projectileCount: world.projectileCount,
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
    world.colliderVersion = (world.colliderVersion || 0) + 1;
    // Joints come back unbroken with their initial multipliers.
    world.jointCount = snap.jointCount;
    world.jointWords.set(snap.joints);
    if (snap.joints.length) device.queue.writeBuffer(bufs[5], 0, snap.joints);
    world.vehicleShadow.set(snap.vehicle);
    device.queue.writeBuffer(bufs[31], 0, snap.vehicle);
    world.vehicles.restore(snap.host);
    world.motionShadow.set(snap.motionPool);
    device.queue.writeBuffer(bufs[33], 0, snap.motionPool);
    world.characters = snap.characters.slice();
    world.projectileCount = snap.projectileCount;
    world.impulses = [];
    world.writeMotionHeader();
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
