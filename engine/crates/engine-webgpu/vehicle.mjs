// Vehicle controller host for the WebGPU demo: layout constants, a line-for-line
// port of engine-vehicle's packer (crates/engine-vehicle/src/pack.rs) and host
// bookkeeping (host.rs). Descriptors come from generated/vehicle_presets.json
// (serde-expanded Rust presets); tests/vehicle_layout.test.mjs checks this file
// against slang/physics/vehicle_layout.slang and the Rust golden pack output.

// ---- Layout (float4 slots), mirrors slang/physics/vehicle_layout.slang ----
export const L = {
  VH_COUNTS: 0, VH_BASES: 1, VH_POOLS: 2, VH_MISC: 3, VH_SLOTS: 4,
  VEH_STRIDE: 40,
  V_CFG: 0, V_FWD: 1, V_UP: 2, V_ENG0: 3, V_ENG1: 4, V_TR0: 5, V_TR1: 6, V_GEARS0: 7, V_GEARS1: 8,
  V_AERO0: 9, V_AERO1: 10, V_STEER: 11, V_RESP: 12, V_POOL: 13, V_POOL2: 14, V_MASS: 15,
  V_IN0: 16, V_IN1: 17, V_IN2: 18, V_IN3: 19, V_D0: 20, V_D1: 21, V_D2: 22, V_D3: 23,
  V_REFP: 24, V_REFQ: 25, V_D4: 26, V_O0: 27, V_O1: 28, V_O2: 29, V_O3: 30, V_ASSIST: 31,
  V_BIKE0: 32, V_BIKE1: 33, V_BIKE_D: 34, V_TRACK: 35, V_TRACK_D: 36, V_QUERY: 37,
  VF_ENABLED: 1, VF_DIRECT_DRIVE: 2, VF_MANUAL_TRANS: 4, VF_CAN_SLEEP: 8, VF_REVERSE_BRAKE: 16, VF_SPHERE_CAST: 32, VF_TRACKED: 64, VF_TRACTION_CONTROL: 128,
  VS_ENGINE_LOST: 1, VS_ACTIVE: 2, VS_SLIPPING: 4,
  GEAR_AUTO: 0x7FFFFFFF,
  WHEEL_STRIDE: 48,
  W_CFG: 0, W_HP: 1, W_DIR: 2, W_STEERAX: 3, W_FWD: 4, W_SUSP: 5, W_SPR: 6, W_INER: 7, W_TIRE: 8,
  W_MISC: 9, W_PARK: 10, W_UP: 11, W_C0: 12, W_C1: 13, W_C2: 14, W_C3: 15, W_C4: 16, W_C5: 17,
  W_K: 18, W_LAM: 19, W_K2: 20, W_Y: 21, W_W: 22, W_F: 23, W_BS: 24, W_HIT: 25, W_GV: 26, W_MU: 27,
  W_O_POS: 28, W_O_ROT: 29, W_O_FM: 30, W_O_TM: 31, W_O_CP: 32, W_O_ST: 33, W_O_N: 34,
  W_LHIT: 35, W_LN: 36, W_FP: 37, W_TJ: 38,
  WF_STEER: 1, WF_DRIVEN: 2, WF_HANDBRAKE: 4, WF_ENABLED: 8, WF_RIGID_BUMP: 16, WF_BOX_FRICTION: 32,
  WD_CONTACT: 1, WD_STICK_L: 2, WD_STICK_T: 4, WD_DETACHED: 8, WD_DISABLED: 16, WD_SUSP: 32,
  WD_SLIDE: 64, WD_BUMP: 128, WD_PENDING_ADD: 256,
  PART_STRIDE: 4, P_CFG: 0, P_MASS: 1, P_COM: 2, P_PARK: 3,
  PR_BODY: 0, PR_ENGINE: 1, PR_DRIVESHAFT: 2, PR_STEERING: 3, PR_BRAKE: 4,
  PS_ATTACHED: 0, PS_DETACHED: 1, PS_DISABLED: 2, PS_LOST: 3,
  DIFF_STRIDE: 2, ANTIROLL_STRIDE: 1, RAY_STRIDE: 4,
  NO_INDEX: 0x7F7FFFFF, MAX_VEHICLE_WHEELS: 16, MAX_DRIVEN_WHEELS: 8, SHELL_SPHERES: 5,
};
export const BODY_VEHICLE = 16;
export const JOINT_TYRE = 7;
export const WHEEL_RIM_MATERIAL = 7;
const RIM_SPHERES_MAX = 32;

// ---- f32 helpers ----
const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
export const fbits = (v) => { f32b[0] = v; return u32b[0]; };
export const bitsf = (u) => { u32b[0] = u >>> 0; return f32b[0]; };
const fr = Math.fround;

// ---- small vector math (f32-rounded) ----
const v3 = (a) => [fr(a[0]), fr(a[1]), fr(a[2])];
const add = (a, b) => [fr(a[0] + b[0]), fr(a[1] + b[1]), fr(a[2] + b[2])];
const sub = (a, b) => [fr(a[0] - b[0]), fr(a[1] - b[1]), fr(a[2] - b[2])];
const scale = (a, s) => [fr(a[0] * s), fr(a[1] * s), fr(a[2] * s)];
const mulv = (a, b) => [fr(a[0] * b[0]), fr(a[1] * b[1]), fr(a[2] * b[2])];
const dot = (a, b) => fr(fr(fr(a[0] * b[0]) + fr(a[1] * b[1])) + fr(a[2] * b[2]));
const cross = (a, b) => [fr(a[1] * b[2] - a[2] * b[1]), fr(a[2] * b[0] - a[0] * b[2]), fr(a[0] * b[1] - a[1] * b[0])];
const length = (a) => fr(Math.sqrt(dot(a, a)));
const normalize = (a) => scale(a, fr(1 / length(a)));
const splat = (s) => [s, s, s];

function shapeVolume(shape) {
  if (shape.type === 'box') return fr(8 * shape.half[0] * shape.half[1] * shape.half[2]);
  if (shape.type === 'sphere') return fr((4 / 3) * Math.PI * shape.radius ** 3);
  return fr(Math.PI * shape.radius * shape.radius * (2 * shape.half_length + (4 / 3) * shape.radius));
}

function shapeInertia(shape, m) {
  if (shape.type === 'box') {
    const [x, y, z] = shape.half.map((h) => 2 * h);
    return [fr(m * (y * y + z * z) / 12), fr(m * (x * x + z * z) / 12), fr(m * (x * x + y * y) / 12)];
  }
  if (shape.type === 'sphere') { const i = fr(0.4 * m * shape.radius * shape.radius); return [i, i, i]; }
  const h = 2 * shape.half_length;
  const ia = fr(0.5 * m * shape.radius * shape.radius);
  const it = fr(m * (3 * shape.radius * shape.radius + h * h) / 12);
  return [it, ia, it];
}

const parallel = (m, d) => [fr(m * (d[1] * d[1] + d[2] * d[2])), fr(m * (d[0] * d[0] + d[2] * d[2])), fr(m * (d[0] * d[0] + d[1] * d[1]))];

export function centerOfMass(desc) {
  const off = desc.center_of_mass_offset || [0, 0, 0];
  if (!desc.auto_center_of_mass) return off;
  const up = normalize(desc.up);
  let m = 0;
  let c = [0, 0, 0];
  for (const p of desc.parts) {
    const vol = Math.max(p.colliders.reduce((s, k) => s + shapeVolume(k.shape), 0), 1e-9);
    for (const k of p.colliders) c = add(c, scale(k.position, p.mass * shapeVolume(k.shape) / vol));
    m += p.mass;
  }
  for (const w of desc.wheels) {
    const dir = w.suspension_dir || scale(up, -1);
    c = add(c, scale(add(w.position, scale(dir, w.suspension_max)), w.mass));
    m += w.mass;
  }
  const com = m > 0 ? scale(c, 1 / m) : [0, 0, 0];
  return add(com, off);
}

export function centered(desc) {
  const o = centerOfMass(desc);
  const d = structuredClone(desc);
  for (const p of d.parts) for (const k of p.colliders) k.position = sub(k.position, o);
  for (const w of d.wheels) {
    w.position = sub(w.position, o);
    if (w.suspension_force_point) w.suspension_force_point = sub(w.suspension_force_point, o);
  }
  d.aero.center = sub(d.aero.center, o);
  d.auto_center_of_mass = false;
  d.center_of_mass_offset = [0, 0, 0];
  return d;
}

const rimCount = (w) => Math.min(Math.max(w.rim_spheres, 1), RIM_SPHERES_MAX);
const curveLen = (c) => (c ? c.length : 0);

export function requirements(desc) {
  const wheelSlots = desc.wheels.length + desc.spare_wheels;
  const spareRims = (desc.wheels.length ? rimCount(desc.wheels[0]) : 12) * desc.spare_wheels;
  let curvePoints = 0;
  for (const w of desc.wheels) curvePoints += curveLen(w.long_friction) + curveLen(w.lat_friction);
  curvePoints += curveLen(desc.steering.speed_curve);
  if (desc.drive.type === 'drivetrain') curvePoints += curveLen(desc.drive.engine.torque_curve);
  const diffs = desc.drive.type === 'drivetrain' ? desc.drive.differentials.length : 0;
  return {
    wheelSlots,
    curvePoints,
    diffs,
    antiRolls: desc.anti_roll.length,
    parts: desc.parts.length,
    bodies: 2 + wheelSlots + desc.parts.length,
    colliders: desc.parts.reduce((s, p) => s + p.colliders.length, 0)
      + desc.wheels.reduce((s, w) => s + rimCount(w), 0) + spareRims + wheelSlots * L.SHELL_SPHERES,
  };
}

export function parkedPosition(index) {
  return [fr(-5000 + 9 * (index % 1024)), -5000, fr(-5000 + 9 * Math.floor(index / 1024))];
}

// Body / collider records as plain objects; writeBodyWords/writeColliderWords encode them.
function dynamicBody(position, rotation, mass, inertia) {
  return {
    position: v3(position), invMass: mass > 0 ? fr(1 / mass) : 0, orientation: rotation.map(fr),
    invInertia: inertia.map((i) => (i > 0 ? fr(1 / i) : 0)), flags: 1 | 8,
    linearVel: [0, 0, 0], sleepTimer: 0, angularVel: [0, 0, 0], generation: 0,
    initialPos: [0, 0, 0], initialPadBits: 0, initialRot: [0, 0, 0, 1],
  };
}

function wheelBodyInertia(w, axle) {
  const r = w.radius;
  const ia = fr(0.5 * w.mass * r * r);
  const im = fr(w.mass * (3 * r * r + w.width * w.width) / 12);
  return add(splat(im), scale(mulv(axle, axle), fr(ia - im)));
}

function parkedBody(index, mass, inertia) {
  const b = dynamicBody(parkedPosition(index), [0, 0, 0, 1], mass, inertia);
  b.flags = 0;
  b.initialPos = b.position.slice();
  return b;
}

export function shellGeometry(radius, width) {
  const rs = fr(Math.max(Math.min(fr(0.5 * width), fr(0.4 * radius)), 1e-3));
  return [rs, fr(Math.max(fr(radius - rs), 0))];
}

/** Wheel shell spheres around `center` in the plane of `down` and `fwd` (60°…300° from down). */
export function shellColliders(w, center, down, fwd, body) {
  const [rs, ring] = shellGeometry(w.radius, w.width);
  const out = [];
  for (let k = 1; k <= L.SHELL_SPHERES; k++) {
    const th = fr((Math.PI * 2 * k) / (L.SHELL_SPHERES + 1));
    const p = add(center, scale(add(scale(down, fr(Math.cos(th))), scale(fwd, fr(Math.sin(th)))), ring));
    out.push({ body, shape: 0, material: w.rim_material, generation: 0, localPos: p, density: 0, localRot: [0, 0, 0, 1], params: [rs, 0, 0, rs] });
  }
  return out;
}

export function rimColliders(w, up, fwd, body, n) {
  const rr = fr(Math.max(Math.min(0.5 * w.width, 0.5 * w.radius), 1e-3));
  const ring = fr(Math.max(w.radius - rr, 0));
  const out = [];
  for (let k = 0; k < n; k++) {
    const th = fr((Math.PI * 2 * k) / n);
    const p = scale(add(scale(up, fr(Math.cos(th))), scale(fwd, fr(Math.sin(th)))), ring);
    out.push({ body, shape: 0, material: w.rim_material, generation: 0, localPos: p, density: 0, localRot: [0, 0, 0, 1], params: [rr, 0, 0, rr] });
  }
  return out;
}

const slot3 = (v, w) => [fbits(v[0]), fbits(v[1]), fbits(v[2]), fbits(w)];
const slot3u = (v, w) => [fbits(v[0]), fbits(v[1]), fbits(v[2]), w >>> 0];

function pushCurve(sink, c) {
  if (!c || c.length === 0) return L.NO_INDEX;
  const start = sink.base + sink.points.length;
  if (start + c.length > 0xFFFF) throw new Error('curve pool exceeds 16-bit point index');
  for (const p of c) sink.points.push([fr(p[0]), fr(p[1])]);
  return (start | (c.length << 16)) >>> 0;
}

export function packWheel(desc, w, vehicle, localIndex, enabled, parkedBodyIdx, parkedColStart, curves) {
  const up = normalize(desc.up);
  const fwd = normalize(desc.forward);
  const axle = normalize(cross(up, fwd));
  const dir = normalize(w.suspension_dir || scale(up, -1));
  const steerAxis = normalize(w.steer_axis || up);
  const wfwd = normalize(w.forward || fwd);
  const wup = normalize(w.up || up);
  const pos = v3(w.position);
  const lat = dot(pos, axle);
  const side = lat > 1e-3 ? 1 : lat < -1e-3 ? -1 : 0;
  let flags = 0;
  if (w.steerable) flags |= L.WF_STEER;
  if (w.driven) flags |= L.WF_DRIVEN;
  if (w.handbrake) flags |= L.WF_HANDBRAKE;
  if (enabled) flags |= L.WF_ENABLED;
  if (w.friction_model === 'box') flags |= L.WF_BOX_FRICTION;
  const sp = w.spring;
  const [mode, k, c] = sp.type === 'frequency_damping' ? [0, sp.hz, sp.zeta] : [1, sp.k, sp.c];
  let bump = [0, 0, 0];
  if (w.bump_stop.type === 'spring') bump = [w.bump_stop.hz, w.bump_stop.zeta, w.bump_stop.max_force];
  else if (w.bump_stop.type === 'rigid') flags |= L.WF_RIGID_BUMP;
  const inertia = w.inertia ?? fr(0.5 * w.mass * w.radius * w.radius);
  const s = Array.from({ length: L.WHEEL_STRIDE }, () => [0, 0, 0, 0]);
  s[L.W_CFG] = [vehicle, flags, localIndex, L.NO_INDEX];
  s[L.W_HP] = slot3(pos, w.radius);
  s[L.W_DIR] = slot3(dir, w.width);
  s[L.W_STEERAX] = slot3(steerAxis, w.max_steer);
  s[L.W_FWD] = slot3(wfwd, side);
  s[L.W_SUSP] = [fbits(w.suspension_min), fbits(w.suspension_max), fbits(w.preload), mode];
  s[L.W_SPR] = [fbits(k), fbits(c), fbits(bump[0]), fbits(bump[1])];
  s[L.W_INER] = [fbits(inertia), fbits(w.angular_damping), fbits(w.max_brake_torque), fbits(w.max_handbrake_torque)];
  s[L.W_TIRE] = [curves[0], curves[1], fbits(w.long_mu_scale), fbits(w.lat_mu_scale)];
  s[L.W_MISC] = [fbits(w.mass), fbits(w.rolling_resistance), fbits(bump[2]), fbits(w.roll_influence)];
  s[L.W_PARK] = [parkedBodyIdx, parkedColStart, rimCount(w), 0];
  s[L.W_UP] = slot3(wup, w.drive_multiplier);
  s[L.W_C0] = [L.NO_INDEX, L.NO_INDEX, enabled ? 0 : L.WD_DISABLED, 0];
  s[L.W_C1] = slot3([0, 0, 0], w.suspension_max);
  s[L.W_C2] = slot3(up, 1);
  s[L.W_O_ROT] = [fbits(0), fbits(0), fbits(0), fbits(1)];
  s[L.W_O_POS] = slot3(add(pos, scale(dir, w.suspension_max)), w.suspension_max);
  s[L.W_FP] = w.suspension_force_point ? slot3(v3(w.suspension_force_point), 1) : slot3([0, 0, 0], 0);
  return s;
}

/// Port of engine_vehicle::pack::pack.
export function pack(desc0, position, rotation, slots) {
  const desc = centered(desc0);
  if (!desc.wheels.length) throw new Error('vehicle needs at least one wheel');
  if (desc.wheels.length + desc.spare_wheels > L.MAX_VEHICLE_WHEELS) throw new Error('too many wheel slots');
  const up = normalize(desc.up);
  const fwd = normalize(desc.forward);
  const axle = normalize(cross(up, fwd));
  const out = { bodies: [], colliders: [], vehicle: null, wheels: [], curvePoints: [], diffs: [], antiRolls: [], parts: [], joints: [], mass: 0, inertia: null };
  const curves = { base: slots.curvePointBase, points: out.curvePoints };

  let mass = 0;
  let inertia = [0, 0, 0];
  const partProps = [];
  for (const p of desc.parts) {
    const vol = Math.max(p.colliders.reduce((s, k) => fr(s + shapeVolume(k.shape)), 0), 1e-9);
    let com = [0, 0, 0];
    for (const k of p.colliders) com = add(com, scale(k.position, fr(shapeVolume(k.shape) / vol)));
    let ip = [0, 0, 0];
    for (const k of p.colliders) {
      const m = fr(p.mass * shapeVolume(k.shape) / vol);
      const d = sub(k.position, com);
      ip = add(ip, add(shapeInertia(k.shape, m), parallel(m, d)));
    }
    inertia = add(inertia, add(ip, parallel(p.mass, com)));
    mass = fr(mass + p.mass);
    partProps.push([com, ip]);
  }
  for (const w of desc.wheels) {
    const d = add(w.position, scale(w.suspension_dir || scale(up, -1), w.suspension_max));
    inertia = add(inertia, add(wheelBodyInertia(w, axle), parallel(w.mass, d)));
    mass = fr(mass + w.mass);
  }
  if (desc.inertia) inertia = v3(desc.inertia);
  out.mass = mass;
  out.inertia = inertia;

  const chassis = slots.bodyBase;
  const anchor = chassis + 1;
  const wheel0 = chassis + 2;
  const wheelSlots = desc.wheels.length + desc.spare_wheels;
  const part0 = wheel0 + wheelSlots;
  const cb = dynamicBody(position, rotation, mass, inertia);
  cb.flags |= BODY_VEHICLE;
  cb.initialPos = cb.position.slice();
  cb.initialRot = cb.orientation.slice();
  cb.initialPadBits = slots.aabbBlock >>> 0;
  out.bodies.push(cb);
  const ab = dynamicBody(parkedPosition(anchor), [0, 0, 0, 1], 0, [0, 0, 0]);
  ab.flags = 0; ab.invMass = 0; ab.invInertia = [0, 0, 0];
  ab.initialPos = ab.position.slice();
  out.bodies.push(ab);
  const spare = desc.wheels[0];
  for (let i = 0; i < wheelSlots; i++) {
    const w = desc.wheels[i] || spare;
    out.bodies.push(parkedBody(wheel0 + i, w.mass, wheelBodyInertia(w, axle)));
  }
  desc.parts.forEach((p, i) => out.bodies.push(parkedBody(part0 + i, p.mass, partProps[i][1])));

  const partCols = [];
  for (const p of desc.parts) {
    const start = slots.colliderBase + out.colliders.length;
    for (const k of p.colliders) {
      let shape, params;
      if (k.shape.type === 'box') { shape = 2; params = [...k.shape.half, length(k.shape.half)]; }
      else if (k.shape.type === 'sphere') { shape = 0; params = [k.shape.radius, 0, 0, k.shape.radius]; }
      else { shape = 1; params = [k.shape.radius, k.shape.half_length, 0, fr(k.shape.radius + k.shape.half_length)]; }
      out.colliders.push({ body: chassis, shape, material: k.material, generation: 0, localPos: v3(k.position), density: 0, localRot: k.rotation.map(fr), params: params.map(fr) });
    }
    partCols.push([start, p.colliders.length]);
  }
  const rimStarts = [];
  for (let i = 0; i < wheelSlots; i++) {
    const w = desc.wheels[i] || spare;
    rimStarts.push(slots.colliderBase + out.colliders.length);
    out.colliders.push(...rimColliders(w, up, fwd, wheel0 + i, rimCount(w)));
  }
  // Wheel shells on the chassis (spare slots park theirs on the static anchor).
  const shellStarts = [];
  for (let i = 0; i < wheelSlots; i++) {
    const real = i < desc.wheels.length;
    const w = desc.wheels[i] || spare;
    const down = normalize(w.suspension_dir || scale(up, -1));
    const center = add(v3(w.position), scale(down, w.suspension_max));
    shellStarts.push(slots.colliderBase + out.colliders.length);
    out.colliders.push(...shellColliders(w, center, down, fwd, real ? chassis : anchor));
  }

  const wheelCurves = desc.wheels.map((w) => [pushCurve(curves, w.long_friction), pushCurve(curves, w.lat_friction)]);
  for (let i = 0; i < wheelSlots; i++) {
    const real = i < desc.wheels.length;
    const w = desc.wheels[i] || spare;
    out.wheels.push(packWheel(desc, w, slots.vehicle, i, real, wheel0 + i, rimStarts[i], real ? wheelCurves[i] : wheelCurves[0]));
  }
  // Tyre joints: graph edges to dynamic ground, parked on the anchor.
  out.wheels.forEach((rec, i) => {
    rec[L.W_TJ] = [slots.jointBase + i, shellStarts[i], L.NO_INDEX, 0];
    out.joints.push({ bodyA: anchor, bodyB: anchor, jointType: JOINT_TYRE });
  });

  // Tracks: members share the driven wheel's spin, inertia and brake.
  if (desc.tracks) {
    for (const tr of [desc.tracks.left, desc.tracks.right]) {
      for (const m of tr.wheels) out.wheels[m][L.W_CFG][3] = tr.driven_wheel;
      const rec = out.wheels[tr.driven_wheel];
      rec[L.W_CFG][1] |= L.WF_DRIVEN;
      rec[L.W_INER] = [fbits(tr.inertia), fbits(tr.angular_damping), fbits(tr.max_brake_torque), fbits(0)];
      rec[L.W_UP][3] = fbits(tr.differential_ratio);
    }
  }

  const steered = desc.wheels.filter((w) => w.steerable);
  const fixed = desc.wheels.filter((w) => !w.steerable);
  const meanFwd = (ws) => (ws.length ? fr(ws.reduce((s, w) => fr(s + dot(w.position, fwd)), 0) / ws.length) : 0);
  const wheelbase = desc.steering.wheelbase ?? Math.abs(fr(meanFwd(steered) - meanFwd(fixed)));
  let track = desc.steering.track;
  if (track == null) {
    const lats = steered.map((w) => dot(w.position, axle));
    track = lats.length < 2 ? 0 : fr(Math.max(...lats) - Math.min(...lats));
  }

  let flags = L.VF_ENABLED;
  if (desc.can_sleep) flags |= L.VF_CAN_SLEEP;
  if (desc.reverse_brake) flags |= L.VF_REVERSE_BRAKE;
  if (desc.wheel_query === 'sphere') flags |= L.VF_SPHERE_CAST;
  if (desc.assists && desc.assists.traction_control) flags |= L.VF_TRACTION_CONTROL;
  if (desc.tracks) flags |= L.VF_TRACKED;
  const v = Array.from({ length: L.VEH_STRIDE }, () => [0, 0, 0, 0]);
  let directTorque = 0, topSpeed = 0;
  if (desc.drive.type === 'direct') {
    flags |= L.VF_DIRECT_DRIVE;
    directTorque = desc.drive.max_torque;
    topSpeed = desc.drive.top_speed;
  } else {
    const d = desc.drive;
    if (d.transmission.mode === 'manual') flags |= L.VF_MANUAL_TRANS;
    const e = d.engine;
    const t = d.transmission;
    const torque = pushCurve(curves, e.torque_curve);
    v[L.V_ENG0] = [fbits(e.max_torque), fbits(e.min_rpm), fbits(e.max_rpm), fbits(e.inertia)];
    v[L.V_ENG1] = [fbits(e.angular_damping), torque, fbits(t.clutch_strength), fbits(d.inter_diff_limited_slip ?? 1e30)];
    v[L.V_TR0] = [fbits(t.switch_time), fbits(t.clutch_release_time), fbits(t.switch_latency), fbits(t.shift_up_rpm)];
    if (t.gear_ratios.length + t.reverse_ratios.length > 8) throw new Error('at most 8 gear ratios');
    v[L.V_TR1] = [fbits(t.shift_down_rpm), t.gear_ratios.length, t.reverse_ratios.length, 0];
    const ratios = [...t.gear_ratios, ...t.reverse_ratios];
    while (ratios.length < 8) ratios.push(0);
    v[L.V_GEARS0] = ratios.slice(0, 4).map(fbits);
    v[L.V_GEARS1] = ratios.slice(4, 8).map(fbits);
    for (const dd of d.differentials) {
      const lr = (x) => (x == null ? L.NO_INDEX : x);
      out.diffs.push([
        [lr(dd.left), lr(dd.right), fbits(dd.ratio), fbits(dd.left_right_split)],
        [fbits(dd.limited_slip_ratio ?? 1e30), fbits(dd.engine_torque_ratio), 0, 0],
      ]);
    }
  }
  const steerCurve = pushCurve(curves, desc.steering.speed_curve);
  v[L.V_CFG] = [chassis, slots.wheelBase, wheelSlots, flags];
  const pitchRollCos = desc.max_pitch_roll_deg >= 179.9 ? -2 : Math.cos((desc.max_pitch_roll_deg * Math.PI) / 180);
  v[L.V_FWD] = slot3(fwd, pitchRollCos);
  v[L.V_UP] = slot3(up, Math.cos((desc.max_slope_deg * Math.PI) / 180));
  v[L.V_AERO0] = [fbits(desc.aero.drag), fbits(desc.aero.downforce), fbits(desc.gravity_scale), steerCurve];
  v[L.V_AERO1] = slot3(desc.aero.center, directTorque);
  v[L.V_STEER] = [fbits(desc.steering.ackermann), fbits(wheelbase), fbits(track), fbits(desc.steering.rate)];
  v[L.V_RESP] = [fbits(desc.response.throttle_rate), fbits(desc.response.brake_rate), fbits(desc.response.handbrake_rate), fbits(topSpeed)];
  v[L.V_POOL] = [slots.diffBase, out.diffs.length, slots.antiRollBase, desc.anti_roll.length];
  v[L.V_POOL2] = [slots.partBase, desc.parts.length, anchor, 0];
  v[L.V_MASS] = [fbits(mass), fbits(inertia[0]), fbits(inertia[1]), fbits(inertia[2])];
  const a = desc.assists || {};
  v[L.V_ASSIST] = [a.air_control_pitch || 0, a.air_control_yaw || 0, a.upright_stiffness || 0, a.upright_damping || 0].map(fbits);
  const m = desc.motorcycle;
  if (m) {
    v[L.V_BIKE0] = [fbits((m.max_lean_deg * Math.PI) / 180), fbits(m.lean_spring), fbits(m.lean_damping), fbits(m.lean_integration)];
    v[L.V_BIKE1] = [fbits(m.lean_integration_decay), fbits(m.lean_smoothing), fbits(m.lean_steering_limit ? 1 : 0), 0];
  }
  v[L.V_BIKE_D] = slot3([0, 1, 0], 0);
  const t = desc.tracks;
  v[L.V_TRACK] = t
    ? [t.left.driven_wheel, t.right.driven_wheel, fbits(t.steer_strength), fbits(0)]
    : [L.NO_INDEX, L.NO_INDEX, fbits(0), fbits(0)];
  // V_TRACK_D.w: lateral-accel budget × track spacing (a·W) for the ratio cap.
  let turnAW = 0;
  if (t) {
    const lateral = (tr) => fr(tr.wheels.reduce((s2, i) => fr(s2 + dot(v3(desc.wheels[i].position), axle)), 0) / Math.max(tr.wheels.length, 1));
    turnAW = fr(fr(t.turn_grip_g * fr(9.81)) * Math.abs(fr(lateral(t.left) - lateral(t.right))));
  }
  v[L.V_TRACK_D] = [fbits(1), fbits(1), fbits(0), fbits(turnAW)];
  v[L.V_QUERY] = [desc.query_interval >>> 0, 0, 0, 0];
  v[L.V_IN0] = [0, 0, 0, 0].map(fbits);
  v[L.V_IN1] = [L.GEAR_AUTO, 0, 0, 0];
  v[L.V_IN2] = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF];
  v[L.V_IN3] = [0, 0, 1, 0];
  const minRpm = desc.drive.type === 'drivetrain' ? desc.drive.engine.min_rpm : 0;
  v[L.V_D0] = [fbits(minRpm / 9.5492966), fbits(0), fbits(1), fbits(0)];
  v[L.V_REFQ] = [fbits(0), fbits(0), fbits(0), fbits(1)];
  v[L.V_O3] = [1, 0, fbits(0), fbits(0)];
  out.vehicle = v;

  for (const a of desc.anti_roll) out.antiRolls.push([a.left, a.right, fbits(a.stiffness), 0]);
  desc.parts.forEach((p, i) => {
    const role = p.role.type;
    const [r, mask] = role === 'engine' ? [L.PR_ENGINE, 0]
      : role === 'drive_shaft' ? [L.PR_DRIVESHAFT, p.role.wheels]
      : role === 'steering' ? [L.PR_STEERING, p.role.wheels]
      : role === 'brake' ? [L.PR_BRAKE, p.role.wheels] : [L.PR_BODY, 0];
    const [com, ip] = partProps[i];
    out.parts.push([
      [partCols[i][0], partCols[i][1], r, mask >>> 0],
      [fbits(p.mass), fbits(ip[0]), fbits(ip[1]), fbits(ip[2])],
      slot3u(com, L.PS_ATTACHED),
      [part0 + i, slots.vehicle, 0, 0],
    ]);
  });
  return out;
}

// ---- record encoding (BodyState 112 B, Collider 64 B) ----
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

// Joint record (128 B): bodyA, bodyB, type, flags, then zeros (tyre joints only).
export function jointWords(j) {
  const u = new Uint32Array(32);
  u[0] = j.bodyA; u[1] = j.bodyB; u[2] = j.jointType; u[3] = j.flags || 0;
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

// ---- pool layout ----
export function vehicleLayout(cap) {
  const vehicleBase = L.VH_SLOTS;
  const wheelBase = vehicleBase + cap.maxVehicles * L.VEH_STRIDE;
  const curveBase = wheelBase + cap.maxWheels * L.WHEEL_STRIDE;
  const diffBase = curveBase + Math.ceil(cap.maxCurvePoints / 2);
  const antiRollBase = diffBase + cap.maxDiffs * L.DIFF_STRIDE;
  const partBase = antiRollBase + cap.maxAntiRolls * L.ANTIROLL_STRIDE;
  const rayBase = partBase + cap.maxParts * L.PART_STRIDE;
  const totalSlots = rayBase + cap.maxRays * L.RAY_STRIDE;
  return { vehicleBase, wheelBase, curveBase, diffBase, antiRollBase, partBase, rayBase, totalSlots };
}

export const DEFAULT_VEHICLE_CAPACITY = {
  maxVehicles: 64, maxWheels: 512, maxCurvePoints: 4096, maxDiffs: 256, maxAntiRolls: 256, maxParts: 512, maxRays: 1024,
};

/**
 * Host-side vehicle bookkeeping (port of engine_vehicle::VehicleHost). `backend`
 * provides bodyCount()/colliderCount(), appendBodies(words[]), appendColliders(words[]),
 * writeBody(index, words), writeCollider(index, words), writeSlots(slot, Uint32Array)
 * and writeBytes(byteOffset, Uint32Array) into gVehicleData.
 */
export class VehicleHost {
  constructor(cap, aabbTail) {
    this.cap = { ...DEFAULT_VEHICLE_CAPACITY, ...cap };
    this.layout = vehicleLayout(this.cap);
    this.aabbTail = aabbTail;
    this.entries = [];
    this.wheelsUsed = 0; this.curvePointsUsed = 0; this.diffsUsed = 0; this.antiRollsUsed = 0; this.partsUsed = 0;
    this.rays = 0;
    // Wheel probes scan every collider instead of walking the broadphase grid (oracle).
    this.fullScanQueries = false;
    this.dirty = true;
  }

  get vehicleCount() { return this.entries.length; }
  get wheelCount() { return this.wheelsUsed; }

  header() {
    const l = this.layout;
    return [
      [this.cap.maxVehicles, this.cap.maxWheels, this.entries.length, this.wheelsUsed],
      [l.vehicleBase, l.wheelBase, l.curveBase, this.aabbTail],
      [l.diffBase, l.antiRollBase, l.partBase, l.rayBase],
      [this.rays, this.cap.maxRays, this.partsUsed, this.fullScanQueries ? 1 : 0],
    ];
  }

  static flat(slots) {
    const u = new Uint32Array(slots.length * 4);
    slots.forEach((s, i) => u.set(s.map((x) => x >>> 0), i * 4));
    return u;
  }

  addVehicle(backend, desc, position, rotation = [0, 0, 0, 1]) {
    const req = requirements(desc);
    const c = this.cap;
    const v = this.entries.length;
    const need = (ok, msg) => { if (!ok) throw new Error(msg); };
    need(v < c.maxVehicles, 'vehicle capacity exhausted');
    need(this.wheelsUsed + req.wheelSlots <= c.maxWheels, 'wheel capacity exhausted');
    need(this.curvePointsUsed + req.curvePoints <= c.maxCurvePoints, 'curve capacity exhausted');
    need(this.diffsUsed + req.diffs <= c.maxDiffs, 'differential capacity exhausted');
    need(this.antiRollsUsed + req.antiRolls <= c.maxAntiRolls, 'anti-roll capacity exhausted');
    need(this.partsUsed + req.parts <= c.maxParts, 'part capacity exhausted');
    const slots = {
      vehicle: v, wheelBase: this.wheelsUsed, curvePointBase: this.curvePointsUsed, diffBase: this.diffsUsed,
      antiRollBase: this.antiRollsUsed, partBase: this.partsUsed, bodyBase: backend.bodyCount(),
      colliderBase: backend.colliderCount(), aabbBlock: this.aabbTail + 4 * v,
      jointBase: backend.jointCount(),
    };
    const p = pack(desc, position, rotation, slots);
    const l = this.layout;
    backend.appendBodies(p.bodies.map(bodyWords));
    backend.appendColliders(p.colliders.map(colliderWords));
    backend.appendJoints(p.joints.map(jointWords));
    backend.writeSlots(l.vehicleBase + v * L.VEH_STRIDE, VehicleHost.flat(p.vehicle));
    p.wheels.forEach((w, i) => backend.writeSlots(l.wheelBase + (slots.wheelBase + i) * L.WHEEL_STRIDE, VehicleHost.flat(w)));
    if (p.curvePoints.length) {
      const u = new Uint32Array(p.curvePoints.length * 2);
      p.curvePoints.forEach((pt, i) => { u[2 * i] = fbits(pt[0]); u[2 * i + 1] = fbits(pt[1]); });
      backend.writeBytes(l.curveBase * 16 + slots.curvePointBase * 8, u);
    }
    p.diffs.forEach((d, i) => backend.writeSlots(l.diffBase + (slots.diffBase + i) * L.DIFF_STRIDE, VehicleHost.flat(d)));
    p.antiRolls.forEach((a, i) => backend.writeSlots(l.antiRollBase + (slots.antiRollBase + i), VehicleHost.flat([a])));
    p.parts.forEach((pr, i) => backend.writeSlots(l.partBase + (slots.partBase + i) * L.PART_STRIDE, VehicleHost.flat(pr)));
    this.wheelsUsed += req.wheelSlots;
    this.curvePointsUsed += p.curvePoints.length;
    this.diffsUsed += p.diffs.length;
    this.antiRollsUsed += p.antiRolls.length;
    this.partsUsed += p.parts.length;
    this.entries.push({
      slots, desc: structuredClone(desc), wheelSlots: req.wheelSlots, wheelsUsed: desc.wheels.length,
      controls: p.vehicle.slice(L.V_IN0, L.V_IN0 + 4).map((s) => s.slice()),
      chassis: slots.bodyBase, firstPartBody: slots.bodyBase + 2 + req.wheelSlots,
      origin: centerOfMass(desc), mass: p.mass,
    });
    backend.writeSlots(0, VehicleHost.flat(this.header()));
    this.dirty = true;
    return v;
  }

  writeControls(backend, h) {
    const e = this.entries[h];
    backend.writeSlots(this.layout.vehicleBase + h * L.VEH_STRIDE + L.V_IN0, VehicleHost.flat(e.controls));
    this.dirty = true;
  }

  setInput(backend, h, { forward = 0, steer = 0, brake = 0, handbrake = 0, gear = null } = {}) {
    const c = this.entries[h].controls;
    const n0 = [forward, steer, brake, handbrake].map(fbits);
    const g = gear == null ? L.GEAR_AUTO : (gear | 0) >>> 0;
    if (c[0].every((x, i) => x === n0[i]) && c[1][0] === g) return;
    c[0] = n0; c[1][0] = g;
    this.writeControls(backend, h);
  }

  setFunctionalState(backend, h, { wheelMask = 0xFFFFFFFF, driveMask = 0xFFFFFFFF, steerMask = 0xFFFFFFFF, brakeMask = 0xFFFFFFFF, engineConnected = true } = {}) {
    const c = this.entries[h].controls;
    c[2] = [wheelMask >>> 0, driveMask >>> 0, steerMask >>> 0, brakeMask >>> 0];
    c[3][2] = ((c[3][2] & ~1) | (engineConnected ? 1 : 0)) >>> 0;
    this.writeControls(backend, h);
  }

  detachWheel(backend, h, w) { this.entries[h].controls[1][2] = (this.entries[h].controls[1][2] | (1 << w)) >>> 0; this.writeControls(backend, h); }
  disableWheel(backend, h, w) { this.entries[h].controls[1][3] = (this.entries[h].controls[1][3] | (1 << w)) >>> 0; this.writeControls(backend, h); }
  detachPart(backend, h, p) { this.entries[h].controls[3][0] = (this.entries[h].controls[3][0] | (1 << p)) >>> 0; this.writeControls(backend, h); }
  disablePart(backend, h, p) { this.entries[h].controls[3][1] = (this.entries[h].controls[3][1] | (1 << p)) >>> 0; this.writeControls(backend, h); }
  wake(backend, h) { const c = this.entries[h].controls; c[3][3] = (c[3][3] + 1) >>> 0; this.writeControls(backend, h); }

  addWheel(backend, h, wheel, readSlots) {
    const e = this.entries[h];
    if (e.wheelsUsed >= e.wheelSlots) throw new Error('no spare wheel slot (VehicleDesc.spare_wheels)');
    const idx = e.wheelsUsed;
    const l = this.layout;
    const rec = l.wheelBase + (e.slots.wheelBase + idx) * L.WHEEL_STRIDE;
    const old = readSlots(rec, L.WHEEL_STRIDE);
    const w = { ...wheel, position: sub(wheel.position, e.origin) };
    const s = packWheel(e.desc, w, h, idx, true, old[L.W_PARK][0], old[L.W_PARK][1], [old[L.W_TIRE][0], old[L.W_TIRE][1]]);
    s[L.W_C0][2] = L.WD_PENDING_ADD;
    const rims = old[L.W_PARK][2];
    s[L.W_PARK][2] = rims;
    const up = normalize(e.desc.up);
    const fwd = normalize(e.desc.forward);
    const axle = normalize(cross(up, fwd));
    backend.writeBody(old[L.W_PARK][0], bodyWords(parkedBody(old[L.W_PARK][0], wheel.mass, wheelBodyInertia(wheel, axle))));
    rimColliders(wheel, up, fwd, old[L.W_PARK][0], rims).forEach((c, k) => backend.writeCollider(old[L.W_PARK][1] + k, colliderWords(c)));
    // Keep the slot's tyre joint and shell; resize the shell (vehicle_post attaches it).
    s[L.W_TJ] = [old[L.W_TJ][0], old[L.W_TJ][1], L.NO_INDEX, 0];
    if (old[L.W_TJ][1] !== L.NO_INDEX) {
      const down = normalize(w.suspension_dir || scale(up, -1));
      const center = add(v3(w.position), scale(down, w.suspension_max));
      shellColliders(w, center, down, fwd, e.chassis + 1).forEach((c, k) => backend.writeCollider(old[L.W_TJ][1] + k, colliderWords(c)));
    }
    backend.writeSlots(rec, VehicleHost.flat(s));
    e.wheelsUsed++;
    e.desc.wheels.push(structuredClone(wheel));
    this.dirty = true;
    return idx;
  }

  chassisBody(h) { return this.entries[h].chassis; }
  wheelBody(h, w) { return this.entries[h].chassis + 2 + w; }
  partBody(h, p) { return this.entries[h].firstPartBody + p; }

  /** Decode outputs from a Uint32Array view of gVehicleData (whole buffer or prefix). */
  readState(words, h) {
    const e = this.entries[h];
    const l = this.layout;
    const f = new Float32Array(words.buffer, words.byteOffset, words.length);
    const vq = (l.vehicleBase + h * L.VEH_STRIDE) * 4;
    const V = (slot, k) => f[vq + slot * 4 + k];
    const VU = (slot, k) => words[vq + slot * 4 + k];
    const st = {
      forwardSpeed: V(L.V_O0, 0), engineRpm: V(L.V_O0, 1), gear: Math.round(V(L.V_O0, 2)),
      lateralSpeed: V(L.V_O1, 0), yawRate: V(L.V_O1, 1), clutch: V(L.V_O1, 2), contacts: VU(L.V_O1, 3),
      wheelMask: VU(L.V_O2, 0), driveMask: VU(L.V_O2, 1), steerMask: VU(L.V_O2, 2), eventSeq: VU(L.V_O2, 3),
      engineConnected: VU(L.V_O3, 0) !== 0, partsLost: VU(L.V_O3, 1), mass: V(L.V_MASS, 0), wheels: [],
    };
    for (let i = 0; i < e.wheelsUsed; i++) {
      const wq = (l.wheelBase + (e.slots.wheelBase + i) * L.WHEEL_STRIDE) * 4;
      const W = (slot, k) => f[wq + slot * 4 + k];
      const WU = (slot, k) => words[wq + slot * 4 + k];
      const flags = WU(L.W_O_ST, 3);
      const dyn = WU(L.W_C0, 2);
      st.wheels.push({
        position: [W(L.W_O_POS, 0), W(L.W_O_POS, 1), W(L.W_O_POS, 2)], suspensionLength: W(L.W_O_POS, 3),
        rotation: [W(L.W_O_ROT, 0), W(L.W_O_ROT, 1), W(L.W_O_ROT, 2), W(L.W_O_ROT, 3)],
        suspensionForce: W(L.W_O_FM, 3), longitudinalForce: W(L.W_O_TM, 3), lateralForce: W(L.W_O_CP, 3),
        mountForce: [W(L.W_O_FM, 0), W(L.W_O_FM, 1), W(L.W_O_FM, 2)],
        contactPoint: [W(L.W_O_CP, 0), W(L.W_O_CP, 1), W(L.W_O_CP, 2)],
        omega: W(L.W_O_ST, 0), slipRatio: W(L.W_O_ST, 1), slipAngleDeg: W(L.W_O_ST, 2),
        inContact: (flags & L.WD_CONTACT) !== 0, sliding: (flags & L.WD_SLIDE) !== 0,
        detached: (dyn & L.WD_DETACHED) !== 0, disabled: (dyn & L.WD_DISABLED) !== 0,
        radius: W(L.W_HP, 3), width: W(L.W_DIR, 3),
      });
    }
    return st;
  }
}
