// Suspension, steering and wheel poses of the dune buggy's render mesh: a
// dependency-free port of vibe-land's visual rig (engine/scenes/dune-buggy/src/
// vehicle-rig.mjs + visual-rig.mjs, Three.js) for the WebGPU demo.
//
// Input per corner (fl, fr, rl, rr): travel (m, + = compression), steering
// (rad, + = left) and wheel spin (rad). Output: each moving part's matrix in the
// recipe frame (metres, +Y up, -Z front), column-major 4×4 as Three.js
// Matrix4.elements, identical to VisualRig.matrixFor (tests/buggy_rig.test.mjs
// checks it against the original). The renderer maps the recipe frame into
// Rubble's (R = 180° about Y) and onto the chassis.

export const CORNER_IDS = ['fl', 'fr', 'rl', 'rr'];

// ---- tiny Three.js-compatible math (arrays) ----
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const quatAxisAngle = (axis, angle) => {
  const s = Math.sin(angle / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
};
const quatMul = (a, b) => [
  a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
  a[1] * b[3] + a[3] * b[1] + a[2] * b[0] - a[0] * b[2],
  a[2] * b[3] + a[3] * b[2] + a[0] * b[1] - a[1] * b[0],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const quatNorm = (q) => { const l = Math.hypot(...q) || 1; return q.map((x) => x / l); };
/** Three.js Quaternion.setFromUnitVectors. */
function quatFromUnitVectors(f, t) {
  let r = dot(f, t) + 1;
  let q;
  if (r < Number.EPSILON) {
    r = 0;
    q = Math.abs(f[0]) > Math.abs(f[2]) ? [-f[1], f[0], 0, r] : [0, -f[2], f[1], r];
  } else {
    q = [f[1] * t[2] - f[2] * t[1], f[2] * t[0] - f[0] * t[2], f[0] * t[1] - f[1] * t[0], r];
  }
  return quatNorm(q);
}
const rotate = (q, v) => {
  const [x, y, z, w] = q;
  const ix = w * v[0] + y * v[2] - z * v[1], iy = w * v[1] + z * v[0] - x * v[2], iz = w * v[2] + x * v[1] - y * v[0], iw = -x * v[0] - y * v[1] - z * v[2];
  return [ix * w + iw * -x + iy * -z - iz * -y, iy * w + iw * -y + iz * -x - ix * -z, iz * w + iw * -z + ix * -y - iy * -x];
};
const axisAngle = (v, axis, angle) => rotate(quatAxisAngle(axis, angle), v);

/** Matrix4.compose(position, quaternion, scale). */
function compose(p, q, s = [1, 1, 1]) {
  const [x, y, z, w] = q, x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2, wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    p[0], p[1], p[2], 1,
  ];
}
export function mul(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
const translation = (p) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, p[0], p[1], p[2], 1];
const rotationZ = (t) => { const c = Math.cos(t), s = Math.sin(t); return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; };
const applyPoint = (m, p) => [
  m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
  m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
  m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
];
/** Rotation `q` about `point` (visual-rig.mjs pivot). */
const pivot = (point, q) => mul(mul(translation(point), compose([0, 0, 0], q)), translation(scale(point, -1)));
/** A 3×4 column-major rest matrix (buggy-mesh.json) as a 4×4. */
export const rest4 = (m) => [m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, m[9], m[10], m[11], 1];

const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];

// ---- the linkage (vehicle-rig.mjs, unchanged) ----
function linkage(h, angle) {
  const ax = Math.abs(h.lowerPivot[0]), ay = h.lowerPivot[1], bx = ax, by = h.upperPivot[1];
  const ll = Math.hypot(Math.abs(h.lower[0]) - ax, h.lower[1] - ay);
  const ul = Math.hypot(Math.abs(h.upper[0]) - bx, h.upper[1] - by);
  const kl = Math.hypot(h.upper[0] - h.lower[0], h.upper[1] - h.lower[1]);
  const x = ax + ll * Math.cos(angle), y = ay + ll * Math.sin(angle);
  const dx = bx - x, dy = by - y, d = Math.hypot(dx, dy);
  if (d >= ul + kl - 1e-5 || d <= Math.abs(ul - kl) + 1e-5) return null;
  const a = (kl * kl - ul * ul + d * d) / (2 * d), v = Math.sqrt(Math.max(0, kl * kl - a * a));
  const ux = x + a * dx / d + v * dy / d, uy = y + a * dy / d - v * dx / d;
  const camber = Math.atan2(uy - y, ux - x) - Math.atan2(h.upper[1] - h.lower[1], Math.abs(h.upper[0]) - Math.abs(h.lower[0]));
  const hx = x + .15 * Math.cos(camber) - .1 * Math.sin(camber), hy = y + .15 * Math.sin(camber) + .1 * Math.cos(camber);
  return { lower: [h.side * x, y, h.z], upper: [h.side * ux, uy, h.z], hub: [h.side * hx, hy, h.z], camber: camber * h.side, angle };
}
function steeredHub(k, steering) {
  if (!steering) return k.hub;
  const l = k.lower, u = k.upper, r = k.hub.map((x, i) => x - l[i]), a = u.map((x, i) => x - l[i]), n = Math.hypot(...a);
  for (let i = 0; i < 3; i++) a[i] /= n;
  const c = Math.cos(steering), sn = Math.sin(steering), d = r[0] * a[0] + r[1] * a[1] + r[2] * a[2];
  const cr = [a[1] * r[2] - a[2] * r[1], a[2] * r[0] - a[0] * r[2], a[0] * r[1] - a[1] * r[0]];
  return r.map((x, i) => l[i] + x * c + cr[i] * sn + a[i] * d * (1 - c));
}
export function solveCorner(h, travel = 0, steering = 0) {
  steering = h.front ? steering : 0;
  const target = h.radius + clamp(travel, h.minTravel, h.maxTravel);
  let a = h.minAngle, b = h.maxAngle;
  for (let i = 0; i < 22; i++) { const mid = (a + b) / 2; if (steeredHub(linkage(h, mid), steering)[1] < target) a = mid; else b = mid; }
  const k = linkage(h, Math.abs(travel) < 1e-10 && Math.abs(steering) < 1e-10 ? h.neutralAngle : (a + b) / 2);
  k.steeredHub = steeredHub(k, steering);
  k.travelM = k.steeredHub[1] - h.radius;
  return k;
}

/** The rig: `definition` is buggy-mesh.json's `rig`. */
export class BuggyRig {
  constructor(definition) {
    this.def = definition;
    this.states = {};
    this.steeringDelta = translation([0, 0, 0]);
    this.applyPose(Object.fromEntries(CORNER_IDS.map((id) => [id, { travelM: 0, steeringRad: 0, rotationRad: 0 }])));
  }

  /** `wheels[id] = {travelM, steeringRad, rotationRad}`; `steeringWheelRad` optional. */
  applyPose(wheels, steeringWheelRad) {
    for (const id of CORNER_IDS) {
      const h = this.def.corners[id], input = wheels[id];
      const k = solveCorner(h, input.travelM, input.steeringRad);
      const lower = k.lower, upper = k.upper;
      const camberQ = quatAxisAngle(Z, k.camber);
      const camber = mul(compose(lower, camberQ), translation(scale(h.lower, -1)));
      const axis = norm(sub(upper, lower));
      const steerQ = quatAxisAngle(axis, h.front ? input.steeringRad : 0);
      const steer = mul(pivot(k.lower, steerQ), camber);
      const hub = applyPoint(steer, h.hub);
      const wheelAxis = rotate(steerQ, rotate(camberQ, X));
      const wheel = mul(pivot(hub, quatAxisAngle(wheelAxis, input.rotationRad)), steer);
      const armAngle = (k.angle - h.neutralAngle) * h.side;
      const bottom = add(axisAngle(sub(h.shockBottom, h.lowerPivot), Z, armAngle), h.lowerPivot);
      const top = h.shockTop, shockDir = norm(sub(bottom, top)), length = len(sub(bottom, top));
      const springMatrix = mul(compose(top, quatFromUnitVectors(Y, shockDir)), [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]);
      const coilLength = length - .32 * h.shockLength;
      const tieOuter = applyPoint(steer, h.tieOuter), tieInner = [...h.tieInner];
      const rodLength = len(sub(h.tieOuter, tieInner));
      tieInner[0] = tieOuter[0] - h.side * Math.sqrt(Math.max(.0001, rodLength ** 2 - (tieOuter[1] - tieInner[1]) ** 2 - (tieOuter[2] - tieInner[2]) ** 2));
      const axleDir = norm(sub(hub, h.axleInner));
      this.states[id] = { h, k, lower, upper, hub, steer, wheel, top, bottom, shockDir, coilLength, springMatrix, tieOuter, tieInner, axleDir, rotation: input.rotationRad };
    }
    const { columnStart, columnEnd } = this.def.steering;
    const angle = steeringWheelRad ?? ((wheels.fl.steeringRad + wheels.fr.steeringRad) * .5 * this.def.steering.ratio);
    this.steeringDelta = pivot(columnEnd, quatAxisAngle(norm(sub(columnEnd, columnStart)), angle));
  }

  beam(a, b, radius) {
    const d = sub(b, a);
    return compose(scale(add(a, b), .5), quatFromUnitVectors(Z, norm(d)), [radius, radius, len(d)]);
  }

  /** The posed matrix of a moving instance ({role, corner, endpoint, radius,
   * matrix}: buggy-mesh.json), recipe frame. The spring's is its frame at the
   * current coil length for a helix modelled at its neutral length (the
   * renderer scales the template along local Z by coil / neutral coil). */
  matrixFor(inst) {
    const rest = rest4(inst.matrix);
    if (inst.role === 'steering') return mul(this.steeringDelta, rest);
    const s = this.states[CORNER_IDS[inst.corner]], h = s.h;
    switch (inst.role) {
      case 'hub': case 'wheel': return mul(s.wheel, rest);
      case 'knuckle': return mul(s.steer, rest);
      case 'lowerArm': return this.beam(inst.endpoint, s.lower, inst.radius);
      case 'upperArm': return this.beam(inst.endpoint, s.upper, inst.radius);
      case 'upright': return this.beam(s.lower, s.upper, inst.radius);
      case 'damper': return this.beam(s.top, add(s.top, scale(s.shockDir, h.shockLength * .67)), inst.radius);
      case 'piston': return this.beam(add(s.top, scale(s.shockDir, h.shockLength * .6)), s.bottom, inst.radius);
      case 'spring': return s.springMatrix;
      case 'topSeat': case 'bottomSeat': {
        const t = inst.role === 'topSeat' ? .02 : s.coilLength;
        return this.beam(add(s.top, scale(s.shockDir, t - .012)), add(s.top, scale(s.shockDir, t + .012)), inst.radius);
      }
      case 'shockEye': return this.beam(add(s.bottom, scale(s.shockDir, -.06)), s.lower, inst.radius);
      case 'tieRod': return this.beam(s.tieInner, s.tieOuter, inst.radius);
      case 'axle': return mul(this.beam(h.axleInner, s.hub, inst.radius), rotationZ(s.rotation * h.side));
      case 'cvBoot': {
        const offset = Math.abs(rest[12] - h.hub[0]);
        const at = add(s.hub, scale(s.axleDir, -offset));
        const q = quatMul(quatFromUnitVectors(X, scale(s.axleDir, h.side)), quatAxisAngle(X, s.rotation));
        return mul(mul(mul(translation(at), compose([0, 0, 0], q)), translation([-rest[12], -rest[13], -rest[14]])), rest);
      }
      default: return rest;
    }
  }

  /** Neutral coil length of a corner (the spring template's modelled length). */
  neutralCoil(corner) {
    return .68 * this.def.corners[CORNER_IDS[corner]].shockLength;
  }
}
