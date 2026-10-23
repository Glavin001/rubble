import {vehicleById,vehicleLift,trailerSpec} from './vehicle-catalog.mjs';
/** DUNE visual pose contract. Metres, +Y up, -Z forward; no physics dependency.
 * Import with Three.js available as the `three` package (or an import map).
 * Positive travel is compression, positive steering turns left.
 */
import * as T from 'three';
export const cornerIds = ['fl', 'fr', 'rl', 'rr'];
export const previewDefaults = { compression: .12, extension: .10, stiffness: 45, damping: .85 };
const V = a => new T.Vector3(...a);
const clamp = T.MathUtils.clamp;
export function cornerHardpoints(p, id) {
  const front = id[0] === 'f', side = id[1] === 'l' ? -1 : 1;
  const z = p.wheelbase / 2 * (front ? -1 : 1), y = p.tireRadius, wx = side * p.track / 2, inner = front ? .39 : .47;
  return { id, front, side, z, radius: y,
    lowerPivot: [side * inner, y + .13, z], upperPivot: [side * inner, y + .37, z],
    lower: [wx - side * .15, y - .1, z], upper: [wx - side * .17, y + .12, z], hub: [wx, y, z],
    shockTop: [side * (front ? .57 : .58), (front ? 1.02 : 1.1)+vehicleLift(p), z + (front ? .32 : -.35)],
    shockBottom: [wx - side * .25, y - .08, z], axleInner: [0, y, z],
    tieInner: [side * .18, y + .07, z + .1], tieOuter: [wx - side * .15, y + .06, z],
  };
}
// Work in an outboard-positive XY plane. The four-bar pivots all have Z axes.
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
// Rodrigues rotation about the physical kingpin, including its inclination.
function steeredHub(k, steering) {
  if (!steering) return k.hub;
  const l=k.lower,u=k.upper,r=k.hub.map((x,i)=>x-l[i]),a=u.map((x,i)=>x-l[i]),len=Math.hypot(...a);for(let i=0;i<3;i++)a[i]/=len;
  const c=Math.cos(steering),sn=Math.sin(steering),dot=r[0]*a[0]+r[1]*a[1]+r[2]*a[2];
  const cross=[a[1]*r[2]-a[2]*r[1],a[2]*r[0]-a[0]*r[2],a[0]*r[1]-a[1]*r[0]];
  return r.map((x,i)=>l[i]+x*c+cross[i]*sn+a[i]*dot*(1-c));
}
function shockLength(h, angle, neutralAngle) {
  const d = (angle - neutralAngle) * h.side, x = h.shockBottom[0] - h.lowerPivot[0], y = h.shockBottom[1] - h.lowerPivot[1];
  return Math.hypot(h.lowerPivot[0] + x * Math.cos(d) - y * Math.sin(d) - h.shockTop[0], h.lowerPivot[1] + x * Math.sin(d) + y * Math.cos(d) - h.shockTop[1], h.shockBottom[2] - h.shockTop[2]);
}
export function createRigDefinition(parameters, parts = []) {
  const corners = {};
  for (const id of cornerIds) {
    const h = cornerHardpoints(parameters, id);
    h.neutralAngle = Math.atan2(h.lower[1] - h.lowerPivot[1], Math.abs(h.lower[0]) - Math.abs(h.lowerPivot[0]));
    h.shockLength = V(h.shockBottom).distanceTo(V(h.shockTop));
    h.spring = { turns: 10, radius: .059, wireRadius: .009, neutralLength: h.shockLength * .68 };
    const safe = angle => {
      const k = linkage(h, angle); if (!k) return false;
      const len = shockLength(h, angle, h.neutralAngle);
      return len > h.shockLength * .75 && len < h.shockLength * 1.22 && len - .32 * h.shockLength > .205;
    };
    h.minAngle = h.maxAngle = h.neutralAngle;
    for (let n = 1; n <= 240; n++) { const a = h.neutralAngle - n * .0025; if (!safe(a)) break; h.minAngle = a; }
    for (let n = 1; n <= 240; n++) { const a = h.neutralAngle + n * .0025; if (!safe(a)) break; h.maxAngle = a; }
    h.minTravel = Math.max(-.25, ...[-.65,0,.65].map(a=>steeredHub(linkage(h,h.minAngle),h.front?a:0)[1]-h.radius));
    h.maxTravel = Math.min(.25, ...[-.65,0,.65].map(a=>steeredHub(linkage(h,h.maxAngle),h.front?a:0)[1]-h.radius));
    corners[id] = h;
  }
  return { version: 1, units: 'metres', axes: { up: '+Y', front: '-Z' }, parameters: { ...parameters }, corners, trailer:trailerSpec(parameters),
    steering: { columnStart: [-.325, .77+vehicleLift(parameters), -parameters.wheelbase / 2 + .6], columnEnd: [-.325, 1.08+vehicleLift(parameters), -.31], ratio: 6, maxCentreRad: (vehicleById(parameters.vehicle)?.handling.steering??25) * Math.PI / 180 },
    bindings: parts.filter(p => p.motion).map(p => ({ partId: p.id, ...p.motion, neutralMatrix: p.matrix.toArray() })), handling: {...vehicleById(parameters.vehicle)?.handling}, previewDefaults: { ...previewDefaults, ...Object.fromEntries(Object.keys(previewDefaults).map(k=>[k,vehicleById(parameters.vehicle)?.handling[k]??previewDefaults[k]])) } };
}
export function solveCorner(h, travel = 0, steering = 0) {
  steering = h.front ? steering : 0;
  const target = h.radius + clamp(travel, h.minTravel, h.maxTravel);
  let a = h.minAngle, b = h.maxAngle;
  for (let i = 0; i < 22; i++) { const mid = (a + b) / 2; if (steeredHub(linkage(h, mid),steering)[1] < target) a = mid; else b = mid; }
  // Exact identity at neutral is important when entering or leaving Play.
  const k = linkage(h, Math.abs(travel) < 1e-10 && Math.abs(steering)<1e-10 ? h.neutralAngle : (a + b) / 2);
  k.steeredHub=steeredHub(k,steering);k.travelM = k.steeredHub[1] - h.radius; return k;
}
export function steeringAngles(centre, wheelbase, track) {
  if (Math.abs(centre) < 1e-8) return { fl: 0, fr: 0, rl: 0, rr: 0 };
  const radius = wheelbase / Math.tan(centre);
  return { fl: Math.atan(wheelbase / (radius - track / 2)), fr: Math.atan(wheelbase / (radius + track / 2)), rl: 0, rr: 0 };
}
export function neutralPose() {
  return { trailer:{yawRad:0,pitchRad:0,rotationRad:0}, chassis: { position: [0, 0, 0], rotation: [0, 0, 0, 1] },
    wheels: Object.fromEntries(cornerIds.map(id => [id, { travelM: 0, steeringRad: 0, rotationRad: 0, grounded: true }])) };
}
