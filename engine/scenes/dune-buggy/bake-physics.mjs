// Bake a vibe-land catalog vehicle into its Rubble preset asset
// (engine/crates/engine-vehicle/assets/dune_buggy.json for the buggy,
// vibe_<model>.json for the others; committed: CI builds the presets without
// bun or this directory's node_modules).
//
//   bun install && bun run bake:physics          # every model
//   bun run bake-physics.mjs trophy               # one
//
// Two builds of the same recipe (src/buggy.mjs), part for part:
// - the exact build (manifold-3d CSG, ~40 s): solid volumes → masses, and the
//   measured interface area of every pair of touching parts → joint strength;
// - the live build (Three.js primitives): each part's primitive and placement
//   → its collision shape.
// parts.mjs says which Rubble part, wheel or rig each recipe part belongs to.

import Module from 'manifold-3d';
import * as T from 'three';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildBuggy, defaults } from './src/buggy.mjs';
import { LiveGeometry } from './src/live-geometry.mjs';
import { createRigDefinition } from './src/vehicle-rig.mjs';
import { jointStrength, SOLVER_MIN_BOND_AREA_M2, STRENGTH_PROFILE_VERSION } from './src/strength-profile.mjs';
import { DENSITY } from './src/real-joint-capacity.mjs';
import { classify, toRubble, partsFor, CORNERS, MODELS } from './parts.mjs';
import { vehicleById, tireWidthScale, trailerSpec } from './src/vehicle-catalog.mjs';
import { buildFrameGraph, attachNodes } from './frame-graph.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const MODEL = process.argv[2] ?? 'buggy';
if (!MODELS.includes(MODEL)) throw new Error(`unknown model ${MODEL}; have ${MODELS.join(', ')}`);
const OUT = resolve(here, `../../crates/engine-vehicle/assets/${MODEL === 'buggy' ? 'dune_buggy' : `vibe_${MODEL}`}.json`);
const PARAMS = { ...defaults, ...vehicleById(MODEL).parameters, vehicle: MODEL };
// Rubble's own wheel shape: the tyre's width (live tread spans 0.30 m; the
// monster's tyres are half as wide again).
const WHEEL_WIDTH = 0.3 * tireWidthScale(PARAMS);

const round = (x, d = 5) => Math.round(x * 10 ** d) / 10 ** d;
const r3 = (a) => a.map((x) => round(x));

// ---- builds ----
const wasm = await Module();
wasm.setup();
const t0 = performance.now();
const exact = buildBuggy(wasm, PARAMS, () => {}, false, { preview: false });
console.error(`exact build: ${exact.parts.length} parts, ${exact.joints.length} joints in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
const liveGeometry = new LiveGeometry();
const live = liveGeometry.build(PARAMS);
if (live.parts.length !== exact.parts.length) throw new Error(`live ${live.parts.length} vs exact ${exact.parts.length} parts`);
live.parts.forEach((p, i) => {
  if (p.name !== exact.parts[i].name) throw new Error(`part ${i}: live ${p.name} vs exact ${exact.parts[i].name}`);
});
const keyOf = new Map([...liveGeometry.templates].map(([k, g]) => [g, k]));

// ---- frame change: R = 180° about Y ----
const Q_R = new T.Quaternion().setFromAxisAngle(new T.Vector3(0, 1, 0), Math.PI);
// Rubble capsules lie along local Y; recipe cylinders along local Z.
const Q_Y_TO_Z = new T.Quaternion().setFromAxisAngle(new T.Vector3(1, 0, 0), Math.PI / 2);

/** Collision shape of a live part, in the Rubble frame (null: no primitive). */
function collider(part) {
  const key = keyOf.get(part.geometry);
  const pos = new T.Vector3(), q = new T.Quaternion(), s = new T.Vector3();
  part.matrix.decompose(pos, q, s);
  const position = r3(toRubble(pos.toArray()));
  const rot = Q_R.clone().multiply(q);
  const quat = (qq) => r3(qq.normalize().toArray());
  if (key.startsWith('box:')) {
    const size = key.slice(4).split(',').map(Number).slice(0, 3);
    return { shape: { type: 'box', half: r3(size.map((x, i) => (x * s.getComponent(i)) / 2)) }, position, rotation: quat(rot), material: 0 };
  }
  if (key.startsWith('cylinder:') || key.startsWith('ring:')) {
    const radius = Math.max(s.x, s.y), length = s.z;
    if (length >= 2 * radius) {
      return {
        shape: { type: 'capsule', radius: round(radius), half_length: round(length / 2 - radius) },
        position, rotation: quat(rot.multiply(Q_Y_TO_Z)), material: 0,
      };
    }
    return { shape: { type: 'box', half: r3([radius, radius, length / 2]) }, position, rotation: quat(rot), material: 0 };
  }
  if (key.startsWith('sphere:')) return { shape: { type: 'sphere', radius: round(s.x) }, position, rotation: [0, 0, 0, 1], material: 0 };
  // A body panel (a door skin, a windshield, a wing): a thin convex plate from
  // its corner points. Vehicle parts take boxes, spheres and capsules, so it
  // gets the box along its principal axes that holds every vertex.
  if (key.startsWith('panel:')) return panelBox(part);
  throw new Error(`no collision primitive for ${part.name} (${key})`);
}

/** Symmetric 3×3 eigenvectors (Jacobi), as columns of a Matrix3-like array. */
function eigenAxes(c) {
  const a = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-14) continue;
      const th = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
      const cs = Math.cos(th), sn = Math.sin(th);
      for (let k = 0; k < 3; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = cs * akp - sn * akq; a[k][q] = sn * akp + cs * akq; }
      for (let k = 0; k < 3; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = cs * apk - sn * aqk; a[q][k] = sn * apk + cs * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = cs * vkp - sn * vkq; v[k][q] = sn * vkp + cs * vkq; }
    }
  }
  return [0, 1, 2].map((j) => new T.Vector3(v[0][j], v[1][j], v[2][j]).normalize());
}

function panelBox(part) {
  const pos = part.geometry.attributes.position;
  const pts = [];
  for (let i = 0; i < pos.count; i++) pts.push(new T.Vector3().fromBufferAttribute(pos, i).applyMatrix4(part.matrix));
  const mean = pts.reduce((m, p) => m.add(p), new T.Vector3()).multiplyScalar(1 / pts.length);
  const c = [0, 0, 0, 0, 0, 0];
  for (const p of pts) {
    const d = p.clone().sub(mean);
    c[0] += d.x * d.x; c[1] += d.x * d.y; c[2] += d.x * d.z; c[3] += d.y * d.y; c[4] += d.y * d.z; c[5] += d.z * d.z;
  }
  let [e0, e1, e2] = eigenAxes(c);
  if (e0.clone().cross(e1).dot(e2) < 0) e2.negate(); // a rotation, not a reflection
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {
    const d = p.clone().sub(mean);
    [e0, e1, e2].forEach((e, k) => { const t = d.dot(e); lo[k] = Math.min(lo[k], t); hi[k] = Math.max(hi[k], t); });
  }
  const centre = mean.clone();
  [e0, e1, e2].forEach((e, k) => centre.addScaledVector(e, (lo[k] + hi[k]) / 2));
  const q = new T.Quaternion().setFromRotationMatrix(new T.Matrix4().makeBasis(e0, e1, e2));
  return {
    shape: { type: 'box', half: r3([0, 1, 2].map((k) => Math.max((hi[k] - lo[k]) / 2, 0.004))) },
    position: r3(toRubble(centre.toArray())),
    rotation: r3(Q_R.clone().multiply(q).normalize().toArray()),
    material: 0,
  };
}

// ---- groups ----
const cls = exact.parts.map((p) => classify(p, MODEL));
const groupOf = (i) => (cls[i].cls === 'wheel' ? `wheel_${cls[i].corner}` : cls[i].name);
const indexById = new Map(exact.parts.map((p, i) => [p.id, i]));

const parts = partsFor(cls).map((p) => ({ name: p.name, role: p.role, mass: 0, colliders: [], members: 0 }));
const partByName = new Map(parts.map((p) => [p.name, p]));
const wheelMass = Object.fromEntries(CORNERS.map((c) => [c, 0]));
const trailerParts = [];
exact.parts.forEach((p, i) => {
  const c = cls[i];
  if (c.cls === 'wheel') {
    wheelMass[c.corner] += p.mass;
    return;
  }
  if (c.cls === 'trailer') {
    trailerParts.push(i);
    return;
  }
  const out = partByName.get(c.name);
  out.mass += p.mass;
  out.members++;
  if (c.collider) out.colliders.push(collider(live.parts[i]));
});

// ---- mounts: measured joints that cross a group boundary ----
// Capacity of one joint = interface area × the weaker material's fatal
// strength (tension or shear, whichever is lower): vibe-land's
// strength-profile.mjs, the limits its PhysX build tore parts off at.
//
// A part's mount is its joints to the frame where it has any (the engine's
// struts, not its flange to the gearbox), else to whatever carries it (a
// headlamp on the nose). A wheel's mount is its corner's suspension where it
// meets the frame: arms, tie rod and damper at their chassis mounts. The hub's
// own bearing and the axle inside it are far stronger; a wheel comes off a
// buggy when its arms tear out.
//
// The measured area of two members that merely overlap (an extinguisher in its
// bracket, a barrel in its crankcase) is far larger than any weld or bolt
// between them. As vibe-land's real-joint-capacity.mjs (section-bound): a joint
// never carries more than the smaller member beside it, so its effective area
// is min(measured area, smaller member section), the section being the
// member's material volume over its longest extent.
const section = exact.parts.map((p) => {
  const rho = DENSITY[p.material];
  const ext = Math.max(...[0, 1, 2].map((k) => p.bounds.max[k] - p.bounds.min[k]));
  return rho && p.mass > 0 && ext > 0 ? p.mass / rho / ext : Infinity;
});
//
// Contacts under SOLVER_MIN_BOND_AREA_M2 are kept. vibe-land drops them because
// its PhysX stress stage floors bond stiffness at that area; a Rubble mount is
// rated as a whole, and a small far contact (a nerf bar's rear return, a light
// bar's second foot) is what gives the part its moment capacity.
const bonds = [];
let dropped = 0, bounded = 0;
for (const j of exact.joints) {
  const a = indexById.get(j.a), b = indexById.get(j.b);
  if (!(j.interfaceAreaM2 > 0)) { dropped++; continue; }
  const st = jointStrength(exact.parts[a].material, exact.parts[b].material);
  const area = Math.min(j.interfaceAreaM2, section[a], section[b]);
  if (area < j.interfaceAreaM2) bounded++;
  bonds.push({ a, b, force: area * Math.min(st.tensionFatal, st.shearFatal), area, anchor: toRubble(j.anchor) });
}
console.error(`${bonds.length} joints (${bounded} section-bound, ${bonds.filter((b) => b.area < SOLVER_MIN_BOND_AREA_M2).length} under ${SOLVER_MIN_BOND_AREA_M2} m²)`);
const isRig = (i) => cls[i].cls === 'rig';
function mountOf(g) {
  let mine;
  const corner = g.startsWith('wheel_') ? g.slice(6) : null;
  if (corner) {
    // Suspension (rig) parts of this corner against fixed frame parts.
    mine = bonds.filter(({ a, b }) => [[a, b], [b, a]].some(([s, o]) => isRig(s) && cls[s].corner === corner && groupOf(o) === 'frame' && !isRig(o)))
      .map((x) => ({ ...x, with: 'frame' }));
  } else {
    const all = bonds.filter(({ a, b }) => (groupOf(a) === g) !== (groupOf(b) === g))
      .map((x) => ({ ...x, with: groupOf(x.a) === g ? groupOf(x.b) : groupOf(x.a) }));
    const toFrame = all.filter((x) => x.with === 'frame');
    mine = toFrame.length ? toFrame : all;
  }
  if (!mine.length) throw new Error(`${g} is not attached to anything`);
  const F = mine.reduce((s, b) => s + b.force, 0);
  const anchor = [0, 1, 2].map((k) => mine.reduce((s, b) => s + b.force * b.anchor[k], 0) / F);
  // A couple of the joints about their force centroid, each also resisting
  // with half its own width: a single weld still has a bending capacity.
  const torque = mine.reduce((s, b) => s + b.force * (Math.hypot(...b.anchor.map((x, k) => x - anchor[k])) + 0.5 * Math.sqrt(b.area)), 0);
  return { break_force: round(F, 0), break_torque: round(torque, 1), anchor: r3(anchor), bonds: mine.length, attached_to: [...new Set(mine.map((b) => b.with))] };
}

// ---- wheels: the rig's hub at neutral, its travel either side ----
const rig = createRigDefinition(PARAMS);
const wheels = CORNERS.map((id) => {
  const h = rig.corners[id];
  const m = mountOf(`wheel_${id}`);
  return {
    corner: id,
    hub: r3(toRubble(h.hub)),
    radius: PARAMS.tireRadius,
    width: WHEEL_WIDTH,
    mass: round(wheelMass[id], 2),
    compression: round(h.maxTravel),
    extension: round(-h.minTravel),
    steerable: h.front,
    mount: { break_force: m.break_force, break_torque: m.break_torque, anchor: r3(toRubble(h.hub)) },
    mount_bonds: m.bonds,
  };
});

// ---- hybrid build: the frame as nodes and beams, bolted parts on joints ----
// The front bumper, rear bumper and nose are node-beam pieces in break groups;
// engine, gearbox and steering column stay on the rigid core (with their
// mounts, so losing them still cuts drive and steering); the rest are rigid
// bodies on breakable ball joints to the three nearest nodes of what carries
// them (headlamps on the nose, rear lamps on the rear bumper, else the frame).
const partMounts = MODEL !== 'buggy' ? null : Object.fromEntries(parts.map((p, i) => [p.name, i === 0 ? null : mountOf(p.name)]));
const graph = MODEL !== 'buggy' ? null : buildFrameGraph(liveGeometry, live, exact.parts.map((p) => p.mass), partMounts);
const CORE_PARTS = ['engine', 'transmission', 'steering_column'];
const SOFT_GROUPS = ['front_bumper', 'rear_bumper', 'nose'];
const rigMass = exact.parts.reduce((s, p, i) => s + (cls[i].cls === 'rig' ? p.mass : 0), 0);
const hybrid = MODEL !== 'buggy' ? null : {
  nodes: graph.nodes,
  beams: graph.beams.map((b) => ({ a: b.a, b: b.b, kind: b.kind, cls: b.cls ?? null, group: b.group ?? null, ...(b.strength ? { strength: Math.round(b.strength) } : {}) })),
  core_nodes: graph.coreNodes,
  core_parts: CORE_PARTS,
  // The moving suspension (arms, uprights, dampers, axles) rides on the core.
  core_root_mass: round(rigMass, 2),
  soft_groups: SOFT_GROUPS,
  rigid_parts: parts.flatMap((p, i) => {
    if (i === 0 || CORE_PARTS.includes(p.name) || SOFT_GROUPS.includes(p.name)) return [];
    const m = partMounts[p.name];
    const carrier = m.attached_to.includes('frame') ? 'frame' : m.attached_to[0];
    const center = [0, 1, 2].map((k) => p.colliders.reduce((s, c) => s + c.position[k], 0) / p.colliders.length);
    const nodes = attachNodes(graph, center, carrier);
    return [{ name: p.name, part: i, carrier, nodes, break_force: round(m.break_force / nodes.length, 0) }];
  }),
};
if (hybrid) console.error(`hybrid frame: ${hybrid.nodes.length} nodes (${round(hybrid.nodes.reduce((s, n) => s + n.mass, 0), 1)} kg), ${hybrid.beams.length} beams, core on ${hybrid.core_nodes.length} nodes, ${hybrid.rigid_parts.length} jointed parts`);

// ---- the semi's trailer: a vehicle of its own, hitched by a ball joint ----
// Its frame, deck and flatbed parts are one rigid body; its two wheels are
// raycast wheels on one axle (the leaf springs are the suspension).
function trailerOf() {
  const spec = trailerSpec(PARAMS);
  if (!spec) return null;
  let mass = 0, wheelM = 0;
  const colliders = [];
  for (const i of trailerParts) {
    if (cls[i].trailerWheel) { wheelM += exact.parts[i].mass; continue; }
    mass += exact.parts[i].mass;
    if (cls[i].collider) colliders.push(collider(live.parts[i]));
  }
  const y = spec.wheelRadius;
  return {
    mass: round(mass, 2),
    colliders,
    // Recipe +Z is the rear: Rubble -Z.
    hitch: r3(toRubble(spec.hitch)),
    wheels: [1, -1].map((s) => ({ side: s > 0 ? 'l' : 'r', hub: r3(toRubble([-s * spec.halfTrack, y, spec.axleZ])), radius: spec.wheelRadius, width: 0.3, mass: round(wheelM / 2, 2) })),
  };
}

function parentOf(i, mount) {
  if (mount.attached_to.includes('frame') || mount.attached_to.length !== 1) return null;
  const k = parts.findIndex((p) => p.name === mount.attached_to[0]);
  if (k < 0 || k >= i) throw new Error(`${parts[i].name}: carried by ${mount.attached_to[0]}, which must come earlier`);
  return k;
}

const out = {
  source: {
    recipe: 'vibe-land client/src/vehicles/dune/buggy.mjs (see scenes/dune-buggy/README.md)',
    model: MODEL,
    name: vehicleById(MODEL).name,
    parameters: PARAMS,
    strength_profile: STRENGTH_PROFILE_VERSION,
    joint_capacity: 'section-bound (vibe-land real-joint-capacity.mjs)',
    frame: 'Rubble chassis space: +Z forward, +Y up, +X left; origin on the ground at the recipe origin',
    generated_by: 'engine/scenes/dune-buggy/bake-physics.mjs',
  },
  // The vehicle's own mass (a semi's trailer is reported in `trailer`).
  total_mass: round(exact.parts.reduce((s, p, i) => s + (cls[i].cls === 'trailer' ? 0 : p.mass), 0), 2),
  parts: parts.map((p, i) => {
    const mount = i === 0 ? null : mountOf(p.name);
    if (!p.colliders.length) throw new Error(`${p.name} has no colliders`);
    return {
      name: p.name,
      role: p.role,
      mass: round(p.mass, 3),
      colliders: p.colliders,
      // A part held only by another part (a headlamp on the nose) is that part's
      // child: it leaves with it, and loads it while both are on.
      mount: mount ? { break_force: mount.break_force, break_torque: mount.break_torque, anchor: mount.anchor, parent: parentOf(i, mount) } : { break_force: 0, break_torque: 0, anchor: null, parent: null },
      report: { members: p.members, ...(mount ? { bonds: mount.bonds, attached_to: mount.attached_to } : {}) },
    };
  }),
  wheels,
  steering: { max_steer_deg: rig.steering.maxCentreRad * 180 / Math.PI },
  dropped_grazes: dropped,
  ...(hybrid ? { hybrid } : {}),
  ...(trailerParts.length ? { trailer: trailerOf() } : {}),
};
// One record per line (a part, a wheel, a node, a beam): readable diffs when
// the recipe or a rule changes, without a line per number.
function layout(v, depth = 0) {
  const pad = ' '.repeat(depth + 1), end = ' '.repeat(depth);
  if (Array.isArray(v) && v.length && v.every((x) => x && typeof x === 'object'))
    return `[\n${v.map((x) => pad + JSON.stringify(x)).join(',\n')}\n${end}]`;
  if (v && typeof v === 'object' && !Array.isArray(v) && depth < 2)
    return `{\n${Object.entries(v).map(([k, x]) => `${pad}${JSON.stringify(k)}: ${layout(x, depth + 1)}`).join(',\n')}\n${end}}`;
  return JSON.stringify(v);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, layout(out) + '\n');
for (const p of out.parts) console.error(`${p.name.padEnd(16)} ${p.mass.toFixed(1).padStart(7)} kg  ${String(p.colliders.length).padStart(3)} colliders  F ${(p.mount.break_force / 1e3).toFixed(0).padStart(5)} kN  τ ${(p.mount.break_torque / 1e3).toFixed(1).padStart(6)} kN·m`);
for (const w of wheels) console.error(`wheel ${w.corner}  ${w.mass.toFixed(1)} kg  travel -${w.extension}/+${w.compression}  F ${(w.mount.break_force / 1e3).toFixed(0)} kN  τ ${(w.mount.break_torque / 1e3).toFixed(1)} kN·m`);
console.error(`wrote ${OUT}`);
