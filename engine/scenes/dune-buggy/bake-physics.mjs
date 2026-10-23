// Bake vibe-land's buggy into the Rubble preset asset
// (engine/crates/engine-vehicle/assets/dune_buggy.json, committed: CI builds the
// preset without bun or this directory's node_modules).
//
//   bun install && bun run bake:physics
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
import { classify, toRubble, RUBBLE_PARTS, CORNERS } from './parts.mjs';
import { buildFrameGraph, attachNodes } from './frame-graph.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '../../crates/engine-vehicle/assets/dune_buggy.json');
const PARAMS = { ...defaults, vehicle: 'buggy' };
// Rubble's own wheel shape: the tyre's width (live tread spans 0.30 m).
const WHEEL_WIDTH = 0.3;

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
  throw new Error(`no collision primitive for ${part.name} (${key})`);
}

// ---- groups ----
const cls = exact.parts.map(classify);
const groupOf = (i) => (cls[i].cls === 'wheel' ? `wheel_${cls[i].corner}` : RUBBLE_PARTS[cls[i].part].name);
const indexById = new Map(exact.parts.map((p, i) => [p.id, i]));

const parts = RUBBLE_PARTS.map((p) => ({ name: p.name, role: p.role, mass: 0, colliders: [], members: 0 }));
const wheelMass = Object.fromEntries(CORNERS.map((c) => [c, 0]));
exact.parts.forEach((p, i) => {
  const c = cls[i];
  if (c.cls === 'wheel') {
    wheelMass[c.corner] += p.mass;
    return;
  }
  const out = parts[c.part];
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
const bonds = [];
let dropped = 0, bounded = 0;
for (const j of exact.joints) {
  const a = indexById.get(j.a), b = indexById.get(j.b);
  if (j.interfaceAreaM2 < SOLVER_MIN_BOND_AREA_M2) { dropped++; continue; }
  const st = jointStrength(exact.parts[a].material, exact.parts[b].material);
  const area = Math.min(j.interfaceAreaM2, section[a], section[b]);
  if (area < j.interfaceAreaM2) bounded++;
  bonds.push({ a, b, force: area * Math.min(st.tensionFatal, st.shearFatal), area, anchor: toRubble(j.anchor) });
}
console.error(`${bonds.length} joints (${bounded} section-bound, ${dropped} grazes under ${SOLVER_MIN_BOND_AREA_M2} m² dropped)`);
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
const partMounts = Object.fromEntries(parts.map((p, i) => [p.name, i === 0 ? null : mountOf(p.name)]));
const graph = buildFrameGraph(liveGeometry, live, exact.parts.map((p) => p.mass), partMounts);
const CORE_PARTS = ['engine', 'transmission', 'steering_column'];
const SOFT_GROUPS = ['front_bumper', 'rear_bumper', 'nose'];
const rigMass = exact.parts.reduce((s, p, i) => s + (cls[i].cls === 'rig' ? p.mass : 0), 0);
const hybrid = {
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
console.error(`hybrid frame: ${hybrid.nodes.length} nodes (${round(hybrid.nodes.reduce((s, n) => s + n.mass, 0), 1)} kg), ${hybrid.beams.length} beams, core on ${hybrid.core_nodes.length} nodes, ${hybrid.rigid_parts.length} jointed parts`);

const out = {
  source: {
    recipe: 'vibe-land client/src/vehicles/dune/buggy.mjs (see scenes/dune-buggy/README.md)',
    parameters: PARAMS,
    strength_profile: STRENGTH_PROFILE_VERSION,
    joint_capacity: 'section-bound (vibe-land real-joint-capacity.mjs)',
    frame: 'Rubble chassis space: +Z forward, +Y up, +X left; origin on the ground at the recipe origin',
    generated_by: 'engine/scenes/dune-buggy/bake-physics.mjs',
  },
  total_mass: round(exact.parts.reduce((s, p) => s + p.mass, 0), 2),
  parts: parts.map((p, i) => {
    const mount = i === 0 ? null : mountOf(p.name);
    if (!p.colliders.length) throw new Error(`${p.name} has no colliders`);
    return {
      name: p.name,
      role: p.role,
      mass: round(p.mass, 3),
      colliders: p.colliders,
      mount: mount ? { break_force: mount.break_force, break_torque: mount.break_torque, anchor: mount.anchor } : { break_force: 0, break_torque: 0, anchor: null },
      report: { members: p.members, ...(mount ? { bonds: mount.bonds, attached_to: mount.attached_to } : {}) },
    };
  }),
  wheels,
  steering: { max_steer_deg: rig.steering.maxCentreRad * 180 / Math.PI },
  dropped_grazes: dropped,
  hybrid,
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
