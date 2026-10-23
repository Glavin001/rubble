// Bake the buggy's render mesh for Rubble's WebGPU demos:
//   engine/crates/engine-webgpu/generated/dune-buggy/buggy-mesh.{json,bin}
// (git-ignored; build-web.sh runs `bun run bake:mesh`).
//
// The live build of the recipe (src/live-geometry.mjs) is ~930 parts over ~50
// shared primitives. The bin holds each primitive once (positions, normals,
// indices); the json lists every part as an instance of one (its rest matrix in
// the recipe frame, its colour, its Rubble part, its wheel corner and its
// suspension role), plus the rig definition the demo poses the suspension with
// (buggy-rig.mjs, a port of src/visual-rig.mjs) and reference poses from the
// original Three.js rig for that port's tests.

import * as T from 'three';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaults, materials } from './src/buggy.mjs';
import { LiveGeometry } from './src/live-geometry.mjs';
import { createRigDefinition, cornerIds } from './src/vehicle-rig.mjs';
import { VisualRig } from './src/visual-rig.mjs';
import { classify, RUBBLE_PARTS, CORNERS, CLASSES } from './parts.mjs';
import { buildFrameGraph } from './frame-graph.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(here, '../../crates/engine-webgpu/generated/dune-buggy');
const PARAMS = { ...defaults, vehicle: 'buggy' };
const r6 = (x) => Math.round(x * 1e6) / 1e6;

const live = new LiveGeometry();
const model = live.build(PARAMS);

// ---- primitives ----
const templates = [];
const templateOf = new Map();
const chunks = [];
let bytes = 0;
const push = (typed) => {
  const at = bytes;
  chunks.push(new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  bytes += typed.byteLength;
  return at;
};
for (const part of model.parts) {
  const g = part.geometry;
  if (templateOf.has(g)) continue;
  const pos = g.attributes.position, nrm = g.attributes.normal;
  if (!nrm) throw new Error(`${part.name}: template without normals`);
  const vertices = new Float32Array(pos.count * 6);
  for (let i = 0; i < pos.count; i++) {
    vertices.set([pos.getX(i), pos.getY(i), pos.getZ(i), nrm.getX(i), nrm.getY(i), nrm.getZ(i)], i * 6);
  }
  const indices = g.index ? Uint32Array.from(g.index.array) : Uint32Array.from({ length: pos.count }, (_, i) => i);
  const key = [...live.templates].find(([, v]) => v === g)?.[0] ?? `template ${templates.length}`;
  templateOf.set(g, templates.length);
  templates.push({ key, vertexCount: pos.count, indexCount: indices.length, vertexOffset: push(vertices), indexOffset: push(indices) });
}

// ---- hybrid build: the frame graph's render bindings (bake-physics builds the
// same graph from the same live build; masses do not change the bindings) ----
const graph = buildFrameGraph(live, model, model.parts.map(() => 0), {});

// ---- instances ----
const palette = Object.keys(materials);
const instances = model.parts.map((p) => {
  const c = classify(p);
  const e = p.matrix.elements;
  const m = p.motion;
  return {
    name: p.name,
    template: templateOf.get(p.geometry),
    // Column-major 3×4 (Three.js elements without the last row).
    matrix: [e[0], e[1], e[2], e[4], e[5], e[6], e[8], e[9], e[10], e[12], e[13], e[14]].map(r6),
    material: palette.indexOf(p.material),
    part: c.part,
    cls: CLASSES.indexOf(c.cls),
    corner: c.corner ? CORNERS.indexOf(c.corner) : -1,
    role: m?.role ?? null,
    ...(m?.endpoints ? { endpoint: m.endpoints[0].map(r6) } : {}),
    ...(m?.radius ? { radius: r6(m.radius) } : {}),
    // Hybrid: a frame tube spans two nodes; other frame parts ride three.
    ...(graph.bindings[model.parts.indexOf(p)] ? { bind: graph.bindings[model.parts.indexOf(p)] } : {}),
  };
});

// ---- rig ----
const rig = createRigDefinition(PARAMS);
const corners = Object.fromEntries(cornerIds.map((id) => {
  const h = rig.corners[id];
  const keep = ['id', 'front', 'side', 'z', 'radius', 'lowerPivot', 'upperPivot', 'lower', 'upper', 'hub', 'shockTop', 'shockBottom',
    'axleInner', 'tieInner', 'tieOuter', 'neutralAngle', 'shockLength', 'minAngle', 'maxAngle', 'minTravel', 'maxTravel'];
  return [id, Object.fromEntries(keep.map((k) => [k, h[k]]))];
}));
const rigDef = { corners, steering: { columnStart: rig.steering.columnStart, columnEnd: rig.steering.columnEnd, ratio: rig.steering.ratio } };

// ---- reference poses from the original rig (buggy-rig.mjs tests) ----
const visual = new VisualRig(model);
const posed = model.parts.map((p, i) => [p, i]).filter(([p]) => p.motion && p.motion.role !== 'wheel' && p.motion.role !== 'hub');
// One wheel and one hub part per corner stand for their groups (same matrix).
for (const id of cornerIds) for (const role of ['wheel', 'hub']) {
  const i = model.parts.findIndex((p) => p.motion?.corner === id && p.motion.role === role);
  posed.push([model.parts[i], i]);
}
const poses = [];
const travels = [-0.2, -0.05, 0, 0.07, 0.3];
for (let k = 0; k < 12; k++) {
  const wheels = {};
  cornerIds.forEach((id, j) => {
    wheels[id] = { travelM: travels[(k + j) % travels.length], steeringRad: [0, 0.3, -0.42, 0.11][(k + j) % 4], rotationRad: 0.7 * k - j, grounded: true };
  });
  const pose = { wheels, chassis: { position: [0, 0, 0], rotation: [0, 0, 0, 1] } };
  visual.applyPose(pose);
  poses.push({
    wheels,
    matrices: posed.map(([p, i]) => [i, Array.from(visual.matrixFor(p).elements).map(r6)]),
    coil: Object.fromEntries(cornerIds.map((id) => [id, r6(visual.states[id].coilLength)])),
  });
}

mkdirSync(OUT, { recursive: true });
const bin = new Uint8Array(bytes);
let off = 0;
for (const c of chunks) { bin.set(c, off); off += c.byteLength; }
writeFileSync(resolve(OUT, 'buggy-mesh.bin'), bin);
writeFileSync(resolve(OUT, 'buggy-mesh.json'), JSON.stringify({
  format: 'rubble-dune-buggy-mesh-1',
  source: 'engine/scenes/dune-buggy (vibe-land buggy.mjs, live geometry)',
  frame: 'recipe frame: metres, +Y up, -Z front, left at x < 0; Rubble = R(x, y, z) = (-x, y, -z)',
  parameters: PARAMS,
  vertexStride: 24,
  palette: palette.map((name) => ({ name, color: materials[name].color, metalness: materials[name].metalness, roughness: materials[name].roughness })),
  parts: RUBBLE_PARTS.map((p) => p.name),
  corners: CORNERS,
  classes: CLASSES,
  templates,
  instances,
  rig: rigDef,
  // Hybrid frame nodes at rest (Rubble design frame), in engine_softbody::buggy order.
  frameNodes: graph.nodes.map((n) => n.position),
}));
writeFileSync(resolve(OUT, 'rig-fixtures.json'), JSON.stringify({ poses }));
const tris = instances.reduce((s, i) => s + templates[i.template].indexCount / 3, 0);
console.error(`buggy mesh: ${templates.length} primitives (${(bytes / 1024).toFixed(0)} KiB), ${instances.length} parts, ${tris} triangles → ${OUT}`);
