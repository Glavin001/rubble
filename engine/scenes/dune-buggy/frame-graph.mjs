// The buggy's frame as a node–beam structure, for the hybrid build (soft frame,
// rigid bolted-on parts). Both bakes call buildFrameGraph on the same live
// build, so physics nodes and render bindings agree.
//
// - Every structural tube of the frame classes (cell, crumple, panel) is a
//   chain of beams between nodes: nodes sit at tube ends, merged within
//   MERGE m, and where a tube end lands on another tube's span that tube is
//   split there (a T-junction).
// - Brace beams (invisible) tie each node to its nearest neighbours so the
//   structure has no free mechanisms: a space frame's joints resist bending,
//   which plain beams between two nodes cannot.
// - The front bumper, rear bumper and nose are detachable: their tubes get
//   nodes of their own (offset off any frame node they start from), tied to the
//   frame by mount beams in one break group each, rated from the part's mount.
// - Frame-class mass goes onto the nodes: a tube's onto its chain, anything
//   else onto the three nodes it is bound to for rendering.
//
// All points are in the Rubble design frame (R applied).

import * as T from 'three';
import { classify, toRubble, RUBBLE_PARTS, PART_INDEX } from './parts.mjs';

const MERGE = 0.035;      // m: tube ends this close share a node
const MIN_TUBE_R = 0.012; // thinner tubes (gauge needles, pedals) are not structure
const BRACE_K = 5;        // brace each node to this many nearest unbraced neighbours
const BRACE_MAX = 1.1;    // m
const BRACE_SPREAD = 25 * Math.PI / 180; // braces differ from a node's other beams by this
const SEG_MAX = 0.42;     // m: longer tube spans get nodes along them (to 4 per tube)
const CHAIN_MAX = 4;      // nodes per tube (the renderer skins a tube along its chain)
export const GROUPS = ['front_bumper', 'rear_bumper', 'nose'];

const v3 = (a) => new T.Vector3(...a);

/** Endpoints (recipe frame) and radius of a live part if it is a tube. */
function tubeOf(part, key) {
  if (!(key.startsWith('ring') || key.startsWith('cylinder'))) return null;
  const pos = new T.Vector3(), q = new T.Quaternion(), s = new T.Vector3();
  part.matrix.decompose(pos, q, s);
  const r = Math.max(s.x, s.y), len = s.z;
  if (len < 2.5 * r || len < 0.08 || r < MIN_TUBE_R) return null;
  const z = new T.Vector3(0, 0, 1).applyQuaternion(q);
  return { a: pos.clone().addScaledVector(z, -len / 2).toArray(), b: pos.clone().addScaledVector(z, len / 2).toArray(), r };
}

/**
 * @param {object} live    LiveGeometry instance after build (templates)
 * @param {object} model   its build result (parts)
 * @param {number[]} masses exact per-part masses (same order)
 * @param {object} mounts  part name → {break_force} (the rigid preset's mounts)
 */
export function buildFrameGraph(live, model, masses, mounts) {
  const keyOf = new Map([...live.templates].map(([k, g]) => [g, k]));
  const cls = model.parts.map(classify);
  const frameish = (i) => ['cell', 'crumple', 'panel'].includes(cls[i].cls);
  const groupOf = (i) => {
    const name = RUBBLE_PARTS[cls[i].part].name;
    return GROUPS.includes(name) ? name : 'frame';
  };

  // ---- tubes ----
  const tubes = [];
  model.parts.forEach((p, i) => {
    if (!frameish(i)) return;
    const t = tubeOf(p, keyOf.get(p.geometry));
    if (t) tubes.push({ instance: i, a: toRubble(t.a), b: toRubble(t.b), r: t.r, group: groupOf(i), cls: cls[i].cls });
  });

  // ---- nodes: tube ends, merged per group (a detachable group never shares a frame node) ----
  const nodes = [];
  const nodeAt = (p, group, cl) => {
    for (let k = 0; k < nodes.length; k++) {
      if (nodes[k].group === group && v3(nodes[k].p).distanceTo(v3(p)) < MERGE) return k;
    }
    nodes.push({ p: [...p], group, cls: cl, mass: 0 });
    return nodes.length - 1;
  };
  for (const t of tubes) {
    let a = t.a, b = t.b;
    if (t.group !== 'frame') {
      // Off any frame node it starts from, by MERGE along the tube.
      const d = v3(b).sub(v3(a)).normalize().multiplyScalar(MERGE * 1.2);
      const nearFrame = (p) => tubes.some((o) => o.group === 'frame' && (v3(o.a).distanceTo(v3(p)) < MERGE || v3(o.b).distanceTo(v3(p)) < MERGE));
      if (nearFrame(a)) a = v3(a).add(d).toArray();
      if (nearFrame(b)) b = v3(b).sub(d).toArray();
    }
    t.na = nodeAt(a, t.group, t.cls);
    t.nb = nodeAt(b, t.group, t.cls);
  }
  // A node strictly inside another tube of its group splits it.
  for (const t of tubes) {
    const A = v3(nodes[t.na].p), B = v3(nodes[t.nb].p), AB = B.clone().sub(A), L2 = AB.lengthSq();
    const inner = [];
    nodes.forEach((n, k) => {
      if (k === t.na || k === t.nb || n.group !== t.group) return;
      const s = v3(n.p).sub(A).dot(AB) / L2;
      if (s <= 0.02 || s >= 0.98) return;
      const closest = A.clone().addScaledVector(AB, s);
      if (closest.distanceTo(v3(n.p)) < MERGE) inner.push([s, k]);
    });
    inner.sort((x, y) => x[0] - y[0]);
    t.chain = [t.na, ...inner.map(([, k]) => k), t.nb];
  }
  // Subdivide long spans: a node sphere every SEG_MAX or so (contact along the
  // tube), and somewhere for it to bend. Longest span first, to CHAIN_MAX nodes.
  for (const t of tubes) {
    for (;;) {
      if (t.chain.length >= CHAIN_MAX) break;
      let worst = -1, wl = SEG_MAX;
      for (let k = 0; k + 1 < t.chain.length; k++) {
        const l = v3(nodes[t.chain[k]].p).distanceTo(v3(nodes[t.chain[k + 1]].p));
        if (l > wl) { wl = l; worst = k; }
      }
      if (worst < 0) break;
      const mid = v3(nodes[t.chain[worst]].p).add(v3(nodes[t.chain[worst + 1]].p)).multiplyScalar(0.5).toArray();
      nodes.push({ p: mid, group: t.group, cls: t.cls, mass: 0 });
      t.chain.splice(worst + 1, 0, nodes.length - 1);
    }
  }

  // ---- beams ----
  const beams = [];
  const has = new Set();
  const key = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const addBeam = (a, b, kind, extra = {}) => {
    if (a === b || has.has(key(a, b))) return null;
    has.add(key(a, b));
    beams.push({ a, b, kind, ...extra });
    return beams.length - 1;
  };
  for (const t of tubes) {
    for (let k = 0; k + 1 < t.chain.length; k++) addBeam(t.chain[k], t.chain[k + 1], 'tube', { cls: t.cls, group: t.group === 'frame' ? null : t.group, radius: t.r });
  }
  // Braces within each group (frame braces only to frame nodes). Beams only
  // resist stretch, so a node needs beams in several directions: a node in the
  // middle of a straight tube has none across it. Each node takes its nearest
  // neighbours whose direction differs by over BRACE_SPREAD from every beam it
  // already has, until it has BRACE_K braces.
  const dirsAt = nodes.map(() => []);
  const unit = (a, b) => v3(nodes[b].p).sub(v3(nodes[a].p)).normalize();
  for (const b of beams) { dirsAt[b.a].push(unit(b.a, b.b)); dirsAt[b.b].push(unit(b.b, b.a)); }
  const spreadCos = Math.cos(BRACE_SPREAD);
  for (let i = 0; i < nodes.length; i++) {
    const near = nodes
      .map((n, k) => [v3(n.p).distanceTo(v3(nodes[i].p)), k])
      .filter(([d, k]) => k !== i && nodes[k].group === nodes[i].group && d < BRACE_MAX)
      .sort((x, y) => x[0] - y[0]);
    let added = 0;
    for (const [, k] of near) {
      if (added >= BRACE_K) break;
      const d = unit(i, k);
      if (dirsAt[i].some((e) => Math.abs(e.dot(d)) > spreadCos)) continue;
      const id = addBeam(i, k, 'brace', { cls: nodes[i].cls === 'crumple' || nodes[k].cls === 'crumple' ? 'crumple' : 'cell', group: nodes[i].group === 'frame' ? null : nodes[i].group });
      if (id === null) continue;
      dirsAt[i].push(d);
      dirsAt[k].push(d.clone().negate());
      added++;
    }
  }
  // Mount beams: each detachable node to its 3 nearest frame nodes, in the group's break group.
  const groupNodes = Object.fromEntries(GROUPS.map((g) => [g, nodes.map((n, k) => (n.group === g ? k : -1)).filter((k) => k >= 0)]));
  const frameNodes = nodes.map((n, k) => (n.group === 'frame' ? k : -1)).filter((k) => k >= 0);
  const nearestFrame = (p, n) => frameNodes.map((k) => [v3(nodes[k].p).distanceTo(v3(p)), k]).sort((x, y) => x[0] - y[0]).slice(0, n).map(([, k]) => k);
  const mountBeams = Object.fromEntries(GROUPS.map((g) => [g, []]));
  for (const g of GROUPS) {
    for (const k of groupNodes[g]) {
      for (const f of nearestFrame(nodes[k].p, 3)) {
        const id = addBeam(k, f, 'mount', { group: g });
        if (id !== null) mountBeams[g].push(id);
      }
    }
  }
  for (const g of GROUPS) {
    const F = mounts[g]?.break_force ?? 0;
    for (const id of mountBeams[g]) beams[id].strength = F / mountBeams[g].length;
  }

  // ---- render bindings and mass ----
  // Frame-class instances that are not tubes ride on three nodes of their group.
  const nearestOfGroup = (p, group, n) => nodes
    .map((nd, k) => [v3(nd.p).distanceTo(v3(p)), k])
    .filter(([, k]) => nodes[k].group === group)
    .sort((x, y) => x[0] - y[0]);
  const triangleOf = (p, group) => {
    // Three nearest non-collinear nodes.
    const list = nearestOfGroup(p, group);
    const [a, b] = [list[0][1], list[1][1]];
    const A = v3(nodes[a].p), AB = v3(nodes[b].p).sub(A).normalize();
    for (let j = 2; j < list.length; j++) {
      const c = list[j][1];
      const AC = v3(nodes[c].p).sub(A);
      if (AC.clone().cross(AB).length() > 0.25 * AC.length()) return [a, b, c];
    }
    throw new Error('no three non-collinear nodes');
  };
  const bindings = model.parts.map(() => null);
  const tubeOfInstance = new Map(tubes.map((t) => [t.instance, t]));
  model.parts.forEach((p, i) => {
    if (!frameish(i)) return;
    const t = tubeOfInstance.get(i);
    if (t) {
      // The whole chain (to CHAIN_MAX nodes) and each node's fraction of the length.
      const lens = t.chain.slice(1).map((k, j) => v3(nodes[t.chain[j]].p).distanceTo(v3(nodes[k].p)));
      const total = lens.reduce((a, b) => a + b, 0);
      let acc = 0;
      bindings[i] = { mode: 'tube', nodes: t.chain.slice(0, CHAIN_MAX), at: [0, ...lens.map((l) => (acc += l) / total)].map((x) => Math.round(x * 1e5) / 1e5) };
      // Mass along the chain, by segment length.
      const segs = t.chain.slice(1).map((k, j) => [t.chain[j], k, v3(nodes[t.chain[j]].p).distanceTo(v3(nodes[k].p))]);
      const L = segs.reduce((s, x) => s + x[2], 0) || 1;
      for (const [a, b, l] of segs) { nodes[a].mass += masses[i] * l / L / 2; nodes[b].mass += masses[i] * l / L / 2; }
    } else {
      const tri = triangleOf(toRubble(p.center), groupOf(i));
      bindings[i] = { mode: 'frame', nodes: tri };
      for (const k of tri) nodes[k].mass += masses[i] / 3;
    }
  });

  // ---- core attachment: the lower wishbone pivots (the suspension's frame mounts) ----
  const pivots = [];
  model.parts.forEach((p, i) => {
    if (!/pivot bolt$/.test(p.name)) return;
    const near = nearestOfGroup(toRubble(p.center), 'frame')[0];
    if (near[0] < 0.06 && !pivots.includes(near[1])) pivots.push(near[1]);
  });

  return {
    nodes: nodes.map((n) => ({ position: n.p.map((x) => Math.round(x * 1e5) / 1e5), mass: Math.round(n.mass * 1e4) / 1e4, group: n.group, cls: n.cls })),
    beams,
    bindings,
    coreNodes: pivots,
    groupNodes,
    nearestOfGroup,
  };
}

/** Nodes a rigid part bolts to: its three nearest nodes of `group`. */
export function attachNodes(graph, center, group) {
  return graph.nearestOfGroup(center, group).slice(0, 3).map(([, k]) => k);
}

export { PART_INDEX };
