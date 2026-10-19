// Hull clean-up for three-pinata pieces (an explicit, recorded tolerance; see README).
//
// three-pinata's Voronoi pieces carry float noise from its f32 slicing: sliver triangles of
// ~0.01 mm² and near-parallel steps up to ~1 mm. Rubble's hull cooker needs exactly planar
// faces and at most 64 vertices. A piece is rebuilt as an intersection of half-spaces:
//
// 1. Start from the planes of the piece's convex hull, each offset out to the farthest
//    original point, so the shape always contains the whole piece.
// 2. Remove small or near-parallel planes, smallest face first, while the shape stays
//    closed (its volume does not drop) and within `tolerance` of the piece.
//    A second pass removes planes the engine cannot separate (near-parallel neighbours,
//    corners closer than 0.1 mm) while the shape stays within the hard maximum.
// 3. The shape's vertices are the feasible intersections of plane triples (exactly planar
//    in f64). A shape over `maxVertices` is cut by planes into exact convex parts.
//
// `outside` reports how far the rebuilt shape extends past the original hull; the exporter
// fails when it exceeds the configured maximum.

import * as THREE from "three";
import { ConvexHull } from "three/examples/jsm/math/ConvexHull.js";

export type P = [number, number, number];
type Plane = { n: P; d: number };

export const HULL_MAX_VERTICES = 64;
/** Faces smaller than this (m^2) are candidates for removal in the clean-up. */
const SMALL_FACE = 1e-2;
/** Corners closer than this (m) are one corner computed from different plane triples. */
const VERTEX_WELD = 1e-6;

const dot = (a: P, b: P) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** A convex part: corners and its faces (corner indices, counter-clockwise from outside). */
export type Part = { points: P[]; faces: number[][] };

export type CleanHull = {
  /** Convex parts; their union is the rebuilt shape. */
  parts: Part[];
  /** Largest distance a rebuilt vertex lies outside the original hull (m). */
  outside: number;
  /** Original hull triangles and the planes they merged into. */
  triangles: number;
  planes: number;
};

export function cleanHull(points: P[], tolerance: number, hardMax: number, maxVertices = HULL_MAX_VERTICES): CleanHull {
  // three's quickhull goes wrong on repeated points (non-indexed piece meshes repeat each
  // vertex per triangle): weld exact duplicates first.
  const unique = new Map<string, P>();
  for (const p of points) unique.set(`${p[0].toFixed(9)},${p[1].toFixed(9)},${p[2].toFixed(9)}`, p);
  const hull = new ConvexHull().setFromPoints([...unique.values()].map((p) => new THREE.Vector3(...p)));
  const faces = hull.faces.map((f) => {
    const verts: P[] = [];
    let e = f.edge;
    do {
      const p = e.head().point;
      verts.push([p.x, p.y, p.z]);
      e = e.next;
    } while (e !== f.edge);
    return { n: [f.normal.x, f.normal.y, f.normal.z] as P, c: f.constant, area: f.area, verts };
  })
    // Zero-area triangles have no usable normal (three gives NaN): they bound nothing.
    .filter((f) => f.area > 1e-14 && f.n.every(Number.isFinite) && Number.isFinite(f.c));
  const hullVerts: P[] = hull.vertices.map((v) => [v.point.x, v.point.y, v.point.z]);
  for (const f of faces)
    for (const v of hullVerts)
      if (dot(v, f.n) - f.c > 1e-7) throw new Error("three ConvexHull returned a non-convex hull");

  const o: P = [0, 0, 0];
  for (const v of hullVerts) for (let k = 0; k < 3; k++) o[k] += v[k] / hullVerts.length;
  // 1. Start from the hull's own planes (coplanar triangles share one), offset out to the
  //    farthest original point: closed, exactly the piece.
  const planes: Plane[] = [];
  for (const f of [...faces].sort((x, y) => y.area - x.area)) {
    if (planes.some((q) => dot(q.n, f.n) > 1 - 1e-12 && Math.abs(q.d - f.c) < 1e-9)) continue;
    planes.push({ n: f.n, d: Math.max(...hullVerts.map((v) => dot(v, f.n))) });
  }
  // Closedness baseline: the volume the starting planes enclose (removing planes only grows it).
  const pieceVolume = volumeOf(dualVertices(planes, o));
  const overshootOf = (verts: P[]) => {
    let by = 0;
    for (const v of verts) for (const f of faces) by = Math.max(by, dot(v, f.n) - f.c);
    return by;
  };
  // 2. Remove small or near-parallel planes (sliver faces, steps), smallest face first,
  //    while the shape stays closed (its volume, which can only grow, does not drop) and
  //    within `tolerance` of the piece.
  {
    const verts = dualVertices(planes, o);
    const area = (pl: Plane) => polygonArea(verts.filter((v) => Math.abs(dot(v, pl.n) - pl.d) <= 1e-7), pl.n);
    const nearParallel = (pl: Plane) => planes.some((q) => q !== pl && dot(q.n, pl.n) > 0.9999);
    const order = planes
      .map((pl) => ({ pl, a: area(pl) }))
      .filter((x) => x.a < SMALL_FACE || nearParallel(x.pl))
      .sort((x, y) => x.a - y.a);
    for (const { pl } of order) {
      if (planes.length <= 4) break;
      const rest = planes.filter((q) => q !== pl);
      const rv = dualVertices(rest, o);
      if (rv.length >= 4 && overshootOf(rv) <= tolerance && volumeOf(rv) >= pieceVolume * (1 - 1e-6))
        planes.splice(planes.indexOf(pl), 1);
    }
  }
  // 2b. Planes the engine's cooker cannot separate: near-parallel neighbours (within
  //     ~0.08 deg it merges them into one face) and planes holding corners closer than
  //     0.1 mm. Remove the smaller face while the shape stays closed and within `hardMax`.
  // Quick screen with the dual corners; only suspect pieces pay for the exact corner search.
  const suspect = (() => {
    const dv = dualVertices(planes, o);
    if (dv.length < 4) return true;
    for (let a = 0; a < planes.length; a++)
      for (let b = a + 1; b < planes.length; b++) if (dot(planes[a].n, planes[b].n) > 1 - 1e-6) return true;
    for (let a = 0; a < dv.length; a++)
      for (let b = a + 1; b < dv.length; b++)
        if (Math.hypot(dv[a][0] - dv[b][0], dv[a][1] - dv[b][1], dv[a][2] - dv[b][2]) < 1e-4) return true;
    return false;
  })();
  for (let guard = 0; suspect && guard < 64; guard++) {
    const verts = vertices(planes);
    const on = (pl: Plane) => verts.filter((v) => Math.abs(dot(v, pl.n) - pl.d) <= 1e-7);
    const problem = new Set<Plane>();
    planes.forEach((a, i) =>
      planes.forEach((b, j) => {
        if (j > i && dot(a.n, b.n) > 1 - 1e-6)
          problem.add(polygonArea(on(a), a.n) < polygonArea(on(b), b.n) ? a : b);
      }),
    );
    // Corners closer than 0.1 mm: every plane through either corner is a candidate (removing
    // any of the nearly concurrent planes collapses the tiny edge).
    const at = (v: P) => planes.filter((pl) => Math.abs(dot(v, pl.n) - pl.d) <= 1e-7);
    for (let a = 0; a < verts.length; a++)
      for (let b = a + 1; b < verts.length; b++)
        if (Math.hypot(verts[a][0] - verts[b][0], verts[a][1] - verts[b][1], verts[a][2] - verts[b][2]) < 1e-4)
          for (const pl of [...at(verts[a]), ...at(verts[b])]) problem.add(pl);
    const tries = [...problem].sort((a, b) => polygonArea(on(a), a.n) - polygonArea(on(b), b.n));
    const removed = tries.find((pl) => {
      if (planes.length <= 4) return false;
      const rv = vertices(planes.filter((q) => q !== pl));
      return rv.length >= 4 && overshootOf(rv) <= hardMax && volumeOf(rv) >= pieceVolume * (1 - 1e-6);
    });
    if (!removed) break;
    planes.splice(planes.indexOf(removed), 1);
  }
  const finalVerts = dualVertices(planes, o);
  if (volumeOf(finalVerts) < pieceVolume * (1 - 1e-6))
    throw new Error("hull clean-up lost volume (the rebuilt shape does not contain the piece)");
  // 4. Split shapes over the vertex limit.
  const parts = split(planes, maxVertices, 0);
  let outside = 0;
  for (const part of parts) outside = Math.max(outside, overshootOf(part.points));
  return { parts, outside, triangles: faces.length, planes: planes.length };
}

function split(planes: Plane[], maxVertices: number, depth: number): Part[] {
  const verts = vertices(planes);
  if (verts.length <= maxVertices) return [{ points: verts, faces: facesOf(planes, verts) }];
  if (depth > 8) throw new Error(`hull with ${verts.length} vertices did not split within 8 cuts`);
  const lo: P = [Infinity, Infinity, Infinity];
  const hi: P = [-Infinity, -Infinity, -Infinity];
  const mid: P = [0, 0, 0];
  for (const v of verts)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], v[k]);
      hi[k] = Math.max(hi[k], v[k]);
      mid[k] += v[k] / verts.length;
    }
  const axis = [0, 1, 2].reduce((a, k) => (hi[k] - lo[k] > hi[a] - lo[a] ? k : a), 0);
  const n: P = [0, 0, 0];
  n[axis] = 1;
  const m: P = [0, 0, 0];
  m[axis] = -1;
  return [
    ...split([...planes, { n, d: mid[axis] }], maxVertices, depth + 1),
    ...split([...planes, { n: m, d: -mid[axis] }], maxVertices, depth + 1),
  ];
}

/**
 * Vertices of the intersection of half-spaces n . x <= d around an interior point `o`, by
 * duality: plane i maps to q_i = n_i / (d_i - n_i . o); each face (N . q = C) of the dual
 * points' hull is a corner o + N / C. Empty when the region is not closed around `o`.
 */
function dualVertices(planes: Plane[], o: P): P[] {
  const q: THREE.Vector3[] = [];
  for (const pl of planes) {
    const h = pl.d - dot(pl.n, o);
    if (!(h > 0)) return [];
    q.push(new THREE.Vector3(pl.n[0] / h, pl.n[1] / h, pl.n[2] / h));
  }
  if (q.length < 4) return [];
  const hull = new ConvexHull().setFromPoints(q);
  const out: P[] = [];
  for (const f of hull.faces) {
    if (!(f.area > 0) || !(f.constant > 1e-12)) return [];
    const x: P = [o[0] + f.normal.x / f.constant, o[1] + f.normal.y / f.constant, o[2] + f.normal.z / f.constant];
    if (out.some((y) => Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) <= VERTEX_WELD)) continue;
    out.push(x);
  }
  return out;
}

/** Vertices of the bounded intersection of half-spaces n . x <= d (every plane triple). */
function vertices(planes: Plane[]): P[] {
  const scale = Math.max(1e-3, ...planes.map((p) => Math.abs(p.d)));
  const eps = scale * 1e-9;
  const out: P[] = [];
  const m = planes.length;
  for (let i = 0; i < m; i++)
    for (let j = i + 1; j < m; j++)
      for (let k = j + 1; k < m; k++) {
        const x = solve(planes[i], planes[j], planes[k]);
        if (!x) continue;
        if (planes.some((p) => dot(x, p.n) > p.d + eps)) continue;
        // Planes meeting at one corner give the same point several times (float noise ~0.1 um).
        if (out.some((y) => Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]) <= VERTEX_WELD)) continue;
        out.push(x);
      }
  return out;
}

function solve(a: Plane, b: Plane, c: Plane): P | null {
  const [n1, n2, n3] = [a.n, b.n, c.n];
  const cross = (u: P, v: P): P => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const c23 = cross(n2, n3);
  const det = dot(n1, c23);
  if (Math.abs(det) < 1e-6) return null;
  const c31 = cross(n3, n1);
  const c12 = cross(n1, n2);
  return [0, 1, 2].map((k) => (a.d * c23[k] + b.d * c31[k] + c.d * c12[k]) / det) as P;
}

/** Area of the convex polygon of points lying on a plane with normal n. */
function polygonArea(points: P[], n: P): number {
  if (points.length < 3) return 0;
  const c: P = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / points.length;
  const u0: P = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cr = (a: P, b: P): P => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const u = cr(n, u0);
  const lu = Math.hypot(...u);
  const uu: P = [u[0] / lu, u[1] / lu, u[2] / lu];
  const vv = cr(n, uu);
  const pts = points
    .map((p) => {
      const d: P = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
      return [dot(d, uu), dot(d, vv)] as [number, number];
    })
    .sort((a, b) => Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0]));
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % pts.length];
    area += x0 * y1 - x1 * y0;
  }
  return Math.abs(area) / 2;
}

/** Volume of the convex hull of points. */
function volumeOf(points: P[]): number {
  if (points.length < 4) return 0;
  const h = new ConvexHull().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  let v = 0;
  for (const f of h.faces) {
    const a = f.edge.head().point;
    let e = f.edge.next;
    while (e.next !== f.edge) {
      const b = e.head().point;
      const c = e.next.head().point;
      v += a.dot(new THREE.Vector3().crossVectors(b, c)) / 6;
      e = e.next;
    }
  }
  return Math.abs(v);
}

/** Each plane's corners as a face, ordered counter-clockwise seen from outside (along n). */
function facesOf(planes: Plane[], verts: P[]): number[][] {
  const faces: number[][] = [];
  for (const pl of planes) {
    const ids = verts.map((v, i) => i).filter((i) => Math.abs(dot(verts[i], pl.n) - pl.d) <= 1e-8 * Math.max(1, Math.abs(pl.d)));
    if (ids.length < 3) continue;
    const c: P = [0, 0, 0];
    for (const i of ids) for (let k = 0; k < 3; k++) c[k] += verts[i][k] / ids.length;
    const n = pl.n;
    const a: P = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const u: P = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]];
    const lu = Math.hypot(...u);
    const uu: P = [u[0] / lu, u[1] / lu, u[2] / lu];
    const vv: P = [n[1] * uu[2] - n[2] * uu[1], n[2] * uu[0] - n[0] * uu[2], n[0] * uu[1] - n[1] * uu[0]];
    const angle = (i: number) => {
      const d: P = [verts[i][0] - c[0], verts[i][1] - c[1], verts[i][2] - c[2]];
      return Math.atan2(dot(d, vv), dot(d, uu));
    };
    faces.push(ids.sort((x, y) => angle(x) - angle(y)));
  }
  return faces;
}
