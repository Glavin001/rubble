//! Edge-against-edge vehicle contact (a Rubble extension, `ExplicitBeams::edge_contact`,
//! off by default; CPU reference first, known-bugs S4).
//!
//! Node-against-triangle contact (spec §5, §36) keeps every node of one car out of the
//! other's skin, but two skins can still cross between their nodes: a hood's edge
//! through the other car's hood with every node outside. For two triangle meshes the
//! complete contact is vertex–face both ways (the node pass, every owner in turn) and
//! edge–edge (this pass). Measured before it (the generated sedans head on at 60 + 60
//! km/h): 62 / 54 skin edges through the other car's triangles, and the cars fused
//! (pushed apart they part 0.21 m against 6.11 m with contact off).
//!
//! The rule, per pair of vehicles A < B with overlapping boxes, per pair of live skin
//! edges (edges of live collision triangles, not anti-clip only) eA of A and eB of B:
//! * the closest points of the two segments, strictly inside both (an end is a node:
//!   the node pass's);
//! * the normal n = eA × eB, pointed out of A by A's edge normal (the sum of its live
//!   triangles' unit normals: the skin faces out of its solid), and required to point
//!   into B by B's edge normal (else the pair is no face-to-face contact: skipped);
//! * the gap g = n·(cB − cA): a contact while −`EDGE_DEPTH` ≤ g ≤ r (the larger
//!   contact range of the two), pushed with depth min(r − g, r) (anti-clip's
//!   `clip_depth`: a crossed pair is pushed as one at the plane);
//! * the response is the node pass's law (`contact_response`; two-body with each
//!   node's share of the pass's edge hits as its sub-body count, dissipative with §37's
//!   position step when either vehicle has it) on the points cA = (1 − s)·a0 + s·a1 and
//!   cB = (1 − t)·b0 + t·b1, B taking f by its weights and A −f by its.

use super::ExplicitBeamsWorld;
use crate::explicit_beams::{
    contact_response, shift_depth, ContactSurface, ExplicitBeams, Node, NODE_IMMOVABLE, STEP,
};
use glam::Vec3;
use std::collections::HashMap;

/// How far behind each other two crossed edges are still pushed apart (m): anti-clip's
/// memory depth (`CLIP_DEPTH`), for the same reason (a deeper crossing is no longer a
/// local contact: its normal and side are not known).
pub const EDGE_DEPTH: f32 = crate::explicit_contact::CLIP_DEPTH;
/// Measurement switches (`EDGE_FACING=0`: no check that B faces A; `EDGE_DEPTH_M`:
/// the crossing depth still pushed, m).
fn facing_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("EDGE_FACING").is_ok_and(|v| v == "0"))
}
fn vertex_edge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EDGE_VERTEX").is_ok_and(|v| v == "1"))
}
fn edge_depth() -> f32 {
    static D: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("EDGE_DEPTH_M").ok().and_then(|v| v.parse().ok()).unwrap_or(EDGE_DEPTH))
}

/// Closest points this close to an end (of the segment's parameter) are the node's.
const END: f32 = 1e-3;
/// Grid cell of the edge broad phase (m): about a skin edge's length.
const CELL: f32 = 0.3;

/// A live skin edge of one vehicle this step: its nodes, world ends and outward normal.
struct SkinEdge {
    i: u32,
    j: u32,
    p: Vec3,
    q: Vec3,
    out: Vec3,
    /// The opposite corners of its live triangles (one on a boundary edge).
    opp: [Option<Vec3>; 2],
    /// Convex (or a boundary edge): the skin turns away from outside at it. A concave
    /// edge's region belongs to its faces (Offset Geometric Contact's edge block is empty
    /// there).
    convex: bool,
}

/// The vehicle's live skin edges with their outward normals (unit sum of the live
/// triangles' unit normals at each edge).
fn skin_edges(v: &ExplicitBeams) -> Vec<SkinEdge> {
    let w = |i: u32| v.origin + v.nodes[i as usize].pos;
    let mut by: HashMap<(u32, u32), (Vec3, Vec<(Vec3, Vec3)>)> = HashMap::new();
    for (t, tri) in v.triangles.iter().enumerate() {
        if v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t) {
            continue;
        }
        let n = tri.nodes;
        if n.iter().any(|&i| v.is_retired(i as usize)) {
            continue;
        }
        let (a, b, c) = (w(n[0]), w(n[1]), w(n[2]));
        let nn = (b - a).cross(c - a).normalize_or_zero();
        let corner = [a, b, c];
        for k in 0..3 {
            let (x, y) = (n[k], n[(k + 1) % 3]);
            let e = by.entry((x.min(y), x.max(y))).or_insert((Vec3::ZERO, Vec::new()));
            e.0 += nn;
            e.1.push((corner[(k + 2) % 3], nn));
        }
    }
    let mut e: Vec<SkinEdge> = by
        .into_iter()
        .filter_map(|((i, j), (s, faces))| {
            let out = s.normalize_or_zero();
            let opp = [faces.first().map(|f| f.0), faces.get(1).map(|f| f.0)];
            // Convex: each face's opposite corner is behind the other face's plane.
            let convex = match faces.as_slice() {
                [(o0, n0), (o1, n1)] => n0.dot(*o1 - w(i)) <= 0.0 && n1.dot(*o0 - w(i)) <= 0.0,
                _ => true,
            };
            (out != Vec3::ZERO).then(|| SkinEdge { i, j, p: w(i), q: w(j), out, opp, convex })
        })
        .collect();
    // (A fixed order: the hits' order sets the sums' order.)
    e.sort_by_key(|x| (x.i, x.j));
    e
}

/// The speed (m/s) at which the points cA = (1 − s)·a0 + s·a1 of vehicle `va` and
/// cB = (1 − t)·b0 + t·b1 of `vb` close along n (out of A); 0 when they part.
fn closing(va: &ExplicitBeams, a: [u32; 2], s: f32, vb: &ExplicitBeams, b: [u32; 2], t: f32, n: Vec3) -> f32 {
    let v = |x: &ExplicitBeams, i: u32| x.nodes[i as usize].vel;
    let ua = v(va, a[0]) * (1.0 - s) + v(va, a[1]) * s;
    let ub = v(vb, b[0]) * (1.0 - t) + v(vb, b[1]) * t;
    (-(ub - ua).dot(n)).max(0.0)
}

fn cell(x: Vec3) -> (i32, i32, i32) {
    let c = (x / CELL).floor();
    (c.x as i32, c.y as i32, c.z as i32)
}

/// One edge hit: A's edge (a0, a1) at s, B's (b0, b1) at t, normal n out of A, depth p.
struct EdgeHit {
    a: [u32; 2],
    b: [u32; 2],
    s: f32,
    t: f32,
    n: Vec3,
    p: f32,
    /// Speculative: the gap past the band the pair may still close this step (m).
    spec: f32,
    /// How far the pair has crossed (m; 0 not crossed).
    behind: f32,
}

impl ExplicitBeamsWorld {
    /// The edge pass over every pair of vehicles with edge contact (either of the two)
    /// and overlapping boxes (after the node pass, from the same positions and the
    /// pass's starting forces).
    pub(super) fn edge_pass(&mut self) {
        let nv = self.vehicles.len();
        if !self.vehicles.iter().any(|v| v.edge_contact) {
            return;
        }
        let edges: Vec<Option<Vec<SkinEdge>>> = (0..nv)
            .map(|a| {
                let skip = self.skip.get(a).copied().unwrap_or(false);
                (!skip).then(|| skin_edges(&self.vehicles[a]))
            })
            .collect();
        for a in 0..nv {
            for b in a + 1..nv {
                if !(self.vehicles[a].edge_contact || self.vehicles[b].edge_contact) {
                    continue;
                }
                let (Some(ea), Some(eb)) = (&edges[a], &edges[b]) else {
                    continue;
                };
                let (alo, ahi) = self.boxes[a];
                let (blo, bhi) = self.boxes[b];
                if ahi.cmplt(blo).any() || bhi.cmplt(alo).any() {
                    continue;
                }
                let hits = self.edge_hits(a, b, ea, eb);
                if !hits.is_empty() {
                    self.edge_apply(a, b, &hits);
                }
                if vertex_edge() {
                    // A node against the other's edge, in its edge's wedge (both ways).
                    for (o, x, eo) in [(a, b, ea), (b, a, eb)] {
                        let hits = self.vertex_edge_hits(o, x, eo);
                        if !hits.is_empty() {
                            self.edge_apply(o, x, &hits);
                        }
                    }
                }
            }
        }
    }

    fn edge_hits(&mut self, a: usize, b: usize, ea: &[SkinEdge], eb: &[SkinEdge]) -> Vec<EdgeHit> {
        let r = self.vv_range.unwrap_or(self.vehicles[a].collision_range.max(self.vehicles[b].collision_range));
        let travel = |v: &ExplicitBeams| v.nodes.iter().map(|n| n.vel.length()).fold(0.0f32, f32::max) * STEP;
        let pad = r + edge_depth() + travel(&self.vehicles[a]) + travel(&self.vehicles[b]);
        // A's edges in the grid, by their padded boxes.
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (k, e) in ea.iter().enumerate() {
            let (lo, hi) = (cell(e.p.min(e.q) - Vec3::splat(pad)), cell(e.p.max(e.q) + Vec3::splat(pad)));
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        grid.entry((x, y, z)).or_default().push(k as u32);
                    }
                }
            }
        }
        let mut out = Vec::new();
        let mut seen: Vec<u32> = Vec::new();
        for f in eb {
            seen.clear();
            let (lo, hi) = (cell(f.p.min(f.q)), cell(f.p.max(f.q)));
            for x in lo.0..=hi.0 {
                for y in lo.1..=hi.1 {
                    for z in lo.2..=hi.2 {
                        if let Some(l) = grid.get(&(x, y, z)) {
                            seen.extend_from_slice(l);
                        }
                    }
                }
            }
            seen.sort_unstable();
            seen.dedup();
            for &k in &seen {
                let e = &ea[k as usize];
                let (s, t) = engine_schema::feature_query::closest_seg_seg(e.p, e.q, f.p, f.q);
                self.edge_stats[0] += 1;
                let key = (a, b, e.i, e.j, f.i, f.j);
                let step = self.steps;
                // EDGE_TRACE (a measurement): each tested pair's outcome this step, and for
                // a hit first seen crossed, the pair's outcome the step before.
                let mut note = |w: &mut Self, code: u8, g: f32| {
                    if let Some(tr) = w.edge_trace.as_mut() {
                        if code == 4 && tr.get(&key).is_some_and(|p| p.0 + 1 == step && p.1 == 5) {
                            // A held pair overrun past EDGE_DEPTH this step.
                            w.edge_first[5] += 1;
                        }
                        if code == 5 && g < 0.0 {
                            let prev = tr.get(&key).filter(|p| p.0 + 1 == step).map(|p| p.1);
                            if prev != Some(5) {
                                w.edge_first[prev.map_or(6, |c| c as usize)] += 1;
                            }
                        }
                        tr.insert(key, (step, code));
                    }
                };
                let a_end = s <= END || s >= 1.0 - END;
                let b_end = t <= END || t >= 1.0 - END;
                if a_end || b_end {
                    self.edge_stats[1] += 1;
                    note(self, 0, 0.0);
                    continue;
                }
                let mut n = (e.q - e.p).cross(f.q - f.p);
                let len = n.length();
                if len < 1e-8 {
                    self.edge_stats[2] += 1;
                    note(self, 1, 0.0);
                    continue;
                }
                n /= len;
                if n.dot(e.out) < 0.0 {
                    n = -n;
                }
                // B's skin must face A here (B's outward against n), else no contact.
                if facing_check() && n.dot(f.out) >= 0.0 {
                    // (Counted only when close: the pairs the check turns away.)
                    let g = n.dot((f.p + (f.q - f.p) * t) - (e.p + (e.q - e.p) * s));
                    if g <= r && g >= -edge_depth() {
                        self.edge_stats[3] += 1;
                    }
                    note(self, 2, g);
                    continue;
                }
                let ca = e.p + (e.q - e.p) * s;
                let cb = f.p + (f.q - f.p) * t;
                let g = n.dot(cb - ca);
                let mut spec = 0.0;
                if g > r {
                    let reach = if crate::explicit_contact::speculative() {
                        closing(&self.vehicles[a], [e.i, e.j], s, &self.vehicles[b], [f.i, f.j], t, n) * STEP
                    } else {
                        0.0
                    };
                    if g - r >= reach {
                        note(self, 3, g);
                        continue;
                    }
                    spec = g - r;
                }
                if g < -edge_depth() {
                    self.edge_stats[4] += 1;
                    note(self, 4, g);
                    continue;
                }
                note(self, 5, g);
                self.edge_stats[5] += 1;
                if g < 0.0 {
                    self.edge_stats[6] += 1;
                }
                out.push(EdgeHit { a: [e.i, e.j], b: [f.i, f.j], s, t, n, p: (r - g).max(0.0).min(r), spec, behind: (-g).max(0.0) });
            }
        }
        out
    }

    /// Vehicle `x`'s contact nodes against owner `o`'s skin edges, each in its edge's
    /// region only (Offset Geometric Contact's edge block): the closest point inside the
    /// edge, the edge convex, the node outside (along the edge's outward normal) and
    /// beyond the edge in both faces' planes: (p − f)·(f − c) ≥ 0 for each face's
    /// opposite corner c and its foot f on the edge's line. Over a face the node is the
    /// face's (the node pass); so no two contacts push the same point two ways.
    fn vertex_edge_hits(&mut self, o: usize, x: usize, eo: &[SkinEdge]) -> Vec<EdgeHit> {
        let r = self.vv_range.unwrap_or(self.vehicles[o].collision_range.max(self.vehicles[x].collision_range));
        let travel = |v: &ExplicitBeams| {
            if crate::explicit_contact::speculative() {
                v.nodes.iter().map(|n| n.vel.length()).fold(0.0f32, f32::max) * STEP
            } else {
                0.0
            }
        };
        let (travel_o, travel_x) = (travel(&self.vehicles[o]), travel(&self.vehicles[x]));
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (k, e) in eo.iter().enumerate() {
            if !e.convex {
                continue;
            }
            let pad = r + travel_o + travel_x;
            let (lo, hi) = (cell(e.p.min(e.q) - Vec3::splat(pad)), cell(e.p.max(e.q) + Vec3::splat(pad)));
            for gx in lo.0..=hi.0 {
                for gy in lo.1..=hi.1 {
                    for gz in lo.2..=hi.2 {
                        grid.entry((gx, gy, gz)).or_default().push(k as u32);
                    }
                }
            }
        }
        let vx = &self.vehicles[x];
        let mut out = Vec::new();
        for &h in &vx.contact_points {
            if vx.is_retired(h as usize) {
                continue;
            }
            let p = vx.origin + vx.nodes[h as usize].pos;
            let Some(list) = grid.get(&cell(p)) else { continue };
            for &k in list {
                let e = &eo[k as usize];
                let d = e.q - e.p;
                let ll = d.dot(d);
                if ll < 1e-12 {
                    continue;
                }
                let s = (p - e.p).dot(d) / ll;
                if s <= END || s >= 1.0 - END {
                    continue;
                }
                let c = e.p + d * s;
                let g = (p - c).length();
                if g < 1e-6 || g > r + travel_o + travel_x {
                    continue;
                }
                let n = (p - c) / g;
                if n.dot(e.out) <= 0.0 {
                    continue;
                }
                let mut spec = 0.0;
                if g > r {
                    let reach = closing(&self.vehicles[o], [e.i, e.j], s, vx, [h, h], 0.0, n) * STEP;
                    if g - r >= reach {
                        continue;
                    }
                    spec = g - r;
                }
                let beyond = e.opp.iter().flatten().all(|&oc| {
                    let f = e.p + d * ((oc - e.p).dot(d) / ll);
                    (p - f).dot(f - oc) >= 0.0
                });
                if !beyond {
                    continue;
                }
                out.push(EdgeHit { a: [e.i, e.j], b: [h, h], s, t: 0.0, n, p: (r - g).max(0.0).min(r), spec, behind: 0.0 });
            }
        }
        out
    }

    fn edge_apply(&mut self, a: usize, b: usize, hits: &[EdgeHit]) {
        let two_body = self.vehicles[a].two_body_contact || self.vehicles[b].two_body_contact;
        let dissipative = self.vehicles[a].dissipative_contact || self.vehicles[b].dissipative_contact;
        // Each node's share of the pass's edge hits (its sub-body count).
        let mut ka: HashMap<u32, u32> = HashMap::new();
        let mut kb: HashMap<u32, u32> = HashMap::new();
        for x in hits {
            for (i, w) in x.a.iter().zip([1.0 - x.s, x.s]) {
                if w > 0.0 {
                    *ka.entry(*i).or_default() += 1;
                }
            }
            for (i, w) in x.b.iter().zip([1.0 - x.t, x.t]) {
                if w > 0.0 {
                    *kb.entry(*i).or_default() += 1;
                }
            }
        }
        let im = |n: &Node, k: u32| if n.has(NODE_IMMOVABLE) { 0.0 } else { k as f32 / n.mass };
        let acc = |f: Vec3, n: &Node| if n.has(NODE_IMMOVABLE) { Vec3::ZERO } else { f / n.mass };
        let force = |v: usize, i: u32| -> Vec3 {
            self.pass_force
                .get(v)
                .and_then(|p| p.get(i as usize))
                .copied()
                .unwrap_or_else(|| self.vehicles[v].force_now(i as usize))
        };
        let mut apply: Vec<(Vec3, Vec3, f32, [f32; 4])> = Vec::with_capacity(hits.len());
        for x in hits {
            let (va, vb) = (&self.vehicles[a], &self.vehicles[b]);
            let na = x.a.map(|i| va.nodes[i as usize]);
            let nb = x.b.map(|i| vb.nodes[i as usize]);
            let wa = [1.0 - x.s, x.s];
            let wb = [1.0 - x.t, x.t];
            let vel = (nb[0].vel * wb[0] + nb[1].vel * wb[1]) - (na[0].vel * wa[0] + na[1].vel * wa[1]) + x.n * (x.spec / STEP);
            let ims = if two_body {
                [
                    im(&na[0], ka.get(&x.a[0]).copied().unwrap_or(1)),
                    im(&na[1], ka.get(&x.a[1]).copied().unwrap_or(1)),
                    im(&nb[0], kb.get(&x.b[0]).copied().unwrap_or(1)),
                    im(&nb[1], kb.get(&x.b[1]).copied().unwrap_or(1)),
                ]
            } else {
                [im(&na[0], 1), im(&na[1], 1), im(&nb[0], 1), im(&nb[1], 1)]
            };
            let inv = wa[0] * wa[0] * ims[0] + wa[1] * wa[1] * ims[1] + wb[0] * wb[0] * ims[2] + wb[1] * wb[1] * ims[3];
            if inv <= 0.0 {
                apply.push((Vec3::ZERO, x.n, 0.0, ims));
                continue;
            }
            let m = 1.0 / inv;
            let rel = (acc(force(b, x.b[0]), &nb[0]) * wb[0] + acc(force(b, x.b[1]), &nb[1]) * wb[1])
                - (acc(force(a, x.a[0]), &na[0]) * wa[0] + acc(force(a, x.a[1]), &na[1]) * wa[1]);
            let fh = rel * m;
            let friction = nb[0].friction.min(nb[1].friction);
            let (f, budget) = contact_response(
                fh,
                vel,
                m,
                x.n,
                STEP,
                &ContactSurface::BOX.exact_if(vb.exact_friction).dissipative_if(dissipative),
                x.p,
                friction,
            );
            let mut delta = if f != Vec3::ZERO && dissipative { shift_depth(x.p, budget, fh.dot(x.n)) } else { 0.0 };
            let rate = crate::explicit_contact::push_out_rate();
            if rate > 0.0 && x.behind > 0.0 {
                delta = delta.max((rate * STEP).min(x.behind));
            }
            apply.push((f, x.n, delta * m, ims));
        }
        for (x, (f, n, q, ims)) in hits.iter().zip(apply) {
            if f == Vec3::ZERO && q <= 0.0 {
                continue;
            }
            let wa = [1.0 - x.s, x.s];
            let wb = [1.0 - x.t, x.t];
            if let Some(l) = self.ledger.as_mut() {
                let fnorm = n * f.dot(n);
                let ft = f - fnorm;
                for (v, i, w) in [(b, x.b[0], wb[0]), (b, x.b[1], wb[1]), (a, x.a[0], -wa[0]), (a, x.a[1], -wa[1])] {
                    l.vc[v][0][i as usize] += fnorm * w;
                    l.vc[v][1][i as usize] += ft * w;
                }
            }
            for k in 0..2 {
                self.vehicles[b].add_contact(x.b[k] as usize, f * wb[k]);
                self.vehicles[a].add_contact(x.a[k] as usize, -(f * wa[k]));
            }
            if q > 0.0 {
                for k in 0..2 {
                    self.vehicles[b].add_shift(x.b[k] as usize, n * (q * wb[k] * ims[2 + k]));
                    self.vehicles[a].add_shift(x.a[k] as usize, n * -(q * wa[k] * ims[k]));
                }
            }
        }
    }
}
