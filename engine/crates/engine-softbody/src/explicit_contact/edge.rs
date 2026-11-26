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
use crate::explicit_beams::{ExplicitBeams, STEP};
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
/// Node against edge with edge contact (on; `EDGE_VERTEX=0`, a measurement: off).
/// An edge-contact pair's contact points at their paint (contact offsets: paint meets
/// paint). Under measurement, `RUBBLE_PAINT_POINTS=1`: better with edge contact over six
/// head-ons (edges through 73 -> 48, overlap 30 -> 25 L); off until the GPU's node turns
/// place the points so too.
pub(crate) fn paint_points() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUBBLE_PAINT_POINTS").is_ok_and(|v| v == "1"))
}

fn vertex_edge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("EDGE_VERTEX").is_ok_and(|v| v == "0"))
}
/// Car-to-car edge contact over the outer skin's edges only (`EDGE_OUTER_ONLY=1`, a
/// measurement; default: every edge, an inside face's taking its side from the partner).
fn edge_outer_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EDGE_OUTER_ONLY").is_ok_and(|v| v == "1"))
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
#[derive(Clone, Debug)]
pub(super) struct SkinEdge {
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
    /// `out` is from the outer skin (its winding is checked); false: an edge of faces inside
    /// the car only, whose side the partner's edge decides.
    outer: bool,
}

/// The vehicle's live skin edges with their outward normals (unit sum of the live
/// triangles' unit normals at each edge).
fn skin_edges(v: &ExplicitBeams, outer_only: bool) -> Vec<SkinEdge> {
    let off = contact_offset_vectors(v);
    let w = |i: u32| v.origin + v.nodes[i as usize].pos + off.get(i as usize).copied().unwrap_or(Vec3::ZERO);
    let mut by: HashMap<(u32, u32), (Vec3, Vec<(Vec3, Vec3)>, bool)> = HashMap::new();
    for (t, tri) in v.triangles.iter().enumerate() {
        if v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t) {
            continue;
        }
        let outer = v.tri_outer.get(t).copied().unwrap_or(true);
        if outer_only && !outer {
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
            let e = by.entry((x.min(y), x.max(y))).or_insert((Vec3::ZERO, Vec::new(), false));
            // The outward normal from the outer skin's faces when the edge has any.
            if outer && !e.2 {
                e.0 = Vec3::ZERO;
                e.1.clear();
                e.2 = true;
            }
            if outer == e.2 {
                e.0 += nn;
                e.1.push((corner[(k + 2) % 3], nn));
            }
        }
    }
    let mut e: Vec<SkinEdge> = by
        .into_iter()
        .filter_map(|((i, j), (s, faces, outer))| {
            let out = s.normalize_or_zero();
            let opp = [faces.first().map(|f| f.0), faces.get(1).map(|f| f.0)];
            // Convex: each face's opposite corner is behind the other face's plane.
            let convex = match faces.as_slice() {
                [(o0, n0), (o1, n1)] => n0.dot(*o1 - w(i)) <= 0.0 && n1.dot(*o0 - w(i)) <= 0.0,
                _ => true,
            };
            (out != Vec3::ZERO).then(|| SkinEdge { i, j, p: w(i), q: w(j), out, opp, convex, outer })
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

/// The speed-scaled band (a measurement, `SPEED_BAND=k`; 0: off; edge contact's passes only, not
/// adopted: docs/contact-design.md): a pair's band `r` widened by k
/// times its closing speed over a step, at most `SPEED_BAND_MAX` (m, default 0.075): thin at rest,
/// as thick as a fast pair needs to be stopped inside it.
pub(crate) fn edge_fade() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EDGE_FADE").is_ok_and(|v| v == "1"))
}

pub(crate) fn speed_band() -> (f32, f32) {
    static K: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        let g = |n: &str| std::env::var(n).ok().and_then(|v| v.parse().ok());
        (g("SPEED_BAND").unwrap_or(0.0), g("SPEED_BAND_MAX").unwrap_or(0.075))
    })
}
pub(crate) fn banded(r: f32, closing: f32) -> f32 {
    let (k, max) = speed_band();
    if k == 0.0 { r } else { r.max((r + k * closing * STEP).min(max)) }
}

fn edge_cone() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("EDGE_CONE").is_ok_and(|v| v == "1"))
}
/// Each face's in-plane direction across edge `e` (perpendicular to it, out of the face).
fn cone_dirs(e: &SkinEdge) -> Vec<Vec3> {
    let d = e.q - e.p;
    let ll = d.dot(d).max(1e-12);
    e.opp.iter().flatten().map(|&oc| { let f = e.p + d * ((oc - e.p).dot(d) / ll); (f - oc).normalize_or_zero() }).collect()
}
/// How far `n` is inside the edge's normal cone: min over its faces of n·u (u across the
/// edge out of the face); negative: outside, past that face's normal.
fn cone_margin(e: &SkinEdge, n: Vec3) -> f32 {
    cone_dirs(e).into_iter().map(|u| n.dot(u)).fold(f32::INFINITY, f32::min)
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
    /// At the contact pass's start: every vehicle's live skin edges (edge contact on
    /// for any vehicle; none otherwise).
    pub(super) fn edge_begin(&mut self) {
        let nv = self.vehicles.len();
        self.skin.clear();
        if !self.vehicles.iter().any(|v| v.edge_contact) {
            return;
        }
        for a in 0..nv {
            let skip = self.skip.get(a).copied().unwrap_or(false);
            self.skin.push((!skip).then(|| skin_edges(&self.vehicles[a], edge_outer_only())));
        }
    }

    /// Every owner's turn of its edge hits, after every owner's node turn, in the same
    /// order (`turns`: each owner with its partners).
    pub(super) fn edge_turns(&mut self, edge_turns: Vec<(usize, Vec<usize>)>) {
        // Edge against edge and node against edge (`edge_contact`): each owner's turn of
        // its edge hits after every owner's node turn, in the same order (the GPU's edge
        // pass, a dispatch of its own after the contact pass: contact-design.md).
        // (The edge turns' first passes start from the forces after every node turn: the
        // GPU's edge dispatch snapshots them once at its start.)
        if !edge_turns.is_empty() && self.vehicles.iter().any(|v| v.two_body_contact) {
            for (pf, v) in self.pass_force.iter_mut().zip(&self.vehicles) {
                pf.clear();
                pf.extend(v.forces_now());
            }
        }
        for (a, partners) in edge_turns {
            let mut hits = Vec::new();
            self.edge_turn(a, &partners, &mut hits);
            if !hits.is_empty() {
                self.turn_hits(a, hits);
            }
        }
    }

    /// Owner `a`'s turn: its edges against each partner's (edge against edge, each
    /// unordered pair once, in the lower index's turn) and against each partner's nodes
    /// (node against edge, `vertex_edge()`), as hits of the owner's list (the contacter
    /// the partner's edge or node, the owner's edge in `tri` as [i, j, j]).
    pub(super) fn edge_turn(&mut self, a: usize, partners: &[usize], hits: &mut Vec<super::Hit>) {
        if self.skin.is_empty() {
            return;
        }
        let skin = std::mem::take(&mut self.skin);
        for &b in partners {
            if !(self.vehicles[a].edge_contact || self.vehicles[b].edge_contact) {
                continue;
            }
            let (Some(ea), Some(eb)) = (&skin[a], &skin[b]) else {
                continue;
            };
            let mut found = Vec::new();
            if a < b {
                found.extend(self.edge_hits(a, b, ea, eb).into_iter().map(|x| (x, 1u8)));
            }
            if vertex_edge() {
                found.extend(self.vertex_edge_hits(a, b, ea).into_iter().map(|x| (x, 2u8)));
            }
            for (x, kind) in found {
                hits.push(super::Hit {
                    tri: [x.a[0], x.a[1], x.a[1]],
                    w: [x.s, 0.0, 1.0 - x.s],
                    b,
                    h: x.b[0],
                    n: x.n,
                    p: x.p,
                    spec: x.spec,
                    behind: x.behind,
                    h2: x.b[1],
                    t: x.t,
                    kind,
                });
            }
        }
        self.skin = skin;
    }

    fn edge_hits(&mut self, a: usize, b: usize, ea: &[SkinEdge], eb: &[SkinEdge]) -> Vec<EdgeHit> {
        let r = self.vv_range.unwrap_or(self.vehicles[a].vv_band().max(self.vehicles[b].vv_band()));
        let travel = |v: &ExplicitBeams| v.nodes.iter().map(|n| n.vel.length()).fold(0.0f32, f32::max) * STEP;
        let pad = r.max(speed_band().1 * (speed_band().0 > 0.0) as u8 as f32) + edge_depth() + travel(&self.vehicles[a]) + travel(&self.vehicles[b]);
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
                let note = |w: &mut Self, code: u8, g: f32| {
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
                // The side from A's outer edge, or from B's when A's is an inside face's
                // (neither outer: no side, no contact).
                match (e.outer, f.outer) {
                    (false, false) => continue,
                    (false, true) => {
                        if n.dot(f.out) > 0.0 {
                            n = -n;
                        }
                    }
                    _ => {
                        if n.dot(e.out) < 0.0 {
                            n = -n;
                        }
                    }
                }
                // B's skin must face A here (B's outward against n), else no contact.
                if facing_check() && e.outer && f.outer && n.dot(f.out) >= 0.0 {
                    // (Counted only when close: the pairs the check turns away.)
                    let g = n.dot((f.p + (f.q - f.p) * t) - (e.p + (e.q - e.p) * s));
                    if g <= r && g >= -edge_depth() {
                        self.edge_stats[3] += 1;
                    }
                    note(self, 2, g);
                    continue;
                }
                // Both outer: an edge-edge contact only where the edges are the touching
                // features, the normal inside each edge's normal cone (between its faces'
                // normals; empty at a concave edge). Elsewhere a face or a node is nearer and
                // that contact is theirs (node against face, or this face's own edges); a push
                // along e x f there points where neither panel faces (docs/contact-design.md;
                // `EDGE_CONE=1`, a measurement, CPU only: it leaves crossed cars inside each other).
                let gap = n.dot((f.p + (f.q - f.p) * t) - (e.p + (e.q - e.p) * s));
                if edge_cone() && e.outer && f.outer && gap <= r && gap >= -edge_depth()
                    && (!e.convex || !f.convex || cone_margin(e, n) < 0.0 || cone_margin(f, -n) < 0.0)
                {
                    continue;
                }
                let ca = e.p + (e.q - e.p) * s;
                let cb = f.p + (f.q - f.p) * t;
                let g = n.dot(cb - ca);
                let r = if speed_band().0 > 0.0 { banded(r, closing(&self.vehicles[a], [e.i, e.j], s, &self.vehicles[b], [f.i, f.j], t, n)) } else { r };
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
                // (EDGE_FADE=1, a measurement: a pair crossed past the band pushes less the deeper it is, to
                // nothing at EDGE_DEPTH, instead of full until it and then none.)
                let p = if edge_fade() && g < -r { r * (1.0 - (-g - r) / (edge_depth() - r).max(1e-6)).max(0.0) } else { (r - g).max(0.0).min(r) };
                out.push(EdgeHit { a: [e.i, e.j], b: [f.i, f.j], s, t, n, p, spec, behind: (-g).max(0.0) });
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
        let r = self.vv_range.unwrap_or(self.vehicles[o].vv_band().max(self.vehicles[x].vv_band()));
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
            if !e.convex || !e.outer {
                continue;
            }
            let pad = r.max(speed_band().1 * (speed_band().0 > 0.0) as u8 as f32) + travel_o + travel_x;
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
        let offx = contact_offset_vectors(vx);
        let mut out = Vec::new();
        for &h in &vx.contact_points {
            if vx.is_retired(h as usize) {
                continue;
            }
            // (At its paint, as the skins' edges.)
            let p = vx.origin + vx.nodes[h as usize].pos + if paint_points() { offx.get(h as usize).copied().unwrap_or(Vec3::ZERO) } else { Vec3::ZERO };
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
                if g < 1e-6 || g > r.max(speed_band().1 * (speed_band().0 > 0.0) as u8 as f32) + travel_o + travel_x {
                    continue;
                }
                let n = (p - c) / g;
                if n.dot(e.out) <= 0.0 {
                    continue;
                }
                let r = if speed_band().0 > 0.0 { banded(r, closing(&self.vehicles[o], [e.i, e.j], s, vx, [h, h], 0.0, n)) } else { r };
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
}

/// Edge against edge within one car (self contact, `edge_contact`): pairs of the outer
/// skin's live edges sharing no node, not both in one self group, not within the contact
/// range at rest (the first pass records those, as node contact's spawn exclusion), by the
/// car-to-car rule (closest points inside both, normal e × f out of the first edge's side
/// and into the second's, gap −EDGE_DEPTH..r). Each pair once, as hits of the self pass.
pub(crate) fn self_edge_hits(v: &mut ExplicitBeams) -> Vec<crate::explicit_beams::SelfHit> {
    let edges = skin_edges(v, true);
    let r = v.collision_range;
    let group = |i: u32| v.self_groups.get(i as usize).copied().unwrap_or(0);
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    let pad = r + EDGE_DEPTH;
    for (k, e) in edges.iter().enumerate() {
        let (lo, hi) = (cell(e.p.min(e.q) - Vec3::splat(pad)), cell(e.p.max(e.q) + Vec3::splat(pad)));
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    grid.entry((x, y, z)).or_default().push(k as u32);
                }
            }
        }
    }
    let first = v.self_edge_excl.is_none();
    let mut excl = v.self_edge_excl.take().unwrap_or_default();
    let mut out = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    for (kf, f) in edges.iter().enumerate() {
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
        for &ke in &seen {
            if ke as usize <= kf {
                continue;
            }
            let e = &edges[ke as usize];
            if e.i == f.i || e.i == f.j || e.j == f.i || e.j == f.j {
                continue;
            }
            let g = group(e.i);
            if g != 0 && group(e.j) == g && group(f.i) == g && group(f.j) == g {
                continue;
            }
            let key = (e.i, e.j, f.i, f.j);
            let (s, t) = engine_schema::feature_query::closest_seg_seg(e.p, e.q, f.p, f.q);
            let (ca, cb) = (e.p + (e.q - e.p) * s, f.p + (f.q - f.p) * t);
            if first {
                if ca.distance(cb) < r + 0.005 {
                    excl.insert(key);
                }
                continue;
            }
            if excl.contains(&key) || s <= END || s >= 1.0 - END || t <= END || t >= 1.0 - END {
                continue;
            }
            let mut n = (e.q - e.p).cross(f.q - f.p);
            let len = n.length();
            if len < 1e-8 {
                continue;
            }
            n /= len;
            if n.dot(e.out) < 0.0 {
                n = -n;
            }
            if n.dot(f.out) >= 0.0 {
                continue;
            }
            let gap = n.dot(cb - ca);
            if gap > r || gap < -EDGE_DEPTH {
                continue;
            }
            out.push(crate::explicit_beams::SelfHit {
                hn: f.i,
                tri: [e.i, e.j, e.j],
                w: [s, 0.0, 1.0 - s],
                depth: (r - gap).min(r),
                n,
                hn2: f.j,
                t,
                kind: 1,
            });
        }
    }
    v.self_edge_excl = Some(excl);
    out
}

/// One skin edge's topology for the GPU's table: its nodes (i < j, the vehicle's own
/// indices) and its triangles in the triangles' order, each with whether it is on the
/// outer skin (anti-clip-only triangles left out, as `skin_edges` leaves them; which
/// triangles are live, which nodes retired and the normals are the step's).
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeTopology {
    pub i: u32,
    pub j: u32,
    pub tris: Vec<(u32, bool)>,
}

/// Every skin edge of `v` in `skin_edges`' order (by (i, j)).
pub fn skin_edge_topology(v: &ExplicitBeams) -> Vec<EdgeTopology> {
    let mut by: std::collections::BTreeMap<(u32, u32), Vec<(u32, bool)>> = Default::default();
    for (t, tri) in v.triangles.iter().enumerate() {
        if v.tri_anti_only(t) {
            continue;
        }
        let outer = v.tri_outer.get(t).copied().unwrap_or(true);
        let n = tri.nodes;
        for k in 0..3 {
            let (x, y) = (n[k], n[(k + 1) % 3]);
            by.entry((x.min(y), x.max(y))).or_default().push((t as u32, outer));
        }
    }
    by.into_iter().map(|((i, j), tris)| EdgeTopology { i, j, tris }).collect()
}

/// Vehicle `v`'s contact offsets as vectors (`ExplicitBeams::contact_offset` along each
/// node's vertex normal: the unit sum of its live outer triangles' unit normals); empty
/// without offsets.
/// Each node's position on the contact surface vehicle contact sees (relative to the
/// origin): the node pushed out by its contact offset (the paint), or the node itself.
pub fn contact_surface_positions(v: &ExplicitBeams) -> Vec<Vec3> {
    let off = contact_offset_vectors(v);
    v.nodes.iter().enumerate().map(|(i, n)| n.pos + off.get(i).copied().unwrap_or(Vec3::ZERO)).collect()
}

pub(crate) fn contact_offset_vectors(v: &ExplicitBeams) -> Vec<Vec3> {
    if v.contact_offset.is_empty() {
        return Vec::new();
    }
    let mut n = vec![Vec3::ZERO; v.nodes.len()];
    for (t, tri) in v.triangles.iter().enumerate() {
        if v.tri_dead.get(t).copied().unwrap_or(false) || v.tri_anti_only(t) || !v.tri_outer.get(t).copied().unwrap_or(true) {
            continue;
        }
        let [a, b, c] = tri.nodes.map(|i| v.nodes[i as usize].pos);
        let nn = (b - a).cross(c - a).normalize_or_zero();
        for &i in &tri.nodes {
            n[i as usize] += nn;
        }
    }
    n.iter()
        .enumerate()
        .map(|(i, s)| s.normalize_or_zero() * v.contact_offset.get(i).copied().unwrap_or(0.0))
        .collect()
}

impl ExplicitBeams {
    /// The band against other vehicles (`vehicle_contact_range`, or `collision_range`, which self
    /// and surface contact keep; an edge pair's is the larger of the two's).
    pub fn vv_band(&self) -> f32 {
        self.vehicle_contact_range.unwrap_or(self.collision_range)
    }
}
