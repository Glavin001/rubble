//! The closed-skin side test (jbeam-semantics-spec.md §10.3, the T-bone leak):
//! a measurement of which side of a struck car's skin a point is on, used by the
//! crash diagnostics. Not in the solver: tried as vehicle contact's side test and
//! closest-feature contact and measured worse there (spec §10.3).
//!
//! RoR's contact decides which side of a struck triangle a contact point belongs
//! on by a vote of the point and its beam neighbours (spec §5.3 of the
//! vehicle-contact spec). On a closed collision skin the side is a fact of
//! geometry instead: a point is inside the skin when it is behind the skin's
//! closest feature (face, edge or vertex) by that feature's angle-weighted
//! pseudo-normal (Bærentzen and Aanæs, "Signed distance computation using the
//! angle weighted pseudonormal", IEEE TVCG 11(3), 2005), which is exact for a
//! closed, oriented 2-manifold.
//!
//! At spawn ([`SkinTopology::build`]) the collision triangles are split into
//! edge-connected components; a component is *closed* when every edge of it is
//! shared by exactly two of its triangles, its windings can be made consistent,
//! and every vertex's triangles form one fan. Its triangles are then oriented
//! outward (positive enclosed volume). Everything else is *open*: its features
//! have no side, and a point whose closest feature is open keeps RoR's vote. A
//! dead (broken, §18) triangle opens its edges and vertices while it is dead.
//!
//! Every operation is written out (one rounding each, no fused multiply-add) so a
//! GPU port could reproduce it bit for bit.

use glam::Vec3;

/// No triangle across an edge.
pub const OPEN: u32 = u32::MAX;

/// The closest-feature code in a side key's two low bits.
pub const SIDE_OUTSIDE: u32 = 0;
pub const SIDE_INSIDE: u32 = 1;
pub const SIDE_UNKNOWN: u32 = 2;
/// The key of a contact point with no candidate triangle.
pub const SIDE_NONE: u32 = u32::MAX;

/// A vehicle's collision-skin topology for the side test.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinTopology {
    /// Per triangle, per edge k (its nodes k → k + 1 mod 3): the triangle across
    /// it in a closed component, else [`OPEN`].
    pub across: Vec<[u32; 3]>,
    /// Per triangle: +1 its winding (n = (p₁ − p₀) × (p₂ − p₀)) points out of its
    /// closed component, −1 into it, 0 its component is open.
    pub orient: Vec<f32>,
    /// Per node, its closed fan: `fan[fan_start[i]..fan_start[i + 1]]`, the
    /// triangles of a closed component around it in triangle order (empty: the
    /// node is on no closed component's triangle).
    pub fan_start: Vec<u32>,
    pub fan: Vec<u32>,
    /// Closed components found.
    pub closed_components: u32,
}

impl SkinTopology {
    /// The topology of triangles `tris` over `nodes` nodes at positions `pos`
    /// (spawn: the components' orientation is their enclosed volume's sign).
    pub fn build(tris: &[[u32; 3]], pos: &[Vec3]) -> Self {
        let nt = tris.len();
        let n = pos.len();
        // Undirected edges → the (triangle, edge) uses.
        let mut uses: std::collections::HashMap<(u32, u32), Vec<(u32, u8)>> =
            std::collections::HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                uses.entry((a.min(b), a.max(b)))
                    .or_default()
                    .push((t as u32, k as u8));
            }
        }
        let degenerate = |t: &[u32; 3]| t[0] == t[1] || t[1] == t[2] || t[2] == t[0];
        let mut across = vec![[OPEN; 3]; nt];
        let mut manifold = vec![true; nt];
        for (t, tri) in tris.iter().enumerate() {
            if degenerate(tri) {
                manifold[t] = false;
                continue;
            }
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let u = &uses[&(a.min(b), a.max(b))];
                if u.len() == 2 {
                    let o = if u[0].0 == t as u32 { u[1] } else { u[0] };
                    if o.0 == t as u32 || degenerate(&tris[o.0 as usize]) {
                        manifold[t] = false;
                    } else {
                        across[t][k] = o.0;
                    }
                } else {
                    manifold[t] = false;
                }
            }
        }
        // Components over manifold edges, with relative windings.
        let mut comp = vec![u32::MAX; nt];
        let mut flip = vec![1.0f32; nt];
        let mut orient = vec![0.0f32; nt];
        let mut closed_components = 0;
        let mut stack = Vec::new();
        for seed in 0..nt {
            if comp[seed] != u32::MAX || degenerate(&tris[seed]) {
                continue;
            }
            let id = seed as u32;
            comp[seed] = id;
            flip[seed] = 1.0;
            stack.push(seed);
            let mut members = Vec::new();
            let mut closed = true;
            while let Some(t) = stack.pop() {
                members.push(t);
                closed &= manifold[t];
                for k in 0..3 {
                    let o = across[t][k];
                    if o == OPEN {
                        continue;
                    }
                    let o = o as usize;
                    // Consistent windings traverse the shared edge oppositely.
                    let (a, b) = (tris[t][k], tris[t][(k + 1) % 3]);
                    let ko = (0..3)
                        .find(|&j| {
                            let (c, d) = (tris[o][j], tris[o][(j + 1) % 3]);
                            (c == a && d == b) || (c == b && d == a)
                        })
                        .unwrap();
                    let same = tris[o][ko] == a;
                    let want = if same { -flip[t] } else { flip[t] };
                    if comp[o] == u32::MAX {
                        comp[o] = id;
                        flip[o] = want;
                        stack.push(o);
                    } else if flip[o] != want {
                        closed = false; // not orientable
                    }
                }
            }
            if !closed {
                continue;
            }
            // Its enclosed volume (spawn positions) sets the outward sign.
            let mut vol = 0.0f64;
            for &t in &members {
                let [a, b, c] = tris[t].map(|i| pos[i as usize].as_dvec3());
                vol += f64::from(flip[t]) * a.dot(b.cross(c));
            }
            if vol == 0.0 {
                continue;
            }
            let s = if vol > 0.0 { 1.0 } else { -1.0 };
            for &t in &members {
                orient[t] = s * flip[t];
            }
            closed_components += 1;
        }
        for t in 0..nt {
            if orient[t] == 0.0 {
                across[t] = [OPEN; 3];
            }
        }
        // Fans: a node's closed triangles, kept only when they are one cycle
        // round it (a manifold vertex).
        let mut inc: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (t, tri) in tris.iter().enumerate() {
            if orient[t] != 0.0 {
                for &i in tri {
                    inc[i as usize].push(t as u32);
                }
            }
        }
        let mut fan_start = Vec::with_capacity(n + 1);
        let mut fan = Vec::new();
        fan_start.push(0);
        for (i, list) in inc.iter().enumerate() {
            if !list.is_empty() && one_fan(i as u32, list, tris, &across) {
                fan.extend_from_slice(list);
            }
            fan_start.push(fan.len() as u32);
        }
        Self {
            across,
            orient,
            fan_start,
            fan,
            closed_components,
        }
    }

    /// Whether any component is closed.
    pub fn any_closed(&self) -> bool {
        self.closed_components > 0
    }
}

/// Whether node `v`'s triangles `list` (all closed) form a single cycle round it.
fn one_fan(v: u32, list: &[u32], tris: &[[u32; 3]], across: &[[u32; 3]]) -> bool {
    let start = list[0];
    let mut seen = 1usize;
    let (mut prev, mut t) = (OPEN, start);
    // Each triangle has two edges at v (k: v → next, k + 2: prev → v); leave by the
    // one not arrived through.
    for _ in 0..list.len() + 1 {
        let tri = tris[t as usize];
        let k = (0..3).find(|&k| tri[k] == v).unwrap();
        let (e1, e2) = (across[t as usize][k], across[t as usize][(k + 2) % 3]);
        let o = if e1 != prev || prev == OPEN { e1 } else { e2 };
        if o == OPEN || !list.contains(&o) {
            return false;
        }
        if o == start {
            return seen == list.len();
        }
        seen += 1;
        (prev, t) = (t, o);
    }
    false
}

// --- The query, in the kernel's operation order -------------------------------

#[inline]
fn dot(a: Vec3, b: Vec3) -> f32 {
    (a.x * b.x + a.y * b.y) + a.z * b.z
}

#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - b.y * a.z,
        a.z * b.x - b.z * a.x,
        a.x * b.y - b.x * a.y,
    )
}

/// The unit normal of triangle (p₀, p₁, p₂) times `s` (the zero vector when the
/// cross product is shorter than 1e-20).
#[inline]
fn unit_normal(p0: Vec3, p1: Vec3, p2: Vec3, s: f32) -> Vec3 {
    let c = cross(p1 - p0, p2 - p0);
    let l = dot(c, c).sqrt();
    if l > 1e-20 {
        c * (s / l)
    } else {
        Vec3::ZERO
    }
}

/// atan2(y, x) for y ≥ 0 (an angle in [0, π]): a degree-11 odd minimax polynomial
/// on [0, 1] (absolute error below 1e-5 rad), folded by octant. Written out so the
/// kernel computes the same bits.
pub fn angle_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let (lo, hi) = if y > ax { (ax, y) } else { (y, ax) };
    if hi <= 0.0 {
        return 0.0;
    }
    let a = lo / hi;
    let s = a * a;
    let mut r = -0.011_721_2_f32;
    r = r * s + 0.052_653_32;
    r = r * s - 0.116_432_87;
    r = r * s + 0.193_543_46;
    r = r * s - 0.332_623_47;
    r = r * s + 0.999_977_26;
    r *= a;
    if y > ax {
        r = std::f32::consts::FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = std::f32::consts::PI - r;
    }
    r
}

/// The interior angle at `v` between edges to `a` and `b`.
#[inline]
fn corner_angle(v: Vec3, a: Vec3, b: Vec3) -> f32 {
    let e1 = a - v;
    let e2 = b - v;
    let c = cross(e1, e2);
    angle_atan2(dot(c, c).sqrt(), dot(e1, e2))
}

/// The closest feature of triangle (a, b, c) to `p` (Ericson, Real-Time Collision
/// Detection §5.1.5): (closest point, feature, weights) with feature 0 the face,
/// 1/2/3 the edges ab/bc/ca (edge k + 1 joins nodes k and k + 1), 4/5/6 the vertices
/// a/b/c, and the closest point's weights on (b, c, a) — the contact's (α, β, γ).
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, u32, [f32; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, 4, [0.0, 0.0, 1.0]);
    }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, 5, [1.0, 0.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, 1, [v, 0.0, 1.0 - v]);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, 6, [0.0, 1.0, 0.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, 3, [0.0, w, 1.0 - w]);
    }
    let va = d3 * d6 - d5 * d4;
    let e43 = d4 - d3;
    let e56 = d5 - d6;
    if va <= 0.0 && e43 >= 0.0 && e56 >= 0.0 {
        let w = e43 / (e43 + e56);
        return (b + (c - b) * w, 2, [1.0 - w, w, 0.0]);
    }
    let denom = 1.0 / ((va + vb) + vc);
    let v = vb * denom;
    let w = vc * denom;
    ((a + ab * v) + ac * w, 0, [v, w, (1.0 - v) - w])
}

/// A side key's squared distance (its low 14 bits cleared: a relative resolution
/// of 2⁻⁹ of d², about 0.1 % of d).
pub fn side_d2(key: u32) -> f32 {
    f32::from_bits(key & !KEY_LOW)
}

/// A side key's triangle (the owner's local index).
pub fn side_triangle(key: u32) -> usize {
    ((key >> 2) & 0xFFF) as usize
}

/// The low bits of a side key holding its triangle (12 bits) and side (2 bits).
pub const KEY_LOW: u32 = 0x3FFF;

/// Collision triangles per vehicle a side key can name.
pub const KEY_TRIANGLES: usize = 4096;

/// Point `p`'s side key for triangle `t` of a skin (world positions `pos`; `dead`
/// its triangles' dead flags): the squared distance's bits with their low 14 bits
/// replaced by the triangle (12 bits) and the side of its closest feature (2 bits:
/// [`SIDE_OUTSIDE`], [`SIDE_INSIDE`], or [`SIDE_UNKNOWN`] when the feature is open).
/// The smallest key over a point's candidate triangles is the skin's closest
/// feature (ties within the distance's resolution: the lowest triangle).
pub fn side_key(
    p: Vec3,
    t: usize,
    tris: &[[u32; 3]],
    pos: &[Vec3],
    dead: &[bool],
    skin: &SkinTopology,
) -> u32 {
    let tri = tris[t];
    let [a, b, c] = tri.map(|i| pos[i as usize]);
    let (q, feature, _) = closest_on_triangle(p, a, b, c);
    let d = p - q;
    let d2 = dot(d, d);
    let s = skin.orient[t];
    let alive = |x: u32| !dead.get(x as usize).copied().unwrap_or(false);
    let code = if s == 0.0 {
        SIDE_UNKNOWN
    } else {
        let normal = match feature {
            0 => Some(cross(b - a, c - a) * s),
            1..=3 => {
                let k = (feature - 1) as usize;
                let o = skin.across[t][k];
                (o != OPEN && alive(o)).then(|| {
                    let [oa, ob, oc] = tris[o as usize].map(|i| pos[i as usize]);
                    unit_normal(a, b, c, s) + unit_normal(oa, ob, oc, skin.orient[o as usize])
                })
            }
            _ => {
                let v = tri[(feature - 4) as usize];
                let (f0, f1) = (
                    skin.fan_start[v as usize] as usize,
                    skin.fan_start[v as usize + 1] as usize,
                );
                let fan = &skin.fan[f0..f1];
                (!fan.is_empty() && fan.iter().all(|&o| alive(o))).then(|| {
                    let mut nsum = Vec3::ZERO;
                    for &o in fan {
                        let ot = tris[o as usize];
                        let k = (0..3).find(|&k| ot[k] == v).unwrap_or(0);
                        let pv = pos[v as usize];
                        let pn = pos[ot[(k + 1) % 3] as usize];
                        let pp = pos[ot[(k + 2) % 3] as usize];
                        let [oa, ob, oc] = ot.map(|i| pos[i as usize]);
                        let n = unit_normal(oa, ob, oc, skin.orient[o as usize]);
                        nsum += n * corner_angle(pv, pn, pp);
                    }
                    nsum
                })
            }
        };
        match normal {
            None => SIDE_UNKNOWN,
            Some(nrm) => {
                if dot(d, nrm) < 0.0 {
                    SIDE_INSIDE
                } else {
                    SIDE_OUTSIDE
                }
            }
        }
    };
    (d2.to_bits() & !KEY_LOW) | ((t as u32 & 0xFFF) << 2) | code
}

// --- The leak metric: crossings of the live skin --------------------------------

/// The T-bone leak metric (jbeam-semantics-spec.md §10.3, fifth pass;
/// `vehicle-acceptance-criteria.md` A1): a tracked point (a striking car's
/// node) has *leaked* when it has penetrated a struck car's live skin: when its
/// net count of crossings of the skin's live triangles, inward (front to back by
/// the closed component's outward orientation) less outward, is not zero (−1: it
/// went in through an opening and out through the skin, or through a triangle the
/// crush has turned over). Followed step by step from a spawn where every point is
/// outside. A step's crossings are the integer jump of the point's winding
/// number: the change of each nearby triangle's solid angle seen from the point
/// (about 4π when the point passes through it, small otherwise), summed and
/// rounded (the sum of a shared edge's two triangles counts a pass at the edge
/// once), and put to the dead or the live triangles by their larger share. A crossing of a dead (broken, §18)
/// triangle is not counted: a point that passes through the opening a door leaves
/// when its shut-line triangles die is in a real opening, not through the skin.
/// With nothing dead the net count is the winding number itself, so it agrees with
/// the old metric (|winding| > 0.5).
///
/// A measurement for the tests (f64 throughout), not in the solver.
#[derive(Clone, Debug)]
pub struct LeakTracker {
    /// Per point: inward minus outward crossings of live triangles.
    pub net: Vec<i32>,
    /// Crossings of live triangles counted, inward and outward.
    pub inward: u64,
    pub outward: u64,
    /// Crossings of dead triangles seen (not counted).
    pub through_dead: u64,
    /// The last update's crossings: (point, triangle, +1 inward / −1 outward / 0
    /// through a dead triangle).
    pub events: Vec<(u32, u32, i8)>,
    /// Points' steps and triangles' sweeps not tracked: non-finite or wider than
    /// [`LEAK_SPAN`] (a diverging vehicle).
    pub skipped: u64,
    prev_pts: Vec<Vec3>,
    prev_skin: Vec<Vec3>,
    grid: std::collections::HashMap<(i32, i32, i32), Vec<u32>>,
    stamp: Vec<u32>,
    epoch: u32,
}

/// The broad phase's cell (m).
const LEAK_CELL: f32 = 0.25;
/// Each triangle's box is grown by this much (m) so that a point passing close by
/// an edge sees both triangles' solid angles change (their sum is the crossing).
const LEAK_MARGIN: f32 = 0.05;

/// Whether a box is finite and spans at most [`LEAK_SPAN`] (a step of a diverging
/// vehicle is not tracked, and is counted in `skipped`).
fn wild(lo: Vec3, hi: Vec3) -> bool {
    lo.is_finite() && hi.is_finite() && (hi - lo).max_element() <= LEAK_SPAN
}

/// The largest box a point's step or a triangle's sweep may span (m).
const LEAK_SPAN: f32 = 4.0;

fn leak_cell(p: Vec3) -> (i32, i32, i32) {
    let q = p / LEAK_CELL;
    (q.x.floor() as i32, q.y.floor() as i32, q.z.floor() as i32)
}

impl LeakTracker {
    /// A tracker of points `pts` against a skin at `skin_pos` (both world positions,
    /// every point outside).
    pub fn new(pts: &[Vec3], skin_pos: &[Vec3]) -> Self {
        Self {
            net: vec![0; pts.len()],
            inward: 0,
            outward: 0,
            through_dead: 0,
            events: Vec::new(),
            skipped: 0,
            prev_pts: pts.to_vec(),
            prev_skin: skin_pos.to_vec(),
            grid: Default::default(),
            stamp: Vec::new(),
            epoch: 0,
        }
    }

    /// One step: the crossings between the last positions and these. `tris` and
    /// `topo` are the skin's triangles and topology (its orientation; triangles of
    /// open components are not tracked), `dead` their dead flags now.
    pub fn update(
        &mut self,
        pts: &[Vec3],
        skin_pos: &[Vec3],
        tris: &[[u32; 3]],
        dead: &[bool],
        topo: &SkinTopology,
    ) {
        self.grid.clear();
        self.events.clear();
        self.stamp.resize(tris.len(), 0);
        for (t, tri) in tris.iter().enumerate() {
            if topo.orient[t] == 0.0 {
                continue;
            }
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for &i in tri {
                for p in [self.prev_skin[i as usize], skin_pos[i as usize]] {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            if !wild(lo, hi) {
                self.skipped += 1;
                continue;
            }
            let (a, b) = (
                leak_cell(lo - Vec3::splat(LEAK_MARGIN)),
                leak_cell(hi + Vec3::splat(LEAK_MARGIN)),
            );
            for x in a.0..=b.0 {
                for y in a.1..=b.1 {
                    for z in a.2..=b.2 {
                        self.grid.entry((x, y, z)).or_default().push(t as u32);
                    }
                }
            }
        }
        for (k, &p1) in pts.iter().enumerate() {
            let p0 = self.prev_pts[k];
            if !wild(p0.min(p1), p0.max(p1)) {
                self.skipped += 1;
                continue;
            }
            let (a, b) = (leak_cell(p0.min(p1)), leak_cell(p0.max(p1)));
            self.epoch = self.epoch.wrapping_add(1);
            // The change of the point's winding number over the nearby triangles,
            // live and dead apart, and the triangle that changed most.
            let (mut live, mut gone) = (0.0f64, 0.0f64);
            let mut top = (0.0f64, 0u32);
            for x in a.0..=b.0 {
                for y in a.1..=b.1 {
                    for z in a.2..=b.2 {
                        let Some(list) = self.grid.get(&(x, y, z)) else {
                            continue;
                        };
                        for &t in list {
                            let t = t as usize;
                            if self.stamp[t] == self.epoch {
                                continue;
                            }
                            self.stamp[t] = self.epoch;
                            let tri = tris[t];
                            let v0 = tri.map(|i| self.prev_skin[i as usize].as_dvec3());
                            let v1 = tri.map(|i| skin_pos[i as usize].as_dvec3());
                            let dw = (solid_angle(p1.as_dvec3(), v1)
                                - solid_angle(p0.as_dvec3(), v0))
                                * topo.orient[t] as f64
                                / (4.0 * std::f64::consts::PI);
                            if dw.abs() > top.0.abs() {
                                top = (dw, t as u32);
                            }
                            if dead.get(t).copied().unwrap_or(false) {
                                gone += dw;
                            } else {
                                live += dw;
                            }
                        }
                    }
                }
            }
            // A step's crossing goes to the larger share (a pass at the rim of an
            // opening splits it between a dead and a live triangle).
            let total = (live + gone).round() as i32;
            let n = if gone.abs() > live.abs() { 0 } else { total };
            if total != 0 && n == 0 {
                self.through_dead += total.unsigned_abs() as u64;
                self.events.push((k as u32, top.1, 0));
            }
            if n != 0 {
                self.net[k] += n;
                if n > 0 {
                    self.inward += n as u64;
                } else {
                    self.outward += (-n) as u64;
                }
                self.events.push((k as u32, top.1, n.signum() as i8));
            }
        }
        self.prev_pts.clear();
        self.prev_pts.extend_from_slice(pts);
        self.prev_skin.clear();
        self.prev_skin.extend_from_slice(skin_pos);
    }

    /// The leaked points (net crossings not zero).
    pub fn leaked(&self) -> Vec<usize> {
        (0..self.net.len()).filter(|&k| self.net[k] != 0).collect()
    }

    /// Point `p`'s distance to the nearest live triangle (its penetration depth
    /// when it has leaked).
    pub fn depth(p: Vec3, tris: &[[u32; 3]], pos: &[Vec3], dead: &[bool]) -> f32 {
        let mut best = f32::INFINITY;
        for (t, tri) in tris.iter().enumerate() {
            if dead.get(t).copied().unwrap_or(false) {
                continue;
            }
            let [a, b, c] = tri.map(|i| pos[i as usize]);
            let (q, _, _) = closest_on_triangle(p, a, b, c);
            best = best.min((p - q).length());
        }
        best
    }
}

/// The solid angle of triangle `v` seen from `p` (Van Oosterom and Strackee),
/// in (−2π, 2π]: negative in front of its normal (v₁ − v₀) × (v₂ − v₀), positive
/// behind; it jumps by about 4π when the point passes through the triangle and
/// changes continuously otherwise.
fn solid_angle(p: glam::DVec3, v: [glam::DVec3; 3]) -> f64 {
    let (a, b, c) = (v[0] - p, v[1] - p, v[2] - p);
    let (la, lb, lc) = (a.length(), b.length(), c.length());
    let num = a.dot(b.cross(c));
    let den = la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la;
    2.0 * num.atan2(den)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> (Vec<[u32; 3]>, Vec<Vec3>) {
        let pos: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32))
            .collect();
        // Outward-wound faces of the unit cube.
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let mut tris = Vec::new();
        for q in quads {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
        }
        (tris, pos)
    }

    fn side(p: Vec3, tris: &[[u32; 3]], pos: &[Vec3], skin: &SkinTopology) -> u32 {
        let dead = vec![false; tris.len()];
        (0..tris.len())
            .map(|t| side_key(p, t, tris, pos, &dead, skin))
            .min()
            .unwrap()
            & 3
    }

    #[test]
    fn a_cube_is_closed_and_oriented_outward_whatever_its_windings() {
        let (mut tris, pos) = cube();
        // Flip half the windings: the orientation is recovered.
        for t in tris.iter_mut().step_by(2) {
            t.swap(1, 2);
        }
        let skin = SkinTopology::build(&tris, &pos);
        assert_eq!(skin.closed_components, 1);
        for (t, tri) in tris.iter().enumerate() {
            let [a, b, c] = tri.map(|i| pos[i as usize]);
            let n = (b - a).cross(c - a) * skin.orient[t];
            let centre = (a + b + c) / 3.0 - Vec3::splat(0.5);
            assert!(n.dot(centre) > 0.0, "triangle {t} not outward");
        }
        for i in 0..8 {
            let f = skin.fan_start[i + 1] - skin.fan_start[i];
            assert!(f >= 3, "vertex {i}: fan {f}");
        }
    }

    #[test]
    fn inside_and_outside_near_every_feature_of_a_cube() {
        let (tris, pos) = cube();
        let skin = SkinTopology::build(&tris, &pos);
        let e = 0.01;
        // Faces, edges, vertices; just inside and just outside.
        let probes = [
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.3, 1.0, 0.7),
            Vec3::new(1.0, 0.4, 1.0),
        ];
        let c = Vec3::splat(0.5);
        for q in probes {
            let out = (q - c).normalize();
            assert_eq!(side(q + out * e, &tris, &pos, &skin), SIDE_OUTSIDE, "{q}");
            assert_eq!(side(q - out * e, &tris, &pos, &skin), SIDE_INSIDE, "{q}");
        }
        // Sideways to a face, beyond an edge (the case the vote got wrong): outside.
        assert_eq!(
            side(Vec3::new(-0.005, 0.5, 0.02), &tris, &pos, &skin),
            SIDE_OUTSIDE
        );
    }

    #[test]
    fn an_open_surface_and_a_dead_triangle_have_no_side() {
        let (tris, pos) = cube();
        let open = SkinTopology::build(&tris[1..], &pos);
        assert_eq!(open.closed_components, 0);
        let p = Vec3::new(0.5, 0.5, -0.01);
        let dead = vec![false; tris.len() - 1];
        let k = (0..tris.len() - 1)
            .map(|t| side_key(p, t, &tris[1..], &pos, &dead, &open))
            .min()
            .unwrap();
        assert_eq!(k & 3, SIDE_UNKNOWN);
        let skin = SkinTopology::build(&tris, &pos);
        let mut dead = vec![false; tris.len()];
        // The top face's triangles 2 and 3 share the diagonal 4–7: with 2 dead, a
        // point just above the diagonal is closest to an open edge.
        dead[2] = true;
        let q = Vec3::new(0.5, 0.5, 1.0);
        let keys: Vec<u32> = (0..tris.len())
            .filter(|&t| !dead[t])
            .map(|t| side_key(q + Vec3::Z * 0.01, t, &tris, &pos, &dead, &skin))
            .collect();
        assert_eq!(keys.iter().min().unwrap() & 3, SIDE_UNKNOWN);
    }

    #[test]
    fn the_angle_polynomial_is_within_1e5_of_atan2() {
        for i in 0..=2000 {
            let th = std::f32::consts::PI * i as f32 / 2000.0;
            let (y, x) = (th.sin().abs(), th.cos());
            let a = angle_atan2(y * 3.0, x * 3.0);
            assert!((a - y.atan2(x)).abs() < 2e-5, "{th}: {a}");
        }
    }

    #[test]
    fn the_leak_tracker_counts_crossings_of_live_triangles_only() {
        let (tris, pos) = cube();
        let skin = SkinTopology::build(&tris, &pos);
        let mut dead = vec![false; tris.len()];
        // Two points: one through the bottom face (z = 0), one through the top.
        let path = |z: f32| [Vec3::new(0.3, 0.6, z), Vec3::new(0.6, 0.3, 1.0 - z)];
        let mut lt = LeakTracker::new(&path(-0.2), &pos);
        // The top face's triangles (4 → 5 → 7, 4 → 7 → 6) dead before anything moves.
        dead[2] = true;
        dead[3] = true;
        for k in 1..=10 {
            lt.update(&path(-0.2 + 0.05 * k as f32), &pos, &tris, &dead, &skin);
        }
        assert_eq!(
            lt.net,
            vec![1, 0],
            "in through the live bottom, the dead top"
        );
        assert_eq!((lt.inward, lt.outward, lt.through_dead), (1, 0, 1));
        // Back out the way each came.
        for k in (0..10).rev() {
            lt.update(&path(-0.2 + 0.05 * k as f32), &pos, &tris, &dead, &skin);
        }
        assert_eq!(lt.net, vec![0, 0]);
        assert_eq!(lt.leaked(), Vec::<usize>::new());
        // The skin moving through a resting point counts too.
        let p = [Vec3::new(0.5, 0.5, 1.1)];
        let mut lt = LeakTracker::new(&p, &pos);
        let lifted: Vec<Vec3> = pos.iter().map(|&q| q + Vec3::Z * 0.2).collect();
        lt.update(&p, &lifted, &tris, &vec![false; tris.len()], &skin);
        assert_eq!(lt.leaked(), vec![0]);
    }
}
