//! Structural members as node–beam trusses, so they bend and buckle the way the
//! real parts do: box-section members along a path (rails, sills, pillars, roof
//! rails, cross members, bumper beams, door intrusion beams), lattice blocks
//! (engine and gearbox), plates (radiator), and a double-wishbone suspension corner.

use super::material::Material;
use super::skeleton::{BeamType, SkShock, SkWheel, Skeleton};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// A box-section truss: four corner nodes per station.
#[derive(Clone, Debug, Default)]
pub struct Truss {
    pub stations: Vec<[u32; 4]>,
    /// Arc fraction of each station along the path (0 at its start).
    pub t: Vec<f32>,
    /// Contact nodes between stations (see [`CATCH`]).
    pub catchers: Vec<u32>,
}

/// The longest gap between a member's nodes along its length (m): a 20 cm pole
/// always meets one.
pub const CATCH: f32 = 0.11;

/// A suspension upright's beams against the link grade.
pub const KNUCKLE: f32 = 4.0;

/// A section ring's stiffness against the member's grade (its ties are
/// short: [`Skeleton::tie`]).
pub const RING: f32 = 0.25;
/// Bend beams (over two stations) against the member's grade.
pub const BEND: f32 = 0.5;
/// Face diagonals against the member's grade.
pub const DIAG: f32 = 0.5;
/// A catcher's ties against the member's grade.
pub const CATCH_TIE: f32 = 0.25;

/// A member's station node this close to an existing node (m) reuses it.
pub const SNAP: f32 = 0.05;

/// A station node this close to another member's frame node (m) is that
/// node: members meet at shared joints, not through gusset ties.
pub const SNAP_FRAME: f32 = 0.15;

impl Truss {
    pub fn nodes(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.stations.iter().flatten().copied().collect();
        v.extend_from_slice(&self.catchers);
        v.sort();
        v.dedup();
        v
    }
    pub fn first(&self) -> [u32; 4] {
        self.stations[0]
    }
    pub fn last(&self) -> [u32; 4] {
        *self.stations.last().unwrap()
    }
}

/// Points along `path` at most `spacing` apart with their arc fractions: every
/// path vertex is a station (members meet at their paths' vertices: a joint
/// node shared with the member crossing there), each segment split evenly.
pub fn resample(path: &[Vec3], spacing: f32) -> Vec<(Vec3, f32)> {
    let mut acc = vec![0.0f32];
    for w in path.windows(2) {
        acc.push(acc.last().unwrap() + w[0].distance(w[1]));
    }
    let total = acc.last().unwrap().max(1e-9);
    let mut out = vec![(path[0], 0.0)];
    for (j, w) in path.windows(2).enumerate() {
        let l = w[0].distance(w[1]);
        let n = (l / spacing - 1e-3).ceil().max(1.0) as usize;
        for i in 1..=n {
            let t = i as f32 / n as f32;
            out.push((w[0].lerp(w[1], t), (acc[j] + l * t) / total));
        }
    }
    out
}

/// A member's description: path, section and material along it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemberSpec {
    pub name: String,
    pub path: Vec<[f32; 3]>,
    /// Section width (across, in the plane of `up`'s normal) and height (along `up`).
    pub width: f32,
    pub height: f32,
    /// Direction the section's height points to (it is made square to the path).
    pub up: [f32; 3],
    /// Material grade name (`Grades` field) and its threshold scale at the path's
    /// start and end (a crush gradient: weak at the bumper, strong at the cell).
    pub grade: String,
    pub scale: [f32; 2],
    /// Every this many stations the section is weaker (crush beads: the member
    /// folds there first, in an accordion). 0: none.
    pub beads: usize,
    /// Built on both sides (the path's x mirrored for the second).
    pub mirrored: bool,
    /// Join each station to this many of the nearest shell nodes (0: none).
    pub attach: usize,
    /// Rides the panel around its middle (door beams, lid frames): built in that
    /// part and attached to it alone.
    #[serde(default)]
    pub ride: bool,
    /// Nodes per station: 4 (a box, for members that buckle in 3-D: rails, sills,
    /// pillars) or 2 (a ladder in the plane of `up`: cross members, bows, frames).
    #[serde(default = "four")]
    pub section: usize,
    /// Station spacing (m); 0: the resolution's (crush members: its crush spacing).
    /// Longer stations make a stiffer member for the same mass: a line's axial
    /// rigidity grows with the square of its station spacing at a fixed mass and step.
    #[serde(default)]
    pub station: f32,
    /// Pole catchers between stations (impact members only: bumper beams, door
    /// beams, sills, B-pillars). See [`CATCH`].
    #[serde(default)]
    pub catch: bool,
    /// Stiffness against the grade's (k·L, damping at the grade's ratio):
    /// the main load paths (rails, sills) are heavier sections of their steel.
    #[serde(default = "unit")]
    pub stiff: f32,
}

fn unit() -> f32 {
    1.0
}

fn four() -> usize {
    4
}

/// Build one box truss along `path`.
pub fn box_truss(
    sk: &mut Skeleton,
    path: &[Vec3],
    width: f32,
    height: f32,
    up: Vec3,
    spacing: f32,
    part: u16,
    mat: &dyn Fn(f32) -> Material,
    beads: usize,
    section: usize,
    catch: bool,
    snap: &[u32],
) -> Truss {
    let pts = resample(path, spacing);
    let mut truss = Truss::default();
    // A station node within `SNAP` of an existing node of its part (the shell,
    // a member built before) is that node: a member is welded where it meets
    // the body, and two coincident nodes joined by a tie would be a pin.
    let frame_snap = SNAP_FRAME;
    let mk = |sk: &mut Skeleton, q: Vec3, used: &[u32]| -> u32 {
        let near = snap
            .iter()
            .copied()
            .filter(|&j| sk.nodes[j as usize].part == part && !used.contains(&j))
            // Never across or onto the centre line from off it (the mirror
            // twin would want the same node).
            .filter(|&j| {
                let x = sk.pos(j).x;
                if q.x.abs() < 1e-3 {
                    x.abs() < 1e-3
                } else {
                    x * q.x > 0.0 && x.abs() > 1e-3
                }
            })
            .min_by(|&a, &b| {
                // Ties broken by position, so mirrored members pick mirrored
                // nodes.
                let key = |i: u32| {
                    let p = sk.pos(i);
                    (
                        (p.distance(q) * 1e4).round() as i64,
                        (p.x.abs() * 1e4).round() as i64,
                        (p.y * 1e4).round() as i64,
                        (p.z * 1e4).round() as i64,
                    )
                };
                key(a).cmp(&key(b))
            });
        match near {
            // Onto another member's frame node from farther (a shared joint
            // instead of a gusset).
            Some(j)
                if sk.pos(j).distance(q) < SNAP
                    || (sk.nodes[j as usize].primary && sk.pos(j).distance(q) < frame_snap) =>
            {
                j
            }
            _ => sk.node(q, part),
        }
    };
    for (i, &(p, t)) in pts.iter().enumerate() {
        // Central differences (symmetric paths give mirrored sections).
        let prev = if i > 0 { pts[i - 1].0 } else { p };
        let next = if i + 1 < pts.len() { pts[i + 1].0 } else { p };
        let tan = (next - prev).normalize_or_zero();
        let mut side = tan.cross(up);
        if side.length_squared() < 1e-6 {
            side = tan.cross(Vec3::X);
        }
        let side = side.normalize();
        let u = side.cross(tan).normalize();
        let hw = 0.5 * width;
        let hh = 0.5 * height;
        let st = if section == 2 {
            let a = mk(sk, p - u * hh, &[]);
            let b = mk(sk, p + u * hh, &[a]);
            [a, b, a, b]
        } else {
            let c = [
                p - side * hw - u * hh,
                p + side * hw - u * hh,
                p + side * hw + u * hh,
                p - side * hw + u * hh,
            ];
            // Snapped in an order by position (mirror-invariant), so mirrored
            // members claim mirrored nodes.
            let mut order = [0usize, 1, 2, 3];
            let key = |q: Vec3| {
                (
                    (q.x.abs() * 1e4).round() as i64,
                    (q.y * 1e4).round() as i64,
                    (q.z * 1e4).round() as i64,
                )
            };
            order.sort_by_key(|&e| key(c[e]));
            let mut used: Vec<u32> = Vec::new();
            let mut out = [0u32; 4];
            for e in order {
                let i = mk(sk, c[e], &used);
                used.push(i);
                out[e] = i;
            }
            out
        };
        truss.stations.push(st);
        truss.t.push(t);
    }
    let n = truss.stations.len();
    // Longitudinals carry the member's load at full stiffness; the section's
    // ring, its diagonal and the bend beams keep its shape at half; the face
    // diagonals only stop shear (a third), so the mass the step needs goes into
    // axial rigidity.
    for i in 0..n {
        let s = truss.stations[i];
        let m = mat(truss.t[i]);
        let ring = if beads > 0 && i % beads == beads / 2 && i > 0 && i + 1 < n {
            m.scaled(0.6)
        } else {
            m
        }
        .stiffer(RING);
        // The section's ring and diagonal are short: ties (their stiffness that of
        // a `TIE_LENGTH` beam), so the mass the step needs goes into the member's
        // length.
        for e in 0..4 {
            sk.tie(s[e], s[(e + 1) % 4], ring);
        }
        sk.tie(s[0], s[2], ring);
        if i + 2 < n {
            // Bend beams over two stations: stiffness against buckling.
            let q = truss.stations[i + 2];
            let mm = mat(truss.t[i + 1]).stiffer(BEND);
            for e in 0..4 {
                sk.beam(s[e], q[e], mm);
            }
        }
        if i + 1 < n {
            let q = truss.stations[i + 1];
            let mm = mat(0.5 * (truss.t[i] + truss.t[i + 1]));
            for e in 0..4 {
                sk.beam(s[e], q[e], mm);
            }
            let diag = mm.stiffer(DIAG);
            for e in 0..4 {
                sk.beam(s[e], q[(e + 1) % 4], diag);
                sk.beam(s[(e + 1) % 4], q[e], diag);
            }
        }
    }
    // Catchers: static contact is per node, so a bay longer than `CATCH` gets
    // nodes along one of its lines at most `CATCH` apart, each braced to both
    // stations of its line and of the opposite line (a pole between two
    // stations still meets nodes that carry its load into the member).
    for i in 0..if catch { n.saturating_sub(1) } else { 0 } {
        let (s0, s1) = (truss.stations[i], truss.stations[i + 1]);
        let m = mat(0.5 * (truss.t[i] + truss.t[i + 1])).stiffer(CATCH_TIE);
        // One line: a vertical pole meets every line of a bay at once (the
        // section's lines are at the same place in plan view, or beside each
        // other along the pole's path).
        // The outer line (mirrored members get mirrored lanes), else the lower.
        let lines_n = if section == 2 { 2 } else { 4 };
        let pick = {
            // By position alone (mirrored members get mirrored lanes): the
            // outermost line, else the lowest, else the frontmost.
            let mid = |e: usize| sk.pos(s0[e]) + sk.pos(s1[e]);
            (0..lines_n)
                .max_by(|&a, &b| {
                    let (pa, pb) = (mid(a), mid(b));
                    let k = |p: Vec3| {
                        (
                            (p.x.abs() * 50.0).round() as i64,
                            (-p.y * 50.0).round() as i64,
                            (p.z * 50.0).round() as i64,
                        )
                    };
                    k(pa).cmp(&k(pb))
                })
                .unwrap()
        };
        // A member on the centre line with its lines either side (the tunnel):
        // both, so it stays symmetric.
        let (a, b) = (sk.pos(s0[0]) + sk.pos(s1[0]), sk.pos(s0[1]) + sk.pos(s1[1]));
        let lines: Vec<usize> = if (a.x + b.x).abs() < 2e-2 && a.x.abs() > 2e-2 {
            vec![0, 1]
        } else {
            vec![pick]
        };
        for c in lines {
            // Each line by its own length (snapped stations bend them).
            let l = sk.pos(s0[c]).distance(sk.pos(s1[c]));
            let cuts = (l / CATCH - 1e-3).ceil() as usize;
            let c2 = if section == 2 { 1 - c } else { 3 - c };
            for j in 1..cuts.max(1) {
                let t = j as f32 / cuts as f32;
                let p = sk.pos(s0[c]).lerp(sk.pos(s1[c]), t);
                let q = sk.node(p, part);
                truss.catchers.push(q);
                for e in [s0[c], s1[c], s0[c2], s1[c2]] {
                    sk.tie(q, e, m);
                }
            }
        }
    }
    truss
}

/// Join every node of `nodes` to its `count` nearest nodes of `pool` (within
/// `reach` m) with ties ([`Skeleton::tie`]).
pub fn attach(
    sk: &mut Skeleton,
    nodes: &[u32],
    pool: &[u32],
    count: usize,
    reach: f32,
    mat: Material,
) {
    for &a in nodes {
        let p = sk.pos(a);
        for b in sk.nearest(p, pool, count) {
            if b != a && sk.pos(b).distance(p) <= reach {
                sk.tie(a, b, mat);
            }
        }
    }
}

/// A rigid lattice block (engine, gearbox, battery): nodes on a grid at most
/// `spacing` apart filling the box `centre ± half`, joined along every cell edge,
/// face diagonal and body diagonal.
pub fn block(
    sk: &mut Skeleton,
    centre: Vec3,
    half: Vec3,
    spacing: f32,
    part: u16,
    mat: Material,
) -> Vec<u32> {
    let cnt = |h: f32| ((2.0 * h / spacing).ceil() as usize).max(1) + 1;
    let (nx, ny, nz) = (cnt(half.x), cnt(half.y), cnt(half.z));
    let mut idx = vec![0u32; nx * ny * nz];
    let at = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let f = |n: usize, c: usize| {
                    if n == 1 {
                        0.0
                    } else {
                        c as f32 / (n - 1) as f32 * 2.0 - 1.0
                    }
                };
                let p = centre + half * Vec3::new(f(nx, i), f(ny, j), f(nz, k));
                idx[at(i, j, k)] = sk.node(p, part);
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let a = idx[at(i, j, k)];
                for di in 0..2usize {
                    for dj in 0..2usize {
                        for dk in 0..2usize {
                            if di + dj + dk == 0 {
                                continue;
                            }
                            let (ii, jj, kk) = (i + di, j + dj, k + dk);
                            if ii < nx && jj < ny && kk < nz {
                                sk.fixed_beam(a, idx[at(ii, jj, kk)], mat);
                            }
                        }
                    }
                    // The cell's other diagonals (from the neighbour corners).
                }
                for (di, dj, dk) in [
                    (1i32, -1i32, 0i32),
                    (1, 0, -1),
                    (0, 1, -1),
                    (1, -1, -1),
                    (1, 1, -1),
                    (1, -1, 1),
                    (-1, 1, 1),
                ] {
                    let (ii, jj, kk) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                    if ii >= 0
                        && jj >= 0
                        && kk >= 0
                        && (ii as usize) < nx
                        && (jj as usize) < ny
                        && (kk as usize) < nz
                    {
                        sk.fixed_beam(a, idx[at(ii as usize, jj as usize, kk as usize)], mat);
                    }
                }
            }
        }
    }
    idx
}

/// A flat braced grid in the plane through `corner` spanned by `u` and `v`.
pub fn plate(
    sk: &mut Skeleton,
    corner: Vec3,
    u: Vec3,
    v: Vec3,
    spacing: f32,
    part: u16,
    mat: Material,
) -> Vec<Vec<u32>> {
    let nu = (u.length() / spacing).ceil().max(1.0) as usize;
    let nv = (v.length() / spacing).ceil().max(1.0) as usize;
    let mut g = vec![vec![0u32; nu + 1]; nv + 1];
    for (j, row) in g.iter_mut().enumerate() {
        for (i, slot) in row.iter_mut().enumerate() {
            *slot = sk.node(
                corner + u * (i as f32 / nu as f32) + v * (j as f32 / nv as f32),
                part,
            );
        }
    }
    for j in 0..=nv {
        for i in 0..=nu {
            if i < nu {
                sk.beam(g[j][i], g[j][i + 1], mat);
            }
            if j < nv {
                sk.beam(g[j][i], g[j + 1][i], mat);
            }
            if i < nu && j < nv {
                sk.beam(g[j][i], g[j + 1][i + 1], mat);
                sk.beam(g[j][i + 1], g[j + 1][i], mat);
            }
        }
    }
    g
}

/// A suspension corner's geometry (left side, x > 0; the right mirrors it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Corner {
    /// Wheel centre at spawn: half track, height, station along the car.
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub tyre_radius: f32,
    pub tyre_width: f32,
    /// Tyre (ring) mass and the hub/knuckle/brake mass (kg).
    pub wheel_mass: f32,
    pub hub_mass: f32,
    /// Control arms and their brackets (kg, on the chassis pivots).
    #[serde(default)]
    pub arm_mass: f32,
    pub rays: u32,
    pub tyre_spring: f32,
    pub tyre_damp: f32,
    /// Spring (N/m at the wheel) and damper (N·s/m).
    pub spring: f32,
    pub damper: f32,
    pub driven: bool,
    pub braked: bool,
}

/// What a corner built: its nodes, and the chassis pivots to tie into the body.
#[derive(Clone, Debug, Default)]
pub struct CornerNodes {
    pub axle: [u32; 2],
    pub knuckle: Vec<u32>,
    pub pivots: Vec<u32>,
    pub shock_top: u32,
}

/// A double-wishbone corner: a rigid five-node knuckle carrying the tyre's axle
/// nodes, lower and upper arms to four chassis pivots, a tie rod, and a
/// spring–damper from the lower ball joint to the tower. `side` is +1 (left) or
/// −1 (right).
pub fn corner(sk: &mut Skeleton, c: &Corner, side: f32, part: u16, rigid: Material) -> CornerNodes {
    let p = |x: f32, y: f32, z: f32| Vec3::new(side * x, y, z);
    let (x, y, z) = (c.x, c.y, c.z);
    let hw = 0.5 * c.tyre_width;
    let a1 = sk.node(p(x - hw, y, z), part);
    let a2 = sk.node(p(x + hw, y, z), part);
    let ku = sk.node(p(x - 0.11, y + 0.17, z), part);
    let kl = sk.node(p(x - 0.09, y - 0.13, z), part);
    let ks = sk.node(p(x - 0.12, y + 0.02, z - 0.13), part);
    let knuckle = vec![a1, a2, ku, kl, ks];
    // The upright is a casting carrying the hub: far stiffer than the arms
    // (its nodes are heavy: the hub's mass).
    for i in 0..knuckle.len() {
        for j in i + 1..knuckle.len() {
            sk.fixed_beam(knuckle[i], knuckle[j], rigid.stiffer(KNUCKLE));
        }
    }
    // The hub, bearing and brake ride the axle nodes (the tyre's spokes need the
    // mass there); the upright's the ball joints.
    sk.add_mass(&[a1, a2], 0.7 * c.hub_mass);
    sk.add_mass(&[ku, kl, ks], 0.3 * c.hub_mass);
    let plf = sk.node(p(0.36, y - 0.12, z + 0.2), part);
    let plr = sk.node(p(0.36, y - 0.12, z - 0.2), part);
    let puf = sk.node(p(0.47, y + 0.2, z + 0.13), part);
    let pur = sk.node(p(0.47, y + 0.2, z - 0.13), part);
    let pt = sk.node(p(0.35, y + 0.02, z - 0.13), part);
    let top = sk.node(p(x - 0.17, y + 0.47, z), part);
    let arm = rigid;
    sk.fixed_beam(kl, plf, arm);
    sk.fixed_beam(kl, plr, arm);
    sk.fixed_beam(ku, puf, arm);
    sk.fixed_beam(ku, pur, arm);
    sk.fixed_beam(ks, pt, arm);
    // The pivots as one subframe bracket.
    let pivots = vec![plf, plr, puf, pur, pt, top];
    // The arms' and subframe brackets' own mass.
    sk.add_mass(&pivots, c.arm_mass);
    for i in 0..pivots.len() {
        for j in i + 1..pivots.len() {
            sk.fixed_beam(pivots[i], pivots[j], rigid);
        }
    }
    let len = sk.pos(kl).distance(sk.pos(top));
    let _ = len;
    sk.shocks.push(SkShock {
        a: kl,
        b: top,
        spring: c.spring,
        damp: c.damper,
        short: 0.3,
        long: 0.3,
        stop: 5.0e5,
        precomp: 1.0,
    });
    sk.wheels.push(SkWheel {
        a1,
        a2,
        radius: c.tyre_radius,
        width: c.tyre_width,
        rays: c.rays,
        mass: c.wheel_mass,
        spring: c.tyre_spring,
        damp: c.tyre_damp,
        propulsed: c.driven,
        braked: c.braked,
    });
    let _ = BeamType::Normal;
    CornerNodes {
        axle: [a1, a2],
        knuckle,
        pivots,
        shock_top: top,
    }
}
