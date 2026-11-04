//! The visual skin: a dense mesh sampled from the body profile (the same curves
//! the skeleton is lofted from), with materials (paint, glass, chrome, black
//! trim, lights, tyres, rims), bound vertex by vertex to the skeleton so it
//! deforms with the crash, flexbody style.
//!
//! Binding: every vertex rides one triangle of nodes (n₀, n₁, n₂) of its own part
//! (a door's vertices ride door nodes, a wheel's its tyre's nodes):
//!
//! ```text
//! e₁ = P₁ − P₀, e₂ = P₂ − P₀, N = normalize(e₁ × e₂)
//! position = P₀ + α·e₁ + β·e₂ + h·N
//! T = normalize(e₁), B = N × T, normal = nₜ·T + n_b·B + n_n·N
//! ```
//!
//! with (α, β, h) and (nₜ, n_b, n_n) fixed at rest. A panel that tears off keeps
//! its vertices on its own nodes, so nothing stretches across a seam; a crumpled
//! panel shows its nodes' folds. `showcase-car-render.mjs` evaluates this on the
//! GPU from the explicit solver's node buffer ([`deform`] is the same on the CPU).

use super::body::Seg;
use super::curve::Outline;
use super::vehicle::Built;
use crate::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Tyre};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// A binding triangle stretched past this × a rest edge has torn: its vertices
/// are dropped (their triangles vanish) instead of stretching across the tear.
pub const TEAR_STRETCH: f32 = 2.5;

/// Material ids of the visual mesh.
pub mod mat {
    pub const PAINT: u8 = 0;
    pub const GLASS: u8 = 1;
    pub const CHROME: u8 = 2;
    pub const TRIM: u8 = 3;
    pub const HEADLIGHT: u8 = 4;
    pub const TAILLIGHT: u8 = 5;
    pub const TYRE: u8 = 6;
    pub const RIM: u8 = 7;
    pub const GAP: u8 = 8;
    pub const UNDERBODY: u8 = 9;
    pub const PLATE: u8 = 10;
    pub const GRILLE: u8 = 11;
    pub const SEAT: u8 = 12;
    pub const DASH: u8 = 13;
    pub const DOORCARD: u8 = 14;
    pub const HEADLINER: u8 = 15;
    pub const CARPET: u8 = 16;
    pub const COUNT: usize = 17;
}

/// Material names, by id.
pub const MATERIAL_NAMES: [&str; mat::COUNT] = [
    "paint",
    "glass",
    "chrome",
    "trim",
    "headlight",
    "taillight",
    "tyre",
    "rim",
    "gap",
    "underbody",
    "plate",
    "grille",
    "seat",
    "dash",
    "doorcard",
    "headliner",
    "carpet",
];

#[derive(Clone, Debug, Default)]
pub struct ShowcaseVisual {
    /// Rest positions in the `.truck`'s coordinates.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub material: Vec<u8>,
    pub indices: Vec<u32>,
    /// Per vertex: the three solver nodes (indices into the explicit solver's
    /// node list, tyre nodes included) and (α, β, h).
    pub bind_nodes: Vec<[u32; 3]>,
    pub bind_coords: Vec<[f32; 3]>,
    /// The rest normal in the triangle's frame (T, B, N).
    pub bind_normal: Vec<[f32; 3]>,
    /// The binding triangle's rest edge lengths |e₁|, |e₂|: a vertex whose
    /// triangle has stretched past [`TEAR_STRETCH`] × either is torn (dropped).
    pub bind_rest: Vec<[f32; 2]>,
    /// 1 on a panel's edge at a seam (the renderer draws a thin groove there).
    pub seam: Vec<f32>,
    /// The solver's node count for this vehicle (tyre nodes included).
    pub node_count: u32,
}

/// Candidate binding triangles of one part, with a coarse grid over their centroids.
struct Binder {
    tris: Vec<[u32; 3]>,
    pos: Vec<Vec3>,
    cell: f32,
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl Binder {
    fn new(tris: Vec<[u32; 3]>, pos: &[Vec3], cell: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
            grid.entry(key(c, cell)).or_default().push(ti);
        }
        Self {
            tris,
            pos: pos.to_vec(),
            cell,
            grid,
        }
    }

    /// The nearest triangle to `p`, and the binding coordinates.
    fn bind(&self, p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3], [f32; 3])> {
        if self.tris.is_empty() {
            return None;
        }
        let k = key(p, self.cell);
        let mut best: Option<(f32, usize)> = None;
        for r in 1..8 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if let Some(list) = self.grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for &ti in list {
                                let t = self.tris[ti];
                                let d = point_triangle(
                                    p,
                                    self.pos[t[0] as usize],
                                    self.pos[t[1] as usize],
                                    self.pos[t[2] as usize],
                                );
                                if best.is_none_or(|b| d < b.0) {
                                    best = Some((d, ti));
                                }
                            }
                        }
                    }
                }
            }
            if best.is_some_and(|b| b.0 < (r as f32 - 1.0) * self.cell) {
                break;
            }
        }
        let (_, ti) = best?;
        let t = self.tris[ti];
        let (p0, p1, p2) = (
            self.pos[t[0] as usize],
            self.pos[t[1] as usize],
            self.pos[t[2] as usize],
        );
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let d = p - p0;
        let h = d.dot(n);
        let q = d - n * h;
        // Solve q = α e₁ + β e₂ in the plane.
        let (a11, a12, a22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
        let (b1, b2) = (q.dot(e1), q.dot(e2));
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-12 {
            return None;
        }
        let alpha = (b1 * a22 - b2 * a12) / det;
        let beta = (a11 * b2 - a12 * b1) / det;
        let tt = e1.normalize();
        let bb = n.cross(tt);
        let nl = [normal.dot(tt), normal.dot(bb), normal.dot(n)];
        Some((t, [alpha, beta, h], nl))
    }
}

fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
    (
        (p.x / cell).floor() as i32,
        (p.y / cell).floor() as i32,
        (p.z / cell).floor() as i32,
    )
}

/// Distance from `p` to triangle (a, b, c) (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// A vertex under construction (definition coordinates).
#[derive(Clone, Copy)]
struct V {
    p: Vec3,
    n: Vec3,
    uv: Vec2,
    mat: u8,
    part: u16,
    seam: f32,
    /// Binds to a structure group's triangles (index into `groups`), not a part.
    grp: i32,
}

/// Build the visual skin of a built vehicle.
pub fn showcase_visual(built: &Built) -> Result<ShowcaseVisual, String> {
    let def = &built.def;
    let prof = &def.profile;
    let look = &def.look;
    let sk = &built.skeleton;
    let len = prof.length;
    let paint = built.params.paint.unwrap_or(look.paint);
    // The solver's nodes (tyre nodes included), in definition coordinates.
    let sim = ExplicitBeams::from_truck(
        &built.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )?;
    let node_pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos - built.shift).collect();
    // Part of a body surface point (the skeleton's panel rule).
    let part_id = |p: Vec3, seg: Seg| -> u16 {
        for pn in &def.panels {
            if pn.region.contains(p, seg, len) {
                let name = if pn.sided {
                    format!("{}_{}", pn.name, if p.x >= 0.0 { "l" } else { "r" })
                } else {
                    pn.name.clone()
                };
                return sk.parts.iter().position(|q| *q == name).unwrap_or(0) as u16;
            }
        }
        0
    };
    let glass_depth = |p: Vec3, seg: Seg| -> Option<f32> {
        match seg {
            Seg::Top => {
                for r in [look.windscreen, look.backlight] {
                    if p.z > r[0] && p.z < r[1] {
                        let edge = prof.top_half_width.eval(p.z) - prof.top_radius - p.x.abs();
                        return Some((p.z - r[0]).min(r[1] - p.z).min(edge + 0.04));
                    }
                }
                None
            }
            Seg::Upper | Seg::Side => {
                for o in &look.side_glass {
                    let d = o.signed_distance(p.z, p.y);
                    if d < 0.0 {
                        return Some(-d);
                    }
                }
                None
            }
            _ => None,
        }
    };
    let side_glass_near = |p: Vec3| -> f32 {
        look.side_glass
            .iter()
            .map(|o| o.signed_distance(p.z, p.y))
            .fold(f32::MAX, f32::min)
    };
    let in_outline = |o: &Outline, x: f32, y: f32| o.contains(x.abs(), y);
    let material_of = |p: Vec3, n: Vec3, seg: Seg| -> u8 {
        if matches!(seg, Seg::Floor | Seg::Well) {
            return mat::UNDERBODY;
        }
        if let Some(d) = glass_depth(p, seg) {
            return if d < look.frit { mat::TRIM } else { mat::GLASS };
        }
        if matches!(seg, Seg::Upper | Seg::Side) {
            let d = side_glass_near(p);
            if d >= 0.0 && d < 0.018 {
                return mat::CHROME;
            }
            // The B-pillar between the windows: black.
            if look.side_glass.len() >= 2 && p.y > 1.0 {
                let zf = look.side_glass[0]
                    .0
                    .iter()
                    .map(|q| q[0])
                    .fold(f32::MAX, f32::min);
                let zr = look.side_glass[1]
                    .0
                    .iter()
                    .map(|q| q[0])
                    .fold(f32::MIN, f32::max);
                if p.z > zr && p.z < zf {
                    return mat::TRIM;
                }
            }
        }
        // Door handles.
        for h in &look.handles {
            let d = Vec2::new((p.z - h[0]) / 0.075, (p.y - h[1]) / 0.022).length();
            if d < 1.0 && n.x.abs() > 0.5 {
                return mat::CHROME;
            }
        }
        let front = p.z > 0.5 * len && n.z > 0.3;
        let rear = p.z < 0.5 * len && n.z < -0.3;
        if (front || (p.z > len - 0.5 && n.z > -0.3)) && in_outline(&look.headlight, p.x, p.y) {
            // A dark bezel, two chrome reflector bowls with bright projector lenses.
            let d = -look.headlight.signed_distance(p.x.abs(), p.y);
            if d < 0.012 {
                return mat::TRIM;
            }
            let o = &look.headlight.0;
            let cx = o.iter().map(|q| q[0]).sum::<f32>() / o.len() as f32;
            let cy = o.iter().map(|q| q[1]).sum::<f32>() / o.len() as f32;
            for dx in [-0.09f32, 0.09] {
                let r = Vec2::new(p.x.abs() - (cx + dx), p.y - cy).length();
                if r < 0.025 {
                    return mat::HEADLIGHT;
                }
                if r < 0.042 {
                    return mat::CHROME;
                }
            }
            return if d < 0.02 { mat::CHROME } else { mat::GLASS };
        }
        if front {
            if in_outline(&look.grille, p.x, p.y) {
                // Slats.
                return if ((p.y * 55.0).floor() as i32) % 2 == 0 {
                    mat::GRILLE
                } else {
                    mat::CHROME
                };
            }
            if in_outline(&look.intake, p.x, p.y) {
                return mat::GRILLE;
            }
            if p.x.abs() < 0.26 && (p.y - look.plate_y[0]).abs() < 0.06 {
                return mat::PLATE;
            }
        }
        if (rear || (p.z < 0.5 && n.z < 0.3)) && in_outline(&look.taillight, p.x, p.y) {
            let d = -look.taillight.signed_distance(p.x.abs(), p.y);
            if d < 0.01 {
                return mat::TRIM;
            }
            // Light bars over a darker lens.
            return if ((p.y * 45.0).floor() as i32) % 2 == 0 && d > 0.02 {
                mat::TAILLIGHT
            } else {
                mat::GAP
            };
        }
        if rear && p.x.abs() < 0.26 && (p.y - look.plate_y[1]).abs() < 0.06 {
            return mat::PLATE;
        }
        // A black sill and lower valance strip.
        if p.y < prof.bottom.eval(p.z) + 0.06 {
            return mat::TRIM;
        }
        mat::PAINT
    };
    let mut verts: Vec<V> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();
    // Body: rings of the profile section, sampled by arc length.
    let (z0, z1) = prof.loft_range();
    let nz = ((z1 - z0) / 0.025).ceil() as usize;
    let nv = 104; // segments per half section
    let section = |z: f32| -> Vec<(Vec3, Seg)> {
        let half = prof.half_section(z);
        let mut acc = vec![0.0f32];
        for w in half.windows(2) {
            acc.push(acc.last().unwrap() + w[0].p.distance(w[1].p));
        }
        let total = *acc.last().unwrap();
        let mut out = Vec::with_capacity(nv + 1);
        let mut j = 0;
        for i in 0..=nv {
            let s = total * i as f32 / nv as f32;
            while j + 2 < acc.len() && acc[j + 1] < s {
                j += 1;
            }
            let t = ((s - acc[j]) / (acc[j + 1] - acc[j]).max(1e-9)).clamp(0.0, 1.0);
            let p = half[j].p.lerp(half[j + 1].p, t);
            let seg = if t < 0.5 {
                half[j].seg
            } else {
                half[j + 1].seg
            };
            let x = if i == 0 || i == nv { 0.0 } else { p.x };
            out.push((Vec3::new(x, p.y, z), seg));
        }
        out
    };
    // Full rings: the right half, then the left mirrored back down.
    let grid: Vec<Vec<(Vec3, Seg)>> = (0..=nz)
        .map(|k| {
            let r = section(z0 + (z1 - z0) * k as f32 / nz as f32);
            let mut v = r.clone();
            for &(p, s) in r[1..r.len() - 1].iter().rev() {
                v.push((Vec3::new(-p.x, p.y, p.z), s));
            }
            v
        })
        .collect();
    let cols = grid[0].len();
    let at = |k: usize, i: usize| grid[k][i % cols];
    let normal_at = |k: usize, i: usize| -> Vec3 {
        let kp = (k + 1).min(nz);
        let km = k.saturating_sub(1);
        let tz = at(kp, i).0 - at(km, i).0;
        let tv = at(k, i + 1).0 - at(k, (i + cols - 1) % cols).0;
        let n = tv.cross(tz).normalize_or_zero();
        let p = at(k, i).0;
        let c = Vec3::new(0.0, 0.5 * (grid[k][0].0.y + grid[k][nv].0.y), p.z);
        if n.dot(p - c) < 0.0 {
            -n
        } else {
            n
        }
    };
    // Per grid quad its part (from its centre); vertices are shared within a part
    // and duplicated across seams (a panel gap, darkened and set in).
    let mut quad_part = vec![vec![0u16; cols]; nz];
    for (k, row) in quad_part.iter_mut().enumerate() {
        for (i, slot) in row.iter_mut().enumerate() {
            let c = (at(k, i).0 + at(k + 1, i).0 + at(k, i + 1).0 + at(k + 1, i + 1).0) / 4.0;
            *slot = part_id(c, at(k, i).1);
        }
    }
    let mut vert_of: HashMap<(usize, usize, u16), u32> = HashMap::new();
    for k in 0..nz {
        for i in 0..cols {
            let part = quad_part[k][i];
            let corners = [
                (k, i),
                (k + 1, i),
                (k + 1, (i + 1) % cols),
                (k, (i + 1) % cols),
            ];
            let mut ids = [0u32; 4];
            for (c, &(kk, ii)) in corners.iter().enumerate() {
                ids[c] = *vert_of.entry((kk, ii, part)).or_insert_with(|| {
                    let (p, seg) = at(kk, ii);
                    let n = normal_at(kk, ii);
                    let around = [
                        (kk.wrapping_sub(1), ii),
                        (kk, ii),
                        (kk.wrapping_sub(1), (ii + cols - 1) % cols),
                        (kk, (ii + cols - 1) % cols),
                    ];
                    let seam = around
                        .iter()
                        .filter(|(a, _)| *a < nz)
                        .any(|&(a, b)| quad_part[a][b] != part);
                    let m = material_of(p, n, seg);
                    let mut q = p;
                    if seam && m != mat::UNDERBODY {
                        q -= n * look.gap;
                    }
                    verts.push(V {
                        p: q,
                        n,
                        uv: Vec2::new(kk as f32 / nz as f32, ii as f32 / cols as f32),
                        mat: m,
                        part,
                        seam: if seam { 1.0 } else { 0.0 },
                        grp: -1,
                    });
                    (verts.len() - 1) as u32
                });
            }
            idx.extend_from_slice(&[ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]]);
        }
    }
    // Caps: a domed fan from each end ring toward its centre (the skeleton's cap
    // dome: sweep × (1 − t²), tapered to the top and bottom rows).
    let cap_first = verts.len();
    for (front, k) in [(false, 0usize), (true, nz)] {
        let ring = &grid[k];
        let zr = ring[0].0.z;
        let sweep = if front {
            prof.front_sweep
        } else {
            -prof.rear_sweep
        };
        let yb = ring[0].0.y;
        let yt = ring[nv].0.y;
        let c2 = Vec2::new(0.0, 0.5 * (yb + yt));
        let steps = 16;
        let mut prev: Vec<u32> = Vec::new();
        for s in 0..=steps {
            let t = 1.0 - s as f32 / steps as f32;
            let mut ids = Vec::with_capacity(cols);
            for item in ring.iter().take(cols) {
                let rim = Vec2::new(item.0.x, item.0.y);
                let q = c2 + (rim - c2) * t;
                let taper = (std::f32::consts::PI
                    * ((q.y - yb) / (yt - yb).max(1e-3)).clamp(0.0, 1.0))
                .sin()
                .sqrt();
                let p = Vec3::new(q.x, q.y, zr + sweep * (1.0 - t * t) * taper);
                let seg = if s == 0 { item.1 } else { Seg::Cap };
                let n = Vec3::new(0.0, 0.0, sweep.signum());
                verts.push(V {
                    p,
                    n,
                    uv: Vec2::new(if front { 1.0 } else { 0.0 }, t),
                    mat: mat::PAINT,
                    part: part_id(p, seg),
                    seam: 0.0,
                    grp: -1,
                });
                ids.push((verts.len() - 1) as u32);
            }
            if !prev.is_empty() {
                for i in 0..cols {
                    let (a, b, c, d) = (prev[i], prev[(i + 1) % cols], ids[(i + 1) % cols], ids[i]);
                    if front {
                        idx.extend_from_slice(&[a, c, b, a, d, c]);
                    } else {
                        idx.extend_from_slice(&[a, b, c, a, c, d]);
                    }
                }
            }
            prev = ids;
        }
    }
    let cap_end = verts.len();
    // Cap normals from their faces, then their materials.
    {
        let mut acc = vec![Vec3::ZERO; verts.len()];
        for t in idx.chunks(3) {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            if a < cap_first {
                continue;
            }
            let n = (verts[b].p - verts[a].p).cross(verts[c].p - verts[a].p);
            for i in [a, b, c] {
                acc[i] += n;
            }
        }
        for i in cap_first..cap_end {
            if acc[i].length_squared() > 0.0 {
                let mut n = acc[i].normalize();
                let out = if verts[i].p.z > 0.5 * len { 1.0 } else { -1.0 };
                if n.z * out < 0.0 {
                    n = -n;
                }
                verts[i].n = n;
            }
            verts[i].mat = material_of(verts[i].p, verts[i].n, Seg::Cap);
        }
    }
    // Mirrors: a rounded wedge at each front door's A-pillar corner.
    for side in [1.0f32, -1.0] {
        let z = look.mirror[0];
        let y = look.mirror[1];
        let hw = prof.half_width.eval(z);
        let c = Vec3::new(side * (hw + 0.06), y + 0.03, z - 0.04);
        let part = part_id(Vec3::new(side * hw, y - 0.05, z - 0.08), Seg::Side);
        let he = Vec3::new(0.09, 0.055, 0.045);
        let first = verts.len() as u32;
        let (n_lat, n_lon) = (8u32, 14u32);
        for a in 0..=n_lat {
            let th = std::f32::consts::PI * a as f32 / n_lat as f32;
            for b in 0..=n_lon {
                let ph = 2.0 * std::f32::consts::PI * b as f32 / n_lon as f32;
                let dir = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
                let m = if dir.z < -0.7 { mat::GLASS } else { mat::PAINT };
                verts.push(V {
                    p: c + dir * he,
                    n: (dir / he).normalize(),
                    uv: Vec2::ZERO,
                    mat: m,
                    part,
                    seam: 0.0,
                    grp: -1,
                });
            }
        }
        for a in 0..n_lat {
            for b in 0..n_lon {
                let i0 = first + a * (n_lon + 1) + b;
                let i2 = i0 + n_lon + 1;
                idx.extend_from_slice(&[i0, i2, i0 + 1, i0 + 1, i2, i2 + 1]);
            }
        }
    }
    // Interior: door cards, headliner and carpet as inner offsets of their panels;
    // seats, dashboard, console and steering wheel riding their frames.
    let groups: Vec<String> = built.bind_tris.keys().cloned().collect();
    let gi = |name: &str| {
        groups
            .iter()
            .position(|g| g == name)
            .map_or(-1, |i| i as i32)
    };
    {
        let roof = (look.backlight[1], look.windscreen[0]);
        for k in 0..nz {
            for i in 0..cols {
                let part = quad_part[k][i];
                let pname = &sk.parts[part as usize];
                let (p0, seg) = at(k, i);
                let c = (at(k, i).0 + at(k + 1, i).0 + at(k, i + 1).0 + at(k + 1, i + 1).0) / 4.0;
                let (m, off) = if pname.starts_with("door")
                    && matches!(seg, Seg::Side | Seg::Lower)
                    && c.y > 0.32
                {
                    (mat::DOORCARD, 0.07)
                } else if seg == Seg::Top && c.z > roof.0 + 0.05 && c.z < roof.1 - 0.05 {
                    (mat::HEADLINER, 0.05)
                } else if seg == Seg::Floor && c.z > 1.4 && c.z < 3.35 {
                    (mat::CARPET, 0.07)
                } else {
                    continue;
                };
                let _ = p0;
                let corners = [
                    (k, i),
                    (k, (i + 1) % cols),
                    (k + 1, (i + 1) % cols),
                    (k + 1, i),
                ];
                let first = verts.len() as u32;
                for &(kk, ii) in &corners {
                    let n = normal_at(kk, ii);
                    verts.push(V {
                        p: at(kk, ii).0 - n * off,
                        n: -n,
                        uv: Vec2::ZERO,
                        mat: m,
                        part,
                        seam: 0.0,
                        grp: -1,
                    });
                }
                idx.extend_from_slice(&[first, first + 1, first + 2, first, first + 2, first + 3]);
            }
        }
        let mut solid = |verts: &mut Vec<V>,
                         idx: &mut Vec<u32>,
                         c: Vec3,
                         h: Vec3,
                         tilt: f32,
                         m: u8,
                         grp: i32| {
            let rot = glam::Quat::from_rotation_x(tilt);
            let faces: [(Vec3, Vec3, Vec3); 6] = [
                (Vec3::X, Vec3::Y, Vec3::Z),
                (-Vec3::X, Vec3::Z, Vec3::Y),
                (Vec3::Y, Vec3::Z, Vec3::X),
                (-Vec3::Y, Vec3::X, Vec3::Z),
                (Vec3::Z, Vec3::X, Vec3::Y),
                (-Vec3::Z, Vec3::Y, Vec3::X),
            ];
            for (n, u, v) in faces {
                let first = verts.len() as u32;
                for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    // Slightly rounded: pull the corners in.
                    let q = (n + u * a * 0.92 + v * b * 0.92) * h;
                    verts.push(V {
                        p: c + rot * q,
                        n: rot * (n + (u * a + v * b) * 0.25).normalize(),
                        uv: Vec2::ZERO,
                        mat: m,
                        part: 0,
                        seam: 0.0,
                        grp,
                    });
                }
                let t = [first, first + 1, first + 2, first, first + 2, first + 3];
                // Wind outward.
                let (pa, pb, pc) = (
                    verts[t[0] as usize].p,
                    verts[t[1] as usize].p,
                    verts[t[2] as usize].p,
                );
                if (pb - pa).cross(pc - pa).dot(rot * n) < 0.0 {
                    idx.extend_from_slice(&[t[0], t[2], t[1], t[0], t[5], t[4]]);
                } else {
                    idx.extend_from_slice(&t);
                }
            }
        };
        for (name, x) in [("seat_fl", 0.38f32), ("seat_fr", -0.38)] {
            let g = gi(name);
            solid(
                &mut verts,
                &mut idx,
                Vec3::new(x, 0.53, 2.62),
                Vec3::new(0.24, 0.07, 0.25),
                0.0,
                mat::SEAT,
                g,
            );
            solid(
                &mut verts,
                &mut idx,
                Vec3::new(x, 0.86, 2.31),
                Vec3::new(0.23, 0.30, 0.07),
                -0.25,
                mat::SEAT,
                g,
            );
            solid(
                &mut verts,
                &mut idx,
                Vec3::new(x, 1.23, 2.22),
                Vec3::new(0.12, 0.08, 0.05),
                -0.25,
                mat::SEAT,
                g,
            );
        }
        let g = gi("seat_rear");
        solid(
            &mut verts,
            &mut idx,
            Vec3::new(0.0, 0.53, 1.80),
            Vec3::new(0.64, 0.07, 0.24),
            0.0,
            mat::SEAT,
            g,
        );
        solid(
            &mut verts,
            &mut idx,
            Vec3::new(0.0, 0.86, 1.52),
            Vec3::new(0.64, 0.30, 0.07),
            -0.35,
            mat::SEAT,
            g,
        );
        let g = gi("dash");
        solid(
            &mut verts,
            &mut idx,
            Vec3::new(0.0, 0.86, 3.22),
            Vec3::new(0.76, 0.11, 0.2),
            0.15,
            mat::DASH,
            g,
        );
        solid(
            &mut verts,
            &mut idx,
            Vec3::new(0.38, 0.98, 3.1),
            Vec3::new(0.17, 0.04, 0.08),
            0.0,
            mat::DASH,
            g,
        );
        let g = gi("tunnel");
        solid(
            &mut verts,
            &mut idx,
            Vec3::new(0.0, 0.45, 2.75),
            Vec3::new(0.11, 0.12, 0.42),
            0.0,
            mat::DASH,
            g,
        );
        // Steering wheel: a torus on the column, tilted toward the driver.
        let g = gi("column");
        let c = Vec3::new(0.38, 0.95, 2.97);
        let axis = Vec3::new(0.0, 0.45, -0.89).normalize();
        let s0 = Vec3::X;
        let s1 = axis.cross(s0).normalize();
        let first = verts.len() as u32;
        let (nu, nv2) = (36u32, 8u32);
        for a in 0..=nu {
            let th = 2.0 * std::f32::consts::PI * a as f32 / nu as f32;
            let dir = s0 * th.cos() + s1 * th.sin();
            for b in 0..=nv2 {
                let ph = 2.0 * std::f32::consts::PI * b as f32 / nv2 as f32;
                let n = dir * ph.cos() + axis * ph.sin();
                verts.push(V {
                    p: c + dir * 0.185 + n * 0.017,
                    n,
                    uv: Vec2::ZERO,
                    mat: mat::DASH,
                    part: 0,
                    seam: 0.0,
                    grp: g,
                });
            }
        }
        for a in 0..nu {
            for b in 0..nv2 {
                let i0 = first + a * (nv2 + 1) + b;
                let i1 = i0 + nv2 + 1;
                idx.extend_from_slice(&[i0, i1, i0 + 1, i0 + 1, i1, i1 + 1]);
            }
        }
        solid(
            &mut verts,
            &mut idx,
            c - axis * 0.03,
            Vec3::new(0.06, 0.05, 0.04),
            -0.47,
            mat::DASH,
            g,
        );
        // Mirror stalks.
        for side in [1.0f32, -1.0] {
            let (z, y) = (look.mirror[0], look.mirror[1]);
            let hw = prof.half_width.eval(z);
            let part = part_id(Vec3::new(side * hw, y - 0.05, z - 0.08), Seg::Side);
            let first = verts.len();
            solid(
                &mut verts,
                &mut idx,
                Vec3::new(side * (hw + 0.005), y + 0.01, z - 0.03),
                Vec3::new(0.035, 0.016, 0.03),
                0.0,
                mat::PAINT,
                -1,
            );
            for v in &mut verts[first..] {
                v.part = part;
            }
        }
    }
    // Bind every vertex to its part's skin triangles.
    let mut by_part: HashMap<u16, Vec<[u32; 3]>> = HashMap::new();
    for t in &sk.skin {
        let p = sk.nodes[t[0] as usize].part;
        if t.iter().all(|&i| sk.nodes[i as usize].part == p) {
            by_part.entry(p).or_default().push(*t);
        }
    }
    let mut out = ShowcaseVisual {
        node_count: node_pos.len() as u32,
        ..Default::default()
    };
    let fallback = Binder::new(sk.skin.clone(), &node_pos, 0.15);
    let mut binders: HashMap<u16, Binder> = HashMap::new();
    let mut group_binders: HashMap<i32, Binder> = HashMap::new();
    for v in &verts {
        if v.grp >= 0 {
            let gb = group_binders.entry(v.grp).or_insert_with(|| {
                Binder::new(
                    built.bind_tris[&groups[v.grp as usize]].clone(),
                    &node_pos,
                    0.25,
                )
            });
            if let Some((t, c, nl)) = gb.bind(v.p, v.n) {
                out.bind_nodes.push(t);
                out.bind_coords.push(c);
                out.bind_normal.push(nl);
                out.bind_rest.push(rest_edges(&node_pos, t));
                out.positions.push((v.p + built.shift).to_array());
                out.normals.push(v.n.to_array());
                out.uvs.push(v.uv.to_array());
                out.material.push(v.mat);
                out.colors.push(color(v.mat, paint));
                out.seam.push(0.0);
                continue;
            }
        }
        let b = binders.entry(v.part).or_insert_with(|| {
            Binder::new(
                by_part.get(&v.part).cloned().unwrap_or_default(),
                &node_pos,
                0.15,
            )
        });
        let Some((t, c, nl)) = b.bind(v.p, v.n).or_else(|| fallback.bind(v.p, v.n)) else {
            return Err(format!("no skin triangle for a vertex at {:?}", v.p));
        };
        out.bind_nodes.push(t);
        out.bind_coords.push(c);
        out.bind_normal.push(nl);
        out.bind_rest.push(rest_edges(&node_pos, t));
        out.positions.push((v.p + built.shift).to_array());
        out.normals.push(v.n.to_array());
        out.uvs.push(v.uv.to_array());
        out.material.push(v.mat);
        out.colors.push(color(v.mat, paint));
        out.seam.push(v.seam);
    }
    out.indices = idx;
    // Wheels: tyre and rim, bound to the tyre's rings.
    for tyre in &sim.tyres {
        wheel(&mut out, tyre, &node_pos, built.shift);
    }
    Ok(out)
}

fn rest_edges(pos: &[Vec3], t: [u32; 3]) -> [f32; 2] {
    let p0 = pos[t[0] as usize];
    [
        pos[t[1] as usize].distance(p0),
        pos[t[2] as usize].distance(p0),
    ]
}

/// A material's base colour (linear RGB).
pub fn color(m: u8, paint: [f32; 3]) -> [f32; 3] {
    match m {
        mat::PAINT => paint,
        mat::GLASS => [0.02, 0.025, 0.03],
        mat::CHROME => [0.75, 0.76, 0.78],
        mat::TRIM => [0.02, 0.02, 0.022],
        mat::HEADLIGHT => [0.85, 0.88, 0.9],
        mat::TAILLIGHT => [0.5, 0.015, 0.015],
        mat::TYRE => [0.025, 0.025, 0.025],
        mat::RIM => [0.55, 0.57, 0.6],
        mat::GAP => [0.1, 0.004, 0.004],
        mat::UNDERBODY => [0.04, 0.04, 0.04],
        mat::PLATE => [0.85, 0.85, 0.8],
        mat::GRILLE => [0.015, 0.015, 0.015],
        mat::SEAT => [0.05, 0.045, 0.04],
        mat::DASH => [0.03, 0.03, 0.032],
        mat::DOORCARD => [0.1, 0.095, 0.09],
        mat::HEADLINER => [0.42, 0.4, 0.37],
        mat::CARPET => [0.05, 0.05, 0.05],
        _ => [1.0, 0.0, 1.0],
    }
}

/// A tyre (tread, shoulders, sidewalls) and five-spoke rims, bound to the tyre's
/// fan triangles (an axle node and two consecutive ring nodes on each face).
fn wheel(out: &mut ShowcaseVisual, tyre: &Tyre, node_pos: &[Vec3], shift: Vec3) {
    let a0 = node_pos[tyre.axle[0] as usize];
    let a1 = node_pos[tyre.axle[1] as usize];
    let axis = (a1 - a0).normalize();
    let centre = 0.5 * (a0 + a1);
    let width = (a1 - a0).length();
    let rays = (tyre.count / 2) as usize;
    let ring = |i: usize, outer: bool| tyre.first + 2 * (i % rays) as u32 + u32::from(!outer);
    let mut tris = Vec::new();
    for i in 0..rays {
        tris.push([tyre.axle[0], ring(i, true), ring(i + 1, true)]);
        tris.push([tyre.axle[1], ring(i, false), ring(i + 1, false)]);
    }
    let binder = Binder::new(tris, node_pos, 0.1);
    let r = tyre.radius;
    let s0 = {
        let d = node_pos[ring(0, true) as usize] - a0;
        (d - axis * d.dot(axis)).normalize()
    };
    let s1 = axis.cross(s0);
    let dir = |a: f32| s0 * a.cos() + s1 * a.sin();
    let hw = 0.5 * width;
    // Profile: (radius, axial offset from the centre plane, material).
    let profile: [(f32, f32, u8); 13] = [
        (0.2, -hw * 0.95, mat::RIM),
        (0.212, -hw * 0.99, mat::TYRE),
        (0.26, -hw * 1.06, mat::TYRE),
        (0.30, -hw * 1.03, mat::TYRE),
        (r - 0.012, -hw * 0.92, mat::TYRE),
        (r, -hw * 0.6, mat::TYRE),
        (r + 0.003, 0.0, mat::TYRE),
        (r, hw * 0.6, mat::TYRE),
        (r - 0.012, hw * 0.92, mat::TYRE),
        (0.30, hw * 1.03, mat::TYRE),
        (0.26, hw * 1.06, mat::TYRE),
        (0.212, hw * 0.99, mat::TYRE),
        (0.2, hw * 0.95, mat::RIM),
    ];
    let segs = 56usize;
    let push = |out: &mut ShowcaseVisual, p: Vec3, n: Vec3, m: u8| -> u32 {
        let (t, c, nl) = binder.bind(p, n).expect("tyre binding");
        out.bind_nodes.push(t);
        out.bind_coords.push(c);
        out.bind_normal.push(nl);
        out.bind_rest.push(rest_edges(node_pos, t));
        out.positions.push((p + shift).to_array());
        out.normals.push(n.to_array());
        out.uvs.push([0.0, 0.0]);
        out.material.push(m);
        out.colors.push(color(m, [0.0; 3]));
        out.seam.push(0.0);
        (out.positions.len() - 1) as u32
    };
    let mut grid: Vec<Vec<u32>> = Vec::new();
    for j in 0..profile.len() {
        let mut row = Vec::new();
        for i in 0..=segs {
            let a = 2.0 * std::f32::consts::PI * i as f32 / segs as f32;
            let (rr, w, m) = profile[j];
            let p = centre + dir(a) * rr + axis * w;
            let (r0, w0, _) = profile[j.saturating_sub(1)];
            let (r1, w1, _) = profile[(j + 1).min(profile.len() - 1)];
            let tang = dir(a) * (r1 - r0) + axis * (w1 - w0);
            let n = tang.cross(dir(a).cross(axis)).normalize_or_zero();
            let n = if n.dot(dir(a) * rr + axis * w) < 0.0 {
                -n
            } else {
                n
            };
            row.push(push(out, p, n, m));
        }
        grid.push(row);
    }
    for j in 0..profile.len() - 1 {
        for i in 0..segs {
            let (a, b, c, d) = (
                grid[j][i],
                grid[j][i + 1],
                grid[j + 1][i + 1],
                grid[j + 1][i],
            );
            out.indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }
    // Rims on both faces: hub, five spokes over a dark dish, the lip.
    for sign in [-1.0f32, 1.0] {
        let plane = centre + axis * (sign * hw * 0.78);
        let radii = [0.0f32, 0.045, 0.07, 0.15, 0.185, 0.2];
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for (ri, &rr) in radii.iter().enumerate() {
            let mut row = Vec::new();
            for i in 0..=segs {
                let a = 2.0 * std::f32::consts::PI * i as f32 / segs as f32;
                let dish = if (2..4).contains(&ri) {
                    -sign * 0.025
                } else {
                    0.0
                };
                let p = plane + dir(a) * rr + axis * dish;
                let k = (a * 5.0 / (2.0 * std::f32::consts::PI)).fract();
                let spoke = (k - 0.5).abs() < 0.16;
                let m = if ri <= 1 {
                    mat::CHROME
                } else if (2..4).contains(&ri) && !spoke {
                    mat::TRIM
                } else {
                    mat::RIM
                };
                row.push(push(out, p, axis * sign, m));
            }
            rows.push(row);
        }
        for j in 0..rows.len() - 1 {
            for i in 0..segs {
                let (a, b, c, d) = (
                    rows[j][i],
                    rows[j][i + 1],
                    rows[j + 1][i + 1],
                    rows[j + 1][i],
                );
                if sign > 0.0 {
                    out.indices.extend_from_slice(&[a, b, c, a, c, d]);
                } else {
                    out.indices.extend_from_slice(&[a, c, b, a, d, c]);
                }
            }
        }
    }
}

/// Deform the rest mesh by node positions (the solver's order): the GPU
/// shader's formula on the CPU, for tests and offline renders.
pub fn deform(v: &ShowcaseVisual, nodes: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut pos = Vec::with_capacity(v.positions.len());
    let mut nrm = Vec::with_capacity(v.positions.len());
    for i in 0..v.positions.len() {
        let [a, b, c] = v.bind_nodes[i];
        let (p0, p1, p2) = (nodes[a as usize], nodes[b as usize], nodes[c as usize]);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let [al, be, h] = v.bind_coords[i];
        let [r1, r2] = v.bind_rest[i];
        if e1.length() > TEAR_STRETCH * r1 || e2.length() > TEAR_STRETCH * r2 {
            // Torn: collapse onto the triangle's first node (zero-area faces).
            pos.push(Vec3::splat(f32::NAN));
            nrm.push(Vec3::Y);
            continue;
        }
        pos.push(p0 + e1 * al + e2 * be + n * h);
        let t = e1.normalize_or_zero();
        let bb = n.cross(t);
        let [x, y, z] = v.bind_normal[i];
        nrm.push((t * x + bb * y + n * z).normalize_or_zero());
    }
    (pos, nrm)
}
