//! Render mesh of the procedural car (design: "Visual mesh"): a smooth, detailed
//! body made of separate parts that match the physics panels, with glass, lights,
//! a grille, wheel arches, an interior and an engine, every vertex bound to a
//! frame of three nodes of its own part ([`crate::flexbody`]). The physics mesh
//! stays coarse (about 150 nodes); this one has ten thousand-odd vertices.
//!
//! A part is bound only to its own nodes (a door to the door panel's nodes), so
//! a panel that tears off carries its mesh away and leaves the opening behind.

use crate::flexbody::{bind, BindOptions, FlexBinding};
use crate::procedural::{CarLayout, CarParams, CAR_RING};
use glam::{Vec2, Vec3};
use std::collections::HashMap;

/// Surface materials (`VisualMesh::tri_material`), shared with the renderer.
pub mod material {
    pub const PAINT: u8 = 0;
    pub const GLASS: u8 = 1;
    pub const CHROME: u8 = 2;
    pub const TRIM: u8 = 3;
    pub const HEADLIGHT: u8 = 4;
    pub const TAILLIGHT: u8 = 5;
    pub const UNDERBODY: u8 = 6;
    pub const INTERIOR: u8 = 7;
    pub const ENGINE: u8 = 8;
    pub const GRILLE: u8 = 9;
}

/// Parts (`VisualMesh::part`). Body parts first, then the panels in
/// [`CarLayout::panels`] order, then the bumpers and the inner parts.
pub mod part {
    pub const BODY: u8 = 0;
    pub const BONNET: u8 = 1;
    pub const BOOT: u8 = 2;
    pub const DOOR_FL: u8 = 3;
    pub const DOOR_RL: u8 = 4;
    pub const DOOR_FR: u8 = 5;
    pub const DOOR_RR: u8 = 6;
    pub const BUMPER_F: u8 = 7;
    pub const BUMPER_R: u8 = 8;
    pub const INTERIOR: u8 = 9;
    pub const ENGINE: u8 = 10;
    pub const ROOF: u8 = 11;
    /// The crash-test car's bay components and crash rails (`CarParams::components`, `rails`).
    pub const GEARBOX: u8 = 12;
    pub const RADIATOR: u8 = 13;
    pub const BATTERY: u8 = 14;
    pub const RAILS: u8 = 15;
}

/// The procedural car's part name ([`crate::procedural::car_desc`]) for each
/// visual part id ([`part`]): the mesh a part binds to. The interior rides the body.
pub const PART_NAMES: [&str; 16] = [
    "body",
    "bonnet",
    "boot",
    "door_fl",
    "door_rl",
    "door_fr",
    "door_rr",
    "bumper_front",
    "bumper_rear",
    "body",
    "engine",
    "roof",
    "gearbox",
    "radiator",
    "battery",
    "rails",
];

#[derive(Clone, Debug, Default)]
pub struct VisualMesh {
    /// Rest positions (desc space).
    pub vertices: Vec<Vec3>,
    pub bindings: Vec<FlexBinding>,
    pub part: Vec<u8>,
    /// The vertex lies on a seam between two parts (panel gap shading).
    pub seam: Vec<bool>,
    pub tris: Vec<[u32; 3]>,
    pub tri_material: Vec<u8>,
}

/// Roof height of the procedural car's profile (procedural::roof).
fn roof(t: f32) -> f32 {
    let pts = [
        (0.0, 0.92),
        (0.12, 0.98),
        (0.28, 1.38),
        (0.55, 1.42),
        (0.72, 1.02),
        (0.85, 0.9),
        (1.0, 0.82),
    ];
    for w in pts.windows(2) {
        let ((t0, y0), (t1, y1)) = (w[0], w[1]);
        if t <= t1 {
            let s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let s = s * s * (3.0 - 2.0 * s);
            return y0 + (y1 - y0) * s;
        }
    }
    0.82
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Open Catmull–Rom through `pts` at `u` ∈ [0, 1) (ends tangent to their neighbours).
fn open_curve(pts: &[Vec2], u: f32) -> Vec2 {
    let n = pts.len();
    if n == 2 {
        return pts[0].lerp(pts[1], u);
    }
    let x = u * (n - 1) as f32;
    let k = (x.floor() as usize).min(n - 2);
    let t = x - k as f32;
    let p1 = pts[k];
    let p2 = pts[k + 1];
    let p0 = if k == 0 { p1 * 2.0 - p2 } else { pts[k - 1] };
    let p3 = if k + 2 < n { pts[k + 2] } else { p2 * 2.0 - p1 };
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Superellipsoid "rounded box" mesh: (vertices, triangles), outward wound.
fn rounded_box(
    centre: Vec3,
    half: Vec3,
    e: f32,
    lat: usize,
    lon: usize,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let f = |a: f32, p: f32| a.signum() * a.abs().powf(p);
    let mut v = Vec::new();
    for i in 0..=lat {
        let th = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / lat as f32;
        for j in 0..lon {
            let ph = std::f32::consts::TAU * j as f32 / lon as f32;
            let x = f(th.cos(), e) * f(ph.cos(), e);
            let z = f(th.cos(), e) * f(ph.sin(), e);
            let y = f(th.sin(), e);
            v.push(centre + Vec3::new(x, y, z) * half);
        }
    }
    let mut t = Vec::new();
    for i in 0..lat {
        for j in 0..lon {
            let a = (i * lon + j) as u32;
            let b = (i * lon + (j + 1) % lon) as u32;
            let c = ((i + 1) * lon + j) as u32;
            let d = ((i + 1) * lon + (j + 1) % lon) as u32;
            t.push([a, c, b]);
            t.push([b, c, d]);
        }
    }
    (v, t)
}

/// Torus around the axis `axis` (a steering wheel).
fn torus(
    centre: Vec3,
    axis: Vec3,
    r: f32,
    tube: f32,
    n: usize,
    m: usize,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let a = axis.normalize();
    let u = a.any_orthonormal_vector();
    let w = a.cross(u);
    let mut v = Vec::new();
    for i in 0..n {
        let th = std::f32::consts::TAU * i as f32 / n as f32;
        let dir = u * th.cos() + w * th.sin();
        for j in 0..m {
            let ph = std::f32::consts::TAU * j as f32 / m as f32;
            v.push(centre + dir * (r + tube * ph.cos()) + a * (tube * ph.sin()));
        }
    }
    let mut t = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let i1 = (i + 1) % n;
            let j1 = (j + 1) % m;
            let q = |i: usize, j: usize| (i * m + j) as u32;
            t.push([q(i, j), q(i1, j), q(i1, j1)]);
            t.push([q(i, j), q(i1, j1), q(i, j1)]);
        }
    }
    (v, t)
}

struct Builder {
    mesh: VisualMesh,
    /// Grid vertex (station, sample, part) → index.
    grid: HashMap<(usize, usize, u8), u32>,
}

impl Builder {
    fn push_vertex(&mut self, p: Vec3, part: u8) -> u32 {
        self.mesh.vertices.push(p);
        self.mesh.part.push(part);
        self.mesh.seam.push(false);
        self.mesh.vertices.len() as u32 - 1
    }

    fn add_mesh(&mut self, (v, t): (Vec<Vec3>, Vec<[u32; 3]>), part: u8, mat: u8) {
        let base = self.mesh.vertices.len() as u32;
        for p in v {
            self.push_vertex(p, part);
        }
        for tri in t {
            self.mesh.tris.push(tri.map(|i| base + i));
            self.mesh.tri_material.push(mat);
        }
    }
}

/// The procedural car's render mesh, bound to `nodes` (the desc's rest node
/// positions) with `layout` from [`crate::procedural::car_truck_layout`].
pub fn car_visual(p: &CarParams, layout: &CarLayout, nodes: &[Vec3]) -> VisualMesh {
    use material::*;
    let n = layout.stations;
    let len = p.length;
    let hw = 0.5 * p.width;
    let ring_z = |k: usize| -0.5 * len + k as f32 / (n - 1) as f32 * len;
    let t_of = |z: f32| ((z + 0.5 * len) / len).clamp(0.0, 1.0);
    // The design's lines (m, in the desc frame): the window sill, the shoulder crease, the
    // sill, and where the glass starts and ends along the profile ([`roof`], the cage's).
    let belt = 0.93;
    let shoulder = 0.80;
    let sill = 0.31;
    let windscreen = (0.558, 0.716);
    let rear_window = (0.128, 0.276);
    // Half width at t: the cage's taper, with the corners rounded in plan.
    let half_width = |t: f32| {
        let taper = 1.0 - 0.08 * (2.0 * t - 1.0).abs().powi(4);
        let corners = 1.0 - 0.06 * smoothstep(0.9, 1.0, t) - 0.05 * smoothstep(0.08, 0.0, t);
        hw * taper * corners
    };
    // How much cabin rises above the sill at t (0 on the bonnet and boot lid, 1 at the roof).
    let cabin = |t: f32| smoothstep(belt + 0.03, belt + 0.2, roof(t));

    // Cross-section at t, the left half from the keel round to the roof's centre, as five
    // runs joined at creases: underbody, body side, shoulder, glasshouse side, top.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Run {
        Under,
        Side,
        Shoulder,
        Glasshouse,
        Top,
    }
    const RUNS: [(Run, usize); 5] = [
        (Run::Under, 6),
        (Run::Side, 16),
        (Run::Shoulder, 5),
        (Run::Glasshouse, 9),
        (Run::Top, 12),
    ];
    let half_count: usize = RUNS.iter().map(|r| r.1).sum();
    let m = 2 * half_count;
    let section = |t: f32| -> Vec<(Vec2, Run)> {
        let sw = half_width(t);
        let top = roof(t);
        let g = cabin(t);
        let tw = hw * 0.8 * (1.0 - 0.08 * (2.0 * t - 1.0).abs().powi(4));
        let deck_edge = Vec2::new(sw - 0.07, top - 0.02);
        let sill_edge = Vec2::new(sw - 0.05, belt);
        let p6 = deck_edge.lerp(sill_edge, g);
        let p7 = Vec2::new(
            (sw - 0.11) + (tw - (sw - 0.11)) * g,
            top - (0.012 + 0.035 * g),
        );
        let ctrl: [(Run, Vec<Vec2>); 5] = [
            (
                Run::Under,
                vec![Vec2::new(0.0, 0.2), Vec2::new(sw - 0.16, 0.2), Vec2::new(sw - 0.03, 0.25)],
            ),
            (
                Run::Side,
                vec![
                    Vec2::new(sw - 0.03, 0.25),
                    Vec2::new(sw, sill),
                    Vec2::new(sw + 0.014, 0.56),
                    Vec2::new(sw + 0.004, shoulder),
                ],
            ),
            (Run::Shoulder, vec![Vec2::new(sw + 0.004, shoulder), p6]),
            (Run::Glasshouse, vec![p6, p7]),
            (
                Run::Top,
                vec![p7, Vec2::new(0.55 * p7.x, top - 0.004), Vec2::new(0.0, top)],
            ),
        ];
        let mut half = Vec::with_capacity(half_count);
        for ((run, k), (_, pts)) in RUNS.iter().zip(ctrl.iter()) {
            for i in 0..*k {
                let u = i as f32 / *k as f32;
                half.push((open_curve(pts, u), *run));
            }
        }
        // The right half, mirrored, from the roof's centre back down to the keel.
        let mut out = half.clone();
        out.push((Vec2::new(0.0, top), Run::Top));
        for i in (1..half_count).rev() {
            let (q, _) = half[i];
            out.push((Vec2::new(-q.x, q.y), half[i - 1].1));
        }
        out.truncate(m);
        out
    };
    // Stations: the cage's length, then a nose and a tail that round over and close.
    let s_count = 96;
    let mut stations: Vec<(f32, f32)> = (0..s_count)
        .map(|i| (-0.5 * len + i as f32 / (s_count - 1) as f32 * len, 1.0))
        .collect();
    for (dz, sc) in [(0.04, 0.985), (0.075, 0.95), (0.1, 0.88), (0.115, 0.76)] {
        stations.insert(0, (-0.5 * len - dz, sc));
        stations.push((0.5 * len + dz, sc));
    }
    let z_of = |k: usize| ring_z(k.min(n - 1));
    let z_mid = z_of(layout.mid);
    let bonnet_z = (z_of(layout.front_bay) - 0.5 * (len / (n - 1) as f32)).max(-0.5 * len + windscreen.1 * len);
    let boot_z = (z_of(layout.rear_bay) + 0.5 * (len / (n - 1) as f32)).min(-0.5 * len + rear_window.0 * len);
    let arch_r = 0.405;
    let in_arch = |z: f32, y: f32| {
        layout
            .axles
            .iter()
            .any(|&a| Vec2::new(z - a, y - layout.axle_y).length() < arch_r - 0.01)
    };
    // Doors: the front one from the B-pillar to ahead of the front arch, the rear one from the
    // B-pillar to ahead of the rear arch.
    let front_arch = layout.axles[0] - arch_r;
    let rear_arch = layout.axles[1] + arch_r;
    let door_f = (z_mid + 0.015, front_arch - 0.02);
    let door_r = (rear_arch + 0.06, z_mid - 0.015);

    let mut b = Builder {
        mesh: VisualMesh::default(),
        grid: HashMap::new(),
    };
    let grid: Vec<Vec<(Vec3, Run)>> = stations
        .iter()
        .map(|&(z, sc)| {
            // Past the cage's ends the section shrinks toward the bumper's middle.
            let c = Vec2::new(0.0, 0.55);
            section(t_of(z))
                .into_iter()
                .map(|(q, run)| {
                    let q = c + (q - c) * Vec2::new(sc, sc.powf(1.6));
                    (Vec3::new(q.x, q.y, z), run)
                })
                .collect()
        })
        .collect();
    // Wheel arches: side and underbody points inside the arch circle (side view) move onto
    // it, so the opening has a clean edge, and its lip flares out a little.
    let grid: Vec<Vec<(Vec3, Run)>> = grid
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(q, run)| {
                    let mut q = q;
                    if matches!(run, Run::Side | Run::Under) && q.x.abs() > hw * 0.55 {
                        for &a in &layout.axles {
                            let d = Vec2::new(q.z - a, q.y - layout.axle_y);
                            let r = d.length();
                            if r < arch_r && d.y > -0.2 {
                                let e = d.normalize_or_zero() * arch_r;
                                let e = if e.y < -0.05 {
                                    Vec2::new(e.x.signum() * arch_r, -0.05)
                                } else {
                                    e
                                };
                                q.z = a + e.x;
                                q.y = layout.axle_y + e.y;
                            }
                            let flare = smoothstep(arch_r + 0.12, arch_r, r) * 0.018;
                            q.x += q.x.signum() * flare;
                        }
                    }
                    (q, run)
                })
                .collect()
        })
        .collect();
    for i in 0..stations.len() - 1 {
        for j in 0..m {
            let j1 = (j + 1) % m;
            let q = [
                grid[i][j].0,
                grid[i + 1][j].0,
                grid[i + 1][j1].0,
                grid[i][j1].0,
            ];
            let c = (q[0] + q[1] + q[2] + q[3]) * 0.25;
            let run = grid[i][j].1;
            let t = t_of(c.z);
            let left = c.x > 0.0;
            let past_ends = c.z.abs() > 0.5 * len;
            if matches!(run, Run::Side | Run::Under) && c.x.abs() > hw * 0.55 && in_arch(c.z, c.y) {
                continue;
            }
            // Distance into the top run from the pillar edge (the A- and C-pillars' width).
            let from_edge = {
                let k = j.min(m - 1 - j);
                let first_top = half_count - RUNS[4].1;
                k.saturating_sub(first_top)
            };
            let top = roof(t);
            let (part, mat) = match run {
                Run::Under => (part::BODY, UNDERBODY),
                Run::Side => {
                    let door = if past_ends {
                        None
                    } else if (door_f.0..door_f.1).contains(&c.z) {
                        Some(if left { part::DOOR_FL } else { part::DOOR_FR })
                    } else if (door_r.0..door_r.1).contains(&c.z) {
                        Some(if left { part::DOOR_RL } else { part::DOOR_RR })
                    } else {
                        None
                    };
                    let mat = if c.y < sill - 0.02 { TRIM } else { PAINT };
                    match door {
                        Some(d) if !layout.panels.is_empty() => (d, mat),
                        _ => (part::BODY, mat),
                    }
                }
                Run::Shoulder if c.z >= bonnet_z && !past_ends => (part::BONNET, PAINT),
                Run::Shoulder if c.z <= boot_z && !past_ends => (part::BOOT, PAINT),
                Run::Shoulder => (part::BODY, PAINT),
                Run::Glasshouse => {
                    let b_pillar = (c.z - z_mid).abs() < 0.045;
                    let glass = cabin(t) > 0.3
                        && c.y > belt + 0.02
                        && c.y < top - 0.07
                        && !past_ends;
                    if b_pillar && glass {
                        (part::BODY, TRIM)
                    } else if glass {
                        (part::BODY, GLASS)
                    } else if c.z >= bonnet_z && !past_ends {
                        (part::BONNET, PAINT)
                    } else if c.z <= boot_z && !past_ends {
                        (part::BOOT, PAINT)
                    } else {
                        (part::BODY, PAINT)
                    }
                }
                Run::Top => {
                    let pillar = from_edge < 2;
                    if past_ends {
                        (part::BODY, PAINT)
                    } else if c.z >= bonnet_z {
                        (part::BONNET, PAINT)
                    } else if c.z <= boot_z {
                        (part::BOOT, PAINT)
                    } else if (windscreen.0..windscreen.1).contains(&t)
                        || (rear_window.0..rear_window.1).contains(&t)
                    {
                        (part::BODY, if pillar { PAINT } else { GLASS })
                    } else if !layout.roof.is_empty()
                        && c.z > z_of(layout.cabin_first)
                        && c.z < z_of(layout.cabin_last)
                    {
                        // The roof panel over the passenger cell: bound to the roof nodes, so
                        // a sheared-off roof carries its mesh away.
                        (part::ROOF, PAINT)
                    } else {
                        (part::BODY, PAINT)
                    }
                }
            };
            let part = if layout.panels.is_empty() && matches!(part, part::BONNET | part::BOOT) {
                part::BODY
            } else {
                part
            };
            let mut idx = [0u32; 4];
            for (k, &(si, sj)) in [(i, j), (i + 1, j), (i + 1, j1), (i, j1)]
                .iter()
                .enumerate()
            {
                let key = (si, sj, part);
                idx[k] = match b.grid.get(&key) {
                    Some(&v) => v,
                    None => {
                        let v = b.push_vertex(grid[si][sj].0, part);
                        b.grid.insert(key, v);
                        v
                    }
                };
            }
            for tri in [[idx[0], idx[1], idx[2]], [idx[0], idx[2], idx[3]]] {
                b.mesh.tris.push(tri);
                b.mesh.tri_material.push(mat);
            }
        }
    }
    // Seams: grid points shared by more than one part.
    let mut owners: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    for (&(si, sj, _), &v) in &b.grid {
        owners.entry((si, sj)).or_default().push(v);
    }
    for vs in owners.values() {
        if vs.len() > 1 {
            for &v in vs {
                b.mesh.seam[v as usize] = true;
            }
        }
    }
    // The nose and tail close with a cap behind the bumpers and lamps.
    for (end, front) in [(stations.len() - 1, true), (0usize, false)] {
        let ring = &grid[end];
        let centre = ring.iter().map(|r| r.0).sum::<Vec3>() / m as f32
            + Vec3::Z * if front { 0.01 } else { -0.01 };
        let cv = b.push_vertex(centre, part::BODY);
        let rv: Vec<u32> = ring
            .iter()
            .map(|r| b.push_vertex(r.0, part::BODY))
            .collect();
        for j in 0..m {
            b.mesh.tris.push([cv, rv[j], rv[(j + 1) % m]]);
            b.mesh.tri_material.push(PAINT);
        }
    }
    // Outward winding for the shell: flip any triangle facing the car's long axis. (The
    // closed meshes added below are wound by construction.)
    for (tri, &mat) in b.mesh.tris.iter_mut().zip(&b.mesh.tri_material) {
        let _ = mat;
        let [a, bb, c] = tri.map(|i| b.mesh.vertices[i as usize]);
        let nrm = (bb - a).cross(c - a);
        let cen = (a + bb + c) / 3.0;
        let axis = Vec3::new(0.0, 0.62, cen.z.clamp(-0.5 * len + 0.3, 0.5 * len - 0.3));
        if nrm.dot(cen - axis) < 0.0 {
            tri.swap(1, 2);
        }
    }

    let fz = 0.5 * len;
    let rounded = |c: Vec3, h: Vec3, e: f32| rounded_box(c, h, e, 8, 20);
    // Front: a body-coloured bumper cover with a black lip and intake, a grille in a chrome
    // surround, and headlamps at the corners under the bonnet's leading edge.
    b.add_mesh(rounded(Vec3::new(0.0, 0.43, fz + 0.06), Vec3::new(hw * 0.95, 0.15, 0.12), 0.22), part::BUMPER_F, PAINT);
    b.add_mesh(rounded(Vec3::new(0.0, 0.29, fz + 0.1), Vec3::new(hw * 0.9, 0.03, 0.09), 0.3), part::BUMPER_F, TRIM);
    b.add_mesh(rounded(Vec3::new(0.0, 0.38, fz + 0.165), Vec3::new(0.42, 0.045, 0.025), 0.3), part::BUMPER_F, TRIM);
    b.add_mesh(rounded(Vec3::new(0.0, 0.65, fz + 0.06), Vec3::new(0.34, 0.085, 0.05), 0.25), part::BODY, CHROME);
    b.add_mesh(rounded(Vec3::new(0.0, 0.65, fz + 0.075), Vec3::new(0.31, 0.068, 0.05), 0.2), part::BODY, GRILLE);
    for s in [1.0f32, -1.0] {
        b.add_mesh(rounded(Vec3::new(s * 0.57, 0.7, fz + 0.035), Vec3::new(0.2, 0.06, 0.07), 0.3), part::BODY, HEADLIGHT);
    }
    // Rear: bumper, diffuser, wrap-round tail lamps, a plate recess and an exhaust.
    b.add_mesh(rounded(Vec3::new(0.0, 0.45, -fz - 0.06), Vec3::new(hw * 0.95, 0.15, 0.12), 0.22), part::BUMPER_R, PAINT);
    b.add_mesh(rounded(Vec3::new(0.0, 0.29, -fz - 0.1), Vec3::new(hw * 0.88, 0.03, 0.09), 0.3), part::BUMPER_R, TRIM);
    b.add_mesh(rounded(Vec3::new(0.0, 0.56, -fz - 0.16), Vec3::new(0.25, 0.055, 0.02), 0.3), part::BUMPER_R, TRIM);
    for s in [1.0f32, -1.0] {
        b.add_mesh(rounded(Vec3::new(s * 0.6, 0.79, -fz - 0.02), Vec3::new(0.2, 0.065, 0.075), 0.3), part::BODY, TAILLIGHT);
    }
    b.add_mesh(
        torus(Vec3::new(0.5, 0.27, -fz - 0.14), Vec3::Z, 0.035, 0.008, 16, 6),
        part::BUMPER_R,
        CHROME,
    );
    // Mirrors on the front doors at the A-pillar's foot, and a handle on every door.
    let mirror_z = -0.5 * len + windscreen.1 * len - 0.12;
    for (s, door) in [(1.0f32, part::DOOR_FL), (-1.0, part::DOOR_FR)] {
        let x = half_width(t_of(mirror_z)) + 0.07;
        b.add_mesh(rounded(Vec3::new(s * x, belt + 0.06, mirror_z), Vec3::new(0.08, 0.05, 0.045), 0.35), door, PAINT);
    }
    for (s, door, z) in [
        (1.0f32, part::DOOR_FL, door_f.0 + 0.16),
        (-1.0, part::DOOR_FR, door_f.0 + 0.16),
        (1.0, part::DOOR_RL, door_r.0 + 0.18),
        (-1.0, part::DOOR_RR, door_r.0 + 0.18),
    ] {
        let x = half_width(t_of(z)) + 0.012;
        b.add_mesh(rounded(Vec3::new(s * x, 0.86, z), Vec3::new(0.012, 0.016, 0.07), 0.35), door, CHROME);
    }
    // Arch liners: black bands over each wheel, seen from both sides.
    for &a in &layout.axles {
        for s in [1.0f32, -1.0] {
            let (k, l) = (24, 3);
            let x_out = half_width(t_of(a)) - 0.005;
            let x_in = x_out - 0.3;
            let mut v = Vec::new();
            for i in 0..=k {
                let th = -0.12 + (std::f32::consts::PI + 0.24) * i as f32 / k as f32;
                for w in 0..=l {
                    let x = x_in + (x_out - x_in) * w as f32 / l as f32;
                    v.push(Vec3::new(s * x, layout.axle_y + (arch_r - 0.01) * th.sin(), a + (arch_r - 0.01) * th.cos()));
                }
            }
            let mut tris = Vec::new();
            for i in 0..k {
                for w in 0..l {
                    let q = |i: usize, w: usize| (i * (l + 1) + w) as u32;
                    for t in [[q(i, w), q(i + 1, w), q(i + 1, w + 1)], [q(i, w), q(i + 1, w + 1), q(i, w + 1)]] {
                        tris.push(t);
                        tris.push([t[0], t[2], t[1]]);
                    }
                }
            }
            b.add_mesh((v, tris), part::BODY, TRIM);
        }
    }

    // Interior: seats, a dashboard, a steering wheel (seen through glass and
    // through torn-open doors).
    let zc = |k: usize| ring_z(k);
    let front_seat_z = 0.5 * (zc(layout.mid) + zc(layout.cabin_last));
    let rear_seat_z = zc(layout.cabin_first);
    for x in [0.38f32, -0.38] {
        b.add_mesh(
            rounded_box(
                Vec3::new(x, 0.45, front_seat_z),
                Vec3::new(0.25, 0.1, 0.25),
                0.3,
                6,
                12,
            ),
            part::INTERIOR,
            INTERIOR,
        );
        b.add_mesh(
            rounded_box(
                Vec3::new(x, 0.8, front_seat_z - 0.27),
                Vec3::new(0.24, 0.3, 0.07),
                0.3,
                6,
                12,
            ),
            part::INTERIOR,
            INTERIOR,
        );
    }
    b.add_mesh(
        rounded_box(
            Vec3::new(0.0, 0.45, rear_seat_z),
            Vec3::new(0.6, 0.1, 0.24),
            0.3,
            6,
            12,
        ),
        part::INTERIOR,
        INTERIOR,
    );
    b.add_mesh(
        rounded_box(
            Vec3::new(0.0, 0.78, rear_seat_z - 0.26),
            Vec3::new(0.6, 0.28, 0.07),
            0.3,
            6,
            12,
        ),
        part::INTERIOR,
        INTERIOR,
    );
    let dash_z = zc(layout.cabin_last) + 0.25;
    b.add_mesh(
        rounded_box(
            Vec3::new(0.0, 0.86, dash_z),
            Vec3::new(hw * 0.88, 0.1, 0.18),
            0.35,
            6,
            16,
        ),
        part::INTERIOR,
        INTERIOR,
    );
    b.add_mesh(
        torus(
            Vec3::new(0.38, 0.95, dash_z - 0.22),
            Vec3::new(0.0, 0.45, 1.0),
            0.17,
            0.018,
            20,
            6,
        ),
        part::INTERIOR,
        TRIM,
    );
    // Engine: block and head in the front bay (seen when the bonnet goes).
    let bay_c = 0.5 * (zc(layout.front_bay) + zc(n - 1));
    b.add_mesh(
        rounded_box(
            Vec3::new(0.0, 0.55, bay_c),
            Vec3::new(0.32, 0.2, 0.26),
            0.2,
            6,
            16,
        ),
        part::ENGINE,
        ENGINE,
    );
    b.add_mesh(
        rounded_box(
            Vec3::new(0.0, 0.78, bay_c),
            Vec3::new(0.26, 0.05, 0.2),
            0.3,
            4,
            12,
        ),
        part::ENGINE,
        CHROME,
    );

    // The bay's components and the crash rails, shaped from their own nodes (so they sit
    // exactly on them and move with them).
    let span = |ids: &[u32]| {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for &i in ids {
            lo = lo.min(nodes[i as usize]);
            hi = hi.max(nodes[i as usize]);
        }
        (0.5 * (lo + hi), 0.5 * (hi - lo))
    };
    if !layout.gearbox.is_empty() {
        let (c, h) = span(&layout.gearbox);
        b.add_mesh(rounded_box(c, h, 0.3, 6, 16), part::GEARBOX, ENGINE);
        // The bell housing, against the engine.
        b.add_mesh(
            rounded_box(c + Vec3::new(0.0, 0.02, h.z * 0.9), Vec3::new(h.x * 1.2, h.y * 1.15, 0.05), 0.2, 6, 16),
            part::GEARBOX,
            ENGINE,
        );
    }
    if !layout.battery.is_empty() {
        let (c, h) = span(&layout.battery);
        b.add_mesh(rounded_box(c, h, 0.15, 6, 16), part::BATTERY, TRIM);
        for dz in [-0.6f32, 0.6] {
            b.add_mesh(
                rounded_box(c + Vec3::new(0.0, h.y + 0.015, dz * h.z), Vec3::new(0.018, 0.018, 0.018), 0.4, 4, 8),
                part::BATTERY,
                CHROME,
            );
        }
    }
    if !layout.radiator.is_empty() {
        let (c, h) = span(&layout.radiator);
        // The core (dark fins) in a metal frame, a few cm deep.
        b.add_mesh(rounded_box(c, Vec3::new(h.x, h.y, 0.03), 0.1, 6, 16), part::RADIATOR, GRILLE);
        b.add_mesh(
            rounded_box(c + Vec3::Y * (h.y + 0.02), Vec3::new(h.x + 0.03, 0.02, 0.04), 0.2, 4, 12),
            part::RADIATOR,
            ENGINE,
        );
        b.add_mesh(
            rounded_box(c - Vec3::Y * (h.y + 0.02), Vec3::new(h.x + 0.03, 0.02, 0.04), 0.2, 4, 12),
            part::RADIATOR,
            ENGINE,
        );
    }
    if !layout.rails.is_empty() {
        // Each rail a box tube through its ring corners (8 nodes a ring: left box, then right).
        let rings = layout.rails.len() / 8;
        for side in 0..2 {
            let corner = |r: usize, c: usize| nodes[layout.rails[r * 8 + side * 4 + c] as usize];
            // Corners in order round the section: inner lower, outer lower, outer upper, inner upper.
            const ROUND: [usize; 4] = [0, 2, 3, 1];
            let base = b.mesh.vertices.len() as u32;
            for r in 0..rings {
                for &c in &ROUND {
                    b.push_vertex(corner(r, c), part::RAILS);
                }
            }
            for r in 0..rings - 1 {
                for e in 0..4 {
                    let q = |rr: usize, ee: usize| base + (rr * 4 + ee % 4) as u32;
                    let (a, bb, c, d) = (q(r, e), q(r + 1, e), q(r + 1, e + 1), q(r, e + 1));
                    // Both windings: the rail is seen from any side through a torn body.
                    for tri in [[a, bb, c], [a, c, d], [a, c, bb], [a, d, c]] {
                        b.mesh.tris.push(tri);
                        b.mesh.tri_material.push(UNDERBODY);
                    }
                }
            }
        }
    }

    // Bind each part to its own nodes.
    let ring_nodes = (n * CAR_RING) as u32;
    let shell: Vec<u32> = (0..ring_nodes).collect();
    let rings = |ks: std::ops::RangeInclusive<usize>, js: &[usize]| -> Vec<u32> {
        ks.flat_map(|k| js.iter().map(move |&j| (k * CAR_RING + j) as u32))
            .collect()
    };
    let allowed = |part: u8| -> Vec<u32> {
        match part {
            part::BONNET..=part::DOOR_RR => layout
                .panels
                .get((part - 1) as usize)
                .cloned()
                .unwrap_or_else(|| shell.clone()),
            part::BUMPER_F if !layout.front_bumper.is_empty() => layout.front_bumper.clone(),
            part::BUMPER_R if !layout.rear_bumper.is_empty() => layout.rear_bumper.clone(),
            part::BUMPER_F => rings(n - 1..=n - 1, &[0, 1, 2, 3, 6]),
            part::BUMPER_R => rings(0..=0, &[0, 1, 2, 3, 6]),
            part::INTERIOR => rings(
                layout.cabin_first.saturating_sub(1)..=layout.cabin_last + 1,
                &[0, 1, 2, 3, 6],
            ),
            part::ENGINE if !layout.engine.is_empty() => layout.engine.clone(),
            part::GEARBOX => layout.gearbox.clone(),
            part::RADIATOR => layout.radiator.clone(),
            part::BATTERY => layout.battery.clone(),
            part::RAILS => layout.rails.clone(),
            part::ROOF => layout.roof.clone(),
            part::ENGINE => rings(
                layout.front_bay.saturating_sub(1)..=n - 1,
                &[0, 1, 2, 3, 6, 7],
            ),
            _ => shell.clone(),
        }
    };
    let opts = BindOptions {
        candidates: 12,
        max_edge: 1.6,
    };
    let mut bindings = vec![None; b.mesh.vertices.len()];
    for pt in 0..=part::RAILS {
        let ids: Vec<usize> = (0..b.mesh.vertices.len())
            .filter(|&i| b.mesh.part[i] == pt)
            .collect();
        if ids.is_empty() {
            continue;
        }
        let verts: Vec<Vec3> = ids.iter().map(|&i| b.mesh.vertices[i]).collect();
        let pool = allowed(pt);
        let got = bind(&verts, nodes, Some(&pool), opts);
        let fallback = bind(&verts, nodes, None, opts);
        for ((&i, g), f) in ids.iter().zip(got).zip(fallback) {
            bindings[i] = g.or(f);
        }
    }
    b.mesh.bindings = bindings
        .into_iter()
        .zip(&b.mesh.vertices)
        .map(|(bd, v)| {
            bd.unwrap_or_else(|| {
                // Degenerate: collapse onto the nearest node.
                let c = nodes
                    .iter()
                    .enumerate()
                    .min_by(|a, bb| {
                        (*a.1 - *v)
                            .length_squared()
                            .total_cmp(&(*bb.1 - *v).length_squared())
                    })
                    .map_or(0, |(i, _)| i as u32);
                FlexBinding {
                    nodes: [c; 3],
                    coords: [0.0; 3],
                }
            })
        })
        .collect();
    b.mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::car_truck_layout;
    use crate::truck::load_truck;

    #[test]
    fn car_visual_binds_every_part_to_its_own_nodes() {
        let p = CarParams::default();
        let (text, layout) = car_truck_layout(&p);
        let d = load_truck(&text);
        let nodes: Vec<Vec3> = d
            .nodes
            .iter()
            .map(|n| Vec3::from_array(n.position))
            .collect();
        let v = car_visual(&p, &layout, &nodes);
        assert!(v.vertices.len() > 5_000, "{} vertices", v.vertices.len());
        assert_eq!(v.tris.len(), v.tri_material.len());
        // Every vertex is reproduced exactly by its binding at rest.
        for (i, (bnd, pos)) in v.bindings.iter().zip(&v.vertices).enumerate() {
            if bnd.nodes[0] == bnd.nodes[1] {
                continue;
            }
            let got = bnd.skin(&nodes);
            assert!(
                (got - *pos).length() < 1e-3,
                "vertex {i} part {}: {got} vs {pos}",
                v.part[i]
            );
        }
        // A door's vertices use only that door's panel nodes.
        let door: std::collections::HashSet<u32> = layout.panels[2].iter().copied().collect();
        let fl: Vec<usize> = (0..v.vertices.len())
            .filter(|&i| v.part[i] == part::DOOR_FL)
            .collect();
        assert!(!fl.is_empty());
        assert!(fl
            .iter()
            .all(|&i| v.bindings[i].nodes.iter().all(|n| door.contains(n))));
        // Glass, lights and seams exist.
        for mat in [
            material::GLASS,
            material::HEADLIGHT,
            material::TAILLIGHT,
            material::GRILLE,
        ] {
            assert!(v.tri_material.contains(&mat), "material {mat}");
        }
        assert!(v.seam.iter().any(|&s| s));
    }

    #[test]
    fn the_sedan_binds_its_components_and_rails_to_their_own_nodes() {
        let p = CarParams::sedan();
        let (text, layout) = car_truck_layout(&p);
        let d = load_truck(&text);
        let nodes: Vec<Vec3> = d.nodes.iter().map(|n| Vec3::from_array(n.position)).collect();
        let v = car_visual(&p, &layout, &nodes);
        for (pt, own) in [
            (part::GEARBOX, &layout.gearbox),
            (part::RADIATOR, &layout.radiator),
            (part::BATTERY, &layout.battery),
            (part::RAILS, &layout.rails),
        ] {
            let own: std::collections::HashSet<u32> = own.iter().copied().collect();
            let ids: Vec<usize> = (0..v.vertices.len()).filter(|&i| v.part[i] == pt).collect();
            assert!(!ids.is_empty(), "part {pt} has a mesh");
            for &i in &ids {
                let bnd = &v.bindings[i];
                assert!(bnd.nodes.iter().all(|n| own.contains(n)), "part {pt} vertex {i} binds outside it");
                assert!((bnd.skin(&nodes) - v.vertices[i]).length() < 1e-3, "part {pt} vertex {i} at rest");
            }
        }
    }

    #[test]
    fn rounded_box_faces_outward() {
        let c = Vec3::new(1.0, 2.0, 3.0);
        let (v, t) = rounded_box(c, Vec3::new(0.5, 0.2, 0.3), 0.3, 6, 12);
        for tri in t {
            let [a, b, d] = tri.map(|i| v[i as usize]);
            let n = (b - a).cross(d - a);
            if n.length() < 1e-3 {
                continue;
            }
            assert!(n.dot((a + b + d) / 3.0 - c) > 0.0, "{a} {b} {d} n {n}");
        }
    }
}

/// The procedural car's render mesh for `p` (the softbody page's hatchback) and
/// the desc's rest node positions it is bound to. Shared by the web and C ABI exports.
pub fn procedural_car_visual(p: &CarParams) -> (VisualMesh, Vec<Vec3>) {
    let (text, layout) = crate::procedural::car_truck_layout(p);
    let d = crate::truck::load_truck(&text);
    let nodes: Vec<Vec3> = d
        .nodes
        .iter()
        .map(|n| Vec3::from_array(n.position))
        .collect();
    (car_visual(p, &layout, &nodes), nodes)
}
