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
}

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

/// Section regions along the loop (control segment kinds).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Region {
    Under,
    Side,
    Greenhouse,
    Roof,
}

/// Closed Catmull–Rom through `pts`, `m` samples, each with the segment it is in.
fn catmull_loop(pts: &[Vec2], m: usize) -> Vec<(Vec2, usize)> {
    let n = pts.len();
    (0..m)
        .map(|i| {
            let u = i as f32 / m as f32 * n as f32;
            let k = u.floor() as usize % n;
            let t = u.fract();
            let p0 = pts[(k + n - 1) % n];
            let p1 = pts[k];
            let p2 = pts[(k + 1) % n];
            let p3 = pts[(k + 2) % n];
            let t2 = t * t;
            let t3 = t2 * t;
            let p = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
            (p, k)
        })
        .collect()
}

/// Superellipsoid "rounded box" mesh: (vertices, triangles), outward wound.
fn rounded_box(centre: Vec3, half: Vec3, e: f32, lat: usize, lon: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
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
fn torus(centre: Vec3, axis: Vec3, r: f32, tube: f32, n: usize, m: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
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
    // Section control polygon at t (0 = rear, 1 = front), left side then right.
    let section = |t: f32| -> (Vec<Vec2>, Vec<Region>) {
        let taper = 1.0 - 0.08 * (2.0 * t - 1.0).abs().powi(4);
        let top = roof(t);
        let cab = smoothstep(1.08, 1.3, top);
        let top_w = hw * (0.93 - 0.13 * cab) * taper;
        let side = hw * taper;
        let half = [
            (Vec2::new(0.0, 0.24), Region::Under),
            (Vec2::new(side * 0.82, 0.23), Region::Under),
            (Vec2::new(side * 0.98, 0.33), Region::Side),
            (Vec2::new(side * 1.015, 0.56), Region::Side),
            (Vec2::new(side, 0.78), Region::Greenhouse),
            (Vec2::new(side - 0.035 - 0.05 * cab, 0.84 + 0.02 * cab), Region::Greenhouse),
            (Vec2::new(top_w, top - 0.03), Region::Roof),
            (Vec2::new(top_w * 0.55, top + 0.035), Region::Roof),
        ];
        let mut pts: Vec<Vec2> = half.iter().map(|h| h.0).collect();
        let mut reg: Vec<Region> = half.iter().map(|h| h.1).collect();
        pts.push(Vec2::new(0.0, top + 0.045));
        reg.push(Region::Roof);
        for h in half.iter().skip(1).rev() {
            pts.push(Vec2::new(-h.0.x, h.0.y));
        }
        // Mirrored segments run the other way: segment i (from point i) on the
        // right side has the region of the left side's segment before it.
        for k in (1..half.len()).rev() {
            reg.push(half[k - 1].1);
        }
        (pts, reg)
    };
    // Stations: the shell, with the ends rounded over the last 12 cm.
    let s_count = 84;
    let m = 72;
    let mut stations: Vec<(f32, f32)> = Vec::new(); // (z, section scale)
    for i in 0..s_count {
        let z = -0.5 * len + i as f32 / (s_count - 1) as f32 * len;
        stations.push((z, 1.0));
    }
    for (k, (dz, sc)) in [(0.05, 0.97), (0.09, 0.9)].iter().enumerate() {
        let _ = k;
        stations.insert(0, (-0.5 * len - dz, *sc));
        stations.push((0.5 * len + dz, *sc));
    }
    let z_of = |k: usize| ring_z(k.min(n - 1));
    let bay_f = z_of(layout.front_bay) - 0.5 * (len / (n - 1) as f32);
    let bay_r = z_of(layout.rear_bay) + 0.5 * (len / (n - 1) as f32);
    let z_mid = z_of(layout.mid);
    let door_f = (z_mid, z_of(layout.cabin_last + 1));
    let door_r = (z_of(layout.cabin_first.saturating_sub(1)), z_mid);
    let t_of = |z: f32| ((z + 0.5 * len) / len).clamp(0.0, 1.0);

    let mut b = Builder { mesh: VisualMesh::default(), grid: HashMap::new() };
    // Grid positions and regions.
    let grid: Vec<Vec<(Vec3, usize)>> = stations
        .iter()
        .map(|&(z, sc)| {
            let (pts, _) = section(t_of(z));
            let c = Vec2::new(0.0, 0.62);
            catmull_loop(&pts, m)
                .into_iter()
                .map(|(q, seg)| {
                    let q = c + (q - c) * sc;
                    (Vec3::new(q.x, q.y, z), seg)
                })
                .collect()
        })
        .collect();
    let regions = section(0.5).1;
    // Wheel arches: side and underbody points inside the arch circle (side view)
    // move onto it, so the opening has a smooth edge; cells left inside collapse.
    let arch_r = 0.42;
    let grid: Vec<Vec<(Vec3, usize)>> = grid
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|(q, seg)| {
                    let side = matches!(regions[seg], Region::Side | Region::Under) && q.x.abs() > hw * 0.55;
                    let mut q = q;
                    if side {
                        for &a in &layout.axles {
                            let d = Vec2::new(q.z - a, q.y - layout.axle_y);
                            if d.length() < arch_r && d.y > -0.2 {
                                let e = d.normalize_or_zero() * arch_r;
                                let e = if e.y < -0.05 { Vec2::new(e.x.signum() * arch_r, -0.05) } else { e };
                                q.z = a + e.x;
                                q.y = layout.axle_y + e.y;
                            }
                        }
                    }
                    (q, seg)
                })
                .collect()
        })
        .collect();
    let windscreen = (0.585, 0.705);
    let rear_window = (0.145, 0.27);
    let side_glass = (0.30, 0.59);
    let arch = |z: f32, y: f32| layout.axles.iter().any(|&a| Vec2::new(z - a, y - layout.axle_y).length() < arch_r - 0.01);
    for i in 0..stations.len() - 1 {
        for j in 0..m {
            let j1 = (j + 1) % m;
            let q = [grid[i][j].0, grid[i + 1][j].0, grid[i + 1][j1].0, grid[i][j1].0];
            let c = (q[0] + q[1] + q[2] + q[3]) * 0.25;
            let region = regions[grid[i][j].1];
            let t = t_of(c.z);
            let left = c.x > 0.0;
            // Wheel arches: the side opens around each wheel.
            if matches!(region, Region::Side | Region::Under) && c.x.abs() > hw * 0.55 && arch(c.z, c.y) {
                continue;
            }
            let (part, mat) = match region {
                Region::Under => (part::BODY, UNDERBODY),
                Region::Roof if c.z >= bay_f => (part::BONNET, PAINT),
                Region::Roof if c.z <= bay_r => (part::BOOT, PAINT),
                Region::Roof if (windscreen.0..windscreen.1).contains(&t) || (rear_window.0..rear_window.1).contains(&t) => {
                    (part::BODY, GLASS)
                }
                Region::Roof => (part::BODY, PAINT),
                Region::Greenhouse => {
                    let pillar = (c.z - z_mid).abs() < 0.06;
                    let glass = (side_glass.0..side_glass.1).contains(&t) && c.y > 0.86 && c.y < roof(t) - 0.06 && !pillar;
                    (part::BODY, if glass { GLASS } else { PAINT })
                }
                Region::Side => {
                    let door = if (door_f.0..door_f.1).contains(&c.z) {
                        Some(if left { part::DOOR_FL } else { part::DOOR_FR })
                    } else if (door_r.0..door_r.1).contains(&c.z) {
                        Some(if left { part::DOOR_RL } else { part::DOOR_RR })
                    } else {
                        None
                    };
                    match door {
                        Some(d) if !layout.panels.is_empty() => (d, PAINT),
                        _ => (part::BODY, PAINT),
                    }
                }
            };
            let part = if layout.panels.is_empty() && matches!(part, part::BONNET | part::BOOT) { part::BODY } else { part };
            let mut idx = [0u32; 4];
            for (k, &(si, sj)) in [(i, j), (i + 1, j), (i + 1, j1), (i, j1)].iter().enumerate() {
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
    // End caps: front (lights, grille, plate area) and rear.
    for (end, front) in [(stations.len() - 1, true), (0usize, false)] {
        let ring = &grid[end];
        let centre = ring.iter().map(|r| r.0).sum::<Vec3>() / m as f32 + Vec3::Z * if front { 0.03 } else { -0.03 };
        let cv = b.push_vertex(centre, part::BODY);
        let rv: Vec<u32> = ring.iter().map(|r| b.push_vertex(r.0, part::BODY)).collect();
        for j in 0..m {
            let (a, c) = (rv[j], rv[(j + 1) % m]);
            let mid = (centre + ring[j].0 + ring[(j + 1) % m].0) / 3.0;
            let lamp = mid.y > 0.6 && mid.y < 0.78 && mid.x.abs() > 0.38;
            let mat = if front {
                if lamp {
                    HEADLIGHT
                } else if mid.y > 0.36 && mid.y < 0.6 && mid.x.abs() < 0.5 {
                    GRILLE
                } else {
                    PAINT
                }
            } else if lamp {
                TAILLIGHT
            } else {
                PAINT
            };
            b.mesh.tris.push([cv, a, c]);
            b.mesh.tri_material.push(mat);
        }
    }
    // Bumpers: rounded bars over the bumper nodes, a chrome strip on the grille.
    let fz = 0.5 * len + 0.12;
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.47, fz), Vec3::new(hw * 0.95, 0.13, 0.1), 0.25, 10, 24), part::BUMPER_F, TRIM);
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.47, -fz), Vec3::new(hw * 0.95, 0.13, 0.1), 0.25, 10, 24), part::BUMPER_R, TRIM);
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.62, 0.5 * len + 0.11), Vec3::new(0.45, 0.012, 0.012), 0.6, 4, 8), part::BODY, CHROME);
    // Interior: seats, a dashboard, a steering wheel (seen through glass and
    // through torn-open doors).
    let zc = |k: usize| ring_z(k);
    let front_seat_z = 0.5 * (zc(layout.mid) + zc(layout.cabin_last));
    let rear_seat_z = zc(layout.cabin_first);
    for x in [0.38f32, -0.38] {
        b.add_mesh(rounded_box(Vec3::new(x, 0.45, front_seat_z), Vec3::new(0.25, 0.1, 0.25), 0.3, 6, 12), part::INTERIOR, INTERIOR);
        b.add_mesh(rounded_box(Vec3::new(x, 0.8, front_seat_z - 0.27), Vec3::new(0.24, 0.3, 0.07), 0.3, 6, 12), part::INTERIOR, INTERIOR);
    }
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.45, rear_seat_z), Vec3::new(0.6, 0.1, 0.24), 0.3, 6, 12), part::INTERIOR, INTERIOR);
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.78, rear_seat_z - 0.26), Vec3::new(0.6, 0.28, 0.07), 0.3, 6, 12), part::INTERIOR, INTERIOR);
    let dash_z = zc(layout.cabin_last) + 0.25;
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.86, dash_z), Vec3::new(hw * 0.88, 0.1, 0.18), 0.35, 6, 16), part::INTERIOR, INTERIOR);
    b.add_mesh(torus(Vec3::new(0.38, 0.95, dash_z - 0.22), Vec3::new(0.0, 0.45, 1.0), 0.17, 0.018, 20, 6), part::INTERIOR, TRIM);
    // Engine: block and head in the front bay (seen when the bonnet goes).
    let bay_c = 0.5 * (zc(layout.front_bay) + zc(n - 1));
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.55, bay_c), Vec3::new(0.32, 0.2, 0.26), 0.2, 6, 16), part::ENGINE, ENGINE);
    b.add_mesh(rounded_box(Vec3::new(0.0, 0.78, bay_c), Vec3::new(0.26, 0.05, 0.2), 0.3, 4, 12), part::ENGINE, CHROME);

    // Outward winding: flip any body triangle facing the car's long axis.
    for (tri, &mat) in b.mesh.tris.iter_mut().zip(&b.mesh.tri_material) {
        let [a, bb, c] = tri.map(|i| b.mesh.vertices[i as usize]);
        let nrm = (bb - a).cross(c - a);
        let cen = (a + bb + c) / 3.0;
        let axis = Vec3::new(0.0, 0.62, cen.z.clamp(-0.5 * len + 0.3, 0.5 * len - 0.3));
        let out = cen - axis;
        if matches!(mat, INTERIOR | ENGINE | TRIM | CHROME) {
            continue; // closed meshes are wound by construction
        }
        if nrm.dot(out) < 0.0 {
            tri.swap(1, 2);
        }
    }

    // Bind each part to its own nodes.
    let ring_nodes = (n * CAR_RING) as u32;
    let shell: Vec<u32> = (0..ring_nodes).collect();
    let rings = |ks: std::ops::RangeInclusive<usize>, js: &[usize]| -> Vec<u32> {
        ks.flat_map(|k| js.iter().map(move |&j| (k * CAR_RING + j) as u32)).collect()
    };
    let allowed = |part: u8| -> Vec<u32> {
        match part {
            part::BONNET..=part::DOOR_RR => layout.panels.get((part - 1) as usize).cloned().unwrap_or_else(|| shell.clone()),
            part::BUMPER_F if !layout.front_bumper.is_empty() => layout.front_bumper.clone(),
            part::BUMPER_R if !layout.rear_bumper.is_empty() => layout.rear_bumper.clone(),
            part::BUMPER_F => rings(n - 1..=n - 1, &[0, 1, 2, 3, 6]),
            part::BUMPER_R => rings(0..=0, &[0, 1, 2, 3, 6]),
            part::INTERIOR => rings(layout.cabin_first.saturating_sub(1)..=layout.cabin_last + 1, &[0, 1, 2, 3, 6]),
            part::ENGINE => rings(layout.front_bay.saturating_sub(1)..=n - 1, &[0, 1, 2, 3, 6, 7]),
            _ => shell.clone(),
        }
    };
    let opts = BindOptions { candidates: 12, max_edge: 1.6 };
    let mut bindings = vec![None; b.mesh.vertices.len()];
    for pt in 0..=part::ENGINE {
        let ids: Vec<usize> = (0..b.mesh.vertices.len()).filter(|&i| b.mesh.part[i] == pt).collect();
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
                    .min_by(|a, bb| (*a.1 - *v).length_squared().total_cmp(&(*bb.1 - *v).length_squared()))
                    .map_or(0, |(i, _)| i as u32);
                FlexBinding { nodes: [c; 3], coords: [0.0; 3] }
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
        let nodes: Vec<Vec3> = d.nodes.iter().map(|n| Vec3::from_array(n.position)).collect();
        let v = car_visual(&p, &layout, &nodes);
        assert!(v.vertices.len() > 5_000, "{} vertices", v.vertices.len());
        assert_eq!(v.tris.len(), v.tri_material.len());
        // Every vertex is reproduced exactly by its binding at rest.
        for (i, (bnd, pos)) in v.bindings.iter().zip(&v.vertices).enumerate() {
            if bnd.nodes[0] == bnd.nodes[1] {
                continue;
            }
            let got = bnd.skin(&nodes);
            assert!((got - *pos).length() < 1e-3, "vertex {i} part {}: {got} vs {pos}", v.part[i]);
        }
        // A door's vertices use only that door's panel nodes.
        let door: std::collections::HashSet<u32> = layout.panels[2].iter().copied().collect();
        let fl: Vec<usize> = (0..v.vertices.len()).filter(|&i| v.part[i] == part::DOOR_FL).collect();
        assert!(!fl.is_empty());
        assert!(fl.iter().all(|&i| v.bindings[i].nodes.iter().all(|n| door.contains(n))));
        // Glass, lights and seams exist.
        for mat in [material::GLASS, material::HEADLIGHT, material::TAILLIGHT, material::GRILLE] {
            assert!(v.tri_material.contains(&mat), "material {mat}");
        }
        assert!(v.seam.iter().any(|&s| s));
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
