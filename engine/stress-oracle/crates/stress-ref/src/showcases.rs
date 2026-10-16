//! Showcase scenes: the spec's comparison table ("Ours" column). Each one isolates a
//! piece of physics that threshold or quasi-static destruction gets wrong; the tests in
//! `tests/showcases.rs` assert the expected behaviour.
//!
//! The lead demos (fast vs slow car, sudden vs gradual column loss, spalling) are the
//! oracle benchmarks 5, 8 and 6 in `builders.rs`.

use crate::builders::*;
use crate::material::{BondKind, FractureEnergy, Material};
use crate::math::{Quat, Vec3};
use crate::scene::*;

/// Concrete with static fatigue but no rate effects (slow, static showcases).
pub fn static_load_concrete() -> Material {
    Material { dif: None, ..Material::concrete() }
}

/// Overhang on a thin connection (load path): a column carries a 2 m cantilevered slab
/// through a neck sized to `utilization` of its bending strength under self weight.
/// Just below 1 the neck holds at first; static fatigue then makes it creak, crack and
/// snap (concrete at 97%: about 8 s), and the slab swings down.
pub fn overhang(utilization: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Cantilevered slab hanging on a thin neck: snaps, sags, swings down.");
    let m = static_load_concrete();
    s.materials.insert("concrete".into(), m.clone());
    let mut chunks = grid(Vec3::new(0.0, 0.0, 0.0), [1, 1, 6], Vec3::splat(0.2), "concrete");
    chunks[0].support = Support::Fixed;
    let slab_mass = 10.0 * 0.008 * m.density;
    // Neck section from M = W * lever = utilization * f_t * w^3 / 6.
    let moment = slab_mass * 9.81 * (1.3 - 0.2);
    let w = (6.0 * moment / (utilization * m.tensile_strength)).cbrt().min(0.2);
    let mut neck = chunk(Vec3::new(0.25, 0.1, 1.1), Vec3::new(0.05, 0.5 * w, 0.5 * w), "concrete");
    neck.groups.push("neck".into());
    chunks.push(neck);
    for c in grid(Vec3::new(0.3, 0.0, 1.0), [10, 1, 1], Vec3::splat(0.2), "concrete") {
        let mut c = c;
        c.groups.push("slab".into());
        chunks.push(c);
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("overhang", chunks, bonds));
    s.sim.duration = 12.0;
    s.probes.push(probe("slab_tip", ProbeKind::ChunkDisplacement { body: "overhang".into(), chunk: 16, axis: [0.0, 0.0, 1.0] }));
    s
}

/// Supports removed one by one (redistribution, damage over time): a 4 m beam on four
/// posts carries a line load; the two outer posts are removed in turn, gradually (no
/// dynamic overshoot). The first removal leaves a short overhang well below strength;
/// the second leaves a 2.4 m overhang just below it (97% at 410 N/m), so the root
/// creaks, cracks and gives way over seconds (static fatigue). (Removing interior posts instead leaves spans restrained at
/// both ends, which arch once cracked and do not come down: correct, but no showcase.)
pub fn supports_one_by_one(line_load: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Beam on four posts, outer two removed in turn: creak, crack, give way over seconds.");
    s.materials.insert("concrete".into(), static_load_concrete());
    let mut chunks = grid(Vec3::new(0.0, 0.0, 0.2), [20, 1, 1], Vec3::splat(0.2), "concrete");
    for (k, i) in [0usize, 7, 13, 19].iter().enumerate() {
        let mut post = chunk(Vec3::new(0.1 + 0.2 * *i as f64, 0.1, 0.1), Vec3::splat(0.1), "concrete");
        post.support = Support::Fixed;
        post.groups.push(format!("post_{k}"));
        chunks.push(post);
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("beam", chunks, bonds));
    for c in 0..20 {
        s.loads.push(LoadDesc::PointForce {
            body: "beam".into(),
            chunk: c,
            direction: [0.0, 0.0, -1.0],
            magnitude: TimeFunction::Constant { value: line_load * 0.2 },
            point: None,
        });
    }
    for (t, k) in [(0.2, 3), (0.8, 2)] {
        s.events.push(EventDesc::RemoveChunks { time: t, duration: 0.3, body: "beam".into(), chunks: ChunkSelector::Group(format!("post_{k}")) });
    }
    s.sim.duration = 12.0;
    s
}

/// Steel vs brick, same car (energy absorption): the same crush-capped car hits two walls
/// whose joints have the same strength but brittle (brick) or ductile, 1000x tougher
/// (steel-like) fracture behaviour.
pub fn car_into_wall(ductile: bool, name: &str) -> Scene {
    let mut s = new_scene(name, "Same car into a brittle or a ductile wall: energy absorption before failure.");
    let mut m = Material { dif: None, static_fatigue: None, ..Material::brick() };
    if ductile {
        m.kind = BondKind::Ductile;
        m.fracture_energy = FractureEnergy {
            tension: m.fracture_energy.tension * 1000.0,
            shear: m.fracture_energy.shear * 1000.0,
            compression: m.fracture_energy.compression,
        };
    }
    s.materials.insert("wall".into(), m);
    s.materials.insert("steel".into(), elastic_steel());
    let counts = [10, 1, 11];
    let mut chunks = grid(Vec3::new(-1.0, 0.0, -0.2), counts, Vec3::splat(0.2), "wall");
    for i in 0..counts[0] {
        chunks[grid_index(counts, i, 0, 0)].support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "wall".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "car".into(),
        shape: ImpactorShape::Box { half_extents: [0.5, 0.4, 0.3] },
        mass: 1500.0,
        position: [0.0, -0.401, 0.6],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 8.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: Some(CrushDesc { max_force: 400e3, energy: 20e3 }),
    });
    s.sim.duration = 0.15;
    s.sim.frame_dt = 1e-3;
    s.probes.push(probe("car_velocity", ProbeKind::ImpactorVelocity { impactor: "car".into(), axis: [0.0, 1.0, 0.0] }));
    s
}

/// Floor falls onto floor below (impact loading): a corner-supported slab is hit by an
/// identical slab dropped from `drop` metres, or (drop = 0) carries the same weight as a
/// static pressure.
pub fn floor_onto_floor(drop: f64, name: &str) -> Scene {
    let mut s = new_scene(name, "Slab dropped onto a corner-supported slab: the lower floor fails (pancake).");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let counts = [10, 10, 1];
    let mut lower = grid(Vec3::new(-1.0, -1.0, 0.0), counts, Vec3::new(0.2, 0.2, 0.1), "concrete");
    for (i, j) in [(0, 0), (9, 0), (0, 9), (9, 9)] {
        lower[grid_index(counts, i, j, 0)].support = Support::Fixed;
    }
    let lb = auto_bonds(&lower, |_, _| "concrete".into());
    s.bodies.push(body("lower", lower, lb));
    if drop > 0.0 {
        let upper = grid(Vec3::new(-1.0, -1.0, 0.1 + drop), counts, Vec3::new(0.2, 0.2, 0.1), "concrete");
        let ub = auto_bonds(&upper, |_, _| "concrete".into());
        s.bodies.push(body("upper", upper, ub));
    } else {
        let weight_pressure = 2400.0 * 0.1 * 9.81;
        s.loads.push(LoadDesc::Pressure {
            body: "lower".into(),
            chunks: ChunkSelector::All,
            face_normal: [0.0, 0.0, 1.0],
            pressure: TimeFunction::Constant { value: weight_pressure },
        });
    }
    s.sim.duration = 0.25 + (2.0 * drop / 9.81).sqrt();
    s.sim.frame_dt = 1e-3;
    s
}

/// Masonry arch, keystone removed (compression, friction): a semicircular arch of 15
/// voussoirs on mortar joints stands by compression under its own weight; removing the
/// keystone at `t_remove` (if any) brings it down.
pub fn arch(t_remove: Option<f64>, name: &str) -> Scene {
    let mut s = new_scene(name, "Semicircular masonry arch: stands by compression; keystone removed, it collapses.");
    s.materials.insert("brick".into(), Material::brick());
    s.materials.insert("mortar".into(), Material::mortar());
    let (radius, depth, width, n) = (1.5f64, 0.3f64, 0.5f64, 15usize);
    let dtheta = std::f64::consts::PI / n as f64;
    let voussoir = |theta: f64| {
        // Local x tangential, z radial, y = -Y (right-handed).
        let (c, sn) = (theta.cos(), theta.sin());
        let t = Vec3::new(-sn, 0.0, c);
        let r = Vec3::new(c, 0.0, sn);
        let q = quat_from_axes(t, Vec3::new(0.0, -1.0, 0.0), r);
        let mut ch = chunk(r * radius, Vec3::new(0.5 * radius * dtheta, 0.5 * width, 0.5 * depth), "brick");
        ch.orientation = q.to_wxyz();
        ch
    };
    let mut chunks = Vec::new();
    // Abutments continue the ring below the springing line and are fixed.
    let mut right = voussoir(-0.5 * dtheta);
    right.support = Support::Fixed;
    chunks.push(right);
    for i in 0..n {
        let mut v = voussoir((i as f64 + 0.5) * dtheta);
        if i == n / 2 {
            v.groups.push("keystone".into());
        }
        chunks.push(v);
    }
    let mut left = voussoir(std::f64::consts::PI + 0.5 * dtheta);
    left.support = Support::Fixed;
    chunks.push(left);
    // Radial joints between consecutive voussoirs.
    let mut bonds = Vec::new();
    for i in 0..=n {
        let theta = i as f64 * dtheta;
        let (c, sn) = (theta.cos(), theta.sin());
        let normal = Vec3::new(-sn, 0.0, c);
        let radial = Vec3::new(c, 0.0, sn);
        bonds.push(BondDesc {
            a: i,
            b: i + 1,
            centroid: (radial * radius).to_array(),
            normal: normal.to_array(),
            tangent: radial.to_array(),
            area: depth * width,
            width: [depth, width],
            material: "mortar".into(),
            rebar: None,
            buckling_length: None,
            level: 0,
            groups: Vec::new(),
            derived: None,
        });
    }
    s.bodies.push(body("arch", chunks, bonds));
    if let Some(t) = t_remove {
        s.events.push(EventDesc::RemoveChunks { time: t, duration: 0.0, body: "arch".into(), chunks: ChunkSelector::Group("keystone".into()) });
    }
    s.sim.duration = t_remove.unwrap_or(0.0) + 0.6;
    s.sim.frame_dt = 1e-3;
    s
}

/// What happens to the masonry house.
#[derive(Clone, Copy, Debug)]
pub enum HouseEvent {
    /// Nothing: the house stands under its own weight (the prestress stress field).
    None,
    /// The footing under the front-right corner settles away over a second.
    Settlement,
    /// A crush-capped car hits the side wall at `speed` m/s.
    Impact { speed: f64 },
    /// `tnt` kg of TNT detonates 3 m in front of the house.
    Blast { tnt: f64 },
}

/// A 3 x 3 m brick house, 2.7 m high, one brick thick (0.3 x 0.15 x 0.15 m bricks in
/// running bond on mortar joints, corners interlocked course by course), with a door,
/// windows and concrete lintels, on footing chunks, carrying a roof load on its top
/// course. The front-right corner footing can settle away (1.65 m along both walls).
pub fn masonry_house(event: HouseEvent, name: &str) -> Scene {
    let mut s = new_scene(name, "Brick house with door and windows: stress paths around openings; settlement, impact or blast.");
    s.materials.insert("brick".into(), Material::brick());
    s.materials.insert("mortar".into(), Material::mortar());
    s.materials.insert("lintel".into(), Material { dif: None, static_fatigue: None, ..Material::concrete() });
    let (bl, bt, bh) = (0.3, 0.15, 0.15);
    let (width, depth, courses) = (3.0, 3.0, 18usize);
    // Wall lines: (runs along x?, coordinate of the wall's centre plane, outward normal sign).
    let walls = [(true, 0.5 * bt), (true, depth - 0.5 * bt), (false, 0.5 * bt), (false, width - 0.5 * bt)];
    // Openings: (wall, along-wall interval, height interval). Edges on the half-brick grid.
    let openings: [(usize, [f64; 2], [f64; 2]); 5] = [
        (0, [1.2, 1.8], [0.0, 2.1]),  // door
        (0, [2.1, 2.7], [0.9, 1.8]),  // front window
        (1, [1.2, 1.8], [1.2, 1.8]),  // back window
        (2, [1.05, 1.95], [0.9, 1.8]), // left window
        (3, [0.9, 2.1], [0.9, 1.8]),  // right window
    ];
    let cut = |pieces: Vec<[f64; 2]>, hole: [f64; 2]| -> Vec<[f64; 2]> {
        let mut out = Vec::new();
        for [a, b] in pieces {
            if b <= hole[0] + 1e-9 || a >= hole[1] - 1e-9 {
                out.push([a, b]);
                continue;
            }
            if hole[0] - a > 1e-9 {
                out.push([a, hole[0]]);
            }
            if b - hole[1] > 1e-9 {
                out.push([hole[1], b]);
            }
        }
        out
    };
    let mut chunks = Vec::new();
    let place = |along_x: bool, at: f64, a0: f64, a1: f64, z: f64, half_h: f64, material: &str| {
        let (c, h) = if along_x {
            (Vec3::new(0.5 * (a0 + a1), at, z), Vec3::new(0.5 * (a1 - a0), 0.5 * bt, half_h))
        } else {
            (Vec3::new(at, 0.5 * (a0 + a1), z), Vec3::new(0.5 * bt, 0.5 * (a1 - a0), half_h))
        };
        chunk(c, h, material)
    };
    for (wi, &(along_x, at)) in walls.iter().enumerate() {
        // Footing: fixed chunks under the full wall line.
        let length = if along_x { width } else { depth };
        let (lo, hi) = if along_x { (0.0, length) } else { (bt, length - bt) };
        let mut a = lo;
        while a < hi - 1e-9 {
            let b = (a + bl).min(hi);
            let mut f = place(along_x, at, a, b, -0.5 * bh, 0.5 * bh, "brick");
            f.support = Support::Fixed;
            let corner = if along_x { 0.5 * (a + b) > width - 1.65 && wi == 0 } else { 0.5 * (a + b) < 1.65 && wi == 3 };
            f.groups.push(if corner { "settling_footing".into() } else { "footing".into() });
            chunks.push(f);
            a = b;
        }
        for k in 0..courses {
            let z = (k as f64 + 0.5) * bh;
            // Corners interlock: x-walls run through on even courses, y-walls on odd ones.
            let through = (k % 2 == 0) == along_x;
            let (lo, hi) = if through { (0.0, length) } else { (bt, length - bt) };
            let mut edges = vec![lo];
            let mut e = lo + if k % 2 == 1 { 0.5 * bl } else { bl };
            while e < hi - 1e-9 {
                edges.push(e);
                e += bl;
            }
            edges.push(hi);
            let mut pieces: Vec<[f64; 2]> = edges.windows(2).map(|w| [w[0], w[1]]).filter(|p| p[1] - p[0] > 1e-9).collect();
            let mut lintels = Vec::new();
            for &(ow, span, height) in &openings {
                if ow != wi {
                    continue;
                }
                if z > height[0] && z < height[1] {
                    pieces = cut(pieces, span);
                } else if (z - 0.5 * bh - height[1]).abs() < 1e-9 {
                    let lintel = [span[0] - 0.5 * bl, span[1] + 0.5 * bl];
                    pieces = cut(pieces, lintel);
                    lintels.push(lintel);
                }
            }
            for [a, b] in pieces {
                let mut c = place(along_x, at, a, b, z, 0.5 * bh, "brick");
                if k + 1 == courses {
                    c.groups.push("roof_bearing".into());
                }
                chunks.push(c);
            }
            for [a, b] in lintels {
                chunks.push(place(along_x, at, a, b, z, 0.5 * bh, "lintel"));
            }
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "mortar".into());
    s.bodies.push(body("house", chunks, bonds));
    // Roof and floor dead load on the top course (30 kPa on its 0.15 m bearing, 4.5 kN/m).
    s.loads.push(LoadDesc::Pressure {
        body: "house".into(),
        chunks: ChunkSelector::Group("roof_bearing".into()),
        face_normal: [0.0, 0.0, 1.0],
        pressure: TimeFunction::Constant { value: 30e3 },
    });
    s.sim.frame_dt = 1e-3;
    match event {
        HouseEvent::None => s.sim.duration = 0.1,
        HouseEvent::Settlement => {
            s.events.push(EventDesc::RemoveSupports {
                time: 0.05,
                duration: 1.0,
                body: "house".into(),
                chunks: ChunkSelector::Group("settling_footing".into()),
            });
            s.sim.duration = 2.0;
            s.sim.frame_dt = 1.0 / 60.0;
        }
        HouseEvent::Impact { speed } => {
            s.materials.insert("steel".into(), elastic_steel());
            s.impactors.push(ImpactorDesc {
                name: "car".into(),
                shape: ImpactorShape::Box { half_extents: [0.4, 0.5, 0.3] },
                mass: 1500.0,
                position: [width + 0.401, 1.5, 0.75],
                orientation: [1.0, 0.0, 0.0, 0.0],
                velocity: [-speed, 0.0, 0.0],
                angular_velocity: [0.0; 3],
                material: "steel".into(),
                crush: Some(CrushDesc { max_force: 400e3, energy: 20e3 }),
            });
            s.sim.duration = 0.3;
        }
        HouseEvent::Blast { tnt } => {
            s.loads.push(LoadDesc::Blast { position: [1.5, -3.0, 1.0], tnt_mass: tnt, time: 0.0 });
            s.sim.duration = 0.1;
        }
    }
    s
}

/// Quaternion of the rotation whose matrix has columns `x`, `y`, `z`.
fn quat_from_axes(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let m = [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]];
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        Quat { w: 0.25 * s, x: (m[2][1] - m[1][2]) / s, y: (m[0][2] - m[2][0]) / s, z: (m[1][0] - m[0][1]) / s }
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        Quat { w: (m[2][1] - m[1][2]) / s, x: 0.25 * s, y: (m[0][1] + m[1][0]) / s, z: (m[0][2] + m[2][0]) / s }
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        Quat { w: (m[0][2] - m[2][0]) / s, x: (m[0][1] + m[1][0]) / s, y: 0.25 * s, z: (m[1][2] + m[2][1]) / s }
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        Quat { w: (m[1][0] - m[0][1]) / s, x: (m[0][2] + m[2][0]) / s, y: (m[1][2] + m[2][1]) / s, z: 0.25 * s }
    };
    q.normalized()
}

/// Every showcase scene, for `gen-scenes`.
pub fn catalog() -> Vec<Scene> {
    vec![
        overhang(0.97, "s_overhang_thin"),
        overhang(0.4, "s_overhang_thick"),
        supports_one_by_one(410.0, "s_supports_one_by_one"),
        car_into_wall(false, "s_car_brick"),
        car_into_wall(true, "s_car_ductile"),
        floor_onto_floor(1.0, "s_floor_drop"),
        floor_onto_floor(0.0, "s_floor_static"),
        arch(Some(0.1), "s_arch_keystone_removed"),
        arch(None, "s_arch"),
        blast_two_walls(30.0, "s_blast_two_walls"),
        masonry_house(HouseEvent::None, "s_house"),
        masonry_house(HouseEvent::Settlement, "s_house_settlement"),
        masonry_house(HouseEvent::Impact { speed: 10.0 }, "s_house_car"),
        masonry_house(HouseEvent::Blast { tnt: 5.0 }, "s_house_blast"),
    ]
}
