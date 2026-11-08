//! One car, one obstacle, the same scene through the explicit solver's native static
//! colliders and through the rigid-body coupling's CPU reference
//! (`engine_softbody::explicit_rigid`, `engine/docs/explicit-vehicles-in-world.md`): a
//! diagnosis of whether the coupling transmits the reaction the native contact does.
//!
//! Scenes (`scene=`):
//! * wall kinds (the car settled 1 s on its raycast wheels on the native ground, then
//!   `kmh=` toward a 10 × 3 × 2 m wall 0.2 m ahead of its nose):
//!   `native` (an axis-aligned `StaticBox`), `shape` (`StaticShape` oriented box),
//!   `proxy` (an infinite-mass rigid proxy), `heavy` (a finite-mass proxy, `mass=`,
//!   default 1e7 kg, through the two-body rule);
//! * `free`: a free proxy box (`mass=`, half extents `half=x,y,z`, default a 25 kg brick
//!   of 1 × 0.5 × 0.5 m at bumper height); with `fly=1` the car flies (no ground: only
//!   gravity acts, vertically), so the horizontal momentum must close (Newton III);
//! * `bricks`: a wall of free 25 kg proxy bricks (`cols=4 rows=3 deep=3`, 1 × 0.5 × 0.5 m,
//!   stacked from the ground), all in one coupling window (the CPU reference has no World:
//!   the bricks do not touch each other inside the window, as inside one World frame);
//! * `drop`: the car without its raycast wheels, its lowest node `h=` (1 m) above a
//!   surface: `on=ground` (the native ground), `box`, `shape`, `proxy`, `heavy`.
//!
//! Reports the external force on the car (−dp/dt of its nodes' momentum, gravity
//! removed, 1 ms and 5 ms means), its impulse, the obstacle's impulse (proxies), crush
//! and damage; `trace=path` writes the force–time series (TSV).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example coupling_probe -- <mods-dir> <name> <config.pc> scene=native [kmh=50]
//! ```
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape, GRAVITY, STEP,
};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use glam::{Quat, Vec3};

fn proxy_box(centre: Vec3, half: Vec3, mass: f32) -> (RigidProxy, ProxyShape) {
    let (inv_mass, inv_inertia) = if mass.is_finite() {
        let i = Vec3::new(
            mass / 3.0 * (half.y * half.y + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.y * half.y),
        );
        (1.0 / mass, Vec3::ONE / i)
    } else {
        (0.0, Vec3::ZERO)
    };
    (
        RigidProxy {
            body: 0,
            pos: centre,
            rot: Quat::IDENTITY,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            share: 1.0,
            count: 0,
        },
        ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            ProxyGeom::Box { half },
            0,
            None,
        ),
    )
}

fn set_of(boxes: Vec<(RigidProxy, ProxyShape)>) -> ProxySet {
    let mut set = ProxySet::default();
    for (k, (mut b, mut s)) in boxes.into_iter().enumerate() {
        b.body = k as u32;
        s.body = k as u32;
        s.collider = k as u32;
        set.bodies.push(b);
        set.shapes.push(s);
    }
    set.begin_window();
    set
}

fn momentum(s: &ExplicitBeams) -> glam::DVec3 {
    s.nodes.iter().map(|n| (n.vel * n.mass).as_dvec3()).sum()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| a.iter().find_map(|x| x.strip_prefix(k)).map(str::to_string);
    let f = |k: &str, d: f32| opt(k).map_or(d, |s| s.parse().unwrap());
    let scene = opt("scene=").unwrap_or_else(|| "native".into());
    let kmh = f("kmh=", 50.0);
    let t_end = f("t=", 0.5);
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let dir = std::path::Path::new(&a[0]);
    let texts = engine_softbody::jbeam::read_vehicle_dir(dir, &a[1]).unwrap();
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pcs = std::fs::read_to_string(dir.join(&a[1]).join(&a[2])).unwrap();
    let desc = engine_softbody::jbeam::load_jbeam(&files, Some(&pcs)).unwrap();
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let fw = engine_softbody::pack::forward_axis(&desc);
    let turn = Quat::from_rotation_y(-fw.x.atan2(fw.z));
    for n in &mut s.nodes {
        n.pos = turn * n.pos;
    }
    let low = s
        .nodes
        .iter()
        .map(|n| n.pos.y)
        .fold(f32::INFINITY, f32::min);
    for n in &mut s.nodes {
        n.pos.y -= low - 0.02;
    }
    let mass: f64 = s.nodes.iter().map(|n| f64::from(n.mass)).sum();
    let fly = opt("fly=").is_some();
    if opt("nowheels=").is_some() {
        // Without the raycast wheels (their rays meet the native colliders, not proxies).
        s.drive = None;
        s.tyres.clear();
    }
    let drop = scene == "drop";
    if drop {
        s.drive = None;
        s.tyres.clear();
    } else if !fly {
        s.run(1.0);
    }
    let (zlo, zhi) = s.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), q| {
        (l.min(q.pos.z), h.max(q.pos.z))
    });
    let len0 = zhi - zlo;
    let rear: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| s.nodes[i].pos.z < zlo + 0.33 * len0)
        .collect();
    let rear_z = |s: &ExplicitBeams| -> f64 {
        let m: f64 = rear.iter().map(|&i| f64::from(s.nodes[i].mass)).sum();
        rear.iter()
            .map(|&i| f64::from(s.nodes[i].mass * s.nodes[i].pos.z))
            .sum::<f64>()
            / m
    };
    let v0 = kmh / 3.6;
    let mut set: Option<ProxySet> = None;
    let face = zhi + 0.2;
    let wall_c = Vec3::new(0.0, 1.5, face + 1.0);
    let wall_h = Vec3::new(5.0, 1.5, 1.0);
    let axis: usize; // the measured force component: 2 (z) or 1 (y)
    if drop {
        axis = 1;
        let h = f("h=", 1.0);
        let on = opt("on=").unwrap_or_else(|| "ground".into());
        let lowy = s
            .nodes
            .iter()
            .map(|n| n.pos.y)
            .fold(f32::INFINITY, f32::min);
        // The surface's top at y = 0; the car's lowest node h above it.
        for n in &mut s.nodes {
            n.pos.y += h - lowy;
        }
        let slab_c = Vec3::new(0.0, -1.0, 0.0);
        let slab_h = Vec3::new(10.0, 1.0, 10.0);
        if on != "ground" {
            s.ground = -1.0e4;
        }
        match on.as_str() {
            "ground" => {}
            "box" => s.boxes.push(StaticBox::centered(slab_c, slab_h)),
            "shape" => s
                .shapes
                .push(StaticShape::oriented_box(slab_c, slab_h, Quat::IDENTITY)),
            "proxy" => set = Some(set_of(vec![proxy_box(slab_c, slab_h, f32::INFINITY)])),
            "heavy" => set = Some(set_of(vec![proxy_box(slab_c, slab_h, f("mass=", 1e7))])),
            o => panic!("on={o}"),
        }
    } else {
        axis = 2;
        if fly {
            s.ground = -1.0e4;
            let lowy = s
                .nodes
                .iter()
                .map(|n| n.pos.y)
                .fold(f32::INFINITY, f32::min);
            for n in &mut s.nodes {
                n.pos.y += 0.3 - lowy;
            }
        }
        for q in &mut s.nodes {
            q.vel.z += v0;
        }
        s.roll_tyres(Vec3::new(0.0, 0.0, v0));
        if let Some(d) = s.drive.as_mut() {
            d.roll(v0);
        }
        match scene.as_str() {
            "native" => s.boxes.push(StaticBox::centered(wall_c, wall_h)),
            "shape" => s
                .shapes
                .push(StaticShape::oriented_box(wall_c, wall_h, Quat::IDENTITY)),
            "proxy" => set = Some(set_of(vec![proxy_box(wall_c, wall_h, f32::INFINITY)])),
            "heavy" => set = Some(set_of(vec![proxy_box(wall_c, wall_h, f("mass=", 1e7))])),
            "free" => {
                let hv: Vec<f32> = opt("half=").map_or(vec![0.5, 0.25, 0.25], |h| {
                    h.split(',').map(|x| x.parse().unwrap()).collect()
                });
                let half = Vec3::new(hv[0], hv[1], hv[2]);
                let y = f(
                    "y=",
                    s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min) + 0.45,
                );
                set = Some(set_of(vec![proxy_box(
                    Vec3::new(0.0, y, face + half.z),
                    half,
                    f("mass=", 25.0),
                )]));
            }
            "bricks" => {
                let (cols, rows, deep) = (
                    f("cols=", 4.0) as i32,
                    f("rows=", 3.0) as i32,
                    f("deep=", 3.0) as i32,
                );
                let half = Vec3::new(0.5, 0.25, 0.25);
                let mut v = Vec::new();
                for d in 0..deep {
                    for r in 0..rows {
                        for c in 0..cols {
                            let x = (c as f32 - 0.5 * (cols - 1) as f32) * 1.0
                                + if r % 2 == 1 { 0.5 } else { 0.0 };
                            v.push(proxy_box(
                                Vec3::new(x, 0.25 + 0.5 * r as f32, face + 0.25 + 0.5 * d as f32),
                                half,
                                f("mass=", 25.0),
                            ));
                        }
                    }
                }
                set = Some(set_of(v));
            }
            o => panic!("scene={o}"),
        }
    }
    let p_obj0: glam::DVec3 = set.as_ref().map_or(glam::DVec3::ZERO, |st| {
        st.bodies
            .iter()
            .filter(|b| !b.is_static())
            .map(|b| (b.vel / b.inv_mass).as_dvec3())
            .sum()
    });
    if let Some(st) = set {
        s.rigid = Some(Box::new(st));
    }
    if opt("surface=").is_some() {
        // Opt-in surface contact (jbeam-semantics-spec.md §38): triangles and beam edges
        // against the shapes and proxies, beside node contact.
        s.surface_contact = true;
    }
    let steps = (t_end / STEP).round() as usize;
    let p0 = momentum(&s);
    let rz0 = rear_z(&s);
    let mut p_prev = p0;
    let mut force: Vec<f64> = Vec::with_capacity(steps);
    let mut rows = String::from("t_ms\tF_N\trear_z\tobj_J\n");
    let mut max_dyn = 0.0f64;
    for k in 0..steps {
        s.step();
        let p = momentum(&s);
        let g = if axis == 1 {
            -mass * f64::from(GRAVITY)
        } else {
            0.0
        };
        // External force on the car from contacts along the axis, gravity removed:
        // dp/dt − m g (axis y), dp/dt (axis z, reported as the wall's push −dp/dt).
        let fz = if axis == 2 {
            -(p.z - p_prev.z) / f64::from(STEP)
        } else {
            (p.y - p_prev.y) / f64::from(STEP) + g
        };
        p_prev = p;
        force.push(fz);
        let rz = rear_z(&s) - rz0 - f64::from(v0) * f64::from(STEP) * (k + 1) as f64 * 0.0;
        max_dyn = max_dyn.max(rz);
        let obj: f64 = s.rigid.as_ref().map_or(0.0, |r| {
            r.shapes.iter().map(|sh| f64::from(sh.impulse[axis])).sum()
        });
        rows.push_str(&format!(
            "{:.1}\t{fz:.0}\t{rz:.4}\t{obj:.1}\n",
            (k + 1) as f32 * STEP * 1e3
        ));
    }
    let mean = |w: usize| -> f64 {
        (w..=force.len())
            .map(|e| force[e - w..e].iter().sum::<f64>() / w as f64)
            .fold(f64::MIN, f64::max)
    };
    let p1 = momentum(&s);
    let dp = p1 - p0;
    let (broken, bent) = s.damage();
    let (lo1, hi1) = s.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), q| {
        (l.min(q.pos.z), h.max(q.pos.z))
    });
    println!(
        "scene {scene}{} {kmh} km/h ({mass:.0} kg, {t_end} s): peak external force 1 ms {:.0} kN, 5 ms {:.0} kN; car impulse dp {:.0} N s (axis {}), dp_x {:.0}",
        if fly { " (flying)" } else { "" },
        mean(2) / 1e3,
        mean(10) / 1e3,
        if axis == 2 { -dp.z } else { dp.y + mass * f64::from(-GRAVITY) * f64::from(t_end) },
        if axis == 2 { "z" } else { "y" },
        dp.x
    );
    if let Some(r) = s.rigid.as_ref() {
        let j: glam::DVec3 = r.shapes.iter().map(|sh| sh.impulse.as_dvec3()).sum();
        let p_obj: glam::DVec3 = r
            .bodies
            .iter()
            .filter(|b| !b.is_static())
            .map(|b| (b.vel / b.inv_mass).as_dvec3())
            .sum();
        let moved = r
            .bodies
            .iter()
            .filter(|b| !b.is_static() && b.vel.length() > 0.1)
            .count();
        let vmax = r.bodies.iter().map(|b| b.vel.length()).fold(0.0, f32::max);
        println!(
            "  obstacle: impulse received {:.1?} N s; bodies' momentum change {:.1?}; car dp + bodies dp = {:.2?} (horizontal residual {:.3} N s); {moved} of {} bodies moving > 0.1 m/s, fastest {vmax:.1} m/s",
            j,
            p_obj - p_obj0,
            dp + (p_obj - p_obj0),
            ((dp + (p_obj - p_obj0)) * glam::DVec3::new(1.0, 0.0, 1.0)).length(),
            r.bodies.len()
        );
    }
    let ymin = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    println!(
        "  rear third travel max {:.3} m; car length now {:.3} (was {len0:.3}); lowest node y {ymin:.3}; broken {broken}, bent {bent}; mean car v_z at end {:.2} m/s",
        max_dyn,
        hi1 - lo1,
        p1.z / mass
    );
    if let Some(path) = opt("trace=") {
        std::fs::write(&path, rows).unwrap();
    }
}
