//! Explicit soft-body vehicles in the World (`engine/docs/explicit-vehicles-in-world.md`):
//! * a vehicle into a free dynamic box: the momentum the vehicle loses the box gains,
//!   through the World's own impulse path (the GPU plan, the explicit frame, the emit and
//!   the rigid step), to the fixed-point rounding; the box ends between the plastic and
//!   elastic bounds;
//! * a vehicle into a prefractured threshold house: its strike loads the stress solver,
//!   bonds break and the house splits, forward-only (the loads of a window break bonds
//!   before the rigid step of that window; nothing is re-simulated).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::*;

/// The repo sedan, its front face's middle at `front` (its lowest node 2 cm above
/// `front.y`), heading along +Z (`turns` quarter turns about +Y from there) at `speed`,
/// on flat ground at `ground`.
fn sedan(speed: f32, ground: f32, front: Vec3, turns: u32) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let b = |s: &ExplicitBeams| s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    // Long axis along +Z, then turned.
    let mut dir = Vec3::Z;
    for _ in 0..turns % 4 {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(n.pos.z, n.pos.y, -n.pos.x);
        }
        dir = Vec3::new(dir.z, 0.0, -dir.x);
    }
    let (lo, hi) = b(&sim);
    let lateral = Vec3::Y.cross(dir).abs();
    let shift = Vec3::new(0.0, front.y + 0.02 - lo.y, 0.0)
        + dir * (front.dot(dir) - sim.nodes.iter().map(|n| n.pos.dot(dir)).fold(f32::MIN, f32::max))
        + lateral * (front.dot(lateral) - 0.5 * (lo + hi).dot(lateral));
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = dir * speed;
    }
    sim.ground = ground;
    sim
}

fn node_momentum(w: &World, masses: &[f32]) -> Vec3 {
    let v = w.explicit_vehicles().unwrap().velocities();
    v[0].iter().zip(masses).map(|(v, m)| *v * *m).sum()
}

#[test]
fn a_vehicle_shoves_a_free_box_and_momentum_closes() {
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            // The explicit solver's own gravity: the box falls with the flying vehicle.
            gravity: [0.0, engine_softbody::explicit_beams::GRAVITY, 0.0],
            enable_sleep: false,
            ..WorldDef::default()
        },
    )
    .unwrap();
    // Flying (no ground under the vehicle; none for the box): no horizontal external force.
    let car = sedan(15.0, -1.0e4, Vec3::new(0.0, 1.0, 0.0), 0);
    let hi_y = car.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let half = [1.2f32, 0.7, 0.5];
    let mass = 1200.0f32;
    let volume = 8.0 * half[0] * half[1] * half[2];
    let b = w
        .create_body(&BodyDef {
            linear_damping: 0.0,
            angular_damping: 0.0,
            ..BodyDef::dynamic([0.0, 0.5 * (1.0 + hi_y), 0.8 + half[2]])
        })
        .unwrap();
    w.create_box_shape(
        b,
        &ShapeDef {
            density: mass / volume,
            ..ShapeDef::default()
        },
        half,
    )
    .unwrap();
    // A box that does not turn (a huge rotational inertia, the same in the proxy and the
    // World): the exchange is the one-dimensional collision whose bounds are exact.
    let big = 1.0e9;
    w.body_set_mass_data(
        b,
        MassData {
            mass,
            center: [0.0; 3],
            rotational_inertia: [[big, 0.0, 0.0], [0.0, big, 0.0], [0.0, 0.0, big]],
        },
    )
    .unwrap();
    let masses: Vec<f32> = car.nodes.iter().map(|n| n.mass).collect();
    let m: f32 = masses.iter().sum();
    w.add_explicit_vehicles(std::slice::from_ref(&car), &ExplicitVehiclesDef::default()).unwrap();
    let box_mass = w.body_mass_data(b).unwrap().unwrap().mass;
    assert!((box_mass - mass).abs() < 1e-2 * mass, "box mass {box_mass}");
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let p0 = horizontal(node_momentum(&w, &masses));
    let mut worst = 0.0f32;
    let dt = w.solver().dt;
    let frames = (1.2 / dt).round() as u32;
    for k in 0..frames {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if std::env::var("EV_TRACE2").is_ok() && k < 30 {
            let imp = w.explicit_window_impulses();
            let j: Vec3 = imp[0].iter().map(|s| Vec3::from(s.1)).sum();
            println!("box step {k}: {} shapes J {j:?}", imp[0].len());
        }
        if k % 4 == 3 {
            let vb = Vec3::from(w.read_body_state(b).unwrap().linear_velocity);
            // The last window's impulses, applied by the next step's rigid step.
            let pending: Vec3 = w.explicit_window_impulses()[0].iter().map(|s| Vec3::from(s.1)).sum();
            let p = horizontal(node_momentum(&w, &masses) + vb * box_mass + pending);
            worst = worst.max((p - p0).length());
            if std::env::var("EV_TRACE").is_ok() {
                let st = w.read_body_state(b).unwrap();
                let pos = w.explicit_vehicles().unwrap().positions();
                let (lo, hi) = pos[0].iter().fold((Vec3::MAX, Vec3::MIN), |(a, c), p| (a.min(*p), c.max(*p)));
                println!("t {:.3}: box at {:?} v {:?} w {:?}; vehicle v {:.3} box {lo:?}..{hi:?}", (k + 1) as f32 * dt, st.transform.position, st.linear_velocity, st.angular_velocity, node_momentum(&w, &masses).z / m);
            }
        }
    }
    let vb = Vec3::from(w.read_body_state(b).unwrap().linear_velocity);
    let vv = node_momentum(&w, &masses).z / m;
    let plastic = m * 15.0 / (m + box_mass);
    println!(
        "dt {dt}: box {:.3} m/s in [{plastic:.3}, {:.3}], vehicle {vv:.3}; worst |dP| {worst:.3} of {:.0} N s; steps {:?}",
        vb.z,
        2.0 * plastic,
        p0.length(),
        w.explicit_vehicle_steps()
    );
    assert!(vb.z > 0.5 * plastic, "the box was shoved: {vb:?}");
    assert!(worst < 2e-4 * p0.length(), "momentum: {worst} of {}", p0.length());
    assert!(vb.z >= 0.99 * plastic && vb.z <= 2.0 * plastic, "box speed {}", vb.z);
    // Kinetic energy is lost, never gained (deformation, damping).
    let e0 = 0.5 * m * 15.0 * 15.0;
    let e1 = 0.5 * m * vv * vv + 0.5 * box_mass * vb.z * vb.z;
    assert!(e1 < e0, "energy {e1} after {e0}");
}

/// The cuboid house of the fracture lab as a threshold destructible (prefractured,
/// stress-solved), its ground; the sedan at 20 m/s into its front wall on its own ground at
/// the same height. The strike's chunk loads (the threshold ingest's slot 1) break bonds
/// and split the house before the rigid step of the window they come from, and the freed
/// pieces carry the vehicle's push.
#[cfg(feature = "stress")]
#[test]
fn a_vehicle_breaks_a_prefractured_house() {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None).unwrap();
    println!("house centre {:?} extents {} x {}", lab.center, lab.extent_x, lab.extent_z);
    let stress = asset.stress.as_ref().unwrap();
    let placement = stress.placement();
    let offset = placement.map(|x| -x as f32);
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut cap = stress_rubble::capacity_for(&scene);
    cap.max_bodies = cap.max_bodies.max(((chunks + 16 + 4) as u32).next_power_of_two());
    cap.max_contacts = stress_rubble::threshold::contact_capacity(chunks, cap.max_contacts, 1 << 30);
    cap.max_candidates = cap.max_candidates.max(cap.max_contacts);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: cap,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let h = stress.ground_height().unwrap() + offset[1];
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: h })
        .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let centre = Vec3::new(
        lab.center[0] as f32 + offset[0],
        lab.center[1] as f32 + offset[1],
        lab.center[2] as f32 + offset[2],
    );
    // Its front: the +Z face of its box, the car 1.5 m short of it, heading −Z.
    let zmax = centre.z + 0.5 * lab.extent_z as f32;
    let car = sedan(20.0, h, Vec3::new(centre.x, h, zmax + 1.5), 2);
    let def = ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity { bodies: 64, shapes: 255, planes: 4096 },
        ..ExplicitVehiclesDef::default()
    };
    let dt = w.solver().dt;
    // The house settles first (a bond or two at the edge of its capacity may go under its
    // own weight): only the strike's breaks are counted.
    let mut settled = 0usize;
    for _ in 0..30 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    settled += bonds.len();
                }
            }
        }
    }
    println!("settling broke {settled} bonds");
    w.add_explicit_vehicles(std::slice::from_ref(&car), &def).unwrap();
    let mut broken = 0usize;
    let mut splits = 0usize;
    let mut first: Option<u32> = None;
    let mut first_contact: Option<u32> = None;
    let mut children = Vec::new();
    for k in 0..(1.5 / dt).round() as u32 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken { destructible, bonds, .. } if *destructible == id => {
                        broken += bonds.len();
                        first.get_or_insert(k);
                    }
                    DestructionEvent::Split { children: c, .. } => {
                        splits += 1;
                        children.extend(c.iter().map(|c| c.bodies[0]));
                    }
                    _ => {}
                }
            }
        }
        let imp = w.explicit_window_impulses();
        let j: f32 = imp[0].iter().map(|s| Vec3::from(s.1).length()).sum();
        if std::env::var("EV_TRACE").is_ok() && k < 40 {
            let p = w.explicit_vehicles().unwrap().positions();
            let zmin = p[0].iter().map(|p| p.z).fold(f32::MAX, f32::min);
            println!("step {k}: {} shapes, |J| {j:.1}, broken {broken}, front z {zmin:.2}", imp[0].len());
        }
        if first_contact.is_none() && j > 0.0 {
            first_contact = Some(k);
        }
    }
    let fastest = children
        .iter()
        .filter(|&&b| w.body_is_valid(b))
        .map(|&b| Vec3::from(w.read_body_state(b).unwrap().linear_velocity).length())
        .fold(0.0f32, f32::max);
    println!("fastest piece {fastest:.2} m/s");
    // No piece leaves faster than an elastic bounce off the 20 m/s vehicle allows.
    assert!(fastest <= 2.0 * 20.0, "a freed piece at {fastest} m/s");
    let moving = children
        .iter()
        .filter(|&&b| w.body_is_valid(b) && Vec3::from(w.read_body_state(b).unwrap().linear_velocity).length() > 0.5)
        .count();
    println!(
        "house: {chunks} chunks; contact at step {first_contact:?}, first break at {first:?}; {broken} bonds, {splits} splits, {} pieces ({moving} moving)",
        children.len()
    );
    assert!(first_contact.is_some(), "the vehicle reached the house");
    assert!(broken > 0 && splits > 0, "the strike breaks the house");
    // Forward-only: the window's loads break bonds in the step that took them (the frame
    // before the rigid step of that window), never by re-running a step.
    assert!(first.unwrap() <= first_contact.unwrap() + 2, "breaks follow the strike at once");
    assert!(moving > 0, "freed pieces carry the vehicle's push");
}

/// The coupling's per-step shape cull (`explicit_beams_rigid.slang` `cplMay`) skips only
/// the proxy shapes a node cannot be inside: the same frames with it on and off are equal
/// bit for bit (every word of the vehicles' state, the proxies' integrated bodies and the
/// window impulses included). Four cars into one house at 120 Hz; the frames compared run
/// once the cars are inside it (tens of chunk shapes each, contact groups of two).
#[test]
fn the_coupling_shape_cull_is_bit_exact() {
    let template = sedan(0.0, 0.0, Vec3::ZERO, 0);
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &template,
        1,
        4,
        15.0,
        8.0,
        30,
        &rubble::explicit_scene::town_coupling(),
        Some(1.0 / 120.0),
        None,
    )
    .unwrap();
    let w = &mut town.world;
    let mut checked = 0;
    for step in 0..150 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if step < 60 || step % 30 != 0 {
            continue;
        }
        let shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
        let x = w.explicit_vehicles_mut().unwrap();
        let snap = x.snapshot();
        let mut run = |cull: bool| {
            x.restore(&snap);
            x.set_coupling_cull(cull);
            for _ in 0..3 {
                x.frame(17);
            }
            x.wait();
            // (The header's cull bit itself is the one word the runs set apart.)
            x.set_coupling_cull(true);
            x.state_words()
        };
        let off = run(false);
        let on = run(true);
        x.restore(&snap);
        x.set_coupling_cull(true);
        println!("step {step}: proxy shapes per vehicle {shapes:?}");
        for ((name, a), (_, b)) in off.iter().zip(&on) {
            let diff = a.iter().zip(b).filter(|(p, q)| p != q).count();
            assert_eq!(diff, 0, "step {step}: {name}: {diff} words differ with the cull on");
        }
        if shapes.iter().any(|&n| n >= 10) {
            checked += 1;
        }
    }
    assert!(checked >= 2, "the comparison ran with the cars among the house's chunks");
}

/// The coupling's workgroup copies (`explicit_beams_rigid.slang` `cplGsAdd`: the proxy
/// spheres and each node's shape range from the count pass in workgroup memory, a small
/// contact group's members' coupling side by side) change no result: the same frames
/// with them on and off are equal bit for bit (every word of the vehicles' state, the
/// proxies' integrated bodies and the window impulses included). Four cars into one
/// house at 120 Hz; the frames compared run once the cars are inside it (tens of chunk
/// shapes each, contact groups of two).
#[test]
fn the_coupling_workgroup_copies_are_bit_exact() {
    let template = sedan(0.0, 0.0, Vec3::ZERO, 0);
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &template,
        1,
        4,
        15.0,
        8.0,
        30,
        &rubble::explicit_scene::town_coupling(),
        Some(1.0 / 120.0),
        None,
    )
    .unwrap();
    let w = &mut town.world;
    let mut checked = 0;
    for step in 0..150 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if step < 60 || step % 30 != 0 {
            continue;
        }
        let shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
        let x = w.explicit_vehicles_mut().unwrap();
        let snap = x.snapshot();
        let mut run = |cache: bool| {
            x.restore(&snap);
            x.set_coupling_workgroup_cache(cache);
            for _ in 0..3 {
                x.frame(17);
            }
            x.wait();
            // (The header's cache bit itself is the one word the runs set apart.)
            x.set_coupling_workgroup_cache(true);
            x.state_words()
        };
        let off = run(false);
        let on = run(true);
        x.restore(&snap);
        x.set_coupling_workgroup_cache(true);
        println!("step {step}: proxy shapes per vehicle {shapes:?}");
        for ((name, a), (_, b)) in off.iter().zip(&on) {
            let diff = a.iter().zip(b).filter(|(p, q)| p != q).count();
            assert_eq!(diff, 0, "step {step}: {name}: {diff} words differ with the workgroup copies on");
        }
        if shapes.iter().any(|&n| n >= 10) {
            checked += 1;
        }
    }
    assert!(checked >= 2, "the comparison ran with the cars among the house's chunks");
}

/// The coupling plan's lane-parallel records (`vehicle_coupling.slang` `cplPlanShapes`)
/// are its serial walk's: from one World state, the plan alone both ways gives the same
/// proxy records (bodies, shapes, hull planes, each body's shape range) and share
/// counts, word for word. Four cars into one house at 120 Hz, compared once they meet
/// its chunks.
#[test]
fn the_coupling_plan_in_parallel_is_the_serial_walk() {
    let template = sedan(0.0, 0.0, Vec3::ZERO, 0);
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &template,
        1,
        4,
        15.0,
        8.0,
        30,
        &rubble::explicit_scene::town_coupling(),
        Some(1.0 / 120.0),
        None,
    )
    .unwrap();
    let w = &mut town.world;
    let mut checked = 0;
    for step in 0..150 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if step < 60 || step % 15 != 0 {
            continue;
        }
        let shapes: Vec<usize> = w.explicit_window_impulses().iter().map(|v| v.len()).collect();
        let parallel = w.explicit_replan_words(false);
        let serial = w.explicit_replan_words(true);
        assert_eq!(parallel.len(), serial.len());
        let diff = parallel.iter().zip(&serial).filter(|(a, b)| a != b).count();
        println!("step {step}: proxy shapes per vehicle {shapes:?}, {diff} words differ");
        assert_eq!(diff, 0, "step {step}: the parallel plan differs from the serial walk");
        if shapes.iter().any(|&n| n >= 10) {
            checked += 1;
        }
    }
    assert!(checked >= 2, "the comparison ran with the cars among the house's chunks");
}
