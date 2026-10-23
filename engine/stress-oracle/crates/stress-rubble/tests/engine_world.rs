//! The world with rubble as its rigid-body engine (`World::with_engine`) against the
//! standalone reference world; ported from `stress-physx/tests/engine_world.rs`.
//!
//!   cargo test --release -p stress-rubble --test engine_world -- --nocapture

use stress_ref::engine::{BodyGeometry, BodyKey, BoxShape, EngineBody, RigidEngine};
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::scene::Scene;
use stress_ref::world::World;
use stress_rubble::RubbleEngine;

fn coupled(scene: &Scene) -> Option<World> {
    let engine = RubbleEngine::boxed(Vec3::from_array(scene.gravity))?;
    Some(World::with_engine(scene, engine))
}

fn cube(key: BodyKey, at: Vec3, chunk: usize, mass: f64) -> EngineBody {
    EngineBody {
        key,
        signature: 1,
        geometry: BodyGeometry::Boxes(vec![BoxShape { local: Pose::default(), half_extents: Vec3::splat(0.5), mass, chunk: Some((0, chunk)), hull: None }].into()),
        fixed: false,
        ccd: false,
        pose: Pose::new(at, Quat::IDENTITY),
        velocity: Vec3::ZERO,
        angular_velocity: Vec3::ZERO,
    }
}

/// A box resting on the ground: the contact impulses rubble reports on its chunk push
/// it up and add up to its weight over a step.
#[test]
fn engine_contact_impulses_carry_the_weight_upwards() {
    let Ok(mut e) = RubbleEngine::new(Vec3::new(0.0, 0.0, -9.81)) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    e.add_ground(0.0);
    let mass = 100.0;
    e.sync_bodies(&[cube(BodyKey::Cluster(1), Vec3::new(0.0, 0.0, 0.5), 7, mass)]);
    let dt = 1.0 / 60.0;
    let mut last = Vec::new();
    for _ in 0..60 {
        last = e.step(dt, &[], &[]);
    }
    let total = last.iter().fold(Vec3::ZERO, |a, c| a + c.impulse);
    println!("closing speeds {:?}", last.iter().map(|c| c.approach_speed).collect::<Vec<_>>());
    println!("{} contacts, impulse {total:?}, points {:?}", last.len(), last.iter().map(|c| c.point.to_array()).collect::<Vec<_>>());
    assert!(!last.is_empty());
    assert!(last.iter().all(|c| c.structure == 0 && c.chunk == 7 && c.other.is_none()));
    assert!(last.iter().all(|c| c.point.z.abs() < 0.02), "contact points on the ground");
    assert!((total.z - mass * 9.81 * dt).abs() < 0.02 * mass * 9.81 * dt, "{total:?}");
    assert!(total.x.abs() + total.y.abs() < 0.01 * mass * 9.81 * dt, "{total:?}");
    let m = e.motion();
    let z = m[0].pose.position.z;
    assert!((z - 0.5).abs() < 0.01, "rests at {z}");
    assert!(m[0].velocity.norm() < 0.01, "{:?}", m[0].velocity);
}

/// Two boxes side by side, one frozen: the other, thrown at it, stops against it
/// (contacts against a frozen body are resolved), while the frozen one does not move;
/// released by `set_motion`, it moves with the state handed back.
#[test]
fn frozen_bodies_hold_still_and_are_released_with_the_given_motion() {
    let Ok(mut e) = RubbleEngine::new(Vec3::ZERO) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let (a, b) = (BodyKey::Cluster(1), BodyKey::Cluster(2));
    let mut thrown = cube(b, Vec3::new(-2.0, 0.0, 0.0), 1, 10.0);
    thrown.velocity = Vec3::new(3.0, 0.0, 0.0);
    e.sync_bodies(&[cube(a, Vec3::ZERO, 0, 10.0), thrown]);
    let dt = 1.0 / 60.0;
    let mut hits = 0;
    for _ in 0..60 {
        let contacts = e.step(dt, &[a], &[]);
        hits += contacts.iter().filter(|c| c.chunk == 1 && c.other == Some(a)).count();
    }
    let m = e.motion();
    let (ma, mb) = (m.iter().find(|m| m.key == a).unwrap(), m.iter().find(|m| m.key == b).unwrap());
    println!("frozen at {:?}, thrown at {:?} moving {:?}; {hits} contacts", ma.pose.position, mb.pose.position, mb.velocity);
    assert!(ma.pose.position.norm() < 1e-6, "the frozen body moved: {:?}", ma.pose.position);
    assert!(hits > 0);
    // Stopped at the frozen body (AVBD's penalty contact hands back a few percent of
    // the closing speed: it rebounds at about 0.1 m/s).
    assert!(mb.pose.position.x < -0.98 && mb.pose.position.x > -1.1, "{:?}", mb.pose.position);
    assert!(mb.velocity.x.abs() < 0.2 * 3.0, "{:?}", mb.velocity);
    // Release with a velocity.
    let handed = stress_ref::engine::BodyMotion { key: a, pose: Pose::new(Vec3::new(0.0, 0.0, 1.0), Quat::IDENTITY), velocity: Vec3::new(0.0, 0.0, 2.0), angular_velocity: Vec3::ZERO };
    e.set_motion(&[handed]);
    e.step(dt, &[], &[]);
    let m = e.motion();
    let ma = m.iter().find(|m| m.key == a).unwrap();
    assert!((ma.velocity.z - 2.0).abs() < 1e-3, "{:?}", ma.velocity);
    assert!((ma.pose.position.z - (1.0 + 2.0 * dt)).abs() < 1e-3, "{:?}", ma.pose.position);
}

/// With TOI CCD (`set_toi_ccd`), a 250 m/s impactor flagged `ccd` stops at a fixed
/// 0.1 m wall with one engine step per frame (no whole-world substeps: time-of-impact
/// contacts in the step); the same impactor without `ccd` passes through it (CCD audit
/// Phase 2).
#[test]
fn a_ccd_impactor_stops_at_a_thin_wall_without_substeps() {
    for ccd in [false, true] {
        let Ok(mut e) = RubbleEngine::new(Vec3::ZERO) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        e.set_toi_ccd(true);
        let wall = EngineBody {
            key: BodyKey::Cluster(1),
            signature: 1,
            geometry: BodyGeometry::Boxes(
                vec![BoxShape { local: Pose::default(), half_extents: Vec3::new(0.05, 1.5, 1.5), mass: 100.0, chunk: Some((0, 0)), hull: None }].into(),
            ),
            fixed: true,
            ccd: false,
            pose: Pose::new(Vec3::new(10.05, 0.0, 0.0), Quat::IDENTITY),
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
        };
        let ball = EngineBody {
            key: BodyKey::Impactor(0),
            signature: 1,
            geometry: BodyGeometry::Sphere { radius: 0.1, mass: 5.0 },
            fixed: false,
            ccd,
            pose: Pose::new(Vec3::new(0.37, 0.03, 0.02), Quat::IDENTITY),
            velocity: Vec3::new(250.0, 0.0, 0.0),
            angular_velocity: Vec3::ZERO,
        };
        e.sync_bodies(&[wall, ball]);
        let mut max_x = f64::MIN;
        let mut hits = 0;
        for _ in 0..12 {
            let contacts = e.step(1.0 / 60.0, &[], &[]);
            hits += contacts.iter().filter(|c| c.chunk == 0).count();
            let m = e.motion();
            max_x = max_x.max(m.iter().find(|m| m.key == BodyKey::Impactor(0)).unwrap().pose.position.x);
            assert_eq!(e.last_substeps, 1, "one engine step per frame");
        }
        println!("ccd {ccd}: impactor reached x {max_x:.3} (wall face at 10.0), {hits} wall contacts");
        if ccd {
            assert!(max_x < 10.0 - 0.1 + 0.02, "the CCD impactor tunnelled or embedded: {max_x}");
            assert!(hits > 0, "its wall contacts are reported (stress loads)");
        } else {
            assert!(max_x > 10.1, "control: without CCD the impactor passes the wall ({max_x})");
        }
    }
}

/// Impulses change a driven body's velocity: linear `J / m`, angular `I^-1 L`.
#[test]
fn impulses_change_the_velocity() {
    let Ok(mut e) = RubbleEngine::new(Vec3::ZERO) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let k = BodyKey::Cluster(1);
    e.sync_bodies(&[cube(k, Vec3::ZERO, 0, 6.0)]);
    // A cube of side 1 and mass 6: I = m s^2 / 6 = 1.
    e.step(1.0 / 60.0, &[], &[(k, Vec3::new(0.0, 12.0, 0.0), Vec3::new(0.0, 0.0, 0.5))]);
    let m = e.motion()[0];
    println!("{:?} {:?}", m.velocity, m.angular_velocity);
    assert!((m.velocity - Vec3::new(0.0, 2.0, 0.0)).norm() < 1e-4, "{:?}", m.velocity);
    assert!((m.angular_velocity - Vec3::new(0.0, 0.0, 0.5)).norm() < 1e-3, "{:?}", m.angular_velocity);
}

/// `restore` returns every body to the saved instant, and the redone steps repeat
/// bit for bit.
#[test]
fn restore_repeats_the_frame_exactly() {
    let Ok(mut e) = RubbleEngine::new(Vec3::new(0.0, 0.0, -9.81)) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    e.add_ground(0.0);
    let mut falling = cube(BodyKey::Cluster(2), Vec3::new(0.3, 0.0, 2.0), 1, 50.0);
    falling.angular_velocity = Vec3::new(0.5, 1.0, 0.0);
    e.sync_bodies(&[cube(BodyKey::Cluster(1), Vec3::new(0.0, 0.0, 0.5), 0, 100.0), falling]);
    let dt = 1.0 / 60.0;
    for _ in 0..20 {
        e.step(dt, &[], &[]);
    }
    e.save();
    let run = |e: &mut RubbleEngine| {
        let mut contacts = 0;
        for _ in 0..30 {
            contacts += e.step(dt, &[BodyKey::Cluster(1)], &[]).len();
        }
        (e.motion(), contacts)
    };
    let (first, c1) = run(&mut e);
    e.restore();
    e.save();
    let (second, c2) = run(&mut e);
    e.discard();
    assert_eq!(c1, c2);
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.pose, b.pose);
        assert_eq!(a.velocity.to_array(), b.velocity.to_array());
    }
}

/// A fast ram through a free wall. With the impact owned by the stress solve, the
/// coupled run matches the reference world: the ram keeps ~35 m/s, the wall breaks into
/// the same fragments, momentum is conserved. (PhysX coupled, REALTIME_ROADMAP A11:
/// 35.05 m/s, 6 fragments, 24 bonds.)
#[test]
#[ignore = "about a minute in release: the explicit stress solve of 0.5 s twice"]
fn ram_through_free_wall_matches_the_reference_world() {
    let scene = ram_scene();
    let Some(mut w) = coupled(&scene) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let mut reference = World::new(&scene);
    let t = std::time::Instant::now();
    let ro = reference.run();
    let ref_secs = t.elapsed().as_secs_f64();
    let p0 = momentum(&w);
    let t = std::time::Instant::now();
    let o = w.run();
    let secs = t.elapsed().as_secs_f64();
    let p1 = momentum(&w);
    let (vr, vp) = (reference.impactors[0].velocity.y, w.impactors[0].velocity.y);
    println!(
        "reference: ram {vr:.2} m/s, {} fragments, {} broken bonds ({ref_secs:.1} s); rubble-coupled: ram {vp:.2} m/s, {} fragments, {} broken bonds, {} island frames (max {} bodies), {} redone ({secs:.1} s)",
        ro.values["fragments"], ro.values["broken_bonds"], o.values["fragments"], o.values["broken_bonds"], w.island_frames, w.max_island_bodies, w.redone_frames
    );
    println!("engine body-frames {}, island body-frames {}", w.engine_body_frames, w.island_body_frames);
    assert!(w.island_frames > 0);
    assert!((vp - vr).abs() < 0.05 * vr, "ram {vp} vs {vr}");
    assert!((p1 - p0).norm() < 1e-3 * p0.norm(), "momentum {p1:?} vs {p0:?}");
    let (fr, fp) = (ro.values["fragments"], o.values["fragments"]);
    assert!((fp - fr).abs() <= 0.25 * fr, "fragments {fp} vs {fr}");
}

/// A free (unsupported) column standing on the ground: rubble carries it, and the
/// ground reaction it reports must reach the stress solve, so the bottom joint carries
/// the weight of the four chunks above it (statics: N = 4 m g), as in the standalone
/// world. Nothing about this scene breaks the rigid assumption, so rubble integrates
/// the column every frame.
#[test]
#[ignore = "about 30 s in release: two explicit runs of 1 s"]
fn free_column_on_the_ground_carries_its_weight_through_rubble_contacts() {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("free_column", "five free chunks standing on the ground");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let chunks = grid(Vec3::new(-0.1, -0.1, 0.0), [1, 1, 5], Vec3::splat(0.2), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("column", chunks, bonds));
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.5, material: "concrete".into() });
    s.sim.duration = 1.0;
    let base_force = |w: &mut World| {
        let mut sum = 0.0;
        let mut n = 0;
        while w.solver.time < s.sim.duration {
            w.step_frame();
            if w.solver.time > 0.5 {
                let b = w.solver.bonds[0].iter().find(|b| b.geometry.a.min(b.geometry.b) == 0 && b.geometry.a.max(b.geometry.b) == 1).unwrap();
                sum += b.force.lin.z;
                n += 1;
            }
        }
        sum / n as f64
    };
    let mass = stress_ref::structure::Structure::from_scene(&s, 0).chunks[0].mass;
    let expected = -4.0 * mass * 9.81;
    let standalone = base_force(&mut World::new(&s));
    let Some(mut w) = coupled(&s) else {
        eprintln!("no wgpu adapter; skipping");
        return;
    };
    let rubble = base_force(&mut w);
    println!(
        "base joint: expected {expected:.1} N, standalone {standalone:.1} N, rubble {rubble:.1} N; engine body-frames {}, island {}",
        w.engine_body_frames, w.island_body_frames
    );
    assert_eq!(w.island_body_frames, 0);
    assert!(w.engine_body_frames > 0);
    assert!((standalone - expected).abs() < 0.01 * expected.abs(), "standalone {standalone} vs {expected}");
    assert!((rubble - expected).abs() < 0.01 * expected.abs(), "rubble {rubble} vs {expected}");
    let z = w.solver.clusters[0].com_world().z;
    assert!((z - 0.5).abs() < 0.01, "column centre at {z}");
}

/// Linear momentum of every body (hidden deformation carries none: mean-axis frame).
fn momentum(w: &World) -> Vec3 {
    let clusters = w.solver.clusters.iter().filter(|c| !c.anchored).fold(Vec3::ZERO, |a, c| a + c.velocity * c.mass);
    w.impactors.iter().fold(clusters, |a, m| a + m.velocity * m.mass)
}

/// The scene of `stress-physx`'s `physx_integration.rs`: a free 1 x 0.2 x 1.1 m
/// concrete wall, a 200 kg steel ram at 40 m/s, no gravity, established methods.
fn ram_scene() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("rubble_ram_free_wall", "free concrete wall hit by a steel ram");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.5;
    s.sim.methods = Methods::established();
    s
}
