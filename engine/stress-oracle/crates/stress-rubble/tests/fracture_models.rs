//! The fracture world's models (stress-gpu fracture.slang; `capability::Profile::LoadPath`
//! and `Profile::ImpactLocal`) on the GPU: each against its f64 host twin (stress-ref
//! load_path.rs, impact_local.rs), and the behaviour each model exists for, checked against
//! statics and the closed forms' own definitions (nothing recorded from a run):
//!
//! - the load path carries a load to where it is resisted: a weight at a cantilever's tip
//!   breaks its root, which the contact-force test never does; an idle structure keeps its
//!   bonds; runs repeat bit for bit;
//! - the analytic strike holes a wall exactly at and above its ballistic limit, the shot
//!   leaving faster the faster it came; equal momentum at a lower speed does not hole it;
//!   thicker walls go from perforation to spall to crater; the outcome does not depend on
//!   the frame rate; runs repeat bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test fracture_models
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::capability::Profile;
use stress_ref::impact_local::{self, Impact, Target};
use stress_ref::load_path::{Dof, LOWER};
use stress_ref::scene::{Methods, Scene};
use stress_rubble::scenes::{cantilever_plate, shot_at, shot_wall};
use stress_rubble::threshold::ThresholdWorld;

fn world(gpu: &Gpu, scene: Scene, profile: Profile, radius: f64, mass: f64) -> ThresholdWorld {
    let mut scene = scene.with_derived();
    scene.sim.methods = Methods::established();
    profile.apply_scene(&mut scene);
    ThresholdWorld::with_profile(gpu, &scene, radius, mass, profile).expect("fracture world")
}

fn dof(x: &[[f32; 4]; 2]) -> Dof {
    [x[0][0] as f64, x[0][1] as f64, x[0][2] as f64, x[1][0] as f64, x[1][1] as f64, x[1][2] as f64]
}

/// FNV-1a of the bonds' states, the chunks' bodies and the counters.
fn digest(gpu: &Gpu, w: &ThresholdWorld) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h = (h ^ x as u64).wrapping_mul(0x100000001b3);
        }
    };
    for b in w.fracture.bonds_intact(gpu) {
        eat(&[b as u8]);
    }
    for c in w.fracture.chunk_bodies(gpu) {
        eat(&c.to_le_bytes());
    }
    let c = w.fracture.counters(gpu);
    eat(format!("{c:?}").as_bytes());
    h
}

// ---- Load path ----

/// One frame of the GPU's sweeps from the warm start, against the host's twin given the
/// same loads (the GPU's relieved ones) and holds: the bonds' forces agree to f32.
#[test]
fn load_path_matches_its_host_twin() {
    let gpu = Gpu::new().expect("GPU");
    let mut w = world(&gpu, cantilever_plate(8), Profile::LoadPath, 0.1, 1.0);
    let net = w.load_path_network();
    // The warm start is the host's equilibrium under gravity: the GPU's first frame (no
    // contact) keeps it.
    w.step_frame(&gpu).expect("frame");
    let x_gpu: Vec<Dof> = w.fracture.lp_displacements(&gpu).iter().map(dof).collect();
    let loads = w.fracture.lp_loads(&gpu);
    let f: Vec<Dof> = loads.iter().map(dof).collect();
    let fixed: Vec<bool> = loads.iter().map(|l| l[0][3] != 0.0).collect();
    assert_eq!(fixed, net.fixed(), "the GPU holds the chunks the host does");
    // Gravity, nothing else: each chunk's load is its weight.
    for (k, c) in net.chunks.iter().enumerate() {
        let wz = -9.81 * c.mass;
        assert!((f[k][2] - wz).abs() <= 1e-5 * wz.abs() && f[k][0].abs() + f[k][1].abs() <= 1e-5 * wz.abs(), "chunk {k}: load {:?}, weight {wz}", f[k]);
    }
    let mut x: Vec<Dof> = w.tables.lp_init.iter().map(dof).collect();
    let mut d = vec![[0.0; 6]; x.len()];
    let alive = vec![true; net.bonds.len()];
    let sweeps = match w.fracture.models().0 {
        1 => stress_ref::capability::LOAD_PATH_SWEEPS,
        _ => unreachable!(),
    };
    assert!(net.sweep(&f, &fixed, &alive, &mut x, &mut d, sweeps, LOWER));
    let scale = (0..net.bonds.len()).map(|i| net.force(i, &x)).map(|q| q.lin.norm() + q.ang.norm()).fold(0.0, f64::max);
    let mut worst = 0.0f64;
    for i in 0..net.bonds.len() {
        let (a, b) = (net.force(i, &x), net.force(i, &x_gpu));
        worst = worst.max((a.lin - b.lin).norm() + (a.ang - b.ang).norm());
    }
    println!("largest bond force {scale:.1}, GPU against host {worst:.3e}");
    assert!(worst <= 1e-3 * scale, "GPU bond forces differ from the host's by {worst:e} (of {scale:e})");
}

/// A weight resting on a cantilever plate's tip breaks the plate at its root (the bonds
/// beside the fixed column) under the load path, and nothing else first. The contact-force
/// test never breaks the root: it sees only the contact force where the weight rests, never
/// the moment it makes at the root.
#[test]
fn a_weight_at_the_tip_breaks_the_root() {
    let gpu = Gpu::new().expect("GPU");
    // The first frame bonds broke, and whether they were all at the root; the digest.
    let run = |profile: Profile| -> (Option<(usize, bool)>, u64) {
        // 250 kg at 0.72 m: the root section's bending moment (with the plate's own) about
        // 1.13 times its three bonds' capacity (concrete's 3 MPa on 0.1 m squares), the next
        // section's 0.93 times.
        let (radius, mass) = (0.15, 250.0);
        let mut w = world(&gpu, cantilever_plate(8), profile, radius, mass);
        // Resting on the tip (the plate's top at z = 1.1), set down from 0.5 mm: it lands
        // slower than gravity adds in a frame, a sustained load, not an impact.
        assert!(w.launch(&gpu, &shot_at(radius, mass, [0.72, 0.0, 1.1 + radius + 5e-4], [0.0; 3])));
        let net = w.load_path_network();
        let at_root = |i: usize| net.chunks[net.bonds[i].a].support || net.chunks[net.bonds[i].b].support;
        let mut first = None;
        for f in 0..180 {
            w.step_frame(&gpu).expect("frame");
            let intact = w.fracture.bonds_intact(&gpu);
            let broken: Vec<usize> = (0..intact.len()).filter(|&i| !intact[i]).collect();
            if first.is_none() && !broken.is_empty() {
                let at = |i: usize| (net.chunks[net.bonds[i].a].centre + net.chunks[net.bonds[i].b].centre) * 0.5;
                println!("{}: frame {f}, first breaks {:?}", profile.name(), broken.iter().map(|&i| (i, at(i).x, at(i).y)).collect::<Vec<_>>());
                first = Some((f, broken.iter().all(|&i| at_root(i))));
            }
            if profile == Profile::Threshold && broken.iter().any(|&i| at_root(i)) {
                panic!("the contact-force test broke a root bond in frame {f}");
            }
        }
        (first, digest(&gpu, &w))
    };
    let (lp, d1) = run(Profile::LoadPath);
    let (frame, root) = lp.expect("the load path breaks the plate");
    assert!(root, "every first break is at the root");
    assert!(frame < 60, "within a second of the weight landing (frame {frame})");
    run(Profile::Threshold);
    let (_, d2) = run(Profile::LoadPath);
    assert_eq!(d1, d2, "the run repeats bit for bit");
}

/// The house-1story pack standing for 10 s under the load path keeps every bond.
#[test]
fn an_idle_house_keeps_its_bonds() {
    let gpu = Gpu::new().expect("GPU");
    let mut scene = stress_rubble::scenes::house();
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    let mut w = world(&gpu, scene, Profile::LoadPath, 0.1, 1.0);
    for _ in 0..600 {
        w.step_frame(&gpu).expect("frame");
    }
    let c = w.fracture.counters(&gpu);
    println!("house: {} bonds broken, peak failure index {:.3}, error {}", c.broken, c.peak_utilization, c.error);
    assert_eq!((c.broken, c.error), (0, 0));
}

// ---- Local impact ----

const R: f64 = 0.05;

fn steel_mass(r: f64) -> f64 {
    7850.0 * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3)
}

struct Strike {
    record: stress_gpu::fracture::StrikeRecord,
    /// The shot's velocity along the path (+y) `frames` after the strike.
    exit: f64,
    digest: u64,
}

/// A shot of `mass` (radius [`R`]) at `speed` along +y into `scene`'s first wall (its face
/// at y = 0), at the centre of a chunk column; the first strike's record, then `frames`
/// frames on.
fn strike(gpu: &Gpu, scene: Scene, profile: Profile, mass: f64, speed: f64, frames: usize) -> Option<Strike> {
    let hz = 1.0 / scene.sim.frame_dt;
    let mut w = world(gpu, scene, profile, R, mass);
    // More than a frame's travel away: the strike is predicted at the end of a frame.
    let y = -(1.5 * speed / hz + 0.2);
    assert!(w.launch(gpu, &shot_at(R, mass, [0.05, y, 0.55], [0.0, speed, 0.0])));
    let wall = w.tables.chunk_body[0] as usize;
    let mut record = None;
    for _ in 0..(2.0 * hz) as usize {
        w.step_frame(gpu).expect("frame");
        let c = w.fracture.counters(gpu);
        if c.bounces + c.craters + c.spalls + c.perforations > 0 {
            record = Some(w.fracture.strike_records(gpu)[wall]);
            break;
        }
    }
    let record = record?;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    assert_eq!(w.fracture.counters(gpu).error, 0, "no error bits");
    Some(Strike { record, exit: w.shot_states(gpu)[0].1[1], digest: digest(gpu, &w) })
}

fn outcome(r: &stress_gpu::fracture::StrikeRecord) -> &'static str {
    if r.outcome & 8 != 0 {
        "perforate"
    } else if r.outcome & 4 != 0 {
        "spall"
    } else if r.outcome & 2 != 0 {
        "crater"
    } else {
        "bounce"
    }
}

fn concrete_target(scene: &Scene) -> Target {
    let m = scene.material("concrete");
    Target { density: m.density, youngs_modulus: m.youngs_modulus * scene.sim.stiffness_scale, poisson_ratio: m.poisson_ratio, tensile: m.tensile_strength, compressive: m.compressive_strength, shear: m.cohesion.min(m.shear_cap.unwrap_or(f64::INFINITY)), dif: m.dif }
}

/// The host's strike on the GPU's thickness (the launch speed: gravity's frame or two
/// turns the path by a millionth).
fn host(scene: &Scene, mass: f64, speed: f64, r: &stress_gpu::fracture::StrikeRecord) -> impact_local::Strike {
    let p = scene.material("projectile");
    let e_shot = p.youngs_modulus * scene.sim.stiffness_scale / (1.0 - p.poisson_ratio * p.poisson_ratio);
    impact_local::strike(&Impact { mass, reduced_mass: mass, radius: R, shot_modulus: e_shot, speed, thickness: r.thickness as f64 }, &concrete_target(scene))
}

/// The GPU's closed forms against the host's on the same strike: the wall's thickness
/// along the path (two chunks, 0.2 m), the ballistic limit and the outcome.
#[test]
fn analytic_strike_matches_its_host_twin() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    for speed in [60.0, 150.0, 400.0] {
        let scene = shot_wall(2, true).with_derived();
        let s = strike(&gpu, scene.clone(), Profile::ImpactLocal, mass, speed, 0).expect("a strike");
        let h = host(&scene, mass, speed, &s.record);
        let r = &s.record;
        println!("{speed} m/s: GPU {r:?}; host V_bl {:.2}, residual {:.2}, outcome damage {} crater {} spall {} perforate {}", h.ballistic_speed, h.residual_speed, h.damage, h.crater, h.spall, h.perforate);
        assert!((r.thickness as f64 - 0.2).abs() <= 1e-3, "thickness {}", r.thickness);
        assert!((r.ballistic_speed as f64 - h.ballistic_speed).abs() <= 1e-4 * h.ballistic_speed, "V_bl {} against {}", r.ballistic_speed, h.ballistic_speed);
        let bits = (h.damage as u32) | (h.crater as u32) << 1 | (h.spall as u32) << 2 | (h.perforate as u32) << 3;
        assert_eq!(r.outcome, bits, "outcome");
        assert!((r.residual_speed as f64 - h.residual_speed).abs() <= 1e-3 * speed, "residual {} against {}", r.residual_speed, h.residual_speed);
    }
}

/// From half to twice the ballistic limit: the wall is holed exactly at and above it, and
/// the shot then leaves faster the faster it came, within 15 % of Recht and Ipson's
/// residual speed; below it the shot at most creeps on
/// (under a tenth of its speed: what a spall freed, the contact test then pushes out). At
/// 2000 Hz: the fracture world steps the engine once a frame with no swept collision, so a
/// shot crossing more than a wall and its own diameter in a frame passes through it.
#[test]
fn holed_at_and_above_the_ballistic_limit_and_exit_speed_rises() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    let fine = || {
        let mut s = shot_wall(2, true);
        s.sim.frame_dt = 1.0 / 2000.0;
        s
    };
    let v_bl = strike(&gpu, fine(), Profile::ImpactLocal, mass, 100.0, 0).expect("a strike").record.ballistic_speed as f64;
    let mut last: Option<f64> = None;
    for k in 0..7 {
        let speed = v_bl * (0.5 + 0.25 * k as f64) * if k == 2 { 1.001 } else { 1.0 };
        let s = strike(&gpu, fine(), Profile::ImpactLocal, mass, speed, 40).expect("a strike");
        let through = s.record.outcome & 8 != 0;
        println!("{speed:.1} m/s ({:.2} V_bl): {}, exit {:.2} m/s (Recht-Ipson {:.2})", speed / v_bl, outcome(&s.record), s.exit, s.record.residual_speed);
        assert_eq!(through, speed >= v_bl, "{speed} m/s against V_bl {v_bl}");
        if !through {
            assert!(s.exit < 0.1 * speed, "below the limit the shot at most creeps on (exit {})", s.exit);
        } else if speed > 1.1 * v_bl {
            // The engine's exit speed (the shot and its plug's chunks leaving together)
            // against Recht and Ipson's with the cone's mass: the chunks resolve the cone.
            let predicted = s.record.residual_speed as f64;
            assert!((s.exit - predicted).abs() <= 0.15 * predicted, "exit {} against Recht-Ipson {predicted}", s.exit);
            if let Some(l) = last {
                assert!(s.exit > l, "faster in, faster out: {} after {l}", s.exit);
            }
            last = Some(s.exit);
        }
    }
}

/// Equal momentum: a light shot at 300 m/s holes the wall, one 37.5 times heavier at 8 m/s
/// (the same momentum, a 37.5th of the energy) does not.
#[test]
fn equal_momentum_only_the_fast_light_shot_holes_the_wall() {
    let gpu = Gpu::new().expect("GPU");
    let light = steel_mass(R);
    let fast = strike(&gpu, shot_wall(2, true), Profile::ImpactLocal, light, 300.0, 0).expect("a strike");
    let slow = strike(&gpu, shot_wall(2, true), Profile::ImpactLocal, light * 37.5, 8.0, 0).expect("a strike");
    println!("fast: {:?}; slow: {:?}", fast.record, slow.record);
    assert!(fast.record.outcome & 8 != 0, "the fast light shot perforates");
    assert!(slow.record.outcome & 8 == 0, "the slow heavy one does not");
}

/// Thicker walls at 120 m/s: perforated, then spalled, then cratered (never a worse
/// outcome for a thicker wall); every wall's thickness is its chunk layers'.
#[test]
fn thicker_walls_go_from_perforation_to_spall_to_crater() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    let rank = |o: &str| match o {
        "perforate" => 0,
        "spall" => 1,
        "crater" => 2,
        _ => 3,
    };
    let mut seen = Vec::new();
    for layers in [1, 2, 4, 6, 8] {
        let s = strike(&gpu, shot_wall(layers, true), Profile::ImpactLocal, mass, 120.0, 0).expect("a strike");
        assert!((s.record.thickness as f64 - 0.1 * layers as f64).abs() <= 1e-3, "{layers} layers: thickness {}", s.record.thickness);
        seen.push((layers, outcome(&s.record)));
    }
    println!("{seen:?}");
    for w in seen.windows(2) {
        assert!(rank(w[0].1) <= rank(w[1].1), "a thicker wall does no worse: {seen:?}");
    }
    assert_eq!(seen[0].1, "perforate");
    assert!(seen.iter().any(|s| s.1 == "spall"), "a spall band: {seen:?}");
    assert_eq!(seen.last().unwrap().1, "crater");
}

/// The strike's outcome does not depend on the frame rate (30, 60, 120 Hz), and a run
/// repeats bit for bit.
#[test]
fn the_outcome_is_the_same_at_any_frame_rate_and_repeats() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    for speed in [100.0, 250.0] {
        let mut outcomes = Vec::new();
        for hz in [30.0, 60.0, 120.0] {
            let mut scene = shot_wall(2, true);
            scene.sim.frame_dt = 1.0 / hz;
            let s = strike(&gpu, scene, Profile::ImpactLocal, mass, speed, 0).expect("a strike");
            outcomes.push(s.record.outcome);
        }
        println!("{speed} m/s at 30, 60, 120 Hz: {outcomes:?}");
        assert!(outcomes.windows(2).all(|w| w[0] == w[1]), "{speed} m/s: {outcomes:?}");
    }
    let a = strike(&gpu, shot_wall(2, true), Profile::ImpactLocal, mass, 250.0, 30).expect("a strike");
    let b = strike(&gpu, shot_wall(2, true), Profile::ImpactLocal, mass, 250.0, 30).expect("a strike");
    assert_eq!((a.digest, a.exit.to_bits()), (b.digest, b.exit.to_bits()), "the run repeats bit for bit");
}


/// The momentum along the path (stress +y) of every body: shots, walls and pieces.
fn momentum_y(gpu: &Gpu, w: &ThresholdWorld) -> f64 {
    let bodies: Vec<engine_schema::BodyState> = gpu.read_range(w.engine.sim.body_buffer(), 0, w.tables.body_count() as usize);
    // rubble is Y-up: the stress frame's +y is rubble's -z.
    bodies.iter().filter(|b| b.inv_mass > 0.0).map(|b| -(b.linear_vel[2] as f64) / b.inv_mass as f64).sum()
}

/// At 60 Hz a shot crosses the 0.2 m wall in a fraction of a frame: below the limit the
/// strike itself resolves the contact (the shot comes to rest on the face), so it does not
/// pass; and into a free wall (no gravity) the shot's momentum is kept, through a spall and
/// through a perforation (the shot, its plug, the wall and every piece).
#[test]
fn a_strike_keeps_momentum_and_a_slow_shot_does_not_pass() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    let s = strike(&gpu, shot_wall(2, true), Profile::ImpactLocal, mass, 100.0, 30).expect("a strike");
    println!("anchored, 100 m/s: {}, exit {:.3} m/s", outcome(&s.record), s.exit);
    assert!(s.record.outcome & 8 == 0 && s.exit.abs() < 1.0, "below the limit the shot stays (exit {})", s.exit);
    for speed in [100.0, 250.0] {
        let mut scene = shot_wall(2, false);
        scene.gravity = [0.0; 3];
        let hz = 1.0 / scene.sim.frame_dt;
        let mut w = world(&gpu, scene, Profile::ImpactLocal, R, mass);
        assert!(w.launch(&gpu, &shot_at(R, mass, [0.05, -(1.5 * speed / hz + 0.2), 0.55], [0.0, speed, 0.0])));
        let p0 = mass * speed;
        for _ in 0..30 {
            w.step_frame(&gpu).expect("frame");
        }
        let c = w.fracture.counters(&gpu);
        let p = momentum_y(&gpu, &w);
        println!("free wall, {speed} m/s: strikes b{} c{} s{} p{}, momentum {p:.3} of {p0:.3}", c.bounces, c.craters, c.spalls, c.perforations);
        assert_eq!(c.bounces + c.craters + c.spalls + c.perforations, 1, "one strike");
        // Measured 2026-10-24: within 0.03 % (spall) and 0.09 % (perforation).
        assert!((p - p0).abs() <= 2e-3 * p0, "momentum {p} against {p0}");
    }
}

/// Two walls 0.3 m apart at 60 Hz (a frame's travel covers both): the shot strikes the
/// first, then the second with the speed the first left it, so it does not hole both.
#[test]
fn a_wall_row_is_struck_one_wall_at_a_time() {
    let gpu = Gpu::new().expect("GPU");
    let mass = steel_mass(R);
    let scene = stress_rubble::scenes::shot_row(2, 2, true);
    let speed = 250.0;
    let mut w = world(&gpu, scene, Profile::ImpactLocal, R, mass);
    assert!(w.launch(&gpu, &shot_at(R, mass, [0.05, -(1.5 * speed / 60.0 + 0.2), 0.55], [0.0, speed, 0.0])));
    for _ in 0..60 {
        w.step_frame(&gpu).expect("frame");
    }
    let c = w.fracture.counters(&gpu);
    println!("wall row, {speed} m/s: strikes b{} c{} s{} p{}, exit {:.2}", c.bounces, c.craters, c.spalls, c.perforations, w.shot_states(&gpu)[0].1[1]);
    assert_eq!(c.perforations, 1, "the first wall is holed, the second is not");
}
