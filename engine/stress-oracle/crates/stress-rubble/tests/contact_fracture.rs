//! The in-step contact fracture (`Profile::contact_fracture`; stress-gpu fracture.slang
//! `fr_cs_*`): within rubble's step, where its broadphase ends, a body about to strike a
//! structure has the strike's load solved by the load path before the step solves; the
//! bonds break by their own strength and the structure splits, or nothing breaks and the
//! striker does not get through. Forward only: no rollback, no second solve.
//!
//! - Against its host twins: the strike's load (the Hertz peak of the impulse that stops
//!   the striker against the whole structure) and the load path's increment.
//! - Must not break: a wall a thousand times stronger takes the strike whole, and the
//!   ball does not get through, on a box wall and on the lab's cuboid house.
//! - The lab's cuboid house at the lab's 18 m/s: through with the pass, not without it.
//! - A strike by debris (no shot): a flying concrete block against a wall.
//! - Against the explicit reference (the CPU oracle, `realtime` configuration): where the
//!   reference is clear (stopped at 15 m/s, through at 120 and 250 m/s) the pass agrees.
//!   Between, it is not yet faithful: it over-breaks (measured 2026-10-24 on this wall:
//!   67 bonds at 15 m/s against the reference's 10, 415 at 40 m/s against 9, where the
//!   reference stops the ball and the pass lets it through). The strike's load is the
//!   rigid-structure upper bound, applied quasi-statically (engine/docs/
//!   stress-threshold-gpu.md, "In-step contact fracture").
//! - Runs repeat bit for bit.
//!
//!   cargo test --release -p stress-rubble --features gpu --test contact_fracture
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::capability::{Profile, CONTACT_SWEEPS};
use stress_ref::load_path::{Dof, IMPACT_LOWER};
use stress_ref::scene::{ImpactorDesc, Methods, Scene};
use stress_rubble::scenes::{shot_at, shot_wall};
use stress_rubble::threshold::ThresholdWorld;

const R: f64 = 0.05;

fn steel_mass(r: f64) -> f64 {
    7850.0 * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3)
}

fn prepared(scene: Scene) -> Scene {
    let mut scene = scene.with_derived();
    scene.sim.methods = Methods::established();
    Profile::LoadPath.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

/// Every structure material `k` times stronger.
fn stronger(mut scene: Scene, k: f64) -> Scene {
    for m in scene.materials.values_mut() {
        m.tensile_strength *= k;
        m.compressive_strength *= k;
        m.cohesion *= k;
        m.shear_cap = m.shear_cap.map(|c| c * k);
    }
    scene
}

/// The lab's cuboid house as stress-web loads it.
fn house() -> Scene {
    let (_lab, json) = engine_fracture::samples::cuboid_stress_scene(None).expect("cuboid house");
    stress_ref::shardwright::import_value_with(&json, "cuboid_houses", true).expect("import")
}

/// The extent of a scene's chunk centres.
fn bounds(scene: &Scene) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for c in scene.bodies.iter().flat_map(|b| &b.chunks) {
        for k in 0..3 {
            lo[k] = lo[k].min(c.center[k]);
            hi[k] = hi[k].max(c.center[k]);
        }
    }
    (lo, hi)
}

/// A shot's path against the wall it is fired at (along `axis`, its near side's chunk
/// centres at `wall`): how far it got before it first turned back.
#[derive(Debug)]
struct Path {
    wall: f64,
    reached: f64,
    turned_back: bool,
}

impl Path {
    fn new(wall: f64) -> Path {
        Path { wall, reached: f64::NEG_INFINITY, turned_back: false }
    }

    fn record(&mut self, x: f64, v: f64) {
        if v < -0.5 {
            self.turned_back = true;
        }
        if !self.turned_back {
            self.reached = self.reached.max(x);
        }
    }

    /// Clear of the wall (a metre past its near side's chunk centres: its thickness and
    /// the ball's radius) before it ever turned back.
    fn through(&self) -> bool {
        self.reached > self.wall + 1.0
    }
}

struct Run {
    path: Path,
    broken: u32,
    strikes: u32,
    contact_breaks: u32,
    digest: u64,
}

/// `desc` fired along `axis` at `scene` (prepared), `frames` frames, the in-step contact
/// fracture `contact` (the old end-of-frame strike prediction off).
fn run(gpu: &Gpu, scene: &Scene, desc: &ImpactorDesc, axis: usize, contact: bool, frames: usize) -> Run {
    let radius = match desc.shape {
        stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
        _ => unreachable!(),
    };
    let mut w = ThresholdWorld::with_contact_fracture(gpu, scene, radius, desc.mass, Profile::LoadPath, Some(contact)).expect("fracture world");
    w.set_strike_prediction(gpu, false).expect("no end-of-frame prediction");
    assert!(w.launch(gpu, desc));
    let (lo, _) = bounds(scene);
    let mut path = Path::new(lo[axis]);
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
        let (p, v, _) = w.shot_states(gpu)[0];
        path.record(p[axis], v[axis]);
    }
    let c = w.fracture.counters(gpu);
    assert_eq!(c.error, 0, "no error bits");
    let mut h: u64 = 0xcbf29ce484222325;
    for x in w.fracture.bonds_intact(gpu).iter().map(|&b| b as u32).chain(w.fracture.chunk_bodies(gpu)) {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    Run { path, broken: c.broken, strikes: c.contact_strikes, contact_breaks: c.contact_breaks, digest: h }
}

/// The 4.1 kg ball (radius 0.05 m) at `speed` along +y at the 0.2 m wall's middle, more
/// than a frame's travel away.
fn ball(speed: f64) -> ImpactorDesc {
    shot_at(R, steel_mass(R), [0.05, -(1.5 * speed / 60.0 + 0.2), 0.55], [0.0, speed, 0.0])
}

/// The lab's ball (500 kg, radius 0.4 m) at `speed` along +x at the cuboid house's -x wall.
fn lab_ball(scene: &Scene, speed: f64) -> ImpactorDesc {
    let (lo, hi) = bounds(scene);
    shot_at(0.4, 500.0, [lo[0] - 1.9, 0.5 * (lo[1] + hi[1]), 1.2], [speed, 0.0, 0.0])
}

/// The strike's load and the load path's increment on the GPU against the host: the
/// Hertz peak of the impulse that stops the ball against the anchored wall
/// (`pi m v / (2 tau)`, `tau` api.rs `hertz_duration`), and every bond's force under the
/// GPU's increment against the host's sweeps from rest on the same loads (f32).
#[test]
fn contact_strike_matches_its_host_twins() {
    let gpu = Gpu::new().expect("GPU");
    let scene = prepared(shot_wall(2, true));
    let speed = 60.0;
    let mass = steel_mass(R);
    let mut w = ThresholdWorld::with_contact_fracture(&gpu, &scene, R, mass, Profile::LoadPath, Some(true)).expect("fracture world");
    assert_eq!(w.contact_fracture(), CONTACT_SWEEPS);
    assert!(!w.strike_prediction(), "strikes are the in-step pass's, not predicted at the end of a frame");
    assert!(w.launch(&gpu, &ball(speed)));
    let wall = w.tables.chunk_body[0] as usize;
    for _ in 0..30 {
        w.step_frame(&gpu).expect("frame");
        if w.fracture.counters(&gpu).contact_strikes > 0 {
            break;
        }
    }
    assert_eq!(w.fracture.counters(&gpu).contact_strikes, 1, "one strike");
    let rec = w.fracture.contact_strikes(&gpu)[wall];
    // The host's load: the ball's mass (the wall is anchored), its speed (gravity's frame
    // or two turns the path by a millionth), Hertz's modulus of the shot (its Young's
    // modulus, as the ingest's table has it) and the chunk's plane-strain one.
    let p = scene.material("projectile");
    let c = scene.material("concrete");
    let e_chunk = c.youngs_modulus * scene.sim.stiffness_scale / (1.0 - c.poisson_ratio * c.poisson_ratio);
    let e_star = 1.0 / (1.0 / p.youngs_modulus + 1.0 / e_chunk);
    let tau = stress_ref::api::hertz_duration(mass, R, e_star, speed);
    let peak = std::f64::consts::PI * mass * speed / (2.0 * tau);
    println!("strike: {rec:?}; host peak {peak:.1} N");
    assert!((rec.force as f64 - peak).abs() <= 1e-3 * peak, "peak {} against {peak}", rec.force);
    assert!((rec.direction[1] as f64 - 1.0).abs() < 1e-4, "along the path");
    // The increment: the host's sweeps from rest on the GPU's loads and holds.
    let net = w.load_path_network();
    let inc = w.fracture.contact_increments(&gpu);
    let dof = |a: [f32; 4], b: [f32; 4]| -> Dof { [a[0] as f64, a[1] as f64, a[2] as f64, b[0] as f64, b[1] as f64, b[2] as f64] };
    let f: Vec<Dof> = inc.iter().map(|c| dof(c[2], c[3])).collect();
    let fixed: Vec<bool> = inc.iter().map(|c| c[2][3] != 0.0).collect();
    let x_gpu: Vec<Dof> = inc.iter().map(|c| dof(c[0], c[1])).collect();
    // The bonds as they were when the strike was solved: all intact.
    let alive = vec![true; net.bonds.len()];
    let mut x = vec![[0.0; 6]; net.chunks.len()];
    let mut d = vec![[0.0; 6]; net.chunks.len()];
    assert!(net.sweep(&f, &fixed, &alive, &mut x, &mut d, CONTACT_SWEEPS, IMPACT_LOWER));
    let size = |q: &stress_ref::bond::Local6| q.lin.norm() + q.ang.norm() / 0.05;
    let scale = (0..net.bonds.len()).map(|i| size(&net.force(i, &x))).fold(0.0, f64::max);
    let worst = (0..net.bonds.len())
        .map(|i| {
            let (a, b) = (net.force(i, &x), net.force(i, &x_gpu));
            (a.lin - b.lin).norm() + (a.ang - b.ang).norm() / 0.05
        })
        .fold(0.0, f64::max);
    println!("increment: largest bond force {scale:.1}, GPU against host {worst:.3e}");
    assert!(scale > 0.0 && worst <= 2e-3 * scale, "GPU increment differs from the host's by {worst:e} (of {scale:e})");
}

/// A wall a thousand times stronger: the strike is solved, nothing breaks, the ball does
/// not get through (a box wall at 120 m/s, the lab's cuboid house at 18 m/s).
#[test]
fn a_strong_enough_wall_takes_the_strike_whole() {
    let gpu = Gpu::new().expect("GPU");
    let wall = prepared(stronger(shot_wall(2, true), 1000.0));
    let a = run(&gpu, &wall, &ball(120.0), 1, true, 30);
    println!("box wall: {:?}, broken {}, strikes {}", a.path, a.broken, a.strikes);
    assert!(a.strikes > 0 && a.broken == 0 && !a.path.through(), "box wall: the strike solved, nothing broken, the ball held");
    let home = prepared(stronger(house(), 1000.0));
    let b = run(&gpu, &home, &lab_ball(&home, 18.0), 0, true, 90);
    println!("cuboid house: {:?}, broken {}, strikes {}", b.path, b.broken, b.strikes);
    assert!(b.strikes > 0 && b.broken == 0 && !b.path.through(), "cuboid house: the strike solved, nothing broken, the ball held");
}

/// The lab's cuboid house and ball at the lab's 18 m/s: through the wall with the in-step
/// pass (its strike broke the bonds, before the step), not without it (no prediction
/// either: the step meets the whole wall).
#[test]
fn the_lab_ball_goes_through_the_house_only_with_the_pass() {
    let gpu = Gpu::new().expect("GPU");
    let home = prepared(house());
    let on = run(&gpu, &home, &lab_ball(&home, 18.0), 0, true, 90);
    let off = run(&gpu, &home, &lab_ball(&home, 18.0), 0, false, 90);
    println!("on: {:?}, strikes {}, contact breaks {}; off: {:?}", on.path, on.strikes, on.contact_breaks, off.path);
    assert!(on.strikes > 0 && on.contact_breaks > 0 && on.path.through(), "with the pass: struck, broken, through");
    assert_eq!(off.strikes, 0);
    assert!(!off.path.through(), "without it the ball does not get through");
}

/// Debris strikes too, with no shot: a free concrete block (2 x 2 x 2 chunks of 0.1 m) at
/// 8 m/s into a 0.4 m wall is a strike the pass solves; against a wall a thousand times
/// stronger nothing breaks.
#[test]
fn a_flying_block_strikes_a_wall() {
    let gpu = Gpu::new().expect("GPU");
    let with_block = |k: f64| {
        use stress_ref::builders::{auto_bonds, body, grid};
        use stress_ref::math::Vec3;
        let mut s = shot_wall(4, true);
        let chunks = grid(Vec3::new(-0.05, -0.6, 0.45), [2, 2, 2], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut block = body("block", chunks, bonds);
        block.linear_velocity = [0.0, 8.0, 0.0];
        s.bodies.push(block);
        prepared(stronger(s, k))
    };
    for (k, breaks) in [(1.0, true), (1000.0, false)] {
        let scene = with_block(k);
        let mut w = ThresholdWorld::with_contact_fracture(&gpu, &scene, R, 1.0, Profile::LoadPath, Some(true)).expect("fracture world");
        for _ in 0..20 {
            w.step_frame(&gpu).expect("frame");
        }
        let c = w.fracture.counters(&gpu);
        println!("strength x{k}: strikes {}, contact breaks {}, broken {}", c.contact_strikes, c.contact_breaks, c.broken);
        assert!(c.contact_strikes > 0, "the block's strike is found among the step's pairs");
        assert_eq!(c.contact_breaks > 0, breaks);
        assert_eq!(c.error, 0);
    }
}

/// Where the explicit reference is clear, the pass agrees: the 4.1 kg ball stops at the
/// 0.2 m wall at 15 m/s and goes through at 120 and 250 m/s, in both.
#[test]
fn agrees_with_the_explicit_reference_where_it_is_clear() {
    let gpu = Gpu::new().expect("GPU");
    for (speed, through) in [(15.0, false), (120.0, true), (250.0, true)] {
        let mut s = shot_wall(2, true);
        s.impactors.push(ball(speed));
        let s = stress_rubble::scenes::realtime(s, 0.25).with_derived();
        let mut o = stress_ref::world::World::new(&s);
        let mut reference = Path::new(0.05);
        for _ in 0..15 {
            o.step_frame();
            let i = &o.impactors[0];
            reference.record(i.pose.position.y, i.velocity.y);
        }
        let pass = run(&gpu, &prepared(shot_wall(2, true)), &ball(speed), 1, true, 15);
        println!("{speed} m/s: reference {reference:?} broken {}; pass {:?} broken {}", o.observation().values["broken_bonds"], pass.path, pass.broken);
        assert_eq!(reference.through(), through, "the reference at {speed} m/s");
        assert_eq!(pass.path.through(), through, "the pass at {speed} m/s");
    }
}

#[test]
fn runs_repeat_bit_for_bit() {
    let gpu = Gpu::new().expect("GPU");
    let home = prepared(house());
    let a = run(&gpu, &home, &lab_ball(&home, 18.0), 0, true, 60);
    let b = run(&gpu, &home, &lab_ball(&home, 18.0), 0, true, 60);
    assert_eq!((a.digest, a.broken, a.strikes), (b.digest, b.broken, b.strikes));
}
