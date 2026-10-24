//! The in-step contact fracture (`Profile::contact_fracture`; stress-gpu fracture.slang
//! `fr_cs_*`): inside rubble's step, after its contacts are made and before it solves them,
//! rubble's own contacts are the strikes. A contact closing on a structure faster than
//! `strike_speed` is judged by the profile's stress solver (the threshold rule, the load
//! path, the analytic strike); the bonds break by its verdict and the structure splits, the
//! moved contacts are re-anchored to the pieces, and the step solves the striker against
//! them. Or nothing breaks, the whole structure takes the contact and the striker rebounds.
//! Forward only: no rollback, no second solve, no detection of its own.
//!
//! For each solver (`threshold`, `load_path`, `impact_local`):
//! - Must bounce: a wall a thousand times stronger takes the strike whole, nothing breaks,
//!   and the ball does not get through: rubble's contact stops it (it is inelastic).
//! - Through: the ball goes through a wall the solver breaks, with the pass; without it
//!   (and no end-of-frame prediction) the same ball is stopped by the same wall.
//! - The verdict is the solver's own: the threshold rule breaks exactly the bonds its share
//!   condemns; the analytic strike's outcome and ballistic limit are its host twin's; the
//!   load path's strike load and increment are its host twins'.
//! - A strike by debris (a flying block, no shot) is found and judged.
//! - After the split no energy comes from nowhere (the ball's kinetic energy is never
//!   exceeded). Momentum is not checked: rubble's contact loses some even when nothing
//!   breaks (measured 2026-10-24: a 300 kg ball at 15 m/s into a free 528 kg wall a
//!   thousand times stronger, no gravity: 17 %), with the pass on or off.
//! - Runs repeat bit for bit.
//!
//! And: pieces resting on the ground stay at rest after the split (their contacts
//! re-anchored); the load path agrees with the explicit reference where the reference is
//! clear (stopped at 15 m/s, through at 120 and 250 m/s). Between, its strike load (the
//! rigid-structure upper bound, applied quasi-statically) over-breaks
//! (engine/docs/stress-threshold-gpu.md).
//!
//!   cargo test --release -p stress-rubble --features gpu --test contact_fracture
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_ref::capability::{Profile, CONTACT_SWEEPS};
use stress_ref::impact_local::{self, Impact, Target};
use stress_ref::load_path::{Dof, IMPACT_LOWER};
use stress_ref::scene::{ImpactorDesc, Methods, Scene};
use stress_rubble::scenes::{shot_at, shot_wall};
use stress_rubble::threshold::{FractureOptions, ThresholdWorld};

const R: f64 = 0.05;
const SOLVERS: [Profile; 3] = [Profile::Threshold, Profile::LoadPath, Profile::ImpactLocal];

fn steel_mass(r: f64) -> f64 {
    7850.0 * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3)
}

fn prepared(scene: Scene, profile: Profile) -> Scene {
    let mut scene = scene.with_derived();
    scene.sim.methods = Methods::established();
    profile.apply_scene(&mut scene);
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
    /// Its last velocity along the path.
    speed: f64,
}

impl Path {
    fn new(wall: f64) -> Path {
        Path { wall, reached: f64::NEG_INFINITY, turned_back: false, speed: 0.0 }
    }

    fn record(&mut self, x: f64, v: f64) {
        self.speed = v;
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

    /// Not through, and no longer moving on into the wall.
    fn held(&self) -> bool {
        !self.through() && self.speed < 0.5
    }
}

struct Run {
    path: Path,
    broken: u32,
    strikes: u32,
    contact_breaks: u32,
    digest: u64,
}

/// The fracture world of `scene` under `profile`, the in-step contact fracture `contact`,
/// the end-of-frame strike prediction off, the shots TOI CCD bodies either way (a 250 m/s
/// ball crosses a 0.2 m wall in a step: without them it would pass with no contact),
/// `desc` launched.
fn world(gpu: &Gpu, scene: &Scene, desc: &ImpactorDesc, profile: Profile, contact: bool) -> ThresholdWorld {
    let radius = match desc.shape {
        stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
        _ => unreachable!(),
    };
    let options = FractureOptions { contact: Some(contact), toi_ccd: Some(true) };
    let mut w = ThresholdWorld::with_options(gpu, scene, radius, desc.mass, profile, options).expect("fracture world");
    w.set_strike_prediction(gpu, false).expect("no end-of-frame prediction");
    assert!(w.launch(gpu, desc));
    w
}

/// `desc` fired along `axis` at `scene` (prepared for `profile`), `frames` frames.
fn run(gpu: &Gpu, scene: &Scene, desc: &ImpactorDesc, axis: usize, profile: Profile, contact: bool, frames: usize) -> Run {
    let mut w = world(gpu, scene, desc, profile, contact);
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

/// Frames until the pass's first strike (at most `limit`); whether there was one.
fn until_struck(gpu: &Gpu, w: &mut ThresholdWorld, limit: usize) -> bool {
    for _ in 0..limit {
        w.step_frame(gpu).expect("frame");
        if w.fracture.counters(gpu).contact_strikes > 0 {
            return true;
        }
    }
    false
}

/// The strike's record is rubble's contact: the ball's mass (the wall is anchored, the
/// contact through the ball's centre: no rotation), its closing speed, the Hertz pulse
/// (`pi m v / (2 tau)`, `tau` api.rs `hertz_duration`, the shot's modulus as the ingest's
/// table has it and the chunk's plane-strain one); and the load path's increment on the
/// GPU against the host's sweeps from rest on the same loads (f32).
#[test]
fn contact_strike_matches_its_host_twins() {
    let gpu = Gpu::new().expect("GPU");
    let scene = prepared(shot_wall(2, true), Profile::LoadPath);
    let speed = 60.0;
    let mass = steel_mass(R);
    let mut w = world(&gpu, &scene, &ball(speed), Profile::LoadPath, true);
    assert_eq!(w.contact_fracture(), CONTACT_SWEEPS);
    let wall = w.tables.chunk_body[0] as usize;
    assert!(until_struck(&gpu, &mut w, 30));
    assert_eq!(w.fracture.counters(&gpu).contact_strikes, 1, "one strike");
    let rec = w.fracture.contact_strikes(&gpu)[wall];
    let p = scene.material("projectile");
    let c = scene.material("concrete");
    let e_chunk = c.youngs_modulus * scene.sim.stiffness_scale / (1.0 - c.poisson_ratio * c.poisson_ratio);
    let e_star = 1.0 / (1.0 / p.youngs_modulus + 1.0 / e_chunk);
    let tau = stress_ref::api::hertz_duration(mass, R, e_star, speed);
    let peak = std::f64::consts::PI * mass * speed / (2.0 * tau);
    println!("strike: {rec:?}; host peak {peak:.1} N");
    assert!((rec.m_eff as f64 - mass).abs() <= 1e-5 * mass, "effective mass {} against the ball's {mass}", rec.m_eff);
    assert!((rec.speed as f64 - speed).abs() <= 1e-3 * speed, "closing speed {}", rec.speed);
    assert!((rec.peak as f64 - peak).abs() <= 1e-4 * peak, "peak {} against {peak}", rec.peak);
    assert!(((rec.hertz as f64) - mass * speed / tau).abs() <= 1e-4 * mass * speed / tau, "Hertz force {}", rec.hertz);
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

/// Must bounce, whichever solver: a wall a thousand times stronger takes the 120 m/s
/// strike whole; nothing breaks and the ball is held (rubble's contact stops it).
#[test]
fn every_solver_holds_a_strong_wall() {
    let gpu = Gpu::new().expect("GPU");
    for profile in SOLVERS {
        let wall = prepared(stronger(shot_wall(2, true), 1000.0), profile);
        let r = run(&gpu, &wall, &ball(120.0), 1, profile, true, 30);
        println!("{}: {:?}, broken {}, strikes {}", profile.name(), r.path, r.broken, r.strikes);
        assert!(r.strikes > 0, "{}: the strike is judged", profile.name());
        assert_eq!(r.broken, 0, "{}: nothing breaks", profile.name());
        assert!(r.path.held(), "{}: the ball is held", profile.name());
    }
}

/// Through, whichever solver: at 250 m/s each breaks the 0.2 m wall within the step the
/// ball reaches it, and the ball goes on through the pieces; without the pass (and no
/// end-of-frame prediction) the same ball is held by the same wall: the step meets it
/// whole, whatever breaks after.
#[test]
fn every_solver_lets_the_ball_through_a_wall_it_breaks_only_in_the_step() {
    let gpu = Gpu::new().expect("GPU");
    for profile in SOLVERS {
        let wall = prepared(shot_wall(2, true), profile);
        let on = run(&gpu, &wall, &ball(250.0), 1, profile, true, 30);
        let off = run(&gpu, &wall, &ball(250.0), 1, profile, false, 30);
        println!("{}: on {:?} strikes {} breaks {}; off {:?}", profile.name(), on.path, on.strikes, on.contact_breaks, off.path);
        assert!(on.strikes > 0 && on.contact_breaks > 0 && on.path.through(), "{}: struck, broken in the step, through", profile.name());
        assert_eq!(off.strikes, 0);
        assert!(off.path.held(), "{}: without the pass the ball is held", profile.name());
    }
}

/// The threshold rule's verdict in the step is its own: the bonds it breaks are exactly
/// those whose share of the strike's Hertz force (`F min(1, A / (2 pi d^2))`, the anchored
/// wall's chunk boxes where they are) beats their tensile capacity (f32: a bond within
/// 1e-4 of its capacity may go either way).
#[test]
fn the_threshold_rule_breaks_exactly_the_bonds_its_share_condemns() {
    let gpu = Gpu::new().expect("GPU");
    let scene = prepared(shot_wall(2, true), Profile::Threshold);
    let mut w = world(&gpu, &scene, &ball(60.0), Profile::Threshold, true);
    assert!(until_struck(&gpu, &mut w, 30));
    let wall = w.tables.chunk_body[0] as usize;
    let rec = w.fracture.contact_strikes(&gpu)[wall];
    let intact = w.fracture.bonds_intact(&gpu);
    let c = w.fracture.counters(&gpu);
    let centre = |k: u32| {
        let g = w.tables.geometry[k as usize][0];
        [g[0] as f64, g[1] as f64, g[2] as f64]
    };
    let (mut condemned, mut borderline) = (Vec::new(), 0);
    for (i, b) in w.tables.bonds.iter().enumerate() {
        let (pa, pb) = (centre(b[0]), centre(b[1]));
        let d2: f64 = (0..3).map(|k| (0.5 * (pa[k] + pb[k]) - rec.contact[k] as f64).powi(2)).sum();
        let area = f32::from_bits(b[3]) as f64;
        let reach = rec.hertz as f64 * (area / (2.0 * std::f64::consts::PI * d2.max(1e-12))).min(1.0);
        let ratio = reach / f32::from_bits(b[2]) as f64;
        if (ratio - 1.0).abs() < 1e-4 {
            borderline += 1;
        } else if ratio > 1.0 {
            condemned.push(i);
        }
    }
    println!("Hertz force {:.1} N: {} condemned, {borderline} borderline, {} broken in the step", rec.hertz, condemned.len(), c.contact_breaks);
    assert!(!condemned.is_empty());
    assert!(condemned.iter().all(|&i| !intact[i]), "every condemned bond broke");
    assert!((c.contact_breaks as usize).abs_diff(condemned.len()) <= borderline, "and only those");
}

/// The analytic strike's verdict in the step is its own: on the step's strike (its
/// effective mass, closing speed, the striker's face and modulus as the record has them,
/// the GPU's thickness along the path), the host's closed forms give the same outcome and
/// ballistic limit, below and above it.
#[test]
fn the_analytic_strike_judges_as_its_host_twin() {
    let gpu = Gpu::new().expect("GPU");
    let scene = prepared(shot_wall(2, true), Profile::ImpactLocal);
    let m = scene.material("concrete");
    let target = Target { density: m.density, youngs_modulus: m.youngs_modulus * scene.sim.stiffness_scale, poisson_ratio: m.poisson_ratio, tensile: m.tensile_strength, compressive: m.compressive_strength, shear: m.cohesion.min(m.shear_cap.unwrap_or(f64::INFINITY)), dif: m.dif };
    let mut outcomes = Vec::new();
    for speed in [30.0, 100.0, 250.0] {
        let mut w = world(&gpu, &scene, &ball(speed), Profile::ImpactLocal, true);
        assert!(until_struck(&gpu, &mut w, 30), "{speed} m/s: struck");
        let wall = w.tables.chunk_body[0] as usize;
        let rec = w.fracture.contact_strikes(&gpu)[wall];
        let out = w.fracture.strike_records(&gpu)[wall];
        let host = impact_local::strike(
            &Impact { mass: rec.mass as f64, reduced_mass: rec.m_eff as f64, radius: rec.face as f64, shot_modulus: rec.modulus as f64, speed: rec.speed as f64, thickness: out.thickness as f64 },
            &target,
        );
        let host_outcome = (host.damage as u32) | (host.crater as u32) << 1 | (host.spall as u32) << 2 | (host.perforate as u32) << 3;
        println!("{speed} m/s: thickness {:.4}, v_bl {:.2} (host {:.2}), outcome {:#x} (host {host_outcome:#x})", out.thickness, out.ballistic_speed, host.ballistic_speed, out.outcome);
        assert!((out.thickness as f64 - 0.2).abs() < 1e-3, "the wall's thickness along the path");
        assert_eq!(out.outcome, host_outcome, "{speed} m/s: the outcome");
        assert!((out.ballistic_speed as f64 - host.ballistic_speed).abs() <= 1e-3 * host.ballistic_speed, "{speed} m/s: the ballistic limit");
        outcomes.push(out.outcome);
    }
    assert!(outcomes[0] & 8 == 0 && outcomes[2] & 8 != 0, "below the limit no hole, above it one");
}

/// Debris strikes too, with no shot, whichever solver: a free concrete block (2 x 2 x 2
/// chunks of 0.1 m) at 8 m/s into a 0.4 m wall is a strike; against a wall a thousand
/// times stronger nothing breaks. (Debris is not a CCD body: much faster, a block crosses
/// the wall within a step and rubble's narrowphase finds it already through.)
#[test]
fn a_flying_block_strikes_a_wall() {
    let gpu = Gpu::new().expect("GPU");
    let with_block = |k: f64, profile: Profile| {
        use stress_ref::builders::{auto_bonds, body, grid};
        use stress_ref::math::Vec3;
        let mut s = shot_wall(4, true);
        let chunks = grid(Vec3::new(-0.05, -0.6, 0.45), [2, 2, 2], Vec3::splat(0.1), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut block = body("block", chunks, bonds);
        block.linear_velocity = [0.0, 8.0, 0.0];
        s.bodies.push(block);
        prepared(stronger(s, k), profile)
    };
    for profile in SOLVERS {
        for k in [1.0, 1000.0] {
            let scene = with_block(k, profile);
            let mut w = ThresholdWorld::with_contact_fracture(&gpu, &scene, R, 1.0, profile, Some(true)).expect("fracture world");
            w.set_strike_prediction(&gpu, false).expect("no prediction");
            for _ in 0..20 {
                w.step_frame(&gpu).expect("frame");
            }
            let c = w.fracture.counters(&gpu);
            println!("{} strength x{k}: strikes {}, contact breaks {}, broken {}", profile.name(), c.contact_strikes, c.contact_breaks, c.broken);
            assert!(c.contact_strikes > 0, "{}: the block's strike is one of the step's contacts", profile.name());
            assert_eq!(c.error, 0);
            if k > 1.0 {
                assert_eq!(c.contact_breaks, 0, "{}: the strong wall holds", profile.name());
            }
        }
    }
}

/// The kinetic energy (J) of every body.
fn energy(gpu: &Gpu, w: &ThresholdWorld) -> f64 {
    let bodies: Vec<engine_schema::BodyState> = gpu.read_range(w.engine.sim.body_buffer(), 0, w.tables.body_count() as usize);
    let mut ke = 0.0;
    for b in bodies.iter().filter(|b| b.inv_mass > 0.0) {
        let m = 1.0 / b.inv_mass as f64;
        let v = b.linear_vel.map(|x| x as f64);
        ke += 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        // The spin about the principal axes (the body's frame; the angular velocity is the
        // world's).
        let q = stress_ref::math::Quat { w: b.orientation[3] as f64, x: b.orientation[0] as f64, y: b.orientation[1] as f64, z: b.orientation[2] as f64 };
        let wl = q.conjugate().rotate(stress_ref::math::Vec3::new(b.angular_vel[0] as f64, b.angular_vel[1] as f64, b.angular_vel[2] as f64)).to_array();
        for k in 0..3 {
            if b.inv_inertia[k] > 0.0 {
                ke += 0.5 * wl[k] * wl[k] / b.inv_inertia[k] as f64;
            }
        }
    }
    ke
}

/// After the step's split nothing comes from nowhere, whichever solver: into a free wall
/// (no gravity, no ground) the ball's kinetic energy is never exceeded, through the strike,
/// the split, the re-anchored contacts and the frames after (anchors left on the old bodies
/// would push the pieces apart).
#[test]
fn a_struck_free_wall_makes_no_energy() {
    let gpu = Gpu::new().expect("GPU");
    for profile in SOLVERS {
        let mut scene = shot_wall(2, false);
        scene.gravity = [0.0; 3];
        let scene = prepared(scene, profile);
        let mass = steel_mass(R);
        let speed = 250.0;
        let mut w = world(&gpu, &scene, &ball(speed), profile, true);
        let ke0 = 0.5 * mass * speed * speed;
        let mut worst = 0.0f64;
        for _ in 0..30 {
            w.step_frame(&gpu).expect("frame");
            worst = worst.max(energy(&gpu, &w));
        }
        let c = w.fracture.counters(&gpu);
        println!("{}: strikes {} breaks {}, largest kinetic energy {worst:.1} of {ke0:.1}", profile.name(), c.contact_strikes, c.contact_breaks);
        assert!(c.contact_strikes > 0 && c.contact_breaks > 0, "{}: struck and broken", profile.name());
        assert!(worst <= 1.001 * ke0, "{}: kinetic energy {worst} exceeds the ball's {ke0}", profile.name());
    }
}

/// The struck pieces keep their contacts: a free wall standing on the ground, struck
/// through its middle; the split moves its chunks' colliders to new bodies within the step,
/// and the rows of the ones resting on the ground are re-anchored to them, so the bottom
/// course far from the hole stays at rest (wrong anchors would throw it).
#[test]
fn pieces_on_the_ground_stay_at_rest_after_the_split() {
    let gpu = Gpu::new().expect("GPU");
    for profile in [Profile::Threshold, Profile::LoadPath] {
        let mut scene = shot_wall(2, false);
        scene.ground = Some(stress_ref::scene::GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
        let scene = prepared(scene, profile);
        let mut w = world(&gpu, &scene, &ball(250.0), profile, true);
        assert!(until_struck(&gpu, &mut w, 30));
        let c = w.fracture.counters(&gpu);
        assert!(c.contact_breaks > 0 && c.pieces > 0, "{}: the strike split the wall", profile.name());
        for _ in 0..3 {
            w.step_frame(&gpu).expect("frame");
        }
        // The bottom course's chunks at the wall's ends (more than 0.3 m from the hole).
        let bodies: Vec<engine_schema::BodyState> = gpu.read_range(w.engine.sim.body_buffer(), 0, w.tables.body_count() as usize);
        let owner = w.fracture.chunk_bodies(&gpu);
        let mut fastest = 0.0f32;
        for (k, g) in w.tables.geometry.iter().enumerate() {
            let c = g[0];
            if c[2] < 0.1 && (c[0] - 0.05).abs() > 0.3 {
                let b = &bodies[owner[k] as usize];
                fastest = fastest.max((b.linear_vel[0].powi(2) + b.linear_vel[1].powi(2) + b.linear_vel[2].powi(2)).sqrt());
            }
        }
        println!("{}: pieces {}, the far bottom course's fastest {fastest:.3} m/s", profile.name(), c.pieces);
        assert!(fastest < 0.5, "{}: the far bottom course moves at {fastest} m/s", profile.name());
    }
}

/// The lab's cuboid house and ball at the lab's 18 m/s under the load path: through the
/// wall with the in-step pass, not without it; a house a thousand times stronger holds it.
#[test]
fn the_lab_ball_goes_through_the_house_only_with_the_pass() {
    let gpu = Gpu::new().expect("GPU");
    let home = prepared(house(), Profile::LoadPath);
    let on = run(&gpu, &home, &lab_ball(&home, 18.0), 0, Profile::LoadPath, true, 90);
    let off = run(&gpu, &home, &lab_ball(&home, 18.0), 0, Profile::LoadPath, false, 90);
    println!("on: {:?}, strikes {}, contact breaks {}; off: {:?}", on.path, on.strikes, on.contact_breaks, off.path);
    assert!(on.strikes > 0 && on.contact_breaks > 0 && on.path.through(), "with the pass: struck, broken, through");
    assert_eq!(off.strikes, 0);
    assert!(!off.path.through(), "without it the ball does not get through");
    let strong = prepared(stronger(house(), 1000.0), Profile::LoadPath);
    let held = run(&gpu, &strong, &lab_ball(&strong, 18.0), 0, Profile::LoadPath, true, 90);
    println!("strong house: {:?}, broken {}, strikes {}", held.path, held.broken, held.strikes);
    assert!(held.strikes > 0 && held.broken == 0 && !held.path.through(), "a strong house holds the ball");
}

/// Where the explicit reference is clear, the load path's pass agrees: the 4.1 kg ball
/// stops at the 0.2 m wall at 15 m/s and goes through at 120 and 250 m/s, in both.
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
        let pass = run(&gpu, &prepared(shot_wall(2, true), Profile::LoadPath), &ball(speed), 1, Profile::LoadPath, true, 15);
        println!("{speed} m/s: reference {reference:?} broken {}; pass {:?} broken {}", o.observation().values["broken_bonds"], pass.path, pass.broken);
        assert_eq!(reference.through(), through, "the reference at {speed} m/s");
        assert_eq!(pass.path.through(), through, "the pass at {speed} m/s");
    }
}

#[test]
fn runs_repeat_bit_for_bit() {
    let gpu = Gpu::new().expect("GPU");
    for profile in SOLVERS {
        let wall = prepared(shot_wall(2, true), profile);
        let a = run(&gpu, &wall, &ball(250.0), 1, profile, true, 30);
        let b = run(&gpu, &wall, &ball(250.0), 1, profile, true, 30);
        assert_eq!((a.digest, a.broken, a.strikes), (b.digest, b.broken, b.strikes), "{}", profile.name());
    }
    let home = prepared(house(), Profile::LoadPath);
    let a = run(&gpu, &home, &lab_ball(&home, 18.0), 0, Profile::LoadPath, true, 60);
    let b = run(&gpu, &home, &lab_ball(&home, 18.0), 0, Profile::LoadPath, true, 60);
    assert_eq!((a.digest, a.broken, a.strikes), (b.digest, b.broken, b.strikes));
}
