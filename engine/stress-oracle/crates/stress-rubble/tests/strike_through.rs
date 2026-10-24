//! A shot through the fracture lab's cuboid house wall, forward only: with strike
//! prediction on, the wall's breaks are made at the end of the frame before the shot
//! reaches it and the wall splits before the engine steps, so the shot meets the pieces
//! and carries on through; off, the engine's contact meets the whole wall first and the
//! shot rebounds off it, however many bonds break after (what PhysX's internal correction
//! pass re-simulates, done here without re-simulating).
//!
//! The lab's scene and ball: the cuboid house as stress-web imports it, its 500 kg ball
//! (radius 0.4 m) at the lab's default 18 m/s, fired level at the middle of a wall. Each
//! mode with prediction on and off:
//! - `blast_parity` on the resident `GpuWorld` (the lab's "Stress solver"), with
//!   `GpuWorld::set_strike_prediction` (`threshold_predict`);
//! - the fracture world's `threshold` and `load_path`, with
//!   `ThresholdWorld::set_strike_prediction`;
//! - `impact_local`: on, at 18 m/s (below its ballistic limit for this wall) the ball comes
//!   to rest against the wall, and at 36 m/s it goes through.
//!
//! Through: the ball gets clear past the wall before it ever turns back. Rebounded: it
//! turns back before that.
//!
//! The pairs run at the lab's default speed. Much faster (36 m/s: 0.6 m a frame) the ball
//! gets through even with prediction off: the engine steps once a frame with no swept
//! collision, so within one step the ball sinks past the wall's middle and the contact
//! pushes it out on the far side (the engine's, not a fracture).
//!
//!   cargo test --release -p stress-rubble --features gpu --test strike_through
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Methods, Scene};
use stress_rubble::threshold::ThresholdWorld;

const RADIUS: f64 = 0.4;
const MASS: f64 = 500.0;
const FRAMES: usize = 90;

/// The lab's cuboid house as stress-web loads it, under `profile`.
fn house(profile: Profile) -> Scene {
    let (_lab, json) = engine_fracture::samples::cuboid_stress_scene(None).expect("cuboid house");
    let scene = stress_ref::shardwright::import_value_with(&json, "cuboid_houses", true).expect("import");
    scene.validate().expect("valid scene");
    let mut scene = scene.with_derived();
    scene.sim.methods = Methods::established();
    profile.apply_scene(&mut scene);
    scene.materials.entry("projectile".into()).or_insert_with(stress_ref::builders::elastic_steel);
    scene
}

/// The ball, level along +x at the middle of the house's -x wall, 1.5 m short of it; and
/// the wall's chunk centres' x (its near side).
fn ball(scene: &Scene, speed: f64) -> (ImpactorDesc, f64) {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for c in scene.bodies.iter().flat_map(|b| &b.chunks) {
        for k in 0..3 {
            lo[k] = lo[k].min(c.center[k]);
            hi[k] = hi[k].max(c.center[k]);
        }
    }
    let desc = ImpactorDesc {
        name: "shot".into(),
        shape: ImpactorShape::Sphere { radius: RADIUS },
        mass: MASS,
        position: [lo[0] - 1.5 - RADIUS, 0.5 * (lo[1] + hi[1]), 1.2],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [speed, 0.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "projectile".into(),
        crush: None,
    };
    (desc, lo[0])
}

/// The ball's path against the struck wall (its chunks' centres at `wall_x`): how far
/// along +x it got before it first turned back (or by the end), whether it turned back,
/// and its slowest velocity along +x.
#[derive(Debug, Default)]
struct Path {
    wall_x: f64,
    reached_x: f64,
    turned_back: bool,
    min_vx: f64,
}

impl Path {
    fn new(wall_x: f64) -> Path {
        Path { wall_x, reached_x: f64::NEG_INFINITY, turned_back: false, min_vx: f64::INFINITY }
    }

    fn record(&mut self, x: f64, vx: f64) {
        self.min_vx = self.min_vx.min(vx);
        if vx < -0.5 {
            self.turned_back = true;
        }
        if !self.turned_back {
            self.reached_x = self.reached_x.max(x);
        }
    }

    /// Through the wall: clear of it (its centre a metre past the wall's chunks' centres,
    /// more than the wall and the ball's radius) before it ever turned back.
    fn through(&self) -> bool {
        self.reached_x > self.wall_x + 1.0
    }

    /// Rebounded off it: turned back before getting clear of it.
    fn bounced(&self) -> bool {
        self.turned_back && !self.through()
    }
}

fn gpu_world(gpu: &Gpu, profile: Profile, predict: bool, speed: f64) -> Path {
    let scene = house(profile);
    let (desc, wall_x) = ball(&scene, speed);
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = profile.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    // As stress-web sets it up for the lab.
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(profile);
    w.set_strike_prediction(predict).expect("strike prediction");
    w.add_impactor(gpu, stress_ref::world::Impactor::from_desc(&desc, &scene)).expect("shot");
    let mut path = Path::new(wall_x);
    for _ in 0..FRAMES {
        w.step_frame(gpu).expect("frame");
        w.sync(gpu);
        let i = &w.solver.impactors[0];
        path.record(i.pose.position.x, i.velocity.x);
    }
    path
}

fn fracture_world(gpu: &Gpu, profile: Profile, predict: bool, speed: f64) -> Path {
    let scene = house(profile);
    let (desc, wall_x) = ball(&scene, speed);
    let mut w = ThresholdWorld::with_profile(gpu, &scene, RADIUS, MASS, profile).expect("fracture world");
    w.set_strike_prediction(gpu, predict).expect("strike prediction");
    assert!(w.launch(gpu, &desc));
    let mut path = Path::new(wall_x);
    for _ in 0..FRAMES {
        w.step_frame(gpu).expect("frame");
        let (p, v, _) = w.shot_states(gpu)[0];
        path.record(p[0], v[0]);
    }
    path
}

fn check(name: &str, off: Path, on: Path) {
    println!("{name}: off {off:?}; on {on:?}");
    assert!(off.bounced(), "{name}, strike prediction off: the ball rebounds off the whole wall ({off:?})");
    assert!(on.through(), "{name}, strike prediction on: the ball carries on through the broken wall ({on:?})");
}

#[test]
fn blast_parity_ball_goes_through_only_with_strike_prediction() {
    let gpu = Gpu::new().expect("GPU");
    check("blast_parity (GpuWorld)", gpu_world(&gpu, Profile::BlastParity, false, 18.0), gpu_world(&gpu, Profile::BlastParity, true, 18.0));
}

#[test]
fn threshold_ball_goes_through_only_with_strike_prediction() {
    let gpu = Gpu::new().expect("GPU");
    check("threshold", fracture_world(&gpu, Profile::Threshold, false, 18.0), fracture_world(&gpu, Profile::Threshold, true, 18.0));
}

#[test]
fn load_path_ball_goes_through_only_with_strike_prediction() {
    let gpu = Gpu::new().expect("GPU");
    check("load_path", fracture_world(&gpu, Profile::LoadPath, false, 18.0), fracture_world(&gpu, Profile::LoadPath, true, 18.0));
}

/// The analytic strike decides by the wall's ballistic limit. At 18 m/s (below it) the
/// ball comes to rest against the wall, never rebounding (off: it rebounds); at 36 m/s
/// (above it) it goes through.
#[test]
fn impact_local_ball_goes_through_above_its_limit_and_never_rebounds() {
    let gpu = Gpu::new().expect("GPU");
    let off = fracture_world(&gpu, Profile::ImpactLocal, false, 18.0);
    let below = fracture_world(&gpu, Profile::ImpactLocal, true, 18.0);
    let above = fracture_world(&gpu, Profile::ImpactLocal, true, 36.0);
    println!("impact_local: 18 m/s off {off:?}; on {below:?}; 36 m/s on {above:?}");
    assert!(off.bounced(), "strike prediction off: the ball rebounds off the whole wall ({off:?})");
    assert!(!below.turned_back && !below.through() && below.min_vx > -0.05, "below the limit the ball stops at the wall, no rebound ({below:?})");
    assert!(above.through(), "above the limit the ball goes through ({above:?})");
}
