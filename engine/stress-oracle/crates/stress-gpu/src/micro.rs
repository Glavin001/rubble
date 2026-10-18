//! The micro tier: tiny targeted scenes run from t = 0 through the production path
//! (`GpuWorld`, `GpuSolver`, `GpuCoupledSolver`) in seconds, each checked for the
//! semantic it exists for. Beside the accuracy gate (`gate::gate`) a case checks
//!
//! - its events (kind, bond) as sorted multisets against the reference's, and their
//!   times within one substep or twice the reference ensemble's own scatter;
//! - its fragment partition (chunk groups, ids relabelled);
//! - chunk velocities (momentum exchange) within the reference's spread;
//! - energy closure `E0 + W_ext = mechanical + dissipated` on the GPU;
//! - its path coverage (`coverage`): the bits it exists for must have been set, so a case
//!   can never pass without running its path.
//!
//! Metamorphic checks run a case under GPU-only scheduling knobs (`Knobs`) against its
//! default run: batch boundaries and sync cadence must give bit-identical state, probes
//! and events (the operation order is the same); wide islands and the contact plan horizon
//! re-associate f32 sums, so they are held to 1e-5 of each probe's range, equal counts and
//! (kind, bond) events, and times within one substep. GPU snapshots record every run's
//! outcome (the GPU is deterministic) and report any change; seeded fuzz scenes widen
//! the net. The reference runs are cached like the gate's (`gate::cached`, keyed by the
//! case and stress-ref's fingerprint).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;
use web_time::Instant;

use stress_ref::api::{ChildBody, ClusterMotion, ContactImpulse, EngineCoupledSolver, FrameInput, StressSolverApi};
use stress_ref::builders::{auto_bonds, body, bond_pull, chunk, grid, new_scene, probe};
use stress_ref::material::{Material, StaticFatigue};
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::observation::Observation;
use stress_ref::scene::{
    BodyDesc, ChunkSelector, CrushDesc, EventDesc, GroundDesc, ImpactorDesc, ImpactorShape, LoadDesc, ProbeKind, Scene, SectionComponent, SolveMode, Support, TimeFunction,
};
use stress_ref::solver::{ReferenceSolver, SolverEvent};
use stress_ref::world::World;

use crate::coupled::GpuStressSolverApi;
use crate::coverage as cov;
use crate::gate::{self, ReferenceSet};
use crate::gpu::Gpu;
use crate::world::GpuWorld;

/// Bump when the reference computation of the micro cases changes (cache key).
const MICRO_VERSION: u32 = 2;

// ------------------------------------------------------------------ knobs

/// GPU-only scheduling knobs (solver.rs): they change when the GPU stops and what it
/// re-lays, never the physics.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Knobs {
    /// `STRESS_GPU_WIDE_CHUNKS`: islands with at least this many chunks run wide.
    pub wide_chunks: Option<usize>,
    /// `STRESS_GPU_BATCH_ROUNDS`: per-substep rounds per submission at most.
    pub batch_rounds: Option<usize>,
    /// `STRESS_GPU_PLAN_STEPS`: steps a contact plan covers.
    pub plan_steps: Option<f64>,
    /// `STRESS_GPU_SYNC_EACH_STEP`: bonds synced after every step.
    pub sync_each_step: bool,
    /// `STRESS_GPU_DEVICE_CONNECT=0`: a disconnection always halts for the host's
    /// connectivity check (the path of islands too big for the device's).
    pub host_connect: bool,
}

impl Knobs {
    pub fn wide(n: usize) -> Knobs {
        Knobs { wide_chunks: Some(n), ..Default::default() }
    }
    pub fn batch(n: usize) -> Knobs {
        Knobs { batch_rounds: Some(n), ..Default::default() }
    }
    pub fn plan(x: f64) -> Knobs {
        Knobs { plan_steps: Some(x), ..Default::default() }
    }
    pub fn sync() -> Knobs {
        Knobs { sync_each_step: true, ..Default::default() }
    }
    pub fn host_connect() -> Knobs {
        Knobs { host_connect: true, ..Default::default() }
    }

    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if let Some(n) = self.wide_chunks {
            parts.push(format!("wide={n}"));
        }
        if let Some(n) = self.batch_rounds {
            parts.push(format!("batch={n}"));
        }
        if let Some(x) = self.plan_steps {
            parts.push(format!("plan={x}"));
        }
        if self.sync_each_step {
            parts.push("sync".into());
        }
        if self.host_connect {
            parts.push("host-connect".into());
        }
        if parts.is_empty() {
            "default".into()
        } else {
            parts.join(",")
        }
    }

    /// Whether a run under these knobs must match the default bit for bit: batch
    /// boundaries and sync cadence keep the operation order; wide islands and the plan
    /// horizon do not (summation order, re-split compensated positions).
    pub fn bitwise(&self) -> bool {
        self.wide_chunks.is_none() && self.plan_steps.is_none()
    }

    fn vars(&self) -> [(&'static str, Option<String>); 5] {
        [
            ("STRESS_GPU_WIDE_CHUNKS", self.wide_chunks.map(|n| n.to_string())),
            ("STRESS_GPU_BATCH_ROUNDS", self.batch_rounds.map(|n| n.to_string())),
            ("STRESS_GPU_PLAN_STEPS", self.plan_steps.map(|x| x.to_string())),
            ("STRESS_GPU_SYNC_EACH_STEP", self.sync_each_step.then(|| "1".to_string())),
            ("STRESS_GPU_DEVICE_CONNECT", self.host_connect.then(|| "0".to_string())),
        ]
    }
}

static KNOB_LOCK: Mutex<()> = Mutex::new(());

/// Restores the knob variables when dropped (also on a panic).
struct KnobGuard(Vec<(&'static str, Option<String>)>);

impl Drop for KnobGuard {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

/// Run `f` with exactly these knobs set (the knob variables not named are unset for the
/// run, so the default is canonical), then restore the environment. Serialized within
/// the process: the variables are global.
pub fn with_knobs<R>(k: &Knobs, f: impl FnOnce() -> R) -> R {
    let _lock = KNOB_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let vars = k.vars();
    let _guard = KnobGuard(vars.iter().map(|(n, _)| (*n, std::env::var(n).ok())).collect());
    for (name, value) in &vars {
        match value {
            Some(v) => std::env::set_var(name, v),
            None => std::env::remove_var(name),
        }
    }
    f()
}

// ------------------------------------------------------------------ cases

/// A change to the solver configuration (fields that are not scene fields), applied
/// identically to the reference's solver and the GPU's mirror.
#[derive(Clone, Copy)]
pub struct Tweak {
    pub id: &'static str,
    pub apply: fn(&mut ReferenceSolver),
}

fn short_active(s: &mut ReferenceSolver, t: f64) {
    s.config.active_time = t;
    for c in &mut s.clusters {
        c.active_timer = c.active_timer.min(t);
    }
}

fn active_10ms(s: &mut ReferenceSolver) {
    short_active(s, 0.01)
}

fn active_6ms(s: &mut ReferenceSolver) {
    short_active(s, 0.006)
}

/// An engine-coupled run: frames of contact impulses and engine motion.
#[derive(Clone)]
pub struct CoupledScript {
    pub scene: Scene,
    /// Run the GPU in adaptive mode (sleeping); the reference stays explicit.
    pub gpu_adaptive: bool,
    pub frames: usize,
    /// Stands for the closures in the cache key: bump when they change.
    pub inputs_version: u32,
    pub contacts: fn(usize, &Scene, f64) -> Vec<ContactImpulse>,
    pub motion: fn(usize, &[ChildBody]) -> Vec<ClusterMotion>,
}

#[derive(Clone)]
pub enum Script {
    World(Scene),
    Coupled(CoupledScript),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum EvKind {
    Cracked,
    Creaked,
    Broken,
    Split,
}

/// What a case must show.
#[derive(Clone, Default)]
pub struct Expect {
    /// Coverage bits that must be set (by the default run or a knob run).
    pub cover: u64,
    /// Events the reference and the GPU must both have at least this many of.
    pub min_events: Vec<(EvKind, usize)>,
    /// Compare chunk velocities (momentum exchange).
    pub velocities: bool,
    /// Check energy closure on the GPU.
    pub energy: bool,
}

#[derive(Clone)]
pub struct MicroCase {
    pub name: &'static str,
    pub semantic: &'static str,
    pub script: Script,
    pub tweak: Option<Tweak>,
    /// The knobs of the case's own run (default for all but the forced-path cases).
    pub base: Knobs,
    pub expect: Expect,
    /// Path-forcing runs, each checked against the base run (metamorphic).
    pub knob_runs: Vec<Knobs>,
}

impl MicroCase {
    fn world(name: &'static str, semantic: &'static str, scene: Scene, expect: Expect, knob_runs: Vec<Knobs>) -> MicroCase {
        MicroCase { name, semantic, script: Script::World(scene), tweak: None, base: Knobs::default(), expect, knob_runs }
    }
    fn with_tweak(mut self, t: Tweak) -> MicroCase {
        self.tweak = Some(t);
        self
    }
    fn with_base(mut self, k: Knobs) -> MicroCase {
        self.base = k;
        self
    }
    pub fn scene(&self) -> &Scene {
        match &self.script {
            Script::World(s) => s,
            Script::Coupled(c) => &c.scene,
        }
    }
}

fn expect(cover: u64) -> Expect {
    Expect { cover, ..Default::default() }
}

fn concrete() -> Material {
    stress_ref::showcases::static_load_concrete()
}

/// A scene of 0.2 m concrete cubes with short frames (fine segment boundaries).
fn scene(name: &str, description: &str, gravity: bool) -> Scene {
    let mut s = new_scene(name, description);
    s.materials.insert("concrete".into(), concrete());
    s.sim.frame_dt = 0.002;
    if !gravity {
        s.gravity = [0.0; 3];
    }
    s
}

/// A row of `n` cubes along x from the origin (centres at 0.1, 0.3, ...), the given
/// chunks fixed.
fn bar(name: &str, n: [usize; 3], fixed: &[usize]) -> BodyDesc {
    let mut chunks = grid(Vec3::new(0.0, -0.1, -0.1), n, Vec3::splat(0.2), "concrete");
    for &k in fixed {
        chunks[k].support = Support::Fixed;
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    body(name, chunks, bonds)
}

/// One free cube at `at` moving at `v`.
fn cube(name: &str, at: [f64; 3], v: [f64; 3]) -> BodyDesc {
    let mut b = body(name, vec![chunk(Vec3::ZERO, Vec3::splat(0.1), "concrete")], Vec::new());
    b.position = at;
    b.linear_velocity = v;
    b
}

fn point_force(body: &str, chunk: usize, direction: [f64; 3], magnitude: TimeFunction) -> LoadDesc {
    LoadDesc::PointForce { body: body.into(), chunk, direction, magnitude, point: None }
}

fn impactor(name: &str, shape: ImpactorShape, mass: f64, position: [f64; 3], velocity: [f64; 3]) -> ImpactorDesc {
    ImpactorDesc { name: name.into(), shape, mass, position, orientation: [0.0, 0.0, 0.0, 1.0], velocity, angular_velocity: [0.0; 3], material: "concrete".into(), crush: None }
}

fn disp(name: &str, body: &str, chunk: usize, axis: [f64; 3]) -> stress_ref::scene::ProbeDesc {
    probe(name, ProbeKind::ChunkDisplacement { body: body.into(), chunk, axis })
}

fn vel(name: &str, body: &str, chunk: usize, axis: [f64; 3]) -> stress_ref::scene::ProbeDesc {
    probe(name, ProbeKind::ChunkVelocity { body: body.into(), chunk, axis })
}

fn ground() -> GroundDesc {
    GroundDesc { height: 0.0, friction: 0.5, material: "concrete".into() }
}

/// A glass pair (bond_pull) whose static fatigue life at stress ratio `s` is `life`.
fn glass_pair(name: &str, s: f64, life: f64, magnitude: impl Fn(f64) -> TimeFunction, duration: f64) -> Scene {
    let mut m = Material::glass();
    let n = 16.0;
    let unit = StaticFatigue { exponent: n, test_time: 1.0 }.lifetime(s);
    m.static_fatigue = Some(StaticFatigue { exponent: n, test_time: life / unit });
    let h = 0.05;
    let f = s * m.tensile_strength * 4.0 * h * h;
    let mut sc = bond_pull(name, m, h, magnitude(f), duration);
    sc.sim.frame_dt = 0.005;
    sc
}

/// Two hexagonal prisms (convex hulls), flat faces towards each other.
fn hex_prism(name: &str, x: f64, v: f64) -> BodyDesc {
    let r = 0.1;
    let mut pts = Vec::new();
    for k in 0..6 {
        let a = std::f64::consts::PI / 6.0 + k as f64 * std::f64::consts::PI / 3.0;
        for z in [-0.1, 0.1] {
            pts.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    let mut c = chunk(Vec3::ZERO, Vec3::new(r * (std::f64::consts::PI / 6.0).cos(), r, 0.1), "concrete");
    c.hull = Some(pts);
    let mut b = body(name, vec![c], Vec::new());
    b.position = [x, 0.0, 0.0];
    b.linear_velocity = [v, 0.0, 0.0];
    b
}

fn wall_contact(scene: &Scene, chunk: usize, impulse: Vec3, approach: f64) -> ContactImpulse {
    let c = &scene.bodies[0].chunks[chunk];
    let dir = impulse * (1.0 / impulse.norm());
    ContactImpulse {
        structure: 0,
        chunk,
        point: Vec3::from_array(c.center) - dir * 0.09,
        impulse,
        other_mass: 300.0,
        approach_speed: approach,
        other_modulus: 30e9,
        other_radius: 0.2,
        crush: None,
        other: Some(1),
    }
}

fn coupled_contacts(f: usize, scene: &Scene, k: f64) -> Vec<ContactImpulse> {
    // The wall's top middle chunk (4 x 1 x 3, index 1 + 4 * 2), pushed along +y.
    let target = 9;
    match f {
        0..=1 => vec![wall_contact(scene, target, Vec3::new(0.0, 900.0 * k / (f as f64 + 1.0), 0.0), 8.0)],
        _ => vec![wall_contact(scene, target, Vec3::new(0.0, 40.0 * k, 0.0), 0.0)],
    }
}

fn coupled_motion(f: usize, clusters: &[ChildBody]) -> Vec<ClusterMotion> {
    // The engine's motion of every free cluster: a slow drift and spin.
    let t = (f + 1) as f64 / 60.0;
    clusters
        .iter()
        .filter(|c| !c.anchored)
        .map(|c| ClusterMotion {
            id: c.id,
            pose: Pose { position: Vec3::new(c.pose[0], c.pose[1], c.pose[2]) + Vec3::new(0.01, 0.0, 0.0), rotation: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.3 * t) },
            velocity: Vec3::new(0.6, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 0.3),
        })
        .collect()
}

fn coupled_scene(name: &str) -> Scene {
    let mut s = scene(name, "A wall struck through engine impulses, beside a free pair the engine moves.", true);
    s.sim.frame_dt = 1.0 / 60.0;
    let mut wall = grid(Vec3::new(0.0, -0.1, 0.0), [4, 1, 3], Vec3::splat(0.2), "concrete");
    for c in wall.iter_mut().filter(|c| c.center[2] < 0.15) {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&wall, |_, _| "concrete".into());
    s.bodies.push(body("wall", wall, bonds));
    let mut pair = bar("pair", [2, 1, 1], &[]);
    pair.position = [0.0, 2.0, 1.0];
    s.bodies.push(pair);
    s
}

/// The micro catalogue.
pub fn micro_cases() -> Vec<MicroCase> {
    let mut out = Vec::new();
    let x = [1.0, 0.0, 0.0];
    let y = [0.0, 1.0, 0.0];
    let z = [0.0, 0.0, 1.0];

    // ---- contact
    let block = |name: &str, desc: &str, speed: f64, mass: f64| {
        let mut s = scene(name, desc, false);
        s.bodies.push(bar("block", [2, 1, 1], &[1]));
        s.impactors.push(impactor("ram", ImpactorShape::Box { half_extents: [0.05; 3] }, mass, [-0.051, 0.0, 0.0], [speed, 0.0, 0.0]));
        s.probes.push(probe("ram_velocity", ProbeKind::ImpactorVelocity { impactor: "ram".into(), axis: x }));
        s.probes.push(disp("face_x", "block", 0, x));
        s.probes.push(probe(
            "axial",
            ProbeKind::SectionForce { body: "block".into(), point: [0.2, 0.0, 0.0], normal: x, region: None, component: SectionComponent::Normal },
        ));
        s.sim.duration = 0.006;
        s
    };
    let onset = block("impactor_onset", "A box 1 mm off a face, moving at 2 m/s: contact starts at first touch.", 2.0, 20.0);
    out.push(MicroCase::world(
        "impactor_onset",
        "an impactor 1 mm off a face pushes from first touch; its momentum goes into the block",
        onset,
        Expect { cover: cov::IMPACTOR_CONTACT | cov::CONTACT_PIPELINE | cov::PROBES, velocities: true, ..Default::default() },
        vec![Knobs::batch(1), Knobs::plan(1.0)],
    ));
    let mut crush = block("impactor_crush", "A soft box (crush cap) at 8 m/s: the cap limits the contact force.", 8.0, 50.0);
    crush.impactors[0].crush = Some(CrushDesc { max_force: 3.0e4, energy: 40.0 });
    out.push(MicroCase::world("impactor_crush", "a soft impactor's crush cap limits the force and books the crush energy", crush, expect(cov::IMPACTOR_CONTACT | cov::CRUSH), vec![Knobs::batch(1)]));

    let mut s = scene("impactor_ground", "A sphere dropped 1 mm onto the ground.", true);
    s.ground = Some(ground());
    s.bodies.push({
        let mut b = bar("post", [1, 1, 1], &[0]);
        b.position = [3.0, 0.0, 0.2];
        b
    });
    s.impactors.push(impactor("ball", ImpactorShape::Sphere { radius: 0.05 }, 5.0, [0.0, 0.0, 0.051], [0.0, 0.0, -0.5]));
    s.probes.push(probe("ball_z", ProbeKind::ImpactorPosition { impactor: "ball".into(), axis: z }));
    s.probes.push(probe("ball_vz", ProbeKind::ImpactorVelocity { impactor: "ball".into(), axis: z }));
    s.sim.duration = 0.01;
    out.push(MicroCase::world("impactor_ground", "an impactor lands on the ground and bounces", s, expect(cov::IMPACTOR_GROUND | cov::CONTACT_PIPELINE), vec![]));

    let mut s = scene("pair_closing", "Two free cubes 2 mm apart closing at 2 m/s.", false);
    s.bodies.push(cube("a", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
    s.bodies.push(cube("b", [0.202, 0.0, 0.0], [-1.0, 0.0, 0.0]));
    s.probes.push(vel("a_vx", "a", 0, x));
    s.probes.push(vel("b_vx", "b", 0, x));
    s.sim.duration = 0.006;
    out.push(MicroCase::world(
        "pair_closing",
        "two free chunks collide face on face and exchange momentum",
        s,
        Expect { cover: cov::PAIR_CONTACT | cov::CONTACT_PIPELINE | cov::PROBES, velocities: true, energy: true, ..Default::default() },
        vec![Knobs::batch(1), Knobs::plan(1.0), Knobs::wide(1)],
    ));

    let mut s = scene("pair_overlap", "Two free cubes overlapping by 1 um, closing at 0.1 m/s.", false);
    s.bodies.push(cube("a", [0.0, 0.0, 0.0], [0.05, 0.0, 0.0]));
    s.bodies.push(cube("b", [0.2 - 1e-6, 0.0, 0.0], [-0.05, 0.0, 0.0]));
    s.probes.push(vel("a_vx", "a", 0, x));
    s.probes.push(vel("b_vx", "b", 0, x));
    s.sim.duration = 0.004;
    out.push(MicroCase::world(
        "pair_overlap",
        "a pre-existing overlap is kept as an offset: only the approach after it pushes",
        s,
        Expect { cover: cov::PAIR_CONTACT, velocities: true, ..Default::default() },
        vec![],
    ));

    let mut s = scene("ground_contact", "A free cube dropped 1 mm onto the ground.", true);
    s.ground = Some(ground());
    s.bodies.push(cube("c", [0.0, 0.0, 0.101], [0.0; 3]));
    s.probes.push(disp("cz", "c", 0, z));
    s.probes.push(vel("cvz", "c", 0, z));
    s.sim.duration = 0.02;
    out.push(MicroCase::world("ground_contact", "a chunk lands on the ground and comes to rest", s, expect(cov::GROUND_CONTACT | cov::CONTACT_PIPELINE), vec![]));

    let mut s = scene("hull_hull", "Two hexagonal prisms (hulls) 1 mm apart closing at 1 m/s.", false);
    let flat = 0.1 * (std::f64::consts::PI / 6.0).cos();
    s.bodies.push(hex_prism("a", 0.0, 0.5));
    s.bodies.push(hex_prism("b", 2.0 * flat + 0.001, -0.5));
    s.probes.push(vel("a_vx", "a", 0, x));
    s.probes.push(vel("b_vx", "b", 0, x));
    s.sim.duration = 0.006;
    out.push(MicroCase::world(
        "hull_hull",
        "convex hull chunks collide face on face",
        s,
        Expect { cover: cov::PAIR_CONTACT, velocities: true, ..Default::default() },
        vec![],
    ));

    // ---- fracture
    let m = Material::analytic_test();
    let f_t = m.tensile_strength * 4.0 * 0.05 * 0.05;
    let mut s = bond_pull("crack_break_split", m, 0.05, TimeFunction::Ramp { t0: 0.0, t1: 0.004, value: 1.25 * f_t }, 0.006);
    s.sim.frame_dt = 0.001;
    s.probes.push(disp("pulled_x", "pair", 1, x));
    out.push(MicroCase::world(
        "crack_break_split",
        "a bond pulled past its strength cracks, softens, breaks, and the pair splits",
        s,
        Expect {
            cover: cov::CRACKED | cov::DISCONNECTED | cov::HOST_SPLIT | cov::LOAD_TERMS | cov::PROBES,
            min_events: vec![(EvKind::Cracked, 1), (EvKind::Broken, 1), (EvKind::Split, 1)],
            energy: true,
            ..Default::default()
        },
        vec![Knobs::batch(1), Knobs::sync(), Knobs::wide(2)],
    ));

    let mut s = scene("short_cascade", "A 4-chunk cantilever whose tip load passes its root capacity.", false);
    s.bodies.push(bar("arm", [4, 1, 1], &[0]));
    s.loads.push(point_force("arm", 3, [0.0, 0.0, -1.0], TimeFunction::Ramp { t0: 0.0, t1: 0.01, value: 12000.0 }));
    s.probes.push(disp("tip_z", "arm", 3, z));
    s.probes.push(probe(
        "root_moment",
        ProbeKind::SectionForce { body: "arm".into(), point: [0.2, 0.0, 0.0], normal: x, region: None, component: SectionComponent::Moment(y) },
    ));
    s.sim.duration = 0.012;
    out.push(MicroCase::world(
        "short_cascade",
        "a cantilever overloaded at the tip: the root cracks and breaks, the arm splits off",
        s,
        Expect { cover: cov::CRACKED | cov::DISCONNECTED | cov::HOST_SPLIT, min_events: vec![(EvKind::Broken, 1), (EvKind::Split, 1)], energy: true, ..Default::default() },
        vec![Knobs::wide(2), Knobs::batch(1)],
    ));

    let s = glass_pair("fatigue_exhaustion", 0.9, 0.1, |f| TimeFunction::Ramp { t0: 0.0, t1: 0.005, value: f }, 0.15);
    out.push(MicroCase::world(
        "fatigue_exhaustion",
        "a glass bond held at 90% of its strength creaks, then its static fatigue life runs out at ~0.1 s and it breaks",
        s,
        Expect { cover: cov::CREAKED | cov::CRACKED | cov::DISCONNECTED | cov::HOST_SPLIT, min_events: vec![(EvKind::Creaked, 1), (EvKind::Broken, 1)], ..Default::default() },
        vec![Knobs::sync()],
    ));

    let mut s = scene("resumed_halt", "A 2x2 ring of cubes: one bond is pulled apart, the ring holds.", false);
    s.bodies.push(bar("ring", [2, 2, 1], &[0]));
    s.loads.push(point_force("ring", 1, x, TimeFunction::Table { points: vec![[0.0, 0.0], [0.003, 3.3e5], [0.0035, 0.0]] }));
    s.probes.push(disp("pulled_x", "ring", 1, x));
    s.sim.duration = 0.006;
    out.push(MicroCase::world(
        "resumed_halt",
        "a bond breaks but its cluster stays one piece: the island's own connectivity check on the GPU finds it whole, no split",
        s,
        Expect { cover: cov::RESUMED_ON_DEVICE | cov::DISCONNECTED | cov::CRACKED, min_events: vec![(EvKind::Broken, 1)], ..Default::default() },
        vec![Knobs::batch(1), Knobs::wide(2)],
    ));
    let mut s = scene("resumed_halt_host", "A 2x2 ring of cubes: one bond is pulled apart, the ring holds.", false);
    s.bodies.push(bar("ring", [2, 2, 1], &[0]));
    s.loads.push(point_force("ring", 1, x, TimeFunction::Table { points: vec![[0.0, 0.0], [0.003, 3.3e5], [0.0035, 0.0]] }));
    s.probes.push(disp("pulled_x", "ring", 1, x));
    s.sim.duration = 0.006;
    out.push(
        MicroCase::world(
            "resumed_halt_host",
            "as resumed_halt with the device's connectivity check off: the island halts, the host finds it whole, the GPU resumes without a split",
            s,
            Expect { cover: cov::RESUMED_HALT | cov::DISCONNECTED | cov::CRACKED, min_events: vec![(EvKind::Broken, 1)], ..Default::default() },
            vec![Knobs::wide(2)],
        )
        .with_base(Knobs::host_connect()),
    );

    let mut s = scene("probe_resolution", "A short cantilever's prestressed sag (1e-7 m), probed at its tip.", true);
    let mut arm = bar("arm", [3, 1, 1], &[0]);
    arm.position = [0.7, 0.0, 0.3];
    s.bodies.push(arm);
    s.probes.push(disp("tip_z", "arm", 2, z));
    s.probes.push(disp("mid_z", "arm", 1, z));
    s.probes.push(disp("tip_x", "arm", 2, x));
    s.sim.duration = 0.004;
    out.push(MicroCase::world(
        "probe_resolution",
        "displacement probes resolve 1e-7 m far below the f32 ulp of the chunk's position (3e-8..1e-7 m)",
        s,
        expect(cov::PROBES),
        vec![],
    ));

    // ---- rigid motion
    let mut s = scene("rotation_precision", "A free bonded pair spinning at 11 rad/s for 26 ms.", false);
    let mut pair = bar("pair", [2, 1, 1], &[]);
    pair.position = [-0.2, 0.0, 0.0];
    pair.angular_velocity = [0.0, 0.0, 11.0];
    s.bodies.push(pair);
    s.probes.push(disp("tip_x", "pair", 1, x));
    s.probes.push(disp("tip_y", "pair", 1, y));
    s.probes.push(vel("tip_vy", "pair", 1, y));
    s.sim.duration = 0.026;
    out.push(MicroCase::world(
        "rotation_precision",
        "a spinning fragment keeps its orientation over thousands of substeps (compensated quaternion)",
        s,
        Expect { cover: cov::PROBES, velocities: true, energy: true, ..Default::default() },
        vec![Knobs::wide(2)],
    ));

    // ---- solve modes
    let mut s = scene("adaptive_settle_wake", "A settled cantilever woken by a load at 1 ms, settling again; an idle one beside it.", true);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.bodies.push(bar("arm", [3, 1, 1], &[0]));
    s.bodies.push({
        let mut b = bar("idle", [2, 1, 1], &[0]);
        b.position = [0.0, 2.0, 0.0];
        b
    });
    s.loads.push(point_force("arm", 2, [0.0, 0.0, -1.0], TimeFunction::Ramp { t0: 0.001, t1: 0.011, value: 2000.0 }));
    s.probes.push(disp("tip_z", "arm", 2, z));
    s.probes.push(probe("root_reaction", ProbeKind::Reaction { body: "arm".into(), chunks: ChunkSelector::Indices(vec![0]), axis: z }));
    s.sim.duration = 0.06;
    out.push(
        MicroCase::world(
            "adaptive_settle_wake",
            "a settled cluster wakes on the GPU when its load changes, then settles again (static solve); an unloaded one stays quiet",
            s,
            expect(cov::SETTLED | cov::WOKE | cov::SETTLE | cov::STATICS | cov::QUIET | cov::SETTLED_FATIGUE | cov::LOAD_TERMS),
            vec![Knobs::sync(), Knobs::batch(1)],
        )
        .with_tweak(Tweak { id: "active_10ms", apply: active_10ms }),
    );

    let mut s = glass_pair("sleeping_fatigue", 0.95, 0.05, |f| TimeFunction::Constant { value: f }, 0.06);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.frame_dt = 0.002;
    out.push(
        MicroCase::world(
            "sleeping_fatigue",
            "a settled glass bond under sustained load fatigues while asleep (GPU settled fatigue), creaks and wakes",
            s,
            Expect { cover: cov::SETTLED | cov::SETTLED_FATIGUE | cov::CREAKED, min_events: vec![(EvKind::Creaked, 1)], ..Default::default() },
            vec![],
        )
        .with_tweak(Tweak { id: "active_10ms", apply: active_10ms }),
    );

    let mut s = scene("quasi_static_cascade", "A quasi-static cantilever under a step load past its root capacity.", true);
    s.sim.solve_mode = SolveMode::QuasiStatic;
    s.bodies.push(bar("arm", [3, 1, 1], &[0]));
    s.loads.push(point_force("arm", 2, [0.0, 0.0, -1.0], TimeFunction::Table { points: vec![[0.0, 0.0], [0.01, 0.0], [0.0101, 20000.0]] }));
    s.probes.push(disp("tip_z", "arm", 2, z));
    s.sim.duration = 0.05;
    out.push(MicroCase::world(
        "quasi_static_cascade",
        "a step load cracks the root at quasi-static equilibrium: damage is committed there and the same-step cascade re-solves (past the peak no equilibrium exists: both keep the last one)",
        s,
        Expect { cover: cov::STATICS | cov::QS_CASCADE | cov::SETTLED | cov::LOAD_TERMS, min_events: vec![(EvKind::Cracked, 1)], ..Default::default() },
        vec![],
    ));

    let bridge = |name: &str, desc: &str| {
        let mut s = scene(name, desc, true);
        s.bodies.push(bar("bridge", [3, 1, 1], &[0, 2]));
        s.probes.push(disp("mid_z", "bridge", 1, z));
        s.probes.push(probe("left_reaction", ProbeKind::Reaction { body: "bridge".into(), chunks: ChunkSelector::Indices(vec![0]), axis: z }));
        s.sim.duration = 0.008;
        s
    };
    let mut s = bridge("remove_chunks", "A 3-chunk bridge whose right support chunk is removed at 1 ms.");
    s.events.push(EventDesc::RemoveChunks { time: 0.001, duration: 0.0, body: "bridge".into(), chunks: ChunkSelector::Indices(vec![2]) });
    out.push(MicroCase::world(
        "remove_chunks",
        "a removed chunk's bonds are replaced by loads (alternate path) and the GPU rebuilds at the event",
        s,
        expect(cov::REBUILD_EVENT | cov::LOAD_TERMS),
        vec![],
    ));
    let mut s = bridge("remove_supports", "A 3-chunk bridge losing its right support at 1 ms.");
    s.events.push(EventDesc::RemoveSupports { time: 0.001, duration: 0.0, body: "bridge".into(), chunks: ChunkSelector::Indices(vec![2]) });
    out.push(MicroCase::world("remove_supports", "a removed support turns the bridge into a cantilever at the event", s, expect(cov::REBUILD_EVENT), vec![]));

    let mut s = scene("blast_1ms", "A small wall struck by a blast arriving at 1 ms.", true);
    let mut wall = grid(Vec3::new(0.0, -0.1, 0.0), [3, 1, 3], Vec3::splat(0.2), "concrete");
    for c in wall.iter_mut().filter(|c| c.center[2] < 0.15) {
        c.support = Support::Fixed;
    }
    let bonds = auto_bonds(&wall, |_, _| "concrete".into());
    s.bodies.push(body("wall", wall, bonds));
    s.loads.push(LoadDesc::Blast { position: [0.3, -1.0, 0.3], tnt_mass: 0.02, time: 0.001 });
    s.probes.push(disp("top_y", "wall", 7, y));
    s.sim.duration = 0.006;
    out.push(MicroCase::world("blast_1ms", "a blast's pressure pulse reaches the wall's exposed faces when it arrives", s, expect(cov::BLAST | cov::LOAD_TERMS), vec![]));

    let mut s = scene("probes_all_kinds", "A cantilever struck sideways, observed by every probe kind.", true);
    s.bodies.push(bar("arm", [3, 1, 1], &[0]));
    s.impactors.push(impactor("ball", ImpactorShape::Sphere { radius: 0.05 }, 10.0, [0.5, -0.151, 0.0], [0.0, 3.0, 0.0]));
    s.probes.push(disp("d", "arm", 2, y));
    s.probes.push(vel("v", "arm", 2, y));
    for (name, component) in [("n", SectionComponent::Normal), ("fy", SectionComponent::Force(y)), ("mz", SectionComponent::Moment(z))] {
        s.probes.push(probe(name, ProbeKind::SectionForce { body: "arm".into(), point: [0.2, 0.0, 0.0], normal: x, region: None, component }));
    }
    s.probes.push(probe("reaction", ProbeKind::Reaction { body: "arm".into(), chunks: ChunkSelector::Indices(vec![0]), axis: y }));
    s.probes.push(probe("ball_v", ProbeKind::ImpactorVelocity { impactor: "ball".into(), axis: y }));
    s.probes.push(probe("ball_x", ProbeKind::ImpactorPosition { impactor: "ball".into(), axis: y }));
    s.sim.duration = 0.006;
    out.push(MicroCase::world("probes_all_kinds", "every probe kind records the same history as the reference's", s, expect(cov::PROBES | cov::IMPACTOR_CONTACT), vec![]));

    let mut s = scene("stop_now_replan", "A 30 m/s sphere launches a free cube out of its contact budget.", false);
    s.bodies.push(cube("c", [0.0, 0.0, 0.0], [0.0; 3]));
    s.impactors.push(impactor("ball", ImpactorShape::Sphere { radius: 0.05 }, 50.0, [-0.152, 0.0, 0.0], [30.0, 0.0, 0.0]));
    s.probes.push(vel("c_vx", "c", 0, x));
    s.probes.push(probe("ball_v", ProbeKind::ImpactorVelocity { impactor: "ball".into(), axis: x }));
    s.sim.duration = 0.004;
    out.push(MicroCase::world(
        "stop_now_replan",
        "a chunk leaving its contact budget stops the batch; the host re-plans mid-step",
        s,
        Expect { cover: cov::STOP_NOW | cov::REPLAN | cov::IMPACTOR_CONTACT, velocities: true, ..Default::default() },
        vec![Knobs::batch(1), Knobs::plan(1.0)],
    ));

    let mut s = scene("settled_wide", "Adaptive: a free pair, wide while active, settles, then a load wakes it.", false);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.bodies.push(bar("pair", [2, 1, 1], &[]));
    s.loads.push(point_force("pair", 1, z, TimeFunction::Ramp { t0: 0.02, t1: 0.03, value: 500.0 }));
    s.probes.push(disp("tip_z", "pair", 1, z));
    s.sim.duration = 0.04;
    out.push(
        MicroCase::world(
            "settled_wide",
            "an active wide island settles (static solve), runs settled, and wakes when loaded",
            s,
            expect(cov::WIDE | cov::SETTLE | cov::STATICS | cov::SETTLED | cov::WOKE | cov::LOAD_TERMS),
            vec![Knobs::default()],
        )
        .with_tweak(Tweak { id: "active_6ms", apply: active_6ms })
        .with_base(Knobs::wide(2)),
    );

    // ---- engine-coupled
    for (name, adaptive) in [("coupled_impulse", false), ("coupled_sleeping", true)] {
        let script = CoupledScript { scene: coupled_scene(name), gpu_adaptive: adaptive, frames: 6, inputs_version: 1, contacts: coupled_contacts, motion: coupled_motion };
        let cover = cov::PULSE | cov::LOAD_TERMS | if adaptive { cov::COUPLED_MAY_WAKE | cov::SETTLED } else { 0 };
        out.push(MicroCase {
            name: if adaptive { "coupled_sleeping" } else { "coupled_impulse" },
            semantic: if adaptive {
                "engine impulses on a sleeping (adaptive) wall: the struck cluster wakes within the frame"
            } else {
                "engine impulses become impact pulses and sustained loads; the engine owns the free pair's motion"
            },
            script: Script::Coupled(script),
            tweak: None,
            base: Knobs::default(),
            expect: expect(cover),
            knob_runs: if adaptive { vec![] } else { vec![Knobs::batch(1)] },
        });
    }
    out
}

// ------------------------------------------------------------------ fuzz

/// A seeded random scene (`stress_ref::rng::uniform(seed, &[field])`): a grid of
/// 2x2x1..6x4x2 cubes of a catalogue material, a support pattern, an optional impactor
/// (shape, speed, angle), a solve mode.
pub fn fuzz_scene(seed: u64) -> Scene {
    let u = |k: u64| stress_ref::rng::uniform(seed, &[k]);
    let pick = |k: u64, n: usize| ((u(k) * n as f64) as usize).min(n - 1);
    let n = [2 + pick(0, 5), 2 + pick(1, 3), 1 + pick(2, 2)];
    let materials = ["concrete", "brick", "glass", "timber", "stone", "steel"];
    let mat = materials[pick(3, materials.len())];
    let mut s = new_scene(&format!("fuzz-{seed:016x}"), "Seeded fuzz scene.");
    let material = match mat {
        "concrete" => Material::concrete(),
        "brick" => Material::brick(),
        "glass" => Material::glass(),
        "timber" => Material::timber(),
        "stone" => Material::stone(),
        _ => Material::steel(),
    };
    s.materials.insert(mat.into(), material);
    let mut chunks = grid(Vec3::new(0.0, 0.0, 0.0), n, Vec3::splat(0.2), mat);
    match pick(4, 3) {
        0 => {
            // Free, resting on the ground.
            s.ground = Some(GroundDesc { height: -0.001, friction: 0.5, material: mat.into() });
        }
        1 => chunks.iter_mut().filter(|c| c.center[0] < 0.2).for_each(|c| c.support = Support::Fixed),
        _ => chunks.iter_mut().filter(|c| c.center[2] < 0.2).for_each(|c| c.support = Support::Fixed),
    }
    let bonds = auto_bonds(&chunks, |_, _| mat.into());
    s.bodies.push(body("structure", chunks, bonds));
    s.sim.frame_dt = 0.004;
    s.sim.duration = 0.024;
    s.sim.solve_mode = match pick(5, 20) {
        0..=11 => SolveMode::Explicit,
        12..=16 => SolveMode::Adaptive,
        _ => SolveMode::QuasiStatic,
    };
    if u(6) < 0.7 {
        let shape = if u(7) < 0.5 { ImpactorShape::Sphere { radius: 0.1 + 0.1 * u(8) } } else { ImpactorShape::Box { half_extents: [0.1, 0.1 + 0.1 * u(8), 0.1 + 0.1 * u(8)] } };
        let speed = 3.0 + 22.0 * u(9);
        let mass = 20.0 + 200.0 * u(10);
        let mut sc = gate::with_impactor(s, shape, mass, speed);
        // Aim off the face normal by up to 30 degrees, from closer in.
        let a = (u(11) - 0.5) * std::f64::consts::PI / 3.0;
        let imp = &mut sc.impactors[0];
        imp.position[0] += 0.15;
        imp.velocity = [speed * a.cos(), 0.0, speed * a.sin()];
        sc
    } else {
        gate::with_chunk_probes(s)
    }
}

// ------------------------------------------------------------------ runs

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Ev {
    pub kind: EvKind,
    pub structure: usize,
    /// The bond (a split: its number of children).
    pub bond: usize,
    pub time: f64,
}

/// What a run produced beside its observation: events, energy terms, substep.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Semantics {
    pub events: Vec<Ev>,
    pub e0: f64,
    pub e1: f64,
    pub dissipated: f64,
    pub external_work: f64,
    pub contact_stored: f64,
    pub dt: f64,
}

fn events_of(list: &[SolverEvent]) -> Vec<Ev> {
    list.iter()
        .filter_map(|e| match e {
            SolverEvent::Cracked { time, structure, bond, .. } => Some(Ev { kind: EvKind::Cracked, structure: *structure, bond: *bond, time: *time }),
            SolverEvent::Creaked { time, structure, bond, .. } => Some(Ev { kind: EvKind::Creaked, structure: *structure, bond: *bond, time: *time }),
            SolverEvent::Broken { time, structure, bond, .. } => Some(Ev { kind: EvKind::Broken, structure: *structure, bond: *bond, time: *time }),
            SolverEvent::Split { time, children, .. } => Some(Ev { kind: EvKind::Split, structure: 0, bond: children.len(), time: *time }),
            SolverEvent::Refined { .. } => None,
        })
        .collect()
}

/// The reference runs of a case: world cases keep the gate's reference set (reference,
/// half step, perturbed runs); coupled cases their per-bond rows (reference, impulses
/// scaled by 1 +- 1e-4). `semantics` follows the same order.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MicroReference {
    pub set: Option<ReferenceSet>,
    pub rows: Vec<Vec<Vec<f64>>>,
    pub fractures: Vec<usize>,
    pub semantics: Vec<Semantics>,
}

fn map_sem(s: &mut Semantics, f: fn(f64) -> f64) {
    for e in &mut s.events {
        e.time = f(e.time);
    }
    for x in [&mut s.e0, &mut s.e1, &mut s.dissipated, &mut s.external_work, &mut s.contact_stored, &mut s.dt] {
        *x = f(*x);
    }
}

fn map_reference(mut r: MicroReference, encode: bool) -> MicroReference {
    let f: fn(f64) -> f64 = if encode { gate::enc } else { gate::dec };
    r.set = r.set.map(|s| if encode { gate::encode(s) } else { gate::decode(s) });
    for rows in &mut r.rows {
        for row in rows.iter_mut() {
            for x in row.iter_mut() {
                *x = f(*x);
            }
        }
    }
    for s in &mut r.semantics {
        map_sem(s, f);
    }
    r
}

fn case_key(case: &MicroCase) -> serde_json::Value {
    let base = match &case.script {
        Script::World(s) => serde_json::json!({ "scene": s }),
        Script::Coupled(c) => serde_json::json!({ "scene": c.scene, "frames": c.frames, "inputs": c.inputs_version }),
    };
    serde_json::json!({ "case": base, "tweak": case.tweak.map(|t| t.id), "perturbations": gate::perturbations(), "version": MICRO_VERSION })
}

/// The scene with its inputs scaled by `1 + eps`: `gate::perturbed_by` (gravity, impactor
/// and body velocities) and the scripted loads' magnitudes too. A scene driven by loads
/// alone (no gravity, nothing moving) is otherwise not perturbed at all, and its
/// reference ensemble would claim no sensitivity to the f32-sized input changes the GPU
/// makes (a fatigue life is ~n = 16 times as sensitive to its load).
pub fn perturbed(scene: &Scene, eps: f64) -> Scene {
    let mut s = gate::perturbed_by(scene, eps);
    let k = 1.0 + eps;
    let scale = |f: &mut TimeFunction| match f {
        TimeFunction::Constant { value } | TimeFunction::Ramp { value, .. } => *value *= k,
        TimeFunction::HalfSine { peak, .. } | TimeFunction::Friedlander { peak, .. } => *peak *= k,
        TimeFunction::Table { points } => points.iter_mut().for_each(|p| p[1] *= k),
    };
    for l in &mut s.loads {
        match l {
            LoadDesc::PointForce { magnitude, .. } => scale(magnitude),
            LoadDesc::Pressure { pressure, .. } => scale(pressure),
            LoadDesc::Blast { tnt_mass, .. } => *tnt_mass *= k,
        }
    }
    s
}

/// The reference World run of a scene with the case's tweak.
fn reference_run(scene: &Scene, tweak: Option<Tweak>) -> (Observation, Semantics) {
    let mut w = World::new(scene);
    if let Some(t) = tweak {
        (t.apply)(&mut w.solver);
    }
    let dt = w.substep_dt();
    let e0 = w.mechanical_energy();
    let obs = w.run();
    let sem = Semantics {
        events: events_of(&w.solver.events),
        e0,
        e1: w.mechanical_energy(),
        dissipated: w.dissipated_energy(),
        external_work: w.solver.energy.external_work,
        contact_stored: w.contact.stored,
        dt,
    };
    (obs, sem)
}

/// Bond rows (damage, utilization, broken) of every structure, fractures and events.
fn drive(c: &CoupledScript, solver: &mut dyn StressSolverApi, scale: f64) -> (Vec<Vec<f64>>, usize, Vec<Ev>) {
    let mut fractures = 0;
    let mut events = Vec::new();
    for f in 0..c.frames {
        let input = FrameInput { dt: 1.0 / 60.0, motion: (c.motion)(f, &solver.clusters()), contacts: (c.contacts)(f, &c.scene, scale) };
        let out = solver.step(&input);
        fractures += out.fractures.len();
        events.extend(events_of(&out.events));
    }
    let mut rows = Vec::new();
    for s in 0..c.scene.bodies.len() {
        rows.extend(solver.bond_reports(s).iter().map(|r| vec![r.damage, r.utilization, if r.broken { 1.0 } else { 0.0 }]));
    }
    (rows, fractures, events)
}

/// The substep of a coupled frame (coupled.rs `step_gpu`: the stable substep fitted to
/// the frame).
fn coupled_dt(c: &CoupledScript) -> f64 {
    let frame = 1.0 / 60.0;
    let n = (frame / ReferenceSolver::new(&c.scene).stable_dt().min(frame)).ceil().max(1.0);
    frame / n
}

fn compute_reference(case: &MicroCase) -> MicroReference {
    match &case.script {
        Script::World(scene) => {
            let tweak = case.tweak;
            let eps = gate::perturbations();
            let (r, h, ps) = std::thread::scope(|sc| {
                let r = sc.spawn(|| {
                    let t = Instant::now();
                    let o = reference_run(scene, tweak);
                    (o, t.elapsed().as_secs_f64())
                });
                let h = sc.spawn(|| reference_run(&gate::refined(scene), tweak));
                let ps: Vec<_> = eps.iter().map(|&e| sc.spawn(move || reference_run(&perturbed(scene, e), tweak))).collect();
                (r.join().unwrap(), h.join().unwrap(), ps.into_iter().map(|p| p.join().unwrap()).collect::<Vec<_>>())
            });
            let mut semantics = vec![r.0 .1, h.1];
            semantics.extend(ps.iter().map(|p| p.1.clone()));
            let set = ReferenceSet { reference: r.0 .0, half: h.0, spreads: ps.into_iter().map(|p| p.0).collect(), reference_seconds: r.1 };
            MicroReference { set: Some(set), rows: Vec::new(), fractures: Vec::new(), semantics }
        }
        Script::Coupled(c) => {
            let runs: Vec<_> = std::thread::scope(|sc| {
                let hs: Vec<_> = [1.0, 1.0 + 1e-4, 1.0 - 1e-4].iter().map(|&k| sc.spawn(move || drive(c, &mut EngineCoupledSolver::new(&c.scene), k))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut out = MicroReference { set: None, rows: Vec::new(), fractures: Vec::new(), semantics: Vec::new() };
            let dt = coupled_dt(c);
            for (rows, fractures, events) in runs {
                out.rows.push(rows);
                out.fractures.push(fractures);
                out.semantics.push(Semantics { events, dt, ..Default::default() });
            }
            out
        }
    }
}

/// The case's reference runs from the cache, or computed (`compute`) and stored.
pub fn micro_reference(case: &MicroCase, compute: bool) -> Option<MicroReference> {
    let label = format!("micro-{}", case.name);
    let key = case_key(case);
    if let Some(r) = gate::lookup::<_, MicroReference>(&label, &key) {
        return Some(map_reference(r, false));
    }
    if !compute {
        return None;
    }
    let r = compute_reference(case);
    gate::store(&label, &key, &map_reference(r.clone(), true));
    Some(r)
}

/// One GPU run's outcome.
#[derive(Clone, Debug)]
pub struct GpuRun {
    pub obs: Option<Observation>,
    pub rows: Vec<Vec<f64>>,
    pub fractures: usize,
    pub sem: Semantics,
    pub coverage: u64,
    /// Hash of the final chunk state, cluster motion and bond history (after sync).
    pub state: u64,
    pub seconds: f64,
}

fn hash_str(s: &str, h: u64) -> u64 {
    gate::fnv(s.as_bytes(), h)
}

/// Every final state of the mirror (synced), and the impactors, as a hash.
fn state_hash(m: &ReferenceSolver, impactors: &[stress_ref::world::Impactor]) -> u64 {
    let mut h = 0xcbf29ce484222325;
    for chunks in &m.chunks {
        for c in chunks {
            h = hash_str(&format!("{:?}{:?}{:?}{:?}{}", c.u, c.th, c.v, c.w, c.active), h);
        }
    }
    for cl in &m.clusters {
        h = hash_str(&format!("{:?}{:?}{:?}{:?}{:?}", cl.chunks, cl.pose, cl.velocity, cl.angular_velocity, cl.activity), h);
    }
    for bonds in &m.bonds {
        for b in bonds {
            h = hash_str(&format!("{:?}{:?}", b.joint, b.force), h);
        }
    }
    for imp in impactors {
        h = hash_str(&format!("{:?}{:?}{:?}{}", imp.pose, imp.velocity, imp.angular_velocity, imp.crush_used), h);
    }
    h
}

/// A world scene on the GPU (the case's tweak applied to the mirror, then rebuilt).
pub fn run_world(gpu: &Gpu, scene: &Scene, tweak: Option<Tweak>) -> Result<GpuRun, String> {
    let t = Instant::now();
    let mut w = GpuWorld::new(gpu, scene)?;
    if let Some(tw) = tweak {
        (tw.apply)(&mut w.solver.mirror);
        w.solver.rebuild(gpu, 0.0)?;
    }
    let dt = w.with_shell(|s| s.substep_dt());
    let e0 = w.with_shell(|s| s.mechanical_energy());
    let obs = w.run(gpu)?;
    let (e1, dissipated, external_work, contact_stored) = w.with_shell(|s| (s.mechanical_energy(), s.dissipated_energy(), s.solver.energy.external_work, s.contact.stored));
    let sem = Semantics { events: events_of(&w.solver.mirror.events), e0, e1, dissipated, external_work, contact_stored, dt };
    Ok(GpuRun {
        obs: Some(obs),
        rows: Vec::new(),
        fractures: 0,
        sem,
        coverage: w.solver.coverage,
        state: state_hash(&w.solver.mirror, &w.solver.impactors),
        seconds: t.elapsed().as_secs_f64(),
    })
}

fn run_coupled(gpu: &Gpu, c: &CoupledScript) -> Result<GpuRun, String> {
    let t = Instant::now();
    let mut scene = c.scene.clone();
    if c.gpu_adaptive {
        scene.sim.solve_mode = SolveMode::Adaptive;
    }
    let mut api = GpuStressSolverApi::with_gpu(gpu.clone(), &scene)?;
    let (rows, fractures, events) = drive(c, &mut api, 1.0);
    let s = &api.coupled.solver;
    Ok(GpuRun {
        obs: None,
        rows,
        fractures,
        sem: Semantics { events, dt: coupled_dt(c), ..Default::default() },
        coverage: s.coverage,
        state: state_hash(&s.mirror, &s.impactors),
        seconds: t.elapsed().as_secs_f64(),
    })
}

/// The case on the GPU under `knobs`.
pub fn run_gpu(gpu: &Gpu, case: &MicroCase, knobs: &Knobs) -> Result<GpuRun, String> {
    with_knobs(knobs, || match &case.script {
        Script::World(scene) => run_world(gpu, scene, case.tweak),
        Script::Coupled(c) => run_coupled(gpu, c),
    })
}

// ------------------------------------------------------------------ checks

type EventKey = (EvKind, usize, usize);

fn keyed(evs: &[Ev]) -> BTreeMap<EventKey, Vec<f64>> {
    let mut m: BTreeMap<EventKey, Vec<f64>> = BTreeMap::new();
    for e in evs {
        m.entry((e.kind, e.structure, e.bond)).or_default().push(e.time);
    }
    for v in m.values_mut() {
        v.sort_by(f64::total_cmp);
    }
    m
}

fn same_set(a: &BTreeMap<EventKey, Vec<f64>>, b: &BTreeMap<EventKey, Vec<f64>>) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| w.len() == v.len()))
}

fn describe(m: &BTreeMap<EventKey, Vec<f64>>) -> String {
    let parts: Vec<String> = m.iter().map(|((k, s, b), v)| if *k == EvKind::Split { format!("{}x Split({b})", v.len()) } else { format!("{}x {k:?} s{s} bond {b}", v.len()) }).collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(", ")
    }
}

/// Events against the reference: the same (kind, bond) multiset, times within one
/// substep or twice the reference ensemble's own scatter. Where the reference's own
/// ensemble does not agree on the multiset, the GPU's must be one of the ensemble's.
fn check_events(refs: &[Semantics], ours: &Semantics) -> Vec<String> {
    let r = keyed(&refs[0].events);
    let g = keyed(&ours.events);
    let ensemble: Vec<_> = refs.iter().map(|s| keyed(&s.events)).collect();
    if !ensemble.iter().all(|e| same_set(e, &r)) {
        if ensemble.iter().any(|e| same_set(e, &g)) {
            return Vec::new();
        }
        return vec![format!("events: GPU {} is none of the reference ensemble's sets (reference {})", describe(&g), describe(&r))];
    }
    if !same_set(&r, &g) {
        return vec![format!("events: GPU {} vs reference {}", describe(&g), describe(&r))];
    }
    let dt = refs[0].dt;
    let mut fails = Vec::new();
    for (key, times) in &r {
        for (i, &tr) in times.iter().enumerate() {
            let tg = g[key][i];
            let dev = ensemble.iter().map(|e| (e[key][i] - tr).abs()).fold(0.0, f64::max);
            let allowed = (1.0001 * dt).max(2.0 * dev);
            if (tg - tr).abs() > allowed {
                let what = if key.0 == EvKind::Split { format!("Split({})", key.2) } else { format!("{:?} bond {}", key.0, key.2) };
                fails.push(format!(
                    "events: {what} at t={tg:.6} (substep {:.0}) vs reference {tr:.6} (substep {:.0}), allowed +-{allowed:.1e}",
                    tg / dt,
                    tr / dt
                ));
            }
        }
    }
    fails
}

/// Each body's fragment ids relabelled by first appearance (-1 stays).
fn partition(o: &Observation) -> BTreeMap<String, Vec<i64>> {
    o.bodies
        .iter()
        .map(|(name, chunks)| {
            let mut map: BTreeMap<i64, i64> = BTreeMap::new();
            let ids = chunks
                .iter()
                .map(|c| {
                    if c.fragment < 0 {
                        return -1;
                    }
                    let n = map.len() as i64;
                    *map.entry(c.fragment).or_insert(n)
                })
                .collect();
            (name.clone(), ids)
        })
        .collect()
}

fn ensemble_obs(set: &ReferenceSet) -> Vec<&Observation> {
    std::iter::once(&set.reference).chain(std::iter::once(&set.half)).chain(set.spreads.iter()).collect()
}

fn check_fragments(set: &ReferenceSet, ours: &Observation) -> Vec<String> {
    let g = partition(ours);
    if ensemble_obs(set).iter().any(|o| partition(o) == g) {
        return Vec::new();
    }
    vec![format!("fragments: GPU partition {:?} vs reference {:?}", g, partition(&set.reference))]
}

fn velocities(o: &Observation) -> Vec<[f64; 3]> {
    o.bodies.values().flat_map(|cs| cs.iter().map(|c| c.velocity)).collect()
}

/// Chunk velocities at the end (momentum exchange) within twice the reference
/// ensemble's scatter, or 1e-3 of the largest speed.
fn check_velocities(set: &ReferenceSet, ours: &Observation) -> Vec<String> {
    let r = velocities(&set.reference);
    let g = velocities(ours);
    let vmax = r.iter().flat_map(|v| v.iter()).fold(0.0f64, |m, x| m.max(x.abs()));
    let mut worst = (0.0f64, 0usize, 0.0f64);
    for (i, (a, b)) in r.iter().zip(&g).enumerate() {
        for k in 0..3 {
            let dev = ensemble_obs(set).iter().map(|o| (velocities(o)[i][k] - a[k]).abs()).fold(0.0, f64::max);
            let allowed = (2.0 * dev).max(1e-3 * vmax).max(1e-9);
            let excess = (b[k] - a[k]).abs() / allowed;
            if excess > worst.0 {
                worst = (excess, i, (b[k] - a[k]).abs());
            }
        }
    }
    if worst.0 > 1.0 {
        return vec![format!("momentum: chunk {} velocity off by {:.3e} m/s ({:.1}x the allowed; speeds up to {vmax:.3e})", worst.1, worst.2, worst.0)];
    }
    Vec::new()
}

/// Energy closure residual relative to the energy that moved.
fn closure(s: &Semantics) -> (f64, f64) {
    let scale = s.external_work.abs().max(s.dissipated).max((s.e1 - s.e0).abs()).max(1e-12);
    ((s.e0 + s.external_work - s.e1 - s.dissipated).abs() / scale, scale)
}

/// `E0 + W_ext = mechanical + dissipated` on the GPU, within twice the reference's own
/// residual or 1e-3 of the energy that moved (contacts still storing energy at the end
/// are skipped: the GPU does not report that term).
fn check_energy(refs: &[Semantics], ours: &Semantics) -> Vec<String> {
    let (r_ref, scale) = closure(&refs[0]);
    if refs[0].contact_stored > 1e-3 * scale {
        return Vec::new();
    }
    let worst_ref = refs.iter().map(|s| closure(s).0).fold(r_ref, f64::max);
    let (r_gpu, _) = closure(ours);
    let allowed = (2.0 * worst_ref).max(1e-3);
    if !(r_gpu <= allowed) {
        return vec![format!(
            "energy: closure residual {r_gpu:.2e} of {scale:.3e} J > {allowed:.1e} (reference {r_ref:.1e}): E0 {:.6e} + W {:.6e} vs mech {:.6e} + dissipated {:.6e}",
            ours.e0, ours.external_work, ours.e1, ours.dissipated
        )];
    }
    Vec::new()
}

fn count(evs: &[Ev], kind: EvKind) -> usize {
    evs.iter().filter(|e| e.kind == kind).count()
}

/// Absolute resolution (m, m/s, N) below which a probe the reference holds at exactly
/// zero counts as zero on the GPU too.
pub const ZERO_PROBE: f64 = 1e-12;

/// `ours` with every probe the reference holds at exactly zero (by symmetry: a sideways
/// displacement of a straight push, an anchored chunk's) taken as the reference's when
/// the GPU's stays within `ZERO_PROBE`: such a probe has no range, so the gate's relative
/// error (range floor 1e-30) would judge f32 noise of 1e-31..1e-14 as a 1e16 error.
/// Anything larger still fails.
fn without_zero_probes(reference: &Observation, ours: &Observation) -> Observation {
    let mut out = ours.clone();
    for (name, r) in &reference.probes {
        let Some(g) = out.probes.get_mut(name) else { continue };
        if r.v.iter().all(|x| *x == 0.0) && g.v.len() == r.v.len() && g.v.iter().all(|x| x.abs() <= ZERO_PROBE) {
            *g = r.clone();
        }
    }
    out
}

/// The accuracy gate, with the chaotic-case ensemble judgement (`stress-gpu-gate`).
pub fn accuracy(gpu: &Gpu, scene: &Scene, tweak: Option<Tweak>, knobs: &Knobs, perturb: fn(&Scene, f64) -> Scene, set: &ReferenceSet, ours: &Observation) -> Vec<String> {
    let ours = &without_zero_probes(&set.reference, ours);
    let fails = gate::gate(&set.reference, ours, &set.half, &set.spreads);
    if fails.is_empty() || !gate::is_chaotic(set) {
        return fails.into_iter().map(|f| format!("accuracy: {f}")).collect();
    }
    let mut runs = vec![ours.clone()];
    for e in gate::perturbations() {
        let s = perturb(scene, e);
        if let Ok(r) = with_knobs(knobs, || run_world(gpu, &s, tweak)) {
            runs.extend(r.obs);
        }
    }
    gate::ensemble_gate(set, &runs).into_iter().map(|f| format!("accuracy (chaotic ensemble): {f}")).collect()
}

/// Coupled: per bond damage and utilization within twice the reference's spread (at
/// least 1e-3), the same broken bonds and fractures unless the spread differs.
fn check_coupled(refs: &MicroReference, ours: &GpuRun) -> Vec<String> {
    let cmp = |a: &[Vec<f64>], b: &[Vec<f64>]| {
        let mut w = (0.0f64, 0.0f64, 0usize);
        for (x, y) in a.iter().zip(b) {
            w.0 = w.0.max((x[0] - y[0]).abs());
            w.1 = w.1.max((x[1] - y[1]).abs() / x[1].abs().max(1e-3));
            w.2 += (x[2] != y[2]) as usize;
        }
        w
    };
    let r = &refs.rows[0];
    let g = cmp(r, &ours.rows);
    let spread = refs.rows[1..].iter().map(|p| cmp(r, p)).fold((0.0f64, 0.0f64, 0usize), |a, b| (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2)));
    let mut fails = Vec::new();
    if g.0 > (2.0 * spread.0).max(1e-3) {
        fails.push(format!("bonds: damage differs by {:.1e} (allowed {:.1e})", g.0, (2.0 * spread.0).max(1e-3)));
    }
    if g.1 > (2.0 * spread.1).max(1e-3) {
        fails.push(format!("bonds: utilization differs by {:.1e} (allowed {:.1e})", g.1, (2.0 * spread.1).max(1e-3)));
    }
    if g.2 > 2 * spread.2 {
        fails.push(format!("bonds: {} broken differently (reference spread {})", g.2, spread.2));
    }
    if !refs.fractures.contains(&ours.fractures) {
        fails.push(format!("fractures: GPU {} vs reference {:?}", ours.fractures, refs.fractures));
    }
    fails
}

/// Every semantic check of a case's run against its reference. `coverage`: the bits of
/// all its runs (the expectation may be met by a knob run).
pub fn check_case(gpu: &Gpu, case: &MicroCase, refs: &MicroReference, run: &GpuRun, coverage: u64) -> Vec<String> {
    let mut fails = Vec::new();
    for &(kind, n) in &case.expect.min_events {
        if count(&refs.semantics[0].events, kind) < n {
            fails.push(format!("scene: the reference shows {} {kind:?} events (< {n}): the case lost its semantic", count(&refs.semantics[0].events, kind)));
        }
        if count(&run.sem.events, kind) < n {
            fails.push(format!("events: GPU has {} {kind:?} (< {n})", count(&run.sem.events, kind)));
        }
    }
    match (&refs.set, &run.obs) {
        (Some(set), Some(obs)) => {
            fails.extend(accuracy(gpu, case.scene(), case.tweak, &case.base, perturbed, set, obs));
            fails.extend(check_fragments(set, obs));
            if case.expect.velocities {
                fails.extend(check_velocities(set, obs));
            }
            fails.extend(check_events(&refs.semantics, &run.sem));
            if case.expect.energy {
                fails.extend(check_energy(&refs.semantics, &run.sem));
            }
        }
        _ => {
            fails.extend(check_coupled(refs, run));
            fails.extend(check_events(&refs.semantics, &run.sem));
        }
    }
    let missing = case.expect.cover & !coverage;
    if missing != 0 {
        fails.push(format!("coverage: {} did not run", cov::names(missing).join(", ")));
    }
    fails
}

/// Observation values the sync cadence may change: f32 per-sync partial sums folded into
/// f64 at another cadence.
const SYNC_SUMS: [&str; 3] = ["bond_dissipation", "damping_dissipation", "softening_overshoot"];
/// Relative bound on those sums under another sync cadence.
pub const SYNC_SUM_BOUND: f64 = 1e-4;
/// Bound on probes (of their range) for wide islands and plan horizons.
pub const BOUNDED_PROBES: f64 = 1e-5;

fn series_equal(a: &Observation, b: &Observation) -> Option<String> {
    for (name, s) in &a.probes {
        let Some(t) = b.probes.get(name) else { return Some(format!("probe {name} missing")) };
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        if bits(&s.v) != bits(&t.v) || bits(&s.t) != bits(&t.t) {
            let i = s.v.iter().zip(&t.v).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap_or(0);
            return Some(format!("probe {name} differs from sample {i} ({:e} vs {:e})", s.v.get(i).copied().unwrap_or(f64::NAN), t.v.get(i).copied().unwrap_or(f64::NAN)));
        }
    }
    None
}

/// A knob run against the base run (no reference): bitwise for batch boundaries and sync
/// cadence; bounded for wide islands and plan horizons (`BOUNDED_PROBES` of each probe's
/// range, equal counts and (kind, bond) events, times within one substep), or, for a
/// scene whose reference itself scatters beyond that bound, judged by the gate.
pub fn metamorphic(gpu: &Gpu, case: &MicroCase, refs: Option<&MicroReference>, base: &GpuRun, other: &GpuRun, knobs: &Knobs) -> Vec<String> {
    let mut fails = Vec::new();
    if knobs.bitwise() {
        if base.state != other.state {
            fails.push("state differs".into());
        }
        if let (Some(a), Some(b)) = (&base.obs, &other.obs) {
            fails.extend(series_equal(a, b));
            for (k, va) in &a.values {
                let vb = b.values.get(k).copied().unwrap_or(f64::NAN);
                let ok = if knobs.sync_each_step && SYNC_SUMS.contains(&k.as_str()) { (va - vb).abs() <= SYNC_SUM_BOUND * va.abs().max(1e-9) } else { va.to_bits() == vb.to_bits() };
                if !ok {
                    fails.push(format!("value {k}: {va:e} vs {vb:e}"));
                }
            }
        }
        let sorted = |e: &[Ev]| {
            let mut v: Vec<(EvKind, usize, usize, u64)> = e.iter().map(|e| (e.kind, e.structure, e.bond, e.time.to_bits())).collect();
            v.sort();
            v
        };
        if sorted(&base.sem.events) != sorted(&other.sem.events) {
            fails.push(format!("events differ: {} vs {}", describe(&keyed(&base.sem.events)), describe(&keyed(&other.sem.events))));
        }
        if base.rows != other.rows {
            fails.push("bond rows differ".into());
        }
        return fails;
    }
    let dt = base.sem.dt;
    if let (Some(a), Some(b)) = (&base.obs, &other.obs) {
        let (err, _) = gate::difference(a, b);
        if !(err <= BOUNDED_PROBES) {
            fails.push(format!("probes differ by {err:.1e} of range (> {BOUNDED_PROBES:.0e})"));
        }
        for k in ["broken_bonds", "fragments"] {
            if a.values.get(k) != b.values.get(k) {
                fails.push(format!("{k}: {:?} vs {:?}", a.values.get(k), b.values.get(k)));
            }
        }
    }
    let (ka, kb) = (keyed(&base.sem.events), keyed(&other.sem.events));
    if !same_set(&ka, &kb) {
        fails.push(format!("events: {} vs {}", describe(&ka), describe(&kb)));
    } else {
        for (key, ta) in &ka {
            for (x, y) in ta.iter().zip(&kb[key]) {
                if (x - y).abs() > 1.0001 * dt.max(1e-12) && dt > 0.0 {
                    fails.push(format!("events: {:?} bond {} at {y:.6} vs {x:.6} (> 1 substep)", key.0, key.2));
                }
            }
        }
    }
    if !base.rows.is_empty() {
        let d = base.rows.iter().zip(&other.rows).map(|(x, y)| (x[0] - y[0]).abs()).fold(0.0, f64::max);
        if d > 1e-5 || base.rows.iter().zip(&other.rows).any(|(x, y)| x[2] != y[2]) {
            fails.push(format!("bond rows: damage differs by {d:.1e}, or broken bonds differ"));
        }
    }
    if !fails.is_empty() {
        // A scene whose own reference scatters beyond the bound amplifies f32
        // re-association too: judge the knob run by the gate instead.
        if let (Some(set), Some(obs)) = (refs.and_then(|r| r.set.as_ref()), &other.obs) {
            let (spread, _) = gate::spread_difference(&set.reference, &set.spreads);
            if spread > BOUNDED_PROBES {
                let g = accuracy(gpu, case.scene(), case.tweak, knobs, perturbed, set, obs);
                return g.into_iter().map(|f| format!("{f} (sensitive scene: reference spread {spread:.1e}, judged by the gate)")).collect();
            }
        }
    }
    fails
}

// ------------------------------------------------------------------ snapshots

/// A GPU run's outcome as recorded: probes (bits), values, events, final state.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub probes: u64,
    // Bits: JSON floats need not read back exactly.
    pub values: BTreeMap<String, u64>,
    pub events: Vec<(EvKind, usize, usize, u64)>,
    pub state: u64,
    pub rows: u64,
    pub coverage: u64,
}

impl Snapshot {
    pub fn of(run: &GpuRun) -> Snapshot {
        let mut h = 0xcbf29ce484222325;
        let mut values = BTreeMap::new();
        if let Some(o) = &run.obs {
            for (name, s) in &o.probes {
                h = hash_str(name, h);
                for x in s.t.iter().chain(&s.v) {
                    h = gate::fnv(&x.to_bits().to_le_bytes(), h);
                }
            }
            values = o.values.iter().map(|(k, v)| (k.clone(), v.to_bits())).collect();
        }
        let mut r = 0xcbf29ce484222325;
        for row in &run.rows {
            for x in row {
                r = gate::fnv(&x.to_bits().to_le_bytes(), r);
            }
        }
        let events = run.sem.events.iter().map(|e| (e.kind, e.structure, e.bond, e.time.to_bits())).collect();
        Snapshot { probes: h, values, events, state: run.state, rows: r, coverage: run.coverage }
    }

    /// The fields that differ from `other`.
    pub fn changes(&self, other: &Snapshot) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.probes != other.probes {
            out.push("probes");
        }
        if self.values != other.values {
            out.push("values");
        }
        if self.events != other.events {
            out.push("events");
        }
        if self.state != other.state {
            out.push("state");
        }
        if self.rows != other.rows {
            out.push("bonds");
        }
        if self.coverage != other.coverage {
            out.push("coverage");
        }
        out
    }
}

/// Where a run's snapshot lives: per case, knobs and GPU adapter.
pub fn snapshot_path(gpu: &Gpu, name: &str, knobs: &Knobs) -> PathBuf {
    let h = hash_str(&gpu.adapter_info.name, hash_str(&knobs.label(), 0xcbf29ce484222325));
    gate::cache_dir().join("gpu-snapshots").join(format!("{name}-{h:016x}.json"))
}

#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotState {
    New,
    Unchanged,
    Changed(Vec<&'static str>),
    Written,
}

/// Compare a run with its snapshot (`accept`: overwrite it); a missing one is written.
pub fn snapshot(gpu: &Gpu, name: &str, knobs: &Knobs, run: &GpuRun, accept: bool) -> SnapshotState {
    let path = snapshot_path(gpu, name, knobs);
    let now = Snapshot::of(run);
    let write = |s: &Snapshot| {
        std::fs::create_dir_all(path.parent().unwrap()).ok();
        std::fs::write(&path, serde_json::to_string_pretty(s).unwrap()).ok();
    };
    let old: Option<Snapshot> = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok());
    match old {
        None => {
            write(&now);
            SnapshotState::New
        }
        Some(o) if o == now => SnapshotState::Unchanged,
        Some(o) => {
            if accept {
                write(&now);
                SnapshotState::Written
            } else {
                SnapshotState::Changed(o.changes(&now))
            }
        }
    }
}

// ------------------------------------------------------------------ the tier

/// One run of the tier (a case's base run, a knob run or a fuzz scene).
#[derive(Clone, Debug)]
pub struct Row {
    pub case: String,
    pub knobs: String,
    pub seconds: f64,
    pub coverage: u64,
    pub fails: Vec<String>,
    pub snapshot: Option<SnapshotState>,
    /// No cached reference (skipped).
    pub missing: bool,
}

#[derive(Default)]
pub struct Report {
    pub rows: Vec<Row>,
    /// Coverage per case (all its runs).
    pub coverage: Vec<(String, u64)>,
}

impl Report {
    pub fn failures(&self) -> usize {
        self.rows.iter().filter(|r| !r.fails.is_empty()).count()
    }
    pub fn missing(&self) -> usize {
        self.rows.iter().filter(|r| r.missing).count()
    }
    pub fn changed_snapshots(&self) -> usize {
        self.rows.iter().filter(|r| matches!(r.snapshot, Some(SnapshotState::Changed(_)))).count()
    }
}

/// Run the micro catalogue (base runs, knob runs with their metamorphic checks,
/// snapshots), then the fuzz regressions. `only`: name prefixes; `accept`: overwrite
/// changed snapshots.
pub fn run_micro(gpu: &Gpu, only: Option<&[String]>, accept: bool, explain: bool) -> Report {
    let mut report = Report::default();
    for case in micro_cases().iter().filter(|c| only.is_none_or(|o| o.iter().any(|p| c.name.starts_with(p.as_str())))) {
        let Some(refs) = micro_reference(case, false) else {
            report.rows.push(Row { case: case.name.into(), knobs: case.base.label(), seconds: 0.0, coverage: 0, fails: Vec::new(), snapshot: None, missing: true });
            continue;
        };
        let base = match run_gpu(gpu, case, &case.base) {
            Ok(r) => r,
            Err(e) => {
                report.rows.push(Row { case: case.name.into(), knobs: case.base.label(), seconds: 0.0, coverage: 0, fails: vec![format!("GPU error: {e}")], snapshot: None, missing: false });
                continue;
            }
        };
        if explain {
            eprintln!("{}: {}", case.name, case.semantic);
            eprintln!("  reference events: {}", describe(&keyed(&refs.semantics[0].events)));
            eprintln!("  gpu events:       {}", describe(&keyed(&base.sem.events)));
            for (i, s) in refs.semantics.iter().enumerate().skip(1) {
                let t: Vec<String> = s.events.iter().map(|e| format!("{:?}{}@{:.6}", e.kind, e.bond, e.time)).collect();
                eprintln!("    ensemble {i}: {}", t.join(" "));
            }
            for e in &refs.semantics[0].events {
                eprintln!("    ref {:?} s{} b{} t={:.6}", e.kind, e.structure, e.bond, e.time);
            }
            for e in &base.sem.events {
                eprintln!("    gpu {:?} s{} b{} t={:.6}", e.kind, e.structure, e.bond, e.time);
            }
            eprintln!("  coverage: {}", cov::names(base.coverage).join(" "));
            if let (Some(set), Some(o)) = (&refs.set, &base.obs) {
                let (_, notes) = gate::difference(&set.reference, o);
                eprintln!("  gpu vs ref: {}", notes.join("; "));
                eprintln!("  ref values: {:?}", set.reference.values);
            }
            let (r, s) = closure(&refs.semantics[0]);
            let (g, _) = closure(&base.sem);
            eprintln!("  energy closure: reference {r:.2e}, gpu {g:.2e} of {s:.3e} J");
        }
        let mut coverage = base.coverage;
        // Snapshots per case and scene: editing a case starts a new one.
        let tag = format!("{}-{:08x}", case.name, gate::fnv(case_key(case).to_string().as_bytes(), 0xcbf29ce484222325) as u32);
        let mut runs = Vec::new();
        for k in &case.knob_runs {
            match run_gpu(gpu, case, k) {
                Ok(r) => {
                    coverage |= r.coverage;
                    runs.push((k.clone(), Ok(r)));
                }
                Err(e) => runs.push((k.clone(), Err(e))),
            }
        }
        let fails = check_case(gpu, case, &refs, &base, coverage);
        let snap = snapshot(gpu, &tag, &case.base, &base, accept);
        report.rows.push(Row { case: case.name.into(), knobs: case.base.label(), seconds: base.seconds, coverage: base.coverage, fails, snapshot: Some(snap), missing: false });
        for (k, r) in runs {
            let (fails, seconds, c, snap) = match r {
                Ok(r) => (metamorphic(gpu, case, Some(&refs), &base, &r, &k), r.seconds, r.coverage, Some(snapshot(gpu, &tag, &k, &r, accept))),
                Err(e) => (vec![format!("GPU error: {e}")], 0.0, 0, None),
            };
            report.rows.push(Row { case: case.name.into(), knobs: k.label(), seconds, coverage: c, fails, snapshot: snap, missing: false });
        }
        report.coverage.push((case.name.into(), coverage));
    }
    if only.is_none() {
        let fuzz = run_fuzz(gpu, gate::FUZZ_REGRESSIONS, accept);
        report.rows.extend(fuzz.rows);
    }
    report
}

/// Fuzz scenes by seed: the gate against the cached reference, batch boundaries
/// bitwise, wide islands (`wide=2`) judged by the gate too (a fracturing fuzz scene
/// amplifies f32 re-association beyond any fixed bound).
pub fn run_fuzz(gpu: &Gpu, seeds: &[u64], accept: bool) -> Report {
    let mut report = Report::default();
    for &seed in seeds {
        let scene = fuzz_scene(seed);
        let name = scene.name.clone();
        let Some(set) = gate::reference_set(&scene, false) else {
            report.rows.push(Row { case: name, knobs: "default".into(), seconds: 0.0, coverage: 0, fails: Vec::new(), snapshot: None, missing: true });
            continue;
        };
        let default = Knobs::default();
        let base = match with_knobs(&default, || run_world(gpu, &scene, None)) {
            Ok(r) => r,
            Err(e) => {
                report.rows.push(Row { case: name, knobs: "default".into(), seconds: 0.0, coverage: 0, fails: vec![format!("GPU error: {e}")], snapshot: None, missing: false });
                continue;
            }
        };
        let fails = accuracy(gpu, &scene, None, &default, gate::perturbed_by, &set, base.obs.as_ref().unwrap());
        if !fails.is_empty() {
            let b = &scene.bodies[0];
            let supports = b.chunks.iter().filter(|c| c.support != Support::None).count();
            eprintln!("{name}: {} chunks of {}, {supports} fixed, ground {}, {:?}, impactor {:?}", b.chunks.len(), b.chunks[0].material, scene.ground.is_some(), scene.sim.solve_mode, scene.impactors.first().map(|i| (&i.shape, i.mass, i.velocity)));
            let (_, notes) = gate::difference(&set.reference, base.obs.as_ref().unwrap());
            let (_, spread) = gate::spread_difference(&set.reference, &set.spreads);
            eprintln!("  gpu: {}", notes.join("; "));
            eprintln!("  ref spread: {}", spread.join("; "));
        }
        let snap = snapshot(gpu, &name, &default, &base, accept);
        report.rows.push(Row { case: name.clone(), knobs: "default".into(), seconds: base.seconds, coverage: base.coverage, fails, snapshot: Some(snap), missing: false });
        // Batches of 3 rounds: boundaries at every phase of the contact pipeline at a third
        // of the host round trips of batch=1 (the micro catalogue runs batch=1).
        for k in [Knobs::batch(3), Knobs::wide(2)] {
            let r = with_knobs(&k, || run_world(gpu, &scene, None));
            let (fails, seconds, c) = match r {
                Ok(r) => {
                    let f = if k.bitwise() {
                        let case = MicroCase::world("fuzz", "", scene.clone(), Expect::default(), vec![]);
                        metamorphic(gpu, &case, None, &base, &r, &k)
                    } else {
                        accuracy(gpu, &scene, None, &k, gate::perturbed_by, &set, r.obs.as_ref().unwrap())
                    };
                    (f, r.seconds, r.coverage)
                }
                Err(e) => (vec![format!("GPU error: {e}")], 0.0, 0),
            };
            report.rows.push(Row { case: name.clone(), knobs: k.label(), seconds, coverage: c, fails, snapshot: None, missing: false });
        }
    }
    report
}

/// The fuzz seeds of a run: scene seeds hashed from (`seed`, i).
pub fn fuzz_seeds(seed: u64, n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| stress_ref::rng::hash(seed, &[i])).collect()
}
