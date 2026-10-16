//! Jolt Physics 5.6 adapter (CPU). A small C++ shim (`src/jolt_shim.cpp`) builds and steps the
//! scene through Jolt's public API; this file drives it over FFI. `build.rs` builds Jolt from
//! `bench/third_party/JoltPhysics` (pinned by `third_party/fetch.sh`) with CMake (Distribution,
//! IPO on, asserts/profiler/debug renderer off, 16-bit object layers, GPU compute off) and
//! compiles the shim with the library's own `JPH_*` defines. See README.md.
//!
//! Matched to the scene (README "Fairness settings"): mass and principal inertia exactly from
//! the scene (`MassAndInertiaProvided`; convex-hull bodies wrapped in an
//! `OffsetCenterOfMassShape` so the body origin stays the scene's centre of mass), friction per
//! body combined by Jolt's default geometric mean sqrt(a·b) (= three-avbd), restitution 0, no
//! damping, max linear/angular velocity raised to 1e4, discrete motion quality (no CCD),
//! sleeping off unless `--sleep on`, joined bodies never collide (a group filter over the
//! scene's joint pairs). `--iterations` = `mNumVelocitySteps` (Jolt default 10); position steps
//! default 2 (`--set pos_steps=N`). Everything else is Jolt's default and recorded in config.
//!
//! `--set` keys: pos_steps=N, max_pairs=N, max_contacts=N, temp_mb=N, max_lin_vel=X,
//! max_ang_vel=X, convex_radius=X (max box/hull convex radius, default 0.05 = cDefaultConvexRadius),
//! convex_radius_frac=F (radius = min(convex_radius, F·min half extent), default 0.1),
//! joints=specialised|sixdof (D6 joints as point/hinge/swing-twist where exact, else 6DOF; or
//! all as SixDOFConstraint), jobs=N, barriers=N.

use anyhow::{bail, ensure, Context, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
struct JBody {
    shape: u32,
    is_static: u32,
    half: [f32; 3],
    hull_first: u32,
    hull_count: u32,
    mass: f32,
    inertia: [f32; 3],
    friction: f32,
    restitution: f32,
    position: [f32; 3],
    rotation: [f32; 4],
    linvel: [f32; 3],
    angvel: [f32; 3],
}

#[repr(C)]
struct JJoint {
    body_a: u32,
    body_b: u32,
    collide: u32,
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
    kind: u32,
    frame_a: [f32; 4],
    frame_b: [f32; 4],
    motion: u32,
    twist: [f32; 2],
    swing: [f32; 2],
    cone: u32,
    linear_x: [f32; 2],
    drive_dof: u32,
    drive_velocity: f32,
    drive_force_limit: f32,
    drive_stiffness: f32,
    drive_damping: f32,
    drive_target: f32,
}

fn j_joint(j: &bench_core::Joint) -> JJoint {
    use bench_core::scene::{JointKind, Motion};
    let mut p = JJoint {
        body_a: j.body_a,
        body_b: j.body_b,
        collide: j.collide as u32,
        anchor_a: j.anchor_a,
        anchor_b: j.anchor_b,
        kind: 0,
        frame_a: [0.0, 0.0, 0.0, 1.0],
        frame_b: [0.0, 0.0, 0.0, 1.0],
        motion: 0,
        twist: [0.0; 2],
        swing: [0.0; 2],
        cone: 0,
        linear_x: [0.0; 2],
        drive_dof: u32::MAX,
        drive_velocity: 0.0,
        drive_force_limit: 0.0,
        drive_stiffness: 0.0,
        drive_damping: 0.0,
        drive_target: 0.0,
    };
    if let Some(d) = j.d6.filter(|_| j.kind == JointKind::D6) {
        p.kind = 1;
        p.frame_a = d.frame_a;
        p.frame_b = d.frame_b;
        p.motion = d.motion.iter().enumerate().fold(0, |m, (i, x)| {
            m | (match x {
                Motion::Locked => 0,
                Motion::Limited => 1,
                Motion::Free => 2,
            } << (2 * i))
        });
        p.twist = d.twist;
        p.swing = d.swing;
        p.cone = d.cone as u32;
        p.linear_x = d.linear_x;
        if let Some(dr) = d.drive {
            p.drive_dof = dr.dof;
            p.drive_velocity = dr.velocity;
            p.drive_force_limit = dr.force_limit;
            p.drive_stiffness = dr.stiffness;
            p.drive_damping = dr.damping;
            p.drive_target = dr.target;
        }
    }
    p
}

#[repr(C)]
struct JHulls {
    n_hulls: u32,
    first: *const u32,
    count: *const u32,
    n_vertices: u32,
    vertices: *const f32,
}

#[repr(C)]
struct JSettings {
    gravity: [f32; 3],
    dt: f32,
    threads: u32,
    vel_steps: u32,
    pos_steps: u32,
    sleep: u32,
    max_bodies: u32,
    max_pairs: u32,
    max_contacts: u32,
    joint_mode: u32,
    temp_bytes: u64,
    max_linear_velocity: f32,
    max_angular_velocity: f32,
    box_radius_frac: f32,
    max_convex_radius: f32,
    plane_half_extent: f32,
    max_jobs: u32,
    max_barriers: u32,
}

#[repr(C)]
#[derive(Default)]
struct JInfo {
    version_major: u32,
    version_minor: u32,
    version_patch: u32,
    version_id_ok: u32,
    baumgarte: f32,
    penetration_slop: f32,
    speculative_distance: f32,
    max_penetration_distance: f32,
    linear_cast_threshold: f32,
    manifold_tolerance: f32,
    min_velocity_for_restitution: f32,
    time_before_sleep: f32,
    point_velocity_sleep_threshold: f32,
    deterministic: u32,
    warm_start: u32,
    body_pair_cache: u32,
    manifold_reduction: u32,
    large_island_splitter: u32,
    check_active_edges: u32,
    vel_steps: u32,
    pos_steps: u32,
    workers: u32,
    max_constraint_size: u32,
    boxes: u32,
    spheres: u32,
    planes: u32,
    box_shapes_unique: u32,
    box_radius_min: f32,
    box_radius_max: f32,
    hulls_built: u32,
    hull_vertices_in: u32,
    hull_vertices_out: u32,
    hulls_changed: u32,
    compounds: u32,
    hull_radius_min: f32,
    hull_radius_max: f32,
    hull_max_outside: f32,
    max_com_shift: f32,
    hull_tolerance: f32,
    hull_max_error_convex_radius: f32,
    n_point: u32,
    n_hinge: u32,
    n_swing_twist: u32,
    n_six_dof: u32,
    n_motors: u32,
    filtered_bodies: u32,
    filtered_pairs: u32,
    default_convex_radius: f32,
}

#[repr(C)]
#[derive(Default)]
struct JCounters {
    bodies: u32,
    active_bodies: u32,
    body_pairs: u32,
    manifolds: u32,
    constraints: u32,
    errors_last: u32,
    errors_sticky: u32,
    steps: u32,
}

extern "C" {
    fn bench_jolt_max_constraint_size() -> u32;
    fn bench_jolt_create(
        bodies: *const JBody,
        nb: u32,
        joints: *const JJoint,
        nj: u32,
        hulls: *const JHulls,
        settings: *const JSettings,
        info: *mut JInfo,
        err: *mut c_char,
        errlen: u32,
    ) -> *mut c_void;
    fn bench_jolt_step(world: *mut c_void) -> u32;
    fn bench_jolt_read_state(world: *mut c_void, out: *mut f32);
    fn bench_jolt_counters(world: *mut c_void, out: *mut JCounters);
    fn bench_jolt_destroy(world: *mut c_void);
}

/// EPhysicsUpdateError bits.
fn update_errors(bits: u32) -> String {
    let mut v = Vec::new();
    if bits & 1 != 0 {
        v.push("ManifoldCacheFull (raise --set max_contacts)");
    }
    if bits & 2 != 0 {
        v.push("BodyPairCacheFull (raise --set max_pairs)");
    }
    if bits & 4 != 0 {
        v.push("ContactConstraintsFull (raise --set max_contacts)");
    }
    v.join(", ")
}

struct Jolt {
    world: *mut c_void,
    n: usize,
    threads: usize,
    iterations: u32,
    sleeping: bool,
    version: String,
    config: BTreeMap<String, String>,
}

impl Drop for Jolt {
    fn drop(&mut self) {
        unsafe { bench_jolt_destroy(self.world) };
    }
}

fn parse<T: std::str::FromStr>(args: &mut Args, key: &str) -> Result<Option<T>>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    args.take(key).map(|v| v.parse::<T>().with_context(|| format!("--set {key}={v}"))).transpose()
}

fn build(scene: &Scene, args: &mut Args) -> Result<Jolt> {
    let n = scene.bodies.len();
    let threads = args.cpu_threads();
    let vel_steps = args.iterations.unwrap_or(10);
    let pos_steps: u32 = parse(args, "pos_steps")?.unwrap_or(2);
    ensure!(vel_steps >= 1 && vel_steps <= 255, "--iterations (velocity steps) must be 1..255");
    ensure!(pos_steps <= 255, "pos_steps must be 0..255");

    let constraint_size = unsafe { bench_jolt_max_constraint_size() } as u64;
    // Jolt limits max contact constraints so that max * cMaxConstraintSize fits in 32 bits
    let contacts_limit = (u32::MAX as u64 / constraint_size) as u32;
    let max_pairs: u32 = parse(args, "max_pairs")?.unwrap_or_else(|| (16 * n as u64).clamp(65536, 1 << 26) as u32);
    let max_contacts: u32 = parse(args, "max_contacts")?.unwrap_or_else(|| (16 * n as u64).clamp(65536, contacts_limit as u64) as u32);
    ensure!(max_contacts <= contacts_limit, "max_contacts {max_contacts} exceeds Jolt's limit {contacts_limit}");
    // The contact constraint buffer (max_contacts × cMaxConstraintSize) is taken from the temp
    // allocator every step; the rest covers islands, broadphase and narrowphase scratch.
    // Untouched pages are never committed, so generous sizing costs nothing.
    let default_temp = max_contacts as u64 * (constraint_size + 4) + n as u64 * 4096 + (256 << 20);
    let temp_bytes: u64 = match parse::<u64>(args, "temp_mb")? {
        Some(mb) => mb << 20,
        None => default_temp,
    };
    let max_lin_vel: f32 = parse(args, "max_lin_vel")?.unwrap_or(1e4);
    let max_ang_vel: f32 = parse(args, "max_ang_vel")?.unwrap_or(1e4);
    let max_convex_radius: f32 = parse(args, "convex_radius")?.unwrap_or(0.05);
    let radius_frac: f32 = parse(args, "convex_radius_frac")?.unwrap_or(0.1);
    let joint_mode = match args.take("joints").as_deref() {
        None | Some("specialised") | Some("specialized") => 0u32,
        Some("sixdof") => 1,
        Some(j) => bail!("--set joints={j}: expected specialised|sixdof"),
    };
    let max_jobs: u32 = parse(args, "jobs")?.unwrap_or(2048); // cMaxPhysicsJobs
    let max_barriers: u32 = parse(args, "barriers")?.unwrap_or(8); // cMaxPhysicsBarriers

    // Plane half extent: Jolt's default 1000, larger if the scene is
    let mut extent: f32 = 0.0;
    for b in &scene.bodies {
        if !matches!(b.shape, Shape::Plane) {
            extent = extent.max(b.position.iter().map(|x| x.abs()).fold(0.0, f32::max) + b.radius());
        }
    }
    let plane_half_extent = (4.0 * extent).max(1000.0);

    let bodies: Vec<JBody> = scene
        .bodies
        .iter()
        .map(|b| {
            let (shape, half, hull_first, hull_count) = match b.shape {
                Shape::Box(h) => (0, h, 0, 0),
                Shape::Sphere(r) => (1, [r, 0.0, 0.0], 0, 0),
                Shape::Hulls { first, count, radius } => (2, [radius, 0.0, 0.0], first, count),
                Shape::Plane => (3, [0.0; 3], 0, 0),
            };
            JBody {
                shape,
                is_static: b.is_static as u32,
                half,
                hull_first,
                hull_count,
                mass: b.mass,
                inertia: b.inertia,
                friction: b.friction,
                restitution: b.restitution,
                position: b.position,
                rotation: b.rotation,
                linvel: b.linvel,
                angvel: b.angvel,
            }
        })
        .collect();
    let joints: Vec<JJoint> = scene.joints.iter().map(j_joint).collect();
    let mut hull_first = Vec::with_capacity(scene.hulls.len());
    let mut hull_count = Vec::with_capacity(scene.hulls.len());
    let mut vertices: Vec<f32> = Vec::new();
    for h in &scene.hulls {
        hull_first.push((vertices.len() / 3) as u32);
        hull_count.push(h.len() as u32);
        vertices.extend(h.iter().flatten());
    }
    let hulls = JHulls {
        n_hulls: scene.hulls.len() as u32,
        first: hull_first.as_ptr(),
        count: hull_count.as_ptr(),
        n_vertices: (vertices.len() / 3) as u32,
        vertices: vertices.as_ptr(),
    };
    let s = JSettings {
        gravity: scene.meta.gravity,
        dt: scene.meta.dt,
        threads: threads as u32,
        vel_steps,
        pos_steps,
        sleep: args.sleep as u32,
        max_bodies: (n + 64) as u32,
        max_pairs,
        max_contacts,
        joint_mode,
        temp_bytes,
        max_linear_velocity: max_lin_vel,
        max_angular_velocity: max_ang_vel,
        box_radius_frac: radius_frac,
        max_convex_radius,
        plane_half_extent,
        max_jobs,
        max_barriers,
    };
    let mut info = JInfo::default();
    let mut err = [0 as c_char; 1024];
    let world = unsafe {
        bench_jolt_create(
            bodies.as_ptr(),
            bodies.len() as u32,
            joints.as_ptr(),
            joints.len() as u32,
            &hulls,
            &s,
            &mut info,
            err.as_mut_ptr(),
            err.len() as u32,
        )
    };
    if world.is_null() {
        let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().to_string();
        bail!("Jolt world creation failed: {msg}");
    }
    ensure!(info.version_id_ok == 1, "Jolt version/feature ID mismatch");

    let rev = env!("BENCH_JOLT_REV");
    let f = |x: f32| format!("{x}");
    let mut config = BTreeMap::new();
    config.insert("source".into(), format!("jrouwe/JoltPhysics v{}.{}.{} @ {rev}", info.version_major, info.version_minor, info.version_patch));
    config.insert("build".into(), format!("CMake {}", env!("BENCH_JOLT_CMAKE")));
    config.insert("defines".into(), format!("{} (VerifyJoltVersionID ok)", env!("BENCH_JOLT_DEFINES")));
    config.insert("shimFlags".into(), env!("BENCH_JOLT_FLAGS").into());
    config.insert("velocitySteps".into(), info.vel_steps.to_string());
    config.insert("positionSteps".into(), info.pos_steps.to_string());
    config.insert("collisionSteps".into(), "1 (Update(dt, 1, ...))".into());
    config.insert(
        "threads".into(),
        format!("{threads} total = calling thread + JobSystemThreadPool({max_jobs} jobs, {max_barriers} barriers, {} workers)", info.workers),
    );
    config.insert("maxBodies".into(), s.max_bodies.to_string());
    config.insert("maxBodyPairs".into(), max_pairs.to_string());
    config.insert("maxContactConstraints".into(), format!("{max_contacts} (cMaxConstraintSize {constraint_size} B)"));
    config.insert("tempAllocator".into(), format!("TempAllocatorImpl {} MiB (no malloc fallback)", temp_bytes >> 20));
    config.insert("layers".into(), "2 object layers (non-moving, moving) = 2 broadphase layers; static-static pairs culled".into());
    config.insert("baumgarte".into(), f(info.baumgarte) + " (default)");
    config.insert("penetrationSlop".into(), f(info.penetration_slop) + " (default)");
    config.insert("speculativeContactDistance".into(), f(info.speculative_distance) + " (default)");
    config.insert("maxPenetrationDistance".into(), f(info.max_penetration_distance) + " (default)");
    config.insert("manifoldTolerance".into(), f(info.manifold_tolerance) + " (default)");
    config.insert("minVelocityForRestitution".into(), f(info.min_velocity_for_restitution) + " (default)");
    config.insert("deterministicSimulation".into(), (info.deterministic != 0).to_string() + " (default)");
    config.insert("constraintWarmStart".into(), (info.warm_start != 0).to_string() + " (default)");
    config.insert("bodyPairContactCache".into(), (info.body_pair_cache != 0).to_string() + " (default)");
    config.insert("manifoldReduction".into(), (info.manifold_reduction != 0).to_string() + " (default)");
    config.insert("largeIslandSplitter".into(), (info.large_island_splitter != 0).to_string() + " (default)");
    config.insert("checkActiveEdges".into(), (info.check_active_edges != 0).to_string() + " (default)");
    config.insert("allowSleeping".into(), args.sleep.to_string());
    if args.sleep {
        config.insert("sleep".into(), format!("timeBeforeSleep {} s, pointVelocitySleepThreshold {} m/s (defaults)", info.time_before_sleep, info.point_velocity_sleep_threshold));
    }
    config.insert("motionQuality".into(), "Discrete (no CCD; linear cast unused)".into());
    config.insert("frictionCombine".into(), "geometric mean sqrt(a·b) (Jolt default; = three-avbd)".into());
    config.insert("restitutionCombine".into(), "max(a, b) (Jolt default); scene restitution 0".into());
    config.insert("linearDamping".into(), "0 (Jolt default 0.05)".into());
    config.insert("angularDamping".into(), "0 (Jolt default 0.05)".into());
    config.insert("maxLinearVelocity".into(), format!("{max_lin_vel} (Jolt default 500)"));
    config.insert("maxAngularVelocity".into(), format!("{max_ang_vel} (Jolt default 0.25·π·60 ≈ 47.1)"));
    config.insert("massProperties".into(), "MassAndInertiaProvided: scene mass, diagonal scene inertia (body frame = principal frame)".into());
    config.insert(
        "convexRadius".into(),
        format!(
            "min({max_convex_radius}, {radius_frac}·smallest half extent) per box / hull (Jolt default {}); boxes keep their outer size, edges rounded",
            info.default_convex_radius
        ),
    );
    if info.boxes > 0 {
        config.insert(
            "boxes".into(),
            format!("{} BoxShape bodies ({} distinct shapes), convex radius {}..{}", info.boxes, info.box_shapes_unique, info.box_radius_min, info.box_radius_max),
        );
    }
    if info.spheres > 0 {
        config.insert("spheres".into(), format!("{} SphereShape bodies", info.spheres));
    }
    if info.hulls_built > 0 {
        config.insert(
            "hullCooking".into(),
            format!(
                "ConvexHullShapeSettings(points, maxConvexRadius) hullTolerance {} maxErrorConvexRadius {} (defaults); several hulls per body -> \
                 StaticCompoundShape (identity sub-transforms); every hull body wrapped in OffsetCenterOfMassShape(-COM)",
                info.hull_tolerance, info.hull_max_error_convex_radius
            ),
        );
        config.insert(
            "hullCookingResult".into(),
            format!(
                "{} hulls: {} input vertices -> {} hull vertices ({} hulls changed count), farthest input point outside its hull {:.2e} m, \
                 convex radius used {:.4}..{:.4} m, {} compound bodies, largest Jolt COM shift compensated {:.2e} m",
                info.hulls_built,
                info.hull_vertices_in,
                info.hull_vertices_out,
                info.hulls_changed,
                info.hull_max_outside,
                info.hull_radius_min,
                info.hull_radius_max,
                info.compounds,
                info.max_com_shift
            ),
        );
    }
    if info.planes > 0 {
        config.insert("plane".into(), format!("static PlaneShape(normal +y, through body origin) at the scene pose, half extent {plane_half_extent}"));
    }
    if !scene.joints.is_empty() {
        config.insert(
            "joints".into(),
            format!(
                "{} PointConstraint, {} HingeConstraint, {} SwingTwistConstraint, {} SixDOFConstraint ({} motors); LocalToBodyCOM frames at the scene anchors{}",
                info.n_point,
                info.n_hinge,
                info.n_swing_twist,
                info.n_six_dof,
                info.n_motors,
                if joint_mode == 1 { "; joints=sixdof" } else { "" }
            ),
        );
        config.insert(
            "jointCollision".into(),
            format!(
                "joined bodies never collide: GroupFilter subclass over the scene's {} non-colliding body pairs ({} bodies carry it)",
                info.filtered_pairs, info.filtered_bodies
            ),
        );
    }
    Ok(Jolt {
        world,
        n,
        threads,
        iterations: vel_steps,
        sleeping: args.sleep,
        version: format!("{}.{}.{} ({})", info.version_major, info.version_minor, info.version_patch, &rev[..12.min(rev.len())]),
        config,
    })
}

impl Engine for Jolt {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "jolt".into(),
            version: self.version.clone(),
            backend: "cpu".into(),
            device: bench_core::machine::cpu_model(),
            threads: self.threads,
            pipelined: false,
            solver: "sequential impulse (velocity + position steps)".into(),
            iterations: self.iterations,
            iterations_meaning: "velocity steps (position steps in config)".into(),
            sleeping: self.sleeping,
            config: self.config.clone(),
        }
    }

    fn step(&mut self) -> Result<()> {
        let bits = unsafe { bench_jolt_step(self.world) };
        ensure!(bits == 0, "Jolt Update reported {} — contacts were dropped", update_errors(bits));
        Ok(())
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        ensure!(out.len() == self.n, "read_state: {} samples for {} bodies", out.len(), self.n);
        const _: () = assert!(std::mem::size_of::<BodySample>() == 13 * 4);
        let floats = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out.len() * 13) };
        unsafe { bench_jolt_read_state(self.world, floats.as_mut_ptr()) };
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        let mut c = JCounters::default();
        unsafe { bench_jolt_counters(self.world, &mut c) };
        Ok(json!({
            "bodies": c.bodies,
            "activeBodies": c.active_bodies,
            "bodyPairs": c.body_pairs,
            "contactManifolds": c.manifolds,
            "joints": c.constraints,
            "updateErrors": c.errors_sticky,
            "overflow": c.errors_sticky != 0,
        }))
    }
}

fn main() -> Result<()> {
    bench_core::main_with(build)
}
