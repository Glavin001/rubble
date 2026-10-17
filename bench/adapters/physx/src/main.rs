//! NVIDIA PhysX 5 adapter. A small C++ shim (`src/physx_shim.cpp`) builds and steps the scene
//! through the public PhysX API; this file drives it over FFI. PhysX comes either from the
//! `physx-sys` crate (default feature `vendored`: PhysX 5.1.3 built from source, CPU only) or
//! from an official NVIDIA-Omniverse/PhysX SDK build (`PHYSX_SDK_DIR`, CPU, or CUDA with
//! feature `gpu`). See README.md.
//!
//! Matched to the scene (README "Fairness"): mass and principal inertia set exactly from the
//! scene file (identity centre-of-mass frame), friction from the scene (static = dynamic)
//! combined with `eAVERAGE` (three-avbd uses sqrt(a·b); the suite's coefficients are 0.5/0.6,
//! so they differ by <0.5%), restitution 0, linear and angular damping 0 (PhysX's default
//! angular damping is 0.05), max angular velocity 1e4, spherical joints with contacts between
//! joined bodies disabled, CCD off, PCM on, no stabilization or enhanced determinism, sleeping
//! off (sleep threshold 0, so the wake counter is reset every step) unless `--sleep on`.
//! Solver TGS; `--iterations` = position iterations (default 4, PhysX's default), velocity
//! iterations 1 (`--set vel_iters=N`). Everything else is PhysX's default.
//!
//! Format v2 scenes: each convex hull is cooked with `PxCreateConvexMesh` (eCOMPUTE_CONVEX,
//! vertex and polygon limit 64; GPU dynamics adds buildGPUData, and eGPU_COMPATIBLE on SDKs that
//! still have it) and becomes one `PxConvexMeshGeometry` shape with an identity local pose (the
//! vertices are in the body frame); a multi-hull body is one actor with several shapes. Mass
//! properties stay the scene's (PhysX never recomputes them from the shapes). How much cooking
//! changed the hulls is reported in `config.hullCookingResult`. The ground plane is a
//! PxRigidStatic at the scene pose with a PxPlaneGeometry whose local pose turns PhysX's +x
//! plane normal onto the body's +y.
//!
//! `--set` keys: solver=tgs|pgs, vel_iters=N, broadphase=abp|sap|mbp|pabp (CPU),
//! dispatch=wait|yield|spin (PxDefaultCpuDispatcher wait mode), max_ang_vel=X,
//! max_depen_vel=X (per-body max depenetration velocity; default: PhysX's, 1e32).
//! GPU (feature `gpu`): gpu=on|off (default on), gpu_device=N, gpu_partitions=N,
//! gpu_contacts=N, gpu_patches=N, gpu_found_lost=N, gpu_agg_found_lost=N, gpu_agg_pairs=N,
//! gpu_heap_mb=N (rounded up to a power of two), gpu_temp_mb=N, gpu_stack_mb=N.

#[cfg(feature = "vendored")]
use physx_sys as _; // links the PhysX static library built by physx-sys

use anyhow::{bail, ensure, Context, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
struct PxBody {
    /// 0 box, 1 sphere, 2 convex hulls, 3 plane
    shape: u32,
    is_static: u32,
    /// box half extents; sphere [r, 0, 0]; hulls [bounding radius, 0, 0]
    half: [f32; 3],
    /// hulls: `PxHulls` entries `hull_first..hull_first + hull_count`
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
struct PxJoint {
    body_a: u32,
    body_b: u32,
    collide: u32,
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
    /// Scene format v3: 1 = PxD6Joint with the fields below; 0 = PxSphericalJoint.
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

fn px_joint(j: &bench_core::Joint) -> PxJoint {
    use bench_core::scene::Motion;
    let mut p = PxJoint {
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
    if let Some(d) = j.d6.filter(|_| j.kind == bench_core::scene::JointKind::D6) {
        p.kind = 1;
        p.frame_a = d.frame_a;
        p.frame_b = d.frame_b;
        p.motion = d.motion.iter().enumerate().fold(0, |m, (i, x)| {
            m | (match x { Motion::Locked => 0, Motion::Limited => 1, Motion::Free => 2 } << (2 * i))
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

/// The scene's convex hulls: hull `h` is vertices `first[h]..first[h] + count[h]` of
/// `vertices` (xyz, body frame).
#[repr(C)]
struct PxHulls {
    n_hulls: u32,
    first: *const u32,
    count: *const u32,
    n_vertices: u32,
    vertices: *const f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PxSettings {
    gravity: [f32; 3],
    dt: f32,
    threads: u32,
    solver: u32,
    pos_iters: u32,
    vel_iters: u32,
    broadphase: u32,
    sleep: u32,
    dispatch_mode: u32,
    max_angular_velocity: f32,
    max_depenetration_velocity: f32,
    gpu: u32,
    gpu_device: i32,
    gpu_max_partitions: u32,
    gpu_max_rigid_contacts: u32,
    gpu_max_rigid_patches: u32,
    gpu_found_lost_pairs: u32,
    gpu_found_lost_aggregate_pairs: u32,
    gpu_total_aggregate_pairs: u32,
    gpu_heap_capacity: u32,
    gpu_collision_stack: u32,
    gpu_temp_buffer: u64,
}

#[repr(C)]
struct PxInfo {
    version_major: u32,
    version_minor: u32,
    version_bugfix: u32,
    gpu_active: u32,
    broadphase: u32,
    solver: u32,
    contact_offset: f32,
    rest_offset: f32,
    max_depenetration_velocity: f32,
    max_angular_velocity: f32,
    max_linear_velocity: f32,
    sleep_threshold: f32,
    wake_counter_reset: f32,
    bounce_threshold: f32,
    friction_offset_threshold: f32,
    friction_correlation_distance: f32,
    tolerance_length: f32,
    tolerance_speed: f32,
    device: [c_char; 128],
    // convex cooking (format v2 scenes)
    hulls_cooked: u32,
    hull_vertices_in: u32,
    hull_vertices_out: u32,
    hulls_vertex_count_changed: u32,
    hull_polygons: u32,
    hull_max_polygons: u32,
    hulls_gpu_incompatible: u32,
    hulls_new_vertices: u32,
    hull_max_outside: f32,
    cook_gpu_flag: u32,       // PxConvexFlag::eGPU_COMPATIBLE set (SDKs that have it)
    cook_build_gpu_data: u32, // PxCookingParams::buildGPUData
    cook_vertex_limit: u32,
    cook_polygon_limit: u32,
    planes: u32,
}

#[repr(C)]
#[derive(Default)]
struct PxCounters {
    awake: u32,
    active_dynamic: u32,
    active_constraints: u32,
    axis_constraints: u32,
    contact_pairs: u32,
    contact_pairs_touching: u32,
    contact_pairs_cache_hits: u32,
    new_pairs: u32,
    lost_pairs: u32,
    new_touches: u32,
    lost_touches: u32,
    partitions: u32,
    dynamic_bodies: u32,
    static_bodies: u32,
    errors: u32,
}

extern "C" {
    fn bench_px_create(
        bodies: *const PxBody,
        nb: u32,
        joints: *const PxJoint,
        nj: u32,
        hulls: *const PxHulls,
        settings: *const PxSettings,
        info: *mut PxInfo,
        err: *mut c_char,
        errlen: u32,
    ) -> *mut c_void;
    fn bench_px_step(world: *mut c_void, dt: f32) -> i32;
    fn bench_px_read_state(world: *mut c_void, out: *mut f32);
    fn bench_px_counters(world: *mut c_void, out: *mut PxCounters);
    fn bench_px_destroy(world: *mut c_void);
}

struct PhysX {
    world: *mut c_void,
    n: usize,
    dt: f32,
    threads: usize,
    gpu: bool,
    device: String,
    version: String,
    solver: &'static str,
    iterations: u32,
    sleeping: bool,
    config: BTreeMap<String, String>,
}

impl Drop for PhysX {
    fn drop(&mut self) {
        unsafe { bench_px_destroy(self.world) };
    }
}

fn on_off(v: &str) -> Result<bool> {
    match v {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        _ => bail!("expected on/off, got {v}"),
    }
}

const BROADPHASES: [&str; 5] = ["abp", "sap", "mbp", "pabp", "gpu"];

fn build(scene: &Scene, args: &mut Args) -> Result<PhysX> {
    let gpu_built = cfg!(feature = "gpu");
    let gpu = match args.take("gpu") {
        Some(v) => on_off(&v)?,
        None => gpu_built,
    };
    ensure!(!gpu || gpu_built, "--set gpu=on needs a build with the `gpu` feature (official SDK, see README)");

    let solver = match args.take("solver").as_deref() {
        None | Some("tgs") => 0u32,
        Some("pgs") => 1,
        Some(s) => bail!("--set solver={s}: expected tgs or pgs"),
    };
    let pos_iters = args.iterations.unwrap_or(4);
    let vel_iters: u32 = args.take("vel_iters").map(|v| v.parse()).transpose().context("vel_iters")?.unwrap_or(1);
    ensure!((1..=255).contains(&pos_iters) && vel_iters <= 255, "PhysX iteration counts: position 1..255, velocity 0..255");
    let broadphase = match args.take("broadphase") {
        None => 0u32,
        Some(b) => BROADPHASES[..4].iter().position(|x| *x == b).with_context(|| format!("--set broadphase={b}: expected abp|sap|mbp|pabp"))? as u32,
    };
    let dispatch_mode = match args.take("dispatch").as_deref() {
        None | Some("wait") => 0u32,
        Some("yield") => 1,
        Some("spin") => 2,
        Some(d) => bail!("--set dispatch={d}: expected wait|yield|spin"),
    };
    let max_angular_velocity: f32 = args.take("max_ang_vel").map(|v| v.parse()).transpose().context("max_ang_vel")?.unwrap_or(1e4);
    let max_depenetration_velocity: f32 = args.take("max_depen_vel").map(|v| v.parse()).transpose().context("max_depen_vel")?.unwrap_or(0.0);
    let threads = args.cpu_threads();

    let mut s = PxSettings {
        gravity: scene.meta.gravity,
        dt: scene.meta.dt,
        threads: threads as u32,
        solver,
        pos_iters,
        vel_iters,
        broadphase,
        sleep: args.sleep as u32,
        dispatch_mode,
        max_angular_velocity,
        max_depenetration_velocity,
        gpu: gpu as u32,
        gpu_device: -1,
        ..Default::default()
    };

    let mut config = BTreeMap::new();
    // GPU buffer sizing from the scene: the large suite scenes have ~12 contacts per body;
    // allow 2x that, with PhysX's defaults as the floor.
    if gpu {
        let n = scene.bodies.len() as u64;
        let nj = scene.joints.len() as u64;
        let take_u32 = |args: &mut Args, key: &str, default: u64| -> Result<u32> {
            let v = match args.take(key) {
                Some(v) => v.parse::<u64>().with_context(|| key.to_string())?,
                None => default,
            };
            Ok(v.min(u32::MAX as u64) as u32)
        };
        let mb = 1024 * 1024;
        s.gpu_device = args.take("gpu_device").map(|v| v.parse()).transpose().context("gpu_device")?.unwrap_or(-1);
        s.gpu_max_partitions = take_u32(args, "gpu_partitions", 8)?;
        s.gpu_max_rigid_contacts = take_u32(args, "gpu_contacts", (24 * n).max(512 * 1024))?;
        s.gpu_max_rigid_patches = take_u32(args, "gpu_patches", (8 * n + 2 * nj).max(80 * 1024))?;
        s.gpu_found_lost_pairs = take_u32(args, "gpu_found_lost", (16 * n).max(256 * 1024))?;
        s.gpu_found_lost_aggregate_pairs = take_u32(args, "gpu_agg_found_lost", 1024)?;
        s.gpu_total_aggregate_pairs = take_u32(args, "gpu_agg_pairs", 1024)?;
        let heap_mb = take_u32(args, "gpu_heap_mb", (n * 1024 / mb).max(64))? as u64;
        s.gpu_heap_capacity = (heap_mb * mb).next_power_of_two().min(1 << 31) as u32;
        s.gpu_collision_stack = take_u32(args, "gpu_stack_mb", ((n * 512) / mb).max(64))?.saturating_mul(mb as u32);
        let temp_mb = take_u32(args, "gpu_temp_mb", ((n * 256) / mb).max(16))? as u64;
        s.gpu_temp_buffer = temp_mb * mb;
        config.insert("gpuMaxNumPartitions".into(), s.gpu_max_partitions.to_string());
        config.insert("gpuMaxRigidContactCount".into(), s.gpu_max_rigid_contacts.to_string());
        config.insert("gpuMaxRigidPatchCount".into(), s.gpu_max_rigid_patches.to_string());
        config.insert("gpuFoundLostPairsCapacity".into(), s.gpu_found_lost_pairs.to_string());
        config.insert("gpuFoundLostAggregatePairsCapacity".into(), s.gpu_found_lost_aggregate_pairs.to_string());
        config.insert("gpuTotalAggregatePairsCapacity".into(), s.gpu_total_aggregate_pairs.to_string());
        config.insert("gpuHeapCapacity".into(), s.gpu_heap_capacity.to_string());
        config.insert("gpuCollisionStackSize".into(), s.gpu_collision_stack.to_string());
        config.insert("gpuTempBufferCapacity".into(), s.gpu_temp_buffer.to_string());
    }

    let bodies: Vec<PxBody> = scene
        .bodies
        .iter()
        .map(|b| {
            let (shape, half, hull_first, hull_count) = match b.shape {
                Shape::Box(h) => (0, h, 0, 0),
                Shape::Sphere(r) => (1, [r, 0.0, 0.0], 0, 0),
                Shape::Hulls { first, count, radius } => (2, [radius, 0.0, 0.0], first, count),
                Shape::Plane => (3, [0.0; 3], 0, 0),
            };
            PxBody {
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
    let joints: Vec<PxJoint> = scene
        .joints
        .iter()
        .map(px_joint)
        .collect();

    // Hull table + flat vertex array (format v2)
    let mut hull_first = Vec::with_capacity(scene.hulls.len());
    let mut hull_count = Vec::with_capacity(scene.hulls.len());
    let mut vertices: Vec<f32> = Vec::new();
    for h in &scene.hulls {
        hull_first.push((vertices.len() / 3) as u32);
        hull_count.push(h.len() as u32);
        vertices.extend(h.iter().flatten());
    }
    let hulls = PxHulls {
        n_hulls: scene.hulls.len() as u32,
        first: hull_first.as_ptr(),
        count: hull_count.as_ptr(),
        n_vertices: (vertices.len() / 3) as u32,
        vertices: vertices.as_ptr(),
    };

    let mut info: PxInfo = unsafe { std::mem::zeroed() };
    let mut err = [0 as c_char; 1024];
    let world = unsafe {
        bench_px_create(
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
        bail!("PhysX world creation failed: {msg}");
    }
    if gpu {
        ensure!(info.gpu_active == 1, "PhysX did not enable GPU dynamics");
    }

    let solver_name = if info.solver == 1 { "PGS" } else { "TGS" };
    let f = |x: f32| format!("{x}");
    config.insert("source".into(), env!("BENCH_PHYSX_SOURCE").into());
    config.insert("solverType".into(), solver_name.into());
    config.insert("positionIterations".into(), pos_iters.to_string());
    config.insert("velocityIterations".into(), vel_iters.to_string());
    config.insert("broadPhase".into(), BROADPHASES.get(info.broadphase as usize).copied().unwrap_or("?").into());
    config.insert("pcm".into(), "true".into());
    config.insert("ccd".into(), "false".into());
    config.insert("stabilization".into(), "false".into());
    config.insert("enhancedDeterminism".into(), "false".into());
    config.insert("frictionType".into(), "patch (default)".into());
    config.insert("frictionCombine".into(), "average".into());
    config.insert("restitutionCombine".into(), "average".into());
    config.insert("staticFriction".into(), "= dynamic friction = scene friction".into());
    config.insert("contactOffset".into(), f(info.contact_offset));
    config.insert("restOffset".into(), f(info.rest_offset));
    let depen_default = if max_depenetration_velocity > 0.0 { "" } else { " (default)" };
    config.insert("maxDepenetrationVelocity".into(), format!("{:e}{depen_default}", info.max_depenetration_velocity));
    config.insert("maxAngularVelocity".into(), f(info.max_angular_velocity));
    config.insert("maxLinearVelocity".into(), format!("{:e} (default)", info.max_linear_velocity));
    config.insert("linearDamping".into(), "0".into());
    config.insert("angularDamping".into(), "0".into());
    config.insert("sleepThreshold".into(), f(info.sleep_threshold));
    config.insert("wakeCounterResetValue".into(), f(info.wake_counter_reset));
    config.insert("bounceThresholdVelocity".into(), f(info.bounce_threshold));
    config.insert("frictionOffsetThreshold".into(), f(info.friction_offset_threshold));
    config.insert("frictionCorrelationDistance".into(), f(info.friction_correlation_distance));
    config.insert("tolerancesScale".into(), format!("length {} speed {}", info.tolerance_length, info.tolerance_speed));
    config.insert("dispatcherWaitMode".into(), ["waitForWork", "yieldThread", "yieldProcessor"][dispatch_mode as usize].into());
    config.insert("joints".into(), if scene.joints.iter().any(|j| j.d6.is_some()) { "PxD6Joint (format v3 frames, motions, limits, drives; GPU-prepared), eCOLLISION_ENABLED off".into() } else { "PxSphericalJoint, local frames at the anchors, eCOLLISION_ENABLED off".into() });
    config.insert("step".into(), "simulate(dt) + fetchResults(true)".into());
    if info.hulls_cooked > 0 {
        let mut flags = String::from("eCOMPUTE_CONVEX");
        if info.cook_gpu_flag != 0 {
            flags += "+eGPU_COMPATIBLE";
        }
        if info.cook_build_gpu_data != 0 {
            flags += ", buildGPUData";
        }
        config.insert(
            "hullCooking".into(),
            format!(
                "PxCreateConvexMesh {flags}, vertex limit {}, polygon limit {}, PxCookingParams(physics tolerances) defaults otherwise; \
                 one PxConvexMeshGeometry shape per hull (scale 1, identity local pose), mass/inertia from the scene",
                info.cook_vertex_limit, info.cook_polygon_limit
            ),
        );
        config.insert(
            "hullCookingResult".into(),
            format!(
                "{} hulls: {} input vertices -> {} cooked ({} hulls changed vertex count: near-coplanar / non-extreme points dropped; {} hulls with a \
                 cooked vertex that is no input point), farthest input point outside its cooked hull {:.2e} m, {} polygons (max {} per hull){}",
                info.hulls_cooked,
                info.hull_vertices_in,
                info.hull_vertices_out,
                info.hulls_vertex_count_changed,
                info.hulls_new_vertices,
                info.hull_max_outside,
                info.hull_polygons,
                info.hull_max_polygons,
                if gpu { format!(", {} not GPU compatible (PhysX collides those on the CPU)", info.hulls_gpu_incompatible) } else { String::new() }
            ),
        );
    }
    if info.planes > 0 {
        config.insert("plane".into(), "PxRigidStatic at the scene pose, PxPlaneGeometry (local pose turns PhysX's +x normal onto body +y)".into());
    }

    let device = if gpu {
        unsafe { CStr::from_ptr(info.device.as_ptr()) }.to_string_lossy().to_string()
    } else {
        bench_core::machine::cpu_model()
    };
    Ok(PhysX {
        world,
        n: bodies.len(),
        dt: scene.meta.dt,
        threads,
        gpu,
        device,
        version: format!("{}.{}.{}", info.version_major, info.version_minor, info.version_bugfix),
        solver: solver_name,
        iterations: pos_iters,
        sleeping: args.sleep,
        config,
    })
}

impl Engine for PhysX {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "physx".into(),
            version: self.version.clone(),
            // CuMetal runs PhysX's CUDA GPU module on Apple Metal (sibling cuda-metal repo)
            backend: match (self.gpu, env!("BENCH_PHYSX_SOURCE").starts_with("CuMetal")) {
                (true, true) => "cuda→metal (CuMetal)",
                (true, false) => "cuda",
                (false, _) => "cpu",
            }
            .into(),
            device: self.device.clone(),
            threads: self.threads,
            pipelined: false,
            solver: self.solver.into(),
            iterations: self.iterations,
            iterations_meaning: "position iterations (velocity iterations in config)".into(),
            sleeping: self.sleeping,
            config: self.config.clone(),
        }
    }

    fn step(&mut self) -> Result<()> {
        let rc = unsafe { bench_px_step(self.world, self.dt) };
        ensure!(rc == 0, "PhysX simulate/fetchResults failed ({rc})");
        Ok(())
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        ensure!(out.len() == self.n, "read_state: {} samples for {} bodies", out.len(), self.n);
        let floats: &mut [f32] = bytemuck_cast(out);
        unsafe { bench_px_read_state(self.world, floats.as_mut_ptr()) };
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        let mut c = PxCounters::default();
        unsafe { bench_px_counters(self.world, &mut c) };
        Ok(json!({
            "awake": c.awake,
            "activeDynamicBodies": c.active_dynamic,
            "dynamicBodies": c.dynamic_bodies,
            "staticBodies": c.static_bodies,
            "activeConstraints": c.active_constraints,
            "axisSolverConstraints": c.axis_constraints,
            "contactPairs": c.contact_pairs,
            "contactPairsTouching": c.contact_pairs_touching,
            "contactPairsCacheHits": c.contact_pairs_cache_hits,
            "newPairs": c.new_pairs,
            "lostPairs": c.lost_pairs,
            "newTouches": c.new_touches,
            "lostTouches": c.lost_touches,
            "partitions": c.partitions,
            "physxErrors": c.errors,
            // PxSimulationStatistics pair/touch counts stay 0 under GPU dynamics (the GPU
            // pipeline does not publish them); the judge measures contacts from poses instead
            "pairStatsValid": !self.gpu,
        }))
    }
}

/// BodySample is #[repr(C)] with 13 f32 fields, matching the shim's output layout.
fn bytemuck_cast(s: &mut [BodySample]) -> &mut [f32] {
    const _: () = assert!(std::mem::size_of::<BodySample>() == 13 * 4);
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut f32, s.len() * 13) }
}

fn main() -> Result<()> {
    bench_core::main_with(build)
}
