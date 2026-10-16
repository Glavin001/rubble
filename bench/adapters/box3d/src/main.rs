//! Box3D adapter (Erin Catto's 3D engine; C17, SIMD, soft-step solver). CPU only. A small C
//! shim (`src/box3d_shim.c`) builds and steps the world; this file drives it over FFI.
//!
//! Matched to the scene (README "Fairness"): exact mass and principal inertia from the scene
//! (`b3Body_SetMassData`, shapes don't touch mass), friction per body combined by Box3D's
//! default sqrt(a·b) (the same rule as three-avbd and rubble), restitution 0, no damping, no
//! per-step rotation clamp (`allowFastRotation`), CCD off, sleeping off unless `--sleep on`,
//! ball joints = `b3SphericalJoint` with joined bodies not colliding. Boxes are box hulls,
//! convex compounds one hull shape per hull (`b3CreateHull` from the scene's points), and the
//! ground plane, which Box3D lacks, a static box whose top face is the plane.
//! Like Box3D's own benchmark, it uses its internal scheduler with `workerCount` = `--threads`
//! and 4 substeps; `--iterations` is the substep count.
//!
//! `--set` keys: contact_hertz=X, contact_damping=X, contact_speed=X (Box3D's soft contact
//! tuning, defaults 30 Hz / 10 / 3 m/s), recycling=on|off (contact recycling, default on),
//! max_speed=X (maximum linear speed, default 1e4 m/s), ccd=on|off.

use anyhow::{bail, ensure, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr};

#[repr(C)]
#[derive(Default)]
struct BxBody {
    shape: u32,
    is_static: u32,
    half: [f32; 3],
    mass: f32,
    inertia: [f32; 3],
    friction: f32,
    pos: [f32; 3],
    quat: [f32; 4],
    linvel: [f32; 3],
    angvel: [f32; 3],
    hull_first: u32,
    hull_count: u32,
}

#[repr(C)]
struct BxJoint {
    a: u32,
    b: u32,
    collide: u32,
    pad: u32,
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
}

#[repr(C)]
struct BxSettings {
    dt: f32,
    gravity: [f32; 3],
    substeps: i32,
    workers: i32,
    sleep: i32,
    continuous: i32,
    recycling: i32,
    contact_hertz: f32,
    contact_damping: f32,
    contact_speed: f32,
    max_linear_speed: f32,
    plane_half_extent: f32,
}

#[repr(C)]
struct BxReport {
    hulls_in: i32,
    hulls_changed: i32,
    vertices_in: i32,
    vertices_out: i32,
    failed: i32,
    error: [c_char; 256],
}

#[repr(C)]
#[derive(Default)]
struct BxCounters {
    awake: i32,
    contacts: i32,
    awake_contacts: i32,
    recycled_contacts: i32,
    islands: i32,
    joints: i32,
    bodies: i32,
    tasks: i32,
}

extern "C" {
    fn bx_create(s: *const BxSettings, bodies: *const BxBody, nb: i32, hull_table: *const u32, nh: i32, verts: *const f32, joints: *const BxJoint, nj: i32, report: *mut BxReport) -> *mut c_void;
    fn bx_step(w: *mut c_void);
    fn bx_read(w: *mut c_void, out: *mut f32);
    fn bx_counters(w: *mut c_void, c: *mut BxCounters);
    fn bx_destroy(w: *mut c_void);
}

struct Box3d {
    world: *mut c_void,
    threads: usize,
    substeps: u32,
    sleeping: bool,
    config: BTreeMap<String, String>,
}

impl Drop for Box3d {
    fn drop(&mut self) {
        unsafe { bx_destroy(self.world) }
    }
}

fn on_off(v: &str) -> Result<bool> {
    match v {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        _ => bail!("expected on/off, got {v}"),
    }
}

fn build(scene: &Scene, args: &mut Args) -> Result<Box3d> {
    let f = |args: &mut Args, k: &str| args.take(k).map(|v| v.parse::<f32>()).transpose();
    let contact_hertz = f(args, "contact_hertz")?;
    let contact_damping = f(args, "contact_damping")?;
    let contact_speed = f(args, "contact_speed")?;
    let max_speed = f(args, "max_speed")?.unwrap_or(1.0e4);
    let recycling = args.take("recycling").map(|v| on_off(&v)).transpose()?.unwrap_or(true);
    let ccd = args.take("ccd").map(|v| on_off(&v)).transpose()?.unwrap_or(false);
    let threads = args.cpu_threads();
    let substeps = args.iterations.unwrap_or(4);
    let extent = scene.bodies.iter().filter(|b| !b.is_static).flat_map(|b| b.position.iter().map(|x| x.abs())).fold(0.0f32, f32::max);

    let bodies: Vec<BxBody> = scene
        .bodies
        .iter()
        .map(|b| {
            let (shape, half, hull_first, hull_count) = match b.shape {
                Shape::Box(h) => (0, h, 0, 0),
                Shape::Sphere(r) => (1, [r, r, r], 0, 0),
                Shape::Hulls { first, count, .. } => (2, [0.0; 3], first, count),
                Shape::Plane => (3, [0.0; 3], 0, 0),
            };
            BxBody { shape, is_static: b.is_static as u32, half, mass: b.mass, inertia: b.inertia, friction: b.friction, pos: b.position, quat: b.rotation, linvel: b.linvel, angvel: b.angvel, hull_first, hull_count }
        })
        .collect();
    let mut table = Vec::with_capacity(scene.hulls.len() * 2);
    let mut verts = Vec::new();
    for h in &scene.hulls {
        table.extend([(verts.len() / 3) as u32, h.len() as u32]);
        verts.extend(h.iter().flatten());
    }
    let d6 = scene.joints.iter().filter(|j| j.kind != bench_core::scene::JointKind::Ball).count();
    ensure!(d6 == 0, "unsupported: {d6} D6 joints (the Box3D adapter maps ball joints only; Box3D's revolute/prismatic/motor joints are not wired up yet)");
    let joints: Vec<BxJoint> = scene.joints.iter().map(|j| BxJoint { a: j.body_a, b: j.body_b, collide: j.collide as u32, pad: 0, anchor_a: j.anchor_a, anchor_b: j.anchor_b }).collect();
    let settings = BxSettings {
        dt: scene.meta.dt,
        gravity: scene.meta.gravity,
        substeps: substeps as i32,
        workers: threads as i32,
        sleep: args.sleep as i32,
        continuous: ccd as i32,
        recycling: recycling as i32,
        contact_hertz: contact_hertz.unwrap_or(-1.0),
        contact_damping: contact_damping.unwrap_or(-1.0),
        contact_speed: contact_speed.unwrap_or(-1.0),
        max_linear_speed: max_speed,
        plane_half_extent: (4.0 * extent).max(200.0),
    };
    let mut report = BxReport { hulls_in: 0, hulls_changed: 0, vertices_in: 0, vertices_out: 0, failed: 0, error: [0; 256] };
    let world = unsafe { bx_create(&settings, bodies.as_ptr(), bodies.len() as i32, table.as_ptr(), scene.hulls.len() as i32, verts.as_ptr(), joints.as_ptr(), joints.len() as i32, &mut report) };
    ensure!(!world.is_null(), "Box3D world creation failed");
    if report.failed > 0 {
        let msg = unsafe { CStr::from_ptr(report.error.as_ptr()) }.to_string_lossy().to_string();
        unsafe { bx_destroy(world) };
        bail!("{} hulls could not be built: {msg}", report.failed);
    }
    let opt = |v: Option<f32>, d: &str| v.map(|x| x.to_string()).unwrap_or_else(|| format!("{d} (default)"));
    let mut config = BTreeMap::new();
    config.insert("substeps".into(), substeps.to_string());
    config.insert("contactHertz".into(), opt(contact_hertz, "30"));
    config.insert("contactDampingRatio".into(), opt(contact_damping, "10"));
    config.insert("contactSpeed".into(), opt(contact_speed, "3 m/s"));
    config.insert("maximumLinearSpeed".into(), max_speed.to_string());
    config.insert("contactRecycling".into(), recycling.to_string());
    config.insert("ccd".into(), ccd.to_string());
    config.insert("allowFastRotation".into(), "true".into());
    config.insert("frictionCombine".into(), "sqrt(a·b) (Box3D default)".into());
    config.insert("scheduler".into(), "Box3D internal (workerCount = threads)".into());
    config.insert("build".into(), "CMake Release, BOX3D_VALIDATE=OFF, BOX3D_PROFILE=OFF".into());
    if scene.bodies.iter().any(|b| matches!(b.shape, Shape::Plane)) {
        config.insert("groundPlane".into(), format!("static box {0}×2×{0} m, top face on the plane (Box3D has no plane shape)", 2.0 * settings.plane_half_extent));
    }
    if report.hulls_in > 0 {
        config.insert("hullCooking".into(), format!("b3CreateHull(points, max {}): {} hulls, {} changed vertex count, vertices {} → {}", 128, report.hulls_in, report.hulls_changed, report.vertices_in, report.vertices_out));
    }
    Ok(Box3d { world, threads, substeps, sleeping: args.sleep, config })
}

impl Engine for Box3d {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "box3d".into(),
            version: format!("0.1.0 ({})", env!("BENCH_BOX3D_REV")),
            backend: "cpu".into(),
            device: bench_core::machine::cpu_model(),
            threads: self.threads,
            pipelined: false,
            solver: "soft step (substepped soft constraints, relax + restitution)".into(),
            iterations: self.substeps,
            iterations_meaning: "substeps per step".into(),
            sleeping: self.sleeping,
            config: self.config.clone(),
        }
    }

    fn step(&mut self) -> Result<()> {
        unsafe { bx_step(self.world) };
        Ok(())
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        const _: () = assert!(std::mem::size_of::<BodySample>() == 13 * 4);
        unsafe { bx_read(self.world, out.as_mut_ptr() as *mut f32) };
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        let mut c = BxCounters::default();
        unsafe { bx_counters(self.world, &mut c) };
        Ok(json!({
            "awake": c.awake, "contacts": c.contacts, "awakeContacts": c.awake_contacts,
            "recycledContacts": c.recycled_contacts, "islands": c.islands, "joints": c.joints, "tasks": c.tasks,
        }))
    }
}

fn main() -> Result<()> {
    bench_core::main_with(build)
}
