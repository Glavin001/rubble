//! rubble adapter: the engine/ AVBD solver, either as Slang → SPIR-V → WGSL on native wgpu
//! (`--set backend=wgpu`, default; the path rubble ships to browsers) or on native Metal
//! (`--set backend=metal`, macOS; the only rubble path with joints today). Convex-hull
//! (Shardwright) scenes run on the native Metal fracture host (`src/fracture.rs`).
//!
//! Matched to the scene: mass/inertia, friction per body through rubble's material table
//! (combined as sqrt(a·b) on the GPU, as three-avbd), hard ball joints (stiffness unbounded),
//! the scene's AVBD α/γ/β, `--iterations` default = the scene's reference count, sleeping off
//! unless `--sleep on` (sleep timer pushed out of reach, as rubble's ring regression does).
//! Both backends read a 64-byte counter block back after every step, so `step()` is
//! synchronous: batch and sync timings measure the same thing.
//!
//! `--set` keys: backend=wgpu|metal, contacts=canonical|compact|persistent (wgpu storage
//! layout), reuse=on|off (three-avbd reuseContacts, default on as in three-avbd),
//! colors=N (colour capacity, default 32), contacts_per_body=N, candidates_per_body=N
//! (capacity sizing), joint_stiffness=K (default unbounded, as three-avbd's hard joints; rubble's
//! own demos use 1e5). ENGINE_* environment variables the engine reads are recorded.

use anyhow::{bail, ensure, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use engine_schema::{AvbdParams, BodyFlags, BodyState, Collider, Material, ShapeType, StepCounters, WorldCapacity, FLAG_REUSE_CONTACTS};
use serde_json::json;
use std::collections::BTreeMap;

enum Backend {
    Wgpu(engine_webgpu::WgpuSim),
    #[cfg(target_os = "macos")]
    Metal(Box<engine_metal::MetalWorld>),
}

struct Rubble {
    backend: Backend,
    params: AvbdParams,
    last: StepCounters,
    config: BTreeMap<String, String>,
    device: String,
    backend_name: &'static str,
    sleeping: bool,
}

/// Hard joint: the dual update caps the penalty at min(stiffness, kMaxN), i.e. at kMaxN.
const HARD: f32 = 1.0e30;

fn body_state(b: &bench_core::Body) -> BodyState {
    let inv = |v: f32| if v > 0.0 { 1.0 / v } else { 0.0 };
    BodyState {
        position: b.position,
        orientation: b.rotation,
        inv_mass: if b.is_static { 0.0 } else { inv(b.mass) },
        inv_inertia: if b.is_static { [0.0; 3] } else { b.inertia.map(inv) },
        flags: if b.is_static { BodyFlags::None as u32 } else { BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32 },
        linear_vel: if b.is_static { [0.0; 3] } else { b.linvel },
        angular_vel: if b.is_static { [0.0; 3] } else { b.angvel },
        initial_pos: b.position,
        initial_rot: b.rotation,
        ..bytemuck::Zeroable::zeroed()
    }
}

fn collider(id: u32, b: &bench_core::Body, material: u32) -> Collider {
    let (shape, params) = match b.shape {
        Shape::Sphere(r) => (ShapeType::Sphere, [r, 0.0, 0.0, r]),
        Shape::Box(h) => (ShapeType::Box, [h[0], h[1], h[2], (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt()]),
        // Plane collider: normal.xyz + offset (the suite's planes are horizontal at the body)
        Shape::Plane => (ShapeType::Plane, [0.0, 1.0, 0.0, b.position[1]]),
        Shape::Hulls { .. } => unreachable!("hull scenes take the fracture host (main)"),
    };
    Collider { body: id, shape: shape as u32, material, density: 1.0, local_rot: [0.0, 0.0, 0.0, 1.0], params, ..bytemuck::Zeroable::zeroed() }
}

fn round_up(n: u64, to: u64) -> u32 {
    (n.div_ceil(to) * to).min(u32::MAX as u64 / 2) as u32
}

fn build(scene: &Scene, args: &mut Args) -> Result<Rubble> {
    let backend_name = args.take("backend").unwrap_or_else(|| "wgpu".into());
    let layout = args.take("contacts").unwrap_or_else(|| "canonical".into());
    let reuse = match args.take("reuse").as_deref() {
        None | Some("on") => true,
        Some("off") => false,
        Some(v) => bail!("reuse={v}: expected on/off"),
    };
    let colors: u32 = args.take("colors").map(|v| v.parse()).transpose()?.unwrap_or(32);
    let contacts_per_body: u64 = args.take("contacts_per_body").map(|v| v.parse()).transpose()?.unwrap_or(12);
    let candidates_per_body: u64 = args.take("candidates_per_body").map(|v| v.parse()).transpose()?.unwrap_or(8);
    let joint_stiffness: f32 = args.take("joint_stiffness").map(|v| v.parse()).transpose()?.unwrap_or(HARD);

    let n = scene.bodies.len() as u64;
    let capacity = WorldCapacity {
        max_bodies: (n as u32 + 16).next_power_of_two().max(4096),
        max_colliders: (n as u32 + 16).next_power_of_two().max(4096),
        max_contacts: round_up(contacts_per_body * n, 1 << 20),
        max_candidates: round_up(candidates_per_body * n, 1 << 20),
        max_joints: (scene.joints.len() as u32 + 16).next_power_of_two().max(8192),
        max_colors: colors,
        ..WorldCapacity::small()
    };

    let mut params = AvbdParams::default();
    params.dt = scene.meta.dt;
    params.gravity_y = scene.meta.gravity[1];
    ensure!(scene.meta.gravity[0] == 0.0 && scene.meta.gravity[2] == 0.0, "rubble takes gravity along y only");
    params.iterations = args.iterations.unwrap_or(scene.meta.iterations);
    params.alpha = scene.meta.avbd.alpha;
    params.gamma = scene.meta.avbd.gamma;
    params.beta_lin = scene.meta.avbd.beta_lin;
    params.beta_ang = scene.meta.avbd.beta_ang;
    if reuse {
        params.flags |= FLAG_REUSE_CONTACTS;
    }
    if !args.sleep {
        params.sleep_time = 1.0e6; // never reached: every body stays in the solve
    }

    // One material per distinct friction coefficient
    let mut frictions: Vec<f32> = Vec::new();
    let material_of: Vec<u32> = scene
        .bodies
        .iter()
        .map(|b| match frictions.iter().position(|&f| f == b.friction) {
            Some(i) => i as u32,
            None => {
                frictions.push(b.friction);
                frictions.len() as u32 - 1
            }
        })
        .collect();
    ensure!(frictions.len() <= 64, "more than 64 distinct friction coefficients");
    let mut mats = vec![Material { friction: 0.5, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 }; 64];
    for (m, &f) in mats.iter_mut().zip(&frictions) {
        m.friction = f;
    }
    let bodies: Vec<BodyState> = scene.bodies.iter().map(body_state).collect();
    let colliders: Vec<Collider> = scene.bodies.iter().enumerate().map(|(i, b)| collider(i as u32, b, material_of[i])).collect();

    let mut config = BTreeMap::new();
    config.insert("reuseContacts".into(), reuse.to_string());
    if !scene.joints.is_empty() {
        config.insert("jointStiffness".into(), if joint_stiffness >= HARD { "hard (capped at kMaxN)".into() } else { joint_stiffness.to_string() });
    }
    config.insert("capacity".into(), format!("bodies {} contacts {} candidates {} colours {}", capacity.max_bodies, capacity.max_contacts, capacity.max_candidates, capacity.max_colors));
    config.insert("alpha/gamma/betaLin/betaAng".into(), format!("{}/{}/{}/{}", params.alpha, params.gamma, params.beta_lin, params.beta_ang));
    for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("ENGINE_")) {
        config.insert(format!("env.{k}"), v);
    }

    let (backend, device, backend_name) = match backend_name.as_str() {
        "wgpu" => {
            let mut sim = match layout.as_str() {
                "canonical" => engine_webgpu::WgpuSim::new_with_compact_contacts(capacity, false)?,
                "compact" => engine_webgpu::WgpuSim::new_with_compact_contacts(capacity, true)?,
                "persistent" => engine_webgpu::WgpuSim::new_with_persistent_contacts(capacity)?,
                other => bail!("contacts={other}: expected canonical, compact or persistent"),
            };
            config.insert("contacts".into(), layout.clone());
            sim.set_params(params);
            sim.set_materials(&mats);
            sim.add_bodies(&bodies);
            sim.add_colliders(&colliders);
            for d in joint_descs(&scene, joint_stiffness) {
                sim.add_joint_desc(&d);
            }
            let device = sim.chip().to_string();
            (Backend::Wgpu(sim), device, "wgpu")
        }
        #[cfg(target_os = "macos")]
        "metal" => {
            use engine_exec::PhysicsBackend;
            let mut world = Box::new(engine_metal::MetalWorld::new(capacity)?);
            world.set_params(params);
            let ids: Vec<u32> = frictions.iter().map(|&f| world.add_material(Material { friction: f, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 })).collect();
            for (i, b) in bodies.iter().enumerate() {
                let id = world.add_body(*b);
                let mut c = colliders[i];
                c.body = id;
                c.material = ids[material_of[i] as usize];
                world.add_collider(c);
            }
            for d in joint_descs(&scene, joint_stiffness) {
                world.add_joint_desc(&d);
            }
            let device = world.chip.clone();
            (Backend::Metal(world), device, "metal")
        }
        other => bail!("backend={other}: expected wgpu or metal (metal needs macOS)"),
    };
    Ok(Rubble { backend, params, last: StepCounters::default(), config, device, backend_name, sleeping: args.sleep })
}

/// v1/v2 joints are hard ball joints (three-avbd `Joint` with stiffnessAng 0); v3 joints carry
/// their D6 frames, motions, limits and drive, mapped one to one onto rubble's `JointDesc`.
fn joint_descs(scene: &Scene, stiffness: f32) -> Vec<engine_schema::JointDesc> {
    use bench_core::scene::Motion as M;
    use engine_schema::{Dof, JointDesc, JointDrive, JointFrame, Motion};
    use glam::{Quat, Vec3};
    scene
        .joints
        .iter()
        .map(|j| {
            let (a, b) = (Vec3::from_array(j.anchor_a), Vec3::from_array(j.anchor_b));
            let Some(d6) = j.d6 else {
                let d = JointDesc::ball(j.body_a, j.body_b, a, b).with_collision(j.collide);
                return if stiffness >= HARD { d } else { d.with_stiffness(stiffness, 0.0) };
            };
            let fa = JointFrame { anchor: a, rot: Quat::from_array(d6.frame_a) };
            let fb = JointFrame { anchor: b, rot: Quat::from_array(d6.frame_b) };
            let mut d = JointDesc::d6(j.body_a, j.body_b, fa, fb).with_collision(j.collide);
            d.motion = d6.motion.map(|m| match m {
                M::Locked => Motion::Locked,
                M::Limited => Motion::Limited,
                M::Free => Motion::Free,
            });
            d.twist = (d6.twist[0], d6.twist[1]);
            d.swing = (d6.swing[0], d6.swing[1]);
            d.cone = d6.cone;
            d.lin_lo.x = d6.linear_x[0];
            d.lin_hi.x = d6.linear_x[1];
            if let Some(dr) = d6.drive {
                let dof = [Dof::X, Dof::Y, Dof::Z, Dof::Twist][dr.dof.min(3) as usize];
                d.drive = Some(JointDrive {
                    dof,
                    target: dr.target,
                    velocity: dr.velocity,
                    stiffness: dr.stiffness,
                    damping: dr.damping,
                    force_limit: dr.force_limit,
                });
            }
            d
        })
        .collect()
}

fn rubble_revision() -> String {
    // branch@commit, plus the dirty files and a hash of their diff so a run can be reproduced
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../engine");
    let git = |a: &[&str]| std::process::Command::new("git").arg("-C").arg(dir).args(a).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    let (branch, rev) = (git(&["rev-parse", "--abbrev-ref", "HEAD"]), git(&["rev-parse", "--short=12", "HEAD"]));
    let dirty = git(&["status", "--porcelain", "--", "."]);
    if dirty.is_empty() {
        return format!("{branch}@{rev}");
    }
    let diff = std::process::Command::new("git").arg("-C").arg(dir).args(["diff", "HEAD", "--", "."]).output().map(|o| o.stdout).unwrap_or_default();
    let h = diff.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100_0000_01b3));
    format!("{branch}@{rev}+dirty(diff fnv {h:016x}; {})", dirty.lines().map(|l| l[3..].to_string()).collect::<Vec<_>>().join(", "))
}

impl Engine for Rubble {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "rubble".into(),
            version: rubble_revision(),
            backend: self.backend_name.into(),
            device: self.device.clone(),
            threads: 1,
            pipelined: false,
            solver: "AVBD".into(),
            iterations: self.params.iterations,
            iterations_meaning: "AVBD iterations (primal+dual) per step".into(),
            sleeping: self.sleeping,
            config: self.config.clone(),
        }
    }

    fn step(&mut self) -> Result<()> {
        let r = match &mut self.backend {
            Backend::Wgpu(sim) => sim.step(),
            #[cfg(target_os = "macos")]
            Backend::Metal(world) => world.step_gpu(),
        };
        match r {
            Ok(c) => {
                self.last = c;
                Ok(())
            }
            Err(e) => bail!("rubble step failed: {e:?} (overflow flags 0x{:x})", self.last.overflow_flags),
        }
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        let bodies = match &mut self.backend {
            Backend::Wgpu(sim) => sim.download_bodies(),
            #[cfg(target_os = "macos")]
            Backend::Metal(world) => world.download_bodies(),
        };
        ensure!(bodies.len() >= out.len(), "engine returned {} bodies", bodies.len());
        for (s, b) in out.iter_mut().zip(&bodies) {
            *s = BodySample { position: b.position, rotation: b.orientation, linvel: b.linear_vel, angvel: b.angular_vel };
        }
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        let c = self.last;
        Ok(json!({
            "awake": c.awake_count,
            "contacts": c.contact_count,
            "manifolds": c.manifold_count,
            "candidates": c.candidate_count,
            "colors": c.color_count,
            "overflow": c.overflow_flags,
        }))
    }
}

#[cfg(all(target_os = "macos", feature = "fracture"))]
mod fracture;

/// Box/sphere scenes run on `Rubble` (wgpu or Metal); convex-hull scenes on the Metal
/// fracture host.
enum Any {
    Rigid(Rubble),
    #[cfg(all(target_os = "macos", feature = "fracture"))]
    Fracture(fracture::Fracture),
}

impl Engine for Any {
    fn info(&self) -> EngineInfo {
        match self {
            Any::Rigid(e) => e.info(),
            #[cfg(all(target_os = "macos", feature = "fracture"))]
            Any::Fracture(e) => e.info(),
        }
    }
    fn step(&mut self) -> Result<()> {
        match self {
            Any::Rigid(e) => e.step(),
            #[cfg(all(target_os = "macos", feature = "fracture"))]
            Any::Fracture(e) => e.step(),
        }
    }
    fn drain(&mut self) -> Result<()> {
        match self {
            Any::Rigid(e) => e.drain(),
            #[cfg(all(target_os = "macos", feature = "fracture"))]
            Any::Fracture(e) => e.drain(),
        }
    }
    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        match self {
            Any::Rigid(e) => e.read_state(out),
            #[cfg(all(target_os = "macos", feature = "fracture"))]
            Any::Fracture(e) => e.read_state(out),
        }
    }
    fn counters(&mut self) -> Result<serde_json::Value> {
        match self {
            Any::Rigid(e) => e.counters(),
            #[cfg(all(target_os = "macos", feature = "fracture"))]
            Any::Fracture(e) => e.counters(),
        }
    }
}

fn main() -> Result<()> {
    bench_core::main_with(|scene, args| {
        if scene.meta.counts.hulls == 0 {
            return build(scene, args).map(Any::Rigid);
        }
        let backend = args.take("backend").unwrap_or_else(|| "metal".into());
        ensure!(backend == "metal", "convex-hull scenes need --set backend=metal: rubble's fracture kernels have no native wgpu host yet (they run in the browser demo)");
        #[cfg(all(target_os = "macos", feature = "fracture"))]
        return fracture::build(scene, args, rubble_revision()).map(Any::Fracture);
        #[cfg(not(all(target_os = "macos", feature = "fracture")))]
        bail!("convex-hull scenes need macOS and the `fracture` feature (engine-viewer's Metal fracture host)")
    })
}
