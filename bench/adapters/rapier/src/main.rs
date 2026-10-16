//! rapier3d adapter. CPU, f32, `parallel` feature (rayon); `--threads N` sizes its pool.
//!
//! Matched to the scene (see README "Fairness"): mass and principal inertia set exactly from
//! the scene file (on the body for convex-hull compounds, one `convex_hull` collider per hull;
//! the ground plane is a half-space), friction from the scene combined with `Average` (three-avbd uses sqrt(a·b);
//! the suite's coefficients are 0.5/0.6, so they differ by <0.5%), restitution 0, no damping,
//! ball joints with contacts between joined bodies disabled, CCD off (no other engine here
//! runs CCD), sleeping off unless `--sleep on`. Everything else is rapier's default; its
//! `num_solver_iterations` (substeps of its soft-TGS solver, default 4) is `--iterations`.
//!
//! `--set` keys: pgs=N (num_internal_pgs_iterations), stabilization=N
//! (num_internal_stabilization_iterations), recycling=on|off (contact_recycling),
//! warmstart_joints=on|off, ccd=on|off.

use anyhow::{bail, Result};
use bench_core::{Args, BodySample, Engine, EngineInfo, Scene, Shape};
use rapier3d::prelude::*;
use serde_json::json;
use std::collections::BTreeMap;

struct Rapier {
    world: PhysicsWorld,
    handles: Vec<RigidBodyHandle>,
    pool: rayon::ThreadPool,
    threads: usize,
    sleeping: bool,
    config: BTreeMap<String, String>,
}

fn on_off(v: &str) -> Result<bool> {
    match v {
        "on" | "true" | "1" => Ok(true),
        "off" | "false" | "0" => Ok(false),
        _ => bail!("expected on/off, got {v}"),
    }
}

fn build(scene: &Scene, args: &mut Args) -> Result<Rapier> {
    let mut world = PhysicsWorld::new();
    let g = scene.meta.gravity;
    world.gravity = Vector::new(g[0], g[1], g[2]);
    let ip = &mut world.integration_parameters;
    ip.dt = scene.meta.dt;
    if let Some(n) = args.iterations {
        ip.num_solver_iterations = n as usize;
    }
    let mut config = BTreeMap::new();
    if let Some(v) = args.take("pgs") {
        ip.num_internal_pgs_iterations = v.parse()?;
    }
    if let Some(v) = args.take("stabilization") {
        ip.num_internal_stabilization_iterations = v.parse()?;
    }
    if let Some(v) = args.take("recycling") {
        ip.contact_recycling = on_off(&v)?;
    }
    if let Some(v) = args.take("warmstart_joints") {
        ip.warmstart_joints = on_off(&v)?;
    }
    let ccd = args.take("ccd").map(|v| on_off(&v)).transpose()?.unwrap_or(false);
    if !ccd {
        ip.max_ccd_substeps = 0;
    }
    config.insert("pgs".into(), ip.num_internal_pgs_iterations.to_string());
    config.insert("stabilization".into(), ip.num_internal_stabilization_iterations.to_string());
    config.insert("recycling".into(), ip.contact_recycling.to_string());
    config.insert("warmstartJoints".into(), ip.warmstart_joints.to_string());
    config.insert("ccd".into(), ccd.to_string());
    config.insert("frictionCombine".into(), "average".into());

    let mut handles = Vec::with_capacity(scene.bodies.len());
    for b in &scene.bodies {
        let pose = Pose::from_parts(Vector::from_array(b.position), Rotation::from_xyzw(b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]));
        let rb = if b.is_static {
            RigidBodyBuilder::fixed().pose(pose)
        } else {
            RigidBodyBuilder::dynamic()
                .pose(pose)
                .linvel(Vector::from_array(b.linvel))
                .angvel(Vector::from_array(b.angvel))
                .linear_damping(0.0)
                .angular_damping(0.0)
                .can_sleep(args.sleep)
        };
        let mass = MassProperties::new(Vector::ZERO, b.mass, Vector::from_array(b.inertia));
        // A compound's pieces carry no mass; the body gets the scene's exact mass properties
        let rb = match b.shape {
            Shape::Hulls { .. } if !b.is_static => rb.additional_mass_properties(mass),
            _ => rb,
        };
        let h = world.bodies.insert(rb);
        let shapes: Vec<ColliderBuilder> = match b.shape {
            Shape::Box(h) => vec![ColliderBuilder::cuboid(h[0], h[1], h[2])],
            Shape::Sphere(r) => vec![ColliderBuilder::ball(r)],
            Shape::Plane => vec![ColliderBuilder::halfspace(rapier3d::na::Unit::new_unchecked(Vector::Y))],
            Shape::Hulls { first, count, .. } => (first..first + count)
                .map(|i| {
                    let pts: Vec<Vector> = scene.hulls[i as usize].iter().map(|v| Vector::from_array(*v)).collect();
                    ColliderBuilder::convex_hull(&pts).ok_or_else(|| anyhow::anyhow!("hull {i}: rapier could not build a convex hull"))
                })
                .collect::<Result<_>>()?,
        };
        for shape in shapes {
            let col = shape
                .friction(b.friction)
                .friction_combine_rule(CoefficientCombineRule::Average)
                .restitution(b.restitution)
                .restitution_combine_rule(CoefficientCombineRule::Average);
            let col = match b.shape {
                Shape::Box(_) | Shape::Sphere(_) if !b.is_static => col.mass_properties(mass),
                _ => col.density(0.0),
            };
            world.colliders.insert_with_parent(col, h, &mut world.bodies);
        }
        handles.push(h);
    }
    for j in &scene.joints {
        let joint = SphericalJointBuilder::new()
            .local_anchor1(Vector::from_array(j.anchor_a))
            .local_anchor2(Vector::from_array(j.anchor_b))
            .contacts_enabled(j.collide);
        world.impulse_joints.insert(handles[j.body_a as usize], handles[j.body_b as usize], joint, true);
    }
    let threads = args.cpu_threads();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    Ok(Rapier { world, handles, pool, threads, sleeping: args.sleep, config })
}

impl Engine for Rapier {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            engine: "rapier".into(),
            version: "0.36.1".into(),
            backend: "cpu".into(),
            device: bench_core::machine::cpu_model(),
            threads: self.threads,
            pipelined: false,
            solver: "soft-TGS (substepped PGS)".into(),
            iterations: self.world.integration_parameters.num_solver_iterations as u32,
            iterations_meaning: "num_solver_iterations (substeps)".into(),
            sleeping: self.sleeping,
            config: self.config.clone(),
        }
    }

    fn step(&mut self) -> Result<()> {
        let world = &mut self.world;
        self.pool.install(|| world.step());
        Ok(())
    }

    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()> {
        for (s, h) in out.iter_mut().zip(&self.handles) {
            let b = &self.world.bodies[*h];
            let q = b.rotation();
            *s = BodySample {
                position: b.translation().to_array(),
                rotation: [q.x, q.y, q.z, q.w],
                linvel: b.linvel().to_array(),
                angvel: b.angvel().to_array(),
            };
        }
        Ok(())
    }

    fn counters(&mut self) -> Result<serde_json::Value> {
        Ok(json!({ "awake": self.world.islands.num_active_bodies() }))
    }
}

fn main() -> Result<()> {
    bench_core::main_with(build)
}
