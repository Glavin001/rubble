#![allow(dead_code)]

use glam::{Mat3, Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use rubble_math::BodyHandle;
use std::fmt;

pub const RUN_KNOWN_FAILURES_ENV: &str = "RUBBLE_RUN_KNOWN_FAILURES";

pub fn run_known_failures() -> bool {
    std::env::var_os(RUN_KNOWN_FAILURES_ENV).is_some()
}

pub fn should_skip_known_failure(name: &str, reason: &str) -> bool {
    if run_known_failures() {
        return false;
    }
    eprintln!(
        "SKIP known failure `{name}`: {reason}. Set {RUN_KNOWN_FAILURES_ENV}=1 to run this scenario."
    );
    true
}

pub fn try_world(config: SimConfig) -> Option<World> {
    World::new(config).ok()
}

pub fn step_n(world: &mut World, steps: usize) {
    for _ in 0..steps {
        world.step();
    }
}

#[derive(Debug, Clone)]
pub struct TrackedBody3D {
    pub label: &'static str,
    pub handle: BodyHandle,
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc,
}

impl TrackedBody3D {
    pub fn is_dynamic(&self) -> bool {
        self.mass > 0.0
    }
}

pub fn add_tracked_body(
    world: &mut World,
    label: &'static str,
    desc: RigidBodyDesc,
) -> TrackedBody3D {
    let handle = world.add_body(&desc);
    TrackedBody3D {
        label,
        handle,
        mass: desc.mass,
        friction: desc.friction,
        shape: desc.shape,
    }
}

#[derive(Debug, Clone)]
pub struct BodySnapshot3D {
    pub label: &'static str,
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc,
    pub position: Vec3,
    pub rotation: Quat,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

pub fn snapshot_body(world: &World, body: &TrackedBody3D) -> BodySnapshot3D {
    BodySnapshot3D {
        label: body.label,
        mass: body.mass,
        friction: body.friction,
        shape: body.shape.clone(),
        position: world
            .get_position(body.handle)
            .expect("tracked 3D body position should exist"),
        rotation: world
            .get_rotation(body.handle)
            .expect("tracked 3D body rotation should exist"),
        linear_velocity: world
            .get_velocity(body.handle)
            .expect("tracked 3D body velocity should exist"),
        angular_velocity: world
            .get_angular_velocity(body.handle)
            .expect("tracked 3D body angular velocity should exist"),
    }
}

pub fn snapshot_all(world: &World, bodies: &[TrackedBody3D]) -> Vec<BodySnapshot3D> {
    bodies
        .iter()
        .map(|body| snapshot_body(world, body))
        .collect()
}

#[derive(Debug, Clone)]
pub struct SystemMetrics3D {
    pub total_mass: f32,
    pub center_of_mass: Vec3,
    pub linear_momentum: Vec3,
    pub angular_momentum: Vec3,
    pub translational_ke: f32,
    pub rotational_ke: f32,
    pub potential_energy: f32,
    pub total_energy: f32,
    pub max_speed: f32,
    pub max_angular_speed: f32,
    pub min_height: f32,
}

pub fn compute_metrics(snapshots: &[BodySnapshot3D], gravity: Vec3) -> SystemMetrics3D {
    let dynamic: Vec<&BodySnapshot3D> = snapshots.iter().filter(|body| body.mass > 0.0).collect();

    let total_mass = dynamic.iter().map(|body| body.mass).sum::<f32>();
    let center_of_mass = if total_mass > 0.0 {
        dynamic
            .iter()
            .fold(Vec3::ZERO, |acc, body| acc + body.position * body.mass)
            / total_mass
    } else {
        Vec3::ZERO
    };

    let gravity_dir = if gravity.length_squared() > 1e-12 {
        -gravity.normalize()
    } else {
        Vec3::Y
    };

    let mut linear_momentum = Vec3::ZERO;
    let mut angular_momentum = Vec3::ZERO;
    let mut translational_ke: f32 = 0.0;
    let mut rotational_ke: f32 = 0.0;
    let mut potential_energy: f32 = 0.0;
    let mut max_speed: f32 = 0.0;
    let mut max_angular_speed: f32 = 0.0;
    let mut min_height = f32::INFINITY;

    for body in dynamic {
        let momentum = body.linear_velocity * body.mass;
        linear_momentum += momentum;
        translational_ke += 0.5 * body.mass * body.linear_velocity.length_squared();

        let inertia_world = world_inertia_tensor(&body.shape, body.mass, body.rotation);
        let spin_momentum = inertia_world * body.angular_velocity;
        angular_momentum += (body.position - center_of_mass).cross(momentum) + spin_momentum;
        rotational_ke += 0.5 * body.angular_velocity.dot(spin_momentum);

        potential_energy += body.mass * (-gravity).dot(body.position);
        max_speed = max_speed.max(body.linear_velocity.length());
        max_angular_speed = max_angular_speed.max(body.angular_velocity.length());
        min_height = min_height.min(body.position.dot(gravity_dir));
    }

    if !min_height.is_finite() {
        min_height = 0.0;
    }

    SystemMetrics3D {
        total_mass,
        center_of_mass,
        linear_momentum,
        angular_momentum,
        translational_ke,
        rotational_ke,
        potential_energy,
        total_energy: translational_ke + rotational_ke + potential_energy,
        max_speed,
        max_angular_speed,
        min_height,
    }
}

#[derive(Debug, Clone)]
pub struct SceneReport3D {
    pub step: usize,
    pub metrics: SystemMetrics3D,
    pub bodies: Vec<BodySnapshot3D>,
}

impl fmt::Display for SceneReport3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "step={} mass={:.4} com={:?} p={:?} L={:?} ke_t={:.6} ke_r={:.6} pe={:.6} total={:.6} max_v={:.6} max_w={:.6} min_h={:.6}",
            self.step,
            self.metrics.total_mass,
            self.metrics.center_of_mass,
            self.metrics.linear_momentum,
            self.metrics.angular_momentum,
            self.metrics.translational_ke,
            self.metrics.rotational_ke,
            self.metrics.potential_energy,
            self.metrics.total_energy,
            self.metrics.max_speed,
            self.metrics.max_angular_speed,
            self.metrics.min_height
        )?;
        for body in &self.bodies {
            writeln!(
                f,
                "  {} pos={:?} vel={:?} rot={:?} omega={:?} mass={:.3} friction={:.3}",
                body.label,
                body.position,
                body.linear_velocity,
                body.rotation,
                body.angular_velocity,
                body.mass,
                body.friction
            )?;
        }
        Ok(())
    }
}

pub fn scene_report(
    world: &World,
    bodies: &[TrackedBody3D],
    gravity: Vec3,
    step: usize,
) -> SceneReport3D {
    let body_snapshots = snapshot_all(world, bodies);
    let metrics = compute_metrics(&body_snapshots, gravity);
    SceneReport3D {
        step,
        metrics,
        bodies: body_snapshots,
    }
}

pub fn collect_reports(
    world: &mut World,
    bodies: &[TrackedBody3D],
    gravity: Vec3,
    steps: usize,
) -> Vec<SceneReport3D> {
    let mut reports = Vec::with_capacity(steps);
    for step in 0..steps {
        world.step();
        reports.push(scene_report(world, bodies, gravity, step + 1));
    }
    reports
}

pub fn cube_hull(extents: Vec3) -> Vec<Vec3> {
    let hx = extents.x;
    let hy = extents.y;
    let hz = extents.z;
    vec![
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ]
}

pub fn octagon_hull(radius: f32, half_height: f32) -> Vec<Vec3> {
    let mut vertices = Vec::new();
    for &y in &[-half_height, half_height] {
        for i in 0..8 {
            let angle = i as f32 * std::f32::consts::TAU / 8.0;
            vertices.push(Vec3::new(radius * angle.cos(), y, radius * angle.sin()));
        }
    }
    vertices
}

fn world_inertia_tensor(shape: &ShapeDesc, mass: f32, rotation: Quat) -> Mat3 {
    if mass <= 0.0 {
        return Mat3::ZERO;
    }

    let local = Mat3::from_diagonal(local_inertia_diagonal(shape, mass));
    let rot = Mat3::from_quat(rotation);
    rot * local * rot.transpose()
}

fn local_inertia_diagonal(shape: &ShapeDesc, mass: f32) -> Vec3 {
    match shape {
        ShapeDesc::Sphere { radius } => {
            let i = (2.0 / 5.0) * mass * radius * radius;
            Vec3::splat(i)
        }
        ShapeDesc::Box { half_extents } => {
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            let d = 2.0 * half_extents.z;
            Vec3::new(
                mass / 12.0 * (h * h + d * d),
                mass / 12.0 * (w * w + d * d),
                mass / 12.0 * (w * w + h * h),
            )
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            let h = 2.0 * half_height;
            let r2 = radius * radius;
            let cyl_mass = mass * h / (h + (4.0 / 3.0) * radius);
            let cap_mass = mass - cyl_mass;
            let iy = cyl_mass * r2 / 2.0 + cap_mass * 2.0 * r2 / 5.0;
            let ix = cyl_mass * (3.0 * r2 + h * h) / 12.0
                + cap_mass * (2.0 * r2 / 5.0 + h * h / 4.0 + 3.0 * h * radius / 8.0);
            Vec3::new(ix, iy, ix)
        }
        ShapeDesc::ConvexHull { vertices } => {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for &vertex in vertices {
                min = min.min(vertex);
                max = max.max(vertex);
            }
            let size = max - min;
            Vec3::new(
                mass / 12.0 * (size.y * size.y + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.y * size.y),
            )
        }
        ShapeDesc::Plane { .. } => Vec3::ZERO,
        ShapeDesc::Compound { children } => {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            for (child_shape, local_pos, _) in children {
                let extent = match child_shape {
                    ShapeDesc::Sphere { radius } => Vec3::splat(*radius),
                    ShapeDesc::Box { half_extents } => *half_extents,
                    ShapeDesc::Capsule {
                        half_height,
                        radius,
                    } => Vec3::new(*radius, *half_height + *radius, *radius),
                    ShapeDesc::ConvexHull { .. } => Vec3::splat(1.0),
                    ShapeDesc::Plane { .. } => Vec3::splat(1.0),
                    ShapeDesc::Compound { .. } => Vec3::splat(1.0),
                };
                min = min.min(*local_pos - extent);
                max = max.max(*local_pos + extent);
            }
            let size = max - min;
            Vec3::new(
                mass / 12.0 * (size.y * size.y + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.z * size.z),
                mass / 12.0 * (size.x * size.x + size.y * size.y),
            )
        }
    }
}
