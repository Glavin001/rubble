#![allow(dead_code)]

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D, World2D};
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

pub fn try_world(config: SimConfig2D) -> Option<World2D> {
    World2D::new(config).ok()
}

pub fn step_n(world: &mut World2D, steps: usize) {
    for _ in 0..steps {
        world.step();
    }
}

#[derive(Debug, Clone)]
pub struct TrackedBody2D {
    pub label: &'static str,
    pub handle: BodyHandle,
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc2D,
}

impl TrackedBody2D {
    pub fn is_dynamic(&self) -> bool {
        self.mass > 0.0
    }
}

pub fn add_tracked_body(
    world: &mut World2D,
    label: &'static str,
    desc: RigidBodyDesc2D,
) -> TrackedBody2D {
    let handle = world.add_body(&desc);
    TrackedBody2D {
        label,
        handle,
        mass: desc.mass,
        friction: desc.friction,
        shape: desc.shape,
    }
}

#[derive(Debug, Clone)]
pub struct BodySnapshot2D {
    pub label: &'static str,
    pub mass: f32,
    pub friction: f32,
    pub shape: ShapeDesc2D,
    pub position: Vec2,
    pub angle: f32,
    pub linear_velocity: Vec2,
    pub angular_velocity: f32,
}

pub fn snapshot_body(world: &World2D, body: &TrackedBody2D) -> BodySnapshot2D {
    BodySnapshot2D {
        label: body.label,
        mass: body.mass,
        friction: body.friction,
        shape: body.shape.clone(),
        position: world
            .get_position(body.handle)
            .expect("tracked 2D body position should exist"),
        angle: world
            .get_angle(body.handle)
            .expect("tracked 2D body angle should exist"),
        linear_velocity: world
            .get_velocity(body.handle)
            .expect("tracked 2D body velocity should exist"),
        angular_velocity: world
            .get_angular_velocity(body.handle)
            .expect("tracked 2D body angular velocity should exist"),
    }
}

pub fn snapshot_all(world: &World2D, bodies: &[TrackedBody2D]) -> Vec<BodySnapshot2D> {
    bodies
        .iter()
        .map(|body| snapshot_body(world, body))
        .collect()
}

#[derive(Debug, Clone)]
pub struct SystemMetrics2D {
    pub total_mass: f32,
    pub center_of_mass: Vec2,
    pub linear_momentum: Vec2,
    pub angular_momentum: f32,
    pub translational_ke: f32,
    pub rotational_ke: f32,
    pub potential_energy: f32,
    pub total_energy: f32,
    pub max_speed: f32,
    pub max_angular_speed: f32,
    pub min_height: f32,
}

pub fn compute_metrics(snapshots: &[BodySnapshot2D], gravity: Vec2) -> SystemMetrics2D {
    let dynamic: Vec<&BodySnapshot2D> = snapshots.iter().filter(|body| body.mass > 0.0).collect();
    let total_mass = dynamic.iter().map(|body| body.mass).sum::<f32>();
    let center_of_mass = if total_mass > 0.0 {
        dynamic
            .iter()
            .fold(Vec2::ZERO, |acc, body| acc + body.position * body.mass)
            / total_mass
    } else {
        Vec2::ZERO
    };

    let gravity_dir = if gravity.length_squared() > 1e-12 {
        -gravity.normalize()
    } else {
        Vec2::Y
    };

    let mut linear_momentum = Vec2::ZERO;
    let mut angular_momentum: f32 = 0.0;
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

        let r = body.position - center_of_mass;
        angular_momentum +=
            cross2(r, momentum) + local_inertia(body.mass, &body.shape) * body.angular_velocity;
        rotational_ke += 0.5
            * local_inertia(body.mass, &body.shape)
            * body.angular_velocity
            * body.angular_velocity;

        potential_energy += body.mass * (-gravity).dot(body.position);
        max_speed = max_speed.max(body.linear_velocity.length());
        max_angular_speed = max_angular_speed.max(body.angular_velocity.abs());
        min_height = min_height.min(body.position.dot(gravity_dir));
    }

    if !min_height.is_finite() {
        min_height = 0.0;
    }

    SystemMetrics2D {
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
pub struct SceneReport2D {
    pub step: usize,
    pub metrics: SystemMetrics2D,
    pub bodies: Vec<BodySnapshot2D>,
}

impl fmt::Display for SceneReport2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "step={} mass={:.4} com={:?} p={:?} L={:.6} ke_t={:.6} ke_r={:.6} pe={:.6} total={:.6} max_v={:.6} max_w={:.6} min_h={:.6}",
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
                "  {} pos={:?} vel={:?} angle={:.6} omega={:.6} mass={:.3} friction={:.3}",
                body.label,
                body.position,
                body.linear_velocity,
                body.angle,
                body.angular_velocity,
                body.mass,
                body.friction
            )?;
        }
        Ok(())
    }
}

pub fn scene_report(
    world: &World2D,
    bodies: &[TrackedBody2D],
    gravity: Vec2,
    step: usize,
) -> SceneReport2D {
    let body_snapshots = snapshot_all(world, bodies);
    let metrics = compute_metrics(&body_snapshots, gravity);
    SceneReport2D {
        step,
        metrics,
        bodies: body_snapshots,
    }
}

pub fn collect_reports(
    world: &mut World2D,
    bodies: &[TrackedBody2D],
    gravity: Vec2,
    steps: usize,
) -> Vec<SceneReport2D> {
    let mut reports = Vec::with_capacity(steps);
    for step in 0..steps {
        world.step();
        reports.push(scene_report(world, bodies, gravity, step + 1));
    }
    reports
}

pub fn square_polygon(extents: Vec2) -> Vec<Vec2> {
    vec![
        Vec2::new(-extents.x, -extents.y),
        Vec2::new(extents.x, -extents.y),
        Vec2::new(extents.x, extents.y),
        Vec2::new(-extents.x, extents.y),
    ]
}

pub fn regular_polygon(radius: f32, count: usize) -> Vec<Vec2> {
    let mut vertices = Vec::with_capacity(count);
    for i in 0..count {
        let angle = i as f32 * std::f32::consts::TAU / count as f32;
        vertices.push(Vec2::new(radius * angle.cos(), radius * angle.sin()));
    }
    vertices
}

fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn local_inertia(mass: f32, shape: &ShapeDesc2D) -> f32 {
    if mass <= 0.0 {
        return 0.0;
    }
    match shape {
        ShapeDesc2D::Circle { radius } => 0.5 * mass * radius * radius,
        ShapeDesc2D::Rect { half_extents } => {
            let w = 2.0 * half_extents.x;
            let h = 2.0 * half_extents.y;
            mass * (w * w + h * h) / 12.0
        }
        ShapeDesc2D::Capsule {
            half_height,
            radius,
        } => {
            let w = 2.0 * radius;
            let h = 2.0 * (half_height + radius);
            mass * (w * w + h * h) / 12.0
        }
        ShapeDesc2D::ConvexPolygon { vertices } => {
            let mut min = Vec2::splat(f32::MAX);
            let mut max = Vec2::splat(f32::NEG_INFINITY);
            for &vertex in vertices {
                min = min.min(vertex);
                max = max.max(vertex);
            }
            let size = max - min;
            mass * (size.x * size.x + size.y * size.y) / 12.0
        }
    }
}
