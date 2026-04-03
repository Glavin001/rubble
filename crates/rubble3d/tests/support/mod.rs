#![allow(dead_code)]

use glam::{Mat3, Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use rubble_math::{BodyHandle, Contact3D};
use std::{
    collections::HashSet,
    fmt,
    fs::OpenOptions,
    io::Write,
    time::{SystemTime, UNIX_EPOCH},
};

pub const RUN_KNOWN_FAILURES_ENV: &str = "RUBBLE_RUN_KNOWN_FAILURES";
const DEBUG_LOG_PATH: &str = "/Users/glavin/Development/rubble/.cursor/debug-543c9a.log";
const DEBUG_SESSION_ID: &str = "543c9a";

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

#[derive(Debug, Clone)]
pub struct DebugTraceConfig3D {
    pub scene: &'static str,
    pub run_id: &'static str,
    pub sample_bodies: Vec<usize>,
    pub monitored_pairs: Vec<(usize, usize)>,
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

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_f32(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".to_string()
    }
}

fn append_debug_log(
    run_id: &str,
    hypothesis_id: &str,
    location: &str,
    message: &str,
    data_json: String,
) {
    let line = format!(
        "{{\"sessionId\":\"{}\",\"runId\":\"{}\",\"hypothesisId\":\"{}\",\"location\":\"{}\",\"message\":\"{}\",\"data\":{},\"timestamp\":{}}}",
        DEBUG_SESSION_ID,
        json_escape(run_id),
        json_escape(hypothesis_id),
        json_escape(location),
        json_escape(message),
        data_json,
        timestamp_millis(),
    );
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(DEBUG_LOG_PATH)
    {
        let _ = writeln!(file, "{line}");
    }
}

fn current_contact_keys(contacts: &[Contact3D]) -> HashSet<(u32, u32, u32)> {
    contacts
        .iter()
        .map(|contact| {
            (
                contact.body_a.min(contact.body_b),
                contact.body_a.max(contact.body_b),
                contact.feature_id,
            )
        })
        .collect()
}

fn sample_bodies_json(report: &SceneReport3D, sample_bodies: &[usize]) -> String {
    let mut entries = Vec::new();
    for &body_idx in sample_bodies {
        if let Some(body) = report.bodies.get(body_idx) {
            entries.push(format!(
                "{{\"index\":{},\"label\":\"{}\",\"pos\":[{},{},{}],\"vel\":[{},{},{}],\"speed\":{},\"mass\":{}}}",
                body_idx,
                json_escape(body.label),
                json_f32(body.position.x),
                json_f32(body.position.y),
                json_f32(body.position.z),
                json_f32(body.linear_velocity.x),
                json_f32(body.linear_velocity.y),
                json_f32(body.linear_velocity.z),
                json_f32(body.linear_velocity.length()),
                json_f32(body.mass),
            ));
        }
    }
    format!("[{}]", entries.join(","))
}

fn monitored_pairs_json(contacts: &[Contact3D], monitored_pairs: &[(usize, usize)]) -> String {
    let mut entries = Vec::new();
    for &(body_a, body_b) in monitored_pairs {
        let pair_a = body_a.min(body_b) as u32;
        let pair_b = body_a.max(body_b) as u32;
        let pair_contacts: Vec<&Contact3D> = contacts
            .iter()
            .filter(|contact| {
                let a = contact.body_a.min(contact.body_b);
                let b = contact.body_a.max(contact.body_b);
                a == pair_a && b == pair_b
            })
            .collect();

        let mut feature_ids: Vec<u32> = pair_contacts
            .iter()
            .map(|contact| contact.feature_id)
            .collect();
        feature_ids.sort_unstable();
        feature_ids.dedup();
        let feature_ids_json = feature_ids
            .iter()
            .take(8)
            .map(|feature_id| feature_id.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let count = pair_contacts.len();
        let avg_normal_y = if count > 0 {
            pair_contacts
                .iter()
                .map(|contact| contact.normal.y)
                .sum::<f32>()
                / count as f32
        } else {
            0.0
        };
        let min_depth = pair_contacts
            .iter()
            .map(|contact| contact.point.w)
            .fold(f32::INFINITY, f32::min);
        let max_depth = pair_contacts
            .iter()
            .map(|contact| contact.point.w)
            .fold(f32::NEG_INFINITY, f32::max);
        let max_lambda_n = pair_contacts
            .iter()
            .map(|contact| contact.lambda.x.abs())
            .fold(0.0, f32::max);
        let max_penalty_n = pair_contacts
            .iter()
            .map(|contact| contact.penalty.x)
            .fold(0.0, f32::max);
        let sticking_count = pair_contacts
            .iter()
            .filter(|contact| contact.flags & rubble_math::CONTACT_FLAG_STICKING != 0)
            .count();

        entries.push(format!(
            "{{\"a\":{},\"b\":{},\"count\":{},\"featureIds\":[{}],\"avgNormalY\":{},\"minDepth\":{},\"maxDepth\":{},\"maxLambdaN\":{},\"maxPenaltyN\":{},\"stickingCount\":{}}}",
            pair_a,
            pair_b,
            count,
            feature_ids_json,
            json_f32(avg_normal_y),
            json_f32(min_depth),
            json_f32(max_depth),
            json_f32(max_lambda_n),
            json_f32(max_penalty_n),
            sticking_count,
        ));
    }
    format!("[{}]", entries.join(","))
}

fn trace_step_json(
    scene: &str,
    report: &SceneReport3D,
    contacts: &[Contact3D],
    prev_keys: &HashSet<(u32, u32, u32)>,
    sample_bodies: &[usize],
    monitored_pairs: &[(usize, usize)],
    timings: &rubble_gpu::StepTimingsMs,
) -> (String, HashSet<(u32, u32, u32)>) {
    let current_keys = current_contact_keys(contacts);
    let persistent_keys = current_keys.intersection(prev_keys).count();
    let unique_pairs: HashSet<(u32, u32)> = contacts
        .iter()
        .map(|contact| {
            (
                contact.body_a.min(contact.body_b),
                contact.body_a.max(contact.body_b),
            )
        })
        .collect();

    let mut centers: Vec<f32> = report
        .bodies
        .iter()
        .filter(|body| body.mass > 0.0)
        .map(|body| body.position.y)
        .collect();
    centers.sort_by(f32::total_cmp);
    let min_gap = centers
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f32::INFINITY, f32::min);
    let lowest_center_y = centers.first().copied().unwrap_or(0.0);
    let highest_center_y = centers.last().copied().unwrap_or(0.0);
    let max_abs_vertical_speed = report
        .bodies
        .iter()
        .filter(|body| body.mass > 0.0)
        .map(|body| body.linear_velocity.y.abs())
        .fold(0.0, f32::max);
    let floor_contact_count = contacts
        .iter()
        .filter(|contact| contact.body_a == 0 || contact.body_b == 0)
        .count();
    let floor_pair_count = unique_pairs
        .iter()
        .filter(|(body_a, body_b)| *body_a == 0 || *body_b == 0)
        .count();
    let max_lambda_n = contacts
        .iter()
        .map(|contact| contact.lambda.x.abs())
        .fold(0.0, f32::max);
    let max_penalty_n = contacts
        .iter()
        .map(|contact| contact.penalty.x)
        .fold(0.0, f32::max);
    let min_depth = contacts
        .iter()
        .map(|contact| contact.point.w)
        .fold(f32::INFINITY, f32::min);
    let max_depth = contacts
        .iter()
        .map(|contact| contact.point.w)
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_abs_normal_y = if contacts.is_empty() {
        0.0
    } else {
        contacts
            .iter()
            .map(|contact| contact.normal.y.abs())
            .sum::<f32>()
            / contacts.len() as f32
    };

    let json = format!(
        "{{\"scene\":\"{}\",\"step\":{},\"contactCount\":{},\"uniquePairs\":{},\"persistentKeys\":{},\"newKeys\":{},\"floorContactCount\":{},\"floorPairCount\":{},\"minDepth\":{},\"maxDepth\":{},\"avgAbsNormalY\":{},\"maxLambdaN\":{},\"maxPenaltyN\":{},\"lowestCenterY\":{},\"highestCenterY\":{},\"minGap\":{},\"maxAbsVerticalSpeed\":{},\"maxSpeed\":{},\"minHeight\":{},\"sampleBodies\":{},\"monitoredPairs\":{},\"timings\":{{\"uploadMs\":{},\"predictAabbMs\":{},\"broadphaseMs\":{},\"narrowphaseMs\":{},\"solveMs\":{},\"extractMs\":{}}}}}",
        json_escape(scene),
        report.step,
        contacts.len(),
        unique_pairs.len(),
        persistent_keys,
        current_keys.len().saturating_sub(persistent_keys),
        floor_contact_count,
        floor_pair_count,
        json_f32(min_depth),
        json_f32(max_depth),
        json_f32(avg_abs_normal_y),
        json_f32(max_lambda_n),
        json_f32(max_penalty_n),
        json_f32(lowest_center_y),
        json_f32(highest_center_y),
        json_f32(min_gap),
        json_f32(max_abs_vertical_speed),
        json_f32(report.metrics.max_speed),
        json_f32(report.metrics.min_height),
        sample_bodies_json(report, sample_bodies),
        monitored_pairs_json(contacts, monitored_pairs),
        json_f32(timings.upload_ms),
        json_f32(timings.predict_aabb_ms),
        json_f32(timings.broadphase_ms),
        json_f32(timings.narrowphase_ms),
        json_f32(timings.solve_ms),
        json_f32(timings.extract_ms),
    );

    (json, current_keys)
}

pub fn collect_reports_with_debug_trace(
    world: &mut World,
    bodies: &[TrackedBody3D],
    gravity: Vec3,
    steps: usize,
    trace: &DebugTraceConfig3D,
) -> Vec<SceneReport3D> {
    let mut reports = Vec::with_capacity(steps);
    let mut prev_keys = HashSet::new();
    let initial = scene_report(world, bodies, gravity, 0);
    let monitored_pairs = trace
        .monitored_pairs
        .iter()
        .map(|(body_a, body_b)| format!("{{\"a\":{},\"b\":{}}}", body_a, body_b))
        .collect::<Vec<_>>()
        .join(",");

    // #region agent log
    append_debug_log(
        trace.run_id,
        "H1,H2,H3,H4",
        "crates/rubble3d/tests/support/mod.rs:collect_reports_with_debug_trace:start",
        "trace_start",
        format!(
            "{{\"scene\":\"{}\",\"steps\":{},\"sampleBodies\":{},\"monitoredPairs\":[{}]}}",
            json_escape(trace.scene),
            steps,
            sample_bodies_json(&initial, &trace.sample_bodies),
            monitored_pairs,
        ),
    );
    // #endregion

    for step in 0..steps {
        world.step();
        let report = scene_report(world, bodies, gravity, step + 1);
        let contacts = world.debug_contacts();
        let (step_json, current_keys) = trace_step_json(
            trace.scene,
            &report,
            contacts,
            &prev_keys,
            &trace.sample_bodies,
            &trace.monitored_pairs,
            world.last_step_timings(),
        );

        // #region agent log
        append_debug_log(
            trace.run_id,
            "H1,H2,H3,H4",
            "crates/rubble3d/tests/support/mod.rs:collect_reports_with_debug_trace:step",
            "scene_step",
            step_json,
        );
        // #endregion

        prev_keys = current_keys;
        reports.push(report);
    }

    if let Some(last) = reports.last() {
        // #region agent log
        append_debug_log(
            trace.run_id,
            "H2,H4",
            "crates/rubble3d/tests/support/mod.rs:collect_reports_with_debug_trace:end",
            "trace_end",
            format!(
                "{{\"scene\":\"{}\",\"finalStep\":{},\"maxSpeed\":{},\"minHeight\":{},\"sampleBodies\":{}}}",
                json_escape(trace.scene),
                last.step,
                json_f32(last.metrics.max_speed),
                json_f32(last.metrics.min_height),
                sample_bodies_json(last, &trace.sample_bodies),
            ),
        );
        // #endregion
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
