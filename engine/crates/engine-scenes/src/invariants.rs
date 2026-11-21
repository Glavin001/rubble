//! Backend-neutral checks of a stepped scene, from its bodies and joints as any host
//! reads them back: one definition, so every host's scene test holds one bar.

use engine_schema::{BodyState, Joint};
use glam::{Quat, Vec3};

/// A scene's state at a checkpoint.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measure {
    /// First body with a non-finite pose or velocity.
    pub nonfinite: Option<usize>,
    /// Lowest dynamic body centre (m).
    pub lowest: f32,
    /// Dynamic bodies' linear kinetic energy (J).
    pub kinetic: f64,
    /// Locked-linear joints measured, and their anchors' separation (m): largest, mean.
    pub locked_joints: usize,
    pub gap_max: f32,
    pub gap_mean: f64,
}

fn world_anchor(b: &BodyState, local: [f32; 3]) -> Vec3 {
    Vec3::from_array(b.position) + Quat::from_array(b.orientation) * Vec3::from_array(local)
}

/// A joint whose three linear DOFs are locked (2 bits each, 0 locked).
pub fn locked_linear(j: &Joint) -> bool {
    j.motion & 0b11_1111 == 0
}

pub fn measure(bodies: &[BodyState], joints: &[Joint]) -> Measure {
    let mut m = Measure {
        lowest: f32::MAX,
        ..Default::default()
    };
    for (id, b) in bodies.iter().enumerate() {
        let finite = b
            .position
            .iter()
            .chain(&b.orientation)
            .chain(&b.linear_vel)
            .chain(&b.angular_vel)
            .all(|v| v.is_finite());
        if !finite && m.nonfinite.is_none() {
            m.nonfinite = Some(id);
        }
        if b.inv_mass > 0.0 {
            m.lowest = m.lowest.min(b.position[1]);
            m.kinetic +=
                0.5 * f64::from(Vec3::from_array(b.linear_vel).length_squared() / b.inv_mass);
        }
    }
    let mut sum = 0.0f64;
    for j in joints.iter().filter(|j| locked_linear(j)) {
        let d = (world_anchor(&bodies[j.body_a as usize], j.anchor_a)
            - world_anchor(&bodies[j.body_b as usize], j.anchor_b))
        .length();
        m.gap_max = m.gap_max.max(d);
        sum += f64::from(d);
        m.locked_joints += 1;
    }
    m.gap_mean = if m.locked_joints == 0 {
        0.0
    } else {
        sum / m.locked_joints as f64
    };
    m
}

/// What a host gives the scene gate: one step (waiting), the bodies, the colouring
/// check, and a pipelined step with its final wait.
pub trait GateHost {
    fn step(&mut self) -> anyhow::Result<engine_schema::StepCounters>;
    fn bodies(&mut self) -> Vec<BodyState>;
    fn validate_coloring(&mut self) -> Result<(), String>;
    fn submit(&mut self) -> anyhow::Result<()>;
    fn drain(&mut self) -> anyhow::Result<engine_schema::StepCounters>;
}

/// The scene gate every host runs the same way: `steps` steps, at step 1 and every 100th
/// the state is finite, the colouring valid, no dynamic body below y = −1 m (through the
/// ground), no locked joint open past `max_gap` m and their mean under
/// [`JOINT_GAP_MEAN_MAX`]; then 120 pipelined steps timed.
/// Returns ms a step of those.
pub fn gate(
    name: &str,
    host: &mut dyn GateHost,
    joints: &[Joint],
    steps: u32,
    max_gap: f32,
) -> f64 {
    for step in 1..=steps {
        let c = host
            .step()
            .unwrap_or_else(|e| panic!("{name} step {step}: {e:?}"));
        if step % 100 != 0 && step != 1 {
            continue;
        }
        host.validate_coloring()
            .unwrap_or_else(|why| panic!("{name} step {step}: {why}"));
        let m = measure(&host.bodies(), joints);
        eprintln!(
            "{name} step {step}: contacts={} colors={} awake={} lowest y={:.3} KE={:.1} joint gap max={:.4} mean={:.5}",
            c.contact_count, c.color_count, c.awake_count, m.lowest, m.kinetic, m.gap_max, m.gap_mean
        );
        assert!(
            m.nonfinite.is_none(),
            "{name} step {step}: body {:?} not finite",
            m.nonfinite
        );
        assert!(
            m.lowest > -1.0,
            "{name} step {step}: a body fell through the ground (y {})",
            m.lowest
        );
        assert!(
            m.locked_joints == 0 || m.gap_max < max_gap,
            "{name} step {step}: a locked joint opened {} m (limit {max_gap})",
            m.gap_max
        );
        assert!(
            m.gap_mean < JOINT_GAP_MEAN_MAX,
            "{name} step {step}: locked joints open {} m on average (limit {JOINT_GAP_MEAN_MAX})",
            m.gap_mean
        );
    }
    let n = 120;
    let t = std::time::Instant::now();
    for _ in 0..n {
        host.submit().expect("submit");
    }
    let c = host.drain().expect("drain");
    let ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(n);
    eprintln!(
        "{name}: {ms:.3} ms a step over {n} pipelined steps ({} awake, {} contacts)",
        c.awake_count, c.contact_count
    );
    ms
}

/// The scene gate's bars for locked joints' anchors (m), the same on every host. Set from
/// the WebGPU reference (2026-11-21, M3 Max, `engine-webgpu/tests/scene_gate.rs`): Ragdoll
/// Pile (10k) piles up with joints open by at most 0.137 m and 0.023 m on average (step
/// 100); Hinge Chains (10k) 0.020 / 0.013. About 1.8x and 1.3x those.
pub const JOINT_GAP_MAX: f32 = 0.25;
pub const JOINT_GAP_MEAN_MAX: f64 = 0.03;
