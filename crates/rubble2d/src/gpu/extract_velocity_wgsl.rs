/// WGSL source for the 2D velocity extraction shader.
///
/// Computes new velocities from position/angle changes:
///   v_new = (pos_new - pos_old) / dt
///   omega_new = (angle_new - angle_old) / dt
pub const EXTRACT_VELOCITY_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct SimParams2D {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body2D>;
@group(0) @binding(1) var<storage, read>       old_states: array<Body2D>;
@group(0) @binding(2) var<uniform>             params:     SimParams2D;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_bodies {
        return;
    }

    let inv_mass = bodies[idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return; // static body
    }

    let inv_dt = 1.0 / params.dt;

    // Linear velocity: v = (pos_new - pos_old) / dt
    let pos_new = bodies[idx].position_inv_mass.xy;
    let pos_old = old_states[idx].position_inv_mass.xy;
    let v_new = (pos_new - pos_old) * inv_dt;

    // Angular velocity: omega = (angle_new - angle_old) / dt
    let angle_new = bodies[idx].position_inv_mass.z;
    let angle_old = old_states[idx].position_inv_mass.z;
    let omega_new = (angle_new - angle_old) * inv_dt;

    bodies[idx].lin_vel = vec4<f32>(v_new, omega_new, 0.0);
}
"#;
