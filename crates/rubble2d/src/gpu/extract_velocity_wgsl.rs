/// WGSL source for the 2D velocity extraction shader.
///
/// Extracts velocity from the solved positions/orientations after the primal-dual
/// position-space solve.
pub const EXTRACT_VELOCITY_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body2D>;
@group(0) @binding(1) var<storage, read>       old_states: array<Body2D>;
@group(0) @binding(2) var<uniform>             params:     SimParams2D;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }

    let inv_mass = bodies[idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return; // static body
    }

    let dt = params.solver.x;
    if dt <= 1e-12 {
        return;
    }

    let pos_old = old_states[idx].position_inv_mass.xy;
    let angle_old = old_states[idx].position_inv_mass.z;
    let pos_new = bodies[idx].position_inv_mass.xy;
    let angle_new = bodies[idx].position_inv_mass.z;
    let pos_delta = pos_new - pos_old;
    // Tiny timesteps can produce sub-ULP position changes even though the
    // integrated velocity is non-zero. Preserve the already-updated velocity in
    // that case so free motion can accumulate until the position changes become
    // representable again.
    var lin_vel = bodies[idx].lin_vel.xy;
    if length(pos_delta) >= 1e-6 {
        lin_vel = pos_delta / dt;
    }
    let angle_delta = angle_new - angle_old;
    var ang_vel = bodies[idx].lin_vel.z;
    if abs(angle_delta) >= 1e-6 {
        ang_vel = angle_delta / dt;
    }
    // Clamp velocities to prevent explosion from solver overcorrection.
    let max_lin_speed = 100.0;
    let lin_speed = length(lin_vel);
    if lin_speed > max_lin_speed {
        lin_vel = lin_vel * (max_lin_speed / lin_speed);
    }
    let max_ang_speed = 100.0;
    ang_vel = clamp(ang_vel, -max_ang_speed, max_ang_speed);

    bodies[idx].lin_vel = vec4<f32>(lin_vel, ang_vel, 0.0);
}
"#;
