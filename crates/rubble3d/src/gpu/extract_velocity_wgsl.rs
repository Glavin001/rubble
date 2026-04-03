/// WGSL source for the velocity extraction shader.
///
/// Extracts velocity from the solved positions/orientations after the
/// position-space AVBD solve.
pub const EXTRACT_VELOCITY_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body>;
@group(0) @binding(1) var<storage, read>       old_states: array<Body>;
@group(0) @binding(2) var<storage, read>       active_bodies: array<u32>;
@group(0) @binding(3) var<uniform>             params:     SimParams;

// Quaternion multiply
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// Quaternion conjugate
fn qconj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

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

    let pos_old = old_states[idx].position_inv_mass.xyz;
    let pos_new = bodies[idx].position_inv_mass.xyz;
    let pos_delta = pos_new - pos_old;
    var lin_vel = vec3<f32>(0.0);
    if length(pos_delta) >= 1e-6 {
        lin_vel = pos_delta / dt;
    } else if active_bodies[idx] == 0u {
        // Preserve free-motion velocity when sub-ULP position changes would
        // otherwise collapse to zero.
        lin_vel = bodies[idx].lin_vel.xyz;
    }
    let q_new = bodies[idx].orientation;
    let q_old = old_states[idx].orientation;
    var dq = qmul(q_new, qconj(q_old));
    if dq.w < 0.0 {
        dq = -dq;
    }
    let imag_len = length(dq.xyz);
    var omega_new = vec3<f32>(0.0);
    if imag_len >= 1e-6 {
        let angle = 2.0 * atan2(imag_len, dq.w);
        omega_new = dq.xyz / imag_len * (angle / dt);
    } else if active_bodies[idx] == 0u {
        omega_new = bodies[idx].ang_vel.xyz;
    }

    // Clamp velocities to prevent explosion from solver overcorrection.
    let max_lin_speed = 100.0;
    let lin_speed = length(lin_vel);
    if lin_speed > max_lin_speed {
        lin_vel = lin_vel * (max_lin_speed / lin_speed);
    }
    let max_ang_speed = 100.0;
    let ang_speed = length(omega_new);
    if ang_speed > max_ang_speed {
        omega_new = omega_new * (max_ang_speed / ang_speed);
    }

    bodies[idx].lin_vel = vec4<f32>(lin_vel, 0.0);
    bodies[idx].ang_vel = vec4<f32>(omega_new, 0.0);
}
"#;
