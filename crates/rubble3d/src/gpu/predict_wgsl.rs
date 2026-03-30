/// WGSL source for the prediction shader.
///
/// Computes predicted positions: x_tilde = pos + dt*vel + dt^2*gravity.
/// Also integrates orientation: q_tilde = normalize(q + 0.5 * dt * omega_quat * q).
/// Copies current state to old_states for velocity extraction.
pub const PREDICT_WGSL: &str = r#"
// ---------- Types matching rubble-math Rust layouts ----------

// RigidBodyState3D: 4 x vec4f = 64 bytes
struct Body {
    position_inv_mass: vec4<f32>, // (x, y, z, 1/m)
    orientation:       vec4<f32>, // quaternion (x, y, z, w)
    lin_vel:           vec4<f32>, // (vx, vy, vz, 0)
    ang_vel:           vec4<f32>, // (wx, wy, wz, 0)
};

struct SimParams {
    gravity:           vec4<f32>, // (gx, gy, gz, 0)
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body>;
@group(0) @binding(1) var<storage, read_write> old_states: array<Body>;
@group(0) @binding(2) var<uniform>             params:     SimParams;

// Quaternion multiply: a * b
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_bodies {
        return;
    }

    let body = bodies[idx];
    let inv_mass = body.position_inv_mass.w;

    // Save old state for velocity extraction.
    old_states[idx] = body;

    if inv_mass <= 0.0 {
        return; // static body — no prediction
    }

    let pos = body.position_inv_mass.xyz;
    let vel = body.lin_vel.xyz;
    let omega = body.ang_vel.xyz;
    let dt = params.dt;
    let gravity = params.gravity.xyz;

    // Position prediction: x_tilde = pos + dt*vel + dt^2 * gravity
    let x_tilde = pos + dt * vel + dt * dt * gravity;
    bodies[idx].position_inv_mass = vec4<f32>(x_tilde, inv_mass);

    // Orientation prediction: q_tilde = normalize(q + 0.5 * dt * omega_quat * q)
    let q = body.orientation;
    let omega_quat = vec4<f32>(omega, 0.0);
    let q_dot = qmul(omega_quat, q);
    let q_new = normalize(q + 0.5 * dt * q_dot);
    bodies[idx].orientation = q_new;
}
"#;
