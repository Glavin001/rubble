/// WGSL source for the prediction shader.
///
/// 1. Saves the current state to `old_states` (used later for velocity extraction).
/// 2. Computes the inertial target `x* = x + v dt + g dt^2`.
/// 3. Computes a VBD-style warmstarted linear position using the previous-step velocity.
/// 4. Advances orientation by angular velocity and writes it to both the warmstarted state and
///    inertial target, matching the CPU reference's identical angular start/target states.
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
    gravity: vec4<f32>, // (gx, gy, gz, 0)
    solver:  vec4<f32>, // (dt, beta, k_start, max_penalty)
    counts:  vec4<u32>, // (num_bodies, solver_iterations, pair_count, flags)
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body>;
@group(0) @binding(1) var<storage, read_write> old_states: array<Body>;
@group(0) @binding(2) var<storage, read_write> inertial_states: array<Body>;
@group(0) @binding(3) var<storage, read>       prev_states:     array<Body>;
@group(0) @binding(4) var<uniform>             params:          SimParams;

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
    let num_bodies = params.counts.x;
    if idx >= num_bodies {
        return;
    }

    let body = bodies[idx];
    let inv_mass = body.position_inv_mass.w;

    // Save old state for velocity extraction.
    old_states[idx] = body;

    let dt = params.solver.x;
    let gravity = params.gravity.xyz;
    if inv_mass <= 0.0 {
        inertial_states[idx] = body;
        return; // static body -- no prediction
    }

    let pos = body.position_inv_mass.xyz;
    let vel = body.lin_vel.xyz;
    let omega = body.ang_vel.xyz;
    let prev_vel = prev_states[idx].lin_vel.xyz;

    let inertial_pos = pos + vel * dt + gravity * (dt * dt);

    let gravity_mag = length(gravity);
    var accel_weight = 0.0;
    if gravity_mag > 1e-6 && dt > 1e-12 {
        let gravity_dir = gravity / gravity_mag;
        let accel = (vel - prev_vel) / dt;
        let accel_ext = dot(accel, gravity_dir);
        accel_weight = clamp(accel_ext / gravity_mag, 0.0, 1.0);
    }
    let warm_pos = pos + vel * dt + gravity * (accel_weight * dt * dt);

    let q = body.orientation;
    let omega_quat = vec4<f32>(omega, 0.0);
    let q_dot = qmul(omega_quat, q);
    let q_new = normalize(q + 0.5 * dt * q_dot);

    inertial_states[idx].position_inv_mass = vec4<f32>(inertial_pos, inv_mass);
    inertial_states[idx].orientation = q_new;
    inertial_states[idx].lin_vel = body.lin_vel;
    inertial_states[idx].ang_vel = body.ang_vel;

    bodies[idx].position_inv_mass = vec4<f32>(warm_pos, inv_mass);
    bodies[idx].orientation = q_new;
    bodies[idx].lin_vel = body.lin_vel;
}
"#;
