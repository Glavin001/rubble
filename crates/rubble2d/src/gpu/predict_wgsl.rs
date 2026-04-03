/// WGSL source for the 2D prediction shader.
///
/// 1. Saves the current state to `old_states` (used later for velocity extraction).
/// 2. Computes the inertial target `x* = x + v dt + g dt^2`.
/// 3. Computes a VBD-style warmstarted primal position using the previous-step velocity.
/// 4. Writes the warmstarted state to `bodies` and the inertial target to `inertial_states`.
pub const PREDICT_2D_WGSL: &str = r#"
// ---------- Types matching rubble-math RigidBodyState2D (64 bytes) ----------

// RigidBodyState2D: 4 x vec4f = 64 bytes
struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct SimParams2D {
    gravity: vec4<f32>, // (gx, gy, 0, 0)
    solver:  vec4<f32>, // (dt, beta, k_start, max_penalty)
    counts:  vec4<u32>, // (num_bodies, solver_iterations, pair_count, flags)
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body2D>;
@group(0) @binding(1) var<storage, read_write> old_states: array<Body2D>;
@group(0) @binding(2) var<storage, read_write> inertial_states: array<Body2D>;
@group(0) @binding(3) var<storage, read>       prev_states:     array<Body2D>;
@group(0) @binding(4) var<uniform>             params:          SimParams2D;

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
    let gravity = params.gravity.xy;
    if inv_mass <= 0.0 {
        inertial_states[idx] = body;
        return; // static body -- no prediction
    }

    let pos = body.position_inv_mass.xy;
    let angle = body.position_inv_mass.z;
    let vel = body.lin_vel.xy;
    let omega = body.lin_vel.z;
    let prev_vel = prev_states[idx].lin_vel.xy;

    // Inertial target used by the primal solve.
    let inertial_pos = pos + vel * dt + gravity * (dt * dt);
    let inertial_angle = angle + dt * omega;

    // Adaptive warmstart weight from motion along the gravity direction.
    let gravity_mag = length(gravity);
    var accel_weight = 0.0;
    if gravity_mag > 1e-6 && dt > 1e-12 {
        let gravity_dir = gravity / gravity_mag;
        let accel = (vel - prev_vel) / dt;
        let accel_ext = dot(accel, gravity_dir);
        accel_weight = clamp(accel_ext / gravity_mag, 0.0, 1.0);
    }

    let warm_pos = pos + vel * dt + gravity * (accel_weight * dt * dt);
    let warm_angle = inertial_angle;

    inertial_states[idx].position_inv_mass = vec4<f32>(inertial_pos, inertial_angle, inv_mass);
    inertial_states[idx].lin_vel = body.lin_vel;
    inertial_states[idx]._pad0 = body._pad0;
    inertial_states[idx]._pad1 = body._pad1;

    bodies[idx].position_inv_mass = vec4<f32>(warm_pos, warm_angle, inv_mass);
    bodies[idx].lin_vel = body.lin_vel;
}
"#;
