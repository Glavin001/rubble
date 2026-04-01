/// WGSL source for the 2D prediction shader.
///
/// 1. Saves current state to old_states (for velocity extraction later).
/// 2. Integrates velocity with gravity: v' = v + dt * g
/// 3. Predicts position: x_tilde = pos + dt * v'
/// 4. Stores BOTH updated velocity AND predicted position for the solver.
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
    gravity:           vec4<f32>, // (gx, gy, 0, 0)
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body2D>;
@group(0) @binding(1) var<storage, read_write> old_states: array<Body2D>;
@group(0) @binding(2) var<uniform>             params:     SimParams2D;

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
        return; // static body -- no prediction
    }

    let pos = body.position_inv_mass.xy;
    let angle = body.position_inv_mass.z;
    let vel = body.lin_vel.xy;
    let omega = body.lin_vel.z;
    let dt = params.dt;
    let gravity = params.gravity.xy;

    // Integrate velocity with gravity: v' = v + dt * g
    let new_vel = vel + dt * gravity;

    // Position prediction: x_tilde = pos + dt * v'
    let x_tilde = pos + dt * new_vel;

    // Angle prediction: angle_tilde = angle + dt * omega
    let angle_tilde = angle + dt * omega;

    // Store predicted position AND gravity-integrated velocity
    bodies[idx].position_inv_mass = vec4<f32>(x_tilde, angle_tilde, inv_mass);
    bodies[idx].lin_vel = vec4<f32>(new_vel, omega, 0.0);
}
"#;
