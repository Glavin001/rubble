/// WGSL source for the velocity extraction shader.
///
/// Recomputes positions from the AVBD-solved velocities:
///   pos_new = pos_old + dt * v_solved
/// This ensures the solver's velocity corrections are reflected in positions.
/// Angular velocity is extracted from orientation delta.
pub const EXTRACT_VELOCITY_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct SimParams {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body>;
@group(0) @binding(1) var<storage, read>       old_states: array<Body>;
@group(0) @binding(2) var<uniform>             params:     SimParams;

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
    if idx >= params.num_bodies {
        return;
    }

    let inv_mass = bodies[idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return; // static body
    }

    let dt = params.dt;

    // The AVBD solver modified velocities. Recompute position from
    // old position + dt * solved velocity, so impulses actually move bodies.
    let v_solved = bodies[idx].lin_vel.xyz;
    let pos_old = old_states[idx].position_inv_mass.xyz;
    let pos_new = pos_old + dt * v_solved;
    bodies[idx].position_inv_mass = vec4<f32>(pos_new, inv_mass);
    // Linear velocity is already correct (v_solved from predict + solver impulses)

    // Angular velocity: omega = 2 * (q_new * conj(q_old)).xyz / dt
    let q_new = bodies[idx].orientation;
    let q_old = old_states[idx].orientation;
    let dq = qmul(q_new, qconj(q_old));
    // Ensure shortest path (positive w)
    let sign = select(-1.0, 1.0, dq.w >= 0.0);
    let omega_new = dq.xyz * sign * 2.0 / dt;
    bodies[idx].ang_vel = vec4<f32>(omega_new, 0.0);
}
"#;
