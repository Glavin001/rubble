/// WGSL source for the 2D velocity extraction shader.
///
/// Recomputes positions from the AVBD-solved velocities:
///   pos_new = pos_old + dt * v_solved
///   angle_new = angle_old + dt * omega_solved
/// This ensures the solver's velocity corrections are reflected in positions.
/// Then extracts velocity: v = (pos_new - pos_old) / dt = v_solved (consistent).
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

    let dt = params.dt;

    // The solver may have modified velocities. Recompute position from
    // old position + dt * solved velocity, so impulses actually move bodies.
    let v_solved = bodies[idx].lin_vel.xy;
    let omega_solved = bodies[idx].lin_vel.z;
    let pos_old = old_states[idx].position_inv_mass.xy;
    let angle_old = old_states[idx].position_inv_mass.z;

    let pos_new = pos_old + dt * v_solved;
    let angle_new = angle_old + dt * omega_solved;

    bodies[idx].position_inv_mass = vec4<f32>(pos_new, angle_new, inv_mass);
    // Velocity is already correct (v_solved from predict + solver impulses)
}
"#;
