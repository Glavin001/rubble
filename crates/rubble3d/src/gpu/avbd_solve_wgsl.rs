/// WGSL source for the AVBD (Averaged Velocity-Based Dynamics) solver.
///
/// Operates on velocities directly. For each contact, computes the velocity
/// constraint violation using averaged velocities, then applies impulses.
///
/// Supports graph-colored dispatch: contacts are sorted by color group,
/// and `solve_range` provides (offset, count) so each dispatch processes
/// only contacts of one color (no two share a body → no data races).
pub const AVBD_SOLVE_WGSL: &str = r#"
// ---------- Types ----------

struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

// RigidBodyProps3D: 3 x vec4 + 4 x f32 = 64 bytes
struct BodyProps {
    inv_inertia_row0: vec4<f32>,
    inv_inertia_row1: vec4<f32>,
    inv_inertia_row2: vec4<f32>,
    friction:         f32,
    shape_type:       u32,
    shape_index:      u32,
    flags:            u32,
};

struct Contact {
    point:      vec4<f32>, // (x, y, z, depth)
    normal:     vec4<f32>, // (nx, ny, nz, 0)
    body_a:     u32,
    body_b:     u32,
    feature_id: u32,
    _pad:       u32,
    lambda_n:   f32,
    lambda_t1:  f32,
    lambda_t2:  f32,
    penalty_k:  f32,
};

struct SimParams {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

struct SolveRange {
    offset: u32,
    count:  u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:     array<Body>;
@group(0) @binding(1) var<storage, read>       old_states: array<Body>;
@group(0) @binding(2) var<storage, read>       props:      array<BodyProps>;
@group(0) @binding(3) var<storage, read_write> contacts:   array<Contact>;
@group(0) @binding(4) var<uniform>             params:     SimParams;
@group(0) @binding(5) var<storage, read>       contact_count_buf: array<u32>;
@group(0) @binding(6) var<uniform>             solve_range: SolveRange;

// Multiply a 3x3 matrix (stored as 3 vec4 rows, .xyz used) by a vec3.
fn mat3_mul(r0: vec4<f32>, r1: vec4<f32>, r2: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // The columns are stored as rows due to Rust column-major storage.
    // inv_inertia_row0 is actually column 0, etc.
    return vec3<f32>(
        r0.x * v.x + r1.x * v.y + r2.x * v.z,
        r0.y * v.x + r1.y * v.y + r2.y * v.z,
        r0.z * v.x + r1.z * v.y + r2.z * v.z,
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }
    let ci = local_idx + solve_range.offset;
    let num_contacts = contact_count_buf[0];
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let a = c.body_a;
    let b = c.body_b;
    let normal = c.normal.xyz;
    let cp = c.point.xyz;

    let im_a = bodies[a].position_inv_mass.w;
    let im_b = bodies[b].position_inv_mass.w;
    let w_sum = im_a + im_b;
    if w_sum <= 0.0 {
        return;
    }

    let pos_a = bodies[a].position_inv_mass.xyz;
    let pos_b = bodies[b].position_inv_mass.xyz;

    // Contact offsets (from body centers to contact point)
    let r_a = cp - pos_a;
    let r_b = cp - pos_b;

    // Inverse inertia tensors
    let props_a = props[a];
    let props_b = props[b];

    // Effective mass: w_eff = inv_m_a + inv_m_b + (inv_I_a * (r_a x n)) . (r_a x n) + ...
    let rn_a = cross(r_a, normal);
    let rn_b = cross(r_b, normal);
    let inv_I_rn_a = mat3_mul(props_a.inv_inertia_row0, props_a.inv_inertia_row1, props_a.inv_inertia_row2, rn_a);
    let inv_I_rn_b = mat3_mul(props_b.inv_inertia_row0, props_b.inv_inertia_row1, props_b.inv_inertia_row2, rn_b);
    let w_rot_a = dot(inv_I_rn_a, rn_a);
    let w_rot_b = dot(inv_I_rn_b, rn_b);
    let w_eff = w_sum + w_rot_a + w_rot_b;
    if w_eff <= 0.0 {
        return;
    }

    // Use contact depth directly (computed at predicted positions by narrowphase)
    let depth = c.point.w;
    if depth >= 0.0 {
        return; // no penetration
    }

    // --- AVBD: velocity-level solve using current velocities ---
    let v_a = bodies[a].lin_vel.xyz;
    let v_b = bodies[b].lin_vel.xyz;
    let w_a = bodies[a].ang_vel.xyz;
    let w_b = bodies[b].ang_vel.xyz;

    // Keep old_states binding alive (dispatch provides it)
    let _keep = old_states[0].position_inv_mass.x;

    // Relative velocity at contact point
    let v_rel = (v_b + cross(w_b, r_b)) - (v_a + cross(w_a, r_a));
    let v_n = dot(v_rel, normal);

    // Baumgarte stabilization: add position correction term
    let beta_stabilize = 0.2;
    let bias = beta_stabilize * depth / params.dt;

    // Velocity constraint violation
    let dC = v_n + bias;
    if dC >= 0.0 {
        return; // separating
    }

    // Compute impulse magnitude with augmented Lagrangian
    let penalty = c.penalty_k;
    let lambda_old = c.lambda_n;
    let impulse = (-dC * penalty + lambda_old) / (w_eff * penalty + 1.0);
    let impulse_clamped = max(impulse, 0.0);

    // Apply linear impulses
    if im_a > 0.0 {
        bodies[a].lin_vel = vec4<f32>(v_a - normal * impulse_clamped * im_a, 0.0);
        bodies[a].ang_vel = vec4<f32>(w_a - inv_I_rn_a * impulse_clamped, 0.0);
    }
    if im_b > 0.0 {
        bodies[b].lin_vel = vec4<f32>(v_b + normal * impulse_clamped * im_b, 0.0);
        bodies[b].ang_vel = vec4<f32>(w_b + inv_I_rn_b * impulse_clamped, 0.0);
    }

    // Update dual variable
    contacts[ci].lambda_n = max(lambda_old + penalty * (-depth), 0.0);

    // --- Friction (tangential impulse) ---
    // Use per-body friction (average of both bodies)
    let mu = (props_a.friction + props_b.friction) * 0.5;
    let v_rel_current = (v_b + cross(w_b, r_b)) - (v_a + cross(w_a, r_a));
    let v_tangent = v_rel_current - normal * dot(v_rel_current, normal);
    let tang_len = length(v_tangent);
    if tang_len > 1e-8 {
        let tang_dir = v_tangent / tang_len;
        let max_tang_impulse = mu * impulse_clamped;
        let tang_impulse = min(tang_len / w_eff, max_tang_impulse);

        if im_a > 0.0 {
            bodies[a].lin_vel = vec4<f32>(bodies[a].lin_vel.xyz + tang_dir * tang_impulse * im_a, 0.0);
        }
        if im_b > 0.0 {
            bodies[b].lin_vel = vec4<f32>(bodies[b].lin_vel.xyz - tang_dir * tang_impulse * im_b, 0.0);
        }
    }

    // Stiffness ramp
    contacts[ci].penalty_k = penalty + 10.0 * (-depth);
}
"#;
