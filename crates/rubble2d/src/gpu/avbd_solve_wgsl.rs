/// WGSL source for the AVBD (Averaged Velocity-Based Dynamics) solver (2D).
///
/// Operates on velocities directly. For each contact, computes the velocity
/// constraint violation using averaged velocities, then applies impulses.
/// 2D: 3 DOF per body (x, y, angle).
pub const AVBD_SOLVE_2D_WGSL: &str = r#"
// ---------- Types ----------

struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct Contact2D {
    point:      vec4<f32>, // (x, y, depth, 0)
    normal:     vec4<f32>, // (nx, ny, 0, 0)
    body_a:     u32,
    body_b:     u32,
    feature_id: u32,
    _pad:       u32,
    lambda_n:   f32,
    lambda_t:   f32,
    penalty_k:  f32,
    _pad2:      f32,
};

struct SimParams2D {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:            array<Body2D>;
@group(0) @binding(1) var<storage, read>       old_states:        array<Body2D>;
@group(0) @binding(2) var<storage, read_write> contacts:          array<Contact2D>;
@group(0) @binding(3) var<uniform>             params:            SimParams2D;
@group(0) @binding(4) var<storage, read>       contact_count_buf: array<u32>;

// 2D cross product: r x n = r.x*n.y - r.y*n.x (scalar)
fn cross2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let num_contacts = contact_count_buf[0];
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let a = c.body_a;
    let b = c.body_b;
    let normal = c.normal.xy;
    let cp = c.point.xy;

    let im_a = bodies[a].position_inv_mass.w;
    let im_b = bodies[b].position_inv_mass.w;
    let w_sum = im_a + im_b;
    if w_sum <= 0.0 {
        return;
    }

    let pos_a = bodies[a].position_inv_mass.xy;
    let pos_b = bodies[b].position_inv_mass.xy;
    let old_pos_a = old_states[a].position_inv_mass.xy;
    let old_pos_b = old_states[b].position_inv_mass.xy;

    // Contact offsets (from body centers to contact point)
    let r_a = cp - pos_a;
    let r_b = cp - pos_b;

    // 2D rotational effective mass: inv_I * (r x n)^2
    // For 2D we approximate inv_inertia ~ inv_mass (simple approximation)
    let rn_a = cross2d(r_a, normal);
    let rn_b = cross2d(r_b, normal);
    let w_rot_a = im_a * rn_a * rn_a;
    let w_rot_b = im_b * rn_b * rn_b;
    let w_eff = w_sum + w_rot_a + w_rot_b;
    if w_eff <= 0.0 {
        return;
    }

    // Penetration depth (recomputed from current positions)
    let depth = c.point.z + dot(pos_b - pos_a, normal) - dot(old_pos_b - old_pos_a, normal);
    if depth >= 0.0 {
        return; // no penetration
    }

    // --- AVBD: velocity-level solve with averaged velocities ---
    let v_a = bodies[a].lin_vel.xy;
    let v_b = bodies[b].lin_vel.xy;
    let w_a = bodies[a].lin_vel.z;
    let w_b = bodies[b].lin_vel.z;

    let v_old_a = old_states[a].lin_vel.xy;
    let v_old_b = old_states[b].lin_vel.xy;
    let w_old_a = old_states[a].lin_vel.z;
    let w_old_b = old_states[b].lin_vel.z;

    // Averaged velocities (the "AV" in AVBD)
    let v_avg_a = 0.5 * (v_old_a + v_a);
    let v_avg_b = 0.5 * (v_old_b + v_b);
    let w_avg_a = 0.5 * (w_old_a + w_a);
    let w_avg_b = 0.5 * (w_old_b + w_b);

    // Velocity at contact point: v + omega x r (in 2D: omega * perp(r))
    // perp(r) = (-r.y, r.x), so v_contact = v + omega * (-r.y, r.x)
    let v_contact_a = v_avg_a + w_avg_a * vec2<f32>(-r_a.y, r_a.x);
    let v_contact_b = v_avg_b + w_avg_b * vec2<f32>(-r_b.y, r_b.x);
    let v_rel = v_contact_b - v_contact_a;
    let v_n = dot(v_rel, normal);

    // Baumgarte stabilization
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

    // Apply linear and angular impulses
    if im_a > 0.0 {
        bodies[a].lin_vel = vec4<f32>(
            v_a - normal * impulse_clamped * im_a,
            w_a - im_a * rn_a * impulse_clamped,
            0.0,
        );
    }
    if im_b > 0.0 {
        bodies[b].lin_vel = vec4<f32>(
            v_b + normal * impulse_clamped * im_b,
            w_b + im_b * rn_b * impulse_clamped,
            0.0,
        );
    }

    // Update dual variable
    contacts[ci].lambda_n = max(lambda_old + penalty * (-depth), 0.0);

    // --- Friction (tangential impulse) ---
    let v_rel_current_a = v_a + w_a * vec2<f32>(-r_a.y, r_a.x);
    let v_rel_current_b = v_b + w_b * vec2<f32>(-r_b.y, r_b.x);
    let v_rel_current = v_rel_current_b - v_rel_current_a;
    let v_tangent = v_rel_current - normal * dot(v_rel_current, normal);
    let tang_len = length(v_tangent);
    if tang_len > 1e-8 {
        let tang_dir = v_tangent / tang_len;
        let mu = 0.5; // friction coefficient
        let max_tang_impulse = mu * impulse_clamped;
        let tang_impulse = min(tang_len / w_eff, max_tang_impulse);

        if im_a > 0.0 {
            bodies[a].lin_vel = vec4<f32>(
                bodies[a].lin_vel.xy + tang_dir * tang_impulse * im_a,
                bodies[a].lin_vel.z,
                0.0,
            );
        }
        if im_b > 0.0 {
            bodies[b].lin_vel = vec4<f32>(
                bodies[b].lin_vel.xy - tang_dir * tang_impulse * im_b,
                bodies[b].lin_vel.z,
                0.0,
            );
        }
    }

    // Stiffness ramp
    contacts[ci].penalty_k = penalty + 10.0 * (-depth);
}
"#;
