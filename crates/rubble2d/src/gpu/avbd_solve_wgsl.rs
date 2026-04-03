/// WGSL source for the 2D AVBD primal solve.
///
/// Each invocation solves one body block in position space using all rows
/// touching that body. Bodies are dispatched in body-color groups so the
/// updates are race free.
pub const AVBD_PRIMAL_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>, // (vx, vy, angular_vel, 0)
    _pad0:             vec4<f32>, // (friction, inv_inertia, 0, 0)
    _pad1:             vec4<f32>,
};

struct Contact2D {
    point:           vec4<f32>, // (x, y, depth, 0)
    normal:          vec4<f32>, // (nx, ny, tx, ty)
    local_anchors:   vec4<f32>, // (rA.x, rA.y, rB.x, rB.y)
    lambda_penalty:  vec4<f32>, // (lambda_n, lambda_t, penalty_n, penalty_t)
    body_a:          u32,
    body_b:          u32,
    feature_id:      u32,
    flags:           u32,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

struct SolveRange {
    offset: u32,
    count:  u32,
};

@group(0) @binding(0) var<storage, read_write> bodies:            array<Body2D>;
@group(0) @binding(1) var<storage, read>       inertial_states:   array<Body2D>;
@group(0) @binding(2) var<storage, read>       contacts:          array<Contact2D>;
@group(0) @binding(3) var<storage, read>       body_order:        array<u32>;
@group(0) @binding(4) var<uniform>             params:            SimParams2D;
@group(0) @binding(5) var<storage, read>       contact_count_buf: array<u32>;
@group(0) @binding(6) var<uniform>             solve_range:       SolveRange;

fn cross2d(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn rotate2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let ca = cos(angle);
    let sa = sin(angle);
    return vec2<f32>(ca * v.x - sa * v.y, sa * v.x + ca * v.y);
}

fn solve_sym_3x3(
    a00: f32, a01: f32, a02: f32,
    a11: f32, a12: f32, a22: f32,
    rhs: vec3<f32>,
) -> vec3<f32> {
    let det_m =
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02);
    if abs(det_m) < 1e-10 {
        return vec3<f32>(0.0);
    }
    let det_x =
        rhs.x * (a11 * a22 - a12 * a12)
        - a01 * (rhs.y * a22 - a12 * rhs.z)
        + a02 * (rhs.y * a12 - a11 * rhs.z);
    let det_y =
        a00 * (rhs.y * a22 - a12 * rhs.z)
        - rhs.x * (a01 * a22 - a12 * a02)
        + a02 * (a01 * rhs.z - rhs.y * a02);
    let det_z =
        a00 * (a11 * rhs.z - rhs.y * a12)
        - a01 * (a01 * rhs.z - rhs.y * a02)
        + rhs.x * (a01 * a12 - a11 * a02);
    return vec3<f32>(det_x, det_y, det_z) / det_m;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }

    let body_idx = body_order[solve_range.offset + local_idx];
    let inv_mass = bodies[body_idx].position_inv_mass.w;
    let inv_inertia = bodies[body_idx]._pad0.y;
    if inv_mass <= 0.0 || inv_inertia <= 0.0 {
        return;
    }

    let pos = bodies[body_idx].position_inv_mass.xy;
    let angle = bodies[body_idx].position_inv_mass.z;
    let inertial_pos = inertial_states[body_idx].position_inv_mass.xy;
    let inertial_angle = inertial_states[body_idx].position_inv_mass.z;

    let dt = params.solver.x;
    let dt2 = dt * dt;
    let mass = 1.0 / inv_mass;
    let inertia = 1.0 / inv_inertia;

    var a00 = mass / dt2;
    var a01 = 0.0;
    var a02 = 0.0;
    var a11 = mass / dt2;
    var a12 = 0.0;
    var a22 = inertia / dt2;
    var rhs = vec3<f32>(
        a00 * (pos.x - inertial_pos.x),
        a11 * (pos.y - inertial_pos.y),
        a22 * (angle - inertial_angle),
    );

    let num_contacts = contact_count_buf[0];
    for (var ci = 0u; ci < num_contacts; ci = ci + 1u) {
        let c = contacts[ci];
        let is_a = c.body_a == body_idx;
        let is_b = c.body_b == body_idx;
        if !is_a && !is_b {
            continue;
        }

        let pos_a = bodies[c.body_a].position_inv_mass.xy;
        let pos_b = bodies[c.body_b].position_inv_mass.xy;
        let angle_a = bodies[c.body_a].position_inv_mass.z;
        let angle_b = bodies[c.body_b].position_inv_mass.z;
        let r_a = rotate2d(c.local_anchors.xy, angle_a);
        let r_b = rotate2d(c.local_anchors.zw, angle_b);
        let world_a = pos_a + r_a;
        let world_b = pos_b + r_b;
        let separation = world_a - world_b;

        let normal = c.normal.xy;
        let tangent = c.normal.zw;
        let c_n = dot(normal, separation);
        let c_t = dot(tangent, separation);
        let lambda_n = c.lambda_penalty.x;
        let lambda_t = c.lambda_penalty.y;
        let k_n = c.lambda_penalty.z;
        let k_t = c.lambda_penalty.w;

        let mu = sqrt(max(bodies[c.body_a]._pad0.x * bodies[c.body_b]._pad0.x, 0.0));
        let f_n = min(k_n * c_n + lambda_n, 0.0);
        let tang_limit = mu * abs(f_n);
        let f_t = clamp(k_t * c_t + lambda_t, -tang_limit, tang_limit);

        var jn = vec3<f32>(normal.x, normal.y, cross2d(r_a, normal));
        var jt = vec3<f32>(tangent.x, tangent.y, cross2d(r_a, tangent));
        if is_b {
            jn = vec3<f32>(-normal.x, -normal.y, -cross2d(r_b, normal));
            jt = vec3<f32>(-tangent.x, -tangent.y, -cross2d(r_b, tangent));
        }

        a00 = a00 + k_n * jn.x * jn.x + k_t * jt.x * jt.x;
        a01 = a01 + k_n * jn.x * jn.y + k_t * jt.x * jt.y;
        a02 = a02 + k_n * jn.x * jn.z + k_t * jt.x * jt.z;
        a11 = a11 + k_n * jn.y * jn.y + k_t * jt.y * jt.y;
        a12 = a12 + k_n * jn.y * jn.z + k_t * jt.y * jt.z;
        a22 = a22 + k_n * jn.z * jn.z + k_t * jt.z * jt.z;

        rhs = rhs + jn * f_n + jt * f_t;
    }

    var delta = solve_sym_3x3(a00, a01, a02, a11, a12, a22, rhs);

    // Clamp corrections to prevent explosion from overcorrection.
    let max_lin_corr = 2.0;
    let lin_corr_len = length(delta.xy);
    if lin_corr_len > max_lin_corr {
        delta = vec3<f32>(delta.xy * (max_lin_corr / lin_corr_len), delta.z);
    }
    let max_ang_corr = 1.0;
    delta.z = clamp(delta.z, -max_ang_corr, max_ang_corr);

    bodies[body_idx].position_inv_mass = vec4<f32>(pos - delta.xy, angle - delta.z, inv_mass);
}
"#;

/// WGSL source for the 2D AVBD dual update.
///
/// The dual pass refreshes the cached contact forces after each primal sweep and
/// grows penalties only while rows remain active.
pub const AVBD_DUAL_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>,
    lin_vel:           vec4<f32>,
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct Contact2D {
    point:           vec4<f32>,
    normal:          vec4<f32>,
    local_anchors:   vec4<f32>,
    lambda_penalty:  vec4<f32>,
    body_a:          u32,
    body_b:          u32,
    feature_id:      u32,
    flags:           u32,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

const CONTACT_FLAG_STICKING: u32 = 1u;

@group(0) @binding(0) var<storage, read>       bodies:            array<Body2D>;
@group(0) @binding(1) var<storage, read_write> contacts:          array<Contact2D>;
@group(0) @binding(2) var<uniform>             params:            SimParams2D;
@group(0) @binding(3) var<storage, read>       contact_count_buf: array<u32>;

fn rotate2d(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let ca = cos(angle);
    let sa = sin(angle);
    return vec2<f32>(ca * v.x - sa * v.y, sa * v.x + ca * v.y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let num_contacts = contact_count_buf[0];
    let _keep_params = params.solver.x;
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let pos_a = bodies[c.body_a].position_inv_mass.xy;
    let pos_b = bodies[c.body_b].position_inv_mass.xy;
    let angle_a = bodies[c.body_a].position_inv_mass.z;
    let angle_b = bodies[c.body_b].position_inv_mass.z;
    let r_a = rotate2d(c.local_anchors.xy, angle_a);
    let r_b = rotate2d(c.local_anchors.zw, angle_b);
    let world_a = pos_a + r_a;
    let world_b = pos_b + r_b;
    let separation = world_a - world_b;

    let normal = c.normal.xy;
    let tangent = c.normal.zw;
    let c_n = dot(normal, separation);
    let c_t = dot(tangent, separation);

    let lambda_n_old = c.lambda_penalty.x;
    let lambda_t_old = c.lambda_penalty.y;
    let k_n = c.lambda_penalty.z;
    let k_t = c.lambda_penalty.w;
    let mu = sqrt(max(bodies[c.body_a]._pad0.x * bodies[c.body_b]._pad0.x, 0.0));
    let beta = params.solver.y;
    let max_penalty = params.solver.w;

    let lambda_n = min(k_n * c_n + lambda_n_old, 0.0);
    let tang_limit = mu * abs(lambda_n);
    let lambda_t = clamp(k_t * c_t + lambda_t_old, -tang_limit, tang_limit);

    var next_k_n = k_n;
    if lambda_n < -1e-6 && c_n < -1e-5 {
        next_k_n = min(k_n + beta * abs(c_n), max_penalty);
    }

    var next_k_t = k_t;
    var flags = 0u;
    if tang_limit > 0.0 && abs(lambda_t) < tang_limit * 0.98 && abs(c_t) < 2e-3 {
        flags = CONTACT_FLAG_STICKING;
        next_k_t = min(k_t + beta * abs(c_t), max_penalty);
    }

    contacts[ci].point = vec4<f32>((world_a + world_b) * 0.5, c_n, 0.0);
    contacts[ci].lambda_penalty = vec4<f32>(lambda_n, lambda_t, next_k_n, next_k_t);
    contacts[ci].flags = flags;
}
"#;
