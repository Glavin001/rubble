/// WGSL source for the 3D AVBD primal solve.
pub const AVBD_PRIMAL_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

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
    point:          vec4<f32>, // (x, y, z, depth)
    normal:         vec4<f32>, // (nx, ny, nz, 0)
    tangent:        vec4<f32>, // tangent 1
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    lambda:         vec4<f32>, // (lambda_n, lambda_t1, lambda_t2, 0)
    penalty:        vec4<f32>, // (k_n, k_t1, k_t2, 0)
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

struct SolveRange {
    offset: u32,
    count:  u32,
};

struct Solve6 {
    lin: vec3<f32>,
    ang: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:            array<Body>;
@group(0) @binding(1) var<storage, read>       inertial_states:   array<Body>;
@group(0) @binding(2) var<storage, read>       props:             array<BodyProps>;
@group(0) @binding(3) var<storage, read>       contacts:          array<Contact>;
@group(0) @binding(4) var<storage, read>       body_order:        array<u32>;
@group(0) @binding(5) var<uniform>             params:            SimParams;
@group(0) @binding(6) var<storage, read>       contact_count_buf: array<u32>;
@group(0) @binding(7) var<uniform>             solve_range:       SolveRange;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    return q / max(length(q), 1e-12);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
    );
}

fn mat3_from_props(p: BodyProps) -> mat3x3<f32> {
    return mat3x3<f32>(p.inv_inertia_row0.xyz, p.inv_inertia_row1.xyz, p.inv_inertia_row2.xyz);
}

fn mat3_transpose(m: mat3x3<f32>) -> mat3x3<f32> {
    return transpose(m);
}

fn world_inv_inertia(p: BodyProps, q: vec4<f32>) -> mat3x3<f32> {
    let r = quat_to_mat3(q);
    let local = mat3_from_props(p);
    return r * local * mat3_transpose(r);
}

fn mat3_inverse_safe(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0].x;
    let d = m[0].y;
    let g = m[0].z;
    let b = m[1].x;
    let e = m[1].y;
    let h = m[1].z;
    let c = m[2].x;
    let f = m[2].y;
    let i = m[2].z;

    let det_m =
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g);
    if abs(det_m) < 1e-10 {
        return mat3x3<f32>(
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
        );
    }
    let inv_det = 1.0 / det_m;
    return mat3x3<f32>(
        vec3<f32>((e * i - f * h) * inv_det, (f * g - d * i) * inv_det, (d * h - e * g) * inv_det),
        vec3<f32>((c * h - b * i) * inv_det, (a * i - c * g) * inv_det, (b * g - a * h) * inv_det),
        vec3<f32>((b * f - c * e) * inv_det, (c * d - a * f) * inv_det, (a * e - b * d) * inv_det),
    );
}

fn small_angle_quat(theta: vec3<f32>) -> vec4<f32> {
    return quat_normalize(vec4<f32>(0.5 * theta, 1.0));
}

fn rotation_delta_vec(current: vec4<f32>, reference: vec4<f32>) -> vec3<f32> {
    var dq = quat_mul(current, quat_conj(reference));
    if dq.w < 0.0 {
        dq = -dq;
    }
    let imag_len = length(dq.xyz);
    if imag_len < 1e-8 {
        return 2.0 * dq.xyz;
    }
    let angle = 2.0 * atan2(imag_len, dq.w);
    return dq.xyz / imag_len * angle;
}

fn solve_6x6(m: array<array<f32, 6>, 6>, rhs: array<f32, 6>) -> Solve6 {
    var aug: array<array<f32, 7>, 6>;
    for (var i = 0u; i < 6u; i = i + 1u) {
        for (var j = 0u; j < 6u; j = j + 1u) {
            aug[i][j] = m[i][j];
        }
        aug[i][6] = rhs[i];
    }

    for (var k = 0u; k < 6u; k = k + 1u) {
        var pivot = k;
        var pivot_abs = abs(aug[k][k]);
        for (var i = k + 1u; i < 6u; i = i + 1u) {
            let cand = abs(aug[i][k]);
            if cand > pivot_abs {
                pivot = i;
                pivot_abs = cand;
            }
        }

        if pivot_abs < 1e-9 {
            return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
        }

        if pivot != k {
            let tmp = aug[k];
            aug[k] = aug[pivot];
            aug[pivot] = tmp;
        }

        let inv_pivot = 1.0 / aug[k][k];
        for (var j = k; j < 7u; j = j + 1u) {
            aug[k][j] = aug[k][j] * inv_pivot;
        }

        for (var i = 0u; i < 6u; i = i + 1u) {
            if i == k {
                continue;
            }
            let factor = aug[i][k];
            if abs(factor) < 1e-12 {
                continue;
            }
            for (var j = k; j < 7u; j = j + 1u) {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    return Solve6(
        vec3<f32>(aug[0][6], aug[1][6], aug[2][6]),
        vec3<f32>(aug[3][6], aug[4][6], aug[5][6]),
    );
}

fn accumulate_row(
    j_lin: vec3<f32>,
    j_ang: vec3<f32>,
    stiffness: f32,
    force: f32,
    m: ptr<function, array<array<f32, 6>, 6>>,
    rhs: ptr<function, array<f32, 6>>,
) {
    let j = array<f32, 6>(j_lin.x, j_lin.y, j_lin.z, j_ang.x, j_ang.y, j_ang.z);
    for (var r = 0u; r < 6u; r = r + 1u) {
        (*rhs)[r] = (*rhs)[r] + j[r] * force;
        for (var c = 0u; c < 6u; c = c + 1u) {
            (*m)[r][c] = (*m)[r][c] + stiffness * j[r] * j[c];
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }

    let body_idx = body_order[solve_range.offset + local_idx];
    let inv_mass = bodies[body_idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return;
    }

    let q = quat_normalize(bodies[body_idx].orientation);
    let q_inertial = quat_normalize(inertial_states[body_idx].orientation);
    let pos = bodies[body_idx].position_inv_mass.xyz;
    let pos_inertial = inertial_states[body_idx].position_inv_mass.xyz;
    let rot_delta = rotation_delta_vec(q, q_inertial);
    let dt = params.solver.x;
    let dt2 = dt * dt;
    let mass = 1.0 / inv_mass;
    let inv_i_world = world_inv_inertia(props[body_idx], q);
    let i_world = mat3_inverse_safe(inv_i_world);

    var mtx: array<array<f32, 6>, 6>;
    var rhs: array<f32, 6>;
    for (var r = 0u; r < 6u; r = r + 1u) {
        rhs[r] = 0.0;
        for (var c = 0u; c < 6u; c = c + 1u) {
            mtx[r][c] = 0.0;
        }
    }

    mtx[0][0] = mass / dt2;
    mtx[1][1] = mass / dt2;
    mtx[2][2] = mass / dt2;
    rhs[0] = mtx[0][0] * (pos.x - pos_inertial.x);
    rhs[1] = mtx[1][1] * (pos.y - pos_inertial.y);
    rhs[2] = mtx[2][2] * (pos.z - pos_inertial.z);

    mtx[3][3] = i_world[0][0] / dt2;
    mtx[3][4] = i_world[0][1] / dt2;
    mtx[3][5] = i_world[0][2] / dt2;
    mtx[4][3] = i_world[1][0] / dt2;
    mtx[4][4] = i_world[1][1] / dt2;
    mtx[4][5] = i_world[1][2] / dt2;
    mtx[5][3] = i_world[2][0] / dt2;
    mtx[5][4] = i_world[2][1] / dt2;
    mtx[5][5] = i_world[2][2] / dt2;
    rhs[3] = (i_world[0][0] * rot_delta.x + i_world[0][1] * rot_delta.y + i_world[0][2] * rot_delta.z) / dt2;
    rhs[4] = (i_world[1][0] * rot_delta.x + i_world[1][1] * rot_delta.y + i_world[1][2] * rot_delta.z) / dt2;
    rhs[5] = (i_world[2][0] * rot_delta.x + i_world[2][1] * rot_delta.y + i_world[2][2] * rot_delta.z) / dt2;

    let num_contacts = contact_count_buf[0];
    for (var ci = 0u; ci < num_contacts; ci = ci + 1u) {
        let c = contacts[ci];
        let is_a = c.body_a == body_idx;
        let is_b = c.body_b == body_idx;
        if !is_a && !is_b {
            continue;
        }

        let pos_a = bodies[c.body_a].position_inv_mass.xyz;
        let pos_b = bodies[c.body_b].position_inv_mass.xyz;
        let q_a = quat_normalize(bodies[c.body_a].orientation);
        let q_b = quat_normalize(bodies[c.body_b].orientation);
        let r_a = quat_rotate(q_a, c.local_anchor_a.xyz);
        let r_b = quat_rotate(q_b, c.local_anchor_b.xyz);
        let world_a = pos_a + r_a;
        let world_b = pos_b + r_b;
        let separation = world_a - world_b;
        let normal = c.normal.xyz;
        let tangent1 = normalize(c.tangent.xyz);
        let tangent2 = normalize(cross(normal, tangent1));
        let c_n = dot(normal, separation);
        let c_t1 = dot(tangent1, separation);
        let c_t2 = dot(tangent2, separation);
        let lambda_n = c.lambda.x;
        let lambda_t1 = c.lambda.y;
        let lambda_t2 = c.lambda.z;
        let k_n = c.penalty.x;
        let k_t1 = c.penalty.y;
        let k_t2 = c.penalty.z;
        let mu = sqrt(max(props[c.body_a].friction * props[c.body_b].friction, 0.0));
        let f_n = min(k_n * c_n + lambda_n, 0.0);
        var tang = vec2<f32>(k_t1 * c_t1 + lambda_t1, k_t2 * c_t2 + lambda_t2);
        let tang_limit = mu * abs(f_n);
        let tang_len = length(tang);
        if tang_len > tang_limit && tang_len > 1e-8 {
            tang = tang * (tang_limit / tang_len);
        }

        var jn_lin = normal;
        var jt1_lin = tangent1;
        var jt2_lin = tangent2;
        var jn_ang = cross(r_a, normal);
        var jt1_ang = cross(r_a, tangent1);
        var jt2_ang = cross(r_a, tangent2);
        if is_b {
            jn_lin = -normal;
            jt1_lin = -tangent1;
            jt2_lin = -tangent2;
            jn_ang = -cross(r_b, normal);
            jt1_ang = -cross(r_b, tangent1);
            jt2_ang = -cross(r_b, tangent2);
        }

        accumulate_row(jn_lin, jn_ang, k_n, f_n, &mtx, &rhs);
        accumulate_row(jt1_lin, jt1_ang, k_t1, tang.x, &mtx, &rhs);
        accumulate_row(jt2_lin, jt2_ang, k_t2, tang.y, &mtx, &rhs);
    }

    let solution = solve_6x6(mtx, rhs);
    bodies[body_idx].position_inv_mass = vec4<f32>(pos - solution.lin, inv_mass);
    bodies[body_idx].orientation = quat_mul(small_angle_quat(-solution.ang), q);
}
"#;

/// WGSL source for the 3D AVBD dual update.
pub const AVBD_DUAL_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

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
    point:          vec4<f32>,
    normal:         vec4<f32>,
    tangent:        vec4<f32>,
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

const CONTACT_FLAG_STICKING: u32 = 1u;

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       bodies:            array<Body>;
@group(0) @binding(1) var<storage, read>       props:             array<BodyProps>;
@group(0) @binding(2) var<storage, read_write> contacts:          array<Contact>;
@group(0) @binding(3) var<uniform>             params:            SimParams;
@group(0) @binding(4) var<storage, read>       contact_count_buf: array<u32>;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let num_contacts = contact_count_buf[0];
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let pos_a = bodies[c.body_a].position_inv_mass.xyz;
    let pos_b = bodies[c.body_b].position_inv_mass.xyz;
    let q_a = normalize(bodies[c.body_a].orientation);
    let q_b = normalize(bodies[c.body_b].orientation);
    let r_a = quat_rotate(q_a, c.local_anchor_a.xyz);
    let r_b = quat_rotate(q_b, c.local_anchor_b.xyz);
    let world_a = pos_a + r_a;
    let world_b = pos_b + r_b;
    let separation = world_a - world_b;
    let normal = c.normal.xyz;
    let tangent1 = normalize(c.tangent.xyz);
    let tangent2 = normalize(cross(normal, tangent1));

    let c_n = dot(normal, separation);
    let c_t1 = dot(tangent1, separation);
    let c_t2 = dot(tangent2, separation);
    let lambda_n = min(c.penalty.x * c_n + c.lambda.x, 0.0);
    var tang = vec2<f32>(
        c.penalty.y * c_t1 + c.lambda.y,
        c.penalty.z * c_t2 + c.lambda.z,
    );
    let mu = sqrt(max(props[c.body_a].friction * props[c.body_b].friction, 0.0));
    let tang_limit = mu * abs(lambda_n);
    let beta = params.solver.y;
    let max_penalty = params.solver.w;
    let tang_len = length(tang);
    if tang_len > tang_limit && tang_len > 1e-8 {
        tang = tang * (tang_limit / tang_len);
    }

    var next_penalty = c.penalty;
    if lambda_n < -1e-6 && c_n < -1e-5 {
        next_penalty.x = min(c.penalty.x + beta * abs(c_n), max_penalty);
    }

    var flags = 0u;
    if tang_limit > 0.0 && tang_len < tang_limit * 0.98 && max(abs(c_t1), abs(c_t2)) < 2e-3 {
        flags = CONTACT_FLAG_STICKING;
        next_penalty.y = min(c.penalty.y + beta * abs(c_t1), max_penalty);
        next_penalty.z = min(c.penalty.z + beta * abs(c_t2), max_penalty);
    }

    contacts[ci].point = vec4<f32>((world_a + world_b) * 0.5, c_n);
    contacts[ci].lambda = vec4<f32>(lambda_n, tang.x, tang.y, 0.0);
    contacts[ci].penalty = next_penalty;
    contacts[ci].flags = flags;
}
"#;
