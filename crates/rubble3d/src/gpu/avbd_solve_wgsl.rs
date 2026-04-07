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
    inertia_row0:     vec4<f32>,
    inertia_row1:     vec4<f32>,
    inertia_row2:     vec4<f32>,
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

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

struct SolveRange {
    offset: u32,
    count:  u32,
};

struct Solve6 {
    lin: vec3<f32>,
    ang: vec3<f32>,
};

const CONTACT_MARGIN: f32 = 0.01;
const BODY_LANES: u32 = 64u;

@group(0) @binding(0) var<storage, read_write> bodies:               array<Body>;
@group(0) @binding(1) var<storage, read>       inertial_states:      array<Body>;
@group(0) @binding(2) var<storage, read>       props:                array<BodyProps>;
@group(0) @binding(3) var<storage, read>       contacts:             array<Contact>;
@group(0) @binding(4) var<storage, read>       body_order:           array<u32>;
@group(0) @binding(5) var<uniform>             params:               SimParams;
@group(0) @binding(6) var<storage, read>       body_contact_ranges:  array<vec2<u32>>;
@group(0) @binding(7) var<storage, read>       body_contact_indices: array<u32>;
@group(0) @binding(8) var<uniform>             solve_range:          SolveRange;

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

fn mat3_from_inertia(p: BodyProps) -> mat3x3<f32> {
    return mat3x3<f32>(p.inertia_row0.xyz, p.inertia_row1.xyz, p.inertia_row2.xyz);
}

fn world_inertia(p: BodyProps, q: vec4<f32>) -> mat3x3<f32> {
    let r = quat_to_mat3(q);
    return r * mat3_from_inertia(p) * transpose(r);
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

// Lower-triangle packed index: lt[r*(r+1)/2 + c] for r >= c
// (0,0)=0  (1,0)=1 (1,1)=2  (2,0)=3 (2,1)=4 (2,2)=5
// (3,0)=6 (3,1)=7 (3,2)=8 (3,3)=9  (4,0)=10 .. (4,4)=14
// (5,0)=15 .. (5,5)=20   Total: 21 elements

fn solve_6x6_lt(lt: array<f32, 21>, rhs: array<f32, 6>) -> Solve6 {
    let eps = 1e-9;

    let A11 = lt[0];
    let A21 = lt[1];
    let A22 = lt[2];
    let A31 = lt[3];
    let A32 = lt[4];
    let A33 = lt[5];
    let A41 = lt[6];
    let A42 = lt[7];
    let A43 = lt[8];
    let A44 = lt[9];
    let A51 = lt[10];
    let A52 = lt[11];
    let A53 = lt[12];
    let A54 = lt[13];
    let A55 = lt[14];
    let A61 = lt[15];
    let A62 = lt[16];
    let A63 = lt[17];
    let A64 = lt[18];
    let A65 = lt[19];
    let A66 = lt[20];

    if A11 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let D1 = A11;
    let L21 = A21 / D1;
    let L31 = A31 / D1;
    let L41 = A41 / D1;
    let L51 = A51 / D1;
    let L61 = A61 / D1;

    let D2 = A22 - L21 * L21 * D1;
    if D2 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let L32 = (A32 - L21 * L31 * D1) / D2;
    let L42 = (A42 - L21 * L41 * D1) / D2;
    let L52 = (A52 - L21 * L51 * D1) / D2;
    let L62 = (A62 - L21 * L61 * D1) / D2;

    let D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
    if D3 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
    let L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
    let L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;

    let D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
    if D4 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
    let L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;

    let D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
    if D5 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;

    let D6 = A66 - (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);
    if D6 <= eps {
        return Solve6(vec3<f32>(0.0), vec3<f32>(0.0));
    }

    let y1 = rhs[0];
    let y2 = rhs[1] - L21 * y1;
    let y3 = rhs[2] - L31 * y1 - L32 * y2;
    let y4 = rhs[3] - L41 * y1 - L42 * y2 - L43 * y3;
    let y5 = rhs[4] - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
    let y6 = rhs[5] - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;

    let z1 = y1 / D1;
    let z2 = y2 / D2;
    let z3 = y3 / D3;
    let z4 = y4 / D4;
    let z5 = y5 / D5;
    let z6 = y6 / D6;

    let x6 = z6;
    let x5 = z5 - L65 * x6;
    let x4 = z4 - L54 * x5 - L64 * x6;
    let x3 = z3 - L43 * x4 - L53 * x5 - L63 * x6;
    let x2 = z2 - L32 * x3 - L42 * x4 - L52 * x5 - L62 * x6;
    let x1 = z1 - L21 * x2 - L31 * x3 - L41 * x4 - L51 * x5 - L61 * x6;

    return Solve6(vec3<f32>(x1, x2, x3), vec3<f32>(x4, x5, x6));
}

fn accumulate_row(
    j_lin: vec3<f32>,
    j_ang: vec3<f32>,
    stiffness: f32,
    force: f32,
    lt: ptr<function, array<f32, 21>>,
    rhs: ptr<function, array<f32, 6>>,
) {
    (*rhs)[0] += j_lin.x * force;
    (*rhs)[1] += j_lin.y * force;
    (*rhs)[2] += j_lin.z * force;
    (*rhs)[3] += j_ang.x * force;
    (*rhs)[4] += j_ang.y * force;
    (*rhs)[5] += j_ang.z * force;

    let sj0 = stiffness * j_lin.x;
    let sj1 = stiffness * j_lin.y;
    let sj2 = stiffness * j_lin.z;
    let sj3 = stiffness * j_ang.x;
    let sj4 = stiffness * j_ang.y;
    let sj5 = stiffness * j_ang.z;

    // Lower triangle only (21 FMAs instead of 36)
    // Row 0: (0,0)
    (*lt)[0]  += sj0 * j_lin.x;
    // Row 1: (1,0) (1,1)
    (*lt)[1]  += sj1 * j_lin.x; (*lt)[2]  += sj1 * j_lin.y;
    // Row 2: (2,0) (2,1) (2,2)
    (*lt)[3]  += sj2 * j_lin.x; (*lt)[4]  += sj2 * j_lin.y; (*lt)[5]  += sj2 * j_lin.z;
    // Row 3: (3,0) (3,1) (3,2) (3,3)
    (*lt)[6]  += sj3 * j_lin.x; (*lt)[7]  += sj3 * j_lin.y; (*lt)[8]  += sj3 * j_lin.z;
    (*lt)[9]  += sj3 * j_ang.x;
    // Row 4: (4,0) (4,1) (4,2) (4,3) (4,4)
    (*lt)[10] += sj4 * j_lin.x; (*lt)[11] += sj4 * j_lin.y; (*lt)[12] += sj4 * j_lin.z;
    (*lt)[13] += sj4 * j_ang.x; (*lt)[14] += sj4 * j_ang.y;
    // Row 5: (5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
    (*lt)[15] += sj5 * j_lin.x; (*lt)[16] += sj5 * j_lin.y; (*lt)[17] += sj5 * j_lin.z;
    (*lt)[18] += sj5 * j_ang.x; (*lt)[19] += sj5 * j_ang.y; (*lt)[20] += sj5 * j_ang.z;
}

@compute @workgroup_size(BODY_LANES)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }

    let body_idx = body_order[solve_range.offset + local_idx];
    let body = bodies[body_idx];
    let inv_mass = body.position_inv_mass.w;
    if inv_mass <= 0.0 {
        return;
    }

    let body_props = props[body_idx];
    let inertial_body = inertial_states[body_idx];
    let q = quat_normalize(body.orientation);
    let q_inertial = quat_normalize(inertial_body.orientation);
    let pos = body.position_inv_mass.xyz;
    let pos_inertial = inertial_body.position_inv_mass.xyz;
    let rot_delta = rotation_delta_vec(q, q_inertial);
    let dt = params.solver.x;
    let dt2 = dt * dt;
    let mass = 1.0 / inv_mass;
    let i_world = world_inertia(body_props, q);

    // Lower-triangle packed matrix (21 elements instead of 36)
    var local_lt = array<f32, 21>(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    var local_rhs = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    let m_dt2 = mass / dt2;
    // Diagonal: (0,0)=0, (1,1)=2, (2,2)=5
    local_lt[0] = m_dt2;
    local_lt[2] = m_dt2;
    local_lt[5] = m_dt2;
    local_rhs[0] = m_dt2 * (pos.x - pos_inertial.x);
    local_rhs[1] = m_dt2 * (pos.y - pos_inertial.y);
    local_rhs[2] = m_dt2 * (pos.z - pos_inertial.z);

    // Inertia block: (3,3)=9, (4,3)=13, (4,4)=14, (5,3)=18, (5,4)=19, (5,5)=20
    local_lt[9]  = i_world[0][0] / dt2;
    local_lt[13] = i_world[1][0] / dt2;
    local_lt[14] = i_world[1][1] / dt2;
    local_lt[18] = i_world[2][0] / dt2;
    local_lt[19] = i_world[2][1] / dt2;
    local_lt[20] = i_world[2][2] / dt2;
    local_rhs[3] =
        (i_world[0][0] * rot_delta.x + i_world[0][1] * rot_delta.y + i_world[0][2] * rot_delta.z) / dt2;
    local_rhs[4] =
        (i_world[1][0] * rot_delta.x + i_world[1][1] * rot_delta.y + i_world[1][2] * rot_delta.z) / dt2;
    local_rhs[5] =
        (i_world[2][0] * rot_delta.x + i_world[2][1] * rot_delta.y + i_world[2][2] * rot_delta.z) / dt2;

    let contact_range = body_contact_ranges[body_idx];
    let range_end = contact_range.x + contact_range.y;
    for (var slot = contact_range.x; slot < range_end; slot = slot + 1u) {
        let c = contacts[body_contact_indices[slot]];
        let is_a = c.body_a == body_idx;
        let is_b = c.body_b == body_idx;
        if !is_a && !is_b {
            continue;
        }

        let body_a = bodies[c.body_a];
        let body_b = bodies[c.body_b];
        let q_a = quat_normalize(body_a.orientation);
        let q_b = quat_normalize(body_b.orientation);
        let r_a = quat_rotate(q_a, c.local_anchor_a.xyz);
        let r_b = quat_rotate(q_b, c.local_anchor_b.xyz);
        let world_a = body_a.position_inv_mass.xyz + r_a;
        let world_b = body_b.position_inv_mass.xyz + r_b;
        let separation = world_a - world_b;
        let normal = c.normal.xyz;
        let tangent1 = c.tangent.xyz;
        let tangent2 = cross(normal, tangent1);
        let lambda_n = c.lambda.x;
        let lambda_t1 = c.lambda.y;
        let lambda_t2 = c.lambda.z;
        let k_n = c.penalty.x;
        let k_t1 = c.penalty.y;
        let k_t2 = c.penalty.z;
        let mu = sqrt(max(props[c.body_a].friction * props[c.body_b].friction, 0.0));
        let c_n = dot(normal, separation) + CONTACT_MARGIN;
        let c_t1 = dot(tangent1, separation);
        let c_t2 = dot(tangent2, separation);
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

        accumulate_row(jn_lin, jn_ang, k_n, f_n, &local_lt, &local_rhs);
        accumulate_row(jt1_lin, jt1_ang, k_t1, tang.x, &local_lt, &local_rhs);
        accumulate_row(jt2_lin, jt2_ang, k_t2, tang.y, &local_lt, &local_rhs);
    }

    let solution = solve_6x6_lt(local_lt, local_rhs);
    bodies[body_idx].position_inv_mass = vec4<f32>(pos - solution.lin, inv_mass);
    bodies[body_idx].orientation = quat_normalize(quat_mul(small_angle_quat(-solution.ang), q));
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
    inertia_row0:     vec4<f32>,
    inertia_row1:     vec4<f32>,
    inertia_row2:     vec4<f32>,
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
const CONTACT_MARGIN: f32 = 0.01;

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       bodies:            array<Body>;
@group(0) @binding(1) var<storage, read>       props:             array<BodyProps>;
@group(0) @binding(2) var<storage, read_write> contacts:          array<Contact>;
@group(0) @binding(3) var<uniform>             params:            SimParams;
@group(0) @binding(4) var<storage, read>       contact_count_buf: array<u32>;

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
    let q_a = quat_normalize(bodies[c.body_a].orientation);
    let q_b = quat_normalize(bodies[c.body_b].orientation);
    let r_a = quat_rotate(q_a, c.local_anchor_a.xyz);
    let r_b = quat_rotate(q_b, c.local_anchor_b.xyz);
    let world_a = pos_a + r_a;
    let world_b = pos_b + r_b;
    let separation = world_a - world_b;
    let normal = c.normal.xyz;
    let tangent1 = c.tangent.xyz;
    let tangent2 = cross(normal, tangent1);
    let geom_c_n = dot(normal, separation);
    let c_n = geom_c_n + CONTACT_MARGIN;
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
    if lambda_n < -1e-6 && geom_c_n < -1e-5 {
        next_penalty.x = min(c.penalty.x + beta * abs(geom_c_n), max_penalty);
    }

    var flags = 0u;
    if tang_limit > 0.0 && tang_len < tang_limit * 0.98 && max(abs(c_t1), abs(c_t2)) < 2e-3 {
        flags = CONTACT_FLAG_STICKING;
        next_penalty.y = min(c.penalty.y + beta * abs(c_t1), max_penalty);
        next_penalty.z = min(c.penalty.z + beta * abs(c_t2), max_penalty);
    }

    contacts[ci].point = vec4<f32>((world_a + world_b) * 0.5, geom_c_n);
    contacts[ci].lambda = vec4<f32>(lambda_n, tang.x, tang.y, 0.0);
    contacts[ci].penalty = next_penalty;
    contacts[ci].flags = flags;
}
"#;
