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
    normal:         vec4<f32>, // (nx, ny, nz, C0_n frozen at solve start)
    tangent:        vec4<f32>, // (t1x, t1y, t1z, v_n_pre for restitution)
    local_anchor_a: vec4<f32>, // (ax, ay, az, C0_t1 frozen at solve start)
    local_anchor_b: vec4<f32>, // (bx, by, bz, C0_t2 frozen at solve start)
    lambda:         vec4<f32>, // (lambda_n, lambda_t1, lambda_t2, 0)
    penalty:        vec4<f32>, // (k_n, k_t1, k_t2, 0)
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,  // (dt, beta, k_start, k_max_normal)
    counts:  vec4<u32>,
    quality: vec4<f32>,  // (contact_offset, restitution_threshold, penetration_slop, alpha_stab)
    solver2: vec4<f32>,  // (k_max_tangential, max_speed, max_ang_speed, dynamic_mu_scale)
};

struct SolveRange {
    // x = color index into the 32-entry (offset,count) header at the head of
    // body_order (written on GPU by the color compaction pass); y unused.
    offset: u32,
    count:  u32,
};

struct Solve6 {
    lin: vec3<f32>,
    ang: vec3<f32>,
};

const CONTACT_MARGIN: f32 = 0.01;
const CONTACT_FLAG_STICKING: u32 = 1u;
// Upper bound on the bounded-augmented-Lagrangian contact force magnitude. The
// effective per-contact cap is scaled by the lighter body's mass so a light body
// cannot stockpile arbitrarily large forces (mirrors avbd-metal's lamCap).
const LAMBDA_MAX: f32 = 1.0e6;
// Max penetration-recovery speed (m/s). Deep initial overlap (e.g. bodies spawned
// intersecting) is recovered at no more than this rate, so it separates gently
// instead of launching. Resisting *further* penetration is unaffected.
const MAX_RECOVERY_VEL: f32 = 6.0;

@group(0) @binding(0) var<storage, read_write> bodies:            array<Body>;
@group(0) @binding(1) var<storage, read>       inertial_states:   array<Body>;
@group(0) @binding(2) var<storage, read>       props:             array<BodyProps>;
@group(0) @binding(3) var<storage, read_write> contacts:          array<Contact>;
@group(0) @binding(4) var<storage, read>       body_order:        array<u32>;
@group(0) @binding(5) var<uniform>             params:            SimParams;
@group(0) @binding(6) var<storage, read>       body_contact_ranges: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read>       body_contact_indices: array<u32>;
@group(0) @binding(8) var<uniform>             solve_range:       SolveRange;
@group(0) @binding(9) var<storage, read>       solve_start_states: array<Body>;

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

// Jacobi-preconditioned LDL^T solve of the (SPD by construction) 6x6 system.
// The raw system mixes m/dt^2 mass terms with penalty stiffnesses up to 1e9+,
// so it is scaled to unit diagonal first (A' = S A S with S = diag(a_ii)^-1/2),
// factored with a pivot floor, and unscaled on the way out. A small relative
// diagonal regularization keeps near-singular geometry (e.g. all contacts
// coplanar through the center of mass) well-posed instead of returning zero.
fn solve_6x6(m: array<array<f32, 6>, 6>, rhs: array<f32, 6>) -> Solve6 {
    var a: array<array<f32, 6>, 6>;
    var b: array<f32, 6>;
    var s: array<f32, 6>;

    for (var i = 0u; i < 6u; i = i + 1u) {
        s[i] = 1.0 / sqrt(max(m[i][i], 1e-12));
    }
    for (var i = 0u; i < 6u; i = i + 1u) {
        for (var j = 0u; j < 6u; j = j + 1u) {
            a[i][j] = m[i][j] * s[i] * s[j];
        }
        a[i][i] = a[i][i] + 1e-6; // tiny anti-singular nudge on the unit diagonal
        b[i] = rhs[i] * s[i];
    }

    // LDL^T: L unit lower triangular in-place (strict lower), D on the diagonal.
    for (var k = 0u; k < 6u; k = k + 1u) {
        var d = a[k][k];
        for (var p = 0u; p < k; p = p + 1u) {
            d = d - a[k][p] * a[k][p] * a[p][p];
        }
        a[k][k] = max(d, 1e-6); // pivot floor: SPD input, so clamp instead of bail
        for (var i = k + 1u; i < 6u; i = i + 1u) {
            var v = a[i][k];
            for (var p = 0u; p < k; p = p + 1u) {
                v = v - a[i][p] * a[k][p] * a[p][p];
            }
            a[i][k] = v / a[k][k];
        }
    }

    // Forward solve L y = b.
    for (var i = 0u; i < 6u; i = i + 1u) {
        var v = b[i];
        for (var p = 0u; p < i; p = p + 1u) {
            v = v - a[i][p] * b[p];
        }
        b[i] = v;
    }
    // Diagonal solve D z = y.
    for (var i = 0u; i < 6u; i = i + 1u) {
        b[i] = b[i] / a[i][i];
    }
    // Backward solve L^T x = z.
    for (var ii = 0u; ii < 6u; ii = ii + 1u) {
        let i = 5u - ii;
        var v = b[i];
        for (var p = i + 1u; p < 6u; p = p + 1u) {
            v = v - a[p][i] * b[p];
        }
        b[i] = v;
    }

    return Solve6(
        vec3<f32>(b[0] * s[0], b[1] * s[1], b[2] * s[2]),
        vec3<f32>(b[3] * s[3], b[4] * s[4], b[5] * s[5]),
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

// Stiffness rescaling (AVBD Eq. 14): once a force is clamped, the quadratic
// penalty model k*C no longer matches the actual (constant) force, which makes
// the Newton step overdamped. Use the secant stiffness |f|/|C| instead so the
// Hessian is consistent with the force the rhs actually applies.
fn effective_stiffness(k: f32, force: f32, c: f32, clamped: bool) -> f32 {
    if !clamped {
        return k;
    }
    return min(k, abs(force) / max(abs(c), 1e-9));
}

const ORDER_HEADER: u32 = 64u;

// One AVBD block-descent step for a single body (called per color; the
// persistent small-scene kernel drives the same function from its in-kernel
// color loop).
fn solve_body(body_idx: u32) {
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

    let contact_range = body_contact_ranges[body_idx];
    let range_end = contact_range.x + contact_range.y;
    for (var slot = contact_range.x; slot < range_end; slot = slot + 1u) {
        let c = contacts[body_contact_indices[slot]];
        let is_a = c.body_a == body_idx;
        let is_b = c.body_b == body_idx;
        if !is_a && !is_b {
            continue;
        }

        // Taylor-cached constraint (AVBD): geometry (anchors, normal, C0) was
        // frozen at solve start by the contact-prep pass. Each iteration only
        // evaluates C = C0 + J*dx against the frozen Jacobian, where dx is the
        // pose delta accumulated since solve start. The trust-region step caps
        // below bound the linearization error.
        let start_a = solve_start_states[c.body_a];
        let start_b = solve_start_states[c.body_b];
        let q_a0 = quat_normalize(start_a.orientation);
        let q_b0 = quat_normalize(start_b.orientation);
        let r_a = quat_rotate(q_a0, c.local_anchor_a.xyz);
        let r_b = quat_rotate(q_b0, c.local_anchor_b.xyz);

        let dpos_a = bodies[c.body_a].position_inv_mass.xyz - start_a.position_inv_mass.xyz;
        let dpos_b = bodies[c.body_b].position_inv_mass.xyz - start_b.position_inv_mass.xyz;
        let dth_a = rotation_delta_vec(quat_normalize(bodies[c.body_a].orientation), q_a0);
        let dth_b = rotation_delta_vec(quat_normalize(bodies[c.body_b].orientation), q_b0);
        let dmotion = (dpos_a + cross(dth_a, r_a)) - (dpos_b + cross(dth_b, r_b));

        let normal = c.normal.xyz;
        let c0_n = c.normal.w;   // frozen normal separation at solve start
        let c0_t1 = c.local_anchor_a.w;
        let c0_t2 = c.local_anchor_b.w;
        let tangent1 = normalize(c.tangent.xyz);
        let tangent2 = normalize(cross(normal, tangent1));
        let lambda_n = c.lambda.x;
        let lambda_t1 = c.lambda.y;
        let lambda_t2 = c.lambda.z;
        let k_n = c.penalty.x;
        let k_t1 = c.penalty.y;
        let k_t2 = c.penalty.z;
        // Alpha-regularized constraint target: C_reg = (1 - alpha)*C0 + J*dx.
        // Only correcting (1 - alpha) of the initial violation per step bleeds
        // penetration out gradually instead of injecting energy by snapping it
        // to zero in one step. The recovery term is additionally clamped so a deep
        // initial overlap recovers at most MAX_RECOVERY_VEL per step (J*dx, which
        // resists further penetration, is left unclamped).
        let alpha_stab = params.quality.w;
        let recovery_target = max((1.0 - alpha_stab) * (c0_n + CONTACT_MARGIN), -MAX_RECOVERY_VEL * dt);
        let c_n = recovery_target + dot(normal, dmotion);
        let c_t1 = c0_t1 + dot(tangent1, dmotion);
        let c_t2 = c0_t2 + dot(tangent2, dmotion);
        // Bounded augmented-Lagrangian: cap the normal force by the lighter body's mass.
        let inv_m_a = bodies[c.body_a].position_inv_mass.w;
        let inv_m_b = bodies[c.body_b].position_inv_mass.w;
        let mass_a = select(1e30, 1.0 / inv_m_a, inv_m_a > 0.0);
        let mass_b = select(1e30, 1.0 / inv_m_b, inv_m_b > 0.0);
        let lam_cap = min(LAMBDA_MAX, max(10.0, 1.0e5 * min(mass_a, mass_b)));
        let f_n_raw = k_n * c_n + lambda_n;
        let f_n = clamp(f_n_raw, -lam_cap, 0.0);
        // Normal row keeps full stiffness in the Hessian even when the force
        // clamps (one-sided force, two-sided stiffness): the phantom stiffness
        // damps the contact-boundary crossing. Rescaling it to the secant
        // (Eq. 14) is for *friction* rows only — applied to the normal row it
        // removes all normal support the moment a contact separates and the
        // stack limit-cycles.
        let k_n_eff = k_n;

        var tang = vec2<f32>(k_t1 * c_t1 + lambda_t1, k_t2 * c_t2 + lambda_t2);
        // Coulomb cone radius uses the larger of the instantaneous normal force
        // and the lagged accumulated normal dual. The lagged dual gives a stable
        // grip from iteration 0 on warm-started resting contacts (instead of the
        // volatile per-iteration f_n collapsing the cone), reducing slip/toppling.
        let mu_static = sqrt(max(props[c.body_a].friction * props[c.body_b].friction, 0.0));
        let mu_dynamic = mu_static * params.solver2.w;
        let f_n_ref = max(abs(f_n), abs(lambda_n));
        let tang_len = length(tang);
        var k_t1_eff = k_t1;
        var k_t2_eff = k_t2;
        if tang_len > mu_static * f_n_ref && tang_len > 1e-8 {
            // Sliding: clamp to the *dynamic* cone and rescale row stiffness so
            // the Hessian matches the now-constant friction force (Eq. 14).
            tang = tang * (mu_dynamic * f_n_ref / tang_len);
            k_t1_eff = effective_stiffness(k_t1, tang.x, c_t1, true);
            k_t2_eff = effective_stiffness(k_t2, tang.y, c_t2, true);
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

        accumulate_row(jn_lin, jn_ang, k_n_eff, f_n, &mtx, &rhs);
        accumulate_row(jt1_lin, jt1_ang, k_t1_eff, tang.x, &mtx, &rhs);
        accumulate_row(jt2_lin, jt2_ang, k_t2_eff, tang.y, &mtx, &rhs);
    }

    let solution = solve_6x6(mtx, rhs);

    // Trust-region step caps (VBD-style): bound a single Newton step so it cannot
    // overshoot or tunnel through a thin contact. The linear cap scales with the
    // body's bounding radius R = sqrt(1.5 * trace(I) / m) (exact for boxes,
    // conservative for spheres); the angular cap is a fixed ~0.5 rad (~29 deg).
    let trace_i = i_world[0][0] + i_world[1][1] + i_world[2][2];
    let bound_r = sqrt(max(1.5 * trace_i * inv_mass, 1e-8));
    let max_lin = 0.35 * bound_r;
    var dlin = solution.lin;
    var dang = solution.ang;
    let lin_len = length(dlin);
    if lin_len > max_lin && lin_len > 1e-12 {
        dlin = dlin * (max_lin / lin_len);
    }
    let ang_len = length(dang);
    if ang_len > 0.5 && ang_len > 1e-12 {
        dang = dang * (0.5 / ang_len);
    }

    bodies[body_idx].position_inv_mass = vec4<f32>(pos - dlin, inv_mass);
    bodies[body_idx].orientation = quat_mul(small_angle_quat(-dang), q);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    let color = solve_range.offset;
    let range_offset = body_order[2u * color];
    let range_count = body_order[2u * color + 1u];
    if local_idx >= range_count {
        return;
    }
    solve_body(body_order[ORDER_HEADER + range_offset + local_idx]);
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
    normal:         vec4<f32>,   // (n, C0_n)
    tangent:        vec4<f32>,   // (t1, v_n_pre)
    local_anchor_a: vec4<f32>,   // (anchor_a, C0_t1)
    local_anchor_b: vec4<f32>,   // (anchor_b, C0_t2)
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

const CONTACT_FLAG_STICKING: u32 = 1u;
const CONTACT_MARGIN: f32 = 0.01;
const LAMBDA_MAX: f32 = 1.0e6;
const MAX_RECOVERY_VEL: f32 = 6.0;

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,  // (dt, beta, k_start, k_max_normal)
    counts:  vec4<u32>,
    quality: vec4<f32>,  // (contact_offset, restitution_threshold, penetration_slop, alpha_stab)
    solver2: vec4<f32>,  // (k_max_tangential, max_speed, max_ang_speed, dynamic_mu_scale)
};

@group(0) @binding(0) var<storage, read>       bodies:            array<Body>;
@group(0) @binding(1) var<storage, read>       props:             array<BodyProps>;
@group(0) @binding(2) var<storage, read_write> contacts:          array<Contact>;
@group(0) @binding(3) var<uniform>             params:            SimParams;
@group(0) @binding(4) var<storage, read>       contact_count_buf: array<u32>;
@group(0) @binding(5) var<storage, read>       solve_start_states: array<Body>;

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

// >>> dual-contact-update
// Verbatim-shared with the persistent solve kernel via source extraction —
// see avbd_persistent_wgsl(); edit only here.
fn dual_update_contact(ci: u32) {
    let c = contacts[ci];

    // Same Taylor-cached constraint evaluation as the primal.
    let start_a = solve_start_states[c.body_a];
    let start_b = solve_start_states[c.body_b];
    let q_a0 = quat_normalize(start_a.orientation);
    let q_b0 = quat_normalize(start_b.orientation);
    let r_a = quat_rotate(q_a0, c.local_anchor_a.xyz);
    let r_b = quat_rotate(q_b0, c.local_anchor_b.xyz);
    let dpos_a = bodies[c.body_a].position_inv_mass.xyz - start_a.position_inv_mass.xyz;
    let dpos_b = bodies[c.body_b].position_inv_mass.xyz - start_b.position_inv_mass.xyz;
    let dth_a = rotation_delta_vec(quat_normalize(bodies[c.body_a].orientation), q_a0);
    let dth_b = rotation_delta_vec(quat_normalize(bodies[c.body_b].orientation), q_b0);
    let dmotion = (dpos_a + cross(dth_a, r_a)) - (dpos_b + cross(dth_b, r_b));

    let normal = c.normal.xyz;
    let c0_n = c.normal.w;
    let c0_t1 = c.local_anchor_a.w;
    let c0_t2 = c.local_anchor_b.w;
    let tangent1 = normalize(c.tangent.xyz);
    let tangent2 = normalize(cross(normal, tangent1));
    let geom_c_n = c0_n + dot(normal, dmotion);
    let alpha_stab = params.quality.w;
    let dt = params.solver.x;
    // Alpha-regularized constraint with bounded penetration recovery (matches the
    // primal). The penalty ramp below still keys off the raw geometric penetration.
    let recovery_target = max((1.0 - alpha_stab) * (c0_n + CONTACT_MARGIN), -MAX_RECOVERY_VEL * dt);
    let c_n = recovery_target + dot(normal, dmotion);
    let c_t1 = c0_t1 + dot(tangent1, dmotion);
    let c_t2 = c0_t2 + dot(tangent2, dmotion);
    let inv_m_a = bodies[c.body_a].position_inv_mass.w;
    let inv_m_b = bodies[c.body_b].position_inv_mass.w;
    let mass_a = select(1e30, 1.0 / inv_m_a, inv_m_a > 0.0);
    let mass_b = select(1e30, 1.0 / inv_m_b, inv_m_b > 0.0);
    let lam_cap = min(LAMBDA_MAX, max(10.0, 1.0e5 * min(mass_a, mass_b)));
    let lambda_n = clamp(c.penalty.x * c_n + c.lambda.x, -lam_cap, 0.0);
    var tang = vec2<f32>(
        c.penalty.y * c_t1 + c.lambda.y,
        c.penalty.z * c_t2 + c.lambda.z,
    );
    let mu_static = sqrt(max(props[c.body_a].friction * props[c.body_b].friction, 0.0));
    let mu_dynamic = mu_static * params.solver2.w;
    let beta = params.solver.y;
    let k_max_normal = params.solver.w;
    let k_max_tangential = params.solver2.x;
    let tang_len = length(tang);
    let stick_limit = mu_static * abs(lambda_n);
    if tang_len > stick_limit && tang_len > 1e-8 {
        // Sliding: accumulate the dynamic-friction force.
        tang = tang * (mu_dynamic * abs(lambda_n) / tang_len);
    }

    var next_penalty = c.penalty;
    if lambda_n < -1e-6 && geom_c_n < -1e-5 {
        next_penalty.x = min(c.penalty.x + beta * abs(geom_c_n), k_max_normal);
    }

    var flags = 0u;
    if stick_limit > 0.0 && tang_len < stick_limit * 0.98 && max(abs(c_t1), abs(c_t2)) < 2e-3 {
        flags = CONTACT_FLAG_STICKING;
        next_penalty.y = min(c.penalty.y + beta * abs(c_t1), k_max_tangential);
        next_penalty.z = min(c.penalty.z + beta * abs(c_t2), k_max_tangential);
    }

    let delta_a = dpos_a + cross(dth_a, r_a);
    let delta_b = dpos_b + cross(dth_b, r_b);
    contacts[ci].point = vec4<f32>(
        (start_a.position_inv_mass.xyz + r_a + start_b.position_inv_mass.xyz + r_b) * 0.5
            + (delta_a + delta_b) * 0.5,
        geom_c_n,
    );
    contacts[ci].lambda = vec4<f32>(lambda_n, tang.x, tang.y, 0.0);
    contacts[ci].penalty = next_penalty;
    contacts[ci].flags = flags;
}
// <<< dual-contact-update

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let num_contacts = contact_count_buf[0];
    if ci >= num_contacts {
        return;
    }
    dual_update_contact(ci);
}
"#;

/// WGSL source for the contact-prep pass, run once per solve after warm-start.
///
/// Freezes the Taylor cache (AVBD): recomputes the solve-start constraint values
/// `C0` for the normal and both tangents from the post-warm-start anchors (sticking
/// carries replace anchors, so the narrowphase-time C0 can be stale), applies the
/// `m/dt^2` penalty floor + split normal/tangential caps, and records the pre-solve
/// relative normal velocity used by the restitution pass.
pub const CONTACT_PREP_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct Contact {
    point:          vec4<f32>,
    normal:         vec4<f32>,   // (n, C0_n out)
    tangent:        vec4<f32>,   // (t1, v_n_pre out)
    local_anchor_a: vec4<f32>,   // (anchor_a, C0_t1 out)
    local_anchor_b: vec4<f32>,   // (anchor_b, C0_t2 out)
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,  // (dt, beta, k_start, k_max_normal)
    counts:  vec4<u32>,
    quality: vec4<f32>,
    solver2: vec4<f32>,  // (k_max_tangential, max_speed, max_ang_speed, dynamic_mu_scale)
};

@group(0) @binding(0) var<storage, read>       solve_start_states: array<Body>;
@group(0) @binding(1) var<storage, read_write> contacts:           array<Contact>;
@group(0) @binding(2) var<uniform>             params:             SimParams;
@group(0) @binding(3) var<storage, read>       contact_count_buf:  array<u32>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    return q / max(length(q), 1e-12);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let num_contacts = contact_count_buf[0];
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let a = solve_start_states[c.body_a];
    let b = solve_start_states[c.body_b];
    let q_a = quat_normalize(a.orientation);
    let q_b = quat_normalize(b.orientation);
    let r_a = quat_rotate(q_a, c.local_anchor_a.xyz);
    let r_b = quat_rotate(q_b, c.local_anchor_b.xyz);
    let world_a = a.position_inv_mass.xyz + r_a;
    let world_b = b.position_inv_mass.xyz + r_b;
    let separation = world_a - world_b;

    let normal = c.normal.xyz;
    let tangent1 = normalize(c.tangent.xyz);
    let tangent2 = normalize(cross(normal, tangent1));

    // Pre-solve relative normal velocity at the contact (for restitution).
    let v_rel = (a.lin_vel.xyz + cross(a.ang_vel.xyz, r_a))
              - (b.lin_vel.xyz + cross(b.ang_vel.xyz, r_b));
    let v_n_pre = dot(normal, v_rel);

    // Penalty floor: a single contact must be stiff enough to hold the lighter
    // body against gravity in one step (k >= m/dt^2), independent of k_start.
    let dt = params.solver.x;
    let inv_m_a = a.position_inv_mass.w;
    let inv_m_b = b.position_inv_mass.w;
    let mass_a = select(1e30, 1.0 / inv_m_a, inv_m_a > 0.0);
    let mass_b = select(1e30, 1.0 / inv_m_b, inv_m_b > 0.0);
    let min_mass = min(mass_a, mass_b);
    let k_floor = select(0.0, min_mass / max(dt * dt, 1e-12), min_mass < 1e29);
    let k_max_normal = params.solver.w;
    let k_max_tangential = params.solver2.x;

    var penalty = c.penalty;
    penalty.x = clamp(max(penalty.x, k_floor), 0.0, k_max_normal);
    penalty.y = clamp(max(penalty.y, min(k_floor, k_max_tangential)), 0.0, k_max_tangential);
    penalty.z = clamp(max(penalty.z, min(k_floor, k_max_tangential)), 0.0, k_max_tangential);

    contacts[ci].normal = vec4<f32>(normal, dot(normal, separation));
    contacts[ci].tangent = vec4<f32>(tangent1, v_n_pre);
    contacts[ci].local_anchor_a = vec4<f32>(c.local_anchor_a.xyz, dot(tangent1, separation));
    contacts[ci].local_anchor_b = vec4<f32>(c.local_anchor_b.xyz, dot(tangent2, separation));
    contacts[ci].penalty = penalty;
}
"#;

/// WGSL source for the restitution pass, run once after velocity extraction.
///
/// Per-body Jacobi pass over that body's contacts (via the CSR adjacency used by
/// the primal): for every contact whose pre-solve approach speed exceeded the
/// restitution threshold, applies a mass-weighted normal impulse to *this body
/// only* so the post-solve separation speed reaches `-e * v_n_pre`. Both bodies
/// run the same formula with the pair's full effective mass, so the split is
/// consistent without cross-body writes (race-free).
pub const RESTITUTION_WGSL: &str = r#"
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
    tangent:        vec4<f32>,   // (t1, v_n_pre)
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
    quality: vec4<f32>,  // (contact_offset, restitution_threshold, penetration_slop, alpha_stab)
    solver2: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:               array<Body>;
@group(0) @binding(1) var<storage, read>       props:                array<BodyProps>;
@group(0) @binding(2) var<storage, read>       contacts:             array<Contact>;
@group(0) @binding(3) var<uniform>             params:               SimParams;
@group(0) @binding(4) var<storage, read>       body_contact_ranges:  array<vec2<u32>>;
@group(0) @binding(5) var<storage, read>       body_contact_indices: array<u32>;
@group(0) @binding(6) var<storage, read>       solve_start_states:   array<Body>;
@group(0) @binding(7) var<storage, read>       pre_pass_bodies:      array<Body>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    return q / max(length(q), 1e-12);
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
    );
}

fn world_inv_inertia(p: BodyProps, q: vec4<f32>) -> mat3x3<f32> {
    let r = quat_to_mat3(q);
    let local = mat3x3<f32>(p.inv_inertia_row0.xyz, p.inv_inertia_row1.xyz, p.inv_inertia_row2.xyz);
    return r * local * transpose(r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body_idx = gid.x;
    if body_idx >= params.counts.x {
        return;
    }
    let inv_mass = bodies[body_idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return;
    }

    let threshold = params.quality.y;
    var lin = bodies[body_idx].lin_vel.xyz;
    var ang = bodies[body_idx].ang_vel.xyz;
    let q = quat_normalize(bodies[body_idx].orientation);
    let inv_i = world_inv_inertia(props[body_idx], q);

    let contact_range = body_contact_ranges[body_idx];
    let range_end = contact_range.x + contact_range.y;
    for (var slot = contact_range.x; slot < range_end; slot = slot + 1u) {
        let c = contacts[body_contact_indices[slot]];
        let is_a = c.body_a == body_idx;
        let is_b = c.body_b == body_idx;
        if !is_a && !is_b {
            continue;
        }
        // Only restitute contacts that ended the solve in compression.
        if c.lambda.x >= 0.0 {
            continue;
        }
        let e = max(props[c.body_a].inv_inertia_row2.w, props[c.body_b].inv_inertia_row2.w);
        if e <= 0.0 {
            continue;
        }
        let v_n_pre = c.tangent.w;
        if v_n_pre >= -threshold {
            continue;
        }

        let start_a = solve_start_states[c.body_a];
        let start_b = solve_start_states[c.body_b];
        let r_a = quat_rotate(quat_normalize(start_a.orientation), c.local_anchor_a.xyz);
        let r_b = quat_rotate(quat_normalize(start_b.orientation), c.local_anchor_b.xyz);
        let n = c.normal.xyz;

        // Post-solve normal speed, read from the immutable pre-pass snapshot so
        // both bodies of a pair see identical values (Jacobi).
        let pa = pre_pass_bodies[c.body_a];
        let pb = pre_pass_bodies[c.body_b];
        let v_post = (pa.lin_vel.xyz + cross(pa.ang_vel.xyz, r_a))
                   - (pb.lin_vel.xyz + cross(pb.ang_vel.xyz, r_b));
        let v_n_post = dot(n, v_post);
        let v_n_target = -e * v_n_pre; // > 0: separating
        let dv = v_n_target - v_n_post;
        if dv <= 0.0 {
            continue;
        }

        // Effective mass of the pair along the contact normal.
        let inv_m_a = pa.position_inv_mass.w;
        let inv_m_b = pb.position_inv_mass.w;
        let qa = quat_normalize(pa.orientation);
        let qb = quat_normalize(pb.orientation);
        var w_sum = 0.0;
        if inv_m_a > 0.0 {
            let inv_i_a = world_inv_inertia(props[c.body_a], qa);
            let ra_x_n = cross(r_a, n);
            w_sum = w_sum + inv_m_a + dot(ra_x_n, inv_i_a * ra_x_n);
        }
        if inv_m_b > 0.0 {
            let inv_i_b = world_inv_inertia(props[c.body_b], qb);
            let rb_x_n = cross(r_b, n);
            w_sum = w_sum + inv_m_b + dot(rb_x_n, inv_i_b * rb_x_n);
        }
        if w_sum <= 1e-12 {
            continue;
        }
        let impulse = dv / w_sum;

        if is_a {
            lin = lin + n * (impulse * inv_mass);
            ang = ang + inv_i * cross(r_a, n * impulse);
        } else {
            lin = lin - n * (impulse * inv_mass);
            ang = ang - inv_i * cross(r_b, n * impulse);
        }
    }

    bodies[body_idx].lin_vel = vec4<f32>(lin, 0.0);
    bodies[body_idx].ang_vel = vec4<f32>(ang, 0.0);
}
"#;

/// Entry point for the single-workgroup persistent solve (D8/W3): the whole
/// `iterations x (colors + dual)` loop in ONE dispatch, replacing hundreds of
/// per-color compute passes for small scenes. Portable by construction: one
/// workgroup, `workgroupBarrier`/`storageBarrier` between phases (the
/// multi-workgroup variant is rejected — WGSL has no device-scope barrier and
/// avbd-metal's deadlocked).
///
/// Selected when total bodies <= 256 (web's max workgroup size); a color's
/// compacted range can then never exceed the workgroup, and contacts stride.
/// `params.counts.y` = solver iterations; `params.counts.w` carries the
/// clamped contact count, patched GPU-side by the args kernel + buffer copy
/// (same trick as the narrowphase pair count — no readback).
const AVBD_PERSISTENT_MAIN_WGSL: &str = r#"
const PERSIST_WG: u32 = 256u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    let iterations = params.counts.y;
    let num_contacts = min(params.counts.w, arrayLength(&contacts));
    for (var it = 0u; it < iterations; it = it + 1u) {
        // Gauss-Seidel across colors: barriers order color N's body writes
        // before color N+1's partner reads (the job pass boundaries do in the
        // multi-dispatch path).
        for (var color = 0u; color < 32u; color = color + 1u) {
            let range_count = body_order[2u * color + 1u];
            if lid < range_count {
                let range_offset = body_order[2u * color];
                solve_body(body_order[ORDER_HEADER + range_offset + lid]);
            }
            workgroupBarrier();
            storageBarrier();
        }
        for (var ci = lid; ci < num_contacts; ci = ci + PERSIST_WG) {
            dual_update_contact(ci);
        }
        workgroupBarrier();
        storageBarrier();
    }
}
"#;

/// Compose the persistent solve kernel from the primal's shared source
/// (types, bindings, constants, `solve_body`) plus the dual's marked
/// `dual_update_contact` — single source of truth by extraction, so the
/// persistent path can never drift from the per-color kernels.
pub fn avbd_persistent_wgsl() -> String {
    let primal_shared = AVBD_PRIMAL_WGSL
        .split("@compute")
        .next()
        .expect("AVBD_PRIMAL_WGSL has an entry point");
    let dual_fn = AVBD_DUAL_WGSL
        .split("// >>> dual-contact-update")
        .nth(1)
        .and_then(|tail| tail.split("// <<< dual-contact-update").next())
        .expect("AVBD_DUAL_WGSL carries the dual-contact-update markers");
    format!("{primal_shared}\n{dual_fn}\n{AVBD_PERSISTENT_MAIN_WGSL}")
}
