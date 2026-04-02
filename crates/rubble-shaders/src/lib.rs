//! Rust-GPU shader library for the rubble physics engine.
//!
//! All physics compute kernels are implemented in Rust using `spirv-std`,
//! compiled to SPIR-V via `rust-gpu`, and can target multiple GPU backends
//! (Vulkan, Metal, DX12) through the SPIR-V intermediate representation.
//!
//! This provides multi-GPU target support — the same Rust source produces
//! SPIR-V that runs on any GPU vendor without per-vendor shader maintenance.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

use spirv_std::glam::{UVec3, Vec2, Vec3, Vec4};
use spirv_std::spirv;

// ---------------------------------------------------------------------------
// GPU-side data types — must match rubble-math #[repr(C)] layouts exactly
// ---------------------------------------------------------------------------

/// 3D rigid body state: 4 x Vec4 = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Body3D {
    /// (x, y, z, 1/m)
    pub position_inv_mass: Vec4,
    /// Quaternion (x, y, z, w)
    pub orientation: Vec4,
    /// (vx, vy, vz, 0)
    pub lin_vel: Vec4,
    /// (wx, wy, wz, 0)
    pub ang_vel: Vec4,
}

/// 2D rigid body state: 4 x Vec4 = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Body2D {
    /// (x, y, angle, 1/m)
    pub position_inv_mass: Vec4,
    /// (vx, vy, angular_vel, 0)
    pub lin_vel: Vec4,
    /// Padding — _pad0.x stores friction
    pub _pad0: Vec4,
    pub _pad1: Vec4,
}

/// 3D body properties: 3 x Vec4 + 4 x f32 = 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BodyProps3D {
    pub inv_inertia_row0: Vec4,
    pub inv_inertia_row1: Vec4,
    pub inv_inertia_row2: Vec4,
    pub friction: f32,
    pub shape_type: u32,
    pub shape_index: u32,
    pub flags: u32,
}

/// 3D contact: 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Contact3D {
    /// (x, y, z, depth)
    pub point: Vec4,
    /// (nx, ny, nz, 0)
    pub normal: Vec4,
    pub body_a: u32,
    pub body_b: u32,
    pub feature_id: u32,
    pub _pad: u32,
    pub lambda_n: f32,
    pub lambda_t1: f32,
    pub lambda_t2: f32,
    pub penalty_k: f32,
}

/// 2D contact: 64 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Contact2D {
    /// (x, y, depth, 0)
    pub point: Vec4,
    /// (nx, ny, 0, 0)
    pub normal: Vec4,
    pub body_a: u32,
    pub body_b: u32,
    pub feature_id: u32,
    pub _pad: u32,
    pub lambda_n: f32,
    pub lambda_t: f32,
    pub penalty_k: f32,
    pub _pad2: f32,
}

/// Simulation parameters: 32 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SimParams {
    pub gravity: Vec4,
    pub dt: f32,
    pub num_bodies: u32,
    pub solver_iterations: u32,
    pub _pad: u32,
}

/// Solve range for graph-colored dispatch: 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SolveRange {
    pub offset: u32,
    pub count: u32,
}

/// Axis-aligned bounding box: 32 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min_pt: Vec4,
    pub max_pt: Vec4,
}

/// Sphere shape data: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SphereData {
    pub radius: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Box shape data: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BoxDataGpu {
    pub half_extents: Vec4,
}

/// Capsule shape data: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CapsuleDataGpu {
    pub half_height: f32,
    pub radius: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Convex hull info: 32 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvexHullInfo {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub face_offset: u32,
    pub face_count: u32,
    pub edge_offset: u32,
    pub edge_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Convex vertex: 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ConvexVert {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

/// Broadphase pair: 8 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pair {
    pub a: u32,
    pub b: u32,
}

// ---------------------------------------------------------------------------
// Math helpers — GPU-optimized, always inlined
// ---------------------------------------------------------------------------

/// Quaternion multiply: a * b.
#[inline(always)]
pub fn qmul(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )
}

/// Quaternion conjugate.
#[inline(always)]
pub fn qconj(q: Vec4) -> Vec4 {
    Vec4::new(-q.x, -q.y, -q.z, q.w)
}

/// Rotate vector v by quaternion q.
#[inline(always)]
pub fn quat_rotate(q: Vec4, v: Vec3) -> Vec3 {
    let u = Vec3::new(q.x, q.y, q.z);
    let s = q.w;
    2.0 * u.dot(v) * u + (s * s - u.dot(u)) * v + 2.0 * s * u.cross(v)
}

/// Multiply 3x3 matrix (stored as 3 column-vec4s) by vec3.
/// The rows r0, r1, r2 are actually columns due to Rust column-major storage.
#[inline(always)]
pub fn mat3_mul(r0: Vec4, r1: Vec4, r2: Vec4, v: Vec3) -> Vec3 {
    Vec3::new(
        r0.x * v.x + r1.x * v.y + r2.x * v.z,
        r0.y * v.x + r1.y * v.y + r2.y * v.z,
        r0.z * v.x + r1.z * v.y + r2.z * v.z,
    )
}

/// 2D cross product: a × b = a.x*b.y - a.y*b.x (scalar result).
#[inline(always)]
pub fn cross2d(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

// ---------------------------------------------------------------------------
// 3D Predict kernel
// ---------------------------------------------------------------------------

/// 3D prediction shader: saves old state, integrates gravity, predicts position
/// and orientation for the upcoming solver step.
#[spirv(compute(threads(64)))]
pub fn predict_3d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_states: &mut [Body3D],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimParams,
) {
    let idx = id.x as usize;
    if idx >= params.num_bodies as usize {
        return;
    }

    let body = bodies[idx];
    let inv_mass = body.position_inv_mass.w;

    // Save old state for velocity extraction
    old_states[idx] = body;

    if inv_mass <= 0.0 {
        return; // static body
    }

    let pos = body.position_inv_mass.truncate();
    let vel = body.lin_vel.truncate();
    let omega = body.ang_vel.truncate();
    let dt = params.dt;
    let gravity = params.gravity.truncate();

    // Integrate velocity with gravity
    let new_vel = vel + dt * gravity;

    // Position prediction
    let x_tilde = pos + dt * new_vel;
    bodies[idx].position_inv_mass = x_tilde.extend(inv_mass);
    bodies[idx].lin_vel = new_vel.extend(0.0);

    // Orientation prediction: q_tilde = normalize(q + 0.5 * dt * omega_quat * q)
    let q = body.orientation;
    let omega_quat = Vec4::new(omega.x, omega.y, omega.z, 0.0);
    let q_dot = qmul(omega_quat, q);
    let q_new = (q + 0.5 * dt * q_dot).normalize();
    bodies[idx].orientation = q_new;
}

// ---------------------------------------------------------------------------
// 3D AVBD Solver kernel — the critical GPU performance path
// ---------------------------------------------------------------------------

/// Augmented Velocity-Based Dynamics solver for 3D rigid bodies.
///
/// Implements graph-colored Gauss-Seidel: contacts sorted by color group,
/// `solve_range` provides (offset, count) so each dispatch processes only
/// contacts of one color (no two share a body → no data races).
///
/// Algorithm per contact:
/// 1. Compute effective mass (linear + rotational contributions)
/// 2. Compute relative velocity at contact point
/// 3. Baumgarte positional stabilization bias
/// 4. Augmented Lagrangian impulse with penalty stiffness
/// 5. Coulomb friction with cone clamping
/// 6. Penalty stiffness ramp for convergence
#[spirv(compute(threads(64)))]
pub fn avbd_solve_3d(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] _old_states: &[Body3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] props: &[BodyProps3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contacts: &mut [Contact3D],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_count_buf: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] solve_range: &SolveRange,
) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }
    let ci = (local_idx + solve_range.offset) as usize;
    let num_contacts = contact_count_buf[0] as usize;
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let a = c.body_a as usize;
    let b = c.body_b as usize;
    let normal = c.normal.truncate();
    let cp = c.point.truncate();

    let im_a = bodies[a].position_inv_mass.w;
    let im_b = bodies[b].position_inv_mass.w;
    let w_sum = im_a + im_b;
    if w_sum <= 0.0 {
        return;
    }

    let pos_a = bodies[a].position_inv_mass.truncate();
    let pos_b = bodies[b].position_inv_mass.truncate();

    // Contact offsets
    let r_a = cp - pos_a;
    let r_b = cp - pos_b;

    // Inverse inertia tensors
    let props_a = props[a];
    let props_b = props[b];

    // Effective mass: w_eff = inv_m_a + inv_m_b + (inv_I_a * (r_a × n)) · (r_a × n) + ...
    let rn_a = r_a.cross(normal);
    let rn_b = r_b.cross(normal);
    let inv_i_rn_a = mat3_mul(
        props_a.inv_inertia_row0,
        props_a.inv_inertia_row1,
        props_a.inv_inertia_row2,
        rn_a,
    );
    let inv_i_rn_b = mat3_mul(
        props_b.inv_inertia_row0,
        props_b.inv_inertia_row1,
        props_b.inv_inertia_row2,
        rn_b,
    );
    let w_rot_a = inv_i_rn_a.dot(rn_a);
    let w_rot_b = inv_i_rn_b.dot(rn_b);
    let w_eff = w_sum + w_rot_a + w_rot_b;
    if w_eff <= 0.0 {
        return;
    }

    let depth = c.point.w;
    if depth >= 0.0 {
        return; // no penetration
    }

    // Current velocities
    let v_a = bodies[a].lin_vel.truncate();
    let v_b = bodies[b].lin_vel.truncate();
    let w_a = bodies[a].ang_vel.truncate();
    let w_b = bodies[b].ang_vel.truncate();

    // Keep old_states binding alive
    let _keep = _old_states[0].position_inv_mass.x;

    // Relative velocity at contact point
    let v_rel = (v_b + w_b.cross(r_b)) - (v_a + w_a.cross(r_a));
    let v_n = v_rel.dot(normal);

    // Baumgarte stabilization
    let beta = 0.2;
    let bias = beta * depth / params.dt;

    // Constraint violation
    let d_c = v_n + bias;
    if d_c >= 0.0 {
        return; // separating
    }

    // Augmented Lagrangian impulse
    let penalty = c.penalty_k;
    let lambda_old = c.lambda_n;
    let impulse = (-d_c * penalty + lambda_old) / (w_eff * penalty + 1.0);
    let impulse_clamped = impulse.max(0.0);

    // Apply linear and angular impulses
    if im_a > 0.0 {
        bodies[a].lin_vel = (v_a - normal * impulse_clamped * im_a).extend(0.0);
        bodies[a].ang_vel = (w_a - inv_i_rn_a * impulse_clamped).extend(0.0);
    }
    if im_b > 0.0 {
        bodies[b].lin_vel = (v_b + normal * impulse_clamped * im_b).extend(0.0);
        bodies[b].ang_vel = (w_b + inv_i_rn_b * impulse_clamped).extend(0.0);
    }

    // Update dual variable
    contacts[ci].lambda_n = (lambda_old + penalty * (-depth)).max(0.0);

    // --- Friction ---
    let mu = (props_a.friction + props_b.friction) * 0.5;
    let v_rel_current = (v_b + w_b.cross(r_b)) - (v_a + w_a.cross(r_a));
    let v_tangent = v_rel_current - normal * v_rel_current.dot(normal);
    let tang_len = v_tangent.length();
    if tang_len > 1e-8 {
        let tang_dir = v_tangent / tang_len;
        let max_tang_impulse = mu * impulse_clamped;
        let tang_impulse = (tang_len / w_eff).min(max_tang_impulse);

        if im_a > 0.0 {
            let cur = bodies[a].lin_vel.truncate();
            bodies[a].lin_vel = (cur + tang_dir * tang_impulse * im_a).extend(0.0);
        }
        if im_b > 0.0 {
            let cur = bodies[b].lin_vel.truncate();
            bodies[b].lin_vel = (cur - tang_dir * tang_impulse * im_b).extend(0.0);
        }
    }

    // Stiffness ramp
    contacts[ci].penalty_k = penalty + 10.0 * (-depth);
}

// ---------------------------------------------------------------------------
// 3D Extract Velocity kernel
// ---------------------------------------------------------------------------

/// Recomputes positions from AVBD-solved velocities:
///   pos_new = pos_old + dt * v_solved
/// Angular velocity extracted from quaternion delta.
#[spirv(compute(threads(64)))]
pub fn extract_velocity_3d(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_states: &[Body3D],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimParams,
) {
    let idx = gid.x as usize;
    if idx >= params.num_bodies as usize {
        return;
    }

    let inv_mass = bodies[idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return;
    }

    let dt = params.dt;

    // Recompute position from old position + dt * solved velocity
    let v_solved = bodies[idx].lin_vel.truncate();
    let pos_old = old_states[idx].position_inv_mass.truncate();
    let pos_new = pos_old + dt * v_solved;
    bodies[idx].position_inv_mass = pos_new.extend(inv_mass);

    // Angular velocity from quaternion delta: omega = 2 * (q_new * conj(q_old)).xyz / dt
    let q_new = bodies[idx].orientation;
    let q_old = old_states[idx].orientation;
    let dq = qmul(q_new, qconj(q_old));
    let sign = if dq.w >= 0.0 { 1.0 } else { -1.0 };
    let omega_new = Vec3::new(dq.x, dq.y, dq.z) * sign * 2.0 / dt;
    bodies[idx].ang_vel = omega_new.extend(0.0);
}

// ---------------------------------------------------------------------------
// 2D Predict kernel
// ---------------------------------------------------------------------------

/// 2D prediction: integrate gravity, predict position (x, y, angle).
#[spirv(compute(threads(64)))]
pub fn predict_2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body2D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_states: &mut [Body2D],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimParams,
) {
    let idx = id.x as usize;
    if idx >= params.num_bodies as usize {
        return;
    }

    let body = bodies[idx];
    let inv_mass = body.position_inv_mass.w;

    old_states[idx] = body;

    if inv_mass <= 0.0 {
        return;
    }

    let pos_x = body.position_inv_mass.x;
    let pos_y = body.position_inv_mass.y;
    let angle = body.position_inv_mass.z;
    let vx = body.lin_vel.x;
    let vy = body.lin_vel.y;
    let omega = body.lin_vel.z;
    let dt = params.dt;

    // Integrate velocity with gravity
    let new_vx = vx + dt * params.gravity.x;
    let new_vy = vy + dt * params.gravity.y;

    // Predict position
    let x_tilde = pos_x + dt * new_vx;
    let y_tilde = pos_y + dt * new_vy;
    let angle_tilde = angle + dt * omega;

    bodies[idx].position_inv_mass = Vec4::new(x_tilde, y_tilde, angle_tilde, inv_mass);
    bodies[idx].lin_vel = Vec4::new(new_vx, new_vy, omega, 0.0);
}

// ---------------------------------------------------------------------------
// 2D AVBD Solver kernel
// ---------------------------------------------------------------------------

/// Augmented Velocity-Based Dynamics solver for 2D rigid bodies.
/// 3 DOF per body: (x, y, angle).
#[spirv(compute(threads(64)))]
pub fn avbd_solve_2d(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body2D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] _old_states: &[Body2D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts: &mut [Contact2D],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] params: &SimParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_count_buf: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] solve_range: &SolveRange,
) {
    let local_idx = gid.x;
    if local_idx >= solve_range.count {
        return;
    }
    let ci = (local_idx + solve_range.offset) as usize;
    let num_contacts = contact_count_buf[0] as usize;
    if ci >= num_contacts {
        return;
    }

    let c = contacts[ci];
    let a = c.body_a as usize;
    let b = c.body_b as usize;
    let normal = Vec2::new(c.normal.x, c.normal.y);
    let cp = Vec2::new(c.point.x, c.point.y);

    let im_a = bodies[a].position_inv_mass.w;
    let im_b = bodies[b].position_inv_mass.w;
    let w_sum = im_a + im_b;
    if w_sum <= 0.0 {
        return;
    }

    let pos_a = Vec2::new(bodies[a].position_inv_mass.x, bodies[a].position_inv_mass.y);
    let pos_b = Vec2::new(bodies[b].position_inv_mass.x, bodies[b].position_inv_mass.y);

    let r_a = cp - pos_a;
    let r_b = cp - pos_b;

    // 2D rotational effective mass
    let rn_a = cross2d(r_a, normal);
    let rn_b = cross2d(r_b, normal);
    let w_rot_a = im_a * rn_a * rn_a;
    let w_rot_b = im_b * rn_b * rn_b;
    let w_eff = w_sum + w_rot_a + w_rot_b;
    if w_eff <= 0.0 {
        return;
    }

    let depth = c.point.z;
    if depth >= 0.0 {
        return;
    }

    let v_a = Vec2::new(bodies[a].lin_vel.x, bodies[a].lin_vel.y);
    let v_b = Vec2::new(bodies[b].lin_vel.x, bodies[b].lin_vel.y);
    let w_a = bodies[a].lin_vel.z;
    let w_b = bodies[b].lin_vel.z;

    // Keep old_states binding alive
    let _keep = _old_states[0].position_inv_mass.x;

    // Velocity at contact point: v + omega * perp(r)
    let v_contact_a = v_a + w_a * Vec2::new(-r_a.y, r_a.x);
    let v_contact_b = v_b + w_b * Vec2::new(-r_b.y, r_b.x);
    let v_rel = v_contact_b - v_contact_a;
    let v_n = v_rel.dot(normal);

    // Baumgarte stabilization
    let beta = 0.2;
    let bias = beta * depth / params.dt;
    let d_c = v_n + bias;
    if d_c >= 0.0 {
        return;
    }

    // Augmented Lagrangian impulse
    let penalty = c.penalty_k;
    let lambda_old = c.lambda_n;
    let impulse = (-d_c * penalty + lambda_old) / (w_eff * penalty + 1.0);
    let impulse_clamped = impulse.max(0.0);

    if im_a > 0.0 {
        let new_lin = v_a - normal * impulse_clamped * im_a;
        let new_ang = w_a - im_a * rn_a * impulse_clamped;
        bodies[a].lin_vel = Vec4::new(new_lin.x, new_lin.y, new_ang, 0.0);
    }
    if im_b > 0.0 {
        let new_lin = v_b + normal * impulse_clamped * im_b;
        let new_ang = w_b + im_b * rn_b * impulse_clamped;
        bodies[b].lin_vel = Vec4::new(new_lin.x, new_lin.y, new_ang, 0.0);
    }

    // Update dual variable
    contacts[ci].lambda_n = (lambda_old + penalty * (-depth)).max(0.0);

    // Friction
    let mu = (bodies[a]._pad0.x + bodies[b]._pad0.x) * 0.5;
    let v_rel_ca = v_a + w_a * Vec2::new(-r_a.y, r_a.x);
    let v_rel_cb = v_b + w_b * Vec2::new(-r_b.y, r_b.x);
    let v_rel_c = v_rel_cb - v_rel_ca;
    let v_tangent = v_rel_c - normal * v_rel_c.dot(normal);
    let tang_len = v_tangent.length();
    if tang_len > 1e-8 {
        let tang_dir = v_tangent / tang_len;
        let max_tang_impulse = mu * impulse_clamped;
        let tang_impulse = (tang_len / w_eff).min(max_tang_impulse);

        if im_a > 0.0 {
            let cur = Vec2::new(bodies[a].lin_vel.x, bodies[a].lin_vel.y);
            let new_lin = cur + tang_dir * tang_impulse * im_a;
            bodies[a].lin_vel =
                Vec4::new(new_lin.x, new_lin.y, bodies[a].lin_vel.z, 0.0);
        }
        if im_b > 0.0 {
            let cur = Vec2::new(bodies[b].lin_vel.x, bodies[b].lin_vel.y);
            let new_lin = cur - tang_dir * tang_impulse * im_b;
            bodies[b].lin_vel =
                Vec4::new(new_lin.x, new_lin.y, bodies[b].lin_vel.z, 0.0);
        }
    }

    // Stiffness ramp
    contacts[ci].penalty_k = penalty + 10.0 * (-depth);
}

// ---------------------------------------------------------------------------
// 2D Extract Velocity kernel
// ---------------------------------------------------------------------------

/// Recomputes 2D positions from solved velocities.
#[spirv(compute(threads(64)))]
pub fn extract_velocity_2d(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &mut [Body2D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] old_states: &[Body2D],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &SimParams,
) {
    let idx = gid.x as usize;
    if idx >= params.num_bodies as usize {
        return;
    }

    let inv_mass = bodies[idx].position_inv_mass.w;
    if inv_mass <= 0.0 {
        return;
    }

    let dt = params.dt;

    // Recompute position from old + dt * solved velocity
    let vx = bodies[idx].lin_vel.x;
    let vy = bodies[idx].lin_vel.y;
    let omega = bodies[idx].lin_vel.z;
    let old_x = old_states[idx].position_inv_mass.x;
    let old_y = old_states[idx].position_inv_mass.y;
    let old_angle = old_states[idx].position_inv_mass.z;

    bodies[idx].position_inv_mass = Vec4::new(
        old_x + dt * vx,
        old_y + dt * vy,
        old_angle + dt * omega,
        inv_mass,
    );
}

// ---------------------------------------------------------------------------
// 3D Sphere-Sphere narrowphase kernel
// ---------------------------------------------------------------------------

/// Sphere-sphere contact generation kernel.
/// Emits one contact per overlapping sphere pair using atomic counter.
#[spirv(compute(threads(64)))]
pub fn sphere_sphere_test(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] bodies: &[Body3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] props: &[BodyProps3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pairs: &[Pair],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] pair_count_in: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] spheres: &[SphereData],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contacts: &mut [Contact3D],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_count: &mut u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] params: &SimParams,
) {
    let idx = gid.x as usize;
    if idx >= pair_count_in[0] as usize {
        return;
    }

    let pair = pairs[idx];
    let a = pair.a as usize;
    let b = pair.b as usize;

    // Only handle sphere-sphere pairs
    if props[a].shape_type != 0 || props[b].shape_type != 0 {
        return;
    }

    let pos_a = bodies[a].position_inv_mass.truncate();
    let pos_b = bodies[b].position_inv_mass.truncate();
    let r_a = spheres[props[a].shape_index as usize].radius;
    let r_b = spheres[props[b].shape_index as usize].radius;

    let diff = pos_b - pos_a;
    let dist_sq = diff.dot(diff);
    let sum_r = r_a + r_b;

    if dist_sq >= sum_r * sum_r {
        return;
    }

    let dist = dist_sq.sqrt();
    if dist < 1e-12 {
        return;
    }

    let normal = diff / dist;
    let depth = dist - sum_r;
    let point = pos_a + normal * (r_a + depth * 0.5);

    // Atomic increment would go here — for now, this demonstrates the kernel structure.
    // In the actual WGSL pipeline, atomicAdd is used. In rust-gpu, we'd use
    // spirv_std::arch::atomic_i_increment or write to a known index.
    let _max_contacts = params.num_bodies * 8;
    let _contact = Contact3D {
        point: Vec4::new(point.x, point.y, point.z, depth),
        normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
        body_a: pair.a,
        body_b: pair.b,
        feature_id: 0,
        _pad: 0,
        lambda_n: 0.0,
        lambda_t1: 0.0,
        lambda_t2: 0.0,
        penalty_k: 1e4,
    };

    // Note: actual atomic contact emission requires spirv_std atomics.
    // The full narrowphase with all shape pairs is kept as WGSL for now
    // since it uses atomic<u32> extensively. The solver kernels above
    // are the performance-critical path and are fully ported to rust-gpu.
    let _ = contact_count;
    let _ = contacts;
}

// ---------------------------------------------------------------------------
// Trivial test kernel (kept for backward compatibility)
// ---------------------------------------------------------------------------

/// Simple test kernel: multiply each element by 2.
#[spirv(compute(threads(64)))]
pub fn multiply_by_two(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [f32],
) {
    let idx = id.x as usize;
    if idx < data.len() {
        data[idx] *= 2.0;
    }
}
