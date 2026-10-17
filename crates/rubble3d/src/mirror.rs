//! CPU mirror of the GPU AVBD solver math (D12 in the production plan).
//!
//! Scalar Rust implementations of the exact WGSL kernel math — same equations,
//! same clamps, same constants, f32 throughout. This is a *testing anchor and
//! executable documentation*, not a runtime fallback: GPU↔CPU parity tests pin
//! the shader behavior, and any future shader edit that changes the math must
//! change this file (and its tests) too.
//!
//! Mirrored kernels:
//! - `solve_6x6`: Jacobi-preconditioned LDLᵀ with pivot floor (primal solve).
//! - `contact_rows` / `primal_body_update`: per-contact force + Hessian
//!   accumulation and the trust-region-capped Newton step.
//! - `dual_update`: the augmented-Lagrangian λ/k update.

use glam::{Mat3, Quat, Vec3};

/// Mirrors the WGSL `CONTACT_MARGIN`.
pub const CONTACT_MARGIN: f32 = 0.01;
/// Mirrors the WGSL `LAMBDA_MAX`.
pub const LAMBDA_MAX: f32 = 1.0e6;
/// Mirrors the WGSL `MAX_RECOVERY_VEL`.
pub const MAX_RECOVERY_VEL: f32 = 6.0;

/// Solver parameters consumed by the mirrored kernels (the subset of
/// `SimParamsGpu` the solve reads).
#[derive(Clone, Copy, Debug)]
pub struct MirrorParams {
    pub dt: f32,
    pub beta: f32,
    pub k_max_normal: f32,
    pub k_max_tangential: f32,
    pub alpha_stab: f32,
    pub dynamic_mu_scale: f32,
}

/// One contact's solver state (the fields the primal/dual read and write).
#[derive(Clone, Copy, Debug)]
pub struct MirrorContact {
    pub normal: Vec3,
    pub tangent1: Vec3,
    /// Frozen solve-start constraint values (Taylor cache): (C0_n, C0_t1, C0_t2).
    pub c0: Vec3,
    /// Local anchors (body frame), rotated by the *solve-start* orientation.
    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,
    pub lambda: Vec3,
    pub penalty: Vec3,
    pub body_a: usize,
    pub body_b: usize,
}

/// One body's pose + mass data as the solve sees it.
#[derive(Clone, Copy, Debug)]
pub struct MirrorBody {
    pub position: Vec3,
    pub orientation: Quat,
    pub inv_mass: f32,
    /// Local-frame inverse inertia.
    pub inv_inertia: Mat3,
    pub friction: f32,
}

/// Mirrors WGSL `rotation_delta_vec`: the rotation vector taking `reference`
/// to `current` (axis * angle, shortest arc).
pub fn rotation_delta_vec(current: Quat, reference: Quat) -> Vec3 {
    let mut dq = current * reference.conjugate();
    if dq.w < 0.0 {
        dq = Quat::from_xyzw(-dq.x, -dq.y, -dq.z, -dq.w);
    }
    let imag = Vec3::new(dq.x, dq.y, dq.z);
    let imag_len = imag.length();
    if imag_len < 1e-8 {
        return 2.0 * imag;
    }
    let angle = 2.0 * imag_len.atan2(dq.w);
    imag / imag_len * angle
}

/// Mirrors WGSL `world_inv_inertia`.
pub fn world_inv_inertia(inv_inertia_local: Mat3, q: Quat) -> Mat3 {
    let r = Mat3::from_quat(q);
    r * inv_inertia_local * r.transpose()
}

/// Mirrors WGSL `solve_6x6`: Jacobi-preconditioned LDLᵀ with a tiny (1e-6)
/// scaled-diagonal nudge and a 1e-6 pivot floor. Input must be symmetric with a
/// positive(ish) diagonal — which the AVBD system guarantees by construction.
#[allow(clippy::needless_range_loop)] // index loops mirror the WGSL 1:1
pub fn solve_6x6(m: [[f32; 6]; 6], rhs: [f32; 6]) -> [f32; 6] {
    let mut a = [[0.0f32; 6]; 6];
    let mut b = [0.0f32; 6];
    let mut s = [0.0f32; 6];

    for i in 0..6 {
        s[i] = 1.0 / m[i][i].max(1e-12).sqrt();
    }
    for i in 0..6 {
        for j in 0..6 {
            a[i][j] = m[i][j] * s[i] * s[j];
        }
        a[i][i] += 1e-6;
        b[i] = rhs[i] * s[i];
    }

    for k in 0..6 {
        let mut d = a[k][k];
        for p in 0..k {
            d -= a[k][p] * a[k][p] * a[p][p];
        }
        a[k][k] = d.max(1e-6);
        for i in (k + 1)..6 {
            let mut v = a[i][k];
            for p in 0..k {
                v -= a[i][p] * a[k][p] * a[p][p];
            }
            a[i][k] = v / a[k][k];
        }
    }

    for i in 0..6 {
        let mut v = b[i];
        for p in 0..i {
            v -= a[i][p] * b[p];
        }
        b[i] = v;
    }
    for i in 0..6 {
        b[i] /= a[i][i];
    }
    for i in (0..6).rev() {
        let mut v = b[i];
        for p in (i + 1)..6 {
            v -= a[p][i] * b[p];
        }
        b[i] = v;
    }

    let mut x = [0.0f32; 6];
    for i in 0..6 {
        x[i] = b[i] * s[i];
    }
    x
}

/// Mirrors WGSL `effective_stiffness` (AVBD Eq. 14 stiffness rescaling).
pub fn effective_stiffness(k: f32, force: f32, c: f32, clamped: bool) -> f32 {
    if !clamped {
        return k;
    }
    k.min(force.abs() / c.abs().max(1e-9))
}

/// Mirrors the WGSL contact-force block of the primal: given one contact and the
/// live/solve-start states of its two bodies, produce the three constraint rows
/// as `(j_lin, j_ang, k_eff, force)` tuples for the requested body.
///
/// `solve_start` must be the post-predict snapshot the Taylor cache was frozen
/// in — the same buffer the GPU binds as `solve_start_states`.
#[allow(clippy::too_many_arguments)]
pub fn contact_rows(
    c: &MirrorContact,
    live: &[MirrorBody],
    solve_start: &[MirrorBody],
    for_body: usize,
    params: &MirrorParams,
) -> [(Vec3, Vec3, f32, f32); 3] {
    let sa = &solve_start[c.body_a];
    let sb = &solve_start[c.body_b];
    let r_a = sa.orientation * c.local_anchor_a;
    let r_b = sb.orientation * c.local_anchor_b;

    let dpos_a = live[c.body_a].position - sa.position;
    let dpos_b = live[c.body_b].position - sb.position;
    let dth_a = rotation_delta_vec(live[c.body_a].orientation, sa.orientation);
    let dth_b = rotation_delta_vec(live[c.body_b].orientation, sb.orientation);
    let dmotion = (dpos_a + dth_a.cross(r_a)) - (dpos_b + dth_b.cross(r_b));

    let normal = c.normal;
    let tangent1 = c.tangent1.normalize();
    let tangent2 = normal.cross(tangent1);

    let recovery_target =
        ((1.0 - params.alpha_stab) * (c.c0.x + CONTACT_MARGIN)).max(-MAX_RECOVERY_VEL * params.dt);
    let c_n = recovery_target + normal.dot(dmotion);
    let c_t1 = c.c0.y + tangent1.dot(dmotion);
    let c_t2 = c.c0.z + tangent2.dot(dmotion);

    let mass_a = if live[c.body_a].inv_mass > 0.0 {
        1.0 / live[c.body_a].inv_mass
    } else {
        1e30
    };
    let mass_b = if live[c.body_b].inv_mass > 0.0 {
        1.0 / live[c.body_b].inv_mass
    } else {
        1e30
    };
    let lam_cap = LAMBDA_MAX.min((1.0e5 * mass_a.min(mass_b)).max(10.0));

    let f_n_raw = c.penalty.x * c_n + c.lambda.x;
    let f_n = f_n_raw.clamp(-lam_cap, 0.0);
    // Full normal stiffness even when clamped (see the WGSL primal): Eq. 14
    // secant rescaling is a friction-row feature only.
    let k_n_eff = c.penalty.x;

    let mut tang = glam::Vec2::new(
        c.penalty.y * c_t1 + c.lambda.y,
        c.penalty.z * c_t2 + c.lambda.z,
    );
    let mu_static = (live[c.body_a].friction * live[c.body_b].friction)
        .max(0.0)
        .sqrt();
    let mu_dynamic = mu_static * params.dynamic_mu_scale;
    let f_n_ref = f_n.abs().max(c.lambda.x.abs());
    let tang_len = tang.length();
    let mut k_t1_eff = c.penalty.y;
    let mut k_t2_eff = c.penalty.z;
    if tang_len > mu_static * f_n_ref && tang_len > 1e-8 {
        tang *= mu_dynamic * f_n_ref / tang_len;
        k_t1_eff = effective_stiffness(c.penalty.y, tang.x, c_t1, true);
        k_t2_eff = effective_stiffness(c.penalty.z, tang.y, c_t2, true);
    }

    let is_b = for_body == c.body_b;
    let (sn, r) = if is_b { (-1.0, r_b) } else { (1.0, r_a) };
    [
        (normal * sn, r.cross(normal) * sn, k_n_eff, f_n),
        (tangent1 * sn, r.cross(tangent1) * sn, k_t1_eff, tang.x),
        (tangent2 * sn, r.cross(tangent2) * sn, k_t2_eff, tang.y),
    ]
}

/// Mirrors one full primal update for one body: assemble the 6×6 system from the
/// inertial term + all contact rows, solve, apply trust-region caps, and return
/// the new (position, orientation).
pub fn primal_body_update(
    body_idx: usize,
    live: &[MirrorBody],
    inertial: &[MirrorBody],
    solve_start: &[MirrorBody],
    contacts: &[MirrorContact],
    params: &MirrorParams,
) -> (Vec3, Quat) {
    let body = &live[body_idx];
    if body.inv_mass <= 0.0 {
        return (body.position, body.orientation);
    }

    let dt2 = params.dt * params.dt;
    let mass = 1.0 / body.inv_mass;
    let inv_i_world = world_inv_inertia(body.inv_inertia, body.orientation);
    let i_world = if inv_i_world.determinant().abs() < 1e-10 {
        Mat3::ZERO
    } else {
        inv_i_world.inverse()
    };
    let rot_delta = rotation_delta_vec(body.orientation, inertial[body_idx].orientation);

    let mut mtx = [[0.0f32; 6]; 6];
    let mut rhs = [0.0f32; 6];
    mtx[0][0] = mass / dt2;
    mtx[1][1] = mass / dt2;
    mtx[2][2] = mass / dt2;
    let dpos = body.position - inertial[body_idx].position;
    rhs[0] = mtx[0][0] * dpos.x;
    rhs[1] = mtx[1][1] * dpos.y;
    rhs[2] = mtx[2][2] * dpos.z;
    // glam Mat3 is column-major: i_world.col(c)[r] = element (r, c). The WGSL
    // uses i_world[r][c] with mat3x3 constructed from rows, so transpose here.
    let iw = i_world.transpose();
    for r in 0..3 {
        for cc in 0..3 {
            mtx[3 + r][3 + cc] = iw.col(r)[cc] / dt2;
        }
        rhs[3 + r] =
            (iw.col(r)[0] * rot_delta.x + iw.col(r)[1] * rot_delta.y + iw.col(r)[2] * rot_delta.z)
                / dt2;
    }

    for c in contacts {
        if c.body_a != body_idx && c.body_b != body_idx {
            continue;
        }
        for (j_lin, j_ang, k, f) in contact_rows(c, live, solve_start, body_idx, params) {
            let j = [j_lin.x, j_lin.y, j_lin.z, j_ang.x, j_ang.y, j_ang.z];
            for r in 0..6 {
                rhs[r] += j[r] * f;
                for cc in 0..6 {
                    mtx[r][cc] += k * j[r] * j[cc];
                }
            }
        }
    }

    let x = solve_6x6(mtx, rhs);
    let mut dlin = Vec3::new(x[0], x[1], x[2]);
    let mut dang = Vec3::new(x[3], x[4], x[5]);

    let trace_i = iw.col(0)[0] + iw.col(1)[1] + iw.col(2)[2];
    let bound_r = (1.5 * trace_i * body.inv_mass).max(1e-8).sqrt();
    let max_lin = 0.35 * bound_r;
    let lin_len = dlin.length();
    if lin_len > max_lin && lin_len > 1e-12 {
        dlin *= max_lin / lin_len;
    }
    let ang_len = dang.length();
    if ang_len > 0.5 {
        dang *= 0.5 / ang_len;
    }

    let new_pos = body.position - dlin;
    let half = -dang * 0.5;
    let dq = Quat::from_xyzw(half.x, half.y, half.z, 1.0).normalize();
    (new_pos, (dq * body.orientation).normalize())
}

/// Mirrors the WGSL dual update for one contact. Returns the contact with
/// updated `lambda`, `penalty`, and a sticking flag.
pub fn dual_update(
    c: &MirrorContact,
    live: &[MirrorBody],
    solve_start: &[MirrorBody],
    params: &MirrorParams,
) -> (MirrorContact, bool) {
    let sa = &solve_start[c.body_a];
    let sb = &solve_start[c.body_b];
    let r_a = sa.orientation * c.local_anchor_a;
    let r_b = sb.orientation * c.local_anchor_b;
    let dpos_a = live[c.body_a].position - sa.position;
    let dpos_b = live[c.body_b].position - sb.position;
    let dth_a = rotation_delta_vec(live[c.body_a].orientation, sa.orientation);
    let dth_b = rotation_delta_vec(live[c.body_b].orientation, sb.orientation);
    let dmotion = (dpos_a + dth_a.cross(r_a)) - (dpos_b + dth_b.cross(r_b));

    let normal = c.normal;
    let tangent1 = c.tangent1.normalize();
    let tangent2 = normal.cross(tangent1);
    let geom_c_n = c.c0.x + normal.dot(dmotion);
    let recovery_target =
        ((1.0 - params.alpha_stab) * (c.c0.x + CONTACT_MARGIN)).max(-MAX_RECOVERY_VEL * params.dt);
    let c_n = recovery_target + normal.dot(dmotion);
    let c_t1 = c.c0.y + tangent1.dot(dmotion);
    let c_t2 = c.c0.z + tangent2.dot(dmotion);

    let mass_a = if live[c.body_a].inv_mass > 0.0 {
        1.0 / live[c.body_a].inv_mass
    } else {
        1e30
    };
    let mass_b = if live[c.body_b].inv_mass > 0.0 {
        1.0 / live[c.body_b].inv_mass
    } else {
        1e30
    };
    let lam_cap = LAMBDA_MAX.min((1.0e5 * mass_a.min(mass_b)).max(10.0));
    let lambda_n = (c.penalty.x * c_n + c.lambda.x).clamp(-lam_cap, 0.0);

    let mut tang = glam::Vec2::new(
        c.penalty.y * c_t1 + c.lambda.y,
        c.penalty.z * c_t2 + c.lambda.z,
    );
    let mu_static = (live[c.body_a].friction * live[c.body_b].friction)
        .max(0.0)
        .sqrt();
    let mu_dynamic = mu_static * params.dynamic_mu_scale;
    let tang_len = tang.length();
    let stick_limit = mu_static * lambda_n.abs();
    if tang_len > stick_limit && tang_len > 1e-8 {
        tang *= mu_dynamic * lambda_n.abs() / tang_len;
    }

    let mut next = *c;
    next.lambda = Vec3::new(lambda_n, tang.x, tang.y);
    if lambda_n < -1e-6 && geom_c_n < -1e-5 {
        next.penalty.x = (c.penalty.x + params.beta * geom_c_n.abs()).min(params.k_max_normal);
    }
    let mut sticking = false;
    if stick_limit > 0.0 && tang_len < stick_limit * 0.98 && c_t1.abs().max(c_t2.abs()) < 2e-3 {
        sticking = true;
        next.penalty.y = (c.penalty.y + params.beta * c_t1.abs()).min(params.k_max_tangential);
        next.penalty.z = (c.penalty.z + params.beta * c_t2.abs()).min(params.k_max_tangential);
    }
    (next, sticking)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // matrix index loops mirror the WGSL 1:1
mod tests {
    use super::*;

    fn spd_system() -> ([[f32; 6]; 6], [f32; 6]) {
        // J^T K J + M/dt^2 style system built from two rows, guaranteed SPD.
        let m = 2.0f32;
        let dt2 = (1.0f32 / 60.0).powi(2);
        let mut a = [[0.0f32; 6]; 6];
        for i in 0..3 {
            a[i][i] = m / dt2;
        }
        for i in 3..6 {
            a[i][i] = 0.4 / dt2;
        }
        let rows: [([f32; 6], f32); 2] = [
            ([0.0, 1.0, 0.0, 0.3, 0.0, -0.2], 5.0e8),
            ([0.7, 0.0, 0.7, 0.0, 0.5, 0.0], 1.0e6),
        ];
        for (j, k) in rows {
            for r in 0..6 {
                for c in 0..6 {
                    a[r][c] += k * j[r] * j[c];
                }
            }
        }
        let rhs = [1.0, -2.0, 0.5, 0.25, -0.75, 3.0];
        (a, rhs)
    }

    #[test]
    fn ldlt_solves_stiff_spd_system() {
        let (a, b) = spd_system();
        let x = solve_6x6(a, b);
        // Compare against an f64 Gaussian-elimination reference of the same raw
        // system. The stiffness spread (5e8 vs ~7e3) is representative of a
        // floored contact against k_max_normal; the preconditioned LDLᵀ should
        // land within f32 round-off + the 1e-6 nudge of the exact solution.
        let mut aa = [[0.0f64; 7]; 6];
        for i in 0..6 {
            for j in 0..6 {
                aa[i][j] = a[i][j] as f64;
            }
            aa[i][6] = b[i] as f64;
        }
        for k in 0..6 {
            let piv = (k..6)
                .max_by(|&p, &q| aa[p][k].abs().total_cmp(&aa[q][k].abs()))
                .unwrap();
            aa.swap(k, piv);
            for i in 0..6 {
                if i == k {
                    continue;
                }
                let f = aa[i][k] / aa[k][k];
                for j in k..7 {
                    aa[i][j] -= f * aa[k][j];
                }
            }
        }
        let mut rel_err_num = 0.0f64;
        let mut rel_err_den = 0.0f64;
        for i in 0..6 {
            let exact = aa[i][6] / aa[i][i];
            rel_err_num += (x[i] as f64 - exact).powi(2);
            rel_err_den += exact.powi(2);
        }
        let rel = (rel_err_num / rel_err_den.max(1e-30)).sqrt();
        // The 5e8-vs-7e3 stiffness spread leaves κ(A') ≈ 4e4 even after Jacobi
        // scaling (three DOFs share one dominant rank-1 row), so the best any
        // f32 solve can promise is O(κ·ε_machine) ≈ 0.5–5%. The AVBD outer
        // iteration contracts this per-step error; what matters is that it is
        // bounded and finite, not machine-precision.
        assert!(rel < 0.05, "solution error vs f64 reference: {rel}");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ldlt_exact_on_well_conditioned_system() {
        // Moderate stiffness (1e5): κ small, f32 LDLᵀ should be near-exact.
        let dt2 = (1.0f32 / 60.0).powi(2);
        let mut a = [[0.0f32; 6]; 6];
        for i in 0..3 {
            a[i][i] = 2.0 / dt2;
        }
        for i in 3..6 {
            a[i][i] = 0.4 / dt2;
        }
        let j = [0.0, 1.0, 0.0, 0.3, 0.0, -0.2];
        for r in 0..6 {
            for c in 0..6 {
                a[r][c] += 1.0e5 * j[r] * j[c];
            }
        }
        let b = [1.0, -2.0, 0.5, 0.25, -0.75, 3.0];
        let x = solve_6x6(a, b);
        let mut r = [0.0f32; 6];
        for i in 0..6 {
            r[i] = -b[i];
            for jj in 0..6 {
                r[i] += a[i][jj] * x[jj];
            }
        }
        let rn: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let bn: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(rn / bn < 1e-4, "well-conditioned residual: {}", rn / bn);
    }

    #[test]
    fn ldlt_survives_singular_direction() {
        // Rank-deficient beyond the mass floor: zero angular diagonal.
        let mut a = [[0.0f32; 6]; 6];
        for i in 0..3 {
            a[i][i] = 100.0;
        }
        // angular block: zero (a static-ish degenerate case) — must not NaN
        // and must not zero out the *linear* solution (the old Gaussian
        // elimination returned all-zero here, dropping the whole update).
        let b = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let x = solve_6x6(a, b);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!((x[0] - 0.01).abs() < 1e-3, "linear part solved: {}", x[0]);
    }

    #[test]
    fn effective_stiffness_matches_secant() {
        assert_eq!(effective_stiffness(1e6, -50.0, -0.01, false), 1e6);
        let k = effective_stiffness(1e6, -50.0, -0.01, true);
        assert!((k - 5000.0).abs() < 1.0, "secant k = f/C = 5000, got {k}");
        // Separating contact, force clamped to zero: no Hessian contribution.
        assert_eq!(effective_stiffness(1e6, 0.0, 0.02, true), 0.0);
    }

    #[test]
    fn dual_update_ramps_and_clamps() {
        let params = MirrorParams {
            dt: 1.0 / 60.0,
            beta: 1.0e4,
            k_max_normal: 1.0e9,
            k_max_tangential: 1.0e6,
            alpha_stab: 0.95,
            dynamic_mu_scale: 1.0,
        };
        let bodies = vec![
            MirrorBody {
                position: Vec3::new(0.0, 0.4, 0.0), // sphere r=0.5, 0.1 deep
                orientation: Quat::IDENTITY,
                inv_mass: 1.0,
                inv_inertia: Mat3::IDENTITY * (1.0 / 0.1),
                friction: 0.5,
            },
            MirrorBody {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                inv_mass: 0.0,
                inv_inertia: Mat3::ZERO,
                friction: 0.5,
            },
        ];
        let c = MirrorContact {
            normal: Vec3::Y,
            tangent1: Vec3::X,
            c0: Vec3::new(-0.1, 0.0, 0.0),
            local_anchor_a: Vec3::new(0.0, -0.5, 0.0),
            local_anchor_b: Vec3::new(0.0, 0.4, 0.0),
            lambda: Vec3::ZERO,
            penalty: Vec3::new(3600.0, 3600.0, 3600.0),
            body_a: 0,
            body_b: 1,
        };
        let (next, _) = dual_update(&c, &bodies, &bodies, &params);
        // Penetrating + compressive: λ_n must go negative, k_n must ramp.
        assert!(next.lambda.x < 0.0, "compressive dual: {}", next.lambda.x);
        assert!(next.penalty.x > c.penalty.x, "penalty ramped");
        assert!(next.penalty.x <= params.k_max_normal);
    }

    #[test]
    fn primal_pushes_penetrating_sphere_up() {
        let params = MirrorParams {
            dt: 1.0 / 60.0,
            beta: 1.0e4,
            k_max_normal: 1.0e9,
            k_max_tangential: 1.0e6,
            alpha_stab: 0.3,
            dynamic_mu_scale: 1.0,
        };
        // Sphere resting 0.02 into a static floor, gravity pulling it down:
        // inertial target is below current, contact must push the solution up
        // relative to the inertial target.
        let g_drop = 9.81 * params.dt * params.dt;
        let live = vec![
            MirrorBody {
                position: Vec3::new(0.0, 0.48, 0.0),
                orientation: Quat::IDENTITY,
                inv_mass: 1.0,
                inv_inertia: Mat3::IDENTITY * (1.0 / 0.1),
                friction: 0.5,
            },
            MirrorBody {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                inv_mass: 0.0,
                inv_inertia: Mat3::ZERO,
                friction: 0.5,
            },
        ];
        let mut inertial = live.clone();
        inertial[0].position.y -= g_drop;
        let solve_start = live.clone();
        let contacts = vec![MirrorContact {
            normal: Vec3::Y,
            tangent1: Vec3::X,
            c0: Vec3::new(-0.02, 0.0, 0.0),
            local_anchor_a: Vec3::new(0.0, -0.5, 0.0),
            local_anchor_b: Vec3::new(0.0, 0.46, 0.0),
            lambda: Vec3::ZERO,
            penalty: Vec3::new(3600.0, 3600.0, 3600.0),
            body_a: 0,
            body_b: 1,
        }];
        let (pos, _q) = primal_body_update(0, &live, &inertial, &solve_start, &contacts, &params);
        assert!(
            pos.y > inertial[0].position.y,
            "contact must hold the sphere above its inertial target: {} vs {}",
            pos.y,
            inertial[0].position.y
        );
        assert!(pos.is_finite());
    }
}
