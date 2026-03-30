//! `rubble-solver3d` — CPU reference constraint solver for 3D rigid bodies.
//!
//! Implements a position-based Gauss-Seidel solver with augmented Lagrangian
//! penalty terms, structured for future GPU AVBD optimization.

use glam::{Quat, Vec3, Vec4};
use rubble_math::{Contact3D, RigidBodyState3D};

// ---------------------------------------------------------------------------
// Graph coloring
// ---------------------------------------------------------------------------

/// Greedy graph coloring on body adjacency from contacts.
/// Returns `(color_per_body, num_colors)`. Bodies with the same color share no contacts.
pub fn greedy_coloring(num_bodies: usize, contacts: &[Contact3D]) -> (Vec<u32>, u32) {
    // Build adjacency list.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_bodies];
    for c in contacts {
        let a = c.body_a as usize;
        let b = c.body_b as usize;
        if a < num_bodies && b < num_bodies && a != b {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    let mut colors: Vec<u32> = vec![u32::MAX; num_bodies];
    let mut num_colors: u32 = 0;

    for body in 0..num_bodies {
        // Collect colors used by neighbors.
        let mut used = Vec::new();
        for &nb in &adj[body] {
            if colors[nb] != u32::MAX {
                used.push(colors[nb]);
            }
        }
        used.sort_unstable();
        used.dedup();

        // Find lowest available color.
        let mut c = 0u32;
        for &u in &used {
            if c == u {
                c += 1;
            } else {
                break;
            }
        }
        colors[body] = c;
        if c + 1 > num_colors {
            num_colors = c + 1;
        }
    }

    // Bodies with no contacts still get color 0; ensure num_colors >= 1 when
    // there are bodies.
    if num_bodies > 0 && num_colors == 0 {
        num_colors = 1;
    }

    (colors, num_colors)
}

// ---------------------------------------------------------------------------
// Solver parameters
// ---------------------------------------------------------------------------

/// Parameters for the AVBD / position-based solver.
pub struct SolverParams {
    /// Number of Gauss-Seidel iterations per step.
    pub iterations: u32,
    /// Augmented Lagrangian stiffness ramp rate.
    pub beta: f32,
    /// Initial penalty stiffness for contacts.
    pub k_start: f32,
    /// Decay factor applied to dual variables across steps.
    pub warmstart_decay: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            iterations: 5,
            beta: 10.0,
            k_start: 1e4,
            warmstart_decay: 0.95,
        }
    }
}

// ---------------------------------------------------------------------------
// 3D Solver
// ---------------------------------------------------------------------------

/// CPU reference position-based constraint solver for 3D rigid bodies.
pub struct Solver3D {
    params: SolverParams,
}

impl Solver3D {
    pub fn new(params: SolverParams) -> Self {
        Self { params }
    }

    /// Run one solver step.
    ///
    /// * Predicts body positions under gravity.
    /// * Iteratively resolves contact penetrations using position-based
    ///   Gauss-Seidel with augmented Lagrangian penalties.
    /// * Extracts velocities from the position change.
    pub fn solve(
        &self,
        dt: f32,
        gravity: Vec3,
        bodies: &mut [RigidBodyState3D],
        inv_inertias: &[glam::Mat3],
        contacts: &mut [Contact3D],
    ) {
        if dt <= 0.0 {
            return;
        }

        let n = bodies.len();

        // Save old positions and orientations for velocity extraction.
        let old_positions: Vec<Vec3> = bodies.iter().map(|b| b.position()).collect();
        let old_quats: Vec<Quat> = bodies.iter().map(|b| b.quat()).collect();

        // -----------------------------------------------------------------
        // 1. Predict
        // -----------------------------------------------------------------
        for body in bodies.iter_mut() {
            let im = body.inv_mass();
            if im <= 0.0 {
                continue; // static body
            }
            let pos = body.position();
            let vel = body.linear_velocity();
            let omega = body.angular_velocity();

            // x_tilde = pos + dt * vel + dt^2 * gravity * inv_mass
            // (inv_mass here is used as a flag; gravity applies to all dynamic
            //  bodies equally, so we just add dt^2 * gravity.)
            let x_tilde = pos + dt * vel + dt * dt * gravity;
            body.position_inv_mass = Vec4::new(x_tilde.x, x_tilde.y, x_tilde.z, im);

            // q_tilde = q + 0.5 * dt * Quat(0, omega) * q
            let q = body.quat();
            let omega_quat = Quat::from_xyzw(omega.x, omega.y, omega.z, 0.0);
            let q_dot = omega_quat * q;
            let q_new = Quat::from_xyzw(
                q.x + 0.5 * dt * q_dot.x,
                q.y + 0.5 * dt * q_dot.y,
                q.z + 0.5 * dt * q_dot.z,
                q.w + 0.5 * dt * q_dot.w,
            )
            .normalize();
            body.orientation = Vec4::new(q_new.x, q_new.y, q_new.z, q_new.w);
        }

        // -----------------------------------------------------------------
        // 2. Initialize contact penalty stiffness
        // -----------------------------------------------------------------
        for c in contacts.iter_mut() {
            if c.penalty_k == 0.0 {
                c.penalty_k = self.params.k_start;
            }
            // Warm-start decay
            c.lambda_n *= self.params.warmstart_decay;
            c.lambda_t1 *= self.params.warmstart_decay;
            c.lambda_t2 *= self.params.warmstart_decay;
        }

        // -----------------------------------------------------------------
        // 3. Graph coloring
        // -----------------------------------------------------------------
        let (_colors, _num_colors) = greedy_coloring(n, contacts);

        // -----------------------------------------------------------------
        // 4. Position-based Gauss-Seidel iterations
        // -----------------------------------------------------------------
        for _iter in 0..self.params.iterations {
            for ci in 0..contacts.len() {
                let c = &contacts[ci];
                let a = c.body_a as usize;
                let b = c.body_b as usize;
                let normal = c.contact_normal();
                let cp = c.contact_point();

                let im_a = bodies[a].inv_mass();
                let im_b = bodies[b].inv_mass();
                let w_sum = im_a + im_b;
                if w_sum <= 0.0 {
                    continue;
                }

                // Compute contact point relative to body centers.
                let r_a = cp - bodies[a].position();
                let r_b = cp - bodies[b].position();

                // Penetration depth: project relative displacement onto normal.
                // depth < 0 means penetrating (bodies overlap).
                // We compute: depth = dot((pos_b + r_b) - (pos_a + r_a), normal) + stored_depth
                // But the contact point and depth are already computed by the
                // narrow-phase. We recompute violation from current positions:
                // separation = dot(pos_b - pos_a, normal) - original_separation
                // For simplicity, use the stored depth and adjust for position changes.
                let depth = c.depth() + (bodies[b].position() - bodies[a].position()).dot(normal)
                    - (old_positions[b] - old_positions[a]).dot(normal);

                if depth >= 0.0 {
                    continue; // no penetration
                }

                // ---- Normal correction (position-based) ----
                let penalty = contacts[ci].penalty_k;
                let lambda_n = contacts[ci].lambda_n;

                // Effective correction magnitude with AL penalty.
                let correction = (-depth * penalty + lambda_n) / (w_sum * penalty + penalty);
                let correction = correction.max(0.0);

                // Apply position correction.
                {
                    let ba = &mut bodies[a];
                    let pos = ba.position();
                    let im = ba.inv_mass();
                    let new_pos = pos - normal * correction * im;
                    ba.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, im);
                }
                {
                    let bb = &mut bodies[b];
                    let pos = bb.position();
                    let im = bb.inv_mass();
                    let new_pos = pos + normal * correction * im;
                    bb.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, im);
                }

                // ---- Rotational correction ----
                // Apply torque-like correction from contact offset.
                let inv_inertia_a = if a < inv_inertias.len() {
                    inv_inertias[a]
                } else {
                    glam::Mat3::ZERO
                };
                let inv_inertia_b = if b < inv_inertias.len() {
                    inv_inertias[b]
                } else {
                    glam::Mat3::ZERO
                };

                if im_a > 0.0 {
                    let torque_a = r_a.cross(normal) * correction;
                    let delta_omega_a = inv_inertia_a * torque_a;
                    let q = bodies[a].quat();
                    let dq = Quat::from_xyzw(
                        -0.5 * delta_omega_a.x,
                        -0.5 * delta_omega_a.y,
                        -0.5 * delta_omega_a.z,
                        0.0,
                    ) * q;
                    let q_new = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w)
                        .normalize();
                    bodies[a].orientation = Vec4::new(q_new.x, q_new.y, q_new.z, q_new.w);
                }
                if im_b > 0.0 {
                    let torque_b = r_b.cross(normal) * correction;
                    let delta_omega_b = inv_inertia_b * torque_b;
                    let q = bodies[b].quat();
                    let dq = Quat::from_xyzw(
                        0.5 * delta_omega_b.x,
                        0.5 * delta_omega_b.y,
                        0.5 * delta_omega_b.z,
                        0.0,
                    ) * q;
                    let q_new = Quat::from_xyzw(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w)
                        .normalize();
                    bodies[b].orientation = Vec4::new(q_new.x, q_new.y, q_new.z, q_new.w);
                }

                // ---- Friction (tangential correction) ----
                // Compute relative tangential displacement at contact point.
                let rel_disp = (bodies[b].position() - old_positions[b])
                    - (bodies[a].position() - old_positions[a]);
                let rel_tangent = rel_disp - normal * rel_disp.dot(normal);
                let tang_len = rel_tangent.length();
                if tang_len > 1e-8 {
                    let mu = 0.5; // default friction coefficient
                    let max_tang = mu * correction;
                    let tang_correction = tang_len.min(max_tang);
                    let tang_dir = rel_tangent / tang_len;

                    if im_a > 0.0 {
                        let ba = &mut bodies[a];
                        let pos = ba.position();
                        let im = ba.inv_mass();
                        let new_pos = pos + tang_dir * tang_correction * im / w_sum;
                        ba.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, im);
                    }
                    if im_b > 0.0 {
                        let bb = &mut bodies[b];
                        let pos = bb.position();
                        let im = bb.inv_mass();
                        let new_pos = pos - tang_dir * tang_correction * im / w_sum;
                        bb.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_pos.z, im);
                    }
                }
            }

            // ---- Dual variable update (after all contacts in this iteration) ----
            for c in contacts.iter_mut() {
                let a = c.body_a as usize;
                let b = c.body_b as usize;
                let normal = c.contact_normal();

                // Recompute violation.
                let depth = c.depth() + (bodies[b].position() - bodies[a].position()).dot(normal)
                    - (old_positions[b] - old_positions[a]).dot(normal);

                if depth < 0.0 {
                    c.lambda_n += c.penalty_k * (-depth);
                    c.lambda_n = c.lambda_n.max(0.0);

                    // Friction dual: project to Coulomb cone.
                    let mu = 0.5;
                    let max_t = mu * c.lambda_n;
                    let t_len = (c.lambda_t1 * c.lambda_t1 + c.lambda_t2 * c.lambda_t2).sqrt();
                    if t_len > max_t && t_len > 0.0 {
                        let scale = max_t / t_len;
                        c.lambda_t1 *= scale;
                        c.lambda_t2 *= scale;
                    }

                    // Stiffness ramp.
                    c.penalty_k += self.params.beta * (-depth);
                }
            }
        }

        // -----------------------------------------------------------------
        // 5. Velocity extraction
        // -----------------------------------------------------------------
        let inv_dt = 1.0 / dt;
        for (i, body) in bodies.iter_mut().enumerate() {
            if body.inv_mass() <= 0.0 {
                continue;
            }
            let v_new = (body.position() - old_positions[i]) * inv_dt;
            body.lin_vel = v_new.extend(0.0);

            // omega = 2 * (q_new * conj(q_old)).xyz / dt
            let q_new = body.quat();
            let q_old_inv = old_quats[i].conjugate();
            let dq = q_new * q_old_inv;
            // Ensure positive w for shortest path.
            let sign = if dq.w < 0.0 { -1.0 } else { 1.0 };
            let omega_new =
                Vec3::new(dq.x * sign, dq.y * sign, dq.z * sign) * 2.0 * inv_dt;
            body.ang_vel = omega_new.extend(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Quat, Vec3, Vec4};
    use rubble_math::Contact3D;

    fn make_body(pos: Vec3, inv_mass: f32) -> RigidBodyState3D {
        RigidBodyState3D::new(pos, inv_mass, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO)
    }

    fn make_contact(a: u32, b: u32, point: Vec3, normal: Vec3, depth: f32) -> Contact3D {
        Contact3D {
            point: Vec4::new(point.x, point.y, point.z, depth),
            normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
            body_a: a,
            body_b: b,
            feature_id: 0,
            _pad: 0,
            lambda_n: 0.0,
            lambda_t1: 0.0,
            lambda_t2: 0.0,
            penalty_k: 0.0,
        }
    }

    // ------------------------------------------------------------------
    // Coloring tests
    // ------------------------------------------------------------------

    #[test]
    fn test_greedy_coloring_triangle() {
        // 3 bodies, all touching each other: A-B, B-C, A-C => need 3 colors.
        let contacts = vec![
            make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.01),
            make_contact(1, 2, Vec3::ZERO, Vec3::Y, -0.01),
            make_contact(0, 2, Vec3::ZERO, Vec3::Y, -0.01),
        ];
        let (colors, num) = greedy_coloring(3, &contacts);
        assert_eq!(num, 3);
        // All colors must be distinct.
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);
    }

    #[test]
    fn test_greedy_coloring_chain() {
        // A-B-C chain: A touches B, B touches C, but A and C don't touch.
        let contacts = vec![
            make_contact(0, 1, Vec3::ZERO, Vec3::Y, -0.01),
            make_contact(1, 2, Vec3::ZERO, Vec3::Y, -0.01),
        ];
        let (colors, num) = greedy_coloring(3, &contacts);
        assert_eq!(num, 2);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_eq!(colors[0], colors[2]); // A and C can share a color.
    }

    // ------------------------------------------------------------------
    // Solver tests
    // ------------------------------------------------------------------

    #[test]
    fn test_penetration_decreases() {
        // Two overlapping bodies on the Y axis.
        let mut bodies = vec![
            make_body(Vec3::new(0.0, 0.0, 0.0), 1.0),
            make_body(Vec3::new(0.0, 0.9, 0.0), 1.0), // overlapping by 0.1 if radius ~0.5
        ];
        let inv_inertias = vec![
            Mat3::from_diagonal(Vec3::splat(1.0)),
            Mat3::from_diagonal(Vec3::splat(1.0)),
        ];
        // Contact at midpoint, normal pointing from A to B, depth = -0.1 (penetrating).
        let mut contacts = vec![make_contact(
            0,
            1,
            Vec3::new(0.0, 0.45, 0.0),
            Vec3::Y,
            -0.1,
        )];

        let solver = Solver3D::new(SolverParams {
            iterations: 1,
            ..Default::default()
        });
        let initial_gap = (bodies[1].position() - bodies[0].position()).dot(Vec3::Y);
        solver.solve(1.0 / 60.0, Vec3::ZERO, &mut bodies, &inv_inertias, &mut contacts);
        let final_gap = (bodies[1].position() - bodies[0].position()).dot(Vec3::Y);

        // After solving, the bodies should be farther apart.
        assert!(
            final_gap > initial_gap,
            "Penetration should decrease: initial gap {initial_gap}, final gap {final_gap}"
        );
    }

    #[test]
    fn test_gravity_free_fall() {
        // Single body under gravity, no contacts.
        let mut bodies = vec![make_body(Vec3::ZERO, 1.0)];
        let inv_inertias = vec![Mat3::from_diagonal(Vec3::splat(1.0))];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;

        let solver = Solver3D::new(SolverParams::default());

        for _ in 0..100 {
            solver.solve(dt, gravity, &mut bodies, &inv_inertias, &mut []);
        }

        let t = 100.0 * dt;
        let expected_y = -0.5 * 9.81 * t * t;
        let actual_y = bodies[0].position().y;
        let error = (actual_y - expected_y).abs() / expected_y.abs();
        assert!(
            error < 0.02,
            "Free fall relative error too large: actual {actual_y}, expected {expected_y}, rel_error {error}"
        );
    }

    #[test]
    fn test_two_body_equilibrium() {
        // Two stacked bodies: bottom is static (inv_mass=0), top is dynamic.
        // Gravity pulls top body down, contact pushes it up.
        let mut bodies = vec![
            make_body(Vec3::new(0.0, 0.0, 0.0), 0.0),  // static floor
            make_body(Vec3::new(0.0, 1.0, 0.0), 1.0),   // dynamic box
        ];
        let inv_inertias = vec![Mat3::ZERO, Mat3::from_diagonal(Vec3::splat(1.0))];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;
        let solver = Solver3D::new(SolverParams {
            iterations: 10,
            ..Default::default()
        });

        // Run 50 steps. Each step we re-create the contact based on current positions.
        for _ in 0..50 {
            let gap = bodies[1].position().y - bodies[0].position().y - 1.0;
            let depth = if gap < 0.0 { gap } else { 0.0 };
            let mut contacts = if depth < 0.0 {
                vec![make_contact(
                    0,
                    1,
                    Vec3::new(0.0, bodies[0].position().y + 0.5, 0.0),
                    Vec3::Y,
                    depth,
                )]
            } else {
                vec![]
            };
            solver.solve(dt, gravity, &mut bodies, &inv_inertias, &mut contacts);
        }

        // The top body should be near y=1.0 (resting on the floor body at y=0).
        let y = bodies[1].position().y;
        assert!(
            (y - 1.0).abs() < 0.2,
            "Body should be near equilibrium at y=1.0, got y={y}"
        );
    }

    #[test]
    fn test_friction_slope() {
        // Body on a slope. With contact + friction the body's downward (Y)
        // displacement should be less than in free-fall (no contacts).
        let angle = std::f32::consts::FRAC_PI_6; // 30 degrees
        let slope_normal = Vec3::new(-angle.sin(), angle.cos(), 0.0);

        let solver = Solver3D::new(SolverParams {
            iterations: 10,
            ..Default::default()
        });

        let inv_inertias = vec![Mat3::ZERO, Mat3::from_diagonal(Vec3::splat(1.0))];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let dt = 1.0 / 60.0;

        // With friction contact: body is held by slope.
        let mut bodies_friction = vec![
            make_body(Vec3::ZERO, 0.0),
            make_body(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let start_pos = bodies_friction[1].position();
        for _ in 0..30 {
            let mut contacts = vec![make_contact(
                0,
                1,
                Vec3::new(0.5, 0.5, 0.0),
                slope_normal,
                -0.01,
            )];
            solver.solve(dt, gravity, &mut bodies_friction, &inv_inertias, &mut contacts);
        }
        let y_disp_friction = (bodies_friction[1].position().y - start_pos.y).abs();

        // Free-fall: no contacts at all.
        let mut bodies_free = vec![
            make_body(Vec3::ZERO, 0.0),
            make_body(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        for _ in 0..30 {
            solver.solve(dt, gravity, &mut bodies_free, &inv_inertias, &mut []);
        }
        let y_disp_free = (bodies_free[1].position().y - start_pos.y).abs();

        // The friction contact should reduce downward displacement compared to free-fall.
        assert!(
            y_disp_friction < y_disp_free,
            "Friction contact should slow fall: with_contact_y_disp={y_disp_friction}, free_fall_y_disp={y_disp_free}"
        );
    }
}
