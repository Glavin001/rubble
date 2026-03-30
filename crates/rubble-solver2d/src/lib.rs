//! `rubble-solver2d` — CPU reference constraint solver for 2D rigid bodies.
//!
//! Implements a position-based Gauss-Seidel solver with augmented Lagrangian
//! penalty terms, structured for future GPU AVBD optimization.

use glam::{Vec2, Vec4};
use rubble_math::{Contact2D, RigidBodyState2D};

// ---------------------------------------------------------------------------
// Graph coloring (reused logic from 3D but for Contact2D)
// ---------------------------------------------------------------------------

/// Greedy graph coloring on body adjacency from contacts.
/// Returns `(color_per_body, num_colors)`. Bodies with the same color share no contacts.
pub fn greedy_coloring(num_bodies: usize, contacts: &[Contact2D]) -> (Vec<u32>, u32) {
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
        let mut used = Vec::new();
        for &nb in &adj[body] {
            if colors[nb] != u32::MAX {
                used.push(colors[nb]);
            }
        }
        used.sort_unstable();
        used.dedup();

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

    if num_bodies > 0 && num_colors == 0 {
        num_colors = 1;
    }

    (colors, num_colors)
}

// ---------------------------------------------------------------------------
// Solver parameters (same as 3D)
// ---------------------------------------------------------------------------

/// Parameters for the AVBD / position-based solver.
pub struct SolverParams {
    pub iterations: u32,
    pub beta: f32,
    pub k_start: f32,
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
// 2D Solver
// ---------------------------------------------------------------------------

/// CPU reference position-based constraint solver for 2D rigid bodies.
///
/// Each body has 3 DOF: (x, y, angle).
pub struct Solver2D {
    params: SolverParams,
}

impl Solver2D {
    pub fn new(params: SolverParams) -> Self {
        Self { params }
    }

    /// Run one solver step.
    pub fn solve(
        &self,
        dt: f32,
        gravity: Vec2,
        bodies: &mut [RigidBodyState2D],
        contacts: &mut [Contact2D],
    ) {
        if dt <= 0.0 {
            return;
        }

        // Save old state for velocity extraction.
        let old_positions: Vec<Vec2> = bodies.iter().map(|b| b.position()).collect();
        let old_angles: Vec<f32> = bodies.iter().map(|b| b.angle()).collect();

        // -----------------------------------------------------------------
        // 1. Predict
        // -----------------------------------------------------------------
        for body in bodies.iter_mut() {
            let im = body.inv_mass();
            if im <= 0.0 {
                continue;
            }
            let pos = body.position();
            let vel = body.linear_velocity();
            let angle = body.angle();
            let omega = body.angular_velocity();

            let x_tilde = pos + dt * vel + dt * dt * gravity;
            let a_tilde = angle + dt * omega;

            body.position_inv_mass = Vec4::new(x_tilde.x, x_tilde.y, a_tilde, im);
        }

        // -----------------------------------------------------------------
        // 2. Initialize contact penalties
        // -----------------------------------------------------------------
        for c in contacts.iter_mut() {
            if c.penalty_k == 0.0 {
                c.penalty_k = self.params.k_start;
            }
            c.lambda_n *= self.params.warmstart_decay;
            c.lambda_t *= self.params.warmstart_decay;
        }

        // -----------------------------------------------------------------
        // 3. Position-based Gauss-Seidel iterations
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

                // Contact arms.
                let r_a = cp - bodies[a].position();
                let r_b = cp - bodies[b].position();

                // Recompute penetration depth from current positions.
                let depth = c.depth()
                    + (bodies[b].position() - bodies[a].position()).dot(normal)
                    - (old_positions[b] - old_positions[a]).dot(normal);

                if depth >= 0.0 {
                    continue;
                }

                // ---- Normal correction ----
                let penalty = contacts[ci].penalty_k;
                let lambda_n = contacts[ci].lambda_n;
                let correction = (-depth * penalty + lambda_n) / (w_sum * penalty + penalty);
                let correction = correction.max(0.0);

                if im_a > 0.0 {
                    let ba = &mut bodies[a];
                    let pos = ba.position();
                    let im = ba.inv_mass();
                    let angle = ba.angle();
                    let new_pos = pos - normal * correction * im;
                    // Rotational correction: 2D cross product r x n = r.x*n.y - r.y*n.x
                    let torque = r_a.x * normal.y - r_a.y * normal.x;
                    // Use a simple inverse inertia estimate (assume inv_I ~ inv_mass for simplicity).
                    let new_angle = angle - torque * correction * im * 0.1;
                    ba.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_angle, im);
                }
                if im_b > 0.0 {
                    let bb = &mut bodies[b];
                    let pos = bb.position();
                    let im = bb.inv_mass();
                    let angle = bb.angle();
                    let new_pos = pos + normal * correction * im;
                    let torque = r_b.x * normal.y - r_b.y * normal.x;
                    let new_angle = angle + torque * correction * im * 0.1;
                    bb.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, new_angle, im);
                }

                // ---- Friction ----
                let rel_disp = (bodies[b].position() - old_positions[b])
                    - (bodies[a].position() - old_positions[a]);
                let rel_tangent = rel_disp - normal * rel_disp.dot(normal);
                let tang_len = rel_tangent.length();
                if tang_len > 1e-8 {
                    let mu = 0.5;
                    let max_tang = mu * correction;
                    let tang_correction = tang_len.min(max_tang);
                    let tang_dir = rel_tangent / tang_len;

                    if im_a > 0.0 {
                        let ba = &mut bodies[a];
                        let pos = ba.position();
                        let im = ba.inv_mass();
                        let angle = ba.angle();
                        let new_pos = pos + tang_dir * tang_correction * im / w_sum;
                        ba.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, angle, im);
                    }
                    if im_b > 0.0 {
                        let bb = &mut bodies[b];
                        let pos = bb.position();
                        let im = bb.inv_mass();
                        let angle = bb.angle();
                        let new_pos = pos - tang_dir * tang_correction * im / w_sum;
                        bb.position_inv_mass = Vec4::new(new_pos.x, new_pos.y, angle, im);
                    }
                }
            }

            // ---- Dual variable update ----
            for c in contacts.iter_mut() {
                let a = c.body_a as usize;
                let b = c.body_b as usize;
                let normal = c.contact_normal();

                let depth = c.depth()
                    + (bodies[b].position() - bodies[a].position()).dot(normal)
                    - (old_positions[b] - old_positions[a]).dot(normal);

                if depth < 0.0 {
                    c.lambda_n += c.penalty_k * (-depth);
                    c.lambda_n = c.lambda_n.max(0.0);

                    let mu = 0.5;
                    let max_t = mu * c.lambda_n;
                    if c.lambda_t.abs() > max_t {
                        c.lambda_t = c.lambda_t.signum() * max_t;
                    }

                    c.penalty_k += self.params.beta * (-depth);
                }
            }
        }

        // -----------------------------------------------------------------
        // 4. Velocity extraction
        // -----------------------------------------------------------------
        let inv_dt = 1.0 / dt;
        for (i, body) in bodies.iter_mut().enumerate() {
            if body.inv_mass() <= 0.0 {
                continue;
            }
            let v_new = (body.position() - old_positions[i]) * inv_dt;
            let omega_new = (body.angle() - old_angles[i]) * inv_dt;
            body.lin_vel = Vec4::new(v_new.x, v_new.y, omega_new, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec2, Vec4};
    use rubble_math::Contact2D;

    fn make_body_2d(x: f32, y: f32, inv_mass: f32) -> RigidBodyState2D {
        RigidBodyState2D::new(x, y, 0.0, inv_mass, 0.0, 0.0, 0.0)
    }

    fn make_contact_2d(a: u32, b: u32, point: Vec2, normal: Vec2, depth: f32) -> Contact2D {
        Contact2D {
            point: Vec4::new(point.x, point.y, depth, 0.0),
            normal: Vec4::new(normal.x, normal.y, 0.0, 0.0),
            body_a: a,
            body_b: b,
            feature_id: 0,
            _pad: 0,
            lambda_n: 0.0,
            lambda_t: 0.0,
            penalty_k: 0.0,
            _pad2: 0.0,
        }
    }

    #[test]
    fn test_2d_coloring() {
        // Triangle: 3 mutually contacting bodies.
        let contacts = vec![
            make_contact_2d(0, 1, Vec2::ZERO, Vec2::Y, -0.01),
            make_contact_2d(1, 2, Vec2::ZERO, Vec2::Y, -0.01),
            make_contact_2d(0, 2, Vec2::ZERO, Vec2::Y, -0.01),
        ];
        let (colors, num) = greedy_coloring(3, &contacts);
        assert_eq!(num, 3);
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_ne!(colors[0], colors[2]);

        // Chain: A-B-C.
        let contacts_chain = vec![
            make_contact_2d(0, 1, Vec2::ZERO, Vec2::Y, -0.01),
            make_contact_2d(1, 2, Vec2::ZERO, Vec2::Y, -0.01),
        ];
        let (colors2, num2) = greedy_coloring(3, &contacts_chain);
        assert_eq!(num2, 2);
        assert_eq!(colors2[0], colors2[2]);
    }

    #[test]
    fn test_2d_penetration_decreases() {
        let mut bodies = vec![
            make_body_2d(0.0, 0.0, 1.0),
            make_body_2d(0.0, 0.9, 1.0),
        ];
        let mut contacts = vec![make_contact_2d(
            0,
            1,
            Vec2::new(0.0, 0.45),
            Vec2::Y,
            -0.1,
        )];

        let solver = Solver2D::new(SolverParams {
            iterations: 1,
            ..Default::default()
        });
        let initial_gap = bodies[1].position().y - bodies[0].position().y;
        solver.solve(1.0 / 60.0, Vec2::ZERO, &mut bodies, &mut contacts);
        let final_gap = bodies[1].position().y - bodies[0].position().y;

        assert!(
            final_gap > initial_gap,
            "Penetration should decrease: initial {initial_gap}, final {final_gap}"
        );
    }

    #[test]
    fn test_2d_gravity() {
        let mut bodies = vec![make_body_2d(0.0, 0.0, 1.0)];
        let gravity = Vec2::new(0.0, -9.81);
        let dt = 1.0 / 60.0;
        let solver = Solver2D::new(SolverParams::default());

        for _ in 0..100 {
            solver.solve(dt, gravity, &mut bodies, &mut []);
        }

        let t = 100.0 * dt;
        let expected_y = -0.5 * 9.81 * t * t;
        let actual_y = bodies[0].position().y;
        let error = (actual_y - expected_y).abs() / expected_y.abs();
        assert!(
            error < 0.02,
            "Free fall relative error: actual {actual_y}, expected {expected_y}, rel_error {error}"
        );
    }
}
