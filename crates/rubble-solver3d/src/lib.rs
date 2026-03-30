use glam::Vec3;
use rubble_math::{Contact3D, RigidBodyState3D};

#[derive(Clone, Copy, Debug)]
pub struct SolverParams {
    pub iterations: u32,
    pub baumgarte: f32,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            iterations: 6,
            baumgarte: 0.2,
        }
    }
}

pub fn solve_contacts(
    states: &mut [RigidBodyState3D],
    contacts: &[Contact3D],
    params: SolverParams,
) {
    for _ in 0..params.iterations {
        for c in contacts {
            let ai = c.body_a as usize;
            let bi = c.body_b as usize;
            let inv_ma = states[ai].position_inv_mass.w;
            let inv_mb = states[bi].position_inv_mass.w;
            if inv_ma + inv_mb <= 0.0 {
                continue;
            }
            let n = c.normal.truncate();
            let depth = c.point.w.max(0.0);
            if depth <= 0.0 {
                continue;
            }
            let correction = n * (depth * params.baumgarte / (inv_ma + inv_mb));
            if inv_ma > 0.0 {
                let mut p = states[ai].position_inv_mass.truncate();
                p -= correction * inv_ma;
                states[ai].position_inv_mass = p.extend(inv_ma);
            }
            if inv_mb > 0.0 {
                let mut p = states[bi].position_inv_mass.truncate();
                p += correction * inv_mb;
                states[bi].position_inv_mass = p.extend(inv_mb);
            }
        }
    }
}

pub fn integrate(states: &mut [RigidBodyState3D], gravity: Vec3, dt: f32) {
    for state in states {
        let inv_m = state.position_inv_mass.w;
        if inv_m == 0.0 {
            continue;
        }
        let mut v = state.lin_vel.truncate();
        v += gravity * dt;
        let mut p = state.position_inv_mass.truncate();
        p += v * dt;
        state.lin_vel = v.extend(0.0);
        state.position_inv_mass = p.extend(inv_m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec4};

    fn body(pos: Vec3, inv_m: f32) -> RigidBodyState3D {
        RigidBodyState3D {
            position_inv_mass: pos.extend(inv_m),
            orientation: Quat::IDENTITY,
            lin_vel: Vec4::ZERO,
            ang_vel: Vec4::ZERO,
        }
    }

    #[test]
    fn penetration_decreases() {
        let mut states = vec![body(Vec3::ZERO, 1.0), body(Vec3::new(1.5, 0.0, 0.0), 1.0)];
        let c = Contact3D {
            point: Vec4::new(1.0, 0.0, 0.0, 0.5),
            normal: Vec4::new(1.0, 0.0, 0.0, 0.0),
            body_a: 0,
            body_b: 1,
            ..Default::default()
        };
        solve_contacts(&mut states, &[c], SolverParams::default());
        assert!(states[0].position_inv_mass.x < 0.0);
        assert!(states[1].position_inv_mass.x > 1.5);
    }
}
