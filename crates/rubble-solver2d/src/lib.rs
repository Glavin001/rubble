use glam::Vec2;
use rubble_narrowphase2d::Contact2D;

pub fn integrate(pos: &mut [Vec2], vel: &mut [Vec2], inv_mass: &[f32], gravity: Vec2, dt: f32) {
    for i in 0..pos.len() {
        if inv_mass[i] == 0.0 {
            continue;
        }
        vel[i] += gravity * dt;
        pos[i] += vel[i] * dt;
    }
}

pub fn solve(pos: &mut [Vec2], inv_mass: &[f32], contacts: &[Contact2D], iterations: usize) {
    for _ in 0..iterations {
        for c in contacts {
            let ia = c.a as usize;
            let ib = c.b as usize;
            let w = inv_mass[ia] + inv_mass[ib];
            if w <= 0.0 {
                continue;
            }
            let corr = c.normal * (c.depth * 0.2 / w);
            pos[ia] -= corr * inv_mass[ia];
            pos[ib] += corr * inv_mass[ib];
        }
    }
}
