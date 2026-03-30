use glam::Vec2;
use rubble_broadphase2d::overlaps;
use rubble_narrowphase2d::contacts;
use rubble_shapes2d::{compute_aabb, Shape2D};
use rubble_solver2d::{integrate, solve};

pub struct World2D {
    pub gravity: Vec2,
    pub dt: f32,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub inv_mass: Vec<f32>,
    pub shapes: Vec<Shape2D>,
}

impl Default for World2D {
    fn default() -> Self {
        Self::new()
    }
}

impl World2D {
    pub fn new() -> Self {
        Self {
            gravity: Vec2::new(0.0, -9.81),
            dt: 1.0 / 60.0,
            pos: vec![],
            vel: vec![],
            inv_mass: vec![],
            shapes: vec![],
        }
    }

    pub fn add_circle(&mut self, pos: Vec2, radius: f32, mass: f32) {
        self.pos.push(pos);
        self.vel.push(Vec2::ZERO);
        self.inv_mass
            .push(if mass <= 0.0 { 0.0 } else { 1.0 / mass });
        self.shapes.push(Shape2D::Circle { radius });
    }

    pub fn step(&mut self) {
        integrate(
            &mut self.pos,
            &mut self.vel,
            &self.inv_mass,
            self.gravity,
            self.dt,
        );
        let aabbs: Vec<_> = self
            .pos
            .iter()
            .zip(self.shapes.iter())
            .map(|(p, s)| compute_aabb(s, *p))
            .collect();
        let pairs = overlaps(&aabbs);
        let cs = contacts(&pairs, &self.pos, &self.shapes);
        solve(&mut self.pos, &self.inv_mass, &cs, 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circles_separate_over_steps() {
        let mut w = World2D::new();
        w.gravity = Vec2::ZERO;
        w.add_circle(Vec2::new(0.0, 0.0), 1.0, 1.0);
        w.add_circle(Vec2::new(1.0, 0.0), 1.0, 1.0);
        for _ in 0..10 {
            w.step();
        }
        assert!(w.pos[0].distance(w.pos[1]) > 1.95);
    }
}
