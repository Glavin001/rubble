//! Balls fired into a scene as threshold strikers: the one implementation the fracture
//! lab's stress models ([`crate::lab::Lab`]) and the range scenes ([`super::range`]:
//! Broadside, the timber yard) share. Each ball is a dynamic sphere registered as a
//! projectile (its breaks are predicted at the end of the frame before it lands: fracture is
//! forward only, AGENTS.md). A pool keeps at most `max_live` balls (the oldest is destroyed
//! past it) and at most `max_striking` in strike prediction's shot table (the oldest leaves
//! it early and rolls on as a plain rigid ball).

use crate::*;

/// One ball's body: its mass, radius, surface and continuous collision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub mass: f32,
    pub radius: f32,
    pub material: SurfaceMaterial,
    /// Continuous collision (`BodyDef::is_bullet`).
    pub ccd: bool,
}

impl Ball {
    /// A solid sphere's mass properties.
    pub fn mass_data(&self) -> MassData {
        let i = 0.4 * self.mass * self.radius * self.radius;
        MassData {
            mass: self.mass,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        }
    }
}

/// The balls in a world, oldest first, with the step each was fired.
#[derive(Clone, Debug)]
pub struct Balls {
    max_live: usize,
    max_striking: usize,
    live: Vec<(BodyId, u64)>,
    striking: Vec<(BodyId, u64)>,
}

impl Balls {
    /// A pool of at most `max_live` balls, `max_striking` of them strikers at once (strike
    /// prediction's shot table holds 64).
    pub fn new(max_live: usize, max_striking: usize) -> Self {
        Self {
            max_live: max_live.max(1),
            max_striking: max_striking.max(1),
            live: Vec::new(),
            striking: Vec::new(),
        }
    }

    /// The balls in the world, oldest first, with the step each was fired.
    pub fn live(&self) -> &[(BodyId, u64)] {
        &self.live
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Forget every ball (the world was restored or rebuilt without them).
    pub fn clear(&mut self) {
        self.live.clear();
        self.striking.clear();
    }

    /// Fire `ball` from `at` with velocity `v` and spin `w` at step `step`. A full pool
    /// destroys its oldest ball first; a full shot table takes its oldest striker out of it.
    pub fn fire(
        &mut self,
        world: &mut World,
        ball: &Ball,
        at: [f32; 3],
        v: [f32; 3],
        w: [f32; 3],
        step: u64,
    ) -> anyhow::Result<BodyId> {
        if self.live.len() >= self.max_live {
            let (old, _) = self.live.remove(0);
            self.striking.retain(|(b, _)| *b != old);
            world.destroy_body(old)?;
        }
        if self.striking.len() >= self.max_striking {
            let (old, _) = self.striking.remove(0);
            if world.body_is_valid(old) {
                world.body_set_projectile(old, None)?;
            }
        }
        let body = world.create_body(&BodyDef {
            linear_velocity: v,
            angular_velocity: w,
            is_bullet: ball.ccd,
            ..BodyDef::dynamic(at)
        })?;
        world.create_sphere_shape(
            body,
            &ShapeDef {
                material: ball.material,
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            ball.radius,
        )?;
        world.body_set_mass_data(body, ball.mass_data())?;
        world.body_set_projectile(body, Some(Projectile::default()))?;
        self.live.push((body, step));
        self.striking.push((body, step));
        Ok(body)
    }

    /// Balls fired `after` steps or more before `now` leave the shot table (spent: through
    /// the structure or stopped in it) and roll on as plain rigid balls.
    pub fn retire_spent(&mut self, world: &mut World, now: u64, after: u64) -> anyhow::Result<()> {
        while self
            .striking
            .first()
            .is_some_and(|&(_, f)| now >= f + after)
        {
            let (ball, _) = self.striking.remove(0);
            if world.body_is_valid(ball) {
                world.body_set_projectile(ball, None)?;
            }
        }
        Ok(())
    }
}
