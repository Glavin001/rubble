//! The brick ring drive (`brick-ring-drive.html`): the 110k brick ring of the rigid demos
//! (`engine-scenes` `brick_ring_110k`, demo.html `?scene=ring110k`) with drivable 2 kHz
//! node–beam cars (the explicit solver on raycast wheels) in the same World. A car drives
//! into the wall: it deforms, and it pushes the bricks, both ways, on the GPU.
//!
//! The ring is the demo's layout ([`RingSpec::demo`]: radius 80, 40 courses, a row less
//! every 4 courses from 10 at the base; 110,441 bricks of 1 × 0.5 × 0.5 m) scaled by
//! [`RingSpec::scale`] about the origin, its bricks' density set by [`RingSpec::density`].
//! The World is the demo's: 1/60 s steps of 4 iterations, persistent contacts, its sparse
//! colour start, no sleep ([`ring_world_def`]).
//!
//! The cars keep their own ground (the explicit vehicles' plane at the ground slab's
//! top): the slab's collider is in its own collision category, which the vehicles' mask
//! leaves out, so it is never a proxy.

use crate::explicit_scene::place_vehicle;
use crate::{
    BodyDef, ContactStorage, ExplicitVehiclesDef, GpuContext, KillBounds, ShapeDef, World, WorldDef,
};
use engine_schema::convex::CollisionFilter;
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::{Quat, Vec3};

/// The ground slab's top (the demo's: bricks rest on y = 0.5).
pub const GROUND: f32 = 0.5;
/// The ground slab's collision category (the vehicles' mask leaves it out).
const GROUND_CATEGORY: u16 = 0x8000;

/// The ring: the demo's `brick_ring(radius, courses, tier, rows)` in brick units (a brick
/// 1 × 0.5 × 0.5, rows 0.52 apart, 1.02 along a course), then scaled by `scale` (m per
/// unit), each brick of `density` kg/m³.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingSpec {
    pub scale: f32,
    pub radius: f32,
    pub courses: u32,
    pub tier: u32,
    pub rows: u32,
    pub density: f32,
    /// The demo's smashing ball (its rolling sphere at 30 m/s); off in the drive.
    pub ball: bool,
}

impl RingSpec {
    /// demo.html's `ring110k`: 0.25 kg bricks (density 1) and the ball.
    pub fn demo() -> Self {
        Self {
            scale: 1.0,
            radius: 80.0,
            courses: 40,
            tier: 4,
            rows: 10,
            density: 1.0,
            ball: true,
        }
    }

    /// The bricks' centres and yaws (about +y), in the demo's order.
    pub fn bricks(&self) -> Vec<(Vec3, f32)> {
        let mut out = Vec::new();
        let s = self.scale;
        for c in 0..self.courses {
            let depth = self.rows.saturating_sub(c / self.tier.max(1)).max(1);
            for j in 0..depth {
                let r = self.radius + 0.25 + 0.52 * j as f32;
                let n = ((2.0 * std::f32::consts::PI * (r - 0.25)) / 1.02)
                    .floor()
                    .max(1.0) as u32;
                for i in 0..n {
                    let a = ((i as f32 + if c % 2 == 1 { 0.5 } else { 0.0 }) / n as f32)
                        * 2.0
                        * std::f32::consts::PI;
                    out.push((
                        Vec3::new(
                            s * r * a.cos(),
                            GROUND + s * (0.25 + 0.5 * c as f32),
                            s * r * a.sin(),
                        ),
                        a + std::f32::consts::FRAC_PI_2,
                    ));
                }
            }
        }
        out
    }

    /// A brick's half extents (m).
    pub fn half(&self) -> [f32; 3] {
        [0.5 * self.scale, 0.25 * self.scale, 0.25 * self.scale]
    }

    /// The wall's inner face (m from the centre) and its height.
    pub fn inner(&self) -> f32 {
        self.scale * self.radius
    }
    pub fn height(&self) -> f32 {
        self.scale * 0.5 * self.courses as f32
    }
}

/// The drive's ring (`brick-ring-drive.html`): the demo's layout and brick count.
pub fn drive_ring() -> RingSpec {
    RingSpec {
        ball: false,
        ..RingSpec::demo()
    }
}

/// Coupling room per car ([`ring_coupling`]): proxy bodies and shapes a car may meet in
/// one World step.
pub const DRIVE_SHAPES: u32 = 1024;

/// The demo's World for `bodies` bodies (demo-config.mjs `ring110k`: `sceneCap`,
/// `demoSolver`, `parseDemoQuery`'s features; `ring_world_def_is_the_demos` holds the values demo-config.mjs gives).
pub fn ring_world_def(bodies: u32) -> WorldDef {
    let b = bodies.max(4096).next_power_of_two();
    let cap_contacts = 1u32 << 21;
    let mut def = WorldDef {
        gravity: [0.0, -10.0, 0.0],
        contacts: ContactStorage::Persistent,
        kill_bounds: Some(KillBounds::below(-50.0)),
        ..WorldDef::default()
    };
    def.capacity.max_bodies = b;
    def.capacity.max_colliders = (2 * bodies).max(8192).next_power_of_two();
    def.capacity.max_contacts = (16 * bodies)
        .max(32768)
        .next_power_of_two()
        .min(cap_contacts);
    def.capacity.max_candidates = (32 * bodies)
        .max(65536)
        .next_power_of_two()
        .min(cap_contacts);
    def.capacity.max_joints = 64;
    def.capacity.max_colors = 32;
    def.capacity.max_commands = 2048;
    def.solver.dt = 1.0 / 60.0;
    def.solver.iterations = 4;
    def.solver.flags = engine_schema::FLAG_REUSE_CONTACTS;
    def.solver.alpha = 0.99;
    def.solver.beta_lin = 10_000.0;
    def.solver.sleep_lin = 0.15;
    def.solver.sleep_ang = 0.15;
    def.solver.sleep_time = 1.0e6;
    def.features.sparse_colors = 10;
    def
}

/// Where a car starts: its front face's middle, its speed (m/s) and quarter turns about +y
/// from +z ([`place_vehicle`]).
#[derive(Clone, Copy, Debug)]
pub struct CarStart {
    pub front: [f32; 3],
    pub speed: f32,
    pub turns: u32,
}

/// The cars' starts for `count` cars (1 or 2) facing the wall's inner face along +z,
/// `run_up` m from it, at `speed`: side by side 2.6 m apart (a second car's front a car
/// length behind, so they meet the wall one after the other).
pub fn car_starts(ring: &RingSpec, count: usize, run_up: f32, speed: f32) -> Vec<CarStart> {
    let face = ring.inner();
    (0..count)
        .map(|k| CarStart {
            front: [
                if count > 1 {
                    (k as f32 - 0.5) * 2.6
                } else {
                    0.0
                },
                GROUND,
                face - run_up - 5.0 * k as f32,
            ],
            speed,
            turns: 0,
        })
        .collect()
}

/// The brick ring drive.
pub struct BrickRingDrive {
    pub world: World,
    pub ring: RingSpec,
    pub cars: Vec<CarStart>,
    /// The vehicles as spawned (the World's explicit solver holds them in this order).
    pub vehicles: Vec<ExplicitBeams>,
    pub bricks: usize,
}

/// The coupling the drive needs per car (`CouplingCapacity`): a car in the wall meets
/// hundreds of bricks; more fails the step with the limit's name.
pub fn ring_coupling(shapes: u32) -> ExplicitVehiclesDef {
    ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity {
            bodies: shapes,
            shapes,
            planes: 16384,
        },
        mask: !GROUND_CATEGORY,
        // A drive: the cars stay awake while driven; sleep off keeps every frame's work
        // the same whatever the input (and the crash lab's bit-for-bit runs).
        sleep: None,
        ..ExplicitVehiclesDef::default()
    }
}

/// The ring on `gpu` with `templates` (car `k` a copy of `templates[k % len]`, facing +z)
/// at `cars`; `tweak` adjusts the World ([`ring_world_def`]: timing, features);
/// `coupling`: [`ring_coupling`]. No templates: the ring alone (the baseline).
pub fn brick_ring_drive(
    gpu: GpuContext,
    ring: RingSpec,
    tweak: impl FnOnce(&mut WorldDef),
    templates: &[ExplicitBeams],
    cars: &[CarStart],
    coupling: &ExplicitVehiclesDef,
) -> anyhow::Result<BrickRingDrive> {
    let bricks = ring.bricks();
    let n = bricks.len() as u32 + 2 + u32::from(ring.ball);
    let mut wdef = ring_world_def(n);
    tweak(&mut wdef);
    let mut world = World::new(gpu, &wdef)?;
    // The demo's ground: a static slab 0.5 m thick under y = 0.5, twice the ring's extent.
    let extent = (2.0 * (ring.inner() + ring.scale * 0.52 * ring.rows as f32 + 20.0)).max(100.0);
    let g = world.create_body(&BodyDef {
        position: [0.0, 0.0, 0.0],
        ..BodyDef::default()
    })?;
    let mat = |friction: f32| crate::SurfaceMaterial {
        friction,
        ..Default::default()
    };
    world.create_box_shape(
        g,
        &ShapeDef {
            material: mat(0.5),
            density: 0.0,
            filter: CollisionFilter {
                categories: GROUND_CATEGORY,
                ..CollisionFilter::default()
            },
            ..ShapeDef::default()
        },
        [extent, 0.5, extent],
    )?;
    let defs: Vec<BodyDef> = bricks
        .iter()
        .map(|(p, yaw)| BodyDef {
            rotation: Quat::from_rotation_y(-yaw).to_array(),
            ..BodyDef::dynamic(p.to_array())
        })
        .collect();
    let ids = world.create_bodies(&defs)?;
    let brick = ShapeDef {
        material: mat(0.5),
        density: ring.density,
        ..ShapeDef::default()
    };
    let half = ring.half();
    for id in &ids {
        world.create_box_shape(*id, &brick, half)?;
    }
    if ring.ball {
        // The demo's ball: radius courses/5, density 10, rolling at 30 m/s toward +z.
        let r = ring.scale * ring.courses as f32 / 5.0;
        let z = -(ring.scale * (ring.radius + 0.52 * ring.rows as f32) + r + 2.0 * ring.scale);
        let ball = world.create_body(&BodyDef {
            linear_velocity: [0.0, 0.0, 30.0],
            angular_velocity: [30.0 / r, 0.0, 0.0],
            ..BodyDef::dynamic([0.0, GROUND + r, z])
        })?;
        world.create_sphere_shape(
            ball,
            &ShapeDef {
                material: mat(0.5),
                density: 10.0,
                ..ShapeDef::default()
            },
            [0.0; 3],
            r,
        )?;
    }
    world.enable_sleep_archive(None)?;
    let vehicles: Vec<ExplicitBeams> = if templates.is_empty() {
        Vec::new()
    } else {
        cars.iter()
            .enumerate()
            .map(|(k, c)| {
                let mut sim = place_vehicle(
                    &templates[k % templates.len()],
                    Vec3::from(c.front),
                    c.turns,
                    c.speed,
                    GROUND,
                );
                if let Some(d) = sim.drive.as_mut() {
                    d.roll(c.speed);
                }
                sim
            })
            .collect()
    };
    if !vehicles.is_empty() {
        world.add_explicit_vehicles(&vehicles, coupling)?;
    }
    Ok(BrickRingDrive {
        world,
        ring,
        cars: cars.to_vec(),
        vehicles,
        bricks: bricks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_demo_ring_has_its_110440_bricks() {
        // demo.html's ring110k: 110,442 bodies with its ground and ball.
        assert_eq!(RingSpec::demo().bricks().len(), 110_440);
    }

    #[test]
    fn ring_world_def_is_the_demos() {
        // demo-config.mjs for ?scene=ring110k (sceneCap(131072), demoSolver with sleep
        // off, parseDemoQuery's features): only the sparse colour start differs from
        // EngineFeatures::default().
        let d = ring_world_def(110_442);
        assert_eq!(d.capacity.max_bodies, 131_072);
        assert_eq!(d.capacity.max_colliders, 262_144);
        assert_eq!(d.capacity.max_contacts, 2_097_152);
        assert_eq!(d.capacity.max_candidates, 2_097_152);
        assert_eq!((d.solver.iterations, d.solver.flags), (4, 1));
        assert_eq!(d.solver.sleep_time, 1.0e6);
        assert!(matches!(d.contacts, ContactStorage::Persistent));
        let mut f = d.features.clone();
        f.sparse_colors = 32;
        assert_eq!(
            format!("{f:?}"),
            format!("{:?}", WorldDef::default().features)
        );
        assert_eq!(d.features.sparse_colors, 10);
    }
}
