//! The legacy rubble web demo's 3D scenes (`crates/rubble-scenes/src/scenes_3d.rs`),
//! rebuilt body for body so the two engines can be compared on the same scenarios.
//!
//! Each scene brings its own ground as legacy did (a 100×1×100 static box, or the plane
//! y = 0), so `populate` adds none. Sizes, masses, positions, velocities and per-body
//! friction match legacy; friction becomes one material per distinct value, and both
//! engines combine a pair as sqrt(μa·μb). [`scene_setup`](crate::scene_setup) applies
//! legacy's `SimConfig` (gravity −9.81, dt 1/60, 12 iterations); legacy also substepped
//! adaptively (up to 8 for the grid scenes), which this engine does not.
//!
//! Legacy drew the random scenes (Scatter, Scatter Boxes, Grid Boxes jitter) from an
//! unseeded RNG, so no two runs matched; these use a fixed seed with the same
//! distributions, so every run, and both hosts, see the same bodies.

use engine_schema::scene::SceneBuilder;
use engine_schema::*;
use glam::{Quat, Vec3};

/// Scene group of the ported scenes; `populate` adds no ground for it.
pub const LEGACY_GROUP: &str = "Legacy rubble";

/// Legacy `SimConfig` gravity.
pub const LEGACY_GRAVITY_Y: f32 = -9.81;
/// Legacy `SimConfig::solver_iterations`.
pub const LEGACY_ITERATIONS: u32 = 12;

/// Legacy shapes with per-body friction, written straight into the world.
struct Legacy<'b, 'a> {
    b: &'b mut SceneBuilder<'a>,
    /// friction → material id; 0.5 is the default material.
    materials: Vec<(f32, u32)>,
}

enum Shape {
    Box(Vec3),
    Sphere(f32),
    Capsule { half_height: f32, radius: f32 },
}

impl<'b, 'a> Legacy<'b, 'a> {
    fn new(b: &'b mut SceneBuilder<'a>) -> Self {
        Self {
            b,
            materials: vec![(0.5, 0)],
        }
    }

    fn material(&mut self, friction: f32) -> u32 {
        if let Some(&(_, id)) = self.materials.iter().find(|(f, _)| *f == friction) {
            return id;
        }
        let id = self.b.world_mut().add_material(Material {
            friction,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        });
        self.materials.push((friction, id));
        id
    }

    #[allow(clippy::too_many_arguments)]
    fn body(
        &mut self,
        pos: Vec3,
        rot: Quat,
        shape: Shape,
        mass: f32,
        friction: f32,
        vel: Vec3,
    ) -> u32 {
        let (inertia, volume, shape_type, params) = match shape {
            Shape::Box(half) => {
                let h = half * 2.0;
                (
                    Vec3::new(
                        (1.0 / 12.0) * mass * (h.y * h.y + h.z * h.z),
                        (1.0 / 12.0) * mass * (h.x * h.x + h.z * h.z),
                        (1.0 / 12.0) * mass * (h.x * h.x + h.y * h.y),
                    ),
                    8.0 * half.x * half.y * half.z,
                    ShapeType::Box,
                    [half.x, half.y, half.z, half.length()],
                )
            }
            Shape::Sphere(r) => (
                Vec3::splat(0.4 * mass * r * r),
                (4.0 / 3.0) * std::f32::consts::PI * r.powi(3),
                ShapeType::Sphere,
                [r, 0.0, 0.0, r],
            ),
            Shape::Capsule {
                half_height,
                radius,
            } => (
                Vec3::splat(0.2 * mass * (half_height + radius).powi(2)),
                std::f32::consts::PI * radius * radius * (2.0 * half_height)
                    + (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3),
                ShapeType::Capsule,
                [radius, half_height, 0.0, radius + half_height],
            ),
        };
        let mut state = BodyState::dynamic(pos, rot, mass, inertia);
        if mass <= 0.0 {
            state.inv_inertia = [0.0; 3];
            state.flags = 0;
        }
        state.linear_vel = vel.to_array();
        let material = self.material(friction);
        let world = self.b.world_mut();
        let body = world.add_body(state);
        world.add_collider(Collider {
            body,
            shape: shape_type as u32,
            material,
            generation: 0,
            local_pos: [0.0; 3],
            density: if mass > 0.0 {
                mass / volume.max(1e-8)
            } else {
                0.0
            },
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params,
        });
        body
    }

    /// Legacy `box_desc`: full width/height/depth.
    #[allow(clippy::too_many_arguments)]
    fn cube(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        w: f32,
        h: f32,
        d: f32,
        mass: f32,
        friction: f32,
    ) -> u32 {
        let half = Vec3::new(w, h, d) * 0.5;
        self.body(
            Vec3::new(x, y, z),
            Quat::IDENTITY,
            Shape::Box(half),
            mass,
            friction,
            Vec3::ZERO,
        )
    }

    /// Legacy's 100×1×100 static ground box centred at height `y`.
    fn ground_box(&mut self, y: f32) {
        self.cube(0.0, y, 0.0, 100.0, 1.0, 100.0, 0.0, 0.5);
    }

    /// Legacy's static plane y = 0 (default material, friction 0.5).
    fn ground_plane(&mut self) {
        self.b.add_ground_plane(0.0);
    }
}

/// Deterministic uniform [0, 1) floats (SplitMix64) standing in for legacy's `rand::rng()`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
}

const SEED: u64 = 0x5275_6262_6C65; // "Rubble"

pub fn empty(_b: &mut SceneBuilder<'_>) {}

pub fn ground(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_box(0.0);
    l.cube(0.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5);
}

/// Eleven boxes launched along +x at 10 m/s, friction 5.0 down to 0.0.
pub fn dynamic_friction(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_box(0.0);
    for x in 0..=10 {
        let friction = 5.0 - (x as f32 / 10.0 * 5.0);
        let half = Vec3::new(1.0, 0.5, 1.0) * 0.5;
        l.body(
            Vec3::new(0.0, 0.75, -30.0 + x as f32 * 2.0),
            Quat::IDENTITY,
            Shape::Box(half),
            1.0,
            friction,
            Vec3::new(10.0, 0.0, 0.0),
        );
    }
}

/// Eleven boxes resting on a 30° ramp (friction 1.0), friction 0.25 up to 0.5.
pub fn static_friction(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    let rot = Quat::from_rotation_z(30.0_f32.to_radians());
    let tangent = rot * Vec3::X;
    let normal = rot * Vec3::Y;
    l.ground_box(0.0);
    l.body(
        Vec3::ZERO,
        rot,
        Shape::Box(Vec3::new(20.0, 0.5, 12.0)),
        0.0,
        1.0,
        Vec3::ZERO,
    );
    for i in 0..=10 {
        let friction = i as f32 / 10.0 * 0.25 + 0.25;
        let p = tangent * -12.0 + Vec3::new(0.0, 0.0, -10.0 + i as f32 * 2.0) + normal * 1.05;
        l.cube(p.x, p.y, p.z, 1.0, 1.0, 1.0, 1.0, friction);
    }
}

/// A 16-row 2D pyramid of 1×0.5×0.5 boxes (136 boxes).
pub fn pyramid(b: &mut SceneBuilder<'_>) {
    const SIZE: usize = 16;
    let mut l = Legacy::new(b);
    l.ground_box(-0.5);
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            l.cube(
                x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                y as f32 * 0.85 + 0.5,
                0.0,
                1.0,
                0.5,
                0.5,
                1.0,
                0.5,
            );
        }
    }
}

/// Ten unit cubes dropped in a column with 0.5 m gaps.
pub fn stack(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_box(0.0);
    for i in 0..10 {
        l.cube(0.0, i as f32 * 1.5 + 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5);
    }
}

/// Four cubes of side 1, 2, 4, 8, all of mass 1, smallest at the bottom.
pub fn stack_ratio(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_box(0.0);
    let mut top_y = 0.5;
    let mut size = 1.0;
    for _ in 0..4 {
        let half = size * 0.5;
        let center_y = top_y + half;
        l.cube(0.0, center_y, 0.0, size, size, size, 1.0, 0.5);
        top_y = center_y + half;
        size *= 2.0;
    }
}

/// 3,000 random spheres, cubes and capsules dropped onto the plane.
pub fn scatter(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_plane();
    let mut rng = Rng(SEED);
    for _ in 0..3000 {
        let x = (rng.f32() - 0.5) * 12.0;
        let y = 3.0 + rng.f32() * 15.0;
        let z = (rng.f32() - 0.5) * 12.0;
        let pos = Vec3::new(x, y, z);
        match rng.below(3) {
            0 => {
                let r = 0.3 + rng.f32() * 0.4;
                l.body(pos, Quat::IDENTITY, Shape::Sphere(r), 1.0, 0.5, Vec3::ZERO);
            }
            1 => {
                let size = 0.4 + rng.f32() * 0.8;
                l.cube(x, y, z, size, size, size, 1.0, 0.5);
            }
            _ => {
                let shape = Shape::Capsule {
                    half_height: 0.3,
                    radius: 0.2,
                };
                l.body(pos, Quat::IDENTITY, shape, 1.0, 0.5, Vec3::ZERO);
            }
        }
    }
}

/// 4,000 random boxes (sides 0.2–0.7) dropped onto the plane.
pub fn scatter_boxes(b: &mut SceneBuilder<'_>) {
    let mut l = Legacy::new(b);
    l.ground_plane();
    let mut rng = Rng(SEED);
    for _ in 0..4000 {
        let x = (rng.f32() - 0.5) * 16.0;
        let y = 2.5 + rng.f32() * 20.0;
        let z = (rng.f32() - 0.5) * 16.0;
        let sx = 0.2 + rng.f32() * 0.5;
        let sy = 0.2 + rng.f32() * 0.5;
        let sz = 0.2 + rng.f32() * 0.5;
        l.cube(x, y, z, sx, sy, sz, 1.0, 0.5);
    }
}

/// Legacy's box grids: `nx × ny × nz` cubes of side 0.42 on a 0.5 pitch, each layer
/// shifted by `layer_shift`, positions jittered by up to ±`jitter`/2.
fn grid(b: &mut SceneBuilder<'_>, n: [usize; 3], layer_shift: Vec3, jitter: f32) {
    let [nx, ny, nz] = n;
    let mut l = Legacy::new(b);
    l.ground_plane();
    let side = 0.42_f32;
    let pitch = side + 0.08;
    let ox = -((nx - 1) as f32 * pitch) * 0.5;
    let oz = -((nz - 1) as f32 * pitch) * 0.5;
    let base_y = side * 0.5 + 0.03;
    let mut rng = Rng(SEED);
    for j in 0..ny {
        let shift = layer_shift * j as f32;
        for i in 0..nx {
            for k in 0..nz {
                let mut p = Vec3::new(
                    ox + i as f32 * pitch + shift.x,
                    base_y + j as f32 * pitch,
                    oz + k as f32 * pitch + shift.z,
                );
                if jitter > 0.0 {
                    p += Vec3::new(rng.f32() - 0.5, rng.f32() - 0.5, rng.f32() - 0.5) * jitter;
                }
                l.cube(p.x, p.y, p.z, side, side, side, 1.0, 0.5);
            }
        }
    }
}

/// 8 × 24 × 8 jittered cubes (1,536).
pub fn grid_boxes(b: &mut SceneBuilder<'_>) {
    grid(b, [8, 24, 8], Vec3::ZERO, 0.012);
}

/// 20 × 25 × 20 cubes (10,000).
pub fn grid_10k_boxes(b: &mut SceneBuilder<'_>) {
    grid(b, [20, 25, 20], Vec3::ZERO, 0.0);
}

/// 50 × 40 × 50 cubes (100,000). Not in the legacy demo: its 10k grid at the scale the
/// engine targets, built the same way (same cubes, pitch and ground plane).
pub fn grid_100k_boxes(b: &mut SceneBuilder<'_>) {
    grid(b, [50, 40, 50], Vec3::ZERO, 0.0);
}

/// 12 × 22 × 12 cubes (3,168), each layer shifted (0.038, 0, 0.026) from the one below.
pub fn slanted_grid_boxes(b: &mut SceneBuilder<'_>) {
    grid(b, [12, 22, 12], Vec3::new(0.038, 0.0, 0.026), 0.0);
}
