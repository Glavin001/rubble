//! The terrain demo's scene (`terrain.html`) placed in a [`World`]: a 40 m heightfield of
//! rolling hills and a ridge (collider 0), and `n` boxes, spheres and capsules on a
//! jittered grid above it. The page keeps its renderer, craters and checks over what
//! [`Placed`] returns.
//!
//! Every value is the page's: its seeded random sequence, its math in f64 (with
//! `Math.hypot` as browsers compute it) cast to f32, the same bodies in the same order.

use crate::*;
use serde::{Deserialize, Serialize};

/// The body counts the page offers.
pub const COUNTS: [u32; 4] = [500, 2000, 5000, 10000];
/// The heightfield: 161 × 161 samples, 0.25 m cells.
pub const NX: u32 = 161;
pub const NZ: u32 = 161;
pub const DX: f64 = 0.25;
pub const DZ: f64 = 0.25;

/// A hills scene: how many bodies.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct HillsDef {
    pub n: u32,
}

impl Default for HillsDef {
    fn default() -> Self {
        Self { n: 2000 }
    }
}

impl HillsDef {
    fn known(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            COUNTS.contains(&self.n),
            "hills: {} bodies is not one of {COUNTS:?}",
            self.n
        );
        Ok(())
    }
}

/// The world the scene needs.
pub fn world_def(def: &HillsDef) -> anyhow::Result<WorldDef> {
    def.known()?;
    let n = def.n + 8;
    Ok(WorldDef {
        gravity: [0.0, -9.81, 0.0],
        contacts: ContactStorage::Canonical,
        capacity: WorldCapacity {
            max_bodies: n,
            max_colliders: n,
            max_commands: 4 * n,
            max_hull_vertices: 65536,
            ..WorldCapacity::default()
        },
        ..WorldDef::default()
    })
}

/// The ground height at (x, z): rolling hills and a ridge along x = 6.
pub fn hills(x: f64, z: f64) -> f64 {
    1.4 * (0.22 * x).sin() * (0.18 * z).cos()
        + 0.5 * (0.55 * x + 1.0).sin() * (0.43 * z).sin()
        + 0.9 * (-((x - 6.0) * (x - 6.0)) / 8.0).exp() * (0.15 * z).cos()
}

/// The heightfield's samples (row-major, x fastest), as the page builds them.
pub fn heights() -> Vec<f32> {
    let (x0, z0) = (-0.5 * (NX - 1) as f64 * DX, -0.5 * (NZ - 1) as f64 * DZ);
    let mut h = Vec::with_capacity((NX * NZ) as usize);
    for j in 0..NZ {
        for i in 0..NX {
            h.push(hills(x0 + i as f64 * DX, z0 + j as f64 * DZ) as f32);
        }
    }
    h
}

/// `Math.hypot` as browsers compute it: scaled by the largest magnitude, the squares
/// summed with compensation.
fn js_hypot(v: &[f64]) -> f64 {
    let max = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for x in v {
        let n = x.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// A placed body: its handle, slot and starting pose.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacedBody {
    pub id: u64,
    pub slot: u32,
    pub p: [f32; 3],
    pub q: [f32; 4],
}

/// What the page needs of the placed scene.
#[derive(Clone, Debug, Serialize)]
pub struct Placed {
    /// The heightfield shape (collider 0) and its grid.
    pub terrain: u64,
    pub nx: u32,
    pub nz: u32,
    pub dx: f32,
    pub dz: f32,
    pub bodies: Vec<PlacedBody>,
    /// The page's random sequence (an LCG) where placement left it: the craters and
    /// throws continue it.
    pub seed: u32,
}

/// Place the scene in `world` (made from [`world_def`], empty): its solver, the
/// heightfield and the bodies.
pub fn place(world: &mut World, def: &HillsDef) -> anyhow::Result<Placed> {
    def.known()?;
    anyhow::ensure!(
        world.body_count() == 0,
        "the hills are placed in an empty world"
    );
    let mut solver = world.solver();
    solver.dt = (1.0f64 / 60.0) as f32;
    solver.iterations = 10;
    solver.flags = engine_schema::FLAG_SPEC_VELOCITY;
    world.set_solver(solver);
    let shape = ShapeDef {
        material: SurfaceMaterial {
            friction: 0.8,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    let ground = world.create_body(&BodyDef::default())?;
    let field = HeightField::new(NX, NZ, DX as f32, DZ as f32, heights())?;
    let terrain = world.create_height_field_shape(ground, &shape, &field, Transform::IDENTITY)?;
    anyhow::ensure!(
        world.shape_collider(terrain)? == 0,
        "the terrain must be collider 0"
    );

    // Bodies on a jittered grid above the hills, each of mass 1 with the inertia given
    // here (not derived from its shape).
    let mut seed: u64 = 12345;
    let mut rand = || {
        seed = (seed * 1664525 + 1013904223) & 0xffff_ffff;
        seed as f64 / 4294967296.0
    };
    let n = def.n;
    let side = (n as f64 / 6.0).sqrt().ceil() as u32;
    let spacing = 1.1;
    let mut bodies = Vec::with_capacity(n as usize);
    for k in 0..n {
        let layer = k / (side * side);
        let cell = k % (side * side);
        let half_side = side as f64 / 2.0;
        let x = ((cell % side) as f64 - half_side + 0.5) * spacing + (rand() - 0.5) * 0.3;
        let z = ((cell / side) as f64 - half_side + 0.5) * spacing + (rand() - 0.5) * 0.3;
        let y = 4.0 + layer as f64 * 1.0 + hills(x, z);
        let (a, b) = (rand() * 6.283, rand() * 6.283);
        let sa = (a / 2.0).sin();
        let q = [b.cos() * sa, b.sin() * sa * 0.5, 0.6 * sa, (a / 2.0).cos()];
        let l = js_hypot(&q);
        let q = q.map(|c| c / l);
        let m = 1.0f64;
        let p = [x as f32, y as f32, z as f32];
        let qf = q.map(|c| c as f32);
        let id = world.create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: p,
            rotation: qf,
            ..BodyDef::default()
        })?;
        let (geometry, inertia) = match k % 3 {
            0 => {
                let h = [
                    0.15 + 0.15 * rand(),
                    0.12 + 0.12 * rand(),
                    0.15 + 0.15 * rand(),
                ];
                let i = [
                    (m * (h[1] * h[1] + h[2] * h[2])) / 3.0,
                    (m * (h[0] * h[0] + h[2] * h[2])) / 3.0,
                    (m * (h[0] * h[0] + h[1] * h[1])) / 3.0,
                ];
                let g = Geometry::Box {
                    half: h.map(|c| c as f32),
                    transform: Transform::IDENTITY,
                };
                (g, i)
            }
            1 => {
                let r = 0.15 + 0.1 * rand();
                let i = (m * r * r) / 2.5;
                let g = Geometry::Sphere {
                    center: [0.0; 3],
                    radius: r as f32,
                };
                (g, [i; 3])
            }
            _ => {
                let r = 0.12f64;
                let hh = 0.18 + 0.1 * rand();
                let len = 2.0 * (hh + r);
                let g = Geometry::Capsule {
                    center: [0.0; 3],
                    radius: r as f32,
                    half_len: hh as f32,
                };
                let across = (m * (3.0 * r * r + len * len)) / 12.0;
                (g, [across, (m * r * r) / 2.0, across])
            }
        };
        world.create_shape(id, &shape, geometry)?;
        let mass = MassData::principal(m as f32, [0.0; 3], inertia.map(|c| c as f32));
        world.body_set_mass_data(id, mass)?;
        bodies.push(PlacedBody {
            id: id.to_bits(),
            slot: id.index(),
            p,
            q: qf,
        });
    }
    Ok(Placed {
        seed: seed as u32,
        terrain: terrain.to_bits(),
        nx: NX,
        nz: NZ,
        dx: DX as f32,
        dz: DZ as f32,
        bodies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hypot_is_the_length() {
        for v in [
            [0.3, -0.2, 0.6, 0.7],
            [1e-200, 2e-200, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
        ] {
            let direct = v.iter().map(|x: &f64| x * x).sum::<f64>().sqrt();
            let h = js_hypot(&v);
            assert!(
                (h - direct).abs() <= 2.0 * f64::EPSILON * direct || direct == 0.0,
                "{h} vs {direct}"
            );
        }
        assert_eq!(js_hypot(&[3.0, 4.0]), 5.0);
        assert_eq!(js_hypot(&[0.0; 4]), 0.0);
    }

    #[test]
    fn heights_cover_the_grid_row_major() {
        let h = heights();
        assert_eq!(h.len(), (NX * NZ) as usize);
        let (x0, z0) = (-20.0, -20.0);
        assert_eq!(h[0], hills(x0, z0) as f32);
        assert_eq!(h[1], hills(x0 + DX, z0) as f32);
        assert_eq!(h[NX as usize], hills(x0, z0 + DZ) as f32);
    }

    #[test]
    fn only_the_offered_counts() {
        assert!(world_def(&HillsDef { n: 2000 }).is_ok());
        assert!(world_def(&HillsDef { n: 2001 }).is_err());
    }
}
