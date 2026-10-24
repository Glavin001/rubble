//! "Gauntlet": a drivable car down a street of masonry facades being shelled.
//! Each shell blasts a patch of facade ahead of the car into the street, and
//! directed debris (`engine_schema::guidance`) steers the free-flying blocks
//! to land around the car, on the parked cars and behind it, rather than
//! wherever the blast threw them. The page (vehicle.html `?scene=gauntlet`)
//! builds the bodies from `layout`, launches `blast`'s blocks and uploads
//! `zones`.
//!
//! Facade blocks are static, held by the building's frame, until a blast
//! frees them: each blasted block becomes dynamic with its launch velocity
//! (as the vehicle page's cannon launches a parked ball), so a facade loses
//! the panel that was hit and nothing else.
//!
//! Street frame: the road runs along +Z, centred on x = 0; facades stand on
//! both sides with static building cores behind them; the ground is y = 0.

use engine_schema::guidance::{GuidanceSettings, ZoneDesc, ZoneShape};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Half the street's width to the facades' street faces (m).
pub const STREET_HALF: f32 = 8.0;
/// Facade block half extents (x thickness, y height, z length) and mass (kg).
pub const BLOCK_HALF: [f32; 3] = [0.25, 0.25, 0.4];
pub const BLOCK_MASS: f32 = 150.0;
/// Rows of blocks per facade (12 m).
pub const ROWS: u32 = 24;
/// Building spans along the street (z from, z to), the same on both sides.
pub const SPANS: [(f32, f32); 4] = [(0.0, 30.0), (40.0, 70.0), (80.0, 110.0), (120.0, 150.0)];
/// Where the player starts (and heads +Z).
pub const SPAWN: [f32; 3] = [-2.5, 3.5, -25.0];
/// Blast radius on the facade (m) and the fastest block it throws (m/s).
const BLAST_RADIUS: f32 = 1.6;
const BLAST_SPEED: f32 = 7.5;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Block {
    pub p: [f32; 3],
    pub half: [f32; 3],
    pub mass: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Parked {
    pub p: [f32; 3],
    pub yaw: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Layout {
    pub spawn: [f32; 3],
    pub blocks: Vec<Block>,
    /// Static building cores behind the facades.
    pub cores: Vec<Block>,
    pub parked: Vec<Parked>,
}

/// The street: facade blocks in running bond, cores and parked cars.
pub fn layout() -> Layout {
    let [hx, hy, hz] = BLOCK_HALF;
    let mut blocks = Vec::new();
    let mut cores = Vec::new();
    for side in [-1.0f32, 1.0] {
        let x = side * (STREET_HALF + hx);
        for (z0, z1) in SPANS {
            for row in 0..ROWS {
                // Odd rows shift half a block and hold one fewer.
                let shift = if row % 2 == 1 { hz } else { 0.0 };
                let n = ((z1 - z0) / (2.0 * hz)).floor() as u32 - (row % 2);
                for k in 0..n {
                    let z = z0 + shift + hz + 2.0 * hz * k as f32;
                    // 1 mm gaps so the stack starts without overlap.
                    let y = hy + row as f32 * (2.0 * hy + 0.001);
                    blocks.push(Block {
                        p: [x, y, z],
                        half: BLOCK_HALF,
                        mass: BLOCK_MASS,
                    });
                }
            }
            let height = ROWS as f32 * 2.0 * hy;
            let depth = 8.0;
            cores.push(Block {
                p: [
                    side * (STREET_HALF + 2.0 * hx + 0.01 + depth),
                    height / 2.0,
                    (z0 + z1) / 2.0,
                ],
                half: [depth, height / 2.0, (z1 - z0) / 2.0],
                mass: 0.0,
            });
        }
    }
    let parked = [
        (-5.6, 22.0, 0.0),
        (5.6, 48.0, std::f32::consts::PI),
        (-5.6, 75.0, 0.0),
        (5.6, 96.0, std::f32::consts::PI),
        (-5.6, 128.0, 0.0),
        (5.6, 140.0, std::f32::consts::PI),
    ]
    .map(|(x, z, yaw)| Parked {
        p: [x, 2.2, z],
        yaw,
    });
    Layout {
        spawn: SPAWN,
        blocks,
        cores,
        parked: parked.to_vec(),
    }
}

/// The zones: a ring around the player car (led ahead of it), each parked
/// car's roof and an arc behind the player. Bodies are world body indices;
/// `chassis_y` is the player chassis centre's height over the ground.
pub fn zones(player: u32, chassis_y: f32, parked: &[u32]) -> Vec<ZoneDesc> {
    use std::f32::consts::PI;
    let mut z = vec![
        // All around the car, clear of it (vehicle frame: +Z forward).
        ZoneDesc {
            anchor_body: Some(player),
            anchor_relative: true,
            offset: [0.0, -chassis_y, 0.0],
            shape: ZoneShape::Ring {
                inner: 2.5,
                outer: 6.0,
                height: 0.0,
            },
            weight: 2.0,
            ..ZoneDesc::default()
        },
        // Behind the car: debris crashing down where it just was.
        ZoneDesc {
            anchor_body: Some(player),
            anchor_relative: true,
            offset: [0.0, -chassis_y, 0.0],
            shape: ZoneShape::Arc {
                inner: 3.0,
                outer: 9.0,
                height: 0.0,
                from: 0.8 * PI,
                to: 1.2 * PI,
            },
            weight: 1.0,
            ..ZoneDesc::default()
        },
    ];
    for &b in parked {
        z.push(ZoneDesc {
            anchor_body: Some(b),
            anchor_relative: true,
            offset: [0.0, 0.8, 0.0],
            shape: ZoneShape::Box {
                half_extents: [0.8, 0.15, 1.9],
            },
            weight: 1.5,
            quota: Some(8),
            ..ZoneDesc::default()
        });
    }
    z
}

/// Steering strength of the page's select: 0 subtle (the shipped defaults),
/// 1 cinematic, 2 exaggerated (to see the mechanism).
pub fn settings(strength: u32) -> GuidanceSettings {
    let base = GuidanceSettings {
        min_flight: 0.5,
        min_speed: 0.5,
        ..GuidanceSettings::default()
    };
    match strength {
        0 => base,
        1 => GuidanceSettings {
            launch_kick: 0.12,
            launch_cone: 6.0f32.to_radians(),
            accel_max_g: 0.08,
            ..base
        },
        _ => GuidanceSettings {
            launch_kick: 0.3,
            launch_cone: 16.0f32.to_radians(),
            accel_max_g: 0.25,
            ..base
        },
    }
}

fn hash(mut x: u32) -> f32 {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    ((x >> 22) ^ x) as f32 / u32::MAX as f32
}

/// The street's shelling: which blocks are still in their facades, so a
/// later blast does not throw a block that already fell.
#[wasm_bindgen]
pub struct Gauntlet {
    layout: Layout,
    spent: Vec<bool>,
    shots: u32,
}

impl Gauntlet {
    /// Blast `k` aimed ahead of a car at `car` (x, z) moving at `vz` along the
    /// street: its centre and per-block launches `(block, velocity, spin)`.
    pub fn blast_at(
        &mut self,
        car: [f32; 2],
        vz: f32,
    ) -> ([f32; 3], Vec<(u32, [f32; 3], [f32; 3])>) {
        let k = self.shots;
        self.shots += 1;
        // Alternate sides, at a seeded height; aim where the car will be when
        // the debris comes down (its flight from that height at the blast's
        // lift), give or take 4 m: the steering has near misses to pull in.
        let side = if k % 2 == 0 { 1.0 } else { -1.0 };
        let height = ROWS as f32 * 2.0 * BLOCK_HALF[1];
        let y = 3.0 + (height - 5.0) * hash(k * 3 + 2);
        let lift = 0.35 * BLAST_SPEED;
        let flight = (lift + (lift * lift + 2.0 * 9.81 * y).sqrt()) / 9.81;
        let want = car[1] + vz.max(0.0) * flight + 8.0 * (hash(k * 3 + 1) - 0.5);
        // On the nearest facade span.
        let z = SPANS
            .iter()
            .map(|&(a, b)| want.clamp(a + 2.0, b - 2.0))
            .min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs()))
            .unwrap();
        let x = side * STREET_HALF;
        let centre = [x, y, z];
        let toward_street = [-side, 0.0, 0.0];
        let mut out = Vec::new();
        for (i, b) in self.layout.blocks.iter().enumerate() {
            if self.spent[i] || (b.p[0] * side) < 0.0 {
                continue;
            }
            let d = [b.p[0] - x, b.p[1] - y, b.p[2] - z];
            let r = (d[1] * d[1] + d[2] * d[2]).sqrt();
            if r > BLAST_RADIUS {
                continue;
            }
            self.spent[i] = true;
            let n = hash(i as u32 ^ (k << 16));
            // Out of the wall into the street, spread radially, a little up.
            let radial = if r > 1e-3 {
                [0.0, d[1] / r, d[2] / r]
            } else {
                [0.0; 3]
            };
            let mut dir = [
                toward_street[0] + 0.7 * radial[0],
                0.35 + 0.4 * radial[1].max(-0.6),
                0.7 * radial[2] + 0.25 * (n - 0.5),
            ];
            let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            dir.iter_mut().for_each(|c| *c /= l);
            let speed = BLAST_SPEED * (1.0 - r / BLAST_RADIUS).sqrt().max(0.35) * (0.8 + 0.4 * n);
            let v = dir.map(|c| c * speed);
            // A slow tumble (up to ¼ turn a second), so neighbours from the
            // same patch fly apart instead of knocking into each other.
            let w = [0, 1, 2].map(|a| 3.0 * (hash(i as u32 * 3 + a + 7) - 0.5));
            out.push((i as u32, v, w));
        }
        (centre, out)
    }
}

#[wasm_bindgen]
impl Gauntlet {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Gauntlet {
        let layout = layout();
        let n = layout.blocks.len();
        Gauntlet {
            layout,
            spent: vec![false; n],
            shots: 0,
        }
    }

    /// The street (`Layout` JSON).
    pub fn layout(&self) -> String {
        serde_json::to_string(&self.layout).unwrap()
    }

    /// Back to an unshelled street (scene reset).
    pub fn reset(&mut self) {
        self.spent.fill(false);
        self.shots = 0;
    }

    /// The next blast ahead of a car at (x, z) moving at `vz` m/s along the
    /// street: `[cx, cy, cz, n, (block, vx, vy, vz, wx, wy, wz) × n]`.
    pub fn blast(&mut self, x: f32, z: f32, vz: f32) -> Vec<f32> {
        let (c, list) = self.blast_at([x, z], vz);
        let mut out = vec![c[0], c[1], c[2], list.len() as f32];
        for (i, v, w) in list {
            out.push(i as f32);
            out.extend_from_slice(&v);
            out.extend_from_slice(&w);
        }
        out
    }

    /// `zones` as `ZoneDesc` JSON.
    pub fn zones(&self, player: u32, chassis_y: f32, parked: &[u32]) -> String {
        serde_json::to_string(&zones(player, chassis_y, parked)).unwrap()
    }

    /// `settings(strength)` as `GuidanceSettings` JSON, `enabled` as given.
    pub fn settings(strength: u32, enabled: bool) -> String {
        serde_json::to_string(&GuidanceSettings {
            enabled,
            ..settings(strength)
        })
        .unwrap()
    }
}

impl Default for Gauntlet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facades_stand_clear_of_the_street_and_cores() {
        let l = layout();
        assert!(
            l.blocks.len() > 5000 && l.blocks.len() < 9000,
            "{}",
            l.blocks.len()
        );
        for b in &l.blocks {
            assert!(
                b.p[0].abs() - b.half[0] >= STREET_HALF - 1e-4,
                "in the street: {:?}",
                b.p
            );
            assert!(b.p[1] - b.half[1] >= -1e-4);
            for c in &l.cores {
                let gap = (b.p[0] - c.p[0]).abs() - b.half[0] - c.half[0];
                assert!(
                    gap > 0.0 || (b.p[2] - c.p[2]).abs() > c.half[2],
                    "overlaps a core"
                );
            }
        }
        for p in &l.parked {
            assert!(p.p[0].abs() < STREET_HALF - 1.5);
        }
        for z in zones(1, 0.6, &[2, 3]) {
            z.to_gpu().unwrap();
        }
        for s in 0..3 {
            settings(s).validate().unwrap();
        }
    }

    #[test]
    fn a_blast_throws_its_patch_into_the_street_once() {
        let mut g = Gauntlet::new();
        let (c, a) = g.blast_at([-2.5, 10.0], 12.0);
        assert!(c[2] > 10.0 && c[2] < 40.0, "ahead of the car: {c:?}");
        assert!(a.len() >= 6, "{} blocks", a.len());
        for (i, j, _) in &a {
            let b = g.layout.blocks[*i as usize];
            let j = j.map(|c| c * b.mass);
            assert!(b.p[0] * c[0] > 0.0, "on the blasted side");
            // Toward the street and up, at most the blast speed.
            assert!(j[0] * c[0] < 0.0 && j[1] > 0.0);
            let v = (j[0] * j[0] + j[1] * j[1] + j[2] * j[2]).sqrt() / b.mass;
            assert!(v <= BLAST_SPEED * 1.21 && v > 2.0, "{v}");
        }
        // The same patch again throws nothing it already threw.
        g.shots = 0;
        let (_, again) = g.blast_at([-2.5, 10.0], 12.0);
        assert!(again.iter().all(|(i, ..)| a.iter().all(|(k, ..)| k != i)));
    }
}
