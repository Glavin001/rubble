//! The vehicle lab's "infinite mountain": a steep, rough, rocky slope that never ends,
//! made of chunks along the fall line. A chunk is a pure function of (seed, index), so
//! chunks can be made ahead of the car and dropped behind it in any order, and a run is
//! the same on every host.
//!
//! Frame: the cliff's lip is the origin; the fall line runs along +z and down. Chunk `k`
//! (k ≥ 0) covers the fall line's arc `[k L, (k + 1) L]`; chunk −1 is the plateau behind
//! the lip. A chunk's shapes are placed relative to its own origin (the fall line's
//! point at its start, f64), so their local coordinates stay small however far down the
//! mountain the chunk lies.
//!
//! Every shape is static: slabs (the ground itself, 3 m thick oriented boxes along a
//! piecewise-linear profile with ledges, drops and cliffs), canyon walls that keep the
//! car on the mountain, and rocks standing on the slabs: spikes (convex hulls from a
//! small palette), boulders (spheres) and blocks (boxes).

use glam::{DVec3, Quat, Vec3};

/// The mountain's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountainSpec {
    pub seed: u64,
    /// The fall line's mean slope (degrees from horizontal), 35–50.
    pub steepness: f32,
    /// 0–1: how many rocks, ledges and cliffs.
    pub roughness: f32,
    /// The slope's width between the canyon walls (m).
    pub width: f32,
    /// A chunk's length along the fall line (m).
    pub chunk: f32,
    /// Chunks kept ahead of and behind the car's chunk.
    pub ahead: u32,
    pub behind: u32,
    /// The car's speed off the lip (m/s).
    pub launch: f32,
    /// Loose rocks (dynamic rigid bodies) per chunk: looks only; see known bug B1.
    pub loose: u32,
    /// When the car comes to rest on the mountain, a boulder is thrown at it from up the
    /// slope (a dynamic rigid body), so the fall goes on.
    pub rockslide: bool,
}

impl Default for MountainSpec {
    fn default() -> Self {
        Self {
            seed: 1,
            // Past the rock's kinetic friction angle (atan 0.75 = 37°) and at its static one
            // (atan 1.2 = 50°): a sliding car keeps sliding.
            steepness: 50.0,
            roughness: 0.6,
            width: 36.0,
            chunk: 24.0,
            ahead: 8,
            behind: 2,
            launch: 14.0,
            loose: 0,
            rockslide: true,
        }
    }
}

/// A static shape of a chunk, in the chunk's frame.
#[derive(Clone, Debug, PartialEq)]
pub enum RockGeom {
    Box {
        half: Vec3,
    },
    Sphere {
        radius: f32,
    },
    /// A hull of the palette ([`MountainSpec::palette`]).
    Spike {
        palette: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MountainShape {
    pub geom: RockGeom,
    pub pos: Vec3,
    pub rot: Quat,
    /// The renderer's material tag (a density: 2400 slab rock, 2300 pale rock).
    pub tag: f32,
    /// A loose rock (dynamic), not part of the static mountain.
    pub loose: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MountainChunk {
    pub index: i64,
    pub origin: DVec3,
    pub shapes: Vec<MountainShape>,
}

/// Spike hulls in the palette.
pub const PALETTE: usize = 12;
const SLAB: f32 = 3.0;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A deterministic stream of numbers in [0, 1) for (seed, chunk, salt).
struct Rng(u64);

impl Rng {
    fn new(seed: u64, k: i64, salt: u64) -> Self {
        Self(mix(mix(seed)
            ^ mix(k as u64 ^ 0xA5A5_0000)
            ^ mix(salt.wrapping_mul(0x1234_5677))))
    }
    fn next(&mut self) -> f32 {
        self.0 = mix(self.0);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

impl MountainSpec {
    /// The spec with its values held to their ranges.
    pub fn clamped(mut self) -> Self {
        self.steepness = self.steepness.clamp(25.0, 60.0);
        self.roughness = self.roughness.clamp(0.0, 1.0);
        self.width = self.width.clamp(16.0, 80.0);
        self.chunk = self.chunk.clamp(12.0, 48.0);
        self.ahead = self.ahead.clamp(2, 16);
        self.behind = self.behind.clamp(1, 8);
        self.launch = self.launch.clamp(0.0, 60.0);
        self.loose = self.loose.min(16);
        self
    }

    fn theta(&self) -> f64 {
        f64::from(self.steepness).to_radians()
    }

    /// The fall line's point at arc length `s` (m) from the lip.
    pub fn line(&self, s: f64) -> DVec3 {
        let t = self.theta();
        DVec3::new(0.0, -s * t.sin(), s * t.cos())
    }

    /// The chunk holding the fall line's point nearest `p` (world).
    pub fn chunk_of(&self, p: DVec3) -> i64 {
        let t = self.theta();
        let s = p.z * t.cos() - p.y * t.sin();
        (s / f64::from(self.chunk)).floor() as i64
    }

    /// The spike hulls' points (each hull's own frame: y up, the base buried).
    pub fn palette(&self) -> Vec<Vec<Vec3>> {
        (0..PALETTE)
            .map(|i| {
                let mut r = Rng::new(self.seed, -1000 - i as i64, 7);
                let sides = 4 + (r.next() * 3.0) as usize;
                let base = r.range(0.35, 1.1);
                let height = r.range(0.6, 2.0) * (0.6 + 0.4 * base);
                let mut pts = Vec::new();
                for k in 0..sides {
                    let a =
                        (k as f32 + r.range(-0.25, 0.25)) / sides as f32 * std::f32::consts::TAU;
                    let rad = base * r.range(0.7, 1.0);
                    pts.push(Vec3::new(rad * a.cos(), -0.6, rad * a.sin()));
                    // A shoulder ring: the spike is a faceted crag, not a cone.
                    let rs = rad * r.range(0.35, 0.6);
                    pts.push(Vec3::new(
                        rs * a.cos(),
                        height * r.range(0.3, 0.55),
                        rs * a.sin(),
                    ));
                }
                pts.push(Vec3::new(
                    base * r.range(-0.3, 0.3),
                    height,
                    base * r.range(-0.3, 0.3),
                ));
                pts
            })
            .collect()
    }

    /// The chunk's ground profile: points (z, y) relative to its origin, from (0, 0) to
    /// the fall line's point a chunk on.
    fn profile(&self, k: i64) -> Vec<[f32; 2]> {
        let t = self.theta() as f32;
        let l = self.chunk;
        let end = [l * t.cos(), -l * t.sin()];
        let mut r = Rng::new(self.seed, k, 1);
        let pick = r.next();
        let rough = self.roughness;
        let ledge = 0.1 + 0.25 * rough;
        let cliff = 0.05 + 0.2 * rough;
        if k == 0 || pick >= ledge + cliff {
            // A slope with a knee: its middle pushed off the line along the normal.
            let n = if k == 0 {
                0.0
            } else {
                r.range(-2.5, 2.5) * rough
            };
            let f = r.range(0.35, 0.65);
            return vec![
                [0.0, 0.0],
                [f * end[0] + n * t.sin(), f * end[1] + n * t.cos()],
                end,
            ];
        }
        if pick < ledge {
            // A shelf, then a near-vertical drop, then the slope again.
            let shelf = r.range(4.0, 8.0);
            let drop = shelf * t.tan() * r.range(0.5, 0.9) + r.range(1.0, 3.0) * rough;
            return vec![[0.0, 0.0], [shelf, 0.0], [shelf + 0.35, -drop], end];
        }
        // A cliff: a gentler run, a 65–75° face, the slope again.
        let run = r.range(0.3, 0.45) * l;
        let gentle = (t - 15f32.to_radians()).max(0.2);
        let a = [run * gentle.cos(), -run * gentle.sin()];
        let face = r.range(65.0f32, 75.0).to_radians();
        let h = r.range(3.0, 7.0) * (0.5 + 0.5 * rough);
        let b = [a[0] + h / face.tan(), a[1] - h];
        // The face must leave room to reach the chunk's end going down.
        if b[0] < end[0] - 2.0 && b[1] > end[1] + 0.5 {
            vec![[0.0, 0.0], a, b, end]
        } else {
            vec![[0.0, 0.0], a, end]
        }
    }

    /// Chunk `k`: the plateau for −1, the slope for k ≥ 0, nothing below −1.
    pub fn chunk(&self, k: i64) -> MountainChunk {
        let l = self.chunk;
        let w = self.width;
        let mut shapes = Vec::new();
        if k < 0 {
            if k == -1 {
                // The plateau: 40 m of flat rock ending at the lip.
                shapes.push(MountainShape {
                    geom: RockGeom::Box {
                        half: Vec3::new(0.5 * w, 0.5 * SLAB, 20.0),
                    },
                    pos: Vec3::new(0.0, -0.5 * SLAB, -20.0),
                    rot: Quat::IDENTITY,
                    tag: 2400.0,
                    loose: false,
                });
            }
            return MountainChunk {
                index: k,
                origin: DVec3::ZERO,
                shapes,
            };
        }
        let origin = self.line(k as f64 * f64::from(l));
        let prof = self.profile(k);
        // Each profile segment: a slab under it, its upward normal and its rotation.
        let mut segs: Vec<([f32; 2], [f32; 2], Quat, Vec3, f32)> = Vec::new();
        for p in prof.windows(2) {
            let (a, b) = (p[0], p[1]);
            let d = Vec3::new(0.0, b[1] - a[1], b[0] - a[0]);
            let len = d.length();
            if len < 1e-3 {
                continue;
            }
            let phi = (-d.y).atan2(d.z);
            let rot = Quat::from_rotation_x(phi);
            let normal = rot * Vec3::Y;
            let mid = Vec3::new(0.0, 0.5 * (a[1] + b[1]), 0.5 * (a[0] + b[0]));
            shapes.push(MountainShape {
                geom: RockGeom::Box {
                    half: Vec3::new(0.5 * w + 2.0, 0.5 * SLAB, 0.5 * len + 0.3),
                },
                pos: mid - normal * (0.5 * SLAB),
                rot,
                tag: 2400.0,
                loose: false,
            });
            segs.push((a, b, rot, normal, len));
        }
        // The canyon walls, along the chunk's own line.
        let t = self.theta() as f32;
        let rot = Quat::from_rotation_x(t);
        let mid = Vec3::new(0.0, -0.5 * l * t.sin(), 0.5 * l * t.cos());
        for side in [-1.0f32, 1.0] {
            shapes.push(MountainShape {
                geom: RockGeom::Box {
                    half: Vec3::new(1.5, 9.0, 0.5 * l + 0.5),
                },
                pos: mid
                    + Vec3::new(side * (0.5 * w + 1.5), 0.0, 0.0)
                    + rot * Vec3::new(0.0, 4.0, 0.0),
                rot,
                tag: 2400.0,
                loose: false,
            });
        }
        // Rocks on the slabs.
        let mut r = Rng::new(self.seed, k, 2);
        let total: f32 = segs.iter().map(|s| s.4).sum();
        let count = (4.0 + 14.0 * self.roughness).round() as usize;
        for i in 0..count + self.loose as usize {
            let loose = i >= count;
            let mut at = r.next() * total;
            let mut seg = &segs[0];
            for s in &segs {
                seg = s;
                if at <= s.4 {
                    break;
                }
                at -= s.4;
            }
            let (a, b, rot, normal, len) = *seg;
            let f = (at / len).clamp(0.0, 1.0);
            let x = r.range(-0.5 * w + 2.5, 0.5 * w - 2.5);
            let on = Vec3::new(x, a[1] + f * (b[1] - a[1]), a[0] + f * (b[0] - a[0]));
            let kind = r.next();
            let turn = Quat::from_rotation_y(r.range(0.0, std::f32::consts::TAU));
            let tilt = Quat::from_rotation_x(r.range(-0.3, 0.3))
                * Quat::from_rotation_z(r.range(-0.3, 0.3));
            // The landing zone under the lip stays clear, so every run starts the same way.
            if k == 0 && on.z < 10.0 {
                // The stream still advances as for a placed rock.
                let _ = (r.next(), r.next(), r.next());
                continue;
            }
            let shape = if loose {
                let h = Vec3::new(r.range(0.15, 0.3), r.range(0.12, 0.25), r.range(0.15, 0.3));
                MountainShape {
                    geom: RockGeom::Box { half: h },
                    pos: on + normal * (h.y + 0.05),
                    rot: rot * turn,
                    tag: 2300.0,
                    loose: true,
                }
            } else if kind < 0.45 {
                let p = (r.next() * PALETTE as f32) as usize % PALETTE;
                let _ = (r.next(), r.next());
                MountainShape {
                    geom: RockGeom::Spike { palette: p },
                    pos: on,
                    rot: rot * turn * tilt,
                    tag: 2300.0,
                    loose: false,
                }
            } else if kind < 0.75 {
                let rad = r.range(0.35, 1.2);
                let _ = (r.next(), r.next());
                MountainShape {
                    geom: RockGeom::Sphere { radius: rad },
                    pos: on + normal * (0.3 * rad),
                    rot: Quat::IDENTITY,
                    tag: 2300.0,
                    loose: false,
                }
            } else {
                let h = Vec3::new(r.range(0.3, 1.0), r.range(0.3, 0.8), r.range(0.3, 1.0));
                MountainShape {
                    geom: RockGeom::Box { half: h },
                    pos: on + normal * 0.2,
                    rot: rot * turn * tilt,
                    tag: 2300.0,
                    loose: false,
                }
            };
            shapes.push(shape);
        }
        MountainChunk {
            index: k,
            origin,
            shapes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_is_a_function_of_seed_and_index() {
        let m = MountainSpec::default();
        assert_eq!(m.chunk(7), m.chunk(7));
        assert_ne!(m.chunk(7).shapes, m.chunk(8).shapes);
        assert_ne!(m.chunk(7), MountainSpec { seed: 2, ..m }.chunk(7));
    }

    #[test]
    fn every_profile_joins_the_fall_line_and_only_goes_down_the_mountain() {
        for seed in 0..8 {
            for rough in [0.0, 0.5, 1.0] {
                let m = MountainSpec {
                    seed,
                    roughness: rough,
                    ..Default::default()
                };
                for k in 0..200 {
                    let p = m.profile(k);
                    let end = m.line(f64::from(m.chunk));
                    let last = p[p.len() - 1];
                    assert!(
                        (f64::from(last[0]) - end.z).abs() < 1e-3
                            && (f64::from(last[1]) - end.y).abs() < 1e-3
                    );
                    for w in p.windows(2) {
                        assert!(
                            w[1][0] > w[0][0],
                            "seed {seed} chunk {k}: a segment going back up the mountain: {p:?}"
                        );
                        assert!(w[1][1] <= w[0][1] + 1e-4 || k > 0, "chunk 0 only descends");
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_coordinates_stay_small_far_down_the_mountain() {
        let m = MountainSpec::default();
        let c = m.chunk(400);
        assert!(c.origin.length() > 9000.0);
        for s in &c.shapes {
            assert!(s.pos.length() < 80.0, "{:?}", s.pos);
        }
        assert_eq!(m.chunk_of(c.origin + DVec3::new(0.0, -1.0, 1.0)), 400);
    }

    #[test]
    fn the_palette_hulls_are_convex_hull_input_of_few_points() {
        for pts in MountainSpec::default().palette() {
            assert!(pts.len() >= 9 && pts.len() <= 13);
        }
    }
}
