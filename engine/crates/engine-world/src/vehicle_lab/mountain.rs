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
//! Every shape is static. Two terrains ([`Terrain`]):
//!
//! - **Ledges** (the first mountain, kept for the measurements made on it): slabs (3 m
//!   thick oriented boxes along a piecewise-linear profile with shelves, drops and
//!   cliffs), canyon walls, and rocks standing on the slabs: spikes (convex hulls from a
//!   small palette), boulders (spheres) and blocks (boxes). A car sliding onto a shelf
//!   stops there.
//! - **Rolling** (the page's default): a gully down a mountainside with no shelf or cliff.
//!   The fall line swells smoothly (a sum of three seeded waves along the normal, its
//!   local slope never more than a few degrees off the mean and never under 38° on the
//!   default 50°), the floor is a flat bed between two banks rising 18° to the walls, and
//!   the rocks are rounded: wide low humps (big buried spheres) that throw a car up and
//!   sideways, boulders, and a few crags on the banks. A car keeps falling and bouncing.

use glam::{DVec3, Quat, Vec3};

/// The mountain's ground: see the module's doc.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Terrain {
    /// Shelves, drops and cliffs (the first mountain; the measurements' scene).
    #[default]
    Ledges,
    /// A smooth gully with rounded rocks and no shelf: the car keeps falling.
    Rolling,
}

impl Terrain {
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "ledges" => Some(Self::Ledges),
            "rolling" => Some(Self::Rolling),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Ledges => "ledges",
            Self::Rolling => "rolling",
        }
    }
}

/// The mountain's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountainSpec {
    pub seed: u64,
    /// The ground's kind. The default stays [`Terrain::Ledges`] so that the offline tools
    /// and the measurements keep their scene; the lab page asks for rolling.
    pub terrain: Terrain,
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
    /// Where the run starts, metres down the fall line from the lip (0: off the lip). A
    /// start kilometres down tests the mountain far from the World's origin.
    pub start: f32,
}

impl Default for MountainSpec {
    fn default() -> Self {
        Self {
            seed: 1,
            terrain: Terrain::Ledges,
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
            start: 0.0,
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
/// The canyon's lid: its height above the chunk's line (the walls' tops, m), and its
/// shapes' tag (a density no rock has: a page draws nothing for it).
pub const LID_UP: f32 = 13.5;
pub const LID_TAG: f32 = 1.0;
/// Rolling: the flat bed's share of the width, the banks' rise to the walls (degrees), a
/// profile segment's length (m) and the lowest local slope the swell may make (degrees:
/// over the rock's kinetic friction angle, 37°, so a sliding car keeps sliding).
const BED: f32 = 0.5;
const BANK_DEG: f32 = 18.0;
const SEGMENT: f32 = 4.0;
const MIN_SLOPE_DEG: f64 = 38.0;

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
        self.start = if self.start.is_finite() {
            self.start.clamp(0.0, 100_000.0)
        } else {
            0.0
        };
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

    /// Rolling: the largest local slope change the swell makes (radians).
    fn swell_slope(&self) -> f64 {
        let want = 4.0 + 10.0 * f64::from(self.roughness);
        let room = (f64::from(self.steepness) - MIN_SLOPE_DEG).max(2.0);
        want.min(room).to_radians()
    }

    /// Rolling: the ground's offset (m) along the fall line's upward normal at arc `s`
    /// from the lip: three seeded waves (71, 29 and 13 m) whose slopes add to at most
    /// [`Self::swell_slope`], faded in over the first 30 m so the run leaves the lip the
    /// same way. A function of the arc alone, so chunks join.
    fn swell(&self, s: f64) -> f64 {
        use std::f64::consts::TAU;
        let dmax = self.swell_slope();
        let mut r = Rng::new(self.seed, -2000, 11);
        let mut d = 0.0;
        for (lambda, w) in [(71.0, 0.5), (29.0, 0.3), (13.0, 0.2)] {
            let phase = f64::from(r.next()) * TAU;
            d += w * dmax * lambda / TAU * (TAU * s / lambda + phase).sin();
        }
        let f = (s / 30.0).clamp(0.0, 1.0);
        d * f * f * (3.0 - 2.0 * f)
    }

    /// Rolling: the chunk's ground profile, points (z, y) relative to its origin (the
    /// fall line's point at its start), every [`SEGMENT`] m of arc.
    fn profile_rolling(&self, k: i64) -> Vec<[f32; 2]> {
        let l = f64::from(self.chunk);
        let n = (l / f64::from(SEGMENT)).round().max(3.0) as usize;
        let t = self.theta();
        let up = DVec3::new(0.0, t.cos(), t.sin());
        let o = self.line(k as f64 * l);
        (0..=n)
            .map(|j| {
                let s = (k as f64 + j as f64 / n as f64) * l;
                let p = self.line(s) - o + up * self.swell(s);
                [p.z as f32, p.y as f32]
            })
            .collect()
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
        if self.terrain == Terrain::Rolling {
            return MountainChunk {
                index: k,
                origin,
                shapes: self.rolling_shapes(k),
            };
        }
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
        // The canyon walls, along the chunk's own line, and the lid: a slab across the
        // canyon on top of its walls (LID_UP m above the chunk's line), so that nothing
        // thrown up in a crash leaves the canyon. A part over a wall had no ground and fell
        // without end (past 1,000 km/h within a run).
        self.walls_and_lid(&mut shapes);
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

    /// The canyon's walls and lid along chunk `k`'s own line (both terrains).
    fn walls_and_lid(&self, shapes: &mut Vec<MountainShape>) {
        let (l, w) = (self.chunk, self.width);
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
        if std::env::var("LAB_NO_LID").is_err() {
            shapes.push(MountainShape {
                geom: RockGeom::Box {
                    half: Vec3::new(0.5 * w + 3.0, 0.5, 0.5 * l + 0.5),
                },
                pos: mid + rot * Vec3::new(0.0, LID_UP, 0.0),
                rot,
                tag: LID_TAG,
                loose: false,
            });
        }
    }

    /// Rolling: chunk `k`'s shapes (k ≥ 0), in its frame. Per profile segment a bed slab
    /// and two bank slabs hinged on its edges; the walls and lid; then humps, boulders,
    /// crags on the banks and loose rocks, from the chunk's own stream.
    fn rolling_shapes(&self, k: i64) -> Vec<MountainShape> {
        let w = self.width;
        let prof = self.profile_rolling(k);
        let bed = BED * w;
        let beta = BANK_DEG.to_radians();
        // A bank reaches into the wall, so nothing falls between them.
        let bank = (0.5 * w - 0.5 * bed) / beta.cos() + 1.0;
        let mut shapes = Vec::new();
        // (a, b, rot, normal, len) per segment.
        let mut segs: Vec<([f32; 2], [f32; 2], Quat, Vec3, f32)> = Vec::new();
        for p in prof.windows(2) {
            let (a, b) = (p[0], p[1]);
            let d = Vec3::new(0.0, b[1] - a[1], b[0] - a[0]);
            let len = d.length();
            let rot = Quat::from_rotation_x((-d.y).atan2(d.z));
            let normal = rot * Vec3::Y;
            let mid = Vec3::new(0.0, 0.5 * (a[1] + b[1]), 0.5 * (a[0] + b[0]));
            // The bed, reaching under the banks. Its overlap with the next segment is
            // short: a joint's lip is the overlap times the change of slope.
            shapes.push(MountainShape {
                geom: RockGeom::Box {
                    half: Vec3::new(0.5 * bed + 0.5, 0.5 * SLAB, 0.5 * len + 0.1),
                },
                pos: mid - normal * (0.5 * SLAB),
                rot,
                tag: 2400.0,
                loose: false,
            });
            for side in [-1.0f32, 1.0] {
                let tilt = rot * Quat::from_rotation_z(side * beta);
                let hinge = mid + rot * Vec3::new(side * 0.5 * bed, 0.0, 0.0);
                shapes.push(MountainShape {
                    geom: RockGeom::Box {
                        half: Vec3::new(0.5 * bank + 0.25, 0.5 * SLAB, 0.5 * len + 0.3),
                    },
                    pos: hinge + tilt * Vec3::new(side * (0.5 * bank - 0.25), -0.5 * SLAB, 0.0),
                    rot: tilt,
                    tag: 2400.0,
                    loose: false,
                });
            }
            segs.push((a, b, rot, normal, len));
        }
        self.walls_and_lid(&mut shapes);
        let total: f32 = segs.iter().map(|s| s.4).sum();
        let mut r = Rng::new(self.seed, k, 3);
        // A point on the ground: `at` m along the chunk's profile, `x` across (on a bank
        // it rises with the bank). Its height over the bed and the segment's frame.
        let ground = |at: f32, x: f32| -> (Vec3, Quat, Vec3) {
            let mut at = at;
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
            let rise = (x.abs() - 0.5 * bed).max(0.0) * beta.tan();
            let on =
                Vec3::new(x, a[1] + f * (b[1] - a[1]), a[0] + f * (b[0] - a[0])) + normal * rise;
            (on, rot, normal)
        };
        let rough = self.roughness;
        // The landing zone under the lip stays clear, so every run starts the same way.
        let clear = |on: Vec3| k == 0 && on.z < 12.0;
        // Humps: wide, low swells of rock (a buried sphere). Their faces are at most 25°
        // off the ground, so a car meets a ramp, not a step, and leaves it up and aside.
        for _ in 0..(1.0 + 3.0 * rough).round() as usize {
            let (at, x) = (r.next() * total, r.range(-0.5 * bed, 0.5 * bed));
            let radius = r.range(5.0, 12.0);
            let h = (radius * r.range(0.04, 0.09)).min(1.1);
            let (on, _, normal) = ground(at, x);
            if clear(on) {
                continue;
            }
            shapes.push(MountainShape {
                geom: RockGeom::Sphere { radius },
                pos: on + normal * (h - radius),
                rot: Quat::IDENTITY,
                tag: 2400.0,
                loose: false,
            });
        }
        // Boulders, part buried, anywhere across.
        for _ in 0..(1.0 + 5.0 * rough).round() as usize {
            let (at, x) = (r.next() * total, r.range(-0.5 * w + 2.0, 0.5 * w - 2.0));
            let radius = r.range(0.35, 1.1);
            let sink = r.range(-0.1, 0.4);
            let (on, _, normal) = ground(at, x);
            if clear(on) {
                continue;
            }
            shapes.push(MountainShape {
                geom: RockGeom::Sphere { radius },
                pos: on - normal * (sink * radius),
                rot: Quat::IDENTITY,
                tag: 2300.0,
                loose: false,
            });
        }
        // Crags on the banks: they turn a car running up a bank back into the gully.
        for _ in 0..(2.0 * rough).round() as usize {
            let side = if r.next() < 0.5 { -1.0 } else { 1.0 };
            let at = r.next() * total;
            let x = side * r.range(0.5 * bed + 2.0, 0.5 * w - 1.5);
            let p = (r.next() * PALETTE as f32) as usize % PALETTE;
            let turn = Quat::from_rotation_y(r.range(0.0, std::f32::consts::TAU));
            let (on, rot, _) = ground(at, x);
            if clear(on) {
                continue;
            }
            shapes.push(MountainShape {
                geom: RockGeom::Spike { palette: p },
                pos: on,
                rot: rot * Quat::from_rotation_z(side * beta) * turn,
                tag: 2300.0,
                loose: false,
            });
        }
        for _ in 0..self.loose {
            let (at, x) = (r.next() * total, r.range(-0.5 * bed, 0.5 * bed));
            let h = Vec3::new(r.range(0.15, 0.3), r.range(0.12, 0.25), r.range(0.15, 0.3));
            let turn = Quat::from_rotation_y(r.range(0.0, std::f32::consts::TAU));
            let (on, rot, normal) = ground(at, x);
            if clear(on) {
                continue;
            }
            shapes.push(MountainShape {
                geom: RockGeom::Box { half: h },
                pos: on + normal * (h.y + 0.05),
                rot: rot * turn,
                tag: 2300.0,
                loose: true,
            });
        }
        shapes
    }

    /// Where a ray (`o` world, `d` unit) first meets chunk `c`'s drawn shapes (boxes and
    /// spheres; a crag is skipped, the ground behind it is met), and where it leaves the
    /// chunk's lid (the lid is not drawn: a shot from a camera above it starts under it).
    pub fn ray(&self, c: &MountainChunk, o: DVec3, d: Vec3) -> (Option<f32>, Option<f32>) {
        let (mut hit, mut lid) = (None::<f32>, None::<f32>);
        for s in &c.shapes {
            if s.loose {
                continue;
            }
            let rel = (o - c.origin).as_vec3() - s.pos;
            let span = match &s.geom {
                RockGeom::Box { half } => {
                    let inv = s.rot.inverse();
                    let (p, v) = (inv * rel, inv * d);
                    let (mut t0, mut t1) = (f32::NEG_INFINITY, f32::INFINITY);
                    let mut miss = false;
                    for i in 0..3 {
                        if v[i].abs() < 1e-9 {
                            if p[i].abs() > half[i] {
                                miss = true;
                            }
                            continue;
                        }
                        let (a, b) = ((-half[i] - p[i]) / v[i], (half[i] - p[i]) / v[i]);
                        t0 = t0.max(a.min(b));
                        t1 = t1.min(a.max(b));
                    }
                    (!miss && t0 <= t1).then_some((t0, t1))
                }
                RockGeom::Sphere { radius } => {
                    let b = rel.dot(d);
                    let q = b * b - (rel.length_squared() - radius * radius);
                    (q >= 0.0).then(|| (-b - q.sqrt(), -b + q.sqrt()))
                }
                RockGeom::Spike { .. } => None,
            };
            let Some((t0, t1)) = span else { continue };
            if t1 <= 0.0 {
                continue;
            }
            if s.tag == LID_TAG {
                lid = Some(lid.map_or(t1, |x: f32| x.max(t1)));
            } else if t0 > 0.0 {
                hit = Some(hit.map_or(t0, |x: f32| x.min(t0)));
            }
        }
        (hit, lid)
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
        let r = MountainSpec {
            terrain: Terrain::Rolling,
            ..m
        };
        assert_eq!(r.chunk(7), r.chunk(7));
        assert_ne!(r.chunk(7).shapes, r.chunk(8).shapes);
    }

    /// Rolling: no shelf. Every profile segment slopes within the swell's bound of the
    /// mean (and never under 35° at the defaults), only goes down the mountain, and a
    /// chunk's end is the next chunk's start.
    #[test]
    fn the_rolling_ground_has_no_shelf_and_chunks_join() {
        for seed in 0..8 {
            for rough in [0.0, 0.5, 1.0] {
                for steep in [35.0, 50.0, 60.0] {
                    let m = MountainSpec {
                        seed,
                        terrain: Terrain::Rolling,
                        roughness: rough,
                        steepness: steep,
                        ..Default::default()
                    };
                    let bound = m.swell_slope().to_degrees() + 1.0;
                    let mut lowest = f64::INFINITY;
                    for k in 0..200 {
                        let p = m.profile_rolling(k);
                        let o = m.line(k as f64 * f64::from(m.chunk));
                        let o1 = m.line((k + 1) as f64 * f64::from(m.chunk));
                        let next = m.profile_rolling(k + 1)[0];
                        let last = p[p.len() - 1];
                        let end = (o.z + f64::from(last[0]), o.y + f64::from(last[1]));
                        let start = (o1.z + f64::from(next[0]), o1.y + f64::from(next[1]));
                        assert!(
                            (end.0 - start.0).abs() < 2e-3 && (end.1 - start.1).abs() < 2e-3,
                            "seed {seed} chunk {k}: {end:?} vs {start:?}"
                        );
                        for w in p.windows(2) {
                            let a = f64::from(w[0][1] - w[1][1])
                                .atan2(f64::from(w[1][0] - w[0][0]))
                                .to_degrees();
                            assert!(w[1][0] > w[0][0] && w[1][1] < w[0][1]);
                            assert!((a - f64::from(steep)).abs() <= bound, "seed {seed} rough {rough} {steep}° chunk {k}: a segment at {a:.1}°");
                            lowest = lowest.min(a);
                        }
                    }
                    if steep == 50.0 {
                        assert!(lowest >= 37.0, "seed {seed} rough {rough}: {lowest:.1}°");
                    }
                }
            }
        }
    }

    #[test]
    fn a_ray_down_onto_the_rolling_mountain_meets_its_ground_not_its_lid() {
        let m = MountainSpec {
            terrain: Terrain::Rolling,
            ..Default::default()
        };
        let c = m.chunk(3);
        // From 20 m over the chunk's middle (above the lid), straight down along the normal.
        let t = m.theta();
        let up = DVec3::new(0.0, t.cos(), t.sin());
        let o = m.line(3.5 * f64::from(m.chunk)) + up * 20.0;
        let (hit, lid) = m.ray(&c, o, (-up).as_vec3());
        let (hit, lid) = (hit.expect("the ground"), lid.expect("the lid"));
        assert!(
            lid > 20.0 - LID_UP - 1.0 && lid < 20.0 - LID_UP + 1.0,
            "{lid}"
        );
        assert!(hit > 17.0 && hit < 23.0, "{hit}");
    }

    #[test]
    fn every_profile_joins_the_fall_line_and_only_goes_down_the_mountain() {
        for seed in 0..8 {
            for rough in [0.0, 0.5, 1.0] {
                let m = MountainSpec {
                    seed,
                    terrain: Terrain::Ledges,
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
