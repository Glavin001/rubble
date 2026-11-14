//! The rigid benchmark scenes of the web demo (`engine-webgpu/demo.html`) and the native
//! viewers: one set of generators (brick ring, brick gables, wall smash, pyramid, box pile,
//! box columns) in the engine's Y-up frame, the demo's scene table (titles, cameras, solver
//! constants, capacities) and the body records `World` takes.
//!
//! The generators compute in f64 and emit pieces (bricks, boxes, balls); the web demo turns
//! them into [`Body`] descriptions ([`bodies`]) and records ([`records`]), the native
//! three-avbd scenes (`lib.rs`) into `SceneBuilder` bodies with their own masses.
//! `tests/web_scenes.rs` checks their geometric and physical invariants.

use serde::Serialize;
use std::f64::consts::PI;

/// A brick: 1 × 0.5 × 0.5 m, its long axis yawed `angle` about +Y.
pub const BRICK_HALF: [f64; 3] = [0.5, 0.25, 0.25];

/// What a generator places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Piece {
    /// A brick ([`BRICK_HALF`]) at `p`, yawed `angle` about +Y.
    Brick {
        p: [f64; 3],
        angle: f64,
        mass: f64,
        density: f64,
    },
    /// A box of half extents `half` at `p`, rotated `q` (x, y, z, w).
    Box {
        p: [f64; 3],
        half: [f64; 3],
        q: [f64; 4],
        mass: f64,
    },
    /// A ball of `radius` and `density` at `p`, launched at `v` with spin `w`.
    Ball {
        p: [f64; 3],
        radius: f64,
        density: f64,
        v: [f64; 3],
        w: [f64; 3],
    },
}

/// A brick's yaw about +Y as a quaternion (x, y, z, w).
pub fn brick_quaternion(angle: f64) -> [f64; 4] {
    [0.0, -(angle / 2.0).sin(), 0.0, (angle / 2.0).cos()]
}

/// The spin that rolls a ball of `radius` along +Z at `speed` without slipping
/// (v + w × (0, −r, 0) = 0).
pub fn rolling_angular_velocity(speed: f64, radius: f64) -> [f64; 3] {
    [speed / radius, 0.0, 0.0]
}

/// A solid sphere's mass.
pub fn sphere_mass(radius: f64, density: f64) -> f64 {
    density * (4.0 / 3.0) * PI * radius.powi(3)
}

/// A solid box's principal moments (half extents `h`).
pub fn box_inertia(mass: f64, h: [f64; 3]) -> [f64; 3] {
    [
        mass * (h[1] * h[1] + h[2] * h[2]) / 3.0,
        mass * (h[0] * h[0] + h[2] * h[2]) / 3.0,
        mass * (h[0] * h[0] + h[1] * h[1]) / 3.0,
    ]
}

/// Paper Fig. 1: running-bond courses of bricks on concentric rings of `radius`, a stepped
/// profile (one ring fewer every `tier` courses, `rows` at the base) and a ball rolling at
/// the ring.
pub fn ring(radius: f64, courses: u32, tier: u32, rows: u32) -> Vec<Piece> {
    let mut out = Vec::new();
    for c in 0..courses {
        let depth = (rows as i64 - (c / tier) as i64).max(1) as u32;
        for j in 0..depth {
            let r = radius + 0.25 + 0.52 * j as f64;
            let n = ((2.0 * PI * (r - 0.25)) / 1.02).floor() as u32;
            for i in 0..n {
                let a = ((i as f64 + (c % 2) as f64 * 0.5) / n as f64) * 2.0 * PI;
                out.push(Piece::Brick {
                    p: [r * a.cos(), 0.75 + 0.5 * c as f64, r * a.sin()],
                    angle: a + PI / 2.0,
                    mass: 0.25,
                    density: 1.0,
                });
            }
        }
    }
    let r = courses as f64 / 5.0;
    out.push(Piece::Ball {
        p: [0.0, 0.5 + r, -(radius + 0.52 * rows as f64 + r + 2.0)],
        radius: r,
        density: 10.0,
        v: [0.0, 0.0, 30.0],
        w: rolling_angular_velocity(30.0, r),
    });
    out
}

/// Paper Fig. 3: stepped brick gables (`base` bricks at the bottom) on a `columns` × `rows`
/// grid, each course centred over the joints below, and two balls rolling at them.
pub fn gables(columns: u32, rows: u32, base: u32) -> Vec<Piece> {
    let (pitch_x, pitch_z) = (base as f64 + 2.0, 5.0);
    let (cols, rws) = (columns as f64, rows as f64);
    let mut out = Vec::new();
    for cx in 0..columns {
        for cz in 0..rows {
            let x0 = (cx as f64 - (cols - 1.0) / 2.0) * pitch_x;
            let z = (cz as f64 - (rws - 1.0) / 2.0) * pitch_z;
            for k in 0..base {
                let m = base - k;
                for i in 0..m {
                    out.push(Piece::Brick {
                        p: [
                            x0 + (i as f64 - (m as f64 - 1.0) / 2.0) * 1.01,
                            0.75 + 0.5 * k as f64,
                            z,
                        ],
                        angle: 0.0,
                        mass: 0.25,
                        density: 1.0,
                    });
                }
            }
        }
    }
    let r = base as f64 / 7.0;
    for cx in [cols / 2.0 - 2.0, cols / 2.0 + 1.0] {
        let x = (cx - (cols - 1.0) / 2.0) * pitch_x;
        let z = -((rws - 1.0) / 2.0) * pitch_z - r - 12.0;
        out.push(Piece::Ball {
            p: [x, 0.5 + r, z],
            radius: r,
            density: 50.0,
            v: [0.0, 0.0, 30.0],
            w: rolling_angular_velocity(30.0, r),
        });
    }
    out
}

/// A running-bond wall `width` bricks long, `courses` high and two deep, of `brick_mass`
/// (`density`) bricks, and a 2 m ball (density 20) thrown at it.
pub fn wall(width: u32, courses: u32, brick_mass: f64, density: f64) -> Vec<Piece> {
    let mut out = Vec::new();
    for course in 0..courses {
        let offset = if course % 2 == 0 { 0.0 } else { 0.5 };
        for x in 0..width {
            for depth in 0..2 {
                out.push(Piece::Brick {
                    p: [
                        (x as f64 - width as f64 / 2.0) + offset,
                        0.75 + course as f64 * 0.5,
                        depth as f64 * 0.5,
                    ],
                    angle: 0.0,
                    mass: brick_mass,
                    density,
                });
            }
        }
    }
    out.push(Piece::Ball {
        p: [0.0, 5.0, -25.0],
        radius: 2.0,
        density: 20.0,
        v: [0.0, 4.0, 30.0],
        w: [0.0; 3],
    });
    out
}

/// three-avbd's brick pyramid: `levels` square courses of bricks.
pub fn pyramid(levels: u32) -> Vec<Piece> {
    let mut out = Vec::new();
    for level in 0..levels {
        let n = levels - level;
        for x in 0..n {
            for z in 0..n {
                out.push(Piece::Brick {
                    p: [
                        (x as f64 - (n as f64 - 1.0) / 2.0) * 1.05,
                        0.75 + level as f64 * 0.5,
                        (z as f64 - (n as f64 - 1.0) / 2.0) * 0.55,
                    ],
                    angle: 0.0,
                    mass: 0.25,
                    density: 1.0,
                });
            }
        }
    }
    out
}

/// The 10-bit Morton spread (bits of `x` three apart).
fn spread(x: u32) -> u32 {
    let mut x = x & 0x3ff;
    x = (x | (x << 16)) & 0x30000ff;
    x = (x | (x << 8)) & 0x300f00f;
    x = (x | (x << 4)) & 0x30c30c3;
    (x | (x << 2)) & 0x9249249
}

/// three-avbd's box pile: an `n` × `n` × `h` lattice of randomly sized and turned boxes
/// (its LCG, seed 12345), generated in its (z, x, y) order and placed in Morton order.
pub fn box_pile(n: u32, h: u32) -> Vec<Piece> {
    let mut seed = 12345u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        seed as f64 / 4294967296.0
    };
    let mut specs = Vec::with_capacity((n * n * h) as usize);
    for z in 0..h {
        for x in 0..n {
            for y in 0..n {
                let half = [
                    0.25 + rand() * 0.35,
                    0.25 + rand() * 0.35,
                    0.25 + rand() * 0.35,
                ];
                let p = [
                    (x as f64 - n as f64 / 2.0) * 1.6,
                    2.0 + z as f64 * 1.6,
                    (y as f64 - n as f64 / 2.0) * 1.6,
                ];
                let q = [rand() - 0.5, rand() - 0.5, rand() - 0.5, rand() - 0.5];
                let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                let l = if l > 0.0 { l } else { 1.0 };
                specs.push((p, half, q.map(|c| c / l)));
            }
        }
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (p, _, _) in &specs {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let morton = |p: &[f64; 3]| {
        let q = |k: usize| {
            ((p[k] - lo[k]) / (hi[k] - lo[k]).max(1e-9) * 1024.0)
                .floor()
                .clamp(0.0, 1023.0) as u32
        };
        spread(q(0)) | (spread(q(1)) << 1) | (spread(q(2)) << 2)
    };
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.sort_by_key(|&i| morton(&specs[i].0));
    order
        .into_iter()
        .map(|i| {
            let (p, half, q) = specs[i];
            Piece::Box {
                p,
                half,
                q,
                mass: 8.0 * half[0] * half[1] * half[2],
            }
        })
        .collect()
}

/// three-avbd's box columns: an `n` × `n` grid of `h`-high unit-box columns.
pub fn box_columns(n: u32, h: u32) -> Vec<Piece> {
    let mut out = Vec::with_capacity((n * n * h) as usize);
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                out.push(Piece::Box {
                    p: [
                        (x as f64 - n as f64 / 2.0) * 1.5,
                        1.0 + y as f64,
                        (z as f64 - n as f64 / 2.0) * 1.5,
                    ],
                    half: [0.5; 3],
                    q: [0.0, 0.0, 0.0, 1.0],
                    mass: 1.0,
                });
            }
        }
    }
    out
}

/// A body's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f64 },
    Box { half: [f64; 3] },
}

/// A body of a built scene (scene order; body 0 is the static ground slab).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub dynamic: bool,
    pub p: [f64; 3],
    pub q: [f64; 4],
    pub v: [f64; 3],
    pub w: [f64; 3],
    pub mass: f64,
    /// Principal moments of inertia (centre at the origin, axes aligned).
    pub inertia: [f64; 3],
    pub shape: Shape,
    /// The shape's density.
    pub density: f64,
    /// Continuous collision.
    pub bullet: bool,
}

/// The ground slab's half height: thin under the ring and gables (three-avbd's), deep
/// elsewhere so soft Jacobi contacts cannot tunnel. Its top is at y 0.5.
pub fn ground_half_height(scene: &str) -> f64 {
    if scene.starts_with("ring") || scene.starts_with("gables") {
        0.5
    } else {
        25.0
    }
}

/// The ground slab's half extent across for a scene of `extent`.
pub fn ground_half_extent(extent: f64) -> f64 {
    100f64.max(2.0 * extent)
}

/// The pieces of built scene `name` (`levels`: the pyramid's).
pub fn pieces(name: &str, levels: u32) -> anyhow::Result<Vec<Piece>> {
    Ok(match name {
        "ring" => ring(12.0, 20, 2, 10),
        "ring110k" => ring(80.0, 40, 4, 10),
        "gables" => gables(4, 8, 20),
        "gables506k" => gables(16, 68, 30),
        "pyramid" => pyramid(levels),
        "pile" => box_pile(40, 20),
        "columns" => box_columns(100, 10),
        "wall" => wall(40, 25, 0.6 * 0.25, 0.6),
        _ => anyhow::bail!("{name} is not a built scene (try {})", BUILT.join(", ")),
    })
}

/// The scenes [`pieces`] builds (the rest of [`scenes`] are snapshots).
pub const BUILT: [&str; 8] = [
    "wall",
    "ring",
    "gables",
    "pyramid",
    "pile",
    "columns",
    "ring110k",
    "gables506k",
];

/// Built scene `name` as bodies: the ground slab (`ground`: the scene's extent), then its
/// pieces; `ccd`: continuous collision for its balls.
pub fn bodies(name: &str, levels: u32, ground: f64, ccd: bool) -> anyhow::Result<Vec<Body>> {
    let pieces = pieces(name, levels)?;
    let gh = ground_half_height(name);
    let gx = ground_half_extent(ground);
    let mut out = Vec::with_capacity(pieces.len() + 1);
    out.push(Body {
        dynamic: false,
        p: [0.0, 0.5 - gh, 0.0],
        q: [0.0, 0.0, 0.0, 1.0],
        v: [0.0; 3],
        w: [0.0; 3],
        mass: 0.0,
        inertia: [0.0; 3],
        shape: Shape::Box { half: [gx, gh, gx] },
        density: 0.0,
        bullet: false,
    });
    for piece in pieces {
        out.push(match piece {
            Piece::Brick {
                p,
                angle,
                mass,
                density,
            } => Body {
                dynamic: true,
                p,
                q: brick_quaternion(angle),
                v: [0.0; 3],
                w: [0.0; 3],
                mass,
                inertia: box_inertia(mass, BRICK_HALF),
                shape: Shape::Box { half: BRICK_HALF },
                density,
                bullet: false,
            },
            Piece::Box { p, half, q, mass } => Body {
                dynamic: true,
                p,
                q,
                v: [0.0; 3],
                w: [0.0; 3],
                mass,
                inertia: box_inertia(mass, half),
                shape: Shape::Box { half },
                density: 1.0,
                bullet: false,
            },
            Piece::Ball {
                p,
                radius,
                density,
                v,
                w,
            } => {
                let mass = sphere_mass(radius, density);
                // The page's sphere inertia: the inverse of its inverse.
                let i = 1.0 / (1.0 / (0.4 * mass * radius * radius));
                Body {
                    dynamic: true,
                    p,
                    q: [0.0, 0.0, 0.0, 1.0],
                    v,
                    w,
                    mass,
                    inertia: [i; 3],
                    shape: Shape::Sphere { radius },
                    density,
                    bullet: ccd,
                }
            }
        });
    }
    Ok(out)
}

/// Floats a body record takes (engine-world `play::rigid::create_bodies`): type (0 static,
/// 2 dynamic), bullet, position xyz, rotation xyzw, linear and angular velocity, mass,
/// principal inertia xyz, shape (0 sphere: radius; 2 box: half extents), its three numbers,
/// then the shape's density.
pub const BODY_RECORD: usize = 24;

/// The GPU upload order of `bodies`: scene order, or (`spatial`) the ground at slot 0 and
/// the rest sorted by their Morton code (X, Z, Y from the least significant bit: vertical
/// varies slowest). Returns `gpu_to_scene` (`gpu_to_scene[slot]` = scene body). Scene ids
/// stay in construction order for checks, impactor tracking and reset.
pub fn upload_order(bodies: &[Body], spatial: bool) -> anyhow::Result<Vec<u32>> {
    let n = bodies.len();
    if !spatial {
        return Ok((0..n as u32).collect());
    }
    anyhow::ensure!(
        n > 0 && !bodies[0].dynamic,
        "Morton upload requires static ground at scene body 0."
    );
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (id, b) in bodies.iter().enumerate() {
        for k in 0..3 {
            anyhow::ensure!(
                b.p[k].is_finite(),
                "Morton upload: body {id}, position axis {k} is not finite."
            );
            lo[k] = lo[k].min(b.p[k]);
            hi[k] = hi[k].max(b.p[k]);
        }
    }
    let codes: Vec<u32> = bodies
        .iter()
        .map(|b| {
            let q = |k: usize| {
                ((b.p[k] - lo[k]) / (hi[k] - lo[k]).max(1e-9) * 1024.0)
                    .floor()
                    .min(1023.0) as u32
            };
            spread(q(0)) | (spread(q(2)) << 1) | (spread(q(1)) << 2)
        })
        .collect();
    let mut rest: Vec<u32> = (1..n as u32).collect();
    rest.sort_by(|&a, &b| codes[a as usize].cmp(&codes[b as usize]).then(a.cmp(&b)));
    Ok(std::iter::once(0).chain(rest).collect())
}

/// `bodies` as [`BODY_RECORD`]-float records in `gpu_to_scene` order.
pub fn records(bodies: &[Body], gpu_to_scene: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bodies.len() * BODY_RECORD);
    for &s in gpu_to_scene {
        let b = &bodies[s as usize];
        let (shape, dims) = match b.shape {
            Shape::Sphere { radius } => (0.0, [radius, 0.0, 0.0]),
            Shape::Box { half } => (2.0, half),
        };
        let row: [f64; BODY_RECORD] = [
            if b.dynamic { 2.0 } else { 0.0 },
            if b.bullet { 1.0 } else { 0.0 },
            b.p[0],
            b.p[1],
            b.p[2],
            b.q[0],
            b.q[1],
            b.q[2],
            b.q[3],
            b.v[0],
            b.v[1],
            b.v[2],
            b.w[0],
            b.w[1],
            b.w[2],
            b.mass,
            b.inertia[0],
            b.inertia[1],
            b.inertia[2],
            shape,
            dims[0],
            dims[1],
            dims[2],
            b.density,
        ];
        out.extend(row.iter().map(|&x| x as f32));
    }
    out
}

/// A world's capacities for a scene of `bodies` bodies (`max_joints` joints), camelCase
/// for the page's `WorldDef`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capacity {
    pub max_bodies: u32,
    pub max_colliders: u32,
    pub max_contacts: u32,
    pub max_joints: u32,
    pub max_candidates: u32,
    pub max_colors: u32,
    /// Teleports, velocity sets and drives are World commands (the drag joint, the
    /// projectile).
    pub max_commands: u32,
}

/// The contact cap that bounds memory for small scenes (the benchmarks give their own).
const DEFAULT_CONTACT_CAP: u32 = 1 << 21;

impl Capacity {
    pub fn for_bodies(bodies: u32, max_joints: u32) -> Capacity {
        let pow2 = |n: u32| n.max(1).next_power_of_two();
        Capacity {
            max_bodies: pow2(bodies.max(4096)),
            max_colliders: pow2((bodies * 2).max(8192)),
            max_contacts: pow2((bodies * 16).max(32768)).min(DEFAULT_CONTACT_CAP),
            max_joints,
            max_candidates: pow2((bodies * 32).max(65536)).min(DEFAULT_CONTACT_CAP),
            max_colors: 32,
            max_commands: 2048,
        }
    }
}

/// A scene of the web demo: what it is, its camera, its solver constants and its world's
/// capacity (built scenes; a snapshot's follows its counts). camelCase, the page's names.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WebScene {
    pub title: &'static str,
    /// A snapshot exported by `scene_export` (`generated/scenes/<id>.bin`), else built here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<&'static str>,
    /// Steps of lead-in before a benchmark.
    pub lead: u32,
    /// Solver iterations (a snapshot's own when none).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u32>,
    pub settled: bool,
    pub sleep_rounds: u32,
    pub friction: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub friction_label: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground_y: Option<f64>,
    pub eye: [f64; 3],
    pub look: [f64; 3],
    pub far: f64,
    /// The scene's extent (the ground slab's size; 0: no ground drawn).
    pub ground: f64,
    /// The pyramid's levels by default (`?levels=` overrides).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flags: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta_lin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sleep_lin: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sleep_ang: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sleep_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap: Option<Capacity>,
}

const fn scene(title: &'static str, lead: u32, settled: bool, friction: f64) -> WebScene {
    WebScene {
        title,
        snapshot: None,
        lead,
        iterations: None,
        settled,
        sleep_rounds: 8,
        friction,
        friction_label: None,
        ground_y: None,
        eye: [0.0; 3],
        look: [0.0; 3],
        far: 400.0,
        ground: 0.0,
        levels: None,
        flags: None,
        alpha: None,
        beta_lin: None,
        sleep_lin: None,
        sleep_ang: None,
        sleep_time: None,
        cap: None,
    }
}

/// FLAG_REUSE_CONTACTS (three-avbd's default).
const REUSE: Option<u32> = Some(1);

/// The demo's scenes by name, in its menu order (capacities mirror three-avbd's
/// bench-cases tiers: small, paper, large, steady).
pub fn scenes() -> Vec<(&'static str, WebScene)> {
    let cap = |n| Some(Capacity::for_bodies(n, 64));
    let legacy = |title, snapshot, ground_y, eye, look, far, ground| WebScene {
        snapshot: Some(snapshot),
        friction_label: Some("per body (legacy)"),
        ground_y: Some(ground_y),
        eye,
        look,
        far,
        ground,
        ..scene(title, 60, false, 0.5)
    };
    let snapshot = |title, snapshot, friction, eye, look, far, ground| WebScene {
        snapshot: Some(snapshot),
        eye,
        look,
        far,
        ground,
        ..scene(title, 60, false, friction)
    };
    let mut gables506k_cap = Capacity::for_bodies(524288, 64);
    gables506k_cap.max_contacts = 6 * 1048576;
    gables506k_cap.max_candidates = 4 * 1048576;
    vec![
        (
            "wall",
            WebScene {
                iterations: Some(10),
                eye: [0.0, 16.0, -42.0],
                look: [0.0, 8.0, 0.0],
                ground: 48.0,
                cap: cap(4096),
                ..scene("wall smash 2k, 10 iterations", 30, false, 0.5)
            },
        ),
        (
            "ring",
            WebScene {
                iterations: Some(4),
                eye: [32.0, 22.0, 14.0],
                look: [0.0, 5.0, 0.0],
                ground: 22.0,
                flags: REUSE,
                sleep_time: Some(1e6),
                // contacts 262144: 131072 overflowed mid-smash.
                cap: cap(16384),
                ..scene("brick ring 9k, 4 iterations", 60, false, 0.5)
            },
        ),
        (
            "gables",
            WebScene {
                iterations: Some(3),
                eye: [0.0, 28.0, -40.0],
                look: [0.0, 6.0, 0.0],
                ground: 44.0,
                flags: REUSE,
                cap: cap(16384),
                ..scene("brick gables 6.7k, 3 iterations", 60, false, 0.5)
            },
        ),
        (
            "pyramid",
            WebScene {
                iterations: Some(10),
                eye: [28.0, 22.0, 28.0],
                look: [0.0, 6.0, 0.0],
                ground: 64.0,
                levels: Some(30),
                flags: REUSE,
                cap: cap(16384),
                ..scene("settled pyramid", 180, true, 0.5)
            },
        ),
        (
            "pile",
            WebScene {
                iterations: Some(4),
                sleep_rounds: 1,
                eye: [48.0, 36.0, 48.0],
                look: [0.0, 8.0, 0.0],
                far: 800.0,
                ground: 96.0,
                // The box pile's defaults (DECISIONS 8af–8av).
                flags: REUSE,
                alpha: Some(0.95),
                beta_lin: Some(2000.0),
                sleep_lin: Some(0.5),
                sleep_ang: Some(0.5),
                sleep_time: Some(0.25),
                cap: cap(65536),
                ..scene("settled pile 32k, 4 iterations", 180, true, 0.7)
            },
        ),
        (
            "columns",
            WebScene {
                iterations: Some(4),
                eye: [90.0, 40.0, 90.0],
                look: [0.0, 6.0, 0.0],
                far: 1200.0,
                ground: 200.0,
                flags: REUSE,
                cap: cap(131072),
                ..scene("box columns 100k, 4 iterations", 180, true, 0.5)
            },
        ),
        (
            "ring110k",
            WebScene {
                iterations: Some(4),
                eye: [140.0, 60.0, 40.0],
                look: [0.0, 12.0, 0.0],
                far: 2000.0,
                ground: 90.0,
                flags: REUSE,
                sleep_time: Some(1e6),
                cap: cap(131072),
                ..scene("brick ring 110k, 4 iterations", 60, false, 0.5)
            },
        ),
        (
            "gables506k",
            WebScene {
                iterations: Some(3),
                eye: [0.0, 80.0, -120.0],
                look: [0.0, 10.0, 0.0],
                far: 4000.0,
                ground: 256.0,
                flags: REUSE,
                sleep_time: Some(1e6),
                cap: Some(gables506k_cap),
                ..scene("brick gables 506k, 3 iterations", 60, false, 0.5)
            },
        ),
        // Joint scenes: built by this crate, exported as RUBSNAP1 records (scene_export).
        (
            "joints",
            snapshot(
                "joints showcase",
                "joints",
                0.5,
                [0.0, 12.0, -22.0],
                [0.0, 2.5, 1.0],
                400.0,
                60.0,
            ),
        ),
        (
            "ragdolls",
            snapshot(
                "ragdolls 40",
                "ragdolls",
                0.6,
                [6.0, 6.0, -11.0],
                [0.0, 1.5, 0.0],
                400.0,
                40.0,
            ),
        ),
        (
            "ragdolls10k",
            snapshot(
                "ragdoll pile 11k bodies, 10k joints",
                "ragdolls10k",
                0.6,
                [16.0, 14.0, -22.0],
                [0.0, 1.0, 0.0],
                600.0,
                60.0,
            ),
        ),
        (
            "hingechains10k",
            snapshot(
                "hinge chains 10k bodies, 10k joints",
                "hingechains10k",
                0.5,
                [40.0, 30.0, -90.0],
                [0.0, 8.0, 0.0],
                800.0,
                120.0,
            ),
        ),
        (
            "motorfield10k",
            snapshot(
                "motor field 10k bodies, 2.5k motors",
                "motorfield10k",
                0.5,
                [30.0, 30.0, -70.0],
                [0.0, 2.0, 0.0],
                800.0,
                80.0,
            ),
        ),
        // The legacy web demo's scenes (`legacy`): the same bodies, masses, velocities and
        // per-body friction as crates/rubble-scenes. Each brings its own ground; static
        // friction draws no floor (legacy placed its boxes on the ramp below the floor slab).
        (
            "legacyEmpty",
            legacy(
                "legacy · empty",
                "legacy-empty",
                0.5,
                [0.0, 6.0, -20.0],
                [0.0; 3],
                600.0,
                50.0,
            ),
        ),
        (
            "legacyGround",
            legacy(
                "legacy · ground",
                "legacy-ground",
                0.5,
                [0.0, 6.0, -16.0],
                [0.0, 1.0, 0.0],
                600.0,
                50.0,
            ),
        ),
        (
            "legacyDynamicFriction",
            legacy(
                "legacy · dynamic friction (μ 5 → 0)",
                "legacy-dynamic-friction",
                0.5,
                [10.0, 18.0, -58.0],
                [10.0, 0.0, -20.0],
                800.0,
                120.0,
            ),
        ),
        (
            "legacyStaticFriction",
            legacy(
                "legacy · static friction ramp (μ 0.25 → 0.5)",
                "legacy-static-friction",
                0.5,
                [-34.0, 8.0, -34.0],
                [-8.0, -3.0, 0.0],
                600.0,
                0.0,
            ),
        ),
        (
            "legacyPyramid",
            legacy(
                "legacy · pyramid 136",
                "legacy-pyramid",
                0.0,
                [0.0, 8.0, -26.0],
                [0.0, 6.0, 0.0],
                600.0,
                50.0,
            ),
        ),
        (
            "legacyStack",
            legacy(
                "legacy · stack 10",
                "legacy-stack",
                0.5,
                [0.0, 8.0, -22.0],
                [0.0, 7.0, 0.0],
                600.0,
                50.0,
            ),
        ),
        (
            "legacyStackRatio",
            legacy(
                "legacy · stack ratio 1:8",
                "legacy-stack-ratio",
                0.5,
                [0.0, 12.0, -36.0],
                [0.0, 8.0, 0.0],
                600.0,
                50.0,
            ),
        ),
        (
            "legacyScatter",
            legacy(
                "legacy · scatter 3k (spheres, boxes, capsules)",
                "legacy-scatter",
                0.0,
                [24.0, 18.0, 32.0],
                [0.0, 3.0, 0.0],
                1200.0,
                300.0,
            ),
        ),
        (
            "legacyScatterBoxes",
            legacy(
                "legacy · scatter boxes 4k",
                "legacy-scatter-boxes",
                0.0,
                [26.0, 20.0, 34.0],
                [0.0, 3.0, 0.0],
                1200.0,
                300.0,
            ),
        ),
        (
            "legacyGridBoxes",
            legacy(
                "legacy · grid boxes 1.5k",
                "legacy-grid-boxes",
                0.0,
                [10.0, 10.0, 14.0],
                [0.0, 5.0, 0.0],
                1200.0,
                300.0,
            ),
        ),
        (
            "legacySlantedGrid",
            legacy(
                "legacy · slanted grid 3.2k",
                "legacy-slanted-grid",
                0.0,
                [14.0, 10.0, 18.0],
                [0.0, 5.0, 0.0],
                1200.0,
                300.0,
            ),
        ),
        (
            "legacyGrid10k",
            legacy(
                "legacy · grid 10k",
                "legacy-grid10k",
                0.0,
                [22.0, 16.0, 28.0],
                [0.0, 5.0, 0.0],
                1200.0,
                300.0,
            ),
        ),
        // Not in the legacy demo: its 10k grid scaled to 100k (50 × 40 × 50).
        (
            "legacyGrid100k",
            legacy(
                "grid 100k (legacy grid ×10)",
                "legacy-grid100k",
                0.0,
                [46.0, 30.0, 58.0],
                [0.0, 9.0, 0.0],
                2000.0,
                300.0,
            ),
        ),
    ]
}

/// The scene table as the page's JSON (`generated/scenes/demo.json`): an array of
/// `[name, scene]` pairs in menu order.
pub fn scenes_json() -> serde_json::Value {
    serde_json::Value::Array(
        scenes()
            .into_iter()
            .map(|(name, s)| serde_json::json!([name, s]))
            .collect(),
    )
}
