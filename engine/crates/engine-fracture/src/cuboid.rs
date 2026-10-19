//! A detailed house built only from cuboids (SHAPE_BOX colliders): running-bond
//! brick walls with interlocking corners, door and window openings with
//! lintels, sills and glass, stepped brick gables, a ridge beam, rafters, roof
//! boards and a ridge cap, an exterior chimney, a porch on posts with a step,
//! and a picket fence with a mailbox. Every piece is its own dynamic body and
//! starts asleep, touching but never overlapping its neighbours.
//!
//! Box pairs take the engine's box-box path instead of the convex-hull SAT.
//! Rafters and boards are loose and held only by friction (mu = 0.5). Woken
//! whole, the house stands and sleeps again at a 12 deg pitch; at 18 deg the
//! settling jolt starts the rafters sliding and the roof falls in. Contacts use
//! zero-gap mode (FLAG_CONTACT_GAP): with the default margin every course rests
//! 2 mm deep, an 11-course wall sinks ~2 cm unevenly against the taller
//! gables, and the roof shifts.

use crate::bonds::{scale_bond_springs, BOND_TICK};
use crate::math::*;
use crate::scene::*;
use anyhow::Result;
use engine_schema::convex::HullArena;

const BRICK_L: f64 = 0.5;
const BRICK_H: f64 = 0.25;
const WALL_T: f64 = 0.25;
const HALF_X: f64 = 4.0;
const HALF_Z: f64 = 3.0;
const COURSES: u32 = 11;
const EAVE: f64 = COURSES as f64 * BRICK_H;
/// Side gap between neighbouring pieces.
const GAP: f64 = 0.001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Material {
    Brick,
    Concrete,
    Wood,
    Glass,
}

impl Material {
    pub fn density(self) -> f64 {
        match self {
            Material::Brick => 1900.0,
            Material::Concrete => 2300.0,
            Material::Wood => 550.0,
            Material::Glass => 2500.0,
        }
    }

    /// Tensile strength (Pa) of a joint of this material (a bond takes its weaker
    /// side's): mortar for brick, a cold joint for concrete, nailing for timber, glazing
    /// for glass. Representative values, not a calibration.
    pub fn joint_strength(self) -> f64 {
        match self {
            Material::Brick => 0.3e6,
            Material::Concrete => 1.0e6,
            Material::Wood => 0.8e6,
            Material::Glass => 0.3e6,
        }
    }

    /// Young's modulus (Pa).
    pub fn youngs_modulus(self) -> f64 {
        match self {
            Material::Brick => 5e9,
            Material::Concrete => 25e9,
            Material::Wood => 10e9,
            Material::Glass => 50e9,
        }
    }
}

/// One cuboid: centre, half extents, rotation about +x (radians).
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub center: [f64; 3],
    pub half: [f64; 3],
    pub material: Material,
    pub color: [f64; 3],
    pub angle: f64,
}

impl Piece {
    /// World corners (rotation about +x).
    pub fn corners(&self) -> [[f64; 3]; 8] {
        let [hx, hy, hz] = self.half;
        let (c, s) = (self.angle.cos(), self.angle.sin());
        let mut out = [[0.0; 3]; 8];
        let mut i = 0;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    let (y, z) = (sy * hy, sz * hz);
                    out[i] = [
                        self.center[0] + sx * hx,
                        self.center[1] + c * y - s * z,
                        self.center[2] + s * y + c * z,
                    ];
                    i += 1;
                }
            }
        }
        out
    }
}

/// Deterministic colour jitter (the same house every load).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 as f64 / 4294967296.0
    }
    fn jitter(&mut self, c: [f64; 3], a: f64) -> [f64; 3] {
        c.map(|x| (x * (1.0 + (self.next() - 0.5) * a)).clamp(0.0, 1.0))
    }
}

pub const PARTS: [&str; 7] = [
    "walls", "openings", "gables", "roof", "chimney", "porch", "fence",
];

#[derive(Clone, Debug)]
pub struct CuboidOptions {
    pub seed: u32,
    pub pitch_deg: f64,
    /// Each box as an 8-vertex convex hull instead of SHAPE_BOX (same shapes
    /// through the convex path, to compare the two).
    pub hulls: bool,
    /// Limits the build (diagnostics): any of `PARTS`.
    pub parts: Vec<&'static str>,
}

impl Default for CuboidOptions {
    fn default() -> Self {
        Self {
            seed: 7,
            pitch_deg: 12.0,
            hulls: false,
            parts: PARTS.to_vec(),
        }
    }
}

#[derive(Clone, Copy)]
struct Opening {
    u0: f64,
    u1: f64,
    c0: u32,
    c1: u32,
    door: bool,
}

#[derive(Clone)]
struct Wall {
    along_x: bool,
    at: f64,
    openings: Vec<Opening>,
}

/// The cuboid house's pieces.
pub fn cuboid_pieces(o: &CuboidOptions) -> Vec<Piece> {
    let pitch = o.pitch_deg.to_radians();
    let (tan, cos, sin) = (pitch.tan(), pitch.cos(), pitch.sin());
    let has = |p: &str| o.parts.contains(&p);
    let mut rand = Rng(o.seed);
    let mut pieces = Vec::new();
    let mut boxed =
        |center: [f64; 3], half: [f64; 3], material: Material, color: [f64; 3], angle: f64| {
            pieces.push(Piece {
                center,
                half,
                material,
                color,
                angle,
            })
        };
    let concrete = [0.62, 0.62, 0.6];
    let trim = [0.9, 0.88, 0.82];
    let opening = |u0, u1, c0, c1, door| Opening {
        u0,
        u1,
        c0,
        c1,
        door,
    };

    // ---- walls ----------------------------------------------------------------
    // A wall runs along u (x for front/back, z for the sides) at a fixed
    // coordinate. Even courses: front/back span the full width and the sides
    // fit between them; odd courses swap, so the corners interlock.
    let walls = [
        Wall {
            along_x: true,
            at: HALF_Z - WALL_T / 2.0,
            openings: vec![
                opening(-0.5, 0.5, 0, 8, true),
                opening(-3.0, -1.8, 3, 8, false),
                opening(1.8, 3.0, 3, 8, false),
            ],
        },
        Wall {
            along_x: true,
            at: -HALF_Z + WALL_T / 2.0,
            openings: vec![
                opening(-2.6, -1.4, 3, 8, false),
                opening(1.4, 2.6, 3, 8, false),
            ],
        },
        Wall {
            along_x: false,
            at: HALF_X - WALL_T / 2.0,
            openings: vec![opening(-0.6, 0.6, 3, 8, false)],
        },
        Wall {
            along_x: false,
            at: -HALF_X + WALL_T / 2.0,
            openings: vec![],
        },
    ];
    let place = |boxed: &mut dyn FnMut([f64; 3], [f64; 3], Material, [f64; 3], f64),
                 wall: &Wall,
                 u: f64,
                 y: f64,
                 du: f64,
                 dy: f64,
                 depth: f64,
                 material: Material,
                 color: [f64; 3]| {
        if wall.along_x {
            boxed(
                [u, y, wall.at],
                [du / 2.0, dy / 2.0, depth / 2.0],
                material,
                color,
                0.0,
            )
        } else {
            boxed(
                [wall.at, y, u],
                [depth / 2.0, dy / 2.0, du / 2.0],
                material,
                color,
                0.0,
            )
        }
    };
    // Fill [a, b] of one course with bricks; odd courses start with a half brick.
    let fill = |boxed: &mut dyn FnMut([f64; 3], [f64; 3], Material, [f64; 3], f64),
                rand: &mut Rng,
                wall: &Wall,
                a: f64,
                b: f64,
                course: u32| {
        let len = b - a;
        if len < 0.1 {
            return;
        }
        let n = (len / BRICK_L).round().max(1.0) as u32;
        let bl = len / n as f64;
        let mut cuts = vec![a];
        if course % 2 == 1 && n > 1 {
            for k in 0..n {
                cuts.push(a + bl * (k as f64 + 0.5));
            }
        } else {
            for k in 1..n {
                cuts.push(a + bl * k as f64);
            }
        }
        cuts.push(b);
        for k in 0..cuts.len() - 1 {
            let color = rand.jitter([0.62, 0.25, 0.18], 0.25);
            place(
                boxed,
                wall,
                (cuts[k] + cuts[k + 1]) / 2.0,
                (course as f64 + 0.5) * BRICK_H,
                cuts[k + 1] - cuts[k] - GAP,
                BRICK_H,
                WALL_T,
                Material::Brick,
                color,
            );
        }
    };
    let minus = |a: f64, b: f64, holes: &[(f64, f64)]| {
        let mut runs = vec![(a, b)];
        for &(h0, h1) in holes {
            runs = runs
                .into_iter()
                .flat_map(|(r0, r1)| {
                    if h1 <= r0 || h0 >= r1 {
                        vec![(r0, r1)]
                    } else {
                        [(r0, r0.max(h0)), (r1.min(h1), r1)]
                            .into_iter()
                            .filter(|(x, y)| y - x > 1e-9)
                            .collect()
                    }
                })
                .collect();
        }
        runs
    };
    if has("walls") {
        for c in 0..COURSES {
            for wall in &walls {
                let full = wall.along_x == (c % 2 == 0);
                let half = if wall.along_x { HALF_X } else { HALF_Z };
                let extent = if full { half } else { half - WALL_T };
                let mut holes = Vec::new();
                for o in &wall.openings {
                    if c >= o.c0 && c < o.c1 {
                        holes.push((o.u0, o.u1));
                    } else if c == o.c1 {
                        holes.push((o.u0 - WALL_T, o.u1 + WALL_T)); // lintel course
                    }
                }
                for (a, b) in minus(-extent, extent, &holes) {
                    fill(&mut boxed, &mut rand, wall, a, b, c);
                }
            }
        }
    }
    if has("openings") {
        for wall in &walls {
            for o in &wall.openings {
                // Lintel over the opening, bearing a brick depth on each side.
                let mid = (o.u0 + o.u1) / 2.0;
                place(
                    &mut boxed,
                    wall,
                    mid,
                    (o.c1 as f64 + 0.5) * BRICK_H,
                    o.u1 - o.u0 + 2.0 * WALL_T - GAP,
                    BRICK_H,
                    WALL_T,
                    Material::Concrete,
                    concrete,
                );
                let (top, bottom, w) = (o.c1 as f64 * BRICK_H, o.c0 as f64 * BRICK_H, o.u1 - o.u0);
                if o.door {
                    place(
                        &mut boxed,
                        wall,
                        mid,
                        (top - 0.01) / 2.0,
                        w - 0.02,
                        top - 0.01,
                        0.05,
                        Material::Wood,
                        [0.2, 0.38, 0.42],
                    );
                } else {
                    let sill = 0.06;
                    place(
                        &mut boxed,
                        wall,
                        mid,
                        bottom + sill / 2.0,
                        w - 0.01,
                        sill,
                        WALL_T,
                        Material::Wood,
                        trim,
                    );
                    place(
                        &mut boxed,
                        wall,
                        mid,
                        (bottom + sill + top - 0.01) / 2.0,
                        w - 0.02,
                        top - 0.01 - bottom - sill,
                        0.03,
                        Material::Glass,
                        [0.62, 0.8, 0.9],
                    );
                }
            }
        }
    }

    // ---- gables and roof --------------------------------------------------------
    // P(z): the plane the rafters sit on, through the outer wall top edges.
    let plane = |z: f64| EAVE + (HALF_Z - z.abs()) * tan + GAP;
    let mut gable_top = EAVE;
    let mut gable_courses = Vec::new();
    let mut c = COURSES;
    loop {
        let y1 = (c + 1) as f64 * BRICK_H;
        let span = HALF_Z - (y1 - EAVE) / tan - 0.002;
        if span < 0.3 {
            break;
        }
        gable_courses.push((c, span));
        gable_top = y1;
        c += 1;
    }
    // Ridge beam: rests on the top gable course, its top corners just under P.
    let mut ridge_h = plane(0.1) - 0.002 - gable_top;
    while ridge_h < 0.06 && !gable_courses.is_empty() {
        gable_courses.pop();
        gable_top = EAVE + gable_courses.len() as f64 * BRICK_H;
        ridge_h = plane(0.1) - 0.002 - gable_top;
    }
    if has("gables") {
        for x in [HALF_X - WALL_T / 2.0, -HALF_X + WALL_T / 2.0] {
            for &(c, span) in &gable_courses {
                fill(
                    &mut boxed,
                    &mut rand,
                    &Wall {
                        along_x: false,
                        at: x,
                        openings: vec![],
                    },
                    -span,
                    span,
                    c,
                );
            }
        }
    }
    if has("roof") {
        boxed(
            [0.0, gable_top + ridge_h / 2.0, 0.0],
            [HALF_X - 0.05, ridge_h / 2.0, 0.1],
            Material::Wood,
            [0.5, 0.36, 0.22],
            0.0,
        );
    }
    // Rafters and roof boards along each slope, from just off the ridge to the eave overhang.
    let (z_lo, z_hi, rafter_t, board_t, strip) = (0.02, HALF_Z + 0.35, 0.12, 0.03, 0.62);
    if has("roof") {
        for side in [1.0, -1.0] {
            let angle = side * pitch;
            // Centre of a piece whose bottom face lies on P + lift between z0 and z1.
            let on_slope = |z0: f64, z1: f64, lift: f64, thick: f64| {
                let zm = (z0 + z1) / 2.0;
                let y = plane(zm) + lift;
                (side * zm + side * sin * thick / 2.0, y + cos * thick / 2.0)
            };
            for x in [-3.6, -1.8, 0.0, 1.8, 3.6] {
                let (z, y) = on_slope(z_lo, z_hi, 0.0, rafter_t);
                boxed(
                    [x, y, z],
                    [0.04, rafter_t / 2.0, (z_hi - z_lo) / cos / 2.0],
                    Material::Wood,
                    [0.55, 0.4, 0.25],
                    angle,
                );
            }
            let lift = rafter_t / cos + GAP;
            let mut z0 = z_lo;
            while z0 < z_hi - 1e-6 {
                let z1 = z_hi.min(z0 + strip * cos);
                let (z, y) = on_slope(z0, z1, lift, board_t);
                let shade = rand.jitter([0.33, 0.29, 0.28], 0.15);
                for (x0, x1) in [(-HALF_X, -GAP), (GAP, HALF_X)] {
                    boxed(
                        [(x0 + x1) / 2.0, y, z],
                        [(x1 - x0) / 2.0, board_t / 2.0, (z1 - z0) / cos / 2.0 - GAP],
                        Material::Wood,
                        shade,
                        angle,
                    );
                }
                z0 += strip * cos;
            }
        }
        let cap_y = plane(z_lo) + rafter_t / cos + GAP + board_t / cos + 0.002;
        boxed(
            [0.0, cap_y + 0.02, 0.0],
            [HALF_X, 0.02, 0.18],
            Material::Wood,
            [0.25, 0.22, 0.22],
            0.0,
        );
    }

    // ---- chimney (outside the left gable) ----------------------------------------
    let chimney_x = -HALF_X - 0.26;
    let chimney_top = plane(0.0) + 0.6;
    if has("chimney") {
        let mut c = 0;
        while (c + 1) as f64 * BRICK_H <= chimney_top {
            let y = (c as f64 + 0.5) * BRICK_H;
            let col = rand.jitter([0.5, 0.2, 0.15], 0.2);
            if c % 2 == 0 {
                for z in [-0.125, 0.125] {
                    boxed(
                        [chimney_x, y, z],
                        [0.25 - GAP, BRICK_H / 2.0, 0.125 - GAP / 2.0],
                        Material::Brick,
                        col,
                        0.0,
                    );
                }
            } else {
                for dx in [-0.125, 0.125] {
                    boxed(
                        [chimney_x + dx, y, 0.0],
                        [0.125 - GAP / 2.0, BRICK_H / 2.0, 0.25 - GAP],
                        Material::Brick,
                        col,
                        0.0,
                    );
                }
            }
            c += 1;
        }
    }

    // ---- porch and step ---------------------------------------------------------------
    let post_h = 2.3;
    if has("porch") {
        for x in [-1.1, 1.1] {
            for z in [HALF_Z + 0.2, HALF_Z + 1.4] {
                boxed(
                    [x, post_h / 2.0, z],
                    [0.075, post_h / 2.0, 0.075],
                    Material::Wood,
                    trim,
                    0.0,
                );
            }
        }
        boxed(
            [0.0, post_h + GAP + 0.04, HALF_Z + 0.825],
            [1.3, 0.04, 0.775],
            Material::Wood,
            [0.4, 0.4, 0.42],
            0.0,
        );
        boxed(
            [0.0, 0.075, HALF_Z + 0.3 + GAP],
            [0.7, 0.075, 0.3],
            Material::Concrete,
            concrete,
            0.0,
        );
    }

    // ---- picket fence, gate gap and mailbox --------------------------------------
    let fence_z = HALF_Z + 3.2;
    let posts = [-5.0, -2.5, -0.75, 0.75, 2.5, 5.0];
    if has("fence") {
        for x in posts {
            boxed(
                [x, 0.475, fence_z],
                [0.05, 0.475, 0.05],
                Material::Wood,
                trim,
                0.0,
            );
        }
        for k in 0..posts.len() - 1 {
            let (a, b) = (posts[k], posts[k + 1]);
            if a == -0.75 {
                continue; // the gate
            }
            boxed(
                [(a + b) / 2.0, 0.95 + GAP + 0.02, fence_z],
                [(b - a) / 2.0 - GAP, 0.02, 0.03],
                Material::Wood,
                trim,
                0.0,
            );
            let mut x = a + 0.15;
            while x < b - 0.12 {
                boxed(
                    [x, 0.4, fence_z],
                    [0.04, 0.4, 0.0125],
                    Material::Wood,
                    trim,
                    0.0,
                );
                x += 0.18;
            }
        }
        boxed(
            [1.3, 0.5, fence_z + 0.45],
            [0.04, 0.5, 0.04],
            Material::Wood,
            [0.45, 0.32, 0.2],
            0.0,
        );
        boxed(
            [1.3, 1.0 + GAP + 0.11, fence_z + 0.45],
            [0.1, 0.11, 0.225],
            Material::Wood,
            [0.18, 0.2, 0.24],
            0.0,
        );
    }
    pieces
}

/// The procedural cuboid house scene.
pub fn build_cuboid_house(o: &CuboidOptions) -> Result<Scene> {
    scene_from_pieces(&cuboid_pieces(o), o.hulls)
}

pub(crate) const FACES: [([f64; 3], [[f64; 3]; 4]); 6] = [
    (
        [1.0, 0.0, 0.0],
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
    ),
    (
        [-1.0, 0.0, 0.0],
        [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
    ),
    (
        [0.0, 1.0, 0.0],
        [
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
    ),
    (
        [0.0, -1.0, 0.0],
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, 1.0],
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, -1.0],
        [
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
        ],
    ),
];
/// Faces of an 8-corner box (corner i: bit 0 x, bit 1 y, bit 2 z).
pub const BOX_FACES: [[u32; 4]; 6] = [
    [0, 2, 6, 4],
    [1, 5, 7, 3],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 1, 3, 2],
    [4, 6, 7, 5],
];

/// Box corners for half extents (corner i: bit 0 x, bit 1 y, bit 2 z).
pub fn box_corners(half: [f64; 3]) -> Vec<[f64; 3]> {
    (0..8)
        .map(|i| {
            [
                if i & 1 != 0 { half[0] } else { -half[0] },
                if i & 2 != 0 { half[1] } else { -half[1] },
                if i & 4 != 0 { half[2] } else { -half[2] },
            ]
        })
        .collect()
}

/// Bodies, box colliders and the render mesh, recentred so the footprint's
/// bounding box is centred on x = z = 0 (layouts assume a centred house).
pub fn scene_from_pieces(pieces: &[Piece], hulls: bool) -> Result<Scene> {
    let mut arena = HullArena::new();
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in pieces {
        for c in p.corners() {
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
    }
    let shift = [-(lo[0] + hi[0]) / 2.0, 0.0, -(lo[2] + hi[2]) / 2.0];
    let mut bodies = vec![Body::fixed([0.0; 3])];
    let mut colliders = vec![Collider::ground()];
    let mut vertices = Vec::new();
    let sizes = [4u32; 6];
    let indices: Vec<u32> = BOX_FACES.iter().flatten().copied().collect();
    for p in pieces {
        let [hx, hy, hz] = p.half;
        let rho = p.material.density();
        let m = rho * 8.0 * hx * hy * hz;
        let id = bodies.len() as u32;
        let q = [(p.angle / 2.0).sin(), 0.0, 0.0, (p.angle / 2.0).cos()];
        bodies.push(Body {
            p: add(p.center, shift),
            q,
            v: [0.0; 3],
            w: [0.0; 3],
            inv_mass: 1.0 / m,
            inv_i: [
                3.0 / (m * (hy * hy + hz * hz)),
                3.0 / (m * (hx * hx + hz * hz)),
                3.0 / (m * (hx * hx + hy * hy)),
            ],
            flags: FLAG_DYNAMIC | FLAG_SLEEPING,
        });
        if hulls {
            let (base, radius, local_pos) =
                append_hull(&mut arena, &box_corners(p.half), &sizes, &indices, 0.0)?;
            colliders.push(Collider::hull(id, rho, base, 1.0, radius, local_pos));
        } else {
            colliders.push(Collider::new(id, SHAPE_BOX, rho, [hx, hy, hz, 0.0]));
        }
        for (n, quad) in FACES {
            for k in [0, 1, 2, 0, 2, 3] {
                let s = quad[k];
                vertex(
                    &mut vertices,
                    [s[0] * hx, s[1] * hy, s[2] * hz],
                    n,
                    p.color,
                    id,
                );
            }
        }
    }
    let span = sub(hi, lo);
    let (joints, lines) = cuboid_bonds(pieces, shift, &bodies);
    let mut s = Scene::from_parts(
        bodies,
        colliders,
        &arena,
        SceneParams::lab(LAB_FLAGS | FLAG_CONTACT_GAP),
    );
    s.bond_count = joints.len() as u32;
    s.joints = joints;
    s.lines = lines;
    add_projectiles(&mut s, 0.4, 2.0);
    s.projectile_mass = 500.0;
    s.vertices = vertices;
    s.pieces = pieces.len() as u32;
    s.selected = (1..=pieces.len() as u32).collect();
    s.extent = span[0].max(span[1]).max(span[2]);
    s.extent_x = span[0];
    s.extent_z = span[2];
    s.height = span[1];
    s.center = [0.0, span[1] / 2.0, 0.0];
    s.grid_cell_size = grid_config(&s.colliders).0;
    Ok(s)
}


/// Contacts closer than this (m) count as touching (pieces sit `GAP` apart or flush).
const TOUCH: f64 = 0.003;
/// The smallest contact area (m^2) that makes a bond.
const MIN_BOND_AREA: f64 = 1e-4;

/// The bond graph of a cuboid house, kept simple: a weld wherever two axis-aligned pieces
/// touch face to face (the contact rectangle from their overlap), and an anchor wherever
/// a piece rests on the ground. Capacities and springs follow the GLB bonds' formulas
/// (`bonds.rs`) from the pieces' materials (`Material::joint_strength`,
/// `youngs_modulus`; the weaker side). Rotated pieces (rafters, roof boards) stay loose,
/// held by friction as before. Body i + 1 is piece i (body 0 the ground).
pub fn cuboid_bonds(pieces: &[Piece], shift: [f64; 3], bodies: &[Body]) -> (Vec<Joint>, Vec<f32>) {
    let mut joints = Vec::new();
    let mut lines = Vec::new();
    // A rectangle contact w0 x w1 with normal along `axis`: capacities and springs.
    let bond = |ma: Material, mb: Option<Material>, da: f64, db: f64, w: [f64; 2]| {
        let ft = mb.map_or(ma.joint_strength(), |b| ma.joint_strength().min(b.joint_strength()));
        let fc = 15.0 * ft;
        let area = w[0] * w[1];
        let (iuu, ivv) = (w[0] * w[1].powi(3) / 12.0, w[1] * w[0].powi(3) / 12.0);
        let c = 0.5 * w[0].max(w[1]);
        let ea = ma.youngs_modulus();
        let ce = da / ea + mb.map_or(0.0, |b| db / b.youngs_modulus());
        let cg = da / (ea / 2.4) + mb.map_or(0.0, |b| db / (b.youngs_modulus() / 2.4));
        let strength = [ft * area, fc * area, 1.5 * ft * area, ft * iuu.min(ivv) / c];
        let k = [area / ce, area / cg, area / cg, iuu / ce, ivv / ce, (iuu + ivv) / cg];
        (strength, k)
    };
    let axis_u = |axis: usize| -> [f64; 3] {
        let mut u = [0.0; 3];
        u[(axis + 1) % 3] = 1.0;
        u
    };
    // A bond's line between its bodies' centres; an anchor's from the piece to its anchor
    // point in the piece's frame (`local`; copies share body 0, so a ground point would sit
    // at the first copy).
    let line = |lines: &mut Vec<f32>, a: u32, b: u32, local: [f64; 3], anchor: bool| {
        let color = if anchor { [0.95, 0.6, 0.2] } else { [0.25, 0.9, 0.8] };
        vertex(lines, [0.0; 3], [0.0, 1.0, 0.0], color, a);
        if b == 0 {
            vertex(lines, local, [0.0, 1.0, 0.0], color, a);
        } else {
            vertex(lines, [0.0; 3], [0.0, 1.0, 0.0], color, b);
        }
    };
    for (i, a) in pieces.iter().enumerate() {
        if a.angle != 0.0 {
            continue;
        }
        let ia = i as u32 + 1;
        // Resting on the ground: an anchor over the footprint.
        if a.center[1] - a.half[1] <= TOUCH {
            let w = [2.0 * a.half[2], 2.0 * a.half[0]];
            let (strength, k) = bond(a.material, None, a.half[1], 0.0, w);
            let at = add([a.center[0], 0.0, a.center[2]], shift);
            joints.push(Joint {
                body_a: ia,
                body_b: 0,
                anchor_a: sub(at, bodies[ia as usize].p),
                anchor_b: at,
                rest: IDENTITY,
                normal: [0.0, -1.0, 0.0],
                frame_u: axis_u(1),
                strength,
                k,
                world: true,
            });
            line(&mut lines, ia, 0, sub(at, bodies[ia as usize].p), true);
        }
        for (j, b) in pieces.iter().enumerate().skip(i + 1) {
            if b.angle != 0.0 {
                continue;
            }
            // Separation per axis (negative: overlap).
            let gap: [f64; 3] = std::array::from_fn(|k| (a.center[k] - b.center[k]).abs() - a.half[k] - b.half[k]);
            let touching: Vec<usize> = (0..3).filter(|&k| gap[k].abs() <= TOUCH).collect();
            let [axis] = touching[..] else { continue };
            let others = [(axis + 1) % 3, (axis + 2) % 3];
            if others.iter().any(|&k| gap[k] > -TOUCH) {
                continue;
            }
            let w = others.map(|k| (-gap[k]).min(2.0 * a.half[k]).min(2.0 * b.half[k]));
            if w[0] * w[1] < MIN_BOND_AREA {
                continue;
            }
            let sign = if b.center[axis] > a.center[axis] { 1.0 } else { -1.0 };
            let mut c = [0.0; 3];
            c[axis] = a.center[axis] + sign * a.half[axis];
            for k in others {
                let lo = (a.center[k] - a.half[k]).max(b.center[k] - b.half[k]);
                let hi = (a.center[k] + a.half[k]).min(b.center[k] + b.half[k]);
                c[k] = 0.5 * (lo + hi);
            }
            let at = add(c, shift);
            let mut normal = [0.0; 3];
            normal[axis] = sign;
            let ib = j as u32 + 1;
            let (strength, k) = bond(a.material, Some(b.material), a.half[axis], b.half[axis], w);
            joints.push(Joint {
                body_a: ia,
                body_b: ib,
                anchor_a: sub(at, bodies[ia as usize].p),
                anchor_b: sub(at, bodies[ib as usize].p),
                rest: IDENTITY,
                normal,
                frame_u: axis_u(axis),
                strength,
                k,
                world: false,
            });
            line(&mut lines, ia, ib, [0.0; 3], false);
        }
    }
    (scale_bond_springs(&joints, bodies, BOND_TICK), lines)
}
