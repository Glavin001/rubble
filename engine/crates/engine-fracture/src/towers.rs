//! A city block of fractured towers under a scripted volley of fast, off-angle
//! balls: the CCD showcase, shared by the native viewer (`fracture_viewer
//! towers --volley`, `towers_capture`, `tests/towers_volley.rs`) and the
//! browser lab ("Fractured towers" scene), so both fire exactly the same shots.
//!
//! The volley first rakes the towers at a glancing angle (balls clip a corner,
//! smash through a few rooms, deflect and carry on into the next tower), then
//! hammers the towers' lower floors from outside the block so they lose their
//! base and fall into their neighbours.

use crate::scene::{build_grid, GridOptions, Scene};
use serde::Serialize;

/// The showcase's defaults: 2 × 2 towers, 6 m streets, a 1 m 30 t ball at 16×
/// the shot speed, bonds at 0.2× the pack's strength, 8 raking and 16 base
/// shots 45 ticks apart.
pub const GAP: f64 = 6.0;
pub const BALL: (f64, f64) = (1.0, 30000.0);
pub const SPEED_SCALE: f64 = 16.0;
pub const BOND_STRENGTH: f64 = 0.2;
pub const VOLLEY: (u32, u32, u32) = (8, 16, 45);

/// `columns × rows` copies of `one` (a pack scene: one tower), `gap` metres of
/// street between them, firing the pack's own ball unless `ball` = (radius m,
/// mass kg) (the lab's grid layout swaps in a 200 t, 1.6 m wrecking ball). The
/// joints `one` carries are kept (`Scene::without_joints` first for loose
/// pieces). The camera frames the block.
pub fn towers_grid(one: &Scene, columns: u32, rows: u32, gap: f64, ball: Option<(f64, f64)>) -> Scene {
    let o = GridOptions {
        columns,
        rows,
        gap,
        projectile_radius: ball.map_or(one.projectile_radius, |b| b.0),
        projectile_mass: ball.map_or(one.projectile_mass, |b| b.1),
        projectile_speed: one.projectile_speed.unwrap_or(40.0),
        ..GridOptions::default()
    };
    let mut s = build_grid(one, &o);
    // The pack's camera frames one tower.
    s.camera = None;
    s
}

#[derive(Clone, Debug, Serialize)]
pub struct Shot {
    /// Physics step at which it is fired.
    pub step: u32,
    pub origin: [f64; 3],
    pub dir: [f64; 3],
    pub label: String,
}

/// The towers' centres and the block's bounds (piece centres).
pub struct Block {
    pub towers: Vec<[f64; 3]>,
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    /// Largest half width of a tower.
    pub half: f64,
}

pub fn block(scene: &Scene) -> Block {
    let per = scene.instance_stride.max(1) as usize;
    let start = scene.projectile_start as usize;
    let copies = (start - 1) / per;
    let (mut lo, mut hi, mut half) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3], 0.0f64);
    let mut towers = Vec::new();
    for t in 0..copies {
        let (mut a, mut b) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for body in &scene.bodies[1 + t * per..1 + (t + 1) * per] {
            for k in 0..3 {
                a[k] = a[k].min(body.p[k]);
                b[k] = b[k].max(body.p[k]);
            }
        }
        towers.push([0, 1, 2].map(|k| 0.5 * (a[k] + b[k])));
        for k in 0..3 {
            lo[k] = lo[k].min(a[k]);
            hi[k] = hi[k].max(b[k]);
        }
        half = half.max(0.5 * (b[0] - a[0]).max(b[2] - a[2]));
    }
    Block { towers, lo, hi, half }
}

fn norm(d: [f64; 3]) -> [f64; 3] {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    d.map(|x| x / l)
}

/// The volley, `every` steps apart: `raking` glancing shots, then `base` shots
/// at the towers' lower floors. Deterministic.
pub fn volley(scene: &Scene, raking: u32, base: u32, every: u32) -> Vec<Shot> {
    let b = block(scene);
    let mut k: u64 = 0x5eed;
    let mut rnd = || {
        k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((k >> 33) & 0xFFFF) as f64 / 65535.0
    };
    let centre = [0.5 * (b.lo[0] + b.hi[0]), 0.0, 0.5 * (b.lo[2] + b.hi[2])];
    let radius = 0.5 * (b.hi[0] - b.lo[0]).max(b.hi[2] - b.lo[2]) + b.half + 8.0;
    let height = |f: f64| b.lo[1] + f * (b.hi[1] - b.lo[1]);
    let mut shots = Vec::new();
    for i in 0..raking + base {
        let t = b.towers[i as usize % b.towers.len()];
        let raking_shot = i < raking;
        // Approach from around the block, 20-45° off the tower's face normal,
        // aimed off the tower's axis so the ball clips it and deflects on.
        let side = (i as usize / b.towers.len() + i as usize) % 4;
        let face = [0.0f64, 0.5, 1.0, 1.5][side] * std::f64::consts::PI;
        let skew = (20.0 + 25.0 * rnd()).to_radians() * if rnd() < 0.5 { -1.0 } else { 1.0 };
        let yaw = face + if raking_shot { skew } else { 0.3 * skew };
        let (fy, pitch) = if raking_shot {
            (0.2 + 0.7 * rnd(), -(5.0 + 15.0 * rnd()).to_radians())
        } else {
            (0.04 + 0.12 * rnd(), -(2.0 * rnd()).to_radians())
        };
        let lateral = if raking_shot {
            (0.25 + 0.4 * rnd()) * b.half * if rnd() < 0.5 { -1.0 } else { 1.0 }
        } else {
            0.3 * (rnd() - 0.5) * b.half
        };
        // Direction of travel (towards the tower), and a point beside its axis.
        let dir = norm([-(yaw.cos()) * pitch.cos(), pitch.sin(), -(yaw.sin()) * pitch.cos()]);
        let across = norm([-dir[2], 0.0, dir[0]]);
        let aim = [t[0] + across[0] * lateral, height(fy), t[2] + across[2] * lateral];
        // Start outside the block on the line through the aim point.
        let to_c = [aim[0] - centre[0], aim[2] - centre[2]];
        let along = to_c[0] * dir[0] + to_c[1] * dir[2];
        let back = radius + along;
        let origin = [aim[0] - dir[0] * back, aim[1] - dir[1] * back, aim[2] - dir[2] * back];
        shots.push(Shot {
            step: 8 + i * every,
            origin,
            dir,
            label: format!("{} {} tower {}", if raking_shot { "raking" } else { "base" }, i, i as usize % b.towers.len()),
        });
    }
    shots
}
