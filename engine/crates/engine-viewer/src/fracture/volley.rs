//! A city block of vibe-city fractured towers under a scripted volley of fast,
//! off-angle balls: the CCD showcase (`fracture_viewer towers --volley`) and
//! its headless twin (`tests/towers_volley.rs`) fire exactly the same shots.
//!
//! The volley first rakes the towers at a glancing angle (balls clip a corner,
//! smash through a few rooms, deflect and carry on into the next tower), then
//! hammers the towers' lower floors from outside the block so they lose their
//! base and fall into their neighbours.

use engine_fracture::pack::build_pack_scene;
use engine_fracture::scene::{build_grid, GridOptions, Scene};
use std::path::PathBuf;

/// The generated vibe-city pack (`bun run export.ts fracturedTower` in
/// engine/scenes/vibe-city).
pub fn tower_pack_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/generated/vibe-city/fracturedTower.json")
}

/// `columns × rows` fractured towers, `gap` metres of street between them,
/// firing the pack's own ball (the lab's grid layout swaps in a 200 t, 1.6 m
/// wrecking ball). `bonded`: the pack's breakable welds between pieces (the
/// lab's "Breakable D6 joints"); else loose pieces (bond mode none).
/// `ball`: (radius m, mass kg) instead of the pack's own. `strength` scales
/// every bond's break limits (a weaker masonry: a material choice of the scene).
pub fn towers_scene(
    columns: u32,
    rows: u32,
    gap: f64,
    bonded: bool,
    ball: Option<(f64, f64)>,
    strength: f64,
) -> anyhow::Result<Scene> {
    let path = tower_pack_path();
    let bytes = std::fs::read(&path).map_err(|e| {
        anyhow::anyhow!(
            "{}: {e} (bun run export.ts fracturedTower in engine/scenes/vibe-city)",
            path.display()
        )
    })?;
    let (one, _) = build_pack_scene(&bytes)?;
    let mut one = if bonded { one } else { one.without_joints() };
    for j in &mut one.joints {
        for x in &mut j.strength {
            *x *= strength;
        }
    }
    let o = GridOptions {
        columns,
        rows,
        gap,
        projectile_radius: ball.map_or(one.projectile_radius, |b| b.0),
        projectile_mass: ball.map_or(one.projectile_mass, |b| b.1),
        projectile_speed: one.projectile_speed.unwrap_or(40.0),
        ..GridOptions::default()
    };
    let mut s = build_grid(&one, &o);
    // The pack's camera frames one tower; frame the block.
    s.camera = None;
    Ok(s)
}

#[derive(Clone, Debug)]
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
    Block {
        towers,
        lo,
        hi,
        half,
    }
}

fn norm(d: [f64; 3]) -> [f64; 3] {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    d.map(|x| x / l)
}

/// The volley, `every` steps apart: `raking` glancing shots, then `base` shots
/// at the towers' lower floors.
pub fn volley(scene: &Scene, raking: u32, base: u32, every: u32) -> Vec<Shot> {
    let b = block(scene);
    let mut k: u64 = 0x5eed;
    let mut rnd = || {
        k = k
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        let dir = norm([
            -(yaw.cos()) * pitch.cos(),
            pitch.sin(),
            -(yaw.sin()) * pitch.cos(),
        ]);
        let across = norm([-dir[2], 0.0, dir[0]]);
        let aim = [
            t[0] + across[0] * lateral,
            height(fy),
            t[2] + across[2] * lateral,
        ];
        // Start outside the block on the line through the aim point.
        let to_c = [aim[0] - centre[0], aim[2] - centre[2]];
        let along = to_c[0] * dir[0] + to_c[1] * dir[2];
        let back = radius + along;
        let origin = [
            aim[0] - dir[0] * back,
            aim[1] - dir[1] * back,
            aim[2] - dir[2] * back,
        ];
        shots.push(Shot {
            step: 8 + i * every,
            origin,
            dir,
            label: format!(
                "{} {} tower {}",
                if raking_shot { "raking" } else { "base" },
                i,
                i as usize % b.towers.len()
            ),
        });
    }
    shots
}

/// Projectile trails, read from the shared body buffer after each tick: line
/// vertices (x, y, z, r, g, b, a) for `Renderer::set_trails`; cyan for CCD
/// shots, red for plain ones.
pub struct Trails {
    paths: Vec<(Vec<[f32; 3]>, bool)>,
}

impl Trails {
    pub fn new(scene: &Scene) -> Self {
        Self {
            paths: vec![(Vec::new(), false); scene.projectile_count as usize],
        }
    }
    /// A shot was fired into slot `slot`.
    pub fn launched(&mut self, slot: usize, ccd: bool) {
        if let Some(p) = self.paths.get_mut(slot) {
            *p = (Vec::new(), ccd);
        }
    }
    pub fn update(&mut self, sim: &super::FractureSim, scene: &Scene) {
        let start = scene.projectile_start as usize;
        let body = sim.body_buffer().contents() as *const f32;
        for (k, (path, _)) in self.paths.iter_mut().enumerate() {
            if (k as u32) >= sim.shots() {
                continue;
            }
            let p = unsafe { std::slice::from_raw_parts(body.add((start + k) * 28), 3) };
            let p = [p[0], p[1], p[2]];
            if p[1] < -100.0 {
                continue;
            }
            if path
                .last()
                .is_none_or(|q| (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f32>() > 1e-4)
            {
                path.push(p);
                if path.len() > 900 {
                    path.remove(0);
                }
            }
        }
    }
    pub fn vertices(&self) -> Vec<f32> {
        let mut verts = Vec::new();
        for (path, ccd) in &self.paths {
            let n = path.len();
            for (i, w) in path.windows(2).enumerate() {
                let a = 0.25 + 0.75 * (i + 1) as f32 / n as f32;
                let c = if *ccd {
                    [0.2, 0.95, 1.0, a]
                } else {
                    [1.0, 0.25, 0.2, a]
                };
                for q in w {
                    verts.extend_from_slice(&[q[0], q[1], q[2], c[0], c[1], c[2], c[3]]);
                }
            }
        }
        verts
    }
}
