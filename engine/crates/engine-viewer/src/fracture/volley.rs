//! The CCD showcase on Metal: the towers block and volley shared with the
//! browser lab (`engine_fracture::towers`), the native pack loader, and ball
//! trails for the viewer (`fracture_viewer towers --volley`, `towers_capture`,
//! `tests/towers_volley.rs`).

use engine_fracture::pack::build_pack_scene;
use engine_fracture::scene::Scene;
pub use engine_fracture::towers::{block, volley, Block, Shot};
use engine_fracture::towers::towers_grid;
use std::path::PathBuf;

/// The generated vibe-city pack (`bun run export.ts fracturedTower` in
/// engine/scenes/vibe-city).
pub fn tower_pack_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/generated/vibe-city/fracturedTower.json")
}

/// `columns × rows` fractured towers (`engine_fracture::towers::towers_grid`),
/// `gap` metres of street between them. `bonded`: the pack's breakable welds
/// between pieces (the lab's "Breakable D6 joints"); else loose pieces (bond
/// mode none). `ball`: (radius m, mass kg) instead of the pack's own.
/// `strength` scales every bond's break limits (a weaker masonry).
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
    let one = if bonded { one } else { one.without_joints() }.scale_bond_strength(strength);
    Ok(towers_grid(&one, columns, rows, gap, ball))
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
