//! What both hosts draw besides the GPU shader (`slang/render/fracture_draw.slang`):
//! the orbit camera and click-to-shoot ray, the floor quad, and the indexed
//! house mesh.

use crate::math::*;
use crate::scene::{vertex, Ray, Scene};
use std::collections::HashMap;

/// Vertical field of view (radians).
pub const FOV_Y: f64 = 0.7;
pub const NEAR: f64 = 0.02;

/// Orbit camera around the scene centre: drag to orbit, scroll to zoom.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub yaw: f64,
    pub pitch: f64,
    pub distance: f64,
    pub target: [f64; 3],
    pub extent: f64,
    pub min_distance: f64,
}

impl Orbit {
    pub fn new(scene: &Scene) -> Self {
        let c = scene.camera;
        Self {
            yaw: c.map_or(0.25, |c| c.yaw),
            pitch: c.map_or(0.22, |c| c.pitch),
            distance: c.map_or(scene.extent * 1.9, |c| c.distance),
            target: scene.center,
            extent: scene.extent,
            min_distance: c.map_or(scene.extent * 0.35, |c| c.min_distance),
        }
    }

    /// Eye position; narrow viewports back off so the scene still fits.
    pub fn eye(&self, aspect: f64) -> [f64; 3] {
        let fitted = self.distance / aspect.min(1.0);
        let (sy, cy, sp, cp) = (
            self.yaw.sin(),
            self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos(),
        );
        add(self.target, scale([sy * cp, sp, cy * cp], fitted))
    }

    /// Column-major view-projection (right-handed, depth 0..1).
    pub fn view_proj(&self, aspect: f64) -> [f32; 16] {
        let (eye, far) = (self.eye(aspect), (self.extent * 50.0).max(500.0));
        let z = norm(sub(eye, self.target));
        let x = norm(cross([0.0, 1.0, 0.0], z));
        let y = cross(z, x);
        let view = [
            x[0],
            y[0],
            z[0],
            0.0,
            x[1],
            y[1],
            z[1],
            0.0,
            x[2],
            y[2],
            z[2],
            0.0,
            -dot(x, eye),
            -dot(y, eye),
            -dot(z, eye),
            1.0,
        ];
        let f = 1.0 / (FOV_Y / 2.0).tan();
        let proj = [
            f / aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            f,
            0.0,
            0.0,
            0.0,
            0.0,
            far / (NEAR - far),
            -1.0,
            0.0,
            0.0,
            NEAR * far / (NEAR - far),
            0.0,
        ];
        std::array::from_fn(|i| {
            (0..4)
                .map(|k| proj[k * 4 + i % 4] * view[(i / 4) * 4 + k])
                .sum::<f64>() as f32
        })
    }

    pub fn drag(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-0.1, 1.3);
    }

    pub fn zoom(&mut self, delta_y: f64) {
        self.distance = (self.distance * (delta_y * 0.001).exp())
            .min(self.extent * 8.0)
            .max(self.min_distance);
    }

    /// Shot through a viewport point (u, v in 0..1 from the top left), starting
    /// just in front of the eye.
    pub fn aim(&self, u: f64, v: f64, aspect: f64, projectile_radius: f64) -> Ray {
        let eye = self.eye(aspect);
        let forward = norm(sub(self.target, eye));
        let right = norm(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        let t = (FOV_Y / 2.0).tan();
        let (x, y) = ((2.0 * u - 1.0) * aspect * t, (1.0 - 2.0 * v) * t);
        let direction = norm(add(forward, add(scale(right, x), scale(up, y))));
        Ray {
            origin: add(eye, scale(direction, projectile_radius * 2.0)),
            direction,
        }
    }
}

/// The ground quad (body 0), sized to the scene.
pub fn floor_vertices(extent: f64) -> Vec<f32> {
    let size = (extent * 8.0).max(20.0);
    let mut v = Vec::new();
    for [x, z] in [
        [-size, -size],
        [-size, size],
        [size, size],
        [-size, -size],
        [size, size],
        [size, -size],
    ] {
        vertex(&mut v, [x, 0.0, z], [0.0, 1.0, 0.0], [0.1, 0.15, 0.18], 0);
    }
    v
}

/// Indexed render mesh: corners repeated by the triangle list are stored once
/// (only exact matches of all ten attributes merge, body id included), so each
/// instanced copy runs far fewer vertex shaders for the same image.
pub fn index_mesh(vertices: &[f32]) -> (Vec<f32>, Vec<u32>) {
    let mut seen: HashMap<[u32; 10], u32> = HashMap::new();
    let (mut unique, mut index) = (Vec::new(), Vec::with_capacity(vertices.len() / 10));
    for v in vertices.chunks_exact(10) {
        let key: [u32; 10] = std::array::from_fn(|k| v[k].to_bits());
        let next = (unique.len() / 10) as u32;
        let k = *seen.entry(key).or_insert_with(|| {
            unique.extend_from_slice(v);
            next
        });
        index.push(k);
    }
    (unique, index)
}
