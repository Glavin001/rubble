//! Debug drawing (Box3D `b3DebugDraw`): the world calls back for every shape
//! at its body's last polled pose, so a host draws colliders with its own
//! renderer. Host-side and frame-latent: no GPU readback.

use crate::defs::Geometry;
use crate::{ShapeMesh, Transform, World};
use glam::{Quat, Vec3};

/// RGBA, 0..1.
pub type DebugColor = [f32; 4];

/// Colours by body state.
pub const COLOR_STATIC: DebugColor = [0.55, 0.55, 0.6, 1.0];
pub const COLOR_AWAKE: DebugColor = [0.35, 0.85, 0.45, 1.0];
pub const COLOR_ASLEEP: DebugColor = [0.35, 0.5, 0.9, 1.0];

/// Callbacks for [`World::draw`]. Shapes come in world space.
pub trait DebugDraw {
    fn draw_box(&mut self, transform: Transform, half: [f32; 3], color: DebugColor);
    fn draw_sphere(&mut self, center: [f32; 3], radius: f32, color: DebugColor);
    /// The segment `p0..p1` swept by `radius`.
    fn draw_capsule(&mut self, p0: [f32; 3], p1: [f32; 3], radius: f32, color: DebugColor);
    /// A convex hull, as a mesh in its own frame placed by `transform`.
    fn draw_mesh(&mut self, transform: Transform, mesh: &ShapeMesh, color: DebugColor);
    /// A body's origin frame (optional).
    fn draw_transform(&mut self, _transform: Transform) {}
}

/// What [`World::draw`] includes.
#[derive(Clone, Copy, Debug)]
pub struct DebugDrawOptions {
    pub shapes: bool,
    pub body_transforms: bool,
    /// Shapes' `SurfaceMaterial::custom_color` (0xRRGGBB, non-zero) override
    /// the state colours.
    pub material_colors: bool,
}

impl Default for DebugDrawOptions {
    fn default() -> Self {
        Self {
            shapes: true,
            body_transforms: false,
            material_colors: true,
        }
    }
}

impl World {
    /// Draw every user shape at its body's last polled pose.
    pub fn draw(&self, draw: &mut impl DebugDraw, options: DebugDrawOptions) {
        for (b, slot) in self.bodies.iter().enumerate() {
            if !slot.alive {
                continue;
            }
            let Some(body) = slot.last_transform else {
                continue;
            };
            if options.body_transforms {
                draw.draw_transform(body);
            }
            if !options.shapes {
                continue;
            }
            let state = match (slot.body_type, slot.last_asleep) {
                (crate::BodyType::Static, _) => COLOR_STATIC,
                (_, true) => COLOR_ASLEEP,
                _ => COLOR_AWAKE,
            };
            let _ = b;
            for &s in &slot.shapes {
                let shape = &self.shapes[s as usize];
                let c = shape.def.material.custom_color;
                let color = if options.material_colors && c != 0 {
                    [
                        ((c >> 16) & 0xFF) as f32 / 255.0,
                        ((c >> 8) & 0xFF) as f32 / 255.0,
                        (c & 0xFF) as f32 / 255.0,
                        1.0,
                    ]
                } else {
                    state
                };
                let place = |local: Transform| compose(body, local);
                match &shape.geometry {
                    Geometry::Sphere { center, radius } => {
                        let p = place(Transform::new(*center, [0.0, 0.0, 0.0, 1.0]));
                        draw.draw_sphere(p.position, *radius, color);
                    }
                    Geometry::Capsule {
                        center,
                        radius,
                        half_len,
                    } => {
                        let q = body.q();
                        let c = body.p() + q * Vec3::from(*center);
                        let axis = q * Vec3::Y * *half_len;
                        draw.draw_capsule(
                            (c - axis).to_array(),
                            (c + axis).to_array(),
                            *radius,
                            color,
                        );
                    }
                    Geometry::Box { half, transform } => {
                        draw.draw_box(place(*transform), *half, color);
                    }
                    // Infinite grounds: hosts draw their own floor.
                    Geometry::Plane { .. } => {}
                    Geometry::Hull { hull, transform } => {
                        let mut triangles = Vec::new();
                        for f in &hull.faces {
                            for k in 1..f.indices.len().saturating_sub(1) {
                                let i = |n: usize| u32::from(f.indices[n]);
                                triangles.push([i(0), i(k), i(k + 1)]);
                            }
                        }
                        let mesh = ShapeMesh {
                            vertices: hull.vertices.iter().map(|v| v.to_array()).collect(),
                            triangles,
                        };
                        draw.draw_mesh(place(*transform), &mesh, color);
                    }
                }
            }
        }
    }
}

fn compose(a: Transform, b: Transform) -> Transform {
    let q = a.q();
    Transform::new(
        (a.p() + q * b.p()).to_array(),
        (q * Quat::from_array(b.rotation)).normalize().to_array(),
    )
}
