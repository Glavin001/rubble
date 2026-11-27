//! The explicit sim's static shapes as world bodies, for rendering only (the sim collides
//! with its own copies): the soft-body page's explicit scenes (`softbody::place`), and the
//! crash lab when a scene is placed again with other values (`WebWorld::placeExplicitBoxes`).

use crate::*;

/// One static body per row (`ExplicitSim::layout().boxes`: centre, half extents, rotation
/// quaternion, kind: 2 a cylinder's hull, 3 a capsule, 4 a sphere, else a box), each with one
/// shape of the page's prop material. Returns the bodies, in row order.
pub fn place(world: &mut World, rows: &[Vec<f64>]) -> anyhow::Result<Vec<BodyId>> {
    let pi = std::f64::consts::PI;
    let f3 = |v: [f64; 3]| v.map(|x| x as f32);
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let g = |i: usize, d: f64| row.get(i).copied().unwrap_or(d);
        let (hx, hy, hz) = (g(3, 0.0), g(4, 0.0), g(5, 0.0));
        let geometry = match g(10, 0.0) as u32 {
            2 => {
                let pts: Vec<glam::Vec3> = (0..32u32)
                    .map(|i| {
                        let a = (i >> 1) as f64 * pi / 8.0;
                        let y = if i & 1 != 0 { hy } else { -hy };
                        glam::Vec3::from_array(f3([hx * a.cos(), y, hx * a.sin()]))
                    })
                    .collect();
                Geometry::Hull { hull: ConvexHull::from_points(&pts).map_err(|e| anyhow::anyhow!("hull: {e:?}"))?, transform: Transform::IDENTITY }
            }
            3 => Geometry::Capsule { center: [0.0; 3], radius: hx as f32, half_len: hy as f32 },
            4 => Geometry::Sphere { center: [0.0; 3], radius: hx as f32 },
            _ => Geometry::Box { half: f3([hx, hy, hz]), transform: Transform::IDENTITY },
        };
        let body = world.create_body(&BodyDef {
            position: f3([g(0, 0.0), g(1, 0.0), g(2, 0.0)]),
            rotation: [g(6, 0.0), g(7, 0.0), g(8, 0.0), g(9, 1.0)].map(|x| x as f32),
            ..BodyDef::default()
        })?;
        let def = ShapeDef { material: SurfaceMaterial { friction: 0.6, ..SurfaceMaterial::default() }, ..ShapeDef::default() };
        world.create_shape(body, &def, geometry)?;
        out.push(body);
    }
    Ok(out)
}
