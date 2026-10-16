//! Collision-cost benchmark scenes (headless `scene=pairs`): `count` separated
//! stacks, each a static base of shape `a` with a dynamic body of shape `b`
//! resting on it, sleep disabled so every pair stays in contact. Measuring two
//! counts and taking the difference gives the marginal GPU cost of one live
//! pair of that shape combination.
//!
//! Shapes: box, sphere, capsule, hull<N> (an N-gon prism, 2N vertices,
//! N = 3-32), box4 (a compound body: four boxes tiling one box's footprint).

use crate::scene::*;
use anyhow::{bail, Result};
use engine_schema::convex::HullArena;

/// Box half extents (1 x 0.5 x 1 m slab).
const HALF: [f64; 3] = [0.5, 0.25, 0.5];
/// Sphere radius; capsule radius and half length.
const RADIUS: f64 = 0.25;

/// N-gon prism circumscribing the box footprint, same height as the box.
fn prism(n: usize) -> (Vec<[f64; 3]>, Vec<u32>, Vec<u32>) {
    use std::f64::consts::PI;
    let r = 0.5 / (PI / n as f64).cos();
    let mut v = Vec::new();
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64 * 2.0 * PI;
        v.push([r * t.cos(), -HALF[1], r * t.sin()]);
    }
    for i in 0..n {
        v.push([v[i][0], HALF[1], v[i][2]]);
    }
    let mut faces: Vec<Vec<u32>> = vec![
        (0..n as u32).collect(),
        (0..n as u32).map(|i| n as u32 + i).rev().collect(),
    ];
    for i in 0..n as u32 {
        let j = (i + 1) % n as u32;
        faces.push(vec![i, n as u32 + i, n as u32 + j, j]);
    }
    (
        v,
        faces.iter().map(|f| f.len() as u32).collect(),
        faces.concat(),
    )
}

fn colliders_for(shape: &str, body: u32, arena: &mut HullArena) -> Result<Vec<Collider>> {
    let c = |shape, params| Collider::new(body, shape, 1000.0, params);
    Ok(match shape {
        "box" => vec![c(SHAPE_BOX, [HALF[0], HALF[1], HALF[2], 0.0])],
        "sphere" => vec![c(SHAPE_SPHERE, [RADIUS, 0.0, 0.0, 0.0])],
        "capsule" => {
            let h = std::f64::consts::FRAC_1_SQRT_2;
            vec![Collider {
                local_rot: [0.0, 0.0, h, h],
                ..c(SHAPE_CAPSULE, [RADIUS, RADIUS, 0.0, 0.0])
            }]
        }
        "box4" => {
            let h = [HALF[0] / 2.0, HALF[1], HALF[2] / 2.0];
            let mut out = Vec::new();
            for sx in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    out.push(Collider {
                        local_pos: [sx * h[0], 0.0, sz * h[2]],
                        ..c(SHAPE_BOX, [h[0], h[1], h[2], 0.0])
                    });
                }
            }
            out
        }
        _ => {
            let Some(n) = shape
                .strip_prefix("hull")
                .and_then(|n| n.parse::<usize>().ok())
            else {
                bail!("Unknown bench shape {shape}");
            };
            let (v, sizes, indices) = prism(n);
            let (base, radius, local_pos) = append_hull(arena, &v, &sizes, &indices, 0.0)?;
            vec![Collider::hull(body, 1000.0, base, 1.0, radius, local_pos)]
        }
    })
}

/// Half height above the base top at which shape `b` rests.
fn rest_half(shape: &str) -> f64 {
    if shape == "sphere" || shape == "capsule" {
        RADIUS
    } else {
        HALF[1]
    }
}

pub fn build_pair_scene(a: &str, b: &str, count: u32, spacing: f64) -> Result<Scene> {
    let mut arena = HullArena::new();
    let (mut bodies, mut colliders) = (Vec::new(), Vec::new());
    let side = (count as f64).sqrt().ceil() as u32;
    for k in 0..count {
        let x = ((k % side) as f64 - side as f64 / 2.0) * spacing;
        let z = ((k / side) as f64 - side as f64 / 2.0) * spacing;
        let base = bodies.len() as u32;
        bodies.push(Body::fixed([x, HALF[1], z]));
        colliders.extend(colliders_for(a, base, &mut arena)?);
        // Every base shape reaches y = 2 * HALF[1] (= 2 * RADIUS) at its top.
        let top = bodies.len() as u32;
        let m = 1000.0 * 8.0 * HALF[0] * HALF[1] * HALF[2];
        let sq = |x: f64| x * x;
        bodies.push(Body {
            p: [x, 2.0 * HALF[1] + rest_half(b), z],
            inv_mass: 1.0 / m,
            inv_i: [
                3.0 / (m * (sq(HALF[1]) + sq(HALF[2]))),
                3.0 / (m * (sq(HALF[0]) + sq(HALF[2]))),
                3.0 / (m * (sq(HALF[0]) + sq(HALF[1]))),
            ],
            flags: FLAG_DYNAMIC,
            ..Body::fixed([0.0; 3])
        });
        colliders.extend(colliders_for(b, top, &mut arena)?);
    }
    let extent = side as f64 * spacing;
    // Flags: world-frame inertia (+1 contact reuse when the host asks). Sleep off.
    let params = SceneParams {
        sleep_lin: -1.0,
        sleep_ang: -1.0,
        sleep_time: 1e9,
        ..SceneParams::lab(FLAG_WORLD_INERTIA)
    };
    let mut s = Scene::from_parts(bodies, colliders, &arena, params);
    s.projectile_radius = 0.25;
    s.extent = extent;
    s.extent_x = extent;
    s.extent_z = extent;
    s.center = [0.0, 0.5, 0.0];
    s.capacity = Budget {
        candidates_per_collider: 8,
        contacts_per_collider: 16,
        ..Budget::default()
    };
    Ok(s)
}
