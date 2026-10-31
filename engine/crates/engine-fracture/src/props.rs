//! Street props for the city demos: parked cars and pedestrians as dynamic
//! box bodies on the streets around a block of towers (`towers.rs`), and the
//! directed-debris zones that make wreckage land on and around them.
//!
//! Props come after the projectile slots, so the pieces stay `1..projectile_start`
//! and the projectiles keep their slots; they start asleep like the pieces.

use crate::cuboid::FACES;
use crate::scene::{
    vertex, Body, Collider, Scene, FLAG_DYNAMIC, FLAG_SLEEPING, SHAPE_BOX,
};
use crate::towers::block;
use engine_schema::guidance::{ZoneDesc, ZoneShape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PropKind {
    Car,
    Pedestrian,
}

/// One prop: what it is, where its base stands and its heading (radians
/// about +Y; a car's length lies along its heading).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prop {
    pub kind: PropKind,
    pub at: [f64; 3],
    pub yaw: f64,
}

impl PropKind {
    /// Half extents (x across, y up, z along the heading) and mass (kg).
    fn size(self) -> ([f64; 3], f64) {
        match self {
            Self::Car => ([0.9, 0.7, 2.2], 1300.0),
            Self::Pedestrian => ([0.22, 0.88, 0.15], 75.0),
        }
    }
}

const CAR_COLORS: [[f64; 3]; 5] = [
    [0.80, 0.14, 0.12],
    [0.15, 0.35, 0.75],
    [0.85, 0.85, 0.82],
    [0.12, 0.13, 0.15],
    [0.90, 0.65, 0.10],
];
const PEDESTRIAN_COLORS: [[f64; 3]; 4] = [
    [0.95, 0.45, 0.65],
    [0.30, 0.80, 0.55],
    [0.98, 0.85, 0.30],
    [0.45, 0.75, 0.98],
];

/// The scene with `city_props` on its streets.
pub fn with_city(scene: &Scene) -> Scene {
    let mut s = scene.clone();
    let props = city_props(&s);
    add_props(&mut s, &props);
    s
}

/// The zones of the scene's props (`prop_zones`).
pub fn city_zones(scene: &Scene) -> Vec<ZoneDesc> {
    prop_zones(&scene.props, &scene.prop_bodies)
}

/// Add `props` to the scene; returns their body indices.
pub fn add_props(scene: &mut Scene, props: &[Prop]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(props.len());
    for (k, p) in props.iter().enumerate() {
        let (h, m) = p.kind.size();
        let id = scene.bodies.len() as u32;
        let q = [0.0, (p.yaw / 2.0).sin(), 0.0, (p.yaw / 2.0).cos()];
        scene.bodies.push(Body {
            p: [p.at[0], p.at[1] + h[1], p.at[2]],
            q,
            v: [0.0; 3],
            w: [0.0; 3],
            inv_mass: 1.0 / m,
            inv_i: [
                3.0 / (m * (h[1] * h[1] + h[2] * h[2])),
                3.0 / (m * (h[0] * h[0] + h[2] * h[2])),
                3.0 / (m * (h[0] * h[0] + h[1] * h[1])),
            ],
            flags: FLAG_DYNAMIC | FLAG_SLEEPING,
        });
        let density = m / (8.0 * h[0] * h[1] * h[2]);
        scene
            .colliders
            .push(Collider::new(id, SHAPE_BOX, density, [h[0], h[1], h[2], 0.0]));
        let color = match p.kind {
            PropKind::Car => CAR_COLORS[k % CAR_COLORS.len()],
            PropKind::Pedestrian => PEDESTRIAN_COLORS[k % PEDESTRIAN_COLORS.len()],
        };
        for (n, quad) in FACES {
            for c in [0, 1, 2, 0, 2, 3] {
                let s = quad[c];
                vertex(
                    &mut scene.prop_vertices,
                    [s[0] * h[0], s[1] * h[1], s[2] * h[2]],
                    n,
                    color,
                    id,
                );
            }
        }
        ids.push(id);
    }
    scene.props.extend_from_slice(props);
    scene.prop_bodies.extend_from_slice(&ids);
    ids
}

/// Cars parked along the street around a towers block and pedestrians on the
/// pavements and in the crossing streets, laid out from the block's bounds.
pub fn city_props(scene: &Scene) -> Vec<Prop> {
    let b = block(scene);
    let ground = 0.0;
    let (lo, hi) = ([b.lo[0], b.lo[2]], [b.hi[0], b.hi[2]]);
    let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let mut props = Vec::new();
    // Each side of the block: cars at the kerb 4 m out, people on the pavement
    // 1.5 m out and a second line across the street 9 m out.
    for side in 0..4 {
        let (along_x, sign) = (side % 2 == 0, if side < 2 { 1.0 } else { -1.0 });
        let (from, to) = if along_x { (lo[0], hi[0]) } else { (lo[1], hi[1]) };
        let edge = |out: f64| {
            if along_x {
                if sign > 0.0 { hi[1] + out } else { lo[1] - out }
            } else if sign > 0.0 {
                hi[0] + out
            } else {
                lo[0] - out
            }
        };
        let at = |t: f64, out: f64| {
            if along_x {
                [t, ground, edge(out)]
            } else {
                [edge(out), ground, t]
            }
        };
        let yaw = if along_x { std::f64::consts::FRAC_PI_2 } else { 0.0 };
        let span = to - from;
        for k in 0..4 {
            let t = from + span * (k as f64 + 0.5) / 4.0;
            props.push(Prop {
                kind: PropKind::Car,
                at: at(t, 4.0),
                yaw,
            });
        }
        for k in 0..6 {
            let t = from + span * (k as f64 + 0.5) / 6.0;
            props.push(Prop {
                kind: PropKind::Pedestrian,
                at: at(t, 1.5),
                yaw: 0.0,
            });
            if k % 2 == 1 {
                props.push(Prop {
                    kind: PropKind::Pedestrian,
                    at: at(t + 1.0, 9.0),
                    yaw: 0.0,
                });
            }
        }
    }
    // The crossing streets between the towers.
    for k in 0..4 {
        let t = (k as f64 + 0.5) / 4.0;
        props.push(Prop {
            kind: PropKind::Pedestrian,
            at: [lo[0] + (hi[0] - lo[0]) * t, ground, centre[1]],
            yaw: 0.0,
        });
        props.push(Prop {
            kind: PropKind::Pedestrian,
            at: [centre[0], ground, lo[1] + (hi[1] - lo[1]) * t],
            yaw: 0.0,
        });
    }
    props
}

/// Zones on the props, anchored to their bodies so they follow a car that is
/// shoved or a pedestrian knocked over: a ring of near misses around each
/// pedestrian, and the roof of each car (a few pieces each).
pub fn prop_zones(props: &[Prop], bodies: &[u32]) -> Vec<ZoneDesc> {
    props
        .iter()
        .zip(bodies)
        .map(|(p, &body)| match p.kind {
            PropKind::Pedestrian => ZoneDesc {
                anchor_body: Some(body),
                // From the body's centre down to the ground.
                offset: [0.0, -0.88, 0.0],
                shape: ZoneShape::Ring {
                    inner: 0.7,
                    outer: 2.5,
                    height: 0.0,
                },
                ..ZoneDesc::default()
            },
            PropKind::Car => ZoneDesc {
                anchor_body: Some(body),
                anchor_relative: true,
                offset: [0.0, 0.9, 0.0],
                shape: ZoneShape::Box {
                    half_extents: [0.8, 0.2, 2.0],
                },
                weight: 1.5,
                quota: Some(6),
                ..ZoneDesc::default()
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_schema::convex::HullArena;

    fn block_scene() -> Scene {
        // Four 6 m pieces as "towers" in a 2 x 2 grid (one body per copy).
        let arena = HullArena::new();
        let mut s = Scene::from_parts(
            vec![Body::fixed([0.0; 3])],
            vec![Collider::ground()],
            &arena,
            Default::default(),
        );
        for (x, z) in [(-6.0, -6.0), (6.0, -6.0), (-6.0, 6.0), (6.0, 6.0)] {
            let mut b = Body::fixed([x, 10.0, z]);
            b.flags = FLAG_DYNAMIC;
            s.bodies.push(b);
        }
        s.instance_stride = 1;
        s.projectile_start = s.bodies.len() as u32;
        s
    }

    #[test]
    fn props_stand_on_streets_outside_the_towers_with_zones() {
        let mut s = block_scene();
        let props = city_props(&s);
        assert!(props.iter().filter(|p| p.kind == PropKind::Car).count() == 16);
        let ids = add_props(&mut s, &props);
        assert_eq!(ids.len(), props.len());
        assert_eq!(s.bodies.len(), 5 + props.len());
        assert_eq!(s.prop_vertices.len(), props.len() * 36 * 10);
        for (p, &id) in props.iter().zip(&ids) {
            let b = s.bodies[id as usize];
            assert!(b.p[1] > 0.0, "standing on the ground");
            // Outside the 6 m half-width towers (or in a crossing street).
            let inside = |c: f64| (c.abs() - 6.0).abs() < 0.5;
            assert!(!(inside(p.at[0]) && inside(p.at[2])), "not inside a tower: {:?}", p.at);
        }
        for z in prop_zones(&props, &ids) {
            z.to_gpu().unwrap();
        }
    }
}
