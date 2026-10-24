//! "Near Miss": the lab's directed-debris preset. Each shot finds a crowd of
//! bystanders downrange, and each bystander gets a ring zone, so debris that
//! can reach one lands around that bystander rather than on them. Also builds the zone outlines for the
//! renderer and counts the pieces that came to rest near each bystander. The
//! steering itself is `engine-schema` guidance (`guide_step.slang`), the same
//! as `World`'s.

use crate::scene::{vertex, Scene};
use engine_schema::guidance::{GuidanceSettings, ZoneDesc, ZoneShape};
use serde::{Deserialize, Serialize};

/// How strongly the preset steers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Strength {
    /// The shipped defaults (6 % kick, 3 % g): meant to go unnoticed.
    Subtle,
    /// Twice the kick, a quarter of the shipped limit on held acceleration.
    Cinematic,
    /// Large enough to see the mechanism at work (not meant to be invisible).
    Exaggerated,
}

/// The settings of a strength. House debris flies short arcs, so the lab
/// steers flights from 0.3 s.
pub fn settings(strength: Strength) -> GuidanceSettings {
    let base = GuidanceSettings {
        min_flight: 0.3,
        min_speed: 0.5,
        ..GuidanceSettings::default()
    };
    match strength {
        Strength::Subtle => base,
        Strength::Cinematic => GuidanceSettings {
            launch_kick: 0.12,
            launch_cone: 6.0f32.to_radians(),
            accel_max_g: 0.08,
            ..base
        },
        Strength::Exaggerated => GuidanceSettings {
            launch_kick: 0.3,
            launch_cone: 16.0f32.to_radians(),
            accel_max_g: 0.25,
            ..base
        },
    }
}

/// Ring radii around a bystander (m): debris lands between them.
pub const RING_INNER: f32 = 0.6;
pub const RING_OUTER: f32 = 2.0;
/// Distances (m) past the impact and bearings (radians off the shot) of the
/// crowd a shot finds downrange.
const FAN_DISTANCES: [f32; 4] = [7.0, 12.0, 18.0, 25.0];
const FAN_BEARINGS: [f32; 4] = [-0.6, -0.2, 0.2, 0.6];

/// Bystanders in a fan downrange of a shot: past the point where the ray
/// passes closest to the scene's centre, along its heading, on the ground.
/// Debris from a hit flies mostly that way, so this is where it can reach.
pub fn downrange(scene: &Scene, origin: [f64; 3], direction: [f64; 3]) -> Vec<[f32; 3]> {
    let flat = [direction[0], direction[2]];
    let len = (flat[0] * flat[0] + flat[1] * flat[1]).sqrt();
    let (hx, hz) = if len > 1e-9 {
        (flat[0] / len, flat[1] / len)
    } else {
        (0.0, -1.0)
    };
    let c = scene.center;
    // The ray's closest approach to the centre, in plan view.
    let t = ((c[0] - origin[0]) * hx + (c[2] - origin[2]) * hz).max(0.0);
    let (ix, iz) = (origin[0] + hx * t, origin[2] + hz * t);
    let half = 0.5 * scene.extent_x.max(scene.extent_z).max(1.0);
    let mut out = Vec::with_capacity(FAN_DISTANCES.len() * FAN_BEARINGS.len());
    for d in FAN_DISTANCES {
        for b in FAN_BEARINGS {
            let (s, cs) = (b as f64).sin_cos();
            let (dx, dz) = (hx * cs - hz * s, hx * s + hz * cs);
            let r = half + d as f64;
            out.push([(ix + dx * r) as f32, 0.0, (iz + dz * r) as f32]);
        }
    }
    out
}

/// One ring zone per bystander of a shot's downrange crowd.
pub fn zones_downrange(scene: &Scene, origin: [f64; 3], direction: [f64; 3]) -> Vec<ZoneDesc> {
    ring_zones(downrange(scene, origin, direction))
}

fn ring_zones(points: Vec<[f32; 3]>) -> Vec<ZoneDesc> {
    points
        .into_iter()
        .map(|p| ZoneDesc {
            position: p,
            shape: ZoneShape::Ring {
                inner: RING_INNER,
                outer: RING_OUTER,
                height: 0.0,
            },
            ..ZoneDesc::default()
        })
        .collect()
}

/// Line-list vertices (10 floats each, on body 0) outlining each zone: its
/// rings, and a 1.8 m post for a ring's bystander. `ground` is body 0's
/// position, which the draw adds back.
pub fn outline_vertices(zones: &[ZoneDesc], ground: [f64; 3]) -> Vec<f32> {
    const SEGMENTS: usize = 32;
    const COLOR: [f64; 3] = [1.0, 0.62, 0.16];
    const POST: [f64; 3] = [1.0, 0.92, 0.55];
    let up = [0.0, 1.0, 0.0];
    let mut out = Vec::new();
    let local = |p: [f32; 3], dx: f64, dy: f64, dz: f64| {
        [
            p[0] as f64 + dx - ground[0],
            p[1] as f64 + dy - ground[1] + 0.02,
            p[2] as f64 + dz - ground[2],
        ]
    };
    for z in zones {
        // Outlines of fixed zones only: a zone on a moving body has no fixed place.
        if z.anchor_body.is_some() {
            continue;
        }
        let p = [z.position[0] + z.offset[0], z.position[1] + z.offset[1], z.position[2] + z.offset[2]];
        let radii: Vec<f64> = match z.shape {
            ZoneShape::Ring { inner, outer, .. } | ZoneShape::Arc { inner, outer, .. } => {
                vec![inner as f64, outer as f64]
            }
            ZoneShape::Sphere { radius } => vec![radius as f64],
            ZoneShape::Box { half_extents } => vec![half_extents[0].max(half_extents[2]) as f64],
            ZoneShape::Point => vec![0.25],
        };
        for r in radii {
            for s in 0..SEGMENTS {
                let a0 = s as f64 / SEGMENTS as f64 * std::f64::consts::TAU;
                let a1 = (s + 1) as f64 / SEGMENTS as f64 * std::f64::consts::TAU;
                vertex(&mut out, local(p, r * a0.cos(), 0.0, r * a0.sin()), up, COLOR, 0);
                vertex(&mut out, local(p, r * a1.cos(), 0.0, r * a1.sin()), up, COLOR, 0);
            }
        }
        if matches!(z.shape, ZoneShape::Ring { .. }) {
            vertex(&mut out, local(p, 0.0, 0.0, 0.0), up, POST, 0);
            vertex(&mut out, local(p, 0.0, 1.8, 0.0), up, POST, 0);
        }
    }
    out
}

/// Floats per body in a body snapshot (`BODY_BYTES` / 4).
const BODY_FLOATS: usize = 28;

/// Pieces `first..last` of a body snapshot that lie near each ring zone: on
/// the ground (below 1.5 m) within its outer radius in plan view.
pub fn near_counts(snapshot: &[f32], first: u32, last: u32, zones: &[ZoneDesc]) -> Vec<u32> {
    let mut counts = vec![0u32; zones.len()];
    for i in first as usize..(last as usize).min(snapshot.len() / BODY_FLOATS) {
        let p = &snapshot[i * BODY_FLOATS..i * BODY_FLOATS + 3];
        if p[1] > 1.5 {
            continue;
        }
        for (k, z) in zones.iter().enumerate() {
            let outer = match z.shape {
                ZoneShape::Ring { outer, .. } | ZoneShape::Arc { outer, .. } => outer,
                ZoneShape::Sphere { radius } => radius,
                ZoneShape::Box { half_extents } => half_extents[0].max(half_extents[2]),
                ZoneShape::Point => 0.5,
            };
            let c = [z.position[0] + z.offset[0], z.position[2] + z.offset[2]];
            let (dx, dz) = (p[0] - c[0], p[2] - c[1]);
            if dx * dx + dz * dz <= outer * outer {
                counts[k] += 1;
            }
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strengths_validate_and_grow() {
        let s: Vec<_> = [Strength::Subtle, Strength::Cinematic, Strength::Exaggerated]
            .iter()
            .map(|&k| settings(k))
            .collect();
        for g in &s {
            g.validate().unwrap();
        }
        assert!(s[0].accel_max_g < s[1].accel_max_g && s[1].accel_max_g < s[2].accel_max_g);
    }

    #[test]
    fn a_shot_finds_its_crowd_downrange() {
        let arena = engine_schema::convex::HullArena::new();
        let mut scene = Scene::from_parts(vec![], vec![], &arena, Default::default());
        scene.extent_x = 6.0;
        scene.extent_z = 4.0;
        scene.center = [0.0, 2.0, 0.0];
        // Fired from +Z toward -Z, passing the centre.
        let z = zones_downrange(&scene, [0.0, 3.0, 20.0], [0.0, -0.1, -1.0]);
        assert_eq!(z.len(), 16);
        for d in &z {
            assert!(d.position[2] < -8.0, "downrange (-Z) of the house: {:?}", d.position);
            assert!(d.position[0].abs() < 0.7 * -d.position[2], "inside the fan: {:?}", d.position);
        }
    }

    #[test]
    fn outlines_close_and_counts_see_the_rings() {
        let arena = engine_schema::convex::HullArena::new();
        let mut scene = Scene::from_parts(vec![], vec![], &arena, Default::default());
        scene.extent_x = 6.0;
        scene.extent_z = 4.0;
        let z = zones_downrange(&scene, [0.0, 3.0, 20.0], [0.0, 0.0, -1.0]);
        for d in &z {
            d.to_gpu().unwrap();
        }
        let v = outline_vertices(&z, [0.0; 3]);
        // Two rings of 32 segments and a post per zone, 2 vertices each.
        assert_eq!(v.len(), z.len() * (2 * 32 + 1) * 2 * 10);
        let mut snap = vec![0.0f32; 3 * BODY_FLOATS];
        snap[BODY_FLOATS..BODY_FLOATS + 3].copy_from_slice(&[z[0].position[0] + 1.0, 0.1, z[0].position[2]]);
        snap[2 * BODY_FLOATS..2 * BODY_FLOATS + 3].copy_from_slice(&[0.0, 0.1, 0.0]);
        let c = near_counts(&snap, 1, 3, &z);
        assert_eq!(c[0], 1);
        assert_eq!(c.iter().sum::<u32>(), 1);
    }
}
