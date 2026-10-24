//! "Meteor range": the plainest test of directed debris. Two identical lanes
//! of target pads side by side; each volley throws the same ballistic meteors
//! into both, every one set to land 1–4 m off a pad. Only the left lane is
//! steered, so the pads hit in each lane show at a glance whether steering
//! works. The page (vehicle.html `?scene=meteors`) builds the pads from
//! `range`, launches `volley` and uploads `zones`.

use engine_schema::guidance::{ZoneDesc, ZoneShape};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Lane centres (x): steered, then unsteered (the steered lane is on the
/// left of a camera looking down the range, +Z).
pub const LANES: [f32; 2] = [15.0, -15.0];
/// Pad centres within a lane (x offset, z) and their half size (m).
pub const PADS: [[f32; 2]; 3] = [[-4.0, 10.0], [4.0, 18.0], [0.0, 28.0]];
pub const PAD_HALF: f32 = 1.0;
/// Meteor radius (m) and mass (kg).
pub const RADIUS: f32 = 0.5;
pub const MASS: f32 = 500.0;
/// Launch point (lane-relative x, y, z) and flight time (s).
const LAUNCH: [f32; 3] = [0.0, 30.0, -50.0];
const FLIGHT: f32 = 3.0;
/// How far off its pad a meteor lands unsteered (m).
const MISS: (f32, f32) = (1.6, 4.0);

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Pad {
    pub p: [f32; 3],
    pub half: [f32; 3],
    pub lane: u32,
}

/// One meteor of a volley: its lane, pad, launch state and the tag mask that
/// steers it toward its pad's zone (0: not steered).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Launch {
    pub lane: u32,
    pub pad: u32,
    pub p: [f32; 3],
    pub v: [f32; 3],
    pub tags: u32,
}

/// Pad centres on the ground, both lanes.
pub fn pads() -> Vec<Pad> {
    LANES
        .iter()
        .enumerate()
        .flat_map(|(l, &x)| {
            PADS.iter().map(move |p| Pad {
                p: [x + p[0], 0.0, p[1]],
                half: [PAD_HALF, 0.02, PAD_HALF],
                lane: l as u32,
            })
        })
        .collect()
}

/// One zone per pad of the steered lane: a disc where the meteor's centre is
/// as it touches the pad (tag bit k for pad k).
pub fn zones() -> Vec<ZoneDesc> {
    PADS.iter()
        .enumerate()
        .map(|(k, p)| ZoneDesc {
            position: [LANES[0] + p[0], RADIUS, p[1]],
            // A flat disc at resting height: an aim above it would carry
            // the meteor on past the aim before it comes down.
            shape: ZoneShape::Ring {
                inner: 0.0,
                outer: 0.6,
                height: 0.0,
            },
            tags: 1 << k,
            ..ZoneDesc::default()
        })
        .collect()
}

fn hash(mut x: u32) -> f32 {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    ((x >> 22) ^ x) as f32 / u32::MAX as f32
}

/// Volley `k`: one meteor per pad per lane. The two lanes get the same arcs
/// (shifted by the lane spacing); each lands MISS off its pad, unsteered.
pub fn volley(k: u32, gravity_y: f32) -> Vec<Launch> {
    let mut out = Vec::new();
    for (pi, pad) in PADS.iter().enumerate() {
        let s = k * 7 + pi as u32;
        let ang = std::f32::consts::TAU * hash(s * 2 + 1);
        let miss = MISS.0 + (MISS.1 - MISS.0) * hash(s * 2 + 2);
        let land = [pad[0] + miss * ang.cos(), RADIUS, pad[1] + miss * ang.sin()];
        // Each pad's meteors launch from their own point (behind the pad), so
        // a volley's meteors never meet in the air.
        let p0 = [
            LAUNCH[0] + pad[0] + 2.0 * (hash(s * 2 + 3) - 0.5),
            LAUNCH[1],
            LAUNCH[2],
        ];
        let v = [0, 1, 2].map(|a| {
            let g = if a == 1 { gravity_y } else { 0.0 };
            (land[a] - p0[a] - 0.5 * g * FLIGHT * FLIGHT) / FLIGHT
        });
        for (l, &x) in LANES.iter().enumerate() {
            out.push(Launch {
                lane: l as u32,
                pad: pi as u32,
                p: [x + p0[0], p0[1], p0[2]],
                v,
                tags: if l == 0 { 1 << pi } else { 0 },
            });
        }
    }
    out
}

/// Whether a meteor at rest at `p` is on pad `pad` of lane `lane`.
pub fn on_pad(lane: u32, pad: u32, p: [f32; 3]) -> bool {
    let c = PADS[pad as usize];
    let x = LANES[lane as usize] + c[0];
    (p[0] - x).abs() <= PAD_HALF && (p[2] - c[1]).abs() <= PAD_HALF
}

#[wasm_bindgen]
pub struct MeteorRange;

#[wasm_bindgen]
impl MeteorRange {
    /// `{pads, radius, mass}` JSON.
    pub fn range() -> String {
        serde_json::json!({ "pads": pads(), "radius": RADIUS, "mass": MASS }).to_string()
    }

    /// The steered lane's zones (`ZoneDesc` JSON).
    pub fn zones() -> String {
        serde_json::to_string(&zones()).unwrap()
    }

    /// Volley `k` (`Launch` JSON array).
    pub fn volley(k: u32, gravity_y: f32) -> String {
        serde_json::to_string(&volley(k, gravity_y)).unwrap()
    }

    /// Whether a meteor of `lane` aimed at `pad` came down on it at (x, y, z).
    pub fn on_pad(lane: u32, pad: u32, x: f32, y: f32, z: f32) -> bool {
        on_pad(lane, pad, [x, y, z])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsteered_meteors_miss_their_pads_and_lanes_match() {
        let g = -10.0;
        for k in 0..20 {
            let v = volley(k, g);
            assert_eq!(v.len(), PADS.len() * 2);
            for pair in v.chunks(2) {
                let (a, b) = (pair[0], pair[1]);
                assert_eq!(a.v, b.v);
                assert!((b.p[0] - a.p[0] - (LANES[1] - LANES[0])).abs() < 1e-4);
                assert!(a.tags != 0 && b.tags == 0);
                // The ballistic landing is off its pad.
                let land = [0, 1, 2].map(|i| {
                    let gi = if i == 1 { g } else { 0.0 };
                    b.p[i] + b.v[i] * FLIGHT + 0.5 * gi * FLIGHT * FLIGHT
                });
                assert!((land[1] - RADIUS).abs() < 1e-3);
                assert!(!on_pad(b.lane, b.pad, land), "{land:?}");
            }
        }
        for z in zones() {
            z.to_gpu().unwrap();
        }
        assert_eq!(pads().len(), 6);
    }
}
