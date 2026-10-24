//! "Meteor range": the plainest test of directed debris. Two identical lanes
//! of target pads side by side; each volley throws the same ballistic meteors
//! into both, every one set to land 1–4 m off a pad. Only the left lane is
//! steered, so the pads hit in each lane show at a glance whether steering
//! works. The page (vehicle.html `?scene=meteors`) builds the pads from
//! `range`, launches `volley` and uploads `zones`.
//!
//! With moving targets every pad is a kinematic cart sliding across its lane
//! (`pad_x`, the same in both lanes); the steered zones ride on the carts, and
//! unsteered meteors still land off where their cart will be. With known
//! routes the zones also carry each cart's path for the next few seconds
//! (`ZoneMotion::Path`), so the planner sees a turn coming; the page uploads
//! them again before every volley.

use engine_schema::guidance::{ZoneDesc, ZoneMotion, ZoneShape, GUIDE_PATH_KEYS};
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
/// Moving targets: cart speed (m/s) and how far either side of its spot it
/// slides (m), reversing at each end; each pad starts at its own phase (s).
pub const CART_SPEED: f32 = 2.0;
pub const CART_SWING: f32 = 6.0;
const CART_PHASE: f32 = 2.3;
/// How far ahead (s) a known route reaches: past every flight's landing.
const ROUTE_AHEAD: f32 = 8.0;

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

/// Pad `pad`'s centre x in lane `lane` at time `t` (s).
pub fn pad_x(lane: u32, pad: u32, t: f32, moving: bool) -> f32 {
    let x = LANES[lane as usize] + PADS[pad as usize][0];
    if !moving {
        return x;
    }
    // A triangle wave in [-1, 1] that changes at CART_SPEED / CART_SWING.
    let u = (t + CART_PHASE * pad as f32) * CART_SPEED / (4.0 * CART_SWING);
    x + CART_SWING * (4.0 * (u.fract() - 0.5).abs() - 1.0)
}

/// Pad `pad`'s route from time `t` to `t + ROUTE_AHEAD`: its position now,
/// at each turn and at the end (the triangle wave's corners, so the straight
/// segments between them are exact).
pub fn route(lane: u32, pad: u32, t: f32) -> Vec<([f32; 3], f32)> {
    let leg = 2.0 * CART_SWING / CART_SPEED;
    let shift = CART_PHASE * pad as f32;
    let at = |u: f32| ([pad_x(lane, pad, u, true), 0.0, PADS[pad as usize][1]], u);
    let mut keys = vec![at(t)];
    let mut n = ((t + shift) / leg).floor() + 1.0;
    while n * leg - shift < t + ROUTE_AHEAD {
        keys.push(at(n * leg - shift));
        n += 1.0;
    }
    keys.push(at(t + ROUTE_AHEAD));
    debug_assert!(keys.len() <= GUIDE_PATH_KEYS);
    keys
}

/// One zone per pad of the steered lane: a disc where the meteor's centre is
/// as it touches the pad (tag bit k for pad k). Moving pads carry their zone
/// (`pad_bodies`: the steered lane's pad bodies, pad centres 0.02 m up);
/// with `routes` the zones know each cart's route from time `t`.
pub fn zones(pad_bodies: &[u32], moving: bool, routes: bool, t: f32) -> Vec<ZoneDesc> {
    PADS.iter()
        .enumerate()
        .map(|(k, p)| ZoneDesc {
            position: [LANES[0] + p[0], RADIUS, p[1]],
            anchor_body: moving.then(|| pad_bodies[k]),
            offset: if moving {
                [0.0, RADIUS - 0.02, 0.0]
            } else {
                [0.0; 3]
            },
            // A flat disc at resting height: an aim above it would carry
            // the meteor on past the aim before it comes down.
            shape: ZoneShape::Ring {
                inner: 0.0,
                outer: 0.6,
                height: 0.0,
            },
            tags: 1 << k,
            motion: if moving && routes {
                ZoneMotion::Path {
                    keys: route(0, k as u32, t),
                }
            } else {
                ZoneMotion::Velocity
            },
            ..ZoneDesc::default()
        })
        .map(|mut z| {
            if z.anchor_body.is_some() {
                z.position = [0.0; 3];
            }
            z
        })
        .collect()
}

fn hash(mut x: u32) -> f32 {
    x = x.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
    ((x >> 22) ^ x) as f32 / u32::MAX as f32
}

/// Volley `k`, launched at time `t`: one meteor per pad per lane. The two
/// lanes get the same arcs (shifted by the lane spacing); each lands MISS off
/// where its pad is when it comes down, unsteered.
pub fn volley(k: u32, gravity_y: f32, t: f32, moving: bool) -> Vec<Launch> {
    let mut out = Vec::new();
    for (pi, pad) in PADS.iter().enumerate() {
        let s = k * 7 + pi as u32;
        let ang = std::f32::consts::TAU * hash(s * 2 + 1);
        let miss = MISS.0 + (MISS.1 - MISS.0) * hash(s * 2 + 2);
        let at = pad_x(0, pi as u32, t + FLIGHT, moving) - LANES[0];
        let land = [at + miss * ang.cos(), RADIUS, pad[1] + miss * ang.sin()];
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

/// Whether a meteor coming down at `p` at time `t` is on pad `pad` of `lane`.
pub fn on_pad(lane: u32, pad: u32, p: [f32; 3], t: f32, moving: bool) -> bool {
    let x = pad_x(lane, pad, t, moving);
    (p[0] - x).abs() <= PAD_HALF && (p[2] - PADS[pad as usize][1]).abs() <= PAD_HALF
}

#[wasm_bindgen]
pub struct MeteorRange;

#[wasm_bindgen]
impl MeteorRange {
    /// `{pads, radius, mass}` JSON.
    pub fn range() -> String {
        serde_json::json!({ "pads": pads(), "radius": RADIUS, "mass": MASS }).to_string()
    }

    /// The steered lane's zones (`ZoneDesc` JSON); `pad_bodies` are that
    /// lane's pad bodies (carried zones when `moving`), with each cart's
    /// route from time `t` when `routes`.
    pub fn zones(pad_bodies: &[u32], moving: bool, routes: bool, t: f32) -> String {
        serde_json::to_string(&zones(pad_bodies, moving, routes, t)).unwrap()
    }

    /// Volley `k` launched at time `t` (`Launch` JSON array).
    pub fn volley(k: u32, gravity_y: f32, t: f32, moving: bool) -> String {
        serde_json::to_string(&volley(k, gravity_y, t, moving)).unwrap()
    }

    /// Pad centre x at time `t`.
    pub fn pad_x(lane: u32, pad: u32, t: f32, moving: bool) -> f32 {
        pad_x(lane, pad, t, moving)
    }

    /// Whether a meteor of `lane` aimed at `pad` came down on it at (x, y, z)
    /// at time `t`.
    pub fn on_pad(lane: u32, pad: u32, x: f32, y: f32, z: f32, t: f32, moving: bool) -> bool {
        on_pad(lane, pad, [x, y, z], t, moving)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsteered_meteors_miss_their_pads_and_lanes_match() {
        let g = -10.0;
        for (k, moving) in (0..40).map(|k| (k, k % 2 == 1)) {
            let t = k as f32 * 2.0;
            let v = volley(k, g, t, moving);
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
                assert!(!on_pad(b.lane, b.pad, land, t + FLIGHT, moving), "{land:?}");
            }
        }
        for (moving, routes) in [(false, false), (true, false), (true, true)] {
            for t in [0.0, 3.7, 11.0] {
                for z in zones(&[1, 2, 3], moving, routes, t) {
                    z.to_gpu().unwrap();
                }
            }
        }
        // A route's straight segments follow the cart exactly.
        for pad in 0..3 {
            for t0 in [0.0, 2.5, 9.1] {
                let keys = route(0, pad, t0);
                for k in 0..=80 {
                    let t = t0 + ROUTE_AHEAD * k as f32 / 80.0;
                    let s = keys.windows(2).find(|w| t <= w[1].1).unwrap();
                    let u = (t - s[0].1) / (s[1].1 - s[0].1);
                    let x = s[0].0[0] + u * (s[1].0[0] - s[0].0[0]);
                    assert!((x - pad_x(0, pad, t, true)).abs() < 1e-3, "pad {pad} t {t}");
                }
            }
        }
        // Carts slide at CART_SPEED within their swing.
        let dt = 1.0 / 60.0;
        for k in 0..600 {
            let t = k as f32 * dt;
            let (a, b) = (pad_x(0, 1, t, true), pad_x(0, 1, t + dt, true));
            assert!(((b - a) / dt).abs() <= CART_SPEED * 1.001);
            assert!((a - pad_x(0, 1, 0.0, false)).abs() <= CART_SWING + 1e-4);
        }
        assert_eq!(pads().len(), 6);
    }
}
