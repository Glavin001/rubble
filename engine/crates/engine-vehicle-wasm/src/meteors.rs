//! wasm-bindgen face of the "Meteor range" (`engine_vehicle::scripts::meteors`):
//! the page (vehicle.html `?scene=meteors`) builds the pads from `range`,
//! launches `volley` and uploads `zones`.

use wasm_bindgen::prelude::*;

pub use engine_vehicle::scripts::meteors::*;

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
