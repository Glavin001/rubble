//! wasm-bindgen face of the "Gauntlet" (`engine_vehicle::scripts::gauntlet`):
//! the page (vehicle.html `?scene=gauntlet`) builds the bodies from `layout`,
//! launches `blast`'s blocks and uploads `zones`.

use engine_schema::guidance::GuidanceSettings;
use wasm_bindgen::prelude::*;

use engine_vehicle::scripts::gauntlet as script;
pub use engine_vehicle::scripts::gauntlet::*;

/// The street's shelling (`engine_vehicle::scripts::gauntlet::Gauntlet`).
#[wasm_bindgen]
pub struct Gauntlet {
    inner: script::Gauntlet,
}

#[wasm_bindgen]
impl Gauntlet {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Gauntlet {
        Gauntlet {
            inner: script::Gauntlet::new(),
        }
    }

    /// The street (`Layout` JSON).
    pub fn layout(&self) -> String {
        serde_json::to_string(self.inner.street()).unwrap()
    }

    /// Back to an unshelled street (scene reset).
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// The next blast ahead of a car at (x, z) moving at `vz` m/s along the
    /// street: `[cx, cy, cz, n, (block, vx, vy, vz, wx, wy, wz) × n]`.
    pub fn blast(&mut self, x: f32, z: f32, vz: f32) -> Vec<f32> {
        let (c, list) = self.inner.blast_at([x, z], vz);
        let mut out = vec![c[0], c[1], c[2], list.len() as f32];
        for (i, v, w) in list {
            out.push(i as f32);
            out.extend_from_slice(&v);
            out.extend_from_slice(&w);
        }
        out
    }

    /// `zones` as `ZoneDesc` JSON.
    pub fn zones(&self, player: u32, chassis_y: f32, parked: &[u32]) -> String {
        serde_json::to_string(&zones(player, chassis_y, parked)).unwrap()
    }

    /// `settings(strength)` as `GuidanceSettings` JSON, `enabled` as given.
    pub fn settings(strength: u32, enabled: bool) -> String {
        serde_json::to_string(&GuidanceSettings {
            enabled,
            ..settings(strength)
        })
        .unwrap()
    }
}

impl Default for Gauntlet {
    fn default() -> Self {
        Self::new()
    }
}
