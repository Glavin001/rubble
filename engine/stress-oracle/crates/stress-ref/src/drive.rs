//! Driven vehicles: raycast vehicles (rubble's `engine-vehicle` presets) placed beside a
//! structure and driven by their wheels, not bonded islands set moving (`launch`). A host
//! that steps the structure in rubble's world API (`World`, the `Threshold` model) creates
//! each one as a vehicle and gives it its throttle from `delay` on; its chassis colliders
//! strike the structure's chunks as a shot does (strike prediction), so it breaks in
//! instead of stopping dead.
//!
//! A pack carries them as `defaults.drives` ([`write`], read with [`drives_of`]), in the
//! packs' frame (Y-up, m):
//!
//! ```json
//! {"name": "sedan", "vehicle": "sedan", "at": {"x": 0.55, "y": 1.25, "z": 33.5},
//!  "forward": {"x": 0, "y": 0, "z": -1}, "throttle": 1, "delay": 0}
//! ```
//!
//! * `vehicle` names an `engine-vehicle` preset (`sedan`, `six_wheel_truck`, ...).
//! * `at` is the vehicle's design origin and `forward` the way its nose points (level).
//! * `throttle` (0..1) is held from `delay` (s after the scene starts) on; until then the
//!   vehicle stands braked.

use serde_json::{json, Value};

/// A driven vehicle (Z-up, the stress scene's frame).
#[derive(Clone, Debug, PartialEq)]
pub struct Drive {
    pub name: String,
    pub vehicle: String,
    pub at: [f64; 3],
    /// Unit, level.
    pub forward: [f64; 3],
    pub throttle: f64,
    pub delay: f64,
}

fn y_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], v[2], -v[1]]
}

fn z_up(v: [f64; 3]) -> [f64; 3] {
    [v[0], -v[2], v[1]]
}

fn xyz(v: [f64; 3]) -> Value {
    let p = y_up(v);
    json!({"x": p[0], "y": p[1], "z": p[2]})
}

fn read(v: &Value) -> Option<[f64; 3]> {
    Some(z_up([v["x"].as_f64()?, v["y"].as_f64()?, v["z"].as_f64()?]))
}

/// Write `drives` into a pack as `defaults.drives` (none: the key is left out).
pub fn write(pack: &mut Value, drives: &[Drive]) {
    if drives.is_empty() {
        return;
    }
    pack["defaults"]["drives"] = drives
        .iter()
        .map(|d| json!({"name": d.name, "vehicle": d.vehicle, "at": xyz(d.at), "forward": xyz(d.forward), "throttle": d.throttle, "delay": d.delay}))
        .collect();
}

/// The drives a pack carries (`defaults.drives`), in the stress scene's frame (Z-up), in the
/// pack's order; empty for a pack without any. An error names a drive without a name, a
/// vehicle, a position or a level forward direction, or a throttle outside 0..1.
pub fn drives_of(pack: &Value) -> Result<Vec<Drive>, String> {
    let Some(list) = pack["defaults"]["drives"].as_array() else { return Ok(Vec::new()) };
    list.iter()
        .map(|d| {
            let name = d["name"].as_str().ok_or("a drive without a name")?.to_string();
            let vehicle = d["vehicle"].as_str().ok_or_else(|| format!("drive {name}: no vehicle"))?.to_string();
            let at = read(&d["at"]).ok_or_else(|| format!("drive {name}: no position (at)"))?;
            let f = read(&d["forward"]).ok_or_else(|| format!("drive {name}: no forward direction"))?;
            let len = (f[0] * f[0] + f[1] * f[1]).sqrt();
            if len < 1e-9 || f[2].abs() > 1e-6 * len.max(1.0) {
                return Err(format!("drive {name}: forward {f:?} is not level"));
            }
            let throttle = d["throttle"].as_f64().unwrap_or(1.0);
            if !(0.0..=1.0).contains(&throttle) {
                return Err(format!("drive {name}: throttle {throttle} outside 0..1"));
            }
            Ok(Drive { name, vehicle, at, forward: [f[0] / len, f[1] / len, 0.0], throttle, delay: d["delay"].as_f64().unwrap_or(0.0) })
        })
        .collect()
}
