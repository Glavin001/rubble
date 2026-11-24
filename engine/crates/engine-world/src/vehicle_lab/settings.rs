//! The lab's engine settings as JSON (a page's controls, a harness's options): vehicle
//! sleep, debris' time to live, and the fast schedules' switches (surface contact on the
//! fast schedules, vehicle contact's wide frames: what `RUBBLE_SURFACE_FAST` and
//! `RUBBLE_VV_WIDE` say natively, `explicit_plan::schedule::host_setting`).

use super::{num, Lab, LabOptions, LAB_DEBRIS_TTL};
use crate::World;
use engine_softbody::explicit_debris::DebrisTtl;
use engine_softbody::explicit_sleep::VehicleSleep;
use serde_json::{json, Value};

/// `{ on, speed, reachMm, dwell }` (missing: `d`'s) as the vehicle sleep setting: on or
/// off, and the thresholds (speed m/s, reach mm, dwell in 0.5 ms steps).
pub fn vehicle_sleep_from(x: &Value, d: (bool, VehicleSleep)) -> (bool, VehicleSleep) {
    let on = match x {
        Value::Bool(b) => *b,
        _ => x.get("on").and_then(Value::as_bool).unwrap_or(d.0),
    };
    let cfg = VehicleSleep {
        speed: (num(x, "speed", f64::from(d.1.speed)) as f32).max(0.0),
        reach: (num(x, "reachMm", f64::from(d.1.reach) * 1e3) as f32 * 1e-3).max(0.0),
        dwell: num(x, "dwell", f64::from(d.1.dwell)).clamp(1.0, 1.0e9) as u32,
    };
    (on, cfg)
}

pub(super) fn vehicle_sleep_json(on: bool, c: VehicleSleep) -> Value {
    json!({ "on": on, "speed": c.speed, "reachMm": f64::from(c.reach) * 1e3, "dwell": c.dwell })
}

/// `{ on, seconds, maxKg, maxNodes }` (missing: the lab's default) as a time to live.
pub fn debris_ttl_from(x: &Value) -> Option<DebrisTtl> {
    let d = LAB_DEBRIS_TTL;
    x.get("on").and_then(Value::as_bool).unwrap_or(true).then(|| DebrisTtl {
        max_mass_kg: (num(x, "maxKg", f64::from(d.max_mass_kg)) as f32).max(0.0),
        max_nodes: (num(x, "maxNodes", f64::from(d.max_nodes)) as u32).max(1),
        seconds: (num(x, "seconds", f64::from(d.seconds)) as f32).max(0.0),
    })
}

pub(super) fn debris_json(t: Option<DebrisTtl>) -> Value {
    match t {
        Some(t) => json!({ "on": true, "seconds": t.seconds, "maxKg": t.max_mass_kg, "maxNodes": t.max_nodes }),
        None => json!({ "on": false }),
    }
}

/// `default` (or null) as `None`, `off`/`0` as 0, `wide`/`1` as 1, `all`/`2` as 2: the
/// lab's `surfaceFast`.
pub(super) fn surface_fast_from(x: Option<&Value>) -> Option<u8> {
    match x? {
        Value::String(t) => match t.as_str() {
            "off" | "0" => Some(0),
            "wide" | "1" => Some(1),
            "all" | "on" | "2" => Some(2),
            _ => None,
        },
        Value::Number(n) => n.as_u64().map(|n| n.min(2) as u8),
        Value::Bool(b) => Some(if *b { 2 } else { 0 }),
        _ => None,
    }
}

/// `default` (or null) as `None`, `on`/`off` (or a bool): the lab's `vvWide`.
pub(super) fn vv_wide_from(x: Option<&Value>) -> Option<bool> {
    match x? {
        Value::String(t) => match t.as_str() {
            "on" | "1" => Some(true),
            "off" | "0" => Some(false),
            _ => None,
        },
        Value::Bool(b) => Some(*b),
        _ => None,
    }
}

impl LabOptions {
    /// The schedule settings ([`Self::surface_fast`], [`Self::vv_wide`]) for this process
    /// (`engine_exec::explicit_plan::schedule::host_setting`): before a browser lists
    /// the kernels to compile and before the World is built.
    pub fn apply_schedule_settings(&self) {
        use engine_exec::explicit_plan::schedule::host_setting;
        host_setting::set_surface_fast(self.surface_fast);
        host_setting::set_vv_wide(self.vv_wide);
    }
}

impl Lab {
    /// `{ surfaceFast, vvWide }`: the schedule switches ([`LabOptions::surface_fast`],
    /// `vv_wide`; `default`, `off`, `wide`, `all` / `default`, `on`, `off`; the same
    /// results bit for bit), from the next frame, kept for the process
    /// (`schedule::host_setting`) so a rebuild keeps them. `rebuild`: a browser host did
    /// not compile the modules a switch now on takes (the page builds again). `None`:
    /// not this command.
    pub(super) fn schedule_command(&mut self, world: &mut World, cmd: &Value) -> Option<anyhow::Result<Value>> {
        if cmd.get("surfaceFast").is_none() && cmd.get("vvWide").is_none() {
            return None;
        }
        if let Some(x) = cmd.get("surfaceFast") {
            self.bench.opts.surface_fast = surface_fast_from(Some(x));
        }
        if let Some(x) = cmd.get("vvWide") {
            self.bench.opts.vv_wide = vv_wide_from(Some(x));
        }
        self.bench.opts.apply_schedule_settings();
        let mut rebuild = false;
        if let Some(x) = world.explicit_vehicles_mut() {
            x.reset_schedule_switches();
            rebuild = !x.schedule_modules_listed();
        }
        Some(Ok(json!({ "surfaceFast": self.bench.opts.surface_fast, "vvWide": self.bench.opts.vv_wide, "rebuild": rebuild })))
    }
}
