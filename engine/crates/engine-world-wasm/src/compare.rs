//! Destruction approaches side by side (`rubble::play::compare`) for the comparison page:
//! the catalogue, each scenario's scene pack and lab def, and a run's verdict. The page
//! places each column with the lab's own bindings (`LabRun.create`, `labPlace`); nothing
//! here decides physics.

use crate::lab::LabScene;
use crate::{err, res, WebWorld};
use rubble::play::compare::{queue_script, Approach, Scenario, Tracker};
use wasm_bindgen::prelude::*;

fn js(v: &serde_json::Value) -> Result<JsValue, JsValue> {
    serde::Serialize::serialize(v, &serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
}

fn scenario(name: &str) -> Result<Scenario, JsValue> {
    Scenario::parse(name).ok_or_else(|| err(format!("unknown scenario {name}")))
}

fn approach(name: &str) -> Result<Approach, JsValue> {
    Approach::parse(name).ok_or_else(|| err(format!("unknown approach {name}")))
}

/// `{ approaches: [{ name, title, blurb }], scenarios: [{ name, title, caption, seconds,
/// physicsHz }] }`.
#[wasm_bindgen(js_name = compareCatalog)]
pub fn compare_catalog() -> Result<JsValue, JsValue> {
    js(&serde_json::json!({
        "approaches": Approach::ALL.iter().map(|a| serde_json::json!({ "name": a.name(), "title": a.title(), "blurb": a.blurb() })).collect::<Vec<_>>(),
        "scenarios": Scenario::ALL.iter().map(|s| serde_json::json!({
            "name": s.name(), "title": s.title(), "caption": s.caption(), "seconds": s.seconds(), "physicsHz": s.physics_hz(),
        })).collect::<Vec<_>>(),
    }))
}

/// The scenario's scene pack (its shots authored in it).
#[wasm_bindgen(js_name = comparePack)]
pub fn compare_pack(name: &str) -> Result<Vec<u8>, JsValue> {
    Ok(scenario(name)?.pack())
}

/// The lab def (`LabRun.create`'s) of `scenario` under `approach`.
#[wasm_bindgen(js_name = compareDef)]
pub fn compare_def(scenario_name: &str, approach_name: &str) -> Result<JsValue, JsValue> {
    let s = approach(approach_name)?.settings(scenario(scenario_name)?.physics_hz());
    js(&serde_json::json!({
        "source": "pack",
        "bonds": s.bonds.name(),
        "physicsHz": s.physics_hz,
        "ccd": s.ccd,
        "contactFracture": s.fracture.contact_fracture,
        "strikePrediction": s.fracture.strike_prediction,
        "strike": s.fracture.strike.map(|r| r.name()),
        "strikeRadius": match s.fracture.strike {
            Some(rubble::StrikeRule::Radial { radius_mm }) => Some(radius_mm as f64 * 1e-3),
            _ => None,
        },
        "impactResponse": s.fracture.impact_response.map(|r| r.pulse.name()),
        "impactScale": s.fracture.impact_response.map(|r| r.scale_pct as f64 * 0.01),
        "supportLoss": s.support_loss,
        "supportLossDynamicFactor": s.support_loss_dynamic_factor,
    }))
}

/// A run's verdict tracker (`compareTrack`).
#[wasm_bindgen]
pub struct CompareTracker {
    scenario: Scenario,
    tracker: Tracker,
}

#[wasm_bindgen]
impl WebWorld {
    /// A tracker for `scenario` placed as `lab` in this world.
    #[wasm_bindgen(js_name = compareTrack)]
    pub fn compare_track(&self, lab: &LabScene, scenario_name: &str) -> Result<CompareTracker, JsValue> {
        let scenario = scenario(scenario_name)?;
        Ok(CompareTracker {
            scenario,
            tracker: res(Tracker::new(scenario, &lab.lab, &self.world))?,
        })
    }

    /// Queue the scenario's script: every authored shot at its time from now.
    #[wasm_bindgen(js_name = compareQueueScript)]
    pub fn compare_queue_script(&self, lab: &mut LabScene, tracker: &CompareTracker) -> Result<(), JsValue> {
        res(queue_script(tracker.scenario, &mut lab.lab, &self.world))
    }

    /// Take in the last polled step.
    #[wasm_bindgen(js_name = compareObserve)]
    pub fn compare_observe(&self, lab: &LabScene, tracker: &mut CompareTracker) -> Result<(), JsValue> {
        res(tracker.tracker.observe(&lab.lab, &self.world))
    }

    /// `{ seconds, bonds, bondsBroken, firstBreak, drop, fell, shots: [{ label, launchSpeed, speed, along, through,
    /// broke }] }`.
    #[wasm_bindgen(js_name = compareVerdict)]
    pub fn compare_verdict(&self, lab: &LabScene, tracker: &CompareTracker) -> Result<JsValue, JsValue> {
        let v = res(tracker.tracker.verdict(&lab.lab, &self.world, false))?;
        js(&serde_json::json!({
            "seconds": v.seconds, "bonds": v.bonds, "bondsBroken": v.bonds_broken, "firstBreak": v.first_break,
            "drop": v.drop, "fell": v.fell,
            "shots": v.shots.iter().map(|s| serde_json::json!({
                "label": s.label, "launchSpeed": s.launch_speed, "speed": s.speed, "along": s.along,
                "through": s.through, "broke": s.broke,
            })).collect::<Vec<_>>(),
        }))
    }
}
