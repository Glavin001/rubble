//! The crash lab's own scenes and presets (`softbody.html?solver=explicit`), as data the page
//! reads (`crashLabCatalog`): each scene of `ExplicitSim`'s spawn (`explicit.rs`
//! `crash_scene`, `arena`) with its parameters, declared with the same [`Param`] as the
//! crash reference families, and each one-click preset as a scene and values. The page has
//! no second definition of either.
//!
//! These scenes spawn whatever vehicle the page loads (a `.truck`, a JBeam folder, the
//! showcase sedan, a converted mod), placed by its collision skin, its front face at the
//! spawn point, with a run-up of 1 m plus half a second of travel; the reference families
//! (`crash_reference::families`) place registered generated designs by their extent with a
//! run-up in seconds. A scene names the family it stages (`family`), so the picker files them
//! together; its geometry stays its own (the lab's recordings and checks are made on it).

use engine_softbody::crash_reference::family::{num, Param, Value, Values};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// One of the lab's scenes: its id (`?xscene=`), title, the reference family it stages, its
/// parameters (their keys are the URL's and the spawn spec's; km/h ones go to the solver in
/// m/s), whether it is a crash (slow motion at impact) or a drive, and its pill (`{key}` the
/// value now).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LabScene {
    pub id: &'static str,
    pub title: &'static str,
    pub family: Option<&'static str>,
    pub params: Vec<Param>,
    pub crash: bool,
    pub drive: bool,
    pub pill: &'static str,
}

/// A one-click preset (`?preset=`): a scene and some of its values, its pills and camera.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LabPreset {
    pub id: &'static str,
    pub label: &'static str,
    pub scene: &'static str,
    pub values: Values,
    pub pills: Vec<&'static str>,
    pub cam: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Catalog {
    pub scenes: Vec<LabScene>,
    pub presets: Vec<LabPreset>,
}

fn kmh(key: &str, label: &str, default: f64) -> Param {
    num(key, label, "km/h", default, 0.0, 200.0, 1.0)
}
fn angle(key: &str, label: &str, default: f64, lo: f64, hi: f64) -> Param {
    num(key, label, "°", default, lo, hi, 1.0)
}
fn run_up() -> Param {
    num("runUp", "run-up", "m", 0.0, 0.0, 60.0, 0.1).help("the gap to what it hits; 0: 1 m plus half a second of travel")
}
fn along() -> Param {
    num("at", "hit at", "m", 0.0, -2.5, 2.5, 0.05).help("along the struck side from its centre (forward +)")
}
fn overlap() -> Param {
    num("overlap", "overlap", "of width", 1.0, 0.05, 1.0, 0.05)
}

fn scene(id: &'static str, title: &'static str, family: Option<&'static str>, params: Vec<Param>, pill: &'static str) -> LabScene {
    LabScene { id, title, family, params, crash: !matches!(id, "idle" | "coast" | "arena" | "yard" | "traffic"), drive: matches!(id, "yard" | "traffic"), pill }
}

/// Every scene, in the picker's order.
pub fn scenes() -> Vec<LabScene> {
    vec![
        scene("idle", "At rest (L launches)", Some("coast"), vec![], "AT REST"),
        scene("coast", "Coasting", Some("coast"), vec![kmh("xspeed", "speed", 100.0)], "COASTING · {xspeed} KM/H"),
        scene("wall", "Into a rigid wall", Some("block"), vec![kmh("speed", "speed", 100.0), overlap(), angle("angle", "angle", 0.0, -90.0, 90.0), run_up()], "WALL · {speed} KM/H"),
        scene("headon", "Head-on, two cars", Some("head_on"), vec![kmh("speed", "car 1", 56.0), kmh("speed2", "car 2", 56.0), overlap(), angle("angle", "car 2 angle", 0.0, -90.0, 90.0)], "HEAD-ON · {speed} + {speed2} KM/H"),
        scene("tbone", "T-bone into a car", Some("tbone"), vec![kmh("speed", "striking", 50.0), kmh("carSpeed", "struck", 0.0), along(), angle("angle", "angle", 90.0, 0.0, 180.0), run_up()], "T-BONE · {speed} KM/H"),
        scene("pole", "Into a 20 cm pole", Some("pole"), vec![kmh("speed", "speed", 60.0), num("offset", "offset", "m", 0.0, -1.5, 1.5, 0.01), angle("angle", "angle", 0.0, -90.0, 90.0), run_up()], "POLE · {speed} KM/H"),
        scene("sidepole", "Sideways into a 20 cm pole", Some("side_pole"), vec![kmh("speed", "speed", 32.0), along(), angle("angle", "travel angle", 90.0, 0.0, 180.0), run_up()], "SIDE POLE · {speed} KM/H · {angle}°"),
        scene("limbo", "Under a low bar", Some("low_bar"), vec![kmh("speed", "speed", 60.0), num("bar", "bar height", "m", 1.0, 0.3, 2.0, 0.05), run_up()], "LOW BAR · {speed} KM/H"),
        scene("drop", "Drop test", Some("drop"), vec![num("height", "height", "m", 8.0, 0.5, 40.0, 0.5), kmh("speed", "forward", 0.0)], "DROPPED FROM {height} M"),
        scene("arena", "Fleet arena (many vehicles)", None, vec![num("n", "vehicles", "", 256.0, 1.0, 4096.0, 1.0), num("headOn", "head-on share", "", 0.0, 0.0, 1.0, 0.05), kmh("speedMin", "min", 90.0), kmh("speedMax", "max", 150.0)], "FLEET ARENA"),
        // FMVSS 214 S7: 33.5 mph (53.9 km/h), the barrier crabbed 27°.
        scene("mdb214", "FMVSS 214 moving barrier", Some("side_barrier"), vec![kmh("speed", "barrier", 53.9), angle("crab", "crab", 27.0, 0.0, 45.0), run_up()], "FMVSS 214 MDB · {speed} KM/H · {crab}° CRAB"),
        scene("doorpull", "Door pull", None, vec![num("pullSpeed", "pull", "m/s", 0.5, 0.05, 3.0, 0.05), num("doorNodes", "door nodes", "", 6.0, 1.0, 24.0, 1.0), num("pullStiffness", "pull", "N/m", 100000.0, 10000.0, 1000000.0, 10000.0)], "DOOR PULL · {pullSpeed} M/S"),
        scene("yard", "Drive yard (WASD, raycast wheels)", None, vec![], "DRIVE YARD · WASD · SPACE HANDBRAKE"),
        scene("traffic", "Traffic derby (self-driving, raycast wheels)", None, vec![num("cars", "cars", "", 20.0, 1.0, 64.0, 1.0)], "TRAFFIC DERBY · {cars} CARS"),
    ]
}

fn preset(id: &'static str, label: &'static str, scene: &'static str, values: &[(&str, f64)], pills: &[&'static str], cam: &'static str) -> LabPreset {
    LabPreset { id, label, scene, values: Values(values.iter().map(|(k, v)| (k.to_string(), Value::Num(*v))).collect()), pills: pills.to_vec(), cam }
}

/// Every preset, in the picker's order (their ids are the recorders' and checks').
pub fn presets() -> Vec<LabPreset> {
    vec![
        preset("wall56", "Wall · 56 km/h", "wall", &[("speed", 56.0)], &["WALL · 56 KM/H"], "three"),
        // An oblique side pole: sliding at 75° to the car's axis into a pole at the door,
        // 0.8 m away (a car sliding on its tyres would stop short of the usual run-up).
        preset("pole29", "Pole · 29 km/h", "sidepole", &[("speed", 29.0), ("angle", 75.0), ("at", 0.3), ("runUp", 0.8)], &["POLE · 29 KM/H"], "threeFront"),
        preset("headon50", "Head-on · 2 x 50 km/h", "headon", &[("speed", 50.0), ("speed2", 50.0)], &["HEAD-ON · 2 x 50 KM/H"], "side"),
        preset("drop15", "Dropped from 15 m", "drop", &[("height", 15.0)], &["DROPPED FROM 15 M"], "side"),
        // engine/docs/beamng-realism-plan.md P10: FMVSS 214's barrier stood in for by a second
        // car (1,450 kg: within 6 %) at 54 km/h, its heading 27° off square, into the front door.
        preset("tbone214", "Side · FMVSS 214-like · 54 km/h", "tbone", &[("speed", 54.0), ("at", 0.55), ("angle", 63.0)], &["SIDE · 54 KM/H · 27° CRAB", "FMVSS 214-LIKE (CAR STRIKER)"], "three"),
        // Car to car at 80 km/h, 90°, at the B-pillar: the severe T-bone in which a door may
        // come off (when its hinges and latch pass their strength).
        preset("tbone80", "T-bone · car to car · 80 km/h", "tbone", &[("speed", 80.0), ("at", 0.15)], &["T-BONE · 80 KM/H"], "three"),
        // IIHS small overlap: 64 km/h, 25 % of the width into a rigid barrier.
        preset("overlap64", "Small overlap · 25 % · 64 km/h", "wall", &[("speed", 64.0), ("overlap", 0.25)], &["SMALL OVERLAP · 25 % · 64 KM/H"], "threeFront"),
        // A post glancing down the side at 50 km/h: the front corner, door and mirror take it
        // (12 cm of the width overlaps the post).
        preset("swipe50", "Post sideswipe · 50 km/h", "pole", &[("speed", 50.0), ("offset", 0.82)], &["POST SIDESWIPE · 50 KM/H"], "threeFront"),
        // FMVSS 214 S7 with the standard's barrier (crash_rigs.rs).
        preset("mdb214", "Side · FMVSS 214 MDB · 54 km/h", "mdb214", &[("speed", 53.9), ("crab", 27.0)], &["FMVSS 214 · MDB · 54 KM/H", "1,368 KG · 27° CRAB"], "three"),
        // The front door's latch edge pulled out through six skin nodes by one rig.
        preset("doorpull", "Door pull · 0.5 m/s", "doorpull", &[("pullSpeed", 0.5), ("doorNodes", 6.0)], &["DOOR PULL · 0.5 M/S"], "threeFront"),
    ]
}

/// The catalogue as the page reads it (JSON text: `{ scenes, presets }`).
#[wasm_bindgen(js_name = crashLabCatalog)]
pub fn crash_lab_catalog() -> String {
    serde_json::to_string(&Catalog { scenes: scenes(), presets: presets() }).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset names a scene and only that scene's parameters, within their ranges; ids
    /// are unique; every family named exists.
    #[test]
    fn presets_are_values_of_their_scenes() {
        let scenes = scenes();
        for p in presets() {
            let s = scenes.iter().find(|s| s.id == p.scene).unwrap_or_else(|| panic!("{}: scene {}", p.id, p.scene));
            for (k, v) in &p.values.0 {
                let par = s.params.iter().find(|x| &x.key == k).unwrap_or_else(|| panic!("{}: {k} is not a parameter of {}", p.id, s.id));
                par.check(v).unwrap_or_else(|e| panic!("{}: {e}", p.id));
            }
        }
        let mut ids: Vec<&str> = presets().iter().map(|p| p.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), presets().len());
        for s in &scenes {
            if let Some(f) = s.family {
                assert!(engine_softbody::crash_reference::family::get(f).is_some(), "{}: family {f}", s.id);
            }
        }
        assert_eq!(scenes.len(), 14);
    }
}
