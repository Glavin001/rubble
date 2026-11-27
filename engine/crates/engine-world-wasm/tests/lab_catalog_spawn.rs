//! The crash lab's scenes and presets moved from the page's tables to Rust's catalogue
//! (`lab_scenes.rs`) without changing what they spawn: for every preset and every crash
//! scene at its defaults, the spec the page builds from the catalogue (every parameter sent,
//! `runUp` 0 meaning the default run-up) spawns the fleet, word for word, that the spec the
//! page built from its own table before (the values below, copied from that table: no
//! `runUp` unless the preset set one) spawned. The car: the procedural sedan.

use engine_world_wasm::lab_scenes::{presets, scenes};
use engine_world_wasm::{explicit_sedan_truck, ExplicitSim};
use serde_json::{json, Map, Value};

/// The page's table before (softbody.html `XPARAMS`, 2026-11-27): each scene's knobs and
/// defaults, km/h ones marked.
fn old_table(scene: &str) -> Vec<(&'static str, f64, bool)> {
    match scene {
        "wall" => vec![("speed", 100.0, true), ("overlap", 1.0, false), ("angle", 0.0, false)],
        "headon" => vec![("speed", 56.0, true), ("speed2", 56.0, true), ("overlap", 1.0, false), ("angle", 0.0, false)],
        "tbone" => vec![("speed", 50.0, true), ("carSpeed", 0.0, true), ("at", 0.0, false), ("angle", 90.0, false)],
        "pole" => vec![("speed", 60.0, true), ("offset", 0.0, false), ("angle", 0.0, false)],
        "sidepole" => vec![("speed", 32.0, true), ("at", 0.0, false), ("angle", 90.0, false)],
        "limbo" => vec![("speed", 60.0, true), ("bar", 1.0, false)],
        "drop" => vec![("height", 8.0, false), ("speed", 0.0, true)],
        "mdb214" => vec![("speed", 53.9, true), ("crab", 27.0, false)],
        "doorpull" => vec![("pullSpeed", 0.5, false), ("doorNodes", 6.0, false), ("pullStiffness", 100000.0, false)],
        _ => vec![],
    }
}

fn base(scene: &str) -> Map<String, Value> {
    let v = json!({ "vehicles": [{ "truck": explicit_sedan_truck(true), "at": [0.0, 0.2], "front": true, "yaw": 0.0 }], "scene": scene, "speed": 100.0 / 3.6, "drag": true, "surfaceContact": false });
    v.as_object().unwrap().clone()
}

/// The spec the page built before: its table's knobs, the preset's values over the defaults,
/// `runUp` only when the preset gave one.
fn old_spec(scene: &str, preset: &[(String, f64)]) -> Value {
    let mut s = base(scene);
    for (k, d, kmh) in old_table(scene) {
        let v = preset.iter().find(|(p, _)| p == k).map_or(d, |(_, x)| *x);
        s.insert(k.into(), json!(if kmh { v / 3.6 } else { v }));
    }
    if let Some((_, r)) = preset.iter().find(|(k, _)| k == "runUp") {
        s.insert("runUp".into(), json!(r));
    }
    Value::Object(s)
}

/// The spec the page builds now: every parameter of the catalogue's scene (km/h ones in m/s).
fn new_spec(scene: &str, preset: &[(String, f64)]) -> Value {
    let mut s = base(scene);
    let sc = scenes().into_iter().find(|x| x.id == scene).unwrap();
    for p in &sc.params {
        let d = match p.kind {
            engine_softbody::crash_reference::family::ParamKind::Number { default, .. } => default,
            _ => unreachable!(),
        };
        let v = preset.iter().find(|(k, _)| *k == p.key).map_or(d, |(_, x)| *x);
        s.insert(p.key.clone(), json!(if p.unit == "km/h" { v / 3.6 } else { v }));
    }
    Value::Object(s)
}

fn fleet(spec: &Value) -> String {
    format!("{:?}", ExplicitSim::spawn_cpu(spec).unwrap())
}

#[test]
fn catalogue_specs_spawn_the_old_fleets() {
    let mut cases: Vec<(String, String, Vec<(String, f64)>)> = presets()
        .into_iter()
        .map(|p| (p.id.to_string(), p.scene.to_string(), p.values.0.iter().map(|(k, v)| (k.clone(), match v { engine_softbody::crash_reference::family::Value::Num(x) => *x, _ => 0.0 })).collect()))
        .collect();
    for s in ["wall", "headon", "tbone", "pole", "sidepole", "limbo", "drop", "mdb214", "doorpull"] {
        cases.push((format!("scene {s}"), s.to_string(), Vec::new()));
    }
    for (name, scene, values) in cases {
        let (a, b) = (fleet(&old_spec(&scene, &values)), fleet(&new_spec(&scene, &values)));
        assert!(a == b, "{name}: the catalogue's spec spawns another fleet");
        println!("{name}: same fleet ({} bytes of state)", a.len());
    }
}
