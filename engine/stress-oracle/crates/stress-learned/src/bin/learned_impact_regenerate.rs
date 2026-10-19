//! Re-record a saved event after an oracle correction, into a fresh directory.
//! Original scenes/settings/recipes are retained; changed trajectories are explicit.
use serde_json::{json, Value};
use std::{fs, path::Path, time::Instant};
use stress_learned::{
    oracle::{record_event_options, LoadBasis},
    validate_event, RecordMeta,
};

fn read(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

// Older context files passed through a JSON parser without float_roundtrip.
// Only descriptive recipe numbers may have that rounding difference; source
// scene/settings are compared exactly, and all recipe identities remain exact.
fn same_recipe(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| same_recipe(v, w)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(v, w)| same_recipe(v, w))
        }
        (Value::Number(a), Value::Number(b)) if a != b => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() <= 4.0 * f64::EPSILON * x.abs().max(y.abs()),
            _ => false,
        },
        _ => a == b,
    }
}

fn run(source: &Path, out: &Path) -> Result<(), String> {
    if out.exists() {
        return Err("output exists; regeneration requires a fresh directory".into());
    }
    let manifest = validate_event(source).map_err(|e| e.to_string())?;
    let context = read(&source.join("context.json"))?;
    if context["load_basis"] != "frame_start_cluster" {
        return Err("regeneration requires an explicit frame-start load contract".into());
    }
    let qs = context["qs_probe"]
        .as_bool()
        .ok_or("missing explicit QS probe contract")?;
    let preview = match context.get("load_preview") {
        None => false,
        Some(v) if v["format"] == "rigid-contact-preview/1" => true,
        _ => return Err("unknown preview contract".into()),
    };
    let original_scene = read(&source.join("scene.json"))?;
    if original_scene["name"] != context["source_scene"] {
        return Err("source scene identity mismatch".into());
    }
    let scene = serde_json::from_value(original_scene.clone()).map_err(|e| e.to_string())?;
    record_event_options(scene, out, RecordMeta {
        event_id: manifest.event_id,
        structure_type: manifest.structure_type,
        load_type: manifest.load_type,
        notes: format!("oracle regeneration from {}; original scene/settings retained; trajectories may differ", source.display()),
    }, manifest.n_frames, qs, LoadBasis::FrameStart, preview)?;
    if read(&out.join("scene.json"))? != original_scene {
        return Err("saved scene/settings changed during regeneration".into());
    }
    let mut new_context = read(&out.join("context.json"))?;
    if !same_recipe(&new_context["generator"], &context["generator"]) {
        return Err("source recipe identity changed during regeneration".into());
    }
    let retained_legacy_recipe_rounding = new_context["generator"] != context["generator"];
    if retained_legacy_recipe_rounding {
        new_context["generator"] = context["generator"].clone();
        fs::write(
            out.join("context.json"),
            serde_json::to_vec_pretty(&new_context).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    validate_event(out).map_err(|e| e.to_string())?;
    let provenance = json!({"format":"learned-oracle-regeneration/1", "source":source.canonicalize().map_err(|e| e.to_string())?,
        "source_scene_and_settings_identical":true, "source_recipe_identical":true,
        "retained_legacy_recipe_rounding":retained_legacy_recipe_rounding,
        "scope":"Fresh oracle run from the same scene. This does not assert original array/replay equality or physical validity; audit energy/motion and retain original split membership before training."});
    fs::write(
        out.join("regeneration.json"),
        serde_json::to_vec_pretty(&provenance).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_recipe_numeric_roundoff_is_compatible() {
        let old = json!({"catalog":"forcing-v8","point":[0.99,0.0]});
        assert!(same_recipe(
            &old,
            &json!({"catalog":"forcing-v8","point":[0.9900000000000001,0.0]})
        ));
        assert!(!same_recipe(
            &old,
            &json!({"catalog":"forcing-v8","point":[0.991,0.0]})
        ));
        assert!(!same_recipe(
            &old,
            &json!({"catalog":"other","point":[0.99,0.0]})
        ));
        assert!(!same_recipe(
            &old,
            &json!({"catalog":"forcing-v8","point":[0.99]})
        ));
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: learned_impact_regenerate SOURCE_EVENT FRESH_OUTPUT");
        std::process::exit(2);
    }
    let start = Instant::now();
    match run(Path::new(&args[0]), Path::new(&args[1])) {
        Ok(()) => println!(
            "{}",
            json!({"status":"completed","seconds":start.elapsed().as_secs_f64(),"output":args[1]})
        ),
        Err(error) => {
            eprintln!("regeneration failed: {error}");
            std::process::exit(1);
        }
    }
}
