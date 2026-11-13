//! Cars of the comparison manifest through the same three frontal crashes
//! on the CPU reference, side by side: the full wall (56 km/h), the 40 %
//! offset barrier (64 km/h) and the centre pole (50 km/h).
//!
//! ```bash
//! RUBBLE_COMPARE_MODS=~/Development/rubble-artifacts/sedan-live/mods SURFACE_CONTACT=1 \
//!   cargo run --release -p engine-softbody --example frontal_compare -- \
//!   --cars sedan_new,etkk [--out DIR] [--scenes wall,offset,pole] [--secs 0.6]
//! ```
//! A name that is not in the manifest is taken as a mod folder holding a
//! vehicle `sedan` (`<name>/sedan/sedan.pc`: a generated car's export) or, with
//! `name=mod:vehicle:pc`, any mod. Prints one markdown table per scene and
//! writes `<out>/<car>.<scene>.json` (everything `vehicle_compare::scene_front`
//! measures). `engine/docs/crash-reference-targets.md` has the real cars' bands
//! for the full wall; there is none for the offset or the centre pole.
use engine_softbody::vehicle_compare::{
    default_manifest, scene_front, Car, CarSpec, Front, Manifest,
};
use serde_json::Value;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1).cloned())
    };
    let cars = opt("--cars").unwrap_or_else(|| "sedan_new,etkk".into());
    let scenes = opt("--scenes").unwrap_or_else(|| "wall,offset,pole".into());
    let secs: f32 = opt("--secs").and_then(|x| x.parse().ok()).unwrap_or(0.6);
    let out = opt("--out").map(PathBuf::from);
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(default_manifest()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut loaded: Vec<(String, Car)> = Vec::new();
    for name in cars.split(',') {
        let spec: CarSpec = match m.cars.iter().find(|c| c.name == name) {
            Some(c) => c.clone(),
            None => {
                let (label, rest) = name.split_once('=').unwrap_or((name, name));
                let v: Vec<&str> = rest.split(':').collect();
                let (md, vehicle, pc) = match v[..] {
                    [md, vehicle, pc] => (md, vehicle, pc.to_string()),
                    _ => (rest, "sedan", "sedan.pc".to_string()),
                };
                serde_json::from_value(serde_json::json!({
                    "name": label, "label": label, "mod": md, "vehicle": vehicle, "pc": pc,
                }))
                .map_err(|e| e.to_string())?
            }
        };
        let t = std::time::Instant::now();
        let car = Car::load(&spec)?;
        eprintln!(
            "{}: loaded in {:.1} s ({} nodes, {} beams)",
            spec.name,
            t.elapsed().as_secs_f32(),
            car.template.nodes.len(),
            car.template.beams.len()
        );
        loaded.push((spec.name.clone(), car));
    }
    for scene in scenes.split(',') {
        let (kmh, barrier) = match scene {
            "wall" => (56.0, Front::Wall),
            "offset" => (64.0, Front::Offset(0.4)),
            "pole" => (50.0, Front::Pole),
            _ => return Err(format!("unknown scene {scene}")),
        };
        let results: Vec<(String, Value)> = loaded
            .iter()
            .map(|(n, c)| {
                let t = std::time::Instant::now();
                let r = scene_front(c, kmh, secs, barrier);
                eprintln!("{n} {scene}: {:.1} s", t.elapsed().as_secs_f32());
                if let Some(o) = &out {
                    std::fs::create_dir_all(o).ok();
                    std::fs::write(
                        o.join(format!("{n}.{scene}.json")),
                        serde_json::to_string_pretty(&r).unwrap(),
                    )
                    .ok();
                }
                (n.clone(), r)
            })
            .collect();
        println!("\n### {scene}, {kmh} km/h ({secs} s after contact)\n");
        println!(
            "| | {} |",
            results
                .iter()
                .map(|r| r.0.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        );
        println!("|---|{}", "---|".repeat(results.len()));
        let row = |label: &str, f: &dyn Fn(&Value) -> String| {
            println!(
                "| {label} | {} |",
                results
                    .iter()
                    .map(|r| f(&r.1))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
        };
        let num = |v: &Value, path: &[&str]| -> String {
            let mut x = v;
            for k in path {
                x = &x[*k];
            }
            match x {
                Value::Null => "–".into(),
                Value::String(s) => s.clone(),
                other => other.to_string(),
            }
        };
        row("mass (kg)", &|v| num(v, &["mass_kg"]));
        row("dynamic crush (mm) at (ms)", &|v| {
            format!(
                "{} at {}",
                num(v, &["dynamic_crush_mm"]),
                num(v, &["t_m_ms"])
            )
        });
        row("peak g (rear cabin, 100 Hz) / mean to stop", &|v| {
            format!("{} / {}", num(v, &["peak_g"]), num(v, &["mean_g_to_tm"]))
        });
        row("peak barrier force (kN) at (ms)", &|v| {
            format!(
                "{} at {}",
                num(v, &["peak_force_kn"]),
                num(v, &["peak_force_ms"])
            )
        });
        row("rebound (m/s), restitution", &|v| {
            format!("{}, {}", num(v, &["rebound_ms"]), num(v, &["restitution"]))
        });
        row("residual crush across the width, −x to +x (mm)", &|v| {
            num(v, &["residual_crush_zones_mm"])
        });
        row("residual crush max / mean (mm)", &|v| {
            format!(
                "{} / {}",
                num(v, &["residual_crush_max_mm"]),
                num(v, &["residual_crush_avg_mm"])
            )
        });
        row("springback", &|v| num(v, &["springback_ratio"]));
        row("intrusion", &|v| {
            v["intrusion"].as_object().map_or("–".into(), |o| {
                o.iter()
                    .map(|(k, x)| format!("{k} {}/{}", x["max_mm"], x["residual_mm"]))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        });
        row("beams bent / broken", &|v| {
            format!(
                "{} / {}",
                num(v, &["damage", "bent_beams"]),
                num(v, &["damage", "broken_beams"])
            )
        });
        row("hardware broken", &|v| {
            format!(
                "{} of {}: {}",
                num(v, &["damage", "hardware_broken"]),
                num(v, &["damage", "hardware_total"]),
                num(v, &["damage", "hardware_broken_names"])
            )
        });
        row("parts detached", &|v| {
            format!(
                "{}: {}",
                num(v, &["damage", "parts_detached"]),
                num(v, &["damage", "parts_detached_names"])
            )
        });
        row("deepest node in the barrier (mm)", &|v| {
            num(v, &["penetration", "nodes", "max_mm"])
        });
        row("energy: closure share, largest rise (J)", &|v| {
            format!(
                "{}, {}",
                num(v, &["energy", "closure_share"]),
                num(v, &["energy", "max_rise_j"])
            )
        });
        row("mesh", &|v| {
            v["visual"].as_object().map_or("–".into(), |o| {
                o.iter()
                    .filter(|(_, x)| !x.is_object() && !x.is_array())
                    .map(|(k, x)| format!("{k} {x}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        });
        row("ripple at the end", &|v| {
            format!(
                "worst {} {} mm",
                num(v, &["ripple_end", "worst_part"]),
                num(v, &["ripple_end", "worst_mm"])
            )
        });
    }
    Ok(())
}
