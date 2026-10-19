//! Backfill visual sidecars by replaying the original scene and verifying saved arrays.
//! usage: learned_impact_replay DATASET OUTPUT [--event ID] [--jobs N]
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use stress_learned::{
    generate::*,
    oracle::{record_event_with_basis, LoadBasis},
    RecordMeta,
};
use stress_ref::{builders, scene::Scene, showcases};

fn read(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn choose(man: &Value) -> Result<(Scene, bool), String> {
    let id = man["event_id"].as_str().ok_or("missing event_id")?;
    let notes = man["notes"].as_str().unwrap_or("");
    let set = notes.strip_prefix("generated set=");
    let jobs = match set {
        Some("smoke") => smoke_jobs(),
        Some("mini") => mini_jobs(),
        Some("walls") => wall_jobs(),
        Some("prove") => prove_jobs(),
        Some("diversity_v1") => diversity_jobs(),
        Some("diversity_v2") => diversity_v2_jobs(),
        _ => Vec::new(),
    };
    if let Some(j) = jobs.into_iter().find(|j| j.event_id == id) {
        return Ok((j.scene, j.qs_probe));
    }
    // Single-event oracle probes used the standard builders. Validation below
    // rejects any changed scene; metadata alone never establishes equivalence.
    let kind = man["structure_type"].as_str().unwrap_or("");
    let load = man["load_type"].as_str().unwrap_or("");
    let speed = load
        .strip_prefix("ram_")
        .or_else(|| load.strip_prefix("car_"))
        .and_then(|s| s.strip_suffix("mps"))
        .and_then(|s| s.parse::<f64>().ok());
    let scene = match (kind, speed) {
        ("wall_b5", Some(v)) => builders::wall_impact(v, "replay_wall"),
        ("masonry_house", Some(v)) => {
            showcases::masonry_house(showcases::HouseEvent::Impact { speed: v }, "replay_house")
        }
        ("bond_tension", _) => builders::bond_tension(),
        _ => return Err(format!("no source scene for {id} ({notes})")),
    };
    Ok((scene, notes.contains("qs_probe=true")))
}
fn verify(a: &Path, b: &Path, frames: usize) -> Result<(), String> {
    // All inputs/labels, including the complete static graph, must be identical.
    // QS probe is read-only and can be on/off in old recordings without metadata;
    // its field is excluded explicitly, because it cannot affect world motion.
    for rel in std::iter::once(PathBuf::new())
        .chain((0..frames).map(|i| PathBuf::from(format!("frames/{i:06}"))))
    {
        let ca = read(&a.join(&rel).join("catalog.json"))?;
        let cb = read(&b.join(&rel).join("catalog.json"))?;
        for (key, ent) in ca["arrays"].as_object().ok_or("invalid catalog")? {
            if key == "bond_qs_force" {
                continue;
            }
            let other = &cb["arrays"][key];
            if ent["shape"] != other["shape"] || ent["dtype"] != other["dtype"] {
                return Err(format!("{rel:?}/{key}: shape or dtype changed"));
            }
            let pa = ent["path"].as_str().ok_or("array path")?;
            let pb = other["path"].as_str().ok_or("array path")?;
            let aa = fs::read(a.join(&rel).join(pa)).map_err(|e| e.to_string())?;
            let bb = fs::read(b.join(&rel).join(pb)).map_err(|e| e.to_string())?;
            if aa != bb {
                return Err(format!(
                    "{rel:?}/{key}: replay differs from recorded data; motion rejected"
                ));
            }
        }
    }
    Ok(())
}
fn one(root: &Path, out: &Path, entry: &Value) -> Result<Value, String> {
    let name = entry["path"].as_str().ok_or("path")?;
    let source = root.join(name);
    let man = read(&source.join("manifest.json"))?;
    let target = out.join(format!("{name}.json"));
    if target.exists() {
        let prior = read(&target)?;
        if prior["source_sha256"] == entry["sha256"] && prior["verification"].as_str().is_some() {
            return Ok(json!({"event":name,"status":"cached"}));
        }
    }
    let start = Instant::now();
    let (mut scene, qs) = choose(&man)?;
    scene.sim.frame_dt = man["dt"].as_f64().ok_or("dt")?;
    scene.sim.seed = man["seed"].as_u64().ok_or("seed")?;
    let frames = man["n_frames"].as_u64().ok_or("frames")? as u32;
    let temp = out.join(".recorded").join(name);
    eprintln!("replay start {name} ({frames} frames)");
    let basis = if source.join("context.json").exists()
        && read(&source.join("context.json"))?["load_basis"] == "frame_start_cluster"
    {
        LoadBasis::FrameStart
    } else {
        LoadBasis::LegacyFrameEnd
    };
    record_event_with_basis(
        scene,
        &temp,
        RecordMeta {
            event_id: man["event_id"].as_str().unwrap().into(),
            structure_type: man["structure_type"].as_str().unwrap().into(),
            load_type: man["load_type"].as_str().unwrap().into(),
            notes: "visual replay verification".into(),
        },
        frames,
        qs,
        basis,
    )?;
    verify(&source, &temp, frames as usize)?;
    let mut replay = read(&temp.join("replay.json"))?;
    replay["source_sha256"] = entry["sha256"].clone();
    replay["verification"]=json!("All stored static and per-frame inputs/labels bit-identical, except read-only bond_qs_force probe; original training files unchanged.");
    fs::write(
        &target,
        serde_json::to_vec(&replay).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "replay verified {name} {:.1}s",
        start.elapsed().as_secs_f64()
    );
    Ok(json!({"event":name,"status":"verified","seconds":start.elapsed().as_secs_f64()}))
}
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: learned_impact_replay DATASET OUTPUT [--event ID] [--jobs N]");
        std::process::exit(2);
    }
    let root = Path::new(&a[0]);
    let out = Path::new(&a[1]);
    fs::create_dir_all(out).unwrap();
    let filter = a
        .windows(2)
        .find(|v| v[0] == "--event")
        .map(|v| v[1].clone());
    let jobs: usize = a
        .windows(2)
        .find(|v| v[0] == "--jobs")
        .map(|v| v[1].parse().expect("jobs integer"))
        .unwrap_or(8);
    let man = read(&root.join("manifest.json")).expect("dataset manifest");
    let entries: Vec<_> = man["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| {
            filter.as_ref().is_none_or(|f| {
                e["event_id"].as_str() == Some(f.as_str()) || e["path"].as_str() == Some(f.as_str())
            })
        })
        .collect();
    if entries.is_empty() {
        eprintln!("no matching events");
        std::process::exit(2);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();
    let result: Vec<_> = pool.install(|| {
        entries
            .par_iter()
            .map(|e| match one(root, out, e) {
                Ok(v) => v,
                Err(error) => {
                    eprintln!("replay unavailable {}: {error}", e["path"]);
                    json!({"event":e["path"],"status":"rejected","error":error})
                }
            })
            .collect()
    });
    let bad = result.iter().filter(|r| r["status"] == "rejected").count();
    fs::write(
        out.join(if filter.is_some() {
            "selected-status.json"
        } else {
            "status.json"
        }),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!(
        "replay completed: {} ready, {bad} rejected",
        result.len() - bad
    );
    if bad > 0 {
        std::process::exit(1);
    }
}
