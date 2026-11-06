//! Every configuration of every converted community vehicle (`community_cache`'s
//! `__mods/index.json`) on the CPU reference as the page spawns it: JBeam mode on
//! raycast wheels (none for a vehicle without wheels), at rest for `secs` (default
//! 1). Prints one markdown row per configuration: loads, wheels, broken and bent
//! beams (those of `beamStrength` 0, which break at once by design, apart), the
//! fastest node, and the parts the configuration names that are not local.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example community_sweep -- [__mods dir] [secs=1] [only=folder,...]
//! ```
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use std::path::{Path, PathBuf};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mods = a
        .iter()
        .find(|x| !x.contains('=') && x.as_str() != "mark")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/__mods"));
    let secs: f32 = a.iter().find_map(|x| x.strip_prefix("secs=")).map_or(1.0, |s| s.parse().unwrap());
    let only: Vec<String> = a
        .iter()
        .find_map(|x| x.strip_prefix("only="))
        .map(|s| s.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    // `mark`: write each configuration's rest result into the index (`rest`:
    // broken, bent, fastest node), which the page's picker shows ("marginal at
    // rest" past 1 m/s or with a beam broken).
    let mark = a.iter().any(|x| x == "mark");
    // `scheme=N`: the damping velocity (4: `ExplicitBeamsSettings::centred_damping`;
    // 1, 2: `ExplicitBeams::damp_scheme`, measurements).
    // `brief`: rows of broken, bent, fastest node, KE and the CPU time per step.
    let scheme: u32 = a
        .iter()
        .find_map(|x| x.strip_prefix("scheme="))
        .map_or(0, |s| s.parse().unwrap());
    let brief = a.iter().any(|x| x == "brief");
    let mut index: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(mods.join("index.json")).unwrap()).unwrap();
    if brief {
        println!("| vehicle | config | broken | bent | fastest node (m/s) | KE (J) | µs/step |");
        println!("|---|---|---|---|---|---|---|");
    } else {
        println!("| vehicle | config | loads | wheels | broken | bent | fastest node (m/s) | missing parts |");
        println!("|---|---|---|---|---|---|---|---|");
    }
    for car in index.iter_mut() {
        let folder = car["folder"].as_str().unwrap().to_string();
        let folder = folder.as_str();
        if !only.is_empty() && !only.iter().any(|o| o == folder) {
            continue;
        }
        let vehicle = car["vehicle"].as_str().unwrap().to_string();
        let vehicle = vehicle.as_str();
        let label = car["label"].as_str().unwrap_or(vehicle).to_string();
        let dir = mods.join(folder);
        let files = engine_softbody::jbeam::read_vehicle_dir(&dir, vehicle).unwrap();
        let refs: Vec<(&str, &str)> = files.iter().map(|(x, y)| (x.as_str(), y.as_str())).collect();
        for k in car["configs"].as_array_mut().unwrap() {
            let pc = k["pc"].as_str().unwrap().to_string();
            let pc = pc.as_str();
            let row = |s: String| println!("| {label} | {} | {s} |", pc.rsplit('/').next().unwrap());
            let Ok(text) = std::fs::read_to_string(dir.join(pc)) else {
                row("no .pc | | | | | ".into());
                continue;
            };
            let desc = match engine_softbody::jbeam::load_jbeam(&refs, Some(&text)) {
                Ok(d) => d,
                Err(e) => {
                    row(format!("**no**: {} | | | | | ", e.replace('|', "/")));
                    continue;
                }
            };
            let missing: Vec<String> = desc
                .warnings
                .iter()
                .filter(|w| w.contains("not found") && !w.contains("bulb") && !w.contains("_W`"))
                .map(|w| w.split('`').nth(3).unwrap_or("").to_string())
                .filter(|p| !p.is_empty())
                .collect();
            let settings = ExplicitBeamsSettings {
                drag: false,
                turbulence: false,
                ..Default::default()
            };
            let spawned = ExplicitBeams::from_desc_jbeam_raycast(&desc, glam::Vec3::ZERO, glam::Vec3::ZERO, settings);
            let (mut s, _) = match spawned {
                Ok(x) => x,
                Err(e) => {
                    row(format!("**spawn**: {} | {} | | | | ", e.replace('|', "/"), desc.wheels.len()));
                    continue;
                }
            };
            // Turned to face +Z and set on the ground, as the page does.
            let f = engine_softbody::pack::forward_axis(&desc);
            let turn = glam::Quat::from_rotation_y(-f.x.atan2(f.z));
            for n in &mut s.nodes {
                n.pos = turn * n.pos;
            }
            let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::INFINITY, f32::min);
            for n in &mut s.nodes {
                n.pos.y -= low - 0.02;
            }
            // 4: the centred damping (`centred_damping`); 1, 2: the measurements.
            if scheme == 4 {
                s.settings.centred_damping = true;
            } else {
                s.damp_scheme = scheme;
            }
            let t0 = std::time::Instant::now();
            let steps0 = s.steps;
            s.run(secs);
            let us = t0.elapsed().as_secs_f64() * 1e6 / (s.steps - steps0).max(1) as f64;
            let ke: f32 = s.nodes.iter().map(|n| 0.5 * n.mass * n.vel.length_squared()).sum();
            let (broken, bent) = s.damage();
            // Beams the file gives no strength (`beamStrength` 0: BeamNG content's
            // sensor beams, e.g. a hood-open switch) break at once by design.
            let designed = s.beams.iter().filter(|b| b.broken && b.strength == 0.0).count();
            let broken = broken - designed;
            let fast = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
            let nan = s.nodes.iter().any(|n| !n.pos.is_finite());
            let mut m = missing.clone();
            m.sort();
            m.dedup();
            if mark {
                k["rest"] = serde_json::json!({ "broken": broken, "bent": bent, "fastest": (fast * 100.0).round() / 100.0, "finite": !nan });
            }
            if brief {
                let nan = if nan { "NaN " } else { "" };
                row(format!("{broken} | {bent} | {nan}{fast:.2} | {ke:.1} | {us:.0}"));
                continue;
            }
            row(format!(
                "yes | {} | {broken}{} | {bent} | {}{fast:.2} | {}",
                desc.wheels.len(),
                if designed > 0 { format!(" (+{designed} of strength 0)") } else { String::new() },
                if nan { "NaN " } else { "" },
                m.join(", ")
            ));
        }
    }
    if mark {
        std::fs::write(mods.join("index.json"), serde_json::to_string_pretty(&index).unwrap()).unwrap();
    }
}
