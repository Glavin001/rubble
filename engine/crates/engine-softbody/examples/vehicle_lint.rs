//! Lints node–beam cars against the hard rules and the census bands
//! (`authoring::lint`, `engine/docs/generated-vehicles.md`).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_lint -- sedan [gen3] [etkk] [label=<mod dir>,<vehicle>,<pc>]
//! ```
//! A bare name is a mod under `engine/crates/engine-webgpu/__mods/`: a
//! vehicle-gen design written by `vehicle_gen_rig` (its `rig.json` gives the
//! design mass and the markers), `gen3` (`__mods/gen2`, `gen2_export`), or
//! `etkk` (BeamNG community content, local only: `RUBBLE_LINT_MODS` names
//! another `__mods` folder to read it from). Community cars are read as data
//! and reported for the bands' self-consistency only.
//! Exit status 1 when a generated car has a hard failure.
use engine_softbody::authoring::gen2::rig::WHEEL_KG;
use engine_softbody::authoring::gen2::vg_input::RigJson;
use engine_softbody::authoring::lint::{self, Expect};
use engine_softbody::vehicle_compare::{Car, CarSpec, Source};
use glam::Vec3;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mods = root.join("crates/engine-webgpu/__mods");
    let other = std::env::var("RUBBLE_LINT_MODS").ok().map(PathBuf::from);
    let names = if a.is_empty() { vec!["sedan".to_string()] } else { a };
    let mut hard_total = 0;
    for name in names {
        let (label, dir, vehicle, pc, generated) = if let Some((l, rest)) = name.split_once('=') {
            let v: Vec<&str> = rest.split(',').collect();
            (
                l.to_string(),
                PathBuf::from(v[0]),
                v.get(1).unwrap_or(&l).to_string(),
                v.get(2).map(|s| s.to_string()),
                false,
            )
        } else if name == "gen3" {
            ("gen3".into(), mods.join("gen2"), "gen2".into(), Some("gen2.pc".into()), true)
        } else if name == "etkk" {
            let d = other.clone().unwrap_or(mods.clone()).join("etkk");
            ("etkk".into(), d, "etkk".into(), Some("kr6.pc".into()), false)
        } else {
            (
                name.clone(),
                mods.join(&name),
                name.clone(),
                Some(format!("{name}.pc")),
                true,
            )
        };
        if !dir.exists() {
            println!("\n## {label}\n\nskipped: {} is absent", dir.display());
            continue;
        }
        let spec: CarSpec = serde_json::from_value(serde_json::json!({
            "name": label, "mod": dir.to_string_lossy(), "vehicle": vehicle, "pc": pc
        }))
        .unwrap();
        let car = match Car::load(&spec) {
            Ok(c) => c,
            Err(e) => {
                println!("\n## {label}\n\nload failed: {e}");
                continue;
            }
        };
        let Source::Jbeam { desc, .. } = &car.source else {
            continue;
        };
        // A vehicle-gen design's rig.json (from its build) gives its design
        // mass and its markers.
        let mut ex = Expect::default();
        let rj = [
            root.join("tools/vehicle-gen/out").join(&label),
            root.join("scenes/vehicles").join(&label),
        ]
        .into_iter()
        .map(|d| d.join(format!("{label}.rig.json")))
        .find(|p| p.exists());
        if let Some(p) = rj.filter(|_| generated && label != "gen3") {
            let r: RigJson = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            let ph: engine_softbody::authoring::gen2::rig::Physics =
                serde_json::from_value(if r.physics.is_null() {
                    serde_json::json!({})
                } else {
                    r.physics.clone()
                })
                .unwrap();
            ex.mass = Some(ph.kerb - 4.0 * WHEEL_KG);
            // glTF to the loader's frame (Rubble: +Y up, +Z forward, the
            // origin at the front axle).
            let af = r.dims.axle_f;
            ex.markers = r
                .markers
                .iter()
                .map(|(n, m)| (n.clone(), Vec3::new(m.pos[0], m.pos[1], m.pos[2] - af)))
                .collect();
        } else if label == "gen3" {
            ex.mass = Some(engine_softbody::authoring::gen2::rig::KERB - 4.0 * WHEEL_KG);
        }
        let f = lint::lint(desc, car.visual.as_ref(), &ex);
        let h = lint::print(&label, &f);
        if generated {
            hard_total += h;
        }
    }
    if hard_total > 0 {
        std::process::exit(1);
    }
}
