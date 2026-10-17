//! Oracle goldens: run the catalogue scenes that are fast enough for `cargo test` and
//! require every gated metric to pass against the committed oracle observations in
//! `golden/<scene>/`. Slow scenes (and those with documented gaps) are covered by
//! `stress-ref check scenes golden`; see README "Status".

mod common;

use std::path::PathBuf;

use stress_ref::builders::catalog;
use stress_ref::metrics::{compare, format_table};
use stress_ref::observation::Observation;

fn goldens(scene: &str) -> Vec<Observation> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden").join(scene);
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else { return out };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for p in paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".json") && !name.starts_with("provenance") && !name.contains("_seed") {
            out.push(Observation::load(&p).unwrap());
        }
    }
    out
}

fn check(names: &[&str]) {
    let scenes = catalog();
    let mut failures = Vec::new();
    for name in names {
        let scene = scenes.iter().find(|s| s.name == *name).unwrap_or_else(|| panic!("no scene {name}"));
        let oracles = goldens(name);
        assert!(!oracles.is_empty(), "no goldens for {name}");
        let obs = common::run(scene);
        let rows = compare(scene, &obs, &oracles);
        println!("{}", format_table(name, &rows));
        failures.extend(rows.iter().filter(|r| r.pass == Some(false)).map(|r| format!("{name}: {} vs {}", r.metric, r.reference_source)));
    }
    assert!(failures.is_empty(), "failing metrics: {failures:#?}");
}

#[test]
fn bond_and_wave_benchmarks_match_opencourant() {
    check(&["b1_bond_tension", "b3_bar_wave"]);
}

#[test]
fn cantilever_and_frame_match_opensees() {
    check(&["b2_cantilever", "b2_cantilever_n10", "b2_cantilever_n40", "b8_frame_sudden", "b8_frame_gradual", "b8_frame_sudden_elastic"]);
}

#[test]
fn pressure_panel_matches_opencourant() {
    check(&["b9_panel_low", "b9_panel_high"]);
}
