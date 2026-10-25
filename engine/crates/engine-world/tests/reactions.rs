//! The reference destruction rigs end to end (`engine/docs/destruction-reactions.md`,
//! "Tests and oracle"): the punch-through wall and the reference building, ported from
//! blast-stress-solver as scene packs, run as threshold destructibles with every reaction
//! counted (`support/reactions.rs`). Gates are physical invariants, not another engine's
//! numbers.
#![cfg(feature = "stress")]

#[path = "support/reactions.rs"]
mod reactions;

use reactions::*;
use rubble::*;

fn spec(scene: &str, shot: Option<&str>, seconds: f64) -> RunSpec {
    RunSpec {
        scene: scene.into(),
        shot: shot.map(str::to_string),
        seconds,
    }
}

#[test]
fn both_reference_scenes_load_with_their_crush_blocks_and_shots() {
    for (key, bytes) in SCENES {
        let json: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let nodes = json["scenario"]["nodes"].as_array().unwrap().len();
        let bonds = json["scenario"]["bonds"].as_array().unwrap().len();
        // The authored crush parameters survive the port (v3 packs).
        let crush = json["defaults"]["solver"]["materials"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m.get("crush").is_some())
            .count();
        assert_eq!(
            crush > 0,
            json["version"] == 3,
            "{key}: crush blocks iff v3"
        );
        // The World loader takes them (unknown fields are not an error).
        let asset = DestructibleAssetDesc::from_pack(bytes).unwrap();
        assert!(asset.name.is_some(), "{key}");
        let stress = asset.stress.as_ref().unwrap();
        if let Err(e) = stress.scene() {
            panic!("{key}: the stress import failed: {e}");
        }
        let scene = stress
            .threshold_scene(1.0, asset.solver_for(StrengthModel::Threshold).unwrap().dt)
            .unwrap_or_else(|e| panic!("{key}: {e}"));
        let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
        assert_eq!(chunks, nodes, "{key}: one stress chunk per pack node");
        assert_eq!(asset.chunks.len() + asset.bonds.len() > 0, true);
        eprintln!(
            "{key}: {nodes} nodes, {bonds} bonds, {} world chunks, {} world bonds",
            asset.chunks.len(),
            asset.bonds.len()
        );
        let shots = engine_fracture::pack::pack_shots(bytes).unwrap();
        assert!(!shots.is_empty(), "{key}: authored shots");
        for s in &shots {
            assert!(s.mass > 0.0 && s.radius > 0.0, "{key} {}", s.label);
            // Every shot flies toward the structure (+x, the source's launcher side).
            assert!(
                s.velocity[0] > 0.0 && s.origin[0] < -5.0,
                "{key} {}",
                s.label
            );
        }
    }
}

#[test]
fn reference_building_stands_under_gravity_for_5_s() {
    let r = run(&spec("reference-building-crush", None, 5.0)).unwrap();
    assert_eq!(r.total_broken(), 0, "gravity alone breaks no bond");
    assert_eq!(r.total_splits(), 0);
    assert_eq!(r.bonds_intact_end, r.bonds);
    assert_eq!(
        r.total_crushed(),
        0,
        "a structure under its own weight never crushes"
    );
}

/// Mass is conserved through every split, frame by frame (no crushing removes it: the
/// charter's default `CrushMass::Conserve`).
fn assert_mass_conserved(r: &Report) {
    let m0 = r.frames[0].system_mass;
    let shot = r.shot.as_ref().map_or(0.0, |s| s.mass);
    for f in &r.frames {
        let expect = if f.impactor_speed.is_some() {
            r.structure_mass_start + shot
        } else {
            r.structure_mass_start
        };
        assert!(
            (f.system_mass - expect).abs() <= 1e-4 * expect,
            "{}: frame {} system mass {} vs {expect} (first {m0})",
            r.scene,
            f.frame,
            f.system_mass
        );
    }
    assert!((r.structure_mass_end - r.structure_mass_start).abs() <= 1e-4 * r.structure_mass_start);
}

#[test]
fn an_ordinary_shot_knocks_chunks_off_and_conserves_mass() {
    for (scene, shot) in [
        ("reference-building-crush", "ordinary"),
        ("crush-wall", "1.0x"),
    ] {
        let r = run(&spec(scene, Some(shot), 4.0)).unwrap();
        eprintln!("{}", to_markdown(std::slice::from_ref(&r)));
        assert!(
            r.total_broken() > 0,
            "{scene} {shot}: the shot breaks bonds"
        );
        assert!(
            r.total_splits() > 0 && r.actors_end() > 1,
            "{scene} {shot}: pieces come off"
        );
        // No break before the shot arrives: everything before launch is gravity.
        let s = r.shot.as_ref().unwrap();
        assert!(
            r.frames[..s.launch_frame as usize]
                .iter()
                .all(|f| f.bonds_broken == 0),
            "{scene}: bonds broke before the shot"
        );
        // The impactor pays for the damage: it leaves slower than it came.
        assert!(
            s.speed_out < s.speed_in && s.ke_out < s.ke_in,
            "{scene}: {s:?}"
        );
        assert_mass_conserved(&r);
    }
}

#[test]
fn two_runs_produce_identical_events() {
    let a = run(&spec("crush-wall", Some("1.4x"), 3.0)).unwrap();
    let b = run(&spec("crush-wall", Some("1.4x"), 3.0)).unwrap();
    assert!(!a.events.is_empty());
    assert_eq!(a.events, b.events, "the same events in the same frames");
    assert_eq!(
        a.frames, b.frames,
        "the same counts, speeds and masses every frame"
    );
}
