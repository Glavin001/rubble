//! ScenePack v3 crush blocks (`materials[].crush`, `nodes[].m`) import as chunk
//! materials' crush parameters; v1/v2 packs are unchanged.

use stress_ref::capability::Profile;
use stress_ref::scene_pack::import_value_with;

fn crush_wall() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../packs/crush-wall.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn crush_wall_chunks_carry_their_materials_crush() {
    let (scene, _) = import_value_with(&crush_wall(), "crush-wall", "wall", "crush-wall.json", Profile::Threshold.import_options()).unwrap();
    let chunks = &scene.bodies[0].chunks;
    let mut crushable = 0;
    for c in chunks {
        let m = scene.material(&c.material);
        match c.support {
            stress_ref::scene::Support::None => {
                let p = m.crush.expect("a block-wall chunk crushes");
                assert_eq!(p.cap_pressure, 20e6);
                assert_eq!(p.crush_energy, 4e6);
                assert_eq!(p.crush_viscosity, 2e5);
                assert_eq!(p.friction_slope, 1.2);
                crushable += 1;
            }
            // Anchors (zero-mass footing nodes) never crush.
            _ => assert!(m.crush.is_none()),
        }
    }
    assert!(crushable > 100, "{crushable} crushable chunks");
}

#[test]
fn a_crush_block_needs_version_3() {
    let mut json = crush_wall();
    json["version"] = 2.into();
    let err = import_value_with(&json, "crush-wall", "wall", "crush-wall.json", Profile::Threshold.import_options()).unwrap_err();
    assert!(err.contains("version 2") && err.contains("crush"), "{err}");
}

#[test]
fn a_crush_block_missing_a_required_field_is_an_error() {
    let mut json = crush_wall();
    json["defaults"]["solver"]["materials"][0]["crush"].as_object_mut().unwrap().remove("crushEnergy");
    let err = import_value_with(&json, "crush-wall", "wall", "crush-wall.json", Profile::Threshold.import_options()).unwrap_err();
    assert!(err.contains("crushEnergy"), "{err}");
}

#[test]
fn packs_without_crush_import_as_before() {
    let mut json = crush_wall();
    let base = import_value_with(&json, "crush-wall", "wall", "crush-wall.json", Profile::Threshold.import_options()).unwrap().0;
    for m in json["defaults"]["solver"]["materials"].as_array_mut().unwrap() {
        m.as_object_mut().unwrap().remove("crush");
    }
    let plain = import_value_with(&json, "crush-wall", "wall", "crush-wall.json", Profile::Threshold.import_options()).unwrap().0;
    // The same scene but for the crush parameters.
    let mut stripped = base.clone();
    for m in stripped.materials.values_mut() {
        m.crush = None;
    }
    assert_eq!(serde_json::to_string(&stripped).unwrap(), serde_json::to_string(&plain).unwrap());
}
