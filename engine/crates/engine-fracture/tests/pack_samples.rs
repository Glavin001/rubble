//! The Scene menu's pack samples (the Wrecking Yard and Topple, `stress_ref::wrecking`):
//! each is a structure the stress oracle builds, its JSON builds as a lab scene (one body
//! per chunk, the footings static, a breakable weld per bond at the calibrated strengths),
//! and its menu entry counts its chunks. Topple's carry their charges.

use engine_fracture::pack::{build_pack_scene, BLAST_DT};
use engine_fracture::samples::{pack_sample_json, GROUPS, PACK_SAMPLES, TOPPLE, WRECKING_YARD};

#[test]
fn pack_samples_build_as_lab_scenes() {
    let keys: Vec<&str> = PACK_SAMPLES.iter().map(|s| s.key).collect();
    let structures: Vec<&str> = stress_ref::wrecking::STRUCTURES
        .iter()
        .map(|s| s.0)
        .collect();
    assert_eq!(keys, structures);
    assert!(GROUPS.ends_with(&[WRECKING_YARD, TOPPLE]), "the pack samples' groups last");
    for s in PACK_SAMPLES {
        assert!([WRECKING_YARD, TOPPLE].contains(&s.group), "{}: {}", s.key, s.group);
        let charges = stress_ref::wrecking::charges_of(&serde_json::from_slice(&pack_sample_json(s.key).unwrap()).unwrap());
        assert_eq!(!charges.is_empty(), s.group == TOPPLE, "{}: Topple's structures and only they carry charges", s.key);
        let structure = stress_ref::wrecking::structure(s.key).expect(s.key);
        let chunks = structure.chunks.len();
        let shown = s
            .menu
            .rsplit(" · ")
            .next()
            .and_then(|n| n.split(' ').next())
            .map(|n| n.replace(',', ""));
        assert_eq!(
            shown,
            Some(chunks.to_string()),
            "{}: the menu counts the chunks",
            s.key
        );
        let (scene, info) = build_pack_scene(&pack_sample_json(s.key).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e:#}", s.key));
        assert_eq!(
            (info.nodes, info.bonds),
            (chunks, structure.bonds.len()),
            "{}",
            s.key
        );
        let fixed = structure
            .chunks
            .iter()
            .filter(|c| c.support == stress_ref::scene::Support::Fixed)
            .count();
        assert_eq!(info.supports, fixed, "{}", s.key);
        let statics = scene.bodies[1..=info.nodes]
            .iter()
            .filter(|b| b.inv_mass == 0.0)
            .count();
        assert_eq!(statics, fixed, "{}: footings are static", s.key);
        assert_eq!(
            scene.joints.len(),
            structure.bonds.len(),
            "{}: a weld per bond",
            s.key
        );
        // The welds' tension capacity is the calibrated strength over the bond's area.
        let joint = if s.key == "wy_brick_tower" {
            stress_ref::material::Material::mortar()
        } else {
            stress_ref::material::Material::concrete()
        };
        for (j, b) in scene.joints.iter().zip(&structure.bonds) {
            let expected = joint.tensile_strength * b.area;
            let got = j.strength[0];
            assert!(
                (got - expected).abs() <= 1e-6 * expected.max(1.0) + 1e-9 / BLAST_DT,
                "{}: {got} N vs {expected} N",
                s.key
            );
        }
        // About the structure's mass (the lab's bodies are f32).
        let mass: f64 = scene.bodies[1..=info.nodes]
            .iter()
            .filter(|b| b.inv_mass > 0.0)
            .map(|b| 1.0 / b.inv_mass as f64)
            .sum();
        let built: f64 = structure
            .chunks
            .iter()
            .map(stress_ref::wrecking::mass)
            .sum();
        assert!(
            (mass - built).abs() < 1e-4 * built,
            "{}: {mass} kg vs {built} kg",
            s.key
        );
    }
    assert!(pack_sample_json("house").is_err());
}
