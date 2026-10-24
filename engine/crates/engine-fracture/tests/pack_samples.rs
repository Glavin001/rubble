//! The Scene menu's pack samples (the Wrecking Yard, Bridge Out, Topple, Ram Raid and
//! Fly-through groups, `stress_ref::wrecking`): each is a structure the stress oracle
//! builds, its JSON builds as a lab scene (one body per chunk, the footings static, a
//! breakable weld per bond at the calibrated strengths of its joint material), and its menu
//! entry counts its chunks. Topple's carry their charges.

use engine_fracture::pack::{build_pack_scene, BLAST_DT};
use engine_fracture::samples::{pack_sample_json, GROUPS, PACK_GROUPS, PACK_SAMPLES, TOPPLE};

#[test]
fn pack_samples_build_as_lab_scenes() {
    let keys: Vec<&str> = PACK_SAMPLES.iter().map(|s| s.key).collect();
    let structures: Vec<&str> = stress_ref::wrecking::STRUCTURES
        .iter()
        .map(|s| s.0)
        .collect();
    assert_eq!(keys, structures);
    assert!(GROUPS.ends_with(PACK_GROUPS), "the pack samples' groups come last");
    // Each group's entries together, in the groups' order.
    let order: Vec<usize> = PACK_SAMPLES
        .iter()
        .map(|s| PACK_GROUPS.iter().position(|g| *g == s.group).expect(s.group))
        .collect();
    assert!(order.windows(2).all(|w| w[0] <= w[1]), "{order:?}");
    for s in PACK_SAMPLES {
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
        // The welds' capacities are their joint material's calibrated strengths over the
        // bond's area (each bond its own material's, in a structure of several).
        for (j, b) in scene.joints.iter().zip(&structure.bonds) {
            let (_, joint) = stress_ref::wrecking::calibrated(&b.material);
            let strengths = [
                joint.tensile_strength,
                joint.compressive_strength,
                joint.cohesion,
            ];
            for (k, strength) in strengths.iter().enumerate() {
                let expected = strength * b.area;
                let got = j.strength[k];
                assert!(
                    (got - expected).abs() <= 1e-6 * expected.max(1.0) + 1e-9 / BLAST_DT,
                    "{}: {} bond, capacity {k}: {got} N vs {expected} N",
                    s.key,
                    b.material
                );
            }
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

/// A pack whose materials carry their own limits gives each bond its material's; a bond
/// whose material has none fails the load with a clear error unless pack-wide limits are
/// there for it to take, and incomplete limits fail too.
#[test]
fn per_material_limits_are_per_bond_or_an_error() {
    let load = |pack: &serde_json::Value| build_pack_scene(&serde_json::to_vec(pack).unwrap());
    let mut pack: serde_json::Value =
        serde_json::from_slice(&pack_sample_json("rr_storefront_row").unwrap()).unwrap();
    let solver = &pack["defaults"]["solver"];
    assert!(
        solver["limits"].is_null(),
        "several joint materials: no pack-wide limits"
    );
    assert_eq!(solver["materials"].as_array().unwrap().len(), 3);
    let bonds = pack["scenario"]["bonds"].as_array().unwrap().clone();
    let (scene, _) = load(&pack).unwrap();
    for (j, b) in scene.joints.iter().zip(&bonds) {
        let m = &pack["defaults"]["solver"]["materials"][b["m"].as_u64().unwrap() as usize];
        let expected =
            m["limits"]["tensionFatal"].as_f64().unwrap() * b["area"].as_f64().unwrap() / BLAST_DT;
        assert!(
            (j.strength[0] - expected).abs() <= 1e-6 * expected,
            "{} N vs {expected} N",
            j.strength[0]
        );
    }
    // Incomplete limits: an error naming what is missing.
    let mut incomplete = pack.clone();
    incomplete["defaults"]["solver"]["materials"][0]["limits"] =
        serde_json::json!({"tensionFatal": 1.0});
    let err = format!("{:#}", load(&incomplete).unwrap_err());
    assert!(err.contains("no compressionFatal"), "{err}");
    // A material without limits, and no pack-wide ones: an error naming the bond.
    pack["defaults"]["solver"]["materials"][0]["limits"] = serde_json::Value::Null;
    let err = format!("{:#}", load(&pack).unwrap_err());
    assert!(err.contains("has none"), "{err}");
    // With pack-wide limits, such a bond takes those.
    pack["defaults"]["solver"]["limits"] =
        serde_json::json!({"tensionFatal": 7.0, "compressionFatal": 8.0, "shearFatal": 9.0});
    let (scene, _) = load(&pack).unwrap();
    let first = bonds.iter().position(|b| b["m"] == 0).unwrap();
    let expected = 7.0 * bonds[first]["area"].as_f64().unwrap() / BLAST_DT;
    assert!((scene.joints[first].strength[0] - expected).abs() <= 1e-6 * expected);
}
