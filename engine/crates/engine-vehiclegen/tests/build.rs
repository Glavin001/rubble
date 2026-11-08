//! Whole-vehicle builds: the reference sedan is pinned to the Python
//! generator's output (counts below come from engine/tools/vehicle-gen,
//! `tools/stats.py sedan`, and match it part for part), and parametric
//! builds keep the art spec's structure.

use engine_vehiclegen::{build, design_from_json, BuildOpts};

fn tris(b: &engine_vehiclegen::Built, name: &str) -> usize {
    b.parts.get(name).unwrap_or_else(|| panic!("no part {name}")).ntris()
}

#[test]
fn reference_sedan_matches_the_python_generator() {
    let d = design_from_json("sedan", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts::default());
    assert_eq!(b.parts.len(), 68);
    assert_eq!(b.markers.len(), 42);
    let total: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    assert_eq!(total, 289_196);
    for (name, n) in [("body_shell", 55_160), ("hood", 34_008), ("door_FL", 11_376), ("door_RL", 9_872), ("trunk_lid", 13_552), ("glass_windshield", 5_280), ("tyre_FL", 7_872), ("rim_FL", 4_840), ("grille", 3_660), ("door_trim_FL", 3_814), ("headliner", 5_310)] {
        assert_eq!(tris(&b, name), n, "{name}");
    }
    // the body is symmetric about x = 0 (to microns: relaxation and insets
    // run over the welded mesh, as in the Python)
    let (lo, hi) = b.parts.get("body_shell").unwrap().bbox();
    assert!((lo.x + hi.x).abs() < 1e-5, "{lo} {hi}");
}

#[test]
fn parametric_builds_keep_the_structure() {
    for p in [
        r#"{"wheelbase":2.55,"front_overhang":0.80,"rear_overhang":0.85,"width":1.78,"height":1.42,"track":1.53,"tyre_d":0.64,"tyre_w":0.205,"rim_in":16,"rim_w_in":6.5}"#,
        r#"{"ride":0.15,"height":1.52,"tyre_d":0.78,"tyre_w":0.265,"rim_in":17,"rim_w_in":8,"track":1.66,"width":1.90,"rim_style":"steel"}"#,
    ] {
        let d = design_from_json("sedan", p).unwrap();
        let b = build(d.as_ref(), BuildOpts { uv: false, ..Default::default() });
        assert_eq!(b.parts.len(), 68, "{p}");
        assert_eq!(b.markers.len(), 42, "{p}");
        let wb: serde_json::Value = serde_json::from_str(p).unwrap();
        let want = wb.get("wheelbase").and_then(|v| v.as_f64()).unwrap_or(2.83);
        let z = |n: &str| b.markers.iter().find(|m| m.name == n).unwrap().pos.z;
        assert!((z("wheel_center_FL") - z("wheel_center_RL") - want).abs() < 1e-9, "{p}");
        for (_, m) in &b.parts.0 {
            assert!(m.v.iter().all(|v| v.is_finite()), "{p}");
        }
    }
}

#[test]
fn out_of_range_parameters_are_refused() {
    assert!(design_from_json("sedan", r#"{"ride":-0.05}"#).is_err());
    assert!(design_from_json("sedan", r#"{"track":1.95}"#).is_err());
    assert!(design_from_json("sedan", r#"{"rim_in":22,"tyre_d":0.6}"#).is_err());
    assert!(design_from_json("boat", "{}").is_err());
}
