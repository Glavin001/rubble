//! The 1969 Mach 1: the legacy sheet build is pinned to the Python
//! generator's (`engine/scenes/vehicles/index.json`: 60 parts, 285,290
//! triangles, 4,722 x 1,870 x 1,300 mm), and the solid build is a two-door car.

use engine_vehiclegen::{build, design_from_json, BuildOpts};

fn bounds(b: &engine_vehiclegen::Built, skip: &dyn Fn(&str) -> bool) -> glam::DVec3 {
    let (mut lo, mut hi) = (glam::DVec3::splat(1e9), glam::DVec3::splat(-1e9));
    for (n, m) in &b.parts.0 {
        if skip(n) {
            continue;
        }
        for v in &m.v {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    hi - lo
}

#[test]
fn the_legacy_sheets_match_the_python_generator() {
    let d = design_from_json("mach1_1969", "{}").unwrap();
    let b = build(d.as_ref(), BuildOpts { solid: false, ..Default::default() });
    assert_eq!(b.parts.len(), 60);
    let total: usize = b.parts.0.iter().map(|(_, m)| m.ntris()).sum();
    assert_eq!(total, 285_290);
    let s = bounds(&b, &|_| false);
    assert!((s.z - 4.722).abs() < 0.001 && (s.x - 1.870).abs() < 0.001 && (s.y - 1.300).abs() < 0.001, "{s:?}");
}

#[test]
fn it_has_two_doors_and_a_boot_lid() {
    let d = design_from_json("mach1_1969", "{}").unwrap();
    let spec = d.rig_spec().unwrap();
    assert_eq!(spec.doors, ["FL", "FR"]);
    assert_eq!(spec.rear_closure, "trunk_lid");
    let b = build(d.as_ref(), BuildOpts::default());
    for n in ["door_FL", "door_FR", "hood", "trunk_lid", "glass_quarter_L", "glass_quarter_R"] {
        assert!(b.parts.get(n).is_some(), "no {n}");
    }
    for n in ["door_RL", "door_RR", "door_trim_RL", "handle_RL"] {
        assert!(b.parts.get(n).is_none(), "{n}: a two-door car");
    }
    for m in &b.markers {
        assert!(!m.name.contains("door_R"), "{}: a marker of a rear door", m.name);
    }
}
