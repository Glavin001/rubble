//! Every registered design, by the registry alone: no design is named here.
//! A new design that is added to `designs::REGISTRY` is covered by these.

use engine_vehiclegen::mesh::Surf;
use engine_vehiclegen::solid::check;
use engine_vehiclegen::{build, default_params_json, design_from_json, design_names, BuildOpts};

#[test]
fn the_registry_lists_designs_and_each_builds_from_its_defaults() {
    let names = design_names();
    assert!(names.contains(&"sedan"));
    for n in names {
        let json = default_params_json(n).unwrap();
        let d = design_from_json(n, &json).unwrap_or_else(|e| panic!("{n}: {e}"));
        let b = build(d.as_ref(), BuildOpts::default());
        assert!(!b.parts.0.is_empty() && !b.markers.is_empty(), "{n}: nothing built");
        assert!(d.rig_spec().is_some(), "{n}: not rigged (no rig_spec)");
        for (name, m) in &b.parts.0 {
            assert!(m.ntris() > 0, "{n}/{name}: empty part");
            assert!(m.v.iter().all(|p| p.is_finite()), "{n}/{name}: non-finite vertex");
            // solid panels: closed, as the sedan's are
            if m.surf.as_ref().is_some_and(|s| s.iter().any(|x| *x != Surf::Outer)) {
                let r = check(m);
                assert!(r.closed(), "{n}/{name}: {} open edges", r.boundary_edges);
            }
        }
    }
}

#[test]
fn unknown_designs_name_the_registry() {
    let e = design_from_json("nope", "{}").err().unwrap();
    assert!(e.contains("sedan") && e.contains("police_sedan"), "{e}");
}

#[test]
fn variants_keep_their_base_parts_and_add_their_own() {
    let base = build(design_from_json("sedan", "{}").unwrap().as_ref(), BuildOpts::default());
    let police = build(design_from_json("police_sedan", "{}").unwrap().as_ref(), BuildOpts::default());
    for (name, _) in &base.parts.0 {
        assert!(police.parts.get(name).is_some(), "police lost {name}");
    }
    for extra in ["lightbar", "lightbar_lens", "push_bar", "spotlight", "antenna"] {
        assert!(police.parts.get(extra).is_some(), "police has no {extra}");
    }
}
