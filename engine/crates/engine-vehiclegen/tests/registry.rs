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
                assert!(
                    r.closed(),
                    "{n}/{name}: {} open edges, {} non-manifold, {} pieces",
                    r.boundary_edges,
                    r.nonmanifold_edges,
                    r.components
                );
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

/// The configurator's controls come from the registry: every design has a schema whose
/// numbers hold their defaults inside the range `validate()` enforces, and a value just
/// outside a range is the validator's error (the range list is the validator's own).
#[test]
fn every_designs_schema_matches_its_validator() {
    for (name, _) in engine_vehiclegen::design_titles() {
        let schema: serde_json::Value =
            serde_json::from_str(&engine_vehiclegen::param_schema_json(name).unwrap()).unwrap();
        let rows = schema.as_array().unwrap();
        assert!(!rows.is_empty(), "{name}");
        let defaults: serde_json::Value =
            serde_json::from_str(&engine_vehiclegen::default_params_json(name).unwrap()).unwrap();
        for r in rows {
            let k = r["key"].as_str().unwrap();
            assert_eq!(r["value"], defaults[k], "{name}.{k}");
            if r["kind"] == "number" {
                let (v, lo, hi) = (r["value"].as_f64().unwrap(), r["min"].as_f64().unwrap(), r["max"].as_f64().unwrap());
                assert!(lo <= v && v <= hi, "{name}.{k} = {v} not in {lo}..{hi}");
                assert!(r["step"].as_f64().unwrap() > 0.0);
                // The validator refuses a value past the top of the range it is built from.
                let mut bad = defaults.clone();
                bad[k] = serde_json::json!(hi + 0.5);
                let err = engine_vehiclegen::validate_params(name, &bad.to_string());
                assert!(err.is_err(), "{name}: {k} = {} should be refused", hi + 0.5);
            }
            if r["kind"] == "choice" {
                assert!(r["options"].as_array().unwrap().contains(&defaults[k]), "{name}.{k}");
            }
        }
        engine_vehiclegen::validate_params(name, "{}").unwrap();
    }
}
