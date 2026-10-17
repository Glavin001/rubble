//! Convex hull chunks: a scene whose boxes are given as hulls (the same 8 corners) must
//! behave as the box scene. Statics, waves, impact contact and fracture are compared.

mod common;

use stress_ref::builders::catalog;
use stress_ref::observation::Observation;
use stress_ref::scene::Scene;
use stress_ref::showcases;

/// The scene with every chunk replaced by the convex hull of its box corners.
fn as_hulls(scene: &Scene) -> Scene {
    let mut s = scene.clone();
    for body in &mut s.bodies {
        for c in &mut body.chunks {
            let h = c.half_extents;
            c.hull = Some(
                (0..8)
                    .map(|i| [if i & 1 == 0 { -h[0] } else { h[0] }, if i & 2 == 0 { -h[1] } else { h[1] }, if i & 4 == 0 { -h[2] } else { h[2] }])
                    .collect(),
            );
        }
    }
    s.validate().expect("hull scene is valid");
    s
}

fn value(o: &Observation, key: &str) -> f64 {
    o.values[key]
}

fn scene(name: &str) -> Scene {
    catalog().into_iter().chain(showcases::catalog()).find(|s| s.name == name).unwrap_or_else(|| panic!("no scene {name}"))
}

#[test]
fn hull_boxes_match_boxes_in_statics_and_waves() {
    for name in ["b2_cantilever", "b3_bar_wave"] {
        let s = scene(name);
        let (a, b) = (common::run(&s), common::run(&as_hulls(&s)));
        for m in &s.metrics {
            let (x, y) = (
                stress_ref::metrics::evaluate(&s, m, &a).unwrap().as_f64().unwrap(),
                stress_ref::metrics::evaluate(&s, m, &b).unwrap().as_f64().unwrap(),
            );
            println!("{name} {}: box {x} hull {y}", m.name);
            assert!(common::rel(y, x) < 1e-9, "{name} {}: box {x} hull {y}", m.name);
        }
    }
}

/// Contact and fracture: same breach, same fragments and bonds broken within a few,
/// the same car speed lost (the sample points are the same but their order differs, so
/// the force sums round differently and the fracture cascade drifts slightly).
#[test]
fn hull_boxes_match_boxes_under_impact() {
    let s = scene("s_car_brick");
    let (a, b) = (common::run(&s), common::run(&as_hulls(&s)));
    println!(
        "box: broken {} fragments {}; hull: broken {} fragments {}",
        value(&a, "broken_bonds"),
        value(&a, "fragments"),
        value(&b, "broken_bonds"),
        value(&b, "fragments")
    );
    let speed = |o: &Observation| *o.probes["car_velocity"].v.last().unwrap();
    assert!((speed(&a) - speed(&b)).abs() < 0.02 * speed(&a).abs().max(1.0), "car {} vs {}", speed(&a), speed(&b));
    assert!(common::rel(value(&b, "broken_bonds"), value(&a, "broken_bonds")) < 0.1);
    assert!(a.bodies["wall"].iter().any(|c| c.detached) && b.bodies["wall"].iter().any(|c| c.detached));
}

/// The authored assets import with their geometry: hull chunks, supports, bonds with
/// patches recomputed from shared faces. Under their own weight (calibrated materials,
/// code-minimum reinforcement) all four stand. The villa's cantilevered stairs and roof
/// parapets crack a few joints on the way (5 of 2464) to an equilibrium that only the
/// globalised tangent re-solve (`sim.methods.tangent_statics`) reaches; the secant
/// iteration alone diverges there and finds none.
#[test]
fn scene_packs_import_and_stand() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    for (file, nodes, stands) in [("rig-portal.json", 18, true), ("comp-wall-bay.json", 35, true), ("house-2story.json", 599, true), ("villa-savoye.json", 1045, false)] {
        let path = dir.join(file);
        if !path.exists() {
            eprintln!("{} not found; skipping", path.display());
            continue;
        }
        let mut s = stress_ref::scene_pack::import(&path, file, "structure").unwrap();
        let stands = stands || s.sim.methods.tangent_statics;
        let b = &s.bodies[0];
        let hulls = b.chunks.iter().filter(|c| c.hull.is_some()).count();
        let supports = b.chunks.iter().filter(|c| c.support == stress_ref::scene::Support::Fixed).count();
        println!("{file}: {} chunks ({hulls} hulls, {supports} supports), {} bonds; {}; {:?}", b.chunks.len(), b.bonds.len(), s.description, stress_ref::scene_pack::summary(&s));
        assert_eq!(b.chunks.len(), nodes);
        assert_eq!(hulls, nodes);
        assert!(supports > 0 && !b.bonds.is_empty());
        let b_bonds = b.bonds.len() as f64;
        // Standing under its own weight is a static question: equilibrium every frame.
        s.sim.solve_mode = stress_ref::scene::SolveMode::QuasiStatic;
        s.sim.duration = 3.0 * s.sim.frame_dt;
        let o = common::run(&s);
        let unconverged = o.values.get("static_unconverged_frames").copied().unwrap_or(0.0);
        println!("{file}: broken {} fragments {} unconverged frames {unconverged}", o.values["broken_bonds"], o.values["fragments"]);
        if !stands {
            assert!(unconverged > 0.0, "{file} now has a static equilibrium with the secant iteration");
            continue;
        }
        // In equilibrium and nothing separates. Single overloaded joints may crack and
        // shed their load (the house's solid steel rails hang on mortar joints the asset
        // calls brick; they crack and the rails stay on their steel and concrete supports).
        assert_eq!(unconverged, 0.0, "{file} has no static equilibrium");
        assert_eq!(o.values["fragments"], 1.0, "{file} does not stand");
        assert!(o.values["broken_bonds"] <= 0.005 * b_bonds, "{file}: {} joints fail under self-weight", o.values["broken_bonds"]);
    }
}

/// Asset bonds whose shared face has no width (faces meeting along an edge) are left out
/// on import and counted; before, the whole pack failed validation. Packs without such
/// bonds import byte for byte as before.
#[test]
fn scene_packs_leave_out_bonds_without_width() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs");
    let path = dir.join("petronas.json");
    if !path.exists() {
        eprintln!("{} not found; skipping", path.display());
        return;
    }
    let s = stress_ref::scene_pack::import(&path, "petronas", "structure").unwrap();
    assert!(s.description.contains("(11 asset bonds left out: their shared face has no width)"), "{}", s.description);
    assert!(s.bodies[0].bonds.iter().all(|b| b.area > 0.0 && b.width[0] > 0.0 && b.width[1] > 0.0));
}
