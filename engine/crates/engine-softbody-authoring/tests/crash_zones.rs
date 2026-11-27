//! Every generated design's `crash.json` (`authoring::gen2::crash_zones`), as the crash
//! reference loads it (`crash_templates::template`): its prefixes give the JBeam's nodes
//! their zones, every load path is present, and its gauges are real node pairs of a
//! plausible length, left on +X.
//!
//! Run: `cargo test --release -p engine-softbody-authoring --test crash_zones -- --nocapture`

use engine_softbody::crash_reference::{Section, Template};
use engine_softbody_authoring::authoring::gen2::crash_zones::lookup;
use engine_softbody_authoring::crash_templates::template;
use glam::Vec3;
use std::collections::BTreeMap;
use std::sync::OnceLock;

fn templates() -> &'static Vec<(&'static str, Template)> {
    static T: OnceLock<Vec<(&'static str, Template)>> = OnceLock::new();
    T.get_or_init(|| {
        engine_vehiclegen::design_titles()
            .into_iter()
            .map(|(d, _)| (d, template(d).unwrap_or_else(|e| panic!("{d}: {e}"))))
            .collect()
    })
}

#[test]
fn every_designs_zones_cover_its_nodes_and_every_load_path() {
    for (design, t) in templates() {
        let z = &t.zones;
        assert!(!z.parts.is_empty(), "{design}: no crash.json");
        let mut by: BTreeMap<Section, usize> = BTreeMap::new();
        let mut unmatched = Vec::new();
        for n in &t.desc.nodes {
            match lookup(&z.parts, &n.name) {
                Some(s) => *by.entry(s).or_default() += 1,
                None => unmatched.push(n.name.clone()),
            }
        }
        let total = t.desc.nodes.len();
        let share = 1.0 - unmatched.len() as f64 / total as f64;
        println!("{design}: {} prefixes, {:.1} % of {total} nodes matched; {by:?}", z.parts.len(), 100.0 * share);
        assert!(share >= 0.95, "{design}: {:.1} % matched; unmatched {:?}", 100.0 * share, &unmatched[..unmatched.len().min(20)]);
        for s in [Section::FrontLower, Section::FrontUpper, Section::Cabin, Section::Rear, Section::Wheel] {
            assert!(by.get(&s).copied().unwrap_or(0) > 0, "{design}: no node in {s:?}");
        }
    }
}

#[test]
fn every_designs_gauges_are_node_pairs_of_a_plausible_length() {
    // (m) a car's: wheelbase, a front door's opening, the cabin between the B-pillars at
    // the belt, sill to roof rail, the front bumper's face to the front axle line
    let bands: [(&str, f32, f32); 7] = [
        ("wheelbase_left", 2.3, 3.2),
        ("wheelbase_right", 2.3, 3.2),
        ("door_left", 0.8, 1.5),
        ("door_right", 0.8, 1.5),
        ("cabin_width", 1.2, 1.9),
        ("roof_height", 0.8, 1.4),
        ("front_overhang", 0.7, 1.5),
    ];
    for (design, t) in templates() {
        let pos = |n: &str| -> Vec3 {
            let i = t.desc.nodes.iter().position(|x| x.name == n).unwrap_or_else(|| panic!("{design}: gauge node {n} is not in the JBeam"));
            Vec3::from(t.desc.nodes[i].position)
        };
        let mut line = Vec::new();
        for (name, lo, hi) in bands {
            let g = t.zones.gauges.iter().find(|g| g.name == name).unwrap_or_else(|| panic!("{design}: no gauge {name}"));
            let (a, b) = (pos(&g.a), pos(&g.b));
            let l = a.distance(b);
            line.push(format!("{name} {} - {} {l:.3}", g.a, g.b));
            assert!((lo..=hi).contains(&l), "{design}: {name} {} - {} is {l:.3} m, outside {lo}-{hi}", g.a, g.b);
            // left is +X in the car's frame (forward +Z, up +Y)
            if name.ends_with("_left") || name == "roof_height" {
                assert!(a.x > 0.0 && b.x > 0.0, "{design}: {name} is not on +X: {a} {b}");
            }
            if name.ends_with("_right") {
                assert!(a.x < 0.0 && b.x < 0.0, "{design}: {name} is not on -X: {a} {b}");
            }
            if name.starts_with("wheelbase") {
                assert!(a.z > b.z, "{design}: {name} does not run front to rear (+Z forward)");
            }
            if name == "roof_height" {
                assert!(b.y > a.y + 0.5, "{design}: roof_height does not run sill to roof (+Y up): {a} {b}");
            }
            if name == "front_overhang" {
                // its rear node stands on the front axle's line (the left wheel's axle node)
                let wb = t.zones.gauges.iter().find(|g| g.name == "wheelbase_left").unwrap();
                let axle = pos(&wb.a);
                assert!((b.z - axle.z).abs() < 0.15 && a.z > axle.z + 0.6, "{design}: front_overhang {a} {b}, front axle at z {}", axle.z);
                line.push(format!("(along the car {:.3})", a.z - axle.z));
            }
        }
        println!("{design}: {}", line.join("; "));
    }
}
