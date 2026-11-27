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
        // A car's load paths; the tractor unit's own (a cab-over on a ladder frame: no hood
        // and fender path, no rear body; the frame and what it carries are `Other`).
        // The semi-trailer's: its rear impact guard (`Rear`), axles and wheels, the rest `Other`.
        let need: &[Section] = if *design == "truck_cab" {
            &[Section::FrontLower, Section::Cabin, Section::Wheel, Section::Other]
        } else if *design == "box_trailer" {
            &[Section::Rear, Section::Wheel, Section::Other]
        } else {
            &[Section::FrontLower, Section::FrontUpper, Section::Cabin, Section::Rear, Section::Wheel]
        };
        for &s in need {
            assert!(by.get(&s).copied().unwrap_or(0) > 0, "{design}: no node in {s:?}");
        }
    }
}

#[test]
fn every_designs_gauges_are_node_pairs_of_a_plausible_length() {
    // (m) a car's: wheelbase, a front door's opening, the cabin between the B-pillars at
    // the belt, sill to roof rail, the front bumper's face to the front axle line
    let car: &[(&str, f32, f32)] = &[
        ("wheelbase_left", 2.3, 3.2),
        ("wheelbase_right", 2.3, 3.2),
        ("door_left", 0.8, 1.5),
        ("door_right", 0.8, 1.5),
        ("cabin_width", 1.2, 1.9),
        ("roof_height", 0.8, 1.4),
        ("front_overhang", 0.7, 1.5),
    ];
    // (m) the 4x2 tractor unit's (`truck_cab`: wheelbase 3.80 m, a 2.5 m wide cab 1.85 m
    // from floor to roof, a 5.7 m frame, the bumper's face 1.365 m ahead of the front axle):
    // wheelbase, the cab's width at mid height, its floor to roof on the left, the left
    // rail's length, the bumper's face to the left rail at the front axle's line
    let truck: &[(&str, f32, f32)] = &[
        ("wheelbase_left", 3.5, 4.1),
        ("wheelbase_right", 3.5, 4.1),
        ("cab_width", 2.2, 2.7),
        ("cab_height", 1.5, 2.1),
        ("frame_length", 5.3, 6.1),
        ("front_overhang", 1.0, 1.7),
    ];
    // (m) the 53 ft tandem dry van's (`box_trailer`: a 49 in tandem spread, the kingpin
    // 40 ft ahead of the rear axle less the plate's 0.45 m half length, a 2.58 m floor
    // between the walls' nodes, 2.96 m from the floor's underside to the roof, the rails
    // 14.3 m, the guard 0.47 m under the rear sill): the tandem's spread (named as a
    // wheelbase: the gauge picks the foremost and rearmost wheels), the kingpin plate to the
    // rear axle, the box's width and height at mid length, the rail's length, the sill to
    // the guard's lower edge
    let trailer: &[(&str, f32, f32)] = &[
        ("wheelbase_left", 1.1, 1.4),
        ("wheelbase_right", 1.1, 1.4),
        ("kingpin_to_axle", 11.3, 12.3),
        ("box_width", 2.4, 2.6),
        ("box_height", 2.7, 3.1),
        ("frame_length", 13.5, 15.5),
        ("guard_drop", 0.35, 0.65),
    ];
    let pos_y = |t: &Template, i: u32| t.desc.nodes[i as usize].position[1];
    for (design, t) in templates() {
        let pos = |n: &str| -> Vec3 {
            let i = t.desc.nodes.iter().position(|x| x.name == n).unwrap_or_else(|| panic!("{design}: gauge node {n} is not in the JBeam"));
            Vec3::from(t.desc.nodes[i].position)
        };
        let mut line = Vec::new();
        if *design == "box_trailer" {
            // FMVSS 224 S5.1.2: the guard's lower edge at most 560 mm over the ground (the
            // template's ground: its wheels' lowest point), across its whole width
            let low = t.desc.wheels.iter().map(|w| 0.5 * (pos_y(t, w.node1) + pos_y(t, w.node2)) - w.radius).fold(f32::MAX, f32::min);
            let guard = t.desc.nodes.iter().filter(|n| n.name.starts_with("gb")).map(|n| n.position[1]).fold(f32::MAX, f32::min);
            line.push(format!("guard's lower edge {:.3} m over the ground", guard - low));
            assert!(guard - low <= 0.56, "{design}: the guard's lower edge stands {:.3} m over the ground (FMVSS 224: at most 0.56)", guard - low);
        }
        let bands = if *design == "truck_cab" {
            truck
        } else if *design == "box_trailer" {
            trailer
        } else {
            car
        };
        for &(name, lo, hi) in bands {
            let g = t.zones.gauges.iter().find(|g| g.name == name).unwrap_or_else(|| panic!("{design}: no gauge {name}"));
            let (a, b) = (pos(&g.a), pos(&g.b));
            let l = a.distance(b);
            line.push(format!("{name} {} - {} {l:.3}", g.a, g.b));
            assert!((lo..=hi).contains(&l), "{design}: {name} {} - {} is {l:.3} m, outside {lo}-{hi}", g.a, g.b);
            // left is +X in the car's frame (forward +Z, up +Y)
            if name.ends_with("_left") || name == "roof_height" || name == "cab_height" {
                assert!(a.x > 0.0 && b.x > 0.0, "{design}: {name} is not on +X: {a} {b}");
            }
            if name.ends_with("_right") {
                assert!(a.x < 0.0 && b.x < 0.0, "{design}: {name} is not on -X: {a} {b}");
            }
            if name.starts_with("wheelbase") {
                assert!(a.z > b.z, "{design}: {name} does not run front to rear (+Z forward)");
            }
            if name == "roof_height" || name == "cab_height" {
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
