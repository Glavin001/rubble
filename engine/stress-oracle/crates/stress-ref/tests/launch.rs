//! Launched bonded bodies (`stress_ref::launch`): a pack's `defaults.launches` round trip,
//! and a free island imports as its own body in both lab profiles.
//!
//!   cargo test --release -p stress-ref --test launch

use serde_json::json;
use stress_ref::capability::Profile;
use stress_ref::launch::{self, Launch, Strike};

#[test]
fn launches_round_trip_through_a_pack() {
    let pack = launch::ram_and_wall();
    let launches = launch::launches_of(&pack).expect("launches");
    assert_eq!(launches.len(), 1);
    let ram = &launches[0];
    assert_eq!((ram.name.as_str(), ram.chunks.as_slice(), ram.delay), ("ram", &[17usize, 18, 19, 20, 21][..], 0.0));
    // Z-up as built: along +y at 30 m/s, a little lift, no spin.
    assert_eq!(ram.velocity[1], 30.0);
    assert!(ram.velocity[2] > 0.0 && ram.velocity[0] == 0.0);
    assert_eq!(ram.angular_velocity, [0.0; 3]);
    assert_eq!(ram.strikes.len(), 1);
    // Written again from what was read: the same JSON.
    let mut again = pack.clone();
    launch::write(&mut again, &launches);
    assert_eq!(again["defaults"]["launches"], pack["defaults"]["launches"]);
    // Y-up in the pack: +y (Z-up) is -z there.
    assert_eq!(pack["defaults"]["launches"][0]["velocity"]["z"], -30.0);
    // The ram is one whole free island; the wall is anchored.
    let (scene, _) = stress_ref::scene_pack::import_value_with(&pack, "ram", "structure", "ram", Profile::Threshold.import_options()).expect("import");
    let body = &scene.bodies[0];
    assert_eq!(launch::free_islands(&body.chunks, &body.bonds), vec![ram.chunks.clone()]);
    launch::check_island(ram, &body.chunks, &body.bonds).expect("one free island");
}

#[test]
fn a_launch_by_group_and_malformed_launches() {
    let mut pack = launch::ram_and_wall();
    let groups: Vec<serde_json::Value> = (0..22).map(|i| json!(if i >= 17 { "car" } else { "" })).collect();
    pack["scenario"]["nodeGroups"] = json!(groups);
    pack["defaults"]["launches"] = json!([{"name": "car", "group": "car", "velocity": {"x": 1, "y": 0, "z": 0}, "delay": 0.5}]);
    let l = launch::launches_of(&pack).expect("by group");
    assert_eq!(l[0].chunks, vec![17, 18, 19, 20, 21]);
    assert_eq!((l[0].velocity, l[0].delay, l[0].strikes.len()), ([1.0, 0.0, 0.0], 0.5, 0));
    pack["defaults"]["launches"] = json!([{"name": "car", "nodes": [3, 99]}]);
    let err = launch::launches_of(&pack).unwrap_err();
    assert!(err.contains("node 99"), "{err}");
    pack["defaults"]["launches"] = json!([{"name": "car"}]);
    assert!(launch::launches_of(&pack).unwrap_err().contains("no nodes"));
    // An island that is not whole, or not free.
    let (scene, _) = stress_ref::scene_pack::import_value_with(&launch::ram_and_wall(), "ram", "structure", "ram", Profile::Threshold.import_options()).expect("import");
    let b = &scene.bodies[0];
    let part = Launch { name: "half".into(), chunks: vec![17, 18], velocity: [0.0; 3], angular_velocity: [0.0; 3], delay: 0.0, strikes: vec![Strike { centre: [0.0; 3], radius: 1.0 }] };
    assert!(launch::check_island(&part, &b.chunks, &b.bonds).is_err());
    let wall = Launch { chunks: (1..17).collect(), ..part };
    assert!(launch::check_island(&wall, &b.chunks, &b.bonds).is_err());
}

/// Both lab profiles' worlds make the ram its own free body (a cluster of its three chunks,
/// not anchored) and the wall one anchored body.
#[test]
fn a_free_island_is_its_own_body() {
    let pack = launch::ram_and_wall();
    for profile in [Profile::BlastParity, Profile::Threshold] {
        let (scene, _) = stress_ref::scene_pack::import_value_with(&pack, "ram", "structure", "ram", profile.import_options()).expect("import");
        let mut scene = scene.with_derived();
        profile.apply_scene(&mut scene);
        let w = stress_ref::world::World::new(&scene);
        let m = &w.solver;
        assert_eq!(m.clusters.len(), 2, "{profile:?}");
        let ram = m.clusters.iter().find(|c| c.chunks == [17, 18, 19, 20, 21]).expect("the ram's cluster");
        assert!(!ram.anchored);
        assert!((ram.mass - 1500.0).abs() < 1e-6, "{}", ram.mass);
        assert!(m.clusters.iter().any(|c| c.anchored && c.chunks.len() == 17));
    }
}
