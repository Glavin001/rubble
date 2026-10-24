//! Bonded vehicles (`stress_ref::vehicles`): the car, bus and airliner as built (parts,
//! masses, size, one bonded island each, the joints that tear first), and the scenes that
//! place them: their packs import as built in both lab profiles, the vehicles free islands
//! with their launches, and the vehicles stand on their wheels in the reference.
//!
//!   cargo test --release -p stress-ref --test vehicles

use stress_ref::capability::Profile;
use stress_ref::scene::{ChunkDesc, Support};
use stress_ref::vehicles::{self, AirlinerLayout, BusLayout, CarLayout, Placement, Vehicle};
use stress_ref::wrecking;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn kind(c: &ChunkDesc) -> &str {
    c.groups.first().map(String::as_str).unwrap_or("")
}

/// The vehicle placed at the origin, with its bonds.
fn built(v: &Vehicle) -> (Vec<ChunkDesc>, Vec<stress_ref::scene::BondDesc>) {
    let s = vehicles::compose("v", "v", empty(), Vec::new(), &[(v.clone(), Placement { at: [0.0; 3], heading: 0.0 }, None)]);
    (s.structure.chunks, s.structure.bonds)
}

fn empty() -> wrecking::Structure {
    wrecking::Structure { key: "v", title: "v", chunks: Vec::new(), bonds: Vec::new(), aim: wrecking::Aim { target: [0.0; 3], direction: [1.0, 0.0, 0.0] }, ball: wrecking::BALLS[0], charges: Vec::new() }
}

#[test]
fn vehicles_as_built() {
    let cases: [(Vehicle, usize, f64, [f64; 3]); 3] = [
        (vehicles::car("car"), 50, CarLayout::MASS, [CarLayout::LENGTH, CarLayout::WIDTH, CarLayout::HEIGHT]),
        (vehicles::bus("bus"), 87, 10_420.0, [BusLayout::LENGTH + 0.4, BusLayout::WIDTH, BusLayout::HEIGHT]),
        (vehicles::airliner("airliner"), 36, 34_700.0, [AirlinerLayout::LENGTH, AirlinerLayout::SPAN, 0.0]),
    ];
    for (v, parts, total, size) in cases {
        let (chunks, bonds) = built(&v);
        let (lo, hi) = wrecking::bounds(&chunks);
        let ext: Vec<f64> = (0..3).map(|k| hi[k] - lo[k]).collect();
        println!("{}: {} parts, {} bonds, {:.0} kg, {:.2} x {:.2} x {:.2} m", v.name, chunks.len(), bonds.len(), v.mass(), ext[0], ext[1], ext[2]);
        assert_eq!(chunks.len(), parts, "{}", v.name);
        assert_eq!(v.masses.len(), parts);
        assert!(v.masses.iter().all(|&m| m > 0.0), "{}: every part has a mass", v.name);
        assert!(rel(v.mass(), total) < 1e-9, "{}: {} kg", v.name, v.mass());
        for k in 0..3 {
            if size[k] > 0.0 {
                assert!((ext[k] - size[k]).abs() < 1e-9, "{}: {ext:?} vs {size:?}", v.name);
            }
        }
        // One bonded island of hull parts, every bond over an area, of the joint the rule gives.
        assert_eq!(stress_ref::launch::free_islands(&chunks, &bonds), vec![(0..parts).collect::<Vec<_>>()], "{}: one free island", v.name);
        assert!(chunks.iter().all(|c| c.hull.is_some() && c.support == Support::None));
        for b in &bonds {
            assert!(b.area > 0.0 && b.width[0] > 0.0 && b.width[1] > 0.0, "{}: bond {}-{}", v.name, b.a, b.b);
            assert_eq!(b.material, vehicles::joint(&chunks[b.a], &chunks[b.b]));
        }
        if v.name != "airliner" {
            // A road vehicle stands on its wheels, which are bolted on.
            assert!(lo[2].abs() < 1e-9, "{}: on the ground", v.name);
            let wheels: Vec<usize> = (0..parts).filter(|&k| kind(&chunks[k]) == "wheel").collect();
            assert_eq!(wheels.len(), 4);
            for &k in &wheels {
                let low = chunks[k].center[2] - chunks[k].half_extents[2];
                assert!(low.abs() < 1e-9, "{}: a wheel off the ground", v.name);
            }
            // No other part reaches the ground.
            assert!((0..parts).filter(|k| !wheels.contains(k)).all(|k| chunks[k].center[2] - chunks[k].half_extents[2] > 0.04), "{}", v.name);
        } else {
            assert!(bonds.iter().all(|b| b.material == "riveted"));
        }
    }
    // The car's doors and wheels hang on bolted joints, weaker than the welded shell in
    // every mode; each door on its hinge and latch pillars (FMVSS 206's 11 kN each), each
    // wheel holding several times its share of the car's weight.
    let (bolted, weld) = (wrecking::calibrated("bolted").1, wrecking::calibrated("spot-weld").1);
    assert!(bolted.tensile_strength < weld.tensile_strength && bolted.cohesion < weld.cohesion && bolted.compressive_strength < weld.compressive_strength);
    let (chunks, bonds) = built(&vehicles::car("car"));
    let capacity = |b: &stress_ref::scene::BondDesc| wrecking::calibrated(&b.material).1.tensile_strength * b.area;
    for part in ["door", "wheel"] {
        for k in (0..chunks.len()).filter(|&k| kind(&chunks[k]) == part) {
            // (A door's window is glazed into it: that joint is the glass's.)
            let held: Vec<&stress_ref::scene::BondDesc> = bonds.iter().filter(|b| (b.a == k || b.b == k) && b.material != "glass").collect();
            assert!(held.iter().all(|b| b.material == "bolted"), "{part} {k}");
            let total: f64 = held.iter().map(|b| capacity(b)).sum();
            if part == "door" {
                assert_eq!(held.len(), 2, "door {k}: its hinge and latch pillars");
                assert!(held.iter().all(|b| (capacity(b) - 11_000.0).abs() < 1_500.0), "door {k}: {:?} N", held.iter().map(|b| capacity(b)).collect::<Vec<_>>());
            } else {
                println!("wheel {k}: held by {total:.0} N");
                assert!(total > 3.0 * CarLayout::MASS * 9.81 / 4.0, "wheel {k}: {total} N");
            }
        }
    }
}

/// The scenes' packs import as built (both profiles): every chunk and bond, the vehicles'
/// masses, each vehicle one free island with its launch, the structure's own chunks and
/// bonds first and unchanged.
#[test]
fn scenes_import_as_built() {
    for &(key, title) in vehicles::SCENES {
        let s = vehicles::scene(key).expect(key);
        assert_eq!((s.structure.key, s.structure.title), (key, title));
        assert_eq!(wrecking::structure(key).map(|x| x.chunks.len()), Some(s.structure.chunks.len()), "{key}: the Scene menu's lookup");
        let pack = s.to_pack();
        assert_eq!(wrecking::pack(key).as_ref(), Some(&pack), "{key}");
        let launches = stress_ref::launch::launches_of(&pack).expect("launches");
        assert_eq!(launches, s.launches, "{key}");
        let (lo, _) = wrecking::bounds(&s.structure.chunks);
        assert!(lo[2] >= -1e-12, "{key}: below grade");
        for profile in [Profile::BlastParity, Profile::Threshold] {
            let (scene, notes) = stress_ref::scene_pack::import_value_with(&pack, key, "structure", key, profile.import_options()).unwrap_or_else(|e| panic!("{key}: {e}"));
            assert!(notes.is_empty(), "{key}: {notes:?}");
            let body = &scene.bodies[0];
            assert_eq!(body.chunks.len(), s.structure.chunks.len());
            assert_eq!(body.bonds.len(), s.structure.bonds.len());
            let derived = scene.with_derived();
            for (k, c) in derived.bodies[0].chunks.iter().enumerate() {
                if c.support == Support::None {
                    let m = derived.materials[&c.material].density * wrecking::volume(c);
                    assert!(rel(m, s.masses[k]) < 1e-9, "{key}: chunk {k} {m} kg vs {} kg", s.masses[k]);
                } else {
                    assert_eq!(s.masses[k], 0.0);
                }
            }
            let free = stress_ref::launch::free_islands(&body.chunks, &body.bonds);
            for (name, ids) in &s.vehicles {
                assert!(free.contains(ids), "{key}: {name} is one free island");
            }
            for l in &launches {
                stress_ref::launch::check_island(l, &body.chunks, &body.bonds).unwrap_or_else(|e| panic!("{key}: {e}"));
            }
        }
    }
}

/// The car stands on its wheels in the reference (the lab profiles' quasi-static scene,
/// Blast's model), unlaunched: after a quarter second no bond has broken and it barely
/// moves. (The bus rocks on its wheels in the reference's own contact and loses them after
/// half a second; on the GPU both stand still in both stress modes, stress-rubble
/// `tests/vehicles.rs`.)
#[test]
fn car_stands_on_its_wheels_in_the_reference() {
    let key = "vh_car_barrier";
    let mut pack = vehicles::pack(key).unwrap();
    pack["defaults"].as_object_mut().unwrap().remove("launches");
    let profile = Profile::BlastParity;
    let (mut scene, _) = stress_ref::scene_pack::import_value_with(&pack, key, "structure", key, profile.import_options()).expect("import");
    scene.sim.duration = 0.25;
    let mut scene = scene.with_derived();
    scene.sim.methods = stress_ref::scene::Methods::established();
    profile.apply_scene(&mut scene);
    let mut w = stress_ref::world::World::new(&scene);
    w.run();
    let chunks = w.solver.structures[0].chunks.len();
    let vmax = (0..chunks).map(|c| w.solver.chunk_velocity(0, c).0.norm()).fold(0.0, f64::max);
    println!("{key}: broken {}, max speed {vmax:.2e} m/s", w.solver.broken_bond_count());
    assert_eq!(w.solver.broken_bond_count(), 0, "{key}: bonds broke under self-weight");
    assert!(vmax < 1e-3, "{key}: max chunk speed {vmax} m/s");
}
