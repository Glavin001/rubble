//! Wrecking Yard structures (`stress_ref::wrecking`): their chunk and bond counts and
//! total masses from the layouts' own geometry, the scene packs they are written as
//! (imported back under the lab profiles' options: every chunk, bond and kilogram kept,
//! nothing left out), and that each stands under its own weight in the reference.
//!
//!   cargo test --release -p stress-ref --test wrecking

use stress_ref::capability::Profile;
use stress_ref::scene::Support;
use stress_ref::wrecking::{self, BrickLayout, FrameLayout, RoundLayout, BALLS, STRUCTURES};

/// Area of a regular n-gon of circumradius r.
fn polygon(n: usize, r: f64) -> f64 {
    0.5 * n as f64 * r * r * (2.0 * std::f64::consts::PI / n as f64).sin()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// Per storey: column-joint 16, joint-beam 48 (each span's two ends), beam halves at
/// midspan 24, slab quarters to the beams 72 (two sides each) and to each other 36 (four
/// a bay); and 16 columns to the joints (or footings) below. The core's wall halves: to
/// each other 4, to the columns at their ends 8, to the beams over them 8 and under them
/// 8 (on the ground storey, to their strip footings 8 and the pad footings they reach 8).
#[test]
fn frame10_counts_and_mass() {
    let s = wrecking::frame10();
    type L = FrameLayout;
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 1264);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 16 + 8);
    assert_eq!(s.bonds.len(), L::STOREYS * (16 + 48 + 24 + 72 + 36 + 16 + 4 + 8 + 8 + 8) + 8);
    // Concrete volume a storey: columns, joints, beams (24 spans), slabs (9 bays), the
    // core's four walls.
    let span = L::BAY - L::COLUMN;
    let storey = 16.0 * L::COLUMN * L::COLUMN * (L::STOREY - L::BEAM_DEPTH) + 16.0 * L::COLUMN * L::COLUMN * L::BEAM_DEPTH + 24.0 * span * L::BEAM_WIDTH * L::BEAM_DEPTH + 9.0 * span * span * L::SLAB + 4.0 * span * L::WALL * (L::STOREY - L::BEAM_DEPTH);
    let expected = 2400.0 * L::STOREYS as f64 * storey;
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!(rel(total, expected) < 1e-12, "{total} kg vs {expected} kg");
    // About 3,380 t.
    assert!((3.3e6..3.45e6).contains(&total), "{total}");
}

/// Per floor: core to the inner ring 16, inner to outer ring 16, neighbours within each
/// ring 16 + 16, columns to the slab above 8 and below (or their footings) 8, the core to
/// the segment (or footing) below 1.
#[test]
fn round_tower_counts_and_mass() {
    let s = wrecking::round_tower();
    type L = RoundLayout;
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 337);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 1 + L::COLUMNS);
    assert_eq!(s.bonds.len(), L::FLOORS * (16 + 16 + 16 + 16 + 8 + 8 + 1));
    let n = L::SECTORS;
    let floor = polygon(n, L::CORE) * L::STOREY + (polygon(n, L::EDGE) - polygon(n, L::CORE)) * L::SLAB + L::COLUMNS as f64 * polygon(8, L::COLUMN) * (L::STOREY - L::SLAB);
    let expected = 2400.0 * L::FLOORS as f64 * floor;
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!(rel(total, expected) < 1e-9, "{total} kg vs {expected} kg");
}

/// Per course: 7 + 7 + 6 + 6 joints along the walls and one at each corner; between
/// courses every brick sits on two (running bond: no joint above a joint, corners
/// included), 60 an interface; the first course on its footing brick by brick, 30.
#[test]
fn brick_tower_counts_and_mass() {
    let s = wrecking::brick_tower();
    type L = BrickLayout;
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 1444);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 4);
    assert_eq!(s.bonds.len(), L::COURSES * 30 + (L::COURSES - 1) * 60 + 30);
    assert!(s.bonds.iter().all(|b| b.material == "brick"), "mortar joints throughout");
    let expected = 1900.0 * (L::COURSES * L::PER_COURSE) as f64 * L::BRICK * L::WALL * L::COURSE;
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!(rel(total, expected) < 1e-12, "{total} kg vs {expected} kg");
}

/// Every structure stands at or above grade, its bonds join touching faces over an area,
/// and its line of attack is horizontal.
#[test]
fn structures_are_well_formed() {
    let keys: Vec<&str> = STRUCTURES.iter().map(|s| s.0).collect();
    assert_eq!(keys, ["wy_frame10", "wy_round_tower", "wy_brick_tower", "topple_tower"]);
    for &(key, title) in STRUCTURES {
        let s = wrecking::structure(key).expect(key);
        assert_eq!((s.key, s.title), (key, title));
        let (lo, _) = wrecking::bounds(&s.chunks);
        assert!(lo[2] >= -1e-12, "{key}: below grade at {}", lo[2]);
        for b in &s.bonds {
            assert!(b.area > 0.0 && b.width[0] > 0.0 && b.width[1] > 0.0, "{key}: bond {}-{} has no section", b.a, b.b);
            assert!(s.chunks[b.a].support == Support::None || s.chunks[b.b].support == Support::None, "{key}: a bond between two supports");
        }
        assert_eq!(s.aim.direction[2], 0.0, "{key}");
    }
}

/// The packs import back (both lab profiles' options: the pack's own fatal limits, masses
/// and modulus) as the structures were built: every chunk, bond and support, the same
/// mass, no import notes, and the calibrated strengths on the joints. The line of attack
/// survives the round trip.
#[test]
fn packs_import_as_built() {
    for &(key, _) in STRUCTURES {
        let s = wrecking::structure(key).expect(key);
        let pack = wrecking::to_pack(&s);
        assert_eq!(wrecking::aim_of(&pack), Some(s.aim), "{key}");
        for profile in [Profile::BlastParity, Profile::Threshold] {
            let (scene, notes) = stress_ref::scene_pack::import_value_with(&pack, key, "structure", key, profile.import_options()).unwrap_or_else(|e| panic!("{key}: {e}"));
            assert!(notes.is_empty(), "{key}: {notes:?}");
            let body = &scene.bodies[0];
            assert_eq!(body.chunks.len(), s.chunks.len(), "{key}");
            assert_eq!(body.bonds.len(), s.bonds.len(), "{key}");
            let fixed = |c: &[stress_ref::scene::ChunkDesc]| c.iter().filter(|c| c.support == Support::Fixed).count();
            assert_eq!(fixed(&body.chunks), fixed(&s.chunks), "{key}");
            let derived = scene.with_derived();
            let mass: f64 = derived.bodies[0].chunks.iter().filter(|c| c.support == Support::None).map(|c| derived.materials[&c.material].density * wrecking::volume(c)).sum();
            let built: f64 = s.chunks.iter().map(wrecking::mass).sum();
            assert!(rel(mass, built) < 1e-9, "{key}: {mass} kg imported, {built} kg built");
            for b in &body.bonds {
                let m = &scene.materials[&b.material];
                let calibrated = if key == "wy_brick_tower" { stress_ref::material::Material::mortar() } else { stress_ref::material::Material::concrete() };
                assert_eq!((m.tensile_strength, m.compressive_strength, m.cohesion, m.youngs_modulus), (calibrated.tensile_strength, calibrated.compressive_strength, calibrated.cohesion, calibrated.youngs_modulus), "{key}: {}", b.material);
                assert_eq!(b.rebar.is_some(), key != "wy_brick_tower", "{key}: concrete joints carry the code-minimum reinforcement");
            }
        }
    }
}

/// Each structure stands under its own weight in the reference, under the lab's
/// profiles' scene settings (quasi-static, Blast's model): after a quarter second no bond
/// has broken, nothing moves and the supports carry the whole weight.
#[test]
fn structures_stand_under_self_weight() {
    for &(key, _) in STRUCTURES {
        let pack = wrecking::pack(key).expect(key);
        let profile = Profile::BlastParity;
        let (mut scene, _) = stress_ref::scene_pack::import_value_with(&pack, key, "structure", key, profile.import_options()).expect("import");
        scene.sim.duration = 0.25;
        let mut scene = scene.with_derived();
        scene.sim.methods = stress_ref::scene::Methods::established();
        profile.apply_scene(&mut scene);
        let mut w = stress_ref::world::World::new(&scene);
        let t = std::time::Instant::now();
        w.run();
        let chunks = w.solver.structures[0].chunks.len();
        let vmax = (0..chunks).map(|c| w.solver.chunk_velocity(0, c).0.norm()).fold(0.0, f64::max);
        // A support no bond joins (the Topple street's tiles) rests on the ground alone.
        let weight: f64 = (0..chunks).filter(|&c| !w.solver.clusters[w.solver.chunks[0][c].cluster].bonds.is_empty()).map(|c| w.solver.structures[0].chunks[c].mass * 9.81).sum();
        let reaction: f64 = (0..chunks).map(|c| w.solver.reaction_world(0, c).0.z).sum();
        println!("{key}: {chunks} chunks, {:.0} t, max speed {vmax:.2e} m/s, reaction / weight {:.9}, {:.2} s", weight / 9.81e3, reaction / weight, t.elapsed().as_secs_f64());
        assert_eq!(w.solver.broken_bond_count(), 0, "{key}: bonds broke under self-weight");
        assert!(vmax < 1e-3, "{key}: max chunk speed {vmax} m/s");
        assert!(rel(reaction, weight) < 1e-4, "{key}: {reaction} N vs {weight} N");
    }
}

/// The wrecking balls are solid steel at the calibrated density; the demolition sphere
/// is a 15 m shell 5 cm thick that rolls in on the ground without slipping.
#[test]
fn ball_presets() {
    let steel = stress_ref::material::Material::steel().density;
    for b in &BALLS[..2] {
        let solid = 4.0 / 3.0 * std::f64::consts::PI * b.radius.powi(3) * steel;
        assert!(rel(solid, b.mass) < 1e-3, "{}: {solid} kg at r = {}", b.key, b.radius);
        assert!(!b.rolling);
    }
    let sphere = wrecking::ball("sphere15m").expect("demolition sphere");
    assert_eq!(2.0 * sphere.radius, 15.0);
    assert!(sphere.rolling && (sphere.mass - 277_442.0).abs() < 1.0, "{}", sphere.mass);
    let s = wrecking::frame10();
    let shot = wrecking::launch(sphere, &s.aim, "sphere");
    assert_eq!(shot.position[2], sphere.radius, "on the ground");
    // No slip at the contact point: v + w x (-r z) = 0.
    let (v, w, r) = (shot.velocity, shot.angular_velocity, sphere.radius);
    let slip = [v[0] - w[1] * r, v[1] + w[0] * r, v[2]];
    assert!(slip.iter().all(|s| s.abs() < 1e-12), "{slip:?}");
    let ball = wrecking::launch(&BALLS[0], &s.aim, "ball");
    assert!(ball.velocity[2] > 0.0 && ball.position[0] < s.aim.target[0] - BALLS[0].radius);
}
