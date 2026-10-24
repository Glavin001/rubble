//! Wrecking Yard structures (`stress_ref::wrecking`, with Bridge Out's viaduct,
//! `stress_ref::viaduct`, and the Ram Raid and Fly-through buildings of
//! `stress_ref::buildings`): their chunk and bond counts and total masses from the layouts'
//! own geometry, the scene packs they are written as (imported back under the lab profiles'
//! options: every chunk, bond and kilogram kept, nothing left out), and that each stands
//! under its own weight in the reference.
//!
//!   cargo test --release -p stress-ref --test wrecking

use stress_ref::buildings::{ApartmentLayout, StorefrontLayout, TowerLayout};
use stress_ref::capability::Profile;
use stress_ref::scene::{ChunkDesc, Support};
use stress_ref::viaduct::{self, ViaductLayout};
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

/// Per span: deck joints along 5 x 6 and across 6 x 5, parapets on the deck 12 and to
/// each other 10. A seat pier: footing-column, 4 column joints, column-cap, 2 cap joints,
/// 12 cap-bearing, 12 bearing-deck, the link to the spans' ends 12 and across itself 5.
/// The hinge pier: footing-column, 4 column joints, column-cap, 2 cap joints, 6
/// cap-bearing, 6 bearing-diaphragm, 5 across the diaphragm, 6 diaphragm-link, 5 across
/// the link and its nibs to the spans 12. An abutment: 6 seat-bearing, 6 bearing-deck. The
/// road: 30 slabs on their sub-bases, 2 x 10 joints along x and 3 x 9 along y (none
/// between the sub-bases: both are supports).
#[test]
fn viaduct_counts_and_mass() {
    let s = viaduct::viaduct();
    type L = ViaductLayout;
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 347);
    // Supports: 3 pad footings, 2 abutments, the road's 30 sub-bases.
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 3 + 2 + 30);
    let span = 30 + 30 + 12 + 10;
    let seat = 1 + 4 + 1 + 2 + 12 + 12 + 12 + 5;
    let hinge = 1 + 4 + 1 + 2 + 6 + 6 + 5 + 6 + 5 + 12;
    let road = 30 + 20 + 27;
    assert_eq!(s.bonds.len(), 4 * span + 2 * seat + hinge + 2 * 12 + road);
    assert_eq!(s.bonds.len(), 575);
    assert!(s.bonds.iter().all(|b| b.material == "reinforced-concrete"));
    // Concrete volume: the deck (the spans' lengths: 15, 14.5, 14.5, 15 m) and its
    // parapets, the columns, the caps (two seat caps, the hinge cap), the bearings (12 on
    // each seat pier, 6 on the hinge pier and on each abutment), the links, the hinge's
    // diaphragm and the road's slabs.
    let deck: f64 = (0..L::SPANS).map(|k| L::span(k).1 - L::span(k).0).sum();
    assert!(rel(deck, 59.0) < 1e-12, "{deck}");
    let volume = deck * (L::WIDTH * L::DECK + 2.0 * L::PARAPET_WIDTH * L::PARAPET_HEIGHT)
        + 3.0 * L::COLUMN * L::COLUMN * (L::cap_bottom() - L::FOOTING_DEPTH)
        + (2.0 * L::SEAT_CAP + L::HINGE_CAP) * L::WIDTH * L::CAP_DEPTH
        + (2 * 12 + 6 + 2 * 6) as f64 * L::BEARING.iter().product::<f64>()
        + 2.0 * L::LINK * L::WIDTH * L::LINK_DEPTH
        + L::HINGE_GAP * L::WIDTH * L::NIB
        + L::DIAPHRAGM * L::WIDTH * (L::DECK - L::NIB)
        + (L::ROAD_ALONG * L::ROAD_ACROSS) as f64 * L::ROAD_SLAB * L::ROAD_SLAB * L::ROAD_DEPTH;
    let expected = 2400.0 * volume;
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!(rel(total, expected) < 1e-12, "{total} kg vs {expected} kg");
    // About 1,890 t.
    assert!((1.85e6..1.95e6).contains(&total), "{total}");
    // Every chunk at or above grade, the deck's top 10 m up, the road at grade under span 2.
    assert!(s.chunks.iter().all(|c| c.center[2] - c.half_extents[2] >= -1e-12));
    let (lo, hi) = wrecking::bounds(&s.chunks);
    assert_eq!(lo[2], 0.0);
    assert!((hi[2] - (L::DECK_TOP + L::PARAPET_HEIGHT)).abs() < 1e-12);
    let road: Vec<_> = s.chunks.iter().filter(|c| c.groups[0] == "road").collect();
    let (x0, x1) = L::span(1);
    assert!(road.iter().all(|c| c.center[0] - c.half_extents[0] > x0 && c.center[0] + c.half_extents[0] < x1));
    // The drop-in spans have no seat on the hinge pier: nothing under their ends there
    // but air down to its cap's level.
    let cap = |p: usize| s.chunks.iter().filter(move |c| c.groups[0] == "cap" && c.groups[1] == format!("pier-{p}"));
    let hinge_x = L::line(L::HINGE_PIER);
    assert!(cap(L::HINGE_PIER).all(|c| (c.center[0] - hinge_x).abs() + c.half_extents[0] < L::end_gap(L::HINGE_PIER)));
    // The seat piers' caps reach a metre under each span's end.
    for p in [1, 3] {
        assert!(cap(p).all(|c| c.half_extents[0] - L::end_gap(p) >= 1.0 - 1e-12));
    }
    // Four spans on the probes, each its 82 bonded pairs and its six end chunks a side.
    let probes = viaduct::span_probes(&s.chunks, &s.bonds);
    assert_eq!(probes.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(), ["span-1", "span-2", "span-3", "span-4"]);
    assert!(probes.iter().all(|p| p.pairs.len() == span && p.west.len() == L::ACROSS && p.east.len() == L::ACROSS));
    // The ball's line: across the deck (+y) into the hinge pier's column face, clear of
    // its footing (the 50 t ball's bottom above the footing's top as it reaches the face).
    assert_eq!(s.aim.direction, [0.0, 1.0, 0.0]);
    assert_eq!(s.aim.target[0], hinge_x);
    assert!(s.aim.target[2] - BALLS[1].radius > L::FOOTING_DEPTH);
}

/// Every structure stands at or above grade, its bonds join touching faces over an area,
/// and its line of attack is horizontal.
#[test]
fn structures_are_well_formed() {
    let keys: Vec<&str> = STRUCTURES.iter().map(|s| s.0).collect();
    assert_eq!(keys, ["wy_frame10", "wy_round_tower", "wy_brick_tower", "bo_viaduct", "topple_tower", "rr_storefront_row", "ft_apartment_block", "ft_glass_tower"]);
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
            // Each bond its own joint material's calibrated strengths (bonds import in order).
            for (b, built) in body.bonds.iter().zip(&s.bonds) {
                assert_eq!((b.a, b.b), (built.a, built.b), "{key}");
                let m = &scene.materials[&b.material];
                let (_, calibrated) = wrecking::calibrated(&built.material);
                assert_eq!((m.tensile_strength, m.compressive_strength, m.cohesion, m.youngs_modulus, m.kind), (calibrated.tensile_strength, calibrated.compressive_strength, calibrated.cohesion, calibrated.youngs_modulus, calibrated.kind), "{key}: {}", b.material);
                assert_eq!(b.rebar.is_some(), built.material == "reinforced-concrete", "{key}: concrete joints, and only they, carry the code-minimum reinforcement");
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
    // The stand-ins fly in level at their target's height; each building's click ball is
    // its stand-in.
    for key in ["car2t", "jet"] {
        assert!(!wrecking::ball(key).expect(key).rolling, "{key}");
    }
    assert_eq!(wrecking::structure("rr_storefront_row").unwrap().ball.key, "car2t");
    assert_eq!(wrecking::structure("ft_apartment_block").unwrap().ball.key, "jet");
    assert_eq!(wrecking::structure("ft_glass_tower").unwrap().ball.key, "jet");
}

// ---- The Ram Raid and Fly-through buildings (`stress_ref::buildings`). ----

const BRICK: f64 = 1900.0;
const CONCRETE: f64 = 2400.0;
const GLASS: f64 = 2500.0;
const STEEL: f64 = 7850.0;

fn kind(c: &ChunkDesc) -> &str {
    c.groups.first().map(String::as_str).unwrap_or("")
}

/// The joint material where two chunks meet, from its definition: mortar wherever brick
/// is, else the glass's wherever a pane is, else the concrete's wherever concrete is, else
/// (steel on steel) the steel's.
fn joint(a: &ChunkDesc, b: &ChunkDesc) -> &'static str {
    let either = |m: &str| a.material == m || b.material == m;
    if either("brick") {
        "brick"
    } else if either("glass") {
        "glass"
    } else if either("reinforced-concrete") || either("footing-anchor") {
        "reinforced-concrete"
    } else {
        "steel"
    }
}

/// The bonds a building of boxes should have, counted here by brute force over every pair
/// (not `auto_bonds`), by joint material: each pair of boxes with faces touching over an
/// area, not two supports, and of a pane only its sill (the pane above) or, at its sides,
/// its joint to a mullion or corner post.
fn expected_bonds(chunks: &[ChunkDesc]) -> std::collections::BTreeMap<&'static str, usize> {
    let mut out = std::collections::BTreeMap::new();
    for i in 0..chunks.len() {
        for j in i + 1..chunks.len() {
            let (a, b) = (&chunks[i], &chunks[j]);
            let gap = |k: usize| (a.center[k] - b.center[k]).abs() - (a.half_extents[k] + b.half_extents[k]);
            let Some(axis) = (0..3).find(|&k| gap(k).abs() < 1e-9) else { continue };
            if (0..3).filter(|&k| k != axis).any(|k| gap(k) > -1e-9) {
                continue;
            }
            if a.support == Support::Fixed && b.support == Support::Fixed {
                continue;
            }
            let glazing = match (a.material == "glass", b.material == "glass") {
                (false, false) => true,
                (true, true) => false,
                _ => {
                    let (pane, other) = if a.material == "glass" { (a, b) } else { (b, a) };
                    if axis == 2 {
                        pane.center[2] > other.center[2]
                    } else {
                        matches!(kind(other), "mullion" | "post")
                    }
                }
            };
            if glazing {
                *out.entry(joint(a, b)).or_insert(0) += 1;
            }
        }
    }
    out
}

/// The built bonds by joint material.
fn bonds_by_material(s: &wrecking::Structure) -> std::collections::BTreeMap<&'static str, usize> {
    let mut out = std::collections::BTreeMap::new();
    for b in &s.bonds {
        let m: &'static str = match b.material.as_str() {
            "brick" => "brick",
            "glass" => "glass",
            "reinforced-concrete" => "reinforced-concrete",
            "steel" => "steel",
            other => panic!("joint material {other}"),
        };
        *out.entry(m).or_insert(0) += 1;
    }
    out
}

/// The total mass by material, at the calibrated densities.
fn mass_of(s: &wrecking::Structure) -> f64 {
    s.chunks.iter().map(wrecking::mass).sum()
}

/// A terrace of five shops. Its volumes from the layout: brick walls less the openings and
/// lintels; the concrete fascia, lintels and two slabs; the panes.
#[test]
fn storefront_row_counts_and_mass() {
    type L = StorefrontLayout;
    let s = wrecking::structure("rr_storefront_row").unwrap();
    assert_eq!(s.chunks.len(), 1203);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 2 * L::SHOPS + L::SHOPS + 1);
    assert_eq!(s.bonds.len(), 3464);
    assert_eq!(bonds_by_material(&s), expected_bonds(&s.chunks));
    // The panes: five doors and ten shop window panes on the ground storey, ten windows above.
    assert_eq!(s.chunks.iter().filter(|c| c.material == "glass").count(), 5 * L::SHOPS);
    let (length, t, c) = (2.0 * L::half_length(), L::WALL, L::BOND.course);
    let (ground, upper) = (L::GROUND_COURSES as f64 * c, L::UPPER_COURSES as f64 * c);
    let front = ground - c;
    let shopfronts: f64 = (0..L::SHOPS).map(|i| L::shopfront(i)).map(|(door, window)| (door.1 - door.0) * front + (window.1 - window.0) * (front - c)).sum();
    let windows = (2 * L::SHOPS) as f64 * L::WINDOW * 3.0 * c;
    let lintels = (2 * L::SHOPS) as f64 * (L::WINDOW + 2.0 * L::BEARING) * c;
    let party = (L::SHOPS + 1) as f64 * (L::DEPTH - 2.0 * t) * (ground + upper);
    let brick = t * (length * front - shopfronts + length * (ground + upper) + party + length * upper - windows - lintels);
    let concrete = t * (length * c + lintels) + 2.0 * length * L::DEPTH * L::SLAB;
    let glass = L::BOND.glass * (shopfronts + windows);
    let expected = BRICK * brick + CONCRETE * concrete + GLASS * glass;
    let total = mass_of(&s);
    assert!(rel(total, expected) < 1e-9, "{total} kg vs {expected} kg");
    // About 500 t.
    assert!((4.5e5..5.5e5).contains(&total), "{total}");
}

/// A seven-storey infilled frame. Per storey: columns, joints, 22 beam spans, 8 slab bays,
/// and per perimeter bay a lintel, masonry less the window, and the pane; the entrance
/// has a full-height pane instead of a sill.
#[test]
fn apartment_block_counts_and_mass() {
    type L = ApartmentLayout;
    let s = wrecking::structure("ft_apartment_block").unwrap();
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 1355);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), L::LINES + L::PERIMETER_BAYS);
    assert_eq!(s.bonds.len(), 2874);
    assert_eq!(bonds_by_material(&s), expected_bonds(&s.chunks));
    let (c, n) = (L::COLUMN, L::STOREYS as f64);
    let span = L::BAY - c;
    let spans = (L::BAYS_Y + 1) * L::BAYS_X + (L::BAYS_X + 1) * L::BAYS_Y;
    let bays = L::PERIMETER_BAYS as f64;
    let window = L::WINDOW * (L::HEAD - L::SILL);
    let storey_concrete = L::LINES as f64 * c * c * L::STOREY + spans as f64 * span * L::BEAM_WIDTH * L::BEAM_DEPTH + (L::BAYS_X * L::BAYS_Y) as f64 * (L::BAY - L::BEAM_WIDTH).powi(2) * L::SLAB + bays * span * L::INFILL * (L::STOREY - L::BEAM_DEPTH - L::HEAD);
    let entrance = L::WINDOW * L::SILL;
    let brick = L::INFILL * (n * bays * (span * L::HEAD - window) - entrance);
    let glass = L::GLASS * (n * bays * window + entrance);
    let expected = CONCRETE * n * storey_concrete + BRICK * brick + GLASS * glass;
    let total = mass_of(&s);
    assert!(rel(total, expected) < 1e-9, "{total} kg vs {expected} kg");
    // About 1,900 t.
    assert!((1.8e6..2.0e6).contains(&total), "{total}");
}

/// An eleven-storey core-and-slab tower in a curtain wall. Per storey: the slab less the
/// core's plan, the core's walls, eight columns, and the curtain wall (mullions, transoms,
/// posts and panes) hung over its stack joint. Its glazing joints from the layout: each
/// pane bonded at its two sides only; its steel joints, each transom at its two ends.
#[test]
fn glass_tower_counts_and_mass() {
    type L = TowerLayout;
    let s = wrecking::structure("ft_glass_tower").unwrap();
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 1332);
    assert_eq!(s.chunks.iter().filter(|c| c.support == Support::Fixed).count(), 12);
    assert_eq!(s.bonds.len(), 2856);
    let by = bonds_by_material(&s);
    assert_eq!(by, expected_bonds(&s.chunks));
    assert_eq!(by["glass"], L::STOREYS * 4 * L::BAYS * 2);
    assert_eq!(by["steel"], L::STOREYS * 4 * L::BAYS * 2);
    let n = L::STOREYS as f64;
    let (side, core, md) = (2.0 * L::HALF, 2.0 * L::CORE, L::MULLION_DEPTH);
    let hung = L::STOREY - L::STACK_JOINT;
    let glazed = side - (L::BAYS - 1) as f64 * L::MULLION;
    let concrete = n * ((side * side - core * core) * L::SLAB + (core * core - (core - 2.0 * L::CORE_WALL).powi(2)) * L::STOREY + 8.0 * L::COLUMN * L::COLUMN * (L::STOREY - L::SLAB));
    let steel = n * 4.0 * ((L::BAYS - 1) as f64 * L::MULLION * md * hung + glazed * md * L::SLAB + md * md * hung);
    let glass = n * 4.0 * glazed * L::GLASS * (hung - L::SLAB);
    let expected = CONCRETE * concrete + STEEL * steel + GLASS * glass;
    let total = mass_of(&s);
    assert!(rel(total, expected) < 1e-9, "{total} kg vs {expected} kg");
    // About 1,840 t.
    assert!((1.75e6..1.95e6).contains(&total), "{total}");
}

/// Every box structure's chunks are solid (no two overlap), its footings at grade with
/// everything else above them, and every chunk is held: bonded, through the others, to a
/// footing. The buildings' glazing joints leave no pane or curtain wall unheld.
#[test]
fn structures_are_solid_and_held() {
    for &(key, _) in STRUCTURES {
        let s = wrecking::structure(key).expect(key);
        let boxes = s.chunks.iter().all(|c| c.hull.is_none());
        if boxes {
            for i in 0..s.chunks.len() {
                for j in i + 1..s.chunks.len() {
                    let (a, b) = (&s.chunks[i], &s.chunks[j]);
                    let inside = (0..3).all(|k| (a.center[k] - b.center[k]).abs() < a.half_extents[k] + b.half_extents[k] - 1e-9);
                    assert!(!inside, "{key}: chunks {i} ({:?}) and {j} ({:?}) overlap", a.groups, b.groups);
                }
            }
        }
        // Everything stands on the footings when every support is one. The viaduct's
        // abutments and road sub-base, and the topple tower's street, are supports that a
        // chunk may sit beside or below the top of.
        let only_footings = s.chunks.iter().filter(|c| c.support == Support::Fixed).all(|c| c.groups.first().map(String::as_str) == Some("footing"));
        let footing_top = s.chunks.iter().filter(|c| c.support == Support::Fixed).map(|c| c.center[2] + c.half_extents[2]).fold(0.0, f64::max);
        for c in &s.chunks {
            let bottom = c.center[2] - c.half_extents[2];
            if c.support == Support::Fixed {
                assert!(bottom.abs() < 1e-9, "{key}: a footing at {bottom}");
            } else if only_footings {
                assert!(bottom >= footing_top - 1e-9, "{key}: {:?} below the footings' top at {bottom}", c.groups);
            }
        }
        let mut root: Vec<usize> = (0..s.chunks.len()).collect();
        fn find(root: &mut [usize], i: usize) -> usize {
            let mut i = i;
            while root[i] != i {
                root[i] = root[root[i]];
                i = root[i];
            }
            i
        }
        for b in &s.bonds {
            let (x, y) = (find(&mut root, b.a), find(&mut root, b.b));
            root[x] = y;
        }
        let held: std::collections::HashSet<usize> = (0..s.chunks.len()).filter(|&c| s.chunks[c].support == Support::Fixed).map(|c| find(&mut root, c)).collect();
        for c in 0..s.chunks.len() {
            assert!(held.contains(&find(&mut root, c)), "{key}: chunk {c} ({:?}) is held by nothing", s.chunks[c].groups);
        }
    }
}

/// The buildings' nodes: `groups[0]` the type, `groups[1]` the storey (`storey-N`, or
/// `footing`), as the hosts and the Wrecking Yard example read them; the line of attack
/// at the storefront's shop windows and the towers' mid-height.
#[test]
fn buildings_are_typed_by_storey_and_aimed() {
    for (key, storeys) in [("rr_storefront_row", 2), ("ft_apartment_block", ApartmentLayout::STOREYS), ("ft_glass_tower", TowerLayout::STOREYS)] {
        let s = wrecking::structure(key).unwrap();
        for c in &s.chunks {
            assert_eq!(c.groups.len(), 2, "{key}");
            let g = &c.groups[1];
            let ok = (c.support == Support::Fixed && g == "footing") || g.strip_prefix("storey-").and_then(|n| n.parse::<usize>().ok()).is_some_and(|n| (1..=storeys).contains(&n));
            assert!(ok, "{key}: {:?}", c.groups);
        }
        let (_, hi) = wrecking::bounds(&s.chunks);
        let z = s.aim.target[2];
        if key == "rr_storefront_row" {
            // Through the middle shop's window, a car's height up.
            assert!((1.0..2.0).contains(&(z - StorefrontLayout::FOOTING_DEPTH)), "{key}: {z}");
        } else {
            assert!((0.4 * hi[2]..0.6 * hi[2]).contains(&z), "{key}: aimed at {z} of {}", hi[2]);
        }
        assert_eq!(s.aim.direction, [0.0, 1.0, 0.0], "{key}");
    }
}
