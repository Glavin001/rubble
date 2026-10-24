//! The Skybridge (`stress_ref::skybridge`): its chunk and bond counts and mass from the
//! layout's own geometry, the walkway's connections (its only bonds to the towers, the
//! seismic joint, the swing clear of the far facade), its charges (what each cuts and what
//! that leaves), and the charges' pack form naming the same bonds after either lab
//! profile's import. The shared Wrecking Yard checks (`tests/wrecking.rs`: well formed,
//! solid and held, the pack's round trip, standing under its own weight in the reference)
//! cover it too.
//!
//!   cargo test --release -p stress-ref --test skybridge

use std::collections::{BTreeMap, HashSet};

use stress_ref::capability::Profile;
use stress_ref::scene::{BondDesc, ChunkDesc, Support};
use stress_ref::skybridge::{self, SkybridgeLayout, EAST, WEST};
use stress_ref::wrecking::{self, FrameLayout};

type L = SkybridgeLayout;
type F = FrameLayout;

fn kind(c: &ChunkDesc) -> &str {
    c.groups.first().map_or("", String::as_str)
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// The bonds the structure should have, by brute force over every pair (not `auto_bonds`),
/// by joint material: boxes with faces touching over an area, not two supports, and of a
/// pane only its sill (the pane above) or its sides to a post.
fn expected_bonds(chunks: &[ChunkDesc]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for i in 0..chunks.len() {
        for j in i + 1..chunks.len() {
            let (a, b) = (&chunks[i], &chunks[j]);
            let gap = |k: usize| (a.center[k] - b.center[k]).abs() - (a.half_extents[k] + b.half_extents[k]);
            let Some(axis) = (0..3).find(|&k| gap(k).abs() < 1e-9) else { continue };
            if (0..3).filter(|&k| k != axis).any(|k| gap(k) > -1e-9) || (a.support == Support::Fixed && b.support == Support::Fixed) {
                continue;
            }
            let either = |m: &str| a.material == m || b.material == m;
            let joint = if either("glass") {
                let (pane, other) = if a.material == "glass" { (a, b) } else { (b, a) };
                let held = if axis == 2 { pane.center[2] > other.center[2] } else { kind(other) == "post" };
                if !held || other.material == "glass" {
                    continue;
                }
                "glass"
            } else if either("reinforced-concrete") || either("footing-anchor") {
                "reinforced-concrete"
            } else {
                "steel"
            };
            *out.entry(joint.to_string()).or_insert(0) += 1;
        }
    }
    out
}

/// Two towers of the Wrecking Yard frame's sections (per storey 9 columns, 12 beam spans,
/// 4 slab bays), the walkway (four chords, the posts, the deck and roof between the
/// chords, the panes between the posts), its connections (two ledges, two bearings and
/// four keepers an end) and the plaza's slabs. 992 chunks, 1,728 bonds, about 2,200 t.
#[test]
fn counts_and_mass() {
    let s = skybridge::skybridge();
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 992);
    let fixed = |k: &str| s.chunks.iter().filter(|c| c.support == Support::Fixed && kind(c) == k).count();
    assert_eq!((fixed("footing"), fixed("subbase")), (2 * L::LINES, L::PLAZA_X * L::PLAZA_Y));
    assert!(s.chunks.iter().all(|c| (c.support == Support::Fixed) == matches!(kind(c), "footing" | "subbase")));
    assert_eq!(s.bonds.len(), 1728);
    let mut built: BTreeMap<String, usize> = BTreeMap::new();
    for b in &s.bonds {
        *built.entry(b.material.clone()).or_insert(0) += 1;
    }
    assert_eq!(built, expected_bonds(&s.chunks));
    // The panes: each bonded at its sill and to the two posts beside it.
    assert_eq!(built["glass"], 2 * L::PANELS * 3);
    let (c, bw, bd) = (F::COLUMN, F::BEAM_WIDTH, F::BEAM_DEPTH);
    let storey = L::LINES as f64 * c * c * F::STOREY + 12.0 * (F::BAY - c) * bw * bd + 4.0 * (F::BAY - bw).powi(2) * F::SLAB;
    let (x0, x1) = L::ends();
    let length = x1 - x0;
    let between = L::WIDTH - 2.0 * L::CHORD;
    let keeper = (L::LEDGE[0] - L::JOINT - L::BEARING[0] - L::KEEPER[0]) * (0.5 * L::BEARING[1] - 0.5 * L::CHORD - L::KEEPER[1]) * (L::BEARING[2] + L::KEEPER[2]);
    let concrete = 2.0 * L::STOREYS as f64 * storey
        + length * between * L::DECK
        + 2.0 * (L::LEDGE[0] * 2.0 * L::LEDGE[1] * L::LEDGE[2] + 4.0 * keeper)
        + (L::PLAZA_X * L::PLAZA_Y) as f64 * L::TILE * L::TILE * L::PLAZA_DEPTH;
    let clear = length - (L::PANELS + 1) as f64 * L::POST;
    let steel = 2.0 * length * L::CHORD * (L::BOTTOM_CHORD + L::TOP_CHORD) + 2.0 * (L::PANELS + 1) as f64 * L::POST * L::CHORD * L::CLEAR + length * between * L::ROOF + 4.0 * L::BEARING.iter().product::<f64>();
    let glass = 2.0 * clear * L::GLASS * L::CLEAR;
    let expected = 2400.0 * concrete + 7850.0 * steel + 2500.0 * glass;
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!(rel(total, expected) < 1e-9, "{total} kg vs {expected} kg");
    let walkway: f64 = skybridge::walkway(&s.chunks).iter().map(|&k| wrecking::mass(&s.chunks[k])).sum();
    println!("total {:.0} t, walkway {:.0} t", total / 1e3, walkway / 1e3);
    assert!((2.1e6..2.3e6).contains(&total), "{total}");
    // The walkway, its bearings included, about 120 t.
    assert!((1.0e5..1.4e5).contains(&walkway), "{walkway}");
}

/// The walkway's connections: it is bonded to the towers only where its four bearings sit
/// on their ledges; its ends stop the seismic joint short of each facade; it can swing
/// down about its west bearings' keepers clear of the east facade; its floor is level with
/// the fifth storey's (on its bearings); the plaza lies under it.
#[test]
fn walkway_hangs_on_its_bearings() {
    let s = skybridge::skybridge();
    let walkway: HashSet<usize> = skybridge::walkway(&s.chunks).into_iter().collect();
    assert_eq!(walkway.len(), L::WALKWAY + 4);
    let out: Vec<&BondDesc> = s.bonds.iter().filter(|b| walkway.contains(&b.a) != walkway.contains(&b.b)).collect();
    assert_eq!(out.len(), 4);
    for b in out {
        let (inside, outside) = if walkway.contains(&b.a) { (b.a, b.b) } else { (b.b, b.a) };
        assert!(kind(&s.chunks[inside]).starts_with("bearing-") && kind(&s.chunks[outside]).starts_with("ledge-"), "{:?} to {:?}", s.chunks[inside].groups, s.chunks[outside].groups);
        assert_eq!(b.normal[2].abs(), 1.0, "a bearing on its ledge");
    }
    // Its box, the bearing plates (wider than the chords) aside.
    let (lo, hi) = wrecking::bounds(&walkway.iter().map(|&k| s.chunks[k].clone()).filter(|c| !kind(c).starts_with("bearing-")).collect::<Vec<_>>());
    let face = 0.5 * L::GAP;
    assert!((lo[0] + face - L::JOINT).abs() < 1e-12 && (face - hi[0] - L::JOINT).abs() < 1e-12, "{lo:?} {hi:?}");
    assert!((hi[1] - lo[1] - L::WIDTH).abs() < 1e-12);
    assert!((L::post_bottom() - L::floor() - L::BEARING[2] - L::BOTTOM_CHORD).abs() < 1e-12);
    // The facades: the towers' chunks nearest the walkway end at x = -GAP / 2 and GAP / 2,
    // the ledges and keepers in between.
    let towers: Vec<&ChunkDesc> = s.chunks.iter().filter(|c| c.groups[1].starts_with("storey-") && !kind(c).starts_with("ledge-") && !kind(c).starts_with("keeper-")).collect();
    assert!(towers.iter().all(|c| (c.center[0].abs() - c.half_extents[0]) >= face - 1e-12));
    // About the west keepers' face at the bearings' top, every corner of the walkway stays
    // clear of the east facade as it swings down.
    let pivot = [-face + L::JOINT + L::BEARING[0] + L::KEEPER[0], L::chord_bottom()];
    let reach = walkway
        .iter()
        .flat_map(|&k| {
            let c = &s.chunks[k];
            [-1.0, 1.0].into_iter().flat_map(move |sx| [-1.0, 1.0].map(|sz| (c.center[0] + sx * c.half_extents[0] - pivot[0]).hypot(c.center[2] + sz * c.half_extents[2] - pivot[1])))
        })
        .fold(0.0, f64::max);
    assert!(pivot[0] + reach < face - 0.1, "the walkway's far corner reaches {} of the east facade at {face}", pivot[0] + reach);
    // Each keeper stands beyond its bearing (within the ledge), beside the chord, clear of
    // both.
    for end in ["west", "east"] {
        let of = |k: &'static str| s.chunks.iter().filter(move |c| kind(c) == format!("{k}-{end}"));
        assert_eq!((of("ledge").count(), of("bearing").count(), of("keeper").count()), (2, 2, 4), "{end}");
        let toward = if end == "west" { 1.0 } else { -1.0 };
        for k in of("keeper") {
            let b = of("bearing").min_by(|a, b| (a.center[1] - k.center[1]).abs().total_cmp(&(b.center[1] - k.center[1]).abs())).unwrap();
            let gap = toward * (k.center[0] - b.center[0]) - k.half_extents[0] - b.half_extents[0];
            assert!((gap - L::KEEPER[0]).abs() < 1e-12, "{end}: keeper {gap} m beyond its bearing");
            assert!(k.center[2] + k.half_extents[2] > L::chord_bottom(), "{end}: a keeper stands above the bearing's top");
        }
    }
    // The plaza under the walkway: across its whole width, and most of its length.
    let plaza: Vec<&ChunkDesc> = s.chunks.iter().filter(|c| kind(c) == "plaza").collect();
    let (plo, phi) = wrecking::bounds(&plaza.iter().map(|c| (*c).clone()).collect::<Vec<_>>());
    assert!(plo[1] < lo[1] && phi[1] > hi[1] && phi[0] - plo[0] > 0.9 * (hi[0] - lo[0]));
}

/// The east charge cuts every bond of the east connection's ledges, bearings and keepers
/// and throws them down and away from the east facade; the west charge cuts the west
/// bearings from their ledges. Fired together, the walkway is one piece bonded to nothing
/// else, resting on the west ledges by its bearings.
#[test]
fn charges_cut_the_walkway_free() {
    let s = skybridge::skybridge();
    let names: Vec<&str> = s.charges.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, [EAST, WEST]);
    assert_eq!((s.charges[0].delay, s.charges[1].delay), (0.0, L::TEAR_DELAY));
    let east = &s.charges[0];
    assert_eq!(east.debris.len(), 8);
    for &k in &east.debris {
        let c = &s.chunks[k];
        assert!(kind(c).ends_with("-east") && c.center[0] > 0.0, "chunk {k}: {:?}", c.groups);
        let own: Vec<usize> = (0..s.bonds.len()).filter(|&i| s.bonds[i].a == k || s.bonds[i].b == k).collect();
        assert!(!own.is_empty() && own.iter().all(|i| east.bonds.contains(i)), "chunk {k}: every bond cut");
    }
    assert!(east.kick[2] < 0.0 && east.kick[0] < 0.0, "down and away from the east facade: {:?}", east.kick);
    let west = &s.charges[1];
    assert_eq!(west.bonds.len(), 2);
    assert!(west.debris.is_empty());
    for &i in &west.bonds {
        let (a, b) = (kind(&s.chunks[s.bonds[i].a]), kind(&s.chunks[s.bonds[i].b]));
        assert!([a, b].contains(&"bearing-west") && [a, b].contains(&"ledge-west"), "bond {i}: {a}-{b}");
    }
    let cut: HashSet<usize> = s.charges.iter().flat_map(|c| c.bonds.iter().copied()).collect();
    let left: Vec<&BondDesc> = (0..s.bonds.len()).filter(|i| !cut.contains(i)).map(|i| &s.bonds[i]).collect();
    let walkway: HashSet<usize> = skybridge::walkway(&s.chunks).into_iter().filter(|k| !east.debris.contains(k)).collect();
    let mut seen = HashSet::from([*walkway.iter().next().unwrap()]);
    let mut stack: Vec<usize> = seen.iter().copied().collect();
    while let Some(k) = stack.pop() {
        for b in &left {
            let other = if b.a == k { b.b } else if b.b == k { b.a } else { continue };
            if seen.insert(other) {
                stack.push(other);
            }
        }
    }
    assert_eq!(seen, walkway, "the walkway one piece, bonded to nothing else");
    // Its west bearings over the west ledges.
    for b in s.chunks.iter().filter(|c| kind(c) == "bearing-west") {
        let ledge = s.chunks.iter().filter(|c| kind(c) == "ledge-west").find(|l| (l.center[1] - b.center[1]).abs() < l.half_extents[1]).expect("a ledge under the bearing");
        assert!((ledge.center[2] + ledge.half_extents[2] - (b.center[2] - b.half_extents[2])).abs() < 1e-12);
        assert!((b.center[0] - ledge.center[0]).abs() + b.half_extents[0] <= ledge.half_extents[0] + 1e-12);
    }
}

/// The pack carries the charges (`defaults.charges`) as node pairs that name the same
/// bonds after either lab profile's import, with their debris, delays and kick.
#[test]
fn charges_survive_the_pack() {
    let s = skybridge::skybridge();
    let pack = wrecking::to_pack(&s);
    let read = wrecking::charges_of(&pack);
    assert_eq!(read.len(), s.charges.len());
    for profile in [Profile::BlastParity, Profile::Threshold] {
        let (scene, notes) = stress_ref::scene_pack::import_value_with(&pack, s.key, "structure", s.key, profile.import_options()).expect("import");
        assert!(notes.is_empty(), "{notes:?}");
        let bonds = &scene.bodies[0].bonds;
        for (c, r) in s.charges.iter().zip(&read) {
            assert_eq!((r.name.as_str(), r.delay, &r.debris), (c.name.as_str(), c.delay, &c.debris));
            assert!(r.kick.iter().zip(c.kick).all(|(a, b)| (a - b).abs() < 1e-12), "{:?} vs {:?}", r.kick, c.kick);
            let got = r.bond_indices(bonds).expect("every pair a bond");
            let pairs = |i: usize, b: &[BondDesc]| (b[i].a.min(b[i].b), b[i].a.max(b[i].b));
            let want: Vec<(usize, usize)> = c.bonds.iter().map(|&i| pairs(i, &s.bonds)).collect();
            assert_eq!(got.iter().map(|&i| pairs(i, bonds)).collect::<Vec<_>>(), want, "{}", c.name);
        }
    }
}
