//! The Topple tower (`stress_ref::topple`): its chunk and bond counts and mass from the
//! layout's own geometry, its charges (what each cuts, what that leaves standing and
//! where the freed tower's weight lies), and the charges' pack form (`defaults.charges`)
//! naming the same bonds after either lab profile's import.
//!
//!   cargo test --release -p stress-ref --test topple

use std::collections::HashSet;

use stress_ref::capability::Profile;
use stress_ref::scene::{BondDesc, ChunkDesc, Support};
use stress_ref::topple::{self, ToppleLayout, BACK_ROW, STREET_SIDE};
use stress_ref::wrecking::{self, FrameLayout};

type L = ToppleLayout;
type F = FrameLayout;

fn kind(c: &ChunkDesc) -> &str {
    c.groups.first().map_or("", String::as_str)
}

/// The chunks joined to `from` by `bonds` (not crossing a support), `from` included.
fn reach(chunks: &[ChunkDesc], bonds: &[&BondDesc], from: usize) -> HashSet<usize> {
    let mut seen = HashSet::from([from]);
    let mut stack = vec![from];
    while let Some(k) = stack.pop() {
        for b in bonds {
            let other = if b.a == k { b.b } else if b.b == k { b.a } else { continue };
            if chunks[other].support == Support::None && seen.insert(other) {
                stack.push(other);
            }
        }
    }
    seen
}

/// Per storey of the frame: columns to joints 6, joints to the columns above (or the pads)
/// 6, joints to beam ends 14, beam halves at midspan 7, slab quarters to beams 16 and to
/// each other 8. Each end wall (second storey up, one bay): halves 1, to the columns at
/// their ends 2, to the beams over and under them 2 + 2. The back wall (ground storey, two
/// bays): per bay halves 1, columns 2, beams 2, strip footings 2, the pads they reach 2.
/// The back wall's outer leaves (per bay): halves 1, columns 2, the inner leaves 2, strip
/// footings 2, pads 2, the kerb 2; and the back columns to the kerb, 1 each.
#[test]
fn counts_and_mass() {
    let s = topple::topple_tower();
    assert_eq!(s.chunks.len(), L::CHUNKS);
    assert_eq!(L::CHUNKS, 425);
    let fixed = |k: &str| s.chunks.iter().filter(|c| c.support == Support::Fixed && kind(c) == k).count();
    assert_eq!((fixed("footing"), fixed("street")), (L::LINES + L::STRIPS + 1, L::TILES));
    assert!(s.chunks.iter().all(|c| (c.support == Support::Fixed) == matches!(kind(c), "footing" | "street")));
    assert_eq!(s.bonds.len(), L::STOREYS * (6 + 6 + 14 + 7 + 16 + 8) + 2 * (L::STOREYS - 1) * (1 + 2 + 2 + 2) + L::BAYS[0] * (1 + 2 + 2 + 2 + 2) + L::BAYS[0] * (1 + 2 + 2 + 2 + 2 + 2) + (L::BAYS[0] + 1));
    // Concrete a storey: columns, joints, beams (7 spans), slabs (2 bays); the end walls
    // (9 storeys) and the back wall (2 bays, the ground storey's); the hinge line's extra
    // depth (the ground storey's back columns, joints and wall).
    let (span, clear) = (F::BAY - F::BEAM_WIDTH, F::STOREY - F::BEAM_DEPTH);
    let storey = 6.0 * F::COLUMN * F::COLUMN * F::STOREY + 7.0 * (F::BAY - F::COLUMN) * F::BEAM_WIDTH * F::BEAM_DEPTH + 2.0 * span * span * F::SLAB;
    let wall = (F::BAY - F::COLUMN) * F::WALL * clear;
    let hinge = 3.0 * F::COLUMN * (L::HINGE - F::COLUMN) * F::STOREY + 2.0 * (F::BAY - F::COLUMN) * clear * (L::HINGE - F::WALL);
    let expected = 2400.0 * (L::STOREYS as f64 * storey + (2.0 * (L::STOREYS - 1) as f64 + 2.0) * wall + hinge);
    let total: f64 = s.chunks.iter().map(wrecking::mass).sum();
    assert!((total - expected).abs() < 1e-9 * expected, "{total} kg vs {expected} kg");
    // About 1,060 t.
    assert!((1.0e6..1.1e6).contains(&total), "{total}");
    // Tall and slender: 35.5 m on a 7 m deep plan (the frame's 6.5 m and the hinge line's
    // extra half metre behind it); the street along its -y side.
    let (lo, hi) = wrecking::bounds(&s.chunks.iter().filter(|c| c.support == Support::None).cloned().collect::<Vec<_>>());
    assert!((hi[2] - F::FOOTING_DEPTH - L::STOREYS as f64 * F::STOREY).abs() < 1e-9, "{}", hi[2]);
    assert!((hi[1] - lo[1] - (F::BAY + L::HINGE)).abs() < 1e-12 && (hi[0] - lo[0] - 12.5).abs() < 1e-12, "{lo:?} {hi:?}");
    let street: Vec<&ChunkDesc> = s.chunks.iter().filter(|c| kind(c) == "street").collect();
    assert!(street.iter().all(|c| c.center[1] + c.half_extents[1] <= lo[1] - L::SIDEWALK + 1e-9 && c.center[2] + c.half_extents[2] <= L::STREET_DEPTH + 1e-12));
}

/// The street-side charge cuts every bond of the street side's three ground-storey
/// columns (they come free, and it throws them out over the street); the back-row charge
/// cuts the back line's bonds to its footings. Fired together they leave the tower one
/// piece, bonded to no support, its weight over the street side of the line it then
/// stands on: it tips toward the street.
#[test]
fn charges_cut_the_tower_free_to_tip() {
    let s = topple::topple_tower();
    let names: Vec<&str> = s.charges.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, [STREET_SIDE, BACK_ROW]);
    assert_eq!((s.charges[0].delay, s.charges[1].delay), (0.0, L::HINGE_DELAY));
    let street = &s.charges[0];
    assert_eq!(street.debris.len(), L::BAYS[0] + 1);
    for &k in &street.debris {
        let c = &s.chunks[k];
        assert!(topple::ground(c, "column") && c.center[1] < 0.0, "chunk {k}: a street-side ground column");
        let own: HashSet<usize> = (0..s.bonds.len()).filter(|&i| s.bonds[i].a == k || s.bonds[i].b == k).collect();
        assert_eq!(own.len(), 2, "chunk {k}: to its footing and its joint");
        assert!(own.iter().all(|i| street.bonds.contains(i)), "chunk {k}: every bond cut");
    }
    assert_eq!(street.bonds.len(), 2 * street.debris.len());
    assert!(street.kick[1] < 0.0 && street.kick[0] == 0.0, "out over the street: {:?}", street.kick);
    let back = &s.charges[1];
    assert_eq!(back.bonds.len(), (L::BAYS[0] + 1) + 2 * 2 * 2 * L::BAYS[0] + (L::BAYS[0] + 1) + 2 * L::BAYS[0], "the back columns' pads, the back wall leaves' strips and pads, the kerb");
    for &i in &back.bonds {
        let (a, b) = (&s.chunks[s.bonds[i].a], &s.chunks[s.bonds[i].b]);
        let (up, down) = if a.support == Support::Fixed { (b, a) } else { (a, b) };
        assert_eq!(kind(down), "footing");
        assert!((topple::ground(up, "column") || topple::ground(up, "wall")) && up.center[1] > 0.0, "bond {i}: the back line's");
    }
    // Both fired: no bond to a support; the tower (without the thrown columns) one piece.
    let cut: HashSet<usize> = s.charges.iter().flat_map(|c| c.bonds.iter().copied()).collect();
    let left: Vec<&BondDesc> = (0..s.bonds.len()).filter(|i| !cut.contains(i)).map(|i| &s.bonds[i]).collect();
    assert!(left.iter().all(|b| s.chunks[b.a].support == Support::None && s.chunks[b.b].support == Support::None), "a bond to a support left");
    let tower: Vec<usize> = (0..s.chunks.len()).filter(|&k| s.chunks[k].support == Support::None && !street.debris.contains(&k)).collect();
    assert_eq!(reach(&s.chunks, &left, tower[0]).len(), tower.len(), "the tower stays one piece");
    // Its centre of mass is over the street side of the back line it stands on.
    let mass: f64 = tower.iter().map(|&k| wrecking::mass(&s.chunks[k])).sum();
    let y: f64 = tower.iter().map(|&k| wrecking::mass(&s.chunks[k]) * s.chunks[k].center[1]).sum::<f64>() / mass;
    let line = 0.5 * L::BAYS[1] as f64 * F::BAY;
    assert!(y < line - 0.5 * F::COLUMN - 2.0, "centre of mass at y = {y}, the back line at {line}");
}

/// The hinge line the freed tower turns on: the back line's ground-storey columns and
/// joints `HINGE` deep from the inner column face, its wall in two leaves as deep (the
/// inner one flush with the frame, under the back beam); the footings under them reach
/// the pads' street-side edge, in front of the edge the tower turns about. The kerb is a
/// support from grade behind it, touching their back faces below the beams and bonded to
/// them (the back-row charge cuts it free): it stops the base sliding back, and a turn
/// toward the street about that edge only moves the hinge's back face away from it.
#[test]
fn hinge_line_and_kerb() {
    let s = topple::topple_tower();
    let line = 0.5 * L::BAYS[1] as f64 * F::BAY;
    let (inner, face, outer) = (line - 0.5 * F::COLUMN, line + 0.5 * F::COLUMN, line - 0.5 * F::COLUMN + L::HINGE);
    let span = |c: &ChunkDesc, ax: usize| (c.center[ax] - c.half_extents[ax], c.center[ax] + c.half_extents[ax]);
    let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12;
    let back = |c: &&ChunkDesc| c.support == Support::None && span(c, 1).1 > face - 1e-12;
    let on = |k: &str, y: (f64, f64)| s.chunks.iter().filter(back).filter(|c| topple::ground(c, k) && near(span(c, 1), y)).count();
    assert_eq!(on("column", (inner, outer)), L::BAYS[0] + 1);
    assert_eq!(on("joint", (inner, outer)), L::BAYS[0] + 1);
    assert_eq!((on("wall", (inner, face)), on("wall", (face, outer))), (2 * L::BAYS[0], 2 * L::BAYS[0]), "the wall's two leaves");
    // Everything else keeps the frame's sections: the back beams over the inner leaves,
    // and nothing else behind the back line's column faces.
    assert_eq!(s.chunks.iter().filter(back).filter(|c| topple::ground(c, "beam") && near(span(c, 1), (inner, face))).count(), 2 * L::BAYS[0]);
    let hinge: Vec<&ChunkDesc> = s.chunks.iter().filter(back).filter(|c| span(c, 1).1 > face + 1e-12).collect();
    assert_eq!(hinge.len(), 2 * (L::BAYS[0] + 1) + 2 * L::BAYS[0]);
    // The footings under the hinge: from the pads' street-side edge to its back face.
    let under: Vec<&ChunkDesc> = s.chunks.iter().filter(|c| kind(c) == "footing" && c.groups[1] == "footing" && c.center[1] > 0.0).collect();
    assert_eq!(under.len(), (L::BAYS[0] + 1) + L::STRIPS);
    assert!(under.iter().all(|c| near(span(c, 1), (line - 0.5 * F::FOOTING, outer))));
    // The kerb.
    let kerbs: Vec<usize> = (0..s.chunks.len()).filter(|&k| s.chunks[k].groups.get(1).map(String::as_str) == Some("kerb")).collect();
    assert_eq!(kerbs.len(), 1);
    let kerb = &s.chunks[kerbs[0]];
    assert_eq!((kind(kerb), kerb.support), ("footing", Support::Fixed));
    assert!(near(span(kerb, 1), (outer, outer + L::KERB[0])));
    let (z0, z1) = span(kerb, 2);
    assert!(z0.abs() < 1e-12 && (z1 - F::FOOTING_DEPTH - L::KERB[1]).abs() < 1e-12 && z1 < F::FOOTING_DEPTH + F::STOREY - F::BEAM_DEPTH);
    let (x0, x1) = span(kerb, 0);
    assert!(hinge.iter().all(|c| span(c, 0).0 >= x0 && span(c, 0).1 <= x1), "along the whole hinge line");
    let kerb_bonds: Vec<usize> = (0..s.bonds.len()).filter(|&i| s.bonds[i].a == kerbs[0] || s.bonds[i].b == kerbs[0]).collect();
    assert_eq!(kerb_bonds.len(), (L::BAYS[0] + 1) + 2 * L::BAYS[0], "the back columns and outer leaves");
    let back_row = &s.charges[1];
    assert!(kerb_bonds.iter().all(|i| back_row.bonds.contains(i)), "the back-row charge cuts the kerb free");
}

/// The pack carries the charges as node pairs (`defaults.charges`) that name the same
/// bonds after either lab profile's import (whatever bonds an import keeps and in what
/// order), with their debris, delays and kick (Y-up in the pack, Z-up read back).
#[test]
fn charges_survive_the_pack() {
    let s = topple::topple_tower();
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
    // A pair no bond joins is an error that names it.
    let mut bad = read[0].clone();
    bad.bonds.push([0, s.chunks.len() - 1]);
    assert!(bad.bond_indices(&s.bonds).unwrap_err().contains("no bond joins"));
    // Structures without charges write none.
    assert!(wrecking::charges_of(&wrecking::pack("wy_frame10").unwrap()).is_empty());
}
