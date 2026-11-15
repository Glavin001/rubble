//! Time to live for small debris (`engine_softbody::explicit_debris`, an explicit setting,
//! off by default): a small piece off the vehicle's main body for its time to live is
//! retired (`ExplicitBeams::retired`): no longer simulated, in no contact. The rule's
//! cases (the main body, too heavy, too large, still joined, node 0, off), and the CPU
//! reference with retired nodes: frozen, out of self-collision, and with nothing
//! retired every state the same as without the setting.

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_debris::{DebrisTracker, DebrisTtl, Pieces};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: false,
    centred_damping: false,
    centred_objective: false,
};

fn sedan() -> ExplicitBeams {
    let text = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Low,
            ..Default::default()
        },
    )
    .unwrap()
    .truck;
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), Vec3::ZERO, EXACT).unwrap();
    s.anti_clip = true;
    s.finish();
    s
}

/// Every beam across the boundary of `cut` broken; returns the cut's mass.
fn tear(s: &mut ExplicitBeams, cut: &[bool]) -> f32 {
    for b in &mut s.beams {
        if cut[b.a as usize] != cut[b.b as usize] {
            b.broken = true;
        }
    }
    s.nodes.iter().zip(cut).filter(|(_, &c)| c).map(|(n, _)| n.mass).sum()
}

fn broken(s: &ExplicitBeams) -> Vec<bool> {
    s.beams.iter().map(|b| b.broken).collect()
}

/// The end away from node 0 (+1 the front, -1 the rear).
fn far_end(s: &ExplicitBeams) -> f32 {
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let rear = s.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
    if front - s.nodes[0].pos.z > s.nodes[0].pos.z - rear { 1.0 } else { -1.0 }
}

/// A corner at the end away from node 0 (the nodes within `depth` of that end and 0.35 m
/// of the right side).
fn corner(s: &ExplicitBeams, depth: f32) -> Vec<bool> {
    let e = far_end(s);
    let end = s.nodes.iter().map(|n| e * n.pos.z).fold(f32::MIN, f32::max);
    let right = s.nodes.iter().map(|n| n.pos.x).fold(f32::MIN, f32::max);
    s.nodes.iter().map(|n| e * n.pos.z > end - depth && n.pos.x > right - 0.35).collect()
}

const TTL: DebrisTtl = DebrisTtl { max_mass_kg: 5.0, max_nodes: 40, seconds: 0.5 };

#[test]
fn a_small_piece_is_retired_after_its_time_and_nothing_else() {
    let mut s = sedan();
    // A small corner (under 5 kg, 40 nodes) and the last 1.2 m of that end (heavier),
    // away from node 0.
    let small = corner(&s, 0.25);
    let e = far_end(&s);
    let end = s.nodes.iter().map(|n| e * n.pos.z).fold(f32::MIN, f32::max);
    let rear: Vec<bool> = s.nodes.iter().zip(&small).map(|(n, &c)| !c && e * n.pos.z > end - 1.2).collect();
    assert!(!small[0] && !rear[0], "node 0 is in the main body");
    let small_mass = tear(&mut s, &small);
    let rear_mass = tear(&mut s, &rear);
    let pieces = Pieces::of(&s, &broken(&s));
    let n_small = small.iter().filter(|&&c| c).count() as u32;
    eprintln!("corner {n_small} nodes {small_mass:.2} kg, rear {rear_mass:.0} kg, {} pieces", pieces.mass.len());
    assert!(small_mass <= TTL.max_mass_kg && n_small <= TTL.max_nodes, "the corner is small: {small_mass} kg, {n_small} nodes");
    assert!(rear_mass > TTL.max_mass_kg, "the rear is heavy: {rear_mass} kg");
    // Off: nothing is ever retired.
    let mut off = DebrisTracker::new(None, s.nodes.len());
    for k in 0..20 {
        assert!(!off.update(&pieces, k as f64 * 0.1));
    }
    assert!(off.retired.iter().all(|&r| !r));
    // On: the corner only, at its time to live from the first reading it was off.
    let mut on = DebrisTracker::new(Some(TTL), s.nodes.len());
    let mut retired_at = None;
    for k in 0..20 {
        let t = 1.0 + k as f64 * 0.1;
        if on.update(&pieces, t) {
            retired_at.get_or_insert(t);
        }
    }
    let at = retired_at.expect("the corner was retired");
    assert!((at - 1.5).abs() < 1e-9, "retired at {at} s (off since 1.0 s, 0.5 s to live)");
    // Retired: exactly the nodes of the small pieces off the main body (the corner, and
    // any fragment the two cuts left).
    let p0 = pieces.piece[0];
    for i in 0..s.nodes.len() {
        let p = pieces.piece[i] as usize;
        let expect = p as u32 != pieces.main && p as u32 != p0 && pieces.mass[p] <= TTL.max_mass_kg && pieces.nodes[p] <= TTL.max_nodes;
        assert_eq!(on.retired[i], expect, "node {i}: retired {} (corner {}, piece {p} {} kg)", on.retired[i], small[i], pieces.mass[p]);
        if small[i] && !pieces.attached(i) {
            assert!(on.retired[i], "corner node {i} retired");
        }
    }
    assert!(!on.retired.iter().zip(&rear).any(|(&r, &x)| r && x), "the heavy rear is not retired");
    let ev = on.take_events();
    assert_eq!(ev.len(), on.pieces as usize);
    assert!((on.mass - ev.iter().map(|e| e.mass).sum::<f32>()).abs() < 1e-4);
    let expect_mass: f32 = (0..s.nodes.len()).filter(|&i| on.retired[i]).map(|i| s.nodes[i].mass).sum();
    assert!((on.mass - expect_mass).abs() < 1e-3, "retired mass {} against the retired nodes' {}", on.mass, expect_mass);
    assert_eq!(on.nodes as usize, on.retired.iter().filter(|&&r| r).count());
    // Too many nodes for a lower limit: kept.
    let mut few = DebrisTracker::new(Some(DebrisTtl { max_nodes: 1, ..TTL }), s.nodes.len());
    for k in 0..20 {
        few.update(&pieces, k as f64 * 0.1);
    }
    let lone = pieces.nodes.iter().filter(|&&c| c == 1).count();
    assert_eq!(few.nodes as usize, lone, "only single-node pieces under a 1-node limit");
}

#[test]
fn the_piece_holding_node_0_and_a_joined_piece_are_never_retired() {
    let mut s = sedan();
    // Node 0's neighbourhood torn off (small) and a corner that stays joined.
    let p0 = s.nodes[0].pos;
    let near0: Vec<bool> = s.nodes.iter().map(|n| (n.pos - p0).length() < 0.12).collect();
    tear(&mut s, &near0);
    let pieces = Pieces::of(&s, &broken(&s));
    assert_ne!(pieces.piece[0], pieces.main, "node 0's piece is off the main body");
    let mut on = DebrisTracker::new(Some(TTL), s.nodes.len());
    for k in 0..40 {
        on.update(&pieces, k as f64 * 0.1);
    }
    let p = pieces.piece[0];
    assert!(
        (0..s.nodes.len()).filter(|&i| pieces.piece[i] == p).all(|i| !on.retired[i]),
        "the piece holding node 0 is kept"
    );
    // The untouched corner is joined: never retired.
    let c = corner(&s, 0.25);
    assert!((0..s.nodes.len()).filter(|&i| c[i] && pieces.attached(i)).all(|i| !on.retired[i]));
    assert!((0..s.nodes.len()).filter(|&i| pieces.attached(i)).all(|i| !on.retired[i]), "the main body is never retired");
}

#[test]
fn retired_nodes_are_frozen_and_out_of_self_collision() {
    let mut s = sedan();
    let small = corner(&s, 0.25);
    tear(&mut s, &small);
    // The torn corner flung up into the car's own path: falling back through it.
    for (n, &c) in s.nodes.iter_mut().zip(&small) {
        if c {
            n.vel = Vec3::new(-3.0, 2.0, -6.0);
        }
    }
    let mut a = s.clone();
    let mut b = s.clone();
    b.retired = vec![false; s.nodes.len()];
    // Nothing retired: the same as without the setting, bit for bit.
    for _ in 0..400 {
        a.step();
        b.step();
    }
    for (x, y) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(x.pos.to_array().map(f32::to_bits), y.pos.to_array().map(f32::to_bits));
        assert_eq!(x.vel.to_array().map(f32::to_bits), y.vel.to_array().map(f32::to_bits));
    }
    // Retire the corner: from the next step it stays where it is, at rest.
    let pieces = Pieces::of(&b, &broken(&b));
    let retired: Vec<bool> = (0..b.nodes.len()).map(|i| small[i] && !pieces.attached(i)).collect();
    assert!(retired.iter().any(|&r| r));
    b.retired = retired.clone();
    let held: Vec<Vec3> = b.nodes.iter().map(|n| n.pos).collect();
    let mut pairs_with = 0u64;
    for _ in 0..400 {
        b.step();
        pairs_with += u64::from(b.last_self_pairs);
        for (i, n) in b.nodes.iter().enumerate() {
            if retired[i] {
                assert_eq!(n.pos, held[i], "retired node {i} moved");
                assert_eq!(n.vel, Vec3::ZERO, "retired node {i} has a velocity");
            }
        }
    }
    eprintln!("400 steps: {pairs_with} pairs tested, retired nodes frozen");
}
