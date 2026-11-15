//! Anti-clip's box padding by each contacter's own travel (`ExplicitBeams::own_travel_reach`,
//! a Rubble setting): a fast node that is neither a contacter nor a collision triangle's
//! node no longer changes which (triangle, contacter) pairs self-collision tests; with
//! the specification's padding (the vehicle's fastest node) it does.

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
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

/// Pairs tested in the first step, with node `fast` (if any) moving up at 300 m/s.
fn pairs(own: bool, fast: Option<usize>) -> u32 {
    let mut s = sedan();
    s.own_travel_reach = own;
    if let Some(i) = fast {
        s.nodes[i].vel = Vec3::new(0.0, 300.0, 0.0);
    }
    s.step();
    s.last_self_pairs
}

#[test]
fn a_fast_unrelated_node_leaves_the_pairs_with_own_travel() {
    let s = sedan();
    let mut related = vec![false; s.nodes.len()];
    for &c in &s.candidates {
        related[c as usize] = true;
    }
    for t in &s.triangles {
        for n in t.nodes {
            related[n as usize] = true;
        }
    }
    let fast = (0..s.nodes.len()).find(|&i| !related[i]).expect("a node neither contacter nor on a triangle");
    let (plain, plain_fast) = (pairs(false, None), pairs(false, Some(fast)));
    let (own, own_fast) = (pairs(true, None), pairs(true, Some(fast)));
    eprintln!("pairs tested: fastest-node padding {plain} (with node {fast} at 300 m/s: {plain_fast}); own travel {own} ({own_fast})");
    assert!(plain > 0, "no self-collision pairs");
    // The specification's padding: the fast node pads every triangle's box.
    assert!(plain_fast > plain, "{plain_fast} pairs with the fast node, {plain} without");
    // Own travel: the same pairs as without it, and never more than the fastest node's.
    assert_eq!(own_fast, own);
    assert!(own <= plain);
}
