//! `World::debris_reach_routing` (`STRESS_DEBRIS_ROUTING=1`): routing extended by the reach
//! of debris launched within the frame. On the wall row with the ram 1.15 m back
//! (`wall_row_late(2)`), the strike frame's sweep of the ram reaches only the first wall;
//! the plug it punches out reaches the second within the same frame
//! (learned-impact/findings/debris-reach-routing-2026-10-20.md). Off, the island is the ram
//! and the first wall; on, the second wall joins it. Nothing changes in a frame without a
//! fast impact.

use std::collections::HashSet;

use stress_ref::math::Vec3;
use stress_ref::world::World;
use stress_rubble::scenes::{realtime, wall_row_late};
use stress_rubble::RubbleEngine;

#[test]
fn debris_reach_joins_the_wall_behind_the_struck_one() {
    let scene = realtime(wall_row_late(2), 0.05);
    let mut w = World::with_engine(&scene, Box::new(RubbleEngine::new(Vec3::from_array(scene.gravity)).expect("rubble engine")));
    let fdt = scene.sim.frame_dt;
    // Frame 1: the ram in free flight, no island either way.
    for on in [false, true] {
        w.debris_reach_routing = on;
        assert_eq!(w.predict_islands(fdt, &HashSet::new()), 0, "no island before the strike (flag {on})");
    }
    w.debris_reach_routing = false;
    w.step_frame();
    // Frame 2: the strike.
    w.debris_reach_routing = false;
    let off = w.predict_islands(fdt, &HashSet::new());
    w.debris_reach_routing = true;
    let on = w.predict_islands(fdt, &HashSet::new());
    assert_eq!(off, 2, "the ram and the first wall");
    assert_eq!(on, 3, "and the second wall, which the plug reaches within the frame");
}
