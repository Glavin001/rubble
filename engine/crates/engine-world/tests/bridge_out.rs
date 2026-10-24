//! Bridge Out with real traffic (`bo_drive_viaduct`), through the pages' host
//! (`rubble::scenario`): the viaduct as a `Threshold` destructible and raycast vehicles
//! (rubble's presets) cruising both ways on its deck and along the lower road under it. The
//! hinge pier's charge (stress-ref `vehicles` `hinge_charge`) is cut between two steps
//! (`World::destructible_cut_bonds`): the drop-in spans come down with the traffic on them.
//! The vehicles' wheels keep finding the deck as it splits (its bodies change under them)
//! and load it (W5).
//!
//! (The lab's ball alone breaks the bonds it meets and no more here: the threshold model
//! breaks a bond only under a contact's load, so the spans, still bonded at their seats,
//! stay up. Under `blast_parity` their weight breaks those links, stress-rubble
//! `wrecking_yard` `viaduct_pier_hit_drops_the_drop_in_spans`.)
#![cfg(feature = "stress")]

use rubble::scenario::Scenario;
use rubble::*;

/// Per drive: (name, the most it came down (m) below where it settled).
type Drops = Vec<(String, f32)>;

/// 9 s of traffic, the hinge pier's charge cut 2 s in (`charge`): each drive's drop and
/// the bonds broken.
fn run(charge: bool) -> (Drops, usize) {
    let pack = engine_fracture::samples::pack_sample_json("bo_drive_viaduct").unwrap();
    let mut s = Scenario::load(GpuContext::Default, &pack).unwrap();
    assert_eq!((s.cars.len(), s.charges.len()), (8, 1), "eight drives and the hinge pier's charge");
    let n = s.cars.len();
    let (mut settled, mut low) = (vec![0.0f32; n], vec![f32::INFINITY; n]);
    for step in 0..540 {
        if charge && step == 120 {
            let c = s.charges[0].clone();
            assert_eq!(s.cut(&c).unwrap(), c.bonds.len(), "every bond of the charge was intact");
        }
        s.step().unwrap();
        s.poll().unwrap();
        for (k, car) in s.cars.iter().enumerate() {
            let y = s.world.read_body_state(car.build.chassis).unwrap().transform.position[1];
            // Settled on its springs half a second in (it starts at full droop).
            if step == 30 {
                settled[k] = y;
            }
            if step >= 30 {
                low[k] = low[k].min(y);
            }
        }
    }
    (s.cars.iter().enumerate().map(|(k, c)| (c.drive.name.clone(), settled[k] - low[k])).collect(), s.broken)
}

/// Without the charge the traffic runs over the bridge and under it and nothing breaks:
/// the deck carries its wheels (W5) and every vehicle stays on its road. With it, the
/// four vehicles on the drop-in spans when it goes come down with them, most of the
/// deck's height (10 m).
#[test]
fn the_drop_in_spans_take_their_traffic_down() {
    let (drops, broken) = run(false);
    assert_eq!(broken, 0, "no charge: {drops:?}");
    for (name, d) in &drops {
        // (A sedan on the lower road runs off its far end, 0.45 m down to the ground.)
        assert!(*d < 0.5, "no charge: {name} came down {d} m");
    }
    let (drops, broken) = run(true);
    assert!(broken > 100, "{broken} bonds");
    for name in ["east-truck", "east-2", "west-truck", "west-2"] {
        let d = drops.iter().find(|x| x.0 == name).unwrap().1;
        assert!(d > 5.0, "{name} came down {d} m with its span: {drops:?}");
    }
}
