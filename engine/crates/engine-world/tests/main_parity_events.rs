//! Fields the branch added to the step results default to main's behaviour: nothing
//! reported for a World without explicit vehicles (PR #105 regression audit). CPU only.

use rubble::{coalesce, Frame};

#[test]
fn a_frame_without_explicit_vehicles_reports_no_sleep_changes() {
    let f = Frame::default();
    assert!(f.explicit_sleep.is_empty());
    let frames: Vec<Frame> = (0..3)
        .map(|k| Frame { step: k, ..Default::default() })
        .collect();
    let c = coalesce(frames);
    assert!(c.explicit_sleep.is_empty(), "coalesce made sleep events from none");
    assert!(c.steps.is_some(), "the folded steps are reported");
}
