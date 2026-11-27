//! Crash zones: which region of the car each part is, for the crash reference suite's
//! measurements (`engine_softbody::crash_reference::Section`; the rigger writes them per
//! node name as `crash.json` beside the JBeam).
//!
//! A part states its zone where it is defined: every mechanical and interior part when it
//! is made ([`crate::mechanical::MechPart::zone`]), every body-in-white member type here
//! ([`member_zone`]), each body panel in the rigger. A member type or part without one
//! fails `tests/crash_zones.rs`, never silently becomes [`Zone::Other`].

use serde::{Deserialize, Serialize};

/// A region of the car (the names and meanings are the crash reference's sections).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Zone {
    /// Bumper beam and fascia, crash boxes, front rails, radiator support, front
    /// subframe: the low load path ahead of the firewall.
    FrontLower,
    /// Hood, fenders, cowl, upper rails, strut towers: the high load path ahead of the
    /// firewall.
    FrontUpper,
    /// Pillars, sills, roof, floor, firewall, dash: the survival cell.
    Cabin,
    Door,
    /// Behind the rear seats: trunk or tailgate, rear rails and floor, quarters, rear
    /// bumper, rear subframe.
    Rear,
    /// Wheels, hubs, arms, struts, links, steering.
    Wheel,
    /// Engine, driveline, exhaust, fuel tank, cooling, seats, glass: carried, not
    /// structure.
    Other,
}

/// The zone of a body-in-white member type ([`crate::structure::Member::kind`]); None: a
/// type with no zone yet (a test fails).
pub fn member_zone(kind: &str) -> Option<Zone> {
    Some(match kind {
        "crash_box" | "front_rail" | "radiator_support" => Zone::FrontLower,
        "upper_rail" | "strut_tower" | "cowl" => Zone::FrontUpper,
        "firewall" | "floor_pan" | "sill" | "seat_crossmember" | "roof_panel" | "a_pillar"
        | "roof_rail" | "b_pillar" | "c_pillar" | "roof_bow" => Zone::Cabin,
        "boot_floor" | "rear_rail" | "rear_tower" | "quarter" | "rear_shelf" | "tail_panel" => {
            Zone::Rear
        }
        _ => return None,
    })
}
