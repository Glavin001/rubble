//! Continuous contact (`docs/continuous-contact-design.md`) in the acceptance crashes that take longest (the rest: `continuous_contact.rs`), 8 runs,
//! generated sedan, edge contact on. The owner's bar (2026-11-26):
//! * **hard**: the collision mesh never overlaps: no edge of one car passes through the
//!   other's collision triangles, and none through its own (checked every 20 ms, every run);
//! * **paint, a guard**: the drawn paint (a flexbody over the coarser node-beam mesh) may
//!   stand a little into the other car while the collision meshes are apart; it is held at
//!   or under [`PAINT_DURING`] during a hit, [`PAINT_AFTER`] from 0.3 s after it and over
//!   1 cm for at most 0.2 s, so it cannot quietly get worse (the owner may tighten them).

mod continuous_common;
use continuous_common::*;

#[test]
fn continuous_head_60_pressed() { accept("head", 60.0, 2.0, true); }
#[test]
fn continuous_head_100() { accept("head", 100.0, 1.0, false); }
#[test]
fn continuous_side_90() { accept("side", 90.0, 1.0, false); }
#[test]
fn continuous_side_60_pressed() { accept("side", 60.0, 2.0, true); }
/// Extremes: both cars at 130 km/h head on, and at 100 km/h with 40 % of the width overlapping.
#[test]
fn continuous_head_130() { accept("head", 130.0, 1.0, false); }
#[test]
fn continuous_offset_100() { accept("offset", 100.0, 1.0, false); }
