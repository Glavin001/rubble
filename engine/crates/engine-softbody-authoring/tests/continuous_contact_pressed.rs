//! Continuous contact (`docs/continuous-contact-design.md`) in the pressed acceptance crashes (2 s each; the others: `continuous_contact.rs`, `continuous_contact_hard.rs`) (the rest: `continuous_contact.rs`), 8 runs,
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
fn continuous_side_60_pressed() { accept("side", 60.0, 2.0, true); }
#[test]
fn continuous_head_60_pressed() { accept("head", 60.0, 2.0, true); }
