//! What a host says in place of an environment variable it does not have (a browser's
//! page, which has controls instead: the vehicle lab's "surface contact on the fast
//! schedules" and "cars apart on the wide passes"), read by the settings' defaults
//! below and [`crate::explicit_plan::vv_wide::vv_wide_default`] before the variable itself. Process
//! wide, as the environment is: set before the fleet's pipelines are listed and the
//! host is built.
//!
//! Also the two surface settings' defaults, which read it (re-exported by
//! [`super`]).

use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
static SURFACE_FAST: AtomicU8 = AtomicU8::new(0);
static VV_WIDE: AtomicU8 = AtomicU8::new(0);

/// `RUBBLE_SURFACE_FAST` for this process: `Some(0)` neither part, `Some(1)` a wide
/// frame's part alone (which no value of the variable says), `Some(2)` both;
/// `None` the variable's own value again.
pub fn set_surface_fast(parts: Option<u8>) {
    SURFACE_FAST.store(parts.map_or(0, |p| 1 + p.min(2)), Relaxed);
}

/// `RUBBLE_VV_WIDE` for this process; `None` the variable's own value again.
pub fn set_vv_wide(on: Option<bool>) {
    VV_WIDE.store(on.map_or(0, |on| 1 + u8::from(on)), Relaxed);
}

/// The parts of surface contact on the fast schedules a host asked for, or the
/// variable's (`1`: both, `0`: neither); `None`: neither says.
pub fn surface_fast() -> Option<u8> {
    match SURFACE_FAST.load(Relaxed) {
        0 => match std::env::var("RUBBLE_SURFACE_FAST").as_deref() {
            Ok("0") => Some(0),
            Ok("1") => Some(2),
            _ => None,
        },
        p => Some(p - 1),
    }
}

/// Vehicle contact's frames without a group on the wide passes as a host asked, or
/// the variable's; `None`: neither says.
pub fn vv_wide() -> Option<bool> {
    match VV_WIDE.load(Relaxed) {
        0 => match std::env::var("RUBBLE_VV_WIDE").as_deref() {
            Ok("1") => Some(true),
            Ok("0") => Some(false),
            _ => None,
        },
        v => Some(v == 2),
    }
}

/// Whether hosts run a fleet with surface contact on the fast schedules, its pass as
/// pipelines ([`crate::explicit_plan::FramePlanner::set_surface_fast`]): on natively, on CUDA and in a browser
/// unless `RUBBLE_SURFACE_FAST=0` ([this module](self)). In Chrome (wasm32) its wide frames'
/// part held bit for bit on Chrome's A/B (AGENTS.md, rung 3: a lone car into a pole and
/// a wall, 40 s each) and took a lone surface car from 17 to 6.5 ms a step
/// (vehicle-perf-first-principles.md §8.64); its vehicle-contact part
/// ([`surface_fast_contact_default`]) stays off there.
pub fn surface_fast_default() -> bool {
    match surface_fast() {
        Some(parts) => parts > 0,
        None => true,
    }
}

/// [`crate::explicit_plan::FramePlanner::set_surface_fast_contact`]'s default: with `RUBBLE_SURFACE_FAST=1`
/// on ([this module](self): as the host asked); otherwise the host's own (`host`). On for
/// CUDA: on a T4 every goal row is faster with it (-6 to -34 %). Off for wgpu: on Metal
/// (M3 Max) two derby rows are 7.5 and 9.7 % slower with it and two head-on rows 3.6
/// and 4.7 %, the others 1 to 17 % faster (vehicle-perf-first-principles.md §8.53); a
/// change that slows a row does not become a default. Off in a browser too: on Chrome's
/// A/B (8 cars in the derby, t 4 s) it differs where two nodes' accelerations are NaN on
/// both schedules (§8.64, known-bugs).
pub fn surface_fast_contact_default(host: bool) -> bool {
    match surface_fast() {
        Some(2) => true,
        Some(_) => false,
        None => host,
    }
}
