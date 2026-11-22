//! engine-softbody-authoring's host-only integration tests in one binary (each file a module): one
//! link instead of one per file (a release test build links each with thin LTO).
//! Run one file's: `cargo test -p engine-softbody-authoring --test host <file>::`.

#[path = "../invariants_common/mod.rs"]
mod invariants_common;
mod authoring_census;
mod crash_rigs;
mod damper_tear;
mod debris_ttl;
mod free_flight;
mod invariants_small;
mod own_travel_reach;
mod sedan_hinges;
mod sedan_parts_collide;
mod shell_lattice;
mod showcase_rest_wobble;
mod vehicle_lint;
