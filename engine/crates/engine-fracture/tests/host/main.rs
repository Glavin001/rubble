//! engine-fracture's host-only integration tests in one binary (each file a module): one
//! link instead of one per file (a release test build links each with thin LTO).
//! Run one file's: `cargo test -p engine-fracture --test host <file>::`.

#[path = "../common/mod.rs"]
mod common;
mod asset_hierarchy;
mod assets;
mod bond_lines;
mod bonds;
mod cook;
mod cuboid;
mod house_levels;
mod hulls;
mod pack;
mod projectile;
mod view;
