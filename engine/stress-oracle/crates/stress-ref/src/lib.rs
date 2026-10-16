//! `stress-ref`: the double-precision CPU reference solver for elastodynamic
//! destruction stress, and the harness that compares it against external oracles.
//!
//! Layering (see `README.md` of the workspace for the full picture):
//! * [`scene`] — the shared scene format every solver and oracle exporter reads;
//! * [`material`], [`bond`], [`joint`] — material laws, bond stiffness and the
//!   damageable joint (failure surface, softening, cracked-joint contact);
//! * [`solver`], [`statics`], [`implicit`] — clusters in a floating frame, explicit
//!   substeps, quasi-static solves, implicit Newmark steps, fracture and splitting;
//! * [`world`] — a standalone rigid world (impactors, ground, contact, blast,
//!   scripted loads and events) that drives the solver for reference runs;
//! * [`observation`], [`metrics`] — solver-independent results and the metrics
//!   computed identically from ours and every oracle's observation;
//! * [`api`] — the engine-facing trait for swapping stress-solver implementations.

// Numeric kernels index several parallel arrays with one loop variable, and the
// validators write `!(x > 0.0)` on purpose so that NaN fails.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod api;
pub mod blast;
pub mod bond;
pub mod builders;
pub mod cli;
pub mod contact;
pub mod engine;
pub mod hull;
pub mod implicit;
pub mod joint;
pub mod material;
pub mod math;
pub mod metrics;
pub mod observation;
pub mod par;
pub mod polytope;
pub mod profile;
pub mod refine;
pub mod rng;
pub mod scene;
pub mod scene_pack;
pub mod shardwright;
pub mod scenes;
pub mod showcases;
pub mod snapshot;
pub mod solver;
pub mod statics;
pub mod structure;
pub mod world;
