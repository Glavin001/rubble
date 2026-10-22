#![recursion_limit = "512"]
//! Learned-impact dataset schema, on-disk IO, and a smoke recorder.
//!
//! See `engine/stress-oracle/learned-impact/SCHEMA.md`. Bulk generation and
//! training live elsewhere; this crate only validates the contract and writes
//! synthetic / recorded events.

pub mod catalog;
pub mod constants;
pub mod generate;
pub mod forcing;
pub mod joint_pack;
pub mod joint_history;
pub mod response_history;
pub mod live;
pub mod layouts;
pub mod oracle;
pub mod oracle_config;
pub mod full_state;
pub mod full_state_calls;
pub mod outcome;
pub mod outcome_calls;
pub mod procedural;
pub mod stage2;
pub mod progress;
pub mod peak;
pub mod permutations;
pub mod recorder;
pub mod replay;
pub mod validate;

pub use constants::*;
pub use generate::{
    diversity_jobs, diversity_v2_jobs, mini_jobs, prove_jobs, run_jobs, run_jobs_with, smoke_jobs,
    wall_jobs, DatasetManifest, Job, RunJobsOpts,
};
pub use live::{
    bond_keys_for_edges, fail_edge_indices, pack_live_features, LiveFeatures, FAIL_BINS, FEATURE_GAPS,
    OUT_PER_EDGE,
};
pub use oracle::{prepare_scene, record_event, OracleSession, RecordMeta};
pub use recorder::{EventRecorder, FrameRecord, StaticFeatures};
pub use validate::{validate_event, ValidationError};

/// Causal native-model snapshots; no predicted state is applied to the solver.
pub mod runtime_snapshot;
/// Experimental independent endpoint application contract.
pub mod runtime_endpoint;
/// Lossless native stress-phase targets for interval-trajectory learning.
pub mod phase_history;
/// Complete learned forecasts applied at the native stress phase boundary.
pub mod runtime_trajectory;
pub mod causal_state;
/// Causal full-graph physical-state transition pairs, before rigid-frame transfer.
pub mod graph_transition;
pub mod graph_live;
