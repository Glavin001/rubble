//! Learned-impact dataset schema, on-disk IO, and a smoke recorder.
//!
//! See `engine/stress-oracle/learned-impact/SCHEMA.md`. Bulk generation and
//! training live elsewhere; this crate only validates the contract and writes
//! synthetic / recorded events.

pub mod catalog;
pub mod constants;
pub mod generate;
pub mod joint_pack;
pub mod live;
pub mod oracle;
pub mod peak;
pub mod recorder;
pub mod validate;

pub use constants::*;
pub use generate::{
    diversity_jobs, mini_jobs, prove_jobs, run_jobs, run_jobs_with, smoke_jobs, wall_jobs,
    DatasetManifest, Job, RunJobsOpts,
};
pub use live::{
    bond_keys_for_edges, fail_edge_indices, pack_live_features, LiveFeatures, FAIL_BINS, FEATURE_GAPS,
    OUT_PER_EDGE,
};
pub use oracle::{prepare_scene, record_event, OracleSession, RecordMeta};
pub use recorder::{EventRecorder, FrameRecord, StaticFeatures};
pub use validate::{validate_event, ValidationError};
