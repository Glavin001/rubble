#![recursion_limit = "512"]
//! Learned-impact dataset schema, on-disk IO, and a smoke recorder.
//!
//! See `engine/stress-oracle/learned-impact/SCHEMA.md`. Bulk generation and
//! training live elsewhere; this crate only validates the contract and writes
//! synthetic / recorded events.

pub mod catalog;
pub mod constants;
pub mod generate;
pub mod joint_pack;
pub mod oracle;
pub mod progress;
pub mod procedural;
pub mod provider_client;
pub mod outcome_calls;
pub mod outcome_decode;
pub mod outcome_model;
pub mod outcome_program;
pub mod outcome;
pub mod oracle_config;
pub mod peak;
pub mod recorder;
pub mod validate;

pub use constants::*;
pub use generate::{mini_jobs, run_jobs, smoke_jobs, DatasetManifest, Job};
pub use oracle::{prepare_scene, record_event, OracleSession, RecordMeta};
pub use recorder::{EventRecorder, FrameRecord, StaticFeatures};
pub use validate::{validate_event, ValidationError};
