//! Deformable terrain host logic (`engine/docs/deformable-terrain.md`): ground at
//! rest is a heightfield; around an event, a local active region of GPU MLS-MPM
//! particles carries the moving material, and bakes back to the heightfield when
//! it settles. This crate holds what the host decides: the engine-wide limits
//! ([`TerrainConfig`]), the plan of an active region ([`RegionPlan`]) and the GPU
//! records ([`layout`]). The simulation runs in Slang kernels
//! (`slang/kernels/terrain_mpm.slang`); the host loads material and reads results.

pub mod config;
pub mod layout;
pub mod region;

pub use config::{TerrainConfig, TerrainError};
pub use region::{RegionPlan, BLOCK_CELLS};
