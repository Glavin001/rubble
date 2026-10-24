//! Deformable terrain host logic (`engine/docs/deformable-terrain.md`): ground at
//! rest is a heightfield; around an event, a local active region of GPU MLS-MPM
//! particles carries the moving material, and bakes back to the heightfield when
//! it settles. This crate holds what the host decides: the engine-wide limits
//! ([`TerrainConfig`]), the plan of an active region ([`RegionPlan`]) and the
//! sand material ([`SandMaterial`]) with the CPU mirror of its Slang functions.
//! The simulation itself runs in Slang kernels; the host never touches particles.

pub mod config;
pub mod material;
pub mod region;

pub use config::{TerrainConfig, TerrainError};
pub use material::SandMaterial;
pub use region::{RegionPlan, BLOCK_CELLS};
