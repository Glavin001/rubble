//! Deformable terrain host logic (`engine/docs/deformable-terrain.md`): ground at
//! rest is a heightfield; around an event, a local active region of GPU MLS-MPM
//! particles carries the moving material, and bakes back to the heightfield when
//! it settles. This crate holds what the host decides: the engine-wide limits
//! ([`TerrainConfig`]), the plan of an active region ([`RegionPlan`]), the sand
//! material ([`SandMaterial`]) with the CPU mirror of its Slang functions, the
//! GPU records ([`layout`]), and the point sources of km/s impacts and
//! explosions ([`source`], [`zmodel`]): π-scaling crater predictions that size
//! the region, and a Z-model initial velocity field. The simulation runs in
//! Slang kernels (`slang/kernels/terrain_mpm.slang`); the host loads material
//! and reads results.

pub mod config;
pub mod layout;
pub mod material;
pub mod region;
pub mod source;
pub mod step;
pub mod zmodel;

pub use config::{TerrainConfig, TerrainError};
pub use material::{SandMaterial, YieldSurface};
pub use region::{RegionPlan, BLOCK_CELLS};
pub use source::{
    CraterPrediction, CraterScaling, CraterTarget, ExplosionSource, Explosive, ImpactSource, Regime,
};
pub use step::substeps_for;
pub use zmodel::{EjectaScaling, ZModel};
