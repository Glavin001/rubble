//! Deformable terrain host logic (`engine/docs/deformable-terrain.md`): ground at
//! rest is a heightfield; around an event, a local active region of GPU MLS-MPM
//! particles carries the moving material, and bakes back to the heightfield when
//! it settles. This crate holds what the host decides: the engine-wide limits
//! ([`TerrainConfig`]), the plan of an active region ([`RegionPlan`]), the
//! materials ([`SandMaterial`] for sand and soil, [`RockMaterial`]) with the CPU
//! mirror of their Slang functions, the GPU records ([`layout`]), and the point
//! sources of km/s impacts and explosions ([`source`], [`zmodel`]): π-scaling
//! crater predictions that size the region, and a Z-model initial velocity
//! field. [`plan`] is the command sequence every host replays. The
//! simulation runs in Slang kernels (`slang/kernels/terrain_mpm.slang`); the
//! host loads material and reads results.

pub mod config;
pub mod layout;
pub mod material;
pub mod plan;
pub mod region;
pub mod source;
pub mod step;
pub mod zmodel;

pub use config::{InitialStress, RegionBoundary, TerrainConfig, TerrainError};
pub use material::{RockMaterial, SandMaterial, TerrainMaterial, YieldSurface};
pub use plan::{RegionClock, TerrainOps, TerrainPlan, TerrainStepReport};
pub use region::{RegionPlan, BLOCK_CELLS};
pub use source::{
    CraterPrediction, CraterScaling, CraterTarget, ExplosionSource, Explosive, ImpactSource, Regime,
};
pub use step::substeps_for;
pub use zmodel::{EjectaScaling, ZModel};
