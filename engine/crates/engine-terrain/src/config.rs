//! Engine-wide terrain limits. Nothing degrades silently (AGENTS.md): a request
//! past any limit fails with [`TerrainError`], naming the limit, what was asked
//! for and what is allowed.

/// Limits of the deformable-terrain subsystem, fixed when a world is created
/// (GPU buffers are sized from them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainConfig {
    /// Active regions alive at once.
    pub max_regions: u32,
    /// MPM particles over all regions.
    pub max_particles: u32,
    /// Sparse-grid blocks ([`crate::BLOCK_CELLS`]³ cells each) over all regions.
    pub max_blocks: u32,
    /// Ejecta parcels one region may throw (0: material leaving the region's box
    /// is an error instead).
    pub max_parcels: u32,
    /// Particles seeded per cell when material is activated (per axis: the
    /// cell holds this many cubed).
    pub particles_per_cell_axis: u32,
    /// Cells across a region's predicted transient radius: sets its cell size.
    pub cells_per_radius: f32,
    /// Smallest cell size a region may use (m).
    pub min_cell_size: f32,
    /// Courant number of the explicit substeps: no material point moves, and
    /// no elastic wave travels, more than this fraction of a cell per substep.
    pub courant: f32,
    /// Substeps one call may run (a frame needing more fails, rather than
    /// taking longer substeps than CFL allows).
    pub max_substeps: u32,
    /// Heightfield samples one region may activate.
    pub max_columns: u32,
    /// Rigid colliders the active region may hold at once (each couples to the
    /// material and takes its reaction).
    pub max_coupled: u32,
    /// Samples of the bed heightfield parcels may land on.
    pub max_field_samples: u32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            max_regions: 4,
            max_particles: 1 << 20,
            max_blocks: 1 << 14,
            max_parcels: 1 << 16,
            particles_per_cell_axis: 2,
            cells_per_radius: 16.0,
            min_cell_size: 0.01,
            courant: 0.3,
            max_substeps: 512,
            max_columns: 1 << 16,
            max_coupled: 256,
            max_field_samples: 1 << 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TerrainError {
    #[error("terrain limit `{limit}` exceeded: {requested} requested, {allowed} allowed")]
    LimitExceeded {
        limit: &'static str,
        requested: u64,
        allowed: u64,
    },
    #[error("cell size {requested} m is below `min_cell_size` ({allowed} m)")]
    CellTooSmall { requested: f32, allowed: f32 },
    #[error("invalid terrain config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid region request: {0}")]
    InvalidRequest(&'static str),
}

impl TerrainConfig {
    pub fn validate(&self) -> Result<(), TerrainError> {
        let bad = |c: bool, why| {
            if c {
                Err(TerrainError::InvalidConfig(why))
            } else {
                Ok(())
            }
        };
        bad(self.max_regions == 0, "max_regions must be at least 1")?;
        bad(self.max_particles == 0, "max_particles must be at least 1")?;
        bad(self.max_blocks == 0, "max_blocks must be at least 1")?;
        bad(
            self.particles_per_cell_axis == 0,
            "particles_per_cell_axis must be at least 1",
        )?;
        bad(
            !(self.cells_per_radius.is_finite() && self.cells_per_radius >= 1.0),
            "cells_per_radius must be finite and at least 1",
        )?;
        bad(
            !(self.min_cell_size.is_finite() && self.min_cell_size > 0.0),
            "min_cell_size must be finite and positive",
        )?;
        bad(
            !(self.courant > 0.0 && self.courant <= 1.0),
            "courant must be in (0, 1]",
        )?;
        bad(self.max_substeps == 0, "max_substeps must be at least 1")?;
        bad(self.max_columns == 0, "max_columns must be at least 1")?;
        bad(
            self.max_coupled == 0 || self.max_coupled > 65535,
            "max_coupled must lie in [1, 65535]",
        )?;
        Ok(())
    }

    /// `requested` of `limit` against `allowed`.
    pub fn check(limit: &'static str, requested: u64, allowed: u64) -> Result<(), TerrainError> {
        if requested > allowed {
            return Err(TerrainError::LimitExceeded {
                limit,
                requested,
                allowed,
            });
        }
        Ok(())
    }
}
