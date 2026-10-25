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
    /// How the ground around an activated region (its bed and side walls)
    /// acts on the region's material.
    pub boundary: RegionBoundary,
    /// The stress activated material starts with.
    pub initial_stress: InitialStress,
    /// Grid nodes an absorbing boundary keeps a displacement for: the node
    /// layers within reach of the region's bed and walls (each region needs
    /// `TerrainPlan::boundary_nodes`).
    pub max_boundary_nodes: u32,
}

/// The ground around an activated region: its bed (the floor the window was
/// lowered to) and side walls (where the window meets the unactivated ground).
/// The region's top, and the free surface, are free either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegionBoundary {
    /// Rigid ground: a separating Coulomb contact (the bed friction), which
    /// reflects elastic waves back into the region.
    #[default]
    Rigid,
    /// The ground beyond as an elastic half-space: Lysmer & Kuhlemeyer's (1969)
    /// viscous dashpots (−ρ·c_p·v_n − ρ·c_s·v_t per unit area) with the
    /// viscous-spring boundary's springs (Liu et al. 2006), on the geostatic
    /// traction the ground carried at activation. Waves leave the region as if
    /// into the surrounding ground; material at rest neither sinks nor creeps.
    /// Needs [`InitialStress::Geostatic`] and a material with stiffness.
    Absorbing,
}

/// The stress activated material starts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitialStress {
    /// Undeformed (F = I): the material sags under gravity once it steps, by
    /// ρ·g·H²/(2(λ + 2μ)) at the surface of a column H deep, and rings.
    #[default]
    Unstressed,
    /// The geostatic (K0) state of an elastic layer under its own weight with
    /// no lateral strain: σ_v = −ρ·g·d at depth d below the column's surface
    /// and σ_h = K0·σ_v, K0 = ν/(1 − ν). Set by the first step after
    /// activation (it needs gravity).
    Geostatic,
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
            boundary: RegionBoundary::Rigid,
            initial_stress: InitialStress::Unstressed,
            max_boundary_nodes: 1 << 18,
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
    /// A rock's crack band does not fit the cell: softening over a band of
    /// cells this large would have to snap back (`RockMaterial::max_cell_size`).
    #[error(
        "cell size {cell} m is not below the rock's crack-band limit E·G_f/f_t² = {max} m (its \
         crack band is 2 cells): refine the region, or give the rock parameters for this scale"
    )]
    CrackBand { cell: f32, max: f32 },
    #[error("invalid terrain config: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid region request: {0}")]
    InvalidRequest(&'static str),
    /// A crater source outside the range its published scaling covers.
    #[error("crater source `{quantity}` = {value} is outside the scaled range [{min}, {max}]")]
    SourceOutOfRange {
        quantity: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
    /// The initial excavation field carries more kinetic energy than the
    /// source delivers: its core radius is too small.
    #[error(
        "excavation field energy {field_j} J exceeds the source's {source_j} J: enlarge the core radius"
    )]
    FieldEnergy { field_j: f64, source_j: f64 },
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
        bad(
            self.boundary == RegionBoundary::Absorbing
                && self.initial_stress != InitialStress::Geostatic,
            "an absorbing boundary carries the geostatic traction: it needs initial_stress Geostatic",
        )?;
        bad(
            self.boundary == RegionBoundary::Absorbing && self.max_boundary_nodes == 0,
            "an absorbing boundary needs max_boundary_nodes",
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
