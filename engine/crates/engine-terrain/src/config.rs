//! Engine-wide terrain limits. Nothing degrades silently (AGENTS.md): a request
//! past any limit fails with [`TerrainError`], naming the limit, what was asked
//! for and what is allowed.

/// Limits of the deformable-terrain subsystem, fixed when a world is created
/// (GPU buffers are sized from them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainConfig {
    /// Active areas open at once (each an activated box over the heightfield
    /// with its own window, absorbing faces and bake; at most
    /// [`crate::layout::MPM_MAX_AREAS`]). Requests whose boxes come near an
    /// open area grow that area instead of opening another.
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
    /// One cell size (m) for every event on this terrain, or 0: the first
    /// event's own (from `cells_per_radius`) while any area is open. All open
    /// areas share one sparse grid, so they share its cell size.
    pub cell_size: f32,
    /// The coarsest a later event may be resolved, as a fraction of the cells
    /// it asks for: an event joins the open grid (or the fixed `cell_size`)
    /// only if that cell is at most its own asked cell / `min_resolution`, so
    /// at 0.5 an impact asking 16 cells per radius runs at no fewer than 8.
    /// A coarser grid fails with [`TerrainError::CellTooCoarse`]; a finer one
    /// is always accepted (it costs particles, which the limits bound).
    pub min_resolution: f32,
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
    /// How far an activated particle moves sideways (in spacings of the
    /// activation lattice, `cells` / `particles_per_cell_axis`) before the
    /// bake treats it as material the MPM moved rather than as the heightfield
    /// it came from.
    ///
    /// Activation lays a sample's material out as a lattice of particles
    /// across the sample's cell. At rest each particle stands for a share of
    /// that one sample: the bake puts it back there with the bilinear tent
    /// (its home sample plus its displacement), and activate-then-bake is the
    /// identity. Material the MPM moved is baked with the kernel P2G gives the
    /// grid its mass with, the quadratic B-spline, at its own position: the
    /// simulation resolves nothing finer, and finer arrangements of its
    /// particles (sub-cell cracks along a crater's mirror planes, columns
    /// stretched over several samples) would otherwise bake as trenches or
    /// ridges a sample wide. The share at home falls linearly from 1 at rest
    /// to 0 at this displacement.
    ///
    /// Half a spacing (the default) is where a particle leaves its own
    /// lattice cell: past it, where it sat in the rest lattice no longer says
    /// where its volume is. On the browser demo's crater the axes then stand
    /// within 0.5 m of their neighbours, against 2.2 m at two spacings
    /// (`engine/docs/deformable-terrain.md`, "The splat bake").
    pub bake_home_fade: f32,
    /// How the ground around an activated region (its bed and side walls)
    /// acts on the region's material.
    pub boundary: RegionBoundary,
    /// The stress activated material starts with.
    pub initial_stress: InitialStress,
    /// Face points an absorbing boundary keeps a displacement for: the node
    /// lattice's columns through the region's bed and walls (each region needs
    /// `TerrainPlan::boundary_nodes`).
    pub max_boundary_nodes: u32,
    /// How far below the event a region planned around a radius
    /// ([`crate::RegionPlan::around`], the π-scaled impact and explosion
    /// regions) puts its bed, in those radii: `None` reaches as deep as it
    /// reaches sideways, radius · (1 + margin). With the absorbing ground the
    /// bed need not sit that deep (`engine/docs/deformable-terrain.md`,
    /// "Performance"); the region's sides and top keep the margin.
    pub bed_depth: Option<f32>,
    /// The open region's frame clock ([`crate::plan::RegionClock`]) bounds a
    /// frame's material speed by this factor times the last reported speed
    /// (or the source's peak, until the first report), plus gravity over the
    /// readback lag and the host's headroom. The GPU checks the bound each
    /// substep (an error past it), so a smaller factor never degrades a run
    /// silently: it takes fewer substeps while speeds fall, and fails a frame
    /// whose material outruns it.
    pub speed_growth: f32,
}

/// Which settings a world's terrain starts from ([`TerrainConfig::profile`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainProfile {
    /// Physical: the settings the validations run at (cells per radius 16,
    /// Courant 0.3, the clock's 1.5× speed growth).
    #[default]
    Physical,
    /// Games: a convincing crater in real time at 120 Hz. Eight cells per
    /// transient radius, the bed 1.25 R deep, Courant 0.45 and no growth on
    /// the clock's speed bound; the sand keeps its physical stiffness. Each
    /// setting's measured effect on the crater and the frame time is in
    /// `engine/docs/deformable-terrain.md`, "Performance".
    Game,
}

/// The ground around an activated region: its bed (the floor the window was
/// lowered to) and side walls (where the window meets the unactivated ground).
/// The region's top, and the free surface, are free either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RegionBoundary {
    /// Rigid ground: a separating Coulomb contact (the bed friction), which
    /// reflects elastic waves back into the region.
    Rigid,
    /// The ground beyond as an elastic half-space: Lysmer & Kuhlemeyer's (1969)
    /// viscous dashpots (−ρ·c_p·v_n − ρ·c_s·v_t per unit area) with the
    /// viscous-spring boundary's springs (Liu et al. 2006), on the geostatic
    /// traction the ground carried at activation. Waves leave the region as if
    /// into the surrounding ground; material at rest neither sinks nor creeps.
    /// Needs [`InitialStress::Geostatic`] and a material with stiffness. The
    /// default: the ground beyond the region is more ground.
    #[default]
    Absorbing,
}

/// The stress activated material starts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitialStress {
    /// Undeformed (F = I): the material sags under gravity once it steps, by
    /// ρ·g·H²/(2(λ + 2μ)) at the surface of a column H deep, and rings.
    Unstressed,
    /// The geostatic (K0) state of an elastic layer under its own weight with
    /// no lateral strain: σ_v = −ρ·g·d at depth d below the column's surface
    /// and σ_h = K0·σ_v, K0 = ν/(1 − ν). Set by the first step after
    /// activation (it needs gravity). The default.
    #[default]
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
            cell_size: 0.0,
            min_resolution: 0.5,
            courant: 0.3,
            max_substeps: 512,
            max_columns: 1 << 16,
            max_coupled: 256,
            max_field_samples: 1 << 20,
            bake_home_fade: 0.5,
            boundary: RegionBoundary::Absorbing,
            initial_stress: InitialStress::Geostatic,
            max_boundary_nodes: 1 << 18,
            bed_depth: Some(1.5),
            speed_growth: 1.5,
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
    /// An event asks for cells of `asked` m, and the open terrain grid (or the
    /// fixed `cell_size`) has cells of `grid` m: coarser than `min_resolution`
    /// allows.
    #[error(
        "the terrain grid's cells ({grid} m) are too coarse for this event, which asks for \
         {asked} m cells (`min_resolution` {min_resolution}): wait for the open areas to bake, \
         or use a finer `cell_size`"
    )]
    CellTooCoarse {
        asked: f32,
        grid: f32,
        min_resolution: f32,
    },
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
    /// The default limits with `profile`'s settings.
    pub fn profile(profile: TerrainProfile) -> Self {
        let d = Self::default();
        match profile {
            TerrainProfile::Physical => d,
            TerrainProfile::Game => Self {
                cells_per_radius: 8.0,
                bed_depth: Some(1.25),
                courant: 0.45,
                speed_growth: 1.0,
                ..d
            },
        }
    }

    /// `profile`'s settings over these limits (the other fields kept).
    pub fn with_profile(self, profile: TerrainProfile) -> Self {
        let p = Self::profile(profile);
        Self {
            cells_per_radius: p.cells_per_radius,
            bed_depth: p.bed_depth,
            courant: p.courant,
            speed_growth: p.speed_growth,
            ..self
        }
    }

    pub fn validate(&self) -> Result<(), TerrainError> {
        let bad = |c: bool, why| {
            if c {
                Err(TerrainError::InvalidConfig(why))
            } else {
                Ok(())
            }
        };
        bad(self.max_regions == 0, "max_regions must be at least 1")?;
        bad(
            self.max_regions > crate::layout::MPM_MAX_AREAS,
            "max_regions must be at most MPM_MAX_AREAS (16)",
        )?;
        bad(
            !(self.cell_size.is_finite()
                && (self.cell_size == 0.0 || self.cell_size >= self.min_cell_size)),
            "cell_size must be 0 (from the events) or at least min_cell_size",
        )?;
        bad(
            !(self.min_resolution > 0.0 && self.min_resolution <= 1.0),
            "min_resolution must be in (0, 1]",
        )?;
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
            !(self.bake_home_fade.is_finite() && self.bake_home_fade > 0.0),
            "bake_home_fade must be finite and positive",
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
        bad(
            self.bed_depth.is_some_and(|d| !(d.is_finite() && d > 0.0)),
            "bed_depth must be finite and positive",
        )?;
        bad(
            !(self.speed_growth.is_finite() && self.speed_growth >= 1.0),
            "speed_growth must be finite and at least 1",
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
