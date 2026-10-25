//! Planning an active region: its cell size from the predicted transient radius
//! and the configured cells per radius, and its box (a whole number of grid
//! blocks) around the event, padded by a margin so the bed and the sides do not
//! limit the cavity. The plan is checked against [`TerrainConfig`] before any
//! GPU work is scheduled.

use crate::config::{TerrainConfig, TerrainError};
use glam::{IVec3, Vec3};

/// Cells along each axis of a sparse-grid block.
pub const BLOCK_CELLS: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionPlan {
    /// Cell size (m).
    pub dx: f32,
    /// World position of the box's minimum corner: a multiple of the block
    /// size, so regions of the same `dx` share one block lattice.
    pub origin: Vec3,
    /// Box size in blocks.
    pub blocks: IVec3,
}

impl RegionPlan {
    /// A region centred horizontally on `center` (the event at the surface),
    /// reaching `radius · (1 + margin)` sideways and above (for material
    /// thrown up), and as deep below it unless [`TerrainConfig::bed_depth`]
    /// sets the depth (`radius · bed_depth`). `active_regions` and `active_blocks`
    /// are what the world already holds.
    pub fn around(
        config: &TerrainConfig,
        center: Vec3,
        radius: f32,
        margin: f32,
        active_regions: u32,
        active_blocks: u32,
    ) -> Result<Self, TerrainError> {
        config.validate()?;
        if !(center.is_finite() && radius.is_finite() && radius > 0.0) {
            return Err(TerrainError::InvalidRequest(
                "center must be finite and radius positive",
            ));
        }
        if !(margin.is_finite() && margin >= 0.0) {
            return Err(TerrainError::InvalidRequest(
                "margin must be finite and non-negative",
            ));
        }
        TerrainConfig::check(
            "max_regions",
            u64::from(active_regions) + 1,
            u64::from(config.max_regions),
        )?;
        let dx = radius / config.cells_per_radius;
        if dx < config.min_cell_size {
            return Err(TerrainError::CellTooSmall {
                requested: dx,
                allowed: config.min_cell_size,
            });
        }
        let block = dx * BLOCK_CELLS as f32;
        let reach = radius * (1.0 + margin);
        let depth = config.bed_depth.map_or(reach, |d| radius * d);
        let lo = ((center - Vec3::new(reach, depth, reach)) / block).floor();
        let hi = ((center + Vec3::splat(reach)) / block).ceil();
        let blocks = (hi - lo).as_ivec3();
        let count = blocks.x as u64 * blocks.y as u64 * blocks.z as u64;
        TerrainConfig::check(
            "max_blocks",
            u64::from(active_blocks) + count,
            u64::from(config.max_blocks),
        )?;
        Ok(Self {
            dx,
            origin: lo * block,
            blocks,
        })
    }

    /// A region of cells of size `dx` covering the box `[lo, hi]` (rounded out
    /// to whole blocks), for events whose extent is known directly.
    pub fn spanning(
        config: &TerrainConfig,
        lo: Vec3,
        hi: Vec3,
        dx: f32,
        active_regions: u32,
        active_blocks: u32,
    ) -> Result<Self, TerrainError> {
        config.validate()?;
        if !(lo.is_finite() && hi.is_finite() && hi.cmpgt(lo).all() && dx.is_finite() && dx > 0.0) {
            return Err(TerrainError::InvalidRequest(
                "box must be finite and non-empty, dx positive",
            ));
        }
        TerrainConfig::check(
            "max_regions",
            u64::from(active_regions) + 1,
            u64::from(config.max_regions),
        )?;
        if dx < config.min_cell_size {
            return Err(TerrainError::CellTooSmall {
                requested: dx,
                allowed: config.min_cell_size,
            });
        }
        let block = dx * BLOCK_CELLS as f32;
        let lo = (lo / block).floor();
        let hi = (hi / block).ceil();
        let blocks = (hi - lo).as_ivec3();
        let count = blocks.x as u64 * blocks.y as u64 * blocks.z as u64;
        TerrainConfig::check(
            "max_blocks",
            u64::from(active_blocks) + count,
            u64::from(config.max_blocks),
        )?;
        Ok(Self {
            dx,
            origin: lo * block,
            blocks,
        })
    }

    /// The box's maximum corner.
    pub fn end(&self) -> Vec3 {
        self.origin + self.blocks.as_vec3() * (self.dx * BLOCK_CELLS as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_box_covers_the_padded_radius_on_the_block_lattice() {
        let c = TerrainConfig::default();
        let center = Vec3::new(3.3, -1.0, 7.9);
        let p = RegionPlan::around(&c, center, 2.0, 0.5, 0, 0).unwrap();
        assert_eq!(p.dx, 2.0 / c.cells_per_radius);
        let block = p.dx * BLOCK_CELLS as f32;
        for k in 0..3 {
            assert!(p.origin[k] <= center[k] - 3.0 + 1e-5 && p.end()[k] >= center[k] + 3.0 - 1e-5);
            assert!(p.end()[k] - p.origin[k] <= 2.0 * 3.0 + 2.0 * block + 1e-4);
            let m = p.origin[k] / block;
            assert!((m - m.round()).abs() < 1e-4, "origin off the lattice");
        }
    }

    #[test]
    fn the_bed_depth_is_its_own_setting() {
        let c = TerrainConfig {
            bed_depth: Some(1.25),
            ..TerrainConfig::default()
        };
        let center = Vec3::new(0.0, 0.0, 0.0);
        let p = RegionPlan::around(&c, center, 2.0, 1.5, 0, 0).unwrap();
        let deep = TerrainConfig {
            bed_depth: None,
            ..TerrainConfig::default()
        };
        let q = RegionPlan::around(&deep, center, 2.0, 1.5, 0, 0).unwrap();
        let block = p.dx * BLOCK_CELLS as f32;
        // The bed within a block below radius · bed_depth; sides and top unchanged.
        assert!(p.origin.y <= -2.5 + 1e-5 && p.origin.y > -2.5 - block - 1e-5);
        assert!(q.origin.y <= -5.0 + 1e-5);
        assert_eq!((p.origin.x, p.origin.z), (q.origin.x, q.origin.z));
        assert_eq!(p.end(), q.end());
        let bad = TerrainConfig {
            bed_depth: Some(0.0),
            ..TerrainConfig::default()
        };
        assert!(RegionPlan::around(&bad, center, 2.0, 1.5, 0, 0).is_err());
    }

    #[test]
    fn requests_past_a_limit_fail_naming_it() {
        let c = TerrainConfig {
            max_blocks: 100,
            ..TerrainConfig::default()
        };
        let err = RegionPlan::around(&c, Vec3::ZERO, 1.0, 0.5, 0, 0).unwrap_err();
        assert!(
            matches!(
                err,
                TerrainError::LimitExceeded {
                    limit: "max_blocks",
                    ..
                }
            ),
            "{err}"
        );
        let c = TerrainConfig::default();
        let err = RegionPlan::around(&c, Vec3::ZERO, 1.0, 0.5, c.max_regions, 0).unwrap_err();
        assert!(
            matches!(
                err,
                TerrainError::LimitExceeded {
                    limit: "max_regions",
                    ..
                }
            ),
            "{err}"
        );
        let err = RegionPlan::around(&c, Vec3::ZERO, 0.05, 0.5, 0, 0).unwrap_err();
        assert!(matches!(err, TerrainError::CellTooSmall { .. }), "{err}");
        assert!(RegionPlan::around(&c, Vec3::NAN, 1.0, 0.5, 0, 0).is_err());
    }
}
