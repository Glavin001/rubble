//! Render-only: the live ground surface over an open region, rebuilt every
//! frame on the GPU from its particles (`slang/kernels/terrain_surface.slang`;
//! `engine/docs/deformable-terrain.md`, "Rendering and showcase").
//!
//! A [`SurfacePlan`] sizes the surface buffer ([`SURFACE_BINDING`]) and turns
//! the open region of a [`TerrainPlan`] into one compute pass: the occupancy
//! cleared, the absorbing walls' give measured (as the bake measures it), every
//! particle splatted as the bake splats it, and one height per window sample
//! resolved from the material attached to the bed. A renderer draws those
//! heights as the ground over the window and the material above them (and the
//! parcels) as debris. The simulation never reads the buffer.

use crate::config::{TerrainConfig, TerrainError};
use crate::layout::{MPM_BOUNDARY_ABSORBING, MPM_NO_BED};
use crate::plan::{TerrainCmd, TerrainOps, TerrainPlan, TerrainWrite, ARGS_PARTICLES};

/// The surface buffer's binding (after the terrain look-ahead's 52).
pub const SURFACE_BINDING: usize = 53;

/// Every kernel a [`SurfacePlan`] dispatches (`mpm_write_args` refreshes the
/// per-particle indirect counts from the counters, which the frame left).
pub const SURFACE_KERNELS: &[&str] = &[
    "mpm_write_args",
    "terrain_surface_clear",
    "terrain_surface_shift",
    "terrain_surface_bin",
    "terrain_surface_resolve",
    "terrain_surface_debris",
    "terrain_surface_publish",
];

/// Header words; the heights follow, one f32 per window sample.
pub const SURFACE_HEADER_WORDS: u32 = 24;
/// Header word set to 1 by every resolve; a host clears it when the region
/// bakes (the heightfield is then the ground again).
pub const SURFACE_VALID_WORD: u32 = 15;
/// Header words the GPU fills: window i0, j0, ni, nj, then the bed's local y
/// and the layer height (f32 bits).
pub const SURFACE_WINDOW_WORD: u32 = 8;
/// The debris draw's indirect arguments (`drawIndirect`: vertex count, which
/// a renderer writes; instance count, the particles listed this frame; 0; 0).
pub const SURFACE_DEBRIS_ARGS_WORD: u32 = 16;
/// Header word: the surface patch's collider (`u32::MAX`: none).
pub const SURFACE_PATCH_WORD: u32 = 6;

/// A static heightfield collider (`nx × nz` samples, the field's spacing)
/// that `terrain_surface_publish` places over the open window with the live
/// surface's heights, so wheel rays and contacts meet the sand; sunk out of
/// reach while no region is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfacePatch {
    pub collider: u32,
    pub nx: u32,
    pub nz: u32,
}

/// Default limit on window samples × layers.
pub const DEFAULT_SURFACE_CELLS: u32 = 1 << 21;

/// The surface buffer's sections and limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePlan {
    max_columns: u32,
    max_particles: u32,
    max_cells: u32,
    ejecta_fall: f32,
    patch: Option<SurfacePatch>,
}

impl SurfacePlan {
    /// For regions within `config`'s `max_columns`, with at most `max_cells`
    /// occupancy cells (window samples × the region's cells in y). Sand
    /// moving faster than a fall of one layer is in flight (debris, not
    /// ground); [`Self::with_ejecta_fall`] changes that height.
    pub fn new(config: &TerrainConfig, max_cells: u32) -> Result<Self, TerrainError> {
        config.validate()?;
        if max_cells == 0 {
            return Err(TerrainError::InvalidRequest(
                "surface max_cells must be positive",
            ));
        }
        Ok(Self {
            max_columns: config.max_columns,
            max_particles: config.max_particles,
            max_cells,
            ejecta_fall: 1.0,
            patch: None,
        })
    }

    /// Publish the surface into `patch` every frame (and sink it at the bake).
    pub fn with_patch(mut self, patch: Option<SurfacePatch>) -> Self {
        self.patch = patch;
        self
    }

    pub fn patch(&self) -> Option<SurfacePatch> {
        self.patch
    }

    fn patch_word(&self) -> u32 {
        self.patch.map_or(u32::MAX, |p| p.collider)
    }

    fn publish(&self, ops: &mut TerrainOps) {
        if let Some(patch) = self.patch {
            ops.cmds.push(TerrainCmd::Dispatch {
                kernel: "terrain_surface_publish",
                threads: u64::from(patch.nx * patch.nz).div_ceil(4),
            });
        }
    }

    /// After a bake: the surface is no longer valid, and the patch sinks out
    /// of reach (no blocks without a patch).
    pub fn retire(&self) -> Option<TerrainOps> {
        self.patch?;
        let mut ops = TerrainOps {
            writes: vec![
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 4 * u64::from(SURFACE_PATCH_WORD),
                    words: vec![self.patch_word()],
                },
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 4 * u64::from(SURFACE_VALID_WORD),
                    words: vec![0],
                },
            ],
            cmds: Vec::new(),
        };
        self.publish(&mut ops);
        Some(ops)
    }

    /// Sand faster than a fall of `layers` layers (√(2 g · layers · dy)) is in
    /// flight; 0 counts only the material a gap separates.
    pub fn with_ejecta_fall(mut self, layers: f32) -> Result<Self, TerrainError> {
        if !(layers.is_finite() && layers >= 0.0) {
            return Err(TerrainError::InvalidRequest(
                "surface ejecta_fall must be finite and non-negative",
            ));
        }
        self.ejecta_fall = layers;
        Ok(self)
    }

    /// First word of the walls' give (two per window edge sample).
    pub fn shift_base(&self) -> u32 {
        SURFACE_HEADER_WORDS + self.max_columns
    }

    /// First word of the occupancy (samples × layers, fixed point).
    pub fn occupancy_base(&self) -> u32 {
        self.shift_base() + 2 * self.max_columns + 2
    }

    /// First word of the debris list (particle indices, one per particle at most).
    pub fn debris_base(&self) -> u32 {
        self.occupancy_base() + self.max_cells
    }

    /// Bytes of the surface buffer.
    pub fn bytes(&self) -> u64 {
        4 * (u64::from(self.debris_base()) + u64::from(self.max_particles))
    }

    pub fn max_cells(&self) -> u32 {
        self.max_cells
    }

    /// The pass that rebuilds the surface of `terrain`'s open region from its
    /// particles as they are now (after the frame's substeps), under `gravity`
    /// (m/s², its magnitude). Fails naming the limit when the region needs
    /// more cells than `max_cells`, and when the region was not opened over a
    /// heightfield (no window).
    pub fn frame(&self, terrain: &TerrainPlan, gravity: f32) -> anyhow::Result<TerrainOps> {
        let p = terrain.params();
        let [_, _, ni, nj] = p.window;
        anyhow::ensure!(
            p.dims[3] != MPM_NO_BED && ni > 0 && nj > 0,
            "the terrain surface needs a region opened over a heightfield"
        );
        let layers = p.dims[1];
        let samples = u64::from(ni) * u64::from(nj);
        TerrainConfig::check(
            "surface max_cells",
            samples * u64::from(layers),
            u64::from(self.max_cells),
        )?;
        if let Some(patch) = self.patch {
            // The window and a ring of the field's samples round it.
            TerrainConfig::check(
                "surface patch samples",
                u64::from(ni.max(nj) + 2),
                u64::from(patch.nx.min(patch.nz)),
            )?;
        }
        let absorbing = p.boundary[0] & MPM_BOUNDARY_ABSORBING != 0;
        let ejecta_speed = (2.0 * gravity.abs() * self.ejecta_fall * p.origin[3]).sqrt();
        let mut ops = TerrainOps {
            writes: vec![TerrainWrite {
                binding: SURFACE_BINDING,
                offset: 0,
                words: vec![
                    self.shift_base(),
                    self.occupancy_base(),
                    layers,
                    u32::from(absorbing),
                    self.debris_base(),
                    ejecta_speed.to_bits(),
                    self.patch_word(),
                ],
            }],
            cmds: Vec::new(),
        };
        let disp = |ops: &mut TerrainOps, kernel, threads| {
            ops.cmds.push(TerrainCmd::Dispatch { kernel, threads })
        };
        disp(
            &mut ops,
            "terrain_surface_clear",
            samples * u64::from(layers),
        );
        if absorbing {
            disp(&mut ops, "terrain_surface_shift", 2 * u64::from(ni + nj));
        }
        disp(&mut ops, "mpm_write_args", 1);
        ops.cmds.push(TerrainCmd::Indirect {
            kernel: "terrain_surface_bin",
            offset: ARGS_PARTICLES,
        });
        disp(&mut ops, "terrain_surface_resolve", samples);
        ops.cmds.push(TerrainCmd::Indirect {
            kernel: "terrain_surface_debris",
            offset: ARGS_PARTICLES,
        });
        self.publish(&mut ops);
        Ok(ops)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::FieldGrid;
    use crate::{InitialStress, RegionBoundary, RegionPlan};

    fn opened(cfg: &TerrainConfig) -> TerrainPlan {
        let mut p = TerrainPlan::new(cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let region = RegionPlan::spanning(
            cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        let grid = FieldGrid {
            nx: 33,
            nz: 33,
            dx: 0.125,
            dz: 0.125,
        };
        p.activate(&region, 0, grid, glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        p
    }

    fn kernels(ops: &TerrainOps) -> Vec<&'static str> {
        ops.cmds
            .iter()
            .map(|c| match *c {
                TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } => kernel,
            })
            .collect()
    }

    #[test]
    fn the_pass_bins_every_particle_then_resolves_every_window_sample() {
        let cfg = TerrainConfig {
            boundary: RegionBoundary::Absorbing,
            initial_stress: InitialStress::Geostatic,
            ..TerrainConfig::default()
        };
        let t = opened(&cfg);
        let s = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS).unwrap();
        let ops = s.frame(&t, 9.81).unwrap();
        assert_eq!(
            kernels(&ops),
            [
                "terrain_surface_clear",
                "terrain_surface_shift",
                "mpm_write_args",
                "terrain_surface_bin",
                "terrain_surface_resolve",
                "terrain_surface_debris"
            ]
        );
        for k in kernels(&ops) {
            assert!(SURFACE_KERNELS.contains(&k), "{k}");
        }
        let [_, _, ni, nj] = t.params().window;
        let layers = t.params().dims[1];
        assert!(
            matches!(ops.cmds[0], TerrainCmd::Dispatch { threads, .. } if threads == u64::from(ni * nj * layers))
        );
        assert!(
            matches!(ops.cmds[4], TerrainCmd::Dispatch { threads, .. } if threads == u64::from(ni * nj))
        );
        // The header names the sections, the layers and the shift.
        assert_eq!(ops.writes.len(), 1);
        assert_eq!(ops.writes[0].binding, SURFACE_BINDING);
        assert_eq!(
            ops.writes[0].words,
            [
                s.shift_base(),
                s.occupancy_base(),
                layers,
                1,
                s.debris_base(),
                (2.0 * 9.81 * t.params().origin[3]).sqrt().to_bits(),
                u32::MAX
            ]
        );
        // Off: only the material a gap separates is debris.
        let off = s.with_ejecta_fall(0.0).unwrap().frame(&t, 9.81).unwrap();
        assert_eq!(off.writes[0].words[5], 0f32.to_bits());
        assert!(s.bytes() >= 4 * u64::from(s.occupancy_base() + ni * nj * layers));
    }

    #[test]
    fn a_region_past_max_cells_fails_naming_the_limit() {
        let cfg = TerrainConfig {
            boundary: RegionBoundary::Rigid,
            ..TerrainConfig::default()
        };
        let t = opened(&cfg);
        let s = SurfacePlan::new(&cfg, 100).unwrap();
        let e = s.frame(&t, 9.81).unwrap_err().to_string();
        assert!(e.contains("surface max_cells"), "{e}");
        // The rigid ground has no walls' give to measure.
        let ops = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS)
            .unwrap()
            .frame(&t, 9.81)
            .unwrap();
        assert!(!kernels(&ops).contains(&"terrain_surface_shift"));
        assert_eq!(ops.writes[0].words[3], 0);
        assert!(SurfacePlan::new(&cfg, 0).is_err());
    }

    #[test]
    fn a_loaded_region_has_no_surface() {
        let cfg = TerrainConfig::default();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        let region = RegionPlan::spanning(
            &cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        p.load(&region, 10, None, 0.5).unwrap();
        let s = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS).unwrap();
        assert!(s.frame(&p, 9.81).is_err());
    }

    #[test]
    fn a_patch_takes_the_surface_while_open_and_sinks_at_the_bake() {
        let cfg = TerrainConfig::default();
        let t = opened(&cfg);
        let [_, _, ni, nj] = t.params().window;
        let patch = SurfacePatch {
            collider: 7,
            nx: ni + 2,
            nz: nj + 2,
        };
        let s = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS)
            .unwrap()
            .with_patch(Some(patch));
        let ops = s.frame(&t, 9.81).unwrap();
        assert_eq!(kernels(&ops).last(), Some(&"terrain_surface_publish"));
        assert_eq!(ops.writes[0].words[6], 7);
        let retire = s.retire().unwrap();
        assert_eq!(kernels(&retire), ["terrain_surface_publish"]);
        assert!(SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS)
            .unwrap()
            .retire()
            .is_none());
        // A patch smaller than the window and its ring fails by name.
        let small = s.with_patch(Some(SurfacePatch { nx: ni, ..patch }));
        let e = small.frame(&t, 9.81).unwrap_err().to_string();
        assert!(e.contains("surface patch samples"), "{e}");
    }
}
