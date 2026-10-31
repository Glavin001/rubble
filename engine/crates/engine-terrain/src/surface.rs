//! Render-only: the live ground surface over every open area, rebuilt every
//! frame on the GPU from its particles (`slang/kernels/terrain_surface.slang`;
//! `engine/docs/deformable-terrain.md`, "Rendering and showcase").
//!
//! A [`SurfacePlan`] sizes the surface buffer ([`SURFACE_BINDING`]) and turns
//! the open areas of a [`TerrainPlan`] into one compute pass. Each open area
//! is a *view*: its window's heights, the walls' give and its occupancy, each
//! in a section of the buffer, found through the view table. Every kernel
//! covers all the views at once: the occupancy cleared, the absorbing walls'
//! give measured (as the bake measures it), every particle splatted into its
//! own area's view as the bake splats it, and one height per window sample
//! resolved from the material attached to the bed. A renderer draws those
//! heights as the ground over each open window and the material above them
//! (and the parcels) as debris. The simulation never reads the buffer; the
//! surface patches (below) are the one output physics reads.

use crate::config::{TerrainConfig, TerrainError};
use crate::layout::{MPM_BOUNDARY_ABSORBING, MPM_MAX_AREAS, MPM_NO_BED};
use crate::plan::{TerrainCmd, TerrainOps, TerrainPlan, TerrainWrite, ARGS_PARTICLES};

/// The surface buffer's binding (after the terrain look-ahead's 52).
pub const SURFACE_BINDING: usize = 53;

/// Every kernel a [`SurfacePlan`] dispatches (`mpm_write_args` refreshes the
/// per-particle indirect counts from the counters, which the frame left).
pub const SURFACE_KERNELS: &[&str] = &[
    "mpm_write_args",
    "terrain_surface_clear",
    "terrain_surface_shift",
    "terrain_surface_shift_mean",
    "terrain_surface_bin",
    "terrain_surface_resolve",
    "terrain_surface_debris",
    "terrain_surface_publish",
];

/// Header words: [0] views to scan (one past the highest open slot), [4] the
/// debris list's first word, [5] the ejecta speed (f32 bits, m/s), [14]
/// frames resolved, [15] valid, [16..20) the debris draw's arguments. The view
/// table follows ([`SURFACE_VIEW_WORD`]).
pub const SURFACE_HEADER_WORDS: u32 = 24;
/// Header word set to 1 by every resolve; a host clears it when the last area
/// bakes (the heightfield is then the ground again).
pub const SURFACE_VALID_WORD: u32 = 15;
/// The debris draw's indirect arguments (`drawIndirect`: vertex count, which
/// a renderer writes; instance count, the particles listed this frame; 0; 0).
pub const SURFACE_DEBRIS_ARGS_WORD: u32 = 16;
/// First word of the view table: one record of [`SURFACE_VIEW_WORDS`] per
/// area slot (all [`MPM_MAX_AREAS`]; a slot not open is not shown).
pub const SURFACE_VIEW_WORD: u32 = SURFACE_HEADER_WORDS;
/// Words of a view record: [0] flags ([`SURFACE_VIEW_OPEN`],
/// [`SURFACE_VIEW_ABSORBING`]), [1] layers, [2] its heights' first word, [3]
/// its shift section's, [4] its occupancy's, [5] its patch collider
/// (`u32::MAX`: none), [6..10) the window i0, j0, ni, nj
/// ([`SURFACE_VIEW_WINDOW`]), [10] the bed's local y (f32 bits), [11] the
/// layer height (f32 bits), [12..16) the view's first thread in the clear,
/// shift-mean, resolve and publish dispatches.
pub const SURFACE_VIEW_WORDS: u32 = 16;
/// View flag: the area is open and its window shown.
pub const SURFACE_VIEW_OPEN: u32 = 1;
/// View flag: the absorbing ground's edge shift is used.
pub const SURFACE_VIEW_ABSORBING: u32 = 2;
/// Word of a view record where its window (i0, j0, ni, nj) starts.
pub const SURFACE_VIEW_WINDOW: u32 = 6;
/// First word after the view table (the sections start here).
pub const SURFACE_DATA_WORD: u32 = SURFACE_VIEW_WORD + SURFACE_VIEW_WORDS * MPM_MAX_AREAS;

const VIEW: usize = SURFACE_VIEW_WORDS as usize;
const SLOTS: usize = MPM_MAX_AREAS as usize;

/// A static heightfield collider (`nx × nz` samples, the field's spacing)
/// that `terrain_surface_publish` places over its area's open window with the
/// live surface's heights, so wheel rays and contacts meet the sand; sunk out
/// of reach while that area is not open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfacePatch {
    pub collider: u32,
    pub nx: u32,
    pub nz: u32,
}

/// Default limit on window samples × layers, summed over the open areas.
pub const DEFAULT_SURFACE_CELLS: u32 = 1 << 21;

/// The surface buffer's sections and limits.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfacePlan {
    max_columns: u32,
    max_particles: u32,
    max_cells: u32,
    ejecta_fall: f32,
    /// Patch `k` serves area slot `k`.
    patches: Vec<SurfacePatch>,
}

impl SurfacePlan {
    /// For areas within `config`'s `max_columns` (summed over them, as the
    /// plan pools them), with at most `max_cells` occupancy cells (window
    /// samples × the area's cells in y, summed over the open areas). Sand
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
            patches: Vec::new(),
        })
    }

    /// One patch, for area slot 0 ([`Self::with_patches`] for several).
    pub fn with_patch(self, patch: Option<SurfacePatch>) -> Self {
        self.with_patches(patch.into_iter().collect())
            .expect("one patch is within MPM_MAX_AREAS")
    }

    /// One patch per area slot: `patches[k]` takes the surface of slot `k`
    /// while it is open and sinks while it is not. Give one per
    /// `max_regions` so every area that can open is drivable: a frame with
    /// an area open past the last patch fails naming "surface patches".
    /// More than [`MPM_MAX_AREAS`] is an error.
    pub fn with_patches(mut self, patches: Vec<SurfacePatch>) -> Result<Self, TerrainError> {
        TerrainConfig::check(
            "surface patches",
            patches.len() as u64,
            u64::from(MPM_MAX_AREAS),
        )?;
        self.patches = patches;
        Ok(self)
    }

    pub fn patches(&self) -> &[SurfacePatch] {
        &self.patches
    }

    /// Patch records `terrain_surface_publish` writes for `patch` (four samples each).
    fn records(patch: &SurfacePatch) -> u64 {
        (u64::from(patch.nx) * u64::from(patch.nz)).div_ceil(4)
    }

    fn publish(&self, ops: &mut TerrainOps) {
        let threads: u64 = self.patches.iter().map(Self::records).sum();
        if threads > 0 {
            ops.cmds.push(TerrainCmd::Dispatch {
                kernel: "terrain_surface_publish",
                threads,
            });
        }
    }

    /// The view table: `views` with each slot's patch and publish range.
    fn table(&self, mut views: [[u32; VIEW]; SLOTS]) -> Vec<u32> {
        let mut publish = 0u64;
        for (k, v) in views.iter_mut().enumerate() {
            v[5] = self.patches.get(k).map_or(u32::MAX, |p| p.collider);
            v[15] = publish as u32;
            if let Some(p) = self.patches.get(k) {
                publish += Self::records(p);
            }
        }
        views.concat()
    }

    /// After the last area bakes: the surface is no longer valid, and every
    /// patch sinks out of reach (`None` without patches; the host clears the
    /// valid word itself then).
    pub fn retire(&self) -> Option<TerrainOps> {
        let mut ops = TerrainOps {
            writes: vec![
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 4 * u64::from(SURFACE_VALID_WORD),
                    words: vec![0],
                },
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 4 * u64::from(SURFACE_VIEW_WORD),
                    words: self.table([[0; VIEW]; SLOTS]),
                },
            ],
            cmds: Vec::new(),
        };
        self.publish(&mut ops);
        (!ops.cmds.is_empty()).then_some(ops)
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

    /// First word of the heights (every open window's, one per sample).
    pub fn heights_base(&self) -> u32 {
        SURFACE_DATA_WORD
    }

    /// First word of the walls' give (per view: two per window edge sample,
    /// then their accumulators).
    pub fn shift_base(&self) -> u32 {
        self.heights_base() + self.max_columns
    }

    /// Words of the shift section (the limit "surface edge samples" names).
    fn shift_words(&self) -> u32 {
        2 * self.max_columns + 2
    }

    /// First word of the occupancy (per view: samples × layers, fixed point).
    pub fn occupancy_base(&self) -> u32 {
        self.shift_base() + self.shift_words()
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

    /// The pass that rebuilds the surface of every open area of `terrain`
    /// from its particles as they are now (after the frame's substeps), under
    /// `gravity` (m/s², its magnitude), publishes each into its slot's patch
    /// and sinks the patches of slots not open. Fails naming the limit when
    /// the open areas together need more cells than `max_cells` or more edge
    /// samples than the shift section holds, when an open area has no patch
    /// (with patches given) or one too small, and when an area was not
    /// opened over a heightfield (no window) or none is open.
    pub fn frame(&self, terrain: &TerrainPlan, gravity: f32) -> anyhow::Result<TerrainOps> {
        let areas = terrain.areas();
        let no_window = "the terrain surface needs a region opened over a heightfield";
        anyhow::ensure!(!areas.is_empty(), no_window);
        let mut views = [[0u32; VIEW]; SLOTS];
        // Running totals: heights, shift words, occupancy cells, and the
        // threads of the clear, shift-mean and resolve dispatches.
        let (mut heights, mut shift, mut cells) = (0u64, 0u64, 0u64);
        let (mut clear, mut means, mut samples_all) = (0u64, 0u64, 0u64);
        let mut absorbing_any = false;
        let mut dy = 0.0f32;
        for a in &areas {
            let p = terrain
                .area_params(a.slot)
                .ok_or_else(|| anyhow::anyhow!(no_window))?;
            let [i0, j0, ni, nj] = p.window;
            anyhow::ensure!(p.dims[3] != MPM_NO_BED && ni > 0 && nj > 0, no_window);
            if !self.patches.is_empty() {
                TerrainConfig::check(
                    "surface patches",
                    u64::from(a.slot) + 1,
                    self.patches.len() as u64,
                )?;
                let patch = self.patches[a.slot as usize];
                // The window and a ring of the field's samples round it.
                TerrainConfig::check(
                    "surface patch samples",
                    u64::from(ni.max(nj) + 2),
                    u64::from(patch.nx.min(patch.nz)),
                )?;
            }
            let layers = p.dims[1];
            let samples = u64::from(ni) * u64::from(nj);
            let edge = 2 * u64::from(ni + nj);
            let absorbing = p.boundary[0] & MPM_BOUNDARY_ABSORBING != 0;
            absorbing_any |= absorbing;
            let v = &mut views[a.slot as usize];
            v[0] = SURFACE_VIEW_OPEN | if absorbing { SURFACE_VIEW_ABSORBING } else { 0 };
            v[1] = layers;
            v[2] = self.heights_base() + heights as u32;
            v[3] = self.shift_base() + shift as u32;
            v[4] = self.occupancy_base() + cells as u32;
            v[6..10].copy_from_slice(&[i0, j0, ni, nj]);
            v[10] = p.column[0].to_bits();
            v[11] = p.origin[3].to_bits();
            v[12] = clear as u32;
            v[13] = means as u32;
            v[14] = samples_all as u32;
            heights += samples;
            // The give (one word per edge sample) and its accumulators (two).
            shift += 3 * edge;
            cells += samples * u64::from(layers);
            // The clear empties the occupancy and the give's accumulators.
            clear += (samples * u64::from(layers)).max(2 * edge);
            means += edge;
            samples_all += samples;
            dy = p.origin[3];
        }
        TerrainConfig::check("surface max_cells", cells, u64::from(self.max_cells))?;
        TerrainConfig::check("surface edge samples", shift, u64::from(self.shift_words()))?;
        // The plan pools the areas' columns within max_columns.
        TerrainConfig::check("max_columns", heights, u64::from(self.max_columns))?;
        let scan = areas.iter().map(|a| a.slot + 1).max().unwrap_or(0);
        // One cell size over the domain: one ejecta speed for every view.
        let ejecta_speed = (2.0 * gravity.abs() * self.ejecta_fall * dy).sqrt();
        let mut ops = TerrainOps {
            writes: vec![
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 0,
                    words: vec![scan, 0, 0, 0, self.debris_base(), ejecta_speed.to_bits()],
                },
                TerrainWrite {
                    binding: SURFACE_BINDING,
                    offset: 4 * u64::from(SURFACE_VIEW_WORD),
                    words: self.table(views),
                },
            ],
            cmds: Vec::new(),
        };
        let disp = |ops: &mut TerrainOps, kernel, threads| {
            ops.cmds.push(TerrainCmd::Dispatch { kernel, threads })
        };
        disp(&mut ops, "terrain_surface_clear", clear);
        disp(&mut ops, "mpm_write_args", 1);
        if absorbing_any {
            // The walls' give, from the particles made at the edge columns.
            ops.cmds.push(TerrainCmd::Indirect {
                kernel: "terrain_surface_shift",
                offset: ARGS_PARTICLES,
            });
            disp(&mut ops, "terrain_surface_shift_mean", means);
        }
        ops.cmds.push(TerrainCmd::Indirect {
            kernel: "terrain_surface_bin",
            offset: ARGS_PARTICLES,
        });
        disp(&mut ops, "terrain_surface_resolve", samples_all);
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

    fn grid() -> FieldGrid {
        FieldGrid {
            nx: 97,
            nz: 33,
            dx: 0.125,
            dz: 0.125,
        }
    }

    /// Areas over [x - 1, x + 1] × [-1, 1] for each x.
    fn opened_at(cfg: &TerrainConfig, xs: &[f32]) -> TerrainPlan {
        let mut p = TerrainPlan::new(cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        for &x in xs {
            let c = glam::Vec3::new(x, 0.0, 0.0);
            let region = RegionPlan::spanning(
                cfg,
                c - glam::Vec3::splat(1.0),
                c + glam::Vec3::splat(1.0),
                0.1,
                0,
                0,
            )
            .unwrap();
            let o = p
                .open(&region, 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
                .unwrap();
            assert!(o.absorbed.is_empty() && o.activated);
        }
        p
    }

    fn opened(cfg: &TerrainConfig) -> TerrainPlan {
        opened_at(cfg, &[0.0])
    }

    fn kernels(ops: &TerrainOps) -> Vec<&'static str> {
        ops.cmds
            .iter()
            .map(|c| match *c {
                TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } => kernel,
            })
            .collect()
    }

    fn threads(ops: &TerrainOps, kernel: &str) -> u64 {
        ops.cmds
            .iter()
            .find_map(|c| match *c {
                TerrainCmd::Dispatch { kernel: k, threads } if k == kernel => Some(threads),
                _ => None,
            })
            .unwrap()
    }

    fn view(ops: &TerrainOps, slot: u32) -> &[u32] {
        &ops.writes[1].words[slot as usize * VIEW..][..VIEW]
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
                "mpm_write_args",
                "terrain_surface_shift",
                "terrain_surface_shift_mean",
                "terrain_surface_bin",
                "terrain_surface_resolve",
                "terrain_surface_debris"
            ]
        );
        for k in kernels(&ops) {
            assert!(SURFACE_KERNELS.contains(&k), "{k}");
        }
        let p = t.area_params(0).unwrap();
        let [i0, j0, ni, nj] = p.window;
        let layers = p.dims[1];
        assert_eq!(
            threads(&ops, "terrain_surface_clear"),
            u64::from(ni * nj * layers)
        );
        assert_eq!(threads(&ops, "terrain_surface_resolve"), u64::from(ni * nj));
        // The header names the views, the debris list and the ejecta speed.
        assert_eq!(ops.writes[0].binding, SURFACE_BINDING);
        assert_eq!(
            ops.writes[0].words,
            [
                1,
                0,
                0,
                0,
                s.debris_base(),
                (2.0 * 9.81 * p.origin[3]).sqrt().to_bits()
            ]
        );
        // The view: open on the absorbing ground, its sections and window.
        assert_eq!(ops.writes[1].offset, 4 * u64::from(SURFACE_VIEW_WORD));
        assert_eq!(ops.writes[1].words.len(), VIEW * SLOTS);
        assert_eq!(
            view(&ops, 0),
            [
                SURFACE_VIEW_OPEN | SURFACE_VIEW_ABSORBING,
                layers,
                s.heights_base(),
                s.shift_base(),
                s.occupancy_base(),
                u32::MAX,
                i0,
                j0,
                ni,
                nj,
                p.column[0].to_bits(),
                p.origin[3].to_bits(),
                0,
                0,
                0,
                0
            ]
        );
        assert_eq!(view(&ops, 1)[0], 0);
        // Off: only the material a gap separates is debris.
        let off = s
            .clone()
            .with_ejecta_fall(0.0)
            .unwrap()
            .frame(&t, 9.81)
            .unwrap();
        assert_eq!(off.writes[0].words[5], 0f32.to_bits());
        assert!(s.bytes() >= 4 * u64::from(s.occupancy_base() + ni * nj * layers));
    }

    #[test]
    fn separate_areas_are_views_of_their_own_in_one_pass() {
        let cfg = TerrainConfig {
            boundary: RegionBoundary::Absorbing,
            ..TerrainConfig::default()
        };
        let t = opened_at(&cfg, &[-4.0, 4.0]);
        assert_eq!(t.areas().len(), 2);
        let s = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS).unwrap();
        let ops = s.frame(&t, 9.81).unwrap();
        assert_eq!(ops.writes[0].words[0], 2, "two views to scan");
        let (a, b) = (t.area_params(0).unwrap(), t.area_params(1).unwrap());
        let samples = |p: &crate::layout::MpmParams| u64::from(p.window[2] * p.window[3]);
        let edge = |p: &crate::layout::MpmParams| 2 * u64::from(p.window[2] + p.window[3]);
        let cells = |p: &crate::layout::MpmParams| samples(p) * u64::from(p.dims[1]);
        // Back to back in each section, and in each dispatch's threads.
        let (va, vb) = (view(&ops, 0), view(&ops, 1));
        assert_eq!(va[0] & SURFACE_VIEW_OPEN, 1);
        assert_eq!(vb[0] & SURFACE_VIEW_OPEN, 1);
        assert_eq!(vb[2], va[2] + samples(a) as u32);
        assert_eq!(vb[3], va[3] + 3 * edge(a) as u32);
        assert_eq!(vb[4], va[4] + cells(a) as u32);
        assert_eq!(vb[12], cells(a).max(2 * edge(a)) as u32);
        assert_eq!(vb[13], edge(a) as u32);
        assert_eq!(vb[14], samples(a) as u32);
        assert_eq!(&vb[6..10], &b.window);
        assert_eq!(
            threads(&ops, "terrain_surface_resolve"),
            samples(a) + samples(b)
        );
        assert_eq!(
            threads(&ops, "terrain_surface_clear"),
            cells(a).max(2 * edge(a)) + cells(b).max(2 * edge(b))
        );
        // The windows do not overlap (the areas keep apart).
        assert!(a.window[0] + a.window[2] < b.window[0] || b.window[0] + b.window[2] < a.window[0]);
        // max_cells is over every open area.
        let one = (cells(a).max(cells(b))) as u32;
        let e = SurfacePlan::new(&cfg, one)
            .unwrap()
            .frame(&t, 9.81)
            .unwrap_err()
            .to_string();
        assert!(e.contains("surface max_cells"), "{e}");
    }

    #[test]
    fn each_area_publishes_into_its_slots_patch_and_a_missing_one_fails_by_name() {
        let cfg = TerrainConfig::default();
        let t = opened_at(&cfg, &[-4.0, 4.0]);
        let [_, _, ni, nj] = t.area_params(1).unwrap().window;
        let patch = |collider| SurfacePatch {
            collider,
            nx: ni + 2,
            nz: nj + 2,
        };
        let s = SurfacePlan::new(&cfg, DEFAULT_SURFACE_CELLS).unwrap();
        let two = s
            .clone()
            .with_patches(vec![patch(7), patch(9), patch(11)])
            .unwrap();
        let ops = two.frame(&t, 9.81).unwrap();
        assert_eq!(kernels(&ops).last(), Some(&"terrain_surface_publish"));
        let records = SurfacePlan::records(&patch(0));
        // Every patch, open or not (slot 2's sinks).
        assert_eq!(threads(&ops, "terrain_surface_publish"), 3 * records);
        assert_eq!(view(&ops, 0)[5], 7);
        assert_eq!(view(&ops, 1)[5], 9);
        assert_eq!(view(&ops, 2)[5], 11);
        assert_eq!(view(&ops, 2)[0], 0);
        assert_eq!(view(&ops, 3)[5], u32::MAX);
        assert_eq!(
            [view(&ops, 0)[15], view(&ops, 1)[15], view(&ops, 2)[15]],
            [0, records as u32, 2 * records as u32]
        );
        // One patch for two open areas: the second has none.
        let e = s
            .clone()
            .with_patch(Some(patch(7)))
            .frame(&t, 9.81)
            .unwrap_err()
            .to_string();
        assert!(e.contains("surface patches"), "{e}");
        // Retire: every view closed, every patch sunk.
        let retire = two.retire().unwrap();
        assert_eq!(kernels(&retire), ["terrain_surface_publish"]);
        assert!(retire.writes[1].words.chunks(VIEW).all(|v| v[0] == 0));
        assert!(s.retire().is_none());
        // A patch smaller than the window and its ring fails by name.
        let small = s
            .clone()
            .with_patches(vec![patch(7), SurfacePatch { nx: ni, ..patch(9) }])
            .unwrap();
        let e = small.frame(&t, 9.81).unwrap_err().to_string();
        assert!(e.contains("surface patch samples"), "{e}");
        // More patches than area slots.
        assert!(s.with_patches(vec![patch(1); 17]).is_err());
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
        assert_eq!(view(&ops, 0)[0], SURFACE_VIEW_OPEN);
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
        assert!(s.frame(&TerrainPlan::new(&cfg).unwrap(), 9.81).is_err());
    }
}
