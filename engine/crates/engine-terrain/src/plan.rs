//! One terrain command sequence for every host: the native wgpu host
//! (`engine-webgpu`'s `WgpuSim`) and the browser (through wasm) replay the
//! same [`TerrainOps`] for activation, the Z-model source, uncoupled and
//! coupled frames of substeps, the parcels' deposits and the bake, so the two
//! cannot drift apart (as `engine_exec::rigid_plan` is for the rigid step).
//!
//! A [`TerrainPlan`] holds the host state of the terrain pool (limits, the
//! `MpmParams` record, the material's wave speed) and turns each call into
//! host writes plus the dispatches of one compute pass. Kernel names are the
//! entry points of `slang/kernels/terrain_mpm.slang` (bindings 37-51); every
//! dispatch is `threads` invocations at 64 per workgroup, and an indirect one
//! reads its workgroup counts from [`MPM_ARGS_BINDING`] at a byte offset.
//!
//! [`RegionClock`] is the open region's frame schedule: each frame's substeps
//! from a speed bound the GPU checks, and when the region has settled.

use crate::config::{InitialStress, RegionBoundary, TerrainConfig, TerrainError};
use crate::layout::*;
use crate::material::{RockMaterial, TerrainMaterial, YieldSurface};
use crate::region::{RegionPlan, BLOCK_CELLS};
use crate::step::substeps_for;
use crate::zmodel::ZModel;
use glam::IVec3;

/// Every kernel a [`TerrainPlan`] dispatches.
pub const KERNELS: &[&str] = &[
    "mpm_clear",
    "mpm_touch",
    "mpm_write_args",
    "mpm_count",
    "mpm_scan_cells",
    "mpm_scatter",
    "mpm_cell_sort",
    "mpm_payload",
    "mpm_p2g",
    "mpm_grid",
    "mpm_g2p",
    "mpm_activate_count",
    "mpm_activate_scan",
    "mpm_activate_emit",
    "mpm_activate_lower",
    "mpm_bake_bin",
    "mpm_bake_raise",
    "mpm_bake_header",
    "mpm_couple_clear",
    "mpm_couple_bin",
    "mpm_couple_gather",
    "mpm_couple_apply",
    "mpm_couple_emit",
    "mpm_couple_finish",
    "mpm_parcels",
    "mpm_deposit_raise",
    "mpm_deposit_header",
    "mpm_source",
    "mpm_absorb_init",
    "mpm_geostatic",
    "mpm_absorb_faces",
    "mpm_bake_shift",
    "mpm_bake_edge_clear",
    "mpm_bake_edge_count",
    "mpm_bake_edge_scan",
    "mpm_bake_edge_scatter",
    "mpm_bake_edge_sort",
    "mpm_source_surface",
];

/// Byte offsets of the indirect workgroup counts in [`MPM_ARGS_BINDING`].
pub const ARGS_BLOCKS: u64 = 0;
pub const ARGS_PARTICLES: u64 = 12;
pub const ARGS_PARCELS: u64 = 24;

/// Quanta per lattice particle: a column's volume is rounded to s³/64 (s the
/// particle spacing), so a sample's height moves by whole quanta.
const QUANTA_PER_LATTICE_CELL: f32 = 64.0;

/// What a run of substeps reported (the counters after it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStepReport {
    /// Blocks the last substep used.
    pub blocks: u32,
    /// Largest particle speed over the frame's substeps (m/s).
    pub max_speed: f32,
    /// Particle slots in the region (live, or left as parcels).
    pub particles: u32,
    /// Parcels thrown so far (in flight or landed).
    pub parcels: u32,
    /// Quanta parcels have deposited on the heightfield.
    pub deposited: u32,
    /// Quanta that left the heightfield's footprint.
    pub escaped: u32,
    /// Parcels still in flight after the last substep.
    pub airborne: u32,
    /// Fastest surface speed of a dynamic body coupled to the region this
    /// frame (m/s; 0 without coupling).
    pub body_speed: f32,
    /// Per area slot: its fastest particle over the frame (m/s).
    pub area_speed: [f32; MPM_MAX_AREAS as usize],
    /// Per area slot: its fastest coupled body this frame (m/s).
    pub area_body_speed: [f32; MPM_MAX_AREAS as usize],
}

impl TerrainStepReport {
    /// The report as area `slot` sees it: its own particles' and bodies'
    /// speeds (the domain's parcels and ledger are shared).
    pub fn area(&self, slot: u32) -> Self {
        let k = slot as usize;
        Self {
            max_speed: self.area_speed.get(k).copied().unwrap_or(0.0),
            body_speed: self.area_body_speed.get(k).copied().unwrap_or(0.0),
            ..*self
        }
    }
}

/// A host write (`queue.write_buffer`) of `words` at byte `offset` of `binding`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainWrite {
    pub binding: usize,
    pub offset: u64,
    pub words: Vec<u32>,
}

/// One dispatch of the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainCmd {
    /// `threads` invocations, 64 per workgroup.
    Dispatch { kernel: &'static str, threads: u64 },
    /// Workgroup counts from [`MPM_ARGS_BINDING`] at byte `offset`.
    Indirect { kernel: &'static str, offset: u64 },
}

/// A call's work: the host writes first (ahead of the submission that carries
/// the commands), then the commands in order, in one compute pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrainOps {
    pub writes: Vec<TerrainWrite>,
    pub cmds: Vec<TerrainCmd>,
}

impl TerrainOps {
    fn write_words(&mut self, binding: usize, offset: u64, words: &[u32]) {
        self.writes.push(TerrainWrite {
            binding,
            offset,
            words: words.to_vec(),
        });
    }

    fn counter(&mut self, word: usize, value: u32) {
        self.write_words(MPM_COUNTERS_BINDING, 4 * word as u64, &[value]);
    }

    fn params(&mut self, params: &MpmParams) {
        self.write_words(
            MPM_PARAMS_BINDING,
            0,
            bytemuck::cast_slice(std::slice::from_ref(params)),
        );
    }

    fn disp(&mut self, kernel: &'static str, threads: u64) {
        self.cmds.push(TerrainCmd::Dispatch { kernel, threads });
    }

    fn indirect(&mut self, kernel: &'static str, offset: u64) {
        self.cmds.push(TerrainCmd::Indirect { kernel, offset });
    }

    /// One workgroup (64 threads) per block in use, counted on the GPU.
    fn blocks(&mut self, kernel: &'static str) {
        self.indirect(kernel, ARGS_BLOCKS);
    }

    /// One thread per particle in the region, counted on the GPU.
    fn particles(&mut self, kernel: &'static str) {
        self.indirect(kernel, ARGS_PARTICLES);
    }
}

/// Size a buffer of `bytes` is allocated with (never empty).
pub fn alloc_bytes(bytes: u64) -> u64 {
    bytes.max(256)
}

/// The size of the heightfield a region opens over: samples and cell size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldGrid {
    pub nx: u32,
    pub nz: u32,
    pub dx: f32,
    pub dz: f32,
}

/// A run `[base, base + len)` of a pooled buffer's records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slice {
    base: u32,
    len: u32,
}

/// First fit of `len` records in `[0, cap)` beside the `used` runs.
fn first_fit(used: &[Slice], len: u32, cap: u32) -> Option<u32> {
    if len == 0 {
        return Some(0);
    }
    let mut runs: Vec<Slice> = used.iter().copied().filter(|s| s.len > 0).collect();
    runs.sort_by_key(|s| s.base);
    let mut at = 0u32;
    for r in runs {
        if r.base >= at && r.base - at >= len {
            return Some(at);
        }
        at = at.max(r.base + r.len);
    }
    (cap.checked_sub(at)? >= len).then_some(at)
}

/// One open area: an activated box over the heightfield (or a host-loaded
/// region), with its own window of samples, absorbing faces and bake.
#[derive(Clone, Debug)]
struct Area {
    plan: RegionPlan,
    /// The box's minimum corner in blocks of the shared lattice.
    block_lo: IVec3,
    /// Its record (`gMpmParams[1 + slot]`).
    record: MpmParams,
    columns: Slice,
    faces: Slice,
    couple: Slice,
    /// Opened over a heightfield window (a host-loaded region has none).
    windowed: bool,
    /// Its particles have been stepped since it opened: a source on it finds
    /// the material's surface on the GPU.
    moved: bool,
}

/// The sparse grid every open area shares: one cell size and block lattice,
/// one bed heightfield.
#[derive(Clone, Copy, Debug)]
struct Domain {
    dx: f32,
    /// The lattice block of the domain's origin (the first area's corner):
    /// block keys are relative to it.
    block_lo: IVec3,
    bed: Option<u32>,
    field: Option<(FieldGrid, glam::Vec3)>,
    density: f32,
}

/// An open area, as hosts see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaInfo {
    /// Its slot (`MPM_C_AREA_SPEED + slot` is its speed counter).
    pub slot: u32,
    /// Its box.
    pub plan: RegionPlan,
    /// Heightfield samples under it: i0, j0, ni, nj (0 for a loaded region).
    pub window: [u32; 4],
    /// Its bed (heightfield-local y).
    pub bed: f32,
}

/// What [`TerrainPlan::open`] did with a request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Opened {
    /// The area that now covers the request.
    pub area: u32,
    /// Open areas it took over (grown into one box with the request), now
    /// closed as areas of their own.
    pub absorbed: Vec<u32>,
    /// Columns were activated (false: an open area already covered it).
    pub activated: bool,
    pub ops: TerrainOps,
}

/// Host state of the terrain pool, and the commands of each call.
///
/// The pool is one sparse domain: every open area shares its cell size, its
/// block lattice (keys relative to the first area's corner) and its bed
/// heightfield. A request whose box comes within [`TerrainPlan::merge_gap`]
/// of open areas grows them into one box covering all of them; otherwise it
/// opens an area of its own. Each area keeps its window, absorbing faces,
/// settling and bake; the substeps run over all their particles at once.
#[derive(Clone, Debug)]
pub struct TerrainPlan {
    config: TerrainConfig,
    /// The domain record (`gMpmParams[0]`).
    params: MpmParams,
    /// Elastic (P-) wave speed of the loaded material (0 without stress).
    wave_speed: f32,
    /// The loaded rock, whose crack band each step checks against the cell size.
    rock: Option<RockMaterial>,
    /// Activated particles wait for the next step's gravity (geostatic start).
    geostatic_pending: bool,
    domain: Option<Domain>,
    areas: Vec<Option<Area>>,
    /// The area records changed since they were last written.
    areas_dirty: bool,
}

impl TerrainPlan {
    /// The pool for `config` (no particles, no material).
    pub fn new(config: &TerrainConfig) -> anyhow::Result<Self> {
        config.validate()?;
        // Per-block and per-particle kernels run in a single dispatch row.
        anyhow::ensure!(
            config.max_blocks <= 65535 && config.max_particles <= 65535 * 64,
            "max_blocks {} or max_particles {} exceeds one indirect dispatch row (65535 workgroups)",
            config.max_blocks,
            config.max_particles
        );
        Ok(Self {
            config: *config,
            params: MpmParams {
                counts: [
                    config.max_particles,
                    hash_slots(config.max_blocks),
                    config.max_blocks,
                    MPM_MATERIAL_NONE,
                ],
                dims: [0, 0, 0, MPM_NO_BED],
                limits: [0.0, config.bake_home_fade, 0.0, 0.0],
                couple: [
                    0.0,
                    f32::from_bits(config.max_coupled),
                    f32::from_bits(config.max_parcels),
                    f32::from_bits(config.max_field_samples),
                ],
                ..MpmParams::default()
            },
            wave_speed: 0.0,
            rock: None,
            geostatic_pending: false,
            domain: None,
            areas: vec![None; MPM_MAX_AREAS as usize],
            areas_dirty: true,
        })
    }

    /// The bytes each binding of the terrain pool needs (before
    /// [`alloc_bytes`]): bindings 37-51. The look-ahead's binding
    /// ([`MPM_WATCH_BINDING`]) is not the pool's: hosts keep it at 256 bytes.
    pub fn buffer_sizes(&self) -> [(usize, u64); MPM_WATCH_BINDING - MPM_PARAMS_BINDING] {
        let config = &self.config;
        let blocks = u64::from(config.max_blocks);
        let particles = u64::from(config.max_particles);
        let slots = u64::from(hash_slots(config.max_blocks));
        let nodes = blocks * u64::from(BLOCK_CELLS.pow(3));
        [
            (MPM_PARAMS_BINDING, MPM_PARAMS_BYTES),
            (MPM_COUNTERS_BINDING, 4 * MPM_COUNTER_WORDS as u64),
            (
                MPM_PARTICLES_BINDING,
                particles * std::mem::size_of::<MpmParticle>() as u64,
            ),
            (MPM_PAYLOAD_BINDING, particles * 80),
            (MPM_HASH_BINDING, slots * 8),
            (MPM_BLOCKS_BINDING, blocks * 16),
            (MPM_CELLS_BINDING, nodes * 16),
            (MPM_SORTED_BINDING, particles * 4),
            (MPM_GRID_BINDING, nodes * 16),
            (MPM_ARGS_BINDING, 36),
            (MPM_COLUMNS_BINDING, u64::from(config.max_columns) * 16),
            (
                MPM_COUPLE_BINDING,
                (4 + blocks * u64::from(1 + MPM_CELL_COLLIDERS)) * 4,
            ),
            (
                MPM_COUPLED_BINDING,
                u64::from(config.max_coupled) * 64 + u64::from(config.max_boundary_nodes) * 16,
            ),
            (MPM_PARCELS_BINDING, u64::from(config.max_parcels) * 32),
            (MPM_DEPOSIT_BINDING, u64::from(config.max_field_samples) * 4),
        ]
    }

    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }

    /// The domain record (`gMpmParams[0]`).
    pub fn params(&self) -> &MpmParams {
        &self.params
    }

    /// Area `slot`'s record (`gMpmParams[1 + slot]`), if it is open.
    pub fn area_params(&self, slot: u32) -> Option<&MpmParams> {
        self.areas
            .get(slot as usize)?
            .as_ref()
            .map(|a| &a.record)
    }

    /// Elastic (P-) wave speed of the loaded material (m/s; 0 without stress).
    pub fn wave_speed(&self) -> f32 {
        self.wave_speed
    }

    /// Face points an absorbing boundary keeps a displacement for in a region
    /// of `cells` (each axis): the node lattice's columns through the floor and
    /// through each of the four sides (`mpmFaceRecord`), against
    /// [`TerrainConfig::max_boundary_nodes`].
    pub fn boundary_nodes(cells: [u32; 3]) -> u64 {
        let [x, y, z] = cells.map(|c| u64::from(c) + 1);
        x * z + 2 * y * (x + z)
    }

    /// The domain's bed heightfield collider, if it has one.
    pub fn bed(&self) -> Option<u32> {
        self.domain.and_then(|d| d.bed)
    }

    /// The open grid's cell size (m), while any area is open.
    pub fn domain_dx(&self) -> Option<f32> {
        self.domain.map(|d| d.dx)
    }

    /// The cell an event asking for cells of `asked` m runs at: the fixed
    /// [`TerrainConfig::cell_size`], else the open grid's, else its own. It
    /// fails ([`TerrainError::CellTooCoarse`]) when that cell resolves the
    /// event coarser than [`TerrainConfig::min_resolution`] allows.
    pub fn event_cell(&self, asked: f32) -> Result<f32, TerrainError> {
        if !(asked.is_finite() && asked > 0.0) {
            return Err(TerrainError::InvalidRequest("cell size must be positive"));
        }
        let grid = if self.config.cell_size > 0.0 {
            self.config.cell_size
        } else {
            self.domain_dx().unwrap_or(asked)
        };
        if grid * self.config.min_resolution > asked * (1.0 + 1e-6) {
            return Err(TerrainError::CellTooCoarse {
                asked,
                grid,
                min_resolution: self.config.min_resolution,
            });
        }
        Ok(grid)
    }

    /// The open areas, by slot.
    pub fn areas(&self) -> Vec<AreaInfo> {
        self.areas
            .iter()
            .enumerate()
            .filter_map(|(k, a)| {
                a.as_ref().map(|a| AreaInfo {
                    slot: k as u32,
                    plan: a.plan,
                    window: a.record.window,
                    bed: a.record.column[0],
                })
            })
            .collect()
    }

    /// The open area whose box holds `point` horizontally.
    pub fn area_at(&self, point: glam::Vec3) -> Option<u32> {
        self.areas.iter().enumerate().find_map(|(k, a)| {
            let a = a.as_ref()?;
            let (lo, hi) = (a.plan.origin, a.plan.end());
            (point.x >= lo.x && point.x < hi.x && point.z >= lo.z && point.z < hi.z)
                .then_some(k as u32)
        })
    }

    /// Blocks of clear ground two areas keep between their boxes (else a
    /// request touching both grows them into one): two blocks, so no grid
    /// block holds nodes of two areas, and at least two heightfield samples,
    /// so one area's bake and deposits stay off another's window.
    pub fn merge_gap(&self) -> i32 {
        let Some(d) = self.domain else { return 2 };
        let block = d.dx * BLOCK_CELLS as f32;
        let samples = d
            .field
            .map_or(0.0, |(f, _)| 2.0 * f.dx.max(f.dz) / block);
        2.max(samples.ceil() as i32)
    }

    fn slots_in_use(&self) -> u32 {
        self.areas
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |k| k as u32 + 1)
    }

    fn open_count(&self) -> u32 {
        self.areas.iter().filter(|a| a.is_some()).count() as u32
    }

    /// The domain record with `sel` selected, `merge` the areas it takes over
    /// and `flags`, written ahead of the pass, with the area records if they
    /// changed (an area record never changes while it is open).
    fn write_params(&mut self, ops: &mut TerrainOps, sel: u32, merge: u32, flags: u32) {
        self.params.area = [self.slots_in_use(), sel, merge, flags];
        ops.params(&self.params);
        if std::mem::take(&mut self.areas_dirty) {
            let mut records = vec![MpmParams::default(); MPM_MAX_AREAS as usize];
            for (k, a) in self.areas.iter().enumerate() {
                if let Some(a) = a {
                    records[k] = a.record;
                }
            }
            ops.write_words(
                MPM_PARAMS_BINDING,
                std::mem::size_of::<MpmParams>() as u64,
                bytemuck::cast_slice(&records),
            );
        }
    }

    /// The domain starts afresh: its parcels are thrown from slot 0 and
    /// counted anew.
    fn reset_ejecta(ops: &mut TerrainOps) {
        for w in [MPM_C_PARCELS, MPM_C_DEPOSITED, MPM_C_ESCAPED] {
            ops.counter(w, 0);
        }
    }

    fn check_lattice(plan: &RegionPlan) -> anyhow::Result<IVec3> {
        let block = plan.dx * BLOCK_CELLS as f32;
        let lo = (plan.origin / block).round().as_ivec3();
        anyhow::ensure!(
            ((plan.origin / block) - lo.as_vec3()).abs().max_element() < 1e-3,
            "region origin {:?} is not on the block lattice of cells {}",
            plan.origin,
            plan.dx
        );
        Ok(lo)
    }

    /// Make `plan` the only area, holding `particles` host-loaded particles
    /// (the host writes them to [`MPM_PARTICLES_BINDING`] itself). Any open
    /// area is dropped.
    pub fn load(
        &mut self,
        plan: &RegionPlan,
        particles: usize,
        bed: Option<u32>,
        bed_friction: f32,
    ) -> anyhow::Result<TerrainOps> {
        let n = u32::try_from(particles).unwrap_or(u32::MAX);
        TerrainConfig::check(
            "max_particles",
            u64::from(n),
            u64::from(self.config.max_particles),
        )?;
        anyhow::ensure!(
            plan.blocks.max_element() as u32 <= MPM_MAX_BLOCKS_PER_AXIS,
            "region spans {:?} blocks; block keys hold {MPM_MAX_BLOCKS_PER_AXIS} per axis",
            plan.blocks
        );
        let block_lo = Self::check_lattice(plan)?;
        self.domain = Some(Domain {
            dx: plan.dx,
            block_lo,
            bed,
            field: None,
            density: 0.0,
        });
        self.areas = vec![None; MPM_MAX_AREAS as usize];
        let cells = plan.blocks * BLOCK_CELLS as i32;
        self.params.origin = [plan.origin.x, plan.origin.y, plan.origin.z, plan.dx];
        self.params.dims = [0, 0, 0, bed.unwrap_or(MPM_NO_BED)];
        self.params.step[3] = bed_friction;
        // No activation window: the bed is the field's surface everywhere, a
        // contact (the absorbing ground is the window's floor and sides).
        let record = MpmParams {
            origin: self.params.origin,
            dims: [
                cells.x as u32,
                cells.y as u32,
                cells.z as u32,
                bed.unwrap_or(MPM_NO_BED),
            ],
            window: [0; 4],
            column: self.params.column,
            boundary: [0; 4],
            area: [0; 4],
            ..self.params
        };
        // Its box's block cells (a step past max_blocks fails on the GPU).
        let count = (plan.blocks.x as u64 * plan.blocks.y as u64 * plan.blocks.z as u64)
            .min(u64::from(self.config.max_blocks)) as u32;
        self.areas[0] = Some(Area {
            plan: *plan,
            block_lo,
            record,
            columns: Slice { base: 0, len: 0 },
            faces: Slice { base: 0, len: 0 },
            couple: Slice {
                base: 0,
                len: count,
            },
            windowed: false,
            moved: false,
        });
        self.areas_dirty = true;
        self.geostatic_pending = false;
        let mut ops = TerrainOps::default();
        ops.counter(MPM_C_PARTICLES, n);
        Self::reset_ejecta(&mut ops);
        Ok(ops)
    }

    /// Open `plan` over heightfield `collider` (see [`Self::open`]) and return
    /// its commands.
    pub fn activate(
        &mut self,
        plan: &RegionPlan,
        collider: u32,
        field: FieldGrid,
        field_origin: glam::Vec3,
        density: f32,
        bed_friction: f32,
    ) -> anyhow::Result<TerrainOps> {
        Ok(self
            .open(plan, collider, field, field_origin, density, bed_friction)?
            .ops)
    }

    /// Open `request` over heightfield `collider` (`field`'s grid, its frame
    /// at `field_origin` with no rotation, y up).
    ///
    /// - Far from every open area, it becomes an area of its own: every sample
    ///   whose cell lies inside its box (a cell in) is lowered to the box's
    ///   bed, and the material above becomes particles of `density` holding
    ///   exactly the volume removed (in whole quanta).
    /// - Within [`Self::merge_gap`] blocks of open areas (overlapping them, on
    ///   their rim, partly outside them), those areas and the request grow
    ///   into one box. Its samples not yet active are activated; a sample that
    ///   was gives the slab between its old bed and a deeper new one. The
    ///   moving material stays as it is.
    /// - Inside one open area, nothing is activated.
    ///
    /// Every area shares the domain's cell size, lattice and bed: a request on
    /// another heightfield, or with other cells, fails. An activation past
    /// `max_particles` changes nothing and fails the next step's check.
    pub fn open(
        &mut self,
        request: &RegionPlan,
        collider: u32,
        field: FieldGrid,
        field_origin: glam::Vec3,
        density: f32,
        bed_friction: f32,
    ) -> anyhow::Result<Opened> {
        let cfg = self.config;
        anyhow::ensure!(
            density > 0.0 && density.is_finite(),
            "density must be positive"
        );
        let dx = request.dx;
        let req_lo = Self::check_lattice(request)?;
        let req_hi = req_lo + request.blocks;
        // A host-loaded region has no window to bake: an activation replaces it.
        if self.areas.iter().flatten().any(|a| !a.windowed) {
            self.domain = None;
            self.areas = vec![None; MPM_MAX_AREAS as usize];
        }
        if let Some(d) = self.domain {
            anyhow::ensure!(
                d.dx.to_bits() == dx.to_bits(),
                "the request's cells ({dx} m) differ from the open terrain grid's ({} m): every \
                 open area shares one grid (TerrainPlan::event_cell picks its cell)",
                d.dx
            );
            anyhow::ensure!(
                d.bed == Some(collider) && d.field == Some((field, field_origin)),
                "terrain areas are open on another heightfield: one heightfield at a time"
            );
            anyhow::ensure!(
                d.density.to_bits() == density.to_bits(),
                "the open terrain's material has density {} kg/m³, not {density}",
                d.density
            );
        } else {
            anyhow::ensure!(
                request.blocks.max_element() as u32 <= MPM_MAX_BLOCKS_PER_AXIS,
                "region spans {:?} blocks; block keys hold {MPM_MAX_BLOCKS_PER_AXIS} per axis",
                request.blocks
            );
            self.domain = Some(Domain {
                dx,
                block_lo: req_lo,
                bed: Some(collider),
                field: Some((field, field_origin)),
                density,
            });
            self.areas = vec![None; MPM_MAX_AREAS as usize];
            self.areas_dirty = true;
            self.params.origin = [request.origin.x, request.origin.y, request.origin.z, dx];
            self.params.dims = [0, 0, 0, collider];
        }
        let fresh_domain = self.open_count() == 0;
        let domain = self.domain.expect("set above");
        // The areas the request reaches, and the box covering them all.
        let gap = self.merge_gap();
        let (mut lo, mut hi) = (req_lo, req_hi);
        let mut merge: Vec<u32> = Vec::new();
        loop {
            let before = merge.len();
            for (k, a) in self.areas.iter().enumerate() {
                let Some(a) = a else { continue };
                if merge.contains(&(k as u32)) {
                    continue;
                }
                let (alo, ahi) = (a.block_lo, a.block_lo + a.plan.blocks);
                let near = alo.x - gap < hi.x
                    && lo.x < ahi.x + gap
                    && alo.z - gap < hi.z
                    && lo.z < ahi.z + gap;
                if near {
                    merge.push(k as u32);
                    lo = lo.min(alo);
                    hi = hi.max(ahi);
                }
            }
            if merge.len() == before {
                break;
            }
        }
        merge.sort_unstable();
        if let [k] = merge[..] {
            let a = self.areas[k as usize].as_ref().expect("open");
            if a.block_lo == lo && a.block_lo + a.plan.blocks == hi {
                return Ok(Opened {
                    area: k,
                    absorbed: Vec::new(),
                    activated: false,
                    ops: TerrainOps::default(),
                });
            }
        }
        TerrainConfig::check(
            "max_regions",
            u64::from(self.open_count() - merge.len() as u32 + 1),
            u64::from(cfg.max_regions),
        )?;
        let rel_lo = lo - domain.block_lo;
        let rel_hi = hi - domain.block_lo;
        let [bx, by, bz] = MPM_KEY_BIAS;
        anyhow::ensure!(
            rel_lo.x >= -bx
                && rel_lo.y >= -by
                && rel_lo.z >= -bz
                && rel_hi.x < bx - 1
                && rel_hi.y < by
                && rel_hi.z < bz - 1,
            "terrain areas would span blocks {rel_lo:?}..{rel_hi:?} from the grid's origin; block \
             keys reach {:?} each way (bake the open areas, or use larger cells)",
            MPM_KEY_BIAS
        );
        let block = dx * BLOCK_CELLS as f32;
        let plan = RegionPlan {
            dx,
            origin: lo.as_vec3() * block,
            blocks: hi - lo,
        };
        let FieldGrid {
            nx,
            nz,
            dx: sx,
            dz: sz,
        } = field;
        let x0 = field_origin.x - 0.5 * (nx - 1) as f32 * sx;
        let z0 = field_origin.z - 0.5 * (nz - 1) as f32 * sz;
        let blo = plan.origin + glam::Vec3::splat(dx);
        let bhi = plan.end() - glam::Vec3::splat(dx);
        // Interior samples (full-area cells) whose cell lies within [blo, bhi].
        let range = |o: f32, d: f32, n: u32, a: f32, b: f32| {
            let first = ((a - o) / d + 0.5).ceil().max(1.0) as u32;
            let last = ((b - o) / d - 0.5).floor().min((n - 2) as f32) as i64;
            (first, (last - i64::from(first) + 1).max(0) as u32)
        };
        let (i0, ni) = range(x0, sx, nx, blo.x, bhi.x);
        let (j0, nj) = range(z0, sz, nz, blo.z, bhi.z);
        anyhow::ensure!(ni > 0 && nj > 0, "the region covers no heightfield samples");
        TerrainConfig::check(
            "max_field_samples",
            u64::from(nx) * u64::from(nz),
            u64::from(cfg.max_field_samples),
        )?;
        let spacing = dx / cfg.particles_per_cell_axis as f32;
        let column = [
            blo.y + 0.5 * dx - field_origin.y,
            spacing.powi(3) / QUANTA_PER_LATTICE_CELL,
            spacing,
            density,
        ];
        let cells = plan.blocks * BLOCK_CELLS as i32;
        let cells = [cells.x as u32, cells.y as u32, cells.z as u32];
        let geostatic = cfg.initial_stress == InitialStress::Geostatic;
        let absorbing = cfg.boundary == RegionBoundary::Absorbing;
        // Pool slices, beside every open area's (the merged ones' included:
        // the activation reads them).
        let live: Vec<&Area> = self.areas.iter().flatten().collect();
        let fit = |limit: &'static str,
                   used: &dyn Fn(&Area) -> Slice,
                   len: u64,
                   cap: u32|
         -> anyhow::Result<Slice> {
            let used: Vec<Slice> = live.iter().map(|a| used(a)).collect();
            let total: u64 = used.iter().map(|s| u64::from(s.len)).sum::<u64>() + len;
            TerrainConfig::check(limit, total, u64::from(cap))?;
            let len = len as u32;
            let base = first_fit(&used, len, cap).ok_or(TerrainError::LimitExceeded {
                limit,
                requested: total,
                allowed: u64::from(cap),
            })?;
            Ok(Slice { base, len })
        };
        let columns = fit(
            "max_columns",
            &|a| a.columns,
            u64::from(ni) * u64::from(nj),
            cfg.max_columns,
        )?;
        let faces = if absorbing {
            // The faces' points; the bake reuses them for the window's edge
            // samples (mpm_bake_shift).
            let n = Self::boundary_nodes(cells).max(2 * u64::from(ni + nj));
            fit("max_boundary_nodes", &|a| a.faces, n, cfg.max_boundary_nodes)?
        } else {
            Slice { base: 0, len: 0 }
        };
        let couple = fit(
            "max_blocks",
            &|a| a.couple,
            plan.blocks.x as u64 * plan.blocks.y as u64 * plan.blocks.z as u64,
            cfg.max_blocks,
        )?;
        let slot = self
            .areas
            .iter()
            .position(Option::is_none)
            .expect("max_regions is at most MPM_MAX_AREAS") as u32;
        let off = rel_lo * BLOCK_CELLS as i32;
        let record = MpmParams {
            origin: [plan.origin.x, plan.origin.y, plan.origin.z, dx],
            dims: [cells[0], cells[1], cells[2], collider],
            window: [i0, j0, ni, nj],
            column,
            boundary: [
                if absorbing { MPM_BOUNDARY_ABSORBING } else { 0 }
                    | if geostatic { MPM_BOUNDARY_GEOSTATIC } else { 0 },
                4 * cfg.max_coupled + faces.base,
                faces.len,
                columns.base,
            ],
            area: [off.x as u32, off.y as u32, off.z as u32, couple.base],
            ..self.params
        };
        let moved = merge
            .iter()
            .any(|&k| self.areas[k as usize].as_ref().is_some_and(|a| a.moved));
        self.areas[slot as usize] = Some(Area {
            plan,
            block_lo: lo,
            record,
            columns,
            faces,
            couple,
            windowed: true,
            moved,
        });
        self.areas_dirty = true;
        // Domain-wide: the material's quantum, spacing and density.
        self.params.column = [0.0, column[1], column[2], column[3]];
        self.params.step[3] = bed_friction;
        let mask = merge.iter().fold(0u32, |m, &k| m | (1 << k));
        let mut ops = TerrainOps::default();
        self.write_params(&mut ops, slot, mask, 0);
        if fresh_domain {
            ops.counter(MPM_C_PARTICLES, 0);
            Self::reset_ejecta(&mut ops);
        }
        let samples = u64::from(ni) * u64::from(nj);
        let first = (j0 * nx + i0) / 4;
        let last = ((j0 + nj - 1) * nx + i0 + ni - 1) / 4;
        ops.disp("mpm_activate_count", samples);
        ops.disp("mpm_activate_scan", 64);
        ops.disp("mpm_activate_emit", samples);
        ops.disp("mpm_activate_lower", u64::from(last - first + 1));
        if faces.len > 0 {
            ops.disp("mpm_absorb_init", u64::from(faces.len));
        }
        // The merged areas are this one now; their records go with the next
        // write.
        for &k in &merge {
            self.areas[k as usize] = None;
            self.areas_dirty = true;
        }
        self.geostatic_pending |= geostatic;
        Ok(Opened {
            area: slot,
            absorbed: merge,
            activated: true,
            ops,
        })
    }

    /// Close every open area (see [`Self::bake_area`]), one pass each.
    pub fn bake(&mut self) -> anyhow::Result<Vec<TerrainOps>> {
        let slots: Vec<u32> = self.areas().iter().map(|a| a.slot).collect();
        anyhow::ensure!(
            !slots.is_empty() && self.areas.iter().flatten().all(|a| a.windowed),
            "terrain_bake needs a region opened by terrain_activate"
        );
        slots.into_iter().map(|k| self.bake_area(k)).collect()
    }

    /// Close area `slot`: each of its particles' quanta are split, in whole
    /// quanta, between the heightfield it came from (the tent at its home
    /// sample moved by its displacement) and material the MPM moved (P2G's
    /// B-spline at its position), onto window samples directly and others
    /// through the deposits. Its samples are rigid ground again; other areas
    /// go on. The last area's bake empties the particle pool.
    pub fn bake_area(&mut self, slot: u32) -> anyhow::Result<TerrainOps> {
        let area = self
            .areas
            .get(slot as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::anyhow!("terrain area {slot} is not open"))?;
        anyhow::ensure!(
            area.windowed,
            "terrain_bake needs a region opened by terrain_activate"
        );
        let rec = area.record;
        let nx = self
            .domain
            .and_then(|d| d.field)
            .map(|(f, _)| f.nx)
            .expect("an activated area has a field");
        let [i0, j0, ni, nj] = rec.window;
        let first = (j0 * nx + i0) / 4;
        let last = ((j0 + nj - 1) * nx + i0 + ni - 1) / 4;
        let close = self.open_count() == 1;
        let mut ops = TerrainOps::default();
        self.write_params(&mut ops, slot, 0, if close { MPM_DOMAIN_CLOSE } else { 0 });
        // On the absorbing ground: the walls' give, measured on the window's
        // edge columns, for the bake to take out (in the face records' space).
        if rec.boundary[2] > 0 {
            let edges = 2 * u64::from(ni + nj);
            ops.disp("mpm_write_args", 1);
            ops.disp("mpm_bake_edge_clear", edges);
            ops.particles("mpm_bake_edge_count");
            ops.disp("mpm_bake_edge_scan", 64);
            ops.particles("mpm_bake_edge_scatter");
            ops.disp("mpm_bake_edge_sort", edges);
            ops.disp("mpm_bake_shift", edges);
        }
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_bake_bin");
        ops.disp("mpm_bake_raise", u64::from(last - first + 1));
        ops.disp("mpm_bake_header", 1);
        self.deposits(&mut ops);
        self.areas[slot as usize] = None;
        self.areas_dirty = true;
        if close {
            self.domain = None;
        }
        Ok(ops)
    }

    /// The loaded material's constitutive model (see `WgpuSim::terrain_set_material`).
    pub fn set_material(&mut self, material: impl Into<TerrainMaterial>) -> anyhow::Result<()> {
        let material = material.into();
        material.validate()?;
        self.rock = None;
        self.wave_speed = material.p_wave_speed();
        let (kind, words) = match material {
            TerrainMaterial::Inert => (MPM_MATERIAL_NONE, [0.0; 4]),
            TerrainMaterial::Sand(m) => {
                let (kind, friction, cohesion) = match m.yield_surface {
                    // Dry sand keeps its kernel (and its results) unchanged.
                    YieldSurface::MohrCoulomb if m.cohesion == 0.0 => {
                        (MPM_MATERIAL_SAND_MC, m.friction_angle.sin(), 0.0)
                    }
                    YieldSurface::MohrCoulomb => (
                        MPM_MATERIAL_SOIL_MC,
                        m.friction_angle.sin(),
                        m.cohesion_cos_phi(),
                    ),
                    YieldSurface::DruckerPrager => (MPM_MATERIAL_SAND, m.alpha(), m.apex_strain()),
                };
                (kind, [m.lambda(), m.mu(), friction, cohesion])
            }
            TerrainMaterial::Rock(m) => {
                self.rock = Some(m);
                (
                    MPM_MATERIAL_ROCK,
                    [m.lambda(), m.mu(), m.tensile_strength, m.band_stress()],
                )
            }
        };
        self.params.counts[3] = kind;
        self.params.sand = words;
        Ok(())
    }

    /// The substeps' parameters (dt, gravity, the CFL speed bound), as a write.
    fn step_params(
        &mut self,
        ops: &mut TerrainOps,
        substeps: u32,
        dt: f32,
        gravity: f32,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            substeps > 0 && dt > 0.0,
            "terrain substeps need a positive count and dt"
        );
        TerrainConfig::check(
            "max_substeps",
            u64::from(substeps),
            u64::from(self.config.max_substeps),
        )?;
        if let Some(rock) = self.rock {
            rock.softening_stress(self.params.origin[3])?;
        }
        anyhow::ensure!(
            self.areas
                .iter()
                .flatten()
                .all(|a| a.record.boundary[0] & MPM_BOUNDARY_ABSORBING == 0)
                || self.wave_speed > 0.0,
            "the absorbing boundary needs a material with stiffness (terrain_set_material)"
        );
        self.params.step[0] = dt;
        self.params.step[1] = gravity;
        // Nodes lighter than this carry no velocity (f32 momentum over mass).
        self.params.step[2] = 1.0e-12;
        // CFL: courant·dx per substep, less what the elastic wave takes.
        let courant_speed = self.config.courant * self.params.origin[3] / dt;
        self.params.limits[0] = courant_speed - self.wave_speed;
        anyhow::ensure!(
            self.params.limits[0] > 0.0,
            "terrain dt {dt} s is above the CFL bound for the material's wave speed ({} m/s): use \
             engine_terrain::substeps_for",
            self.wave_speed
        );
        self.write_params(ops, 0, 0, 0);
        for a in self.areas.iter_mut().flatten() {
            a.moved = true;
        }
        // Freshly activated particles take their stress before the first
        // substep.
        if std::mem::take(&mut self.geostatic_pending) {
            ops.disp("mpm_write_args", 1);
            ops.particles("mpm_geostatic");
        }
        Ok(())
    }

    /// `substeps` substeps of `dt` under `gravity` (m/s² along y) with no
    /// rigid coupling, then the parcels' deposits.
    pub fn step(&mut self, substeps: u32, dt: f32, gravity: f32) -> anyhow::Result<TerrainOps> {
        let mut ops = TerrainOps::default();
        self.step_params(&mut ops, substeps, dt, gravity)?;
        // No colliders listed: the grid holds no rigid contacts.
        ops.disp("mpm_couple_clear", u64::from(self.config.max_blocks));
        for _ in 0..substeps {
            self.substep(&mut ops, None);
        }
        self.deposits(&mut ops);
        Ok(ops)
    }

    /// A coupled frame (explicit, staggered) ahead of the rigid step: the
    /// rigid colliders (`colliders` in the world) overlapping the open areas
    /// are listed, each substep holds the material to them, and their
    /// reactions become impulse records the rigid step applies at its start
    /// (so the step must run with its impulse pass on).
    pub fn coupled(
        &mut self,
        substeps: u32,
        dt: f32,
        gravity: f32,
        colliders: u32,
    ) -> anyhow::Result<TerrainOps> {
        self.params.couple[0] = substeps as f32 * dt;
        let mut ops = TerrainOps::default();
        self.step_params(&mut ops, substeps, dt, gravity)?;
        let slots = u64::from(self.config.max_coupled);
        ops.disp("mpm_couple_clear", u64::from(self.config.max_blocks));
        ops.disp("mpm_couple_bin", u64::from(colliders));
        for _ in 0..substeps {
            self.substep(&mut ops, Some(slots));
        }
        self.deposits(&mut ops);
        ops.disp("mpm_couple_emit", slots);
        ops.disp("mpm_couple_finish", 1);
        Ok(ops)
    }

    /// One past the last absorbing face record any open area uses.
    fn faces_end(&self) -> u32 {
        self.areas
            .iter()
            .flatten()
            .map(|a| a.faces.base + a.faces.len)
            .max()
            .unwrap_or(0)
    }

    /// One MPM substep's dispatches. With `coupled` (the coupling slots), the
    /// rigid contacts' exchange runs between the grid update and G2P.
    fn substep(&self, ops: &mut TerrainOps, coupled: Option<u64>) {
        // Hash words to empty.
        let clear = 2 * u64::from(self.params.counts[1]);
        ops.disp("mpm_clear", clear);
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_touch");
        ops.disp("mpm_write_args", 1);
        ops.particles("mpm_count");
        ops.blocks("mpm_scan_cells");
        ops.particles("mpm_scatter");
        ops.blocks("mpm_cell_sort");
        ops.particles("mpm_payload");
        ops.blocks("mpm_p2g");
        ops.blocks("mpm_grid");
        if let Some(slots) = coupled {
            ops.disp("mpm_couple_gather", slots * 64);
            ops.blocks("mpm_couple_apply");
        }
        ops.particles("mpm_g2p");
        // The absorbing faces move with the material at them.
        let faces = u64::from(self.faces_end());
        if faces > 0 {
            ops.disp("mpm_absorb_faces", faces);
        }
        ops.indirect("mpm_parcels", ARGS_PARCELS);
    }

    /// Once per frame: the parcels' deposits onto the heightfield.
    fn deposits(&self, ops: &mut TerrainOps) {
        let records = u64::from(self.config.max_field_samples).div_ceil(4);
        ops.disp("mpm_deposit_raise", records);
        ops.disp("mpm_deposit_header", 1);
    }

    /// Start a km/s impact or explosion at `point` (on the surface) in the
    /// open area under it: see [`Self::apply_source_to`].
    pub fn apply_source(
        &mut self,
        field: &ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<TerrainOps> {
        let area = self
            .area_at(point)
            .ok_or_else(|| anyhow::anyhow!("no open terrain area under the source point"))?;
        self.apply_source_to(area, field, point)
    }

    /// Start a km/s impact or explosion at `point` in area `slot`: every live
    /// particle of the area within the field's extent takes `field`'s
    /// excavation velocity, added to its own (the impulses of several
    /// sources superpose on the moving material). On an area whose material
    /// has moved since it opened, the source sits on the material's surface
    /// under `point` (its highest particle there, found on the GPU), not at
    /// `point`'s height.
    pub fn apply_source_to(
        &mut self,
        slot: u32,
        field: &ZModel,
        point: glam::Vec3,
    ) -> anyhow::Result<TerrainOps> {
        anyhow::ensure!(point.is_finite(), "source point must be finite");
        let area = self
            .areas
            .get(slot as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow::anyhow!("terrain area {slot} is not open"))?;
        let surface = area.moved;
        self.params.source = [
            [point.x, point.y, point.z, field.z as f32],
            [
                field.flow_center_depth as f32,
                field.core_radius as f32,
                field.alpha as f32,
                if field.extent.is_finite() {
                    field.extent as f32
                } else {
                    f32::MAX
                },
            ],
        ];
        let mut ops = TerrainOps::default();
        let flags = if surface { MPM_DOMAIN_SURFACE } else { 0 };
        self.write_params(&mut ops, slot, 0, flags);
        ops.disp("mpm_write_args", 1);
        if surface {
            ops.counter(MPM_C_SURFACE, 0);
            ops.particles("mpm_source_surface");
        }
        ops.particles("mpm_source");
        Ok(ops)
    }

    /// After a host has read the counters: their flags start clear again.
    pub fn flags_reset(&self) -> TerrainWrite {
        TerrainWrite {
            binding: MPM_COUNTERS_BINDING,
            offset: 4 * MPM_C_FLAGS as u64,
            words: vec![0],
        }
    }

    /// The flags word as a `clear_buffer` range (binding, byte offset, bytes),
    /// for pipelined hosts that clear it after copying the counters.
    pub fn flags_clear(&self) -> (usize, u64, u64) {
        (MPM_COUNTERS_BINDING, 4 * MPM_C_FLAGS as u64, 4)
    }

    /// The report a copy of the counters (`MPM_COUNTER_WORDS` words) carries,
    /// or the error a flag names.
    pub fn check(&self, c: &[u32]) -> anyhow::Result<TerrainStepReport> {
        let (cfg, params) = (self.config, self.params);
        anyhow::ensure!(
            c.len() >= MPM_COUNTER_WORDS,
            "terrain counters are {MPM_COUNTER_WORDS} words"
        );
        let flags = c[MPM_C_FLAGS];
        if flags & MPM_OVERFLOW_PARTICLES != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_particles",
                requested: u64::from(cfg.max_particles) + 1,
                allowed: u64::from(cfg.max_particles),
            }
            .into());
        }
        if flags & MPM_OVERFLOW_BLOCKS != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_blocks",
                requested: u64::from(c[MPM_C_BLOCKS]),
                allowed: u64::from(cfg.max_blocks),
            }
            .into());
        }
        if flags & MPM_OVERFLOW_PARCELS != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_parcels",
                requested: u64::from(c[MPM_C_PARCELS]),
                allowed: u64::from(cfg.max_parcels),
            }
            .into());
        }
        anyhow::ensure!(
            flags & MPM_OVERFLOW_FIELD == 0,
            "the bed heightfield has more samples than max_field_samples ({})",
            cfg.max_field_samples
        );
        if flags & MPM_OVERFLOW_COUPLED != 0 {
            return Err(TerrainError::LimitExceeded {
                limit: "max_coupled",
                requested: u64::from(cfg.max_coupled) + 1,
                allowed: u64::from(cfg.max_coupled),
            }
            .into());
        }
        anyhow::ensure!(
            flags & MPM_OVERFLOW_CELL == 0,
            "more than {MPM_CELL_COLLIDERS} rigid colliders overlap one terrain block (4 cells across)"
        );
        anyhow::ensure!(
            flags & MPM_OVERFLOW_IMPULSES == 0,
            "the rigid step's impulse records are full: terrain reactions were not all applied"
        );
        anyhow::ensure!(
            flags & MPM_OVERFLOW_HASH == 0,
            "terrain block hash full ({} slots)",
            params.counts[1]
        );
        anyhow::ensure!(
            flags & MPM_OUTSIDE == 0,
            "terrain material left the active region (one cell inside its box): enlarge the region's margin"
        );
        anyhow::ensure!(
            flags & MPM_CFL == 0,
            "terrain material moved faster ({} m/s) than the substep's CFL bound ({} m/s): take \
             more substeps (engine_terrain::substeps_for with a higher max_speed)",
            f32::from_bits(c[MPM_C_MAX_SPEED]),
            params.limits[0]
        );
        anyhow::ensure!(
            flags & MPM_INVALID == 0,
            "a terrain particle's deformation or plastic state is not finite"
        );
        Ok(TerrainStepReport {
            blocks: c[MPM_C_BLOCKS],
            max_speed: f32::from_bits(c[MPM_C_MAX_SPEED]),
            particles: c[MPM_C_PARTICLES],
            parcels: c[MPM_C_PARCELS].min(cfg.max_parcels),
            deposited: c[MPM_C_DEPOSITED],
            escaped: c[MPM_C_ESCAPED],
            airborne: c[MPM_C_AIRBORNE],
            body_speed: f32::from_bits(c[MPM_C_BODY_SPEED]),
            area_speed: std::array::from_fn(|k| f32::from_bits(c[MPM_C_AREA_SPEED + k])),
            area_body_speed: std::array::from_fn(|k| f32::from_bits(c[MPM_C_AREA_BODY + k])),
        })
    }
}

/// The open region's frame schedule (`engine-world`'s terrain host): each
/// frame's substeps cover the frame at the CFL bound for a speed bound the
/// GPU checks, from the last reported speed; the region is settled once its
/// fastest particle has stayed under a rest speed, with nothing in flight,
/// for a number of frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionClock {
    /// Speed bound basis: the source's peak, then each frame's report.
    speed: f32,
    /// The step whose report set `speed`.
    speed_step: u64,
    /// Consecutive settled frames.
    quiet: u32,
    /// No settling before this step (a strike still on its way in).
    hold_until: u64,
}

impl RegionClock {
    /// A region opened at `step` whose material starts at up to `speed` m/s.
    pub fn new(speed: f32, step: u64) -> Self {
        Self {
            speed,
            speed_step: step,
            quiet: 0,
            hold_until: 0,
        }
    }

    /// Keep the region open (not settled) until `step`.
    pub fn hold_until(&mut self, step: u64) {
        self.hold_until = self.hold_until.max(step);
    }

    /// A source adds up to `speed` m/s to material already moving (its
    /// velocity superposes on the flow): from `step` the bound starts from
    /// the sum, and reports from before it are ignored.
    pub fn kick(&mut self, speed: f32, step: u64) {
        self.speed += speed;
        self.speed_step = self.speed_step.max(step);
        self.quiet = 0;
    }

    /// Areas grown into one: the merged area's clock starts at `step` from
    /// the fastest of theirs, held as long as the longest hold.
    pub fn merged(clocks: &[RegionClock], step: u64) -> Self {
        Self {
            speed: clocks.iter().map(|c| c.speed).fold(0.0, f32::max),
            speed_step: step,
            quiet: 0,
            hold_until: clocks.iter().map(|c| c.hold_until).max().unwrap_or(0),
        }
    }

    /// `(substeps, dt)` for the frame at `step`: the bound is 1.5 × the last
    /// reported speed, plus what gravity can add over the readback lag, plus
    /// `headroom` (bodies entering faster than the sand).
    #[allow(clippy::too_many_arguments)]
    pub fn substeps(
        &self,
        config: &TerrainConfig,
        step: u64,
        frame_dt: f32,
        gravity: f32,
        dx: f32,
        wave_speed: f32,
        headroom: f32,
    ) -> Result<(u32, f32), TerrainError> {
        let lag = (step - self.speed_step + 1) as f32;
        let bound = 1.5 * self.speed + gravity.abs() * frame_dt * lag + headroom;
        substeps_for(config, frame_dt, dx, wave_speed, bound)
    }

    /// The report of the frame at `step` (reports arrive in order; one from
    /// before the region opened is ignored). A frame is quiet when its
    /// material and every body coupled to it move slower than `settle_speed`,
    /// nothing is in flight, and no hold is on.
    pub fn report(&mut self, step: u64, report: &TerrainStepReport, settle_speed: f32) {
        if step < self.speed_step {
            return;
        }
        self.speed = report.max_speed;
        self.speed_step = step;
        if report.max_speed < settle_speed
            && report.body_speed < settle_speed
            && report.airborne == 0
            && step >= self.hold_until
        {
            self.quiet += 1;
        } else {
            self.quiet = 0;
        }
    }

    /// Settled for `frames` frames: bake it.
    pub fn settled(&self, frames: u32) -> bool {
        self.quiet >= frames
    }

    /// The speed the next frame's bound starts from (m/s).
    pub fn speed(&self) -> f32 {
        self.speed
    }
}

/// The largest speed a Z-model excavation field starts: at its core's edge.
pub fn source_peak_speed(model: &ZModel) -> f32 {
    let core = model.core_radius as f32;
    let depth = model.flow_center_depth as f32;
    let deep = model
        .velocity_f32(glam::Vec3::new(0.0, -depth, core))
        .length();
    deep.max(model.surface_speed(model.core_radius) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> FieldGrid {
        FieldGrid {
            nx: 33,
            nz: 33,
            dx: 0.125,
            dz: 0.125,
        }
    }

    #[test]
    fn a_coupled_frame_lists_colliders_once_and_gathers_every_substep() {
        let cfg = TerrainConfig::default();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let region = RegionPlan::spanning(
            &cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        p.activate(&region, 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        let ops = p.coupled(3, 1.0e-4, -9.81, 5).unwrap();
        let count = |k: &str| {
            ops.cmds
                .iter()
                .filter(|c| matches!(c, TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } if *kernel == k))
                .count()
        };
        assert_eq!(count("mpm_couple_bin"), 1);
        assert_eq!(count("mpm_couple_gather"), 3);
        assert_eq!(count("mpm_couple_apply"), 3);
        assert_eq!(count("mpm_p2g"), 3);
        assert_eq!(count("mpm_couple_finish"), 1);
        // The params ride ahead of the pass.
        assert_eq!(ops.writes.len(), 1);
        assert_eq!(ops.writes[0].binding, MPM_PARAMS_BINDING);
        // Every kernel the plan names is in KERNELS.
        for c in &ops.cmds {
            let (TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. }) = c;
            assert!(KERNELS.contains(kernel), "{kernel}");
        }
    }

    #[test]
    fn a_step_past_the_cfl_bound_fails_before_any_work() {
        let cfg = TerrainConfig::default();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
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
        assert!(p.step(1, 1.0 / 60.0, -9.81).is_err());
        assert!(
            p.bake().is_err(),
            "a loaded region has no window to bake into"
        );
    }

    #[test]
    fn the_clock_settles_after_quiet_frames_and_resets_on_motion() {
        let mut c = RegionClock::new(100.0, 10);
        let r = |s: f32, air: u32| TerrainStepReport {
            blocks: 1,
            max_speed: s,
            particles: 1,
            parcels: 0,
            deposited: 0,
            escaped: 0,
            airborne: air,
            body_speed: 0.0,
            area_speed: [s; MPM_MAX_AREAS as usize],
            area_body_speed: [0.0; MPM_MAX_AREAS as usize],
        };
        c.report(9, &r(0.0, 0), 0.01);
        assert_eq!(
            c.speed(),
            100.0,
            "a report from before the region is ignored"
        );
        c.report(10, &r(0.001, 0), 0.01);
        c.report(11, &r(0.001, 1), 0.01);
        assert!(!c.settled(1), "a parcel in flight keeps it open");
        c.report(12, &r(0.001, 0), 0.01);
        c.report(13, &r(0.001, 0), 0.01);
        assert!(c.settled(2));
        let cfg = TerrainConfig::default();
        // The bound grows by g·dt per frame of readback lag.
        let a = c
            .substeps(&cfg, 14, 1.0 / 60.0, -9.81, 1.0, 200.0, 0.0)
            .unwrap();
        let b = c
            .substeps(&cfg, 400, 1.0 / 60.0, -9.81, 1.0, 200.0, 0.0)
            .unwrap();
        assert!(b.0 >= a.0);
    }

    fn kernels(ops: &TerrainOps) -> Vec<&'static str> {
        ops.cmds
            .iter()
            .map(|c| match *c {
                TerrainCmd::Dispatch { kernel, .. } | TerrainCmd::Indirect { kernel, .. } => kernel,
            })
            .collect()
    }

    fn absorbing() -> TerrainConfig {
        TerrainConfig {
            boundary: RegionBoundary::Absorbing,
            initial_stress: InitialStress::Geostatic,
            ..TerrainConfig::default()
        }
    }

    #[test]
    fn an_absorbing_region_sets_up_its_node_records_and_starts_geostatic_once() {
        let cfg = absorbing();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let region = RegionPlan::spanning(
            &cfg,
            glam::Vec3::splat(-1.0),
            glam::Vec3::splat(1.0),
            0.1,
            0,
            0,
        )
        .unwrap();
        let ops = p
            .activate(&region, 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        let a = *p.area_params(0).unwrap();
        let cells = [a.dims[0], a.dims[1], a.dims[2]];
        let records = TerrainPlan::boundary_nodes(cells);
        assert_eq!(
            ops.cmds.last(),
            Some(&TerrainCmd::Dispatch {
                kernel: "mpm_absorb_init",
                threads: records
            })
        );
        let b = a.boundary;
        assert_eq!(b[0], MPM_BOUNDARY_ABSORBING | MPM_BOUNDARY_GEOSTATIC);
        // The records follow the coupling slots in their binding, which holds them all.
        assert_eq!(b[1], 4 * cfg.max_coupled);
        assert_eq!(u64::from(b[2]), records);
        let coupled = p
            .buffer_sizes()
            .iter()
            .find(|(k, _)| *k == MPM_COUPLED_BINDING)
            .unwrap()
            .1;
        assert!(coupled >= 16 * (u64::from(b[1]) + records));
        // The first step sets the geostatic stress, before its substeps; later ones do not.
        let first = kernels(&p.step(2, 1.0e-4, -9.81).unwrap());
        assert_eq!(first.iter().filter(|k| **k == "mpm_geostatic").count(), 1);
        assert!(
            first.iter().position(|k| *k == "mpm_geostatic")
                < first.iter().position(|k| *k == "mpm_p2g")
        );
        let next = kernels(&p.coupled(2, 1.0e-4, -9.81, 3).unwrap());
        assert!(!next.contains(&"mpm_geostatic"));
        // A rigid, unstressed region does neither.
        let mut q = TerrainPlan::new(&TerrainConfig {
            boundary: RegionBoundary::Rigid,
            initial_stress: InitialStress::Unstressed,
            ..TerrainConfig::default()
        })
        .unwrap();
        q.set_material(Some(&crate::SandMaterial::default()))
            .unwrap();
        let ops = q
            .activate(&region, 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        assert!(!kernels(&ops).contains(&"mpm_absorb_init"));
        let b = q.area_params(0).unwrap().boundary;
        assert_eq!((b[0], b[2]), (0, 0));
        assert!(!kernels(&q.step(2, 1.0e-4, -9.81).unwrap()).contains(&"mpm_geostatic"));
    }

    #[test]
    fn the_absorbing_ground_fails_by_name_past_its_limits_and_without_stiffness() {
        let region = |cfg: &TerrainConfig| {
            RegionPlan::spanning(
                cfg,
                glam::Vec3::splat(-1.0),
                glam::Vec3::splat(1.0),
                0.1,
                0,
                0,
            )
            .unwrap()
        };
        let cfg = TerrainConfig {
            max_boundary_nodes: 100,
            ..absorbing()
        };
        let mut p = TerrainPlan::new(&cfg).unwrap();
        let err = p
            .activate(&region(&cfg), 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap_err();
        assert!(err.to_string().contains("max_boundary_nodes"), "{err}");
        // Its static traction is the geostatic stress.
        let bad = TerrainConfig {
            initial_stress: InitialStress::Unstressed,
            ..absorbing()
        };
        assert!(matches!(
            bad.validate(),
            Err(TerrainError::InvalidConfig(_))
        ));
        // A material with no stiffness has no wave speed for the dashpots.
        let cfg = absorbing();
        let mut p = TerrainPlan::new(&cfg).unwrap();
        p.activate(&region(&cfg), 0, grid(), glam::Vec3::ZERO, 1600.0, 0.6)
            .unwrap();
        let err = p.step(1, 1.0e-4, -9.81).unwrap_err();
        assert!(err.to_string().contains("stiffness"), "{err}");
        // A host-loaded region has no window: its bed is the field's surface.
        p.load(&region(&cfg), 10, Some(0), 0.5).unwrap();
        assert_eq!(p.area_params(0).unwrap().boundary, [0; 4]);
    }
}
