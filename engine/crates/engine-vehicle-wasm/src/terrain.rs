//! Deformable terrain for a browser host (`engine-webgpu/terrain-host.mjs`): the
//! shared command plan (`engine_terrain::plan`) the native `WgpuSim` replays,
//! plus the open areas' lifecycle: π-scaled impact and explosion regions with
//! their Z-model source, each frame's coupled substeps from a speed bound the
//! GPU checks, the counters' check, and the bake once the region settles. The
//! JS host only executes the commands and copies the counters back.
//!
//! Commands come back as words: `[blocks, block…]`, each block
//! `[writes, cmds, write…, cmd…]` with a write `[binding, byte offset, n,
//! words × n]` and a command `[op, kernel, value]` (op 1: dispatch of `value`
//! threads, 64 per workgroup; op 2: indirect, workgroup counts at byte `value`
//! of the args binding). A block's writes go ahead of its commands' submission;
//! its commands are one compute pass. Kernel indices are into [`TerrainStep::kernels`].

use super::err;
use engine_terrain::layout::*;
use engine_terrain::plan::{
    alloc_bytes, source_peak_speed, FieldGrid, RegionClock, TerrainCmd, TerrainOps, TerrainPlan,
    TerrainStepReport, KERNELS,
};
use engine_terrain::surface::{
    SurfacePatch, SurfacePlan, DEFAULT_SURFACE_CELLS, SURFACE_BINDING, SURFACE_DEBRIS_ARGS_WORD,
    SURFACE_HEADER_WORDS, SURFACE_KERNELS, SURFACE_VALID_WORD, SURFACE_WINDOW_WORD,
};
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    CraterPrediction, CraterTarget, ExplosionSource, ImpactSource, InitialStress, RegionBoundary,
    RegionPlan, SandMaterial, TerrainConfig, TerrainError, TerrainProfile, ZModel,
};
use glam::{DVec3, Vec3};
use serde::Deserialize;
use serde_json::json;
use wasm_bindgen::prelude::*;

/// `TerrainConfig` fields a page may set (the rest keep their defaults).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ConfigJson {
    max_particles: Option<u32>,
    max_blocks: Option<u32>,
    max_parcels: Option<u32>,
    particles_per_cell_axis: Option<u32>,
    cells_per_radius: Option<f32>,
    min_cell_size: Option<f32>,
    courant: Option<f32>,
    max_substeps: Option<u32>,
    max_columns: Option<u32>,
    max_coupled: Option<u32>,
    max_field_samples: Option<u32>,
    bake_home_fade: Option<f32>,
    /// "rigid" or "absorbing" (`RegionBoundary`).
    boundary: Option<String>,
    /// "unstressed" or "geostatic" (`InitialStress`).
    initial_stress: Option<String>,
    max_boundary_nodes: Option<u32>,
    max_regions: Option<u32>,
    cell_size: Option<f32>,
    min_resolution: Option<f32>,
    /// Bed depth below the event in transient radii (`TerrainConfig::bed_depth`).
    bed_depth: Option<f32>,
    speed_growth: Option<f32>,
    /// "physical" (the default) or "game" (`TerrainProfile`): the settings
    /// the other fields then override.
    profile: Option<String>,
}

/// Dry sand's parameters (the default sand when absent).
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SandJson {
    density: Option<f32>,
    youngs_modulus: Option<f32>,
    poisson_ratio: Option<f32>,
    friction_angle_deg: Option<f32>,
}

/// The settings of `engine-world`'s `TerrainDef`, as JSON.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SettingsJson {
    #[serde(default)]
    config: ConfigJson,
    #[serde(default)]
    sand: SandJson,
    region_margin: Option<f64>,
    z: Option<f64>,
    speed_headroom: Option<f32>,
    settle_speed: Option<f32>,
    settle_frames: Option<u32>,
    /// The render-only surface's limit on window samples × layers
    /// (`engine_terrain::surface`).
    surface_cells: Option<u32>,
    /// Sand faster than a fall of this many layers is in flight (debris);
    /// 0: only what a gap separates (`SurfacePlan::with_ejecta_fall`).
    surface_ejecta_fall: Option<f32>,
}

/// An open area.
#[derive(Clone)]
struct Open {
    /// Its slot in the plan (`AreaInfo::slot`).
    slot: u32,
    plan: RegionPlan,
    clock: RegionClock,
    prediction: Option<CraterPrediction>,
    source_speed: f32,
}

/// The terrain pool of a browser world: its plan, material and open areas.
#[wasm_bindgen]
pub struct TerrainStep {
    plan: TerrainPlan,
    sand: SandMaterial,
    margin: f64,
    z: Option<f64>,
    headroom: f32,
    settle_speed: f32,
    settle_frames: u32,
    /// The open areas, by slot.
    areas: Vec<Open>,
    surface: SurfacePlan,
    /// Frames encoded (the step ids reports come back with).
    frame: u64,
    last: Option<TerrainStepReport>,
}

/// Op words of a block's command.
const OP_DISPATCH: u32 = 1;
const OP_INDIRECT: u32 = 2;

fn kernel_index(kernels: &[&str], name: &str) -> u32 {
    kernels
        .iter()
        .position(|k| *k == name)
        .expect("plan kernels are in their kernel list") as u32
}

/// `[blocks, block…]` for a sequence of the plan's ops.
fn encode(blocks: &[TerrainOps]) -> Vec<u32> {
    encode_in(blocks, KERNELS)
}

/// [`encode`] with kernel indices into `kernels`.
fn encode_in(blocks: &[TerrainOps], kernels: &[&str]) -> Vec<u32> {
    let mut out = vec![blocks.len() as u32];
    for ops in blocks {
        out.push(ops.writes.len() as u32);
        out.push(ops.cmds.len() as u32);
        for w in &ops.writes {
            out.extend([w.binding as u32, w.offset as u32, w.words.len() as u32]);
            out.extend_from_slice(&w.words);
        }
        for c in &ops.cmds {
            match *c {
                TerrainCmd::Dispatch { kernel, threads } => out.extend([
                    OP_DISPATCH,
                    kernel_index(kernels, kernel),
                    u32::try_from(threads).expect("a dispatch's threads fit u32"),
                ]),
                TerrainCmd::Indirect { kernel, offset } => {
                    out.extend([OP_INDIRECT, kernel_index(kernels, kernel), offset as u32])
                }
            }
        }
    }
    out
}

/// A heightfield `[nx, nz, dx, dz]` (samples as f32) and its frame's origin.
fn grid(field: &[f32]) -> Result<(FieldGrid, Vec3), JsError> {
    if field.len() != 7 {
        return Err(JsError::new(
            "field is [nx, nz, dx, dz, origin x, origin y, origin z]",
        ));
    }
    Ok((
        FieldGrid {
            nx: field[0] as u32,
            nz: field[1] as u32,
            dx: field[2],
            dz: field[3],
        },
        Vec3::new(field[4], field[5], field[6]),
    ))
}

fn point(p: &[f32]) -> Result<Vec3, JsError> {
    if p.len() != 3 {
        return Err(JsError::new("point is [x, y, z]"));
    }
    Ok(Vec3::from_slice(p))
}

fn prediction_json(c: &CraterPrediction) -> serde_json::Value {
    json!({
        "radius": c.radius, "depth": c.depth, "rim_radius": c.rim_radius,
        "final_rim_radius": c.final_rim_radius, "volume": c.volume,
        "formation_time": c.formation_time, "complex": c.complex, "pi_v": c.pi_v,
        "regime": format!("{:?}", c.regime),
    })
}

#[wasm_bindgen]
impl TerrainStep {
    /// The pool for `settings` (JSON: `config` with `TerrainConfig` fields,
    /// `sand` with density, youngs_modulus, poisson_ratio, friction_angle_deg,
    /// and `region_margin`, `z`, `speed_headroom`, `settle_speed`,
    /// `settle_frames` as in `engine-world`'s `TerrainDef`; absent fields take
    /// its defaults).
    #[wasm_bindgen(constructor)]
    pub fn new(settings: &str) -> Result<TerrainStep, JsError> {
        let s: SettingsJson = super::parse(settings)?;
        let c = s.config;
        let d = TerrainConfig::profile(match c.profile.as_deref() {
            None | Some("physical") => TerrainProfile::Physical,
            Some("game") => TerrainProfile::Game,
            Some(other) => {
                return Err(JsError::new(&format!(
                    "config.profile is \"physical\" or \"game\", not {other:?}"
                )))
            }
        });
        let boundary = match c.boundary.as_deref() {
            None => d.boundary,
            Some("rigid") => RegionBoundary::Rigid,
            Some("absorbing") => RegionBoundary::Absorbing,
            Some(other) => {
                return Err(JsError::new(&format!(
                    "config.boundary is \"rigid\" or \"absorbing\", not {other:?}"
                )))
            }
        };
        let initial_stress = match c.initial_stress.as_deref() {
            None => d.initial_stress,
            Some("unstressed") => InitialStress::Unstressed,
            Some("geostatic") => InitialStress::Geostatic,
            Some(other) => {
                return Err(JsError::new(&format!(
                    "config.initial_stress is \"unstressed\" or \"geostatic\", not {other:?}"
                )))
            }
        };
        let config = TerrainConfig {
            max_particles: c.max_particles.unwrap_or(d.max_particles),
            max_blocks: c.max_blocks.unwrap_or(d.max_blocks),
            max_parcels: c.max_parcels.unwrap_or(d.max_parcels),
            particles_per_cell_axis: c
                .particles_per_cell_axis
                .unwrap_or(d.particles_per_cell_axis),
            cells_per_radius: c.cells_per_radius.unwrap_or(d.cells_per_radius),
            min_cell_size: c.min_cell_size.unwrap_or(d.min_cell_size),
            courant: c.courant.unwrap_or(d.courant),
            max_substeps: c.max_substeps.unwrap_or(d.max_substeps),
            max_columns: c.max_columns.unwrap_or(d.max_columns),
            max_coupled: c.max_coupled.unwrap_or(d.max_coupled),
            max_field_samples: c.max_field_samples.unwrap_or(d.max_field_samples),
            bake_home_fade: c.bake_home_fade.unwrap_or(d.bake_home_fade),
            boundary,
            initial_stress,
            max_boundary_nodes: c.max_boundary_nodes.unwrap_or(d.max_boundary_nodes),
            max_regions: c.max_regions.unwrap_or(d.max_regions),
            cell_size: c.cell_size.unwrap_or(d.cell_size),
            min_resolution: c.min_resolution.unwrap_or(d.min_resolution),
            bed_depth: c.bed_depth.or(d.bed_depth),
            speed_growth: c.speed_growth.unwrap_or(d.speed_growth),
            ..d
        };
        let ds = SandMaterial::default();
        let sand = SandMaterial {
            density: s.sand.density.unwrap_or(ds.density),
            youngs_modulus: s.sand.youngs_modulus.unwrap_or(ds.youngs_modulus),
            poisson_ratio: s.sand.poisson_ratio.unwrap_or(ds.poisson_ratio),
            friction_angle: s
                .sand
                .friction_angle_deg
                .map_or(ds.friction_angle, f32::to_radians),
            ..ds
        };
        let mut plan = TerrainPlan::new(&config).map_err(err)?;
        let surface = SurfacePlan::new(&config, s.surface_cells.unwrap_or(DEFAULT_SURFACE_CELLS))
            .and_then(|p| p.with_ejecta_fall(s.surface_ejecta_fall.unwrap_or(1.0)))
            .map_err(err)?;
        plan.set_material(Some(&sand)).map_err(err)?;
        // engine-world's TerrainDef defaults and checks.
        let margin = s.region_margin.unwrap_or(2.0);
        let headroom = s.speed_headroom.unwrap_or(20.0);
        let settle_speed = s.settle_speed.unwrap_or(0.01);
        let settle_frames = s.settle_frames.unwrap_or(30);
        if !(margin.is_finite() && margin > 0.0) {
            return Err(JsError::new("region_margin must be positive"));
        }
        if !(headroom.is_finite() && headroom >= 0.0) {
            return Err(JsError::new(
                "speed_headroom must be finite and non-negative",
            ));
        }
        if !(settle_speed.is_finite() && settle_speed > 0.0 && settle_frames > 0) {
            return Err(JsError::new(
                "settle_speed must be positive and settle_frames at least 1",
            ));
        }
        Ok(TerrainStep {
            plan,
            sand,
            margin,
            z: s.z,
            headroom,
            settle_speed,
            settle_frames,
            areas: Vec::new(),
            surface,
            frame: 0,
            last: None,
        })
    }

    /// Kernel entry names (JSON array); command words index into it.
    pub fn kernels() -> String {
        serde_json::to_string(KERNELS).unwrap()
    }

    /// `[binding, bytes to allocate, bytes needed]` for every terrain binding.
    pub fn buffer_sizes(&self) -> Vec<f64> {
        self.plan
            .buffer_sizes()
            .iter()
            .flat_map(|&(b, bytes)| [b as f64, alloc_bytes(bytes) as f64, bytes as f64])
            .collect()
    }

    /// Bindings, counter words and record sizes (JSON object).
    pub fn constants() -> String {
        json!({
            "PARAMS": MPM_PARAMS_BINDING, "COUNTERS": MPM_COUNTERS_BINDING,
            "PARTICLES": MPM_PARTICLES_BINDING, "ARGS": MPM_ARGS_BINDING,
            "PARCELS": MPM_PARCELS_BINDING, "WATCH": MPM_WATCH_BINDING,
            "BINDING_END": MPM_BINDING_END,
            "COUNTER_WORDS": MPM_COUNTER_WORDS, "C_PARTICLES": MPM_C_PARTICLES,
            "C_AREA_SPEED": MPM_C_AREA_SPEED, "C_AREA_BODY": MPM_C_AREA_BODY,
            "MAX_AREAS": MPM_MAX_AREAS,
            "C_PARCELS": MPM_C_PARCELS, "C_FLAGS": MPM_C_FLAGS,
            "PARTICLE_BYTES": std::mem::size_of::<MpmParticle>(),
            "PARCEL_BYTES": std::mem::size_of::<MpmParcel>(),
            "PARTICLE_DEAD": MPM_PARTICLE_DEAD,
            "SURFACE": SURFACE_BINDING, "SURFACE_HEADER_WORDS": SURFACE_HEADER_WORDS,
            "SURFACE_VALID_WORD": SURFACE_VALID_WORD, "SURFACE_WINDOW_WORD": SURFACE_WINDOW_WORD,
            "SURFACE_DEBRIS_ARGS_WORD": SURFACE_DEBRIS_ARGS_WORD,
        })
        .to_string()
    }

    /// The material and limits in use (JSON).
    pub fn settings(&self) -> String {
        let c = self.plan.config();
        json!({
            "sand": {"density": self.sand.density, "youngs_modulus": self.sand.youngs_modulus,
                "poisson_ratio": self.sand.poisson_ratio,
                "friction_angle_deg": self.sand.friction_angle.to_degrees(),
                "p_wave_speed": self.sand.p_wave_speed()},
            "config": {"max_particles": c.max_particles, "max_blocks": c.max_blocks,
                "max_parcels": c.max_parcels, "particles_per_cell_axis": c.particles_per_cell_axis,
                "cells_per_radius": c.cells_per_radius, "courant": c.courant,
                "max_substeps": c.max_substeps, "max_columns": c.max_columns,
                "max_coupled": c.max_coupled, "max_field_samples": c.max_field_samples,
                "bake_home_fade": c.bake_home_fade,
                "boundary": match c.boundary {
                    RegionBoundary::Rigid => "rigid",
                    RegionBoundary::Absorbing => "absorbing",
                },
                "initial_stress": match c.initial_stress {
                    InitialStress::Unstressed => "unstressed",
                    InitialStress::Geostatic => "geostatic",
                },
                "max_boundary_nodes": c.max_boundary_nodes, "max_regions": c.max_regions,
                "cell_size": c.cell_size, "min_resolution": c.min_resolution,
                "bed_depth": c.bed_depth, "speed_growth": c.speed_growth},
            "region_margin": self.margin, "speed_headroom": self.headroom,
            "settle_speed": self.settle_speed, "settle_frames": self.settle_frames,
        })
        .to_string()
    }

    fn impact_source(
        &self,
        radius: f64,
        density: f64,
        speed: f64,
    ) -> Result<ImpactSource, TerrainError> {
        Ok(ImpactSource {
            impactor_radius: radius,
            impactor_density: density,
            // Vertical: the Z-model source is axisymmetric.
            velocity: DVec3::new(0.0, -speed, 0.0),
            target: CraterTarget::sand(&self.sand)?,
        })
    }

    /// The π-scaling prediction for a vertical impact (JSON), without opening
    /// a region: the crater, the impactor's mass and energy, the region's cell.
    pub fn predict_impact(
        &self,
        radius: f64,
        density: f64,
        speed: f64,
        gravity: f64,
    ) -> Result<String, JsError> {
        let src = self.impact_source(radius, density, speed).map_err(err)?;
        let crater = src.crater(gravity).map_err(err)?;
        let field = src.field(gravity, self.z_for(src.target)).map_err(err)?;
        let plan = crater
            .region(self.plan.config(), Vec3::ZERO, self.margin, 0, 0)
            .map_err(err)?;
        Ok(json!({
            "crater": prediction_json(&crater), "mass": src.mass(), "energy": src.kinetic_energy(),
            "dx": plan.dx, "blocks": plan.blocks.to_array(), "source_speed": source_peak_speed(&field),
        })
        .to_string())
    }

    /// The prediction for a buried charge of `kg` TNT at `depth` m (JSON).
    pub fn predict_explosion(&self, kg: f64, depth: f64, gravity: f64) -> Result<String, JsError> {
        let target = CraterTarget::sand(&self.sand).map_err(err)?;
        let ex = ExplosionSource::tnt(kg, depth, target);
        let crater = ex.crater(gravity).map_err(err)?;
        let field = ex.field(gravity, self.z_for(target)).map_err(err)?;
        let plan = crater
            .region(self.plan.config(), Vec3::ZERO, self.margin, 0, 0)
            .map_err(err)?;
        Ok(json!({
            "crater": prediction_json(&crater), "energy": ex.energy, "dx": plan.dx,
            "blocks": plan.blocks.to_array(), "source_speed": source_peak_speed(&field),
        })
        .to_string())
    }

    fn z_for(&self, target: CraterTarget) -> f64 {
        self.z.unwrap_or_else(|| point_source_z(target.ejecta.mu))
    }

    /// A vertical km/s impact at `at` (on the surface of heightfield collider
    /// `collider`, `field` = [nx, nz, dx, dz, origin x, y, z]) of an impactor
    /// of `radius` m and `density` kg/m³ at `speed` m/s: opens a region sized
    /// by π-scaling and starts the Z-model excavation flow in it (two blocks:
    /// activation, then the source). `World::terrain_impact`.
    #[allow(clippy::too_many_arguments)]
    pub fn impact(
        &mut self,
        collider: u32,
        field: &[f32],
        at: &[f32],
        radius: f64,
        density: f64,
        speed: f64,
        gravity: f64,
    ) -> Result<Vec<u32>, JsError> {
        let src = self.impact_source(radius, density, speed).map_err(err)?;
        let crater = src.crater(gravity).map_err(err)?;
        let model = src.field(gravity, self.z_for(src.target)).map_err(err)?;
        self.open_source(collider, field, at, crater, &model)
    }

    /// A buried charge of `kg` TNT at `depth` m under `at`; see [`Self::impact`].
    pub fn explode(
        &mut self,
        collider: u32,
        field: &[f32],
        at: &[f32],
        kg: f64,
        depth: f64,
        gravity: f64,
    ) -> Result<Vec<u32>, JsError> {
        let target = CraterTarget::sand(&self.sand).map_err(err)?;
        let ex = ExplosionSource::tnt(kg, depth, target);
        let crater = ex.crater(gravity).map_err(err)?;
        let model = ex.field(gravity, self.z_for(target)).map_err(err)?;
        self.open_source(collider, field, at, crater, &model)
    }

    fn open_source(
        &mut self,
        collider: u32,
        field: &[f32],
        at: &[f32],
        crater: CraterPrediction,
        model: &ZModel,
    ) -> Result<Vec<u32>, JsError> {
        let at = point(at)?;
        let dx = self
            .plan
            .event_cell(crater.asked_cell(self.plan.config()))
            .map_err(err)?;
        let plan = crater
            .region_with_cell(self.plan.config(), at, self.margin, dx)
            .map_err(err)?;
        let (slot, activate) = self.open(collider, field, plan, Some(crater))?;
        let source = self.plan.apply_source_to(slot, model, at).map_err(err)?;
        let speed = source_peak_speed(model);
        let frame = self.frame;
        let area = self.area_mut(slot);
        // Its velocity adds to the material's: the bound adds its peak.
        area.clock.kick(speed, frame);
        area.source_speed = area.source_speed.max(speed);
        let mut blocks = Vec::new();
        blocks.extend(activate);
        blocks.push(source);
        Ok(encode(&blocks))
    }

    /// A region over the box `[lo, hi]` with cells of `dx` and no source
    /// (heavy bodies move the sand). `World::terrain_open`.
    pub fn open_box(
        &mut self,
        collider: u32,
        field: &[f32],
        lo: &[f32],
        hi: &[f32],
        dx: f32,
    ) -> Result<Vec<u32>, JsError> {
        let dx = self.plan.event_cell(dx).map_err(err)?;
        let plan = RegionPlan::spanning(self.plan.config(), point(lo)?, point(hi)?, dx, 0, 0)
            .map_err(err)?;
        let (_, activate) = self.open(collider, field, plan, None)?;
        Ok(encode(&activate))
    }

    fn area_mut(&mut self, slot: u32) -> &mut Open {
        self.areas
            .iter_mut()
            .find(|a| a.slot == slot)
            .expect("the plan's area is tracked")
    }

    /// Open `plan` beside, into or around the open areas (`TerrainPlan::open`):
    /// the covering area and its activation (none if an area covered it).
    fn open(
        &mut self,
        collider: u32,
        field: &[f32],
        plan: RegionPlan,
        prediction: Option<CraterPrediction>,
    ) -> Result<(u32, Vec<TerrainOps>), JsError> {
        let (grid, origin) = grid(field)?;
        if self.areas.is_empty() {
            self.last = None;
        }
        let opened = self
            .plan
            .open(
                &plan,
                collider,
                grid,
                origin,
                self.sand.density,
                self.sand.friction_angle.tan(),
            )
            .map_err(err)?;
        let frame = self.frame;
        // The covering area's clock: a new area's, the merged areas'
        // combined, or the covering one's own.
        let mut clock = RegionClock::new(0.0, frame);
        let mut source_speed = 0.0f32;
        let mut pred = prediction;
        if opened.activated {
            let merged: Vec<&Open> = self
                .areas
                .iter()
                .filter(|a| opened.absorbed.contains(&a.slot))
                .collect();
            if !merged.is_empty() {
                let clocks: Vec<RegionClock> = merged.iter().map(|a| a.clock).collect();
                clock = RegionClock::merged(&clocks, frame);
                source_speed = merged.iter().map(|a| a.source_speed).fold(0.0, f32::max);
                pred = pred.or_else(|| merged.iter().find_map(|a| a.prediction));
            }
            self.areas.retain(|a| !opened.absorbed.contains(&a.slot));
        } else if let Some(at) = self.areas.iter().position(|a| a.slot == opened.area) {
            let old = self.areas.remove(at);
            clock = old.clock;
            source_speed = old.source_speed;
            pred = old.prediction.or(pred);
        }
        let plan = self
            .plan
            .areas()
            .into_iter()
            .find(|a| a.slot == opened.area)
            .map_or(plan, |a| a.plan);
        self.areas.push(Open {
            slot: opened.area,
            plan,
            clock,
            prediction: pred,
            source_speed,
        });
        self.areas.sort_by_key(|a| a.slot);
        // A grown area's old walls first (a block each), then the activation.
        let mut blocks = opened.pre;
        if opened.activated {
            blocks.push(opened.ops);
        }
        Ok((opened.area, blocks))
    }

    /// This frame's terrain work, ahead of the rigid step: `[kind, step,
    /// blocks…]`.
    /// - Kind 0: no area open (no blocks).
    /// - Kind 1: the coupled substeps of every open area (one block, encoded
    ///   into the frame's submission ahead of the rigid step, which must run
    ///   its impulse pass; copy the counters after the step and clear their
    ///   flags, then hand the copy to [`Self::report`] with `step`).
    /// - Kind 2: every open area settled and bakes (a block each, each its
    ///   own submission); no substeps.
    /// - Kind 3: some areas settled: their bakes (a block each, each its own
    ///   submission, first), then the others' coupled substeps as kind 1 (the
    ///   last block).
    pub fn frame(
        &mut self,
        frame_dt: f32,
        gravity: f32,
        colliders: u32,
    ) -> Result<Vec<u32>, JsError> {
        let step = self.frame;
        if self.areas.is_empty() {
            return Ok(vec![0, step as u32, 0]);
        }
        let settled: Vec<u32> = self
            .areas
            .iter()
            .filter(|a| a.clock.settled(self.settle_frames))
            .map(|a| a.slot)
            .collect();
        let mut blocks = Vec::new();
        for slot in &settled {
            blocks.push(self.close_area_ops(*slot)?);
        }
        if self.areas.is_empty() {
            let mut out = vec![2, step as u32];
            out.extend(encode(&blocks));
            return Ok(out);
        }
        // One schedule for the domain: the busiest area's bound.
        let mut sched = (1u32, frame_dt);
        for a in &self.areas {
            let (n, dt) = a
                .clock
                .substeps(
                    self.plan.config(),
                    step,
                    frame_dt,
                    gravity,
                    a.plan.dx,
                    self.sand.p_wave_speed(),
                    self.headroom,
                )
                .map_err(err)?;
            if n > sched.0 {
                sched = (n, dt);
            }
        }
        let ops = self
            .plan
            .coupled(sched.0, sched.1, gravity, colliders)
            .map_err(err)?;
        self.frame += 1;
        let kind = if blocks.is_empty() { 1 } else { 3 };
        blocks.push(ops);
        let mut out = vec![kind, step as u32];
        out.extend(encode(&blocks));
        Ok(out)
    }

    fn close_area_ops(&mut self, slot: u32) -> Result<TerrainOps, JsError> {
        let Some(at) = self.areas.iter().position(|a| a.slot == slot) else {
            return Err(JsError::new(&format!("terrain area {slot} is not open")));
        };
        self.areas.remove(at);
        self.plan.bake_area(slot).map_err(err)
    }

    /// Bake every open area now (each otherwise bakes once settled): a block
    /// each, each its own submission. `World::terrain_close`.
    pub fn close(&mut self) -> Result<Vec<u32>, JsError> {
        if self.areas.is_empty() {
            return Err(JsError::new("no terrain region is open"));
        }
        let slots: Vec<u32> = self.areas.iter().map(|a| a.slot).collect();
        let mut blocks = Vec::new();
        for slot in slots {
            blocks.push(self.close_area_ops(slot)?);
        }
        Ok(encode(&blocks))
    }

    /// Bake open area `slot` now (one block, its own submission); the
    /// others go on. `World::terrain_close_area`.
    pub fn close_area(&mut self, slot: u32) -> Result<Vec<u32>, JsError> {
        Ok(encode(&[self.close_area_ops(slot)?]))
    }

    /// The counters copied after frame `step` (`COUNTER_WORDS` words): the
    /// report as JSON, or the error a flag names. Tracks each area's settling.
    pub fn report(&mut self, step: u32, counters: &[u32]) -> Result<String, JsError> {
        let r = self
            .plan
            .check(counters)
            .map_err(|e| JsError::new(&format!("step {step}: terrain: {e}")))?;
        let settle = self.settle_speed;
        for open in &mut self.areas {
            open.clock
                .report(u64::from(step), &r.area(open.slot), settle);
        }
        self.last = Some(r);
        let areas: Vec<serde_json::Value> = self
            .areas
            .iter()
            .map(|a| {
                json!({"slot": a.slot, "max_speed": r.area_speed[a.slot as usize],
                    "body_speed": r.area_body_speed[a.slot as usize]})
            })
            .collect();
        Ok(json!({
            "blocks": r.blocks, "max_speed": r.max_speed, "particles": r.particles,
            "parcels": r.parcels, "deposited": r.deposited, "escaped": r.escaped,
            "airborne": r.airborne, "body_speed": r.body_speed, "areas": areas,
        })
        .to_string())
    }

    /// The flags word to clear after copying the counters: `[binding, byte offset, bytes]`.
    pub fn flags_clear(&self) -> Vec<u32> {
        let (b, o, n) = self.plan.flags_clear();
        vec![b as u32, o as u32, n as u32]
    }

    fn area_json(&self, r: &Open) -> serde_json::Value {
        let p = self.plan.area_params(r.slot).copied().unwrap_or_default();
        json!({
            "slot": r.slot,
            "origin": r.plan.origin.to_array(), "end": r.plan.end().to_array(), "dx": r.plan.dx,
            "blocks": r.plan.blocks.to_array(), "window": p.window, "bed": p.column[0],
            "spacing": p.column[2], "prediction": r.prediction.as_ref().map(prediction_json),
            "source_speed": r.source_speed, "speed_bound_basis": r.clock.speed(),
        })
    }

    /// The first open area (JSON, or `null`): its box, cell, the activation
    /// window and bed height, the prediction that sized it and the source's
    /// peak speed. [`Self::areas`] lists them all.
    pub fn region(&self) -> String {
        let Some(r) = self.areas.first() else {
            return "null".into();
        };
        self.area_json(r).to_string()
    }

    /// Every open area (JSON array of [`Self::region`]'s objects, by slot).
    pub fn areas(&self) -> String {
        serde_json::Value::Array(self.areas.iter().map(|a| self.area_json(a)).collect()).to_string()
    }

    /// The render-only surface pass's kernel entry names (JSON array); its
    /// command words index into it.
    pub fn surface_kernels() -> String {
        serde_json::to_string(SURFACE_KERNELS).unwrap()
    }

    /// Bytes of the surface buffer (binding `SURFACE`).
    pub fn surface_bytes(&self) -> f64 {
        alloc_bytes(self.surface.bytes()) as f64
    }

    /// The pass that rebuilds the live surface of the open region from its
    /// particles under `gravity` (m/s²; one block, after the frame's
    /// substeps; kernel indices into [`Self::surface_kernels`]), or no blocks
    /// without a region. With several areas open it shows the first (by
    /// slot); [`Self::surface_area`] picks one.
    pub fn surface(&self, gravity: f32) -> Result<Vec<u32>, JsError> {
        match self.areas.first() {
            None => Ok(vec![0]),
            Some(a) => self.surface_area(a.slot, gravity),
        }
    }

    /// [`Self::surface`] for open area `slot` (the surface is one window's).
    pub fn surface_area(&self, slot: u32, gravity: f32) -> Result<Vec<u32>, JsError> {
        Ok(encode_in(
            &[self
                .surface
                .frame_area(&self.plan, slot, gravity)
                .map_err(err)?],
            SURFACE_KERNELS,
        ))
    }

    /// Publish the live surface into a static heightfield collider of
    /// `nx × nz` samples (the field's spacing, at least two more than the
    /// window each way) so wheel rays and contacts meet the sand while a
    /// region is open (`SurfacePatch`).
    pub fn set_surface_patch(&mut self, collider: u32, nx: u32, nz: u32) {
        self.surface = self
            .surface
            .with_patch(Some(SurfacePatch { collider, nx, nz }));
    }

    /// After a bake: one block that sinks the patch (no blocks without one).
    pub fn surface_retire(&self) -> Vec<u32> {
        match self.surface.retire() {
            Some(ops) => encode_in(&[ops], SURFACE_KERNELS),
            None => vec![0],
        }
    }

    /// Frames encoded since the pool was created.
    pub fn frames(&self) -> u32 {
        self.frame as u32
    }
}
