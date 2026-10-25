//! Deformable terrain in a world (`engine/docs/deformable-terrain.md`): a
//! heightfield shape turns into MPM sand where an impact, an explosion or a
//! heavy body needs it to move, the sand and the rigid bodies step together
//! (explicit, staggered: every frame's MPM substeps run on the GPU ahead of
//! the rigid step, in the same submission), and once the region settles its
//! material is baked back into the heightfield. Terrain data never comes back
//! to the CPU on the step path: frames read the region's counters, and a bake
//! copies the new heights back only to refresh the host's mirror of the field.

use crate::readback;
use crate::{unsupported, Extra, Geometry, ShapeId, World};
use engine_terrain::plan::source_peak_speed;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    CraterPrediction, ExplosionSource, ImpactSource, RegionClock, RegionPlan, SandMaterial,
    TerrainConfig, ZModel,
};
use engine_webgpu::{TerrainStepReport, TerrainStrike, TerrainWatch};
use glam::{Quat, Vec3};

/// A world's terrain: limits, the material, and when a region closes.
#[derive(Clone, Debug)]
pub struct TerrainDef {
    pub config: TerrainConfig,
    pub material: SandMaterial,
    /// How far an impact or explosion region reaches past the predicted
    /// transient radius, in transient radii (at least to the final rim). Too
    /// small and the region's box limits the cavity; the crater validation
    /// measures the effect.
    pub region_margin: f64,
    /// Z-model exponent of the excavation field (`None`: 1/μ of the target's
    /// ejecta scaling, `point_source_z`).
    pub z: Option<f64>,
    /// Added to 1.5 × the last reported material speed to bound each frame's
    /// speed for its CFL substeps (m/s): covers bodies entering the region
    /// faster than the sand moves. A frame that exceeds its bound fails (the
    /// GPU checks every substep); nothing is clamped.
    pub speed_headroom: f32,
    /// A region bakes after its fastest particle has stayed below this speed
    /// (m/s), with no parcel in flight, for `settle_frames` frames. Baking
    /// stops whatever creep is left, so this is the region's rest tolerance.
    pub settle_speed: f32,
    pub settle_frames: u32,
    /// What counts as a heavy-body strike on a watched heightfield
    /// ([`World::terrain_watch_shape`]).
    pub strike: StrikeDef,
}

/// The look-ahead's thresholds and the region a strike opens.
#[derive(Clone, Debug, PartialEq)]
pub struct StrikeDef {
    /// A strike opens a region at this impact speed (m/s) ...
    pub min_speed: f32,
    /// ... and this kinetic energy (J) or more.
    pub min_energy: f32,
    /// Frames ahead the look-ahead sees, past the readback lag.
    pub horizon_frames: u32,
    /// Samples of each path over the horizon.
    pub samples: u32,
    /// Cells across the striking body's bounding diameter.
    pub cells_per_body: f32,
}

impl Default for StrikeDef {
    fn default() -> Self {
        Self {
            min_speed: 3.0,
            min_energy: 100.0,
            horizon_frames: 4,
            samples: 16,
            cells_per_body: 8.0,
        }
    }
}

impl Default for TerrainDef {
    fn default() -> Self {
        Self {
            config: TerrainConfig::default(),
            material: SandMaterial::default(),
            region_margin: 2.0,
            z: None,
            speed_headroom: 20.0,
            settle_speed: 0.01,
            settle_frames: 30,
            strike: StrikeDef::default(),
        }
    }
}

/// An open area: on which heightfield shape, its box, and how it is settling.
#[derive(Clone, Debug)]
struct ActiveArea {
    /// Its slot in the terrain pool (`engine_terrain::plan::AreaInfo::slot`).
    slot: u32,
    shape: ShapeId,
    plan: RegionPlan,
    /// Its frame schedule (substeps, settling): engine-terrain's, shared with
    /// the browser host.
    clock: RegionClock,
}

/// An open terrain area ([`World::terrain_areas`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainArea {
    /// Its slot: [`World::terrain_close_area`] takes it, and
    /// `TerrainStepReport::area_speed[slot]` is its speed.
    pub slot: u32,
    pub shape: ShapeId,
    /// Its box (cells of `plan.dx`).
    pub plan: RegionPlan,
}

pub(crate) struct TerrainHost {
    def: TerrainDef,
    /// The open areas, by slot.
    areas: Vec<ActiveArea>,
    /// Heightfield shapes whose heights a step should copy back (after a bake).
    mirror_due: Vec<ShapeId>,
    /// Copies encoded, not yet read.
    mirror_in_flight: Vec<ShapeId>,
    /// Heightfield shapes the look-ahead watches.
    watched: Vec<ShapeId>,
    /// A predicted strike to open a region for before the next step.
    pending_strike: Option<(ShapeId, TerrainStrike)>,
    /// Strikes predicted too late to open a region before them.
    missed_strikes: u64,
}

impl World {
    /// Turn on deformable terrain: size its GPU buffers by `def.config` and
    /// load the material. Heightfield shapes stay rigid until a region opens.
    pub fn enable_terrain(&mut self, def: &TerrainDef) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.terrain.as_ref().is_none_or(|t| t.areas.is_empty()),
            "a terrain region is open"
        );
        anyhow::ensure!(
            def.region_margin.is_finite() && def.region_margin > 0.0,
            "region_margin must be positive"
        );
        anyhow::ensure!(
            def.speed_headroom.is_finite() && def.speed_headroom >= 0.0,
            "speed_headroom must be finite and non-negative"
        );
        anyhow::ensure!(
            def.settle_speed.is_finite() && def.settle_speed > 0.0 && def.settle_frames > 0,
            "settle_speed must be positive and settle_frames at least 1"
        );
        self.sim.terrain_enable(&def.config)?;
        self.sim.terrain_set_material(Some(&def.material))?;
        self.terrain = Some(TerrainHost {
            def: def.clone(),
            areas: Vec::new(),
            mirror_due: Vec::new(),
            mirror_in_flight: Vec::new(),
            watched: Vec::new(),
            pending_strike: None,
            missed_strikes: 0,
        });
        Ok(())
    }

    /// A km/s impact at `point` on heightfield shape `ground`: opens a region
    /// sized by π-scaling and starts the Z-model excavation flow in it. The
    /// crater is computed, not prescribed. Returns the scaling prediction.
    ///
    /// Several may be under way at once. Far from the open areas it opens an
    /// area of its own; near them (inside a crater still forming, on its rim,
    /// partly outside it) they grow into one box with it and its field adds to
    /// the material's motion there (the sources superpose). Every area shares
    /// one grid: the impact runs at the open grid's cells (or
    /// `TerrainConfig::cell_size`), and fails with `TerrainError::CellTooCoarse`
    /// if those resolve it coarser than `TerrainConfig::min_resolution` allows.
    pub fn terrain_impact(
        &mut self,
        ground: ShapeId,
        impact: &ImpactSource,
        point: [f32; 3],
    ) -> anyhow::Result<CraterPrediction> {
        let g = self.terrain_gravity()?;
        let (margin, z) = self.terrain_source_settings(impact.target.ejecta.mu)?;
        let crater = impact.crater(g)?;
        let field = impact.field(g, z)?;
        let plan = self.terrain_event_region(&crater, point, margin)?;
        self.terrain_open_with(ground, plan, Some((&field, Vec3::from(point))))?;
        Ok(crater)
    }

    /// The region of an event's crater on the terrain's grid.
    fn terrain_event_region(
        &self,
        crater: &CraterPrediction,
        point: [f32; 3],
        margin: f64,
    ) -> anyhow::Result<RegionPlan> {
        let config = self.terrain_def()?.config;
        let dx = self.terrain_cell(crater.asked_cell(&config))?;
        Ok(crater.region_with_cell(&config, Vec3::from(point), margin, dx)?)
    }

    /// The cell an event asking for `asked` runs at (`TerrainPlan::event_cell`).
    fn terrain_cell(&self, asked: f32) -> anyhow::Result<f32> {
        let plan = self
            .sim
            .terrain_plan_ref()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call enable_terrain)"))?;
        Ok(plan.event_cell(asked)?)
    }

    /// A buried charge under `point` (on the surface of `ground`); see
    /// [`Self::terrain_impact`].
    pub fn terrain_explode(
        &mut self,
        ground: ShapeId,
        charge: &ExplosionSource,
        point: [f32; 3],
    ) -> anyhow::Result<CraterPrediction> {
        let g = self.terrain_gravity()?;
        let (margin, z) = self.terrain_source_settings(charge.target.ejecta.mu)?;
        let crater = charge.crater(g)?;
        let field = charge.field(g, z)?;
        let plan = self.terrain_event_region(&crater, point, margin)?;
        self.terrain_open_with(ground, plan, Some((&field, Vec3::from(point))))?;
        Ok(crater)
    }

    /// Open a region over the box `[lo, hi]` with cells of `dx`, with no
    /// source: heavy bodies landing or driving on it move the sand.
    pub fn terrain_open(
        &mut self,
        ground: ShapeId,
        lo: [f32; 3],
        hi: [f32; 3],
        dx: f32,
    ) -> anyhow::Result<()> {
        let config = self.terrain_def()?.config;
        let dx = self.terrain_cell(dx)?;
        let plan = RegionPlan::spanning(&config, Vec3::from(lo), Vec3::from(hi), dx, 0, 0)?;
        self.terrain_open_with(ground, plan, None).map(|_| ())
    }

    /// Bake every open area now (each otherwise bakes once settled): their
    /// material goes back into the heightfield and the shape is rigid again.
    pub fn terrain_close(&mut self) -> anyhow::Result<()> {
        let host = self.terrain_host()?;
        anyhow::ensure!(!host.areas.is_empty(), "no terrain region is open");
        for a in std::mem::take(&mut host.areas) {
            if !host.mirror_due.contains(&a.shape) {
                host.mirror_due.push(a.shape);
            }
        }
        self.sim.terrain_bake()?;
        self.dirty = true;
        Ok(())
    }

    /// Bake open area `slot` now; the others go on.
    pub fn terrain_close_area(&mut self, slot: u32) -> anyhow::Result<()> {
        let host = self.terrain_host()?;
        let Some(at) = host.areas.iter().position(|a| a.slot == slot) else {
            anyhow::bail!("terrain area {slot} is not open");
        };
        let a = host.areas.remove(at);
        if !host.mirror_due.contains(&a.shape) {
            host.mirror_due.push(a.shape);
        }
        self.sim.terrain_bake_area(slot)?;
        self.dirty = true;
        Ok(())
    }

    /// An open area's heightfield shape and box (the first, by slot), if any
    /// is open.
    pub fn terrain_region(&self) -> Option<(ShapeId, RegionPlan)> {
        let r = self.terrain.as_ref()?.areas.first()?;
        Some((r.shape, r.plan))
    }

    /// The open areas, by slot.
    pub fn terrain_areas(&self) -> Vec<TerrainArea> {
        self.terrain.as_ref().map_or_else(Vec::new, |t| {
            t.areas
                .iter()
                .map(|a| TerrainArea {
                    slot: a.slot,
                    shape: a.shape,
                    plan: a.plan,
                })
                .collect()
        })
    }

    fn terrain_host(&mut self) -> anyhow::Result<&mut TerrainHost> {
        self.terrain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call enable_terrain)"))
    }

    /// The settings `enable_terrain` was given, if terrain is on.
    pub fn terrain_settings(&self) -> Option<&TerrainDef> {
        self.terrain.as_ref().map(|t| &t.def)
    }

    fn terrain_def(&self) -> anyhow::Result<&TerrainDef> {
        self.terrain
            .as_ref()
            .map(|t| &t.def)
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call enable_terrain)"))
    }

    fn terrain_gravity(&self) -> anyhow::Result<f64> {
        let g = -f64::from(self.sim.params().gravity_y);
        anyhow::ensure!(g > 0.0, "terrain sources need downward gravity");
        Ok(g)
    }

    fn terrain_source_settings(&self, mu: f64) -> anyhow::Result<(f64, f64)> {
        let def = self.terrain_def()?;
        Ok((
            def.region_margin,
            def.z.unwrap_or_else(|| point_source_z(mu)),
        ))
    }

    /// Open `plan` on `ground` (beside, into or around the open areas) and
    /// start `source` there. Returns the area that covers it.
    fn terrain_open_with(
        &mut self,
        ground: ShapeId,
        plan: RegionPlan,
        source: Option<(&ZModel, Vec3)>,
    ) -> anyhow::Result<u32> {
        let s = self.live_shape(ground)?;
        let host = self.terrain_host()?;
        if let Some(a) = host.areas.first() {
            anyhow::ensure!(
                a.shape == ground,
                "terrain areas are open on heightfield shape {}: one heightfield at a time",
                a.shape.to_bits()
            );
        }
        let density = host.def.material.density;
        let bed_friction = host.def.material.friction_angle.tan();
        let Geometry::HeightField { field, transform } = &self.shapes[s].geometry else {
            anyhow::bail!("shape {} is not a heightfield", ground.to_bits());
        };
        let body = &self.bodies[self.shapes[s].body as usize];
        let at = body.origin;
        if !(quat_is_identity(at.q()) && quat_is_identity(transform.q())) {
            return unsupported("terrain regions on a rotated heightfield");
        }
        // Statics have the identity solver frame: world space is solver space.
        let origin = at.p() + transform.p();
        let collider = self.shapes[s].collider;
        let field = field.clone();
        let opened =
            self.sim
                .terrain_open_area(&plan, collider, &field, origin, density, bed_friction)?;
        let step = self.step;
        let host = self.terrain_host()?;
        // The covering area's clock: a new area's, the merged areas' combined,
        // or the covering one's own.
        let mut clock = RegionClock::new(0.0, step);
        if opened.activated {
            let merged: Vec<RegionClock> = host
                .areas
                .iter()
                .filter(|a| opened.absorbed.contains(&a.slot))
                .map(|a| a.clock)
                .collect();
            if !merged.is_empty() {
                clock = RegionClock::merged(&merged, step);
            }
            host.areas.retain(|a| !opened.absorbed.contains(&a.slot));
        } else if let Some(at) = host.areas.iter().position(|a| a.slot == opened.area) {
            clock = host.areas.remove(at).clock;
        }
        let area_plan = self
            .sim
            .terrain_areas()
            .into_iter()
            .find(|a| a.slot == opened.area)
            .map_or(plan, |a| a.plan);
        if let Some((model, point)) = source {
            self.sim
                .terrain_apply_source_to(opened.area, model, point)?;
            // Its velocity adds to the material's: the bound adds its peak.
            clock.kick(source_peak_speed(model), step);
        }
        let host = self.terrain_host()?;
        host.areas.push(ActiveArea {
            slot: opened.area,
            shape: ground,
            plan: area_plan,
            clock,
        });
        host.areas.sort_by_key(|a| a.slot);
        self.dirty = true;
        Ok(opened.area)
    }

    /// Before the rigid step: bake a settled region, or encode this frame's
    /// coupled MPM substeps. Returns whether the counters should be read back.
    pub(crate) fn terrain_encode(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<bool> {
        let Some(host) = self.terrain.as_ref() else {
            return Ok(false);
        };
        // Settled areas bake (each its own submission); the rest step on.
        let settled: Vec<u32> = host
            .areas
            .iter()
            .filter(|a| a.clock.settled(host.def.settle_frames))
            .map(|a| a.slot)
            .collect();
        for slot in settled {
            self.terrain_close_area(slot)?;
        }
        let host = self.terrain.as_ref().expect("terrain enabled");
        if host.areas.is_empty() {
            return Ok(false);
        }
        // One schedule for the domain: the busiest area's bound.
        let p = self.sim.params();
        let mut frame = (1u32, p.dt);
        for a in &host.areas {
            let (n, dt) = a.clock.substeps(
                &host.def.config,
                self.step,
                p.dt,
                p.gravity_y,
                a.plan.dx,
                host.def.material.p_wave_speed(),
                host.def.speed_headroom,
            )?;
            if n > frame.0 {
                frame = (n, dt);
            }
        }
        self.sim.terrain_encode_coupled(enc, frame.0, frame.1)?;
        Ok(true)
    }

    /// The terrain copies this step's readback carries.
    pub(crate) fn terrain_extras(
        &mut self,
        counters: bool,
        watched: bool,
    ) -> Vec<(Extra, u64, u64)> {
        let mut out = Vec::new();
        if counters {
            out.push((
                Extra::TerrainCounters,
                0,
                4 * engine_terrain::layout::MPM_COUNTER_WORDS as u64,
            ));
        }
        if watched {
            let (_, offset, size) = self.sim.terrain_watch_buffer();
            out.push((Extra::TerrainWatch, offset, size));
        }
        let Some(host) = self.terrain.as_mut() else {
            return out;
        };
        for id in std::mem::take(&mut host.mirror_due) {
            let Some(s) = self
                .shapes
                .get(id.index() as usize)
                .filter(|s| s.alive && s.generation == id.generation())
            else {
                continue;
            };
            let Geometry::HeightField { field, .. } = &s.geometry else {
                continue;
            };
            let base = engine_schema::convex::convex_params_base(
                self.sim.collider_record(s.collider).params,
            );
            let (nx, nz) = field.samples();
            // The arena counts 16-byte records; a readback copy counts bytes.
            out.push((
                Extra::TerrainHeights(id),
                u64::from(base + 2) * 16,
                u64::from(nx * nz) * 4,
            ));
            host.mirror_in_flight.push(id);
        }
        out
    }

    /// The readback copies for [`Self::terrain_extras`]' list.
    pub(crate) fn terrain_extra_copies<'a>(
        sim: &'a engine_webgpu::WgpuSim,
        list: Vec<(Extra, u64, u64)>,
    ) -> Vec<readback::Extra<'a, Extra>> {
        list.into_iter()
            .map(|(kind, offset, size)| readback::Extra {
                buffer: match kind {
                    Extra::TerrainHeights(_) => sim.hull_arena_buffer(),
                    Extra::TerrainWatch => sim.terrain_watch_buffer().0,
                    _ => sim.terrain_counters_buffer(),
                },
                kind,
                offset,
                size,
            })
            .collect()
    }

    /// A frame's terrain counters: errors on any limit or CFL flag, and
    /// tracks the region's settling.
    pub(crate) fn terrain_counters(
        &mut self,
        step: u64,
        bytes: &[u8],
    ) -> anyhow::Result<TerrainStepReport> {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let report = self
            .sim
            .terrain_check(&words)
            .map_err(|e| anyhow::anyhow!("step {step}: terrain: {e}"))?;
        if let Some(host) = self.terrain.as_mut() {
            let settle = host.def.settle_speed;
            for a in &mut host.areas {
                a.clock.report(step, &report.area(a.slot), settle);
            }
        }
        Ok(report)
    }

    /// A baked heightfield's samples: the host mirror follows the GPU (unless
    /// an edit already refreshed it, after which this copy is stale).
    pub(crate) fn terrain_heights(&mut self, id: ShapeId, bytes: &[u8]) {
        let Some(host) = self.terrain.as_mut() else {
            return;
        };
        let Some(at) = host.mirror_in_flight.iter().position(|&m| m == id) else {
            return;
        };
        host.mirror_in_flight.swap_remove(at);
        let heights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        self.set_mirror(id, &heights);
    }

    fn set_mirror(&mut self, id: ShapeId, heights: &[f32]) {
        let Some(s) = self
            .shapes
            .get_mut(id.index() as usize)
            .filter(|s| s.alive && s.generation == id.generation())
        else {
            return;
        };
        let Geometry::HeightField { field, .. } = &mut s.geometry else {
            return;
        };
        let (nx, nz) = field.samples();
        // Same dimensions, finite values: only a corrupt copy could fail.
        field
            .set_region(0, 0, nx, nz, heights)
            .expect("baked heights match the field");
    }

    /// Before a host edit of heightfield shape `id`: refused while a region is
    /// open on it; after a bake whose copy has not arrived, the mirror is read
    /// now (blocking: the edit would otherwise write stale heights back).
    pub(crate) fn terrain_before_edit(&mut self, id: ShapeId) -> anyhow::Result<()> {
        let Some(host) = self.terrain.as_mut() else {
            return Ok(());
        };
        anyhow::ensure!(
            host.areas.iter().all(|a| a.shape != id),
            "shape {} has an open terrain region",
            id.to_bits()
        );
        let due = host.mirror_due.contains(&id) || host.mirror_in_flight.contains(&id);
        if !due {
            return Ok(());
        }
        host.mirror_due.retain(|&m| m != id);
        host.mirror_in_flight.retain(|&m| m != id);
        let heights = self.read_height_field(id)?;
        self.set_mirror(id, &heights);
        Ok(())
    }

    /// Watch (or stop watching) heightfield shape `ground` for heavy-body
    /// strikes: after each step the GPU predicts every awake dynamic body's
    /// ballistic path, and a strike past `TerrainDef::strike`'s thresholds
    /// opens a region around it before the body arrives (forward only; hit
    /// events come after the rigid step has already stopped the body). At most
    /// 4 shapes, axis-aligned like every terrain region.
    pub fn terrain_watch_shape(&mut self, ground: ShapeId, watch: bool) -> anyhow::Result<()> {
        let s = self.live_shape(ground)?;
        anyhow::ensure!(
            matches!(self.shapes[s].geometry, Geometry::HeightField { .. }),
            "shape {} is not a heightfield",
            ground.to_bits()
        );
        let host = self.terrain_host()?;
        host.watched.retain(|&w| w != ground);
        if watch {
            host.watched.push(ground);
        }
        self.terrain_update_watch()
    }

    /// Strikes the look-ahead saw too late to open a region before (each a
    /// body that met rigid ground instead).
    pub fn terrain_missed_strikes(&self) -> u64 {
        self.terrain.as_ref().map_or(0, |t| t.missed_strikes)
    }

    fn terrain_update_watch(&mut self) -> anyhow::Result<()> {
        let dt = self.sim.params().dt;
        let lag = self.readback.frames() as f32;
        let host = self.terrain.as_mut().expect("terrain enabled");
        host.watched.retain(|id| {
            self.shapes
                .get(id.index() as usize)
                .is_some_and(|s| s.alive && s.generation == id.generation())
        });
        let fields = host
            .watched
            .iter()
            .map(|id| self.shapes[id.index() as usize].collider)
            .collect();
        let d = &host.def.strike;
        let watch = TerrainWatch {
            fields,
            // A pick reaches the host up to `lag` frames later; the region
            // opens before the step after that.
            horizon: (lag + d.horizon_frames as f32) * dt,
            samples: d.samples,
            min_speed: d.min_speed,
            min_energy: d.min_energy,
            min_time: dt,
        };
        self.sim.terrain_set_watch(&watch)
    }

    /// Whether this step should run the look-ahead (open areas or not: a
    /// strike beside them opens another, one into them grows or holds them).
    pub(crate) fn terrain_watching(&self) -> bool {
        self.sim.terrain_watch_enabled()
            && self
                .terrain
                .as_ref()
                .is_some_and(|t| t.pending_strike.is_none())
    }

    /// A frame's look-ahead pick (predicted at the end of `step`).
    pub(crate) fn terrain_strike(&mut self, step: u64, bytes: &[u8]) {
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let Some(strike) = engine_webgpu::WgpuSim::decode_terrain_strike(&words) else {
            return;
        };
        let dt = self.sim.params().dt;
        let shape = self.collider_shape.get(strike.field as usize).copied();
        let Some(host) = self.terrain.as_mut() else {
            return;
        };
        if host.pending_strike.is_some() {
            return;
        }
        // Steps already submitted since the prediction run before the region
        // can exist: the strike must come after them.
        let ahead = (self.step - 1 - step) as f32 * dt;
        let Some(s) = shape else {
            return;
        };
        let id = ShapeId::new(s, self.shapes[s as usize].generation);
        if strike.time <= ahead {
            host.missed_strikes += 1;
            return;
        }
        host.pending_strike = Some((id, strike));
    }

    /// Before a step: open the region a predicted strike asks for.
    pub(crate) fn terrain_open_strike(&mut self) -> anyhow::Result<()> {
        let Some(host) = self.terrain.as_mut() else {
            return Ok(());
        };
        let Some((ground, strike)) = host.pending_strike.take() else {
            return Ok(());
        };
        if host.areas.first().is_some_and(|a| a.shape != ground) {
            // Areas are open on another heightfield: one at a time.
            host.missed_strikes += 1;
            return Ok(());
        }
        let def = host.def.clone();
        let g = self.terrain_gravity()?;
        let s = self.live_shape(ground)?;
        let Geometry::HeightField { field, transform } = &self.shapes[s].geometry else {
            return Ok(());
        };
        let origin = self.bodies[self.shapes[s].body as usize].origin.p() + transform.p();
        let p = Vec3::from(strike.point);
        let surface = field
            .surface(p.x - origin.x, p.z - origin.z)
            .map_or(p.y - strike.radius, |(h, _)| h + origin.y);
        let size = strike_region(&def, g, &strike);
        let lo = Vec3::new(
            p.x - size.half_width,
            surface - size.depth,
            p.z - size.half_width,
        );
        let hi = Vec3::new(
            p.x + size.half_width,
            surface + size.above,
            p.z + size.half_width,
        );
        let dx = self.terrain_cell(size.dx)?;
        let plan = RegionPlan::spanning(&def.config, lo, hi, dx, 0, 0)?;
        let slot = self.terrain_open_with(ground, plan, None)?;
        // Faster than the sand: the frame's speed bound covers the body's.
        let dt = self.sim.params().dt;
        let step = self.step;
        if let Some(r) = self
            .terrain
            .as_mut()
            .and_then(|t| t.areas.iter_mut().find(|a| a.slot == slot))
        {
            r.clock.raise(Vec3::from(strike.velocity).length(), step);
            r.clock
                .hold_until(step + (strike.time / dt).ceil() as u64 + 1);
        }
        Ok(())
    }

    /// A heightfield shape's samples as the GPU holds them now (blocking: it
    /// waits for every step submitted). For tools and tests; frames keep the
    /// shape's geometry current after a bake without this.
    pub fn read_height_field(&self, id: ShapeId) -> anyhow::Result<Vec<f32>> {
        let s = self.live_shape(id)?;
        let Geometry::HeightField { field, .. } = &self.shapes[s].geometry else {
            anyhow::bail!("shape {} is not a heightfield", id.to_bits());
        };
        let (nx, nz) = field.samples();
        let base = engine_schema::convex::convex_params_base(
            self.sim.collider_record(self.shapes[s].collider).params,
        );
        // The arena counts 16-byte records; read_words counts words.
        let words = self
            .sim
            .read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
        Ok(words.into_iter().map(f32::from_bits).collect())
    }
}

/// A strike's region, from the scaling of spheres dropped into dry sand
/// (Uehara et al. 2003, PRL 90, 194301): penetration depth
/// d = (0.14/μ)·(ρ_b/ρ_g)^½·D^⅔·H^⅓ and crater diameter
/// D_c = 0.90·(ρ_b/ρ_g)^¼·D^¾·H^¼, with H the drop height plus d (H = v²/2g
/// for an impact at v), μ = tan φ, D the body's bounding diameter and ρ_b its
/// mass over that sphere. The box reaches `region_margin` past both.
struct StrikeRegion {
    half_width: f32,
    depth: f32,
    above: f32,
    dx: f32,
}

fn strike_region(def: &TerrainDef, g: f64, strike: &TerrainStrike) -> StrikeRegion {
    let d_b = 2.0 * f64::from(strike.radius);
    let rho_b = f64::from(strike.mass) / (std::f64::consts::PI / 6.0 * d_b.powi(3));
    let ratio = rho_b / f64::from(def.material.density);
    let mu = f64::from(def.material.friction_angle).tan();
    let v = Vec3::from(strike.velocity).as_dvec3().length();
    let drop = v * v / (2.0 * g);
    let mut d = 0.0;
    for _ in 0..8 {
        d = 0.14 / mu * ratio.sqrt() * d_b.powf(2.0 / 3.0) * (drop + d).cbrt();
    }
    let crater = 0.90 * ratio.powf(0.25) * d_b.powf(0.75) * (drop + d).powf(0.25);
    let m = 1.0 + def.region_margin;
    StrikeRegion {
        half_width: (m * (0.5 * crater).max(d_b)) as f32,
        depth: (m * (d + d_b)) as f32,
        above: (m * d_b) as f32,
        dx: ((d_b / f64::from(def.strike.cells_per_body)) as f32).max(def.config.min_cell_size),
    }
}

fn quat_is_identity(q: Quat) -> bool {
    q.abs_diff_eq(Quat::IDENTITY, 1e-6) || q.abs_diff_eq(-Quat::IDENTITY, 1e-6)
}
