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
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterPrediction, ExplosionSource, ImpactSource, RegionPlan, SandMaterial,
    TerrainConfig, ZModel,
};
use engine_webgpu::TerrainStepReport;
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
        }
    }
}

/// The open region: on which heightfield shape, and how it is settling.
#[derive(Clone, Debug)]
struct ActiveRegion {
    shape: ShapeId,
    plan: RegionPlan,
    /// Speed bound basis: the source's peak, then each frame's report.
    speed: f32,
    /// The step whose report set `speed`.
    speed_step: u64,
    /// Consecutive settled frames.
    quiet: u32,
}

pub(crate) struct TerrainHost {
    def: TerrainDef,
    region: Option<ActiveRegion>,
    /// Heightfield shapes whose heights a step should copy back (after a bake).
    mirror_due: Vec<ShapeId>,
    /// Copies encoded, not yet read.
    mirror_in_flight: Vec<ShapeId>,
}

impl World {
    /// Turn on deformable terrain: size its GPU buffers by `def.config` and
    /// load the material. Heightfield shapes stay rigid until a region opens.
    pub fn enable_terrain(&mut self, def: &TerrainDef) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.terrain.as_ref().is_none_or(|t| t.region.is_none()),
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
            region: None,
            mirror_due: Vec::new(),
            mirror_in_flight: Vec::new(),
        });
        Ok(())
    }

    /// A km/s impact at `point` on heightfield shape `ground`: opens a region
    /// sized by π-scaling and starts the Z-model excavation flow in it. The
    /// crater is computed, not prescribed. Returns the scaling prediction.
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
        let plan = crater.region(&self.terrain_def()?.config, Vec3::from(point), margin, 0, 0)?;
        self.terrain_open_with(ground, plan, Some((&field, Vec3::from(point))))?;
        Ok(crater)
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
        let plan = crater.region(&self.terrain_def()?.config, Vec3::from(point), margin, 0, 0)?;
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
        let plan = RegionPlan::spanning(&config, Vec3::from(lo), Vec3::from(hi), dx, 0, 0)?;
        self.terrain_open_with(ground, plan, None)
    }

    /// Bake the open region now (it otherwise bakes once settled): its
    /// material goes back into the heightfield and the shape is rigid again.
    pub fn terrain_close(&mut self) -> anyhow::Result<()> {
        let host = self.terrain_host()?;
        let Some(region) = host.region.take() else {
            anyhow::bail!("no terrain region is open");
        };
        host.mirror_due.push(region.shape);
        self.sim.terrain_bake()?;
        self.dirty = true;
        Ok(())
    }

    /// The open region's heightfield shape and box, if one is open.
    pub fn terrain_region(&self) -> Option<(ShapeId, RegionPlan)> {
        let r = self.terrain.as_ref()?.region.as_ref()?;
        Some((r.shape, r.plan))
    }

    fn terrain_host(&mut self) -> anyhow::Result<&mut TerrainHost> {
        self.terrain
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (call enable_terrain)"))
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

    fn terrain_open_with(
        &mut self,
        ground: ShapeId,
        plan: RegionPlan,
        source: Option<(&ZModel, Vec3)>,
    ) -> anyhow::Result<()> {
        let s = self.live_shape(ground)?;
        let host = self.terrain_host()?;
        if host.region.is_some() {
            // One region at a time on the GPU: a second waits for the bake.
            return Err(engine_terrain::TerrainError::LimitExceeded {
                limit: "open terrain regions",
                requested: 2,
                allowed: 1,
            }
            .into());
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
        self.sim
            .terrain_activate(&plan, collider, &field, origin, density, bed_friction)?;
        let mut speed = 0.0;
        if let Some((model, point)) = source {
            self.sim.terrain_apply_source(model, point)?;
            speed = source_peak_speed(model);
        }
        let step = self.step;
        self.terrain_host()?.region = Some(ActiveRegion {
            shape: ground,
            plan,
            speed,
            speed_step: step,
            quiet: 0,
        });
        self.dirty = true;
        Ok(())
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
        let Some(region) = host.region.as_ref() else {
            return Ok(false);
        };
        if region.quiet >= host.def.settle_frames {
            self.terrain_close()?;
            return Ok(false);
        }
        let p = self.sim.params();
        let frame_dt = p.dt;
        // Each frame the report is older adds what gravity can add to a speed.
        let lag = (self.step - region.speed_step + 1) as f32;
        let bound =
            1.5 * region.speed + p.gravity_y.abs() * frame_dt * lag + host.def.speed_headroom;
        let (substeps, dt) = substeps_for(
            &host.def.config,
            frame_dt,
            region.plan.dx,
            host.def.material.p_wave_speed(),
            bound,
        )?;
        self.sim.terrain_encode_coupled(enc, substeps, dt)?;
        Ok(true)
    }

    /// The terrain copies this step's readback carries.
    pub(crate) fn terrain_extras(&mut self, counters: bool) -> Vec<(Extra, u64, u64, bool)> {
        let mut out = Vec::new();
        if counters {
            out.push((
                Extra::TerrainCounters,
                0,
                4 * engine_terrain::layout::MPM_COUNTER_WORDS as u64,
                false,
            ));
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
                true,
            ));
            host.mirror_in_flight.push(id);
        }
        out
    }

    /// The readback copies for [`Self::terrain_extras`]' list.
    pub(crate) fn terrain_extra_copies<'a>(
        sim: &'a engine_webgpu::WgpuSim,
        list: Vec<(Extra, u64, u64, bool)>,
    ) -> Vec<readback::Extra<'a, Extra>> {
        list.into_iter()
            .map(|(kind, offset, size, arena)| readback::Extra {
                kind,
                buffer: if arena {
                    sim.hull_arena_buffer()
                } else {
                    sim.terrain_counters_buffer()
                },
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
            if let Some(r) = host.region.as_mut().filter(|r| step >= r.speed_step) {
                r.speed = report.max_speed;
                r.speed_step = step;
                if report.max_speed < settle && report.airborne == 0 {
                    r.quiet += 1;
                } else {
                    r.quiet = 0;
                }
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
            host.region.as_ref().is_none_or(|r| r.shape != id),
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

/// The largest speed the excavation field starts: at its core's edge.
fn source_peak_speed(model: &ZModel) -> f32 {
    let core = model.core_radius as f32;
    let depth = model.flow_center_depth as f32;
    let deep = model.velocity_f32(Vec3::new(0.0, -depth, core)).length();
    deep.max(model.surface_speed(model.core_radius) as f32)
}

fn quat_is_identity(q: Quat) -> bool {
    q.abs_diff_eq(Quat::IDENTITY, 1e-6) || q.abs_diff_eq(-Quat::IDENTITY, 1e-6)
}
