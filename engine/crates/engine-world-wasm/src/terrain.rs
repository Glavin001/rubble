//! Deformable terrain (`World::enable_terrain`, `terrain_impact`, ...): the
//! settings from a plain object, events, the open areas, predictions without a
//! world (to size a heightfield before the world exists), and the terrain's GPU
//! buffers for renderers that draw the sand, parcels and live surface in place.

use crate::defs::Fields;
use crate::{bits, err, res, WebWorld};
use engine_terrain::layout::*;
use engine_terrain::surface::{
    SURFACE_BINDING, SURFACE_DEBRIS_ARGS_WORD, SURFACE_HEADER_WORDS, SURFACE_VALID_WORD,
    SURFACE_VIEW_OPEN, SURFACE_VIEW_WINDOW, SURFACE_VIEW_WORD, SURFACE_VIEW_WORDS,
};
use rubble::*;
use wasm_bindgen::prelude::*;

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

fn profile(s: &str) -> Result<TerrainProfile, JsValue> {
    match s {
        "physical" => Ok(TerrainProfile::Physical),
        "game" => Ok(TerrainProfile::Game),
        _ => Err(err(format!("TerrainConfig.profile: \"physical\" or \"game\", not {s:?}"))),
    }
}

/// `TerrainConfig` fields over `profile`'s settings (the physical profile, the
/// default, when absent).
fn config(mut f: Fields) -> Result<TerrainConfig, JsValue> {
    let p = f.string("profile")?.map(|s| profile(&s)).transpose()?;
    let mut c = TerrainConfig::profile(p.unwrap_or_default());
    f.u32("maxRegions", &mut c.max_regions)?;
    f.u32("maxParticles", &mut c.max_particles)?;
    f.u32("maxBlocks", &mut c.max_blocks)?;
    f.u32("maxParcels", &mut c.max_parcels)?;
    f.u32("particlesPerCellAxis", &mut c.particles_per_cell_axis)?;
    f.f32("cellsPerRadius", &mut c.cells_per_radius)?;
    f.f32("minCellSize", &mut c.min_cell_size)?;
    f.f32("cellSize", &mut c.cell_size)?;
    f.f32("minResolution", &mut c.min_resolution)?;
    f.f32("courant", &mut c.courant)?;
    f.u32("maxSubsteps", &mut c.max_substeps)?;
    f.u32("maxColumns", &mut c.max_columns)?;
    f.u32("maxCoupled", &mut c.max_coupled)?;
    f.u32("maxWheelLoads", &mut c.max_wheel_loads)?;
    f.u32("maxFieldSamples", &mut c.max_field_samples)?;
    f.f32("bakeHomeFade", &mut c.bake_home_fade)?;
    if let Some(s) = f.string("boundary")? {
        c.boundary = match s.as_str() {
            "rigid" => RegionBoundary::Rigid,
            "absorbing" => RegionBoundary::Absorbing,
            _ => return Err(err(format!("TerrainConfig.boundary: unknown {s}"))),
        };
    }
    if let Some(s) = f.string("initialStress")? {
        c.initial_stress = match s.as_str() {
            "unstressed" => InitialStress::Unstressed,
            "geostatic" => InitialStress::Geostatic,
            _ => return Err(err(format!("TerrainConfig.initialStress: unknown {s}"))),
        };
    }
    f.u32("maxBoundaryNodes", &mut c.max_boundary_nodes)?;
    if let Some(v) = f.take("bedDepth") {
        c.bed_depth = Some(v.as_f64().ok_or_else(|| err("TerrainConfig.bedDepth: expected a number"))? as f32);
    }
    f.f32("speedGrowth", &mut c.speed_growth)?;
    f.bool("areaClocks", &mut c.area_clocks)?;
    f.f32("sleepSpeed", &mut c.sleep_speed)?;
    f.u32("sleepAfter", &mut c.sleep_after)?;
    f.finish()?;
    Ok(c)
}

/// A `TerrainDef` (camelCase, every field optional): `{ config: { profile,
/// maxParticles, ... }, material: { density, youngsModulus, poissonRatio,
/// frictionAngle (rad), cohesion }, regionMargin, z, speedHeadroom,
/// settleSpeed, settleFrames, settleRule ('strict' | 'local'), surface: {
/// cells, ejectaFall, patchSamples, maxDebris } (absent: off), strike: {
/// minSpeed, minEnergy, horizonFrames, samples, cellsPerBody } }`.
pub fn terrain_def(v: JsValue) -> Result<TerrainDef, JsValue> {
    let mut f = Fields::of("TerrainDef", v)?;
    let mut d = TerrainDef::default();
    if let Some(c) = f.object("config")? {
        d.config = config(c)?;
    }
    if let Some(mut m) = f.object("material")? {
        let s = &mut d.material;
        m.f32("density", &mut s.density)?;
        m.f32("youngsModulus", &mut s.youngs_modulus)?;
        m.f32("poissonRatio", &mut s.poisson_ratio)?;
        m.f32("frictionAngle", &mut s.friction_angle)?;
        m.f32("cohesion", &mut s.cohesion)?;
        m.finish()?;
    }
    let mut margin = d.region_margin as f32;
    f.f32("regionMargin", &mut margin)?;
    d.region_margin = f64::from(margin);
    if let Some(z) = f.take("z") {
        d.z = Some(z.as_f64().ok_or_else(|| err("TerrainDef.z: expected a number"))?);
    }
    f.f32("speedHeadroom", &mut d.speed_headroom)?;
    f.f32("settleSpeed", &mut d.settle_speed)?;
    f.u32("settleFrames", &mut d.settle_frames)?;
    if let Some(s) = f.string("settleRule")? {
        d.settle_rule = match s.as_str() {
            "strict" => SettleRule::Strict,
            "local" => SettleRule::Local,
            _ => return Err(err(format!("TerrainDef.settleRule: unknown {s}"))),
        };
    }
    if let Some(mut s) = f.object("surface")? {
        let mut sd = TerrainSurfaceDef::default();
        s.u32("cells", &mut sd.cells)?;
        s.f32("ejectaFall", &mut sd.ejecta_fall)?;
        s.u32("patchSamples", &mut sd.patch_samples)?;
        s.u32("maxDebris", &mut sd.max_debris)?;
        s.finish()?;
        d.surface = Some(sd);
    }
    if let Some(mut s) = f.object("strike")? {
        let k = &mut d.strike;
        s.f32("minSpeed", &mut k.min_speed)?;
        s.f32("minEnergy", &mut k.min_energy)?;
        s.u32("horizonFrames", &mut k.horizon_frames)?;
        s.u32("samples", &mut k.samples)?;
        s.f32("cellsPerBody", &mut k.cells_per_body)?;
        s.finish()?;
    }
    f.finish()?;
    Ok(d)
}

/// The settings in use, camelCase, with the sand's P-wave speed.
fn def_js(d: &TerrainDef) -> JsValue {
    let c = &d.config;
    let m = &d.material;
    js(serde_json::json!({
        "config": {
            "maxRegions": c.max_regions, "maxParticles": c.max_particles, "maxBlocks": c.max_blocks,
            "maxParcels": c.max_parcels, "particlesPerCellAxis": c.particles_per_cell_axis,
            "cellsPerRadius": c.cells_per_radius, "minCellSize": c.min_cell_size, "cellSize": c.cell_size,
            "minResolution": c.min_resolution, "courant": c.courant, "maxSubsteps": c.max_substeps,
            "maxColumns": c.max_columns, "maxCoupled": c.max_coupled, "maxWheelLoads": c.max_wheel_loads,
            "maxFieldSamples": c.max_field_samples, "bakeHomeFade": c.bake_home_fade,
            "boundary": if c.boundary == RegionBoundary::Rigid { "rigid" } else { "absorbing" },
            "initialStress": if c.initial_stress == InitialStress::Unstressed { "unstressed" } else { "geostatic" },
            "maxBoundaryNodes": c.max_boundary_nodes, "bedDepth": c.bed_depth, "speedGrowth": c.speed_growth,
            "areaClocks": c.area_clocks, "sleepSpeed": c.sleep_speed, "sleepAfter": c.sleep_after,
        },
        "material": {
            "density": m.density, "youngsModulus": m.youngs_modulus, "poissonRatio": m.poisson_ratio,
            "frictionAngle": m.friction_angle, "cohesion": m.cohesion, "pWaveSpeed": m.p_wave_speed(),
        },
        "regionMargin": d.region_margin, "z": d.z, "speedHeadroom": d.speed_headroom,
        "settleSpeed": d.settle_speed, "settleFrames": d.settle_frames,
        "settleRule": if d.settle_rule == SettleRule::Local { "local" } else { "strict" },
        "surface": d.surface.as_ref().map(|s| serde_json::json!({
            "cells": s.cells, "ejectaFall": s.ejecta_fall, "patchSamples": s.patch_samples, "maxDebris": s.max_debris,
        })),
    }))
}

fn crater_json(c: &CraterPrediction) -> serde_json::Value {
    serde_json::json!({
        "radius": c.radius, "depth": c.depth, "rimRadius": c.rim_radius,
        "finalRimRadius": c.final_rim_radius, "volume": c.volume,
        "formationTime": c.formation_time, "complex": c.complex, "piV": c.pi_v,
        "regime": format!("{:?}", c.regime),
    })
}

fn prediction_json(p: &TerrainPrediction) -> serde_json::Value {
    serde_json::json!({
        "crater": crater_json(&p.crater), "dx": p.dx, "blocks": p.blocks, "sourceSpeed": p.source_speed,
    })
}

/// A `TerrainDef` with every field resolved (`terrainDef` rules), and the
/// sand's P-wave speed: what a world would run with.
#[wasm_bindgen(js_name = terrainSettings)]
pub fn terrain_settings(def: JsValue) -> Result<JsValue, JsValue> {
    Ok(def_js(&terrain_def(def)?))
}

/// A vertical impact's crater and region under the settings `def` and
/// gravity `g` (m/s²), without a world: `{ crater: { radius, depth,
/// finalRimRadius, formationTime, ... }, dx, blocks, sourceSpeed, mass,
/// energy }` (`TerrainDef::predict_impact`).
#[wasm_bindgen(js_name = terrainPredictImpact)]
pub fn terrain_predict_impact(def: JsValue, radius: f64, density: f64, speed: f64, g: f64) -> Result<JsValue, JsValue> {
    let d = terrain_def(def)?;
    let src = res(d.vertical_impact(radius, density, speed))?;
    let p = res(d.predict_impact(&src, g))?;
    let mut o = prediction_json(&p);
    o["mass"] = src.mass().into();
    o["energy"] = src.kinetic_energy().into();
    Ok(js(o))
}

/// A buried charge of `kg` TNT `depth` m down; see [`terrain_predict_impact`]
/// (`energy`, no `mass`).
#[wasm_bindgen(js_name = terrainPredictExplosion)]
pub fn terrain_predict_explosion(def: JsValue, kg: f64, depth: f64, g: f64) -> Result<JsValue, JsValue> {
    let d = terrain_def(def)?;
    let src = res(d.tnt_charge(kg, depth))?;
    let p = res(d.predict_explosion(&src, g))?;
    let mut o = prediction_json(&p);
    o["energy"] = src.energy.into();
    Ok(js(o))
}

/// The terrain buffers' layout for renderers: bindings, counter words, record
/// sizes and the live surface's header words (`engine_terrain::layout`,
/// `engine_terrain::surface`).
#[wasm_bindgen(js_name = terrainLayout)]
pub fn terrain_layout() -> JsValue {
    js(serde_json::json!({
        "COUNTER_WORDS": MPM_COUNTER_WORDS, "C_PARTICLES": MPM_C_PARTICLES,
        "C_PARCELS": MPM_C_PARCELS, "C_FLAGS": MPM_C_FLAGS,
        "C_AREA_SPEED": MPM_C_AREA_SPEED, "MAX_AREAS": MPM_MAX_AREAS,
        "PARTICLE_BYTES": std::mem::size_of::<MpmParticle>(),
        "PARCEL_BYTES": std::mem::size_of::<MpmParcel>(),
        "PARTICLE_DEAD": MPM_PARTICLE_DEAD,
        "SURFACE": SURFACE_BINDING, "SURFACE_HEADER_WORDS": SURFACE_HEADER_WORDS,
        "SURFACE_VALID_WORD": SURFACE_VALID_WORD, "SURFACE_VIEW_WORD": SURFACE_VIEW_WORD,
        "SURFACE_VIEW_WORDS": SURFACE_VIEW_WORDS, "SURFACE_VIEW_WINDOW": SURFACE_VIEW_WINDOW,
        "SURFACE_VIEW_OPEN": SURFACE_VIEW_OPEN, "MAX_SURFACE_VIEWS": MPM_MAX_AREAS,
        "SURFACE_DEBRIS_ARGS_WORD": SURFACE_DEBRIS_ARGS_WORD,
    }))
}

#[wasm_bindgen]
impl WebWorld {
    /// Turn on deformable terrain (`World::enable_terrain`; `def`: see
    /// [`terrain_def`]).
    #[wasm_bindgen(js_name = enableTerrain)]
    pub fn enable_terrain(&mut self, def: JsValue) -> Result<(), JsValue> {
        let d = terrain_def(def)?;
        res(self.world.enable_terrain(&d))
    }

    /// The settings `enableTerrain` was given, resolved (`null`: terrain off).
    #[wasm_bindgen(js_name = terrainSettings)]
    pub fn terrain_settings(&self) -> JsValue {
        self.world.terrain_settings().map_or(JsValue::NULL, def_js)
    }

    /// A vertical impact at `point` on heightfield shape `ground` of a sphere
    /// of `radius` m, `density` kg/m³ at `speed` m/s (`World::terrain_impact`):
    /// returns the crater π-scaling predicts.
    #[wasm_bindgen(js_name = terrainImpact)]
    pub fn terrain_impact(&mut self, ground: f64, point: &[f32], radius: f64, density: f64, speed: f64) -> Result<JsValue, JsValue> {
        let d = self.world.terrain_settings().ok_or_else(|| err("terrain is not enabled (call enableTerrain)"))?;
        let src = res(d.vertical_impact(radius, density, speed))?;
        let c = res(self.world.terrain_impact(ShapeId::from_bits(bits(ground)), &src, crate::vec3(point)?))?;
        Ok(js(crater_json(&c)))
    }

    /// A buried charge of `kg` TNT `depth` m under `point` (`World::terrain_explode`).
    #[wasm_bindgen(js_name = terrainExplode)]
    pub fn terrain_explode(&mut self, ground: f64, point: &[f32], kg: f64, depth: f64) -> Result<JsValue, JsValue> {
        let d = self.world.terrain_settings().ok_or_else(|| err("terrain is not enabled (call enableTerrain)"))?;
        let src = res(d.tnt_charge(kg, depth))?;
        let c = res(self.world.terrain_explode(ShapeId::from_bits(bits(ground)), &src, crate::vec3(point)?))?;
        Ok(js(crater_json(&c)))
    }

    /// A sourceless region over `[lo, hi]` with cells of `dx` (`World::terrain_open`).
    #[wasm_bindgen(js_name = terrainOpen)]
    pub fn terrain_open(&mut self, ground: f64, lo: &[f32], hi: &[f32], dx: f32) -> Result<(), JsValue> {
        res(self.world.terrain_open(ShapeId::from_bits(bits(ground)), crate::vec3(lo)?, crate::vec3(hi)?, dx))
    }

    /// Bake every open area now (`World::terrain_close`).
    #[wasm_bindgen(js_name = terrainClose)]
    pub fn terrain_close(&mut self) -> Result<(), JsValue> {
        res(self.world.terrain_close())
    }

    /// Bake open area `slot` now; the others go on.
    #[wasm_bindgen(js_name = terrainCloseArea)]
    pub fn terrain_close_area(&mut self, slot: u32) -> Result<(), JsValue> {
        res(self.world.terrain_close_area(slot))
    }

    /// The open areas, by slot, as the host has them now (an area that baked in
    /// the last step is gone): `[{ slot, shape, origin, end, dx, blocks,
    /// window: [i0, j0, ni, nj], age, bed, spacing }]`, `bed` the bed's height
    /// and `spacing` the particle columns' spacing (m).
    #[wasm_bindgen(js_name = terrainAreas)]
    pub fn terrain_areas(&self) -> JsValue {
        let plan = self.world.sim().terrain_plan_ref();
        js(serde_json::Value::Array(
            self.world
                .terrain_areas()
                .iter()
                .map(|a| {
                    let p = plan.and_then(|p| p.area_params(a.slot));
                    serde_json::json!({
                        "slot": a.slot, "shape": crate::handle(a.shape.to_bits()),
                        "origin": a.plan.origin.to_array(), "end": a.plan.end().to_array(),
                        "dx": a.plan.dx, "blocks": a.plan.blocks.to_array(), "window": a.window,
                        "age": a.age, "bed": p.map(|p| p.column[0]), "spacing": p.map(|p| p.column[2]),
                    })
                })
                .collect(),
        ))
    }

    /// The MPM substeps the last step ran (0: no area coupled).
    #[wasm_bindgen(js_name = terrainLastSubsteps)]
    pub fn terrain_last_substeps(&self) -> u32 {
        self.world.terrain_last_substeps()
    }

    /// The surface patches' shapes (`World::terrain_surface_patches`), once the
    /// first area opened.
    #[wasm_bindgen(js_name = terrainSurfacePatches)]
    pub fn terrain_surface_patches(&self) -> Vec<f64> {
        self.world
            .terrain_surface_patches()
            .iter()
            .map(|s| crate::handle(s.to_bits()))
            .collect()
    }

    /// A heightfield's samples as `World` holds them (row-major): after a bake,
    /// once a frame listed it in `terrainBakes`.
    #[wasm_bindgen(js_name = heightFieldHeights)]
    pub fn height_field_heights(&self, shape: f64) -> Result<Vec<f32>, JsValue> {
        match res(self.world.shape_geometry(ShapeId::from_bits(bits(shape))))? {
            Geometry::HeightField { field, .. } => Ok(field.heights().to_vec()),
            _ => Err(err("not a heightfield")),
        }
    }

    // ---- GPU buffers, for renderers (rebind when one is a new object) ----

    /// The particle pool (`PARTICLE_BYTES` records: xm, vj, c0-c2, f0-f2).
    #[wasm_bindgen(js_name = terrainParticleBuffer)]
    pub fn terrain_particle_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().terrain_particle_buffer())
    }

    /// The ejecta parcels (`PARCEL_BYTES` records: x, y, z, quanta; vx, vy, vz,
    /// mass, 0 once landed).
    #[wasm_bindgen(js_name = terrainParcelBuffer)]
    pub fn terrain_parcel_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().terrain_parcel_buffer())
    }

    /// The terrain counters (`COUNTER_WORDS` words; `C_PARTICLES` bounds the pool).
    #[wasm_bindgen(js_name = terrainCounterBuffer)]
    pub fn terrain_counter_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().terrain_counters_buffer())
    }

    /// The live surface (binding `SURFACE`): the header, a view per area slot,
    /// the heights and the debris list; its debris draw's arguments at word
    /// `SURFACE_DEBRIS_ARGS_WORD` (the page writes the vertex count there).
    #[wasm_bindgen(js_name = terrainSurfaceBuffer)]
    pub fn terrain_surface_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().terrain_surface_buffer())
    }

    /// Words `[word, word + n)` of the live surface buffer, resolved once
    /// mapped (never waits in place): for tools that read a view.
    #[wasm_bindgen(js_name = readTerrainSurface)]
    pub fn read_terrain_surface(&self, word: u32, n: u32) -> js_sys::Promise {
        let buf = self.world.sim().terrain_surface_buffer().clone();
        crate::read::read_ranges(&self.world, vec![(buf, 4 * u64::from(word), 4 * u64::from(n))])
    }

    /// Particle records `[first, first + n)`, resolved once mapped (words).
    #[wasm_bindgen(js_name = readTerrainParticles)]
    pub fn read_terrain_particles(&self, first: u32, n: u32) -> js_sys::Promise {
        let buf = self.world.sim().terrain_particle_buffer().clone();
        let r = std::mem::size_of::<MpmParticle>() as u64;
        crate::read::read_ranges(&self.world, vec![(buf, r * u64::from(first), r * u64::from(n))])
    }

    /// Parcel records `[first, first + n)`, resolved once mapped (words).
    #[wasm_bindgen(js_name = readTerrainParcels)]
    pub fn read_terrain_parcels(&self, first: u32, n: u32) -> js_sys::Promise {
        let buf = self.world.sim().terrain_parcel_buffer().clone();
        let r = std::mem::size_of::<MpmParcel>() as u64;
        crate::read::read_ranges(&self.world, vec![(buf, r * u64::from(first), r * u64::from(n))])
    }
}
