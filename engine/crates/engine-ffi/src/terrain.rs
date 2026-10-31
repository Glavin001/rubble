//! Deformable terrain (`engine/docs/deformable-terrain.md`): impacts, buried
//! charges and heavy-body strikes turn a heightfield shape's ground into MPM
//! sand where it must move, and bake it back once it settles.
//!
//! Lengths, speeds, densities and energies are in the world's coordinates and
//! units, like the rest of the ABI; Young's modulus is in Pa and angles in
//! radians whatever the units.

use super::*;

/// Deformable-terrain settings (from [`rbDefaultTerrainDef`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbTerrainDef {
    /// Limits: exceeding one fails the step with its name.
    pub max_particles: u32,
    pub max_blocks: u32,
    /// Ejecta parcels a region may throw (0: material leaving the region's box
    /// is an error).
    pub max_parcels: u32,
    pub max_substeps: u32,
    /// Cells across an impact's predicted transient radius.
    pub cells_per_radius: f32,
    /// Smallest cell a region may use (length).
    pub min_cell_size: f32,
    /// Areas open at once: impacts far apart open areas of their own, one
    /// near an open area grows it.
    pub max_regions: u32,
    /// One cell size (length) for every event, or 0: the first event's own
    /// while any area is open (all open areas share one grid).
    pub cell_size: f32,
    /// The coarsest a later event may run on the open grid, as a fraction of
    /// the cells it asks for (0, 1]; coarser fails.
    pub min_resolution: f32,
    /// Dry sand: density, Young's modulus (Pa), Poisson's ratio, friction angle
    /// (rad).
    pub density: f32,
    pub youngs_modulus: f32,
    pub poisson_ratio: f32,
    pub friction_angle: f32,
    /// How far an impact's region reaches past its predicted transient radius,
    /// in those radii.
    pub region_margin: f32,
    /// A region bakes once its sand and every body in it stay slower than this
    /// (speed), with nothing in flight, for `settle_frames` frames.
    pub settle_speed: f32,
    pub settle_frames: u32,
    /// Added to each frame's speed bound for its substeps (speed).
    pub speed_headroom: f32,
    /// Strikes on watched shapes ([`rbTerrain_WatchShape`]): the impact speed
    /// and kinetic energy that open a region, and cells across the striking
    /// body.
    pub strike_min_speed: f32,
    pub strike_min_energy: f32,
    pub strike_cells_per_body: f32,
    pub internal_value: u32,
}

/// π-scaling's prediction for an impact or charge (lengths in world units).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RbCraterPrediction {
    pub transient_radius: f32,
    pub transient_depth: f32,
    pub final_rim_radius: f32,
    /// Seconds.
    pub formation_time: f32,
}

/// Defaults in metres (the default coordinates); scale lengths, speeds,
/// densities and energies for other units.
#[no_mangle]
pub extern "C" fn rbDefaultTerrainDef() -> RbTerrainDef {
    let d = TerrainDef::default();
    RbTerrainDef {
        max_particles: d.config.max_particles,
        max_blocks: d.config.max_blocks,
        max_parcels: d.config.max_parcels,
        max_substeps: d.config.max_substeps,
        cells_per_radius: d.config.cells_per_radius,
        min_cell_size: d.config.min_cell_size,
        max_regions: d.config.max_regions,
        cell_size: d.config.cell_size,
        min_resolution: d.config.min_resolution,
        density: d.material.density,
        youngs_modulus: d.material.youngs_modulus,
        poisson_ratio: d.material.poisson_ratio,
        friction_angle: d.material.friction_angle,
        region_margin: d.region_margin as f32,
        settle_speed: d.settle_speed,
        settle_frames: d.settle_frames,
        speed_headroom: d.speed_headroom,
        strike_min_speed: d.strike.min_speed,
        strike_min_energy: d.strike.min_energy,
        strike_cells_per_body: d.strike.cells_per_body,
        internal_value: COOKIE,
    }
}

/// Turn on deformable terrain for the world.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_EnableTerrain(
    world: *mut RbWorld,
    def: *const RbTerrainDef,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null terrain def")?;
        check_cookie(d.internal_value, "terrain def not from rbDefaultTerrainDef")?;
        let c = &w.coords;
        let mut t = TerrainDef::default();
        t.config.max_particles = d.max_particles;
        t.config.max_blocks = d.max_blocks;
        t.config.max_parcels = d.max_parcels;
        t.config.max_substeps = d.max_substeps;
        t.config.cells_per_radius = d.cells_per_radius;
        t.config.min_cell_size = c.length_in(d.min_cell_size);
        t.config.max_regions = d.max_regions;
        t.config.cell_size = c.length_in(d.cell_size);
        t.config.min_resolution = d.min_resolution;
        t.material.density = c.density_in(d.density);
        t.material.youngs_modulus = d.youngs_modulus;
        t.material.poisson_ratio = d.poisson_ratio;
        t.material.friction_angle = d.friction_angle;
        t.region_margin = f64::from(d.region_margin);
        t.settle_speed = c.length_in(d.settle_speed);
        t.settle_frames = d.settle_frames;
        t.speed_headroom = c.length_in(d.speed_headroom);
        t.strike.min_speed = c.length_in(d.strike_min_speed);
        t.strike.min_energy = d.strike_min_energy / c.energy_out(1.0);
        t.strike.cells_per_body = d.strike_cells_per_body;
        w.world.enable_terrain(&t)
    })
}

unsafe fn write_prediction(c: &coords::Coords, p: &CraterPrediction, out: *mut RbCraterPrediction) {
    if let Some(o) = out.as_mut() {
        *o = RbCraterPrediction {
            transient_radius: c.length_out(p.radius as f32),
            transient_depth: c.length_out(p.depth as f32),
            final_rim_radius: c.length_out(p.final_rim_radius as f32),
            formation_time: p.formation_time as f32,
        };
    }
}

/// A hypervelocity impact at `point` on heightfield `shape`: a sphere of
/// `impactor_radius` and `impactor_density` arriving at `velocity` (its
/// downward part; the excavation field is vertical). `out` (may be null)
/// receives π-scaling's prediction; the crater itself is simulated. Several
/// may be under way: one far from the open areas opens its own, one near an
/// open area (into a crater still forming, on its rim) grows it and adds its
/// flow to the material's.
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_Impact(
    world: *mut RbWorld,
    shape: RbShapeId,
    impactor_radius: f32,
    impactor_density: f32,
    velocity: RbVec3,
    point: RbVec3,
    out: *mut RbCraterPrediction,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let c = w.coords;
        let def = w
            .world
            .terrain_settings()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (rbWorld_EnableTerrain)"))?
            .clone();
        let v = c.linear_in([velocity.x, velocity.y, velocity.z]);
        let impact = ImpactSource {
            impactor_radius: f64::from(c.length_in(impactor_radius)),
            impactor_density: f64::from(c.density_in(impactor_density)),
            velocity: glam::DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])),
            target: CraterTarget::sand(&def.material)?,
        };
        let p = c.position_in([point.x, point.y, point.z]);
        let crater = w
            .world
            .terrain_impact(ShapeId::from_bits(shape), &impact, p)?;
        write_prediction(&c, &crater, out);
        Ok(())
    })
}

/// A buried charge of `tnt_kg` TNT-equivalent, its centre `depth_of_burial`
/// below `point` on heightfield `shape`. `out` as [`rbTerrain_Impact`].
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_Explode(
    world: *mut RbWorld,
    shape: RbShapeId,
    tnt_kg: f32,
    depth_of_burial: f32,
    point: RbVec3,
    out: *mut RbCraterPrediction,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let c = w.coords;
        let def = w
            .world
            .terrain_settings()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled (rbWorld_EnableTerrain)"))?
            .clone();
        let charge = ExplosionSource::tnt(
            f64::from(tnt_kg),
            f64::from(c.length_in(depth_of_burial)),
            CraterTarget::sand(&def.material)?,
        );
        let p = c.position_in([point.x, point.y, point.z]);
        let crater = w
            .world
            .terrain_explode(ShapeId::from_bits(shape), &charge, p)?;
        write_prediction(&c, &crater, out);
        Ok(())
    })
}

/// Open a region over the box between corners `a` and `b` with cells of
/// `cell_size`, with no source (for bodies landing or driving on it).
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_Open(
    world: *mut RbWorld,
    shape: RbShapeId,
    a: RbVec3,
    b: RbVec3,
    cell_size: f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let c = w.coords;
        let (pa, pb) = (
            c.position_in([a.x, a.y, a.z]),
            c.position_in([b.x, b.y, b.z]),
        );
        let lo = [pa[0].min(pb[0]), pa[1].min(pb[1]), pa[2].min(pb[2])];
        let hi = [pa[0].max(pb[0]), pa[1].max(pb[1]), pa[2].max(pb[2])];
        w.world
            .terrain_open(ShapeId::from_bits(shape), lo, hi, c.length_in(cell_size))
    })
}

/// Bake every open area now (each otherwise bakes once settled).
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_Close(world: *mut RbWorld) -> RbResult {
    guard_args(|| world_mut(world)?.world.terrain_close())
}

/// Watch heightfield `shape` for heavy-body strikes: one predicted to pass the
/// def's speed and energy opens a region before the body arrives.
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_WatchShape(
    world: *mut RbWorld,
    shape: RbShapeId,
    watch: bool,
) -> RbResult {
    guard_args(|| {
        world_mut(world)?
            .world
            .terrain_watch_shape(ShapeId::from_bits(shape), watch)
    })
}

/// Whether any area is open now (`*out_open`).
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_IsOpen(world: *mut RbWorld, out_open: *mut bool) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out_open, w.world.terrain_region().is_some())
    })
}

/// Terrain areas open now (`*out_count`).
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_AreaCount(world: *mut RbWorld, out_count: *mut u32) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out_count, w.world.terrain_areas().len() as u32)
    })
}

/// Strikes seen too late to open a region before them (`*out_count`).
#[no_mangle]
pub unsafe extern "C" fn rbTerrain_MissedStrikes(
    world: *mut RbWorld,
    out_count: *mut u64,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        write(out_count, w.world.terrain_missed_strikes())
    })
}

/// A heightfield shape's samples as the host's copy holds them (updated by
/// the frames that follow a bake), in the order and units of
/// `rbCreateHeightFieldShape` (`count_a × count_b`, A fastest).
#[no_mangle]
pub unsafe extern "C" fn rbHeightField_GetHeights(
    world: *mut RbWorld,
    shape: RbShapeId,
    count_a: u32,
    count_b: u32,
    out_heights: *mut f32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        if out_heights.is_null() {
            return Err(invalid("null heights output"));
        }
        let Geometry::HeightField { field, .. } =
            w.world.shape_geometry(ShapeId::from_bits(shape))?
        else {
            return Err(invalid("shape is not a heightfield"));
        };
        let unreal = w.coords.unreal();
        let grid = HostGrid { count_a, count_b };
        if grid.rubble_size(unreal) != field.samples() {
            return Err(invalid("sample counts differ from the field's"));
        }
        let out = std::slice::from_raw_parts_mut(out_heights, count_a as usize * count_b as usize);
        for b in 0..count_b {
            for a in 0..count_a {
                let (i, j) = grid.to_rubble(unreal, a, b);
                out[(b * count_a + a) as usize] = w.coords.length_out(field.height(i, j));
            }
        }
        Ok(())
    })
}
