//! The explicit vehicles' crash scene (`engine/docs/explicit-vehicles-in-world.md`): many
//! soft-body vehicles driving into prefractured, stress-solved houses (the fracture lab's
//! cuboid house as a threshold destructible) past loose crates, in one World. One builder
//! for the native benchmark (`examples/explicit_vehicles_bench.rs`) and the page
//! (`explicit-world.html`, engine-world-wasm `WebWorld::vehicleCrashTown`).

use crate::{
    BodyDef, BodyId, DestructibleAssetDesc, DestructibleDef, DestructibleId, ExplicitVehiclesDef, Geometry,
    GpuContext, ShapeDef, StrengthModel, Transform, World, WorldDef,
};
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::Vec3;

/// The vehicle `template` turned so its long axis runs along `heading` (a whole number of
/// quarter turns: `turns` about +Y from +Z), its front face's middle at `front` (its lowest
/// node 2 cm above `front.y`), every node at `speed` along the heading, its own flat ground
/// at `ground`, its tyres (if any) rolling. A template whose node box is longer in x than in z is first turned so its
/// long axis is z (its forward is taken along +z).
pub fn place_vehicle(template: &ExplicitBeams, front: Vec3, turns: u32, speed: f32, ground: f32) -> ExplicitBeams {
    let mut sim = template.clone();
    let b = |s: &ExplicitBeams| {
        s.nodes
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)))
    };
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let mut dir = Vec3::Z;
    for _ in 0..turns % 4 {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(n.pos.z, n.pos.y, -n.pos.x);
        }
        dir = Vec3::new(dir.z, 0.0, -dir.x);
    }
    let (lo, hi) = b(&sim);
    let lateral = Vec3::Y.cross(dir).abs();
    let ahead = sim.nodes.iter().map(|n| n.pos.dot(dir)).fold(f32::MIN, f32::max);
    let shift = Vec3::new(0.0, front.y + sim.spawn_clearance() - lo.y, 0.0)
        + dir * (front.dot(dir) - ahead)
        + lateral * (front.dot(lateral) - 0.5 * (lo + hi).dot(lateral));
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = dir * speed;
    }
    sim.origin = Vec3::ZERO;
    sim.ground = ground;
    // Soft tyres rolling with the vehicle, not skidding.
    if !sim.tyres.is_empty() && speed != 0.0 {
        sim.roll_tyres(dir * speed);
    }
    sim
}

/// Lot `k` of `houses` (x, z): a ceil(sqrt) grid at [`PITCH`].
fn lot(k: usize, houses: usize) -> [f32; 2] {
    let cols = (houses as f64).sqrt().ceil() as usize;
    [(k % cols) as f32 * PITCH, (k / cols) as f32 * PITCH]
}

/// The town's lot pitch (m).
pub const PITCH: f32 = 32.0;

fn house_centre(lab: &engine_fracture::Scene, offset: [f32; 3], k: usize, houses: usize) -> [f32; 3] {
    let at = lot(k, houses);
    [
        lab.center[0] as f32 + offset[0] + at[0],
        lab.center[1] as f32 + offset[1],
        lab.center[2] as f32 + offset[2] + at[1],
    ]
}

/// The houses (at least one per four vehicles), their centres and the ground height of a
/// [`crash_town`] of `count` vehicles: for a page's camera.
pub fn town_layout(houses: usize, count: usize) -> anyhow::Result<(Vec<[f32; 3]>, f32)> {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default())?;
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None)?;
    let stress = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("crash town: no stress source"))?;
    let offset = stress.placement().map(|x| -x as f32);
    let houses = houses.max(count.div_ceil(4)).max(1);
    let centres = (0..houses).map(|k| house_centre(&lab, offset, k, houses)).collect();
    Ok((centres, stress.ground_height().unwrap_or(0.0) + offset[1]))
}

/// The crash town ([`crash_town`]).
pub struct CrashTown {
    pub world: World,
    pub houses: Vec<DestructibleId>,
    pub crates: Vec<BodyId>,
    /// The vehicles as spawned (in the world's explicit solver, same order).
    pub vehicles: Vec<ExplicitBeams>,
    /// Ground height and the houses' centres.
    pub ground: f32,
    pub centres: Vec<[f32; 3]>,
}

/// `count` copies of `template` driving at `speed` m/s (each a little faster than the
/// last, up to +20 %) into cuboid houses: `houses` on a 32 m grid (at least one per four
/// vehicles), one vehicle from each side, starting `run_up` metres from its wall, a 300 kg
/// crate on the way. On `gpu`; the houses settle `settle` steps before the vehicles are
/// added. No vehicles: the houses alone (a baseline). [`town_coupling`] is the coupling
/// room the scene needs.
#[allow(clippy::too_many_arguments)]
pub fn crash_town(
    gpu: GpuContext,
    template: &ExplicitBeams,
    houses: usize,
    count: usize,
    speed: f32,
    run_up: f32,
    settle: u32,
    def: &ExplicitVehiclesDef,
) -> anyhow::Result<CrashTown> {
    crash_town_at(gpu, template, houses, count, speed, run_up, settle, def, None, None)
}

/// [`crash_town`] with the World's step `dt` (s; `None`: the houses' solver's).
#[allow(clippy::too_many_arguments)]
pub fn crash_town_at(
    gpu: GpuContext,
    template: &ExplicitBeams,
    houses: usize,
    count: usize,
    speed: f32,
    run_up: f32,
    settle: u32,
    def: &ExplicitVehiclesDef,
    dt: Option<f32>,
    max_houses: Option<usize>,
) -> anyhow::Result<CrashTown> {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default())?;
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None)?;
    let stress = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("crash town: no stress source"))?;
    let offset = stress.placement().map(|x| -x as f32);
    let mut solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("crash town: no threshold solver"))?;
    if let Some(dt) = dt {
        solver.dt = dt;
    }
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    // `max_houses`: past four vehicles a house, the rest meet head on in pairs in
    // open lanes south of the town (a fleet larger than the fracture's capacity
    // of houses).
    let houses = houses.max(count.div_ceil(4)).max(1);
    let houses = max_houses.map_or(houses, |m| houses.min(m.max(1)));
    let mut cap = stress_rubble::capacity_for(&scene);
    let h = houses as u32;
    // Room for the houses by `capacity_for`'s own rule (three times the live slots: a
    // split's children take new slots until compaction reclaims the retired ones) per
    // house, plus a crate per vehicle and the ground; not `capacity_for`'s one-scene
    // floor (16,384 bodies, 65,536 colliders) times the houses, which put 32 houses at
    // 2^20 body slots (every per-slot pass of the step ran over a million slots) and 64
    // past the fracture's scans (`GpuFracture::with_capacity`, at most 2^20).
    let parts: usize = scene
        .bodies
        .iter()
        .flat_map(|b| &b.chunks)
        .map(|c| c.compound.as_ref().map_or(1, |p| p.len().max(1)))
        .sum();
    let extra = 3 * (4 * h + 64);
    cap.max_bodies = (3 * (chunks as u32 + 2) * h + extra).next_power_of_two();
    cap.max_colliders = (3 * (parts as u32 + 2) * h + extra).next_power_of_two();
    cap.max_hull_vertices *= h;
    cap.max_contacts = stress_rubble::threshold::contact_capacity(chunks * houses, cap.max_contacts, 1 << 30);
    cap.max_candidates = cap.max_candidates.max(cap.max_contacts);
    let mut world = World::new(
        gpu,
        &WorldDef {
            capacity: cap,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: crate::ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )?;
    let ground = stress.ground_height().unwrap_or(0.0) + offset[1];
    let g = world.create_body(&BodyDef::default())?;
    world.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: ground,
        },
    )?;
    let a = world.create_destructible_asset(&asset)?;
    let half = [0.5 * lab.extent_x as f32, 0.5 * lab.extent_z as f32];
    let mut ids = Vec::new();
    let centres: Vec<[f32; 3]> = (0..houses).map(|k| house_centre(&lab, offset, k, houses)).collect();
    for k in 0..houses {
        let at = lot(k, houses);
        ids.push(world.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new([offset[0] + at[0], offset[1], offset[2] + at[1]], [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })?);
    }
    for _ in 0..settle {
        world.step()?;
        world.poll_blocking()?;
    }
    let mut vehicles = Vec::new();
    let mut crates = Vec::new();
    let south = centres.iter().map(|c| c[2]).fold(f32::INFINITY, f32::min) - 60.0;
    for i in 0..count {
        if i >= 4 * houses {
            let e = i - 4 * houses;
            let pair = e / 2;
            let at = Vec3::new(
                centres[0][0] + (pair % 16) as f32 * 6.0,
                ground,
                south - (pair / 16) as f32 * (4.0 * run_up + 20.0),
            );
            let v = speed * (1.0 + 0.2 * ((i as f32 * 0.618_034).fract()));
            let (dz, turns) = if e % 2 == 0 { (-1.0, 0) } else { (1.0, 2) };
            vehicles.push(place_vehicle(template, at + Vec3::Z * (dz * (run_up + 3.0)), turns, v, ground));
            continue;
        }
        let c = Vec3::from(centres[i / 4]);
        // From +Z heading −Z, then from +X heading −X, −Z heading +Z, −X heading +X.
        let (out, turns, reach) = match i % 4 {
            0 => (Vec3::Z, 2, half[1]),
            1 => (Vec3::X, 3, half[0]),
            2 => (Vec3::NEG_Z, 0, half[1]),
            _ => (Vec3::NEG_X, 1, half[0]),
        };
        let wall = Vec3::new(c.x, ground, c.z) + out * reach;
        let v = speed * (1.0 + 0.2 * ((i as f32 * 0.618_034).fract()));
        vehicles.push(place_vehicle(template, wall + out * run_up, turns, v, ground));
        let crate_at = wall + out * (0.45 * run_up) + Vec3::Y * 0.5;
        let b = world.create_body(&BodyDef::dynamic(crate_at.to_array()))?;
        let half_box = [0.5f32; 3];
        world.create_box_shape(
            b,
            &ShapeDef {
                density: 300.0,
                ..ShapeDef::default()
            },
            half_box,
        )?;
        crates.push(b);
    }
    if !vehicles.is_empty() {
        world.add_explicit_vehicles(&vehicles, def)?;
    }
    Ok(CrashTown {
        world,
        houses: ids,
        crates,
        vehicles,
        ground,
        centres,
    })
}

/// The coupling room [`crash_town`]'s vehicles need (a vehicle inside a broken house meets
/// many pieces at once).
pub fn town_coupling() -> ExplicitVehiclesDef {
    ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity {
            bodies: 1024,
            shapes: 1024,
            planes: 16384,
        },
        // A crash scene: vehicle sleep off, so its strikes (and §8's measurements) are
        // unchanged.
        sleep: None,
        ..ExplicitVehiclesDef::default()
    }
}
