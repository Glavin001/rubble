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
/// at `ground`. A template whose node box is longer in x than in z is first turned so its
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
    let shift = Vec3::new(0.0, front.y + 0.02 - lo.y, 0.0)
        + dir * (front.dot(dir) - ahead)
        + lateral * (front.dot(lateral) - 0.5 * (lo + hi).dot(lateral));
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = dir * speed;
    }
    sim.origin = Vec3::ZERO;
    sim.ground = ground;
    sim
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
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default())?;
    let (lab, _) = engine_fracture::samples::cuboid_stress_scene(None)?;
    let stress = asset.stress.as_ref().ok_or_else(|| anyhow::anyhow!("crash town: no stress source"))?;
    let offset = stress.placement().map(|x| -x as f32);
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("crash town: no threshold solver"))?;
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let houses = houses.max(count.div_ceil(4)).max(1);
    let mut cap = stress_rubble::capacity_for(&scene);
    let h = houses as u32;
    cap.max_bodies = (cap.max_bodies.max((chunks + 20) as u32) * h + 4 * h + 64).next_power_of_two();
    cap.max_colliders = (cap.max_colliders * h + 4 * h + 64).next_power_of_two();
    cap.max_hull_vertices *= h;
    cap.max_contacts = stress_rubble::threshold::contact_capacity(chunks * houses, cap.max_contacts * h, 1 << 30);
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
    let cols = (houses as f64).sqrt().ceil() as usize;
    let pitch = 32.0f32;
    let half = [0.5 * lab.extent_x as f32, 0.5 * lab.extent_z as f32];
    let mut ids = Vec::new();
    let mut centres = Vec::new();
    for k in 0..houses {
        let at = [
            (k % cols) as f32 * pitch,
            0.0,
            (k / cols) as f32 * pitch,
        ];
        ids.push(world.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new([offset[0] + at[0], offset[1], offset[2] + at[2]], [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })?);
        centres.push([
            lab.center[0] as f32 + offset[0] + at[0],
            lab.center[1] as f32 + offset[1],
            lab.center[2] as f32 + offset[2] + at[2],
        ]);
    }
    for _ in 0..settle {
        world.step()?;
        world.poll_blocking()?;
    }
    let mut vehicles = Vec::new();
    let mut crates = Vec::new();
    for i in 0..count {
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
            bodies: 64,
            shapes: 255,
            planes: 4096,
        },
        ..ExplicitVehiclesDef::default()
    }
}
