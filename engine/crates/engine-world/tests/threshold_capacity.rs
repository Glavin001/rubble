//! The shared threshold ingest's substep slots (regression vs main, PR #105 audit R3 and
//! the `rebuild_shared` mismatch): a World with no explicit vehicles allocates what main
//! did (one slot), a World with vehicles has their slot (two) however its fracture was
//! laid out, and steps. And R2: a threshold destructible in a World of more than 2^20
//! body slots is refused when it is made, as on main (there by the fracture's own check,
//! "the scans hold at most 2^20 of each"; now by the capacity's name).
#![cfg(feature = "stress")]

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::*;

fn sedan(ground: f32, front: Vec3) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
        centred_damping: false,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let lo = sim.nodes.iter().fold(Vec3::MAX, |a, n| a.min(n.pos));
    let shift = Vec3::new(front.x - lo.x, front.y + 0.02 - lo.y, front.z - lo.z);
    for n in &mut sim.nodes {
        n.pos += shift;
    }
    sim.ground = ground;
    sim
}

/// The fracture lab's cuboid house as a `model` destructible on its ground, in its own
/// World; `vehicle_first` adds the sedan (parked 30 m off) before the house is placed.
fn house_world(model: StrengthModel, vehicle: Option<bool>) -> World {
    house_world_with(model, vehicle, 0).unwrap()
}

/// [`house_world`] with at least `min_bodies` body slots; Err is the destructible's.
fn house_world_with(model: StrengthModel, vehicle: Option<bool>, min_bodies: u32) -> anyhow::Result<World> {
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let solver = asset.solver_for(model).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut cap = stress_rubble::capacity_for(&scene);
    cap.max_bodies = cap.max_bodies.max(((chunks + 16 + 4) as u32).next_power_of_two()).max(min_bodies);
    cap.max_contacts = stress_rubble::threshold::contact_capacity(chunks, cap.max_contacts, 1 << 30);
    cap.max_candidates = cap.max_candidates.max(cap.max_contacts);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: cap,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let h = stress.ground_height().unwrap() + offset[1];
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: h })
        .unwrap();
    let car = sedan(h, Vec3::new(30.0, h, 30.0));
    let def = ExplicitVehiclesDef {
        capacity: engine_webgpu::CouplingCapacity { bodies: 64, shapes: 255, planes: 4096 },
        ..ExplicitVehiclesDef::default()
    };
    if vehicle == Some(true) {
        w.add_explicit_vehicles(std::slice::from_ref(&car), &def).unwrap();
    }
    let a = w.create_destructible_asset(&asset).unwrap();
    w.create_destructible(&DestructibleDef {
        model,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        asleep: false,
        ..DestructibleDef::new(a)
    })?;
    if vehicle == Some(false) {
        w.add_explicit_vehicles(std::slice::from_ref(&car), &def).unwrap();
    }
    Ok(w)
}

fn step(w: &mut World, n: u32) {
    for _ in 0..n {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
}

/// No explicit vehicles: one slot, main's allocation (`EngineIngest::new(.., 1, ..)`),
/// for a fracture that is appended to (threshold) and one laid out anew (load path).
#[test]
fn a_world_without_vehicles_has_mains_one_slot() {
    for model in [StrengthModel::Threshold, StrengthModel::LoadPath] {
        let mut w = house_world(model, None);
        assert_eq!(w.threshold_ingest_slots(), Some(1), "{model:?}");
        step(&mut w, 3);
        assert_eq!(w.threshold_ingest_slots(), Some(1), "{model:?} after steps");
    }
}

/// Explicit vehicles need the second slot, whichever came first, and the frame runs
/// (the merge of the vehicles' rows asserts the slot is there).
#[test]
fn a_world_with_vehicles_has_their_slot_and_steps() {
    for model in [StrengthModel::Threshold, StrengthModel::LoadPath] {
        for vehicle_first in [false, true] {
            let mut w = house_world(model, Some(vehicle_first));
            println!("{model:?}, vehicle first {vehicle_first}: {:?} slots", w.threshold_ingest_slots());
            step(&mut w, 3);
            assert_eq!(w.threshold_ingest_slots(), Some(2), "{model:?}, vehicle first {vehicle_first}");
        }
    }
}

/// Vehicles added to a fracture that has already stepped (the slot is grown in place).
#[test]
fn vehicles_added_after_steps_get_their_slot() {
    let mut w = house_world(StrengthModel::Threshold, None);
    step(&mut w, 3);
    let asset = DestructibleAssetDesc::cuboid_house(&Default::default()).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let ground = stress.ground_height().unwrap() - stress.placement()[1] as f32;
    let car = sedan(ground, Vec3::new(30.0, ground, 30.0));
    w.add_explicit_vehicles(std::slice::from_ref(&car), &ExplicitVehiclesDef::default()).unwrap();
    assert_eq!(w.threshold_ingest_slots(), Some(2));
    step(&mut w, 3);
}

/// More than 2^20 body slots: the fracture's split scans cannot rank them, so the
/// destructible is refused when made (main refused it at the same call), and 2^20 is
/// accepted.
#[test]
fn a_threshold_world_past_2_20_body_slots_is_refused() {
    let err = house_world_with(StrengthModel::Threshold, None, (1 << 20) + 1).err().expect("refused");
    let text = format!("{err:#}");
    assert!(text.contains("2^20"), "{text}");
    assert!(text.contains("max_bodies"), "the error names the capacity: {text}");
    house_world_with(StrengthModel::Threshold, None, 1 << 20).expect("2^20 body slots fit");
}
