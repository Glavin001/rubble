//! A raycast vehicle's wheels load the chunks they stand on (W5): the tyres' forces reach
//! the fracture's contact ingest (stress-gpu `ingest_wheels`) though the engine's contact
//! rows never see them. The Bridge Out viaduct as a `Threshold` destructible, its bonds
//! weakened until a parked six-wheel truck's weight is more than a deck bond holds.
#![cfg(feature = "stress")]

use rubble::*;

fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c.max_bodies.max(((chunks + 64) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

/// Bonds broken in 2 s with the truck parked on span 3 (`truck`) or the deck empty, every
/// bond's capacity scaled by `strength`.
fn parked(truck: bool, strength: f32) -> usize {
    let pack = engine_fracture::samples::pack_sample_json("bo_viaduct").unwrap();
    let asset = DestructibleAssetDesc::from_pack(&pack).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: capacity(&asset),
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )
    .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    w.create_destructible(&DestructibleDef {
        model: StrengthModel::Threshold,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        strength_scale: strength,
        asleep: false,
        ..DestructibleDef::new(a)
    })
    .unwrap();
    if truck {
        // Span 3 (stress x 1..15.5), on the deck's centre line, heading along the bridge
        // (+x; the design nose is +z), on its wheels at full droop: only they touch.
        use stress_ref::viaduct::ViaductLayout as L;
        let desc = engine_vehicle::presets::six_wheel_truck();
        let at = [8.0 + offset[0], L::DECK_TOP as f32 + desc.droop_height() + offset[1], offset[2]];
        let yaw = std::f32::consts::FRAC_PI_2;
        let b = w
            .create_vehicle(&VehicleDef {
                model: VehicleModel::Rigid(desc),
                transform: Transform::new(at, glam::Quat::from_rotation_y(yaw).to_array()),
                parts: vec![],
                connections: vec![],
                user_data: 0,
            })
            .unwrap();
        w.set_vehicle_input(b.vehicle.unwrap(), VehicleInput { brake: 1.0, ..VehicleInput::default() });
    }
    let mut broken = 0;
    for _ in 0..120 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    broken += bonds.len();
                }
            }
        }
    }
    broken
}

/// At a thousandth of its strength a deck bond holds about 4 kN, less than one of the
/// truck's wheels carries (5.5 t on six). Nothing but the wheels touches the deck, so
/// without their loads nothing would break; with them the deck under the truck goes. The
/// empty deck at the same strength keeps every bond, and at full strength the truck
/// breaks nothing (its wheels carry tens of kN against MN bonds).
#[test]
fn a_parked_truck_loads_the_deck() {
    assert_eq!(parked(false, 1e-3), 0, "the empty deck holds");
    let broken = parked(true, 1e-3);
    assert!(broken > 0, "the truck's wheels break nothing: {broken} bonds");
    assert_eq!(parked(true, 1.0), 0, "full strength: the deck carries the truck");
}
