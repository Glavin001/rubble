//! A driven car through a shop front: a rigid raycast vehicle (the sedan preset, its
//! wheels driving it on the street) at the Ram Raid storefront row as a `Threshold`
//! destructible. The car's contacts break the shop window's bonds and it drives in.
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

/// The sedan driven from 30 m out at full throttle into the middle shop's window, its
/// chassis a projectile (`predict`) or not: (how far in its centre gets, bonds broken).
fn drive(predict: bool) -> (f32, usize) {
    let pack = engine_fracture::samples::pack_sample_json("rr_storefront_row").unwrap();
    let asset = DestructibleAssetDesc::from_pack(&pack).unwrap();
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let offset = stress.placement().map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    // The street is level with the footings' tops (the Ram Raid scenes' street).
    let ground = stress_ref::buildings::StorefrontLayout::FOOTING_DEPTH as f32 + offset[1];
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: ground })
        .unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })
        .unwrap();
    // The middle shop's window (stress frame x), the front face at stress y = -3.5, i.e.
    // world z = +3.5 (Y up: (x, z, -y)).
    let x = {
        use stress_ref::buildings::StorefrontLayout as L;
        let (_, win) = L::shopfront(2);
        0.5 * (win.0 + win.1) as f32
    };
    let start = [x + offset[0], ground + 0.75, 3.5 + 30.0 + offset[2]];
    // Facing -z (the sedan's forward is +z).
    let car = w
        .create_vehicle(&VehicleDef {
            model: VehicleModel::Rigid(engine_vehicle::presets::sedan()),
            transform: Transform::new(start, [0.0, 1.0, 0.0, 0.0]),
            parts: vec![],
            connections: vec![],
            user_data: 0,
        })
        .unwrap();
    let h = car.vehicle.unwrap();
    if predict {
        w.body_set_projectile(car.chassis, Some(Projectile::default())).unwrap();
    }
    let mut broken = 0usize;
    let mut deepest = f32::INFINITY;
    for n in 0..360 {
        w.set_vehicle_input(h, VehicleInput { forward: 1.0, ..VehicleInput::default() });
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    broken += bonds.len();
                }
            }
        }
        deepest = deepest.min(w.read_body_state(car.chassis).unwrap().transform.position[2]);
        if n == 200 {
            // Its wheels have driven it up to speed on the street.
            let v = w.read_body_state(car.chassis).unwrap().linear_velocity;
            assert!(v[2] < -10.0, "{v:?}");
        }
    }
    let stats = w.destructible_stats(id).unwrap();
    assert_eq!(stats.bonds - stats.bonds_intact, broken as u32, "events report every break");
    (3.5 + offset[2] - deepest, broken)
}

/// Strike prediction is what lets the car in: its boxes' strike breaks the window's bonds
/// before the step that meets them. Without it the shop front stops the car dead in that
/// step and only then breaks.
#[test]
fn a_driven_car_breaks_into_a_shop() {
    let (into, broken) = drive(true);
    // The car's centre ends past the front wall, inside the 7 m deep shop.
    assert!(into > 3.0 && broken > 20, "{into} m in, {broken} bonds");
    let (into, broken) = drive(false);
    assert!(into < 0.0 && broken < 20, "without prediction: {into} m in, {broken} bonds");
}
