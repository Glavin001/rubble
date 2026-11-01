//! Scene save and restore, body type changes and pinned material slots (the
//! web pages' reset key, parked projectiles and vehicle colliders).

use rubble::*;

fn world() -> (World, BodyId) {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [20.0, 0.5, 20.0])
        .unwrap();
    let b = w
        .create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: [0.0, 3.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5, 0.5, 0.5])
        .unwrap();
    (w, b)
}

fn y_after(w: &mut World, b: BodyId, steps: u32) -> f32 {
    for _ in 0..steps {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    w.read_body_state(b).unwrap().transform.position[1]
}

/// A restore returns the world to the save: the same steps fall the same way.
#[test]
fn restore_repeats_the_steps_after_a_save() {
    let (mut w, b) = world();
    let s = w.save_state().unwrap();
    let first = y_after(&mut w, b, 40);
    assert!(first < 2.0, "the box fell ({first})");
    w.restore_state(&s).unwrap();
    assert!((w.read_body_state(b).unwrap().transform.position[1] - 3.0).abs() < 1e-6);
    let again = y_after(&mut w, b, 40);
    assert!((again - first).abs() < 1e-3, "{again} vs {first}");
}

/// A dynamic body parked static holds still; launched again it flies with its
/// mass data intact.
#[test]
fn a_body_parked_static_and_launched_again() {
    let (mut w, b) = world();
    w.body_set_mass_data(b, MassData::principal(2.0, [0.0; 3], [0.4; 3]))
        .unwrap();
    let park = Transform::new([10.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    w.body_set_type(b, BodyType::Static, park, [0.0; 3], [0.0; 3])
        .unwrap();
    assert!(
        (y_after(&mut w, b, 20) - 5.0).abs() < 1e-6,
        "a static body stays put"
    );
    let at = Transform::new([0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
    w.body_set_type(b, BodyType::Dynamic, at, [0.0, 4.0, 0.0], [0.0; 3])
        .unwrap();
    assert_eq!(w.body_mass_data(b).unwrap().map(|m| m.mass), Some(2.0));
    let y = y_after(&mut w, b, 12);
    assert!(y > 5.0, "launched upward ({y})");
}

/// A pinned slot holds its material, and shapes with that material share it.
#[test]
fn pinned_material_slots_are_shared() {
    let (mut w, _) = world();
    let ice = SurfaceMaterial {
        friction: 0.08,
        ..SurfaceMaterial::default()
    };
    w.set_material_slot(7, ice.clone()).unwrap();
    let body = w.create_body(&BodyDef::default()).unwrap();
    let def = ShapeDef {
        material: ice,
        ..ShapeDef::default()
    };
    let s = w.create_box_shape(body, &def, [1.0; 3]).unwrap();
    let c = w.shape_collider(s).unwrap();
    assert_eq!(w.sim().collider_record(c).material & 0xFFFF, 7);
}
