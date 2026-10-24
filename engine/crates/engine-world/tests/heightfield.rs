//! Heightfield shapes through the public API: terrain on a static body, its
//! render mesh, and in-place edits that wake what rests on them.

use rubble::*;

fn world() -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 256,
            max_colliders: 256,
            max_contacts: 16_384,
            max_joints: 16,
            max_candidates: 32_768,
            max_colors: 32,
            max_commands: 64,
            max_hull_vertices: 64,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    w
}

fn run(w: &mut World, steps: u32) {
    for _ in 0..steps {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
}

fn terrain(w: &mut World, field: &HeightField) -> ShapeId {
    let ground = w.create_body(&BodyDef::default()).unwrap();
    w.create_height_field_shape(ground, &ShapeDef::default(), field, Transform::IDENTITY)
        .unwrap()
}

fn crate_box(w: &mut World, at: [f32; 3], half: f32) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: at,
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [half; 3])
        .unwrap();
    b
}

#[test]
fn heightfields_go_on_static_bodies_only() {
    let mut w = world();
    let field = HeightField::new(4, 4, 1.0, 1.0, vec![0.0; 16]).unwrap();
    let b = w
        .create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            ..BodyDef::default()
        })
        .unwrap();
    let err = w
        .create_height_field_shape(b, &ShapeDef::default(), &field, Transform::IDENTITY)
        .unwrap_err();
    assert!(err.to_string().contains("static"), "{err}");
}

#[test]
fn the_render_mesh_is_the_collision_surface() {
    let mut w = world();
    let field = HeightField::from_fn(5, 3, 0.5, 2.0, |x, z| x * 0.3 - z * 0.1).unwrap();
    let ground = w.create_body(&BodyDef::default()).unwrap();
    let shift = Transform::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
    let s = w
        .create_height_field_shape(ground, &ShapeDef::default(), &field, shift)
        .unwrap();
    let mesh = w.shape_mesh(s).unwrap().expect("a heightfield has a mesh");
    assert_eq!(mesh.vertices.len(), 15);
    assert_eq!(mesh.triangles.len(), 4 * 2 * 2);
    for (k, v) in mesh.vertices.iter().enumerate() {
        let (i, j) = (k as u32 % 5, k as u32 / 5);
        let local = field.vertex(i, j);
        assert_eq!(*v, [local.x + 1.0, local.y + 2.0, local.z + 3.0]);
    }
    // Counter-clockwise seen from above: every triangle faces up on this field.
    for t in &mesh.triangles {
        let p = |n: usize| glam::Vec3::from(mesh.vertices[t[n] as usize]);
        assert!((p(1) - p(0)).cross(p(2) - p(0)).y > 0.0);
    }
}

#[test]
fn digging_a_pit_wakes_the_sleeping_box_above_it() {
    let mut w = world();
    let mut field = HeightField::new(21, 21, 0.5, 0.5, vec![0.0; 21 * 21]).unwrap();
    let ground = terrain(&mut w, &field);
    let b = crate_box(&mut w, [0.0, 1.0, 0.0], 0.3);
    run(&mut w, 240);
    assert!(
        !w.body_is_awake(b).unwrap(),
        "the box should have gone to sleep"
    );
    // The world's default solver rests a box a few cm deep, on a plane too.
    let rest = w.body_transform(b).unwrap().unwrap().position[1];
    assert!((rest - 0.3).abs() < 0.05, "rests at {rest}");
    // A pit 0.5 m deep under the box: samples 7..14 in both directions (the
    // pit's flat floor spans x, z in [-1, 1]; the box is 0.6 m wide).
    field.set_region(7, 7, 7, 7, &[-0.5; 49]).unwrap();
    w.set_height_field_region(ground, 7, 7, 7, 7, &[-0.5; 49])
        .unwrap();
    run(&mut w, 240);
    let fallen = w.body_transform(b).unwrap().unwrap().position[1];
    // Woken and fallen onto the pit floor (same rest band as on the flat).
    assert!(w.body_transform(b).unwrap().is_some());
    assert!(
        (fallen - (-0.5 + 0.3)).abs() < 0.05,
        "after the edit the box is at {fallen} (rested at {rest})"
    );
    match w.shape_geometry(ground).unwrap() {
        Geometry::HeightField { field: stored, .. } => assert_eq!(stored, &field),
        g => panic!("{g:?}"),
    }
    let err = w
        .set_height_field_region(ground, 18, 0, 5, 1, &[0.0; 5])
        .unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
}

#[test]
fn digging_wakes_the_box_on_terrain_away_from_the_origin() {
    // The terrain's body sits at (30, 2, -20): the edit's region is found
    // from that pose, not the origin.
    let mut w = world();
    let mut field = HeightField::new(21, 21, 0.5, 0.5, vec![0.0; 21 * 21]).unwrap();
    let ground_body = w
        .create_body(&BodyDef {
            position: [30.0, 2.0, -20.0],
            ..BodyDef::default()
        })
        .unwrap();
    let ground = w
        .create_height_field_shape(
            ground_body,
            &ShapeDef::default(),
            &field,
            Transform::IDENTITY,
        )
        .unwrap();
    let b = crate_box(&mut w, [30.0, 3.0, -20.0], 0.3);
    run(&mut w, 240);
    assert!(
        !w.body_is_awake(b).unwrap(),
        "the box should have gone to sleep"
    );
    field.set_region(7, 7, 7, 7, &[-0.5; 49]).unwrap();
    w.set_height_field_region(ground, 7, 7, 7, 7, &[-0.5; 49])
        .unwrap();
    run(&mut w, 240);
    let fallen = w.body_transform(b).unwrap().unwrap().position[1];
    assert!(
        (fallen - (2.0 - 0.5 + 0.3)).abs() < 0.05,
        "after the edit the box is at {fallen}"
    );
}
