//! Characters, casts and projectiles through the World: their state and
//! events arrive with each step's frame.

use rubble::*;

fn world() -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 1024,
            max_colliders: 1024,
            max_contacts: 32_768,
            max_joints: 64,
            max_candidates: 65_536,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 1024,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [40.0, 0.5, 40.0])
        .unwrap();
    w
}

fn wall(w: &mut World, x: f32, half_thickness: f32) -> (BodyId, ShapeId) {
    let b = w
        .create_body(&BodyDef {
            position: [x, 2.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let s = w
        .create_box_shape(b, &ShapeDef::default(), [half_thickness, 2.0, 2.0])
        .unwrap();
    (b, s)
}

fn run(w: &mut World, steps: usize) -> Vec<Frame> {
    for _ in 0..steps {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap()
}

fn character_def(position: [f32; 3]) -> CharacterDef {
    CharacterDef {
        position,
        user_data: 7,
        ..CharacterDef::default()
    }
}

#[test]
fn a_character_lands_stands_and_walks() {
    let mut w = world();
    let c = w.create_character(&character_def([0.0, 2.0, 0.0])).unwrap();
    let frames = run(&mut w, 90);
    let landed: Vec<_> = frames
        .iter()
        .flat_map(|f| &f.character_events)
        .filter(|e| e.kind == CharacterEventKind::Landed)
        .collect();
    assert_eq!(landed.len(), 1, "one landing: {landed:?}");
    assert_eq!((landed[0].character, landed[0].user_data), (c, 7));
    let s = w.character_state(c).unwrap().expect("reported every frame");
    let d = CharacterDesc::default();
    // Centre height standing on the floor (y = 0) a skin above it.
    let stand = d.radius + d.half_height + d.skin;
    assert_eq!(s.ground, GroundState::OnGround);
    assert!(
        (s.position[1] - stand).abs() < 2e-3,
        "capsule rests on the floor: {:?}",
        s.position
    );
    assert!(!s.ground_body.is_null(), "stands on the ground body");
    // The inner capsule is no body move: characters report through their own channel.
    let inner = w.character_body(c).unwrap();
    assert!(frames
        .iter()
        .flat_map(|f| &f.body_events.moves)
        .all(|m| m.body != inner));

    w.character_set_input(c, [2.0, 0.0, 0.0], false).unwrap();
    run(&mut w, 60);
    let walked = w.character_state(c).unwrap().unwrap();
    assert!(
        (walked.position[0] - 2.0).abs() < 0.25,
        "one second at 2 m/s: {:?}",
        walked.position
    );
    assert!((walked.position[1] - stand).abs() < 2e-3);
}

#[test]
fn a_character_jumps_once_per_request() {
    let mut w = world();
    let c = w.create_character(&character_def([0.0, 1.0, 0.0])).unwrap();
    run(&mut w, 60);
    let floor = w.character_state(c).unwrap().unwrap().position[1];
    w.character_set_input(c, [0.0; 3], true).unwrap();
    let frames = run(&mut w, 120);
    let peak = frames
        .iter()
        .flat_map(|f| &f.characters)
        .map(|s| s.position[1])
        .fold(f32::MIN, f32::max);
    // v²/2g with the default jump speed (5 m/s).
    let expected = 25.0 / (2.0 * 9.81);
    assert!(
        (peak - floor - expected).abs() < 0.15,
        "peak {:.3} above the floor, expected {expected:.3}",
        peak - floor
    );
    let kinds: Vec<_> = frames
        .iter()
        .flat_map(|f| &f.character_events)
        .map(|e| e.kind)
        .collect();
    // A jump is not a lost support: only the landing is reported.
    assert_eq!(kinds, vec![CharacterEventKind::Landed]);
    // The same input again is not a second jump.
    w.character_set_input(c, [0.0; 3], false).unwrap();
    let again = run(&mut w, 30);
    assert!(again.iter().all(|f| f.character_events.is_empty()));
}

#[test]
fn a_character_pushes_light_bodies_and_is_blocked_by_walls() {
    let mut w = world();
    wall(&mut w, 3.0, 0.25);
    let c = w.create_character(&character_def([0.0, 1.0, 0.0])).unwrap();
    run(&mut w, 30);
    w.character_set_input(c, [3.0, 0.0, 0.0], false).unwrap();
    run(&mut w, 120);
    let s = w.character_state(c).unwrap().unwrap();
    let d = CharacterDesc::default();
    // Wall face at x = 2.75: the capsule stops a radius (and its skin) short.
    assert!(
        s.position[0] < 2.75 - d.radius + 0.01 && s.position[0] > 2.75 - d.radius - 0.1,
        "stopped at the wall: {:?}",
        s.position
    );
}

#[test]
fn destroyed_characters_are_stale_and_their_slots_reused() {
    let mut w = world();
    let a = w.create_character(&character_def([0.0, 1.0, 0.0])).unwrap();
    run(&mut w, 5);
    w.destroy_character(a).unwrap();
    assert!(!w.character_is_valid(a));
    assert!(w.character_set_input(a, [1.0, 0.0, 0.0], false).is_err());
    let f = run(&mut w, 3);
    assert!(
        f.last().unwrap().characters.is_empty(),
        "no longer reported"
    );
    let b = w.create_character(&character_def([5.0, 1.0, 0.0])).unwrap();
    assert_eq!(b.index(), a.index(), "slot reused after quarantine");
    assert_ne!(b, a);
    run(&mut w, 60);
    let s = w.character_state(b).unwrap().unwrap();
    assert!((s.position[0] - 5.0).abs() < 1e-3 && s.ground == GroundState::OnGround);
}

#[test]
fn casts_hit_the_closest_shape_with_the_step() {
    let mut w = world();
    let (_, near) = wall(&mut w, 5.0, 0.5);
    let (far_body, far) = wall(&mut w, 9.0, 0.5);
    w.step_sync().unwrap();
    let ray = w
        .cast(&CastDef {
            origin: [0.0, 2.0, 0.0],
            direction: [1.0, 0.0, 0.0],
            max_distance: 20.0,
            ..CastDef::default()
        })
        .unwrap();
    let sphere = w
        .cast(&CastDef {
            origin: [0.0, 2.0, 0.0],
            direction: [2.0, 0.0, 0.0],
            max_distance: 20.0,
            shape: CastShape::Sphere { radius: 0.5 },
            ..CastDef::default()
        })
        .unwrap();
    let ignoring = w
        .cast(&CastDef {
            origin: [7.0, 2.0, 0.0],
            direction: [1.0, 0.0, 0.0],
            max_distance: 20.0,
            ignore_body: far_body,
            ..CastDef::default()
        })
        .unwrap();
    let miss = w
        .cast(&CastDef {
            origin: [0.0, 2.0, 0.0],
            direction: [-1.0, 0.0, 0.0],
            max_distance: 20.0,
            ..CastDef::default()
        })
        .unwrap();
    let f = w.step_sync().unwrap();
    let get = |id| f.cast_results.iter().find(|r| r.cast == id).unwrap().hit;
    let r = get(ray).expect("the ray hits the near wall");
    assert_eq!(r.shape, near);
    assert!((r.distance - 4.5).abs() < 1e-3 && (r.point[0] - 4.5).abs() < 1e-3);
    assert!((r.normal[0] + 1.0).abs() < 1e-3);
    let s = get(sphere).expect("the sphere hits the near wall");
    assert_eq!(s.shape, near);
    assert!(
        (s.distance - 4.0).abs() < 1e-2,
        "centre stops a radius short: {s:?}"
    );
    assert_eq!(get(ignoring), None, "its only target is ignored");
    assert_eq!(get(miss), None);
    let _ = far;
    // Results are one-shot.
    assert!(w.step_sync().unwrap().cast_results.is_empty());
}

#[test]
fn casts_answer_in_a_sleeping_world() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic([0.0, 0.5, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    run(&mut w, 300);
    assert!(!w.body_is_awake(b).unwrap(), "the box fell asleep");
    let id = w
        .cast(&CastDef {
            origin: [0.0, 5.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            max_distance: 10.0,
            ..CastDef::default()
        })
        .unwrap();
    let f = w.step_sync().unwrap();
    let hit = f.cast_results[0].hit.expect("hits the sleeping box");
    assert_eq!((f.cast_results[0].cast, hit.body), (id, b));
    assert!((hit.distance - 4.0).abs() < 1e-2);
}

#[test]
fn projectiles_pass_thin_walls_and_stop_in_thick_ones() {
    let mut w = world();
    let (_, thin) = wall(&mut w, 5.0, 0.05);
    let (_, thick) = wall(&mut w, 10.0, 1.0);
    w.step_sync().unwrap();
    // 0.01 kg at 400 m/s: 800 J. Crossing costs 2000 J/m: 0.1 m takes 200 J,
    // 2 m would take 4000 J.
    let p = w
        .spawn_projectile(&ProjectileDef {
            position: [0.0, 2.0, 0.0],
            velocity: [400.0, 0.0, 0.0],
            mass: 0.01,
            penetration_cost: 2000.0,
            user_data: 3,
            ..ProjectileDef::default()
        })
        .unwrap();
    let frames = run(&mut w, 10);
    let events: Vec<_> = frames.iter().flat_map(|f| &f.projectile_events).collect();
    assert_eq!(events.len(), 2, "{events:?}");
    assert_eq!((events[0].projectile, events[0].shape), (p, thin));
    match events[0].kind {
        ProjectileEventKind::Passed { thickness } => {
            assert!((thickness - 0.1).abs() < 1e-3, "{thickness}")
        }
        k => panic!("passed the thin wall: {k:?}"),
    }
    assert_eq!(
        (events[1].shape, events[1].kind, events[1].user_data),
        (thick, ProjectileEventKind::Stopped, 3)
    );
    let last = frames
        .iter()
        .flat_map(|f| &f.projectiles)
        .filter(|o| o.projectile == p)
        .last()
        .unwrap();
    assert!(
        !last.alive && (last.position[0] - 9.0).abs() < 1e-2,
        "{last:?}"
    );
    // Reported once more as stopped, then no more.
    assert!(run(&mut w, 3).iter().all(|f| f.projectiles.is_empty()));
    // A stopped projectile's slot is reused with a new handle.
    let q = w
        .spawn_projectile(&ProjectileDef {
            position: [0.0, 2.0, 0.0],
            velocity: [0.0, 0.0, 50.0],
            ..ProjectileDef::default()
        })
        .unwrap();
    assert_eq!(q.index(), p.index());
    assert_ne!(q, p);
}

#[test]
fn projectile_hits_push_dynamic_bodies() {
    let mut w = world();
    let b = w.create_body(&BodyDef::dynamic([5.0, 0.5, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    run(&mut w, 30);
    w.spawn_projectile(&ProjectileDef {
        position: [0.0, 0.5, 0.0],
        velocity: [300.0, 0.0, 0.0],
        mass: 0.05,
        penetration_cost: 1e6,
        ..ProjectileDef::default()
    })
    .unwrap();
    let frames = run(&mut w, 5);
    let hit = frames
        .iter()
        .flat_map(|f| &f.projectile_events)
        .find(|e| e.body == b)
        .expect("hits the box");
    assert_eq!(hit.kind, ProjectileEventKind::Stopped);
    let x = w.body_transform(b).unwrap().unwrap().position[0];
    assert!(x > 5.0 + 1e-3, "the box was pushed along +x: {x}");
}

#[test]
fn bad_definitions_are_refused() {
    let mut w = world();
    let mut def = character_def([0.0, 1.0, 0.0]);
    def.desc.radius = 0.0;
    assert!(w.create_character(&def).is_err());
    let none = QueryFilter {
        categories: 0xFFFF,
        mask: 0,
    };
    assert!(w
        .cast(&CastDef {
            filter: none,
            ..CastDef::default()
        })
        .is_err());
    assert!(w
        .cast(&CastDef {
            direction: [0.0; 3],
            ..CastDef::default()
        })
        .is_err());
    assert!(w
        .spawn_projectile(&ProjectileDef {
            mass: 0.0,
            ..ProjectileDef::default()
        })
        .is_err());
}

/// Sphere and capsule casts touch where the margins meet, also against a
/// floor box tens of metres across (GJK on the cores, the radii as a margin;
/// on the rounded shapes it stalled millimetres short or missed).
#[test]
fn rounded_casts_are_exact_against_large_boxes() {
    for half in [5.0f32, 40.0, 100.0] {
        let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
        let g = w
            .create_body(&BodyDef {
                position: [0.0, -0.5, 0.0],
                ..BodyDef::default()
            })
            .unwrap();
        w.create_box_shape(g, &ShapeDef::default(), [half, 0.5, half])
            .unwrap();
        w.step_sync().unwrap();
        let mut expected = Vec::new();
        for gap in [0.005f32, 0.0167, 0.0367, 0.1] {
            for (shape, hh) in [
                (CastShape::Capsule { radius: 0.3, half_height: 0.6 }, 0.6),
                (CastShape::Sphere { radius: 0.3 }, 0.0),
            ] {
                w.cast(&CastDef {
                    origin: [0.3, 0.3 + hh + gap, -0.2],
                    direction: [0.0, -1.0, 0.0],
                    max_distance: 0.39,
                    shape,
                    ..CastDef::default()
                })
                .unwrap();
                expected.push(gap);
            }
        }
        let f = w.step_sync().unwrap();
        for (r, gap) in f.cast_results.iter().zip(expected) {
            let hit = r.hit.unwrap_or_else(|| panic!("half {half}, gap {gap}: missed"));
            assert!(
                (hit.distance - gap).abs() < 1e-3,
                "half {half}: cast stopped at {} for a {gap} gap",
                hit.distance
            );
            assert!((hit.normal[1] - 1.0).abs() < 1e-4, "{hit:?}");
        }
    }
}
