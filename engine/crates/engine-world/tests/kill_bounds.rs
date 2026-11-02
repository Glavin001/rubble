//! `WorldDef::kill_bounds`: a body that falls out of the world is destroyed and
//! reported once; bodies inside are untouched, and the world can then sleep.

use rubble::*;

fn world(bounds: Option<KillBounds>) -> (World, BodyId, BodyId) {
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            kill_bounds: bounds,
            ..WorldDef::default()
        },
    )
    .unwrap();
    // A 4 m ledge: one box rests on it, one sphere starts beside it and falls.
    let ground = w.create_body(&BodyDef::default()).unwrap();
    w.create_box_shape(ground, &ShapeDef::default(), [2.0, 0.5, 2.0])
        .unwrap();
    let resting = w.create_body(&BodyDef::dynamic([0.0, 1.0, 0.0])).unwrap();
    w.create_box_shape(resting, &ShapeDef::default(), [0.5, 0.5, 0.5])
        .unwrap();
    let falling = w
        .create_body(&BodyDef {
            user_data: 7,
            ..BodyDef::dynamic([6.0, 1.0, 0.0])
        })
        .unwrap();
    w.create_sphere_shape(falling, &ShapeDef::default(), [0.0; 3], 0.25)
        .unwrap();
    (w, resting, falling)
}

#[test]
fn a_falling_body_leaves_the_world_once() {
    let (mut w, resting, falling) = world(Some(KillBounds::below(-20.0)));
    let mut left = Vec::new();
    let mut asleep_steps = 0;
    for _ in 0..600 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            left.extend(f.body_events.left_world);
            if f.counters.is_some_and(|c| c.awake_count == 0) {
                asleep_steps += 1;
            }
        }
    }
    assert_eq!(left.len(), 1, "reported once: {left:?}");
    assert_eq!(left[0].body, falling);
    assert_eq!(left[0].user_data, 7);
    assert!(left[0].position[1] < -20.0, "{:?}", left[0].position);
    assert!(!w.body_is_valid(falling), "destroyed");
    assert!(w.body_is_valid(resting), "the body inside stays");
    let rest = w.read_body_state(resting).unwrap();
    assert!((rest.transform.position[1] - 1.0).abs() < 0.05, "{rest:?}");
    assert!(asleep_steps > 0, "with the faller gone, the world sleeps");
}

#[test]
fn without_bounds_nothing_is_removed() {
    let (mut w, _, falling) = world(None);
    for _ in 0..300 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            assert!(f.body_events.left_world.is_empty());
        }
    }
    assert!(w.body_is_valid(falling));
    assert!(w.read_body_state(falling).unwrap().transform.position[1] < -20.0);
}
