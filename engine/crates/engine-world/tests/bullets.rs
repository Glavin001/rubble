//! Bullet bodies (`BodyDef::is_bullet`, the kernels' continuous collision):
//! fast bodies do not tunnel through thin walls, and still pass through
//! sensors.

use rubble::*;

fn world() -> World {
    let def = WorldDef {
        gravity: [0.0, 0.0, 0.0],
        ..WorldDef::default()
    };
    World::new(GpuContext::Default, &def).unwrap()
}

/// A thin static wall across x = 5 (2 cm thick), or a sensor volume there.
fn wall(w: &mut World, sensor: bool) -> ShapeId {
    let b = w
        .create_body(&BodyDef {
            position: [5.0, 0.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let def = ShapeDef {
        is_sensor: sensor,
        enable_sensor_events: sensor,
        ..ShapeDef::default()
    };
    w.create_box_shape(b, &def, [0.01, 3.0, 3.0]).unwrap()
}

/// A 5 cm sphere at 200 m/s along +x (3.3 m a step at 60 Hz).
fn shot(w: &mut World, bullet: bool) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [200.0, 0.0, 0.0],
            is_bullet: bullet,
            ..BodyDef::dynamic([0.0, 0.0, 0.0])
        })
        .unwrap();
    w.create_sphere_shape(b, &ShapeDef::default(), [0.0; 3], 0.05)
        .unwrap();
    b
}

fn run(w: &mut World, steps: u32) -> SensorEvents {
    let mut events = SensorEvents::default();
    for _ in 0..steps {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            events.begin.extend(f.sensor_events.begin);
            events.end.extend(f.sensor_events.end);
        }
    }
    for f in w.poll_blocking().unwrap() {
        events.begin.extend(f.sensor_events.begin);
        events.end.extend(f.sensor_events.end);
    }
    events
}

fn x(w: &World, b: BodyId) -> f32 {
    w.read_body_state(b).unwrap().transform.position[0]
}

#[test]
fn plain_fast_bodies_tunnel_and_bullets_do_not() {
    let mut w = world();
    wall(&mut w, false);
    let plain = shot(&mut w, false);
    run(&mut w, 10);
    assert!(
        x(&w, plain) > 5.5,
        "without CCD it passes the wall: {}",
        x(&w, plain)
    );

    let mut w = world();
    wall(&mut w, false);
    let bullet = shot(&mut w, true);
    run(&mut w, 10);
    assert!(
        x(&w, bullet) < 5.0,
        "a bullet stops at the wall: {}",
        x(&w, bullet)
    );
}

#[test]
fn bullets_pass_through_sensors() {
    let mut w = world();
    let sensor = wall(&mut w, true);
    let bullet = shot(&mut w, true);
    let events = run(&mut w, 10);
    assert!(
        x(&w, bullet) > 15.0,
        "sensors do not stop bullets: {}",
        x(&w, bullet)
    );
    assert!(
        events.begin.iter().all(|e| e.sensor_shape == sensor),
        "{events:?}"
    );
}

#[test]
fn a_bullet_hit_is_one_begin_at_the_impact_speed() {
    let mut w = world();
    let wall_shape = wall(&mut w, false);
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [200.0, 0.0, 0.0],
            is_bullet: true,
            ..BodyDef::dynamic([0.0, 0.0, 0.0])
        })
        .unwrap();
    let def = ShapeDef {
        enable_contact_events: true,
        enable_hit_events: true,
        ..ShapeDef::default()
    };
    let s = w.create_sphere_shape(b, &def, [0.0; 3], 0.05).unwrap();
    let mut begins = Vec::new();
    let mut hits = Vec::new();
    for _ in 0..10 {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            begins.extend(f.contact_events.begin);
            hits.extend(f.contact_events.hits);
        }
    }
    for f in w.poll_blocking().unwrap() {
        begins.extend(f.contact_events.begin);
        hits.extend(f.contact_events.hits);
    }
    assert_eq!(begins.len(), 1, "{begins:?}");
    let e = begins[0];
    assert!([e.shape_a, e.shape_b].contains(&s) && [e.shape_a, e.shape_b].contains(&wall_shape));
    assert_eq!(hits.len(), 1);
    assert!(hits[0].approach_speed > 150.0, "{:?}", hits[0]);
}
