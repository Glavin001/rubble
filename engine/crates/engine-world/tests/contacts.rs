//! Contact begin, end and hit events (P4), in persistent and canonical
//! contact storage.

use rubble::*;

fn world(contacts: ContactStorage) -> (World, ShapeId) {
    let def = WorldDef {
        contacts,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let ground = w
        .create_box_shape(g, &ShapeDef::default(), [20.0, 0.5, 20.0])
        .unwrap();
    (w, ground)
}

fn events_def() -> ShapeDef {
    ShapeDef {
        enable_contact_events: true,
        enable_hit_events: true,
        ..ShapeDef::default()
    }
}

fn run(w: &mut World, steps: u32) -> ContactEvents {
    let mut all = ContactEvents::default();
    for _ in 0..steps {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            all.begin.extend(f.contact_events.begin);
            all.end.extend(f.contact_events.end);
            all.hits.extend(f.contact_events.hits);
        }
    }
    for f in w.poll_blocking().unwrap() {
        all.begin.extend(f.contact_events.begin);
        all.end.extend(f.contact_events.end);
        all.hits.extend(f.contact_events.hits);
    }
    all
}

fn involves(e: &ContactBeginEvent, a: ShapeId, b: ShapeId) -> bool {
    (e.shape_a == a && e.shape_b == b) || (e.shape_a == b && e.shape_b == a)
}

fn drop_box(contacts: ContactStorage) {
    let (mut w, ground) = world(contacts);
    let b = w.create_body(&BodyDef::dynamic([0.0, 2.5, 0.0])).unwrap();
    let s = w.create_box_shape(b, &events_def(), [0.5; 3]).unwrap();
    let e = run(&mut w, 240);
    assert_eq!(e.begin.len(), 1, "{contacts:?}: one begin: {:?}", e.begin);
    let begin = e.begin[0];
    assert!(involves(&begin, s, ground));
    // The normal points from A to B: up when the ground is A.
    let up = if begin.shape_a == ground { 1.0 } else { -1.0 };
    assert!(
        begin.normal[1] * up > 0.9,
        "{contacts:?}: normal {:?}",
        begin.normal
    );
    assert!(
        begin.point[1].abs() < 0.1,
        "{contacts:?}: at the floor: {:?}",
        begin.point
    );
    // Fell 2 m: about sqrt(2 g h) = 6.3 m/s.
    assert_eq!(e.hits.len(), 1, "{contacts:?}: one hit");
    let v = e.hits[0].approach_speed;
    assert!((v - 6.3).abs() < 0.8, "{contacts:?}: approach speed {v}");
    assert!(
        e.end.is_empty(),
        "{contacts:?}: no end while resting or asleep"
    );
    assert!(!w.body_is_awake(b).unwrap(), "{contacts:?}: it fell asleep");

    // Lift it off: the pair ends.
    w.body_set_transform(b, Transform::new([0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let e = run(&mut w, 5);
    assert_eq!(e.end.len(), 1, "{contacts:?}: one end: {:?}", e.end);
    assert!(e.begin.is_empty());
}

#[test]
fn a_dropped_box_begins_hits_and_ends_persistent() {
    drop_box(ContactStorage::Persistent);
}

#[test]
fn a_dropped_box_begins_hits_and_ends_canonical() {
    drop_box(ContactStorage::Canonical);
}

#[test]
fn slow_touches_begin_without_hits_and_destroyed_shapes_end() {
    let (mut w, ground) = world(ContactStorage::Persistent);
    let b = w.create_body(&BodyDef::dynamic([0.0, 0.51, 0.0])).unwrap();
    let s = w.create_box_shape(b, &events_def(), [0.5; 3]).unwrap();
    let e = run(&mut w, 30);
    assert_eq!(e.begin.len(), 1);
    assert!(e.hits.is_empty(), "too slow to be a hit: {:?}", e.begin);
    w.destroy_shape(s, true).unwrap();
    let e = run(&mut w, 2);
    assert_eq!(e.end.len(), 1, "destroying a touching shape ends it");
    assert!(
        e.end[0].shape_a == ground || e.end[0].shape_b == ground,
        "{:?}",
        e.end
    );
}

#[test]
fn shapes_without_flags_report_nothing() {
    let (mut w, _) = world(ContactStorage::Persistent);
    let b = w.create_body(&BodyDef::dynamic([0.0, 2.0, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    let e = run(&mut w, 120);
    assert!(e.begin.is_empty() && e.end.is_empty() && e.hits.is_empty());
}
