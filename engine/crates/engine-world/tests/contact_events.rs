//! Contact and hit events (P4) from their definitions: a touch begins once,
//! ends when the shapes separate or one is destroyed, never on sleep; a hit
//! is a begin at least `hit_event_threshold` fast, its approach speed the
//! closing speed at impact (√(2gh) for a drop from rest through h).

use rubble::*;
use std::collections::BTreeMap;

const G: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;

fn world_with(contacts: ContactStorage, max_bodies: u32, threshold: f32) -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies,
            max_colliders: max_bodies + 64,
            max_contacts: 8 * max_bodies + 4096,
            max_joints: 16,
            max_candidates: 8 * max_bodies + 8192,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 64,
        },
        contacts,
        hit_event_threshold: threshold,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -G, 0.0]).unwrap();
    w
}

fn world(threshold: f32) -> World {
    world_with(ContactStorage::Persistent, 256, threshold)
}

fn events_def() -> ShapeDef {
    ShapeDef {
        enable_contact_events: true,
        enable_hit_events: true,
        ..ShapeDef::default()
    }
}

fn ground_plane(w: &mut World, def: &ShapeDef) -> ShapeId {
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(
        g,
        def,
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )
    .unwrap()
}

fn ground_field(w: &mut World, def: &ShapeDef) -> ShapeId {
    let g = w.create_body(&BodyDef::default()).unwrap();
    let field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    w.create_height_field_shape(g, def, &field, Transform::IDENTITY)
        .unwrap()
}

fn falling_box(w: &mut World, def: &ShapeDef, at: [f32; 3], half: f32) -> (BodyId, ShapeId) {
    let b = w.create_body(&BodyDef::dynamic(at)).unwrap();
    let s = w.create_box_shape(b, def, [half; 3]).unwrap();
    (b, s)
}

/// Every frame's contact events over `steps` steps.
fn run(w: &mut World, steps: u32) -> Vec<Frame> {
    let mut frames = Vec::new();
    for _ in 0..steps {
        w.step().unwrap();
        frames.extend(w.poll().unwrap());
    }
    frames.extend(w.poll_blocking().unwrap());
    frames
}

#[derive(Default, Debug)]
struct Tally {
    begins: Vec<(u64, ContactBeginEvent)>,
    ends: Vec<(u64, ContactEndEvent)>,
    hits: Vec<(u64, ContactHitEvent)>,
}

/// Events per shape pair (lower handle first, as reported).
fn tally(frames: &[Frame]) -> BTreeMap<(ShapeId, ShapeId), Tally> {
    let mut out: BTreeMap<_, Tally> = BTreeMap::new();
    for f in frames {
        let e = &f.contact_events;
        for b in &e.begins {
            assert!(b.shape_a < b.shape_b);
            out.entry((b.shape_a, b.shape_b))
                .or_default()
                .begins
                .push((f.step, *b));
        }
        for x in &e.ends {
            out.entry((x.shape_a, x.shape_b))
                .or_default()
                .ends
                .push((f.step, *x));
        }
        for h in &e.hits {
            out.entry((h.shape_a, h.shape_b))
                .or_default()
                .hits
                .push((f.step, *h));
        }
    }
    out
}

fn pair(a: ShapeId, b: ShapeId) -> (ShapeId, ShapeId) {
    (a.min(b), a.max(b))
}

/// Drop heights (gap under the box) that put the first overlap at every phase
/// of a step: the touch must not depend on where in a step it lands.
const HEIGHTS: [f32; 8] = [0.3, 0.47, 0.61, 0.8, 1.05, 1.33, 1.6, 2.05];

/// One step of gravity (the speed resolution of a fixed step) plus 2%.
fn speed_tolerance(v: f32) -> f32 {
    G * DT + 0.02 * v
}

fn drop_onto(contacts: ContactStorage, terrain: bool) {
    drop_onto_with(contacts, terrain, 0);
}

/// `flags`: extra solver flags (`FLAG_CONTACT_GAP`: the zero-gap contacts
/// destructible packs and houses use).
fn drop_onto_with(contacts: ContactStorage, terrain: bool, flags: u32) {
    let mut w = world_with(contacts, 256, 0.5);
    if flags != 0 {
        let mut p = w.solver();
        p.flags |= flags;
        w.set_solver(p);
    }
    let ground = if terrain {
        ground_field(&mut w, &ShapeDef::default())
    } else {
        ground_plane(&mut w, &ShapeDef::default())
    };
    let half = 0.25;
    let boxes: Vec<_> = HEIGHTS
        .iter()
        .enumerate()
        .map(|(k, h)| {
            let x = -7.0 + 2.0 * k as f32;
            falling_box(&mut w, &events_def(), [x, half + h, 0.0], half).1
        })
        .collect();
    assert!(w.sim().has_contact_events());
    let frames = run(&mut w, 150);
    let t = tally(&frames);
    for (h, s) in HEIGHTS.iter().zip(&boxes) {
        let e = t.get(&pair(ground, *s)).unwrap_or_else(|| {
            panic!("{contacts:?} terrain={terrain}: no events for the box dropped {h} m")
        });
        let v = (2.0 * G * h).sqrt();
        assert_eq!(
            e.begins.len(),
            1,
            "{contacts:?} h={h}: one begin, got {:?}",
            e.begins
        );
        assert!(
            e.ends.is_empty(),
            "{contacts:?} h={h}: the box rests (asleep), no end: {:?}",
            e.ends
        );
        assert_eq!(
            e.hits.len(),
            1,
            "{contacts:?} h={h}: one hit (threshold 0.5 m/s, impact {v} m/s)"
        );
        let (step, hit) = e.hits[0];
        assert_eq!(step, e.begins[0].0, "the hit is the begin");
        let tol = speed_tolerance(v);
        assert!(
            (hit.approach_speed - v).abs() <= tol,
            "{contacts:?} terrain={terrain} h={h}: approach {} m/s, √(2gh) = {v} (±{tol})",
            hit.approach_speed
        );
        // Normal from the lower handle to the higher: the ground was made first.
        assert!(ground < *s);
        assert!(
            hit.normal[1] > 0.99,
            "normal {:?} points up, ground to box",
            hit.normal
        );
        assert!(
            hit.point[1].abs() < 0.1,
            "point {:?} at the ground",
            hit.point
        );
    }
}

#[test]
fn a_box_dropped_on_a_plane_begins_once_and_hits_at_the_impact_speed() {
    drop_onto(ContactStorage::Persistent, false);
}

#[test]
fn a_box_dropped_on_a_heightfield_begins_once_and_hits_at_the_impact_speed() {
    drop_onto(ContactStorage::Persistent, true);
}

#[test]
fn every_contact_storage_reports_the_same_touch() {
    for storage in [ContactStorage::Canonical, ContactStorage::Compact] {
        drop_onto(storage, false);
    }
}

/// `engine_schema::FLAG_CONTACT_GAP` (zero-gap contacts, as destructible
/// houses and packs use): resting contacts sit at zero gap instead of a
/// margin deep, and touches are reported the same way.
const FLAG_CONTACT_GAP: u32 = 32;

#[test]
fn zero_gap_contacts_report_the_same_touch() {
    drop_onto_with(ContactStorage::Persistent, false, FLAG_CONTACT_GAP);
    drop_onto_with(ContactStorage::Persistent, true, FLAG_CONTACT_GAP);
}

#[test]
fn a_box_set_down_on_a_box_touches_in_both_contact_modes() {
    // Placed at each mode's rest depth (a margin deep, or at zero gap), where
    // the contact's c0N is 0: it touches from the first step, once.
    for (flags, y) in [(0, 1.24), (FLAG_CONTACT_GAP, 1.25)] {
        let mut w = world(0.5);
        let mut p = w.solver();
        p.flags |= flags;
        w.set_solver(p);
        let base = w
            .create_body(&BodyDef {
                position: [0.0, 0.5, 0.0],
                ..BodyDef::default()
            })
            .unwrap();
        let ledge = w
            .create_box_shape(base, &ShapeDef::default(), [1.0, 0.5, 1.0])
            .unwrap();
        let (b, s) = falling_box(&mut w, &events_def(), [0.0, y, 0.0], 0.25);
        let t = tally(&run(&mut w, 240));
        let e = &t[&pair(ledge, s)];
        assert_eq!(e.begins.len(), 1, "flags {flags}: {:?}", e.begins);
        assert_eq!(e.begins[0].0, 0, "touching from the first step");
        assert!(
            e.ends.is_empty() && e.hits.is_empty(),
            "flags {flags}: {e:?}"
        );
        assert!(!w.body_is_awake(b).unwrap());
    }
}

#[test]
fn no_hit_below_the_threshold() {
    let h = 1.0;
    let v = (2.0 * G * h).sqrt();
    let mut w = world(v + 1.0);
    let ground = ground_plane(&mut w, &ShapeDef::default());
    let (_, s) = falling_box(&mut w, &events_def(), [0.0, 0.25 + h, 0.0], 0.25);
    let t = tally(&run(&mut w, 90));
    let e = &t[&pair(ground, s)];
    assert_eq!(e.begins.len(), 1);
    assert!(
        e.hits.is_empty(),
        "impact {v} m/s is under the threshold {}: {:?}",
        v + 1.0,
        e.hits
    );
    assert!(e.begins[0].1.approach_speed < v + 1.0);
}

#[test]
fn hit_events_alone_report_hits_without_begins() {
    let mut w = world(0.5);
    let ground = ground_plane(&mut w, &ShapeDef::default());
    let def = ShapeDef {
        enable_hit_events: true,
        ..ShapeDef::default()
    };
    let (_, s) = falling_box(&mut w, &def, [0.0, 1.25, 0.0], 0.25);
    let t = tally(&run(&mut w, 90));
    let e = &t[&pair(ground, s)];
    assert!(e.begins.is_empty() && e.ends.is_empty());
    assert_eq!(e.hits.len(), 1);
}

#[test]
fn resting_bodies_do_not_begin_again() {
    let mut w = world(0.5);
    let ground = ground_plane(&mut w, &ShapeDef::default());
    // Already resting on the ground: one begin in the first step, then
    // nothing while it sits (and after it falls asleep).
    let (b, s) = falling_box(&mut w, &events_def(), [0.0, 0.24, 0.0], 0.25);
    let frames = run(&mut w, 400);
    let t = tally(&frames);
    let e = &t[&pair(ground, s)];
    assert_eq!(e.begins.len(), 1, "{:?}", e.begins);
    assert!(e.ends.is_empty(), "{:?}", e.ends);
    assert!(
        e.hits.is_empty(),
        "a box set down at rest hits nothing: {:?}",
        e.hits
    );
    assert!(!w.body_is_awake(b).unwrap(), "the box fell asleep");
    // Waking it does not begin the touch again either.
    w.body_set_awake(b, true).unwrap();
    let again = tally(&run(&mut w, 30));
    assert!(again.is_empty(), "{again:?}");
}

#[test]
fn a_box_sliding_off_a_ledge_ends_its_touch() {
    let mut w = world(0.5);
    let ice = ShapeDef {
        material: SurfaceMaterial {
            friction: 0.0,
            ..SurfaceMaterial::default()
        },
        ..events_def()
    };
    // A 2 m ledge, its top at y = 1, its right edge at x = 1.
    let ledge_body = w
        .create_body(&BodyDef {
            position: [0.0, 0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    let ledge = w
        .create_box_shape(ledge_body, &ice, [1.0, 0.5, 1.0])
        .unwrap();
    let floor = ground_plane(&mut w, &ShapeDef::default());
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [3.0, 0.0, 0.0],
            ..BodyDef::dynamic([0.0, 1.24, 0.0])
        })
        .unwrap();
    let s = w.create_box_shape(b, &ice, [0.25; 3]).unwrap();
    let t = tally(&run(&mut w, 120));
    let on_ledge = &t[&pair(ledge, s)];
    assert_eq!(on_ledge.begins.len(), 1, "{:?}", on_ledge.begins);
    assert_eq!(on_ledge.ends.len(), 1, "slid off the edge: one end");
    let (end_step, _) = on_ledge.ends[0];
    assert!(end_step > on_ledge.begins[0].0);
    // It slid ~1 m at 3 m/s before tipping off: past step 15, before it lands.
    let on_floor = &t[&pair(floor, s)];
    assert_eq!(on_floor.begins.len(), 1);
    assert!(
        end_step >= 15 && end_step < on_floor.begins[0].0,
        "ended at {end_step}"
    );
}

#[test]
fn destroying_a_touching_shape_ends_its_touch() {
    let mut w = world(0.5);
    let ground = ground_plane(&mut w, &ShapeDef::default());
    let (b, s) = falling_box(&mut w, &events_def(), [0.0, 0.24, 0.0], 0.25);
    let t = tally(&run(&mut w, 10));
    assert_eq!(t[&pair(ground, s)].begins.len(), 1);
    w.destroy_body(b).unwrap();
    let t = tally(&run(&mut w, 10));
    let e = &t[&pair(ground, s)];
    assert_eq!(e.ends.len(), 1, "{e:?}");
    assert!(e.begins.is_empty());
    assert!(
        !w.shape_is_valid(e.ends[0].1.shape_b),
        "the box's handle is stale"
    );
}

#[test]
fn shape_flags_keep_the_material() {
    // A box thrown along grippy ground stops within v²/(2μg) whether or not
    // its shape reports events: the flags (material bits 12..15) never reach
    // the material table lookup.
    let mut stops = Vec::new();
    for flagged in [false, true] {
        let mut w = world(0.5);
        let grip = SurfaceMaterial {
            friction: 1.0,
            ..SurfaceMaterial::default()
        };
        let def = ShapeDef {
            material: grip,
            enable_contact_events: flagged,
            enable_hit_events: flagged,
            ..ShapeDef::default()
        };
        ground_plane(
            &mut w,
            &ShapeDef {
                material: grip,
                ..ShapeDef::default()
            },
        );
        let b = w
            .create_body(&BodyDef {
                linear_velocity: [3.0, 0.0, 0.0],
                ..BodyDef::dynamic([0.0, 0.24, 0.0])
            })
            .unwrap();
        w.create_box_shape(b, &def, [0.25; 3]).unwrap();
        assert_eq!(w.sim().has_contact_events(), flagged);
        run(&mut w, 120);
        stops.push(w.body_transform(b).unwrap().unwrap().position[0]);
    }
    let bound = 3.0f32 * 3.0 / (2.0 * 1.0 * G);
    for x in &stops {
        assert!(*x < 1.5 * bound, "stopped at {x} m (μ = 1 bound {bound} m)");
    }
    assert_eq!(stops[0], stops[1], "flags change nothing the solver reads");
}

#[test]
fn overflow_is_counted_not_silent() {
    // More touching manifolds than CONTACT_EVENT_CAP in one step: spheres set
    // into the ground, every one touching from the first step.
    let n = CONTACT_EVENT_CAP + 100;
    let side = (n as f32).sqrt().ceil() as u32;
    let mut w = world_with(ContactStorage::Persistent, n + 16, 0.5);
    let ground = ground_plane(&mut w, &ShapeDef::default());
    let def = ShapeDef {
        enable_contact_events: true,
        ..ShapeDef::default()
    };
    let r = 0.2;
    let defs: Vec<BodyDef> = (0..n)
        .map(|k| {
            let (i, j) = (k % side, k / side);
            BodyDef::dynamic([i as f32 * 0.5, r - 0.02, j as f32 * 0.5])
        })
        .collect();
    let bodies = w.create_bodies(&defs).unwrap();
    for b in &bodies {
        w.create_sphere_shape(b.to_owned(), &def, [0.0; 3], r)
            .unwrap();
    }
    let frames = run(&mut w, 3);
    let first = frames.iter().find(|f| f.ran).unwrap();
    assert!(
        first.lost_events >= 100,
        "{} touching, {} reported: lost {}",
        n,
        first.contact_events.begins.len(),
        first.lost_events
    );
    assert_eq!(first.contact_events.begins.len() as u32, CONTACT_EVENT_CAP);
    // A frame that lost records ends nothing.
    for f in &frames {
        if f.lost_events > 0 {
            assert!(f.contact_events.ends.is_empty());
        }
    }
    let _ = ground;
}
