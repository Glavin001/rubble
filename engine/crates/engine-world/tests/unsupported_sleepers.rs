//! Nothing is left in the air: a sleeping body whose support is taken away wakes and
//! falls, though no awake body touches it.
//!
//! A sleeping body is as good as fixed in space, so whatever rests on it stays where it is
//! for as long as it sleeps. Removing a support is not a contact: a wake rule that only
//! fires on a touch from an awake body misses it, and the piece above hangs. The cases here
//! are that failure's simplest forms, each stated as what must be true afterwards.
//!
//! **All three fail today (2026-11-12) and are ignored until the engine wakes what rested on
//! a destroyed body:** each sleeper stays exactly where it was, in the air
//! (`engine/docs/lattice-demo-audit-2026-11-12.md`, "Finding: sleepers left in the air").
//!
//!   cargo test --release -p engine-world --test unsupported_sleepers -- --ignored --test-threads=1 --nocapture

use rubble::*;

const HALF: f32 = 0.5;

fn world() -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    let mut solver = w.solver();
    solver.sleep_time = 0.5;
    w.set_solver(solver);
    let ground = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(
        ground,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )
    .unwrap();
    w
}

/// A unit cube with its centre at height `y` over the origin.
fn cube(w: &mut World, y: f32, body_type: BodyType) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            body_type,
            ..BodyDef::dynamic([0.0, y, 0.0])
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [HALF; 3]).unwrap();
    b
}

fn run(w: &mut World, steps: u32) {
    for _ in 0..steps {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
}

fn height(w: &World, b: BodyId) -> f32 {
    w.body_transform(b).unwrap().expect("a polled pose").position[1]
}

/// Steps until every one of `bodies` sleeps (at most 600), then 30 more.
fn settle(w: &mut World, bodies: &[BodyId]) {
    for _ in 0..600 {
        run(w, 1);
        if bodies.iter().all(|&b| !w.body_is_awake(b).unwrap()) {
            run(w, 30);
            return;
        }
    }
    panic!("the bodies never fell asleep");
}

/// A cube asleep on a static pedestal; the pedestal is destroyed. The cube comes down to the
/// ground.
#[test]
#[ignore = "fails: a sleeper is left in the air when its support is destroyed"]
fn a_sleeper_falls_when_its_static_support_is_destroyed() {
    let mut w = world();
    let pedestal = cube(&mut w, HALF, BodyType::Static);
    let top = cube(&mut w, 3.0 * HALF + 0.01, BodyType::Dynamic);
    settle(&mut w, &[top]);
    let rested = height(&w, top);
    assert!((rested - 3.0 * HALF).abs() < 0.05, "it rests on the pedestal: {rested}");
    w.destroy_body(pedestal).unwrap();
    run(&mut w, 240);
    let after = height(&w, top);
    eprintln!("static support destroyed: {rested:.3} m -> {after:.3} m");
    assert!((after - HALF).abs() < 0.05, "the cube is left in the air at {after} m (it rested at {rested} m)");
}

/// Three cubes stacked and asleep; the bottom one (dynamic, asleep) is destroyed. Both above
/// come down by its height: the wake-up reaches the one that never touched it.
#[test]
#[ignore = "fails: a sleeper is left in the air when its support is destroyed"]
fn a_sleeping_stack_falls_when_its_bottom_sleeper_is_destroyed() {
    let mut w = world();
    let bottom = cube(&mut w, HALF + 0.01, BodyType::Dynamic);
    let middle = cube(&mut w, 3.0 * HALF + 0.02, BodyType::Dynamic);
    let top = cube(&mut w, 5.0 * HALF + 0.03, BodyType::Dynamic);
    settle(&mut w, &[bottom, middle, top]);
    let rested = [height(&w, middle), height(&w, top)];
    w.destroy_body(bottom).unwrap();
    run(&mut w, 300);
    let after = [height(&w, middle), height(&w, top)];
    eprintln!("bottom sleeper destroyed: middle {:.3} -> {:.3} m, top {:.3} -> {:.3} m", rested[0], after[0], rested[1], after[1]);
    assert!((after[0] - HALF).abs() < 0.05, "the middle cube is left at {} m", after[0]);
    assert!((after[1] - 3.0 * HALF).abs() < 0.08, "the top cube is left at {} m", after[1]);
}

/// A plank asleep across two static posts; one post is destroyed. The plank cannot stay
/// level on one post: its free end comes down.
#[test]
#[ignore = "fails: a sleeper is left in the air when its support is destroyed"]
fn a_sleeping_plank_tips_when_one_of_its_two_posts_is_destroyed() {
    let mut w = world();
    let post = |w: &mut World, x: f32| {
        let b = w
            .create_body(&BodyDef {
                body_type: BodyType::Static,
                ..BodyDef::dynamic([x, 1.0, 0.0])
            })
            .unwrap();
        w.create_box_shape(b, &ShapeDef::default(), [0.2, 1.0, 0.2]).unwrap();
        b
    };
    let (left, _right) = (post(&mut w, -1.5), post(&mut w, 1.5));
    let plank = w.create_body(&BodyDef::dynamic([0.0, 2.11, 0.0])).unwrap();
    w.create_box_shape(plank, &ShapeDef::default(), [1.9, 0.1, 0.3]).unwrap();
    settle(&mut w, &[plank]);
    let rested = height(&w, plank);
    w.destroy_body(left).unwrap();
    run(&mut w, 300);
    let after = height(&w, plank);
    eprintln!("one post destroyed: plank centre {rested:.3} m -> {after:.3} m");
    assert!(after < rested - 0.3, "the plank still lies level at {after} m on one post");
}
