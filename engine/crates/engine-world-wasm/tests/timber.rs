//! The timber yard natively: a cannonball volley breaks the plank wall it hits and nothing
//! it misses, a volley into the house's front breaks the front while its back stands, and
//! the raze wrecks the whole yard.

use engine_world_wasm::timber_yard::*;
use rubble::GpuContext;
use std::time::Instant;

const GLB: &[u8] = include_bytes!("../../engine-webgpu/fracture/assets/timber-yard.glb");

fn level() -> u8 {
    std::env::var("TIMBER_LEVEL").map_or(LEVEL, |s| s.parse().unwrap())
}

/// The yard settled one step, and its rest poses.
fn yard(take: YardTake) -> (TimberYard, Vec<rubble::Transform>) {
    let mut y = TimberYard::new(GpuContext::Default, GLB, level(), take).unwrap();
    y.step().unwrap();
    y.poll_blocking().unwrap();
    let rest = y.chunk_poses();
    (y, rest)
}

/// Run `seconds` of steps; mean ms a step.
fn run(y: &mut TimberYard, seconds: f32) -> f64 {
    let frames = (seconds / y.dt).round() as u32;
    let t0 = Instant::now();
    for _ in 0..frames {
        y.step().unwrap();
        y.poll_blocking().unwrap();
    }
    t0.elapsed().as_secs_f64() * 1e3 / frames.max(1) as f64
}

/// Fire a ball from `STAND_OFF` m in front of `target` (−z) straight at it.
fn shoot_from_front(y: &mut TimberYard, target: [f32; 3]) {
    let from = [target[0], target[1], target[2] - STAND_OFF];
    let d = glam::Vec3::from(target) - glam::Vec3::from(from);
    y.shoot(from, (d.normalize() * BALL_SPEED).to_array());
}

/// Regions by rest position (lo, hi).
fn regions() -> Vec<(&'static str, [f32; 3], [f32; 3])> {
    use yard::*;
    let big = 1.0e3;
    let wall = |x: [f32; 2]| {
        (
            [x[0] - 0.01, WALL_Y[0], WALL_FACE_Z - 0.01],
            [x[1] + 0.01, big, WALL_FACE_Z + WALL_DEPTH + 0.01],
        )
    };
    let (pl, ph) = wall(PLAIN_WALL_X);
    let (wl, wh) = wall(WINDOW_WALL_X);
    vec![
        ("plain wall", pl, ph),
        ("window wall", wl, wh),
        (
            "house",
            [HOUSE_X[0] - 0.5, PLINTH, HOUSE_Z[0] - 0.5],
            [HOUSE_X[1] + 0.5, big, HOUSE_Z[1] + 0.5],
        ),
        // The ground storey's front and back walls (frame and boards).
        (
            "front, ground storey",
            [HOUSE_X[0], PLINTH, HOUSE_Z[0]],
            [HOUSE_X[1], UPPER_FLOOR - 0.3, HOUSE_Z[0] + 0.2],
        ),
        (
            "back, ground storey",
            [HOUSE_X[0], PLINTH, HOUSE_Z[1] - 0.2],
            [HOUSE_X[1], UPPER_FLOOR - 0.3, HOUSE_Z[1]],
        ),
        ("roof", [-big, CEILING + 0.2, -big], [big, big, big]),
        ("whole yard", [-big, PLINTH, -big], [big, big, big]),
    ]
}

/// Each region's chunk count and the share of them loose: crushed, or moved over 0.15 m.
fn loose(y: &TimberYard, rest: &[rubble::Transform]) -> Vec<(&'static str, usize, f32)> {
    let now = y.chunk_poses();
    regions()
        .into_iter()
        .map(|(name, lo, hi)| {
            let inside: Vec<usize> = (0..rest.len())
                .filter(|&c| {
                    (0..3).all(|k| rest[c].position[k] >= lo[k] && rest[c].position[k] <= hi[k])
                })
                .collect();
            let gone = inside
                .iter()
                .filter(|&&c| {
                    y.crushed[c]
                        || (glam::Vec3::from(now[c].position) - glam::Vec3::from(rest[c].position))
                            .length()
                            > 0.15
                })
                .count();
            (name, inside.len(), gone as f32 / inside.len().max(1) as f32)
        })
        .collect()
}

fn share(l: &[(&'static str, usize, f32)], name: &str) -> f32 {
    l.iter().find(|r| r.0 == name).map(|r| r.2).unwrap()
}

fn report(what: &str, y: &TimberYard, ms: f64, l: &[(&'static str, usize, f32)]) {
    eprintln!(
        "{what}: L{} {} chunks, {ms:.1} ms/step (blocking); {} shots, {} bonds broken, {} crushed, {} splinters",
        level(),
        y.chunks,
        y.total.shots,
        y.total.broken,
        y.total.crushed,
        y.splinters.len()
    );
    for (name, n, s) in l {
        eprintln!("  {name}: {:.1}% of {n} loose", 100.0 * s);
    }
}

#[test]
fn a_volley_breaks_the_wall_it_hits_and_nothing_else() {
    let (mut y, rest) = yard(YardTake::Free);
    let x = 0.5 * (yard::PLAIN_WALL_X[0] + yard::PLAIN_WALL_X[1]);
    for (dx, h) in [(-1.0, 0.8), (0.0, 1.4), (1.0, 0.8), (-0.5, 2.0), (0.5, 2.0)] {
        shoot_from_front(&mut y, [x + dx, h, yard::WALL_FACE_Z]);
    }
    let ms = run(&mut y, 2.0);
    let l = loose(&y, &rest);
    report("plain wall volley", &y, ms, &l);
    assert!(y.total.broken > 0, "the balls break bonds");
    assert!(
        share(&l, "plain wall") > 0.05,
        "the plain wall breaks where it is hit"
    );
    assert!(
        share(&l, "window wall") < 0.005,
        "the window wall is untouched"
    );
    assert!(share(&l, "house") < 0.005, "the house is untouched");
}

#[test]
fn a_volley_into_the_front_leaves_the_back_standing() {
    let (mut y, rest) = yard(YardTake::Free);
    // Low on the ground-floor front, between the left window and the door, and around them.
    for (x, h) in [
        (0.6, 0.6),
        (3.0, 0.6),
        (0.6, 1.7),
        (3.0, 1.7),
        (1.8, 2.5),
        (7.4, 0.6),
        (5.2, 1.0),
        (7.4, 1.8),
    ] {
        shoot_from_front(&mut y, [x, h, yard::HOUSE_Z[0]]);
    }
    let ms = run(&mut y, 3.0);
    let l = loose(&y, &rest);
    report("house front volley", &y, ms, &l);
    // Each ball punches a hole through boards and studs (about 2.5% of the storey for eight).
    assert!(
        share(&l, "front, ground storey") > 0.01,
        "the front breaks where it is hit"
    );
    assert!(
        share(&l, "plain wall") < 0.005 && share(&l, "window wall") < 0.005,
        "the range walls are untouched"
    );
    assert!(
        share(&l, "back, ground storey") < share(&l, "front, ground storey"),
        "the back breaks less than the front"
    );
}

#[test]
fn the_raze() {
    let (mut y, rest) = yard(YardTake::Raze);
    let mut ms = 0.0;
    let mut seconds = 0.0;
    while !y.over() && seconds < RAZE_SECONDS + 10.0 {
        ms += run(&mut y, 1.0);
        seconds += 1.0;
    }
    let l = loose(&y, &rest);
    report("raze", &y, ms / seconds as f64, &l);
    assert!(y.finale.is_some(), "the closing volley fired");
    assert!(share(&l, "whole yard") > 0.8, "the yard is wrecked");
    for name in ["plain wall", "window wall", "house", "roof"] {
        assert!(share(&l, name) > 0.5, "{name} is wrecked");
    }
}

/// Every level loads into its world (the capacity fits its colliders: a coarse level's
/// chunks are compounds of several hulls) and steps with a shot in flight.
#[test]
fn every_level_loads_and_steps() {
    for level in 0..=2 {
        let mut y = TimberYard::new(GpuContext::Default, GLB, level, YardTake::Free).unwrap();
        shoot_from_front(&mut y, [-3.0, 1.2, yard::WALL_FACE_Z]);
        run(&mut y, 0.5);
        eprintln!(
            "L{level}: {} chunks, {} bonds broken",
            y.chunks, y.total.broken
        );
        assert!(y.total.broken > 0, "L{level}: the shot breaks bonds");
    }
}
