//! What vibe-land's rubble bridge does through `WgpuSim` (blocking read-modify-
//! writes, parking bodies for removal), done through the world API: its
//! acceptance cases, with commands instead of readbacks and removal instead of
//! parking.

use glam::Vec3;
use rubble::*;

const G: f32 = 9.81;
const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn world(gravity: f32) -> World {
    world_with(gravity, EngineFeatures::default())
}

fn world_with(gravity: f32, features: EngineFeatures) -> World {
    let mut w = World::new(
        GpuContext::Default,
        WorldConfig {
            capacity: WorldCapacity {
                max_candidates: 131_072,
                ..WorldCapacity::small()
            },
            gravity: [0.0, gravity, 0.0],
            features,
            ..WorldConfig::default()
        },
    )
    .expect("wgpu device");
    w.set_material(
        MaterialId::DEFAULT,
        MaterialDef {
            friction: 0.6,
            ..MaterialDef::default()
        },
    )
    .unwrap();
    w
}

fn ground(w: &mut World) -> BodyId {
    // Top face at y = 0.
    w.create_body(
        &BodyDef {
            kind: BodyKind::Static,
            position: [0.0, -5.0, 0.0],
            ..BodyDef::default()
        },
        &[ShapeDef::new(Shape::Box {
            half: [200.0, 5.0, 200.0],
        })],
    )
    .unwrap()
}

fn sphere(w: &mut World, position: [f32; 3], radius: f32, mass: f32) -> BodyId {
    w.create_body(
        &BodyDef {
            position,
            mass: MassSpec::Mass(mass),
            ..BodyDef::default()
        },
        &[ShapeDef::new(Shape::Sphere { radius })],
    )
    .unwrap()
}

fn steps(w: &mut World, n: u32) -> Vec<Frame> {
    for _ in 0..n {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap()
}

fn state(w: &World, id: BodyId) -> engine_schema::BodyState {
    w.sim().read_body(id.index())
}

#[test]
fn a_central_impulse_changes_momentum_by_j() {
    let mut w = world(0.0);
    let m = 4.0;
    let ball = sphere(&mut w, [0.0, 10.0, 0.0], 0.5, m);
    w.apply_impulse(ball, [8.0, 0.0, 0.0], [0.0, 10.0, 0.0])
        .unwrap();
    steps(&mut w, 1);
    let b = state(&w, ball);
    assert!(
        (b.linear_vel[0] - 8.0 / m).abs() < 1e-5,
        "{:?}",
        b.linear_vel
    );
    assert!(Vec3::from(b.angular_vel).length() < 1e-6);
}

#[test]
fn free_fall_matches_the_implicit_integrator() {
    let mut w = world(-G);
    ground(&mut w);
    let ball = sphere(&mut w, [0.0, 50.0, 0.0], 0.5, 1.0);
    let n = 30;
    steps(&mut w, n);
    let dt = AvbdParams::default().dt;
    let expected = 50.0 - G * dt * dt * (n * (n + 1)) as f32 / 2.0;
    let y = w.pose(ball).unwrap().unwrap().position[1];
    assert!((y - expected).abs() < 1e-3, "y {y} vs {expected}");
}

#[test]
fn a_removed_sphere_is_gone_and_its_slot_comes_back_as_a_new_body() {
    let mut w = world(-G);
    ground(&mut w);
    let ball = sphere(&mut w, [0.0, 3.0, 0.0], 0.5, 2.0);
    steps(&mut w, 1);
    w.remove_body(ball).unwrap();
    let frames = steps(&mut w, 10);
    assert!(
        frames
            .iter()
            .all(|f| f.moved.iter().all(|m| m.body != ball)),
        "a removed body is never reported"
    );
    let parked = state(&w, ball);
    assert!(parked.inv_mass == 0.0 && parked.flags == 0, "{parked:?}");
    let again = sphere(&mut w, [4.0, 5.0, 0.0], 0.5, 2.0);
    assert_eq!(
        again.index(),
        ball.index(),
        "the quarantined slot is reused"
    );
    assert_ne!(again, ball);
    let frames = steps(&mut w, 10);
    let first = frames[0].moved.iter().find(|m| m.body == again).unwrap();
    assert!(
        first.previous.is_none(),
        "a reused slot has no pose to interpolate from"
    );
    let p = w.pose(again).unwrap().unwrap().position;
    assert!(p[1] < 5.0 && (p[0] - 4.0).abs() < 1e-4, "{p:?}");
}

#[test]
fn a_city_of_large_statics_steps_beside_small_balls() {
    // The hierarchical grid, as the bridge sets it: a uniform grid sized from
    // the 0.3 m ball puts every 2 m chunk in the oversized-static list, and the
    // step then fails with OVERFLOW_LARGE_STATICS (loudly, never dropped).
    let mut w = world_with(
        -G,
        EngineFeatures {
            grid_levels: Some(true),
            ..EngineFeatures::default()
        },
    );
    ground(&mut w);
    for i in 0..1100 {
        let (x, z) = ((i % 40) as f32 * 4.0 - 80.0, (i / 40) as f32 * 4.0 - 60.0);
        w.create_body(
            &BodyDef {
                kind: BodyKind::Static,
                position: [x, 1.0, z],
                ..BodyDef::default()
            },
            &[ShapeDef::new(Shape::Box { half: [1.0; 3] })],
        )
        .unwrap();
    }
    // Above the box at (0, 1, 0), whose top is y = 2.
    let ball = sphere(&mut w, [0.0, 4.0, 0.0], 0.3, 50.0);
    steps(&mut w, 120);
    let y = w.pose(ball).unwrap().unwrap().position[1];
    assert!(y > 2.0 && y < 2.6, "the ball rests on the box top: y {y}");
}

/// 1 m cubes, created asleep, 6 wide and `rows` high.
fn sleeping_wall(w: &mut World, rows: u32) -> Vec<BodyId> {
    (0..rows)
        .flat_map(|row| (0..6).map(move |col| (row, col)))
        .map(|(row, col)| {
            w.create_body(
                &BodyDef {
                    position: [0.0, 0.5 + row as f32, col as f32 - 2.5],
                    mass: MassSpec::Mass(100.0),
                    asleep: true,
                    ..BodyDef::default()
                },
                &[ShapeDef::new(Shape::Box { half: [0.5; 3] })],
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn a_sleeping_wall_stays_put_and_is_never_reported_again() {
    let mut w = world(-G);
    ground(&mut w);
    let wall = sleeping_wall(&mut w, 4);
    let frames = steps(&mut w, 120);
    let first: Vec<BodyId> = frames[0].moved.iter().map(|m| m.body).collect();
    assert!(
        wall.iter().all(|b| first.contains(b)),
        "first frame reports every body"
    );
    for f in &frames[1..] {
        assert!(
            f.moved.iter().all(|m| !wall.contains(&m.body)),
            "step {}: a sleeping box moved",
            f.step
        );
    }
}

#[test]
fn a_ball_knocks_a_sleeping_wall_loose() {
    let mut w = world(-G);
    ground(&mut w);
    let rows = 4;
    let wall = sleeping_wall(&mut w, rows);
    let speed = 30.0;
    let ball = sphere(&mut w, [6.0, 1.5, 0.0], 0.3, 50.0);
    w.set_velocity(ball, [-speed, 0.0, 0.0], [0.0; 3]).unwrap();
    let bound = speed + (2.0 * G * rows as f32).sqrt();
    let mut fastest = 0.0f32;
    let mut woke = 0;
    for _ in 0..180 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            woke += f
                .events
                .iter()
                .filter(|e| matches!(e, Event::BodyWoke { body, .. } if wall.contains(body)))
                .count();
        }
        let bodies = w.sim().download_bodies();
        for b in &wall {
            fastest = fastest.max(Vec3::from(bodies[b.index() as usize].linear_vel).length());
        }
    }
    assert!(woke > 0, "the ball woke nothing");
    assert!(
        fastest <= bound,
        "a box reached {fastest} m/s, over the energy bound {bound}"
    );
}

#[test]
fn bodies_added_to_a_running_world_are_hit_by_rays() {
    rays_find_a_late_body(false);
}

/// Needs vibe-land's WgpuSim fix "bounds for colliders added asleep to a running
/// world" (af24d153 on feat/vibe-land-backend), not yet on main: a raw WgpuSim
/// misses the box the same way.
#[test]
#[ignore = "rubble gap: a collider added asleep to a running world gets no ray bounds (fix af24d153 unmerged)"]
fn bodies_added_asleep_to_a_running_world_are_hit_by_rays() {
    rays_find_a_late_body(true);
}

fn rays_find_a_late_body(asleep: bool) {
    {
        let mut w = world(-G);
        ground(&mut w);
        steps(&mut w, 1);
        let late = w
            .create_body(
                &BodyDef {
                    position: [0.0, 1.5, 0.0],
                    asleep,
                    mass: MassSpec::Mass(10.0),
                    ..BodyDef::default()
                },
                &[ShapeDef::new(Shape::Box { half: [0.5; 3] })],
            )
            .unwrap();
        steps(&mut w, 1);
        let ray = Ray {
            origin: [5.0, 1.5, 0.0],
            dir: [-1.0, 0.0, 0.0],
            max_dist: 20.0,
            exclude_body: None,
        };
        let hit = w.raycast(&[ray])[0];
        assert_eq!(hit.map(|h| h.body), Some(late), "asleep={asleep}");
    }
}

#[test]
fn a_ray_down_hits_the_ground_and_names_its_body() {
    let mut w = world(-G);
    let g = ground(&mut w);
    let ray = Ray {
        origin: [3.0, 4.0, 1.0],
        dir: [0.0, -2.0, 0.0],
        max_dist: 10.0,
        exclude_body: None,
    };
    let hit = w.raycast(&[ray])[0].expect("ground hit");
    assert_eq!(hit.body, g);
    assert!(
        (hit.distance - 4.0).abs() < 1e-3,
        "distance {}",
        hit.distance
    );
    let _ = IDENTITY;
}
