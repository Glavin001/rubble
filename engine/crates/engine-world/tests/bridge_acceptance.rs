//! What vibe-land's rubble bridge does through `WgpuSim` (blocking read-modify-
//! writes, parking bodies for removal), done through the world API: its
//! acceptance cases, with commands instead of readbacks and destruction
//! instead of parking.

use glam::Vec3;
use rubble::*;

const G: f32 = 9.81;

fn world_with(gravity: f32, features: EngineFeatures) -> World {
    World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: WorldCapacity {
                max_candidates: 131_072,
                ..WorldCapacity::small()
            },
            gravity: [0.0, gravity, 0.0],
            features,
            ..WorldDef::default()
        },
    )
    .expect("wgpu device")
}

fn world(gravity: f32) -> World {
    world_with(gravity, EngineFeatures::default())
}

fn rough() -> ShapeDef {
    ShapeDef {
        material: SurfaceMaterial {
            friction: 0.6,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    }
}

fn ground(w: &mut World) -> BodyId {
    // Top face at y = 0.
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -5.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &rough(), [200.0, 5.0, 200.0])
        .unwrap();
    g
}

fn sphere(w: &mut World, position: [f32; 3], radius: f32, mass: f32) -> BodyId {
    let b = w.create_body(&BodyDef::dynamic(position)).unwrap();
    let volume = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let s = ShapeDef {
        density: mass / volume,
        ..rough()
    };
    w.create_sphere_shape(b, &s, [0.0; 3], radius).unwrap();
    b
}

fn cube(w: &mut World, position: [f32; 3], half: f32, mass: f32, awake: bool) -> BodyId {
    let b = w
        .create_body(&BodyDef {
            is_awake: awake,
            ..BodyDef::dynamic(position)
        })
        .unwrap();
    let s = ShapeDef {
        density: mass / (8.0 * half.powi(3)),
        ..rough()
    };
    w.create_box_shape(b, &s, [half; 3]).unwrap();
    b
}

fn steps(w: &mut World, n: u32) -> Vec<Frame> {
    for _ in 0..n {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap()
}

fn pos(w: &World, id: BodyId) -> [f32; 3] {
    w.body_transform(id).unwrap().unwrap().position
}

#[test]
fn a_central_impulse_changes_momentum_by_j() {
    let mut w = world(0.0);
    let m = 4.0;
    let ball = sphere(&mut w, [0.0, 10.0, 0.0], 0.5, m);
    w.body_apply_linear_impulse(ball, [8.0, 0.0, 0.0], [0.0, 10.0, 0.0], true)
        .unwrap();
    steps(&mut w, 1);
    let b = w.read_body_state(ball).unwrap();
    assert!(
        (b.linear_velocity[0] - 8.0 / m).abs() < 1e-4,
        "{:?}",
        b.linear_velocity
    );
    assert!(Vec3::from(b.angular_velocity).length() < 1e-6);
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
    let y = pos(&w, ball)[1];
    assert!((y - expected).abs() < 1e-3, "y {y} vs {expected}");
}

#[test]
fn a_destroyed_sphere_is_gone_and_its_slot_comes_back_as_a_new_body() {
    let mut w = world(-G);
    ground(&mut w);
    let ball = sphere(&mut w, [0.0, 3.0, 0.0], 0.5, 2.0);
    steps(&mut w, 1);
    w.destroy_body(ball).unwrap();
    let frames = steps(&mut w, 10);
    assert!(
        frames
            .iter()
            .all(|f| f.body_events.moves.iter().all(|m| m.body != ball)),
        "a destroyed body is never reported"
    );
    let parked = w.sim().read_body(ball.index());
    assert!(parked.inv_mass == 0.0 && parked.flags == 0, "{parked:?}");
    let again = sphere(&mut w, [4.0, 5.0, 0.0], 0.5, 2.0);
    assert_eq!(
        again.index(),
        ball.index(),
        "the quarantined slot is reused"
    );
    assert_ne!(again, ball);
    let frames = steps(&mut w, 10);
    let first = frames[0]
        .body_events
        .moves
        .iter()
        .find(|m| m.body == again)
        .unwrap();
    assert!(
        first.previous.is_none(),
        "a reused slot has no pose to interpolate from"
    );
    let p = pos(&w, again);
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
    let defs: Vec<BodyDef> = (0..1100)
        .map(|i| BodyDef {
            position: [
                (i % 40) as f32 * 4.0 - 80.0,
                1.0,
                (i / 40) as f32 * 4.0 - 60.0,
            ],
            ..BodyDef::default()
        })
        .collect();
    for b in w.create_bodies(&defs).unwrap() {
        w.create_box_shape(b, &rough(), [1.0; 3]).unwrap();
    }
    // Above the box at (0, 1, 0), whose top is y = 2.
    let ball = sphere(&mut w, [0.0, 4.0, 0.0], 0.3, 50.0);
    steps(&mut w, 120);
    let y = pos(&w, ball)[1];
    assert!(y > 2.0 && y < 2.6, "the ball rests on the box top: y {y}");
}

/// 1 m cubes, created asleep, 6 wide and `rows` high.
fn sleeping_wall(w: &mut World, rows: u32) -> Vec<BodyId> {
    (0..rows)
        .flat_map(|row| (0..6).map(move |col| (row, col)))
        .map(|(row, col)| {
            cube(
                w,
                [0.0, 0.5 + row as f32, col as f32 - 2.5],
                0.5,
                100.0,
                false,
            )
        })
        .collect()
}

#[test]
fn a_sleeping_wall_stays_put_and_is_never_reported_again() {
    let mut w = world(-G);
    ground(&mut w);
    let wall = sleeping_wall(&mut w, 4);
    let frames = steps(&mut w, 120);
    let first: Vec<BodyId> = frames[0].body_events.moves.iter().map(|m| m.body).collect();
    assert!(
        wall.iter().all(|b| first.contains(b)),
        "the first frame reports every body"
    );
    for f in &frames[1..] {
        assert!(
            f.body_events.moves.iter().all(|m| !wall.contains(&m.body)),
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
    w.body_set_linear_velocity(ball, [-speed, 0.0, 0.0])
        .unwrap();
    let bound = speed + (2.0 * G * rows as f32).sqrt();
    let mut fastest = 0.0f32;
    let mut woke = 0;
    for _ in 0..180 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            woke += f
                .body_events
                .moves
                .iter()
                .filter(|m| m.woke && wall.contains(&m.body))
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

fn rays_find_a_late_body(awake: bool) {
    let mut w = world(-G);
    ground(&mut w);
    steps(&mut w, 1);
    let late = cube(&mut w, [0.0, 1.5, 0.0], 0.5, 10.0, awake);
    steps(&mut w, 1);
    let ray = RayInput {
        origin: [5.0, 1.5, 0.0],
        translation: [-20.0, 0.0, 0.0],
        ignore_body: None,
    };
    let hit = w.cast_ray_blocking(&[ray])[0];
    assert_eq!(hit.map(|h| h.body), Some(late), "awake={awake}");
}

#[test]
fn bodies_added_to_a_running_world_are_hit_by_rays() {
    rays_find_a_late_body(true);
}

/// Needs vibe-land's WgpuSim fix "bounds for colliders added asleep to a
/// running world" (af24d153 on feat/vibe-land-backend), not yet on main: a raw
/// WgpuSim misses the box the same way.
#[test]
fn bodies_added_asleep_to_a_running_world_are_hit_by_rays() {
    rays_find_a_late_body(false);
}

#[test]
fn a_ray_down_hits_the_ground_and_names_its_body() {
    let mut w = world(-G);
    let g = ground(&mut w);
    let ray = RayInput {
        origin: [3.0, 4.0, 1.0],
        translation: [0.0, -10.0, 0.0],
        ignore_body: None,
    };
    let hit = w.cast_ray_blocking(&[ray])[0].expect("ground hit");
    assert_eq!(hit.body, g);
    assert!(
        (hit.distance - 4.0).abs() < 1e-3,
        "distance {}",
        hit.distance
    );
}
