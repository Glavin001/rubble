//! The terrain look-ahead (`terrain_watch.slang`) against ballistics: a body
//! falling onto a heightfield is predicted to strike when, where and how fast
//! free fall says; strikes too soon to act on, too slow or too weak are not
//! reported; and of several, the most energetic wins (the lower collider on a
//! tie), so the pick is deterministic.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::{TerrainStrike, TerrainWatch, WgpuSim};
use glam::{Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;
const G: f32 = -9.81;

fn sim() -> WgpuSim {
    let mut s = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter");
    let mut p = AvbdParams {
        dt: FRAME,
        ..AvbdParams::default()
    };
    p.gravity_y = G;
    p.sleep_time = 1.0e9;
    s.set_params(p);
    s
}

fn ground(s: &mut WgpuSim) -> u32 {
    let field = HeightField::new(41, 41, 0.5, 0.5, vec![0.0; 41 * 41]).unwrap();
    let mut arena = HullArena::new();
    s.add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap()
        .1
}

fn ball(s: &mut WgpuSim, pos: Vec3, r: f32, mass: f32, v: Vec3) -> u32 {
    let i = 0.4 * mass * r * r;
    let mut state = BodyState::dynamic(pos, Quat::IDENTITY, mass, Vec3::splat(i));
    state.linear_vel = v.to_array();
    let body = s.add_body(state);
    s.add_collider(Collider {
        body,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [r, 0.0, 0.0, r],
    });
    body
}

fn watch(field: u32) -> TerrainWatch {
    TerrainWatch {
        fields: vec![field],
        horizon: 2.0,
        samples: 32,
        min_speed: 0.0,
        min_energy: 0.0,
        min_time: 0.0,
    }
}

/// One rigid step, then the look-ahead; its pick.
fn step_and_watch(s: &mut WgpuSim) -> Option<TerrainStrike> {
    s.step().unwrap();
    let mut enc = s
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    s.terrain_encode_watch(&mut enc);
    s.queue().submit([enc.finish()]);
    let words = s.read_words(engine_terrain::layout::MPM_WATCH_BINDING, 16, 14);
    WgpuSim::decode_terrain_strike(&words)
}

#[test]
fn a_falling_ball_is_predicted_to_strike_as_free_fall_says() {
    let mut s = sim();
    let field = ground(&mut s);
    let r = 0.2;
    let body = ball(
        &mut s,
        Vec3::new(1.0, 5.0, -2.0),
        r,
        10.0,
        Vec3::new(0.5, 0.0, 0.0),
    );
    s.terrain_set_watch(&watch(field)).unwrap();
    let hit = step_and_watch(&mut s).expect("a strike within 2 s");
    // From the step's end state: y + v t + g t²/2 = r.
    // BodyState rows (28 words): position at 0, linear velocity at 12.
    let w: Vec<f32> = s
        .read_words(0, u64::from(body) * 28, 28)
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let (x, y, vy) = (w[0], w[1], w[13]);
    let t = (-vy - (vy * vy - 2.0 * G * (y - r)).sqrt()) / G;
    assert_eq!(hit.body, body);
    assert_eq!(hit.field, field);
    assert!(
        (hit.time - t).abs() < 1e-3,
        "strike at {} s, free fall {t} s",
        hit.time
    );
    assert!((hit.point[1] - r).abs() < 2e-3, "centre at {:?}", hit.point);
    assert!((hit.point[0] - (x + 0.5 * t)).abs() < 2e-3);
    let speed = vy + G * t;
    assert!(
        (hit.velocity[1] - speed).abs() < 1e-2,
        "{:?} vs {speed}",
        hit.velocity
    );
    assert!((hit.energy - 0.5 * 10.0 * (speed * speed + 0.25)).abs() < 0.01 * hit.energy);
    assert_eq!(hit.mass, 10.0);
    assert_eq!(hit.radius, r);
}

#[test]
fn strikes_too_soon_too_slow_too_weak_or_away_are_not_reported() {
    let mut s = sim();
    let field = ground(&mut s);
    ball(&mut s, Vec3::new(0.0, 2.0, 0.0), 0.2, 10.0, Vec3::ZERO);
    // Free fall from 1.8 m takes 0.6 s; the step adds g·dt² shy of it.
    for (w, why) in [
        (
            TerrainWatch {
                min_time: 0.7,
                ..watch(field)
            },
            "too soon",
        ),
        (
            TerrainWatch {
                min_speed: 7.0,
                ..watch(field)
            },
            "too slow",
        ),
        (
            TerrainWatch {
                min_energy: 200.0,
                ..watch(field)
            },
            "too weak",
        ),
        (
            TerrainWatch {
                horizon: 0.3,
                ..watch(field)
            },
            "beyond the horizon",
        ),
    ] {
        s.terrain_set_watch(&w).unwrap();
        assert_eq!(step_and_watch(&mut s), None, "{why}");
    }
    s.terrain_set_watch(&watch(field)).unwrap();
    assert!(step_and_watch(&mut s).is_some());

    // Thrown upward faster than it can come back within the horizon.
    let mut s = sim();
    let field = ground(&mut s);
    ball(
        &mut s,
        Vec3::new(0.0, 2.0, 0.0),
        0.2,
        10.0,
        Vec3::new(0.0, 12.0, 0.0),
    );
    s.terrain_set_watch(&watch(field)).unwrap();
    assert_eq!(step_and_watch(&mut s), None);
}

#[test]
fn the_most_energetic_strike_wins_and_ties_go_to_the_lower_collider() {
    let mut s = sim();
    let field = ground(&mut s);
    let light = ball(&mut s, Vec3::new(-3.0, 4.0, 0.0), 0.2, 1.0, Vec3::ZERO);
    let heavy = ball(&mut s, Vec3::new(3.0, 3.0, 0.0), 0.2, 50.0, Vec3::ZERO);
    s.terrain_set_watch(&watch(field)).unwrap();
    assert_eq!(step_and_watch(&mut s).unwrap().body, heavy);
    let _ = light;

    let mut s = sim();
    let field = ground(&mut s);
    let a = ball(&mut s, Vec3::new(-3.0, 3.0, 0.0), 0.2, 5.0, Vec3::ZERO);
    let _b = ball(&mut s, Vec3::new(3.0, 3.0, 0.0), 0.2, 5.0, Vec3::ZERO);
    s.terrain_set_watch(&watch(field)).unwrap();
    for _ in 0..3 {
        assert_eq!(step_and_watch(&mut s).unwrap().body, a);
    }
}
