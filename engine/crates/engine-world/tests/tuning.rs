//! Per-body damping, gravity scale and motion locks (P3).

use rubble::*;

fn world(gravity: f32) -> World {
    let def = WorldDef {
        gravity: [0.0, gravity, 0.0],
        ..WorldDef::default()
    };
    World::new(GpuContext::Default, &def).unwrap()
}

fn cube(w: &mut World, def: BodyDef) -> BodyId {
    let b = w.create_body(&def).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])
        .unwrap();
    b
}

fn run(w: &mut World, steps: u32) {
    for _ in 0..steps {
        w.step().unwrap();
        w.poll().unwrap();
    }
    w.poll_blocking().unwrap();
}

#[test]
fn damping_follows_box3d() {
    let mut w = world(0.0);
    // A slow spin: the solver's first-order rotation integration loses a
    // little angular speed at high rates, damped or not.
    let b = cube(
        &mut w,
        BodyDef {
            linear_velocity: [10.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.5, 0.0],
            linear_damping: 1.0,
            angular_damping: 0.5,
            ..BodyDef::dynamic([0.0, 5.0, 0.0])
        },
    );
    run(&mut w, 60);
    let s = w.read_body_state(b).unwrap();
    let h = w.solver().dt;
    let v = 10.0 / (1.0 + h).powi(60);
    let wv = 0.5 / (1.0 + 0.5 * h).powi(60);
    assert!((s.linear_velocity[0] - v).abs() < 0.005 * v, "{s:?} vs {v}");
    assert!(
        (s.angular_velocity[1] - wv).abs() < 0.005 * wv,
        "{s:?} vs {wv}"
    );
}

fn y(w: &World, b: BodyId) -> f32 {
    w.read_body_state(b).unwrap().transform.position[1]
}

#[test]
fn gravity_scale_scales_the_fall() {
    let mut w = world(-10.0);
    let normal = cube(&mut w, BodyDef::dynamic([0.0, 50.0, 0.0]));
    let double = cube(
        &mut w,
        BodyDef {
            gravity_scale: 2.0,
            ..BodyDef::dynamic([5.0, 50.0, 0.0])
        },
    );
    let floating = cube(
        &mut w,
        BodyDef {
            gravity_scale: 0.0,
            ..BodyDef::dynamic([10.0, 50.0, 0.0])
        },
    );
    run(&mut w, 60);
    let (dn, dd, df) = (
        50.0 - y(&w, normal),
        50.0 - y(&w, double),
        50.0 - y(&w, floating),
    );
    assert!(dn > 4.0, "fell: {dn}");
    assert!((dd / dn - 2.0).abs() < 0.02, "twice the drop: {dd} vs {dn}");
    assert!(df.abs() < 1e-4, "floats: {df}");
    w.body_set_gravity_scale(floating, 1.0).unwrap();
    run(&mut w, 30);
    assert!(50.0 - y(&w, floating) > 1.0, "falls once scaled back");
}

#[test]
fn motion_locks_hold_their_axes() {
    let mut w = world(0.0);
    let locks = MotionLocks {
        linear_x: true,
        angular_x: true,
        angular_y: true,
        angular_z: true,
        ..MotionLocks::default()
    };
    let b = cube(
        &mut w,
        BodyDef {
            motion_locks: locks,
            ..BodyDef::dynamic([0.0, 5.0, 0.0])
        },
    );
    // Off-centre impulse: would move it along x and z and spin it.
    w.body_apply_linear_impulse(b, [3.0, 0.0, 3.0], [0.0, 5.4, 0.3], true)
        .unwrap();
    run(&mut w, 30);
    let s = w.read_body_state(b).unwrap();
    assert!(s.transform.position[0].abs() < 1e-5, "x locked: {s:?}");
    assert!(s.transform.position[2] > 0.5, "z free: {s:?}");
    let q = s.transform.rotation;
    assert!(q[3].abs() > 0.99999, "no rotation: {q:?}");
    assert_eq!(w.body_motion_locks(b).unwrap(), locks);
    // Unlock: the next impulse moves it along x.
    w.body_set_motion_locks(b, MotionLocks::default()).unwrap();
    w.body_apply_linear_impulse_to_center(b, [3.0, 0.0, 0.0], true)
        .unwrap();
    run(&mut w, 30);
    assert!(w.read_body_state(b).unwrap().transform.position[0] > 0.5);
}

#[test]
fn damped_worlds_still_fall_asleep() {
    let mut w = world(-10.0);
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [10.0, 0.5, 10.0])
        .unwrap();
    cube(
        &mut w,
        BodyDef {
            linear_damping: 0.5,
            ..BodyDef::dynamic([0.0, 1.0, 0.0])
        },
    );
    run(&mut w, 600);
    let f = w.step_sync().unwrap();
    assert!(
        !f.ran,
        "the damped body slept and the world skips its steps"
    );
}
