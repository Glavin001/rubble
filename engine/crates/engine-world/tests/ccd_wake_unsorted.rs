//! `ccd_wake` with colliders out of body order. A sleeper in a fast CCD body's path
//! wakes before the narrowphase when it lies within 1.25 of the CCD collider's radii,
//! plus its own extent, of the swept segment. A body that got its shape after a later
//! body got one (pooled body slots: the city's houses) has its collider out of body
//! order. Main's per-body pass found a body's colliders by binary search and a
//! contiguous run, so for such a body it found none, took its extent as zero and
//! left it asleep (city war, first shell step: 6 of 22 sleepers). The per-collider
//! pass (`EngineFeatures::ccd_by_collider`) tests each collider.

use rubble::*;

#[test]
fn ccd_wake_finds_a_sleeper_whose_collider_is_out_of_body_order() {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 64,
            max_colliders: 64,
            max_contacts: 4096,
            max_candidates: 8192,
            max_colors: 32,
            ..WorldCapacity::small()
        },
        enable_sleep: true,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    let ground = w
        .create_body(&BodyDef {
            position: [0.0, -1.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(ground, &ShapeDef::default(), [100.0, 1.0, 100.0])
        .unwrap();
    let asleep = |p| BodyDef {
        is_awake: false,
        ..BodyDef::dynamic(p)
    };
    // The sleeper's body first, its shape last: colliders [ground, other, sleeper].
    let sleeper = w.create_body(&asleep([0.0, 1.0, 0.0])).unwrap();
    let other = w.create_body(&asleep([30.0, 1.0, 0.0])).unwrap();
    w.create_box_shape(other, &ShapeDef::default(), [1.0; 3])
        .unwrap();
    w.create_box_shape(sleeper, &ShapeDef::default(), [1.0; 3])
        .unwrap();
    // A 0.5 m bullet passing the sleeper's centre 2 m off in one step: within
    // 1.25 x 0.5 + sqrt(3) = 2.36 m of the sweep, 0.5 m clear of the box.
    let ball = w
        .create_body(&BodyDef {
            linear_velocity: [600.0, 0.0, 0.0],
            is_bullet: true,
            gravity_scale: 0.0,
            ..BodyDef::dynamic([-6.0, 1.0, 2.0])
        })
        .unwrap();
    w.create_sphere_shape(ball, &ShapeDef::default(), [0.0; 3], 0.5)
        .unwrap();
    // The GPU's sleep bit (World::body_is_awake follows the frames' sleep events).
    let asleep_now = |w: &World, id| {
        let slot = w.body_pose_index(id).unwrap() as usize;
        w.sim().download_bodies()[slot].flags & engine_schema::BodyFlags::Sleeping as u32 != 0
    };
    assert!(asleep_now(&w, sleeper) && asleep_now(&w, other));
    w.step().unwrap();
    w.poll_blocking().unwrap();
    assert!(
        !asleep_now(&w, sleeper),
        "the sleeper in the bullet's path wakes (ccd_wake)"
    );
    assert!(asleep_now(&w, other), "a far sleeper stays asleep");
}
