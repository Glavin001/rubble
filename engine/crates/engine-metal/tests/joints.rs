//! Joint behaviour on Metal: each D6 preset holds what it locks, frees what it
//! frees, and limits / drives / breaking / yield act at their thresholds.

use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::*;
use glam::{Quat, Vec3};

const STEPS: usize = 180;

fn world() -> MetalWorld {
    let mut w = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    // Joint tests watch motion; keep everything awake.
    p.sleep_time = 1.0e9;
    w.set_params(p);
    w
}

fn make_static(b: &mut SceneBuilder<'_>, id: u32) {
    let mut s = b.world_mut().body(id);
    s.inv_mass = 0.0;
    s.inv_inertia = [0.0; 3];
    s.flags = 0;
    b.world_mut().set_body(id, s);
}

fn static_box(b: &mut SceneBuilder<'_>, pos: Vec3) -> u32 {
    let id = b.add_box(pos, Vec3::splat(0.1), 1.0);
    make_static(b, id);
    id
}

fn pose(w: &MetalWorld, id: u32) -> (Vec3, Quat) {
    let s = w.body(id);
    (
        Vec3::from_array(s.position),
        Quat::from_array(s.orientation),
    )
}

fn anchor_gap(w: &MetalWorld, j: &Joint) -> f32 {
    let (pa, qa) = pose(w, j.body_a);
    let (pb, qb) = pose(w, j.body_b);
    let wa = pa + qa * Vec3::from_array(j.anchor_a);
    let wb = pb + qb * Vec3::from_array(j.anchor_b);
    (wa - wb).length()
}

fn run(w: &mut MetalWorld, steps: usize, mut each: impl FnMut(&MetalWorld)) {
    for _ in 0..steps {
        w.step().unwrap();
        each(w);
    }
}

#[test]
fn hinge_swings_only_about_its_axis() {
    let mut w = world();
    let (j, bar) = {
        let mut b = SceneBuilder::new(&mut w);
        let pivot = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        // Bar starts horizontal along +X so gravity swings it about Z.
        let bar = b.add_box(Vec3::new(1.0, 5.0, 0.0), Vec3::new(0.9, 0.1, 0.1), 1.0);
        let (fa, fb) = b.world_frames(pivot, bar, Vec3::new(0.0, 5.0, 0.0), Vec3::Z);
        (b.add_joint(JointDesc::hinge(pivot, bar, fa, fb)), bar)
    };
    let mut max_gap = 0.0f32;
    let mut min_axis = 1.0f32;
    let mut min_y = f32::MAX;
    run(&mut w, STEPS, |w| {
        max_gap = max_gap.max(anchor_gap(w, &w.joint(j)));
        let (p, q) = pose(w, bar);
        min_axis = min_axis.min((q * Vec3::Z).dot(Vec3::Z));
        min_y = min_y.min(p.y);
    });
    assert!(max_gap < 2.0e-3, "hinge anchor drifted {max_gap}");
    assert!(min_axis > 0.9995, "hinge axis tilted, dot={min_axis}");
    assert!(min_y < 4.2, "bar never swung down (min y {min_y})");
}

#[test]
fn hinge_twist_limit_stops_the_swing() {
    let mut w = world();
    let bar = {
        let mut b = SceneBuilder::new(&mut w);
        let pivot = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let bar = b.add_box(Vec3::new(1.0, 5.0, 0.0), Vec3::new(0.9, 0.1, 0.1), 1.0);
        let (fa, fb) = b.world_frames(pivot, bar, Vec3::new(0.0, 5.0, 0.0), Vec3::Z);
        b.add_joint(JointDesc::hinge(pivot, bar, fa, fb).with_twist_limit(-0.5, 0.5));
        bar
    };
    let mut max_angle = 0.0f32;
    run(&mut w, STEPS, |w| {
        let (_, q) = pose(w, bar);
        max_angle = max_angle.max((2.0 * q.z.atan2(q.w)).abs());
    });
    assert!(max_angle < 0.55, "twist limit exceeded: {max_angle} rad");
    assert!(
        max_angle > 0.4,
        "bar never reached the limit: {max_angle} rad"
    );
}

#[test]
fn hinge_motor_reaches_target_speed() {
    let mut w = world();
    let wheel = {
        let mut b = SceneBuilder::new(&mut w);
        let hub = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        // Vertical axis: gravity exerts no torque about it.
        let wheel = b.add_box(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.8, 0.05, 0.2), 2.0);
        let (fa, fb) = b.world_frames(hub, wheel, Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
        b.add_joint(
            JointDesc::hinge(hub, wheel, fa, fb).with_drive(JointDrive::motor(
                Dof::Twist,
                3.0,
                1000.0,
            )),
        );
        wheel
    };
    run(&mut w, 120, |_| {});
    let s = w.body(wheel);
    let wy = s.angular_vel[1];
    // B twists positively relative to A about +Y.
    assert!((wy - 3.0).abs() < 0.15, "motor speed {wy} rad/s, want 3");
    assert!(s.angular_vel[0].abs() < 0.05 && s.angular_vel[2].abs() < 0.05);
}

#[test]
fn hinge_motor_respects_force_limit() {
    let mut w = world();
    let wheel = {
        let mut b = SceneBuilder::new(&mut w);
        let hub = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let wheel = b.add_box(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.8, 0.05, 0.2), 2.0);
        let (fa, fb) = b.world_frames(hub, wheel, Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
        b.add_joint(
            JointDesc::hinge(hub, wheel, fa, fb).with_drive(JointDrive::motor(
                Dof::Twist,
                50.0,
                0.5,
            )),
        );
        wheel
    };
    // Iyy of the 1.6 x 0.1 x 0.4 m, 2 kg box; a 0.5 N·m cap accelerates at τ/I.
    let iyy = 2.0 / 12.0 * (1.6f32 * 1.6 + 0.4 * 0.4);
    let t = 30.0 / 60.0;
    run(&mut w, 30, |_| {});
    let wy = w.body(wheel).angular_vel[1];
    let cap = 0.5 / iyy * t;
    assert!(
        wy > 0.5 * cap && wy < 1.3 * cap,
        "limited motor spun to {wy}, cap ~{cap}"
    );
}

#[test]
fn slider_moves_only_along_its_axis_and_stops_at_limits() {
    let mut w = world();
    let (j, block, start) = {
        let mut b = SceneBuilder::new(&mut w);
        let rail = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let start = Vec3::new(0.0, 5.0, 0.0);
        let block = b.add_box(start, Vec3::splat(0.2), 1.0);
        // Inclined rail: gravity pulls the block down-slope along -axis.
        let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
        let (fa, fb) = b.world_frames(rail, block, start, axis);
        let j = b.add_joint(JointDesc::slider(rail, block, fa, fb).with_linear_limit(
            Dof::X,
            -1.0,
            0.5,
        ));
        (j, block, start)
    };
    let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
    let mut max_perp = 0.0f32;
    let mut min_s = 0.0f32;
    let mut max_tilt = 0.0f32;
    run(&mut w, STEPS, |w| {
        let (p, q) = pose(w, block);
        let d = p - start;
        let s = d.dot(axis);
        max_perp = max_perp.max((d - axis * s).length());
        min_s = min_s.min(s);
        max_tilt = max_tilt.max(q.angle_between(Quat::IDENTITY));
    });
    let _ = j;
    assert!(max_perp < 2.0e-3, "slider left its axis by {max_perp}");
    assert!(max_tilt < 2.0e-3, "slider rotated {max_tilt} rad");
    assert!(
        min_s > -1.03 && min_s < -0.95,
        "slider travel {min_s}, limit -1"
    );
}

#[test]
fn fixed_joint_holds_a_cantilever() {
    let mut w = world();
    let (j, beam) = {
        let mut b = SceneBuilder::new(&mut w);
        let wall = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let beam = b.add_box(Vec3::new(1.0, 5.0, 0.0), Vec3::new(0.9, 0.1, 0.1), 3.0);
        let (fa, fb) = b.world_frames(wall, beam, Vec3::new(0.1, 5.0, 0.0), Vec3::X);
        (b.add_joint(JointDesc::fixed(wall, beam, fa, fb)), beam)
    };
    let mut max_tilt = 0.0f32;
    let mut max_gap = 0.0f32;
    run(&mut w, STEPS, |w| {
        let (_, q) = pose(w, beam);
        max_tilt = max_tilt.max(q.angle_between(Quat::IDENTITY));
        max_gap = max_gap.max(anchor_gap(w, &w.joint(j)));
    });
    assert!(max_tilt < 0.02, "weld sagged {max_tilt} rad");
    assert!(max_gap < 2.0e-3, "weld anchor gap {max_gap}");
}

#[test]
fn fixed_joint_resists_twist() {
    // A weld must lock rotation about every axis; the old cross-product rows
    // left rotation about the joint axis free.
    let mut w = world();
    let beam = {
        let mut b = SceneBuilder::new(&mut w);
        let wall = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        // Mass offset along Z gives a torque about the beam's own (X) axis.
        let beam = b.add_compound_box_sphere(
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(0.9, 0.1, 0.1),
            Vec3::new(0.0, 0.0, 0.6),
            0.2,
            3.0,
        );
        let (fa, fb) = b.world_frames(wall, beam, Vec3::new(0.1, 5.0, 0.0), Vec3::X);
        b.add_joint(JointDesc::fixed(wall, beam, fa, fb));
        beam
    };
    let mut max_tilt = 0.0f32;
    run(&mut w, STEPS, |w| {
        let (_, q) = pose(w, beam);
        max_tilt = max_tilt.max(q.angle_between(Quat::IDENTITY));
    });
    assert!(max_tilt < 0.03, "weld twisted {max_tilt} rad");
}

#[test]
fn cone_limit_holds_a_pendulum() {
    let mut w = world();
    let bob = {
        let mut b = SceneBuilder::new(&mut w);
        let pivot = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        // Rod hangs along -Y; the cone axis (frame X) is -Y.
        let bob = b.add_box(Vec3::new(0.0, 4.0, 0.0), Vec3::new(0.05, 0.9, 0.05), 1.0);
        let mut s = b.world_mut().body(bob);
        s.linear_vel = [6.0, 0.0, 3.0];
        b.world_mut().set_body(bob, s);
        let (fa, fb) = b.world_frames(pivot, bob, Vec3::new(0.0, 5.0, 0.0), -Vec3::Y);
        b.add_joint(JointDesc::cone(pivot, bob, fa, fb, 0.4, 0.4));
        bob
    };
    let mut max_angle = 0.0f32;
    run(&mut w, STEPS, |w| {
        let (p, _) = pose(w, bob);
        let d = (p - Vec3::new(0.0, 5.0, 0.0)).normalize();
        max_angle = max_angle.max(d.angle_between(-Vec3::Y));
    });
    assert!(max_angle < 0.45, "cone exceeded: {max_angle} rad");
    assert!(
        max_angle > 0.3,
        "pendulum never reached the cone: {max_angle} rad"
    );
}

#[test]
fn distance_and_tether_lengths_hold() {
    let mut w = world();
    let (rod, tether, a0, b1, b2) = {
        let mut b = SceneBuilder::new(&mut w);
        let a0 = static_box(&mut b, Vec3::new(0.0, 6.0, 0.0));
        let b1 = b.add_sphere(Vec3::new(1.0, 6.0, 0.0), 0.1, 1.0);
        let rod = b.add_distance(a0, b1, Vec3::ZERO, Vec3::ZERO, 1.0);
        let b2 = b.add_sphere(Vec3::new(3.0, 6.0, 2.0), 0.1, 1.0);
        let tether = b.add_joint(JointDesc::max_distance(a0, b2, Vec3::ZERO, Vec3::ZERO, 4.0));
        (rod, tether, a0, b1, b2)
    };
    let _ = (rod, tether);
    let mut rod_err = 0.0f32;
    let mut tether_max = 0.0f32;
    run(&mut w, STEPS, |w| {
        let p0 = pose(w, a0).0;
        rod_err = rod_err.max(((pose(w, b1).0 - p0).length() - 1.0).abs());
        tether_max = tether_max.max((pose(w, b2).0 - p0).length());
    });
    assert!(rod_err < 3.0e-3, "distance error {rod_err}");
    assert!(
        tether_max < 4.01 && tether_max > 3.9,
        "tether length {tether_max}, max 4"
    );
}

#[test]
fn spring_settles_at_hookes_extension() {
    let mut w = world();
    let mass = 2.0f32;
    let k = 400.0f32;
    let bob = {
        let mut b = SceneBuilder::new(&mut w);
        let top = static_box(&mut b, Vec3::new(0.0, 6.0, 0.0));
        let bob = b.add_sphere(Vec3::new(0.0, 5.0, 0.0), 0.1, mass);
        b.add_joint(JointDesc::spring(top, bob, Vec3::ZERO, Vec3::ZERO, 1.0, k));
        // Damp the oscillation with a soft servo-free approach: let it ring
        // down through AVBD's implicit integration.
        bob
    };
    run(&mut w, 600, |_| {});
    let y = pose(&w, bob).0.y;
    let want = 6.0 - 1.0 - mass * 10.0 / k;
    // Implicit Euler damps the ringing; the mean sits at the static stretch.
    assert!((y - want).abs() < 0.03, "spring rest y={y}, want {want}");
}

#[test]
fn breakable_joint_breaks_and_slot_is_reused() {
    let mut w = world();
    let (j, heavy) = {
        let mut b = SceneBuilder::new(&mut w);
        let top = static_box(&mut b, Vec3::new(0.0, 6.0, 0.0));
        let heavy = b.add_box(Vec3::new(0.0, 5.5, 0.0), Vec3::splat(0.2), 10.0);
        // Holding 10 kg takes ~100 N; break at 50 N.
        let j = b.add_joint(
            JointDesc::fixed(
                top,
                heavy,
                JointFrame::at(Vec3::new(0.0, -0.5, 0.0)),
                JointFrame::at(Vec3::ZERO),
            )
            .with_break(50.0, JOINT_BIG),
        );
        (j, heavy)
    };
    run(&mut w, 30, |_| {});
    assert!(
        w.joint(j).is_broken(),
        "joint should break under 2x its load"
    );
    assert_eq!(w.drain_broken_joints(), vec![j]);
    run(&mut w, 30, |_| {});
    assert!(
        pose(&w, heavy).0.y < 5.0,
        "broken joint still holds the box"
    );
    // Released slot is reused.
    let reused = {
        let mut b = SceneBuilder::new(&mut w);
        let a = static_box(&mut b, Vec3::new(5.0, 6.0, 0.0));
        let c = b.add_sphere(Vec3::new(5.0, 5.0, 0.0), 0.1, 1.0);
        b.add_ball_joint(a, c, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))
    };
    assert_eq!(reused, j);
}

#[test]
fn strong_joint_does_not_break() {
    let mut w = world();
    let j = {
        let mut b = SceneBuilder::new(&mut w);
        let top = static_box(&mut b, Vec3::new(0.0, 6.0, 0.0));
        let heavy = b.add_box(Vec3::new(0.0, 5.5, 0.0), Vec3::splat(0.2), 10.0);
        b.add_joint(
            JointDesc::fixed(
                top,
                heavy,
                JointFrame::at(Vec3::new(0.0, -0.5, 0.0)),
                JointFrame::at(Vec3::ZERO),
            )
            .with_break(300.0, JOINT_BIG),
        )
    };
    run(&mut w, 60, |_| {});
    assert!(!w.joint(j).is_broken());
    let f = w.joint(j).linear_force();
    assert!((f - 100.0).abs() < 10.0, "joint force {f} N, want ~100");
}

#[test]
fn plastic_weld_yields_then_breaks_past_bend() {
    let mut w = world();
    let (soft, brittle) = {
        let mut b = SceneBuilder::new(&mut w);
        let wall = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let beam = b.add_box(Vec3::new(1.0, 5.0, 0.0), Vec3::new(0.9, 0.1, 0.1), 3.0);
        let (fa, fb) = b.world_frames(wall, beam, Vec3::new(0.1, 5.0, 0.0), Vec3::X);
        let soft = b.add_joint(JointDesc::fixed(wall, beam, fa, fb).with_yield(5.0, JOINT_BIG));
        let wall2 = static_box(&mut b, Vec3::new(0.0, 5.0, 4.0));
        let beam2 = b.add_box(Vec3::new(1.0, 5.0, 4.0), Vec3::new(0.9, 0.1, 0.1), 3.0);
        let (fa, fb) = b.world_frames(wall2, beam2, Vec3::new(0.1, 5.0, 4.0), Vec3::X);
        let brittle = b.add_joint(JointDesc::fixed(wall2, beam2, fa, fb).with_yield(5.0, 0.3));
        (soft, brittle)
    };
    run(&mut w, STEPS, |_| {});
    let js = w.joint(soft);
    assert!(!js.is_broken(), "plastic weld without bend limit broke");
    assert!(js.bend() > 0.3, "plastic weld bent only {} rad", js.bend());
    assert!(
        w.joint(brittle).is_broken(),
        "bend limit should break the weld"
    );
}

#[test]
fn soft_joint_stretches_by_stiffness() {
    let mut w = world();
    let (top, bob) = {
        let mut b = SceneBuilder::new(&mut w);
        let top = static_box(&mut b, Vec3::new(0.0, 6.0, 0.0));
        let bob = b.add_sphere(Vec3::new(0.0, 5.0, 0.0), 0.1, 1.0);
        b.add_joint(
            JointDesc::ball(top, bob, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))
                .with_stiffness(500.0, 0.0),
        );
        (top, bob)
    };
    let _ = top;
    run(&mut w, 600, |_| {});
    let sag = 5.0 - pose(&w, bob).0.y;
    let want = 10.0 / 500.0;
    assert!(
        (sag - want).abs() < 0.01,
        "soft joint sag {sag}, want {want}"
    );
}

#[test]
fn running_motor_never_sleeps_and_still_chain_sleeps_together() {
    let mut w = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    p.sleep_time = 0.3;
    w.set_params(p);
    let (wheel, chain) = {
        let mut b = SceneBuilder::new(&mut w);
        let hub = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let wheel = b.add_box(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.6, 0.05, 0.2), 2.0);
        let (fa, fb) = b.world_frames(hub, wheel, Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
        b.add_joint(
            JointDesc::hinge(hub, wheel, fa, fb).with_drive(JointDrive::motor(
                Dof::Twist,
                1.0,
                100.0,
            )),
        );
        // Hanging chain at rest.
        let top = static_box(&mut b, Vec3::new(5.0, 6.0, 0.0));
        let mut prev = top;
        let mut chain = Vec::new();
        for i in 0..4 {
            let s = b.add_sphere(Vec3::new(5.0, 5.5 - 0.5 * i as f32, 0.0), 0.1, 1.0);
            let anchor_a = if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.0, -0.25, 0.0)
            };
            b.add_ball_joint(
                prev,
                s,
                anchor_a,
                Vec3::new(0.0, if i == 0 { 0.5 } else { 0.25 }, 0.0),
            );
            chain.push(s);
            prev = s;
        }
        (wheel, chain)
    };
    run(&mut w, 300, |_| {});
    assert!(!w.body(wheel).is_sleeping(), "driven wheel fell asleep");
    let asleep: Vec<bool> = chain.iter().map(|&c| w.body(c).is_sleeping()).collect();
    assert!(
        asleep.iter().all(|&s| s),
        "still chain should sleep as one island: {asleep:?}"
    );
}

#[test]
fn slider_servo_tracks_its_target() {
    let mut w = world();
    let block = {
        let mut b = SceneBuilder::new(&mut w);
        let rail = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let block = b.add_box(Vec3::new(0.0, 5.0, 0.0), Vec3::splat(0.2), 1.0);
        let (fa, fb) = b.world_frames(rail, block, Vec3::new(0.0, 5.0, 0.0), Vec3::X);
        b.add_joint(
            JointDesc::slider(rail, block, fa, fb).with_drive(JointDrive::servo(
                Dof::X,
                0.7,
                300.0,
                30.0,
                1.0e4,
            )),
        );
        block
    };
    run(&mut w, 240, |_| {});
    let x = pose(&w, block).0.x;
    assert!((x - 0.7).abs() < 0.01, "servo settled at {x}, target 0.7");
}

#[test]
fn collide_flag_controls_contact_between_jointed_bodies() {
    for collide in [false, true] {
        let mut w = world();
        {
            let mut b = SceneBuilder::new(&mut w);
            let a = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
            let c = b.add_box(Vec3::new(0.15, 5.0, 0.0), Vec3::splat(0.1), 1.0);
            b.add_joint(
                JointDesc::ball(a, c, Vec3::ZERO, Vec3::new(-0.15, 0.0, 0.0))
                    .with_collision(collide),
            );
        }
        run(&mut w, 2, |_| {});
        let contacts = w.last_counters().manifold_count;
        assert_eq!(
            contacts > 0,
            collide,
            "collide={collide}: manifolds={contacts}"
        );
    }
}

#[test]
fn twist_limit_with_free_swing_is_stable() {
    // Spherical joint with only a twist limit (PhysX D6: swings free, twist
    // limited). Spin the rod about its own axis and swing it.
    let mut w = world();
    let rod = {
        let mut b = SceneBuilder::new(&mut w);
        let pivot = static_box(&mut b, Vec3::new(0.0, 5.0, 0.0));
        let rod = b.add_box(Vec3::new(0.0, 4.0, 0.0), Vec3::new(0.1, 0.9, 0.3), 1.0);
        let mut s = b.world_mut().body(rod);
        s.linear_vel = [3.0, 0.0, 1.0];
        s.angular_vel = [0.0, 6.0, 0.0];
        b.world_mut().set_body(rod, s);
        let (fa, fb) = b.world_frames(pivot, rod, Vec3::new(0.0, 5.0, 0.0), -Vec3::Y);
        let d = JointDesc::ball(pivot, rod, fa.anchor, fb.anchor);
        let mut d = JointDesc {
            frame_a: fa,
            frame_b: fb,
            ..d
        };
        d.motion[Dof::Twist as usize] = Motion::Limited;
        d.twist = (-0.4, 0.4);
        b.add_joint(d);
        rod
    };
    let mut max_v = 0.0f32;
    run(&mut w, 300, |w| {
        max_v = max_v.max(Vec3::from_array(w.body(rod).linear_vel).length());
    });
    assert!(
        max_v < 8.0,
        "twist-limited ball joint blew up: max |v| {max_v}"
    );
}
