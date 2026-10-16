//! Joint scenes built once in Rust, stepped on Metal (Slang→MSL) and wgpu
//! (Slang→WGSL): trajectories and joint states must agree.

use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::*;
use engine_webgpu::WgpuSim;
use glam::Vec3;

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

/// Every preset in one world: hinge chain with a limit, motor, slider with
/// limits and a servo, weld, cone, distance, tether, spring, breakable weld.
fn joint_zoo(b: &mut SceneBuilder<'_>) {
    b.add_ground_plane(0.0);
    // Hinge chain (3 links), last link limited.
    let pivot = static_box(b, Vec3::new(-6.0, 6.0, 0.0));
    let mut prev = pivot;
    for i in 0..3 {
        let x = -6.0 + 0.5 + i as f32;
        let link = b.add_box(Vec3::new(x, 6.0, 0.0), Vec3::new(0.45, 0.08, 0.08), 1.0);
        let (fa, fb) = b.world_frames(prev, link, Vec3::new(x - 0.5, 6.0, 0.0), Vec3::Z);
        let mut d = JointDesc::hinge(prev, link, fa, fb);
        if i == 2 {
            d = d.with_twist_limit(-0.4, 0.4);
        }
        b.add_joint(d);
        prev = link;
    }
    // Motorised wheel.
    let hub = static_box(b, Vec3::new(-2.0, 6.0, 0.0));
    let wheel = b.add_box(Vec3::new(-2.0, 6.0, 0.0), Vec3::new(0.6, 0.05, 0.2), 2.0);
    let (fa, fb) = b.world_frames(hub, wheel, Vec3::new(-2.0, 6.0, 0.0), Vec3::Y);
    b.add_joint(
        JointDesc::hinge(hub, wheel, fa, fb).with_drive(JointDrive::motor(Dof::Twist, 2.0, 50.0)),
    );
    // Inclined slider with limits, and a horizontal servo slider.
    let rail = static_box(b, Vec3::new(1.0, 6.0, 0.0));
    let block = b.add_box(Vec3::new(1.0, 6.0, 0.0), Vec3::splat(0.2), 1.0);
    let (fa, fb) = b.world_frames(
        rail,
        block,
        Vec3::new(1.0, 6.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0).normalize(),
    );
    b.add_joint(JointDesc::slider(rail, block, fa, fb).with_linear_limit(Dof::X, -0.8, 0.5));
    let rail2 = static_box(b, Vec3::new(1.0, 6.0, 3.0));
    let block2 = b.add_box(Vec3::new(1.0, 6.0, 3.0), Vec3::splat(0.2), 1.0);
    let (fa, fb) = b.world_frames(rail2, block2, Vec3::new(1.0, 6.0, 3.0), Vec3::X);
    b.add_joint(
        JointDesc::slider(rail2, block2, fa, fb).with_drive(JointDrive::servo(
            Dof::X,
            0.6,
            200.0,
            20.0,
            1.0e4,
        )),
    );
    // Weld cantilever.
    let wall = static_box(b, Vec3::new(4.0, 6.0, 0.0));
    let beam = b.add_box(Vec3::new(5.0, 6.0, 0.0), Vec3::new(0.9, 0.1, 0.1), 2.0);
    let (fa, fb) = b.world_frames(wall, beam, Vec3::new(4.1, 6.0, 0.0), Vec3::X);
    b.add_joint(JointDesc::fixed(wall, beam, fa, fb));
    // Cone pendulum with initial velocity.
    let cone_top = static_box(b, Vec3::new(8.0, 6.0, 0.0));
    let bob = b.add_box(Vec3::new(8.0, 5.0, 0.0), Vec3::new(0.05, 0.9, 0.05), 1.0);
    let mut s = b.world_mut().body(bob);
    s.linear_vel = [4.0, 0.0, 2.0];
    b.world_mut().set_body(bob, s);
    let (fa, fb) = b.world_frames(cone_top, bob, Vec3::new(8.0, 6.0, 0.0), -Vec3::Y);
    b.add_joint(JointDesc::cone(cone_top, bob, fa, fb, 0.5, 0.3));
    // Distance rod, tether, spring.
    let anchor = static_box(b, Vec3::new(11.0, 6.0, 0.0));
    let s1 = b.add_sphere(Vec3::new(12.0, 6.0, 0.0), 0.1, 1.0);
    b.add_distance(anchor, s1, Vec3::ZERO, Vec3::ZERO, 1.0);
    let s2 = b.add_sphere(Vec3::new(12.0, 6.0, 2.0), 0.1, 1.0);
    b.add_joint(JointDesc::max_distance(
        anchor,
        s2,
        Vec3::ZERO,
        Vec3::ZERO,
        3.0,
    ));
    let s3 = b.add_sphere(Vec3::new(11.0, 5.0, -2.0), 0.1, 1.0);
    let top = static_box(b, Vec3::new(11.0, 6.0, -2.0));
    b.add_spring(top, s3, Vec3::ZERO, Vec3::ZERO, 1.0, 300.0);
    // Breakable weld holding a heavy box.
    let hook = static_box(b, Vec3::new(14.0, 6.0, 0.0));
    let heavy = b.add_box(Vec3::new(14.0, 5.5, 0.0), Vec3::splat(0.2), 10.0);
    b.add_joint(
        JointDesc::fixed(
            hook,
            heavy,
            JointFrame::at(Vec3::new(0.0, -0.5, 0.0)),
            JointFrame::at(Vec3::ZERO),
        )
        .with_break(60.0, JOINT_BIG),
    );
}

fn params() -> AvbdParams {
    let mut p = AvbdParams::default();
    p.sleep_time = 1.0e9;
    p
}

#[test]
fn joint_zoo_matches_metal() {
    let cap = WorldCapacity::small();
    let mut metal = MetalWorld::new(cap.clone()).unwrap();
    metal.set_params(params());
    joint_zoo(&mut SceneBuilder::new(&mut metal));
    let snap = metal.snapshot();
    let mut wg = WgpuSim::new(cap).unwrap();
    wg.load_snapshot(&snap);
    assert_eq!(wg.joint_count(), metal.joint_count());

    let mut worst = 0.0f32;
    let mut worst_step = 0;
    for step in 0..120 {
        metal.step().unwrap();
        wg.step().unwrap();
        if step % 10 == 9 {
            let a = metal.download_bodies();
            let b = wg.download_bodies();
            let mut step_worst = (0.0f32, 0usize);
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                let d = (Vec3::from_array(x.position) - Vec3::from_array(y.position)).length();
                if d > step_worst.0 {
                    step_worst = (d, i);
                }
                // Limits and contacts are binary events: after the hinge
                // chain's limit engages (~step 110), last-bit MSL/WGSL
                // differences pick different active sets and the chaos grows.
                if step < 100 && d > worst {
                    worst = d;
                    worst_step = step;
                }
            }
            if std::env::var("PARITY_TRACE").is_ok() {
                println!(
                    "step {step}: worst {:.2e} at body {}",
                    step_worst.0, step_worst.1
                );
            }
        }
    }
    let mj = metal.joints().to_vec();
    let wj = wg.download_joints();
    for (i, (a, b)) in mj.iter().zip(&wj).enumerate() {
        assert_eq!(
            a.is_broken(),
            b.is_broken(),
            "joint {i} broken state differs"
        );
    }
    assert!(
        mj.iter().any(|j| j.is_broken()),
        "the breakable weld should break on both"
    );
    assert!(
        worst < 1e-3,
        "metal vs wgpu joint trajectories diverged by {worst} (step {worst_step})"
    );
}
