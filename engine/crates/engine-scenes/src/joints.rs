//! Joint demo and stress scenes (Y-up; `build_world` adds the ground box).
//! Every body owns exactly one collider,
//! added in body order, so the browser renderer can draw instance i as body i.

use engine_metal::SceneBuilder;
use engine_schema::*;
use glam::Vec3;

fn make_static(b: &mut SceneBuilder<'_>, id: u32) {
    let mut s = b.world_mut().body(id);
    s.inv_mass = 0.0;
    s.inv_inertia = [0.0; 3];
    s.flags = 0;
    b.world_mut().set_body(id, s);
}

fn post(b: &mut SceneBuilder<'_>, pos: Vec3, half: Vec3) -> u32 {
    let id = b.add_box(pos, half, 1.0);
    make_static(b, id);
    id
}

fn set_velocity(b: &mut SceneBuilder<'_>, id: u32, v: Vec3, w: Vec3) {
    let mut s = b.world_mut().body(id);
    s.linear_vel = v.to_array();
    s.angular_vel = w.to_array();
    b.world_mut().set_body(id, s);
}

fn hinge_at(b: &mut SceneBuilder<'_>, a: u32, c: u32, at: Vec3, axis: Vec3) -> JointDesc {
    let (fa, fb) = b.world_frames(a, c, at, axis);
    JointDesc::hinge(a, c, fa, fb)
}

fn cone_at(
    b: &mut SceneBuilder<'_>,
    a: u32,
    c: u32,
    at: Vec3,
    axis: Vec3,
    sy: f32,
    sz: f32,
) -> JointDesc {
    let (fa, fb) = b.world_frames(a, c, at, axis);
    JointDesc::cone(a, c, fa, fb, sy, sz)
}

/// Ten-body ragdoll standing at `origin` (feet on it), facing +Z. Cone joints
/// with twist limits at spine / neck / shoulders / hips, limited hinges at the
/// elbows and knees. Returns the pelvis id.
pub fn ragdoll(b: &mut SceneBuilder<'_>, origin: Vec3, scale: f32) -> u32 {
    let s = scale;
    let p = |x: f32, y: f32, z: f32| origin + Vec3::new(x, y, z) * s;
    let pelvis = b.add_box(
        p(0.0, 1.0, 0.0),
        Vec3::new(0.16, 0.1, 0.1) * s,
        8.0 * s * s * s,
    );
    let chest = b.add_box(
        p(0.0, 1.32, 0.0),
        Vec3::new(0.18, 0.18, 0.1) * s,
        14.0 * s * s * s,
    );
    let head = b.add_sphere(p(0.0, 1.66, 0.0), 0.11 * s, 4.0 * s * s * s);
    let ua_l = b.add_box(
        p(-0.27, 1.34, 0.0),
        Vec3::new(0.05, 0.13, 0.05) * s,
        2.0 * s * s * s,
    );
    let la_l = b.add_box(
        p(-0.27, 1.05, 0.0),
        Vec3::new(0.045, 0.13, 0.045) * s,
        1.5 * s * s * s,
    );
    let ua_r = b.add_box(
        p(0.27, 1.34, 0.0),
        Vec3::new(0.05, 0.13, 0.05) * s,
        2.0 * s * s * s,
    );
    let la_r = b.add_box(
        p(0.27, 1.05, 0.0),
        Vec3::new(0.045, 0.13, 0.045) * s,
        1.5 * s * s * s,
    );
    let th_l = b.add_box(
        p(-0.09, 0.7, 0.0),
        Vec3::new(0.065, 0.18, 0.065) * s,
        6.0 * s * s * s,
    );
    let sh_l = b.add_box(
        p(-0.09, 0.28, 0.0),
        Vec3::new(0.055, 0.19, 0.055) * s,
        4.0 * s * s * s,
    );
    let th_r = b.add_box(
        p(0.09, 0.7, 0.0),
        Vec3::new(0.065, 0.18, 0.065) * s,
        6.0 * s * s * s,
    );
    let sh_r = b.add_box(
        p(0.09, 0.28, 0.0),
        Vec3::new(0.055, 0.19, 0.055) * s,
        4.0 * s * s * s,
    );

    let spine = cone_at(b, pelvis, chest, p(0.0, 1.12, 0.0), Vec3::Y, 0.5, 0.35)
        .with_twist_limit(-0.4, 0.4);
    b.add_joint(spine);
    let neck =
        cone_at(b, chest, head, p(0.0, 1.52, 0.0), Vec3::Y, 0.6, 0.6).with_twist_limit(-0.8, 0.8);
    b.add_joint(neck);
    for (ua, la, x) in [(ua_l, la_l, -0.27f32), (ua_r, la_r, 0.27)] {
        let shoulder =
            cone_at(b, chest, ua, p(x, 1.48, 0.0), -Vec3::Y, 1.3, 1.3).with_twist_limit(-0.6, 0.6);
        b.add_joint(shoulder);
        let elbow = hinge_at(b, ua, la, p(x, 1.19, 0.0), Vec3::X).with_twist_limit(-2.4, 0.05);
        b.add_joint(elbow);
    }
    for (th, sh, x) in [(th_l, sh_l, -0.09f32), (th_r, sh_r, 0.09)] {
        let hip =
            cone_at(b, pelvis, th, p(x, 0.9, 0.0), -Vec3::Y, 0.9, 0.5).with_twist_limit(-0.3, 0.3);
        b.add_joint(hip);
        let knee = hinge_at(b, th, sh, p(x, 0.5, 0.0), Vec3::X).with_twist_limit(-0.05, 2.3);
        b.add_joint(knee);
    }
    pelvis
}

/// `n` ragdolls on a grid, stacked in layers, dropped onto a box ground.
/// ~10 bodies and 9 joints each: 1000 ragdolls = 10k bodies, 9k joints.
pub fn ragdoll_pile(b: &mut SceneBuilder<'_>, n: u32) {
    let side = ((n as f32 / 4.0).sqrt().ceil() as u32).max(1);
    let mut placed = 0;
    let mut layer = 0;
    while placed < n {
        for x in 0..side {
            for z in 0..side {
                if placed >= n {
                    return;
                }
                let origin = Vec3::new(
                    (x as f32 - side as f32 * 0.5) * 0.9 + (layer % 2) as f32 * 0.3,
                    1.0 + layer as f32 * 2.0,
                    (z as f32 - side as f32 * 0.5) * 0.7,
                );
                let pelvis = ragdoll(b, origin, 1.0);
                // A little spin so the pile tangles instead of stacking upright.
                let w = Vec3::new(
                    ((placed * 7) % 5) as f32 - 2.0,
                    0.0,
                    ((placed * 3) % 5) as f32 - 2.0,
                ) * 0.8;
                set_velocity(b, pelvis, Vec3::ZERO, w);
                placed += 1;
            }
        }
        layer += 1;
    }
}

/// `chains` hanging chains of `links` hinged planks, alternating hinge axes,
/// started horizontal so they swing down and wrap.
pub fn hinge_chains(b: &mut SceneBuilder<'_>, chains: u32, links: u32) {
    // Columns of chains along X (each chain spans links·0.5 m), rows along Z
    // close enough that neighbouring chains tangle as they swing down.
    let columns = 10u32;
    let rows = chains.div_ceil(columns);
    let span = links as f32 * 0.5 + 1.5;
    let top = 2.0 + links as f32 * 0.5;
    for c in 0..chains {
        let x0 = (c % columns) as f32 * span - columns as f32 * span * 0.5;
        let z0 = (c / columns) as f32 * 0.35 - rows as f32 * 0.35 * 0.5;
        let anchor = post(b, Vec3::new(x0, top, z0), Vec3::splat(0.1));
        let mut prev = anchor;
        for i in 0..links {
            let x = x0 + 0.25 + i as f32 * 0.5;
            let link = b.add_box(Vec3::new(x, top, z0), Vec3::new(0.22, 0.04, 0.08), 0.5);
            let axis = if i % 2 == 0 { Vec3::Z } else { Vec3::Y };
            let d = hinge_at(b, prev, link, Vec3::new(x - 0.25, top, z0), axis);
            b.add_joint(d);
            prev = link;
        }
    }
}

/// Grid of motorised windmills (hub + rotor + two weights on the rotor tips
/// hanging from ball joints). Each rotor runs a velocity motor.
pub fn motor_field(b: &mut SceneBuilder<'_>, n: u32) {
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let x = (i % side) as f32 * 2.2 - side as f32 * 1.1;
        let z = (i / side) as f32 * 2.2 - side as f32 * 1.1;
        let hub_pos = Vec3::new(x, 2.5, z);
        let hub = post(b, hub_pos, Vec3::splat(0.08));
        let rotor = b.add_box(
            hub_pos + Vec3::new(0.0, 0.0, 0.2),
            Vec3::new(0.8, 0.06, 0.04),
            1.0,
        );
        let speed = 1.0 + (i % 7) as f32 * 0.4;
        let d = hinge_at(b, hub, rotor, hub_pos + Vec3::new(0.0, 0.0, 0.2), Vec3::Z).with_drive(
            JointDrive::motor(Dof::Twist, if i % 2 == 0 { speed } else { -speed }, 60.0),
        );
        b.add_joint(d);
        for tip in [-0.75f32, 0.75] {
            let w = b.add_sphere(hub_pos + Vec3::new(tip, -0.3, 0.2), 0.08, 0.3);
            b.add_ball_joint(rotor, w, Vec3::new(tip, 0.0, 0.0), Vec3::new(0.0, 0.3, 0.0));
        }
    }
}

/// Station origins on a 4 x 3 grid (6 m cells, ground level), front row at
/// -Z so a camera at (0, 13, -24) sees every station.
fn station(i: u32) -> Vec3 {
    let col = (i % 4) as f32;
    let row = (i / 4) as f32;
    Vec3::new((col - 1.5) * 6.0, 0.5, row * 6.0 - 6.0)
}

/// One station per joint feature: rope, cone fan, door, windmill motor,
/// sliders + servo, weld tower, breakable wall, spring chain, distance cradle
/// + tether, plastic beam, hinge bridge, ragdolls.
pub fn joints_showcase(b: &mut SceneBuilder<'_>) {
    // 0. Ball-joint rope released horizontally.
    let o = station(0);
    let top = post(b, o + Vec3::new(-2.5, 6.0, 0.0), Vec3::splat(0.12));
    let mut prev = top;
    for i in 0..12 {
        let s = b.add_sphere(
            o + Vec3::new(-2.5 + 0.35 * (i + 1) as f32, 6.0, 0.0),
            0.13,
            0.5,
        );
        let anchor_a = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.175, 0.0, 0.0)
        };
        b.add_ball_joint(
            prev,
            s,
            anchor_a,
            Vec3::new(if i == 0 { -0.35 } else { -0.175 }, 0.0, 0.0),
        );
        prev = s;
    }

    // 1. Cone-limited pendulum fan, limits 0.2 .. 1.0 rad.
    let o = station(1);
    for k in 0..5 {
        let p = o + Vec3::new(-2.0 + k as f32 * 1.0, 4.5, 0.0);
        let pivot = post(b, p, Vec3::splat(0.08));
        let rod = b.add_box(
            p - Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.06, 0.9, 0.06),
            1.0,
        );
        let lim = 0.2 + 0.2 * k as f32;
        let d = cone_at(b, pivot, rod, p, -Vec3::Y, lim, lim);
        b.add_joint(d);
        set_velocity(b, rod, Vec3::new(3.0, 0.0, 2.5), Vec3::ZERO);
    }

    // 2. Hinged door with a twist limit, kicked open.
    let o = station(2);
    let frame = post(b, o + Vec3::new(-0.8, 1.8, 0.0), Vec3::new(0.08, 1.8, 0.08));
    let door = b.add_box(o + Vec3::new(0.0, 1.8, 0.0), Vec3::new(0.7, 1.5, 0.05), 6.0);
    let d = hinge_at(b, frame, door, o + Vec3::new(-0.72, 1.8, 0.0), Vec3::Y)
        .with_twist_limit(-1.4, 1.4);
    b.add_joint(d);
    set_velocity(b, door, Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0));

    // 3. Motorised windmill (velocity motor about Z).
    let o = station(3);
    let mast = post(b, o + Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.12, 2.0, 0.12));
    let rotor = b.add_box(
        o + Vec3::new(0.0, 3.8, -0.3),
        Vec3::new(1.8, 0.12, 0.06),
        3.0,
    );
    let d = hinge_at(b, mast, rotor, o + Vec3::new(0.0, 3.8, -0.3), Vec3::Z)
        .with_drive(JointDrive::motor(Dof::Twist, 1.5, 400.0));
    b.add_joint(d);

    // 4. Slider pistons: one limited on an incline, one sweeping position servo.
    let o = station(4);
    let rail = post(b, o + Vec3::new(-1.0, 1.5, 0.0), Vec3::splat(0.1));
    let block = b.add_box(o + Vec3::new(-1.0, 1.5, 0.0), Vec3::splat(0.25), 1.0);
    let axis = Vec3::new(1.0, 1.0, 0.0).normalize();
    let (fa, fb) = b.world_frames(rail, block, o + Vec3::new(-1.0, 1.5, 0.0), axis);
    b.add_joint(JointDesc::slider(rail, block, fa, fb).with_linear_limit(Dof::X, -0.9, 1.2));
    let rail2 = post(b, o + Vec3::new(0.0, 4.0, 0.0), Vec3::splat(0.1));
    let piston = b.add_box(o + Vec3::new(0.0, 4.0, 0.0), Vec3::new(0.3, 0.2, 0.2), 1.0);
    let (fa, fb) = b.world_frames(rail2, piston, o + Vec3::new(0.0, 4.0, 0.0), Vec3::X);
    b.add_joint(
        JointDesc::slider(rail2, piston, fa, fb)
            .with_linear_limit(Dof::X, -1.8, 1.8)
            .with_drive(JointDrive::servo(Dof::X, 1.5, 400.0, 40.0, 2000.0)),
    );

    // 5. Welded tower (fixed joints).
    let o = station(5);
    let mut below = None;
    for i in 0..8 {
        let p = o + Vec3::new(0.0, 0.25 + 0.5 * i as f32, 0.0);
        let blk = b.add_box(p, Vec3::new(0.25, 0.25, 0.25), 1.0);
        if let Some(lo) = below {
            let (fa, fb) = b.world_frames(lo, blk, p - Vec3::new(0.0, 0.25, 0.0), Vec3::X);
            b.add_joint(JointDesc::fixed(lo, blk, fa, fb));
        }
        below = Some(blk);
    }

    // 6. Breakable welded wall (break force / torque).
    let o = station(6) + Vec3::new(-1.25, 0.0, 0.0);
    let mut grid = vec![vec![0u32; 6]; 5];
    for (row, line) in grid.iter_mut().enumerate() {
        for (col, cell) in line.iter_mut().enumerate() {
            *cell = b.add_box(
                o + Vec3::new(col as f32 * 0.5, 0.25 + row as f32 * 0.5, 0.0),
                Vec3::splat(0.24),
                0.6,
            );
        }
    }
    for row in 0..5 {
        for col in 0..6 {
            let me = grid[row][col];
            let here = o + Vec3::new(col as f32 * 0.5, 0.25 + row as f32 * 0.5, 0.0);
            if col + 1 < 6 {
                let (fa, fb) = b.world_frames(
                    me,
                    grid[row][col + 1],
                    here + Vec3::new(0.25, 0.0, 0.0),
                    Vec3::X,
                );
                b.add_joint(
                    JointDesc::fixed(me, grid[row][col + 1], fa, fb).with_break(40.0, 25.0),
                );
            }
            if row + 1 < 5 {
                let (fa, fb) = b.world_frames(
                    me,
                    grid[row + 1][col],
                    here + Vec3::new(0.0, 0.25, 0.0),
                    Vec3::X,
                );
                b.add_joint(
                    JointDesc::fixed(me, grid[row + 1][col], fa, fb).with_break(40.0, 25.0),
                );
            }
        }
    }

    // 7. Spring chain hanging from a beam.
    let o = station(7);
    let beam = post(b, o + Vec3::new(0.0, 5.5, 0.0), Vec3::new(0.3, 0.1, 0.1));
    let mut prev = beam;
    for i in 0..5 {
        let s = b.add_sphere(o + Vec3::new(0.0, 4.8 - 0.7 * i as f32, 0.0), 0.15, 0.6);
        b.add_spring(prev, s, Vec3::ZERO, Vec3::ZERO, 0.7, 120.0);
        prev = s;
    }

    // 8. Distance-rod pendulums (Newton's-cradle row) and a max-distance tether.
    let o = station(8);
    for k in 0..5 {
        let p = o + Vec3::new(-1.5 + k as f32 * 0.42, 4.5, 0.0);
        let pivot = post(b, p, Vec3::splat(0.05));
        let bob = b.add_sphere(p - Vec3::new(0.0, 2.0, 0.0), 0.2, 1.0);
        b.add_distance(pivot, bob, Vec3::ZERO, Vec3::ZERO, 2.0);
        if k == 0 {
            set_velocity(b, bob, Vec3::new(-5.0, 0.0, 0.0), Vec3::ZERO);
        }
    }
    let anchor = post(b, o + Vec3::new(1.8, 0.3, 0.0), Vec3::splat(0.12));
    let tethered = b.add_sphere(o + Vec3::new(1.8, 3.0, 0.0), 0.3, 1.0);
    b.add_joint(JointDesc::max_distance(
        anchor,
        tethered,
        Vec3::ZERO,
        Vec3::ZERO,
        2.8,
    ));
    set_velocity(b, tethered, Vec3::new(3.0, 5.0, 2.0), Vec3::ZERO);

    // 9. Plastic beam: yields under a dropped weight and keeps the bend.
    let o = station(9);
    let wall = post(b, o + Vec3::new(-1.2, 2.5, 0.0), Vec3::new(0.15, 1.0, 0.4));
    let beam = b.add_box(
        o + Vec3::new(0.0, 2.5, 0.0),
        Vec3::new(1.05, 0.08, 0.25),
        2.0,
    );
    let (fa, fb) = b.world_frames(wall, beam, o + Vec3::new(-1.05, 2.5, 0.0), Vec3::X);
    b.add_joint(JointDesc::fixed(wall, beam, fa, fb).with_yield(30.0, 1.2));
    b.add_box(o + Vec3::new(0.7, 5.0, 0.0), Vec3::splat(0.2), 6.0);

    // 10. Hinged plank bridge between two posts, crates dropped on it.
    let o = station(10) + Vec3::new(-2.6, 0.0, 0.0);
    let l = post(b, o + Vec3::new(0.0, 2.5, 0.0), Vec3::new(0.15, 0.15, 0.8));
    let mut prev = l;
    let planks = 10;
    for i in 0..planks {
        let px = 0.3 + i as f32 * 0.5;
        let plank = b.add_box(o + Vec3::new(px, 2.5, 0.0), Vec3::new(0.24, 0.05, 0.7), 0.6);
        let d = hinge_at(b, prev, plank, o + Vec3::new(px - 0.25, 2.5, 0.0), Vec3::Z);
        b.add_joint(d);
        prev = plank;
    }
    let end_x = 0.3 + planks as f32 * 0.5;
    let r = post(
        b,
        o + Vec3::new(end_x, 2.5, 0.0),
        Vec3::new(0.15, 0.15, 0.8),
    );
    let d = hinge_at(b, prev, r, o + Vec3::new(end_x - 0.25, 2.5, 0.0), Vec3::Z);
    b.add_joint(d);
    for k in 0..3 {
        b.add_box(
            o + Vec3::new(1.5 + k as f32 * 1.2, 4.5 + k as f32, 0.0),
            Vec3::splat(0.25),
            1.5,
        );
    }

    // 11. Ragdolls dropped from a height.
    let o = station(11);
    for k in 0..3 {
        let pelvis = ragdoll(
            b,
            o + Vec3::new(-1.5 + k as f32 * 1.5, 3.0 + k as f32, 0.0),
            1.0,
        );
        set_velocity(
            b,
            pelvis,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.5 * (k as f32 - 1.0)),
        );
    }
}
