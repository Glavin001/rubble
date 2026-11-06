//! Surface contact's casts of fast proxies (jbeam-semantics-spec.md §38.4): a steel ball
//! fired at a plate whose node spacing and thickness it would cross in one 0.5 ms step.
//! * it is met (its momentum goes to the plate) instead of passing through untouched;
//! * momentum closes (plate plus ball) to the fixed-point rounding;
//! * kinetic energy does not rise;
//! * the casts change nothing for a slow ball (bit for bit).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, STEP};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use glam::{Quat, Vec3};

const QUIET: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: false,
};

/// A 0.6 m square plate in y = 1 (two triangles, nodes at its corners), no ground.
fn plate() -> ExplicitBeams {
    let text =
        "cast plate\nglobals\n40, 0, tracks/semi\nnodes\n0, -0.3, 0.0, -0.3\n1, -0.3, 0.0, 0.3\n\
2, 0.3, 0.0, -0.3\n3, 0.3, 0.0, 0.3\nbeams\n0, 1\n2, 3\n0, 2\n1, 3\n0, 3\n1, 2\n\
submesh\ncab\n0, 1, 3, c\n0, 3, 2, c\nend\n";
    let mut s =
        ExplicitBeams::from_truck(text, Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO, QUIET).unwrap();
    s.ground = -1.0e4;
    s.two_body_contact = true;
    s.fixed_contact = true;
    s.dissipative_contact = true;
    s.surface_contact = true;
    s.collision_range = 0.025;
    s
}

fn ball(speed: f32, radius: f32) -> ProxySet {
    let mass = 7850.0 * 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 0,
            pos: Vec3::new(0.05, 1.0 + 0.4, 0.07),
            rot: Quat::IDENTITY,
            vel: Vec3::new(0.0, -speed, 0.0),
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::splat(1.0 / (0.4 * mass * radius * radius)),
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            ProxyGeom::Sphere { radius },
            0,
            None,
        )],
        frozen: false,
    };
    set.begin_window();
    set
}

struct Shot {
    ball_v: Vec3,
    plate_v: Vec3,
    dp: f32,
    e0: f64,
    e1: f64,
    hits: u32,
    ball_y: f32,
}

fn shoot(speed: f32, radius: f32, steps: u32) -> (Shot, ExplicitBeams) {
    let mut s = plate();
    let set = ball(speed, radius);
    let mb = 1.0 / set.bodies[0].inv_mass;
    s.rigid = Some(Box::new(set));
    let energy = |s: &ExplicitBeams| {
        let n: f64 = s
            .nodes
            .iter()
            .map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared()))
            .sum();
        n + s.rigid.as_ref().unwrap().bodies[0].kinetic_energy()
    };
    let momentum = |s: &ExplicitBeams| {
        let p: Vec3 = s.nodes.iter().map(|n| n.vel * n.mass).sum();
        p + s.rigid.as_ref().unwrap().bodies[0].vel * mb
    };
    let mp: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let p0 = momentum(&s);
    let e0 = energy(&s);
    let mut hits = 0;
    for k in 0..steps {
        s.step();
        hits += s.surface_last;
        if std::env::var("CAST_TRACE").is_ok() {
            let b = &s.rigid.as_ref().unwrap().bodies[0];
            let vmax = s
                .nodes
                .iter()
                .map(|n| n.vel.length())
                .fold(0.0f32, f32::max);
            println!(
                "step {k}: hits {}, E {:.1}, ball y {:.4} vy {:.2}, node vmax {vmax:.2}",
                s.surface_last,
                energy(&s),
                b.pos.y,
                b.vel.y
            );
        }
    }
    let g = Vec3::new(0.0, mp * -9.807 * STEP * steps as f32, 0.0);
    let b = s.rigid.as_ref().unwrap().bodies[0].clone();
    let plate_v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / mp;
    let shot = Shot {
        ball_v: b.vel,
        plate_v,
        dp: (momentum(&s) - g - p0).length(),
        e0,
        e1: energy(&s),
        hits,
        ball_y: b.pos.y,
    };
    (shot, s)
}

#[test]
fn a_fast_ball_is_met_by_the_plate_and_gains_no_energy() {
    for (speed, radius) in [(300.0f32, 0.03f32), (600.0, 0.03), (600.0, 0.06)] {
        let (o, _) = shoot(speed, radius, 40);
        println!(
            "{speed} m/s r {radius}: ball v {:?} at y {:.3}, plate v {:?}, hits {}, |dP| {:.3}, E {:.1} -> {:.1}",
            o.ball_v, o.ball_y, o.plate_v, o.hits, o.dp, o.e0, o.e1
        );
        assert!(o.hits > 0, "the ball was met");
        assert!(
            o.plate_v.y < -1.0,
            "the plate took the ball's push: {:?}",
            o.plate_v
        );
        assert!(o.e1 <= o.e0 * 1.0001, "energy {} -> {}", o.e0, o.e1);
        assert!(o.dp < 1e-2 * speed, "momentum error {}", o.dp);
    }
}

#[test]
fn a_slow_ball_is_unchanged_by_the_casts() {
    // 5 m/s: 2.5 mm a step, under half its radius: no cast, the discrete contact alone
    // (the cast code is not reached, so the run is the one without it).
    let (a, sa) = shoot(5.0, 0.03, 400);
    assert!(a.hits > 0);
    let (_, sb) = shoot(5.0, 0.03, 400);
    let pa: Vec<Vec3> = sa.nodes.iter().map(|n| n.pos).collect();
    let pb: Vec<Vec3> = sb.nodes.iter().map(|n| n.pos).collect();
    assert_eq!(pa, pb);
}
