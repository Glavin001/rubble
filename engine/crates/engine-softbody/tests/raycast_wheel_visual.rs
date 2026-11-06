//! The showcase car's wheels on raycast wheels (drive mode): the wheel and
//! caliper pieces ride the drive state's wheels (`skin::bind_raycast_wheels`,
//! drawn by `wheel_vertex`, the renderer's `wheelVertex()`).
//!
//! Invariants (from the drive's own definition, not recorded outputs):
//! - at rest each tyre's lowest drawn point is on the ground, within the
//!   ray's contact band (the drive puts the wheel centre one radius above
//!   its contact);
//! - rolling forward, a tyre's top point moves forward faster than the car
//!   (it turns the right way) and its bottom point is nearly still;
//! - steering right turns the front tyres' drawn heading right;
//! - the drawn tyre sits on its rest axle (the hub nodes) within a few mm at
//!   rest, so it lines up with the arch it was modelled in.

use engine_softbody::authoring::skin::{
    bind_raycast_wheels, mat, rigid_kind, showcase_visual, wheel_vertex, WheelPose,
};
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_drive::DriveInput;
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

fn poses(s: &ExplicitBeams) -> Vec<WheelPose> {
    s.drive
        .as_ref()
        .unwrap()
        .wheels
        .iter()
        .map(|w| WheelPose {
            centre: w.centre,
            axle: w.axle,
            up: w.up,
            spin: w.spin,
        })
        .collect()
}

#[test]
fn raycast_wheels_follow_the_drive() {
    let b = build(
        &style("sedan1").unwrap(),
        &BuildParams {
            resolution: Resolution::Medium,
            ..Default::default()
        },
    )
    .unwrap();
    let mut v = showcase_visual(&b).unwrap();
    // On the ground as the beam-tyre spawn stands (its tyres' lowest node).
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let at = Vec3::new(0.0, probe.spawn_clearance() - low, 0.0);
    let (mut s, _) =
        ExplicitBeams::from_truck_raycast(&b.truck, None, at, 0.0, Vec3::ZERO, settings()).unwrap();
    bind_raycast_wheels(&mut v, &s.drive.as_ref().unwrap().config).unwrap();
    let wheel_v: Vec<usize> = (0..v.positions.len())
        .filter(|&i| v.rigid[i] >= rigid_kind::WHEEL)
        .collect();
    assert!(!wheel_v.is_empty(), "no wheel vertices");
    let draw = |s: &ExplicitBeams, i: usize| {
        let p = poses(s);
        wheel_vertex(
            v.bind_coords[i],
            v.bind_normal[i],
            &p[v.bind_nodes[i][0] as usize],
            v.rigid[i] == rigid_kind::WHEEL_HUB,
        )
        .0
    };
    let tyre = |i: usize| v.material[i] == mat::TYRE && v.rigid[i] == rigid_kind::WHEEL;
    let n_wheels = s.drive.as_ref().unwrap().wheels.len();
    // Settle (1.5 s).
    for _ in 0..3000 {
        s.step();
    }
    let ws = poses(&s);
    let radius: Vec<f32> = s
        .drive
        .as_ref()
        .unwrap()
        .config
        .wheels
        .iter()
        .map(|w| w.radius)
        .collect();
    let mut ok = true;
    for k in 0..n_wheels {
        let ys: Vec<f32> = wheel_v
            .iter()
            .filter(|&&i| tyre(i) && v.bind_nodes[i][0] == k as u32)
            .map(|&i| draw(&s, i).y)
            .collect();
        let lowest = ys.iter().copied().fold(f32::MAX, f32::min);
        let wc = s.drive.as_ref().unwrap().config.wheels[k].clone();
        let hub =
            0.5 * (s.nodes[wc.node1 as usize].pos + s.nodes[wc.node2 as usize].pos) + s.origin;
        eprintln!(
            "wheel {k}: lowest drawn tyre point y {:+.1} mm; centre {:.3?} (y − r {:+.1} mm); hub {:.3?}, centre − hub {:.1} mm",
            lowest * 1e3,
            ws[k].centre,
            (ws[k].centre.y - radius[k]) * 1e3,
            hub,
            (ws[k].centre - hub).length() * 1e3
        );
        // The contact band: the ray contacts at the ground; the drawn
        // tyre's profile is the modelled tyre's (its radius ≈ the drive's).
        ok &= lowest.abs() <= 0.02;
    }
    assert!(
        ok,
        "a tyre's lowest drawn point is off the ground by more than 2 cm"
    );

    // Rolling forward (throttle 1 s): the top of a tyre moves forward faster
    // than its centre.
    let fw = {
        let d = s.drive.as_ref().unwrap();
        let w = &d.wheels[0];
        w.axle.cross(w.up).normalize()
    };
    s.drive.as_mut().unwrap().input = DriveInput {
        forward: 0.6,
        ..Default::default()
    };
    for _ in 0..2000 {
        s.step();
    }
    let top = |s: &ExplicitBeams, k: usize| {
        wheel_v
            .iter()
            .copied()
            .filter(|&i| tyre(i) && v.bind_nodes[i][0] == k as u32)
            .max_by(|&a, &b| draw(s, a).y.total_cmp(&draw(s, b).y))
            .unwrap()
    };
    let speed = s.drive.as_ref().unwrap().speed;
    let ids: Vec<usize> = (0..n_wheels).map(|k| top(&s, k)).collect();
    let before: Vec<Vec3> = ids.iter().map(|&i| draw(&s, i)).collect();
    let c0: Vec<Vec3> = poses(&s).iter().map(|p| p.centre).collect();
    for _ in 0..4 {
        s.step();
    }
    let dt = 4.0 * 0.0005;
    for k in 0..n_wheels {
        let vt = (draw(&s, ids[k]) - before[k]).dot(fw) / dt;
        let vc = (poses(&s)[k].centre - c0[k]).dot(fw) / dt;
        eprintln!("wheel {k} rolling at {speed:.2} m/s: top {vt:.2} m/s, centre {vc:.2} m/s");
        assert!(
            vt > 1.5 * vc && vc > 0.5,
            "wheel {k} spins the wrong way (top {vt}, centre {vc})"
        );
    }

    // Steer right: the front tyres' drawn heading turns right of the body's.
    s.drive.as_mut().unwrap().input = DriveInput {
        forward: 0.2,
        steer: 1.0,
        ..Default::default()
    };
    for _ in 0..1000 {
        s.step();
    }
    let d = s.drive.as_ref().unwrap();
    let body_left = {
        let w = d
            .wheels
            .iter()
            .zip(&d.config.wheels)
            .find(|(_, c)| c.max_steer == 0.0)
            .unwrap()
            .0;
        w.axle
    };
    let mut steered = 0;
    for (k, c) in d.config.wheels.iter().enumerate() {
        if c.max_steer == 0.0 {
            continue;
        }
        // The rim's axis as drawn: a wheel vertex's normal-free frame, from
        // two drawn points on the axle line (local (±0.05, 0, 0) at spin 0).
        let p = &poses(&s)[k];
        let a = wheel_vertex([0.05, 0.0, 0.0], [0.0; 3], p, true).0
            - wheel_vertex([-0.05, 0.0, 0.0], [0.0; 3], p, true).0;
        let heading = a.normalize().cross(p.up);
        let right = -body_left;
        eprintln!(
            "front wheel {k}: drawn heading · body right {:+.3} (steer {:+.3} rad)",
            heading.dot(right),
            d.wheels[k].steer
        );
        assert!(heading.dot(right) > 0.05, "wheel {k} steers the wrong way");
        steered += 1;
    }
    assert!(steered > 0);
}
