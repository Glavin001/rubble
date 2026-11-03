//! Vehicle–vehicle contact on the GPU (`ExplicitBeamsGpu`) against the CPU
//! reference (`engine_softbody::explicit_contact::ExplicitBeamsWorld`): the contact
//! throttle and the fast-approach reset step for step on tiny structures; vehicles
//! that never meet run as without contact; two procedural sedans in a rear-end
//! crash agree before contact to millimetres and after it in broken and bent
//! counts; a node leaving its frame bounds is reported.

use engine_softbody::explicit_beams::*;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
};

fn gpu(fleet: &[ExplicitBeams]) -> Option<ExplicitBeamsGpu> {
    let (_, device, queue) = default_device().ok()?;
    Some(ExplicitBeamsGpu::new(&device, &queue, fleet))
}

fn vehicle(points: &[Vec3]) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    for &p in points {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: 0,
        });
    }
    s
}

/// An immovable triangle in y = 0 and an immovable node at `extra`.
fn floor(extra: Vec3) -> ExplicitBeams {
    let mut s = vehicle(&[
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        extra,
    ]);
    for n in &mut s.nodes {
        n.flags |= NODE_IMMOVABLE;
    }
    s.triangles.push(Triangle {
        nodes: [0, 1, 2],
        wait: 0,
        idle: 0,
    });
    s.finish();
    s
}

fn point(p: Vec3, v: Vec3, immovable: bool) -> ExplicitBeams {
    let mut s = vehicle(&[p]);
    s.nodes[0].vel = v;
    if immovable {
        s.nodes[0].flags |= NODE_IMMOVABLE;
    }
    s.finish();
    s
}

fn rms(a: &[Vec3], b: &[Vec3]) -> f32 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x.distance_squared(*y)))
        .sum();
    (s / a.len().max(1) as f64).sqrt() as f32
}

#[test]
fn contact_throttles_step_like_the_cpu() {
    // No candidates (backing off), candidates without a hit, and a hit, each a
    // one-step frame and a 17-step frame on the GPU.
    let cases = [
        (Vec3::new(2.5, 2.5, 2.5), Vec3::ZERO, true),
        (Vec3::new(0.9, 0.01, 1.5), Vec3::ZERO, true),
        (Vec3::new(0.0, 0.01, 0.0), Vec3::ZERO, false),
    ];
    for (p, v, still) in cases {
        let fleet = vec![floor(Vec3::new(3.0, 3.0, 3.0)), point(p, v, still)];
        let Some(mut g) = gpu(&fleet) else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut w = ExplicitBeamsWorld::new(fleet);
        for (frames, steps) in [(30, 1), (6, 17)] {
            for _ in 0..frames {
                w.frame(steps);
                g.frame(steps);
                g.wait();
                assert_eq!(
                    g.contact_throttles()[0],
                    w.vehicles[0].contact_throttle,
                    "point {p}, after {} steps",
                    w.steps
                );
            }
        }
        // The GPU accumulates contact forces in 1/32 N fixed point.
        let d = rms(&g.positions()[1], &w.vehicles[1].positions());
        assert!(d < 1e-4, "point {p}: GPU–CPU {d} m");
        g.contact_status().unwrap();
    }
}

#[test]
fn a_fast_approach_resets_both_vehicles_throttles_on_the_gpu() {
    let a = floor(Vec3::new(3.0, 3.0, 3.0));
    let mut b = floor(Vec3::new(3.0, 3.0, 3.0));
    for n in &mut b.nodes {
        n.pos += Vec3::new(0.0, 2.0, 0.0);
    }
    b.nodes[0].vel = Vec3::new(5.0, 0.0, 0.0);
    let mut fleet = vec![a, b];
    fleet[0].contact_throttle[0] = [9, 4];
    fleet[1].contact_throttle[0] = [6, 1];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet);
    for _ in 0..3 {
        w.frame(1);
        g.frame(1);
    }
    g.wait();
    let ct = g.contact_throttles();
    assert_eq!(ct[0], w.vehicles[0].contact_throttle);
    assert_eq!(ct[1], w.vehicles[1].contact_throttle);
}

#[test]
fn vehicles_that_never_meet_run_as_without_contact() {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let fleet: Vec<ExplicitBeams> = (0..3)
        .map(|k| {
            ExplicitBeams::from_truck(
                &text,
                Vec3::new(8.0 * k as f32, 0.6, 0.0),
                Vec3::new(0.0, 0.0, 5.0),
                settings,
            )
            .unwrap()
        })
        .collect();
    let Some(mut on) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut off = gpu(&fleet).unwrap();
    off.set_vehicle_contact(false);
    assert!(on.contact_active() && !off.contact_active());
    for _ in 0..40 {
        on.frame(17);
        off.frame(17);
    }
    on.wait();
    off.wait();
    on.contact_status().unwrap();
    let (a, b) = (on.positions(), off.positions());
    for k in 0..3 {
        let d = rms(&a[k], &b[k]);
        assert!(d < 1e-6, "vehicle {k}: contact on vs off {d} m");
    }
}

fn sedan_text() -> String {
    car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    })
}

/// Two sedans in one lane, the rear one at `speed` m/s, `gap` m apart.
fn rear_end(speed: f32, gap: f32) -> Vec<ExplicitBeams> {
    let text = sedan_text();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
        (a.min(n.pos.z), b.max(n.pos.z))
    });
    let length = zmax - zmin;
    let v = Vec3::new(0.0, 0.0, speed);
    let mut rear =
        ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings).unwrap();
    rear.roll_tyres(v);
    let front = ExplicitBeams::from_truck(
        &text,
        Vec3::new(0.0, 0.01 - low, length + gap),
        Vec3::ZERO,
        settings,
    )
    .unwrap();
    vec![front, rear]
}

#[test]
fn two_sedans_rear_end_match_cpu() {
    let fleet = rear_end(15.0, 0.5);
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet);
    // Before contact: the same to a fraction of a millimetre.
    let mut first = None;
    for f in 0..36 {
        w.frame(17);
        g.frame(17);
        if w.first_hit_step.is_some() && first.is_none() {
            first = Some(f);
            g.wait();
            let p = g.positions();
            for k in 0..2 {
                let d = rms(&p[k], &w.vehicles[k].positions());
                eprintln!(
                    "frame {f} (contact starts): vehicle {k} GPU–CPU {:.3} mm",
                    d * 1e3
                );
                assert!(d < 5e-4, "vehicle {k}: {d} m at contact");
            }
        }
    }
    g.wait();
    g.contact_status().unwrap();
    assert!(first.is_some(), "they met");
    assert!(w.total.hits > 100);
    let beams = g.beams();
    for k in 0..2 {
        let (c_broken, c_bent) = w.vehicles[k].damage();
        let g_broken = beams[k].iter().filter(|b| b.broken).count();
        let g_bent = beams[k].iter().filter(|b| b.bent()).count();
        let d = rms(&g.positions()[k], &w.vehicles[k].positions());
        eprintln!(
            "vehicle {k} after 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} \
             bent {g_bent}; positions rms {:.1} mm",
            d * 1e3
        );
        assert!(c_bent > 20, "the crash bends vehicle {k} ({c_bent})");
        let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.1 * a.max(b) as f32 + 2.0;
        assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
        assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
        // A crash is chaotic: positions are only a sanity check.
        assert!(d < 0.5, "vehicle {k}: positions {d} m apart");
    }
    // Without contact on either, the rear one drives through the front one.
    let fleet = rear_end(15.0, 0.5);
    let mut g = gpu(&fleet).unwrap();
    g.set_vehicle_contact(false);
    for _ in 0..36 {
        g.frame(17);
    }
    let through = g.positions()[1].iter().map(|p| p.z).sum::<f32>() / fleet[1].nodes.len() as f32;
    let held =
        w.vehicles[1].positions().iter().map(|p| p.z).sum::<f32>() / fleet[1].nodes.len() as f32;
    assert!(
        through > held + 1.0,
        "contact holds the rear one back ({held} vs {through})"
    );
}

#[test]
fn a_node_leaving_its_frame_bounds_is_reported() {
    // Margin 0 and a vehicle spawned at rest in the air: its nodes fall out of
    // their frame bounds (the box itself) within the frame.
    let mut a = floor(Vec3::new(3.0, 3.0, 3.0));
    for n in &mut a.nodes {
        n.flags &= !NODE_IMMOVABLE;
    }
    let b = point(Vec3::new(50.0, 0.0, 0.0), Vec3::ZERO, true);
    let Some(mut g) = gpu(&[a, b]) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_contact_margin(0.0);
    g.frame(17);
    let _ = g.positions();
    let e = g.contact_status().unwrap_err();
    assert!(e.contains("vehicle 0 left its frame bounds"), "{e}");
}
