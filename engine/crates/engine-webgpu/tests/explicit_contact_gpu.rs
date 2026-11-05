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
    triangle_throttle: true,
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
    // After it, broken and bent counts: the GPU is not deterministic (atomic
    // order) and the crash is chaotic, so the median of 5 GPU runs.
    let mut runs = vec![(g.beams(), g.positions())];
    for _ in 0..4 {
        let mut g = gpu(&rear_end(15.0, 0.5)).unwrap();
        for _ in 0..36 {
            g.frame(17);
        }
        g.wait();
        runs.push((g.beams(), g.positions()));
    }
    for k in 0..2 {
        let (c_broken, c_bent) = w.vehicles[k].damage();
        let median = |f: &dyn Fn(&engine_webgpu::GpuBeam) -> bool| {
            let mut v: Vec<usize> = runs
                .iter()
                .map(|r| r.0[k].iter().filter(|b| f(b)).count())
                .collect();
            v.sort();
            v[v.len() / 2]
        };
        let g_broken = median(&|b| b.broken);
        let g_bent = median(&|b| b.bent());
        let d = rms(&runs[0].1[k], &w.vehicles[k].positions());
        eprintln!(
            "vehicle {k} after 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU median broken \
             {g_broken} bent {g_bent}; positions rms {:.1} mm",
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

#[test]
fn a_group_over_the_limit_runs_alone_and_is_reported() {
    // A chain of 70 still single-node vehicles 8 cm apart (margin 0: each padded
    // box overlaps only its neighbours'): one group of 70, more than 64.
    let fleet: Vec<ExplicitBeams> = (0..70)
        .map(|k| point(Vec3::new(0.08 * k as f32, 5.0, 0.0), Vec3::ZERO, true))
        .collect();
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_contact_margin(0.0);
    g.frame(17);
    let e = g.contact_status().unwrap_err();
    assert!(e.contains("more than 64"), "{e}");
    // 60 are one group: no error.
    let mut g = gpu(&fleet[..60]).unwrap();
    g.set_contact_margin(0.0);
    g.frame(17);
    g.contact_status().unwrap();
}

/// A node leaving its frame bounds never stops a running sim: the error is handed
/// to the host once ([`ExplicitBeamsGpu::take_contact_errors`]), the margin widens
/// so later frames cover the fall, and frames go on. (The CPU reference has no
/// frame bounds: it tests every pair every step.)
#[test]
fn a_running_sim_survives_leaving_its_frame_bounds() {
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
    let mut errors = Vec::new();
    for _ in 0..20 {
        g.frame(17);
        g.wait();
        errors.extend(g.take_contact_errors());
    }
    assert!(!errors.is_empty() && errors[0].contains("left its frame bounds"), "{errors:?}");
    // The margin widened: the last frames report nothing more.
    for _ in 0..5 {
        g.frame(17);
        g.wait();
    }
    assert!(g.take_contact_errors().is_empty());
    assert!(g.positions().iter().flatten().all(|p| p.is_finite()));
}

/// A procedural sedan at `speed` m/s into a standing pressure-wheel JBeam car (the
/// fixture car, whose tyres carry LBEAMs, spec §24; every node a contacter) `gap` m
/// ahead: [sedan, car].
fn sedan_into_lbeam_car(speed: f32, gap: f32) -> Vec<ExplicitBeams> {
    use engine_softbody::jbeam::load_jbeam;
    let settings = ExplicitBeamsSettings { turbulence: false, ..Default::default() };
    let desc = load_jbeam(&[("car.jbeam", engine_softbody::jbeam_fixture::PART2_CAR)], None).unwrap();
    let (mut car, _) = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    assert!(car.beams.iter().any(|b| b.kind == ExplicitKind::JbeamLbeam), "the fixture's tyres have LBEAMs");
    assert!(car.two_body_contact);
    let text = sedan_text();
    let v = Vec3::new(0.0, 0.0, speed);
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let slow = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let front = probe.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let mut sedan = ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - slow, 0.0), v, settings).unwrap();
    sedan.roll_tyres(v);
    // The car broadside on, its nearest node `gap` ahead of the sedan's front.
    let low = car.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let back = car.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
    for n in &mut car.nodes {
        n.pos += Vec3::new(0.0, 0.01 - low, front + gap - back);
    }
    vec![sedan, car]
}

fn com(v: &ExplicitBeams, pos: &[Vec3], vel: &[Vec3]) -> (Vec3, Vec3) {
    let m: f32 = v.nodes.iter().map(|n| n.mass).sum();
    let p = pos.iter().zip(&v.nodes).map(|(p, n)| *p * n.mass).sum::<Vec3>() / m;
    let u = vel.iter().zip(&v.nodes).map(|(u, n)| *u * n.mass).sum::<Vec3>() / m;
    (p, u)
}

/// Pressure wheels' LBEAMs do not cost vehicle contact (2026-11-05: the barrier's
/// LBEAMs, a follower force before spec §24's gradient form, flew its face apart in
/// free flight and it passed through the car). A sedan hits a pressure-wheel JBeam
/// car broadside: the car is pushed and the sedan never passes its centre, on the
/// CPU and the GPU (with two-body contact, §10.3: the JBeam car opts in).
#[test]
fn a_pressure_wheel_jbeam_car_is_pushed_by_the_vehicle_that_hits_it() {
    let fleet = sedan_into_lbeam_car(10.0, 0.3);
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet.clone());
    let mut first = None;
    for f in 0..60 {
        w.frame(17);
        g.frame(17);
        if w.first_hit_step.is_some() && first.is_none() {
            first = Some(f);
            g.wait();
            let p = g.positions();
            for k in 0..2 {
                let d = rms(&p[k], &w.vehicles[k].positions());
                assert!(d < 5e-4, "vehicle {k}: GPU–CPU {d} m at contact");
            }
        }
    }
    g.wait();
    g.contact_status().unwrap();
    assert!(first.is_some(), "they met");
    let cpu_vel: Vec<Vec<Vec3>> = w.vehicles.iter().map(|v| v.nodes.iter().map(|n| n.vel).collect()).collect();
    let cpu_pos: Vec<Vec<Vec3>> = w.vehicles.iter().map(|v| v.positions()).collect();
    for (label, pos, vel) in [("CPU", cpu_pos, cpu_vel), ("GPU", g.positions(), g.velocities())] {
        let (ps, vs) = com(&fleet[0], &pos[0], &vel[0]);
        let (pc, vc) = com(&fleet[1], &pos[1], &vel[1]);
        let sedan_front = pos[0].iter().map(|p| p.z).fold(f32::MIN, f32::max);
        eprintln!(
            "{label} after 1 s: sedan centre z {:.2} m ({:.2} m/s), front {sedan_front:.2} m; car centre {:.2} m ({:.2} m/s); CPU hits {}",
            ps.z, vs.z, pc.z, vc.z, w.total.hits
        );
        assert!(pos.iter().flatten().all(|p| p.is_finite()), "{label}: finite");
        assert!(vc.z > 0.5, "{label}: the car is pushed ({} m/s)", vc.z);
        assert!(sedan_front < pc.z, "{label}: the sedan passes into the car ({sedan_front} vs {})", pc.z);
    }
}

/// Two-body contact (jbeam-semantics-spec.md §10.3) on the GPU as on the CPU: three
/// heavy points (10 kg) falling onto one light, free triangle (0.1, 0.2, 5 kg) of
/// another vehicle, every step, a frame of 1 and of 17 steps.
#[test]
fn two_body_contact_steps_like_the_cpu() {
    let mut tri = vehicle(&[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 2.0)]);
    for (n, m) in tri.nodes.iter_mut().zip([0.1, 0.2, 5.0]) {
        n.mass = m;
    }
    tri.triangles.push(Triangle { nodes: [0, 1, 2], wait: 0, idle: 0 });
    tri.finish();
    tri.two_body_contact = true;
    let mut pts = vehicle(&[Vec3::new(0.1, 0.02, 0.0), Vec3::new(-0.3, 0.025, -0.5), Vec3::new(0.2, 0.015, 0.8)]);
    for n in &mut pts.nodes {
        n.mass = 10.0;
        n.vel = Vec3::new(0.0, -1.0, 0.0);
    }
    pts.finish();
    // (The contact-force log on, spec §32: vehicle contact's forces logged alike.)
    tri.enable_contact_force();
    pts.enable_contact_force();
    let fleet = vec![tri, pts];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet);
    for (frames, steps) in [(30, 1), (6, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            g.frame(steps);
        }
    }
    g.wait();
    g.contact_status().unwrap();
    assert!(w.total.hits > 10, "hits {}", w.total.hits);
    for k in 0..2 {
        let d = rms(&g.positions()[k], &w.vehicles[k].positions());
        eprintln!("vehicle {k}: GPU–CPU {d:.2e} m after {} steps, {} hits", w.steps, w.total.hits);
        assert!(d < 1e-4, "vehicle {k}: GPU–CPU {d} m");
    }
    let logs = g.contact_forces();
    for k in 0..2 {
        let (_, peak, _) = logs[k].clone().unwrap();
        let cpu = w.vehicles[k].take_contact_force_peak().unwrap();
        eprintln!("vehicle {k}: contact-force peaks CPU {cpu:?} GPU {peak:?} N");
        assert!(cpu.iter().any(|&f| f > 1.0));
        for (a, b) in cpu.iter().zip(&peak) {
            assert!((a - b).abs() <= 0.1 + 1e-3 * a, "CPU {a} GPU {b}");
        }
    }
}

/// The contact-force log and a contact break (jbeam-semantics-spec.md §32) on the
/// GPU as on the CPU: a two-node pane (break group 1, 50 N on node 0) dropped
/// 0.5 m onto the ground breaks in the same step, and a second pane resting on
/// the ground logs the same contact forces; one-step frames, then 17-step ones.
#[test]
fn contact_breaks_step_like_the_cpu() {
    let pane = |x0: f32, y: f32| {
        let mut s = vehicle(&[Vec3::new(x0, y, 0.0), Vec3::new(x0 + 1.0, y, 0.0)]);
        s.ground = 0.0;
        let mut b = Beam::new(0, 1, ExplicitKind::Plain, ExplicitType::Deformable, 1.0e5, 100.0, 1.0, 1.0e9, 1.0e9, 0.0);
        b.group = 1;
        s.beams.push(b);
        s.finish();
        s
    };
    let mut drop = pane(0.0, 0.5);
    drop.set_contact_breaks(&[(1, 50.0, vec![0])]).unwrap();
    let mut rest = pane(5.0, 0.005);
    rest.enable_contact_force();
    let fleet = vec![drop, rest];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet);
    let (mut cpu_at, mut gpu_at) = (None, None);
    for k in 0..1000u32 {
        w.frame(1);
        g.frame(1);
        g.wait();
        if cpu_at.is_none() && w.vehicles[0].beams[0].broken {
            cpu_at = Some(k);
        }
        if gpu_at.is_none() && g.beams()[0][0].broken {
            gpu_at = Some(k);
        }
    }
    for _ in 0..20 {
        w.frame(17);
        g.frame(17);
    }
    g.wait();
    eprintln!("broke: CPU step {cpu_at:?}, GPU step {gpu_at:?}");
    assert!(cpu_at.is_some());
    assert_eq!(cpu_at, gpu_at);
    let logs = g.contact_forces();
    let (last, peak, states) = logs[0].clone().unwrap();
    assert_eq!(states[1], 2);
    let cpu = w.vehicles[1].contact_forces().unwrap();
    let (gl, _, _) = logs[1].clone().unwrap();
    eprintln!("resting pane: CPU {cpu:?} N, GPU {gl:?} N; dropped pane peak {peak:?} last {last:?}");
    for (a, b) in cpu.iter().zip(&gl) {
        assert!((a - b).abs() <= 0.05 + 1e-3 * a.abs(), "CPU {a} GPU {b}");
    }
    let cpu_peak = w.vehicles[0].take_contact_force_peak().unwrap();
    assert!((cpu_peak[0] - peak[0]).abs() <= 0.05 + 1e-3 * cpu_peak[0], "peak CPU {} GPU {}", cpu_peak[0], peak[0]);
}

/// Self-collision's two-body rule (jbeam-semantics-spec.md §36) on the GPU against the
/// CPU: the authored sedan (low resolution) at 15 m/s into a wall, two-body contact on (self and
/// vehicle), the BeamNG profile's no-throttle switch on. Through the frame its
/// self-contacts start the two agree to a fraction of a millimetre; after 0.6 s
/// (chaotic, GPU sums order-free in fixed point) broken and bent counts close.
#[test]
fn self_collision_two_body_steps_like_the_cpu() {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let b = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Low,
            ..Default::default()
        },
    )
    .unwrap();
    let text = b.truck;
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let v = Vec3::new(0.0, 0.0, 15.0);
    let mut s = ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, EXACT).unwrap();
    s.roll_tyres(v);
    s.two_body_contact = true;
    s.settings.triangle_throttle = false;
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    s.finish();
    let Some(mut g) = gpu(std::slice::from_ref(&s)) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut first = None;
    let mut contacts = 0u32;
    for f in 0..36 {
        let mut hit = false;
        for _ in 0..17 {
            s.step();
            hit |= s.last_contacts > 0;
            contacts += s.last_contacts;
        }
        g.frame(17);
        if hit && first.is_none() {
            first = Some(f);
            g.wait();
            let d = rms(&g.positions()[0], &s.positions());
            eprintln!("frame {f} (self-contacts start): GPU–CPU {:.4} mm", d * 1e3);
            assert!(d < 5e-4, "{d} m when self-contact starts");
        }
    }
    g.wait();
    assert!(first.is_some() && contacts > 100, "self-contacts: {contacts}");
    let (cb, cn) = s.damage();
    let gb = g.beams()[0].iter().filter(|b| b.broken).count();
    let gn = g.beams()[0].iter().filter(|b| b.bent()).count();
    let d = rms(&g.positions()[0], &s.positions());
    eprintln!("after 0.6 s ({contacts} CPU self-contacts): CPU broken {cb} bent {cn}; GPU broken {gb} bent {gn}; rms {:.1} mm", d * 1e3);
    let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.1 * a.max(b) as f32 + 2.0;
    assert!(close(cb, gb) && close(cn, gn));
}

/// Anti-clip (jbeam-semantics-spec.md §36) on the GPU against the CPU, step by step
/// with one-step frames, then 17-step ones: points at 30–150 m/s through another
/// vehicle's floor (the swept test catches those faster than the band per step),
/// a self-collision node through its own vehicle's triangle, and an anti-clip-only
/// (NONCOLLIDABLE) triangle. The side memories are equal word for word after every
/// frame (the contact decisions), the positions to the accumulator's rounding.
#[test]
fn anti_clip_steps_like_the_cpu() {
    // Three vehicles (a small contact group), then six (the owner-by-owner pass).
    anti_clip_scene(0);
    anti_clip_scene(3);
}

fn anti_clip_scene(extra: usize) {
    let mut fl = floor(Vec3::new(3.0, 3.0, 3.0));
    fl.anti_clip = true;
    fl.two_body_contact = true;
    fl.settings.triangle_throttle = false;
    let mut pts = vehicle(&[
        Vec3::new(0.1, 0.3, 0.0),
        Vec3::new(-0.3, 0.2, -0.5),
        Vec3::new(0.2, 0.25, 0.8),
        Vec3::new(0.0, -0.2, 0.1),
    ]);
    for (n, v) in pts.nodes.iter_mut().zip([-30.0, -70.0, -150.0, 100.0]) {
        n.vel = Vec3::new(0.0, v, 0.0);
        n.flags |= NODE_NO_GROUND | NODE_CONTACTER;
    }
    pts.finish();
    // Self-collision: a node through its own vehicle's triangle at 90 m/s, and one
    // falling onto an anti-clip-only triangle.
    let mut own = vehicle(&[
        Vec3::new(10.0, 0.0, -1.0),
        Vec3::new(12.0, 0.0, -1.0),
        Vec3::new(11.0, 0.0, 2.0),
        Vec3::new(11.0, 0.3, 0.0),
        Vec3::new(10.0, 0.0, 3.0),
        Vec3::new(12.0, 0.0, 3.0),
        Vec3::new(11.0, 0.0, 6.0),
        Vec3::new(11.0, 0.05, 4.0),
    ]);
    own.settings.triangle_throttle = false;
    for i in [0, 1, 2, 4, 5, 6] {
        own.nodes[i].flags |= NODE_IMMOVABLE | NODE_ON_TRIANGLE;
    }
    for (i, v) in [(3, -90.0), (7, -1.0)] {
        own.nodes[i].flags |= NODE_CONTACTER | NODE_NO_GROUND;
        own.nodes[i].vel = Vec3::new(0.0, v, 0.0);
    }
    own.triangles.push(Triangle {
        nodes: [0, 1, 2],
        wait: 0,
        idle: 0,
    });
    own.triangles.push(Triangle {
        nodes: [4, 5, 6],
        wait: 0,
        idle: 0,
    });
    own.anti_only_tris = vec![false, true];
    own.anti_clip = true;
    own.two_body_contact = true;
    own.finish();
    let mut fleet = vec![fl, pts, own];
    for k in 0..extra {
        let mut p = point(
            Vec3::new(-0.5 + 0.3 * k as f32, 0.4, -0.2),
            Vec3::new(0.0, -120.0, 0.0),
            false,
        );
        p.nodes[0].flags |= NODE_NO_GROUND;
        fleet.push(p);
    }
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut w = ExplicitBeamsWorld::new(fleet);
    let mut recorded = 0usize;
    for (frames, steps) in [(60, 1), (6, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            g.frame(steps);
            let gs = g.clip_sides();
            for k in 0..w.vehicles.len() {
                let cs = &w.vehicles[k].clip_sides;
                for (i, c) in gs[k].iter().enumerate() {
                    let cpu = cs
                        .get(i)
                        .copied()
                        .unwrap_or([engine_softbody::explicit_contact::CLIP_EMPTY; 4]);
                    assert_eq!(
                        *c, cpu,
                        "vehicle {k} node {i} side memory after {} steps",
                        w.steps
                    );
                    recorded += c[2..]
                        .iter()
                        .filter(|&&x| x != engine_softbody::explicit_contact::CLIP_EMPTY)
                        .count();
                }
            }
        }
    }
    g.wait();
    g.contact_status().unwrap();
    assert!(recorded >= 10, "contacts recorded: {recorded}");
    let pos = g.positions();
    for k in 0..w.vehicles.len() {
        let d = rms(&pos[k], &w.vehicles[k].positions());
        eprintln!("vehicle {k}: GPU–CPU {d:.2e} m after {} steps", w.steps);
        assert!(d < 1e-4, "vehicle {k}: GPU–CPU {d} m");
    }
    // Every fast point is above the floor (caught on its side), the self-collision
    // node above its triangle, the other held at the anti-clip-only plane.
    for (i, p) in pos[1].iter().enumerate() {
        let above = i != 3;
        assert!((p.y > -0.01) == above, "point {i} at y {}", p.y);
    }
    for k in 3..3 + extra {
        assert!(pos[k][0].y > -0.01, "extra point {k} at y {}", pos[k][0].y);
    }
    eprintln!(
        "{} vehicles: {recorded} contacts recorded, {} contact groups",
        w.vehicles.len(),
        g.contact_groups()
    );
    assert!(
        pos[2][3].y > -0.01,
        "self-collision node at {}",
        pos[2][3].y
    );
    assert!(
        pos[2][7].y.abs() < 0.005,
        "anti-clip-only plane: {}",
        pos[2][7].y
    );
}
