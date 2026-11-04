//! The node grab on the GPU (`ExplicitBeamsGpu::set_grab`) against the CPU
//! reference (`ExplicitBeams::grab`): a lone grabbed node steps bit for bit like
//! the CPU in the one-workgroup frame and in a contact group (vehicle contact on,
//! two vehicles whose boxes overlap); the wide passes leave the frame's state bit
//! for bit with a grab on; a grabbed sedan agrees with the CPU to millimetres and a
//! strong grab tears it on both; releasing the grab stops the force.

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

/// Unit-mass nodes at `points` (flags as given), finished.
fn vehicle(points: &[Vec3], flags: u32) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    for &p in points {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags,
        });
    }
    s.finish();
    s
}

/// An immovable collision triangle in y = 0.
fn floor() -> ExplicitBeams {
    let mut s = vehicle(
        &[
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ],
        NODE_IMMOVABLE,
    );
    s.triangles.push(Triangle {
        nodes: [0, 1, 2],
        wait: 0,
        idle: 0,
    });
    s.finish();
    s
}

fn sedan(at: Vec3) -> ExplicitBeams {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    ExplicitBeams::from_truck(
        &text,
        at + Vec3::new(0.0, 0.01 - low, 0.0),
        Vec3::ZERO,
        settings,
    )
    .unwrap()
}

fn rms_max(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        sum += f64::from(d).powi(2);
        max = max.max(d);
    }
    ((sum / a.len().max(1) as f64).sqrt() as f32, max)
}

fn world(gpu: &ExplicitBeamsGpu, v: usize) -> Vec<Vec3> {
    let o = gpu.origins()[v];
    gpu.positions()[v].iter().map(|p| o + *p).collect()
}

#[test]
fn a_grabbed_node_steps_like_the_cpu_bit_for_bit() {
    let mut cpu = vehicle(&[Vec3::new(0.3, 1.0, -0.2)], NODE_NO_GROUND);
    let Some(mut g) = gpu(std::slice::from_ref(&cpu)) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let grab = Grab {
        node: 0,
        target: Vec3::new(1.5, 2.0, 0.5),
        k: 300.0,
        c: 4.0,
    };
    cpu.grab = Some(grab);
    g.set_grab(Some((0, grab)));
    for (frames, steps) in [(20, 1), (10, 17), (5, 34)] {
        for _ in 0..frames {
            for _ in 0..steps {
                cpu.step();
            }
            g.frame(steps);
        }
        g.wait();
        assert_eq!(
            g.positions()[0],
            cpu.positions(),
            "after {} steps",
            cpu.steps
        );
        assert_eq!(g.velocities()[0][0], cpu.nodes[0].vel);
    }
    // It moved: pulled up from where gravity alone would have dropped it.
    assert!(cpu.nodes[0].pos.y > 1.0, "{}", cpu.nodes[0].pos);
    // Released: gravity alone from here on, on both.
    cpu.grab = None;
    g.set_grab(None);
    for _ in 0..10 {
        for _ in 0..17 {
            cpu.step();
        }
        g.frame(17);
    }
    g.wait();
    assert_eq!(g.positions()[0], cpu.positions());
}

#[test]
fn a_grab_in_a_contact_group_steps_like_the_cpu() {
    // A point over an immovable triangle: their boxes overlap, so the GPU steps
    // them as a contact group; the grab lifts the point and swings it.
    let fleet = vec![
        floor(),
        vehicle(&[Vec3::new(0.2, 0.3, 0.1)], NODE_NO_GROUND),
    ];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    assert!(g.contact_active());
    let mut w = ExplicitBeamsWorld::new(fleet);
    let grab = Grab {
        node: 0,
        target: Vec3::new(0.5, 0.6, 0.3),
        k: 200.0,
        c: 2.0,
    };
    w.vehicles[1].grab = Some(grab);
    g.set_grab(Some((1, grab)));
    for (frames, steps) in [(30, 1), (12, 17)] {
        for _ in 0..frames {
            w.frame(steps);
            g.frame(steps);
        }
        g.wait();
        let (d, _) = rms_max(&world(&g, 1), &w.vehicles[1].positions());
        assert!(d < 1e-6, "GPU–CPU {d} m after {} steps", w.steps);
    }
    g.contact_status().unwrap();
    let p = w.vehicles[1].origin + w.vehicles[1].nodes[0].pos;
    assert!(p.y > 0.4, "the grab lifted it ({p})");
}

#[test]
fn wide_passes_match_the_frame_bitwise_with_a_grab() {
    let fleet = [sedan(Vec3::new(50.0, 0.0, 50.0))];
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_vehicle_contact(false);
    if !g.set_wide(true) {
        eprintln!("the sedan does not run the wide passes: skipped");
        return;
    }
    let front = (0..fleet[0].nodes.len())
        .max_by(|&a, &b| fleet[0].nodes[a].pos.z.total_cmp(&fleet[0].nodes[b].pos.z))
        .unwrap();
    let start = fleet[0].origin + fleet[0].nodes[front].pos;
    g.set_grab(Some((
        0,
        Grab {
            node: front as u32,
            target: start + Vec3::new(0.0, 1.5, 3.0),
            k: 1.0e5,
            c: 0.5 * fleet[0].nodes[front].mass / STEP,
        },
    )));
    for f in 0..40 {
        let snap = g.snapshot();
        g.set_wide(false);
        g.frame(17);
        g.wait();
        let a = g.state_words();
        g.restore(&snap);
        g.set_wide(true);
        g.frame(17);
        g.wait();
        let b = g.state_words();
        for ((name, x), (_, y)) in a.iter().zip(&b) {
            if let Some(i) = x.iter().zip(y).position(|(p, q)| p != q) {
                panic!("frame {f}: the wide passes differ from the frame in {name} word {i}");
            }
        }
    }
    let moved = world(&g, 0)[front].distance(start);
    eprintln!("wide = frame with a grab; the grabbed node moved {moved:.2} m");
    assert!(moved > 0.5);
}

#[test]
fn a_grabbed_sedan_matches_the_cpu() {
    for (k, lift, frames, tears) in [
        (2.0e4f32, Vec3::new(0.0, 0.6, 0.0), 60, false),
        (2.0e5, Vec3::new(0.0, 2.0, 4.0), 60, true),
    ] {
        let mut cpu = sedan(Vec3::ZERO);
        let Some(mut g) = gpu(std::slice::from_ref(&cpu)) else {
            eprintln!("no GPU: skipped");
            return;
        };
        g.set_vehicle_contact(false);
        g.set_wide(false);
        let front = (0..cpu.nodes.len())
            .max_by(|&a, &b| cpu.nodes[a].pos.z.total_cmp(&cpu.nodes[b].pos.z))
            .unwrap();
        let start = cpu.origin + cpu.nodes[front].pos;
        let grab = Grab {
            node: front as u32,
            target: start + lift,
            k,
            c: 0.5 * cpu.nodes[front].mass / STEP,
        };
        cpu.grab = Some(grab);
        g.set_grab(Some((0, grab)));
        let mut clock = FrameClock::default();
        for _ in 0..frames {
            cpu.frame();
            g.frame(clock.next());
        }
        g.wait();
        let (c_broken, c_bent) = cpu.damage();
        let beams = &g.beams()[0];
        let g_broken = beams.iter().filter(|b| b.broken).count();
        let g_bent = beams.iter().filter(|b| b.bent()).count();
        let (rms, max) = rms_max(&cpu.positions(), &g.positions()[0]);
        let moved = (cpu.origin + cpu.nodes[front].pos).distance(start);
        eprintln!(
            "grab k {k}: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
             positions rms {:.2} mm max {:.2} mm; node moved {moved:.2} m",
            rms * 1e3,
            max * 1e3
        );
        assert!(
            moved > if tears { 0.3 } else { 0.01 },
            "the grab moves the node"
        );
        if tears {
            assert!(
                c_broken > 0 && g_broken > 0,
                "a strong grab tears it on both"
            );
            let close =
                |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.1 * a.max(b) as f32 + 3.0;
            assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
            assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
        } else {
            assert_eq!((c_broken, g_broken), (0, 0));
            assert!(rms < 3e-3 && max < 1e-2, "rms {rms} max {max}");
        }
    }
}
