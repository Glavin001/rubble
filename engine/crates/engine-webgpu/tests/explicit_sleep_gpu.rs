//! Vehicle sleep (`engine_softbody::explicit_sleep`) on the GPU against the CPU reference,
//! on the JBeam fixture car (pressure wheels: CPU and GPU step it bit for bit):
//! * the same frames of the same rule: every node, every frame, bit for bit, the same
//!   still counts, the same frame of falling asleep and of waking;
//! * a sleeping vehicle does no solver work: no state word changes, its step counter
//!   stands;
//! * off, and on for vehicles that never sleep, frames are bit for bit those without it;
//! * another vehicle's approach wakes a sleeping one (its contact group), on the GPU and
//!   the CPU in the same frame, and from the moment of waking it runs as one that never
//!   slept, within the frozen jitter.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_sleep::VehicleSleep;
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::PART2_CAR;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

/// The fixture car at `at` (lowest node 2 cm above its ground), at velocity `v`.
fn car(at: Vec3, v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", PART2_CAR)], None).unwrap();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_with(&d, JbeamWheels::Pressure, Vec3::ZERO, v, settings)
            .unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos += at + Vec3::Y * (0.02 - low);
    }
    s.roll_tyres(v);
    s
}

/// The fixture car parked at `at` and settled on the CPU for `secs` (its lightly damped
/// pressure wheels rock for about 50 s; `examples/sleep_probe.rs fixture=1`).
fn parked(at: Vec3, secs: f32) -> ExplicitBeams {
    let mut s = car(at, Vec3::ZERO);
    for _ in 0..(secs / engine_softbody::explicit_beams::STEP).round() as u32 {
        s.step();
    }
    s
}

fn bits(v: &[Vec3]) -> Vec<[u32; 3]> {
    v.iter().map(|x| x.to_array().map(f32::to_bits)).collect()
}

/// Steps per World frame at 120 Hz (17 and 16 alternately, as the World's clock).
fn steps(frame: u32) -> u32 {
    if frame % 2 == 0 {
        17
    } else {
        16
    }
}

#[test]
fn sleep_steps_like_the_cpu_and_freezes() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let cfg = VehicleSleep::default();
    let mut cpu = ExplicitBeamsWorld::new(vec![parked(Vec3::ZERO, 45.0)]);
    cpu.sleep = Some(cfg);
    let mut gpu = ExplicitBeamsGpu::with_sleep(&device, &queue, &cpu.vehicles, None, Some(cfg));
    let mut slept_at = None;
    let mut frame = 0u32;
    // Until asleep (at most 10 s more), then 20 frames asleep.
    while frame < 1200 {
        let k = steps(frame);
        cpu.frame(k);
        gpu.frame(k);
        frame += 1;
        let c = &cpu.vehicles[0];
        assert_eq!(
            bits(&c.positions()),
            bits(&gpu.positions()[0]),
            "frame {frame}: positions"
        );
        assert_eq!(
            bits(&c.nodes.iter().map(|n| n.vel).collect::<Vec<_>>()),
            bits(&gpu.velocities()[0]),
            "frame {frame}: velocities"
        );
        let g = gpu.sleep_states()[0];
        assert_eq!(
            (c.sleep.still, c.sleep.asleep),
            g,
            "frame {frame}: sleep state"
        );
        if g.1 && slept_at.is_none() {
            slept_at = Some(frame);
        }
        if slept_at.is_some_and(|s| frame >= s + 20) {
            break;
        }
    }
    let slept = slept_at.expect("the parked car never fell asleep in 10 s");
    eprintln!(
        "asleep after frame {slept} ({:.2} s), still count {}",
        slept as f32 / 120.0,
        cpu.vehicles[0].sleep.still
    );
    // Asleep: no state word changes, the step counter stands.
    let before = gpu.state_words();
    let steps_before = cpu.vehicles[0].steps;
    for f in 0..10 {
        cpu.frame(steps(frame + f));
        gpu.frame(steps(frame + f));
    }
    frame += 10;
    assert_eq!(
        cpu.vehicles[0].steps, steps_before,
        "CPU stepped a sleeping vehicle"
    );
    for ((name, a), (_, b)) in before.iter().zip(gpu.state_words()) {
        assert!(*a == b, "a sleeping vehicle's {name} changed");
    }
    // Woken (a host write), both resume from the frozen state alike.
    cpu.wake(0);
    gpu.wake(0);
    for f in 0..60 {
        cpu.frame(steps(frame + f));
        gpu.frame(steps(frame + f));
        let c = &cpu.vehicles[0];
        assert_eq!(
            bits(&c.positions()),
            bits(&gpu.positions()[0]),
            "after waking, frame {f}: positions"
        );
        assert_eq!(
            (c.sleep.still, c.sleep.asleep),
            gpu.sleep_states()[0],
            "after waking, frame {f}"
        );
    }
    assert!(
        cpu.vehicles[0].steps > steps_before,
        "woken but not stepped"
    );
}

#[test]
fn sleep_off_and_awake_vehicles_are_bit_for_bit() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // A car driving (never still) and one settling, apart: contact active (two vehicles),
    // no group. Three fleets: no sleep room, room with sleep off, sleep on.
    let fleet = vec![
        car(Vec3::ZERO, Vec3::new(0.0, 0.0, 8.0)),
        car(Vec3::new(30.0, 0.0, 0.0), Vec3::ZERO),
    ];
    let mut plain = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    let mut off =
        ExplicitBeamsGpu::with_sleep(&device, &queue, &fleet, None, Some(VehicleSleep::default()));
    assert!(!off.set_sleep(false));
    let mut on =
        ExplicitBeamsGpu::with_sleep(&device, &queue, &fleet, None, Some(VehicleSleep::default()));
    for f in 0..240 {
        for g in [&mut plain, &mut off, &mut on] {
            g.frame(steps(f));
        }
        let (p, o, n) = (plain.positions(), off.positions(), on.positions());
        for v in 0..2 {
            assert_eq!(
                bits(&p[v]),
                bits(&o[v]),
                "frame {f} vehicle {v}: sleep off differs"
            );
        }
        // The driving car never sleeps: bit for bit with sleep on.
        assert_eq!(
            bits(&p[0]),
            bits(&n[0]),
            "frame {f}: the awake vehicle differs with sleep on"
        );
        if !on.sleep_states()[1].1 {
            assert_eq!(
                bits(&p[1]),
                bits(&n[1]),
                "frame {f}: the settling vehicle differs before it sleeps"
            );
        }
    }
}

#[test]
fn an_approaching_vehicle_wakes_a_sleeping_one() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let cfg = VehicleSleep::default();
    // A parked car, and one 40 m behind it that sets off at 6 m/s once the first sleeps.
    let fleet = vec![
        parked(Vec3::ZERO, 45.0),
        parked(Vec3::new(0.0, 0.0, -40.0), 45.0),
    ];
    let mut cpu = ExplicitBeamsWorld::new(fleet.clone());
    cpu.sleep = Some(cfg);
    let mut gpu = ExplicitBeamsGpu::with_sleep(&device, &queue, &fleet, None, Some(cfg));
    let mut never = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    let mut frame = 0u32;
    let mut launched = None;
    let mut woke = None;
    let launch = |g: &ExplicitBeamsGpu| {
        for i in 0..fleet[1].nodes.len() as u32 {
            g.set_velocity(1, i, Vec3::new(0.0, 0.0, 6.0));
        }
    };
    while frame < 3600 && woke.is_none() {
        let k = steps(frame);
        // The World's plan schedule: a contact plan every frame.
        gpu.request_contact_replan();
        never.request_contact_replan();
        cpu.frame(k);
        gpu.frame(k);
        never.frame(k);
        frame += 1;
        let (c, g) = (&cpu.vehicles, gpu.sleep_states());
        assert_eq!(
            g[0],
            (c[0].sleep.still, c[0].sleep.asleep),
            "frame {frame}: the parked car's sleep state"
        );
        if launched.is_none() && g[0].1 && g[1].1 {
            for n in &mut cpu.vehicles[1].nodes {
                n.vel = Vec3::new(0.0, 0.0, 6.0);
            }
            cpu.wake(1);
            launch(&gpu);
            launch(&never);
            launched = Some(frame);
        }
        if launched.is_some() && !g[0].1 {
            woke = Some(frame);
        }
    }
    let (launched, woke) = (
        launched.expect("never both asleep"),
        woke.expect("the parked car never woke"),
    );
    let gap = {
        let p = gpu.positions();
        let front = p[1].iter().map(|x| x.z).fold(f32::MIN, f32::max);
        let back = p[0].iter().map(|x| x.z).fold(f32::MAX, f32::min);
        back - front
    };
    eprintln!("launched at frame {launched}, the parked car woke at frame {woke} with {gap:.2} m between them");
    assert!(gap > 0.0, "woke after contact");
    // The woken car against one that never slept: the frozen jitter.
    let (a, b) = (gpu.positions(), never.positions());
    let d = a[0]
        .iter()
        .zip(&b[0])
        .map(|(x, y)| (*x - *y).abs().max_element())
        .fold(0.0f32, f32::max);
    eprintln!("woken against never slept, at waking: {:.3} mm", d * 1e3);
    assert!(d < 0.5e-3, "{d} m");
    // Up to the cars' contact (and 0.1 s past it).
    let mut worst = d;
    for f in 0..240 {
        gpu.request_contact_replan();
        never.request_contact_replan();
        gpu.frame(steps(frame + f));
        never.frame(steps(frame + f));
        let (a, b) = (gpu.positions(), never.positions());
        worst = worst.max(
            a[0].iter()
                .zip(&b[0])
                .map(|(x, y)| (*x - *y).abs().max_element())
                .fold(0.0f32, f32::max),
        );
        let front = a[1].iter().map(|x| x.z).fold(f32::MIN, f32::max);
        let back = a[0].iter().map(|x| x.z).fold(f32::MAX, f32::min);
        if front > back + 0.6 {
            eprintln!(
                "contact plus 0.1 s at frame {}: woken against never slept {:.3} mm at most",
                frame + f,
                worst * 1e3
            );
            break;
        }
    }
}
