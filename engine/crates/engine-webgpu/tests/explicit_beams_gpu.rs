//! Explicit beams on the GPU against the CPU reference
//! (`engine_softbody::explicit_beams`) on the procedural crash-test sedan: at rest
//! the two agree to millimetres, and a 56 km/h crash into a rigid box breaks and
//! bends the same number of beams. Turbulence is off on both (the GPU draws its
//! noise differently).

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, StaticBox,
};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(velocity: Vec3) -> ExplicitBeams {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    // Lowest node 1 cm above the ground.
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, velocity, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), velocity, settings).unwrap()
}

fn rms_max(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        sum += (d as f64).powi(2);
        max = max.max(d);
    }
    ((sum / a.len() as f64).sqrt() as f32, max)
}

/// Run both for `frames` 60 fps frames; returns the GPU host.
fn run_both(cpu: &mut ExplicitBeams, frames: u32) -> Option<ExplicitBeamsGpu> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(cpu));
    let mut clock = FrameClock::default();
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
    }
    gpu.wait();
    Some(gpu)
}

#[test]
fn sedan_at_rest_matches_cpu() {
    let mut cpu = sedan(Vec3::ZERO);
    let Some(gpu) = run_both(&mut cpu, 60) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "at rest after 1 s: GPU vs CPU rms {:.3} mm, max {:.3} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(rms < 3e-3, "rms {rms} m");
    assert!(max < 1e-2, "max {max} m");
    // It is resting: nothing broke and the speed is small.
    assert_eq!(cpu.damage().0, 0);
    let vmax = gpu.velocities()[0]
        .iter()
        .map(|v| v.length())
        .fold(0.0, f32::max);
    assert!(vmax < 0.5, "GPU node speed {vmax} m/s at rest");
}

#[test]
fn sedan_crash_matches_cpu_counts() {
    let speed = 56.0 / 3.6;
    let mut cpu = sedan(Vec3::new(0.0, 0.0, speed));
    let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    // A rigid block 5 cm in front of the car, wider and taller than it.
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    let Some(gpu) = run_both(&mut cpu, 36) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (c_broken, c_bent) = cpu.damage();
    let beams = &gpu.beams()[0];
    let g_broken = beams.iter().filter(|b| b.broken).count();
    let g_bent = beams.iter().filter(|b| b.bent()).count();
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "crash 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
         positions rms {:.1} mm max {:.1} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(c_bent > 100, "the crash bends the front ({c_bent})");
    let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 1.0;
    assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
    assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
}
