//! The showcase sedan (`engine_softbody::authoring`) on `ExplicitBeamsGpu` against
//! the CPU reference: at rest, into a wall at 56 km/h, and sideways into a pole at
//! 32 km/h. Turbulence off on both (the GPU draws its noise differently).

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, StaticBox, StaticShape,
};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(velocity: Vec3) -> ExplicitBeams {
    let b = build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Low,
            ..Default::default()
        },
    )
    .unwrap();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, velocity, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut s = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::new(0.0, 0.02 - low, 0.0),
        velocity,
        settings,
    )
    .unwrap();
    s.roll_tyres(velocity);
    s
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

fn close(a: usize, b: usize, rel: f32) -> bool {
    (a as f32 - b as f32).abs() <= rel * a.max(b) as f32 + 3.0
}

#[test]
fn showcase_sedan_at_rest_matches_cpu() {
    let mut cpu = sedan(Vec3::ZERO);
    let Some(gpu) = run_both(&mut cpu, 120) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    let vmax = gpu.velocities()[0]
        .iter()
        .map(|v| v.length())
        .fold(0.0, f32::max);
    let g_broken = gpu.beams()[0].iter().filter(|b| b.broken).count();
    eprintln!(
        "(a) at rest 2 s: GPU vs CPU rms {:.3} mm, max {:.3} mm; GPU broken {g_broken}, max node speed {vmax:.2} m/s",
        rms * 1e3,
        max * 1e3
    );
    // Millimetres overall; a tyre node chattering on the ground can differ by
    // centimetres (contact is chaotic at rest; native wgpu is not bit-exact).
    assert!(rms < 1e-2, "rms {rms} m");
    assert!(max < 0.1, "max {max} m");
    assert_eq!(cpu.damage().0, 0);
    assert_eq!(g_broken, 0);
    assert!(vmax < 2.0);
}

#[test]
fn showcase_sedan_wall_crash_matches_cpu_counts() {
    let mut cpu = sedan(Vec3::new(0.0, 0.0, 56.0 / 3.6));
    let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
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
    let gp = &gpu.positions()[0];
    let c_front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    let g_back = gp.iter().map(|p| p.z).fold(f32::MAX, f32::min);
    let c_back = cpu.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min);
    eprintln!(
        "(b) wall 56 km/h, 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; rear end CPU {c_back:.3} GPU {g_back:.3} m (front {c_front:.3})"
    );
    assert!(c_bent > 200);
    assert!(
        close(c_broken, g_broken, 0.15),
        "broken {c_broken} vs {g_broken}"
    );
    assert!(close(c_bent, g_bent, 0.1), "bent {c_bent} vs {g_bent}");
    assert!(
        (c_back - g_back).abs() < 0.1,
        "rear end {c_back} vs {g_back}"
    );
}

#[test]
fn showcase_sedan_side_pole_matches_cpu_counts() {
    let mut cpu = sedan(Vec3::new(32.0 / 3.6, 0.0, 0.0));
    let side = cpu.positions().iter().map(|p| p.x).fold(f32::MIN, f32::max);
    cpu.shapes.push(StaticShape::cylinder(
        Vec3::new(side + 0.05 + 0.125, 2.0, 0.1),
        0.125,
        2.0,
    ));
    // 0.1 s: the car has stopped against the pole (its mean x at its
    // furthest) and the hit has done most of its damage. After that the
    // crash bifurcates: nine CPU runs whose nodes start 1e-6 m apart agree
    // to 4 mm in mean x at 0.1 s but spread 3 cm at 0.15 s and 4-10 cm at
    // 0.3 s (78-97 broken beams), since the restyled sedan of 8ae71777
    // (before the firewall and rail changes too). The GPU, not bit-exact,
    // lands on one of those branches; at 0.3 s it was 4 cm outside the nine.
    let Some(gpu) = run_both(&mut cpu, 6) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (c_broken, c_bent) = cpu.damage();
    let beams = &gpu.beams()[0];
    let g_broken = beams.iter().filter(|b| b.broken).count();
    let g_bent = beams.iter().filter(|b| b.bent()).count();
    let mean = |p: &[Vec3]| p.iter().copied().sum::<Vec3>() / p.len() as f32;
    let (cm, gm) = (mean(&cpu.positions()), mean(&gpu.positions()[0]));
    eprintln!(
        "(c) side pole 32 km/h, 0.1 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; mean x CPU {:.3} GPU {:.3} m",
        cm.x, gm.x
    );
    assert!(c_bent > 50);
    assert!(
        close(c_broken, g_broken, 0.2),
        "broken {c_broken} vs {g_broken}"
    );
    assert!(close(c_bent, g_bent, 0.15), "bent {c_bent} vs {g_bent}");
    assert!((cm.x - gm.x).abs() < 0.05);
}
