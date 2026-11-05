//! The explicit node–beam solver on the native Metal host (`MetalExplicitBeams`:
//! the shared plan, `engine_exec::explicit_plan`, replayed with Slang → MSL kernels)
//! against the CPU reference (`engine_softbody::explicit_beams`), with the same
//! fixtures and bars as engine-webgpu's `explicit_beams_gpu`, `explicit_jbeam_ext_gpu`
//! and `explicit_contact_gpu`: the procedural sedan at rest and into a wall at
//! 56 km/h (per-vehicle frame and wide passes), the JBeam-mode pressure-wheel car
//! (`jbeam_fixture::PART2_CAR`) at rest and into a wall, and two sedans in a rear-end
//! crash (the vehicle-contact plan and its indirect dispatches). Turbulence off on
//! both (the GPU draws its noise differently).

use engine_metal::MetalExplicitBeams;
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, JbeamWheels, StaticBox,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::PART2_CAR;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

fn sedan_text() -> String {
    car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    })
}

fn sedan(velocity: Vec3) -> ExplicitBeams {
    let text = sedan_text();
    // Lowest node 1 cm above the ground.
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, velocity, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), velocity, settings()).unwrap()
}

fn jbeam_car(v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", PART2_CAR)], None).unwrap();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_with(&d, JbeamWheels::Pressure, Vec3::ZERO, v, settings).unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(v);
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

fn metal(fleet: &[ExplicitBeams]) -> Option<MetalExplicitBeams> {
    match MetalExplicitBeams::new(fleet) {
        Ok(m) => Some(m),
        Err(e) if e.to_string().contains("no Metal device") => {
            eprintln!("no Metal device: skipped");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// Run both for `frames` 60 fps frames; returns the Metal host.
fn run_both(cpu: &mut ExplicitBeams, frames: u32, wide: Option<bool>) -> Option<MetalExplicitBeams> {
    let mut gpu = metal(std::slice::from_ref(cpu))?;
    if let Some(w) = wide {
        gpu.set_wide(w);
    }
    let mut clock = FrameClock::default();
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
    }
    gpu.wait();
    Some(gpu)
}

fn close(a: usize, b: usize) -> bool {
    (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 1.0
}

#[test]
fn metal_sedan_at_rest_matches_cpu() {
    for wide in [false, true] {
        let mut cpu = sedan(Vec3::ZERO);
        let Some(mut gpu) = run_both(&mut cpu, 60, Some(wide)) else {
            return;
        };
        assert_eq!(gpu.is_wide(), wide);
        let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
        eprintln!(
            "Metal ({}) at rest after 1 s: GPU vs CPU rms {:.3} mm, max {:.3} mm",
            if wide { "wide" } else { "frame" },
            rms * 1e3,
            max * 1e3
        );
        assert!(rms < 3e-3, "rms {rms} m");
        assert!(max < 1e-2, "max {max} m");
        assert_eq!(cpu.damage().0, 0);
        let vmax = gpu.velocities()[0].iter().map(|v| v.length()).fold(0.0, f32::max);
        assert!(vmax < 0.5, "GPU node speed {vmax} m/s at rest");
    }
}

#[test]
fn metal_sedan_crash_matches_cpu_counts() {
    for wide in [false, true] {
        let speed = 56.0 / 3.6;
        let mut cpu = sedan(Vec3::new(0.0, 0.0, speed));
        let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        // A rigid block 5 cm in front of the car, wider and taller than it.
        cpu.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        let Some(mut gpu) = run_both(&mut cpu, 36, Some(wide)) else {
            return;
        };
        let (c_broken, c_bent) = cpu.damage();
        let beams = &gpu.beams()[0];
        let g_broken = beams.iter().filter(|b| b.broken).count();
        let g_bent = beams.iter().filter(|b| b.bent()).count();
        let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
        eprintln!(
            "Metal ({}) crash 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
             positions rms {:.1} mm max {:.1} mm",
            if wide { "wide" } else { "frame" },
            rms * 1e3,
            max * 1e3
        );
        assert!(c_bent > 100, "the crash bends the front ({c_bent})");
        assert!(close(c_broken, g_broken), "broken {c_broken} vs {g_broken}");
        assert!(close(c_bent, g_bent), "bent {c_bent} vs {g_bent}");
    }
}

/// Both for `frames` frames; (max position difference over the run, CPU broken,
/// GPU broken).
fn compare_jbeam(label: &str, cpu: &mut ExplicitBeams, frames: u32) -> Option<(f32, usize, usize)> {
    let mut gpu = metal(std::slice::from_ref(cpu))?;
    let mut clock = FrameClock::default();
    let mut worst = 0.0f32;
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
        let g = &gpu.positions()[0];
        let d = cpu.positions().iter().zip(g).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max);
        worst = worst.max(d);
    }
    let gb = gpu.beams()[0].iter().filter(|b| b.broken).count();
    let cb = cpu.damage().0;
    let gd = gpu.deform_triggered().remove(0);
    let cd = cpu.deform_triggered().to_vec();
    assert!(gpu.positions()[0].iter().all(|p| p.is_finite()), "{label}: Metal NaN");
    eprintln!(
        "Metal {label}: max position difference {:.4} mm; broken CPU {cb} GPU {gb}; deform CPU {cd:?} GPU {gd:?}",
        worst * 1e3
    );
    Some((worst, cb, gb))
}

#[test]
fn metal_jbeam_pressure_car_matches_at_rest_and_into_a_wall() {
    let mut s = jbeam_car(Vec3::ZERO);
    let Some((worst, cb, gb)) = compare_jbeam("JBeam pressure car at rest", &mut s, 120) else {
        return;
    };
    assert_eq!((cb, gb), (0, 0));
    assert!(worst < 5e-3, "at rest: {worst}");
    let v = Vec3::new(0.0, 0.0, 56.0 / 3.6);
    let mut s = jbeam_car(v);
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.3 + 1.0),
        Vec3::new(4.0, 3.0, 1.0),
    ));
    let (worst, cb, gb) = compare_jbeam("JBeam pressure car wall", &mut s, 60).unwrap();
    assert!(close(cb, gb), "broken {cb} vs {gb}");
    eprintln!("wall run: max difference {worst} m (a crash is chaotic: reported, not judged)");
}

/// Two sedans in one lane, the rear one at `speed` m/s, `gap` m apart.
fn rear_end(speed: f32, gap: f32) -> Vec<ExplicitBeams> {
    let text = sedan_text();
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe
        .nodes
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)));
    let length = zmax - zmin;
    let v = Vec3::new(0.0, 0.0, speed);
    let mut rear = ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings()).unwrap();
    rear.roll_tyres(v);
    let front =
        ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, length + gap), Vec3::ZERO, settings()).unwrap();
    vec![front, rear]
}

#[test]
fn metal_two_sedans_rear_end_match_cpu() {
    let fleet = rear_end(15.0, 0.5);
    let Some(mut g) = metal(&fleet) else {
        return;
    };
    assert!(g.contact_active());
    let mut w = ExplicitBeamsWorld::new(fleet);
    let mut first = None;
    for f in 0..36 {
        w.frame(17);
        g.frame(17);
        if w.first_hit_step.is_some() && first.is_none() {
            first = Some(f);
            let p = g.positions();
            for k in 0..2 {
                let (d, _) = rms_max(&p[k], &w.vehicles[k].positions());
                eprintln!("Metal frame {f} (contact starts): vehicle {k} GPU–CPU {:.3} mm", d * 1e3);
                assert!(d < 5e-4, "vehicle {k}: {d} m at contact");
            }
        }
    }
    g.contact_status().unwrap();
    assert!(first.is_some(), "they met");
    let beams = g.beams();
    let p = g.positions();
    for k in 0..2 {
        let (c_broken, c_bent) = w.vehicles[k].damage();
        let g_broken = beams[k].iter().filter(|b| b.broken).count();
        let g_bent = beams[k].iter().filter(|b| b.bent()).count();
        let (d, _) = rms_max(&p[k], &w.vehicles[k].positions());
        eprintln!(
            "Metal vehicle {k} after 0.6 s: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} \
             bent {g_bent}; positions rms {:.1} mm",
            d * 1e3
        );
        assert!(c_bent > 20, "the crash bends vehicle {k} ({c_bent})");
        // One GPU run (the wgpu test takes the median of 5): the wider bar.
        let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.15 * a.max(b) as f32 + 3.0;
        assert!(close(c_broken, g_broken), "vehicle {k} broken {c_broken} vs {g_broken}");
        assert!(close(c_bent, g_bent), "vehicle {k} bent {c_bent} vs {g_bent}");
    }
}
