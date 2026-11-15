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
        ExplicitBeams::from_desc_jbeam_with(&d, JbeamWheels::Pressure, Vec3::ZERO, v, settings)
            .unwrap();
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
fn run_both(
    cpu: &mut ExplicitBeams,
    frames: u32,
    wide: Option<bool>,
) -> Option<MetalExplicitBeams> {
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
        let vmax = gpu.velocities()[0]
            .iter()
            .map(|v| v.length())
            .fold(0.0, f32::max);
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
        let d = cpu
            .positions()
            .iter()
            .zip(g)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0, f32::max);
        worst = worst.max(d);
    }
    let gb = gpu.beams()[0].iter().filter(|b| b.broken).count();
    let cb = cpu.damage().0;
    let gd = gpu.deform_triggered().remove(0);
    let cd = cpu.deform_triggered().to_vec();
    assert!(
        gpu.positions()[0].iter().all(|p| p.is_finite()),
        "{label}: Metal NaN"
    );
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
    let (zmin, zmax) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
        (a.min(n.pos.z), b.max(n.pos.z))
    });
    let length = zmax - zmin;
    let v = Vec3::new(0.0, 0.0, speed);
    let mut rear =
        ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings()).unwrap();
    rear.roll_tyres(v);
    let front = ExplicitBeams::from_truck(
        &text,
        Vec3::new(0.0, 0.01 - low, length + gap),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
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
                eprintln!(
                    "Metal frame {f} (contact starts): vehicle {k} GPU–CPU {:.3} mm",
                    d * 1e3
                );
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
        let close =
            |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.15 * a.max(b) as f32 + 3.0;
        assert!(
            close(c_broken, g_broken),
            "vehicle {k} broken {c_broken} vs {g_broken}"
        );
        assert!(
            close(c_bent, g_bent),
            "vehicle {k} bent {c_bent} vs {g_bent}"
        );
    }
}

/// JBeam Part II (jbeam-semantics-spec.md §16–§19) on Metal: break groups, breakable
/// triangles and deform groups (`jbeam_fixture::PART2_GROUPS`, `PART2_PANEL`), the
/// bars of engine-webgpu's `explicit_jbeam_ext_gpu`: the same breaks and deform
/// groups as the CPU, positions within 1 mm.
#[test]
fn metal_jbeam_part2_break_and_deform_groups_match() {
    use engine_softbody::jbeam_fixture::{PART2_GROUPS, PART2_PANEL};
    use engine_softbody::truck_export::TyreDefaults;
    for (label, text, pull) in [
        ("groups a", PART2_GROUPS, "a1"),
        ("groups b", PART2_GROUPS, "b1"),
        ("panel", PART2_PANEL, "p1"),
    ] {
        let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
        let settings = ExplicitBeamsSettings {
            turbulence: false,
            drag: false,
            ..Default::default()
        };
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
            &d,
            JbeamWheels::Soft(TyreDefaults::for_vehicle(&d)),
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        )
        .unwrap();
        let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        for n in &mut s.nodes {
            n.pos.y += 0.02 - low;
        }
        let i = d.nodes.iter().position(|n| n.name == pull).unwrap();
        s.nodes[i].vel = Vec3::new(40.0, 0.0, 0.0);
        let mut gpu = match metal(std::slice::from_ref(&s)) {
            Some(g) => g,
            None => return,
        };
        let mut clock = FrameClock::default();
        let mut worst = 0.0f32;
        for _ in 0..30 {
            s.frame();
            gpu.frame(clock.next());
            let g = &gpu.positions()[0];
            worst = worst.max(
                s.positions()
                    .iter()
                    .zip(g)
                    .map(|(a, b)| a.distance(*b))
                    .fold(0.0, f32::max),
            );
        }
        let gb = gpu.beams()[0].iter().filter(|b| b.broken).count();
        let cb = s.damage().0;
        let gd = gpu.deform_triggered().remove(0);
        let cd = s.deform_triggered().to_vec();
        eprintln!("Metal Part II {label}: max {:.4} mm; broken CPU {cb} GPU {gb}; deform CPU {cd:?} GPU {gd:?}", worst * 1e3);
        assert!(cb > 0, "{label}: something breaks");
        assert_eq!(cb, gb, "{label}");
        assert_eq!(cd, gd, "{label}");
        assert!(worst < 1e-3, "{label}: {worst}");
    }
}

/// The vehicle ↔ rigid proxy coupling (`explicit_beams_rigid.slang`) on Metal: the repo
/// sedan flying (no ground) into a free 1200 kg box, against the CPU reference
/// (`engine_softbody::explicit_rigid`), with engine-webgpu's `explicit_rigid_gpu` bars:
/// the first frames of contact within 1 mm, horizontal momentum closed to 2e-4, box and
/// vehicle speeds within 1 %.
#[test]
fn metal_coupled_sedan_into_a_rigid_box_matches_cpu() {
    use engine_exec::explicit_plan::CouplingCapacity;
    use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
    use glam::Quat;
    const STEPS: u32 = 17;
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
        centred_damping: false,
        centred_objective: false,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let bounds = |s: &ExplicitBeams| {
        s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        })
    };
    let (lo, hi) = bounds(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = bounds(&sim);
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 1.0 - lo.y, -hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, 15.0);
    }
    sim.ground = -1.0e4;
    let hi_y = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let (centre, half, mass) = (
        Vec3::new(0.0, 0.5 * (1.0 + hi_y), 1.3),
        Vec3::new(1.2, 0.7, 0.5),
        1200.0f32,
    );
    let inertia = Vec3::new(
        mass / 3.0 * (half.y * half.y + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.y * half.y),
    );
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 7,
            pos: centre,
            rot: Quat::from_rotation_y(0.2),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::ONE / inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            ProxyGeom::Box { half },
            3,
            None,
        )],
        frozen: false,
    };
    set.begin_window();
    let m: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    let cap = CouplingCapacity {
        bodies: 4,
        shapes: 8,
        planes: 64,
    };
    let mut gpu = match MetalExplicitBeams::with_coupling(std::slice::from_ref(&sim), Some(cap)) {
        Ok(g) => g,
        Err(e) if e.to_string().contains("no Metal device") => return,
        Err(e) => panic!("{e}"),
    };
    gpu.write_proxies(0, &set).unwrap();
    gpu.set_coupling_on(true);
    assert!(!gpu.is_wide());
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let p0 = horizontal(sim.nodes.iter().map(|n| n.vel * n.mass).sum());
    let mut first_contact = None;
    let mut worst_early = 0.0f32;
    for frame in 0..48 {
        gpu.frame(STEPS);
        for _ in 0..STEPS {
            cpu.step();
        }
        if first_contact.is_none() && cpu.rigid.as_ref().unwrap().shapes[0].impulse.length() > 0.0 {
            first_contact = Some(frame);
        }
        if first_contact.is_some_and(|f| frame <= f + 1) {
            let pos = gpu.positions();
            for (a, n) in pos[0].iter().zip(&cpu.nodes) {
                worst_early = worst_early.max((*a - (cpu.origin + n.pos)).length());
            }
        }
    }
    assert!(first_contact.is_some(), "the box was hit");
    let mut gset = set.clone();
    gpu.read_proxies(0, &mut gset);
    let cset = cpu.rigid.take().unwrap();
    let gvel = gpu.velocities();
    let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
    let pg = horizontal(
        gvel[0]
            .iter()
            .zip(&masses)
            .map(|(v, m)| *v * *m)
            .sum::<Vec3>(),
    ) + horizontal(gset.bodies[0].momentum().0);
    let vg = gvel[0]
        .iter()
        .zip(&masses)
        .map(|(v, m)| v.z * m)
        .sum::<f32>()
        / m;
    let vc = cpu.nodes.iter().map(|n| n.vel.z * n.mass).sum::<f32>() / m;
    eprintln!(
        "Metal coupled: early worst {worst_early:.2e} m; box GPU {:.4} CPU {:.4} m/s; vehicle GPU {vg:.4} CPU {vc:.4}; |dP| {:.4} of {:.0}",
        gset.bodies[0].vel.z,
        cset.bodies[0].vel.z,
        (pg - p0).length(),
        p0.length()
    );
    assert!(worst_early < 1e-3, "early frames apart by {worst_early} m");
    assert!(
        (pg - p0).length() < 2e-4 * p0.length(),
        "Metal momentum {pg} vs {p0}"
    );
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-3);
    assert!(rel(gset.bodies[0].vel.z, cset.bodies[0].vel.z) < 0.01);
    assert!(rel(vg, vc) < 0.01);
}

/// `rubble_exact_lengths` (jbeam-semantics-spec.md §33) on Metal: the procedural
/// sedan at rest, exact lengths and fast, every node's position and velocity the
/// CPU's float after each of 2,400 steps (1.2 s), as on wgpu
/// (`showcase_car_gpu.rs` `exact_lengths_step_bit_for_bit`).
#[test]
fn metal_exact_lengths_step_bit_for_bit() {
    for fast in [false, true] {
        let s = ExplicitBeamsSettings {
            fast_inv_sqrt: fast,
            ..settings()
        };
        let text = sedan_text();
        let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, s).unwrap();
        let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        let mut cpu =
            ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), Vec3::ZERO, s)
                .unwrap();
        let Some(mut gpu) = metal(std::slice::from_ref(&cpu)) else {
            return;
        };
        for step in 0..2400 {
            cpu.step();
            gpu.frame(1);
            gpu.wait();
            let (p, v) = (gpu.positions(), gpu.velocities());
            let bits = |a: Vec3| a.to_array().map(f32::to_bits);
            let bad = (0..cpu.nodes.len())
                .filter(|&i| {
                    bits(cpu.nodes[i].pos) != bits(p[0][i])
                        || bits(cpu.nodes[i].vel) != bits(v[0][i])
                })
                .count();
            assert_eq!(
                bad, 0,
                "fast {fast}: step {step}: {bad} nodes differ from the CPU"
            );
        }
    }
}
