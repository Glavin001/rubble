//! Community BeamNG vehicles in the explicit solver's JBeam mode with pressure
//! wheels (`engine/docs/jbeam-semantics-spec.md` §29), GPU against the CPU
//! reference: 4 s at rest, then a rigid wall at 56 km/h for 1.5 s.
//!
//! Opt-in: `RUBBLE_BEAMNG_MODS=<inspiration/beamng-mods>` (the layout its
//! `SOURCES.md` records) and, for the legocar, `RUBBLE_JBEAM_DIR`. The content is
//! not in the repository; the test skips without it. Run one vehicle with
//! `RUBBLE_BEAMNG_ONLY=<name>`.
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, JbeamWheels, StaticBox,
};
use engine_softbody::jbeam::{load_jbeam, read_vehicle_dir};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn spawn(desc: &engine_softbody::desc::SoftVehicleDesc, v: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_with(desc, JbeamWheels::Pressure, Vec3::ZERO, v, settings)
            .unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y -= low - 0.02;
    }
    s.roll_tyres(v);
    s
}

struct Summary {
    broken: usize,
    bent: usize,
    ke: f32,
    vmax: f32,
    finite: bool,
}

fn summary(pos: &[Vec3], vel: &[Vec3], mass: &[f32], broken: usize, bent: usize) -> Summary {
    Summary {
        broken,
        bent,
        ke: vel.iter().zip(mass).map(|(v, m)| 0.5 * m * v.length_squared()).sum(),
        vmax: vel.iter().map(|v| v.length()).fold(0.0, f32::max),
        finite: pos.iter().all(|p| p.is_finite()),
    }
}

/// Both engines for `frames` 60 fps frames: (CPU, GPU, rms and max position gap).
fn run(cpu: &mut ExplicitBeams, frames: u32) -> Option<(Summary, Summary, f32, f32)> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(cpu));
    let mut clock = FrameClock::default();
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
    }
    gpu.wait();
    let mass: Vec<f32> = cpu.nodes.iter().map(|n| n.mass).collect();
    let (cb, cbent) = cpu.damage();
    let c = summary(
        &cpu.positions(),
        &cpu.nodes.iter().map(|n| n.vel).collect::<Vec<_>>(),
        &mass,
        cb,
        cbent,
    );
    let beams = &gpu.beams()[0];
    let gp = gpu.positions().remove(0);
    let gv = gpu.velocities().remove(0);
    let g = summary(
        &gp,
        &gv,
        &mass,
        beams.iter().filter(|b| b.broken).count(),
        beams.iter().filter(|b| b.bent()).count(),
    );
    let cp = cpu.positions();
    let d: Vec<f32> = cp.iter().zip(&gp).map(|(a, b)| a.distance(*b)).collect();
    let rms = (d.iter().map(|x| x * x).sum::<f32>() / d.len() as f32).sqrt();
    Some((c, g, rms, d.iter().cloned().fold(0.0, f32::max)))
}

fn vehicle(dir: &std::path::Path, name: &str, pc: &str) {
    if std::env::var("RUBBLE_BEAMNG_ONLY").is_ok_and(|o| o != name) {
        return;
    }
    let Ok(files) = read_vehicle_dir(dir, name) else {
        eprintln!("skip: {name} not in {}", dir.display());
        return;
    };
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
    let desc = load_jbeam(&refs, Some(&pcs)).unwrap();
    let line = |tag: &str, s: &Summary| {
        format!(
            "{tag} broken {} bent {} KE {:.1} J vmax {:.2} m/s",
            s.broken, s.bent, s.ke, s.vmax
        )
    };
    // At rest, 4 s.
    let mut cpu = spawn(&desc, Vec3::ZERO);
    let Some((c, g, rms, max)) = run(&mut cpu, 240) else {
        eprintln!("no GPU: skipped");
        return;
    };
    eprintln!(
        "{name} {pc} at rest 4 s: {}; {}; positions rms {:.2} mm max {:.2} mm",
        line("CPU", &c),
        line("GPU", &g),
        rms * 1e3,
        max * 1e3
    );
    assert!(c.finite && g.finite, "{name}: NaN at rest");
    assert_eq!((c.broken, g.broken), (0, 0), "{name}: breaks at rest");
    // Standing still: no node faster than 10 m/s, no energy beyond chatter.
    for s in [&c, &g] {
        assert!(s.vmax < 10.0 && s.ke < 500.0, "{name}: not at rest");
    }
    // 56 km/h into a rigid wall, 1.5 s.
    let v = Vec3::new(0.0, 0.0, 56.0 / 3.6);
    let mut cpu = spawn(&desc, v);
    let front = cpu.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.5, front + 0.3 + 1.0),
        Vec3::new(5.0, 3.0, 1.0),
    ));
    let mass: f32 = cpu.nodes.iter().map(|n| n.mass).sum();
    let ke0 = 0.5 * mass * v.length_squared();
    let (c, g, rms, max) = run(&mut cpu, 90).unwrap();
    eprintln!(
        "{name} {pc} 56 km/h wall 1.5 s (KE {:.0} J at impact): {}; {}; positions rms {:.1} mm max {:.1} mm",
        ke0,
        line("CPU", &c),
        line("GPU", &g),
        rms * 1e3,
        max * 1e3
    );
    assert!(c.finite && g.finite, "{name}: NaN in the crash");
    // Survives: the energy left is a small part of the impact's (no blow-up).
    for s in [&c, &g] {
        assert!(s.ke < 0.25 * ke0, "{name}: energy {} of {ke0}", s.ke);
    }
    // A crash is chaotic: GPU and CPU break counts within 25 % of each other.
    let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.25 * a.max(b) as f32 + 5.0;
    assert!(close(c.broken, g.broken), "{name}: broken CPU {} GPU {}", c.broken, g.broken);
}

#[test]
fn community_vehicles_rest_and_crash_on_the_gpu() {
    if let Ok(mods) = std::env::var("RUBBLE_BEAMNG_MODS") {
        let m = std::path::Path::new(&mods);
        vehicle(&m.join("monster_supercar/vehicles"), "supercar", "config.pc");
        vehicle(
            &m.join("qlourie_astral_r/vehicles"),
            "qlourie_astral_r",
            "wrc17_tarmac_balanced.pc",
        );
    } else {
        eprintln!("skip: RUBBLE_BEAMNG_MODS not set");
    }
    if let Ok(dir) = std::env::var("RUBBLE_JBEAM_DIR") {
        vehicle(std::path::Path::new(&dir), "agenty_legocar", "start.pc");
    }
}
