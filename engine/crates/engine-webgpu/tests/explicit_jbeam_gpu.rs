//! The explicit solver's JBeam mode on the GPU against the CPU reference
//! (`engine/docs/jbeam-semantics-spec.md`): Rubble's latched-box fixture at rest and
//! in a crash, and, opt-in, BeamNG's BX coupé body (`RUBBLE_JBEAM_BX=<vehicles dir
//! with bx/ and common/>`, config `rubble_body_only.pc`) and the legocar
//! (`RUBBLE_JBEAM_DIR=<vehicles dir with agenty_legocar/>`, `start.pc`). BeamNG
//! content is not in the repository; those tests skip without it.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, StaticBox,
};
use engine_softbody::jbeam::{load_jbeam, read_vehicle_dir};
use engine_softbody::jbeam_fixture::latched_box_jbeam;
use engine_softbody::truck_export::TyreDefaults;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

/// A JBeam vehicle with its lowest node `lift` above the ground, moving at `v`.
fn spawn(files: &[(&str, &str)], pc: Option<&str>, lift: f32, v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(files, pc).unwrap();
    let tyres = TyreDefaults::for_vehicle(&d);
    let (mut s, _) = ExplicitBeams::from_desc_jbeam(&d, tyres, Vec3::ZERO, v, settings()).unwrap();
    // Every node (tyre nodes too) above the ground.
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += lift - low;
    }
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

fn wall(cpu: &mut ExplicitBeams) {
    let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(4.0, 3.0, 1.0),
    ));
}

/// Compare the two after a run: positions, and broken and bent counts.
fn report(label: &str, cpu: &ExplicitBeams, gpu: &ExplicitBeamsGpu) -> (f32, f32, [usize; 4]) {
    let (c_broken, c_bent) = cpu.damage();
    let beams = &gpu.beams()[0];
    let g_broken = beams.iter().filter(|b| b.broken).count();
    let g_bent = beams.iter().filter(|b| b.bent()).count();
    let (rms, max) = rms_max(&cpu.positions(), &gpu.positions()[0]);
    eprintln!(
        "{label}: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}; \
         positions rms {:.3} mm max {:.3} mm",
        rms * 1e3,
        max * 1e3
    );
    assert!(
        gpu.positions()[0].iter().all(|p| p.is_finite()),
        "{label}: GPU NaN"
    );
    (rms, max, [c_broken, c_bent, g_broken, g_bent])
}

fn close(a: usize, b: usize) -> bool {
    (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 1.0
}

#[test]
fn latched_box_at_rest_matches_cpu() {
    let text = latched_box_jbeam();
    let mut cpu = spawn(&[("box.jbeam", &text)], None, 0.01, Vec3::ZERO);
    let Some(gpu) = run_both(&mut cpu, 120) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, max, [cb, cbent, gb, gbent]) = report("latched box at rest 2 s", &cpu, &gpu);
    assert_eq!((cb, cbent, gb, gbent), (0, 0, 0, 0));
    assert!(rms < 1e-3 && max < 3e-3, "rms {rms} max {max}");
}

#[test]
fn latched_box_crash_matches_cpu() {
    let text = latched_box_jbeam();
    let mut cpu = spawn(
        &[("box.jbeam", &text)],
        None,
        0.01,
        Vec3::new(0.0, 0.0, 20.0),
    );
    wall(&mut cpu);
    let Some(gpu) = run_both(&mut cpu, 30) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (_, _, [cb, cbent, gb, gbent]) =
        report("latched box 20 m/s into a wall, 0.5 s", &cpu, &gpu);
    assert!(cbent > 0, "the crash deforms the box");
    assert!(close(cb, gb) && close(cbent, gbent));
}

fn beamng(env: &str, name: &str, pc: &str) -> Option<(Vec<(String, String)>, String)> {
    let Ok(dir) = std::env::var(env) else {
        eprintln!("skip: {env} not set");
        return None;
    };
    let dir = std::path::Path::new(&dir);
    let files = read_vehicle_dir(dir, name).ok()?;
    let pc = std::fs::read_to_string(dir.join(name).join(pc)).ok()?;
    Some((files, pc))
}

fn real(env: &str, name: &str, pc: &str) {
    let Some((files, pcs)) = beamng(env, name, pc) else {
        return;
    };
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    // The rules: without self-collision (whose GPU forces are fixed point) the two
    // agree to well under a micrometre for the first frames (about 100 steps; the
    // ground contact's and the torsion angle's last bits differ, and later the
    // parked car's chatter takes them apart).
    let mut cpu = spawn(&refs, Some(&pcs), 0.02, Vec3::ZERO);
    cpu.candidates.clear();
    let Some(gpu) = run_both(&mut cpu, 3) else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (rms, _, _) = report(
        &format!("{name} {pc} at rest, no self-collision, 3 frames"),
        &cpu,
        &gpu,
    );
    assert!(rms < 5e-6, "rms {rms} m");
    let mut cpu = spawn(&refs, Some(&pcs), 0.02, Vec3::ZERO);
    let Some(gpu) = run_both(&mut cpu, 120) else {
        return;
    };
    let (_, _, [cb, _, gb, _]) = report(&format!("{name} {pc} at rest 2 s"), &cpu, &gpu);
    assert!(close(cb, gb));
    // A crash is chaotic: the GPU's counts against the spread of CPU runs spawned
    // 0.1 mm/s × k faster (k = 0..4), widened 10 %.
    let crash = |k: u32| {
        let v = 100.0 / 3.6 + 1e-4 * k as f32;
        let mut cpu = spawn(&refs, Some(&pcs), 0.02, Vec3::new(0.0, 0.0, v));
        wall(&mut cpu);
        cpu
    };
    let mut cpu = crash(0);
    let Some(gpu) = run_both(&mut cpu, 30) else {
        return;
    };
    let (_, _, [cb, cbent, gb, gbent]) = report(
        &format!("{name} {pc} 100 km/h into a wall, 0.5 s"),
        &cpu,
        &gpu,
    );
    let (mut broken, mut bent) = (vec![cb], vec![cbent]);
    for k in 1..5 {
        let mut c = crash(k);
        for _ in 0..30 {
            c.frame();
        }
        let (b, t) = c.damage();
        broken.push(b);
        bent.push(t);
    }
    let within = |x: usize, v: &[usize]| {
        let (lo, hi) = (
            *v.iter().min().unwrap() as f32,
            *v.iter().max().unwrap() as f32,
        );
        let pad = 0.1 * hi + 1.0;
        (x as f32) >= lo - pad && (x as f32) <= hi + pad
    };
    eprintln!("  CPU ensemble broken {broken:?} bent {bent:?}");
    assert!(
        within(gb, &broken),
        "GPU broken {gb} outside the CPU spread {broken:?}"
    );
    assert!(
        within(gbent, &bent),
        "GPU bent {gbent} outside the CPU spread {bent:?}"
    );
}

#[test]
fn bx_body_matches_cpu() {
    real("RUBBLE_JBEAM_BX", "bx", "rubble_body_only.pc");
}

#[test]
fn legocar_matches_cpu() {
    real("RUBBLE_JBEAM_DIR", "agenty_legocar", "start.pc");
}

/// Diagnostic (`RUBBLE_JBEAM_TRACE=bx|legocar`): GPU − CPU per frame.
#[test]
fn divergence_trace() {
    let Ok(which) = std::env::var("RUBBLE_JBEAM_TRACE") else {
        return;
    };
    let (env, name, pc) = if which == "bx" {
        ("RUBBLE_JBEAM_BX", "bx", "rubble_body_only.pc")
    } else {
        ("RUBBLE_JBEAM_DIR", "agenty_legocar", "start.pc")
    };
    let Some((files, pcs)) = beamng(env, name, pc) else {
        return;
    };
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let mut cpu = spawn(&refs, Some(&pcs), 0.02, Vec3::ZERO);
    if std::env::var("RUBBLE_JBEAM_NOSELF").is_ok() {
        cpu.candidates.clear();
    }
    let (_, device, queue) = default_device().unwrap();
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    for f in 0..40 {
        for _ in 0..8 {
            cpu.step();
        }
        gpu.frame(8);
        gpu.wait();
        let g = gpu.positions();
        let c = cpu.positions();
        let (rms, max) = rms_max(&c, &g[0]);
        let worst = (0..c.len())
            .max_by(|&a, &b| c[a].distance(g[0][a]).total_cmp(&c[b].distance(g[0][b])))
            .unwrap();
        eprintln!(
            "frame {f}: rms {rms:.3e} max {max:.3e} at node {worst} (contacts {})",
            cpu.last_contacts
        );
    }
}
