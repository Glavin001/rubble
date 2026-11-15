//! The explicit node–beam solver on CUDA (`CudaExplicitBeams`: the shared plan,
//! `engine_exec::explicit_plan`, replayed with Slang → CUDA kernels) against the CPU
//! reference, with the bars of engine-webgpu's `explicit_beams_gpu`,
//! `explicit_jbeam_ext_gpu` and `explicit_contact_gpu` and engine-metal's
//! `explicit_parity`: the procedural sedan at rest and into a wall at 56 km/h (frame
//! and wide passes), the JBeam-mode pressure-wheel car (`jbeam_fixture::PART2_CAR`)
//! at rest and into a wall, two sedans in a rear-end crash (the contact plan, its
//! indirect arguments read back).
//!
//! Every build checks that each plan module's CUDA source was emitted with every
//! binding mapped. GPU tests run when `ENGINE_CUDA_DEVICE=1` (an NVIDIA host with
//! nvcc at build time) and fail if no device loads; without it they skip, saying so.

use engine_cuda::{CudaExplicitBeams, CUDA_EXPLICIT_KERNELS};
use engine_exec::explicit_plan::{Kernel, BIND_COUNT, MODULES};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, JbeamWheels, StaticBox,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::PART2_CAR;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host (built with nvcc)");
        return false;
    }
    assert!(
        CudaExplicitBeams::has_ptx(),
        "ENGINE_CUDA_DEVICE=1 but this build embedded no explicit-beams PTX (no nvcc)"
    );
    assert!(CudaExplicitBeams::available(), "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded");
    true
}

#[test]
fn cuda_explicit_sources_emitted() {
    // Every module's source, also in a build that embeds device code for only some
    // (ENGINE_CUDA_EXPLICIT_MODULES).
    for m in MODULES {
        let k = CUDA_EXPLICIT_KERNELS
            .iter()
            .find(|k| k.name == m.name)
            .unwrap_or_else(|| panic!("{}: no CUDA module", m.name));
        assert_eq!(k.entry, m.entry);
        assert!(k.cu_bytes > 0, "{}: empty CUDA source", m.name);
        assert!(k.threads[0] > 0, "{}: no thread group size", m.name);
        assert!(k.binds.iter().all(|&(_, _, slot)| (slot as usize) < BIND_COUNT), "{}", m.name);
        // Every module reads the parameters and the node state.
        for slot in [0, 3] {
            assert!(k.binds.iter().any(|b| b.2 == slot), "{}: binding {slot} not mapped", m.name);
        }
    }
    // The frame modules run one workgroup per vehicle at their lane count.
    for (aero, lanes) in [(false, 256), (false, 512), (true, 256), (true, 512)] {
        for kernel in [Kernel::Frame, Kernel::Solo, Kernel::Contact] {
            let name = kernel.module(aero, lanes);
            let k = CUDA_EXPLICIT_KERNELS.iter().find(|k| k.name == name).unwrap();
            assert_eq!(k.threads[0], lanes, "{name}");
        }
    }
    eprintln!(
        "{} explicit CUDA modules emitted; device code embedded: {}",
        CUDA_EXPLICIT_KERNELS.len(),
        CudaExplicitBeams::has_ptx()
    );
}

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

fn sedan_crash() -> ExplicitBeams {
    let mut cpu = sedan(Vec3::new(0.0, 0.0, 56.0 / 3.6));
    let front = cpu.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    cpu.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
        Vec3::new(3.0, 2.0, 1.0),
    ));
    cpu
}

fn jbeam_crash() -> ExplicitBeams {
    let mut cpu = jbeam_car(Vec3::new(0.0, 0.0, 56.0 / 3.6));
    let front = cpu.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    cpu.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.0, front + 0.3 + 1.0), Vec3::new(4.0, 3.0, 1.0)));
    cpu
}

/// Two sedans, the rear one at 15 m/s half a metre behind the other.
fn two_sedans() -> Vec<ExplicitBeams> {
    let text = sedan_text();
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let (zmin, zmax) = probe
        .nodes
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), n| (a.min(n.pos.z), b.max(n.pos.z)));
    let v = Vec3::new(0.0, 0.0, 15.0);
    let mut rear = ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), v, settings()).unwrap();
    rear.roll_tyres(v);
    let front = ExplicitBeams::from_truck(
        &text,
        Vec3::new(0.0, 0.01 - low, zmax - zmin + 0.5),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    vec![front, rear]
}

/// Every fleet this file's GPU tests upload.
fn gpu_test_fleets() -> Vec<Vec<ExplicitBeams>> {
    vec![
        vec![sedan(Vec3::ZERO)],
        vec![sedan_crash()],
        vec![jbeam_car(Vec3::ZERO)],
        vec![jbeam_crash()],
        two_sedans(),
    ]
}

/// The explicit modules CI embeds device code for (`ENGINE_CUDA_EXPLICIT_MODULES` in
/// .github/workflows/engine-cuda.yml, read from `ci_explicit_modules.txt`) are exactly
/// those this file's GPU tests load, derived from their fleets the way
/// `CudaExplicitBeams::new` picks them. Fails on the CPU, before any GPU run, when a test
/// needs a module CI would leave out (or the list names one no test loads).
#[test]
fn ci_explicit_modules_are_those_the_gpu_tests_load() {
    let need: std::collections::BTreeSet<&str> = gpu_test_fleets()
        .iter()
        .flat_map(|f| CudaExplicitBeams::modules_for(f, None))
        .collect();
    let rewrite = "ENGINE_CUDA_WRITE_CI_MODULES=1 cargo test -p engine-cuda --release --test explicit_parity ci_explicit";
    if std::env::var("ENGINE_CUDA_WRITE_CI_MODULES").as_deref() == Ok("1") {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/ci_explicit_modules.txt");
        let mut text = String::from(
            "# Explicit modules CI compiles to device code: those tests/explicit_parity.rs's GPU\n\
             # tests load. Generated; rewrite with\n",
        );
        text.push_str(&format!("# {rewrite}\n"));
        for n in &need {
            text.push_str(n);
            text.push('\n');
        }
        std::fs::write(path, text).unwrap();
        eprintln!("wrote {path}");
        return;
    }
    let listed: std::collections::BTreeSet<&str> = include_str!("../ci_explicit_modules.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let missing: Vec<_> = need.difference(&listed).collect();
    let unused: Vec<_> = listed.difference(&need).collect();
    assert!(
        missing.is_empty() && unused.is_empty(),
        "ci_explicit_modules.txt is stale: the GPU tests need {missing:?} it leaves out, and it lists \
         {unused:?} they do not load. Rewrite it with `{rewrite}`"
    );
    for name in &listed {
        assert!(MODULES.iter().any(|m| m.name == *name), "ci_explicit_modules.txt: {name} is not a plan module");
    }
    // This build embeds them (a build filtered to other modules fails here, not on the GPU).
    if CudaExplicitBeams::has_ptx() {
        let absent: Vec<_> = need
            .iter()
            .filter(|n| CUDA_EXPLICIT_KERNELS.iter().any(|k| k.name == **n && k.image.is_empty()))
            .collect();
        assert!(absent.is_empty(), "this build has no device code for {absent:?}, which the GPU tests load");
    }
    eprintln!("{} explicit modules for the GPU tests: {}", need.len(), need.iter().copied().collect::<Vec<_>>().join(","));
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

fn close(a: usize, b: usize) -> bool {
    (a as f32 - b as f32).abs() <= 0.05 * a.max(b) as f32 + 1.0
}

fn run_both(cpu: &mut ExplicitBeams, frames: u32, wide: bool) -> CudaExplicitBeams {
    let mut g = CudaExplicitBeams::new(std::slice::from_ref(cpu)).unwrap();
    g.set_wide(wide);
    let mut clock = FrameClock::default();
    for _ in 0..frames {
        cpu.frame();
        g.frame(clock.next()).unwrap();
    }
    g.wait().unwrap();
    g
}

#[test]
fn cuda_sedan_at_rest_and_crash_match_cpu() {
    if !gpu() {
        return;
    }
    for wide in [false, true] {
        let mut cpu = sedan(Vec3::ZERO);
        let mut g = run_both(&mut cpu, 60, wide);
        let (rms, max) = rms_max(&cpu.positions(), &g.positions().unwrap()[0]);
        eprintln!("CUDA (wide {wide}) at rest 1 s: rms {:.3} mm max {:.3} mm", rms * 1e3, max * 1e3);
        assert!(rms < 3e-3 && max < 1e-2, "rms {rms} max {max}");

        let mut cpu = sedan_crash();
        let mut g = run_both(&mut cpu, 36, wide);
        let (c_broken, c_bent) = cpu.damage();
        let beams = &g.beams().unwrap()[0];
        let g_broken = beams.iter().filter(|b| b.broken).count();
        let g_bent = beams.iter().filter(|b| b.bent()).count();
        eprintln!("CUDA (wide {wide}) crash: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}");
        assert!(close(c_broken, g_broken) && close(c_bent, g_bent));
    }
}

#[test]
fn cuda_jbeam_pressure_car_matches_cpu() {
    if !gpu() {
        return;
    }
    let mut cpu = jbeam_car(Vec3::ZERO);
    let mut g = run_both(&mut cpu, 120, false);
    let (_, max) = rms_max(&cpu.positions(), &g.positions().unwrap()[0]);
    let gb = g.beams().unwrap()[0].iter().filter(|b| b.broken).count();
    eprintln!("CUDA JBeam car at rest 2 s: max {:.3} mm, broken {gb}", max * 1e3);
    assert_eq!((cpu.damage().0, gb), (0, 0));
    assert!(max < 5e-3, "{max}");
    let mut cpu = jbeam_crash();
    let mut g = run_both(&mut cpu, 60, false);
    let gb = g.beams().unwrap()[0].iter().filter(|b| b.broken).count();
    let cb = cpu.damage().0;
    eprintln!("CUDA JBeam car wall: broken CPU {cb} GPU {gb}");
    assert!(close(cb, gb));
}

#[test]
fn cuda_two_sedans_rear_end_match_cpu() {
    if !gpu() {
        return;
    }
    let fleet = two_sedans();
    let mut g = CudaExplicitBeams::new(&fleet).unwrap();
    assert!(g.contact_active());
    let mut w = ExplicitBeamsWorld::new(fleet);
    for _ in 0..36 {
        w.frame(17);
        g.frame(17).unwrap();
    }
    g.wait().unwrap();
    let beams = g.beams().unwrap();
    for k in 0..2 {
        let (c_broken, c_bent) = w.vehicles[k].damage();
        let g_broken = beams[k].iter().filter(|b| b.broken).count();
        let g_bent = beams[k].iter().filter(|b| b.bent()).count();
        eprintln!("CUDA vehicle {k}: CPU broken {c_broken} bent {c_bent}; GPU broken {g_broken} bent {g_bent}");
        let close = |a: usize, b: usize| (a as f32 - b as f32).abs() <= 0.15 * a.max(b) as f32 + 3.0;
        assert!(close(c_broken, g_broken) && close(c_bent, g_bent), "vehicle {k}");
    }
    eprintln!("CUDA indirect-argument reads: {}", g.args_syncs);
}
