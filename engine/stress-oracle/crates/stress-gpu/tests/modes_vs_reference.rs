//! The GPU world against stress-ref's `World` in the adaptive and quasi-static solve
//! modes: catalogue and showcase scenes with their solve mode overridden, compared like
//! the catalogue (`world_vs_reference.rs`, same gate). Adaptive mode keeps settled
//! clusters at equilibrium until their loads change (an impact or blast wakes them on
//! the GPU); the overhang showcase creaks, cracks and snaps by static fatigue of its
//! settled cluster. `STRESS_GPU_SCENES=b8,...` restricts the cases.

use stress_gpu::gpu::Gpu;
use stress_gpu::world::{unsupported, GpuWorld};
use stress_ref::scene::{Scene, SolveMode};

mod common;
use common::{difference, gate, spread_difference};

fn scene(name: &str) -> Scene {
    stress_ref::builders::catalog()
        .into_iter()
        .chain(stress_ref::showcases::catalog())
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scene {name}"))
}

#[test]
fn solve_modes_match_the_reference_world() {
    let gpu = Gpu::new().expect("GPU");
    let filter: Option<Vec<String>> = std::env::var("STRESS_GPU_SCENES").ok().map(|v| v.split(',').map(str::to_string).collect());
    let cases: [(&str, SolveMode, f64); 8] = [
        ("b2_cantilever", SolveMode::QuasiStatic, 0.1),
        ("b8_frame_sudden", SolveMode::QuasiStatic, 0.3),
        ("s_floor_static", SolveMode::QuasiStatic, 0.25),
        ("b8_frame_sudden", SolveMode::Adaptive, 0.3),
        ("b9_panel_high", SolveMode::Adaptive, 0.06),
        ("b5_wall_impact_v10", SolveMode::Adaptive, 0.03),
        ("s_house_car", SolveMode::Adaptive, 0.1),
        ("s_overhang_thin", SolveMode::Adaptive, 12.0),
    ];
    let mut ran = 0;
    let mut failures = Vec::new();
    for (name, mode, duration) in cases {
        let label = format!("{name} ({mode:?})");
        if filter.as_ref().is_some_and(|f| !f.iter().any(|p| name.starts_with(p.as_str()))) {
            continue;
        }
        let mut s = scene(name);
        s.sim.solve_mode = mode;
        s.sim.duration = s.sim.duration.min(duration);
        if let Some(why) = unsupported(&s) {
            eprintln!("{label:<34} skipped: {why}");
            continue;
        }
        let t = std::time::Instant::now();
        let ours = GpuWorld::new(&gpu, &s).and_then(|mut w| w.run(&gpu)).expect("GPU world");
        let gpu_time = t.elapsed().as_secs_f64();
        // The reference's runs come from the cache (computed on a miss): see gate.rs.
        let refs = stress_gpu::gate::reference_set(&s, true).expect("reference set");
        let (reference, spread, half, ref_time) = (refs.reference, refs.spreads, refs.half, refs.reference_seconds);
        let (gpu_err, gpu_notes) = difference(&reference, &ours);
        let (self_err, _) = spread_difference(&reference, &spread);
        let (half_err, _) = difference(&reference, &half);
        ran += 1;
        eprintln!("{label:<34} gpu {gpu_err:.1e} vs reference's spread {self_err:.1e}, half step {half_err:.1e} | time gpu {gpu_time:.1} s, reference {ref_time:.1} s");
        eprintln!("{:<34}   gpu: {}", "", gpu_notes.join("; "));
        let substeps = |o: &stress_ref::observation::Observation| o.values.get("substeps").copied().unwrap_or(0.0);
        eprintln!("{:<34}   substeps: gpu {} reference {}", "", substeps(&ours), substeps(&reference));
        let fails = gate(&reference, &ours, &half, &spread);
        if !fails.is_empty() {
            eprintln!("{:<34}   GATE: {}", "", fails.join("; "));
            failures.push(format!("{label}: {}", fails.join("; ")));
        }
    }
    assert!(ran > 0);
    if std::env::var("STRESS_GPU_GATE").as_deref() != Ok("report") {
        assert!(failures.is_empty(), "accuracy gate:\n{}", failures.join("\n"));
    }
}
