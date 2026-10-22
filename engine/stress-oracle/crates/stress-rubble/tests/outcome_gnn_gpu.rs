//! P3: P2's trained model on the GPU (`stress_gpu::outcome_gnn`, the Slang kernels) computes
//! what the Rust CPU reference computes (`stress_learned::outcome_model`, itself checked
//! against PyTorch), on the parity fixture's real calls, with and without the node-projection
//! lever: the same break set, the outputs to f32 rounding. A correctness check, not a timing.
#![cfg(feature = "gpu")]

use std::fs;
use std::path::{Path, PathBuf};

use stress_gpu::gpu::Gpu;
use stress_gpu::outcome_gnn::{OutcomeKernels, OutcomeRun};
use stress_learned::outcome_model::{read_event, structure_hierarchies, Model, Options, Weights};
use stress_learned::outcome_program::compile;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../stress-learned/tests/fixtures/p3-model")
}

#[test]
fn the_gpu_model_matches_the_cpu_reference() {
    let gpu = Gpu::new().expect("GPU");
    let kernels = OutcomeKernels::new(&gpu);
    let w = Weights::load(&fixture()).unwrap();
    let tau = w.manifest["decode"]["tau"].as_f64().unwrap_or(1.0).ln() as f32;
    let mut dirs: Vec<PathBuf> = fs::read_dir(fixture().join("parity")).unwrap().map(|e| e.unwrap().path()).collect();
    dirs.sort();
    for (lever, tiled, fused) in [(false, false, false), (true, false, false), (true, true, false), (true, true, true), (false, false, true)] {
        let model = Model::new(&w, Options { node_projection: lever, tiled_mlp: tiled, fused_sums: fused }).unwrap();
        for dir in &dirs {
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            let ev = read_event(&dir.join("call")).unwrap();
            let hier = structure_hierarchies(&ev, model.config.levels);
            let want = model.forward(&ev, &hier, 0).unwrap();
            let prog = compile(&model, &ev, &hier, 0).unwrap();
            let ops: Vec<_> = prog.ops.iter().map(|o| (o.params, o.groups)).collect();
            let run = OutcomeRun::new(&gpu, &kernels, &prog.arena, &prog.weights, &prog.table, &ops);
            let arena = run.run(&gpu, &kernels);
            assert!(arena.iter().all(|x| x.is_finite()), "{name}: non-finite arena");
            let got = prog.outputs(&arena);
            let mut report = Vec::new();
            for ((k, _, a), (_, _, b)) in got.named().into_iter().zip(want.named()) {
                let e = a.iter().zip(&b).map(|(x, y)| (x - y).abs() / (y.abs() + 1.0)).fold(0.0f32, f32::max);
                report.push(format!("{k} {e:.1e}"));
                assert!(e < 1e-4, "{name} (lever {lever}): {k} differs by {e:.2e}; {}", report.join(", "));
            }
            let flips = got.bond_log_q.iter().zip(&want.bond_log_q).filter(|(a, b)| (**a >= tau) != (**b >= tau)).count();
            assert_eq!(flips, 0, "{name}: break set differs on {flips} bonds");
            eprintln!("{name} lever {lever}: {} dispatches; {}", run.dispatches(), report.join(", "));
        }
    }
}
