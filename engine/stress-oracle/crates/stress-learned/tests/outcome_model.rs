//! P3 parity: the Rust CPU reference of P2's model (`outcome_model`) against PyTorch on fixed
//! weights. The fixture (`tests/fixtures/p3-model`, from P2's `outcome.export` through
//! `learned-impact/scripts/p3_fixture.py`) holds the trained checkpoint's weights and real
//! validation calls with PyTorch's CPU outputs.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use stress_learned::outcome_model::{read_event, structure_hierarchies, Model, Options, Weights};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/p3-model")
}

fn parity_dirs() -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(fixture().join("parity")).unwrap().map(|e| e.unwrap().path()).collect();
    d.sort();
    d
}

fn torch_outputs(dir: &Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let idx: Value = serde_json::from_slice(&fs::read(dir.join("outputs.json")).unwrap()).unwrap();
    let blob = fs::read(dir.join("outputs.bin")).unwrap();
    idx.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let n: usize = shape.iter().product();
            let off = v["offset"].as_u64().unwrap() as usize;
            let d = blob[off..off + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            (k.clone(), (shape, d))
        })
        .collect()
}

/// Largest |a − b| / (|b| + 1) over an array (f32 noise on O(1) values ≈ 1e-6).
fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / (y.abs() + 1.0)).fold(0.0, f32::max)
}

fn check(options: Options, tol: f32) {
    let w = Weights::load(&fixture()).unwrap();
    let model = Model::new(&w, options).unwrap();
    let dirs = parity_dirs();
    assert!(dirs.len() >= 4, "parity calls: {}", dirs.len());
    for dir in dirs {
        let ev = read_event(&dir.join("call")).unwrap();
        let hier = structure_hierarchies(&ev, model.config.levels);
        let out = model.forward(&ev, &hier, 0).unwrap();
        let want = torch_outputs(&dir);
        let mut report = Vec::new();
        for (name, shape, got) in out.named() {
            let (ws, wd) = &want[name];
            let n: usize = ws.iter().product();
            assert_eq!(shape.iter().product::<usize>(), n, "{}: {name} shape {shape:?} vs {ws:?}", dir.display());
            let e = worst(&got, wd);
            report.push(format!("{name} {e:.1e}"));
            assert!(e <= tol, "{} ({options:?}): {name} differs by {e:.3e} (tol {tol:.0e}); {}", dir.display(), report.join(", "));
        }
        // The deployed break decision (log q ≥ log τ) and the time bins agree exactly.
        let tau = w.manifest["decode"]["tau"].as_f64().unwrap_or(1.0).ln() as f32;
        let (_, q) = &want["bond_log_q"];
        let flips = out.bond_log_q.iter().zip(q).filter(|(a, b)| (**a >= tau) != (**b >= tau)).count();
        assert_eq!(flips, 0, "{}: break set differs on {flips} bonds", dir.display());
        eprintln!("{} ({options:?}): {}", dir.file_name().unwrap().to_string_lossy(), report.join(", "));
    }
}

#[test]
fn the_reference_forward_pass_matches_pytorch_on_p2s_parity_calls() {
    check(Options::default(), 1e-4);
}

#[test]
fn node_projection_gives_the_same_outputs() {
    check(Options { node_projection: true }, 1e-4);
}
