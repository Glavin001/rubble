//! P3 parity: the Rust CPU reference of P2's model (`outcome_model`) against PyTorch on fixed
//! weights. The fixture (`tests/fixtures/p3-model`, from P2's `outcome.export` through
//! `learned-impact/scripts/p3_fixture.py`) holds the trained checkpoint's weights and real
//! validation calls with PyTorch's CPU outputs.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use stress_learned::outcome_decode::{decode, DecodeConfig};
use stress_learned::outcome_model::{read_event, structure_hierarchies, Model, Options, Outputs, Weights};
use stress_learned::outcome_program::compile;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/p3-model")
}

fn parity_dirs() -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = fs::read_dir(fixture().join("parity")).unwrap().map(|e| e.unwrap().path()).collect();
    d.sort();
    d
}

/// PyTorch's outputs, and whether they were altered after the forward pass (`@kick`
/// fixtures, for the decode only).
fn torch_outputs(dir: &Path) -> (HashMap<String, (Vec<usize>, Vec<f32>)>, bool) {
    let idx: Value = serde_json::from_slice(&fs::read(dir.join("outputs.json")).unwrap()).unwrap();
    let blob = fs::read(dir.join("outputs.bin")).unwrap();
    let modified = idx.get("_modified").is_some();
    let m = idx
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| {
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let n: usize = shape.iter().product();
            let off = v["offset"].as_u64().unwrap() as usize;
            let d = blob[off..off + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            (k.clone(), (shape, d))
        })
        .collect();
    (m, modified)
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
        let (want, modified) = torch_outputs(&dir);
        if modified {
            continue;
        }
        let ev = read_event(&dir.join("call")).unwrap();
        let hier = structure_hierarchies(&ev, model.config.levels);
        let out = model.forward(&ev, &hier, 0).unwrap();
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

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (b.abs() + 1e-3)
}

/// The decode on PyTorch's own outputs gives P2's `decoded.json`: the same keys, break set,
/// bins and flags, and the f64 motion and damage to ~1e-9 (relative, floor 1e-3 m/s).
#[test]
fn the_decode_matches_p2s_decoded_states() {
    let w = Weights::load(&fixture()).unwrap();
    let cfg = DecodeConfig::from_manifest(&w.manifest).unwrap();
    let (mut n_free, mut n_bound) = (0, 0);
    for dir in parity_dirs() {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let ev = read_event(&dir.join("call")).unwrap();
        let (named, _) = torch_outputs(&dir);
        let out = Outputs::from_named(&named).unwrap();
        let (got, flags) = decode(&ev, 0, &out, &cfg, Value::Null).unwrap();
        let want: Value = serde_json::from_slice(&fs::read(dir.join("decoded.json")).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&got.flags).unwrap(), want["flags"], "{name}: flags");
        n_free += usize::from(flags.free);
        n_bound += usize::from(flags.energy_bound_active);
        let got_v = serde_json::to_value(&got).unwrap();
        for group in ["chunks", "impactors"] {
            let (g, x) = (got_v[group].as_array().unwrap(), want[group].as_array().unwrap());
            assert_eq!(g.len(), x.len(), "{name}: {group}");
            for (a, b) in g.iter().zip(x) {
                assert_eq!(a["key"], b["key"], "{name}: {group} key");
                for f in ["v", "w", "dx", "drot"] {
                    for k in 0..3 {
                        let (p, q) = (a[f][k].as_f64().unwrap(), b[f][k].as_f64().unwrap());
                        let floor = if f == "dx" || f == "drot" { 1e-3 / 60.0 } else { 1e-3 };
                        assert!((p - q).abs() <= 1e-9 * q.abs() + 1e-9 * floor, "{name}: {group} {} {f}[{k}] {p} vs {q}", a["key"]);
                    }
                }
                if let Some(q) = b.get("p_perforate") {
                    assert!(close(a["p_perforate"].as_f64().unwrap(), q.as_f64().unwrap(), 1e-6), "{name}: p_perforate");
                }
            }
        }
        assert_eq!(got_v["bonds_broken"], want["bonds_broken"], "{name}: break set and bins");
        let (g, x) = (got_v["bond_damage"].as_array().unwrap(), want["bond_damage"].as_array().unwrap());
        assert_eq!(g.len(), x.len());
        for (a, b) in g.iter().zip(x) {
            assert_eq!(a["key"], b["key"]);
            assert_eq!(a["damage"].as_f64(), b["damage"].as_f64(), "{name}: damage of bond {}", a["key"]);
        }
    }
    assert!(n_free >= 2 && n_bound >= 2, "coverage: {n_free} free islands, {n_bound} energy-bound activations");
}

/// Forward pass then decode, both in Rust, against P2's deployed decoded state: the same
/// break set and bins, the motion to f32 forward-pass precision.
#[test]
fn rust_forward_and_decode_reproduce_p2s_decoded_states() {
    let w = Weights::load(&fixture()).unwrap();
    let cfg = DecodeConfig::from_manifest(&w.manifest).unwrap();
    let model = Model::new(&w, Options { node_projection: true }).unwrap();
    for dir in parity_dirs() {
        if torch_outputs(&dir).1 {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let ev = read_event(&dir.join("call")).unwrap();
        let out = model.forward(&ev, &structure_hierarchies(&ev, model.config.levels), 0).unwrap();
        let got = serde_json::to_value(decode(&ev, 0, &out, &cfg, Value::Null).unwrap().0).unwrap();
        let want: Value = serde_json::from_slice(&fs::read(dir.join("decoded.json")).unwrap()).unwrap();
        assert_eq!(got["bonds_broken"], want["bonds_broken"], "{name}");
        assert_eq!(got["flags"], want["flags"], "{name}");
        // relative to each field's largest magnitude in the call (f32 noise near zero)
        let mut worst = 0.0f64;
        for f in ["v", "w", "dx", "drot"] {
            let rows = |d: &Value| -> Vec<[f64; 3]> {
                ["chunks", "impactors"].iter().flat_map(|g| d[g].as_array().unwrap().iter().map(|b| [0, 1, 2].map(|k| b[f][k].as_f64().unwrap())).collect::<Vec<_>>()).collect()
            };
            let (g, x) = (rows(&got), rows(&want));
            let scale = x.iter().flatten().fold(1e-12f64, |m, v| m.max(v.abs()));
            for (p, q) in g.iter().flatten().zip(x.iter().flatten()) {
                worst = worst.max((p - q).abs() / scale);
            }
        }
        assert!(worst < 1e-5, "{name}: motion differs by {worst:.2e} of its largest value");
    }
}

/// The GPU program (`outcome_program`), interpreted on the CPU, computes the reference's
/// forward pass: every op of the Slang kernels' arithmetic, with and without the
/// node-projection lever.
#[test]
fn the_gpu_program_interpreted_on_the_cpu_matches_the_reference() {
    let w = Weights::load(&fixture()).unwrap();
    for lever in [false, true] {
        let model = Model::new(&w, Options { node_projection: lever }).unwrap();
        for dir in parity_dirs() {
            if torch_outputs(&dir).1 {
                continue;
            }
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            let ev = read_event(&dir.join("call")).unwrap();
            let hier = structure_hierarchies(&ev, model.config.levels);
            let want = model.forward(&ev, &hier, 0).unwrap();
            let prog = compile(&model, &ev, &hier, 0).unwrap();
            let got = prog.outputs(&prog.run_cpu());
            for ((k, _, a), (_, _, b)) in got.named().into_iter().zip(want.named()) {
                assert_eq!(a.len(), b.len(), "{name}: {k}");
                let e = worst(&a, &b);
                assert!(e < 1e-5, "{name} (lever {lever}): {k} differs by {e:.2e} ({} ops)", prog.ops.len());
            }
        }
    }
}
