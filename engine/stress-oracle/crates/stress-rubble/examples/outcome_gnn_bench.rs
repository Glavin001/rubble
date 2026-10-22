//! P3 timing of P2's trained model through the Slang port (`stress_gpu::outcome_gnn`, the op
//! program of `stress_learned::outcome_program`): the network call alone, per workload, on
//! this machine's GPU, with and without the node-projection lever. One JSON line per
//! (workload, lever) on stdout, progress on stderr. With `--write-programs DIR`, also each
//! compiled program as `DIR/<workload>_<lever>.ognn` for the browser page
//! (`learned-impact/web-probe/outcome.html`, the same programs on WebGPU).
//!
//!   outcome_gnn_bench --export DIR [--warmup 5] [--repeats 30] [--only NAME,...]
//!       [--write-programs DIR] [--no-lever | --lever-only] [--check]
//!
//! `--check` first compares each program's GPU outputs with the CPU interpreter's.
//!
//! Workloads:
//! - `m5_free_wall`: the M5 free wall's contact call (220 chunks, 508 bonds);
//! - `m5_wall_row`: the wall row's contact call (440 chunks, 1016 bonds);
//! - `batch_220_{1,5,20,50}`: that many distinct real ~220-chunk contact calls (P1's
//!   validation calls) in one batch program;
//! - `island_10k`: one synthetic 10,000-chunk island (a 40×25×10 bonded grid, the calls'
//!   feature rows cycled from the M5 free-wall call, one impactor in contact), for the size
//!   the plan targets; its values are not physical, its shape is.
//!
//! Inputs: `LEARNED_IMPACT_P3_CALLS` (default `learned-impact/runs/p3-bench/calls`, the M5
//! calls built by `learned_impact_outcome_calls`) and `LEARNED_IMPACT_P3_BATCH` (a list of
//! `EVENT_DIR:CALL` lines; default: the parity fixture's calls, repeated).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use stress_gpu::gpu::Gpu;
use stress_gpu::outcome_gnn::{encode_program, OutcomeKernels, OutcomeRun};
use stress_learned::outcome_calls::{Arr, Call, CallEvent};
use stress_learned::outcome_model::{read_event, structure_hierarchies, Hierarchy, Model, Options, Weights};
use stress_learned::outcome_program::compile_batch;

fn stats(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    json!({"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "min": s[0], "n": s.len()})
}

/// A synthetic island of `nx × ny × nz` chunks (a bonded grid at the template's spacing),
/// with the template call's feature rows cycled and one impactor touching chunk 0.
fn synthetic_island(t: &CallEvent, nx: usize, ny: usize, nz: usize) -> CallEvent {
    let tc = &t.calls[0];
    let n = nx * ny * nz;
    let id = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let mut ends = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if x + 1 < nx {
                    ends.push((id(x, y, z), id(x + 1, y, z)));
                }
                if y + 1 < ny {
                    ends.push((id(x, y, z), id(x, y + 1, z)));
                }
                if z + 1 < nz {
                    ends.push((id(x, y, z), id(x, y, z + 1)));
                }
            }
        }
    }
    let l = tc.scales.get("L").copied().unwrap_or(0.1) as f32;
    let pos: Vec<f32> = (0..n).flat_map(|i| [(i % nx) as f32 * l, ((i / nx) % ny) as f32 * l, (i / (nx * ny)) as f32 * l]).collect();
    let cyc = |name: &str, rows: usize| -> Arr {
        match tc.array(name).expect(name) {
            Arr::F32(s, d) => {
                let w: usize = s[1..].iter().product::<usize>().max(1);
                let r0 = s[0].max(1);
                let mut out = Vec::with_capacity(rows * w);
                for r in 0..rows {
                    out.extend_from_slice(&d[(r % r0) * w..(r % r0) * w + w]);
                }
                let mut shape = s.clone();
                shape[0] = rows;
                Arr::F32(shape, out)
            }
            other => other.clone(),
        }
    };
    let nb = ends.len();
    let mut arrays: Vec<(String, Arr)> = Vec::new();
    let mut put = |k: &str, a: Arr| arrays.push((k.to_string(), a));
    for k in ["chunk_feat", "chunk_qs_feat", "chunk_inertia_body", "chunk_v", "chunk_w"] {
        put(k, cyc(k, n));
    }
    put("chunk_mass", cyc("chunk_mass", n));
    put("chunk_pos", Arr::F32(vec![n, 3], pos.clone()));
    put("chunk_q_world", Arr::F32(vec![n, 4], (0..n).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect()));
    put("chunk_key", Arr::I64(vec![n], (0..n as i64).collect()));
    put("chunk_anchored", Arr::I32(vec![n], vec![0; n]));
    for k in ["impactor_feat", "impactor_qs_feat", "impactor_inertia_body", "impactor_mass", "impactor_pos", "impactor_q_world", "impactor_v", "impactor_w", "impactor_key"] {
        put(k, tc.array(k).expect(k).clone());
    }
    put("global_feat", tc.array("global_feat").unwrap().clone());
    put("bond_src", Arr::I32(vec![2 * nb], ends.iter().flat_map(|&(a, b)| [a as i32, b as i32]).collect()));
    put("bond_dst", Arr::I32(vec![2 * nb], ends.iter().flat_map(|&(a, b)| [b as i32, a as i32]).collect()));
    put("bond_feat", cyc("bond_feat", 2 * nb));
    put("bond_qs_feat", cyc("bond_qs_feat", 2 * nb));
    put("bond_key", Arr::I64(vec![nb], (0..nb as i64).collect()));
    // the impactor (node n) against chunk 0, both directions
    put("impactor_contact_src", Arr::I32(vec![2], vec![n as i32, 0]));
    put("impactor_contact_dst", Arr::I32(vec![2], vec![0, n as i32]));
    put("impactor_contact_feat", cyc("impactor_contact_feat", 2));
    for t in ["cracked", "structure_contact"] {
        put(&format!("{t}_src"), Arr::I32(vec![0], vec![]));
        put(&format!("{t}_dst"), Arr::I32(vec![0], vec![]));
        put(&format!("{t}_feat"), cyc(&format!("{t}_feat"), 0));
    }
    let mut counts = BTreeMap::new();
    counts.insert("chunks".to_string(), n);
    counts.insert("impactors".to_string(), 1);
    counts.insert("bond".to_string(), nb);
    let call = Call { frame: 0, prev_call: None, phase: "contact".into(), transition: false, anchored: false, counts, scales: tc.scales.clone(), energy: tc.energy.clone(), arrays };
    let structures = vec![(0usize, (0..n as i64).collect(), (0..nb as i64).collect(), ends.iter().flat_map(|&(a, b)| [a as i32, b as i32]).collect(), pos)];
    CallEvent { manifest: t.manifest.clone(), calls: vec![call], structures }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut export, mut warmup, mut repeats, mut only, mut write, mut levers, mut check) = (None, 5usize, 30usize, None::<Vec<String>>, None::<PathBuf>, vec![false, true], false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--export" => export = args.next().map(PathBuf::from),
            "--warmup" => warmup = args.next().and_then(|v| v.parse().ok()).expect("--warmup N"),
            "--repeats" => repeats = args.next().and_then(|v| v.parse().ok()).expect("--repeats N"),
            "--only" => only = args.next().map(|v| v.split(',').map(str::to_string).collect()),
            "--write-programs" => write = args.next().map(PathBuf::from),
            "--no-lever" => levers = vec![false],
            "--lever-only" => levers = vec![true],
            "--check" => check = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let export = export.expect("--export DIR (a learned-impact-weights/1 export)");
    let weights = Weights::load(&export).unwrap_or_else(|e| panic!("{e}"));
    let calls_dir = PathBuf::from(std::env::var("LEARNED_IMPACT_P3_CALLS").unwrap_or_else(|_| "learned-impact/runs/p3-bench/calls".into()));
    let load = |d: &Path| -> CallEvent { read_event(d).unwrap_or_else(|e| panic!("{}: {e}", d.display())) };
    let fw = load(&calls_dir.join("m5_free_wall"));
    let wr = load(&calls_dir.join("m5_wall_row"));
    // the batch's calls: EVENT_DIR:CALL lines
    let batch_src: Vec<(PathBuf, usize)> = match std::env::var("LEARNED_IMPACT_P3_BATCH") {
        Ok(f) => std::fs::read_to_string(&f)
            .expect("batch list")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (d, c) = l.rsplit_once(':').expect("EVENT_DIR:CALL");
                (PathBuf::from(d), c.parse().expect("call index"))
            })
            .collect(),
        Err(_) => vec![(calls_dir.join("m5_free_wall"), 0)],
    };
    let mut batch_evs: Vec<(CallEvent, usize)> = Vec::new();
    for (d, c) in batch_src.iter().take(50) {
        let ev = load(d);
        // `CALL` is the call's position in its event (its `call_index`)
        let ci = (*c).min(ev.calls.len() - 1);
        batch_evs.push((ev, ci));
    }
    while batch_evs.len() < 50 {
        let k = batch_evs.len() % batch_src.len().max(1);
        batch_evs.push((load(&batch_src[k].0), batch_src[k].1));
    }
    let island = synthetic_island(&fw, 40, 25, 10);
    let levels = weights.config.levels;
    let hier_of = |ev: &CallEvent| -> Vec<Hierarchy> { structure_hierarchies(ev, levels) };
    eprintln!("[bench] building hierarchies");
    let fw_h = hier_of(&fw);
    let wr_h = hier_of(&wr);
    let batch_h: Vec<Vec<Hierarchy>> = batch_evs.iter().map(|(e, _)| hier_of(e)).collect();
    let island_h = hier_of(&island);
    let gpu = Gpu::new().expect("GPU");
    let kernels = OutcomeKernels::new(&gpu);
    let info = json!({"adapter": gpu.adapter_info.name, "backend": format!("{:?}", gpu.adapter_info.backend), "shaders": format!("{:?}", gpu.shader_path), "timestamps": gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY), "export": export.display().to_string(), "model": weights.manifest["model_config"]});
    println!("{}", json!({"info": info}));
    let workloads: Vec<(String, Vec<(&CallEvent, &[Hierarchy], usize)>)> = {
        let mut w: Vec<(String, Vec<(&CallEvent, &[Hierarchy], usize)>)> = vec![
            ("m5_free_wall".into(), vec![(&fw, &fw_h[..], 0)]),
            ("m5_wall_row".into(), vec![(&wr, &wr_h[..], 0)]),
        ];
        for nb in [1usize, 5, 20, 50] {
            w.push((format!("batch_220_{nb}"), (0..nb).map(|k| (&batch_evs[k].0, &batch_h[k][..], batch_evs[k].1)).collect()));
        }
        w.push(("island_10k".into(), vec![(&island, &island_h[..], 0)]));
        w
    };
    for (name, calls) in &workloads {
        if only.as_ref().is_some_and(|o| !o.contains(name)) {
            continue;
        }
        for &lever in &levers {
            let model = Model::new(&weights, Options { node_projection: lever }).unwrap();
            let t0 = std::time::Instant::now();
            let prog = compile_batch(&model, calls).unwrap_or_else(|e| panic!("{name}: {e}"));
            let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
            let ops: Vec<_> = prog.ops.iter().map(|o| (o.params, o.groups)).collect();
            if let Some(dir) = &write {
                std::fs::create_dir_all(dir).unwrap();
                std::fs::write(dir.join(format!("{name}_{}.ognn", if lever { "lever" } else { "plain" })), encode_program(&prog.arena, &prog.weights, &prog.table, &ops)).unwrap();
            }
            let t1 = std::time::Instant::now();
            let run = OutcomeRun::new(&gpu, &kernels, &prog.arena, &prog.weights, &prog.table, &ops);
            let upload_ms = t1.elapsed().as_secs_f64() * 1e3;
            let mut worst = Value::Null;
            if check {
                // The GPU's outputs against the program interpreted on the CPU, every call.
                let got = run.run(&gpu, &kernels);
                let want = prog.run_cpu();
                let mut w = 0.0f32;
                for k in 0..calls.len() {
                    for ((_, _, a), (_, _, b)) in prog.outputs_of(&got, k).named().into_iter().zip(prog.outputs_of(&want, k).named()) {
                        w = a.iter().zip(&b).map(|(x, y)| (x - y).abs() / (y.abs() + 1.0)).fold(w, f32::max);
                    }
                }
                assert!(w < 1e-4 && w.is_finite(), "{name} lever {lever}: GPU and CPU differ by {w:.2e}");
                worst = json!(w);
            }
            let (wall, gt) = run.time(&gpu, &kernels, warmup, repeats);
            let (nc, ni, nb): (usize, usize, usize) = prog.layout.calls.iter().fold((0, 0, 0), |a, c| (a.0 + c.0, a.1 + c.1, a.2 + c.2));
            let line = json!({"workload": name, "lever": lever, "calls": calls.len(), "chunks": nc, "impactors": ni, "bonds": nb, "dispatches": run.dispatches(),
                "arena_mb": prog.arena.len() as f64 * 4e-6, "table_mb": prog.table.len() as f64 * 4e-6, "compile_ms": compile_ms, "upload_ms": upload_ms,
                "wall_ms": stats(&wall), "gpu_ms": stats(&gt), "gpu_vs_cpu": worst});
            eprintln!("[bench] {name} lever {lever}: {nc} chunks, {} dispatches, gpu {:.3} ms, wall {:.3} ms (compile {compile_ms:.0} ms)", run.dispatches(), line["gpu_ms"]["median"].as_f64().unwrap_or(f64::NAN), line["wall_ms"]["median"].as_f64().unwrap_or(f64::NAN));
            println!("{line}");
        }
    }
}
