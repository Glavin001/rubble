//! The learned impact path's input features on the GPU (`learned_feat.slang`) against P1's
//! Rust call builder (stress-learned `outcome_calls::live_call`) on the same frame-start
//! state: every routed frame's batch is exported (`Learned::export`) and the builder runs on
//! the world as the GPU left it (synced), for the same chunk and impactor keys. Edge sets
//! must be identical; every column must agree within f32 tolerance, except the documented
//! ones (`SKIP`) whose source the engine does not keep. Dry batches
//! (`STRESS_LEARNED_DRY=1`), so the state after the frame is the batch's.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_features -- --nocapture
#![cfg(feature = "gpu")]

use std::collections::{BTreeMap, HashMap};

use stress_gpu::gpu::Gpu;
use stress_gpu::learned::IMPACTOR_KEY_BIT;
use stress_gpu::world::GpuWorld;
use stress_learned::outcome_calls::{feature_columns, live_call, Arr};
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime, wall_ram_late, wall_row_late};
use stress_rubble::RubbleEngine;

/// Columns whose source the engine does not keep (documented in the P3 findings):
/// the failure margin's running maximum and the previous frame's peak utilisation (the
/// oracle's per-evaluation histories), the converged Hertz-peak QS prior, and when a cracked
/// face broke (the engine's break events).
const SKIP: &[(&str, &str)] = &[("bond_feat", "fail_margin"), ("bond_feat", "log1p_peak_util_prev"), ("bond_feat", "log_qs_hertz_util"), ("cracked_feat", "frames_since_broken")];

fn f32s(a: &Arr) -> (&[usize], &[f32]) {
    match a {
        Arr::F32(s, d) => (s, d),
        _ => panic!("not an f32 array"),
    }
}

fn i64s(a: &Arr) -> &[i64] {
    match a {
        Arr::I64(_, d) => d,
        _ => panic!("not an i64 array"),
    }
}

fn i32s(a: &Arr) -> &[i32] {
    match a {
        Arr::I32(_, d) => d,
        _ => panic!("not an i32 array"),
    }
}

#[derive(Default)]
struct Worst {
    cols: BTreeMap<(String, String), (f64, f64, f64)>,
}

impl Worst {
    fn see(&mut self, array: &str, col: &str, gpu: f32, cpu: f32) {
        let d = (gpu as f64 - cpu as f64).abs();
        let rel = d / (cpu as f64).abs().max(1.0);
        let e = self.cols.entry((array.to_string(), col.to_string())).or_insert((0.0, 0.0, 0.0));
        if rel > e.0 {
            *e = (rel, gpu as f64, cpu as f64);
        }
    }
}

/// The scene with its impact point moved across the ram's face (the oracle ensembles'
/// perturbation): the unperturbed M5 scenes are exactly symmetric, where the struck chunk
/// is a tie that f32 and f64 break differently.
fn perturbed(mut scene: stress_ref::scene::Scene, dx: f64, dz: f64) -> stress_ref::scene::Scene {
    for imp in &mut scene.impactors {
        imp.position[0] += dx;
        imp.position[2] += dz;
    }
    scene
}

fn compare(scene: stress_ref::scene::Scene, worst: &mut Worst) -> usize {
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    let mut scene = scene;
    p.apply_scene(&mut scene);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.set_tiered(true).expect("tiered coupling");
    let cols = feature_columns();
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut compared = 0;
    let mut batches = 0;
    for f in 0..frames {
        w.step_frame(&gpu).expect("frame");
        let Some(l) = w.learned.as_ref() else { continue };
        if l.totals.batches == batches {
            continue;
        }
        batches = l.totals.batches;
        let export = l.export(&gpu);
        if export.is_empty() {
            continue;
        }
        w.sync(&gpu);
        let mut world = w.with_shell(|s| s.standalone_copy());
        for call in &export {
            let ev = live_call(&mut world, "parity", &serde_json::json!({}), &call.chunk_keys, &call.impactor_keys, f, false).expect("live call");
            let c = &ev.calls[0];
            let get = |n: &str| c.arrays.iter().find(|(k, _)| k == n).map(|(_, a)| a).unwrap_or_else(|| panic!("builder array {n}"));
            let ck = i64s(get("chunk_key")).to_vec();
            let ik = i64s(get("impactor_key")).to_vec();
            let node_key = |j: i32| -> i64 {
                let j = j as usize;
                if j < ck.len() { ck[j] } else { IMPACTOR_KEY_BIT | ik[j - ck.len()] }
            };
            let mut gck = call.chunk_keys.clone();
            gck.sort_unstable();
            let mut sck = ck.clone();
            sck.sort_unstable();
            assert_eq!(gck, sck, "frame {f}: the call's chunks");
            // node arrays by key
            for (array, keys, gpu_keys) in [("chunk_feat", &ck, &call.chunk_keys), ("chunk_qs_feat", &ck, &call.chunk_keys), ("impactor_feat", &ik, &call.impactor_keys), ("impactor_qs_feat", &ik, &call.impactor_keys)] {
                let (shape, cpu) = f32s(get(array));
                let (gshape, gv) = &call.arrays[array];
                assert_eq!(shape[1], gshape[1], "{array} width");
                let w_ = shape[1];
                let at: HashMap<i64, usize> = gpu_keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
                for (r, k) in keys.iter().enumerate() {
                    let gr = at[k];
                    for col in 0..w_ {
                        worst.see(array, &cols[array][col], gv[gr * w_ + col], cpu[r * w_ + col]);
                    }
                }
            }
            {
                let (_, cpu) = f32s(get("global_feat"));
                let (_, gv) = &call.arrays["global_feat"];
                for col in 0..cpu.len() {
                    worst.see("global_feat", &cols["global_feat"][col], gv[col], cpu[col]);
                }
            }
            for (t, arrays) in [("bond", vec!["bond_feat", "bond_qs_feat"]), ("cracked", vec!["cracked_feat"]), ("impactor_contact", vec!["impactor_contact_feat"]), ("structure_contact", vec!["structure_contact_feat"])] {
                let src = i32s(get(&format!("{t}_src")));
                let dst = i32s(get(&format!("{t}_dst")));
                let bkeys = i64s(get(&format!("{t}_key")));
                let n_e = src.len() / 2;
                // edge identity: (src key, dst key) of the first twin (and the bond's key)
                let cpu_id: Vec<(i64, i64)> = (0..n_e).map(|k| (node_key(src[2 * k]), node_key(dst[2 * k]))).collect();
                let gpu_id: Vec<(i64, i64)> = call.edges[t].iter().map(|e| (e.0, e.1)).collect();
                let mut a = cpu_id.clone();
                a.sort_unstable();
                let mut b = gpu_id.clone();
                b.sort_unstable();
                if a != b {
                    for r in &call.contacts {
                        let hit = [r[1][0], r[1][1], r[1][2]];
                        let d = [r[2][0], r[2][1], r[2][2]];
                        let mut best = (f32::INFINITY, 0usize);
                        for (j, p) in call.chunk_pos.iter().enumerate() {
                            let q = [p[0] - hit[0], p[1] - hit[1], p[2] - hit[2]];
                            let along = q[0] * d[0] + q[1] * d[1] + q[2] * d[2];
                            let l = [q[0] - d[0] * along, q[1] - d[1] * along, q[2] - d[2] * along];
                            let dist = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt() + along.abs() * 1e-3;
                            if dist < best.0 {
                                best = (dist, j);
                            }
                        }
                        let mut diffs = Vec::new();
                        for (j, p) in call.chunk_pos.iter().enumerate() {
                            let q = [p[0] - hit[0], p[1] - hit[1], p[2] - hit[2]];
                            let along = q[0] * d[0] + q[1] * d[1] + q[2] * d[2];
                            let l = [q[0] - d[0] * along, q[1] - d[1] * along, q[2] - d[2] * along];
                            let dist = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt() + along.abs() * 1e-3;
                            if (dist - call.work[j][15]).abs() > 1e-4 {
                                diffs.push((j, dist, call.work[j][15]));
                            }
                        }
                        let dbg = &call.debug;
                        let distinct = |a: &[f32]| { let mut v: Vec<String> = a.iter().map(|x| format!("{x}")).collect(); v.dedup(); v };
                        eprintln!("threads saw n {:?} t0 {:?} lo.x {:?} mass {:?}", distinct(&dbg[0..256]), distinct(&dbg[256..512]), distinct(&dbg[512..768]), distinct(&dbg[768..1024]));
                        let dbg = &call.debug;
                        for (name, r) in [("ctl", 0..64), ("mask", 64..128), ("n", 128..192), ("t0", 192..256), ("lo", 256..320)] {
                            eprintln!("  {name}: {:?}", &dbg[r]);
                        }
                        eprintln!("dist mismatches ({}): {:?}", diffs.len(), &diffs[..diffs.len().min(12)]);
                        let gmin = call.work.iter().enumerate().fold((f32::INFINITY, 0usize), |a, (j, w)| if w[15] < a.0 { (w[15], j) } else { a });
                        eprintln!("host nearest: key {} ({:.4}) ; GPU entry node {} ; GPU dists min {:?} at key {} ; dist[0] {}", call.chunk_keys[best.1], best.0, r[0][1].to_bits(), gmin.0, call.chunk_keys[gmin.1], call.work[0][15]);
                    }
                    eprintln!("GPU contact rows: {:?}", call.contacts);
                    eprintln!("chunk keys {:?}", &call.chunk_keys[..call.chunk_keys.len().min(8)]);
                }
                assert_eq!(a, b, "frame {f}: {t} edge sets differ ({} builder, {} GPU)", a.len(), b.len());
                if t == "bond" || t == "cracked" {
                    for (k, e) in call.edges[t].iter().enumerate() {
                        let ci = cpu_id.iter().position(|&x| x == (e.0, e.1)).unwrap();
                        assert_eq!(bkeys[ci], e.2, "{t} edge {k}: key");
                    }
                }
                let at: HashMap<(i64, i64), usize> = gpu_id.iter().enumerate().map(|(i, &k)| (k, i)).collect();
                for array in arrays {
                    let (shape, cpu) = f32s(get(array));
                    let (_, gv) = &call.arrays[array];
                    let w_ = shape[1];
                    for (k, id) in cpu_id.iter().enumerate() {
                        let gk = at[id];
                        for twin in 0..2 {
                            for col in 0..w_ {
                                worst.see(array, &cols[array][col], gv[(2 * gk + twin) * w_ + col], cpu[(2 * k + twin) * w_ + col]);
                            }
                        }
                    }
                }
            }
            compared += 1;
        }
    }
    compared
}

#[test]
fn gpu_features_match_the_rust_call_builder() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    std::env::set_var("STRESS_LEARNED_STUB", "zero");
    std::env::set_var("STRESS_LEARNED_DRY", "1");
    std::env::set_var("STRESS_DEBRIS_ROUTING", "1");
    // The engine's columns are the builder's, by name.
    let kernel = stress_gpu::learned_feat::schema_columns();
    for (k, v) in feature_columns() {
        assert_eq!(kernel.get(k).unwrap_or_else(|| panic!("kernel lacks {k}")), &v, "{k}: columns differ from the builder's");
    }
    let mut worst = Worst::default();
    let mut n = compare(perturbed(realtime(later(wall_ram_late(), 2), 0.25), 7.3e-4, -4.1e-4), &mut worst);
    n += compare(perturbed(realtime(later(wall_row_late(2), 2), 0.25), -5.2e-4, 8.9e-4), &mut worst);
    assert!(n > 0, "no call compared");
    let mut bad = Vec::new();
    for ((a, c), (rel, g, cpu)) in &worst.cols {
        let skip = SKIP.iter().any(|(x, y)| x == a && y == c);
        let line = format!("{a}.{c}: worst {rel:.2e} (gpu {g:.6}, builder {cpu:.6}){}", if skip { " [skipped: documented]" } else { "" });
        eprintln!("{line}");
        if !skip && *rel > 2e-3 {
            bad.push(line);
        }
    }
    eprintln!("{n} calls compared");
    assert!(bad.is_empty(), "columns over tolerance:\n  {}", bad.join("\n  "));
}
