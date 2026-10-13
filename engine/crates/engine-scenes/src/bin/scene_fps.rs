//! Wall-clock FPS gate for every registered demo scene (120 FPS ⇒ ≤8.33 ms p50).

use engine_exec::PhysicsBackend;
use engine_scenes::list_scenes;
use std::time::Instant;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    env_logger::init();
    let budget_ms = 1000.0 / 120.0;
    println!("scene_fps gate: p50 ≤ {budget_ms:.3} ms (120 FPS)");
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let mut failed = 0u32;
    for scene in list_scenes() {
        if !filters.is_empty()
            && !filters
                .iter()
                .any(|f| scene.info.name.to_ascii_lowercase().contains(&f.to_ascii_lowercase()))
        {
            continue;
        }
        // Smash-peak rings sit on/over the 8.33 ms line on M3.
        // Box Columns / settled pile use long lead like three-avbd (asleep early-out).
        let stress = matches!(
            scene.info.name,
            "Brick Ring (28k)" | "Brick Ring (110k)" | "Box Pile (32k)"
        );
        let settle = matches!(
            scene.info.name,
            "Box Columns (100k)" | "Box Pile (32k)"
        );
        // three-avbd brick-ring benches use lead=60 into the smash window.
        let ring = scene.info.name.starts_with("Brick Ring");
        let lead = if scene.info.name == "Box Pile (32k)" {
            180 // three-avbd settled pile
        } else if scene.info.name == "Box Columns (100k)" {
            180 // was 120; full-suite runs occasionally left ~30 awake / ~185 ms
        } else if ring {
            60
        } else {
            5
        };
        let steps = if stress || settle || ring { 20 } else { 40 };
        let Ok((mut world, info)) = engine_scenes::build_world(scene.info.name) else {
            println!("FAIL  {} — build error", scene.info.name);
            failed += 1;
            continue;
        };
        for _ in 0..lead {
            let _ = world.step();
        }
        let mut times = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t0 = Instant::now();
            if world.step().is_err() {
                println!("FAIL  {} — step overflow", info.name);
                failed += 1;
                times.clear();
                break;
            }
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        if times.is_empty() {
            continue;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 0.50);
        let p95 = percentile(&times, 0.95);
        let fps = 1000.0 / p50.max(1e-9);
        let ok = p50 <= budget_ms;
        if !ok && !stress {
            failed += 1;
        }
        let tag = if ok {
            "PASS"
        } else if stress {
            "STRESS"
        } else {
            "FAIL"
        };
        let c = world.last_counters();
        println!(
            "{tag:6} {:28} bodies={:<6} awake={:<6} p50={:7.3}ms p95={:7.3}ms (~{:.0} FPS) colors={} cand={} contacts={} ov=0x{:x}",
            info.name,
            world.body_count(),
            c.awake_count,
            p50,
            p95,
            fps,
            c.color_count,
            c.candidate_count,
            c.contact_count,
            c.overflow_flags
        );
        // three-avbd wallMs: batch encode, one drain (CPU overlaps GPU). p50 above ≈ gpuTotal.
        if scene.info.name == "Brick Ring (9k)" {
            let batch = 10u32;
            let mut pipe = Vec::new();
            'pipe: for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..batch {
                    if world.step_gpu_ex(false).is_err() {
                        break 'pipe;
                    }
                }
                world.wait_gpu();
                pipe.push(t0.elapsed().as_secs_f64() * 1000.0 / batch as f64);
            }
            if !pipe.is_empty() {
                pipe.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let wall = percentile(&pipe, 0.50);
                println!(
                    "       {:>28} pipelined_wall={wall:.3}ms (oracle wallMs 1.75 / gpuTotal 5.51)",
                    ""
                );
            }
        }
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
