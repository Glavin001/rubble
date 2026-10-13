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
    let mut failed = 0u32;
    for scene in list_scenes() {
        // Smash-peak paper scenes exceed 120 FPS on M3 with ~1e5–3e5 contacts; still reported.
        let stress = matches!(
            scene.info.name,
            "Brick Ring (28k)"
                | "Brick Ring (110k)"
                | "Brick Walls (27k)"
                | "Box Columns (100k)"
                | "Box Pile (32k)"
        );
        let steps = if stress { 20 } else { 40 };
        let Ok((mut world, info)) = engine_scenes::build_world(scene.info.name) else {
            println!("FAIL  {} — build error", scene.info.name);
            failed += 1;
            continue;
        };
        for _ in 0..5 {
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
        println!(
            "{tag:6} {:28} bodies={:<6} p50={:7.3}ms p95={:7.3}ms (~{:.0} FPS) colors={} contacts={}",
            info.name,
            world.body_count(),
            p50,
            p95,
            fps,
            world.last_counters().color_count,
            world.last_counters().contact_count
        );
    }
    if failed > 0 {
        std::process::exit(1);
    }
}
