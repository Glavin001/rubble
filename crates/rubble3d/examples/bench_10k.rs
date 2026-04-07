//! Headless GPU benchmark for the 10k Grid scene.
//! Usage: RUBBLE_PRECISE_GPU_TIMING=1 cargo run --release --example bench_10k
//! Outputs JSON timing stats to stdout.

use rubble3d::{SimConfig, World};
use rubble_scenes::scenes_3d::scene_grid_10k_boxes;

fn main() {
    // Set precise timing env if not already set
    if std::env::var("RUBBLE_PRECISE_GPU_TIMING").is_err() {
        unsafe { std::env::set_var("RUBBLE_PRECISE_GPU_TIMING", "1") };
    }

    let config = SimConfig {
        max_bodies: 11000,
        ..Default::default()
    };
    let mut world = World::new(config).expect("GPU adapter required");

    // Load 10k Grid scene
    let bodies = scene_grid_10k_boxes();
    for desc in &bodies {
        world.add_body(desc);
    }
    eprintln!("Loaded {} bodies", world.body_count());

    let warmup_steps = 30;
    let bench_steps = 60;

    // Warmup (let contacts stabilize + GPU shader compilation)
    for _ in 0..warmup_steps {
        world.step();
    }
    eprintln!("Warmup complete ({warmup_steps} steps)");

    // Benchmark
    let mut step_times = Vec::with_capacity(bench_steps);
    let mut solve_times = Vec::with_capacity(bench_steps);
    let mut swap_times = Vec::with_capacity(bench_steps);
    let mut iter_times = Vec::with_capacity(bench_steps);
    let mut coloring_times = Vec::with_capacity(bench_steps);

    for i in 0..bench_steps {
        world.step();
        let t = world.last_step_timings();
        let sb = &t.solve_breakdown;
        step_times.push(t.as_array().iter().sum::<f32>());
        solve_times.push(t.solve_ms);
        swap_times.push(sb.swap_ms);
        iter_times.push(sb.iterations_ms);
        coloring_times.push(sb.coloring_ms);

        if i % 20 == 0 {
            eprintln!(
                "  step {i}: step={:.1}ms solve={:.1}ms swap={:.1}ms iter={:.1}ms",
                step_times.last().unwrap(),
                solve_times.last().unwrap(),
                swap_times.last().unwrap(),
                iter_times.last().unwrap()
            );
        }
    }

    fn median(v: &mut Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        if n % 2 == 0 {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        }
    }

    let result = serde_json::json!({
        "bodies": world.body_count(),
        "warmup_steps": warmup_steps,
        "bench_steps": bench_steps,
        "median_step_ms": median(&mut step_times),
        "median_solve_ms": median(&mut solve_times),
        "median_swap_ms": median(&mut swap_times),
        "median_iterations_ms": median(&mut iter_times),
        "median_coloring_ms": median(&mut coloring_times),
        "precise_gpu": world.last_step_timings().precise_gpu,
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
