use engine_exec::PhysicsBackend;
use std::time::Instant;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "Box Columns (100k)".into());
    let lead: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(180);
    let steps: u32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let (mut world, info) = engine_scenes::build_world(&name).expect("build");
    println!("settle_probe {name} lead={lead} steps={steps} iters={}", info.iterations);
    for i in 0..lead {
        world.step().expect("lead");
        if i % 30 == 29 {
            let c = world.last_counters();
            println!("  lead {i}: awake={} contacts={} colors={} cand={}", c.awake_count, c.contact_count, c.color_count, c.candidate_count);
        }
    }
    let mut times = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let t0 = Instant::now();
        world.step().expect("step");
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let c = world.last_counters();
    println!(
        "RESULT p50={:.3}ms p95={:.3}ms awake={} contacts={} colors={} cand={} ov=0x{:x}",
        percentile(&times, 0.5),
        percentile(&times, 0.95),
        c.awake_count,
        c.contact_count,
        c.color_count,
        c.candidate_count,
        c.overflow_flags
    );
}
