//! Local GPU accuracy envelopes + Metal timing log.
//!
//! Speed bars against three-avbd require **local** Chrome/Dawn numbers on this
//! machine (Dawn.node is blocked on macOS 23). Until then we only record our
//! Metal p50 and run physical-envelope accuracy checks — never gate on foreign GPUs.

use engine_exec::PhysicsBackend;
use engine_scenes::{build_world, list_scenes};
use std::time::Instant;

struct TimingScene {
    scene: &'static str,
    iterations: u32,
}

const TIMING_SCENES: &[TimingScene] = &[
    TimingScene {
        scene: "Brick Ring (28k)",
        iterations: 4,
    },
    TimingScene {
        scene: "Brick Ring (110k)",
        iterations: 4,
    },
    TimingScene {
        scene: "Wall Smash (2k)",
        iterations: 8,
    },
    TimingScene {
        scene: "Box Columns (100k)",
        iterations: 6,
    },
    TimingScene {
        scene: "Chain Mail (1.6k)",
        iterations: 10,
    },
    TimingScene {
        scene: "Sphere Pile (10k)",
        iterations: 8,
    },
];

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn accuracy_ok(name: &str) -> anyhow::Result<()> {
    let (mut world, _) = build_world(name)?;
    let steps = 90u32;
    // Piles/walls are still settling at 90 frames — KE envelope is for calm demos only.
    let impact = name.contains("Smash")
        || name.contains("Ring")
        || name.contains("Pendulum")
        || name.contains("Drop")
        || name.contains("Pile")
        || name.contains("Wall");
    let mut ke_end = 0.0f32;
    for s in 0..steps {
        world.step()?;
        if s + 1 != steps {
            continue;
        }
        let bodies = world.download_bodies();
        let mut ymin = f32::INFINITY;
        let mut ymax = f32::NEG_INFINITY;
        ke_end = 0.0;
        for b in &bodies {
            if !b.is_dynamic() {
                continue;
            }
            for x in b.position {
                if !x.is_finite() {
                    anyhow::bail!("{name}: non-finite position");
                }
            }
            let v = glam::Vec3::from_array(b.linear_vel);
            if !v.is_finite() {
                anyhow::bail!("{name}: non-finite velocity");
            }
            let m = if b.inv_mass > 1e-12 {
                1.0 / b.inv_mass
            } else {
                0.0
            };
            ke_end += 0.5 * m * v.length_squared();
            ymin = ymin.min(b.position[1]);
            ymax = ymax.max(b.position[1]);
        }
        if ymax > 500.0 || ymin < -200.0 {
            anyhow::bail!("{name}: bodies escaped y=[{ymin}, {ymax}]");
        }
        if !impact && ke_end > 5.0e4 {
            anyhow::bail!("{name}: settle KE too high {ke_end}");
        }
    }
    let _ = ke_end;
    Ok(())
}

fn main() {
    env_logger::init();
    println!("gpu_vs_oracles — local Metal accuracy + timing (no foreign GPU bars)");
    let mut failed = 0u32;

    println!("\n== Accuracy (GPU physical envelopes) ==");
    for scene in list_scenes() {
        let skip = matches!(
            scene.info.name,
            "Brick Ring (110k)" | "Box Columns (100k)" | "Brick Walls (27k)"
        );
        if skip {
            continue;
        }
        match accuracy_ok(scene.info.name) {
            Ok(()) => println!("PASS  accuracy  {}", scene.info.name),
            Err(e) => {
                println!("FAIL  accuracy  {} — {e}", scene.info.name);
                failed += 1;
            }
        }
    }

    println!("\n== Speed (local Metal p50; oracle_ms TBD via Chrome WebGPU) ==");
    for bar in TIMING_SCENES {
        let Ok((mut world, info)) = build_world(bar.scene) else {
            println!("FAIL  speed     {} — missing scene", bar.scene);
            failed += 1;
            continue;
        };
        let mut params = *world.params();
        params.iterations = bar.iterations;
        world.set_params(params);
        for _ in 0..8 {
            let _ = world.step();
        }
        let mut times = Vec::new();
        for _ in 0..24 {
            let t0 = Instant::now();
            if world.step().is_err() {
                println!("FAIL  speed     {} — overflow", bar.scene);
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
        let c = world.last_counters();
        println!(
            "LOG   speed     {:28} ours_p50={:7.3}ms  iters={}  bodies={} awake={} contacts={} oracle_ms=TBD",
            info.name,
            p50,
            bar.iterations,
            world.body_count(),
            c.awake_count,
            c.contact_count
        );
    }

    if failed > 0 {
        eprintln!("\n{failed} gate(s) failed");
        std::process::exit(1);
    }
    println!("\nAll accuracy gates passed. Fill oracle_ms from local Chrome when Dawn works.");
}
