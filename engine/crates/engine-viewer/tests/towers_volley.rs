//! The CCD showcase's volley (`fracture_viewer towers --volley`), headless, in
//! the lab's step plan on Metal: a block of vibe-city fractured towers under
//! fast off-angle balls, then shots at their lower floors until they fall into
//! each other.
//!
//! - Timing: the volley as the viewer runs it (one tick per 120 Hz frame,
//!   submitted and waited), ms per tick against the 8.33 ms real-time budget.
//! - Correctness: the same volley one step at a time with every ball checked
//!   every step for a phase-through (its centre path meeting a piece that did
//!   not move that step); with CCD off as the control.
//!
//!   cargo test --release -p engine-viewer --test towers_volley -- --ignored --nocapture
//!   VOLLEY_GRID=3x2 VOLLEY_SPEED=24 VOLLEY_LOOSE=1 cargo test ...  (defaults: the
//!   showcase: 2×2 bonded towers at 0.2× bond strength, 30 t 1 m balls at 16×)

use engine_viewer::fracture::volley::{towers_scene, volley};
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::time::Instant;

#[path = "support/phase.rs"]
mod phase;

fn env(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

#[test]
#[ignore = "benchmark: needs the generated fracturedTower pack, a few minutes"]
fn the_volley_never_phases_through_and_runs_in_real_time() {
    let grid = env("VOLLEY_GRID", "2x2");
    let (columns, rows) = grid
        .split_once('x')
        .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
        .unwrap();
    let scale: f64 = env("VOLLEY_SPEED", "16").parse().unwrap();
    let (raking, base, every): (u32, u32, u32) = (
        env("VOLLEY_RAKING", "8").parse().unwrap(),
        env("VOLLEY_BASE", "16").parse().unwrap(),
        env("VOLLEY_EVERY", "45").parse().unwrap(),
    );
    let gap: f64 = env("VOLLEY_GAP", "6").parse().unwrap();
    let scene = match towers_scene(
        columns,
        rows,
        gap,
        std::env::var("VOLLEY_LOOSE").is_err(),
        Some(env("VOLLEY_BALL", "1.0,30000")).and_then(|b| {
            b.split_once(',')
                .and_then(|(r, m)| Some((r.parse().ok()?, m.parse().ok()?)))
        }),
        env("VOLLEY_STRENGTH", "0.2").parse().unwrap(),
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    let shots = volley(&scene, raking, base, every);
    let steps = shots.last().unwrap().step + 360;
    let n = scene.bodies.len();
    let hz = 1.0 / scene.params.dt as f64;
    println!(
        "{columns}×{rows} towers, {} pieces, {} shots at {scale}×, {steps} steps at {hz:.0} Hz",
        scene.projectile_start - 1,
        shots.len()
    );
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
    let b = engine_viewer::fracture::volley::block(&scene);
    println!(
        "shot speed {:.0} m/s, ball radius {:.2} m, mass {:.0} kg; tower {:.1} m wide, {:.1} m tall",
        sim.plan.shot_speed() * scale,
        scene.projectile_radius,
        scene.projectile_mass,
        2.0 * b.half,
        b.hi[1] - b.lo[1]
    );

    // Timing, as the viewer runs it: one tick submitted and waited per frame.
    let mut slow = Vec::new();
    for ccd in [false, true] {
        sim.reset();
        sim.plan.set_projectile_ccd(ccd);
        sim.plan.set_projectile_speed_scale(scale).unwrap();
        let mut next = 0;
        let (mut total, mut worst, mut over, mut peak_awake) = (0.0f64, 0.0f64, 0, 0);
        let mut window = Vec::new();
        for step in 0..steps {
            while next < shots.len() && shots[next].step == step {
                sim.launch(shots[next].origin, shots[next].dir).unwrap();
                next += 1;
            }
            let t = Instant::now();
            sim.submit(1).unwrap();
            sim.wait().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(sim.last.overflow, 0, "GPU capacity overflow at step {step}");
            total += ms;
            worst = worst.max(ms);
            window.push(ms);
            if ms > 1e3 / hz {
                over += 1;
            }
            peak_awake = peak_awake.max(sim.last.awake);
            if ccd && std::env::var("VOLLEY_PROFILE").is_ok() && step % 300 == 150 {
                let p = sim.profile_step().unwrap();
                let total: f64 = p.iter().map(|x| x.1).sum();
                println!("  step {step} ({} awake, {} colours, {} dispatches/tick): {total:.2} ms split: {}", sim.last.awake, sim.last.color, sim.profile_dispatches, p.iter().map(|(n, ms)| format!("{n} {ms:.2}")).collect::<Vec<_>>().join(", "));
            }
        }
        // Worst one-second window (120 ticks): sustained real time.
        let per_second = window
            .chunks(hz as usize)
            .map(|c| c.iter().sum::<f64>() / c.len() as f64)
            .fold(0.0f64, f64::max);
        println!(
            "CCD {ccd:<5}: {:.2} ms/tick mean, worst second {:.2} ms/tick, worst tick {:.1} ms, {over} of {steps} ticks over {:.2} ms, peak awake {peak_awake}",
            total / steps as f64,
            per_second,
            worst,
            1e3 / hz
        );
        if ccd {
            slow.push(per_second);
        }
    }

    // Correctness: one step at a time, every ball every step.
    let pieces = phase::pieces(&scene);
    let start = scene.projectile_start as usize;
    let slots: Vec<usize> = (start..start + scene.projectile_count as usize).collect();
    let mut counts = Vec::new();
    for ccd in [false, true] {
        sim.reset();
        sim.plan.set_projectile_ccd(ccd);
        sim.plan.set_projectile_speed_scale(scale).unwrap();
        let mut prev = sim.snapshot(n).unwrap();
        let mut next = 0;
        let mut failures = Vec::new();
        for step in 0..steps {
            while next < shots.len() && shots[next].step == step {
                sim.launch(shots[next].origin, shots[next].dir).unwrap();
                next += 1;
                prev = sim.snapshot(n).unwrap();
            }
            sim.step(1).unwrap();
            let now = sim.snapshot(n).unwrap();
            for &ball in &slots {
                if let Some(f) = phase::crossing(&scene, &pieces, &prev, &now, ball) {
                    failures.push(format!("step {step}: {f}"));
                }
            }
            prev = now;
        }
        println!(
            "CCD {ccd:<5}: {} phase-throughs over {steps} steps{}",
            failures.len(),
            failures
                .first()
                .map(|f| format!(" (first: {f})"))
                .unwrap_or_default()
        );
        counts.push(failures);
    }
    assert!(
        !counts[0].is_empty(),
        "control: without CCD the volley should phase through pieces"
    );
    assert!(
        counts[1].is_empty(),
        "CCD balls phased through:\n{}",
        counts[1].join("\n")
    );
    if std::env::var("VOLLEY_REALTIME").is_ok() {
        assert!(
            slow[0] <= 1e3 / 120.0,
            "worst second {:.2} ms/tick is over the 120 Hz budget",
            slow[0]
        );
    }
}
