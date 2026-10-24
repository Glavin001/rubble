//! CCD at scale: an N×N grid of vibe-city fractured towers (3,711 pieces each)
//! under a stream of balls, in the fracture lab's own step plan on Metal.
//!
//! Timing: ms per physics step (ticks batched four at a time, as the lab runs)
//! with no shots, with plain balls and with CCD balls. Correctness: the same
//! stream with CCD balls, every step checked for a phase-through (the ball's
//! centre path meeting a piece that did not move that step).
//!
//!   cargo test --release -p engine-viewer --test fracture_scale_ccd -- --ignored --nocapture
//!   SCALE_GRID=5 SCALE_STEPS=480 cargo test ... (25 towers, 92,775 pieces)

use engine_fracture::pack::build_pack_scene;
use engine_fracture::samples::{layout_scene, Layout};
use engine_fracture::scene::Scene;
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::path::PathBuf;
use std::time::Instant;

#[path = "support/phase.rs"]
mod phase;

fn env(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Stream {
    towers: Vec<[f64; 3]>,
    lo_y: f64,
    hi_y: f64,
    /// The grid's bounds (piece centres, padded by a tower's half extent).
    lo: [f64; 3],
    hi: [f64; 3],
    k: u64,
}

impl Stream {
    /// Next shot: a tower, a side and a height, deterministically.
    fn next(&mut self) -> ([f64; 3], [f64; 3]) {
        self.k = self.k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = |s: u32| ((self.k >> s) & 0xFFFF) as f64 / 65535.0;
        let t = self.towers[(r(16) * self.towers.len() as f64) as usize % self.towers.len()];
        let side = ((r(32) * 4.0) as usize).min(3);
        let d: [f64; 3] = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]][side];
        let y = self.lo_y + (0.1 + 0.8 * r(48)) * (self.hi_y - self.lo_y);
        // From outside the whole grid (a spawn inside a neighbour is no test):
        // the ball crosses the nearer towers on its way.
        let mut o = [t[0] + 0.137, y, t[2] + 0.071];
        for k in [0usize, 2] {
            if d[k] > 0.0 {
                o[k] = self.lo[k] - 6.0;
            } else if d[k] < 0.0 {
                o[k] = self.hi[k] + 6.0;
            }
        }
        (o, d)
    }
}

fn scene(grid: u32) -> Option<(Scene, Vec<[f64; 3]>, f64, f64, f64)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/generated/vibe-city/fracturedTower.json");
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skip: {} missing (bun run export.ts fracturedTower in engine/scenes/vibe-city)", path.display());
        return None;
    };
    let (one, _) = build_pack_scene(&bytes).unwrap();
    let one = one.without_joints();
    let per = one.projectile_start as usize - 1;
    let s = layout_scene(one, Some(Layout { suburb: false, columns: grid, rows: grid }), false);
    // Tower centres: each copy's pieces are contiguous.
    let start = s.projectile_start as usize;
    let mut towers = Vec::new();
    let (mut lo_y, mut hi_y, mut half) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for t in 0..(start - 1) / per {
        let bodies = &s.bodies[1 + t * per..1 + (t + 1) * per];
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for b in bodies {
            for k in 0..3 {
                lo[k] = lo[k].min(b.p[k]);
                hi[k] = hi[k].max(b.p[k]);
            }
        }
        towers.push([0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k])));
        lo_y = lo_y.min(lo[1]);
        hi_y = hi_y.max(hi[1]);
        half = half.max(0.5 * (hi[0] - lo[0]).max(hi[2] - lo[2]));
    }
    Some((s, towers, lo_y, hi_y, half))
}

#[test]
#[ignore = "benchmark: about a minute at the default 3x3 grid"]
fn ccd_balls_at_scale_timing_and_no_phase_through() {
    let grid = env("SCALE_GRID", 3);
    let steps = env("SCALE_STEPS", 240);
    let every = env("SCALE_EVERY", 15);
    let scale = env("SCALE_SPEED", 16) as f64;
    let Some((scene, towers, lo_y, hi_y, half)) = scene(grid) else { return };
    let n = scene.bodies.len();
    let start = scene.projectile_start as usize;
    println!(
        "{} towers, {} pieces, {} projectile slots, a ball every {every} steps at {scale}x, {steps} steps",
        towers.len(),
        start - 1,
        scene.projectile_count
    );
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
    let grid_lo = [towers.iter().map(|t| t[0]).fold(f64::INFINITY, f64::min) - half, 0.0, towers.iter().map(|t| t[2]).fold(f64::INFINITY, f64::min) - half];
    let grid_hi = [towers.iter().map(|t| t[0]).fold(f64::NEG_INFINITY, f64::max) + half, 0.0, towers.iter().map(|t| t[2]).fold(f64::NEG_INFINITY, f64::max) + half];
    // Timing: batches of four ticks, a shot between batches when due.
    let mut timings = Vec::new();
    for (label, shots, ccd) in [("no shots", false, false), ("plain balls", true, false), ("CCD balls", true, true)] {
        sim.reset();
        sim.step(4).unwrap();
        sim.plan.set_projectile_ccd(ccd);
        sim.plan.set_projectile_speed_scale(scale).unwrap();
        let mut stream = Stream { towers: towers.clone(), lo_y, hi_y, lo: grid_lo, hi: grid_hi, k: env("SCALE_SEED", 7) as u64 };
        let (mut peak_awake, mut peak_contacts) = (0, 0);
        let t0 = Instant::now();
        let mut done = 0;
        while done < steps {
            if shots && done % every < 4 {
                let (o, d) = stream.next();
                sim.launch(o, d).unwrap();
            }
            sim.submit(4).unwrap();
            sim.wait().unwrap();
            assert_eq!(sim.last.overflow, 0, "{label}: GPU capacity overflow at step {done}");
            peak_awake = peak_awake.max(sim.last.awake);
            peak_contacts = peak_contacts.max(sim.last.contact);
            done += 4;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        println!("{label:<12}: {ms:6.2} ms/step, peak awake {peak_awake}, peak contacts {peak_contacts}");
        timings.push(ms);
    }
    println!("CCD cost over plain balls: {:+.2} ms/step ({:+.0}%)", timings[2] - timings[1], 100.0 * (timings[2] / timings[1] - 1.0));
    // Correctness: one step at a time, every ball checked every step.
    let pieces = phase::pieces(&scene);
    sim.reset();
    sim.step(4).unwrap();
    sim.plan.set_projectile_ccd(true);
    sim.plan.set_projectile_speed_scale(scale).unwrap();
    let mut stream = Stream { towers, lo_y, hi_y, lo: grid_lo, hi: grid_hi, k: env("SCALE_SEED", 7) as u64 };
    let slots: Vec<usize> = (start..start + scene.projectile_count as usize).collect();
    let mut prev = sim.snapshot(n).unwrap();
    let mut failures = Vec::new();
    for step in 0..steps {
        if step % every == 0 {
            let (o, d) = stream.next();
            sim.launch(o, d).unwrap();
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
    println!("{} phase-throughs over {steps} steps", failures.len());
    assert!(failures.is_empty(), "CCD balls phased through:\n{}", failures.join("\n"));
}
