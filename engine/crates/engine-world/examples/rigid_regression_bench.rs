//! Rigid-only regression benchmark: the World's rigid step at scale, with no vehicles,
//! soft bodies or destructibles, so two builds can be compared on the same scenes
//! (`engine/docs/measurements/rigid-regression-2026-11-07.md`). Only API present on
//! origin/main before the soft-body vehicle work is used, so the file builds unchanged
//! on both sides of a comparison.
//!
//! Scenes (one per run; one CSV line per run to stdout):
//!   sparse N   N unit boxes on a ground plane, 1 % awake (dropped from 2..92 m, so
//!              they land in a stream through the measured steps), the rest asleep at
//!              rest (large maps: cost should follow the awake 1 %)
//!   awake N    N unit boxes resting on the plane, sleep off: every body awake each step
//!   stacks T H T towers of H boxes, awake, sleep on (RR_SLEEP=0: off, the stacks held
//!              awake every step)
//!   map L      an idle L m square map: buildings (400 per km²) asleep, nothing awake
//!   map1 L     the same map plus one box falling (the step runs; its cost should not
//!              follow the sleeping buildings)
//!   demo NAME  a demo scene (`engine_scenes::build_host`, e.g. "Box Columns (100k)",
//!              "Box Pile (32k)", "Pyramid 30", "100k Grid"), imported as the equivalence
//!              harness does (tests/equivalence.rs), sleep on (RR_SLEEP=0: off)
//!
//! Each step's wall time is `World::step` plus `poll_blocking` (the GPU idle at each
//! submit, so this is the step's GPU span plus the submit and readback); the GPU column
//! is the step's own pass from timestamp queries (`GpuTiming::Step`; none for a step an
//! all-asleep world skips). After the measured steps every body's position, linear
//! velocity and flags are hashed (FNV-1a over their bits) for a determinism check.
//!
//! CSV: label, scene, param, bodies, wall p50 ms, wall p95 ms, gpu p50 ms, steps that
//! ran, awake at the end, contacts at the end, state hash.
//!
//! Environment: RR_WARM (steps before the measured ones: 30 for `awake` and the maps, else
//! 0 so the settling is what is measured), RR_STEPS (measured, 300),
//! RR_LABEL (first CSV column), RR_DUMP=path (write the final bodies, 28 bytes each).
//!
//! perl -e 'alarm 600; exec @ARGV' cargo run --release -p engine-world \
//!   --example rigid_regression_bench -- sparse 100000

use rubble::*;
use std::time::Instant;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * q).round() as usize]
}

fn world(bodies: u32, sleep: bool) -> anyhow::Result<World> {
    let mut cap = WorldCapacity::small();
    let b = (bodies + 1024).next_power_of_two().max(cap.max_bodies);
    cap.max_bodies = b;
    cap.max_colliders = (2 * b).max(cap.max_colliders);
    cap.max_contacts = cap.max_contacts.max(8 * b);
    cap.max_candidates = cap.max_candidates.max(16 * b);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: cap,
            enable_sleep: sleep,
            gpu_timing: GpuTiming::Step,
            ..WorldDef::default()
        },
    )?;
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )?;
    Ok(w)
}

fn boxes(w: &mut World, defs: &[BodyDef], half: [f32; 3]) -> anyhow::Result<()> {
    for b in w.create_bodies(defs)? {
        w.create_box_shape(b, &ShapeDef::default(), half)?;
    }
    Ok(())
}

struct Stats {
    wall: Vec<f64>,
    gpu: Vec<f64>,
    ran: usize,
}

fn measure(w: &mut World, warm: u32, steps: u32) -> anyhow::Result<Stats> {
    for _ in 0..warm {
        w.step()?;
        w.poll_blocking()?;
    }
    let mut s = Stats {
        wall: Vec::new(),
        gpu: Vec::new(),
        ran: 0,
    };
    for _ in 0..steps {
        let t = Instant::now();
        w.step()?;
        let frames = w.poll_blocking()?;
        s.wall.push(t.elapsed().as_secs_f64() * 1e3);
        for f in frames {
            if let Some(g) = f.gpu_time {
                s.gpu.push(g.total_ms);
                s.ran += 1;
            }
        }
    }
    Ok(s)
}

fn hash(w: &World, dump: Option<String>) -> u64 {
    let mut bytes = Vec::new();
    for b in w.sim().download_bodies() {
        for v in b.position.iter().chain(b.linear_vel.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&b.flags.to_le_bytes());
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in &bytes {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    if let Some(p) = dump {
        std::fs::write(p, &bytes).expect("dump");
    }
    h
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let scene = args.first().cloned().unwrap_or_else(|| "sparse".into());
    let nums: Vec<usize> = args.iter().skip(1).filter_map(|a| a.parse().ok()).collect();
    let n0 = |d: usize| nums.first().copied().unwrap_or(d);
    let warm: u32 = env(
        "RR_WARM",
        if matches!(scene.as_str(), "awake" | "map" | "map1") {
            30
        } else {
            0
        },
    );
    let steps: u32 = env("RR_STEPS", 300);
    let (mut w, bodies, param) = match scene.as_str() {
        "sparse" | "awake" => {
            let n = n0(10_000);
            let sparse = scene == "sparse";
            let mut w = world(n as u32, sparse)?;
            let cols = (n as f32).sqrt().ceil() as usize;
            let defs: Vec<BodyDef> = (0..n)
                .map(|i| {
                    let falling = sparse && i % 100 == 0;
                    let p = [
                        (i % cols) as f32 * 1.5,
                        if falling {
                            2.0 + ((i / 100) % 10) as f32 * 10.0
                        } else {
                            0.5
                        },
                        (i / cols) as f32 * 1.5,
                    ];
                    BodyDef {
                        is_awake: !sparse || falling,
                        ..BodyDef::dynamic(p)
                    }
                })
                .collect();
            boxes(&mut w, &defs, [0.5; 3])?;
            (w, n, n.to_string())
        }
        "stacks" => {
            let t = n0(400);
            let h = nums.get(1).copied().unwrap_or(10);
            let n = t * h;
            let mut w = world(n as u32, env("RR_SLEEP", 1u32) == 1)?;
            let cols = (t as f32).sqrt().ceil() as usize;
            let defs: Vec<BodyDef> = (0..n)
                .map(|i| {
                    let (tower, level) = (i / h, i % h);
                    BodyDef::dynamic([
                        (tower % cols) as f32 * 3.0,
                        0.5 + level as f32,
                        (tower / cols) as f32 * 3.0,
                    ])
                })
                .collect();
            boxes(&mut w, &defs, [0.5; 3])?;
            (w, n, format!("{t}x{h}"))
        }
        "map" | "map1" => {
            let l = n0(2000) as f32;
            let nb = (400.0 * l * l / 1e6).round() as usize;
            let mut w = world(nb as u32 + 1, true)?;
            let cols = (nb as f32).sqrt().ceil() as usize;
            let pitch = l / cols as f32;
            let defs: Vec<BodyDef> = (0..nb)
                .map(|i| BodyDef {
                    is_awake: false,
                    ..BodyDef::dynamic([
                        (i % cols) as f32 * pitch - 0.5 * l,
                        4.0,
                        (i / cols) as f32 * pitch - 0.5 * l,
                    ])
                })
                .collect();
            boxes(&mut w, &defs, [2.0, 4.0, 2.0])?;
            if scene == "map1" {
                // Between the first buildings, high enough to fall the whole run.
                let b = w.create_body(&BodyDef::dynamic([
                    -0.5 * l + 0.5 * pitch,
                    200.0,
                    -0.5 * l + 0.5 * pitch,
                ]))?;
                w.create_box_shape(b, &ShapeDef::default(), [0.5; 3])?;
            }
            (w, nb, format!("{l}"))
        }
        "demo" => {
            let name = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "Box Pile (32k)".into());
            let sleep = env("RR_SLEEP", 1u32) == 1;
            let (host, _) = engine_scenes::build_host(&name)?;
            let snap = host.snapshot();
            let n = snap.bodies.len() as u32 + 8;
            let per_body = 16;
            let def = WorldDef {
                capacity: WorldCapacity {
                    max_bodies: n.next_power_of_two(),
                    max_colliders: n.next_power_of_two(),
                    max_contacts: (per_body * n).next_power_of_two(),
                    max_candidates: (per_body * n).next_power_of_two(),
                    max_colors: 32,
                    ..WorldCapacity::small()
                },
                enable_sleep: sleep,
                gpu_timing: GpuTiming::Step,
                ..WorldDef::default()
            };
            let mut w = World::new(GpuContext::Default, &def)?;
            w.import_snapshot(&snap)?;
            // The snapshot's solver replaces the world's: set sleep again.
            let mut solver = w.solver();
            solver.sleep_time = if sleep { 0.5 } else { 1.0e6 };
            w.set_solver(solver);
            (w, snap.bodies.len(), name.replace(',', ";"))
        }
        s => anyhow::bail!("unknown scene {s}"),
    };
    let s = measure(&mut w, warm, steps)?;
    let c = w.counters();
    let h = hash(&w, std::env::var("RR_DUMP").ok());
    let label = std::env::var("RR_LABEL").unwrap_or_default();
    println!(
        "{label},{scene},{param},{bodies},{:.3},{:.3},{:.3},{},{},{},{h:016x}",
        pct(&s.wall, 0.5),
        pct(&s.wall, 0.95),
        pct(&s.gpu, 0.5),
        s.ran,
        c.map_or(0, |c| c.awake_count),
        c.map_or(0, |c| c.contact_count),
    );
    Ok(())
}
