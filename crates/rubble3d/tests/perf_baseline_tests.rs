//! M0 perf harness: fixed deterministic scenes, per-phase `StepTimingsMs` medians,
//! JSON report + comparison against committed baselines.
//!
//! Run explicitly (ignored by default — wall-clock timing is meaningless in the
//! normal parallel test lane):
//!
//! ```bash
//! cargo test -p rubble3d --test perf_baseline_tests --release -- --ignored --nocapture
//! ```
//!
//! Behavior:
//! - Always writes `target/perf-report.json` (workspace target dir) and prints a table.
//! - If `tests/perf/baselines.json` exists, prints per-scene deltas vs baseline.
//! - `RUBBLE_PERF_UPDATE=1` rewrites the committed baselines file from this run.
//! - `RUBBLE_PERF_ENFORCE=1` fails the test when a scene's median total step time
//!   regresses more than `ENFORCE_FACTOR`× the baseline (generous: CI runners are noisy).

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

const WARMUP_STEPS: usize = 20;
const MEASURE_STEPS: usize = 60;
const ENFORCE_FACTOR: f32 = 2.0;

// ---------------------------------------------------------------------------
// Deterministic scenes (no rand: perf scenes must be bit-identical run to run)
// ---------------------------------------------------------------------------

/// SplitMix64-derived unit float in [0, 1). Deterministic jitter without `rand`.
fn hash_unit(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

fn ground() -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }
}

fn box_body(pos: Vec3, half: f32) -> RigidBodyDesc {
    RigidBodyDesc {
        position: pos,
        rotation: Quat::IDENTITY,
        mass: 1.0,
        friction: 0.5,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(half),
        },
        ..Default::default()
    }
}

/// N×NY×N grid of resting boxes over a ground plane, tiny deterministic jitter.
fn scene_grid(nx: usize, ny: usize, nz: usize) -> Vec<RigidBodyDesc> {
    let mut descs = vec![ground()];
    let side = 0.42_f32;
    let pitch = side + 0.08;
    let half = side * 0.5;
    let ox = -((nx - 1) as f32 * pitch) * 0.5;
    let oz = -((nz - 1) as f32 * pitch) * 0.5;
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                let seed = (j * nx * nz + i * nz + k) as u64;
                let jit = |s: u64| (hash_unit(seed * 3 + s) - 0.5) * 0.012;
                descs.push(box_body(
                    Vec3::new(
                        ox + i as f32 * pitch + jit(0),
                        half + 0.03 + j as f32 * pitch + jit(1),
                        oz + k as f32 * pitch + jit(2),
                    ),
                    half,
                ));
            }
        }
    }
    descs
}

/// Boxes dropped from a deterministic cloud into a pile.
fn scene_pile(count: usize) -> Vec<RigidBodyDesc> {
    let mut descs = vec![ground()];
    for i in 0..count {
        let s = i as u64;
        descs.push(box_body(
            Vec3::new(
                (hash_unit(s * 7 + 1) - 0.5) * 14.0,
                2.0 + hash_unit(s * 7 + 2) * 18.0,
                (hash_unit(s * 7 + 3) - 0.5) * 14.0,
            ),
            0.15 + hash_unit(s * 7 + 4) * 0.25,
        ));
    }
    descs
}

/// 10-box vertical stack on a wide static box (mirrors rubble-scenes `scene_stack`).
fn scene_stack() -> Vec<RigidBodyDesc> {
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        },
        ..Default::default()
    }];
    for i in 0..10 {
        descs.push(box_body(Vec3::new(0.0, i as f32 * 1.5 + 1.0, 0.0), 0.5));
    }
    descs
}

/// 16-row 2D pyramid of boxes (mirrors rubble-scenes `scene_pyramid`).
fn scene_pyramid() -> Vec<RigidBodyDesc> {
    const SIZE: usize = 16;
    let mut descs = vec![RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        },
        ..Default::default()
    }];
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            descs.push(RigidBodyDesc {
                position: Vec3::new(
                    x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                    y as f32 * 0.85 + 0.5,
                    0.0,
                ),
                mass: 1.0,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(0.5, 0.25, 0.25) * 2.0 * 0.5,
                },
                ..Default::default()
            });
        }
    }
    descs
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

const PHASES: [&str; 7] = [
    "upload",
    "predict",
    "broadphase",
    "narrowphase",
    "contact_fetch",
    "solve",
    "extract",
];

#[derive(Clone, Debug)]
struct SceneResult {
    name: String,
    bodies: usize,
    contacts: usize,
    /// Median per-phase ms over the measurement window, order matches `PHASES`.
    phase_ms: [f32; 7],
    total_ms: f32,
}

fn median(samples: &mut [f32]) -> f32 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if samples.is_empty() {
        0.0
    } else {
        samples[samples.len() / 2]
    }
}

fn measure_scene(name: &str, descs: &[RigidBodyDesc], iterations: u32) -> Option<SceneResult> {
    let config = SimConfig {
        solver_iterations: iterations,
        max_bodies: (descs.len() + 64).next_power_of_two(),
        ..Default::default()
    };
    let mut world = match World::new(config) {
        Ok(w) => w,
        Err(_) => {
            eprintln!("SKIP: no GPU adapter");
            return None;
        }
    };
    for d in descs {
        world.add_body(d);
    }

    for _ in 0..WARMUP_STEPS {
        world.step();
    }

    let mut samples: Vec<[f32; 7]> = Vec::with_capacity(MEASURE_STEPS);
    for _ in 0..MEASURE_STEPS {
        world.step();
        samples.push(world.last_step_timings().as_array());
    }

    let mut phase_ms = [0.0f32; 7];
    for (p, slot) in phase_ms.iter_mut().enumerate() {
        let mut col: Vec<f32> = samples.iter().map(|s| s[p]).collect();
        *slot = median(&mut col);
    }
    let mut totals: Vec<f32> = samples.iter().map(|s| s.iter().sum()).collect();
    let total_ms = median(&mut totals);

    Some(SceneResult {
        name: name.to_string(),
        bodies: descs.len(),
        contacts: world.fetch_last_contacts().len(),
        phase_ms,
        total_ms,
    })
}

// ---------------------------------------------------------------------------
// JSON (hand-rolled: flat schema, avoids a serde_json dev-dependency)
// ---------------------------------------------------------------------------

fn to_json(results: &[SceneResult]) -> String {
    let mut out = String::from("{\n  \"scenes\": {\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "    \"{}\": {{ \"bodies\": {}, \"contacts\": {}, \"total_ms\": {:.3}",
            r.name, r.bodies, r.contacts, r.total_ms
        ));
        for (p, label) in PHASES.iter().enumerate() {
            out.push_str(&format!(", \"{}_ms\": {:.3}", label, r.phase_ms[p]));
        }
        out.push_str(" }");
        out.push_str(if i + 1 < results.len() { ",\n" } else { "\n" });
    }
    out.push_str("  }\n}\n");
    out
}

/// Minimal extraction of `"scene": { ... "total_ms": X ... }` pairs from the
/// baseline file. Only `total_ms` is enforced; full JSON parsing is unnecessary.
fn parse_baseline_totals(json: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    // Scan line-by-line: `"name": { ... "total_ms": <v>`
    for line in json.lines() {
        let t = line.trim();
        if !t.starts_with('"') {
            continue;
        }
        let Some(name_end) = t[1..].find('"') else {
            continue;
        };
        let name = &t[1..1 + name_end];
        let Some(tm_pos) = t.find("\"total_ms\":") else {
            continue;
        };
        let rest = &t[tm_pos + 11..];
        let num: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();
        if let Ok(v) = num.parse::<f32>() {
            out.push((name.to_string(), v));
        }
    }
    out
}

fn baselines_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/perf/baselines.json")
}

fn report_path() -> std::path::PathBuf {
    // target/ two levels up from the crate manifest (workspace layout).
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/perf-report.json")
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

#[test]
#[ignore = "perf harness: run explicitly with --ignored (timing-sensitive)"]
fn perf_baseline_harness() {
    let iterations = SimConfig::default().solver_iterations;
    let scenes: Vec<(&str, Vec<RigidBodyDesc>)> = vec![
        ("stack_10", scene_stack()),
        ("pyramid_16", scene_pyramid()),
        ("grid_256", scene_grid(8, 4, 8)),
        ("grid_1k", scene_grid(8, 16, 8)),
        ("grid_4k", scene_grid(16, 16, 16)),
        ("pile_1k", scene_pile(1024)),
    ];

    let mut results = Vec::new();
    for (name, descs) in &scenes {
        let Some(r) = measure_scene(name, descs, iterations) else {
            return; // no GPU adapter: skip quietly like the rest of the suite
        };
        println!(
            "{:<12} bodies={:<6} contacts={:<6} total={:>8.3} ms  [{}]",
            r.name,
            r.bodies,
            r.contacts,
            r.total_ms,
            PHASES
                .iter()
                .zip(r.phase_ms.iter())
                .map(|(l, v)| format!("{l}={v:.2}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        results.push(r);
    }

    // Always write the machine-readable report.
    let json = to_json(&results);
    let report = report_path();
    if let Some(dir) = report.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&report, &json).expect("write perf report");
    println!("wrote {}", report.display());

    // Optionally refresh the committed baselines.
    if std::env::var("RUBBLE_PERF_UPDATE").as_deref() == Ok("1") {
        let path = baselines_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &json).expect("write baselines");
        println!("updated {}", path.display());
        return;
    }

    // Compare against committed baselines when present.
    let Ok(baseline_json) = std::fs::read_to_string(baselines_path()) else {
        println!("no committed baselines (tests/perf/baselines.json); run with RUBBLE_PERF_UPDATE=1 to create");
        return;
    };
    let baselines = parse_baseline_totals(&baseline_json);
    let enforce = std::env::var("RUBBLE_PERF_ENFORCE").as_deref() == Ok("1");
    let mut regressions = Vec::new();
    for r in &results {
        if let Some((_, base)) = baselines.iter().find(|(n, _)| n == &r.name) {
            let ratio = if *base > 0.0 { r.total_ms / base } else { 0.0 };
            println!(
                "{:<12} total {:>8.3} ms vs baseline {:>8.3} ms  ({:>5.2}x)",
                r.name, r.total_ms, base, ratio
            );
            if ratio > ENFORCE_FACTOR {
                regressions.push(format!(
                    "{}: {:.3} ms vs baseline {:.3} ms ({:.2}x > {:.1}x)",
                    r.name, r.total_ms, base, ratio, ENFORCE_FACTOR
                ));
            }
        }
    }
    if enforce {
        assert!(
            regressions.is_empty(),
            "perf regressions:\n{}",
            regressions.join("\n")
        );
    } else if !regressions.is_empty() {
        println!(
            "perf regressions (non-enforcing):\n{}",
            regressions.join("\n")
        );
    }
}
