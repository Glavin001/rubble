//! Probe (ignored): the per-step cost floor of a large sleeping world. `FLOOR_SCENE` (an
//! engine-scenes name, default "100k Grid") imported as the demo does, sleep on (0.5 s),
//! stepped `FLOOR_STEPS` times (default 600); with `RUBBLE_GPU_TIMING=1
//! RUBBLE_GPU_TIMING_DISPATCH=1 RUBBLE_GPU_TIMING_PRINT=1` every step prints each
//! dispatch's GPU ms. `FLOOR_WAKE=n`: from step 300 on, n boxes (evenly spaced) are kept
//! awake by a small push each step, so the floor can be read with some bodies awake.
//! `FLOOR_FEATURES_ENV=1` takes the features from `ENGINE_*`.

use rubble::*;

fn env_u(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[test]
#[ignore]
fn floor_probe() {
    let scene = std::env::var("FLOOR_SCENE").unwrap_or_else(|_| "100k Grid".into());
    let (host, _) = engine_scenes::build_host(&scene).unwrap();
    let snap = host.snapshot();
    let n = snap.bodies.len() as u32 + 4;
    let mut def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: n.next_power_of_two(),
            max_colliders: n.next_power_of_two(),
            max_contacts: (16 * n).next_power_of_two().min(1 << 23),
            max_candidates: (16 * n).next_power_of_two().min(1 << 23),
            max_colors: 32,
            ..WorldCapacity::small()
        },
        ..WorldDef::default()
    };
    if std::env::var("FLOOR_FEATURES_ENV").is_ok() {
        def.features = engine_webgpu::EngineFeatures::from_env();
    }
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let imported = w.import_snapshot(&snap).unwrap();
    let mut solver = w.solver();
    solver.sleep_time = 0.5;
    w.set_solver(solver);
    let bodies: Vec<BodyId> = imported.bodies.iter().copied().skip(1).collect();
    let wake = env_u("FLOOR_WAKE", 0) as usize;
    let steps = env_u("FLOOR_STEPS", 600);
    for s in 1..=steps {
        if wake > 0 && s >= 300 {
            let stride = (bodies.len() / wake).max(1);
            for k in 0..wake {
                let b = bodies[(k * stride).min(bodies.len() - 1)];
                w.body_set_linear_velocity(b, [0.0, 0.3, 0.0]).unwrap();
            }
        }
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if s % 100 == 0 {
            let c = w.counters().unwrap();
            eprintln!("floor step {s}: awake {} contacts {}", c.awake_count, c.contact_count);
        }
    }
}
