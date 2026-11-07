//! Rigid-only gate: the demo's rigid scenes in a World with no vehicles, the same on any
//! build (only API present on origin/main is used), for an A/B of two builds.
//!
//! ```bash
//! # hash of every body (bytes of `download_bodies`) at checkpoints up to STEPS steps
//! GATE_MODE=hash cargo run --release -p engine-world --example rigid_gate
//! # ms per step (step + poll_blocking), median of STEPS after WARM, per scene
//! GATE_MODE=time GATE_SCENES="Brick Ring (110k)" cargo run --release -p engine-world --example rigid_gate
//! ```
//!
//! `GATE_SCENES` (comma-separated scene names), `GATE_SLEEP=on|off` (default on), `STEPS`,
//! `WARM`. A ball is fired into the scene at step `STEPS / 3` so the hash covers impacts.
//! Native wgpu is not bit-reproducible run to run once a scene's contacts pile up (atomic
//! ordering), so the hash mode prints a hash at each checkpoint step: compare two builds
//! up to the step where two runs of the same build still agree. (The convex scenes are
//! left out: a snapshot does not carry hull data.)
use rubble::*;

const SCENES: [&str; 11] = [
    "Brick Ring (28k)",
    "Brick Ring (110k)",
    "Brick Walls (27k)",
    "Wall Smash (2k)",
    "Chain Mail (1.6k)",
    "Pyramid 30",
    "Box Pile (32k)",
    "Ragdoll Pile (10k)",
    "Hinge Chains (10k)",
    "Sphere Pile (10k)",
    "100k Grid",
];

/// The hash mode's checkpoint steps (and the last step).
const CHECKS: [u32; 12] = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300];

fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn world(name: &str, sleep: bool) -> World {
    // The scene's own capacity (the demo's), with room for the ball.
    let (host, info) = engine_scenes::build_host(name).unwrap();
    let snap = host.snapshot();
    let c = info.capacity.clone();
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: c.max_bodies.max(snap.bodies.len() as u32 + 8),
            max_colliders: c.max_colliders + 8,
            ..c
        },
        enable_sleep: sleep,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.import_snapshot(&snap).unwrap();
    let mut solver = w.solver();
    solver.sleep_time = if sleep { 0.5 } else { 1.0e6 };
    w.set_solver(solver);
    w
}

fn ball(w: &mut World) {
    let b = w
        .create_body(&BodyDef {
            linear_velocity: [30.0, -2.0, 0.0],
            is_bullet: true,
            ..BodyDef::dynamic([-40.0, 6.0, 0.0])
        })
        .unwrap();
    let r = 1.0;
    let d = ShapeDef {
        density: 2000.0 / (4.0 / 3.0 * std::f32::consts::PI * r * r * r),
        ..ShapeDef::default()
    };
    w.create_sphere_shape(b, &d, [0.0; 3], r).unwrap();
}

/// FNV-1a 64 of `bytes`.
fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn main() {
    let mode = std::env::var("GATE_MODE").unwrap_or_else(|_| "hash".into());
    let sleep = std::env::var("GATE_SLEEP").map_or(true, |v| v != "off");
    let scenes: Vec<String> = match std::env::var("GATE_SCENES") {
        Ok(s) if !s.is_empty() => s.split(',').map(|x| x.trim().to_string()).collect(),
        _ => SCENES.iter().map(|s| s.to_string()).collect(),
    };
    let on_off = if sleep { "on" } else { "off" };
    for name in &scenes {
        let mut w = world(name, sleep);
        if mode == "time" {
            let (warm, steps) = (env_u32("WARM", 60), env_u32("STEPS", 300));
            for _ in 0..warm {
                w.step().unwrap();
                w.poll_blocking().unwrap();
            }
            let mut ms = Vec::with_capacity(steps as usize);
            for _ in 0..steps {
                let t = std::time::Instant::now();
                w.step().unwrap();
                w.poll_blocking().unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
            let awake = w.counters().map_or(0, |c| c.awake_count);
            println!(
                "time\t{name}\tsleep={on_off}\tmedian_ms={:.3}\tp10={:.3}\tp90={:.3}\tawake={awake}",
                q(0.5),
                q(0.1),
                q(0.9)
            );
        } else {
            let steps = env_u32("STEPS", 300);
            let mut line = String::new();
            for s in 1..=steps {
                if s == steps / 3 {
                    ball(&mut w);
                }
                w.step().unwrap();
                w.poll_blocking().unwrap();
                if CHECKS.contains(&s) || s == steps {
                    let bodies = w.sim().download_bodies();
                    line += &format!("\t{s}:{:016x}", fnv(bytemuck::cast_slice(&bodies)));
                }
            }
            let c = w.counters();
            println!(
                "hash\t{name}\tsleep={on_off}\tbodies={}\tawake={}\tcontacts={}{line}",
                w.sim().download_bodies().len(),
                c.map_or(0, |c| c.awake_count),
                c.map_or(0, |c| c.contact_count),
            );
        }
    }
}
