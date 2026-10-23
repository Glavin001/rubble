//! Soft-body scale benchmark on the native wgpu host (the browser's WGSL).
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin soft_bench -- [cars …]   # default 1 10 50
//! ```
//! N procedural node–beam cars (102 particles, 572 beams, 4 raycast wheels each)
//! drive in a grid at 240 Hz × 10 iterations, all awake and deforming-capable.
//! Reports wall time per step and per 60 fps frame (4 steps), after warm-up.
use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::procedural::{car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_softbody::{soft_scene_params, SoftCapacity, SpawnOptions};
use engine_vehicle::{VehicleCapacity, VehicleInput};
use engine_webgpu::WgpuSim;

fn main() -> anyhow::Result<()> {
    let rigid = std::env::args().any(|a| a == "--rigid");
    let counts: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let counts = if counts.is_empty() {
        vec![1, 10, 50]
    } else {
        counts
    };
    let desc = engine_softbody::procedural::car_desc(&CarParams::default()).0;
    for &n in &counts {
        let cap = WorldCapacity {
            max_bodies: 49152,
            max_colliders: 98304,
            max_contacts: 524_288,
            max_joints: 4096,
            max_candidates: 1_048_576,
            max_colors: 32,
            max_commands: 64,
            max_hull_vertices: 64,
        };
        let soft = SoftCapacity {
            max_bodies: 256,
            max_beams: 262_144,
            max_particles: 49152,
            max_triangles: 131_072,
            ..Default::default()
        };
        let mut sim = WgpuSim::new_with_soft(
            cap,
            VehicleCapacity {
                max_vehicles: 256,
                max_wheels: 1024,
                ..Default::default()
            },
            soft,
        )?;
        let iters = std::env::var("ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let hz: f32 = std::env::var("HZ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(240.0);
        sim.set_params(soft_scene_params(hz, iters));
        let mut st = engine_softbody::SoftSettings::default();
        if let Ok(r) = std::env::var("CHEB") {
            st.chebyshev_rho = r.parse().unwrap_or(0.9);
        }
        if let Ok(r) = std::env::var("SWEEPS") {
            st.sweeps = r.parse().unwrap_or(1);
        }
        if std::env::var("CONTACTS").as_deref() == Ok("0") {
            st.contacts = false;
        }
        sim.set_soft_settings(st);
        sim.set_material(
            0,
            Material {
                friction: 1.0,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        sim.set_material(
            9,
            Material {
                friction: 0.8,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        let ground = sim.add_body(BodyState::static_plane_support());
        sim.add_collider(Collider {
            body: ground,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0, -5.0, 0.0],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [500.0, 5.0, 500.0, 707.0],
        });
        let cols = (n as f32).sqrt().ceil() as usize;
        let mut cores = Vec::new();
        for i in 0..n {
            let (x, z) = ((i % cols) as f32 * 6.0, (i / cols) as f32 * 9.0);
            if rigid {
                // Baseline: the rigid sedan preset, same count, no soft bodies.
                cores.push(
                    sim.add_vehicle(
                        &engine_vehicle::presets::sedan(),
                        [x, 1.0, z],
                        [0.0, 0.0, 0.0, 1.0],
                    )
                    .map_err(anyhow::Error::msg)?,
                );
                continue;
            }
            let h = sim
                .add_soft_vehicle(
                    &desc,
                    [x, 0.05, z],
                    [0.0, 0.0, 0.0, 1.0],
                    &SpawnOptions::default(),
                )
                .map_err(anyhow::Error::msg)?;
            cores.push(sim.soft_host().entry(h).core.unwrap().0);
        }
        for &c in &cores {
            sim.set_vehicle_input(
                c,
                VehicleInput {
                    forward: 0.5,
                    steer: 0.3,
                    ..Default::default()
                },
            );
        }
        for _ in 0..120 {
            sim.step()?;
        }
        let steps = 240;
        let t = std::time::Instant::now();
        for _ in 0..steps {
            sim.step()?;
        }
        let sync_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        // Pipelined, as the browser runs: submit without reading counters back.
        let last = sim.last_counters();
        sim.set_plan_hints(last.contact_count, last.color_count);
        let t = std::time::Instant::now();
        for _ in 0..steps {
            sim.step_submit()?;
        }
        sim.drain()?;
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        let c = sim.soft_host().counts();
        let rc = sim.last_counters();
        eprintln!(
            "  rigid colours {} awake {} contacts {}",
            rc.color_count, rc.awake_count, rc.contact_count
        );
        println!(
            "{}{n:>3} cars: {:>5} particles, {:>6} beams, {:>2} particle colours · pipelined {ms:5.2} ms/step ({:5.1} ms per 60 fps frame) · synchronous {sync_ms:5.2} ms/step · {}",
            if rigid { "rigid baseline " } else { "" },
            c.particles_total,
            c.beams,
            c.colors,
            ms * f64::from(hz) / 60.0,
            sim.chip()
        );
    }
    Ok(())
}
