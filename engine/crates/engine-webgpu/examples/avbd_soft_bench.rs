//! GPU cost of JBeam vehicles on Rubble's existing AVBD soft-body path
//! (`soft_step.slang`: particles in the World's AVBD step) at a World rate and
//! sweep count, for the AVBD vehicle study (`engine/docs/avbd-vehicles.md` §6).
//! N copies of the ETK (or the repo sedan) parked in a grid on a ground box;
//! each World step is submitted and waited for (`WgpuSim::step` reads its
//! counters back), so a step's wall time is its GPU time plus the submit and
//! readback latency (an upper bound, as `vehicle_scale_bench`).
//!
//! ```bash
//! perl -e 'alarm 600; exec @ARGV' cargo run --release -p engine-webgpu \
//!     --example avbd_soft_bench -- <cars> <hz> <iterations> [steps]
//! ```
//!
//! Environment: AB_CAR=etkk (default; `RUBBLE_ETK` its mod folder, default the
//! crash lab's `__mods/etkk_tex`, `AB_PC` its config, default kc6.pc) or sedan.
//! Takes the GPU lock (`default_device_with`). Prints one CSV line.
use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::{SoftCapacity, SoftVehicleDesc, SpawnOptions};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use std::time::Instant;

fn desc() -> SoftVehicleDesc {
    match std::env::var("AB_CAR").as_deref() {
        Ok("sedan") => {
            let text = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
            ))
            .expect("sedan");
            engine_softbody::truck::load_truck(&text)
        }
        _ => {
            let dir = std::env::var("RUBBLE_ETK")
                .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/__mods/etkk_tex").into());
            let dir = std::path::Path::new(&dir);
            let files = engine_softbody::jbeam::read_vehicle_dir(dir, "etkk").expect("etkk");
            let refs: Vec<(&str, &str)> = files
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let pc_name = std::env::var("AB_PC").unwrap_or_else(|_| "kc6.pc".into());
            let pc = std::fs::read_to_string(dir.join("etkk").join(pc_name)).expect("pc");
            engine_softbody::jbeam::load_jbeam(&refs, Some(&pc)).expect("load")
        }
    }
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cars: u32 = a.first().map_or(1, |x| x.parse().unwrap());
    let hz: f32 = a.get(1).map_or(120.0, |x| x.parse().unwrap());
    let iters: u32 = a.get(2).map_or(10, |x| x.parse().unwrap());
    let steps: usize = a.get(3).map_or(120, |x| x.parse().unwrap());
    let d = desc();
    let nodes = d.nodes.len() as u32;
    let beams = d.beams.len() as u32;
    let cap = WorldCapacity {
        max_bodies: (cars * (nodes + 8) + 64).max(4096),
        max_colliders: (cars * (nodes + 8) + 64).max(8192),
        max_contacts: (cars * nodes * 8).max(65_536),
        max_joints: 512 + cars * 64,
        max_candidates: (cars * nodes * 16).max(131_072),
        max_colors: 64,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let soft = SoftCapacity {
        max_bodies: cars.max(1),
        max_beams: (cars * beams).max(65_536),
        max_particles: (cars * nodes).max(16_384),
        max_triangles: (cars * d.triangles.len() as u32 * 2).max(32_768),
        max_elements: (cars * 512).max(8192),
        max_rail_nodes: (cars * 512).max(8192),
        max_groups: (cars * 256).max(4096),
        ..SoftCapacity::default()
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), soft)?;
    sim.set_params(engine_softbody::soft_scene_params(hz, iters));
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    let ground = sim.add_body(BodyState {
        position: [0.0, -5.0, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0, -5.0, 0.0],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [2000.0, 5.0, 2000.0, 2000.0],
    });
    let low = d
        .nodes
        .iter()
        .map(|n| n.position[1])
        .fold(f32::MAX, f32::min);
    let side = (cars as f32).sqrt().ceil() as u32;
    let opts = SpawnOptions {
        rigid_core: false,
        ..Default::default()
    };
    let mut handles = Vec::new();
    let t_spawn = Instant::now();
    for k in 0..cars {
        let (gx, gz) = (k % side, k / side);
        let h = sim
            .add_soft_vehicle(
                &d,
                [gx as f32 * 6.0, 0.05 - low, gz as f32 * 9.0],
                [0.0, 0.0, 0.0, 1.0],
                &opts,
            )
            .map_err(anyhow::Error::msg)?;
        handles.push(h);
    }
    let spawn_s = t_spawn.elapsed().as_secs_f64();
    let warm = 10.min(steps);
    for _ in 0..warm {
        sim.step()?;
    }
    let mut ms = Vec::with_capacity(steps);
    for _ in 0..steps {
        let t = Instant::now();
        sim.step()?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let finite = handles.iter().all(|&h| {
        sim.read_soft_particles(h)
            .iter()
            .all(|b| b.position.iter().all(|x| x.is_finite()))
    });
    ms.sort_by(f64::total_cmp);
    let q = |p: f64| ms[((ms.len() - 1) as f64 * p) as usize];
    println!("car,cars,nodes,beams,hz,iterations,steps,p50_ms,p95_ms,p99_ms,finite,spawn_s");
    println!(
        "{},{cars},{nodes},{beams},{hz},{iters},{steps},{:.3},{:.3},{:.3},{finite},{spawn_s:.1}",
        std::env::var("AB_CAR").unwrap_or_else(|_| "etkk".into()),
        q(0.5),
        q(0.95),
        q(0.99)
    );
    Ok(())
}
