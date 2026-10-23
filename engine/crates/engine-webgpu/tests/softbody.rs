//! Soft-body (node–beam) vehicles on the native wgpu host, the same WGSL the
//! browser runs: imported `.truck` content resting, sleeping, matching the CPU
//! reference, driving on raycast wheels, and crushing differently against a
//! wall, a pole and a side impact.

use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::cpu::CpuSoft;
use engine_softbody::procedural::{box_truck, car_truck, CarParams};
use engine_softbody::truck::load_truck;
use engine_softbody::{SoftCapacity, SoftHandle, SoftSettings, SoftVehicleDesc, SpawnOptions};
use engine_vehicle::{VehicleCapacity, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::Vec3;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn static_body(sim: &mut WgpuSim, pos: [f32; 3]) -> u32 {
    sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    })
}

fn add_box_collider(sim: &mut WgpuSim, body: u32, half: [f32; 3]) {
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()],
    });
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3]) -> u32 {
    let b = static_body(sim, pos);
    add_box_collider(sim, b, half);
    b
}

/// Flat ground (top at y = 0), soft arena, 120 Hz.
fn world(hz: f32, iterations: u32) -> WgpuSim {
    let mut sim = WgpuSim::new_with_soft(capacity(), VehicleCapacity::default(), SoftCapacity::default()).expect("wgpu");
    sim.set_params(engine_softbody::soft_scene_params(hz, iterations));
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(9, Material { friction: 0.8, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    static_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0]);
    sim
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h).iter().map(|b| Vec3::from_array(b.position)).collect()
}

fn kinetic(sim: &WgpuSim, h: SoftHandle) -> f32 {
    sim.read_soft_particles(h)
        .iter()
        .filter(|b| b.inv_mass > 0.0)
        .map(|b| 0.5 / b.inv_mass * Vec3::from_array(b.linear_vel).length_squared())
        .sum()
}

fn car_desc() -> SoftVehicleDesc {
    load_truck(&car_truck(&CarParams::default()))
}

#[test]
fn soft_box_lands_rests_and_sleeps() {
    let mut sim = world(120.0, 10);
    let d = load_truck(&box_truck(3, 3, 3, 0.4, 200.0, 2.0e6, 4.0e5, 4.0e6));
    let opts = SpawnOptions { rigid_core: false, ..Default::default() };
    let h = sim.add_soft_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0], &opts).expect("spawn");
    let rest: Vec<Vec3> = positions(&sim, h);
    run(&mut sim, 240);
    let p = positions(&sim, h);
    assert!(p.iter().all(|x| x.is_finite()), "finite");
    let lowest = p.iter().map(|x| x.y).fold(f32::MAX, f32::min);
    assert!(lowest > 0.0 && lowest < 0.12, "box rests on the ground: lowest node {lowest}");
    // Shape preserved: edge lengths within 2 % of rest (elastic sag only).
    let beams = sim.read_soft_beams(h);
    let mut worst = 0.0f32;
    let first = sim.soft_host().entry(h).first_body;
    for b in &beams {
        let l = (p[(b.a - first) as usize] - p[(b.b - first) as usize]).length();
        let l0 = (rest[(b.a - first) as usize] - rest[(b.b - first) as usize]).length();
        worst = worst.max((l - l0).abs() / l0);
    }
    assert!(worst < 0.02, "max strain at rest {worst}");
    assert!(beams.iter().all(|b| !b.broken()), "no breaks at rest");
    let yielded = beams.iter().filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0).count();
    assert_eq!(yielded, 0, "a gentle landing does not yield");
    // Falls asleep within 2 s of stopping.
    run(&mut sim, 240);
    let asleep = sim.read_soft_particles(h).iter().all(|b| b.flags & 4 != 0);
    assert!(asleep, "soft body sleeps as a whole once settled");
}

/// The GPU kernels against the CPU reference on a contact-free clamped cantilever:
/// the static sag both converge to must agree.
#[test]
fn cantilever_sag_matches_cpu_reference() {
    let mut d = load_truck(&box_truck(6, 2, 2, 0.4, 400.0, 3.0e6, f32::INFINITY, f32::INFINITY));
    for n in &mut d.nodes {
        if n.position[0] < 0.01 {
            n.fixed = true;
        }
        n.collision = false;
    }
    let hz = 120.0;
    let iters = 40;
    let mut sim = world(hz, iters);
    sim.set_soft_settings(SoftSettings { plasticity: false, breaking: false, sleep_sync: false, chebyshev_rho: 0.0, ..Default::default() });
    let mut p = engine_softbody::soft_scene_params(hz, iters);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    let opts = SpawnOptions { rigid_core: false, ..Default::default() };
    let h = sim.add_soft_vehicle(&d, [0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0], &opts).expect("spawn");
    let rest = positions(&sim, h);
    run(&mut sim, (hz * 5.0) as u32);
    let gpu = positions(&sim, h);
    let mut cpu = CpuSoft::from_desc(&d, Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
    cpu.h = 1.0 / hz;
    cpu.iterations = iters;
    cpu.settings.plasticity = false;
    cpu.settings.breaking = false;
    for _ in 0..(hz * 5.0) as u32 {
        cpu.step();
    }
    let tip = |x: &[Vec3]| x.iter().zip(&rest).filter(|(_, r)| r.x > 1.9).map(|(p, r)| r.y - p.y).sum::<f32>() / 4.0;
    let (g, c) = (tip(&gpu), tip(&cpu.x));
    eprintln!("tip sag: gpu {g:.5} m, cpu {c:.5} m");
    assert!(g > 0.0 && c > 0.0);
    assert!((g - c).abs() < 0.1 * c, "GPU and CPU agree within 10%: {g} vs {c}");
}

#[test]
#[ignore]
fn debug_box_drop() {
    let mut sim = world(120.0, 10);
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let mut p = engine_softbody::soft_scene_params(120.0, 10);
    p.k_start = env("KSTART", p.k_start);
    p.beta_lin = env("BETA", p.beta_lin);
    p.alpha = env("ALPHA", p.alpha);
    p.flags = env("FLAGS", 0.0) as u32;
    sim.set_params(p);
    let d = load_truck(&box_truck(3, 3, 3, 0.4, 200.0, 2.0e6, 4.0e5, 4.0e6));
    let opts = SpawnOptions { rigid_core: false, ..Default::default() };
    let h = sim.add_soft_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0], &opts).expect("spawn");
    eprintln!("counts {:?}", sim.soft_host().counts());
    use engine_softbody::SoftBackend;
    let hdr = sim.read_soft_slots(0, 8);
    eprintln!("header {hdr:?}");
    let l = SoftCapacity::default().layout();
    eprintln!("colors {:?}", sim.read_soft_slots(l.color_base, 4));
    eprintln!("list {:?}", sim.read_soft_slots(l.list_base, 4));
    for k in 0..240 {
        sim.step().unwrap();
        if k % 10 != 0 { continue; }
        let p = sim.read_soft_particles(h);
        let c = sim.last_counters();
        eprintln!("step {k}: y0 {:.4} vy {:.3} flags {} timer {:.3} | contacts {} awake {} colors {}", p[0].position[1], p[0].linear_vel[1], p[0].flags, p[0].sleep_timer, c.contact_count, c.awake_count, c.color_count);
    }
}
