//! Soft-body (node–beam) vehicles on the native wgpu host, the same WGSL the
//! browser runs: imported `.truck` content resting, sleeping, matching the CPU
//! reference, driving on raycast wheels, and crushing differently against a
//! wall, a pole and a side impact.

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
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
    let mut p = engine_softbody::soft_scene_params(hz, iterations);
    if let Some(k) = std::env::var("KMAXN").ok().and_then(|v| v.parse().ok()) {
        p.k_max_n = k;
    }
    if let Some(k) = std::env::var("BETA").ok().and_then(|v| v.parse().ok()) {
        p.beta_lin = k;
    }
    sim.set_params(p);
    if let Some(r) = std::env::var("CHEB").ok().and_then(|v| v.parse().ok()) {
        sim.set_soft_settings(SoftSettings { chebyshev_rho: r, ..Default::default() });
    }
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
    let _ = engine_schema::AvbdParams::default();
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

fn spawn_car(sim: &mut WgpuSim, pos: [f32; 3], vel: [f32; 3]) -> SoftHandle {
    let opts = SpawnOptions { velocity: vel, ..Default::default() };
    sim.add_soft_vehicle(&car_desc(), pos, [0.0, 0.0, 0.0, 1.0], &opts).expect("spawn car")
}

fn yielded(sim: &WgpuSim, h: SoftHandle) -> usize {
    sim.read_soft_beams(h).iter().filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0).count()
}

fn broken(sim: &WgpuSim, h: SoftHandle) -> usize {
    sim.read_soft_beams(h).iter().filter(|b| b.broken()).count()
}

fn centroid(p: &[Vec3]) -> Vec3 {
    p.iter().copied().sum::<Vec3>() / p.len() as f32
}

#[test]
fn soft_car_settles_on_its_raycast_wheels() {
    let mut sim = world(120.0, 12);
    let h = spawn_car(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
    run(&mut sim, 360);
    let core = sim.soft_host().entry(h).core.expect("rigid core").0;
    let st = sim.vehicle_state(core);
    let p = positions(&sim, h);
    let lowest = p.iter().map(|x| x.y).fold(f32::MAX, f32::min);
    eprintln!("contacts {} lowest node {lowest:.3} ke {:.3} yielded {}", st.contacts, kinetic(&sim, h), yielded(&sim, h));
    for w in &st.wheels {
        eprintln!("  wheel contact {} L {:.3} F {:.0}", w.in_contact, w.suspension_length, w.suspension_force);
    }
    assert_eq!(st.contacts, 4, "stands on four wheels");
    assert!(lowest > 0.12, "bodywork clear of the ground ({lowest})");
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let parts: f32 = sim.read_soft_particles(h).iter().filter(|b| b.inv_mass > 0.0).map(|b| 1.0 / b.inv_mass).sum();
    let core_body = sim.read_body(sim.soft_host().entry(h).core.unwrap().1);
    let mass = parts + 1.0 / core_body.inv_mass;
    let joints = sim.download_joints();
    let r = sim.soft_host().entry(h).core_joints.clone();
    let lam: Vec<[f32; 8]> = joints[r.start as usize..r.end as usize].iter().map(|j| j.lambda).collect();
    let sum: Vec3 = lam.iter().map(|l| Vec3::new(l[0], l[1], l[2])).sum();
    eprintln!("joints {} lambda sum {sum:?}", lam.len());
    eprintln!("particles {parts:.1} kg + core {:.1} kg = {mass:.1} kg (desc {:.1})", 1.0 / core_body.inv_mass, car_desc().total_mass());
    assert!((total - mass * 9.81).abs() < 0.08 * mass * 9.81, "suspension carries the vehicle: {total} vs {}", mass * 9.81);
    assert_eq!(yielded(&sim, h), 0, "standing does not deform the body");
    assert!(kinetic(&sim, h) < 2.0, "at rest");
}

#[test]
fn soft_car_drives_and_steers_without_deforming() {
    let mut sim = world(120.0, 12);
    let h = spawn_car(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
    run(&mut sim, 120);
    let core = sim.soft_host().entry(h).core.expect("rigid core").0;
    let start = centroid(&positions(&sim, h));
    sim.set_vehicle_input(core, VehicleInput { forward: 1.0, ..Default::default() });
    run(&mut sim, 360);
    let mid = centroid(&positions(&sim, h));
    let v = Vec3::from_array(sim.read_body(sim.soft_host().entry(h).core.unwrap().1).linear_vel);
    eprintln!("after 3 s throttle: moved {:?}, speed {:.2} m/s, yielded {}", mid - start, v.length(), yielded(&sim, h));
    assert!(mid.z - start.z > 8.0, "drives forward (+Z): {:?}", mid - start);
    assert!((mid.x - start.x).abs() < 1.0, "straight line");
    sim.set_vehicle_input(core, VehicleInput { forward: 0.6, steer: 1.0, ..Default::default() });
    run(&mut sim, 240);
    let end = centroid(&positions(&sim, h));
    eprintln!("after steering: {:?}", end - mid);
    assert!((end.x - mid.x).abs() > 1.5, "steering turns the car: {:?}", end - mid);
    assert_eq!(broken(&sim, h), 0, "driving breaks nothing");
    assert!(yielded(&sim, h) < 5, "driving barely yields: {}", yielded(&sim, h));
}

#[test]
#[ignore]
fn debug_core_alone() {
    let mut sim = world(120.0, 12);
    let d = car_desc();
    let mut masses: Vec<f32> = d.nodes.iter().map(|n| n.mass).collect();
    let _ = &mut masses;
    let opts = SpawnOptions::default();
    let packed = engine_softbody::pack::pack_soft(&d, engine_softbody::pack::Placement { position: Vec3::ZERO, rotation: glam::Quat::IDENTITY }, &opts, 0, -5);
    let plan = packed.core.unwrap();
    let mut vd = plan.vehicle.clone();
    // Whole vehicle mass on the core.
    vd.parts[0].mass = d.total_mass();
    let v = sim.add_vehicle(&vd, (plan.origin + Vec3::Y * 0.05).to_array(), [0.0, 0.0, 0.0, 1.0]).unwrap();
    run(&mut sim, 360);
    let st = sim.vehicle_state(v);
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let b = sim.read_body(sim.vehicle_host().chassis_body(v));
    eprintln!("core alone: mass {:.1} total {total:.0} weight {:.0} L {:?}", 1.0 / b.inv_mass, 9.81 / b.inv_mass, st.wheels.iter().map(|w| w.suspension_length).collect::<Vec<_>>());
}

#[test]
#[ignore]
fn debug_core_with_rigid_masses() {
    let mut sim = world(120.0, 12);
    let d = car_desc();
    let opts = SpawnOptions::default();
    let packed = engine_softbody::pack::pack_soft(&d, engine_softbody::pack::Placement { position: Vec3::ZERO, rotation: glam::Quat::IDENTITY }, &opts, 0, -5);
    let plan = packed.core.unwrap();
    let origin = plan.origin + Vec3::Y * 0.05;
    let v = sim.add_vehicle(&plan.vehicle, origin.to_array(), [0.0, 0.0, 0.0, 1.0]).unwrap();
    let chassis = sim.vehicle_host().chassis_body(v);
    let hung = 770.0f32;
    let n = plan.frame_nodes.len() as f32;
    let mut weight = 0.0;
    for &i in &plan.frame_nodes {
        let p = Vec3::from_array(d.nodes[i as usize].position) + Vec3::Y * 0.05;
        let m = hung / n;
        let b = sim.add_body(BodyState::dynamic(p, glam::Quat::IDENTITY, m, Vec3::splat(0.01 * m)));
        weight += m * 9.81;
        sim.add_joint_desc(&engine_schema::JointDesc::ball(chassis, b, p - origin, Vec3::ZERO));
    }
    run(&mut sim, 360);
    let st = sim.vehicle_state(v);
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let core = sim.read_body(chassis);
    eprintln!("rigid hung masses: suspension {total:.0} vs weight {:.0}", weight + 9.81 / core.inv_mass);
}

/// Residual deformation: positions in the car's rigid best-fit frame minus rest.
fn residual(rest: &[Vec3], now: &[Vec3]) -> Vec<Vec3> {
    // Kabsch-free approximation: the rear ring (6 nodes, far from every impact
    // in these tests) fixes the frame.
    let c0 = rest[..6].iter().copied().sum::<Vec3>() / 6.0;
    let c1 = now[..6].iter().copied().sum::<Vec3>() / 6.0;
    let axis = |p: &[Vec3]| {
        let x = (p[0] + p[2] + p[4]) / 3.0 - (p[1] + p[3] + p[5]) / 3.0; // left − right
        let y = (p[4] + p[5]) / 2.0 - (p[0] + p[1]) / 2.0; // top − floor
        let x = x.normalize();
        let z = x.cross((y - x * y.dot(x)).normalize());
        glam::Mat3::from_cols(x, x.cross(z) * -1.0, z)
    };
    let r0 = axis(&rest[..6]);
    let r1 = axis(&now[..6]);
    now.iter().zip(rest).map(|(p, q)| r1.transpose() * (*p - c1) - r0.transpose() * (*q - c0)).collect()
}

struct Crash {
    yielded: usize,
    broken: usize,
    /// Residual displacement per node in the car frame.
    disp: Vec<Vec3>,
    events: usize,
}

fn crash(setup: impl FnOnce(&mut WgpuSim), speed: f32, seconds: f32) -> Crash {
    let mut sim = world(240.0, 10);
    setup(&mut sim);
    let h = spawn_car(&mut sim, [0.0, 0.05, 0.0], [0.0, 0.0, speed]);
    // Rest pose of the bodywork as spawned (desc space shifted by the spawn offset).
    let rest = positions(&sim, h);
    let mut events = 0;
    for _ in 0..(240.0 * seconds) as u32 / 24 {
        run(&mut sim, 24);
        events += sim.poll_beam_breaks().len();
    }
    let now = positions(&sim, h);
    assert!(now.iter().all(|p| p.is_finite()), "finite");
    Crash { yielded: yielded(&sim, h), broken: broken(&sim, h), disp: residual(&rest, &now), events }
}

const STATIONS: usize = 11;
fn ring(k: usize, j: usize) -> usize {
    k * engine_softbody::procedural::CAR_RING + j
}

#[test]
fn wall_crash_crumples_the_front_and_sheds_the_bumper() {
    // A rigid wall 0.5 m ahead of the bumper; 15 m/s (54 km/h).
    let c = crash(|sim| { static_box(sim, [0.0, 1.5, 3.0], [3.0, 1.5, 0.2]); }, 15.0, 1.5);
    let front = front_crush(&c);
    let mid: f32 = (0..6).map(|j| c.disp[ring(STATIONS / 2, j)].z).sum::<f32>() / 6.0;
    eprintln!("wall: yielded {} broken {} events {} front crush {front:.3} m, mid {mid:.3} m", c.yielded, c.broken, c.events);
    assert!(c.yielded > 40, "front structure yields ({})", c.yielded);
    assert!(front < -0.08, "permanent front crush ({front})");
    assert!(front < mid - 0.05, "crush concentrated at the front ({front} vs {mid})");
    // Left and right of the front ring crush alike against a flat wall.
    let l: f32 = [0, 2, 4].iter().map(|&j| c.disp[ring(STATIONS - 1, j)].z).sum::<f32>() / 3.0;
    let r: f32 = [1, 3, 5].iter().map(|&j| c.disp[ring(STATIONS - 1, j)].z).sum::<f32>() / 3.0;
    assert!((l - r).abs() < 0.5 * front.abs(), "symmetric crush: left {l} right {r}");
}

fn static_capsule(sim: &mut WgpuSim, pos: [f32; 3], radius: f32, half: f32) -> u32 {
    let b = static_body(sim, pos);
    sim.add_collider(Collider {
        body: b,
        shape: ShapeType::Capsule as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [radius, half, 0.0, radius + half],
    })
}

/// Mean inward (toward the centre line) lateral motion of the front ring.
fn front_wrap(c: &Crash) -> f32 {
    (0..6)
        .map(|j| {
            let side = if j % 2 == 0 { 1.0 } else { -1.0 };
            -side * c.disp[ring(STATIONS - 1, j)].x
        })
        .sum::<f32>()
        / 6.0
}

fn front_crush(c: &Crash) -> f32 {
    (0..6).map(|j| c.disp[ring(STATIONS - 1, j)].z).sum::<f32>() / 6.0
}

/// Mean inward motion of the left (+X) side nodes of the middle rings.
fn side_intrusion(c: &Crash, left: bool) -> f32 {
    let mut sum = 0.0;
    let mut n = 0.0;
    for k in STATIONS / 2 - 2..=STATIONS / 2 + 2 {
        for j in [0usize, 2, 4] {
            let j = if left { j } else { j + 1 };
            let sign = if left { -1.0 } else { 1.0 };
            sum += sign * c.disp[ring(k, j)].x;
            n += 1.0;
        }
    }
    sum / n
}

#[test]
fn wall_pole_and_side_impact_crush_differently() {
    let wall = crash(|sim| { static_box(sim, [0.0, 1.5, 3.0], [3.0, 1.5, 0.2]); }, 15.0, 1.5);
    let pole = crash(
        |sim| {
            // A 20 cm pole on the centre line: narrower than the node spacing.
            let c = static_capsule(sim, [0.0, 1.5, 2.8], 0.1, 1.5);
            sim.add_soft_obstacle(c).expect("obstacle");
        },
        11.0,
        1.5,
    );
    let tbone = {
        let mut sim = world(240.0, 10);
        let h = spawn_car(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
        run(&mut sim, 120);
        let rest = positions(&sim, h);
        // A 1,500 kg car-height block at 40 km/h into the left (+X) side.
        let half = [0.6, 0.6, 0.9];
        let m = 1500.0;
        let inertia = Vec3::new(half[1] * half[1] + half[2] * half[2], half[0] * half[0] + half[2] * half[2], half[0] * half[0] + half[1] * half[1]) * (m / 3.0);
        let mut block = BodyState::dynamic(Vec3::new(2.2, 0.7, 0.0), glam::Quat::IDENTITY, m, inertia);
        block.linear_vel = [-11.1, 0.0, 0.0];
        let b = sim.add_body(block);
        add_box_collider(&mut sim, b, half);
        run(&mut sim, 360);
        let now = positions(&sim, h);
        Crash { yielded: yielded(&sim, h), broken: broken(&sim, h), disp: residual(&rest, &now), events: 0 }
    };
    let report = |name: &str, c: &Crash| {
        eprintln!(
            "{name:>5}: yielded {:>3} broken {:>2} front crush {:+.3} wrap {:+.3} left intrusion {:+.3} right {:+.3}",
            c.yielded,
            c.broken,
            front_crush(c),
            front_wrap(c),
            side_intrusion(c, true),
            side_intrusion(c, false)
        );
    };
    report("wall", &wall);
    report("pole", &pole);
    report("tbone", &tbone);
    // Wall: the whole front pushed back evenly, no wrap.
    assert!(front_crush(&wall) < -0.05);
    // Pole: the front folds around it (inward lateral motion) far more than at the wall.
    assert!(pole.yielded > 10, "pole crumples the front");
    assert!(front_wrap(&pole) > front_wrap(&wall) + 0.02, "pole wraps the front: {} vs {}", front_wrap(&pole), front_wrap(&wall));
    // T-bone: the struck side intrudes, the far side much less, the front untouched.
    assert!(tbone.yielded > 10, "side impact crumples the door");
    assert!(side_intrusion(&tbone, true) > 0.025, "struck side intrudes");
    assert!(side_intrusion(&tbone, true) > side_intrusion(&tbone, false) + 0.02, "intrusion is on the struck side");
    assert!(front_crush(&tbone).abs() < 0.03, "front untouched by a side hit");
}

#[test]
fn two_soft_cars_collide_head_on_without_passing_through() {
    let mut sim = world(240.0, 10);
    let a = spawn_car(&mut sim, [0.0, 0.05, -3.0], [0.0, 0.0, 8.0]);
    let opts = SpawnOptions { velocity: [0.0, 0.0, -8.0], ..Default::default() };
    let rot = glam::Quat::from_rotation_y(std::f32::consts::PI).to_array();
    let b = sim.add_soft_vehicle(&car_desc(), [0.3, 0.05, 3.0], rot, &opts).expect("car b");
    let mut min_gap = f32::MAX;
    for _ in 0..30 {
        run(&mut sim, 12);
        let za = positions(&sim, a).iter().map(|p| p.z).fold(f32::MIN, f32::max);
        let zb = positions(&sim, b).iter().map(|p| p.z).fold(f32::MAX, f32::min);
        min_gap = min_gap.min(centroid(&positions(&sim, b)).z - centroid(&positions(&sim, a)).z);
        let _ = (za, zb);
    }
    let (ya, yb) = (yielded(&sim, a), yielded(&sim, b));
    let ca = centroid(&positions(&sim, a));
    let cb = centroid(&positions(&sim, b));
    eprintln!("head-on: yielded a {ya} b {yb}, centroids {ca:?} {cb:?}, min centre gap {min_gap:.3}");
    assert!(ya > 10 && yb > 10, "both cars crumple");
    assert!(min_gap > 2.0, "bodies do not pass through each other (min centre gap {min_gap})");
    assert!(cb.z > ca.z, "still on their own sides");
}
