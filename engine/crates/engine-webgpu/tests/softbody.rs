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
        params: [
            half[0],
            half[1],
            half[2],
            (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ],
    });
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3]) -> u32 {
    let b = static_body(sim, pos);
    add_box_collider(sim, b, half);
    b
}

/// Flat ground (top at y = 0), soft arena, 120 Hz.
fn world(hz: f32, iterations: u32) -> WgpuSim {
    let mut sim = WgpuSim::new_with_soft(
        capacity(),
        VehicleCapacity::default(),
        SoftCapacity::default(),
    )
    .expect("wgpu");
    sim.set_params(engine_softbody::soft_scene_params(hz, iterations));
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
    static_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0]);
    sim
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h)
        .iter()
        .map(|b| Vec3::from_array(b.position))
        .collect()
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
    let opts = SpawnOptions {
        rigid_core: false,
        ..Default::default()
    };
    let h = sim
        .add_soft_vehicle(&d, [0.0, 0.6, 0.0], [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn");
    let rest: Vec<Vec3> = positions(&sim, h);
    run(&mut sim, 240);
    let p = positions(&sim, h);
    assert!(p.iter().all(|x| x.is_finite()), "finite");
    let lowest = p.iter().map(|x| x.y).fold(f32::MAX, f32::min);
    assert!(
        lowest > 0.0 && lowest < 0.12,
        "box rests on the ground: lowest node {lowest}"
    );
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
    let yielded = beams
        .iter()
        .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
        .count();
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
    let mut d = load_truck(&box_truck(
        6,
        2,
        2,
        0.4,
        400.0,
        3.0e6,
        f32::INFINITY,
        f32::INFINITY,
    ));
    for n in &mut d.nodes {
        if n.position[0] < 0.01 {
            n.fixed = true;
        }
        n.collision = false;
    }
    let hz = 120.0;
    let iters = 40;
    let mut sim = world(hz, iters);
    sim.set_soft_settings(SoftSettings {
        plasticity: false,
        breaking: false,
        sleep_sync: false,
        chebyshev_rho: 0.0,
        ..Default::default()
    });
    let mut p = engine_softbody::soft_scene_params(hz, iters);
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    let opts = SpawnOptions {
        rigid_core: false,
        ..Default::default()
    };
    let h = sim
        .add_soft_vehicle(&d, [0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn");
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
    let tip = |x: &[Vec3]| {
        x.iter()
            .zip(&rest)
            .filter(|(_, r)| r.x > 1.9)
            .map(|(p, r)| r.y - p.y)
            .sum::<f32>()
            / 4.0
    };
    let (g, c) = (tip(&gpu), tip(&cpu.x));
    eprintln!("tip sag: gpu {g:.5} m, cpu {c:.5} m");
    assert!(g > 0.0 && c > 0.0);
    assert!(
        (g - c).abs() < 0.1 * c,
        "GPU and CPU agree within 10%: {g} vs {c}"
    );
}

fn spawn_car(sim: &mut WgpuSim, pos: [f32; 3], vel: [f32; 3]) -> SoftHandle {
    let opts = SpawnOptions {
        velocity: vel,
        ..Default::default()
    };
    sim.add_soft_vehicle(&car_desc(), pos, [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn car")
}

fn yielded(sim: &WgpuSim, h: SoftHandle) -> usize {
    sim.read_soft_beams(h)
        .iter()
        .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
        .count()
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
    eprintln!(
        "contacts {} lowest node {lowest:.3} ke {:.3} yielded {}",
        st.contacts,
        kinetic(&sim, h),
        yielded(&sim, h)
    );
    for w in &st.wheels {
        eprintln!(
            "  wheel contact {} L {:.3} F {:.0}",
            w.in_contact, w.suspension_length, w.suspension_force
        );
    }
    assert_eq!(st.contacts, 4, "stands on four wheels");
    assert!(lowest > 0.12, "bodywork clear of the ground ({lowest})");
    let total: f32 = st.wheels.iter().map(|w| w.suspension_force).sum();
    let parts: f32 = sim
        .read_soft_particles(h)
        .iter()
        .filter(|b| b.inv_mass > 0.0)
        .map(|b| 1.0 / b.inv_mass)
        .sum();
    let core_body = sim.read_body(sim.soft_host().entry(h).core.unwrap().1);
    let mass = parts + 1.0 / core_body.inv_mass;
    let joints = sim.download_joints();
    let r = sim.soft_host().entry(h).core_joints.clone();
    let lam: Vec<[f32; 8]> = joints[r.start as usize..r.end as usize]
        .iter()
        .map(|j| j.lambda)
        .collect();
    let sum: Vec3 = lam.iter().map(|l| Vec3::new(l[0], l[1], l[2])).sum();
    eprintln!("joints {} lambda sum {sum:?}", lam.len());
    eprintln!(
        "particles {parts:.1} kg + core {:.1} kg = {mass:.1} kg (desc {:.1})",
        1.0 / core_body.inv_mass,
        car_desc().total_mass()
    );
    assert!(
        (total - mass * 9.81).abs() < 0.08 * mass * 9.81,
        "suspension carries the vehicle: {total} vs {}",
        mass * 9.81
    );
    assert_eq!(yielded(&sim, h), 0, "standing does not deform the body");
    assert!(kinetic(&sim, h) < 2.0, "at rest");
}

#[test]
fn soft_car_drives_and_steers_without_deforming() {
    // 240 Hz as the demo: 120 Hz × 12 is under-converged and acts as drag.
    let mut sim = world(240.0, 10);
    let h = spawn_car(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
    run(&mut sim, 240);
    let core = sim.soft_host().entry(h).core.expect("rigid core").0;
    let start = centroid(&positions(&sim, h));
    sim.set_vehicle_input(
        core,
        VehicleInput {
            forward: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 720);
    let mid = centroid(&positions(&sim, h));
    let v = Vec3::from_array(
        sim.read_body(sim.soft_host().entry(h).core.unwrap().1)
            .linear_vel,
    );
    eprintln!(
        "after 3 s throttle: moved {:?}, speed {:.2} m/s, yielded {}",
        mid - start,
        v.length(),
        yielded(&sim, h)
    );
    assert!(
        mid.z - start.z > 8.0,
        "drives forward (+Z): {:?}",
        mid - start
    );
    assert!((mid.x - start.x).abs() < 1.0, "straight line");
    sim.set_vehicle_input(
        core,
        VehicleInput {
            forward: 0.6,
            steer: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 480);
    let end = centroid(&positions(&sim, h));
    eprintln!("after steering: {:?}", end - mid);
    assert!(
        (end.x - mid.x).abs() > 1.5,
        "steering turns the car: {:?}",
        end - mid
    );
    assert_eq!(broken(&sim, h), 0, "driving breaks nothing");
    assert!(
        yielded(&sim, h) < 5,
        "driving barely yields: {}",
        yielded(&sim, h)
    );
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
    now.iter()
        .zip(rest)
        .map(|(p, q)| r1.transpose() * (*p - c1) - r0.transpose() * (*q - c0))
        .collect()
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
        run(&mut sim, 48);
        events += sim.poll_beam_breaks().len();
    }
    let now = positions(&sim, h);
    assert!(now.iter().all(|p| p.is_finite()), "finite");
    Crash {
        yielded: yielded(&sim, h),
        broken: broken(&sim, h),
        disp: residual(&rest, &now),
        events,
    }
}

const STATIONS: usize = 11;
fn ring(k: usize, j: usize) -> usize {
    k * engine_softbody::procedural::CAR_RING + j
}

#[test]
fn wall_crash_crumples_the_front_and_sheds_the_bumper() {
    // A rigid wall 0.5 m ahead of the bumper; 15 m/s (54 km/h).
    let c = crash(
        |sim| {
            static_box(sim, [0.0, 1.5, 3.0], [3.0, 1.5, 0.2]);
        },
        15.0,
        1.5,
    );
    let front = front_crush(&c);
    let mid: f32 = (0..6).map(|j| c.disp[ring(STATIONS / 2, j)].z).sum::<f32>() / 6.0;
    eprintln!(
        "wall: yielded {} broken {} events {} front crush {front:.3} m, mid {mid:.3} m",
        c.yielded, c.broken, c.events
    );
    assert!(c.yielded > 40, "front structure yields ({})", c.yielded);
    assert!(front < -0.08, "permanent front crush ({front})");
    assert!(
        front < mid - 0.05,
        "crush concentrated at the front ({front} vs {mid})"
    );
    // Left and right of the front ring crush alike against a flat wall.
    let l: f32 = [0, 2, 4]
        .iter()
        .map(|&j| c.disp[ring(STATIONS - 1, j)].z)
        .sum::<f32>()
        / 3.0;
    let r: f32 = [1, 3, 5]
        .iter()
        .map(|&j| c.disp[ring(STATIONS - 1, j)].z)
        .sum::<f32>()
        / 3.0;
    assert!(
        (l - r).abs() < 0.5 * front.abs(),
        "symmetric crush: left {l} right {r}"
    );
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
    let wall = crash(
        |sim| {
            static_box(sim, [0.0, 1.5, 3.0], [3.0, 1.5, 0.2]);
        },
        15.0,
        1.5,
    );
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
        let inertia = Vec3::new(
            half[1] * half[1] + half[2] * half[2],
            half[0] * half[0] + half[2] * half[2],
            half[0] * half[0] + half[1] * half[1],
        ) * (m / 3.0);
        let mut block =
            BodyState::dynamic(Vec3::new(2.2, 0.7, 0.0), glam::Quat::IDENTITY, m, inertia);
        block.linear_vel = [-11.1, 0.0, 0.0];
        let b = sim.add_body(block);
        add_box_collider(&mut sim, b, half);
        let collider = sim.collider_count() - 1;
        sim.add_soft_obstacle(collider).expect("obstacle");
        run(&mut sim, 360);
        let now = positions(&sim, h);
        Crash {
            yielded: yielded(&sim, h),
            broken: broken(&sim, h),
            disp: residual(&rest, &now),
            events: 0,
        }
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
    assert!(
        front_wrap(&pole) > front_wrap(&wall) + 0.02,
        "pole wraps the front: {} vs {}",
        front_wrap(&pole),
        front_wrap(&wall)
    );
    // T-bone: the struck side intrudes, the far side much less, the front untouched.
    assert!(tbone.yielded > 10, "side impact crumples the door");
    assert!(side_intrusion(&tbone, true) > 0.015, "struck side intrudes");
    assert!(
        side_intrusion(&tbone, true) > side_intrusion(&tbone, false) + 0.015,
        "intrusion is on the struck side"
    );
    assert!(
        front_crush(&tbone).abs() < 0.03,
        "front untouched by a side hit"
    );
}

#[test]
fn two_soft_cars_collide_head_on_without_passing_through() {
    let mut sim = world(240.0, 10);
    let a = spawn_car(&mut sim, [0.0, 0.05, -3.0], [0.0, 0.0, 8.0]);
    let opts = SpawnOptions {
        velocity: [0.0, 0.0, -8.0],
        ..Default::default()
    };
    let rot = glam::Quat::from_rotation_y(std::f32::consts::PI).to_array();
    let b = sim
        .add_soft_vehicle(&car_desc(), [0.3, 0.05, 3.0], rot, &opts)
        .expect("car b");
    let mut min_gap = f32::MAX;
    for _ in 0..30 {
        run(&mut sim, 12);
        let za = positions(&sim, a)
            .iter()
            .map(|p| p.z)
            .fold(f32::MIN, f32::max);
        let zb = positions(&sim, b)
            .iter()
            .map(|p| p.z)
            .fold(f32::MAX, f32::min);
        min_gap = min_gap.min(centroid(&positions(&sim, b)).z - centroid(&positions(&sim, a)).z);
        let _ = (za, zb);
    }
    let (ya, yb) = (yielded(&sim, a), yielded(&sim, b));
    let ca = centroid(&positions(&sim, a));
    let cb = centroid(&positions(&sim, b));
    eprintln!(
        "head-on: yielded a {ya} b {yb}, centroids {ca:?} {cb:?}, min centre gap {min_gap:.3}"
    );
    assert!(ya > 10 && yb > 10, "both cars crumple");
    assert!(
        min_gap > 2.0,
        "bodies do not pass through each other (min centre gap {min_gap})"
    );
    assert!(cb.z > ca.z, "still on their own sides");
}

/// Real BeamNG content, opt-in (not redistributable, so not in the repo):
/// RUBBLE_JBEAM_DIR=<vehicles dir with agenty_legocar/ and common/>.
#[test]
#[ignore]
fn jbeam_legocar_settles_and_drives() {
    let Ok(dir) = std::env::var("RUBBLE_JBEAM_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    fn collect(d: &std::path::Path, out: &mut Vec<(String, String)>) {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else if p.extension().is_some_and(|x| x == "jbeam") {
                out.push((
                    p.display().to_string(),
                    std::fs::read_to_string(&p).unwrap_or_default(),
                ));
            }
        }
    }
    let mut files = Vec::new();
    collect(&dir.join("agenty_legocar"), &mut files);
    collect(&dir.join("common"), &mut files);
    let pc = std::fs::read_to_string(dir.join("agenty_legocar/start.pc")).unwrap();
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let d = engine_softbody::jbeam::load_jbeam(&refs, Some(&pc)).unwrap();
    let (mn, _) = d.bounds();
    let mut sim = world(240.0, 10);
    let h = sim
        .add_soft_vehicle(
            &d,
            [0.0, 0.25 - mn[1], 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .expect("spawn");
    run(&mut sim, 480);
    let core = sim.soft_host().entry(h).core.expect("core").0;
    let st = sim.vehicle_state(core);
    eprintln!(
        "legocar: {} nodes, contacts {}, yielded {}, ke {:.2}",
        d.nodes.len(),
        st.contacts,
        yielded(&sim, h),
        kinetic(&sim, h)
    );
    let start = centroid(&positions(&sim, h));
    sim.set_vehicle_input(
        core,
        VehicleInput {
            forward: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 720);
    let end = centroid(&positions(&sim, h));
    eprintln!(
        "legocar drove {:?}, yielded {}, broken {}",
        end - start,
        yielded(&sim, h),
        broken(&sim, h)
    );
    assert!(positions(&sim, h).iter().all(|p| p.is_finite()));
    assert!((end - start).length() > 3.0, "drives");
}

/// Real Rigs of Rods content, opt-in (GPL, so not in the repo):
/// RUBBLE_ROR_DIR=<content-master from RigsOfRods/content>.
#[test]
#[ignore]
fn ror_bus_and_semi_settle_drive_and_steer() {
    let Ok(dir) = std::env::var("RUBBLE_ROR_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    for file in [
        "agora/95bbUID-agoras.truck",
        "agora/95bbUID-agoral.truck",
        "dafsemi/b6b0UID-semi.truck",
    ] {
        let d = load_truck(&std::fs::read_to_string(dir.join(file)).unwrap());
        let (mn, mx) = d.bounds();
        let mut sim = world(240.0, 10);
        let centre = [-(mn[0] + mx[0]) * 0.5, 0.25 - mn[1], -(mn[2] + mx[2]) * 0.5];
        let h = sim
            .add_soft_vehicle(&d, centre, [0.0, 0.0, 0.0, 1.0], &SpawnOptions::default())
            .expect("spawn");
        run(&mut sim, 480);
        let core = sim.soft_host().entry(h).core.expect("core").0;
        let st = sim.vehicle_state(core);
        let settled = yielded(&sim, h);
        let start = centroid(&positions(&sim, h));
        sim.set_vehicle_input(
            core,
            VehicleInput {
                forward: 1.0,
                ..Default::default()
            },
        );
        run(&mut sim, 960);
        let mid = centroid(&positions(&sim, h));
        sim.set_vehicle_input(
            core,
            VehicleInput {
                forward: 0.5,
                steer: 1.0,
                ..Default::default()
            },
        );
        run(&mut sim, 960);
        let end = centroid(&positions(&sim, h));
        let straight = mid - start;
        let turn = (end - mid)
            .normalize_or_zero()
            .dot(straight.normalize_or_zero());
        eprintln!(
            "{}: contacts {} at rest, bent {settled}; drove {:.1} m in 4 s; heading change cos {turn:.2}; bent {} broken {}",
            d.name,
            st.contacts,
            straight.length(),
            yielded(&sim, h),
            broken(&sim, h)
        );
        assert!(positions(&sim, h).iter().all(|p| p.is_finite()));
        assert_eq!(
            st.contacts as usize,
            d.wheels.len(),
            "{}: every wheel on the ground",
            d.name
        );
        assert!(straight.length() > 3.0, "{} drives", d.name);
        assert!(turn < 0.97, "{} turns", d.name);
    }
}

#[test]
fn destruction_events_name_the_parts_that_tear_off() {
    // A 1,500 kg block at 60 km/h into the front-left corner pulls the front
    // bumper sideways: its mounts tear (one detacher group) and the poll reports
    // a piece holding the `bumper_front` part.
    let mut sim = world(240.0, 10);
    let (desc, _) = engine_softbody::procedural::car_desc(&CarParams::default());
    let h = sim
        .add_soft_vehicle(
            &desc,
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions::default(),
        )
        .expect("spawn");
    run(&mut sim, 120);
    assert!(
        sim.poll_soft_events().pieces.is_empty(),
        "nothing tears while settling"
    );
    let half = [0.6, 0.6, 0.9];
    let m = 1500.0;
    let inertia = Vec3::new(
        half[1] * half[1] + half[2] * half[2],
        half[0] * half[0] + half[2] * half[2],
        half[0] * half[0] + half[1] * half[1],
    ) * (m / 3.0);
    let mut block = BodyState::dynamic(Vec3::new(1.8, 0.75, 1.6), glam::Quat::IDENTITY, m, inertia);
    block.linear_vel = [-16.7, 0.0, 0.0];
    let b = sim.add_body(block);
    add_box_collider(&mut sim, b, half);
    let collider = sim.collider_count() - 1;
    sim.add_soft_obstacle(collider).expect("obstacle");
    let (mut breaks, mut pieces) = (0, Vec::new());
    for _ in 0..24 {
        run(&mut sim, 20);
        let ev = sim.poll_soft_events();
        breaks += ev.breaks.len();
        pieces.extend(ev.pieces);
    }
    for p in &pieces {
        eprintln!(
            "piece: {} nodes, {:.1} kg, parts {:?}",
            p.nodes.len(),
            p.mass,
            p.parts
        );
    }
    assert!(breaks >= 10, "{breaks} breaks");
    assert!(
        pieces
            .iter()
            .any(|p| p.parts.iter().any(|n| n == "bumper_front")),
        "the front bumper is reported as a detached piece"
    );
    // Each node is reported in at most one piece.
    let mut seen = std::collections::HashSet::new();
    assert!(pieces
        .iter()
        .flat_map(|p| &p.nodes)
        .all(|n| seen.insert(*n)));
    let d = sim.read_soft_damage(h);
    let detached = sim.soft_host().entry(h).detached.clone();
    let max_def = d
        .node_deformation
        .iter()
        .zip(&detached)
        .filter(|(_, &x)| !x)
        .map(|(v, _)| *v)
        .fold(0.0, f32::max);
    eprintln!(
        "damage: bent {} broken {} plastic work {:.0} J max node deformation {max_def:.2} m",
        d.bent, d.broken, d.plastic_work
    );
    assert!(
        d.bent > 20 && d.broken >= 10 && d.plastic_work > 100.0 && max_def > 0.05 && max_def < 1.5
    );
}

/// Distance from `p` to triangle (a, b, c).
fn point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (p - a).length();
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (p - b).length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (p - (a + ab * (d1 / (d1 - d3)))).length();
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (p - c).length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (p - (a + ac * (d2 / (d2 - d6)))).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).length();
    }
    let denom = 1.0 / (va + vb + vc);
    (p - (a + ab * (vb * denom) + ac * (vc * denom))).length()
}

#[test]
fn skinned_render_mesh_follows_the_physics_surface_through_a_crash() {
    // The detailed render mesh, bound part by part through the public API and
    // skinned on the GPU, must (a) match the CPU formula and (b) stay on the
    // deformed physics skin: its distance to the nearest collision triangle may
    // not grow by more than a few cm where the car crumples.
    use engine_softbody::visual::{car_visual, material, part};
    let p = CarParams::default();
    let (desc, layout) = engine_softbody::procedural::car_desc(&p);
    let rest_nodes: Vec<Vec3> = desc
        .nodes
        .iter()
        .map(|n| Vec3::from_array(n.position))
        .collect();
    let vis = car_visual(&p, &layout, &rest_nodes);
    let mut sim = world(240.0, 10);
    static_box(&mut sim, [0.0, 1.5, 2.6], [3.0, 1.5, 0.2]);
    let h = sim
        .add_soft_vehicle(
            &desc,
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            &SpawnOptions {
                velocity: [0.0, 0.0, 15.6],
                ..Default::default()
            },
        )
        .expect("spawn");
    let name = |pt: u8| match pt {
        part::BONNET => "bonnet",
        part::BOOT => "boot",
        part::DOOR_FL => "door_fl",
        part::DOOR_RL => "door_rl",
        part::DOOR_FR => "door_fr",
        part::DOOR_RR => "door_rr",
        part::BUMPER_F => "bumper_front",
        part::BUMPER_R => "bumper_rear",
        part::ENGINE => "engine",
        part::ROOF => "roof",
        _ => "body",
    };
    let mut bindings = vec![None; vis.vertices.len()];
    for pt in 0..=part::ROOF {
        let ids: Vec<usize> = (0..vis.vertices.len())
            .filter(|&i| vis.part[i] == pt)
            .collect();
        if ids.is_empty() {
            continue;
        }
        let verts: Vec<Vec3> = ids.iter().map(|&i| vis.vertices[i]).collect();
        let got = sim
            .soft_host()
            .bind_mesh(h, &verts, Some(name(pt)))
            .expect("bind");
        for (i, b) in ids.into_iter().zip(got) {
            bindings[i] = Some(b);
        }
    }
    let bindings: Vec<_> = bindings.into_iter().map(Option::unwrap).collect();
    let skinner = engine_webgpu::SoftSkinner::new(&sim, &bindings, &vis.vertices, &vis.tris);
    let first = sim.soft_host().entry(h).first_body;
    // Paint vertices of the shell and the panels (what the eye reads as the body).
    let check: Vec<usize> = (0..vis.vertices.len())
        .filter(|&i| vis.part[i] <= part::DOOR_RR || vis.part[i] == part::ROOF)
        .filter(|&i| {
            vis.tris
                .iter()
                .zip(&vis.tri_material)
                .any(|(t, &m)| m == material::PAINT && t.contains(&(i as u32)))
        })
        .collect();
    let surface_distance = |sim: &WgpuSim, pos: &[[f32; 4]]| -> Vec<f32> {
        let x: Vec<Vec3> = sim
            .read_soft_particles(h)
            .iter()
            .map(|b| Vec3::from_array(b.position))
            .collect();
        check
            .iter()
            .map(|&i| {
                let q = Vec3::new(pos[i][0], pos[i][1], pos[i][2]);
                desc.triangles
                    .iter()
                    .map(|t| {
                        point_triangle(
                            q,
                            x[t.nodes[0] as usize],
                            x[t.nodes[1] as usize],
                            x[t.nodes[2] as usize],
                        )
                    })
                    .fold(f32::MAX, f32::min)
            })
            .collect()
    };
    let (pos0, _) = skinner.skin_and_read(&sim);
    let d0 = surface_distance(&sim, &pos0);
    run(&mut sim, 360);
    let (pos1, nrm1) = skinner.skin_and_read(&sim);
    // (a) GPU skinning equals SkinBinding::skin on the read-back particles.
    let parts = sim.read_soft_particles(h);
    let at = |g: u32| Vec3::from_array(parts[(g - first) as usize].position);
    let mut worst = 0.0f32;
    for (b, p) in bindings.iter().zip(&pos1) {
        let (cpu, torn) = b.skin(at);
        assert_eq!(torn, p[3] > 0.5);
        worst = worst.max((cpu - Vec3::new(p[0], p[1], p[2])).length());
    }
    assert!(worst < 1e-3, "GPU vs CPU skinning {worst}");
    assert!(nrm1.iter().all(|n| n.iter().all(|x| x.is_finite())));
    // (b) The render surface stays on the physics surface.
    let d1 = surface_distance(&sim, &pos1);
    let mut grow: Vec<f32> = check
        .iter()
        .enumerate()
        .filter(|&(_, &i)| pos1[i][3] < 0.5)
        .map(|(k, _)| d1[k] - d0[k])
        .collect();
    grow.sort_by(f32::total_cmp);
    let q = |f: f32| grow[((grow.len() - 1) as f32 * f) as usize];
    eprintln!(
        "render-to-physics surface drift after a 56 km/h wall hit ({} vertices): median {:+.3} m, 95% {:+.3} m, 99% {:+.3} m, max {:+.3} m",
        grow.len(),
        q(0.5),
        q(0.95),
        q(0.99),
        q(1.0)
    );
    // The brittle A-pillars partly snap at 56 km/h, so the roof shell (bound to
    // the roof nodes) moves against the stretched windscreen faces by a few cm.
    assert!(
        q(0.95) < 0.06,
        "95% of the body stays within 6 cm of the physics skin"
    );
    assert!(q(0.5) < 0.01);
}

#[test]
fn rigid_mode_rides_the_core_while_quiet_and_crashes_like_soft() {
    // Rigid mode: a car cruising with nothing near turns rigid within its quiet
    // time; hitting a wall turns it soft in the same step, and the crush matches
    // the always-soft car's.
    let run_crash = |rigid: bool| -> (bool, bool, f32) {
        let mut sim = world(240.0, 10);
        sim.set_soft_settings(SoftSettings {
            rigid_mode: rigid,
            ..Default::default()
        });
        static_box(&mut sim, [0.0, 1.5, 22.5], [3.0, 1.5, 0.2]);
        let (desc, _) = engine_softbody::procedural::car_desc(&CarParams::default());
        let h = sim
            .add_soft_vehicle(
                &desc,
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                &SpawnOptions {
                    velocity: [0.0, 0.0, 15.6],
                    ..Default::default()
                },
            )
            .expect("spawn");
        let rest = positions(&sim, h);
        run(&mut sim, 240); // 1 s of cruising, 15.6 m: still 4 m short of the wall
        let was_rigid = sim.read_soft_rigid(h);
        run(&mut sim, 480);
        let now = positions(&sim, h);
        let ring = engine_softbody::procedural::CAR_RING;
        let n = CarParams::default().stations;
        let len = |x: &[Vec3]| {
            let c = |k: usize| (0..6).map(|j| x[k * ring + j]).sum::<Vec3>() / 6.0;
            (c(n - 1) - c(0)).length()
        };
        (was_rigid, sim.read_soft_rigid(h), len(&rest) - len(&now))
    };
    let (soft_before, _, soft_crush) = run_crash(false);
    let (rigid_before, rigid_after, rigid_crush) = run_crash(true);
    eprintln!("crush: always soft {soft_crush:.3} m, rigid mode {rigid_crush:.3} m");
    assert!(!soft_before, "rigid mode off: never rigid");
    assert!(rigid_before, "cruising quietly for 1 s: rigid");
    assert!(!rigid_after, "after the wall: soft");
    assert!(soft_crush > 0.3);
    assert!(
        (soft_crush - rigid_crush).abs() < 0.05,
        "same crush within 5 cm"
    );
}
