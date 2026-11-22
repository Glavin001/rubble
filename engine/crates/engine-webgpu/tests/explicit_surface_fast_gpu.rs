//! Surface contact on the fast schedules (`ExplicitBeamsGpu::set_surface_fast`: the wide
//! passes with the reused pair list, self-collision split with vehicle contact; the
//! surface pass as its own pipelines after each step's hits, each kind's find and apply a
//! thread an element sized on the GPU) against the pass inside the frame kernels (a
//! workgroup a vehicle or a contact group, or the frame in segments): each frame from one
//! saved state both ways, every state word and every coupling word the same, and the
//! fast schedule in use (vehicle-perf-first-principles.md §8.53).
//!
//! * a lone car into a static wall slot and a proxy pole (static, then free): the wide
//!   passes against the frame in segments;
//! * three cars with vehicle contact, two head-on (a contact group forms) and one alone
//!   into its own wall, each with a pole: self split with vehicle contact against the
//!   groups' and the solo vehicles' own workgroups;
//! * the same with vehicle sleep on and the awake list.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_webgpu::explicit_coupling::write_proxies;
use engine_webgpu::{default_device, CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const CAP: CouplingCapacity = CouplingCapacity { bodies: 4, shapes: 8, planes: 128 };
const HALF: Vec3 = Vec3::new(0.1, 0.9, 0.1);

fn first_diff(a: &[(&str, Vec<u32>)], b: &[(&str, Vec<u32>)]) -> Option<String> {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if x.len() != y.len() {
            return Some(format!("{name}: lengths {} and {}", x.len(), y.len()));
        }
        if let Some(i) = x.iter().zip(y).position(|(p, q)| p != q) {
            let count = x.iter().zip(y).filter(|(p, q)| p != q).count();
            return Some(format!("{name}[{i}] ({:#x} vs {:#x}), {count} words differ", x[i], y[i]));
        }
    }
    None
}

/// The repo sedan with surface contact, facing +z (yaw 0) or −z, its front at `at`,
/// 2 cm off the ground, moving at `speed` along its heading, with a wall slot `wall` m
/// ahead of its front (0: none).
fn sedan(at: Vec3, back: bool, speed: f32, wall: f32) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let bounds = |s: &ExplicitBeams| s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let (lo, hi) = bounds(&s);
    // Its length along z, then turned about y for the heading.
    let along_x = hi.x - lo.x > hi.z - lo.z;
    let dir = if back { -1.0 } else { 1.0 };
    for n in &mut s.nodes {
        let p = if along_x { Vec3::new(-n.pos.z, n.pos.y, n.pos.x) } else { n.pos };
        n.pos = Vec3::new(dir * p.x, p.y, dir * p.z);
    }
    let (lo, hi) = bounds(&s);
    let front = if back { lo.z } else { hi.z };
    let shift = Vec3::new(at.x - 0.5 * (lo.x + hi.x), at.y + 0.02 - lo.y, at.z - front);
    for n in &mut s.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, dir * speed);
    }
    s.ground = at.y;
    s.surface_contact = true;
    s.dissipative_contact = true;
    if wall > 0.0 {
        let c = Vec3::new(at.x, at.y + 1.0, at.z + dir * (wall + 0.5));
        s.boxes.push(StaticBox::centered(c, Vec3::new(2.0, 1.0, 0.5)));
    }
    s
}

/// A 20 cm box pole at `at`: static (`mass` 0) or free.
fn pole(at: Vec3, mass: f32) -> ProxySet {
    let (inv_mass, inv_inertia) = if mass > 0.0 {
        let h = HALF;
        let i = Vec3::new(h.y * h.y + h.z * h.z, h.x * h.x + h.z * h.z, h.x * h.x + h.y * h.y) * (mass / 3.0);
        (1.0 / mass, Vec3::ONE / i)
    } else {
        (0.0, Vec3::ZERO)
    };
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 7,
            pos: at,
            rot: Quat::from_rotation_y(0.3),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half: HALF }, 3, None)],
        frozen: false,
    };
    set.begin_window();
    set
}

/// `frames` frames of `steps` steps, each from one saved state with the fast schedules
/// off then on (continuing from the on half): every state and coupling word the same.
/// `on_path` says whether the on half ran the schedule the case is about; the count of
/// frames it did is returned (`None` without a device).
fn compare(
    label: &str,
    gpu: &mut ExplicitBeamsGpu,
    frames: u32,
    steps: u32,
    on_path: &dyn Fn(&ExplicitBeamsGpu) -> bool,
    after: &mut dyn FnMut(&ExplicitBeamsGpu),
) -> u32 {
    let mut taken = 0;
    for f in 0..frames {
        let snap = gpu.snapshot();
        gpu.set_surface_fast(false);
        gpu.frame(steps);
        gpu.wait();
        assert!(!gpu.surface_fast() && !on_path(gpu), "{label}: frame {f}: the fast schedule ran with it off");
        let (old, old_cpl) = (gpu.state_words(), gpu.coupling_words());
        gpu.restore(&snap);
        gpu.set_surface_fast(true);
        gpu.frame(steps);
        gpu.wait();
        assert!(gpu.surface_fast());
        taken += u32::from(on_path(gpu));
        after(gpu);
        let (new, new_cpl) = (gpu.state_words(), gpu.coupling_words());
        if let Some(d) = first_diff(&old, &new).or_else(|| first_diff(&old_cpl, &new_cpl)) {
            panic!("{label}: frame {f}: the fast schedules differ from the pass in the kernels at {d}");
        }
    }
    eprintln!("{label}: {frames} frames of {steps} steps bit for bit, the fast schedule on {taken}");
    taken
}

fn lone(label: &str, mass: f32, frames: u32, steps: u32) {
    let Ok((_, device, queue)) = default_device() else { return };
    let car = sedan(Vec3::new(20.0, 0.0, -30.0), false, 10.0, 0.6);
    let front = car.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&car), Some(CAP));
    write_proxies(&gpu, &queue, 0, &pole(Vec3::new(20.4, 0.9, front + 0.25), mass)).unwrap();
    gpu.set_coupling_on(true);
    gpu.set_self_split(true);
    gpu.set_wide_list(true);
    gpu.set_wide_fused(true);
    gpu.set_wide(true);
    let taken = compare(label, &mut gpu, frames, steps, &|g| g.is_wide(), &mut |_| {});
    assert_eq!(taken, frames, "{label}: the wide passes ran on {taken} of {frames} frames");
}

/// The first check of a new kernel version: two frames of a few steps.
#[test]
fn tiny_lone_car_wide_is_bit_for_bit() {
    lone("tiny lone car", 0.0, 2, 4);
}

#[test]
fn lone_car_into_a_wall_and_a_static_pole_wide_is_bit_for_bit() {
    lone("lone car, wall and static pole", 0.0, 24, 17);
}

#[test]
fn lone_car_into_a_wall_and_a_free_pole_wide_is_bit_for_bit() {
    lone("lone car, wall and free 60 kg pole", 60.0, 24, 17);
}

fn fleet(label: &str, sleep: bool, frames: u32, steps: u32) {
    let Ok((_, device, queue)) = default_device() else { return };
    // Two head-on, 1 m apart, and one alone into its own wall.
    let a = sedan(Vec3::new(0.0, 0.0, 0.0), false, 12.0, 0.0);
    let b = sedan(Vec3::new(0.0, 0.0, 1.0), true, 12.0, 0.0);
    let c = sedan(Vec3::new(12.0, 0.0, 0.0), false, 10.0, 0.5);
    let cars = [a, b, c];
    let cfg = sleep.then(engine_softbody::explicit_sleep::VehicleSleep::default);
    let mut gpu = ExplicitBeamsGpu::with_sleep(&device, &queue, &cars, Some(CAP), cfg);
    // A pole beside the pair's meeting point, and one before the lone car's wall.
    write_proxies(&gpu, &queue, 0, &pole(Vec3::new(0.7, 0.9, 0.4), 60.0)).unwrap();
    write_proxies(&gpu, &queue, 1, &pole(Vec3::new(-0.7, 0.9, 0.6), 0.0)).unwrap();
    write_proxies(&gpu, &queue, 2, &pole(Vec3::new(12.4, 0.9, 0.3), 0.0)).unwrap();
    gpu.set_coupling_on(true);
    gpu.set_vehicle_contact(true);
    gpu.set_group_split(true);
    gpu.set_self_split(true);
    let mut grouped = 0;
    let taken = compare(label, &mut gpu, frames, steps, &|g| g.self_split_used(), &mut |g| {
        grouped += u32::from(!g.contact_group_sizes().is_empty());
    });
    assert_eq!(taken, frames, "{label}: self split with vehicle contact ran on {taken} of {frames} frames");
    assert!(grouped > 0, "{label}: no contact group formed");
    eprintln!("{label}: a contact group on {grouped} of {frames} frames");
}

/// The first check of a new kernel version: two frames of a few steps.
#[test]
fn tiny_fleet_self_split_is_bit_for_bit() {
    let Ok((_, device, queue)) = default_device() else { return };
    let cars = [sedan(Vec3::ZERO, false, 12.0, 0.0), sedan(Vec3::new(0.0, 0.0, 1.0), true, 12.0, 0.0)];
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, &cars, Some(CAP));
    gpu.set_coupling_on(true);
    gpu.set_vehicle_contact(true);
    gpu.set_group_split(true);
    gpu.set_self_split(true);
    let taken = compare("tiny fleet", &mut gpu, 2, 4, &|g| g.self_split_used(), &mut |_| {});
    assert_eq!(taken, 2);
}

#[test]
fn fleet_in_contact_self_split_is_bit_for_bit() {
    fleet("three cars, a head-on pair and one alone", false, 30, 17);
}

#[test]
fn fleet_in_contact_with_vehicle_sleep_self_split_is_bit_for_bit() {
    fleet("three cars with vehicle sleep", true, 30, 17);
}
