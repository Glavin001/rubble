//! A fleet with vehicle contact on the wide passes while its contact plan holds no group
//! (`ExplicitBeamsGpu::set_vv_wide`; vehicle-perf-first-principles.md §8.57) against the
//! path it replaces (a workgroup a vehicle): each frame from one saved state both ways,
//! every state word and every coupling word the same, and the schedule in use.
//!
//! * cars apart, without surface contact (self split with vehicle contact: the walk, the
//!   pairs and the vehicles' step shared with today's launches) and with it (the wgpu
//!   host's default keeps the pass in the kernels with vehicle contact: the wide frame
//!   brings its own walk, pairs and surface pipelines);
//! * two cars closing head-on and one alone: frames without a group, then with one, the
//!   host's read of the plan a few frames behind, so that some frames hold the wide
//!   schedule while the plan forms a group (the gate then gives today's launches their
//!   workgroups);
//! * a vehicle falling out of its frame bounds: reported from the wide passes as from
//!   the solo frame.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Node, StaticBox, Triangle};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_webgpu::explicit_coupling::write_proxies;
use engine_webgpu::{default_device, CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const CAP: CouplingCapacity = CouplingCapacity { bodies: 4, shapes: 8, planes: 128 };

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

/// The repo sedan facing +z (or −z), its front at `at`, 2 cm off the ground, moving at
/// `speed` along its heading; `surface`: with surface contact; a wall slot `wall` m ahead
/// of its front (0: none).
fn sedan(at: Vec3, back: bool, speed: f32, surface: bool, wall: f32) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let bounds = |s: &ExplicitBeams| s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let (lo, hi) = bounds(&s);
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
    s.surface_contact = surface;
    s.dissipative_contact = true;
    if wall > 0.0 {
        let c = Vec3::new(at.x, at.y + 1.0, at.z + dir * (wall + 0.5));
        s.boxes.push(StaticBox::centered(c, Vec3::new(2.0, 1.0, 0.5)));
    }
    s
}

/// A static 20 cm box pole at `at`.
fn pole(at: Vec3) -> ProxySet {
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 7,
            pos: at,
            rot: Quat::from_rotation_y(0.3),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 0.0,
            inv_inertia: Vec3::ZERO,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half: Vec3::new(0.1, 0.9, 0.1) }, 3, None)],
        frozen: false,
    };
    set.begin_window();
    set
}

/// The host's switches the wide frame needs beside vehicle contact.
fn fast_paths(gpu: &mut ExplicitBeamsGpu) {
    gpu.set_vehicle_contact(true);
    gpu.set_group_split(true);
    gpu.set_self_split(true);
    gpu.set_wide_list(true);
    gpu.set_wide_fused(true);
    gpu.set_wide(true);
}

/// What the on halves of [`compare`]'s frames were: how many held the wide schedule,
/// how many of those while the plan held a group, and how many had a group at all.
#[derive(Default, Debug)]
struct Taken {
    wide: u32,
    wide_with_group: u32,
    grouped: u32,
}

/// `frames` frames of `steps` steps, each from one saved state with the setting off then
/// on (continuing from the on half): every state and coupling word the same.
fn compare(label: &str, gpu: &mut ExplicitBeamsGpu, frames: u32, steps: u32) -> Taken {
    let mut taken = Taken::default();
    for f in 0..frames {
        let snap = gpu.snapshot();
        gpu.set_vv_wide(false);
        gpu.frame(steps);
        gpu.wait();
        assert!(!gpu.vv_wide_used(), "{label}: frame {f}: the wide schedule with the setting off");
        let (old, old_cpl) = (gpu.state_words(), gpu.coupling_words());
        gpu.restore(&snap);
        gpu.set_vv_wide(true);
        gpu.frame(steps);
        gpu.wait();
        let group = !gpu.contact_group_sizes().is_empty();
        taken.wide += u32::from(gpu.vv_wide_used());
        taken.wide_with_group += u32::from(gpu.vv_wide_used() && group);
        taken.grouped += u32::from(group);
        let (new, new_cpl) = (gpu.state_words(), gpu.coupling_words());
        if let Some(d) = first_diff(&old, &new).or_else(|| first_diff(&old_cpl, &new_cpl)) {
            panic!("{label}: frame {f} (wide schedule {}, a group {group}): differs from today's path at {d}", gpu.vv_wide_used());
        }
    }
    eprintln!("{label}: {frames} frames of {steps} steps bit for bit; {taken:?}");
    taken
}

fn apart(label: &str, surface: bool, frames: u32, steps: u32) {
    let Ok((_, device, queue)) = default_device() else { return };
    // Three cars 12 m apart: one driving on, one into a wall slot, one dropped.
    let a = sedan(Vec3::new(0.0, 0.0, 0.0), false, 12.0, surface, 0.0);
    let b = sedan(Vec3::new(12.0, 0.0, 0.0), false, 10.0, surface, 0.5);
    let c = sedan(Vec3::new(24.0, 0.6, 0.0), true, 6.0, surface, 0.0);
    let cars = [a, b, c];
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, &cars, Some(CAP));
    if surface {
        write_proxies(&gpu, &queue, 0, &pole(Vec3::new(0.4, 0.9, 0.3))).unwrap();
        write_proxies(&gpu, &queue, 1, &pole(Vec3::new(12.4, 0.9, 0.3))).unwrap();
        gpu.set_coupling_on(true);
    }
    fast_paths(&mut gpu);
    let taken = compare(label, &mut gpu, frames, steps);
    assert_eq!(taken.grouped, 0, "{label}: a contact group formed");
    // (The first frames run today's schedule until the host has read a plan back.)
    assert!(taken.wide + 3 >= frames, "{label}: the wide schedule on {} of {frames} frames", taken.wide);
}

/// The first check of a new kernel version: two frames of a few steps, the plan's count
/// given as if read back.
#[test]
fn tiny_cars_apart_wide_is_bit_for_bit() {
    let Ok((_, device, queue)) = default_device() else { return };
    let cars = [sedan(Vec3::ZERO, false, 12.0, false, 0.0), sedan(Vec3::new(12.0, 0.0, 0.0), false, 10.0, false, 0.0)];
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &cars);
    fast_paths(&mut gpu);
    gpu.set_vv_wide_seen(Some(0));
    let taken = compare("tiny cars apart", &mut gpu, 2, 4);
    assert_eq!((taken.wide, taken.grouped), (2, 0));
}

#[test]
fn cars_apart_wide_is_bit_for_bit() {
    apart("three cars apart", false, 24, 17);
}

#[test]
fn cars_apart_with_surface_contact_wide_is_bit_for_bit() {
    apart("three cars apart, surface contact", true, 24, 17);
}

fn closing(label: &str, surface: bool) {
    let Ok((_, device, queue)) = default_device() else { return };
    // Two head-on from 5 m at 12 m/s each (they meet after about 0.2 s), one alone.
    let a = sedan(Vec3::new(0.0, 0.0, 0.0), false, 12.0, surface, 0.0);
    let b = sedan(Vec3::new(0.0, 0.0, 5.0), true, 12.0, surface, 0.0);
    let c = sedan(Vec3::new(14.0, 0.0, 0.0), false, 10.0, surface, 0.5);
    let cars = [a, b, c];
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, &cars, Some(CAP));
    fast_paths(&mut gpu);
    // (A small margin: the plan groups the pair only when they are close.)
    gpu.set_contact_margin(0.05);
    let taken = compare(label, &mut gpu, 40, 17);
    assert!(taken.wide > taken.wide_with_group, "{label}: no frame without a group on the wide passes ({taken:?})");
    assert!(taken.grouped > 0, "{label}: no contact group formed ({taken:?})");
    assert!(taken.wide_with_group > 0, "{label}: no frame held the wide schedule while the plan formed a group ({taken:?})");
}

#[test]
fn cars_closing_into_a_group_are_bit_for_bit() {
    closing("a pair closing head-on and one alone", false);
}

#[test]
fn cars_closing_into_a_group_with_surface_contact_are_bit_for_bit() {
    closing("a pair closing head-on and one alone, surface contact", true);
}

/// Margin 0 and a vehicle spawned at rest in the air: its nodes fall out of their frame
/// bounds (the box itself) within the frame, and the wide passes report it as the solo
/// frame does (explicit_contact_gpu's a_node_leaving_its_frame_bounds_is_reported).
#[test]
fn a_node_leaving_its_frame_bounds_is_reported_from_the_wide_passes() {
    let Ok((_, device, queue)) = default_device() else { return };
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: false,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
        centred_damping: false,
        centred_objective: false,
    };
    let body = |points: &[Vec3]| {
        let mut s = ExplicitBeams::empty(settings);
        s.ground = -1.0e3;
        for &p in points {
            s.nodes.push(Node { pos: p, vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 1.0, flags: 0 });
        }
        s
    };
    let mut a = body(&[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 2.0), Vec3::new(3.0, 3.0, 3.0)]);
    a.triangles.push(Triangle { nodes: [0, 1, 2], wait: 0, idle: 0 });
    a.finish();
    let mut b = body(&[Vec3::new(50.0, 0.0, 0.0)]);
    b.finish();
    let fleet = [a, b];
    // (Both fall; the status names the highest vehicle outside its bounds. A third
    // fleet whose first vehicle alone can fall: the report names it.)
    let mut pinned = fleet.clone();
    for n in &mut pinned[1].nodes {
        n.flags |= engine_softbody::explicit_beams::NODE_IMMOVABLE;
    }
    for (fleet, who) in [(&fleet, "vehicle 1"), (&pinned, "vehicle 0")] {
        let errors: Vec<String> = [false, true]
            .into_iter()
            .map(|wide| {
                let mut g = ExplicitBeamsGpu::new(&device, &queue, fleet);
                fast_paths(&mut g);
                g.set_vv_wide(wide);
                g.set_vv_wide_seen(Some(0));
                g.set_contact_margin(0.0);
                g.frame(17);
                g.wait();
                assert_eq!(g.vv_wide_used(), wide);
                g.contact_status().unwrap_err()
            })
            .collect();
        assert!(errors[1].contains(&format!("{who} left its frame bounds")), "the wide passes: {}", errors[1]);
        assert_eq!(errors[0], errors[1], "the solo frame's report and the wide passes'");
    }
}
