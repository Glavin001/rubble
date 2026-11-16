//! Contact groups split across workgroups (`ExplicitBeamsGpu::set_group_split`,
//! `explicit_beams_split.slang`, engine/docs/measurements/contact-groups-design.md): each
//! substep the members' steps in a workgroup per vehicle, then the groups' contact passes
//! (their triangle tests across workgroups), instead of a group's whole frame in one
//! workgroup. A change of how, not what: every frame from one saved state with the group's
//! workgroup, then split (workgroup memory not zeroed natively), split with each group's
//! pass whole in its workgroup, and split with workgroup memory zeroed: every state word
//! the same.

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

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

/// The generated sedan (low resolution: a self-collision grid of its own), spawned at
/// `at` turned `yaw` degrees, moving at `vel`, its lowest node 2 cm off the ground.
fn sedan(text: &str, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = if yaw == 0.0 {
        ExplicitBeams::from_truck_config(text, None, at, vel, settings)
    } else {
        ExplicitBeams::from_truck_config_yaw(text, None, at, yaw, vel, settings)
    }
    .expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(vel);
    s.finish();
    s
}

/// `frames` frames of `steps` steps, each from one saved state three ways: the groups'
/// own workgroup, split, split with workgroup memory zeroed. Returns the frames that
/// had a contact group and the largest group seen.
fn split_matches(fleet: Vec<ExplicitBeams>, frames: u32, steps: u32) -> (u32, u32) {
    let Some((_, device, queue)) = default_device().ok() else { return (u32::MAX, 0) };
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    let (mut grouped, mut largest) = (0u32, 0u32);
    for f in 0..frames {
        let snap = gpu.snapshot();
        gpu.set_group_split(false);
        gpu.frame(steps);
        gpu.wait();
        let old = gpu.state_words();
        let sizes = gpu.contact_group_sizes();
        for (zero, tests) in [(false, true), (false, false), (true, true)] {
            gpu.restore(&snap);
            gpu.set_group_split(true);
            gpu.set_group_split_zero(zero);
            gpu.set_group_split_tests(tests);
            gpu.frame(steps);
            gpu.wait();
            let new = gpu.state_words();
            if let Some(d) = first_diff(&old, &new) {
                panic!("frame {f} (groups {sizes:?}): split (zeroing {zero}, tests across workgroups {tests}) differs from the groups' workgroup at {d}");
            }
        }
        gpu.set_group_split_zero(false);
        gpu.set_group_split_tests(true);
        grouped += u32::from(!sizes.is_empty());
        largest = largest.max(sizes.iter().copied().max().unwrap_or(0));
    }
    (grouped, largest)
}

fn generated() -> String {
    build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Low, ..Default::default() }).unwrap().truck
}

#[test]
fn two_sedans_head_on_split_is_bit_for_bit() {
    let text = generated();
    let a = sedan(&text, Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 12.0));
    let front = a.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let probe = sedan(&text, Vec3::ZERO, 180.0, Vec3::ZERO);
    let back = probe.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
    let b = sedan(&text, Vec3::new(0.0, 0.0, front + 1.0 - back), 180.0, Vec3::new(0.0, 0.0, -12.0));
    let (grouped, largest) = split_matches(vec![a, b], 40, 17);
    if grouped == u32::MAX {
        return;
    }
    eprintln!("two sedans head-on: 40 frames bit for bit, {grouped} with a contact group (largest {largest})");
    assert!(grouped > 10, "the cars met in a contact group ({grouped} frames)");
}

/// Six sedans nose to tail in a column, the last pushed into the others: groups past
/// four members (the larger groups' pass, members one after another before) and the
/// small ones as the column closes.
#[test]
fn a_column_of_six_split_is_bit_for_bit() {
    let text = generated();
    let probe = sedan(&text, Vec3::ZERO, 0.0, Vec3::ZERO);
    let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
    let pitch = hi - lo + 0.3;
    let fleet: Vec<ExplicitBeams> = (0..6)
        .map(|i| {
            let vel = if i == 0 { Vec3::new(0.0, 0.0, 15.0) } else { Vec3::ZERO };
            sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, vel)
        })
        .collect();
    let (grouped, largest) = split_matches(fleet, 30, 17);
    if grouped == u32::MAX {
        return;
    }
    eprintln!("a column of six: 30 frames bit for bit, {grouped} with a contact group, largest {largest} members");
    assert!(grouped > 10, "the cars met in a contact group ({grouped} frames)");
    assert!(largest > 4, "a group past four members formed (largest {largest})");
}
