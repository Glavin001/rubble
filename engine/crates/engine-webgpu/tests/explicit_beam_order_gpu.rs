//! The beam phase's records sorted by kind (`explicit_plan::W_BEAM_ORDER`, the kernel's
//! beamOrder: the hot JBeam beams on their packed records, the rest by kind, a
//! two-record element on one lane) against the beams in file order (beamUpdate for
//! each): each frame from one saved state, every state word the same, bit for bit. The
//! order changes which lane evaluates a beam, never what it computes.
//!
//! Vehicles: the generated sedan (in memory, as the rigger exports it: JBeam plain,
//! support and bounded beams, couplers, torsion bars, extension records, aero
//! triangles) into a wall at 56 km/h (yields, breaks and group cascades), and beside a
//! Rigs of Rods truck (plain kinds, the cold path) in one upload.

use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

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

fn sedan_desc() -> SoftVehicleDesc {
    let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
    let r = lattice::rig_with(&inp);
    let files = vg_input::physics_files(&inp, &r);
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = texts
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = texts.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| b.as_str());
    engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
}

/// The generated sedan facing +Z at `at` (its lowest node 2 cm up), driving at `speed`
/// into a wall `gap` m ahead of its front (no wall with `gap` 0).
fn sedan(at: Vec3, speed: f32, gap: f32) -> ExplicitBeams {
    let d = sedan_desc();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let v = Vec3::Z * speed;
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings).expect("spawn");
    let f = engine_softbody::pack::forward_axis(&d);
    let q = Quat::from_rotation_y(-f.x.atan2(f.z));
    let o = s.origin;
    s.origin = Vec3::ZERO;
    for n in &mut s.nodes {
        n.pos = q * (n.pos + o);
        n.vel = v;
    }
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos += Vec3::new(at.x, 0.02 - low, at.z);
    }
    s.roll_tyres(v);
    if gap > 0.0 {
        let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(Vec3::new(at.x, 2.0, front + gap + 0.5), Vec3::new(3.0, 2.0, 0.5)));
    }
    s
}

fn truck(at: Vec3) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("the RoR sedan");
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let mut s = ExplicitBeams::from_truck_config(&text, None, at, Vec3::ZERO, settings).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s
}

/// Per segment (frames, steps a frame), each frame from one saved state: the sorted
/// order twice (deterministic) and file order, every state word the same; continues
/// from the sorted order's frame. Returns the broken beams at the end.
fn compare(label: &str, fleet: &[ExplicitBeams], segments: &[(u32, u32)]) -> Option<usize> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, fleet);
    gpu.set_wide(false);
    assert_ne!(
        gpu.layout().words[engine_exec::explicit_plan::W_BEAM_ORDER],
        0,
        "{label}: the sorted beam records are uploaded"
    );
    let run = |gpu: &mut ExplicitBeamsGpu, snap: &engine_webgpu::GpuSnapshot, sorted: bool, steps: u32| {
        gpu.restore(snap);
        gpu.set_beam_order(sorted);
        gpu.frame(steps);
        gpu.wait();
        gpu.state_words()
    };
    let mut f = 0;
    for &(frames, steps) in segments {
        for _ in 0..frames {
            let snap = gpu.snapshot();
            let file = run(&mut gpu, &snap, false, steps);
            let sorted = run(&mut gpu, &snap, true, steps);
            if let Some(d) = first_diff(&sorted, &run(&mut gpu, &snap, true, steps)) {
                panic!("{label}: frame {f}: the sorted beam order is not deterministic: {d}");
            }
            if let Some(d) = first_diff(&file, &sorted) {
                panic!("{label}: frame {f}: the sorted beam order differs from file order at {d}");
            }
            f += 1;
        }
    }
    let beams = gpu.beams();
    let broken: usize = beams.iter().map(|v| v.iter().filter(|b| b.broken).count()).sum();
    let bent: usize = beams.iter().map(|v| v.iter().filter(|b| b.bent()).count()).sum();
    eprintln!("{label}: {f} frames bit for bit; {broken} beams broken, {bent} yielded");
    Some(broken)
}

#[test]
fn generated_sedan_into_a_wall_sorted_order_is_file_order() {
    let car = sedan(Vec3::ZERO, 15.6, 0.3);
    let Some(broken) = compare("generated sedan wall", &[car], &[(2, 17), (120, 1), (40, 17)]) else {
        return;
    };
    assert!(broken > 0, "the crash breaks beams (so breaks were compared)");
}

#[test]
fn sedan_and_truck_in_one_upload_sorted_order_is_file_order() {
    let fleet = [sedan(Vec3::ZERO, 0.0, 0.0), truck(Vec3::new(10.0, 0.0, 0.0))];
    compare("sedan and truck at rest", &fleet, &[(30, 17)]);
}
