//! Self-collision's listed pairs tested across workgroups (`explicit_beams_selfsplit.slang`,
//! engine/docs/measurements/selfcol-design.md) against the same tests in each vehicle's
//! workgroup: each frame from one saved state with the split off then on, every state word
//! the same, and the split path (with the pair list reused) in use.
//!
//! Adversarial cases: a spin at 20 rad/s free and into a wall, a piece torn off mid-list
//! (with the host's piece labels), nodes retired mid-run, a triangle larger than the
//! grid's 64 cells, a node at 250 m/s through the body, and a fleet of four.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticBox};
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

/// Between frames (frame index, the GPU): the host's changes (retirement, labels).
type Hook<'a> = &'a mut dyn FnMut(u32, &mut ExplicitBeamsGpu);

/// `frames` frames of `steps` steps, each from one saved state with the split off then on;
/// continues from the split's frame. Returns (frames the split ran, lists built, reused).
fn compare(label: &str, fleet: &[ExplicitBeams], frames: u32, steps: u32, hook: Hook) -> Option<(u32, u64, u64)> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, fleet);
    gpu.set_wide(false);
    gpu.take_pair_reuse_stats();
    let mut split = 0;
    for f in 0..frames {
        hook(f, &mut gpu);
        let snap = gpu.snapshot();
        gpu.restore(&snap);
        gpu.set_self_split(false);
        gpu.frame(steps);
        gpu.wait();
        assert!(!gpu.self_split_used(), "{label}: the split ran with it off");
        let whole = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_self_split(true);
        gpu.frame(steps);
        gpu.wait();
        split += gpu.self_split_used() as u32;
        let cut = gpu.state_words();
        if let Some(d) = first_diff(&whole, &cut) {
            panic!("{label}: frame {f}: the split tests differ from the workgroup's at {d}");
        }
    }
    let s = gpu.take_pair_reuse_stats();
    eprintln!(
        "{label}: {frames} frames of {steps} steps bit for bit ({split} split); lists built {}, steps reused {}, pairs a build {:.0}",
        s[0],
        s[1],
        s[2] as f64 / s[0].max(1) as f64,
    );
    Some((split, s[0], s[1]))
}

fn gen_sedan() -> String {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    build(
        &style("sedan").unwrap(),
        &BuildParams {
            resolution: Resolution::Low,
            ..Default::default()
        },
    )
    .unwrap()
    .truck
}

/// The procedural sedan with anti-clip, its lowest node `lift` m up at `at`, moving at
/// `vel`, optionally a wall 1 m ahead along +z.
fn spawn(at: Vec3, vel: Vec3, lift: f32, wall: bool) -> ExplicitBeams {
    let mut s =
        ExplicitBeams::from_truck_config(&gen_sedan(), None, at, vel, ExplicitBeamsSettings::default()).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += lift - low;
    }
    s.roll_tyres(vel);
    if wall {
        let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(at + Vec3::new(0.0, 1.0, front + 2.0), Vec3::new(3.0, 2.0, 1.0)));
    }
    s.anti_clip = true;
    s.fixed_contact = true;
    s.finish();
    s
}

/// Spun at `w` rad/s about `axis` through its centre, plus `vel`.
fn spin(s: &mut ExplicitBeams, axis: Vec3, w: f32, vel: Vec3) {
    let c = s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / s.nodes.len() as f32;
    for n in &mut s.nodes {
        n.vel = vel + axis.normalize().cross(n.pos - c) * w;
    }
}

fn check(label: &str, r: Option<(u32, u64, u64)>, frames: u32) {
    let Some((split, built, reused)) = r else { return };
    assert_eq!(split, frames, "{label}: the split ran on {split} of {frames} frames");
    assert!(built > 0 && reused > 0, "{label}: lists built {built}, reused {reused}: the listed pairs were not tested");
}

/// The first check of a new kernel version: two frames of a few steps, one car.
#[test]
fn tiny_split_is_bit_for_bit() {
    let o = Vec3::new(20.0, 0.0, -30.0);
    let s = spawn(o, Vec3::new(0.0, 0.0, 20.0), 0.02, true);
    let r = compare("tiny", std::slice::from_ref(&s), 2, 4, &mut |_, _| {});
    if let Some((split, _, _)) = r {
        assert_eq!(split, 2, "tiny: the split ran on {split} of 2 frames");
    }
}

#[test]
fn wall_crash_split_is_bit_for_bit() {
    let o = Vec3::new(20.0, 0.0, -30.0);
    let s = spawn(o, Vec3::new(0.0, 0.0, 20.0), 0.02, true);
    check("sedan into a wall at 72 km/h", compare("sedan wall", std::slice::from_ref(&s), 30, 34, &mut |_, _| {}), 30);
}

#[test]
fn spin_free_and_into_a_wall_split_is_bit_for_bit() {
    let o = Vec3::new(20.0, 0.0, -30.0);
    // 20 rad/s about a tilted axis, 3 m up, free (then the ground).
    let mut d = spawn(o, Vec3::ZERO, 3.0, false);
    spin(&mut d, Vec3::new(0.3, 1.0, 0.2), 20.0, Vec3::new(0.0, -2.0, 0.0));
    check("spin 20 rad/s free", compare("spin free", std::slice::from_ref(&d), 30, 34, &mut |_, _| {}), 30);
    // 20 rad/s about y, at 15 m/s into the wall.
    let mut w = spawn(o, Vec3::ZERO, 0.3, true);
    spin(&mut w, Vec3::Y, 20.0, Vec3::new(0.0, 0.0, 15.0));
    check("spin 20 rad/s into a wall", compare("spin wall", std::slice::from_ref(&w), 30, 34, &mut |_, _| {}), 30);
}

#[test]
fn torn_piece_and_retired_nodes_split_is_bit_for_bit() {
    use engine_softbody::explicit_debris::{piece_labels, Pieces};
    let o = Vec3::new(20.0, 0.0, -30.0);
    // Free fall 40 m up, every triangle tested every step (no throttle); after 4 frames
    // its front 0.5 m is torn off (every beam across the cut broken) and flung at 60 m/s:
    // mid-list, labelled as its own piece from then on.
    let settings = ExplicitBeamsSettings { triangle_throttle: false, ..Default::default() };
    let mut d = ExplicitBeams::from_truck_config(&gen_sedan(), None, o, Vec3::ZERO, settings).expect("spawn");
    let low = d.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut d.nodes {
        n.pos.y += 40.0 - low;
    }
    d.anti_clip = true;
    d.fixed_contact = true;
    d.finish();
    let front = d.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let cut: Vec<bool> = d.nodes.iter().map(|n| n.pos.z > front - 0.5).collect();
    let mut torn = d.clone();
    for b in &mut torn.beams {
        if cut[b.a as usize] != cut[b.b as usize] {
            b.broken = true;
        }
    }
    for (n, &c) in torn.nodes.iter_mut().zip(&cut) {
        if c {
            n.vel = Vec3::new(50.0, -33.0, 0.0);
        }
    }
    let broken: Vec<bool> = torn.beams.iter().map(|b| b.broken).collect();
    let pieces = Pieces::of(&torn, &broken);
    let labels = piece_labels(&torn, &pieces, &broken);
    assert!(labels.iter().any(|&l| l != 0), "the cut leaves no piece off the main body");
    let r = compare("torn piece", std::slice::from_ref(&torn), 20, 34, &mut |f, g| {
        if f == 2 {
            g.set_piece_labels(0, &labels);
        }
    });
    check("torn piece", r, 20);
    // Retired mid-run: a third of the torn piece's nodes at frame 6.
    let retired: Vec<bool> = cut.iter().enumerate().map(|(i, &c)| c && i % 3 == 0).collect();
    let r = compare("retired nodes", std::slice::from_ref(&torn), 16, 34, &mut |f, g| {
        if f == 2 {
            g.set_piece_labels(0, &labels);
        }
        if f == 6 {
            g.set_retired(0, &retired).expect("retire");
        }
    });
    check("retired nodes", r, 16);
}

#[test]
fn stretched_triangle_and_fast_node_split_is_bit_for_bit() {
    let o = Vec3::new(20.0, 0.0, -30.0);
    // One collision triangle's node 6 m out (its box past the grid's 64 cells: every
    // candidate tested), its beams broken so it stays out; and one contacter at 250 m/s
    // through the body.
    let mut s = spawn(o, Vec3::new(0.0, 0.0, 10.0), 0.02, true);
    let far = s.triangles[0].nodes[0];
    let fast = *s.candidates.iter().find(|&&c| !s.triangles[0].nodes.contains(&c)).expect("a contacter");
    for b in &mut s.beams {
        if b.a == far || b.b == far || b.a == fast || b.b == fast {
            b.broken = true;
        }
    }
    s.nodes[far as usize].pos += Vec3::new(0.0, 6.0, 0.0);
    let c = s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / s.nodes.len() as f32;
    let to = (c - s.nodes[fast as usize].pos).normalize();
    s.nodes[fast as usize].vel = to * 250.0;
    check("stretched triangle, fast node", compare("stretched, fast", std::slice::from_ref(&s), 20, 34, &mut |_, _| {}), 20);
}

#[test]
fn fleet_split_is_bit_for_bit() {
    // Four sedans side by side into walls at different speeds: one pairs dispatch for all.
    let fleet: Vec<ExplicitBeams> = (0..4)
        .map(|k| {
            let mut s = spawn(Vec3::new(20.0 + 8.0 * k as f32, 0.0, -30.0), Vec3::new(0.0, 0.0, 8.0 + 6.0 * k as f32), 0.02, true);
            if k % 2 == 1 {
                spin(&mut s, Vec3::Y, 6.0, Vec3::new(0.0, 0.0, 12.0));
            }
            s
        })
        .collect();
    check("fleet of four", compare("fleet of four", &fleet, 20, 34, &mut |_, _| {}), 20);
}
