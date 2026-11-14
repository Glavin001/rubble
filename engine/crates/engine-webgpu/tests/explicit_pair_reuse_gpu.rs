//! The self-collision grid walk's pair list built once and reused over steps
//! (`explicit_beams.slang` pairReuse*, lossless) against the walk listing its pairs every
//! step: each frame from the same saved state, every state word the same, and the list
//! reused on most steps (so the reuse path is what was compared).
//!
//! Vehicles: the procedural sedan into a wall, and with anti-clip (§36: every triangle's
//! box padded by its own and the vehicle's fastest node's travel) into a wall and dropped
//! tumbling.

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

/// `frames` frames of `steps` steps, each from one saved state with the reuse off then
/// on; continues from the reuse's frame. Returns the reuse statistics.
fn compare(label: &str, fleet: &[ExplicitBeams], frames: u32, steps: u32) -> Option<[u64; 4]> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, fleet);
    gpu.set_wide(false);
    gpu.take_pair_reuse_stats();
    for f in 0..frames {
        let snap = gpu.snapshot();
        gpu.restore(&snap);
        gpu.set_pair_reuse(false);
        gpu.frame(steps);
        gpu.wait();
        let walk = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_pair_reuse(true);
        gpu.frame(steps);
        gpu.wait();
        let reuse = gpu.state_words();
        if let Some(d) = first_diff(&walk, &reuse) {
            panic!("{label}: frame {f}: the reused pair list differs from the walk at {d}");
        }
    }
    let s = gpu.take_pair_reuse_stats();
    eprintln!(
        "{label}: {frames} frames of {steps} steps bit for bit; lists built {}, steps reused {}, pairs a build {:.0}, most {}",
        s[0],
        s[1],
        s[2] as f64 / s[0].max(1) as f64,
        s[3]
    );
    Some(s)
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

/// Lowest node `lift` m up at `at`, moving at `vel`, with a wall 1 m ahead along +z.
fn spawn(text: &str, at: Vec3, vel: Vec3, lift: f32, wall: bool) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings::default();
    let mut s = ExplicitBeams::from_truck_config(text, None, at, vel, settings).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += lift - low;
    }
    s.roll_tyres(vel);
    if wall {
        let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            at + Vec3::new(0.0, 1.0, front + 1.0 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
    }
    s.finish();
    s
}

/// The list was built and reused (`most`: on more steps than it was built).
fn assert_reused(label: &str, s: [u64; 4], most: bool) {
    assert!(s[0] > 0, "{label}: no list was built (the grid's pair list is not in use)");
    let enough = if most { s[1] > s[0] } else { s[1] > 0 };
    assert!(enough, "{label}: reused on {} steps for {} builds", s[1], s[0]);
}

#[test]
fn reused_pair_list_is_the_walk_bit_for_bit() {
    let o = Vec3::new(20.0, 0.0, -30.0);
    // The procedural sedan at 100 km/h into a wall: 30 frames of 34 steps. (The
    // repository's RoR sedan lists no contacters: no self-collision.)
    let s = spawn(&gen_sedan(), o, Vec3::new(0.0, 0.0, 27.78), 0.02, true);
    if let Some(st) = compare("procedural sedan wall", std::slice::from_ref(&s), 30, 34) {
        assert_reused("procedural sedan wall", st, true);
    }
    // The procedural sedan with anti-clip at 72 km/h into a wall.
    let mut g = spawn(&gen_sedan(), o, Vec3::new(0.0, 0.0, 20.0), 0.02, true);
    g.anti_clip = true;
    g.fixed_contact = true;
    g.finish();
    if let Some(st) = compare("procedural sedan wall, anti-clip", std::slice::from_ref(&g), 30, 34) {
        assert_reused("procedural sedan wall, anti-clip", st, true);
    }
    // Dropped from 3 m tumbling (4 rad/s about z) onto the ground, anti-clip: the spin
    // moves nodes up to 10 m/s relative to one another (5 mm a step, the 5 cm margin
    // within a frame), so the list is mostly built again.
    let mut d = spawn(&gen_sedan(), o, Vec3::ZERO, 3.0, false);
    let c = d.nodes.iter().map(|n| n.pos).sum::<Vec3>() / d.nodes.len() as f32;
    for n in &mut d.nodes {
        n.vel = Vec3::new(0.0, -4.0, 0.0) + Vec3::Z.cross(n.pos - c) * 4.0;
    }
    d.anti_clip = true;
    d.fixed_contact = true;
    d.finish();
    if let Some(st) = compare("procedural sedan tumbling drop, anti-clip", std::slice::from_ref(&d), 40, 34) {
        assert_reused("procedural sedan tumbling drop, anti-clip", st, false);
    }
}

/// A piece torn off and flying away (`ExplicitBeamsGpu::set_piece_labels`): the list with
/// the host's piece labels against the same list with none and against the walk, each frame
/// from one saved state, every state word the same; with the labels the piece no longer
/// invalidates the main body's list (reused on most steps, against few without).
#[test]
fn piece_labels_keep_the_list_bit_for_bit() {
    use engine_softbody::explicit_debris::{piece_labels, Pieces};
    let o = Vec3::new(20.0, 0.0, -30.0);
    // The procedural sedan in free fall 40 m up with anti-clip, every triangle tested every
    // step (no throttle: the grid each step); its front 0.5 m torn off (every beam across
    // the cut broken) and flung sideways and down at 60 m/s.
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
    for b in &mut d.beams {
        if cut[b.a as usize] != cut[b.b as usize] {
            b.broken = true;
        }
    }
    for (n, &c) in d.nodes.iter_mut().zip(&cut) {
        if c {
            n.vel = Vec3::new(50.0, -33.0, 0.0);
        }
    }
    let broken: Vec<bool> = d.beams.iter().map(|b| b.broken).collect();
    let pieces = Pieces::of(&d, &broken);
    let labels = piece_labels(&d, &pieces, &broken);
    assert!(labels.iter().any(|&l| l != 0), "the cut leaves no piece off the main body");
    let Some((_, device, queue)) = default_device().ok() else { return };
    let fleet = std::slice::from_ref(&d);
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, fleet);
    gpu.set_wide(false);
    let zeros = vec![0u8; d.nodes.len()];
    let (mut with, mut without) = ([0u64; 4], [0u64; 4]);
    // The first frames the piece is still beside the cut; from frame 8 it is metres away.
    let (mut late_with, mut late_without) = ([0u64; 2], [0u64; 2]);
    for f in 0..20 {
        let snap = gpu.snapshot();
        let mut run = |gpu: &mut ExplicitBeamsGpu, reuse: bool, l: &[u8]| {
            gpu.restore(&snap);
            gpu.set_pair_reuse(reuse);
            gpu.set_piece_labels(0, l);
            gpu.take_pair_reuse_stats();
            gpu.frame(34);
            gpu.wait();
            let st = gpu.take_pair_reuse_stats();
            gpu.set_piece_labels(0, &zeros);
            gpu.set_pair_reuse(true);
            (gpu.state_words(), st)
        };
        let (walk, _) = run(&mut gpu, false, &zeros);
        let (none, s0) = run(&mut gpu, true, &zeros);
        let (lab, s1) = run(&mut gpu, true, &labels);
        if let Some(diff) = first_diff(&walk, &none) {
            panic!("torn piece: frame {f}: the list without labels differs from the walk at {diff}");
        }
        if let Some(diff) = first_diff(&walk, &lab) {
            panic!("torn piece: frame {f}: the list with piece labels differs from the walk at {diff}");
        }
        for k in 0..2 {
            with[k] += s1[k];
            without[k] += s0[k];
            if f >= 8 {
                late_with[k] += s1[k];
                late_without[k] += s0[k];
            }
        }
    }
    eprintln!(
        "torn piece: 20 frames of 34 steps bit for bit; with labels built {} reused {}, without built {} reused {}; frames 8..20: with built {} reused {}, without built {} reused {}",
        with[0], with[1], without[0], without[1], late_with[0], late_with[1], late_without[0], late_without[1]
    );
    assert!(with[1] > with[0], "with labels the list was reused on {} steps for {} builds", with[1], with[0]);
    // Once the piece is away, the labelled list is built a few times a frame (the cut
    // body's own ringing; measured 45 builds in 12 frames, 136 without the labels).
    assert!(late_with[0] <= 5 * 12, "frames 8..20 with labels: {} builds", late_with[0]);
    assert!(late_without[0] > 2 * late_with[0], "frames 8..20 builds: with labels {}, without {}", late_with[0], late_without[0]);
}
