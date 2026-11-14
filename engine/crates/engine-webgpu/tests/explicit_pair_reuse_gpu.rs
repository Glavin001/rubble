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
