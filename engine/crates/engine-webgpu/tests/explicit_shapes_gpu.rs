//! Static shapes beyond the axis-aligned box on the GPU (the kernel's
//! `nodeUpdateShapes`) against the CPU reference (`StaticShape`,
//! `engine_softbody::explicit_beams`):
//! * a real car (the Rigs of Rods Viper, with `RUBBLE_INSPIRATION=<dir>`; skipped
//!   otherwise) at 100 km/h up a 10° rotated ramp and into a round pole: per frame
//!   from the CPU's state, the GPU frame within float noise of the CPU with the
//!   GPU's fixed-point self-collision accumulators modelled (as
//!   `explicit_beams_bitwise.rs`'s `cpu_reference_per_frame`);
//! * a box given as an identity-rotation `StaticShape` (the shapes path) and as a
//!   `StaticBox` (the plain path): the same GPU state, bit for bit, every frame.

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape,
};
use engine_softbody::truck::parse_truck;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

/// Steps per frame (120 fps).
const STEPS: u32 = 17;
const KMH100: f32 = 100.0 / 3.6;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let (_, d, q) = default_device().ok()?;
    Some((d, q))
}

fn repo_sedan() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan")
}

/// A Rigs of Rods content file under `RUBBLE_INSPIRATION`, if set and present.
fn ror(rel: &str) -> Option<String> {
    let Ok(dir) = std::env::var("RUBBLE_INSPIRATION") else {
        eprintln!("skip: RUBBLE_INSPIRATION not set");
        return None;
    };
    let text = std::fs::read_to_string(format!("{dir}/{rel}")).ok();
    if text.is_none() {
        eprintln!("skip: {dir}/{rel} not found");
    }
    text
}

/// Spawned at `at` with velocity `vel`, lowest node 2 cm up, tyres rolling; no air
/// drag or turbulence (the GPU's turbulence draws are its own sequence).
fn spawn(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings::default();
    let mut sim = ExplicitBeams::from_truck_config(text, None, at, vel, settings).expect("spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(vel);
    sim
}

/// Forward (from the first cinecam, else −x) and lateral axes, and the node
/// extents along them (relative to the spawn point).
fn axes(text: &str) -> (Vec3, Vec3, (f32, f32), (f32, f32)) {
    let sim = spawn(text, Vec3::ZERO, Vec3::ZERO);
    let file = parse_truck(text);
    let p = |i: u32| sim.nodes[i as usize].pos;
    let forward = file
        .cameras
        .as_ref()
        .map(|c| {
            let d = p(c.center) - p(c.back);
            Vec3::new(d.x, 0.0, d.z).normalize()
        })
        .unwrap_or(-Vec3::X);
    let lateral = Vec3::Y.cross(forward).normalize();
    let span = |axis: Vec3| {
        sim.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
            (a.min(n.pos.dot(axis)), b.max(n.pos.dot(axis)))
        })
    };
    (forward, lateral, span(forward), span(lateral))
}

/// One CPU step with the GPU's self-collision accumulator (1/32 N fixed point),
/// as `explicit_beams_bitwise.rs`'s `step_fixed_point`.
fn step_fixed_point(cpu: &mut ExplicitBeams) {
    let mut bare = cpu.clone();
    bare.candidates.clear();
    cpu.step();
    bare.step();
    let q = |x: f32| (x * 32.0).round() / 32.0;
    for (n, b) in cpu.nodes.iter_mut().zip(&bare.nodes) {
        let c = n.force - b.force;
        n.force = b.force + Vec3::new(q(c.x), q(c.y), q(c.z));
    }
}

#[test]
fn viper_up_a_rotated_ramp_into_a_pole_matches_cpu() {
    let Some(text) = ror("ror-repo/viper/ViperGTS.car") else {
        return;
    };
    let Some((device, queue)) = device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let o = Vec3::new(100.0, 0.0, 100.0);
    let (f, l, along, across) = axes(&text);
    let mut cpu = spawn(&text, o, f * KMH100);
    // A 4 m ramp rising 10° along forward, its top face's leading edge on the
    // ground 0.5 m ahead of the front, a metre wider than the car each side.
    let slope = 10f32.to_radians();
    let mut q = Quat::from_axis_angle(l, -slope);
    if (q * f).y < 0.0 {
        q = q.inverse();
    }
    let (len, thick, side) = (2.0, 0.5, 0.5 * (across.1 - across.0) + 1.0);
    let mid = o.dot(l) + 0.5 * (across.0 + across.1);
    let edge = f * (o.dot(f) + along.1 + 0.5) + l * mid;
    let up_slope = q * f;
    let normal = q * Vec3::Y;
    let centre = edge + up_slope * len - normal * thick;
    let half = if f.x.abs() > 0.5 {
        Vec3::new(len, thick, side)
    } else {
        Vec3::new(side, thick, len)
    };
    cpu.shapes.push(StaticShape::oriented_box(centre, half, q));
    // A 0.15 m pole 3 m past the ramp's end, 0.3 m off the car's centre line.
    let pole_at = edge + f * (2.0 * len * slope.cos() + 3.0) + l * 0.3 + Vec3::Y * 1.5;
    cpu.shapes.push(StaticShape::cylinder(pole_at, 0.15, 1.5));

    let frames = 70;
    let (mut worst, mut top, mut pole_hit) = (0.0f32, f32::MIN, false);
    let mut sum = 0.0f32;
    for frame in 0..frames {
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
        gpu.set_wide(false);
        gpu.frame(STEPS);
        gpu.wait();
        let end = gpu.positions().remove(0);
        cpu.check_wings();
        cpu.rebase_origin();
        for _ in 0..STEPS {
            step_fixed_point(&mut cpu);
        }
        let c = cpu.positions();
        let rms = (end
            .iter()
            .zip(&c)
            .map(|(a, b)| (*a - *b).length_squared())
            .sum::<f32>()
            / c.len() as f32)
            .sqrt();
        worst = worst.max(rms);
        sum += rms;
        top = top.max(c.iter().map(|p| p.y).fold(f32::MIN, f32::max));
        for p in &c {
            let d = *p - pole_at;
            if Vec3::new(d.x, 0.0, d.z).length() < 0.3 && d.y.abs() < 1.5 {
                pole_hit = true;
            }
        }
        if frame % 10 == 9 {
            eprintln!(
                "frame {}: GPU vs fixed-point CPU node RMS {:.5} mm",
                frame + 1,
                rms * 1e3
            );
        }
    }
    let speed = cpu.nodes.iter().map(|n| n.vel.dot(f)).sum::<f32>() / cpu.nodes.len() as f32;
    let (broken, bent) = cpu.damage();
    eprintln!(
        "Viper ramp + pole, {frames} frames: per-frame RMS mean {:.5} mm, max {:.5} mm; \
         highest node {top:.2} m, reached the pole {pole_hit}, forward speed {speed:.1} m/s, \
         broken {broken}, bent {bent}",
        sum / frames as f32 * 1e3,
        worst * 1e3
    );
    // The car went up the ramp (above the ramp's top edge height, ~0.7 m) and into
    // the pole, which slowed it and bent its front.
    assert!(top > 0.8, "never climbed: highest node {top} m");
    assert!(pole_hit, "never reached the pole");
    assert!(
        speed < 0.9 * KMH100,
        "the pole did not slow it: {speed} m/s"
    );
    assert!(bent > 20, "the pole bent nothing ({bent})");
    // Float noise from the fixed-point CPU (cpu_reference_per_frame's bound).
    assert!(
        worst < 2e-5,
        "a frame {} mm RMS from the fixed-point CPU",
        worst * 1e3
    );
}

#[test]
fn identity_rotation_box_is_the_plain_box_on_the_gpu_bit_for_bit() {
    let s = repo_sedan();
    let Some((device, queue)) = device() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let o = Vec3::new(100.0, 0.0, 100.0);
    let (f, l, along, across) = axes(&s);
    let car = spawn(&s, o, f * KMH100);
    let mid = o.dot(l) + 0.5 * (across.0 + across.1);
    let centre = f * (o.dot(f) + along.1 + 1.5) + l * mid + Vec3::Y * 3.0;
    let side = 0.5 * (across.1 - across.0) + 1.0;
    let half = if f.x.abs() > 0.5 {
        Vec3::new(0.5, 3.0, side)
    } else {
        Vec3::new(side, 3.0, 0.5)
    };
    let mut plain = car.clone();
    plain.boxes.push(StaticBox::centered(centre, half));
    let mut shaped = car;
    shaped
        .shapes
        .push(StaticShape::oriented_box(centre, half, Quat::IDENTITY));
    let run = |v: &ExplicitBeams| {
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(v));
        gpu.set_wide(false);
        let mut out = Vec::new();
        for _ in 0..40 {
            gpu.frame(STEPS);
            gpu.wait();
            out.push(gpu.state_words());
        }
        out
    };
    let (a, b) = (run(&plain), run(&shaped));
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        for ((name, p), (_, q)) in x.iter().zip(y) {
            assert!(p == q, "frame {i}: {name} differs");
        }
    }
    eprintln!("identity-rotation box = plain box: 40 frames bit for bit");
}
