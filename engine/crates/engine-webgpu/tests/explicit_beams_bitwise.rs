//! The explicit-beams GPU kernel against its frozen baseline
//! (`slang/kernels/explicit_beams_baseline.slang`, the kernel before the GPU
//! parallelism work), bit for bit, frame by frame, on the same GPU state.
//!
//! Native wgpu worlds are not bit-reproducible run to run (the kernel has two known
//! scheduling races), so two worlds stepped in lockstep drift. Instead each frame
//! starts from one saved state and runs three times: the baseline, the baseline
//! with its race probe (which reports whether the frame met a race), and the
//! current kernel. A frame without a race must leave exactly the baseline's state:
//! every node record, beam record, incidence force, self-collision accumulator,
//! group slot, throttle and per-vehicle record (scratch whose order follows the lanes
//! is left out). The next frame starts from the current kernel's state.
//!
//! Every frame is also run on both kernels compiled with `EB_DET_HITS` (tests only):
//! self-collision and vehicle-contact hits on one lane in a fixed order, so there is
//! no race and every frame must match exactly, including heavy crashes (the E36's
//! self-collision races in nearly every frame of the plain build).
//!
//! The races (both documented in `engine/docs/explicit-beams.md`): a contact
//! candidate whose force is read while another hit of the same phase adds to it,
//! and the break exception's live-beam count read in the beam phase that breaks a
//! beam at that node.
//!
//! Vehicles: the repository's sedan (`rubble_sedan_ror.truck`), the procedural
//! sedan, and with `RUBBLE_INSPIRATION=<dir>` (the local, gitignored `inspiration/`
//! folder; tests skip them otherwise) the Viper, E36, Mustang and Agora S from Rigs
//! of Rods' content (Viper and E36 also need `RUBBLE_AIRFOIL_DIR`).

use engine_softbody::explicit_beams::{yaw_offset, ExplicitBeams, ExplicitBeamsSettings, StaticBox};
use engine_softbody::truck::parse_truck;
use engine_webgpu::{default_device, ExplicitBeamsGpu, KernelVariant};
use glam::Vec3;

/// Steps per frame (120 fps).
const STEPS: u32 = 17;

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
    let dir = std::env::var("RUBBLE_INSPIRATION").ok()?;
    let text = std::fs::read_to_string(format!("{dir}/{rel}")).ok();
    if text.is_none() {
        eprintln!("skip: {dir}/{rel} not found");
    }
    text
}

const VIPER: &str = "ror-repo/viper/ViperGTS.car";
const E36: &str = "ror-repo/e36/E36Sedan.truck";
const MUSTANG: &str = "ror-repo/mustang/1982-Ford-Mustang-GT-351W.truck";
const AGORA_S: &str = "ror-content/agora/95bbUID-agoras.truck";

/// The forward axis (from the first cinecam, else −x), lateral axis and extents.
struct Geo {
    forward: Vec3,
    lateral: Vec3,
    along: (f32, f32),
    across: (f32, f32),
}

fn geo(text: &str) -> Geo {
    let sim = spawn(text, Vec3::ZERO, 0.0, Vec3::ZERO);
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
    Geo {
        forward,
        lateral,
        along: span(forward),
        across: span(lateral),
    }
}

/// Spawned at `at` (turned by `yaw` degrees), lowest node 2 cm up, tyres rolling.
fn spawn(text: &str, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: true,
        turbulence: true,
        ..Default::default()
    };
    let at = Vec3::new(at.x, 0.0, at.z);
    let mut sim = if yaw == 0.0 {
        ExplicitBeams::from_truck_config(text, None, at, vel, settings)
    } else {
        ExplicitBeams::from_truck_config_yaw(text, None, at, yaw, vel, settings)
    }
    .expect("spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(vel);
    sim
}

/// Driving at `speed` into its own wall `gap` m ahead of its front, at `origin`.
fn into_wall(text: &str, origin: Vec3, speed: f32, gap: f32) -> ExplicitBeams {
    let g = geo(text);
    let mut v = spawn(text, origin, 0.0, g.forward * speed);
    let face = origin.dot(g.forward) + g.along.1 + gap;
    let mid = origin.dot(g.lateral) + 0.5 * (g.across.0 + g.across.1);
    let side = 0.5 * (g.across.1 - g.across.0) + 1.0;
    let centre = g.forward * (face + 0.5) + g.lateral * mid + Vec3::Y * 3.0;
    let half = if g.forward.x.abs() > 0.5 {
        Vec3::new(0.5, 3.0, side)
    } else {
        Vec3::new(side, 3.0, 0.5)
    };
    v.boxes.push(StaticBox::centered(centre, half));
    v
}

/// Two cars meeting `gap` m apart: head-on (the second turned 180° and driving
/// towards the first) or T-bone (the second turned 90°, standing across the lane).
fn pair(text: &str, origin: Vec3, speed: f32, gap: f32, tbone: bool) -> Vec<ExplicitBeams> {
    let g = geo(text);
    let a = spawn(text, origin, 0.0, g.forward * speed);
    let yaw = if tbone { 90.0 } else { 180.0 };
    // The turned car's extent along forward and its centre across.
    let sim = spawn(text, Vec3::ZERO, 0.0, Vec3::ZERO);
    let t: Vec<Vec3> = sim.nodes.iter().map(|n| yaw_offset(n.pos, yaw)).collect();
    let lo = t.iter().map(|q| q.dot(g.forward)).fold(f32::MAX, f32::min);
    let (clo, chi) = t.iter().fold((f32::MAX, f32::MIN), |(a, b), q| {
        (a.min(q.dot(g.lateral)), b.max(q.dot(g.lateral)))
    });
    let mid = 0.5 * (g.across.0 + g.across.1);
    let at = origin + g.forward * (g.along.1 + gap - lo) + g.lateral * (mid - 0.5 * (clo + chi));
    let vel = if tbone { Vec3::ZERO } else { -g.forward * speed };
    vec![a, spawn(text, at, yaw, vel)]
}

/// Per frame: baseline, baseline with the race probe, current kernel, each from the
/// same saved state. Returns (frames compared exactly, frames with a race).
fn compare(label: &str, fleet: &[ExplicitBeams], frames: u32, contact: bool) -> Option<(u32, u32)> {
    compare_steps(label, fleet, &[(frames, STEPS)], contact)
}

/// [`compare`] over segments of (frames, steps per frame): 1-step frames isolate a
/// race to its step, so a crash's other steps are still compared.
fn compare_steps(
    label: &str,
    fleet: &[ExplicitBeams],
    segments: &[(u32, u32)],
    contact: bool,
) -> Option<(u32, u32)> {
    let (device, queue) = device()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, fleet);
    gpu.set_vehicle_contact(contact);
    gpu.set_wide(false);
    let mut exact = 0;
    let mut racy = 0;
    let mut racy_same = 0;
    let mut det = 0;
    let mut groups = 0;
    let frames: u32 = segments.iter().map(|s| s.0).sum();
    let plan: Vec<u32> = segments.iter().flat_map(|&(n, k)| std::iter::repeat_n(k, n as usize)).collect();
    for (f, &steps) in plan.iter().enumerate() {
        let snap = gpu.snapshot();
        gpu.set_kernel_variant(KernelVariant::Baseline);
        gpu.frame(steps);
        gpu.wait();
        let a = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::BaselineProbe);
        gpu.take_race_flag();
        gpu.frame(steps);
        gpu.wait();
        let race = gpu.take_race_flag();
        let a2 = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::Current);
        gpu.frame(steps);
        gpu.wait();
        let b = gpu.state_words();
        groups = groups.max(gpu.contact_groups());
        // Both kernels with the hits on one lane in a fixed order: no race, so every
        // frame must match exactly.
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::BaselineDet);
        gpu.frame(steps);
        gpu.wait();
        let c = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::CurrentDet);
        gpu.frame(steps);
        gpu.wait();
        let d = gpu.state_words();
        if let Some(x) = first_diff(&c, &d) {
            panic!("{label}: frame {f}: with fixed-order hits the current kernel differs from the baseline at {x}");
        }
        det += 1;
        // Continue from the current kernel's own (racy) frame.
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::Current);
        gpu.frame(steps);
        gpu.wait();
        assert_ne!(race, 2, "{label}: the race probe gave up at frame {f}");
        if race != 0 {
            racy += 1;
            racy_same += (first_diff(&a, &b).is_none()) as u32;
            continue;
        }
        // The probe changes no arithmetic: a race-free frame is the baseline's.
        if let Some(d) = first_diff(&a, &a2) {
            panic!("{label}: frame {f}: the probe build differs from the baseline at {d}");
        }
        if let Some(d) = first_diff(&a, &b) {
            panic!("{label}: frame {f}: the current kernel differs from the baseline at {d}");
        }
        exact += 1;
    }
    eprintln!(
        "{label}: {exact} frames bit-identical, {racy} frames with a known race ({racy_same} \
         of them identical anyway); fixed-order hits: {det} of {frames} frames bit-identical; \
         contact groups seen {groups}"
    );
    // Racy frames may differ (the documented exception); in practice the lanes of
    // one SIMD group read before any of them adds, so they usually match too.
    // Every frame was compared exactly with fixed-order hits (above).
    assert_eq!(det, frames);
    Some((exact, racy))
}

fn first_diff(a: &[(&str, Vec<u32>)], b: &[(&str, Vec<u32>)]) -> Option<String> {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if x.len() != y.len() {
            return Some(format!("{name}: lengths {} and {}", x.len(), y.len()));
        }
        if let Some(i) = x.iter().zip(y).position(|(p, q)| p != q) {
            let count = x.iter().zip(y).filter(|(p, q)| p != q).count();
            return Some(format!(
                "{name}[{i}] ({:#x} vs {:#x}; {:e} vs {:e}), {count} words differ",
                x[i],
                y[i],
                f32::from_bits(x[i]),
                f32::from_bits(y[i])
            ));
        }
    }
    None
}

const SPEED: f32 = 27.78;

#[test]
fn sedan_alone_and_into_a_wall_bitwise() {
    let s = repo_sedan();
    compare("sedan idle", &[spawn(&s, Vec3::new(100.0, 0.0, 100.0), 0.0, Vec3::ZERO)], 40, true);
    compare_steps("sedan wall", &[into_wall(&s, Vec3::new(100.0, 0.0, 100.0), SPEED, 1.0)], &[(3, STEPS), (200, 1), (60, STEPS)], true);
}

#[test]
fn sedans_head_on_and_t_bone_bitwise() {
    let s = repo_sedan();
    let o = Vec3::new(100.0, 0.0, 100.0);
    compare_steps("sedan head-on", &pair(&s, o, 0.85 * SPEED, 1.5, false), &[(2, STEPS), (300, 1), (40, STEPS)], true);
    compare_steps("sedan T-bone", &pair(&s, o, 0.85 * SPEED, 1.0, true), &[(2, STEPS), (300, 1), (40, STEPS)], true);
}

fn real_single(rel: &str, name: &str) {
    let Some(t) = ror(rel) else { return };
    let o = Vec3::new(100.0, 0.0, 100.0);
    compare(&format!("{name} idle"), &[spawn(&t, o, 0.0, Vec3::ZERO)], 30, true);
    compare(&format!("{name} drive"), &[spawn(&t, o, 0.0, geo(&t).forward * SPEED)], 30, true);
    compare_steps(&format!("{name} impact"), &[into_wall(&t, o, SPEED, 1.0)], &[(3, STEPS), (200, 1), (60, STEPS)], true);
}

#[test]
fn viper_single_bitwise() {
    real_single(VIPER, "Viper");
}

#[test]
fn e36_single_bitwise() {
    real_single(E36, "E36");
}

#[test]
fn mustang_single_bitwise() {
    real_single(MUSTANG, "Mustang");
}

#[test]
fn agora_s_single_bitwise() {
    real_single(AGORA_S, "Agora S");
}

#[test]
fn real_cars_head_on_and_t_bone_bitwise() {
    for (rel, name) in [(VIPER, "Viper"), (E36, "E36"), (MUSTANG, "Mustang"), (AGORA_S, "Agora S")] {
        let Some(t) = ror(rel) else { continue };
        let o = Vec3::new(100.0, 0.0, 100.0);
        compare_steps(&format!("{name} head-on"), &pair(&t, o, 0.85 * SPEED, 1.5, false), &[(2, STEPS), (250, 1), (30, STEPS)], true);
        compare_steps(&format!("{name} T-bone"), &pair(&t, o, 0.85 * SPEED, 1.0, true), &[(2, STEPS), (250, 1), (30, STEPS)], true);
    }
}

/// Every real car into its own wall, far apart: the solo frame with contact on, and
/// the plain frame with it off; and a mixed pair of crashes side by side.
#[test]
fn mixed_fleet_bitwise() {
    let cars: Vec<String> = [VIPER, E36, MUSTANG, AGORA_S].iter().filter_map(|r| ror(r)).collect();
    if cars.is_empty() {
        return;
    }
    let fleet: Vec<ExplicitBeams> = cars
        .iter()
        .enumerate()
        .map(|(k, t)| into_wall(t, Vec3::new(100.0 + 40.0 * k as f32, 0.0, 100.0), SPEED * (0.7 + 0.1 * k as f32), 1.0))
        .collect();
    compare("mixed fleet, contact on", &fleet, 60, true);
    compare("mixed fleet, contact off", &fleet, 60, false);
    // Two crash pairs (different cars) and two cars alone.
    let mut fleet: Vec<ExplicitBeams> = Vec::new();
    for (k, t) in cars.iter().enumerate() {
        let o = Vec3::new(100.0 + 60.0 * k as f32, 0.0, 100.0);
        if k < 2 {
            fleet.extend(pair(t, o, 0.85 * SPEED, 1.5, k == 1));
        } else {
            fleet.push(into_wall(t, o, SPEED, 1.0));
        }
    }
    compare("mixed pairs and solos", &fleet, 70, true);
}
