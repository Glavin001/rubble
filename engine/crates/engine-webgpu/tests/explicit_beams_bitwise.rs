//! The explicit-beams GPU kernel against its frozen baseline
//! (`slang/kernels/explicit_beams_baseline.slang`, the kernel before the GPU
//! parallelism work), bit for bit, frame by frame, on the same GPU state.
//!
//! The baseline has two scheduling races (both documented in
//! `engine/docs/explicit-beams.md`): a self-collision or contact candidate whose
//! force is read while another hit of the same phase adds to it, and the break
//! exception's live-beam count read in the beam phase that breaks a beam at that
//! node. The current kernel applies hits in the specification's order and resolves
//! deferred breaks in spawn order, so it is deterministic. Native wgpu worlds are
//! still compared from one saved state per frame: each frame starts from a snapshot
//! and runs the baseline (with and without its race probe, which reports which race
//! the frame met), both kernels compiled with `EB_DET_HITS` (tests only), and the
//! current kernel twice. Checks per frame:
//! * the current kernel twice: the same state (deterministic);
//! * the frozen reference (`slang/kernels/explicit_beams_ref.slang`, the kernel at
//!   feat/softbody-beamng-parity dcaa231b, before the latency and throughput work):
//!   the same state as the current kernel, bit for bit, on every frame;
//! * the baseline with `EB_DET_HITS` applies the hits found by its lanes on one lane
//!   sorted by (triangle, node), the specification's order; the current kernel with
//!   `EB_DET_HITS` (its ordered, parallel application with the baseline's hit
//!   arithmetic): the same state, bit for bit, on every frame without a
//!   break-exception race (every node record, beam record, incidence force,
//!   accumulator, group slot, throttle and per-vehicle record; scratch whose order
//!   follows the lanes is left out). The grid skip is on in the current kernel, so
//!   this also shows the skip leaves the results unchanged.
//! * a race-free frame of the probe build is the baseline's.
//! The shipped kernel differs from its `EB_DET_HITS` build only in the hit force's
//! arithmetic (strict, as the CPU reference; tests/explicit_beams_gpu.rs compares it
//! with the CPU).
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

/// Per frame, each from the same saved state: the baseline (and with its race
/// probe), both kernels with `EB_DET_HITS`, and the current kernel twice. Returns
/// (frames whose fixed-order hits matched the baseline's, frames with a
/// break-exception race).
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
    let mut contact_races = 0;
    let mut break_races = 0;
    let mut det = 0;
    let mut retried = 0;
    let mut groups = 0;
    let mut members = 0;
    let mut same_as_ref = 0;
    let mut ref_retries = 0;
    let frames: u32 = segments.iter().map(|s| s.0).sum();
    let plan: Vec<u32> = segments.iter().flat_map(|&(n, k)| std::iter::repeat_n(k, n as usize)).collect();
    let run = |gpu: &mut ExplicitBeamsGpu, snap: &engine_webgpu::GpuSnapshot, var: KernelVariant, steps: u32| {
        gpu.restore(snap);
        gpu.set_kernel_variant(var);
        gpu.frame(steps);
        gpu.wait();
        gpu.state_words()
    };
    for (f, &steps) in plan.iter().enumerate() {
        let snap = gpu.snapshot();
        let a = run(&mut gpu, &snap, KernelVariant::Baseline, steps);
        gpu.take_race_flag();
        let a2 = run(&mut gpu, &snap, KernelVariant::BaselineProbe, steps);
        let race = gpu.take_race_flag();
        assert_eq!(race & 2, 0, "{label}: the race probe gave up at frame {f}");
        // The probe changes no arithmetic: a race-free frame is the baseline's.
        if race == 0 {
            if let Some(d) = first_diff(&a, &a2) {
                panic!("{label}: frame {f}: the probe build differs from the baseline at {d}");
            }
        }
        // The current kernel is deterministic: the same frame twice is the same,
        // contact groups included, with no retry (the group path's rare race, a
        // contact triangle's marks stored as components of one record, is fixed; the
        // frozen kernels below keep it, so their comparisons retry a group frame).
        let b = run(&mut gpu, &snap, KernelVariant::Current, steps);
        let grouped = gpu.contact_groups();
        groups = groups.max(grouped);
        members = members.max(gpu.contact_group_sizes().into_iter().max().unwrap_or(0));
        if let Some(d) = first_diff(&b, &run(&mut gpu, &snap, KernelVariant::Current, steps)) {
            panic!("{label}: frame {f}: the current kernel is not deterministic: {d}");
        }
        // The frozen reference (the kernel before the latency work): the same state,
        // bit for bit (a group frame may hit the group path's rare race: retried).
        let mut refd = first_diff(&b, &run(&mut gpu, &snap, KernelVariant::Reference, steps));
        for _ in 0..4 {
            if refd.is_none() || grouped == 0 {
                break;
            }
            ref_retries += 1;
            let b2 = run(&mut gpu, &snap, KernelVariant::Current, steps);
            refd = first_diff(&b2, &run(&mut gpu, &snap, KernelVariant::Reference, steps));
        }
        if let Some(d) = refd {
            panic!("{label}: frame {f}: the current kernel differs from the frozen reference: {d}");
        }
        same_as_ref += 1;
        // The current kernel's ordered hit application (with the baseline's
        // arithmetic, EB_DET_HITS) against the baseline's hits applied on one lane in
        // the specification's order: the same state, bit for bit, unless the
        // baseline's break exception raced (the current kernel resolves it in spawn
        // order; the baseline reads whatever the other lanes have written).
        let d = run(&mut gpu, &snap, KernelVariant::CurrentDet, steps);
        let c = run(&mut gpu, &snap, KernelVariant::BaselineDet, steps);
        assert!(!gpu.take_det_overflow(), "{label}: frame {f}: the fixed-order hit list overflowed");
        contact_races += race & 1;
        if race & 4 != 0 {
            break_races += 1;
        } else {
            // The baseline's contact-group path is itself, rarely, not reproducible
            // with fixed-order hits (the baseline against itself shows the same rare
            // mismatches: RUBBLE_DET_AA=1). A logic difference is deterministic, so a
            // mismatching frame is retried; it must match on a retry.
            let mut diff = first_diff(&c, &d);
            for _ in 0..4 {
                if diff.is_none() {
                    break;
                }
                let c2 = run(&mut gpu, &snap, KernelVariant::BaselineDet, steps);
                let d2 = run(&mut gpu, &snap, KernelVariant::CurrentDet, steps);
                diff = first_diff(&c2, &d2);
                retried += 1;
            }
            if let Some(x) = diff {
                panic!("{label}: frame {f}: the ordered hits differ from the baseline's one-lane order at {x}");
            }
            det += 1;
        }
        // Continue from the current kernel's frame.
        gpu.restore(&snap);
        gpu.set_kernel_variant(KernelVariant::Current);
        gpu.frame(steps);
        gpu.wait();
    }
    eprintln!(
        "{label}: {frames} frames deterministic, {same_as_ref} = the \
         frozen reference ({ref_retries} retries); ordered hits = \
         the baseline's one-lane order in {det} ({retried} retries); {break_races} frames with a \
         break-exception race (not compared), {contact_races} with a contact race in the plain \
         baseline; contact groups seen {groups}, the largest of {members} members"
    );
    assert_eq!(det + break_races, frames);
    assert_eq!(same_as_ref, frames);
    Some((det, break_races))
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
    // At the World origin: vehicle contact runs in the owner's frame (vvRel) while the
    // frozen reference runs it in World coordinates; the two are the same arithmetic
    // only while every vehicle's origin is 0 (never re-based: within REBASE_RADIUS).
    // (explicit_far_origin_gpu.rs covers contact away from the origin.)
    let o = Vec3::ZERO;
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

/// Three cars in a chain (one driving into two standing nose to tail) and four (two
/// head-on pairs side by side): contact groups of three and four, the owners' levels.
fn chain_and_four(text: &str, name: &str) {
    let g = geo(text);
    // At the World origin: vehicle contact runs in the owner's frame (vvRel) while the
    // frozen reference runs it in World coordinates; the two are the same arithmetic
    // only while every vehicle's origin is 0 (never re-based: within REBASE_RADIUS).
    // (explicit_far_origin_gpu.rs covers contact away from the origin.)
    let o = Vec3::ZERO;
    let len = g.along.1 - g.along.0;
    let chain = vec![
        spawn(text, o, 0.0, g.forward * (0.85 * SPEED)),
        spawn(text, o + g.forward * (len + 1.5), 0.0, Vec3::ZERO),
        spawn(text, o + g.forward * (2.0 * len + 2.0), 0.0, Vec3::ZERO),
    ];
    compare_steps(&format!("{name} chain of three"), &chain, &[(2, STEPS), (200, 1), (30, STEPS)], true);
    let width = g.across.1 - g.across.0;
    let mut four = pair(text, o, 0.85 * SPEED, 1.5, false);
    four.extend(pair(text, o + g.lateral * (width + 0.6), 0.7 * SPEED, 1.5, false));
    compare_steps(&format!("{name} two pairs side by side"), &four, &[(2, STEPS), (200, 1), (30, STEPS)], true);
}

/// Five cars nose to tail, the first driving into the four standing: a contact group
/// of more than four members (the larger groups' module, Kernel::ContactBig).
#[test]
fn a_group_of_five_bitwise() {
    let text = repo_sedan();
    let g = geo(&text);
    let o = Vec3::ZERO;
    let len = g.along.1 - g.along.0;
    let mut chain = vec![spawn(&text, o, 0.0, g.forward * (0.85 * SPEED))];
    for k in 1..5 {
        chain.push(spawn(&text, o + g.forward * (k as f32 * (len + 0.5) + 1.0), 0.0, Vec3::ZERO));
    }
    compare_steps("sedan chain of five", &chain, &[(2, STEPS), (120, 1), (30, STEPS)], true);
}

#[test]
fn groups_of_three_and_four_bitwise() {
    chain_and_four(&repo_sedan(), "sedan");
    for (rel, name) in [(VIPER, "Viper"), (MUSTANG, "Mustang")] {
        if let Some(t) = ror(rel) {
            chain_and_four(&t, name);
        }
    }
}

/// One CPU step with the GPU's self-collision accumulator: the step's
/// self-collision forces on each node (left in its force for the next step) rounded
/// to 1/32 N per component, as the GPU's fixed-point gAcc holds them. The
/// collision part is the step's force less the same step's without candidates.
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

/// Per frame from the CPU reference's state: the GPU (current kernel, and the
/// baseline for comparison) uploads the CPU's vehicle and runs one frame; the CPU
/// runs it too, as it is and with the GPU's fixed-point self-collision forces
/// ([`step_fixed_point`]). Returns per frame the node RMS (m) of the current
/// kernel to the CPU, of the baseline to the CPU, and of the current kernel to the
/// fixed-point CPU. No air drag or turbulence (the GPU's turbulence draws are its
/// own sequence).
fn cpu_per_frame(text: &str, drive: bool, frames: u32) -> Option<Vec<(f32, f32, f32)>> {
    let (device, queue) = device()?;
    let settings = ExplicitBeamsSettings::default();
    let o = Vec3::new(100.0, 0.0, 100.0);
    let g = geo(text);
    let vel = if drive { g.forward * SPEED } else { Vec3::ZERO };
    let mut cpu = ExplicitBeams::from_truck_config(text, None, o, vel, settings).expect("spawn");
    let lowest = cpu.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut cpu.nodes {
        n.pos.y += 0.02 - lowest;
    }
    cpu.roll_tyres(vel);
    if drive {
        let face = o.dot(g.forward) + g.along.1 + 1.0;
        let mid = o.dot(g.lateral) + 0.5 * (g.across.0 + g.across.1);
        let side = 0.5 * (g.across.1 - g.across.0) + 1.0;
        let centre = g.forward * (face + 0.5) + g.lateral * mid + Vec3::Y * 3.0;
        let half = if g.forward.x.abs() > 0.5 { Vec3::new(0.5, 3.0, side) } else { Vec3::new(side, 3.0, 0.5) };
        cpu.boxes.push(StaticBox::centered(centre, half));
    }
    let mut out = Vec::new();
    for _ in 0..frames {
        let mut ends = Vec::new();
        for var in [KernelVariant::Current, KernelVariant::Baseline] {
            let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
            gpu.set_wide(false);
            gpu.set_kernel_variant(var);
            gpu.frame(STEPS);
            gpu.wait();
            ends.push(gpu.positions().remove(0));
        }
        cpu.check_wings();
        cpu.rebase_origin();
        let mut fixed = cpu.clone();
        for _ in 0..STEPS {
            cpu.step();
            step_fixed_point(&mut fixed);
        }
        let rms = |p: &[Vec3], c: &[Vec3]| {
            (p.iter().zip(c).map(|(a, b)| (*a - *b).length_squared()).sum::<f32>() / c.len() as f32).sqrt()
        };
        let (c, f) = (cpu.positions(), fixed.positions());
        out.push((rms(&ends[0], &c), rms(&ends[1], &c), rms(&ends[0], &f)));
    }
    Some(out)
}

/// The current kernel against the CPU reference, one frame at a time from the CPU's
/// state: the self-collision order is the CPU's, so a frame lands within float noise
/// of the CPU with the GPU's one rule of its own modelled, the self-collision
/// accumulators' 1/32 N fixed point ([`step_fixed_point`]); far closer than the
/// baseline's lane-order hits. Against the plain CPU it is usually as close, but
/// that rounding (up to 1/64 N a component) can tip a contact's discrete switch
/// within a frame: the Viper's wall crash without its airfoil tables (no
/// `RUBBLE_AIRFOIL_DIR`) does at frame 38, step 7 (nodes 107 and 100 jump 0.64 m/s
/// apart in one step, 0.040 mm RMS by the frame's end), and the fixed-point CPU
/// lands on the GPU's frame there (reported, not judged).
#[test]
fn cpu_reference_per_frame() {
    let mut cars = vec![("sedan".to_string(), repo_sedan())];
    for (rel, name) in [(VIPER, "Viper"), (E36, "E36"), (MUSTANG, "Mustang"), (AGORA_S, "Agora S")] {
        if let Some(t) = ror(rel) {
            cars.push((name.to_string(), t));
        }
    }
    for (name, text) in &cars {
        for drive in [false, true] {
            let Some(r) = cpu_per_frame(text, drive, 40) else { return };
            let pick = |x: &(f32, f32, f32), k: usize| [x.0, x.1, x.2][k];
            let med = |k: usize| {
                let mut v: Vec<f32> = r.iter().map(|x| pick(x, k)).collect();
                v.sort_by(f32::total_cmp);
                v[v.len() / 2]
            };
            let max = |k: usize| r.iter().map(|x| pick(x, k)).fold(0.0, f32::max);
            eprintln!(
                "{name} {}: per-frame node RMS to the CPU, current kernel median {:.5} mm max {:.5} mm; baseline median {:.5} mm max {:.5} mm; current kernel to the fixed-point CPU median {:.5} mm max {:.5} mm",
                if drive { "into a wall" } else { "idle" },
                med(0) * 1e3,
                max(0) * 1e3,
                med(1) * 1e3,
                max(1) * 1e3,
                med(2) * 1e3,
                max(2) * 1e3
            );
            // Float noise: at most 20 µm RMS in any frame from the fixed-point CPU
            // (the baseline: up to 3.9 mm per frame from the CPU in the E36's wall
            // crash).
            assert!(max(2) < 2e-5, "{name}: a frame {} mm RMS from the fixed-point CPU", max(2) * 1e3);
        }
    }
}

/// The wide passes (a dispatch per pass, the hits applied by their own pass) against
/// the one-workgroup frame, from the same state each frame: the same hits in the
/// same order, so the same state bit for bit (the sedan into a wall: self-collision
/// hits, deferred breaks and group breaks).
#[test]
fn wide_passes_match_the_frame_bitwise() {
    let s = repo_sedan();
    let Some((device, queue)) = device() else { return };
    let fleet = [into_wall(&s, Vec3::new(100.0, 0.0, 100.0), SPEED, 1.0)];
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(false);
    assert!(gpu.set_wide(true), "the sedan runs the wide passes");
    let plan: Vec<u32> = [(3u32, STEPS), (200, 1), (60, STEPS)]
        .iter()
        .flat_map(|&(n, k)| std::iter::repeat_n(k, n as usize))
        .collect();
    for (f, &steps) in plan.iter().enumerate() {
        let snap = gpu.snapshot();
        gpu.set_wide(false);
        gpu.frame(steps);
        gpu.wait();
        let a = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_wide(true);
        gpu.frame(steps);
        gpu.wait();
        let b = gpu.state_words();
        if let Some(d) = first_diff(&a, &b) {
            panic!("frame {f}: the wide passes differ from the frame at {d}");
        }
    }
    let broken = gpu.beams()[0].iter().filter(|b| b.broken).count();
    eprintln!("sedan wall, wide passes = frame on {} frames; {broken} beams broken", plan.len());
    assert!(broken > 0);
}
