//! Vehicle sleep (`engine_softbody::explicit_sleep`) on the fast vehicle paths: contact
//! groups across workgroups, self-collision split, vehicle contact's state by vehicle and
//! its front across the fleet (`ExplicitBeamsGpu::set_group_split`, `set_self_split`,
//! `explicit_plan::PERF_FLEET_STATE`, `PERF_FLEET_FRONT`). The reference for sleep's rule
//! is the path it was built on, one workgroup a vehicle or contact group:
//! * every frame from one saved state on the fast paths (the awake-indexed dispatch, and
//!   every vehicle dispatched) and on the reference path: every state word and every
//!   sleep word (records, anchors) the same;
//! * with sleep on and nobody asleep, the physics state is that of sleep off;
//! * a contact group of two parked cars falls asleep and stays frozen, word for word,
//!   beside an awake car and a car asleep on its own; a car driven into the group wakes
//!   it; the woken cars start from their state at sleep, word for word; the frames that
//!   wake them and step them run the fast path.
//! * a lone vehicle: the fleet-wide passes (with and without the reused pair list) and
//!   self-collision split against the vehicle's workgroup, through falling asleep, frames
//!   asleep and a wake by the host.
//! In a file of its own: a process of its own (see `fleet_contact_gpu`'s note).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_sleep::VehicleSleep;
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

/// The repo sedan (beam tyres; parked, it sleeps: explicit-vehicles-in-world.md §10),
/// spawned at `at`, moving at `vel`, its lowest node 2 cm off the ground.
fn sedan(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = ExplicitBeams::from_truck(text, Vec3::ZERO, vel, settings).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos += at + Vec3::Y * (0.02 - low);
    }
    s.roll_tyres(vel);
    s.finish();
    s
}

fn repo_sedan() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck")).expect("the repo sedan")
}

/// Each vehicle's fastest node along an axis (m/s): what the rule's speed bound sees.
fn fastest(gpu: &ExplicitBeamsGpu) -> Vec<f32> {
    gpu.velocities().iter().map(|v| v.iter().map(|x| x.abs().max_element()).fold(0.0f32, f32::max)).collect()
}

/// Every state word and every sleep word.
fn words(gpu: &ExplicitBeamsGpu) -> Vec<(&'static str, Vec<u32>)> {
    let mut w = gpu.state_words();
    w.extend(gpu.sleep_words());
    w
}

/// Vehicle `v`'s own words of `w` ([`words`]): its node records (position, velocity,
/// force: 12 words a node), its beam records (8 words a beam) and its sleep record.
fn vehicle_words(gpu: &ExplicitBeamsGpu, w: &[(&'static str, Vec<u32>)], v: usize) -> Vec<u32> {
    let span = gpu.layout().spans[v];
    let run = |name: &str| &w.iter().find(|r| r.0 == name).unwrap_or_else(|| panic!("no {name} words")).1;
    let (n0, n1) = (12 * span.node_off as usize, 12 * (span.node_off + span.nodes) as usize);
    let (b0, b1) = (8 * span.beam_off as usize, 8 * (span.beam_off + span.beams) as usize);
    let mut out = run("nodes")[n0..n1].to_vec();
    out.extend_from_slice(&run("beams")[b0..b1]);
    out.extend_from_slice(&run("sleep records")[4 * v..4 * v + 4]);
    out
}

/// How a frame is run: the fast paths with the awake-indexed dispatch, the fast paths with
/// every vehicle dispatched, the reference (a workgroup a vehicle or group).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Fast,
    FastUnlisted,
    Reference,
}

fn set_path(gpu: &mut ExplicitBeamsGpu, p: Path) {
    gpu.set_group_split(p != Path::Reference);
    gpu.set_self_split(p != Path::Reference);
    gpu.set_sleep_listing(p != Path::FastUnlisted);
}

/// One frame of `steps` steps from the state the fleet is in, on every path: the same
/// words; the fleet is left as the fast path left it. Returns those words.
fn frame_every_path(gpu: &mut ExplicitBeamsGpu, steps: u32, label: &str) -> Vec<(&'static str, Vec<u32>)> {
    gpu.wait();
    let snap = gpu.snapshot();
    let mut out: Vec<(Path, Vec<(&'static str, Vec<u32>)>)> = Vec::new();
    for p in [Path::Reference, Path::FastUnlisted, Path::Fast] {
        gpu.restore(&snap);
        set_path(gpu, p);
        // The World's plan schedule: a contact plan every frame, on every path (the
        // plan's records are state words).
        gpu.request_contact_replan();
        gpu.frame(steps);
        gpu.wait();
        // (The fast paths run split whenever a vehicle may be awake; the reference never.)
        assert_eq!(gpu.split_path_used(), p != Path::Reference, "{label}: path {p:?}");
        out.push((p, words(gpu)));
    }
    for (p, w) in &out[1..] {
        if let Some(d) = first_diff(&out[0].1, w) {
            panic!("{label}: {p:?} differs from the reference path at {d}; sleep states {:?}", gpu.sleep_states());
        }
    }
    if let Some(e) = gpu.take_contact_errors().into_iter().next() {
        panic!("{label}: {e}");
    }
    out.pop().unwrap().1
}

#[test]
fn a_sleeping_group_is_frozen_and_wakes_on_the_fast_path() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let text = repo_sedan();
    let probe = sedan(&text, Vec3::ZERO, Vec3::ZERO);
    let (lo, hi) = probe.nodes.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), n| (l.min(n.pos), h.max(n.pos)));
    let (width, length) = (hi.x - lo.x, hi.z - lo.z);
    // Cars 0 and 1 parked side by side 0.3 m apart (one contact group: their frame bounds
    // overlap); car 2 behind car 0 with 6 m between them, parked until the group sleeps;
    // car 3 parked 60 m away, on its own.
    let fleet = vec![
        sedan(&text, Vec3::ZERO, Vec3::ZERO),
        sedan(&text, Vec3::new(width + 0.3, 0.0, 0.0), Vec3::ZERO),
        sedan(&text, Vec3::new(0.0, 0.0, -(length + 6.0)), Vec3::ZERO),
        sedan(&text, Vec3::new(60.0, 0.0, 0.0), Vec3::ZERO),
    ];
    // Thresholds loose enough for a car to sleep soon after its 2 cm drop (the rule and
    // the paths are what is tested here, not the defaults).
    let cfg = VehicleSleep { speed: 1.0, reach: 0.02, dwell: 150 };
    let mut gpu = ExplicitBeamsGpu::with_sleep(&device, &queue, &fleet, None, Some(cfg));
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    const STEPS: u32 = 17;

    // Sleep on with nobody asleep against sleep off: the physics state, bit for bit (the
    // first frames: every still count far below the dwell).
    for f in 0..4 {
        gpu.wait();
        let snap = gpu.snapshot();
        set_path(&mut gpu, Path::Fast);
        gpu.set_sleep(false);
        gpu.request_contact_replan();
        gpu.frame(STEPS);
        gpu.wait();
        let off = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_sleep(true);
        // (Turning sleep on wakes every vehicle: the snapshot's records again.)
        gpu.restore(&snap);
        gpu.request_contact_replan();
        gpu.frame(STEPS);
        gpu.wait();
        assert!(gpu.sleep_states().iter().all(|s| !s.1), "frame {f}: asleep within the dwell");
        if let Some(d) = first_diff(&off, &gpu.state_words()) {
            panic!("frame {f}: sleep on with nobody asleep differs from sleep off at {d}");
        }
    }

    // Until the group of two and the car on its own sleep.
    let mut frame = 0u32;
    let mut last = words(&gpu);
    while frame < 400 {
        last = frame_every_path(&mut gpu, STEPS, &format!("settling, frame {frame}"));
        frame += 1;
        let s = gpu.sleep_states();
        // (A group's members fall asleep together.)
        assert_eq!(s[0].1, s[1].1, "frame {frame}: one member of the group asleep without the other");
        if s[0].1 && s[3].1 {
            break;
        }
    }
    let s = gpu.sleep_states();
    assert!(s[0].1 && s[1].1 && s[3].1, "the parked cars never slept in {frame} frames: {s:?}; fastest nodes {:?} m/s", fastest(&gpu));
    assert_eq!(gpu.contact_group_sizes(), vec![2], "cars 0 and 1 are one contact group");
    let slept_at = frame;
    let at_sleep: Vec<Vec<u32>> = (0..4).map(|v| vehicle_words(&gpu, &last, v)).collect();

    // Car 2 sets off at 10 m/s (a host write wakes it); the sleepers stay frozen, word for
    // word, until its contact group reaches them.
    for i in 0..fleet[2].nodes.len() as u32 {
        gpu.set_velocity(2, i, Vec3::new(0.0, 0.0, 10.0));
    }
    // (The host's wake is written with the next frame, which a restored snapshot would
    // not hold: one frame on the fast path alone, then every path again.)
    set_path(&mut gpu, Path::Fast);
    gpu.request_contact_replan();
    gpu.frame(STEPS);
    gpu.wait();
    frame += 1;
    last = words(&gpu);
    for v in [0, 1, 3] {
        assert_eq!(vehicle_words(&gpu, &last, v), at_sleep[v], "sleeping car {v} changed when car 2 set off");
    }
    let mut woke_at = None;
    let mut frozen = 0u32;
    while frame < slept_at + 400 {
        let before = last;
        last = frame_every_path(&mut gpu, STEPS, &format!("approach, frame {frame}"));
        frame += 1;
        let s = gpu.sleep_states();
        assert!(!s[2].1, "frame {frame}: the driven car fell asleep");
        assert_eq!(s[0].1, s[1].1, "frame {frame}: one member of the group awake without the other");
        // The car on its own sleeps through all of it.
        assert!(s[3].1, "frame {frame}: the car on its own woke");
        assert_eq!(vehicle_words(&gpu, &last, 3), at_sleep[3], "frame {frame}: the car asleep on its own changed");
        if s[0].1 {
            for v in [0, 1] {
                assert_eq!(vehicle_words(&gpu, &last, v), at_sleep[v], "frame {frame}: sleeping car {v} changed");
            }
            frozen += 1;
            continue;
        }
        // Woken this frame: the state the frame started from is the state at sleep, word
        // for word (the frame itself matched the reference path's above).
        for v in [0, 1] {
            assert_eq!(vehicle_words(&gpu, &before, v), at_sleep[v], "frame {frame}: car {v} did not wake from its state at sleep");
        }
        let sizes = gpu.contact_group_sizes();
        assert_eq!(sizes, vec![3], "frame {frame}: woken by the driven car's contact group");
        woke_at = Some(frame);
        break;
    }
    let woke_at = woke_at.expect("the driven car never woke the group");
    assert!(frozen > 5, "the group slept {frozen} frames beside the awake car");
    let gap = {
        let p = gpu.positions();
        let front = p[2].iter().map(|x| x.z).fold(f32::MIN, f32::max);
        let back = p[0].iter().map(|x| x.z).fold(f32::MAX, f32::min);
        back - front
    };
    assert!(gap > 0.0, "woken after contact ({gap} m)");
    // Through the crash: every frame on every path the same, the fast path taken.
    let mut hits = 0u32;
    for f in 0..60 {
        frame_every_path(&mut gpu, STEPS, &format!("crash, frame {}", frame + f));
        hits = hits.max(gpu.contact_group_counts().iter().flat_map(|g| g.2.iter().map(|m| m[1])).max().unwrap_or(0));
        assert!(gpu.sleep_states()[3].1, "the car on its own woke");
    }
    assert!(hits > 0, "the driven car hit the group");
    eprintln!(
        "asleep after {slept_at} frames; {frozen} frames frozen beside the awake car; woken at frame {woke_at} with {gap:.2} m to go; 60 frames of the crash ({hits} hits an owner at most): every frame on the fast paths (listed and not) equal to the reference path, word for word"
    );
}

/// A lone vehicle's frame: its workgroup (the reference), self-collision split, the
/// fleet-wide passes, and those with the reused pair list.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lone {
    Workgroup,
    SelfSplit,
    Wide,
    WideList,
}

fn set_lone(gpu: &mut ExplicitBeamsGpu, p: Lone) {
    gpu.set_self_split(p == Lone::SelfSplit);
    gpu.set_wide_list(p == Lone::WideList);
    gpu.set_wide_fused(p == Lone::WideList);
    let wide = gpu.set_wide(matches!(p, Lone::Wide | Lone::WideList));
    assert_eq!(wide, matches!(p, Lone::Wide | Lone::WideList), "{p:?}: the wide passes with sleep, a lone vehicle");
}

/// One frame from the state the vehicle is in on every path. The wide passes keep other
/// scratch in the accumulators and misc records than a workgroup: the nodes, beams,
/// groups and sleep's own words are compared there; every word for self-collision split.
/// The vehicle is left as the wide passes with the pair list left it.
fn lone_frame_every_path(gpu: &mut ExplicitBeamsGpu, steps: u32, label: &str) -> Vec<(&'static str, Vec<u32>)> {
    gpu.wait();
    let snap = gpu.snapshot();
    let physics = |w: &[(&'static str, Vec<u32>)]| -> Vec<(&'static str, Vec<u32>)> {
        w.iter().filter(|r| ["nodes", "beams", "groups", "sleep records", "sleep anchors"].contains(&r.0)).cloned().collect()
    };
    let mut reference = Vec::new();
    let mut last = Vec::new();
    for p in [Lone::Workgroup, Lone::SelfSplit, Lone::Wide, Lone::WideList] {
        gpu.restore(&snap);
        set_lone(gpu, p);
        gpu.frame(steps);
        gpu.wait();
        let w = words(gpu);
        match p {
            Lone::Workgroup => reference = w.clone(),
            Lone::SelfSplit => {
                assert!(gpu.split_path_used(), "{label}: self-collision split with sleep");
                if let Some(d) = first_diff(&reference, &w) {
                    panic!("{label}: self-collision split differs from the vehicle's workgroup at {d}");
                }
            }
            _ => {
                if let Some(d) = first_diff(&physics(&reference), &physics(&w)) {
                    panic!("{label}: {p:?} differs from the vehicle's workgroup at {d}");
                }
            }
        }
        last = w;
    }
    last
}

#[test]
fn a_lone_vehicle_sleeps_and_wakes_on_every_path() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let text = repo_sedan();
    let fleet = vec![sedan(&text, Vec3::ZERO, Vec3::ZERO)];
    let cfg = VehicleSleep { speed: 1.0, reach: 0.02, dwell: 150 };
    let mut gpu = ExplicitBeamsGpu::with_sleep(&device, &queue, &fleet, None, Some(cfg));
    const STEPS: u32 = 17;
    let own = |w: &[(&'static str, Vec<u32>)]| -> Vec<u32> {
        ["nodes", "beams", "sleep records", "sleep anchors"].iter().flat_map(|n| w.iter().find(|r| r.0 == *n).unwrap().1.clone()).collect()
    };
    let mut frame = 0u32;
    let mut last = Vec::new();
    while frame < 400 {
        last = lone_frame_every_path(&mut gpu, STEPS, &format!("settling, frame {frame}"));
        frame += 1;
        if gpu.sleep_states()[0].1 {
            break;
        }
    }
    assert!(gpu.sleep_states()[0].1, "the parked car never slept in {frame} frames; fastest node {:?} m/s", fastest(&gpu));
    let slept_at = frame;
    let at_sleep = own(&last);
    // Asleep: nothing changes, on any path.
    for f in 0..10 {
        let w = lone_frame_every_path(&mut gpu, STEPS, &format!("asleep, frame {}", frame + f));
        assert_eq!(own(&w), at_sleep, "frame {}: the sleeping car changed", frame + f);
    }
    // The host sets it moving (a write wakes it; written with the next frame, which a
    // restored snapshot would not hold: one frame on the wide passes alone).
    for i in 0..fleet[0].nodes.len() as u32 {
        gpu.set_velocity(0, i, Vec3::new(0.0, 0.0, 3.0));
    }
    set_lone(&mut gpu, Lone::WideList);
    gpu.frame(STEPS);
    gpu.wait();
    assert!(!gpu.sleep_states()[0].1, "the host's write did not wake the car");
    let z0 = gpu.positions()[0][0].z;
    for f in 0..30 {
        lone_frame_every_path(&mut gpu, STEPS, &format!("woken, frame {}", frame + 11 + f));
    }
    let z1 = gpu.positions()[0][0].z;
    assert!(z1 > z0 + 0.3, "the woken car moved ({z0} to {z1} m)");
    eprintln!("a lone car: asleep after {slept_at} frames; 10 frames asleep unchanged; woken by a host write, 30 frames: self-collision split equal to its workgroup in every word, the wide passes (with and without the pair list) in every node, beam, group and sleep word");
}
