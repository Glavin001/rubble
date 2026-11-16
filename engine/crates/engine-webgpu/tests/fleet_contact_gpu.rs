//! Vehicle contact's state by owner vehicle and its front as fleet-wide dispatches
//! (`explicit_plan::PERF_FLEET_STATE`, `PERF_FLEET_FRONT`; `explicit_beams_fleet.slang`,
//! engine/docs/measurements/contact-groups-design.md §8): a change of how, not what. Every
//! frame from one saved state on the split group path with the switches off (a contact
//! group's slots, its front and tests a group's dispatches) then on: every state word the
//! same. In a file of its own: a process of its own (see the note on `group_split_gpu`'s
//! tests run beside another flag-toggling test, vehicle-perf-first-principles.md §8.25).

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

fn generated() -> String {
    build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Low, ..Default::default() }).unwrap().truck
}

/// Vehicle contact's state by owner vehicle (`explicit_plan::PERF_FLEET_STATE`,
/// explicit_beams_fleet.slang; contact-groups-design.md §8): each frame from one saved
/// state on the split path with the switches of `mask` off (a contact group's slots) then
/// on, every state word the same. Returns the frames with a contact group, the largest
/// group and the most hits an owner listed in a frame's last pass with the switches on.
fn fleet_matches(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, mask: u32, self_split: bool) -> Option<[u32; 3]> {
    fleet_matches_kept(label, fleet, frames, steps, mask, self_split).map(|r| r.0)
}

/// [`fleet_matches`], with the kept candidates' counts of the frames with the switches on
/// (`ExplicitBeamsGpu::take_fleet_keep_stats`: builds, passes that reused the kept pairs,
/// pairs kept over the builds, passes on the front's own path, the most pairs an owner
/// kept).
fn fleet_matches_kept(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, mask: u32, self_split: bool) -> Option<([u32; 3], [u64; 10])> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    gpu.set_group_split(true);
    gpu.set_self_split(self_split);
    let flags = gpu.perf_flags();
    let (mut grouped, mut largest, mut hits) = (0u32, 0u32, 0u32);
    let mut kept = [0u64; 10];
    for f in 0..frames {
        gpu.wait();
        let snap = gpu.snapshot();
        gpu.set_perf_flags(flags & !mask);
        gpu.frame(steps);
        gpu.wait();
        let off = gpu.state_words();
        let sizes = gpu.contact_group_sizes();
        gpu.restore(&snap);
        // (The counts are in the snapshot: this frame's alone.)
        gpu.take_fleet_keep_stats();
        gpu.set_perf_flags(flags | mask);
        gpu.frame(steps);
        gpu.wait();
        let on = gpu.state_words();
        let k = gpu.take_fleet_keep_stats();
        for i in 0..10 {
            kept[i] = if matches!(i, 4 | 7 | 8) { kept[i].max(k[i]) } else { kept[i] + k[i] };
        }
        if let Some(d) = first_diff(&off, &on) {
            panic!(
                "{label} (self split {self_split}): frame {f} (groups {sizes:?}): perf flags {mask} on differ from off at {d}; counts {:?}",
                gpu.contact_group_counts()
            );
        }
        if let Some(e) = gpu.take_contact_errors().into_iter().next() {
            panic!("{label}: frame {f}: {e}");
        }
        hits = hits.max(gpu.contact_group_counts().iter().flat_map(|g| g.2.iter().map(|m| m[1])).max().unwrap_or(0));
        grouped += u32::from(!sizes.is_empty());
        largest = largest.max(sizes.iter().copied().max().unwrap_or(0));
    }
    gpu.set_perf_flags(flags);
    eprintln!(
        "{label} (self split {self_split}, perf flags {mask}): {frames} frames bit for bit, {grouped} with a contact group (largest {largest}), most hits an owner {hits}"
    );
    Some(([grouped, largest, hits], kept))
}

#[test]
fn owners_state_by_vehicle_is_bit_for_bit() {
    use engine_exec::explicit_plan::{PERF_FLEET_FRONT, PERF_FLEET_KEEP, PERF_FLEET_STATE};
    let text = generated();
    // The state by vehicle under the groups' dispatches (stage 1), then with the front and
    // the tests as fleet-wide dispatches too (stage 2), each against the groups' slots.
    for (mask, self_split) in [(PERF_FLEET_STATE, false), (PERF_FLEET_STATE, true), (PERF_FLEET_STATE | PERF_FLEET_FRONT, false), (PERF_FLEET_STATE | PERF_FLEET_FRONT, true)] {
        let a = sedan(&text, Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 12.0));
        let front = a.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        let probe = sedan(&text, Vec3::ZERO, 180.0, Vec3::ZERO);
        let back = probe.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
        let b = sedan(&text, Vec3::new(0.0, 0.0, front + 1.0 - back), 180.0, Vec3::new(0.0, 0.0, -12.0));
        let Some([grouped, _, hits]) = fleet_matches("two sedans head-on", vec![a, b], 40, 17, mask, self_split) else { return };
        assert!(grouped > 10 && hits > 0, "the cars met and hit ({grouped} frames grouped, {hits} hits)");
        // A column of six, the first pushed into the others: groups of two to six, owners
        // without a partner beside owners with hits.
        let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
        let pitch = hi - lo + 0.3;
        let fleet: Vec<ExplicitBeams> = (0..6)
            .map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, if i == 0 { Vec3::new(0.0, 0.0, 15.0) } else { Vec3::ZERO }))
            .collect();
        let Some([grouped, largest, hits]) = fleet_matches("a column of six", fleet, 30, 17, mask, self_split) else { return };
        assert!(grouped > 10 && largest > 4 && hits > 0, "a group past four members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    }
    // Vehicle contact's candidates kept across substeps (`PERF_FLEET_KEEP`,
    // contact-groups-design.md §8.8), on top of the fleet-wide front: the same two
    // scenes, the list kept from frame to frame (it is in the snapshot), every state word
    // the same as with the front's own search every substep; and the list was reused, not
    // only built. (In this test, not one of its own: see the head of the file.)
    for self_split in [false, true] {
        let a = sedan(&text, Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 12.0));
        let front = a.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        let probe = sedan(&text, Vec3::ZERO, 180.0, Vec3::ZERO);
        let back = probe.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
        let b = sedan(&text, Vec3::new(0.0, 0.0, front + 1.0 - back), 180.0, Vec3::new(0.0, 0.0, -12.0));
        let Some(([grouped, _, hits], kept)) = fleet_matches_kept("two sedans head-on, candidates kept", vec![a, b], 60, 17, PERF_FLEET_KEEP, self_split) else { return };
        eprintln!("   kept candidates: {kept:?} (builds, passes reused, pairs kept, passes on the front's own path, most an owner, builds for a calm node, for cover, largest bound and the allowance between vehicles as float bits, builds for two vehicles' motion)");
        assert!(grouped > 10 && hits > 0, "the cars met and hit ({grouped} frames grouped, {hits} hits)");
        assert!(kept[0] > 0 && kept[1] > kept[0], "the kept list was built and reused more often than built: {kept:?}");
        let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
        let pitch = hi - lo + 0.3;
        let fleet: Vec<ExplicitBeams> = (0..6)
            .map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, if i == 0 { Vec3::new(0.0, 0.0, 15.0) } else { Vec3::ZERO }))
            .collect();
        let Some(([grouped, largest, hits], kept)) = fleet_matches_kept("a column of six, candidates kept", fleet, 40, 17, PERF_FLEET_KEEP, self_split) else { return };
        eprintln!("   kept candidates: {kept:?}");
        assert!(grouped > 10 && largest > 4 && hits > 0, "a group past four members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
        assert!(kept[0] > 0 && kept[1] > kept[0], "the kept list was built and reused more often than built: {kept:?}");
    }
}
