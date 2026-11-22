//! Vehicle contact's state by owner vehicle and its front as fleet-wide dispatches
//! (`explicit_plan::PERF_FLEET_STATE`, `PERF_FLEET_FRONT`; `explicit_beams_fleet.slang`,
//! engine/docs/measurements/contact-groups-design.md §8): a change of how, not what. Every
//! frame from one saved state on the split group path with the switches off (a contact
//! group's slots, its front and tests a group's dispatches) then on: every state word the
//! same. In a file of its own: a process of its own (see the note on `group_split_gpu`'s
//! tests run beside another flag-toggling test, vehicle-perf-first-principles.md §8.25).

use engine_softbody_authoring::authoring::{build, style, BuildParams, Resolution};
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
/// group, the most hits an owner listed in a frame's last pass with the switches on, and
/// the highest level of a frame's last pass (with the levels for any number of owners).
fn fleet_matches(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, mask: u32, self_split: bool) -> Option<[u32; 4]> {
    fleet_matches_with(label, fleet, frames, steps, mask, self_split, 0).map(|r| r.0)
}

/// [`fleet_matches`], with the kept candidates' counts of the frames with the switches on
/// (`ExplicitBeamsGpu::take_fleet_keep_stats`: builds, passes that reused the kept pairs,
/// pairs kept over the builds, passes on the front's own path, the most pairs an owner
/// kept).
fn fleet_matches_kept(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, mask: u32, self_split: bool) -> Option<([u32; 4], [u64; 10])> {
    fleet_matches_with(label, fleet, frames, steps, mask, self_split, 0)
}

/// [`fleet_matches_kept`] with the second measurement word `word2` (bit 31: the levels for
/// any number of owners for the small groups too, with `PERF_FLEET_ANY`).
fn fleet_matches_with(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, mask: u32, self_split: bool, word2: u32) -> Option<([u32; 4], [u64; 10])> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    gpu.set_group_split(true);
    gpu.set_self_split(self_split);
    gpu.set_ablation2(word2);
    let flags = gpu.perf_flags();
    let (mut grouped, mut largest, mut hits, mut top) = (0u32, 0u32, 0u32, 0u32);
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
        top = top.max(gpu.contact_levels().iter().map(|p| p.1).max().unwrap_or(0));
        grouped += u32::from(!sizes.is_empty());
        largest = largest.max(sizes.iter().copied().max().unwrap_or(0));
    }
    gpu.set_perf_flags(flags);
    eprintln!(
        "{label} (self split {self_split}, perf flags {mask}): {frames} frames bit for bit, {grouped} with a contact group (largest {largest}), most hits an owner {hits}, highest level {top}"
    );
    Some(([grouped, largest, hits, top], kept))
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
        let Some([grouped, _, hits, _]) = fleet_matches("two sedans head-on", vec![a, b], 40, 17, mask, self_split) else { return };
        assert!(grouped > 10 && hits > 0, "the cars met and hit ({grouped} frames grouped, {hits} hits)");
        // A column of six, the first pushed into the others: groups of two to six, owners
        // without a partner beside owners with hits.
        let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
        let pitch = hi - lo + 0.3;
        let fleet: Vec<ExplicitBeams> = (0..6)
            .map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, if i == 0 { Vec3::new(0.0, 0.0, 15.0) } else { Vec3::ZERO }))
            .collect();
        let Some([grouped, largest, hits, _]) = fleet_matches("a column of six", fleet, 30, 17, mask, self_split) else { return };
        assert!(grouped > 10 && largest > 4 && hits > 0, "a group past four members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    }
    // Vehicle contact's candidates kept across substeps (`PERF_FLEET_KEEP`,
    // contact-groups-design.md §8.10), on top of the fleet-wide front: the same two
    // scenes, the list kept from frame to frame (it is in the snapshot), every state word
    // the same as with the front's own search every substep; and the list was reused, not
    // only built. (In this test, not one of its own: see the head of the file.)
    for self_split in [false, true] {
        let a = sedan(&text, Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 12.0));
        let front = a.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
        let probe = sedan(&text, Vec3::ZERO, 180.0, Vec3::ZERO);
        let back = probe.nodes.iter().map(|n| n.pos.z).fold(f32::MAX, f32::min);
        let b = sedan(&text, Vec3::new(0.0, 0.0, front + 1.0 - back), 180.0, Vec3::new(0.0, 0.0, -12.0));
        let Some(([grouped, _, hits, _], kept)) = fleet_matches_kept("two sedans head-on, candidates kept", vec![a, b], 60, 17, PERF_FLEET_KEEP, self_split) else { return };
        eprintln!("   kept candidates: {kept:?} (builds, passes reused, pairs kept, passes on the front's own path, most an owner, builds for a calm node, for cover, largest bound and the allowance between vehicles as float bits, builds for two vehicles' motion)");
        assert!(grouped > 10 && hits > 0, "the cars met and hit ({grouped} frames grouped, {hits} hits)");
        assert!(kept[0] > 0 && kept[1] > kept[0], "the kept list was built and reused more often than built: {kept:?}");
        let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
        let pitch = hi - lo + 0.3;
        let fleet: Vec<ExplicitBeams> = (0..6)
            .map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, if i == 0 { Vec3::new(0.0, 0.0, 15.0) } else { Vec3::ZERO }))
            .collect();
        let Some(([grouped, largest, hits, _], kept)) = fleet_matches_kept("a column of six, candidates kept", fleet, 40, 17, PERF_FLEET_KEEP, self_split) else { return };
        eprintln!("   kept candidates: {kept:?}");
        assert!(grouped > 10 && largest > 4 && hits > 0, "a group past four members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
        assert!(kept[0] > 0 && kept[1] > kept[0], "the kept list was built and reused more often than built: {kept:?}");
    }
    groups_of_any_size_are_bit_for_bit();
}

/// Contact groups of any size on the fleet-wide passes (`explicit_plan::PERF_FLEET_ANY`,
/// the kernel's fleetLevelsAny; contact-groups-design.md §8 stage 3): a column of ten, a
/// group past eight members, against the larger groups' pass in the group's workgroup
/// (the switch off); and the same levels taken for small groups (the second measurement
/// word's bit 31) against the small groups' own. (Run from the one test above, after it:
/// two such tests side by side in one process diverge, vehicle-perf-first-principles.md
/// §8.25's open finding.)
fn groups_of_any_size_are_bit_for_bit() {
    use engine_exec::explicit_plan::PERF_FLEET_ANY;
    let text = generated();
    let probe = sedan(&text, Vec3::ZERO, 0.0, Vec3::ZERO);
    let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
    let pitch = hi - lo + 0.3;
    let column = |n: usize, speed: f32| -> Vec<ExplicitBeams> {
        (0..n)
            .map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), 0.0, if i == 0 { Vec3::new(0.0, 0.0, speed) } else { Vec3::ZERO }))
            .collect()
    };
    for self_split in [true, false] {
        let Some([grouped, largest, hits, _]) = fleet_matches("a column of ten", column(10, 15.0), 40, 17, PERF_FLEET_ANY, self_split) else { return };
        assert!(grouped > 10 && largest > 8 && hits > 0, "a group past eight members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
        let Some(([grouped, largest, hits, _], _)) = fleet_matches_with("a column of six, the levels for any size", column(6, 15.0), 30, 17, PERF_FLEET_ANY, self_split, 0x8000_0000) else { return };
        assert!(grouped > 10 && largest > 4 && hits > 0, "a group formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    }
    // The listed triangles' throttle updates as threads over the fleet
    // (`explicit_plan::PERF_FLEET_POST`, the kernel's fleetPost): with the levels for any
    // size (a column of ten, against neither) and alone (a column of six, the small groups'
    // own levels).
    use engine_exec::explicit_plan::PERF_FLEET_POST;
    let Some([grouped, largest, hits, _]) = fleet_matches("a column of ten, throttles fleet-wide", column(10, 15.0), 40, 17, PERF_FLEET_ANY | PERF_FLEET_POST, true) else { return };
    assert!(grouped > 10 && largest > 8 && hits > 0, "a group past eight members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    let Some([grouped, largest, hits, _]) = fleet_matches("a column of six, throttles fleet-wide", column(6, 15.0), 30, 17, PERF_FLEET_POST, true) else { return };
    assert!(grouped > 10 && largest > 4 && hits > 0, "a group formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    levels_across_the_gpu_are_bit_for_bit(&column);
}

/// The levels across the GPU (`explicit_plan::PERF_FLEET_LEVELS`,
/// explicit_beams_fleet_levels.slang; contact-groups-design.md §8.7, stage 3c) against the
/// group's workgroup (the switch off): a column of ten pushed from behind (a chain of
/// owners, many levels) and two columns of six side by side (two groups in the same
/// dispatches), with every level slot, and with two and one (the levels past them in the
/// end dispatch's workgroup), with and without self split.
fn levels_across_the_gpu_are_bit_for_bit(column: &dyn Fn(usize, f32) -> Vec<ExplicitBeams>) {
    use engine_exec::explicit_plan::PERF_FLEET_LEVELS;
    let mut deepest = 0;
    for slots in ["16", "2", "1"] {
        std::env::set_var("RUBBLE_FLEET_LEVEL_SLOTS", slots);
        for self_split in [true, false] {
            let label = format!("a column of ten, levels across the GPU, {slots} slots");
            let Some([grouped, largest, hits, top]) = fleet_matches(&label, column(10, 15.0), 40, 17, PERF_FLEET_LEVELS, self_split) else { return };
            assert!(grouped > 10 && largest > 8 && hits > 0, "a group past eight members formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
            deepest = deepest.max(top);
        }
        // Two columns of six, ten metres apart: two groups, their levels in one dispatch.
        let mut two = column(6, 15.0);
        for mut v in column(6, 12.0) {
            for n in &mut v.nodes {
                n.pos.x += 10.0;
            }
            two.push(v);
        }
        let label = format!("two columns of six, levels across the GPU, {slots} slots");
        let Some([grouped, largest, hits, _]) = fleet_matches(&label, two, 30, 17, PERF_FLEET_LEVELS, true) else { return };
        assert!(grouped > 10 && largest > 4 && hits > 0, "groups formed and hit ({grouped} frames, largest {largest}, {hits} hits)");
    }
    std::env::remove_var("RUBBLE_FLEET_LEVEL_SLOTS");
    assert!(deepest >= 3, "the column's levels reached {deepest}: too few to pass a slot");
    // One hit's latency cut in the group's levels (`explicit_plan::PERF_FLEET_HITLAT`,
    // fleetLevelsLat) and the front's fused dispatches (`PERF_FLEET_FUSE`), each against
    // the switch off: the column of ten and the two columns of six.
    use engine_exec::explicit_plan::{PERF_FLEET_FUSE, PERF_FLEET_HITLAT};
    for mask in [PERF_FLEET_HITLAT, PERF_FLEET_FUSE, PERF_FLEET_HITLAT | PERF_FLEET_FUSE] {
        for self_split in [true, false] {
            let Some([grouped, largest, hits, top]) = fleet_matches("a column of ten", column(10, 15.0), 40, 17, mask, self_split) else { return };
            assert!(grouped > 10 && largest > 8 && hits > 0 && top >= 3, "a deep group formed and hit ({grouped} frames, largest {largest}, {hits} hits, level {top})");
        }
        let mut two = column(6, 15.0);
        for mut v in column(6, 12.0) {
            for n in &mut v.nodes {
                n.pos.x += 10.0;
            }
            two.push(v);
        }
        let Some([grouped, _, hits, _]) = fleet_matches("two columns of six", two, 30, 17, mask, true) else { return };
        assert!(grouped > 10 && hits > 0, "groups formed and hit ({grouped} frames, {hits} hits)");
    }
    // Each hit worked out once in the group's levels (`explicit_plan::PERF_LEVELS_ONCE`,
    // fleetLevelsOnce, with PERF_FLEET_HITLAT on: the default), against it off: the column of
    // ten (at least 3 levels deep), with and without self split, and the two columns of six.
    use engine_exec::explicit_plan::PERF_LEVELS_ONCE;
    for self_split in [true, false] {
        let Some([grouped, largest, hits, top]) =
            fleet_matches("a column of ten, hits once", column(10, 15.0), 40, 17, PERF_LEVELS_ONCE, self_split)
        else {
            return;
        };
        assert!(grouped > 10 && largest > 8 && hits > 0 && top >= 3, "a deep group formed and hit ({grouped} frames, largest {largest}, {hits} hits, level {top})");
    }
    let mut two = column(6, 15.0);
    for mut v in column(6, 12.0) {
        for n in &mut v.nodes {
            n.pos.x += 10.0;
        }
        two.push(v);
    }
    let Some([grouped, _, hits, _]) = fleet_matches("two columns of six, hits once", two, 30, 17, PERF_LEVELS_ONCE, true) else { return };
    assert!(grouped > 10 && hits > 0, "groups formed and hit ({grouped} frames, {hits} hits)");
}
