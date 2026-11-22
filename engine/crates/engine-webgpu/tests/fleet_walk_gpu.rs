//! The fleet-wide walk with each candidate's position tested before its records are loaded
//! and its big triangles given threads of their own (`explicit_plan::PERF_FLEET_WALK_BOX`,
//! `explicit_beams_fleet.slang` flWalkBoxed, flWalkBig; vehicle-perf-first-principles.md
//! §8.56): a change of how, not what. Every frame from one saved state with the switch off
//! then on, the kept candidates off both ways so that every contact pass walks: every
//! state word the same. In a file of its own: a process of its own (see
//! `fleet_contact_gpu`'s head).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody_authoring::authoring::{build, style, BuildParams, Resolution};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn first_diff(a: &[(&str, Vec<u32>)], b: &[(&str, Vec<u32>)]) -> Option<String> {
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if let Some(i) = x.iter().zip(y).position(|(p, q)| p != q) {
            return Some(format!("{name}[{i}] ({:#x} vs {:#x})", x[i], y[i]));
        }
        if x.len() != y.len() {
            return Some(format!("{name}: lengths {} and {}", x.len(), y.len()));
        }
    }
    None
}

/// The generated sedan (low resolution), spawned at `at` moving at `vel`, its lowest node
/// 2 cm off the ground.
fn sedan(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = ExplicitBeams::from_truck_config(text, None, at, vel, settings).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(vel);
    s.finish();
    s
}

/// `frames` frames of `steps` steps, each from one saved state with the switch off then
/// on. Returns the frames with a contact group, the most hits an owner listed in a frame's
/// last pass, and the most big triangles a frame's last pass gave threads of their own.
fn walk_matches(label: &str, fleet: Vec<ExplicitBeams>, frames: u32, steps: u32, self_split: bool) -> Option<[u32; 3]> {
    use engine_exec::explicit_plan::{PERF_FLEET_KEEP, PERF_FLEET_WALK_BOX};
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_wide(false);
    gpu.set_group_split(true);
    gpu.set_self_split(self_split);
    let flags = gpu.perf_flags() & !PERF_FLEET_KEEP;
    let (mut grouped, mut hits, mut big) = (0u32, 0u32, 0u32);
    for f in 0..frames {
        gpu.wait();
        let snap = gpu.snapshot();
        gpu.set_perf_flags(flags & !PERF_FLEET_WALK_BOX);
        gpu.frame(steps);
        gpu.wait();
        let off = gpu.state_words();
        gpu.restore(&snap);
        gpu.set_perf_flags(flags | PERF_FLEET_WALK_BOX);
        gpu.frame(steps);
        gpu.wait();
        if let Some(d) = first_diff(&off, &gpu.state_words()) {
            panic!("{label} (self split {self_split}): frame {f}: the boxed walk differs from the walk at {d}; counts {:?}", gpu.contact_group_counts());
        }
        if let Some(e) = gpu.take_contact_errors().into_iter().next() {
            panic!("{label}: frame {f}: {e}");
        }
        hits = hits.max(gpu.contact_group_counts().iter().flat_map(|g| g.2.iter().map(|m| m[1])).max().unwrap_or(0));
        grouped += u32::from(!gpu.contact_group_sizes().is_empty());
        big = big.max(gpu.fleet_big_triangles());
    }
    eprintln!("{label} (self split {self_split}): {frames} frames bit for bit, {grouped} with a contact group, most hits an owner {hits}, most big triangles a pass {big}");
    Some([grouped, hits, big])
}

#[test]
fn boxed_walk_is_bit_for_bit() {
    let text = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Low, ..Default::default() }).unwrap().truck;
    let probe = sedan(&text, Vec3::ZERO, Vec3::ZERO);
    let (lo, hi) = probe.nodes.iter().fold((f32::MAX, f32::MIN), |(l, h), n| (l.min(n.pos.z), h.max(n.pos.z)));
    let pitch = hi - lo + 0.3;
    let column = |n: usize, speed: f32| -> Vec<ExplicitBeams> {
        (0..n).map(|i| sedan(&text, Vec3::new(0.0, 0.0, i as f32 * pitch), if i == 0 { Vec3::new(0.0, 0.0, speed) } else { Vec3::ZERO })).collect()
    };
    for self_split in [true, false] {
        // A column of ten, the first pushed into the others: every pass walks its
        // listed triangles' cells.
        let Some([grouped, hits, _]) = walk_matches("a column of ten", column(10, 15.0), 40, 17, self_split) else { return };
        assert!(grouped > 10 && hits > 0, "a group formed and hit ({grouped} frames grouped, {hits} hits)");
        // The same six with a torn panel: a triangle's node of the second and of the
        // fourth car metres away, so that their triangles' boxes are over the cells the
        // walk takes by cell and they test every candidate of their owner.
        let mut fleet = column(6, 15.0);
        for v in [1usize, 3] {
            for t in [10usize, 200] {
                let n = fleet[v].triangles[t % fleet[v].triangles.len()].nodes[0] as usize;
                fleet[v].nodes[n].pos += Vec3::new(4.0, 4.0, 4.0);
            }
        }
        let Some([grouped, hits, big]) = walk_matches("a column of six, torn panels", fleet, 40, 17, self_split) else { return };
        assert!(grouped > 10 && hits > 0, "a group formed and hit ({grouped} frames grouped, {hits} hits)");
        assert!(big > 0, "a triangle over the cells had threads of its own");
    }
}
