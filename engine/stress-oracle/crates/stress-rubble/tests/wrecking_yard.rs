//! The Wrecking Yard structures (`stress_ref::wrecking::STRUCTURES`, the Ram Raid and
//! Fly-through buildings of `stress_ref::buildings` among them) stand and settle on the GPU in
//! both of the lab's stress modes before anything hits them, set up as stress-web sets
//! them up: `blast_parity` on the resident GpuWorld (impact islands on), `threshold` on
//! the GPU-resident threshold world. Two seconds at 60 Hz with no shot: no bond breaks,
//! the structure stays one body, and no chunk has moved more than a centimetre.
//!
//!   cargo test --release -p stress-rubble --features gpu --test wrecking_yard
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::wrecking::STRUCTURES;
use stress_rubble::scenes::wrecking_yard;
use stress_rubble::RubbleEngine;

const SECONDS: f64 = 2.0;
/// How far a chunk may move while the structure settles (m).
const SETTLE: f64 = 0.01;

fn blast_parity(gpu: &Gpu, key: &str) {
    let p = Profile::BlastParity;
    let scene = wrecking_yard(key, p, SECONDS);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.sync(gpu);
    let fragments = w.observation().values["fragments"];
    let frames = (SECONDS / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.sync(gpu);
    let o = w.observation();
    let moved = o.bodies.values().flatten().map(|c| Vec3::from_array(c.displacement).norm()).fold(0.0, f64::max);
    println!("{key} blast_parity: {frames} frames, broken {}, fragments {} (from {fragments}), largest move {moved:.2e} m", o.values["broken_bonds"], o.values["fragments"]);
    assert_eq!(o.values["broken_bonds"], 0.0, "{key}: bonds broke while settling");
    assert_eq!(o.values["fragments"], fragments, "{key}: the structure came apart");
    assert!(moved < SETTLE, "{key}: a chunk moved {moved} m");
}

fn threshold(gpu: &Gpu, key: &str) {
    let p = Profile::Threshold;
    let scene = wrecking_yard(key, p, SECONDS);
    let ball = stress_ref::wrecking::BALLS[0];
    let mut w = stress_rubble::threshold::ThresholdWorld::new(gpu, &scene, ball.radius, ball.mass).expect("threshold world");
    let start = w.chunk_centres(gpu);
    let frames = (SECONDS / w.frame_dt).round() as usize;
    // Each frame waited for and its counters read (the ring copies them into a slot
    // the host has read).
    let mut st = w.poll(gpu);
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        st = w.poll(gpu);
    }
    let end = w.chunk_centres(gpu);
    let moved = start.iter().zip(&end).map(|((a, _), (b, _))| (Vec3::from_array(*a) - Vec3::from_array(*b)).norm()).fold(0.0, f64::max);
    println!("{key} threshold: {frames} frames (stats of frame {}), broken {}, bodies {}, errors {:#x}/{:#x}, largest move {moved:.2e} m", st.frame, st.broken, st.bodies, st.error, st.engine_overflow);
    assert_eq!(st.frame, frames as u64, "{key}: the counters are the last frame's");
    assert_eq!((st.error, st.engine_overflow), (0, 0), "{key}");
    assert_eq!(st.broken, 0, "{key}: bonds broke while settling");
    assert_eq!(st.split_frames, 0, "{key}: the structure came apart");
    assert!(moved < SETTLE, "{key}: a chunk moved {moved} m");
}

/// The viaduct alone (also in `structures_settle_in_both_stress_modes`):
///   cargo test --release -p stress-rubble --features gpu --test wrecking_yard viaduct
#[test]
fn viaduct_settles_in_both_stress_modes() {
    let gpu = Gpu::new().expect("GPU");
    blast_parity(&gpu, "bo_viaduct");
    threshold(&gpu, "bo_viaduct");
}

/// Bridge Out's beat under `blast_parity`, as the lab fires it: the viaduct's own ball
/// (50 t at 20 m/s) along its line into the hinge pier's base, fired at frame 10. It
/// arrives at about frame 28.
/// - At frame 90, the two drop-in spans (2 and 3) are swinging down about their seats:
///   the end at the hinge pier more than 4 m down, the seated end less than 1 m, and at
///   least 90% of each span's bonded pairs still at their distance (one slab). Spans 1
///   and 4 still stand.
/// - At frame 150, both have come down on the ground at the hinge end and broken up there
///   (fewer than half their pairs intact).
#[test]
fn viaduct_pier_hit_drops_the_drop_in_spans() {
    const FIRE_AT: usize = 10;
    const AIR: usize = 90;
    const GROUND: usize = 150;
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParity;
    let structure = stress_ref::wrecking::structure("bo_viaduct").expect("viaduct");
    let scene = wrecking_yard("bo_viaduct", p, GROUND as f64 / 60.0);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    let body = &scene.bodies[0];
    let probes = stress_ref::viaduct::span_probes(&body.chunks, &body.bonds);
    let start: Vec<[f64; 3]> = body.chunks.iter().map(|c| c.center).collect();
    let shot = stress_ref::wrecking::launch(&structure.ball, &structure.aim, "ball");
    let states = |w: &mut GpuWorld| {
        w.sync(&gpu);
        let o = w.observation();
        let now: Vec<[f64; 3]> = o.bodies.values().next().expect("the structure").iter().zip(&start).map(|(c, s)| [s[0] + c.displacement[0], s[1] + c.displacement[1], s[2] + c.displacement[2]]).collect();
        probes.iter().map(|s| s.state(&start, &now)).collect::<Vec<_>>()
    };
    let mut at = Vec::new();
    for f in 0..GROUND {
        if f == FIRE_AT {
            let imp = stress_ref::world::Impactor::from_desc(&shot, &w.scene);
            w.add_impactor(&gpu, imp).expect("shot");
        }
        w.step_frame(&gpu).expect("frame");
        // Each frame's splits followed and the mirror synced, as the lab's view does every
        // display frame (stress-web `view_prepare`). Without it, until the first sync, the
        // drop-in spans came down later (2.1 m down 1 s after the shot, not 5.2 m).
        w.sync(&gpu);
        if f + 1 == AIR || f + 1 == GROUND {
            let s = states(&mut w);
            println!("frame {}: {s:?}", f + 1);
            at.push(s);
        }
    }
    // The spans' ends at the hinge pier: span 2's east, span 3's west.
    let hinge_end = |s: &stress_ref::viaduct::SpanState, k: usize| if k == 1 { (s.east_drop, s.west_drop) } else { (s.west_drop, s.east_drop) };
    for k in [1, 2] {
        let (free, seated) = hinge_end(&at[0][k], k);
        let s = &at[0][k];
        assert!(free > 4.0 && seated < 1.0, "span {}: hinge end down {free} m, seated end {seated} m", k + 1);
        assert!(s.intact * 10 >= s.pairs * 9, "span {}: {} of {} pairs intact in the air", k + 1, s.intact, s.pairs);
        let (free, _) = hinge_end(&at[1][k], k);
        let s = &at[1][k];
        assert!(free > 8.0, "span {}: hinge end down {free} m", k + 1);
        assert!(s.intact * 2 < s.pairs, "span {}: {} of {} pairs intact on the ground", k + 1, s.intact, s.pairs);
    }
    for k in [0, 3] {
        let s = &at[0][k];
        assert!(s.west_drop.abs().max(s.east_drop.abs()) < 0.3 && s.intact == s.pairs, "span {}: {s:?}", k + 1);
    }
}

/// The Skybridge's beat (`stress_ref::skybridge`) in both stress modes, its charges set
/// off before frame 10 (each at its delay) as the lab's Detonate does, each frame's splits
/// followed as the lab's view does:
/// - At frame 60 (0.83 s later) the walkway hangs from its west end: the east end of its
///   last section more than 2.5 m down, the west end of its first less than 0.5 m, every
///   section still at least 90% intact (one piece).
/// - At frame 180 every section has come down on the plaza at both ends (more than 12 m),
///   while the towers' top storeys have not moved.
#[test]
fn skybridge_charges_hang_then_drop_the_walkway() {
    const AT: usize = 10;
    const HANG: usize = 60;
    const DOWN: usize = 180;
    let gpu = Gpu::new().expect("GPU");
    for p in [Profile::BlastParity, Profile::Threshold] {
        let key = "sb_skybridge";
        let scene = wrecking_yard(key, p, DOWN as f64 / 60.0);
        let mut demolition = stress_rubble::demolition::Demolition::of_pack(&stress_ref::wrecking::pack(key).expect("pack"));
        let body = &scene.bodies[0];
        let probes = stress_ref::viaduct::span_probes(&body.chunks, &body.bonds);
        assert_eq!(probes.len(), 4);
        let start: Vec<[f64; 3]> = body.chunks.iter().map(|c| c.center).collect();
        let top: Vec<usize> = (0..start.len()).filter(|&k| body.chunks[k].groups.get(1).is_some_and(|g| g == "storey-7")).collect();
        let dt = scene.sim.frame_dt;
        let mut at: Vec<(Vec<stress_ref::viaduct::SpanState>, f64)> = Vec::new();
        let mut record = |now: &[[f64; 3]]| {
            let moved = top.iter().map(|&k| (Vec3::from_array(now[k]) - Vec3::from_array(start[k])).norm()).fold(0.0, f64::max);
            at.push((probes.iter().map(|s| s.state(&start, now)).collect(), moved));
        };
        if p == Profile::Threshold {
            let ball = stress_ref::wrecking::BALLS[0];
            let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
            for f in 0..DOWN {
                if f == AT {
                    demolition.arm("", f as u64, dt);
                }
                for c in demolition.due(f as u64) {
                    w.detonate(&gpu, &c).expect("detonate");
                }
                w.step_frame(&gpu).expect("frame");
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                let st = w.poll(&gpu);
                assert_eq!((st.error, st.engine_overflow), (0, 0), "frame {f}");
                if f + 1 == HANG || f + 1 == DOWN {
                    let now: Vec<[f64; 3]> = w.chunk_centres(&gpu).into_iter().map(|(c, _)| c).collect();
                    record(&now);
                }
            }
        } else {
            let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
            let (iterations, substeps) = p.engine_policy();
            e.set_policy(iterations, substeps);
            let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("resident world");
            w.redo_fractured_frames = false;
            w.set_impact_islands(true);
            w.apply_profile(p);
            for f in 0..DOWN {
                if f == AT {
                    demolition.arm("", f as u64, dt);
                }
                for c in demolition.due(f as u64) {
                    let bonds = c.bond_indices(&scene.bodies[0].bonds).expect("charge bonds");
                    w.cut_bonds(&gpu, 0, &bonds, &c.debris, c.kick).expect("detonate");
                }
                w.step_frame(&gpu).expect("frame");
                w.sync(&gpu);
                if f + 1 == HANG || f + 1 == DOWN {
                    let o = w.observation();
                    let now: Vec<[f64; 3]> = o.bodies.values().next().expect("the structure").iter().zip(&start).map(|(c, s)| [s[0] + c.displacement[0], s[1] + c.displacement[1], s[2] + c.displacement[2]]).collect();
                    record(&now);
                }
            }
        }
        println!("{} frame {HANG}: {:?}\n{} frame {DOWN}: {:?}", p.name(), at[0], p.name(), at[1]);
        let (hang, _) = &at[0];
        assert!(hang[3].east_drop > 2.5 && hang[0].west_drop < 0.5, "{}: east end down {} m, west end {} m", p.name(), hang[3].east_drop, hang[0].west_drop);
        for (k, s) in hang.iter().enumerate() {
            assert!(s.intact * 10 >= s.pairs * 9, "{}: section {} {} of {} pairs intact while it hangs", p.name(), k + 1, s.intact, s.pairs);
        }
        let (down, moved) = &at[1];
        for (k, s) in down.iter().enumerate() {
            assert!(s.west_drop > 12.0 && s.east_drop > 12.0, "{}: section {} {s:?}", p.name(), k + 1);
        }
        assert!(*moved < 0.01, "{}: a top storey chunk moved {moved} m", p.name());
    }
}

#[test]
fn structures_settle_in_both_stress_modes() {
    let gpu = Gpu::new().expect("GPU");
    for &(key, _) in STRUCTURES {
        blast_parity(&gpu, key);
        threshold(&gpu, key);
    }
}
