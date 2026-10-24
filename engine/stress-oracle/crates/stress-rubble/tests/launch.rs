//! Launched bonded bodies on the GPU (`stress_ref::launch`, a pack's `defaults.launches`),
//! released between frames as stress-web releases them (`stress_launch`): the check scene's
//! free concrete ram (`stress_ref::launch::ram_and_wall`) flies at its launch velocity and
//! breaks bonds of the wall and of itself where it hits.
//!
//! * threshold (the GPU-resident world, `ThresholdWorld::launch_island`, its strike sphere
//!   in the strike prediction);
//! * blast_parity (the resident GpuWorld, `GpuWorld::launch_island`).
//!
//!   cargo test --release -p stress-rubble --features gpu --test launch
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::launch::Launcher;
use stress_rubble::scenes::pack_scene;

/// The ram is released before this frame (a few frames standing first).
const AT: u64 = 5;
/// Frames after the release at which its flight is measured (before it reaches the wall:
/// 4 m at 30 m/s is 8 frames).
const FLIGHT: u64 = 6;
const WALL: std::ops::Range<usize> = 1..17;
const RAM: std::ops::Range<usize> = 17..22;

fn mean(pos: &[Vec3], ks: std::ops::Range<usize>) -> Vec3 {
    let n = ks.len() as f64;
    ks.fold(Vec3::ZERO, |a, k| a + pos[k]) * (1.0 / n)
}

#[test]
fn launched_ram_flies_and_breaks_the_wall_and_itself_threshold() {
    let gpu = Gpu::new().expect("GPU");
    let pack = stress_ref::launch::ram_and_wall();
    let frames = 60;
    let scene = pack_scene(&pack, "launch_ram_wall", Profile::Threshold, frames as f64 / 60.0);
    let mut launcher = Launcher::of_pack(&pack).expect("launches");
    let v = launcher.launches[0].velocity;
    let ball = stress_ref::wrecking::BALLS[0];
    let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
    let centres = |w: &stress_rubble::threshold::ThresholdWorld| -> Vec<Vec3> { w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect() };
    let mut released = None;
    for f in 0..frames {
        if w.frame == AT {
            assert!(launcher.arm("", w.frame, w.frame_dt));
            assert!(!launcher.arm("nothing", w.frame, w.frame_dt));
        }
        for l in launcher.due(w.frame) {
            assert_eq!(w.launch_island(&gpu, 0, &l).expect("launch"), 1, "the ram is one body");
            released = Some(centres(&w));
        }
        w.step_frame(&gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let st = w.poll(&gpu);
        assert_eq!((st.error, st.engine_overflow), (0, 0), "frame {f}");
        if w.frame == AT {
            assert_eq!(st.broken, 0, "nothing broken before the ram arrives");
        }
        if w.frame == AT + FLIGHT {
            // In flight: its centre has moved by its launch velocity (and gravity's fall).
            let (a, b) = (mean(released.as_ref().unwrap(), RAM), mean(&centres(&w), RAM));
            let t = FLIGHT as f64 * w.frame_dt;
            let expected = Vec3::from_array(v) * t + Vec3::new(0.0, 0.0, -0.5 * 9.81 * t * t);
            let err = (b - a - expected).norm();
            println!("threshold: after {FLIGHT} frames the ram moved {:.3?} m, expected {:.3?} (off by {err:.4} m)", (b - a).to_array(), expected.to_array());
            assert!(err < 0.02 * expected.norm(), "flies at its launch velocity: off by {err} m");
        }
    }
    let st = w.poll(&gpu);
    let body = w.fracture.chunk_bodies(&gpu);
    let wall_bodies: std::collections::BTreeSet<u32> = WALL.map(|k| body[k]).collect();
    let ram_bodies: std::collections::BTreeSet<u32> = RAM.map(|k| body[k]).collect();
    let pos = centres(&w);
    println!("threshold: {frames} frames, broken {} of {}, bodies {}; the wall in {} bodies, the ram in {}; the ram at y {:.2} m", st.broken, scene.bodies[0].bonds.len(), st.bodies, wall_bodies.len(), ram_bodies.len(), mean(&pos, RAM).y);
    assert!(wall_bodies.len() > 1, "the wall broke where the ram hit");
    assert!(ram_bodies.len() > 1, "the ram broke on the wall");
}

#[test]
fn launched_ram_flies_and_breaks_the_wall_and_itself_blast_parity() {
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParity;
    let pack = stress_ref::launch::ram_and_wall();
    let frames = 60;
    let scene = pack_scene(&pack, "launch_ram_wall", p, frames as f64 / 60.0);
    let mut launcher = Launcher::of_pack(&pack).expect("launches");
    let v = launcher.launches[0].velocity;
    let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
    let (iterations, substeps) = p.engine_policy();
    e.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    let displacement = |w: &mut GpuWorld| -> Vec<Vec3> {
        w.sync(&gpu);
        w.observation().bodies.values().next().unwrap().iter().map(|c| Vec3::from_array(c.displacement)).collect()
    };
    let mut released = None;
    for _ in 0..frames {
        if w.frame == AT {
            assert!(launcher.arm("ram", w.frame, scene.sim.frame_dt));
        }
        for l in launcher.due(w.frame) {
            assert_eq!(w.launch_island(&gpu, 0, &l.chunks, l.velocity, l.angular_velocity).expect("launch"), 1);
            assert!(w.launch_island(&gpu, 0, &[1, 2], l.velocity, l.angular_velocity).is_err(), "wall chunks are no free island");
            released = Some(displacement(&mut w));
        }
        w.step_frame(&gpu).expect("frame");
        if w.frame == AT + FLIGHT {
            let (a, b) = (mean(released.as_ref().unwrap(), RAM), mean(&displacement(&mut w), RAM));
            let t = FLIGHT as f64 * scene.sim.frame_dt;
            let expected = Vec3::from_array(v) * t + Vec3::new(0.0, 0.0, -0.5 * 9.81 * t * t);
            let err = (b - a - expected).norm();
            println!("blast_parity: after {FLIGHT} frames the ram moved {:.3?} m, expected {:.3?} (off by {err:.4} m)", (b - a).to_array(), expected.to_array());
            assert!(err < 0.02 * expected.norm(), "flies at its launch velocity: off by {err} m");
            assert_eq!(w.observation().values["broken_bonds"], 0.0, "nothing broken before the ram arrives");
        }
    }
    w.sync(&gpu);
    let bonds = &scene.bodies[0].bonds;
    let m = &w.solver.mirror;
    let broken = |pick: &dyn Fn(usize) -> bool| (0..bonds.len()).filter(|&b| pick(bonds[b].a) && pick(bonds[b].b) && !m.bonds[0][b].connected()).count();
    let (wall, ram) = (broken(&|k| WALL.contains(&k)), broken(&|k| RAM.contains(&k)));
    println!("blast_parity: {frames} frames, broken {} (wall {wall}, ram {ram}), fragments {}", w.observation().values["broken_bonds"], w.observation().values["fragments"]);
    assert!(wall > 0, "the wall broke where the ram hit");
    assert!(ram > 0, "the ram broke on the wall");
}
