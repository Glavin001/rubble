//! What continuous collision costs at scale in the GPU-resident frame (the fracture lab's
//! `blast_parity` path): `--bodies` free concrete pieces (0.2 m cubes, one chunk each) resting
//! on the ground in a square field, `--ccd-bodies` of them marked as fast movers
//! (`BodyDesc::ccd`), and `--balls` CCD steel balls fired low across the field at
//! `--speed` m/s, so the frames hold TOI pairs, pieces launched within a step (the late
//! sweep) and pieces broken off CCD bodies. Every frame timed end to end (the GPU waited
//! on); one JSON line per run: median, p95 and worst frame, and the frames' resident share.
//!
//!   cargo run --release -p stress-rubble --features gpu --example ccd_scale_cost -- \
//!       [--bodies 10000] [--ccd-bodies 0] [--balls 0] [--speed 250] [--frames 60]

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::builders::*;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_rubble::RubbleEngine;

fn scene(bodies: usize, ccd_bodies: usize, balls: usize, speed: f64, seconds: f64) -> Scene {
    let mut s = new_scene("ccd_scale_cost", "a field of pieces swept by fast balls");
    s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let side = (bodies as f64).sqrt().ceil() as usize;
    let pitch = 0.35;
    for i in 0..bodies {
        let (x, y) = ((i % side) as f64 * pitch, (i / side) as f64 * pitch);
        let c = grid(Vec3::new(x, y, 0.0), [1, 1, 1], Vec3::splat(0.2), "concrete");
        let mut b = body(&format!("piece{i}"), c, Vec::new());
        // Spread the marked ones over the field.
        b.ccd = ccd_bodies > 0 && i % (bodies / ccd_bodies).max(1) == 0 && i / (bodies / ccd_bodies).max(1) < ccd_bodies;
        s.bodies.push(b);
    }
    // Balls along x, one per row (cycling), at the pieces' height, starting just off the field.
    for k in 0..balls {
        let row = (k * side / balls.max(1)).min(side - 1);
        s.impactors.push(ImpactorDesc {
            name: format!("ball{k}"),
            shape: ImpactorShape::Sphere { radius: 0.08 },
            mass: 20.0,
            position: [-1.0 - 0.3 * (k / side) as f64, row as f64 * pitch + 0.1, 0.1],
            orientation: [1.0, 0.0, 0.0, 0.0],
            velocity: [speed, 0.0, 0.0],
            angular_velocity: [0.0; 3],
            material: "steel".into(),
            crush: None,
            ccd: true,
        });
    }
    let mut s = stress_rubble::scenes::realtime(s, seconds);
    Profile::BlastParity.apply_scene(&mut s);
    s
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let (mut bodies, mut ccd_bodies, mut balls, mut speed, mut frames) = (10_000usize, 0usize, 0usize, 250.0f64, 60usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("a value");
        match a.as_str() {
            "--bodies" => bodies = v().parse().expect("--bodies N"),
            "--ccd-bodies" => ccd_bodies = v().parse().expect("--ccd-bodies N"),
            "--balls" => balls = v().parse().expect("--balls N"),
            "--speed" => speed = v().parse().expect("--speed M/S"),
            "--frames" => frames = v().parse().expect("--frames N"),
            other => panic!("unknown argument {other}"),
        }
    }
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let gpu = Gpu::new().expect("GPU");
    let profile = Profile::BlastParity;
    let s = scene(bodies, ccd_bodies, balls, speed, frames as f64 / 60.0);
    let mut e = RubbleEngine::with_device_capacity(Vec3::from_array(s.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&s)).expect("rubble engine");
    e.apply_profile(profile);
    let mut w = GpuWorld::with_resident_engine(&gpu, &s, e).expect("resident world");
    w.redo_fractured_frames = false;
    w.apply_profile(profile);
    let mut times = Vec::with_capacity(frames);
    for _ in 0..frames {
        let t = Instant::now();
        w.step_frame(&gpu).expect("frame");
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    // The first frames build the tables and pipelines: skipped.
    let mut warm = times[frames.min(5)..].to_vec();
    warm.sort_by(f64::total_cmp);
    println!(
        "{{\"bodies\":{bodies},\"ccd_bodies\":{ccd_bodies},\"balls\":{balls},\"speed\":{speed},\"frames\":{},\"resident_frames\":{},\"median_ms\":{:.3},\"p95_ms\":{:.3},\"worst_ms\":{:.3},\"first_ms\":{:.1}}}",
        warm.len(),
        w.resident_frames,
        percentile(&warm, 0.5),
        percentile(&warm, 0.95),
        warm[warm.len() - 1],
        times[0]
    );
}
