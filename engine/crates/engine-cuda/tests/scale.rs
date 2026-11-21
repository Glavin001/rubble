//! The WebGPU host's rigid scale gates on CUDA (`engine-webgpu/tests/ring_hold.rs`,
//! `ccd_scale.rs`): the same scenes, the same assertions, run by `CudaSim`. The ring
//! comes from `engine-scenes` (the shared builder the demos and Metal use), the CCD
//! field is built here as in `ccd_scale.rs`, and phase-throughs are found by the same
//! detector (`engine-webgpu/tests/support/phase.rs`).
//!
//! GPU tests run when `ENGINE_CUDA_DEVICE=1` (Modal sets it); `ENGINE_CUDA_PERF=1`
//! prints ms a step.
use engine_cuda::CudaSim;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use std::collections::HashMap;
use std::time::Instant;

#[path = "../../engine-webgpu/tests/support/phase.rs"]
mod phase;

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host");
        return false;
    }
    assert!(CudaSim::available(), "ENGINE_CUDA_DEVICE=1 but no device code or no NVIDIA device");
    true
}

/// A scene of `engine-scenes` on CUDA: its bodies, colliders, materials and settings.
/// CudaSim has no joints or convex hulls yet (docs/world-on-cuda.md, C3).
fn scene(name: &str, capacity: WorldCapacity) -> (CudaSim, AvbdParams) {
    let (host, info) = engine_scenes::build_host(name).expect("scene");
    assert!(host.joints.is_empty(), "{name}: CudaSim has no joints yet");
    let mut sim = CudaSim::new(capacity).expect("CudaSim");
    let mut params = host.params;
    params.iterations = info.iterations;
    sim.set_params(params);
    sim.set_materials(&host.materials).expect("materials");
    sim.add_bodies(&host.bodies);
    sim.add_colliders(&host.colliders);
    (sim, params)
}

fn finite(b: &BodyState) -> bool {
    b.position.iter().chain(&b.orientation).chain(&b.linear_vel).chain(&b.angular_vel).all(|v| v.is_finite())
}

/// `ring_hold.rs`'s gate: the paper's 110k brick ring, every body awake (sleep off),
/// a rolling sphere through both walls in 8 s while the far flanks stand.
#[test]
fn cuda_ring_110k_stands_away_from_impact_and_ball_rolls_through() {
    if !gpu() {
        return;
    }
    let (mut sim, mut p) = scene(
        "Brick Ring (110k)",
        WorldCapacity {
            max_bodies: 131072,
            max_colliders: 131072,
            max_contacts: 2097152,
            max_candidates: 2097152,
            max_colors: 32,
            ..WorldCapacity::small()
        },
    );
    p.sleep_time = 1.0e6; // Accuracy and cost must not be hidden by sleeping the ring.
    sim.set_params(p);
    let initial = sim.download_bodies();
    let ball = initial.len() as u32 - 1;
    assert_eq!(ball, 110441, "ring benchmark body count changed");
    assert!(initial[ball as usize].linear_vel[2] > 0.0, "the last body is the rolling sphere");
    let t0 = Instant::now();
    for step in 0..480 {
        let counters = sim.step().unwrap_or_else(|e| panic!("ring step {step}: capacity/dispatch failure: {e:?}"));
        assert_eq!(counters.awake_count, ball, "ring step {step}: active workload unexpectedly slept");
        if ![0, 29, 59, 119, 239, 479].contains(&step) {
            continue;
        }
        sim.validate_contact_coloring().unwrap_or_else(|why| panic!("ring step {step}: invalid parallel solve: {why}"));
        let bodies = sim.download_bodies();
        let (mut max_drift, mut worst, mut total, mut flank) = (0.0f32, 0usize, 0.0f32, 0u32);
        for (id, b) in bodies.iter().enumerate().skip(1) {
            assert!(finite(b), "ring step {step}, body {id}: nonfinite pose or velocity");
            if initial[id].position[0].abs() > 40.0 {
                let drift = (b.position[1] - initial[id].position[1]).abs();
                total += drift;
                flank += 1;
                if drift > max_drift {
                    max_drift = drift;
                    worst = id;
                }
            }
        }
        let mean = total / flank as f32;
        let b = &bodies[ball as usize];
        eprintln!(
            "ring step {step}: contacts={} colors={} flank drift={max_drift:.4} mean={mean:.4} worst={worst} ball={:?}",
            counters.contact_count, counters.color_count, b.position
        );
        if step == 0 {
            assert!(
                counters.contact_count > 1_250_000 && counters.contact_count < 1_320_000,
                "first ring frame has {} contacts; the reference has ~1.285M",
                counters.contact_count
            );
        }
        let limit = if step <= 119 { 0.2 } else if step <= 239 { 0.5 } else { 1.6 };
        assert!(mean < limit, "ring step {step}: widespread collapse: mean flank drift {mean} m, limit {limit} m");
        assert!(step > 119 || max_drift < 0.5, "ring step {step}: untouched flank body {worst} fell {max_drift:.3} m");
        assert!(b.position[1] > 7.5 && b.position[1] < 9.5, "ring step {step}: sphere left the floor: {:?}", b.position);
        if step == 479 {
            assert!(b.position[2] > 90.0, "sphere failed to traverse both ring walls after 8 s: {:?}", b.position);
        }
    }
    eprintln!(
        "ring 110k on {}: 480 steps, {:.2} ms a step (with checkpoints' readbacks)",
        sim.device_name(),
        t0.elapsed().as_secs_f64() * 1e3 / 480.0
    );
}

const DT: f32 = 1.0 / 60.0;

fn body(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: inv_i, flags, linear_vel: v, sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn collider(body: u32, shape: ShapeType, p: [f32; 4]) -> Collider {
    Collider { body, shape: shape as u32, material: 0, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: p }
}

/// `ccd_scale.rs`'s field: `side`² walls of 200 bricks, `balls` spheres parked far away.
fn field(side: u32, balls: u32, ccd: bool) -> (CudaSim, Vec<u32>, Vec<([f32; 3], [f32; 3])>) {
    let bricks = side * side * 200;
    let mut sim = CudaSim::new(WorldCapacity {
        max_bodies: bricks + balls + 64,
        max_colliders: bricks + balls + 64,
        max_contacts: 1 << 21,
        max_candidates: 1 << 21,
        max_colors: 32,
        ..WorldCapacity::small()
    })
    .expect("CudaSim");
    let mut p = AvbdParams::default();
    p.dt = DT;
    p.flags |= FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    let mut bodies = vec![body([0.0, -1.0, 0.0], 0.0, [0.0; 3], 0, [0.0; 3])];
    let mut cols = vec![collider(0, ShapeType::Box, [400.0, 1.0, 400.0, 400.0])];
    let (hx, hy, hz) = (0.2f32, 0.1f32, 0.1f32);
    let m = 1.0f32;
    let ii = [3.0 / (m * (hy * hy + hz * hz)), 3.0 / (m * (hx * hx + hz * hz)), 3.0 / (m * (hx * hx + hy * hy))];
    let off = (side as f32 - 1.0) * 3.0;
    for wx in 0..side {
        for wz in 0..side {
            let (cx, cz) = (wx as f32 * 6.0 - off, wz as f32 * 6.0 - off);
            for course in 0..10 {
                for i in 0..20 {
                    let x = cx - 4.0 + i as f32 * 0.4 + if course % 2 == 1 { 0.2 } else { 0.0 };
                    let id = bodies.len() as u32;
                    bodies.push(body([x, hy + course as f32 * 2.0 * hy, cz], 1.0 / m, ii, 9, [0.0; 3]));
                    cols.push(collider(id, ShapeType::Box, [hx, hy, hz, (hx * hx + hy * hy + hz * hz).sqrt()]));
                }
            }
        }
    }
    let mut ids = Vec::new();
    for k in 0..balls {
        let id = bodies.len() as u32;
        bodies.push(body([1000.0 + k as f32, -500.0, 0.0], 0.0, [0.0; 3], if ccd { BODY_CCD } else { 0 }, [0.0; 3]));
        cols.push(collider(id, ShapeType::Sphere, [0.15, 0.0, 0.0, 0.15]));
        ids.push(id);
    }
    sim.add_bodies(&bodies);
    sim.add_colliders(&cols);
    for i in 0..90 {
        sim.step().unwrap_or_else(|e| panic!("settle step {i} (ccd {ccd}): {e:?}"));
    }
    let mut k = 12345u64;
    let mut shots = Vec::new();
    for _ in 0..balls {
        k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = |s: u32| ((k >> s) & 0xFFFF) as f32 / 65535.0;
        let x = -off - 4.0 + r(16) * (2.0 * off + 8.0);
        let y = 0.25 + r(32) * 1.5;
        let dir = if r(48) < 0.5 { 1.0 } else { -1.0 };
        shots.push(([x, y, -dir * (off + 8.0)], [0.0, 0.0, dir * 200.0]));
    }
    (sim, ids, shots)
}

fn run(side: u32, balls: u32, ccd: bool, steps: u32, check: bool) -> (f64, Vec<String>) {
    let (mut sim, ids, shots) = field(side, balls, ccd);
    let mut all = sim.download_bodies();
    for (i, &id) in ids.iter().enumerate() {
        let s = &mut all[id as usize];
        let (p, v) = shots[i];
        let i_ball = 2.5 / (5.0 * 0.15 * 0.15);
        s.position = p;
        s.initial_pos = p;
        s.linear_vel = v;
        s.inv_mass = 0.2;
        s.inv_inertia = [i_ball; 3];
        s.flags = 9 | (s.flags & BODY_CCD);
        let one = [*s];
        sim.write_bodies(id, &one);
    }
    let cols: Vec<Collider> = sim.colliders().to_vec();
    let mut prev = sim.download_bodies();
    let (mut failures, mut ms) = (Vec::new(), 0.0);
    for step in 0..steps {
        let t = Instant::now();
        sim.step().unwrap_or_else(|e| panic!("step {step} (ccd {ccd}, {balls} balls): {e:?}"));
        ms += t.elapsed().as_secs_f64() * 1e3;
        if !check {
            continue;
        }
        let now = sim.download_bodies();
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (ci, c) in cols.iter().enumerate() {
            if c.shape == ShapeType::Box as u32 && c.body != 0 {
                let p = prev[c.body as usize].position;
                grid.entry((p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32)).or_default().push(ci);
            }
        }
        for &ball in &ids {
            let (a, b) = (prev[ball as usize].position, now[ball as usize].position);
            let lo = [0, 1, 2].map(|k| a[k].min(b[k]).floor() as i32 - 1);
            let hi = [0, 1, 2].map(|k| a[k].max(b[k]).floor() as i32 + 1);
            if (0..3).map(|k| (hi[k] - lo[k] + 1) as i64).product::<i64>() > 200_000 {
                continue;
            }
            let mut near = Vec::new();
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        if let Some(v) = grid.get(&(x, y, z)) {
                            near.extend(v.iter().map(|&ci| cols[ci]));
                        }
                    }
                }
            }
            if let Some(why) = phase::crossing(&near, &prev, &now, ball as usize) {
                failures.push(format!("step {step}: {why}"));
            }
        }
        prev = now;
    }
    (ms / steps as f64, failures)
}

fn compare(side: u32, balls: u32, steps: u32) {
    let (_, plain_through) = run(side, balls, false, steps, true);
    let (_, ccd_through) = run(side, balls, true, steps, true);
    let (plain_t, _) = run(side, balls, false, steps, false);
    let (ccd_t, _) = run(side, balls, true, steps, false);
    println!(
        "CUDA: {} walls ({} bricks), {balls} balls at 200 m/s: plain {plain_t:.2} ms/step ({} phase-throughs), CCD {ccd_t:.2} ms/step ({} phase-throughs)",
        side * side,
        side * side * 200,
        plain_through.len(),
        ccd_through.len()
    );
    assert!(plain_through.len() > balls as usize / 4, "control: plain balls should phase through bricks ({})", plain_through.len());
    assert!(ccd_through.is_empty(), "CCD balls phased through:\n{}", ccd_through[..ccd_through.len().min(10)].join("\n"));
}

#[test]
fn cuda_a_hundred_ccd_balls_through_a_field_of_brick_walls() {
    if gpu() {
        compare(4, 128, 40);
    }
}

#[test]
fn cuda_five_hundred_ccd_balls_through_sixty_four_walls() {
    if gpu() {
        compare(8, 512, 40);
    }
}
