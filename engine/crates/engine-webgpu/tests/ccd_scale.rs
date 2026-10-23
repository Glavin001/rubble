//! CCD at scale on the shared engine (WgpuSim, rigid_plan): a field of brick
//! walls (dynamic, asleep) and hundreds of balls fired across it at 200 m/s.
//! Timing with plain and with CCD balls; every CCD ball checked every step for
//! a phase-through (its centre path meeting a brick that did not move).
//!
//!   cargo test --release -p engine-webgpu --test ccd_scale -- --nocapture
//!   (the 512-ball benchmark: -- --ignored)

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use engine_webgpu::WgpuSim;
use std::collections::HashMap;
use std::time::Instant;

#[path = "support/phase.rs"]
mod phase;

const DT: f32 = 1.0 / 60.0;

fn body(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: inv_i, flags, linear_vel: v, sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn collider(body: u32, shape: ShapeType, p: [f32; 4]) -> Collider {
    Collider { body, shape: shape as u32, material: 0, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: p }
}

struct Field {
    sim: WgpuSim,
    balls: Vec<u32>,
    shots: Vec<([f32; 3], [f32; 3])>,
}

/// `side`×`side` walls of 20×10 bricks (0.4×0.2×0.2 m, 1 kg), 6 m apart, then
/// `balls` parked balls (0.15 m, 5 kg); settled until the walls sleep.
fn field(side: u32, balls: u32, ccd: bool) -> Field {
    let bricks = side * side * 200;
    let cap = WorldCapacity {
        max_bodies: bricks + balls + 64,
        max_colliders: bricks + balls + 64,
        max_contacts: 1 << 21,
        max_candidates: 1 << 21,
        max_colors: 32,
        ..WorldCapacity::small()
    };
    let mut sim = WgpuSim::new(cap).expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = DT;
    p.flags |= FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    let g = sim.add_body(body([0.0, -1.0, 0.0], 0.0, [0.0; 3], 0, [0.0; 3]));
    sim.add_collider(collider(g, ShapeType::Box, [400.0, 1.0, 400.0, 400.0]));
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
                    let id = sim.add_body(body([x, hy + course as f32 * 2.0 * hy, cz], 1.0 / m, ii, 9, [0.0; 3]));
                    sim.add_collider(collider(id, ShapeType::Box, [hx, hy, hz, (hx * hx + hy * hy + hz * hz).sqrt()]));
                }
            }
        }
    }
    let mut ids = Vec::new();
    for k in 0..balls {
        let id = sim.add_body(body([1000.0 + k as f32, -500.0, 0.0], 0.0, [0.0; 3], if ccd { BODY_CCD } else { 0 }, [0.0; 3]));
        sim.add_collider(collider(id, ShapeType::Sphere, [0.15, 0.0, 0.0, 0.15]));
        ids.push(id);
    }
    // Settle until the walls sleep.
    for i in 0..90 {
        sim.step().unwrap_or_else(|e| panic!("settle step {i} (ccd {ccd}): {e:?}"));
    }
    // Shots: from outside the field along ±z (through the walls' faces), at
    // brick heights, deterministic.
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
    Field { sim, balls: ids, shots }
}

fn launch(f: &mut Field) {
    let mut all = f.sim.download_bodies();
    for (i, &id) in f.balls.iter().enumerate() {
        let s = &mut all[id as usize];
        let (p, v) = f.shots[i];
        let i_ball = 2.5 / (5.0 * 0.15 * 0.15);
        s.position = p;
        s.initial_pos = p;
        s.linear_vel = v;
        s.inv_mass = 0.2;
        s.inv_inertia = [i_ball; 3];
        s.flags = 9 | (s.flags & BODY_CCD);
        f.sim.write_bodies(id, &[*s]);
    }
}

/// ms/step over `steps` after the launch, and (when `check`) the phase-throughs.
fn run(side: u32, balls: u32, ccd: bool, steps: u32, check: bool) -> (f64, Vec<String>) {
    let mut f = field(side, balls, ccd);
    launch(&mut f);
    let cols: Vec<Collider> = (0..f.sim.collider_count()).map(|i| f.sim.collider(i)).collect();
    let mut prev = f.sim.download_bodies();
    let mut failures = Vec::new();
    let mut ms = 0.0;
    for step in 0..steps {
        let t = Instant::now();
        let res = f.sim.step();
        res.unwrap_or_else(|e| panic!("step {step} (ccd {ccd}, {balls} balls): {e:?}"));
        ms += t.elapsed().as_secs_f64() * 1e3;
        if check {
            let now = f.sim.download_bodies();
            // Bricks bucketed by 1 m cells: only those near each path are tested.
            let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
            for (ci, c) in cols.iter().enumerate() {
                if c.shape == ShapeType::Box as u32 && c.body != 0 {
                    let p = prev[c.body as usize].position;
                    grid.entry((p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32)).or_default().push(ci);
                }
            }
            for &ball in &f.balls {
                let (a, b) = (prev[ball as usize].position, now[ball as usize].position);
                let mut near = Vec::new();
                let lo = [0, 1, 2].map(|k| a[k].min(b[k]).floor() as i32 - 1);
                let hi = [0, 1, 2].map(|k| a[k].max(b[k]).floor() as i32 + 1);
                if (0..3).map(|k| (hi[k] - lo[k] + 1) as i64).product::<i64>() > 200_000 {
                    continue;
                }
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
    }
    (ms / steps as f64, failures)
}

fn compare(side: u32, balls: u32, steps: u32) {
    let (plain, plain_through) = run(side, balls, false, steps, true);
    let (ccd, ccd_through) = run(side, balls, true, steps, true);
    let (plain_t, _) = run(side, balls, false, steps, false);
    let (ccd_t, _) = run(side, balls, true, steps, false);
    println!(
        "{} walls ({} bricks), {balls} balls at 200 m/s: plain {plain_t:.2} ms/step ({} phase-throughs), CCD {ccd_t:.2} ms/step ({} phase-throughs), CCD {:+.2} ms ({:+.0}%)",
        side * side,
        side * side * 200,
        plain_through.len(),
        ccd_through.len(),
        ccd_t - plain_t,
        100.0 * (ccd_t / plain_t - 1.0)
    );
    let _ = (plain, ccd);
    assert!(plain_through.len() > balls as usize / 4, "control: plain balls should phase through bricks ({})", plain_through.len());
    assert!(ccd_through.is_empty(), "CCD balls phased through:\n{}", ccd_through[..ccd_through.len().min(10)].join("\n"));
}

#[test]
fn a_hundred_ccd_balls_through_a_field_of_brick_walls() {
    compare(4, 128, 40);
}

#[test]
#[ignore = "benchmark: 64 walls, 512 balls"]
fn five_hundred_ccd_balls_through_sixty_four_walls() {
    compare(8, 512, 40);
}
