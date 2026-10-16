//! MetalSim replays the shared rigid step graph (engine_exec::rigid_plan) on
//! native Metal. Same settled-columns gate as engine-webgpu's columns_hold, so
//! both hosts are held to one bar; plus pipelined submission must match
//! step-by-step submission on a deterministic scene.
use engine_metal::MetalSim;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_REUSE_CONTACTS};

fn columns(n: u32, sleep_enabled: bool) -> (MetalSim, Vec<BodyState>) {
    let h = 10u32;
    let bodies = n * n * h + 1;
    let mut sim = MetalSim::new(WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: bodies.next_power_of_two().max(4096),
        max_contacts: (bodies * 16).next_power_of_two().min(1 << 21),
        max_joints: 64,
        max_candidates: (bodies * 32).next_power_of_two().min(1 << 21),
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("Metal device");
    let mut params = AvbdParams::default();
    params.iterations = 4;
    params.flags |= FLAG_REUSE_CONTACTS;
    if !sleep_enabled {
        params.sleep_time = 1.0e6;
    }
    sim.set_params(params);
    let (half_h, hx) = (25.0f32, 200.0f32);
    let mut all = vec![BodyState {
        position: [0.0, 0.5 - half_h, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0; 3],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    }];
    let mut cols = vec![Collider {
        body: 0,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [hx, half_h, hx, hx.max(half_h)],
    }];
    let inertia = 1.0f32 / 6.0;
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                let pos = [(x as f32 - n as f32 / 2.0) * 1.5, 1.0 + y as f32, (z as f32 - n as f32 / 2.0) * 1.5];
                cols.push(Collider {
                    body: all.len() as u32,
                    shape: ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 1.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [0.5, 0.5, 0.5, 0.75f32.sqrt()],
                });
                all.push(BodyState {
                    position: pos,
                    inv_mass: 1.0,
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    inv_inertia: [1.0 / inertia; 3],
                    flags: 1 | 8,
                    linear_vel: [0.0; 3],
                    sleep_timer: 0.0,
                    angular_vel: [0.0; 3],
                    generation: 0,
                    initial_pos: pos,
                    initial_pad: 0.0,
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                });
            }
        }
    }
    sim.add_bodies(&all);
    sim.add_colliders(&cols);
    (sim, all)
}

fn columns_hold(n: u32, sleep_enabled: bool) {
    let (mut sim, initial) = columns(n, sleep_enabled);
    let dynamic = n * n * 10;
    for s in 0..180u32 {
        let c = sim.step().expect("step without overflow");
        if !sleep_enabled {
            assert_eq!(c.awake_count, dynamic, "step {s}: active-body coverage lost to sleep");
        }
        if s == 0 || s % 30 == 29 {
            let snap = sim.download_bodies();
            let (mut ke, mut max_v) = (0.0f32, 0.0f32);
            for (id, b) in snap.iter().enumerate().skip(1) {
                assert!(b.position.iter().chain(&b.linear_vel).all(|x| x.is_finite()), "step {s}: body {id} nonfinite");
                let drift = (b.position[1] - initial[id].position[1]).abs();
                assert!(drift < 0.15, "step {s}, body {id}: column layer drifted {drift:.4} m");
                for axis in [0, 2] {
                    assert!((b.position[axis] - initial[id].position[axis]).abs() < 0.1, "step {s}, body {id}: column toppled sideways");
                }
                let v2: f32 = b.linear_vel.iter().map(|v| v * v).sum();
                ke += 0.5 * v2 / b.inv_mass;
                max_v = max_v.max(v2.sqrt());
            }
            if s == 179 {
                assert!(max_v < 0.1, "columns did not settle: maxV={max_v}");
                assert!(ke < 0.001 * dynamic as f32, "columns gained energy: KE={ke}");
            }
        }
    }
}

#[test]
fn metalsim_columns_10k_hold_height() {
    columns_hold(32, true);
}

#[test]
fn metalsim_columns_100k_stay_stable_with_sleep_disabled() {
    columns_hold(100, false);
}

/// Pipelined submission forwards last step's counts on the GPU; it must give
/// the same trajectory as waiting for every step.
#[test]
fn metalsim_pipelined_matches_waiting() {
    let (mut waited, _) = columns(16, false);
    let (mut piped, _) = columns(16, false);
    for _ in 0..40 {
        waited.step().expect("step");
        piped.step_submit().expect("submit");
    }
    piped.drain().expect("drain");
    let (a, b) = (waited.download_bodies(), piped.download_bodies());
    for (id, (x, y)) in a.iter().zip(&b).enumerate() {
        for k in 0..3 {
            assert!((x.position[k] - y.position[k]).abs() < 1e-5, "body {id}: waited {:?} vs pipelined {:?}", x.position, y.position);
        }
    }
}
