//! Settled columns must preserve every layer across GPU workgroups, without energy growth.
use engine_schema::{
    AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_REUSE_CONTACTS,
};
use engine_webgpu::WgpuSim;

fn columns_hold(n: u32, sleep_enabled: bool) {
    let h = 10u32;
    let bodies = n * n * h + 1;
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: bodies.next_power_of_two().max(4096),
        max_contacts: (bodies * 16).next_power_of_two().min(1 << 21),
        max_joints: 64,
        max_candidates: (bodies * 32).next_power_of_two().min(1 << 21),
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut params = AvbdParams::default();
    params.iterations = 4;
    params.flags |= FLAG_REUSE_CONTACTS;
    if !sleep_enabled {
        params.sleep_time = 1.0e6;
    }
    sim.set_params(params);

    let half_h = 25.0f32;
    let hx = 200.0f32;
    sim.add_body(BodyState {
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
    });
    sim.add_collider(Collider {
        body: 0,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [hx, half_h, hx, hx.max(half_h)],
    });

    let hx_b = 0.5f32;
    let mass = 1.0f32;
    let inertia = (1.0 / 6.0) * mass;
    let r = (0.75f32).sqrt();
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                let pos = [
                    (x as f32 - n as f32 / 2.0) * 1.5,
                    1.0 + y as f32,
                    (z as f32 - n as f32 / 2.0) * 1.5,
                ];
                let id = sim.add_body(BodyState {
                    position: pos,
                    inv_mass: 1.0 / mass,
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
                sim.add_collider(Collider {
                    body: id,
                    shape: ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 1.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [hx_b, hx_b, hx_b, r],
                });
            }
        }
    }

    let initial = sim.download_bodies();
    let dynamic_count = n * n * h;
    // Three seconds includes warm start, sleep transition, and subsequent steps.
    // 15 cm permits normal contact compression but rejects a lost box layer.
    for s in 0..180 {
        let c = sim.step().expect("step without overflow");
        if !sleep_enabled {
            assert_eq!(
                c.awake_count, dynamic_count,
                "step {s}: active-body coverage lost to sleep"
            );
        }
        if s == 0 || s % 30 == 29 {
            let snapshot = sim.download_bodies();
            let mut max_drift = 0.0f32;
            let mut ymax = f32::NEG_INFINITY;
            let mut ke = 0.0f32;
            let mut max_v = 0.0f32;
            for (i, b) in snapshot.iter().skip(1).enumerate() {
                assert!(
                    b.position
                        .iter()
                        .chain(b.orientation.iter())
                        .chain(b.linear_vel.iter())
                        .chain(b.angular_vel.iter())
                        .all(|x| x.is_finite()),
                    "step {s}: body {} has a nonfinite pose or velocity",
                    i + 1
                );
                let id = i + 1;
                let expected_y = initial[id].position[1];
                let drift = (b.position[1] - expected_y).abs();
                assert!(drift < 0.15,
                    "step {s}, body {id}/{dynamic_count}: column layer drifted {drift:.4} m; expected y={expected_y}, actual pose={:?}. Check CSR offsets and warm-start support anchors.", b.position);
                for axis in [0, 2] {
                    assert!(
                        (b.position[axis] - initial[id].position[axis]).abs() < 0.1,
                        "step {s}, body {id}: column toppled sideways: initial={:?}, actual={:?}",
                        initial[id].position,
                        b.position
                    );
                }
                max_drift = max_drift.max((b.position[1] - expected_y).abs());
                ymax = ymax.max(b.position[1]);
                let v2 = b.linear_vel.iter().map(|v| v * v).sum::<f32>();
                ke += 0.5 * v2 / b.inv_mass;
                max_v = max_v.max(v2.sqrt());
            }
            println!("n={} t={s}: maxY={ymax:.4} drift={max_drift:.4} KE={ke:.4} maxV={max_v:.4} contacts={} awake={} ov=0x{:x}", n*n*h, c.contact_count, c.awake_count, c.overflow_flags);
            if s == 179 {
                assert!(max_v < 0.1, "columns did not settle: maxV={max_v}");
                assert!(
                    ke < 0.001 * (n * n * h) as f32,
                    "columns gained energy: KE={ke}"
                );
            }
        }
    }
}

#[test]
fn columns_10k_holds_height() {
    columns_hold(32, true);
}

#[test]
fn columns_25k_holds_height() {
    columns_hold(50, true);
}

#[test]
fn columns_100k_holds_height() {
    columns_hold(100, true);
}

#[test]
fn columns_100k_stays_stable_with_sleep_disabled() {
    columns_hold(100, false);
}
