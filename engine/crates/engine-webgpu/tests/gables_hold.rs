//! Full-size support and impact regression against pinned three-avbd checkpoints.
//! The exact initial contact count also detects truncation beyond the ring's capacity.
use engine_schema::{AvbdParams, WorldCapacity, FLAG_REUSE_CONTACTS};
use engine_webgpu::WgpuSim;
mod support;
use support::make_body_collider;

#[test]
fn gables_506k_keeps_unhit_walls_standing_and_both_balls_advancing() {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 524288,
        max_colliders: 524288,
        max_contacts: 6 * 1048576,
        max_candidates: 4 * 1048576,
        max_colors: 32,
        ..WorldCapacity::small()
    })
    .expect("GPU adapter required for full-size 506k gables regression");
    let mut p = AvbdParams::default();
    p.iterations = 3;
    p.flags |= FLAG_REUSE_CONTACTS;
    p.sleep_time = 1.0e6;
    sim.set_params(p);
    let mut upload_bodies = Vec::with_capacity(505923);
    let mut upload_colliders = Vec::with_capacity(505923);
    let mut add = |pos, half, mass, q, sphere| {
        let id = upload_bodies.len() as u32;
        let (body, collider) = make_body_collider(id, pos, half, mass, q, sphere);
        upload_bodies.push(body);
        upload_colliders.push(collider);
    };
    let identity = [0.0, 0.0, 0.0, 1.0];
    add([0.0; 3], [512.0, 0.5, 512.0], 0.0, identity, false);
    let mut initial = vec![[0.0; 3]];
    // Pinned brickGables defaults: 16 columns × 68 rows × (30+...+1) bricks.
    for cx in 0..16 {
        for cy in 0..68 {
            let x0 = (cx as f64 - 7.5) * 32.0;
            let z = (cy as f64 - 33.5) * 5.0;
            for k in 0..30 {
                let m = 30 - k;
                for i in 0..m {
                    let pos = [
                        (x0 + (i as f64 - (m - 1) as f64 / 2.0) * 1.01) as f32,
                        0.75 + 0.5 * k as f32,
                        z as f32,
                    ];
                    add(pos, [0.5, 0.25, 0.25], 0.25, identity, false);
                    initial.push(pos);
                }
            }
        }
    }
    let first_ball = initial.len();
    let radius = 30.0f64 / 7.0;
    let mass = 50.0 * (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    for cx in [6.0, 9.0] {
        let pos = [
            ((cx - 7.5) * 32.0) as f32,
            (0.5 + radius) as f32,
            (-167.5 - radius - 12.0) as f32,
        ];
        add(pos, [radius as f32; 3], mass as f32, identity, true);
        initial.push(pos);
    }
    assert_eq!(
        initial.len(),
        505923,
        "pinned full gables scene must contain 505,920 bricks, two balls and the ground"
    );
    drop(add);
    assert_eq!(sim.add_bodies(&upload_bodies), 0..505923);
    assert_eq!(sim.add_colliders(&upload_colliders), 0..505923);
    eprintln!("gables: uploaded 505923 bodies and colliders in three contiguous GPU writes");
    for step in 1..=240 {
        let counters = sim.step().expect("full-size gables GPU step");
        assert_eq!(counters.overflow_flags, 0,
            "gables step {step}: overflow 0x{:x}; do not truncate contacts, candidates or coloring to fit the scene", counters.overflow_flags);
        assert_eq!(
            counters.awake_count, 505922,
            "gables step {step}: all bodies must remain active for scale verification"
        );
        if step == 1 {
            // Every upper brick has two rectangular supports (2×435 per gable),
            // plus 30 ground supports. Four points each, and two sphere contacts.
            assert_eq!(counters.contact_count, 1088 * (2 * 435 + 30) * 4 + 2,
                "gables first frame lost exact-touch supports: expected four points per rectangular overlap and one per ball; inspect clipping, tiled dispatch and capacity");
        }
        if ![1, 10, 20, 60, 120, 240].contains(&step) {
            continue;
        }
        sim.validate_contact_coloring()
            .expect("gables active contact endpoints must have distinct valid colors");
        let bodies = sim.download_bodies();
        let mut max_drift = 0.0f32;
        let mut unhit_total = 0.0f64;
        let mut unhit_count = 0u32;
        let mut unhit_max = 0.0f32;
        let mut unhit_worst = 1usize;
        let mut unhit_over_limit = 0u32;
        for (id, body) in bodies.iter().enumerate().skip(1) {
            assert!(
                body.position
                    .iter()
                    .chain(body.orientation.iter())
                    .chain(body.linear_vel.iter())
                    .chain(body.angular_vel.iter())
                    .all(|v| v.is_finite()),
                "gables step {step}: body {id} has nonfinite pose or velocity"
            );
            if id < first_ball {
                let drift = (body.position[1] - initial[id][1]).abs();
                max_drift = max_drift.max(drift);
                let column = (id - 1) / (68 * 465);
                if column != 6 && column != 9 {
                    unhit_total += f64::from(drift);
                    unhit_count += 1;
                    if drift > unhit_max {
                        unhit_max = drift;
                        unhit_worst = id;
                    }
                    if drift >= 0.6 {
                        unhit_over_limit += 1;
                    }
                }
                assert!(step > 20 || drift < 0.3,
                    "gables step {step}: brick {id} moved vertically {drift} m before impact; initial={:?}, actual={:?}; check support contacts and warm starts", initial[id], body.position);
            }
        }
        let mean_drift = unhit_total / f64::from(unhit_count);
        assert!(mean_drift < 0.2 && unhit_max < 0.6,
            "gables step {step}: unhit columns collapsed (mean height drift={mean_drift:.6} m, max={unhit_max:.6} m, bricks over 0.60 m={unhit_over_limit}); worst brick {unhit_worst}, column={}, initial={:?}, actual={:?}, velocity={:?}; pinned three-avbd at step 240 is mean 0.118 m / max 0.328 m; check support, warm starts and coloring",
            (unhit_worst - 1) / (68 * 465), initial[unhit_worst], bodies[unhit_worst].position, bodies[unhit_worst].linear_vel);
        let reference_forward = match step {
            1 => -183.28572,
            10 => -178.78572,
            20 => -173.78595,
            60 => -154.0713,
            120 => -126.1219,
            240 => -74.0568,
            _ => unreachable!(),
        };
        for (side, id) in [first_ball, first_ball + 1].into_iter().enumerate() {
            let pos = bodies[id].position;
            assert!((pos[0] - initial[id][0]).abs() < 1.0 && (4.6..4.95).contains(&pos[1])
                && (pos[2] - reference_forward).abs() < 1.5,
                "gables step {step}: impactor {side} diverged from pinned three-avbd: actual={pos:?}, expected forward={reference_forward:.4} ±1.5 m, height 4.6..4.95 m; check rolling direction, mass, friction and reused contact anchors");
        }
        eprintln!("gables step {step}: contacts={} awake={} recolor={} colors={} max brick drift={max_drift:.6} unhit mean={:.6} max={unhit_max:.6} balls={:?}/{:?}", counters.contact_count, counters.awake_count, counters._pad0, counters.color_count, unhit_total / f64::from(unhit_count), bodies[first_ball].position, bodies[first_ball+1].position);
    }
}
