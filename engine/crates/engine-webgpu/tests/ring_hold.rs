//! Full paper-scale impact, with all bodies active. The opposite side must stand
//! while the sphere crosses the near wall; max-height alone cannot detect collapse.
use engine_schema::{AvbdParams, WorldCapacity, FLAG_REUSE_CONTACTS};
use engine_webgpu::WgpuSim;

mod support;
use support::add;

#[test]
fn ring_110k_stands_away_from_impact_and_ball_rolls_through() {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 131072,
        max_colliders: 131072,
        max_contacts: 2097152,
        max_candidates: 2097152,
        max_colors: 32,
        ..WorldCapacity::small()
    })
    .expect("GPU adapter required for the full 110k ring regression");
    let mut p = AvbdParams::default();
    p.iterations = 4;
    p.flags |= FLAG_REUSE_CONTACTS;
    p.sleep_time = 1.0e6; // Accuracy and cost must not be hidden by sleeping the ring.
    sim.set_params(p);
    add(
        &mut sim,
        [0.0; 3],
        [180.0, 0.5, 180.0],
        0.0,
        [0.0, 0.0, 0.0, 1.0],
        false,
    );
    for course in 0..40 {
        for row in 0..10 - course / 4 {
            let r = 80.25 + 0.52 * row as f64;
            let count = (std::f64::consts::TAU * (r - 0.25) / 1.02).floor() as u32;
            for i in 0..count {
                let a =
                    (i as f64 + (course % 2) as f64 * 0.5) / count as f64 * std::f64::consts::TAU;
                let yaw = -(a + std::f64::consts::FRAC_PI_2);
                add(
                    &mut sim,
                    [
                        (r * a.cos()) as f32,
                        0.75 + 0.5 * course as f32,
                        (r * a.sin()) as f32,
                    ],
                    [0.5, 0.25, 0.25],
                    0.25,
                    [0.0, (yaw * 0.5).sin() as f32, 0.0, (yaw * 0.5).cos() as f32],
                    false,
                );
            }
        }
    }
    let ball = add(
        &mut sim,
        [0.0, 8.5, -95.2],
        [8.0; 3],
        10.0 * 4.0 / 3.0 * std::f32::consts::PI * 8.0f32.powi(3),
        [0.0, 0.0, 0.0, 1.0],
        true,
    );
    assert_eq!(ball, 110441, "ring benchmark body count changed");
    let initial = sim.download_bodies();
    for step in 0..480 {
        let counters = sim
            .step()
            .unwrap_or_else(|e| panic!("ring step {step}: capacity/dispatch failure: {e:?}"));
        assert_eq!(
            counters.awake_count, ball,
            "ring step {step}: active workload unexpectedly slept"
        );
        if [0, 29, 59, 119, 239, 479].contains(&step) {
            sim.validate_contact_coloring()
                .unwrap_or_else(|why| panic!("ring step {step}: invalid parallel solve: {why}"));
            let bodies = sim.download_bodies();
            let mut max_drift = 0.0f32;
            let mut worst = 0;
            let mut total_drift = 0.0f32;
            let mut flank_count = 0;
            for (id, b) in bodies.iter().enumerate().skip(1) {
                assert!(
                    b.position
                        .iter()
                        .chain(b.orientation.iter())
                        .chain(b.linear_vel.iter())
                        .chain(b.angular_vel.iter())
                        .all(|v| v.is_finite()),
                    "ring step {step}, body {id}: nonfinite pose or velocity"
                );
                // Both flanks are >40 m from the impactor's center line.
                if initial[id].position[0].abs() > 40.0 {
                    let drift = (b.position[1] - initial[id].position[1]).abs();
                    total_drift += drift;
                    flank_count += 1;
                    if drift > max_drift {
                        max_drift = drift;
                        worst = id;
                    }
                }
            }
            let mean_drift = total_drift / flank_count as f32;
            let b = &bodies[ball as usize];
            eprintln!("ring step {step}: contacts={} pairs={} recolor={} colors={} flank drift={max_drift:.4} mean={mean_drift:.4} worst={worst} ball={:?} velocity={:?}",counters.contact_count,counters.candidate_count,counters._pad0,counters.color_count,b.position,b.linear_vel);
            if step == 0 {
                assert!(counters.contact_count > 1_250_000 && counters.contact_count < 1_320_000,
                    "first ring frame has {} contacts; pinned three-avbd has ~1.285M. Check exact-touch SAT and full clipped polygons", counters.contact_count);
            }
            // The reference's impact spreads to some flank bricks after 2 s. Its
            // mean flank drift is ~0.23 m at 4 s and ~1.02 m at 8 s (see regression notes).
            let mean_limit = if step <= 119 {
                0.2
            } else if step <= 239 {
                0.5
            } else {
                1.6
            };
            assert!(mean_drift<mean_limit,"ring step {step}: widespread collapse: mean flank drift={mean_drift} m, reference-based limit={mean_limit} m");
            assert!(step > 119 || max_drift<0.5,"ring step {step}: untouched flank body {worst} fell {max_drift:.3} m; initial={:?}, actual={:?}. Check scene handedness, current-step dispatch coverage, and warm-start support points.",initial[worst].position,bodies[worst].position);
            assert!(
                b.position[1] > 7.5 && b.position[1] < 9.5,
                "ring step {step}: rolling sphere sank into or launched off the floor: {:?}",
                b.position
            );
            if step == 479 {
                assert!(b.position[2]>90.0,"sphere failed to traverse both ring walls after 8 s: position={:?}, velocity={:?}; check mass and angular-velocity sign",b.position,b.linear_vel);
            }
        }
    }
}
