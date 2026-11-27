//! S14: the US NCAP side test: the FMVSS 214 moving deformable barrier (design `mdb214`,
//! 1,367.6 kg, `crate::crash_rigs`) crabbed 27° at 38.5 mph (61.96 km/h) into the driver's
//! (left) side of the standing generated sedan (NCAP side MDB, the FMVSS 214 S7 configuration at
//! the NCAP speed). Reference data: NHTSA tests 7517 (2012 Camry, 1,641 kg test mass, barrier
//! 1,362 kg, 62.3 km/h) and 8560 (2014 Camry, 1,649 kg, 61.94 km/h),
//! `engine/docs/crash-reference/nhtsa/<test>.json`.
//!
//! Set up as the web crash lab's `mdb214` scene (`engine-world-wasm` `explicit.rs`) and S03:
//! the barrier faces −X (yaw −90°, square to the car's side), its wheels crabbed 27° so it
//! moves along (−cos 27°, 0, −sin 27°): across the car and toward its rear; the face's forward
//! edge on FMVSS 214's impact reference line (S8.10: 940 mm ahead of the wheelbase's middle up
//! to a 2,896 mm wheelbase) when it reaches the side's outermost plane, after a run-up of
//! [`RUN_UP_S`] in which the car settles on its suspension.
//!
//! The struck car is 1,448 kg, 12 % lighter than the tests' 1,641–1,649 kg; with the same
//! barrier, a plastic hit sends it off at M·V_x/(M + m) across, faster than a Camry. Every curve
//! is reported; only the laws are asserted (the 14 sedans of FMVSS 214, targets §10, are at
//! 54 km/h, not the NCAP 62).
//!
//! First principles. Momentum M·V = 23,538 N·s along the barrier's travel is shared, ground
//! friction and drag only taking it away; no collision sends the car off faster than an
//! elastic one, 2·M·V/(M + m) = 16.72 m/s; energy (202.6 kJ) only leaves; the struck side
//! takes the work.

use super::*;
use crate::crash_rigs::{MDB_CRAB_DEG, MDB_FACE};

/// NCAP side MDB: 38.5 mph.
pub const SPEED: f32 = 38.5 * 0.447_04;
/// The barrier's run-up before contact (s), as S03's.
const RUN_UP_S: f32 = 0.4;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S14",
        title: "NCAP side: moving deformable barrier crabbed 27° into a standing sedan, 62 km/h",
        reference: Reference { video: VIDEO, at_s: &[], real_test: "NHTSA NCAP side MDB, 62 km/h crabbed 27° (tests 7517, 8560: 2012-14 Toyota Camry)" },
        designs: &["sedan", "mdb214"],
        layout: |extent, j| {
            let (e, b) = (extent("sedan"), extent("mdb214"));
            let v = SPEED * j.speed;
            let (front, rear) = e.axles.unwrap_or((e.lo.z + 0.2 * e.length(), e.lo.z + 0.8 * e.length()));
            let irl = if front - rear <= 2.896 { 0.5 * (front + rear) + 0.94 } else { front - 0.508 };
            let face_z = irl - 0.5 * MDB_FACE[0] + j.lateral;
            let crab = MDB_CRAB_DEG.to_radians();
            let dir = Vec3::new(-crab.cos(), 0.0, -crab.sin());
            // Its origin when its face reaches the side's outermost plane, and a run-up back
            // along its travel.
            let contact = Vec3::new(e.hi.x + b.hi.z, 0.0, face_z);
            let mut barrier = Body::new("barrier", "mdb214", contact - dir * (v * RUN_UP_S), -0.5 * std::f32::consts::PI + j.yaw, v);
            barrier.vel = Some(dir * v);
            Layout { bodies: vec![Body::new("car", "sedan", Vec3::ZERO, 0.0, 0.0), barrier], boxes: Vec::new(), impact_s: RUN_UP_S }
        },
        secs_after_impact: 0.5,
        cameras: &[
            // As S03's: ahead of the car and out on the struck side; above; on its axis ahead.
            Camera { name: "front three-quarter", eye: [5.5, 1.6, 3.5], target: [0.9, 0.7, -1.3] },
            Camera { name: "top", eye: [0.01, 12.0, -1.6], target: [0.0, 0.0, -1.6] },
            Camera { name: "side", eye: [0.0, 1.0, 8.0], target: [0.0, 0.7, -1.4] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "peak", after_impact_ms: 40.0 },
            Moment { name: "deepest", after_impact_ms: 80.0 },
            Moment { name: "rest", after_impact_ms: 450.0 },
        ],
        // M = 1,367.6 kg, m = 1,448.4 kg, V = 17.211 m/s (jitter: speed ±2 %).
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(203.0), "0.1 % of the barrier's 202.6 kJ (½·1,367.6·17.211²)")
                    .law("mechanical energy never rises in a crash: contact, plasticity, damping and friction only remove it"),
                Criterion::new(Invariant, "momentum along the barrier's travel", "all.momentum_along_end", Some(0.0), Some(24_033.0),
                    "starts at M·V = 23,538 N·s (24,009 at +2 % speed; 24,033 with 0.1 % for rounding); ground friction and drag only take it away and cannot reverse it")
                    .law("momentum is conserved between car and barrier; the only outside forces (ground friction, air drag) oppose the motion"),
                Criterion::new(Physics, "the car leaves no faster than an elastic hit", "car.speed_peak", None, Some(17.05),
                    "2·M·V/(M + m) = 16.72 m/s, +2 % for the speed jitter: no collision gives more")
                    .law("an elastic collision transfers the most momentum: v_car ≤ 2·M·V/(M + m)"),
                Criterion::new(Physics, "the struck side absorbs more than the far side", "car.zone.lr_ratio", Some(1.0), None, "left (struck) plastic work over right")
                    .law("the load enters through the struck side; the far side sees only what passes through the cabin"),
                // Reported: two Camrys (heavier by 12 %).
                Criterion::new(Shape, "car's lateral velocity change, largest (Camry)", "car.com.dvy_peak", Some(6.60), Some(6.79), "7517 6.60 m/s at 88 ms, 8560 6.79 m/s at 110 ms (CG Y, +right); a plastic hit sends a 1,448 kg car off at M·V cos 27°/(M + m) = 7.45 m/s across"),
                Criterion::new(Shape, "car's lateral velocity change by 150 ms (Camry)", "car.com.dvy_150ms", Some(6.22), Some(6.35), "7517 6.22, 8560 6.35 m/s"),
                Criterion::new(Shape, "car's lateral peak acceleration at the CG, sampled as Rubble samples (Camry)", "car.com.ay_peak_g", Some(23.6), Some(27.4), "7517 23.6 g, 8560 27.4 g: their CG Y velocity (CFC 60) every 4 ms, differenced as the suite does (CFC 60 itself: 31.8, 35.6 g; ours on the CPU: car.com.cfc60_ay_peak_g)"),
                Criterion::new(Shape, "barrier's velocity change by 150 ms (Camry)", "barrier.com.dvx_150ms", Some(-8.56), Some(-8.15), "7517 −8.56, 8560 −8.15 m/s along its axis (MDB CG X)"),
                Criterion::new(Shape, "struck side's residual crush (Camry)", "car.side_crush_left_residual", Some(0.258), Some(0.275), "7517 275 mm, 8560 258 mm (the reports' maximum exterior crush, mid door)"),
                Criterion::new(Shape, "no rollover", "car.roll_peak_deg", None, Some(25.0), "the face's middle (0.56 m up) is near the car's centre of mass"),
            ]
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::crash_rigs::MDB_MASS;

    /// The bands' numbers follow from the masses and the speed (the sedan's 1,448.4 kg, S03's).
    #[test]
    fn s14_bands_from_masses() {
        let (bm, m, v) = (f64::from(MDB_MASS), 1448.4, f64::from(super::SPEED));
        let near = |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol, "{a} vs {b}");
        near(bm * v, 23_538.0, 1.0);
        near(bm * v * 1.02 * 1.001, 24_033.0, 1.0);
        near(2.0 * bm * v / (bm + m), 16.72, 5e-3);
        near(2.0 * bm * v / (bm + m) * 1.02, 17.05, 5e-3);
        near(0.5 * bm * v * v * 1e-3, 202.6, 0.1);
        near(bm * v * 27f64.to_radians().cos() / (bm + m), 7.45, 5e-3);
    }
}
