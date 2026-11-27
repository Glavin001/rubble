//! S12: the generated sedan into a full-width rigid wall at 40 km/h (NHTSA's FMVSS 208 frontal
//! rigid-barrier compliance tests at 40 km/h; the CCSA Camry's 40 km/h validation). Reference
//! data: NHTSA tests 9692 (2015 Camry, 1,690 kg, 39.6
//! km/h) and 7793 (2011 Camry Hybrid, previous body, 1,798 kg, 39.6 km/h),
//! `engine/docs/crash-reference/nhtsa/<test>.json`.
//!
//! The car is the 1,448 kg generated sedan, 14–19 % lighter than those two, and no fleet data
//! at 40 km/h is held: only the laws are asserted; every curve and scalar of the two tests is
//! reported beside ours.
//!
//! First principles: 11.11 m/s, 89 kJ; crush scales with speed above the no-damage speed
//! (Campbell, targets §2.2): (40 − 8)/(56.3 − 8) = 0.66 of the 56 km/h crush.

use super::s11_ncap_frontal_56::{frontal_laws, wall_layout};
use super::*;

pub const KMH: f32 = 40.0;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S12",
        title: "Sedan into a full-width rigid wall, 40 km/h",
        reference: Reference { video: VIDEO, at_s: &[], real_test: "NHTSA FMVSS 208 rigid barrier, 40 km/h (tests 9692, 7793: Toyota Camry)" },
        designs: &["sedan"],
        layout: |extent, j| wall_layout(extent("sedan"), KMH / 3.6 * j.speed, j),
        secs_after_impact: 0.4,
        cameras: &[
            Camera { name: "side", eye: [9.0, 1.2, -1.5], target: [0.0, 0.7, -1.5] },
            Camera { name: "top", eye: [0.01, 12.0, -1.5], target: [0.0, 0.0, -1.5] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "peak", after_impact_ms: 50.0 },
            Moment { name: "max_crush", after_impact_ms: 72.0 },
            Moment { name: "rest", after_impact_ms: 350.0 },
        ],
        criteria: || {
            use Tier::*;
            let mut c = frontal_laws(89.0);
            c.extend([
                // Reported: two Camrys, not a fleet.
                Criterion::new(Shape, "dynamic crush (Camry)", "car.ref.dyn_crush", Some(0.475), Some(0.489), "9692 0.475 m, 7793 0.489 m"),
                Criterion::new(Shape, "time of maximum crush (Camry)", "car.ref.stop_ms", Some(69.3), Some(74.7), "9692 69.3 ms, 7793 74.7 ms"),
                Criterion::new(Shape, "restitution (Camry)", "car.ref.restitution", Some(0.107), Some(0.149), "9692 0.149, 7793 0.107 (−min v from t_m to 200 ms over v0)"),
                Criterion::new(Shape, "peak deceleration, sampled as Rubble samples (Camry)", "car.ref.peak_decel_g", Some(26.1), Some(27.1), "9692 27.1 g, 7793 26.1 g (CFC 60 itself: 34.1, 31.5 g)"),
                Criterion::new(Shape, "residual crush (Camry, overall length)", "car.crush_residual", Some(0.279), Some(0.304), "9692 304 mm, 7793 279 mm"),
            ]);
            c
        },
    }
}
