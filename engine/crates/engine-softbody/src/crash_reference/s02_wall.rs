//! S2: an SUV into a full-width rigid wall at 50 km/h (Euro NCAP full width; video 2:10): the
//! generated 4x4 (`suv_4x4`, 2,000 kg kerb; `engine/docs/generated-vehicles-lessons.md` §22).
//!
//! First principles: 13.9 m/s stopped by a front built for 17–33 g gives 0.3–0.6 m of crush;
//! the load is symmetric, so the car neither yaws nor drifts; the wall does not move.

use super::*;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S02".into(),
        title: "SUV into a full-width rigid wall, 50 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[130], real_test: "Euro NCAP full-width rigid barrier, 50 km/h" },
        designs: names(&["suv_4x4"]),
        recreated: false,
        modified_from: None,
        // A wall 4 m tall and 1 m deep, a metre past each side, a tenth of a second ahead.
        layout: family::preset(&families::BLOCK, &[("car", "suv_4x4".into()), ("speed", 50.0.into()), ("overlap", 1.0.into()), ("height", 4.0.into()), ("depth", 1.0.into()), ("lead", 0.1.into())]),
        secs_after_impact: 0.5,
        cameras: vec![
            // Framed as the reference footage (60° vertical field of view, 16:9): the SUV (z −2.3 to
            // 2.2 at impact) fills about two thirds of the width, the wall at the left of the side view.
            Camera { name: "side".into(), eye: [4.6, 1.1, -0.3], target: [0.0, 0.8, -0.3] },
            // Straight down, the car's length across the frame, travelling right to left into the wall
            // at the left as in the reference's top view (the eye a centimetre to +x of the target
            // fixes the picture's up as −x).
            Camera { name: "top".into(), eye: [0.01, 5.6, 0.0], target: [0.0, 0.8, 0.0] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            Moment { name: "peak".into(), after_impact_ms: 70.0 },
            Moment { name: "rest".into(), after_impact_ms: 450.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Physics, "crush at its deepest", "car.crush_peak", Some(0.25), Some(0.8), "v²/2a at 13.9 m/s and 17–33 g, widened; NHTSA 56 km/h dynamic crush 0.62–0.75 m"),
                // Reported, not asserted: 0.13–0.20 is the p10–p90 of sedans at 56 km/h (a tenth of real
                // cars are above it), and this run is at 50 km/h, measured 0.5 s after contact.
                Criterion::new(Physics, "rebound", "car.rebound_ratio", Some(-0.05), Some(0.2), "a plastic crush: NHTSA restitution 0.13–0.20 at 56 km/h (crash-reference-targets.md)"),
                Criterion::new(Invariant, "the crush stays", "car.crush_residual", Some(0.1), None, "metal does not spring back"),
            ]
        },
    }
}
