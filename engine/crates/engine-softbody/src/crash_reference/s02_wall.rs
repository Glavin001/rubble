//! S2: an SUV into a full-width rigid wall at 50 km/h (Euro NCAP full width; video 2:10): the
//! generated 4x4 (`suv_4x4`, 2,000 kg kerb; `engine/docs/generated-vehicles-lessons.md` §21).
//!
//! First principles: 13.9 m/s stopped by a front built for 17–33 g gives 0.3–0.6 m of crush;
//! the load is symmetric, so the car neither yaws nor drifts; the wall does not move.

use super::*;

const KMH: f32 = 50.0;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S02",
        title: "SUV into a full-width rigid wall, 50 km/h",
        reference: Reference { video: VIDEO, at_s: &[130], real_test: "Euro NCAP full-width rigid barrier, 50 km/h" },
        designs: &["suv_4x4"],
        layout: |extent, j| {
            let e = extent("suv_4x4");
            let v = KMH / 3.6 * j.speed;
            // A tenth of a second of travel before the wall.
            let gap = 0.1 * v;
            let face = e.hi.z + gap;
            Layout {
                bodies: vec![Body::new("car", "suv_4x4", Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v)],
                boxes: vec![(Vec3::new(0.0, 2.0, face + 0.5), Vec3::new(0.5 * e.width() + 1.0, 2.0, 0.5))],
                impact_s: 0.1,
            }
        },
        secs_after_impact: 0.5,
        cameras: &[
            Camera { name: "side", eye: [9.0, 1.2, -1.5], target: [0.0, 0.7, -1.5] },
            Camera { name: "top", eye: [0.01, 12.0, -1.5], target: [0.0, 0.0, -1.5] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "peak", after_impact_ms: 70.0 },
            Moment { name: "rest", after_impact_ms: 450.0 },
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
