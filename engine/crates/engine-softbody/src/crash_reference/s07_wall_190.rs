//! S7: full-width rigid concrete block, 190 km/h (video 5:32).
//!
//! First principles: 52.8 m/s carries (190/50)² = 14× the energy of a 50 km/h test (2.0 MJ on
//! the 1,448 kg wagon). The crush zone absorbs about a 60–70 km/h impact's worth
//! ((65/190)² ≈ 12 %), so the rest goes through the cabin and the rear; stopping within about
//! 2 m is v²/2d ≈ 70 g for 2d/v ≈ 80 ms. The centre of mass rides above the crush force, so
//! the rear pitches up hard, toward standing the car on its nose.
//!
//! Sign conventions as S01 (pitch positive nose down).

use super::*;

const KMH: f32 = 190.0;
/// The generated wagon's mass as the CPU reference spawns it (kg, measured 2026-11-27).
pub(super) const WAGON_KG: f64 = 1448.0;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S07",
        title: "Full-width rigid block, 190 km/h",
        reference: Reference { video: VIDEO, at_s: &[332], real_test: "rigid concrete block, full width, 190 km/h (no standard test)" },
        designs: &["wagon"],
        layout: |extent, j| {
            let e = extent("wagon");
            let v = KMH / 3.6 * j.speed;
            // A tenth of a second of travel before the block.
            let gap = 0.1 * v;
            let face = e.hi.z + gap;
            Layout {
                bodies: vec![Body::new("car", "wagon", Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v)],
                // A concrete block 2 m tall (above the roof), 2 m deep, a metre past each side.
                boxes: vec![(Vec3::new(0.0, 1.0, face + 1.0), Vec3::new(0.5 * e.width() + 1.0, 1.0, 1.0))],
                impact_s: 0.1,
            }
        },
        // Pitched up to its peak by 0.6 s, back on its wheels by 1.25 s, at rest (under 1 kJ)
        // by 1.8 s.
        secs_after_impact: 2.0,
        // World coordinates: the wagon's front is 0.96 m ahead of its origin, the block's face
        // 5.28 m beyond (z ≈ 6.2).
        cameras: &[
            Camera { name: "side", eye: [11.0, 1.8, 4.0], target: [0.0, 1.0, 4.0] },
            Camera { name: "aftermath", eye: [-6.0, 3.5, -1.0], target: [0.0, 0.5, 4.5] },
            Camera { name: "top", eye: [0.01, 14.0, 4.0], target: [0.0, 0.0, 4.0] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "front_gone", after_impact_ms: 20.0 },
            Moment { name: "peak", after_impact_ms: 60.0 },
            Moment { name: "on_its_nose", after_impact_ms: 600.0 },
            Moment { name: "back_down", after_impact_ms: 1250.0 },
            Moment { name: "rest", after_impact_ms: 1950.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(KE_J * 1e-3), "at most 0.1 % of the 2.02 MJ the 1,448 kg wagon brings at 52.8 m/s").law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
                Criterion::new(Invariant, "nothing passes into the block", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside the block: a node moves 2.6 cm a step at 52.8 m/s, so 5 cm is two steps of a missed contact").law("a static box is impenetrable"),
                Criterion::new(Invariant, "no explosion", "car.max_node_speed", None, Some(158.0), "3 × the 52.8 m/s approach (see S01)"),
                Criterion::new(Physics, "the car is crushed to under 60 % of its length", "car.gauge.length.shrink_frac", Some(0.4), None, "a 1.4–2.9 m crush (v²/2a at 50–100 g) of the 4.85 m car; a crush zone sized for 65 km/h takes 0.6 m of it"),
                Criterion::new(Physics, "the crush stays", "car.crush_residual", Some(1.3), None, "metal does not spring back: 70 % of the 0.4 × 4.85 m = 1.94 m floor above (along the first heading: thrown-off parts and the pitch distort it)"),
                Criterion::new(Physics, "the cabin is crushed", "car.zone.cabin.work_frac", Some(0.3), None, "the front absorbs about 12 % of the energy ((65/190)²), the cabin most of the rest: 0.3 is a floor at ×0.5 of 0.6"),
                Criterion::new(Physics, "left door opening collapses", "car.gauge.door_left.shrink_frac", Some(0.3), None, "the crush runs through the cabin: the door apertures close by a third or more"),
                Criterion::new(Physics, "right door opening collapses", "car.gauge.door_right.shrink_frac", Some(0.3), None, "full width and symmetric: the right as the left"),
                Criterion::new(Physics, "the rear is damaged too", "car.zone.rear.bent_frac", Some(0.01), None, "at 60–70 g the rear's own inertia and the cabin's collapse reach it: at least 1 % of its beams yield (S10's must be lower)"),
                Criterion::new(Physics, "parts come off", "car.detached_parts", Some(1.0), None, "14× a 50 km/h test's energy tears at least one part (door, wheel, panel) from the body"),
                Criterion::new(Physics, "rebound", "car.rebound_ratio", None, Some(0.15), "restitution falls with impact speed (Monson & Germane, crash-reference-targets.md §1): below the 0.13–0.20 measured at 56 km/h"),
                Criterion::new(Physics, "the rear pitches up", "car.pitch_peak_deg", Some(20.0), None, "the centre of mass above the crush force's line: its moment over the pulse pitches the rear up past 20°"),
                Criterion::new(Shape, "stands on its nose", "car.pitch_peak_deg", Some(60.0), Some(90.0), "the frames show the car standing on its nose"),
                Criterion::new(Shape, "parts on the ground", "car.detached_parts", Some(3.0), None, "doors and a wheel on the ground in the aftermath"),
                Criterion::new(Shape, "half its length left", "car.gauge.length.shrink_frac", Some(0.4), Some(0.65), "the wreck is about half the car's length"),
            ]
        },
    }
}

/// ½ m v² of the wagon at 52.8 m/s (J).
const KE_J: f64 = 0.5 * WAGON_KG * (KMH as f64 / 3.6) * (KMH as f64 / 3.6);
