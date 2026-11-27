//! S4: a mid-size sedan into the rear of a parked semi-trailer, full overlap, 56 km/h
//! (video 2:59; the IIHS underride test: a 2010 Chevrolet Malibu into a parked trailer).
//!
//! The car is the generated sedan (`car`, m = 1,448.4 kg, as S03 measures it); the trailer
//! the generated 53 ft dry van (`box_trailer`, `trailer`, M = 15,000 kg laden: kerb 6,350 +
//! 8,650 on the deck; `generated-vehicles-lessons.md` §24), parked on its landing gear and
//! its tandem, its rear impact guard built to FMVSS 223/224 (the bar's lower edge 0.52 m up
//! at design, its supports collapsing at the regulation's 350 kN distributed). Centred, the
//! car square to the trailer. Nothing brakes: the scenario sets no inputs, so the trailer's
//! tandem rolls free (a parked trailer's spring brakes are on: not modelled) and only its
//! landing gear's feet grip.
//!
//! First principles. The trailer is ten times the car (M/m = 10.4): close to a wall, but one
//! whose load path, the guard and above it the rear sill at deck height (1.0-1.27 m), starts
//! above the car's bumper beam. A plastic hit leaves both at m·V/(m + M) = 1.37 m/s; with
//! restitution e the trailer leaves at (1 + e)·m·V/(m + M), at most the elastic 2·m·V/(m +
//! M) = 2.74 m/s (e ≤ 1), and friction only lowers it. The car cannot end faster forward
//! than it came: the trailer only pushes it back and the ground only resists.
//!
//! Masses and bands: `tests::bands_from_masses`.

use super::*;

// The car's run-up (0.5 s): the trailer settles on its landing gear and tandem from the
// spawn's 2 cm drop, and the car on its wheels, before the hit.
pub const AFTER_IMPACT_S: f32 = 0.8;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S04".into(),
        title: "Sedan into the rear of a parked semi-trailer, 56 km/h, full overlap".into(),
        reference: Reference { video: VIDEO, at_s: &[179, 181], real_test: "IIHS rear underride guard test, full width, a mid-size sedan at 56 km/h into a parked trailer" },
        designs: names(&["sedan", "box_trailer"]),
        recreated: false,
        modified_from: None,
        // The trailer heads +Z, its rear a run-up ahead of the car's front, full overlap.
        layout: family::preset(&families::UNDERRIDE, &[("car", "sedan".into()), ("trailer", "box_trailer".into()), ("speed", 56.0.into()), ("overlap", 1.0.into()), ("run_up", 0.5.into())]),
        secs_after_impact: AFTER_IMPACT_S,
        // The car's front reaches the trailer's rear at z ≈ 1.0 + 7.8 = 8.8 (the sedan's
        // front is 0.96 m ahead of its origin).
        cameras: vec![
            Camera { name: "side".into(), eye: [-9.5, 1.3, 8.0], target: [0.0, 0.9, 8.0] },
            Camera { name: "aftermath".into(), eye: [-5.5, 2.4, 3.5], target: [0.0, 0.8, 8.0] },
            Camera { name: "top".into(), eye: [0.01, 13.0, 7.0], target: [0.0, 0.0, 7.0] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            // the guard meets the bumper; the hood and fenders crush above it
            Moment { name: "early_crush".into(), after_impact_ms: 30.0 },
            Moment { name: "peak".into(), after_impact_ms: 80.0 },
            Moment { name: "rest".into(), after_impact_ms: 750.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                // Asserted (laws).
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(182.0), "0.1 % of the car's 175.2 kJ (½·1,448.4·15.556²), 182 kJ at +2 % speed").law("mechanical energy never rises in a crash: contact, plasticity, damping and friction only remove it")
                    .known_failure(&[Contact::Edge], "CPU reference 2026-11-27, 8 of 8 runs: 2.2-43 kJ (engine/docs/crash-reference.md, Currently failing)")
                    .known_intermittent(&[Contact::Continuous], "CPU reference 2026-11-27: 3 of 8 runs (0, 1, 3), up to 516 J (engine/docs/crash-reference.md, Currently failing)")
                    .known_intermittent(&[Contact::Discrete], "CPU reference 2026-11-27: 1 of 8 runs (2), 221 J (engine/docs/crash-reference.md, Currently failing)"),
                Criterion::new(Invariant, "momentum along the road", "all.momentum_z_end", Some(-9_500.0), Some(23_400.0),
                    "starts at m·V = 22,531 kg·m/s (22,982 at +2 % speed). Only the ground acts from outside: the landing gear's feet, sliding once pushed, at most μ·N = 0.6 × 57.0 kN (the ground's static 0.85 × the nodes' 0.7; the feet carry 5,810 kg) over the 0.8 s after contact, 27.4 kN·s, and every tyre's rolling resistance, at most 0.02 × 161 kN × 1.3 s = 4.2 kN·s, either way; a backward-rolling car's resistance adds at most 0.02 × 14.2 kN × 1.3 s = 370 N·s")
                    .law("momentum changes only by the external impulse, the ground's friction and rolling resistance, bounded by μ·N and c_rr·W over the run"),
                Criterion::new(Physics, "the trailer leaves no faster than an elastic hit sends it", "trailer.speed_peak", None, Some(2.80),
                    "(1 + e)·m·V/(m + M) ≤ 2·m·V/(m + M) = 2.74 m/s for e ≤ 1 (2.80 at +2 % speed); friction only lowers it")
                    .law("momentum and restitution e ≤ 1 between two bodies: the struck body's speed is (1 + e)·m·V/(m + M)"),
                Criterion::new(Physics, "the car ends no faster forward than it came", "car.v_along_end", None, Some(15.87),
                    "15.556 m/s (15.87 at +2 % speed): the trailer only pushes it back, the ground only resists")
                    .law("no force on the car points along its travel: the trailer's contact pushes it back, friction and drag oppose its motion"),
                // Reported (estimates and the frames).
                Criterion::new(Physics, "the trailer hardly moves", "trailer.speed_peak", None, Some(1.40), "a plastic hit's common velocity m·V/(m + M) = 1.37 m/s (+2 %): the guard's crush makes the hit nearly plastic"),
                Criterion::new(Physics, "the car stops", "car.rebound_ratio", Some(-0.1), Some(0.2), "rebound under 20 % of the approach (the car meets a near-wall)"),
                Criterion::new(Physics, "front crush", "car.crush_peak", Some(0.5), Some(1.3), "the frames: hood and front crushed back to the screen base (crash-reference.md S4)"),
                Criterion::new(Invariant, "the crush stays", "car.crush_residual", Some(0.35), None, "metal does not spring back: 70 % of the 0.5 m floor"),
                Criterion::new(Physics, "the upper front does the work", "car.zone.front_upper.work_frac", Some(0.3), None, "the load path starts above the bumper: hood, fenders and cowl take a larger share than in a wall hit (S02's car is the 4x4: compare with the sedan's own wall hit when measured)"),
                Criterion::new(Physics, "upper over lower front work", "car.zone.front_upper_lower_ratio", Some(1.0), None, "proposed metric: (front_upper + 1 J) / (front_lower + 1 J) work"),
                Criterion::new(Shape, "guard held: roof height kept", "car.gauge.roof_height.shrink_frac", None, Some(0.10), "guard-held outcome: roof and A-pillars keep ≥ 90 % of roof height; if the guard fails the greenhouse is sheared (the other valid outcome)"),
                Criterion::new(Shape, "guard held: door openings kept", "car.gauge.door_left.shrink_frac", None, Some(0.10), "the cabin stays out from under the box"),
                Criterion::new(Shape, "guard held: right door opening kept", "car.gauge.door_right.shrink_frac", None, Some(0.10), "the cabin stays out from under the box"),
                Criterion::new(Shape, "underride depth", "pair.car_trailer.underride_m", None, Some(0.6), "proposed metric: how far the car's foremost attached node gets past the plane of the trailer's rear face (in the trailer's frame); guard held: the hood's length at most"),
                Criterion::new(Shape, "the car's skin past the trailer's centre of mass", "pair.car_trailer.skin_reach_max", None, Some(-5.0), "the trailer's centre of mass is about 7.6 m ahead of its rear: a car under the box would reach toward it"),
                Criterion::new(Shape, "guard: beams yielded", "trailer.zone.rear.bent_frac", None, None, "the guard (the trailer's rear zone: its bar and supports) yields as it is built to"),
                Criterion::new(Shape, "guard held: attachments intact", "trailer.zone.rear.broken_frac", None, Some(0.0), "guard held: none of its beams broken; failed: its supports' break groups gone"),
                Criterion::new(Shape, "guard's plastic work", "trailer.zone.rear.work_j", None, None, "what the guard absorbed (FMVSS 223: at least 20 kJ within 125 mm under its distributed load)"),
                Criterion::new(Shape, "guard's peak load", "trailer.zone.rear.force_peak_n", None, None, "proposed metric: the guard's supports' largest total axial force along the trailer's heading; its regulatory strength is 350 kN distributed (its quasi-static collapse measured 400-460 kN, tests/box_trailer.rs)"),
                Criterion::new(Shape, "the car's deceleration", "car.decel_peak_g", None, None, "× the car's mass: the largest force on it (an upper bound on the guard's share)"),
                Criterion::new(Physics, "rear of the car undamaged", "car.zone.rear.bent_frac", None, Some(0.02), "a frontal hit leaves the rear in shape"),
                Criterion::new(Physics, "no rollover", "car.roll_peak_deg", None, Some(20.0), "a centred frontal hit"),
                Criterion::new(Invariant, "collision meshes cross", "pair.car_trailer.crossings_max", None, Some(0.0), "the repo's hard rule (0): reported here as in the suite's table"),
            ]
        },
    }
}

#[cfg(test)]
mod tests {
    /// The bands' numbers follow from the masses and the speed (the generated sedan's mass,
    /// as S03 measures it; the trailer's design gross).
    #[test]
    fn bands_from_masses() {
        let (m, mm, v) = (1448.4f64, 15_000.0f64, 56.0 / 3.6);
        let near = |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol, "{a} vs {b}");
        near(0.5 * m * v * v * 1e-3, 175.2, 0.1);
        near(0.5 * m * (1.02 * v).powi(2) * 1e-3, 182.3, 0.2);
        near(m * v, 22_531.0, 2.0);
        near(m * v * 1.02, 22_982.0, 2.0);
        near(m * v / (m + mm), 1.37, 5e-3);
        near(2.0 * m * v / (m + mm), 2.74, 5e-3);
        near(2.0 * m * v * 1.02 / (m + mm), 2.794, 5e-3);
        near(0.6 * 5_810.0 * 9.81 * 0.8 * 1e-3, 27.4, 0.1);
        near(0.02 * (m + mm) * 9.81 * 1.3 * 1e-3, 4.2, 0.05);
        // the band's ends: 22,982 − 27,400 − 4,200 ≥ −9,500 and 22,982 + 370 ≤ 23,400
        assert!(m * v * 1.02 - 27_400.0 - 4_200.0 >= -9_500.0 && m * v * 1.02 + 370.0 <= 23_400.0);
    }
}
