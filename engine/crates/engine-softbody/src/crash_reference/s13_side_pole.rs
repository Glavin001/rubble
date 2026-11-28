//! S13: the NCAP / FMVSS 214 oblique side pole test: the generated sedan carried sideways at
//! 32 km/h, its travel 75° from its forward axis (toward its left, a little forward: the
//! database's crab angle 285°), into a rigid 254 mm pole aligned with the driver's head (49 CFR
//! 571.214 S9; NCAP side pole). Reference data: NHTSA tests 7523 (2012 Camry, 1,566 kg test
//! mass, 32.1 km/h) and 8558 (2014 Camry, 1,579 kg, 31.95 km/h),
//! `engine/docs/crash-reference/nhtsa/<test>.json`; and the 116 sedans of
//! `crash-reference-targets.md` §1 (residual crush at the pole).
//!
//! Set up as the web crash lab's `sidepole` scene (`engine-world-wasm` `explicit.rs`): the car
//! on a carrier ([`crate::crash_rigs::on_carrier`], its tyres on low-friction pads as in the
//! real test, so its own tyres do not stop it before the pole), its side's face square to one
//! of the pole's faces. **Choices:** the suite's static boxes are axis-aligned, so the pole is
//! a square post 254 mm wide (the lab's is 200 mm; the test's is a 254 mm cylinder): a face
//! parallel to the car's side meets it along a 254 mm vertical strip, as the cylinder's
//! tangent does at first contact; the cylinder's wrapping as the side crushes round it is not
//! modelled.
//!
//! **Where the pole strikes.** The procedure aligns the pole with the seated dummy's head
//! centre of gravity along the travel; both reports give where that put it on the car: the
//! vertical impact reference line on the struck side 1,101 mm (7523) and 1,125 mm (8558; 1,128
//! intended) behind the front axle. On a Camry that is the front door, about 0.37 m ahead of the
//! B-pillar and 0.06-0.09 m ahead of the car's centre of gravity. The generated sedan is laid
//! out differently (rear-wheel drive: its centre of mass is 1.393 m behind its front axle, 0.49
//! of its wheelbase, against the Camrys' 1.16-1.19 m, 0.42; its B-pillar 1.635 m, its front
//! door 0.57-1.60 m), so one line cannot match on the axle, the door and the centre of mass at
//! once. The scenario keeps what decides the crash: **the impact line 0.075 m ahead of the
//! centre of mass** (the pole's lever arm, so the car's yaw) **and on the front door**, 0.32 m
//! ahead of the B-pillar: 1.318 m behind the front axle. The aim, on the travel line through
//! that line, is 0.38 m left of the centre line and 1.469 m behind the front axle
//! (1.318 + (0.943 - 0.38) / tan 75 deg; the side's outermost plane is 0.943 m out). A change
//! of the sedan's mass distribution moves this number: it was 1.355 m (aim 1.506 m) while the
//! front seats stood behind the B-pillar and the centre of mass was 1.430 m back (the seats
//! moved beside the front door on 2026-11-27, the centre of mass 37 mm forward with them).
//!
//! Measured 2026-11-27 on the car before the seats moved (CPU reference, run 0, edge contact,
//! the `beamng_feel` strength), the impact line at three places:
//! 1.996 m behind the front axle (the aim until then, read from the generated sedan's driver
//! seat, which the generator stands behind the B-pillar, beside the rear door: seat 1.64-2.25
//! m): the pole behind the centre of mass, the car pivots 38.7 deg nose-left by 300 ms, 307 kN.
//! 1.113 m (the tests' line by the axle): 0.31 m ahead of the centre of mass, 19.9 deg
//! nose-right, 342 kN. 1.355 m (0.075 m ahead of that car's centre of mass): 3.6 deg nose-right, 275 kN. The tests: 8-10 deg
//! nose-left (two accelerometers, good to 30 %), 191-197 kN.
//!
//! First principles: 8.89 m/s, 57 kJ; the pole only pushes (the car leaves along the travel
//! no faster than it came, and the carrier only slows it); the struck (left) side takes the
//! work; nothing enters the pole.

use super::*;

pub const KMH: f32 = 32.0;
/// The travel's angle from the car's forward axis (deg).
pub const ANGLE_DEG: f32 = 75.0;
// The pole: 254 mm wide. The aim: 0.38 m left of the centre line, 1.469 m behind the front
// axle, which puts the impact line on the side 1.318 m behind it (see the module notes).

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S13".into(),
        title: "NCAP side pole: sedan carried into a 254 mm pole at 75°, 32 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[], real_test: "NHTSA NCAP / FMVSS 214 oblique pole, 32 km/h, 75°, 254 mm (tests 7523, 8558: 2012-14 Toyota Camry)" },
        designs: names(&["sedan"]),
        recreated: false,
        modified_from: None,
        // Travel 75° from the forward axis toward the car's left, the pole's centre on the
        // travel line through the driver's head, a tenth of a second ahead.
        layout: family::preset(&families::SIDE_POLE, &[("car", "sedan".into()), ("speed", f64::from(KMH).into()), ("angle", f64::from(ANGLE_DEG).into()), ("pole_width", 0.254.into()), ("aim_left", 0.38.into()), ("aim_behind_axle", 1.469.into()), ("lead", 0.1.into())]),
        secs_after_impact: 0.3,
        // World coordinates at contact: the car's origin at 0, the pole at about (1.07, −1.32).
        cameras: vec![
            Camera { name: "front three-quarter".into(), eye: [4.5, 1.6, 2.6], target: [0.9, 0.7, -1.3] },
            Camera { name: "top".into(), eye: [0.01, 12.0, -1.4], target: [0.0, 0.0, -1.4] },
            Camera { name: "behind the pole".into(), eye: [5.5, 1.2, -1.32], target: [0.9, 0.8, -1.32] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            Moment { name: "peak".into(), after_impact_ms: 40.0 },
            Moment { name: "deepest".into(), after_impact_ms: 100.0 },
            Moment { name: "rest".into(), after_impact_ms: 280.0 },
        ],
        // 8.89 m/s nominal, +2 % jitter: 9.07 m/s.
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(57.0), "0.1 % of the 57.2 kJ the 1,448 kg sedan brings at 8.89 m/s")
                    .law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule")
                    .known_intermittent(&[Contact::Continuous], "new in the 2026-11-27 sweep on the sedan as defined, for review: CPU reference, 2 of 8 runs (0, 4), 67-74 J against 57 (edge 9-38 J, holds) (engine/docs/crash-reference.md, Currently failing)"),
                Criterion::new(Invariant, "nothing passes into the pole", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside the pole (a node moves 4.5 mm a step)").law("a static box is impenetrable"),
                Criterion::new(Physics, "the pole throws it back no faster than it came", "car.v_travel_end", Some(-9.07), None, "the COM's velocity along the travel at the end, at least −v0 (+2 % jitter)")
                    .law("a fixed pole does no work: the car's kinetic energy after is at most its energy before"),
                Criterion::new(Physics, "the pole and the carrier only slow it", "car.v_travel_end", None, Some(9.07), "at most v0 along the travel (+2 % jitter)")
                    .law("the pole only pushes back along the travel and the carrier's friction opposes the motion; gravity has no horizontal part"),
                Criterion::new(Physics, "the struck side absorbs more than the far side", "car.zone.lr_ratio", Some(1.0), None, "left (struck) plastic work over right")
                    .law("the load enters through the struck side; the far side sees only what passes through the cabin"),
                // Reported.
                Criterion::new(Physics, "residual crush at the pole (fleet)", "car.side_crush_left_residual", Some(0.169), Some(0.537),
                    "116 sedans at 32 km/h, 75°, 254 mm: min–max 169–537 mm, p10–p90 297–445 (targets §1; Camry 394, 379 mm). Not asserted: NHTSA's maximumCrushDistance is read on the exterior at the levels of its crush table, ours is the door band's deepest skin node (metrics::SIDE_BAND), and the post is square"),
                Criterion::new(Shape, "pole force peak (Camry)", "all.box_force_peak_n", Some(191e3), Some(197e3), "7523 191 kN, 8558 197 kN (8 load cells, CFC 60)"),
                Criterion::new(Shape, "lateral peak acceleration at the CG, sampled as Rubble samples (Camry)", "car.com.ay_peak_g", Some(13.15), Some(13.25), "7523 13.15 g, 8558 13.25 g: their CG Y velocity (CFC 60) every 4 ms, differenced as the suite does (CFC 60 itself: 16.5, 17.9 g; ours on the CPU: car.com.cfc60_ay_peak_g)"),
                Criterion::new(Shape, "lateral velocity change by 150 ms (Camry)", "car.com.dvy_150ms", Some(10.26), Some(10.82), "7523 10.26 m/s, 8558 10.82 m/s (CG Y, +right)"),
                Criterion::new(Shape, "no rollover", "car.roll_peak_deg", None, Some(25.0), "the pole strikes at the door band, near the centre of mass's height"),
            ]
        },
    }
}
