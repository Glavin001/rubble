//! S5: a heavy truck T-bones a stationary car at 70 km/h (video 3:23 and 3:29: a TV test,
//! a loaded conventional truck into a stationary car's side).
//!
//! The truck is the generated 4x2 cab-over tractor solo, at its kerb (`truck_cab`,
//! M = 7,500 kg; `generated-vehicles-lessons.md` §23-24). Its `ballast_kg` (a load on the
//! fifth wheel) is not used: the suite's templates are each design's defaults, and a
//! registered laden variant (4.5 t of ballast, 12 t) failed the design gate, its frame 13.5 mm
//! out of shape parked (lessons §24). M/m = 5.2 against the generated sedan (`car`,
//! m = 1,448.4 kg). The truck runs square into the car's left (+X) side, its centre line on
//! the middle of the car's wheelbase. The car is raised onto its wheels and settles for
//! [`RUN_UP_S`] before the truck reaches it. Nothing brakes.
//!
//! First principles. Momentum M·V is shared: a plastic hit carries the car off at
//! M·V/(M + m) = 16.30 m/s (0.84 V) while the truck slows by m·V/(M + m) = 3.15 m/s; with
//! restitution e the car leaves at (1 + e)·M·V/(M + m), at most the elastic 2·M·V/(M + m)
//! = 32.6 m/s, and the truck keeps V·(M − e·m)/(M + m) > 0: the heavier striker never
//! reverses (M > m ≥ e·m). The car's side cannot absorb the 230 kJ the plastic hit loses,
//! so its section collapses round the truck's face.
//!
//! Masses and bands: `tests::bands_from_masses`.

use super::*;

pub const AFTER_IMPACT_S: f32 = 1.0;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S05".into(),
        title: "Tractor unit T-bones a stationary sedan, 70 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[203, 209], real_test: "TV test: a loaded truck at about 70-80 km/h into the side of a stationary car" },
        designs: names(&["truck_cab", "sedan"]),
        recreated: false,
        modified_from: None,
        // The truck faces −X (yaw −90°), its bumper a run-up off the car's left side, its centre
        // line on the middle of the car's wheelbase; the car stands.
        layout: family::preset_named(&families::TBONE, &["truck", "car"], &[("striker", "truck_cab".into()), ("struck", "sedan".into()), ("speed", 70.0.into()), ("struck_speed", 0.0.into()), ("aim", 0.0.into()), ("angle", 0.0.into()), ("run_up", 0.4.into())]),
        secs_after_impact: AFTER_IMPACT_S,
        // The truck strikes the car's left side (x ≈ 0.9) at z ≈ −1.4 and carries it along −X.
        cameras: vec![
            Camera { name: "front of car, low".into(), eye: [-1.0, 0.6, 6.0], target: [-1.0, 0.7, -1.4] },
            Camera { name: "aftermath".into(), eye: [-14.0, 3.0, 6.0], target: [-10.0, 0.8, -1.4] },
            Camera { name: "top".into(), eye: [-5.0, 16.0, -1.39], target: [-5.0, 0.0, -1.4] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            Moment { name: "peak".into(), after_impact_ms: 60.0 },
            // the car on the truck's face, carried along
            Moment { name: "carried".into(), after_impact_ms: 250.0 },
            Moment { name: "rest".into(), after_impact_ms: 950.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                // Asserted (laws).
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(1_476.0), "0.1 % of the truck's 1,418 kJ (½·7,500·19.444²), 1,475 kJ at +2 % speed").law("mechanical energy never rises in a crash: contact, plasticity, damping and friction only remove it"),
                Criterion::new(Invariant, "momentum along the truck's travel", "all.momentum_x_end", Some(-148_900.0), Some(0.0),
                    "starts at −M·V = −145,833 kg·m/s (−148,750 at +2 % speed, 0.1 % rounding); ground friction and drag only take it away and cannot reverse it")
                    .law("momentum is conserved between truck and car; the only outside forces (ground friction, air drag) oppose the motion"),
                Criterion::new(Physics, "the car ends moving the truck's way", "car.vx_end", None, Some(0.0), "the truck pushes along −X; friction can stop the car but not reverse it")
                    .law("the only push on the car along X comes from the truck (−X); friction opposes motion"),
                Criterion::new(Physics, "the truck never reverses", "truck.v_along_end", Some(0.0), None, "V·(M − e·m)/(M + m) > 0 for M > m and e ≤ 1; friction only slows it to rest")
                    .law("momentum with restitution e ≤ 1: the heavier striker keeps a forward velocity V·(M − e·m)/(M + m)"),
                Criterion::new(Physics, "the car leaves no faster than an elastic hit", "car.speed_peak", None, Some(33.25), "2·M·V/(M + m) = 32.59 m/s, +2 % for the speed jitter: no collision gives more")
                    .law("an elastic collision transfers the most momentum: v_car ≤ 2·M·V/(M + m)"),
                Criterion::new(Physics, "the struck side absorbs more than the far side", "car.zone.lr_ratio", Some(1.0), None, "left (struck) plastic work over right")
                    .law("the load enters through the struck side; the far side sees only what passes through the cabin"),
                // Reported (estimates and the frames).
                Criterion::new(Physics, "the car is carried at about the plastic speed", "car.speed_peak", Some(14.7), Some(19.8), "plastic M·V/(M + m) = 16.30 m/s (0.84 V) −10 % (tyres and friction during the pulse) to the truck's own 19.83 m/s at +2 %"),
                Criterion::new(Physics, "the truck slows by the matching momentum", "truck.dv_along", Some(-5.5), Some(0.0), "plastic: −m·V/(M + m) = −3.15 m/s; restitution and friction through the car over the second after add"),
                Criterion::new(Physics, "the truck never reverses, at any sample", "truck.v_along_min", Some(0.0), None, "proposed metric: the least COM velocity along its first heading over the run (law once measured)"),
                Criterion::new(Physics, "cabin width at the B-pillar, residual", "car.gauge.cabin_width.shrink_frac", Some(0.35), None, "the frames: the struck side caved in past the car's middle"),
                Criterion::new(Physics, "plan-view bend", "car.bend_plan_m", Some(0.3), None, "proposed metric: how far the car's nose and tail (mean) end ahead of its struck mid-section along the striker's heading; the frames show a V"),
                Criterion::new(Physics, "the cabin carries the work", "car.zone.cabin.work_frac", Some(0.4), None, "the hit is between the axles: cabin cage and doors"),
                Criterion::new(Shape, "the car tilts", "car.roll_peak_deg", Some(10.0), Some(45.0), "the frames: the car tilted as it is carried"),
                Criterion::new(Shape, "the underrun bumper against the sill", "pair.truck_car.bumper_over_sill_m", None, None, "proposed metric: the striker's lowest front-lower node height minus the struck car's lowest cabin node height on the struck side at first contact (+: over the sill)"),
                Criterion::new(Shape, "some part comes off", "car.detached_parts", Some(1.0), None, "the frames: panels and a door separate"),
                Criterion::new(Invariant, "no node outruns the truck much", "car.max_node_speed", None, Some(29.75), "1.5 × the truck's 19.83 m/s (+2 %)"),
                Criterion::new(Invariant, "collision meshes cross", "pair.truck_car.crossings_max", None, Some(0.0), "the repo's hard rule (0): reported here as in the suite's table"),
            ]
        },
    }
}

#[cfg(test)]
mod tests {
    /// The bands' numbers follow from the masses and the speed (the tractor's design kerb;
    /// the generated sedan's mass as S03 measures it).
    #[test]
    fn bands_from_masses() {
        let (mm, m, v) = (7_500.0f64, 1448.4f64, 70.0 / 3.6);
        let near = |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol, "{a} vs {b}");
        near(0.5 * mm * v * v * 1e-3, 1_417.8, 0.5);
        near(0.5 * mm * (1.02 * v).powi(2) * 1e-3, 1_475.1, 0.5);
        near(mm * v, 145_833.0, 1.0);
        near(mm * v * 1.02 * 1.001, 148_899.0, 1.0);
        near(mm * v / (mm + m), 16.30, 5e-3);
        near(m * v / (mm + m), 3.147, 5e-3);
        near(2.0 * mm * v / (mm + m), 32.59, 5e-3);
        near(2.0 * mm * v * 1.02 / (mm + m), 33.25, 5e-3);
        near(0.5 * mm * m / (mm + m) * v * v * 1e-3, 229.5, 0.5);
        near(1.5 * 1.02 * v, 29.75, 5e-3);
    }
}
