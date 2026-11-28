//! S11: the US NCAP frontal test: the generated sedan into a full-width rigid wall at
//! 56.3 km/h (35 mph; 49 CFR 571.208 S5.1 configuration, NHTSA NCAP frontal). Reference data:
//! NHTSA tests 8545 (2014 Camry, 1,664 kg test mass, 56.7 km/h) and 7520 (2012 Camry, 1,662 kg,
//! 56.3 km/h), derived into `engine/docs/crash-reference/nhtsa/<test>.json` by
//! `engine/tools/crash-reference/nhtsa_reference.py`, and the 56 km/h sedan fleet of
//! `engine/docs/crash-reference-targets.md` (§1: 82 sedans' static crush; §8: seven sedans'
//! pulses).
//!
//! **Not a Camry.** The generated sedan is 1,448 kg as spawned (`car.mass`), 13 % lighter than
//! the tests' 1,660 kg test mass (curb 1,460 kg plus two dummies and ballast), and a different
//! structure. A Camry's curve is a comparison, never a band: only the laws and the fleet's
//! whole range are asserted.
//!
//! First principles. 15.64 m/s into a wall that does not move: the car stops (its reference
//! location's velocity reaches zero, at t_m, with dx(t_m) the dynamic crush) and springs back
//! by its elastic part; the wall's impulse is m (v0 + v_rebound) less tyre friction; nothing
//! enters the wall; energy only leaves (177 kJ at the start).
//!
//! Measured like the tests (`metrics::series`): the reference location is the cabin floor
//! behind the B-pillar (`metrics::series::reference_nodes`: the rear-seat crossmember's place),
//! sampled every 4 ms; the wall force from the fleet's momentum.

use super::*;

/// 35 mph.
pub const KMH: f64 = 56.3;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S11".into(),
        title: "NHTSA NCAP frontal: sedan into a full-width rigid wall, 56.3 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[], real_test: "NHTSA NCAP frontal rigid barrier, 56.3 km/h (tests 8545, 7520: 2012-14 Toyota Camry)" },
        designs: names(&["sedan"]),
        recreated: false,
        modified_from: None,
        layout: wall_preset(KMH),
        secs_after_impact: 0.4,
        cameras: vec![
            Camera { name: "side".into(), eye: [9.0, 1.2, -1.5], target: [0.0, 0.7, -1.5] },
            Camera { name: "top".into(), eye: [0.01, 12.0, -1.5], target: [0.0, 0.0, -1.5] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            Moment { name: "peak".into(), after_impact_ms: 45.0 },
            Moment { name: "max_crush".into(), after_impact_ms: 72.0 },
            Moment { name: "rest".into(), after_impact_ms: 350.0 },
        ],
        criteria: || {
            use Tier::*;
            let mut c = frontal_laws(177.0);
            // (continuous contact, on the car as defined: one run a joule over the allowance)
            if let Some(i) = c.iter().position(|x| x.name == "no energy created") {
                let k = c.remove(i);
                c.insert(i, k.known_intermittent(&[Contact::Continuous], "new in the 2026-11-27 sweep on the sedan as defined, for review: CPU reference, 1 of 8 runs (2), 178 J against 177 (engine/docs/crash-reference.md, Currently failing)"));
            }
            c.extend([
                Criterion::new(Physics, "restitution within the 56 km/h sedan fleet's", "car.ref.restitution", None, Some(0.25),
                    "−min v from t_m to 200 ms over v0 at the rear-seat crossmember: nine sedans at 56 km/h into the rigid wall span 0.057–0.194 (seven of targets §8 re-checked 2026-11-27: 0.125–0.194; Camry 8545 0.095, 7520 0.057), plus 0.05 for the measurement (7520's two crossmember channels differ by 0.095)")
                    .cited("NHTSA NCAP frontal rigid barrier 56 km/h, sedans 8407, 9065, 9593, 9764, 9793, 10688, 11574 (crash-reference-targets.md §8, re-checked) and 8545, 7520 (engine/docs/crash-reference/nhtsa): rear-seat crossmember X, CFC 60, integrated; rebound = −min v from t_m to 200 ms")
                    .known_intermittent(&[Contact::Discrete, Contact::Edge, Contact::Continuous], "CPU reference 2026-11-27, the sedan as defined (real_crash, the default since 2026-11-27, front seats beside the front door): 3 of 8 runs discrete and edge (0.232-0.260), 4 of 8 continuous (0.234-0.255) against the fleet's 0.057-0.194 + 0.05; it was 8 of 8 at 0.30-0.36 on the beamng_feel car (engine/docs/crash-reference.md, Currently failing)"),
                // Reported: the fleet's ranges, which a different car need not meet.
                Criterion::new(Physics, "dynamic crush (fleet)", "car.ref.dyn_crush", Some(0.617), Some(0.753), "the nine sedans' dx at t_m: 0.617–0.753 m (Camry 0.663, 0.685)"),
                Criterion::new(Physics, "time of maximum crush (fleet)", "car.ref.stop_ms", Some(62.7), Some(76.3), "the nine sedans' t_m: 62.7–76.3 ms (Camry 70.9, 75.9)"),
                Criterion::new(Physics, "mean deceleration to t_m (fleet)", "car.ref.mean_decel_g", Some(21.0), Some(25.4), "v0 / t_m: 21.0–25.4 g (Camry 22.7, 21.0)"),
                Criterion::new(Physics, "peak deceleration, sampled as Rubble samples (fleet)", "car.ref.peak_decel_g", Some(31.9), Some(44.5), "the nine sedans' CFC 60 velocity sampled every 4 ms and differenced as the suite does: 31.9–44.5 g (CFC 60 itself: 35.5–47.3 g; Camry 36.8 and 40.7, CFC 60 39.8 and 45.0)"),
                Criterion::new(Physics, "residual crush (fleet, overall length)", "car.crush_residual", Some(0.311), Some(0.602), "82 sedans' overall length loss p10–p90 311–602 mm (targets §1; Camry 554, 579 mm)"),
                Criterion::new(Shape, "wall force peak (Camry)", "all.box_force_peak_n", Some(830e3), Some(921e3), "Camry 8545 921 kN, 7520 830 kN (load cells, CFC 60); a 1,448 kg car need not match"),
            ]);
            c
        },
    }
}

/// The generated sedan at `kmh` into a full-width rigid wall (4 m tall, 1 m deep, a metre
/// past each side) a tenth of a second ahead of its front (S02's wall).
pub(super) fn wall_preset(kmh: f64) -> LayoutSource {
    family::preset(&families::BLOCK, &[("car", "sedan".into()), ("speed", kmh.into()), ("overlap", 1.0.into()), ("height", 4.0.into()), ("depth", 1.0.into()), ("lead", 0.1.into())])
}

/// The laws of a car into a fixed wall: finite, no energy created (0.1 % of `ke_kj` kJ),
/// nothing into the wall, no rebound past an elastic one.
pub(super) fn frontal_laws(ke_kj: f64) -> Vec<Criterion> {
    use Tier::*;
    vec![
        Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
        Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(ke_kj), "at most 0.1 % of the kinetic energy the car brings (nominal speed)")
            .law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
        Criterion::new(Invariant, "nothing passes into the wall", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside the wall (a node moves under 1 cm a step)").law("a static box is impenetrable"),
        Criterion::new(Physics, "no rebound past an elastic one", "car.rebound_ratio", None, Some(1.0), "the car leaves no faster than it came")
            .law("a fixed wall does no work: the car's kinetic energy after is at most its energy before"),
        Criterion::new(Physics, "the wall stops the car", "car.ref.stop_ms", Some(0.0), None, "the reference location's velocity along the heading reaches 0 after contact (missing when it never does)")
            .law("the wall is impenetrable and does not move: the car cannot keep its forward velocity through it"),
    ]
}
