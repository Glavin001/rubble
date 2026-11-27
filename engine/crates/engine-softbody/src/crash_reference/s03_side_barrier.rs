//! S3: a moving barrier into the side of a stationary car (video 2:26 and 2:31).
//!
//! The reference frames are an IIHS / Euro NCAP style side test: the barrier runs square to
//! the car (no crab) at about 50 km/h. Ours is the **FMVSS 214 moving deformable barrier**
//! (`crate::crash_rigs`, 1,367.6 kg, its 1,676 × 559 mm aluminium honeycomb face 279 mm up;
//! design `mdb214`), run perpendicular at 50 km/h. (The IIHS barrier is taller and heavier,
//! 1,900 kg; Euro NCAP's AE-MDB 1,400 kg: the FMVSS one is what Rubble has.) It strikes the
//! generated sedan's left (+X, driver's) side, its face's forward edge on FMVSS 214's impact
//! reference line (940 mm ahead of the wheelbase's middle up to a 2,896 mm wheelbase, else
//! 508 mm behind the front axle; S8.10), as the web crash lab's `mdb214` scene places it:
//! the face's middle 0.10 m ahead of the wheelbase's middle, square between the axles (about
//! where Euro NCAP puts the barrier's centre, on the driver's R-point).
//!
//! First principles. A plastic hit: the car leaves sideways at v = M·V/(M + m) and the
//! barrier slows to the same speed (momentum M·V shared); restitution of a crushing side
//! (e ≈ 0.1–0.2) sends the car up to e·M·V/(M + m) faster, and an elastic hit, the most
//! any collision can, at 2·M·V/(M + m). Masses: the barrier M = 1,367.6 kg (49 CFR 587.6),
//! the generated sedan m = 1,448.4 kg (`car.mass`, measured 2026-11-27; the test below
//! re-derives the bands from both), so at V = 13.889 m/s
//! v = 6.745 m/s and the barrier's plastic speed drop m·V/(M + m) = 7.144 m/s. The energy
//! lost is ½·M·m/(M + m)·V² = 67.7 kJ of the barrier's 131.9 kJ, taken by the car's side
//! structure (15–30 cm deep) and the barrier's face.
//!
//! The run. The car is raised onto its wheels (their lowest point 2 cm up) and settles for
//! [`RUN_UP_S`] before the barrier reaches it (it stands still to a few cm/s by then). From
//! first contact its tyres grip sideways, so its sideways speed is judged by its peak COM
//! speed (`car.speed_peak`, reached as the pulse ends, about 0.1 s after contact) and not
//! at the run's end; friction and drag can only take momentum along the barrier's travel
//! away, never add it or reverse it, which bounds `all.momentum_x_end`.
//!
//! Real data. NHTSA's FMVSS 214 dynamic tests (crash-reference-targets.md §10: residual side
//! crush, 14 sedans 126–206 mm) are for the barrier crabbed 27° at 54 km/h; this run is
//! perpendicular at 50 km/h, so none of its bands cite them: they are estimates, reported.

use super::*;
use crate::crash_rigs::MDB_FACE;

const KMH: f32 = 50.0;
/// Simulated seconds after first contact: the pulse (about 0.1 s) and the car shoved clear.
pub const AFTER_IMPACT_S: f32 = 0.6;
/// The barrier's run-up before contact (s).
const RUN_UP_S: f32 = 0.4;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S03".into(),
        title: "Moving barrier into the side of a stationary car, 50 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[146, 151], real_test: "IIHS / Euro NCAP side impact (run with the FMVSS 214 barrier, perpendicular), 50 km/h" },
        designs: names(&["sedan", "mdb214"]),
        recreated: false,
        layout: code(|extent, j| {
            let (e, b) = (extent("sedan"), extent("mdb214"));
            let v = KMH / 3.6 * j.speed;
            let (front, rear) = e.axles.unwrap_or((e.lo.z + 0.2 * e.length(), e.lo.z + 0.8 * e.length()));
            // The spawn puts the lowest node 2 cm up; the car stands on its raycast wheels,
            // so it is raised until their lowest point is (the web lab's spawn rule), and
            // only settles by its suspension's sag before the hit.
            let mid = 0.5 * (front + rear);
            // FMVSS 214 S8.10: the face's forward edge on the impact reference line.
            let irl = if front - rear <= 2.896 { mid + 0.94 } else { front - 0.508 };
            let face_z = irl - 0.5 * MDB_FACE[0] + j.lateral;
            // The barrier faces −X (yaw −90°): its bumper's front, `b.hi.z` ahead of its
            // origin, a run-up short of the car's left side.
            let x = e.hi.x + RUN_UP_S * v + b.hi.z;
            Layout { box_yaw: Vec::new(),
                bodies: vec![
                    Body::new("car", "sedan", Vec3::ZERO, 0.0, 0.0),
                    Body::new("barrier", "mdb214", Vec3::new(x, 0.0, face_z), -0.5 * std::f32::consts::PI + j.yaw, v),
                ],
                boxes: Vec::new(),
                impact_s: RUN_UP_S,
            }
        }),
        secs_after_impact: AFTER_IMPACT_S,
        cameras: vec![
            // Ahead of the car and out on the struck side, looking back at the doors.
            // (World frame: the sedan spans z −3.9 to 1.0, the face's middle at z ≈ −1.3.)
            // Closer than the first cut, as the reference frame: the doors and the barrier's face fill it.
            Camera { name: "front three-quarter".into(), eye: [4.0, 1.5, 2.6], target: [0.2, 0.7, -1.2] },
            // Straight down, the car's length across the frame (the eye a centimetre to −x fixes the
            // picture's up as +x: the barrier comes from the top); wide enough for the car's slide
            // (2.5 to 4 m to −x by the run's end, more with discrete contact).
            Camera { name: "top".into(), eye: [-1.41, 9.0, -1.4], target: [-1.4, 0.7, -1.4] },
            // On the car's axis ahead of it: the side hit in elevation.
            Camera { name: "side".into(), eye: [-0.5, 1.3, 5.0], target: [-0.5, 0.7, -0.5] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            // Deepest intrusion, about where the car's sideways speed peaks.
            Moment { name: "peak".into(), after_impact_ms: 70.0 },
            // The run's end: the car is still sliding (about 4 m/s), the barrier behind it.
            Moment { name: "rest".into(), after_impact_ms: 550.0 },
        ],
        // Worked from M = 1,367.6 kg, m = 1,448.4 kg, V = 13.889 m/s (jitter: speed ±2 %).
        criteria: || {
            use Tier::*;
            vec![
                // Asserted (laws).
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes")
                    .law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(132.0), "0.1 % of the barrier's 131.9 kJ (½·1,367.6·13.889²), rounding only")
                    .law("mechanical energy never rises in a crash: contact, plasticity, damping and friction only remove it")
                    .known_failure(&[Contact::Edge, Contact::Continuous], "CPU reference 2026-11-27, 8 of 8 runs: edge 14.1-211 kJ, continuous 6.7-29.7 kJ (discrete holds); edge-contact hits between barrier and door (engine/docs/crash-reference.md, Currently failing)"),
                Criterion::new(Invariant, "momentum along the barrier's travel", "all.momentum_x_end", Some(-19_400.0), Some(0.0),
                    "starts at −M·V = −18,994 N·s (−19,374 at +2 % speed, 0.1 % rounding); ground friction and drag only take it away and cannot reverse it")
                    .law("momentum is conserved between car and barrier; the only outside forces (ground friction, air drag) oppose the motion"),
                Criterion::new(Physics, "the car leaves no faster than an elastic hit", "car.speed_peak", None, Some(13.76),
                    "2·M·V/(M + m) = 13.49 m/s, +2 % for the speed jitter: no collision gives more")
                    .law("an elastic collision transfers the most momentum: v_car ≤ 2·M·V/(M + m)"),
                Criterion::new(Physics, "the car ends moving the barrier's way", "car.vx_end", None, Some(0.0),
                    "the barrier pushes along −X; friction can stop the car but not reverse it")
                    .law("the only push on the car along X comes from the barrier (−X); friction opposes motion"),
                Criterion::new(Physics, "the barrier slows, never speeds up or rebounds past elastic", "barrier.dv_along", Some(-14.57), Some(0.0),
                    "at most 2·m·V/(M + m) = 14.29 m/s (+2 % jitter) lost in an elastic hit, none gained")
                    .law("momentum: the car can only take the barrier's momentum, at most what an elastic hit takes"),
                Criterion::new(Physics, "the struck side absorbs more than the far side", "car.zone.lr_ratio", Some(1.0), None,
                    "left (struck) plastic work over right")
                    .law("the load enters through the struck side; the far side sees only what passes through the cabin"),
                // Reported (estimates).
                Criterion::new(Invariant, "no node outruns the barrier much", "car.max_node_speed", None, Some(21.25),
                    "1.5 × the barrier's 14.17 m/s (+2 %): a node driven by the barrier's face moves about its speed"),
                Criterion::new(Physics, "car's sideways speed after contact", "car.speed_peak", Some(5.06), Some(8.43),
                    "plastic hit M·V/(M + m) = 6.745 m/s ± 25 % (restitution e ≈ 0.1–0.2 raises it, tyre friction during the 0.1 s pulse lowers it)"),
                Criterion::new(Physics, "barrier slowed by the matching momentum", "barrier.dv_along", Some(-14.17), Some(-5.36),
                    "loses at least 0.75 × m·V/(M + m) = 0.75 × 7.144 m/s, and does not reverse (V = 14.17 m/s at +2 %); friction through the car adds to the loss by the run's end"),
                Criterion::new(Physics, "cabin width at the B-pillar, residual", "car.gauge.cabin_width.change_residual", Some(-0.45), Some(-0.10),
                    "a side has 15–30 cm of structure, so intrusion is large (no matching data: FMVSS 214's 126–206 mm crush is for a crabbed 54 km/h test)"),
                Criterion::new(Physics, "struck side takes the damage", "car.zone.lr_ratio", Some(3.0), None, "left (struck) work at least 3 × the right's"),
                Criterion::new(Physics, "the cabin and doors take it", "car.zone.cabin.work_frac", Some(0.6), None, "the barrier strikes between the axles: cabin cage and doors"),
                Criterion::new(Physics, "nose unharmed", "car.zone.front.bent_frac", None, Some(0.05), "the face spans the doors, not the front"),
                Criterion::new(Physics, "tail unharmed", "car.zone.rear.bent_frac", None, Some(0.05), "the face spans the doors, not the rear"),
                Criterion::new(Physics, "no rollover", "car.roll_peak_deg", None, Some(25.0), "the face's middle (0.56 m up) is near the car's centre of mass (0.51 m): the push shoves more than it tips; only the tyres' grip below tips it"),
                Criterion::new(Physics, "barrier face stops short of the car's centreline", "pair.barrier_car.reach_max", None, Some(0.0),
                    "a side has 15–30 cm of structure and the cabin's width behind it: no barrier node gets past the car's centre of mass"),
                Criterion::new(Shape, "little yaw (face centred between the axles)", "car.yaw_abs_peak_deg", None, Some(10.0), "the face's middle 0.10 m ahead of the wheelbase's middle"),
                Criterion::new(Shape, "roof rail bowed a little", "car.gauge.roof_height.change_residual", Some(-0.10), Some(-0.002), "the frames show the struck roof rail bowed in; the roof itself stays up"),
            ]
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::crash_rigs::MDB_MASS;

    /// The generated sedan's mass (kg), as `car.mass` reports it (2026-11-27).
    const CAR_MASS: f64 = 1448.4;

    /// The bands' numbers follow from the masses and the speed.
    #[test]
    fn bands_from_masses() {
        let (bm, m, v) = (f64::from(MDB_MASS), CAR_MASS, 50.0 / 3.6);
        let near = |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol, "{a} vs {b}");
        near(bm * v / (bm + m), 6.745, 5e-3);
        near(0.75 * bm * v / (bm + m), 5.06, 5e-3);
        near(1.25 * bm * v / (bm + m), 8.43, 5e-3);
        near(2.0 * bm * v / (bm + m) * 1.02, 13.76, 5e-3);
        near(m * v / (bm + m), 7.144, 5e-3);
        near(0.75 * m * v / (bm + m), 5.36, 5e-3);
        near(2.0 * m * v / (bm + m) * 1.02, 14.57, 5e-3);
        near(bm * v * 1.02 * 1.001, 19_393.0, 1.0);
        near(0.5 * bm * v * v * 1e-3, 131.9, 0.1);
        near(1.5 * 1.02 * v, 21.25, 5e-3);
    }
}
