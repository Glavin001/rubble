//! S6: a car crushed between a moving truck and a parked semi-trailer (video 4:06: DEKRA,
//! a truck closing on a queue).
//!
//! The parked trailer is the generated 53 ft dry van (`box_trailer`, 15,000 kg laden, its
//! FMVSS 223 guard); the car the generated sedan (`car`, 1,448.4 kg), stationary with its
//! nose [`GAP`] behind the trailer's rear; the truck the tractor solo at its kerb
//! (`truck_cab`, 7,500 kg: see S05 for why not ballasted) driving into the car's rear at
//! 45 km/h. The speed: the reference shows a truck at moderate speed closing a queue, read as
//! 40-50 km/h (crash-reference.md §1, an estimate from the footage), so the middle of that
//! range. All three heading +Z, centred. Nothing brakes (the scenario sets no inputs): the
//! car and the trailer's tandem roll free; a parked trailer's spring brakes are not modelled.
//!
//! First principles. Both partners are five to ten times the car and far stiffer: the car is
//! the only thing that can shorten, so its final length is set by the gap the truck leaves,
//! not by its strength. The truck's momentum M·V = 93,750 kg·m/s goes to the three bodies
//! less the ground's friction; the car cannot leave the space between the two except by
//! passing through (or under, or over) one of them.
//!
//! Masses and bands: `tests::bands_from_masses`.

use super::*;

const KMH: f32 = 45.0;
/// The car's nose to the trailer's rear (m): a queue's gap (estimate).
pub const GAP: f32 = 1.0;
/// The truck's run-up (s): the car and the trailer settle on their wheels first.
const RUN_UP_S: f32 = 0.5;
pub const AFTER_IMPACT_S: f32 = 1.4;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S06",
        title: "Sedan crushed between a laden tractor at 45 km/h and a parked semi-trailer",
        reference: Reference { video: VIDEO, at_s: &[246, 248, 255], real_test: "DEKRA: a truck into a queue, a car between it and a parked truck" },
        designs: &["truck_cab", "sedan", "box_trailer"],
        layout: |extent, j| {
            let (c, t, tr) = (extent("sedan"), extent("truck_cab"), extent("box_trailer"));
            let v = KMH / 3.6 * j.speed;
            // the trailer's rear GAP ahead of the car's nose
            let z_trailer = c.hi.z + GAP - tr.lo.z;
            // the truck's front a run-up behind the car's tail
            let z_truck = c.lo.z - RUN_UP_S * v - t.hi.z;
            Layout {
                bodies: vec![
                    Body::new("truck", "truck_cab", Vec3::new(j.lateral, 0.0, z_truck), j.yaw, v),
                    Body::new("car", "sedan", Vec3::ZERO, 0.0, 0.0),
                    Body::new("trailer", "box_trailer", Vec3::new(0.0, 0.0, z_trailer), 0.0, 0.0),
                ],
                boxes: vec![],
                impact_s: RUN_UP_S,
            }
        },
        secs_after_impact: AFTER_IMPACT_S,
        // The car spans z −3.9 to 1.0; the trailer's rear at z ≈ 2.0.
        cameras: &[
            Camera { name: "side", eye: [-11.0, 1.6, -1.0], target: [0.0, 1.0, -1.0] },
            Camera { name: "aftermath", eye: [-6.0, 2.6, -6.0], target: [0.0, 0.7, -0.5] },
            Camera { name: "top", eye: [0.01, 15.0, -1.0], target: [0.0, 0.0, -1.0] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "first_contact", after_impact_ms: 20.0 },
            // the car's nose reaches the trailer's guard
            Moment { name: "nose_under_trailer", after_impact_ms: 150.0 },
            Moment { name: "peak", after_impact_ms: 400.0 },
            Moment { name: "aftermath", after_impact_ms: 1_350.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                // Asserted (laws).
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(610.0), "0.1 % of the truck's 585.9 kJ (½·7,500·12.5²), 609.6 kJ at +2 % speed").law("mechanical energy never rises in a crash: contact, plasticity, damping and friction only remove it"),
                Criterion::new(Invariant, "momentum along the road", "all.momentum_z_end", Some(18_000.0), Some(105_200.0),
                    "starts at M·V = 93,750 kg·m/s (91,875-95,625 with the ±2 % speed). From outside only the ground acts: the trailer's landing-gear feet, sliding once pushed, at most μ·N = 0.6 × 57.0 kN over the 1.5 s after contact (51.3 kN·s); the crushed car's body, if all of it slid on the ground, 0.6 × 14.2 kN × 1.5 s = 12.8 kN·s; and every tyre's rolling resistance either way, at most 0.02 × 235 kN × 2.0 s = 9.4 kN·s")
                    .law("momentum changes only by the external impulse, the ground's friction and rolling resistance, bounded by μ·N and c_rr·W over the run"),
                Criterion::new(Invariant, "the car stays ahead of the truck", "pair.truck_car.com_order_min", Some(0.0), None,
                    "proposed metric: the least (car's COM − truck's COM) along the truck's first heading over the run")
                    .law("no pass-through: the car's mass cannot get behind the truck's centre of mass except through, under or over the truck"),
                Criterion::new(Invariant, "the car stays behind the trailer", "pair.car_trailer.com_order_min", Some(0.0), None,
                    "proposed metric: the least (trailer's COM − car's COM) along the car's first heading over the run")
                    .law("no pass-through: the car's mass cannot get past the trailer's centre of mass except through, under or over the trailer"),
                // Reported (estimates and the frames).
                Criterion::new(Physics, "the car ends about half its length", "car.gauge.length.shrink_frac", Some(0.4), None, "the frames: the aftermath car about half length (residual ≤ 60 %)"),
                Criterion::new(Physics, "roof flattened", "car.gauge.roof_height.shrink_frac", Some(0.4), None, "the frames: roof flattened to the beltline (≤ 60 % of the roof height)"),
                Criterion::new(Physics, "most of the car yields", "car.bent_frac", Some(0.5), None, "everything folded"),
                Criterion::new(Physics, "the front yields", "car.zone.front.bent_frac", Some(0.01), None, "every zone has yielded beams"),
                Criterion::new(Physics, "the cabin yields", "car.zone.cabin.bent_frac", Some(0.01), None, "every zone has yielded beams"),
                Criterion::new(Physics, "the rear yields", "car.zone.rear.bent_frac", Some(0.01), None, "every zone has yielded beams"),
                Criterion::new(Physics, "the trailer is pushed", "trailer.travel_along", None, None, "proposed metric: (COM at the end − at the start) along its first heading (m)"),
                Criterion::new(Physics, "the trailer's speed", "trailer.speed_peak", None, Some(7.0), "the three-body plastic M·V/(M + m + M_trailer) = 3.9 m/s; restitution raises it (estimate: up to about 1.8 ×)"),
                Criterion::new(Shape, "nose under the trailer's deck", "pair.car_trailer.underride_m", Some(0.0), None, "proposed metric (see S04): the frames show the nose under the deck"),
                Criterion::new(Shape, "guard: beams yielded", "trailer.zone.rear.bent_frac", None, None, "the trailer's guard"),
                Criterion::new(Shape, "guard: attachments broken", "trailer.zone.rear.broken_frac", None, None, "the trailer's guard: held (0) or failed"),
                Criterion::new(Invariant, "no explosion", "car.max_node_speed", None, Some(19.13), "1.5 × the truck's 12.75 m/s (+2 %)"),
                Criterion::new(Invariant, "collision meshes cross: truck and car", "pair.truck_car.crossings_max", None, Some(0.0), "the repo's hard rule (0): reported here as in the suite's table"),
                Criterion::new(Invariant, "collision meshes cross: car and trailer", "pair.car_trailer.crossings_max", None, Some(0.0), "the repo's hard rule (0): reported here as in the suite's table"),
                Criterion::new(Invariant, "collision meshes cross: truck and trailer", "pair.truck_trailer.crossings_max", None, Some(0.0), "the repo's hard rule (0): reported here as in the suite's table"),
            ]
        },
    }
}

#[cfg(test)]
mod tests {
    /// The bands' numbers follow from the masses and the speed (the tractor's kerb, the
    /// trailer's design gross; the generated sedan's mass as S03 measures it).
    #[test]
    fn bands_from_masses() {
        let (mm, m, mt, v) = (7_500.0f64, 1448.4f64, 15_000.0f64, 45.0 / 3.6);
        let near = |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol, "{a} vs {b}");
        near(0.5 * mm * v * v * 1e-3, 585.9, 0.1);
        near(0.5 * mm * (1.02 * v).powi(2) * 1e-3, 609.6, 0.1);
        near(mm * v, 93_750.0, 1.0);
        near(0.6 * 5_810.0 * 9.81 * 1.5 * 1e-3, 51.3, 0.1);
        near(0.02 * (mm + m + mt) * 9.81 * 2.0 * 1e-3, 9.4, 0.05);
        near(0.6 * m * 9.81 * 1.5 * 1e-3, 12.8, 0.05);
        // the band's ends
        assert!(mm * v * 0.98 - 51_300.0 - 12_800.0 - 9_400.0 >= 18_000.0);
        assert!(mm * v * 1.02 * 1.001 + 9_400.0 <= 105_200.0);
        near(mm * v / (mm + m + mt), 3.915, 5e-3);
        near(1.5 * 1.02 * v, 19.13, 5e-3);
    }
}
