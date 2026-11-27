//! S9: a car and a heavier, taller SUV head-on, each at 55 km/h, 50 % overlap on their
//! drivers' (left) sides (video 6:15): the generated `wagon` (body `car`) and the generated
//! 4x4 (`suv_4x4`, body `suv`; its mass built up part by part,
//! `engine/docs/generated-vehicles-lessons.md` §21).
//!
//! Reference frames: the SUV rides over the car's hood; the car's front is crushed back to the
//! screen and its rear is thrown up; the SUV's front is damaged much less; the pair moves on
//! in the SUV's direction.
//!
//! First principles. The masses as spawned (wheels on their hubs) are m = 1,448 kg (car) and
//! M = 1,974 kg (SUV), M/m = 1.36, both at v = 15.28 m/s toward each other (399.5 kJ). The
//! car heads +Z, the SUV −Z. The forces between them are equal and opposite, so while they are
//! in contact m·Δv_car = −M·Δv_suv: the car's change of velocity is M/m times the SUV's,
//! whatever the restitution. With restitution e (0 ≤ e ≤ 1: no energy created) the car leaves
//! at v(m − M − 2eM)/(M + m) along its heading, which is below zero for every e since m < M:
//! the car is driven back in the SUV's direction, and Coulomb friction, which only brings a
//! body toward rest, never turns it forward again.
//!
//! The only external horizontal force is the ground's friction, at most μ·N with μ ≤ 1.2 (the
//! largest coefficient in the scene: the tyres' peak and the concrete's static; the ground's
//! is 0.85). Over the run the ground's normal impulse is the pair's weight × time (their
//! vertical momentum is zero at both ends), so the friction impulse is at most
//! 1.2 × 9.81 m/s² × 3,422 kg × 1.1 s = 44.3 kN·s: the bound on the change of momentum along
//! and across the road (the start's momentum along it is (M − m)·v = 8.04 kN·s toward −Z).

use super::*;

const KMH: f32 = 55.0;
/// The share of the car's width that meets the SUV's.
const OVERLAP: f32 = 0.5;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S09",
        title: "Car vs heavier SUV head-on, 2 × 55 km/h, 50 % overlap",
        reference: Reference { video: VIDEO, at_s: &[375], real_test: "car-to-SUV frontal offset, 50 % overlap, each at 55 km/h" },
        designs: &["wagon", "suv_4x4"],
        layout: |extent, j| {
            let (ec, es) = (extent("wagon"), extent("suv_4x4"));
            let v = KMH / 3.6 * j.speed;
            // A tenth of a second of closing (both move) before the fronts meet.
            let gap = 0.1 * 2.0 * v;
            // The SUV faces the car (yaw π: its left is world −X). Its world x span is
            // [xs − es.hi.x, xs − es.lo.x]; its left edge stands OVERLAP of the car's width in
            // from the car's left (+X) edge, so their left sides overlap.
            let xs = ec.hi.x - OVERLAP * ec.width() + es.hi.x;
            let zs = ec.hi.z + es.hi.z + gap;
            Layout {
                bodies: vec![
                    Body::new("car", "wagon", Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v),
                    Body::new("suv", "suv_4x4", Vec3::new(xs, 0.0, zs), std::f32::consts::PI, v),
                ],
                boxes: vec![],
                impact_s: 0.1,
            }
        },
        secs_after_impact: 1.0,
        // The fronts meet at z ≈ 2.2 (the wagon's front is 0.96 m ahead of its origin), x ≈ 0.5.
        cameras: &[
            Camera { name: "side", eye: [11.0, 1.4, 2.2], target: [0.5, 0.8, 2.2] },
            Camera { name: "top", eye: [0.5, 14.0, 2.21], target: [0.5, 0.0, 2.2] },
            Camera { name: "quarter", eye: [-6.0, 3.0, -3.0], target: [0.5, 0.7, 2.2] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "override", after_impact_ms: 60.0 },
            Moment { name: "car_rear_up", after_impact_ms: 120.0 },
            Moment { name: "rest", after_impact_ms: 950.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                // -- asserted: laws
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(400.0), "energy is never created: at most 0.1 % of the 399.5 kJ the 1,448 kg car and the 1,974 kg SUV bring at 15.28 m/s").law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
                Criterion::new(Invariant, "momentum along the road", "all.momentum_z_end", Some(-52_400.0), Some(36_300.0), "the start's −8.04 kN·s ((M − m)·v toward the SUV's heading), changed by at most the ground's friction impulse, 44.3 kN·s").law("momentum is conserved between the bodies; the only external horizontal impulse is the ground's friction, at most μ (1.2) × the weight's impulse (3,422 kg × 9.81 m/s² × 1.1 s)"),
                Criterion::new(Invariant, "momentum across the road", "all.momentum_x_end", Some(-44_600.0), Some(44_600.0), "zero at the start up to the car's 0.5° jitter (190 kg·m/s), changed by at most the ground's friction impulse, 44.3 kN·s").law("momentum is conserved (as along the road)"),
                Criterion::new(Invariant, "the car is driven back", "car.v_along_end", None, Some(0.0), "the car leaves at v(m − M − 2eM)/(M + m) < 0 along its heading for every restitution 0 ≤ e ≤ 1").law("momentum and no energy created (0 ≤ e ≤ 1) put the car's velocity after the hit in the heavier SUV's direction (m < M); Coulomb friction brings a body toward rest and never reverses it"),
                Criterion::new(Physics, "the car's overlapping side takes more", "car.zone.front_lr_ratio", Some(1.0), None, "only the left half of each front meets the other vehicle").law("the right front is loaded only through the structure from the left, which passes on less than it receives: left work > right work"),
                Criterion::new(Physics, "the SUV's overlapping side takes more", "suv.zone.front_lr_ratio", Some(1.0), None, "as the car's").law("as the car's"),
                // -- reported: the velocity split's size (a ratio metric is not yet measured:
                // `pair.car_suv.dv_ratio` = |car.dv_along| / |suv.dv_along|; exactly M/m = 1.36
                // between two free bodies, so its direction ≥ 1 becomes a law once measured)
                Criterion::new(Physics, "the lighter car's velocity changes more", "pair.car_suv.dv_ratio", Some(1.02), Some(1.70), "M/m = 1.36 ± 25 % (momentum; the ground's friction moves it)"),
                Criterion::new(Physics, "car's velocity change", "car.dv_along", Some(-30.6), Some(-17.6), "−2Mv/(M + m) (e = 0, 17.6 m/s) to −4Mv/(M + m) (e = 1), the ground's friction aside"),
                Criterion::new(Physics, "SUV's velocity change", "suv.dv_along", Some(-25.9), Some(-12.9), "−2mv/(M + m) (e = 0, 12.9 m/s) to −4mv/(M + m) (e = 1), the ground's friction aside"),
                Criterion::new(Physics, "the pair moves on in the SUV's direction", "all.momentum_z_end", None, Some(0.0), "(M − m)·v = 8.04 kN·s toward −Z at the start; a law only while the SUV keeps moving forward after the hit (e < (M − m)/2m = 0.18), so reported"),
                // -- reported: estimates from the frames and the mass ratio
                Criterion::new(Physics, "car absorbs more", "pair.car_suv.plastic_work_ratio", Some(1.3), None, "the car's softer, lower front takes the SUV's rails above its own (the frames); car / SUV"),
                Criterion::new(Physics, "car crushes more", "pair.car_suv.crush_peak_ratio", Some(1.0), None, "the car's front crushed back to the screen, the SUV's much less (the frames); car / SUV"),
                Criterion::new(Physics, "car's overlapping side's crush", "car.gauge.length_left.change_peak", Some(0.4), Some(1.2), "crushed back to the screen: more than S08's 0.3-0.95 m at the same speed (the struck side's length: crush_peak is the whole car's, which the unstruck corner holds)"),
                Criterion::new(Physics, "SUV's overlapping side's crush", "suv.gauge.length_left.change_peak", Some(0.1), Some(0.7), "damaged much less than the car"),
                Criterion::new(Physics, "car's crush (whole length)", "car.crush_peak", None, None, "reported: the unstruck corner holds the length"),
                Criterion::new(Physics, "SUV's crush (whole length)", "suv.crush_peak", None, None, "reported"),
                Criterion::new(Physics, "car's plastic work", "car.plastic_work_j", None, None, "reported"),
                Criterion::new(Physics, "SUV's plastic work", "suv.plastic_work_j", None, None, "reported"),
                Criterion::new(Physics, "SUV's struck door opening", "suv.gauge.door_left.shrink_frac", None, Some(0.05), "the heavier vehicle's cabin stays whole"),
                Criterion::new(Physics, "car's struck door opening", "car.gauge.door_left.shrink_frac", None, Some(0.10), "as S08's"),
                Criterion::new(Physics, "car's cabin carries little", "car.zone.cabin.work_frac", None, Some(0.25), "more than S08's 0.15: the crush reaches the screen"),
                Criterion::new(Physics, "the cabins stay apart", "pair.car_suv.com_gap_min", Some(2.5), None, "as S08's: cabins that met would put the centres of mass under about 2.6 m"),
                Criterion::new(Invariant, "nothing passes through", "pair.car_suv.crossings_max", None, Some(0.0), "the collision meshes never cross"),
                Criterion::new(Invariant, "no explosion (car)", "car.max_node_speed", None, Some(46.0), "3 × the 15.3 m/s approach (see S01)"),
                Criterion::new(Invariant, "no explosion (SUV)", "suv.max_node_speed", None, Some(46.0), "3 × the 15.3 m/s approach"),
                Criterion::new(Shape, "the car's rear is thrown up", "car.pitch_peak_deg", Some(3.0), Some(20.0), "the frames: the car's rear lifts (pitch positive nose down), more than the SUV's"),
                Criterion::new(Shape, "the SUV rides over (nose up)", "suv.pitch_peak_deg", Some(-15.0), Some(3.0), "the frames: the SUV climbs the car's hood, its nose up (pitch negative) or level; an override metric is proposed: `pair.suv_car.override_max`, the largest height of the SUV's lowest front-section node above the car's highest front-section node at the same sample, minus its value at the first sample"),
                Criterion::new(Shape, "the car turns the same sense as in S08", "car.yaw_peak_deg", Some(0.0), Some(25.0), "the force left of its centre of mass turns it positively about world +Y"),
            ]
        },
    }
}
