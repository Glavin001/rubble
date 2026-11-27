//! S8: two equal sedans head-on, each at 55 km/h, 50 % overlap on their driver's (left)
//! sides (video 5:57).
//!
//! First principles: equal masses at equal and opposite speeds, so the common centre of mass
//! stays still and the contact acts as a wall moving with it: each car sees a 55 km/h
//! rigid-barrier test on the overlapping half of its front (15.3 m/s, 169 kJ each). Every
//! number for car_a holds for car_b: symmetry between them is the test. The forces act left of
//! each centre of mass, so both cars turn the same way about the vertical (positive yaw in
//! world terms, nose toward its own left) and stay nearly in line.

use super::*;

const KMH: f32 = 55.0;
/// The share of each car's width that meets the other's.
const OVERLAP: f32 = 0.5;

/// The same bands for each car (metric names are static, so they are built by `concat!`).
macro_rules! per_car {
    ($car:literal) => {{
        use Tier::*;
        [
            Criterion::new(Invariant, "no explosion", concat!($car, ".max_node_speed"), None, Some(46.0), "3 × the 15.3 m/s approach (see S01)"),
            Criterion::new(Physics, "the overlapping side takes more", concat!($car, ".zone.front_lr_ratio"), Some(1.0), None, "only the left half of each front meets the other car").law("the right front is loaded only through the structure from the left, which passes on less than it receives: left work > right work"),
            Criterion::new(Physics, "turns the same sense as the other", concat!($car, ".yaw_peak_deg"), Some(0.0), None, "each car's contact force acts left of its centre of mass").law("angular momentum: an impulse against the motion at the car's own left turns it positively about world +Y; for car_b (heading −Z, left −X) the same sign by point symmetry"),
            Criterion::new(Physics, "crush at its deepest", concat!($car, ".crush_peak"), Some(0.3), Some(0.95), "a 55 km/h offset barrier test each: S01's 0.4–1.1 m at 64 km/h scaled by Campbell's (55 − 8)/(64 − 8) = 0.84"),
            Criterion::new(Physics, "the crush stays", concat!($car, ".crush_residual"), Some(0.21), None, "metal does not spring back: 70 % of the 0.3 m peak floor"),
            Criterion::new(Physics, "overlapping side's crush", concat!($car, ".gauge.length_left.shrink_frac"), Some(0.05), Some(0.22), "S01's 0.06–0.25 of the left side scaled by Campbell's 0.84; crush_peak is the whole car's length, which the unstruck right corner holds"),
            Criterion::new(Physics, "each car stops", concat!($car, ".speed_end"), None, Some(3.1), "each car meets a wall at its own speed: rebound at most 20 % of 15.3 m/s (NHTSA restitution 0.13–0.20)"),
            Criterion::new(Physics, "the front absorbs it", concat!($car, ".zone.front.work_frac"), Some(0.6), None, "a 55 km/h frontal is what the crush zone is built for"),
            Criterion::new(Physics, "the cabin carries it", concat!($car, ".zone.cabin.work_frac"), None, Some(0.15), "the survival cell passes the load on and stays nearly whole"),
            Criterion::new(Physics, "struck door opening", concat!($car, ".gauge.door_left.shrink_frac"), None, Some(0.10), "the overlapping side's aperture closes at most 10 % (as S01, a lower speed)"),
            Criterion::new(Physics, "no rollover", concat!($car, ".roll_peak_deg"), Some(-45.0), Some(45.0), "a frontal force below the centre of mass does not turn the car over"),
            Criterion::new(Shape, "the rear lifts", concat!($car, ".pitch_peak_deg"), Some(3.0), Some(15.0), "both rears lift off the ground (pitch positive nose down)"),
            Criterion::new(Shape, "nearly in line, the same sense", concat!($car, ".yaw_peak_deg"), Some(0.0), Some(20.0), "the force left of each centre of mass turns both cars positively about world +Y"),
        ]
    }};
}

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S08",
        title: "Two sedans head-on, 2 × 55 km/h, 50 % overlap",
        reference: Reference { video: VIDEO, at_s: &[357], real_test: "car-to-car frontal offset, 50 % overlap, equal cars at 55 km/h each" },
        designs: &["sedan"],
        layout: |extent, j| {
            let e = extent("sedan");
            let v = KMH / 3.6 * j.speed;
            // A tenth of a second of closing (both cars move) before the fronts meet.
            let gap = 0.1 * 2.0 * v;
            // car_b faces car_a (yaw π: its left is world −X). Its world x span is
            // [xb − hi.x, xb − lo.x]; overlapping car_a's left (+X) side by OVERLAP of the width.
            let xb = 2.0 * e.hi.x - OVERLAP * e.width();
            let zb = 2.0 * e.hi.z + gap;
            Layout {
                bodies: vec![
                    Body::new("car_a", "sedan", Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v),
                    Body::new("car_b", "sedan", Vec3::new(xb, 0.0, zb), std::f32::consts::PI, v),
                ],
                boxes: vec![],
                impact_s: 0.1,
            }
        },
        secs_after_impact: 1.0,
        // The fronts meet at z ≈ 2.5 (the sedan's front is 0.96 m ahead of its origin), x ≈ 0.5.
        cameras: &[
            // Framed as the reference footage (60° vertical field of view, 16:9): the two cars (z −2.4
            // to 7.4 at impact, meeting at z 2.5) fill about two thirds of the width.
            Camera { name: "side", eye: [8.3, 1.1, 2.5], target: [0.47, 0.7, 2.5] },
            // Straight down, the cars' length across the frame (the eye a centimetre to −x fixes the
            // picture's up as +x).
            Camera { name: "top", eye: [0.46, 8.5, 2.5], target: [0.47, 0.7, 2.5] },
            Camera { name: "quarter", eye: [-7.0, 5.0, -1.5], target: [0.47, 0.5, 2.5] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "peak", after_impact_ms: 70.0 },
            Moment { name: "rears_lifted", after_impact_ms: 120.0 },
            Moment { name: "rest", after_impact_ms: 950.0 },
        ],
        criteria: || {
            use Tier::*;
            let mut c = vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(340.0), "energy is never created: at most 0.1 % of the 338 kJ two 1,448 kg sedans bring at 15.3 m/s").law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
                Criterion::new(Invariant, "momentum along the road", "all.momentum_z_end", Some(-450.0), Some(450.0), "equal and opposite momenta (22.1 kN·s each) sum to zero, and contact and a symmetric ground keep it there: 2 % of one car's").law("momentum is conserved: the cars' forces on each other cancel, and the ground's friction, the only external horizontal force, is symmetric up to the jitter"),
                Criterion::new(Invariant, "momentum across the road", "all.momentum_x_end", Some(-450.0), Some(450.0), "zero at the start up to car_a's 0.5° jitter (190 kg·m/s); 2 % of one car's momentum").law("momentum is conserved (as along the road)"),
                Criterion::new(Invariant, "the cabins stay apart", "pair.car_a_car_b.com_gap_min", Some(2.5), None, "the centres of mass start 4.9 m apart at contact, each 1.3 m behind its firewall; cabins that met would put them under 2 × 1.3 m = 2.6 m (with the 0.94 m offset)"),
            ];
            c.extend(per_car!("car_a"));
            c.extend(per_car!("car_b"));
            c.extend([
                Criterion::new(Physics, "symmetric crush", "pair.car_a_car_b.crush_peak_ratio", None, Some(1.15), "equal cars at equal speeds: the larger crush within 15 % of the smaller (the 8-run spread of S02 is a few %)"),
                Criterion::new(Physics, "symmetric work", "pair.car_a_car_b.plastic_work_ratio", None, Some(1.15), "equal cars: each absorbs half of the 338 kJ, the larger within 15 % of the smaller"),
                Criterion::new(Physics, "symmetric turn", "pair.car_a_car_b.yaw_peak_diff_deg", None, Some(5.0), "point symmetry about the contact maps one car's turn onto the other's"),
                Criterion::new(Invariant, "the common centre of mass stays still", "all.com_speed_end", None, Some(0.3), "zero total momentum stays zero: 2 % of the 15.3 m/s approach").law("a symmetric head-on's common centre of mass stays at rest (equal masses, equal and opposite velocities)"),
            ]);
            c
        },
    }
}
