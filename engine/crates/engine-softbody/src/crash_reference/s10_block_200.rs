//! S10: narrow rigid block, 200 km/h, 40 % overlap on the left front (video 6:33).
//!
//! First principles: 55.6 m/s, the same energy class as S07 (15.4× a 50 km/h test), but only
//! the left 40 % of the width meets the block. That side is cut open back through the front
//! door while the right carries on: the force acts left of and below the centre of mass, so
//! the car yaws about the block toward +X and pitches its rear up, and much of the energy
//! leaves as rotation and residual speed rather than crush, which is why the rear survives
//! here and not in S07.

use super::*;
use super::s07_wall_190::WAGON_KG;

const KMH: f32 = 200.0;
/// The share of the car's width the block covers, from its left (+X) side.
const OVERLAP: f32 = 0.4;
/// The block's width (m): it covers OVERLAP of the car and stands out past its left side.
const BLOCK_WIDTH: f32 = 0.9;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S10",
        title: "Narrow rigid block, 200 km/h, 40 % overlap",
        reference: Reference { video: VIDEO, at_s: &[393], real_test: "narrow rigid block, 40 % of the width, 200 km/h (no standard test)" },
        designs: &["wagon"],
        layout: |extent, j| {
            let e = extent("wagon");
            let v = KMH / 3.6 * j.speed;
            // A tenth of a second of travel before the block.
            let gap = 0.1 * v;
            let face = e.hi.z + gap;
            let x0 = e.hi.x - OVERLAP * e.width();
            Layout {
                bodies: vec![Body::new("car", "wagon", Vec3::new(j.lateral, 0.0, 0.0), j.yaw, v)],
                // A concrete block 0.9 m wide, 2 m tall and 2 m deep.
                boxes: vec![(Vec3::new(x0 + 0.5 * BLOCK_WIDTH, 1.0, face + 1.0), Vec3::new(0.5 * BLOCK_WIDTH, 1.0, 1.0))],
                impact_s: 0.1,
            }
        },
        secs_after_impact: 2.0,
        cameras: &[
            // Close, from the side away from the block (the wagon spans z 1.7 to 6.5 at impact): from
            // the struck side the block hides the car swinging round it. High enough for the rear
            // rising (60° vertical field of view).
            Camera { name: "side", eye: [-5.2, 1.6, 4.6], target: [-0.3, 1.4, 4.6] },
            // On the wreck where it ends: past the block, about 8 m to −x (its centre near z 17.8).
            Camera { name: "aftermath", eye: [-4.3, 2.4, 21.6], target: [-7.8, 0.6, 17.8] },
            // Straight down on the whole path, travelling left to right (the eye a centimetre to −x
            // of the target fixes the picture's up as +x): the block, and the wreck about 10 m past
            // it and 8 m to the side.
            Camera { name: "top", eye: [-4.01, 12.8, 10.5], target: [-4.0, 0.6, 10.5] },
        ],
        moments: &[
            Moment { name: "before", after_impact_ms: -10.0 },
            Moment { name: "door_reached", after_impact_ms: 30.0 },
            Moment { name: "peak", after_impact_ms: 60.0 },
            Moment { name: "rear_up", after_impact_ms: 250.0 },
            Moment { name: "rest", after_impact_ms: 1950.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(WAGON_KG * 0.5 * (KMH as f64 / 3.6).powi(2) * 1e-3), "at most 0.1 % of the 2.24 MJ the 1,448 kg wagon brings at 55.6 m/s").law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
                Criterion::new(Invariant, "nothing passes into the block", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside the block (a node moves 2.8 cm a step at 55.6 m/s)").law("a static box is impenetrable"),
                Criterion::new(Invariant, "no explosion", "car.max_node_speed", None, Some(167.0), "3 × the 55.6 m/s approach (see S01)"),
                Criterion::new(Physics, "struck door opening collapses", "car.gauge.door_left.shrink_frac", Some(0.4), None, "the left side is destroyed back through the front door: its aperture closes by at least 40 %"),
                Criterion::new(Physics, "the struck side takes more", "car.zone.lr_ratio", Some(1.0), None, "only the left 40 % touches the block").law("the right side is loaded only through the structure from the left, which passes on less than it receives: left work > right work"),
                Criterion::new(Physics, "the struck side takes the load", "car.zone.lr_ratio", Some(2.0), None, "only the left 40 % meets the block: the left's work at least twice the right's"),
                Criterion::new(Physics, "the struck side is shortened", "car.gauge.length_left.shrink_frac", Some(0.2), None, "everything ahead of the B-pillar (about 2 m of the 4.85 m car, 0.41) crushed or torn off on the left; ×0.5"),
                Criterion::new(Physics, "parts come off", "car.detached_parts", Some(1.0), None, "a side cut open at 200 km/h tears at least one part (wheel, panel) away"),
                Criterion::new(Physics, "the rear pitches up", "car.pitch_peak_deg", Some(20.0), None, "the block's force below the centre of mass pitches the rear up (pitch positive nose down)"),
                Criterion::new(Physics, "the car turns toward the block", "car.yaw_peak_deg", Some(0.0), None, "the block's impulse acts left of the centre of mass").law("angular momentum: an impulse −Z applied at +X of the centre of mass turns it positively about +Y, the nose toward +X"),
                Criterion::new(Physics, "the car turns at least 10°", "car.yaw_peak_deg", Some(10.0), None, "the force acts about 0.55 m left of the centre of mass at 55.6 m/s: a large angular impulse"),
                Criterion::new(Physics, "the rear survives better than in S07", "car.zone.rear.bent_frac", None, Some(0.05), "energy leaves as rotation and residual speed, so the rear is barely reached: at most 5 % of its beams yield (to be compared with S07's measured value, which must be higher)"),
                Criterion::new(Shape, "rear up as the frames show", "car.pitch_peak_deg", Some(40.0), Some(70.0), "the rear rises to 40–60°"),
                Criterion::new(Shape, "the wreck ends beside or past the block", "car.com_end_past_box0", Some(0.0), None, "the car swings round the block: its centre of mass ends at or past the block's near face"),
                Criterion::new(Shape, "parts on the ground", "car.detached_parts", Some(2.0), None, "a wheel and a panel on the ground in the aftermath"),
            ]
        },
    }
}
