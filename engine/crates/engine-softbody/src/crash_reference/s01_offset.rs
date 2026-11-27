//! S1: 40 % offset rigid block, 64 km/h (Euro NCAP offset deformable barrier; the BeamNG half
//! of the video uses a rigid block; video 1:52 and 2:06).
//!
//! First principles: 17.8 m/s (229 kJ on the 1,448 kg sedan) stopped by the struck 40 % of
//! the front alone, so the crush is deeper than full width (17.8²/2a = 0.54–1.1 m at 15–30 g);
//! the force acts at the left front corner, off the centre of mass, so the car yaws toward
//! the block (+X, positive yaw) and the rear swings out; the cabin and the rear carry the
//! load through and keep their shape.
//!
//! Sign conventions the criteria assume: yaw positive from +Z toward +X (as [`Body::yaw`]),
//! pitch positive nose down (rear rising), roll positive left side up.

use super::*;

pub(super) fn scenario() -> CrashScenario {
    CrashScenario {
        id: "S01".into(),
        title: "40 % offset rigid block, 64 km/h".into(),
        reference: Reference { video: VIDEO, at_s: &[112, 126], real_test: "Euro NCAP offset deformable barrier (ODB), 40 % overlap, 64 km/h" },
        designs: names(&["sedan"]),
        recreated: false,
        modified_from: None,
        // A rigid block 2 m tall and 1 m deep over the left 40 % of the car's width, running a
        // metre past its side, a tenth of a second of travel ahead.
        layout: family::preset(&families::BLOCK, &[("car", "sedan".into()), ("speed", 64.0.into()), ("overlap", 0.4.into()), ("height", 2.0.into()), ("depth", 1.0.into()), ("lead", 0.1.into())]),
        // The car is back off the block by 0.1 s and rolls away on free wheels; the yaw is
        // settled by 0.4 s.
        secs_after_impact: 1.0,
        // World coordinates: the sedan's front is 0.96 m ahead of its origin, the block's face
        // 1.78 m beyond (z ≈ 2.7), its inner edge at x ≈ 0.19.
        cameras: vec![
            // Framed as the reference footage (60° vertical field of view, 16:9): the side and top
            // views fill about two thirds of the width with the car at impact (it spans z −2.1 to
            // 2.7 then), the barrier at the left edge of the side view as in the film, and room on the
            // right for the rebound (the car's centre ends near z −2.6).
            Camera { name: "side".into(), eye: [5.2, 1.0, -0.6], target: [0.0, 0.7, -0.6] },
            // Straight down, the car's length across the frame, travelling left to right (struck side up):
            // the eye a centimetre to −x of the target fixes the picture's up as +x.
            Camera { name: "top".into(), eye: [-0.01, 6.4, -0.6], target: [0.0, 0.7, -0.6] },
            Camera { name: "front_quarter".into(), eye: [-3.4, 1.9, 5.2], target: [0.2, 0.6, 1.4] },
        ],
        moments: vec![
            Moment { name: "before".into(), after_impact_ms: -10.0 },
            // The reference's "early crush" frame: bumper and lamp gone, the hood starting to buckle.
            Moment { name: "early_crush".into(), after_impact_ms: 20.0 },
            Moment { name: "wheel_at_sill".into(), after_impact_ms: 40.0 },
            Moment { name: "peak".into(), after_impact_ms: 60.0 },
            // The reference's "rebound" frame: the front destroyed, the car moving back off the block.
            Moment { name: "rebound".into(), after_impact_ms: 150.0 },
            Moment { name: "rear_swing".into(), after_impact_ms: 300.0 },
            Moment { name: "rest".into(), after_impact_ms: 950.0 },
        ],
        criteria: || {
            use Tier::*;
            vec![
                Criterion::new(Invariant, "state stays finite", "all.finite", Some(1.0), None, "a finite state is the least a run owes").law("every position and velocity is a real number"),
                Criterion::new(Invariant, "no energy created", "all.energy_gain_j", None, Some(230.0), "at most 0.1 % of the 229 kJ the 1,448 kg sedan brings at 17.8 m/s").law("energy is never created; 0.1 % is the 1.001 rounding allowance of the energy rule"),
                Criterion::new(Invariant, "nothing passes into the block", "all.box_depth_m", None, Some(0.05), "no node deeper than 5 cm inside the block (a node moves 0.9 cm a step at 17.8 m/s)").law("a static box is impenetrable"),
                Criterion::new(Invariant, "no explosion", "car.max_node_speed", None, Some(53.0), "3 × the 17.8 m/s approach: a light piece struck by the oncoming body leaves at up to 2× the closing speed (ball off a bat) and a breaking beam's stored energy adds tens of m/s to a node; an instability grows without bound"),
                Criterion::new(Physics, "the struck side takes more", "car.zone.front_lr_ratio", Some(1.0), None, "only the left 40 % touches the block").law("the right front is loaded only through the structure from the left, which passes on less than it receives: left work > right work"),
                Criterion::new(Physics, "the car turns toward the block", "car.yaw_peak_deg", Some(0.0), None, "the block's impulse acts at the left front corner, left of the centre of mass").law("angular momentum: an impulse −Z applied at +X of the centre of mass turns it positively about +Y, the nose toward +X"),
                Criterion::new(Physics, "crush at its deepest", "car.crush_peak", Some(0.4), Some(1.1), "17.8²/2a over 40 % of the front: 0.54–1.1 m at 15–30 g, floor at S02's 0.4 m; this metric is the whole car's length along its heading, which the unstruck right corner holds, so an offset crash reads low"),
                Criterion::new(Physics, "struck side's crush", "car.gauge.length_left.shrink_frac", Some(0.06), Some(0.25), "residual 0.3–1.2 m (70 % of 0.4–1.1 m peak, plus margin) of the 4.89 m left side"),
                Criterion::new(Physics, "the crush stays", "car.crush_residual", Some(0.28), None, "metal does not spring back: 70 % of the 0.4 m peak floor (NHTSA springback is about 25 % of the dynamic crush); along the first heading, so the yaw and the unstruck corner shorten it"),
                Criterion::new(Physics, "rebound", "car.rebound_ratio", None, Some(0.2), "a plastic crush: NHTSA restitution 0.13–0.20 at 56 km/h full width (crash-reference-targets.md §8); less when 40 % is loaded and the car turns away"),
                Criterion::new(Physics, "the struck side takes the load", "car.zone.front_lr_ratio", Some(2.0), None, "only the left 40 % meets the block: the left front's work at least twice the right's"),
                Criterion::new(Physics, "the front absorbs it", "car.zone.front.work_frac", Some(0.6), None, "the crush zone is built to take a 64 km/h offset test: most plastic work ahead of the firewall"),
                Criterion::new(Physics, "the cabin carries it", "car.zone.cabin.work_frac", None, Some(0.15), "the survival cell passes the load on and stays nearly whole"),
                Criterion::new(Physics, "struck door opening", "car.gauge.door_left.shrink_frac", None, Some(0.10), "the struck side's aperture closes at most 10 % (about 0.1 m)"),
                Criterion::new(Physics, "far door opening", "car.gauge.door_right.shrink_frac", None, Some(0.03), "the unloaded side's aperture is unchanged: NHTSA full-width p90 is 9 mm (§1), so 3 % (3 cm) is generous"),
                Criterion::new(Physics, "the rear keeps its shape", "car.zone.rear.bent_frac", None, Some(0.02), "nothing loads the rear but its own inertia (20–30 g on its mass): at most 2 % of its beams yield"),
                Criterion::new(Physics, "the car turns at least 5°", "car.yaw_peak_deg", Some(5.0), None, "the block's force about 0.6 m left of the centre of mass over the pulse"),
                Criterion::new(Physics, "no rollover", "car.roll_peak_deg", Some(-45.0), Some(45.0), "a frontal force below the centre of mass does not turn the car over"),
                Criterion::new(Shape, "yaw as the frames show", "car.yaw_peak_deg", Some(10.0), Some(35.0), "from above the car turns about the struck corner by 20–30°"),
                Criterion::new(Shape, "struck wheel pushed to the sill", "car.gauge.wheelbase_left.shrink_frac", Some(0.018), Some(0.09), "the left front wheel moves back 5–25 cm on the 2.83 m wheelbase"),
                Criterion::new(Shape, "the rear lifts a little", "car.pitch_peak_deg", Some(0.5), Some(15.0), "the frames show the rear lifting a little"),
                Criterion::new(Shape, "mean deceleration", "car.decel_mean_g", Some(12.0), Some(30.0), "Δv of 17.8–21 m/s (with rebound) over the 80–160 ms pulse below"),
                Criterion::new(Shape, "pulse", "car.pulse_ms", Some(80.0), Some(160.0), "a 40 % offset pulse is longer than full width's 63–76 ms to peak crush (§8)"),
            ]
        },
    }
}
