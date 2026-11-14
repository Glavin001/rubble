//! The generated sedan's raycast wheels in the vehicle lab's compactor (the CPU
//! reference, `vehicle_lab::cpu`): the top plate presses the car down, and a wheel
//! is a solid, not a spring that lets its hub into the ground.
//!
//! The owner's report (2026-11-13, `vehicle-lab.html`, Compactor, the top plate at
//! 0.10 m/s, 535 kN at 0.52 m of travel): the wheels' rims and tyres stood half
//! below the floor. Measured before the fix (`examples/wheel_floor_probe`): the
//! raycast suspension's bump stop sits 0.183 m of hub travel below the hub's
//! height at rest (its length 0.02 m against the rest length 0.203 m), so pressed
//! down the hubs went to 0.145 m over the ground: 0.185 m under a 0.33 m tyre and
//! 0.055 m under the 0.2 m rim. The floor then reached the ground and took the
//! plate's 500 kN; each wheel carried only its spring's 10 kN.
//!
//! Invariants (from the wheel's geometry and Newton's laws, not recorded outputs):
//! - while the plate is above the wheel's own parts (the rim is intact), the hub
//!   never comes closer to the ground than the rim's radius (less 1 cm: one step of
//!   approach at most, the stop is a velocity-level contact). So the rim's lowest
//!   point is never more than 1 cm under the floor, and the tyre's undeformed
//!   outline (drawn on the hub) never more than its own section height, r − rim:
//!   a fully deflected tyre;
//! - the load reaches the hub's mounts: while a wheel bears on its rim, the beams
//!   that join the wheel and hub to the car push it down with the wheel's load (the
//!   wheel is in equilibrium: what the ground gives it, the mounts take);
//! - pressed flat, the mounts yield or break (the wheel folds into the body or
//!   leaves it) instead of the hub passing through the floor.
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::LabOptions;
use serde_json::json;

fn sedan() -> SoftVehicleDesc {
    let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
    let r = lattice::rig_with(&inp);
    let files = vg_input::physics_files(&inp, &r);
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = texts
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = texts
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| b.as_str());
    engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
}

/// Per raycast wheel: (rim radius from the content, the wheel's and hub's own nodes).
struct Corner {
    rim: f32,
    radius: f32,
    own: Vec<usize>,
}

fn corners(d: &SoftVehicleDesc, s: &ExplicitBeams) -> Vec<Corner> {
    let drive = s.drive.as_ref().expect("raycast wheels");
    drive
        .config
        .wheels
        .iter()
        .map(|w| {
            let pw = d
                .pressure_wheels
                .iter()
                .find(|p| {
                    (p.node1, p.node2) == (w.node1, w.node2)
                        || (p.node1, p.node2) == (w.node2, w.node1)
                })
                .expect("the wheel's pressureWheels row");
            let rim = *pw.values.get("hubRadius").expect("the row's hubRadius");
            // `whefl_ai` → corner `fl`: the wheel's (`whefl_*`) and hub's (`hubfl_*`) nodes.
            let n1 = &d.nodes[w.node1 as usize].name;
            let c = &n1.split('_').next().unwrap()[3..];
            let own = d
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    n.name.starts_with(&format!("whe{c}_"))
                        || n.name.starts_with(&format!("hub{c}_"))
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            assert!(own.len() >= 6, "corner {c}: {} own nodes", own.len());
            Corner {
                rim,
                radius: w.radius,
                own,
            }
        })
        .collect()
}

struct Press {
    /// Lowest hub height (m) over the ground before the plate, or the body it
    /// crushed, first reached the wheel's own nodes, and that wheel's rim radius.
    /// From then the crush bears on the wheel itself: the plate is kinematic (any
    /// force), and the body's contact with the wheel's nodes separates them by
    /// position steps (§37), which the rim stop, a force as the solver's ground is,
    /// does not resist (`known-bugs.md` W1).
    hub_low: f32,
    rim: f32,
    radius: f32,
    /// Largest wheel load (N) and the mounts' downward push on that wheel then (N).
    load_peak: f32,
    mounts_then: f32,
    /// Per wheel: mount beams yielded or broken when the crush first reached the
    /// wheel (or at the end, if it never did).
    folded: Vec<usize>,
    /// Lowest hub (m) at the end, and whether the run stayed finite.
    hub_end: f32,
}

fn press(speed: f32, gap: f32, secs: f32) -> Press {
    let d = sedan();
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let (template, _) =
        ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings)
            .expect("spawn");
    let cs = corners(&d, &template);
    let script = json!({ "mode": "compactor", "scene": "flat", "surface": true,
        "events": [{ "t": 0.5, "plates": { "axis": "top", "speed": speed, "gap": gap } }] });
    let mut lab = CpuLab::new(&template, LabOptions::from_json(&script)).expect("lab");
    lab.bench.set_script(&script).expect("script");
    let rest0: Vec<f32> = lab.sim.beams.iter().map(|b| b.rest).collect();
    let mut touched = vec![false; cs.len()];
    let mut folded_at = vec![None; cs.len()];
    let mounts_moved = |s: &ExplicitBeams, c: &Corner| {
        s.beams
            .iter()
            .zip(&rest0)
            .filter(|(b, &r0)| {
                c.own.contains(&(b.a as usize)) != c.own.contains(&(b.b as usize))
                    && (b.broken || (b.rest - r0).abs() > 1e-3 * r0)
            })
            .count()
    };
    let mut out = Press {
        hub_low: f32::MAX,
        rim: 0.0,
        radius: 0.0,
        load_peak: 0.0,
        mounts_then: 0.0,
        folded: Vec::new(),
        hub_end: f32::MAX,
    };
    for _ in 0..(secs * 60.0) as usize {
        lab.step().expect("finite");
        let s = &lab.sim;
        let g = s.ground;
        let y = |i: usize| s.nodes[i].pos.y + s.origin.y - g;
        let plate_bottom = lab
            .bench
            .plates
            .first()
            .map_or(f32::MAX, |p| gap + p.stroke - p.travel);
        let dr = s.drive.as_ref().unwrap();
        for (k, (c, (w, st))) in cs
            .iter()
            .zip(dr.config.wheels.iter().zip(&dr.wheels))
            .enumerate()
        {
            let hub = 0.5 * (y(w.node1 as usize) + y(w.node2 as usize));
            let top = c.own.iter().map(|&i| y(i)).fold(f32::MIN, f32::max);
            // Reached: the plate is at the wheel's nodes, or the body it crushed is on
            // them (a contact's position step on one of them, §37).
            let pressed = c
                .own
                .iter()
                .any(|&i| s.contact_shift.get(i).is_some_and(|a| *a != [0, 0, 0]));
            if !touched[k] && (plate_bottom <= top + 0.01 || pressed) {
                touched[k] = true;
                folded_at[k] = Some(mounts_moved(s, c));
            }
            if !touched[k] && hub < out.hub_low {
                out.hub_low = hub;
                out.rim = c.rim;
                out.radius = c.radius;
            }
            // The mounts: beams from the wheel's and hub's own nodes to the rest of
            // the car, their push on the wheel along up (Beam::force > 0: compression).
            if st.load > out.load_peak && hub < c.rim + 0.02 {
                let mut down = 0.0f32;
                for b in &s.beams {
                    let (a, bb) = (b.a as usize, b.b as usize);
                    let (ina, inb) = (c.own.contains(&a), c.own.contains(&bb));
                    if b.broken || ina == inb {
                        continue;
                    }
                    let (inside, outside) = if ina { (a, bb) } else { (bb, a) };
                    let dir = (s.nodes[inside].pos - s.nodes[outside].pos).normalize_or_zero();
                    // Compression pushes the inside end away from the outside one.
                    down -= b.force * dir.y;
                }
                out.load_peak = st.load;
                out.mounts_then = down;
            }
        }
    }
    let s = &lab.sim;
    let dr = s.drive.as_ref().unwrap();
    for (k, (c, w)) in cs.iter().zip(&dr.config.wheels).enumerate() {
        out.folded
            .push(folded_at[k].unwrap_or_else(|| mounts_moved(s, c)));
        let hub = 0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y)
            + s.origin.y
            - s.ground;
        out.hub_end = out.hub_end.min(hub);
    }
    out
}

/// The plate down at 0.25 m/s to 0.7 m over the ground: the body crushed onto the
/// wheels, the plate never on them.
#[test]
fn pressed_to_a_stop_the_hubs_stay_a_rim_over_the_ground() {
    let p = press(0.25, 0.7, 5.5);
    eprintln!(
        "hub lowest {:.3} m (rim {:.3}, tyre {:.3}); peak load {:.1} kN, mounts {:.1} kN; mounts yielded/broken {:?}; hub at end {:.3}",
        p.hub_low,
        p.rim,
        p.radius,
        p.load_peak / 1e3,
        p.mounts_then / 1e3,
        p.folded,
        p.hub_end
    );
    assert!(
        p.hub_low >= p.rim - 0.01,
        "a hub came to {:.3} m over the ground, under its rim's {:.3} m: the rim is {:.3} m and the tyre {:.3} m into the floor",
        p.hub_low,
        p.rim,
        p.rim - p.hub_low,
        p.radius - p.hub_low
    );
    assert!(
        p.load_peak > 0.0 && (p.mounts_then - p.load_peak).abs() <= 0.25 * p.load_peak,
        "a wheel on its rim carried {:.1} kN but its mounts pushed it down with {:.1} kN",
        p.load_peak / 1e3,
        p.mounts_then / 1e3
    );
}

/// The plate down at 0.25 m/s to 5 cm (flat): the rims hold until the crush (the
/// plate, or the body it flattened) reaches the wheels themselves, and by then the
/// wheels' mounts have yielded or broken under the load the rims passed into them.
#[test]
fn pressed_flat_the_wheels_fold_instead_of_sinking() {
    let p = press(0.25, 0.05, 8.0);
    eprintln!(
        "hub lowest {:.3} m (rim {:.3}, tyre {:.3}) before the crush reached the wheels; peak load {:.1} kN, mounts {:.1} kN; mounts yielded/broken {:?}; hub at end {:.3}",
        p.hub_low,
        p.rim,
        p.radius,
        p.load_peak / 1e3,
        p.mounts_then / 1e3,
        p.folded,
        p.hub_end
    );
    assert!(
        p.hub_low >= p.rim - 0.01,
        "a hub came to {:.3} m over the ground under its rim's {:.3} m before the crush reached the wheel",
        p.hub_low,
        p.rim
    );
    assert!(
        p.folded.iter().all(|&n| n > 0),
        "pressed flat, a wheel's mounts neither yielded nor broke before the crush reached it: {:?}",
        p.folded
    );
}
