//! The generated sedan must not create energy, however hard it is hit
//! (known-bugs S3; `examples/sidepole_sweep` is the measuring tool).
//!
//! **Bug 1, the gapped stops (authoring, fixed in d136f77f).** The rigger's 42
//! stops (bumper covers on the impact beams, the front wheels on the sill and
//! A-pillar) were BOUNDED beams with no base spring, a 2e6 N/m limit spring and
//! a long bound of 100 lengths. A bounded beam's limit spring acts on its whole
//! extension once a bound is crossed, so each stop was a 1 MN force wall where
//! it engaged and a tether a hundred lengths out: in the crash lab's side pole
//! at 186 km/h the front right wheel, torn off, reached 55 m at 1.27 s and the
//! beams' work rose by 1.8e8 J in 20 ms. `RIG_BOUNDED_STOPS=1` rigs the car that
//! way again: `a_stop_between_separated_parts_does_no_work` and
//! `a_torn_off_wheel_adds_no_energy` then fail (checked 2026-11-13: a force of
//! 1.1e8 N at 101 lengths; the wheel's run not finite).
//!
//! **Bug 2, centred damping on anything that spins (solver; the objective
//! form, `ExplicitBeamsSettings::centred_objective`).** See
//! `tests/centred_damping_objective.rs`. The whole-car scenes below run with
//! the objective form on; with it off the side pole at 186 km/h is not finite
//! at 3.0 s and the 42° wall at 106 km/h doubles its kinetic energy every
//! 0.1 s from 2.0 s (the page's report of 2026-11-13).
//!
//! **The invariant:** nothing drives these scenes, so the car's kinetic +
//! gravitational + elastic energy never rises above where it started.
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{
    jbeam_force, ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, StaticBox, STEP,
};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};
use std::sync::OnceLock;

/// The sedan as the rigger exports it (JBeam, in memory).
fn sedan() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
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
    })
}

/// The sedan on raycast wheels, facing +Z turned by `yaw`, its lowest wheel
/// 2 cm up, its middle over the origin; the objective damping on.
fn spawn(yaw: f32) -> ExplicitBeams {
    let d = sedan();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::ZERO, Vec3::ZERO, settings).expect("spawn");
    s.settings.centred_objective = true;
    let f = engine_softbody::pack::forward_axis(d);
    let q = Quat::from_rotation_y(yaw - f.x.atan2(f.z));
    let o = s.origin;
    s.origin = Vec3::ZERO;
    for n in &mut s.nodes {
        n.pos = q * (n.pos + o);
    }
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    let wheel_low = s
        .drive
        .as_ref()
        .map(|dr| {
            dr.config
                .wheels
                .iter()
                .map(|w| {
                    0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y)
                        - w.radius
                })
                .fold(f32::MAX, f32::min)
        })
        .expect("raycast wheels");
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 0.02 - wheel_low, -0.5 * (lo.z + hi.z));
    for n in &mut s.nodes {
        n.pos += shift;
    }
    s
}

fn names() -> Vec<String> {
    sedan().nodes.iter().map(|n| n.name.clone()).collect()
}

/// Runs `s` with the ledger: the largest rise of kinetic + gravitational +
/// elastic above its start (J), the start (J), and whether it stayed finite.
fn run(s: ExplicitBeams, seconds: f32) -> (f64, f64, bool) {
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().mechanical();
    let mut rise = 0.0f64;
    for _ in 0..(seconds / STEP) as usize {
        w.step();
        let e = w.ledger.as_ref().unwrap().mechanical();
        if !e.is_finite() {
            return (f64::INFINITY, e0, false);
        }
        rise = rise.max(e - e0);
    }
    (rise, e0, true)
}

/// A gapped stop is a beam with no force at its spawn length or a little
/// either side of it. However far its ends part it must stay without force
/// (the parts have separated: there is nothing between them), and where it
/// engages its force must rise from zero, not appear: no step of 1 mm may
/// change it by more than 5 kN (a 5e6 N/m spring; the stops are 2e6).
/// **Failed before d136f77f:** 1.1e8 N at 101 lengths, and 55 kN in the first
/// millimetre past the short bound.
#[test]
fn a_stop_between_separated_parts_does_no_work() {
    let s = spawn(0.0);
    let names = names();
    let mut stops = 0;
    for (j, b) in s.beams.iter().enumerate() {
        if b.broken
            || !matches!(
                b.kind,
                ExplicitKind::JbeamBounded | ExplicitKind::JbeamSupport
            )
        {
            continue;
        }
        let (a, e) = (b.a as usize, b.b as usize);
        let (pa, pe) = (s.nodes[a].pos, s.nodes[e].pos);
        let l = pa.distance(pe);
        let dir = (pe - pa) / l;
        let force_at = |len: f32| -> f32 {
            let mut beam = *b;
            let (mut na, mut ne) = (s.nodes[a], s.nodes[e]);
            na.vel = Vec3::ZERO;
            ne.vel = Vec3::ZERO;
            ne.pos = pa + dir * len;
            let out = jbeam_force(&mut beam, &na, &ne, false);
            if beam.broken {
                0.0
            } else {
                out.force
            }
        };
        // Gapped: no force at the spawn length or 2 mm either side.
        if [l - 0.002, l, l + 0.002]
            .iter()
            .any(|&x| force_at(x) != 0.0)
        {
            continue;
        }
        stops += 1;
        let what = format!("stop {j} ({} to {}, {l:.3} m)", names[a], names[e]);
        for times in [1.5f32, 2.0, 10.0, 99.0, 101.0, 102.0, 150.0, 1000.0] {
            let f = force_at(l * times);
            assert!(
                f == 0.0,
                "{what}: {f:.3e} N with its ends {times} lengths apart"
            );
        }
        let mut last = 0.0f32;
        let mut len = l;
        while len > 0.05 * l {
            len -= 0.001;
            // (Past its yield a stop crushes: plastic, not a spring. The
            // elastic range is what must be continuous.)
            let mut probe = *b;
            let (mut na, mut ne) = (s.nodes[a], s.nodes[e]);
            na.vel = Vec3::ZERO;
            ne.vel = Vec3::ZERO;
            ne.pos = pa + dir * len;
            jbeam_force(&mut probe, &na, &ne, false);
            if probe.rest != b.rest {
                break;
            }
            let f = force_at(len);
            assert!(
                (f - last).abs() <= 5.0e3,
                "{what}: its force steps from {last:.0} N to {f:.0} N in 1 mm at {len:.3} m"
            );
            last = f;
        }
    }
    assert!(
        stops >= 12,
        "{stops} gapped stops found: the front wheels' alone are 12"
    );
}

/// The front right wheel leaves the car at 60 m/s (its mounts broken, the
/// stops left as they are) and travels 90 m in 1.5 s, past the 55 m at which
/// the old stops' long bound caught it. Nothing may add energy.
/// **Failed before d136f77f:** not finite (the beams' work +1.8e8 J in 20 ms
/// in the side pole).
#[test]
fn a_torn_off_wheel_adds_no_energy() {
    let mut s = spawn(0.0);
    let names = names();
    let wheel: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| names[i].starts_with("whefr") || names[i].starts_with("hubcfr"))
        .collect();
    assert!(
        wheel.len() >= 5,
        "the front right wheel's nodes: {}",
        wheel.len()
    );
    let on = |i: u32| wheel.contains(&(i as usize));
    let l0: Vec<f32> = s
        .beams
        .iter()
        .map(|b| {
            s.nodes[b.a as usize]
                .pos
                .distance(s.nodes[b.b as usize].pos)
        })
        .collect();
    let mut cut = 0;
    for (j, b) in s.beams.iter_mut().enumerate() {
        if on(b.a) == on(b.b) || b.broken {
            continue;
        }
        // Its mounts are the beams that hold it at rest; a gapped stop holds nothing there.
        let holds = !matches!(
            b.kind,
            ExplicitKind::JbeamBounded | ExplicitKind::JbeamSupport
        ) || (b.kind == ExplicitKind::JbeamBounded && b.k > 0.0)
            || (b.kind == ExplicitKind::JbeamSupport && b.rest >= l0[j]);
        if holds {
            b.broken = true;
            cut += 1;
        }
    }
    assert!(cut >= 8, "{cut} mount beams cut");
    let centre = wheel.iter().map(|&i| s.nodes[i].pos).sum::<Vec3>() / wheel.len() as f32;
    let away = Vec3::new(centre.x, 0.0, 0.0).normalize() * 60.0 + Vec3::Y * 3.0;
    for &i in &wheel {
        s.nodes[i].vel = away;
    }
    let (rise, e0, finite) = run(s, 1.5);
    assert!(finite, "not finite");
    assert!(
        rise <= 1e-3 * e0,
        "energy rose {rise:.0} J above its start ({e0:.0} J)"
    );
}

/// The crash lab's side pole at 186 km/h, 1.05 m ahead of the middle (the
/// report of 2026-11-13: warping, then 4,861 km/h and a lost GPU device): the
/// car carried sideways into a 20 cm pole, 3 s.
#[test]
fn the_side_pole_at_186_kmh_never_gains_energy() {
    let mut s = spawn(std::f32::consts::FRAC_PI_2);
    let v = 186.0 / 3.6;
    for n in &mut s.nodes {
        n.vel = Vec3::Z * v;
    }
    engine_softbody::crash_rigs::on_carrier(&mut s);
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(1.05, 2.0, front + 1.0 + 0.1),
        Vec3::new(0.1, 2.0, 0.1),
    ));
    s.surface_contact = true;
    let (rise, e0, finite) = run(s, 3.0);
    assert!(finite, "not finite");
    assert!(
        rise <= 1e-3 * e0,
        "energy rose {rise:.0} J above its start ({e0:.0} J)"
    );
}

/// The crash lab's wall at 106 km/h, the car turned 42° to it (the report of
/// 2026-11-13: nearly stopped, then its kinetic energy doubling every 0.1 s):
/// 4 s.
#[test]
fn the_42_degree_wall_at_106_kmh_never_gains_energy() {
    let yaw = (-42.0f32).to_radians();
    let mut s = spawn(yaw);
    let fw = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    for n in &mut s.nodes {
        n.vel = fw * (106.0 / 3.6);
    }
    if let Some(d) = s.drive.as_mut() {
        d.roll(106.0 / 3.6);
    }
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 2.0, front + 1.0 + 0.5),
        Vec3::new(12.0, 2.0, 0.5),
    ));
    let (rise, e0, finite) = run(s, 4.0);
    assert!(finite, "not finite");
    assert!(
        rise <= 1e-3 * e0,
        "energy rose {rise:.0} J above its start ({e0:.0} J)"
    );
}

/// Two sedans, one parked side-on, the other bumping it at 25 km/h: `offset`
/// is the parked car's shift along its own length (m). Runs 5 s past the
/// contact with the ledger; returns (largest rise J, start J, finite) and
/// prints which parts' joints broke.
fn bump(offset: f32, objective: bool) -> (f64, f64, bool) {
    let v = 25.0 / 3.6;
    let mut player = spawn(0.0);
    let mut parked = spawn(std::f32::consts::FRAC_PI_2);
    player.settings.centred_objective = objective;
    parked.settings.centred_objective = objective;
    let front = player
        .nodes
        .iter()
        .map(|n| n.pos.z)
        .fold(f32::MIN, f32::max);
    let side = parked
        .nodes
        .iter()
        .map(|n| n.pos.z)
        .fold(f32::MAX, f32::min);
    for n in &mut parked.nodes {
        n.pos += Vec3::new(offset, 0.0, front - side + 1.0);
    }
    for n in &mut player.nodes {
        n.vel = Vec3::Z * v;
    }
    if let Some(d) = player.drive.as_mut() {
        d.roll(v);
    }
    let mut w = ExplicitBeamsWorld::new(vec![player, parked]);
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().mechanical();
    let mut rise = 0.0f64;
    let mut finite = true;
    for _ in 0..((1.0 / v + 5.0) / STEP) as usize {
        w.step();
        let e = w.ledger.as_ref().unwrap().mechanical();
        if !e.is_finite() {
            finite = false;
            break;
        }
        rise = rise.max(e - e0);
    }
    let d = sedan();
    let mut part = vec![usize::MAX; d.nodes.len()];
    for (k, p) in d.parts.iter().enumerate() {
        for &i in &p.nodes {
            if part[i as usize] == usize::MAX {
                part[i as usize] = k;
            }
        }
    }
    for (vi, car) in w.vehicles.iter().enumerate() {
        let mut pairs: std::collections::BTreeMap<(String, String), usize> = Default::default();
        let name = |i: u32| {
            part.get(i as usize)
                .and_then(|&k| d.parts.get(k))
                .map_or("?".to_string(), |p| p.name.clone())
        };
        for b in car.beams.iter().filter(|b| b.broken) {
            let (a, e) = (name(b.a), name(b.b));
            if a != e {
                let key = if a < e { (a, e) } else { (e, a) };
                *pairs.entry(key).or_default() += 1;
            }
        }
        let fastest = car
            .nodes
            .iter()
            .map(|n| n.vel.length())
            .fold(0.0f32, f32::max);
        eprintln!("offset {offset} objective {objective} car {vi}: fastest node {fastest:.1} m/s; broken joints {pairs:?}");
    }
    (rise, e0, finite)
}

/// An ordinary bump (the yard's report of 2026-11-13: after a 25 km/h bump
/// the player's kinetic energy went 155 kJ → 7,569 kJ in 0.65 s): corner to
/// corner and onto a wheel, 5 s after contact. The two cars' energy never
/// rises.
#[test]
fn a_25_kmh_bump_between_two_sedans_never_gains_energy() {
    for offset in [2.6f32, -1.2] {
        let (rise, e0, finite) = bump(offset, true);
        assert!(finite, "offset {offset}: not finite");
        assert!(
            rise <= 2e-3 * e0,
            "offset {offset}: energy rose {rise:.0} J above its start ({e0:.0} J)"
        );
    }
}

/// A measurement, not a check (`--ignored --nocapture`): the same bump with
/// the end-direction form (the default until the kernels carry the fix).
#[test]
#[ignore]
fn measure_the_25_kmh_bump_with_the_end_direction_form() {
    for offset in [2.6f32, -1.2] {
        let (rise, e0, finite) = bump(offset, false);
        eprintln!("offset {offset}: finite {finite}, rise {rise:.3e} J of {e0:.0} J");
    }
}

/// One part torn off the standing car and flung: every beam that holds it to
/// another part broken (gapped stops left as they are), the part leaving at
/// 15 m/s, outward and up, spinning at `rim` m/s at its farthest node. Runs
/// 1 s; returns (largest rise J, the part's own kinetic energy at the start
/// J, finite), or None when the part cannot leave (a beam that holds it
/// never breaks) or is not a part of its own (fewer than 3 nodes).
fn fling(part: &str, rim: f32, objective: bool) -> Option<(f64, f64, bool)> {
    let d = sedan();
    let p = d.parts.iter().position(|p| p.name == part)?;
    let mut owner = vec![usize::MAX; d.nodes.len()];
    for (k, q) in d.parts.iter().enumerate() {
        for &i in &q.nodes {
            if owner[i as usize] == usize::MAX {
                owner[i as usize] = k;
            }
        }
    }
    let mut s = spawn(0.0);
    s.settings.centred_objective = objective;
    let own: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| owner.get(i) == Some(&p))
        .collect();
    if own.len() < 3 {
        return None;
    }
    let on = |i: u32| owner.get(i as usize) == Some(&p);
    let l0: Vec<f32> = s
        .beams
        .iter()
        .map(|b| {
            s.nodes[b.a as usize]
                .pos
                .distance(s.nodes[b.b as usize].pos)
        })
        .collect();
    for (j, b) in s.beams.iter_mut().enumerate() {
        if on(b.a) == on(b.b) || b.broken {
            continue;
        }
        let holds = !matches!(
            b.kind,
            ExplicitKind::JbeamBounded | ExplicitKind::JbeamSupport
        ) || (b.kind == ExplicitKind::JbeamBounded && b.k > 0.0)
            || (b.kind == ExplicitKind::JbeamSupport && b.rest >= l0[j]);
        if holds {
            if !b.strength.is_finite() && b.group == 0 {
                // It never breaks: the part cannot leave.
                return None;
            }
            b.broken = true;
        }
    }
    let n = own.len() as f32;
    let centre = own.iter().map(|&i| s.nodes[i].pos).sum::<Vec3>() / n;
    let car = s.nodes.iter().map(|x| x.pos).sum::<Vec3>() / s.nodes.len() as f32;
    let out = Vec3::new(centre.x - car.x, 0.0, centre.z - car.z).normalize_or(Vec3::X);
    let away = out * 15.0 + Vec3::Y * 4.0;
    let axis = (out + Vec3::Y * 0.5).normalize();
    let far = own
        .iter()
        .map(|&i| axis.cross(s.nodes[i].pos - centre).length())
        .fold(1e-6f32, f32::max);
    let mut ke = 0.0f64;
    for &i in &own {
        let v = away + axis.cross(s.nodes[i].pos - centre) * (rim / far);
        s.nodes[i].vel = v;
        ke += 0.5 * f64::from(s.nodes[i].mass) * f64::from(v.length_squared());
    }
    let (rise, _, finite) = run(s, 1.0);
    Some((rise, ke, finite))
}

/// **The lint's dynamic check** (`engine/docs/generated-vehicles.md`, energy
/// bombs): every part that can leave the car is torn off, flung and spun
/// (5 and 20 m/s at its rim), and nothing may add energy: not a stop that
/// still reaches for it, not its own beams' damping as it turns. The rise
/// allowed is 2 % of the part's own kinetic energy (it lands on the car and
/// the ground: contacts and the plain step's own single-precision wander).
/// **Fails with the end-direction damping** (the last assertion keeps the
/// record: the front left hub cap, spun at 20 m/s, is not finite).
#[test]
fn every_part_torn_off_and_spun_adds_no_energy() {
    let parts: Vec<String> = sedan().parts.iter().map(|p| p.name.clone()).collect();
    let mut flung = 0;
    for part in &parts {
        for rim in [5.0f32, 20.0] {
            let Some((rise, ke, finite)) = fling(part, rim, true) else {
                continue;
            };
            flung += 1;
            assert!(finite, "{part} spun at {rim} m/s: not finite");
            assert!(
                rise <= 0.02 * ke + 20.0,
                "{part} spun at {rim} m/s: energy rose {rise:.0} J (its own kinetic energy {ke:.0} J)"
            );
        }
    }
    assert!(
        flung >= 40,
        "{flung} part runs: the doors, lids, fenders, bumpers, wheels and caps alone are 36"
    );
    let (rise, ke, finite) = fling("sedan_hub_cap_FL", 20.0, false).expect("the hub cap leaves");
    assert!(
        !finite || rise > 10.0 * ke,
        "the end-direction form no longer pumps a flung hub cap (rise {rise:.0} J of {ke:.0} J): update known-bugs S3"
    );
}
