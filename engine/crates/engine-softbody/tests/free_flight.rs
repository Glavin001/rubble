//! A soft-body car in free flight obeys Newton (the vehicle lab's report of
//! 2026-11-17: a torn-off door seemed to orbit the body instead of flying away;
//! a car rolling fast in mid-air seemed to lose its spin).
//!
//! **The invariants** (from the physics, no recorded outputs), on the generated
//! sedan (the lab's `car=gen`, JBeam, raycast wheels) far above any ground:
//! 1. Free flight: total momentum changes by exactly M·g·t; the angular
//!    momentum about the centre of mass is constant without air (the beams'
//!    forces are equal, opposite and along the beam, and the centred damping's
//!    correction is too); the kinetic energy of the rigid rotation is not damped
//!    by the internal damping (only the ring-down of the centrifugal stretch is).
//!    With air drag the loss is the air's, and it is measured.
//! 2. A part whose every beam to the body is broken moves ballistically: its
//!    centre of mass follows x₀ + v₀·t + ½·g·t² without air, and its motion is
//!    the same bit for bit whatever the body does (no force between them unless
//!    they touch).
//! 3. A car spun fast enough to tear parts off by their centripetal load: each
//!    torn piece leaves on the tangent (its centre of mass moves on a straight
//!    line in the free-falling frame once its last beam has broken), it does not
//!    orbit the body.
//!
//! Each runs with centred damping on and off and air drag on and off; "lab" is
//! the page's settings (drag and turbulence asked for, the BeamNG profile then
//! turns the drag into per-triangle aero and centred damping on).
//!
//! **Measured (2026-11-17, CPU reference; all pass, nothing to fix).**
//! 2: 50 records cut, no aero triangle spans the cut; the door's every node is
//! the same to the bit with the body yawing and moving away or not; its centre of
//! mass is 0.033 mm off x₀ + v₀·t + ½·g·t² after 1 s without air (f32), 28 mm
//! with the page's air (its own drag). 3 (10 rev/s, 0.6 s): the four hub caps
//! leave at 5 ms and sweep 77–81° about the body's centre (a point leaving a
//! circle on its tangent sweeps less than 90°), straight to 0.02 mm; the right
//! front door leaves at 65 ms (centred damping on; 39 ms off) and is then held
//! only by stretched SUPPORT beams (the shut-line stops), which carry no force in
//! tension (its centre of mass's acceleration is g to 1e-3 m/s²) and break at their
//! long bound, 100 lengths (6 to 31 m) out: the structural pieces count it
//! attached until 0.47 s. Without air it flies straight to 0.00 mm, sweeping 83°;
//! the page's air slows it along its path (≈60 m/s) and bends it 4.5 mm across.
//! The GPU steps these scenes bit for bit with this reference
//! (`engine-webgpu/tests/free_flight_gpu.rs`).
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, GRAVITY, STEP};
use engine_softbody::explicit_debris::Pieces;
use glam::{DVec3, Vec3};
use std::sync::OnceLock;

/// The sedan as the rigger exports it (JBeam, in memory; what
/// `build-web.sh` writes to `__mods/sedan_new` for the lab).
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

/// The solver settings of a run.
#[derive(Clone, Copy, Debug)]
struct Air {
    centred: bool,
    drag: bool,
}

const VARIANTS: [(&str, Air); 4] = [
    (
        "lab (drag, centred)",
        Air {
            centred: true,
            drag: true,
        },
    ),
    (
        "no drag, centred",
        Air {
            centred: true,
            drag: false,
        },
    ),
    (
        "drag, not centred",
        Air {
            centred: false,
            drag: true,
        },
    ),
    (
        "no drag, not centred",
        Air {
            centred: false,
            drag: false,
        },
    ),
];

/// The sedan at the origin (the ground 10 km below), at rest, with `air`.
fn spawn(air: Air) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: air.drag,
        turbulence: air.drag,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(sedan(), Vec3::ZERO, Vec3::ZERO, settings)
            .expect("spawn");
    s.settings.centred_damping = air.centred;
    s.ground = -10_000.0;
    s
}

/// Mass, centre of mass, momentum, angular momentum about the centre of mass and
/// kinetic energy about it, of the nodes `which` (all: None), in f64.
#[derive(Clone, Copy, Debug)]
struct Motion {
    m: f64,
    x: DVec3,
    p: DVec3,
    l: DVec3,
    ke_rel: f64,
}

fn motion(s: &ExplicitBeams, which: Option<&dyn Fn(usize) -> bool>) -> Motion {
    let o = s.origin.as_dvec3();
    let pick = |i: usize| which.map_or(true, |f| f(i));
    let (mut m, mut mx, mut p) = (0.0, DVec3::ZERO, DVec3::ZERO);
    for (i, n) in s.nodes.iter().enumerate() {
        if !pick(i) {
            continue;
        }
        let k = n.mass as f64;
        m += k;
        mx += k * (o + n.pos.as_dvec3());
        p += k * n.vel.as_dvec3();
    }
    let (x, v) = (mx / m, p / m);
    let (mut l, mut ke) = (DVec3::ZERO, 0.0);
    for (i, n) in s.nodes.iter().enumerate() {
        if !pick(i) {
            continue;
        }
        let k = n.mass as f64;
        let r = o + n.pos.as_dvec3() - x;
        let u = n.vel.as_dvec3() - v;
        l += k * r.cross(u);
        ke += 0.5 * k * u.length_squared();
    }
    Motion {
        m,
        x,
        p,
        l,
        ke_rel: ke,
    }
}

/// Adds a rigid spin `w` (rad/s) about the centre of mass to every node.
fn spin(s: &mut ExplicitBeams, w: DVec3) {
    let c = motion(s, None).x - s.origin.as_dvec3();
    for n in &mut s.nodes {
        n.vel += w.cross(n.pos.as_dvec3() - c).as_vec3();
    }
}

/// The car's long axis (the JBeam forward axis turned to the solver's frame is
/// its longest horizontal extent).
fn long_axis(s: &ExplicitBeams) -> DVec3 {
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    let d = hi - lo;
    if d.x > d.z {
        DVec3::X
    } else {
        DVec3::Z
    }
}

/// What [`fly`] measured.
#[derive(Clone, Copy, Debug)]
struct Spun {
    l0: f64,
    l_end: f64,
    /// The largest |L(t) − L(0)| / |L(0)|.
    worst_l: f64,
    ke0: f64,
    ke_half: f64,
    ke_end: f64,
    /// The largest |P(t) − P(t₁) − M·g·(t − t₁)| in units of one step's weight
    /// impulse M·|g|·h (t₁ the first step: a spawned node starts without its
    /// weight, the solver restarts each force with it after the node update).
    worst_p: f64,
    broken: usize,
}

/// The intact sedan spun at `w` (rad/s, about its centre of mass) for 1 s.
fn fly(air: Air, w: DVec3) -> Spun {
    let g = DVec3::new(0.0, GRAVITY as f64, 0.0);
    let h = STEP as f64;
    let mut s = spawn(air);
    spin(&mut s, w);
    let m0 = motion(&s, None);
    let steps = (1.0 / STEP).round() as usize;
    let (mut worst_l, mut worst_p) = (0.0f64, 0.0f64);
    let mut p1 = DVec3::ZERO;
    let mut ke_half = 0.0;
    for k in 1..=steps {
        s.step();
        let mk = motion(&s, None);
        if k == 1 {
            p1 = mk.p;
        }
        let t = (k - 1) as f64 * h;
        worst_p = worst_p.max((mk.p - p1 - m0.m * g * t).length() / (m0.m * g.length() * h));
        worst_l = worst_l.max((mk.l - m0.l).length() / m0.l.length());
        if k == steps / 2 {
            ke_half = mk.ke_rel;
        }
    }
    let m1 = motion(&s, None);
    Spun {
        l0: m0.l.length(),
        l_end: m1.l.length(),
        worst_l,
        ke0: m0.ke_rel,
        ke_half,
        ke_end: m1.ke_rel,
        worst_p,
        broken: s.beams.iter().filter(|b| b.broken).count(),
    }
}

/// The air's torque about the centre of mass (N·m) by the solver's own
/// per-triangle law (`aero_tri_force`, spec §31) at the present state.
fn aero_torque(s: &ExplicitBeams) -> DVec3 {
    use engine_softbody::explicit_beams::ExplicitKind as K;
    use engine_softbody::explicit_jbeam_ext::aero_tri_force;
    let c = motion(s, None).x - s.origin.as_dvec3();
    let mut t = DVec3::ZERO;
    for (j, b) in s.beams.iter().enumerate() {
        if b.kind != K::JbeamAeroTri {
            continue;
        }
        let ext = &s.beam_ext[s.ext_of[j] as usize - 1];
        let ids = [b.a as usize, b.b as usize, ext.index(0) as usize];
        let f = aero_tri_force(
            ids.map(|i| s.nodes[i].pos),
            ids.map(|i| s.nodes[i].vel),
            ext,
        );
        for i in ids {
            t += (s.nodes[i].pos.as_dvec3() - c).cross(f.as_dvec3());
        }
    }
    t
}

/// Where a free-flying car's spin goes (the lab's report: a car rolling in the
/// air loses its spin). The car flies at 25 m/s and rolls at 1 rev/s (the
/// mountain's speeds). Without air its angular momentum about the centre of mass
/// is constant; with the page's air every bit of the change is the air's torque
/// by the solver's own aero law, evaluated from the state step by step and
/// integrated (nothing else, such as internal damping, acts on the rigid
/// rotation).
///
/// Measured (2026-11-17, CPU reference): with the page's air |L| 3,246.8 →
/// 3,203.5 kg·m²/s in 1 s (−1.33 %); the air's torque integrated is 48.6 and
/// differs from ΔL by 0.04 kg·m²/s (0.1 %). Without air ΔL is 0.02 kg·m²/s
/// (f32). In the GPU World on the mountain (`examples/door_orbit`,
/// `tools/free-flight/spin_ledger.py`) the body's clean airborne spans lose
/// 3–9 % a second, and the same law's torque accounts for it to 0.1–1 % of |L|
/// a second: the air, at the speeds and turn rates of a tumbling car.
#[test]
fn a_flying_car_loses_spin_only_to_the_air() {
    for (name, air) in VARIANTS {
        if !air.centred {
            continue;
        }
        let mut s = spawn(air);
        let axis = long_axis(&s);
        let fwd = axis.as_vec3();
        for n in &mut s.nodes {
            n.vel += fwd * 25.0;
        }
        spin(&mut s, axis * std::f64::consts::TAU);
        let l0 = motion(&s, None).l;
        let h = STEP as f64;
        let mut by_air = DVec3::ZERO;
        let steps = (1.0 / STEP).round() as usize;
        for _ in 0..steps {
            // The forces of a step act at its start state (the force restart
            // after the node update; the aero records in the beam pass).
            s.step();
            by_air += aero_torque(&s) * h;
        }
        let l1 = motion(&s, None).l;
        let dl = l1 - l0;
        eprintln!(
            "{name}, 25 m/s rolling at 1 rev/s: |L| {:.1} → {:.1} kg·m²/s ({:+.2} % in 1 s); the air's torque integrated {:.1}, |ΔL − its| {:.2} kg·m²/s",
            l0.length(),
            l1.length(),
            100.0 * (l1.length() / l0.length() - 1.0),
            by_air.length(),
            (dl - by_air).length()
        );
        if air.drag {
            assert!(l1.length() < l0.length(), "{name}: the air added spin");
            assert!(
                (dl - by_air).length() < 0.05 * dl.length(),
                "{name}: ΔL {dl:?} is not the air's torque {by_air:?}"
            );
        } else {
            assert!(
                (dl.length() / l0.length()) < 1e-4,
                "{name}: spin lost without air: {dl:?}"
            );
        }
    }
}

/// 1. The intact car rolling at 2 rev/s about its long axis, and tumbling at
/// 2 rev/s about a skew axis, for 1 s.
///
/// Measured (2026-11-17, CPU reference): without air |L| moves by at most
/// 3.5e-5 of itself (roll) and 1.2e-5 (tumble), f32 rounding; the kinetic
/// energy about the centre of mass falls 0.67 % (roll) and 1.6 % (tumble) in
/// the second, the internal damping of the centrifugal stretch and, in the
/// tumble, of a body turning about an axis that is not a principal one (a
/// deformable body dissipates there: physical). The page's air (per-triangle
/// aero) takes 1.2 % (roll) and 1.5 % (tumble) of |L| in 1 s. With centred
/// damping off (not the lab's: JBeam content always runs it) the spin breaks
/// 36 beams and the energy about the centre rises 31 % (roll): the damped
/// explicit step past its limit (spec §39), which is what centred damping is
/// for.
#[test]
fn a_free_flying_car_keeps_its_momentum_and_spin() {
    for (name, air) in VARIANTS {
        let probe = spawn(air);
        for (axis_name, axis) in [
            ("roll", long_axis(&probe)),
            ("tumble", DVec3::new(0.3, 1.0, 0.5)),
        ] {
            let r = fly(air, axis.normalize() * (2.0 * std::f64::consts::TAU));
            eprintln!(
                "{name}, {axis_name}: |L| {:.1} → {:.1} kg·m²/s ({:+.3} %), worst |ΔL|/|L| {:.2e}; KE about the centre {:.0} → {:.0} (½ s) → {:.0} J ({:+.3} %); momentum off M·g·t by at most {:.2} of a step's weight; {} beams broken",
                r.l0,
                r.l_end,
                100.0 * (r.l_end / r.l0 - 1.0),
                r.worst_l,
                r.ke0,
                r.ke_half,
                r.ke_end,
                100.0 * (r.ke_end / r.ke0 - 1.0),
                r.worst_p,
                r.broken,
            );
            // Momentum: only gravity changes it (and the air, which acts on the
            // centre of mass too: with drag the bound is the air's, below).
            if !air.drag {
                assert!(
                    r.worst_p < 0.05,
                    "{name}, {axis_name}: momentum off M·g·t by {:.3} of a step's weight",
                    r.worst_p
                );
                assert!(
                    r.worst_l < 1e-4,
                    "{name}, {axis_name}: angular momentum moved by {:.2e} of itself",
                    r.worst_l
                );
            } else {
                // The air only takes spin away, and slowly at 2 rev/s.
                assert!(r.l_end < r.l0, "{name}, {axis_name}: the air added spin");
                assert!(
                    r.l_end > 0.97 * r.l0,
                    "{name}, {axis_name}: the air took {:.1} % of the spin in 1 s",
                    100.0 * (1.0 - r.l_end / r.l0)
                );
            }
            if air.centred {
                assert_eq!(r.broken, 0, "{name}, {axis_name}: the spin broke beams");
                // The internal damping does not damp the rigid rotation (only the
                // deformation's ring-down): at most 2 % of the energy about the
                // centre without air, and never a rise.
                let floor = if air.drag { 0.94 } else { 0.98 };
                assert!(
                    r.ke_end > floor * r.ke0 && r.ke_end <= r.ke0 * 1.0001,
                    "{name}, {axis_name}: energy about the centre {:.0} → {:.0} J",
                    r.ke0,
                    r.ke_end
                );
            }
        }
    }
}

/// The nodes of part `name` that are in no other part.
fn own_nodes(name: &str) -> Vec<bool> {
    let d = sedan();
    let mut own = vec![false; d.nodes.len()];
    for &i in &d.part(name).expect("the part").nodes {
        own[i as usize] = true;
    }
    for p in &d.parts {
        if p.name != name {
            for &i in &p.nodes {
                own[i as usize] = false;
            }
        }
    }
    own
}

/// Breaks every record joining a node of `set` to one outside it; returns how
/// many, and the aero triangles that span the cut (records that never break).
fn cut(s: &mut ExplicitBeams, set: &[bool]) -> (usize, usize) {
    use engine_softbody::explicit_beams::ExplicitKind as K;
    let (mut n, mut aero) = (0, 0);
    for j in 0..s.beams.len() {
        let b = s.beams[j];
        let ends = [b.a as usize, b.b as usize];
        let third = match b.kind {
            K::JbeamAeroTri | K::JbeamPressureTri => {
                let x = s.ext_of[j] as usize - 1;
                Some(s.beam_ext[x].index(0) as usize)
            }
            _ => None,
        };
        let inside = ends.iter().chain(third.iter()).filter(|&&i| set[i]).count();
        let all = 2 + third.is_some() as usize;
        if inside == 0 || inside == all {
            continue;
        }
        if b.kind == K::JbeamAeroTri {
            aero += 1;
            continue;
        }
        if !b.broken {
            s.beams[j].broken = true;
            n += 1;
        }
    }
    (n, aero)
}

/// The pieces joined by structural records (beams, couplers, torsion: every
/// record but the aero and pressure triangles' bookkeeping, which never break).
fn pieces(s: &ExplicitBeams) -> Pieces {
    use engine_softbody::explicit_beams::ExplicitKind as K;
    let masses: Vec<f32> = s.nodes.iter().map(|n| n.mass).collect();
    Pieces::from_beams(
        &masses,
        s.beams.iter().map(|b| {
            let skip = matches!(
                b.kind,
                K::JbeamAeroTri | K::JbeamAeroTri2 | K::JbeamPressureTri | K::JbeamPressureTri2
            );
            (b.a, b.b, b.broken || skip)
        }),
    )
}

/// The front left door cut free (every record to the rest broken), moved 0.25 m
/// out and thrown out sideways at 3 m/s, tumbling at 3 rad/s; the body given
/// `away` m/s the other way and a yaw of `body_w` rad/s. Returns the door's node positions every step (world, f64),
/// the cut's size, the aero triangles spanning it, and the closest the door's
/// nodes came to the body's.
fn thrown_door(
    air: Air,
    away: f64,
    body_w: f64,
    steps: usize,
) -> (Vec<Vec<DVec3>>, usize, usize, f64) {
    let mut s = spawn(air);
    let door = own_nodes("sedan_door_FL");
    let (n, aero) = cut(&mut s, &door);
    let p = pieces(&s);
    let first = door.iter().position(|&d| d).unwrap();
    for (i, &d) in door.iter().enumerate() {
        assert_eq!(
            p.piece[i] == p.piece[first],
            d,
            "node {i}: the door is one piece of its own after the cut"
        );
    }
    let is_door = |i: usize| door[i];
    let md = motion(&s, Some(&is_door));
    let mb = motion(&s, Some(&|i: usize| !door[i]));
    let out = (md.x - mb.x).with_y(0.0).normalize();
    for (i, nd) in s.nodes.iter_mut().enumerate() {
        if door[i] {
            nd.pos += (out * 0.25).as_vec3();
        }
    }
    let md = motion(&s, Some(&is_door));
    let wdoor = DVec3::new(0.4, 1.0, 0.2).normalize() * 3.0;
    let o = s.origin.as_dvec3();
    for (i, nd) in s.nodes.iter_mut().enumerate() {
        let r = o + nd.pos.as_dvec3();
        if door[i] {
            nd.vel += (out * 3.0 + wdoor.cross(r - md.x)).as_vec3();
        } else {
            nd.vel += (-out * away + (DVec3::Y * body_w).cross(r - mb.x)).as_vec3();
        }
    }
    let mut path = Vec::with_capacity(steps + 1);
    let mut closest = f64::MAX;
    let grab = |s: &ExplicitBeams| -> Vec<DVec3> {
        let o = s.origin.as_dvec3();
        (0..s.nodes.len())
            .filter(|&i| door[i])
            .map(|i| o + s.nodes[i].pos.as_dvec3())
            .collect()
    };
    path.push(grab(&s));
    for _ in 0..steps {
        s.step();
        path.push(grab(&s));
        for i in (0..s.nodes.len()).filter(|&i| door[i]) {
            for k in (0..s.nodes.len()).filter(|&k| !door[k]) {
                closest = closest.min(s.nodes[i].pos.distance(s.nodes[k].pos) as f64);
            }
        }
    }
    (path, n, aero, closest)
}

/// 2. A part with every beam to the body broken moves ballistically and does not
/// feel the body.
#[test]
fn a_detached_door_flies_ballistically_and_does_not_feel_the_body() {
    let d = sedan();
    let door = own_nodes("sedan_door_FL");
    let masses: Vec<f64> = d.nodes.iter().map(|n| n.mass as f64).collect();
    let md: f64 = (0..d.nodes.len())
        .filter(|&i| door[i])
        .map(|i| masses[i])
        .sum();
    let wd: Vec<f64> = (0..d.nodes.len())
        .filter(|&i| door[i])
        .map(|i| masses[i] / md)
        .collect();
    let com = |p: &[DVec3]| p.iter().zip(&wd).map(|(x, w)| *x * *w).sum::<DVec3>();
    let g = DVec3::new(0.0, GRAVITY as f64, 0.0);
    let steps = (1.0 / STEP).round() as usize;
    let h = STEP as f64;
    for (name, air) in VARIANTS {
        let (a, n, aero, close_a) = thrown_door(air, 0.0, 0.0, steps);
        let (b, _, _, close_b) = thrown_door(air, 3.0, 1.0, steps);
        // Ballistic: x(t) = x₀ + v₀·t + ½·g·t² (the first step's gravity starts a
        // step late: the solver restarts each node's force with its weight after
        // the node update, and a spawned node starts with none).
        let x0 = com(&a[0]);
        let v0 = (com(&a[1]) - x0) / h;
        let mut worst_ballistic = 0.0f64;
        for (k, p) in a.iter().enumerate().skip(1) {
            let t = (k - 1) as f64 * h;
            let want = com(&a[1]) + v0 * t + 0.5 * g * t * t + 0.5 * g * h * t;
            worst_ballistic = worst_ballistic.max((com(p) - want).length());
        }
        let mut worst_ab = 0.0f64;
        let mut first_ab = None;
        for (k, (pa, pb)) in a.iter().zip(&b).enumerate() {
            let e = pa
                .iter()
                .zip(pb)
                .map(|(x, y)| (*x - *y).length())
                .fold(0.0, f64::max);
            if e > 0.0 && first_ab.is_none() {
                first_ab = Some(k);
            }
            worst_ab = worst_ab.max(e);
        }
        eprintln!(
            "{name}: {n} records cut, {aero} aero triangles span the cut; door CoM off the ballistic path by at most {:.3} mm over 1 s; door's nodes with the body spinning vs not: at most {:.3} mm apart (first differ at step {first_ab:?}); closest approach {close_a:.2} / {close_b:.2} m",
            worst_ballistic * 1e3,
            worst_ab * 1e3,
        );
        assert!(
            close_a > 0.1 && close_b > 0.1,
            "{name}: the door touched the body"
        );
        // No force between them: the door's every node is the same to the bit
        // whatever the body does (air drag is per triangle, and no triangle spans
        // the cut).
        assert_eq!(aero, 0, "{name}: an aero triangle spans the cut");
        assert_eq!(
            first_ab,
            None,
            "{name}: the door felt the body ({:.3} mm)",
            worst_ab * 1e3
        );
        if air.drag {
            // The door's own air drag (≈5 m/s, 29.5 kg): centimetres in 1 s.
            assert!(
                worst_ballistic < 0.1,
                "{name}: {:.1} mm off the ballistic path",
                worst_ballistic * 1e3
            );
        } else {
            // f32 rounding of positions near the origin.
            assert!(
                worst_ballistic < 1e-4,
                "{name}: {:.3} mm off the ballistic path",
                worst_ballistic * 1e3
            );
        }
    }
}

/// One torn piece's flight.
#[derive(Debug)]
struct Flight {
    nodes: usize,
    kg: f64,
    /// The step its last record to the rest broke (the structural pieces), and the
    /// step its last record that can pull broke (every record but SUPPORT beams,
    /// which push only: a stretched one carries no force, and breaks only at its
    /// long bound).
    since: usize,
    free: usize,
    /// Its centre of mass off uniform motion on a straight line in the
    /// free-falling frame, from 20 ms after `free` to the end (m); and off the
    /// line alone (across it: the air slows a piece along its path, an orbit
    /// bends it).
    off_line: f64,
    bend: f64,
    /// The angle it swept about the body's centre of mass in the free-falling
    /// frame after `free` (rad), and its distance from it then and at the end (m).
    swept: f64,
    r0: f64,
    r1: f64,
    /// One of its nodes came within 10 cm of another piece's from 20 ms after
    /// `free` (contact may then push it).
    touched: bool,
}

/// Spins the sedan at `w` rad/s about its long axis for `seconds`, and follows
/// every piece that tears off.
fn spun_apart(air: Air, w: f64, seconds: f64) -> (Vec<Flight>, usize) {
    use engine_softbody::explicit_beams::ExplicitKind as K;
    let mut s = spawn(air);
    let axis = long_axis(&s);
    spin(&mut s, axis * w);
    let steps = (seconds / STEP as f64).round() as usize;
    let n = s.nodes.len();
    let h = STEP as f64;
    let g = DVec3::new(0.0, GRAVITY as f64, 0.0);
    let mass: Vec<f64> = s.nodes.iter().map(|n| n.mass as f64).collect();
    let snap = |s: &ExplicitBeams| -> Vec<DVec3> {
        let o = s.origin.as_dvec3();
        s.nodes.iter().map(|n| o + n.pos.as_dvec3()).collect()
    };
    let mut xs: Vec<Vec<DVec3>> = Vec::with_capacity(steps + 1);
    xs.push(snap(&s));
    // Per record the step it broke.
    let mut broke_at: Vec<Option<usize>> = s.beams.iter().map(|b| b.broken.then_some(0)).collect();
    for t in 1..=steps {
        s.step();
        xs.push(snap(&s));
        for (j, b) in s.beams.iter().enumerate() {
            if b.broken && broke_at[j].is_none() {
                broke_at[j] = Some(t);
            }
        }
    }
    let end = pieces(&s);
    let broken = s.beams.iter().filter(|b| b.broken).count();
    let com = |x: &[DVec3], set: &[usize]| {
        let m: f64 = set.iter().map(|&i| mass[i]).sum();
        set.iter().map(|&i| x[i] * mass[i]).sum::<DVec3>() / m
    };
    let mut out = Vec::new();
    for k in 0..end.mass.len() as u32 {
        if k == end.main {
            continue;
        }
        let inside = |i: usize| end.piece[i] == k;
        let set: Vec<usize> = (0..n).filter(|&i| inside(i)).collect();
        let (mut since, mut free) = (0, 0);
        for (j, b) in s.beams.iter().enumerate() {
            if matches!(
                b.kind,
                K::JbeamAeroTri | K::JbeamAeroTri2 | K::JbeamPressureTri | K::JbeamPressureTri2
            ) {
                continue;
            }
            if inside(b.a as usize) == inside(b.b as usize) {
                continue;
            }
            let t = broke_at[j].expect("a record across the cut is broken");
            since = since.max(t);
            if b.kind != K::JbeamSupport {
                free = free.max(t);
            }
        }
        let body: Vec<usize> = (0..n).filter(|&i| end.piece[i] == end.main).collect();
        // In the free-falling frame: x − ½·g·t².
        let fall = |t: usize| 0.5 * g * (t as f64 * h).powi(2);
        let start = (free + 40).min(steps);
        let c = |t: usize| com(&xs[t], &set) - fall(t);
        let v = (c(steps) - c(start)) / ((steps - start).max(1) as f64 * h);
        let (mut off_line, mut bend) = (0.0f64, 0.0f64);
        let along = v.normalize_or_zero();
        for t in start..=steps {
            let want = c(start) + v * ((t - start) as f64 * h);
            off_line = off_line.max((c(t) - want).length());
            let d = c(t) - c(start);
            bend = bend.max((d - along * d.dot(along)).length());
        }
        let rel = |t: usize| c(t) - (com(&xs[t], &body) - fall(t));
        let r0 = rel(free);
        let mut swept = 0.0;
        let mut last = r0;
        for t in free + 1..=steps {
            let r = rel(t);
            swept += last.angle_between(r);
            last = r;
        }
        let mut touched = false;
        'time: for t in (start..=steps).step_by(10) {
            for &i in &set {
                for j in 0..n {
                    if !inside(j) && xs[t][i].distance(xs[t][j]) < 0.1 {
                        touched = true;
                        break 'time;
                    }
                }
            }
        }
        out.push(Flight {
            nodes: set.len(),
            kg: set.iter().map(|&i| mass[i]).sum(),
            since,
            free,
            off_line,
            bend,
            swept,
            r0: r0.length(),
            r1: rel(steps).length(),
            touched,
        });
    }
    (out, broken)
}

/// 3. Spun fast enough to tear parts off (10 rev/s about the long axis; 6 rev/s
/// tears nothing with centred damping), each piece leaves on its tangent.
///
/// Measured: the file's doc comment.
#[test]
fn parts_torn_off_by_a_spin_leave_on_the_tangent() {
    for (name, air) in VARIANTS {
        let (flights, broken) = spun_apart(air, 10.0 * std::f64::consts::TAU, 0.6);
        eprintln!(
            "{name}, 10 rev/s: {broken} records broken, {} pieces off",
            flights.len()
        );
        assert!(!flights.is_empty(), "{name}: nothing tore off");
        for f in &flights {
            eprintln!(
                "  {} nodes {:.1} kg, free at {:.3} s (last record {:.3} s): off uniform straight motion by {:.2} mm, across its line {:.2} mm; swept {:.0}° about the body, {:.2} → {:.2} m from it{}",
                f.nodes,
                f.kg,
                f.free as f64 * STEP as f64,
                f.since as f64 * STEP as f64,
                f.off_line * 1e3,
                f.bend * 1e3,
                f.swept.to_degrees(),
                f.r0,
                f.r1,
                if f.touched { " (touched another piece)" } else { "" }
            );
            assert!(!f.touched, "{name}: a torn piece touched another");
            assert!(
                f.r1 > f.r0,
                "{name}: a torn piece came back toward the body"
            );
            assert!(
                f.swept < 90f64.to_radians(),
                "{name}: a torn piece swept {:.0}° about the body",
                f.swept.to_degrees()
            );
            if air.drag {
                // The air slows it along its path; it does not bend it round.
                assert!(
                    f.bend < 0.05,
                    "{name}: a torn piece {:.2} mm across its straight line",
                    f.bend * 1e3
                );
            } else {
                assert!(
                    f.off_line < 1e-4,
                    "{name}: a torn piece {:.3} mm off uniform straight motion",
                    f.off_line * 1e3
                );
            }
        }
    }
}
