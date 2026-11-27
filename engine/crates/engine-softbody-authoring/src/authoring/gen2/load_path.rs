//! The front load path ([`engine_vehiclegen::physics::FrontRules::load_path`]), designed from how
//! the 2012 Camry's front is built (`engine/docs/front-load-path.md`). With the switch off nothing
//! here changes a node or a beam. With it on, a side's front structure is one member from the
//! bumper beam to the firewall:
//!
//! - **stations**: the crash box's front and rear plates (`sh_a1_*`, `sh_a2_*`) and the rail's two
//!   stations (`sh_a4_*` at the subframe's front corners, `sh_a5_*` at its rear ones), each with a
//!   node at every corner of its section (the definitions' node stays at one corner and keeps its
//!   name; the others share its mass, so the car's mass and centre of mass do not move);
//! - **bays**: between consecutive stations a closed cell (four corner chords, one diagonal on each
//!   long face, each end face closed), its axial yield its section's mean crush force
//!   ([`engine_vehiclegen::structure::Member::crush_n`]) times the character's front multiple and
//!   the crash-rate factor where the character is `real_crash`, flat to `deformLimit`
//!   [`EFFECTIVE_CRUSH_FRACTION`] (or the solver's least rest length);
//! - **braces**: every other shell beam from a station's node across a bay carries, with the others
//!   across that bay and along the car, at most [`CRASH_BOX_BAY_BRACE_SHARE`] of the bay's force;
//! - **upper rail**: its chain (radiator support top, its two nodes, the hinge pillar's top) at its
//!   own section's force, flat;
//! - the bumper beam's bolted plate and the subframe's mounts are the definitions'
//!   ([`mechanical`]); the engine's stops engage where the block meets what they bear on ([`apply`]).

use super::build::{Part, Role};
use super::rig::{Rig, RigInput};
use super::shell::{self, Class, Lattice, ShellBeam, Zone};
use engine_vehiclegen::front_rules as vr;
use engine_vehiclegen::mechanical::MechPart;
use engine_vehiclegen::physics::{Physics, CRASH_BOX_BAY_BRACE_SHARE};
use engine_vehiclegen::structure::{Member, Section, ShellNode, Structure};
use engine_vehiclegen::thin_wall::crush::EFFECTIVE_CRUSH_FRACTION;
use glam::DVec3;

/// The stations along a side's rail, front to rear: (group tag, the corner tags in the order
/// outboard top, outboard bottom, inboard top, inboard bottom). The definitions' nodes are the
/// crash box's `u`/`d` (top and bottom on the lane) and the rail's bottom node (tag "").
const STATIONS: [(&str, [&str; 4]); 4] = [("a1", ["u", "d", "ui", "di"]), ("a2", ["u", "d", "ui", "di"]), ("a4", ["u", "", "ui", "di"]), ("a5", ["u", "", "ui", "di"])];
/// The bays: (front station, rear station, the member's kind, its zone).
const BAYS: [(usize, usize, &str, Zone); 3] = [(0, 1, "crash_box", Zone::CrashBox), (1, 2, "front_rail", Zone::FrontRail), (2, 3, "front_rail", Zone::FrontRail)];
/// The long faces' diagonals of a bay carry together at most this share of its force along the
/// member (engineering estimate: a quarter, the chords the rest; `front-load-path.md` §3.2).
pub const DIAGONAL_SHARE: f64 = 0.25;
/// A bay's long-face diagonals (front corner, rear corner): outboard, inboard, top, bottom faces.
const FACES: [(usize, usize); 4] = [(0, 1), (2, 3), (0, 2), (1, 3)];
/// The upper rail's chain, front to rear (`{s}` the side).
const UPPER: [&str; 4] = ["sh_a3_u{s}", "sh_a8_f{s}", "sh_a8_m{s}", "sh_b2_o{s}"];

/// A step of the design ([`engine_vehiclegen::physics::load_path_step`]: all of them unless the
/// measurement switch `LOAD_PATH_STEPS` lists some).
fn step(name: &str) -> bool {
    engine_vehiclegen::physics::load_path_step(name)
}

fn on(ph: &Physics) -> bool {
    ph.front_rules.load_path
}

/// A station's corner node name.
fn corner(stn: &str, tag: &str, s: &str) -> String {
    format!("sh_{stn}_{tag}{s}")
}

fn section(m: &Member) -> Option<(f64, f64)> {
    match m.section {
        Section::Box { w, h } | Section::Hat { w, h } => Some((w, h)),
        Section::Panel => None,
    }
}

/// The body-in-white with the stations' corner nodes (unchanged when the switch is off or the
/// car has no crash box and front rail).
pub fn structure(st: &Structure, ph: &Physics) -> Structure {
    let mut out = st.clone();
    if !on(ph) || !step("cells") {
        return out;
    }
    let (Some(cb), Some(rail)) = (st.members.iter().find(|m| m.kind == "crash_box"), st.members.iter().find(|m| m.kind == "front_rail")) else { return out };
    let (Some((cw, _)), Some((rw, rh))) = (section(cb), section(rail)) else { return out };
    let mut added = Vec::new();
    for n in out.nodes.iter_mut() {
        let out_x = n.pos.x.signum();
        let side = if out_x > 0.0 { "l" } else { "r" };
        // the crash box's ends: top and bottom on the lane → the outboard corners, and an
        // inboard twin each
        for stn in ["a1", "a2"] {
            for tag in ["u", "d"] {
                if n.name == corner(stn, tag, side) {
                    let c = n.pos;
                    n.pos = c + DVec3::X * (out_x * 0.5 * cw);
                    n.mass *= 0.5;
                    added.push(ShellNode { name: corner(stn, &format!("{tag}i"), side), group: n.group, pos: c - DVec3::X * (out_x * 0.5 * cw), mass: n.mass, member: n.member });
                }
            }
        }
        // the rail's stations: the bottom node on the lane → its outboard bottom corner, and the
        // other three corners
        for stn in ["a4", "a5"] {
            if n.name == corner(stn, "", side) {
                let c = n.pos;
                n.mass *= 0.25;
                n.pos = c + DVec3::X * (out_x * 0.5 * rw);
                for (tag, dx, dy) in [("u", 0.5, rh), ("ui", -0.5, rh), ("di", -0.5, 0.0)] {
                    added.push(ShellNode { name: corner(stn, tag, side), group: n.group, pos: c + DVec3::X * (out_x * dx * rw) + DVec3::Y * dy, mass: n.mass, member: n.member });
                }
            }
        }
    }
    out.nodes.extend(added);
    out
}

/// The crash-rate factor the character asks for: the section's in `real_crash` (front multiple
/// 1), 1 in a character whose multiple stands for it ([`engine_vehiclegen::physics::CrashFeel`]).
fn rate(m: &Member, ph: &Physics) -> f64 {
    if ph.crash.front_crush == 1.0 {
        vr::crash_rate_factor(m)
    } else {
        1.0
    }
}

/// The chords whose stroke is flat ([`apply`]): node-name pairs.
pub fn flat_chords(st: &Structure, ph: &Physics) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if !on(ph) {
        return out;
    }
    for s in ["l", "r"] {
        if step("cells") {
            for (f, r, _, _) in BAYS {
                let n = |k: usize, c: usize| corner(STATIONS[k].0, STATIONS[k].1[c], s);
                for c in 0..4 {
                    out.push((n(f, c), n(r, c)));
                }
                // (the long faces' diagonals carry their share of the bay's axial force: flat too)
                for (a, b) in FACES {
                    out.push((n(f, a), n(r, b)));
                }
            }
        }
        if step("upper") {
            for w in UPPER.windows(2) {
                out.push((w[0].replace("{s}", s), w[1].replace("{s}", s)));
            }
        }
    }
    let have = |n: &str| st.nodes.iter().any(|x| x.name == n);
    out.retain(|(a, b)| have(a) && have(b));
    out
}

/// The cells, the braces across the bays and the upper rail on the shell's lattice (after the
/// character's multiples and the earlier rules); the springs are fitted to the node budget again.
pub fn lattice(st: &Structure, l: &mut Lattice, ph: &Physics) {
    if !on(ph) {
        return;
    }
    let id = |n: &str| st.nodes.iter().position(|x| x.name == n);
    let member = |kind: &str| st.members.iter().find(|m| m.kind == kind);
    let feel = f64::from(ph.crash.front_crush);
    let along = |a: usize, b: usize| {
        let d = st.nodes[a].pos - st.nodes[b].pos;
        d.z.abs() / d.length()
    };
    let log = std::env::var("FRONT_RULES_LOG").is_ok();
    let mut changed = false;
    for s in ["l", "r"] {
        if !step("cells") {
            break;
        }
        let Some(stn): Option<Vec<[usize; 4]>> = STATIONS.iter().map(|(g, tags)| Some([id(&corner(g, tags[0], s))?, id(&corner(g, tags[1], s))?, id(&corner(g, tags[2], s))?, id(&corner(g, tags[3], s))?])).collect() else { continue };
        let lane: Vec<usize> = stn.iter().flatten().copied().collect();
        // every beam between two of the side's station nodes is the cells' (the chain's skip
        // beams included: the cells carry the bending they stood for)
        l.beams.retain(|b| !(lane.contains(&b.a) && lane.contains(&b.b)));
        let z = |k: usize| stn[k].iter().map(|&i| st.nodes[i].pos.z).sum::<f64>() / 4.0;
        // each station's end face, closed and held in shape: a thin-walled section keeps its
        // cross-section until its walls fold (the shape the mean crush force assumes), so the face's
        // ties yield at no less than the force of the bays it closes (a face that distorts lets the
        // bay pivot about it)
        let bay_target = |kind: &str| member(kind).map(|m| m.crush_n() * feel * rate(m, ph)).unwrap_or(0.0);
        for (k, e) in stn.iter().enumerate() {
            let zone = if k < 2 { Zone::CrashBox } else { Zone::FrontRail };
            let (kk, c, deform, lim) = zone.values();
            let tie = (deform * feel).max(if k == 0 { bay_target("crash_box") } else { bay_target("front_rail") });
            let [uo, dn, ui, di] = *e;
            for (a, b, class) in [(uo, dn, Class::Edge), (ui, di, Class::Edge), (uo, ui, Class::Edge), (dn, di, Class::Edge), (uo, di, Class::Brace), (ui, dn, Class::Brace)] {
                l.beams.push(ShellBeam { a, b, class, zone, k: kk, c, deform: tie, limit_expansion: lim });
            }
        }
        let mut bay_force = [0.0f64; 3];
        for (bi, (fi, ri, kind, zone)) in BAYS.into_iter().enumerate() {
            let Some(m) = member(kind) else { continue };
            let (f, r) = (stn[fi], stn[ri]);
            let (kk, c, deform, lim) = zone.values();
            let target = m.crush_n() * feel * rate(m, ph);
            bay_force[bi] = target;
            let faces = FACES.map(|(a, b)| (f[a], r[b]));
            // The walls' folding force goes through their steel along the member, which the
            // corner chords stand for; the long faces' diagonals stand for the walls' shear and
            // keep the zone's yield, together at most [`DIAGONAL_SHARE`] of the force along the
            // member. (With the zone's yield alone a 0.38 m rail bay's diagonals took two thirds,
            // and its chords' 9 kN left the cell a third of its bending.)
            let cos_sum: f64 = faces.iter().map(|&(a, b)| along(a, b)).sum();
            let tie = (deform * feel).min(DIAGONAL_SHARE * target / cos_sum);
            let diag_axial: f64 = faces.iter().map(|&(a, b)| tie * along(a, b)).sum();
            for (a, b) in faces {
                l.beams.push(ShellBeam { a, b, class: Class::Brace, zone, k: kk, c, deform: tie, limit_expansion: lim });
            }
            // the chords take what the diagonals leave of the section's force
            let chord = (target - diag_axial) / 4.0;
            for x in 0..4 {
                l.beams.push(ShellBeam { a: f[x], b: r[x], class: Class::Edge, zone, k: kk, c, deform: chord, limit_expansion: lim });
            }
            if log {
                eprintln!("load path {s}: bay {kind} {:.3} m, target {:.1} kN, diagonals' axial {:.1} kN, chords 4 × {:.1} kN", z(fi) - z(ri), target / 1e3, diag_axial / 1e3, chord / 1e3);
            }
        }
        // the braces across each bay (one end on a station's node, the other off the lane, the
        // beam spanning part of the bay along the car)
        for (bi, (fi, ri, _, _)) in BAYS.into_iter().enumerate() {
            let (z0, z1) = (z(ri), z(fi));
            let crosses = |b: &ShellBeam| {
                if lane.contains(&b.a) == lane.contains(&b.b) || !b.deform.is_finite() {
                    return false;
                }
                let (za, zb) = (st.nodes[b.a].pos.z, st.nodes[b.b].pos.z);
                let (lo, hi) = (za.min(zb).max(z0), za.max(zb).min(z1));
                hi - lo > 0.02
            };
            let across: Vec<usize> = (0..l.beams.len()).filter(|&j| crosses(&l.beams[j])).collect();
            let cos: f64 = across.iter().map(|&j| along(l.beams[j].a, l.beams[j].b)).sum();
            if across.is_empty() || cos <= 0.0 || bay_force[bi] <= 0.0 {
                continue;
            }
            let per = CRASH_BOX_BAY_BRACE_SHARE * bay_force[bi] / cos;
            let before: f64 = across.iter().map(|&j| l.beams[j].deform * along(l.beams[j].a, l.beams[j].b)).sum();
            for &j in &across {
                l.beams[j].deform = l.beams[j].deform.min(per);
            }
            if log {
                let after: f64 = across.iter().map(|&j| l.beams[j].deform * along(l.beams[j].a, l.beams[j].b)).sum();
                eprintln!("load path {s}: {} braces across bay {bi}: {:.1} → {:.1} kN along the car", across.len(), before / 1e3, after / 1e3);
            }
        }
        changed = true;
    }
    // the upper rail: its chain at its own section's force, its skip beams at the brace share
    if step("upper") {
        if let Some(m) = member("upper_rail") {
            let f = m.crush_n() * feel * rate(m, ph);
            for s in ["l", "r"] {
                let Some(ids): Option<Vec<usize>> = UPPER.iter().map(|n| id(&n.replace("{s}", s))).collect() else { continue };
                for b in l.beams.iter_mut() {
                    let pair = |x: usize, y: usize| (b.a == x && b.b == y) || (b.a == y && b.b == x);
                    if ids.windows(2).any(|w| pair(w[0], w[1])) {
                        b.deform = f / along(b.a, b.b).max(0.5);
                    } else if ids.windows(3).any(|w| pair(w[0], w[2])) {
                        b.deform = b.deform.min(CRASH_BOX_BAY_BRACE_SHARE * f);
                    }
                }
                if log {
                    eprintln!("load path {s}: upper rail chain at {:.1} kN", f / 1e3);
                }
            }
            changed = true;
        }
    }
    if changed {
        shell::budget(st, &mut l.beams);
    }
}

/// The definitions with the bumper beam's bolted plate and the subframe's mounts on the rail's
/// stations ([`vr::impact_beam_bolted_plate`], [`vr::subframe_on_rail_stations`]). `st` is the
/// structure with the stations' corners ([`structure`]).
pub fn mechanical(ph: &Physics, st: &Structure, mut parts: Vec<MechPart>) -> Vec<MechPart> {
    if !on(ph) {
        return parts;
    }
    if step("plate") && step("cells") {
        let side = |s: &str| -> Option<[(String, DVec3); 4]> {
            let c = |k: usize| -> Option<(String, DVec3)> {
                let n = corner("a1", STATIONS[0].1[k], s);
                let p = st.node(&n)?.pos;
                Some((n, p))
            };
            Some([c(0)?, c(1)?, c(2)?, c(3)?])
        };
        if let (Some(l), Some(r), Some(cb)) = (side("l"), side("r"), st.members.iter().find(|m| m.kind == "crash_box")) {
            let force = cb.crush_n() * f64::from(ph.crash.front_crush) * rate(cb, ph);
            vr::impact_beam_bolted_plate(&mut parts, [l, r], force);
        }
    }
    if step("mounts") && step("cells") {
        let t = STATIONS[2].1;
        let front: Vec<String> = t.iter().map(|tag| format!("sh_a4_{tag}{{s}}")).collect();
        let rear: Vec<String> = t.iter().map(|tag| format!("sh_a5_{tag}{{s}}")).collect();
        let f: Vec<&str> = front.iter().map(String::as_str).collect();
        let r: Vec<&str> = rear.iter().map(String::as_str).collect();
        vr::subframe_on_rail_stations(&mut parts, &f, &r);
    }
    parts
}

/// On the rigged car: the cells' and the upper rail's chords crush flat (`deformLimitStress` =
/// their yield, `deformLimit` [`EFFECTIVE_CRUSH_FRACTION`]); the engine's stops engage where the
/// block meets what they bear on (the gap along the car closed to the powertrain's clearance,
/// [`Physics::pt_clearance`]), not after a tenth of their length.
pub fn apply(r: &mut Rig, inp: &dyn RigInput) {
    let ph = inp.physics();
    if !on(ph) {
        return;
    }
    let Some(d) = inp.definitions() else { return };
    let chords = flat_chords(d.structure, ph);
    let car = &mut r.car;
    let name = |car: &super::build::Car, i: usize| car.nodes[i].name.clone();
    let mut n_flat = 0;
    for j in 0..car.beams.len() {
        if !car.fixed.contains_key(&j) {
            continue;
        }
        let (a, b) = (name(car, car.beams[j].a), name(car, car.beams[j].b));
        if chords.iter().any(|(x, y)| (x == &a && y == &b) || (x == &b && y == &a)) {
            let tc = car.beams[j].deform;
            car.fixed_limits.insert(j, (EFFECTIVE_CRUSH_FRACTION as f32, tc));
            n_flat += 1;
        }
    }
    let mut n_stops = 0;
    if step("engine") {
        let clr = ph.pt_clearance;
        for j in 0..car.beams.len() {
            let bm = &car.beams[j];
            if bm.role != Role::Bearing || bm.sub.as_deref() != Some("engine") {
                continue;
            }
            let (pa, pb) = (car.nodes[bm.a].p, car.nodes[bm.b].p);
            let l = pa.distance(pb);
            // (JBeam frame: y along the car) the offset across the car stays; the gap along it
            // closes to the clearance; never earlier than before, never under the solver's least
            // rest length with a margin
            let d = pa - pb;
            let engage = (d.x * d.x + d.z * d.z + clr * clr).sqrt().max(MIN_ENGAGE);
            let old = car.precomp.get(&j).copied().unwrap_or(1.0);
            let pc = (engage / l).min(old);
            car.precomp.insert(j, pc);
            n_stops += 1;
        }
    }
    if std::env::var("FRONT_RULES_LOG").is_ok() {
        eprintln!("load path: {n_flat} chords flat, {n_stops} engine stops engage at the block's gap");
    }
    let _ = Part::Body;
}

/// An engine stop's rest length is at least this (m): the solver keeps a yielding beam's rest at
/// 0.1 m or more (`MIN_REST`), so a stop resting shorter would be pushed back to it.
const MIN_ENGAGE: f32 = 0.12;
