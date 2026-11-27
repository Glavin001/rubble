//! The crash box's lattice cell as a closed box ([`engine_vehiclegen::physics::FrontRules::crash_box_cell`],
//! `engine/docs/generated-vehicles-lessons.md` §26).
//!
//! The definitions place each crash box's shell nodes at the top and bottom of its section, both
//! on the rail's lane (`structure::unibody`: `sh_a1_u*`, `sh_a1_d*` at its front, `sh_a2_*` at its
//! rear): four nodes in one vertical plane, a cell with no depth across the car, which swings
//! about its rear nodes instead of folding (§25.2). With the switch on, each end has a node at
//! every corner of the section: the existing nodes move to the outboard corners and an inboard
//! pair is added (`sh_a1_ui*`, `sh_a1_di*`, `sh_a2_ui*`, `sh_a2_di*`), each end's two nodes at
//! a corner sharing the node's group mass (the car's mass and the group's do not change). The
//! cell then has, a side: four corner chords; each end face closed (four edges and both
//! diagonals); one diagonal on each of the four long faces (shear in every face: bending about
//! both axes and torsion).
//!
//! **What it matches** (`engine/docs/section-matching.md` §2, rule 3 "by role"): a crash box
//! crushes along its axis, so the cell's axial yield, the four chords plus the long faces'
//! diagonals in their axial direction, is the section's mean crush force
//! ([`engine_vehiclegen::structure::Member::crush_n`], Abramowicz–Jones) × the character's front
//! multiple (× the crash-rate factor with rule 6 in `real_crash`). The diagonals keep the zone's
//! yield (`shell::Zone::CrashBox`, as the planar cell's did) and the chords take the rest
//! equally. Its bending then follows from corner chords at that force: about a horizontal axis
//! 2·F_chord·h plus the diagonals' share, well under the section's σ_y·Z, which a crash box does
//! not need to match (it is not loaded in bending; §26 has the numbers).

use super::shell::{self, Class, Lattice, ShellBeam, Zone};
use engine_vehiclegen::physics::Physics;
use engine_vehiclegen::structure::{Section, ShellNode, Structure};
use glam::DVec3;

/// The corners at an end: (tag of the existing node, tag of the added one).
const CORNERS: [(&str, &str); 2] = [("u", "ui"), ("d", "di")];

/// The body-in-white with the crash boxes' corner nodes (unchanged when the switch is off or
/// the car has no crash box).
pub fn structure(st: &Structure, ph: &Physics) -> Structure {
    let mut out = st.clone();
    if !ph.front_rules.crash_box_cell {
        return out;
    }
    let Some(mi) = st.members.iter().position(|m| m.kind == "crash_box") else { return out };
    let w = match st.members[mi].section {
        Section::Box { w, .. } | Section::Hat { w, .. } => w,
        Section::Panel => return out,
    };
    let mut added = Vec::new();
    for n in out.nodes.iter_mut() {
        let Some(rest) = n.name.strip_prefix("sh_a1_").or_else(|| n.name.strip_prefix("sh_a2_")) else { continue };
        let (corner, side) = rest.split_at(rest.len() - 1);
        let Some(&(_, inner)) = CORNERS.iter().find(|c| c.0 == corner) else { continue };
        let out_x = n.pos.x.signum();
        let centre = n.pos;
        n.pos = centre + DVec3::X * (out_x * 0.5 * w);
        n.mass *= 0.5;
        added.push(ShellNode {
            name: format!("{}{inner}{side}", &n.name[..6]),
            group: n.group,
            pos: centre - DVec3::X * (out_x * 0.5 * w),
            mass: n.mass,
            member: n.member,
        });
    }
    out.nodes.extend(added);
    out
}

/// The cell's beams on the shell's lattice (after the character's multiples): every beam
/// between two of a side's eight cell nodes is replaced by the cell's own, and the springs are
/// fitted to the node budget again. The box's rear face is bolted to the rail: the rail's first
/// bay runs from all four of its corners to the rail's next station (`sh_a4_*`), a pyramid that
/// carries the rail's mean crush force (without it the closed box turns about its rear end on
/// the rail's single chord, measured). `rate`, `rail_rate`: the crash-rate factors (1 unless
/// rule 6 applies).
pub fn lattice(st: &Structure, l: &mut Lattice, ph: &Physics, rate: f64, rail_rate: f64) {
    if !ph.front_rules.crash_box_cell {
        return;
    }
    let Some(m) = st.members.iter().find(|m| m.kind == "crash_box") else { return };
    let id = |n: &str| st.nodes.iter().position(|x| x.name == n);
    let feel = f64::from(ph.crash.front_crush);
    let (k, c, zone_deform, lim) = Zone::CrashBox.values();
    let tie = zone_deform * feel;
    let target = m.crush_n() * feel * rate;
    for s in ["l", "r"] {
        // [end][uo, do, ui, di]
        let at = |e: &str| -> Option<[usize; 4]> { Some([id(&format!("sh_{e}_u{s}"))?, id(&format!("sh_{e}_d{s}"))?, id(&format!("sh_{e}_ui{s}"))?, id(&format!("sh_{e}_di{s}"))?]) };
        let (Some(f), Some(r)) = (at("a1"), at("a2")) else { continue };
        let cell: Vec<usize> = f.iter().chain(&r).copied().collect();
        l.beams.retain(|b| !(cell.contains(&b.a) && cell.contains(&b.b)));
        let mut add = |a: usize, b: usize, class: Class, deform: f64| l.beams.push(ShellBeam { a, b, class, zone: Zone::CrashBox, k, c, deform, limit_expansion: lim });
        for e in [f, r] {
            let [uo, dn, ui, di] = e;
            for (a, b) in [(uo, dn), (ui, di), (uo, ui), (dn, di)] {
                add(a, b, Class::Edge, tie);
            }
            add(uo, di, Class::Brace, tie);
            add(ui, dn, Class::Brace, tie);
        }
        // one diagonal a long face: outboard, inboard, top, bottom
        let faces = [(f[0], r[1]), (f[2], r[3]), (f[0], r[2]), (f[1], r[3])];
        let cos = |a: usize, b: usize| {
            let d = st.nodes[a].pos - st.nodes[b].pos;
            d.z.abs() / d.length()
        };
        // (`CELL_DIAG_SHEAR=1`, a measurement: each long face's diagonal at its wall's shear
        // yield, τ_y·(wall width)·t / sin θ, τ_y = σ_y/√3: the section's shear, not its crush)
        let shear = std::env::var("CELL_DIAG_SHEAR").is_ok_and(|v| v == "1");
        let diag = |a: usize, b: usize| {
            if !shear {
                return tie;
            }
            let d = st.nodes[a].pos - st.nodes[b].pos;
            let sin = (1.0 - (d.z.abs() / d.length()).powi(2)).sqrt();
            let wall = (d - DVec3::Z * d.z).length();
            m.grade.yield_pa() / 3f64.sqrt() * wall * m.gauge * feel / sin
        };
        let diag_axial: f64 = faces.iter().map(|&(a, b)| diag(a, b) * cos(a, b)).sum();
        for (a, b) in faces {
            add(a, b, Class::Brace, diag(a, b));
        }
        // the chords take what the diagonals leave of the section's force (never under a
        // quarter of it: a diagonal yields first)
        let chord = ((target - diag_axial) / 4.0).max(0.25 * target / 4.0);
        for x in 0..4 {
            add(f[x], r[x], Class::Edge, chord);
        }
        // the rail's first bay: the box's rear corners to the rail station behind it
        let (Some(rail), Some(a4)) = (st.members.iter().find(|m| m.kind == "front_rail"), id(&format!("sh_a4_{s}"))) else { continue };
        l.beams.retain(|b| !((r.contains(&b.a) && b.b == a4) || (r.contains(&b.b) && b.a == a4)));
        let (rk, rc, _, rlim) = Zone::FrontRail.values();
        let rcos: f64 = r.iter().map(|&a| cos(a, a4)).sum();
        let per = rail.crush_n() * feel * rail_rate / rcos;
        for &a in &r {
            l.beams.push(ShellBeam { a, b: a4, class: Class::Edge, zone: Zone::FrontRail, k: rk, c: rc, deform: per, limit_expansion: rlim });
        }
        if std::env::var("FRONT_RULES_LOG").is_ok() {
            eprintln!("crash box cell {s}: rail's first bay 4 × {:.1} kN", per / 1e3);
        }
        if std::env::var("FRONT_RULES_LOG").is_ok() {
            eprintln!("crash box cell {s}: target {:.1} kN, diagonals' axial {:.1} kN, chords 4 × {:.1} kN", target / 1e3, diag_axial / 1e3, chord / 1e3);
        }
    }
    shell::budget(st, &mut l.beams);
}
