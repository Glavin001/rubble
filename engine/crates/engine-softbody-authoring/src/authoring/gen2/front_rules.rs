//! The front crash structure's rules on the rigged car ([`engine_vehiclegen::physics::FrontRules`],
//! each a switch of the design's physics block). With every switch off nothing here changes a
//! beam, so the car is byte for byte the one before the rules. Each rule finds its members by
//! the definitions' node names (the shell's groups `sh_a1_*` crash box front, `sh_a2_*` crash
//! box rear, `sh_b1_*` firewall) and by part; a car without them is untouched.
//!
//! Derivations and measurements: `engine/docs/generated-vehicles-lessons.md` §25; the evidence
//! for each: `engine/docs/sedan-front-audit.md`.

use super::build::{Part, Role};
use super::rig::{Rig, RigInput};
use super::shell::{self, Class, Lattice, Zone};
use engine_vehiclegen::front_rules as vr;
use engine_vehiclegen::mechanical::MechPart;
use engine_vehiclegen::physics::{FrontRules, Physics, CRASH_BOX_BAY_BRACE_SHARE, HOOD_FOLD_ROW_M};
use engine_vehiclegen::structure::Structure;
use engine_vehiclegen::thin_wall::crush::EFFECTIVE_CRUSH_FRACTION;
use glam::Vec3;

/// The mechanical parts as the rules define them (rule 1).
pub fn mechanical(rules: &FrontRules, parts: &[MechPart]) -> Vec<MechPart> {
    let mut out = parts.to_vec();
    if rules.subframe_mounts_off_crash_box {
        vr::subframe_mounts_off_crash_box(&mut out);
    }
    out
}

/// The shell's lattice in the design's crash character ([`shell::lattice_with`]) with rules 6
/// and 2 on it.
pub fn lattice(st: &Structure, ph: &Physics) -> Lattice {
    let mut l = shell::lattice_with(st, &ph.crash);
    let rules = &ph.front_rules;
    let group = |i: usize| st.nodes[i].group;
    let along = |b: &shell::ShellBeam| {
        let d = (st.nodes[b.a].pos - st.nodes[b.b].pos).normalize();
        d.z.abs()
    };
    // The crush members' chords, as `shell` marks them: lengthwise edges inside the member's
    // groups (crash box A1–A2, front rail A2–A4–A5).
    let chord = |b: &shell::ShellBeam, gs: &[&str]| b.class == Class::Edge && along(b) > 0.707 && gs.contains(&group(b.a)) && gs.contains(&group(b.b)) && st.nodes[b.a].pos.x * st.nodes[b.b].pos.x > 0.0;
    // Rule 6: in the real_crash character (front multiple 1) the chords at crash rate.
    if rules.crush_from_section && ph.crash.front_crush == 1.0 {
        for (kind, gs) in [("crash_box", &["A1", "A2"][..]), ("front_rail", &["A2", "A4", "A5"][..])] {
            let Some(m) = st.members.iter().find(|m| m.kind == kind) else { continue };
            let f = vr::crash_rate_factor(m);
            for b in l.beams.iter_mut().filter(|b| chord(b, gs) && matches!(b.zone, Zone::CrashBox | Zone::FrontRail)) {
                b.deform *= f;
            }
        }
    }
    // Rule 2: the beams that cross the crash box's bay without being the box (one end on the
    // box's front nodes, the other behind its rear nodes) carry, together and in their axial
    // direction, a share of the box's own chords.
    if rules.crash_box_bay_alone {
        for side in [1.0f64, -1.0] {
            let on = |i: usize| st.nodes[i].pos.x * side > 0.0;
            let box_chords: f64 = l.beams.iter().filter(|b| on(b.a) && chord(b, &["A1", "A2"])).map(|b| b.deform).sum();
            let across: Vec<usize> = (0..l.beams.len())
                .filter(|&j| {
                    let b = &l.beams[j];
                    let (ga, gb) = (group(b.a), group(b.b));
                    let front = |g: &str| g == "A1";
                    let boxed = |g: &str| g == "A1" || g == "A2";
                    ((front(ga) && !boxed(gb) && on(b.a)) || (front(gb) && !boxed(ga) && on(b.b))) && b.deform.is_finite()
                })
                .collect();
            let cos: f64 = across.iter().map(|&j| along(&l.beams[j])).sum();
            if across.is_empty() || box_chords <= 0.0 || cos <= 0.0 {
                continue;
            }
            // each the same axial force: F_axial = share × box / n, beam force = F_axial / cos
            let per = CRASH_BOX_BAY_BRACE_SHARE * box_chords / cos;
            for j in across {
                l.beams[j].deform = l.beams[j].deform.min(per);
            }
        }
    }
    l
}

/// The hood's node rows (glTF z, rear to front) with the fold row of rule 5.
pub fn hood_rows(mut rows: Vec<f32>, ph: &Physics) -> Vec<f32> {
    if ph.front_rules.hood_fold_row && rows.len() >= 2 {
        let front = rows[rows.len() - 1];
        let z = front - HOOD_FOLD_ROW_M;
        // (only where it falls between the last two rows, clear of both)
        if z > rows[rows.len() - 2] + 0.05 {
            rows.insert(rows.len() - 1, z);
        }
    }
    rows
}

/// Rules 3, 4 and 5 (the panels' yields) on the rigged car, after every yield is set.
pub fn apply(r: &mut Rig, inp: &dyn RigInput) {
    let ph = inp.physics();
    let rules = &ph.front_rules;
    if rules.is_off() {
        return;
    }
    let (car, member) = (&mut r.car, &r.member);
    let name = |car: &super::build::Car, i: usize| car.nodes[i].name.clone();
    // Rule 3: the crash box's chords crush flat to their stroke.
    if rules.crash_box_stroke {
        for j in 0..car.beams.len() {
            let (a, b) = (name(car, car.beams[j].a), name(car, car.beams[j].b));
            let chord = |x: &str, y: &str| x.starts_with("sh_a1_") && y.starts_with("sh_a2_") && x[6..] == y[6..];
            if (chord(&a, &b) || chord(&b, &a)) && car.fixed.contains_key(&j) {
                let tc = car.beams[j].deform;
                car.fixed_limits.insert(j, (EFFECTIVE_CRUSH_FRACTION as f32, tc));
            }
        }
    }
    // The front crush zone: ahead of the firewall (the shell's `sh_b1_*` nodes).
    let fw: Vec<f32> = (0..car.nodes.len()).filter(|&i| car.nodes[i].name.starts_with("sh_b1_")).map(|i| car.nodes[i].p.y).collect();
    let Some(fw_y) = fw.iter().copied().reduce(f32::min) else { return };
    let mid = |car: &super::build::Car, j: usize| 0.5 * (car.nodes[car.beams[j].a].p + car.nodes[car.beams[j].b].p);
    // Rule 4: a SUPPORT stop or seal in the front crush zone yields at the force its weaker end
    // can bear: the largest yield among the panel or frame beams at that node (never more than
    // the metal under it).
    if rules.yielding_stops {
        let bears = |car: &super::build::Car, i: usize, skip: usize| {
            car.beams
                .iter()
                .enumerate()
                .filter(|&(k, b)| k != skip && (b.a == i || b.b == i) && b.deform.is_finite() && !b.role.spec(b.part).support && !matches!(b.role, Role::Stop | Role::Bolt | Role::Clip | Role::Hinge) && !car.fixed.get(&k).is_some_and(|f| f.kind != super::build::FixedKind::Normal))
                .map(|(_, b)| b.deform)
                .fold(0.0f32, f32::max)
        };
        let stops: Vec<usize> = (0..car.beams.len())
            .filter(|&j| {
                let b = &car.beams[j];
                matches!(b.role, Role::EdgeStop | Role::Bearing) && !car.fixed.contains_key(&j) && !b.deform.is_finite() && mid(car, j).y < fw_y
            })
            .collect();
        for j in stops {
            let (a, b) = (car.beams[j].a, car.beams[j].b);
            let y = bears(car, a, j).min(bears(car, b, j));
            if y > 0.0 {
                car.beams[j].deform = y;
            }
        }
    }
    // Rule 5: the hood's and fenders' lengthwise beams at their panel's buckling force.
    if rules.hood_fold_row {
        for (part, sheet, across) in [(Part::Hood, vr::HOOD_SHEET, 0usize), (Part::FenderL, vr::FENDER_SHEET, 2), (Part::FenderR, vr::FENDER_SHEET, 2)] {
            panel_yield(car, member, part, sheet, across, ph.crash.panel_yield);
        }
    }
}

/// The beams of `part` that a lengthwise load crushes (|cos| to the car's axis ≥ 0.5, panel
/// and frame roles, finite yield) scaled down so the largest cut across the panel carries no
/// more than its sheet's buckling force ([`vr::panel_buckling_force`]) times the character's
/// panel multiple. The strips are the gaps between the panel's node lines (its skin's edges
/// along the car, |cos| ≥ 0.9, both ends skin nodes) where the cut crosses them, in order along
/// `across` (0: x, across a hood; 2: z, up a fender).
fn panel_yield(car: &mut super::build::Car, member: &[Option<Role>], part: Part, sheet: (f64, f64, f64, f64), across: usize, feel: f32) {
    let sheet_role = |r: Role| matches!(r, Role::Sheet | Role::Bend | Role::DoorFrame | Role::EdgeTie);
    let beams: Vec<usize> = (0..car.beams.len())
        .filter(|&j| {
            let b = &car.beams[j];
            let d = car.nodes[b.a].p - car.nodes[b.b].p;
            b.part == part && sheet_role(b.role) && b.deform.is_finite() && (d.y.abs() / d.length().max(1e-6)) >= 0.5
        })
        .collect();
    if beams.is_empty() {
        return;
    }
    let ys: Vec<f32> = beams.iter().flat_map(|&j| [car.nodes[car.beams[j].a].p.y, car.nodes[car.beams[j].b].p.y]).collect();
    let (y0, y1) = (ys.iter().copied().fold(f32::MAX, f32::min), ys.iter().copied().fold(f32::MIN, f32::max));
    let (mut worst, mut strips_at) = (0.0f64, Vec::new());
    for k in 1..40 {
        let y = y0 + (y1 - y0) * k as f32 / 40.0;
        let mut cut = 0.0f64;
        let mut pts: Vec<Vec3> = Vec::new();
        for &j in &beams {
            let (pa, pb) = (car.nodes[car.beams[j].a].p, car.nodes[car.beams[j].b].p);
            if (pa.y - y) * (pb.y - y) >= 0.0 {
                continue;
            }
            let cos = f64::from((pa.y - pb.y).abs() / pa.distance(pb));
            cut += f64::from(car.beams[j].deform) * cos;
            let skin = |i: usize| member.get(i).copied().flatten() == Some(Role::Sheet);
            if cos >= 0.9 && skin(car.beams[j].a) && skin(car.beams[j].b) {
                pts.push(pa + (pb - pa) * ((y - pa.y) / (pb.y - pa.y)));
            }
        }
        if cut > worst {
            worst = cut;
            strips_at = pts;
        }
    }
    // the node lines the cut crosses (within 3 cm is one line), in order across the panel
    let coord = |p: &Vec3| [p.x, p.y, p.z][across];
    strips_at.sort_by(|a, b| coord(a).total_cmp(&coord(b)));
    let mut lines: Vec<Vec3> = Vec::new();
    for p in strips_at {
        if lines.last().is_none_or(|q| q.distance(p) > 0.03) {
            lines.push(p);
        }
    }
    let strips: Vec<f64> = lines.windows(2).map(|w| f64::from(w[0].distance(w[1]))).collect();
    let target = f64::from(feel) * vr::panel_buckling_force(&strips, sheet);
    if std::env::var("FRONT_RULES_LOG").is_ok() {
        eprintln!("front rule 5: {part:?} worst cut {:.1} kN, buckling {:.1} kN × {feel} ({} strips {:?} m)", worst / 1e3, target / 1e3 / f64::from(feel), strips.len(), strips);
    }
    if worst <= 0.0 || target >= worst {
        return;
    }
    let s = (target / worst) as f32;
    for j in beams {
        car.beams[j].deform *= s;
    }
    if std::env::var("RIG_LOG").is_ok() || std::env::var("FRONT_RULES_LOG").is_ok() {
        eprintln!("front rule 5: {part:?} cut {:.1} kN to {:.1} kN ({} strips {:?} m)", worst / 1e3, target / 1e3, strips.len(), strips);
    }
}
