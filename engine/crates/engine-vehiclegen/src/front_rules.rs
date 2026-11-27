//! The front crash structure's rules ([`crate::physics::FrontRules`]): the parts of each rule
//! that are statements about the definitions (which nodes a mount lands on, a member's force
//! at crash rate, a panel's buckling force). The rigger (`engine-softbody-authoring`
//! `authoring::gen2::front_rules`) applies them to the lattice. Derivations and measurements:
//! `engine/docs/generated-vehicles-lessons.md` §25; the evidence: `engine/docs/sedan-front-audit.md`.

use crate::mechanical::MechPart;
use crate::physics::CRASH_RATE_SPEED;
use crate::structure::{Member, Section};
use crate::thin_wall::crush;

/// [`crate::physics::FrontRules::subframe_mounts_off_crash_box`]: the front subframe's mount
/// groups no longer list the crash boxes' rear nodes (`sh_a2_d*`, `sh_a2_u*`). A real front
/// subframe bolts to the rails at its own front corners, under the rail station directly above
/// them (`sh_a4_*`, which the mounts already list); the beams from its front corners forward
/// and up to the crash boxes' rear crossed the rail's whole first bay at 157 kN each (277 kN
/// of the bay's ~580 kN, audit §1), so the rail could not fold. The mount keeps its beam count
/// (picked among the remaining nodes). A design without a `subframe_front` is unchanged.
pub fn subframe_mounts_off_crash_box(parts: &mut [MechPart]) {
    for p in parts.iter_mut().filter(|p| p.name == "subframe_front") {
        for m in &mut p.mounts {
            m.to.retain(|n| !(n.starts_with("sh_a2_d") || n.starts_with("sh_a2_u")));
        }
    }
}

/// [`crate::physics::FrontRules::crash_box_cell`]: the front impact beam's mounts reach all four
/// corners of each crash box's front plate (the inboard corner nodes `sh_a1_ui*`, `sh_a1_di*`
/// the closed cell adds) as a bumper beam is bolted to the box's whole flange. The mount keeps
/// its beam count.
pub fn impact_beam_on_crash_box_corners(parts: &mut [MechPart]) {
    for p in parts.iter_mut().filter(|p| p.name == "impact_beam_front") {
        for m in &mut p.mounts {
            let extra: Vec<String> = m.to.iter().filter_map(|n| n.strip_prefix("sh_a1_").map(|r| format!("sh_a1_{}i{}", &r[..1], &r[1..]))).collect();
            m.to.extend(extra);
        }
    }
}

/// [`crate::physics::FrontRules::load_path`]: the bumper beam's joint to each crash box, a
/// bolted plate. Real cars of this class bolt the beam's bracket flange to the box's front plate
/// (the 2012 Camry: three bolts a side, `engine/docs/front-load-path.md` §1.2): the joint
/// carries the box's whole axial force, shear and moment, so the beam cannot slide off the box's
/// end. Its proof load, three M10 class 10.9 bolts (ISO 898-1: 58 mm² × 830 MPa = 48 kN each),
/// is an estimate of the bolt size (the deck's rigid bodies do not state it).
pub const BUMPER_PLATE_STRENGTH_N: f64 = 3.0 * 48e3;

/// [`crate::physics::FrontRules::load_path`] on the definitions: the front impact beam gets a
/// node at each corner of each crash box's front plate (on its own centreline, in front of the
/// corner: `corners` are the box's four front corner nodes, name and position, a side,
/// `[left, right]`), its
/// mass spread evenly over all its nodes (the part's mass does not change), each new node braced
/// to the beam's nodes either side and to the side's other three; and its mount to the box becomes
/// a plate: all sixteen pairs between those four nodes and the box's four corners (`to`), yielding
/// at the box's own force `box_force` (the plate does not fold before the box) and breaking as a
/// group at [`BUMPER_PLATE_STRENGTH_N`]. A design without a front impact beam is unchanged.
pub fn impact_beam_bolted_plate(parts: &mut [MechPart], corners: [[(String, glam::DVec3); 4]; 2], box_force: f64) {
    let Some(p) = parts.iter_mut().find(|p| p.name == "impact_beam_front") else { return };
    let mass = p.mass();
    let old = p.nodes.len();
    let mut new = [Vec::new(), Vec::new()];
    for (k, (s, side)) in [("l", &corners[0]), ("r", &corners[1])].into_iter().enumerate() {
        for (ci, (_, pos)) in side.iter().enumerate() {
            // (on the beam's centreline: the nearest of its own nodes gives the depth along the car)
            let z = p.nodes[..old].iter().min_by(|a, b| (a.pos.x - pos.x).abs().total_cmp(&(b.pos.x - pos.x).abs())).map(|n| n.pos.z).unwrap_or(pos.z);
            let name = format!("{}_x{ci}{s}", p.nodes[0].name.split('_').next().unwrap_or("impbf"));
            p.nodes.push(crate::mechanical::PartNode { name, pos: glam::DVec3::new(pos.x, pos.y, z.max(pos.z)), mass: 0.0, collide: true });
            new[k].push(p.nodes.len() - 1);
        }
    }
    let n = p.nodes.len() as f64;
    for q in &mut p.nodes {
        q.mass = mass / n;
    }
    let v = p.internal[0].v;
    let mut pairs = Vec::new();
    for side in &new {
        for (a, &i) in side.iter().enumerate() {
            for &j in &side[a + 1..] {
                pairs.push((i, j));
            }
            // the beam's own nodes either side of the box, top and bottom
            let pi = p.nodes[i].pos;
            let mut near: Vec<usize> = (0..old).collect();
            near.sort_by(|&a, &b| (p.nodes[a].pos - pi).length().total_cmp(&(p.nodes[b].pos - pi).length()));
            for &j in near.iter().take(4) {
                pairs.push((j, i));
            }
        }
    }
    p.internal.push(crate::mechanical::Internal { pairs, v, deform_group: None });
    for (k, s) in ["l", "r"].into_iter().enumerate() {
        let Some(m) = p.mounts.iter_mut().find(|m| m.group == format!("impact_beam_front_{s}")) else { continue };
        m.from = new[k].iter().map(|&i| p.nodes[i].name.clone()).collect();
        m.to = corners[k].iter().map(|c| c.0.clone()).collect();
        m.beams = m.from.len() * m.to.len();
        m.v = crate::mechanical::vals(m.v.k.unwrap_or(0.0), m.v.c, box_force, BUMPER_PLATE_STRENGTH_N);
    }
}

/// [`crate::physics::FrontRules::load_path`] on the definitions: the front subframe is bolted
/// under the rail at its own corners, as real cars of this class mount it (the 2012 Camry: the
/// front bushings on hangers under the rail just behind the crash box, the rear ones under the
/// toe pan where the rail kicks down; `front-load-path.md` §1.5): its mounts land on the rail
/// stations above its corners (`front` the station over its front corners, `rear` over its rear
/// ones, each station's four corner nodes a side, `{s}` the side) and on the firewall's and the
/// floor's nodes behind it, not on the radiator support, the apron or the crash box. Its beam
/// counts and values are the definitions'.
pub fn subframe_on_rail_stations(parts: &mut [MechPart], front: &[&str], rear: &[&str]) {
    for p in parts.iter_mut().filter(|p| p.name == "subframe_front") {
        for m in &mut p.mounts {
            let Some(s) = m.group.strip_prefix("subframe_front_").map(str::to_string) else { continue };
            let mut to: Vec<String> = front.iter().chain(rear).map(|t| t.replace("{s}", &s)).collect();
            to.extend([format!("sh_c1_1t{s}"), format!("sh_b1_i{s}")]);
            m.to = to;
        }
    }
}

/// The mean crush force at crash rate over the quasi-static one for a boxed member
/// ([`crush::dynamic_factor`] at [`CRASH_RATE_SPEED`], C its mean side, the Camry deck's
/// Cowper–Symonds constants [`crush::CS_CAMRY_DECK`]); 1 for a panel. The fold strain-rate
/// coefficient inside it (0.33 V/C) is to be verified against its source
/// (`engine/docs/section-matching.md` §1).
pub fn crash_rate_factor(m: &Member) -> f64 {
    match m.section {
        Section::Box { w, h } | Section::Hat { w, h } => crush::dynamic_factor(CRASH_RATE_SPEED, 0.5 * (w + h), crush::CS_CAMRY_DECK),
        Section::Panel => 1.0,
    }
}

/// A closure panel's sheet, for its buckling force: (gauge m, yield Pa, Young's modulus Pa,
/// skins). Engineering estimates, not definitions (the closures have none yet): drawing-quality
/// mild steel (σ_y 180 MPa, the generator's [`crate::structure::Grade::Mild`]) at 0.7 mm, the
/// usual 0.65–0.75 mm of a steel hood or fender skin. The hood counts its outer skin and its
/// inner panel (two sheets of 0.7 mm over the hood's ~1.6 m² weigh 17.6 kg against the
/// design's 17.9 kg hood); a fender is one skin.
pub const HOOD_SHEET: (f64, f64, f64, f64) = (0.0007, 180e6, 210e9, 2.0);
pub const FENDER_SHEET: (f64, f64, f64, f64) = (0.0007, 180e6, 210e9, 1.0);

/// The axial force at which a panel's sheet buckles and carries no more: per strip of sheet
/// between two of its lengthwise node lines (each line a stiffener: a crease, a flange, an
/// inner panel's rib), σ_y · t · b_e with b_e Winter's effective width of the strip
/// ([`crush::winter_effective_width`], simply supported on both long edges: AISI S100 and
/// EN 1993-1-5 §4.4), summed over the strips and the sheet's skins. `strips` are the strips'
/// widths (m). A thin wide strip keeps about b_e ≈ 1.9 t √(E/σ_y) whatever its width (45 mm at
/// 0.7 mm mild steel): a 0.35 m strip carries 5.6 kN, so a hood of four strips and two skins
/// about 45 kN. Confidence: the formula is established; the sheet's gauge, grade and the
/// strips' support (a lattice line taken as a stiffener) are estimates.
pub fn panel_buckling_force(strips: &[f64], (t, sigma_y, e, skins): (f64, f64, f64, f64)) -> f64 {
    skins * strips.iter().map(|&b| sigma_y * t * b * crush::winter_effective_width(b, t, e, sigma_y)).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_thin_strip_keeps_its_effective_width() {
        // b_e → 1.9 t √(E/σ_y) for λ ≫ 1 (Winter's ρ ≈ 1/λ − 0.22/λ²)
        let (t, s, e) = (0.0007, 180e6, 210e9);
        let f = panel_buckling_force(&[0.35], (t, s, e, 1.0));
        let be = f / (s * t);
        assert!((0.040..0.048).contains(&be), "effective width {be}");
        // two strips of half the width carry more (each keeps its own effective width)
        assert!(panel_buckling_force(&[0.175, 0.175], (t, s, e, 1.0)) > f);
        // a narrow strip is fully effective
        let n = panel_buckling_force(&[0.02], (t, s, e, 1.0));
        assert!((n - s * t * 0.02).abs() < 1e-9 * n);
    }
}
