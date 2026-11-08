//! The generated mid-size sedan (`gen2`): its node–beam layout on the visual
//! features of [`super::shape`], one tier, BeamNG's pattern (structural nodes
//! about 0.5 m apart on the edges, creases, pillars and door seams; light
//! contact-only nodes in the pole lanes on short ties), real masses (the 2011
//! Accord's: NHTSA DOT HS 811 666), designed crash structure and hardware.

use super::build::*;
use super::shape::*;
use crate::authoring::sedan::{
    ACCORD_BIW, ACCORD_DECKLID, ACCORD_DOOR_F, ACCORD_DOOR_R, ACCORD_ENGINE_KG, ACCORD_FENDER,
    ACCORD_HOOD, ACCORD_TRANSAXLE_KG,
};
use glam::Vec3;
use std::collections::HashMap;

/// Kerb mass with the driver (kg): the authored sedans' reference.
pub const KERB: f32 = 1450.0;
/// Mass each raycast wheel adds to its two hub nodes (the loader's 20 kg).
pub const WHEEL_KG: f32 = 20.0;
/// Bumper beam assemblies (kg): front 7.96, rear 7.84 (DOT HS 811 666 §5.10.7),
/// and their fascias (3 mm plastic, about 2.5 and 1.3 kg).
pub const BUMPER_F_KG: f32 = 7.96 + 2.5;
pub const BUMPER_R_KG: f32 = 7.84 + 1.3;
/// Unsprung mass per corner on the hub nodes besides the wheel (kg): knuckle,
/// hub, brake, half the links.
pub const UNSPRUNG_KG: f32 = 30.0 - RIM_NODES as f32 * RIM_NODE_KG;
/// Wheel proxy rim nodes per wheel, their radius (m) and mass (kg each, out
/// of the unsprung mass).
pub const RIM_NODES: usize = 6;
pub const RIM_R: f32 = 0.25;
pub const RIM_NODE_KG: f32 = 2.0;
/// Plastic compression limit of the crush zones' beams (deformLimit): a
/// folded section stacks at 1 − 0.7 = 30 % of its length.
pub const STACK_LIMIT: f32 = 0.7;

fn lr(s: f32) -> &'static str {
    if s > 0.0 {
        "l"
    } else {
        "r"
    }
}

fn m(p: Vec3, s: f32) -> Vec3 {
    Vec3::new(p.x * s, p.y, p.z)
}

/// x of the plan outline's front (`front`) or rear end at half width x.
pub fn end_y(x: f32, front: bool) -> f32 {
    let mid = 0.5 * (Y_FRONT + Y_REAR);
    let half = 0.5 * (Y_REAR - Y_FRONT);
    let t = (1.0 - (x / HALF_W).clamp(0.0, 0.9999).powf(6.0)).powf(1.0 / 6.0);
    if front {
        mid - half * t
    } else {
        mid + half * t
    }
}

/// Inset a side point inward by `d` (m), side sign `s`.
fn inset(p: Vec3, d: f32) -> Vec3 {
    Vec3::new(p.x - d * p.x.signum(), p.y, p.z)
}

/// Triangulate between two chains ordered along y (a zipper), sheet edges.
fn strip(c: &mut Car, a: &[usize], b: &[usize], part: Part, role: Role) {
    let (mut i, mut j) = (0usize, 0usize);
    c.chain(a, role, part);
    c.chain(b, role, part);
    while i + 1 < a.len() || j + 1 < b.len() {
        let adv_a = if j + 1 >= b.len() {
            true
        } else if i + 1 >= a.len() {
            false
        } else {
            c.p(a[i + 1]).y < c.p(b[j + 1]).y
        };
        if adv_a {
            c.beam(a[i + 1], b[j], role, part);
            c.tri(a[i], a[i + 1], b[j], part);
            i += 1;
        } else {
            c.beam(a[i], b[j + 1], role, part);
            c.tri(a[i], b[j + 1], b[j], part);
            j += 1;
        }
    }
}

/// The generated sedan.
pub fn sedan() -> Car {
    let mut c = Car::default();
    let body = Part::Body;
    let n = |c: &mut Car, name: &str, p: Vec3, part: Part| c.node(name, p, part);

    // ------------------------------------------------------------------
    // Centre-line body nodes
    // ------------------------------------------------------------------
    let fw_y = 0.56; // dash panel (firewall)
    let cowl_y = 0.84; // windshield base
    n(&mut c, "fwLoC", Vec3::new(0.0, fw_y, 0.34), body);
    n(&mut c, "fwMidC", Vec3::new(0.0, fw_y + 0.04, 0.62), body);
    n(&mut c, "cowlC", top(cowl_y, 0.0) - Vec3::Z * 0.02, body);
    n(&mut c, "rsC", Vec3::new(0.0, -0.66, 0.74), body);
    n(&mut c, "rsLoC", Vec3::new(0.0, -0.70, 0.36), body);
    let sill_y = [0.80, 1.34, 1.90, 2.42];
    for (k, &y) in sill_y.iter().enumerate().skip(1) {
        n(&mut c, &format!("tn{k}"), Vec3::new(0.0, y, 0.34), body);
    }
    let roof_y = [1.58, 1.90, 2.35, 2.80];
    for (k, &y) in roof_y.iter().enumerate() {
        n(
            &mut c,
            &format!("rfC{k}"),
            top(y, 0.0) - Vec3::Z * 0.01,
            body,
        );
    }
    let deck_y = 3.36;
    n(&mut c, "dkC", top(deck_y, 0.0) - Vec3::Z * 0.02, body);
    n(&mut c, "rpC", Vec3::new(0.0, 3.90, 0.80), body);
    n(&mut c, "tfC0", Vec3::new(0.0, 3.00, 0.30), body);
    n(&mut c, "tfC1", Vec3::new(0.0, 3.62, 0.34), body);
    // Bumper beams (centre).
    n(&mut c, "bbLoC", Vec3::new(0.0, -0.885, 0.40), Part::BumperF);
    n(&mut c, "bbHiC", Vec3::new(0.0, -0.885, 0.56), Part::BumperF);
    n(&mut c, "rbLoC", Vec3::new(0.0, 3.835, 0.40), Part::BumperR);
    n(&mut c, "rbHiC", Vec3::new(0.0, 3.835, 0.54), Part::BumperR);

    // ------------------------------------------------------------------
    // Per side: body nodes
    // ------------------------------------------------------------------
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let nm = |b: &str| format!("{b}{x}");
        // Front rails: two-node stations (low, high), x = 0.46.
        for (k, &y) in [-0.72f32, -0.36, 0.02, 0.36].iter().enumerate() {
            c.node(
                &nm(&format!("rlLo{k}")),
                m(Vec3::new(0.46, y, 0.36), s),
                body,
            );
            c.node(
                &nm(&format!("rlHi{k}")),
                m(Vec3::new(0.46, y, 0.54), s),
                body,
            );
        }
        c.node(&nm("fwLo"), m(Vec3::new(0.46, fw_y, 0.26), s), body);
        c.node(&nm("fwMid"), m(Vec3::new(0.44, fw_y + 0.04, 0.60), s), body);
        // Radiator support top corner, apron (upper load path), strut tower.
        c.node(&nm("apr0"), m(Vec3::new(0.62, -0.64, 0.73), s), body);
        c.node(&nm("apr1"), m(Vec3::new(0.70, -0.22, 0.77), s), body);
        c.node(&nm("twr"), m(Vec3::new(0.58, 0.04, 0.82), s), body);
        c.node(&nm("apr2"), m(Vec3::new(0.72, 0.38, 0.83), s), body);
        // Cowl: A-pillar base top (outer), cowl inner.
        c.node(
            &nm("cowlO"),
            inset(m(side(0.82, z_belt(0.82) - 0.01), s), 0.06),
            body,
        );
        c.node(&nm("cowlI"), top(cowl_y, 0.48 * s) - Vec3::Z * 0.02, body);
        // Sills: outer bottom, outer top, inner.
        for (k, &y) in sill_y.iter().enumerate() {
            c.node(
                &nm(&format!("soB{k}")),
                inset(m(side(y, 0.19), s), 0.01),
                body,
            );
            c.node(
                &nm(&format!("soT{k}")),
                inset(m(side(y, 0.33), s), 0.035),
                body,
            );
            c.node(
                &nm(&format!("si{k}")),
                m(Vec3::new(side(y, 0.3).x - 0.17, y, 0.26), s),
                body,
            );
            if k > 0 {
                c.node(&nm(&format!("fs{k}")), m(Vec3::new(0.46, y, 0.21), s), body);
            }
        }
        // Hinge pillar mid, A-pillar mid, B-pillar, roof rails, C-pillar.
        c.node(&nm("hp"), inset(m(side(0.80, 0.62), s), 0.05), body);
        c.node(&nm("ap1"), inset(m(glass(1.20, 0.55), s), 0.03), body);
        c.node(&nm("bp1"), inset(m(side(1.90, 0.64), s), 0.07), body);
        c.node(&nm("bp2"), inset(m(side(1.90, 0.97), s), 0.06), body);
        for (k, &y) in roof_y.iter().enumerate() {
            c.node(
                &nm(&format!("rr{k}")),
                inset(m(glass(y, 1.0), s), 0.015),
                body,
            );
        }
        c.node(&nm("cpLo"), inset(m(side(2.90, 0.80), s), 0.03), body);
        c.node(&nm("cp0"), inset(m(side(2.90, 0.99), s), 0.02), body);
        c.node(&nm("dk"), top(deck_y, 0.95 * s) - Vec3::Z * 0.02, body);
        // Rear rails (x = 0.50); the floor side member's last station is the
        // rear rail's first low node.
        for (k, &(y, zl, zh)) in [
            (2.42f32, 0.24f32, 0.40f32),
            (2.95, 0.30, 0.46),
            (3.40, 0.32, 0.48),
            (3.70, 0.36, 0.52),
        ]
        .iter()
        .enumerate()
        {
            if k > 0 {
                c.node(&nm(&format!("rrLo{k}")), m(Vec3::new(0.50, y, zl), s), body);
            }
            c.node(&nm(&format!("rrHi{k}")), m(Vec3::new(0.50, y, zh), s), body);
        }
        c.node(&nm("wh"), m(Vec3::new(0.64, 2.80, 0.78), s), body);
        // Quarter panel and rear arch lip.
        let lip = |ya: f32, deg: f32| {
            let a = deg.to_radians();
            let (y, z) = (
                ya + (ARCH_R + 0.02) * a.cos(),
                WHEEL_R + (ARCH_R + 0.02) * a.sin(),
            );
            side(y, z)
        };
        c.node(&nm("aR3"), m(lip(AXLE_R, 52.0), s), body);
        c.node(&nm("aR4"), m(lip(AXLE_R, 12.0), s), body);
        c.node(&nm("qr0"), m(side(3.72, 0.42), s), body);
        c.node(&nm("qm0"), m(side(3.30, 0.72), s), body);
        c.node(&nm("qr1"), m(side(3.74, 0.72), s), body);
        c.node(&nm("qb1"), m(side(3.40, z_belt(3.40) - 0.015), s), body);
        c.node(&nm("qb2"), m(side(3.80, z_belt(3.80) - 0.015), s), body);
        c.node(&nm("rp"), inset(m(side(3.86, 0.80), s), 0.03), body);
        // Hubs (raycast wheels' axle nodes).
        for (w, y) in [("F", AXLE_F), ("R", AXLE_R)] {
            c.node(
                &nm(&format!("hub{w}i")),
                m(Vec3::new(TRACK_HALF - 0.09, y, WHEEL_R), s),
                body,
            );
            c.node(
                &nm(&format!("hub{w}o")),
                m(Vec3::new(TRACK_HALF + 0.07, y, WHEEL_R), s),
                body,
            );
        }
        // Bumper beams.
        c.node(
            &nm("bbLo1"),
            m(Vec3::new(0.46, -0.88, 0.40), s),
            Part::BumperF,
        );
        c.node(
            &nm("bbHi1"),
            m(Vec3::new(0.46, -0.88, 0.56), s),
            Part::BumperF,
        );
        c.node(
            &nm("bbLo0"),
            m(Vec3::new(0.74, -0.84, 0.41), s),
            Part::BumperF,
        );
        c.node(
            &nm("bbHi0"),
            m(Vec3::new(0.74, -0.84, 0.55), s),
            Part::BumperF,
        );
        c.node(
            &nm("rbLo1"),
            m(Vec3::new(0.50, 3.83, 0.40), s),
            Part::BumperR,
        );
        c.node(
            &nm("rbHi1"),
            m(Vec3::new(0.50, 3.83, 0.54), s),
            Part::BumperR,
        );
        c.node(
            &nm("rbLo0"),
            m(Vec3::new(0.70, 3.79, 0.41), s),
            Part::BumperR,
        );
        c.node(
            &nm("rbHi0"),
            m(Vec3::new(0.70, 3.79, 0.53), s),
            Part::BumperR,
        );
    }
    let id = |c: &Car, name: &str| c.id(name);
    let both = |name: &str| [format!("{name}l"), format!("{name}r")];

    // ------------------------------------------------------------------
    // Body members, per side
    // ------------------------------------------------------------------
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let g = |c: &Car, b: &str| c.id(&format!("{b}{x}"));
        let lo: Vec<usize> = (0..4).map(|k| g(&c, &format!("rlLo{k}"))).collect();
        let hi: Vec<usize> = (0..4).map(|k| g(&c, &format!("rlHi{k}"))).collect();
        // Front rail: the tip bay is the crush can's trigger (weaker).
        c.ladder(&lo, &hi, Role::Rail, Role::Web, Role::Web, body);
        for (a, b) in [(lo[0], lo[1]), (hi[0], hi[1])] {
            let j = c.beam(a, b, Role::CrushCan, body);
            c.beams[j].role = Role::CrushCan;
        }
        let (fwlo, fwmid) = (g(&c, "fwLo"), g(&c, "fwMid"));
        c.fan(fwlo, &[lo[3], hi[3]], Role::Rail, body);
        c.fan(fwmid, &[hi[3], lo[3]], Role::Rail, body);
        c.beam(fwlo, fwmid, Role::Firewall, body);
        // Torque box: the rail's end braced to the sill and onto the floor's
        // side member, so the kick-down at the dash cannot fold.
        let (si0b, sob0b, sot0b, fs1b) = (g(&c, "si0"), g(&c, "soB0"), g(&c, "soT0"), g(&c, "fs1"));
        c.fan(lo[3], &[si0b, sob0b, fs1b], Role::Sill, body);
        c.fan(hi[3], &[si0b, sot0b], Role::Sill, body);
        // Apron (upper load path) to the radiator support and the cowl.
        let apr: Vec<usize> = ["apr0", "apr1", "twr", "apr2", "cowlO"]
            .iter()
            .map(|b| g(&c, b))
            .collect();
        c.chain(&[apr[0], apr[1], apr[3], apr[4]], Role::Apron, body);
        c.fan(
            apr[2],
            &[apr[1], apr[3], hi[2], hi[1], hi[3], lo[2]],
            Role::Apron,
            body,
        );
        // Apron walls down to the rail (webs).
        c.fan(apr[0], &[hi[0], lo[0], hi[1]], Role::Web, body);
        c.fan(apr[1], &[hi[1], hi[0], hi[2]], Role::Web, body);
        c.fan(apr[3], &[hi[3], hi[2], fwmid], Role::Web, body);
        // Cowl, hinge pillar, sill front.
        let (cowlo, cowli, hp) = (g(&c, "cowlO"), g(&c, "cowlI"), g(&c, "hp"));
        let (so_t0, so_b0, si0) = (g(&c, "soT0"), g(&c, "soB0"), g(&c, "si0"));
        c.chain(&[so_t0, hp, cowlo], Role::PillarA, body);
        c.fan(hp, &[so_b0, si0, fwmid, fwlo, apr[3]], Role::Web, body);
        c.fan(cowlo, &[cowli, fwmid, apr[3]], Role::Firewall, body);
        c.fan(
            cowli,
            &[fwmid, c.id("cowlC"), c.id("fwMidC")],
            Role::Firewall,
            body,
        );
        c.fan(
            fwlo,
            &[si0, so_b0, c.id("fwLoC"), c.id("fwMidC")],
            Role::Firewall,
            body,
        );
        c.fan(fwmid, &[c.id("fwMidC"), si0], Role::Firewall, body);
        // Sill: triangular section, stations 0.5 m.
        let sob: Vec<usize> = (0..4).map(|k| g(&c, &format!("soB{k}"))).collect();
        let sot: Vec<usize> = (0..4).map(|k| g(&c, &format!("soT{k}"))).collect();
        let si: Vec<usize> = (0..4).map(|k| g(&c, &format!("si{k}"))).collect();
        c.chain(&sob, Role::Sill, body);
        c.chain(&sot, Role::Sill, body);
        c.chain(&si, Role::Sill, body);
        for k in 0..4 {
            c.fan(si[k], &[sob[k], sot[k]], Role::Cross, body);
            c.beam(sob[k], sot[k], Role::Cross, body);
            if k < 3 {
                c.beam(sob[k], sot[k + 1], Role::Web, body);
                c.beam(si[k], sot[k + 1], Role::Web, body);
                c.beam(si[k], sob[k + 1], Role::Web, body);
            }
        }
        // Floor: side member (the front rail's continuation), cross members.
        let fs: Vec<usize> = std::iter::once(fwlo)
            .chain((1..4).map(|k| g(&c, &format!("fs{k}"))))
            .collect();
        c.chain(&fs, Role::Rail, body);
        for k in 0..4 {
            c.beam(fs[k], si[k], Role::Floor, body);
            if k < 3 {
                c.beam(fs[k], si[k + 1], Role::Floor, body);
                c.beam(si[k], fs[k + 1], Role::Floor, body);
            }
            let t = if k == 0 {
                c.id("fwLoC")
            } else {
                c.id(&format!("tn{k}"))
            };
            c.beam(fs[k], t, Role::Floor, body);
            if k < 3 {
                let t1 = c.id(&format!("tn{}", k + 1));
                c.beam(fs[k], t1, Role::Floor, body);
                c.beam(t, fs[k + 1], Role::Floor, body);
            }
        }
        // A-pillar, B-pillar, roof rail, C-pillar.
        let rr: Vec<usize> = (0..4).map(|k| g(&c, &format!("rr{k}"))).collect();
        let (ap1, bp1, bp2) = (g(&c, "ap1"), g(&c, "bp1"), g(&c, "bp2"));
        c.chain(&[cowlo, ap1, rr[0]], Role::PillarA, body);
        c.fan(ap1, &[cowli, hp], Role::Web, body);
        c.chain(&[sot[2], bp1, bp2, rr[1]], Role::PillarB, body);
        c.fan(bp1, &[si[2], sob[2]], Role::PillarB, body);
        c.fan(bp2, &[rr[0], rr[2]], Role::Web, body);
        c.chain(&rr, Role::RoofRail, body);
        let (cplo, cp0, dk) = (g(&c, "cpLo"), g(&c, "cp0"), g(&c, "dk"));
        c.chain(&[rr[3], cp0, cplo], Role::PillarC, body);
        c.chain(&[rr[3], dk], Role::PillarC, body);
        c.beam(cp0, dk, Role::PillarC, body);
        // Rear wheelhouse, rear rails, quarter.
        let wh = g(&c, "wh");
        let rlo: Vec<usize> = std::iter::once(fs[3])
            .chain((1..4).map(|k| g(&c, &format!("rrLo{k}"))))
            .collect();
        let rhi: Vec<usize> = (0..4).map(|k| g(&c, &format!("rrHi{k}"))).collect();
        c.ladder(&rlo, &rhi, Role::RearRail, Role::Web, Role::Web, body);
        c.fan(rhi[0], &[si[3], sot[3]], Role::Cross, body);
        c.fan(
            wh,
            &[rhi[0], rhi[1], rhi[2], cplo, sot[3], cp0],
            Role::Cross,
            body,
        );
        let (ar3, ar4, qr0, qm0, qr1, qb1, qb2, rp) = (
            g(&c, "aR3"),
            g(&c, "aR4"),
            g(&c, "qr0"),
            g(&c, "qm0"),
            g(&c, "qr1"),
            g(&c, "qb1"),
            g(&c, "qb2"),
            g(&c, "rp"),
        );
        c.panel(
            &[
                vec![Some(ar3), Some(ar4), Some(qr0)],
                vec![Some(cplo), Some(qm0), Some(qr1)],
                vec![Some(cp0), Some(qb1), Some(qb2)],
            ],
            body,
            &[Role::Sheet, Role::Sheet, Role::PillarC],
            Role::Sheet,
            false,
        );
        c.fan(ar3, &[wh, rhi[1]], Role::Cross, body);
        c.fan(ar4, &[rhi[1], rhi[2], rlo[2]], Role::Cross, body);
        c.fan(qm0, &[rhi[2], wh], Role::Web, body);
        c.fan(qr0, &[rlo[3], rhi[3], rlo[2]], Role::Cross, body);
        c.fan(qr1, &[rhi[3], rp], Role::Web, body);
        c.fan(qb1, &[dk, rhi[2]], Role::Cross, body);
        c.fan(qb2, &[rp, dk, rhi[3]], Role::Cross, body);
        c.fan(rp, &[rhi[3], c.id("rpC")], Role::Cross, body);
        // Sill outer face: a sheet panel over the sill (collision skin).
        c.panel(
            &[
                sob.iter().map(|&i| Some(i)).collect(),
                sot.iter().map(|&i| Some(i)).collect(),
            ],
            body,
            &[Role::Sill, Role::Sill],
            Role::Cross,
            false,
        );
        // Hubs: links to the rails (front) and rear rails (rear).
        let (hfi, hfo) = (g(&c, "hubFi"), g(&c, "hubFo"));
        c.beam(hfi, hfo, Role::Susp, body);
        for h in [hfi, hfo] {
            c.fan(h, &[lo[1], lo[2], lo[3], apr[2], hi[2]], Role::Susp, body);
        }
        let (hri, hro) = (g(&c, "hubRi"), g(&c, "hubRo"));
        c.beam(hri, hro, Role::Susp, body);
        for h in [hri, hro] {
            c.fan(h, &[rlo[0], rlo[1], rlo[2], wh, rhi[1]], Role::Susp, body);
        }
        // Bumper beams on the crush cans.
        let gb = |c: &Car, b: &str| c.id(&format!("{b}{x}"));
        let (bl1, bh1, bl0, bh0) = (
            gb(&c, "bbLo1"),
            gb(&c, "bbHi1"),
            gb(&c, "bbLo0"),
            gb(&c, "bbHi0"),
        );
        c.ladder(
            &[bl0, bl1, c.id("bbLoC")],
            &[bh0, bh1, c.id("bbHiC")],
            Role::BumperBeam,
            Role::Web,
            Role::Web,
            Part::BumperF,
        );
        c.fan(bl1, &[lo[0], hi[0]], Role::CrushCan, body);
        c.fan(bh1, &[hi[0], lo[0]], Role::CrushCan, body);
        c.fan(bl0, &[lo[0]], Role::CrushCan, body);
        c.fan(bh0, &[hi[0], apr[0]], Role::CrushCan, body);
        let (rl1, rh1, rl0, rh0) = (
            gb(&c, "rbLo1"),
            gb(&c, "rbHi1"),
            gb(&c, "rbLo0"),
            gb(&c, "rbHi0"),
        );
        c.ladder(
            &[rl0, rl1, c.id("rbLoC")],
            &[rh0, rh1, c.id("rbHiC")],
            Role::BumperBeam,
            Role::Web,
            Role::Web,
            Part::BumperR,
        );
        c.fan(rl1, &[rlo[3], rhi[3]], Role::CrushCan, body);
        c.fan(rh1, &[rhi[3], rlo[3]], Role::CrushCan, body);
        c.fan(rl0, &[rlo[3], qr0], Role::CrushCan, body);
        c.fan(rh0, &[rhi[3], qr0], Role::CrushCan, body);
    }

    // ------------------------------------------------------------------
    // Body cross members and centre line
    // ------------------------------------------------------------------
    let pair = |c: &mut Car, b: &str, role: Role| {
        let [l, r] = both(b);
        let (a, d) = (c.id(&l), c.id(&r));
        c.beam(a, d, role, Part::Body);
    };
    let to_c = |c: &mut Car, b: &str, cen: &str, role: Role| {
        let [l, r] = both(b);
        let k = c.id(cen);
        for side_name in [l, r] {
            let a = c.id(&side_name);
            c.beam(a, k, role, Part::Body);
        }
    };
    // Front: radiator support, rail ties, the front subframe.
    to_c(&mut c, "rlLo0", "rsLoC", Role::Cross);
    to_c(&mut c, "rlHi0", "rsLoC", Role::Web);
    to_c(&mut c, "apr0", "rsC", Role::Cross);
    to_c(&mut c, "rlHi0", "rsC", Role::Web);
    c.beam(id(&c, "rsC"), id(&c, "rsLoC"), Role::Cross, body);
    pair(&mut c, "rlLo2", Role::Cross); // subframe cross member
    pair(&mut c, "rlLo3", Role::Cross);
    pair(&mut c, "twr", Role::Cross); // strut brace
    for (a, b) in [
        ("rlLo2l", "rlLo3r"),
        ("rlLo2r", "rlLo3l"),
        ("rlLo1l", "rlLo2r"),
        ("rlLo1r", "rlLo2l"),
    ] {
        let (a, b) = (id(&c, a), id(&c, b));
        c.beam(a, b, Role::Web, body);
    }
    // Firewall and cowl.
    to_c(&mut c, "fwLo", "fwLoC", Role::Firewall);
    to_c(&mut c, "fwMid", "fwMidC", Role::Firewall);
    to_c(&mut c, "cowlI", "cowlC", Role::Firewall);
    c.beam(id(&c, "fwLoC"), id(&c, "fwMidC"), Role::Firewall, body);
    c.beam(id(&c, "fwMidC"), id(&c, "cowlC"), Role::Firewall, body);
    // Tunnel and floor cross members.
    let tn: Vec<usize> = std::iter::once(id(&c, "fwLoC"))
        .chain((1..4).map(|k| id(&c, &format!("tn{k}"))))
        .collect();
    c.chain(&tn, Role::Tunnel, body);
    for k in 1..4 {
        pair(&mut c, &format!("si{k}"), Role::Cross);
    }
    // Roof: bows and centre line, header, rear header.
    let rfc: Vec<usize> = (0..4).map(|k| id(&c, &format!("rfC{k}"))).collect();
    c.chain(&rfc, Role::RoofBow, body);
    for k in 0..4 {
        to_c(&mut c, &format!("rr{k}"), &format!("rfC{k}"), Role::RoofBow);
    }
    for s in ["l", "r"] {
        let rows: Vec<usize> = (0..4).map(|k| id(&c, &format!("rr{k}{s}"))).collect();
        for k in 0..3 {
            c.beam(rows[k], rfc[k + 1], Role::Sheet, body);
            c.beam(rfc[k], rows[k + 1], Role::Sheet, body);
        }
    }
    // Roof skin (collision): rows rr_l, rfC, rr_r along the car.
    {
        let l: Vec<Option<usize>> = (0..4).map(|k| Some(id(&c, &format!("rr{k}l")))).collect();
        let cc: Vec<Option<usize>> = rfc.iter().map(|&i| Some(i)).collect();
        let r: Vec<Option<usize>> = (0..4).map(|k| Some(id(&c, &format!("rr{k}r")))).collect();
        c.panel(
            &[r, cc, l],
            body,
            &[Role::RoofRail, Role::RoofBow, Role::RoofRail],
            Role::RoofBow,
            false,
        );
    }
    // Windshield frame (cowl row to header) and its glass as collision skin.
    {
        let cowl = [
            id(&c, "cowlOr"),
            id(&c, "cowlIr"),
            id(&c, "cowlC"),
            id(&c, "cowlIl"),
            id(&c, "cowlOl"),
        ];
        let head = [id(&c, "rr0r"), rfc[0], id(&c, "rr0l")];
        // Sort both chains along x for the zipper (strip works along y; use a
        // local mapping: triangulate by index).
        let tris = [(0, 1, 0), (1, 2, 1), (2, 3, 1), (3, 4, 2)];
        for (a, b, h) in tris {
            c.tri(cowl[a], cowl[b], head[h], body);
            c.beam(cowl[b], head[h], Role::Sheet, body);
            c.beam(cowl[a], head[h], Role::Sheet, body);
        }
        c.tri(cowl[1], head[1], head[0], body);
        c.tri(cowl[3], head[2], head[1], body);
        c.chain(&cowl, Role::Firewall, body);
    }
    // Rear window and the parcel shelf / deck front.
    {
        to_c(&mut c, "dk", "dkC", Role::Cross);
        let rear = [id(&c, "rr3r"), rfc[3], id(&c, "rr3l")];
        let deck = [id(&c, "dkr"), id(&c, "dkC"), id(&c, "dkl")];
        for k in 0..2 {
            c.tri(rear[k], rear[k + 1], deck[k + 1], body);
            c.tri(rear[k], deck[k + 1], deck[k], body);
            c.beam(rear[k], deck[k + 1], Role::Sheet, body);
            c.beam(rear[k + 1], deck[k], Role::Sheet, body);
        }
        c.chain(&deck, Role::Cross, body);
        for k in 0..3 {
            c.beam(rear[k], deck[k], Role::PillarC, body);
        }
        // Parcel shelf down to the rear rails, trunk floor.
        to_c(&mut c, "rrHi2", "dkC", Role::Web);
        to_c(&mut c, "rrHi1", "tfC0", Role::Cross);
        to_c(&mut c, "rrLo1", "tfC0", Role::Floor);
        to_c(&mut c, "rrLo3", "tfC1", Role::Floor);
        to_c(&mut c, "rrHi2", "tfC1", Role::Cross);
        to_c(&mut c, "rrLo2", "tfC1", Role::Floor);
        to_c(&mut c, "rrLo2", "tfC0", Role::Floor);
        c.beam(id(&c, "tn3"), id(&c, "tfC0"), Role::Tunnel, body);
        c.beam(id(&c, "tfC0"), id(&c, "tfC1"), Role::Floor, body);
        to_c(&mut c, "rp", "rpC", Role::Cross);
        to_c(&mut c, "rrHi3", "rpC", Role::Web);
        pair(&mut c, "wh", Role::Cross);
        pair(&mut c, "rrHi1", Role::Cross);
    }
    // C-pillar sails (collision skin above the belt behind the rear door).
    for s in ["l", "r"] {
        let (a, b, d, e) = (
            id(&c, &format!("cp0{s}")),
            id(&c, &format!("rr3{s}")),
            id(&c, &format!("dk{s}")),
            id(&c, &format!("qb1{s}")),
        );
        c.tri(a, b, d, body);
        c.tri(a, d, e, body);
        c.beam(d, e, Role::Sheet, body);
    }
    // Firewall skin (the engine's collision target) and the underbody.
    for s in ["l", "r"] {
        let (lo, mid, co) = (
            id(&c, &format!("fwLo{s}")),
            id(&c, &format!("fwMid{s}")),
            id(&c, &format!("cowlI{s}")),
        );
        let (loc, midc) = (id(&c, "fwLoC"), id(&c, "fwMidC"));
        c.tri(lo, loc, midc, body);
        c.tri(lo, midc, mid, body);
        c.tri(mid, midc, id(&c, "cowlC"), body);
        c.tri(mid, id(&c, "cowlC"), co, body);
        for k in 0..3 {
            let (si, si1, fs) = (
                id(&c, &format!("si{k}{s}")),
                id(&c, &format!("si{}{s}", k + 1)),
                id(&c, &format!("fs{}{s}", k + 1)),
            );
            let fs0 = if k == 0 {
                id(&c, &format!("fwLo{s}"))
            } else {
                id(&c, &format!("fs{k}{s}"))
            };
            c.tri(si, si1, fs, body);
            c.tri(si, fs, fs0, body);
        }
    }

    // ------------------------------------------------------------------
    // Engine (rigid block) and mounts
    // ------------------------------------------------------------------
    {
        let e = Part::Engine;
        let mut en: Vec<usize> = Vec::new();
        // About 12 lumped nodes of 22 kg (census: ~10 nodes of 10–23 kg): the
        // 2.4 L engine and transaxle, 0.39 m ahead of the axle to 0.12 m behind.
        for (k, (xx, y, z)) in [0.30f32, -0.30]
            .iter()
            .flat_map(|&xx| {
                [-0.39f32, -0.13, 0.12]
                    .into_iter()
                    .flat_map(move |y| [0.38f32, 0.84].into_iter().map(move |z| (xx, y, z)))
            })
            .enumerate()
        {
            en.push(c.node(&format!("en{k}"), Vec3::new(xx, y, z), e));
        }
        for a in 0..en.len() {
            for b in a + 1..en.len() {
                c.beam(en[a], en[b], Role::Engine, e);
            }
        }
        // Mounts: the block's lower nodes to the rails beside them, its upper
        // front nodes to the rails' upper chords.
        let at = |xx: f32, y: f32, z: f32| {
            let s = if xx > 0.0 { 0usize } else { 6 };
            let yi = [-0.39f32, -0.13, 0.12]
                .iter()
                .position(|&v| (v - y).abs() < 1e-3)
                .unwrap();
            s + yi * 2 + usize::from(z > 0.5)
        };
        let mounts = [
            (at(0.3, -0.39, 0.38), "rlLo1l"),
            (at(0.3, -0.39, 0.38), "rlHi1l"),
            (at(0.3, 0.12, 0.38), "rlLo2l"),
            (at(0.3, 0.12, 0.38), "rlHi2l"),
            (at(-0.3, -0.39, 0.38), "rlLo1r"),
            (at(-0.3, -0.39, 0.38), "rlHi1r"),
            (at(-0.3, 0.12, 0.38), "rlLo2r"),
            (at(-0.3, 0.12, 0.38), "rlHi2r"),
            (at(0.3, -0.39, 0.84), "rlHi1l"),
            (at(-0.3, -0.39, 0.84), "rlHi1r"),
        ];
        for (k, b) in mounts {
            let bi = c.id(b);
            c.beam(en[k], bi, Role::Mount, e);
        }
        // The dash panel stops the engine: gapped BOUNDED stops from the
        // block's rear face to the firewall, free until 8 cm are left.
        for (xs, sx) in [(0.3f32, "l"), (-0.3, "r")] {
            let (lo, hi) = (en[at(xs, 0.12, 0.38)], en[at(xs, 0.12, 0.84)]);
            for (a, to) in [
                (
                    lo,
                    vec![
                        format!("fwLo{sx}"),
                        "fwLoC".to_string(),
                        format!("fwMid{sx}"),
                    ],
                ),
                (
                    hi,
                    vec![
                        format!("fwMid{sx}"),
                        "fwMidC".to_string(),
                        format!("cowlI{sx}"),
                    ],
                ),
            ] {
                for t in to {
                    let ti = c.id(&t);
                    let j = c.beam(a, ti, Role::Stop, e);
                    c.beams[j].role = Role::Stop;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Doors
    // ------------------------------------------------------------------
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        for front in [true, false] {
            let part = Part::door(front, s > 0.0);
            let tag = if front { "F" } else { "R" };
            let p = |y: f32, z: f32| m(side(y, z), s);
            // Outer skin rows (bottom, intrusion beam, belt); inner frame rows.
            let (cols, rows): (Vec<f32>, Vec<Vec<Option<(f32, f32)>>>) = if front {
                (
                    vec![0.82, 1.34, 1.86],
                    vec![
                        vec![Some((0.82, 0.36)), Some((1.34, 0.36)), Some((1.86, 0.36))],
                        vec![Some((0.82, 0.66)), Some((1.34, 0.66)), Some((1.86, 0.66))],
                        vec![Some((0.82, 0.95)), Some((1.34, 0.96)), Some((1.86, 0.97))],
                    ],
                )
            } else {
                (
                    vec![1.94, 2.36, 2.66],
                    vec![
                        vec![Some((1.94, 0.36)), Some((2.36, 0.36)), None],
                        vec![Some((1.94, 0.66)), Some((2.36, 0.66)), Some((2.66, 0.80))],
                        vec![Some((1.94, 0.97)), Some((2.36, 0.98)), Some((2.84, 0.99))],
                    ],
                )
            };
            let _ = cols;
            let mut ids: Vec<Vec<Option<usize>>> = Vec::new();
            for (r, row) in rows.iter().enumerate() {
                let mut v = Vec::new();
                for (k, cell) in row.iter().enumerate() {
                    v.push(cell.map(|(y, z)| c.node(&format!("d{tag}{r}{k}{x}"), p(y, z), part)));
                }
                ids.push(v);
            }
            c.panel(
                &ids,
                part,
                &[Role::Sheet, Role::Intrusion, Role::DoorFrame],
                Role::DoorFrame,
                false,
            );
            // Inner frame: two rows 0.11 m inboard (hinge face, latch face,
            // the inner panel's top and bottom).
            let mut inner: Vec<Vec<usize>> = Vec::new();
            for (r, &rr) in [0usize, 2].iter().enumerate() {
                let mut v = Vec::new();
                for (k, cell) in rows[rr].iter().enumerate() {
                    let (y, z) = match cell {
                        Some(yz) => *yz,
                        None => rows[1][k].unwrap(),
                    };
                    let z = if rr == 0 { z + 0.05 } else { z - 0.04 };
                    let q = inset(p(y, z), 0.11);
                    v.push(c.node(&format!("d{tag}i{r}{k}{x}"), q, part));
                }
                inner.push(v);
            }
            c.ladder(
                &inner[0],
                &inner[1],
                Role::DoorFrame,
                Role::DoorFrame,
                Role::Sheet,
                part,
            );
            // Depth: each inner node to the outer nodes around it.
            for (r, &rr) in [0usize, 2].iter().enumerate() {
                for k in 0..inner[r].len() {
                    let me = inner[r][k];
                    let near: Vec<usize> = [rr as isize - 1, rr as isize, rr as isize + 1]
                        .iter()
                        .filter(|&&q| (0..3).contains(&q))
                        .flat_map(|&q| {
                            let row = &ids[q as usize];
                            [k as isize - 1, k as isize, k as isize + 1]
                                .iter()
                                .filter_map(|&kk| {
                                    (kk >= 0)
                                        .then(|| row.get(kk as usize).copied().flatten())
                                        .flatten()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    c.fan(me, &near, Role::DoorFrame, part);
                }
            }
            // Window frame: two nodes at the roof rail (B-pillar / C-pillar top).
            let (wy0, wy1) = if front { (1.34, 1.84) } else { (2.36, 2.76) };
            let w0 = c.node(
                &format!("d{tag}w0{x}"),
                inset(m(glass(wy0, 0.97), s), 0.02),
                part,
            );
            let w1 = c.node(
                &format!("d{tag}w1{x}"),
                inset(m(glass(wy1, if front { 0.97 } else { 0.80 }), s), 0.02),
                part,
            );
            let belt: Vec<usize> = ids[2].iter().flatten().copied().collect();
            c.chain(&[belt[0], w0, w1, belt[2]], Role::DoorFrame, part);
            c.fan(
                w0,
                &[belt[1], belt[2], inner[1][0], inner[1][1]],
                Role::DoorFrame,
                part,
            );
            c.fan(
                w1,
                &[belt[1], inner[1][2], inner[1][1]],
                Role::DoorFrame,
                part,
            );
            c.tri(belt[0], belt[1], w0, part);
            c.tri(belt[1], w1, w0, part);
            c.tri(belt[1], belt[2], w1, part);
            // Hinges (front edge, two pins) on the hinge pillar (front door)
            // or the B-pillar (rear door): break groups, 20 kN a beam.
            let gs = |c: &Car, b: &str| c.id(&format!("{b}{x}"));
            let (pil_lo, pil_mid, pil_hi) = if front {
                (gs(&c, "soT0"), gs(&c, "hp"), gs(&c, "cowlO"))
            } else {
                (gs(&c, "soT2"), gs(&c, "bp1"), gs(&c, "bp2"))
            };
            let dn = part.name().trim_start_matches("gen2_").to_string();
            // Census: two groups of 4–14 beams at 30–98 kN a beam.
            for (pins, to, grp) in [
                ([inner[0][0], ids[0][0].unwrap()], [pil_lo, pil_mid], "lo"),
                ([inner[1][0], ids[2][0].unwrap()], [pil_mid, pil_hi], "up"),
            ] {
                for pin in pins {
                    for t in to {
                        c.hardware(
                            pin,
                            t,
                            Role::Hinge,
                            part,
                            &format!("{dn}_hinge_{grp}"),
                            40e3,
                        );
                    }
                }
            }
            // Latch: a node coupler at the rear edge (BeamNG's couplerTag on the
            // door to a tag node on the pillar), plus a secondary catch beam.
            let (ly, lz) = if front { (1.86, 0.74) } else { (2.62, 0.86) };
            let lpos = inset(p(ly, lz), 0.07);
            let latch = c.node(&format!("d{tag}latch{x}"), lpos, part);
            let strike = c.node(&format!("b{tag}strike{x}"), lpos, body);
            c.nodes[latch].no_group = true;
            c.nodes[strike].no_group = true;
            let ltag = format!("{dn}_latch");
            c.nodes[latch].opts = vec![
                ("couplerTag".into(), serde_json::json!(ltag)),
                ("couplerStrength".into(), serde_json::json!(40000)),
                ("couplerStartRadius".into(), serde_json::json!(0.02)),
                (
                    "breakGroup".into(),
                    serde_json::json!(format!("{dn}_latch")),
                ),
            ];
            c.nodes[strike].opts = vec![("tag".into(), serde_json::json!(ltag))];
            let lnear: Vec<usize> = if front {
                vec![
                    ids[1][2].unwrap(),
                    ids[2][2].unwrap(),
                    inner[0][2],
                    inner[1][2],
                    ids[1][1].unwrap(),
                ]
            } else {
                vec![
                    ids[1][2].unwrap(),
                    ids[2][2].unwrap(),
                    inner[1][2],
                    inner[0][2],
                    ids[1][1].unwrap(),
                ]
            };
            c.fan(latch, &lnear, Role::DoorFrame, part);
            let snear: Vec<usize> = if front {
                vec![gs(&c, "bp1"), gs(&c, "bp2"), gs(&c, "soT2"), gs(&c, "rr1")]
            } else {
                vec![gs(&c, "cpLo"), gs(&c, "cp0"), gs(&c, "wh"), gs(&c, "rr3")]
            };
            c.fan(strike, &snear, Role::PillarB, body);
            // Bearing: the door's edges and bottom bear on the pillars and sill
            // (compression only).
            let rear_edge: Vec<usize> = [ids[0].get(2).copied().flatten(), ids[1][2], ids[2][2]]
                .into_iter()
                .flatten()
                .collect();
            let front_edge: Vec<usize> = [ids[0][0], ids[1][0], ids[2][0]]
                .into_iter()
                .flatten()
                .collect();
            for &d in &front_edge {
                let pd = c.p(d);
                let mut best: Vec<(f32, usize)> = [pil_lo, pil_mid, pil_hi]
                    .iter()
                    .map(|&b| (c.p(b).distance(pd), b))
                    .collect();
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(_, b) in best.iter().take(2) {
                    c.beam(d, b, Role::Bearing, part);
                }
            }
            let rear_pil: Vec<usize> = if front {
                vec![gs(&c, "soT2"), gs(&c, "bp1"), gs(&c, "bp2")]
            } else {
                vec![gs(&c, "soT3"), gs(&c, "cpLo"), gs(&c, "cp0"), gs(&c, "aR3")]
            };
            for &d in &rear_edge {
                let pd = c.p(d);
                let mut best: Vec<(f32, usize)> =
                    rear_pil.iter().map(|&b| (c.p(b).distance(pd), b)).collect();
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
                for &(_, b) in best.iter().take(2) {
                    c.beam(d, b, Role::Bearing, part);
                }
            }
            let sills: Vec<usize> = (0..4).map(|k| gs(&c, &format!("soT{k}"))).collect();
            for d in ids[0]
                .iter()
                .flatten()
                .copied()
                .chain(inner[0].iter().copied())
                .collect::<Vec<_>>()
            {
                let pd = c.p(d);
                let mut best: Vec<(f32, usize)> =
                    sills.iter().map(|&b| (c.p(b).distance(pd), b)).collect();
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
                c.beam(d, best[0].1, Role::Bearing, part);
            }
            // Window frame top bears on the roof rail.
            let rails: Vec<usize> = (0..4).map(|k| gs(&c, &format!("rr{k}"))).collect();
            for w in [w0, w1] {
                let pw = c.p(w);
                let mut best: Vec<(f32, usize)> =
                    rails.iter().map(|&b| (c.p(b).distance(pw), b)).collect();
                best.sort_by(|a, b| a.0.total_cmp(&b.0));
                c.beam(w, best[0].1, Role::Bearing, part);
            }
            // Pole lanes on the door's outer skin, backed by the inner frame.
            let back: Vec<usize> = inner.iter().flatten().copied().collect();
            c.lanes(
                part,
                &back,
                &|q: Vec3| {
                    if q.z > z_belt(q.y) - 0.005 {
                        q
                    } else {
                        Vec3::new(side(q.y, q.z).x * q.x.signum(), q.y, q.z)
                    }
                },
                &|_, _| true,
            );
        }
    }

    // ------------------------------------------------------------------
    // Fenders
    // ------------------------------------------------------------------
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        let part = if s > 0.0 {
            Part::FenderL
        } else {
            Part::FenderR
        };
        let p = |y: f32, z: f32| m(side(y, z), s);
        let top_row: Vec<usize> = [-0.78f32, -0.30, 0.20, 0.72]
            .iter()
            .enumerate()
            .map(|(k, &y)| c.node(&format!("fdT{k}{x}"), p(y, z_belt(y) - 0.015), part))
            .collect();
        let mut low = vec![c.node(&format!("fdL0{x}"), p(-0.80, 0.44), part)];
        for (k, deg) in [160.0f32, 122.0, 90.0, 58.0, 20.0].iter().enumerate() {
            let a = deg.to_radians();
            let (y, z) = (
                AXLE_F + (ARCH_R + 0.02) * a.cos(),
                WHEEL_R + (ARCH_R + 0.02) * a.sin(),
            );
            low.push(c.node(&format!("fdA{k}{x}"), p(y, z), part));
        }
        low.push(c.node(&format!("fdL1{x}"), p(0.74, 0.40), part));
        strip(&mut c, &low, &top_row, part, Role::Sheet);
        for w in top_row.windows(3) {
            c.beam(w[0], w[2], Role::Bend, part);
        }
        for w in low.windows(3) {
            c.beam(w[0], w[2], Role::Bend, part);
        }
        // Bolts along the apron, at the cowl and the hinge pillar; bearing on
        // the apron (compression only).
        let gs = |c: &Car, b: &str| c.id(&format!("{b}{x}"));
        let grp = format!("{}_bolts", part.name().trim_start_matches("gen2_"));
        let aprons = [
            gs(&c, "apr0"),
            gs(&c, "apr1"),
            gs(&c, "apr2"),
            gs(&c, "cowlO"),
        ];
        for (k, &t) in top_row.iter().enumerate() {
            c.hardware(t, aprons[k], Role::Bolt, part, &grp, 9e3);
        }
        let hp = gs(&c, "hp");
        let sot0 = gs(&c, "soT0");
        c.hardware(*low.last().unwrap(), hp, Role::Bolt, part, &grp, 9e3);
        c.hardware(*low.last().unwrap(), sot0, Role::Bolt, part, &grp, 9e3);
        let bb0 = gs(&c, "bbHi0");
        c.hardware(low[0], bb0, Role::Bolt, part, &grp, 6e3);
        for &t in &low[1..4] {
            let pt = c.p(t);
            let mut best: Vec<(f32, usize)> = [
                gs(&c, "apr1"),
                gs(&c, "twr"),
                gs(&c, "apr2"),
                gs(&c, "rlHi1"),
                gs(&c, "rlHi2"),
            ]
            .iter()
            .map(|&b| (c.p(b).distance(pt), b))
            .collect();
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            c.beam(t, best[0].1, Role::Bearing, part);
        }
        let back = [
            gs(&c, "apr0"),
            gs(&c, "apr1"),
            gs(&c, "apr2"),
            gs(&c, "twr"),
            gs(&c, "rlHi0"),
            gs(&c, "rlHi1"),
            gs(&c, "rlHi3"),
            gs(&c, "hp"),
            gs(&c, "bbHi0"),
        ];
        c.lanes(
            part,
            &back,
            &|q: Vec3| Vec3::new(side(q.y, q.z).x * q.x.signum(), q.y, q.z),
            &|_, _| true,
        );
    }

    // ------------------------------------------------------------------
    // Hood
    // ------------------------------------------------------------------
    {
        let part = Part::Hood;
        let ys = [-0.80f32, -0.36, 0.12, 0.64];
        let us = [-0.95f32, -0.33, 0.33, 0.95];
        let mut rows: Vec<Vec<Option<usize>>> = Vec::new();
        for (r, &y) in ys.iter().enumerate() {
            rows.push(
                us.iter()
                    .enumerate()
                    .map(|(k, &u)| {
                        Some(c.node(&format!("hd{r}{k}"), top(y, u) + Vec3::Z * 0.004, part))
                    })
                    .collect(),
            );
        }
        c.panel(&rows, part, &[Role::Sheet; 4], Role::Sheet, true);
        let g = |r: usize, k: usize| rows[r][k].unwrap();
        // Hinges: rear corners to the cowl and the apron.
        for (k, x) in [(0usize, "r"), (3, "l")] {
            let (co, ap) = (c.id(&format!("cowlO{x}")), c.id(&format!("apr2{x}")));
            c.hardware(
                g(3, k),
                co,
                Role::Hinge,
                part,
                &format!("hood_hinge_{x}"),
                40e3,
            );
            c.hardware(
                g(3, k),
                ap,
                Role::Hinge,
                part,
                &format!("hood_hinge_{x}"),
                40e3,
            );
            c.hardware(
                g(2, k),
                ap,
                Role::Hinge,
                part,
                &format!("hood_hinge_{x}"),
                40e3,
            );
        }
        // Latch: a coupler at the front centre to the radiator support.
        let lp = top(-0.78, 0.0) - Vec3::Z * 0.03;
        let latch = c.node("hdLatch", lp, part);
        let strike = c.node("hdStrike", lp, body);
        c.nodes[latch].no_group = true;
        c.nodes[strike].no_group = true;
        c.nodes[latch].opts = vec![
            ("couplerTag".into(), serde_json::json!("hood_latch")),
            ("couplerStrength".into(), serde_json::json!(30000)),
            ("couplerStartRadius".into(), serde_json::json!(0.02)),
        ];
        c.nodes[strike].opts = vec![("tag".into(), serde_json::json!("hood_latch"))];
        c.fan(
            latch,
            &[g(0, 1), g(0, 2), g(1, 1), g(1, 2)],
            Role::Sheet,
            part,
        );
        let (rsc, a0l, a0r) = (c.id("rsC"), c.id("apr0l"), c.id("apr0r"));
        c.fan(strike, &[rsc, a0l, a0r], Role::Cross, body);
        // Bearing on the aprons, the radiator support and the cowl.
        let seats: Vec<usize> = [
            "apr0l", "apr1l", "twrl", "apr2l", "apr0r", "apr1r", "twrr", "apr2r", "rsC", "cowlIl",
            "cowlIr", "cowlC",
        ]
        .iter()
        .map(|b| c.id(b))
        .collect();
        let hood_nodes: Vec<usize> = rows.iter().flatten().flatten().copied().collect();
        for h in hood_nodes {
            let ph = c.p(h);
            let mut best: Vec<(f32, usize)> =
                seats.iter().map(|&b| (c.p(b).distance(ph), b)).collect();
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            if best[0].0 < 0.35 {
                c.beam(h, best[0].1, Role::Bearing, part);
            }
        }
        // Pole lanes along the hood's outer edges (a pole at the front
        // corner meets them), backed by the aprons.
        let back: Vec<usize> = [
            "apr0l", "apr1l", "twrl", "apr2l", "apr0r", "apr1r", "twrr", "apr2r",
        ]
        .iter()
        .map(|b| c.id(b))
        .collect();
        c.lanes(part, &back, &|q: Vec3| q, &|a: Vec3, b: Vec3| {
            a.x.abs() > 0.5 && b.x.abs() > 0.5
        });
    }

    // ------------------------------------------------------------------
    // Trunk lid
    // ------------------------------------------------------------------
    {
        let part = Part::Trunk;
        let ys = [3.40f32, 3.65, 3.88];
        let us = [-0.94f32, 0.0, 0.94];
        let mut rows: Vec<Vec<Option<usize>>> = Vec::new();
        for (r, &y) in ys.iter().enumerate() {
            rows.push(
                us.iter()
                    .enumerate()
                    .map(|(k, &u)| {
                        Some(c.node(&format!("tk{r}{k}"), top(y, u) + Vec3::Z * 0.004, part))
                    })
                    .collect(),
            );
        }
        c.panel(&rows, part, &[Role::Sheet; 3], Role::Sheet, true);
        let g = |r: usize, k: usize| rows[r][k].unwrap();
        for (k, x) in [(0usize, "r"), (2, "l")] {
            let (dk, qb) = (c.id(&format!("dk{x}")), c.id(&format!("qb1{x}")));
            c.hardware(
                g(0, k),
                dk,
                Role::Hinge,
                part,
                &format!("trunk_hinge_{x}"),
                25e3,
            );
            c.hardware(
                g(0, k),
                qb,
                Role::Hinge,
                part,
                &format!("trunk_hinge_{x}"),
                25e3,
            );
            c.hardware(
                g(1, k),
                qb,
                Role::Hinge,
                part,
                &format!("trunk_hinge_{x}"),
                25e3,
            );
        }
        let lp = top(3.88, 0.0) - Vec3::Z * 0.04;
        let latch = c.node("tkLatch", lp, part);
        let strike = c.node("tkStrike", lp, body);
        c.nodes[latch].no_group = true;
        c.nodes[strike].no_group = true;
        c.nodes[latch].opts = vec![
            ("couplerTag".into(), serde_json::json!("trunk_latch")),
            ("couplerStrength".into(), serde_json::json!(45000)),
            ("couplerStartRadius".into(), serde_json::json!(0.02)),
        ];
        c.nodes[strike].opts = vec![("tag".into(), serde_json::json!("trunk_latch"))];
        c.fan(
            latch,
            &[g(2, 1), g(2, 0), g(2, 2), g(1, 1)],
            Role::Sheet,
            part,
        );
        let (rpc, rpl, rpr) = (c.id("rpC"), c.id("rpl"), c.id("rpr"));
        c.fan(strike, &[rpc, rpl, rpr], Role::Cross, body);
        let seats: Vec<usize> = [
            "dkl", "dkr", "dkC", "qb1l", "qb1r", "qb2l", "qb2r", "rpl", "rpr", "rpC",
        ]
        .iter()
        .map(|b| c.id(b))
        .collect();
        for h in rows.iter().flatten().flatten().copied().collect::<Vec<_>>() {
            let ph = c.p(h);
            let mut best: Vec<(f32, usize)> =
                seats.iter().map(|&b| (c.p(b).distance(ph), b)).collect();
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            if best[0].0 < 0.35 {
                c.beam(h, best[0].1, Role::Bearing, part);
            }
        }
    }

    // ------------------------------------------------------------------
    // Bumper fascias (front and rear): 3 rows across the nose and tail.
    // ------------------------------------------------------------------
    for front in [true, false] {
        let part = if front { Part::BumperF } else { Part::BumperR };
        let pre = if front { "fF" } else { "fR" };
        let xs = [-0.82f32, -0.62, -0.34, 0.0, 0.34, 0.62, 0.82];
        let zs: [f32; 3] = if front {
            [0.30, 0.47, 0.64]
        } else {
            [0.36, 0.52, 0.70]
        };
        let mut rows: Vec<Vec<Option<usize>>> = Vec::new();
        for (r, &z) in zs.iter().enumerate() {
            let mut v = Vec::new();
            for (k, &xx) in xs.iter().enumerate() {
                let y = end_y(xx.abs(), front);
                let q = Vec3::new(side(y, z).x * xx.signum(), y, z);
                let q = if xx == 0.0 {
                    Vec3::new(
                        0.0,
                        if front {
                            Y_FRONT + 0.005
                        } else {
                            Y_REAR - 0.005
                        },
                        z,
                    )
                } else {
                    q
                };
                v.push(Some(c.node(&format!("{pre}{r}{k}"), q, part)));
            }
            rows.push(v);
        }
        c.panel(&rows, part, &[Role::Sheet; 3], Role::Sheet, true);
        // Clips to the bumper beam (compression bearing plus light clips).
        let beam: Vec<usize> = if front {
            [
                "bbLo0r", "bbHi0r", "bbLo1r", "bbHi1r", "bbLoC", "bbHiC", "bbLo1l", "bbHi1l",
                "bbLo0l", "bbHi0l", "rsC", "apr0l", "apr0r",
            ]
            .iter()
            .map(|b| c.id(b))
            .collect()
        } else {
            [
                "rbLo0r", "rbHi0r", "rbLo1r", "rbHi1r", "rbLoC", "rbHiC", "rbLo1l", "rbHi1l",
                "rbLo0l", "rbHi0l", "rpl", "rpr", "rpC", "qr0l", "qr0r",
            ]
            .iter()
            .map(|b| c.id(b))
            .collect()
        };
        let grp = format!("{}_clips", part.name().trim_start_matches("gen2_"));
        for f in rows.iter().flatten().flatten().copied().collect::<Vec<_>>() {
            let pf = c.p(f);
            let mut best: Vec<(f32, usize)> =
                beam.iter().map(|&b| (c.p(b).distance(pf), b)).collect();
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
            c.beam(f, best[0].1, Role::Bearing, part);
            c.hardware(f, best[1].1, Role::Bolt, part, &grp, 2.5e3);
        }
    }

    // ------------------------------------------------------------------
    // Sill and quarter pole lanes (body), backed by the inner sill and the
    // rear structure.
    // ------------------------------------------------------------------
    {
        let back: Vec<usize> = [
            "si0", "si1", "si2", "si3", "wh", "rrHi1", "rrHi2", "rrHi3", "rrLo2", "rrLo3", "bp1",
            "hp", "cpLo",
        ]
        .iter()
        .flat_map(|b| both(b))
        .map(|b| c.id(&b))
        .collect();
        // Only the side-facing body triangles get lanes: split the body's
        // triangles into a temporary part.
        let side_tris: Vec<usize> = (0..c.tris.len())
            .filter(|&t| {
                let tr = &c.tris[t];
                if tr.part != body {
                    return false;
                }
                let [a, b, d] = tr.n;
                let nrm = (c.p(b) - c.p(a)).cross(c.p(d) - c.p(a)).normalize_or_zero();
                nrm.x.abs() > 0.8 && [a, b, d].iter().all(|&i| c.p(i).z < 1.05)
            })
            .collect();
        for &t in &side_tris {
            c.tris[t].part = Part::Engine; // placeholder tag while splitting
        }
        // Split under the Engine tag, then return them (and their lane nodes)
        // to the body.
        let first_node = c.nodes.len();
        c.lanes(
            Part::Engine,
            &back,
            &|q: Vec3| Vec3::new(side(q.y, q.z).x * q.x.signum(), q.y, q.z),
            &|_, _| true,
        );
        for i in first_node..c.nodes.len() {
            c.nodes[i].part = body;
            let nm = c.nodes[i].name.replace("xengine", "xbody");
            c.nodes[i].name = nm;
        }
        for t in &mut c.tris {
            if t.part == Part::Engine {
                t.part = body;
            }
        }
        for b in &mut c.beams {
            if b.part == Part::Engine && (c.nodes[b.a].part == body || c.nodes[b.b].part == body) {
                b.part = body;
            }
        }
    }

    // ------------------------------------------------------------------
    // Wheel proxies: the raycast wheels have no geometry, so each hub carries
    // a ring of rim nodes (contacters, no triangles, as BeamNG's wheel nodes)
    // in the wheel's plane at the rim's radius. In a frontal the wheel stacks
    // against the hinge pillar and sill, the outer load path real cars have.
    // ------------------------------------------------------------------
    let mut rims = Vec::new();
    for s in [1.0f32, -1.0] {
        let x = lr(s);
        for (w, ya) in [("F", AXLE_F), ("R", AXLE_R)] {
            let (hi, ho) = (c.id(&format!("hub{w}i{x}")), c.id(&format!("hub{w}o{x}")));
            let ring: Vec<usize> = (0..RIM_NODES)
                .map(|k| {
                    let a = (k as f32 + 0.5) * std::f32::consts::TAU / RIM_NODES as f32;
                    let q = Vec3::new(
                        TRACK_HALF * s,
                        ya + RIM_R * a.cos(),
                        WHEEL_R + RIM_R * a.sin(),
                    );
                    c.node(&format!("rim{w}{k}{x}"), q, body)
                })
                .collect();
            for (k, &r) in ring.iter().enumerate() {
                c.nodes[r].no_group = true;
                c.nodes[r].fixed_mass = RIM_NODE_KG;
                // Off the axle line too, or the ring spins freely about it.
                let arm = if w == "F" {
                    c.id(&format!("twr{x}"))
                } else {
                    c.id(&format!("wh{x}"))
                };
                c.fan(r, &[hi, ho, arm], Role::Susp, body);
                c.beam(r, ring[(k + 1) % ring.len()], Role::Susp, body);
                c.beam(r, ring[(k + 2) % ring.len()], Role::Susp, body);
                rims.push(r);
            }
        }
    }

    // ------------------------------------------------------------------
    // Self-collision: every skin node (triangles' nodes), the engine and the
    // wheel proxies.
    // ------------------------------------------------------------------
    let mut skin = vec![false; c.nodes.len()];
    for t in &c.tris {
        for &i in &t.n {
            skin[i] = true;
        }
    }
    for &r in &rims {
        skin[r] = true;
    }
    for (i, nd) in c.nodes.iter_mut().enumerate() {
        nd.self_collision = skin[i] || nd.part == Part::Engine;
    }

    // ------------------------------------------------------------------
    // Masses
    // ------------------------------------------------------------------
    let named = vec![
        ("driver", 75.0, Vec3::new(0.38, 1.70, 0.50), 0.6),
        ("front_seats", 35.0, Vec3::new(0.0, 1.70, 0.40), 0.75),
        ("rear_seat", 17.5, Vec3::new(0.0, 2.45, 0.40), 0.65),
        ("dashboard_steering", 40.0, Vec3::new(0.0, 0.95, 0.80), 0.6),
        ("fuel", 45.0, Vec3::new(0.0, 2.55, 0.30), 0.55),
        ("interior", 50.0, Vec3::new(0.0, 2.00, 0.60), 1.1),
        ("exhaust", 15.0, Vec3::new(0.0, 1.80, 0.25), 1.5),
        ("spare_wheel", 15.0, Vec3::new(0.0, 3.55, 0.35), 0.45),
        ("battery", 12.4, Vec3::new(0.55, -0.45, 0.70), 0.4),
        ("radiator_cooling", 15.0, Vec3::new(0.0, -0.68, 0.55), 0.4),
        ("windshield", 13.0, Vec3::new(0.0, 1.20, 1.20), 0.7),
        ("rear_glass", 9.0, Vec3::new(0.0, 3.05, 1.25), 0.6),
        (
            "front_subframe_steering",
            45.0,
            Vec3::new(0.0, 0.0, 0.36),
            0.6,
        ),
        ("rear_subframe", 22.0, Vec3::new(0.0, 2.85, 0.35), 0.65),
        // HVAC module and the dash's wiring sit behind the dash panel.
        ("hvac_wiring", 40.0, Vec3::new(0.0, 0.85, 0.65), 0.6),
    ];
    for (name, kg, at, radius) in named {
        c.named.push(Named {
            name,
            kg,
            at,
            radius,
            part: body,
        });
    }
    // Unsprung (front drive: the front corners carry the drive shafts).
    for s in [1.0f32, -1.0] {
        for (y, kg) in [(AXLE_F, UNSPRUNG_KG + 6.0), (AXLE_R, UNSPRUNG_KG - 6.0)] {
            c.named.push(Named {
                name: "unsprung",
                kg,
                at: Vec3::new(TRACK_HALF * s, y, WHEEL_R),
                radius: 0.1,
                part: body,
            });
        }
    }
    let fixed: f32 = BUMPER_F_KG
        + BUMPER_R_KG
        + 2.0 * (ACCORD_DOOR_F + ACCORD_DOOR_R + ACCORD_FENDER)
        + ACCORD_HOOD
        + ACCORD_DECKLID
        + ACCORD_ENGINE_KG
        + ACCORD_TRANSAXLE_KG
        + ACCORD_BIW
        + 4.0 * RIM_NODES as f32 * RIM_NODE_KG
        + c.named.iter().map(|m| m.kg).sum::<f32>()
        + 4.0 * WHEEL_KG;
    // What is left of the kerb is a front-drive car's front-end module (air
    // conditioning condenser and compressor, lamps, horns, washer and coolant
    // fluids, brake booster and ABS unit), ahead of the dash.
    let module = KERB - fixed;
    assert!(module >= 0.0, "masses exceed the kerb by {} kg", -module);
    c.named.push(Named {
        name: "front_end_module",
        kg: module,
        at: Vec3::new(0.0, -0.40, 0.62),
        radius: 0.55,
        part: body,
    });
    let body_kg = ACCORD_BIW;
    c.part_mass = [
        (Part::Body, body_kg),
        (Part::Engine, ACCORD_ENGINE_KG + ACCORD_TRANSAXLE_KG),
        (Part::DoorFL, ACCORD_DOOR_F),
        (Part::DoorFR, ACCORD_DOOR_F),
        (Part::DoorRL, ACCORD_DOOR_R),
        (Part::DoorRR, ACCORD_DOOR_R),
        (Part::Hood, ACCORD_HOOD),
        (Part::Trunk, ACCORD_DECKLID),
        (Part::FenderL, ACCORD_FENDER),
        (Part::FenderR, ACCORD_FENDER),
        (Part::BumperF, BUMPER_F_KG),
        (Part::BumperR, BUMPER_R_KG),
    ]
    .into_iter()
    .collect();
    c.assign_masses();

    // Wheels and reference nodes.
    for (name, w, s, y) in [
        ("FL", "F", 1.0f32, AXLE_F),
        ("FR", "F", -1.0, AXLE_F),
        ("RL", "R", 1.0, AXLE_R),
        ("RR", "R", -1.0, AXLE_R),
    ] {
        let x = lr(s);
        let _ = y;
        let (n1, n2) = (c.id(&format!("hub{w}i{x}")), c.id(&format!("hub{w}o{x}")));
        let arm = if w == "F" {
            c.id(&format!("twr{x}"))
        } else {
            c.id(&format!("wh{x}"))
        };
        c.wheels.push(Wheel {
            name,
            n1,
            n2,
            arm,
            dir: s,
        });
    }
    c.refs = [c.id("tn1"), c.id("tn2"), c.id("si1l"), c.id("rfC1")];
    let hub = hub_extra(&c);
    c.assign_springs(&hub);
    // Yield by zone: a member's webs and ties are its section as much as its
    // chords, so the front crush zone's webs and cross ties take the crush
    // multiple and the cabin's (the safety cage: DP980/PHS) the cabin multiple.
    for b in &mut c.beams {
        if !matches!(b.part, Part::Body | Part::BumperF | Part::BumperR) || !b.deform.is_finite() {
            continue;
        }
        let mid = 0.5 * (c.nodes[b.a].p + c.nodes[b.b].p);
        // Crushed sheet steel stacks at about a third of its length (a folded
        // crush can's solid length): JBeam's deformLimit on the crush zones'
        // beams, so a crushed bay stacks instead of inverting.
        if (mid.y < fw_y - 0.05 || mid.y > 2.9)
            && !matches!(b.role, Role::Sheet | Role::Bend | Role::Tie | Role::Backing)
        {
            b.deform_limit = STACK_LIMIT;
        }
        if b.part != body {
            continue;
        }
        let web = matches!(
            b.role,
            Role::Web
                | Role::Cross
                | Role::Firewall
                | Role::Floor
                | Role::Tunnel
                | Role::RoofBow
                | Role::Sheet
        );
        if !web {
            continue;
        }
        if mid.y < fw_y - 0.05 {
            b.deform *= crush_mult();
        } else if mid.y < 2.9 {
            b.deform *= cabin_mult();
        } else {
            b.deform *= crush_mult();
        }
    }
    c
}

/// The raycast wheels' mass on their hub nodes (half a wheel each).
pub fn hub_extra(c: &Car) -> HashMap<usize, f32> {
    let mut h = HashMap::new();
    for w in &c.wheels {
        *h.entry(w.n1).or_insert(0.0) += 0.5 * WHEEL_KG;
        *h.entry(w.n2).or_insert(0.0) += 0.5 * WHEEL_KG;
    }
    h
}
