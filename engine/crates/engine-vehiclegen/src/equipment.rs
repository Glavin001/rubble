//! What a real car has besides its body, running gear and seats: the engine
//! bay's equipment, the brakes' and suspension's visible members, the
//! controls, the body's hardware and the interior's trim.
//!
//! Each is a [`MechPart`] defined once, as in [`crate::mechanical`]: a closed
//! solid, and either its own nodes with real masses, internal beams and
//! mounts in named break groups with finite strengths, or (a part that is
//! fixed inside or onto another: a caliper on its hub, a door's intrusion
//! beam) no nodes, a solid that follows its host's, and the mass it accounts
//! for inside the host's listed mass (`carried_kg`).
//!
//! Masses are typical mid-size sedan component masses. A part with nodes
//! takes its mass out of the shell's fixed-trim allowance
//! ([`crate::structure::Structure::fixed_trim`]), so the kerb mass stays.

use glam::DVec3;

use crate::frame::Frame;
use crate::mechanical::{cyl, sh, vals, BeamKind, Gear, Internal, MechPart, Mount, Vals, INF};
use crate::mesh::{box_, closed_lathe, rbox, Mat, Mesh};
use crate::parts::rot_to;
use crate::structure::{box_stations, cyl_along, Structure};

/// A solid of revolution about `axis` from `at`: profile points are
/// (radius, distance along the axis).
pub(crate) fn lathe_at(
    profile: &[(f64, f64)],
    segs: usize,
    mat: Mat,
    axis: DVec3,
    at: DVec3,
) -> Mesh {
    let mut m = closed_lathe(profile, segs, mat);
    m.transform(Some(rot_to(axis.normalize())), at);
    m
}

/// A drum of radius `r` from `a` to `b`.
pub(crate) fn drum(a: DVec3, b: DVec3, r: f64, segs: usize, mat: Mat) -> Mesh {
    let l = (b - a).length();
    lathe_at(
        &[(0.0, 0.0), (r, 0.0), (r, l), (0.0, l)],
        segs,
        mat,
        b - a,
        a,
    )
}

/// A flat strap from `a` to `b`: `w` wide along `across`, `t` thick.
pub(crate) fn strap(a: DVec3, b: DVec3, across: DVec3, w: f64, t: f64, mat: Mat) -> Mesh {
    let up = across.cross(b - a).normalize();
    let mut m = box_stations(&[(a, across, up, w, t), (b, across, up, w, t)]);
    m.set_mat(mat);
    m
}

impl MechPart {
    /// Four nodes on alternate corners of the box `c` ± `h` (a tetrahedron:
    /// the least that holds a solid's position and attitude), `kg` between
    /// them, fully braced.
    pub(crate) fn tetra(&mut self, c: DVec3, h: DVec3, kg: f64, v: Vals) {
        for (i, s) in [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]
        .into_iter()
        .enumerate()
        {
            self.node(
                &format!("{}", i + 1),
                c + DVec3::new(s.0 * h.x, s.1 * h.y, s.2 * h.z),
                kg / 4.0,
            );
        }
        self.internal.push(Internal {
            pairs: self.complete(),
            v,
            deform_group: None,
        });
    }
    /// A mount from all the part's nodes to the named shell nodes (`sh_…`
    /// without the prefix).
    pub(crate) fn hold(&mut self, group: &str, to: &[&str], beams: usize, kind: BeamKind, v: Vals) {
        let to = to.iter().map(|n| format!("sh_{n}")).collect();
        self.mounts.push(Mount {
            group: group.into(),
            from: self.names(),
            to,
            beams,
            kind,
            v,
            precomp: 1.0,
            to_part: None,
        });
    }
}

/// The belt's run between two pulleys (centre, radius) in the plane z: the
/// outer tangent on the side of `out`.
fn belt_run(c1: DVec3, r1: f64, c2: DVec3, r2: f64, out: DVec3, w: f64) -> Mesh {
    let d = c2 - c1;
    let l = d.length();
    let d = d / l;
    let mut n = DVec3::Z.cross(d);
    if n.dot(out) < 0.0 {
        n = -n;
    }
    let s = (r1 - r2) / l;
    let u = d * s + n * (1.0 - s * s).sqrt();
    strap(c1 + u * r1, c2 + u * r2, DVec3::Z, w, 0.004, Mat::Rubber)
}

/// The engine's solid: block, head, cam cover with its coils and filler cap,
/// sump, intake plenum with runners and throttle body, exhaust manifold,
/// crank and water-pump pulleys, alternator, compressor and their belt, and
/// the two mount ears. Reference coordinates: the block is x ± `ex`,
/// z `ez0`..`ez1`; `zm` the ears' station.
pub(crate) fn engine_mesh(f: &Frame, ex: f64, ez0: f64, ez1: f64) -> Mesh {
    let r = |x: f64, y: f64, z: f64| f.p(x, y, z);
    let (zc, hz) = (0.5 * (ez0 + ez1), 0.5 * (ez1 - ez0));
    let mut m = Mesh::new();
    m.add(&rbox(
        r(0.0, 0.50, zc),
        DVec3::new(f.x(ex), 0.115, hz),
        0.02,
        2,
        Mat::MetalBare,
    ));
    m.add(&rbox(
        r(0.0, 0.660, zc),
        DVec3::new(f.x(0.16), 0.045, hz - 0.01),
        0.015,
        2,
        Mat::MetalBare,
    ));
    m.add(&rbox(
        r(0.0, 0.722, zc),
        DVec3::new(f.x(0.125), 0.020, hz - 0.03),
        0.012,
        2,
        Mat::TrimBlack,
    ));
    m.add(&drum(
        r(0.05, 0.740, zc + 0.22),
        r(0.05, 0.756, zc + 0.22),
        0.022,
        12,
        Mat::TrimBlack,
    ));
    for k in [-0.21, -0.07, 0.07, 0.21] {
        m.add(&box_(
            r(-0.02, 0.746, zc + k),
            DVec3::new(0.020, 0.008, 0.030),
            Mat::TrimBlack,
            None,
        ));
    }
    m.add(&rbox(
        r(0.0, 0.358, zc + 0.03),
        DVec3::new(f.x(0.15), 0.025, hz - 0.06),
        0.012,
        2,
        Mat::MetalBare,
    ));
    // intake: plenum, four runners into the head, the throttle body in front
    let xi = f.x(0.245);
    m.add(&rbox(
        DVec3::new(-xi, f.y(0.56), f.z(zc)),
        DVec3::new(0.040, 0.045, hz - 0.05),
        0.02,
        2,
        Mat::TrimBlack,
    ));
    let xh = f.x(0.16);
    for k in [-0.21, -0.07, 0.07, 0.21] {
        m.add(&cyl(
            DVec3::new(-xi + 0.03, f.y(0.585), f.z(zc + k)),
            DVec3::new(-xh + 0.01, f.y(0.655), f.z(zc + k)),
            0.017,
        ));
    }
    m.add(&drum(
        DVec3::new(-xi, f.y(0.56), f.z(zc + hz - 0.055)),
        DVec3::new(-xi, f.y(0.56), f.z(zc + hz - 0.015)),
        0.034,
        12,
        Mat::MetalBare,
    ));
    // exhaust manifold: four stubs into a log, the collector down to the
    // downpipe's flange
    let xm = f.x(0.235);
    m.add(&cyl(
        DVec3::new(xm, f.y(0.60), f.z(zc - 0.24)),
        DVec3::new(xm, f.y(0.60), f.z(zc + 0.18)),
        0.024,
    ));
    for k in [-0.21, -0.09, 0.03, 0.15] {
        m.add(&cyl(
            DVec3::new(xh - 0.01, f.y(0.645), f.z(zc + k)),
            DVec3::new(xm, f.y(0.60), f.z(zc + k)),
            0.018,
        ));
    }
    m.add(&cyl(
        DVec3::new(xm, f.y(0.60), f.z(zc - 0.035)),
        DVec3::new(xm + 0.020, f.y(0.508), f.z(zc - 0.035)),
        0.024,
    ));
    // accessory drive on the front face
    let zf = f.z(ez1);
    let at = |x: f64, y: f64, dz: f64| DVec3::new(x, f.y(y), zf + dz);
    let (crank, pump, alt, comp) = (
        at(0.0, 0.42, 0.0),
        at(0.0, 0.585, 0.0),
        at(-f.x(0.255), 0.44, 0.0),
        at(f.x(0.255), 0.44, 0.0),
    );
    let (r_crank, r_pump, r_alt, r_comp) = (0.070, 0.050, 0.028, 0.045);
    let fwd = DVec3::Z;
    for (c, rr) in [(crank, r_crank), (pump, r_pump)] {
        m.add(&lathe_at(
            &[
                (0.0, 0.0),
                (rr + 0.004, 0.0),
                (rr + 0.004, 0.003),
                (rr, 0.005),
                (rr, 0.023),
                (rr + 0.004, 0.025),
                (rr + 0.004, 0.028),
                (0.0, 0.028),
            ],
            20,
            Mat::MetalBare,
            fwd,
            c,
        ));
    }
    // (their bodies hang beside the block on brackets, their pulleys in the
    // belt's plane)
    for (c, rr, body) in [(alt, r_alt, 0.058), (comp, r_comp, 0.052)] {
        m.add(&lathe_at(
            &[
                (0.0, -0.100),
                (body - 0.008, -0.100),
                (body, -0.090),
                (body, -0.014),
                (body - 0.010, -0.004),
                (0.0, -0.004),
            ],
            16,
            Mat::MetalBare,
            fwd,
            c,
        ));
        m.add(&lathe_at(
            &[
                (0.0, -0.004),
                (rr + 0.004, -0.004),
                (rr + 0.004, 0.003),
                (rr, 0.005),
                (rr, 0.023),
                (rr + 0.004, 0.025),
                (rr + 0.004, 0.028),
                (0.0, 0.028),
            ],
            16,
            Mat::MetalBare,
            fwd,
            c,
        ));
        let s = c.x.signum();
        m.add(&box_(
            DVec3::new(s * (f.x(ex) + 0.012), c.y, zf - 0.055),
            DVec3::new(0.020, 0.030, 0.030),
            Mat::MetalBare,
            None,
        ));
    }
    let zb = DVec3::Z * 0.014;
    for (a, ra, b, rb, out) in [
        (crank, r_crank, alt, r_alt, -DVec3::Y),
        (alt, r_alt, pump, r_pump, DVec3::new(-1.0, 1.0, 0.0)),
        (pump, r_pump, comp, r_comp, DVec3::new(1.0, 1.0, 0.0)),
        (comp, r_comp, crank, r_crank, -DVec3::Y),
    ] {
        m.add(&belt_run(a + zb, ra, b + zb, rb, out, 0.016));
    }
    // the mount ears
    let zm = 0.5 * (ez0 + ez1);
    for sgn in [1.0, -1.0] {
        m.add(&box_(
            r(sgn * 0.262, 0.40, zm),
            DVec3::new(0.058, 0.02, 0.04),
            Mat::MetalBare,
            None,
        ));
    }
    m
}

/// The parts this module adds to the sedan's ([`crate::mechanical::sedan_parts`]).
pub(crate) fn sedan_equipment(g: &Gear, st: &Structure, out: &mut Vec<MechPart>) {
    let n0 = out.len();
    engine_bay(g, st, out);
    for p in &mut out[n0..] {
        p.from_trim = !p.nodes.is_empty();
    }
}

// ---- 1. the engine bay ---------------------------------------------------------------------

fn engine_bay(g: &Gear, st: &Structure, out: &mut Vec<MechPart>) {
    let f = &g.f;
    let r = |x: f64, y: f64, z: f64| f.p(x, y, z);
    let af = g.axle_f;
    let soft = |deform: f64| vals(0.0, 40.0, deform, INF);

    // battery on its tray over the left rail, ahead of the strut tower
    {
        let mut p = MechPart::new("battery", "electrical");
        let (c, h) = (r(0.36, 0.617, af + 0.405), DVec3::new(0.0875, 0.095, 0.120));
        p.mesh.add(&rbox(c, h, 0.008, 2, Mat::TrimBlack));
        // tray, hold-down bar and the two posts
        p.mesh.add(&box_(
            c - DVec3::Y * (h.y + 0.006),
            DVec3::new(h.x + 0.010, 0.004, h.z + 0.010),
            Mat::MetalBare,
            None,
        ));
        p.mesh.add(&box_(
            c + DVec3::Y * (h.y + 0.005),
            DVec3::new(h.x + 0.006, 0.004, 0.010),
            Mat::MetalBare,
            None,
        ));
        for (dz, mat) in [(0.085, Mat::MetalBare), (-0.085, Mat::MetalBare)] {
            p.mesh.add(&drum(
                c + DVec3::new(0.045, h.y, dz),
                c + DVec3::new(0.045, h.y + 0.018, dz),
                0.009,
                10,
                mat,
            ));
        }
        p.tetra(c, h, 17.0, soft(4000.0));
        p.hold(
            "battery",
            &["a8_fl", "a2_ul", "a4_l", "a7_l", "a3_ul"],
            8,
            BeamKind::Normal,
            vals(0.0, 100.0, 3000.0, 9000.0),
        );
        out.push(p);
    }
    // air box over the right rail, its duct to the throttle body, its
    // snorkel to the radiator support
    {
        let mut p = MechPart::new("airbox", "engine_bay");
        let (c, h) = (
            r(-0.365, 0.600, af + 0.405),
            DVec3::new(0.080, 0.075, 0.110),
        );
        p.mesh.add(&rbox(c, h, 0.015, 2, Mat::TrimBlack));
        p.mesh.add(&box_(
            c + DVec3::Y * 0.012,
            DVec3::new(h.x + 0.004, 0.004, h.z + 0.004),
            Mat::TrimBlack,
            None,
        ));
        let xi = -f.x(0.245);
        let duct = [
            c + DVec3::new(h.x - 0.01, 0.0, 0.03),
            DVec3::new(xi + 0.01, c.y - 0.01, c.z + 0.035),
            DVec3::new(xi, f.y(0.56), c.z + 0.005),
            DVec3::new(xi, f.y(0.56), f.z(af + 0.385 - 0.009)),
        ];
        p.mesh.add(&cyl_along(&duct, 0.032, 10, Mat::Rubber));
        p.mesh.add(&cyl_along(
            &[
                c + DVec3::new(-0.02, 0.02, h.z - 0.01),
                c + DVec3::new(-0.02, 0.02, h.z + 0.020),
            ],
            0.028,
            10,
            Mat::TrimBlack,
        ));
        p.tetra(c, h, 2.4, soft(600.0));
        p.hold(
            "airbox",
            &["a8_fr", "a2_ur", "a4_r", "a7_r", "a3_ur"],
            8,
            BeamKind::Normal,
            vals(0.0, 40.0, 800.0, 2500.0),
        );
        out.push(p);
    }
    // the cooling pack round the radiator: fan and shroud behind it, the
    // condenser in front
    let a3 = sh(st, "sh_a3_uc");
    let z_rad = a3.z + 0.062;
    let (hx, y0, y1) = (f.x(0.36), sh(st, "sh_a3_dc").y + 0.035, a3.y - 0.045);
    let yc = 0.5 * (y0 + y1);
    {
        let mut p = MechPart::new("fan_shroud", "cooling");
        let (zs, hy) = (z_rad - 0.037, 0.5 * (y1 - y0) - 0.008);
        // the frame, the motor on four arms, the ring and seven blades
        for sy in [1.0, -1.0] {
            p.mesh.add(&box_(
                DVec3::new(0.0, yc + sy * (hy - 0.012), zs),
                DVec3::new(hx - 0.005, 0.012, 0.015),
                Mat::TrimBlack,
                None,
            ));
        }
        for sx in [1.0, -1.0] {
            p.mesh.add(&box_(
                DVec3::new(sx * (hx - 0.017), yc, zs),
                DVec3::new(0.012, hy - 0.024, 0.015),
                Mat::TrimBlack,
                None,
            ));
        }
        let hub = DVec3::new(0.0, yc, zs);
        let rf = (hy - 0.030).min(0.19);
        for k in 0..4 {
            let a = std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2 * k as f64;
            let d = DVec3::new(a.cos(), a.sin(), 0.0);
            let reach = (hx - 0.03).min((hy - 0.03) / d.y.abs());
            p.mesh.add(&strap(
                hub + d * 0.03 - DVec3::Z * 0.010,
                hub + d * reach - DVec3::Z * 0.010,
                DVec3::Z,
                0.008,
                0.014,
                Mat::TrimBlack,
            ));
        }
        p.mesh.add(&lathe_at(
            &[
                (0.0, -0.060),
                (0.045, -0.060),
                (0.045, -0.006),
                (0.0, -0.006),
            ],
            14,
            Mat::MetalBare,
            DVec3::Z,
            hub,
        ));
        p.mesh.add(&lathe_at(
            &[(0.0, -0.004), (0.050, -0.004), (0.050, 0.013), (0.0, 0.013)],
            14,
            Mat::TrimBlack,
            DVec3::Z,
            hub,
        ));
        p.mesh.add(&lathe_at(
            &[
                (rf, -0.012),
                (rf + 0.006, -0.012),
                (rf + 0.006, 0.014),
                (rf, 0.014),
            ],
            28,
            Mat::TrimBlack,
            DVec3::Z,
            hub,
        ));
        for k in 0..7 {
            let a = std::f64::consts::TAU * k as f64 / 7.0;
            let d = DVec3::new(a.cos(), a.sin(), 0.0);
            let across = (DVec3::Z.cross(d) + DVec3::Z * 0.45).normalize();
            p.mesh.add(&strap(
                hub + d * 0.046 + DVec3::Z * 0.004,
                hub + d * (rf - 0.004) + DVec3::Z * 0.004,
                across,
                0.030,
                0.003,
                Mat::TrimBlack,
            ));
        }
        for (tg, x, y) in [
            ("ul", hx - 0.02, yc + hy),
            ("ur", -hx + 0.02, yc + hy),
            ("dl", hx - 0.02, yc - hy),
            ("dr", -hx + 0.02, yc - hy),
        ] {
            p.node(tg, DVec3::new(x, y, zs), 1.05);
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: soft(1500.0),
            deform_group: None,
        });
        p.mounts.push(Mount {
            group: "fan_shroud".into(),
            from: p.names(),
            to: ["ul", "ur", "dl", "dr"]
                .iter()
                .map(|t| format!("rad_{t}"))
                .collect(),
            beams: 8,
            kind: BeamKind::Normal,
            v: vals(0.0, 40.0, 1000.0, 3000.0),
            precomp: 1.0,
            to_part: None,
        });
        out.push(p);
    }
    {
        let mut p = MechPart::new("condenser", "cooling");
        let (zk, hy, hxc) = (z_rad + 0.040, 0.5 * (y1 - y0) - 0.012, hx - 0.025);
        p.mesh.add(&box_(
            DVec3::new(0.0, yc, zk),
            DVec3::new(hxc, hy, 0.008),
            Mat::MetalBare,
            None,
        ));
        // header tanks and the receiver-drier
        for sx in [1.0, -1.0] {
            p.mesh.add(&box_(
                DVec3::new(sx * (hxc + 0.008), yc, zk),
                DVec3::new(0.008, hy, 0.011),
                Mat::MetalBare,
                None,
            ));
        }
        p.mesh.add(&drum(
            DVec3::new(hxc + 0.008, yc - hy + 0.02, zk + 0.034),
            DVec3::new(hxc + 0.008, yc + hy - 0.04, zk + 0.034),
            0.020,
            10,
            Mat::MetalBare,
        ));
        for (tg, x, y) in [
            ("ul", hxc, yc + hy),
            ("ur", -hxc, yc + hy),
            ("dl", hxc, yc - hy),
            ("dr", -hxc, yc - hy),
        ] {
            p.node(tg, DVec3::new(x, y, zk), 0.65);
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: soft(1200.0),
            deform_group: Some(("condenser".into(), 0.05)),
        });
        p.mounts.push(Mount {
            group: "condenser".into(),
            from: p.names(),
            to: ["ul", "ur", "dl", "dr"]
                .iter()
                .map(|t| format!("rad_{t}"))
                .collect(),
            beams: 8,
            kind: BeamKind::Normal,
            v: vals(0.0, 40.0, 1000.0, 3000.0),
            precomp: 1.0,
            to_part: None,
        });
        out.push(p);
    }
    // reservoirs: coolant on the right, screen wash on the left
    {
        let mut p = MechPart::new("coolant_reservoir", "cooling");
        let (c, h) = (r(-0.40, 0.630, af + 0.185), DVec3::new(0.055, 0.070, 0.060));
        p.mesh.add(&rbox(c, h, 0.02, 2, Mat::InteriorPlastic));
        p.mesh.add(&drum(
            c + DVec3::Y * (h.y - 0.002),
            c + DVec3::Y * (h.y + 0.020),
            0.024,
            12,
            Mat::TrimBlack,
        ));
        p.tetra(c, h, 1.6, soft(400.0));
        p.hold(
            "coolant_reservoir",
            &["a7_r", "a6_r", "a8_fr", "a8_mr", "a4_r"],
            6,
            BeamKind::Normal,
            vals(0.0, 30.0, 500.0, 1500.0),
        );
        out.push(p);
    }
    {
        let mut p = MechPart::new("washer_reservoir", "engine_bay");
        let (c, h) = (r(0.40, 0.620, af + 0.195), DVec3::new(0.055, 0.070, 0.070));
        p.mesh.add(&rbox(c, h, 0.02, 2, Mat::InteriorPlastic));
        p.mesh.add(&cyl_along(
            &[
                c + DVec3::new(0.0, h.y - 0.005, 0.03),
                c + DVec3::new(0.02, h.y + 0.07, 0.04),
            ],
            0.016,
            10,
            Mat::InteriorPlastic,
        ));
        p.mesh.add(&drum(
            c + DVec3::new(0.02, h.y + 0.07, 0.04),
            c + DVec3::new(0.023, h.y + 0.082, 0.0415),
            0.021,
            12,
            Mat::TrimBlack,
        ));
        p.tetra(c, h, 3.0, soft(500.0));
        p.hold(
            "washer_reservoir",
            &["a7_l", "a6_l", "a8_fl", "a8_ml", "a4_l"],
            6,
            BeamKind::Normal,
            vals(0.0, 30.0, 700.0, 2200.0),
        );
        out.push(p);
    }
    // brake booster and master cylinder on the firewall, driver's side
    let z_fw = sh(st, "sh_b1_ol").z;
    {
        let mut p = MechPart::new("brake_booster", "controls");
        let c = DVec3::new(f.x(0.38), f.y(0.665), z_fw + 0.008);
        p.mesh.add(&lathe_at(
            &[
                (0.0, 0.0),
                (0.095, 0.0),
                (0.118, 0.018),
                (0.118, 0.085),
                (0.090, 0.112),
                (0.030, 0.118),
                (0.0, 0.118),
            ],
            24,
            Mat::TrimBlack,
            DVec3::Z,
            c,
        ));
        p.mesh.add(&drum(
            c + DVec3::Z * 0.118,
            c + DVec3::Z * 0.245,
            0.022,
            12,
            Mat::MetalBare,
        ));
        p.mesh.add(&rbox(
            c + DVec3::new(0.0, 0.050, 0.185),
            DVec3::new(0.035, 0.028, 0.050),
            0.010,
            2,
            Mat::InteriorPlastic,
        ));
        p.mesh.add(&drum(
            c + DVec3::new(0.0, 0.076, 0.185),
            c + DVec3::new(0.0, 0.090, 0.185),
            0.022,
            12,
            Mat::TrimBlack,
        ));
        p.tetra(
            c + DVec3::Z * 0.085,
            DVec3::new(0.085, 0.085, 0.075),
            4.4,
            soft(1500.0),
        );
        p.hold(
            "brake_booster",
            &["b1_ol", "b1_il", "b2_ol", "b2_il", "a8_ml"],
            8,
            BeamKind::Normal,
            vals(0.0, 60.0, 2000.0, 6000.0),
        );
        out.push(p);
    }
    // fuse and relay box by the firewall on the right
    {
        let mut p = MechPart::new("fuse_box", "electrical");
        let (c, h) = (
            DVec3::new(-f.x(0.375), f.y(0.690), z_fw + 0.175),
            DVec3::new(0.075, 0.040, 0.085),
        );
        p.mesh.add(&rbox(c, h, 0.008, 2, Mat::TrimBlack));
        p.mesh.add(&box_(
            c + DVec3::Y * (h.y + 0.003),
            DVec3::new(h.x - 0.008, 0.004, h.z - 0.008),
            Mat::TrimBlack,
            None,
        ));
        p.tetra(c, h, 1.4, soft(300.0));
        p.hold(
            "fuse_box",
            &["b1_or", "b1_ir", "a8_mr", "a7_r", "b2_or"],
            6,
            BeamKind::Normal,
            vals(0.0, 30.0, 400.0, 1200.0),
        );
        out.push(p);
    }
    // strut-tower brace: a tube from tower to tower over the engine
    {
        let mut p = MechPart::new("tower_brace", "suspension");
        let (tl, tr) = (sh(st, "sh_a6_l"), sh(st, "sh_a6_r"));
        let up = |x: f64, dy: f64| DVec3::new(x, tl.y + dy, tl.z);
        p.mesh.add(&cyl_along(
            &[
                up(tr.x + 0.01, 0.022),
                up(0.7 * tr.x, 0.036),
                up(0.0, 0.040),
                up(0.7 * tl.x, 0.036),
                up(tl.x - 0.01, 0.022),
            ],
            0.014,
            8,
            Mat::MetalBare,
        ));
        for t in [tl, tr] {
            p.mesh.add(&box_(
                t + DVec3::Y * 0.009,
                DVec3::new(0.045, 0.004, 0.045),
                Mat::MetalBare,
                None,
            ));
            p.mesh.add(&box_(
                t + DVec3::Y * 0.020,
                DVec3::new(0.016, 0.010, 0.016),
                Mat::MetalBare,
                None,
            ));
        }
        p.node("l", up(tl.x, 0.022), 1.2);
        p.node("r", up(tr.x, 0.022), 1.2);
        p.internal.push(Internal {
            pairs: vec![(0, 1)],
            v: vals(0.0, 60.0, 30e3, INF),
            deform_group: None,
        });
        for (s, i) in [("l", 0), ("r", 1)] {
            p.mounts.push(Mount {
                group: format!("tower_brace_{s}"),
                from: vec![p.nodes[i].name.clone()],
                to: vec![
                    format!("sh_a6_{s}"),
                    format!("sh_a8_f{s}"),
                    format!("sh_a8_m{s}"),
                    format!("sh_a7_{s}"),
                ],
                beams: 4,
                kind: BeamKind::Normal,
                v: vals(0.0, 60.0, 12e3, 32e3),
                precomp: 1.0,
                to_part: None,
            });
        }
        out.push(p);
    }
}
