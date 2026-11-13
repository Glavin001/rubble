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
        DVec3::new(xm + 0.020, f.y(0.518), f.z(zc - 0.035)),
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
    body_hardware(g, st, out);
    running_gear(g, st, out);
    controls(g, st, out);
    interior(g, st, out);
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

// ---- 5. the body's hardware ----------------------------------------------------------------

/// A point of the body side at (z, y) and `d` in from the skin, level with
/// the ground.
fn inside(g: &Gear, z: f64, y: f64, sg: f64, d: f64) -> DVec3 {
    let (p, _) = crate::parts::surface_point(&g.tube, z, y, sg, (1.5, 5.5));
    p - DVec3::X * (sg * d)
}

/// The top surface's height at (z, x): hood, roof, deck.
fn top_y(g: &Gear, z: f64, x: f64) -> f64 {
    let v = g.tube.v_at_x(z, x.abs(), 6.0, 9.0);
    g.tube.eval(z, v).y
}

/// A box given by its faces along the three axes.
fn span(x: (f64, f64), y: (f64, f64), z: (f64, f64), mat: Mat) -> Mesh {
    box_(
        DVec3::new(0.5 * (x.0 + x.1), 0.5 * (y.0 + y.1), 0.5 * (z.0 + z.1)),
        DVec3::new(
            0.5 * (x.1 - x.0).abs(),
            0.5 * (y.1 - y.0).abs(),
            0.5 * (z.1 - z.0).abs(),
        ),
        mat,
        None,
    )
}

impl MechPart {
    /// A mount from the named nodes of this part to a body panel's own
    /// lattice.
    fn hold_panel(&mut self, group: &str, from: Vec<String>, panel: &str, beams: usize, v: Vals) {
        self.mounts.push(Mount {
            group: group.into(),
            from,
            to: Vec::new(),
            beams,
            kind: BeamKind::Normal,
            v,
            precomp: 1.0,
            to_part: Some(panel.into()),
        });
    }
    fn hold_from(&mut self, group: &str, from: Vec<String>, to: &[String], beams: usize, v: Vals) {
        self.mounts.push(Mount {
            group: group.into(),
            from,
            to: to.to_vec(),
            beams,
            kind: BeamKind::Normal,
            v,
            precomp: 1.0,
            to_part: None,
        });
    }
}

fn shn(tags: &[String]) -> Vec<String> {
    tags.iter().map(|t| format!("sh_{t}")).collect()
}

fn body_hardware(g: &Gear, st: &Structure, out: &mut Vec<MechPart>) {
    let f = &g.f;
    let steel = Mat::MetalBare;
    // (reference z of the shut lines: the front door's front edge, the
    // B-pillar, the rear door's rear edge)
    let (z_df, z_b, z_dr) = (0.88, -0.22, -1.02);
    for (s, sg) in [("L", 1.0), ("R", -1.0)] {
        let sl = s.to_lowercase();
        // ---- door hinges: a leaf on the pillar, a pin, a leaf on the door
        for (door, zl, pillar) in [("F", f.z(z_df), false), ("R", f.z(z_b), true)] {
            for (nm, y) in [("upper", 0.82), ("lower", 0.46)] {
                let mut p = MechPart::new(&format!("hinge_door_{door}{s}_{nm}"), "hardware");
                let yy = f.y(y);
                let at = |dz: f64, d: f64| {
                    let q = inside(g, zl + dz, yy, sg, d);
                    DVec3::new(q.x, yy, zl + dz)
                };
                let slab = |z0: f64, z1: f64, d0: f64, d1: f64, hy: f64| {
                    let (a, b) = (at(0.5 * (z0 + z1), d0), at(0.5 * (z0 + z1), d1));
                    span((a.x, b.x), (yy - hy, yy + hy), (zl + z0, zl + z1), steel)
                };
                // (the hinge pillar's outer face is 64 mm in and ahead of the
                // shut line; the B-pillar's is 72 mm in, either side of it)
                // (measured in from the skin, level: the door's inner face is
                // within 57 mm, the hinge pillar's outer face at 61 mm and
                // ahead of the shut line, the B-pillar's at 69 mm either
                // side of it; the door's edge is 28 mm deep)
                let pin = if pillar {
                    p.mesh.add(&slab(
                        0.003,
                        0.031,
                        0.059,
                        if nm == "lower" { 0.065 } else { 0.0675 },
                        0.035,
                    ));
                    p.mesh.add(&slab(-0.006, 0.010, 0.059, 0.063, 0.020));
                    -0.004
                } else {
                    p.mesh.add(&slab(0.008, 0.064, 0.050, 0.0595, 0.035));
                    p.mesh.add(&slab(0.002, 0.012, 0.052, 0.058, 0.020));
                    0.004
                };
                let c = at(pin, 0.0555);
                p.mesh.add(&drum(
                    c - DVec3::Y * 0.036,
                    c + DVec3::Y * 0.036,
                    0.004,
                    8,
                    steel,
                ));
                p.mesh
                    .add(&slab(pin - 0.079, pin - 0.010, 0.0585, 0.0625, 0.028));
                p.mesh
                    .add(&slab(pin - 0.012, pin + 0.002, 0.054, 0.0595, 0.028));
                let (a, b) = (
                    p.node("u", c + DVec3::Y * 0.03, 0.4),
                    p.node("d", c - DVec3::Y * 0.03, 0.4),
                );
                p.internal.push(Internal {
                    pairs: vec![(0, 1)],
                    v: vals(0.0, 50.0, 20e3, INF),
                    deform_group: None,
                });
                let to: Vec<String> = match (pillar, nm) {
                    (false, "upper") => vec![
                        format!("b2_o{sl}"),
                        format!("c2_1{sl}"),
                        format!("b1_o{sl}"),
                        format!("a8_m{sl}"),
                    ],
                    (false, _) => vec![
                        format!("c2_1{sl}"),
                        format!("c1_1o{sl}"),
                        format!("b2_o{sl}"),
                        format!("b1_o{sl}"),
                    ],
                    (true, "upper") => vec![
                        format!("d2_b{sl}"),
                        format!("c2_3{sl}"),
                        format!("d2_t{sl}"),
                        format!("c1_3o{sl}"),
                    ],
                    (true, _) => vec![
                        format!("c2_3{sl}"),
                        format!("c1_3o{sl}"),
                        format!("d2_b{sl}"),
                        format!("c2_2{sl}"),
                    ],
                };
                let grp = format!("hinge_door_{}{sl}_{nm}", door.to_lowercase());
                let v = vals(0.0, 100.0, 9e3, 16e3);
                p.hold_from(
                    &format!("{grp}_body"),
                    vec![a.clone(), b.clone()],
                    &shn(&to),
                    4,
                    v,
                );
                p.hold_panel(
                    &format!("{grp}_door"),
                    vec![a, b],
                    &format!("door_{door}{s}"),
                    4,
                    v,
                );
                out.push(p);
            }
        }
        // ---- door latches: the latch in the door's rear edge, its striker
        // on the pillar behind
        // (the rear door's rear edge leans back round the wheel arch: under
        // the shoulder it is 0.296 m ahead of the axle, and leaning)
        for (door, zl, y, shell) in [
            ("F", f.z(z_b), 0.64, false),
            ("R", g.zr + 0.296, 0.85, true),
        ] {
            let mut p = MechPart::new(&format!("latch_door_{door}{s}"), "hardware");
            let yy = f.y(y);
            let at = |dz: f64, d: f64| {
                let q = inside(g, zl + dz, yy, sg, d);
                DVec3::new(q.x, yy, zl + dz)
            };
            // (the door is 28 mm deep at its edge and 50 mm from 35 mm in)
            let (b0, b1, bd, bh) = if shell {
                (0.034, 0.059, 0.030, 0.020)
            } else {
                (0.020, 0.050, 0.034, 0.035)
            };
            let (a, b) = (at(0.5 * (b0 + b1), 0.010), at(0.5 * (b0 + b1), bd));
            p.mesh.add(&span(
                (a.x, b.x),
                (yy - bh, yy + bh),
                (zl + b0, zl + b1),
                steel,
            ));
            // the striker: a plate and a loop standing towards the door
            let (d0, d1, dl) = if shell {
                (0.038, 0.0415, 0.046)
            } else {
                (0.064, 0.0675, 0.056)
            };
            let (z0, z1) = if shell {
                (-0.053, -0.029)
            } else {
                (0.006, 0.030)
            };
            let (a, b) = (at(0.5 * (z0 + z1), d0), at(0.5 * (z0 + z1), d1));
            p.mesh.add(&span(
                (a.x, b.x),
                (yy - 0.025, yy + 0.025),
                (zl + z0, zl + z1),
                steel,
            ));
            let (q0, q1) = (at(z0 + 0.005, dl), at(z1 - 0.005, dl));
            for q in [q0, q1] {
                p.mesh.add(&cyl(
                    DVec3::new(at(q.z - zl, d0 + 0.001).x, yy, q.z),
                    q,
                    0.0035,
                ));
            }
            p.mesh
                .add(&cyl(q0 - DVec3::Z * 0.003, q1 + DVec3::Z * 0.003, 0.0035));
            let latch = p.node("l", at(0.5 * (b0 + b1), 0.020), 0.7);
            let strike = p.node("s", at(0.5 * (z0 + z1), dl), 0.2);
            p.internal.push(Internal {
                pairs: vec![(0, 1)],
                v: vals(0.0, 50.0, 7e3, 11e3 * 1.5),
                deform_group: None,
            });
            let to: Vec<String> = if shell {
                vec![
                    format!("e5_1d{sl}"),
                    format!("e5_1u{sl}"),
                    format!("c2_5{sl}"),
                    format!("d3_b{sl}"),
                ]
            } else {
                vec![
                    format!("d2_b{sl}"),
                    format!("c2_3{sl}"),
                    format!("c1_3o{sl}"),
                    format!("d2_t{sl}"),
                ]
            };
            let grp = format!("latch_door_{}{sl}", door.to_lowercase());
            let v = vals(0.0, 100.0, 7e3, 11e3);
            p.hold_from(&format!("{grp}_body"), vec![strike], &shn(&to), 3, v);
            p.hold_panel(
                &format!("{grp}_door"),
                vec![latch],
                &format!("door_{door}{s}"),
                3,
                v,
            );
            out.push(p);
        }
        // ---- inside each door: the intrusion beam and the window regulator
        for (door, z0, z1) in [
            ("F", z_df - 0.07, z_b + 0.07),
            ("R", z_b - 0.08, z_dr + 0.07),
        ] {
            let mut p = MechPart::new(&format!("door_beam_{door}{s}"), "hardware");
            let n = 9;
            let path: Vec<DVec3> = (0..=n)
                .map(|i| {
                    inside(
                        g,
                        f.z(z0 + (z1 - z0) * i as f64 / n as f64),
                        f.y(0.500),
                        sg,
                        0.026,
                    )
                })
                .collect();
            // (in the pressed beam band of the door's inner panel, 50 mm deep)
            p.mesh.add(&cyl_along(&path, 0.012, 8, steel));
            for e in [path[0], path[n]] {
                p.mesh
                    .add(&box_(e, DVec3::new(0.004, 0.016, 0.022), steel, None));
            }
            p.host = Some(format!("door_{door}{s}"));
            p.carried_kg = 2.2;
            out.push(p);
            let mut p = MechPart::new(&format!("window_regulator_{door}{s}"), "hardware");
            let zc = 0.5 * (z0 + z1);
            // (in the pocket of the door's inner panel, 36 mm deep)
            let c = inside(g, f.z(zc), f.y(0.70), sg, 0.022);
            p.mesh
                .add(&box_(c, DVec3::new(0.0015, 0.055, 0.150), steel, None));
            for dz in [-0.11, 0.11] {
                p.mesh.add(&box_(
                    c + DVec3::new(0.0, 0.01, dz),
                    DVec3::new(0.003, 0.070, 0.010),
                    steel,
                    None,
                ));
            }
            p.mesh.add(&drum(
                c + DVec3::new(-sg * 0.002, -0.02, 0.0),
                c + DVec3::new(-sg * 0.008, -0.02, 0.0),
                0.030,
                12,
                Mat::TrimBlack,
            ));
            p.host = Some(format!("door_{door}{s}"));
            p.carried_kg = 1.8;
            out.push(p);
        }
        // ---- hood hinge on the cowl's end, its arm forward under the hood
        {
            let mut p = MechPart::new(&format!("hinge_hood_{s}"), "hardware");
            let b2 = sh(st, &format!("sh_b2_o{sl}"));
            let x = sg * f.x(0.66);
            let zr = f.z(z_df);
            let yu = |z: f64| top_y(g, z, x) - 0.047;
            let base = DVec3::new(x, b2.y + 0.006, zr + 0.02);
            p.mesh
                .add(&box_(base, DVec3::new(0.014, 0.004, 0.030), steel, None));
            let pivot = DVec3::new(x, (yu(zr + 0.03) - 0.006).min(b2.y + 0.030), zr + 0.03);
            p.mesh.add(&span(
                (x - 0.004, x + 0.004),
                (b2.y + 0.008, pivot.y + 0.006),
                (zr + 0.018, zr + 0.042),
                steel,
            ));
            let arm: Vec<DVec3> = [0.03, 0.09, 0.16, 0.24]
                .iter()
                .map(|dz| {
                    DVec3::new(
                        x,
                        if *dz < 0.05 {
                            pivot.y
                        } else {
                            yu(zr + dz) - 0.004
                        },
                        zr + dz,
                    )
                })
                .collect();
            p.mesh
                .add(&crate::structure::box_along(&arm, DVec3::Y, 0.022, 0.006));
            let a = p.node("p", pivot, 0.6);
            let b = p.node("a", arm[3], 0.5);
            p.internal.push(Internal {
                pairs: vec![(0, 1)],
                v: vals(0.0, 50.0, 8e3, INF),
                deform_group: None,
            });
            let v = vals(0.0, 100.0, 6e3, 10e3);
            p.hold_from(
                &format!("hinge_hood_{sl}_body"),
                vec![a.clone(), b.clone()],
                &shn(&[
                    format!("b2_o{sl}"),
                    format!("a8_m{sl}"),
                    format!("b2_i{sl}"),
                ]),
                4,
                v,
            );
            p.hold_panel(&format!("hinge_hood_{sl}_hood"), vec![a, b], "hood", 4, v);
            out.push(p);
        }
        // ---- hood gas strut along the hood's side
        {
            let mut p = MechPart::new(&format!("gas_strut_{s}"), "hardware");
            let x = sg * f.x(0.635);
            let (za, zb) = (f.z(1.03), f.z(1.50));
            let (a, b) = (
                DVec3::new(x, top_y(g, za, x) - 0.100, za),
                DVec3::new(x - sg * 0.015, top_y(g, zb, x) - 0.058, zb),
            );
            let d = (b - a).normalize();
            let mid = a + (b - a) * 0.58;
            p.mesh.add(&cyl(a, mid, 0.010));
            p.mesh.add(&cyl(mid, b, 0.0045));
            for (e, dir) in [(a, -1.0), (b, 1.0)] {
                p.mesh.add(&box_(
                    e + d * dir * 0.008,
                    DVec3::new(0.008, 0.008, 0.008),
                    steel,
                    None,
                ));
            }
            let na = p.node("b", a, 0.2);
            let nb = p.node("r", b, 0.15);
            p.internal.push(Internal {
                pairs: vec![(0, 1)],
                v: vals(0.0, 30.0, 900.0, INF),
                deform_group: None,
            });
            let v = vals(0.0, 30.0, 800.0, 2000.0);
            p.hold_from(
                &format!("gas_strut_{sl}_body"),
                vec![na],
                &shn(&[format!("a8_m{sl}"), format!("a6_{sl}"), format!("b2_o{sl}")]),
                3,
                v,
            );
            p.hold_panel(&format!("gas_strut_{sl}_hood"), vec![nb], "hood", 3, v);
            out.push(p);
        }
        // ---- boot lid hinge: a bracket under the shelf's rear edge and a
        // gooseneck back under the lid
        {
            let mut p = MechPart::new(&format!("hinge_trunk_{s}"), "hardware");
            let e6 = sh(st, &format!("sh_e6_o{sl}"));
            let x = sg * f.x(0.55);
            let z0 = sh(st, &format!("sh_e6_r{sl}")).z - 0.030;
            let ys = e6.y;
            p.mesh.add(&span(
                (x - 0.015, x + 0.015),
                (ys - 0.050, ys - 0.012),
                (z0 - 0.020, z0 + 0.012),
                steel,
            ));
            let yl = |z: f64| top_y(g, z, x) - 0.050;
            let neck: Vec<DVec3> = [
                (0.0, ys - 0.031),
                (-0.07, ys - 0.085),
                (-0.16, ys - 0.085),
                (-0.22, yl(z0 - 0.22) - 0.010),
                (-0.30, yl(z0 - 0.30) - 0.010),
            ]
            .iter()
            .map(|&(dz, y)| DVec3::new(x, y, z0 - 0.02 + dz))
            .collect();
            p.mesh
                .add(&crate::structure::box_along(&neck, DVec3::Y, 0.020, 0.010));
            let a = p.node("p", neck[0], 0.5);
            let b = p.node("a", neck[4], 0.4);
            p.internal.push(Internal {
                pairs: vec![(0, 1)],
                v: vals(0.0, 50.0, 6e3, INF),
                deform_group: None,
            });
            let v = vals(0.0, 100.0, 5e3, 9e3);
            p.hold_from(
                &format!("hinge_trunk_{sl}_body"),
                vec![a.clone(), b.clone()],
                &shn(&[format!("e6_o{sl}"), format!("e6_r{sl}"), format!("e7_{sl}")]),
                4,
                v,
            );
            p.hold_panel(
                &format!("hinge_trunk_{sl}_lid"),
                vec![a, b],
                "trunk_lid",
                4,
                v,
            );
            out.push(p);
        }
    }
    // ---- hood latch on the radiator support's tie bar
    {
        let mut p = MechPart::new("latch_hood", "hardware");
        let a3 = sh(st, "sh_a3_uc");
        let top = (top_y(g, a3.z, 0.0) - 0.050).min(a3.y + 0.040);
        p.mesh.add(&span(
            (-0.035, 0.035),
            (a3.y + 0.002, top - 0.006),
            (a3.z - 0.018, a3.z + 0.018),
            steel,
        ));
        p.mesh.add(&drum(
            DVec3::new(0.0, top - 0.008, a3.z),
            DVec3::new(0.0, top, a3.z),
            0.006,
            8,
            steel,
        ));
        let a = p.node("l", DVec3::new(0.0, a3.y + 0.015, a3.z), 0.6);
        let b = p.node("s", DVec3::new(0.0, top, a3.z), 0.2);
        p.internal.push(Internal {
            pairs: vec![(0, 1)],
            v: vals(0.0, 50.0, 5e3, INF),
            deform_group: None,
        });
        let v = vals(0.0, 100.0, 5e3, 9e3);
        p.hold_from(
            "latch_hood_body",
            vec![a],
            &shn(&["a3_uc".into(), "a3_ul".into(), "a3_ur".into()]),
            3,
            v,
        );
        p.hold_panel("latch_hood_hood", vec![b], "hood", 3, v);
        out.push(p);
    }
    // ---- boot latch: on a stay up from the tail panel, under the lid's edge
    {
        let mut p = MechPart::new("latch_trunk", "hardware");
        let e4 = sh(st, "sh_e4_uc");
        let zt = f.z(-2.38);
        let top = top_y(g, zt, 0.0) - 0.055;
        p.mesh.add(&span(
            (-0.030, 0.030),
            (top - 0.045, top),
            (zt - 0.012, zt + 0.012),
            steel,
        ));
        p.mesh.add(&crate::structure::box_along(
            &[
                DVec3::new(0.0, top - 0.040, zt),
                DVec3::new(0.0, e4.y + 0.06, 0.5 * (zt + e4.z) + 0.01),
                DVec3::new(0.0, e4.y + 0.024, e4.z + 0.012),
            ],
            DVec3::Z,
            0.030,
            0.005,
        ));
        let a = p.node("l", DVec3::new(0.0, top - 0.02, zt), 0.5);
        let b = p.node("s", DVec3::new(0.0, top, zt), 0.2);
        p.internal.push(Internal {
            pairs: vec![(0, 1)],
            v: vals(0.0, 50.0, 5e3, INF),
            deform_group: None,
        });
        let v = vals(0.0, 100.0, 5e3, 9e3);
        p.hold_from(
            "latch_trunk_body",
            vec![a],
            &shn(&["e4_uc".into(), "e4_ul".into(), "e4_ur".into()]),
            3,
            v,
        );
        p.hold_panel("latch_trunk_lid", vec![b], "trunk_lid", 3, v);
        out.push(p);
    }
    // ---- bumper absorbers: foam between each cover and its beam
    for (front, name, yb, h, dz, w, kg) in [
        (true, "absorber_front", 0.465, 0.050, -0.0635, 0.026, 0.7),
        (false, "absorber_rear", 0.430, 0.060, 0.060, 0.034, 0.7),
    ] {
        let mut p = MechPart::new(name, "bumper");
        // (the front one behind the lower intake's back plate, which leans
        // with the nose; the rear one above the cover's tuck)
        let face = if front { &g.z_nose } else { &g.z_tail };
        let yf = if front { 0.45 } else { 0.414 };
        let at = |x: f64| DVec3::new(f.x(x), f.y(yb), face(f.x(x.abs()), f.y(yf)) + dz);
        let path: Vec<DVec3> = (0..=16)
            .map(|i| at(-0.60 + 1.20 * i as f64 / 16.0))
            .collect();
        let mut m = crate::structure::box_along(&path, DVec3::Y, w, h);
        m.set_mat(Mat::TrimBlack);
        p.mesh = m;
        for (i, (x, dy)) in [(-0.55, 1.0), (-0.18, -1.0), (0.18, 1.0), (0.55, -1.0)]
            .into_iter()
            .enumerate()
        {
            p.node(
                &format!("{}", i + 1),
                at(x) + DVec3::Y * dy * 0.4 * h,
                kg / 4.0,
            );
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: vals(0.0, 30.0, 300.0, INF),
            deform_group: None,
        });
        let beam = if front { "impbf" } else { "impbr" };
        let to: Vec<String> = (1..=5)
            .flat_map(|i| [format!("{beam}_{i}u"), format!("{beam}_{i}d")])
            .collect();
        p.hold_from(name, p.names(), &to, 8, vals(0.0, 30.0, 300.0, 800.0));
        out.push(p);
    }
    // ---- splash shield under the engine bay
    {
        let mut p = MechPart::new("undertray", "body");
        let (z0, z1) = (g.axle_f + 0.275, f.zi((g.z_nose)(0.0, f.y(0.30))) - 0.10);
        let hx = 0.40;
        let y = |z: f64| 0.222 + 0.070 * ((z - z0) / (z1 - z0)).powi(2);
        let (nu, nv) = (15, 11);
        let (mut m, _) = crate::structure::panel(
            nu,
            nv,
            |i, j| {
                let z = z0 + (z1 - z0) * j as f64 / (nv - 1) as f64;
                f.p(-hx + 2.0 * hx * i as f64 / (nu - 1) as f64, y(z), z)
            },
            |_, _| true,
        );
        m.set_mat(Mat::TrimBlack);
        p.mesh = m;
        for (tg, x, z) in [
            ("fl", hx, z1),
            ("fr", -hx, z1),
            ("rl", hx, z0),
            ("rr", -hx, z0),
        ] {
            p.node(tg, f.p(x, y(z), z), 0.65);
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: vals(0.0, 30.0, 500.0, INF),
            deform_group: None,
        });
        p.hold(
            "undertray",
            &["a1_dl", "a1_dr", "a2_dl", "a2_dr", "a3_dc", "a4_l", "a4_r"],
            10,
            BeamKind::Normal,
            vals(0.0, 30.0, 500.0, 1500.0),
        );
        out.push(p);
    }
    // ---- fuel filler: the neck from the tank up to the right quarter, and
    // its flap on the skin
    {
        let zf = g.axle_r - 0.535;
        let (cup, n) = crate::parts::surface_point(&g.tube, f.z(zf), f.y(0.80), -1.0, (1.5, 5.5));
        let mut p = MechPart::new("fuel_filler", "fuel");
        let mut path: Vec<DVec3> = [
            (-0.424, 0.368),
            (-0.447, 0.337),
            (-0.560, 0.337),
            (-0.630, 0.430),
            (-0.700, 0.600),
        ]
        .iter()
        .map(|&(x, y)| f.p(x, y, zf))
        .collect();
        path.push(cup - n * 0.120 - DVec3::Y * 0.02);
        path.push(cup - n * 0.062);
        p.mesh.add(&cyl_along(&path, 0.021, 10, steel));
        p.mesh.add(&lathe_at(
            &[
                (0.0, 0.0),
                (0.030, 0.0),
                (0.045, 0.014),
                (0.045, 0.020),
                (0.0, 0.020),
            ],
            14,
            Mat::TrimBlack,
            n,
            cup - n * 0.066,
        ));
        let nn: Vec<String> = [
            ("b", path[1], 0.5),
            ("m", path[4], 0.4),
            ("t", cup - n * 0.06, 0.4),
        ]
        .into_iter()
        .map(|(t, q, kg)| p.node(t, q, kg))
        .collect();
        p.internal.push(Internal {
            pairs: p.complete(),
            v: vals(0.0, 30.0, 1500.0, INF),
            deform_group: None,
        });
        let tank: Vec<String> = ["fuet_1", "fuet_3", "fuet_5", "fuet_7"]
            .iter()
            .map(|n| n.to_string())
            .collect();
        p.hold_from(
            "fuel_filler_tank",
            vec![nn[0].clone(), nn[1].clone()],
            &tank,
            4,
            vals(0.0, 30.0, 1500.0, 4000.0),
        );
        p.hold_from(
            "fuel_filler_body",
            vec![nn[2].clone(), nn[1].clone()],
            &shn(&[
                "e5_4ur".into(),
                "e5_5ur".into(),
                "e5_4dr".into(),
                "e5_3ur".into(),
            ]),
            4,
            vals(0.0, 30.0, 1500.0, 4000.0),
        );
        out.push(p);
        let mut p = MechPart::new("fuel_flap", "body");
        p.mesh = lathe_at(
            &[
                (0.0, 0.0025),
                (0.068, 0.0025),
                (0.070, 0.0045),
                (0.068, 0.0065),
                (0.0, 0.0065),
            ],
            20,
            Mat::Paint,
            n,
            cup,
        );
        p.host = Some("body_shell".into());
        p.carried_kg = 0.25;
        out.push(p);
    }
    // ---- number plates
    for (name, host, face, sgn, y) in [
        ("plate_front", "bumper_front", &g.z_nose, 1.0, 0.585),
        ("plate_rear", "body_shell", &g.z_tail, -1.0, 0.68),
    ] {
        let mut p = MechPart::new(name, "body");
        let (hx, hy) = (0.155, 0.0775);
        let zs = [
            (0.0, y - hy),
            (0.0, y),
            (0.0, y + hy),
            (hx, y - hy),
            (hx, y),
            (hx, y + hy),
        ]
        .map(|(x, yy)| face(x, f.y(yy)));
        let zo = if sgn > 0.0 {
            zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            zs.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        p.mesh.add(&box_(
            DVec3::new(0.0, f.y(y), zo + sgn * 0.0045),
            DVec3::new(hx, hy, 0.0015),
            Mat::Chrome,
            None,
        ));
        p.mesh.add(&box_(
            DVec3::new(0.0, f.y(y), zo + sgn * 0.0020),
            DVec3::new(hx + 0.006, hy + 0.006, 0.0010),
            Mat::TrimBlack,
            None,
        ));
        p.host = Some(host.into());
        p.carried_kg = 0.2;
        out.push(p);
    }
}

// ---- 2. brakes, springs, anti-roll bars ----------------------------------------------------

/// A coil spring round the axis `a`→`b`: coil radius `rc`, wire radius
/// `rw`, with a seat disc at each end.
pub(crate) fn coil_spring(a: DVec3, b: DVec3, rc: f64, rw: f64, turns: f64) -> Mesh {
    use crate::structure::{sweep_rings, Ring};
    let d = (b - a).normalize();
    let u = d.cross(DVec3::Z).normalize();
    let v = d.cross(u);
    let n = (turns * 12.0).round() as usize;
    let sides = 6;
    let pts: Vec<glam::DVec2> = (0..sides)
        .map(|k| {
            glam::DVec2::from_angle(std::f64::consts::TAU * (k as f64 + 0.5) / sides as f64) * rw
        })
        .collect();
    let rings: Vec<Ring> = (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let phi = std::f64::consts::TAU * turns * t;
            let e = u * phi.cos() + v * phi.sin();
            Ring {
                c: a + (b - a) * t + e * rc,
                a: e,
                b: d,
                pts: pts.clone(),
            }
        })
        .collect();
    let caps: Vec<[usize; 4]> = (0..sides / 2 - 1)
        .map(|k| [sides - 1 - k, k, k + 1, sides - 2 - k])
        .collect();
    let mut m = sweep_rings(&rings, &caps, Mat::MetalBare);
    for (e, s) in [(a, -1.0), (b, 1.0)] {
        m.add(&drum(
            e + d * s * (rw + 0.001),
            e + d * s * (rw + 0.007),
            rc + rw + 0.002,
            14,
            Mat::MetalBare,
        ));
    }
    m
}

fn running_gear(g: &Gear, _st: &Structure, out: &mut Vec<MechPart>) {
    let f = &g.f;
    let (t, wy) = (g.half_track, g.wheel_y);
    let disc_r = 0.16;
    // ---- calipers: a floating caliper astride each disc, on the knuckle's
    // bracket (behind the axle in front, ahead of it at the rear)
    for (cn, sg, front) in [
        ("FL", 1.0, true),
        ("FR", -1.0, true),
        ("RL", 1.0, false),
        ("RR", -1.0, false),
    ] {
        let c = cn.to_lowercase();
        let mut p = MechPart::new(&format!("brake_caliper_{cn}"), "suspension");
        let (hz, body) = if front {
            (0.058, 0.020)
        } else {
            (0.046, 0.016)
        };
        // (built on top of the disc in the wheel's frame, x outboard: the
        // disc is x -52..-28 mm; then turned round the axle)
        let mut m = Mesh::new();
        let xi = -0.052 - 0.004 - body;
        m.add(&rbox(
            DVec3::new(xi, disc_r - 0.026, 0.0),
            DVec3::new(body, 0.030, hz),
            0.006,
            2,
            Mat::Paint,
        ));
        m.add(&drum(
            DVec3::new(xi - body + 0.002, disc_r - 0.030, 0.0),
            DVec3::new(xi - body - 0.016, disc_r - 0.030, 0.0),
            0.026,
            12,
            Mat::MetalBare,
        ));
        m.add(&box_(
            DVec3::new(0.5 * (xi - body - 0.004), disc_r + 0.012, 0.0),
            DVec3::new(0.5 * (-0.004 - (xi - body)), 0.008, hz - 0.006),
            Mat::Paint,
            None,
        ));
        m.add(&rbox(
            DVec3::new(-0.014, disc_r - 0.026, 0.0),
            DVec3::new(0.010, 0.030, hz - 0.006),
            0.005,
            2,
            Mat::Paint,
        ));
        let a = (if front { -1.0f64 } else { 1.0 }) * 55.0f64.to_radians();
        m.transform(Some(crate::mesh::rot_x(a)), DVec3::ZERO);
        if sg < 0.0 {
            m.mirror_x();
        }
        m.transform(
            None,
            DVec3::new(sg * t, wy, if front { g.zf } else { g.zr }),
        );
        p.mesh = m;
        p.binds = ["lo", "up", "toe", "n4"]
            .iter()
            .map(|n| format!("hub{c}_{n}"))
            .collect();
        p.carried_kg = if front { 4.5 } else { 3.2 };
        out.push(p);
    }
    // ---- anti-roll bars: a bar in two bushes on the subframe, its ends on
    // drop links to the lower joints
    let rod = Mat::MetalBare;
    for (front, name, sub, z_bar, y_bar, x_bush, ends, kg) in [
        (
            true,
            "antiroll_bar_front",
            "subf",
            g.axle_f + 0.280,
            0.300,
            0.30,
            [
                (0.46, 0.300, 0.280),
                (0.52, 0.262, 0.185),
                (0.55, 0.250, 0.085),
            ],
            5.5,
        ),
        (
            false,
            "antiroll_bar_rear",
            "subr",
            g.axle_r + 0.265,
            0.215,
            0.33,
            [
                (0.46, 0.215, 0.265),
                (0.50, 0.222, 0.200),
                (0.53, 0.230, 0.145),
            ],
            3.5,
        ),
    ] {
        let mut p = MechPart::new(name, "suspension");
        let ax = if front { g.axle_f } else { g.axle_r };
        let mut path: Vec<DVec3> = ends
            .iter()
            .rev()
            .map(|&(x, y, dz)| f.p(-x, y, ax + dz))
            .collect();
        path.extend(ends.iter().map(|&(x, y, dz)| f.p(x, y, ax + dz)));
        let r_bar = if front { 0.011 } else { 0.009 };
        p.mesh.add(&cyl_along(&path, r_bar, 8, rod));
        let wheel_z = if front { g.zf } else { g.zr };
        let mut names = Vec::new();
        for (s, sg) in [("l", 1.0), ("r", -1.0)] {
            let bush = f.p(sg * x_bush, y_bar, z_bar);
            p.mesh.add(&box_(
                bush,
                DVec3::new(0.016, r_bar + 0.008, r_bar + 0.008),
                Mat::Rubber,
                None,
            ));
            p.mesh.add(&box_(
                bush + DVec3::Y * (r_bar + 0.010),
                DVec3::new(0.020, 0.002, r_bar + 0.020),
                rod,
                None,
            ));
            // the drop link: from the bar's end down towards the lower joint
            let end = path[if sg > 0.0 { path.len() - 1 } else { 0 }];
            let lo = DVec3::new(sg * (t - 0.16), wy - 0.14, wheel_z);
            let foot = end + (lo - end).normalize() * 0.055;
            p.mesh.add(&cyl(end, foot, 0.006));
            p.mesh.add(&drum(
                end - DVec3::X * 0.010,
                end + DVec3::X * 0.010,
                0.012,
                8,
                rod,
            ));
            let nb = p.node(&format!("b{s}"), bush, 0.28 * kg);
            let ne = p.node(&format!("e{s}"), end, 0.22 * kg);
            names.push((s, nb, ne));
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: vals(0.0, 60.0, 9e3, INF),
            deform_group: None,
        });
        for (s, nb, ne) in names {
            let c = format!("{}{s}", if front { "f" } else { "r" });
            let to: Vec<String> = if front {
                vec![
                    format!("{sub}_f{s}"),
                    format!("{sub}_k{s}"),
                    format!("sh_a4_{s}"),
                ]
            } else {
                vec![
                    format!("{sub}_f{s}"),
                    "subr_cf".to_string(),
                    format!("sh_e1_1d{s}"),
                ]
            };
            let tag = if front { "front" } else { "rear" };
            p.mounts.push(Mount {
                group: format!("antiroll_bush_{tag}_{s}"),
                from: vec![nb],
                to,
                beams: 3,
                kind: BeamKind::Bounded,
                v: vals(0.0, 80.0, 6e3, 12e3),
                precomp: 1.0,
                to_part: None,
            });
            p.mounts.push(Mount {
                group: format!("antiroll_link_{c}"),
                from: vec![ne],
                to: vec![format!("hub{c}_lo")],
                beams: 1,
                kind: BeamKind::Normal,
                v: vals(0.0, 60.0, 4e3, 8e3),
                precomp: 1.0,
                to_part: None,
            });
        }
        out.push(p);
    }
}

// ---- 3. the exhaust's sections -------------------------------------------------------------

/// The exhaust's solid along its 13 nodes: the downpipe from the manifold's
/// flange, the catalytic converter, the centre pipe, the resonator under
/// the floor, the pipe over the rear axle, the rear silencer and the
/// tailpipe with its tip; a flange at every joint and a rubber hanger at
/// each of the four hanger mounts (`hang`: node indices).
pub(crate) fn exhaust_mesh(path: &[DVec3], hang: &[usize]) -> Mesh {
    let steel = Mat::MetalBare;
    let mut m = cyl_along(path, 0.022, 10, steel);
    let dir =
        |i: usize| (path[(i + 1).min(path.len() - 1)] - path[i.saturating_sub(1)]).normalize();
    let flange = |m: &mut Mesh, at: DVec3, d: DVec3| {
        if at.y < 0.22 {
            // (under the floor: a two-bolt flange lying flat)
            let x = DVec3::Y.cross(d).normalize();
            let mut k = box_stations(&[
                (at - d * 0.005, x, d.cross(x), 0.080, 0.046),
                (at + d * 0.005, x, d.cross(x), 0.080, 0.046),
            ]);
            k.set_mat(steel);
            m.add(&k);
        } else {
            m.add(&drum(at - d * 0.005, at + d * 0.005, 0.034, 12, steel));
        }
    };
    // the downpipe's flange on the manifold
    let d0 = (path[1] - path[0]).normalize();
    flange(&mut m, path[0] + d0 * 0.006, d0);
    // a flat body between two points, tapering to the pipe at each end
    // (the floor is close above: bodies are oval, wider than high)
    let body = |m: &mut Mesh, a: DVec3, b: DVec3, w: f64, h: f64, cone: f64| {
        let d = (b - a).normalize();
        let x = DVec3::Y.cross(d).normalize();
        let y = d.cross(x);
        let st: Vec<(DVec3, DVec3, DVec3, f64, f64)> = [
            (0.0, 0.050, 0.042),
            (cone, w, h),
            ((b - a).length() - cone, w, h),
            ((b - a).length(), 0.050, 0.042),
        ]
        .iter()
        .map(|&(s, ww, hh)| (a + d * s, x, y, ww, hh))
        .collect();
        let mut k = box_stations(&st);
        k.set_mat(steel);
        m.add(&k);
        flange(m, a - d * 0.012, d);
        flange(m, b + d * 0.012, d);
    };
    // catalytic converter between nodes 3 and 4
    let (a, b) = (
        path[2] + (path[3] - path[2]) * 0.10,
        path[2] + (path[3] - path[2]) * 0.92,
    );
    body(&mut m, a, b, 0.130, 0.043, 0.035);
    // resonator between nodes 5 and 7
    let (a, b) = (
        path[4] + (path[5] - path[4]) * 0.55,
        path[5] + (path[6] - path[5]) * 0.70,
    );
    body(&mut m, a, b, 0.150, 0.043, 0.050);
    // rear silencer between nodes 11 and 12
    let (a, b) = (
        path[10] + (path[11] - path[10]) * 0.12,
        path[10] + (path[11] - path[10]) * 0.92,
    );
    body(&mut m, a, b, 0.190, 0.095, 0.045);
    // the joint ahead of the axle, and the tailpipe's tip
    flange(&mut m, path[7], dir(7));
    let n = path.len() - 1;
    let dt = (path[n] - path[n - 1]).normalize();
    m.add(&drum(
        path[n] - dt * 0.070,
        path[n] + dt * 0.012,
        0.027,
        14,
        Mat::Chrome,
    ));
    // hangers: a rod off the pipe into a rubber block (sideways under the
    // floor, where there is no room above)
    for &i in hang {
        let p = path[i];
        let side = if p.y < 0.22 { DVec3::X } else { DVec3::Y };
        m.add(&cyl(p + side * 0.018, p + side * 0.062, 0.005));
        m.add(&box_(
            p + side * 0.066,
            DVec3::new(0.012, 0.012, 0.018) + side * 0.004,
            Mat::Rubber,
            None,
        ));
    }
    m
}

// ---- 4. steering column and pedals ---------------------------------------------------------

/// The firewall's openings (reference x and y of each one's centre): the
/// steering shaft's and the brake push rod's. [`crate::structure::unibody`]
/// leaves the panel's cells there out.
pub const FIREWALL_HOLES: [(f64, f64); 2] = [(0.38, 0.493), (0.38, 0.665)];

fn controls(g: &Gear, st: &Structure, out: &mut Vec<MechPart>) {
    let f = &g.f;
    let steel = Mat::MetalBare;
    let z_fw = sh(st, "sh_b1_ol").z;
    let x = f.x(FIREWALL_HOLES[0].0);
    // ---- steering column: the tube from under the instrument panel to the
    // firewall on the wheel's axis, a joint each side of the firewall, and
    // the intermediate shaft down to the rack's pinion
    {
        let mut p = MechPart::new("steering_column", "controls");
        // (the wheel's centre and axis: `Sedan::make_parts`)
        let wheel = DVec3::new(x, f.y(0.84), f.z(0.17));
        let tilt = 24.0f64.to_radians();
        let d = DVec3::new(0.0, -tilt.sin(), tilt.cos());
        let on = |z: f64| wheel + d * ((z - wheel.z) / d.z);
        // from under the panel (its underside is 0.144 m below the wheel's centre)
        let top = wheel + d * (0.186 / tilt.sin());
        let (wall_in, wall_out) = (on(z_fw - 0.050), on(z_fw + 0.045));
        p.mesh.add(&drum(top, wall_in, 0.022, 12, Mat::TrimBlack));
        p.mesh.add(&cyl(wall_in, wall_out, 0.010));
        p.mesh.add(&drum(
            wall_in - d * 0.004,
            wall_in + d * 0.010,
            0.026,
            12,
            steel,
        ));
        // the pinion on the rack, left of the engine; the shaft drops to a
        // second joint and runs in under the engine's mount ear
        let rack = f.p(0.262, 0.338, g.axle_f + 0.135);
        let pin = rack + DVec3::new(0.0, 0.024, -0.014);
        p.mesh.add(&drum(rack, pin, 0.017, 10, steel));
        let mid = f.p(0.33, 0.360, g.axle_f - 0.115);
        let run = [wall_out, mid, pin];
        for w in run.windows(2) {
            let dj = (w[1] - w[0]).normalize();
            p.mesh
                .add(&cyl(w[0] + dj * 0.012, w[1] - dj * 0.012, 0.010));
        }
        for j in [wall_out, mid] {
            p.mesh.add(&box_(j, DVec3::splat(0.015), steel, None));
        }
        let nu = p.node("u", top + d * 0.03, 2.2);
        let nf = p.node("f", on(z_fw), 1.6);
        let np = p.node("p", pin, 1.2);
        p.internal.push(Internal {
            pairs: p.complete(),
            v: vals(0.0, 60.0, 4e3, INF),
            deform_group: None,
        });
        let dash: Vec<String> = ["das_ml", "das_tl", "das_c", "das_bl"]
            .iter()
            .map(|n| n.to_string())
            .collect();
        p.hold_from(
            "steering_column_dash",
            vec![nu],
            &dash,
            3,
            vals(0.0, 60.0, 4e3, 8e3),
        );
        p.hold_from(
            "steering_column_firewall",
            vec![nf],
            &shn(&["b1_ol".into(), "b1_il".into(), "c1_1tl".into()]),
            3,
            vals(0.0, 60.0, 3e3, 6e3),
        );
        let gear: Vec<String> = ["ster_l", "subf_kl", "subf_fl"]
            .iter()
            .map(|n| n.to_string())
            .collect();
        p.hold_from(
            "steering_column_rack",
            vec![np],
            &gear,
            3,
            vals(0.0, 60.0, 5e3, 10e3),
        );
        out.push(p);
    }
    // ---- pedal box on the firewall: bracket, pivot shaft, the brake pedal
    // with its push rod through to the booster, the accelerator
    {
        let mut p = MechPart::new("pedal_box", "controls");
        let (yb, zb) = (f.y(FIREWALL_HOLES[1].1), z_fw - 0.045);
        let (xb, xa) = (f.x(0.33), f.x(0.215));
        p.mesh.add(&span(
            (xa - 0.025, x + 0.020),
            (yb - 0.022, yb + 0.030),
            (zb + 0.012, z_fw - 0.008),
            steel,
        ));
        p.mesh.add(&drum(
            DVec3::new(xa - 0.020, yb, zb),
            DVec3::new(x + 0.015, yb, zb),
            0.008,
            8,
            steel,
        ));
        // brake: the arm down to its pad, a lever to the push rod
        let pad = DVec3::new(xb, f.y(0.40), zb - 0.105);
        p.mesh.add(&strap(
            DVec3::new(xb, yb, zb),
            pad,
            DVec3::X,
            0.016,
            0.008,
            steel,
        ));
        p.mesh.add(&box_(
            pad,
            DVec3::new(0.032, 0.026, 0.006),
            Mat::Rubber,
            Some(crate::mesh::rot_x(-0.38)),
        ));
        p.mesh.add(&cyl(
            DVec3::new(x, yb, zb + 0.004),
            DVec3::new(x, yb, z_fw + 0.006),
            0.006,
        ));
        // accelerator: a slim arm and a long pad
        let pad_a = DVec3::new(xa, f.y(0.37), zb - 0.060);
        p.mesh.add(&strap(
            DVec3::new(xa, yb - 0.004, zb),
            pad_a,
            DVec3::X,
            0.010,
            0.006,
            steel,
        ));
        p.mesh.add(&box_(
            pad_a,
            DVec3::new(0.020, 0.050, 0.005),
            Mat::Rubber,
            Some(crate::mesh::rot_x(-0.30)),
        ));
        p.tetra(
            DVec3::new(0.5 * (xa + x), 0.5 * (yb + pad.y), zb - 0.04),
            DVec3::new(0.5 * (x - xa), 0.5 * (yb - pad.y), 0.045),
            3.2,
            vals(0.0, 40.0, 2e3, INF),
        );
        p.hold(
            "pedal_box",
            &["b1_ol", "b1_il", "c1_1tl", "b2_ol", "c2_1l"],
            6,
            BeamKind::Normal,
            vals(0.0, 60.0, 2500.0, 5000.0),
        );
        out.push(p);
    }
}

// ---- 6. the interior's trim ----------------------------------------------------------------

/// A formed sheet as a closed solid: `nu` × `nv` points `p(i, j)` on one
/// face, the other face `off(i, j)` away, the cells where `keep(i, j)`.
fn sheet_solid(
    nu: usize,
    nv: usize,
    p: impl Fn(usize, usize) -> DVec3,
    off: impl Fn(usize, usize) -> DVec3,
    keep: impl Fn(usize, usize) -> bool,
    mat: Mat,
) -> Mesh {
    let mut ids: Vec<Option<(u32, u32)>> = vec![None; nu * nv];
    let mut v: Vec<DVec3> = Vec::new();
    let mut q: Vec<[u32; 4]> = Vec::new();
    let kept = |i: isize, j: isize| {
        i >= 0
            && j >= 0
            && (i as usize) < nu - 1
            && (j as usize) < nv - 1
            && keep(i as usize, j as usize)
    };
    let mut id = |i: usize, j: usize, v: &mut Vec<DVec3>| {
        *ids[i * nv + j].get_or_insert_with(|| {
            let a = p(i, j);
            v.push(a);
            v.push(a + off(i, j));
            (v.len() as u32 - 2, v.len() as u32 - 1)
        })
    };
    for i in 0..nu - 1 {
        for j in 0..nv - 1 {
            if !kept(i as isize, j as isize) {
                continue;
            }
            let c = [
                id(i, j, &mut v),
                id(i + 1, j, &mut v),
                id(i + 1, j + 1, &mut v),
                id(i, j + 1, &mut v),
            ];
            q.push([c[0].0, c[1].0, c[2].0, c[3].0]);
            q.push([c[3].1, c[2].1, c[1].1, c[0].1]);
            let nb = [(0, -1), (1, 0), (0, 1), (-1, 0)];
            for k in 0..4 {
                if !kept(i as isize + nb[k].0, j as isize + nb[k].1) {
                    let (a, b) = (c[k], c[(k + 1) % 4]);
                    q.push([b.0, a.0, a.1, b.1]);
                }
            }
        }
    }
    crate::mesh::orient_outward(Mesh::from_faces(v, q, vec![], mat, None, -1))
}

fn lin(a: f64, b: f64, i: usize, n: usize) -> f64 {
    a + (b - a) * i as f64 / (n - 1) as f64
}

/// The instrument panel's features on its rear face (`zr`, towards the
/// cabin) at height `yp`: the cluster's two dials under its hood, the centre
/// screen and vents, the glovebox door with its handle.
pub(crate) fn dash_features(f: &Frame, yp: f64, zr: f64) -> Mesh {
    let mut m = Mesh::new();
    let xd = f.x(0.38);
    m.add(&box_(
        DVec3::new(xd, yp + 0.060, zr - 0.003),
        DVec3::new(0.135, 0.040, 0.004),
        Mat::TrimBlack,
        None,
    ));
    for dx in [-0.062, 0.062] {
        m.add(&lathe_at(
            &[
                (0.0, 0.0),
                (0.046, 0.0),
                (0.050, 0.006),
                (0.044, 0.008),
                (0.0, 0.005),
            ],
            20,
            Mat::Chrome,
            -DVec3::Z,
            DVec3::new(xd + dx, yp + 0.060, zr - 0.007),
        ));
    }
    m.add(&box_(
        DVec3::new(0.0, yp + 0.040, zr - 0.004),
        DVec3::new(0.105, 0.060, 0.005),
        Mat::TrimBlack,
        None,
    ));
    for dx in [-0.165, 0.165] {
        m.add(&box_(
            DVec3::new(dx, yp + 0.060, zr - 0.003),
            DVec3::new(0.035, 0.022, 0.004),
            Mat::TrimBlack,
            None,
        ));
    }
    m.add(&box_(
        DVec3::new(-xd, yp - 0.020, zr - 0.003),
        DVec3::new(0.170, 0.055, 0.004),
        Mat::InteriorPlastic,
        None,
    ));
    m.add(&box_(
        DVec3::new(-xd, yp + 0.018, zr - 0.009),
        DVec3::new(0.040, 0.006, 0.003),
        Mat::Chrome,
        None,
    ));
    m
}

fn interior(g: &Gear, st: &Structure, out: &mut Vec<MechPart>) {
    let f = &g.f;
    let clip = |deform: f64, strength: f64| vals(0.0, 30.0, deform, strength);
    let tube = &g.tube;
    let rows: Vec<DVec3> = (1..=5).map(|k| sh(st, &format!("sh_c3_{k}"))).collect();
    let z_fw = sh(st, "sh_b1_ol").z;
    let yp = sh(st, "sh_c1_3tl").y + 0.004;
    // ---- floor carpet: one moulding over the pan, up the toe board, over
    // the tunnel and the floor's longitudinals; cut out round the seat cross
    // members and the console
    {
        let mut p = MechPart::new("carpet", "interior");
        let xs = (1..=5)
            .map(|k| sh(st, &format!("sh_c1_{k}ol")).x)
            .fold(f64::INFINITY, f64::min)
            - 0.0325
            - 0.016;
        let (z0, z1) = (z_fw - 0.016, rows[4].z + 0.014);
        let toe = (f.y(0.42) - f.y(0.222)) / (z_fw - rows[0].z);
        let xl = f.x(0.43);
        let height = |x: f64, z: f64| {
            let mut h = yp;
            if z > rows[0].z {
                h = yp + (z - rows[0].z) * toe;
            }
            if x.abs() < 0.090 || (z > rows[0].z - 0.02 && x.abs() < 0.125) {
                h = h.max(yp + 0.166);
            }
            if (x.abs() - xl).abs() < 0.052 && z > rows[2].z + 0.04 && z < rows[0].z {
                h = h.max(yp + 0.050);
            }
            h
        };
        let (nu, nv) = (61, 66);
        let (dx, dz) = (2.0 * xs / (nu - 1) as f64, (z1 - z0) / (nv - 1) as f64);
        let dza = dz.abs();
        let (xa, za) = (
            |i: usize| -xs + dx * i as f64,
            |j: usize| z0 + dz * j as f64,
        );
        // (the highest of a point's neighbourhood: the moulding bridges
        // every step instead of cutting its corner)
        let hd = |i: usize, j: usize| {
            let mut h: f64 = 0.0;
            for a in [-1.0, 0.0, 1.0] {
                for b in [-1.0, 0.0, 1.0] {
                    h = h.max(height(xa(i) + a * dx, za(j) + b * dz));
                }
            }
            h
        };
        let (c2, c4) = (rows[1].z, rows[3].z);
        let bench_front = sh(st, "sh_e3_1").z + 0.185 + 0.20;
        let keep = |i: usize, j: usize| {
            let (x, z) = (xa(i) + 0.5 * dx, za(j) + 0.5 * dz);
            let member = x.abs() > 0.110
                && ((z - rows[2].z).abs() < 0.062 + dza || (z - rows[3].z).abs() < 0.062 + dza);
            let console =
                x.abs() < 0.100 + dx && z < c2 - 0.10 + 0.02 + dza && z > c4 + 0.05 - 0.02 - dza;
            // (the rear bench's cushion sits 4 mm over the tunnel)
            let bench = x.abs() < 0.125 + 2.0 * dx && z < bench_front + 0.02 + dza;
            !member && !console && !bench
        };
        p.mesh = sheet_solid(
            nu,
            nv,
            |i, j| DVec3::new(xa(i), hd(i, j) + 0.006, za(j)),
            |_, _| DVec3::Y * 0.006,
            keep,
            Mat::InteriorFabric,
        );
        for (k, z) in [
            0.5 * (rows[0].z + rows[1].z),
            0.5 * (rows[2].z + rows[3].z),
            0.5 * (rows[3].z + rows[4].z),
        ]
        .into_iter()
        .enumerate()
        {
            for (s, sg) in [("l", 1.0), ("r", -1.0)] {
                p.node(
                    &format!("{}{s}", k + 1),
                    DVec3::new(sg * 0.40, yp + 0.009, z),
                    1.5,
                );
            }
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: clip(300.0, INF),
            deform_group: None,
        });
        p.hold(
            "carpet",
            &[
                "c1_1tl", "c1_1tr", "c1_2tl", "c1_2tr", "c1_3tl", "c1_3tr", "c1_4tl", "c1_4tr",
                "c1_5tl", "c1_5tr",
            ],
            12,
            BeamKind::Normal,
            clip(300.0, 600.0),
        );
        out.push(p);
    }
    // ---- pillar trims: A and C under the shell's own section, B on the
    // pillar's cabin face
    let zb = f.z(-0.22);
    for (s, sg) in [("L", 1.0), ("R", -1.0)] {
        let sl = s.to_lowercase();
        let vv = |v: f64| if sg > 0.0 { v } else { tube.vm - v };
        for (nm, za, zc, to) in [
            (
                "a",
                0.74,
                -0.03,
                [
                    format!("d1_m{sl}"),
                    format!("d1_t{sl}"),
                    format!("b2_o{sl}"),
                ],
            ),
            (
                "c",
                -0.99,
                -1.42,
                [
                    format!("d3_t{sl}"),
                    format!("d3_b{sl}"),
                    format!("e5_1u{sl}"),
                ],
            ),
        ] {
            let mut p = MechPart::new(&format!("pillar_trim_{nm}_{s}"), "interior");
            let (nu, nv) = (13, 4);
            let at = |i: usize, j: usize| {
                tube.eval_n(f.z(lin(za, zc, i, nu)), vv(lin(7.12, 7.88, j, nv)))
            };
            p.mesh = sheet_solid(
                nu,
                nv,
                |i, j| at(i, j).0 - at(i, j).1 * 0.036,
                |i, j| -at(i, j).1 * 0.005,
                |_, _| true,
                Mat::InteriorPlastic,
            );
            for (t, i) in [("t", 1), ("m", 6), ("b", 11)] {
                p.node(t, at(i, 1).0 - at(i, 1).1 * 0.038, 0.2);
            }
            p.internal.push(Internal {
                pairs: p.complete(),
                v: clip(200.0, INF),
                deform_group: None,
            });
            let to: Vec<&str> = to.iter().map(|x| x.as_str()).collect();
            p.hold(
                &*format!("pillar_trim_{nm}_{sl}"),
                &to,
                5,
                BeamKind::Normal,
                clip(200.0, 400.0),
            );
            out.push(p);
        }
        {
            let mut p = MechPart::new(&format!("pillar_trim_b_{s}"), "interior");
            let (nu, nv) = (17, 3);
            let at = |i: usize, j: usize| {
                tube.eval(zb + lin(-0.033, 0.033, j, nv), vv(lin(2.3, 6.8, i, nu)))
                    - DVec3::X * (sg * 0.138)
            };
            p.mesh = sheet_solid(
                nu,
                nv,
                at,
                |_, _| -DVec3::X * (sg * 0.005),
                |_, _| true,
                Mat::InteriorPlastic,
            );
            for (t, i) in [("t", 14), ("m", 8), ("b", 2)] {
                p.node(t, at(i, 1) - DVec3::X * (sg * 0.002), 0.2);
            }
            p.internal.push(Internal {
                pairs: p.complete(),
                v: clip(200.0, INF),
                deform_group: None,
            });
            p.hold(
                &*format!("pillar_trim_b_{sl}"),
                &[
                    &*format!("d2_b{sl}"),
                    &*format!("d2_t{sl}"),
                    &*format!("c2_3{sl}"),
                ],
                5,
                BeamKind::Normal,
                clip(200.0, 400.0),
            );
            out.push(p);
        }
        // ---- sun visor under the header
        {
            let mut p = MechPart::new(&format!("sun_visor_{s}"), "interior");
            let (nu, nv) = (9, 5);
            let at = |i: usize, j: usize| {
                let (x, z) = (
                    sg * f.x(lin(0.20, 0.54, i, nu)),
                    f.z(lin(-0.075, -0.19, j, nv)),
                );
                DVec3::new(x, top_y(g, z, x) - 0.074, z)
            };
            p.mesh = sheet_solid(
                nu,
                nv,
                at,
                |_, _| DVec3::Y * 0.014,
                |_, _| true,
                Mat::InteriorFabric,
            );
            for (t, i, j) in [("o", 7, 0), ("i", 1, 0), ("r", 4, 4)] {
                p.node(t, at(i, j) + DVec3::Y * 0.007, 0.35 / 3.0);
            }
            p.internal.push(Internal {
                pairs: p.complete(),
                v: clip(100.0, INF),
                deform_group: None,
            });
            p.hold(
                &*format!("sun_visor_{sl}"),
                &["d4_h", &*format!("d1_t{sl}"), &*format!("d2_t{sl}")],
                4,
                BeamKind::Normal,
                clip(100.0, 250.0),
            );
            out.push(p);
        }
        // ---- front seat belt: the reel at the B-pillar's foot, the webbing
        // up the pillar to its ring, the buckle on its stalk by the tunnel
        {
            let mut p = MechPart::new(&format!("seat_belt_F{s}"), "interior");
            let reel = inside(g, zb, f.y(0.42), sg, 0.178);
            p.mesh.add(&box_(
                reel,
                DVec3::new(0.020, 0.040, 0.030),
                Mat::TrimBlack,
                None,
            ));
            let web: Vec<DVec3> = (0..14)
                .map(|k| inside(g, zb, f.y(0.47 + 0.03 * k as f64), sg, 0.1490))
                .collect();
            let mut w = crate::structure::box_along(&web, DVec3::X, 0.046, 0.002);
            w.set_mat(Mat::InteriorFabric);
            p.mesh.add(&w);
            let ring = inside(g, zb, f.y(0.875), sg, 0.157);
            p.mesh.add(&box_(
                ring,
                DVec3::new(0.006, 0.018, 0.026),
                Mat::TrimBlack,
                None,
            ));
            let zs = 0.5 * (rows[2].z + rows[3].z) - 0.18;
            let buckle = DVec3::new(sg * 0.110, yp + 0.255, zs);
            p.mesh.add(&box_(
                buckle,
                DVec3::new(0.011, 0.030, 0.018),
                Mat::TrimBlack,
                None,
            ));
            p.mesh.add(&box_(
                buckle + DVec3::Y * 0.033,
                DVec3::new(0.008, 0.004, 0.012),
                Mat::Paint,
                None,
            ));
            p.mesh.add(&box_(
                buckle - DVec3::Y * 0.050,
                DVec3::new(0.003, 0.021, 0.010),
                Mat::MetalBare,
                None,
            ));
            let nr = p.node("r", reel, 1.0);
            let ng = p.node("g", ring, 0.2);
            let nb = p.node("b", buckle, 0.3);
            p.internal.push(Internal {
                pairs: p.complete(),
                v: vals(0.0, 30.0, 4e3, INF),
                deform_group: None,
            });
            let anchor = vals(0.0, 60.0, 8e3, 15e3);
            p.hold_from(
                &format!("seat_belt_f{sl}_reel"),
                vec![nr],
                &shn(&[
                    format!("c2_3{sl}"),
                    format!("c1_3o{sl}"),
                    format!("d2_b{sl}"),
                ]),
                3,
                anchor,
            );
            p.hold_from(
                &format!("seat_belt_f{sl}_ring"),
                vec![ng],
                &shn(&[format!("d2_b{sl}"), format!("d2_t{sl}")]),
                2,
                anchor,
            );
            p.hold_from(
                &format!("seat_belt_f{sl}_buckle"),
                vec![nb],
                &shn(&["c3_3".into(), "c3_4".into(), format!("c1_3t{sl}")]),
                3,
                anchor,
            );
            out.push(p);
        }
        // ---- boot side trim: a board inboard of the wheel well and tower
        {
            let mut p = MechPart::new(&format!("boot_trim_{s}"), "interior");
            let x = sg * f.x(0.505);
            let (za, zc) = (g.zr - 0.085, sh(st, "sh_e4_dc").z + 0.040);
            let (ya, yc) = (sh(st, "sh_e3_2").y + 0.020, sh(st, "sh_e6_ol").y - 0.060);
            let (nu, nv) = (10, 6);
            let at = |i: usize, j: usize| DVec3::new(x, lin(ya, yc, j, nv), lin(za, zc, i, nu));
            p.mesh = sheet_solid(
                nu,
                nv,
                at,
                |_, _| -DVec3::X * (sg * 0.006),
                |_, _| true,
                Mat::InteriorFabric,
            );
            for (t, i, j) in [("f", 1, 1), ("r", 8, 1), ("t", 4, 4)] {
                p.node(t, at(i, j), 1.4 / 3.0);
            }
            p.internal.push(Internal {
                pairs: p.complete(),
                v: clip(200.0, INF),
                deform_group: None,
            });
            p.hold(
                &*format!("boot_trim_{sl}"),
                &[
                    &*format!("e2_{sl}"),
                    &*format!("e1_2u{sl}"),
                    &*format!("e1_3u{sl}"),
                    &*format!("e5_3u{sl}"),
                    &*format!("e5_4u{sl}"),
                ],
                6,
                BeamKind::Normal,
                clip(200.0, 500.0),
            );
            out.push(p);
        }
    }
    // ---- interior mirror under the windscreen's top edge
    {
        let mut p = MechPart::new("interior_mirror", "interior");
        let z = f.z(-0.02);
        let c = DVec3::new(0.0, tube.eval(z, 9.0).y - 0.085, z - 0.010);
        p.mesh.add(&rbox(
            c,
            DVec3::new(0.110, 0.030, 0.010),
            0.008,
            2,
            Mat::TrimBlack,
        ));
        p.mesh.add(&box_(
            c - DVec3::Z * 0.0115,
            DVec3::new(0.100, 0.024, 0.001),
            Mat::Chrome,
            None,
        ));
        p.mesh.add(&cyl(
            c + DVec3::new(0.0, 0.020, 0.008),
            c + DVec3::new(0.0, 0.052, 0.022),
            0.006,
        ));
        for (t, x) in [("l", 0.09), ("r", -0.09), ("s", 0.0)] {
            p.node(
                t,
                c + DVec3::new(x, if x == 0.0 { 0.045 } else { 0.0 }, 0.0),
                0.1,
            );
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: clip(100.0, INF),
            deform_group: None,
        });
        p.hold(
            "interior_mirror",
            &["d4_h", "d1_tl", "d1_tr"],
            3,
            BeamKind::Normal,
            clip(100.0, 250.0),
        );
        out.push(p);
    }
    // ---- parcel shelf trim with its two speaker grilles
    let (e6, e6r, e7) = (sh(st, "sh_e6_ol"), sh(st, "sh_e6_rl"), sh(st, "sh_e7_l"));
    {
        let mut p = MechPart::new("parcel_shelf_trim", "interior");
        let (hx, za, zc) = (e6.x - 0.010, e7.z - 0.014, e6r.z + 0.010);
        let (nu, nv) = (21, 6);
        let at =
            |i: usize, j: usize| DVec3::new(lin(-hx, hx, i, nu), e6.y + 0.008, lin(za, zc, j, nv));
        p.mesh = sheet_solid(
            nu,
            nv,
            at,
            |_, _| DVec3::Y * 0.008,
            |_, _| true,
            Mat::InteriorFabric,
        );
        for sg in [1.0, -1.0] {
            p.mesh.add(&lathe_at(
                &[(0.0, 0.0), (0.068, 0.0), (0.064, 0.005), (0.0, 0.007)],
                18,
                Mat::TrimBlack,
                DVec3::Y,
                DVec3::new(sg * 0.40, e6.y + 0.0165, 0.5 * (za + zc)),
            ));
        }
        for (t, i, j) in [("fl", 19, 0), ("fr", 1, 0), ("rl", 19, 5), ("rr", 1, 5)] {
            p.node(t, at(i, j) + DVec3::Y * 0.004, 0.5);
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: clip(200.0, INF),
            deform_group: None,
        });
        p.hold(
            "parcel_shelf_trim",
            &["e6_ol", "e6_or", "e6_rl", "e6_rr", "e7_l", "e7_r"],
            8,
            BeamKind::Normal,
            clip(200.0, 500.0),
        );
        out.push(p);
    }
    // ---- boot floor lining
    {
        let mut p = MechPart::new("boot_lining", "interior");
        let e3 = sh(st, "sh_e3_2");
        let (hx, za, zc) = (f.x(0.497), g.zr - 0.012, sh(st, "sh_e4_dc").z + 0.036);
        let (nu, nv) = (15, 12);
        let at =
            |i: usize, j: usize| DVec3::new(lin(-hx, hx, i, nu), e3.y + 0.008, lin(za, zc, j, nv));
        p.mesh = sheet_solid(
            nu,
            nv,
            at,
            |_, _| DVec3::Y * 0.008,
            |_, _| true,
            Mat::InteriorFabric,
        );
        for (t, i, j) in [("fl", 13, 1), ("fr", 1, 1), ("rl", 13, 10), ("rr", 1, 10)] {
            p.node(t, at(i, j) + DVec3::Y * 0.004, 0.75);
        }
        p.internal.push(Internal {
            pairs: p.complete(),
            v: clip(200.0, INF),
            deform_group: None,
        });
        p.hold(
            "boot_lining",
            &["e3_2", "e3_3", "e1_2ul", "e1_2ur", "e1_3ul", "e1_3ur"],
            8,
            BeamKind::Normal,
            clip(200.0, 500.0),
        );
        out.push(p);
    }
}
