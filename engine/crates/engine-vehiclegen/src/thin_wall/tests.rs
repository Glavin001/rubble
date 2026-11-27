//! Closed forms the calculator must reproduce exactly, its invariances, and the two
//! sections checked by hand on 2026-11-27 (the truck's C rail, the Camry's front rail).

use super::crush::*;
use super::*;
use std::f64::consts::PI;

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-30)
}

#[track_caller]
fn assert_close(what: &str, a: f64, b: f64, rel: f64) {
    assert!(close(a, b, rel), "{what}: {a:e} against {b:e} (rel {:e})", (a - b) / b);
}

#[test]
fn a_rectangular_tube_is_its_closed_form() {
    let (b, h, t) = (0.100, 0.060, 0.003);
    let p = ThinSection::rect_tube(b, h, t).properties().unwrap();
    let (bi, hi) = (b - 2.0 * t, h - 2.0 * t);
    assert_close("area", p.area, b * h - bi * hi, 1e-12);
    assert_close("Ixx", p.ixx, (b * h.powi(3) - bi * hi.powi(3)) / 12.0, 1e-12);
    assert_close("Iyy", p.iyy, (h * b.powi(3) - hi * bi.powi(3)) / 12.0, 1e-12);
    assert!(p.ixy.abs() < 1e-20 && p.centroid[0].abs() < 1e-15 && p.centroid[1].abs() < 1e-15);
    assert_close("Wx", p.wx, (b * h.powi(3) - bi * hi.powi(3)) / 12.0 / (h / 2.0), 1e-12);
    assert_close("Zx", p.zx, (b * h * h - bi * hi * hi) / 4.0, 1e-9);
    assert_close("Zy", p.zy, (h * b * b - hi * bi * bi) / 4.0, 1e-9);
    assert!(p.pna_y.abs() < 1e-12);
}

#[test]
fn a_circular_tube_is_its_closed_form() {
    let (d, t) = (0.100, 0.004);
    let p = ThinSection::circular_tube(d, t, 20_000).properties().unwrap();
    let (ro, ri) = (d / 2.0, d / 2.0 - t);
    // chords: the area within (π/n)²/6 = 4e-9, the second moments within about 5 × that
    assert_close("area", p.area, PI * (ro * ro - ri * ri), 1e-8);
    assert_close("I", p.ixx, PI / 4.0 * (ro.powi(4) - ri.powi(4)), 5e-8);
    assert_close("I", p.iyy, p.ixx, 1e-9);
    assert_close("Z", p.zx, ((2.0 * ro).powi(3) - (2.0 * ri).powi(3)) / 6.0, 5e-8);
    assert_close("W", p.wx, PI / 4.0 * (ro.powi(4) - ri.powi(4)) / ro, 1e-7);
}

fn rotated(s: &ThinSection, a: f64) -> ThinSection {
    let (sn, c) = a.sin_cos();
    let mut o = s.clone();
    for w in &mut o.walls {
        for p in &mut w.points {
            *p = [c * p[0] - sn * p[1] + 0.3, sn * p[0] + c * p[1] - 0.2];
        }
    }
    o
}

#[test]
fn properties_do_not_depend_on_where_the_section_is_or_how_it_is_turned() {
    let s = ThinSection::c_channel(0.266, 0.090, 0.008);
    let p = s.properties().unwrap();
    for a in [0.3, 1.0, 2.2] {
        let r = rotated(&s, a).properties().unwrap();
        assert_close("area", r.area, p.area, 1e-12);
        assert_close("I1", r.i1, p.i1, 1e-10);
        assert_close("I2", r.i2, p.i2, 1e-9);
        // the plastic modulus about the turned x axis is the original's about x
        let (_, z) = rotated(&s, a).plastic_modulus_at(a).unwrap();
        assert_close("Z turned", z, p.zx, 1e-9);
        let (_, z) = rotated(&s, a).plastic_modulus_at(a + PI / 2.0).unwrap();
        assert_close("Z turned, other axis", z, p.zy, 1e-9);
    }
    // a section symmetric about x has no product of inertia and its principal axes are x, y
    assert!(p.ixy.abs() < 1e-18 && p.principal_angle.abs() < 1e-9);
    // mirrored left to right: the same second moments, the product changes sign
    let un = ThinSection::hat(0.10, 0.06, 0.02, 0.0015, Some(0.0012));
    let mut m = un.clone();
    for w in &mut m.walls {
        for p in &mut w.points {
            p[0] = -p[0];
        }
        w.points.reverse();
    }
    let (a, b) = (un.properties().unwrap(), m.properties().unwrap());
    assert_close("mirror Ixx", a.ixx, b.ixx, 1e-12);
    assert_close("mirror Zx", a.zx, b.zx, 1e-9);
}

/// The truck's side member (`generated-vehicles-lessons.md` §23): C 266 × 90 × 8 mm in
/// S500MC. By hand: A 3,440 mm², I 3.44e7 mm⁴, Z_pl 3.11e5 mm³, M_p 155 kN·m.
#[test]
fn the_truck_rail_c_channel() {
    let s = ThinSection::c_channel(0.266, 0.090, 0.008);
    let p = s.properties().unwrap();
    assert_close("area", p.area, 3440e-6, 1e-12);
    assert_close("I", p.ixx, (0.090 * 0.266f64.powi(3) - 0.082 * 0.250f64.powi(3)) / 12.0, 1e-12);
    assert!(close(p.ixx, 3.44e-5, 2e-3));
    assert_close("Z_pl", p.zx, 2.0 * 0.090 * 0.008 * 0.129 + 0.008 * 0.250 * 0.250 / 4.0, 1e-9);
    let st = p.strengths(&Material::steel(500e6, 600e6), 1.0);
    assert!(close(st.mp_x, 155e3, 3e-3), "M_p {}", st.mp_x);
    assert_close("squash", st.squash_n, 1.72e6, 1e-12);
    // the elastic modulus over the plastic: the C's shape factor 1.20
    assert!(close(p.zx / p.wx, 1.20, 0.01), "{}", p.zx / p.wx);
}

/// The Camry's front rail at its crash box (`fe-import-spike.md` §4): 92 × 166 mm, 1.69 mm,
/// σ_y 360, σ₀ 499 MPa: 57 kN at σ_y, 79 kN at σ₀, 120 kN with the deck's rate scaling at
/// 15.6 m/s. A regression of the formulas' implementation, not of their truth.
#[test]
fn the_camry_rail_crush_numbers() {
    let (b, h, t) = (0.092, 0.166, 0.00169);
    let s0 = rect_tube_mean_crush(499e6, t, b, h);
    assert!(close(s0, 79e3, 0.01), "{s0}");
    assert!(close(rect_tube_mean_crush(360e6, t, b, h), 57e3, 0.01));
    let f = dynamic_factor(15.6, 0.5 * (b + h), CS_CAMRY_DECK);
    assert!(close(f, 1.515, 0.005), "{f}");
    assert!(close(s0 * f, 120e3, 0.01), "{}", s0 * f);
    // the classic mild-steel pair would give about 2.0
    let fm = dynamic_factor(15.6, 0.5 * (b + h), CS_MILD_STEEL);
    assert!(fm > 1.9 && fm < 2.1, "{fm}");
    // the superfolding element is the same law without the effective crush distance
    assert!(close(wierzbicki_abramowicz(499e6, t, 0.129) / abramowicz_jones(499e6, t, 0.129), 0.73, 0.01));
}

/// `Member::crush_n` is the calculator's Abramowicz–Jones force, bit for bit.
#[test]
fn the_generators_crush_force_is_the_calculators() {
    use crate::structure::{Grade, Member, Section};
    for (w, h, g, grade) in [(0.07, 0.11, 1.5e-3, Grade::Hsla), (0.04, 0.06, 1.0e-3, Grade::Mild), (0.06, 0.09, 1.6e-3, Grade::Dp600)] {
        let m = Member { name: "m".into(), kind: "front_rail", part: "x", section: Section::Box { w, h }, gauge: g, grade, length: 1.0, sheet_area: 1.0, mesh: crate::mesh::Mesh::new(), on_shell: false };
        assert_eq!(m.crush_n().to_bits(), rect_tube_mean_crush(grade.flow_pa(), g, w, h).to_bits());
    }
}

#[test]
fn a_hat_on_its_plate_adds_the_plate() {
    let (w, h, fl, t, tp) = (0.10, 0.06, 0.02, 0.0015, 0.0012);
    let open = ThinSection::hat(w, h, fl, t, None).properties().unwrap();
    let closed = ThinSection::hat(w, h, fl, t, Some(tp)).properties().unwrap();
    // the hat's own area: its midline length × t (mitred corners keep it exact)
    let hat_len = 2.0 * fl + (w - t) + 2.0 * (h - t);
    assert_close("hat area", open.area, hat_len * t, 1e-12);
    assert_close("plate", closed.area - open.area, (w + 2.0 * fl) * tp, 1e-12);
    assert!(closed.centroid[1] < open.centroid[1]);
}

#[test]
fn effective_width_and_box_bending() {
    let m = Material::steel(340e6, 450e6);
    // a stocky flange is fully effective, a slender one is not
    assert_eq!(winter_effective_width(0.03, 0.002, m.e, m.yield_pa), 1.0);
    let r = winter_effective_width(0.20, 0.001, m.e, m.yield_pa);
    assert!(r > 0.2 && r < 0.6, "{r}");
    let (mc, rho) = box_bending_collapse(0.20, 0.10, 0.001, &m).unwrap();
    let mp = m.yield_pa * ThinSection::rect_tube(0.20, 0.10, 0.001).properties().unwrap().zx;
    assert!(rho < 1.0 && mc < mp && mc > 0.5 * mp, "{mc} {mp} {rho}");
    let (mc, rho) = box_bending_collapse(0.07, 0.11, 0.0018, &m).unwrap();
    let mp = m.yield_pa * ThinSection::rect_tube(0.07, 0.11, 0.0018).properties().unwrap().zx;
    assert!(rho <= 1.0 && mc <= mp * (1.0 + 1e-12));
}

#[test]
fn bad_walls_are_refused() {
    let w = |points: Vec<P2>, closed| ThinSection { walls: vec![Wall { points, t: 0.002, closed }] }.properties();
    assert!(w(vec![[0.0, 0.0]], false).is_err());
    assert!(w(vec![[0.0, 0.0], [0.0, 0.0]], false).is_err());
    assert!(w(vec![[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]], false).is_err());
}
