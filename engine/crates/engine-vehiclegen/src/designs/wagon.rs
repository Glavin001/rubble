//! Estate wagon: the sedan's nose, doors and wheels with the roof carried to
//! the tail, long cargo windows behind the rear doors, a near-vertical
//! tailgate carrying the back light between tall lamps, and roof rails.
//!
//! A *variant* of the sedan: three key-line tables, a tail face, a panel
//! rule, a tail layout and the roof rails. Port of
//! `tools/vehicle-gen/designs/wagon.py`.

use glam::{DVec2, DVec3};

use crate::assemble::{Design, RigSpec};
use crate::carbody::{CapPt, CarBody, Side};
use crate::curves::{linspace, Pchip};
use crate::export::Marker;
use crate::fx::{FxHashMap, FxHashSet};
use crate::mechanical::MechPart;
use crate::mesh::{rbox, Mat, Mesh};
use crate::parts as p;
use crate::structure::Structure;
use crate::tube::Tube;
use crate::uvpack::Parts;

use super::sedan::{Sedan, SedanParams, Variant};

const ZREAR: f64 = -2.470;

pub fn livery(mut p: SedanParams) -> SedanParams {
    if p.paint == SedanParams::default().paint {
        p.paint = [0.30, 0.31, 0.32];
        p.metallic = 0.8;
        p.roughness = 0.30;
    }
    p
}

/// Tailgate face: leans forward 0.12 m from the bumper to the roof's rear
/// edge (reference coordinates).
fn rear_ref(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let tuck = 0.060 * ((0.44 - y) / 0.14).clamp(0.0, 1.0);
    let lean = 0.120 * ((y - 0.90) / 0.50).clamp(0.0, 1.0).powf(1.4);
    let corner = 0.090 * (x / 0.89).clamp(0.0, 1.0).powi(4);
    ZREAR + tuck + lean + corner
}

/// Cargo windows from the rear door to the D-pillar; no boot lid.
fn assign(zf: &str, lo: i64, side: Side) -> Option<String> {
    let s = side.tag();
    if matches!(zf, "AR1" | "AR2" | "AR3") && lo == 8 {
        return Some(format!("glass_quarter_{s}"));
    }
    // (the rear door's window column, RQ at 8 and 9, stays the door's: the
    // sedan's rule, so the door keeps its outline)
    if matches!(zf, "RQ" | "AR1" | "AR2" | "AR3" | "TL") && lo >= if zf == "RQ" { 10 } else { 8 } {
        return Some("body_shell".into());
    }
    None
}

/// Tall lamps in the D-pillar columns, the tailgate between them with the
/// back light, bumper cover below.
fn build_tail(_s: &Sedan, cb: &mut CarBody) {
    let it = cb.zi("T");
    let j = cb.jn;
    let nt = |k: usize| cb.nid(it, k % j);
    let left: Vec<CapPt> = (1..=10).map(|k| CapPt::N(nt(k))).collect();
    let right: Vec<CapPt> = (1..=10).map(|k| CapPt::N(nt(j - k))).collect();
    let top_ids = [nt(10), nt(11), nt(12), nt(j - 11), nt(j - 10)];
    let top: Vec<CapPt> = top_ids.iter().map(|&n| CapPt::N(n)).collect();
    let y0 = cb.l.node_pos(nt(1)).y;
    let xg = 0.66;
    let bottom = vec![CapPt::N(nt(1)), CapPt::P(xg, y0), CapPt::P(0.0, y0), CapPt::P(-xg, y0), CapPt::N(nt(j - 1))];
    let r_max = left.len() - 1;
    let mut ov: FxHashMap<(usize, usize), DVec2> = FxHashMap::default();
    // rows up to the belt run level; the back-light and frame rows keep the
    // pillar's distances below the roof edge
    let lpos: Vec<DVec2> = (1..=10).map(|k| cb.l.node_pos(nt(k))).collect::<Vec<_>>().into_iter().map(|q| DVec2::new(q.x, q.y)).collect();
    let tpos: Vec<DVec2> = top_ids.iter().map(|&n| cb.l.node_pos(n)).map(|q| DVec2::new(q.x, q.y)).collect();
    let mut tsorted = tpos.clone();
    tsorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let (tx, ty): (Vec<f64>, Vec<f64>) = (tsorted.iter().map(|q| q.x).collect(), tsorted.iter().map(|q| q.y).collect());
    for r in 1..r_max {
        // the D-pillar narrows going up with the roof edge
        let x1 = xg.min(lpos[r].x - 0.05);
        for (c, x) in [(1usize, x1), (2, 0.0), (3, -x1)] {
            let y = if r >= 7 {
                crate::curves::interp(x, &tx, &ty) - (lpos[r_max].y - lpos[r].y)
            } else {
                lpos[r].y
            };
            ov.insert((r, c), DVec2::new(x, y));
        }
    }
    let rpart = move |r: usize, c: usize| -> Option<String> {
        if r <= 1 {
            return Some("bumper_rear".into());
        }
        if c == 0 || c == 3 {
            if (3..=6).contains(&r) {
                return Some(format!("taillight_{}_lens", if c == 0 { "L" } else { "R" }));
            }
            return Some("body_shell".into());
        }
        if r == 7 {
            return Some("glass_rear".into());
        }
        if r == r_max - 1 {
            return Some("body_shell".into());
        }
        Some("tailgate".into())
    };
    let s2 = _s.clone();
    cb.cap("capR", std::rc::Rc::new(move |x, y| s2.rear_z(x, y)), -1.0, &left, &right, &bottom, &top, &rpart, &FxHashSet::default(), &ov, Vec::new());
}

fn variant() -> Variant {
    let t = |v: &[(f64, f64)]| Pchip::new(v);
    Variant {
        keys: vec![
            // roof, rail, glass edge and centre line: the sedan's forward of
            // the B-pillar, then level to the tail
            (
                7,
                t(&[(2.45, 0.480), (2.0, 0.560), (1.415, 0.620), (0.88, 0.665), (0.78, 0.705), (0.0, 0.650), (-0.5, 0.645), (-1.2, 0.648), (-2.0, 0.640), (-2.55, 0.625)]),
                t(&[(2.45, 0.798), (2.20, 0.834), (1.80, 0.874), (1.415, 0.908), (0.88, 0.945), (0.78, 0.985), (0.30, 1.235), (-0.10, 1.355), (-0.5, 1.385), (-1.2, 1.380), (-2.0, 1.360), (-2.55, 1.340)]),
            ),
            (
                8,
                t(&[(2.45, 0.400), (2.0, 0.500), (1.415, 0.560), (0.88, 0.600), (0.78, 0.645), (0.0, 0.595), (-0.5, 0.590), (-1.2, 0.592), (-2.0, 0.585), (-2.55, 0.570)]),
                t(&[(2.45, 0.802), (2.20, 0.838), (1.80, 0.878), (1.415, 0.913), (0.88, 0.950), (0.78, 0.990), (0.30, 1.250), (-0.10, 1.385), (-0.5, 1.412), (-1.2, 1.408), (-2.0, 1.388), (-2.55, 1.368)]),
            ),
            (
                9,
                Pchip::constant(0.0),
                t(&[(2.45, 0.808), (2.20, 0.845), (1.80, 0.886), (1.415, 0.922), (0.88, 0.958), (0.78, 0.995), (0.30, 1.262), (-0.10, 1.415), (-0.40, 1.450), (-1.2, 1.445), (-2.0, 1.425), (-2.55, 1.405)]),
            ),
        ],
        rear_ref: Some(rear_ref),
        assign: Some(assign),
        tail: Some(build_tail),
        tailgate: true,
    }
}

pub struct Wagon {
    pub base: Sedan,
}

impl Wagon {
    pub fn new(p: SedanParams) -> Wagon {
        Wagon { base: Sedan::with_variant(livery(p), variant()) }
    }

    fn roof_rails(&self, cb: &CarBody, out: &mut Parts) {
        let tube = cb.tube();
        let eval_at = |z: f64| tube.eval(z, tube.v_at_x(z, 0.56, 7.0, 9.0)).y;
        let mut rails = Mesh::new();
        for sg in [1.0, -1.0] {
            let pts: Vec<DVec3> = linspace(0.05, -2.15, 12).iter().map(|&z| DVec3::new(sg * 0.56, eval_at(z) + 0.055, z)).collect();
            rails.add(&p::tube_pipe(&pts, 0.013, Mat::TrimBlack, 3, 8));
            for z in [0.0, -2.10] {
                rails.add(&rbox(DVec3::new(sg * 0.56, eval_at(z) + 0.030, z), DVec3::new(0.018, 0.026, 0.045), 0.010, 2, Mat::TrimBlack));
            }
        }
        out.insert("roof_rails", rails);
    }
}

impl Design for Wagon {
    fn name(&self) -> String {
        "wagon".into()
    }
    fn paint(&self) -> ([f64; 3], f64, f64) {
        self.base.paint()
    }
    fn tube(&self) -> Tube {
        self.base.tube()
    }
    fn zlines(&self) -> Vec<(&'static str, f64)> {
        self.base.zlines()
    }
    fn vlines(&self) -> Vec<f64> {
        self.base.vlines()
    }
    fn assign(&self, zf: &str, zt: &str, lo: usize, hi: usize, side: Side) -> Option<(String, Mat)> {
        self.base.assign(zf, zt, lo, hi, side)
    }
    fn build_layout(&self, cb: &mut CarBody) {
        self.base.build_layout(cb)
    }
    fn build_caps(&self, cb: &mut CarBody) {
        self.base.build_caps(cb)
    }
    fn lens_tint(&self, name: &str) -> [f64; 4] {
        self.base.lens_tint(name)
    }
    fn parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let (mut out, extras, mut mk) = self.base.parts(cb);
        self.roof_rails(cb, &mut out);
        // the boot-lid markers become tailgate markers
        mk.retain(|m| !m.name.starts_with("hinge_trunk_") && m.name != "latch_trunk");
        let f = &self.base.f;
        for (s, sg) in [("L", 1.0), ("R", -1.0)] {
            mk.push(Marker { name: format!("hinge_tailgate_{s}"), pos: f.p(sg * 0.45, 1.36, -2.40), xaxis: DVec3::X });
        }
        let z = self.base.rear_z(0.0, f.y(0.62)) + 0.03;
        mk.push(Marker { name: "latch_tailgate".into(), pos: DVec3::new(0.0, f.y(0.62), z), xaxis: DVec3::X });
        (out, extras, mk)
    }
    fn post(&self, parts: &mut Parts) {
        self.base.post(parts)
    }
    fn rig_spec(&self) -> Option<RigSpec> {
        self.base.rig_spec()
    }
    fn structure(&self) -> Option<Structure> {
        self.base.structure()
    }
    fn mechanical(&self, st: &Structure) -> Vec<MechPart> {
        self.base.mechanical(st)
    }
    fn solid_style(&self, name: &str) -> crate::solid::Style {
        self.base.solid_style(name)
    }
}
