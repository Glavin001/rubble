//! Police interceptor: the sedan in gloss black on steel wheels with a roof
//! light bar, a push bar, an A-pillar spotlight and a boot-lid antenna.
//!
//! A *variant*: it is the sedan (every table, panel, structure and rig
//! fact) plus a paint/rim override and one function adding parts. That is
//! the whole design; port of `tools/vehicle-gen/designs/police_sedan.py`.

use glam::DVec3;

use crate::assemble::{Design, RigSpec};
use crate::carbody::{CarBody, Side};
use crate::export::Marker;
use crate::mesh::{closed_lathe, rbox, Mat, Mesh};
use crate::mechanical::MechPart;
use crate::parts as p;
use crate::structure::Structure;
use crate::tube::Tube;
use crate::uvpack::Parts;

use super::sedan::{Sedan, SedanParams};

const RED: [f64; 4] = [0.85, 0.03, 0.02, 1.0];
const BLUE: [f64; 4] = [0.02, 0.10, 0.90, 1.0];

/// The sedan's parameters with the police livery (a caller's own proportions
/// and physics are kept; only the unset livery fields change).
pub fn livery(mut p: SedanParams) -> SedanParams {
    let d = SedanParams::default();
    if p.paint == d.paint {
        p.paint = [0.010, 0.010, 0.012];
        p.metallic = 0.2;
        p.roughness = 0.18;
    }
    if p.rim_style == d.rim_style {
        p.rim_style = "steel".into();
    }
    p
}

pub struct Police {
    pub base: Sedan,
}

impl Police {
    pub fn new(p: SedanParams) -> Police {
        Police { base: Sedan::new(livery(p)) }
    }

    fn extra_parts(&self, cb: &CarBody, out: &mut Parts) {
        let tube = cb.tube();
        let roof_y = |z: f64, x: f64| {
            let v = tube.v_at_x(z, x.abs(), 7.5, 9.0);
            tube.eval(z, v).y
        };
        // light bar over the front seats: black base on four feet, red lens
        // on the left, blue on the right
        let zc = -0.28;
        let yb = roof_y(zc, 0.0) + 0.045;
        let mut bar = Mesh::new();
        let mut lens = Mesh::new();
        bar.add(&rbox(DVec3::new(0.0, yb, zc), DVec3::new(0.60, 0.016, 0.125), 0.012, 2, Mat::TrimBlack));
        for x in [-0.45, 0.45] {
            for dz in [-0.08, 0.08] {
                let ys = roof_y(zc + dz, x);
                bar.add(&rbox(
                    DVec3::new(x, (ys + 0.004 + yb) / 2.0, zc + dz),
                    DVec3::new(0.022, (yb - ys - 0.004) / 2.0, 0.022),
                    0.006,
                    1,
                    Mat::TrimBlack,
                ));
            }
        }
        for (sg, tint) in [(1.0, RED), (-1.0, BLUE)] {
            let mut l = rbox(DVec3::new(sg * 0.300, yb + 0.049, zc), DVec3::new(0.282, 0.030, 0.112), 0.024, 3, Mat::LampLens);
            l.tint(tint);
            lens.add(&l);
        }
        bar.add(&rbox(DVec3::new(0.0, yb + 0.049, zc), DVec3::new(0.010, 0.030, 0.105), 0.005, 1, Mat::TrimBlack));
        out.insert("lightbar", bar);

        // push bar: two uprights following the nose, two cross tubes, pads
        let mut pb = Mesh::new();
        let ys: Vec<f64> = (0..9).map(|i| 0.30 + 0.44 * i as f64 / 8.0).collect();
        for x in [-0.32f64, 0.32] {
            let pts: Vec<DVec3> = ys.iter().map(|&y| DVec3::new(x, y, self.base.front_z(x.abs(), y) + 0.075)).collect();
            pb.add(&p::tube_pipe(&pts, 0.024, Mat::TrimBlack, 3, 8));
            pb.add(&rbox(
                DVec3::new(x, 0.52, self.base.front_z(x.abs(), 0.52) + 0.112),
                DVec3::new(0.030, 0.17, 0.010),
                0.008,
                1,
                Mat::Rubber,
            ));
            pb.add(&rbox(
                DVec3::new(x, 0.40, self.base.front_z(x.abs(), 0.40) + 0.030),
                DVec3::new(0.030, 0.025, 0.040),
                0.006,
                1,
                Mat::TrimBlack,
            ));
        }
        for y in [0.45, 0.68] {
            let zs: Vec<f64> = (0..9).map(|i| self.base.front_z((-0.32 + 0.64 * i as f64 / 8.0f64).abs(), y) + 0.075).collect();
            let zmax = zs.iter().cloned().fold(f64::MIN, f64::max);
            let pts: Vec<DVec3> = zs
                .iter()
                .enumerate()
                .map(|(i, &z)| DVec3::new(-0.32 + 0.64 * i as f64 / 8.0, y, z.max(zmax - 0.010)))
                .collect();
            pb.add(&p::tube_pipe(&pts, 0.020, Mat::TrimBlack, 3, 8));
        }
        out.insert("push_bar", pb);

        // A-pillar spotlight (driver's side)
        let (base, bn) = p::surface_point(&tube, 0.66, 0.985, 1.0, (5.5, 7.5));
        let c = base + bn * 0.065 + DVec3::new(0.0, 0.02, 0.0);
        let axis = DVec3::new(0.15, 0.05, 1.0);
        let mut sp = Mesh::new();
        let mut can = closed_lathe(&[(0.0, 0.040), (0.050, 0.040), (0.056, 0.0), (0.040, -0.075), (0.0, -0.075)], 28, Mat::Chrome);
        can.transform(Some(p::rot_to(axis)), c);
        sp.add(&can);
        sp.add(&p::tube_pipe(&[base + bn * 0.004, base + bn * 0.035, c - DVec3::new(0.0, 0.0, 0.03)], 0.010, Mat::Chrome, 2, 8));
        out.insert("spotlight", sp);
        let mut lens_ = closed_lathe(&[(0.0, 0.006), (0.044, 0.006), (0.044, 0.002), (0.0, 0.002)], 28, Mat::LampLens);
        lens_.transform(Some(p::rot_to(axis)), c + axis / axis.length() * 0.040);
        lens.add(&lens_);
        out.insert("lightbar_lens", lens);

        // boot-lid antenna
        let (za, xa) = (-2.05, 0.35);
        let ya = tube.eval(za, tube.v_at_x(za, xa, 6.0, 9.0)).y;
        let mut ant = Mesh::new();
        ant.add(&rbox(DVec3::new(xa, ya + 0.014, za), DVec3::new(0.018, 0.010, 0.018), 0.006, 1, Mat::TrimBlack));
        ant.add(&p::tube_pipe(
            &[DVec3::new(xa, ya + 0.024, za), DVec3::new(xa, ya + 0.35, za - 0.02), DVec3::new(xa, ya + 0.70, za - 0.06)],
            0.0025,
            Mat::TrimBlack,
            3,
            8,
        ));
        out.insert("antenna", ant);
    }
}

impl Design for Police {
    fn name(&self) -> String {
        "police_sedan".into()
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
        let (mut out, extras, mk) = self.base.parts(cb);
        self.extra_parts(cb, &mut out);
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
