//! Drag-strip gasser built from the 1969 fastback: the body raked nose-up on
//! skinny front tyres, fat rear slicks on wide rims, a supercharger standing
//! through the bonnet, no wing, flat primer paint.
//!
//! A *variant* of the Mach 1: its livery, its own running gear
//! ([`Running`]: each axle's wheel, placed so the finished car stands on the
//! ground), a [`Pose`] (the body pitched nose-up about the rear axle), no
//! bonnet scoop or wing and the blower. The design is built in its body
//! frame, wheels included, and the pose puts the whole car in the world;
//! the rigger rigs the body frame and poses the rig. Port of
//! `tools/vehicle-gen/designs/mach1_gasser.py`.

use glam::DVec3;

use crate::assemble::{Design, Pose, RigSpec};
use crate::carbody::{CarBody, Side};
use crate::export::Marker;
use crate::mechanical::MechPart;
use crate::mesh::{closed_lathe, rbox, Mat, Mesh};
use crate::parts::{self as p, RimStyle};
use crate::structure::Structure;
use crate::tube::Tube;
use crate::uvpack::Parts;

use super::mach1::{AxleWheel, Mach1, Mach1Params, Running, WHEEL_Y, ZF_AX, ZR_AX};

/// The body pitches nose-up by this much about the rear axle.
const RAKE_DEG: f64 = 2.6;
/// Skinny fronts on steel rims (diameter and width of the tyre, rim).
const FRONT: (f64, f64, f64, f64) = (0.620, 0.165, 15.0 * 0.0254, 4.5 * 0.0254);
/// Slicks on wide magnum rims.
const REAR: (f64, f64, f64, f64) = (0.720, 0.300, 15.0 * 0.0254, 10.0 * 0.0254);

/// The Mach 1's parameters with the primer livery (a caller's own paint is
/// kept). The rim style and the wing are the design's: steel fronts, magnum
/// rears, no wing.
pub fn livery(mut p: Mach1Params) -> Mach1Params {
    if p.paint == Mach1Params::default().paint {
        p.paint = [0.055, 0.058, 0.062];
        p.metallic = 0.0;
        p.roughness = 0.75;
    }
    p.wing = false;
    p
}

/// The stance: pitched about the rear axle, lifted so the rear tyres (the
/// bigger) stand on the ground.
fn pose() -> Pose {
    Pose { pivot_y: WHEEL_Y, pivot_z: ZR_AX, pitch: RAKE_DEG.to_radians(), lift: REAR.0 / 2.0 - WHEEL_Y }
}

/// Each wheel sits on the ground at its own radius, under the car as it
/// stands; in the body frame that is the world's centre pushed back through
/// the pose.
fn running() -> Running {
    let ps = pose();
    let wheel = |t: (f64, f64, f64, f64), z: f64, style: RimStyle| {
        let c = ps.invert(DVec3::new(0.0, t.0 / 2.0, z));
        AxleWheel { y: c.y, z: c.z, tyre_d: t.0, tyre_w: t.1, rim_d: t.2, rim_w: t.3, style }
    };
    Running { front: wheel(FRONT, ZF_AX, RimStyle::Steel), rear: wheel(REAR, ZR_AX, RimStyle::Magnum) }
}

pub struct Gasser {
    pub base: Mach1,
}

impl Gasser {
    pub fn new(p: Mach1Params) -> Gasser {
        Gasser { base: Mach1::with_running(livery(p), running()) }
    }

    /// The blower: a ribbed case on a black manifold, standing through the
    /// bonnet, a chrome bug-catcher scoop on top, drive pulleys and belt.
    fn blower(&self, cb: &CarBody) -> Mesh {
        let tube = cb.tube();
        let (z0, z1) = (1.18, 1.62);
        let mut ymax = f64::MIN;
        for z in crate::curves::linspace(z0, z1, 7) {
            for x in [-0.15f64, 0.0, 0.15] {
                let v = tube.v_at_x(z, x.abs(), 7.0, 9.0);
                ymax = ymax.max(tube.eval(z, v).y);
            }
        }
        let yb = ymax + 0.006;
        let zc = (z0 + z1) / 2.0;
        let mut bl = Mesh::new();
        bl.add(&rbox(DVec3::new(0.0, yb + 0.012, zc), DVec3::new(0.15, 0.012, 0.23), 0.006, 1, Mat::TrimBlack));
        bl.add(&rbox(DVec3::new(0.0, yb + 0.100, zc), DVec3::new(0.13, 0.076, 0.21), 0.035, 3, Mat::MetalBare));
        for z in crate::curves::linspace(zc - 0.17, zc + 0.17, 8) {
            bl.add(&rbox(DVec3::new(0.0, yb + 0.178, z), DVec3::new(0.10, 0.004, 0.006), 0.002, 1, Mat::MetalBare));
        }
        let mut sc = Mesh::new();
        sc.add(&rbox(DVec3::new(0.0, yb + 0.240, zc + 0.02), DVec3::new(0.11, 0.058, 0.13), 0.018, 2, Mat::Chrome));
        sc.add(&rbox(DVec3::new(0.0, yb + 0.252, zc + 0.152), DVec3::new(0.096, 0.044, 0.002), 0.002, 1, Mat::TrimBlack));
        bl.add(&sc);
        for (y, r) in [(yb + 0.160, 0.045), (yb + 0.078, 0.056)] {
            let mut pul = closed_lathe(&[(0.004, 0.0), (r, 0.0), (r, 0.030), (0.004, 0.030)], 28, Mat::Chrome);
            pul.transform(Some(p::rot_to(DVec3::Z)), DVec3::new(0.0, y, z1 + 0.012));
            bl.add(&pul);
        }
        bl.add(&rbox(DVec3::new(0.0, yb + 0.118, z1 + 0.027), DVec3::new(0.060, 0.072, 0.016), 0.03, 2, Mat::Rubber));
        bl
    }
}

impl Design for Gasser {
    fn name(&self) -> String {
        "mach1_gasser".into()
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
        let (mut out, mut extras, mk) = self.base.parts(cb);
        // no bonnet scoop, pins or wing
        extras.insert("hood", Mesh::new());
        extras.insert("trunk_lid", Mesh::new());
        out.insert("blower", self.blower(cb));
        (out, extras, mk)
    }
    fn post(&self, parts: &mut Parts) {
        self.base.post(parts)
    }
    fn pose(&self) -> Option<Pose> {
        Some(pose())
    }
    fn rig_spec(&self) -> Option<RigSpec> {
        self.base.rig_spec().map(|s| RigSpec { pose: Some(pose()), ..s })
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
