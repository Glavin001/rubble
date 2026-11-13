//! The art spec's car: a generic modern mid-size four-door sedan, our own
//! design (no make's styling, badges or logos), with its proportions as
//! parameters. Port of `engine/tools/vehicle-gen/designs/sedan.py`: the
//! tables are the reference car (wheelbase 2.83 m, length 4.88 m, width
//! 1.86 m, height 1.45 m, 235/45 R18) and [`Frame`] maps them onto the
//! requested proportions. At the reference parameters the output matches
//! the Python generator.

use crate::fx::{FxHashMap, FxHashSet};

use std::rc::Rc;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::assemble::{Design, RigSpec};
use crate::physics::Physics;
use crate::carbody::{CapPt, CarBody, Side};
use crate::curves::{bump, interp, linspace, sign, smoothstep, Pchip};
use crate::export::Marker;
use crate::frame::{Frame, RefDims, Target};
use crate::layout::{Curve, FaceFn};
use crate::mesh::{box_, closed_lathe, grid, orient_outward, rbox, rot_x, rrect, sweep, Grid2, Mat, Mesh, SweepOpts};
use crate::panels::{boundary_loops, inner_panel};
use crate::parts::{self as p, RimStyle, WheelSpec};
use crate::tube::{Handle, Key, Tube};
use crate::uvpack::Parts;

// -- the reference car ---------------------------------------------------------------
const R_WHEELBASE: f64 = 2.830;
const R_TRACK: f64 = 1.600;
const R_TYRE_D: f64 = 0.683;
const R_FRONT: f64 = 2.385; // ZFRONT
const R_REAR: f64 = -2.495; // ZREAR
const R_WIDTH: f64 = 1.860;
const R_HEIGHT: f64 = 1.450;
const R_ARCH_R: f64 = 0.398;

fn refdims() -> RefDims {
    RefDims { axle_f: R_WHEELBASE / 2.0, axle_r: -R_WHEELBASE / 2.0, tip: R_FRONT, tail: R_REAR, arch_zone: 0.55 }
}

/// The sedan's parameters (metres unless noted). Defaults are the reference
/// car.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SedanParams {
    pub wheelbase: f64,
    pub front_overhang: f64,
    pub rear_overhang: f64,
    /// body width (excluding mirrors)
    pub width: f64,
    pub track: f64,
    /// roof height above the ground at the reference ride height
    pub height: f64,
    /// body raised (+) or lowered (-) relative to the wheels
    pub ride: f64,
    pub tyre_d: f64,
    pub tyre_w: f64,
    /// rim diameter and width in inches
    pub rim_in: f64,
    pub rim_w_in: f64,
    pub rim_style: String,
    /// linear RGB
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// The crash physics block (masses, zone yields, hardware; census
    /// defaults): `engine/docs/generated-vehicles.md`.
    pub physics: Physics,
}

impl Default for SedanParams {
    fn default() -> Self {
        SedanParams {
            wheelbase: R_WHEELBASE,
            front_overhang: R_FRONT - R_WHEELBASE / 2.0,
            rear_overhang: -R_WHEELBASE / 2.0 - R_REAR,
            width: R_WIDTH,
            track: R_TRACK,
            height: R_HEIGHT,
            ride: 0.0,
            tyre_d: R_TYRE_D,
            tyre_w: 0.235,
            rim_in: 18.0,
            rim_w_in: 8.0,
            rim_style: "alloy".into(),
            paint: [0.22, 0.006, 0.012],
            metallic: 0.5,
            roughness: 0.24,
            physics: Physics::default(),
        }
    }
}

impl SedanParams {
    /// Ranges the design holds together in (each wheel arch keeps its
    /// undistorted zone, the cabin keeps its door lines in order).
    pub fn validate(&self) -> Result<(), String> {
        let chk = |name: &str, v: f64, lo: f64, hi: f64| {
            if v.is_finite() && v >= lo && v <= hi {
                Ok(())
            } else {
                Err(format!("{name} = {v} is outside {lo}..{hi}"))
            }
        };
        chk("wheelbase", self.wheelbase, 2.2, 3.4)?;
        chk("front_overhang", self.front_overhang, 0.70, 1.30)?;
        chk("rear_overhang", self.rear_overhang, 0.75, 1.40)?;
        chk("width", self.width, 1.55, 2.10)?;
        chk("track", self.track, 1.35, 1.85)?;
        chk("height", self.height, 1.20, 1.80)?;
        // lowering uses up the gap between the tyre and the wheel-well liner
        // (it grows with the tyre: 33.5 mm at every tyre size; keep 13.5)
        chk("ride", self.ride, -0.020, 0.20)?;
        chk("tyre_d", self.tyre_d, 0.55, 0.85)?;
        chk("tyre_w", self.tyre_w, 0.15, 0.35)?;
        chk("rim_in", self.rim_in, 13.0, 22.0)?;
        chk("rim_w_in", self.rim_w_in, 5.0, 11.0)?;
        chk("metallic", self.metallic, 0.0, 1.0)?;
        chk("roughness", self.roughness, 0.0, 1.0)?;
        if self.rim_in * 0.0254 > self.tyre_d - 0.12 {
            return Err(format!("rim {} in leaves no tyre sidewall on a {} m tyre", self.rim_in, self.tyre_d));
        }
        if self.track / 2.0 + self.tyre_w / 2.0 > self.width / 2.0 + 0.03 {
            return Err("the wheels stick out of the body: widen the body or narrow the track".into());
        }
        if self.track / 2.0 - self.tyre_w / 2.0 < 0.56 * self.width / 1.86 + 0.04 {
            return Err("the wheels hit the wheel wells: widen the track or narrow the tyres".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Sedan {
    pub p: SedanParams,
    pub f: Frame,
    zf_ax: f64,
    zr_ax: f64,
    wheel_y: f64,
    arch_r: f64,
}

impl Sedan {
    pub fn new(p: SedanParams) -> Sedan {
        let wheel_y = p.tyre_d / 2.0;
        let dy = p.ride + (wheel_y - R_TYRE_D / 2.0);
        // roof: scale above the shoulder so the top lands at `height` (+ dy)
        let y0 = 0.95;
        let blend = |y: f64| if y <= y0 { 0.0 } else if y < y0 + 0.1 { (y - y0).powi(2) / 0.2 } else { y - y0 - 0.05 };
        let kr = 1.0 + (p.height - R_HEIGHT) / blend(R_HEIGHT);
        let arch_r = R_ARCH_R + (p.tyre_d - R_TYRE_D) / 2.0;
        let f = Frame::new(
            refdims(),
            Target {
                wheelbase: p.wheelbase,
                front_overhang: p.front_overhang,
                rear_overhang: p.rear_overhang,
                sx: p.width / R_WIDTH,
                dy,
                kr,
                y0,
                arch_k: arch_r / R_ARCH_R,
            },
        );
        Sedan { zf_ax: p.wheelbase / 2.0, zr_ax: -p.wheelbase / 2.0, wheel_y, arch_r, f, p }
    }

    fn arch_centres(&self) -> [(f64, f64); 2] {
        let yc = self.f.y(R_TYRE_D / 2.0);
        [(self.zf_ax, yc), (self.zr_ax, yc)]
    }

    // -- end faces (reference functions mapped) -------------------------------------
    fn front_z_ref(x: f64, y: f64) -> f64 {
        let x = x.abs();
        let lean = 0.075 * ((0.58 - y) / 0.28).clamp(0.0, 1.0).powf(1.3);
        let top = 0.085 * ((y - 0.62) / 0.19).clamp(0.0, 1.0).powf(1.6);
        let corner = 0.150 * (x / 0.88).clamp(0.0, 1.0).powi(3);
        R_FRONT - lean - top - corner
    }

    fn rear_z_ref(x: f64, y: f64) -> f64 {
        let x = x.abs();
        let tuck = 0.060 * ((0.44 - y) / 0.14).clamp(0.0, 1.0);
        let top = 0.030 * ((y - 0.90) / 0.12).clamp(0.0, 1.0);
        let corner = 0.120 * (x / 0.89).clamp(0.0, 1.0).powi(3);
        R_REAR + tuck + top + corner
    }

    pub fn front_z(&self, x: f64, y: f64) -> f64 {
        self.f.z(Self::front_z_ref(self.f.xi(x), self.f.yi(y)))
    }

    pub fn rear_z(&self, x: f64, y: f64) -> f64 {
        self.f.z(Self::rear_z_ref(self.f.xi(x), self.f.yi(y)))
    }

    fn front_fn(&self) -> FaceFn {
        let s = self.clone();
        Rc::new(move |x, y| s.front_z(x, y))
    }

    fn rear_fn(&self) -> FaceFn {
        let s = self.clone();
        Rc::new(move |x, y| s.rear_z(x, y))
    }

    fn wheel_spec(&self) -> WheelSpec {
        let rim_style = match self.p.rim_style.as_str() {
            "steel" => RimStyle::Steel,
            "magnum" => RimStyle::Magnum,
            _ => RimStyle::Alloy,
        };
        WheelSpec {
            track: self.p.track,
            wheel_y: self.wheel_y,
            tyre_d: self.p.tyre_d,
            tyre_w: self.p.tyre_w,
            rim_d: self.p.rim_in * 0.0254,
            rim_w: self.p.rim_w_in * 0.0254,
            rim_style,
            disc_d: 0.32,
        }
    }

    /// The reference key tables: (x table or constant, y table).
    fn ref_keys() -> Vec<(Pchip, Pchip)> {
        let w = Pchip::new(&[(2.42, 0.835), (2.34, 0.885), (2.15, 0.915), (1.415, 0.929), (0.0, 0.930), (-1.415, 0.930), (-2.10, 0.916), (-2.38, 0.890), (-2.52, 0.850)]);
        let zs = linspace(-2.6, 2.5, 52);
        let off = |d: f64| Pchip::new(&zs.iter().map(|&z| (z, w.eval(z) - d)).collect::<Vec<_>>());
        let c = Pchip::constant;
        let t = |v: &[(f64, f64)]| Pchip::new(v);
        vec![
            (c(0.0), c(0.195)),
            // 1 sill bottom / bumper bottom
            (off(0.070), t(&[(2.45, 0.300), (1.90, 0.300), (1.0, 0.205), (-1.0, 0.205), (-1.90, 0.320), (-2.55, 0.320)])),
            // 2 door bottom
            (off(0.034), t(&[(2.45, 0.360), (1.90, 0.360), (1.0, 0.272), (-1.0, 0.272), (-1.90, 0.400), (-2.55, 0.420)])),
            // 3 widest line / bumper top front
            (off(0.0), t(&[(2.45, 0.500), (1.0, 0.480), (-1.0, 0.500), (-2.55, 0.560)])),
            // 4 waist crease (rising), headlamp bottom on the nose
            (off(0.004), t(&[(2.45, 0.665), (2.20, 0.700), (1.80, 0.735), (1.415, 0.760), (0.0, 0.785), (-1.415, 0.805), (-2.55, 0.820)])),
            // 5 shoulder
            (
                off(0.036),
                t(&[(2.45, 0.775), (2.20, 0.805), (1.80, 0.840), (1.415, 0.865), (0.88, 0.905), (0.0, 0.930), (-1.415, 0.950), (-2.00, 0.958), (-2.55, 0.948)]),
            ),
            // 6 belt (glass base) / bonnet shut line / boot lid edge
            (
                t(&[(2.45, 0.600), (2.20, 0.660), (1.80, 0.720), (0.88, 0.790), (0.78, 0.800), (0.0, 0.808), (-1.0, 0.818), (-1.60, 0.818), (-2.55, 0.800)]),
                t(&[
                    (2.45, 0.790),
                    (2.20, 0.825),
                    (1.80, 0.865),
                    (1.415, 0.900),
                    (0.88, 0.940),
                    (0.78, 0.975),
                    (0.0, 0.990),
                    (-1.0, 1.010),
                    (-1.60, 1.022),
                    (-2.0, 1.012),
                    (-2.55, 1.000),
                ]),
            ),
            // 7 roof rail outer / daylight-opening top / C-pillar outer
            (
                t(&[
                    (2.45, 0.480),
                    (2.0, 0.560),
                    (1.415, 0.620),
                    (0.88, 0.665),
                    (0.78, 0.705),
                    (0.0, 0.650),
                    (-0.5, 0.645),
                    (-1.0, 0.665),
                    (-1.3, 0.705),
                    (-1.70, 0.770),
                    (-2.0, 0.640),
                    (-2.55, 0.580),
                ]),
                t(&[
                    (2.45, 0.798),
                    (2.20, 0.834),
                    (1.80, 0.874),
                    (1.415, 0.908),
                    (0.88, 0.945),
                    (0.78, 0.985),
                    (0.30, 1.235),
                    (-0.10, 1.355),
                    (-0.5, 1.385),
                    (-1.0, 1.345),
                    (-1.3, 1.250),
                    (-1.70, 1.066),
                    (-2.0, 1.058),
                    (-2.55, 1.052),
                ]),
            ),
            // 8 windscreen / back-light edge
            (
                t(&[
                    (2.45, 0.400),
                    (2.0, 0.500),
                    (1.415, 0.560),
                    (0.88, 0.600),
                    (0.78, 0.645),
                    (0.0, 0.595),
                    (-0.5, 0.590),
                    (-1.0, 0.595),
                    (-1.3, 0.625),
                    (-1.70, 0.665),
                    (-2.0, 0.520),
                    (-2.55, 0.470),
                ]),
                t(&[
                    (2.45, 0.802),
                    (2.20, 0.838),
                    (1.80, 0.878),
                    (1.415, 0.913),
                    (0.88, 0.950),
                    (0.78, 0.990),
                    (0.30, 1.250),
                    (-0.10, 1.385),
                    (-0.5, 1.412),
                    (-1.0, 1.372),
                    (-1.3, 1.262),
                    (-1.70, 1.075),
                    (-2.0, 1.064),
                    (-2.55, 1.058),
                ]),
            ),
            // 9 centre line
            (
                c(0.0),
                t(&[
                    (2.45, 0.808),
                    (2.20, 0.845),
                    (1.80, 0.886),
                    (1.415, 0.922),
                    (0.88, 0.958),
                    (0.78, 0.995),
                    (0.30, 1.262),
                    (-0.10, 1.415),
                    (-0.40, 1.450),
                    (-0.70, 1.440),
                    (-1.00, 1.395),
                    (-1.30, 1.272),
                    (-1.70, 1.082),
                    (-2.00, 1.070),
                    (-2.55, 1.064),
                ]),
            ),
        ]
    }
}

const VLINES: [f64; 13] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.33, 5.67, 6.0, 6.85, 7.0, 8.0, 9.0];
const ZLINES: [(&str, f64); 17] = [
    ("F", 2.37),
    ("HL", 2.06),
    ("AF1", 1.81),
    ("AF2", R_WHEELBASE / 2.0),
    ("AF3", 1.02),
    ("DF", 0.88),
    ("WB", 0.78),
    ("WT", -0.10),
    ("FG", -0.17),
    ("BP", -0.22),
    ("RG", -0.27),
    ("RQ", -0.97),
    ("AR1", -1.02),
    ("AR2", -R_WHEELBASE / 2.0),
    ("AR3", -1.81),
    ("TL", -2.20),
    ("T", -2.47),
];

impl Design for Sedan {
    fn name(&self) -> String {
        "sedan".into()
    }

    fn paint(&self) -> ([f64; 3], f64, f64) {
        (self.p.paint, self.p.metallic, self.p.roughness)
    }

    fn tube(&self) -> Tube {
        let f = Rc::new(self.f.clone());
        let keys: Vec<Key> = Self::ref_keys()
            .into_iter()
            .map(|(px, py)| {
                if f.is_identity() {
                    Key { x: Rc::new(move |z| px.eval(z)), y: Rc::new(move |z| py.eval(z)) }
                } else {
                    let (f1, f2) = (f.clone(), f.clone());
                    Key { x: Rc::new(move |z| f1.x(px.eval(f1.zi(z)))), y: Rc::new(move |z| f2.y(py.eval(f2.zi(z)))) }
                }
            })
            .collect();
        let hs = [(1.0 / 3.0, 1.0 / 3.0), (0.30, 0.30), (0.33, 0.33), (0.36, 0.36), (0.18, 0.18), (0.22, 0.22), (0.24, 0.26), (0.33, 0.33), (0.33, 0.33), (1.0 / 3.0, 1.0 / 3.0)];
        let handles = hs.iter().map(|&(a, b)| (Handle::C(a), Handle::C(b))).collect();
        // displacements, shaped in reference coordinates
        let arches = self.arch_centres();
        let rr = self.arch_r;
        let arch_lip: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
            let side = if n.x.abs() > 0.5 { 1.0 } else { 0.0 };
            let mut out = DVec3::ZERO;
            for (zc, yc) in arches {
                let r = ((z - zc).powi(2) + (pp.y - yc).powi(2)).sqrt();
                let lip = 0.006 * bump(r, rr + 0.010, 0.050) * if pp.y > yc - 0.1 { 1.0 } else { 0.0 };
                out.x += sign(pp.x) * lip * side;
            }
            out
        });
        let f2 = f.clone();
        let lower: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
            let zr = f2.zi(z);
            let side = ((n.x.abs() - 0.6) / 0.3).clamp(0.0, 1.0);
            let zw = ((zr + 1.0) / 0.2).clamp(0.0, 1.0) * ((1.0 - zr) / 0.2).clamp(0.0, 1.0);
            let yl = f2.y(0.40 + 0.04 * (1.0 - zr) / 2.0);
            let prof = -0.007 * bump(pp.y, yl, 0.10);
            DVec3::new(sign(pp.x) * prof * side * zw, 0.0, 0.0)
        });
        let f3 = f.clone();
        let vee: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
            let zr = f3.zi(z);
            let ax = f3.xi(pp.x.abs());
            let w = smoothstep(1.0, 1.2, zr) * smoothstep(2.40, 2.25, zr) * if n.y.abs() > 0.6 { 1.0 } else { 0.0 };
            DVec3::new(0.0, 0.006 * bump(ax, 0.30, 0.14) * w, 0.0)
        });
        Tube::new(keys, handles, vec![arch_lip, lower, vee])
    }

    fn zlines(&self) -> Vec<(&'static str, f64)> {
        ZLINES.iter().map(|&(n, z)| (n, if n == "AF2" { self.zf_ax } else if n == "AR2" { self.zr_ax } else { self.f.z(z) })).collect()
    }

    fn vlines(&self) -> Vec<f64> {
        VLINES.to_vec()
    }

    fn lens_tint(&self, name: &str) -> [f64; 4] {
        if name.starts_with("taillight") {
            [0.78, 0.02, 0.02, 1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
    }

    /// v-line indices: 0 floor, 1 sill, 2 door bottom, 3 widest, 4 waist
    /// crease, 5 shoulder, 6-7 shoulder strip, 8 belt, 9 door-frame line,
    /// 10 roof rail, 11 glass edge, 12 centre.
    fn assign(&self, zf: &str, _zt: &str, lo: usize, _hi: usize, side: Side) -> Option<(String, Mat)> {
        let s = side.tag();
        let i = zf;
        let front = matches!(i, "F" | "HL" | "AF1" | "AF2" | "AF3");
        let fdoor = matches!(i, "DF" | "WB" | "WT" | "FG");
        let rdoor = matches!(i, "BP" | "RG" | "RQ");
        let r = |n: String| Some((n, Mat::Paint));
        if lo == 0 {
            return if fdoor || rdoor { r("body_shell".into()) } else { None };
        }
        if lo <= 7 {
            if i == "F" {
                if lo == 4 {
                    return r(format!("headlight_{s}_lens"));
                }
                return r(if lo <= 3 { "bumper_front".into() } else { format!("fender_F{s}") });
            }
            if i == "HL" {
                return r(if lo <= 3 { "bumper_front".into() } else { format!("fender_F{s}") });
            }
            if i == "TL" && lo == 4 {
                return r(format!("taillight_{s}_lens"));
            }
            if i == "AR3" || i == "TL" {
                return r(if lo <= 2 { "bumper_rear".into() } else { "body_shell".into() });
            }
            if front {
                return r(format!("fender_F{s}"));
            }
            if fdoor {
                return r(if lo == 1 { "body_shell".into() } else { format!("door_F{s}") });
            }
            if rdoor {
                return r(if lo == 1 { "body_shell".into() } else { format!("door_R{s}") });
            }
            return r("body_shell".into());
        }
        if front {
            return r("hood".into());
        }
        if i == "AR3" || i == "TL" {
            return r("trunk_lid".into());
        }
        if lo == 8 {
            return r(match i {
                "WB" | "WT" => format!("glass_door_F{s}"),
                "DF" | "FG" => format!("door_F{s}"),
                "RG" => format!("glass_door_R{s}"),
                "BP" | "RQ" => format!("door_R{s}"),
                "AR1" => format!("glass_quarter_{s}"),
                _ => "body_shell".into(),
            });
        }
        if lo == 9 {
            return r(if fdoor {
                format!("door_F{s}")
            } else if rdoor {
                format!("door_R{s}")
            } else {
                "body_shell".into()
            });
        }
        if lo == 11 {
            if i == "WB" {
                return r("glass_windshield".into());
            }
            if matches!(i, "RQ" | "AR1" | "AR2") {
                return r("glass_rear".into());
            }
        }
        r("body_shell".into())
    }

    fn build_layout(&self, cb: &mut CarBody) {
        // back light, quarter glass and C-pillar
        let ov: [(&str, &[(usize, f64)]); 4] = [
            ("RQ", &[(11, -1.00), (12, -1.00)]),
            ("AR1", &[(11, -1.20), (12, -1.20)]),
            ("AR2", &[(8, -1.32), (9, -1.22), (10, -1.18), (11, -1.45), (12, -1.45)]),
            ("AR3", &[(8, -1.80), (9, -1.75), (10, -1.70), (11, -1.70), (12, -1.70)]),
        ];
        for (name, zs) in ov {
            let i = cb.zi(name);
            for &(jl, z) in zs {
                cb.set_z(i, jl, self.f.z(z));
            }
        }
        let (fi, ti) = (cb.zi("F"), cb.zi("T"));
        let s = self.clone();
        cb.end_fit(fi, &|x, y| s.front_z(x, y));
        cb.end_fit(ti, &|x, y| s.rear_z(x, y));
        let [af, ar] = self.arch_centres();
        let (af1, af2, af3) = (cb.zi("AF1"), cb.zi("AF2"), cb.zi("AF3"));
        cb.arch(af, self.arch_r, (af1, af2, af3), &[1, 2, 3, 4], &[10.0, 22.0, 40.0, 62.0, 90.0, 118.0, 145.0, 172.0, 196.5], (0.6, 5.2));
        let (ar1, ar2, ar3) = (cb.zi("AR1"), cb.zi("AR2"), cb.zi("AR3"));
        cb.arch(ar, self.arch_r, (ar1, ar2, ar3), &[1, 2, 3, 4], &[-16.5, 8.0, 35.0, 62.0, 90.0, 118.0, 145.0, 162.0, 177.0], (0.6, 5.2));
    }

    fn build_caps(&self, cb: &mut CarBody) {
        self.build_nose(cb);
        self.build_tail(cb);
    }

    fn parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        self.make_parts(cb)
    }
    fn rig_spec(&self) -> Option<RigSpec> {
        let (a, b) = (self.clone(), self.clone());
        let [(zf, yf), (zr, yr)] = self.arch_centres();
        Some(RigSpec {
            wheelbase: self.p.wheelbase,
            track: self.p.track,
            axle_f: self.zf_ax,
            axle_r: self.zr_ax,
            tyre_d: self.p.tyre_d,
            tyre_w: self.p.tyre_w,
            rim_d: self.p.rim_in * 0.0254,
            wheel_y: self.wheel_y,
            z_front: self.f.z(R_FRONT),
            z_rear: self.f.z(R_REAR),
            doors: ["FL", "FR", "RL", "RR"].iter().map(|s| s.to_string()).collect(),
            rear_closure: "trunk_lid".into(),
            arches: vec![(zf, yf, self.arch_r), (zr, yr, self.arch_r)],
            front_z: Rc::new(move |x, y| a.front_z(x, y)),
            rear_z: Rc::new(move |x, y| b.rear_z(x, y)),
            physics: self.p.physics.clone(),
            surfaces: Vec::new(),
        })
    }
}

impl Sedan {
    fn build_nose(&self, cb: &mut CarBody) {
        let i0 = cb.zi("F");
        let j = cb.jn;
        let nf = |k: usize| CapPt::N(cb.nid(i0, k % j));
        let left: Vec<CapPt> = (1..6).map(nf).collect();
        let right: Vec<CapPt> = (1..6).map(|k| nf(j - k)).collect();
        let top: Vec<CapPt> = (5..j - 4).map(nf).collect();
        let c = top.len() - 1;
        let y0 = cb.l.node_pos(cb.nid(i0, 1)).y;
        let xs: Vec<f64> = (5..j - 4).map(|k| cb.l.node_pos(cb.nid(i0, k % j)).x).collect();
        let mut bottom = vec![nf(1)];
        bottom.extend(xs[1..xs.len() - 1].iter().map(|&x| CapPt::P(x, y0)));
        bottom.push(nf(j - 1));
        let fpart = move |r: usize, cc: usize| -> Option<String> {
            if r == 3 && cc < 3 {
                return Some("headlight_L_lens".into());
            }
            if r == 3 && cc >= c - 3 && cc < c {
                return Some("headlight_R_lens".into());
            }
            let grille = (r == 3 && cc >= 3 && cc < c - 3) || (r == 1 && cc >= 4 && cc < c - 4);
            if grille {
                return None;
            }
            Some("bumper_front".into())
        };
        cb.cap("capF", self.front_fn(), 1.0, &left, &right, &bottom, &top, &fpart, &FxHashSet::default(), &FxHashMap::default(), vec![]);
        cb.extra.insert("grille_cols".into(), vec![xs[3], xs[c - 3], xs[4], xs[c - 4]]);
    }

    fn build_tail(&self, cb: &mut CarBody) {
        // tail: bumper cover low, tail panel, wrap-round lamps under the lid
        let it = cb.zi("T");
        let j = cb.jn;
        let nt = |k: usize| cb.nid(it, k % j);
        let left: Vec<CapPt> = (1..6).map(|k| CapPt::N(nt(k))).collect();
        let right: Vec<CapPt> = (1..6).map(|k| CapPt::N(nt(j - k))).collect();
        let y0 = cb.l.node_pos(nt(1)).y;
        let xk1 = cb.l.node_pos(nt(1)).x;
        const DROP: f64 = 0.0125;
        let xsr: Vec<f64> = [0.55, 0.28, 0.0, -0.28, -0.55].iter().map(|&x| self.f.x(x)).collect();
        let zl = self.f.z(-2.47);
        let tube = cb.tube();
        let lid: Vec<f64> = xsr
            .iter()
            .map(|&x| {
                let v = tube.v_at_x(zl, x.abs(), 6.0, 9.0);
                tube.eval(zl, v).y - DROP
            })
            .collect();
        let pk = cb.l.node_pos(nt(8));
        let mut top = vec![CapPt::N(nt(5)), CapPt::N(nt(6)), CapPt::N(nt(7)), CapPt::P(pk.x, pk.y - DROP)];
        top.extend(xsr.iter().zip(&lid).map(|(&x, &y)| CapPt::P(x, y)));
        top.extend([CapPt::P(-pk.x, pk.y - DROP), CapPt::N(nt(j - 7)), CapPt::N(nt(j - 6)), CapPt::N(nt(j - 5))]);
        let x55 = self.f.x(0.55);
        let fan: Vec<f64> = [1.0, 2.0, 3.0].iter().map(|k| xk1 - (xk1 - x55) * k / 4.0).collect();
        let mut bottom = vec![CapPt::N(nt(1))];
        bottom.extend(fan.iter().map(|&x| CapPt::P(x, y0)));
        bottom.extend(xsr.iter().map(|&x| CapPt::P(x, y0)));
        bottom.extend(fan.iter().rev().map(|&x| CapPt::P(-x, y0)));
        bottom.push(CapPt::N(nt(j - 1)));
        // the lid's rear edge, densely, sorted by x
        let lidp: Vec<DVec2> = (8..j - 7).map(|k| cb.l.nodes[&nt(k)].1).collect();
        let mut dense: Vec<DVec2> = Vec::new();
        for w in lidp.windows(2) {
            for t in 0..16 {
                dense.push(w[0] + (w[1] - w[0]) * (t as f64 / 16.0));
            }
        }
        dense.push(*lidp.last().unwrap());
        let mut lp: Vec<DVec3> = dense.iter().map(|q| tube.eval(q.x, q.y)).collect();
        lp.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        let lx: Rc<Vec<f64>> = Rc::new(lp.iter().map(|q| q.x).collect());
        let ly: Rc<Vec<f64>> = Rc::new(lp.iter().map(|q| q.y).collect());
        let lid_curve = |xa: f64, xb: f64| -> Curve {
            let (lx, ly) = (lx.clone(), ly.clone());
            Rc::new(move |t| {
                let x = xa + (xb - xa) * t;
                DVec2::new(x, interp(x, &lx, &ly) - DROP)
            })
        };
        let mut tx = vec![pk.x];
        tx.extend(xsr.iter().cloned());
        tx.push(-pk.x);
        let arcs: Vec<((usize, usize), (usize, usize), Curve)> = (0..tx.len() - 1).map(|k| ((4, 3 + k), (4, 4 + k), lid_curve(tx[k], tx[k + 1]))).collect();
        let c = top.len() - 1;
        let rpart = move |r: usize, cc: usize| -> Option<String> {
            if r == 3 && cc <= 3 {
                return Some("taillight_L_lens".into());
            }
            if r == 3 && cc >= c - 4 {
                return Some("taillight_R_lens".into());
            }
            if r <= 1 {
                return Some("bumper_rear".into());
            }
            Some("body_shell".into())
        };
        cb.cap("capR", self.rear_fn(), -1.0, &left, &right, &bottom, &top, &rpart, &FxHashSet::default(), &FxHashMap::default(), arcs);
    }

    fn make_parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let f = &self.f;
        let tube = cb.tube();
        let vm = tube.vm;
        let mut out = Parts::default();
        let mut extras = Parts::default();
        let mut mk: Vec<Marker> = Vec::new();
        // solid build: liners, door cards and the headliner are closed slabs
        // too, not sheets. Still sheets: the lamp reflectors (an inner panel
        // of the lens with its squeezed faces dropped, which can leave two
        // faces meeting at a point: a slab is not manifold there) and the
        // grille's back plates (closing them takes the lamps-glass-trim
        // group 552 triangles over the art spec's 60,000)
        let solid = cb.extra.contains_key("solid");
        let thick = |m: Mesh, th: f64| if solid { crate::solid::slab(&m, th) } else { m };
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        let sides = [("L", 1.0f64), ("R", -1.0f64)];

        // wheels
        let spec = self.wheel_spec();
        for (tag, z) in [("F", self.zf_ax), ("R", self.zr_ax)] {
            for (s, sg) in sides {
                let (rim, ty, di, c) = p::wheel(&spec, sg, z);
                out.insert(&format!("rim_{tag}{s}"), rim);
                out.insert(&format!("tyre_{tag}{s}"), ty);
                out.insert(&format!("brake_disc_{tag}{s}"), di);
                marker(format!("wheel_center_{tag}{s}"), c, DVec3::new(sg, 0.0, 0.0));
            }
        }

        // lamp housings: a reflector shell following each lens 32 mm behind
        // it, chrome projector cups in the headlamps
        for (s, sg) in sides {
            for kind in ["headlight", "taillight"] {
                let sk = cb.skins[&format!("{kind}_{s}_lens")].clone();
                let mut isb = vec![false; sk.v.len()];
                for lp in boundary_loops(&sk) {
                    for i in lp {
                        isb[i as usize] = true;
                    }
                }
                let mut h = inner_panel(&sk, &isb, 0.032, -1_000_000, 1, Mat::LampEmissive);
                h.flip();
                if kind == "headlight" {
                    let c0 = sk.v.iter().copied().sum::<DVec3>() / sk.v.len() as f64;
                    let vn = sk.vertex_normals();
                    for k in [-1.0, 1.0] {
                        let q = c0 + DVec3::new(k * sg * f.x(0.075), 0.0, 0.0);
                        let mut best = 0;
                        for (i, v) in sk.v.iter().enumerate() {
                            if (*v - q).length() < (sk.v[best] - q).length() {
                                best = i;
                            }
                        }
                        let n = vn[best];
                        let pc = sk.v[best] - n * 0.034;
                        let mut cup = closed_lathe(&[(0.0, 0.002), (0.020, 0.002), (0.024, -0.008), (0.008, -0.016), (0.0, -0.016)], 24, Mat::Chrome);
                        cup.transform(Some(p::rot_to(n)), pc);
                        h.add(&cup);
                    }
                }
                out.insert(&format!("{kind}_{s}_housing"), h);
            }
        }

        // grille: honeycomb following the nose behind the slim upper opening
        // and the lower intake, each with a dark back plate
        let mut g = Mesh::new();
        let gc = &cb.extra["grille_cols"];
        let (gu, gl) = (gc[0], gc[2]);
        for (gx, ya, yb) in [(gu.abs() - 0.025, f.y(0.680), f.y(0.765)), (gl.abs() - 0.025, f.y(0.375), f.y(0.485))] {
            let xs = linspace(-gx, gx, 40);
            let ys = linspace(ya, yb, 6);
            let pl = Grid2::from_fn(xs.len(), ys.len(), |i, jj| DVec3::new(xs[i], ys[jj], self.front_z(xs[i], ys[jj]) - 0.045));
            g.add(&grid(&pl, None, Mat::TrimBlack, false, false, -1));
            for (i, y) in arange(ya + 0.012, yb - 0.006, 0.024).into_iter().enumerate() {
                let x0 = -gx + 0.02 + if i % 2 == 1 { 0.016 } else { 0.0 };
                for x in arange(x0, gx - 0.015, 0.032) {
                    let zc = self.front_z(x, y) - 0.026;
                    g.add(&box_(DVec3::new(x, y, zc), DVec3::new(0.011, 0.007, 0.008), Mat::TrimBlack, None));
                }
            }
        }
        out.insert("grille", g);

        // mirrors on the door sail, handles, wipers
        for (s, sg) in sides {
            let v = if sg > 0.0 { 6.4 } else { vm - 6.4 };
            let (base, bn) = tube.eval_n(f.z(0.80), v);
            let mut m = Mesh::new();
            m.add(&rbox(base + bn * 0.026, DVec3::new(0.010, 0.018, 0.035), 0.008, 2, Mat::Paint));
            let hc = base + bn * 0.016 + DVec3::new(sg * 0.095, 0.060, -0.020);
            m.add(&rbox(hc, DVec3::new(0.085, 0.055, 0.045), 0.035, 4, Mat::Paint));
            m.add(&rbox(hc + DVec3::new(0.0, 0.0, -0.046), DVec3::new(0.072, 0.044, 0.0015), 0.0014, 1, Mat::Chrome));
            m.add(&rbox(base + bn * 0.016 + DVec3::new(sg * 0.045, 0.040, -0.010), DVec3::new(0.040, 0.012, 0.020), 0.010, 2, Mat::TrimBlack));
            out.insert(&format!("mirror_{s}"), m);
            marker(format!("mount_mirror_{s}"), base, DVec3::new(sg, 0.0, 0.0));
            out.insert(&format!("handle_F{s}"), p::door_handle(tube, f.z(0.05), f.y(0.865), sg, 0.16));
            out.insert(&format!("handle_R{s}"), p::door_handle(tube, f.z(-0.80), f.y(0.880), sg, 0.16));
        }
        for (s, x0, x1) in [("L", 0.62, 0.08), ("R", 0.02, -0.52)] {
            let zs = f.z(0.72);
            let pts: Vec<DVec3> = linspace(x0, x1, 8)
                .into_iter()
                .map(|x| {
                    let mut v = tube.v_at_x(zs, f.x(x.abs()), 8.0, 9.0);
                    if x < 0.0 {
                        v = vm - v;
                    }
                    let (pp, nn) = tube.eval_n(zs, v);
                    pp + nn * 0.004
                })
                .collect();
            out.insert(&format!("wiper_{s}"), p::wiper(&pts, Mat::TrimBlack));
        }

        // body trim: black sill strips and the windscreen reveal
        let mut bx = Mesh::new();
        for side in [1.0, -1.0] {
            let sv: Vec<f64> = linspace(0.80, -0.95, 34).into_iter().map(|z| f.z(z)).collect();
            let ys = vec![f.y(0.240); sv.len()];
            let (pp, _) = p::path_on_surface(tube, &sv, &ys, side, 0.004, (1.5, 5.5));
            bx.add(&orient_outward(sweep(&pp, &rrect(0.004, 0.022, 0.0015, 1), Mat::TrimBlack, &SweepOpts::default())));
        }
        let ws: Vec<DVec2> = [(0.78, 8.0), (0.34, 8.0), (-0.10, 8.0), (-0.10, 9.0), (-0.10, 10.0), (0.34, 10.0), (0.78, 10.0), (0.78, 9.0)]
            .iter()
            .map(|&(z, v)| DVec2::new(f.z(z), v))
            .collect();
        bx.add(&p::molding(tube, &ws, 0.0015, (0.008, 0.004), true, 6));
        extras.insert("body_shell", bx);

        // interior
        let hf = f.y(0.215);
        let dhf = hf - 0.215;
        for (s, sg) in sides {
            out.insert(&format!("seat_F{s}"), p::bucket_seat(sg * f.x(0.38), f.z(-0.40), hf + 0.02, Mat::InteriorFabric, true));
            marker(format!("seat_H_point_F{s}"), f.p(sg * 0.38, 0.215 + 0.20, -0.53) + DVec3::new(0.0, dhf - (f.y(0.415) - 0.415), 0.0), DVec3::X);
        }
        let mut rs = Mesh::new();
        rs.add(&rbox(DVec3::new(0.0, hf + 0.22, f.z(-1.08)), DVec3::new(f.x(0.52), 0.07, 0.24), 0.05, 3, Mat::InteriorFabric));
        let mut back = rbox(DVec3::ZERO, DVec3::new(f.x(0.52), 0.29, 0.06), 0.05, 3, Mat::InteriorFabric);
        back.transform(Some(rot_x((-22.0f64).to_radians())), DVec3::new(0.0, hf + 0.56, f.z(-1.38)));
        rs.add(&back);
        out.insert("seat_rear", rs);
        let mut dash = Mesh::new();
        dash.add(&rbox(DVec3::new(0.0, f.y(0.78), f.z(0.48)), DVec3::new(f.x(0.76), 0.12, 0.16), 0.06, 3, Mat::InteriorPlastic));
        dash.add(&rbox(DVec3::new(f.x(0.38), f.y(0.92), f.z(0.42)), DVec3::new(0.18, 0.03, 0.08), 0.025, 2, Mat::InteriorPlastic));
        dash.add(&rbox(DVec3::new(0.0, f.y(0.86), f.z(0.33)), DVec3::new(0.13, 0.08, 0.02), 0.01, 2, Mat::TrimBlack));
        out.insert("dash", dash);
        out.insert("steering_wheel", p::steering_wheel(DVec3::new(f.x(0.38), f.y(0.84), f.z(0.17)), 0.185, 24.0, Mat::InteriorPlastic, 0.13));
        out.insert("console", rbox(DVec3::new(0.0, f.y(0.42), f.z(-0.15)), DVec3::new(0.11, 0.12, 0.45), 0.03, 2, Mat::InteriorPlastic));
        for s in ["FL", "FR", "RL", "RR"] {
            let sk = cb.skins[&format!("door_{s}")].clone();
            let mut isb = vec![false; sk.v.len()];
            for lp in boundary_loops(&sk) {
                for i in lp {
                    isb[i as usize] = true;
                }
            }
            let trim = inner_panel(&sk, &isb, 0.085, -1_000_000, 2, Mat::InteriorPlastic);
            // solid: a moulded card 18 mm thick, its back 17 mm off the door's inner panel
            out.insert(&format!("door_trim_{s}"), thick(trim, 0.018));
        }
        let zz: Vec<f64> = linspace(-0.05, -0.95, 46).into_iter().map(|z| f.z(z)).collect();
        let vv = linspace(7.35, vm - 7.35, 60);
        let hp = Grid2::from_fn(zz.len(), vv.len(), |i, k| {
            let (pp, nn) = tube.eval_n(zz[i], vv[k]);
            pp - nn * 0.040
        });
        let mut hl = grid(&hp, None, Mat::InteriorFabric, false, false, -1);
        let my: f64 = hl.vertex_normals().iter().map(|n| n.y).sum::<f64>() / hl.v.len() as f64;
        if my > 0.0 {
            hl.flip();
        }
        out.insert("headliner", thick(hl, 0.008));

        // bay: transverse four with gearbox, radiator, firewall
        let mut eb = Mesh::new();
        eb.add(&rbox(f.p(0.05, 0.50, 1.55), DVec3::new(0.30, 0.17, 0.13), 0.03, 2, Mat::MetalBare));
        eb.add(&rbox(f.p(0.05, 0.71, 1.55), DVec3::new(0.28, 0.04, 0.10), 0.02, 2, Mat::TrimBlack));
        eb.add(&rbox(f.p(-0.36, 0.42, 1.52), DVec3::new(0.12, 0.13, 0.12), 0.04, 2, Mat::MetalBare));
        eb.add(&rbox(f.p(0.05, 0.62, 1.74), DVec3::new(0.30, 0.07, 0.04), 0.02, 2, Mat::TrimBlack));
        out.insert("engine_block", eb);
        let mut rad = Mesh::new();
        rad.add(&box_(f.p(0.0, 0.56, 2.12), DVec3::new(f.x(0.36), 0.16, 0.024), Mat::TrimBlack, None));
        rad.add(&box_(f.p(0.0, 0.735, 2.12), DVec3::new(f.x(0.37), 0.014, 0.030), Mat::MetalBare, None));
        out.insert("radiator", rad);
        out.insert("firewall", box_(f.p(0.0, 0.60, 0.94), DVec3::new(f.x(0.60), 0.28, 0.003), Mat::TrimBlack, None));
        // boxes along the car span their mapped ends (a shorter overhang
        // shortens the tray under it)
        let zspan = |zc: f64, hz: f64| ((f.z(zc - hz) + f.z(zc + hz)) / 2.0, (f.z(zc + hz) - f.z(zc - hz)) / 2.0);
        let mut ub = Mesh::new();
        let (zc, hz) = zspan(1.60, 0.45);
        ub.add(&box_(DVec3::new(0.0, f.y(0.150), zc), DVec3::new(f.x(0.46), 0.003, hz), Mat::TrimBlack, None));
        let (zc, hz) = zspan(-2.05, 0.28);
        ub.add(&box_(DVec3::new(0.0, f.y(0.330), zc), DVec3::new(f.x(0.50), 0.004, hz), Mat::TrimBlack, None));
        let (zc, hz) = zspan(-1.90, 0.22);
        ub.add(&rbox(DVec3::new(f.x(-0.09), f.y(0.30), zc), DVec3::new(0.21, 0.05, hz), 0.02, 2, Mat::MetalBare));
        out.insert("underbody", ub);
        let mut ex = Mesh::new();
        let pts: Vec<DVec3> = [(0.10, 0.36, 1.30), (0.12, 0.25, 1.12), (0.10, 0.168, 0.95), (0.10, 0.166, -0.70), (0.16, 0.175, -1.20), (0.30, 0.20, -1.60), (0.42, 0.25, -2.10), (0.48, 0.26, -2.42)]
            .iter()
            .map(|&(x, y, z)| f.p(x, y, z))
            .collect();
        ex.add(&p::tube_pipe(&pts, 0.022, Mat::MetalBare, 5, 14));
        ex.add(&rbox(f.p(0.12, 0.168, -0.20), DVec3::new(0.09, 0.016, 0.25), 0.012, 2, Mat::MetalBare));
        let (zc, hz) = zspan(-1.92, 0.13);
        ex.add(&rbox(DVec3::new(f.x(0.36), f.y(0.230), zc), DVec3::new(0.20, 0.06, hz), 0.04, 2, Mat::MetalBare));
        out.insert("exhaust", ex);
        let dr = self.arch_r - R_ARCH_R;
        for (tag, zc) in [("F", self.zf_ax), ("R", self.zr_ax)] {
            for (s, sg) in sides {
                out.insert(
                    &format!("wheel_well_{tag}{s}"),
                    thick(p::wheel_well(DVec3::new(0.0, f.y(R_TYRE_D / 2.0), zc), 0.375 + dr, f.x(0.56), f.x(0.86), sg, 92.0, 92.0), 0.004),
                );
            }
        }

        // markers
        for (s, sg) in sides {
            for (nm, y) in [("upper", 0.82), ("lower", 0.46)] {
                let (pp, nn) = p::surface_point(tube, f.z(0.86), f.y(y), sg, (1.5, 5.5));
                marker(format!("hinge_door_F{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
                let (pp, nn) = p::surface_point(tube, f.z(-0.24), f.y(y), sg, (1.5, 5.5));
                marker(format!("hinge_door_R{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            }
            let (pp, nn) = p::surface_point(tube, f.z(-0.20), f.y(0.64), sg, (1.5, 5.5));
            marker(format!("latch_door_F{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            let (pp, nn) = p::surface_point(tube, f.z(-1.00), f.y(0.70), sg, (1.5, 5.5));
            marker(format!("latch_door_R{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            marker(format!("hinge_hood_{s}"), f.p(sg * 0.68, 0.905, 0.92), DVec3::X);
            marker(format!("hinge_trunk_{s}"), f.p(sg * 0.55, 1.02, -1.75), DVec3::X);
            for (k, (z, y)) in [(2.15, 0.83), (1.80, 0.86), (1.30, 0.89), (0.95, 0.91)].iter().enumerate() {
                let zn = f.z(*z);
                let x = tube.key(6, zn).x + f.x(0.03);
                marker(format!("bolt_fender_F{s}_{}", k + 1), DVec3::new(sg * x, f.y(y - 0.03), zn), DVec3::X);
            }
        }
        for (k, x) in [-0.70, -0.35, 0.35, 0.70].iter().enumerate() {
            let xn = f.x(*x);
            let zz = self.front_z(xn.abs(), f.y(0.42));
            marker(format!("bolt_bumper_front_{}", k + 1), DVec3::new(xn, f.y(0.42), zz - 0.03), DVec3::X);
            let zz = self.rear_z(xn.abs(), f.y(0.45));
            marker(format!("bolt_bumper_rear_{}", k + 1), DVec3::new(xn, f.y(0.45), zz + 0.03), DVec3::X);
        }
        marker("latch_hood".into(), f.p(0.0, 0.80, 2.30), DVec3::X);
        marker("latch_trunk".into(), f.p(0.0, 0.98, -2.45), DVec3::X);
        (out, extras, mk)
    }
}

/// numpy.arange(a, b, step) for floats.
fn arange(a: f64, b: f64, step: f64) -> Vec<f64> {
    let n = ((b - a) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| a + i as f64 * step).collect()
}
