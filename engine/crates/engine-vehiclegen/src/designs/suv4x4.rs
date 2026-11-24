//! A boxy off-road 4x4: a two-box body with an upright windscreen, long
//! cargo windows, a tailgate carrying the spare wheel, flared wheel
//! openings, steel wheels on 32-inch tyres, a roof rack and rock sliders.
//! The kind of tall, flat-panelled vehicle that crumples dramatically.
//! Port of `tools/vehicle-gen/designs/suv_4x4.py`.
//!
//! Targets: length ~4.6 m (body), width 1.86 m, height ~1.86 m (1.92 over
//! the rack), wheelbase 2.70 m, track 1.56 m, tyre 265/75 R16 (802 mm).
//!
//! A standalone design (its own tube, layout, caps and parts) rigged by the
//! same pipeline as the sedan. Its body-in-white and running gear are the
//! sedan's lattice placed by a [`Frame`] (the car's wheelbase, overhangs, the
//! body raised on its big wheels, a taller cabin), so every station of the
//! lattice follows the design's own key lines through [`structure::Lines`]
//! and [`mechanical::Gear`].

use std::rc::Rc;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::assemble::{Design, RigSpec};
use crate::carbody::{CapPt, CarBody, Side};
use crate::curves::{bump, linspace, sign, smoothstep, Pchip};
use crate::export::Marker;
use crate::frame::{Frame, Target};
use crate::fx::{FxHashMap, FxHashSet};
use crate::layout::FaceFn;
use crate::mechanical::{self, MechPart};
use crate::mesh::{box_, closed_lathe, grid, rbox, rot_x, rot_y, Grid2, Mat, Mesh};
use crate::panels::{boundary_loops, inner_panel};
use crate::parts::{self as p, RimStyle, WheelSpec};
use crate::physics::Physics;
use crate::structure::{self, Structure};
use crate::tube::{Handle, Key, Tube};
use crate::uvpack::Parts;

use super::sedan::{self, R_ARCH_R, R_HEIGHT};

// -- the car ----------------------------------------------------------------------------------
const WHEELBASE: f64 = 2.700;
const TRACK: f64 = 1.560;
const ZF_AX: f64 = WHEELBASE / 2.0;
const ZR_AX: f64 = -WHEELBASE / 2.0;
const TYRE_D: f64 = 0.802;
const TYRE_W: f64 = 0.265;
const RIM_D: f64 = 16.0 * 0.0254;
const RIM_W: f64 = 7.0 * 0.0254;
const WHEEL_Y: f64 = TYRE_D / 2.0;
const ZFRONT: f64 = 2.150;
const ZREAR: f64 = -2.300;
/// The roof's height over the ground (m).
const ROOF: f64 = 1.86;
/// The skin's floor line at the centre (key 0).
const FLOOR_Y: f64 = 0.400;
/// The floor pan stands this far above the skin's floor line at the centre:
/// the skin rises 13 mm from there to the sills' inner edge, which the pan,
/// the sills and the firewall's foot stand on (the sedan's floor is flat to
/// the sill and takes 27 mm).
const PAN_ABOVE_FLOOR: f64 = 0.040;
const ARCH_RADIUS: f64 = 0.460;
const ARCH_F: (f64, f64, f64) = (ZF_AX, WHEEL_Y, ARCH_RADIUS);
const ARCH_R: (f64, f64, f64) = (ZR_AX, WHEEL_Y, ARCH_RADIUS);
/// The wheel-well liner's inner wall (x) and radius.
const WELL_X: f64 = 0.53;
const WELL_R: f64 = 0.44;

/// The 4x4's parameters (metres unless noted). Defaults are the olive car
/// with its roof rack and spare wheel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Suv4x4Params {
    /// rim style: "steel", "alloy" or "magnum"
    pub rim_style: String,
    /// the roof rack's rails and cross bars
    pub roof_rack: bool,
    /// the spare wheel on the tailgate
    pub spare_wheel: bool,
    /// linear RGB
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// The crash physics block: `engine/docs/generated-vehicles.md`.
    pub physics: Physics,
}

impl Default for Suv4x4Params {
    fn default() -> Self {
        Suv4x4Params { rim_style: "steel".into(), roof_rack: true, spare_wheel: true, paint: [0.10, 0.16, 0.06], metallic: 0.15, roughness: 0.38, physics: Physics::default() }
    }
}

/// Each numeric parameter's range (for the web configurator; paint channels are 0..1).
pub const RANGES: &[(&str, f64, f64)] = &[("paint", 0.0, 1.0), ("metallic", 0.0, 1.0), ("roughness", 0.0, 1.0)];

/// The rim styles this design accepts.
pub const RIM_STYLES: &[&str] = &["magnum", "alloy", "steel"];

impl Suv4x4Params {
    pub fn validate(&self) -> Result<(), String> {
        let chk = |name: &str, v: f64| if v.is_finite() && (0.0..=1.0).contains(&v) { Ok(()) } else { Err(format!("{name} = {v} is outside 0..1")) };
        chk("metallic", self.metallic)?;
        chk("roughness", self.roughness)?;
        for (k, c) in self.paint.iter().enumerate() {
            chk(&format!("paint[{k}]"), *c)?;
        }
        if !matches!(self.rim_style.as_str(), "steel" | "alloy" | "magnum") {
            return Err(format!("rim_style '{}' is not steel, alloy or magnum", self.rim_style));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Suv4x4 {
    pub p: Suv4x4Params,
    pub f: Frame,
}

// -- end faces ----------------------------------------------------------------------------------

pub fn front_z(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let lean = 0.030 * ((0.80 - y) / 0.25).clamp(0.0, 1.0);
    let top = 0.040 * ((y - 1.02) / 0.10).clamp(0.0, 1.0).powf(1.5);
    let corner = 0.090 * (x / 0.90).clamp(0.0, 1.0).powi(5);
    ZFRONT - lean - top - corner
}

pub fn rear_z(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let tuck = 0.030 * ((0.66 - y) / 0.10).clamp(0.0, 1.0);
    let top = 0.040 * ((y - 1.70) / 0.15).clamp(0.0, 1.0).powf(1.5);
    let corner = 0.070 * (x / 0.90).clamp(0.0, 1.0).powi(5);
    ZREAR + tuck + top + corner
}

// -- the surface --------------------------------------------------------------------------------

fn tbl(v: &[(f64, f64)]) -> Pchip {
    Pchip::new(v)
}

/// The plan width's table offset by `d`, at 48 stations.
fn width_off(d: f64) -> Vec<(f64, f64)> {
    let w = tbl(&[(2.20, 0.860), (2.10, 0.905), (1.90, 0.925), (1.35, 0.930), (0.0, 0.930), (-1.35, 0.930), (-2.10, 0.925), (-2.25, 0.905), (-2.35, 0.870)]);
    linspace(-2.4, 2.3, 48).iter().map(|&z| (z, w.eval(z) - d)).collect()
}

fn suv_tube() -> Tube {
    let keys = vec![
        Key::consts(0.0, FLOOR_Y),
        Key::tables(&width_off(0.055), &[(2.30, 0.560), (1.85, 0.560), (0.95, 0.450), (-0.95, 0.450), (-1.85, 0.560), (-2.40, 0.560)]),
        Key::tables(&width_off(0.026), &[(2.30, 0.630), (1.85, 0.630), (0.95, 0.520), (-0.95, 0.520), (-1.85, 0.640), (-2.40, 0.640)]),
        Key::tables(&width_off(0.0), &[(2.30, 0.760), (1.0, 0.740), (-1.0, 0.740), (-2.40, 0.760)]),
        Key::tables(&width_off(0.002), &[(2.30, 0.970), (1.35, 0.985), (0.0, 0.985), (-1.35, 0.990), (-2.40, 0.990)]),
        Key::tables(&width_off(0.024), &[(2.30, 1.100), (1.35, 1.120), (0.80, 1.130), (0.0, 1.135), (-2.40, 1.140)]),
        Key::tables(
            &[(2.30, 0.740), (2.05, 0.770), (1.60, 0.795), (0.80, 0.810), (0.72, 0.840), (0.0, 0.850), (-2.40, 0.850)],
            &[(2.30, 1.115), (1.35, 1.135), (0.80, 1.150), (0.72, 1.170), (0.0, 1.170), (-2.40, 1.170)],
        ),
        Key::tables(
            &[(2.30, 0.620), (1.35, 0.680), (0.80, 0.730), (0.72, 0.800), (0.30, 0.790), (0.0, 0.790), (-2.40, 0.790)],
            &[(2.30, 1.120), (1.35, 1.140), (0.80, 1.155), (0.72, 1.175), (0.30, 1.770), (0.0, 1.790), (-2.40, 1.790)],
        ),
        Key::tables(
            &[(2.30, 0.500), (1.35, 0.580), (0.80, 0.650), (0.72, 0.730), (0.30, 0.740), (0.0, 0.740), (-2.40, 0.740)],
            &[(2.30, 1.124), (1.35, 1.144), (0.80, 1.158), (0.72, 1.180), (0.30, 1.825), (0.0, 1.845), (-2.40, 1.845)],
        ),
        Key::xconst(0.0, &[(2.30, 1.130), (1.35, 1.150), (0.80, 1.165), (0.72, 1.190), (0.30, 1.840), (0.0, 1.860), (-2.40, 1.860)]),
    ];
    let h = |a: f64, b: f64| (Handle::C(a), Handle::C(b));
    let handles = vec![h(1.0 / 3.0, 1.0 / 3.0), h(0.30, 0.30), h(0.33, 0.33), h(0.36, 0.36), h(0.15, 0.15), h(0.14, 0.14), h(0.20, 0.22), h(0.30, 0.30), h(0.20, 0.20), h(1.0 / 3.0, 1.0 / 3.0)];
    // box flares round the wheel openings
    let arches = [ARCH_F, ARCH_R];
    let flares: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let side = ((n.x.abs() - 0.5) / 0.3).clamp(0.0, 1.0);
        let mut out = DVec3::ZERO;
        for (zc, yc, r) in arches {
            let d = ((z - zc).powi(2) + (pp.y - yc).powi(2)).sqrt();
            let fl = 0.022 * smoothstep(r + 0.13, r + 0.03, d) * if pp.y > yc - 0.15 { 1.0 } else { 0.0 };
            out.x += sign(pp.x) * fl * side;
        }
        out
    });
    let ribs: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let w = smoothstep(0.85, 1.0, z) * smoothstep(2.20, 2.05, z) * if n.y.abs() > 0.6 { 1.0 } else { 0.0 };
        DVec3::new(0.0, 0.008 * bump(pp.x.abs(), 0.36, 0.08) * w, 0.0)
    });
    Tube::new(keys, handles, vec![flares, ribs])
}

// -- layout --------------------------------------------------------------------------------------

const ZLINES: [(&str, f64); 17] = [
    ("F", 2.14),
    ("HL", 1.98),
    ("AF1", 1.81),
    ("AF2", ZF_AX),
    ("AF3", 0.89),
    ("DF", 0.80),
    ("WB", 0.72),
    ("WT", 0.30),
    ("FG", 0.0),
    ("BP", -0.05),
    ("RG", -0.10),
    ("RQ", -0.80),
    ("AR1", -0.89),
    ("AR2", ZR_AX),
    ("AR3", -1.81),
    ("QG", -2.06),
    ("T", -2.28),
];
const VLINES: [f64; 13] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.33, 5.67, 6.0, 6.85, 7.0, 8.0, 9.0];
/// The tailgate's glass: its top and base on the centre line (z), for the
/// body-in-white's rear header and shelf.
const REAR_LINES: (f64, f64) = (-2.18, -2.22);

impl Suv4x4 {
    pub fn new(p: Suv4x4Params) -> Suv4x4 {
        // The sedan's lattice, placed by the car's own proportions: the
        // body's floor line sets the lift (the pan stands on it), the roof
        // its scale above the shoulder, the wheel opening the arch zone.
        let dy = FLOOR_Y - 0.195;
        let y0 = 0.95;
        let blend = |y: f64| if y <= y0 { 0.0 } else if y < y0 + 0.1 { (y - y0).powi(2) / 0.2 } else { y - y0 - 0.05 };
        let kr = 1.0 + (ROOF - R_HEIGHT - dy) / blend(R_HEIGHT);
        let f = Frame::new(
            sedan::refdims(),
            Target { wheelbase: WHEELBASE, front_overhang: ZFRONT - ZF_AX, rear_overhang: -ZREAR - ZF_AX, sx: 1.0, dy, kr, y0, arch_k: ARCH_RADIUS / R_ARCH_R },
        );
        Suv4x4 { p, f }
    }

    fn wheel_spec(&self) -> WheelSpec {
        let rim_style = match self.p.rim_style.as_str() {
            "alloy" => RimStyle::Alloy,
            "magnum" => RimStyle::Magnum,
            _ => RimStyle::Steel,
        };
        WheelSpec { track: TRACK, wheel_y: WHEEL_Y, tyre_d: TYRE_D, tyre_w: TYRE_W, rim_d: RIM_D, rim_w: RIM_W, rim_style, disc_d: 0.30 }
    }

    /// The lines the body-in-white is placed from ([`structure::unibody`]):
    /// the design's own door, screen and pillar lines (mapped back to the
    /// lattice's reference z), its wheel opening, its floor.
    pub fn structure_lines(&self) -> structure::Lines {
        let f = &self.f;
        let z = |n: &str| f.zi(ZLINES.iter().find(|l| l.0 == n).unwrap().1);
        structure::Lines {
            f: f.clone(),
            axle_f: f.zi(ZF_AX),
            axle_r: f.zi(ZR_AX),
            z_door_front: z("DF"),
            z_screen_base: z("WB"),
            z_screen_top: z("WT"),
            z_b: z("BP"),
            z_seat: z("BP"),
            z_door_rear: z("AR1"),
            z_rear_top: f.zi(REAR_LINES.0),
            z_rear_base: f.zi(REAR_LINES.1),
            well_x: WELL_X,
            well_r: WELL_R,
            wheel_y: WHEEL_Y,
            y_pan: f.yi(FLOOR_Y + PAN_ABOVE_FLOOR),
        }
    }

    /// The running gear's dimensions for [`mechanical::sedan_parts`].
    pub fn gear(&self) -> mechanical::Gear {
        let f = &self.f;
        mechanical::Gear {
            f: f.clone(),
            half_track: TRACK / 2.0,
            wheel_y: WHEEL_Y,
            zf: ZF_AX,
            zr: ZR_AX,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: RIM_D,
            rim_w: RIM_W,
            axle_f: f.zi(ZF_AX),
            axle_r: f.zi(ZR_AX),
            y_pan: f.yi(FLOOR_Y + PAN_ABOVE_FLOOR),
            z_nose: Rc::new(front_z),
            z_tail: Rc::new(rear_z),
            rear: None,
        }
    }

    pub fn unibody(&self) -> Structure {
        let tube = suv_tube();
        structure::unibody(&self.structure_lines(), &structure::Skin { tube: &tube, front_z: &front_z, rear_z: &rear_z })
    }
}

impl Design for Suv4x4 {
    fn name(&self) -> String {
        "suv_4x4".into()
    }

    fn paint(&self) -> ([f64; 3], f64, f64) {
        (self.p.paint, self.p.metallic, self.p.roughness)
    }

    fn tube(&self) -> Tube {
        suv_tube()
    }

    fn zlines(&self) -> Vec<(&'static str, f64)> {
        ZLINES.to_vec()
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

    /// v-line indices: 0 floor, 1 sill, 2 door bottom, 3 widest, 4 waist,
    /// 5 shoulder, 6-7 shoulder strip, 8 belt, 9 roof rail, 10 glass edge,
    /// 11 centre.
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
                return r(if lo <= 2 { "bumper_front".into() } else { format!("fender_F{s}") });
            }
            if i == "HL" {
                return r(if lo <= 2 { "bumper_front".into() } else { format!("fender_F{s}") });
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
            if matches!(i, "AR3" | "QG") {
                return r(if lo <= 1 { "bumper_rear".into() } else { "body_shell".into() });
            }
            return r("body_shell".into());
        }
        if front {
            return r("hood".into());
        }
        if lo == 8 {
            return r(match i {
                "WB" | "WT" => format!("glass_door_F{s}"),
                "DF" | "FG" => format!("door_F{s}"),
                "RG" => format!("glass_door_R{s}"),
                "BP" | "RQ" => format!("door_R{s}"),
                "AR1" | "AR2" | "AR3" => format!("glass_quarter_{s}"),
                _ => "body_shell".into(),
            });
        }
        if lo == 9 {
            if fdoor {
                return r(format!("door_F{s}"));
            }
            if rdoor {
                return r(format!("door_R{s}"));
            }
            return r("body_shell".into());
        }
        if lo == 11 && i == "WB" {
            return r("glass_windshield".into());
        }
        r("body_shell".into())
    }

    fn build_layout(&self, cb: &mut CarBody) {
        let (fi, ti) = (cb.zi("F"), cb.zi("T"));
        cb.end_fit(fi, &front_z);
        cb.end_fit(ti, &rear_z);
        let (af1, af2, af3) = (cb.zi("AF1"), cb.zi("AF2"), cb.zi("AF3"));
        cb.arch((ARCH_F.0, ARCH_F.1), ARCH_F.2, (af1, af2, af3), &[1, 2, 3, 4], &[16.0, 26.0, 42.0, 62.0, 90.0, 118.0, 145.0, 168.0, 192.0], (0.6, 5.2));
        let (ar1, ar2, ar3) = (cb.zi("AR1"), cb.zi("AR2"), cb.zi("AR3"));
        cb.arch((ARCH_R.0, ARCH_R.1), ARCH_R.2, (ar1, ar2, ar3), &[1, 2, 3, 4], &[-12.0, 12.0, 35.0, 62.0, 90.0, 118.0, 145.0, 168.0, 192.0], (0.6, 5.2));
    }

    fn build_caps(&self, cb: &mut CarBody) {
        let j = cb.jn;
        // nose: square lamps either side of a tall grille, apron below
        {
            let i0 = cb.zi("F");
            let nf = |k: usize| cb.nid(i0, k % j);
            let left: Vec<CapPt> = (1..6).map(|k| CapPt::N(nf(k))).collect();
            let right: Vec<CapPt> = (1..6).map(|k| CapPt::N(nf(j - k))).collect();
            let top: Vec<CapPt> = (5..j - 4).map(|k| CapPt::N(nf(k))).collect();
            let c = top.len() - 1;
            let y0 = cb.l.node_pos(nf(1)).y;
            let xs: Vec<f64> = (5..j - 4).map(|k| cb.l.node_pos(nf(k)).x).collect();
            let mut bottom = vec![CapPt::N(nf(1))];
            bottom.extend(xs[1..xs.len() - 1].iter().map(|&x| CapPt::P(x, y0)));
            bottom.push(CapPt::N(nf(j - 1)));
            let fpart = move |r: usize, cc: usize| -> Option<String> {
                if r == 3 && cc < 5 {
                    return Some("headlight_L_lens".into());
                }
                if r == 3 && cc + 5 >= c && cc < c {
                    return Some("headlight_R_lens".into());
                }
                if (r == 2 || r == 3) && cc >= 5 && cc + 5 < c {
                    return None;
                }
                if r <= 1 {
                    return Some("bumper_front".into());
                }
                Some(if (cc as f64) < c as f64 / 2.0 { "fender_FL".into() } else { "fender_FR".into() })
            };
            let fz: FaceFn = Rc::new(front_z);
            cb.cap("capF", fz, 1.0, &left, &right, &bottom, &top, &fpart, &FxHashSet::default(), &FxHashMap::default(), vec![]);
            cb.extra.insert("grille_cols".into(), vec![xs[5], xs[c - 5]]);
        }
        // tail: D-pillars with tall lamps, the tailgate between them
        // carrying the back light, bumper below
        {
            let it = cb.zi("T");
            let nt = |k: usize| cb.nid(it, k % j);
            let left: Vec<CapPt> = (1..=10).map(|k| CapPt::N(nt(k))).collect();
            let right: Vec<CapPt> = (1..=10).map(|k| CapPt::N(nt(j - k))).collect();
            let top: Vec<CapPt> = [nt(10), nt(11), nt(12), nt(j - 11), nt(j - 10)].iter().map(|&n| CapPt::N(n)).collect();
            let y0 = cb.l.node_pos(nt(1)).y;
            let xg = 0.70;
            let bottom = vec![CapPt::N(nt(1)), CapPt::P(xg, y0), CapPt::P(0.0, y0), CapPt::P(-xg, y0), CapPt::N(nt(j - 1))];
            let r_max = left.len() - 1;
            let mut ov: FxHashMap<(usize, usize), DVec2> = FxHashMap::default();
            for r in 1..r_max {
                let yr = cb.l.node_pos(nt(r + 1)).y;
                ov.insert((r, 1), DVec2::new(xg, yr));
                ov.insert((r, 2), DVec2::new(0.0, yr));
                ov.insert((r, 3), DVec2::new(-xg, yr));
            }
            let rpart = move |r: usize, c: usize| -> Option<String> {
                if r == 0 {
                    return Some("bumper_rear".into());
                }
                if c == 0 || c == 3 {
                    if r == 2 || r == 3 {
                        return Some(format!("taillight_{}_lens", if c == 0 { "L" } else { "R" }));
                    }
                    return Some("body_shell".into());
                }
                if r == 7 {
                    return Some("glass_rear".into());
                }
                Some("tailgate".into())
            };
            let rz: FaceFn = Rc::new(rear_z);
            cb.cap("capR", rz, -1.0, &left, &right, &bottom, &top, &rpart, &FxHashSet::default(), &ov, Vec::new());
        }
    }

    fn parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        self.make_parts(cb)
    }

    fn rig_spec(&self) -> Option<RigSpec> {
        Some(RigSpec {
            wheelbase: WHEELBASE,
            track: TRACK,
            axle_f: ZF_AX,
            axle_r: ZR_AX,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: RIM_D,
            wheel_y: WHEEL_Y,
            rear_wheel: None,
            pose: None,
            z_front: ZFRONT,
            z_rear: ZREAR,
            doors: ["FL", "FR", "RL", "RR"].iter().map(|s| s.to_string()).collect(),
            rear_closure: "tailgate".into(),
            arches: vec![ARCH_F, ARCH_R],
            front_z: Rc::new(front_z),
            rear_z: Rc::new(rear_z),
            physics: self.p.physics.clone(),
            surfaces: Vec::new(),
            structure: None,
            mechanical: Rc::new(Vec::new()),
        })
    }

    fn structure(&self) -> Option<Structure> {
        Some(self.unibody())
    }

    fn mechanical(&self, st: &Structure) -> Vec<MechPart> {
        mechanical::sedan_parts(&self.gear(), st)
    }
}

mod parts;
