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

impl Suv4x4 {
    fn make_parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let tube = cb.tube();
        let vm = tube.vm;
        let mut out = Parts::default();
        let mut extras = Parts::default();
        let mut mk: Vec<Marker> = Vec::new();
        let solid = cb.extra.contains_key("solid");
        let thick = |m: Mesh, th: f64| if solid { crate::solid::slab(&m, th) } else { m };
        let mut marker = |name: String, pos: DVec3, xa: DVec3| mk.push(Marker { name, pos, xaxis: xa });
        let sides = [("L", 1.0f64), ("R", -1.0f64)];

        // wheels
        let spec = self.wheel_spec();
        for (tag, z) in [("F", ZF_AX), ("R", ZR_AX)] {
            for (s, sg) in sides {
                let (rim, ty, di, c) = p::wheel(&spec, sg, z);
                out.insert(&format!("rim_{tag}{s}"), rim);
                out.insert(&format!("tyre_{tag}{s}"), ty);
                out.insert(&format!("brake_disc_{tag}{s}"), di);
                marker(format!("wheel_center_{tag}{s}"), c, DVec3::new(sg, 0.0, 0.0));
            }
        }

        // lamp housings: reflector shells behind the lenses, chrome cups in front
        for (s, _sg) in sides {
            for kind in ["headlight", "taillight"] {
                let sk = cb.skins[&format!("{kind}_{s}_lens")].clone();
                let mut isb = vec![false; sk.v.len()];
                for lp in boundary_loops(&sk) {
                    for i in lp {
                        isb[i as usize] = true;
                    }
                }
                let mut h = if solid {
                    crate::solid::reflector(&sk, 0.034, 0.003, Mat::LampEmissive)
                } else {
                    let mut h = inner_panel(&sk, &isb, 0.034, -1_000_000, 1, Mat::LampEmissive);
                    h.flip();
                    h
                };
                if kind == "headlight" {
                    let c0 = sk.v.iter().copied().sum::<DVec3>() / sk.v.len() as f64;
                    let vn = sk.vertex_normals();
                    let mut best = 0;
                    for (i, v) in sk.v.iter().enumerate() {
                        if (*v - c0).length() < (sk.v[best] - c0).length() {
                            best = i;
                        }
                    }
                    let n = vn[best];
                    let mut cup = closed_lathe(&[(0.0, 0.002), (0.045, 0.002), (0.050, -0.010), (0.016, -0.022), (0.0, -0.022)], 32, Mat::Chrome);
                    cup.transform(Some(p::rot_to(n)), sk.v[best] - n * 0.036);
                    h.add(&cup);
                }
                out.insert(&format!("{kind}_{s}_housing"), h);
            }
        }

        // grille: vertical slats in the opening, a dark back plate
        let mut g = Mesh::new();
        let gx = cb.extra["grille_cols"][0].abs() - 0.03;
        let (ya, yb) = (0.785, 1.080);
        let xs = linspace(-gx, gx, 30);
        let ys = linspace(ya, yb, 8);
        let pl = Grid2::from_fn(xs.len(), ys.len(), |i, j| DVec3::new(xs[i], ys[j], front_z(xs[i], ys[j]) - 0.050));
        g.add(&grid(&pl, None, Mat::TrimBlack, false, false, -1));
        let mut x = -gx + 0.02;
        while x < gx - 0.01 {
            let zc = front_z(x, (ya + yb) / 2.0) - 0.030;
            g.add(&box_(DVec3::new(x, (ya + yb) / 2.0, zc), DVec3::new(0.010, (yb - ya) / 2.0 - 0.010, 0.012), Mat::TrimBlack, None));
            x += 0.045;
        }
        out.insert("grille", g);

        // steel bumpers: a square tube bar on brackets, front and rear
        let bar = |zf: f64, y: f64, sgn: f64, half_w: f64| {
            let mut b = Mesh::new();
            b.add(&rbox(DVec3::new(0.0, y, zf + sgn * 0.095), DVec3::new(half_w, 0.065, 0.055), 0.012, 2, Mat::TrimBlack));
            for x in [-0.55, 0.55] {
                b.add(&rbox(DVec3::new(x, y - 0.02, zf + sgn * 0.030), DVec3::new(0.04, 0.04, 0.022), 0.006, 1, Mat::TrimBlack));
            }
            b
        };
        extras.insert("bumper_front", bar(front_z(0.5, 0.60), 0.600, 1.0, 0.92));
        extras.insert("bumper_rear", bar(rear_z(0.5, 0.60), 0.600, -1.0, 0.90));

        // spare wheel on the tailgate
        if self.p.spare_wheel {
            let zt = rear_z(0.0, 1.05);
            let mut spare = Mesh::new();
            spare.add(&p::rim_steel(RIM_D, RIM_W, 64));
            spare.add(&p::tyre(TYRE_D, TYRE_W, RIM_D, RIM_W, 96, 3));
            spare.transform(Some(rot_y(std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 1.09, zt - TYRE_W / 2.0 - 0.045));
            spare.add(&rbox(DVec3::new(0.0, 1.09, zt - 0.024), DVec3::new(0.06, 0.06, 0.018), 0.01, 1, Mat::MetalBare));
            out.insert("spare_wheel", spare);
        }

        // mirrors, handles, wipers
        for (s, sg) in sides {
            let v = if sg > 0.0 { 5.8 } else { vm - 5.8 };
            let (base, bn) = tube.eval_n(0.62, v);
            let mut m = Mesh::new();
            m.add(&rbox(base + bn * 0.028, DVec3::new(0.012, 0.022, 0.035), 0.008, 2, Mat::TrimBlack));
            let hc = base + bn * 0.028 + DVec3::new(sg * 0.11, 0.07, -0.02);
            m.add(&rbox(hc, DVec3::new(0.10, 0.075, 0.035), 0.020, 3, Mat::TrimBlack));
            m.add(&rbox(hc + DVec3::new(0.0, 0.0, -0.036), DVec3::new(0.088, 0.064, 0.0015), 0.0014, 1, Mat::Chrome));
            m.add(&rbox(base + bn * 0.028 + DVec3::new(sg * 0.05, 0.045, -0.01), DVec3::new(0.045, 0.012, 0.016), 0.008, 2, Mat::TrimBlack));
            out.insert(&format!("mirror_{s}"), m);
            marker(format!("mount_mirror_{s}"), base, DVec3::new(sg, 0.0, 0.0));
            out.insert(&format!("handle_F{s}"), p::door_handle(tube, 0.04, 1.06, sg, 0.15));
            out.insert(&format!("handle_R{s}"), p::door_handle(tube, -0.72, 1.06, sg, 0.15));
        }
        for (s, x0, x1) in [("L", 0.62, 0.06), ("R", 0.02, -0.54)] {
            let zs = 0.66;
            let pts: Vec<DVec3> = linspace(x0, x1, 8)
                .into_iter()
                .map(|x: f64| {
                    let mut v = tube.v_at_x(zs, x.abs(), 8.0, 9.0);
                    if x < 0.0 {
                        v = vm - v;
                    }
                    let (pp, nn) = tube.eval_n(zs, v);
                    pp + nn * 0.005
                })
                .collect();
            out.insert(&format!("wiper_{s}"), p::wiper(&pts, Mat::TrimBlack));
        }

        // body extras: rock sliders, roof rack, windscreen reveal
        let mut bx = Mesh::new();
        for sg in [1.0f64, -1.0] {
            let zs_ = linspace(0.78, -0.82, 24);
            let (pp, _) = p::path_on_surface(tube, &zs_, &[0.50], sg, 0.060, (1.5, 5.5));
            bx.add(&p::tube_pipe(&pp, 0.030, Mat::TrimBlack, 1, 14));
            for z in [0.55, -0.10, -0.65] {
                let (q, nq) = p::surface_point(tube, z, 0.50, sg, (1.5, 5.5));
                bx.add(&p::tube_pipe(&[q + nq * 0.002, q + nq * 0.045], 0.016, Mat::TrimBlack, 1, 14));
            }
            if self.p.roof_rack {
                let v = if sg > 0.0 { 7.55 } else { vm - 7.55 };
                let rail: Vec<DVec3> = linspace(0.10, -2.12, 26).into_iter().map(|z| tube.eval(z, v) + DVec3::new(0.0, 0.055, 0.0)).collect();
                bx.add(&p::tube_pipe(&rail, 0.016, Mat::TrimBlack, 1, 14));
                for z in [0.0, -2.05] {
                    let q = tube.eval(z, v);
                    bx.add(&p::tube_pipe(&[q + DVec3::new(0.0, 0.004, 0.0), q + DVec3::new(0.0, 0.055, 0.0)], 0.014, Mat::TrimBlack, 1, 14));
                }
            }
        }
        if self.p.roof_rack {
            for z in [-0.35, -0.95, -1.55] {
                let a = tube.eval(z, 7.55) + DVec3::new(0.0, 0.055, 0.0);
                let b = tube.eval(z, vm - 7.55) + DVec3::new(0.0, 0.055, 0.0);
                bx.add(&p::tube_pipe(&[a, (a + b) / 2.0 + DVec3::new(0.0, 0.012, 0.0), b], 0.014, Mat::TrimBlack, 2, 14));
            }
        }
        bx.set_mat(Mat::TrimBlack);
        let ws: Vec<DVec2> = [(0.72, 8.0), (0.51, 8.0), (0.30, 8.0), (0.30, 9.0), (0.30, 10.0), (0.51, 10.0), (0.72, 10.0), (0.72, 9.0)].iter().map(|&(z, v)| DVec2::new(z, v)).collect();
        let mut rev = p::molding(tube, &ws, 0.0015, (0.010, 0.005), true, 6);
        rev.set_mat(Mat::TrimBlack);
        bx.add(&rev);
        // solid build: the B-pillar, behind the doors' shut line
        if solid {
            let z_b = ZLINES.iter().find(|l| l.0 == "BP").unwrap().1;
            for side in [1.0, -1.0] {
                bx.add(&super::sedan::b_pillar(tube, z_b, side));
            }
        }
        extras.insert("body_shell", bx);

        // interior
        let hf = 0.430;
        for (s, sg) in sides {
            // (solid build: the seats, the bench, the dash and the console
            // are parts on the structure: `mechanical`)
            if !solid {
                out.insert(&format!("seat_F{s}"), p::bucket_seat(sg * 0.38, -0.30, hf, Mat::InteriorFabric, true));
            }
            marker(format!("seat_H_point_F{s}"), DVec3::new(sg * 0.38, hf + 0.18, -0.43), DVec3::X);
        }
        if !solid {
            let mut rs = Mesh::new();
            rs.add(&rbox(DVec3::new(0.0, hf + 0.20, -1.05), DVec3::new(0.52, 0.07, 0.24), 0.05, 3, Mat::InteriorFabric));
            let mut back = rbox(DVec3::ZERO, DVec3::new(0.52, 0.30, 0.06), 0.05, 3, Mat::InteriorFabric);
            back.transform(Some(rot_x((-12.0f64).to_radians())), DVec3::new(0.0, hf + 0.56, -1.32));
            rs.add(&back);
            out.insert("seat_rear", rs);
            let mut dash = Mesh::new();
            dash.add(&rbox(DVec3::new(0.0, 1.00, 0.52), DVec3::new(0.76, 0.13, 0.14), 0.04, 3, Mat::InteriorPlastic));
            dash.add(&rbox(DVec3::new(0.38, 1.15, 0.46), DVec3::new(0.17, 0.03, 0.07), 0.02, 2, Mat::InteriorPlastic));
            out.insert("dash", dash);
            out.insert("console", rbox(DVec3::new(0.0, 0.62, -0.05), DVec3::new(0.11, 0.10, 0.42), 0.03, 2, Mat::InteriorPlastic));
        }
        out.insert("steering_wheel", p::steering_wheel(DVec3::new(0.38, 1.04, 0.22), 0.19, 30.0, Mat::InteriorPlastic, if solid { 0.19 } else { 0.12 }));
        for s in ["FL", "FR", "RL", "RR"] {
            let sk = cb.skins[&format!("door_{s}")].clone();
            let mut isb = vec![false; sk.v.len()];
            for lp in boundary_loops(&sk) {
                for i in lp {
                    isb[i as usize] = true;
                }
            }
            let trim = if solid { crate::solid::door_card(&sk, 0.085, 0.018, Mat::InteriorPlastic) } else { inner_panel(&sk, &isb, 0.085, -1_000_000, 2, Mat::InteriorPlastic) };
            out.insert(&format!("door_trim_{s}"), trim);
        }
        let zz = linspace(0.24, -2.15, 80);
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

        // engine bay and underbody
        if solid {
            // the body-in-white: every member a closed solid, and the
            // mechanical and interior parts on it
            let st = self.unibody();
            for (name, m) in st.parts() {
                out.insert(&name, m);
            }
            for part in self.mechanical(&st) {
                if !part.mesh.v.is_empty() {
                    out.insert(&part.name, part.mesh);
                }
            }
        } else {
            // inline six, radiator, firewall; ladder frame, solid axles,
            // transfer case and skid plates under it all
            let mut eb = Mesh::new();
            eb.add(&rbox(DVec3::new(0.0, 0.78, 1.38), DVec3::new(0.17, 0.20, 0.42), 0.03, 2, Mat::MetalBare));
            eb.add(&rbox(DVec3::new(0.0, 1.00, 1.38), DVec3::new(0.13, 0.03, 0.40), 0.015, 2, Mat::TrimBlack));
            eb.add(&rbox(DVec3::new(0.24, 0.85, 1.38), DVec3::new(0.06, 0.08, 0.32), 0.02, 2, Mat::MetalBare));
            eb.add(&rbox(DVec3::new(0.0, 0.70, 0.92), DVec3::new(0.17, 0.17, 0.11), 0.04, 2, Mat::MetalBare));
            out.insert("engine_block", eb);
            let mut rad = Mesh::new();
            rad.add(&box_(DVec3::new(0.0, 0.82, 1.92), DVec3::new(0.36, 0.18, 0.028), Mat::TrimBlack, None));
            rad.add(&box_(DVec3::new(0.0, 1.015, 1.92), DVec3::new(0.37, 0.015, 0.032), Mat::MetalBare, None));
            out.insert("radiator", rad);
            out.insert("firewall", box_(DVec3::new(0.0, 0.84, 0.775), DVec3::new(0.60, 0.28, 0.003), Mat::TrimBlack, None));
            let mut ub = Mesh::new();
            for sg in [1.0, -1.0] {
                ub.add(&box_(DVec3::new(sg * 0.42, 0.345, 0.0), DVec3::new(0.035, 0.045, 2.05), Mat::TrimBlack, None));
            }
            for z in [1.95, 0.70, -0.60, -2.10] {
                ub.add(&box_(DVec3::new(0.0, 0.345, z), DVec3::new(0.385, 0.035, 0.03), Mat::TrimBlack, None));
            }
            ub.add(&box_(DVec3::new(0.0, 0.300, 1.45), DVec3::new(0.32, 0.004, 0.40), Mat::MetalBare, None));
            for zc in [ZF_AX, ZR_AX] {
                let mut ax = closed_lathe(&[(0.0, 0.62), (0.050, 0.62), (0.050, -0.62), (0.0, -0.62)], 20, Mat::MetalBare);
                ax.transform(None, DVec3::new(0.0, WHEEL_Y, zc));
                ub.add(&ax);
                ub.add(&rbox(DVec3::new(0.08, WHEEL_Y, zc), DVec3::new(0.13, 0.12, 0.11), 0.06, 3, Mat::MetalBare));
            }
            ub.add(&rbox(DVec3::new(0.0, 0.315, 0.40), DVec3::new(0.12, 0.060, 0.16), 0.03, 2, Mat::MetalBare));
            ub.add(&p::tube_pipe(&[DVec3::new(0.03, 0.300, 0.20), DVec3::new(0.06, 0.290, -0.60), DVec3::new(0.08, 0.300, -1.24)], 0.030, Mat::MetalBare, 2, 14));
            ub.add(&p::tube_pipe(&[DVec3::new(0.03, 0.300, 0.56), DVec3::new(0.06, 0.300, 1.24)], 0.028, Mat::MetalBare, 1, 14));
            ub.add(&rbox(DVec3::new(-0.20, 0.30, -1.75), DVec3::new(0.30, 0.05, 0.25), 0.03, 2, Mat::MetalBare));
            out.insert("underbody", ub);
            let mut ex = Mesh::new();
            let pts: Vec<DVec3> = [(0.24, 0.66, 1.30), (0.28, 0.40, 1.08), (0.25, 0.282, 0.85), (0.25, 0.280, -0.70), (0.30, 0.282, -1.15), (0.34, 0.285, -1.55), (0.38, 0.290, -2.05), (0.50, 0.292, -2.25)]
                .iter()
                .map(|&(x, y, z)| DVec3::new(x, y, z))
                .collect();
            ex.add(&p::tube_pipe(&pts, 0.026, Mat::MetalBare, 5, 14));
            ex.add(&rbox(DVec3::new(0.25, 0.282, -0.10), DVec3::new(0.10, 0.026, 0.28), 0.02, 2, Mat::MetalBare));
            out.insert("exhaust", ex);
        }
        for (tag, zc) in [("F", ZF_AX), ("R", ZR_AX)] {
            for (s, sg) in sides {
                // (solid build: the inner wall's opening lets the arms through)
                out.insert(&format!("wheel_well_{tag}{s}"), thick(p::wheel_well_with(DVec3::new(0.0, WHEEL_Y, zc), WELL_R, WELL_X, 0.85, sg, 92.0, 92.0, if solid { 0.74 } else { 0.55 }), 0.004));
            }
        }

        // markers
        for (s, sg) in sides {
            for (nm, y) in [("upper", 1.02), ("lower", 0.66)] {
                let (pp, nn) = p::surface_point(tube, 0.78, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_F{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
                let (pp, nn) = p::surface_point(tube, -0.07, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_R{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            }
            let (pp, nn) = p::surface_point(tube, -0.03, 0.86, sg, (1.5, 5.5));
            marker(format!("latch_door_F{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            let (pp, nn) = p::surface_point(tube, -0.86, 0.86, sg, (1.5, 5.5));
            marker(format!("latch_door_R{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            marker(format!("hinge_hood_{s}"), DVec3::new(sg * 0.68, 1.12, 0.84), DVec3::X);
            marker(format!("hinge_tailgate_{s}"), DVec3::new(sg * 0.45, 1.80, -2.24), DVec3::X);
            for (k, (z, y)) in [(2.00, 1.09), (1.60, 1.10), (1.20, 1.11), (0.88, 1.12)].iter().enumerate() {
                let x = tube.key(6, *z).x + 0.03;
                marker(format!("bolt_fender_F{s}_{}", k + 1), DVec3::new(sg * x, y - 0.03, *z), DVec3::X);
            }
        }
        for (k, x) in [-0.55f64, -0.25, 0.25, 0.55].into_iter().enumerate() {
            let zz = front_z(x.abs(), 0.60);
            marker(format!("bolt_bumper_front_{}", k + 1), DVec3::new(x, 0.60, zz - 0.03), DVec3::X);
            let zz = rear_z(x.abs(), 0.60);
            marker(format!("bolt_bumper_rear_{}", k + 1), DVec3::new(x, 0.60, zz + 0.03), DVec3::X);
        }
        marker("latch_hood".into(), DVec3::new(0.0, 1.10, 2.08), DVec3::X);
        marker("latch_tailgate".into(), DVec3::new(0.0, 0.70, -2.27), DVec3::X);
        (out, extras, mk)
    }
}
