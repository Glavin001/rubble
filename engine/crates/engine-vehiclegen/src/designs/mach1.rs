//! The 1969 fastback ("SportsRoof") Mach 1, electric blue: a two-door
//! fastback with a long bonnet, a square nose with four round lamps, a
//! flat-backed tail with three tall lamps a side and a body side drawn in at
//! the sill. Port of `tools/vehicle-gen/designs/mach1_1969.py`.
//!
//! Profiles were read off photographs of real cars and scaled to the
//! published dimensions: wheelbase 108 in (2,743 mm), length 187.4 in
//! (4,760 mm), width 71.3 in (1,811 mm), height about 51 in (1,300 mm),
//! track 58.5 in (1,486 mm). No badges or logos are modelled.
//!
//! z is forward from the midpoint between the axles; y up from the ground;
//! x to the car's left. A standalone design (its own tube, layout, caps and
//! parts), rigged by the same pipeline as the sedan: [`RigSpec`] lists two
//! doors, and the body-in-white and the running gear come from its own
//! [`structure::Lines`] and [`mechanical::Gear`].

use crate::fx::{FxHashMap, FxHashSet};

use std::rc::Rc;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::assemble::{Design, RigSpec};
use crate::carbody::{circle_arc, CapPt, CarBody, Side};
use crate::curves::{bump, interp, linspace, sign, smoothstep, Pchip};
use crate::export::Marker;
use crate::frame::Frame;
use crate::layout::{Curve, FaceFn};
use crate::mechanical::{self, MechPart};
use crate::mesh::{box_, closed_lathe, grid, loft, orient_outward, rbox, rot_x, rot_y, rrect, sweep, Grid2, Mat, Mesh, SweepOpts};
use crate::panels::{boundary_loops, inner_panel};
use crate::parts::{self as p, RimStyle, WheelSpec};
use crate::physics::Physics;
use crate::structure::{self, Structure};
use crate::tube::{Handle, Key, Tube};
use crate::uvpack::Parts;

// -- the car ---------------------------------------------------------------------------------
const WHEELBASE: f64 = 2.743;
const TRACK: f64 = 1.486;
const ZF_AX: f64 = WHEELBASE / 2.0;
const ZR_AX: f64 = -WHEELBASE / 2.0;
const TYRE_D: f64 = 0.662;
const TYRE_W: f64 = 0.235;
const WHEEL_Y: f64 = TYRE_D / 2.0;
const Z_FRONT: f64 = 2.26;
const Z_REAR: f64 = -2.43;

const ARCH_F: (f64, f64, f64) = (ZF_AX, WHEEL_Y, 0.380);
const ARCH_R: (f64, f64, f64) = (ZR_AX, WHEEL_Y, 0.392);

const RED: [f64; 4] = [0.80, 0.03, 0.02, 1.0];
const CLEAR: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
const AMBER: [f64; 4] = [1.0, 0.45, 0.05, 1.0];

const HEADLIGHT: (f64, f64, f64) = (0.790, 0.622, 0.088); // x, y, R
const HEADLIGHT_INNER: (f64, f64, f64) = (0.535, 0.620, 0.068);
const TAILLIGHT: (f64, f64, f64, f64) = (0.400, 0.685, 0.650, 0.800); // x0, x1, y0, y1

/// The Mach 1's parameters (metres unless noted). Defaults are the 1969
/// car in electric blue.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Mach1Params {
    /// rim style: "magnum", "alloy" or "steel"
    pub rim_style: String,
    /// the rear wing (two uprights and a blade on the boot lid)
    pub wing: bool,
    /// linear RGB
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// The crash physics block: `engine/docs/generated-vehicles.md`.
    pub physics: Physics,
}

impl Default for Mach1Params {
    fn default() -> Self {
        Mach1Params { rim_style: "magnum".into(), wing: true, paint: [0.004, 0.045, 0.62], metallic: 0.45, roughness: 0.26, physics: Physics::default() }
    }
}

impl Mach1Params {
    pub fn validate(&self) -> Result<(), String> {
        let chk = |name: &str, v: f64| if v.is_finite() && (0.0..=1.0).contains(&v) { Ok(()) } else { Err(format!("{name} = {v} is outside 0..1")) };
        chk("metallic", self.metallic)?;
        chk("roughness", self.roughness)?;
        if !matches!(self.rim_style.as_str(), "magnum" | "alloy" | "steel") {
            return Err(format!("rim_style '{}' is not magnum, alloy or steel", self.rim_style));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Mach1 {
    pub p: Mach1Params,
}

// -- end faces -------------------------------------------------------------------------------

/// The outer headlamps sit at the back of tunnels in the fender ends, open
/// to the side and under a brow.
fn lamp_tunnel(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let px = smoothstep(0.620, 0.664, x);
    let py = smoothstep(0.497, 0.532, y) * smoothstep(0.752, 0.716, y);
    px * py
}

pub fn front_z(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let lean = 0.095 * ((0.80 - y) / 0.40).clamp(0.0, 1.0).powf(1.3);
    // square nose: corners roll only at the very edge
    let corner = 0.040 * (x / 0.905).powi(8);
    let vee = 0.012 * (1.0 - (x / 0.85).clamp(0.0, 1.0).powi(2));
    let w = lamp_tunnel(x, y);
    Z_FRONT - lean - corner * (1.0 - w) + vee - 0.165 * w
}

/// The tail's outer envelope: quarter-panel ends, lid lip and bumper line.
pub fn rear_z_body(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let tuck = 0.035 * ((0.52 - y) / 0.18).clamp(0.0, 1.0);
    let dish = 0.022 * (1.0 - (x / 0.80).clamp(0.0, 1.0).powi(2));
    let corner = 0.110 * (x / 0.88).clamp(0.0, 1.05).powi(4);
    Z_REAR + dish + tuck + corner
}

/// The tail panel is dished ~0.09 m into that envelope, between the
/// quarter-panel ends and under the boot lid's lip.
fn tail_recess(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let px = smoothstep(0.88, 0.72, x);
    let py = smoothstep(0.52, 0.64, y) * smoothstep(0.915, 0.83, y);
    0.090 * px * py
}

pub fn rear_z(x: f64, y: f64) -> f64 {
    rear_z_body(x, y) + tail_recess(x, y)
}

// -- the surface -----------------------------------------------------------------------------

fn tbl(v: &[(f64, f64)]) -> Pchip {
    Pchip::new(v)
}

fn key_of(px: Pchip, py: Pchip) -> Key {
    Key { x: Rc::new(move |z| px.eval(z)), y: Rc::new(move |z| py.eval(z)) }
}

/// The plan width's table, offset by `d` at 50 stations.
fn width_off(d: f64) -> Pchip {
    let w = tbl(&[(2.32, 0.885), (2.24, 0.897), (2.05, 0.900), (1.37, 0.902), (0.60, 0.902), (-0.70, 0.904), (-1.37, 0.906), (-1.90, 0.901), (-2.20, 0.892), (-2.44, 0.874)]);
    let zs = linspace(-2.5, 2.4, 50);
    tbl(&zs.iter().map(|&z| (z, w.eval(z) - d)).collect::<Vec<_>>())
}

/// The shoulder's x: the hip's top rolls inboard over the rear wheel.
fn shoulder_x() -> Pchip {
    let w = width_off(0.0);
    let hip = tbl(&[(-2.6, 0.040), (-1.55, 0.040), (-0.95, 0.007), (2.5, 0.007)]);
    let zs = linspace(-2.5, 2.4, 50);
    tbl(&zs.iter().map(|&z| (z, w.eval(z) - hip.eval(z))).collect::<Vec<_>>())
}

fn mach_tube() -> Tube {
    let c = Pchip::constant;
    let keys = vec![
        // 0 floor centre
        key_of(c(0.0), c(0.215)),
        // 1 rocker / sill bottom
        key_of(width_off(0.050), tbl(&[(2.4, 0.40), (1.80, 0.40), (1.0, 0.235), (-0.98, 0.235), (-1.80, 0.33), (-2.5, 0.30)])),
        // 2 door bottom line / rocker top
        key_of(width_off(0.016), tbl(&[(2.4, 0.47), (1.80, 0.46), (1.0, 0.30), (-0.98, 0.30), (-1.80, 0.48), (-2.5, 0.560)])),
        // 3 widest line
        key_of(width_off(0.0), tbl(&[(2.3, 0.52), (1.0, 0.52), (-0.98, 0.52), (-1.80, 0.62), (-2.5, 0.690)])),
        // 4 arch-top line (upper body side)
        key_of(width_off(0.002), tbl(&[(2.3, 0.69), (2.1, 0.705), (1.37, 0.74), (0.6, 0.72), (-0.7, 0.73), (-1.37, 0.77), (-2.5, 0.830)])),
        // 5 shoulder crease; at the nose it drops away convexly
        key_of(
            shoulder_x(),
            tbl(&[(2.26, 0.745), (2.10, 0.790), (1.75, 0.818), (1.37, 0.835), (1.0, 0.848), (0.585, 0.862), (-0.24, 0.872), (-0.715, 0.885), (-0.95, 0.908), (-1.37, 0.922), (-1.77, 0.918), (-2.10, 0.910), (-2.45, 0.900)]),
        ),
        // 6 belt (side-glass sill, 0.93 m) / bonnet shut line / quarter top
        key_of(
            tbl(&[(2.26, 0.665), (2.15, 0.672), (1.95, 0.700), (1.75, 0.762), (1.37, 0.786), (0.60, 0.782), (0.40, 0.790), (-0.715, 0.800), (-1.37, 0.795), (-2.0, 0.780), (-2.45, 0.770)]),
            tbl(&[(2.26, 0.760), (2.10, 0.808), (1.75, 0.840), (1.37, 0.858), (0.60, 0.895), (0.40, 0.925), (0.12, 0.928), (-0.62, 0.935), (-0.715, 0.937), (-0.95, 0.955), (-1.37, 0.963), (-1.77, 0.952), (-2.10, 0.942), (-2.45, 0.935)]),
        ),
        // 7 daylight-opening top (roof rail outer); the A-pillar foot is at z 0.40
        key_of(
            tbl(&[(2.26, 0.500), (1.95, 0.580), (1.37, 0.620), (0.80, 0.660), (0.47, 0.700), (0.40, 0.712), (0.066, 0.640), (-0.24, 0.635), (-0.62, 0.640), (-0.90, 0.660), (-1.37, 0.700), (-1.77, 0.730), (-2.00, 0.660), (-2.20, 0.600), (-2.45, 0.570)]),
            tbl(&[(2.26, 0.765), (2.10, 0.812), (1.37, 0.866), (0.60, 0.900), (0.47, 0.912), (0.40, 0.926), (0.066, 1.218), (-0.24, 1.232), (-0.62, 1.226), (-0.73, 1.218), (-0.90, 1.175), (-1.37, 1.040), (-1.77, 0.957), (-2.10, 0.947), (-2.45, 0.943)]),
        ),
        // 8 glass edge (windscreen / back light) / bonnet line / boot lid
        key_of(
            tbl(&[(2.26, 0.380), (1.95, 0.440), (1.37, 0.460), (0.80, 0.490), (0.52, 0.540), (0.42, 0.640), (0.09, 0.585), (-0.24, 0.580), (-0.62, 0.580), (-0.90, 0.585), (-1.37, 0.600), (-1.77, 0.630), (-2.00, 0.540), (-2.20, 0.490), (-2.45, 0.470)]),
            tbl(&[(2.26, 0.770), (2.10, 0.818), (1.37, 0.870), (0.60, 0.905), (0.52, 0.912), (0.42, 0.928), (0.09, 1.238), (-0.24, 1.262), (-0.62, 1.243), (-0.90, 1.192), (-1.37, 1.065), (-1.77, 0.958), (-2.10, 0.949), (-2.45, 0.944)]),
        ),
        // 9 top centre line: the bonnet droops to the nose
        key_of(
            c(0.0),
            tbl(&[(2.26, 0.775), (2.20, 0.795), (2.10, 0.822), (1.75, 0.852), (1.37, 0.868), (1.00, 0.886), (0.60, 0.905), (0.52, 0.912), (0.47, 0.928), (0.12, 1.262), (-0.24, 1.300), (-0.62, 1.268), (-0.90, 1.215), (-1.37, 1.090), (-1.77, 0.975), (-2.10, 0.950), (-2.45, 0.945)]),
        ),
    ];
    // the shoulder is a crisp crease along the front fender and door and
    // rolls into a round hip over the rear wheel
    let hip = Rc::new(tbl(&[(-2.6, 1.0), (-1.55, 1.0), (-0.95, 0.0), (2.5, 0.0)]));
    let f = |h: &Rc<Pchip>, a: f64, b: f64| {
        let h = h.clone();
        Handle::F(Rc::new(move |z| a + b * h.eval(z)))
    };
    let k = |a: f64| (Handle::C(a), Handle::C(a));
    let handles = vec![
        k(1.0 / 3.0),
        k(0.30),
        k(0.33),
        k(0.36),
        k(0.33),
        (f(&hip, 0.14, 0.20), f(&hip, 0.12, 0.24)),
        (f(&hip, 0.22, 0.10), Handle::C(0.25)),
        k(0.33),
        k(0.33),
        k(1.0 / 3.0),
    ];
    // displacements
    let arches = [ARCH_F, ARCH_R];
    let arch_lip: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let side = if n.x.abs() > 0.5 { 1.0 } else { 0.0 };
        let mut out = DVec3::ZERO;
        for (zc, yc, r) in arches {
            let d = ((z - zc).powi(2) + (pp.y - yc).powi(2)).sqrt();
            let lip = 0.007 * bump(d, r + 0.012, 0.045) * if pp.y > yc - 0.1 { 1.0 } else { 0.0 };
            out.x += sign(pp.x) * lip * side;
        }
        out
    });
    // the raised centre of the bonnet, fading out at the cowl and the nose
    let bonnet_bulge: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let ax = pp.x.abs();
        let w = smoothstep(0.42, 0.34, ax) * smoothstep(0.62, 0.75, z) * smoothstep(2.32, 2.20, z);
        DVec3::new(0.0, 0.012 * w * if n.y.abs() > 0.6 { 1.0 } else { 0.0 }, 0.0)
    });
    // the '69 rear quarter swells out over the rear wheel
    let quarter_hip: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let side = if n.x.abs() > 0.3 { 1.0 } else { 0.0 };
        let w = bump(z, -1.45, 0.75) * bump(pp.y, 0.70, 0.30);
        DVec3::new(sign(pp.x) * 0.008 * w * side, 0.0, 0.0)
    });
    // a crisp light-catching ledge under the shoulder from the headlamp
    // bucket back to the quarter scoop, the panel below it drawn in
    let side_sculpt: crate::tube::Displace = Rc::new(move |pp: DVec3, n: DVec3, z: f64, _v: f64| {
        let side = ((n.x.abs() - 0.6) / 0.3).clamp(0.0, 1.0);
        let zw = ((z + 1.05) / 0.25).clamp(0.0, 1.0) * ((2.30 - z) / 0.15).clamp(0.0, 1.0);
        let prof = 0.004 * bump(pp.y, 0.775, 0.045) - 0.006 * bump(pp.y, 0.66, 0.11);
        DVec3::new(sign(pp.x) * prof * side * zw, 0.0, 0.0)
    });
    Tube::new(keys, handles, vec![arch_lip, bonnet_bulge, quarter_hip, side_sculpt])
}

// -- layout ----------------------------------------------------------------------------------

const ZLINES: [(&str, f64); 14] = [
    ("F", 2.26),
    ("AF1", 1.76),
    ("AF2", ZF_AX),
    ("AF3", 0.99),
    ("DF", 0.585),
    ("WB", 0.40),
    ("WT", 0.12),
    ("GR", -0.63),
    ("DR", -0.715),
    ("Q", -0.80),
    ("AR1", -0.97),
    ("AR2", ZR_AX),
    ("AR3", -1.77),
    ("T", -2.43),
];
const VLINES: [f64; 12] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.33, 5.67, 6.0, 7.0, 8.0, 9.0];

impl Mach1 {
    pub fn new(p: Mach1Params) -> Mach1 {
        Mach1 { p }
    }

    fn wheel_spec(&self) -> WheelSpec {
        let rim_style = match self.p.rim_style.as_str() {
            "steel" => RimStyle::Steel,
            "alloy" => RimStyle::Alloy,
            _ => RimStyle::Magnum,
        };
        WheelSpec { track: TRACK, wheel_y: WHEEL_Y, tyre_d: TYRE_D, tyre_w: TYRE_W, rim_d: 15.0 * 0.0254, rim_w: 7.0 * 0.0254, rim_style, disc_d: 0.28 }
    }

    /// The lines the body-in-white is placed from ([`structure::unibody`]).
    pub fn structure_lines(&self) -> structure::Lines {
        let z = |n: &str| ZLINES.iter().find(|l| l.0 == n).unwrap().1;
        structure::Lines {
            f: Frame::identity(),
            axle_f: ZF_AX,
            axle_r: ZR_AX,
            z_door_front: z("DF"),
            z_screen_base: z("WB"),
            z_screen_top: z("WT"),
            z_b: z("Q"),
            z_door_rear: z("AR1"),
            z_rear_top: -0.90,
            z_rear_base: z("AR3"),
            well_x: 0.535,
            well_r: 0.366,
            wheel_y: WHEEL_Y,
        }
    }

    /// The running gear's dimensions for [`mechanical::sedan_parts`].
    pub fn gear(&self) -> mechanical::Gear {
        mechanical::Gear {
            f: Frame::identity(),
            half_track: TRACK / 2.0,
            wheel_y: WHEEL_Y,
            zf: ZF_AX,
            zr: ZR_AX,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: 15.0 * 0.0254,
            rim_w: 7.0 * 0.0254,
            axle_f: ZF_AX,
            axle_r: ZR_AX,
            z_nose: Rc::new(front_z),
            z_tail: Rc::new(rear_z_body),
        }
    }

    pub fn unibody(&self) -> Structure {
        let tube = mach_tube();
        structure::unibody(&self.structure_lines(), &structure::Skin { tube: &tube, front_z: &front_z, rear_z: &rear_z })
    }

    fn build_nose(&self, cb: &mut CarBody) {
        let i0 = cb.zi("F");
        let hl = HEADLIGHT;
        let (r, d) = (hl.2, hl.2 * 0.5f64.sqrt());
        for (s, side) in [("L", Side::L), ("R", Side::R)] {
            let sg = side.sign();
            let n = |jl: usize| cb.nid(i0, cb.j(jl, side));
            let pos: Vec<DVec3> = (0..=8).map(|k| if k == 0 { DVec3::ZERO } else { cb.l.node_pos(n(k)) }).collect();
            let xr = pos[8].x;
            let left: Vec<CapPt> = (1..=5).map(|k| CapPt::N(n(k))).collect();
            let mut right: Vec<CapPt> = (1..=4).map(|k| CapPt::P(xr, pos[k].y)).collect();
            right.push(CapPt::N(n(8)));
            let bottom = vec![CapPt::N(n(1)), CapPt::P(pos[6].x, pos[1].y), CapPt::P(pos[7].x, pos[1].y), CapPt::P(xr, pos[1].y)];
            let top: Vec<CapPt> = (5..=8).map(|k| CapPt::N(n(k))).collect();
            let (xc, yc) = (sg * hl.0, hl.1);
            let mut ov: FxHashMap<(usize, usize), DVec2> = FxHashMap::default();
            ov.insert((2, 1), DVec2::new(xc + sg * d, yc - d));
            ov.insert((2, 2), DVec2::new(xc - sg * d, yc - d));
            ov.insert((3, 1), DVec2::new(xc + sg * d, yc + d));
            ov.insert((3, 2), DVec2::new(xc - sg * d, yc + d));
            let arcs: Vec<((usize, usize), (usize, usize), Curve)> = if side == Side::L {
                vec![
                    ((2, 1), (2, 2), circle_arc(xc, yc, r, -45.0, -135.0)),
                    ((2, 2), (3, 2), circle_arc(xc, yc, r, -135.0, -225.0)),
                    ((3, 1), (3, 2), circle_arc(xc, yc, r, 45.0, 135.0)),
                    ((2, 1), (3, 1), circle_arc(xc, yc, r, -45.0, 45.0)),
                ]
            } else {
                vec![
                    ((2, 1), (2, 2), circle_arc(xc, yc, r, -135.0, -45.0)),
                    ((2, 2), (3, 2), circle_arc(xc, yc, r, -45.0, 45.0)),
                    ((3, 1), (3, 2), circle_arc(xc, yc, r, 135.0, 45.0)),
                    ((2, 1), (3, 1), circle_arc(xc, yc, r, -135.0, -225.0)),
                ]
            };
            let holes: FxHashSet<(usize, usize)> = [(2, 1)].into_iter().collect();
            let name = format!("fender_F{s}");
            let fz: FaceFn = Rc::new(front_z);
            cb.cap(&format!("capF{s}"), fz, 1.0, &left, &right, &bottom, &top, &move |_r, _c| Some(name.clone()), &holes, &ov, arcs);
        }
    }

    fn build_tail(&self, cb: &mut CarBody) {
        // tail panel: one cap across the back, taillight openings left open
        let it = cb.zi("T");
        let j = cb.jn;
        let tl = TAILLIGHT;
        let nt = |k: usize| cb.nid(it, k % j);
        let left: Vec<CapPt> = (1..=5).map(|k| CapPt::N(nt(k))).collect();
        let right: Vec<CapPt> = (1..=5).map(|k| CapPt::N(nt(j - k))).collect();
        let y0 = cb.l.node_pos(nt(1)).y;
        let xk1 = cb.l.node_pos(nt(1)).x;
        let xs = [tl.1, tl.0, 0.0, -tl.0, -tl.1];
        let tube = cb.tube();
        const DROP: f64 = 0.0125;
        let lid: Vec<f64> = xs
            .iter()
            .map(|&x| {
                let v = tube.v_at_x(-2.43, x.abs(), 6.0, 9.0);
                tube.eval(-2.43, v).y - DROP
            })
            .collect();
        // the tail panel tucks 7 mm under the lid's rear edge all the way to
        // the corners (the lid overhangs it), so its top-row corners are its own
        let pk = cb.l.node_pos(nt(8));
        let mut top = vec![CapPt::N(nt(5)), CapPt::N(nt(6)), CapPt::N(nt(7)), CapPt::P(pk.x, pk.y - DROP)];
        top.extend(xs.iter().zip(&lid).map(|(&x, &y)| CapPt::P(x, y)));
        top.extend([CapPt::P(-pk.x, pk.y - DROP), CapPt::N(nt(j - 7)), CapPt::N(nt(j - 6)), CapPt::N(nt(j - 5))]);
        let fan: Vec<f64> = [1.0, 2.0, 3.0].iter().map(|k| xk1 - (xk1 - tl.1) * k / 4.0).collect();
        let mut bottom = vec![CapPt::N(nt(1))];
        bottom.extend(fan.iter().map(|&x| CapPt::P(x, y0)));
        bottom.extend(xs.iter().map(|&x| CapPt::P(x, y0)));
        bottom.extend(fan.iter().rev().map(|&x| CapPt::P(-x, y0)));
        bottom.push(CapPt::N(nt(j - 1)));
        let mut ov: FxHashMap<(usize, usize), DVec2> = FxHashMap::default();
        let ovs = [
            ((2, 4), (tl.1, tl.2)),
            ((2, 5), (tl.0, tl.2)),
            ((3, 4), (tl.1, tl.3)),
            ((3, 5), (tl.0, tl.3)),
            ((2, 7), (-tl.0, tl.2)),
            ((2, 8), (-tl.1, tl.2)),
            ((3, 7), (-tl.0, tl.3)),
            ((3, 8), (-tl.1, tl.3)),
        ];
        for (k, (x, y)) in ovs {
            ov.insert(k, DVec2::new(x, y));
        }
        // the tail panel's top edges follow the boot lid's rear edge
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
        tx.extend(xs.iter().cloned());
        tx.push(-pk.x);
        let arcs: Vec<((usize, usize), (usize, usize), Curve)> = (0..tx.len() - 1).map(|k| ((4, 3 + k), (4, 4 + k), lid_curve(tx[k], tx[k + 1]))).collect();
        let holes: FxHashSet<(usize, usize)> = [(2, 4), (2, 7)].into_iter().collect();
        let fz: FaceFn = Rc::new(rear_z);
        cb.cap("capR", fz, -1.0, &left, &right, &bottom, &top, &|_r, _c| Some("body_shell".into()), &holes, &ov, arcs);
    }
}

impl Design for Mach1 {
    fn name(&self) -> String {
        "mach1_1969".into()
    }

    fn paint(&self) -> ([f64; 3], f64, f64) {
        (self.p.paint, self.p.metallic, self.p.roughness)
    }

    fn tube(&self) -> Tube {
        mach_tube()
    }

    fn zlines(&self) -> Vec<(&'static str, f64)> {
        ZLINES.to_vec()
    }

    fn vlines(&self) -> Vec<f64> {
        VLINES.to_vec()
    }

    /// Part of the tube block between z-lines `zf`..`zt` and left v-lines
    /// lo..hi. v-line indices: 0 floor, 1 rocker, 2 door bottom, 3 widest,
    /// 4 arch line, 5 shoulder, 6-7 shoulder strip, 8 belt, 9 roof rail,
    /// 10 glass edge, 11 centre.
    fn assign(&self, zf: &str, _zt: &str, lo: usize, _hi: usize, side: Side) -> Option<(String, Mat)> {
        let s = side.tag();
        let i = zf;
        let front = matches!(i, "F" | "AF1" | "AF2" | "AF3");
        let cabin = matches!(i, "DF" | "WB" | "WT" | "GR");
        let r = |n: String| Some((n, Mat::Paint));
        if lo == 0 {
            return if matches!(i, "DF" | "WB" | "WT" | "GR" | "DR" | "Q") { r("body_shell".into()) } else { None };
        }
        if lo <= 7 {
            if front {
                return r(format!("fender_F{s}"));
            }
            if cabin {
                return r(if lo == 1 { "body_shell".into() } else { format!("door_F{s}") });
            }
            return r("body_shell".into());
        }
        if front {
            return r("hood".into());
        }
        if i == "AR3" {
            return r("trunk_lid".into());
        }
        if lo == 8 {
            return r(match i {
                // the mirror sail at the door's top front
                "DF" => format!("door_F{s}"),
                "WB" | "WT" => format!("glass_door_F{s}"),
                "GR" | "DR" => format!("glass_quarter_{s}"),
                _ => "body_shell".into(),
            });
        }
        if i == "DF" {
            return r("body_shell".into());
        }
        if lo == 10 {
            if i == "WB" {
                return r("glass_windshield".into());
            }
            if matches!(i, "Q" | "AR1" | "AR2") {
                return r("glass_rear".into());
            }
        }
        r("body_shell".into())
    }

    fn build_layout(&self, cb: &mut CarBody) {
        // bonnet rear edge, windscreen base and A-pillar (traced off the
        // side photograph): the side glass meets the pillar at z 0.40, the
        // windscreen base bows forward to 0.47 at the centre; the quarter
        // glass's rear edge leans forward going up
        let ov: [(&str, &[(usize, f64)]); 5] = [
            ("DF", &[(8, 0.56), (9, 0.545), (10, 0.53), (11, 0.52)]),
            ("WB", &[(10, 0.42), (11, 0.47)]),
            ("WT", &[(8, 0.10), (9, 0.066), (10, 0.09), (11, 0.12)]),
            ("DR", &[(9, -0.68), (10, -0.80), (11, -0.81)]),
            ("Q", &[(8, -0.881), (9, -0.726), (10, -0.89), (11, -0.90)]),
        ];
        for (name, zs) in ov {
            let i = cb.zi(name);
            for &(jl, z) in zs {
                cb.set_z(i, jl, z);
            }
        }
        let iq = cb.zi("Q");
        let (za, zb) = (-0.881, -0.726);
        let (va, vb) = (6.0, 7.0);
        let vm = cb.tube().vm;
        for side in [Side::L, Side::R] {
            let curve: Curve = Rc::new(move |t: f64| {
                let z = za + (zb - za) * t - 0.040 * (std::f64::consts::PI * t).sin();
                let mut v = va + (vb - va) * t;
                if side == Side::R {
                    v = vm - v;
                }
                DVec2::new(z, v)
            });
            let (a, b) = (cb.nid(iq, cb.j(8, side)), cb.nid(iq, cb.j(9, side)));
            cb.arcs.push((a, b, curve));
        }
        let (fi, ti) = (cb.zi("F"), cb.zi("T"));
        cb.end_fit(fi, &front_z);
        cb.end_fit(ti, &rear_z);
        let (af1, af2, af3) = (cb.zi("AF1"), cb.zi("AF2"), cb.zi("AF3"));
        cb.arch((ARCH_F.0, ARCH_F.1), ARCH_F.2, (af1, af2, af3), &[1, 2, 3, 4], &[12.0, 21.0, 40.0, 62.0, 90.0, 118.0, 145.0, 172.0, 193.5], (0.6, 5.2));
        let (ar1, ar2, ar3) = (cb.zi("AR1"), cb.zi("AR2"), cb.zi("AR3"));
        cb.arch((ARCH_R.0, ARCH_R.1), ARCH_R.2, (ar1, ar2, ar3), &[1, 2, 3, 4], &[-13.4, 8.0, 35.0, 62.0, 90.0, 118.0, 145.0, 160.0, 175.0], (0.6, 5.2));
    }

    fn build_caps(&self, cb: &mut CarBody) {
        self.build_nose(cb);
        self.build_tail(cb);
    }

    fn parts(&self, cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        self.make_parts(cb)
    }

    fn post(&self, parts: &mut Parts) {
        // Mach 1 bonnet: the matte-black centre between the glass-edge
        // lines (a mesh row, so the colour break is a clean line)
        let tube = mach_tube();
        let Some(m) = parts.get_mut("hood") else { return };
        let skin_face = |m: &Mesh, f: &[u32]| match &m.surf {
            Some(s) => f.iter().all(|&i| s[i as usize] == crate::mesh::Surf::Outer),
            None => true,
        };
        let black = |m: &Mesh, f: &[u32]| {
            let c = f.iter().map(|&i| m.v[i as usize]).sum::<DVec3>() / f.len() as f64;
            tube.v_at_x(c.z, c.x.abs(), 7.0, 9.0) > 8.0
        };
        let qsel: Vec<bool> = m.q.iter().map(|f| skin_face(m, f) && black(m, f)).collect();
        let tsel: Vec<bool> = m.t.iter().map(|f| skin_face(m, f) && black(m, f)).collect();
        for (k, s) in qsel.into_iter().enumerate() {
            if s {
                m.qm[k] = Mat::TrimBlack;
            }
        }
        for (k, s) in tsel.into_iter().enumerate() {
            if s {
                m.tm[k] = Mat::TrimBlack;
            }
        }
    }

    fn rig_spec(&self) -> Option<RigSpec> {
        Some(RigSpec {
            wheelbase: WHEELBASE,
            track: TRACK,
            axle_f: ZF_AX,
            axle_r: ZR_AX,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: 15.0 * 0.0254,
            wheel_y: WHEEL_Y,
            z_front: Z_FRONT,
            z_rear: Z_REAR,
            doors: ["FL", "FR"].iter().map(|s| s.to_string()).collect(),
            rear_closure: "trunk_lid".into(),
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

/// numpy.arange(a, b, step) for floats.
fn arange(a: f64, b: f64, step: f64) -> Vec<f64> {
    let n = ((b - a) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| a + i as f64 * step).collect()
}

/// A path round the whole nose or tail from the half outline (x >= 0).
fn full(pxz: &[DVec2]) -> Vec<DVec2> {
    let mut r: Vec<DVec2> = pxz.iter().rev().map(|q| DVec2::new(-q.x, q.y)).collect();
    r.pop();
    r.extend_from_slice(pxz);
    r
}

impl Mach1 {
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
        let up_y = DVec3::Y;
        let _ = up_y;

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

        // headlamps: outer in the fender ends, inner in the grille
        let zg = front_z(0.60, 0.65) - 0.045;
        for (s, sg) in sides {
            let hl = HEADLIGHT;
            let zs = front_z(hl.0, hl.1);
            let g = p::graph_normal(&front_z, hl.0, hl.1, 1.0);
            let nrm = DVec3::new(g.x * sg, g.y, g.z);
            // the lamp sits at the bottom of a scooped bucket under the fender brow
            const DEEP: f64 = 0.020;
            let cen = DVec3::new(sg * hl.0, hl.1, zs) - nrm * DEEP;
            let (mut l1, h1) = p::round_lamp(cen, hl.2 - 0.020, 1.0, CLEAR, 0.07, Some(nrm));
            let rt = p::rot_to(nrm);
            let rings: Vec<Vec<DVec3>> = [(0.0, hl.2 - 0.003, 0.003), (0.5, hl.2 - 0.006, 0.007), (1.0, hl.2 - 0.010, 0.011)]
                .iter()
                .map(|&(_t, rr, dd)| {
                    (0..48)
                        .map(|k| {
                            let a = std::f64::consts::TAU * k as f64 / 48.0;
                            let off = rt * DVec3::new(0.0, rr * a.cos(), rr * a.sin());
                            let mut q = DVec3::new(sg * hl.0, hl.1, 0.0) + off;
                            q.z = front_z(q.x.abs(), q.y);
                            q - nrm * dd
                        })
                        .collect()
                })
                .collect();
            let gr = Grid2::from_fn(rings.len(), 48, |i, j| rings[i][j]);
            let mut bucket = grid(&gr, None, Mat::Paint, false, true, -1);
            let cn = bucket.vertex_normals().iter().copied().sum::<DVec3>() / bucket.v.len() as f64;
            if cn.dot(nrm) < 0.0 {
                bucket.flip();
            }
            // the tunnel's inner wall, from its floor forward to behind the
            // grille surround
            let ys0 = linspace(0.497, 0.752, 60);
            let zb0: Vec<f64> = ys0.iter().map(|&y| front_z(0.665, y)).collect();
            let zfw = zg - 0.008;
            let ok: Vec<usize> = (0..60).filter(|&k| zfw - zb0[k] > 0.025).collect();
            let ys = linspace(ys0[ok[0]], ys0[*ok.last().unwrap()], 11);
            let tt = linspace(0.0, 1.0, 4);
            let pw = Grid2::from_fn(ys.len(), tt.len(), |i, j| {
                let zb = front_z(0.665, ys[i]);
                DVec3::new(sg * 0.654, ys[i], zb + (zfw - zb) * tt[j])
            });
            let mut wall = grid(&pw, None, Mat::Paint, false, false, -1);
            if sg < 0.0 {
                wall.flip();
            }
            let hi = HEADLIGHT_INNER;
            let (l2, h2) = p::round_lamp(DVec3::new(sg * hi.0, hi.1, zg + 0.020), hi.2, 1.0, CLEAR, 0.014, None);
            // side markers: amber at the front of the fender, red at the tail
            let (fm, fn_) = p::surface_point(tube, 1.97, 0.560, sg, (1.5, 5.5));
            let mut mk_ = rbox(DVec3::ZERO, DVec3::new(0.004, 0.016, 0.034), 0.003, 1, Mat::LampLens);
            mk_.transform(Some(p::rot_to(fn_)), fm + fn_ * 0.005);
            mk_.tint(AMBER);
            l1.add(&l2).add(&mk_);
            // (the bucket and the tunnel wall are sheets: they ride the lamp
            // housing, not the closed fender solid)
            let mut h1 = h1;
            h1.add(&h2).add(&bucket).add(&wall);
            out.insert(&format!("headlight_{s}_lens"), l1);
            out.insert(&format!("headlight_{s}_housing"), h1);
        }

        // grille: black egg-crate in a thin bright surround, recessed between
        // the fender ends, closed behind by a back plate
        let mut g = Mesh::new();
        let (gx, gy0, gy1) = (0.652, 0.495, 0.735);
        let gm = (gy0 + gy1) / 2.0;
        let outline = rrect(2.0 * gx, gy1 - gy0, 0.035, 5);
        let path: Vec<DVec3> = outline.iter().map(|q| DVec3::new(q.x, q.y + gm, zg + 0.004)).collect();
        g.add(&orient_outward(sweep(&path, &rrect(0.010, 0.022, 0.004, 2), Mat::Chrome, &SweepOpts { closed: true, up: DVec3::Z, ..Default::default() })));
        g.add(&box_(DVec3::new(0.0, gm, zg - 0.045), DVec3::new(gx - 0.004, (gy1 - gy0) / 2.0 - 0.004, 0.003), Mat::TrimBlack, None));
        for x in arange(-0.420, 0.4201, 0.035) {
            g.add(&box_(DVec3::new(x, gm, zg - 0.018), DVec3::new(0.0022, (gy1 - gy0) / 2.0 - 0.012, 0.022), Mat::TrimBlack, None));
        }
        for y in linspace(gy0 + 0.04, gy1 - 0.04, 6) {
            g.add(&box_(DVec3::new(0.0, y, zg - 0.018), DVec3::new(0.43, 0.0022, 0.022), Mat::TrimBlack, None));
        }
        // solid black end panels round the inner lamps
        for sg in [1.0, -1.0] {
            g.add(&box_(DVec3::new(sg * 0.545, gm, zg - 0.006), DVec3::new(0.100, (gy1 - gy0) / 2.0 - 0.010, 0.004), Mat::TrimBlack, None));
        }
        out.insert("grille", g);

        // front bumper: chrome blade wrapping round to the wheel openings,
        // painted valance and a black chin spoiler below it
        let yb = 0.452;
        let ols: Vec<Vec<DVec2>> = [0.40, 0.452, 0.50].iter().map(|&y| full(&p::plan_outline(tube, &front_z, y, 1.0, 2.3, 0.14, 28, 10))).collect();
        let pp = p::fit_path(&ols, 0.022, 1.0, 3, Some(0.018));
        let path: Vec<DVec3> = pp.iter().map(|q| DVec3::new(q.x, yb, q.y)).collect();
        let mut bf = orient_outward(sweep(&path, &rrect(0.032, 0.084, 0.016, 4), Mat::Chrome, &SweepOpts::default()));
        let ols = vec![full(&p::plan_outline(tube, &front_z, 0.40, 1.0, 2.3, 0.06, 28, 10))];
        let pv = p::fit_path(&ols, -0.022, 1.0, 3, Some(0.018));
        let pv = &pv[4..pv.len() - 4];
        let vpath: Vec<DVec3> = pv.iter().map(|q| DVec3::new(q.x, 0.0, q.y)).collect();
        let vsec = [DVec2::new(0.004, 0.392), DVec2::new(-0.026, 0.392), DVec2::new(-0.046, 0.310), DVec2::new(-0.058, 0.282), DVec2::new(-0.040, 0.280), DVec2::new(-0.014, 0.310)];
        bf.add(&orient_outward(sweep(&vpath, &p::densify(&vsec, 0.02), Mat::Paint, &SweepOpts::default())));
        let csec = [DVec2::new(0.070, 0.272), DVec2::new(-0.050, 0.272), DVec2::new(-0.050, 0.260), DVec2::new(0.070, 0.262)];
        let k = vpath.len() / 8;
        let cpath: Vec<DVec3> = vpath[k..vpath.len() - k].iter().map(|q| DVec3::new(q.x * 0.97, q.y, q.z)).collect();
        bf.add(&orient_outward(sweep(&cpath, &csec, Mat::TrimBlack, &SweepOpts::default())));
        out.insert("bumper_front", bf);
        // parking lamps: wide amber lenses in the valance just under the blade
        for (s, sg) in sides {
            let kx = (0..vpath.len()).min_by(|&a, &b| (vpath[a].x - sg * 0.582).abs().total_cmp(&(vpath[b].x - sg * 0.582).abs())).unwrap();
            let zv = vpath[kx].z;
            let mut pl = rbox(DVec3::new(sg * 0.582, 0.366, zv + 0.017), DVec3::new(0.072, 0.022, 0.004), 0.007, 1, Mat::LampLens);
            pl.tint(AMBER);
            out.get_mut(&format!("headlight_{s}_lens")).unwrap().add(&pl);
            out.get_mut(&format!("headlight_{s}_housing")).unwrap().add(&rbox(DVec3::new(sg * 0.582, 0.366, zv + 0.009), DVec3::new(0.078, 0.027, 0.0035), 0.008, 1, Mat::TrimBlack));
        }
        for (k_, x) in [-0.62f64, -0.30, 0.30, 0.62].iter().enumerate() {
            let zz = front_z(x.abs(), yb);
            marker(format!("bolt_bumper_front_{}", k_ + 1), DVec3::new(*x, yb, zz - 0.004), DVec3::X);
        }

        // rear bumper
        let yr = 0.600;
        let ols: Vec<Vec<DVec2>> = [0.56, 0.60, 0.64].iter().map(|&y| full(&p::plan_outline(tube, &rear_z_body, y, -1.0, -2.4, 0.15, 28, 10))).collect();
        let mut pr = p::fit_path(&ols, 0.023, -1.0, 3, Some(0.018));
        pr.reverse();
        let path: Vec<DVec3> = pr.iter().map(|q| DVec3::new(q.x, yr, q.y)).collect();
        out.insert("bumper_rear", orient_outward(sweep(&path, &rrect(0.040, 0.088, 0.018, 4), Mat::Chrome, &SweepOpts::default())));
        for (k_, x) in [-0.62f64, -0.30, 0.30, 0.62].iter().enumerate() {
            let zz = rear_z_body(x.abs(), yr);
            marker(format!("bolt_bumper_rear_{}", k_ + 1), DVec3::new(*x, yr, zz + 0.004), DVec3::X);
        }

        // taillamps: three lenses each side in a bright bezel, a reflector box behind
        let tl = TAILLIGHT;
        for (s, sg) in sides {
            let xm = (tl.0 + tl.1) / 2.0;
            let ym = (tl.2 + tl.3) / 2.0;
            let zt = rear_z(xm, ym);
            let w = tl.1 - tl.0;
            let hgt = tl.3 - tl.2;
            let mut lens = Mesh::new();
            let mut hous = Mesh::new();
            let gapx = 0.012;
            let lw = (w - 0.020 - 2.0 * gapx) / 3.0;
            hous.add(&box_(DVec3::new(sg * xm, ym, zt + 0.046), DVec3::new(w / 2.0 - 0.005, hgt / 2.0 - 0.005, 0.006), Mat::TrimBlack, None));
            for k in 0..3 {
                let x = tl.0 + 0.010 + lw / 2.0 + k as f64 * (lw + gapx);
                let zk = rear_z(x, ym);
                let lh = hgt - 0.016;
                lens.add(&rbox(DVec3::new(sg * x, ym, zk + 0.016), DVec3::new(lw / 2.0 - 0.007, lh / 2.0 - 0.008, 0.006), (lw / 2.0 - 0.008).min(0.022), 3, Mat::LampLens));
                let rr = rrect(lw - 0.002, lh + 0.002, (lw / 2.0 - 0.002).min(0.024), 4);
                let path: Vec<DVec3> = rr.iter().map(|q| DVec3::new(sg * x + q.x, ym + q.y, zk + 0.008)).collect();
                hous.add(&orient_outward(sweep(&path, &rrect(0.003, 0.010, 0.0013, 1), Mat::Chrome, &SweepOpts { closed: true, up: DVec3::Z, ..Default::default() })));
                hous.add(&box_(DVec3::new(sg * x, ym, zk + 0.032), DVec3::new(lw / 2.0 - 0.006, lh / 2.0 - 0.006, 0.003), Mat::LampEmissive, None));
            }
            lens.tint(RED);
            let (rm, rn) = p::surface_point(tube, -2.12, 0.605, sg, (1.5, 5.5));
            let mut mr = rbox(DVec3::ZERO, DVec3::new(0.004, 0.016, 0.040), 0.003, 1, Mat::LampLens);
            mr.transform(Some(p::rot_to(rn)), rm + rn * 0.007);
            mr.tint(RED);
            lens.add(&mr);
            out.insert(&format!("taillight_{s}_lens"), lens);
            out.insert(&format!("taillight_{s}_housing"), hous);
        }

        // mirrors, handles, wipers
        for (s, sg) in sides {
            let v = if sg > 0.0 { 6.35 } else { vm - 6.35 };
            let base = tube.eval(0.50, v);
            out.insert(&format!("mirror_{s}"), p::sport_mirror(base, sg, Mat::Paint));
            marker(format!("mount_mirror_{s}"), base, DVec3::new(sg, 0.0, 0.0));
            out.insert(&format!("handle_F{s}"), p::door_handle(tube, -0.53, 0.826, sg, 0.13));
        }
        for (s, x0, x1) in [("L", 0.56, 0.10), ("R", 0.02, -0.44)] {
            let zs = 0.40;
            let pts: Vec<DVec3> = linspace(x0, x1, 8)
                .into_iter()
                .map(|x: f64| {
                    let mut v = tube.v_at_x(zs, x.abs(), 7.9, 9.0);
                    if x < 0.0 {
                        v = vm - v;
                    }
                    let (pp, nn) = tube.eval_n(zs, v);
                    pp + nn * 0.004
                })
                .collect();
            out.insert(&format!("wiper_{s}"), p::wiper(&pts, Mat::TrimBlack));
        }

        // body extras: quarter scoops, rocker mouldings, window reveals, fuel cap
        let mut bx = Mesh::new();
        for sg in [1.0, -1.0] {
            // (the right scoop is the left one mirrored: lofted on the right
            // its caps wind against its sides)
            let mut bump = p::surface_bump(tube, -0.735, -1.00, 0.775, 0.925, 0.030, 1.0, Mat::Paint);
            if sg < 0.0 {
                bump.mirror_x();
            }
            bx.add(&bump);
            // the scoop's dark mouth facing forward
            let (q, nq) = p::surface_point(tube, -0.742, 0.850, sg, (1.5, 5.5));
            let nq = nq / nq.length();
            let mut mouth = rbox(DVec3::ZERO, DVec3::new(0.011, 0.058, 0.0015), 0.0014, 1, Mat::TrimBlack);
            mouth.transform(None, q + nq * 0.016);
            bx.add(&mouth);
            let sv = linspace(0.52, -0.66, 30);
            let (pp, _) = p::path_on_surface(tube, &sv, &[0.285], sg, 0.004, (1.5, 5.5));
            bx.add(&orient_outward(sweep(&pp, &rrect(0.005, 0.016, 0.002, 1), Mat::Chrome, &SweepOpts::default())));
        }
        for side in [1.0, -1.0] {
            let f = |v: f64| if side > 0.0 { v } else { vm - v };
            let drip: Vec<DVec2> = [0.38, 0.066, -0.30, -0.62, -0.86].iter().map(|&z| DVec2::new(z, f(7.25))).collect();
            bx.add(&p::molding(tube, &drip, 0.003, (0.008, 0.005), false, 4));
        }
        let zv = |a: &[(f64, f64)]| a.iter().map(|&(z, v)| DVec2::new(z, v)).collect::<Vec<_>>();
        let ws = zv(&[(0.42, 8.0), (0.255, 8.0), (0.09, 8.0), (0.12, 9.0), (0.09, 10.0), (0.255, 10.0), (0.42, 10.0), (0.47, 9.0)]);
        bx.add(&p::molding(tube, &ws, 0.002, (0.010, 0.005), true, 6));
        let rw = zv(&[(-0.89, 8.0), (-1.33, 8.0), (-1.742, 8.0), (-1.742, 9.0), (-1.742, 10.0), (-1.33, 10.0), (-0.89, 10.0), (-0.90, 9.0)]);
        bx.add(&p::molding(tube, &rw, 0.002, (0.010, 0.005), true, 6));
        let zt = rear_z(0.0, 0.735);
        let mut cap = closed_lathe(&[(0.0, 0.012), (0.045, 0.010), (0.058, 0.0), (0.056, -0.004), (0.0, -0.004)], 40, Mat::Chrome);
        cap.transform(Some(rot_y(std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 0.735, zt - 0.004));
        bx.add(&cap);
        extras.extend("body_shell", &bx);

        // bright lip mouldings round the wheel openings (fender front, body rear)
        for (arch, owner) in [(ARCH_F, "fender_F"), (ARCH_R, "body_shell")] {
            let (zc, yc, r) = arch;
            let ra = r + 0.004;
            for (s, sg) in sides {
                let lo_y = if owner == "fender_F" { 0.30 } else { 0.26 };
                let a0 = ((lo_y - yc) / ra).clamp(-1.0, 1.0).asin();
                let angs = linspace(a0, std::f64::consts::PI - a0, 60);
                let zs_: Vec<f64> = angs.iter().map(|a| zc + ra * a.cos()).collect();
                let ys_: Vec<f64> = angs.iter().map(|a| yc + ra * a.sin()).collect();
                let (pp, nn) = p::path_on_surface(tube, &zs_, &ys_, sg, 0.004, (1.5, 5.5));
                let mold = orient_outward(sweep(&pp, &rrect(0.004, 0.012, 0.0018, 1), Mat::Chrome, &SweepOpts { twist_up: Some(nn), ..Default::default() }));
                let key = if owner == "fender_F" { format!("fender_F{s}") } else { owner.to_string() };
                extras.extend(&key, &mold);
            }
        }

        // bonnet: scoop and pins
        let mut hx = Mesh::new();
        let nr = 12;
        let rings: Vec<Vec<DVec3>> = linspace(0.0, 1.0, nr)
            .into_iter()
            .map(|t: f64| {
                let z = 1.10 + 0.32 * t;
                let hgt = 0.010 + 0.042 * t.powf(1.2);
                let wdt = 0.15 + 0.025 * t;
                let yb_ = tube.eval(z, 9.0).y + 0.010;
                let mut ring: Vec<DVec3> = linspace(0.0, std::f64::consts::PI, 11).into_iter().map(|a: f64| DVec3::new(wdt * a.cos(), yb_ + hgt * a.sin().max(0.0).powf(0.6), z)).collect();
                ring.push(DVec3::new(-wdt * 0.98, yb_ - 0.006, z));
                ring.push(DVec3::new(wdt * 0.98, yb_ - 0.006, z));
                ring
            })
            .collect();
        let nrg = rings[0].len();
        let gr = Grid2::from_fn(rings.len(), nrg, |i, j| rings[i][nrg - 1 - j]);
        hx.add(&orient_outward(loft(&gr, Mat::TrimBlack, true)));
        hx.add(&box_(DVec3::new(0.0, tube.eval(1.42, 9.0).y + 0.034, 1.424), DVec3::new(0.15, 0.016, 0.002), Mat::InteriorPlastic, None));
        for sg in [1.0, -1.0] {
            let z = 2.08;
            let mut v = tube.v_at_x(z, 0.47, 6.0, 9.0);
            if sg < 0.0 {
                v = vm - v;
            }
            let pp = tube.eval(z, v);
            let mut pin = closed_lathe(&[(0.0, 0.016), (0.010, 0.016), (0.012, 0.010), (0.012, -0.004), (0.0, -0.004)], 16, Mat::Chrome);
            for q in &mut pin.v {
                *q = DVec3::new(q.y, q.x, q.z);
            }
            pin.flip();
            pin.transform(None, pp + DVec3::new(0.0, 0.001, 0.0));
            hx.add(&orient_outward(pin));
        }
        extras.extend("hood", &hx);

        // the rear wing: two black uprights on the lid, a cambered blade
        if self.p.wing {
            let mut wx = Mesh::new();
            let zw = -2.27;
            for sg in [1.0, -1.0] {
                let mut v = tube.v_at_x(zw, 0.72, 6.0, 9.0);
                if sg < 0.0 {
                    v = vm - v;
                }
                let q = tube.eval(zw, v);
                let mut upr = rbox(DVec3::ZERO, DVec3::new(0.010, 0.030, 0.045), 0.007, 2, Mat::TrimBlack);
                upr.transform(Some(rot_x(8.0f64.to_radians())), q + DVec3::new(0.0, 0.032, -0.005));
                wx.add(&upr);
            }
            let yb_ = tube.eval(zw, 9.0).y + 0.068;
            let blade: Vec<DVec3> = linspace(-0.80, 0.80, 33)
                .into_iter()
                .map(|x: f64| {
                    let droop = 0.030 * ((x.abs() - 0.62) / 0.18).max(0.0).powi(2);
                    DVec3::new(x, yb_ + 0.006 * (1.0 - (x / 0.8).powi(2)) - droop, zw - 0.005)
                })
                .collect();
            let sec = [DVec2::new(0.065, 0.004), DVec2::new(0.030, 0.012), DVec2::new(-0.040, 0.010), DVec2::new(-0.065, -0.004), DVec2::new(-0.040, -0.006), DVec2::new(0.030, -0.004)];
            wx.add(&orient_outward(sweep(&blade, &sec, Mat::TrimBlack, &SweepOpts::default())));
            extras.extend("trunk_lid", &wx);
        }

        // interior
        let hf = 0.255;
        for (s, sg) in sides {
            // (solid build: the seats, the dash and the console are parts on
            // the structure: `mechanical`)
            if !solid {
                out.insert(&format!("seat_F{s}"), p::bucket_seat(sg * 0.37, -0.34, hf, Mat::InteriorFabric, true));
            }
            marker(format!("seat_H_point_F{s}"), DVec3::new(sg * 0.37, hf + 0.155, -0.47), DVec3::X);
        }
        if !solid {
            let mut rs = Mesh::new();
            rs.add(&rbox(DVec3::new(0.0, 0.40, -1.00), DVec3::new(0.50, 0.065, 0.20), 0.05, 3, Mat::InteriorFabric));
            let mut back = rbox(DVec3::ZERO, DVec3::new(0.50, 0.22, 0.05), 0.05, 3, Mat::InteriorFabric);
            back.transform(Some(rot_x((-38.0f64).to_radians())), DVec3::new(0.0, 0.67, -1.28));
            rs.add(&back);
            out.insert("seat_rear", rs);
            let mut dash = Mesh::new();
            dash.add(&rbox(DVec3::new(0.0, 0.735, 0.40), DVec3::new(0.68, 0.115, 0.105), 0.05, 3, Mat::InteriorPlastic));
            for sg in [1.0, -1.0] {
                dash.add(&rbox(DVec3::new(sg * 0.37, 0.86, 0.36), DVec3::new(0.20, 0.035, 0.07), 0.03, 2, Mat::InteriorPlastic));
            }
            for k in 0..2 {
                let mut g_ = closed_lathe(&[(0.0, 0.002), (0.040, 0.002), (0.040, -0.004), (0.0, -0.004)], 24, Mat::Chrome);
                g_.transform(Some(rot_y(-std::f64::consts::FRAC_PI_2)), DVec3::new(0.37 + (k as f64 - 0.5) * 0.10, 0.80, 0.302));
                dash.add(&g_);
            }
            out.insert("dash", dash);
            out.insert("console", rbox(DVec3::new(0.0, 0.405, -0.10), DVec3::new(0.09, 0.10, 0.40), 0.03, 2, Mat::InteriorPlastic));
        }
        out.insert("steering_wheel", p::steering_wheel(DVec3::new(0.37, 0.80, 0.11), 0.19, 24.0, Mat::InteriorPlastic, if solid { 0.19 } else { 0.14 }));
        for s in ["FL", "FR"] {
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
        let zz = linspace(0.06, -0.84, 46);
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
            let mut eb = Mesh::new();
            eb.add(&rbox(DVec3::new(0.0, 0.43, 1.30), DVec3::new(0.21, 0.15, 0.30), 0.03, 2, Mat::MetalBare));
            for sg in [1.0f64, -1.0] {
                let rot = glam::DMat3::from_cols(DVec3::new((sg * 0.6).cos(), (sg * 0.6).sin(), 0.0), DVec3::new(-(sg * 0.6).sin(), (sg * 0.6).cos(), 0.0), DVec3::Z);
                let mut hd = rbox(DVec3::ZERO, DVec3::new(0.07, 0.06, 0.29), 0.02, 2, Mat::MetalBare);
                hd.transform(Some(rot), DVec3::new(sg * 0.22, 0.60, 1.30));
                eb.add(&hd);
                let mut vc = rbox(DVec3::ZERO, DVec3::new(0.05, 0.025, 0.27), 0.015, 2, Mat::Chrome);
                vc.transform(Some(rot), DVec3::new(sg * 0.26, 0.665, 1.30));
                eb.add(&vc);
            }
            eb.add(&rbox(DVec3::new(0.0, 0.61, 1.30), DVec3::new(0.14, 0.05, 0.24), 0.02, 2, Mat::MetalBare));
            let mut ac = closed_lathe(&[(0.0, 0.05), (0.18, 0.05), (0.19, 0.03), (0.19, -0.03), (0.18, -0.04), (0.0, -0.04)], 48, Mat::Chrome);
            for q in &mut ac.v {
                *q = DVec3::new(q.y, q.x, q.z);
            }
            ac.flip();
            ac.transform(None, DVec3::new(0.0, 0.715, 1.30));
            eb.add(&orient_outward(ac));
            eb.add(&rbox(DVec3::new(0.0, 0.25, 1.30), DVec3::new(0.17, 0.05, 0.27), 0.02, 2, Mat::MetalBare));
            let mut fan = closed_lathe(&[(0.0, 0.01), (0.17, 0.01), (0.17, -0.01), (0.0, -0.01)], 32, Mat::TrimBlack);
            fan.transform(Some(rot_y(-std::f64::consts::FRAC_PI_2)), DVec3::new(0.0, 0.48, 1.66));
            eb.add(&fan);
            out.insert("engine_block", eb);
            let mut rad = Mesh::new();
            rad.add(&box_(DVec3::new(0.0, 0.56, 2.06), DVec3::new(0.37, 0.17, 0.028), Mat::TrimBlack, None));
            rad.add(&box_(DVec3::new(0.0, 0.745, 2.06), DVec3::new(0.385, 0.016, 0.034), Mat::MetalBare, None));
            rad.add(&box_(DVec3::new(0.0, 0.375, 2.06), DVec3::new(0.385, 0.016, 0.034), Mat::MetalBare, None));
            out.insert("radiator", rad);
            out.insert("firewall", box_(DVec3::new(0.0, 0.57, 0.625), DVec3::new(0.60, 0.27, 0.003), Mat::TrimBlack, None));
            let mut ub = Mesh::new();
            ub.add(&box_(DVec3::new(0.0, 0.165, 1.56), DVec3::new(0.44, 0.003, 0.41), Mat::TrimBlack, None));
            ub.add(&box_(DVec3::new(0.0, 0.375, -2.06), DVec3::new(0.48, 0.004, 0.27), Mat::TrimBlack, None));
            ub.add(&rbox(DVec3::new(0.0, 0.43, -1.98), DVec3::new(0.42, 0.05, 0.20), 0.02, 2, Mat::MetalBare));
            let mut ax = closed_lathe(&[(0.0, 0.64), (0.045, 0.64), (0.045, -0.64), (0.0, -0.64)], 20, Mat::MetalBare);
            ax.transform(None, DVec3::new(0.0, WHEEL_Y, ZR_AX));
            ub.add(&ax);
            ub.add(&rbox(DVec3::new(0.0, WHEEL_Y, ZR_AX + 0.02), DVec3::new(0.11, 0.10, 0.08), 0.05, 3, Mat::MetalBare));
            ub.add(&p::tube_pipe(&[DVec3::new(0.0, 0.27, 0.90), DVec3::new(0.0, 0.255, -0.40), DVec3::new(0.0, 0.27, -1.24)], 0.033, Mat::MetalBare, 2, 14));
            out.insert("underbody", ub);
            let mut ex = Mesh::new();
            for sg in [1.0f64, -1.0] {
                let pts: Vec<DVec3> = [(0.30, 0.34, 1.32), (0.33, 0.25, 1.13), (0.31, 0.181, 0.98), (0.30, 0.178, 0.30), (0.30, 0.175, -0.60), (0.36, 0.185, -1.15), (0.40, 0.20, -1.45), (0.44, 0.22, -1.85), (0.50, 0.30, -2.10), (0.57, 0.415, -2.25), (0.58, 0.425, -2.355)]
                    .iter()
                    .map(|&(x, y, z)| DVec3::new(sg * x, y, z))
                    .collect();
                ex.add(&p::tube_pipe(&pts, 0.026, Mat::MetalBare, 5, 14));
                ex.add(&rbox(DVec3::new(sg * 0.30, 0.177, -0.85), DVec3::new(0.10, 0.026, 0.22), 0.02, 2, Mat::MetalBare));
                // the pipes end behind the rear valance; a bright bezel and a
                // dark opening on its face mark the cut-outs they leave through
                let ce = DVec3::new(sg * 0.58, 0.425, rear_z(0.58, 0.425));
                let g = p::graph_normal(&rear_z, 0.58, 0.425, -1.0);
                let ne = DVec3::new(g.x * sg, g.y, g.z);
                let mut ring = closed_lathe(&[(0.031, 0.0), (0.041, 0.0), (0.041, 0.003), (0.031, 0.003)], 32, Mat::Chrome);
                let mut hole = closed_lathe(&[(0.002, 0.0), (0.0305, 0.0), (0.0305, 0.0015), (0.002, 0.0015)], 32, Mat::TrimBlack);
                for m_ in [&mut ring, &mut hole] {
                    m_.transform(Some(p::rot_to(ne)), ce + ne * 0.0015);
                    ex.add(m_);
                }
            }
            out.insert("exhaust", ex);
        }
        for (tag, zc, r) in [("F", ZF_AX, 0.360), ("R", ZR_AX, 0.372)] {
            for (s, sg) in sides {
                // (solid build: the inner wall's opening lets the arms through)
                out.insert(&format!("wheel_well_{tag}{s}"), thick(p::wheel_well_with(DVec3::new(0.0, WHEEL_Y, zc), r, 0.535, 0.835, sg, 92.0, 92.0, if solid { 0.74 } else { 0.55 }), 0.004));
            }
        }

        // markers: closures
        for (s, sg) in sides {
            for (nm, y) in [("upper", 0.76), ("lower", 0.42)] {
                let (pp, nn) = p::surface_point(tube, 0.565, y, sg, (1.5, 5.5));
                marker(format!("hinge_door_F{s}_{nm}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            }
            let (pp, nn) = p::surface_point(tube, -0.700, 0.62, sg, (1.5, 5.5));
            marker(format!("latch_door_F{s}"), pp - nn * 0.035, DVec3::new(sg, 0.0, 0.0));
            marker(format!("hinge_hood_{s}"), DVec3::new(sg * 0.68, 0.860, 0.60), DVec3::X);
            marker(format!("hinge_trunk_{s}"), DVec3::new(sg * 0.55, 0.925, -1.72), DVec3::X);
            for (k, (z, y)) in [(2.10, 0.79), (1.70, 0.82), (1.10, 0.85), (0.70, 0.87)].iter().enumerate() {
                let x = if *z < 2.0 { 0.78 } else { 0.70 };
                marker(format!("bolt_fender_F{s}_{}", k + 1), DVec3::new(sg * x, y - 0.03, *z), DVec3::X);
            }
        }
        marker("latch_hood".into(), DVec3::new(0.0, 0.775, 2.25), DVec3::X);
        marker("latch_trunk".into(), DVec3::new(0.0, 0.905, -2.39), DVec3::X);
        (out, extras, mk)
    }
}
