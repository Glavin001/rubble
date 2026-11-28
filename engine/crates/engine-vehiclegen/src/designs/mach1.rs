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
pub const ZF_AX: f64 = WHEELBASE / 2.0;
pub const ZR_AX: f64 = -WHEELBASE / 2.0;
pub const TYRE_D: f64 = 0.662;
const TYRE_W: f64 = 0.235;
pub const WHEEL_Y: f64 = TYRE_D / 2.0;
/// The floor pan's height: the skin's floor line (0.215) and 27 mm, as the sedan's.
const Y_PAN: f64 = 0.242;
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

/// Each numeric parameter's range (for the web configurator; paint channels are 0..1).
pub const RANGES: &[(&str, f64, f64)] = &[("paint", 0.0, 1.0), ("metallic", 0.0, 1.0), ("roughness", 0.0, 1.0)];

/// The rim styles this design accepts.
pub const RIM_STYLES: &[&str] = &["magnum", "alloy", "steel"];

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

/// One wheel of the running gear (the axle's pair alike), in the car's body
/// frame: its centre's height and z, tyre and rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxleWheel {
    pub y: f64,
    pub z: f64,
    pub tyre_d: f64,
    pub tyre_w: f64,
    pub rim_d: f64,
    pub rim_w: f64,
    pub style: RimStyle,
}

/// The two axles' wheels. The stock car's are alike; a variant sets its own
/// (staggered tyres, a raked stance) and every consumer (the wheel meshes,
/// the suspension, the rig data) follows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Running {
    pub front: AxleWheel,
    pub rear: AxleWheel,
}

impl Running {
    /// The 1969 car's: 215 section on 15 x 7 in rims, both axles.
    pub fn stock(p: &Mach1Params) -> Running {
        let style = match p.rim_style.as_str() {
            "steel" => RimStyle::Steel,
            "alloy" => RimStyle::Alloy,
            _ => RimStyle::Magnum,
        };
        let w = AxleWheel { y: WHEEL_Y, z: ZF_AX, tyre_d: TYRE_D, tyre_w: TYRE_W, rim_d: 15.0 * 0.0254, rim_w: 7.0 * 0.0254, style };
        Running { front: w, rear: AxleWheel { z: ZR_AX, ..w } }
    }

    fn uniform(&self) -> bool {
        self.rear == AxleWheel { z: self.rear.z, ..self.front }
    }
}

#[derive(Clone)]
pub struct Mach1 {
    pub p: Mach1Params,
    pub run: Running,
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
        let run = Running::stock(&p);
        Mach1 { p, run }
    }

    /// The same car on other wheels (a variant's running gear).
    pub fn with_running(p: Mach1Params, run: Running) -> Mach1 {
        Mach1 { p, run }
    }

    fn wheel_spec(w: &AxleWheel) -> WheelSpec {
        WheelSpec { track: TRACK, wheel_y: w.y, tyre_d: w.tyre_d, tyre_w: w.tyre_w, rim_d: w.rim_d, rim_w: w.rim_w, rim_style: w.style, disc_d: 0.28 }
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
            z_seat: -0.30,
            z_door_rear: z("AR1"),
            z_rear_top: -0.90,
            z_rear_base: z("AR3"),
            well_x: 0.535,
            well_r: 0.366,
            wheel_y: WHEEL_Y,
            y_pan: Y_PAN,
            seat_row: 3,
        }
    }

    /// The running gear's dimensions for [`mechanical::sedan_parts`].
    pub fn gear(&self) -> mechanical::Gear {
        mechanical::Gear {
            f: Frame::identity(),
            half_track: TRACK / 2.0,
            wheel_y: self.run.front.y,
            zf: self.run.front.z,
            zr: self.run.rear.z,
            tyre_d: self.run.front.tyre_d,
            tyre_w: self.run.front.tyre_w,
            rim_d: self.run.front.rim_d,
            rim_w: self.run.front.rim_w,
            axle_f: ZF_AX,
            axle_r: ZR_AX,
            y_pan: Y_PAN,
            seat_row: 3,
            z_nose: Rc::new(front_z),
            z_tail: Rc::new(rear_z_body),
            rear: (!self.run.uniform()).then(|| {
                let r = &self.run.rear;
                mechanical::AxleGear { wheel_y: r.y, tyre_d: r.tyre_d, tyre_w: r.tyre_w, rim_d: r.rim_d, rim_w: r.rim_w }
            }),
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
            tyre_d: self.run.front.tyre_d,
            tyre_w: self.run.front.tyre_w,
            rim_d: self.run.front.rim_d,
            wheel_y: self.run.front.y,
            rear_wheel: (!self.run.uniform()).then(|| {
                let r = &self.run.rear;
                crate::assemble::AxleWheel { wheel_y: r.y, tyre_d: r.tyre_d, tyre_w: r.tyre_w, rim_d: r.rim_d }
            }),
            pose: None,
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

mod parts;
