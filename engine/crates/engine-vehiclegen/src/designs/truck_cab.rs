//! A two-axle (4x2) cab-over tractor unit: a ladder frame of two C-channel
//! side members, a cab on four mounts, the engine between the rails under it,
//! a front underrun protection beam, a fuel tank and a battery/aftertreatment
//! box on the frame's sides, a fifth wheel, a steered beam axle in front and a
//! drive axle on twin tyres behind. Every part is defined once in
//! [`crate::heavy`] (its nodes, masses, beams, mounts and its solid); the
//! rigger's parts-only path rigs it.
//!
//! **The class and its figures** (`engine/docs/generated-vehicles-lessons.md`
//! §22, with sources): a European 4x2 tractor of the Renault Trucks T class
//! (its personalised specification sheet, model code NLT_10XA036STD00):
//! wheelbase 3.80 m, front overhang 1.365 m, rear overhang 0.825 m, overall
//! chassis length 5.99 m, frame width 851 mm, side member 266 mm × 8 mm, frame
//! top 0.94–0.96 m, fifth wheel 1.152 m over the ground and 675 mm ahead of the
//! drive axle, track 2.109 / 1.854 m, width over the rear wheels 2.518 m; its
//! kerb weight splits 69.8 % front / 30.2 % rear (5,930 / 2,569 of 8,499 kg,
//! with a high sleeper cab and 90 % fuel). Ours carries a day cab with a
//! standard roof (3.0 m), so its kerb is 7,500 kg at the same split
//! (5,235 / 2,265 kg). Tyres 315/70 R22.5 (1,013 mm) on 9.00 × 22.5 steel
//! wheels. Front underrun lower edge 0.40 m (UN R93: at most 400 mm).
//!
//! glTF frame, the origin midway between the axles on the ground.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::assemble::{AxleWheel, Design, RigSpec};
use crate::carbody::{CarBody, Side};
use crate::export::Marker;
use crate::heavy::{self, BeamAxle, Cab, LadderFrame, Underrun};
use crate::mechanical::{vals, MechPart};
use crate::mesh::{box_, Mat, Mesh};
use crate::physics::Physics;
use crate::structure::Structure;
use crate::tube::{Handle, Key, Tube};
use crate::uvpack::Parts;

pub const WHEELBASE: f64 = 3.80;
pub const Z_FA: f64 = WHEELBASE / 2.0;
pub const Z_RA: f64 = -WHEELBASE / 2.0;
pub const FRONT_OVERHANG: f64 = 1.365;
pub const REAR_OVERHANG: f64 = 0.825;
pub const TRACK_F: f64 = 2.109;
pub const TRACK_R: f64 = 1.854;
/// 315/70 R22.5: 571.5 + 2 × 0.70 × 315 mm.
pub const TYRE_D: f64 = 1.0125;
pub const TYRE_W: f64 = 0.315;
pub const RIM_D: f64 = 22.5 * 0.0254;
pub const RIM_W: f64 = 9.0 * 0.0254;
pub const WHEEL_Y: f64 = TYRE_D / 2.0;
/// Frame: outer width 851 mm, side member 266 × 90 × 8 mm (the flange's 90 mm
/// is typical; the sheet gives height and gauge), top 0.95 m.
pub const FRAME_HALF_W: f64 = 0.4255;
pub const FRAME_TOP: f64 = 0.95;
/// Cab: a day cab with a standard roof.
pub const CAB_FRONT: f64 = Z_FA + 1.30;
pub const CAB_REAR: f64 = Z_FA - 1.00;
pub const CAB_FLOOR: f64 = 1.15;
pub const CAB_ROOF: f64 = 3.00;
pub const CAB_HALF_W: f64 = 1.25;
/// Mass budget (kg): the parts as defined. Sum 7,500.
pub const KG_CAB: f64 = 1400.0;
pub const KG_ENGINE: f64 = 2000.0;
pub const KG_AXLE_F: f64 = 600.0;
pub const KG_AXLE_R: f64 = 1000.0;
pub const KG_WHEEL_F: f64 = 105.0;
pub const KG_WHEEL_R: f64 = 200.0;
pub const KG_FRAME: f64 = 840.0;
pub const KG_BUMPER: f64 = 120.0;
pub const KG_FUEL: f64 = 350.0;
pub const KG_BOX: f64 = 400.0;
pub const KG_FIFTH: f64 = 180.0;
/// The design's kerb (kg) and its published front-axle share.
pub const KERB: f64 = 7500.0;
pub const FRONT_SHARE: f64 = 5930.0 / 8499.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TruckCabParams {
    /// linear RGB
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// Ballast on the fifth wheel (kg): a crash scenario's stated gross
    /// (0: kerb).
    pub ballast_kg: f64,
    /// The crash physics block (`kerb` is the design's kerb plus the ballast).
    pub physics: Physics,
}

impl Default for TruckCabParams {
    fn default() -> Self {
        let physics = Physics {
            kerb: KERB as f32,
            drag_cd: 0.60,
            ..Physics::default()
        };
        TruckCabParams {
            paint: [0.55, 0.03, 0.02],
            metallic: 0.2,
            roughness: 0.35,
            ballast_kg: 0.0,
            physics,
        }
    }
}

pub const RANGES: &[(&str, f64, f64)] = &[
    ("paint", 0.0, 1.0),
    ("metallic", 0.0, 1.0),
    ("roughness", 0.0, 1.0),
    ("ballast_kg", 0.0, 12000.0),
];
pub const RIM_STYLES: &[&str] = &["steel"];

impl TruckCabParams {
    pub fn validate(&self) -> Result<(), String> {
        let chk = |name: &str, v: f64, hi: f64| {
            if v.is_finite() && (0.0..=hi).contains(&v) {
                Ok(())
            } else {
                Err(format!("{name} = {v} is outside 0..{hi}"))
            }
        };
        chk("metallic", self.metallic, 1.0)?;
        chk("roughness", self.roughness, 1.0)?;
        chk("ballast_kg", self.ballast_kg, 12000.0)?;
        for (k, c) in self.paint.iter().enumerate() {
            chk(&format!("paint[{k}]"), *c, 1.0)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct TruckCab {
    pub p: TruckCabParams,
}

impl TruckCab {
    pub fn new(p: TruckCabParams) -> Self {
        TruckCab { p }
    }

    pub fn frame(&self) -> LadderFrame {
        let steel = LadderFrame {
            // (40 mm behind the underrun beam's back: outside each other's
            // contact range)
            z_front: Z_FA + 1.075,
            z_rear: Z_RA - REAR_OVERHANG,
            bays: 6,
            half_width: FRAME_HALF_W,
            web_h: 0.266,
            flange_w: 0.090,
            gauge: 0.008,
            top_y: FRAME_TOP,
            // S500MC-class frame steel
            yield_pa: 500e6,
            ultimate_pa: 600e6,
            cross_at: vec![0, 3, 4, 5, 6],
            cross_kg: 25.0,
            equipment_kg: 0.0,
        };
        let equipment = KG_FRAME - steel.mass();
        assert!(
            equipment > 0.0,
            "the frame's steel ({:.0} kg) is over its budget",
            steel.mass()
        );
        LadderFrame {
            equipment_kg: equipment,
            ..steel
        }
    }

    pub fn axle(&self, front: bool) -> BeamAxle {
        BeamAxle {
            tag: if front { "F" } else { "R" },
            z: if front { Z_FA } else { Z_RA },
            wheel_y: WHEEL_Y,
            track: if front { TRACK_F } else { TRACK_R },
            seat_x: 0.38,
            kg: if front { KG_AXLE_F } else { KG_AXLE_R },
            hanger: 0.95,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: RIM_D,
            rim_w: RIM_W,
            dual: !front,
            wheel_kg: if front { KG_WHEEL_F } else { KG_WHEEL_R },
            // (the lint's and the loader's raycast wheel)
            raycast_kg: 20.0,
            // a forged axle beam; spring hangers and shackles
            axle: (400e3, 800e3),
            hangers: (150e3, 300e3),
        }
    }

    pub fn cab_spec(&self) -> Cab {
        Cab {
            z_front: CAB_FRONT,
            z_rear: CAB_REAR,
            z_pillar: Z_FA - 0.40,
            floor_y: CAB_FLOOR,
            belt_y: 2.05,
            roof_y: CAB_ROOF,
            half_w: CAB_HALF_W,
            radius: 0.14,
            kg: KG_CAB,
            mount_x: 0.38,
            // the cab gives way before the frame (its chords: 430 kN)
            members: (60e3, 150e3),
            floor: (120e3, 250e3),
        }
    }

    pub fn underrun(&self) -> Underrun {
        Underrun {
            z_front: Z_FA + FRONT_OVERHANG,
            depth: 0.25,
            lower_y: 0.40,
            upper_y: 1.10,
            half_w: 1.22,
            kg: KG_BUMPER,
            beam: (150e3, 300e3),
            // R93's 160 kN test load at a rail's line holds elastically on 8
            // bracket beams of 40 kN
            bracket: (40e3, 80e3),
            bracket_beams: 8,
        }
    }

    /// Every part (definitions), and the meshes that ride them.
    pub fn assembly(&self) -> (Vec<MechPart>, Vec<(String, Mesh)>) {
        let paint = Mat::Paint;
        let f = self.frame();
        let mut parts = heavy::ladder_frame(&f);
        let mut extra: Vec<(String, Mesh)> = Vec::new();
        // the rear mudguards and tail lamps ride the cross-members
        {
            let xm = parts
                .iter_mut()
                .find(|p| p.name == "frame_crossmembers")
                .unwrap();
            for s in [1.0f64, -1.0] {
                let sd = if s > 0.0 { "L" } else { "R" };
                let mut mg = box_(
                    DVec3::new(s * 0.93, TYRE_D + 0.08, Z_RA),
                    DVec3::new(0.34, 0.008, 0.62),
                    Mat::TrimBlack,
                    None,
                );
                mg.add(&box_(
                    DVec3::new(s * 0.93, TYRE_D - 0.10, Z_RA - 0.615),
                    DVec3::new(0.34, 0.18, 0.008),
                    Mat::TrimBlack,
                    None,
                ));
                extra.push((format!("mudguard_R{sd}"), mg));
                let zt = f.z_rear - 0.03;
                extra.push((
                    format!("taillight_{sd}"),
                    box_(
                        DVec3::new(s * 0.75, 0.80, zt),
                        DVec3::new(0.18, 0.06, 0.012),
                        Mat::LampLens,
                        None,
                    ),
                ));
                xm.meshes.push(format!("mudguard_R{sd}"));
                xm.meshes.push(format!("taillight_{sd}"));
            }
        }
        for front in [true, false] {
            let a = self.axle(front);
            parts.extend(heavy::beam_axle(&a, &f));
            extra.extend(heavy::axle_meshes(&a, &f));
        }
        let (cab, cab_extra) = heavy::cab(&self.cab_spec(), paint);
        extra.extend(cab_extra);
        let mut cab = cab;
        for (s, z0, z1) in [
            (1.0f64, CAB_FRONT - 0.70, CAB_FRONT),
            (-1.0, CAB_FRONT - 0.70, CAB_FRONT),
            (1.0, CAB_REAR, CAB_REAR + 0.65),
            (-1.0, CAB_REAR, CAB_REAR + 0.65),
        ] {
            let k = if z1 >= CAB_FRONT { 0 } else { 2 };
            let sd = if s > 0.0 { "L" } else { "R" };
            let from = vec![format!("cab{k}i{sd}"), format!("cab{k}f{sd}")];
            let to = heavy::rail_nodes(&f, s, &["ot", "it"], z0 - 0.6, z1);
            let fr = if k == 0 { "F" } else { "R" };
            heavy::mount(
                &mut cab,
                &format!("cab_mount_{fr}{sd}"),
                from,
                to,
                6,
                vals(0.0, 0.0, 60e3, 120e3),
            );
        }
        parts.push(cab);
        let (bumper, bumper_extra) = heavy::front_underrun(&self.underrun(), &f, paint);
        parts.push(bumper);
        extra.extend(bumper_extra);
        // the engine with its gearbox and cooling module, on four mounts
        let rigid = vals(0.0, 0.0, f64::INFINITY, 2.0e6);
        let mut eng = heavy::block(
            "engine",
            "engine",
            DVec3::new(-0.30, 0.58, Z_FA - 1.05),
            DVec3::new(0.30, 1.12, Z_FA + 0.65),
            KG_ENGINE,
            0.02,
            Mat::MetalBare,
            rigid,
        );
        for (s, zf, z0, z1) in [
            (1.0f64, 1, Z_FA + 0.10, Z_FA + 1.00),
            (-1.0, 1, Z_FA + 0.10, Z_FA + 1.00),
            (1.0, 0, Z_FA - 1.50, Z_FA - 0.60),
            (-1.0, 0, Z_FA - 1.50, Z_FA - 0.60),
        ] {
            let k = if s > 0.0 { 1 } else { 0 };
            let from = vec![format!("eng{k}0{zf}"), format!("eng{k}1{zf}")];
            let to = heavy::rail_nodes(&f, s, &["it", "ib"], z0, z1);
            let fr = if zf == 1 { "F" } else { "R" };
            let sd = if s > 0.0 { "L" } else { "R" };
            heavy::mount(
                &mut eng,
                &format!("engine_mount_{fr}{sd}"),
                from,
                to,
                4,
                vals(0.0, 0.0, 80e3, 160e3),
            );
        }
        parts.push(eng);
        // the fuel tank (aluminium, D-profile 710 mm, 405 L at 90 %) on the
        // left rail; batteries, AdBlue and the silencer box on the right
        for (name, s, kg, mat) in [
            ("fuel_tank", 1.0f64, KG_FUEL, Mat::Chrome),
            ("battery_box", -1.0, KG_BOX, Mat::TrimBlack),
        ] {
            let (x0, x1) = (0.47, 1.17);
            // (its nodes on two of the frame's stations: its brackets in their planes)
            let st = f.stations();
            let lo = DVec3::new(if s > 0.0 { x0 } else { -x1 }, 0.42, st[3] - 0.02);
            let hi = DVec3::new(if s > 0.0 { x1 } else { -x0 }, 1.06, st[2] + 0.02);
            let mut t = heavy::block(
                name,
                "fuel",
                lo,
                hi,
                kg,
                0.02,
                mat,
                vals(0.0, 0.0, 200e3, 400e3),
            );
            let tag = heavy::short_tag(name);
            let k = if s > 0.0 { 0 } else { 1 };
            // the inner face on the rail's web, and the brackets under the
            // tank from its outer foot to the rail's flanges (a cantilever
            // held in depth, not a hinge on one plane)
            let ko = 1 - k;
            let from = (0..4)
                .map(|q| format!("{tag}{k}{}{}", q / 2, q % 2))
                .chain((0..4).map(|q| format!("{tag}{ko}{}{}", q / 2, q % 2)))
                .collect();
            let to =
                heavy::rail_nodes(&f, s, &["ot", "it", "ib", "ob"], st[3] - 0.01, st[2] + 0.01);
            heavy::mount(
                &mut t,
                &format!("{name}_straps"),
                from,
                to,
                16,
                vals(0.0, 0.0, 30e3, 60e3),
            );
            // the brackets' fore-and-aft legs: the inner face's front corners
            // to the rail at the rear station and the rear ones to the front
            // (in the stations' planes alone the tank slid along the rail at
            // 2 Hz)
            for (i, sz) in [(1, 3), (0, 2)] {
                let from = (0..2).map(|j| format!("{tag}{k}{j}{i}")).collect();
                let to = heavy::rail_nodes(
                    &f,
                    s,
                    &["ot", "it", "ib", "ob"],
                    st[sz] - 0.01,
                    st[sz] + 0.01,
                );
                heavy::mount(
                    &mut t,
                    &format!("{name}_bracket_{}", if i == 1 { "f" } else { "r" }),
                    from,
                    to,
                    4,
                    vals(0.0, 0.0, 30e3, 60e3),
                );
            }
            parts.push(t);
        }
        // the fifth wheel (its top at 1.152 m, 675 mm ahead of the drive axle)
        let zc = Z_RA + 0.675;
        let mut fw = heavy::block(
            "fifth_wheel",
            "coupling",
            DVec3::new(-0.45, 0.975, zc - 0.45),
            DVec3::new(0.45, 1.152, zc + 0.45),
            KG_FIFTH + self.p.ballast_kg,
            0.015,
            Mat::TrimBlack,
            vals(0.0, 0.0, 300e3, 600e3),
        );
        for s in [1.0f64, -1.0] {
            let k = if s > 0.0 { 1 } else { 0 };
            let from = (0..4)
                .map(|q| format!("fifw{k}{}{}", q / 2, q % 2))
                .collect::<Vec<_>>();
            let to = heavy::rail_nodes(&f, s, &["ot", "it"], zc - 0.9, zc + 0.9);
            let sd = if s > 0.0 { "L" } else { "R" };
            heavy::mount(
                &mut fw,
                &format!("fifth_wheel_bolts_{sd}"),
                from,
                to,
                8,
                vals(0.0, 0.0, 100e3, 250e3),
            );
        }
        parts.push(fw);
        (parts, extra)
    }

    fn markers(&self) -> Vec<Marker> {
        let mut out = Vec::new();
        for (tag, z, track) in [("F", Z_FA, TRACK_F), ("R", Z_RA, TRACK_R)] {
            for s in [1.0f64, -1.0] {
                let sd = if s > 0.0 { "L" } else { "R" };
                out.push(Marker {
                    name: format!("wheel_center_{tag}{sd}"),
                    pos: DVec3::new(s * 0.5 * track, WHEEL_Y, z),
                    xaxis: DVec3::X * s,
                });
            }
        }
        let f = self.frame();
        let u = self.underrun();
        for s in [1.0f64, -1.0] {
            let sd = if s > 0.0 { "L" } else { "R" };
            out.push(Marker {
                name: format!("bolt_fup_{sd}"),
                pos: DVec3::new(
                    s * FRAME_HALF_W,
                    f.bottom_y() + 0.1,
                    f.z_front + 0.5 * (u.z_front - u.depth - f.z_front),
                ),
                xaxis: DVec3::X,
            });
        }
        out
    }
}

/// The cab's section as a tube (the rig data's key lines: sill, door bottom,
/// belt, roof rail …). No panel is cut from it: every part is a definition.
fn cab_tube() -> Tube {
    let w = CAB_HALF_W;
    let keys = vec![
        Key::consts(0.0, CAB_FLOOR),
        Key::consts(w - 0.05, CAB_FLOOR + 0.02),
        Key::consts(w, CAB_FLOOR + 0.15),
        Key::consts(w, 1.60),
        Key::consts(w, 1.85),
        Key::consts(w, 2.00),
        Key::consts(w, 2.05),
        Key::consts(w - 0.05, CAB_ROOF - 0.05),
        Key::consts(w - 0.25, CAB_ROOF),
        Key::consts(0.0, CAB_ROOF),
    ];
    let h = |a: f64, b: f64| (Handle::C(a), Handle::C(b));
    let handles = (0..keys.len()).map(|_| h(1.0 / 3.0, 1.0 / 3.0)).collect();
    Tube::new(keys, handles, vec![])
}

impl Design for TruckCab {
    fn name(&self) -> String {
        "truck_cab".into()
    }
    fn paint(&self) -> ([f64; 3], f64, f64) {
        (self.p.paint, self.p.metallic, self.p.roughness)
    }
    fn tube(&self) -> Tube {
        cab_tube()
    }
    fn zlines(&self) -> Vec<(&'static str, f64)> {
        vec![("CR", CAB_REAR), ("CF", CAB_FRONT)]
    }
    fn vlines(&self) -> Vec<f64> {
        vec![0.0, 3.0, 6.0, 9.0]
    }
    fn assign(
        &self,
        _zf: &str,
        _zt: &str,
        _lo: usize,
        _hi: usize,
        _side: Side,
    ) -> Option<(String, Mat)> {
        None
    }
    fn build_layout(&self, _cb: &mut CarBody) {}
    fn build_caps(&self, _cb: &mut CarBody) {}
    fn parts(&self, _cb: &CarBody) -> (Parts, Parts, Vec<Marker>) {
        let (defs, extra) = self.assembly();
        let mut out = Parts::default();
        for p in &defs {
            if !p.mesh.v.is_empty() {
                out.insert(&p.name, p.mesh.clone());
            }
        }
        for (n, m) in extra {
            out.insert(&n, m);
        }
        (out, Parts::default(), self.markers())
    }
    fn rig_spec(&self) -> Option<RigSpec> {
        let mut physics = self.p.physics.clone();
        physics.kerb += self.p.ballast_kg as f32;
        let a = self.axle(false);
        let (cf, cr) = (CAB_FRONT, Z_RA - REAR_OVERHANG);
        Some(RigSpec {
            wheelbase: WHEELBASE,
            track: TRACK_F,
            axle_f: Z_FA,
            axle_r: Z_RA,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: RIM_D,
            wheel_y: WHEEL_Y,
            rear_wheel: Some(AxleWheel {
                wheel_y: WHEEL_Y,
                tyre_d: TYRE_D,
                tyre_w: a.contact_w(),
                rim_d: RIM_D,
            }),
            pose: None,
            z_front: Z_FA + FRONT_OVERHANG,
            z_rear: cr,
            doors: vec![],
            rear_closure: "trunk_lid".into(),
            arches: vec![(Z_FA, WHEEL_Y, 0.56), (Z_RA, WHEEL_Y, 0.56)],
            front_z: std::rc::Rc::new(move |_, _| cf),
            rear_z: std::rc::Rc::new(move |_, _| cr),
            physics,
            surfaces: Vec::new(),
            structure: None,
            mechanical: std::rc::Rc::new(Vec::new()),
        })
    }
    /// No unibody: an empty body-in-white, so the rigger builds the truck from
    /// its parts alone.
    fn structure(&self) -> Option<Structure> {
        Some(Structure::default())
    }
    fn mechanical(&self, _st: &Structure) -> Vec<MechPart> {
        self.assembly().0
    }
    fn solid_style(&self, _name: &str) -> crate::solid::Style {
        crate::solid::Style::Structure
    }
}
