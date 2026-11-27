//! A 53 ft (16.15 m) tandem-axle dry-van semi-trailer, parked: standing on its landing
//! gear in front and its two axles behind, with a rear impact guard to FMVSS 223/224.
//! Every part is defined once in [`crate::heavy`] (its nodes, masses, beams, mounts and
//! its solid); the rigger's parts-only path rigs it.
//!
//! **The class and its figures** (`engine/docs/generated-vehicles-lessons.md` §24, with
//! sources; *estimate* marks an engineering estimate): the US standard dry van, 53 ft
//! long, 102 in (2.59 m) wide, 13 ft 6 in (4.11 m) high; kingpin 36 in (0.914 m) behind
//! the front; a sliding tandem with a 49 in (1.245 m) spread, set at 40 ft (12.19 m) from
//! the kingpin to the rear axle (the California kingpin-to-rear-axle limit); empty about
//! 14,000 lb (6,350 kg; 53 ft vans run 13,500–15,500 lb); floor 50 in (1.27 m) over the
//! ground (*estimate*, 47–50 in is usual). Tyres 295/75 R22.5 (1,014 mm) on 8.25 × 22.5
//! steel wheels, twin, on a 1.82 m track (*estimate*). Laden to 15 t gross for the crash
//! reference: [`BoxTrailerParams::ballast_kg`] 8,650 kg spread over the deck.
//!
//! The rear impact guard: [`crate::heavy::RearGuard`] (its bar's lower edge 0.52 m over
//! the ground, its ends 75 mm in from the sides, its rear face 0.10 m ahead of the rear
//! extremity; its strengths from the regulation's loads, `heavy/guard.rs`).
//!
//! glTF frame, the origin on the ground midway between the tandem's axles.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::assemble::{AxleWheel, Design, RigSpec};
use crate::carbody::{CarBody, Side};
use crate::crash::Zone;
use crate::export::Marker;
use crate::heavy::{self, BeamAxle, CargoBox, LadderFrame, LandingGear, RearFrame, RearGuard};
use crate::mechanical::{vals, MechPart};
use crate::mesh::{box_, closed_lathe, Mat, Mesh};
use crate::physics::Physics;
use crate::structure::Structure;
use crate::tube::{Handle, Key, Tube};
use crate::uvpack::Parts;

pub const LENGTH: f64 = 53.0 * 0.3048;
pub const HALF_W: f64 = 0.5 * 102.0 * 0.0254;
pub const ROOF_Y: f64 = 13.5 * 0.3048;
/// The tandem's spread and where it stands: 40 ft from the kingpin to the rear axle.
pub const TANDEM: f64 = 49.0 * 0.0254;
pub const Z_FA: f64 = 0.5 * TANDEM;
pub const Z_RA: f64 = -0.5 * TANDEM;
pub const KPRA: f64 = 40.0 * 0.3048;
pub const Z_KING: f64 = Z_RA + KPRA;
pub const Z_FRONT: f64 = Z_KING + 36.0 * 0.0254;
pub const Z_REAR: f64 = Z_FRONT - LENGTH;
/// The deck (the floor's top), and the floor structure's underside: 35 mm of floor on
/// 80 mm crossmembers (*estimate*).
pub const DECK_Y: f64 = 1.27;
pub const FLOOR_Y: f64 = DECK_Y - 0.115;
/// 295/75 R22.5: 571.5 + 2 × 0.75 × 295 mm; 8.25 × 22.5 rims.
pub const TYRE_D: f64 = 1.014;
pub const TYRE_W: f64 = 0.295;
pub const RIM_D: f64 = 22.5 * 0.0254;
pub const RIM_W: f64 = 8.25 * 0.0254;
pub const WHEEL_Y: f64 = TYRE_D / 2.0;
pub const TRACK: f64 = 1.82;
/// The rails under the floor (the slider and the longitudinal beams): 300 × 100 × 6 mm
/// steel (*estimate*), 60 mm under the floor's underside (two skins closer than 5 cm read
/// as a doubled layer).
pub const RAIL_TOP: f64 = FLOOR_Y - 0.06;
pub const RAIL_HALF_W: f64 = 0.45;
/// Mass budget (kg): the parts as defined (*estimates*, summing to the published kerb).
pub const KERB: f64 = 6350.0;
pub const KG_FRAME: f64 = 900.0;
pub const KG_AXLE: f64 = 450.0;
pub const KG_WHEEL: f64 = 175.0;
pub const KG_LANDING: f64 = 200.0;
pub const KG_KINGPIN: f64 = 250.0;
pub const KG_GUARD: f64 = 100.0;
pub const KG_REAR_FRAME: f64 = 350.0;
pub const KG_FLOOR: f64 = 1900.0;
pub const KG_SHELL: f64 = 1050.0;
/// The crash reference's gross (kg).
pub const GROSS: f64 = 15000.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BoxTrailerParams {
    /// linear RGB
    pub paint: [f64; 3],
    pub metallic: f64,
    pub roughness: f64,
    /// Load on the deck (kg), spread over its length: the default lades it to the crash
    /// reference's 15 t gross (0: empty).
    pub ballast_kg: f64,
    /// The crash physics block (`kerb` is the design's kerb plus the ballast).
    pub physics: Physics,
}

impl Default for BoxTrailerParams {
    fn default() -> Self {
        let physics = Physics {
            kerb: KERB as f32,
            drag_cd: 0.70,
            ..Physics::default()
        };
        BoxTrailerParams {
            paint: [0.80, 0.80, 0.78],
            metallic: 0.1,
            roughness: 0.4,
            ballast_kg: GROSS - KERB,
            physics,
        }
    }
}

pub const RANGES: &[(&str, f64, f64)] = &[
    ("paint", 0.0, 1.0),
    ("metallic", 0.0, 1.0),
    ("roughness", 0.0, 1.0),
    ("ballast_kg", 0.0, 24000.0),
];
pub const RIM_STYLES: &[&str] = &["steel"];

impl BoxTrailerParams {
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
        chk("ballast_kg", self.ballast_kg, 24000.0)?;
        for (k, c) in self.paint.iter().enumerate() {
            chk(&format!("paint[{k}]"), *c, 1.0)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct BoxTrailer {
    pub p: BoxTrailerParams,
}

impl BoxTrailer {
    pub fn new(p: BoxTrailerParams) -> Self {
        BoxTrailer { p }
    }

    pub fn frame(&self) -> LadderFrame {
        let steel = LadderFrame {
            // behind the upper coupler, to 5 cm ahead of the rear frame
            z_front: Z_KING - 0.70,
            z_rear: Z_REAR + 0.20,
            bays: 18,
            half_width: RAIL_HALF_W,
            web_h: 0.30,
            flange_w: 0.10,
            gauge: 0.006,
            top_y: RAIL_TOP,
            // ASTM A572 Grade 50-class steel (345 / 450 MPa)
            yield_pa: 345e6,
            ultimate_pa: 450e6,
            cross_at: vec![0, 4, 8, 12, 16, 18],
            cross_kg: 25.0,
            equipment_kg: 0.0,
        };
        let equipment = KG_FRAME - steel.mass();
        assert!(equipment > 0.0, "the frame's steel ({:.0} kg) is over its budget", steel.mass());
        LadderFrame { equipment_kg: equipment, ..steel }
    }

    pub fn axle(&self, front: bool) -> BeamAxle {
        BeamAxle {
            tag: if front { "F" } else { "R" },
            z: if front { Z_FA } else { Z_RA },
            wheel_y: WHEEL_Y,
            track: TRACK,
            seat_x: 0.40,
            kg: KG_AXLE,
            // (reaching two of the rails' stations, 0.80 m apart: on one, the front axle
            // hung as a hinge and drifted 19 mm parked)
            hanger: 0.85,
            tyre_d: TYRE_D,
            tyre_w: TYRE_W,
            rim_d: RIM_D,
            rim_w: RIM_W,
            dual: true,
            wheel_kg: KG_WHEEL,
            raycast_kg: 20.0,
            axle: (400e3, 800e3),
            hangers: (150e3, 300e3),
        }
    }

    pub fn guard(&self) -> RearGuard {
        let f = self.frame();
        RearGuard {
            z_rear: Z_REAR + 0.10,
            lower_y: 0.52,
            half_w: HALF_W - 0.075,
            section: 0.100,
            // the lightest standard 100 mm square tube whose plastic moment carries P1
            // (6.3 mm: M_p 28.7 kN m against P1's 26 kN m; 5 mm: 23.4)
            wall: 0.0063,
            // ASTM A500 Grade C tube (345 / 427 MPa: 50 / 62 ksi)
            yield_pa: 345e6,
            ultimate_pa: 427e6,
            // the supports outboard of the rails, on brackets to them (*estimate*: under
            // the rails, the bar's ends cantilevered 0.8 m and rang in plan at 6.7 Hz,
            // their modal damping 0.5 %, for seconds after the trailer was set down)
            support_x: 0.70,
            post_station: f.bays,
            brace_station: f.bays - 1,
            udl_n: 350e3,
            kg: KG_GUARD,
        }
    }

    pub fn cargo(&self) -> CargoBox {
        CargoBox {
            z_front: Z_FRONT - 0.005,
            z_rear: Z_REAR + 0.20,
            // (0.80 m: the rails' stations, so each seat stands over a rail station)
            bays: 20,
            floor_y: FLOOR_Y,
            roof_y: ROOF_Y,
            half_w: HALF_W - 0.004,
            rail_x: RAIL_HALF_W - 0.05,
            floor_kg: KG_FLOOR,
            load_kg: self.p.ballast_kg,
            shell_kg: KG_SHELL,
            // composite side walls and an aluminium roof on bows (*estimate*)
            members: (80e3, 160e3),
            ties: (60e3, 120e3),
        }
    }

    pub fn rear_frame(&self) -> RearFrame {
        RearFrame {
            z_rear: Z_REAR,
            depth: 0.15,
            // the rear sill: 0.27 m deep under the deck (*estimate*)
            sill_y: DECK_Y - 0.27,
            top_y: ROOF_Y,
            half_w: HALF_W,
            kg: KG_REAR_FRAME,
            members: (200e3, 400e3),
            mounts: (100e3, 200e3),
        }
    }

    pub fn landing(&self) -> LandingGear {
        LandingGear {
            // about 1.7 m behind the kingpin (*estimate*)
            z: Z_KING - 1.70,
            x: 0.60,
            leg: 0.14,
            kg: KG_LANDING,
            // a pair's static rating about 700 kN (*estimate*, the usual 155,000–160,000 lb)
            // (each leg's beams and its brackets' at a leg's share, 350 kN over four)
            legs: (90e3, 180e3),
            brackets: (90e3, 180e3),
            brace: (50e3, 100e3),
            mount_to: [1.0, -1.0].map(|s| self.floor_over(Z_KING - 1.70, s)),
        }
    }

    /// The cargo box's floor nodes on side `s` at the two stations nearest `z`.
    fn floor_over(&self, z: f64, s: f64) -> Vec<String> {
        let st = self.cargo().stations();
        let mut k: Vec<usize> = (0..st.len()).collect();
        k.sort_by(|&a, &b| (st[a] - z).abs().total_cmp(&(st[b] - z).abs()));
        k[..2].iter().flat_map(|&k| ['i', 'f'].map(|c| heavy::box_node(k, c, s))).collect()
    }

    /// Every part (definitions), and the meshes that ride them.
    pub fn assembly(&self) -> (Vec<MechPart>, Vec<(String, Mesh)>) {
        let f = self.frame();
        let mut parts = heavy::ladder_frame(&f);
        let mut extra: Vec<(String, Mesh)> = Vec::new();
        for front in [true, false] {
            let a = self.axle(front);
            let mut ax = heavy::beam_axle(&a, &f);
            // a trailer's axles: plain round beams, no differential
            ax[0].mesh = axle_tube(&a);
            parts.extend(ax);
            extra.extend(heavy::axle_meshes(&a, &f));
        }
        let b = self.cargo();
        let mut cargo = heavy::cargo_box(&b, &f, Mat::Paint);
        let box_extra = box_meshes(&b);
        cargo.meshes = box_extra.iter().map(|e| e.0.clone()).collect();
        extra.extend(box_extra);
        parts.push(cargo);
        let r = self.rear_frame();
        let mut rear = heavy::rear_frame(&r, &b, &f);
        let rear_extra = rear_meshes(&r);
        rear.meshes = rear_extra.iter().map(|e| e.0.clone()).collect();
        extra.extend(rear_extra);
        parts.push(rear);
        let (guard, guard_extra) = heavy::rear_guard(&self.guard(), &f);
        parts.push(guard);
        extra.extend(guard_extra);
        let (legs, leg_extra) = heavy::landing_gear(&self.landing(), &f);
        parts.extend(legs);
        extra.extend(leg_extra);
        parts.push(self.kingpin_plate(&b));
        extra.push(("kingpin".into(), kingpin_mesh()));
        // mud flaps behind the rear axle, on the cross-members
        let xm = parts.iter_mut().find(|p| p.name == "frame_crossmembers").unwrap();
        for s in [1.0f64, -1.0] {
            let sd = if s > 0.0 { "L" } else { "R" };
            let m = box_(
                DVec3::new(s * 0.5 * TRACK, 0.5 * (0.25 + FLOOR_Y - 0.05), Z_RA - 0.62),
                DVec3::new(0.32, 0.5 * (FLOOR_Y - 0.30), 0.006),
                Mat::Rubber,
                None,
            );
            extra.push((format!("mudflap_{sd}"), m));
            xm.meshes.push(format!("mudflap_{sd}"));
        }
        (parts, extra)
    }

    /// The upper coupler: the kingpin's plate under the front of the floor, bolted to the
    /// floor's first two stations.
    fn kingpin_plate(&self, b: &CargoBox) -> MechPart {
        let mut p = heavy::block(
            "kingpin_plate",
            "kingpin",
            // (its top 25 mm under the floor: outside the 20 mm contact range)
            DVec3::new(-0.45, FLOOR_Y - 0.075, Z_KING - 0.45),
            DVec3::new(0.45, FLOOR_Y - 0.025, Z_KING + 0.45),
            KG_KINGPIN,
            0.003,
            Mat::MetalBare,
            vals(0.0, 0.0, f64::INFINITY, 2.0e6),
            Zone::Other,
        );
        let from: Vec<String> = p.nodes.iter().map(|n| n.name.clone()).collect();
        // (to the floor over the rails and its edges at the stations about it: on the
        // edges alone, its bolts nearly level, the 250 kg plate rocked for seconds)
        let st = b.stations();
        let near: Vec<usize> = (0..st.len()).filter(|&k| (st[k] - Z_KING).abs() < 1.3).collect();
        let to: Vec<String> = [1.0f64, -1.0]
            .iter()
            .flat_map(|&s| near.iter().flat_map(move |&k| ['i', 'f'].map(|c| heavy::box_node(k, c, s))))
            .collect();
        heavy::mount(&mut p, "kingpin_plate_bolts", from, to, 16, vals(0.0, 0.0, 200e3, 400e3));
        p.meshes = vec!["kingpin".into()];
        p
    }

    fn markers(&self) -> Vec<Marker> {
        let mut out = Vec::new();
        for (tag, z) in [("F", Z_FA), ("R", Z_RA)] {
            for s in [1.0f64, -1.0] {
                let sd = if s > 0.0 { "L" } else { "R" };
                out.push(Marker {
                    name: format!("wheel_center_{tag}{sd}"),
                    pos: DVec3::new(s * 0.5 * TRACK, WHEEL_Y, z),
                    xaxis: DVec3::X * s,
                });
            }
        }
        let g = self.guard();
        let f = self.frame();
        for s in [1.0f64, -1.0] {
            let sd = if s > 0.0 { "L" } else { "R" };
            out.push(Marker {
                name: format!("bolt_guard_{sd}"),
                pos: DVec3::new(s * g.support_x, 0.5 * (g.lower_y + g.section + f.bottom_y()), g.z_rear + 0.5 * g.section),
                xaxis: DVec3::X,
            });
        }
        out
    }
}

/// A trailer axle's beam: a round tube between the hubs.
fn axle_tube(a: &BeamAxle) -> Mesh {
    let kp = a.seat_x + 0.26;
    let mut m = closed_lathe(&[(0.0001, -kp), (0.065, -kp), (0.065, kp), (0.0001, kp)], 24, Mat::TrimBlack);
    // (the lathe's axis is +y; turned to +x)
    let r = glam::DMat3::from_cols(DVec3::Y, -DVec3::X, DVec3::Z);
    m.transform(Some(r), DVec3::new(0.0, a.wheel_y, a.z));
    m
}

/// What rides the box: its bottom side rails, the front corner caps and the side marker lamps.
fn box_meshes(b: &CargoBox) -> Vec<(String, Mesh)> {
    let len = b.z_front - b.z_rear;
    let zc = 0.5 * (b.z_front + b.z_rear);
    let mut trim = Mesh::new();
    let mut lamps = Mesh::new();
    for s in [1.0f64, -1.0] {
        // the bottom side rail, proud of the wall
        trim.add(&box_(
            DVec3::new(s * (b.half_w + 0.006), b.floor_y + 0.07, zc),
            DVec3::new(0.006, 0.07, 0.5 * len - 0.01),
            Mat::MetalBare,
            None,
        ));
        // the front corner posts
        trim.add(&box_(
            DVec3::new(s * (b.half_w - 0.04), 0.5 * (b.floor_y + b.roof_y), b.z_front + 0.006),
            DVec3::new(0.04, 0.5 * (b.roof_y - b.floor_y) - 0.03, 0.006),
            Mat::MetalBare,
            None,
        ));
        for k in 0..5 {
            let z = b.z_rear + 0.3 + (len - 0.6) * k as f64 / 4.0;
            lamps.add(&box_(DVec3::new(s * (b.half_w + 0.014), b.floor_y + 0.07, z), DVec3::new(0.002, 0.025, 0.04), Mat::LampLens, None));
        }
    }
    vec![("box_trim".into(), trim), ("marker_lamps".into(), lamps)]
}

/// What rides the rear frame: the two doors, their lock rods, the hinges, the tail lamps.
fn rear_meshes(r: &RearFrame) -> Vec<(String, Mesh)> {
    let z = r.z_rear - 0.004;
    let (y0, y1) = (r.sill_y + 0.30, r.top_y - 0.12);
    let w = r.half_w - 0.10;
    let mut doors = Mesh::new();
    let mut rods = Mesh::new();
    let mut hinges = Mesh::new();
    for s in [1.0f64, -1.0] {
        doors.add(&box_(DVec3::new(s * 0.5 * (w + 0.006), 0.5 * (y0 + y1), z), DVec3::new(0.5 * w - 0.006, 0.5 * (y1 - y0), 0.004), Mat::Paint, None));
        for x in [0.25, 0.75] {
            rods.add(&box_(DVec3::new(s * x * w, 0.5 * (y0 + y1), z - 0.016), DVec3::new(0.014, 0.5 * (y1 - y0) + 0.04, 0.012), Mat::Chrome, None));
        }
        for k in 0..4 {
            let y = y0 + 0.2 + (y1 - y0 - 0.4) * k as f64 / 3.0;
            hinges.add(&box_(DVec3::new(s * (w + 0.03), y, z - 0.008), DVec3::new(0.04, 0.05, 0.008), Mat::TrimBlack, None));
        }
    }
    let mut lamps = Mesh::new();
    for s in [1.0f64, -1.0] {
        for dx in [0.0, 0.16] {
            let mut l = closed_lathe(&[(0.0001, -0.008), (0.05, -0.008), (0.05, 0.008), (0.0001, 0.008)], 24, Mat::LampLens);
            let rot = glam::DMat3::from_cols(DVec3::X, DVec3::Z, -DVec3::Y);
            l.transform(Some(rot), DVec3::new(s * (w - 0.10 - dx), r.sill_y + 0.13, r.z_rear - 0.008));
            lamps.add(&l);
        }
    }
    vec![
        ("rear_doors".into(), doors),
        ("door_rods".into(), rods),
        ("door_hinges".into(), hinges),
        ("taillights".into(), lamps),
    ]
}

/// The kingpin (2 in) under its plate.
fn kingpin_mesh() -> Mesh {
    let mut m = closed_lathe(&[(0.0001, 0.0), (0.0254, 0.0), (0.0254, 0.08), (0.0001, 0.08)], 24, Mat::MetalBare);
    m.translate(DVec3::new(0.0, FLOOR_Y - 0.075 - 0.0805, Z_KING));
    m
}

/// The box's section as a tube (the rig data's key lines). No panel is cut from it: every
/// part is a definition.
fn box_tube() -> Tube {
    let w = HALF_W;
    let keys = vec![
        Key::consts(0.0, FLOOR_Y),
        Key::consts(w - 0.05, FLOOR_Y),
        Key::consts(w, FLOOR_Y + 0.05),
        Key::consts(w, DECK_Y),
        Key::consts(w, 2.5),
        Key::consts(w, ROOF_Y - 0.05),
        Key::consts(w - 0.05, ROOF_Y),
        Key::consts(0.0, ROOF_Y),
    ];
    let h = |a: f64, b: f64| (Handle::C(a), Handle::C(b));
    let handles = (0..keys.len()).map(|_| h(1.0 / 3.0, 1.0 / 3.0)).collect();
    Tube::new(keys, handles, vec![])
}

impl Design for BoxTrailer {
    fn name(&self) -> String {
        "box_trailer".into()
    }
    fn paint(&self) -> ([f64; 3], f64, f64) {
        (self.p.paint, self.p.metallic, self.p.roughness)
    }
    fn tube(&self) -> Tube {
        box_tube()
    }
    fn zlines(&self) -> Vec<(&'static str, f64)> {
        vec![("BR", Z_REAR), ("BF", Z_FRONT)]
    }
    fn vlines(&self) -> Vec<f64> {
        vec![0.0, 3.0, 6.0, 9.0]
    }
    fn assign(&self, _zf: &str, _zt: &str, _lo: usize, _hi: usize, _side: Side) -> Option<(String, Mat)> {
        None
    }
    fn build_layout(&self, _cb: &mut CarBody) {}
    fn build_caps(&self, _cb: &mut CarBody) {}
    fn lens_tint(&self, name: &str) -> [f64; 4] {
        if name.starts_with("taillight") || name == "guard_tape_red" {
            [0.85, 0.03, 0.02, 1.0]
        } else if name == "marker_lamps" {
            [1.0, 0.45, 0.02, 1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
    }
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
        Some(RigSpec {
            wheelbase: TANDEM,
            track: TRACK,
            axle_f: Z_FA,
            axle_r: Z_RA,
            tyre_d: TYRE_D,
            tyre_w: a.contact_w(),
            rim_d: RIM_D,
            wheel_y: WHEEL_Y,
            rear_wheel: Some(AxleWheel { wheel_y: WHEEL_Y, tyre_d: TYRE_D, tyre_w: a.contact_w(), rim_d: RIM_D }),
            pose: None,
            z_front: Z_FRONT,
            z_rear: Z_REAR,
            doors: vec![],
            rear_closure: "trunk_lid".into(),
            arches: vec![(Z_FA, WHEEL_Y, 0.56), (Z_RA, WHEEL_Y, 0.56)],
            front_z: std::rc::Rc::new(|_, _| Z_FRONT),
            rear_z: std::rc::Rc::new(|_, _| Z_REAR),
            physics,
            surfaces: Vec::new(),
            structure: None,
            mechanical: std::rc::Rc::new(Vec::new()),
        })
    }
    /// No unibody: an empty body-in-white, so the rigger builds the trailer from its parts
    /// alone.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The figures the design states.
    #[test]
    fn published_figures() {
        let t = BoxTrailer::new(BoxTrailerParams::default());
        assert!((LENGTH - 16.154).abs() < 1e-3 && (2.0 * HALF_W - 2.591).abs() < 1e-3);
        assert!((Z_KING - Z_RA - 12.192).abs() < 1e-3);
        let g = t.guard();
        // FMVSS 224: lower edge ≤ 560 mm, ends within 100 mm of the sides, rear face ≤ 305 mm
        assert!(g.lower_y <= 0.56 && HALF_W - g.half_w <= 0.10 && g.z_rear - Z_REAR <= 0.305);
        // FMVSS 223: the section ≥ 100 mm; P1's moment within the tube's plastic moment
        let arm = g.p1_x() - g.support_x;
        assert!(g.section >= 0.1 && g.m_p() >= 50e3 * arm, "M_p {} against {}", g.m_p(), 50e3 * arm);
        // the supports' beams hold the uniform load: their forces balance it
        let f = t.frame();
        let beams: Vec<_> = [-1.0, 1.0].iter().flat_map(|&s| heavy::support_beams(&g, &f, s)).collect();
        let forces = heavy::support_forces(&g, &f);
        let mut sum = DVec3::ZERO;
        for (b, x) in beams.iter().zip(&forces) {
            // (tension pulls the bar toward the rail node)
            sum += (b.4 - b.3).normalize() * *x;
            println!("{} {}-{}: {:.1} kN", b.0, b.1, b.2, x * 1e-3);
        }
        assert!((sum + DVec3::new(0.0, 0.0, g.udl_n)).length() < 1.0, "the supports' forces sum to {sum}");
        let (defs, _) = t.assembly();
        let kg: f64 = defs.iter().flat_map(|p| p.nodes.iter()).map(|n| n.mass).sum::<f64>() + 4.0 * 20.0;
        assert!((kg - GROSS).abs() < 1.0, "{kg} kg");
    }
}
