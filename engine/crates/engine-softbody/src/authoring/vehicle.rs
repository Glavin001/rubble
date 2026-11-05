//! A vehicle definition (shape, panels, structure, masses, suspension, look) and
//! the build that turns it into a node–beam skeleton and `.truck` text.
//!
//! Coordinates: x across (+x the car's left), y up, z along the car from the rear
//! end (0) to the front (`profile.length`); the `.truck` is centred on the length
//! (z − length / 2), its forward axis +z.

use super::body::{build_shell, check_closed, BodyProfile, Seg, Shell, ShellBeam, Spacing};
use super::curve::{Curve, Outline};
use super::material::{Grades, Material};
use super::members::{attach, block_spaced, box_truss, corner, plate, Corner, MemberSpec, Truss};
use super::skeleton::{BeamType, Skeleton};
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Where on the shell a panel or zone lies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Region {
    /// Side surfaces (sill corner, side, greenhouse side) inside an outline drawn
    /// in side view, (z, y).
    Side(Outline),
    /// The top surface between two stations.
    Top { z0: f32, z1: f32 },
    /// An end cap (nose when `front`) between two heights.
    Cap { front: bool, y0: f32, y1: f32 },
    /// Any shell node (but the underbody's) inside a box: |x| in [x0, x1], y, z.
    Box {
        x: [f32; 2],
        y: [f32; 2],
        z: [f32; 2],
    },
    /// The underbody.
    Floor { z0: f32, z1: f32 },
    /// Any of these.
    Any(Vec<Region>),
}

impl Region {
    pub fn contains(&self, p: Vec3, seg: Seg, length: f32) -> bool {
        match self {
            Region::Side(o) => {
                matches!(seg, Seg::Lower | Seg::Side | Seg::Upper) && o.contains(p.z, p.y)
            }
            Region::Top { z0, z1 } => seg == Seg::Top && p.z >= *z0 && p.z <= *z1,
            Region::Cap { front, y0, y1 } => {
                seg == Seg::Cap && (p.z > 0.5 * length) == *front && p.y >= *y0 && p.y <= *y1
            }
            Region::Box { x, y, z } => {
                seg != Seg::Floor
                    && p.x.abs() >= x[0]
                    && p.x.abs() <= x[1]
                    && p.y >= y[0]
                    && p.y <= y[1]
                    && p.z >= z[0]
                    && p.z <= z[1]
            }
            Region::Floor { z0, z1 } => {
                matches!(seg, Seg::Floor | Seg::Well) && p.z >= *z0 && p.z <= *z1
            }
            Region::Any(v) => v.iter().any(|r| r.contains(p, seg, length)),
        }
    }
}

/// Which edge of a hinged panel carries the hinges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Edge {
    /// The panel's edge toward the front (largest z).
    Front,
    /// Toward the rear.
    Rear,
}

/// How a panel is joined to what it meets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Joint {
    /// Spot-welded: the weaker of the two materials.
    Weld,
    /// Bolted on, `scale` × the weaker material; with `group`, one detacher
    /// group: when one mount fails the panel comes off whole.
    Bolt { scale: f32, group: bool },
    /// Hinges along one edge, a latch at the other, compression-only seal
    /// contacts elsewhere (a door, bonnet or boot lid).
    Hinged {
        hinge: Edge,
        hinge_scale: f32,
        latch_scale: f32,
        /// Latch zone across the opposite edge: heights (doors) or |x| below a
        /// limit (lids).
        latch_y: Option<[f32; 2]>,
        latch_x: Option<f32>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Panel {
    pub name: String,
    pub region: Region,
    /// Material grade (a [`Grades`] field name).
    pub grade: String,
    pub joint: Joint,
    /// One part per side (`<name>_l`, `<name>_r`): doors.
    pub sided: bool,
}

/// A shell zone's material (glass, floor pan, sills) inside a part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Zone {
    pub region: Region,
    pub grade: String,
}

/// A lattice block of named mass (engine, gearbox, battery).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockSpec {
    pub name: String,
    pub centre: [f32; 3],
    pub half: [f32; 3],
    pub spacing: f32,
    /// Spacing along z (m; 0: `spacing`): a block's front and rear faces
    /// keep `spacing` across, its depth needs fewer layers.
    #[serde(default)]
    pub depth: f32,
    /// The lattice's stiffness against the rigid grade's (a heavy block's
    /// own mass pays for a stiffer lattice: an engine block is rigid).
    #[serde(default = "one")]
    pub stiff: f32,
    pub kg: f32,
    /// Mounts: each outer node to this many nearest body nodes within `reach`.
    pub mounts: usize,
    pub reach: f32,
    pub mount_grade: String,
    pub mount_scale: f32,
    /// Bearing (m, 0: none): the block's rear face pushes on the structure
    /// behind it within this reach (compression-only beams, as an engine
    /// driven back meets the firewall and tunnel in a crash).
    #[serde(default)]
    pub bear: f32,
}

/// A long diagonal across a panel the body's torsion runs through (and its
/// mirror): a shear panel lumped into an X.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BraceSpec {
    pub name: String,
    pub from: [f32; 3],
    pub to: [f32; 3],
    pub grade: String,
    /// Axial rigidity k·L (N): a lumped panel is a long beam, so its k·L is the
    /// panel's (a grade's is a strip's); the grade gives its damping.
    pub kl: f32,
    /// Yield force (N) of the whole brace (0: at the grade's yield strain);
    /// it breaks at four times that. Braces into a crumple zone (strut towers to
    /// the cowl) yield as the zone folds rather than carry the crash into the
    /// cabin.
    #[serde(default)]
    pub strength: f32,
    /// The members `from` and `to` lie on (their left-side names): each end
    /// is the station of that member nearest its point, on whichever side
    /// the point falls. Explicit ends, so a member added elsewhere never
    /// moves a brace. Empty: the frame node nearest the point and its
    /// station (the first member's through it).
    #[serde(default)]
    pub on: [String; 2],
}

/// A shear panel between two members (a face of the body's box: roof between
/// the roof rails, floor between the sills, a door ring between sill and roof
/// rail): each bay of `a` is crossed by an X to the stations of `b` nearest
/// its two stations along `axis` (0 x, 1 y, 2 z), between the nodes of each
/// station that face each other. A triangulated face: its shear stiffness is
/// the diagonals' (`kl`, the grade's yield strain).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShearSpec {
    pub a: String,
    pub b: String,
    pub axis: usize,
    pub grade: String,
    pub kl: f32,
    /// Yield force (N) of each diagonal (0: at the grade's yield strain; it
    /// breaks at four times that): glass cracks.
    #[serde(default)]
    pub strength: f32,
}

/// A named mass spread over the nodes inside a box (seats, fuel, passengers).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NamedMass {
    pub name: String,
    pub kg: f32,
    pub lo: [f32; 3],
    pub hi: [f32; 3],
}

/// A flat braced plate (radiator) with its mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlateSpec {
    pub name: String,
    pub corner: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub kg: f32,
    pub grade: String,
    pub mount_scale: f32,
}

/// The definition's look (the visual mesh reads it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Look {
    pub paint: [f32; 3],
    /// Side windows (side view).
    pub side_glass: Vec<Outline>,
    /// Windscreen and rear window: (z0, z1) of the top surface, and the black
    /// border's width.
    pub windscreen: [f32; 2],
    pub backlight: [f32; 2],
    pub frit: f32,
    /// Headlights and grille in front view (x ≥ 0, y), tail lights in rear view.
    pub headlight: Outline,
    pub grille: Outline,
    pub intake: Outline,
    pub taillight: Outline,
    /// Door handles: (z, y) centres in side view.
    pub handles: Vec<[f32; 2]>,
    /// Panel gap (m).
    pub gap: f32,
    /// Plates (front, rear): centre (y) and half size.
    pub plate_y: [f32; 2],
    /// Mirror: (z, y) on the A-pillar base.
    pub mirror: [f32; 2],
}

/// A hinged or bolted part's hardware: each one detacher group (when any of
/// its beams breaks, all do), as BeamNG authors a hinge, latch or bolt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hardware {
    HingeUpper,
    HingeLower,
    /// The latch fully latched (its primary position).
    Latch,
    /// The latch's secondary position: it holds the door once the primary
    /// has let go.
    LatchSecondary,
}

/// A hardware group as built: its beams (indices into `skeleton.beams`).
#[derive(Clone, Debug)]
pub struct HardwareGroup {
    /// `<part>.hinge_upper` and so on.
    pub name: String,
    pub part: u16,
    pub kind: Hardware,
    /// The solver's detacher group.
    pub group: u32,
    pub beams: Vec<usize>,
    /// Its rig's pulls: along the car away from the body edge it holds to,
    /// and outward (a door pulled open, a lid lifted).
    pub along: Vec3,
    pub outward: Vec3,
    /// It also carries the part's weight (a door's; a lid rests on its
    /// seals).
    pub weight: bool,
}

/// What a damage unit is (the renderer's damage events follow it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DamageKind {
    Glass,
    Lamp,
    Mirror,
    Hinge,
    Latch,
    Mount,
    Part,
}

/// A unit of damage the renderer follows: broken when any of its beams
/// (indices into `skeleton.beams`) is broken.
#[derive(Clone, Debug)]
pub struct DamageUnit {
    pub name: String,
    pub kind: DamageKind,
    pub part: u16,
    pub beams: Vec<u32>,
}

/// A FMVSS 206 test's ultimate loads (N): Load Test One (longitudinal, along
/// the car) and Load Test Two (transverse, the door pulled outward).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Loads {
    pub longitudinal: f32,
    pub transverse: f32,
}

/// Measured door hardware strengths: the medians of the FMVSS 206
/// compliance reports loaded to failure (`crash-reference-targets.md` §9.3
/// and §9.4, 2005–2012 hardware). The latch fully latched (primary) fails at
/// 28.0 / 20.4 kN, in its secondary position at 18.9 / 10.3 kN. No hinge
/// system separated in any report: every hinge value is a floor. §9.4 takes
/// the pair at or above the latch's (median floor 28 / 21 kN); the pair is
/// taken at the highest load any pair held without separating, 36.2 / 29.8
/// kN (2005 Camry, report 637726), the best-supported floor, and each hinge
/// holds it on its own (`hardware_rigs`), so the door opens at its latch
/// before a hinge lets go, as every report found.
/// The depth of a lid's rigidifier node under its centroid (m): the inner
/// frame's 100 mm hat (a Choice).
pub const RIGIDIFIER_DEPTH: f32 = 0.10;

pub const PRIMARY_LATCH: Loads = Loads {
    longitudinal: 28_000.0,
    transverse: 20_500.0,
};
pub const SECONDARY_LATCH: Loads = Loads {
    longitudinal: 19_000.0,
    transverse: 10_300.0,
};
pub const HINGE_SYSTEM: Loads = Loads {
    longitudinal: 36_179.0,
    transverse: 29_813.0,
};

/// FMVSS 206 (49 CFR 571.206) minimums the hardware must hold: S4.1.1.1–2
/// (latch fully latched 11,000 N longitudinal, 9,000 N transverse), S4.1.1.3
/// (secondary 4,500 N both), S4.1.2.1 (hinge system 11,000 / 9,000 N).
pub const FMVSS_206_LATCH: Loads = Loads {
    longitudinal: 11_000.0,
    transverse: 9_000.0,
};
pub const FMVSS_206_SECONDARY: Loads = Loads {
    longitudinal: 4_500.0,
    transverse: 4_500.0,
};
pub const FMVSS_206_HINGES: Loads = Loads {
    longitudinal: 11_000.0,
    transverse: 9_000.0,
};

/// The secondary latch's engagement stiffness against the primary's (halved
/// at build until, fully latched, it no longer goes first: the assembly
/// fails with its primary).
pub const SECONDARY_STIFF: f32 = 0.5;

/// Directions in a hardware rig's fan, from longitudinal to transverse.
pub const RIG_FAN: usize = 6;

/// A seal's (compression-only) break on stretch, against its rest length.
pub const SEAL_BREAK: f32 = 10.0;

/// A pillar's door flange: the strip of sheet a door end bears on, pushed in
/// (`t` gauge, `width` the strip a door member's end bears on, `sigma_y`
/// its yield stress, Pa). Its in-plane capacity is σ_y·t·width: 1.8 mm
/// press-hardened steel (σ_y 1,100 MPa) over 50 mm carries 99 kN.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FlangeSpec {
    /// The member (pillar) it is part of.
    pub member: String,
    pub t: f32,
    pub width: f32,
    pub sigma_y: f32,
}

impl FlangeSpec {
    /// Its crushing load (N).
    pub fn capacity(&self) -> f32 {
        self.sigma_y * self.t * self.width
    }
}

/// A panel's sheet layers: `(gauge m, share of the panel's area, density
/// kg/m³)` (an outer skin is its whole area, an inner frame's stamping the
/// share its cut-outs leave).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SheetSpec {
    /// Part name prefix (`fender` takes both).
    pub part: String,
    pub layers: Vec<[f32; 3]>,
}

impl SheetSpec {
    /// Mass per square metre of the panel (kg/m²).
    pub fn areal(&self) -> f32 {
        self.layers.iter().map(|l| l[0] * l[1] * l[2]).sum()
    }
}

/// Each part's skin area (m²): every skin triangle's area, a third to each
/// of its nodes' parts.
pub fn part_areas(sk: &Skeleton, tris: &[[u32; 3]]) -> Vec<f32> {
    let mut area = vec![0.0f32; sk.parts.len()];
    for t in tris {
        let (p0, p1, p2) = (sk.pos(t[0]), sk.pos(t[1]), sk.pos(t[2]));
        let a = 0.5 * (p1 - p0).cross(p2 - p0).length();
        for &i in t {
            area[sk.nodes[i as usize].part as usize] += a / 3.0;
        }
    }
    area
}

/// A part's sheet mass (kg), if it has a sheet spec.
pub fn sheet_mass(def: &VehicleDef, sk: &Skeleton, tris: &[[u32; 3]], part: u16) -> Option<f32> {
    let name = &sk.parts[part as usize];
    let spec = def
        .sheets
        .iter()
        .find(|s| name.starts_with(s.part.as_str()))?;
    Some(part_areas(sk, tris)[part as usize] * spec.areal())
}

/// The flange's offset inboard of a door member end's inner node (m): the
/// door seal's section between them.
pub const FLANGE_GAP: f32 = 0.03;

/// A door frame's ends bearing on the pillars (compression only): thresholds
/// × this against the door frame's own (a side load goes into the pillars
/// and sill, as FMVSS 214's crush does).
pub const DOOR_BEAR: f32 = 3.0;

/// A catcher's body dashpot against `body_damping`.
pub const CATCH_DAMPING: f32 = 3.0;

/// Catchers in the frame tier (their ties at the member's grade × `CATCH_TIE`,
/// not scaled with the skin).
pub const CATCHERS_IN_FRAME: bool = false;

/// A block's bearing beams' thresholds against the UHSS grade: an engine
/// driven back crushes into the firewall rather than stopping the car
/// against it. 0.6 since the re-node (heavier frame nodes, 2026-11-05): the
/// engine meeting the firewall gives the 56 km/h pulse its peak (median 35.5 g
/// against NCAP's 34–47; 0.3 gave 34).
pub const BEAR_SCALE: f32 = 0.6;

/// A plate's node spacing (m): the radiator's six nodes, each corner and
/// mid-edge on its mounts (a four-node plate on the 1 m frame rattled on
/// its mounts at rest).
pub const PLATE_SPACING: f32 = 0.6;

/// The shock top's ties to its tower against the link grade.
pub const SHOCK_TOP: f32 = 4.0;

/// A gusset tie's stiffness against the UHSS grade (members that meet
/// without sharing a node: each end-station node to its two nearest of the
/// other member's station).
pub const GUSSET: f32 = 1.0;

/// One tier: a member's lane on the shell reaches this far past its
/// section's half depth (m), and takes the shell's beams within 37° of its
/// path (cos ≥ [`LANE_COS`]): Choices.
pub const LANE_REACH: f32 = 0.08;
pub const LANE_COS: f32 = 0.8;

/// A side pole's width (m): the NCAP and FMVSS 214 pole is 254 mm. One
/// tier: the shell's spacing and the outer members' stations are at most
/// this (static contact is per node).
pub const POLE_LANE: f32 = 0.24;

/// The skin's spacing against the frame's era (25–40 cm): the light skin
/// tier is denser than the frame, so dents and creases are local.
pub const SKIN_DENSITY: f32 = 1.0;

/// Node spacing per resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resolution {
    Low,
    Medium,
    High,
}

impl Resolution {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    /// Shell spacing. The shell is the body-in-white's outer surface, and its
    /// nodes carry the body's mass (trim, glass, interior lumped onto them), so
    /// it is sparse: 25–40 cm, as real node–beam cars are. The 0.5 ms step allows
    /// a beam's stiffness in proportion to its nodes' mass, and a line's axial
    /// rigidity at a fixed mass grows with the square of its node spacing: a
    /// sparse body is a stiff one. Impact faces get their 11 cm node lanes from
    /// the structure (bumper beams, door beams, sills and B-pillars' catchers).
    pub fn spacing(self) -> Spacing {
        let f = SKIN_DENSITY;
        let s = match self {
            Resolution::Low => Spacing {
                ring: 0.36,
                side: 0.30,
                top: 0.36,
                floor: 0.42,
                cap: 0.30,
            },
            Resolution::Medium => Spacing {
                ring: 0.30,
                side: 0.25,
                top: 0.30,
                floor: 0.35,
                cap: 0.25,
            },
            Resolution::High => Spacing {
                ring: 0.28,
                side: 0.24,
                top: 0.28,
                floor: 0.33,
                cap: 0.24,
            },
        };
        Spacing {
            ring: s.ring * f,
            side: s.side * f,
            top: s.top * f,
            floor: s.floor * f,
            cap: s.cap * f,
        }
    }

    /// Station spacing of the structural trusses (at most; every path vertex
    /// is a station). 1 m: a frame of about 330 heavy nodes (1–4 kg) besides
    /// the doors' frames, as BeamNG's cars are built. At a fixed mass and
    /// step a line's axial rigidity grows with the square of its station
    /// spacing, so long stations make a stiff frame; the impact members that
    /// a pole loads mid-span set their own (`MemberSpec::station`: half their
    /// span, so the centre hinge has its node). The same at every
    /// resolution: the resolution sets the skin.
    pub fn truss(self) -> f32 {
        1.0
    }

    /// Station spacing of crush members (beaded rails): long enough beams to
    /// yield in compression (the solver keeps every rest length ≥ 0.1 m).
    pub fn crush(self) -> f32 {
        0.6
    }
}

/// A complete vehicle definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VehicleDef {
    pub name: String,
    pub profile: BodyProfile,
    pub grades: Grades,
    /// Ring stations kept exactly (door seams, pillars, screens).
    pub stations: Vec<f32>,
    /// Panels in priority order (the first whose region holds a node owns it).
    pub panels: Vec<Panel>,
    /// The shell's default grade (outside every zone) and its zones.
    pub shell_grade: String,
    pub zones: Vec<Zone>,
    /// Threshold scale of the shell's beams along z (crumple zones weak, the cell
    /// strong).
    pub crumple: Curve,
    pub members: Vec<MemberSpec>,
    pub blocks: Vec<BlockSpec>,
    pub plates: Vec<PlateSpec>,
    pub masses: Vec<NamedMass>,
    /// Suspension corners (left side; each mirrored).
    pub corners: Vec<Corner>,
    /// Kerb mass (kg): everything, wheels included.
    pub kerb_mass: f32,
    pub look: Look,
    /// The shell's frame: every this many rings (and at the ends) about 14 hard
    /// points around the section are joined into a braced tube of long, stiff
    /// beams (the body's deep section), panel by panel; the skin between rides it.
    #[serde(default = "three")]
    pub frame_every: usize,
    #[serde(default = "hsla")]
    pub frame_grade: String,
    /// Shear webs between members (inner wheelhouses between rails and aprons):
    /// every station of the first joined to the nearest station of the second
    /// within reach (m), every pair of their nodes.
    #[serde(default)]
    pub webs: Vec<(String, String, f32)>,
    /// Shear panels as long diagonals (floor pan, roof, firewall, rear
    /// bulkhead): each between the body nodes nearest its two points, and its
    /// mirror (an X).
    #[serde(default)]
    pub braces: Vec<BraceSpec>,
    /// Panels backed by a frame (doors: their own intrusion beam, belt and
    /// inner rail; fenders: the body's apron and towers): every skin node of
    /// a part named so (prefix) is tied to its two nearest frame nodes of the
    /// part (or of the body, for a part without a frame) within 0.3 m, so a
    /// push on the skin meets structure a few centimetres behind it.
    #[serde(default)]
    pub backed: Vec<String>,
    /// Shear panels between members ([`ShearSpec`]).
    #[serde(default)]
    pub shear: Vec<ShearSpec>,
    /// Frame beams may cross into these panels (closed doors carry load, as a
    /// real door ring does).
    #[serde(default)]
    pub frame_through: Vec<String>,
    /// X-brace the shell's quads (their second diagonal). A sparse shell's
    /// triangles hold their shape without it.
    #[serde(default = "yes")]
    pub shell_braces: bool,
    /// Body damping: a dashpot of this many N·s/m from each node to one about
    /// `body_damping_reach` m away in its part (0: none), torn off past
    /// `body_damping_strength` N.
    #[serde(default)]
    pub body_damping: f32,
    #[serde(default = "one")]
    pub body_damping_reach: f32,
    #[serde(default = "five_kn")]
    pub body_damping_strength: f32,
    /// Fit the step per tier: the primary structure (the frame: members'
    /// stations) at its grades, the rest scaled to what is left
    /// ([`Skeleton::fit_tiers`]); off: one scale for every beam. The frame's
    /// nodes take the structure's mass ([`super::skeleton::PRIMARY_WEIGHT`])
    /// and the named masses either way.
    #[serde(default)]
    pub tiered: bool,
    /// The pillars' door flanges: a door member's ends bear on them inboard
    /// (compression only), as a pushed-in door bears on the A- and
    /// B-pillars' flanges.
    #[serde(default)]
    pub flanges: Vec<FlangeSpec>,
    /// Panels whose mass is their sheet's: area (their skin's, each
    /// triangle's third to each node's part) × Σ gauge × share × density.
    #[serde(default)]
    pub sheets: Vec<SheetSpec>,
    /// One tier (BeamNG's construction): the body shell's nodes are the
    /// frame's. Members drawn on the body share the shell's nodes (their
    /// stations snap onto it), there are no catchers or skin-to-frame ties,
    /// every node weighs its real local mass ([`SkNode::local`]) and every
    /// beam keeps its stiffness unless the step's bound requires less
    /// ([`Skeleton::cap_to_bound`]). Off: the two-tier build.
    #[serde(default)]
    pub one_tier: bool,
    /// One tier: the painted body-in-white's measured mass (kg), spread
    /// over the body's shell and members by their sheet and sections.
    #[serde(default)]
    pub biw_mass: f32,
    /// One tier: parts' measured assembly masses (part name prefix, kg each:
    /// structure, glass, trim, hardware), spread over each part's nodes by
    /// its sheet and members.
    #[serde(default)]
    pub part_masses: Vec<(String, f32)>,
    /// One tier: beam damping as a damping ratio on the beam's own node
    /// masses ([`DampingRatios`]); `None` keeps the grades' damping.
    #[serde(default)]
    pub damping: Option<DampingRatios>,
    /// One tier: crush members that fold progressively ([`CrushSpec`]).
    #[serde(default)]
    pub crush: Vec<CrushSpec>,
    /// One tier: the floor and the roof, which no pole reaches, at the
    /// interior's coarser spacing ([`super::body::Interior`]); `None`: the
    /// pole lanes everywhere.
    #[serde(default)]
    pub interior: Option<super::body::Interior>,
}

/// One tier: a crush member (a rail, a bumper beam) that folds
/// progressively at its section's mean crush force and stays folded. Every
/// bay's cut (the beams of the member crossing the plane between two
/// stations) yields axially at `mean_force` (Σ yield × |cos| of each beam
/// to the member's axis); `mean_force` 0 keeps the built yields. Its beams
/// take JBeam's plastic limits (`rubble_jbeam_plastic`,
/// `jbeam-semantics-spec.md` §3): deformLimitStress = their yield (a fold
/// does not harden) and `deform_limit` (the stack of folds bottoms out).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrushSpec {
    pub member: String,
    /// The section's mean progressive crush force (N).
    pub mean_force: f32,
    /// JBeam's deformLimit: the share of a beam's length it may crush.
    pub deform_limit: f32,
}

/// Abramowicz and Jones's mean progressive crush force of a thin-walled
/// rectangular steel tube (N): P_m = 13.06·σ₀·t^(5/3)·C^(1/3), C = (b + d)/2,
/// σ₀ = (σ_y + σ_u)/2 the flow stress (W. Abramowicz, N. Jones, "Dynamic
/// axial crushing of square tubes", Int. J. Impact Eng. 2 (1984) 179–208,
/// static, with the effective crushing distance 0.73 of the fold length;
/// Wierzbicki and Abramowicz's 9.56 without it, J. Appl. Mech. 50 (1983)
/// 727–734). `b`, `d`, `t` in m, stresses in Pa.
pub fn progressive_crush_force(b: f32, d: f32, t: f32, sigma_y: f32, sigma_u: f32) -> f32 {
    let c = 0.5 * (b + d);
    13.06 * 0.5 * (sigma_y + sigma_u) * t.powf(5.0 / 3.0) * c.powf(1.0 / 3.0)
}

/// Abramowicz and Jones's effective crushing distance: the share of a
/// tube's length a progressive fold stack crushes before it bottoms out
/// (ibid.): a crush member's beams' deformLimit.
pub const CRUSH_STROKE: f32 = 0.73;

/// One tier: each beam's damping as a ratio ζ = c / (2·√(k·m₁m₂/(m₁+m₂)))
/// of its own two-mass mode, per kind of part, so the ratio BeamNG's content
/// uses carries over to this car's real masses. Defaults: the medians
/// measured on the BX's default configuration (`examples/jbeam_damping.rs`,
/// `vehicle-authoring.md` "Where we differ"): never tuned to a reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DampingRatios {
    /// The body's shell and members (`bx_body`: median 0.0275, p10–p90
    /// 0.018–0.052).
    pub body: f32,
    /// Crush members ([`Self::crush_members`]): the BX's front frame rails
    /// (0.0177, 0.015–0.018) and its bumper bar (0.0158).
    pub crush: f32,
    /// Members whose beams count as crush structure.
    pub crush_members: Vec<String>,
    /// Doors (`bx_door`: 0.0334, 0.025–0.040).
    pub door: f32,
    /// Bonnet (`bx_hood`: 0.0593, 0.045–0.080).
    pub bonnet: f32,
    /// Boot lid (`bx_tailgate`: 0.0567, 0.038–0.065).
    pub bootlid: f32,
    /// Fenders (`bx_fender`: 0.0422, 0.039–0.053).
    pub fender: f32,
    /// Bumper fascias (`bx_bumper_F`/`_R`: 0.089–0.091).
    pub bumper: f32,
    /// The body's glass (`bx_windshield`: 0.0200, 0.017–0.026).
    pub glass_body: f32,
    /// A door's glass (`bx_doorglass`: 0.0545, 0.046–0.057).
    pub glass_side: f32,
}

impl Default for DampingRatios {
    fn default() -> Self {
        Self {
            body: 0.0275,
            crush: 0.0177,
            crush_members: [
                "rail_f",
                "rail_r",
                "rail_r_tail",
                "bumper_beam_f",
                "bumper_beam_r",
            ]
            .map(String::from)
            .to_vec(),
            door: 0.0334,
            bonnet: 0.0593,
            bootlid: 0.0567,
            fender: 0.0422,
            bumper: 0.0893,
            glass_body: 0.0200,
            glass_side: 0.0545,
        }
    }
}

impl DampingRatios {
    /// The ratio for a beam in part `part` (its name), glass or not, crush
    /// structure or not.
    pub fn of(&self, part: &str, glass: bool, crush: bool) -> f32 {
        let body = part == "body" || part.is_empty();
        if glass {
            return if body {
                self.glass_body
            } else {
                self.glass_side
            };
        }
        if crush {
            return self.crush;
        }
        let pick = [
            ("door", self.door),
            ("bonnet", self.bonnet),
            ("bootlid", self.bootlid),
            ("fender", self.fender),
            ("bumper", self.bumper),
        ];
        pick.iter()
            .find(|(n, _)| part.starts_with(n))
            .map_or(self.body, |x| x.1)
    }
}

fn one() -> f32 {
    1.0
}

fn five_kn() -> f32 {
    5000.0
}

fn yes() -> bool {
    true
}

fn three() -> usize {
    3
}

fn hsla() -> String {
    "hsla".into()
}

/// Build parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuildParams {
    pub resolution: Resolution,
    /// Every yield and break threshold × this.
    pub strength: f32,
    pub paint: Option<[f32; 3]>,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            resolution: Resolution::Medium,
            strength: 1.0,
            paint: None,
        }
    }
}

/// The built vehicle.
#[derive(Clone, Debug)]
pub struct Built {
    pub def: VehicleDef,
    pub params: BuildParams,
    pub skeleton: Skeleton,
    pub shell: Shell,
    /// The `.truck` text (centred on the length).
    pub truck: String,
    /// Shift from definition coordinates to the `.truck`'s: (0, 0, −length / 2).
    pub shift: Vec3,
    /// Each skeleton node's design position (definition coordinates), before
    /// the car settled into its rest pose (the skeleton's).
    pub design: Vec<Vec3>,
    /// Node indices per named member, block and corner.
    pub groups: HashMap<String, Vec<u32>>,
    /// Binding triangles over each group's nodes (for visual parts riding the
    /// structure: seats on their frames, the dashboard on its cross beam).
    pub bind_tris: HashMap<String, Vec<[u32; 3]>>,
    /// The primary structure's beams' stiffness, damping and thresholds were
    /// scaled by this so the explicit step stays stable within the kerb mass (1:
    /// the materials as defined), and every other beam's by `skin_scale`.
    pub stiffness_scale: f32,
    pub skin_scale: f32,
    /// Every hinge, latch and mount group.
    pub hardware: Vec<HardwareGroup>,
    /// Hardware, glass panes, lamps and mirrors as damage units.
    pub damage_units: Vec<DamageUnit>,
    /// One tier: where the kerb mass is (kg by item: the body-in-white, its
    /// glass, each part, the equipment spread over the body, and the
    /// scales that took the sections and sheet to the measured masses).
    pub masses: Vec<(String, f32)>,
}

/// Medians of the built skeleton's materials (the measures real node–beam cars
/// are compared by: `tests/showcase_rest_wobble.rs`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub nodes: usize,
    pub beams: usize,
    pub node_mass: f32,
    /// k × rest length (N) over the structure's beams and over every beam.
    pub kl_structure: f32,
    pub kl_all: f32,
    /// Damping ratio c / 2√(k μ), μ the node pair's reduced mass.
    pub zeta_structure: f32,
    pub zeta_all: f32,
    /// The primary tier (the frame): its nodes (members' stations; catchers
    /// are in the tier but not counted), their median mass and its beams'
    /// median k·L.
    pub primary_nodes: usize,
    pub primary_node_mass: f32,
    pub kl_primary: f32,
}

fn median(mut v: Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

impl Built {
    pub fn stats(&self) -> Stats {
        let sk = &self.skeleton;
        let m = |i: u32| sk.nodes[i as usize].mass;
        let per = |b: &super::skeleton::SkBeam| -> (f32, f32) {
            let l = sk.pos(b.a).distance(sk.pos(b.b));
            let k = b.mat.k_at(l);
            let mu = m(b.a) * m(b.b) / (m(b.a) + m(b.b));
            (k * l, b.mat.damp / (2.0 * (k * mu).sqrt()))
        };
        let all: Vec<(f32, f32)> = sk.beams.iter().filter(|b| !b.damper).map(per).collect();
        let st: Vec<(f32, f32)> = sk.beams.iter().filter(|b| b.structure).map(per).collect();
        Stats {
            nodes: sk.nodes.len(),
            beams: sk.beams.len(),
            node_mass: median(sk.nodes.iter().map(|n| n.mass).collect()),
            kl_structure: median(st.iter().map(|x| x.0).collect()),
            kl_all: median(all.iter().map(|x| x.0).collect()),
            zeta_structure: median(st.iter().map(|x| x.1).collect()),
            zeta_all: median(all.iter().map(|x| x.1).collect()),
            primary_nodes: sk.nodes.iter().filter(|n| n.primary && !n.catcher).count(),
            primary_node_mass: median(
                sk.nodes
                    .iter()
                    .filter(|n| n.primary && !n.catcher)
                    .map(|n| n.mass)
                    .collect(),
            ),
            kl_primary: median(
                sk.beams
                    .iter()
                    .filter(|b| !b.fixed && !b.tie && !b.damper && sk.beam_primary(b))
                    .map(|b| per(b).0)
                    .collect(),
            ),
        }
    }
}

impl Grades {
    pub fn get(&self, name: &str) -> Result<Material, String> {
        Ok(match name {
            "panel" => self.panel,
            "floor" => self.floor,
            "frame" => self.frame,
            "crush" => self.crush,
            "hsla" => self.hsla,
            "uhss" => self.uhss,
            "boron" => self.boron,
            "alu" => self.alu,
            "plastic" => self.plastic,
            "glass" => self.glass,
            "rigid" => self.rigid,
            "link" => self.link,
            "mount" => self.mount,
            _ => return Err(format!("unknown material grade `{name}`")),
        })
    }

    /// Every grade's thresholds × `s` (the build's strength scale; the rigid
    /// grade stays rigid).
    pub fn scaled(&self, s: f32) -> Self {
        Self {
            panel: self.panel.scaled(s),
            floor: self.floor.scaled(s),
            frame: self.frame.scaled(s),
            crush: self.crush.scaled(s),
            hsla: self.hsla.scaled(s),
            uhss: self.uhss.scaled(s),
            boron: self.boron.scaled(s),
            alu: self.alu.scaled(s),
            plastic: self.plastic.scaled(s),
            glass: self.glass.scaled(s),
            rigid: self.rigid,
            link: self.link,
            mount: self.mount.scaled(s),
        }
    }
}

/// A point's distance to a polyline, the polyline's unit tangent there and
/// the point's arc fraction along it.
fn path_distance(path: &[Vec3], q: Vec3) -> (f32, Vec3, f32) {
    let mut best = (f32::MAX, Vec3::Z, 0.0);
    let total: f32 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
    let mut acc = 0.0;
    for w in path.windows(2) {
        let d = w[1] - w[0];
        let l = d.length().max(1e-6);
        let u = ((q - w[0]).dot(d) / (l * l)).clamp(0.0, 1.0);
        let e = q.distance(w[0] + d * u);
        if e < best.0 {
            best = (e, d / l, (acc + u * l) / total.max(1e-6));
        }
        acc += l;
    }
    best
}

/// Build the skeleton and its `.truck`.
pub fn build(def: &VehicleDef, params: &BuildParams) -> Result<Built, String> {
    let grades = def.grades.scaled(params.strength);
    let res = params.resolution;
    let len = def.profile.length;
    let mut sk = Skeleton::new(&def.name);
    let one = def.one_tier;
    let mut spacing = if one {
        res.spacing().pole_lanes()
    } else {
        res.spacing()
    };
    let interior = def.interior.filter(|_| one);
    if let Some(it) = interior.as_ref().filter(|it| it.floor > 0.0) {
        spacing.floor = it.floor;
    }
    let shell = build_shell(
        &def.profile,
        &spacing,
        &def.stations,
        interior.as_ref(),
        &mut sk,
    );
    check_closed(&shell.tris).map_err(|e| format!("shell skin: {e}"))?;
    // Parts: the first panel whose region holds a node owns it.
    let mut panel_of: HashMap<u32, usize> = HashMap::new();
    for &i in &shell.nodes {
        let p = sk.pos(i);
        let seg = shell.seg[&i];
        if let Some(pi) = def
            .panels
            .iter()
            .position(|pn| pn.region.contains(p, seg, len))
        {
            let pn = &def.panels[pi];
            let name = if pn.sided {
                format!("{}_{}", pn.name, if p.x >= 0.0 { "l" } else { "r" })
            } else {
                pn.name.clone()
            };
            let part = sk.part(&name);
            sk.nodes[i as usize].part = part;
            panel_of.insert(i, pi);
        }
    }
    // Node grades: zones first, then the owning panel's grade, else the shell's.
    let shell_grade = grades.get(&def.shell_grade)?;
    let mut grade_of: HashMap<u32, Material> = HashMap::new();
    for &i in &shell.nodes {
        let p = sk.pos(i);
        let seg = shell.seg[&i];
        let m = if let Some(z) = def.zones.iter().find(|z| z.region.contains(p, seg, len)) {
            grades.get(&z.grade)?
        } else if let Some(&pi) = panel_of.get(&i) {
            grades.get(&def.panels[pi].grade)?
        } else {
            shell_grade
        };
        grade_of.insert(i, m);
    }
    // One tier: the shell's nodes are the frame's (members snap onto them,
    // `members::SNAP_FRAME`).
    if one {
        for &i in &shell.nodes {
            sk.nodes[i as usize].primary = true;
        }
    }
    // Panel extents along z (hinge and latch edges).
    let mut extent: HashMap<u16, (f32, f32)> = HashMap::new();
    for &i in &shell.nodes {
        let n = &sk.nodes[i as usize];
        if n.part != 0 {
            let e = extent.entry(n.part).or_insert((f32::MAX, f32::MIN));
            e.0 = e.0.min(n.p.z);
            e.1 = e.1.max(n.p.z);
        }
    }
    let mut group_ids: HashMap<u16, u32> = HashMap::new();
    // Detacher groups made so far.
    let mut groups_made = 0u32;
    for (a, b, kind) in shell.beams.clone() {
        // A brace spans a shared edge between the two triangles' opposite
        // vertices: on a panel (doors, bonnet, boot lid, fenders) it is the
        // panel's hinge beam, its out-of-plane stiffness (a single layer of
        // triangles is a membrane), and it yields, so a crease stays.
        let panel_brace = {
            let (qa, qb) = (sk.nodes[a as usize].part, sk.nodes[b as usize].part);
            qa == qb && qa != 0
        };
        if kind == ShellBeam::Brace && !def.shell_braces && !panel_brace {
            continue;
        }
        let (pa, pb) = (sk.pos(a), sk.pos(b));
        let zmid = 0.5 * (pa.z + pb.z);
        let crumple = def.crumple.eval(zmid);
        let mut m = grade_of[&a].weaker(grade_of[&b]);
        if kind == ShellBeam::Bend {
            // Bend beams carry the panel's bending, not its membrane: softer. They
            // are long (two bays), so cheap for the step.
            m = m.stiffer(0.7);
        }
        m = m.scaled(crumple);
        let (qa, qb) = (sk.nodes[a as usize].part, sk.nodes[b as usize].part);
        if qa == qb {
            sk.beam(a, b, m);
            continue;
        }
        if kind == ShellBeam::Bend {
            // A bend beam across a panel gap is not a seam.
            continue;
        }
        // A seam: the joint of the panel side (the non-body side, or the first).
        let (pn_node, other) = if panel_of.contains_key(&a) {
            (a, b)
        } else {
            (b, a)
        };
        let Some(&pi) = panel_of.get(&pn_node) else {
            sk.beam(a, b, m);
            continue;
        };
        let part = sk.nodes[pn_node as usize].part;
        let _ = other;
        match &def.panels[pi].joint {
            Joint::Weld => {
                sk.beam(a, b, m);
            }
            Joint::Bolt { scale, group } => {
                let g = if *group {
                    *group_ids.entry(part).or_insert_with(|| {
                        groups_made += 1;
                        groups_made
                    })
                } else {
                    0
                };
                sk.beam_with(a, b, m.scaled(*scale), BeamType::Normal, g);
            }
            Joint::Hinged {
                hinge,
                hinge_scale,
                latch_scale,
                latch_y,
                latch_x,
            } => {
                let (z0, z1) = extent[&part];
                let q = sk.pos(pn_node);
                let zone = 0.14;
                let at_hinge = match hinge {
                    Edge::Front => q.z > z1 - zone,
                    Edge::Rear => q.z < z0 + zone,
                };
                let at_latch_edge = match hinge {
                    Edge::Front => q.z < z0 + zone,
                    Edge::Rear => q.z > z1 - zone,
                };
                let in_latch = at_latch_edge
                    && latch_y.is_none_or(|r| q.y >= r[0] && q.y <= r[1])
                    && latch_x.is_none_or(|x| q.x.abs() <= x);
                // A panel with its own frame (doors) hangs on its frame's
                // hardware; its skin's edge is all seal: it bears (the shut
                // line keeps its gap under a push) and carries no tension.
                let framed = def
                    .frame_through
                    .iter()
                    .any(|t| sk.parts[part as usize].starts_with(t.as_str()));
                if framed {
                    // The shut faces (hinge and lock faces) bear on their
                    // pillars through the hemmed edge (the outer skin folded
                    // over the inner panel: two sheets); the rest is seal.
                    let face = at_hinge || at_latch_edge;
                    if face {
                        sk.beam_with(
                            a,
                            b,
                            Material::weld(m, grades.hsla),
                            BeamType::Support(SEAL_BREAK),
                            0,
                        );
                    } else {
                        sk.beam_with(a, b, m.scaled(0.5), BeamType::Support(SEAL_BREAK), 0);
                    }
                } else {
                    // A lid (bonnet, boot lid): it hangs on its hinges and
                    // latch (placed below); its edge bears on its seals.
                    let _ = (hinge_scale, latch_scale, in_latch);
                    let mm = if at_hinge || at_latch_edge {
                        Material::weld(m, grades.hsla)
                    } else {
                        m.scaled(0.5)
                    };
                    sk.beam_with(a, b, mm, BeamType::Support(SEAL_BREAK), 0);
                }
            }
        }
    }
    // The frame: hard points on every few rings joined into a braced tube.
    let glass = |i: u32| -> bool {
        let p = sk.pos(i);
        let seg = shell.seg[&i];
        def.zones
            .iter()
            .any(|z| z.grade == "glass" && z.region.contains(p, seg, len))
    };
    let frame_mat = grades.get(&def.frame_grade)?;
    // `frame_every` 0: no frame (a sparse shell is its own frame).
    let mut frame_rings: Vec<usize> = if def.frame_every == 0 {
        Vec::new()
    } else {
        (0..shell.rings.len()).step_by(def.frame_every).collect()
    };
    if !frame_rings.is_empty() && *frame_rings.last().unwrap() != shell.rings.len() - 1 {
        frame_rings.push(shell.rings.len() - 1);
    }
    let mut hard: Vec<Vec<u32>> = Vec::new();
    for &k in &frame_rings {
        let r = &shell.rings[k];
        let segs: Vec<Seg> = r.half.iter().map(|i| shell.seg[i]).collect();
        let x = |i: u32| sk.pos(i).x;
        let pick_x = |target: f32, want: &[Seg]| -> Option<u32> {
            r.half
                .iter()
                .copied()
                .filter(|i| want.contains(&shell.seg[i]))
                .min_by(|&a, &b| (x(a) - target).abs().total_cmp(&(x(b) - target).abs()))
        };
        let mut pts: Vec<u32> = vec![r.half[0]];
        if let Some(i) = pick_x(0.45, &[Seg::Floor]) {
            pts.push(i);
        }
        if let Some(j) = segs.iter().position(|s| *s == Seg::Lower) {
            pts.push(r.half[j]);
        }
        let side: Vec<u32> = r
            .half
            .iter()
            .copied()
            .filter(|i| shell.seg[i] == Seg::Side)
            .collect();
        if !side.is_empty() {
            pts.push(side[side.len() / 2]);
            pts.push(*side.last().unwrap());
        }
        if let Some(j) = segs.iter().position(|s| *s == Seg::Top) {
            pts.push(r.half[j]);
        }
        if let Some(i) = pick_x(0.3, &[Seg::Top]) {
            pts.push(i);
        }
        pts.push(*r.half.last().unwrap());
        pts.dedup();
        // Screens are not structure: drop their middle points (the pillar stays).
        let edge = segs.iter().position(|s| *s == Seg::Top).map(|j| r.half[j]);
        pts.retain(|&i| Some(i) == edge || !glass(i));
        // The whole ring: the right half, then the left half's mirrors back down.
        let mut ring: Vec<u32> = pts.clone();
        for &i in pts[1..pts.len() - 1].iter().rev() {
            ring.push(shell.mirror[&i]);
        }
        if pts.len() < 2 {
            ring = pts;
        }
        hard.push(ring);
    }
    let through: Vec<u16> = sk
        .parts
        .iter()
        .enumerate()
        .filter(|(_, n)| def.frame_through.iter().any(|t| n.starts_with(t.as_str())))
        .map(|(i, _)| i as u16)
        .collect();
    let same_part = |sk: &Skeleton, a: u32, b: u32| {
        let (pa, pb) = (sk.nodes[a as usize].part, sk.nodes[b as usize].part);
        pa == pb || (pa == 0 && through.contains(&pb)) || (pb == 0 && through.contains(&pa))
    };
    let fmat = |sk: &Skeleton, a: u32, b: u32| -> Material {
        let z = 0.5 * (sk.pos(a).z + sk.pos(b).z);
        let m = frame_mat.scaled(def.crumple.eval(z));
        // Into a closed door: it carries load until torn (a fifth of the frame's
        // thresholds), so a hard enough pull or hit takes the door off.
        if sk.nodes[a as usize].part != sk.nodes[b as usize].part {
            m.scaled(0.2)
        } else {
            m
        }
    };
    for (fi, ring) in hard.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            if same_part(&sk, a, b) {
                let m = fmat(&sk, a, b).stiffer(0.5);
                sk.set_beam(a, b, m, BeamType::Normal, 0);
            }
        }
        if fi + 1 < hard.len() {
            let next = &hard[fi + 1];
            // Corresponding points by their place around the section.
            let nearest_in = |sk: &Skeleton, p: Vec3, list: &[u32]| -> u32 {
                *list
                    .iter()
                    .min_by(|&&a, &&b| {
                        let da = Vec3::new(sk.pos(a).x - p.x, sk.pos(a).y - p.y, 0.0).length();
                        let db = Vec3::new(sk.pos(b).x - p.x, sk.pos(b).y - p.y, 0.0).length();
                        da.total_cmp(&db)
                    })
                    .unwrap()
            };
            for i in 0..n {
                let a = ring[i];
                let b = nearest_in(&sk, sk.pos(a), next);
                let a2 = ring[(i + 1) % n];
                let b2 = nearest_in(&sk, sk.pos(a2), next);
                for (j, (p, q)) in [(a, b), (a, b2), (a2, b)].into_iter().enumerate() {
                    if p != q && same_part(&sk, p, q) {
                        let m = fmat(&sk, p, q);
                        // Diagonals stop shear: a third of the stiffness.
                        let m = if j == 0 { m } else { m.stiffer(0.35) };
                        sk.set_beam(p, q, m, BeamType::Normal, 0);
                    }
                }
            }
        }
    }
    // Bulkheads where the frame closes: both ends and the firewall ring get
    // braces across the section.
    let ends: Vec<usize> = if hard.is_empty() {
        Vec::new()
    } else {
        vec![0, hard.len() - 1]
    };
    for fi in ends {
        let ring = &hard[fi];
        let n = ring.len();
        for i in 0..n {
            let j = (i + n / 2) % n;
            let (a, b) = (ring[i], ring[j]);
            if same_part(&sk, a, b) {
                let m = fmat(&sk, a, b);
                sk.beam(a, b, m);
            }
        }
    }
    let frame_nodes: Vec<u32> = hard.iter().flatten().copied().collect();
    // The tube's hard points are the frame tier's (heavy) nodes.
    for &i in &frame_nodes {
        sk.nodes[i as usize].primary = true;
    }
    let body_pool: Vec<u32> = shell
        .nodes
        .iter()
        .copied()
        .filter(|&i| sk.nodes[i as usize].part == 0)
        .collect();
    let mut groups: HashMap<String, Vec<u32>> = HashMap::new();
    // What ties a backed panel's skin to its own frame (doors).
    let backing = Material::steel(2.0e5, super::material::ZETA, 2.5e3, 0.3, 1.0)
        .plastic(super::material::PLASTIC);
    // Members.
    let mut structure_nodes: Vec<u32> = Vec::new();
    // Door members' end stations (part, nodes).
    let mut door_ends: Vec<(u16, Vec<u32>)> = Vec::new();
    let mut trusses: Vec<Truss> = Vec::new();
    // Each truss's steel per metre and part (one tier's masses).
    let mut truss_kgm: Vec<(f32, u16)> = Vec::new();
    // One tier: shell beams that are members' outer chords, and their webs
    // (beam, the chord's steel area m²).
    let mut lane_beams: Vec<(usize, f32)> = Vec::new();
    let mut truss_names: Vec<String> = Vec::new();
    let mut snap_pool: Vec<u32> = shell.nodes.clone();
    // One tier: the shell's bend beams (two bays long) in a member's lane
    // stay the sheet's bending: the lane's edges are its chord already, and
    // a chord twice over in parallel would carry twice its section's yield.
    // `LANE_BENDS=1` makes them chords too (a measurement switch).
    let shell_bends: std::collections::HashSet<(u32, u32)> = shell
        .beams
        .iter()
        .filter(|b| b.2 == ShellBeam::Bend)
        .map(|b| (b.0.min(b.1), b.0.max(b.1)))
        .collect();
    let lane_bends_stay = std::env::var("LANE_BENDS").map_or(true, |v| v != "1");
    for m in &def.members {
        let base = grades.get(&m.grade)?.stiffer(m.stiff);
        let sides: &[f32] = if m.mirrored { &[1.0, -1.0] } else { &[1.0] };
        for &side in sides {
            let up = Vec3::new(side * m.up[0], m.up[1], m.up[2]);
            let path: Vec<Vec3> = m
                .path
                .iter()
                .map(|p| Vec3::new(side * p[0], p[1], p[2]))
                .collect();
            // Part: the panel the member's middle lies in (door beams ride the door).
            let mid = path[path.len() / 2];
            // (A riding member's panel: the nearest panel node, a member at a
            // shut line rides the panel, not the body beside it.)
            let panels: Vec<u32> = shell
                .nodes
                .iter()
                .copied()
                .filter(|&i| {
                    let q = sk.nodes[i as usize].part;
                    // (A pane is its own part: members ride the panel
                    // round it, not the glass.)
                    q != 0 && !sk.parts[q as usize].starts_with(GLASS_PART)
                })
                .collect();
            let part = if m.ride {
                sk.nodes[sk.nearest(mid, &panels, 1)[0] as usize].part
            } else {
                0
            };
            // A member riding a panel meets only its own part's frame (it
            // must not weld the panel to the body across a shut line).
            let pool: Vec<u32> = if m.ride {
                snap_pool
                    .iter()
                    .copied()
                    .filter(|&i| sk.nodes[i as usize].part == part)
                    .collect()
            } else {
                snap_pool.clone()
            };
            let spec = m.clone();
            let mat = move |t: f32| base.scaled(spec.scale_at(t));
            // One tier: the shell nodes in the member's lane (below), known
            // before its truss so the truss leaves out what the lane already
            // is: its longitudinals and face diagonals between two lane
            // nodes (the lane's shell beams are that chord and that face).
            // `LANE_TRUSS=1` keeps them (a measurement switch).
            let lane_pre: std::collections::HashSet<u32> =
                if one && m.attach > 0 && std::env::var("LANE_TRUSS").map_or(true, |v| v != "1") {
                    let r = 0.5 * m.width.max(m.height) + LANE_REACH;
                    shell
                        .nodes
                        .iter()
                        .copied()
                        .filter(|&i| {
                            sk.nodes[i as usize].part == part
                                && path_distance(&path, sk.pos(i)).0 <= r
                        })
                        .collect()
                } else {
                    Default::default()
                };
            let truss: Truss = box_truss(
                &mut sk,
                &path,
                m.width,
                m.height,
                up,
                {
                    let s = if m.station > 0.0 {
                        m.station
                    } else if m.beads > 0 {
                        res.crush()
                    } else {
                        res.truss()
                    };
                    s
                },
                part,
                &mat,
                m.beads,
                m.section,
                m.catch && !one,
                &pool,
                &lane_pre,
            );
            snap_pool.extend(truss.stations.iter().flatten().copied());
            // The frame: every member's stations (not its catchers); a door's
            // beams are its own heavy frame.
            for &i in truss.stations.iter().flatten() {
                sk.nodes[i as usize].primary = !m.skin;
            }
            // Catchers too: a pole's whole load goes through their ties into
            // the member (scaled with the light skin they would let it pass).
            for &i in &truss.catchers {
                sk.nodes[i as usize].primary = CATCHERS_IN_FRAME;
            }
            let nodes = truss.nodes();
            trusses.push(truss.clone());
            truss_kgm.push((m.mass_per_m(), part));
            truss_names.push(if side > 0.0 || !m.mirrored {
                m.name.clone()
            } else {
                format!("{}_r", m.name)
            });
            if m.attach > 0 {
                // The stations (catchers ride their bays), but those snapped onto
                // the shell (already part of it).
                let on_shell: std::collections::HashSet<u32> =
                    shell.nodes.iter().copied().collect();
                // (One tier: only a station with no node on the shell, by
                // beams of their own length: what is on the shell is
                // welded to it already.)
                let nodes: Vec<u32> = truss
                    .stations
                    .iter()
                    .filter(|st| !one || !st.iter().any(|i| on_shell.contains(i)))
                    .flatten()
                    .copied()
                    .filter(|i| !on_shell.contains(i))
                    .collect();
                let pool: Vec<u32> = shell
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&i| sk.nodes[i as usize].part == part)
                    .collect();
                super::members::attach_with(
                    &mut sk,
                    &nodes,
                    &pool,
                    m.attach,
                    0.35,
                    // Welded along its flanges: the panel's stiffness, half the
                    // member's strength (the load path holds while the member
                    // folds).
                    Material {
                        deform: 0.5 * base.deform,
                        strength: 0.5 * base.strength,
                        plastic: base.plastic,
                        // A panel's own frame (doors) backs its skin.
                        ..if m.ride { backing } else { grades.panel }
                    },
                    !one,
                );
            }
            if m.ride {
                // A door beam's ends overlap the pillars it closes against:
                // pushed in, they bear on them (compression only, so the door
                // still opens and tears off), carrying a side hit into the
                // pillars and sill instead of the door skin.
                let body: Vec<u32> = structure_nodes
                    .iter()
                    .copied()
                    .filter(|&i| {
                        sk.nodes[i as usize].part == 0
                            && sk.nodes[i as usize].primary
                            && !sk.nodes[i as usize].catcher
                    })
                    .collect();
                for st in [truss.first(), truss.last()] {
                    let mut ends: Vec<u32> = st.to_vec();
                    ends.sort();
                    ends.dedup();
                    // Hinges and latch are placed once every door member is
                    // built (which end carries which is the door's).
                    if !m.hardware_ends {
                        continue;
                    }
                    if through.contains(&part) {
                        door_ends.push((part, ends.clone()));
                    }
                    for &a in &ends {
                        for b in sk.nearest(sk.pos(a), &body, 2) {
                            let l = sk.pos(a).distance(sk.pos(b));
                            // (A door end bears on the pillar beside it
                            // along the car, the aperture closing in a
                            // frontal; never mostly outward on what is
                            // outboard of it: that would hold it shut.)
                            let d = sk.pos(b) - sk.pos(a);
                            let outboard =
                                through.contains(&part) && d.x * sk.pos(a).x.signum() > d.z.abs();
                            if l <= 0.3 && !outboard {
                                let sc = l.max(super::material::REF_LENGTH)
                                    / super::skeleton::TIE_LENGTH.max(l);
                                // The mount grade's stiffness (the door is
                                // light), the beam's strength.
                                let m = Material {
                                    k: grades.mount.k,
                                    damp: grades.mount.damp,
                                    ..base
                                };
                                let m = Material {
                                    deform: DOOR_BEAR * m.deform,
                                    strength: DOOR_BEAR * m.strength,
                                    ..m
                                };
                                sk.beam_with(a, b, m.stiffer(sc), BeamType::Support(10.0), 0);
                            }
                        }
                    }
                }
            }
            // One tier: a member drawn on the body has the shell along its
            // path as its outer chord (the sill's outer panel, a pillar's
            // outer): the shell's beams in its lane, along its path, take
            // its chords' material, and each lane node is joined to its
            // nearest stations' nodes (the section's webs). The shell is at
            // most `POLE_LANE` apart, so a pole meets the member's chord in
            // every lane.
            let mut nodes = nodes;
            if one && m.attach > 0 {
                let r = 0.5 * m.width.max(m.height) + LANE_REACH;
                let dist = |q: Vec3| -> (f32, Vec3, f32) {
                    // (Distance to the path, its tangent there, its arc fraction.)
                    let mut best = (f32::MAX, Vec3::Z, 0.0);
                    let total: f32 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
                    let mut acc = 0.0;
                    for w in path.windows(2) {
                        let d = w[1] - w[0];
                        let l = d.length().max(1e-6);
                        let u = ((q - w[0]).dot(d) / (l * l)).clamp(0.0, 1.0);
                        let e = q.distance(w[0] + d * u);
                        if e < best.0 {
                            best = (e, d / l, (acc + u * l) / total.max(1e-6));
                        }
                        acc += l;
                    }
                    best
                };
                let lane: Vec<u32> = shell
                    .nodes
                    .iter()
                    .copied()
                    .filter(|&i| sk.nodes[i as usize].part == part && dist(sk.pos(i)).0 <= r)
                    .collect();
                let lane_set: std::collections::HashSet<u32> = lane.iter().copied().collect();
                let chords = if m.section == 2 { 2.0 } else { 4.0 };
                let area = m.mass_per_m() / super::members::STEEL_DENSITY / chords;
                for j in 0..sk.beams.len() {
                    let (a, b) = (sk.beams[j].a, sk.beams[j].b);
                    if !(lane_set.contains(&a) && lane_set.contains(&b))
                        || sk.beams[j].ty != BeamType::Normal
                        || sk.beams[j].group != 0
                        || sk.beams[j].tie
                        || (lane_bends_stay && shell_bends.contains(&(a.min(b), a.max(b))))
                    {
                        continue;
                    }
                    let mid = 0.5 * (sk.pos(a) + sk.pos(b));
                    let (_, tan, t) = dist(mid);
                    let d = (sk.pos(b) - sk.pos(a)).normalize_or_zero();
                    if d.dot(tan).abs() >= LANE_COS {
                        sk.beams[j].mat = mat(t);
                        lane_beams.push((j, area));
                    }
                }
                let stations: Vec<u32> = truss
                    .stations
                    .iter()
                    .flatten()
                    .copied()
                    .filter(|i| !lane_set.contains(i))
                    .collect();
                for &a in &lane {
                    let (_, _, t) = dist(sk.pos(a));
                    let web = mat(t)
                        .stiffer(super::members::DIAG)
                        .scaled(super::members::DIAG);
                    for b in sk.nearest(sk.pos(a), &stations, 2) {
                        if sk.pos(b).distance(sk.pos(a)) <= 0.6 && sk.beam(a, b, web) {
                            lane_beams.push((sk.beams.len() - 1, area));
                        }
                    }
                }
                nodes.extend(lane);
                nodes.sort();
                nodes.dedup();
            }
            let key = if side > 0.0 || !m.mirrored {
                m.name.clone()
            } else {
                format!("{}_r", m.name)
            };
            groups.insert(key, nodes.clone());
            structure_nodes.extend(nodes);
        }
    }
    // Door hardware and bearing. At each door's hinge edge the highest
    // member end carries the upper hinge and the lowest the lower hinge; at
    // its latch edge the end nearest the latch zone's middle carries the
    // latch. Each is one detacher group of ropes (tension only) to the
    // pillar nodes beside it; their strengths are set from the FMVSS 206
    // rigs once the stiffnesses are final. Every member end also bears on
    // its pillar's flange, inboard (compression only).
    let mut hardware: Vec<HardwareGroup> = Vec::new();
    {
        let body: Vec<u32> = structure_nodes
            .iter()
            .copied()
            .filter(|&i| {
                sk.nodes[i as usize].part == 0
                    && sk.nodes[i as usize].primary
                    && !sk.nodes[i as usize].catcher
            })
            .collect();
        let member_body: Vec<u32> = {
            let mut on: std::collections::HashSet<u32> = Default::default();
            for g in groups.values() {
                on.extend(g.iter().copied());
            }
            let mut v: Vec<u32> = body.iter().copied().filter(|i| on.contains(i)).collect();
            v.sort();
            v.dedup();
            v
        };
        let mut parts: Vec<u16> = door_ends.iter().map(|e| e.0).collect();
        parts.sort();
        parts.dedup();
        let centre =
            |sk: &Skeleton, v: &[u32]| v.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / v.len() as f32;
        // The flanged members' frame nodes, and each one's flange.
        let mut flange_of: HashMap<u32, FlangeSpec> = HashMap::new();
        for f in &def.flanges {
            let m = def
                .members
                .iter()
                .find(|m| m.name == f.member)
                .ok_or_else(|| format!("flange on `{}`: no such member", f.member))?;
            for key in [m.name.clone(), format!("{}_r", m.name)] {
                for &i in groups.get(&key).into_iter().flatten() {
                    if body.contains(&i) {
                        flange_of.entry(i).or_insert_with(|| f.clone());
                    }
                }
            }
        }
        let flanged: Vec<u32> = {
            let mut v: Vec<u32> = flange_of.keys().copied().collect();
            v.sort();
            v
        };
        let mut flange_nodes: Vec<(String, u32)> = Vec::new();
        for part in parts {
            let name = sk.parts[part as usize].clone();
            let Some(Joint::Hinged { hinge, latch_y, .. }) = def
                .panels
                .iter()
                .find(|p| name.starts_with(p.name.as_str()))
                .map(|p| p.joint.clone())
            else {
                return Err(format!("door part {name}: its frame needs a hinged joint"));
            };
            // (Members whose ends meet share their nodes: one end.)
            let mut ends: Vec<(Vec<u32>, Vec3)> = Vec::new();
            for e in door_ends.iter().filter(|e| e.0 == part) {
                if !ends.iter().any(|x| x.0 == e.1) {
                    ends.push((e.1.clone(), centre(&sk, &e.1)));
                }
            }
            let zmid = ends.iter().map(|e| e.1.z).sum::<f32>() / ends.len() as f32;
            let at_hinge = |c: Vec3| (c.z > zmid) == (hinge == Edge::Front);
            // Away from the pillar beyond the door's edge the end is at.
            let dl_of = |sk: &Skeleton, v: &[u32]| {
                if centre(sk, v).z > zmid {
                    -Vec3::Z
                } else {
                    Vec3::Z
                }
            };
            let hinge_ends: Vec<&(Vec<u32>, Vec3)> =
                ends.iter().filter(|e| at_hinge(e.1)).collect();
            let latch_ends: Vec<&(Vec<u32>, Vec3)> =
                ends.iter().filter(|e| !at_hinge(e.1)).collect();
            let by_y = |v: &[&(Vec<u32>, Vec3)], high: bool| -> Option<Vec<u32>> {
                v.iter()
                    .max_by(|a, b| {
                        let o = a.1.y.total_cmp(&b.1.y);
                        if high {
                            o
                        } else {
                            o.reverse()
                        }
                    })
                    .map(|e| e.0.clone())
            };
            let latch_mid = latch_y.map_or(0.75, |r| 0.5 * (r[0] + r[1]));
            let latch = latch_ends
                .iter()
                .min_by(|a, b| {
                    (a.1.y - latch_mid)
                        .abs()
                        .total_cmp(&(b.1.y - latch_mid).abs())
                })
                .map(|e| e.0.clone());
            let mut kinds: Vec<(Hardware, Vec<u32>)> = Vec::new();
            if let Some(v) = by_y(&hinge_ends, true) {
                kinds.push((Hardware::HingeUpper, v));
            }
            if let Some(v) = by_y(&hinge_ends, false) {
                kinds.push((Hardware::HingeLower, v));
            }
            if let Some(v) = latch {
                kinds.push((Hardware::Latch, v.clone()));
                kinds.push((Hardware::LatchSecondary, v));
            }
            for (kind, nodes) in kinds {
                groups_made += 1;
                let g = groups_made;
                let mut beams = Vec::new();
                // A hinge's pin (and the latch's striker) is one point: the
                // end's outer node, so the door turns about its hinge line
                // instead of the hardware holding its end square.
                let pin = *nodes
                    .iter()
                    .max_by(|&&a, &&b| sk.pos(a).x.abs().total_cmp(&sk.pos(b).x.abs()))
                    .unwrap();
                for &a in &[pin] {
                    // The pillar nodes past the two the end bears on: the
                    // primary's the next two, the secondary's the one after.
                    // The pillar nodes (not the two it bears on) that hold
                    // it both ways: inboard (the door pulled open) and
                    // toward the pillar (pulled along the car), the best
                    // two by the lesser of the two cosines; the secondary's
                    // the next.
                    let pa = sk.pos(a);
                    // (The pillar is beyond the door's edge.)
                    let toward = if centre(&sk, &nodes).z > zmid {
                        1.0
                    } else {
                        -1.0
                    };
                    let (dl, dt) = (Vec3::Z * -toward, Vec3::X * pa.x.signum());
                    // One tier: on the members' nodes only (pillar, sill,
                    // roof rail, cross members), not the shell's plain
                    // nodes, so the hardware lands on structure wherever
                    // the shell's lattice puts its nodes.
                    let near = if one {
                        sk.nearest(pa, &member_body, 12)
                    } else {
                        sk.nearest(pa, &body, 12)
                    };
                    // (The secondary's catch reaches a little farther.)
                    let reach = if kind == Hardware::LatchSecondary {
                        0.45
                    } else {
                        0.35
                    };
                    let bears: Vec<u32> = near.iter().copied().take(2).collect();
                    let pool: Vec<u32> = near
                        .into_iter()
                        .filter(|&b| {
                            !bears.contains(&b)
                                && !sk.has_beam(a, b)
                                && sk.pos(b).distance(pa) <= reach
                        })
                        .collect();
                    // Three load paths, each along one pull: toward the
                    // pillar (pulled along the car), inboard (pulled open)
                    // and up (the door's weight), so a pull meets the beam
                    // it was sized for whatever the end does across it.
                    let best = |dir: Vec3, skip: &[u32]| -> Option<u32> {
                        pool.iter()
                            .copied()
                            .filter(|b| !skip.contains(b))
                            .map(|b| ((sk.pos(b) - pa).normalize().dot(dir), b))
                            .filter(|c| c.0 > 0.3)
                            .max_by(|x, y| x.0.total_cmp(&y.0))
                            .map(|c| c.1)
                    };
                    let mut pick: Vec<u32> = Vec::new();
                    let order = if kind == Hardware::LatchSecondary {
                        vec![-dl, -dt]
                    } else {
                        vec![-dl, -dt, Vec3::Y]
                    };
                    for d in order {
                        if let Some(b) = best(d, &pick) {
                            pick.push(b);
                        }
                    }
                    for b in pick {
                        let l = sk.pos(a).distance(sk.pos(b));
                        let sc =
                            l.max(super::material::REF_LENGTH) / super::skeleton::TIE_LENGTH.max(l);
                        let sc = if kind == Hardware::LatchSecondary {
                            sc * SECONDARY_STIFF
                        } else {
                            sc
                        };
                        if sk.beam_with(a, b, grades.mount.stiffer(sc), BeamType::Rope, g) {
                            beams.push(sk.beams.len() - 1);
                        }
                    }
                }
                if beams.is_empty() {
                    let pa = sk.pos(pin);
                    let near: Vec<String> = sk
                        .nearest(pa, if one { &member_body } else { &body }, 4)
                        .iter()
                        .map(|&b| {
                            let g = groups
                                .iter()
                                .find(|g| g.1.contains(&b))
                                .map_or("-", |g| g.0.as_str());
                            format!("{b} {g} {:.2} m", sk.pos(b).distance(pa))
                        })
                        .collect();
                    return Err(format!(
                        "{name}: no pillar node near its {kind:?} (pin at {pa:?}; nearest {near:?})"
                    ));
                }
                let suffix = match kind {
                    Hardware::HingeUpper => "hinge_upper",
                    Hardware::HingeLower => "hinge_lower",
                    Hardware::Latch => "latch",
                    Hardware::LatchSecondary => "latch_secondary",
                };
                hardware.push(HardwareGroup {
                    name: format!("{name}.{suffix}"),
                    part,
                    kind,
                    group: g,
                    beams,
                    along: dl_of(&sk, &nodes),
                    outward: Vec3::X * sk.pos(pin).x.signum(),
                    weight: true,
                });
            }
            // Bearing: at each of the door's edges one flange node of the
            // nearest pillar, `FLANGE_GAP` inboard of the ends' inner nodes;
            // each member end's outer node bears on it (one flange strip a
            // member end).
            for hinge_edge in [true, false] {
                let edge: Vec<&(Vec<u32>, Vec3)> = ends
                    .iter()
                    .filter(|e| at_hinge(e.1) == hinge_edge)
                    .collect();
                if edge.is_empty() || flanged.is_empty() {
                    continue;
                }
                let abs_x = |sk: &Skeleton, i: u32| sk.pos(i).x.abs();
                let side = edge[0].1.x.signum();
                let xin = edge
                    .iter()
                    .flat_map(|e| e.0.iter().copied())
                    .map(|i| abs_x(&sk, i))
                    .fold(f32::MAX, f32::min);
                // The flange is a strip down the pillar behind the door's
                // edge: a node of it behind each member end, welded to the
                // pillar and to its neighbours along the strip.
                let mut strip: Vec<u32> = Vec::new();
                for (nodes, c) in &edge {
                    let pf = Vec3::new(side * (xin - FLANGE_GAP), c.y, c.z);
                    let flange = flange_of[&sk.nearest(pf, &flanged, 1)[0]].clone();
                    let root: Vec<u32> = sk
                        .nearest(pf, &body, 4)
                        .into_iter()
                        .filter(|&r| sk.pos(r).distance(pf) <= 0.35)
                        .collect();
                    // (Three or more not in one plane with it, or this end
                    // has no flange behind it.)
                    let solid = root.len() >= 3 && {
                        let q: Vec<Vec3> = root.iter().map(|&r| sk.pos(r) - pf).collect();
                        (0..q.len()).any(|i| {
                            (i + 1..q.len()).any(|j| {
                                (j + 1..q.len()).any(|k| q[i].cross(q[j]).dot(q[k]).abs() > 1e-4)
                            })
                        })
                    };
                    if !solid {
                        continue;
                    }
                    let grade = &def
                        .members
                        .iter()
                        .find(|m| m.name == flange.member)
                        .unwrap()
                        .grade;
                    let base = grades.get(grade)?;
                    let cap = flange.capacity() * params.strength;
                    let f = sk.node(pf, 0);
                    sk.nodes[f as usize].primary = true;
                    // Welded to the pillar along its root, at the bearing's
                    // stiffness (in series with it).
                    let weld = Material {
                        k: grades.mount.k,
                        damp: grades.mount.damp,
                        deform: cap,
                        strength: cap + (base.strength - base.deform),
                        ..base
                    };
                    for r in root {
                        sk.tie(f, r, weld);
                    }
                    if let Some(&prev) = strip.last() {
                        sk.tie(f, prev, weld);
                    }
                    strip.push(f);
                    // The end's chords bear on it (pushed in, only).
                    for &a in nodes.iter() {
                        let l = sk.pos(a).distance(pf);
                        let sc =
                            l.max(super::material::REF_LENGTH) / super::skeleton::TIE_LENGTH.max(l);
                        let bear = Material {
                            k: grades.mount.k,
                            damp: grades.mount.damp,
                            deform: cap,
                            strength: cap + (base.strength - base.deform),
                            ..base
                        };
                        sk.beam_with(a, f, bear.stiffer(sc), BeamType::Support(SEAL_BREAK), 0);
                    }
                    flange_nodes.push((format!("{name}_flanges"), f));
                }
            }
        }
        for (k, f) in flange_nodes {
            groups.entry(k).or_default().push(f);
        }
        // Lids (bonnet, boot lid): a hinge each side of the hinge edge and a
        // latch at the middle of the other, each one pin (a lid node) on
        // ropes to the body beside it: toward the body edge (pulled along
        // the car) and down (lifted).
        let pool: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| sk.nodes[i as usize].part == 0 && !sk.nodes[i as usize].catcher)
            .collect();
        for part in 1..sk.parts.len() as u16 {
            let name = sk.parts[part as usize].clone();
            let Some(Joint::Hinged { hinge, .. }) = def
                .panels
                .iter()
                .find(|p| name.starts_with(p.name.as_str()))
                .map(|p| p.joint.clone())
            else {
                continue;
            };
            if through.contains(&part) {
                continue;
            }
            let lid: Vec<u32> = shell
                .nodes
                .iter()
                .copied()
                .filter(|&i| sk.nodes[i as usize].part == part)
                .collect();
            let (z0, z1) = extent[&part];
            let zone = 0.14;
            let hinge_edge = |z: f32| match hinge {
                Edge::Front => z > z1 - zone,
                Edge::Rear => z < z0 + zone,
            };
            let hinge_side: Vec<u32> = lid
                .iter()
                .copied()
                .filter(|&i| hinge_edge(sk.pos(i).z))
                .collect();
            let latch_side: Vec<u32> = lid
                .iter()
                .copied()
                .filter(|&i| !hinge_edge(sk.pos(i).z))
                .collect();
            // Away from the body at each edge: the hinge edge's body lies
            // beyond the hinge edge, the latch's beyond the other.
            let toward_hinge = if hinge == Edge::Front {
                Vec3::Z
            } else {
                -Vec3::Z
            };
            let farthest = |v: &[u32], sign: f32| -> Option<u32> {
                v.iter()
                    .copied()
                    .filter(|&i| sk.pos(i).x * sign > 0.0)
                    .max_by(|&a, &b| (sign * sk.pos(a).x).total_cmp(&(sign * sk.pos(b).x)))
            };
            let latch_pin = latch_side
                .iter()
                .copied()
                .filter(|&i| {
                    let z = sk.pos(i).z;
                    if hinge == Edge::Front {
                        z < z0 + zone
                    } else {
                        z > z1 - zone
                    }
                })
                // (The latch sits low on the lid's edge, on the centre line.)
                .min_by(|&a, &b| {
                    let k = |i: u32| (sk.pos(i).x.abs() > 0.15, sk.pos(i).y);
                    let (ka, kb) = (k(a), k(b));
                    ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
                });
            let pins = [
                (
                    Hardware::HingeUpper,
                    "hinge_l",
                    farthest(&hinge_side, 1.0),
                    -toward_hinge,
                ),
                (
                    Hardware::HingeLower,
                    "hinge_r",
                    farthest(&hinge_side, -1.0),
                    -toward_hinge,
                ),
                (Hardware::Latch, "latch", latch_pin, toward_hinge),
            ];
            for (kind, suffix, pin, along) in pins {
                let Some(a) = pin else {
                    return Err(format!("{name}: no node for its {suffix}"));
                };
                let pa = sk.pos(a);
                let near: Vec<u32> = sk
                    .nearest(pa, &pool, 16)
                    .into_iter()
                    .filter(|&b| sk.pos(b).distance(pa) <= 0.45)
                    .collect();
                let best = |dir: Vec3, skip: &[u32]| -> Option<u32> {
                    near.iter()
                        .copied()
                        .filter(|b| !skip.contains(b))
                        .map(|b| ((sk.pos(b) - pa).normalize().dot(dir), b))
                        .filter(|c| c.0 > 0.3)
                        .max_by(|x, y| x.0.total_cmp(&y.0))
                        .map(|c| c.1)
                };
                let mut pick: Vec<u32> = Vec::new();
                for d in [-along, -Vec3::Y] {
                    if let Some(b) = best(d, &pick) {
                        pick.push(b);
                    }
                }
                groups_made += 1;
                let g = groups_made;
                let mut beams = Vec::new();
                for b in pick {
                    let l = pa.distance(sk.pos(b));
                    let sc =
                        l.max(super::material::REF_LENGTH) / super::skeleton::TIE_LENGTH.max(l);
                    // (A seal already there becomes the hardware.)
                    sk.set_beam(a, b, grades.mount.stiffer(sc), BeamType::Rope, g);
                    beams.push(sk.beam_index(a, b).unwrap());
                }
                if beams.is_empty() {
                    let near: Vec<String> = sk
                        .nearest(pa, &pool, 8)
                        .iter()
                        .map(|&b| format!("{b} {:.2?} {}", sk.pos(b), sk.has_beam(a, b)))
                        .collect();
                    return Err(format!(
                        "{name}: no body node to hold its {suffix} (pin {a} at {pa:.2?}; {near:?})"
                    ));
                }
                hardware.push(HardwareGroup {
                    name: format!("{name}.{suffix}"),
                    part,
                    kind,
                    group: g,
                    beams,
                    along,
                    outward: Vec3::Y,
                    weight: false,
                });
            }
        }
    }
    // One tier: a rigidifier node under each lid (BeamNG's documented
    // practice for flat panels, "a rigidifier node offset from the surface
    // and connected to every panel node"). A real lid is a bonded double
    // shell, outer and inner frame, whose depth gives it bending stiffness;
    // the lattice's single sheet has none but its curvature's, so in the
    // 56 km/h wall the boot lid's free corners flapped up 160–280 mm. The
    // node sits `RIGIDIFIER_DEPTH` under the lid's centroid (a Choice: the
    // inner frame's 100 mm hat), joined to every skin node of the lid by
    // beams of the lid's own sheet material. `RIGIDIFIER=0`: none.
    if one && std::env::var("RIGIDIFIER").map_or(true, |v| v != "0") {
        let on_shell: std::collections::HashSet<u32> = shell.nodes.iter().copied().collect();
        for name in ["bonnet", "bootlid"] {
            let Some(pt) = sk.parts.iter().position(|p| p == name) else {
                continue;
            };
            let pt = pt as u16;
            let skin: Vec<u32> = (0..sk.nodes.len() as u32)
                .filter(|&i| sk.nodes[i as usize].part == pt && on_shell.contains(&i))
                .collect();
            if skin.len() < 4 {
                continue;
            }
            // The lid's own sheet beam (its median stiffness) as the material.
            let mut mats: Vec<Material> = sk
                .beams
                .iter()
                .filter(|b| {
                    b.ty == BeamType::Normal
                        && b.group == 0
                        && !b.damper
                        && skin.contains(&b.a)
                        && skin.contains(&b.b)
                })
                .map(|b| b.mat)
                .collect();
            if mats.is_empty() {
                continue;
            }
            mats.sort_by(|a, b| a.k.total_cmp(&b.k));
            let mat = mats[mats.len() / 2];
            let c = skin.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / skin.len() as f32;
            let r = sk.node(c - Vec3::Y * RIGIDIFIER_DEPTH, pt);
            for &i in &skin {
                sk.beam(r, i, mat);
            }
            groups.insert(format!("{name}_rigidifier"), vec![r]);
        }
    }
    // Backed panels: each skin node tied to the part's frame behind it.
    {
        let parts: Vec<u16> = sk
            .parts
            .iter()
            .enumerate()
            .filter(|(_, n)| def.backed.iter().any(|t| n.starts_with(t.as_str())))
            .map(|(i, _)| i as u16)
            .collect();
        let on_shell: std::collections::HashSet<u32> = shell.nodes.iter().copied().collect();
        // (One tier: the members behind the panel, not the shell beside it.)
        let member_set: std::collections::HashSet<u32> = structure_nodes.iter().copied().collect();
        let body_frame: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| {
                sk.nodes[i as usize].part == 0
                    && sk.nodes[i as usize].primary
                    && !sk.nodes[i as usize].catcher
                    && (!one || (member_set.contains(&i) && !on_shell.contains(&i)))
            })
            .collect();
        for &pt in &parts {
            // Its own frame (a door's beams and rails, a lid's inner frame):
            // the skin is tied to it, within the part.
            let own: Vec<u32> = (0..sk.nodes.len() as u32)
                .filter(|&i| sk.nodes[i as usize].part == pt && !on_shell.contains(&i))
                .collect();
            let skin: Vec<u32> = shell
                .nodes
                .iter()
                .copied()
                .filter(|&i| {
                    sk.nodes[i as usize].part == pt && (one || !sk.nodes[i as usize].primary)
                })
                .collect();
            // (One tier: the panel's members are drawn on its skin; no ties.)
            if !own.is_empty() && !one {
                attach(&mut sk, &skin, &own, 2, 0.3, backing);
            }
            // The body's structure behind it (apron, tower, radiator support,
            // bumper beam) it bears on: compression only, so it holds the
            // panel's push but none of its pull (only its hardware does).
            if !through.contains(&pt) {
                for &a in &skin {
                    for b in sk.nearest(sk.pos(a), &body_frame, 2) {
                        if sk.pos(b).distance(sk.pos(a)) <= 0.3 {
                            let l = sk.pos(a).distance(sk.pos(b));
                            let sc = l.max(super::material::REF_LENGTH)
                                / super::skeleton::TIE_LENGTH.max(l);
                            sk.beam_with(
                                a,
                                b,
                                backing.stiffer(sc),
                                BeamType::Support(SEAL_BREAK),
                                0,
                            );
                        }
                    }
                }
            }
        }
    }
    // Members meet where they touch: a station within reach of another member's
    // station is joined to it rigidly (every pair of their nodes: a gusset).
    {
        let stations: Vec<(usize, Vec<u32>, Vec3, bool)> = trusses
            .iter()
            .enumerate()
            .flat_map(|(mi, t)| {
                let n = t.stations.len();
                t.stations
                    .iter()
                    .enumerate()
                    .map(|(si, st)| {
                        let mut nodes: Vec<u32> = st.to_vec();
                        nodes.sort();
                        nodes.dedup();
                        (mi, nodes, Vec3::ZERO, si == 0 || si + 1 == n)
                    })
                    .collect::<Vec<_>>()
            })
            .map(|(mi, nodes, _, end)| {
                let c = nodes.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / nodes.len() as f32;
                (mi, nodes, c, end)
            })
            .collect();
        let member_nodes: Vec<std::collections::HashSet<u32>> = trusses
            .iter()
            .map(|t| t.stations.iter().flatten().copied().collect())
            .collect();
        let mat = grades.uhss.stiffer(GUSSET);
        for (ia, a) in stations.iter().enumerate() {
            // A member's end butts into another: join its end stations.
            if !a.3 {
                continue;
            }
            let reach = 0.3;
            let mut best: Vec<(f32, usize)> = stations
                .iter()
                .enumerate()
                .filter(|(_, b)| b.0 != a.0)
                .map(|(ib, b)| (b.2.distance(a.2), ib))
                .filter(|&(d, _)| d < reach)
                .collect();
            best.sort_by(|x, y| x.0.total_cmp(&y.0));
            // The nearest station of each other member.
            let mut seen = std::collections::HashSet::new();
            for (_, ib) in best {
                let b = &stations[ib];
                if !seen.insert(b.0) {
                    continue;
                }
                // Already joined: the member snapped onto the other's nodes.
                if member_nodes[a.0]
                    .iter()
                    .any(|i| member_nodes[b.0].contains(i))
                {
                    continue;
                }
                if sk.nodes[a.1[0] as usize].part != sk.nodes[b.1[0] as usize].part {
                    continue;
                }
                // Each node to its two nearest of the other station (a gusset
                // without the full 16-beam web).
                // (A two-node end gets two each, so it is not a hinge.)
                let per = 2;
                for &p in &a.1 {
                    for q in sk.nearest(sk.pos(p), &b.1, per) {
                        sk.tie(p, q, mat);
                    }
                }
                let _ = ia;
            }
        }
    }
    // A frame node's member station (its section's nodes): a lumped panel
    // bears on the whole section, not one corner of it.
    let mut station_of: HashMap<u32, (usize, Vec<u32>)> = HashMap::new();
    for (ti, t) in trusses.iter().enumerate() {
        for st in &t.stations {
            let mut v: Vec<u32> = st.to_vec();
            v.sort();
            v.dedup();
            for &i in &v {
                station_of.entry(i).or_insert_with(|| (ti, v.clone()));
            }
        }
    }
    let grades_strength = params.strength;
    // Shear panels (the members' and shell's body nodes).
    for br in &def.braces {
        let mat = grades.get(&br.grade)?;
        // The frame's nodes (a shear panel's corners are the frame's).
        let is_frame: Vec<bool> = sk
            .nodes
            .iter()
            .map(|n| n.part == 0 && n.primary && !n.catcher)
            .collect();
        let frame = |i: u32| is_frame[i as usize];
        let body: Vec<u32> = (0..sk.nodes.len() as u32).filter(|&i| frame(i)).collect();
        // A named member's trusses: (its left or centre one, its right one).
        let member = |name: &str| -> Result<(usize, usize), String> {
            let find = |n: &str| truss_names.iter().position(|x| x == n);
            let l = find(name).ok_or_else(|| format!("brace {}: no member {name}", br.name))?;
            Ok((l, find(&format!("{name}_r")).unwrap_or(l)))
        };
        let on = [&br.on[0], &br.on[1]].map(|n| (!n.is_empty()).then(|| member(n)).transpose());
        let on = [on[0].clone()?, on[1].clone()?];
        for side in [1.0f32, -1.0] {
            // Each point's node is found on the +x side and mirrored (ties in
            // distance would otherwise pick different nodes on each side, and
            // braces sharing a corner would miss each other). With its member
            // named, among that member's nodes, and its end is that member's
            // station through it: nothing built elsewhere moves it.
            let pick = |q: [f32; 3], on: Option<(usize, usize)>| -> (u32, Vec<u32>) {
                let pool = |ti: usize| -> Vec<u32> {
                    let mut v: Vec<u32> = trusses[ti]
                        .stations
                        .iter()
                        .flatten()
                        .copied()
                        .filter(|&i| frame(i))
                        .collect();
                    v.sort();
                    v.dedup();
                    v
                };
                let right = side * q[0].signum() < 0.0;
                let (left_pool, side_pool) = match on {
                    Some((l, r)) => (pool(l), pool(if right { r } else { l })),
                    None => (body.clone(), body.clone()),
                };
                let left = Vec3::new(q[0].abs(), q[1], q[2]);
                let n = sk.nearest(left, &left_pool, 1)[0];
                let want = Vec3::new(
                    sk.pos(n).x * (side * q[0].signum()),
                    sk.pos(n).y,
                    sk.pos(n).z,
                );
                let i = sk.nearest(want, &side_pool, 1)[0];
                let station = match on {
                    Some((l, r)) => {
                        let ti = if right { r } else { l };
                        let st = trusses[ti]
                            .stations
                            .iter()
                            .find(|st| st.contains(&i))
                            .unwrap();
                        let mut v = st.to_vec();
                        v.sort();
                        v.dedup();
                        v
                    }
                    None => station_of
                        .get(&i)
                        .map(|x| x.1.clone())
                        .unwrap_or_else(|| vec![i]),
                };
                (i, station)
            };
            let (a, ea) = pick(br.from, on[0]);
            let (b, eb) = pick(br.to, on[1]);
            if std::env::var("SHOWCASE_BRACES").is_ok() {
                let deg = |i: u32| sk.beams.iter().filter(|x| x.a == i || x.b == i).count();
                let owner = |i: u32| {
                    station_of
                        .get(&i)
                        .map(|x| truss_names[x.0].clone())
                        .unwrap_or_default()
                };
                eprintln!(
                    "brace {} {side}: {a} at {:.2?} (degree {}, {}) to {b} at {:.2?} (degree {}, {}); ends {ea:?} {eb:?}",
                    br.name,
                    sk.pos(a),
                    deg(a),
                    owner(a),
                    sk.pos(b),
                    deg(b),
                    owner(b)
                );
            }
            let share = 1.0 / (ea.len() * eb.len()) as f32;
            for &p in &ea {
                for &q in &eb {
                    if p == q || ea.contains(&q) || eb.contains(&p) {
                        continue;
                    }
                    let mut m = mat
                        .stiffer(share * br.kl / mat.kl())
                        .scaled(share * br.kl / mat.kl())
                        .scaled(def.crumple.eval(0.5 * (sk.pos(p).z + sk.pos(q).z)));
                    if br.strength > 0.0 {
                        m.deform = share * br.strength * grades_strength;
                        m.strength = 4.0 * m.deform;
                    }
                    sk.beam(p, q, m);
                }
            }
        }
    }
    // Shear panels between members.
    for sp in &def.shear {
        let mat = grades.get(&sp.grade)?;
        let find = |n: &str| truss_names.iter().position(|x| x == n);
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for suffix in ["", "_r"] {
            let na = format!("{}{suffix}", sp.a);
            // `b` may name the other side itself (`rail_r` from `rail`).
            let nb = if sp.b.ends_with("_r") || sp.b.ends_with("_l") {
                if suffix.is_empty() {
                    sp.b.clone()
                } else {
                    continue;
                }
            } else {
                format!("{}{suffix}", sp.b)
            };
            match (find(&na), find(&nb)) {
                (Some(a), Some(b)) => pairs.push((a, b)),
                // A centre member (not mirrored) against a mirrored one.
                (None, Some(b)) if !suffix.is_empty() => {
                    if let Some(a) = find(&sp.a) {
                        pairs.push((a, b));
                    }
                }
                (Some(a), None) if !suffix.is_empty() => {
                    if let Some(b) = find(&sp.b) {
                        pairs.push((a, b));
                    }
                }
                _ => {}
            }
        }
        if pairs.is_empty() {
            return Err(format!("shear panel {}–{}: no such members", sp.a, sp.b));
        }
        for (ia, ib) in pairs {
            let centre = |st: &[u32; 4]| st.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / 4.0;
            let sa: Vec<Vec3> = trusses[ia].stations.iter().map(centre).collect();
            let sb: Vec<Vec3> = trusses[ib].stations.iter().map(centre).collect();
            let near_b = |p: Vec3| -> usize {
                (0..sb.len())
                    .min_by(|&x, &y| {
                        (sb[x][sp.axis] - p[sp.axis])
                            .abs()
                            .total_cmp(&(sb[y][sp.axis] - p[sp.axis]).abs())
                    })
                    .unwrap()
            };
            // The node of station `st` nearest to `toward`.
            let facing = |sk: &Skeleton, st: &[u32; 4], toward: Vec3| -> u32 {
                *st.iter()
                    .min_by(|&&x, &&y| {
                        sk.pos(x)
                            .distance(toward)
                            .total_cmp(&sk.pos(y).distance(toward))
                    })
                    .unwrap()
            };
            for i in 0..sa.len().saturating_sub(1) {
                let (j0, j1) = (near_b(sa[i]), near_b(sa[i + 1]));
                let (a0, a1) = (trusses[ia].stations[i], trusses[ia].stations[i + 1]);
                let (b0, b1) = (trusses[ib].stations[j0], trusses[ib].stations[j1]);
                let add = |sk: &mut Skeleton, sa_: [u32; 4], ca: Vec3, sb_: [u32; 4], cb: Vec3| {
                    let p = facing(sk, &sa_, cb);
                    let q = facing(sk, &sb_, ca);
                    if p != q && sk.pos(p).distance(sk.pos(q)) > 0.15 {
                        let mut m = mat
                            .stiffer(sp.kl / mat.kl())
                            .scaled(sp.kl / mat.kl())
                            .scaled(def.crumple.eval(0.5 * (sk.pos(p).z + sk.pos(q).z)));
                        if sp.strength > 0.0 {
                            m.deform = sp.strength * params.strength;
                            m.strength = 4.0 * m.deform;
                        }
                        sk.beam(p, q, m);
                    }
                };
                add(&mut sk, a0, sa[i], b1, sb[j1]);
                add(&mut sk, a1, sa[i + 1], b0, sb[j0]);
                if j0 == j1 {
                    // Both ends on one station of `b`: the bay's own diagonal.
                    add(&mut sk, a0, sa[i], b0, sb[j0]);
                    add(&mut sk, a1, sa[i + 1], b0, sb[j0]);
                }
            }
        }
    }
    // Shear webs.
    for (na, nb, reach) in &def.webs {
        for suffix in ["", "_r"] {
            let (Some(ia), Some(ib)) = (
                truss_names
                    .iter()
                    .position(|n| *n == format!("{na}{suffix}")),
                truss_names
                    .iter()
                    .position(|n| *n == format!("{nb}{suffix}")),
            ) else {
                continue;
            };
            let web = grades.hsla.stiffer(0.5);
            for sa in &trusses[ia].stations {
                let ca = sa.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / 4.0;
                let best = trusses[ib]
                    .stations
                    .iter()
                    .map(|sb| (sb.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / 4.0, sb))
                    .min_by(|x, y| x.0.distance(ca).total_cmp(&y.0.distance(ca)));
                if let Some((cb, sb)) = best {
                    if cb.distance(ca) <= *reach {
                        for &p in sa {
                            for q in sk.nearest(sk.pos(p), sb, 2) {
                                sk.tie(p, q, web);
                            }
                        }
                    }
                }
            }
        }
    }
    // What parts bolt to: the structure (trusses and the frame's hard points of
    // the body), never the light skin.
    let mut chassis: Vec<u32> = structure_nodes
        .iter()
        .copied()
        .filter(|&i| {
            sk.nodes[i as usize].part == 0
                && sk.nodes[i as usize].primary
                && !sk.nodes[i as usize].catcher
        })
        .collect();
    chassis.extend(
        frame_nodes
            .iter()
            .copied()
            .filter(|&i| sk.nodes[i as usize].part == 0),
    );
    let _ = &body_pool;
    // Plates (radiator).
    for pl in &def.plates {
        let part = sk.part(&pl.name);
        let mat = grades.get(&pl.grade)?;
        let g = plate(
            &mut sk,
            Vec3::from_array(pl.corner),
            Vec3::from_array(pl.u),
            Vec3::from_array(pl.v),
            PLATE_SPACING,
            part,
            mat,
        );
        let nodes: Vec<u32> = g.iter().flatten().copied().collect();
        sk.add_mass(&nodes, pl.kg);
        let edge: Vec<u32> = nodes.clone();
        // Mounts at the mount grade's stiffness (kept as built: the plate's
        // own mass steadies it), its grade's strength.
        let before = sk.beams.len();
        let m = Material {
            k: grades.mount.k,
            damp: grades.mount.damp,
            ..mat.scaled(pl.mount_scale)
        };
        attach(&mut sk, &edge, &chassis, 3, 0.45, m);
        for bm in &mut sk.beams[before..] {
            bm.fixed = true;
        }
        groups.insert(pl.name.clone(), nodes);
    }
    // Blocks.
    for bl in &def.blocks {
        let part = sk.part(&bl.name);
        let nodes = block_spaced(
            &mut sk,
            Vec3::from_array(bl.centre),
            Vec3::from_array(bl.half),
            Vec3::new(
                bl.spacing,
                bl.spacing,
                if bl.depth > 0.0 { bl.depth } else { bl.spacing },
            ),
            part,
            grades.rigid.stiffer(bl.stiff),
        );
        sk.add_mass(&nodes, bl.kg);
        let mount = grades.get(&bl.mount_grade)?.scaled(bl.mount_scale);
        let c = Vec3::from_array(bl.centre);
        let h = Vec3::from_array(bl.half);
        let outer: Vec<u32> = nodes
            .iter()
            .copied()
            .filter(|&i| {
                let d = (sk.pos(i) - c).abs();
                d.x > h.x - 1e-3 || d.y > h.y - 1e-3 || d.z > h.z - 1e-3
            })
            .collect();
        // Mounts from the block's outer nodes to the structure within reach.
        let before = sk.beams.len();
        attach(&mut sk, &outer, &chassis, bl.mounts, bl.reach, mount);
        // Kept as built (the block's mass steadies them).
        for bm in &mut sk.beams[before..] {
            bm.fixed = true;
        }
        if bl.bear > 0.0 {
            // The rear face on what is behind it, the side faces on what is
            // beside them (rails, subframe, towers): compression-only beams
            // that never tear on stretch, so a block knocked loose still meets
            // the structure instead of passing through it.
            // Stiff enough to carry the block, strong as the structure.
            let bear = Material {
                k: grades.mount.k,
                damp: grades.mount.damp,
                ..grades.uhss.scaled(BEAR_SCALE)
            };
            for face in [-Vec3::Z, Vec3::X, -Vec3::X] {
                let lim = c.dot(face) + h.dot(face.abs());
                let on: Vec<u32> = outer
                    .iter()
                    .copied()
                    .filter(|&i| sk.pos(i).dot(face) > lim - 1e-3)
                    .collect();
                let beyond: Vec<u32> = chassis
                    .iter()
                    .copied()
                    .filter(|&j| sk.pos(j).dot(face) > lim)
                    .collect();
                for &a in &on {
                    for b in sk.nearest(sk.pos(a), &beyond, 2) {
                        let l = sk.pos(a).distance(sk.pos(b));
                        if l <= bl.bear {
                            let st = l.max(super::material::REF_LENGTH)
                                / super::skeleton::TIE_LENGTH.max(l);
                            sk.beam_with(a, b, bear.stiffer(st), BeamType::Support(10.0), 0);
                        }
                    }
                }
            }
        }
        groups.insert(bl.name.clone(), nodes);
    }
    // Suspension.
    let susp = sk.part("suspension");
    let mut wheel_nodes = Vec::new();
    for c in &def.corners {
        let mut sides: Vec<Vec<u32>> = Vec::new();
        for side in [1.0f32, -1.0] {
            let cn = corner(&mut sk, c, side, susp, grades.link);
            sides.push(cn.pivots.clone());
            // Pivots and the tower top into the body.
            let mut ties = cn.pivots.clone();
            ties.push(cn.shock_top);
            for &t in &ties {
                // The strut top is bolted to its tower (the spring's whole
                // load): stiffer than the arms' pivots.
                let m = if t == cn.shock_top {
                    grades.link.stiffer(SHOCK_TOP)
                } else {
                    grades.link
                };
                for c2 in sk.nearest(sk.pos(t), &chassis, 4) {
                    if sk.pos(c2).distance(sk.pos(t)) <= 0.35 {
                        sk.fixed_beam(t, c2, m);
                    }
                }
            }
            wheel_nodes.extend(cn.knuckle.iter().copied());
            wheel_nodes.extend(cn.pivots.iter().copied());
        }
        if c.cross > 0.0 {
            // The subframe's cross tubes: each arm pivot (not the strut top)
            // to its mirror.
            let tube = grades.link.stiffer(c.cross / grades.link.kl());
            for i in 0..sides[0].len() - 1 {
                sk.fixed_beam(sides[0][i], sides[1][i], tube);
            }
        }
    }
    groups.insert("suspension".into(), wheel_nodes);
    // Named masses.
    let mut named = 0.0;
    // Each node's stability floor at the built stiffnesses (what its beams
    // need, whatever it carries): a named mass fills the frame nodes' unmet
    // floors first, so it steadies the stiffness that is there instead of
    // weighing on nodes that already carry enough (mass the step budget
    // would otherwise pay twice for).
    let need: Vec<f32> = {
        let mut z = sk.clone();
        for n in &mut z.nodes {
            n.fixed = 0.0;
        }
        z.floors(1.0)
    };
    for m in &def.masses {
        let (lo, hi) = (Vec3::from_array(m.lo), Vec3::from_array(m.hi));
        let nodes: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| {
                let n = &sk.nodes[i as usize];
                n.part == 0 && n.p.cmpge(lo).all() && n.p.cmple(hi).all()
            })
            .collect();
        if nodes.is_empty() {
            return Err(format!("mass `{}`: no body node in its box", m.name));
        }
        // On the frame where it has nodes in the box (seats, fuel and trim bolt
        // to the structure, which carries and steadies them).
        // (One tier: on the members in the box, evenly: where it bolts, not
        // where the step's bound would want it.)
        let frame: Vec<u32> = nodes
            .iter()
            .copied()
            .filter(|&i| {
                if one {
                    structure_nodes.contains(&i)
                } else {
                    sk.nodes[i as usize].primary
                }
            })
            .collect();
        let nodes = if frame.is_empty() { nodes } else { frame };
        if one {
            sk.add_mass(&nodes, m.kg);
        } else {
            sk.add_mass_by_need(&nodes, m.kg, &need);
        }
        named += m.kg;
    }
    let blocks: f32 =
        def.blocks.iter().map(|b| b.kg).sum::<f32>() + def.plates.iter().map(|p| p.kg).sum::<f32>();
    let wheels: f32 = def
        .corners
        .iter()
        .map(|c| 2.0 * (c.wheel_mass + c.hub_mass + c.arm_mass))
        .sum();
    // Panels' sheet masses: on their own nodes (their stability floors
    // first), no share of the structure's.
    let mut sheets = 0.0;
    for part in 1..sk.parts.len() as u16 {
        if one {
            break;
        }
        let Some(kg) = sheet_mass(def, &sk, &shell.tris, part) else {
            continue;
        };
        let nodes: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| sk.nodes[i as usize].part == part)
            .collect();
        for &i in &nodes {
            sk.nodes[i as usize].sheet = true;
        }
        sk.add_mass_by_need(&nodes, kg, &need);
        sheets += kg;
    }
    let structure = def.kerb_mass - named - blocks - wheels - sheets;
    if structure <= 0.0 {
        return Err(format!(
            "kerb mass {} is under the named masses",
            def.kerb_mass
        ));
    }
    // Body damping: dashpots across about a metre of each part. The beams' own
    // dampers damp each beam's mode (hundreds of hertz) but barely the body's
    // (10–30 Hz: bending, torsion, panels), which would ring for seconds after
    // a crash; real bodies' joints, sealer and trim damp those. Each node gets
    // one, to the node of its part nearest `reach` m away along an axis (x, y,
    // z in turn).
    // (Each body dashpot and the node it is from.)
    let mut body_dash: Vec<(usize, u32)> = Vec::new();
    if def.body_damping > 0.0 {
        let n = sk.nodes.len() as u32;
        let susp_part = sk
            .parts
            .iter()
            .position(|p| p == "suspension")
            .map(|x| x as u16);
        let reach = def.body_damping_reach;
        // One tier: a node takes at most `BODY_DAMPING_FAN_IN` dashpots from
        // others (otherwise a light node that is many nodes' best partner
        // carried 16 of them, h·Σc/m 1.29 past `C_RATIO`'s 0.75).
        let mut fan_in = vec![0usize; n as usize];
        body_dash.clear();
        for i in 0..n {
            let ni = &sk.nodes[i as usize];
            if Some(ni.part) == susp_part {
                continue;
            }
            let axis = [Vec3::X, Vec3::Y, Vec3::Z][(i % 3) as usize];
            let pi = ni.p;
            let best = (0..n)
                .filter(|&j| j != i && sk.nodes[j as usize].part == ni.part)
                .filter(|&j| !one || fan_in[j as usize] < BODY_DAMPING_FAN_IN)
                .filter_map(|j| {
                    let d = sk.pos(j) - pi;
                    let l = d.length();
                    // (Small parts, a door: their farthest nodes.)
                    if l < 0.3 || l > 1.6 * reach {
                        return None;
                    }
                    // Along the axis, near `reach`.
                    let along = (d.dot(axis) / l).abs();
                    Some(((l - reach).abs() / reach + (1.0 - along), j))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            if let Some((_, j)) = best {
                // A catcher (a light node on four ties) gets a stronger one.
                let c = if ni.catcher {
                    CATCH_DAMPING * def.body_damping
                } else {
                    def.body_damping
                };
                fan_in[j as usize] += 1;
                if sk.dashpot(i, j, c, def.body_damping_strength) {
                    body_dash.push((sk.beams.len() - 1, i));
                }
            }
        }
    }
    // The structure: beams within the body part (shell and members), but glass.
    let glass_grade = grades.glass;
    let glassy: std::collections::HashSet<u32> = grade_of
        .iter()
        .filter(|(_, m)| **m == glass_grade)
        .map(|(i, _)| *i)
        .collect();
    for bm in &mut sk.beams {
        bm.structure = !bm.fixed
            && !bm.tie
            && !bm.damper
            && sk.nodes[bm.a as usize].part == 0
            && sk.nodes[bm.b as usize].part == 0
            && !glassy.contains(&bm.a)
            && !glassy.contains(&bm.b);
    }
    if std::env::var("SHOWCASE_FLOORS").is_ok() {
        let fl = sk.floors(1.0);
        let floor = |i: u32| fl[i as usize];
        let mut names: Vec<&String> = groups.keys().collect();
        names.sort();
        let mut in_group = vec![false; sk.nodes.len()];
        for n in &names {
            let g = &groups[*n];
            for &i in g {
                in_group[i as usize] = true;
            }
            eprintln!(
                "  {n:>16}: {:4} nodes floor {:7.1} kg",
                g.len(),
                g.iter().map(|&i| floor(i)).sum::<f32>()
            );
        }
        let rest: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| !in_group[i as usize])
            .collect();
        let frame: std::collections::HashSet<u32> = frame_nodes.iter().copied().collect();
        let (fr, sk_rest): (Vec<u32>, Vec<u32>) = rest.iter().partition(|i| frame.contains(i));
        eprintln!(
            "  {:>16}: {:4} nodes floor {:7.1} kg",
            "frame",
            fr.len(),
            fr.iter().map(|&i| floor(i)).sum::<f32>()
        );
        eprintln!(
            "  {:>16}: {:4} nodes floor {:7.1} kg",
            "skin+other",
            sk_rest.len(),
            sk_rest.iter().map(|&i| floor(i)).sum::<f32>()
        );
        {
            let (mut p, mut q) = (0.0, 0.0);
            for (i, n) in sk.nodes.iter().enumerate() {
                if n.primary {
                    p += fl[i];
                } else {
                    q += fl[i];
                }
            }
            eprintln!("  floors: primary {p:.0} kg, rest {q:.0} kg");
        }
        eprintln!(
            "  total floors {:.0} kg, budget {:.0} kg",
            fl.iter().sum::<f32>(),
            structure + sk.nodes.iter().map(|n| n.fixed).sum::<f32>()
        );
        // What the floors pay for: Σ (h²k + 2hc) at both ends by beam kind, with
        // the beam counts and median k·L.
        let shell_set: std::collections::HashSet<u32> = shell.nodes.iter().copied().collect();
        let mut of_group: HashMap<u32, &str> = HashMap::new();
        for (k, g) in &groups {
            for &i in g {
                of_group.insert(i, k.as_str());
            }
        }
        let h = super::skeleton::STEP;
        let mut cats: std::collections::BTreeMap<String, (f32, usize, Vec<f32>)> =
            Default::default();
        for bm in &sk.beams {
            let l = sk.pos(bm.a).distance(sk.pos(bm.b));
            let q = 2.0 * (h * h * bm.mat.k_at(l) + 2.0 * h * bm.mat.damp);
            let (ga, gb) = (of_group.get(&bm.a), of_group.get(&bm.b));
            let (sa, sb) = (shell_set.contains(&bm.a), shell_set.contains(&bm.b));
            let cat = match (ga, gb) {
                _ if bm.fixed => "fixed".to_string(),
                (None, None) if sa && sb => {
                    if sk.nodes[bm.a as usize].part == sk.nodes[bm.b as usize].part {
                        "shell".into()
                    } else {
                        "shell seam".into()
                    }
                }
                (Some(a), Some(b)) if a == b => "member".into(),
                (Some(_), Some(_)) => "member-member".into(),
                (Some(_), None) | (None, Some(_)) => "member-shell tie".into(),
                _ => "other".into(),
            };
            let cat = if bm.structure {
                cat
            } else {
                format!("{cat} (non-str)")
            };
            let tier = match (
                sk.nodes[bm.a as usize].primary,
                sk.nodes[bm.b as usize].primary,
            ) {
                (true, true) => "P",
                (false, false) => "S",
                _ => "PS",
            };
            let cat = format!("{tier} {cat}");
            let e = cats.entry(cat).or_default();
            e.0 += q;
            e.1 += 1;
            e.2.push(bm.mat.k_at(l) * l);
        }
        // Per group: Σq at its nodes' ends, and its internal beams.
        let mut per: std::collections::BTreeMap<&str, (f32, usize)> = Default::default();
        for bm in &sk.beams {
            let l = sk.pos(bm.a).distance(sk.pos(bm.b));
            let q = h * h * bm.mat.k_at(l) + 2.0 * h * bm.mat.damp;
            for i in [bm.a, bm.b] {
                let g = of_group
                    .get(&i)
                    .copied()
                    .unwrap_or(if shell_set.contains(&i) {
                        "shell"
                    } else {
                        "other"
                    });
                let e = per.entry(g).or_default();
                e.0 += q;
                e.1 += 1;
            }
        }
        let mut per: Vec<_> = per.into_iter().collect();
        per.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
        for (g, (q, n)) in per.iter().take(24) {
            eprintln!(
                "  {g:>16}: Σq {q:6.1} (≈{:.0} kg) over {n} beam ends",
                q / 2.4
            );
        }
        for (c, (q, n, kl)) in cats {
            let below = kl.iter().filter(|&&x| x < 1.25e5).count();
            eprintln!(
                "  {c:>28}: {n:5} beams, Σq {q:7.1} (≈{:.0} kg of floor), median k·L {:.2e}, {below} under 1.25e5",
                q / 2.4,
                median(kl)
            );
        }
    }
    let mut masses: Vec<(String, f32)> = Vec::new();
    let (stiffness_scale, skin_scale) = if one {
        // Real local masses, then the step's bound on the stiffness.
        masses = one_tier_masses(def, &mut sk, &shell, &trusses, &truss_kgm)?;
        // Each beam's elongation at break (its break force is its yield +
        // elong × k·L: it tears after stretching that share of its length),
        // kept through the bound's cut so it tears at the same strain.
        let elong: Vec<f32> = sk
            .beams
            .iter()
            .map(|b| {
                let kl = b.mat.k * super::material::REF_LENGTH;
                if b.mat.deform >= 1.0e8 || kl <= 0.0 {
                    -1.0
                } else {
                    ((b.mat.strength - b.mat.deform) / kl).max(0.0)
                }
            })
            .collect();
        one_tier_real_stiffness(
            &mut sk,
            &shell,
            &trusses,
            &truss_kgm,
            &lane_beams,
            &grade_of,
            &grades,
        );
        for c in &def.crush {
            one_tier_crush(&mut sk, c, &trusses, &truss_names, &groups)?;
        }
        if let Some(z) = &def.damping {
            one_tier_damping(&mut sk, z, &groups, &grade_of, &grades);
            // The body dashpots at the BX body's damping-only beams' rate per
            // kilogram of body ([`BX_BODY_DASHPOT_RATE`]), each on the mass
            // of the node it is from.
            for &(j, i) in &body_dash {
                sk.beams[j].mat.damp = BX_BODY_DASHPOT_RATE * sk.nodes[i as usize].mass;
            }
        }
        sk.exact_lengths = true;
        // The profile it ships with (spec §36). `BEAMNG_PROFILE=0` leaves the
        // line out (a measurement switch: the RoR profile's contact rule).
        sk.beamng_profile = std::env::var("BEAMNG_PROFILE").map_or(true, |v| v != "0");
        sk.cap_to_bound()?;
        for (b, &e) in sk.beams.iter_mut().zip(&elong) {
            if e >= 0.0 {
                // (A sheet edge given its real yield: the strength it had
                // was the yield's.)
                let kl = b.mat.k * super::material::REF_LENGTH;
                b.mat.strength = b.mat.deform.max(b.tension_yield) + e * kl;
            }
        }
        // (`BX_YIELD=1`, a measurement: each part's beams yield no later
        // than the Ibishu BX's median yield strain for that part type,
        // beamDeform / (beamSpring·L), at their built k: BeamNG's practice of
        // setting deform forces for the achievable k.)
        if std::env::var("BX_YIELD").map_or(false, |v| v != "0") {
            let scale: f32 = std::env::var("BX_YIELD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0);
            let nodes = sk.nodes.clone();
            let parts = sk.parts.clone();
            for (b, &e) in sk.beams.iter_mut().zip(&elong) {
                let (pa, pb) = (nodes[b.a as usize].part, nodes[b.b as usize].part);
                if pa != pb || b.ty != BeamType::Normal || b.group != 0 || b.mat.deform >= 1.0e8 {
                    continue;
                }
                // (`BX_YIELD_SHEETS=1`: only the sheets' edges, the members
                // keeping their sections' forces.)
                if std::env::var("BX_YIELD_SHEETS").is_ok() && b.tension_yield <= 0.0 {
                    continue;
                }
                let name = parts[pa as usize].as_str();
                let eps = match name {
                    "body" => 0.0100,
                    n if n.starts_with("door") => 0.0125,
                    "bonnet" => 0.0553,
                    "bootlid" => 0.0298,
                    n if n.starts_with("fender") => 0.0455,
                    n if n.starts_with("bumper") => 0.1553,
                    _ => continue,
                } * scale;
                let l = nodes[b.a as usize].p.distance(nodes[b.b as usize].p);
                if l < super::skeleton::MIN_YIELD_LENGTH {
                    continue;
                }
                let y = eps * b.mat.k_at(l) * l;
                if y < b.mat.deform.max(b.tension_yield) {
                    b.mat.deform = b.mat.deform.min(y);
                    b.tension_yield = b.tension_yield.min(y);
                    if e >= 0.0 {
                        let kl = b.mat.k * super::material::REF_LENGTH;
                        b.mat.strength = b.mat.deform.max(b.tension_yield) + e * kl;
                    }
                }
            }
        }
        (1.0, 1.0)
    } else if def.tiered {
        sk.fit_tiers(structure, 0.95)
    } else {
        let s = sk.fit_stiffness(structure, 0.95);
        (s, s)
    };
    if !one {
        sk.solve_masses(structure)?;
    }
    size_hardware(&mut sk, &hardware, params.strength)?;
    let panes = brittle_glass(
        def,
        &mut sk,
        &shell,
        &groups,
        &mut groups_made,
        params.strength,
    );
    if one {
        // (What the glass added.)
        sk.cap_to_bound()?;
    }
    preload(&mut sk, def);
    // Collision skin: the shell's skin on a coarser lattice, its nodes contacters.
    sk.collision = coarse_skin(&sk, &shell)?;
    for t in sk.collision.clone() {
        for i in t {
            sk.nodes[i as usize].contacter = true;
        }
    }
    shut_line_breaks(&mut sk, &shell, &hardware);
    sk.skin = shell.tris.clone();
    // Reference nodes: floor centre, a metre back, and left.
    let floor: Vec<u32> = shell
        .nodes
        .iter()
        .copied()
        .filter(|&i| shell.seg[&i] == Seg::Floor)
        .collect();
    let zc = 0.5 * len;
    let yb = def.profile.bottom.eval(zc);
    let c = sk.nearest(Vec3::new(0.0, yb, zc), &floor, 1)[0];
    let b = sk.nearest(Vec3::new(0.0, yb, zc - 1.0), &floor, 1)[0];
    let l = sk.nearest(Vec3::new(0.6, yb, zc), &floor, 1)[0];
    sk.cameras = [c, b, l];
    // Centre on the length.
    let shift = Vec3::new(0.0, 0.0, -0.5 * len);
    let mut out = sk.clone();
    for n in &mut out.nodes {
        n.p += shift;
    }
    let design: Vec<Vec3> = sk.nodes.iter().map(|n| n.p).collect();
    // The car spawns at rest (its glass's installation pose too).
    let glass: Vec<usize> = panes
        .iter()
        .flat_map(|u| u.beams.iter().map(|&j| j as usize))
        .collect();
    // (`REST_POSE=0`: the design pose instead, to measure what the rest
    // pose changes.)
    if std::env::var("REST_POSE").map_or(true, |v| v != "0") {
        rest_pose(&mut out, &glass)?;
        // The hardware's ropes are sized on the car as it stands (their
        // shares follow their angles, which the car's weight moves).
        size_hardware_once(&mut out, &hardware, params.strength)?;
        for (b, o) in sk.beams.iter_mut().zip(&out.beams) {
            b.mat = o.mat;
        }
    }
    for (n, o) in sk.nodes.iter_mut().zip(&out.nodes) {
        n.p = o.p - shift;
    }
    for (b, o) in sk.beams.iter_mut().zip(&out.beams) {
        b.rest_scale = o.rest_scale;
    }
    for (b, o) in sk.shocks.iter_mut().zip(&out.shocks) {
        b.precomp = o.precomp;
    }
    for (b, o) in sk.straps.iter_mut().zip(&out.straps) {
        b.precomp = o.precomp;
    }
    sk.rest_ground = out.rest_ground;
    sk.wheels = out.wheels.clone();
    sk.spawn_contacts = out.spawn_contacts.clone();
    sk.tyre_rest_axle = out
        .tyre_rest_axle
        .iter()
        .map(|[a, b]| [*a - shift, *b - shift])
        .collect();
    sk.tyre_pose = out
        .tyre_pose
        .iter()
        .map(|t| t.iter().map(|&p| p - shift).collect())
        .collect();
    let truck = out.to_truck();
    let bind_tris: HashMap<String, Vec<[u32; 3]>> = groups
        .iter()
        .map(|(k, nodes)| (k.clone(), cloud_tris(&sk, nodes)))
        .collect();
    Ok(Built {
        def: def.clone(),
        params: params.clone(),
        skeleton: sk,
        shell,
        truck,
        shift,
        design,
        bind_tris,
        groups,
        stiffness_scale,
        skin_scale,
        damage_units: {
            let mut u = damage_units(&hardware);
            u.extend(panes);
            u
        },
        hardware,
        masses,
    })
}

/// A pane that is its own part is named this followed by the panel it sits
/// in (`glass_door_f_l`).
pub const GLASS_PART: &str = "glass_";

/// One tier: the body dashpots' damping per kilogram of the node each is
/// from (1/s): the BX body's damping-only beams (`bx_body_hatch`, "main
/// frame damping": 12 beams of 500 N·s/m, k 0) over its 328.9 kg, Σc/Σm
/// = 6,000 / 328.9 = 18.2 /s. Measured, not tuned.
pub const BX_BODY_DASHPOT_RATE: f32 = 6000.0 / 328.9;

/// One tier: the most body dashpots a node takes from other nodes (a Choice
/// that keeps every node within `C_RATIO`).
pub const BODY_DAMPING_FAN_IN: usize = 2;

/// One tier: the body shell's sheet (m), a Choice typical of a body side
/// and floor (the body's sum is set by its measured mass, so only the shell's
/// share against the members' rests on it).
pub const SHELL_GAUGE: f32 = 0.0008;
/// Glass's density (kg/m³).
pub const GLASS_DENSITY: f32 = 2500.0;
/// One tier: a node on no sheet or member (a flange strip, a bracket) weighs
/// its beams' half-lengths of a 50 × 1.2 mm steel strip (kg/m), a Choice.
pub const STRIP_KG_PER_M: f32 = 0.05 * 0.0012 * 7850.0;

/// Steel's, glass's and a bumper fascia's (talc-filled polypropylene, a
/// Choice in its 1–2 GPa range) Young's moduli (Pa).
pub const E_STEEL: f32 = 210e9;
pub const E_FASCIA: f32 = 1.5e9;
/// A panel's (door, lid, fender skin) sheet (m): the 0.7–0.9 mm outer.
pub const PANEL_GAUGE: f32 = 0.00075;
/// A bumper fascia's wall (m).
pub const FASCIA_WALL: f32 = 0.003;
/// A sheet's yield stress (Pa): the outer and inner panels' bake-hardening
/// and drawing grades (BH210, DC04 class), a Choice; a fascia's polypropylene
/// compound about 25 MPa.
pub const SHEET_YIELD: f32 = 210e6;
pub const FASCIA_YIELD: f32 = 25e6;

/// Poisson's ratio of steel and of the fascia's compound in
/// [`sheet_ultimate_stress`] (0.3, a Choice for the compound).
pub const SHEET_POISSON: f32 = 0.3;
/// The plate buckling coefficient of a bay in [`sheet_ultimate_stress`]: 4,
/// a long plate simply supported on both unloaded edges (its neighbouring
/// bays and members hold them in line), a Choice between the clamped 6.97
/// and the free edge's 0.425.
pub const SHEET_BUCKLING_K: f32 = 4.0;

/// A thin sheet bay's ultimate stress in compression (Pa): it buckles
/// elastically at σ_cr = k·π²·E/(12(1 − ν²))·(t/b)² (Bryan 1891; Timoshenko
/// and Gere, Theory of Elastic Stability, 1961, §9.2), then carries load on
/// Winter's effective width b_e/b = √(σ_cr/σ_y)·(1 − 0.22·√(σ_cr/σ_y))
/// (G. Winter, "Strength of thin steel compression flanges", Trans. ASCE 112
/// (1947) 527–554; AISI S100 §B2.1) at its yield, so σ_u = σ_y·b_e/b,
/// never above σ_y. `b` is the bay's width: the lattice edge's length.
/// 0.8 mm steel on 0.2 m bays: σ_cr 12 MPa, σ_u 48 MPa (σ_y 210).
pub fn sheet_ultimate_stress(e: f32, sigma_y: f32, t: f32, b: f32) -> f32 {
    if sigma_y <= 0.0 {
        return 0.0;
    }
    let pi2 = std::f32::consts::PI * std::f32::consts::PI;
    let cr = SHEET_BUCKLING_K * pi2 * e / (12.0 * (1.0 - SHEET_POISSON * SHEET_POISSON))
        * (t / b.max(1e-3)).powi(2);
    let lam = (cr / sigma_y).sqrt();
    let width = (lam * (1.0 - 0.22 * lam)).min(1.0);
    sigma_y * width
}

/// One tier: each steel, glass and fascia beam's target stiffness is its real
/// one before the step's bound is applied ([`Skeleton::cap_to_bound`]):
/// - a member's beams (chords, rings, diagonals, bends): E·A/L with A its
///   section's steel ([`MemberSpec::mass_per_m`] / 7,850) shared by its
///   chords (4 a box, 2 a ladder; its webs are the same sheet, a Choice);
/// - a sheet's edges (the shell's beams within a part): an isotropic
///   membrane as a triangular lattice, (√3/4)·E·t per edge (exact for an
///   equilateral lattice, Poisson's ratio 1/3), its t the body's
///   [`SHELL_GAUGE`], a panel's [`PANEL_GAUGE`], a pane's glass or a
///   fascia's wall.
///
/// Every other beam (seals, ropes, bearing, mounts, the lumped shear panels'
/// racking from exact frame physics, gauges) keeps its built stiffness.
/// Damping keeps its ratio. A beam is never made softer here.
fn one_tier_real_stiffness(
    sk: &mut Skeleton,
    shell: &Shell,
    trusses: &[Truss],
    kgm: &[(f32, u16)],
    lanes: &[(usize, f32)],
    grade_of: &HashMap<u32, Material>,
    grades: &Grades,
) {
    let lane_of: HashMap<usize, f32> = lanes.iter().copied().collect();
    use super::material::REF_LENGTH;
    use super::members::STEEL_DENSITY;
    let n = sk.nodes.len();
    // Per node: the trusses it is in, with each one's chord area (m²).
    let mut of: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for (ti, (t, &(kg_m, _))) in trusses.iter().zip(kgm).enumerate() {
        let chords = {
            let st = t.stations[0];
            if st[0] == st[2] && st[1] == st[3] {
                2.0
            } else {
                4.0
            }
        };
        let area = kg_m / STEEL_DENSITY / chords;
        for &i in t.stations.iter().flatten() {
            if !of[i as usize].iter().any(|x| x.0 == ti) {
                of[i as usize].push((ti, area));
            }
        }
    }
    let on_shell: std::collections::HashSet<u32> = shell.nodes.iter().copied().collect();
    for (j, b) in sk.beams.iter_mut().enumerate() {
        if b.damper || b.fixed || b.ty != BeamType::Normal || b.group != 0 {
            continue;
        }
        let (a, c) = (b.a as usize, b.b as usize);
        let l = sk.nodes[a].p.distance(sk.nodes[c].p).max(1e-3);
        let mut real = lane_of.get(&j).map_or(0.0, |&area| E_STEEL * area / l);
        let lane = real > 0.0;
        for &(ti, area) in &of[a] {
            if of[c].iter().any(|x| x.0 == ti) {
                real = real.max(E_STEEL * area / l);
            }
        }
        let mut sheet_yield = 0.0f32;
        let mut tension_yield = 0.0f32;
        if on_shell.contains(&b.a)
            && on_shell.contains(&b.b)
            && sk.nodes[a].part == sk.nodes[c].part
        {
            let ga = grade_of[&b.a];
            let gb = grade_of[&b.b];
            let glass = ga == grades.glass || gb == grades.glass;
            let (e, t, sy) = if glass {
                (GLASS_E, TEMPERED_THICKNESS, 0.0)
            } else if ga == grades.plastic || gb == grades.plastic {
                (E_FASCIA, FASCIA_WALL, FASCIA_YIELD)
            } else if sk.nodes[a].part == 0 {
                (E_STEEL, SHELL_GAUGE, SHEET_YIELD)
            } else {
                (E_STEEL, PANEL_GAUGE, SHEET_YIELD)
            };
            let lattice = 3f32.sqrt() / 4.0;
            if !lane && lattice * e * t > real {
                real = lattice * e * t;
                // Its yield force: in compression the bay's ultimate stress
                // (a thin plate buckles, then carries Winter's effective
                // width, [`sheet_ultimate_stress`]), (√3/4)·σ_u·t·L; in
                // tension the sheet's own yield, (√3/4)·σ_y·t·L (a tension
                // field does not buckle; `rubble_tension_yield`).
                sheet_yield = lattice * sheet_ultimate_stress(e, sy, t, l) * t * l;
                tension_yield = lattice * sy * t * l;
            }
        }
        if sheet_yield > 0.0 && real > 0.0 {
            b.mat.deform = sheet_yield;
            b.mat.strength = sheet_yield;
            // (`TENSION_YIELD=0`: the buckled yield both ways, as before
            // `rubble_tension_yield`, to measure what it changes.)
            if std::env::var("TENSION_YIELD").map_or(true, |v| v != "0") {
                b.tension_yield = tension_yield;
            }
        }
        if real == 0.0 {
            continue;
        }
        let now = b.mat.k_at(l);
        if real > now {
            let f = real / now;
            b.mat.k *= f;
            b.mat.damp *= f.sqrt();
        }
        let _ = REF_LENGTH;
    }
}

/// One tier: a crush member's bays yield axially at its mean crush force
/// and its beams take JBeam's plastic limits ([`CrushSpec`]). The bays'
/// cuts share bend beams (over two stations), so the scaling is iterated
/// until every cut is within 1 % of the force.
fn one_tier_crush(
    sk: &mut Skeleton,
    c: &CrushSpec,
    trusses: &[Truss],
    names: &[String],
    groups: &HashMap<String, Vec<u32>>,
) -> Result<(), String> {
    let mut found = false;
    for (ti, name) in names.iter().enumerate() {
        if name != &c.member && name != &format!("{}_r", c.member) {
            continue;
        }
        found = true;
        let set: std::collections::HashSet<u32> =
            groups.get(name).into_iter().flatten().copied().collect();
        let beams: Vec<usize> = (0..sk.beams.len())
            .filter(|&j| {
                let b = &sk.beams[j];
                b.ty == BeamType::Normal
                    && b.group == 0
                    && !b.damper
                    && set.contains(&b.a)
                    && set.contains(&b.b)
            })
            .collect();
        for &j in &beams {
            sk.beams[j].plastic_limits = Some((1.0, c.deform_limit));
        }
        if c.mean_force <= 0.0 {
            continue;
        }
        let t = &trusses[ti];
        let centre = |st: &[u32; 4]| st.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / 4.0;
        // Per bay: the crossing beams and their cosines to the axis.
        let mut cuts: Vec<Vec<(usize, f32)>> = Vec::new();
        for w in t.stations.windows(2) {
            let (c0, c1) = (centre(&w[0]), centre(&w[1]));
            let n = (c1 - c0).normalize_or_zero();
            let m = 0.5 * (c0 + c1);
            let cut: Vec<(usize, f32)> = beams
                .iter()
                .copied()
                .filter_map(|j| {
                    let (pa, pb) = (sk.pos(sk.beams[j].a), sk.pos(sk.beams[j].b));
                    let (sa, sb) = ((pa - m).dot(n), (pb - m).dot(n));
                    (sa * sb < 0.0).then(|| (j, (pb - pa).normalize_or_zero().dot(n).abs()))
                })
                .collect();
            if !cut.is_empty() {
                cuts.push(cut);
            }
        }
        let cap = |sk: &Skeleton, cut: &[(usize, f32)]| -> f32 {
            cut.iter().map(|&(j, cs)| sk.beams[j].mat.deform * cs).sum()
        };
        for _ in 0..50 {
            let mut worst = 0.0f32;
            for cut in &cuts {
                let f = c.mean_force / cap(sk, cut).max(1e-3);
                worst = worst.max((f - 1.0).abs());
                for &(j, _) in cut {
                    sk.beams[j].mat.deform *= f.sqrt();
                }
            }
            if worst < 0.01 {
                break;
            }
        }
        let off = cuts
            .iter()
            .map(|cut| (cap(sk, cut) / c.mean_force - 1.0).abs())
            .fold(0.0f32, f32::max);
        if off > 0.05 {
            return Err(format!(
                "crush member {name}: a bay's cut is {:.0} % off its mean force",
                off * 100.0
            ));
        }
    }
    if !found {
        return Err(format!("crush member {}: no such member", c.member));
    }
    Ok(())
}

/// One tier: every steel, glass and fascia beam's damping set to its part's
/// ratio ([`DampingRatios`]) on its own two-mass mode at its target
/// stiffness: c = 2ζ·√(k·μ), μ = m₁m₂/(m₁+m₂) of its nodes' real masses.
/// [`Skeleton::cap_to_bound`] then keeps the ratio (c ∝ √k). Every beam of
/// a part but the fixed ones (blocks' and plates' mounts, the suspension)
/// and the dashpots: the part's seals, supports and hardware ropes too (the
/// BX's door file's supports and latch beams sit in its door ratio's range).
/// A beam across two parts takes the non-body part's.
fn one_tier_damping(
    sk: &mut Skeleton,
    z: &DampingRatios,
    groups: &HashMap<String, Vec<u32>>,
    grade_of: &HashMap<u32, Material>,
    grades: &Grades,
) {
    let mut crush: std::collections::HashSet<(u32, u32)> = Default::default();
    {
        let mut of: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut k = 0usize;
        for m in &z.crush_members {
            for key in [m.clone(), format!("{m}_r")] {
                if let Some(g) = groups.get(&key) {
                    for &i in g {
                        of.entry(i).or_default().push(k);
                    }
                    k += 1;
                }
            }
        }
        for b in &sk.beams {
            if let (Some(x), Some(y)) = (of.get(&b.a), of.get(&b.b)) {
                if x.iter().any(|t| y.contains(t)) {
                    crush.insert((b.a, b.b));
                }
            }
        }
    }
    let names = sk.parts.clone();
    for i in 0..sk.beams.len() {
        let b = &sk.beams[i];
        if b.damper || b.fixed || b.mat.k <= 0.0 {
            continue;
        }
        let (a, c) = (b.a as usize, b.b as usize);
        let l = sk.nodes[a].p.distance(sk.nodes[c].p).max(1e-3);
        let (ma, mc) = (sk.nodes[a].mass, sk.nodes[c].mass);
        if !(ma > 0.0 && mc > 0.0) {
            continue;
        }
        let glass =
            grade_of.get(&b.a) == Some(&grades.glass) || grade_of.get(&b.b) == Some(&grades.glass);
        let part = &names[sk.nodes[a].part.max(sk.nodes[c].part) as usize];
        let zeta = z.of(part, glass, crush.contains(&(b.a, b.b)));
        let mu = ma * mc / (ma + mc);
        let k = b.mat.k_at(l);
        sk.beams[i].mat.damp = 2.0 * zeta * (k * mu).sqrt();
    }
}

/// One tier: every node's real local mass. The named masses, blocks, plates
/// and the suspension's hubs and arms are where they were put (`fixed`). The
/// body-in-white's measured mass ([`VehicleDef::biw_mass`]) is spread over
/// the body's shell (its area × [`SHELL_GAUGE`]) and members (their length
/// × [`MemberSpec::mass_per_m`]); each part's measured mass
/// ([`VehicleDef::part_masses`], else its [`SheetSpec`]) over its own sheet
/// (its glass at its thickness) and members; the body's glass is its pane's
/// area × thickness × [`GLASS_DENSITY`]. What the kerb mass leaves (wiring,
/// HVAC, carpet, sealer, lamps, the rest of the equipment) is spread over the
/// body's shell by area. Sets each node's mass and [`SkNode::local`] to it:
/// nothing is moved to meet the step's bound. Returns the ledger.
fn one_tier_masses(
    def: &VehicleDef,
    sk: &mut Skeleton,
    shell: &Shell,
    trusses: &[Truss],
    kgm: &[(f32, u16)],
) -> Result<Vec<(String, f32)>, String> {
    use super::members::STEEL_DENSITY;
    let n = sk.nodes.len();
    let len = def.profile.length;
    let mut area = vec![0.0f32; n];
    for t in &shell.tris {
        let (a, b, c) = (sk.pos(t[0]), sk.pos(t[1]), sk.pos(t[2]));
        let ar = 0.5 * (b - a).cross(c - a).length();
        for &i in t {
            area[i as usize] += ar / 3.0;
        }
    }
    // A glass zone's pane thickness: the windscreen (the top zone ahead of
    // the middle) laminated, the rest tempered.
    let glass_t = |sk: &Skeleton, i: u32| -> Option<f32> {
        let p = sk.pos(i);
        let seg = shell.seg[&i];
        def.zones
            .iter()
            .find(|z| z.grade == "glass" && z.region.contains(p, seg, len))
            .map(|z| match z.region {
                Region::Top { z0, .. } if z0 > 0.5 * len => LAMINATED_GLASS,
                _ => TEMPERED_THICKNESS,
            })
    };
    let mut w = vec![0.0f32; n];
    let mut glass = vec![0.0f32; n];
    for &i in &shell.nodes {
        let iu = i as usize;
        match glass_t(sk, i) {
            Some(t) if sk.nodes[iu].part == 0 => glass[iu] = area[iu] * t * GLASS_DENSITY,
            Some(t) => w[iu] += area[iu] * t * GLASS_DENSITY,
            None => w[iu] += area[iu] * SHELL_GAUGE * STEEL_DENSITY,
        }
    }
    for (t, &(kg_m, _)) in trusses.iter().zip(kgm) {
        let sets: Vec<Vec<u32>> = t
            .stations
            .iter()
            .map(|st| {
                let mut v = st.to_vec();
                v.sort();
                v.dedup();
                v
            })
            .collect();
        let c = |v: &[u32]| v.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / v.len() as f32;
        for k in 0..sets.len().saturating_sub(1) {
            let kg = c(&sets[k]).distance(c(&sets[k + 1])) * kg_m;
            for v in [&sets[k], &sets[k + 1]] {
                for &i in v.iter() {
                    w[i as usize] += 0.5 * kg / v.len() as f32;
                }
            }
        }
    }
    let mut strip = vec![0.0f32; n];
    for b in &sk.beams {
        let l = sk.pos(b.a).distance(sk.pos(b.b));
        for i in [b.a, b.b] {
            strip[i as usize] += 0.5 * l * STRIP_KG_PER_M;
        }
    }
    for i in 0..n {
        if w[i] == 0.0 && glass[i] == 0.0 && sk.nodes[i].fixed == 0.0 {
            w[i] = strip[i];
        }
    }
    let mut ledger: Vec<(String, f32)> = Vec::new();
    let mut parts_kg = 0.0f32;
    // A pane that is its own part (`glass_<panel>`) weighs its glass; the
    // panel it sits in has its measured assembly mass less that.
    let panes: HashMap<String, f32> = sk
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.starts_with(GLASS_PART))
        .map(|(q, p)| {
            let kg = (0..n)
                .filter(|&i| sk.nodes[i].part as usize == q)
                .map(|i| w[i])
                .sum();
            (p[GLASS_PART.len()..].to_string(), kg)
        })
        .collect();
    let pane_kg = |name: &str| -> f32 { panes.get(name).copied().unwrap_or(0.0) };
    for part in 0..sk.parts.len() as u16 {
        let name = sk.parts[part as usize].clone();
        let nodes: Vec<usize> = (0..n).filter(|&i| sk.nodes[i].part == part).collect();
        let sum: f32 = nodes.iter().map(|&i| w[i]).sum();
        if sum == 0.0 {
            continue;
        }
        let target = if part == 0 {
            Some(def.biw_mass)
        } else if let Some((_, kg)) = def
            .part_masses
            .iter()
            .find(|(p, _)| name.starts_with(p.as_str()))
        {
            Some(*kg - pane_kg(&name))
        } else {
            sheet_mass(def, sk, &shell.tris, part)
        };
        let Some(target) = target.filter(|&t| t > 0.0) else {
            // (Blocks' and the suspension's own brackets: as computed.)
            parts_kg += sum;
            continue;
        };
        let f = target / sum;
        for &i in &nodes {
            w[i] *= f;
        }
        parts_kg += target;
        ledger.push((format!("{name} (sections and sheet ×{f:.2})"), target));
    }
    let glass_kg: f32 = glass.iter().sum();
    ledger.push(("body glass".into(), glass_kg));
    let fixed: f32 = sk.nodes.iter().map(|x| x.fixed).sum();
    let wheels: f32 = sk.wheels.iter().map(|x| x.mass).sum();
    ledger.push(("named masses, blocks, hubs, arms".into(), fixed));
    ledger.push(("tyres and rims".into(), wheels));
    let rest = def.kerb_mass - fixed - wheels - parts_kg - glass_kg;
    if rest < 0.0 {
        return Err(format!(
            "one tier: the kerb mass {} kg is {:.1} kg under its parts ({ledger:?})",
            def.kerb_mass, -rest
        ));
    }
    let body_area: f32 = shell
        .nodes
        .iter()
        .filter(|&&i| sk.nodes[i as usize].part == 0 && glass[i as usize] == 0.0)
        .map(|&i| area[i as usize])
        .sum();
    let mut equip = vec![0.0f32; n];
    for &i in &shell.nodes {
        let iu = i as usize;
        if sk.nodes[iu].part == 0 && glass[iu] == 0.0 {
            equip[iu] = rest * area[iu] / body_area;
        }
    }
    ledger.push(("equipment spread over the body's shell".into(), rest));
    for i in 0..n {
        let m = sk.nodes[i].fixed + w[i] + glass[i] + equip[i];
        if !(m > 0.0) {
            return Err(format!(
                "one tier: node {i} at {:?} has no mass",
                sk.nodes[i].p
            ));
        }
        sk.nodes[i].mass = m;
        sk.nodes[i].local = m;
    }
    Ok(ledger)
}

/// The collision skin across the shut lines (A1: no collision triangle
/// outlives the joint it spans). The skin stays the closed lattice, so
/// nothing passes between two parts while they are joined; each triangle
/// with nodes in two parts breaks (spec §18) when a beam along its edges
/// across the shut line breaks (the seam's seal, bolt or bearing beams) or
/// when the nearest hinge
/// or latch group of its parts lets go. Triangles inside a part stay plain:
/// a part that comes off keeps its own skin and collides as a part. Every
/// shell node of a part that can come off is a contacter, and the car's file
/// asks for contact from every node (spec §10.2).
fn shut_line_breaks(sk: &mut Skeleton, shell: &Shell, hardware: &[HardwareGroup]) {
    let pins: Vec<(u16, Vec3, u32)> = hardware
        .iter()
        .map(|h| {
            let b = &sk.beams[h.beams[0]];
            let pin = if sk.nodes[b.a as usize].part == h.part {
                b.a
            } else {
                b.b
            };
            (h.part, sk.pos(pin), h.group)
        })
        .collect();
    let mut breaks = Vec::with_capacity(sk.collision.len());
    for t in &sk.collision {
        let parts: Vec<u16> = t.iter().map(|&i| sk.nodes[i as usize].part).collect();
        if parts.iter().all(|&p| p == parts[0]) {
            breaks.push(None);
            continue;
        }
        let c = (sk.pos(t[0]) + sk.pos(t[1]) + sk.pos(t[2])) / 3.0;
        let g = pins
            .iter()
            .filter(|p| parts.contains(&p.0))
            .min_by(|a, b| a.1.distance(c).total_cmp(&b.1.distance(c)))
            .map_or(0, |p| p.2);
        // Its edges across the shut line (a torn skin beam inside a part
        // leaves the part's own surface there).
        let mask = (0..3)
            .filter(|&k| parts[k] != parts[(k + 1) % 3])
            .fold(0u8, |m, k| m | 1 << k);
        breaks.push(Some((g, mask)));
    }
    sk.collision_breaks = breaks;
    sk.full_contact = true;
    let detachable: std::collections::HashSet<u16> = hardware.iter().map(|h| h.part).collect();
    for &i in &shell.nodes {
        if detachable.contains(&sk.nodes[i as usize].part) {
            sk.nodes[i as usize].contacter = true;
        }
    }
}

/// A pane's aperture gauges: the frame (primary) nodes nearest the opening's
/// four corners (the extremes of its mounts' outer ends in the pane's two
/// longest directions), joined by its edges and both diagonals. Each is a
/// beam of negligible stiffness ([`GAUGE_K`], density 0, kept by the step
/// fit) that breaks at `strain` of its length, in `group`. Returns them.
/// (Compression shortens one diagonal as it stretches the other.)
pub const GAUGE_K: f32 = 1.0e4;

/// The frame members round each pane's opening (`frame` for
/// [`aperture_gauges`]): the screen pillars, header and cowl round the
/// windscreen; the C-pillars, rear header and parcel shelf round the
/// backlight; each door's sash and belt round its glass.
pub fn aperture_members(pane: &str) -> Vec<String> {
    let names = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    match pane {
        "glass_windscreen" => names(&["pillar_a", "pillar_a_r", "header_f", "cowl"]),
        "glass_backlight" => names(&["pillar_c", "pillar_c_r", "header_r", "parcel"]),
        _ => {
            let rest = pane.strip_prefix("glass_side_").unwrap_or("");
            let mut it = rest.split('_');
            match (it.next(), it.next()) {
                (Some(door), Some(s)) => {
                    let r = if s == "r" { "_r" } else { "" };
                    vec![
                        format!("door_sash_{door}{r}"),
                        format!("door_belt_{door}{r}"),
                    ]
                }
                _ => Vec::new(),
            }
        }
    }
}

fn aperture_gauges(
    sk: &mut Skeleton,
    pane: &[u32],
    mounts: &[usize],
    strain: f32,
    group: u32,
    frame: &[u32],
    thickness: f32,
) -> Vec<usize> {
    let set: std::collections::HashSet<u32> = pane.iter().copied().collect();
    // The opening's outline: the pane and the shell round it (its mounts'
    // outer ends).
    let mut rim: Vec<u32> = mounts
        .iter()
        .map(|&j| {
            let b = &sk.beams[j];
            if set.contains(&b.a) {
                b.b
            } else {
                b.a
            }
        })
        .collect();
    rim.extend_from_slice(pane);
    if rim.is_empty() {
        return Vec::new();
    }
    let c = rim.iter().map(|&i| sk.pos(i)).sum::<Vec3>() / rim.len() as f32;
    let (lo, hi) = rim.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), &i| {
        (a.min(sk.pos(i)), b.max(sk.pos(i)))
    });
    let ext = hi - lo;
    let mut axes = [(ext.x, Vec3::X), (ext.y, Vec3::Y), (ext.z, Vec3::Z)];
    axes.sort_by(|a, b| b.0.total_cmp(&a.0));
    // (Each axis over the opening's extent along it: the diagonals of the
    // opening, not of a square.)
    let (u, v) = (
        axes[0].1 / axes[0].0.max(1e-3),
        axes[1].1 / axes[1].0.max(1e-3),
    );
    if frame.is_empty() {
        return Vec::new();
    }
    // The opening's corners: the frame's own (its members' nodes about the
    // opening, within 30 cm of its outline), farthest along each diagonal of
    // the opening's plane.
    let near: Vec<u32> = frame
        .iter()
        .copied()
        .filter(|&i| rim.iter().any(|&r| sk.pos(r).distance(sk.pos(i)) < 0.3))
        .collect();
    let mut corners: Vec<u32> = Vec::new();
    for (su, sv) in [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)] {
        let f = |i: u32| (sk.pos(i) - c).dot(u * su + v * sv);
        if let Some(&q) = near.iter().max_by(|&&a, &&b| f(a).total_cmp(&f(b))) {
            if !corners.contains(&q) {
                corners.push(q);
            }
        }
    }
    let mut out = Vec::new();
    if std::env::var("GLASS_STRAIN").is_ok() {
        eprintln!(
            "  corners {corners:?} at {:?}",
            corners.iter().map(|&i| sk.pos(i)).collect::<Vec<_>>()
        );
    }
    for i in 0..corners.len() {
        for k in i + 1..corners.len() {
            let (a, b) = (corners[i], corners[k]);
            if sk.has_beam(a, b) {
                if std::env::var("GLASS_STRAIN").is_ok() {
                    eprintln!("  corner pair {a}-{b} already joined: no gauge");
                }
                continue;
            }
            // (A plain beam, so a break sets off its group: k·strain·L of
            // pull, elastic all the way; k a thousandth of a frame span's.)
            let l = sk.pos(a).distance(sk.pos(b));
            // The glass sits in a compliant mount (the bead, or the run
            // channel) on both edges, in series with the pane: across a span
            // L the opening racks ε = ε_g (1 + 2 E t t_b / (G w L)) for a
            // strain ε_g in the glass (exact for two shear layers in series
            // with an elastic strip). The pane breaks at ε_g = σ/E.
            let series =
                1.0 + 2.0 * GLASS_E * thickness * BEAD_THICKNESS / (URETHANE_G * BEAD_WIDTH * l);
            let k = GAUGE_K;
            let f = k * super::material::REF_LENGTH / l.max(super::material::REF_LENGTH)
                * strain
                * series
                * l;
            let m = Material {
                k,
                damp: 0.0,
                deform: f,
                strength: f,
                plastic: 0.0,
                density: 0.0,
            };
            if sk.beam_with(a, b, m, BeamType::Normal, group) {
                let j = sk.beams.len() - 1;
                sk.beams[j].fixed = true;
                out.push(j);
            }
        }
    }
    out
}

/// Seconds the car settles on its tyres before it is written ([`rest_pose`]).
pub const INSTALL_SETTLE: f32 = 2.0;

/// The car's rest pose (a Choice, `vehicle-authoring.md`): the car spawns
/// standing on its tyres, so nothing in it moves at spawn and its glass's
/// installation pose (a pane is bonded into a body already standing under
/// its own weight) is its spawn pose. The car stands on flat ground on the
/// CPU reference (its tyres touching, no drop), with nothing of its glass
/// breakable, and settles [`INSTALL_SETTLE`] s ([`settle`]). Its nodes then
/// take their settled positions, its tyres theirs (`rubble_tyre_pose`, built
/// round their axles' design positions); every beam and shock keeps the
/// solver's rest length (its `rest_scale` / precompression over the settled
/// length), so the body carries the same loads as when it settled and stands
/// in equilibrium; `gauges` (beams that only measure, of negligible
/// stiffness) take the settled lengths as their rest. Deterministic: the
/// same skeleton gives the same pose, and the `.truck` carries it
/// (`rubble_rest_pose`, `rubble_tyre_pose`, `rubble_spawn_contacts` and the
/// beams' fifth field), so every consumer loads the same car.
fn rest_pose(sk: &mut Skeleton, glass: &[usize]) -> Result<(), String> {
    let n = sk.nodes.len();
    let design = sk.clone();
    // The design pose as the solver reads it (the file's rounding), and the
    // node–triangle pairs it never collides (in range at spawn, spec §10.1).
    let pairs =
        |s: &crate::explicit_beams::ExplicitBeams| -> std::collections::HashSet<([u32; 3], u32)> {
            s.tri_excl
                .iter()
                .enumerate()
                .flat_map(|(t, l)| l.iter().map(move |&c| (t, c)))
                .map(|(t, c)| (s.triangles[t].nodes, c))
                .collect()
        };
    let (p0, excluded) = {
        let t = design.to_truck();
        let probe = crate::explicit_beams::ExplicitBeams::from_truck(
            &t,
            Vec3::ZERO,
            Vec3::ZERO,
            Default::default(),
        )?;
        (probe.positions()[..n].to_vec(), pairs(&probe))
    };
    let gauges: std::collections::HashSet<usize> = glass
        .iter()
        .copied()
        .filter(|&j| sk.beams[j].fixed && sk.beams[j].mat.density == 0.0)
        .collect();
    let settled = settle(sk, glass)?;
    let exact = sk.exact_lengths;
    let p1 = &settled.nodes;
    // Each beam's and shock's rest length is the solver's after the settle
    // (the design's, or where it yielded settling).
    let rest = |a: u32, c: u32| settled.rests[&(a.min(c), a.max(c))];
    for (j, b) in sk.beams.iter_mut().enumerate() {
        let (a, c) = (b.a as usize, b.b as usize);
        b.rest_scale = if gauges.contains(&j) {
            // A gauge's rest is its length here as the solver measures it:
            // the default (RoR-compatible) profile's fast inverse square
            // root is off by up to 0.18 % from the exact length the loader
            // gives a rest, more than the glass's break strain.
            // (On the exact path, `rubble_exact_lengths`, it is exact.)
            let q = p1[a].distance_squared(p1[c]);
            if exact {
                1.0
            } else {
                q * crate::explicit_beams::fast_inv_sqrt(q) / q.sqrt()
            }
        } else {
            rest(b.a, b.b) / p1[a].distance(p1[c])
        };
    }
    for x in sk.shocks.iter_mut().chain(sk.straps.iter_mut()) {
        let (a, c) = (x.a as usize, x.b as usize);
        x.precomp = rest(x.a, x.b) / p1[a].distance(p1[c]);
    }
    for (node, &p) in sk.nodes.iter_mut().zip(p1) {
        node.p = p;
    }
    // (A posed tyre is built from its line's first axle node: the one its
    // settle built it from.)
    for (w, a) in sk.wheels.iter_mut().zip(&settled.axles) {
        [w.a1, w.a2] = *a;
    }
    sk.tyre_rest_axle = sk
        .wheels
        .iter()
        .map(|w| [p0[w.a1 as usize], p0[w.a2 as usize]])
        .collect();
    sk.tyre_pose = settled.tyres;
    sk.rest_ground = Some(settled.ground);
    // The pairs that came into range settling rest in contact: they keep
    // colliding.
    let rest = crate::explicit_beams::ExplicitBeams::from_truck(
        &sk.to_truck(),
        Vec3::ZERO,
        Vec3::ZERO,
        Default::default(),
    )?;
    let mut contacts: Vec<([u32; 3], u32)> = pairs(&rest).difference(&excluded).copied().collect();
    contacts.sort_unstable();
    sk.spawn_contacts = contacts;
    if std::env::var("GLASS_STRAIN").is_ok() {
        let moved = p0
            .iter()
            .zip(p1)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        eprintln!(
            "rest pose: nodes moved up to {:.1} mm from the design; ground at y {:.4}; {} pairs rest in contact",
            moved * 1e3,
            settled.ground,
            sk.spawn_contacts.len()
        );
    }
    Ok(())
}

struct Settled {
    /// The skeleton's nodes, in the file's frame.
    nodes: Vec<Vec3>,
    /// Each tyre's nodes.
    tyres: Vec<Vec<Vec3>>,
    /// Each tyre's axle nodes, the one it was built from first.
    axles: Vec<[u32; 2]>,
    /// The ground's height.
    ground: f32,
    /// Each beam's rest length at the end, by its sorted node pair (the
    /// first beam on the pair): the design's, or where it yielded settling.
    rests: std::collections::HashMap<(u32, u32), f32>,
}

/// The car standing on flat ground on the CPU reference (its design pose,
/// its tyres touching), with nothing of its glass breakable, after
/// [`INSTALL_SETTLE`] s: in the file's frame, to the µm the file writes.
fn settle(sk: &Skeleton, glass: &[usize]) -> Result<Settled, String> {
    use crate::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
    let n = sk.nodes.len();
    let mut probe_sk = sk.clone();
    probe_sk.rest_ground = None;
    probe_sk.tyre_pose.clear();
    probe_sk.tyre_rest_axle.clear();
    probe_sk.spawn_contacts.clear();
    for &j in glass {
        probe_sk.beams[j].mat.deform = 1.0e9;
        probe_sk.beams[j].mat.strength = 1.0e9;
        probe_sk.beams[j].group = 0;
    }
    let truck = probe_sk.to_truck();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&truck, Vec3::ZERO, Vec3::ZERO, settings.clone())?;
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    // Its tyres touching the ground (no drop: a landing would yield its
    // weakest panels), it settles [`INSTALL_SETTLE`] s.
    let lift = Vec3::new(0.0, -low, 0.0);
    let mut s = ExplicitBeams::from_truck(&truck, lift, Vec3::ZERO, settings)?;
    let h = crate::explicit_beams::STEP;
    for k in 0..(INSTALL_SETTLE / h) as usize {
        s.step();
        if std::env::var("GLASS_STRAIN").is_ok() && k % 1000 == 999 {
            let (iv, vm) = (0..n)
                .map(|i| (i, s.nodes[i].vel.length()))
                .fold((0, 0.0f32), |a, x| if x.1 > a.1 { x } else { a });
            let ke: f32 = s
                .nodes
                .iter()
                .map(|q| 0.5 * q.mass * q.vel.length_squared())
                .sum();
            let yielded = s
                .beams
                .iter()
                .filter(|b| {
                    b.rest != b.rest0
                        && b.kind != crate::explicit_beams::ExplicitKind::LimitedSpring
                })
                .count();
            eprintln!(
                "settle {:.1} s: KE {ke:.3} J, fastest {vm:.4} m/s (node {iv} {}), {yielded} beams yielded",
                (k + 1) as f32 * h,
                sk.parts[sk.nodes[iv].part as usize]
            );
        }
    }
    if s.beams.iter().any(|b| b.broken) {
        let which: Vec<String> = s
            .beams
            .iter()
            .filter(|b| b.broken)
            .take(6)
            .map(|b| {
                let name = |i: u32| match sk.nodes.get(i as usize) {
                    Some(nd) => format!("{i} {} {:.2?}", sk.parts[nd.part as usize], nd.p),
                    None => format!("{i} wheel"),
                };
                format!(
                    "{}-{} (rest {:.2} m, k {:.0}, strength {:.0} N, {:?})",
                    name(b.a),
                    name(b.b),
                    b.rest0,
                    b.k,
                    b.strength,
                    b.kind
                )
            })
            .collect();
        return Err(format!(
            "the car broke a beam settling on its tyres: {which:?}"
        ));
    }
    let um = |x: f32| ((x as f64 * 1.0e6).round() / 1.0e6) as f32;
    let file = |p: Vec3| {
        let q = p - lift;
        Vec3::new(um(q.x), um(q.y), um(q.z))
    };
    let p = s.positions();
    Ok(Settled {
        nodes: p[..n].iter().map(|&q| file(q)).collect(),
        tyres: s
            .tyres
            .iter()
            .map(|t| {
                (t.first..t.first + t.count)
                    .map(|i| file(p[i as usize]))
                    .collect()
            })
            .collect(),
        axles: s.tyres.iter().map(|t| t.axle).collect(),
        rests: s
            .beams
            .iter()
            .rev()
            .map(|b| ((b.a.min(b.b), b.a.max(b.b)), b.rest))
            .collect(),
        ground: um(-lift.y),
    })
}

/// Tempered glass's break strain: characteristic bending strength ≈ 120 MPa
/// over E ≈ 70 GPa (`beamng-realism-plan.md` P6 [GLASS-TUE, GLASS-MIF]):
/// 0.17 %, with no plastic range.
pub const TEMPERED_STRAIN: f32 = TEMPERED_STRENGTH / GLASS_E;
/// Tempered glass's characteristic strength (Pa) [GLASS-TUE].
pub const TEMPERED_STRENGTH: f32 = 120.0e6;
/// Annealed glass's (a laminated pane's plies) (Pa) [GLASS-TUE].
pub const ANNEALED_STRENGTH: f32 = 45.0e6;
/// Glass's Young's modulus (Pa) [GLASS-MIF].
pub const GLASS_E: f32 = 70.0e9;
/// Tempered side, quarter and backlight glazing (m): NHTSA DOT 88988 table 2.
pub const TEMPERED_THICKNESS: f32 = 0.0035;
/// A laminated windscreen's glass (m): two 2.1 mm plies on PVB, ibid.
pub const LAMINATED_GLASS: f32 = 0.0042;
/// Annealed glass (a laminated windscreen's plies): ≈ 45 MPa over 70 GPa,
/// 0.064 %: its plies crack there and its interlayer holds them.
pub const ANNEALED_STRAIN: f32 = ANNEALED_STRENGTH / GLASS_E;

/// Glass from its material constants. Each pane's beams (those on its nodes
/// to the shell around it) break at the glass's strain: tempered panes
/// (side windows, backlight) are brittle (they break, never yield, at
/// 0.17 %) and fail whole (one detacher group a pane); the laminated
/// windscreen cracks (yields) at its plies' 0.064 % and its interlayer keeps
/// it in place (its beams keep their tear strength). The beams are as stiff
/// as the step allows, far softer than real glass, so the break is set as a
/// strain: the pane breaks when its frame racks it 0.17 %, whatever force
/// that takes. Returns the panes as damage units.
fn brittle_glass(
    def: &VehicleDef,
    sk: &mut Skeleton,
    shell: &Shell,
    groups: &HashMap<String, Vec<u32>>,
    groups_made: &mut u32,
    scale: f32,
) -> Vec<DamageUnit> {
    let len = def.profile.length;
    let mut panes: Vec<(String, bool, Vec<u32>)> = Vec::new();
    let mut side = 0usize;
    for z in def.zones.iter().filter(|z| z.grade == "glass") {
        let nodes: Vec<u32> = shell
            .nodes
            .iter()
            .copied()
            .filter(|&i| z.region.contains(sk.pos(i), shell.seg[&i], len))
            .collect();
        if nodes.is_empty() {
            continue;
        }
        match &z.region {
            Region::Side(_) => {
                let which = ["f", "r", "q"].get(side).copied().unwrap_or("q");
                side += 1;
                for (s, sign) in [("l", 1.0f32), ("r", -1.0)] {
                    let v: Vec<u32> = nodes
                        .iter()
                        .copied()
                        .filter(|&i| sk.pos(i).x * sign > 0.0)
                        .collect();
                    panes.push((format!("glass_side_{which}_{s}"), false, v));
                }
            }
            _ => {
                let front = nodes.iter().map(|&i| sk.pos(i).z).sum::<f32>() / nodes.len() as f32
                    > 0.5 * len;
                if front {
                    panes.push(("glass_windscreen".into(), true, nodes));
                } else {
                    panes.push(("glass_backlight".into(), false, nodes));
                }
            }
        }
    }
    let on_shell: std::collections::HashSet<u32> = shell.nodes.iter().copied().collect();
    let mut out = Vec::new();
    for (name, laminated, nodes) in panes {
        let set: std::collections::HashSet<u32> = nodes.iter().copied().collect();
        let bonded = !name.starts_with("glass_side");
        let g = if laminated {
            0
        } else {
            *groups_made += 1;
            *groups_made
        };
        let mut pane = Vec::new();
        let mut mounts = Vec::new();
        for (j, b) in sk.beams.iter().enumerate() {
            if b.tie || b.damper || b.ty != BeamType::Normal || b.group != 0 {
                continue;
            }
            if !on_shell.contains(&b.a) || !on_shell.contains(&b.b) {
                continue;
            }
            match (set.contains(&b.a), set.contains(&b.b)) {
                (true, true) => pane.push(j),
                (true, false) | (false, true) => mounts.push(j),
                _ => {}
            }
        }
        // The pane: brittle at its strain (tempered), or cracking there and
        // held by its interlayer (laminated). Off ([`BRITTLE_GLASS`]): the
        // panes are skin-tier beams, far softer than glass, and rack with
        // the settling body: at 0.17 % they break at rest (158 beams in
        // `a`). The units name each pane's beams either way.
        if !brittle_glass_on() {
            let part = sk.nodes[nodes[0] as usize].part;
            out.push(DamageUnit {
                name,
                kind: DamageKind::Glass,
                part,
                beams: pane.iter().chain(&mounts).map(|&j| j as u32).collect(),
            });
            continue;
        }
        // The pane's own beams are skin-tier, ~10⁵ × softer than glass:
        // they carry its weight and break with its group, never on their
        // own strain (that would be a ~0.3 N break force).
        // As a strip of the real pane (a Choice: the strip's width is the
        // beam's length) a beam holds σ·t·L in its plane: tempered 120 MPa
        // over 3.5 mm (NHTSA DOT 88988, table 2: side, quarter and backlight
        // glazing 3.5 mm), about 0.1 MN at 0.25 m, far above anything the
        // step-softened beam carries. So it breaks only with its pane's group
        // (tempered) or, laminated (two 2.1 mm plies on PVB), yields where
        // its plies crack (45 MPa over 4.2 mm) and holds on its interlayer.
        for &j in &pane {
            let l = sk.pos(sk.beams[j].a).distance(sk.pos(sk.beams[j].b));
            let b = &mut sk.beams[j];
            if laminated {
                b.mat.deform = scale * ANNEALED_STRENGTH * LAMINATED_GLASS * l;
                b.mat.strength = f32::MAX;
            } else {
                b.mat.deform = scale * TEMPERED_STRENGTH * TEMPERED_THICKNESS * l;
                b.mat.strength = b.mat.deform;
                b.group = g;
            }
        }
        // Its mounts: each carries the bead (or the channel) along its share
        // of the pane's edge, L (its spacing along the edge, about its own
        // length). A bead holds τ·w·L before it lets go (no yield). Its
        // stiffness G·w·L/t is far above what the step allows the pane's
        // light nodes, so a mount keeps the stiffness the step fit gave it
        // (min of the two).
        let edge = mounts
            .iter()
            .map(|&j| sk.pos(sk.beams[j].a).distance(sk.pos(sk.beams[j].b)))
            .sum::<f32>()
            / mounts.len().max(1) as f32;
        for &j in &mounts {
            let b = &mut sk.beams[j];
            let l = sk.nodes[b.a as usize].p.distance(sk.nodes[b.b as usize].p);
            if bonded {
                let real = URETHANE_G * BEAD_WIDTH / BEAD_THICKNESS * edge;
                if b.mat.k_at(l) > real {
                    b.mat = b.mat.stiffer(real / b.mat.k_at(l));
                }
                b.mat.deform = scale * URETHANE_TAU * BEAD_WIDTH * edge;
                b.mat.strength = b.mat.deform;
                b.group = g;
            } else {
                // Side glass sits in its door's run channel: it bears there
                // (compression only), held by its regulator along the bottom
                // (those beams keep their tension), and breaks with the pane.
                let low = nodes
                    .iter()
                    .map(|&i| sk.nodes[i as usize].p.y)
                    .fold(f32::MAX, f32::min);
                let inner = if set.contains(&b.a) { b.a } else { b.b };
                if sk.nodes[inner as usize].p.y > low + 0.05 {
                    b.ty = BeamType::Support(SEAL_BREAK);
                }
                let _ = l;
                b.group = g;
            }
        }
        // The aperture's racking (a Choice, `vehicle-authoring.md`): the
        // frame nodes at the opening's four corners carry near-real
        // stiffness. Gauges across them (its edges and diagonals) are
        // supports of negligible stiffness that break past σ/E of
        // extension against the installed pose: the strain the frame
        // imposes on the glass reaching its strength (tempered 120 MPa,
        // laminated 45 MPa: it cracks, in no group, and holds).
        // The aperture's frame: the members round the opening.
        let frame: Vec<u32> = aperture_members(&name)
            .iter()
            .filter_map(|m| groups.get(m.as_str()))
            .flatten()
            .copied()
            .filter(|&i| sk.nodes[i as usize].primary && !sk.nodes[i as usize].catcher)
            .collect();
        if std::env::var("GLASS_STRAIN").is_ok() {
            eprintln!(
                "glass {name}: {} pane nodes, {} mounts, aperture frame {} nodes ({:?})",
                nodes.len(),
                mounts.len(),
                frame.len(),
                aperture_members(&name)
                    .iter()
                    .map(|m| (m.clone(), groups.get(m.as_str()).map_or(0, |g| g.len())))
                    .collect::<Vec<_>>()
            );
        }
        let gauges = aperture_gauges(
            sk,
            &nodes,
            &mounts,
            if laminated {
                ANNEALED_STRAIN
            } else {
                TEMPERED_STRAIN
            },
            if laminated { 0 } else { g },
            &frame,
            if laminated {
                LAMINATED_GLASS
            } else {
                TEMPERED_THICKNESS
            },
        );
        let part = sk.nodes[nodes[0] as usize].part;
        let beams: Vec<u32> = pane
            .iter()
            .chain(&mounts)
            .chain(&gauges)
            .map(|&j| j as u32)
            .collect();
        out.push(DamageUnit {
            name,
            kind: DamageKind::Glass,
            part,
            beams,
        });
    }
    out
}

/// Brittle panes and sized mounts (A9): off until the panes stop racking
/// with the body at rest (`vehicle-authoring.md`, glass).
pub const BRITTLE_GLASS: bool = false;

/// [`BRITTLE_GLASS`], or `BRITTLE_GLASS=1` in the environment (to measure it
/// while it is off).
pub fn brittle_glass_on() -> bool {
    BRITTLE_GLASS || std::env::var("BRITTLE_GLASS").is_ok_and(|v| v == "1")
}

/// The windscreen and backlight's urethane bead: TEROSON PU 9096 PL (Henkel
/// technical data sheet, October 2016): shear modulus about 1.2 MPa (DIN
/// 54451, 5 mm layer), shear strength 5–6 MPa fully cured (5.5 taken).
/// The bead is 5 mm thick (the datasheet's layer; 5–10 mm in cars) and 10
/// mm wide (a stated geometry).
pub const URETHANE_G: f32 = 1.2e6;
pub const URETHANE_TAU: f32 = 5.5e6;
pub const BEAD_THICKNESS: f32 = 0.005;
pub const BEAD_WIDTH: f32 = 0.010;

/// The damage units: each hinge and latch group (its secondary latch is its
/// latch's unit too: the latch is broken when either position let go).
fn damage_units(hw: &[HardwareGroup]) -> Vec<DamageUnit> {
    let mut out: Vec<DamageUnit> = Vec::new();
    for h in hw {
        let name = h.name.replace(".latch_secondary", ".latch");
        let kind = match h.kind {
            Hardware::HingeUpper | Hardware::HingeLower => DamageKind::Hinge,
            Hardware::Latch | Hardware::LatchSecondary => DamageKind::Latch,
        };
        let beams = h.beams.iter().map(|&j| j as u32);
        if let Some(u) = out.iter_mut().find(|u| u.name == name) {
            u.beams.extend(beams);
        } else {
            out.push(DamageUnit {
                name,
                kind,
                part: h.part,
                beams: beams.collect(),
            });
        }
    }
    out
}

/// A rig's beams' tensions per newton of load along `d` on the part: its
/// hardware nodes moved as one along `d` (FMVSS 206's fixture guides the
/// load), the body held. A rope at cosine c = −u·d to the pull (u from the
/// part's node to the body's) stretches c·δ, so it carries k·c·δ and
/// resists k·c²·δ; slack ones (c ≤ 0) carry nothing.
pub fn rig_tensions(sk: &Skeleton, beams: &[usize], part: u16, d: Vec3) -> Vec<f32> {
    let kc: Vec<(f32, f32)> = beams
        .iter()
        .map(|&j| {
            let b = &sk.beams[j];
            let (p, q) = if sk.nodes[b.a as usize].part == part {
                (b.a, b.b)
            } else {
                (b.b, b.a)
            };
            let v = sk.pos(q) - sk.pos(p);
            (b.mat.k_at(v.length()), (-v.normalize().dot(d)).max(0.0))
        })
        .collect();
    let stiff: f32 = kc.iter().map(|&(k, c)| k * c * c).sum();
    kc.iter()
        .map(|&(k, c)| if stiff > 0.0 { k * c / stiff } else { 0.0 })
        .collect()
}

/// Cosine of a hardware beam to a pull `d` on its part (−u·d, u from the
/// part's node to the body's).
pub fn rig_cos(sk: &Skeleton, j: usize, part: u16, d: Vec3) -> f32 {
    let b = &sk.beams[j];
    let (p, q) = if sk.nodes[b.a as usize].part == part {
        (b.a, b.b)
    } else {
        (b.b, b.a)
    };
    -(sk.pos(q) - sk.pos(p)).normalize().dot(d)
}

/// The load (N) at which a rig's hardware separates when pulled along `d`
/// (its first beam reaching its strength takes its group): the static
/// prediction the FMVSS 206 rig test measures.
pub fn rig_strength(sk: &Skeleton, hw: &[HardwareGroup], groups: &[usize], d: Vec3) -> f32 {
    let beams: Vec<usize> = groups.iter().flat_map(|&g| hw[g].beams.clone()).collect();
    let t = rig_tensions(sk, &beams, hw[groups[0]].part, d);
    beams
        .iter()
        .zip(&t)
        .filter(|(_, &x)| x > 0.0)
        .map(|(&j, &x)| sk.beams[j].mat.strength / x)
        .fold(f32::INFINITY, f32::min)
}

/// The rigs a part's hardware is tested in (FMVSS 206 S4.1.1, S4.1.2, as
/// TP-206 runs them): the hinge system (both hinges), the latch fully latched
/// (primary and secondary together) and the latch in its secondary position
/// (alone), each with its measured loads. Returns (rig, groups in it, loads).
pub fn hardware_rigs(hw: &[HardwareGroup]) -> Vec<(String, Vec<usize>, Loads)> {
    let mut parts: Vec<u16> = hw.iter().map(|h| h.part).collect();
    parts.sort();
    parts.dedup();
    let mut out = Vec::new();
    for p in parts {
        let of = |k: &[Hardware]| -> Vec<usize> {
            (0..hw.len())
                .filter(|&i| hw[i].part == p && k.contains(&hw[i].kind))
                .collect()
        };
        let name = |i: &[usize]| hw[i[0]].name.split('.').next().unwrap().to_string();
        let hinges = of(&[Hardware::HingeUpper, Hardware::HingeLower]);
        if !hinges.is_empty() {
            out.push((
                format!("{}.hinges", name(&hinges)),
                hinges.clone(),
                HINGE_SYSTEM,
            ));
            // Each hinge on its own holds the pair's floor: the reports'
            // fixtures loaded the pair, and no pair let go up to it, so
            // neither hinge did (a door's intrusion beam pulls its own hinge
            // alone: at half the pair's floor, the side pole tore the lower
            // hinge off every time).
            let share = 1.0;
            for &h in &hinges {
                out.push((
                    hw[h].name.clone(),
                    vec![h],
                    Loads {
                        longitudinal: share * HINGE_SYSTEM.longitudinal,
                        transverse: share * HINGE_SYSTEM.transverse,
                    },
                ));
            }
        }
        let latch = of(&[Hardware::Latch, Hardware::LatchSecondary]);
        if !latch.is_empty() {
            out.push((format!("{}.latch", name(&latch)), latch, PRIMARY_LATCH));
        }
        let sec = of(&[Hardware::LatchSecondary]);
        if !sec.is_empty() {
            out.push((
                format!("{}.latch_secondary", name(&sec)),
                sec,
                SECONDARY_LATCH,
            ));
        }
    }
    out
}

/// The rig directions: longitudinal along the car away from the pillar the
/// hardware holds to (toward it the part bears on its supports), transverse
/// outward (the door pulled open). (direction, longitudinal)
pub fn rig_directions(_sk: &Skeleton, h: &HardwareGroup) -> [(Vec3, bool); 2] {
    [(h.along, true), (h.outward, false)]
}

/// Each hardware beam's strength from its rigs: a beam carrying `t` per
/// newton of the rig's load breaks at `load × t`, the largest over the rig's
/// directions; so in each direction the group separates at the measured load
/// (its most loaded beams together), never through one beam's uneven share.
/// Elastic to the break: hardware does not yield. The secondary latch is
/// sized alone and must not be the first to go fully latched.
fn size_hardware(sk: &mut Skeleton, hw: &[HardwareGroup], scale: f32) -> Result<(), String> {
    // The secondary engages softer until it no longer goes first (real ones
    // carry nothing until the primary lets go).
    let mut last = String::new();
    for _ in 0..10 {
        match size_hardware_once(sk, hw, scale) {
            Ok(()) => return Ok(()),
            Err(e) if e.contains("secondary latch would go first") => {
                last = e;
                for h in hw.iter().filter(|h| h.kind == Hardware::LatchSecondary) {
                    for &j in &h.beams {
                        sk.beams[j].mat = sk.beams[j].mat.stiffer(0.5);
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

fn size_hardware_once(sk: &mut Skeleton, hw: &[HardwareGroup], scale: f32) -> Result<(), String> {
    let mut strength: HashMap<usize, f32> = HashMap::new();
    let rigs = hardware_rigs(hw);
    // Secondary alone first, then the assemblies (the primary's beams take
    // their size from the fully latched rig, the secondary keeps its own).
    let mut order: Vec<&(String, Vec<usize>, Loads)> = rigs.iter().collect();
    order.sort_by_key(|r| !r.0.ends_with("latch_secondary"));
    for (name, groups, loads) in order {
        let beams: Vec<usize> = groups.iter().flat_map(|&g| hw[g].beams.clone()).collect();
        let part = hw[groups[0]].part;
        // The hinge pair's rig is the test's; each hinge is sized alone.
        if name.ends_with(".hinges") {
            continue;
        }
        // A fan of pulls between the two tested directions, the load on the
        // ellipse through the two measured ones (so no direction between
        // them finds a weaker beam).
        // Pushed down (the door's weight, across the striker's bolt or the
        // hinge pin as Load Test One loads them along the car) the
        // longitudinal load.
        let [(dl, _), (dt, _)] = rig_directions(sk, &hw[groups[0]]);
        let mut fan: Vec<(Vec3, f32)> = Vec::new();
        let mut arcs = vec![(dl, loads.longitudinal, dt, loads.transverse)];
        if hw[groups[0]].weight {
            arcs.push((-Vec3::Y, loads.longitudinal, dt, loads.transverse));
            arcs.push((-Vec3::Y, loads.longitudinal, dl, loads.longitudinal));
        }
        for (a, fa, b, fb) in arcs {
            for k in 0..=RIG_FAN {
                let th = std::f32::consts::FRAC_PI_2 * k as f32 / RIG_FAN as f32;
                let (c, sn) = (th.cos(), th.sin());
                fan.push((
                    (a * c + b * sn).normalize(),
                    1.0 / ((c / fa).powi(2) + (sn / fb).powi(2)).sqrt(),
                ));
            }
        }
        // (The secondary position is tested along and across only.)
        if name.ends_with("latch_secondary") {
            fan.truncate(RIG_FAN + 1);
        }
        for (d, f) in fan {
            // A pull no beam lies within 60° of is not this hardware's to
            // hold (the part's other hardware and its bearing take it).
            if !beams.iter().any(|&j| rig_cos(sk, j, part, d) >= 0.5) {
                continue;
            }
            let load = scale * f;
            let t = rig_tensions(sk, &beams, part, d);
            for (&j, &tj) in beams.iter().zip(&t) {
                let sec = hw
                    .iter()
                    .any(|h| h.kind == Hardware::LatchSecondary && h.beams.contains(&j));
                if sec && !name.ends_with("latch_secondary") {
                    // Fully latched: the secondary only shares.
                    if let Some(&s) = strength.get(&j) {
                        if load * tj >= s {
                            return Err(format!(
                                "{name}: the secondary latch would go first ({:.0} N of its {s:.0})",
                                load * tj
                            ));
                        }
                    }
                    continue;
                }
                let e = strength.entry(j).or_insert(0.0);
                *e = e.max(load * tj);
            }
        }
    }
    // A rope no rig pull loads (it lies across every tested direction: in
    // the standing car's pose a rope square to the pulls may tilt just past
    // square) holds nothing the rigs measure; it takes its group's weakest
    // rope's strength, so no rig's result changes.
    let weakest: HashMap<usize, f32> = hw
        .iter()
        .enumerate()
        .filter_map(|(g, h)| {
            h.beams
                .iter()
                .filter_map(|j| strength.get(j).copied())
                .filter(|&s| s > 0.0)
                .min_by(|a, b| a.total_cmp(b))
                .map(|s| (g, s))
        })
        .collect();
    for (j, mut s) in strength {
        if s <= 0.0 {
            match hw
                .iter()
                .position(|h| h.beams.contains(&j))
                .and_then(|g| weakest.get(&g))
            {
                Some(&w) => s = w,
                None => {
                    let (a, b) = (sk.beams[j].a, sk.beams[j].b);
                    return Err(format!(
                        "hardware beam {j} ({a} at {:.2?} to {b} at {:.2?}) carries nothing in its rigs",
                        sk.pos(a),
                        sk.pos(b)
                    ));
                }
            }
        }
        sk.beams[j].mat.deform = s;
        sk.beams[j].mat.strength = s;
    }
    Ok(())
}

/// Preload each spring so it carries its corner's share of the sprung mass at
/// the spawn's geometry (the car spawns at its ride height).
fn preload(sk: &mut Skeleton, def: &VehicleDef) {
    if def.corners.len() < 2 {
        return;
    }
    let m: f32 = sk.nodes.iter().map(|n| n.mass).sum();
    let zc = sk.nodes.iter().map(|n| n.mass * n.p.z).sum::<f32>() / m;
    let unsprung: Vec<f32> = def
        .corners
        .iter()
        .map(|c| c.wheel_mass + c.hub_mass)
        .collect();
    let (zf, zr) = (def.corners[0].z, def.corners[1].z);
    let total = m + 2.0 * unsprung.iter().sum::<f32>();
    let front = total * (zc - zr) / (zf - zr);
    let axle = [front, total - front];
    for (si, sh) in sk.shocks.iter_mut().enumerate() {
        let ci = (si / 2).min(1);
        let load = 0.5 * (axle[ci] - 2.0 * unsprung[ci]) * 9.807;
        let len = sk.nodes[sh.a as usize]
            .p
            .distance(sk.nodes[sh.b as usize].p);
        sh.precomp = 1.0 + load / (sh.spring * len);
    }
}

/// The collision skin: rings and cap rows thinned (every other ring, every other
/// node) until it fits the solver's 3,000 triangles; closed like the fine skin.
pub fn coarse_skin(sk: &Skeleton, shell: &Shell) -> Result<Vec<[u32; 3]>, String> {
    use super::body::orient;
    // Minor rings ([`super::body::Interior`]): the fine skin itself (the
    // strides below assume full rings).
    if shell.rings.iter().any(|r| r.minor) {
        if shell.tris.len() > 2900 {
            return Err("the collision skin does not fit 3,000 triangles".into());
        }
        let tris = orient(shell.tris.clone(), &|i| sk.pos(i));
        check_closed(&tris).map_err(|e| format!("collision skin: {e}"))?;
        return Ok(tris);
    }
    for (ring_stride, node_stride) in [(1usize, 1usize), (2, 1), (1, 2), (2, 2), (3, 2), (3, 3)] {
        let tris = coarse_with(sk, shell, ring_stride, node_stride);
        if tris.len() <= 2900 {
            let tris = orient(tris, &|i| sk.pos(i));
            check_closed(&tris).map_err(|e| format!("collision skin: {e}"))?;
            return Ok(tris);
        }
    }
    Err("the collision skin does not fit 3,000 triangles".into())
}

#[allow(dead_code)]
fn thin(v: &[u32], stride: usize) -> Vec<u32> {
    let mut out: Vec<u32> = v.iter().copied().step_by(stride).collect();
    if *out.last().unwrap() != *v.last().unwrap() {
        out.push(*v.last().unwrap());
    }
    out
}

fn coarse_with(sk: &Skeleton, shell: &Shell, rs: usize, ns: usize) -> Vec<[u32; 3]> {
    let idx: Vec<usize> = {
        let mut v: Vec<usize> = (0..shell.rings.len()).step_by(rs).collect();
        if *v.last().unwrap() != shell.rings.len() - 1 {
            v.push(shell.rings.len() - 1);
        }
        v
    };
    let mut halves: Vec<(Vec<u32>, Vec<f32>)> = Vec::new();
    for &k in &idx {
        let r = &shell.rings[k];
        let end = k == 0 || k == shell.rings.len() - 1;
        let ns = if end { 1 } else { ns };
        let keep: Vec<usize> = {
            let all: Vec<usize> = (0..r.half.len()).collect();
            let mut v: Vec<usize> = all.iter().copied().step_by(ns).collect();
            if *v.last().unwrap() != r.half.len() - 1 {
                v.push(r.half.len() - 1);
            }
            v
        };
        halves.push((
            keep.iter().map(|&i| r.half[i]).collect(),
            keep.iter().map(|&i| r.v[i]).collect(),
        ));
    }
    let mut tris = Vec::new();
    let mut loft = Vec::new();
    for w in halves.windows(2) {
        super::body::zipper_pub(&w[0].0, &w[0].1, &w[1].0, &w[1].1, &mut loft);
    }
    for t in &loft {
        tris.push(*t);
        tris.push([
            shell.mirror[&t[0]],
            shell.mirror[&t[2]],
            shell.mirror[&t[1]],
        ]);
    }
    // Caps at full resolution (the end rings are kept whole).
    tris.extend_from_slice(&shell.cap_tris);
    tris
}

/// Triangles over a cloud of nodes: each node with pairs of its nearest
/// neighbours, skipping slivers (for binding visual parts to structure).
pub fn cloud_tris(sk: &Skeleton, nodes: &[u32]) -> Vec<[u32; 3]> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &a in nodes {
        let pa = sk.pos(a);
        let near = sk.nearest(pa, nodes, 6);
        let near: Vec<u32> = near.into_iter().filter(|&b| b != a).collect();
        for i in 0..near.len() {
            for j in i + 1..near.len() {
                let (b, c) = (near[i], near[j]);
                let (u, w) = (sk.pos(b) - pa, sk.pos(c) - pa);
                let s = u.cross(w).length() / (u.length() * w.length()).max(1e-9);
                if s < 0.4 {
                    continue;
                }
                let mut k = [a, b, c];
                k.sort();
                if seen.insert(k) {
                    out.push([a, b, c]);
                }
            }
        }
    }
    out
}
