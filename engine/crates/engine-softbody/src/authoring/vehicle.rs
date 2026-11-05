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

/// A door frame's hinge and latch beams to the pillars: yield force (N) of
/// each (two to four a door end), above FMVSS 206's 11.1 kN for a hinge or
/// latch system. With them the door ring carries a side pole into the
/// pillars: the NCAP side pole's residual crush lands in the measured
/// sedans' band (the `o` test). A 60 kN pull still tears the door off (`g`).
pub const DOOR_HOLD: f32 = 1.5e4;

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

/// Build the skeleton and its `.truck`.
pub fn build(def: &VehicleDef, params: &BuildParams) -> Result<Built, String> {
    let grades = def.grades.scaled(params.strength);
    let res = params.resolution;
    let len = def.profile.length;
    let mut sk = Skeleton::new(&def.name);
    let shell = build_shell(&def.profile, &res.spacing(), &def.stations, &mut sk);
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
                    let n = group_ids.len() as u32 + 1;
                    *group_ids.entry(part).or_insert(n)
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
                if at_hinge && (latch_y.is_none_or(|r| q.y <= r[1] + 0.05)) {
                    sk.beam(a, b, Material::weld(m, grades.hsla).scaled(*hinge_scale));
                } else if in_latch {
                    sk.beam(a, b, Material::weld(m, grades.hsla).scaled(*latch_scale));
                } else {
                    // The closed panel's flange and seal along the rest of its edge:
                    // holds it in its frame until a crash tears it.
                    sk.beam(a, b, m.scaled(0.5));
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
    let mut trusses: Vec<Truss> = Vec::new();
    let mut truss_names: Vec<String> = Vec::new();
    let mut snap_pool: Vec<u32> = shell.nodes.clone();
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
            let host = sk.nearest(mid, &shell.nodes, 1)[0];
            let part = if m.ride {
                sk.nodes[host as usize].part
            } else {
                0
            };
            let spec = m.clone();
            let mat = move |t: f32| base.scaled(spec.scale_at(t));
            let truss: Truss = box_truss(
                &mut sk,
                &path,
                m.width,
                m.height,
                up,
                if m.station > 0.0 {
                    m.station
                } else if m.beads > 0 {
                    res.crush()
                } else {
                    res.truss()
                },
                part,
                &mat,
                m.beads,
                m.section,
                m.catch,
                &snap_pool,
            );
            snap_pool.extend(truss.stations.iter().flatten().copied());
            // The frame: every member's stations (not its catchers); a door's
            // beams are its own heavy frame.
            for &i in truss.stations.iter().flatten() {
                sk.nodes[i as usize].primary = true;
            }
            // Catchers too: a pole's whole load goes through their ties into
            // the member (scaled with the light skin they would let it pass).
            for &i in &truss.catchers {
                sk.nodes[i as usize].primary = CATCHERS_IN_FRAME;
            }
            let nodes = truss.nodes();
            trusses.push(truss.clone());
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
                let nodes: Vec<u32> = truss
                    .stations
                    .iter()
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
                attach(
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
                    for &a in &ends {
                        // Hinges and latch: the door frame held to the
                        // pillars both ways (the next nearest pillar nodes),
                        // torn off past a few kN each.
                        for b in sk.nearest(sk.pos(a), &body, 4).into_iter().skip(2) {
                            let l = sk.pos(a).distance(sk.pos(b));
                            if l <= 0.35 {
                                let sc = l.max(super::material::REF_LENGTH)
                                    / super::skeleton::TIE_LENGTH.max(l);
                                let m = Material {
                                    deform: DOOR_HOLD,
                                    strength: 1.5 * DOOR_HOLD,
                                    ..grades.mount
                                };
                                sk.beam(a, b, m.stiffer(sc));
                            }
                        }
                        for b in sk.nearest(sk.pos(a), &body, 2) {
                            let l = sk.pos(a).distance(sk.pos(b));
                            if l <= 0.3 {
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
            let key = if side > 0.0 || !m.mirrored {
                m.name.clone()
            } else {
                format!("{}_r", m.name)
            };
            groups.insert(key, nodes.clone());
            structure_nodes.extend(nodes);
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
        for &pt in &parts {
            let mut frame: Vec<u32> = (0..sk.nodes.len() as u32)
                .filter(|&i| sk.nodes[i as usize].part == pt && sk.nodes[i as usize].primary)
                .collect();
            // A panel without a frame of its own (a fender) rides the body's
            // behind it (apron, tower, radiator support).
            if frame.is_empty() {
                frame = (0..sk.nodes.len() as u32)
                    .filter(|&i| {
                        sk.nodes[i as usize].part == 0
                            && sk.nodes[i as usize].primary
                            && !sk.nodes[i as usize].catcher
                    })
                    .collect();
            }
            let skin: Vec<u32> = shell
                .nodes
                .iter()
                .copied()
                .filter(|&i| sk.nodes[i as usize].part == pt && !sk.nodes[i as usize].primary)
                .collect();
            attach(&mut sk, &skin, &frame, 2, 0.3, backing);
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
    let mut station_of: HashMap<u32, Vec<u32>> = HashMap::new();
    for t in &trusses {
        for st in &t.stations {
            let mut v: Vec<u32> = st.to_vec();
            v.sort();
            v.dedup();
            for &i in &v {
                station_of.entry(i).or_insert_with(|| v.clone());
            }
        }
    }
    let grades_strength = params.strength;
    let spread = true;
    // Shear panels (the members' and shell's body nodes).
    for br in &def.braces {
        let mat = grades.get(&br.grade)?;
        // The frame's nodes (a shear panel's corners are the frame's).
        let body: Vec<u32> = (0..sk.nodes.len() as u32)
            .filter(|&i| {
                sk.nodes[i as usize].part == 0
                    && sk.nodes[i as usize].primary
                    && !sk.nodes[i as usize].catcher
            })
            .collect();
        for side in [1.0f32, -1.0] {
            let p = |q: [f32; 3]| Vec3::new(side * q[0], q[1], q[2]);
            // Each point's node is found on the +x side and mirrored (ties in
            // distance would otherwise pick different nodes on each side, and
            // braces sharing a corner would miss each other).
            let pick = |q: [f32; 3]| -> u32 {
                let left = Vec3::new(q[0].abs(), q[1], q[2]);
                let n = sk.nearest(left, &body, 1)[0];
                let want = Vec3::new(
                    sk.pos(n).x * (side * q[0].signum()),
                    sk.pos(n).y,
                    sk.pos(n).z,
                );
                sk.nearest(want, &body, 1)[0]
            };
            let _ = p;
            let a = pick(br.from);
            let b = pick(br.to);
            if std::env::var("SHOWCASE_BRACES").is_ok() {
                let deg = |i: u32| sk.beams.iter().filter(|x| x.a == i || x.b == i).count();
                eprintln!(
                    "brace {} {side}: {a} at {:.2?} (degree {}) to {b} at {:.2?} (degree {})",
                    br.name,
                    sk.pos(a),
                    deg(a),
                    sk.pos(b),
                    deg(b)
                );
            }
            // The panel's yield and break at the grade's strains: its forces
            // scale with its rigidity (a lumped panel is many strips).
            let ends = |i: u32| -> Vec<u32> {
                match station_of.get(&i) {
                    Some(v) if spread => v.clone(),
                    _ => vec![i],
                }
            };
            let (ea, eb) = (ends(a), ends(b));
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
        let frame: Vec<u32> = nodes
            .iter()
            .copied()
            .filter(|&i| sk.nodes[i as usize].primary)
            .collect();
        let nodes = if frame.is_empty() { nodes } else { frame };
        sk.add_mass_by_need(&nodes, m.kg, &need);
        named += m.kg;
    }
    let blocks: f32 =
        def.blocks.iter().map(|b| b.kg).sum::<f32>() + def.plates.iter().map(|p| p.kg).sum::<f32>();
    let wheels: f32 = def
        .corners
        .iter()
        .map(|c| 2.0 * (c.wheel_mass + c.hub_mass + c.arm_mass))
        .sum();
    let structure = def.kerb_mass - named - blocks - wheels;
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
    if def.body_damping > 0.0 {
        let n = sk.nodes.len() as u32;
        let susp_part = sk
            .parts
            .iter()
            .position(|p| p == "suspension")
            .map(|x| x as u16);
        let reach = def.body_damping_reach;
        for i in 0..n {
            let ni = &sk.nodes[i as usize];
            if Some(ni.part) == susp_part {
                continue;
            }
            let axis = [Vec3::X, Vec3::Y, Vec3::Z][(i % 3) as usize];
            let pi = ni.p;
            let best = (0..n)
                .filter(|&j| j != i && sk.nodes[j as usize].part == ni.part)
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
                sk.dashpot(i, j, c, def.body_damping_strength);
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
    let (stiffness_scale, skin_scale) = if def.tiered {
        sk.fit_tiers(structure, 0.95)
    } else {
        let s = sk.fit_stiffness(structure, 0.95);
        (s, s)
    };
    sk.solve_masses(structure)?;
    preload(&mut sk, def);
    // Collision skin: the shell's skin on a coarser lattice, its nodes contacters.
    sk.collision = coarse_skin(&sk, &shell)?;
    for t in sk.collision.clone() {
        for i in t {
            sk.nodes[i as usize].contacter = true;
        }
    }
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
        bind_tris,
        groups,
        stiffness_scale,
        skin_scale,
    })
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
