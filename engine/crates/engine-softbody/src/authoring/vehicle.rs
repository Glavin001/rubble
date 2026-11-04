//! A vehicle definition (shape, panels, structure, masses, suspension, look) and
//! the build that turns it into a node–beam skeleton and `.truck` text.
//!
//! Coordinates: x across (+x the car's left), y up, z along the car from the rear
//! end (0) to the front (`profile.length`); the `.truck` is centred on the length
//! (z − length / 2), its forward axis +z.

use super::body::{build_shell, check_closed, BodyProfile, Seg, Shell, ShellBeam, Spacing};
use super::curve::{Curve, Outline};
use super::material::{Grades, Material};
use super::members::{attach, block, box_truss, corner, plate, Corner, MemberSpec, Truss};
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
    pub kg: f32,
    /// Mounts: each outer node to this many nearest body nodes within `reach`.
    pub mounts: usize,
    pub reach: f32,
    pub mount_grade: String,
    pub mount_scale: f32,
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

    pub fn spacing(self) -> Spacing {
        match self {
            Resolution::Low => Spacing {
                ring: 0.11,
                side: 0.11,
                top: 0.16,
                floor: 0.26,
                cap: 0.11,
            },
            Resolution::Medium => Spacing {
                ring: 0.10,
                side: 0.10,
                top: 0.12,
                floor: 0.2,
                cap: 0.10,
            },
            Resolution::High => Spacing {
                ring: 0.09,
                side: 0.09,
                top: 0.1,
                floor: 0.14,
                cap: 0.09,
            },
        }
    }

    /// Station spacing of the structural trusses.
    pub fn truss(self) -> f32 {
        match self {
            Resolution::Low => 0.25,
            Resolution::Medium => 0.25,
            Resolution::High => 0.22,
        }
    }

    /// Station spacing of crush members (beaded rails): long enough beams to
    /// yield in compression (the solver keeps every rest length ≥ 0.1 m).
    pub fn crush(self) -> f32 {
        0.3
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
    /// Frame beams may cross into these panels (closed doors carry load, as a
    /// real door ring does).
    #[serde(default)]
    pub frame_through: Vec<String>,
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
    /// Every beam's stiffness and damping were scaled by this so the explicit
    /// step stays stable within the kerb mass (1: the materials as defined).
    pub stiffness_scale: f32,
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
        let (pa, pb) = (sk.pos(a), sk.pos(b));
        let zmid = 0.5 * (pa.z + pb.z);
        let crumple = def.crumple.eval(zmid);
        let mut m = grade_of[&a].weaker(grade_of[&b]);
        if kind == ShellBeam::Bend {
            // Bend beams carry the panel's bending, not its membrane: softer.
            m = m.stiffer(0.35);
        }
        m = m.scaled(crumple);
        let (qa, qb) = (sk.nodes[a as usize].part, sk.nodes[b as usize].part);
        if qa == qb {
            sk.beam(a, b, m);
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
    let mut frame_rings: Vec<usize> = (0..shell.rings.len())
        .step_by(def.frame_every.max(1))
        .collect();
    if *frame_rings.last().unwrap() != shell.rings.len() - 1 {
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
        frame_mat.scaled(def.crumple.eval(z))
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
    for fi in [0, hard.len() - 1] {
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
    let body_pool: Vec<u32> = shell
        .nodes
        .iter()
        .copied()
        .filter(|&i| sk.nodes[i as usize].part == 0)
        .collect();
    let mut groups: HashMap<String, Vec<u32>> = HashMap::new();
    // Members.
    let mut structure_nodes: Vec<u32> = Vec::new();
    let mut trusses: Vec<Truss> = Vec::new();
    let mut truss_names: Vec<String> = Vec::new();
    for m in &def.members {
        let base = grades.get(&m.grade)?;
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
            let (s0, s1) = (m.scale[0], m.scale[1]);
            let mat = move |t: f32| base.scaled(s0 + (s1 - s0) * t);
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
            );
            let nodes = truss.nodes();
            trusses.push(truss.clone());
            truss_names.push(if side > 0.0 || !m.mirrored {
                m.name.clone()
            } else {
                format!("{}_r", m.name)
            });
            if m.attach > 0 {
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
                    0.2,
                    Material::weld(grades.panel, base),
                );
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
        let mat = grades.uhss.stiffer(0.3);
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
                if sk.nodes[a.1[0] as usize].part != sk.nodes[b.1[0] as usize].part {
                    continue;
                }
                for &p in &a.1 {
                    for &q in &b.1 {
                        sk.beam(p, q, mat);
                    }
                }
                let _ = ia;
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
                            for &q in sb {
                                sk.beam(p, q, web);
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
        .filter(|&i| sk.nodes[i as usize].part == 0)
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
            res.truss(),
            part,
            mat,
        );
        let nodes: Vec<u32> = g.iter().flatten().copied().collect();
        sk.add_mass(&nodes, pl.kg);
        let edge: Vec<u32> = nodes.clone();
        attach(
            &mut sk,
            &edge,
            &chassis,
            1,
            0.15,
            mat.scaled(pl.mount_scale),
        );
        groups.insert(pl.name.clone(), nodes);
    }
    // Blocks.
    for bl in &def.blocks {
        let part = sk.part(&bl.name);
        let nodes = block(
            &mut sk,
            Vec3::from_array(bl.centre),
            Vec3::from_array(bl.half),
            bl.spacing,
            part,
            grades.rigid,
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
        attach(&mut sk, &outer, &chassis, bl.mounts, bl.reach, mount);
        groups.insert(bl.name.clone(), nodes);
    }
    // Suspension.
    let susp = sk.part("suspension");
    let mut wheel_nodes = Vec::new();
    for c in &def.corners {
        for side in [1.0f32, -1.0] {
            let cn = corner(&mut sk, c, side, susp, grades.rigid);
            // Pivots and the tower top into the body.
            let mut ties = cn.pivots.clone();
            ties.push(cn.shock_top);
            for &t in &ties {
                for c2 in sk.nearest(sk.pos(t), &chassis, 4) {
                    if sk.pos(c2).distance(sk.pos(t)) <= 0.35 {
                        sk.fixed_beam(t, c2, grades.rigid);
                    }
                }
            }
            wheel_nodes.extend(cn.knuckle.iter().copied());
            wheel_nodes.extend(cn.pivots.iter().copied());
        }
    }
    groups.insert("suspension".into(), wheel_nodes);
    // Named masses.
    let mut named = 0.0;
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
        sk.add_mass_by_stiffness(&nodes, m.kg);
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
        eprintln!(
            "  total floors {:.0} kg, budget {:.0} kg",
            fl.iter().sum::<f32>(),
            structure + sk.nodes.iter().map(|n| n.fixed).sum::<f32>()
        );
    }
    let stiffness_scale = sk.fit_stiffness(structure, 0.9);
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
    Ok(Built {
        def: def.clone(),
        params: params.clone(),
        skeleton: sk,
        shell,
        truck,
        shift,
        groups,
        stiffness_scale,
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
