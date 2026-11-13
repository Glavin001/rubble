//! The body-in-white: every structural member of the unibody defined once.
//!
//! One definition per member (`engine/docs/beamng-structure-study.md` §10.2,
//! §10.3): its centreline or surface from the design's key lines, its
//! section, gauge and steel grade. From that definition come
//!
//! - its **visual**: a closed solid (a box or hat section swept along the
//!   centreline, or a pressed panel as a slab), grouped into the
//!   `structure_*` render parts;
//! - its **physics**: the shell nodes that sit on its surface, with the
//!   masses of the study's 131-node map (24 groups), and the member's steel
//!   mass, squash load (σ_y·A) and mean crush force, which the rigger turns
//!   into beam yield forces.
//!
//! Members whose visible surface is the body shell's own skin (the roof
//! panel, the quarter panels, the floor pan's underside, the B-pillars built
//! in the design) carry no mesh here (`on_shell`): their nodes lie on that
//! skin.
//!
//! Section sizes are the study's starting values (typical steel-unibody
//! practice), placed from the design's own lines: every coordinate below is
//! a reference coordinate mapped by the design's [`Frame`], or a point taken
//! from the skin itself.

use glam::{DVec2, DVec3};

use crate::frame::Frame;
use crate::mesh::{orient_outward, Mat, Mesh};
use crate::tube::Tube;

pub const RHO_STEEL: f64 = 7850.0;

/// Sheet steel grades (yield and ultimate strength, Pa).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    /// drawing steel: outer panels, floor, firewall
    Mild,
    /// high-strength low-alloy: rails, sills, crash boxes
    Hsla,
    /// dual phase: pillars, roof rails, towers
    Dp600,
    /// press-hardened: B-pillar reinforcement
    Boron,
}

impl Grade {
    pub fn yield_pa(self) -> f64 {
        match self {
            Grade::Mild => 180e6,
            Grade::Hsla => 340e6,
            Grade::Dp600 => 380e6,
            Grade::Boron => 1000e6,
        }
    }
    pub fn ultimate_pa(self) -> f64 {
        match self {
            Grade::Mild => 300e6,
            Grade::Hsla => 450e6,
            Grade::Dp600 => 600e6,
            Grade::Boron => 1400e6,
        }
    }
    /// The flow stress a folding thin wall works at.
    pub fn flow_pa(self) -> f64 {
        0.5 * (self.yield_pa() + self.ultimate_pa())
    }
    pub fn name(self) -> &'static str {
        match self {
            Grade::Mild => "mild",
            Grade::Hsla => "hsla",
            Grade::Dp600 => "dp600",
            Grade::Boron => "boron",
        }
    }
}

/// A member's cross section (metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Section {
    /// closed box, `w` across by `h`
    Box { w: f64, h: f64 },
    /// hat (top hat on a panel), `w` across by `h` high
    Hat { w: f64, h: f64 },
    /// pressed panel
    Panel,
}

/// One body-in-white member.
#[derive(Clone, Debug)]
pub struct Member {
    /// unique: `front_rail_L`
    pub name: String,
    /// one of the study's member types: `front_rail`
    pub kind: &'static str,
    /// the render part its solid is merged into (`structure_front`, …);
    /// `body_shell` for a member whose surface is the shell's skin
    pub part: &'static str,
    pub section: Section,
    /// sheet thickness (m)
    pub gauge: f64,
    pub grade: Grade,
    /// centreline length (m; 0 for a panel)
    pub length: f64,
    /// sheet area (m²): a section's perimeter × length, a panel's face
    pub sheet_area: f64,
    /// the closed visual solid (empty when `on_shell`)
    pub mesh: Mesh,
    pub on_shell: bool,
}

impl Member {
    /// Steel cross-section area (m²; 0 for a panel).
    pub fn steel_area(&self) -> f64 {
        if self.length > 0.0 {
            self.sheet_area / self.length * self.gauge
        } else {
            0.0
        }
    }
    pub fn steel_kg(&self) -> f64 {
        self.sheet_area * self.gauge * RHO_STEEL
    }
    /// σ_y·A: the axial force at which the whole section yields (N).
    pub fn squash_n(&self) -> f64 {
        self.grade.yield_pa() * self.steel_area()
    }
    /// Mean axial crush force of a thin-walled closed section folding
    /// progressively, P = 13.06·σ₀·t^(5/3)·b^(1/3) with b the mean side (the
    /// standard square-tube result; a hat on its panel folds as a box). N.
    pub fn crush_n(&self) -> f64 {
        let b = match self.section {
            Section::Box { w, h } | Section::Hat { w, h } => 0.5 * (w + h),
            Section::Panel => return 0.0,
        };
        13.06 * self.grade.flow_pa() * self.gauge.powf(5.0 / 3.0) * b.powf(1.0 / 3.0)
    }
}

/// A shell node: a point on a member's surface, with its mass.
#[derive(Clone, Debug)]
pub struct ShellNode {
    /// `sh_a1_ul`
    pub name: String,
    /// the study's group (§10.3): `A1` … `E7`
    pub group: &'static str,
    pub pos: DVec3,
    pub mass: f64,
    /// index into [`Structure::members`]: the member it sits on
    pub member: usize,
}

/// The 24 groups of the study's shell lattice: (group, nodes, kg each).
pub const GROUPS: [(&str, usize, f64); 24] = [
    ("A1", 4, 3.5),
    ("A2", 4, 4.0),
    ("A3", 4, 3.0),
    ("A4", 2, 6.0),
    ("A5", 2, 6.0),
    ("A6", 2, 8.5),
    ("A7", 2, 4.0),
    ("A8", 4, 3.5),
    ("B1", 4, 3.5),
    ("B2", 4, 4.2),
    ("C1", 20, 4.5),
    ("C2", 10, 3.5),
    ("C3", 5, 3.5),
    ("D1", 4, 1.75),
    ("D2", 4, 2.5),
    ("D3", 4, 2.5),
    ("D4", 3, 1.7),
    ("E1", 12, 5.0),
    ("E2", 2, 6.0),
    ("E3", 3, 4.5),
    ("E4", 6, 3.0),
    ("E5", 20, 2.8),
    ("E6", 4, 2.8),
    ("E7", 2, 3.0),
];

pub fn group_mass(g: &str) -> f64 {
    GROUPS.iter().find(|x| x.0 == g).unwrap_or_else(|| panic!("no shell group {g}")).2
}

/// The body-in-white: members and the shell's nodes.
#[derive(Clone, Debug, Default)]
pub struct Structure {
    pub members: Vec<Member>,
    pub nodes: Vec<ShellNode>,
    /// Fixed trim that became parts of its own (kg): taken off the nodes
    /// that carried it ([`Structure::take_trim`]).
    pub trim_removed: f64,
}

/// The groups whose nodes carry the fixed trim (carpet, insulation, wiring,
/// small equipment): the floor and tunnel, the firewall, the boot floor and
/// the quarters.
pub const TRIM_GROUPS: [&str; 5] = ["C1", "C3", "B1", "E3", "E5"];

impl Structure {
    /// The members' solids merged per render part, in a stable order.
    pub fn parts(&self) -> Vec<(String, Mesh)> {
        let mut out: Vec<(String, Mesh)> = Vec::new();
        for m in &self.members {
            if m.on_shell || m.mesh.v.is_empty() {
                continue;
            }
            match out.iter_mut().find(|p| p.0 == m.part) {
                Some(p) => {
                    p.1.add(&m.mesh);
                }
                None => out.push((m.part.to_string(), m.mesh.clone())),
            }
        }
        out
    }

    pub fn member(&self, name: &str) -> Option<&Member> {
        self.members.iter().find(|m| m.name == name)
    }

    pub fn node(&self, name: &str) -> Option<&ShellNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// Σ node mass (kg).
    pub fn node_mass(&self) -> f64 {
        self.nodes.iter().map(|n| n.mass).sum()
    }

    /// Trim and equipment that had no part of its own was mass on the
    /// shell's nodes (the study's "fixed trim", §10.1). When `kg` of it
    /// becomes parts with their own nodes, the nodes of [`TRIM_GROUPS`] give
    /// it up in proportion to their mass, so the car's mass stays.
    pub fn take_trim(&mut self, kg: f64) {
        let carried: f64 = self.nodes.iter().filter(|n| TRIM_GROUPS.contains(&n.group)).map(|n| n.mass).sum();
        assert!(kg < 0.6 * carried, "{kg} kg of trim is more than the floor, firewall and quarter nodes can give ({carried} kg)");
        let k = 1.0 - kg / carried;
        for n in self.nodes.iter_mut().filter(|n| TRIM_GROUPS.contains(&n.group)) {
            n.mass *= k;
        }
        self.trim_removed += kg;
    }

    /// The fixed trim still on the shell's nodes (kg), for a body-in-white
    /// of `biw` kg (the design's `physics.biw`).
    pub fn fixed_trim(&self, biw: f64) -> f64 {
        self.node_mass() - biw
    }

    /// Σ steel of the members that have a section or a face here (kg): the
    /// modelled part of the body-in-white. The rest of the shell's node mass
    /// is the outer skin (its own solid) and the fixed trim (sealer, wiring,
    /// carpet, insulation, paint), listed as such, not hidden.
    pub fn steel_mass(&self) -> f64 {
        self.members.iter().map(|m| m.steel_kg()).sum()
    }
}

/// The skin a structure is built inside: the design's surface and end faces.
pub struct Skin<'a> {
    pub tube: &'a Tube,
    pub front_z: &'a dyn Fn(f64, f64) -> f64,
    pub rear_z: &'a dyn Fn(f64, f64) -> f64,
}

impl Skin<'_> {
    /// The left side's point and outward normal at height `y` (between the
    /// sill and the shoulder).
    pub fn side(&self, z: f64, y: f64) -> (DVec3, DVec3) {
        let v = self.tube.v_at_height(z, y, 1.0, 6.0);
        self.tube.eval_n(z, v)
    }
    /// The left side's point and normal at the surface parameter `v`.
    pub fn at(&self, z: f64, v: f64) -> (DVec3, DVec3) {
        self.tube.eval_n(z, v)
    }
    /// The top surface's height at half width `x` (hood, roof, deck).
    pub fn top_y(&self, z: f64, x: f64) -> f64 {
        let v = self.tube.v_at_x(z, x.abs(), 6.0, 9.0);
        self.tube.eval(z, v).y
    }
    /// How far inside the skin a point lies (m; negative outside): the
    /// distance to the section's outline at its z, and to the end faces.
    pub fn clearance(&self, p: DVec3) -> f64 {
        let n = 144;
        let vm = self.tube.vm;
        let poly: Vec<DVec2> = (0..n)
            .map(|i| {
                let q = self.tube.eval(p.z, vm * i as f64 / n as f64);
                DVec2::new(q.x, q.y)
            })
            .collect();
        let q = DVec2::new(p.x, p.y);
        let mut inside = false;
        let mut d = f64::INFINITY;
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if (a.y > q.y) != (b.y > q.y) && q.x < a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                inside = !inside;
            }
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.length_squared().max(1e-18)).clamp(0.0, 1.0);
            d = d.min((a + ab * t - q).length());
        }
        let d = if inside { d } else { -d };
        d.min((self.front_z)(p.x.abs(), p.y) - p.z).min(p.z - (self.rear_z)(p.x.abs(), p.y))
    }
}

/// The lines a sedan-type unibody is placed from. Reference coordinates are
/// mapped through `f`; the z values here are reference z (before mapping).
#[derive(Clone, Debug)]
pub struct Lines {
    pub f: Frame,
    /// reference axle z, front and rear
    pub axle_f: f64,
    pub axle_r: f64,
    /// the door's front shut line (the hinge pillar stands ahead of it)
    pub z_door_front: f64,
    /// the windscreen's base and top (the header)
    pub z_screen_base: f64,
    pub z_screen_top: f64,
    /// the B-pillar
    pub z_b: f64,
    /// the rear door's rear shut line (the C-pillar's foot; the floor pan's
    /// skin runs between the two door lines)
    pub z_door_rear: f64,
    /// the rear screen's top (the rear header) and base (the shelf's rear edge)
    pub z_rear_top: f64,
    pub z_rear_base: f64,
    /// the wheel-well liner's inner wall (reference x) and radius (mapped m)
    pub well_x: f64,
    pub well_r: f64,
    /// the wheel centre's height (mapped m)
    pub wheel_y: f64,
}

impl Lines {
    /// Reference z of a station s (0 the front axle, 1 the rear axle).
    pub fn station(&self, s: f64) -> f64 {
        self.axle_f + s * (self.axle_r - self.axle_f)
    }
    /// The five floor rows (reference z): the toe board's foot, mid front
    /// footwell, the B-pillar, mid rear footwell, the heel kick.
    pub fn rows(&self) -> [f64; 5] {
        let r1 = self.z_door_front - 0.03;
        let r3 = self.z_b;
        let r5 = self.z_door_rear + 0.035;
        [r1, 0.5 * (r1 + r3), r3, 0.5 * (r3 + r5), r5]
    }
}

// -- solids ------------------------------------------------------------------------------

/// One station of a swept section: centre, the section's two axes (unit) and
/// its outline in them.
#[derive(Clone)]
pub(crate) struct Ring {
    pub c: DVec3,
    pub a: DVec3,
    pub b: DVec3,
    pub pts: Vec<DVec2>,
}

/// Sweep rings into a closed solid; `caps` are the end faces as quads over
/// ring indices.
pub(crate) fn sweep_rings(rings: &[Ring], caps: &[[usize; 4]], mat: Mat) -> Mesh {
    let n = rings[0].pts.len();
    let mut m = Mesh::new();
    for r in rings {
        for p in &r.pts {
            m.v.push(r.c + r.a * p.x + r.b * p.y);
        }
    }
    let id = |i: usize, j: usize| (i * n + j % n) as u32;
    let mut q = Vec::new();
    for i in 0..rings.len() - 1 {
        for j in 0..n {
            q.push([id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j)]);
        }
    }
    for c in caps {
        q.push([id(0, c[3]), id(0, c[2]), id(0, c[1]), id(0, c[0])]);
        let l = rings.len() - 1;
        q.push([id(l, c[0]), id(l, c[1]), id(l, c[2]), id(l, c[3])]);
    }
    let v = std::mem::take(&mut m.v);
    orient_outward(Mesh::from_faces(v, q, vec![], mat, None, -1))
}

/// Points every `step` or closer along a polyline (its corners kept).
pub(crate) fn resample(path: &[DVec3], step: f64) -> Vec<DVec3> {
    let mut out = vec![path[0]];
    for w in path.windows(2) {
        let n = ((w[1] - w[0]).length() / step).ceil().max(1.0) as usize;
        for k in 1..=n {
            out.push(w[0] + (w[1] - w[0]) * (k as f64 / n as f64));
        }
    }
    out
}

fn path_len(p: &[DVec3]) -> f64 {
    p.windows(2).map(|w| (w[1] - w[0]).length()).sum()
}

/// Section frames along a path: `a` across (up × tangent), `b` the up hint
/// made square to the path.
pub(crate) fn frames(path: &[DVec3], up: DVec3) -> Vec<(DVec3, DVec3)> {
    (0..path.len())
        .map(|i| {
            let t = (path[(i + 1).min(path.len() - 1)] - path[i.saturating_sub(1)]).normalize();
            let a = up.cross(t).normalize();
            (a, t.cross(a).normalize())
        })
        .collect()
}

const STEP: f64 = 0.04;

/// A closed round bar of radius `r` (`n` sides, even) along a path.
pub(crate) fn cyl_along(path: &[DVec3], r: f64, n: usize, mat: Mat) -> Mesh {
    let p = resample(path, STEP);
    let up = if (p[p.len() - 1] - p[0]).normalize().dot(DVec3::Y).abs() > 0.9 { DVec3::Z } else { DVec3::Y };
    let fr = frames(&p, up);
    let pts: Vec<DVec2> = (0..n).map(|k| DVec2::from_angle(std::f64::consts::TAU * (k as f64 + 0.5) / n as f64) * r).collect();
    let caps: Vec<[usize; 4]> = (0..n / 2 - 1).map(|k| [n - 1 - k, k, k + 1, n - 2 - k]).collect();
    let rings: Vec<Ring> = p.iter().zip(&fr).map(|(c, f)| Ring { c: *c, a: f.0, b: f.1, pts: pts.clone() }).collect();
    sweep_rings(&rings, &caps, mat)
}

/// A closed box section `w` across by `h` (along `up`) swept along a path.
pub(crate) fn box_along(path: &[DVec3], up: DVec3, w: f64, h: f64) -> Mesh {
    let p = resample(path, STEP);
    let fr = frames(&p, up);
    let pts = vec![DVec2::new(-w / 2.0, -h / 2.0), DVec2::new(w / 2.0, -h / 2.0), DVec2::new(w / 2.0, h / 2.0), DVec2::new(-w / 2.0, h / 2.0)];
    let rings: Vec<Ring> = p.iter().zip(&fr).map(|(c, f)| Ring { c: *c, a: f.0, b: f.1, pts: pts.clone() }).collect();
    sweep_rings(&rings, &[[0, 1, 2, 3]], Mat::MetalBare)
}

/// A hat section (`w` across, `h` high, flanges `fl` wide, wall `th`) swept
/// along a path lying on the panel it is welded to: the brim at the path,
/// the crown `h` along `up`.
fn hat_along(path: &[DVec3], up: DVec3, w: f64, h: f64, fl: f64, th: f64) -> Mesh {
    let p = resample(path, STEP);
    let fr = frames(&p, up);
    // centreline of the wall, flange to flange
    let c = [(-w / 2.0 - fl, 0.0), (-w / 2.0, 0.0), (-w / 2.0 + 0.008, h), (w / 2.0 - 0.008, h), (w / 2.0, 0.0), (w / 2.0 + fl, 0.0)].map(|(x, y)| DVec2::new(x, y + th / 2.0));
    let n = c.len();
    let nrm: Vec<DVec2> = (0..n)
        .map(|i| {
            let t = (c[(i + 1).min(n - 1)] - c[i.saturating_sub(1)]).normalize();
            DVec2::new(-t.y, t.x)
        })
        .collect();
    let mut pts: Vec<DVec2> = (0..n).map(|i| c[i] + nrm[i] * (th / 2.0)).collect();
    pts.extend((0..n).rev().map(|i| c[i] - nrm[i] * (th / 2.0)));
    let caps: Vec<[usize; 4]> = (0..n - 1).map(|k| [k, k + 1, 2 * n - 2 - k, 2 * n - 1 - k]).collect();
    let rings: Vec<Ring> = p.iter().zip(&fr).map(|(c, f)| Ring { c: *c, a: f.0, b: f.1, pts: pts.clone() }).collect();
    sweep_rings(&rings, &caps, Mat::MetalBare)
}

/// A box whose section changes along its length: stations of (centre, across
/// axis, up axis, w, h).
pub(crate) fn box_stations(st: &[(DVec3, DVec3, DVec3, f64, f64)]) -> Mesh {
    let rings: Vec<Ring> = st
        .iter()
        .map(|&(c, a, b, w, h)| Ring { c, a, b, pts: vec![DVec2::new(-w / 2.0, -h / 2.0), DVec2::new(w / 2.0, -h / 2.0), DVec2::new(w / 2.0, h / 2.0), DVec2::new(-w / 2.0, h / 2.0)] })
        .collect();
    sweep_rings(&rings, &[[0, 1, 2, 3]], Mat::MetalBare)
}

/// Visual wall of a pressed panel (the art spec's least wall is 3 mm; the
/// gauge is the physics).
pub const PANEL_WALL: f64 = 0.003;

/// A pressed panel as a closed slab: `nu` × `nv` points from `p(i, j)`, cells
/// kept where `keep(i, j)` (cell indices).
pub(crate) fn panel(nu: usize, nv: usize, p: impl Fn(usize, usize) -> DVec3, keep: impl Fn(usize, usize) -> bool) -> (Mesh, f64) {
    let mut v = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            v.push(p(i, j));
        }
    }
    let id = |i: usize, j: usize| (i * nv + j) as u32;
    let mut q = Vec::new();
    let mut area = 0.0;
    for i in 0..nu - 1 {
        for j in 0..nv - 1 {
            if keep(i, j) {
                let f = [id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
                let (a, b, c, d) = (v[f[0] as usize], v[f[1] as usize], v[f[2] as usize], v[f[3] as usize]);
                area += 0.5 * ((b - a).cross(c - a).length() + (c - a).cross(d - a).length());
                q.push(f);
            }
        }
    }
    let mut sheet = Mesh::from_faces(v, q, vec![], Mat::MetalBare, None, -1);
    // drop the points no cell uses (a hole's inside)
    let mut used = vec![false; sheet.v.len()];
    for f in &sheet.q {
        for &i in f {
            used[i as usize] = true;
        }
    }
    if used.iter().any(|u| !u) {
        let mut map = vec![0u32; sheet.v.len()];
        let mut nv2 = Vec::new();
        for (i, u) in used.iter().enumerate() {
            if *u {
                map[i] = nv2.len() as u32;
                nv2.push(sheet.v[i]);
            }
        }
        sheet.v = nv2;
        for f in &mut sheet.q {
            *f = f.map(|i| map[i as usize]);
        }
    }
    let mut m = crate::solid::slab(&sheet, PANEL_WALL);
    m.surf = None;
    (orient_outward(m), area)
}

pub(crate) fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

pub(crate) fn mx(p: DVec3, sg: f64) -> DVec3 {
    DVec3::new(p.x * sg, p.y, p.z)
}

// -- the unibody -------------------------------------------------------------------------

struct Builder {
    s: Structure,
}

impl Builder {
    fn boxed(&mut self, name: &str, kind: &'static str, part: &'static str, path: &[DVec3], up: DVec3, w: f64, h: f64, gauge_mm: f64, grade: Grade) -> usize {
        let length = path_len(path);
        self.s.members.push(Member {
            name: name.into(),
            kind,
            part,
            section: Section::Box { w, h },
            gauge: gauge_mm * 1e-3,
            grade,
            length,
            sheet_area: 2.0 * (w + h) * length,
            mesh: box_along(path, up, w, h),
            on_shell: false,
        });
        self.s.members.len() - 1
    }
    fn hat(&mut self, name: &str, kind: &'static str, part: &'static str, path: &[DVec3], up: DVec3, w: f64, h: f64, gauge_mm: f64, grade: Grade) -> usize {
        let length = path_len(path);
        let fl = 0.018;
        self.s.members.push(Member {
            name: name.into(),
            kind,
            part,
            section: Section::Hat { w, h },
            gauge: gauge_mm * 1e-3,
            grade,
            length,
            sheet_area: (w + 2.0 * h + 2.0 * fl) * length,
            mesh: hat_along(path, up, w, h, fl, PANEL_WALL),
            on_shell: false,
        });
        self.s.members.len() - 1
    }
    fn solid(&mut self, name: &str, kind: &'static str, part: &'static str, section: Section, mesh: Mesh, length: f64, sheet_area: f64, gauge_mm: f64, grade: Grade) -> usize {
        self.s.members.push(Member { name: name.into(), kind, part, section, gauge: gauge_mm * 1e-3, grade, length, sheet_area, mesh, on_shell: false });
        self.s.members.len() - 1
    }
    fn shell(&mut self, name: &str, kind: &'static str, sheet_area: f64, gauge_mm: f64, grade: Grade) -> usize {
        self.s.members.push(Member {
            name: name.into(),
            kind,
            part: "body_shell",
            section: Section::Panel,
            gauge: gauge_mm * 1e-3,
            grade,
            length: 0.0,
            sheet_area,
            mesh: Mesh::new(),
            on_shell: true,
        });
        self.s.members.len() - 1
    }
    fn node(&mut self, group: &'static str, tag: &str, pos: DVec3, member: usize) {
        self.s.nodes.push(ShellNode { name: format!("sh_{}_{tag}", group.to_lowercase()), group, pos, mass: group_mass(group), member });
    }
}

const SIDES: [(&str, f64); 2] = [("l", 1.0), ("r", -1.0)];

/// A sedan-type steel unibody inside `skin`, placed from `l`.
///
/// Reference dimensions used below (m, before the frame's mapping): the
/// front rails' lane x 0.43 at bumper height (y 0.44), the rear rails' lane
/// x 0.49, the floor pan at y 0.222, the firewall at the door's front line
/// + 0.07.
pub fn unibody(l: &Lines, skin: &Skin) -> Structure {
    let f = &l.f;
    let r = |x: f64, y: f64, z: f64| f.p(x, y, z);
    let mut b = Builder { s: Structure::default() };
    let up = DVec3::Y;
    let rows = l.rows();

    // ---- front: crash boxes, rails, radiator support, upper rails, towers, aprons
    let (xr, yr) = (0.43, 0.44);
    let (rw, rh) = (0.070, 0.110);
    // the bumper beam's rear face (the beam is its own part): 0.13 behind
    // the nose at the rail lane
    let z_cb0 = f.zi((skin.front_z)(f.x(xr), f.y(yr)) - 0.136);
    let z_cb1 = z_cb0 - 0.18;
    let z_fw = l.z_door_front + 0.07;
    let z_kick = l.axle_f - 0.33;
    let z_rad = z_cb1 - 0.035;
    // the hood's underside over the radiator support, less the hood's inner
    // frame and a gap
    let y_tb = f.yi((skin.top_y(f.z(z_rad), 0.0)).min(skin.top_y(f.z(z_rad), f.x(0.60))) - 0.105);
    for (s, sg) in SIDES {
        let m = b.boxed(&format!("crash_box_{s}"), "crash_box", "structure_front", &[mx(r(xr, yr, z_cb0), sg), mx(r(xr, yr, z_cb1), sg)], up, rw, rh, 1.5, Grade::Hsla);
        for (t, dy) in [("u", rh / 2.0), ("d", -rh / 2.0)] {
            b.node("A1", &format!("{t}{s}"), mx(r(xr, yr, z_cb0), sg) + up * dy, m);
            b.node("A2", &format!("{t}{s}"), mx(r(xr, yr, z_cb1), sg) + up * dy, m);
        }
        // rail: level back to the kick-down, then down to the toe board
        let path = [r(xr, yr, z_cb1), r(xr, yr, z_kick), r(xr, yr - 0.075, z_fw + 0.062)].map(|p| mx(p, sg));
        let m = b.boxed(&format!("front_rail_{s}"), "front_rail", "structure_front", &path, up, rw, rh, 1.8, Grade::Hsla);
        b.node("A4", s, mx(r(xr, yr, l.station(-0.08)), sg) - up * (rh / 2.0), m);
        b.node("A5", s, mx(r(xr, yr, l.station(0.08)), sg) - up * (rh / 2.0), m);
        // radiator support upright
        let foot = r(xr, yr, z_rad) + up * (rh / 2.0);
        b.boxed(&format!("rad_support_upright_{s}"), "radiator_support", "structure_front", &[mx(foot, sg), mx(r(xr, y_tb - 0.02, z_rad), sg)], DVec3::Z, 0.040, 0.060, 1.0, Grade::Mild);
        // upper fender rail: under the hood's shut line, from the hinge
        // pillar to the radiator support's tie bar
        let zs = linspace(f.zi(f.z(l.z_door_front + 0.0405) + 0.0445 + 0.008), z_rad - 0.048, 12);
        let path: Vec<DVec3> = zs
            .iter()
            .map(|&z| {
                let p = skin.at(f.z(z), 6.0).0;
                let t = ((z_fw - z) / (z_fw - z_rad)).clamp(0.0, 1.0);
                // (at the front it turns in to the tie bar's end)
                mx(DVec3::new(p.x - 0.045 - 0.03 * t * t, p.y - 0.115, p.z), sg)
            })
            .collect();
        let m = b.boxed(&format!("upper_rail_{s}"), "upper_rail", "structure_front", &path, up, 0.040, 0.060, 1.2, Grade::Hsla);
        for (t, st) in [("f", -0.12), ("m", 0.05)] {
            let z = l.station(st);
            let p = skin.at(f.z(z), 6.0).0;
            let tt = ((z_fw - z) / (z_fw - z_rad)).clamp(0.0, 1.0);
            b.node("A8", &format!("{t}{s}"), mx(DVec3::new(p.x - 0.045 - 0.03 * tt * tt, p.y - 0.115 + 0.030, p.z), sg), m);
        }
        // strut tower: a pressed cup over the wheel well, and the inner
        // apron from the rail up to it
        let zt = l.station(0.02);
        let y_base = f.yi(l.wheel_y + l.well_r + 0.018);
        let xt = l.well_x + 0.020;
        let y_top = y_base + 0.085;
        let st: Vec<(DVec3, DVec3, DVec3, f64, f64)> = (0..=3)
            .map(|k| {
                let t = k as f64 / 3.0;
                (mx(r(xt, y_base + (y_top - y_base) * t, zt), sg), DVec3::X, DVec3::Z, 0.18 - 0.05 * t, 0.20 - 0.06 * t)
            })
            .collect();
        let tower = b.solid(&format!("strut_tower_{s}"), "strut_tower", "structure_front", Section::Panel, box_stations(&st), 0.0, 4.0 * 0.17 * 0.085 + 0.14 * 0.14, 2.0, Grade::Dp600);
        b.node("A6", s, mx(r(xt, y_top, zt), sg), tower);
        let (za0, za1) = (z_kick + 0.02, z_rad - 0.036);
        let y_ap = y_base - 0.006;
        let x_ap = |y: f64| xr + rw / 2.0 + 0.004 + (l.well_x - 0.045 - xr - rw / 2.0) * ((y - yr - rh / 2.0) / (y_ap - yr - rh / 2.0)).clamp(0.0, 1.0);
        let (nu, nv) = (((za1 - za0) / 0.05) as usize + 2, 6);
        let (mesh, area) = panel(nu, nv, |i, j| {
                let y = yr + rh / 2.0 + 0.004 + (y_ap - yr - rh / 2.0 - 0.004) * j as f64 / (nv - 1) as f64;
                mx(r(x_ap(y), y, za0 + (za1 - za0) * i as f64 / (nu - 1) as f64), sg)
            }, |_, _| true);
        let apron = b.solid(&format!("inner_apron_{s}"), "strut_tower", "structure_front", Section::Panel, mesh, 0.0, area, 0.9, Grade::Mild);
        let ya = 0.5 * (yr + rh / 2.0 + y_ap);
        b.node("A7", s, mx(r(x_ap(ya), ya, zt), sg), apron);
    }
    // radiator support: the upper tie bar and the lower cross member
    let x_tb = 0.66;
    let m = b.boxed("rad_support_upper", "radiator_support", "structure_front", &[r(-x_tb, y_tb, z_rad), r(0.0, y_tb, z_rad), r(x_tb, y_tb, z_rad)], up, 0.060, 0.040, 1.0, Grade::Mild);
    for (t, x) in [("l", xr), ("c", 0.0), ("r", -xr)] {
        b.node("A3", &format!("u{t}"), r(x, y_tb, z_rad) + up * 0.020, m);
    }
    let y_lc = yr - rh / 2.0 - 0.022;
    let m = b.boxed("rad_support_lower", "radiator_support", "structure_front", &[r(-xr - 0.03, y_lc, z_rad), r(0.0, y_lc, z_rad), r(xr + 0.03, y_lc, z_rad)], up, 0.060, 0.040, 1.0, Grade::Mild);
    b.node("A3", "dc", r(0.0, y_lc, z_rad) + up * 0.020, m);

    // ---- firewall with the toe board, and the cowl box
    let y_pan = 0.222;
    let (tun_w, tun_h) = (0.18, 0.16);
    let y_cowl = f.yi(skin.top_y(f.z(z_fw - 0.06), f.x(0.5)) - 0.075);
    let (cowl_h, cowl_d) = (0.100, 0.150);
    let y_fw_top = y_cowl - cowl_h;
    // the sill's inner box, inboard of the B-pillar's foot
    let (sw, sh) = (0.065, 0.140);
    let x_sill = |z: f64| f.xi(skin.side(f.z(z), f.y(0.25)).0.x) - 0.142 - sw / 2.0;
    // the firewall and the cowl span between the sills and the hinge pillars
    let x_fw = x_sill(z_fw) - sw / 2.0 - 0.008;
    {
        // profile up the panel: the floor's edge, the toe board, the wall
        let prof: Vec<(f64, f64)> = {
            let mut p = Vec::new();
            let (z0, y0, y1) = (rows[0], y_pan, 0.42);
            for k in 0..=5 {
                let t = k as f64 / 5.0;
                p.push((z0 + (z_fw - z0) * t, y0 + (y1 - y0) * t));
            }
            for y in linspace(y1, y_fw_top, 10).into_iter().skip(1) {
                p.push((z_fw, y));
            }
            p
        };
        let xs = linspace(-x_fw, x_fw, 37);
        let notch = |i: usize, j: usize| {
            // the transmission's opening: the tunnel's mouth
            let xm = 0.5 * (xs[i] + xs[i + 1]);
            let ym = 0.5 * (prof[j].1 + prof[j + 1].1);
            xm.abs() < tun_w / 2.0 + 0.045 && ym < y_pan + tun_h + 0.02
        };
        let (mesh, area) = panel(xs.len(), prof.len(), |i, j| r(xs[i], prof[j].1, prof[j].0), |i, j| !notch(i, j));
        let m = b.solid("firewall", "firewall", "structure_cabin", Section::Panel, mesh, 0.0, area, 0.9, Grade::Mild);
        for (t, x) in [("ol", 0.60), ("il", 0.24), ("ir", -0.24), ("or", -0.60)] {
            b.node("B1", t, r(x, 0.5 * (0.42 + y_fw_top), z_fw), m);
        }
    }
    let z_cowl = z_fw - cowl_d / 2.0;
    let m = b.boxed("cowl", "cowl", "structure_cabin", &[r(-x_fw, y_cowl - cowl_h / 2.0, z_cowl), r(0.0, y_cowl - cowl_h / 2.0, z_cowl), r(x_fw, y_cowl - cowl_h / 2.0, z_cowl)], up, cowl_d, cowl_h, 0.9, Grade::Mild);
    for (t, x) in [("ol", x_fw - 0.02), ("il", 0.25), ("ir", -0.25), ("or", -x_fw + 0.02)] {
        b.node("B2", t, r(x, y_cowl, z_cowl), m);
    }

    // ---- floor: tunnel, longitudinals, seat cross members; sills
    let floor = b.shell("floor_pan", "floor_pan", 2.0 * 0.70 * (rows[0] - rows[4]), 0.8, Grade::Mild);
    let x_tf = tun_w / 2.0 + 0.03;
    // (the floor skin's solid has an end wall 30 mm high at each door line:
    // the members on the pan stop inside them)
    let (z_p0, z_p1) = (l.z_door_front - 0.030, l.z_door_rear + 0.041);
    let tunnel = b.hat("tunnel", "floor_pan", "structure_floor", &[r(0.0, y_pan, z_p0), r(0.0, y_pan, z_p1)], up, tun_w, tun_h, 1.0, Grade::Hsla);
    let y_sill = y_pan + sh / 2.0 + 0.008;
    let mut sills = [0; 2];
    for (k, (s, sg)) in SIDES.into_iter().enumerate() {
        let zs = linspace(z_fw + 0.016, z_p1, 14);
        let path: Vec<DVec3> = zs.iter().map(|&z| mx(r(x_sill(z), y_sill, z), sg)).collect();
        sills[k] = b.boxed(&format!("sill_{s}"), "sill", "structure_floor", &path, up, sw, sh, 1.4, Grade::Hsla);
        b.hat(&format!("floor_longitudinal_{s}"), "floor_pan", "structure_floor", &[mx(r(xr, y_pan, rows[0] - 0.012), sg), mx(r(xr, y_pan, rows[2] + 0.050), sg)], up, 0.060, 0.045, 1.2, Grade::Hsla);
        for (n, row) in [(3, 2), (4, 3)] {
            let z = rows[row];
            b.hat(&format!("seat_crossmember_{n}{s}"), "seat_crossmember", "structure_floor", &[mx(r(x_tf + 0.005, y_pan, z), sg), mx(r(x_sill(z) - sw / 2.0 - 0.004, y_pan, z), sg)], up, 0.050, 0.060, 1.2, Grade::Hsla);
        }
    }
    for (k, &z) in rows.iter().enumerate() {
        let n = k + 1;
        for (j, (s, sg)) in SIDES.into_iter().enumerate() {
            b.node("C1", &format!("{n}o{s}"), mx(r(x_sill(z), y_sill - sh / 2.0, z), sg), sills[j]);
            b.node("C1", &format!("{n}t{s}"), mx(r(x_tf, y_pan - 0.004, z), sg), floor);
            b.node("C2", &format!("{n}{s}"), mx(r(x_sill(z), y_sill + sh / 2.0, z), sg), sills[j]);
        }
        b.node("C3", &format!("{n}"), r(0.0, y_pan + tun_h + PANEL_WALL, z), tunnel);
    }

    // ---- greenhouse. The shell's own solid is the closed section of the
    // upper A-pillars, the C-pillars, the roof rails, the headers and the
    // bow: its skin and its inner face 30 mm under it, with the headliner
    // straight below and the door frames beside, so there is no room for a
    // second box. Their nodes lie inside that section. The hinge pillars
    // (the A-pillars' lower halves) are boxes; the B-pillars are the
    // design's own solids in the body shell.
    b.shell("roof_panel", "roof_panel", 1.2 * (l.z_screen_top - l.z_rear_top), 0.75, Grade::Mild);
    // a point `d` under the skin at parameter v
    let under = |z: f64, v: f64, d: f64| {
        let (p, n) = skin.at(f.z(z), v);
        p - n * d
    };
    let (va, da) = (7.45, 0.014);
    let z_hdr = l.z_screen_top - 0.045;
    let z_rhdr = l.z_rear_top + 0.045;
    // a closed section's sheet area per metre: the shell's 30 mm box
    let sec = |w: f64| 2.0 * (w + 0.030);
    for (s, sg) in SIDES {
        // hinge pillar: on the sill, up to the cowl, between the firewall's
        // edge and the fender's and the door's inner faces
        let zh = l.z_door_front + 0.0405;
        let x_in = x_fw + 0.006;
        let ys = linspace(y_sill + sh / 2.0 + 0.004, y_cowl - 0.004, 9);
        let st: Vec<(DVec3, DVec3, DVec3, f64, f64)> = ys
            .iter()
            .map(|&y| {
                let x_out = f.xi(skin.side(f.z(zh), f.y(y)).0.x) - 0.064;
                (mx(r(0.5 * (x_in + x_out), y, zh), sg), DVec3::X, DVec3::Z, x_out - x_in, 0.089)
            })
            .collect();
        let len = ys[ys.len() - 1] - ys[0];
        b.solid(&format!("hinge_pillar_{s}"), "a_pillar", "structure_cabin", Section::Box { w: 0.12, h: 0.089 }, box_stations(&st), len, 2.0 * (0.12 + 0.089) * len, 1.6, Grade::Dp600);
        // A-pillar upper: cowl to the header
        let la = (under(l.z_screen_base, va, da) - under(l.z_screen_top, va, da)).length();
        let m = b.shell(&format!("a_pillar_{s}"), "a_pillar", sec(0.07) * la, 1.6, Grade::Dp600);
        let zm = 0.5 * (l.z_screen_base + l.z_screen_top);
        b.node("D1", &format!("m{s}"), mx(under(zm, va, da), sg), m);
        b.node("D1", &format!("t{s}"), mx(under(l.z_screen_top, va, da), sg), m);
        // roof rail: A top to C top
        let rail = b.shell(&format!("roof_rail_{s}"), "roof_rail", sec(0.06) * (l.z_screen_top - l.z_rear_top), 1.2, Grade::Dp600);
        // B-pillar: belt and top
        let bp = b.shell(&format!("b_pillar_{s}"), "b_pillar", sec(0.10) * 1.1, 1.6, Grade::Boron);
        let belt = skin.side(f.z(l.z_b), f.y(0.93)).0;
        b.node("D2", &format!("b{s}"), mx(belt - DVec3::X * 0.075, sg), bp);
        b.node("D2", &format!("t{s}"), mx(under(l.z_b, va, da), sg), rail);
        // C-pillar: the rear header's end down to the belt at the deck
        let (zc0, zc1) = (l.z_rear_top, l.z_rear_base + 0.16);
        let lc = (under(zc0, va, da) - under(zc1, va, da)).length();
        let m = b.shell(&format!("c_pillar_{s}"), "c_pillar", sec(0.10) * lc, 1.2, Grade::Dp600);
        b.node("D3", &format!("t{s}"), mx(under(zc0, va, da), sg), m);
        b.node("D3", &format!("b{s}"), mx(under(zc1, va, da), sg), m);
    }
    for (name, tag, z, w, g) in [("windscreen_header", "h", z_hdr, 0.080, 1.0), ("roof_bow", "b", 0.5 * (z_hdr + z_rhdr), 0.060, 0.9), ("rear_header", "r", z_rhdr, 0.080, 1.0)] {
        let vm = skin.tube.vm;
        let span = 2.0 * under(z, va, da).x;
        let m = b.shell(name, "roof_bow", sec(w) * span, g, Grade::Dp600);
        b.node("D4", tag, under(z, vm / 2.0, da), m);
    }

    // ---- rear: seat pan and boot floor, rails, towers, bulkhead, shelf, tail panel
    let (xq, qw, qh) = (0.49, 0.060, 0.090);
    let z_tail = |x: f64, y: f64| f.zi((skin.rear_z)(f.x(x), f.y(y)));
    let (y_seat, y_boot) = (0.35, 0.47);
    let (z_k0, z_k1, z_s0, z_s1) = (rows[4], rows[4] - 0.03, l.axle_r + 0.115, l.axle_r + 0.045);
    let z_end = z_tail(0.3, 0.5) + 0.075;
    // floor profile (z, y) from the heel kick to the tail panel
    let fprof: Vec<(f64, f64)> = {
        // (the heel kick stands on the floor skin's end wall)
        let y0 = y_pan + 0.022;
        let mut p = vec![(z_k0, y0)];
        for k in 1..=3 {
            p.push((z_k0 + (z_k1 - z_k0) * k as f64 / 3.0, y0 + (y_seat - y0) * k as f64 / 3.0));
        }
        for z in linspace(z_k1, z_s0, 8).into_iter().skip(1) {
            p.push((z, y_seat));
        }
        for k in 1..=3 {
            p.push((z_s0 + (z_s1 - z_s0) * k as f64 / 3.0, y_seat + (y_boot - y_seat) * k as f64 / 3.0));
        }
        let n = ((z_s1 - z_end) / 0.045) as usize + 2;
        for z in linspace(z_s1, z_end, n).into_iter().skip(1) {
            p.push((z, y_boot));
        }
        p
    };
    let x_rf = xq + qw / 2.0 + 0.006;
    let xs = linspace(-x_rf, x_rf, 27);
    // (the heel kick is open at the tunnel's mouth: the driveshaft's way)
    let (mesh, area) = panel(xs.len(), fprof.len(), |i, j| r(xs[i], fprof[j].1, fprof[j].0), |i, j| !((0.5 * (xs[i] + xs[i + 1])).abs() < tun_w / 2.0 + 0.045 && j < 3));
    let rear_floor = b.solid("rear_floor", "boot_floor", "structure_rear", Section::Panel, mesh, 0.0, area, 0.8, Grade::Mild);
    for (k, st) in [0.95, 1.10, 1.26].into_iter().enumerate() {
        let z = l.station(st);
        let y = if z > z_s0 { y_seat } else { y_boot };
        b.node("E3", &format!("{}", k + 1), r(0.0, y, z), rear_floor);
    }
    for (s, sg) in SIDES {
        // rear rail under the seat pan and the boot floor
        // (it rises with the floor's step, a little behind it so it stays
        // under the panel, and is at the boot's level over the half-shaft)
        let (z_q0, z_q1) = (z_s0 - 0.005, z_s1 + 0.005);
        let yq = |z: f64| (if z > z_q0 { y_seat } else if z > z_q1 { y_seat + (y_boot - y_seat) * (z_q0 - z) / (z_q0 - z_q1) } else { y_boot }) - qh / 2.0 - 0.016;
        // (upright sections all the way, so its top keeps its distance
        // under the floor on the step)
        let (za, zb) = (z_k1 - 0.01, z_tail(xq, y_boot - 0.06) + 0.148);
        let n = ((za - zb) / 0.03).ceil() as usize;
        let st: Vec<(DVec3, DVec3, DVec3, f64, f64)> = (0..=n)
            .map(|i| {
                let z = za + (zb - za) * i as f64 / n as f64;
                (mx(r(xq, yq(z), z), sg), DVec3::X, DVec3::Y, qw, qh)
            })
            .collect();
        let len: f64 = st.windows(2).map(|w| (w[1].0 - w[0].0).length()).sum();
        let m = b.solid(&format!("rear_rail_{s}"), "rear_rail", "structure_rear", Section::Box { w: qw, h: qh }, box_stations(&st), len, 2.0 * (qw + qh) * len, 1.6, Grade::Hsla);
        for (k, st) in [0.92, 1.08, 1.26].into_iter().enumerate() {
            let z = l.station(st);
            b.node("E1", &format!("{}d{s}", k + 1), mx(r(xq, yq(z) - qh / 2.0, z), sg), m);
            b.node("E1", &format!("{}u{s}", k + 1), mx(r(xq, yq(z) + qh / 2.0, z), sg), m);
        }
        // rear strut tower
        let zt = l.station(1.0);
        let y_base = f.yi(l.wheel_y + l.well_r + 0.018);
        let xt = l.well_x + 0.050;
        let y_top = y_base + 0.070;
        let st: Vec<(DVec3, DVec3, DVec3, f64, f64)> = (0..=3)
            .map(|k| {
                let t = k as f64 / 3.0;
                (mx(r(xt, y_base + (y_top - y_base) * t, zt), sg), DVec3::X, DVec3::Z, 0.19 - 0.05 * t, 0.19 - 0.05 * t)
            })
            .collect();
        let m = b.solid(&format!("rear_tower_{s}"), "rear_tower", "structure_rear", Section::Panel, box_stations(&st), 0.0, 4.0 * 0.165 * 0.07 + 0.14 * 0.14, 1.8, Grade::Dp600);
        b.node("E2", s, mx(r(xt, y_top, zt), sg), m);
        // quarter panel (the shell's skin): five stations at two heights
        let q = b.shell(&format!("quarter_{s}"), "quarter", 0.55 * (l.z_door_rear - l.station(1.26)), 0.75, Grade::Mild);
        let (zq0, zq1) = (l.z_door_rear - 0.040, l.station(1.25));
        for k in 0..5 {
            let z = zq0 + (zq1 - zq0) * k as f64 / 4.0;
            // the lower row clears the wheel opening
            let dz = f.z(z) - f.z(l.axle_r);
            let ra = l.well_r + 0.060;
            let arch = if dz.abs() < ra { l.wheel_y + (ra * ra - dz * dz).sqrt() } else { 0.0 };
            // (and the bumper cover, which reaches the widest line)
            // (ahead of the opening the rear door's dogleg covers the lower
            // quarter up to the waist crease: the first station is above it)
            let y_lo = (skin.at(f.z(z), 3.0).0.y + 0.045).max(arch + 0.03).max(if k == 0 { skin.at(f.z(z), 4.0).0.y + 0.05 } else { 0.0 });
            let y_hi = skin.at(f.z(z), 5.0).0.y - 0.015;
            for (t, y) in [("d", y_lo), ("u", y_hi.max(y_lo + 0.10))] {
                let (p, n) = skin.side(f.z(z), y);
                b.node("E5", &format!("{}{t}{s}", k + 1), mx(p - n * 0.006, sg), q);
            }
        }
    }
    // rear seat bulkhead (leaning with the seat back) and the parcel shelf
    let y_shelf = f.yi(skin.top_y(f.z(l.z_rear_base + 0.02), f.x(0.3))) - 0.060;
    let (zb0, zb1) = (z_s1 - 0.02, l.z_rear_base + 0.21);
    let x_bh = l.well_x - 0.06;
    {
        let ys = linspace(y_boot + PANEL_WALL, y_shelf, 12);
        let xs = linspace(-x_bh, x_bh, 25);
        let (mesh, area) = panel(xs.len(), ys.len(), |i, j| r(xs[i], ys[j], zb0 + (zb1 - zb0) * j as f64 / 11.0), |_, _| true);
        let m = b.solid("rear_bulkhead", "rear_shelf", "structure_rear", Section::Panel, mesh, 0.0, area, 0.8, Grade::Mild);
        for (s, sg) in SIDES {
            b.node("E7", s, mx(r(0.40, y_shelf, zb1), sg), m);
        }
        let z_sh = l.z_rear_base - 0.01;
        let zs = linspace(zb1 - 0.004, z_sh, 6);
        let x_sh = 0.66;
        let xs = linspace(-x_sh, x_sh, 29);
        let (mesh, area) = panel(xs.len(), zs.len(), |i, j| r(xs[i], y_shelf, zs[j]), |_, _| true);
        let m = b.solid("parcel_shelf", "rear_shelf", "structure_rear", Section::Panel, mesh, 0.0, area, 0.8, Grade::Mild);
        for (s, sg) in SIDES {
            b.node("E6", &format!("r{s}"), mx(r(0.22, y_shelf, z_sh), sg), m);
            b.node("E6", &format!("o{s}"), mx(r(x_sh - 0.03, y_shelf, 0.5 * (zb1 + z_sh)), sg), m);
        }
    }
    // tail panel inner
    {
        let (y0, y1) = (y_boot + PANEL_WALL, 0.80);
        let x_tp = 0.56;
        let xs = linspace(-x_tp, x_tp, 25);
        let ys = linspace(y0, y1, 9);
        let zt = |x: f64, y: f64| z_tail(x.abs(), y) + 0.060;
        let (mesh, area) = panel(xs.len(), ys.len(), |i, j| r(xs[i], ys[j], zt(xs[i], ys[j])), |_, _| true);
        let m = b.solid("tail_panel", "tail_panel", "structure_rear", Section::Panel, mesh, 0.0, area, 0.8, Grade::Mild);
        for (t, x) in [("l", 0.50), ("c", 0.0), ("r", -0.50)] {
            for (h, y) in [("d", y0 + 0.03), ("u", y1 - 0.02)] {
                b.node("E4", &format!("{h}{t}"), r(x, y, zt(x, y)), m);
            }
        }
    }
    b.s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_sum_to_the_studys_map() {
        let n: usize = GROUPS.iter().map(|g| g.1).sum();
        let kg: f64 = GROUPS.iter().map(|g| g.1 as f64 * g.2).sum();
        assert_eq!(n, 131);
        assert!((kg - 487.1).abs() < 0.11, "{kg}");
    }

    #[test]
    fn a_rail_section_crushes_at_tens_of_kilonewtons() {
        let m = Member { name: "r".into(), kind: "front_rail", part: "x", section: Section::Box { w: 0.07, h: 0.11 }, gauge: 1.8e-3, grade: Grade::Hsla, length: 1.0, sheet_area: 0.36, mesh: Mesh::new(), on_shell: false };
        assert!((m.steel_kg() - 0.36 * 1.8e-3 * 7850.0).abs() < 1e-9);
        assert!((m.squash_n() - 340e6 * 0.36 * 1.8e-3).abs() < 1.0);
        let c = m.crush_n();
        assert!((40e3..80e3).contains(&c), "{c}");
    }
}
