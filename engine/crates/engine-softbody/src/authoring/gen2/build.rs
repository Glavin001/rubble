//! The generator's builder: nodes, beams with roles, panels (skin triangles with
//! their sheet edges, cross braces and two-bay bends), contact-only nodes in the
//! pole lanes, and the rules that set what the file carries, adopted from the
//! BeamNG authoring census (`engine/docs/beamng-authoring-census.md`, "Authoring
//! rules"):
//!
//! * **mass** ([`Car::assign_masses`]): each part's real mass over its structural
//!   nodes by their share of the part's members (kg/m × length, half to each end)
//!   and of its sheet (area, a third to each corner), the named masses (engine,
//!   occupants, fuel, …) on the structure that carries them, the contact-only
//!   nodes at a fixed light mass taken out of their part's budget;
//! * **stiffness** ([`Car::assign_springs`]): each role's census k·L band
//!   (k = k·L / L), capped only where a node's Gershgorin row would pass
//!   [`row_cap`] × sedan1's bound (the light lane nodes, measured margin);
//! * **damping**: every spring beam at ζ = [`ZETA`] of its pair (no dashpots);
//! * **yield** ([`RoleSpec::eps`]): beamDeform = ε_y·k·L, ε_y the census's
//!   (body 1.15 %, doors 1.45 %, skins 3 %), the crash structure's multiple
//!   calibrated on the frontal; the structure FLT_MAX, hardware in break groups.

use glam::Vec3;
use std::collections::{BTreeMap, HashMap};

/// The fixed explicit step (s).
pub const H: f32 = 0.0005;
/// sedan1's Gershgorin row bound (`skeleton::G_LIMIT` × h²: 1.2 × 4).
pub const ROW_BOUND: f32 = 4.8;
/// Mass of a contact-only node (kg): the agreed pattern's light lane nodes.
pub const CONTACT_MASS: f32 = 0.15;
/// Mass of a shut line's edge node (kg), from its part's budget.
pub const EDGE_MASS: f32 = 0.15;
/// Largest spacing of contact points along a pole lane (m).
pub const LANE: f32 = 0.20;
/// Beams shorter than this take the spring of one this long (BeamNG springs
/// are near uniform within a part: a short beam is not made stiffer).
pub const SPRING_LENGTH: f32 = 0.4;
/// Lane nodes a backing node takes before the next nearest takes them.
pub const BACKING_LOAD: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Part {
    Body,
    Engine,
    DoorFL,
    DoorFR,
    DoorRL,
    DoorRR,
    Hood,
    Trunk,
    FenderL,
    FenderR,
    BumperF,
    BumperR,
}

impl Part {
    pub const ALL: [Part; 12] = [
        Part::Body,
        Part::Engine,
        Part::DoorFL,
        Part::DoorFR,
        Part::DoorRL,
        Part::DoorRR,
        Part::Hood,
        Part::Trunk,
        Part::FenderL,
        Part::FenderR,
        Part::BumperF,
        Part::BumperR,
    ];
    /// JBeam part (and slot, and flexbody group) name.
    pub fn name(self) -> &'static str {
        match self {
            Part::Body => "gen2_body",
            Part::Engine => "gen2_engine",
            Part::DoorFL => "gen2_door_FL",
            Part::DoorFR => "gen2_door_FR",
            Part::DoorRL => "gen2_door_RL",
            Part::DoorRR => "gen2_door_RR",
            Part::Hood => "gen2_hood",
            Part::Trunk => "gen2_trunk",
            Part::FenderL => "gen2_fender_L",
            Part::FenderR => "gen2_fender_R",
            Part::BumperF => "gen2_bumper_F",
            Part::BumperR => "gen2_bumper_R",
        }
    }
    pub fn door(front: bool, left: bool) -> Part {
        match (front, left) {
            (true, true) => Part::DoorFL,
            (true, false) => Part::DoorFR,
            (false, true) => Part::DoorRL,
            (false, false) => Part::DoorRR,
        }
    }
    pub fn is_door(self) -> bool {
        matches!(
            self,
            Part::DoorFL | Part::DoorFR | Part::DoorRL | Part::DoorRR
        )
    }
}

/// What a beam is: its census k·L, yield strain, strength and mass per metre.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// Front rail chords (DP600 hat boxes).
    Rail,
    /// Crush cans: the rails' triggered tips, weaker.
    CrushCan,
    BumperBeam,
    /// Upper load path (apron rail, shotgun) and strut towers.
    Apron,
    Firewall,
    Floor,
    Tunnel,
    Sill,
    PillarA,
    PillarB,
    PillarC,
    RoofRail,
    RoofBow,
    RearRail,
    /// Cross members and shear webs between members.
    Cross,
    /// Diagonals of the members' webs.
    Web,
    /// Panel sheet edges and their cross braces.
    Sheet,
    /// Two-bay bends of single-skin panels.
    Bend,
    /// A door's inner frame and its depth.
    DoorFrame,
    /// Side intrusion beam (door mid row).
    Intrusion,
    /// Contact-only node ties.
    Tie,
    /// A contact-only node's tie to the structure behind it.
    Backing,
    /// Door and lid hinges (break groups).
    Hinge,
    /// Bolts: fenders, bumper fascias.
    Bolt,
    /// A clip or slotted mount (bumper covers): ANISOTROPIC, [`CLIP_COMPRESSION`]
    /// of its spring in compression (the cover slides back on it), the bolt's
    /// spring and strength in tension.
    Clip,
    /// Compression-only seals and stops (door edges on pillars, lids on seats).
    Bearing,
    /// A shut line's edge node tied into its own panel's frame (the panel's
    /// sheet; the step's bound softens these before the panel's own beams).
    EdgeTie,
    /// A SUPPORT stop across a shut line between two edge nodes (the
    /// bound softens it before the panels' own beams).
    EdgeStop,
    /// The powertrain block (rigid).
    Engine,
    /// Engine mounts.
    Mount,
    /// A gapped stop (BOUNDED, no base spring): the engine against the dash.
    Stop,
    /// Suspension links holding the hubs (raycast wheels' axle nodes).
    Susp,
}

/// Which role keeps a pair asked for twice.
fn role_rank(r: Role) -> u8 {
    match r {
        Role::Tie | Role::Backing | Role::Bend => 0,
        Role::Sheet | Role::Bearing | Role::EdgeTie | Role::EdgeStop => 1,
        Role::Web
        | Role::Cross
        | Role::Floor
        | Role::Firewall
        | Role::RoofBow
        | Role::DoorFrame => 2,
        _ => 3,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RoleSpec {
    /// Target k·L (N): the census band (spring = k·L / L).
    pub kl: f32,
    /// A fixed spring instead (N/m), for hardware whose length is incidental.
    pub k_fixed: f32,
    /// Yield strain: beamDeform = ε_y · k · L (∞: never yields).
    pub eps: f32,
    /// Strength (break force), N (∞: never).
    pub strength: f32,
    /// Mass per metre for the mass rule (kg/m), 0 for none.
    pub kg_per_m: f32,
    /// Compression-only.
    pub support: bool,
    /// Carries deformLimitExpansion 1.2 (BeamNG: on 37–49 % of body beams).
    pub limit_expansion: bool,
}

/// Census k·L bands of a part's sheet (N): body 0.8–1.1e6, doors 0.4–1.0e6,
/// bonnet 2.2–4.7e5, boot 2.2–5e5, fenders 1.5–4.3e5, bumper covers 0.4–1.3e5.
pub fn sheet_kl(part: Part) -> f32 {
    match part {
        Part::Body => 0.8e6,
        Part::DoorFL | Part::DoorFR | Part::DoorRL | Part::DoorRR => 0.65e6,
        Part::Hood => 3.3e5,
        Part::Trunk => 3.5e5,
        Part::FenderL | Part::FenderR => 2.8e5,
        Part::BumperF | Part::BumperR => 0.9e5,
        Part::Engine => 1.0e6,
    }
}

/// Census yield strain of a part's sheet: body 1.0–1.3 %, doors 1.3–1.6 %,
/// bonnet 1.5–5.5 %, boot 2–5 %, fenders 1.4–4.6 %, bumper covers 3–21 %.
pub fn sheet_eps(part: Part) -> f32 {
    match part {
        Part::Body => 0.0115,
        Part::DoorFL | Part::DoorFR | Part::DoorRL | Part::DoorRR => 0.0145,
        Part::Hood | Part::Trunk | Part::FenderL | Part::FenderR => 0.03,
        Part::BumperF | Part::BumperR => 0.08,
        Part::Engine => f32::INFINITY,
    }
}

pub fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// The crash structure's yield multiple over the body's 1.15 % (rails, cans,
/// aprons, rear rails), `GEN2_CRUSH`: calibrated on wall56 against NHTSA's
/// frontal residual crush (`crash-reference-targets.md` §1).
pub fn crush_mult() -> f32 {
    // gen3 (2026-11-06): the census yield (the first lattice needed 4.5).
    env_f("GEN2_CRUSH", 1.0)
}

/// The crush cans' yield against the rails' (`GEN2_CAN`): their trigger.
pub fn can_mult() -> f32 {
    env_f("GEN2_CAN", 0.6)
}

/// The cabin's yield multiple (sills, pillars, roof rails), `GEN2_CABIN`.
pub fn cabin_mult() -> f32 {
    // gen3 (2026-11-06): 1.2 (the first lattice needed 6).
    env_f("GEN2_CABIN", 1.2)
}

/// The Gershgorin row cap as a multiple of sedan1's (`GEN2_ROW`, default 2.0:
/// the census's BeamNG parts run at p50 1.1–1.6, p90 1.4–2.6 in JBeam mode).
pub fn row_cap() -> f32 {
    env_f("GEN2_ROW", 3.0)
}

/// Stiffness multiple on every census band (`GEN2_STIFF`, the margin sweep).
pub fn stiff_mult() -> f32 {
    env_f("GEN2_STIFF", 1.0)
}

/// The smallest eigenvalue of a symmetric 3 × 3 and its axis (power
/// iteration on the shifted matrix).
pub fn weakest_axis(m: glam::Mat3) -> (f32, Vec3) {
    let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let shifted = glam::Mat3::from_diagonal(Vec3::splat(tr + 1e-3)) - m;
    // The largest eigenvector of (tr·I − M) is M's smallest.
    let mut v = Vec3::new(0.577, 0.577, 0.577);
    for _ in 0..60 {
        v = (shifted * v).normalize_or(Vec3::X);
    }
    (v.dot(m * v), v)
}

/// A [`Role::Clip`]'s spring in compression as a share of its tension spring.
pub const CLIP_COMPRESSION: f32 = 0.05;

/// Damping ratio of every spring beam (census ζ 0.03–0.05).
pub const ZETA: f32 = 0.04;

/// The explicit step's real bound with margin: the highest spring mode at
/// ω_max·h ≤ 1.85 (symplectic Euler diverges past 2).
pub const OMEGA_H: f32 = 1.85;
/// The bound under centred damping (§39: stable below 2 at any ζ), with margin.
pub const OMEGA_H_CENTRED: f32 = 1.92;

impl Role {
    /// Lane ties and shut-line edge ties: the springs the stability bound
    /// and the row cap soften first.
    pub fn yields_first(self) -> bool {
        matches!(
            self,
            Role::Tie | Role::Backing | Role::EdgeTie | Role::EdgeStop
        )
    }
    pub fn spec(self, part: Part) -> RoleSpec {
        let inf = f32::INFINITY;
        let s = |kl: f32, eps: f32, kg_per_m: f32, lim: bool| RoleSpec {
            kl,
            k_fixed: 0.0,
            eps,
            strength: inf,
            kg_per_m,
            support: false,
            limit_expansion: lim,
        };
        let fixed = |k: f32, eps: f32, strength: f32| RoleSpec {
            k_fixed: k,
            strength,
            ..s(0.0, eps, 0.0, false)
        };
        let e = sheet_eps(part);
        let body = 0.0115;
        match self {
            Role::Rail => s(1.05e6, body * crush_mult(), 5.5, true),
            Role::CrushCan => s(0.9e6, body * can_mult() * crush_mult(), 3.0, true),
            Role::BumperBeam => s(0.9e6, body * 2.0, 4.0, true),
            Role::Apron => s(0.9e6, body * crush_mult(), 3.0, true),
            Role::Firewall => s(0.8e6, body, 4.0, false),
            Role::Floor => s(0.8e6, body, 5.0, false),
            Role::Tunnel => s(1.0e6, body * 1.3, 4.0, true),
            Role::Sill => s(1.1e6, body * cabin_mult(), 6.0, true),
            Role::PillarA => s(1.05e6, body * cabin_mult(), 4.0, true),
            Role::PillarB => s(1.1e6, body * cabin_mult(), 5.0, true),
            Role::PillarC => s(0.95e6, body * 1.2, 3.5, true),
            Role::RoofRail => s(1.0e6, body * cabin_mult(), 3.5, true),
            Role::RoofBow => s(0.8e6, body, 2.0, false),
            Role::RearRail => s(1.0e6, body * crush_mult(), 4.5, true),
            Role::Cross => s(0.85e6, body, 2.5, false),
            Role::Web => s(0.8e6, body, 1.5, false),
            Role::Sheet => s(sheet_kl(part), e, 0.0, false),
            Role::Bend => s(0.6 * sheet_kl(part), e, 0.0, false),
            Role::DoorFrame => s(0.75e6, 0.0145, 0.0, true),
            Role::Intrusion => s(0.9e6, 0.0145 * 2.0, 0.0, true),
            // Elastic: a light lane node follows the panel (its own beams yield).
            Role::Tie | Role::Backing => s(sheet_kl(part), f32::INFINITY, 0.0, false),
            // Hardware (census: hinges 30–98 kN a beam, k 0.5–3e6).
            Role::Hinge => fixed(1.5e6, inf, 40e3),
            Role::Bolt => fixed(0.8e6, inf, 9e3),
            Role::Clip => fixed(0.8e6, inf, 9e3),
            Role::EdgeTie => s(sheet_kl(part), e, 0.0, false),
            Role::EdgeStop => RoleSpec {
                support: true,
                ..fixed(0.6e6, inf, 150e3)
            },
            // Seals and stops: SUPPORT, 100–280 kN.
            Role::Bearing => RoleSpec {
                support: true,
                ..fixed(0.6e6, env_f("GEN3_BEARING_EPS", inf), 150e3)
            },
            Role::Engine => fixed(6e6, inf, inf),
            Role::Mount => fixed(2.0e6, inf, 60e3),
            Role::Susp => fixed(3.0e6, 0.03, 250e3),
            // Free until engaged: no share of the bound at rest.
            Role::Stop => fixed(1.0, f32::INFINITY, f32::INFINITY),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub p: Vec3,
    pub part: Part,
    /// Contact-only (light lane node on short ties).
    pub contact: bool,
    pub mass: f32,
    /// Extra JBeam node options (couplers).
    pub opts: Vec<(String, serde_json::Value)>,
    /// Self-collision (skin nodes).
    pub self_collision: bool,
    /// Excluded from the flexbody groups (latch nodes coincide with tag nodes).
    pub no_group: bool,
    /// A mass of its own (kg), outside its part's budget (0: none).
    pub fixed_mass: f32,
    /// A shut line's edge node: [`EDGE_MASS`] taken from its part's budget
    /// (the panel's sheet stays with the nodes that carried it, so adding
    /// edge nodes does not lighten them; the edge ties give way to the
    /// step's bound instead).
    pub edge: bool,
}

#[derive(Clone, Debug)]
pub struct Beam {
    pub a: usize,
    pub b: usize,
    pub role: Role,
    pub part: Part,
    pub k: f32,
    pub c: f32,
    pub deform: f32,
    pub strength: f32,
    pub break_group: Option<String>,
    /// JBeam deformLimit (compression: L ≥ (1 − d)·L₀), 0 for none.
    pub deform_limit: f32,
}

#[derive(Clone, Debug)]
pub struct Tri {
    pub n: [usize; 3],
    pub part: Part,
    /// JBeam `breakGroup`: the triangle dies with that group (a collision
    /// skin triangle bridging a seam dies with its part's fastening).
    pub group: Option<String>,
}

/// A point mass the structure carries.
#[derive(Clone, Debug)]
pub struct Named {
    pub name: &'static str,
    pub kg: f32,
    /// Nodes it goes on (weights by inverse distance to `at`).
    pub at: Vec3,
    pub radius: f32,
    pub part: Part,
}

#[derive(Clone, Debug)]
pub struct Wheel {
    pub name: &'static str,
    pub n1: usize,
    pub n2: usize,
    pub arm: usize,
    pub dir: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Car {
    pub nodes: Vec<Node>,
    pub beams: Vec<Beam>,
    pub tris: Vec<Tri>,
    pub named: Vec<Named>,
    pub wheels: Vec<Wheel>,
    /// Real part masses (kg) the structural nodes share.
    pub part_mass: BTreeMap<Part, f32>,
    pub refs: [usize; 4],
    /// The highest spring mode after [`Car::assign_springs`] (ω_max·h).
    pub omega_h: f32,
    /// Per node the render surface it stands for (m²; its Voronoi share of
    /// its part's meshes), when the car is rigged from a render model: the
    /// mass rule's sheet term (else the collision triangles' areas).
    pub sheet_area: Option<Vec<f32>>,
    /// The mass rule's floor as a share of a part's mean structural node
    /// (None: 0.5).
    pub mass_floor: Option<f32>,
    /// The Gershgorin row cap only on contact-only nodes' ties (the
    /// structure is left to the measured bound, the power iteration).
    pub row_cap_contact_only: bool,
    /// The car runs with centred damping (the BeamNG profile's default,
    /// jbeam-semantics-spec.md §39): the step is stable to ω·h < 2 whatever
    /// the damping, so the bound is the springs' highest mode at
    /// [`OMEGA_H_CENTRED`], damping not counted.
    pub centred_damping: bool,
    /// The collision triangles' aerodynamics (JBeam `dragCoef`, `liftCoef`),
    /// written as a modifier row before them; None: the JBeam defaults.
    pub tri_aero: Option<(f32, f32)>,
    /// Per [`Role::Stop`] beam: the closure (m) after which it engages
    /// (absent: [`super::export::STOP_LEFT`] of its length left).
    pub stop_close: HashMap<usize, f32>,
    /// Damping ratio of the bumpers' beams (None: [`ZETA`] as every part;
    /// the census's bumper covers run at 0.07-0.33).
    pub bumper_zeta: Option<f32>,
    /// Per beam: JBeam `beamPrecompression` (a preloaded seal or bump stop).
    pub precomp: HashMap<usize, f32>,
    by_name: HashMap<String, usize>,
    pair: HashMap<(usize, usize), usize>,
}

impl Car {
    pub fn node(&mut self, name: &str, p: Vec3, part: Part) -> usize {
        assert!(!self.by_name.contains_key(name), "node {name} twice");
        assert!(p.is_finite(), "node {name} at {p}");
        self.nodes.push(Node {
            name: name.to_string(),
            p,
            part,
            contact: false,
            mass: 0.0,
            opts: Vec::new(),
            self_collision: false,
            no_group: false,
            fixed_mass: 0.0,
            edge: false,
        });
        let i = self.nodes.len() - 1;
        self.by_name.insert(name.to_string(), i);
        i
    }

    pub fn id(&self, name: &str) -> usize {
        *self
            .by_name
            .get(name)
            .unwrap_or_else(|| panic!("no node {name}"))
    }

    pub fn p(&self, i: usize) -> Vec3 {
        self.nodes[i].p
    }

    /// A beam (once per pair: a second request keeps the first's role unless
    /// the new role carries more of the bound).
    pub fn beam(&mut self, a: usize, b: usize, role: Role, part: Part) -> usize {
        assert_ne!(a, b, "beam onto itself: {}", self.nodes[a].name);
        let key = (a.min(b), a.max(b));
        if let Some(&j) = self.pair.get(&key) {
            if role_rank(role) > role_rank(self.beams[j].role)
                && self.beams[j].break_group.is_none()
            {
                self.beams[j].role = role;
            }
            return j;
        }
        let sp = role.spec(part);
        self.beams.push(Beam {
            a,
            b,
            role,
            part,
            k: 0.0,
            c: 0.0,
            deform: 0.0,
            strength: sp.strength,
            break_group: None,
            deform_limit: 0.0,
        });
        let j = self.beams.len() - 1;
        self.pair.insert(key, j);
        j
    }

    pub fn has_beam(&self, a: usize, b: usize) -> bool {
        self.pair.contains_key(&(a.min(b), a.max(b)))
    }

    /// A hardware beam in a break group, with its own strength.
    pub fn hardware(
        &mut self,
        a: usize,
        b: usize,
        role: Role,
        part: Part,
        group: &str,
        strength: f32,
    ) -> usize {
        let j = self.beam(a, b, role, part);
        self.beams[j].role = role;
        self.beams[j].break_group = Some(group.to_string());
        self.beams[j].strength = strength;
        j
    }

    pub fn chain(&mut self, ids: &[usize], role: Role, part: Part) {
        for w in ids.windows(2) {
            self.beam(w[0], w[1], role, part);
        }
    }

    /// Beams from `a` to each of `to`.
    pub fn fan(&mut self, a: usize, to: &[usize], role: Role, part: Part) {
        for &b in to {
            if b != a {
                self.beam(a, b, role, part);
            }
        }
    }

    /// Ladder between two parallel chains: rungs and both diagonals.
    pub fn ladder(
        &mut self,
        a: &[usize],
        b: &[usize],
        chord: Role,
        rung: Role,
        web: Role,
        part: Part,
    ) {
        self.chain(a, chord, part);
        self.chain(b, chord, part);
        for k in 0..a.len().min(b.len()) {
            self.beam(a[k], b[k], rung, part);
            if k + 1 < a.len().min(b.len()) {
                self.beam(a[k], b[k + 1], web, part);
                self.beam(b[k], a[k + 1], web, part);
            }
        }
    }

    /// A collision triangle, oriented outward (away from the car's long axis).
    pub fn tri(&mut self, a: usize, b: usize, c: usize, part: Part) {
        let (pa, pb, pc) = (self.p(a), self.p(b), self.p(c));
        let n = (pb - pa).cross(pc - pa);
        if n.length() < 1e-6 {
            return;
        }
        let cen = (pa + pb + pc) / 3.0;
        let axis = Vec3::new(0.0, cen.y, 0.70);
        let out = if (cen - axis).length() < 0.05 {
            Vec3::Z
        } else {
            cen - axis
        };
        if n.dot(out) >= 0.0 {
            self.tris.push(Tri {
                n: [a, b, c],
                part,
                group: None,
            });
        } else {
            self.tris.push(Tri {
                n: [a, c, b],
                part,
                group: None,
            });
        }
    }

    /// A panel from rows of optional nodes (rows bottom → top or front → rear,
    /// columns in order): sheet edges along rows and columns, both diagonals of
    /// each full cell (sheet), collision triangles, and two-bay bends along rows
    /// and columns when `bends`.
    pub fn panel(
        &mut self,
        rows: &[Vec<Option<usize>>],
        part: Part,
        row_role: &[Role],
        col_role: Role,
        bends: bool,
    ) {
        for (r, row) in rows.iter().enumerate() {
            let ids: Vec<usize> = row.iter().flatten().copied().collect();
            let role = row_role.get(r).copied().unwrap_or(Role::Sheet);
            self.chain(&ids, role, part);
            if bends {
                for w in ids.windows(3) {
                    self.beam(w[0], w[2], Role::Bend, part);
                }
            }
        }
        let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
        for c in 0..cols {
            let ids: Vec<usize> = rows
                .iter()
                .filter_map(|r| r.get(c).copied().flatten())
                .collect();
            self.chain(&ids, col_role, part);
            if bends {
                for w in ids.windows(3) {
                    self.beam(w[0], w[2], Role::Bend, part);
                }
            }
        }
        for r in 0..rows.len().saturating_sub(1) {
            for c in 0..cols.saturating_sub(1) {
                let g = |rr: usize, cc: usize| rows[rr].get(cc).copied().flatten();
                let (a, b, d, e) = (g(r, c), g(r, c + 1), g(r + 1, c), g(r + 1, c + 1));
                match (a, b, d, e) {
                    (Some(a), Some(b), Some(d), Some(e)) => {
                        self.beam(a, e, Role::Sheet, part);
                        self.beam(b, d, Role::Sheet, part);
                        self.tri(a, b, e, part);
                        self.tri(a, e, d, part);
                    }
                    _ => {
                        let v: Vec<usize> = [a, b, e, d].into_iter().flatten().collect();
                        if v.len() == 3 {
                            self.chain(&[v[0], v[1], v[2], v[0]], Role::Sheet, part);
                            self.tri(v[0], v[1], v[2], part);
                        }
                    }
                }
            }
        }
    }

    /// Contact-only nodes in the pole lanes of `part`'s collision triangles:
    /// every triangle edge with a run along the car longer than [`LANE`] gets
    /// evenly spaced lane nodes on the surface (`surf` projects a point onto
    /// it), each triangle so split gets a centre node and is re-triangulated as
    /// a fan, and every lane node is tied to its neighbours along the edge, to
    /// the triangle corners, to the centre nodes, and to the nearest `backing`
    /// node (its support across the panel).
    pub fn lanes(
        &mut self,
        part: Part,
        backing: &[usize],
        surf: &dyn Fn(Vec3) -> Vec3,
        edge_ok: &dyn Fn(Vec3, Vec3) -> bool,
    ) {
        let tris: Vec<usize> = (0..self.tris.len())
            .filter(|&t| self.tris[t].part == part)
            .collect();
        let mut edge_pts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut serial = 0usize;
        let pname = part.name().trim_start_matches("gen2_").to_string();
        let mut new_tris = Vec::new();
        let mut keep = vec![true; self.tris.len()];
        for &t in &tris {
            let [a, b, c] = self.tris[t].n;
            let mut perim: Vec<usize> = Vec::new();
            let mut split = 0usize;
            let mut opposite = a;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                perim.push(u);
                let key = (u.min(v), u.max(v));
                let (pu, pv) = (self.p(key.0), self.p(key.1));
                // Row-like edges only (a run along the car, little rise): the
                // pole lanes cross them; diagonals stay single beams.
                let d = pv - pu;
                let run = d.y.abs();
                let n = ((run / LANE).ceil() as usize).max(1);
                if n <= 1 || d.z.abs() > 0.35 * run || !edge_ok(pu, pv) {
                    continue;
                }
                split += 1;
                opposite = [a, b, c].into_iter().find(|&w| w != u && w != v).unwrap();
                let pts = if let Some(p) = edge_pts.get(&key) {
                    p.clone()
                } else {
                    let mut ids = Vec::new();
                    for k in 1..n {
                        let q = surf(pu.lerp(pv, k as f32 / n as f32));
                        serial += 1;
                        let id = self.node(&format!("x{pname}{serial}"), q, part);
                        self.nodes[id].contact = true;
                        ids.push(id);
                    }
                    let mut chain = vec![key.0];
                    chain.extend(&ids);
                    chain.push(key.1);
                    self.chain(&chain, Role::Tie, part);
                    edge_pts.insert(key, ids.clone());
                    ids
                };
                if u == key.0 {
                    perim.extend(pts.iter().copied());
                } else {
                    perim.extend(pts.iter().rev().copied());
                }
                for &q in &pts {
                    for &corner in &[a, b, c] {
                        if corner != u && corner != v {
                            self.beam(q, corner, Role::Tie, part);
                        }
                    }
                }
            }
            if split == 0 {
                continue;
            }
            keep[t] = false;
            if split == 1 {
                // One split edge: a fan from the corner opposite it.
                let k0 = perim.iter().position(|&q| q == opposite).unwrap();
                let ring: Vec<usize> = (0..perim.len())
                    .map(|k| perim[(k0 + k) % perim.len()])
                    .collect();
                for k in 1..ring.len() - 1 {
                    new_tris.push([ring[0], ring[k], ring[k + 1]]);
                }
                continue;
            }
            let cen = surf((self.p(a) + self.p(b) + self.p(c)) / 3.0);
            serial += 1;
            let ci = self.node(&format!("x{pname}{serial}"), cen, part);
            self.nodes[ci].contact = true;
            for &q in &perim {
                self.beam(ci, q, Role::Tie, part);
            }
            for k in 0..perim.len() {
                new_tris.push([ci, perim[k], perim[(k + 1) % perim.len()]]);
            }
        }
        let old = std::mem::take(&mut self.tris);
        self.tris = old
            .into_iter()
            .zip(keep)
            .filter(|(_, k)| *k)
            .map(|(t, _)| t)
            .collect();
        for t in new_tris {
            self.tri(t[0], t[1], t[2], part);
        }
        // Backing ties: every lane node to its nearest backing node.
        if !backing.is_empty() {
            let lane: Vec<usize> = (0..self.nodes.len())
                .filter(|&i| self.nodes[i].contact && self.nodes[i].part == part)
                .collect();
            let mut load: HashMap<usize, usize> = HashMap::new();
            for i in lane {
                let p = self.p(i);
                let mut best: Vec<(f32, usize)> = backing
                    .iter()
                    .map(|&b| (self.p(b).distance(p), b))
                    .collect();
                best.sort_by(|x, y| x.0.total_cmp(&y.0));
                // The nearest backing node with room (at most `BACKING_LOAD`
                // lane nodes each, so no one node's row carries a whole lane;
                // within 1.5× the nearest's distance), else the nearest.
                // Two backing ties (a lane node on a strip's edge has only
                // its edge and one corner otherwise: soft across the panel).
                let d0 = best.first().map_or(0.0, |b| b.0);
                let mut picked = 0;
                for &(d, b) in &best {
                    if picked == 2 {
                        break;
                    }
                    if d <= 1.5 * d0 + 1e-6 && load.get(&b).copied().unwrap_or(0) < BACKING_LOAD {
                        *load.entry(b).or_insert(0) += 1;
                        self.beam(i, b, Role::Backing, part);
                        picked += 1;
                    }
                }
                if picked == 0 {
                    if let Some(&(_, b)) = best.first() {
                        self.beam(i, b, Role::Backing, part);
                    }
                }
                // Ties that span all three directions: a lane node whose ties
                // lie in one plane (its edge, the opposite corner and a
                // backing node in line with it) has a free mode across that
                // plane and sags under gravity (a 0.65 m sill bay: 29 mm in
                // 10 s). While their directions' second moment has a weak
                // axis, the backing node most along it (within 1.5× the
                // nearest's distance, then within 0.5 m) is tied too.
                for _ in 0..2 {
                    let ties: Vec<Vec3> = self
                        .beams
                        .iter()
                        .filter(|b| b.a == i || b.b == i)
                        .map(|b| {
                            let o = if b.a == i { b.b } else { b.a };
                            (self.p(o) - p).normalize_or_zero()
                        })
                        .collect();
                    let m = ties.iter().fold(glam::Mat3::ZERO, |m, d| {
                        m + glam::Mat3::from_cols(*d * d.x, *d * d.y, *d * d.z)
                    });
                    let (lam, axis) = weakest_axis(m);
                    if lam > 0.08 * ties.len() as f32 {
                        break;
                    }
                    let pick = best
                        .iter()
                        .filter(|&&(d, b)| d <= 0.5 && !self.has_beam(i, b))
                        .max_by(|x, y| {
                            let f = |b: usize| (self.p(b) - p).normalize_or_zero().dot(axis).abs();
                            f(x.1).total_cmp(&f(y.1))
                        })
                        .map(|x| x.1);
                    match pick {
                        Some(b) => {
                            self.beam(i, b, Role::Backing, part);
                        }
                        None => break,
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Mass rule
    // ---------------------------------------------------------------------

    /// Masses: see the module docs. Returns the total.
    pub fn assign_masses(&mut self) -> f32 {
        let n = self.nodes.len();
        let mut w = vec![0.0f32; n];
        for b in &self.beams {
            let kgm = b.role.spec(b.part).kg_per_m;
            if kgm == 0.0 {
                continue;
            }
            let l = self.p(b.a).distance(self.p(b.b));
            for e in [b.a, b.b] {
                if self.nodes[e].part == b.part && !self.nodes[e].contact && !self.nodes[e].edge {
                    w[e] += 0.5 * kgm * l;
                }
            }
        }
        // Sheet: a third of each triangle's area to its structural corners
        // (a lane triangle's area to the structural nodes of its part nearest
        // its centre: the structural nodes stand for the sheet around them).
        let structural_of: HashMap<Part, Vec<usize>> = Part::ALL
            .iter()
            .map(|&p| {
                (
                    p,
                    (0..n)
                        .filter(|&i| self.nodes[i].part == p && !self.nodes[i].contact)
                        .collect(),
                )
            })
            .collect();
        let dens = 0.8 * 7.85 * 3.0; // 0.8 mm steel ≈ 6.3 kg/m², relative weight only
        if let Some(area) = &self.sheet_area {
            for (i, a) in area.iter().enumerate() {
                if i < n && !self.nodes[i].contact {
                    w[i] += a * dens;
                }
            }
        }
        for t in self.tris.iter().filter(|_| self.sheet_area.is_none()) {
            let [a, b, c] = t.n;
            let area = 0.5
                * (self.p(b) - self.p(a))
                    .cross(self.p(c) - self.p(a))
                    .length();
            let dens = 0.8 * 7.85 * 3.0; // 0.8 mm steel ≈ 6.3 kg/m², relative weight only
            for &e in &[a, b, c] {
                let e = if self.nodes[e].contact {
                    let p = self.p(e);
                    let list = &structural_of[&t.part];
                    *list
                        .iter()
                        .min_by(|&&x, &&y| self.p(x).distance(p).total_cmp(&self.p(y).distance(p)))
                        .unwrap()
                } else {
                    e
                };
                if self.nodes[e].part == t.part {
                    w[e] += area * dens / 3.0;
                }
            }
        }
        for i in 0..n {
            if self.nodes[i].fixed_mass > 0.0 {
                self.nodes[i].mass = self.nodes[i].fixed_mass;
            }
        }
        for (&part, &kg) in &self.part_mass.clone() {
            let edges = (0..n)
                .filter(|&i| self.nodes[i].part == part && self.nodes[i].edge)
                .count();
            for i in 0..n {
                if self.nodes[i].part == part && self.nodes[i].edge {
                    self.nodes[i].mass = EDGE_MASS;
                }
            }
            let ids: Vec<usize> = (0..n)
                .filter(|&i| {
                    self.nodes[i].part == part
                        && self.nodes[i].fixed_mass == 0.0
                        && !self.nodes[i].edge
                })
                .collect();
            let contact = ids.iter().filter(|&&i| self.nodes[i].contact).count();
            let budget = kg - contact as f32 * CONTACT_MASS - edges as f32 * EDGE_MASS;
            assert!(
                budget > 0.0,
                "{part:?}: {kg} kg for {contact} contact nodes"
            );
            let mut sw: f32 = ids
                .iter()
                .filter(|&&i| !self.nodes[i].contact)
                .map(|&i| w[i])
                .sum();
            let structural = ids.iter().filter(|&&i| !self.nodes[i].contact).count();
            if sw <= 0.0 {
                for &i in &ids {
                    w[i] = 1.0;
                }
                sw = structural as f32;
            }
            // A floor of half the mean share (BeamNG closures are near uniform): no node is left
            // as light as a lane node.
            let floor = self.mass_floor.unwrap_or(0.5) * sw / structural as f32;
            let sw2: f32 = ids
                .iter()
                .filter(|&&i| !self.nodes[i].contact)
                .map(|&i| w[i].max(floor))
                .sum();
            for &i in &ids {
                self.nodes[i].mass = if self.nodes[i].contact {
                    CONTACT_MASS
                } else {
                    budget * w[i].max(floor) / sw2
                };
            }
        }
        for m in self.named.clone() {
            let ids: Vec<usize> = (0..n)
                .filter(|&i| {
                    self.nodes[i].part == m.part
                        && !self.nodes[i].contact
                        && self.p(i).distance(m.at) <= m.radius
                })
                .collect();
            assert!(
                !ids.is_empty(),
                "named mass {} has no nodes within {} m",
                m.name,
                m.radius
            );
            let ws: Vec<f32> = ids
                .iter()
                .map(|&i| 1.0 / (self.p(i).distance(m.at) + 0.15))
                .collect();
            let s: f32 = ws.iter().sum();
            for (&i, &wi) in ids.iter().zip(&ws) {
                self.nodes[i].mass += m.kg * wi / s;
            }
        }
        self.nodes.iter().map(|n| n.mass).sum()
    }

    // Stiffness: census bands, the stability margin
    // ---------------------------------------------------------------------

    /// Springs, damping and yield: see the module docs. Hub nodes' extra wheel
    /// mass (`hub_extra`, kg per node) counts as the solver adds it.
    pub fn assign_springs(&mut self, hub_extra: &HashMap<usize, f32>) {
        let n = self.nodes.len();
        let m: Vec<f32> = (0..n)
            .map(|i| self.nodes[i].mass + hub_extra.get(&i).copied().unwrap_or(0.0))
            .collect();
        let len: Vec<f32> = self
            .beams
            .iter()
            .map(|b| self.p(b.a).distance(self.p(b.b)).max(0.02))
            .collect();
        let mu = |b: &Beam| m[b.a] * m[b.b] / (m[b.a] + m[b.b]);
        let mult = stiff_mult();
        let mut ks: Vec<f32> = self
            .beams
            .iter()
            .zip(&len)
            .map(|(b, &l)| {
                let sp = b.role.spec(b.part);
                mult * if sp.k_fixed > 0.0 {
                    sp.k_fixed
                } else {
                    sp.kl / l.max(SPRING_LENGTH)
                }
            })
            .collect();
        let bz = self.bumper_zeta;
        let zeta_of = move |b: &Beam| match (b.part, bz) {
            (Part::BumperF | Part::BumperR, Some(z)) => z,
            _ => ZETA,
        };
        let damp = |k: f32, b: &Beam| 2.0 * zeta_of(b) * (k * mu(b)).sqrt();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (j, b) in self.beams.iter().enumerate() {
            adj[b.a].push(j);
            adj[b.b].push(j);
        }
        // The cap: rows past row_cap × sedan1's bound scale their beams back
        // (the lane nodes' ties, mostly).
        let cap = row_cap() * ROW_BOUND;
        for _ in 0..40 {
            let mut over = false;
            for i in 0..n {
                let row: f32 = adj[i]
                    .iter()
                    .map(|&j| {
                        let b = &self.beams[j];
                        let o = if b.a == i { b.b } else { b.a };
                        let q = H * H * ks[j] + 2.0 * H * damp(ks[j], b);
                        q * (1.0 / m[i] + 1.0 / (m[i] * m[o]).sqrt())
                    })
                    .sum();
                let contact_only = self.row_cap_contact_only
                    && !self.nodes[i].contact
                    && adj[i].iter().all(|&j| !self.beams[j].role.yields_first());
                if row > cap * 1.0001 && !contact_only {
                    over = true;
                    // The lane ties give way first: a structural node keeps its
                    // members' census stiffness while its ties can absorb the excess.
                    let tie = |j: usize| self.beams[j].role.yields_first();
                    let part_row = |want_tie: bool| -> f32 {
                        adj[i]
                            .iter()
                            .filter(|&&j| tie(j) == want_tie)
                            .map(|&j| {
                                let b = &self.beams[j];
                                let o = if b.a == i { b.b } else { b.a };
                                let q = H * H * ks[j] + 2.0 * H * damp(ks[j], b);
                                q * (1.0 / m[i] + 1.0 / (m[i] * m[o]).sqrt())
                            })
                            .sum()
                    };
                    let (rt, rs) = (part_row(true), part_row(false));
                    if self.row_cap_contact_only && !self.nodes[i].contact {
                        // The structure keeps its members: only its ties give.
                        let f = ((cap - rs).max(0.2 * rt) / rt.max(1e-30)).min(1.0);
                        for &j in adj[i].iter().filter(|&&j| tie(j)) {
                            ks[j] *= f;
                        }
                    } else if !self.nodes[i].contact && rt > 0.0 && rs < cap {
                        let f = ((cap - rs) / rt).min(1.0).powf(1.05);
                        for &j in adj[i].iter().filter(|&&j| tie(j)) {
                            ks[j] *= f;
                        }
                    } else {
                        let f = (cap / row).powf(1.05);
                        for &j in &adj[i] {
                            ks[j] *= f;
                        }
                    }
                }
            }
            if !over {
                break;
            }
        }
        // The real bound: the highest mode of M⁻¹K (power iteration), the
        // beams of the nodes that carry it softened until ω_max·h ≤ OMEGA_H.
        let dirs: Vec<Vec3> = self
            .beams
            .iter()
            .map(|b| (self.p(b.b) - self.p(b.a)).normalize_or_zero())
            .collect();
        let mut x: Vec<Vec3> = (0..n)
            .map(|i| {
                Vec3::new(
                    (i % 7) as f32 - 3.0,
                    (i % 5) as f32 - 2.0,
                    (i % 3) as f32 - 1.5,
                )
            })
            .collect();
        // Damping counts as stiffness 2c/h (symplectic Euler with damping is
        // stable while h²ω² + 2hγ < 4); the loader's latch couplers (spec §7:
        // k = 0.16 μ/h², c = √(kμ), isotropic) are in the model too.
        let mut couplers: Vec<(usize, usize, f32)> = Vec::new();
        for i in 0..n {
            let tag = |k: &str| {
                self.nodes[i]
                    .opts
                    .iter()
                    .find(|(o, _)| o == k)
                    .and_then(|(_, v)| v.as_str().map(str::to_string))
            };
            if let Some(t) = tag("couplerTag") {
                if let Some(o) = (0..n).find(|&o| {
                    self.nodes[o]
                        .opts
                        .iter()
                        .any(|(k, v)| k == "tag" && v.as_str() == Some(&t))
                }) {
                    let mu = m[i] * m[o] / (m[i] + m[o]);
                    couplers.push((i, o, 0.96 * mu / (H * H)));
                }
            }
        }
        let mut omega_h = 0.0;
        let centred = self.centred_damping;
        for round in 0..2000 {
            let iters = if round == 0 { 400 } else { 40 };
            let mut lam = 0.0f32;
            for _ in 0..iters {
                let mut y = vec![Vec3::ZERO; n];
                for (j, b) in self.beams.iter().enumerate() {
                    let d = dirs[j];
                    let keff = ks[j]
                        + if centred {
                            0.0
                        } else {
                            2.0 * damp(ks[j], b) / H
                        };
                    let f = d * (keff * d.dot(x[b.a] - x[b.b]));
                    y[b.a] += f;
                    y[b.b] -= f;
                }
                for &(a, b, k) in &couplers {
                    let f = (x[a] - x[b]) * k;
                    y[a] += f;
                    y[b] -= f;
                }
                for i in 0..n {
                    y[i] /= m[i];
                }
                let ny = y.iter().map(|v| v.length_squared()).sum::<f32>().sqrt();
                let nx = x.iter().map(|v| v.length_squared()).sum::<f32>().sqrt();
                lam = ny / nx.max(1e-30);
                x = y.into_iter().map(|v| v / ny.max(1e-30)).collect();
            }
            omega_h = lam.sqrt() * H;
            let limit = if centred { OMEGA_H_CENTRED } else { OMEGA_H };
            if std::env::var("GEN3_MODE").is_ok()
                && (round < 3 || round % 50 == 0 || omega_h <= OMEGA_H)
            {
                let mut v: Vec<(f32, usize)> = (0..n).map(|i| (x[i].length(), i)).collect();
                v.sort_by(|a, b| b.0.total_cmp(&a.0));
                let top: Vec<String> = v
                    .iter()
                    .take(6)
                    .map(|&(a, i)| format!("{} {:.2} ({:.2} kg)", self.nodes[i].name, a, m[i]))
                    .collect();
                eprintln!(
                    "mode round {round}: omega_h {omega_h:.3}: {}",
                    top.join(", ")
                );
            }
            if omega_h <= limit {
                break;
            }
            // Soften the beams that carry the mode: those with at least a
            // quarter of the largest beam's strain energy in it (not every
            // beam of the nodes that move: a light node's mode then softens
            // only its own beams, not its heavy neighbours' members).
            let e: Vec<f32> = self
                .beams
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    let keff = ks[j]
                        + if centred {
                            0.0
                        } else {
                            2.0 * damp(ks[j], b) / H
                        };
                    keff * dirs[j].dot(x[b.a] - x[b.b]).powi(2)
                })
                .collect();
            let emax = e.iter().copied().fold(0.0, f32::max);
            // Where the mode's carriers include a shut line's edge ties or
            // stops, only those give (the panel keeps its census springs).
            let edge_first = (0..e.len()).any(|j| {
                e[j] >= 0.25 * emax && matches!(self.beams[j].role, Role::EdgeTie | Role::EdgeStop)
            });
            for j in 0..e.len() {
                let edge = matches!(self.beams[j].role, Role::EdgeTie | Role::EdgeStop);
                if e[j] >= 0.25 * emax && (edge || !edge_first) {
                    ks[j] *= 0.97;
                }
            }
        }
        self.omega_h = omega_h;
        for (j, b) in self.beams.iter_mut().enumerate() {
            let sp = b.role.spec(b.part);
            b.k = ks[j];
            let z = match (b.part, bz) {
                (Part::BumperF | Part::BumperR, Some(z)) => z,
                _ => ZETA,
            };
            b.c = 2.0 * z * (ks[j] * (m[b.a] * m[b.b] / (m[b.a] + m[b.b]))).sqrt();
            b.deform = if sp.eps.is_finite() {
                sp.eps * ks[j] * len[j].max(SPRING_LENGTH)
            } else {
                f32::INFINITY
            };
        }
    }

    /// Gershgorin row of every node as a multiple of sedan1's bound.
    pub fn rows(&self, hub_extra: &HashMap<usize, f32>) -> Vec<f32> {
        let n = self.nodes.len();
        let m: Vec<f32> = (0..n)
            .map(|i| self.nodes[i].mass + hub_extra.get(&i).copied().unwrap_or(0.0))
            .collect();
        let mut r = vec![0.0f32; n];
        for b in &self.beams {
            let q = H * H * b.k + 2.0 * H * b.c;
            r[b.a] += q * (1.0 / m[b.a] + 1.0 / (m[b.a] * m[b.b]).sqrt());
            r[b.b] += q * (1.0 / m[b.b] + 1.0 / (m[b.a] * m[b.b]).sqrt());
        }
        r.iter().map(|x| x / ROW_BOUND).collect()
    }
}
