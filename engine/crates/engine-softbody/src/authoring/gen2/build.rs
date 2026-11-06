//! The generator's builder: nodes, beams with roles, panels (skin triangles with
//! their sheet edges, cross braces and two-bay bends), contact-only nodes in the
//! pole lanes, and the three rules that set what the file carries:
//!
//! * **mass** ([`Car::assign_masses`]): each part's real mass over its structural
//!   nodes by their share of the part's members (kg/m × length, half to each end)
//!   and of its sheet (area × areal density, a third to each corner), the named
//!   masses (engine, occupants, fuel, …) on the structure that carries them, the
//!   contact-only nodes at a fixed light mass taken out of their part's budget;
//! * **stiffness** ([`Car::assign_springs`]): every beam as stiff as the fixed
//!   2 kHz step allows, by Gershgorin rows (`skeleton::G_LIMIT`, the rule the
//!   authored sedans use): Σⱼ qᵢⱼ (1/mᵢ + 1/√(mᵢmⱼ)) ≤ 4·1.2 with
//!   qᵢⱼ = h²kᵢⱼ + 2h·cᵢⱼ, shared between a node's beams by their role weights;
//! * **yield** ([`Role::spec`]): each role's deform force from its section, its
//!   strength from the part (hardware breaks; the structure bends).

use glam::Vec3;
use std::collections::{BTreeMap, HashMap};

/// The fixed explicit step (s).
pub const H: f32 = 0.0005;
/// Gershgorin row bound (`skeleton::G_LIMIT` × h²: 1.2 × 4).
pub const ROW_BOUND: f32 = 4.8;
/// Mass of a contact-only node (kg): the agreed pattern's light lane nodes.
pub const CONTACT_MASS: f32 = 0.15;
/// Largest spacing of contact points along a pole lane (m).
pub const LANE: f32 = 0.20;

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
}

/// What a beam is: sets its share of the bound, its yield and strength, its
/// mass per metre (the mass rule) and its damping time.
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
    /// Bolts: fenders, bumper fascias, bumper beams.
    Bolt,
    /// Compression-only bearing (door edges on pillars, lids on their seats).
    Bearing,
    /// The powertrain block (rigid).
    Engine,
    /// Engine mounts.
    Mount,
    /// Suspension links holding the hubs (raycast wheels' axle nodes).
    Susp,
}

#[derive(Clone, Copy, Debug)]
pub struct RoleSpec {
    /// Share of a node's bound (relative).
    pub weight: f32,
    /// Deform (yield) force, N.
    pub deform: f32,
    /// Strength (break force), N (∞: never).
    pub strength: f32,
    /// Mass per metre for the mass rule (kg/m), 0 for none.
    pub kg_per_m: f32,
    /// Damping as a time constant (c = τ·k, s).
    pub tau: f32,
    /// Compression-only.
    pub support: bool,
    /// Hard cap on k (N/m).
    pub k_max: f32,
}

impl Role {
    pub fn spec(self) -> RoleSpec {
        let inf = f32::INFINITY;
        let s = |weight, deform, strength, kg_per_m| RoleSpec {
            weight,
            deform,
            strength,
            kg_per_m,
            tau: 6.0e-5,
            support: false,
            k_max: inf,
        };
        match self {
            // Sections (DOT HS 811 666 teardown grades; members.rs): a chord's
            // yield is its share of the member's plastic collapse force.
            Role::Rail => s(1.0, 42e3, inf, 5.5),
            Role::CrushCan => s(0.8, 22e3, inf, 3.0),
            Role::BumperBeam => s(1.0, 30e3, inf, 4.0),
            Role::Apron => s(0.8, 16e3, inf, 3.0),
            Role::Firewall => s(0.7, 14e3, inf, 4.0),
            Role::Floor => s(0.6, 12e3, inf, 5.0),
            Role::Tunnel => s(0.8, 24e3, inf, 4.0),
            Role::Sill => s(1.0, 48e3, inf, 6.0),
            Role::PillarA => s(1.0, 38e3, inf, 4.0),
            Role::PillarB => s(1.0, 44e3, inf, 5.0),
            Role::PillarC => s(0.9, 26e3, inf, 3.5),
            Role::RoofRail => s(1.0, 30e3, inf, 3.5),
            Role::RoofBow => s(0.7, 12e3, inf, 2.0),
            Role::RearRail => s(1.0, 32e3, inf, 4.5),
            Role::Cross => s(0.6, 14e3, inf, 2.5),
            Role::Web => s(0.5, 10e3, inf, 1.5),
            Role::Sheet => s(0.5, 4.5e3, inf, 0.0),
            Role::Bend => s(0.3, 2.5e3, inf, 0.0),
            Role::DoorFrame => s(0.8, 9e3, inf, 0.0),
            Role::Intrusion => s(1.0, 24e3, inf, 0.0),
            Role::Tie => RoleSpec { tau: 3.0e-5, ..s(0.35, 3e3, inf, 0.0) },
            Role::Backing => RoleSpec { tau: 3.0e-5, ..s(0.6, 4e3, inf, 0.0) },
            Role::Hinge => s(0.8, inf, 18e3, 0.0),
            Role::Bolt => s(0.6, inf, 9e3, 0.0),
            Role::Bearing => RoleSpec { support: true, ..s(0.5, 15e3, inf, 0.0) },
            Role::Engine => s(1.0, inf, inf, 0.0),
            Role::Mount => s(0.8, inf, 45e3, 0.0),
            Role::Susp => s(1.0, 60e3, 95e3, 0.0),
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
}

#[derive(Clone, Debug)]
pub struct Tri {
    pub n: [usize; 3],
    pub part: Part,
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
        });
        let i = self.nodes.len() - 1;
        self.by_name.insert(name.to_string(), i);
        i
    }

    pub fn id(&self, name: &str) -> usize {
        *self.by_name.get(name).unwrap_or_else(|| panic!("no node {name}"))
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
            if role.spec().weight > self.beams[j].role.spec().weight && self.beams[j].break_group.is_none() {
                self.beams[j].role = role;
            }
            return j;
        }
        let sp = role.spec();
        self.beams.push(Beam {
            a,
            b,
            role,
            part,
            k: 0.0,
            c: 0.0,
            deform: sp.deform,
            strength: sp.strength,
            break_group: None,
        });
        let j = self.beams.len() - 1;
        self.pair.insert(key, j);
        j
    }

    pub fn has_beam(&self, a: usize, b: usize) -> bool {
        self.pair.contains_key(&(a.min(b), a.max(b)))
    }

    /// A hardware beam in a break group, with its own strength.
    pub fn hardware(&mut self, a: usize, b: usize, role: Role, part: Part, group: &str, strength: f32) -> usize {
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
    pub fn ladder(&mut self, a: &[usize], b: &[usize], chord: Role, rung: Role, web: Role, part: Part) {
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
        let out = if (cen - axis).length() < 0.05 { Vec3::Z } else { cen - axis };
        if n.dot(out) >= 0.0 {
            self.tris.push(Tri { n: [a, b, c], part });
        } else {
            self.tris.push(Tri { n: [a, c, b], part });
        }
    }

    /// A panel from rows of optional nodes (rows bottom → top or front → rear,
    /// columns in order): sheet edges along rows and columns, both diagonals of
    /// each full cell (sheet), collision triangles, and two-bay bends along rows
    /// and columns when `bends`.
    pub fn panel(&mut self, rows: &[Vec<Option<usize>>], part: Part, row_role: &[Role], col_role: Role, bends: bool) {
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
            let ids: Vec<usize> = rows.iter().filter_map(|r| r.get(c).copied().flatten()).collect();
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
    pub fn lanes(&mut self, part: Part, backing: &[usize], surf: &dyn Fn(Vec3) -> Vec3) {
        let tris: Vec<usize> = (0..self.tris.len()).filter(|&t| self.tris[t].part == part).collect();
        let mut edge_pts: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut serial = 0usize;
        let pname = part.name().trim_start_matches("gen2_").to_string();
        let mut new_tris = Vec::new();
        let mut keep = vec![true; self.tris.len()];
        for &t in &tris {
            let [a, b, c] = self.tris[t].n;
            let mut perim: Vec<usize> = Vec::new();
            let mut split = false;
            for (u, v) in [(a, b), (b, c), (c, a)] {
                perim.push(u);
                let key = (u.min(v), u.max(v));
                let (pu, pv) = (self.p(key.0), self.p(key.1));
                let run = (pv - pu).y.abs();
                let n = ((run / LANE).ceil() as usize).max(1);
                if n <= 1 {
                    continue;
                }
                split = true;
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
            if !split {
                continue;
            }
            keep[t] = false;
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
        self.tris = old.into_iter().zip(keep).filter(|(_, k)| *k).map(|(t, _)| t).collect();
        for t in new_tris {
            self.tri(t[0], t[1], t[2], part);
        }
        // Backing ties: every lane node to its nearest backing node.
        if !backing.is_empty() {
            let lane: Vec<usize> = (0..self.nodes.len()).filter(|&i| self.nodes[i].contact && self.nodes[i].part == part).collect();
            for i in lane {
                let p = self.p(i);
                let mut best: Vec<(f32, usize)> = backing.iter().map(|&b| (self.p(b).distance(p), b)).collect();
                best.sort_by(|x, y| x.0.total_cmp(&y.0));
                for &(_, b) in best.iter().take(1) {
                    self.beam(i, b, Role::Backing, part);
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
            let kgm = b.role.spec().kg_per_m;
            if kgm == 0.0 {
                continue;
            }
            let l = self.p(b.a).distance(self.p(b.b));
            for e in [b.a, b.b] {
                if self.nodes[e].part == b.part && !self.nodes[e].contact {
                    w[e] += 0.5 * kgm * l;
                }
            }
        }
        // Sheet: a third of each triangle's area to its structural corners
        // (a lane triangle's area to the structural nodes of its part nearest
        // its centre: the structural nodes stand for the sheet around them).
        let structural_of: HashMap<Part, Vec<usize>> = Part::ALL
            .iter()
            .map(|&p| (p, (0..n).filter(|&i| self.nodes[i].part == p && !self.nodes[i].contact).collect()))
            .collect();
        for t in &self.tris {
            let [a, b, c] = t.n;
            let area = 0.5 * (self.p(b) - self.p(a)).cross(self.p(c) - self.p(a)).length();
            let dens = 0.8 * 7.85 * 3.0; // 0.8 mm steel ≈ 6.3 kg/m², relative weight only
            for &e in &[a, b, c] {
                let e = if self.nodes[e].contact {
                    let p = self.p(e);
                    let list = &structural_of[&t.part];
                    *list.iter().min_by(|&&x, &&y| self.p(x).distance(p).total_cmp(&self.p(y).distance(p))).unwrap()
                } else {
                    e
                };
                if self.nodes[e].part == t.part {
                    w[e] += area * dens / 3.0;
                }
            }
        }
        for (&part, &kg) in &self.part_mass.clone() {
            let ids: Vec<usize> = (0..n).filter(|&i| self.nodes[i].part == part).collect();
            let contact = ids.iter().filter(|&&i| self.nodes[i].contact).count();
            let budget = kg - contact as f32 * CONTACT_MASS;
            assert!(budget > 0.0, "{part:?}: {kg} kg for {contact} contact nodes");
            let mut sw: f32 = ids.iter().filter(|&&i| !self.nodes[i].contact).map(|&i| w[i]).sum();
            let structural = ids.iter().filter(|&&i| !self.nodes[i].contact).count();
            if sw <= 0.0 {
                for &i in &ids {
                    w[i] = 1.0;
                }
                sw = structural as f32;
            }
            // A floor of a fifth of the mean share: no structural node is left
            // as light as a lane node.
            let floor = 0.2 * sw / structural as f32;
            let sw2: f32 = ids.iter().filter(|&&i| !self.nodes[i].contact).map(|&i| w[i].max(floor)).sum();
            for &i in &ids {
                self.nodes[i].mass = if self.nodes[i].contact { CONTACT_MASS } else { budget * w[i].max(floor) / sw2 };
            }
        }
        for m in self.named.clone() {
            let ids: Vec<usize> = (0..n)
                .filter(|&i| self.nodes[i].part == m.part && !self.nodes[i].contact && self.p(i).distance(m.at) <= m.radius)
                .collect();
            assert!(!ids.is_empty(), "named mass {} has no nodes within {} m", m.name, m.radius);
            let ws: Vec<f32> = ids.iter().map(|&i| 1.0 / (self.p(i).distance(m.at) + 0.15)).collect();
            let s: f32 = ws.iter().sum();
            for (&i, &wi) in ids.iter().zip(&ws) {
                self.nodes[i].mass += m.kg * wi / s;
            }
        }
        self.nodes.iter().map(|n| n.mass).sum()
    }

    // ---------------------------------------------------------------------
    // Stiffness at the bound
    // ---------------------------------------------------------------------

    /// Springs and damping: see the module docs. Hub nodes' extra wheel mass
    /// (`hub_extra`, kg per node) counts in the bound as the solver adds it.
    pub fn assign_springs(&mut self, hub_extra: &HashMap<usize, f32>) {
        let n = self.nodes.len();
        let m: Vec<f32> = (0..n).map(|i| self.nodes[i].mass + hub_extra.get(&i).copied().unwrap_or(0.0)).collect();
        let qf = |tau: f32| H * (H + 2.0 * tau);
        // coefficient per beam end: q per unit k times (1/m_i + 1/sqrt(m_i m_j))
        let coef = |i: usize, j: usize, tau: f32| qf(tau) * (1.0 / m[i] + 1.0 / (m[i] * m[j]).sqrt());
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (j, b) in self.beams.iter().enumerate() {
            adj[b.a].push(j);
            adj[b.b].push(j);
        }
        let weight = |b: &Beam| b.role.spec().weight;
        let mut s: Vec<f32> = (0..n)
            .map(|i| {
                let r: f32 = adj[i]
                    .iter()
                    .map(|&j| {
                        let b = &self.beams[j];
                        let o = if b.a == i { b.b } else { b.a };
                        weight(b) * coef(i, o, b.role.spec().tau)
                    })
                    .sum();
                if r > 0.0 { ROW_BOUND / r } else { 0.0 }
            })
            .collect();
        let k_of = |s: &[f32], b: &Beam| (weight(b) * s[b.a].min(s[b.b])).min(b.role.spec().k_max);
        for _ in 0..60 {
            let ks: Vec<f32> = self.beams.iter().map(|b| k_of(&s, b)).collect();
            let mut changed = false;
            for i in 0..n {
                let row: f32 = adj[i]
                    .iter()
                    .map(|&j| {
                        let b = &self.beams[j];
                        let o = if b.a == i { b.b } else { b.a };
                        ks[j] * coef(i, o, b.role.spec().tau)
                    })
                    .sum();
                if row > 0.0 && row < 0.995 * ROW_BOUND {
                    s[i] *= (ROW_BOUND / row).min(1.5);
                    changed = true;
                }
            }
            // Never past the bound: scale back any row that went over.
            let ks: Vec<f32> = self.beams.iter().map(|b| k_of(&s, b)).collect();
            for i in 0..n {
                let row: f32 = adj[i]
                    .iter()
                    .map(|&j| {
                        let b = &self.beams[j];
                        let o = if b.a == i { b.b } else { b.a };
                        ks[j] * coef(i, o, b.role.spec().tau)
                    })
                    .sum();
                if row > ROW_BOUND {
                    s[i] *= ROW_BOUND / row;
                }
            }
            if !changed {
                break;
            }
        }
        // Final pass: the exact rows within the bound.
        let mut ks: Vec<f32> = self.beams.iter().map(|b| k_of(&s, b)).collect();
        for _ in 0..20 {
            let mut over = false;
            for i in 0..n {
                let row: f32 = adj[i]
                    .iter()
                    .map(|&j| {
                        let b = &self.beams[j];
                        let o = if b.a == i { b.b } else { b.a };
                        ks[j] * coef(i, o, b.role.spec().tau)
                    })
                    .sum();
                if row > ROW_BOUND * 1.0001 {
                    over = true;
                    let f = ROW_BOUND / row;
                    for &j in &adj[i] {
                        ks[j] *= f;
                    }
                }
            }
            if !over {
                break;
            }
        }
        for (b, k) in self.beams.iter_mut().zip(ks) {
            b.k = k;
            b.c = b.role.spec().tau * k;
        }
    }

    /// Gershgorin row of every node (fraction of the bound).
    pub fn rows(&self, hub_extra: &HashMap<usize, f32>) -> Vec<f32> {
        let n = self.nodes.len();
        let m: Vec<f32> = (0..n).map(|i| self.nodes[i].mass + hub_extra.get(&i).copied().unwrap_or(0.0)).collect();
        let mut r = vec![0.0f32; n];
        for b in &self.beams {
            let q = H * H * b.k + 2.0 * H * b.c;
            r[b.a] += q * (1.0 / m[b.a] + 1.0 / (m[b.a] * m[b.b]).sqrt());
            r[b.b] += q * (1.0 / m[b.b] + 1.0 / (m[b.a] * m[b.b]).sqrt());
        }
        r.iter().map(|x| x / ROW_BOUND).collect()
    }
}
