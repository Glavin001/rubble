//! The node–beam skeleton under construction, and its `.truck` text.
//!
//! Builders add nodes (each in a named part), beams (each with a [`Material`]),
//! shocks, wheels and triangles. Beams are deduplicated (the first one between
//! two nodes wins). Node masses are solved at the end ([`Skeleton::solve_masses`]):
//! named masses stay where they were put, the structure's mass is spread by beam
//! length × material density and scaled to its target, and no node may be lighter
//! than its beams need for a stable 0.5 ms explicit step.

use super::material::Material;
use glam::Vec3;
use std::collections::HashMap;
use std::fmt::Write;

/// The explicit solver's step (s).
pub const STEP: f32 = 0.0005;
/// Largest Σk/m a node may carry (1/s²) on its own (the per-node bound
/// [`Skeleton::floors`] starts from).
pub const K_RATIO: f32 = 1.0e7;
/// Bound on the Gershgorin row estimate of h²ω² + 2hγ per node, as a rate
/// (symplectic Euler with damping is stable while h²ω² + 2hγ < 4). The row sum
/// overestimates a lattice's highest mode, so the bound is 1.2 × 4/h²: checked
/// against 10 s at rest and crashes at every resolution (1.5 × diverges, 1.0 ×
/// leaves the body a third as stiff). [`Skeleton::floors`].
pub const G_LIMIT: f32 = 1.2 * 4.0 / (STEP * STEP);
/// Largest Σc/m a node may carry (1/s): h·Σc/m ≤ 0.75.
pub const C_RATIO: f32 = 1500.0;
/// Length a tie's stiffness is set for (m): see [`Skeleton::tie`].
pub const TIE_LENGTH: f32 = 0.3;
/// Damping of a shock's bump and rebound stops (N·s/m).
pub const STOP_DAMP: f32 = 600.0;
/// The structure's mass leans this many times harder on primary nodes (trim,
/// sealer, wiring and the rest are lumped onto the frame, as real node–beam cars
/// do).
pub const PRIMARY_WEIGHT: f32 = 6.0;
/// Lightest node (kg).
pub const MIN_NODE_MASS: f32 = 0.05;
/// A beam shorter than [`MIN_YIELD_LENGTH`] breaks at this many times its yield
/// force (it cannot yield).
pub const SHORT_BREAK: f32 = 2.0;
/// Shortest beam a yield threshold may act on: the solver keeps every yielded
/// beam at least this long, so a shorter one is made elastic up to its break.
pub const MIN_YIELD_LENGTH: f32 = 0.1;

/// A ductile beam tears at this many times its ultimate force
/// ([`Skeleton::rupture`]): just past the hardening cap, so the hardened
/// beam carries its ultimate force and the step's overshoot does not tear it.
pub const RUPTURE_MARGIN: f32 = 1.05;

/// How far a collision triangle's outward ray looks for a triangle covering
/// it ([`Skeleton::exposed_triangles`], m): past the car's half width.
pub const AERO_REACH: f32 = 3.0;

/// A ductile beam's rupture ([`Skeleton::rupture`]).
#[derive(Clone, Copy, Debug)]
pub struct Rupture {
    /// deformLimitStress (N): the ultimate force.
    pub cap: f32,
    /// beamStrength (N).
    pub strength: f32,
    /// deformLimitExpansion (fraction of the rest length).
    pub expansion: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BeamType {
    Normal,
    /// Compression only; breaks past `factor` × its rest length of stretch.
    Support(f32),
    /// Tension only (a rope): hardware that holds a part on (a hinge, latch
    /// or bolt) but does not carry its push (the part bears on supports).
    Rope,
}

#[derive(Clone, Debug)]
pub struct SkNode {
    pub p: Vec3,
    pub part: u16,
    /// Structural mass weight (beam length × density), scaled at the end.
    pub weight: f32,
    /// Named mass on this node (kg): engine, seats, fuel, passengers.
    pub fixed: f32,
    /// Final mass (kg), after [`Skeleton::solve_masses`].
    pub mass: f32,
    pub contacter: bool,
    /// Ground and static-shape contact (all nodes by default).
    pub ground: bool,
    /// The primary structure (heavy frame nodes: rails, sills, pillars, roof
    /// rails, cross members, towers, bumper beams): its beams are fitted to
    /// the step on their own ([`Skeleton::fit_tiers`]) and the structure's mass
    /// leans on it ([`PRIMARY_WEIGHT`]).
    pub primary: bool,
    /// A pole catcher between a member's stations (frame tier, but never a
    /// joint: braces, mounts and ties pick stations).
    pub catcher: bool,
    /// Its mass is its part's own (a panel's sheet mass, set as `fixed`):
    /// no share of the structure's.
    pub sheet: bool,
    /// One tier ([`super::vehicle::VehicleDef::one_tier`]): the node's real
    /// local mass (kg), what its sheet, member sections, equipment and named
    /// masses weigh there. Its final mass is exactly this: nothing is moved
    /// to meet the step's bound ([`Skeleton::cap_to_bound`] softens beams
    /// instead). 0 in the two-tier build.
    pub local: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SkBeam {
    pub a: u32,
    pub b: u32,
    pub mat: Material,
    pub ty: BeamType,
    /// Detacher group (0: none): when one beam of a group breaks, all do.
    pub group: u32,
    /// Kept at its material's stiffness by [`Skeleton::fit_stiffness`] (parts
    /// whose own mass steadies them: suspension, engine).
    pub fixed: bool,
    /// A tie ([`Skeleton::tie`]): a joint (weld, gusset, mount, catcher brace,
    /// a member section's ring), not a span of the structure.
    pub tie: bool,
    /// A dashpot (no spring): see [`Skeleton::dashpot`].
    pub damper: bool,
    /// Part of the body's load-carrying structure: the body-in-white's shell and
    /// members' spans (not glass, panels, ties, mounts or the rigid blocks). Set
    /// by the build.
    pub structure: bool,
    /// Rest length over its spawn length (1: the spawn pose). Glass panes
    /// and their beads take their installation pose: unstressed in the car
    /// settled under its own weight (`.truck` beams' fifth field, Rubble's
    /// extension).
    pub rest_scale: f32,
    /// One tier: the stiffness the beam was built for ([`Material::k`]: N/m
    /// at [`super::material::REF_LENGTH`]) before [`Skeleton::cap_to_bound`]
    /// (0: not capped).
    pub target_k: f32,
    /// JBeam plastic limits (`rubble_jbeam_plastic`): its deformLimitStress
    /// as a multiple of its yield force (1: it never hardens) and its
    /// deformLimit (its rest length never shortens past 1 − this of its
    /// spawn rest length). None: the `.truck` yield rule.
    pub plastic_limits: Option<(f32, f32)>,
    /// Its yield force in tension (N) when it differs from its compression
    /// yield `mat.deform` (`rubble_tension_yield`): a sheet edge buckles in
    /// compression but carries its material's yield in tension. 0: the same
    /// both ways.
    pub tension_yield: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SkShock {
    pub a: u32,
    pub b: u32,
    pub spring: f32,
    pub damp: f32,
    /// Bounds as fractions of the rest length, and the bump-stop spring past them.
    pub short: f32,
    pub long: f32,
    pub stop: f32,
    /// Rest length over the spawn length (> 1: preloaded to carry the car at the
    /// spawn's ride height).
    pub precomp: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SkWheel {
    /// Axle nodes: inner and outer face of the tyre.
    pub a1: u32,
    pub a2: u32,
    pub radius: f32,
    pub width: f32,
    pub rays: u32,
    pub mass: f32,
    pub spring: f32,
    pub damp: f32,
    pub propulsed: bool,
    pub braked: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub name: String,
    pub nodes: Vec<SkNode>,
    pub beams: Vec<SkBeam>,
    index: HashMap<(u32, u32), usize>,
    pub shocks: Vec<SkShock>,
    /// Limit straps: shocks without a spring or damper, free within their
    /// bounds and on their stops past them (a suspension's articulation
    /// stops). Written after the shocks.
    pub straps: Vec<SkShock>,
    pub wheels: Vec<SkWheel>,
    /// Fine skin triangles (wound outward): the visual binds to these.
    pub skin: Vec<[u32; 3]>,
    /// Collision triangles (wound outward): a coarser closed skin.
    pub collision: Vec<[u32; 3]>,
    /// Per collision triangle (or none: all plain) whether it breaks (spec
    /// §18): with the detacher group given (0 none) and the beams of the
    /// edges in the mask (bit k: nodes k to k+1): the triangles across the
    /// shut lines.
    pub collision_breaks: Vec<Option<(u32, u8)>>,
    /// Contact from every node (spec §10.2): `rubble_full_contact`.
    pub full_contact: bool,
    /// Exact beam lengths (spec §33): `rubble_exact_lengths`.
    pub exact_lengths: bool,
    /// The BeamNG profile (spec §36): `rubble_beamng_profile` (exact
    /// lengths, no idle-triangle throttle, two-body vehicle and self
    /// contact).
    pub beamng_profile: bool,
    /// Ductile beams tear at their elongation at break
    /// ([`Skeleton::rupture`]): one tier.
    pub ductile: bool,
    /// The aero triangles' coefficients (`rubble_drag_coef` dragCoef,
    /// liftCoef): none, the profile's default.
    pub drag_coef: Option<(f32, f32)>,
    /// Per-triangle aerodynamics (`rubble_aero_tri`): the exposed collision
    /// triangles ([`Skeleton::exposed_triangles`]) at these (dragCoef,
    /// liftCoef), every other one (covered, inner) none.
    pub aero_exposed: Option<(f32, f32)>,
    /// The nodes are the vehicle at rest on flat ground at this height
    /// (`rubble_rest_pose`): it spawns standing, its beams' rest lengths
    /// (`rest_scale`) the unloaded design's.
    pub rest_ground: Option<f32>,
    /// Its tyres at rest (`rubble_tyre_pose`): per wheel, its generated
    /// nodes' positions (empty: built undeflected round the axles).
    pub tyre_pose: Vec<Vec<Vec3>>,
    /// With a tyre pose, each wheel's axle nodes' rest (design) positions,
    /// [`SkWheel::a1`]'s first.
    pub tyre_rest_axle: Vec<[Vec3; 2]>,
    /// In a rest pose, the (collision triangle, node) pairs that rest in
    /// contact: they collide though they start in range
    /// (`rubble_spawn_contacts`).
    pub spawn_contacts: Vec<([u32; 3], u32)>,
    pub parts: Vec<String>,
    /// Reference nodes: centre, back, left (`cameras`).
    pub cameras: [u32; 3],
}

impl Skeleton {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            parts: vec!["body".into()],
            ..Default::default()
        }
    }

    /// Moves the whole car by `f` (a rigid motion): its nodes and its tyres'
    /// rest pose. A motion that does not keep the ground under its tyres (a
    /// turn off the vertical) must also clear `rest_ground`.
    pub fn map_positions(&mut self, f: impl Fn(Vec3) -> Vec3) {
        for n in &mut self.nodes {
            n.p = f(n.p);
        }
        for t in &mut self.tyre_pose {
            for p in t {
                *p = f(*p);
            }
        }
        for [a, b] in &mut self.tyre_rest_axle {
            *a = f(*a);
            *b = f(*b);
        }
    }

    /// The part's id, created on first use.
    pub fn part(&mut self, name: &str) -> u16 {
        if let Some(i) = self.parts.iter().position(|p| p == name) {
            return i as u16;
        }
        self.parts.push(name.into());
        (self.parts.len() - 1) as u16
    }

    pub fn node(&mut self, p: Vec3, part: u16) -> u32 {
        self.nodes.push(SkNode {
            p,
            part,
            weight: 0.0,
            fixed: 0.0,
            mass: 0.0,
            contacter: false,
            ground: true,
            primary: false,
            catcher: false,
            sheet: false,
            local: 0.0,
        });
        (self.nodes.len() - 1) as u32
    }

    pub fn pos(&self, i: u32) -> Vec3 {
        self.nodes[i as usize].p
    }

    /// The index of the beam between `a` and `b`, if any.
    pub fn beam_index(&self, a: u32, b: u32) -> Option<usize> {
        self.index.get(&(a.min(b), a.max(b))).copied()
    }

    pub fn has_beam(&self, a: u32, b: u32) -> bool {
        self.index.contains_key(&(a.min(b), a.max(b)))
    }

    /// A ductile beam's rupture (one tier; `RUPTURE=0` leaves it out, a
    /// measurement): a deformable plain beam of a grade with an elongation
    /// at break ([`Material::ductile`]) hardens at most to its ultimate force
    /// (Rm/Re × its yield, tension's when that is higher) and tears just past
    /// it (×[`RUPTURE_MARGIN`]); its plastic stretch is capped
    /// (`deformLimitExpansion`) so that its strain when it tears, plastic
    /// plus the elastic stretch at that force, is the elongation at break:
    /// dₑ = A − S/(k·L), at least 0 (a beam whose elastic stretch at S is
    /// already past A, at the bound's stiffness, tears at S).
    pub fn rupture(&self, b: &SkBeam) -> Option<Rupture> {
        if !self.ductile || b.ty != BeamType::Normal || b.damper || b.mat.k <= 0.0 {
            return None;
        }
        let l = self.pos(b.a).distance(self.pos(b.b)) * b.rest_scale;
        if l < MIN_YIELD_LENGTH {
            return None;
        }
        let t = b.mat.deform.max(b.tension_yield);
        let kl = b.mat.k_at(l) * l;
        if b.mat.elong <= 0.0 || b.mat.uts <= 0.0 {
            // Not graded metal (hardware, mounts, bolts, bearing pads): its
            // strength as built. (Capping its stretch at RoR's tear stretch
            // (S − T)/k with a strength just past its hardened force tore
            // 38 rear-third beams in the 56 km/h wall, in compression: a
            // step's overshoot past a near cap; measured, not adopted.)
            return None;
        }
        let cap = b.mat.uts * t;
        let margin = std::env::var("RUPTURE_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(RUPTURE_MARGIN);
        let strength = margin * cap;
        let expansion = (b.mat.elong - strength / kl).max(0.0);
        Some(Rupture {
            cap,
            strength,
            expansion,
        })
    }

    /// The collision triangles the air reaches: a ray from a triangle's
    /// centre along its outward normal meets no other collision triangle
    /// within [`AERO_REACH`]. Covered ones (a door's jamb behind its skin, a
    /// panel's inner face, the floor's top inside the cabin) are not.
    pub fn exposed_triangles(&self) -> Vec<usize> {
        let tri: Vec<[Vec3; 3]> = self
            .collision
            .iter()
            .map(|t| t.map(|i| self.pos(i)))
            .collect();
        let hit = |o: Vec3, d: Vec3, t: &[Vec3; 3]| -> bool {
            // Möller–Trumbore, either face.
            let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
            let p = d.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-9 {
                return false;
            }
            let inv = 1.0 / det;
            let sv = o - t[0];
            let u = sv.dot(p) * inv;
            if !(0.0..=1.0).contains(&u) {
                return false;
            }
            let q = sv.cross(e1);
            let v = d.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                return false;
            }
            let dist = e2.dot(q) * inv;
            dist > 1e-4 && dist < AERO_REACH
        };
        (0..tri.len())
            .filter(|&k| {
                let t = &tri[k];
                let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
                if n == Vec3::ZERO {
                    return false;
                }
                let o = (t[0] + t[1] + t[2]) / 3.0 + n * 0.005;
                !(0..tri.len()).any(|j| {
                    j != k && {
                        let shares = self.collision[j]
                            .iter()
                            .filter(|i| self.collision[k].contains(i))
                            .count();
                        shares < 2 && hit(o, n, &tri[j])
                    }
                })
            })
            .collect()
    }

    /// A beam between `a` and `b` unless they are already joined; returns whether
    /// it was added.
    pub fn beam(&mut self, a: u32, b: u32, mat: Material) -> bool {
        self.beam_with(a, b, mat, BeamType::Normal, 0)
    }

    pub fn beam_with(&mut self, a: u32, b: u32, mat: Material, ty: BeamType, group: u32) -> bool {
        if a == b {
            return false;
        }
        let key = (a.min(b), a.max(b));
        if self.index.contains_key(&key) {
            return false;
        }
        if self.pos(a).distance(self.pos(b)) < 0.01 {
            return false;
        }
        self.index.insert(key, self.beams.len());
        self.beams.push(SkBeam {
            a,
            b,
            mat,
            ty,
            group,
            fixed: false,
            tie: false,
            damper: false,
            structure: false,
            rest_scale: 1.0,
            target_k: 0.0,
            plastic_limits: None,
            tension_yield: 0.0,
        });
        true
    }

    /// A tie (spot weld, bolt, mount, gusset, catcher brace): as stiff as a
    /// [`TIE_LENGTH`] beam of `mat` however short it is. What a tie joins sits
    /// close by, and a short beam's stiffness would cost its nodes mass for no
    /// gain (the step's limit is on k/m). Its yield and break forces are `mat`'s.
    pub fn tie(&mut self, a: u32, b: u32, mat: Material) -> bool {
        let l = self.pos(a).distance(self.pos(b));
        let s = l.max(super::material::REF_LENGTH) / TIE_LENGTH.max(l);
        let added = self.beam(a, b, mat.stiffer(s));
        if added {
            self.beams.last_mut().unwrap().tie = true;
        }
        added
    }

    /// A dashpot between `a` and `b`: damping `c` (N·s/m) and no spring, torn
    /// off past `strength` N. Kept as is by [`Self::fit_stiffness`].
    pub fn dashpot(&mut self, a: u32, b: u32, c: f32, strength: f32) -> bool {
        let mat = Material {
            k: 0.0,
            damp: c,
            deform: strength,
            strength,
            plastic: 0.0,
            density: 0.0,
            elong: 0.0,
            uts: 0.0,
        };
        let added = self.beam(a, b, mat);
        if added {
            let bm = self.beams.last_mut().unwrap();
            bm.damper = true;
            bm.fixed = true;
        }
        added
    }

    /// Replace the material (and type and group) of the beam between `a` and `b`,
    /// adding it if absent.
    pub fn set_beam(&mut self, a: u32, b: u32, mat: Material, ty: BeamType, group: u32) {
        let key = (a.min(b), a.max(b));
        if let Some(&i) = self.index.get(&key) {
            let bm = &mut self.beams[i];
            bm.mat = mat;
            bm.ty = ty;
            bm.group = group;
        } else {
            self.beam_with(a, b, mat, ty, group);
        }
    }

    /// A beam kept at full stiffness by [`Self::fit_stiffness`].
    pub fn fixed_beam(&mut self, a: u32, b: u32, mat: Material) {
        if self.beam(a, b, mat) {
            self.beams.last_mut().unwrap().fixed = true;
        }
    }

    /// Append `other` (another vehicle) with its nodes moved by `at`: one
    /// skeleton of both, so the solver runs them together and their
    /// contacters meet each other's collision triangles. Its parts are
    /// renamed `<prefix><part>`.
    pub fn append(&mut self, other: &Skeleton, prefix: &str, at: &dyn Fn(Vec3) -> Vec3) {
        let base = self.nodes.len() as u32;
        let part_of: Vec<u16> = other
            .parts
            .iter()
            .map(|p| self.part(&format!("{prefix}{p}")))
            .collect();
        for n in &other.nodes {
            let mut m = n.clone();
            m.p = at(n.p);
            m.part = part_of[n.part as usize];
            self.nodes.push(m);
        }
        for b in &other.beams {
            let mut c = *b;
            c.a += base;
            c.b += base;
            // (Its detacher groups numbered past ours.)
            if c.group != 0 {
                c.group += 10_000;
            }
            self.index
                .insert((c.a.min(c.b), c.a.max(c.b)), self.beams.len());
            self.beams.push(c);
        }
        let shift = |mut x: SkShock| {
            x.a += base;
            x.b += base;
            x
        };
        self.shocks.extend(other.shocks.iter().copied().map(shift));
        self.straps.extend(other.straps.iter().copied().map(shift));
        for w in &other.wheels {
            let mut v = *w;
            v.a1 += base;
            v.a2 += base;
            self.wheels.push(v);
        }
        let tri = |t: &[u32; 3]| [t[0] + base, t[1] + base, t[2] + base];
        self.skin.extend(other.skin.iter().map(tri));
        if !(self.collision_breaks.is_empty() && other.collision_breaks.is_empty()) {
            self.collision_breaks.resize(self.collision.len(), None);
            let mut o = other.collision_breaks.clone();
            o.resize(other.collision.len(), None);
            self.collision_breaks.extend(o);
        }
        self.collision.extend(other.collision.iter().map(tri));
    }

    /// Remove the beam between `a` and `b` (if any).
    pub fn remove_beam(&mut self, a: u32, b: u32) {
        let key = (a.min(b), a.max(b));
        if let Some(i) = self.index.remove(&key) {
            self.beams.swap_remove(i);
            if i < self.beams.len() {
                let m = self.beams[i];
                self.index.insert((m.a.min(m.b), m.a.max(m.b)), i);
            }
        }
    }

    /// Triangle (a, b, c) wound so its normal faces away from `inside`.
    pub fn oriented(&self, a: u32, b: u32, c: u32, inside: Vec3) -> [u32; 3] {
        let (pa, pb, pc) = (self.pos(a), self.pos(b), self.pos(c));
        let n = (pb - pa).cross(pc - pa);
        if n.dot((pa + pb + pc) / 3.0 - inside) >= 0.0 {
            [a, b, c]
        } else {
            [a, c, b]
        }
    }

    /// The `count` nodes of `pool` nearest to `p`.
    pub fn nearest(&self, p: Vec3, pool: &[u32], count: usize) -> Vec<u32> {
        let mut v: Vec<(f32, u32)> = pool
            .iter()
            .map(|&i| (self.pos(i).distance_squared(p), i))
            .collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        v.into_iter().take(count).map(|x| x.1).collect()
    }

    /// Spread `kg` over `nodes` evenly (a named mass).
    pub fn add_mass(&mut self, nodes: &[u32], kg: f32) {
        if nodes.is_empty() {
            return;
        }
        let each = kg / nodes.len() as f32;
        for &i in nodes {
            self.nodes[i as usize].fixed += each;
        }
    }

    /// Spread `kg` over `nodes` in proportion to their beams' stiffness (a named
    /// mass sits on the structure that carries it, which also steadies it).
    pub fn add_mass_by_stiffness(&mut self, nodes: &[u32], kg: f32) {
        let (k, _) = self.node_rates();
        let total: f32 = nodes.iter().map(|&i| k[i as usize].max(1.0)).sum();
        for &i in nodes {
            self.nodes[i as usize].fixed += kg * k[i as usize].max(1.0) / total;
        }
    }

    /// Spread `kg` over `nodes` to fill their unmet stability floors first
    /// (`need` per node, less what it already carries), the rest in
    /// proportion to their beams' stiffness.
    pub fn add_mass_by_need(&mut self, nodes: &[u32], kg: f32, need: &[f32]) {
        let gap: Vec<f32> = nodes
            .iter()
            .map(|&i| (need[i as usize] - self.nodes[i as usize].fixed).max(0.0))
            .collect();
        let total: f32 = gap.iter().sum();
        let fill = kg.min(total);
        if total > 0.0 {
            for (&i, &g) in nodes.iter().zip(&gap) {
                self.nodes[i as usize].fixed += fill * g / total;
            }
        }
        if kg > fill {
            self.add_mass_by_stiffness(nodes, kg - fill);
        }
    }

    /// Per node the sum of its beams' stiffness and damping (with shocks and the
    /// tyre spokes on axle nodes).
    pub fn node_rates(&self) -> (Vec<f32>, Vec<f32>) {
        let n = self.nodes.len();
        let mut k = vec![0.0f32; n];
        let mut c = vec![0.0f32; n];
        for b in &self.beams {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let kk = b.mat.k_at(l);
            for i in [b.a, b.b] {
                k[i as usize] += kk;
                c[i as usize] += b.mat.damp;
            }
        }
        for s in &self.shocks {
            for i in [s.a, s.b] {
                k[i as usize] += s.spring;
                c[i as usize] += s.damp;
            }
        }
        for w in &self.wheels {
            for i in [w.a1, w.a2] {
                k[i as usize] += 2.0 * w.rays as f32 * w.spring;
                c[i as usize] += 2.0 * w.rays as f32 * w.damp;
            }
        }
        (k, c)
    }

    /// The lightest stable mass of every node with every beam's stiffness and
    /// damping × `s`: its Gershgorin row within [`G_LIMIT`] given its neighbours'
    /// masses (iterated to a fixed point from below), Σc/m within [`C_RATIO`],
    /// never under its named mass or [`MIN_NODE_MASS`].
    pub fn floors(&self, s: f32) -> Vec<f32> {
        self.floors_tiers(s, s)
    }

    /// Whether a beam belongs to the primary tier (both ends primary).
    pub fn beam_primary(&self, b: &SkBeam) -> bool {
        self.nodes[b.a as usize].primary && self.nodes[b.b as usize].primary
    }

    /// [`Self::floors`] with the primary tier's beams × `sp`, the rest × `ss`.
    pub fn floors_tiers(&self, sp: f32, ss: f32) -> Vec<f32> {
        let n = self.nodes.len();
        // Per node: (neighbour, stiffness, damping).
        let mut adj: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); n];
        for b in &self.beams {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let f = if b.fixed {
                1.0
            } else if self.beam_primary(b) {
                sp
            } else {
                ss
            };
            let kk = f * b.mat.k_at(l);
            let cc = f * b.mat.damp;
            adj[b.a as usize].push((b.b, kk, cc));
            adj[b.b as usize].push((b.a, kk, cc));
        }
        for sh in &self.shocks {
            adj[sh.a as usize].push((sh.b, sh.spring, sh.damp));
            adj[sh.b as usize].push((sh.a, sh.spring, sh.damp));
        }
        // Tyre spokes: to ring nodes of the wheel's own mass share (a, b terms).
        let mut tyre: Vec<(f32, f32)> = vec![(0.0, 0.0); n];
        let h = STEP;
        for w in &self.wheels {
            let mt = w.mass / (2.0 * w.rays as f32);
            for i in [w.a1, w.a2] {
                let k = 2.0 * w.rays as f32 * w.spring;
                let c = 2.0 * w.rays as f32 * w.damp;
                let q = h * h * k + 2.0 * h * c;
                tyre[i as usize].0 += q;
                tyre[i as usize].1 += q / mt.sqrt();
            }
        }
        // Symplectic Euler with damping is stable while h²ω² + 2hγ < 4; bound
        // both by Gershgorin rows: Σⱼ qᵢⱼ (1/mᵢ + 1/√(mᵢmⱼ)) ≤ 4·safety with
        // qᵢⱼ = h²kᵢⱼ + 2h·cᵢⱼ.
        let limit = 4.0 * G_LIMIT * h * h / 4.0;
        let q = |k: f32, c: f32| h * h * k + 2.0 * h * c;
        let mut m: Vec<f32> = (0..n)
            .map(|i| {
                let a: f32 = adj[i].iter().map(|x| q(x.1, x.2)).sum::<f32>() + tyre[i].0;
                (a / limit).max(MIN_NODE_MASS).max(self.nodes[i].fixed)
            })
            .collect();
        for _ in 0..40 {
            let mut changed = false;
            for i in 0..n {
                let a: f32 = adj[i].iter().map(|x| q(x.1, x.2)).sum::<f32>() + tyre[i].0;
                if a == 0.0 {
                    continue;
                }
                let b: f32 = adj[i]
                    .iter()
                    .map(|&(j, k, c)| q(k, c) / m[j as usize].sqrt())
                    .sum::<f32>()
                    + tyre[i].1;
                // a·x² + b·x = limit, x = 1/√m.
                let x = (-b + (b * b + 4.0 * a * limit).sqrt()) / (2.0 * a);
                let need = 1.0 / (x * x);
                if need > m[i] * 1.0001 {
                    m[i] = need;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        m
    }

    /// The largest Gershgorin bound over the nodes, as ω·h (stable below 2).
    pub fn worst_omega_h(&self) -> f32 {
        let n = self.nodes.len();
        let mut g = vec![0.0f32; n];
        let mass = |i: u32| self.nodes[i as usize].mass;
        for b in &self.beams {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let k = b.mat.k_at(l);
            let cross = k / (mass(b.a) * mass(b.b)).sqrt();
            g[b.a as usize] += k / mass(b.a) + cross;
            g[b.b as usize] += k / mass(b.b) + cross;
        }
        for sh in &self.shocks {
            let cross = sh.spring / (mass(sh.a) * mass(sh.b)).sqrt();
            g[sh.a as usize] += sh.spring / mass(sh.a) + cross;
            g[sh.b as usize] += sh.spring / mass(sh.b) + cross;
        }
        for w in &self.wheels {
            let mt = w.mass / (2.0 * w.rays as f32);
            for i in [w.a1, w.a2] {
                let k = 2.0 * w.rays as f32 * w.spring;
                g[i as usize] += k / mass(i) + k / (mass(i) * mt).sqrt();
            }
        }
        g.iter().fold(0.0f32, |a, &x| a.max(x)).sqrt() * STEP
    }

    /// Scale every beam's stiffness and damping (not the shocks' or tyres') so the
    /// nodes' stability floors take at most `share` of the mass budget (`fixed` +
    /// `structure` kg). Returns the scale (1 when they already fit).
    pub fn fit_stiffness(&mut self, structure: f32, share: f32) -> f32 {
        let fixed: f32 = self.nodes.iter().map(|n| n.fixed).sum();
        let want = share * (fixed + structure);
        let total = |sk: &Skeleton, s: f32| -> f32 { sk.floors(s).iter().sum() };
        if total(self, 1.0) <= want {
            return 1.0;
        }
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if total(self, mid) > want {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        // Thresholds with the stiffness: every grade keeps its yield strain.
        for b in self.beams.iter_mut().filter(|b| !b.fixed) {
            b.mat.k *= lo;
            b.mat.damp *= lo;
            if b.mat.deform < 1.0e8 {
                b.mat.deform *= lo;
                b.mat.strength *= lo;
            }
        }
        lo
    }

    /// [`Self::fit_stiffness`] per tier: the primary structure kept as defined
    /// while the rest (skin, panels, catchers, ties to them) is scaled down to
    /// fit; only if that is not enough are both scaled together. Returns
    /// (primary, rest).
    pub fn fit_tiers(&mut self, structure: f32, share: f32) -> (f32, f32) {
        let fixed: f32 = self.nodes.iter().map(|n| n.fixed).sum();
        let want = share * (fixed + structure);
        let total =
            |sk: &Skeleton, sp: f32, ss: f32| -> f32 { sk.floors_tiers(sp, ss).iter().sum() };
        let search = |f: &dyn Fn(f32) -> f32| -> f32 {
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if f(mid) > want {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            lo
        };
        let (sp, ss) = if total(self, 1.0, 1.0) <= want {
            (1.0, 1.0)
        } else if total(self, 1.0, 0.05) <= want {
            (1.0, search(&|x| total(self, 1.0, x)))
        } else {
            let x = search(&|x| total(self, x, 0.05 * x));
            (x, 0.05 * x)
        };
        // The frame keeps its grades' yield strains (thresholds with the
        // stiffness); the skin keeps its yield and break forces (what holds a
        // panel on and where it dents are forces, not strains).
        let prim: Vec<bool> = self.beams.iter().map(|b| self.beam_primary(b)).collect();
        for (b, &p) in self.beams.iter_mut().zip(&prim) {
            if b.fixed {
                continue;
            }
            let f = if p { sp } else { ss };
            b.mat.k *= f;
            b.mat.damp *= f;
            if p && b.mat.deform < 1.0e8 {
                b.mat.deform *= f;
                b.mat.strength *= f;
            }
        }
        (sp, ss)
    }

    /// Gershgorin rows of every node at the current masses and stiffnesses,
    /// as a fraction of the step's bound (stable while every row is ≤ 1):
    /// Σⱼ qᵢⱼ (1/mᵢ + 1/√(mᵢmⱼ)) / (h²·G_LIMIT), qᵢⱼ = h²kᵢⱼ + 2h·cᵢⱼ, with
    /// the shocks, dashpots and tyre spokes.
    pub fn bound_rows(&self) -> Vec<f32> {
        let h = STEP;
        let limit = G_LIMIT * h * h;
        let n = self.nodes.len();
        let m = |i: u32| self.nodes[i as usize].mass.max(1e-6);
        let mut r = vec![0.0f32; n];
        let mut add = |a: u32, b: u32, k: f32, c: f32| {
            let q = h * h * k + 2.0 * h * c;
            let cross = 1.0 / (m(a) * m(b)).sqrt();
            r[a as usize] += q * (1.0 / m(a) + cross);
            r[b as usize] += q * (1.0 / m(b) + cross);
        };
        for b in &self.beams {
            let l = self.pos(b.a).distance(self.pos(b.b));
            add(b.a, b.b, b.mat.k_at(l), b.mat.damp);
        }
        for s in &self.shocks {
            add(s.a, s.b, s.spring, s.damp);
        }
        for w in &self.wheels {
            let mt = w.mass / (2.0 * w.rays as f32);
            for i in [w.a1, w.a2] {
                let k = 2.0 * w.rays as f32 * w.spring;
                let c = 2.0 * w.rays as f32 * w.damp;
                let q = h * h * k + 2.0 * h * c;
                r[i as usize] += q * (1.0 / m(i) + 1.0 / (m(i) * mt).sqrt());
            }
        }
        r.iter().map(|x| x / limit).collect()
    }

    /// One tier: the node masses are fixed (each node's real local mass,
    /// [`SkNode::local`]) and every beam keeps the stiffness it was built
    /// for unless the explicit step's bound at one of its nodes requires
    /// less. The bound is the Gershgorin row of [`Self::floors`]
    /// (h²ω² + 2hγ within [`G_LIMIT`]): numerics BeamNG's content obeys
    /// too (about 1.1 MN/m on 1 kg nodes). The cut is max-min fair: at a
    /// node that would exceed its bound, its stiffest beams come down to a
    /// common level, beams already under that level keep theirs, and a beam
    /// is softened only where one of its nodes is saturated. Damping keeps
    /// its ratio (c ∝ √k). Yield and break forces are unchanged: strengths
    /// stay what they were built as. Shocks and tyres are kept (their real
    /// rates); dashpots too. Records each beam's target in
    /// [`SkBeam::target_k`]. Returns an error naming a node whose kept
    /// springs alone exceed its bound.
    pub fn cap_to_bound(&mut self) -> Result<(), String> {
        let h = STEP;
        let limit = G_LIMIT * h * h;
        let n = self.nodes.len();
        let m: Vec<f32> = self.nodes.iter().map(|x| x.mass).collect();
        if let Some(i) = m.iter().position(|&x| !(x > 0.0)) {
            return Err(format!("node {i} at {:?} has no mass", self.nodes[i].p));
        }
        // Per beam: target k at its length, damping at the target.
        let nb = self.beams.len();
        let mut kt = vec![0.0f32; nb];
        let mut ct = vec![0.0f32; nb];
        let mut capped = vec![false; nb];
        for (j, b) in self.beams.iter().enumerate() {
            let l = self.pos(b.a).distance(self.pos(b.b));
            kt[j] = b.mat.k_at(l);
            ct[j] = b.mat.damp;
            capped[j] = !b.damper && kt[j] > 0.0;
        }
        // Kept contributions (shocks, tyres, dashpots) per node.
        let w = |a: usize, b: usize| 1.0 / m[a] + 1.0 / (m[a] * m[b]).sqrt();
        let mut kept = vec![0.0f32; n];
        for (j, b) in self.beams.iter().enumerate() {
            if !capped[j] {
                let q = h * h * kt[j] + 2.0 * h * ct[j];
                kept[b.a as usize] += q * w(b.a as usize, b.b as usize);
                kept[b.b as usize] += q * w(b.b as usize, b.a as usize);
            }
        }
        for s in &self.shocks {
            let q = h * h * s.spring + 2.0 * h * s.damp;
            kept[s.a as usize] += q * w(s.a as usize, s.b as usize);
            kept[s.b as usize] += q * w(s.b as usize, s.a as usize);
        }
        // Tyres: each axle node's spokes as one spring to a ring node of the
        // tyre's mass share, capped like a beam (the spoke rate is the tyre
        // model's, not a measured rate).
        let mut tyres: Vec<Vec<(usize, f32, f32, f32)>> = vec![Vec::new(); n];
        for (t, wh) in self.wheels.iter().enumerate() {
            let mt = wh.mass / (2.0 * wh.rays as f32);
            for i in [wh.a1, wh.a2] {
                let iu = i as usize;
                let k = 2.0 * wh.rays as f32 * wh.spring;
                let c = 2.0 * wh.rays as f32 * wh.damp;
                tyres[iu].push((t, k, c, 1.0 / m[iu] + 1.0 / (m[iu] * mt).sqrt()));
            }
        }
        if let Some(i) = (0..n).find(|&i| kept[i] > limit) {
            return Err(format!(
                "node {i} at {:?} ({:.3} kg): its shocks, tyres and dashpots alone exceed the step's bound ({:.2}×)",
                self.nodes[i].p,
                m[i],
                kept[i] / limit
            ));
        }
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for (j, b) in self.beams.iter().enumerate() {
            if capped[j] {
                adj[b.a as usize].push((j, b.b as usize));
                adj[b.b as usize].push((j, b.a as usize));
            }
        }
        let q = |j: usize, k: f32| h * h * k + 2.0 * h * ct[j] * (k / kt[j]).sqrt();
        // Each node's level: the common k its beams may take (beams under it
        // keep their own), given each beam's cap from its other end.
        let mut level = vec![f32::INFINITY; n];
        let qt = |k: f32, c: f32, kk: f32| h * h * kk + 2.0 * h * c * (kk / k).sqrt();
        let tyre_row = |i: usize, lam: f32| -> f32 {
            tyres[i]
                .iter()
                .map(|&(_, k, c, wf)| qt(k, c, k.min(lam)) * wf)
                .sum::<f32>()
        };
        let row = |i: usize, lam: f32, cap: &dyn Fn(usize) -> f32| -> f32 {
            kept[i]
                + tyre_row(i, lam)
                + adj[i]
                    .iter()
                    .map(|&(j, o)| q(j, kt[j].min(lam).min(cap(j))) * w(i, o))
                    .sum::<f32>()
        };
        for _ in 0..60 {
            let prev = level.clone();
            let mut moved = 0.0f32;
            for i in 0..n {
                if adj[i].is_empty() && tyres[i].is_empty() {
                    continue;
                }
                let cap = |j: usize| {
                    let b = &self.beams[j];
                    let o = if b.a as usize == i { b.b } else { b.a } as usize;
                    prev[o]
                };
                let lam = if row(i, f32::INFINITY, &cap) <= limit {
                    f32::INFINITY
                } else {
                    let (mut lo, mut hi) = (
                        0.0f32,
                        adj[i]
                            .iter()
                            .map(|x| kt[x.0])
                            .chain(tyres[i].iter().map(|x| x.1))
                            .fold(0.0, f32::max),
                    );
                    for _ in 0..50 {
                        let mid = 0.5 * (lo + hi);
                        if row(i, mid, &cap) > limit {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    lo
                };
                if lam.is_finite() != prev[i].is_finite()
                    || (lam.is_finite() && (lam - prev[i]).abs() > 1e-4 * lam)
                {
                    moved += 1.0;
                }
                level[i] = lam;
            }
            if moved == 0.0 {
                break;
            }
        }
        let mut k: Vec<f32> = (0..nb)
            .map(|j| {
                if !capped[j] {
                    return kt[j];
                }
                let b = &self.beams[j];
                kt[j].min(level[b.a as usize]).min(level[b.b as usize])
            })
            .collect();
        // Each tyre at the lower of its axle nodes' levels.
        let mut tyre_f = vec![1.0f32; self.wheels.len()];
        for i in 0..n {
            for &(t, k, _, _) in &tyres[i] {
                tyre_f[t] = tyre_f[t].min(k.min(level[i]) / k);
            }
        }
        let tyre_now = |i: usize| -> f32 {
            tyres[i]
                .iter()
                .map(|&(t, k, c, wf)| qt(k, c, k * tyre_f[t]) * wf)
                .sum::<f32>()
        };
        let kept: Vec<f32> = (0..n).map(|i| kept[i] + tyre_now(i)).collect();
        // Feasibility: any node still over its bound (the levels' fixed point
        // not reached exactly) takes its beams down by a common factor.
        for _ in 0..50 {
            let mut over = false;
            for i in 0..n {
                let r = kept[i]
                    + adj[i]
                        .iter()
                        .map(|&(j, o)| q(j, k[j]) * w(i, o))
                        .sum::<f32>();
                if r > limit * 1.000_01 {
                    over = true;
                    let (mut lo, mut hi) = (0.0f32, 1.0f32);
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        let r = kept[i]
                            + adj[i]
                                .iter()
                                .map(|&(j, o)| q(j, k[j] * mid) * w(i, o))
                                .sum::<f32>();
                        if r > limit {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    for &(j, _) in &adj[i] {
                        k[j] *= lo;
                    }
                }
            }
            if !over {
                break;
            }
        }
        // Raise: a beam still under its target with room at both ends (the
        // levels' iteration is not exact) takes what its ends allow, so every
        // softened beam has a node at the bound.
        let mut rows: Vec<f32> = (0..n)
            .map(|i| {
                kept[i]
                    + adj[i]
                        .iter()
                        .map(|&(j, o)| q(j, k[j]) * w(i, o))
                        .sum::<f32>()
            })
            .collect();
        for _ in 0..3 {
            for j in 0..nb {
                if !capped[j] || k[j] >= kt[j] {
                    continue;
                }
                let (a, b) = (self.beams[j].a as usize, self.beams[j].b as usize);
                let room = |kk: f32| {
                    let d = q(j, kk) - q(j, k[j]);
                    rows[a] + d * w(a, b) <= limit && rows[b] + d * w(b, a) <= limit
                };
                let new = if room(kt[j]) {
                    kt[j]
                } else {
                    let (mut lo, mut hi) = (k[j], kt[j]);
                    for _ in 0..40 {
                        let mid = 0.5 * (lo + hi);
                        if room(mid) {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    lo
                };
                let d = q(j, new) - q(j, k[j]);
                rows[a] += d * w(a, b);
                rows[b] += d * w(b, a);
                k[j] = new;
            }
        }
        for (wh, &f) in self.wheels.iter_mut().zip(&tyre_f) {
            wh.spring *= f;
            wh.damp *= f.sqrt();
        }
        for (j, b) in self.beams.iter_mut().enumerate() {
            if b.target_k == 0.0 {
                b.target_k = b.mat.k;
            }
            if capped[j] && k[j] < kt[j] {
                let f = k[j] / kt[j];
                b.mat.k *= f;
                b.mat.damp *= f.sqrt();
            }
        }
        Ok(())
    }

    /// Final node masses: named masses where they were put, `structure` kg spread
    /// by the structural weights, every node at least its stability floor. Returns
    /// an error when the floors alone outweigh the budget.
    pub fn solve_masses(&mut self, structure: f32) -> Result<(), String> {
        for b in self.beams.clone() {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let w = 0.5 * l * b.mat.density;
            for i in [b.a, b.b] {
                if self.nodes[i as usize].sheet {
                    continue;
                }
                let f = if self.nodes[i as usize].primary {
                    PRIMARY_WEIGHT
                } else {
                    1.0
                };
                self.nodes[i as usize].weight += w * f;
            }
        }
        let (k, _) = self.node_rates();
        let floor: Vec<f32> = self.floors(1.0);
        let fixed: f32 = self.nodes.iter().map(|n| n.fixed).sum();
        let total = |s: f32| -> f32 {
            self.nodes
                .iter()
                .zip(&floor)
                .map(|(n, &f)| (n.fixed + s * n.weight).max(f))
                .sum()
        };
        let want = fixed + structure;
        if total(0.0) > want {
            let mut by_part = vec![0.0f32; self.parts.len()];
            for (n, &f) in self.nodes.iter().zip(&floor) {
                by_part[n.part as usize] += n.fixed.max(f);
            }
            let parts: Vec<String> = self
                .parts
                .iter()
                .zip(&by_part)
                .map(|(p, kg)| format!("{p} {kg:.0}"))
                .collect();
            let body: Vec<usize> = (0..self.nodes.len())
                .filter(|&i| self.nodes[i].part == 0)
                .collect();
            let mut deg = vec![0usize; self.nodes.len()];
            for b in &self.beams {
                deg[b.a as usize] += 1;
                deg[b.b as usize] += 1;
            }
            let worst = *body
                .iter()
                .max_by(|&&a, &&b| floor[a].total_cmp(&floor[b]))
                .unwrap();
            eprintln!(
                "body nodes {} mean floor {:.3} kg mean degree {:.1}; worst node {} at {:?}: floor {:.2} kg, degree {}, Σk {:.2e}",
                body.len(),
                body.iter().map(|&i| floor[i]).sum::<f32>() / body.len() as f32,
                body.iter().map(|&i| deg[i] as f32).sum::<f32>() / body.len() as f32,
                worst,
                self.nodes[worst].p,
                floor[worst],
                deg[worst],
                k[worst]
            );
            return Err(format!(
                "the stability floors need {:.0} kg, more than the {:.0} kg budget ({})",
                total(0.0),
                want,
                parts.join(", ")
            ));
        }
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        while total(hi) < want {
            hi *= 2.0;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if total(mid) < want {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let s = 0.5 * (lo + hi);
        for (n, &f) in self.nodes.iter_mut().zip(&floor) {
            n.mass = (n.fixed + s * n.weight).max(f);
        }
        Ok(())
    }

    /// The highest Σk/m and Σc/m over the nodes (1/s², 1/s).
    pub fn worst_rates(&self) -> (f32, f32) {
        let (k, c) = self.node_rates();
        let mut wk = 0.0f32;
        let mut wc = 0.0f32;
        for (i, n) in self.nodes.iter().enumerate() {
            wk = wk.max(k[i] / n.mass);
            wc = wc.max(c[i] / n.mass);
        }
        (wk, wc)
    }

    pub fn total_mass(&self) -> f32 {
        self.nodes.iter().map(|n| n.mass).sum::<f32>()
            + self.wheels.iter().map(|w| w.mass).sum::<f32>()
    }

    /// The skeleton as Rigs of Rods `.truck` text.
    pub fn to_truck(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "{}", self.name);
        let _ = writeln!(
            s,
            "; Generated by engine-softbody::authoring (original Rubble content).\n; {} nodes, {} beams, {} collision triangles, {} wheels.",
            self.nodes.len(),
            self.beams.len(),
            self.collision.len(),
            self.wheels.len()
        );
        let _ = writeln!(s, "fileformatversion 450");
        // Every node carries its own mass (`l`, kg): no dry mass to spread.
        let _ = writeln!(s, "globals\n0.0, 0.0, tracks/semi\n");
        let _ = writeln!(s, "minimass\n0.05\n");
        let _ = writeln!(s, "enable_advanced_deformation\n");
        if self.exact_lengths {
            let _ = writeln!(s, "rubble_exact_lengths\n");
        }
        if self.beamng_profile {
            let _ = writeln!(s, "rubble_beamng_profile\n");
            if let Some((d, l)) = self.drag_coef {
                let _ = writeln!(s, "rubble_drag_coef {d} {l}\n");
            }
        }
        if self.full_contact {
            let _ = writeln!(s, "rubble_full_contact\n");
        }
        if let Some(g) = self.rest_ground {
            let _ = writeln!(s, "rubble_rest_pose {g:.6}\n");
        }
        let _ = writeln!(s, "nodes");
        for (i, n) in self.nodes.iter().enumerate() {
            let opt = if n.ground { "l" } else { "lc" };
            let _ = writeln!(
                s,
                "{i}, {:.6}, {:.6}, {:.6}, {opt}, {:.3}",
                n.p.x, n.p.y, n.p.z, n.mass
            );
        }
        let _ = writeln!(s, "\nbeams");
        // Beams of one material together, so the defaults change rarely.
        struct Line {
            k: f32,
            c: f32,
            d: f32,
            st: f32,
            p: f32,
            group: u32,
            support: Option<f32>,
            rope: bool,
            pre: f32,
            a: u32,
            b: u32,
        }
        let mut lines: Vec<Line> = self
            .beams
            .iter()
            .map(|b| {
                // (Its rest length: the spawn length unless the pose is not
                // its rest, `rest_scale`.)
                let l = self.pos(b.a).distance(self.pos(b.b)) * b.rest_scale;
                let k = round2(b.mat.k_at(l));
                // Too short to yield safely: elastic up to its break, which is
                // then near the yield force (a ductile grade's break is far past
                // its yield: elastic all that way, a short beam would store the
                // crash and spring it back).
                let (d, st) = if l < MIN_YIELD_LENGTH {
                    let st = b.mat.strength.min(SHORT_BREAK * b.mat.deform);
                    (st, st)
                } else if let Some(r) = self.rupture(b) {
                    (b.mat.deform, r.strength)
                } else {
                    (b.mat.deform, b.mat.strength)
                };
                Line {
                    k,
                    c: round2(b.mat.damp),
                    d: round2(d),
                    st: round2(st),
                    p: b.mat.plastic,
                    group: b.group,
                    support: match b.ty {
                        BeamType::Support(f) => Some(f),
                        _ => None,
                    },
                    rope: b.ty == BeamType::Rope,
                    pre: b.rest_scale,
                    a: b.a,
                    b: b.b,
                }
            })
            .collect();
        lines.sort_by(|x, y| {
            x.group
                .cmp(&y.group)
                .then(x.k.total_cmp(&y.k))
                .then(x.c.total_cmp(&y.c))
                .then(x.d.total_cmp(&y.d))
                .then(x.st.total_cmp(&y.st))
                .then(x.p.total_cmp(&y.p))
                .then(x.a.cmp(&y.a))
                .then(x.b.cmp(&y.b))
        });
        let mut cur: Option<(f32, f32, f32, f32, f32)> = None;
        let mut group = 0u32;
        for l in &lines {
            if l.group != group {
                if l.group == 0 {
                    let _ = writeln!(s, "detacher_group end");
                } else {
                    let _ = writeln!(s, "detacher_group {}", l.group);
                }
                group = l.group;
            }
            let props = (l.k, l.c, l.d, l.st, l.p);
            if cur != Some(props) {
                let _ = writeln!(
                    s,
                    "set_beam_defaults {}, {}, {}, {}, 0.05, tracks/beam, {}",
                    l.k, l.c, l.d, l.st, l.p
                );
                cur = Some(props);
            }
            // (A rest length off the spawn pose: Rubble's fifth field.)
            let pre = if l.pre != 1.0 {
                format!(", {:.7}", l.pre)
            } else {
                String::new()
            };
            match l.support {
                Some(f) => {
                    let _ = writeln!(s, "{}, {}, is, {f}{pre}", l.a, l.b);
                }
                None if l.rope => {
                    let _ = writeln!(
                        s,
                        "{}, {}, ir{}",
                        l.a,
                        l.b,
                        if pre.is_empty() {
                            String::new()
                        } else {
                            format!(", 0{pre}")
                        }
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "{}, {}, i{}",
                        l.a,
                        l.b,
                        if pre.is_empty() {
                            String::new()
                        } else {
                            format!(", 0{pre}")
                        }
                    );
                }
            }
        }
        if group != 0 {
            let _ = writeln!(s, "detacher_group end");
        }
        // (Rubble's extension: JBeam's plastic limits on those beams.)
        let limited: Vec<&SkBeam> = self
            .beams
            .iter()
            .filter(|b| {
                (b.plastic_limits.is_some() || self.rupture(b).is_some())
                    && b.ty == BeamType::Normal
                    && self.pos(b.a).distance(self.pos(b.b)) * b.rest_scale >= MIN_YIELD_LENGTH
            })
            .collect();
        if !limited.is_empty() {
            let _ = writeln!(
                s,
                "\nrubble_jbeam_plastic\n; a, b, deformLimitStress (N), deformLimit, deformLimitExpansion"
            );
            for b in limited {
                let r = self.rupture(b);
                let (cap, d) = b.plastic_limits.unwrap_or((-1.0, -1.0));
                let stress = match (b.plastic_limits, r) {
                    (Some(_), _) => round2(cap * round2(b.mat.deform)),
                    (None, Some(r)) => round2(r.cap),
                    (None, None) => -1.0,
                };
                let de = r.map_or(-1.0, |r| r.expansion);
                let _ = writeln!(s, "{}, {}, {stress}, {d}, {de}", b.a, b.b);
            }
        }
        // (Rubble's extension: a tension-side yield on those beams.)
        let tension: Vec<&SkBeam> = self
            .beams
            .iter()
            .filter(|b| {
                b.tension_yield > b.mat.deform
                    && b.ty == BeamType::Normal
                    && self.pos(b.a).distance(self.pos(b.b)) * b.rest_scale >= MIN_YIELD_LENGTH
            })
            .collect();
        if !tension.is_empty() {
            let _ = writeln!(s, "\nrubble_tension_yield\n; a, b, Tt (N)");
            for b in tension {
                let _ = writeln!(s, "{}, {}, {}", b.a, b.b, round2(b.tension_yield));
            }
        }
        if !self.shocks.is_empty() {
            // Past its bounds a shock moves to the beam defaults' spring and
            // damping (the bump and rebound stops): kept within what its
            // knuckle's light nodes allow for the step (a 12,000 N·s/m stop on
            // a 1.4 kg node would kick the knuckle over when the suspension
            // tops out).
            let stop = self.shocks.iter().map(|x| x.stop).fold(0.0, f32::max);
            let _ = writeln!(
                s,
                "\nset_beam_defaults {stop}, {STOP_DAMP}, 2000000, 1000000, 0.05, tracks/beam, 0\nshocks"
            );
            for x in self.shocks.iter().chain(&self.straps) {
                let _ = writeln!(
                    s,
                    "{}, {}, {}, {}, {}, {}, {:.4}, i",
                    x.a, x.b, x.spring, x.damp, x.short, x.long, x.precomp
                );
            }
        }
        let contacters: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| self.nodes[i].contacter)
            .collect();
        if !contacters.is_empty() {
            let _ = writeln!(s, "\ncontacters");
            for i in contacters {
                let _ = writeln!(s, "{i}");
            }
        }
        let _ = writeln!(s, "\nsubmesh\ncab");
        for (k, t) in self.collision.iter().enumerate() {
            match self.collision_breaks.get(k).copied().flatten() {
                // (Rubble's extension: a fifth field, its break group.)
                Some((g, m)) => {
                    let _ = writeln!(s, "{}, {}, {}, c, {g}, {m}", t[0], t[1], t[2]);
                }
                None => {
                    let _ = writeln!(s, "{}, {}, {}, c", t[0], t[1], t[2]);
                }
            }
        }
        if let Some((d, l)) = self.aero_exposed {
            let _ = writeln!(s, "\nrubble_aero_tri\n; a, b, c, dragCoef, liftCoef");
            for k in self.exposed_triangles() {
                let t = self.collision[k];
                let _ = writeln!(s, "{}, {}, {}, {d}, {l}", t[0], t[1], t[2]);
            }
        }
        let [c, b, l] = self.cameras;
        let _ = writeln!(s, "\ncameras\n{c}, {b}, {l}");
        if !self.wheels.is_empty() {
            // Tyres take the beam defaults in force: back to the format's.
            let _ = writeln!(s, "\nset_beam_defaults -1, -1, -1, -1\nwheels");
            for w in &self.wheels {
                let _ = writeln!(
                    s,
                    "{:.3}, {:.3}, {}, {}, {}, 9999, {}, {}, 9999, {:.1}, {:.0}, {:.0}, tracks/wheelface, tracks/wheelband1",
                    w.radius,
                    w.width,
                    w.rays,
                    w.a1,
                    w.a2,
                    w.braked as u32,
                    w.propulsed as u32,
                    w.mass,
                    w.spring,
                    w.damp
                );
            }
        }
        if !self.tyre_pose.is_empty() {
            let _ = writeln!(s, "\nrubble_tyre_pose");
            for (w, [a, b]) in self.tyre_rest_axle.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "{w}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
                    a.x, a.y, a.z, b.x, b.y, b.z
                );
            }
            for (w, ps) in self.tyre_pose.iter().enumerate() {
                for (k, p) in ps.iter().enumerate() {
                    let _ = writeln!(s, "{w}, {k}, {:.6}, {:.6}, {:.6}", p.x, p.y, p.z);
                }
            }
        }
        if !self.spawn_contacts.is_empty() {
            let _ = writeln!(s, "\nrubble_spawn_contacts");
            for ([a, b, c], n) in &self.spawn_contacts {
                let _ = writeln!(s, "{a}, {b}, {c}, {n}");
            }
        }
        let _ = writeln!(s, "\nend");
        s
    }
}

/// Two significant digits, so material lines repeat.
pub fn round2(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = 10f32.powi(x.abs().log10().floor() as i32 - 1);
    (x / e).round() * e
}
