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
/// Bound on h²ω² + 2hγ per node (the stability of a damped mode under
/// symplectic Euler needs it under 4), as a rate: 0.9 × 4/h², the Gershgorin row
/// estimate of both, [`Skeleton::floors`].
pub const G_LIMIT: f32 = 0.9 * 4.0 / (STEP * STEP);
/// Largest Σc/m a node may carry (1/s): h·Σc/m ≤ 0.75.
pub const C_RATIO: f32 = 1500.0;
/// Lightest node (kg).
pub const MIN_NODE_MASS: f32 = 0.05;
/// Shortest beam a yield threshold may act on: the solver keeps every yielded
/// beam at least this long, so a shorter one is made elastic up to its break.
pub const MIN_YIELD_LENGTH: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BeamType {
    Normal,
    /// Compression only; breaks past `factor` × its rest length of stretch.
    Support(f32),
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
    pub wheels: Vec<SkWheel>,
    /// Fine skin triangles (wound outward): the visual binds to these.
    pub skin: Vec<[u32; 3]>,
    /// Collision triangles (wound outward): a coarser closed skin.
    pub collision: Vec<[u32; 3]>,
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
        });
        (self.nodes.len() - 1) as u32
    }

    pub fn pos(&self, i: u32) -> Vec3 {
        self.nodes[i as usize].p
    }

    pub fn has_beam(&self, a: u32, b: u32) -> bool {
        self.index.contains_key(&(a.min(b), a.max(b)))
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
        });
        true
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
        let n = self.nodes.len();
        // Per node: (neighbour, stiffness, damping).
        let mut adj: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); n];
        for b in &self.beams {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let f = if b.fixed { 1.0 } else { s };
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

    /// Final node masses: named masses where they were put, `structure` kg spread
    /// by the structural weights, every node at least its stability floor. Returns
    /// an error when the floors alone outweigh the budget.
    pub fn solve_masses(&mut self, structure: f32) -> Result<(), String> {
        for b in self.beams.clone() {
            let l = self.pos(b.a).distance(self.pos(b.b));
            let w = 0.5 * l * b.mat.density;
            self.nodes[b.a as usize].weight += w;
            self.nodes[b.b as usize].weight += w;
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
        let _ = writeln!(s, "nodes");
        for (i, n) in self.nodes.iter().enumerate() {
            let opt = if n.ground { "l" } else { "lc" };
            let _ = writeln!(
                s,
                "{i}, {:.4}, {:.4}, {:.4}, {opt}, {:.3}",
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
            a: u32,
            b: u32,
        }
        let mut lines: Vec<Line> = self
            .beams
            .iter()
            .map(|b| {
                let l = self.pos(b.a).distance(self.pos(b.b));
                let k = round2(b.mat.k_at(l));
                // Too short to yield safely: elastic up to its break.
                let d = if l < MIN_YIELD_LENGTH {
                    b.mat.strength.max(b.mat.deform)
                } else {
                    b.mat.deform
                };
                Line {
                    k,
                    c: round2(b.mat.damp),
                    d: round2(d),
                    st: round2(b.mat.strength),
                    p: b.mat.plastic,
                    group: b.group,
                    support: match b.ty {
                        BeamType::Support(f) => Some(f),
                        BeamType::Normal => None,
                    },
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
            match l.support {
                Some(f) => {
                    let _ = writeln!(s, "{}, {}, is, {f}", l.a, l.b);
                }
                None => {
                    let _ = writeln!(s, "{}, {}, i", l.a, l.b);
                }
            }
        }
        if group != 0 {
            let _ = writeln!(s, "detacher_group end");
        }
        if !self.shocks.is_empty() {
            let stop = self.shocks.iter().map(|x| x.stop).fold(0.0, f32::max);
            let _ = writeln!(
                s,
                "\nset_beam_defaults 9000000, 12000, {stop}, 1000000, 0.05, tracks/beam, 0\nshocks"
            );
            for x in &self.shocks {
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
        for t in &self.collision {
            let _ = writeln!(s, "{}, {}, {}, c", t[0], t[1], t[2]);
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
