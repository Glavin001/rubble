//! Scripted loads and probes of a scene as the GPU evaluates them: load terms (a time
//! function times a world-fixed force or a body-fixed face normal, with a lever arm in
//! the body frame) and probe items, built on the host from the scene and the solver
//! mirror. Exposure, clearing and the blast cache follow stress-ref `World` exactly; the
//! blast physics is the reference's own (`stress_ref::blast`).

use std::collections::{BinaryHeap, HashMap};

use stress_ref::blast::FaceBlast;
use stress_ref::contact::OBox;
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::scene::{ChunkSelector, LoadDesc, ProbeKind, Scene, SectionComponent, TimeFunction};
use stress_ref::solver::ReferenceSolver;
use stress_ref::structure::Face;

/// A time function as the GPU evaluates it (`world.slang`, FN_*).
#[derive(Clone, Debug)]
pub enum Function {
    Constant(f64),
    Ramp { t0: f64, t1: f64, value: f64 },
    HalfSine { start: f64, duration: f64, peak: f64 },
    Friedlander { arrival: f64, peak: f64, duration: f64, decay: f64 },
    Table(Vec<[f64; 2]>),
    Blast(FaceBlast),
    Replacement { start: f64, duration: f64 },
    /// api.rs impact `Pulse` (from `start`): `plateau` for `plateau_time`, then a
    /// half-sine of `peak` over `duration`.
    Pulse { start: f64, plateau: f64, plateau_time: f64, peak: f64, duration: f64 },
}

impl Function {
    pub fn of(f: &TimeFunction) -> Function {
        match f {
            TimeFunction::Constant { value } => Function::Constant(*value),
            TimeFunction::Ramp { t0, t1, value } => Function::Ramp { t0: *t0, t1: *t1, value: *value },
            TimeFunction::HalfSine { start, duration, peak } => Function::HalfSine { start: *start, duration: *duration, peak: *peak },
            TimeFunction::Friedlander { arrival, peak, duration, decay } => Function::Friedlander { arrival: *arrival, peak: *peak, duration: *duration, decay: *decay },
            TimeFunction::Table { points } => Function::Table(points.clone()),
        }
    }

    /// The function's value at `t` (on the host, as the reference evaluates it).
    pub fn value(&self, t: f64) -> f64 {
        match self {
            Function::Constant(value) => TimeFunction::Constant { value: *value }.eval(t),
            Function::Ramp { t0, t1, value } => TimeFunction::Ramp { t0: *t0, t1: *t1, value: *value }.eval(t),
            Function::HalfSine { start, duration, peak } => TimeFunction::HalfSine { start: *start, duration: *duration, peak: *peak }.eval(t),
            Function::Friedlander { arrival, peak, duration, decay } => {
                TimeFunction::Friedlander { arrival: *arrival, peak: *peak, duration: *duration, decay: *decay }.eval(t)
            }
            Function::Table(points) => TimeFunction::Table { points: points.clone() }.eval(t),
            Function::Blast(fb) => fb.pressure(t),
            Function::Replacement { .. } => 0.0,
            Function::Pulse { start, plateau, plateau_time, peak, duration } => {
                let tau = t - start;
                if tau < 0.0 {
                    0.0
                } else if tau < *plateau_time {
                    *plateau
                } else if tau - plateau_time > *duration {
                    0.0
                } else {
                    peak * (std::f64::consts::PI * (tau - plateau_time) / duration).sin()
                }
            }
        }
    }
}

pub const LOAD_WORLD_FORCE: u32 = 0;
pub const LOAD_FACE: u32 = 1;
/// A world-fixed force at a point fixed in the cluster frame: `arm` is the point's offset
/// from the chunk's rest centre (body frame); the lever follows the chunk's hidden
/// displacement (`ChunkLoads::add_at` at the chunk's current position).
pub const LOAD_POINT: u32 = 3;
pub const LOAD_REPLACEMENT: u32 = 2;

/// One load term on a chunk, in the order the reference adds it.
#[derive(Clone, Debug)]
pub struct LoadTerm {
    pub structure: usize,
    pub chunk: usize,
    pub kind: u32,
    /// World direction (world force), body outward normal (face), or body force (replacement).
    pub dir: Vec3,
    /// Face area (face terms).
    pub area: f64,
    /// Lever arm in the body frame (world force, face), or body moment (replacement).
    pub arm: Vec3,
    pub function: Function,
}

pub const PROBE_DISPLACEMENT: u32 = 0;
pub const PROBE_VELOCITY: u32 = 1;
pub const PROBE_SECTION_BOND: u32 = 2;
pub const PROBE_REACTION: u32 = 3;

/// What a probe item reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Chunk(usize, usize),
    Bond(usize, usize),
}

/// One contribution to a probe (the probe's value is the sum of its items).
#[derive(Clone, Debug)]
pub struct ProbeItem {
    pub probe: usize,
    pub kind: u32,
    pub target: Target,
    /// Section bonds: 0 = side a, 1 = side b.
    pub side: u32,
    /// Axis (displacement, velocity, reaction) or force weight (section).
    pub a: Vec3,
    /// Moment weight (section).
    pub b: Vec3,
    /// Initial world position (displacement).
    pub x0: Vec3,
}

impl ProbeItem {
    /// Identity across rebuilds (topology changes).
    pub fn key(&self) -> (usize, u32, Target, u32) {
        (self.probe, self.kind, self.target, self.side)
    }
}

fn face_towards(faces: &[Face], dir: Vec3) -> Option<usize> {
    let best = (0..faces.len()).max_by(|&i, &j| faces[i].normal.dot(dir).total_cmp(&faces[j].normal.dot(dir)))?;
    (faces[best].normal.dot(dir) > 0.99).then_some(best)
}

/// Exposed area per structure, chunk and face (world.rs `refresh_exposure`).
pub fn exposure(m: &ReferenceSolver) -> Vec<Vec<Vec<f64>>> {
    m.structures
        .iter()
        .enumerate()
        .map(|(s, st)| {
            let mut e: Vec<Vec<f64>> = st.chunks.iter().map(|c| c.faces.iter().map(|f| f.area).collect()).collect();
            for b in m.bonds[s].iter().filter(|b| b.alive) {
                let sb = &st.bonds[b.source];
                let (a, bb) = (b.geometry.a, b.geometry.b);
                let fa = face_towards(&st.chunks[a].faces, b.geometry.normal).unwrap_or(sb.face_a);
                let fb = face_towards(&st.chunks[bb].faces, -b.geometry.normal).unwrap_or(sb.face_b);
                e[a][fa] -= b.geometry.area;
                e[bb][fb] -= b.geometry.area;
            }
            for (c, row) in e.iter_mut().enumerate() {
                if !m.chunks[s][c].active {
                    row.iter_mut().for_each(|v| *v = 0.0);
                }
                for v in row.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            e
        })
        .collect()
}

/// World centre and outward normal of a chunk face.
fn face_world(m: &ReferenceSolver, s: usize, c: usize, f: usize) -> (Vec3, Vec3) {
    let ch = &m.structures[s].chunks[c];
    let cl = &m.clusters[m.chunks[s][c].cluster];
    let face = ch.faces[f];
    (cl.pose.transform_point(ch.center + m.chunks[s][c].u + face.offset), cl.pose.transform_vector(face.normal))
}

fn chunk_box(m: &ReferenceSolver, s: usize, c: usize) -> OBox {
    let ch = &m.structures[s].chunks[c];
    let st = &m.chunks[s][c];
    let cl = &m.clusters[st.cluster];
    let hidden = Quat::from_axis_angle(st.th, st.th.norm()).to_mat3();
    OBox { center: m.chunk_position(s, c), rotation: cl.rotation() * hidden * ch.rotation, half: ch.half_extents, hull: ch.hull.clone(), parts: ch.parts.clone() }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct OrdF64(f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Distance from each exposed face to the nearest free edge (world.rs `clearing_distances`).
pub fn clearing_distances(m: &ReferenceSolver, exposure: &[Vec<Vec<f64>>], s: usize) -> HashMap<(usize, usize), f64> {
    let st = &m.structures[s];
    let exposed = |c: usize, f: usize| exposure[s][c][f] > 0.5 * st.chunks[c].faces[f].area;
    type F = (usize, usize);
    let mut neighbours: HashMap<F, Vec<(F, f64)>> = HashMap::new();
    for (c, ch) in st.chunks.iter().enumerate() {
        if !m.chunks[s][c].active {
            continue;
        }
        for f in 0..ch.faces.len() {
            if !exposed(c, f) {
                continue;
            }
            let n = ch.faces[f].normal;
            let mut list = Vec::new();
            for &bi in &m.chunk_bonds[s][c] {
                let b = &m.bonds[s][bi];
                if b.geometry.normal.dot(n).abs() > 0.1 {
                    continue;
                }
                let o = if b.geometry.a == c { b.geometry.b } else { b.geometry.a };
                if let Some(of) = face_towards(&st.chunks[o].faces, n) {
                    if st.chunks[o].faces[of].normal.dot(n) > 0.99 && exposed(o, of) {
                        let d = (st.chunks[o].center + st.chunks[o].faces[of].offset - ch.center - ch.faces[f].offset).norm();
                        list.push(((o, of), d));
                    }
                }
            }
            neighbours.insert((c, f), list);
        }
    }
    let mut dist: HashMap<F, f64> = HashMap::new();
    let mut heap = BinaryHeap::new();
    for (k, list) in &neighbours {
        if list.len() < 4 {
            let half = 0.5 * st.chunks[k.0].faces[k.1].area.sqrt();
            dist.insert(*k, half);
            heap.push(std::cmp::Reverse((OrdF64(half), *k)));
        }
    }
    while let Some(std::cmp::Reverse((OrdF64(d), k))) = heap.pop() {
        if dist.get(&k).is_some_and(|&x| x < d) {
            continue;
        }
        for &(o, w) in &neighbours[&k] {
            let nd = d + w;
            if dist.get(&o).is_none_or(|&x| nd < x) {
                dist.insert(o, nd);
                heap.push(std::cmp::Reverse((OrdF64(nd), o)));
            }
        }
    }
    dist
}

/// Faces of a blast as the reference caches them (world.rs `rebuild_blast`).
fn blast_faces(scene: &Scene, m: &ReferenceSolver, exposure: &[Vec<Vec<f64>>], position: Vec3, tnt: f64, time: f64) -> Vec<(usize, usize, usize, f64, FaceBlast)> {
    let mut boxes = Vec::new();
    for (s, st) in m.structures.iter().enumerate() {
        for c in 0..st.chunks.len() {
            if m.chunks[s][c].active {
                boxes.push((s, c, chunk_box(m, s, c)));
            }
        }
    }
    let mut faces = Vec::new();
    for s in 0..m.structures.len() {
        let clearing = clearing_distances(m, exposure, s);
        for c in 0..m.structures[s].chunks.len() {
            if !m.chunks[s][c].active {
                continue;
            }
            for f in 0..exposure[s][c].len() {
                let area = exposure[s][c][f];
                if area <= 0.0 {
                    continue;
                }
                let (x, n) = face_world(m, s, c, f);
                let to_charge = position - x;
                let r = to_charge.norm().max(1e-6);
                let cos_theta = n.dot(to_charge) / r;
                let start = x + n * 1e-6;
                let features = &scene.sim.features;
                let shadowed = features.blast_shadowing && cos_theta > 0.0 && boxes.iter().any(|(bs, bc, b)| (*bs, *bc) != (s, c) && b.segment_hit(start, position).is_some());
                let s_clear = if features.blast_clearing { clearing.get(&(c, f)).copied().unwrap_or(0.0) } else { f64::INFINITY };
                faces.push((s, c, f, area, FaceBlast::new(tnt, time, r, cos_theta, shadowed, s_clear)));
            }
        }
    }
    faces
}

/// The scene's loads and probes, with the reference's caching of blast faces (rebuilt
/// only when the topology changed since they were built).
pub struct WorldLoads {
    pub scene: Scene,
    /// Bumped by the host whenever topology changes (splits, removals).
    pub topology_version: u64,
    blast: Vec<Option<(u64, Vec<(usize, usize, usize, f64, FaceBlast)>)>>,
    pub initial_positions: Vec<Vec<Vec3>>,
}

impl WorldLoads {
    pub fn new(scene: &Scene, m: &ReferenceSolver) -> WorldLoads {
        let initial_positions = m
            .structures
            .iter()
            .map(|s| {
                let pose = Pose::new(s.initial_position, s.initial_rotation);
                s.chunks.iter().map(|c| pose.transform_point(c.center)).collect()
            })
            .collect();
        WorldLoads { scene: scene.clone(), topology_version: 0, blast: vec![None; scene.loads.len()], initial_positions }
    }

    /// Load terms in effect from time `t` (the segment start) on.
    pub fn terms(&mut self, m: &ReferenceSolver, t: f64) -> Vec<LoadTerm> {
        let mut out = Vec::new();
        let mut exp: Option<Vec<Vec<Vec<f64>>>> = None;
        for li in 0..self.scene.loads.len() {
            match self.scene.loads[li].clone() {
                LoadDesc::PointForce { body, chunk, direction, magnitude, point } => {
                    let s = m.structure_index(&body).expect("load body");
                    if !m.chunks[s][chunk].active {
                        continue;
                    }
                    let center = m.structures[s].chunks[chunk].center;
                    let arm = point.map(|p| Vec3::from_array(p) - center).unwrap_or(Vec3::ZERO);
                    out.push(LoadTerm {
                        structure: s,
                        chunk,
                        kind: LOAD_WORLD_FORCE,
                        dir: Vec3::from_array(direction).normalized(),
                        area: 1.0,
                        arm,
                        function: Function::of(&magnitude),
                    });
                }
                LoadDesc::Pressure { body, chunks, face_normal, pressure } => {
                    let exposure = exp.get_or_insert_with(|| exposure(m));
                    let s = m.structure_index(&body).expect("load body");
                    let n = Vec3::from_array(face_normal).normalized();
                    for c in chunks.select(&self.scene.bodies[s]) {
                        if !m.chunks[s][c].active {
                            continue;
                        }
                        for f in 0..exposure[s][c].len() {
                            let area = exposure[s][c][f];
                            let face = m.structures[s].chunks[c].faces[f];
                            if area <= 0.0 || face.normal.dot(n) < 0.99 {
                                continue;
                            }
                            out.push(LoadTerm { structure: s, chunk: c, kind: LOAD_FACE, dir: face.normal, area, arm: face.offset, function: Function::of(&pressure) });
                        }
                    }
                }
                LoadDesc::Blast { position, tnt_mass, time } => {
                    if t + 1e-12 < time {
                        continue;
                    }
                    let stale = self.blast[li].as_ref().is_none_or(|b| b.0 != self.topology_version);
                    if stale {
                        let exposure = exp.get_or_insert_with(|| exposure(m));
                        let faces = blast_faces(&self.scene, m, exposure, Vec3::from_array(position), tnt_mass, time);
                        self.blast[li] = Some((self.topology_version, faces));
                    }
                    for &(s, c, f, area, fb) in &self.blast[li].as_ref().unwrap().1 {
                        if !m.chunks[s][c].active {
                            continue;
                        }
                        let face = m.structures[s].chunks[c].faces[f];
                        out.push(LoadTerm { structure: s, chunk: c, kind: LOAD_FACE, dir: face.normal, area, arm: face.offset, function: Function::Blast(fb) });
                    }
                }
            }
        }
        for rl in &m.replacements {
            if !m.chunks[rl.structure][rl.chunk].active {
                continue;
            }
            out.push(LoadTerm {
                structure: rl.structure,
                chunk: rl.chunk,
                kind: LOAD_REPLACEMENT,
                dir: rl.force,
                area: 1.0,
                arm: rl.moment,
                function: Function::Replacement { start: rl.start, duration: rl.duration },
            });
        }
        out
    }

    /// Times at which a blast first acts (segments start there so its faces are cached
    /// with the geometry of that moment, as the reference does).
    /// The scripted loads at time `t` (world.rs `apply_scripted_loads`): world forces,
    /// torques about the chunk centres, as the GPU applies them.
    pub fn chunk_loads(&mut self, m: &ReferenceSolver, t: f64) -> stress_ref::solver::ChunkLoads {
        let mut l = stress_ref::solver::ChunkLoads::new(m);
        for term in self.terms(m, t) {
            if term.kind == LOAD_REPLACEMENT {
                continue;
            }
            let (s, c) = (term.structure, term.chunk);
            let rot = m.clusters[m.chunks[s][c].cluster].rotation();
            let value = term.function.value(t);
            let f = if term.kind == LOAD_WORLD_FORCE || term.kind == LOAD_POINT { term.dir * value } else { rot * term.dir * (-value * term.area) };
            l.force[s][c] += f;
            let lever = if term.kind == LOAD_POINT { term.arm - m.chunks[s][c].u } else { term.arm };
            l.torque[s][c] += (rot * lever).cross(f);
        }
        l
    }

    pub fn blast_times(&self) -> Vec<f64> {
        self.scene
            .loads
            .iter()
            .filter_map(|l| match l {
                LoadDesc::Blast { time, .. } => Some(*time),
                _ => None,
            })
            .collect()
    }

    /// Probe items for the current topology.
    pub fn probe_items(&self, m: &ReferenceSolver) -> Vec<ProbeItem> {
        let mut out = Vec::new();
        for (p, probe) in self.scene.probes.iter().enumerate() {
            match &probe.kind {
                ProbeKind::ChunkDisplacement { body, chunk, axis } => {
                    let s = m.structure_index(body).expect("probe body");
                    out.push(ProbeItem {
                        probe: p,
                        kind: PROBE_DISPLACEMENT,
                        target: Target::Chunk(s, *chunk),
                        side: 0,
                        a: Vec3::from_array(*axis).normalized(),
                        b: Vec3::ZERO,
                        x0: self.initial_positions[s][*chunk],
                    });
                }
                ProbeKind::ChunkVelocity { body, chunk, axis } => {
                    let s = m.structure_index(body).expect("probe body");
                    out.push(ProbeItem { probe: p, kind: PROBE_VELOCITY, target: Target::Chunk(s, *chunk), side: 0, a: Vec3::from_array(*axis).normalized(), b: Vec3::ZERO, x0: Vec3::ZERO });
                }
                ProbeKind::SectionForce { body, point, normal, region, component } => {
                    let s = m.structure_index(body).expect("probe body");
                    let pt = Vec3::from_array(*point);
                    let n = Vec3::from_array(*normal).normalized();
                    let (cf, cm) = match component {
                        SectionComponent::Normal => (-n, Vec3::ZERO),
                        SectionComponent::Force(a) => (Vec3::from_array(*a).normalized(), Vec3::ZERO),
                        SectionComponent::Moment(a) => (Vec3::ZERO, Vec3::from_array(*a).normalized()),
                    };
                    let st = &m.structures[s];
                    for (bi, b) in m.bonds[s].iter().enumerate() {
                        if !b.alive {
                            continue;
                        }
                        let g = &b.geometry;
                        if let Some(r) = region {
                            if !r.contains(g.centroid) {
                                continue;
                            }
                        }
                        let side_a = (st.chunks[g.a].center - pt).dot(n);
                        let side_b = (st.chunks[g.b].center - pt).dot(n);
                        if side_a * side_b >= 0.0 {
                            continue;
                        }
                        let (side, xc) = if side_a > 0.0 { (0, st.chunks[g.a].center) } else { (1, st.chunks[g.b].center) };
                        out.push(ProbeItem { probe: p, kind: PROBE_SECTION_BOND, target: Target::Bond(s, bi), side, a: cf + cm.cross(xc - pt), b: cm, x0: Vec3::ZERO });
                    }
                }
                ProbeKind::Reaction { body, chunks, axis } => {
                    let s = m.structure_index(body).expect("probe body");
                    for c in select(chunks, &self.scene, s) {
                        out.push(ProbeItem { probe: p, kind: PROBE_REACTION, target: Target::Chunk(s, c), side: 0, a: Vec3::from_array(*axis).normalized(), b: Vec3::ZERO, x0: Vec3::ZERO });
                    }
                }
                ProbeKind::ImpactorVelocity { .. } | ProbeKind::ImpactorPosition { .. } => {
                    // Impactors are not on the GPU yet: the probe stays NaN.
                }
            }
        }
        out
    }
}

fn select(sel: &ChunkSelector, scene: &Scene, s: usize) -> Vec<usize> {
    sel.select(&scene.bodies[s])
}
