//! The world coupled to an external rigid-body engine; see `engine.rs` for the design.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use crate::engine::RestOverlap;
use crate::solver::SolverEvent;

use super::*;

/// A body taking part in island selection (or the ground).
struct Part {
    key: Option<BodyKey>,
    /// Bounds swept over the frame.
    lo: Vec3,
    hi: Vec3,
    /// Centre of mass and bounds at the start of the frame (the ground: a half-space).
    center: Vec3,
    box_lo: Vec3,
    box_hi: Vec3,
    velocity: Vec3,
    angular_velocity: Vec3,
    /// Largest distance of its surface from the centre of mass (for rotation sweeps).
    reach: f64,
    /// Infinite for supported clusters and the ground.
    mass: f64,
    /// Plane-strain modulus `E / (1 - nu^2)` of its (softest) material, as simulated.
    modulus: f64,
    /// Radius of curvature at a contact (sphere radius, half the smallest box size).
    contact_radius: f64,
    /// Present for clusters with bonds: what makes the rigid assumption fail on them.
    struck: Option<Struck>,
    /// Its bounds were grown by the reach of debris an impactor may launch from it this
    /// frame (`World::debris_reach_routing`).
    debris: bool,
}

/// An impactor contact the routing sweep predicts for the coming frame
/// (`World::predict_contacts`).
#[derive(Clone, Debug)]
pub struct PredictedContact {
    pub impactor: usize,
    pub cluster: u64,
    pub cluster_chunks: usize,
    pub structure: usize,
    /// The struck chunk (the cluster's nearest to the impactor's path).
    pub entry: usize,
    /// Time within the frame at which the bounds first meet, and the impactor's centre then.
    pub time: f64,
    pub point: Vec3,
    /// Unit closing direction (impactor relative to the cluster) and closing speed.
    pub direction: Vec3,
    pub speed: f64,
    /// Reduced mass of the impactor and the whole cluster (the impactor's for an anchored
    /// one), the contact's plane-strain modulus and the impactor's face radius.
    pub reduced_mass: f64,
    pub modulus: f64,
    pub face_radius: f64,
    /// Hertz contact time and the half-sine peak force of an elastic impulse over it.
    pub hertz_duration: f64,
    pub peak_force: f64,
    /// The punch's plug (`None`: the cone takes the whole cluster).
    pub plug: Option<PunchPlug>,
}

/// The plug the punch would push out of a struck cluster.
#[derive(Clone, Debug)]
pub struct PunchPlug {
    /// Plug chunks (sorted) and the connected bonds holding it in (cluster order).
    pub chunks: Vec<usize>,
    pub interface: Vec<usize>,
    /// Summed tensile capacity of `interface` (N), plug mass, the inelastic impulse on it,
    /// its Hertz time and the mean force over it.
    pub capacity: f64,
    pub mass: f64,
    pub impulse: f64,
    pub duration: f64,
    pub force: f64,
}

impl PunchPlug {
    pub fn breaks(&self) -> bool {
        self.force > self.capacity
    }
}

/// Properties of a breakable body that decide whether a contact on it may be rigid.
#[derive(Clone, Copy)]
struct Struck {
    /// Time for a bar wave to cross the body and back (s).
    round_trip: f64,
    /// Tensile capacity of its weakest bond (N).
    capacity: f64,
}

/// Whether the rigid-body assumption fails for `a` striking breakable `b`: the impact
/// (Hertz contact time at the closing speed) is shorter than a wave round trip across
/// `b`, so only part of `b` takes part, or its peak force (elastic impulse over a
/// half-sine of that duration) exceeds `b`'s weakest bond, so `b` breaks during the
/// contact. Both follow from the bodies' masses, moduli, sizes and strengths.
fn rigid_assumption_fails(a: &Part, b: &Part, breaking_only: bool) -> bool {
    let Some(struck) = b.struck else { return false };
    let speed = closing_speed(a, b);
    let m = match (a.mass.is_finite(), b.mass.is_finite()) {
        (true, true) => a.mass * b.mass / (a.mass + b.mass),
        (true, false) => a.mass,
        (false, true) => b.mass,
        (false, false) => return false,
    };
    if speed <= 0.0 {
        return false;
    }
    let modulus = 1.0 / (1.0 / a.modulus + 1.0 / b.modulus);
    let duration = hertz_duration(m, a.contact_radius.min(b.contact_radius), modulus, speed);
    let peak_force = std::f64::consts::PI * (2.0 * m * speed) / (2.0 * duration);
    (duration < struck.round_trip && !breaking_only) || peak_force > struck.capacity
}

/// Speed at which `a` and `b` approach. The contact normal is estimated from each
/// body's centre to the nearest point of the other's bounds (both ways, the larger
/// closing speed kept), which stays right for a small body landing anywhere on a large
/// one; spin adds its full surface speed. If either centre lies inside the other's
/// bounds there is no normal to estimate and the relative speed is used.
fn closing_speed(a: &Part, b: &Part) -> f64 {
    let spin = a.angular_velocity.norm() * a.reach + b.angular_velocity.norm() * b.reach;
    let rel = a.velocity - b.velocity;
    // The ground is a half-space: the normal is vertical wherever the body is.
    if b.key.is_none() {
        return -rel.z + spin;
    }
    if a.key.is_none() {
        return rel.z + spin;
    }
    let towards = |from: Vec3, lo: Vec3, hi: Vec3| {
        let nearest = from.component_max(lo).component_min(hi);
        let d = nearest - from;
        (d.norm() > 0.0).then(|| d.normalized())
    };
    let normals = [towards(a.center, b.box_lo, b.box_hi), towards(b.center, a.box_lo, a.box_hi).map(|n| -n)];
    if normals.iter().any(Option::is_none) {
        return rel.norm() + spin;
    }
    normals.iter().flatten().map(|n| rel.dot(*n)).fold(f64::NEG_INFINITY, f64::max) + spin
}

/// The tiered coupling's impact speed (m/s): an impactor contact closing slower is a
/// load (`STRESS_TIERED_SPEED`, default 5; a measurement knob until calibrated).
/// A cluster's engine-body signature: its structure, chunks and anchoring (the body
/// changes when any of them does).
pub fn cluster_signature(cl: &crate::solver::Cluster) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (cl.structure, &cl.chunks, cl.anchored).hash(&mut hasher);
    hasher.finish()
}

pub fn tiered_impact_speed() -> f64 {
    std::env::var("STRESS_TIERED_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0)
}

fn overlaps(a: &Part, b: &Part) -> bool {
    (0..3).all(|i| a.lo[i] <= b.hi[i] && b.lo[i] <= a.hi[i])
}

fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
        r = parent[r];
    }
    let mut j = i;
    while parent[j] != r {
        let next = parent[j];
        parent[j] = r;
        j = next;
    }
    r
}

fn interpolate(a: &BodyMotion, b: &BodyMotion, s: f64) -> (Pose, Vec3, Vec3) {
    let mut qb = b.pose.rotation;
    let qa = a.pose.rotation;
    if qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z < 0.0 {
        qb = Quat { w: -qb.w, x: -qb.x, y: -qb.y, z: -qb.z };
    }
    let q = Quat {
        w: qa.w + (qb.w - qa.w) * s,
        x: qa.x + (qb.x - qa.x) * s,
        y: qa.y + (qb.y - qa.y) * s,
        z: qa.z + (qb.z - qa.z) * s,
    }
    .normalized();
    let p = a.pose.position + (b.pose.position - a.pose.position) * s;
    (Pose::new(p, q), a.velocity + (b.velocity - a.velocity) * s, a.angular_velocity + (b.angular_velocity - a.angular_velocity) * s)
}

impl World {
    /// The world coupled to an external rigid-body engine, which then owns rigid motion
    /// and contact except in impact islands (see `engine.rs`). Runs the explicit solve.
    pub fn with_engine(scene: &Scene, engine: Box<dyn RigidEngine>) -> World {
        assert!(
            matches!(scene.sim.solve_mode, SolveMode::Explicit | SolveMode::Adaptive | SolveMode::QuasiStatic),
            "the engine-coupled world runs the explicit, adaptive or quasi-static solve"
        );
        let mut w = World::new(scene);
        w.attach_engine(engine);
        w
    }

    /// Couple this world (at its current state) to an engine: the ground, then a body per
    /// cluster and impactor. Another solver can run the world's coupled frames on its own
    /// state through this world (`coupled_prepare`), as the GPU world does with its
    /// mirror.
    pub fn attach_engine(&mut self, mut engine: Box<dyn RigidEngine>) {
        if let Some(g) = &self.scene.ground {
            engine.add_ground(g.height);
        }
        self.coupling = Some(Coupling {
            engine: EngineCell(Some(engine)),
            filter: ContactLoadFilter::new(),
            start: HashMap::new(),
            end: HashMap::new(),
            handed: Vec::new(),
            kept: HashMap::new(),
            shapes: HashMap::new(),
            routing: HashMap::new(),
        });
        self.sync_engine_bodies();
    }

    /// Add an impactor to the running world (a projectile fired into it), coupled to the
    /// engine at once when there is one. Returns its index. Impactors are keyed by index
    /// (`BodyKey::Impactor`), so they are only ever appended.
    pub fn add_impactor(&mut self, imp: Impactor) -> usize {
        self.impactors.push(imp);
        if self.coupling.is_some() {
            self.sync_engine_bodies();
        }
        self.impactors.len() - 1
    }

    /// Name of the coupled engine, if any.
    pub fn engine_name(&self) -> Option<&str> {
        self.coupling.as_ref().map(|c| c.engine.name())
    }

    pub fn impactor_radius(imp: &Impactor) -> f64 {
        match imp.shape {
            ImpactorShape::Sphere { radius } => radius,
            ImpactorShape::Box { half_extents } => half_extents.iter().copied().fold(f64::INFINITY, f64::min),
        }
    }

    /// Every body as the engine should have it: one per cluster (boxes per chunk) and
    /// one per impactor. A cluster's shapes are built when its signature is new and
    /// shared from `cache` after that (they depend only on its chunks).
    fn engine_bodies(&self, cache: &mut HashMap<u64, (u64, std::sync::Arc<Vec<BoxShape>>)>) -> Vec<EngineBody> {
        let mut out = Vec::with_capacity(self.solver.clusters.len() + self.impactors.len());
        let mut live = std::collections::HashSet::with_capacity(self.solver.clusters.len());
        for cl in &self.solver.clusters {
            let st = &self.solver.structures[cl.structure];
            let signature = cluster_signature(cl);
            live.insert(cl.id);
            if let Some((sig, shapes)) = cache.get(&cl.id) {
                if *sig == signature {
                    out.push(EngineBody {
                        key: BodyKey::Cluster(cl.id),
                        signature,
                        geometry: BodyGeometry::Boxes(shapes.clone()),
                        fixed: cl.anchored,
                        ccd: false,
                        pose: cl.pose,
                        velocity: cl.velocity,
                        angular_velocity: cl.angular_velocity,
                    });
                    continue;
                }
            }
            let shapes: Vec<BoxShape> = cl
                .chunks
                .iter()
                .flat_map(|&c| {
                    let ch = &st.chunks[c];
                    let rotation = rotation_of(&ch.rotation);
                    match &ch.parts {
                        // A compound chunk: one hull shape per part, reported on the chunk,
                        // with the chunk's mass shared by the parts' volumes.
                        Some(parts) => {
                            let volumes: Vec<f64> = parts.iter().map(|p| p.hull.unit_mass().volume).collect();
                            let total: f64 = volumes.iter().sum();
                            parts
                                .iter()
                                .zip(volumes)
                                .map(|(p, v)| BoxShape {
                                    local: Pose::new(ch.center + ch.rotation * p.offset, rotation),
                                    half_extents: p.hull.half_extents(),
                                    mass: ch.mass * v / total,
                                    chunk: Some((cl.structure, c)),
                                    hull: Some(p.hull.vertices.clone()),
                                })
                                .collect::<Vec<_>>()
                        }
                        None => vec![BoxShape {
                            local: Pose::new(ch.center, rotation),
                            half_extents: ch.half_extents,
                            mass: ch.mass,
                            chunk: Some((cl.structure, c)),
                            hull: ch.hull.as_ref().map(|h| h.vertices.clone()),
                        }],
                    }
                })
                .collect();
            let shapes = std::sync::Arc::new(shapes);
            cache.insert(cl.id, (signature, shapes.clone()));
            out.push(EngineBody {
                key: BodyKey::Cluster(cl.id),
                signature,
                geometry: BodyGeometry::Boxes(shapes),
                fixed: cl.anchored,
                ccd: false,
                pose: cl.pose,
                velocity: cl.velocity,
                angular_velocity: cl.angular_velocity,
            });
        }
        for (i, imp) in self.impactors.iter().enumerate() {
            let geometry = match imp.shape {
                ImpactorShape::Sphere { radius } => BodyGeometry::Sphere { radius, mass: imp.mass },
                ImpactorShape::Box { half_extents } => BodyGeometry::Boxes(std::sync::Arc::new(vec![BoxShape {
                    local: Pose::default(),
                    half_extents: Vec3::from_array(half_extents),
                    mass: imp.mass,
                    chunk: None,
                    hull: None,
                }])),
            };
            out.push(EngineBody {
                key: BodyKey::Impactor(i),
                signature: 0,
                geometry,
                fixed: false,
                ccd: true,
                pose: imp.pose,
                velocity: imp.velocity,
                angular_velocity: imp.angular_velocity,
            });
        }
        cache.retain(|id, _| live.contains(id));
        out
    }

    fn sync_engine_bodies(&mut self) {
        let mut cache = std::mem::take(&mut self.coupling.as_mut().expect("coupled").shapes);
        let bodies = self.engine_bodies(&mut cache);
        let coupling = self.coupling.as_mut().expect("coupled");
        coupling.shapes = cache;
        coupling.engine.sync_bodies(&bodies);
    }

    /// The world's own view of every body's motion.
    fn world_motion(&self) -> HashMap<BodyKey, BodyMotion> {
        let mut m = HashMap::new();
        for cl in &self.solver.clusters {
            let key = BodyKey::Cluster(cl.id);
            m.insert(key, BodyMotion { key, pose: cl.pose, velocity: cl.velocity, angular_velocity: cl.angular_velocity });
        }
        for (i, imp) in self.impactors.iter().enumerate() {
            let key = BodyKey::Impactor(i);
            m.insert(key, BodyMotion { key, pose: imp.pose, velocity: imp.velocity, angular_velocity: imp.angular_velocity });
        }
        m
    }

    /// Mark this frame's impact islands (bodies the world simulates) and the driven
    /// bodies (the engine's); `forced` bodies seed islands whatever the prediction.
    /// Returns the number of bodies in islands.
    fn select_islands(&mut self, dt: f64, forced: &HashSet<BodyKey>) -> usize {
        let scale = self.scene.sim.stiffness_scale;
        let plane_strain = |m: &Material| m.youngs_modulus * scale / (1.0 - m.poisson_ratio * m.poisson_ratio);
        let mut parts: Vec<Part> = Vec::new();
        // Standalone (no engine: the learned-impact recorder's routing prediction), the
        // per-cluster cache is this call's own.
        let mut routing = self.coupling.as_mut().map(|c| std::mem::take(&mut c.routing)).unwrap_or_default();
        for ci in 0..self.solver.clusters.len() {
            let cl = &self.solver.clusters[ci];
            let s = cl.structure;
            let com = cl.com_world();
            // A cluster the caller vouches for (`ReferenceSolver::unchanged`) keeps last
            // frame's bounds and material data.
            let reuse = if self.solver.unchanged.contains(&cl.id) { routing.get(&cl.id).copied() } else { None };
            let (lo, hi, reach, modulus, speed, radius, capacity) = match reuse {
                Some(r) => r,
                None => {
            let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
            let (mut reach, mut modulus, mut speed, mut radius): (f64, f64, f64, f64) = (0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY);
            for &c in &cl.chunks {
                // `chunk_box`'s centre and bounding radius (`OBox::bounding_radius` is the
                // half extents' norm), without building the box.
                let (center, half) = (self.solver.chunk_position(s, c), self.solver.structures[s].chunks[c].half_extents);
                let r = half.norm();
                lo = lo.component_min(center - Vec3::splat(r));
                hi = hi.component_max(center + Vec3::splat(r));
                reach = reach.max((center - com).norm() + r);
                let m = self.chunk_material(s, c);
                modulus = modulus.min(plane_strain(m));
                speed = speed.min((m.youngs_modulus * scale / m.density).sqrt());
                radius = radius.min(half.min_elem());
            }
            let capacity = cl
                .bonds
                .iter()
                .map(|&bi| {
                    let b = &self.solver.bonds[s][bi];
                    b.strength.tensile * b.geometry.area * b.weibull
                })
                .fold(f64::INFINITY, f64::min);
                    let r = (lo, hi, reach, modulus, speed, radius, capacity);
                    routing.insert(cl.id, r);
                    r
                }
            };
            let extent = (hi - lo).max_elem();
            let struck = (!cl.bonds.is_empty()).then_some(Struck { round_trip: 2.0 * extent / speed, capacity });
            let sweep = cl.velocity.abs() * dt + Vec3::splat(cl.angular_velocity.norm() * reach * dt);
            parts.push(Part {
                key: Some(BodyKey::Cluster(cl.id)),
                lo: lo - sweep,
                hi: hi + sweep,
                center: com,
                box_lo: lo,
                box_hi: hi,
                velocity: cl.velocity,
                angular_velocity: cl.angular_velocity,
                reach,
                mass: if cl.anchored { f64::INFINITY } else { cl.mass },
                modulus,
                contact_radius: radius,
                struck,
                debris: false,
            });
        }
        let live: std::collections::HashSet<u64> = self.solver.clusters.iter().map(|cl| cl.id).collect();
        routing.retain(|id, _| live.contains(id));
        if let Some(c) = self.coupling.as_mut() {
            c.routing = routing;
        }
        for (i, imp) in self.impactors.iter().enumerate() {
            let b = imp.obox();
            let r = b.bounding_radius();
            let sweep = imp.velocity.abs() * dt + Vec3::splat(imp.angular_velocity.norm() * r * dt);
            parts.push(Part {
                key: Some(BodyKey::Impactor(i)),
                lo: b.center - Vec3::splat(r) - sweep,
                hi: b.center + Vec3::splat(r) + sweep,
                center: b.center,
                box_lo: b.center - Vec3::splat(r),
                box_hi: b.center + Vec3::splat(r),
                velocity: imp.velocity,
                angular_velocity: imp.angular_velocity,
                reach: r,
                mass: imp.mass,
                modulus: plane_strain(&imp.material),
                contact_radius: Self::impactor_radius(imp),
                struck: None,
                debris: false,
            });
        }
        if let Some(g) = &self.scene.ground {
            // A half-space (see `closing_speed`).
            parts.push(Part {
                key: None,
                lo: Vec3::splat(f64::NEG_INFINITY),
                hi: Vec3::new(f64::INFINITY, f64::INFINITY, g.height),
                center: Vec3::new(0.0, 0.0, g.height),
                box_lo: Vec3::splat(f64::NEG_INFINITY),
                box_hi: Vec3::new(f64::INFINITY, f64::INFINITY, g.height),
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                reach: 0.0,
                mass: f64::INFINITY,
                modulus: plane_strain(self.scene.material(&g.material)),
                contact_radius: f64::INFINITY,
                struck: None,
                debris: false,
            });
        }

        // `debris_reach_routing`: a bonded cluster an impactor reaches this frame, closing
        // faster than the tiered impact speed, may launch pieces at up to twice the closing
        // velocity (relative to the cluster: an elastic hit by a heavier body bounds it); its
        // swept bounds grow by that velocity times `dt` along it, so a structure the pieces
        // can reach within the frame overlaps it and joins its island.
        if self.debris_reach_routing {
            let clusters = self.solver.clusters.len();
            let impactors: Vec<usize> = (clusters..clusters + self.impactors.len()).collect();
            for c in 0..clusters {
                if parts[c].struck.is_none() {
                    continue;
                }
                let mut grow = (Vec3::ZERO, Vec3::ZERO);
                for &k in &impactors {
                    if !overlaps(&parts[k], &parts[c]) || closing_speed(&parts[k], &parts[c]) <= tiered_impact_speed() {
                        continue;
                    }
                    let d = (parts[k].velocity - parts[c].velocity) * (2.0 * dt);
                    grow.0 = grow.0.component_min(d);
                    grow.1 = grow.1.component_max(d);
                }
                if grow != (Vec3::ZERO, Vec3::ZERO) {
                    parts[c].lo += grow.0;
                    parts[c].hi += grow.1;
                    parts[c].debris = true;
                }
            }
        }

        // Hand-back: which clusters an impactor closes on fast; a handed-back cluster with
        // such a contact is routed again, the others stay out of every island.
        let mut out_of_routing = vec![false; parts.len()];
        if self.hand_back_after.is_some() {
            let clusters = self.solver.clusters.len();
            self.fast_contact.clear();
            for c in 0..clusters {
                let fast = (clusters..clusters + self.impactors.len()).any(|k| overlaps(&parts[k], &parts[c]) && closing_speed(&parts[k], &parts[c]) > tiered_impact_speed());
                let id = self.solver.clusters[c].id;
                if fast {
                    self.fast_contact.insert(id);
                    self.handed_back.remove(&id);
                    self.quiet_frames.remove(&id);
                }
                out_of_routing[c] = self.handed_back.contains(&id);
            }
        }

        // Owned pairs seed islands; any overlap of swept bounds joins bodies (the ground
        // and pairs of immovable bodies never do: they cannot exchange motion).
        let n = parts.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let mut seed = vec![false; n];
        // The overlapping pairs, found by sweeping the parts sorted along x (the same set
        // as testing every pair), then taken in (i, j) order as the double loop over all
        // pairs would.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| parts[a].lo.x.total_cmp(&parts[b].lo.x));
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for (k, &a) in order.iter().enumerate() {
            for &b in &order[k + 1..] {
                if parts[b].lo.x > parts[a].hi.x {
                    break;
                }
                if overlaps(&parts[a], &parts[b]) {
                    pairs.push((a.min(b), a.max(b)));
                }
            }
        }
        pairs.sort_unstable();
        pairs.retain(|&(i, j)| !out_of_routing[i] && !out_of_routing[j]);
        for (i, j) in pairs {
            {
                // Tiered: only an impactor's contact is an impact (debris and resting
                // contacts reach the static solve as the engine's loads).
                // `routing_tiered`: the deployed routing's rule (`blast_parity_routed`) with any
                // solve mode, so the oracle's own physics can be routed as the engine is.
                let tiered = self.routing_tiered || (self.qs_islands && self.scene.sim.solve_mode == SolveMode::QuasiStatic);
                let impactor = matches!(parts[i].key, Some(BodyKey::Impactor(_))) || matches!(parts[j].key, Some(BodyKey::Impactor(_)));
                let fast = !tiered || closing_speed(&parts[i], &parts[j]).max(closing_speed(&parts[j], &parts[i])) > tiered_impact_speed();
                let owned = (!tiered || impactor) && fast && (rigid_assumption_fails(&parts[i], &parts[j], false) || rigid_assumption_fails(&parts[j], &parts[i], false));
                let both_immovable = !parts[i].mass.is_finite() && !parts[j].mass.is_finite();
                if parts[i].key.is_none() || parts[j].key.is_none() {
                    // Against the ground: a breaking landing makes the body an island seed.
                    if owned {
                        seed[if parts[i].key.is_some() { i } else { j }] = true;
                    }
                    continue;
                }
                if both_immovable {
                    continue;
                }
                // `tiered_fast_joins`: a slow contact between clusters joins nothing.
                if tiered && self.tiered_fast_joins && !impactor && !fast && !(parts[i].debris || parts[j].debris) {
                    continue;
                }
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                parent[ri] = rj;
                if owned {
                    seed[i] = true;
                    seed[j] = true;
                }
            }
        }
        for (i, p) in parts.iter().enumerate() {
            if p.key.is_some_and(|k| forced.contains(&k)) {
                seed[i] = true;
            }
        }
        let mut island_root = vec![false; n];
        for i in 0..n {
            if seed[i] {
                let r = find(&mut parent, i);
                island_root[r] = true;
            }
        }
        let mut count = 0;
        for i in 0..n {
            let in_island = parts[i].key.is_some() && island_root[find(&mut parent, i)];
            match parts[i].key {
                // Parts are the clusters in order, then the impactors, then the ground.
                Some(BodyKey::Cluster(id)) => {
                    let cl = &mut self.solver.clusters[i];
                    debug_assert_eq!(cl.id, id);
                    cl.driven = !in_island;
                }
                Some(BodyKey::Impactor(k)) => self.impactors[k].driven = !in_island,
                None => {}
            }
            count += usize::from(in_island);
        }
        count
    }

    /// The number of bodies the next frame's routing would put in islands (`select_islands`
    /// over `dt` with `forced`), with every body left driven: a prediction made at the end of
    /// a frame, from the state the next one starts with.
    pub fn predict_islands(&mut self, dt: f64, forced: &HashSet<BodyKey>) -> usize {
        let n = self.select_islands(dt, forced);
        for cl in &mut self.solver.clusters {
            cl.driven = true;
        }
        for imp in &mut self.impactors {
            imp.driven = true;
        }
        n
    }

    /// The bodies the next frame's routing would put in impact islands (`select_islands`
    /// over `dt` with `forced`), as (cluster ids, impactor indices), every body's `driven`
    /// flag left as it was. Coupled or standalone (a standalone world simulates every body
    /// itself; this is then a prediction only, for the learned-impact recorder).
    pub fn predict_island_members(&mut self, dt: f64, forced: &HashSet<BodyKey>) -> (Vec<u64>, Vec<usize>) {
        let saved: (Vec<bool>, Vec<bool>) =
            (self.solver.clusters.iter().map(|c| c.driven).collect(), self.impactors.iter().map(|i| i.driven).collect());
        self.select_islands(dt, forced);
        let clusters = self.solver.clusters.iter().filter(|c| !c.driven).map(|c| c.id).collect();
        let impactors = (0..self.impactors.len()).filter(|&k| !self.impactors[k].driven).collect();
        for (cl, d) in self.solver.clusters.iter_mut().zip(saved.0) {
            cl.driven = d;
        }
        for (imp, d) in self.impactors.iter_mut().zip(saved.1) {
            imp.driven = d;
        }
        (clusters, impactors)
    }

    /// The earliest time within `dt` at which an impactor (its bounding sphere) reaches the
    /// bounds of a bonded cluster, moving at their relative velocity (0: already inside;
    /// `dt`: none). Bounds as `select_islands` takes them, so the estimate is early, never
    /// late, for a cluster that does not spin.
    pub fn first_contact_time(&self, dt: f64) -> f64 {
        self.first_contact_time_with(dt, false)
    }

    /// `first_contact_time`; with `new_only`, pairs that already overlap (an impact going
    /// on) are left out: the earliest new approach.
    pub fn first_contact_time_with(&self, dt: f64, new_only: bool) -> f64 {
        let mut t_min = dt;
        for imp in &self.impactors {
            let r = imp.obox().bounding_radius();
            let c = imp.pose.position;
            for cl in self.solver.clusters.iter().filter(|cl| !cl.bonds.is_empty()) {
                let s = cl.structure;
                let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
                for &k in &cl.chunks {
                    let (p, h) = (self.solver.chunk_position(s, k), self.solver.structures[s].chunks[k].half_extents.norm());
                    lo = lo.component_min(p - Vec3::splat(h + r));
                    hi = hi.component_max(p + Vec3::splat(h + r));
                }
                let v = imp.velocity - cl.velocity;
                // Slab test of c + v t against [lo, hi], t in [0, t_min].
                let (mut t0, mut t1) = (0.0f64, t_min);
                for a in 0..3 {
                    if v[a].abs() < 1e-12 {
                        if c[a] < lo[a] || c[a] > hi[a] {
                            t0 = f64::INFINITY;
                        }
                        continue;
                    }
                    let (ta, tb) = ((lo[a] - c[a]) / v[a], (hi[a] - c[a]) / v[a]);
                    t0 = t0.max(ta.min(tb));
                    t1 = t1.min(ta.max(tb));
                }
                if t0 <= t1 && !(new_only && t0 <= 0.0) {
                    t_min = t_min.min(t0);
                }
            }
        }
        t_min
    }

    /// The predictive punch (`STRESS_IMPACT_PUNCH=1`, a model under measurement): for each
    /// impactor that reaches a bonded cluster within `dt` closing faster than the tiered
    /// impact speed, the plug its face would punch out (the cluster's chunks inside a 45°
    /// cone from the face along the closing velocity, through the cluster, connected by
    /// bonds to the struck chunk) and the bonds holding it in. The plug breaks out when the
    /// force of an inelastic hit on it, its impulse over the Hertz contact time, exceeds
    /// those bonds' summed tensile capacity: then they are returned, to be broken before
    /// the engine steps, so the engine's rigid contact meets the free plug.
    pub fn predict_punch(&self, dt: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for c in self.predict_contacts(dt, tiered_impact_speed()) {
            let Some(plug) = c.plug else { continue };
            if std::env::var("STRESS_IMPACT_PUNCH_TRACE").is_ok() {
                eprintln!(
                    "punch: cluster {} ({} chunks): plug {} chunks {:.1} kg, {} bonds holding {:.3e} N; hit {:.1} m/s, impulse {:.0} N s over {:.2e} s = {:.3e} N: {}",
                    c.cluster, c.cluster_chunks, plug.chunks.len(), plug.mass, plug.interface.len(), plug.capacity, c.speed, plug.impulse, plug.duration, plug.force,
                    if plug.breaks() { "breaks out" } else { "holds" }
                );
            }
            if plug.breaks() {
                out.extend(plug.interface.into_iter().map(|bi| (c.structure, bi)));
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Every impactor contact with a bonded cluster the routing sweep predicts within `dt`
    /// (closing faster than `min_speed`, strictly), in impactor then cluster order, with
    /// what the punch would make of it (`predict_punch` takes those with the tiered impact
    /// speed). Read-only: the learned-impact recorder's causal contact inputs.
    pub fn predict_contacts(&self, dt: f64, min_speed: f64) -> Vec<PredictedContact> {
        let scale = self.scene.sim.stiffness_scale;
        let plane_strain = |m: &Material| m.youngs_modulus * scale / (1.0 - m.poisson_ratio * m.poisson_ratio);
        let mut out = Vec::new();
        for (ii, imp) in self.impactors.iter().enumerate() {
            let r = imp.obox().bounding_radius();
            let face = Self::impactor_radius(imp);
            let c = imp.pose.position;
            for cl in self.solver.clusters.iter().filter(|cl| !cl.bonds.is_empty()) {
                let v = imp.velocity - cl.velocity;
                let speed = v.norm();
                if speed <= min_speed {
                    continue;
                }
                let s = cl.structure;
                let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
                for &k in &cl.chunks {
                    let (p, h) = (self.solver.chunk_position(s, k), self.solver.structures[s].chunks[k].half_extents.norm());
                    lo = lo.component_min(p - Vec3::splat(h + r));
                    hi = hi.component_max(p + Vec3::splat(h + r));
                }
                let (mut t0, mut t1) = (0.0f64, dt);
                for a in 0..3 {
                    if v[a].abs() < 1e-12 {
                        if c[a] < lo[a] || c[a] > hi[a] {
                            t0 = f64::INFINITY;
                        }
                        continue;
                    }
                    let (ta, tb) = ((lo[a] - c[a]) / v[a], (hi[a] - c[a]) / v[a]);
                    t0 = t0.max(ta.min(tb));
                    t1 = t1.min(ta.max(tb));
                }
                if t0 > t1 {
                    continue;
                }
                let d = v * (1.0 / speed);
                let hit = c + v * t0;
                // The struck chunk: the cluster's nearest to the impactor's path.
                let entry = cl
                    .chunks
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let dist = |k: usize| {
                            let q = self.solver.chunk_position(s, k) - hit;
                            let along = q.dot(d);
                            (q - d * along).norm() + along.abs() * 1e-3
                        };
                        dist(a).total_cmp(&dist(b))
                    })
                    .expect("a cluster with bonds has chunks");
                let entry_depth = (self.solver.chunk_position(s, entry) - hit).dot(d);
                let in_cone = |k: usize| {
                    let q = self.solver.chunk_position(s, k) - hit;
                    let along = q.dot(d);
                    let h = self.solver.structures[s].chunks[k].half_extents.norm();
                    let lateral = (q - d * along).norm();
                    // The chunk's centre inside the cone (punching shear's 45° from the
                    // face's edge), from the struck chunk's depth on.
                    along >= entry_depth - h && lateral <= face + (along - entry_depth).max(0.0)
                };
                // The cone's chunks connected to the struck one (by the cluster's bonds).
                let member: std::collections::HashSet<usize> = cl.chunks.iter().copied().collect();
                let mut plug: std::collections::HashSet<usize> = std::collections::HashSet::new();
                let mut stack = vec![entry];
                plug.insert(entry);
                while let Some(k) = stack.pop() {
                    for &bi in &self.solver.structures[s].chunks[k].bonds {
                        let b = &self.solver.bonds[s][bi];
                        if !b.connected() {
                            continue;
                        }
                        let o = if b.geometry.a == k { b.geometry.b } else { b.geometry.a };
                        if member.contains(&o) && !plug.contains(&o) && in_cone(o) {
                            plug.insert(o);
                            stack.push(o);
                        }
                    }
                }
                let modulus = 1.0 / (1.0 / plane_strain(&imp.material) + 1.0 / plane_strain(self.chunk_material(s, entry)));
                // The routing's view of the contact: the whole cluster's reduced mass, an
                // elastic impulse over a half-sine of the Hertz time (`rigid_assumption_fails`).
                let m_red = if cl.anchored { imp.mass } else { cl.mass * imp.mass / (cl.mass + imp.mass) };
                let hertz = crate::api::hertz_duration(m_red, face, modulus, speed);
                let peak_force = std::f64::consts::PI * (2.0 * m_red * speed) / (2.0 * hertz);
                let plug = (plug.len() != cl.chunks.len()).then(|| {
                    let interface: Vec<usize> = cl
                        .bonds
                        .iter()
                        .copied()
                        .filter(|&bi| {
                            let g = &self.solver.bonds[s][bi].geometry;
                            self.solver.bonds[s][bi].connected() && (plug.contains(&g.a) != plug.contains(&g.b))
                        })
                        .collect();
                    let capacity: f64 = interface.iter().map(|&bi| {
                        let b = &self.solver.bonds[s][bi];
                        b.strength.tensile * b.geometry.area * b.weibull
                    }).sum();
                    let m_plug: f64 = plug.iter().map(|&k| self.solver.structures[s].chunks[k].mass).sum();
                    let m_eff = m_plug * imp.mass / (m_plug + imp.mass);
                    let impulse = m_eff * speed;
                    let tau = crate::api::hertz_duration(m_eff, face, modulus, speed);
                    let mut chunks: Vec<usize> = plug.iter().copied().collect();
                    chunks.sort_unstable();
                    PunchPlug { chunks, interface, capacity, mass: m_plug, impulse, duration: tau, force: impulse / tau }
                });
                out.push(PredictedContact {
                    impactor: ii,
                    cluster: cl.id,
                    cluster_chunks: cl.chunks.len(),
                    structure: s,
                    entry,
                    time: t0,
                    point: hit,
                    direction: d,
                    speed,
                    reduced_mass: m_red,
                    modulus,
                    face_radius: face,
                    hertz_duration: hertz,
                    peak_force,
                    plug,
                });
            }
        }
        out
    }

    /// Every body's motion taken from the engine as it is (after frames the engine stepped
    /// without the world following it, such as a resident engine's).
    pub fn take_engine_motion(&mut self) {
        let coupling = self.coupling.as_mut().expect("coupled");
        let end: HashMap<BodyKey, BodyMotion> = coupling.engine.motion().into_iter().map(|m| (m.key, m)).collect();
        coupling.start = end.clone();
        coupling.end = end;
        for cl in &mut self.solver.clusters {
            cl.driven = true;
        }
        for imp in &mut self.impactors {
            imp.driven = true;
        }
        self.set_driven_motion(1.0);
    }

    /// Impulse of the scripted loads (and removal ramps) over the frame on each driven
    /// free cluster, for the engine (gravity excluded: the engine applies it).
    fn scripted_impulses(&mut self, t0: f64, h: f64, n: usize) -> Vec<(BodyKey, Vec3, Vec3)> {
        let free: Vec<usize> =
            (0..self.solver.clusters.len()).filter(|&ci| self.solver.clusters[ci].driven && !self.solver.clusters[ci].anchored).collect();
        if free.is_empty() || (self.scene.loads.is_empty() && self.solver.replacements.is_empty()) {
            return Vec::new();
        }
        let g = self.solver.config.gravity;
        let mut acc = vec![(Vec3::ZERO, Vec3::ZERO); free.len()];
        for k in 0..n {
            self.loads.resize(&self.solver);
            self.loads.clear();
            self.apply_scripted_loads(t0 + (k as f64 + 0.5) * h);
            for (i, &ci) in free.iter().enumerate() {
                let (f, tq) = self.solver.net_load(ci, &self.loads);
                acc[i].0 += (f - g * self.solver.clusters[ci].mass) * h;
                acc[i].1 += tq * h;
            }
        }
        free.iter().zip(acc).map(|(&ci, (j, l))| (BodyKey::Cluster(self.solver.clusters[ci].id), j, l)).collect()
    }

    /// Driven bodies at fraction `s` of the frame, between the engine's start and end.
    pub fn set_driven_motion(&mut self, s: f64) {
        let c = self.coupling.as_ref().expect("coupled");
        for cl in self.solver.clusters.iter_mut().filter(|c| c.driven && !c.anchored) {
            let key = BodyKey::Cluster(cl.id);
            if let (Some(a), Some(b)) = (c.start.get(&key), c.end.get(&key)) {
                (cl.pose, cl.velocity, cl.angular_velocity) = interpolate(a, b, s);
            }
        }
        for (i, imp) in self.impactors.iter_mut().enumerate().filter(|(_, m)| m.driven) {
            let key = BodyKey::Impactor(i);
            if let (Some(a), Some(b)) = (c.start.get(&key), c.end.get(&key)) {
                (imp.pose, imp.velocity, imp.angular_velocity) = interpolate(a, b, s);
            }
        }
    }

    /// Layer contact: hand the engine the indentation of every chunk pair it steps this
    /// frame (either chunk's cluster driven): the overlap the layer contact holds at rest
    /// (`Stick::indent`), which the engine then holds as rest too, so the handover adds
    /// no energy (tests `handover_energy.rs`). The world's own pair loop does not see
    /// those pairs this frame (they are the engine's); their contact state comes back
    /// with the engine's end of the frame (`take_rest_overlaps_from_engine`).
    fn hand_rest_overlaps_to_engine(&mut self) {
        if !self.scene.sim.methods.layer_contact {
            return;
        }
        let solver = &self.solver;
        let engine_owned = |s: usize, c: usize| solver.chunks[s][c].active && solver.clusters[solver.chunks[s][c].cluster].driven;
        let keys: Vec<ContactKey> = self
            .sticks
            .iter()
            .filter(|(k, st)| st.indent > 0.0 && matches!(k, ContactKey::Pair(sa, ca, sb, cb) if engine_owned(*sa, *ca) || engine_owned(*sb, *cb)))
            .map(|(k, _)| *k)
            .collect();
        let mut handed: Vec<(ContactKey, Stick)> = keys.into_iter().map(|k| (k, self.sticks.remove(&k).expect("listed"))).collect();
        handed.sort_by_key(|(k, _)| match k {
            ContactKey::Pair(sa, ca, sb, cb) => (*sa, *ca, *sb, *cb),
            _ => unreachable!("pair keys only"),
        });
        let overlaps: Vec<RestOverlap> = handed
            .iter()
            .map(|(k, st)| match k {
                ContactKey::Pair(sa, ca, sb, cb) => RestOverlap { a: (*sa, *ca), b: (*sb, *cb), indent: st.indent },
                _ => unreachable!("pair keys only"),
            })
            .collect();
        let coupling = self.coupling.as_mut().expect("coupled");
        coupling.kept.clear();
        if overlaps.is_empty() || coupling.engine.set_rest_overlaps(&overlaps) {
            coupling.handed = handed;
        } else {
            // An engine that cannot hold them: the world keeps them.
            coupling.handed.clear();
            for (k, st) in handed {
                self.sticks.insert(k, st);
            }
        }
    }

    /// The indentations the engine still holds return to the world's contact state (the
    /// pair's friction layer, which the engine owned this frame, starts unstrained); a
    /// pair the engine separated has forgotten its own, as the layer contact forgets it.
    fn take_rest_overlaps_from_engine(&mut self) {
        let coupling = self.coupling.as_mut().expect("coupled");
        let handed = std::mem::take(&mut coupling.handed);
        let kept = std::mem::take(&mut coupling.kept);
        for (k, st) in handed {
            let ContactKey::Pair(sa, ca, sb, cb) = k else { continue };
            if let Some(&indent) = kept.get(&((sa, ca), (sb, cb))) {
                self.sticks.entry(k).or_insert(Stick { indent, crushed: st.crushed, normal: st.normal, ..Default::default() });
            }
        }
    }

    /// Engine contacts on driven clusters as contact impulses for the load filter.
    fn engine_contacts_to_impulses(&self, contacts: &[crate::engine::EngineContact]) -> Vec<ContactImpulse> {
        let ground_modulus = self.scene.ground.as_ref().map(|g| self.scene.material(&g.material).youngs_modulus).unwrap_or(3e10);
        contacts
            .iter()
            .filter(|c| self.solver.chunks[c.structure][c.chunk].active && self.solver.clusters[self.solver.chunks[c.structure][c.chunk].cluster].driven)
            .map(|c| {
                let (other_mass, other_modulus, other_radius) = match c.other {
                    Some(BodyKey::Impactor(i)) => {
                        let imp = &self.impactors[i];
                        (imp.mass, imp.material.youngs_modulus, Self::impactor_radius(imp))
                    }
                    Some(BodyKey::Cluster(id)) => match self.solver.clusters.iter().find(|cl| cl.id == id) {
                        Some(cl) => {
                            let ch = &self.solver.structures[cl.structure].chunks[cl.chunks[0]];
                            (if cl.anchored { f64::INFINITY } else { cl.mass }, ch.youngs_modulus, ch.half_extents.min_elem())
                        }
                        None => (f64::INFINITY, ground_modulus, 1.0),
                    },
                    None => (f64::INFINITY, ground_modulus, 1.0),
                };
                ContactImpulse {
                    structure: c.structure,
                    chunk: c.chunk,
                    point: c.point,
                    impulse: c.impulse,
                    other_mass,
                    approach_speed: c.approach_speed,
                    other_modulus,
                    other_radius,
                    crush: None,
                    other: c.other.map(|k| match k {
                        BodyKey::Cluster(id) => 2 * id,
                        BodyKey::Impactor(i) => 2 * i as u64 + 1,
                    }),
                }
            })
            .collect()
    }

    /// One frame with the engine: islands, engine step, substeps, verification (redo
    /// with more islands if a body the engine moved fractured), write-back, sync.
    pub(super) fn step_frame_coupled(&mut self) {
        let mut forced: HashSet<BodyKey> = HashSet::new();
        let frozen = loop {
            let saved = self.clone();
            self.engine_save();
            let (frozen, driven) = self.coupled_attempt(&forced);
            let broke = self.fractured_since(&saved, &driven);
            if broke.is_empty() || broke.is_subset(&forced) {
                self.engine_discard();
                break frozen;
            }
            // The engine's assumption failed for these bodies: solve the frame again
            // with them in islands.
            let engine = std::mem::replace(&mut self.coupling.as_mut().expect("coupled").engine, EngineCell(None));
            *self = saved;
            let c = self.coupling.as_mut().expect("coupled");
            c.engine = engine;
            c.engine.restore();
            self.redone_frames += 1;
            forced.extend(broke);
        };
        self.coupled_write_back(&frozen);
    }

    /// Remember the engine's motion (the start of a frame that may be redone).
    pub fn engine_save(&mut self) {
        self.coupling.as_mut().expect("coupled").engine.save();
    }

    /// Forget the engine's saved motion.
    pub fn engine_discard(&mut self) {
        self.coupling.as_mut().expect("coupled").engine.discard();
    }

    /// Return the engine to its saved motion (the world's own state is restored by the
    /// caller).
    pub fn engine_restore(&mut self) {
        self.coupling.as_mut().expect("coupled").engine.restore();
    }

    /// world.rs `contact_stable_dt` over the contacts that can engage this frame only: a
    /// certified bound on the spectral radius of the mass-normalised penalty stiffness
    /// (`contact_network_radius`), each pair at its largest total stiffness `k` (the
    /// ground immovable). A penalty pair never exceeds its face-pair stiffness
    /// (`k / max(points, 10)` per point; the ground `k / max(points, 5)`), and
    /// `pair_stiffness_bound` bounds that over every direction the pair can press along. Same deformation inertia as the
    /// standalone bound, which instead presses every chunk on six faces by a copy of
    /// itself in the scene's stiffest material. Penalty contact (not the layer
    /// contact's lattice bound). Infinite with nothing engaged.
    ///
    /// A chunk pair whose two chunks both have a `PlannedPose` is bounded over the
    /// directions it can press along while the plan stands (`pair_stiffness_within`)
    /// instead of every direction.
    pub fn engaged_contact_dt(
        &self,
        chunks: &[((usize, usize), Vec<ContactPartner>)],
        impactors: &[(usize, Vec<(usize, usize)>)],
        poses: &HashMap<(usize, usize), PlannedPose>,
        impactor_poses: &HashMap<usize, PlannedPose>,
    ) -> f64 {
        let scale = self.scene.sim.stiffness_scale;
        let chunk_box = |s: usize, c: usize| {
            let ch = &self.solver.structures[s].chunks[c];
            (OBox { center: Vec3::ZERO, rotation: ch.rotation, half: ch.half_extents, hull: None, parts: None }, ch.mass * self.solver.chunks[s][c].inertia_scale)
        };
        let ground = self.scene.ground.as_ref().map(|g| self.scene.material(&g.material));
        // The contact network as a graph of bodies: `diag` the sum of each body's contact
        // stiffnesses over its mass, `edges` each pair's `k / sqrt(m_a m_b)` once.
        let mut node: HashMap<ContactPartner, usize> = HashMap::new();
        let mut mass: Vec<f64> = Vec::new();
        let mut index = |p: ContactPartner, m: f64, mass: &mut Vec<f64>| -> usize {
            *node.entry(p).or_insert_with(|| {
                mass.push(m);
                mass.len() - 1
            })
        };
        let mut diag: Vec<f64> = Vec::new();
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        let mut add = |i: usize, j: Option<usize>, k: f64, mass: &Vec<f64>, diag: &mut Vec<f64>| {
            diag.resize(mass.len(), 0.0);
            diag[i] += k / mass[i];
            if let Some(j) = j {
                diag[j] += k / mass[j];
                edges.push((i, j, k / (mass[i] * mass[j]).sqrt()));
            }
        };
        let g = Vec3::from_array(self.scene.gravity).norm();
        for ((s, c), partners) in chunks {
            let (s, c) = (*s, *c);
            let (b, m) = chunk_box(s, c);
            let mat = self.chunk_material(s, c);
            let i = index(ContactPartner::Chunk(s, c), m, &mut mass);
            for p in partners {
                match *p {
                    // Each pair once (both chunks list it).
                    ContactPartner::Chunk(ps, pc) if (s, c) < (ps, pc) => {
                        let (pb, pm) = chunk_box(ps, pc);
                        let k = match (poses.get(&(s, c)), poses.get(&(ps, pc))) {
                            (Some(pa), Some(pp)) => pair_stiffness_within(mat, &b.half, pa, self.chunk_material(ps, pc), &pb.half, pp, g, scale),
                            _ => pair_stiffness_bound(mat, &b, self.chunk_material(ps, pc), &pb, scale),
                        };
                        let j = index(ContactPartner::Chunk(ps, pc), pm, &mut mass);
                        add(i, Some(j), k, &mass, &mut diag);
                    }
                    ContactPartner::Chunk(..) => {}
                    ContactPartner::Impactor(ii) => {
                        let imp = &self.impactors[ii];
                        let ib = imp.obox();
                        let k = match (poses.get(&(s, c)), impactor_poses.get(&ii)) {
                            (Some(pa), Some(pi)) => pair_stiffness_within(mat, &b.half, pa, &imp.material, &ib.half, pi, g, scale),
                            _ => pair_stiffness_bound(mat, &b, &imp.material, &ib, scale),
                        };
                        let j = index(ContactPartner::Impactor(ii), imp.mass, &mut mass);
                        add(i, Some(j), k, &mass, &mut diag);
                    }
                    // The ground presses with the chunk's own box on its side (world.rs).
                    ContactPartner::Ground => {
                        if let Some(gm) = ground {
                            add(i, None, pair_stiffness_bound(mat, &b, gm, &b, scale), &mass, &mut diag);
                        }
                    }
                }
            }
        }
        for (ii, _) in impactors {
            let imp = &self.impactors[*ii];
            let i = index(ContactPartner::Impactor(*ii), imp.mass, &mut mass);
            // On the ground a box presses up to its 8 corners at `k / min(points, 5)` each.
            if let Some(gm) = ground {
                let ib = imp.obox();
                add(i, None, 8.0 / 5.0 * pair_stiffness_bound(&imp.material, &ib, gm, &ib, scale), &mass, &mut diag);
            }
        }
        diag.resize(mass.len(), 0.0);
        let w2 = contact_network_radius(&diag, &edges);
        if w2 <= 0.0 {
            f64::INFINITY
        } else {
            2.0 / w2.sqrt() * self.scene.sim.courant_safety
        }
    }

    /// The end of a coupled frame: island bodies that still exist go back to the engine
    /// with the world's state; new fragments are created by the sync with theirs.
    pub fn coupled_write_back(&mut self, frozen: &[BodyKey]) {
        let t = web_time::Instant::now();
        self.take_rest_overlaps_from_engine();
        let t1 = t.elapsed().as_secs_f64() * 1e3;
        let motion = self.world_motion();
        let t2 = t.elapsed().as_secs_f64() * 1e3;
        let back: Vec<BodyMotion> = frozen.iter().filter_map(|k| motion.get(k).copied()).collect();
        self.coupling.as_mut().expect("coupled").engine.set_motion(&back);
        let t3 = t.elapsed().as_secs_f64() * 1e3;
        self.sync_engine_bodies();
        if std::env::var("STRESS_WRITE_BACK_TRACE").is_ok() {
            eprintln!("write-back: overlaps {t1:.2}, motion {:.2}, set {:.2}, sync bodies {:.2} ms", t2 - t1, t3 - t2, t.elapsed().as_secs_f64() * 1e3 - t3);
        }
    }

    /// Bodies (by their key at the start of the frame, in `driven`) with a bond broken
    /// since `saved`.
    pub(super) fn fractured_since(&self, saved: &World, driven: &HashSet<BodyKey>) -> HashSet<BodyKey> {
        fractured_since(&saved.solver, &self.solver.events[saved.solver.events.len()..], driven)
    }

    /// One attempt at the frame. Returns the bodies held frozen in the engine and the
    /// clusters it drove.
    fn coupled_attempt(&mut self, forced: &HashSet<BodyKey>) -> (Vec<BodyKey>, HashSet<BodyKey>) {
        let f = self.coupled_prepare(forced);
        self.coupled_substeps(&f);
        (f.frozen, f.driven)
    }

    /// The second half of a coupled frame (after `coupled_prepare`): the substeps, which
    /// simulate the impact islands with the driven bodies following the engine's motion,
    /// and the frame's tail. Needs no engine: it runs as well on a clone of the prepared
    /// world (the coupled reference's oracle island step, `step_frame_with_frame_provider`).
    pub fn coupled_substeps(&mut self, f: &CoupledFrame) {
        self.frame_loads.resize(&self.solver);
        let splits_before = self.solver.events.len();
        for i in 0..f.n {
            self.set_driven_motion((i + 1) as f64 / f.n as f64);
            self.substep(f.h);
            let mut fl = std::mem::take(&mut self.frame_loads);
            fl.add_scaled(&self.loads, 1.0 / f.n as f64);
            self.frame_loads = fl;
        }
        self.frame += 1;
        if self.solver.events.len() != splits_before {
            self.topology_version += 1;
        }
        if self.scene.sim.solve_mode == SolveMode::Adaptive {
            self.settle_quiet_clusters();
            self.advance_settled_fatigue();
        }
        if let Some(u) = self.scene.sim.refine_utilization {
            self.refine_where_needed(u);
        }
        self.frame_loads.clear();
    }

    /// The first half of a coupled frame, before its substeps: the impact islands, the
    /// engine's step with the island bodies frozen, and the engine's contacts on driven
    /// clusters taken into the load filter. The substeps (`n` of `h`) then follow the
    /// driven bodies' motion (`set_driven_motion`) and simulate the islands; another
    /// solver (the GPU's) can run them on its own state, as this world's do.
    pub fn coupled_prepare(&mut self, forced: &HashSet<BodyKey>) -> CoupledFrame {
        let fdt = self.scene.sim.frame_dt;
        let t0 = self.solver.time;
        // Without impact islands (and always in quasi-static mode, which has no explicit
        // dynamics to resolve one) the engine drives every body.
        let island_bodies = if self.impact_islands && (self.scene.sim.solve_mode != SolveMode::QuasiStatic || self.qs_islands) {
            self.select_islands(fdt, forced)
        } else {
            for cl in &mut self.solver.clusters {
                cl.driven = true;
            }
            for imp in &mut self.impactors {
                imp.driven = true;
            }
            0
        };
        if island_bodies > 0 {
            self.island_frames += 1;
            self.max_island_bodies = self.max_island_bodies.max(island_bodies);
        }
        let movable = self.solver.clusters.iter().filter(|c| !c.anchored).map(|c| c.driven).chain(self.impactors.iter().map(|m| m.driven));
        for driven in movable {
            if driven {
                self.engine_body_frames += 1;
            } else {
                self.island_body_frames += 1;
            }
        }
        let driven: HashSet<BodyKey> = self.solver.clusters.iter().filter(|c| c.driven).map(|c| BodyKey::Cluster(c.id)).collect();
        // Adaptive: a cluster in an impact island is integrated explicitly.
        if self.scene.sim.solve_mode == SolveMode::Adaptive {
            for cl in self.solver.clusters.iter_mut().filter(|c| !c.driven && c.activity != Activity::Active) {
                cl.activity = Activity::Active;
                cl.active_timer = self.solver.config.active_time;
            }
        }
        // The same substep as the standalone world (`substep_dt`), contacts or not: the
        // engine only owns rigid motion, so the stress solution must not depend on it.
        // (Taking only the stress limit without islands gave a larger step than the
        // standalone world's under `scaled_step_bound`, and b3's reflected tension moved
        // 0.6 % with it; with the established bound the two coincided.)
        let mut dt = self.substep_dt().min(fdt);
        if let Some(m) = self.scene.sim.max_substep {
            dt = dt.min(m);
        }
        let n = (fdt / dt).ceil().max(1.0) as usize;
        let h = fdt / n as f64;

        let impulses = self.scripted_impulses(t0, h, n);
        let start = self.world_motion();
        let frozen: Vec<BodyKey> = self
            .solver
            .clusters
            .iter()
            .filter(|c| !c.driven)
            .map(|c| BodyKey::Cluster(c.id))
            .chain(self.impactors.iter().enumerate().filter(|(_, m)| !m.driven).map(|(i, _)| BodyKey::Impactor(i)))
            .collect();
        self.hand_rest_overlaps_to_engine();
        let coupling = self.coupling.as_mut().expect("coupled");
        let contacts = coupling.engine.step(fdt, &frozen, &impulses);
        if !coupling.handed.is_empty() {
            coupling.kept = coupling.engine.rest_overlaps().into_iter().map(|r| ((r.a, r.b), r.indent)).collect();
        }
        let end: HashMap<BodyKey, BodyMotion> = coupling.engine.motion().into_iter().map(|m| (m.key, m)).collect();
        if let Some(watch) = std::env::var("STRESS_REF_WATCH_ENGINE_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()) {
            // Diagnostic: the engine's contacts on one chunk (structure 0) this frame.
            for c in contacts.iter().filter(|c| c.structure == 0 && c.chunk == watch) {
                eprintln!(
                    "engine contact t {t0:.4}: chunk {watch} at {:?}, impulse {:?} N s (|J| {:.3e}), other {:?}, approach {:.3e} m/s",
                    c.point.to_array(),
                    c.impulse.to_array(),
                    c.impulse.norm(),
                    c.other,
                    c.approach_speed
                );
            }
        }
        coupling.start = start;
        coupling.end = end;
        let impulses_in = self.engine_contacts_to_impulses(&contacts);
        let coupling = self.coupling.as_mut().expect("coupled");
        coupling.filter.ingest(&self.solver, t0, fdt, &impulses_in);
        CoupledFrame { t0, n, h, frozen, driven, impulses: impulses_in }
    }
}

/// A coupled frame's schedule after the engine stepped (`World::coupled_prepare`).
#[derive(Clone, Debug)]
pub struct CoupledFrame {
    pub t0: f64,
    /// Substeps of the frame and their length.
    pub n: usize,
    pub h: f64,
    /// Bodies the engine held frozen (the impact islands).
    pub frozen: Vec<BodyKey>,
    /// Clusters the engine drove.
    pub driven: HashSet<BodyKey>,
    /// The engine's contact impulses on driven clusters, as the load filter took them.
    pub impulses: Vec<ContactImpulse>,
}

/// Driven bodies (by their key in `saved`, at the start of the frame) with a bond broken
/// among `events` (those since `saved`).
/// The cluster id of every chunk, by structure (what `fractured_since` reads of a saved
/// solver, without copying it).
pub fn chunk_clusters(m: &ReferenceSolver) -> Vec<Vec<u64>> {
    m.chunks.iter().map(|cs| cs.iter().map(|c| m.clusters.get(c.cluster).map_or(u64::MAX, |cl| cl.id)).collect()).collect()
}

/// `fractured_since` from the frame-start `chunk_clusters` and bond endpoints (bonds are
/// never added, so the current solver's geometry is the saved one's).
pub fn fractured_since_clusters(start: &[Vec<u64>], m: &ReferenceSolver, events: &[SolverEvent], driven: &HashSet<BodyKey>) -> HashSet<BodyKey> {
    events
        .iter()
        .filter_map(|e| match e {
            SolverEvent::Broken { structure, bond, .. } => {
                let chunk = m.bonds[*structure][*bond].geometry.a;
                let key = BodyKey::Cluster(start[*structure][chunk]);
                driven.contains(&key).then_some(key)
            }
            _ => None,
        })
        .collect()
}

pub fn fractured_since(saved: &ReferenceSolver, events: &[SolverEvent], driven: &HashSet<BodyKey>) -> HashSet<BodyKey> {
    events
        .iter()
        .filter_map(|e| match e {
            SolverEvent::Broken { structure, bond, .. } => {
                let chunk = saved.bonds[*structure][*bond].geometry.a;
                let key = BodyKey::Cluster(saved.clusters[saved.chunks[*structure][chunk].cluster].id);
                driven.contains(&key).then_some(key)
            }
            _ => None,
        })
        .collect()
}

/// Unit quaternion of an orthonormal rotation matrix.
fn rotation_of(m: &Mat3) -> Quat {
    let a = &m.m;
    let tr = m.trace();
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        Quat { w: 0.25 * s, x: (a[2][1] - a[1][2]) / s, y: (a[0][2] - a[2][0]) / s, z: (a[1][0] - a[0][1]) / s }
    } else if a[0][0] > a[1][1] && a[0][0] > a[2][2] {
        let s = (1.0 + a[0][0] - a[1][1] - a[2][2]).sqrt() * 2.0;
        Quat { w: (a[2][1] - a[1][2]) / s, x: 0.25 * s, y: (a[0][1] + a[1][0]) / s, z: (a[0][2] + a[2][0]) / s }
    } else if a[1][1] > a[2][2] {
        let s = (1.0 + a[1][1] - a[0][0] - a[2][2]).sqrt() * 2.0;
        Quat { w: (a[0][2] - a[2][0]) / s, x: (a[0][1] + a[1][0]) / s, y: 0.25 * s, z: (a[1][2] + a[2][1]) / s }
    } else {
        let s = (1.0 + a[2][2] - a[0][0] - a[1][1]).sqrt() * 2.0;
        Quat { w: (a[1][0] - a[0][1]) / s, x: (a[0][2] + a[2][0]) / s, y: (a[1][2] + a[2][1]) / s, z: 0.25 * s }
    };
    q.normalized()
}

/// A body a chunk may touch within a frame (`World::engaged_contact_dt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContactPartner {
    Chunk(usize, usize),
    Impactor(usize),
    Ground,
}

/// The largest world.rs `contact_stiffness` of two boxes over every direction: along a
/// unit `d` a box's projected area `sum |d_k| A_k` is at most `|A|` (Cauchy-Schwarz) and
/// its half-thickness `sum |d_k| h_k` at least its smallest half-extent (`sum |d_k| >= 1`).
fn pair_stiffness_bound(a: &Material, box_a: &OBox, b: &Material, box_b: &OBox, scale: f64) -> f64 {
    let e = |m: &Material| m.youngs_modulus * scale / (1.0 - m.poisson_ratio * m.poisson_ratio);
    let area = |x: &OBox| Vec3::new(x.half.y * x.half.z, x.half.x * x.half.z, x.half.x * x.half.y).norm() * 4.0;
    let thin = |x: &OBox| x.half.x.min(x.half.y).min(x.half.z);
    area(box_a).min(area(box_b)) / (thin(box_a) / e(a) + thin(box_b) / e(b))
}

/// A chunk's pose and motion when a contact plan was made, and its budget under it: no
/// point of the chunk strays more than `budget` from its ballistic path
/// `center + velocity t + g t^2 / 2` (gravity if it `falls`) within `horizon` (the GPU's
/// travel check: deviation of the centre plus turn angle times the bounding radius).
#[derive(Clone, Copy, Debug)]
pub struct PlannedPose {
    pub center: Vec3,
    /// World rotation of the chunk's box (world.rs `chunk_box`).
    pub rotation: Mat3,
    pub budget: f64,
    pub velocity: Vec3,
    pub falls: bool,
    pub horizon: f64,
}

/// world.rs `contact_stiffness` of a chunk pair bounded over the directions it can press
/// along while the plan stands: the line between the centres follows the pair's relative
/// path `rel + dv t + dg t^2 / 2` (`dg`: gravity on one of the two only) give or take
/// both budgets, so it stays within `slack` of the path's middle `m` and within
/// `phi = asin(slack / |m|)` of its direction; each box turns by at most
/// `budget / bounding radius`, so in a box's frame the direction is within
/// `psi = phi + turn` of `m`'s, and each direction cosine within `psi` of its value
/// there (a chord is shorter than its arc). The projected area `sum |d_k| A_k` and
/// half-thickness `sum |d_k| h_k` are bounded with those cosines (and by
/// `pair_stiffness_bound`'s direction-free limits). Every direction when the slack could
/// bring the centres together.
#[allow(clippy::too_many_arguments)]
fn pair_stiffness_within(a: &Material, half_a: &Vec3, pa: &PlannedPose, b: &Material, half_b: &Vec3, pb: &PlannedPose, g: f64, scale: f64) -> f64 {
    let e = |m: &Material| m.youngs_modulus * scale / (1.0 - m.poisson_ratio * m.poisson_ratio);
    let horizon = pa.horizon.max(pb.horizon);
    let dv = pb.velocity - pa.velocity;
    let dg = if pa.falls == pb.falls { 0.0 } else { g };
    let line = pb.center - pa.center + dv * (0.5 * horizon);
    let distance = line.norm();
    let slack = pa.budget + pb.budget + 0.5 * dv.norm() * horizon + 0.5 * dg * horizon * horizon;
    let phi = if distance > slack { (slack / distance).asin() } else { std::f64::consts::PI };
    let d0 = if distance > 0.0 { line * (1.0 / distance) } else { Vec3::Z };
    let reach = |half: &Vec3, pose: &PlannedPose| -> (f64, f64) {
        let areas = [4.0 * half.y * half.z, 4.0 * half.x * half.z, 4.0 * half.x * half.y];
        let free_area = Vec3::new(areas[0], areas[1], areas[2]).norm();
        let thin = half.x.min(half.y).min(half.z);
        let psi = phi + pose.budget / half.norm();
        if psi >= std::f64::consts::FRAC_PI_2 {
            return (free_area, thin);
        }
        let mut area = 0.0;
        let mut h = 0.0;
        for k in 0..3 {
            let c0 = d0.dot(pose.rotation.col(k)).abs();
            area += (c0 + psi).min(1.0) * areas[k];
            h += (c0 - psi).max(0.0) * half[k];
        }
        (area.min(free_area), h.max(thin))
    };
    let (area_a, h_a) = reach(half_a, pa);
    let (area_b, h_b) = reach(half_b, pb);
    area_a.min(area_b) / (h_a / e(a) + h_b / e(b))
}

/// Certified upper bound of the largest eigenvalue of the mass-normalised contact
/// stiffness `M^-1/2 K M^-1/2`, from the network's scalar form `B` (`diag`: each body's
/// `sum k / m`; `edges`: each pair's `k / sqrt(m_a m_b)`). A pair's 3 x 3 blocks are
/// `k n n^T` scaled, of norm at most those entries, and for a symmetric block matrix
/// `lambda_max(A) <= rho(B)` (`x^T A x <= y^T B y` with `y_i = |x_i|`). For any positive
/// `v`, `rho(B) <= max_i (B v)_i / v_i` (Collatz-Wielandt): the first iterate (`v = 1`)
/// is the Gershgorin row bound, and power iteration sharpens it towards `rho(B)`; every
/// iterate's bound holds, the smallest is kept. A heavy impactor on many light chunks (a
/// star) is the case where rows overstate most.
pub(crate) fn contact_network_radius(diag: &[f64], edges: &[(usize, usize, f64)]) -> f64 {
    const ITERATIONS: usize = 200;
    const CONVERGED: f64 = 1e-3;
    let n = diag.len();
    if n == 0 {
        return 0.0;
    }
    let mut v = vec![1.0f64; n];
    let mut bound = f64::INFINITY;
    for _ in 0..ITERATIONS {
        let mut y: Vec<f64> = (0..n).map(|i| diag[i] * v[i]).collect();
        for &(i, j, w) in edges {
            y[i] += w * v[j];
            y[j] += w * v[i];
        }
        let (mut hi, mut lo, mut top) = (0.0f64, f64::INFINITY, 0.0f64);
        for i in 0..n {
            if y[i] > 0.0 {
                let ratio = if v[i] > 0.0 { y[i] / v[i] } else { f64::INFINITY };
                hi = hi.max(ratio);
                lo = lo.min(ratio);
            }
            top = top.max(y[i]);
        }
        bound = bound.min(hi);
        if !(top > 0.0) || hi - lo <= CONVERGED * hi {
            break;
        }
        v = y.iter().map(|x| x / top).collect();
    }
    if bound.is_finite() { bound } else { 0.0 }
}

#[cfg(test)]
mod network_tests {
    use super::contact_network_radius;

    /// A heavy body on n light ones (a ram on a wall): the bound lies between the star's
    /// exact largest eigenvalue and the Gershgorin row bound, close to the former.
    #[test]
    fn star_bound_is_certified_and_sharp() {
        let (k, mh, ml, n) = (8e9, 200.0, 2.4, 230usize);
        let mut diag = vec![n as f64 * k / mh];
        diag.extend(std::iter::repeat_n(k / ml, n));
        let edges: Vec<(usize, usize, f64)> = (1..=n).map(|j| (0, j, k / (mh * ml).sqrt())).collect();
        let (a, b) = (diag[0], diag[1]);
        let exact = 0.5 * (a + b) + (0.25 * (a - b) * (a - b) + n as f64 * k * k / (mh * ml)).sqrt();
        let rows = a + n as f64 * k / (mh * ml).sqrt();
        let bound = contact_network_radius(&diag, &edges);
        assert!(bound >= exact * (1.0 - 1e-12), "{bound} below the exact {exact}");
        assert!(bound <= exact * 1.002, "{bound} not sharp against {exact} (rows {rows})");
        assert!(rows > 5.0 * exact);
    }
}
