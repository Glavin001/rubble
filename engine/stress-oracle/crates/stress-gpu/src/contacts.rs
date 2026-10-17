//! Contact planning for a segment (`shaders/world.slang`): candidate chunk pairs,
//! impactor and ground candidates from bounds swept over the segment, the contact data
//! of every chunk, impactors, and the per-pair memory of pre-existing overlap.
//!
//! Candidates are a superset of what the reference tests in any substep of the segment:
//! each chunk gets a travel budget (twice its speed bound over the segment, plus gravity
//! and a margin), pairs within the sum of radii and budgets are candidates, and the GPU
//! stops the batch if a chunk leaves its budget, so the host re-plans from the state
//! then reached. The GPU then applies the reference's own tests to every candidate.

use std::collections::HashMap;

use stress_ref::math::{Mat3, Vec3};
use stress_ref::scene::{ImpactorShape, Scene};
use stress_ref::solver::ReferenceSolver;
use stress_ref::world::Impactor;

/// A chunk pair key as the reference keys `pair_offsets`: (structure, chunk) of the chunk
/// in the lower-index cluster first.
pub type PairKey = (usize, usize, usize, usize);

/// The sample-point offsets of a pair, a's samples then b's (offset, normal; NaN offset:
/// not in contact).
pub type PairState = Vec<[f32; 4]>;

/// Contact sample points of a chunk (contact.rs `sample_count`).
pub fn sample_count(chunk: &stress_ref::structure::ChunkData) -> usize {
    chunk.hull.as_ref().map_or(stress_ref::contact::SAMPLE_POINTS, |h| h.vertices.len() + h.faces.len())
}

pub const NAN_STATE: [f32; 4] = [f32::NAN, 0.0, 0.0, 0.0];

/// The contact plan of a segment.
#[derive(Clone, Default)]
pub struct Plan {
    /// Candidate pairs (key, gpu chunk a, gpu chunk b, reduced mass, friction).
    pub pairs: Vec<(PairKey, u32, u32, f64, f64)>,
    /// Per impactor: candidate gpu chunks in the reference's order.
    pub impactor_chunks: Vec<Vec<u32>>,
    /// Gpu chunks that may touch the ground.
    pub ground: Vec<u32>,
    /// Clusters (by index) in the contact pipeline this segment.
    pub contact_clusters: Vec<bool>,
    /// Travel budget per gpu chunk (m).
    pub budget: Vec<f64>,
    /// Gpu chunks with a pair that only their boxes rule out (the bounding spheres would
    /// not): the GPU also checks how far they turn (`travel_check`), since a box, unlike
    /// a sphere, does not bound the chunk at another orientation.
    pub rotation_checked: Vec<bool>,
    /// Pose (centre, box rotation) of every chunk in a candidate pair when planned.
    pub poses: HashMap<u32, (Vec3, stress_ref::math::Mat3)>,
    /// Each chunk's velocity when planned, and whether gravity moves it (its cluster is
    /// free): the budget bounds how far it strays from `x0 + v0 t + g t^2 / 2`.
    pub velocity: Vec<Vec3>,
    pub falls: Vec<bool>,
    /// Seconds the plan covers.
    pub horizon: f64,
    /// Per impactor: its plan (as a chunk's: its ballistic path from where it is, the
    /// budget of its deviation, checked on the GPU), None for a driven one.
    pub impactor_plan: Vec<Option<ImpactorPlan>>,
}

/// An impactor's place in a contact plan.
#[derive(Clone, Copy, Debug)]
pub struct ImpactorPlan {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: stress_ref::math::Quat,
    pub budget: f64,
}

/// What a contact plan lets touch, by (structure, chunk), and each chunk's travel budget
/// (enforced on the GPU by `travel_check`): the contacts a step bound must cover while
/// the plan stands.
#[derive(Clone, Debug, Default)]
pub struct PlanView {
    pub pairs: Vec<PairKey>,
    /// Per impactor: the chunks it may touch.
    pub impactor_chunks: Vec<Vec<(usize, usize)>>,
    pub ground: Vec<(usize, usize)>,
    pub budget: HashMap<(usize, usize), f64>,
    /// The planned pose and budget of every chunk in a pair or an impactor's candidate.
    pub poses: HashMap<(usize, usize), stress_ref::world::PlannedPose>,
    /// The planned pose and budget of every impactor the solver integrates.
    pub impactor_poses: HashMap<usize, stress_ref::world::PlannedPose>,
}

impl PlanView {
    pub fn of(plan: &Plan, order: &[(usize, usize)]) -> PlanView {
        PlanView {
            pairs: plan.pairs.iter().map(|p| p.0).collect(),
            impactor_chunks: plan.impactor_chunks.iter().map(|l| l.iter().map(|&k| order[k as usize]).collect()).collect(),
            ground: plan.ground.iter().map(|&k| order[k as usize]).collect(),
            budget: order.iter().copied().zip(plan.budget.iter().copied()).collect(),
            impactor_poses: HashMap::new(),
            poses: plan
                .poses
                .iter()
                .map(|(&k, &(center, rotation))| {
                    let k = k as usize;
                    (order[k], stress_ref::world::PlannedPose { center, rotation, budget: plan.budget[k], velocity: plan.velocity[k], falls: plan.falls[k], horizon: plan.horizon })
                })
                .collect(),
        }
    }
}

/// Plane-strain modulus (scaled) of a material, as world.rs uses it for contacts.
pub fn modulus(scene: &Scene, material: &str) -> f64 {
    let m = scene.material(material);
    m.youngs_modulus * scene.sim.stiffness_scale / (1.0 - m.poisson_ratio * m.poisson_ratio)
}

/// Candidates for the next `duration` seconds from the mirror's current state.
///
/// Each chunk is expected on its ballistic path `x0 + v0 t + g t^2 / 2` (gravity for free
/// clusters) and its budget bounds how far any of its points strays from it (the GPU's
/// `travel_check` halts the batch beyond, and the host re-plans): twice the surface
/// speed its spin gives it about its cluster's centre and its own, a fall from rest
/// (a supported chunk strays from free fall by up to that), and a margin. Pairs are
/// swept along their relative path, on which fragments flying together barely move.
pub fn plan(scene: &Scene, m: &ReferenceSolver, impactors: &[Impactor], chunk_order: &[(usize, usize)], duration: f64) -> Plan {
    let gravity = m.config.gravity;
    let g = gravity.norm();
    let n = chunk_order.len();
    let mut gpu_index: HashMap<(usize, usize), u32> = HashMap::new();
    for (k, &sc) in chunk_order.iter().enumerate() {
        gpu_index.insert(sc, k as u32);
    }
    // Bounds, motion and deviation budgets of every active chunk.
    let mut pos = vec![Vec3::ZERO; n];
    let mut radius = vec![0.0; n];
    let mut budget = vec![0.0; n];
    let mut velocity = vec![Vec3::ZERO; n];
    let mut falls = vec![false; n];
    for (k, &(s, c)) in chunk_order.iter().enumerate() {
        let r = m.structures[s].chunks[c].half_extents.norm();
        let (v, w) = m.chunk_velocity(s, c);
        let cl = &m.clusters[m.chunks[s][c].cluster];
        pos[k] = m.chunk_position(s, c);
        let arm = (pos[k] - cl.com_world()).norm();
        radius[k] = r;
        velocity[k] = v;
        falls[k] = !cl.anchored;
        budget[k] = 2.0 * w.norm() * (arm + r) * duration + g * duration * duration + 0.05 * r + 1e-3;
    }
    let mut plan = Plan {
        contact_clusters: vec![false; m.clusters.len()],
        budget,
        rotation_checked: vec![false; n],
        velocity: velocity.clone(),
        falls: falls.clone(),
        horizon: duration,
        ..Default::default()
    };
    // Each chunk's box (world.rs `chunk_box`) grown by its budget on every side: it holds
    // the chunk anywhere within the budget of its path at its current orientation (the
    // budget's ball fits in the cube of the same half-size).
    let mut grown: Vec<Option<stress_ref::contact::OBox>> = vec![None; n];
    let mut grown_box = |k: usize, grow: f64| -> stress_ref::contact::OBox {
        let mut b = grown[k]
            .get_or_insert_with(|| {
                let (s, c) = chunk_order[k];
                let ch = &m.structures[s].chunks[c];
                let st = &m.chunks[s][c];
                let hidden = stress_ref::math::Quat::from_axis_angle(st.th, st.th.norm()).to_mat3();
                let rotation = m.clusters[st.cluster].rotation() * hidden * ch.rotation;
                stress_ref::contact::OBox { center: pos[k], rotation, half: ch.half_extents, hull: None, parts: None }
            })
            .clone();
        b.half += Vec3::splat(grow);
        b
    };
    // Chunk pairs of different clusters (world.rs `apply_contacts`, any substep).
    // A quiet cluster left out of the GPU tables (`GpuSolver::quiet_clusters`) has no
    // GPU chunks; it is engine-driven, so every pair with it is skipped below anyway.
    let by_cluster: Vec<Vec<u32>> = m
        .clusters
        .iter()
        .map(|cl| {
            let ks: Vec<u32> = cl.chunks.iter().filter_map(|&c| gpu_index.get(&(cl.structure, c)).copied()).collect();
            assert!(ks.len() == cl.chunks.len() || cl.driven, "cluster {} has chunks missing from the GPU tables but is not engine-driven", cl.id);
            ks
        })
        .collect();
    let mut sorted_by_cluster: Vec<Vec<u32>> = by_cluster.clone();
    for (ci, list) in sorted_by_cluster.iter_mut().enumerate() {
        // The reference iterates a cluster's chunks in active order (structure, chunk).
        let s = m.clusters[ci].structure;
        list.sort_by_key(|&k| (s, chunk_order[k as usize].1));
    }
    let mu_of = |s: usize, c: usize| scene.material(&m.structures[s].chunks[c].material).friction;
    // A pair needs both clusters owned (not engine-driven): only those are visited, in
    // the order of the loop over every pair.
    let owned: Vec<usize> = (0..m.clusters.len()).filter(|&c| !m.clusters[c].driven).collect();
    // Per owned cluster, for a conservative whole-cluster test: a centre, its chunks'
    // largest distance from it, largest radius + budget, and largest speed.
    let reach: Vec<(Vec3, f64, f64, f64)> = owned
        .iter()
        .map(|&c| {
            let ks = &sorted_by_cluster[c];
            let center = ks.first().map_or(Vec3::ZERO, |&k| pos[k as usize]);
            ks.iter().fold((center, 0.0f64, 0.0f64, 0.0f64), |(o, s, rb, v), &k| {
                let k = k as usize;
                (o, s.max((pos[k] - o).norm()), rb.max(radius[k] + plan.budget[k]), v.max(velocity[k].norm()))
            })
        })
        .collect();
    for (i, &ca) in owned.iter().enumerate() {
        for (j, &cb) in owned.iter().enumerate().skip(i + 1) {
            let (cla, clb) = (&m.clusters[ca], &m.clusters[cb]);
            if (cla.anchored && clb.anchored) || cla.driven || clb.driven {
                continue;
            }
            // Every chunk pair's `middle` is at least the centres' distance less both
            // spreads and half the largest relative travel, and every pair's limit at
            // most both largest radii + budgets and the largest sweep (`falls` is the
            // cluster's): beyond that, with a margin for rounding, no chunk pair passes.
            {
                let ((oa, sa, rba, va), (ob, sb, rbb, vb)) = (reach[i], reach[j]);
                let dg = if cla.anchored == clb.anchored { 0.0 } else { g };
                let travel = (va + vb) * duration;
                let near = (ob - oa).norm() - sa - sb - 0.5 * travel;
                let limit = rba + rbb + 0.5 * travel + 0.5 * dg * duration * duration;
                if near > limit + 1e-9 * (near.abs() + limit + sa + sb + 1.0) {
                    continue;
                }
            }
            for &ka in &sorted_by_cluster[ca] {
                for &kb in &sorted_by_cluster[cb] {
                    let (a, b) = (ka as usize, kb as usize);
                    // b relative to a over the plan: from `rel` along `dv t + dg t^2 / 2`,
                    // within `sweep` of the middle of its path, give or take the budgets.
                    let dv = velocity[b] - velocity[a];
                    let dg = if falls[b] == falls[a] { 0.0 } else { g };
                    let middle = pos[b] - pos[a] + dv * (0.5 * duration);
                    let sweep = 0.5 * dv.norm() * duration + 0.5 * dg * duration * duration;
                    if middle.norm() > radius[a] + radius[b] + plan.budget[a] + plan.budget[b] + sweep {
                        continue;
                    }
                    // Every pair chunk's turn is bounded on the GPU: a ruled-out pair
                    // relies on it, and so does the step bound of a kept one (its
                    // stiffness over the directions it can press along).
                    plan.rotation_checked[a] = true;
                    plan.rotation_checked[b] = true;
                    let box_a = grown_box(a, plan.budget[a]);
                    let mut box_b = grown_box(b, plan.budget[b] + sweep);
                    box_b.center = pos[a] + middle;
                    if !box_a.may_overlap(&box_b) {
                        continue;
                    }
                    plan.poses.insert(ka, (pos[a], box_a.rotation));
                    plan.poses.insert(kb, (pos[b], box_b.rotation));
                    let (sa, chunk_a) = chunk_order[a];
                    let (sb, chunk_b) = chunk_order[b];
                    let (ma, mb) = (m.structures[sa].chunks[chunk_a].mass, m.structures[sb].chunks[chunk_b].mass);
                    let mu = scene.sim.contact_friction.unwrap_or(mu_of(sa, chunk_a).min(mu_of(sb, chunk_b)));
                    plan.pairs.push(((sa, chunk_a, sb, chunk_b), ka, kb, ma * mb / (ma + mb), mu));
                    plan.contact_clusters[ca] = true;
                    plan.contact_clusters[cb] = true;
                }
            }
        }
    }
    // Impactors against chunks, in active order.
    let mut active_order: Vec<u32> = (0..n as u32).collect();
    active_order.sort_by_key(|&k| chunk_order[k as usize]);
    for imp in impactors {
        let mut list = Vec::new();
        if !imp.driven {
            let reach = match imp.shape {
                ImpactorShape::Sphere { radius } => radius * 3f64.sqrt(),
                ImpactorShape::Box { half_extents } => Vec3::from_array(half_extents).norm(),
            };
            // Its travel from where it is (about a free fall from rest), budgeted twice its
            // surface speed; the GPU's `impactor_travel_check` re-plans beyond.
            let imp_budget = 2.0 * (imp.velocity.norm() + imp.angular_velocity.norm() * reach) * duration + g * duration * duration + 1e-3;
            plan.impactor_plan.push(Some(ImpactorPlan { position: imp.pose.position, velocity: Vec3::ZERO, rotation: imp.pose.rotation, budget: imp_budget }));
            for &k in &active_order {
                let (s, c) = chunk_order[k as usize];
                let ci = m.chunks[s][c].cluster;
                if m.clusters[ci].driven {
                    continue;
                }
                let kk = k as usize;
                // The chunk's reach: its path's farthest point plus its budget.
                let chunk_reach = velocity[kk].norm() * duration + if falls[kk] { 0.5 * g * duration * duration } else { 0.0 } + plan.budget[kk];
                if (pos[kk] - imp.pose.position).norm() <= reach + radius[kk] + chunk_reach + imp_budget {
                    list.push(k);
                    plan.contact_clusters[ci] = true;
                }
            }
        } else {
            plan.impactor_plan.push(None);
        }
        plan.impactor_chunks.push(list);
    }
    // Ground.
    if let Some(ground) = &scene.ground {
        for &k in &active_order {
            let (s, c) = chunk_order[k as usize];
            let cl = &m.clusters[m.chunks[s][c].cluster];
            if cl.anchored || cl.driven {
                continue;
            }
            let kk = k as usize;
            // The lowest its path gets within the plan, give or take its budget.
            let (z0, vz, az) = (pos[kk].z, velocity[kk].z, if falls[kk] { gravity.z } else { 0.0 });
            let lowest = [0.0, duration, if az != 0.0 { (-vz / az).clamp(0.0, duration) } else { 0.0 }].iter().map(|&t| z0 + vz * t + 0.5 * az * t * t).fold(f64::INFINITY, f64::min);
            if lowest - radius[kk] - plan.budget[kk] <= ground.height {
                plan.ground.push(k);
                plan.contact_clusters[m.chunks[s][c].cluster] = true;
            }
        }
    }
    plan
}

/// The impactor's inertia and inverse in its body frame.
pub fn impactor_inertia(imp: &Impactor) -> (Mat3, Mat3) {
    (imp.inertia, imp.inertia.inverse().unwrap_or(Mat3::ZERO))
}
