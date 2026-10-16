//! The reference stress solver.
//!
//! **State.** Every structure is a set of chunks connected by bonds. Chunks are
//! grouped into *clusters*: connected components of intact bonds, each one rigid body
//! for the physics engine. Every chunk carries a hidden 6-DOF displacement
//! `(u, theta)` and its rate `(v, w)` in its cluster's body frame (floating frame of
//! reference). Rendering stays rigid; the displacements exist only to compute stress.
//!
//! **Loads.** The chunk loads of a free cluster are the external forces minus the
//! inertial forces of the cluster's rigid motion (linear, angular and centrifugal
//! terms), so the deformation problem is self-equilibrated. Anchored clusters (with
//! supported chunks) live in the static world frame and report support reactions.
//!
//! **Integration.** Explicit central differences with light stiffness-proportional
//! damping; the substep is a safety fraction of the smallest per-chunk damped
//! central-difference limit, from Gershgorin bounds on each chunk's stiffness and
//! dashpot rows. Settled clusters may instead be solved quasi-statically
//! (`statics.rs`) and idle clusters sleep (`SolveMode::Adaptive`).
//!
//! **Fracture.** Bonds are evaluated every substep (`joint.rs`). When a bond stops
//! connecting its chunks, the cluster's connectivity is recomputed and it splits into
//! child clusters that inherit each chunk's exact velocity (`v + w x r` of the parent
//! plus the hidden deformation velocity), conserving linear and angular momentum.

use crate::bond::{BondGeometry, BondStiffness, ChunkTurn, Local6};
use crate::joint::{fatigue_factor, FailureMode, JointModel, JointResponse, JointState, JointStrength, RebarParams, StressMeasures};

/// A chunk's hidden rotation vector after it turns by the spatial increment `dw`: the
/// exact composition `exp(dw) exp(th)` with `corotational_bonds` (the bonds' moments are
/// conjugate to it), else the small-rotation sum.
fn turn(th: Vec3, dw: Vec3, exact: bool) -> Vec3 {
    if exact {
        (crate::math::Quat::from_rotation_vector(dw) * crate::math::Quat::from_rotation_vector(th)).to_rotation_vector()
    } else {
        th + dw
    }
}

/// A bond's response over a substep: the joint law's, the generalised force with its
/// dashpot, the dashpot's dissipation and the chunk loads `(f_a, m_a, f_b, m_b)`.
type BondResponse = (JointResponse, Local6, f64, (Vec3, Vec3, Vec3, Vec3));
use crate::math::{Mat3, Pose, Quat, Vec3};
use crate::scene::{Features, Scene, SolveMode, Support};
use crate::structure::{BondData, Structure};

/// Solver settings (from the scene's `sim` block).
/// Residual static-fatigue strength factor at which a bond "creaks" ([`SolverEvent::Creaked`]).
pub const CREAK_STRENGTH: f64 = 0.99;

#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub gravity: Vec3,
    pub fracture: bool,
    pub mode: SolveMode,
    /// Integrate cluster rigid motion here (standalone) instead of taking it from an engine.
    pub integrate_rigid: bool,
    pub courant_safety: f64,
    pub refine_utilization: Option<f64>,
    /// Adaptive mode: time a cluster stays explicit after its last dynamic load.
    pub active_time: f64,
    /// Adaptive mode: relative load change that wakes a settled cluster.
    pub wake_threshold: f64,
    pub features: Features,
    pub methods: crate::scene::Methods,
}

impl SolverConfig {
    pub fn from_scene(scene: &Scene) -> SolverConfig {
        SolverConfig {
            gravity: Vec3::from_array(scene.gravity),
            fracture: scene.sim.fracture,
            mode: scene.sim.solve_mode,
            integrate_rigid: true,
            // The stress bound's fraction (`stable_dt`, mass scaling); contacts take
            // `sim.courant_safety` (world.rs).
            courant_safety: scene.sim.stress_courant_safety.unwrap_or(scene.sim.courant_safety),
            refine_utilization: scene.sim.refine_utilization,
            active_time: 0.5,
            wake_threshold: 0.02,
            features: scene.sim.features,
            methods: scene.sim.methods,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    /// Explicit dynamic substeps.
    Active,
    /// Held at quasi-static equilibrium; re-solved when its loads change.
    Settled,
}

/// A rigid cluster of chunks.
#[derive(Clone, Debug)]
pub struct Cluster {
    /// Unique, never reused.
    pub id: u64,
    pub parent: Option<u64>,
    pub structure: usize,
    /// Active chunk indices (into the structure).
    pub chunks: Vec<usize>,
    /// Live runtime bonds with both chunks in this cluster (intact or cracked-contact).
    pub bonds: Vec<usize>,
    /// Holds supported chunks: the frame is the static world frame.
    pub anchored: bool,
    /// Pose of the structure's body frame for this cluster.
    pub pose: Pose,
    /// Centre of mass in the body frame (undeformed).
    pub com: Vec3,
    pub mass: f64,
    /// Inertia about the centre of mass, body frame.
    pub inertia: Mat3,
    /// Velocity of the centre of mass and angular velocity (world).
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub activity: Activity,
    /// Implicit mode with `implicit_contact_windows` or `implicit_failure_windows`:
    /// integrated explicitly this frame.
    pub explicit_window: bool,
    /// `implicit_failure_windows`: the highest failure index of an intact bond at the
    /// last window decision.
    pub window_utilization: f64,
    pub active_timer: f64,
    /// Rigid motion owned by an external engine this frame (see `engine.rs`): the
    /// solver neither integrates it nor folds hidden motion into it.
    pub driven: bool,
    /// External chunk loads at the last quasi-static solve (for change detection).
    pub settled_load_norm: f64,
}

impl Cluster {
    pub fn com_world(&self) -> Vec3 {
        self.pose.transform_point(self.com)
    }
    pub fn rotation(&self) -> Mat3 {
        self.pose.rotation.to_mat3()
    }
    pub fn world_inertia(&self) -> Mat3 {
        let r = self.rotation();
        r * self.inertia * r.transpose()
    }
}

/// Hidden deformation state of a chunk (body frame of its cluster).
#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkState {
    pub u: Vec3,
    pub th: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    /// Multiplier on the chunk's deformation inertia (selective mass scaling; 1 = none).
    pub inertia_scale: f64,
    /// `SolveMode::Implicit` only: hidden accelerations of the last implicit step, and
    /// the hidden state at its end (the next step starts from it; between steps `u`,
    /// `th`, `v`, `w` hold the predictor, see `implicit.rs`).
    pub a: Vec3,
    pub alpha: Vec3,
    pub step_start: [Vec3; 4],
    /// Represented in the simulation (not removed, and on the current refinement level).
    pub active: bool,
    pub removed: bool,
    /// Index into `ReferenceSolver::clusters` (valid when active).
    pub cluster: usize,
    /// Support reaction (body frame force and moment) at the last substep.
    pub reaction: (Vec3, Vec3),
}

/// A bond as simulated: a static bond, possibly re-attached to coarser chunks.
#[derive(Clone, Debug)]
pub struct RtBond {
    /// Static bond in the structure this was created from.
    pub source: usize,
    pub geometry: BondGeometry,
    pub stiffness: BondStiffness,
    pub strength: JointStrength,
    pub rebar: Option<RebarParams>,
    pub weibull: f64,
    pub damping: Local6,
    pub level: u32,
    pub alive: bool,
    pub joint: JointState,
    /// Last actual generalised force (incl. damping) and effective stress measures.
    pub force: Local6,
    /// Its elastic part (no damping): the joint's spring forces.
    pub elastic: Local6,
    pub measures: StressMeasures,
    pub stored: f64,
}

impl RtBond {
    fn from_static(index: usize, b: &BondData) -> RtBond {
        RtBond {
            source: index,
            geometry: b.geometry.clone(),
            stiffness: b.stiffness,
            strength: b.strength.clone(),
            rebar: b.rebar,
            weibull: b.weibull,
            damping: b.damping,
            level: b.level,
            alive: true,
            joint: JointState::new(),
            force: Local6::ZERO,
            elastic: Local6::ZERO,
            measures: StressMeasures::default(),
            stored: 0.0,
        }
    }

    pub fn model(&self) -> JointModel<'_> {
        JointModel {
            geometry: &self.geometry,
            stiffness: &self.stiffness,
            strength: &self.strength,
            rebar: self.rebar.as_ref(),
            weibull: self.weibull,
        }
    }

    pub fn connected(&self) -> bool {
        self.alive && self.joint.connected(self.rebar.is_some())
    }
}

/// Things that happened, for rendering (crack decals), audio and VFX.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SolverEvent {
    /// A bond first took damage (crack initiation).
    Cracked { time: f64, structure: usize, bond: usize, mode: Option<FailureMode>, position: [f64; 3] },
    /// Static fatigue cost a bond its first percent of strength ([`CREAK_STRENGTH`]):
    /// the onset of measurable strength loss, heard as creaking (acoustic emission).
    Creaked { time: f64, structure: usize, bond: usize, position: [f64; 3] },
    /// A bond stopped connecting its chunks.
    Broken { time: f64, structure: usize, bond: usize, mode: Option<FailureMode>, position: [f64; 3], normal: [f64; 3], dissipated: f64 },
    Split { time: f64, parent: u64, children: Vec<u64> },
    Refined { time: f64, structure: usize, chunk: usize },
}

/// A joint broken by a split, as the world needs it to hand its compression over to the
/// contact of its two chunks.
#[derive(Clone, Copy, Debug)]
pub struct Handover {
    pub structure: usize,
    pub a: usize,
    pub b: usize,
    /// The joint's elastic compressive normal force at the split (N, >= 0).
    pub compression: f64,
    /// The joint's stored elastic energy at the split (J).
    pub stored: f64,
    /// The joint's elastic force on chunk `a` and its couple (world frame), whose
    /// shear and torsion the contact's friction takes over.
    pub force: Vec3,
    pub couple: Vec3,
}

/// Cumulative energy terms (J). See `energy_balance` in `world.rs` for the closed form.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct EnergyLedger {
    /// Damage, friction and rebar plasticity in bonds.
    pub bond_dissipation: f64,
    /// Bond dashpots.
    pub damping_dissipation: f64,
    /// Work done on chunks by external (non-gravity) loads.
    pub external_work: f64,
    /// Stored energy released beyond the softening law by bonds snapping within one
    /// substep (time-discretisation; shrinks with the substep).
    pub softening_overshoot: f64,
    /// Energy stored in bonds between chunks that a split separates; the pieces keep
    /// their overlap instead of being pushed apart (see `contact.rs`).
    pub split_release: f64,
    /// Mechanical energy the model removes by putting clusters into static equilibrium
    /// (adaptive settling and settled fatigue, the quasi-static solve) or by refining
    /// them: hidden motion and the elastic energy the equilibrium does not hold, net of
    /// what those steps book as bond dissipation. Without it the balance of those modes
    /// does not close.
    pub static_discard: f64,
}

/// Force-replacement load of a removed member (alternate-path procedure).
#[derive(Clone, Debug)]
pub struct ReplacementLoad {
    pub structure: usize,
    pub chunk: usize,
    /// Body-frame force and moment at `start`.
    pub force: Vec3,
    pub moment: Vec3,
    pub start: f64,
    pub duration: f64,
}

impl ReplacementLoad {
    pub fn factor(&self, t: f64) -> f64 {
        if self.duration <= 0.0 {
            0.0
        } else {
            (1.0 - (t - self.start) / self.duration).clamp(0.0, 1.0)
        }
    }
}

/// External loads on chunks for one substep (world frame; torque about the chunk centre).
#[derive(Clone, Debug, Default)]
pub struct ChunkLoads {
    pub force: Vec<Vec<Vec3>>,
    pub torque: Vec<Vec<Vec3>>,
}

impl ChunkLoads {
    pub fn new(solver: &ReferenceSolver) -> ChunkLoads {
        let mut l = ChunkLoads::default();
        l.resize(solver);
        l
    }
    pub fn resize(&mut self, solver: &ReferenceSolver) {
        self.force = solver.structures.iter().map(|s| vec![Vec3::ZERO; s.chunks.len()]).collect();
        self.torque = self.force.clone();
    }
    /// Resize (zeroed) only if the solver's chunk counts differ; keeps the values otherwise.
    pub fn ensure_shape(&mut self, solver: &ReferenceSolver) {
        let same = self.force.len() == solver.structures.len()
            && self.force.iter().zip(&solver.structures).all(|(f, s)| f.len() == s.chunks.len());
        if !same {
            self.resize(solver);
        }
    }
    pub fn clear(&mut self) {
        self.force.iter_mut().flatten().for_each(|f| *f = Vec3::ZERO);
        self.torque.iter_mut().flatten().for_each(|f| *f = Vec3::ZERO);
    }
    /// `self += other * k`.
    pub fn add_scaled(&mut self, other: &ChunkLoads, k: f64) {
        for (a, b) in self.force.iter_mut().flatten().zip(other.force.iter().flatten()) {
            *a += *b * k;
        }
        for (a, b) in self.torque.iter_mut().flatten().zip(other.torque.iter().flatten()) {
            *a += *b * k;
        }
    }
    pub fn scale(&mut self, k: f64) {
        self.force.iter_mut().flatten().for_each(|f| *f *= k);
        self.torque.iter_mut().flatten().for_each(|f| *f *= k);
    }
    /// Add a world force applied at world point `at` on a chunk centred at `center`.
    pub fn add_at(&mut self, structure: usize, chunk: usize, force: Vec3, at: Vec3, center: Vec3) {
        self.force[structure][chunk] += force;
        self.torque[structure][chunk] += (at - center).cross(force);
    }
}

#[derive(Clone)]
pub struct ReferenceSolver {
    pub config: SolverConfig,
    /// Shared between clones (snapshots copy the mirror often); written through
    /// `Arc::make_mut`, which copies a structure only while it is shared.
    pub structures: Vec<std::sync::Arc<Structure>>,
    pub chunks: Vec<Vec<ChunkState>>,
    pub bonds: Vec<Vec<RtBond>>,
    /// Runtime bonds touching each chunk.
    pub chunk_bonds: Vec<Vec<Vec<usize>>>,
    pub clusters: Vec<Cluster>,
    pub time: f64,
    pub energy: EnergyLedger,
    pub events: Vec<SolverEvent>,
    pub replacements: Vec<ReplacementLoad>,
    /// Joints broken by splits since the world last took them (`sim.methods.layer_contact`):
    /// their compression is handed over to the chunks' contact (`World`).
    pub handovers: Vec<Handover>,
    next_cluster_id: u64,
    /// Scratch: internal forces per chunk.
    f_int: Vec<Vec<(Vec3, Vec3)>>,
    /// Set when a bond disconnects during a substep.
    pending_split: Vec<usize>,
    /// Substep count, for diagnostics.
    pub substeps: u64,
    /// Largest bond failure index seen so far (meaningful with fracture disabled too).
    pub max_utilization: f64,
    /// `sim.methods.true_step`: each structure's stable stress substep (before the safety
    /// factor) from its bond network as last refreshed.
    true_steps: Vec<f64>,
    /// `corotational_bonds`: every chunk's finite rotation at the current substep's
    /// start, shared by its bonds.
    turns: Vec<Vec<ChunkTurn>>,
    /// `rayleigh_damping`: the mass-proportional coefficient per structure (1/s).
    pub mass_damping: Vec<f64>,
    /// Each cluster's Gershgorin step bound, kept while its bonds, chunks and their
    /// inertia scales are unchanged (`stable_dt`; the bound uses static stiffness and
    /// damping, so nothing else moves it).
    step_bounds: StepBounds,
    /// `sim.methods.multirate`: the bonds' rates and schedule in the current frame.
    multirate: MultirateWindow,
    /// Bond evaluations so far (diagnostics: the work that `multirate` saves).
    pub bond_evaluations: u64,
}

/// `stable_dt`'s per-cluster cache: (bonds, chunks, inertia-scale bits, bound) by cluster id.
#[derive(Default)]
struct StepBounds(std::sync::Mutex<std::collections::HashMap<u64, (Vec<usize>, Vec<usize>, Vec<u64>, f64)>>);

impl Clone for StepBounds {
    fn clone(&self) -> StepBounds {
        StepBounds(std::sync::Mutex::new(self.0.lock().unwrap().clone()))
    }
}

impl StepBounds {
    fn clear(&mut self) {
        self.0.get_mut().unwrap().clear();
    }
}

/// `sim.methods.multirate`: a window of fine substeps (one frame) in which each bond is
/// evaluated at its own rate. Closed (`ticks` 0) or used up, every bond is evaluated at
/// every substep with weight 1, which is the single-rate integrator.
#[derive(Clone, Debug, Default)]
struct MultirateWindow {
    /// Substeps in the window, and how many have been taken.
    ticks: u32,
    tick: u32,
    /// Per structure and runtime bond: substeps per evaluation (a power of two), the
    /// substep of its next evaluation and of its last one (`u32::MAX`: none yet).
    rate: Vec<Vec<u32>>,
    next: Vec<Vec<u32>>,
    last: Vec<Vec<u32>>,
}

impl MultirateWindow {
    fn open(&self) -> bool {
        self.tick < self.ticks
    }
}

/// The largest power of two `r` (at most `cap`) with `r dt <= limit`, at least 1.
fn rate_within(limit: f64, dt: f64, cap: u32) -> u32 {
    let mut r = 1u32;
    while r < cap && 2.0 * r as f64 * dt <= limit {
        r *= 2;
    }
    r
}

impl ReferenceSolver {
    pub fn new(scene: &Scene) -> ReferenceSolver {
        let config = SolverConfig::from_scene(scene);
        let structures: Vec<std::sync::Arc<Structure>> = (0..scene.bodies.len()).map(|i| std::sync::Arc::new(Structure::from_scene(scene, i))).collect();
        let mut solver = ReferenceSolver {
            config,
            chunks: Vec::new(),
            bonds: Vec::new(),
            chunk_bonds: Vec::new(),
            clusters: Vec::new(),
            time: 0.0,
            energy: EnergyLedger::default(),
            events: Vec::new(),
            replacements: Vec::new(),
            handovers: Vec::new(),
            next_cluster_id: 0,
            f_int: Vec::new(),
            pending_split: Vec::new(),
            substeps: 0,
            max_utilization: 0.0,
            true_steps: Vec::new(),
            turns: Vec::new(),
            mass_damping: Vec::new(),
            step_bounds: StepBounds::default(),
            multirate: MultirateWindow::default(),
            bond_evaluations: 0,
            structures,
        };
        for si in 0..solver.structures.len() {
            let s = &solver.structures[si];
            let chunks: Vec<ChunkState> = s
                .chunks
                .iter()
                .map(|c| ChunkState { active: c.level == 0, inertia_scale: 1.0, ..Default::default() })
                .collect();
            let bonds: Vec<RtBond> =
                s.bonds.iter().enumerate().filter(|(_, b)| b.level == 0).map(|(i, b)| RtBond::from_static(i, b)).collect();
            solver.chunks.push(chunks);
            solver.bonds.push(bonds);
            solver.chunk_bonds.push(Vec::new());
            solver.f_int.push(vec![(Vec3::ZERO, Vec3::ZERO); s.chunks.len()]);
            solver.rebuild_adjacency(si);
            let s = &solver.structures[si];
            let pose = Pose::new(s.initial_position, s.initial_rotation);
            let (v, w) = (s.initial_velocity, s.initial_angular_velocity);
            let active: Vec<usize> = (0..s.chunks.len()).filter(|&c| solver.chunks[si][c].active).collect();
            for comp in solver.components(si, &active) {
                solver.add_cluster(si, comp, None, pose, |_| (v, w));
            }
        }
        solver
    }

    pub(crate) fn rebuild_adjacency(&mut self, s: usize) {
        let mut adj = vec![Vec::new(); self.structures[s].chunks.len()];
        for (i, b) in self.bonds[s].iter().enumerate() {
            if b.alive {
                adj[b.geometry.a].push(i);
                adj[b.geometry.b].push(i);
            }
        }
        self.chunk_bonds[s] = adj;
    }

    /// Connected components of `chunks` over connected bonds (deterministic order).
    fn components(&self, s: usize, chunks: &[usize]) -> Vec<Vec<usize>> {
        let n = self.structures[s].chunks.len();
        let mut member = vec![false; n];
        chunks.iter().for_each(|&c| member[c] = true);
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for &start in chunks {
            if seen[start] {
                continue;
            }
            let mut comp = vec![start];
            seen[start] = true;
            let mut head = 0;
            while head < comp.len() {
                let c = comp[head];
                head += 1;
                for &bi in &self.chunk_bonds[s][c] {
                    let b = &self.bonds[s][bi];
                    if !b.connected() {
                        continue;
                    }
                    let o = if b.geometry.a == c { b.geometry.b } else { b.geometry.a };
                    if member[o] && !seen[o] {
                        seen[o] = true;
                        comp.push(o);
                    }
                }
            }
            comp.sort_unstable();
            out.push(comp);
        }
        out
    }

    /// Create a cluster from chunks; `velocity_of(com_world)` supplies its rigid motion.
    fn add_cluster(
        &mut self,
        s: usize,
        chunks: Vec<usize>,
        parent: Option<u64>,
        pose: Pose,
        velocity_of: impl Fn(Vec3) -> (Vec3, Vec3),
    ) -> usize {
        let st = &self.structures[s];
        let mass: f64 = chunks.iter().map(|&c| st.chunks[c].mass).sum();
        let com = chunks.iter().map(|&c| st.chunks[c].center * st.chunks[c].mass).fold(Vec3::ZERO, |a, b| a + b) / mass;
        let mut inertia = Mat3::ZERO;
        for &c in &chunks {
            let ch = &st.chunks[c];
            let r = ch.center - com;
            inertia += ch.inertia + (Mat3::IDENTITY * r.norm2() - Mat3::outer(r, r)) * ch.mass;
        }
        let anchored = chunks.iter().any(|&c| st.chunks[c].support != Support::None);
        let (velocity, angular_velocity) =
            if anchored { (Vec3::ZERO, Vec3::ZERO) } else { velocity_of(pose.transform_point(com)) };
        let idx = self.clusters.len();
        for &c in &chunks {
            self.chunks[s][c].cluster = idx;
        }
        let set: std::collections::HashSet<usize> = chunks.iter().copied().collect();
        let mut bonds: Vec<usize> = chunks
            .iter()
            .flat_map(|&c| self.chunk_bonds[s][c].iter().copied())
            .filter(|&b| {
                let g = &self.bonds[s][b].geometry;
                set.contains(&g.a) && set.contains(&g.b)
            })
            .collect();
        bonds.sort_unstable();
        bonds.dedup();
        self.clusters.push(Cluster {
            id: self.next_cluster_id,
            parent,
            structure: s,
            chunks,
            bonds,
            anchored,
            pose,
            com,
            mass,
            inertia,
            velocity,
            angular_velocity,
            activity: Activity::Active,
            explicit_window: false,
            window_utilization: 0.0,
            active_timer: self.config.active_time,
            driven: false,
            settled_load_norm: 0.0,
        });
        self.next_cluster_id += 1;
        idx
    }

    /// Recompute a cluster's chunk/bond lists and mass properties after its chunks
    /// changed in place (refinement); its rigid motion is kept.
    pub(crate) fn refresh_cluster_membership(&mut self, ci: usize) {
        let s = self.clusters[ci].structure;
        let chunks: Vec<usize> =
            (0..self.structures[s].chunks.len()).filter(|&c| self.chunks[s][c].active && self.chunks[s][c].cluster == ci).collect();
        let st = &self.structures[s];
        let mass: f64 = chunks.iter().map(|&c| st.chunks[c].mass).sum();
        let com = chunks.iter().map(|&c| st.chunks[c].center * st.chunks[c].mass).fold(Vec3::ZERO, |a, b| a + b) / mass;
        let mut inertia = Mat3::ZERO;
        for &c in &chunks {
            let ch = &st.chunks[c];
            let r = ch.center - com;
            inertia += ch.inertia + (Mat3::IDENTITY * r.norm2() - Mat3::outer(r, r)) * ch.mass;
        }
        let set: std::collections::HashSet<usize> = chunks.iter().copied().collect();
        let mut bonds: Vec<usize> = chunks
            .iter()
            .flat_map(|&c| self.chunk_bonds[s][c].iter().copied())
            .filter(|&b| {
                let g = &self.bonds[s][b].geometry;
                set.contains(&g.a) && set.contains(&g.b)
            })
            .collect();
        bonds.sort_unstable();
        bonds.dedup();
        let cl = &mut self.clusters[ci];
        cl.chunks = chunks;
        cl.bonds = bonds;
        cl.mass = mass;
        cl.com = com;
        cl.inertia = inertia;
    }

    // ------------------------------------------------------------------ kinematics

    /// World position of a chunk centre (including its hidden displacement).
    pub fn chunk_position(&self, s: usize, c: usize) -> Vec3 {
        let cl = &self.clusters[self.chunks[s][c].cluster];
        cl.pose.transform_point(self.structures[s].chunks[c].center + self.chunks[s][c].u)
    }

    /// World velocity and angular velocity of a chunk.
    pub fn chunk_velocity(&self, s: usize, c: usize) -> (Vec3, Vec3) {
        let st = &self.chunks[s][c];
        let cl = &self.clusters[st.cluster];
        let r = cl.pose.transform_vector(self.structures[s].chunks[c].center + st.u - cl.com);
        (
            cl.velocity + cl.angular_velocity.cross(r) + cl.pose.transform_vector(st.v),
            cl.angular_velocity + cl.pose.transform_vector(st.w),
        )
    }

    /// Velocity of a world point rigidly attached to a chunk.
    pub fn point_velocity(&self, s: usize, c: usize, p: Vec3) -> Vec3 {
        let (v, w) = self.chunk_velocity(s, c);
        v + w.cross(p - self.chunk_position(s, c))
    }

    /// Total kinetic energy of all active chunks (world frame).
    pub fn kinetic_energy(&self) -> f64 {
        let mut e = 0.0;
        for (s, st) in self.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                if !self.chunks[s][c].active {
                    continue;
                }
                let (v, w) = self.chunk_velocity(s, c);
                let cl = &self.clusters[self.chunks[s][c].cluster];
                let r = cl.rotation();
                let iw = r * ch.inertia * r.transpose();
                e += 0.5 * ch.mass * v.norm2() + 0.5 * w.dot(iw * w);
            }
        }
        e
    }

    /// Recoverable elastic energy stored in bonds at the last evaluation (under
    /// `multirate`, a bond waiting for its next evaluation at the current state).
    pub fn elastic_energy(&self) -> f64 {
        if self.multirate.last.is_empty() {
            return self.bonds.iter().flatten().filter(|b| b.alive).map(|b| b.stored).sum();
        }
        let mut e = 0.0;
        for s in 0..self.bonds.len() {
            for bi in 0..self.bonds[s].len() {
                let b = &self.bonds[s][bi];
                if b.alive {
                    e += if self.bond_is_stale(s, bi) { self.bond_stored_now(s, bi) } else { b.stored };
                }
            }
        }
        e
    }

    /// Gravitational potential energy `-m g . x` of all active chunks.
    pub fn gravity_potential(&self) -> f64 {
        let g = self.config.gravity;
        let mut e = 0.0;
        for (s, st) in self.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                if self.chunks[s][c].active {
                    e -= ch.mass * g.dot(self.chunk_position(s, c));
                }
            }
        }
        e
    }

    // ------------------------------------------------------------------ timestep

    /// Gershgorin row sums of a cluster's bond matrix for one per-bond coefficient set
    /// (`coeffs` gives the 6 local components: shear t1, shear t2, axial, bending t1,
    /// bending t2, torsion): per chunk, the 3 translational then 3 rotational rows.
    fn gershgorin_rows(&self, ci: usize, bonds: &[usize], coeffs: impl Fn(&RtBond) -> [f64; 6]) -> Vec<[f64; 6]> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let mut row = vec![[0.0f64; 6]; self.structures[s].chunks.len()];
        for &bi in bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let ks = coeffs(b);
            let axes = [g.t1, g.t2, g.normal];
            for comp in 0..6 {
                let t = axes[comp % 3];
                // 12-vector of this component's kinematic row: [u_a, th_a, u_b, th_b].
                let row_vec: [Vec3; 4] = if comp < 3 {
                    [-t, -(g.ra.cross(t)), t, g.rb.cross(t)]
                } else {
                    [Vec3::ZERO, -t, Vec3::ZERO, t]
                };
                let l1: f64 = row_vec.iter().map(|v| v.x.abs() + v.y.abs() + v.z.abs()).sum();
                for (blk, chunk) in [(0usize, g.a), (2usize, g.b)] {
                    for d in 0..3 {
                        row[chunk][d] += ks[comp] * row_vec[blk][d].abs() * l1;
                        row[chunk][3 + d] += ks[comp] * row_vec[blk + 1][d].abs() * l1;
                    }
                }
            }
        }
        row
    }

    /// Gershgorin row sums of the mass-normalised matrix `M^-1/2 A M^-1/2` (`A` the
    /// bonds' stiffness or damping per `coeffs`), per chunk and degree of freedom:
    /// every entry has the units of `A / M` whatever the units of length, so the bound
    /// (and the substep it sets) is a property of the physics, not of the units. Rotations
    /// use the smallest principal moment (conservative: a smaller inertia only raises the
    /// bound), masses the deformation inertia (`inertia_scale`).
    fn normalized_gershgorin_rows(&self, ci: usize, bonds: &[usize], coeffs: impl Fn(&RtBond) -> [f64; 6]) -> Vec<[f64; 6]> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let st = &self.structures[s];
        let inv_sqrt = |c: usize| {
            let mu = self.chunks[s][c].inertia_scale;
            (1.0 / (mu * st.chunks[c].mass).sqrt(), 1.0 / (mu * smallest_principal_moment(&st.chunks[c].inertia)).sqrt())
        };
        let mut row = vec![[0.0f64; 6]; st.chunks.len()];
        for &bi in bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let ks = coeffs(b);
            let axes = [g.t1, g.t2, g.normal];
            let ((ta, ra), (tb, rb)) = (inv_sqrt(g.a), inv_sqrt(g.b));
            for comp in 0..6 {
                let t = axes[comp % 3];
                let row_vec: [Vec3; 4] = if comp < 3 {
                    [-t, -(g.ra.cross(t)), t, g.rb.cross(t)]
                } else {
                    [Vec3::ZERO, -t, Vec3::ZERO, t]
                };
                // Each entry divided by the square root of its degree of freedom's inertia.
                let scale = [ta, ra, tb, rb];
                let hat: [Vec3; 4] = std::array::from_fn(|k| row_vec[k].abs() * scale[k]);
                let l1: f64 = hat.iter().map(|v| v.x + v.y + v.z).sum();
                for (blk, chunk) in [(0usize, g.a), (2usize, g.b)] {
                    for d in 0..3 {
                        row[chunk][d] += ks[comp] * hat[blk][d] * l1;
                        row[chunk][3 + d] += ks[comp] * hat[blk + 1][d] * l1;
                    }
                }
            }
        }
        row
    }

    /// Full stiffness of a bond's 6 components (concrete plus rebar).
    fn bond_stiffness6(b: &RtBond) -> [f64; 6] {
        let mut k = b.stiffness.as_local();
        if let Some(r) = &b.rebar {
            k.lin.z += r.k_axial;
            k.lin.x += r.k_dowel;
            k.lin.y += r.k_dowel;
        }
        [k.lin.x, k.lin.y, k.lin.z, k.ang.x, k.ang.y, k.ang.z]
    }

    /// Gershgorin bound on each chunk's squared natural frequency (rad^2/s^2) in a
    /// cluster, with every bond at full stiffness (damage only lowers it) and the
    /// chunk's (possibly scaled) deformation inertia. Returns `(chunk, omega^2)`.
    pub fn chunk_frequencies(&self, ci: usize) -> Vec<(usize, f64)> {
        if self.config.methods.scaled_step_bound {
            let row = self.normalized_gershgorin_rows(ci, &self.clusters[ci].bonds, Self::bond_stiffness6);
            return self.clusters[ci].chunks.iter().map(|&c| (c, row[c].iter().copied().fold(0.0, f64::max))).collect();
        }
        let row = self.gershgorin_rows(ci, &self.clusters[ci].bonds, Self::bond_stiffness6);
        let st = &self.structures[self.clusters[ci].structure];
        let s = self.clusters[ci].structure;
        self.clusters[ci]
            .chunks
            .iter()
            .map(|&c| {
                let ch = &st.chunks[c];
                let mu = self.chunks[s][c].inertia_scale;
                let i_min = smallest_principal_moment(&ch.inertia);
                let w2 = (0..3).map(|d| (row[c][d] / ch.mass).max(row[c][3 + d] / i_min)).fold(0.0, f64::max);
                (c, w2 / mu)
            })
            .collect()
    }

    /// Stable central-difference substep of each chunk of a cluster (before the safety
    /// factor), `2/w (sqrt(1 + zeta^2) - zeta)` for every translational and rotational
    /// row, with `w` and `zeta` from that row's own stiffness and dashpot sums
    /// (Gershgorin). Each bond's dashpot is proportional to its own stiffness at its own
    /// frequency, so a soft bond adds little damping: pairing one bond's damping ratio
    /// with another chunk's frequency would overstate the damping.
    pub fn chunk_stable_dts(&self, ci: usize) -> Vec<(usize, f64)> {
        self.chunk_stable_dts_over(ci, &self.clusters[ci].bonds, &self.clusters[ci].chunks)
    }

    /// `chunk_stable_dts` of the given chunks from the given bonds (all a chunk's rows
    /// come from its own bonds, so a chunk's substep needs only those).
    fn chunk_stable_dts_over(&self, ci: usize, bonds: &[usize], chunks: &[usize]) -> Vec<(usize, f64)> {
        if self.config.methods.scaled_step_bound {
            let k = self.normalized_gershgorin_rows(ci, bonds, Self::bond_stiffness6);
            let c = self.normalized_gershgorin_rows(ci, bonds, |b| [b.damping.lin.x, b.damping.lin.y, b.damping.lin.z, b.damping.ang.x, b.damping.ang.y, b.damping.ang.z]);
            return chunks
                .iter()
                .map(|&ch| {
                    let dt = (0..6)
                        .filter(|&d| k[ch][d] > 0.0)
                        .map(|d| {
                            let w = k[ch][d].sqrt();
                            let zeta = c[ch][d] / (2.0 * w);
                            2.0 / w * ((1.0 + zeta * zeta).sqrt() - zeta)
                        })
                        .fold(f64::INFINITY, f64::min);
                    (ch, dt)
                })
                .collect();
        }
        let k = self.gershgorin_rows(ci, bonds, Self::bond_stiffness6);
        let c = self.gershgorin_rows(ci, bonds, |b| [b.damping.lin.x, b.damping.lin.y, b.damping.lin.z, b.damping.ang.x, b.damping.ang.y, b.damping.ang.z]);
        let s = self.clusters[ci].structure;
        chunks
            .iter()
            .map(|&ch| {
                let data = &self.structures[s].chunks[ch];
                let mu = self.chunks[s][ch].inertia_scale;
                let i_min = smallest_principal_moment(&data.inertia);
                let dt = (0..6)
                    .filter(|&d| k[ch][d] > 0.0)
                    .map(|d| {
                        let m = mu * if d < 3 { data.mass } else { i_min };
                        let w = (k[ch][d] / m).sqrt();
                        let zeta = c[ch][d] / (2.0 * m * w);
                        2.0 / w * ((1.0 + zeta * zeta).sqrt() - zeta)
                    })
                    .fold(f64::INFINITY, f64::min);
                (ch, dt)
            })
            .collect()
    }

    /// Selective mass scaling (explicit solve): give every chunk whose own stable
    /// substep is below `target_dt` just enough extra *deformation* inertia to reach it,
    /// as explicit FE codes do instead of softening the material. Stiffness, strengths,
    /// loads and rigid-body masses stay physical; only the response of the smallest chunks
    /// to fast loading is slowed. Each chunk's factor is found by bisection on its own
    /// stable substep (the same bound, damping and safety as `stable_dt`), the others
    /// held: that substep only grows with the chunk's inertia, and scaling a neighbour
    /// afterwards only lowers a chunk's rows, so after one sweep `stable_dt` meets the
    /// target exactly. (A one-shot `(target / dt)^2` ignored the dashpots and, with the
    /// mass-normalised bound, the neighbours' inertia: it fell 5 % short.) Returns the
    /// added mass as a fraction of the total.
    pub fn apply_mass_scaling(&mut self, target_dt: f64) -> f64 {
        let safety = self.config.courant_safety;
        let (mut added, mut total) = (0.0, 0.0);
        for ci in 0..self.clusters.len() {
            let s = self.clusters[ci].structure;
            let chunks = self.clusters[ci].chunks.clone();
            for c in chunks {
                let bonds: Vec<usize> = self.chunk_bonds[s][c].iter().copied().filter(|b| self.clusters[ci].bonds.binary_search(b).is_ok()).collect();
                if !bonds.is_empty() {
                    let meets = |sv: &mut Self, mu: f64| {
                        sv.chunks[s][c].inertia_scale = mu;
                        sv.chunk_stable_dts_over(ci, &bonds, &[c])[0].1 * safety >= target_dt
                    };
                    let mu0 = self.chunks[s][c].inertia_scale;
                    let mut mu = mu0;
                    if !meets(self, mu0) {
                        let (mut lo, mut hi) = (mu0, 2.0 * mu0);
                        while !meets(self, hi) {
                            lo = hi;
                            hi *= 2.0;
                        }
                        loop {
                            let mid = 0.5 * (lo + hi);
                            if mid <= lo || mid >= hi {
                                break;
                            }
                            if meets(self, mid) {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        mu = hi;
                    }
                    self.chunks[s][c].inertia_scale = mu;
                }
                let m = self.structures[s].chunks[c].mass;
                added += (self.chunks[s][c].inertia_scale - 1.0) * m;
                total += m;
            }
        }
        self.refresh_true_steps();
        if total > 0.0 {
            added / total
        } else {
            0.0
        }
    }

    /// Largest stable explicit substep over all active clusters (with damping and safety).
    pub fn stable_dt(&self) -> f64 {
        let mut dt = f64::INFINITY;
        for ci in 0..self.clusters.len() {
            // Only clusters integrated explicitly limit the step (implicit: those in a
            // window). Adaptive: settled clusters too, since a load can wake one at any
            // substep of the frame, and the frame's step was chosen at its start.
            let settled = match self.config.mode {
                SolveMode::Implicit => !self.clusters[ci].explicit_window,
                _ => false,
            };
            if self.clusters[ci].bonds.is_empty() || settled {
                continue;
            }
            let cl = &self.clusters[ci];
            let scales: Vec<u64> = cl.chunks.iter().map(|&c| self.chunks[cl.structure][c].inertia_scale.to_bits()).collect();
            let cached = {
                let cache = self.step_bounds.0.lock().unwrap();
                cache.get(&cl.id).and_then(|(b, c, sc, d)| (*b == cl.bonds && *c == cl.chunks && *sc == scales).then_some(*d))
            };
            let gershgorin = match cached {
                Some(d) => d,
                None => {
                    let d = self.chunk_stable_dts(ci).iter().map(|&(_, d)| d).fold(f64::INFINITY, f64::min);
                    self.step_bounds.0.lock().unwrap().insert(cl.id, (cl.bonds.clone(), cl.chunks.clone(), scales, d));
                    d
                }
            };
            // Both are stable steps of this cluster: the larger holds.
            let own = match self.true_steps.get(self.clusters[ci].structure) {
                Some(&t) if self.config.methods.true_step => gershgorin.max(t),
                _ => gershgorin,
            };
            dt = dt.min(own);
        }
        dt * self.config.courant_safety
    }

    /// `sim.methods.multirate`: each chunk's own stable substep in cluster `ci`, with
    /// the safety factor: its Gershgorin step (`chunk_stable_dts`), no smaller than the
    /// structure's certified step under `true_step` (both are stable steps of the
    /// chunk's rows), combined with the chunk's own contact bound `contact(s, c)` (with
    /// its safety) as `World::substep_dt` combines the global two. Returns `(chunk, dt)`.
    pub fn chunk_native_dts(&self, ci: usize, contact: &impl Fn(usize, usize) -> f64) -> Vec<(usize, f64)> {
        let s = self.clusters[ci].structure;
        let floor = match self.true_steps.get(s) {
            Some(&t) if self.config.methods.true_step => t,
            _ => 0.0,
        };
        let safety = self.config.courant_safety;
        let scaled = self.config.methods.scaled_step_bound;
        self.chunk_stable_dts(ci)
            .into_iter()
            .map(|(c, d)| {
                let stress = d.max(floor) * safety;
                let k = contact(s, c);
                (c, if scaled { 1.0 / (stress.powi(-2) + k.powi(-2)).sqrt() } else { stress.min(k) })
            })
            .collect()
    }

    /// `sim.methods.multirate`: the rate of every bond of cluster `ci` at substep `dt`
    /// (substeps per evaluation, a power of two up to `cap`): that of its finer chunk,
    /// the largest power of two `r` with `r dt` within the chunk's `chunk_native_dts`.
    /// Returns `(bond, rate)` in the cluster's bond order.
    pub fn bond_rates(&self, ci: usize, dt: f64, cap: u32, contact: &impl Fn(usize, usize) -> f64) -> Vec<(usize, u32)> {
        let s = self.clusters[ci].structure;
        let mut native = vec![f64::INFINITY; self.structures[s].chunks.len()];
        for (c, d) in self.chunk_native_dts(ci, contact) {
            native[c] = d;
        }
        self.clusters[ci]
            .bonds
            .iter()
            .map(|&bi| {
                let g = &self.bonds[s][bi].geometry;
                (bi, rate_within(native[g.a].min(native[g.b]), dt, cap))
            })
            .collect()
    }

    /// `sim.methods.multirate`: open a window of `ticks` substeps of `dt` (a frame): every
    /// bond's rate (`bond_rates`, at most the window's length rounded up to a power of
    /// two), every bond due at the first substep. Off, in implicit mode or for a single
    /// substep the window stays closed (single rate).
    pub fn open_multirate_window(&mut self, dt: f64, ticks: usize, contact: impl Fn(usize, usize) -> f64) {
        self.multirate.ticks = 0;
        self.multirate.tick = 0;
        if !self.config.methods.multirate || self.config.mode == SolveMode::Implicit || ticks <= 1 {
            return;
        }
        let ticks = ticks.min(u32::MAX as usize / 2) as u32;
        let cap = ticks.next_power_of_two();
        let mut rate: Vec<Vec<u32>> = self.bonds.iter().map(|b| vec![1; b.len()]).collect();
        for ci in 0..self.clusters.len() {
            if self.clusters[ci].bonds.is_empty() {
                continue;
            }
            let s = self.clusters[ci].structure;
            for (bi, r) in self.bond_rates(ci, dt, cap, &contact) {
                rate[s][bi] = r;
            }
        }
        self.multirate.next = self.bonds.iter().map(|b| vec![0; b.len()]).collect();
        self.multirate.last = self.bonds.iter().map(|b| vec![u32::MAX; b.len()]).collect();
        self.multirate.rate = rate;
        self.multirate.ticks = ticks;
    }

    /// The bonds of each explicit cluster evaluated this substep, with their weight (the
    /// substeps to their next evaluation): in the cluster's bond order, all of them with
    /// weight 1 outside a multirate window. Advances the window's schedule.
    fn firing_bonds(&mut self, explicit: &[usize]) -> Vec<Vec<(usize, f64)>> {
        let open = self.multirate.open();
        let (tick, ticks) = (self.multirate.tick, self.multirate.ticks);
        if !open {
            // Every bond is evaluated: none waits any more.
            self.multirate.last.clear();
        }
        let mut out = Vec::with_capacity(explicit.len());
        for &ci in explicit {
            let cl = &self.clusters[ci];
            let s = cl.structure;
            if !open {
                out.push(cl.bonds.iter().map(|&bi| (bi, 1.0)).collect());
                continue;
            }
            let mr = &mut self.multirate;
            let mut fired = Vec::new();
            for &bi in &cl.bonds {
                // A bond created inside the window (none are today) runs at every substep.
                let r = mr.rate[s].get(bi).copied().unwrap_or(1);
                if mr.next[s].get(bi).is_some_and(|&n| n > tick) {
                    continue;
                }
                // To the next multiple of its rate (the rates' common grid), or the end.
                let w = (r - tick % r).min(ticks - tick);
                if let Some(n) = mr.next[s].get_mut(bi) {
                    *n = tick + w;
                    mr.last[s][bi] = tick;
                }
                fired.push((bi, w as f64));
            }
            out.push(fired);
        }
        out
    }

    /// Whether bond `bi`'s stored response is older than the last substep's start (a
    /// bond waiting for its next multirate evaluation).
    fn bond_is_stale(&self, s: usize, bi: usize) -> bool {
        match self.multirate.last.get(s).and_then(|l| l.get(bi)) {
            Some(&last) => last != u32::MAX && last + 1 < self.multirate.tick.min(self.multirate.ticks),
            None => false,
        }
    }

    /// Elastic energy of bond `bi` at the current chunk state, from its committed joint
    /// state (nothing is committed).
    fn bond_stored_now(&self, s: usize, bi: usize) -> f64 {
        let b = &self.bonds[s][bi];
        let (sa, sb) = (self.chunks[s][b.geometry.a], self.chunks[s][b.geometry.b]);
        let d = if self.config.methods.corotational_bonds {
            b.geometry.corotation(sa.u, sa.th, sb.u, sb.th).deformation()
        } else {
            b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th)
        };
        b.model().evaluate(&b.joint, &d, 0.0, false).stored
    }

    /// `rayleigh_damping`: fit each anchored structure's damping to the two-frequency
    /// Rayleigh form `zeta(w) = alpha / (2 w) + beta w / 2` with the bonds' ratio
    /// `zeta` at both the fundamental `w1` and each bond's axial frequency `w_b`: the
    /// mass-proportional coefficient `alpha = 2 zeta w1 w_b / (w1 + w_b)` (bonds' mean
    /// `w_b`) and every bond's stiffness-proportional dashpot scaled by
    /// `w_b / (w1 + w_b)`. Called once, at the start.
    pub fn fit_mass_damping(&mut self) {
        self.mass_damping = vec![0.0; self.structures.len()];
        let mut fitted = vec![false; self.structures.len()];
        for ci in 0..self.clusters.len() {
            let s = self.clusters[ci].structure;
            if fitted[s] {
                continue;
            }
            let Some(period) = self.fundamental_period(ci) else { continue };
            fitted[s] = true;
            let w1 = 2.0 * std::f64::consts::PI / period;
            // Each bond's axial frequency, from its dashpot: `beta = 2 zeta / w_b`.
            let bond_w: Vec<Option<f64>> = self.structures[s]
                .bonds
                .iter()
                .map(|b| {
                    let beta = b.damping.lin.z / b.stiffness.kn;
                    (beta > 0.0).then(|| 2.0 * b.material.damping_ratio / beta)
                })
                .collect();
            let (mut zeta, mut wb, mut n) = (0.0, 0.0, 0.0);
            for (b, w) in self.structures[s].bonds.iter().zip(&bond_w) {
                if let Some(w) = w {
                    zeta += b.material.damping_ratio;
                    wb += w;
                    n += 1.0;
                }
            }
            if n == 0.0 {
                continue;
            }
            let (zeta, wb) = (zeta / n, wb / n);
            self.mass_damping[s] = 2.0 * zeta * w1 * wb / (w1 + wb);
            // The bonds' dashpots change: every cached step bound is stale.
            self.step_bounds.clear();
            for (k, w) in bond_w.iter().enumerate() {
                if let Some(w) = w {
                    let b = &mut std::sync::Arc::make_mut(&mut self.structures[s]).bonds[k];
                    b.damping = b.damping.scale(w / (w1 + w));
                }
            }
            for rb in self.bonds[s].iter_mut() {
                if let Some(w) = bond_w.get(rb.source).copied().flatten() {
                    rb.damping = rb.damping.scale(w / (w1 + w));
                }
            }
        }
    }

    /// `rayleigh_damping`: the mass-proportional dashpot `(-alpha m v, -alpha I w)` on each
    /// of cluster `ci`'s chunks' hidden motion, in the cluster's chunk order (zero on
    /// supported chunks). On a free cluster the hidden field carries a little net
    /// momentum about its deformed configuration between drift removals, so there the
    /// forces are made self-equilibrated about that configuration (inertia relief, as
    /// `project_load`, with the deformed positions, their mass centre and inertia):
    /// internal damping exerts no net force or moment, and against deformation
    /// velocities the correction does no work.
    pub(crate) fn mass_damping_loads(&self, ci: usize) -> Vec<(Vec3, Vec3)> {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        // Free clusters with `corotational_bonds` only: the projection balances moments
        // about the deformed configuration, as only the co-rotational bonds do
        // (DECISIONS.md 60).
        let alpha = if cl.anchored || self.config.methods.corotational_bonds { self.mass_damping.get(s).copied().unwrap_or(0.0) } else { 0.0 };
        let st = &self.structures[s];
        let mut out: Vec<(Vec3, Vec3)> = cl
            .chunks
            .iter()
            .map(|&c| {
                let (ch, cs) = (&st.chunks[c], &self.chunks[s][c]);
                if alpha > 0.0 && ch.support == Support::None {
                    (cs.v * (-alpha * ch.mass), ch.inertia * cs.w * (-alpha))
                } else {
                    (Vec3::ZERO, Vec3::ZERO)
                }
            })
            .collect();
        if alpha > 0.0 && !cl.anchored {
            let pos: Vec<Vec3> = cl.chunks.iter().map(|&c| st.chunks[c].center + self.chunks[s][c].u).collect();
            let mass: f64 = cl.chunks.iter().map(|&c| st.chunks[c].mass).sum();
            let centre = cl.chunks.iter().zip(&pos).fold(Vec3::ZERO, |acc, (&c, x)| acc + *x * st.chunks[c].mass) / mass;
            let (mut net_f, mut net_m, mut inertia) = (Vec3::ZERO, Vec3::ZERO, Mat3::default());
            for ((&c, (f, m)), x) in cl.chunks.iter().zip(&out).zip(&pos) {
                let (ch, r) = (&st.chunks[c], *x - centre);
                net_f += *f;
                net_m += r.cross(*f) + *m;
                inertia += ch.inertia + (Mat3::IDENTITY * r.norm2() - Mat3::outer(r, r)) * ch.mass;
            }
            let a = net_f / mass;
            let alpha_r = inertia.inverse().expect("invertible") * net_m;
            for ((&c, (f, m)), x) in cl.chunks.iter().zip(out.iter_mut()).zip(&pos) {
                let ch = &st.chunks[c];
                *f -= (a + alpha_r.cross(*x - centre)) * ch.mass;
                *m -= ch.inertia * alpha_r;
            }
        }
        out
    }

    /// `sim.methods.true_step`: refresh every structure's stable stress substep from its
    /// active chunks and live bonds (called where stiffness can grow or inertia change:
    /// at the start, after mass scaling and after refinement).
    pub fn refresh_true_steps(&mut self) {
        if !self.config.methods.true_step {
            return;
        }
        self.true_steps = (0..self.structures.len()).map(|s| self.structure_true_step(s)).collect();
    }

    /// The stable central-difference substep of structure `s`'s bond network at full
    /// stiffness (rebar included), before the safety factor, from the certified frequency
    /// bound of `structure_frequency_bound`. Damping: `C <= beta K` with `beta` the
    /// largest dashpot-to-stiffness ratio of any bond component, so the step
    /// `(2 / w) (sqrt(1 + z^2) - z)` with `z = beta w / 2` is a lower bound of every
    /// mode's.
    pub fn structure_true_step(&self, s: usize) -> f64 {
        let Some((w2, _)) = self.structure_frequency_bound(s) else { return f64::INFINITY };
        let beta = (0..self.bonds[s].len())
            .filter(|&b| self.bonds[s][b].alive)
            .flat_map(|b| {
                let k = Self::bond_stiffness6(&self.bonds[s][b]);
                let c = &self.bonds[s][b].damping;
                let c6 = [c.lin.x, c.lin.y, c.lin.z, c.ang.x, c.ang.y, c.ang.z];
                (0..6).filter(move |&d| k[d] > 0.0).map(move |d| c6[d] / k[d])
            })
            .fold(0.0f64, f64::max);
        let w = w2.sqrt();
        let z = 0.5 * beta * w;
        2.0 / w * ((1.0 + z * z).sqrt() - z)
    }

    /// Certified upper bound of the highest squared natural frequency of structure `s`'s
    /// live bond network at full stiffness, and the scaled Gershgorin bound for
    /// comparison. `A = L^-1 K L^-T` (`M = L L^T` chunk by chunk: `sqrt(m)` and the
    /// Cholesky factor of the inertia, both scaled) is similar to `M^-1 K`, and for any
    /// positive `v`, `lambda_max(A) <= rho(|A|) <= max_i (|A| v)_i / v_i`
    /// (Collatz-Wielandt), so every power-iteration iterate on `|A|` gives a bound, from
    /// its row sums (`v = 1`) down towards `rho(|A|)`. On networks bonded face to face
    /// `|A|` is `A` up to the signs of a diagonal similarity and the bound is the
    /// eigenvalue itself. Supports are treated as free (holding a degree of freedom only
    /// lowers the spectrum). None without stiffness.
    pub fn structure_frequency_bound(&self, s: usize) -> Option<(f64, f64)> {
        const ITERATIONS: usize = 2000;
        const CONVERGED: f64 = 1e-4;
        type Block = [[f64; 6]; 6];
        let st = &self.structures[s];
        let n = st.chunks.len();
        let bonds: Vec<usize> = (0..self.bonds[s].len()).filter(|&b| self.bonds[s][b].alive).collect();
        if bonds.is_empty() {
            return None;
        }
        // L^-1 per chunk.
        let linv: Vec<Block> = (0..n)
            .map(|c| {
                let ch = &st.chunks[c];
                let mu = self.chunks[s][c].inertia_scale;
                let mut out = [[0.0f64; 6]; 6];
                for t in 0..3 {
                    out[t][t] = 1.0 / (ch.mass * mu).sqrt();
                }
                let a: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| ch.inertia.m[i][j] * mu));
                let mut l = [[0.0f64; 3]; 3];
                for i in 0..3 {
                    for j in 0..=i {
                        let sum: f64 = (0..j).map(|p| l[i][p] * l[j][p]).sum();
                        l[i][j] = if i == j { (a[i][i] - sum).max(0.0).sqrt() } else { (a[i][j] - sum) / l[j][j] };
                    }
                }
                for i in 0..3 {
                    out[3 + i][3 + i] = 1.0 / l[i][i];
                    for j in 0..i {
                        let sum: f64 = (j..i).map(|p| l[i][p] * out[3 + p][3 + j]).sum();
                        out[3 + i][3 + j] = -sum / l[i][i];
                    }
                }
                out
            })
            .collect();
        // (L_a^-1 X L_b^-T).
        let congruent = |la: &Block, x: &Block, lb: &Block| -> Block {
            let t: Block = std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|p| la[i][p] * x[p][j]).sum()));
            std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|p| t[i][p] * lb[j][p]).sum()))
        };
        // Each bond's B^T k B as four 6 x 6 blocks, transformed and gathered by chunk pair.
        let mut diag = vec![[[0.0f64; 6]; 6]; n];
        let mut off: Vec<((usize, usize), Block)> = Vec::with_capacity(bonds.len());
        for &bi in &bonds {
            let b = &self.bonds[s][bi];
            let g = &b.geometry;
            let ks = Self::bond_stiffness6(b);
            let axes = [g.t1, g.t2, g.normal];
            let rows: [[f64; 12]; 6] = std::array::from_fn(|comp| {
                let t = axes[comp % 3];
                let r: [Vec3; 4] = if comp < 3 { [-t, -(g.ra.cross(t)), t, g.rb.cross(t)] } else { [Vec3::ZERO, -t, Vec3::ZERO, t] };
                std::array::from_fn(|x| r[x / 3].to_array()[x % 3])
            });
            let kb: [[f64; 12]; 12] = std::array::from_fn(|x| std::array::from_fn(|y| (0..6).map(|c| ks[c] * rows[c][x] * rows[c][y]).sum()));
            let blk = |r0: usize, c0: usize| -> Block { std::array::from_fn(|i| std::array::from_fn(|j| kb[r0 + i][c0 + j])) };
            let (a, bb) = (g.a, g.b);
            let aa = congruent(&linv[a], &blk(0, 0), &linv[a]);
            let bbk = congruent(&linv[bb], &blk(6, 6), &linv[bb]);
            for i in 0..6 {
                for j in 0..6 {
                    diag[a][i][j] += aa[i][j];
                    diag[bb][i][j] += bbk[i][j];
                }
            }
            // The block (row chunk, column chunk) with row chunk < column chunk.
            let (lo, hi, x) = if a < bb { (a, bb, blk(0, 6)) } else { (bb, a, blk(6, 0)) };
            off.push(((lo, hi), congruent(&linv[lo], &x, &linv[hi])));
        }
        off.sort_by_key(|e| e.0);
        let mut merged: Vec<((usize, usize), Block)> = Vec::with_capacity(off.len());
        for (key, x) in off {
            match merged.last_mut() {
                Some((k, acc)) if *k == key => {
                    for i in 0..6 {
                        for j in 0..6 {
                            acc[i][j] += x[i][j];
                        }
                    }
                }
                _ => merged.push((key, x)),
            }
        }
        let abs = |x: &Block| -> Block { x.map(|r| r.map(f64::abs)) };
        let diag: Vec<Block> = diag.iter().map(abs).collect();
        let merged: Vec<((usize, usize), Block)> = merged.iter().map(|(k, x)| (*k, abs(x))).collect();
        let apply = |v: &[[f64; 6]]| -> Vec<[f64; 6]> {
            let mut y = vec![[0.0f64; 6]; n];
            for c in 0..n {
                for i in 0..6 {
                    y[c][i] = (0..6).map(|j| diag[c][i][j] * v[c][j]).sum();
                }
            }
            for &((a, b), x) in &merged {
                for i in 0..6 {
                    for j in 0..6 {
                        y[a][i] += x[i][j] * v[b][j];
                        y[b][j] += x[i][j] * v[a][i];
                    }
                }
            }
            y
        };
        let mut v = vec![[1.0f64; 6]; n];
        let mut bound = f64::INFINITY;
        for _ in 0..ITERATIONS {
            let y = apply(&v);
            let (mut hi, mut lo, mut top) = (0.0f64, f64::INFINITY, 0.0f64);
            for c in 0..n {
                for d in 0..6 {
                    if y[c][d] > 0.0 {
                        // A positive row with a zero entry of v gives no bound this round.
                        let ratio = if v[c][d] > 0.0 { y[c][d] / v[c][d] } else { f64::INFINITY };
                        hi = hi.max(ratio);
                        lo = lo.min(ratio);
                    }
                    top = top.max(y[c][d]);
                }
            }
            bound = bound.min(hi);
            if !(top > 0.0) || hi - lo <= CONVERGED * hi {
                break;
            }
            v = y.iter().map(|r| r.map(|x| x / top)).collect();
        }
        let gershgorin = (0..self.clusters.len())
            .filter(|&ci| self.clusters[ci].structure == s && !self.clusters[ci].bonds.is_empty())
            .flat_map(|ci| self.normalized_gershgorin_rows(ci, &self.clusters[ci].bonds, Self::bond_stiffness6))
            .flat_map(|row| row.into_iter())
            .fold(0.0f64, f64::max);
        (bound > 0.0 && bound.is_finite()).then_some((bound, gershgorin))
    }

    // ------------------------------------------------------------------ substep

    /// Advance every cluster by one explicit substep under the given external loads.
    pub fn substep(&mut self, dt: f64, loads: &ChunkLoads) {
        let t_mid = self.time + dt;
        // Clusters integrated explicitly this substep. Their bond responses and chunk
        // loads depend only on the state at the start of the substep, so all of them
        // (across every cluster) are evaluated in parallel; the results are then applied
        // cluster by cluster in index order, bond by bond, exactly as a sequential sweep
        // would (bit-identical for any thread count).
        let explicit: Vec<usize> = (0..self.clusters.len()).filter(|&ci| self.integrates_explicitly(ci)).collect();
        crate::profile::count(crate::profile::Counter::ExplicitClusters, explicit.len() as u64);
        let t_accel = crate::profile::time(crate::profile::Stage::RigidAccel);
        let accel = crate::par::map(&explicit, |&ci| self.rigid_acceleration(ci, loads));
        drop(t_accel);
        let fracture = self.config.fracture;
        if self.config.methods.corotational_bonds {
            let turns = self.chunks.iter().map(|cs| crate::par::map(cs, |c| ChunkTurn::new(c.th))).collect();
            self.turns = turns;
        }
        // The bonds evaluated this substep and their weights (`multirate`; else all, 1).
        let fired = self.firing_bonds(&explicit);
        let bond_jobs: Vec<(usize, usize, f64)> = explicit
            .iter()
            .zip(&fired)
            .flat_map(|(&ci, list)| {
                let s = self.clusters[ci].structure;
                list.iter().map(move |&(bi, w)| (s, bi, w))
            })
            .collect();
        crate::profile::count(crate::profile::Counter::BondEvaluations, bond_jobs.len() as u64);
        self.bond_evaluations += bond_jobs.len() as u64;
        if crate::par::serial(bond_jobs.len()) {
            // On one thread each response is applied as it is computed (no list of them):
            // a response reads only its bond and the chunks' motion, which applying the
            // others does not change, so this is the same sweep.
            let _t = crate::profile::time(crate::profile::Stage::BondResponses);
            for (&ci, list) in explicit.iter().zip(&fired) {
                self.apply_bond_responses(ci, list, &mut None::<std::iter::Empty<_>>, (dt, fracture), t_mid);
            }
        } else {
            let t_bonds = crate::profile::time(crate::profile::Stage::BondResponses);
            let responses = crate::par::map(&bond_jobs, |&(s, bi, w)| self.bond_response(s, bi, dt * w, fracture)).into_iter();
            drop(t_bonds);
            let _t = crate::profile::time(crate::profile::Stage::ApplyBonds);
            let mut responses = Some(responses);
            for (&ci, list) in explicit.iter().zip(&fired) {
                self.apply_bond_responses(ci, list, &mut responses, (dt, fracture), t_mid);
            }
        }
        let t_loads = crate::profile::time(crate::profile::Stage::ChunkLoads);
        let chunk_jobs: Vec<(usize, usize)> =
            explicit.iter().enumerate().flat_map(|(k, &ci)| self.clusters[ci].chunks.iter().map(move |&c| (k, c))).collect();
        let mut frame_loads = crate::par::map(&chunk_jobs, |&(k, c)| {
            let (a, alpha) = accel[k];
            self.frame_loads(explicit[k], c, loads, a, alpha, t_mid)
        })
        .into_iter();
        drop(t_loads);
        let t_integrate = crate::profile::time(crate::profile::Stage::Integrate);
        let mut next = explicit.iter().zip(&accel).peekable();
        for ci in 0..self.clusters.len() {
            match next.peek() {
                Some(&(&e, &(a, alpha))) if e == ci => {
                    next.next();
                    self.integrate_cluster(ci, dt, loads, a, alpha, &mut frame_loads);
                }
                _ => {
                    if self.config.mode == SolveMode::Implicit {
                        self.implicit_predict(ci, dt, loads, t_mid);
                    }
                    self.advance_rigid_only(ci, dt, loads);
                }
            }
        }
        drop(t_integrate);
        self.time += dt;
        self.substeps += 1;
        if self.multirate.open() {
            self.multirate.tick += 1;
        }
        if !self.pending_split.is_empty() {
            let _t = crate::profile::time(crate::profile::Stage::Splits);
            self.process_splits();
        }
    }

    /// Whether cluster `ci` takes explicit substeps (the others are quasi-static,
    /// implicit or settled and only move rigidly here).
    fn integrates_explicitly(&self, ci: usize) -> bool {
        match self.config.mode {
            SolveMode::Explicit => true,
            SolveMode::QuasiStatic => false,
            SolveMode::Implicit => self.clusters[ci].explicit_window,
            SolveMode::Adaptive => self.clusters[ci].activity == Activity::Active,
        }
    }

    /// External loads of a chunk in the body frame of its cluster, minus the inertial
    /// loads of the cluster's rigid motion (`a`, `alpha` world).
    pub(crate) fn frame_loads(&self, ci: usize, c: usize, loads: &ChunkLoads, a: Vec3, alpha: Vec3, t: f64) -> (Vec3, Vec3) {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let ch = &self.structures[s].chunks[c];
        let r_world = cl.pose.transform_vector(ch.center + self.chunks[s][c].u - cl.com);
        let w = cl.angular_velocity;
        let rot = cl.rotation();
        let iw = rot * ch.inertia * rot.transpose();
        let mut f_world = loads.force[s][c] + self.config.gravity * ch.mass;
        let mut t_world = loads.torque[s][c];
        if self.config.features.rigid_motion_loads {
            f_world -= (a + alpha.cross(r_world) + w.cross(w.cross(r_world))) * ch.mass;
            t_world -= iw * alpha + w.cross(iw * w);
        }
        let mut f = cl.pose.inverse_transform_vector(f_world);
        let mut m = cl.pose.inverse_transform_vector(t_world);
        if self.config.features.rigid_motion_loads {
            // Coupling of the hidden velocities with the frame rotation (body frame):
            // Coriolis force on the chunk and the gyroscopic cross terms of its spin.
            let wb = cl.pose.inverse_transform_vector(w);
            let st = &self.chunks[s][c];
            f -= wb.cross(st.v) * (2.0 * ch.mass);
            m -= wb.cross(ch.inertia * st.w) + st.w.cross(ch.inertia * wb) + st.w.cross(ch.inertia * st.w);
        }
        for rl in &self.replacements {
            if rl.structure == s && rl.chunk == c {
                let k = rl.factor(t);
                f += rl.force * k;
                m += rl.moment * k;
            }
        }
        (f, m)
    }

    /// Net external force and torque (about the com) on a cluster, world frame, gravity included.
    pub(crate) fn net_load(&self, ci: usize, loads: &ChunkLoads) -> (Vec3, Vec3) {
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let com = cl.com_world();
        let mut f = Vec3::ZERO;
        let mut t = Vec3::ZERO;
        for &c in &cl.chunks {
            let ch = &self.structures[s].chunks[c];
            let fc = loads.force[s][c] + self.config.gravity * ch.mass;
            let x = cl.pose.transform_point(ch.center + self.chunks[s][c].u);
            f += fc;
            t += (x - com).cross(fc) + loads.torque[s][c];
        }
        for rl in &self.replacements {
            if rl.structure == s && self.chunks[s][rl.chunk].active && self.chunks[s][rl.chunk].cluster == ci {
                let k = rl.factor(self.time);
                let x = cl.pose.transform_point(self.structures[s].chunks[rl.chunk].center);
                let fw = cl.pose.transform_vector(rl.force * k);
                f += fw;
                t += (x - com).cross(fw) + cl.pose.transform_vector(rl.moment * k);
            }
        }
        (f, t)
    }

    /// Rigid acceleration of a free cluster under its net load.
    pub(crate) fn rigid_acceleration(&self, ci: usize, loads: &ChunkLoads) -> (Vec3, Vec3) {
        let cl = &self.clusters[ci];
        if cl.anchored {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let (f, t) = self.net_load(ci, loads);
        let iw = cl.world_inertia();
        let w = cl.angular_velocity;
        let alpha = iw.inverse().expect("cluster inertia invertible") * (t - w.cross(iw * w));
        (f / cl.mass, alpha)
    }

    /// Symplectic rigid update. The angular part integrates angular momentum (exactly
    /// conserved when the net torque vanishes) and recovers the angular velocity from
    /// the rotated inertia.
    fn integrate_rigid(&mut self, ci: usize, dt: f64, a: Vec3, alpha: Vec3) {
        if !self.config.integrate_rigid || self.clusters[ci].anchored || self.clusters[ci].driven {
            return;
        }
        let cl = &mut self.clusters[ci];
        let com_world = cl.com_world();
        let iw = cl.world_inertia();
        // alpha = Iw^-1 (tau - w x Iw w)  =>  dL/dt = tau = Iw alpha + w x Iw w.
        let torque = iw * alpha + cl.angular_velocity.cross(iw * cl.angular_velocity);
        let l = iw * cl.angular_velocity + torque * dt;
        cl.velocity += a * dt;
        let new_com = com_world + cl.velocity * dt;
        // Rotate with the momentum-consistent angular velocity, then refresh it.
        let w_mid = iw.inverse().expect("invertible") * l;
        cl.pose.rotation = cl.pose.rotation.integrate(w_mid, dt);
        cl.pose.position = new_com - cl.pose.rotation.rotate(cl.com);
        cl.angular_velocity = cl.world_inertia().inverse().expect("invertible") * l;
    }

    /// Rigid update only (no deformation), e.g. between implicit steps.
    pub(crate) fn advance_rigid(&mut self, ci: usize, dt: f64, loads: &ChunkLoads) {
        self.advance_rigid_only(ci, dt, loads);
    }

    /// An impulse `impulse` (world, N s) at world point `point` on chunk `c`, as an
    /// instantaneous change of the chunk's velocity: the velocity condition of an impact
    /// with inertia in the stress solve. Under an engine (`integrate_rigid` off) the
    /// engine has already given the cluster its rigid share, so the chunk's hidden
    /// velocity takes the change and the floating frame removes the rigid share again.
    /// In implicit mode the next step starts from the kicked velocity.
    pub fn apply_chunk_impulse(&mut self, s: usize, c: usize, impulse: Vec3, point: Vec3) {
        let ci = self.chunks[s][c].cluster;
        let pose = self.clusters[ci].pose;
        let ch = &self.structures[s].chunks[c];
        if ch.support != Support::None {
            return;
        }
        let x = self.chunk_position(s, c);
        let st = &mut self.chunks[s][c];
        let dv = pose.inverse_transform_vector(impulse) * (1.0 / (ch.mass * st.inertia_scale));
        let dw = ch.inv_inertia * pose.inverse_transform_vector((point - x).cross(impulse)) * (1.0 / st.inertia_scale);
        st.v += dv;
        st.w += dw;
        st.step_start[2] += dv;
        st.step_start[3] += dw;
    }

    fn advance_rigid_only(&mut self, ci: usize, dt: f64, loads: &ChunkLoads) {
        let (a, alpha) = self.rigid_acceleration(ci, loads);
        self.integrate_rigid(ci, dt, a, alpha);
    }

    /// Response of bond `bi` of structure `s` to the current chunk displacements, with
    /// its dashpot force (from the lagged velocities) and the energy that dissipates.
    fn bond_response(&self, s: usize, bi: usize, dt: f64, fracture: bool) -> BondResponse {
        let b = &self.bonds[s][bi];
        let (sa, sb) = (self.chunks[s][b.geometry.a], self.chunks[s][b.geometry.b]);
        let corotation = self.config.methods.corotational_bonds.then(|| b.geometry.corotation_with(sa.u, &self.turns[s][b.geometry.a], sb.u, &self.turns[s][b.geometry.b]));
        let (d, rate) = match &corotation {
            Some(cr) => (cr.deformation(), cr.variation(sa.v, sa.w, sb.v, sb.w)),
            None => (b.geometry.kinematics(sa.u, sa.th, sb.u, sb.th), b.geometry.kinematics(sa.v, sa.w, sb.v, sb.w)),
        };
        let resp = b.model().evaluate(&b.joint, &d, dt, fracture);
        let factors = b.model().secant_factors(&resp.state, &d);
        let q_damp = rate.mul_elem(&b.damping).mul_elem(&factors);
        let q = resp.force.add(&q_damp);
        let damped = q_damp.dot(&rate) * dt;
        let loads = match &corotation {
            Some(cr) => cr.loads(&q),
            None => b.geometry.chunk_loads(&q),
        };
        (resp, q, damped, loads)
    }

    /// Chunk loads of bond `bi`'s generalised force `q` at the current chunk state.
    fn bond_chunk_loads(&self, s: usize, bi: usize, q: &Local6) -> (Vec3, Vec3, Vec3, Vec3) {
        let g = &self.bonds[s][bi].geometry;
        if self.config.methods.corotational_bonds {
            let (sa, sb) = (self.chunks[s][g.a], self.chunks[s][g.b]);
            g.corotation(sa.u, sa.th, sb.u, sb.th).loads(q)
        } else {
            g.chunk_loads(q)
        }
    }

    /// Commit the responses of cluster `ci`'s bonds `fired` (bond, weight; taken in that
    /// order from `responses`, or computed here, over `(weight dt, fracture)`, without
    /// one): joint states, internal chunk forces, energies and events. A bond of weight
    /// `w` (`multirate`) acts for `w` substeps: its impulse over them is given in this one.
    fn apply_bond_responses(&mut self, ci: usize, fired: &[(usize, f64)], responses: &mut Option<impl Iterator<Item = BondResponse>>, (dt, fracture): (f64, bool), t: f64) {
        let s = self.clusters[ci].structure;
        for &c in &self.clusters[ci].chunks {
            self.f_int[s][c] = (Vec3::ZERO, Vec3::ZERO);
        }
        let mut dissipated = 0.0;
        let mut damped = 0.0;
        for &(bi, w) in fired {
            let (resp, q, bond_damped, (mut fa, mut ma, mut fb, mut mb)) = match responses {
                Some(list) => list.next().expect("one response per bond"),
                None => self.bond_response(s, bi, dt * w, fracture),
            };
            if w != 1.0 {
                (fa, ma, fb, mb) = (fa * w, ma * w, fb * w, mb * w);
            }
            let (ga, gb) = {
                let g = &self.bonds[s][bi].geometry;
                (g.a, g.b)
            };
            damped += bond_damped;
            dissipated += resp.dissipated;
            self.energy.softening_overshoot += resp.overshoot;
            self.max_utilization = self.max_utilization.max(resp.state.utilization);
            let bm = &mut self.bonds[s][bi];
            let previous = std::mem::replace(&mut bm.joint, resp.state);
            bm.force = q;
            bm.elastic = resp.force;
            bm.measures = resp.measures;
            bm.stored = resp.stored;
            self.f_int[s][ga].0 += fa;
            self.f_int[s][ga].1 += ma;
            self.f_int[s][gb].0 += fb;
            self.f_int[s][gb].1 += mb;
            if self.record_bond_events(ci, bi, &previous, resp.disconnected, t) && !self.pending_split.contains(&ci) {
                self.pending_split.push(ci);
            }
        }
        self.energy.bond_dissipation += dissipated;
        self.energy.damping_dissipation += damped;
    }

    /// Central-difference update of cluster `ci`'s chunk deformation (velocities at half
    /// steps) under the internal forces and its chunk loads (taken in chunk order from
    /// `frame_loads`), then its rigid motion.
    fn integrate_cluster(&mut self, ci: usize, dt: f64, loads: &ChunkLoads, a: Vec3, alpha: Vec3, frame_loads: &mut impl Iterator<Item = (Vec3, Vec3)>) {
        let s = self.clusters[ci].structure;
        let (pose, cv, cw, com) = {
            let cl = &self.clusters[ci];
            (cl.pose, cl.velocity, cl.angular_velocity, cl.com)
        };
        let mut work = 0.0;
        let mass_damping = ((self.clusters[ci].anchored || self.config.methods.corotational_bonds) && self.mass_damping.get(s).copied().unwrap_or(0.0) > 0.0).then(|| self.mass_damping_loads(ci));
        let mut damped = 0.0;
        let implicit = self.config.mode == SolveMode::Implicit;
        let corotational = self.config.methods.corotational_bonds;
        for k in 0..self.clusters[ci].chunks.len() {
            let c = self.clusters[ci].chunks[k];
            let (f_ext, m_ext) = frame_loads.next().expect("one load per chunk");
            let (mut fi, mut mi) = self.f_int[s][c];
            if let Some(md) = &mass_damping {
                // `rayleigh_damping`: the mass-proportional dashpot on the hidden motion.
                let ((fd, mdm), st) = (md[k], &self.chunks[s][c]);
                damped -= (fd.dot(st.v) + mdm.dot(st.w)) * dt;
                fi += fd;
                mi += mdm;
            }
            let ch = &self.structures[s].chunks[c];
            let st = &mut self.chunks[s][c];
            match ch.support {
                Support::Fixed => {
                    st.reaction = (-(f_ext + fi), -(m_ext + mi));
                    st.v = Vec3::ZERO;
                    st.w = Vec3::ZERO;
                }
                Support::Pinned => {
                    st.reaction = (-(f_ext + fi), Vec3::ZERO);
                    st.v = Vec3::ZERO;
                    st.w += ch.inv_inertia * (m_ext + mi) * (dt / st.inertia_scale);
                    st.th = turn(st.th, st.w * dt, corotational);
                    if implicit {
                        (st.a, st.alpha) = (Vec3::ZERO, ch.inv_inertia * (m_ext + mi) * (1.0 / st.inertia_scale));
                    }
                }
                Support::None => {
                    st.v += (f_ext + fi) * (dt / (ch.mass * st.inertia_scale));
                    st.w += ch.inv_inertia * (m_ext + mi) * (dt / st.inertia_scale);
                    st.u += st.v * dt;
                    st.th = turn(st.th, st.w * dt, corotational);
                    if implicit {
                        // An explicit window hands Newmark its `a_n` at the frame's end.
                        (st.a, st.alpha) = ((f_ext + fi) * (1.0 / (ch.mass * st.inertia_scale)), ch.inv_inertia * (m_ext + mi) * (1.0 / st.inertia_scale));
                    }
                }
            }
            // Work of the external (non-gravity) loads at the chunk's world velocity.
            let r = pose.transform_vector(ch.center + st.u - com);
            let v_world = cv + cw.cross(r) + pose.transform_vector(st.v);
            let w_world = cw + pose.transform_vector(st.w);
            work += (loads.force[s][c].dot(v_world) + loads.torque[s][c].dot(w_world)) * dt;
        }
        self.energy.external_work += work;
        self.energy.damping_dissipation += damped;
        self.integrate_rigid(ci, dt, a, alpha);
        if !self.clusters[ci].anchored {
            self.remove_rigid_drift(ci);
        }
    }

    /// Emit the events of bond `bi` (in cluster `ci`) changing from `previous` to its
    /// current state at time `t`: first damage (`Cracked`), first percent of strength
    /// lost to static fatigue (`Creaked`), disconnection (`Broken`). Returns whether the
    /// bond disconnected (the cluster must then be split).
    pub(crate) fn record_bond_events(&mut self, ci: usize, bi: usize, previous: &JointState, disconnected: bool, t: f64) -> bool {
        let s = self.clusters[ci].structure;
        let b = &self.bonds[s][bi];
        let pose = self.clusters[ci].pose;
        let position = pose.transform_point(b.geometry.centroid).to_array();
        if !previous.is_damaged() && b.joint.is_damaged() {
            let mode = b.joint.mode;
            self.events.push(SolverEvent::Cracked { time: t, structure: s, bond: bi, mode, position });
        }
        let b = &self.bonds[s][bi];
        if fatigue_factor(&b.strength, previous) > CREAK_STRENGTH && fatigue_factor(&b.strength, &b.joint) <= CREAK_STRENGTH {
            self.events.push(SolverEvent::Creaked { time: t, structure: s, bond: bi, position });
        }
        let b = &self.bonds[s][bi];
        if disconnected {
            self.events.push(SolverEvent::Broken {
                time: t,
                structure: s,
                bond: bi,
                mode: b.joint.mode,
                position,
                normal: pose.transform_vector(b.geometry.normal).to_array(),
                dissipated: b.joint.dissipated,
            });
        }
        disconnected
    }

    /// Disconnect the given bonds now (a fracture verdict decided elsewhere, e.g. by a
    /// trial evaluation of the same tick) and split the clusters they held together.
    /// The energy the bonds store is booked as `split_release`; each bond gets a
    /// `Broken` event. Children inherit the exact momentum of their chunks.
    pub fn break_bonds(&mut self, bonds: &[(usize, usize)]) {
        let mut clusters = Vec::new();
        for &(s, bi) in bonds {
            if !self.bonds[s][bi].connected() {
                continue;
            }
            let previous = self.bonds[s][bi].joint.clone();
            let b = &mut self.bonds[s][bi];
            b.joint.damage = 1.0;
            b.joint.rebar_broken = true;
            let ci = self.chunks[s][b.geometry.a].cluster;
            self.record_bond_events(ci, bi, &previous, true, self.time);
            if !clusters.contains(&ci) {
                clusters.push(ci);
            }
        }
        // Highest index first: splitting swap-removes the parent.
        clusters.sort_unstable();
        for &ci in clusters.iter().rev() {
            self.split_cluster(ci);
        }
    }

    /// Keep the floating frame on the cluster (a mean-axis frame): fold the best-fit
    /// rigid translation and rotation of the hidden displacements into the cluster pose,
    /// and the net linear and angular momentum of the hidden velocities into its rigid
    /// velocity. Chunk positions are unchanged to second order in the (tiny, per-substep)
    /// correction and chunk velocities exactly, so total momentum is conserved.
    ///
    /// Without this the small-rotation bond model sees a growing rigid rotation of the
    /// hidden field whose bond forces do not turn with it, which is unstable in a
    /// spinning body. Moving only the velocities (not the rotation) is inconsistent too:
    /// the hidden field then integrates the opposite rotation. Under an engine (which owns
    /// rigid motion) the hidden rigid motion is removed instead.
    pub(crate) fn remove_rigid_drift(&mut self, ci: usize) {
        self.fold_rigid_drift(ci, false);
    }

    /// `remove_rigid_drift`; with `exact`, the transfer keeps every chunk's world
    /// position, orientation, velocity and spin exactly (for the finite fold of a new
    /// fragment's hidden rigid motion; per substep the fold is O(dt) and the linear
    /// transfer, free of cancellation, is the accurate one).
    fn fold_rigid_drift(&mut self, ci: usize, exact: bool) {
        let weight_of = |sv: &Self, s: usize, c: usize| sv.chunks[s][c].inertia_scale;
        let cl = &self.clusters[ci];
        let s = cl.structure;
        let st = &self.structures[s];
        // Deformation-inertia weights (true masses unless mass scaling is on): the
        // frame follows the hidden field's momentum under the inertia that integrates it.
        let weight = |c: usize| self.chunks[s][c].inertia_scale;
        let scaled = cl.chunks.iter().any(|&c| weight(c) != 1.0);
        let (mass, com, inertia) = if scaled {
            let mass: f64 = cl.chunks.iter().map(|&c| st.chunks[c].mass * weight(c)).sum();
            let com = cl.chunks.iter().map(|&c| st.chunks[c].center * (st.chunks[c].mass * weight(c))).fold(Vec3::ZERO, |a, b| a + b)
                / mass;
            let mut inertia = Mat3::ZERO;
            for &c in &cl.chunks {
                let ch = &st.chunks[c];
                let r = ch.center - com;
                inertia += (ch.inertia + (Mat3::IDENTITY * r.norm2() - Mat3::outer(r, r)) * ch.mass) * weight(c);
            }
            (mass, com, inertia)
        } else {
            (cl.mass, cl.com, cl.inertia)
        };
        let inv_i = inertia.inverse().expect("invertible");
        let (mut tu, mut pv) = (Vec3::ZERO, Vec3::ZERO);
        for &c in &cl.chunks {
            let m = st.chunks[c].mass * weight(c);
            tu += self.chunks[s][c].u * m;
            pv += self.chunks[s][c].v * m;
        }
        let (t, dv) = (tu / mass, pv / mass);
        let (mut lu, mut lv) = (Vec3::ZERO, Vec3::ZERO);
        for &c in &cl.chunks {
            let ch = &st.chunks[c];
            let cs = &self.chunks[s][c];
            let r = ch.center - com;
            let k = weight(c);
            lu += (r.cross(cs.u - t) * ch.mass + ch.inertia * cs.th) * k;
            lv += (r.cross(cs.v - dv) * ch.mass + ch.inertia * cs.w) * k;
        }
        let (phi, dw) = (inv_i * lu, inv_i * lv);
        let chunks = cl.chunks.clone();
        let true_com = cl.com;
        if exact && self.config.integrate_rigid && !self.clusters[ci].driven {
            // Every chunk keeps its world position, orientation, velocity and spin exactly:
            // record them, move the frame, and re-express them in it. The linearised
            // transfer below moves positions by O(phi^2) and velocities by O(phi v): a
            // finite fold (the first after a split) changed momentum and energy.
            let world: Vec<(Vec3, Quat, Vec3, Vec3)> = chunks
                .iter()
                .map(|&c| {
                    let (v, w) = self.chunk_velocity(s, c);
                    let th = self.chunks[s][c].th;
                    (self.chunk_position(s, c), self.clusters[ci].pose.rotation * Quat::from_rotation_vector(th), v, w)
                })
                .collect();
            let cl = &mut self.clusters[ci];
            let rot = cl.pose.rotation;
            cl.pose.position += rot.rotate(t - phi.cross(com));
            cl.pose.rotation = (rot * Quat::from_rotation_vector(phi)).normalized();
            cl.velocity += rot.rotate(dv + dw.cross(true_com - com));
            cl.angular_velocity += rot.rotate(dw);
            let cl = self.clusters[ci].clone();
            let back = cl.pose.rotation.conjugate();
            for (&c, (x, q, v, w)) in chunks.iter().zip(world) {
                let center = self.structures[s].chunks[c].center;
                let u = cl.pose.inverse_transform_point(x) - center;
                let r = cl.pose.transform_vector(center + u - cl.com);
                let cs = &mut self.chunks[s][c];
                cs.u = u;
                cs.th = (back * q).to_rotation_vector();
                cs.v = back.rotate(v - cl.velocity - cl.angular_velocity.cross(r));
                cs.w = back.rotate(w - cl.angular_velocity);
            }
            // Re-centre (the scheme conserves momentum exactly only with the hidden mean
            // displacement and velocity at zero): the frame takes them, the chunks keep
            // their world positions and velocities.
            let (mut tu, mut pv) = (Vec3::ZERO, Vec3::ZERO);
            for &c in &chunks {
                let m = self.structures[s].chunks[c].mass * weight_of(self, s, c);
                tu += self.chunks[s][c].u * m;
                pv += self.chunks[s][c].v * m;
            }
            let (t2, v2) = (tu / mass, pv / mass);
            for &c in &chunks {
                self.chunks[s][c].u -= t2;
                self.chunks[s][c].v -= v2;
            }
            let rot = self.clusters[ci].pose.rotation;
            let shift = rot.rotate(t2);
            self.clusters[ci].pose.position += shift;
            // The chunks' lever arms from the frame's centre moved by `-shift`: the
            // frame's velocity takes the rigid field's change there too.
            let w = self.clusters[ci].angular_velocity;
            self.clusters[ci].velocity += rot.rotate(v2) + w.cross(shift);
            return;
        }
        for c in chunks {
            let r = st.chunks[c].center - com;
            let cs = &mut self.chunks[s][c];
            cs.u -= t + phi.cross(r);
            cs.th -= phi;
            cs.v -= dv + dw.cross(r);
            cs.w -= dw;
        }
        if self.config.integrate_rigid && !self.clusters[ci].driven {
            // The frame turns by `phi` about the fold's centre `com`, not about the body
            // origin: `(exp(phi) - I) com` exactly (Rodrigues), where the first-order
            // `phi x com` left every chunk off by `1/2 phi x (phi x c)`, `c` its distance
            // from the body origin, an arbitrary choice of the scene (9.5 cm for a split
            // child's 0.09 rad at 23.6 m in fractured-bridge). The chunks' hidden
            // correction above is first order about `com` either way.
            let cl = &mut self.clusters[ci];
            let rot = cl.pose.rotation;
            cl.pose.position += rot.rotate(t - turn_difference(phi, com));
            cl.pose.rotation = (rot * Quat::from_axis_angle(phi, phi.norm())).normalized();
            // The removed velocity field `dv + dw x (x - com)` evaluated at the true com.
            cl.velocity += rot.rotate(dv + dw.cross(true_com - com));
            cl.angular_velocity += rot.rotate(dw);
        }
    }

    // ------------------------------------------------------------------ fracture

    fn process_splits(&mut self) {
        let mut pending = std::mem::take(&mut self.pending_split);
        pending.sort_unstable();
        pending.dedup();
        // Highest index first so swap-removal does not disturb lower pending indices.
        for &ci in pending.iter().rev() {
            self.split_cluster(ci);
        }
    }

    /// Recompute the connectivity of cluster `ci` and split it into children.
    pub fn split_cluster(&mut self, ci: usize) {
        crate::profile::count(crate::profile::Counter::Splits, 1);
        crate::profile::count(crate::profile::Counter::SplitChunks, self.clusters[ci].chunks.len() as u64);
        let s = self.clusters[ci].structure;
        let comps = self.components(s, &self.clusters[ci].chunks.clone());
        if comps.len() <= 1 {
            return;
        }
        let parent = self.clusters.swap_remove(ci);
        if ci < self.clusters.len() {
            // Re-point the chunks of the cluster that moved into slot `ci`.
            let moved = self.clusters[ci].clone();
            for &c in &moved.chunks {
                self.chunks[moved.structure][c].cluster = ci;
            }
        }
        // Mass data of the parent's chunks (copied: the loop below mutates `self`).
        // (Dense by chunk: only looked up.)
        let n_chunks = self.structures[s].chunks.len();
        let mut lite = vec![(Vec3::ZERO, 0.0, Mat3::default()); n_chunks];
        for &c in &parent.chunks {
            let ch = &self.structures[s].chunks[c];
            lite[c] = (ch.center, ch.mass, ch.inertia);
        }
        let rot = parent.rotation();
        // Exact world velocity of every chunk before the split.
        let mut vel = vec![(Vec3::ZERO, Vec3::ZERO); n_chunks];
        let com_world = parent.com_world();
        for &c in &parent.chunks {
            let cs = &self.chunks[s][c];
            let x = parent.pose.transform_point(lite[c].0 + cs.u);
            let v = parent.velocity + parent.angular_velocity.cross(x - com_world) + rot * cs.v;
            let w = parent.angular_velocity + rot * cs.w;
            vel[c] = (v, w);
        }
        // Bonds crossing between children are handed over to rigid-body contact.
        let mut comp_of = vec![usize::MAX; n_chunks];
        for (k, comp) in comps.iter().enumerate() {
            for &c in comp {
                comp_of[c] = k;
            }
        }
        for &bi in &parent.bonds {
            let g = &self.bonds[s][bi].geometry;
            if comp_of[g.a] != comp_of[g.b] {
                let b = &self.bonds[s][bi];
                if self.config.methods.layer_contact {
                    let e = b.elastic;
                    self.handovers.push(Handover {
                        structure: s,
                        a: g.a,
                        b: g.b,
                        compression: (-e.lin.z).max(0.0),
                        stored: b.stored,
                        force: rot * g.to_body(e.lin),
                        couple: rot * g.to_body(e.ang),
                    });
                } else {
                    self.energy.split_release += b.stored;
                }
                self.bonds[s][bi].alive = false;
            }
        }
        self.rebuild_adjacency(s);
        let mut children = Vec::new();
        for comp in comps {
            // Linear and angular momentum of the child from its chunks' exact velocities.
            let mass: f64 = comp.iter().map(|&c| lite[c].1).sum();
            let com_local =
                comp.iter().map(|&c| lite[c].0 * lite[c].1).fold(Vec3::ZERO, |a, b| a + b) / mass;
            let mut p = Vec3::ZERO;
            let mut l = Vec3::ZERO;
            let mut i_local = Mat3::ZERO;
            for &c in &comp {
                let (center, ch_mass, ch_inertia) = lite[c];
                let (v, w) = vel[c];
                let r = parent.pose.transform_vector(center - com_local);
                p += v * ch_mass;
                l += r.cross(v) * ch_mass + rot * ch_inertia * rot.transpose() * w;
                let rl = center - com_local;
                i_local += ch_inertia + (Mat3::IDENTITY * rl.norm2() - Mat3::outer(rl, rl)) * ch_mass;
            }
            let iw = rot * i_local * rot.transpose();
            let (cv, cw) = (p / mass, iw.inverse().expect("invertible") * l);
            let idx = self.add_cluster(s, comp.clone(), Some(parent.id), parent.pose, |_| (cv, cw));
            self.clusters[idx].explicit_window = parent.explicit_window;
            self.clusters[idx].window_utilization = parent.window_utilization;
            let child = self.clusters[idx].clone();
            // Hidden velocities: whatever the child's rigid motion does not carry.
            for &c in &comp {
                let (v, w) = vel[c];
                let r = parent.pose.transform_vector(lite[c].0 - child.com);
                let cs = &mut self.chunks[s][c];
                cs.v = rot.transpose() * (v - child.velocity - child.angular_velocity.cross(r));
                cs.w = rot.transpose() * (w - child.angular_velocity);
            }
            // Fold the mean hidden translation of a free child into its pose.
            if !child.anchored {
                let mean = comp.iter().map(|&c| self.chunks[s][c].u * lite[c].1).fold(Vec3::ZERO, |a, b| a + b)
                    / mass;
                for &c in &comp {
                    self.chunks[s][c].u -= mean;
                }
                self.clusters[idx].pose.position += parent.pose.transform_vector(mean);
            }
            if self.config.methods.exact_frame_transfer {
                // Every chunk keeps its world velocity and spin exactly: the lever arms
                // above leave out the hidden displacement (and the fold moves it), which
                // shifted velocities by `w x u` and changed the momentum at every split.
                let child = self.clusters[idx].clone();
                let back = child.pose.rotation.conjugate();
                for &c in &comp {
                    let (v, w) = vel[c];
                    let r = child.pose.transform_vector(lite[c].0 + self.chunks[s][c].u - child.com);
                    let cs = &mut self.chunks[s][c];
                    cs.v = back.rotate(v - child.velocity - child.angular_velocity.cross(r));
                    cs.w = back.rotate(w - child.angular_velocity);
                }
                // The child's hidden field holds whatever rigid motion it had inside the
                // parent, which can be finite (a loose piece turning on a cracked hinge):
                // fold it into the child's frame exactly, once.
                if !child.anchored {
                    self.fold_rigid_drift(idx, true);
                }
            }
            children.push(self.clusters[idx].id);
        }
        self.events.push(SolverEvent::Split { time: self.time, parent: parent.id, children });
    }

    // ------------------------------------------------------------------ events

    /// Remove chunks; the forces their bonds exerted on the rest are replaced by
    /// external loads that ramp to zero over `duration` (0 = sudden).
    pub fn remove_chunks(&mut self, s: usize, chunks: &[usize], duration: f64) {
        let removed: std::collections::HashSet<usize> =
            chunks.iter().copied().filter(|&c| self.chunks[s][c].active).collect();
        if removed.is_empty() {
            return;
        }
        for bi in 0..self.bonds[s].len() {
            let b = &self.bonds[s][bi];
            if !b.alive {
                continue;
            }
            let (a, bb) = (b.geometry.a, b.geometry.b);
            let ra = removed.contains(&a);
            let rb = removed.contains(&bb);
            if !(ra || rb) {
                continue;
            }
            if ra != rb {
                let (fa, ma, fb, mb) = self.bond_chunk_loads(s, bi, &b.force);
                let (kept, f, m) = if ra { (bb, fb, mb) } else { (a, fa, ma) };
                self.replacements.push(ReplacementLoad {
                    structure: s,
                    chunk: kept,
                    force: f,
                    moment: m,
                    start: self.time,
                    duration,
                });
            }
            self.bonds[s][bi].alive = false;
        }
        for &c in &removed {
            self.chunks[s][c].active = false;
            self.chunks[s][c].removed = true;
        }
        self.rebuild_adjacency(s);
        self.rebuild_structure_clusters(s);
    }

    /// Remove the supports of chunks, replacing their reactions by ramped loads.
    pub fn remove_supports(&mut self, s: usize, chunks: &[usize], duration: f64) {
        for &c in chunks {
            if self.structures[s].chunks[c].support == Support::None {
                continue;
            }
            let (f, m) = self.chunks[s][c].reaction;
            self.replacements.push(ReplacementLoad { structure: s, chunk: c, force: f, moment: m, start: self.time, duration });
            std::sync::Arc::make_mut(&mut self.structures[s]).chunks[c].support = Support::None;
            self.chunks[s][c].reaction = (Vec3::ZERO, Vec3::ZERO);
        }
        self.rebuild_structure_clusters(s);
    }

    /// Rebuild all clusters of a structure from current connectivity (keeps motion).
    fn rebuild_structure_clusters(&mut self, s: usize) {
        let affected: Vec<usize> = (0..self.clusters.len()).filter(|&i| self.clusters[i].structure == s).collect();
        for &ci in affected.iter().rev() {
            // Drop removed chunks from the cluster, then split/re-anchor by connectivity.
            let keep: Vec<usize> = self.clusters[ci].chunks.iter().copied().filter(|&c| self.chunks[s][c].active).collect();
            if keep.is_empty() {
                self.clusters.swap_remove(ci);
                if ci < self.clusters.len() {
                    let moved = self.clusters[ci].clone();
                    for &c in &moved.chunks {
                        self.chunks[moved.structure][c].cluster = ci;
                    }
                }
                continue;
            }
            let old = self.clusters[ci].clone();
            let pose = old.pose;
            let (v, w, com_w) = (old.velocity, old.angular_velocity, old.com_world());
            let comps = self.components(s, &keep);
            self.clusters.swap_remove(ci);
            if ci < self.clusters.len() {
                let moved = self.clusters[ci].clone();
                for &c in &moved.chunks {
                    self.chunks[moved.structure][c].cluster = ci;
                }
            }
            let mut children = Vec::new();
            for comp in comps {
                let idx = self.add_cluster(s, comp, Some(old.id), pose, |x| (v + w.cross(x - com_w), w));
                self.clusters[idx].explicit_window = old.explicit_window;
                self.clusters[idx].window_utilization = old.window_utilization;
                children.push(self.clusters[idx].id);
            }
            if children.len() > 1 {
                self.events.push(SolverEvent::Split { time: self.time, parent: old.id, children });
            }
        }
    }

    // ------------------------------------------------------------------ queries

    /// Cluster index of an active chunk.
    pub fn cluster_of(&self, s: usize, c: usize) -> Option<usize> {
        let st = &self.chunks[s][c];
        st.active.then_some(st.cluster)
    }

    pub fn structure_index(&self, name: &str) -> Option<usize> {
        self.structures.iter().position(|s| s.name == name)
    }

    /// Number of bonds broken by fracture (bonds of removed chunks do not count).
    pub fn broken_bond_count(&self) -> usize {
        self.bonds.iter().flatten().filter(|b| !b.joint.connected(b.rebar.is_some())).count()
    }

    /// Support reaction of a chunk in world frame.
    pub fn reaction_world(&self, s: usize, c: usize) -> (Vec3, Vec3) {
        let st = &self.chunks[s][c];
        if !st.active {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let cl = &self.clusters[st.cluster];
        (cl.pose.transform_vector(st.reaction.0), cl.pose.transform_vector(st.reaction.1))
    }
}

/// Smallest eigenvalue of a symmetric positive 3x3 matrix (closed form).
pub fn smallest_principal_moment(m: &Mat3) -> f64 {
    let a = &m.m;
    let p1 = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if p1 < 1e-30 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()).powi(2) {
        return a[0][0].min(a[1][1]).min(a[2][2]);
    }
    let q = m.trace() / 3.0;
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let b = (*m - Mat3::IDENTITY * q) * (1.0 / p);
    let r = (b.det() / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos()
}

/// `exp(phi) v - v` for the turn by the rotation vector `phi` (Rodrigues), without the
/// cancellation of the difference of a rotated (possibly long) vector and itself; zero
/// for no turn. world.slang `turn_difference`.
pub fn turn_difference(phi: Vec3, v: Vec3) -> Vec3 {
    let angle = phi.norm();
    if angle == 0.0 {
        return Vec3::ZERO;
    }
    let a = phi / angle;
    let s = (0.5 * angle).sin();
    let av = a.cross(v);
    av * angle.sin() + a.cross(av) * (2.0 * s * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `true_step`'s frequency bound on the cantilever (21 chunks, 126 degrees of
    /// freedom) lies between the highest eigenvalue of `K x = lambda M x`, found
    /// independently (dense cyclic Jacobi on `L^-1 K L^-T`, `M = L L^T` per chunk), and the
    /// Gershgorin bound. (A Ritz value plus its residual bounds only the eigenvalue
    /// nearest it: from this start it gave 1.37e8 against the true 1.99e8.)
    #[test]
    fn true_step_bounds_the_highest_frequency() {
        let mut scene = crate::builders::cantilever(20, "cantilever");
        scene.sim.methods.true_step = true;
        let sv = ReferenceSolver::new(&scene);
        let (bound, gershgorin) = sv.structure_frequency_bound(0).expect("bonds");
        let st = &sv.structures[0];
        let m = st.chunks.len();
        let n = 6 * m;
        // K's columns from unit displacements (full bond stiffness, rebar included).
        let mut k = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            let mut x = vec![(Vec3::ZERO, Vec3::ZERO); m];
            let mut e = [0.0; 6];
            e[j % 6] = 1.0;
            x[j / 6] = (Vec3::new(e[0], e[1], e[2]), Vec3::new(e[3], e[4], e[5]));
            for b in sv.bonds[0].iter().filter(|b| b.alive) {
                let g = &b.geometry;
                let d = g.kinematics(x[g.a].0, x[g.a].1, x[g.b].0, x[g.b].1);
                let k6 = ReferenceSolver::bond_stiffness6(b);
                let q = crate::bond::Local6 { lin: Vec3::new(k6[0] * d.lin.x, k6[1] * d.lin.y, k6[2] * d.lin.z), ang: Vec3::new(k6[3] * d.ang.x, k6[4] * d.ang.y, k6[5] * d.ang.z) };
                let (fa, ma, fb, mb) = g.chunk_loads(&q);
                for (c, f, mo) in [(g.a, fa, ma), (g.b, fb, mb)] {
                    for t in 0..3 {
                        k[6 * c + t][j] -= f.to_array()[t];
                        k[6 * c + 3 + t][j] -= mo.to_array()[t];
                    }
                }
            }
        }
        // L^-1 per chunk: 1/sqrt(m) and the inverse Cholesky factor of the inertia.
        let mut linv = vec![[[0.0f64; 6]; 6]; m];
        for c in 0..m {
            let ch = &st.chunks[c];
            let mu = sv.chunks[0][c].inertia_scale;
            for t in 0..3 {
                linv[c][t][t] = 1.0 / (ch.mass * mu).sqrt();
            }
            let a: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| ch.inertia.m[i][j] * mu));
            let mut l = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..=i {
                    let s: f64 = (0..j).map(|p| l[i][p] * l[j][p]).sum();
                    l[i][j] = if i == j { (a[i][i] - s).sqrt() } else { (a[i][j] - s) / l[j][j] };
                }
            }
            // Inverse of the lower-triangular factor.
            let mut li = [[0.0f64; 3]; 3];
            for i in 0..3 {
                li[i][i] = 1.0 / l[i][i];
                for j in 0..i {
                    let s: f64 = (j..i).map(|p| l[i][p] * li[p][j]).sum();
                    li[i][j] = -s / l[i][i];
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    linv[c][3 + i][3 + j] = li[i][j];
                }
            }
        }
        let lmat = |r: usize, c: usize| if r / 6 == c / 6 { linv[r / 6][r % 6][c % 6] } else { 0.0 };
        // A = L^-1 K L^-T.
        let mut tmp = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                tmp[i][j] = (6 * (i / 6)..6 * (i / 6) + 6).map(|p| lmat(i, p) * k[p][j]).sum();
            }
        }
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = (6 * (j / 6)..6 * (j / 6) + 6).map(|p| tmp[i][p] * lmat(j, p)).sum();
            }
        }
        // Cyclic Jacobi.
        for _ in 0..30 {
            let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
            if off < 1e-30 * (0..n).map(|i| a[i][i] * a[i][i]).sum::<f64>() {
                break;
            }
            for p in 0..n {
                for q in p + 1..n {
                    if a[p][q] == 0.0 {
                        continue;
                    }
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for r in 0..n {
                        let (arp, arq) = (a[r][p], a[r][q]);
                        a[r][p] = c * arp - s * arq;
                        a[r][q] = s * arp + c * arq;
                    }
                    for r in 0..n {
                        let (apr, aqr) = (a[p][r], a[q][r]);
                        a[p][r] = c * apr - s * aqr;
                        a[q][r] = s * apr + c * aqr;
                    }
                }
            }
        }
        let top = (0..n).map(|i| a[i][i]).fold(f64::MIN, f64::max);
        println!("top eigenvalue {top:.6e}, bound {bound:.6e}, Gershgorin {gershgorin:.6e}");
        assert!(bound >= top * (1.0 - 1e-9), "bound {bound:.6e} below the top eigenvalue {top:.6e}");
        assert!(bound <= gershgorin * (1.0 + 1e-12));
    }

    /// A free fragment and a 3-chunk free cluster at the same world place, their body
    /// origin either beside them or 25 m away (`at_origin`), each chunk with a finite
    /// hidden rotation and displacement, as a split leaves a child (house-1story rammed
    /// and fractured-bridge left 0.09 rad).
    fn fragments(origin_far: bool) -> ReferenceSolver {
        use crate::builders::*;
        let mut s = new_scene("fold", "rigid drift fold");
        s.gravity = [0.0; 3];
        s.materials.insert("concrete".into(), oracle_concrete(None));
        let world = Vec3::new(25.0, -3.0, 1.0);
        let (offset, position) = if origin_far { (world, [0.0; 3]) } else { (Vec3::ZERO, world.to_array()) };
        let mut one = body("one", grid(offset + Vec3::new(-0.5, -0.5, -0.5), [1, 1, 1], Vec3::splat(1.0), "concrete"), Vec::new());
        one.position = position;
        let chunks = grid(offset + Vec3::new(-1.5, 2.0, -0.5), [3, 1, 1], Vec3::splat(1.0), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut three = body("three", chunks, bonds);
        three.position = position;
        s.bodies = vec![one, three];
        let mut sv = ReferenceSolver::new(&s);
        for (k, st) in sv.chunks.iter_mut().flatten().enumerate() {
            let k = k as f64;
            st.th = Vec3::new(0.02 - 0.01 * k, 0.09, -0.01 + 0.005 * k);
            st.u = Vec3::new(1e-3 * k, -2e-3, 5e-4 * k);
        }
        sv
    }

    /// World centre and orientation of every chunk.
    fn chunk_poses(sv: &ReferenceSolver) -> Vec<(Vec3, Quat)> {
        let mut out = Vec::new();
        for s in 0..sv.structures.len() {
            for c in 0..sv.structures[s].chunks.len() {
                let st = &sv.chunks[s][c];
                let q = sv.clusters[st.cluster].pose.rotation * Quat::from_axis_angle(st.th, st.th.norm());
                out.push((sv.chunk_position(s, c), q));
            }
        }
        out
    }

    /// `remove_rigid_drift` changes the floating frame, not the chunks: it must not depend
    /// on where the body's origin lies (an arbitrary choice of the scene), and it leaves a
    /// one-chunk cluster where it is (its hidden rotation becomes the frame's exactly).
    /// The world positions and orientations after the fold are compared at f64 rounding
    /// (gamma_n of the coordinates involved, n = 64 operations).
    #[test]
    fn rigid_drift_fold_is_independent_of_the_body_origin() {
        let gamma = |n: f64| n * (f64::EPSILON / 2.0) / (1.0 - n * f64::EPSILON / 2.0);
        let mut after = Vec::new();
        for far in [false, true] {
            let mut sv = fragments(far);
            let before = chunk_poses(&sv);
            for ci in 0..sv.clusters.len() {
                sv.remove_rigid_drift(ci);
            }
            let now = chunk_poses(&sv);
            // The one-chunk cluster (structure 0): unchanged.
            let (x0, q0) = before[0];
            let (x1, q1) = now[0];
            let scale = x0.norm() + 30.0;
            assert!((x1 - x0).norm() <= gamma(64.0) * scale, "origin far {far}: the fold moved a one-chunk cluster by {:.3e} m", (x1 - x0).norm());
            let turn = 2.0 * (q0.conjugate() * q1).normalized().x.abs().max((q0.conjugate() * q1).normalized().y.abs()).max((q0.conjugate() * q1).normalized().z.abs());
            assert!(turn <= gamma(64.0), "origin far {far}: the fold turned a one-chunk cluster by {turn:.3e} rad");
            after.push((before, now));
        }
        // Every chunk: the same world pose before (by construction) and after the fold,
        // whichever the origin.
        let (near, far) = (&after[0], &after[1]);
        for k in 0..near.0.len() {
            let scale = near.0[k].0.norm() + 30.0;
            assert!((near.0[k].0 - far.0[k].0).norm() <= gamma(64.0) * scale, "chunk {k}: different start");
            let moved_near = near.1[k].0 - near.0[k].0;
            let moved_far = far.1[k].0 - far.0[k].0;
            assert!(
                (moved_near - moved_far).norm() <= gamma(64.0) * scale,
                "chunk {k}: the fold moved it by {:?} with the origin beside it, {:?} with the origin 25 m away",
                moved_near.to_array(),
                moved_far.to_array()
            );
        }
    }
}
