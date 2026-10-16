//! The scene format (`stress-scene/1`): the single description every solver runs.
//!
//! A scene is plain JSON. The reference solver loads it directly; each external
//! oracle has an exporter that turns the same file into that tool's input deck, so
//! both sides simulate the identical setup. Bond section properties and stiffnesses
//! are written into `derived` by [`Scene::with_derived`] for exporters that need
//! them; the solver recomputes them and ignores the field on load.
//!
//! Conventions: SI units, z-up (default gravity `[0, 0, -9.81]`), quaternions as
//! `[w, x, y, z]`. Chunk and bond geometry is in the owning body's frame.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::material::Material;
use crate::math::{Quat, Vec3};

pub const SCENE_FORMAT: &str = "stress-scene/1";

fn default_gravity() -> [f64; 3] {
    [0.0, 0.0, -9.81]
}
fn identity_quat() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}
fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub format: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// The validation benchmark this scene belongs to (spec table number), if any.
    #[serde(default)]
    pub benchmark: Option<u32>,
    #[serde(default = "default_gravity")]
    pub gravity: [f64; 3],
    pub materials: BTreeMap<String, Material>,
    pub bodies: Vec<BodyDesc>,
    #[serde(default)]
    pub impactors: Vec<ImpactorDesc>,
    #[serde(default)]
    pub ground: Option<GroundDesc>,
    #[serde(default)]
    pub loads: Vec<LoadDesc>,
    #[serde(default)]
    pub events: Vec<EventDesc>,
    pub sim: SimDesc,
    #[serde(default)]
    pub probes: Vec<ProbeDesc>,
    #[serde(default)]
    pub metrics: Vec<MetricDesc>,
    /// Free-form per-oracle settings (mesh density, element formulation, ...), keyed by oracle name.
    #[serde(default)]
    pub oracle: BTreeMap<String, serde_json::Value>,
}

/// One rigid cluster of chunks: a single rigid body until fracture splits it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyDesc {
    pub name: String,
    #[serde(default)]
    pub position: [f64; 3],
    #[serde(default = "identity_quat")]
    pub orientation: [f64; 4],
    #[serde(default)]
    pub linear_velocity: [f64; 3],
    #[serde(default)]
    pub angular_velocity: [f64; 3],
    pub chunks: Vec<ChunkDesc>,
    pub bonds: Vec<BondDesc>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Support {
    #[default]
    None,
    /// All six degrees of freedom held.
    Fixed,
    /// Translations held, rotations free.
    Pinned,
}

/// A box-shaped chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkDesc {
    pub center: [f64; 3],
    pub half_extents: [f64; 3],
    #[serde(default = "identity_quat")]
    pub orientation: [f64; 4],
    pub material: String,
    #[serde(default)]
    pub support: Support,
    /// Named groups for selection (events, probes, metrics, exporters).
    #[serde(default)]
    pub groups: Vec<String>,
    /// Pre-fracture level: 0 is the coarse structural level.
    #[serde(default)]
    pub level: u32,
    /// The coarser chunk this one refines, for level > 0.
    #[serde(default)]
    pub parent: Option<usize>,
    /// Convex hull chunk: up to 64 points in the chunk frame, centred on the chunk's
    /// centre of mass (`half_extents` then bound it). Absent: a box of `half_extents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hull: Option<Vec<[f64; 3]>>,
    /// Compound chunk (a non-convex solid): non-overlapping convex parts, each up to 64
    /// points in the chunk frame. Their union's centroid is the chunk's centre and
    /// `half_extents` bound them; mass and inertia are the parts' at the material's
    /// density. Contact is per part (penalty contact only: `layer_contact` off).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compound: Option<Vec<PartDesc>>,
}

/// One convex part of a compound chunk: up to 64 points in the chunk frame and, when
/// the asset gives them, its polygon faces (indices into `points`, counter-clockwise
/// seen from outside), which are then taken as given (`ConvexHull::from_faces`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartDesc {
    pub points: Vec<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<Vec<Vec<usize>>>,
}

impl PartDesc {
    pub fn hull(&self) -> Result<crate::hull::ConvexHull, String> {
        let pts: Vec<Vec3> = self.points.iter().map(|p| Vec3::from_array(*p)).collect();
        crate::hull::ConvexHull::of_part(&pts, self.faces.as_deref())
    }
}

/// A bond between two touching chunks of the same body.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BondDesc {
    pub a: usize,
    pub b: usize,
    /// Centre of the contact patch.
    pub centroid: [f64; 3],
    /// Unit normal of the contact patch, pointing from `a` to `b`.
    pub normal: [f64; 3],
    /// Unit tangent of the patch; `width[0]` is measured along it, `width[1]` along `normal x tangent`.
    pub tangent: [f64; 3],
    pub area: f64,
    pub width: [f64; 2],
    /// Material of the joint (strength, stiffness, fracture energy).
    pub material: String,
    #[serde(default)]
    pub rebar: Option<RebarSpec>,
    /// Effective (Euler) buckling length of the member this bond belongs to; enables the buckling cap.
    #[serde(default)]
    pub buckling_length: Option<f64>,
    #[serde(default)]
    pub level: u32,
    #[serde(default)]
    pub groups: Vec<String>,
    /// Exporter convenience, written by [`Scene::with_derived`]; ignored on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<serde_json::Value>,
}

/// Rebar crossing a bond: an elastic-plastic axial tie in parallel with the joint,
/// holding cracked concrete together until it ruptures.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RebarSpec {
    /// Total steel cross-section crossing the joint (m^2).
    pub area: f64,
    /// Material supplying E, the yield stress (`tensile_strength`) and the rupture
    /// energy (`fracture_energy.tension`, per unit steel area).
    pub material: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImpactorShape {
    Sphere { radius: f64 },
    Box { half_extents: [f64; 3] },
}

/// Crush law of a soft impactor (vehicle front, crate): the impactor's own
/// force-deformation curve, rigid-perfectly plastic. While its crushable energy lasts
/// the impactor deforms (its contact surface recedes) instead of transmitting more than
/// `max_force`; then it bottoms out and contacts with its material stiffness. This is a
/// material law of the impactor acting in its contact with the structure, not a cap on
/// rigid-body contacts in general. For vehicles the plateau force and the crush energy
/// (crush depth = energy / max_force) come from crash-test force-deformation curves;
/// EN 1991-1-7 Annex C idealises the same behaviour as an equivalent elastic stiffness
/// (`F = v sqrt(k m)`), which gives the same impulse with a different pulse shape.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrushDesc {
    pub max_force: f64,
    pub energy: f64,
}

/// A rigid, unbreakable projectile coupled to the stress solve through contact.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImpactorDesc {
    pub name: String,
    pub shape: ImpactorShape,
    pub mass: f64,
    pub position: [f64; 3],
    #[serde(default = "identity_quat")]
    pub orientation: [f64; 4],
    pub velocity: [f64; 3],
    #[serde(default)]
    pub angular_velocity: [f64; 3],
    /// Material supplying the contact stiffness and friction.
    pub material: String,
    #[serde(default)]
    pub crush: Option<CrushDesc>,
}

/// Horizontal ground plane `z = height`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundDesc {
    pub height: f64,
    pub friction: f64,
    pub material: String,
}

/// Scalar function of time.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeFunction {
    Constant { value: f64 },
    /// 0 before `t0`, linear to `value` at `t1`, then held.
    Ramp { t0: f64, t1: f64, value: f64 },
    /// `peak * sin(pi (t - start)/duration)` on `[start, start + duration]`, else 0.
    HalfSine { start: f64, duration: f64, peak: f64 },
    /// Friedlander blast waveform `peak (1 - s) exp(-decay s)`, `s = (t - arrival)/duration`, for `0 <= s <= 1`.
    Friedlander { arrival: f64, peak: f64, duration: f64, decay: f64 },
    /// Piecewise-linear `[t, value]` table, clamped at both ends.
    Table { points: Vec<[f64; 2]> },
}

impl TimeFunction {
    pub fn eval(&self, t: f64) -> f64 {
        match *self {
            TimeFunction::Constant { value } => value,
            TimeFunction::Ramp { t0, t1, value } => {
                if t <= t0 {
                    0.0
                } else if t >= t1 {
                    value
                } else {
                    value * (t - t0) / (t1 - t0)
                }
            }
            TimeFunction::HalfSine { start, duration, peak } => {
                if t < start || t > start + duration {
                    0.0
                } else {
                    peak * (std::f64::consts::PI * (t - start) / duration).sin()
                }
            }
            TimeFunction::Friedlander { arrival, peak, duration, decay } => {
                let s = (t - arrival) / duration;
                if !(0.0..=1.0).contains(&s) {
                    0.0
                } else {
                    peak * (1.0 - s) * (-decay * s).exp()
                }
            }
            TimeFunction::Table { ref points } => {
                if points.is_empty() {
                    return 0.0;
                }
                if t <= points[0][0] {
                    return points[0][1];
                }
                for w in points.windows(2) {
                    if t <= w[1][0] {
                        let f = (t - w[0][0]) / (w[1][0] - w[0][0]).max(1e-300);
                        return w[0][1] + f * (w[1][1] - w[0][1]);
                    }
                }
                points[points.len() - 1][1]
            }
        }
    }
}

/// Axis-aligned box in a body frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Region {
    pub fn contains(&self, p: Vec3) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }
}

/// Selects chunks of one body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ChunkSelector {
    All,
    Group(String),
    Indices(Vec<usize>),
    /// Chunks whose centre lies in the region (body frame).
    Region(Region),
}

impl ChunkSelector {
    pub fn matches(&self, index: usize, chunk: &ChunkDesc) -> bool {
        match self {
            ChunkSelector::All => true,
            ChunkSelector::Group(g) => chunk.groups.iter().any(|x| x == g),
            ChunkSelector::Indices(v) => v.contains(&index),
            ChunkSelector::Region(r) => r.contains(Vec3::from_array(chunk.center)),
        }
    }
    pub fn select(&self, body: &BodyDesc) -> Vec<usize> {
        body.chunks.iter().enumerate().filter(|(i, c)| self.matches(*i, c)).map(|(i, _)| i).collect()
    }
}

/// Scripted external loads.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum LoadDesc {
    /// Force (world-frame direction, magnitude `magnitude(t)`) on a chunk, applied at
    /// `point` (body frame; default the chunk centre).
    PointForce {
        body: String,
        chunk: usize,
        direction: [f64; 3],
        magnitude: TimeFunction,
        #[serde(default)]
        point: Option<[f64; 3]>,
    },
    /// Uniform pressure on the exposed faces of the selected chunks whose outward
    /// normal (body frame) matches `face_normal`. Positive pressure pushes into the face.
    Pressure { body: String, chunks: ChunkSelector, face_normal: [f64; 3], pressure: TimeFunction },
    /// Free-air spherical charge: empirical pressure-time pulse on every exposed face,
    /// with distance falloff, angle of incidence, shadowing and clearing (venting).
    Blast { position: [f64; 3], tnt_mass: f64, time: f64 },
}

/// Scripted structural changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventDesc {
    /// Remove chunks from a structure (e.g. a column). At `time` the forces their bonds
    /// exert on the rest are replaced by equal external forces, which ramp to zero over
    /// `duration` (0 = sudden removal). The alternate-path procedure of GSA/UFC 4-023-03.
    RemoveChunks { time: f64, duration: f64, body: String, chunks: ChunkSelector },
    /// Remove supports, with the same force-replacement ramp.
    RemoveSupports { time: f64, duration: f64, body: String, chunks: ChunkSelector },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveMode {
    /// Explicit central-difference substeps everywhere, every frame (ground truth).
    #[default]
    Explicit,
    /// Explicit for recently loaded clusters, quasi-static for settled ones, sleep when idle.
    Adaptive,
    /// Quasi-static equilibrium every frame with same-step cascade (no inertia in the stress solve).
    QuasiStatic,
    /// One implicit Newmark (average acceleration) step per frame: inertia and true
    /// stiffness at the frame step; failures cascade over steps (see `implicit.rs`).
    Implicit,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimDesc {
    pub duration: f64,
    #[serde(default = "default_frame_dt")]
    pub frame_dt: f64,
    /// Multiplier on every Young's and shear modulus (<= 1 lowers the cost of the explicit solve).
    #[serde(default = "default_one")]
    pub stiffness_scale: f64,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub solve_mode: SolveMode,
    /// Settle every structure under gravity (static equilibrium) before t = 0.
    #[serde(default = "default_true")]
    pub gravity_prestress: bool,
    /// Fraction of the critical explicit timestep used.
    #[serde(default = "default_courant")]
    pub courant_safety: f64,
    /// The stress (bond network) step's own fraction of its critical step, the contact
    /// step keeping `courant_safety`: the two converge at different fractions (a
    /// fracturing impact needs the penalty contact well resolved; the certified stress
    /// bound is a stability limit). None: `courant_safety` for both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stress_courant_safety: Option<f64>,
    /// Upper bound on the substep (for timestep-convergence studies).
    #[serde(default)]
    pub max_substep: Option<f64>,
    /// Allow damage and fracture (false = linear elastic).
    #[serde(default = "default_true")]
    pub fracture: bool,
    /// Probe sampling interval (defaults to `frame_dt`).
    #[serde(default)]
    pub sample_interval: Option<f64>,
    /// Multi-level pre-fracture: refine a coarse chunk when a bond touching it exceeds
    /// this utilization (None disables refinement; level-0 only).
    #[serde(default)]
    pub refine_utilization: Option<f64>,
    /// Coefficient of restitution of every penalty contact (sets the contact dashpot).
    #[serde(default = "default_restitution")]
    pub contact_restitution: f64,
    /// Contact friction: the smaller of the two materials' `friction`, unless overridden.
    #[serde(default)]
    pub contact_friction: Option<f64>,
    /// Selective mass scaling (explicit solve): chunks whose stable substep is below
    /// this get extra deformation inertia to reach it (an alternative to lowering
    /// `stiffness_scale`; see `ReferenceSolver::apply_mass_scaling`).
    #[serde(default)]
    pub mass_scaling_dt: Option<f64>,
    /// `SolveMode::Implicit`: the implicit step (defaults to `frame_dt`). It must resolve
    /// the periods whose dynamics matter (about 10 steps per period for a few percent).
    #[serde(default)]
    pub implicit_dt: Option<f64>,
    /// Model capabilities (all on by default); switch one off to see what it contributes.
    #[serde(default)]
    pub features: Features,
    /// Methods under evaluation, each against the established one (see `IMPROVEMENTS.md`).
    /// Written only when they differ from the built-in default, so a written scene
    /// follows `STRESS_METHODS`.
    #[serde(default, skip_serializing_if = "Methods::is_proven")]
    pub methods: Methods,
}

/// Switches selecting a replacement for an established method. Each defaults to the
/// established method until the A/B diff of its switch over the scene catalogue and the
/// test suite meets its bar (`IMPROVEMENTS.md`); then its default flips. Set like the
/// model switches: `--feature NAME=on|off`, `--set sim.methods.NAME=true`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Methods {
    /// Elastic-layer contact (`polytope.rs`): normal force from the overlap volume of
    /// the two shapes at its centroid, anchored stick-slip Coulomb friction (tangential
    /// and torsional), dashpots bounded by a damped stable substep. Off: sample-point
    /// penalty contact with regularised friction and per-point stiffness shares.
    pub layer_contact: bool,
    /// Stable-substep bound from the Gershgorin rows of the mass-normalised stiffness
    /// `M^-1/2 K M^-1/2` (unit-consistent). Off: rows of `K` mixing translational and
    /// rotational entries, divided by each row's own inertia, which changes with the
    /// unit of length (1.3-1.4x on the benchmark scenes for metres vs half-metres).
    pub scaled_step_bound: bool,
    /// A cracked joint's no-tension contact layer integrated exactly over its patch
    /// (compressed region clipped by the opening line, polygon moments). Off: a 6 x 6
    /// grid of springs at cell centres, whose outermost spring sits at 0.417 of the
    /// width (rocking capacity 17 % low) and whose bending stiffness is 2.8 % low.
    pub exact_joint_patch: bool,
    /// A joint's contact patch is evaluated only once it has cracked (item 2a): no
    /// friction slip is accumulated on an interface that does not exist yet. Off: the
    /// patch runs at every evaluation and commits slip even at zero damage.
    pub patch_after_damage: bool,
    /// The strain-rate filter (rate effects) integrated exactly over each step,
    /// `1 - exp(-dt / tau)`. Off: `min(dt / tau, 1)`, which passes the raw one-step rate
    /// whenever the step exceeds `tau` (implicit, quasi-static and settled steps).
    pub exact_rate_filter: bool,
    /// With `layer_contact`: the contact substep computed every frame from the contacts
    /// that can engage within it (a Gershgorin bound), infinite when nothing can touch.
    /// Off: once at the start, every chunk pressed against the stiffest material on all
    /// six faces whether or not anything ever touches it.
    pub frame_contact_step: bool,
    /// The floating frame's drift removal keeps every chunk's world position,
    /// orientation, velocity and spin exactly (they are re-expressed in the moved frame).
    /// Off: hidden displacements, rotations and velocities are corrected linearly, which
    /// moves chunks by O(phi^2) and their velocities by O(phi v) when the folded rotation
    /// `phi` is finite (the first fold after a split), changing angular momentum and
    /// energy by amounts that do not vanish with the substep.
    pub exact_frame_transfer: bool,
    /// With `layer_contact`: a chunk pair's contact crushes where the layer's pressure
    /// `k''_f s` exceeds the weaker material's compressive strength (its permanent
    /// indentation ratchets to the depth where the peak pressure is the strength). Off:
    /// the contact that replaces a broken joint has the joint's stiffness but unlimited
    /// strength, so hinges between fragments never crush (over-strong arching).
    pub contact_crushing: bool,
    /// A static (quasi-static and settling) solve that the secant iteration leaves
    /// unconverged is solved again from the same start by a globalised Newton: steps with
    /// each joint's consistent tangent (`JointModel::tangent`: the intact share elastic,
    /// the cracked share's active contact springs, sticking or sliding friction, rebar),
    /// else the secant, scaled back until the residual falls (Armijo). Off: the secant
    /// iteration alone, which converges only linearly and can diverge; where it fails the
    /// cluster keeps its last state even when an equilibrium exists
    /// (s_arch_keystone_removed in quasi_static: 109 of 1309 solves converged, against 2087
    /// of 2089 with the re-solve). Every solve the secant converges is unchanged, and
    /// either way a solve is accepted only by the same residual test.
    pub tangent_statics: bool,
    /// The explicit stress substep from a certified bound on the highest natural frequency
    /// of each structure's intact bond network: Collatz-Wielandt power iteration on
    /// `|L^-1 K L^-T|` (`M = L L^T`), every iterate an upper bound, and exact on networks
    /// bonded face to face (the cantilever: the eigenvalue itself, against 1.6x for
    /// Gershgorin). Every bond's dashpot-to-stiffness ratio is taken at the structure's
    /// largest. Damage, cracks and splits only remove stiffness, so the bound holds for
    /// every later fragment; a cluster takes the larger of it and its own Gershgorin
    /// step. Off: the Gershgorin rows alone. A step change: the truncation error grows
    /// with the step. Contact usually binds the step, so the gain is about 9 % of CPU over
    /// the catalogue (`scripts/fidelity.py`: 34/35 scenes pass every gameplay metric).
    pub true_step: bool,
    /// With `layer_contact`: the global contact substep from the densest contact network,
    /// a chunk pressed on all six faces by equal neighbours moving against it. Along each
    /// box axis `d` its two faces' normal layers and its four other faces' tangential
    /// layers act, `w^2 = 1.02 max_d (4 k_d + 4 sum_(e != d) k_t,e) / m`. The 2 % is the
    /// rotation coupling, from Bloch waves on a cube lattice (at most 1.25 %, at nu = 0).
    /// Off: `6 k / m`, which leaves out the tangential layers. The cube lattice's top
    /// eigenvalue is 7.48 k/m at nu = 0.15 (brick, mortar): on fully broken masonry that
    /// step holds at safety 0.9 but blows up at 0.95 (`examples/contact_lattice`, 12^3),
    /// a margin under 5 % where 0.9 means 10 %. On (the default): stable at 1.0, CPU
    /// +1 %.
    pub lattice_contact_step: bool,
    /// Implicit mode: a bonded cluster takes its implicit step at the end of every
    /// substep in which a contact acts on it, not only once per `implicit_dt`. Contacts
    /// act every substep while bonds advance once per implicit step, and in between the
    /// predictor moves the hidden DOFs under the contact loads with no bond resistance:
    /// an impactor meets a wall that gives way, then the step puts the wall back into it
    /// and the overlap is energy from nothing (a 0.3 m brick wall struck at 15 m/s
    /// reached chunks at 1e18 m/s). The mode's own condition is that `implicit_dt`
    /// resolves the dynamics that matter; a contact's do, while it lasts. Off: one step
    /// per `implicit_dt` whatever acts.
    pub implicit_contact_steps: bool,
    /// Implicit mode: explicit windows for impacts. At each frame's start, a bonded
    /// cluster that a body can strike within the frame (bounding spheres closing faster
    /// than the engine's resting band `|g| frame_dt`, gap under that speed bound times
    /// the frame) takes the frame's substeps with the explicit integrator, the
    /// reference, instead of the implicit step; the others stay implicit. Frame
    /// boundaries are implicit-step boundaries, so the hand-over meets a consistent
    /// state, and fragments of a windowed cluster stay explicit to the frame's end.
    /// Implicit mode at the frame step misses punch-through (b5 above 15-20 m/s) and,
    /// with substep contacts, leaks contact energy where its step relocates struck
    /// chunks; inside the window neither happens. The substep then also honours the
    /// windowed clusters' stable stress step. Off: every bonded cluster implicit.
    pub implicit_contact_windows: bool,
    /// Implicit mode: explicit windows for failure. A bonded cluster whose most utilised
    /// intact bond is failing (failure index at 1) or will reach 1 within two frames at
    /// its rate over the last frame takes the frame in the explicit integrator. Fracture
    /// runs on a chunk's wave transit; the implicit step commits damage once per step
    /// and spreads a cascade over steps (b9's blast breach starts 0.7 ms late at a 1 ms
    /// step, and breaks 224 bonds where the explicit solve breaks 60). Elastic dynamics
    /// stay implicit. Off: failure in the implicit step.
    pub implicit_failure_windows: bool,
    /// Geometrically exact (co-rotational) bond kinematics in the explicit integrator
    /// (`bond::CoRotation`): each chunk turns by the finite rotation of its rotation
    /// vector, and the bond deforms by its patch points' separation and relative rotation
    /// in their mean frame. The small-strain map is invariant under infinitesimal
    /// rotations only, so its forces turn the system by `sum (u_a - u_b) x f` (about
    /// 5e-4 of `|L0|` in a 40 m/s impact, DECISIONS.md 36); these forces balance moments
    /// about the deformed configuration and stay the gradient of the bond energy. The
    /// static and implicit solves keep the small-strain map. Off: small strain.
    pub corotational_bonds: bool,
    /// Rayleigh damping, `zeta(w) = alpha / (2 w) + beta w / 2`, fitted per anchored
    /// structure to the bonds' damping ratio at both its fundamental `w1` (from
    /// `fundamental_period`) and each bond's axial frequency `w_b`.
    /// - A mass-proportional dashpot `alpha = 2 zeta w1 w_b / (w1 + w_b)` acts on the
    ///   chunks' hidden (deformation) velocities, never on rigid motion.
    /// - Each bond's stiffness-proportional dashpot is scaled by `w_b / (w1 + w_b)`.
    ///
    /// The bond dashpots alone give a mode at `w` the ratio `zeta w / w_b`, so global
    /// modes are almost undamped: the 2.7 Hz cantilever measures about 1e-4 at a
    /// material 0.05. Both the explicit integrator and Newmark apply it, to anchored
    /// clusters, and with `corotational_bonds` to free clusters and fragments too, made
    /// self-equilibrated about the deformed configuration so it conserves momentum
    /// (DECISIONS.md 58-60). Off: stiffness-proportional dashpots only.
    pub rayleigh_damping: bool,
    /// Explicit multirate integration of the bond network (asynchronous variational
    /// integration, Lew et al. 2003, at power-of-two step ratios). Each frame, every
    /// bond of an explicitly integrated cluster takes the rate of its finer chunk: the
    /// largest power of two `r` with `r dt` within that chunk's own stable step (its
    /// Gershgorin rows, `chunk_stable_dts`, floored at the structure's `true_step` bound
    /// when that is on, combined with the chunk's own contact bound as `substep_dt`
    /// combines them, all at their safety factors). A bond is evaluated only at its own
    /// ticks and kicks both chunks with the impulse `w dt f` of the interval `w dt` to
    /// its next tick (its rate, cut at the frame's end, where every bond starts again).
    /// Chunks drift and take their external, inertial and mass-damping loads at every
    /// substep, so slivers (light chunks with stiff bonds) and their bonds run at the
    /// substep while the rest of the network runs at its own larger step. Every bond's
    /// impulse acts on both of its chunks at once with the balanced force and moment of
    /// the single-rate integrator, so linear and angular momentum are exactly those of
    /// single rate. The joint law runs on each bond's own interval (rate effects,
    /// fatigue). A step change, judged by `scripts/fidelity.py` against single rate:
    /// the truncation error of the coarse bonds grows with their step. Inert in
    /// implicit mode and in the engine-coupled frame (single rate there). Off: every
    /// bond at every substep.
    pub multirate: bool,
}

impl Default for Methods {
    /// The proven methods (`Methods::proven`), adjusted by the environment variable
    /// `STRESS_METHODS` (comma-separated: `NAME` or `NAME=on` switches one on, `-NAME` or
    /// `NAME=off` off, `established` starts from every established method): the A/B
    /// hook that runs the whole test suite and catalogue with switches changed, without
    /// editing any scene or test. Scenes that set `sim.methods` explicitly keep their
    /// setting.
    fn default() -> Self {
        let mut m = Methods::proven();
        if let Ok(list) = std::env::var("STRESS_METHODS") {
            for item in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
                if item == "established" {
                    m = Methods::established();
                    continue;
                }
                let (name, on) = match (item.strip_prefix('-'), item.split_once('=')) {
                    (Some(n), _) => (n, false),
                    (None, Some((n, v))) => (n, matches!(v, "on" | "true")),
                    (None, None) => (item, true),
                };
                m.set(name, on).unwrap_or_else(|e| panic!("STRESS_METHODS: {e}"));
            }
        }
        m
    }
}

impl Methods {
    /// Every switch at the established method.
    pub const fn established() -> Methods {
        Methods {
            layer_contact: false,
            scaled_step_bound: false,
            exact_joint_patch: false,
            patch_after_damage: false,
            exact_rate_filter: false,
            frame_contact_step: false,
            exact_frame_transfer: false,
            contact_crushing: false,
            tangent_statics: false,
            true_step: false,
            lattice_contact_step: false,
            implicit_contact_steps: false,
            implicit_contact_windows: false,
            implicit_failure_windows: false,
            corotational_bonds: false,
            rayleigh_damping: false,
            multirate: false,
        }
    }

    /// The default: the replacements that met their bar (IMPROVEMENTS.md, DECISIONS.md
    /// 40-41): the whole test suite's exact targets, the oracle goldens and the
    /// analytic benchmarks pass with them; the established methods fail the exact
    /// targets (contact equilibrium, sticking, Coulomb sliding and stopping, unit
    /// rescaling, convergence, angular momentum, motion kept through a split).
    pub const fn proven() -> Methods {
        Methods { layer_contact: true, scaled_step_bound: true, exact_joint_patch: true, patch_after_damage: true, exact_rate_filter: true, exact_frame_transfer: true, contact_crushing: true, tangent_statics: true, lattice_contact_step: true, implicit_contact_steps: true, implicit_contact_windows: true, implicit_failure_windows: true, rayleigh_damping: true, corotational_bonds: true, ..Methods::established() }
    }

    /// Every switch at the established method (whatever `STRESS_METHODS` says).
    pub fn is_established(&self) -> bool {
        *self == Methods::established()
    }

    /// The built-in default (`proven`), whatever `STRESS_METHODS` says.
    pub fn is_proven(&self) -> bool {
        *self == Methods::proven()
    }

    /// Names of every switch, in declaration order.
    pub const NAMES: [&'static str; 17] = ["layer_contact", "scaled_step_bound", "exact_joint_patch", "patch_after_damage", "exact_rate_filter", "frame_contact_step", "exact_frame_transfer", "contact_crushing", "tangent_statics", "true_step", "lattice_contact_step", "implicit_contact_steps", "implicit_contact_windows", "implicit_failure_windows", "corotational_bonds", "rayleigh_damping", "multirate"];

    /// Set a switch by name.
    pub fn set(&mut self, name: &str, on: bool) -> Result<(), String> {
        let slot = match name {
            "layer_contact" => &mut self.layer_contact,
            "scaled_step_bound" => &mut self.scaled_step_bound,
            "exact_joint_patch" => &mut self.exact_joint_patch,
            "patch_after_damage" => &mut self.patch_after_damage,
            "exact_rate_filter" => &mut self.exact_rate_filter,
            "frame_contact_step" => &mut self.frame_contact_step,
            "exact_frame_transfer" => &mut self.exact_frame_transfer,
            "contact_crushing" => &mut self.contact_crushing,
            "tangent_statics" => &mut self.tangent_statics,
            "true_step" => &mut self.true_step,
            "lattice_contact_step" => &mut self.lattice_contact_step,
            "implicit_contact_steps" => &mut self.implicit_contact_steps,
            "implicit_contact_windows" => &mut self.implicit_contact_windows,
            "implicit_failure_windows" => &mut self.implicit_failure_windows,
            "corotational_bonds" => &mut self.corotational_bonds,
            "rayleigh_damping" => &mut self.rayleigh_damping,
            "multirate" => &mut self.multirate,
            _ => return Err(format!("unknown method '{name}' (known: {})", Self::NAMES.join(", "))),
        };
        *slot = on;
        Ok(())
    }
}

/// Switches for the model's capabilities, all on by default.
///
/// Each switch removes one mechanism and leaves the others unchanged, so running a scene
/// with and without it shows what that mechanism contributes. A switch only disables
/// what the scene's materials provide (e.g. `rate_effects` does nothing for a material
/// without a DIF). Inertia in the stress solve is selected by [`SolveMode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Features {
    /// Strength rises with loading rate (dynamic increase factor).
    pub rate_effects: bool,
    /// Delayed failure under sustained load (static fatigue).
    pub static_fatigue: bool,
    /// Per-bond Weibull strengths (off: every bond at the mean strength).
    pub weibull: bool,
    /// Fracture-energy softening (off: a bond breaks the instant it reaches its
    /// strength and releases all its stored energy, like a threshold model).
    pub softening: bool,
    /// Cracked joints keep carrying compression and friction through a no-tension
    /// contact patch (off: the cracked share of a joint carries nothing).
    pub crack_contact: bool,
    /// Compressive crushing of bonds (off: unlimited compressive strength, as in an
    /// elastic continuum without a crushing law).
    pub crushing: bool,
    /// Euler buckling cap on bonds of slender members.
    pub buckling: bool,
    /// Rebar crossing bonds.
    pub rebar: bool,
    /// Inertial loads of a cluster's rigid motion on its chunks (linear, angular,
    /// centrifugal, Coriolis); off, chunks feel only applied loads, as if the cluster
    /// were at rest.
    pub rigid_motion_loads: bool,
    /// Blast: faces behind other chunks receive only the diffracted pressure.
    pub blast_shadowing: bool,
    /// Blast: reflected pressure clears from the free edges (and from new holes).
    pub blast_clearing: bool,
    /// Bond dashpots (light stiffness-proportional damping).
    pub damping: bool,
}

impl Default for Features {
    fn default() -> Self {
        Features {
            rate_effects: true,
            static_fatigue: true,
            weibull: true,
            softening: true,
            crack_contact: true,
            crushing: true,
            buckling: true,
            rebar: true,
            rigid_motion_loads: true,
            blast_shadowing: true,
            blast_clearing: true,
            damping: true,
        }
    }
}

impl Features {
    /// Names of every switch, in declaration order.
    pub const NAMES: [&'static str; 12] = [
        "rate_effects",
        "static_fatigue",
        "weibull",
        "softening",
        "crack_contact",
        "crushing",
        "buckling",
        "rebar",
        "rigid_motion_loads",
        "blast_shadowing",
        "blast_clearing",
        "damping",
    ];

    /// Set a switch by name.
    pub fn set(&mut self, name: &str, on: bool) -> Result<(), String> {
        let slot = match name {
            "rate_effects" => &mut self.rate_effects,
            "static_fatigue" => &mut self.static_fatigue,
            "weibull" => &mut self.weibull,
            "softening" => &mut self.softening,
            "crack_contact" => &mut self.crack_contact,
            "crushing" => &mut self.crushing,
            "buckling" => &mut self.buckling,
            "rebar" => &mut self.rebar,
            "rigid_motion_loads" => &mut self.rigid_motion_loads,
            "blast_shadowing" => &mut self.blast_shadowing,
            "blast_clearing" => &mut self.blast_clearing,
            "damping" => &mut self.damping,
            _ => return Err(format!("unknown feature '{name}' (known: {})", Self::NAMES.join(", "))),
        };
        *slot = on;
        Ok(())
    }
}

fn default_restitution() -> f64 {
    0.2
}

fn default_frame_dt() -> f64 {
    1.0 / 60.0
}
fn default_one() -> f64 {
    1.0
}
fn default_courant() -> f64 {
    0.5
}

impl SimDesc {
    /// Both step fractions scaled by `factor` (refinement and spread runs).
    pub fn scale_safety(&mut self, factor: f64) {
        self.courant_safety *= factor;
        if let Some(s) = self.stress_courant_safety.as_mut() {
            *s *= factor;
        }
    }
}

impl Default for SimDesc {
    fn default() -> Self {
        SimDesc {
            duration: 1.0,
            frame_dt: default_frame_dt(),
            stiffness_scale: 1.0,
            seed: 0,
            solve_mode: SolveMode::Explicit,
            gravity_prestress: true,
            courant_safety: default_courant(),
            stress_courant_safety: None,
            max_substep: None,
            fracture: true,
            sample_interval: None,
            refine_utilization: None,
            contact_restitution: default_restitution(),
            contact_friction: None,
            implicit_dt: None,
            mass_scaling_dt: None,
            features: Features::default(),
            methods: Methods::default(),
        }
    }
}

/// Generalised force component across a section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SectionComponent {
    /// Force along the section normal (tension positive).
    Normal,
    /// Force along `axis`.
    Force([f64; 3]),
    /// Moment about `axis` through the section point.
    Moment([f64; 3]),
}

/// A measured quantity, sampled over time. Directions are world-frame unless noted.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProbeKind {
    /// Displacement of a chunk centre from its initial position, along `axis`.
    ChunkDisplacement { body: String, chunk: usize, axis: [f64; 3] },
    ChunkVelocity { body: String, chunk: usize, axis: [f64; 3] },
    /// Internal force across the plane through `point` with `normal` (body frame),
    /// restricted to bonds whose centroid lies in `region`. Acts on the `+normal` side.
    SectionForce { body: String, point: [f64; 3], normal: [f64; 3], region: Option<Region>, component: SectionComponent },
    /// Sum of the support reactions of the selected chunks, along `axis`.
    Reaction { body: String, chunks: ChunkSelector, axis: [f64; 3] },
    ImpactorVelocity { impactor: String, axis: [f64; 3] },
    ImpactorPosition { impactor: String, axis: [f64; 3] },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
// `flatten` cannot be combined with `deny_unknown_fields`.
pub struct ProbeDesc {
    pub name: String,
    #[serde(flatten)]
    pub kind: ProbeKind,
}

/// How a metric is compared against an oracle.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Tolerance {
    /// `|ours - oracle| <= rel * |oracle|`.
    Relative(f64),
    Absolute(f64),
    /// Booleans and categories must be equal.
    Exact,
    /// Recorded, not gated.
    Report,
}

/// Quantities derived from an observation (probe series plus final chunk states).
/// Computed by the same code for every solver (see `metrics.rs`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricKind {
    ProbeFinal { probe: String },
    /// Max |value| over the run (the solver tracks it every substep).
    ProbePeak { probe: String },
    /// Largest / smallest value over the run (signed, envelope-aware).
    ProbeMax { probe: String },
    ProbeMin { probe: String },
    /// Max |value| of `probe` over max |value| of `reference`.
    ProbePeakRatioOf { probe: String, reference: String },
    /// `distance / (t_b - t_a)` where `t_x` is the first time |probe x| reaches `threshold`.
    WaveSpeed { probe_a: String, probe_b: String, distance: f64, threshold: f64 },
    /// Max |value| over the run divided by |value| at `reference_time`.
    ProbePeakRatio { probe: String, reference_time: f64 },
    /// Value at a time (linear interpolation of samples).
    ProbeAt { probe: String, time: f64 },
    /// First time |value| reaches `threshold`.
    ProbeArrival { probe: String, threshold: f64 },
    /// Dominant frequency (Hz) of the probe about its mean after `after`, from zero crossings.
    ProbeFrequency { probe: String, after: f64 },
    /// Initial minus final value.
    ProbeDrop { probe: String },
    /// Dynamic amplification `(peak - v(before)) / (v(end) - v(before))` of a load change
    /// at `before`, with the peak taken after `before` (sign-aware).
    DynamicAmplification { probe: String, before: f64 },
    /// Projected area (perpendicular to `axis`) of detached chunks of `body` within `region`.
    DetachedArea { body: String, region: Option<Region>, axis: [f64; 3] },
    DetachedMass { body: String, region: Option<Region> },
    /// Whether any chunk of `body` in `region` detached.
    DetachedAny { body: String, region: Option<Region> },
    /// Mass-weighted mean (or max) speed along `axis` of detached chunks in `region`; 0 if none.
    DetachedSpeed { body: String, region: Option<Region>, axis: [f64; 3], max: bool },
    /// Mass-weighted mean velocity along `axis` of every (non-removed) chunk of `body`
    /// that started in `region`, attached or not.
    RegionSpeed { body: String, region: Region, axis: [f64; 3] },
    /// Whether the chunks that started in `outer` move away from those in `inner` (along
    /// `axis`) faster than `threshold` m/s at the end: a layer separating across a crack
    /// plane, whether or not ligaments elsewhere still connect it (e.g. a spall layer).
    Separating { body: String, outer: Region, inner: Region, axis: [f64; 3], threshold: f64 },
    /// "hole", "push_over" or "intact" (see `metrics::failure_mode`).
    FailureMode { body: String, impact_region: Region },
    /// Boolean flag recorded by the solver (e.g. `collapse`, `any_bond_broken`).
    Flag { flag: String },
    /// Scalar recorded by the solver (e.g. `bond_dissipation`).
    Value { key: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
// `flatten` cannot be combined with `deny_unknown_fields`.
pub struct MetricDesc {
    pub name: String,
    #[serde(flatten)]
    pub kind: MetricKind,
    /// Tolerance against the analytic `expected` value (and against oracles unless
    /// `oracle_tolerance` is given).
    pub tolerance: Tolerance,
    /// Tolerance against oracle observations when it differs (another formulation:
    /// a continuum or a different beam theory converges to a slightly different value).
    #[serde(default)]
    pub oracle_tolerance: Option<Tolerance>,
    /// Which oracles this metric is compared against ("analytic" uses `expected`).
    #[serde(default)]
    pub oracles: Vec<String>,
    /// Closed-form expected value, for analytic benchmarks.
    #[serde(default)]
    pub expected: Option<serde_json::Value>,
}

impl MetricDesc {
    pub fn with_oracle_tolerance(mut self, t: Tolerance) -> MetricDesc {
        self.oracle_tolerance = Some(t);
        self
    }

    /// The tolerance against a reference (`"analytic"` or an oracle's name).
    pub fn tolerance_for(&self, source: &str) -> Tolerance {
        if source == "analytic" {
            self.tolerance
        } else {
            self.oracle_tolerance.unwrap_or(self.tolerance)
        }
    }
}

impl Scene {
    pub fn from_json(text: &str) -> Result<Scene, String> {
        let scene: Scene = serde_json::from_str(text).map_err(|e| format!("scene parse error: {e}"))?;
        scene.validate()?;
        Ok(scene)
    }

    pub fn load(path: &std::path::Path) -> Result<Scene, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Scene::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("scene serializes")
    }

    /// The scene with one field replaced: `path` is a dotted path into the scene's JSON
    /// (object keys and array indices, e.g. `sim.stiffness_scale`, `impactors.0.velocity`,
    /// `materials.concrete.tensile_strength`) and `value` is JSON (a bare word that is not
    /// valid JSON is taken as a string, so `sim.solve_mode=adaptive` works).
    pub fn with_override(&self, path: &str, value: &str) -> Result<Scene, String> {
        let mut root = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let new: serde_json::Value =
            serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
        let mut node = &mut root;
        for key in path.split('.') {
            node = match node {
                serde_json::Value::Object(map) => map.entry(key.to_string()).or_insert(serde_json::Value::Null),
                serde_json::Value::Array(items) => {
                    let i: usize = key.parse().map_err(|_| format!("'{path}': '{key}' is not an array index"))?;
                    let len = items.len();
                    items.get_mut(i).ok_or_else(|| format!("'{path}': index {i} out of range ({len})"))?
                }
                _ => return Err(format!("'{path}': '{key}' is inside a non-container value")),
            };
        }
        *node = new;
        let scene: Scene = serde_json::from_value(root).map_err(|e| format!("'{path}={value}': {e}"))?;
        scene.validate()?;
        Ok(scene)
    }

    /// The scene with a model feature switched on or off (see [`Features`]).
    /// The scene with a model switch ([`Features`]) or a method switch ([`Methods`]) set.
    pub fn with_feature(&self, name: &str, on: bool) -> Result<Scene, String> {
        let mut scene = self.clone();
        if Methods::NAMES.contains(&name) {
            scene.sim.methods.set(name, on)?;
        } else {
            scene.sim.features.set(name, on).map_err(|e| format!("{e}; methods: {}", Methods::NAMES.join(", ")))?;
        }
        Ok(scene)
    }

    pub fn material(&self, name: &str) -> &Material {
        &self.materials[name]
    }

    pub fn body(&self, name: &str) -> Option<(usize, &BodyDesc)> {
        self.bodies.iter().enumerate().find(|(_, b)| b.name == name)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.format != SCENE_FORMAT {
            return Err(format!("unsupported scene format '{}', expected '{SCENE_FORMAT}'", self.format));
        }
        for (name, m) in &self.materials {
            m.validate(name)?;
        }
        let known = |m: &str| -> Result<(), String> {
            if self.materials.contains_key(m) {
                Ok(())
            } else {
                Err(format!("unknown material '{m}'"))
            }
        };
        let mut names = std::collections::BTreeSet::new();
        for body in &self.bodies {
            if !names.insert(body.name.clone()) {
                return Err(format!("duplicate body name '{}'", body.name));
            }
            for (i, c) in body.chunks.iter().enumerate() {
                known(&c.material)?;
                if c.half_extents.iter().any(|h| !(*h > 0.0)) {
                    return Err(format!("body '{}' chunk {i}: half extents must be positive", body.name));
                }
                if let Some(points) = &c.hull {
                    let pts: Vec<Vec3> = points.iter().map(|p| Vec3::from_array(*p)).collect();
                    let hull = crate::hull::ConvexHull::new(&pts).map_err(|e| format!("body '{}' chunk {i}: {e}", body.name))?;
                    let um = hull.unit_mass();
                    let size = hull.half_extents().norm();
                    if um.centroid.norm() > 1e-6 * size {
                        return Err(format!("body '{}' chunk {i}: hull centroid {:?} is not at the chunk centre", body.name, um.centroid));
                    }
                    let h = hull.half_extents();
                    if (0..3).any(|k| h[k] > c.half_extents[k] * (1.0 + 1e-9)) {
                        return Err(format!("body '{}' chunk {i}: half extents do not bound the hull", body.name));
                    }
                    if c.parent.is_some() || body.chunks.iter().any(|o| o.parent == Some(i)) {
                        return Err(format!("body '{}' chunk {i}: hull chunks cannot be pre-fractured levels", body.name));
                    }
                }
                if let Some(parts) = &c.compound {
                    let err = |m: String| format!("body '{}' chunk {i}: {m}", body.name);
                    if c.hull.is_some() || parts.is_empty() {
                        return Err(err("a compound chunk has one or more parts and no hull".into()));
                    }
                    if self.sim.methods.layer_contact {
                        return Err(err("compound chunks take penalty contact: turn layer_contact off".into()));
                    }
                    if c.parent.is_some() || body.chunks.iter().any(|o| o.parent == Some(i)) {
                        return Err(err("compound chunks cannot be pre-fractured levels".into()));
                    }
                    let (mut volume, mut moment, mut size) = (0.0, Vec3::ZERO, 0.0f64);
                    for (k, part) in parts.iter().enumerate() {
                        let hull = part.hull().map_err(|e| err(format!("part {k}: {e}")))?;
                        let um = hull.unit_mass();
                        volume += um.volume;
                        moment += um.centroid * um.volume;
                        let h = hull.half_extents();
                        size = size.max(h.norm());
                        if (0..3).any(|a| h[a] > c.half_extents[a] * (1.0 + 1e-9)) {
                            return Err(err(format!("half extents do not bound part {k}")));
                        }
                    }
                    if (moment / volume).norm() > 1e-6 * size {
                        return Err(err(format!("parts' centroid {:?} is not at the chunk centre", moment / volume)));
                    }
                }
                if let Some(p) = c.parent {
                    if p >= body.chunks.len() || body.chunks[p].level + 1 != c.level {
                        return Err(format!("body '{}' chunk {i}: bad parent {p}", body.name));
                    }
                }
            }
            for (i, b) in body.bonds.iter().enumerate() {
                known(&b.material)?;
                if b.a >= body.chunks.len() || b.b >= body.chunks.len() || b.a == b.b {
                    return Err(format!("body '{}' bond {i}: bad chunk indices", body.name));
                }
                if !(b.area > 0.0) || b.width.iter().any(|w| !(*w > 0.0)) {
                    return Err(format!("body '{}' bond {i}: area and width must be positive", body.name));
                }
                let n = Vec3::from_array(b.normal);
                let t = Vec3::from_array(b.tangent);
                if (n.norm() - 1.0).abs() > 1e-6 || (t.norm() - 1.0).abs() > 1e-6 || n.dot(t).abs() > 1e-6 {
                    return Err(format!("body '{}' bond {i}: normal/tangent must be orthonormal", body.name));
                }
                if let Some(r) = &b.rebar {
                    known(&r.material)?;
                }
                let la = body.chunks[b.a].level;
                let lb = body.chunks[b.b].level;
                if la != b.level || lb != b.level {
                    return Err(format!("body '{}' bond {i}: chunks must be on the bond's level", body.name));
                }
            }
        }
        for imp in &self.impactors {
            known(&imp.material)?;
            if !(imp.mass > 0.0) {
                return Err(format!("impactor '{}': mass must be positive", imp.name));
            }
        }
        if let Some(g) = &self.ground {
            known(&g.material)?;
        }
        let body_known = |b: &str| -> Result<(), String> {
            if names.contains(b) {
                Ok(())
            } else {
                Err(format!("unknown body '{b}'"))
            }
        };
        for l in &self.loads {
            match l {
                LoadDesc::PointForce { body, .. } | LoadDesc::Pressure { body, .. } => body_known(body)?,
                LoadDesc::Blast { .. } => {}
            }
        }
        for e in &self.events {
            match e {
                EventDesc::RemoveChunks { body, .. } | EventDesc::RemoveSupports { body, .. } => body_known(body)?,
            }
        }
        let mut probe_names = std::collections::BTreeSet::new();
        for p in &self.probes {
            if !probe_names.insert(p.name.clone()) {
                return Err(format!("duplicate probe '{}'", p.name));
            }
        }
        if !(self.sim.frame_dt > 0.0 && self.sim.duration >= 0.0 && self.sim.stiffness_scale > 0.0) {
            return Err("sim: frame_dt and stiffness_scale must be positive, duration non-negative".into());
        }
        Ok(())
    }

    /// Copy with each bond's `derived` section properties filled in (for exporters).
    pub fn with_derived(&self) -> Scene {
        let mut s = self.clone();
        for (body_index, body) in s.bodies.iter_mut().enumerate() {
            let chunks = body.chunks.clone();
            for (bond_index, bond) in body.bonds.iter_mut().enumerate() {
                let m = &self.materials[&bond.material];
                let g = crate::bond::BondGeometry::from_desc(bond, &chunks);
                let k = crate::bond::BondStiffness::new(&g, m, 1.0);
                let weibull = crate::material::bond_strength_factor(m.weibull_modulus, self.sim.seed, body_index, bond_index);
                bond.derived = Some(serde_json::json!({
                    "length": g.length,
                    "i_t1": g.i_t1, "i_t2": g.i_t2, "torsion_constant": g.torsion_constant,
                    "section_modulus_t1": g.s_t1, "section_modulus_t2": g.s_t2,
                    "torsion_modulus": g.torsion_modulus,
                    "kn": k.kn, "ks": k.ks, "kb_t1": k.kb_t1, "kb_t2": k.kb_t2, "kt": k.kt,
                    "weibull": weibull,
                }));
            }
        }
        s
    }
}

/// Mass and body-frame inertia (about its centre) of a box chunk.
pub fn box_mass_inertia(half_extents: [f64; 3], orientation: [f64; 4], density: f64) -> (f64, crate::math::Mat3) {
    let h = Vec3::from_array(half_extents);
    let mass = density * 8.0 * h.x * h.y * h.z;
    let (a, b, c) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
    let local = Vec3::new(b * b + c * c, a * a + c * c, a * a + b * b) * (mass / 12.0);
    let r = Quat::from_wxyz(orientation).to_mat3();
    (mass, r * crate::math::Mat3::diag(local) * r.transpose())
}

/// Observation summary written by every solver; see `observation.rs`.
pub use crate::observation::Observation;
