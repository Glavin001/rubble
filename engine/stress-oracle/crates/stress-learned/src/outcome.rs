//! The impact-outcome recorder (plan P1, `learned_impact_outcome_record`): one event's whole
//! life cycle, frame by frame, with on every frame the model's causal inputs, the oracle's
//! outcome and the baselines' outcomes of the same frame-start state.
//!
//! Per frame `f` (schema `learned-impact-frames/1`, `learned-impact/FRAMES_SCHEMA.md`; P2's per-call
//! `learned-impact-outcome/1` records are derived from it):
//! - **inputs, taken before the step**: the full state (`state_*`, index `f`), the routing's
//!   predicted islands, every impactor contact the routing sweep predicts (time within the
//!   frame, closing speed and direction, reduced mass, Hertz time and peak force), the
//!   punch's plug and bonds, and the QS prior: a converged static solve of the oracle's own
//!   bond model under each predicted contact's Hertz peak force (run on a copy);
//! - **the oracle's outcome**: breaks with their time bin within the frame, each bond's peak
//!   failure index over the frame, the frame-end state (`state_*`, index `f + 1`), the
//!   impactors' contact timing, the energy ledger and a phase tag;
//! - **baselines on the same state**: `blast_parity`'s quasi-static frame (`qs_*`: the
//!   state folded to rigid bodies, then `blast_parity`'s model and switches for one frame)
//!   and the punch's (`punch_*`: the same, with the punch's bonds broken first), recorded on
//!   the frames where the punch breaks something.
//!
//! A recording runs until every impactor has stopped or left the scene's bounds, no bond has
//! broken for `quiet_frames` frames and the kinetic energy left in the bounds is below
//! `ke_fraction` of the input energy (or every bonded cluster there sleeps). `max_seconds`
//! caps it, and an event that hits the cap is `truncated`. Anything nonfinite, a change of
//! the bond or chunk set (refinement), or energy far beyond the input fails the event.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use stress_ref::math::{Quat, Vec3};
use stress_ref::scene::Scene;
use stress_ref::solver::{Activity, ChunkLoads, SolverEvent};
use stress_ref::statics::StaticOptions;
use stress_ref::world::{tiered_impact_speed, World};

use crate::catalog::{self, Catalog};
use crate::oracle_config::{qs_template, OracleConfig, QS_BASELINE};

pub const OUTCOME_SCHEMA: &str = "learned-impact-frames/1";
/// Bumped whenever what the recorder writes, or how, changes.
pub const RECORDER_REVISION: u32 = 9;
/// Break-time bins within a frame.
pub const TIME_BINS: usize = 8;
/// Floats per predicted contact (`contact_f32`).
pub const CONTACT_F32: usize = 17;
/// Floats per impactor state: position, orientation (w, x, y, z), velocity, angular velocity.
pub const IMPACTOR_STATE: usize = 13;
/// Floats per impactor contact summary (`impactor_contact`).
pub const IMPACTOR_CONTACT: usize = 4;

/// Phase tags, in `frames.json` and `frame_phase`.
pub const PHASES: [&str; 5] = ["pre_contact", "contact", "secondary", "separation", "settling"];

/// One scenario as the generator plans it (one line of `recipes.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeRecipe {
    pub event_id: String,
    pub family: String,
    /// Multi-load tier (plan P1): 1 one fast impactor, 2 two to four.
    pub tier: u32,
    /// Train/evaluation partition unit: descendants of one physical recipe share it.
    pub split_group: String,
    /// Held-out unit for whole layouts.
    pub layout_group: String,
    pub parameters: Value,
    /// The scene, before the oracle configuration is applied.
    pub scene: Scene,
}

/// The stopping rule and caps (the plan's "whole life cycle").
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StopRule {
    pub max_seconds: f64,
    pub quiet_frames: u32,
    pub ke_fraction: f64,
    /// An impactor slower than this (m/s) has stopped.
    pub stop_speed: f64,
    /// Bounds margin: max(this, `margin_fraction` x the bounds' largest extent) (m).
    pub margin: f64,
    pub margin_fraction: f64,
}

impl Default for StopRule {
    fn default() -> Self {
        StopRule { max_seconds: 10.0, quiet_frames: 30, ke_fraction: 0.01, stop_speed: 0.05, margin: 0.3, margin_fraction: 0.25 }
    }
}

#[derive(Clone, Debug)]
pub struct RecordOptions {
    pub oracle: OracleConfig,
    pub storage: Storage,
    pub stop: StopRule,
    /// Record the QS and punch baselines (on by default; off only for cost measurement).
    pub baselines: bool,
    /// Kinetic energy above this multiple of the input energy (plus gravity's work bound)
    /// fails the event as nonphysical.
    pub energy_blowup: f64,
}

impl RecordOptions {
    pub fn new(oracle: OracleConfig) -> RecordOptions {
        RecordOptions { oracle, storage: Storage::default(), stop: StopRule::default(), baselines: true, energy_blowup: 10.0 }
    }
}

/// A progress report from inside a running event (the generator rate-limits these into
/// the 30 s heartbeat).
#[derive(Clone, Debug)]
pub struct Beat<'a> {
    pub event_id: &'a str,
    pub frame: usize,
    pub time: f64,
    pub phase: &'a str,
    pub breaks: usize,
}

/// What an event's recording came to (`outcome.json`, and the generator's line per event).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeSummary {
    pub event_id: String,
    pub family: String,
    pub tier: u32,
    pub frames: usize,
    pub sim_seconds: f64,
    pub truncated: bool,
    pub stop_reason: String,
    pub broken_bonds: usize,
    pub fragments: usize,
    pub largest_fraction: f64,
    pub last_break_time: Option<f64>,
    /// Bonds that broke in the last 0.5 s of the recording (0 unless truncated).
    pub breaks_last_half_second: usize,
    pub input_energy: f64,
    pub impactors: Vec<Value>,
    pub frames_per_phase: BTreeMap<String, usize>,
    /// Wall-clock seconds: whole event, the oracle's steps, the QS branch, the punch
    /// branch, the QS prior and the recording overhead (state capture, routing, IO).
    pub cost: BTreeMap<String, f64>,
    pub oracle_substeps: u64,
    pub fingerprint: String,
}

/// Growable per-frame arrays, written through `catalog`.
#[derive(Default)]
pub(crate) struct Arrays {
    pub(crate) f32s: BTreeMap<String, (Vec<usize>, Vec<f32>, usize)>,
    pub(crate) i32s: BTreeMap<String, (Vec<usize>, Vec<i32>, usize)>,
    pub(crate) f64s: BTreeMap<String, (Vec<usize>, Vec<f64>, usize)>,
}

impl Arrays {
    pub(crate) fn f(&mut self, name: &str, tail: &[usize], data: &[f32]) -> Result<(), String> {
        let n: usize = tail.iter().product();
        if data.len() != n {
            return Err(format!("{name}: {} values, want {n}", data.len()));
        }
        if let Some(x) = data.iter().find(|x| !x.is_finite()) {
            return Err(format!("{name}: nonfinite value {x}"));
        }
        let e = self.f32s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
        if e.0 != tail {
            return Err(format!("{name}: shape changed {:?} -> {tail:?}", e.0));
        }
        e.1.extend_from_slice(data);
        e.2 += 1;
        Ok(())
    }

    pub(crate) fn i(&mut self, name: &str, tail: &[usize], data: &[i32]) -> Result<(), String> {
        let n: usize = tail.iter().product();
        if data.len() != n {
            return Err(format!("{name}: {} values, want {n}", data.len()));
        }
        let e = self.i32s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
        if e.0 != tail {
            return Err(format!("{name}: shape changed {:?} -> {tail:?}", e.0));
        }
        e.1.extend_from_slice(data);
        e.2 += 1;
        Ok(())
    }

    /// An exact f64 row (state where f32 loses digits, break times, impulses).
    pub(crate) fn d(&mut self, name: &str, tail: &[usize], data: &[f64]) -> Result<(), String> {
        let n: usize = tail.iter().product();
        if data.len() != n {
            return Err(format!("{name}: {} values, want {n}", data.len()));
        }
        if let Some(x) = data.iter().find(|x| !x.is_finite()) {
            return Err(format!("{name}: nonfinite value {x}"));
        }
        let e = self.f64s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
        if e.0 != tail {
            return Err(format!("{name}: shape changed {:?} -> {tail:?}", e.0));
        }
        e.1.extend_from_slice(data);
        e.2 += 1;
        Ok(())
    }

    pub(crate) fn declare_d(&mut self, name: &str, tail: &[usize]) {
        self.f64s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
    }

    /// Declare an array that may get no rows (a table that can be empty).
    pub(crate) fn declare_f(&mut self, name: &str, tail: &[usize]) {
        self.f32s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
    }

    pub(crate) fn declare_i(&mut self, name: &str, tail: &[usize]) {
        self.i32s.entry(name.into()).or_insert_with(|| (tail.to_vec(), Vec::new(), 0));
    }

    /// Write every array (`compress`: one zstd frame per file, `.zst`), keeping only the
    /// rows of `sel`'s frames when given: per-frame arrays (`frame_*`, `qs_*`) their kept
    /// frames, `state_*` the frame boundaries those frames need (f and f + 1), the punch
    /// branch and the contact and pair tables the rows of kept frames. Static arrays and the
    /// trajectory sidecar (`traj_*`) are always whole.
    fn write(&self, root: &Path, sel: Option<&Selection>, compress: bool) -> Result<Catalog, String> {
        let frame_col = |name: &str| -> Option<Vec<i32>> {
            let (_, d, rows) = self.i32s.get(name)?;
            let w = d.len() / (*rows).max(1);
            Some((0..*rows).map(|r| d[r * w]).collect())
        };
        let contact_frames = frame_col("contact_i32");
        let pair_frames = frame_col("pair_i32");
        let punch_frames = frame_col("punch_frame");
        let break_frames = frame_col("break_i32");
        let peak_frames = frame_col("peak_i32");
        let mid_frames = frame_col("mid_frame");
        let keep_rows = |name: &str, rows: usize| -> Option<Vec<usize>> {
            let sel = sel?;
            let kept = |frames: &Option<Vec<i32>>| frames.as_ref().map(|v| (0..rows).filter(|&r| sel.keeps(v[r] as usize)).collect());
            if name.starts_with("static_") || name.starts_with("traj_") {
                None
            } else if name.starts_with("state_") {
                Some(sel.state_rows.clone())
            } else if name.starts_with("frame_") || name.starts_with("qs_") {
                Some(sel.frames.clone())
            } else if name.starts_with("punch_") {
                kept(&punch_frames)
            } else if name.starts_with("contact_") {
                kept(&contact_frames)
            } else if name.starts_with("pair_") {
                kept(&pair_frames)
            } else if name.starts_with("break_") {
                kept(&break_frames)
            } else if name.starts_with("peak_") {
                kept(&peak_frames)
            } else if name.starts_with("mid_") {
                kept(&mid_frames)
            } else {
                None
            }
        };
        let mut cat = Catalog::new();
        let mut put = |name: &str, dtype: &str, tail: &[usize], rows: usize, bytes: Vec<u8>, width: usize| -> Result<(), String> {
            let (bytes, rows) = match keep_rows(name, rows) {
                Some(keep) => (keep.iter().flat_map(|&r| bytes[r * width..(r + 1) * width].to_vec()).collect(), keep.len()),
                None => (bytes, rows),
            };
            let shape: Vec<usize> = std::iter::once(rows).chain(tail.iter().copied()).collect();
            let rel = format!("arrays/{name}.{dtype}{}", if compress { ".zst" } else { "" });
            if compress {
                catalog::write_zstd(root, &rel, &bytes)?;
            } else {
                std::fs::create_dir_all(root.join("arrays")).map_err(|e| e.to_string())?;
                std::fs::write(root.join(&rel), &bytes).map_err(|e| format!("{rel}: {e}"))?;
            }
            cat.insert(name, dtype, &shape, &rel);
            Ok(())
        };
        for (name, (tail, data, rows)) in &self.f32s {
            put(name, "f32", tail, *rows, catalog::f32_bytes(data), 4 * tail.iter().product::<usize>())?;
        }
        for (name, (tail, data, rows)) in &self.i32s {
            put(name, "i32", tail, *rows, catalog::i32_bytes(data), 4 * tail.iter().product::<usize>())?;
        }
        for (name, (tail, data, rows)) in &self.f64s {
            put(name, "f64", tail, *rows, catalog::f64_bytes(data), 8 * tail.iter().product::<usize>())?;
        }
        Ok(cat)
    }
}

/// The frames a training event keeps (the rest regenerate from its recipe, bit for bit).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub frames: Vec<usize>,
    /// Frame boundaries kept for those frames: f and f + 1.
    pub state_rows: Vec<usize>,
}

impl Selection {
    pub fn keeps(&self, f: usize) -> bool {
        self.frames.binary_search(&f).is_ok()
    }
}

/// Storage of a recording (the user's 20 GB budget; planning decision of 2026-10-21).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Storage {
    /// One zstd frame per array file.
    pub compress: bool,
    /// Training events that are not ensemble members keep only the frames training uses
    /// (`select_frames`); validation, test, sealed and ensemble events keep every frame.
    pub select_training_frames: bool,
    /// Of the settling frames, keep one in this many.
    pub settling_stride: usize,
    /// The oracle's complete state at every frame boundary (`full_state/`, the
    /// full-state contract), every `full_state_every`-th self-contained, zstd level
    /// `full_state_level`.
    #[serde(default)]
    pub full_state: bool,
    #[serde(default = "default_full_state_every")]
    pub full_state_every: usize,
    #[serde(default = "default_full_state_level")]
    pub full_state_level: i32,
}

fn default_full_state_every() -> usize {
    10
}

fn default_full_state_level() -> i32 {
    19
}

/// Generation runs once (the data retention spec, 2026-10-22): every frame kept, with the
/// oracle's complete state at every boundary.
impl Default for Storage {
    fn default() -> Self {
        Storage { compress: true, select_training_frames: false, settling_stride: 8, full_state: true, full_state_every: 10, full_state_level: 19 }
    }
}

/// The frames a training event keeps: every pre-contact, contact, secondary and separation
/// frame, every frame with a break or a change of phase (and the frame before it), the last
/// frame, and one settling frame in `stride`: the phases the trainer's quotas draw from, and
/// every transition. What is dropped is settling, regenerable from the recipe.
pub fn select_frames(phases: &[&str], breaks: &[usize], stride: usize) -> Selection {
    let n = phases.len();
    let mut keep = vec![false; n];
    for f in 0..n {
        let transition = (f > 0 && phases[f] != phases[f - 1]) || breaks[f] > 0;
        if phases[f] != "settling" || transition || f + 1 == n || (f % stride.max(1)) == 0 {
            keep[f] = true;
            if transition && f > 0 {
                keep[f - 1] = true;
            }
        }
    }
    let frames: Vec<usize> = (0..n).filter(|&f| keep[f]).collect();
    let mut state_rows: Vec<usize> = frames.iter().flat_map(|&f| [f, f + 1]).collect();
    state_rows.dedup();
    Selection { frames, state_rows }
}

/// Flat chunk and bond numbering over the structures.
pub(crate) struct Index {
    pub(crate) chunk0: Vec<usize>,
    pub(crate) bond0: Vec<usize>,
    pub(crate) n: usize,
    pub(crate) e: usize,
}

impl Index {
    pub(crate) fn new(w: &World) -> Index {
        let (mut chunk0, mut bond0, mut n, mut e) = (Vec::new(), Vec::new(), 0, 0);
        for s in 0..w.solver.structures.len() {
            chunk0.push(n);
            bond0.push(e);
            n += w.solver.structures[s].chunks.len();
            e += w.solver.bonds[s].len();
        }
        Index { chunk0, bond0, n, e }
    }

    fn check(&self, w: &World) -> Result<(), String> {
        let again = Index::new(w);
        if again.n != self.n || again.e != self.e {
            return Err(format!("chunk/bond set changed ({} / {} -> {} / {}): the fixed-graph recorder rejects it", self.n, self.e, again.n, again.e));
        }
        Ok(())
    }
}

fn v3(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// The axis-aligned box of every structure's chunks at the start, expanded by the margin.
#[derive(Clone, Copy, Debug)]
struct Bounds {
    lo: Vec3,
    hi: Vec3,
}

impl Bounds {
    fn of(w: &World, rule: &StopRule) -> Bounds {
        let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
        for s in 0..w.solver.structures.len() {
            for c in 0..w.solver.structures[s].chunks.len() {
                let p = w.solver.chunk_position(s, c);
                let h = Vec3::splat(w.solver.structures[s].chunks[c].half_extents.norm());
                lo = lo.component_min(p - h);
                hi = hi.component_max(p + h);
            }
        }
        let margin = rule.margin.max(rule.margin_fraction * (hi - lo).max_elem());
        Bounds { lo: lo - Vec3::splat(margin), hi: hi + Vec3::splat(margin) }
    }

    /// A body at `x` moving at `v` has left: outside the bounds and not moving back in.
    /// Under gravity only the horizontal extent counts (a body above or below may still
    /// fall onto the structure or rest on the ground beside it).
    fn left(&self, x: Vec3, v: Vec3, gravity: Vec3) -> bool {
        let c = (self.lo + self.hi) * 0.5;
        let (mut d, mut vel, mut lo, mut hi, mut p) = (x - c, v, self.lo, self.hi, x);
        if gravity.norm() > 0.0 {
            let g = gravity.normalized();
            let flat = |u: Vec3| u - g * u.dot(g);
            d = flat(d);
            vel = flat(vel);
            // Compare in the horizontal plane: project the box corners too.
            p = c + d;
            lo = c + flat(self.lo - c);
            hi = c + flat(self.hi - c);
            let (a, b) = (lo.component_min(hi), lo.component_max(hi));
            lo = a;
            hi = b;
            // A vertical axis has zero extent after projection: never "outside" along it.
            for k in 0..3 {
                if (hi[k] - lo[k]).abs() < 1e-12 {
                    lo[k] = f64::NEG_INFINITY;
                    hi[k] = f64::INFINITY;
                }
            }
        }
        let outside = (0..3).any(|k| p[k] < lo[k] || p[k] > hi[k]);
        outside && vel.dot(d) >= 0.0
    }
}

/// Chunk world velocity and angular velocity, and kinetic energy.
fn chunk_ke(w: &World, s: usize, c: usize) -> f64 {
    let (v, om) = w.solver.chunk_velocity(s, c);
    let ch = &w.solver.structures[s].chunks[c];
    let cl = &w.solver.clusters[w.solver.chunks[s][c].cluster];
    // ChunkData.inertia is in the structure's body frame: it turns with the cluster alone.
    let r = cl.rotation();
    let iw = r * ch.inertia * r.transpose();
    0.5 * ch.mass * v.norm2() + 0.5 * om.dot(iw * om)
}

/// Total kinetic energy of the bodies that have not left, and whether every bonded
/// cluster among them sleeps (quasi-static) and every loose chunk among them is slower
/// than the stop speed.
fn energy_in_bounds(w: &World, b: &Bounds, rule: &StopRule) -> (f64, bool) {
    let g = w.solver.config.gravity;
    let (mut ke, mut asleep) = (0.0, true);
    for cl in &w.solver.clusters {
        let s = cl.structure;
        if b.left(cl.com_world(), cl.velocity, g) {
            continue;
        }
        let e: f64 = cl.chunks.iter().map(|&c| chunk_ke(w, s, c)).sum();
        ke += e;
        let sleeping = if cl.bonds.is_empty() { cl.velocity.norm() < rule.stop_speed && cl.angular_velocity.norm() * 0.1 < rule.stop_speed } else { cl.activity == Activity::Settled };
        asleep &= cl.anchored && cl.bonds.is_empty() || sleeping;
    }
    for imp in &w.impactors {
        if !b.left(imp.pose.position, imp.velocity, g) {
            ke += imp.kinetic_energy();
        }
    }
    (ke, asleep)
}

/// Without gravity: whether no two bodies that have not left can still meet. Each cluster
/// and impactor is a bounding sphere about its centre of mass (it bounds the body in every
/// orientation) moving at its velocity, which no force changes while nothing touches; a
/// pair may interact when the spheres come within reach of each other in the next
/// `horizon` seconds (the rest of the recording) while moving relative to each other at the
/// stop speed or faster. Pieces coasting apart in free space are the zero-gravity
/// counterpart of sleeping, and an impactor coasting where it can meet nothing has left the
/// interaction.
fn interactions_over(w: &World, b: &Bounds, rule: &StopRule, horizon: f64) -> bool {
    let g = w.solver.config.gravity;
    if g.norm() > 0.0 {
        return false;
    }
    let mut bodies: Vec<(Vec3, f64, Vec3)> = Vec::new();
    for cl in &w.solver.clusters {
        let (com, s) = (cl.com_world(), cl.structure);
        if b.left(com, cl.velocity, g) {
            continue;
        }
        let r = cl.chunks.iter().map(|&c| (w.solver.chunk_position(s, c) - com).norm() + w.solver.structures[s].chunks[c].half_extents.norm()).fold(0.0, f64::max);
        // A sphere about the centre of mass bounds the body in every orientation.
        bodies.push((com, r, cl.velocity));
    }
    for imp in &w.impactors {
        if !b.left(imp.pose.position, imp.velocity, g) {
            let r = match imp.shape {
                stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
                stress_ref::scene::ImpactorShape::Box { half_extents } => Vec3::from_array(half_extents).norm(),
            };
            bodies.push((imp.pose.position, r, imp.velocity));
        }
    }
    for i in 0..bodies.len() {
        for j in i + 1..bodies.len() {
            let (dx, dv) = (bodies[j].0 - bodies[i].0, bodies[j].2 - bodies[i].2);
            if dv.norm() < rule.stop_speed {
                continue;
            }
            let t = (-dx.dot(dv) / dv.norm2()).clamp(0.0, horizon);
            if (dx + dv * t).norm() < bodies[i].1 + bodies[j].1 {
                return false;
            }
        }
    }
    true
}

/// Each structure's momentum over its active chunks (hidden motion included): linear, and
/// angular about the world origin (x x m v + R I R^T w).
fn structure_momenta(w: &World) -> Vec<[f64; 6]> {
    (0..w.solver.structures.len())
        .map(|s| {
            let (mut p, mut l) = (Vec3::ZERO, Vec3::ZERO);
            for (c, ch) in w.solver.structures[s].chunks.iter().enumerate() {
                let st = &w.solver.chunks[s][c];
                if !st.active {
                    continue;
                }
                let (v, om) = w.solver.chunk_velocity(s, c);
                let x = w.solver.chunk_position(s, c);
                let r = w.solver.clusters[st.cluster].rotation();
                p += v * ch.mass;
                l += x.cross(v * ch.mass) + r * ch.inertia * r.transpose() * om;
            }
            [p.x, p.y, p.z, l.x, l.y, l.z]
        })
        .collect()
}

fn structure_ke(w: &World) -> f64 {
    (0..w.solver.structures.len()).flat_map(|s| (0..w.solver.structures[s].chunks.len()).map(move |c| (s, c))).filter(|&(s, c)| w.solver.chunks[s][c].active).map(|(s, c)| chunk_ke(w, s, c)).sum()
}

fn impactor_state(w: &World) -> Vec<f64> {
    let mut out = Vec::with_capacity(w.impactors.len() * IMPACTOR_STATE);
    for imp in &w.impactors {
        out.extend(imp.pose.position.to_array());
        out.extend(imp.pose.rotation.to_wxyz());
        out.extend(imp.velocity.to_array());
        out.extend(imp.angular_velocity.to_array());
    }
    out
}

fn to32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

/// Frame-end (or frame-start) chunk motion: positions, velocities, angular velocities.
fn chunk_motion(w: &World, ix: &Index) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut p, mut v, mut om) = (Vec::with_capacity(ix.n * 3), Vec::with_capacity(ix.n * 3), Vec::with_capacity(ix.n * 3));
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            p.extend(w.solver.chunk_position(s, c).to_array());
            let (a, b) = w.solver.chunk_velocity(s, c);
            v.extend(a.to_array());
            om.extend(b.to_array());
        }
    }
    (p, v, om)
}

fn bond_connected(w: &World) -> Vec<i32> {
    w.solver.bonds.iter().flatten().map(|b| i32::from(b.connected())).collect()
}

/// Chunk pairs of different clusters whose bounding spheres overlap (in or near contact),
/// as flat chunk indices `a < b`: the candidates for structure-contact and cracked-face
/// edges, which a call builder refines with the recorded poses. A uniform grid of the
/// largest sphere's diameter keeps it linear in chunks.
pub(crate) fn near_pairs(w: &World, ix: &Index) -> Vec<(usize, usize)> {
    let mut items: Vec<(usize, Vec3, f64, usize)> = Vec::new();
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            let st = &w.solver.chunks[s][c];
            if st.active {
                items.push((ix.chunk0[s] + c, w.solver.chunk_position(s, c), w.solver.structures[s].chunks[c].half_extents.norm(), st.cluster));
            }
        }
    }
    let cell = 2.0 * items.iter().map(|i| i.2).fold(1e-6, f64::max);
    let key = |p: Vec3| [(p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64];
    let mut grid: std::collections::HashMap<[i64; 3], Vec<usize>> = std::collections::HashMap::new();
    for (k, it) in items.iter().enumerate() {
        grid.entry(key(it.1)).or_default().push(k);
    }
    let mut out = Vec::new();
    for a in items.iter() {
        let [x, y, z] = key(a.1);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid.get(&[x + dx, y + dy, z + dz]).into_iter().flatten() {
                        let b = &items[j];
                        if b.0 > a.0 && b.3 != a.3 && (b.1 - a.1).norm() < a.2 + b.2 {
                            out.push((a.0, b.0));
                        }
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// The unit quaternion of a rotation matrix (Shepperd's method).
fn quat_of(r: &stress_ref::math::Mat3) -> Quat {
    let m = &r.m;
    let t = m[0][0] + m[1][1] + m[2][2];
    let q = if t > 0.0 {
        let s = (t + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    Quat::from_wxyz(q).normalized()
}

/// The whole state at a frame boundary (`state_*`, one row per boundary).
pub(crate) fn capture_state(w: &World, ix: &Index, a: &mut Arrays) -> Result<(), String> {
    let (n, e, ni) = (ix.n, ix.e, w.impactors.len());
    let (p, v, om) = chunk_motion(w, ix);
    a.f("state_chunk_pos", &[n, 3], &to32(&p))?;
    a.f("state_chunk_vel", &[n, 3], &to32(&v))?;
    a.f("state_chunk_angvel", &[n, 3], &to32(&om))?;
    let mut quat = Vec::with_capacity(n * 4);
    let (mut hidden, mut cluster, mut active, mut anchored) = (Vec::with_capacity(n * 6), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            let st = &w.solver.chunks[s][c];
            let cl = &w.solver.clusters[st.cluster];
            quat.extend(cl.pose.rotation.to_wxyz());
            hidden.extend(st.u.to_array());
            hidden.extend(st.th.to_array());
            cluster.push(st.cluster as i32);
            active.push(i32::from(st.active));
            anchored.push(i32::from(cl.anchored));
        }
    }
    a.f("state_chunk_cluster_quat", &[n, 4], &to32(&quat))?;
    // Each chunk's own world orientation: cluster, hidden rotation, chunk's rest rotation.
    let mut qw = Vec::with_capacity(n * 4);
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            let st = &w.solver.chunks[s][c];
            let cl = &w.solver.clusters[st.cluster];
            let hidden = Quat::from_rotation_vector(st.th);
            let rest = quat_of(&w.solver.structures[s].chunks[c].rotation);
            qw.extend((cl.pose.rotation * hidden * rest).normalized().to_wxyz());
        }
    }
    a.f("state_chunk_q_world", &[n, 4], &to32(&qw))?;
    a.f("state_chunk_hidden", &[n, 6], &to32(&hidden))?;
    a.i("state_chunk_cluster", &[n], &cluster)?;
    a.i("state_chunk_active", &[n], &active)?;
    a.i("state_chunk_anchored", &[n], &anchored)?;
    let mut joint = Vec::with_capacity(e * 4);
    for b in w.solver.bonds.iter().flatten() {
        joint.extend([b.joint.damage, b.joint.crush, b.joint.fatigue, b.joint.utilization]);
    }
    a.f("state_bond_joint", &[e, 4], &to32(&joint))?;
    // Cumulative per-bond energy (J): joint dissipation (fracture, friction, rebar), dashpot
    // dissipation, stored elastic energy; and the failure mode (0 none, 1 tension, 2 shear,
    // 3 compression, 4 buckling).
    let mut energy = Vec::with_capacity(e * 3);
    let mut mode = Vec::with_capacity(e);
    for b in w.solver.bonds.iter().flatten() {
        energy.extend([b.joint.dissipated, b.damped, b.stored]);
        mode.push(match b.joint.mode {
            None => 0,
            Some(stress_ref::joint::FailureMode::Tension) => 1,
            Some(stress_ref::joint::FailureMode::Shear) => 2,
            Some(stress_ref::joint::FailureMode::Compression) => 3,
            Some(stress_ref::joint::FailureMode::Buckling) => 4,
        });
    }
    a.f("state_bond_energy", &[e, 3], &to32(&energy))?;
    // Failure margin (`RtBond::fail_margin`, a running maximum: >= 1 exactly when the
    // bond disconnected). Written so that the f32 value keeps that rule: a connected
    // bond's margin is never rounded up to 1.
    let margin: Vec<f32> = w
        .solver
        .bonds
        .iter()
        .flatten()
        .map(|b| {
            let q = b.fail_margin() as f32;
            if b.connected() && q >= 1.0 { 1.0f32 - f32::EPSILON / 2.0 } else { q }
        })
        .collect();
    for (b, &q) in w.solver.bonds.iter().flatten().zip(&margin) {
        if (q >= 1.0) == b.connected() {
            return Err(format!("failure margin {q} disagrees with connected {} (the joint law's rule)", b.connected()));
        }
    }
    a.f("state_bond_fail_margin", &[e], &margin)?;
    a.i("state_bond_mode", &[e], &mode)?;
    a.i("state_bond_connected", &[e], &bond_connected(w))?;
    a.f("state_impactor", &[ni, IMPACTOR_STATE], &to32(&impactor_state(w)))?;
    Ok(())
}

/// The compact trajectory at a frame boundary (`traj_*`, never thinned): every cluster's
/// chunk count, mass, centre of mass, velocity and angular velocity, and every impactor's
/// state, so whole-life-cycle trajectory metrics work on any event.
fn capture_traj(w: &World, boundary: usize, a: &mut Arrays) -> Result<(), String> {
    for (ci, cl) in w.solver.clusters.iter().enumerate() {
        a.i("traj_cluster_i32", &[3], &[boundary as i32, ci as i32, cl.chunks.len() as i32])?;
        let mut row = vec![cl.mass as f32];
        row.extend(v3(cl.com_world()));
        row.extend(v3(cl.velocity));
        row.extend(v3(cl.angular_velocity));
        a.f("traj_cluster_f32", &[10], &row)?;
    }
    a.f("traj_impactor", &[w.impactors.len(), IMPACTOR_STATE], &to32(&impactor_state(w)))
}

/// The static arrays of the event (`static_*`, one row).
pub(crate) fn capture_static(w: &World, ix: &Index, a: &mut Arrays) -> Result<(), String> {
    let (n, e) = (ix.n, ix.e);
    let mut inertia_full = Vec::new();
    let (mut mass, mut inertia, mut half, mut center, mut support, mut structure) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for (s, st) in w.solver.structures.iter().enumerate() {
        for ch in &st.chunks {
            mass.push(ch.mass as f32);
            inertia.extend([ch.inertia.col(0).x as f32, ch.inertia.col(1).y as f32, ch.inertia.col(2).z as f32]);
            let m = &ch.inertia.m;
            inertia_full.extend([m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]].map(|x| x as f32));
            half.extend(v3(ch.half_extents));
            center.extend(v3(ch.center));
            support.push(i32::from(ch.support != stress_ref::scene::Support::None));
            structure.push(s as i32);
        }
    }
    a.f("static_chunk_mass", &[n], &mass)?;
    a.f("static_chunk_inertia_diag", &[n, 3], &inertia)?;
    // The full body-frame inertia (xx, yy, zz, xy, xz, yz); the diagonal above is not
    // principal for a chunk with a rest rotation.
    a.f("static_chunk_inertia", &[n, 6], &inertia_full)?;
    a.f("static_chunk_half_extents", &[n, 3], &half)?;
    a.f("static_chunk_center_local", &[n, 3], &center)?;
    a.i("static_chunk_support", &[n], &support)?;
    a.i("static_chunk_structure", &[n], &structure)?;
    let (mut ends, mut geom, mut stiff, mut strength, mut weibull) = (vec![], vec![], vec![], vec![], vec![]);
    for (s, bonds) in w.solver.bonds.iter().enumerate() {
        for b in bonds {
            let g = &b.geometry;
            ends.extend([(ix.chunk0[s] + g.a) as i32, (ix.chunk0[s] + g.b) as i32]);
            geom.extend(v3(g.centroid));
            geom.extend(v3(g.normal));
            geom.extend(v3(g.t1));
            geom.extend([g.area as f32, g.width[0] as f32, g.width[1] as f32, g.length as f32]);
            let k = &b.stiffness;
            stiff.extend([k.kn, k.ks, k.kb_t1, k.kb_t2, k.kt].map(|x| x as f32));
            let t = &b.strength;
            // An infinite limit (no shear cap) is written as -1.
            strength.extend([t.tensile, t.compressive, t.cohesion, t.friction, t.shear_cap, t.g_tension, t.g_shear, t.g_compression].map(|x| if x.is_finite() { x as f32 } else { -1.0 }));
            weibull.push(b.weibull as f32);
        }
    }
    a.i("static_bond_endpoints", &[e, 2], &ends)?;
    a.f("static_bond_geometry", &[e, 13], &geom)?;
    a.f("static_bond_stiffness", &[e, 5], &stiff)?;
    a.f("static_bond_strength", &[e, 8], &strength)?;
    a.f("static_bond_weibull", &[e], &weibull)?;
    Ok(())
}

/// Frame outcome arrays of a branch world stepped one frame from the frame-start state.
pub(crate) fn capture_branch(prefix: &str, start_connected: &[i32], w: &World, ix: &Index, a: &mut Arrays) -> Result<(), String> {
    ix.check(w)?;
    let (n, e, ni) = (ix.n, ix.e, w.impactors.len());
    let now = bond_connected(w);
    let broken: Vec<i32> = start_connected.iter().zip(&now).map(|(&a, &b)| i32::from(a == 1 && b == 0)).collect();
    a.i(&format!("{prefix}_bond_broken"), &[e], &broken)?;
    let peak: Vec<f32> = w.solver.bonds.iter().flatten().map(|b| b.peak_utilization as f32).collect();
    a.f(&format!("{prefix}_bond_peak_util"), &[e], &peak)?;
    let (p, v, om) = chunk_motion(w, ix);
    a.f(&format!("{prefix}_chunk_pos"), &[n, 3], &to32(&p))?;
    a.f(&format!("{prefix}_chunk_vel"), &[n, 3], &to32(&v))?;
    a.f(&format!("{prefix}_chunk_angvel"), &[n, 3], &to32(&om))?;
    a.f(&format!("{prefix}_impactor"), &[ni, IMPACTOR_STATE], &to32(&impactor_state(w)))?;
    Ok(())
}

/// A frame of `w` under `template`'s model, from `w`'s state folded to rigid bodies, with
/// `breaks` broken first.
pub(crate) fn branch_frame(w: &World, template: &World, breaks: &[(usize, usize)]) -> Result<World, String> {
    let mut b = w.clone();
    b.reduce_to_rigid();
    b.adopt_model(template)?;
    for &(s, bi) in breaks {
        if b.solver.bonds[s][bi].connected() {
            b.break_bond(s, bi);
        }
    }
    b.solver.reset_peak_utilizations();
    b.step_frame();
    Ok(b)
}

/// The QS prior: a converged static solve of the oracle's bond model with each predicted
/// contact's Hertz peak force on its struck chunk (along the closing direction), on a
/// copy; one damage commit (no cascade), so each bond's failure index is that of the
/// equilibrium. Returns the failure index per bond (-1 for bonds not connected at the
/// start) and whether every cluster's solve converged.
fn qs_prior(w: &World, contacts: &[stress_ref::world::PredictedContact]) -> (Vec<f32>, bool) {
    let mut q = w.clone();
    let mut loads = ChunkLoads::new(&q.solver);
    for c in contacts {
        loads.force[c.structure][c.entry] += c.direction * c.peak_force;
    }
    q.solver.reset_peak_utilizations();
    let opts = StaticOptions { cascade: true, max_cascade: 1, ..Default::default() };
    let report = q.solver.solve_static_all(&loads, &opts, 0.0);
    let util = w
        .solver
        .bonds
        .iter()
        .zip(&q.solver.bonds)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| if x.connected() { y.peak_utilization as f32 } else { -1.0 }))
        .collect();
    (util, report.converged)
}

fn fnv(h: &mut u64, x: f64) {
    for b in x.to_bits().to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// FNV-1a over every chunk's position and velocity, every bond's damage and every
/// impactor's state (the event's end fingerprint, for replay checks).
pub fn fingerprint(w: &World) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for s in 0..w.solver.structures.len() {
        for c in 0..w.solver.structures[s].chunks.len() {
            let (v, om) = w.solver.chunk_velocity(s, c);
            for x in [w.solver.chunk_position(s, c), v, om] {
                fnv(&mut h, x.x);
                fnv(&mut h, x.y);
                fnv(&mut h, x.z);
            }
        }
    }
    for b in w.solver.bonds.iter().flatten() {
        fnv(&mut h, b.joint.damage);
    }
    for imp in &w.impactors {
        for x in [imp.pose.position, imp.velocity, imp.angular_velocity] {
            fnv(&mut h, x.x);
            fnv(&mut h, x.y);
            fnv(&mut h, x.z);
        }
    }
    format!("{h:016x}")
}

/// FNV-1a of a scene's JSON (the scene hash P0 quotes).
pub fn scene_hash(scene: &Scene) -> String {
    let text = serde_json::to_string(scene).expect("scene serialises");
    let mut h = 0xcbf29ce484222325u64;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Per-impactor contact within the frame, read after each substep: how many substeps the
/// impactor took a contact impulse in, the first and last, and the structures it touched.
struct ContactWatch {
    prev: Vec<Vec3>,
    substeps: usize,
    count: Vec<usize>,
    first: Vec<Option<usize>>,
    last: Vec<Option<usize>>,
    structures: Vec<HashSet<usize>>,
    /// Contact impulse on each impactor over the frame (N s): its momentum change less
    /// gravity's, summed per substep.
    impulse: Vec<Vec3>,
    /// Largest substep-mean contact force on each impactor (N).
    peak_force: Vec<f64>,
    // Frame timing: start, length, the last substep's end; substep sizes and how many
    // substeps the contact step limited.
    t0: f64,
    frame_dt: f64,
    prev_time: f64,
    dt_min: f64,
    dt_max: f64,
    contact_limited: usize,
    /// Per bond (flat): the largest failure index seen at a substep end, the generalised
    /// force then (shear t1, shear t2, axial; bending t1, bending t2, torsion) and its time.
    peak_util: Vec<f64>,
    peak_load: Vec<[f64; 6]>,
    peak_time: Vec<f64>,
    /// Impactor states at the first substep ends at or past k/8 of the frame, k = 1..8:
    /// per impactor (time, position, orientation wxyz, velocity, angular velocity).
    subframe: Vec<Vec<[f64; 14]>>,
    /// Every chunk's position and velocity at the first substep end at or past mid-frame.
    mid: Option<(f64, Vec<f64>, Vec<f64>)>,
}

fn impactor_row(t: f64, imp: &stress_ref::world::Impactor) -> [f64; 14] {
    let q = imp.pose.rotation.to_wxyz();
    let (p, v, w) = (imp.pose.position, imp.velocity, imp.angular_velocity);
    [t, p.x, p.y, p.z, q[0], q[1], q[2], q[3], v.x, v.y, v.z, w.x, w.y, w.z]
}

impl ContactWatch {
    fn new(w: &World) -> ContactWatch {
        let k = w.impactors.len();
        let e = w.solver.bonds.iter().map(Vec::len).sum();
        ContactWatch {
            prev: w.impactors.iter().map(|i| i.velocity).collect(),
            substeps: 0,
            count: vec![0; k],
            first: vec![None; k],
            last: vec![None; k],
            structures: vec![HashSet::new(); k],
            impulse: vec![Vec3::ZERO; k],
            peak_force: vec![0.0; k],
            t0: w.solver.time,
            frame_dt: w.scene.sim.frame_dt,
            prev_time: w.solver.time,
            dt_min: f64::INFINITY,
            dt_max: 0.0,
            contact_limited: 0,
            peak_util: vec![f64::NEG_INFINITY; e],
            peak_load: vec![[0.0; 6]; e],
            peak_time: vec![w.solver.time; e],
            subframe: Vec::with_capacity(8),
            mid: None,
        }
    }

    fn observe(&mut self, w: &World) {
        let g = w.solver.config.gravity;
        let i = self.substeps;
        self.substeps += 1;
        let t = w.solver.time;
        let dt = t - self.prev_time;
        self.prev_time = t;
        self.dt_min = self.dt_min.min(dt);
        self.dt_max = self.dt_max.max(dt);
        let (stress_dt, contact_dt) = w.step_limits();
        self.contact_limited += usize::from(contact_dt < stress_dt);
        for (gi, b) in w.solver.bonds.iter().flatten().enumerate() {
            let u = b.joint.utilization;
            if u > self.peak_util[gi] {
                self.peak_util[gi] = u;
                let (l, a) = (b.force.lin, b.force.ang);
                self.peak_load[gi] = [l.x, l.y, l.z, a.x, a.y, a.z];
                self.peak_time[gi] = t;
            }
        }
        let eps = 1e-9 * self.frame_dt;
        while self.subframe.len() < 8 && t >= self.t0 + (self.subframe.len() + 1) as f64 * self.frame_dt / 8.0 - eps {
            self.subframe.push(w.impactors.iter().map(|imp| impactor_row(t, imp)).collect());
        }
        if self.mid.is_none() && t >= self.t0 + 0.5 * self.frame_dt - eps {
            let (mut p, mut v) = (Vec::new(), Vec::new());
            for s in 0..w.solver.structures.len() {
                for c in 0..w.solver.structures[s].chunks.len() {
                    p.extend(w.solver.chunk_position(s, c).to_array());
                    v.extend(w.solver.chunk_velocity(s, c).0.to_array());
                }
            }
            self.mid = Some((t, p, v));
        }
        for (k, imp) in w.impactors.iter().enumerate() {
            let dv = imp.velocity - self.prev[k] - g * dt;
            self.prev[k] = imp.velocity;
            self.impulse[k] += dv * imp.mass;
            self.peak_force[k] = self.peak_force[k].max(imp.mass * dv.norm() / dt);
            if dv.norm() <= 1e-12 * imp.velocity.norm().max(1.0) {
                continue;
            }
            self.count[k] += 1;
            self.first[k].get_or_insert(i);
            self.last[k] = Some(i);
            let reach = match imp.shape {
                stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
                stress_ref::scene::ImpactorShape::Box { half_extents } => Vec3::from_array(half_extents).norm(),
            };
            for (s, hits) in w.chunk_contact_hits().iter().enumerate() {
                if self.structures[k].contains(&s) {
                    continue;
                }
                let near = hits.iter().enumerate().any(|(c, &h)| {
                    h > 0.0 && (w.solver.chunk_position(s, c) - imp.pose.position).norm() <= reach + w.solver.structures[s].chunks[c].half_extents.norm() * 1.01
                });
                if near {
                    self.structures[k].insert(s);
                }
            }
        }
    }
}

/// The causal inputs of the coming frame, captured before it is stepped: the routing's
/// predicted islands (or the caller's, `routing`: chunk and impactor flags), cross-cluster
/// near pairs, every predicted impactor contact with the punch's plug, and the QS prior.
/// Written into `a` as frame `f`'s rows. The recorder and the call builder's live entry
/// (closed loop) both use this, so their inputs are one computation.
pub(crate) struct FrameInputs {
    pub chunk_routed: Vec<i32>,
    pub impactor_routed: Vec<i32>,
    pub pairs: usize,
    pub contacts: Vec<stress_ref::world::PredictedContact>,
    pub punch_breaks: Vec<(usize, usize)>,
    pub prior_converged: bool,
    pub prior_seconds: f64,
}

pub(crate) fn capture_inputs(world: &mut World, ix: &Index, f: usize, routing: Option<(Vec<i32>, Vec<i32>)>, a: &mut Arrays) -> Result<FrameInputs, String> {
    let frame_dt = world.scene.sim.frame_dt;
    let (chunk_routed, imp_routed) = match routing {
        Some(r) => r,
        None => {
            let (routed_clusters, routed_impactors) = world.predict_island_members(frame_dt, &HashSet::new());
            let routed: HashSet<u64> = routed_clusters.into_iter().collect();
            let mut chunk_routed = vec![0i32; ix.n];
            for s in 0..world.solver.structures.len() {
                for c in 0..world.solver.structures[s].chunks.len() {
                    let st = &world.solver.chunks[s][c];
                    chunk_routed[ix.chunk0[s] + c] = i32::from(st.active && routed.contains(&world.solver.clusters[st.cluster].id));
                }
            }
            let imp_routed: Vec<i32> = (0..world.impactors.len()).map(|k| i32::from(routed_impactors.contains(&k))).collect();
            (chunk_routed, imp_routed)
        }
    };
    if chunk_routed.len() != ix.n || imp_routed.len() != world.impactors.len() {
        return Err("routing flags do not match the world's chunks and impactors".into());
    }
    a.i("frame_chunk_routed", &[ix.n], &chunk_routed)?;
    a.i("frame_impactor_routed", &[world.impactors.len()], &imp_routed)?;
    let pairs = near_pairs(world, &ix);
    for &(pa, pb) in &pairs {
        a.i("pair_i32", &[3], &[f as i32, pa as i32, pb as i32])?;
    }
    let contacts = world.predict_contacts(frame_dt, 0.0);
    let tiered = tiered_impact_speed();
    let mut punch_breaks: Vec<(usize, usize)> = Vec::new();
    let mut plug_of = vec![0i32; ix.n];
    let mut bond_punch = vec![0i32; ix.e];
    for (k, c) in contacts.iter().enumerate() {
        let cluster_index = world.solver.clusters.iter().position(|cl| cl.id == c.cluster).expect("predicted cluster exists") as i32;
        a.i("contact_i32", &[5], &[f as i32, c.impactor as i32, c.structure as i32, (ix.chunk0[c.structure] + c.entry) as i32, cluster_index])?;
        let (ratio, mass, breaks) = match (&c.plug, c.speed > tiered) {
            (Some(p), true) => (p.force / p.capacity.max(1e-300), p.mass, p.breaks()),
            (Some(p), false) => (p.force / p.capacity.max(1e-300), p.mass, false),
            _ => (0.0, 0.0, false),
        };
        let mut row = vec![c.time as f32];
        row.extend(v3(c.point));
        row.extend(v3(c.direction));
        row.extend([c.speed, c.reduced_mass, c.modulus, c.face_radius, c.hertz_duration, c.peak_force, ratio, mass, f64::from(u8::from(breaks)), c.cluster_chunks as f64].map(|x| x as f32));
        a.f("contact_f32", &[CONTACT_F32], &row)?;
        if let (Some(p), true) = (&c.plug, c.speed > tiered) {
            for &ch in &p.chunks {
                plug_of[ix.chunk0[c.structure] + ch] = k as i32 + 1;
            }
            if breaks {
                for &bi in &p.interface {
                    bond_punch[ix.bond0[c.structure] + bi] = 1;
                    punch_breaks.push((c.structure, bi));
                }
            }
        }
    }
    punch_breaks.sort_unstable();
    punch_breaks.dedup();
    debug_assert_eq!(punch_breaks, world.predict_punch(frame_dt));
    a.i("frame_chunk_punch_plug", &[ix.n], &plug_of)?;
    a.i("frame_bond_punch", &[ix.e], &bond_punch)?;
    let t_prior = Instant::now();
    let (prior, prior_converged) = if contacts.is_empty() { (vec![-1.0f32; ix.e], true) } else { qs_prior(world, &contacts) };
    a.f("frame_bond_qs_prior_util", &[ix.e], &prior)?;

    Ok(FrameInputs { chunk_routed, impactor_routed: imp_routed, pairs: pairs.len(), contacts, punch_breaks, prior_converged, prior_seconds: t_prior.elapsed().as_secs_f64() })
}

/// Record one event into `out/<event_id>` (written to `<event_id>.partial` and renamed
/// when complete).
/// One oracle frame captured as the recorder captures the rows a call's targets read
/// (`frame_bond_break_bin`, the peak utilisations, the dissipation, then the frame-end
/// state and trajectory), stepped as the recorder steps (`full_state::oracle_frame`). The
/// relabelled call's targets (`outcome_calls::live_call_labelled`); the test
/// `an_own_state_relabel_is_the_recorded_call` holds it to the recorder bit for bit.
pub(crate) fn labelled_frame(world: &mut World, ix: &Index, f: usize, peak_prev: &[f32], a: &mut Arrays) -> Result<(), String> {
    let frame_dt = world.scene.sim.frame_dt;
    let t0 = world.solver.time;
    let start_energy: Vec<(f64, f64)> = world.solver.bonds.iter().flatten().map(|b| (b.joint.dissipated, b.damped)).collect();
    let events_before = world.solver.events.len();
    crate::full_state::oracle_frame(world, |_| {})?;
    ix.check(world)?;
    let mut break_bin = vec![-1i32; ix.e];
    for ev in &world.solver.events[events_before..] {
        if let SolverEvent::Broken { time, structure, bond, .. } = ev {
            let bin = (((time - t0) / frame_dt) * TIME_BINS as f64).floor().clamp(0.0, (TIME_BINS - 1) as f64) as i32;
            let g = ix.bond0[*structure] + bond;
            if break_bin[g] < 0 {
                break_bin[g] = bin;
            }
        }
    }
    a.i("frame_bond_break_bin", &[ix.e], &break_bin)?;
    let peak: Vec<f32> = world.solver.bonds.iter().flatten().map(|b| b.peak_utilization as f32).collect();
    a.f("frame_bond_peak_util", &[ix.e], &peak)?;
    a.f("frame_bond_peak_util_prev", &[ix.e], peak_prev)?;
    let dissipated: Vec<f32> = world.solver.bonds.iter().flatten().zip(&start_energy).flat_map(|(b, &(j, d))| [(b.joint.dissipated - j) as f32, (b.damped - d) as f32]).collect();
    a.f("frame_bond_dissipated", &[ix.e, 2], &dissipated)?;
    capture_state(world, ix, a)?;
    capture_traj(world, f + 1, a)?;
    Ok(())
}

pub fn record_outcome_event(recipe: &OutcomeRecipe, out: &Path, opts: &RecordOptions, beat: &dyn Fn(&Beat)) -> Result<OutcomeSummary, String> {
    let t_event = Instant::now();
    let final_dir = out.join(&recipe.event_id);
    let dir = out.join(format!("{}.partial", recipe.event_id));
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| format!("clear {}: {e}", dir.display()))?;
    }
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let mut scene = recipe.scene.clone();
    opts.oracle.apply(&mut scene);
    if scene.sim.refine_utilization.is_some() {
        return Err("refinement changes the bond set; the fixed-graph recorder rejects it".into());
    }
    if scene.sim.methods.layer_contact {
        return Err("the layer contact is not supported by the contact watch".into());
    }
    let frame_dt = scene.sim.frame_dt;
    let max_frames = (opts.stop.max_seconds / frame_dt).round() as usize;
    scene.sim.duration = max_frames as f64 * frame_dt;
    let mut world = World::new(&scene);
    world.solver.record_load_impulse = true;
    let template = opts.baselines.then(|| qs_template(&scene));
    let ix = Index::new(&world);
    let bounds = Bounds::of(&world, &opts.stop);
    let gravity = world.solver.config.gravity;
    let input_energy = world.impactors.iter().map(|i| i.kinetic_energy()).sum::<f64>() + structure_ke(&world);
    if !(input_energy > 0.0) {
        return Err(format!("input energy {input_energy} J: nothing to record"));
    }
    // Gravity's largest possible work: every body falling the bounds' height and 10 m more.
    let all_mass = world.impactors.iter().map(|i| i.mass).sum::<f64>() + world.solver.structures.iter().flat_map(|s| s.chunks.iter()).map(|c| c.mass).sum::<f64>();
    let gravity_work_bound = all_mass * gravity.norm() * ((bounds.hi - bounds.lo).max_elem() + 10.0);
    let v0: Vec<Vec3> = world.impactors.iter().map(|i| i.velocity).collect();

    let mut statics = Arrays::default();
    capture_static(&world, &ix, &mut statics)?;
    let mut a = Arrays::default();
    a.declare_i("contact_i32", &[5]);
    a.declare_f("contact_f32", &[CONTACT_F32]);
    a.declare_i("punch_frame", &[]);
    a.declare_i("pair_i32", &[3]);
    a.declare_i("break_i32", &[4]);
    a.declare_d("break_f64", &[]);
    a.declare_i("peak_i32", &[2]);
    a.declare_d("peak_f64", &[9]);
    a.declare_i("mid_frame", &[]);
    a.declare_d("mid_time", &[]);
    a.declare_d("mid_chunk_pos", &[ix.n, 3]);
    a.declare_d("mid_chunk_vel", &[ix.n, 3]);
    capture_state(&world, &ix, &mut a)?;
    capture_traj(&world, 0, &mut a)?;
    let mut peak_prev = vec![0f32; ix.e];

    let mut frames: Vec<Value> = Vec::new();
    let mut cost: BTreeMap<String, f64> = ["oracle", "qs_branch", "punch_branch", "qs_prior", "overhead"].iter().map(|k| (k.to_string(), 0.0)).collect();
    let mut per_phase: BTreeMap<String, usize> = PHASES.iter().map(|p| (p.to_string(), 0)).collect();
    let (mut ever_contact, mut engagement_over) = (false, false);
    let mut first_struck: HashSet<usize> = HashSet::new();
    let (mut last_event_frame, mut last_break_frame): (Option<usize>, Option<usize>) = (None, None);
    let mut break_times: Vec<f64> = Vec::new();
    let mut stop_reason = String::new();
    let mut truncated = false;
    let mut substeps_total = 0u64;
    let mut full_state = opts.storage.full_state.then(|| crate::full_state::FullStateWriter::new(opts.storage.full_state_every, opts.storage.full_state_level));
    let mut break_order = 0i32;

    for f in 0..max_frames {
        let t_over = Instant::now();
        let t0 = world.solver.time;
        // The oracle's complete state at this boundary (the full-state contract).
        if let Some(fs) = full_state.as_mut() {
            fs.push(&dir, f, &world)?;
        }
        // Inputs, before the step.
        let inputs = capture_inputs(&mut world, &ix, f, None, &mut a)?;
        let (chunk_routed, imp_routed, pairs_n, contacts, punch_breaks, prior_converged) =
            (inputs.chunk_routed, inputs.impactor_routed, inputs.pairs, inputs.contacts, inputs.punch_breaks, inputs.prior_converged);
        *cost.get_mut("qs_prior").unwrap() += inputs.prior_seconds;
        let start_connected = bond_connected(&world);
        let start_energy: Vec<(f64, f64)> = world.solver.bonds.iter().flatten().map(|b| (b.joint.dissipated, b.damped)).collect();
        let start_momentum = structure_momenta(&world);
        *cost.get_mut("overhead").unwrap() += t_over.elapsed().as_secs_f64() - inputs.prior_seconds;

        if let Some(t) = &template {
            let t_qs = Instant::now();
            let q = branch_frame(&world, t, &[])?;
            capture_branch("qs", &start_connected, &q, &ix, &mut a)?;
            *cost.get_mut("qs_branch").unwrap() += t_qs.elapsed().as_secs_f64();
            if !punch_breaks.is_empty() {
                let t_p = Instant::now();
                let p = branch_frame(&world, t, &punch_breaks)?;
                capture_branch("punch", &start_connected, &p, &ix, &mut a)?;
                a.i("punch_frame", &[], &[f as i32])?;
                *cost.get_mut("punch_branch").unwrap() += t_p.elapsed().as_secs_f64();
            }
        }

        // The oracle's frame.
        let t_oracle = Instant::now();
        let events_before = world.solver.events.len();
        let substeps_before = world.solver.substeps;
        let mut watch = ContactWatch::new(&world);
        crate::full_state::oracle_frame(&mut world, |w| watch.observe(w))?;
        let oracle_seconds = t_oracle.elapsed().as_secs_f64();
        *cost.get_mut("oracle").unwrap() += oracle_seconds;
        let t_over = Instant::now();
        ix.check(&world)?;
        let substeps = world.solver.substeps - substeps_before;
        substeps_total += substeps;

        let mut break_bin = vec![-1i32; ix.e];
        let mut n_breaks = 0usize;
        for ev in &world.solver.events[events_before..] {
            if let SolverEvent::Broken { time, structure, bond, .. } = ev {
                let bin = (((time - t0) / frame_dt) * TIME_BINS as f64).floor().clamp(0.0, (TIME_BINS - 1) as f64) as i32;
                let g = ix.bond0[*structure] + bond;
                if break_bin[g] < 0 {
                    break_bin[g] = bin;
                    n_breaks += 1;
                    break_times.push(*time);
                    let mode = match world.solver.bonds[*structure][*bond].joint.mode {
                        None => 0,
                        Some(stress_ref::joint::FailureMode::Tension) => 1,
                        Some(stress_ref::joint::FailureMode::Shear) => 2,
                        Some(stress_ref::joint::FailureMode::Compression) => 3,
                        Some(stress_ref::joint::FailureMode::Buckling) => 4,
                    };
                    a.i("break_i32", &[4], &[f as i32, g as i32, break_order, mode])?;
                    a.d("break_f64", &[], &[*time])?;
                    break_order += 1;
                }
            }
        }
        a.i("frame_bond_break_bin", &[ix.e], &break_bin)?;
        let peak: Vec<f32> = world.solver.bonds.iter().flatten().map(|b| b.peak_utilization as f32).collect();
        a.f("frame_bond_peak_util", &[ix.e], &peak)?;
        // The previous frame's peak, per frame, so a kept frame never needs a dropped one.
        a.f("frame_bond_peak_util_prev", &[ix.e], &peak_prev)?;
        peak_prev = peak;
        // Per-bond dissipation over the frame (J): joint (fracture, friction, rebar) and
        // dashpot, differenced in f64 (the f32 running totals lose digits).
        let dissipated: Vec<f32> = world.solver.bonds.iter().flatten().zip(&start_energy).flat_map(|(b, &(j, d))| [(b.joint.dissipated - j) as f32, (b.damped - d) as f32]).collect();
        a.f("frame_bond_dissipated", &[ix.e, 2], &dissipated)?;
        let mut contact_summary = Vec::with_capacity(world.impactors.len() * IMPACTOR_CONTACT);
        let n_sub = watch.substeps.max(1) as f64;
        for k in 0..world.impactors.len() {
            let bin = |i: Option<usize>| i.map_or(-1.0, |i| ((i as f64 / n_sub) * TIME_BINS as f64).floor());
            contact_summary.extend([watch.count[k] as f32 / n_sub as f32, bin(watch.first[k]) as f32, bin(watch.last[k]) as f32, watch.structures[k].iter().map(|&s| 1u32 << s.min(30)).sum::<u32>() as f32]);
        }
        a.f("frame_impactor_contact", &[world.impactors.len(), IMPACTOR_CONTACT], &contact_summary)?;
        a.d("frame_impactor_impulse", &[world.impactors.len(), 3], &watch.impulse.iter().flat_map(|j| j.to_array()).collect::<Vec<_>>())?;
        a.d("frame_impactor_peak_force", &[world.impactors.len()], &watch.peak_force)?;
        // Impactor states at k/8 of the frame (k = 1..8; the last is the frame end).
        let mut sub = watch.subframe.clone();
        while sub.len() < 8 {
            sub.push(world.impactors.iter().map(|imp| impactor_row(world.solver.time, imp)).collect());
        }
        let ni = world.impactors.len();
        a.d("frame_impactor_subframe", &[ni, 8, 14], &(0..ni).flat_map(|k| sub.iter().flat_map(move |row| row[k])).collect::<Vec<_>>())?;
        // Sparse bond peaks: bonds whose peak failure index passed 0.2 or that broke.
        for (gi, b) in world.solver.bonds.iter().flatten().enumerate() {
            if b.peak_utilization > 0.2 || break_bin[gi] >= 0 {
                a.i("peak_i32", &[2], &[f as i32, gi as i32])?;
                let l = watch.peak_load[gi];
                let tp = if watch.peak_util[gi].is_finite() { watch.peak_time[gi] } else { world.solver.time };
                a.d("peak_f64", &[9], &[tp, b.peak_utilization, watch.peak_util[gi].max(0.0), l[0], l[1], l[2], l[3], l[4], l[5]])?;
            }
        }
        // Cost of the frame.
        a.i("frame_cost_i32", &[2], &[substeps as i32, watch.contact_limited as i32])?;
        a.d("frame_cost_f64", &[2], &[if watch.dt_min.is_finite() { watch.dt_min } else { 0.0 }, watch.dt_max])?;
        // Wall-clock: not reproducible, so `regenerate` does not compare it.
        a.d("frame_oracle_seconds", &[], &[oracle_seconds])?;
        // Momentum accounting per structure: external-load impulse (gravity, contact and
        // scripted loads) and the supports' reaction impulse (the momentum change less it),
        // linear and angular about the world origin.
        let end_momentum = structure_momenta(&world);
        let ns = world.solver.structures.len();
        let load = &world.solver.load_impulse;
        let mut load_rows = Vec::with_capacity(ns * 6);
        let mut support_rows = Vec::with_capacity(ns * 6);
        for s_i in 0..ns {
            let li = load.get(s_i).copied().unwrap_or([0.0; 6]);
            load_rows.extend(li);
            support_rows.extend((0..6).map(|k| end_momentum[s_i][k] - start_momentum[s_i][k] - li[k]));
        }
        a.d("frame_structure_load_impulse", &[ns, 6], &load_rows)?;
        a.d("frame_structure_support_impulse", &[ns, 6], &support_rows)?;
        // The energy ledger at the frame end (cumulative J).
        {
            let e = &world.solver.energy;
            let c = &world.contact;
            a.d("frame_energy", &[14], &[e.bond_dissipation, e.damping_dissipation, e.external_work, e.softening_overshoot, e.split_release, e.static_discard,
                c.stored, c.dissipated, c.crush, c.work, world.solver.elastic_energy(), structure_ke(&world), world.impactors.iter().map(|i| i.kinetic_energy()).sum(), world.mechanical_energy()])?;
        }
        capture_state(&world, &ix, &mut a)?;
        capture_traj(&world, f + 1, &mut a)?;

        // Phase.
        let touched: HashSet<usize> = watch.structures.iter().flatten().copied().collect();
        let contact_now = watch.count.iter().any(|&c| c > 0);
        if contact_now && !ever_contact {
            ever_contact = true;
            first_struck = touched.clone();
        }
        let phase = if !ever_contact && n_breaks == 0 {
            "pre_contact"
        } else if contact_now && !engagement_over && touched.iter().all(|s| first_struck.contains(s)) {
            first_struck.extend(touched.iter().copied());
            "contact"
        } else if contact_now || n_breaks > 0 {
            engagement_over |= !contact_now || !touched.iter().all(|s| first_struck.contains(s));
            "secondary"
        } else {
            engagement_over |= ever_contact;
            if last_event_frame.is_some_and(|l| f - l < 6) { "separation" } else { "settling" }
        };
        if contact_now || n_breaks > 0 {
            last_event_frame = Some(f);
        }
        if n_breaks > 0 {
            last_break_frame = Some(f);
        }
        *per_phase.get_mut(phase).unwrap() += 1;
        a.i("frame_phase", &[], &[PHASES.iter().position(|p| *p == phase).unwrap() as i32])?;
        if matches!(phase, "contact" | "secondary" | "separation") {
            if let Some((tm, p, v)) = &watch.mid {
                a.i("mid_frame", &[], &[f as i32])?;
                a.d("mid_time", &[], &[*tm])?;
                a.d("mid_chunk_pos", &[ix.n, 3], p)?;
                a.d("mid_chunk_vel", &[ix.n, 3], v)?;
            }
        }

        // Energy ledger and the stopping rule.
        let e = &world.solver.energy;
        let ke_struct = structure_ke(&world);
        let ke_imp: f64 = world.impactors.iter().map(|i| i.kinetic_energy()).sum();
        let (ke_bounds, asleep) = energy_in_bounds(&world, &bounds, &opts.stop);
        if !(ke_struct + ke_imp).is_finite() || ke_struct + ke_imp > opts.energy_blowup * (input_energy + gravity_work_bound) {
            return Err(format!("nonphysical kinetic energy {:.3e} J against input {:.3e} J at frame {f}", ke_struct + ke_imp, input_energy));
        }
        // Stopped: at rest, or at rest against the remnant (the largest piece; without gravity
        // a pushed free wall never comes to absolute rest), or gone.
        let remnant_v = world.solver.clusters.iter().max_by(|a, b| a.mass.total_cmp(&b.mass)).map_or(Vec3::ZERO, |c| c.velocity);
        let impactors_done = world.impactors.iter().all(|i| {
            i.velocity.norm() < opts.stop.stop_speed || (i.velocity - remnant_v).norm() < opts.stop.stop_speed || bounds.left(i.pose.position, i.velocity, gravity)
        });
        let quiet = f + 1 >= opts.stop.quiet_frames as usize && last_break_frame.map_or(true, |l| f - l >= opts.stop.quiet_frames as usize);
        // Checked only when it can end the event (the pair test is quadratic in bodies).
        let coasting = quiet && !(impactors_done && (ke_bounds < opts.stop.ke_fraction * input_energy || asleep))
            && interactions_over(&world, &bounds, &opts.stop, (max_frames - f - 1) as f64 * frame_dt);
        frames.push(json!({
            "frame": f, "time": t0, "phase": phase, "breaks": n_breaks, "substeps": substeps,
            "routed_chunks": chunk_routed.iter().filter(|&&r| r == 1).count(), "routed_impactors": imp_routed.iter().filter(|&&r| r == 1).count(),
            "predicted_contacts": contacts.len(), "near_pairs": pairs_n, "punch_bonds": punch_breaks.len(), "qs_prior_converged": prior_converged,
            "impactor_contact": contact_now, "struck_structures": touched.iter().copied().collect::<std::collections::BTreeSet<_>>(),
            "energy": {"kinetic_structure": ke_struct, "kinetic_impactors": ke_imp, "kinetic_in_bounds": ke_bounds,
                "mechanical": world.mechanical_energy(), "dissipated": world.dissipated_energy(),
                "bond_dissipation": e.bond_dissipation, "damping_dissipation": e.damping_dissipation, "external_work": e.external_work,
                "softening_overshoot": e.softening_overshoot, "split_release": e.split_release, "static_discard": e.static_discard,
                "contact_stored": world.contact.stored, "contact_dissipated": world.contact.dissipated, "contact_crush": world.contact.crush,
                "elastic": world.solver.elastic_energy()},
            "asleep": asleep, "coasting": coasting,
        }));
        let calm = ke_bounds < opts.stop.ke_fraction * input_energy || asleep || coasting;
        *cost.get_mut("overhead").unwrap() += t_over.elapsed().as_secs_f64();
        if quiet && ((impactors_done && calm) || coasting) {
            let impactors = if world.impactors.iter().all(|i| i.velocity.norm() < opts.stop.stop_speed || (i.velocity - remnant_v).norm() < opts.stop.stop_speed) {
                "stopped"
            } else if impactors_done {
                "stopped or left"
            } else {
                "coasting"
            };
            let why = if !impactors_done || !calm {
                "no two bodies can meet before the cap (no gravity)".to_string()
            } else if ke_bounds < opts.stop.ke_fraction * input_energy {
                format!("KE in bounds {:.2}% of input", 100.0 * ke_bounds / input_energy)
            } else if asleep {
                "asleep".into()
            } else {
                "no two bodies can meet before the cap (no gravity)".into()
            };
            stop_reason = format!("impactors {impactors}; no break for {} frames; {why}", opts.stop.quiet_frames);
            break;
        }
        if f + 1 == max_frames {
            truncated = true;
            stop_reason = format!("cap {} s (impactors done {impactors_done}, quiet {quiet}, calm {calm})", opts.stop.max_seconds);
        }
        beat(&Beat { event_id: &recipe.event_id, frame: f + 1, time: world.solver.time, phase, breaks: break_times.len() });
    }

    // The last boundary.
    if let Some(fs) = full_state.as_mut() {
        fs.push(&dir, frames.len(), &world)?;
    }

    // Full-run outcome.
    let sim_seconds = world.solver.time;
    let total_mass: f64 = world.solver.structures.iter().flat_map(|s| s.chunks.iter()).map(|c| c.mass).sum();
    let largest = world.solver.clusters.iter().map(|cl| cl.mass).fold(0.0, f64::max);
    let breaks_last = break_times.iter().filter(|&&t| t >= sim_seconds - 0.5).count();
    // Regimes are geometric, relative to the remnant (the largest fragment at the end): an
    // impactor that ended beyond the remnant's far extent along its approach and moving on
    // perforated; one moving back from it rebounded; one moving with it (or stopped)
    // arrested; anything else deflected.
    let remnant = world.solver.clusters.iter().max_by(|a, b| a.mass.total_cmp(&b.mass)).expect("a cluster");
    let impactors: Vec<Value> = world
        .impactors
        .iter()
        .zip(&v0)
        .map(|(imp, v0)| {
            let d = v0.normalized();
            let c = remnant.com_world();
            let s = remnant.structure;
            let far = remnant.chunks.iter().map(|&k| (world.solver.chunk_position(s, k) - c).dot(d) + world.solver.structures[s].chunks[k].half_extents.norm()).fold(f64::NEG_INFINITY, f64::max);
            let behind = (imp.pose.position - c).dot(d);
            let rel = (imp.velocity - remnant.velocity).dot(d);
            let along = imp.velocity.dot(d);
            let regime = if behind > far && rel > -opts.stop.stop_speed && along > opts.stop.stop_speed {
                "perforate"
            } else if rel < -opts.stop.stop_speed {
                "rebound"
            } else if rel.abs() <= opts.stop.stop_speed.max(0.05 * v0.norm()) {
                "arrest"
            } else {
                "deflect"
            };
            json!({"name": imp.name, "v0": v0.to_array(), "v_end": imp.velocity.to_array(), "speed_along_v0": along, "speed_relative_to_remnant": rel,
                "beyond_remnant_m": behind - far, "regime": regime, "position_end": imp.pose.position.to_array()})
        })
        .collect();
    let summary = OutcomeSummary {
        event_id: recipe.event_id.clone(),
        family: recipe.family.clone(),
        tier: recipe.tier,
        frames: frames.len(),
        sim_seconds,
        truncated,
        stop_reason,
        broken_bonds: break_times.len(),
        fragments: world.solver.clusters.len(),
        largest_fraction: largest / total_mass,
        last_break_time: break_times.iter().copied().fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t)))),
        breaks_last_half_second: breaks_last,
        input_energy,
        impactors,
        frames_per_phase: per_phase,
        cost: {
            let mut c = cost;
            c.insert("event".into(), t_event.elapsed().as_secs_f64());
            c
        },
        oracle_substeps: substeps_total,
        fingerprint: fingerprint(&world),
    };

    // Files.
    let write_json = |name: &str, v: &Value| -> Result<(), String> { fs::write(dir.join(name), serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?).map_err(|e| format!("{name}: {e}")) };
    // Storage: training events that are not ensemble members keep only the frames training uses.
    let split = recipe.parameters.get("split").and_then(|x| x.as_str()).unwrap_or("");
    let member = recipe.parameters.get("ensemble").and_then(|e| e.get("id")).is_some_and(|x| !x.is_null());
    let selection = (opts.storage.select_training_frames && split == "train" && !member).then(|| {
        let phases: Vec<&str> = frames.iter().map(|x| x["phase"].as_str().unwrap_or("")).collect();
        let breaks: Vec<usize> = frames.iter().map(|x| x["breaks"].as_u64().unwrap_or(0) as usize).collect();
        select_frames(&phases, &breaks, opts.storage.settling_stride)
    });
    let mut cat = statics.write(&dir, None, opts.storage.compress)?;
    let frame_cat = a.write(&dir, selection.as_ref(), opts.storage.compress)?;
    cat.arrays.extend(frame_cat.arrays);
    cat.save(&dir.join("catalog.json"))?;
    write_json("scene.json", &serde_json::to_value(&scene).map_err(|e| e.to_string())?)?;
    let chunk_keys: Vec<Value> = world.solver.structures.iter().enumerate().flat_map(|(s, st)| (0..st.chunks.len()).map(move |c| json!([s, c]))).collect();
    let bond_keys: Vec<Value> = world.solver.bonds.iter().enumerate().flat_map(|(s, bs)| bs.iter().map(move |b| json!([s, b.source, b.geometry.a, b.geometry.b]))).collect();
    write_json("context.json", &json!({
        "schema": OUTCOME_SCHEMA, "recorder_revision": RECORDER_REVISION,
        "oracle_config": opts.oracle, "qs_baseline": if opts.baselines { Some(QS_BASELINE) } else { None },
        "punch": {"model": "World::predict_punch (CPU)", "tiered_impact_speed": tiered_impact_speed()},
        "stop_rule": opts.stop, "time_bins": TIME_BINS, "phases": PHASES,
        "event_id": recipe.event_id, "family": recipe.family, "tier": recipe.tier, "split_group": recipe.split_group,
        "layout_group": recipe.layout_group, "parameters": recipe.parameters, "scene_hash": scene_hash(&scene),
        "structures": world.solver.structures.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
        "chunk_keys": chunk_keys, "bond_keys": bond_keys,
        "n_chunks": ix.n, "n_bonds": ix.e, "n_impactors": world.impactors.len(),
        "contact_f32_columns": ["time","point_x","point_y","point_z","dir_x","dir_y","dir_z","speed","reduced_mass","modulus","face_radius","hertz_duration","peak_force","punch_ratio","plug_mass","punch_breaks","cluster_chunks"],
        "contact_i32_columns": ["frame","impactor","structure","entry_chunk","cluster_index"],
        "pair_i32_columns": ["frame","chunk_a","chunk_b"],
        "traj_cluster_i32_columns": ["boundary","cluster_index","n_chunks"],
        "traj_cluster_f32_columns": ["mass","com_x","com_y","com_z","v_x","v_y","v_z","w_x","w_y","w_z"],
        "impactor_contact_columns": ["contact_substep_fraction","first_bin","last_bin","structure_mask"],
        "bond_joint_columns": ["damage","crush","fatigue","utilization"],
        "bond_energy_columns": ["dissipated_joint","dissipated_damping","stored"],
        "bond_dissipated_columns": ["joint","damping"],
        "break_i32_columns": ["frame","bond","order","mode"],
        "break_f64_columns": ["time"],
        "peak_i32_columns": ["frame","bond"],
        "peak_f64_columns": ["time_of_peak","peak_utilization","observed_peak_utilization","shear_t1","shear_t2","axial","bending_t1","bending_t2","torsion"],
        "peak_rule": "bonds whose peak failure index over the frame exceeds 0.2 or that broke in it; the generalised force (N, N m, bond frame) at the substep end with the largest failure index seen",
        "impactor_subframe_columns": ["time","pos_x","pos_y","pos_z","q_w","q_x","q_y","q_z","v_x","v_y","v_z","w_x","w_y","w_z"],
        "impactor_subframe_rule": "the first substep end at or past k/8 of the frame, k = 1..8",
        "mid_rule": "contact, secondary and separation frames: every chunk's position and velocity at the first substep end at or past mid-frame (mid_time)",
        "cost_i32_columns": ["substeps","contact_limited_substeps"],
        "cost_f64_columns": ["dt_min","dt_max"],
        "oracle_seconds": "frame_oracle_seconds: wall-clock seconds of the oracle's frame (all threads; not reproducible)",
        "structure_impulse_columns": ["x","y","z","about_origin_x","about_origin_y","about_origin_z"],
        "support_rule": "frame_structure_support_impulse = the change of the structure's momentum (active chunks, hidden motion included; angular about the world origin) less frame_structure_load_impulse (gravity, contact and scripted loads, per substep): the supports' reaction (~0 for a structure without supports)",
        "energy_columns": ["bond_dissipation","damping_dissipation","external_work","softening_overshoot","split_release","static_discard","contact_stored","contact_dissipated","contact_crush","contact_work","elastic","ke_structure","ke_impactors","mechanical"],
        "full_state": full_state.as_ref().map(|fs| json!({
            "schema": crate::full_state::FULL_STATE_SCHEMA,
            "format": "full_state/<boundary>.fs.zst: the World's every serialized field, one column per field path (stress_learned::full_state); delta_of: zstd with that boundary's encoding as dictionary; full_state::load_frame decodes and full_state::restore rebuilds the World (with scene.json), bit for bit",
            "boundaries": fs.entries, "raw_bytes": fs.raw_bytes, "stored_bytes": fs.stored_bytes})),
        "provenance": provenance(),
        "bond_mode_values": ["none","tension","shear","compression","buckling"],
        "ensemble": recipe.parameters.get("ensemble").cloned().unwrap_or(Value::Null),
        "split": recipe.parameters.get("split").cloned().unwrap_or(Value::Null),
        "storage": {"compress": opts.storage.compress, "select_training_frames": opts.storage.select_training_frames,
            "full_state": opts.storage.full_state, "full_state_every": opts.storage.full_state_every, "full_state_level": opts.storage.full_state_level,
            "settling_stride": opts.storage.settling_stride, "frames": frames.len(),
            "kept_frames": selection.as_ref().map(|x| x.frames.clone()), "state_rows": selection.as_ref().map(|x| x.state_rows.clone()),
            "dropped_frames": selection.as_ref().map_or(0, |x| frames.len() - x.frames.len()),
            "regenerate": "learned_impact_outcome_record --regenerate <event dir> --out <dir> (bit-identical replay of the recipe)"},
        "energy_notes": "Under the penalty contact (layer_contact off) the booked contact dissipation (contact_dissipated) is the dashpot quadrature, not its work: the ledger does not close (+0.3% to +4.7% of E0, P0). Use kinetic energies (exact) and per-bond energies as labels; contact_dissipated is a diagnostic only.",
        "bond_geometry_columns": ["centroid_x","centroid_y","centroid_z","normal_x","normal_y","normal_z","t1_x","t1_y","t1_z","area","width_0","width_1","length"],
        "bond_stiffness_columns": ["kn","ks","kb_t1","kb_t2","kt"],
        "bond_strength_columns": ["tensile","compressive","cohesion","friction","shear_cap","g_tension","g_shear","g_compression"],
    }))?;
    write_json("replay.json", &json!({
        "recipe": recipe, "oracle_config": opts.oracle.name, "frames": summary.frames,
        "input_energy": input_energy, "end_fingerprint": summary.fingerprint, "sim_seconds": sim_seconds,
    }))?;
    write_json("frames.json", &Value::Array(frames))?;
    write_json("outcome.json", &serde_json::to_value(&summary).map_err(|e| e.to_string())?)?;
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir).map_err(|e| e.to_string())?;
    }
    fs::rename(&dir, &final_dir).map_err(|e| format!("finish {}: {e}", final_dir.display()))?;
    Ok(summary)
}

/// Replay an event's recipe (from `replay.json`) under the oracle configuration recorded in
/// its `context.json`, oracle only, and return the end fingerprint and frame count.
pub fn replay_fingerprint(event: &Path) -> Result<(String, usize), String> {
    let read = |n: &str| -> Result<Value, String> { serde_json::from_slice(&fs::read(event.join(n)).map_err(|e| format!("{n}: {e}"))?).map_err(|e| format!("{n}: {e}")) };
    let ctx = read("context.json")?;
    let replay = read("replay.json")?;
    let oracle: OracleConfig = serde_json::from_value(ctx["oracle_config"].clone()).map_err(|e| e.to_string())?;
    let recipe: OutcomeRecipe = serde_json::from_value(replay["recipe"].clone()).map_err(|e| e.to_string())?;
    let frames = replay["frames"].as_u64().ok_or("replay.json: frames")? as usize;
    let mut scene = recipe.scene;
    oracle.apply(&mut scene);
    let stop: StopRule = serde_json::from_value(ctx["stop_rule"].clone()).map_err(|e| e.to_string())?;
    scene.sim.duration = (stop.max_seconds / scene.sim.frame_dt).round() * scene.sim.frame_dt;
    let mut w = World::new(&scene);
    for _ in 0..frames {
        w.step_frame();
    }
    Ok((fingerprint(&w), frames))
}

/// The event directories under `out` that hold a finished recording.
pub fn finished_events(out: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(out).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.join("outcome.json").exists()).collect();
    v.sort();
    v
}

/// Where a recording came from: the source commit the binary was built from, the
/// binary's own hash (FNV-1a 64 of its bytes) and the revisions.
pub fn provenance() -> Value {
    static BIN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let bin = BIN.get_or_init(|| {
        std::env::current_exe().ok().and_then(|p| fs::read(p).ok()).map_or("unknown".into(), |b| {
            let mut h: u64 = 0xcbf29ce484222325;
            for x in b {
                h = (h ^ u64::from(x)).wrapping_mul(0x100000001b3);
            }
            format!("{h:016x}")
        })
    });
    json!({"git": env!("STRESS_LEARNED_GIT"), "binary_fnv64": bin, "recorder_revision": RECORDER_REVISION, "full_state_schema": crate::full_state::FULL_STATE_SCHEMA,
        "seeds_and_perturbations": "context.json parameters (recipe seed, ensemble perturbation and Weibull seed) and replay.json recipe"})
}

/// Arrays that differ run to run (wall-clock time); everything else regenerates bit for bit.
pub const NOT_REPRODUCIBLE: &[&str] = &["frame_oracle_seconds"];

/// What `regenerate` restored and checked.
#[derive(Clone, Debug, Serialize)]
pub struct Regenerated {
    pub event: PathBuf,
    pub frames: usize,
    pub kept_frames: usize,
    pub arrays_checked: usize,
    pub rows_checked: usize,
}

/// Re-record a (possibly thinned) event from its recipe with every frame kept, into
/// `out/<event_id>`, and check that every row the stored event holds is byte-identical in
/// the regenerated one (the oracle is deterministic; scenes round-trip exactly). Errors on
/// the first difference.
pub fn regenerate(event: &Path, out: &Path) -> Result<Regenerated, String> {
    let read = |n: &str| -> Result<Value, String> { serde_json::from_slice(&fs::read(event.join(n)).map_err(|e| format!("{n}: {e}"))?).map_err(|e| format!("{n}: {e}")) };
    let ctx = read("context.json")?;
    let replay = read("replay.json")?;
    let recipe: OutcomeRecipe = serde_json::from_value(replay["recipe"].clone()).map_err(|e| e.to_string())?;
    let mut opts = RecordOptions::new(serde_json::from_value(ctx["oracle_config"].clone()).map_err(|e| e.to_string())?);
    opts.stop = serde_json::from_value(ctx["stop_rule"].clone()).map_err(|e| e.to_string())?;
    opts.baselines = !ctx["qs_baseline"].is_null();
    opts.storage = Storage { compress: ctx["storage"]["compress"].as_bool().unwrap_or(false), select_training_frames: false, settling_stride: 1, ..Storage::default() };
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let summary = record_outcome_event(&recipe, out, &opts, &|_| {})?;
    let fresh = out.join(&recipe.event_id);
    let fp = |dir: &Path| -> Result<String, String> { Ok(serde_json::from_slice::<Value>(&fs::read(dir.join("outcome.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?["fingerprint"].as_str().unwrap_or("").to_string()) };
    if fp(event)? != summary.fingerprint {
        return Err(format!("regenerated end fingerprint {} differs from the stored {}", summary.fingerprint, fp(event)?));
    }
    let (old_cat, new_cat) = (Catalog::load(&event.join("catalog.json"))?, Catalog::load(&fresh.join("catalog.json"))?);
    let rows_of = |k: &str| ctx["storage"][k].as_array().map(|v| v.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect::<Vec<_>>());
    let (kept, state_rows) = (rows_of("kept_frames"), rows_of("state_rows"));
    let new_bytes = |n: &str| -> Result<(Vec<u8>, usize), String> {
        let e = new_cat.arrays.get(n).ok_or_else(|| format!("regenerated event lacks {n}"))?;
        let b = catalog::read_bytes(&fresh, e)?;
        let w = b.len() / e.shape[0].max(1);
        Ok((b, w))
    };
    let frame_col = |n: &str| -> Result<Option<Vec<i32>>, String> {
        match new_cat.arrays.get(n) {
            Some(e) => {
                let d = catalog::read_i32(&fresh, e)?;
                let w = d.len() / e.shape[0].max(1);
                Ok(Some((0..e.shape[0]).map(|r| d[r * w]).collect()))
            }
            None => Ok(None),
        }
    };
    let (contact_f, pair_f, punch_f, break_f) = (frame_col("contact_i32")?, frame_col("pair_i32")?, frame_col("punch_frame")?, frame_col("break_i32")?);
    let (peak_f, mid_f) = (frame_col("peak_i32")?, frame_col("mid_frame")?);
    let (mut arrays_checked, mut rows_checked) = (0, 0);
    for (name, e) in &old_cat.arrays {
        if NOT_REPRODUCIBLE.contains(&name.as_str()) {
            continue;
        }
        let stored = catalog::read_bytes(event, e)?;
        let (fresh_bytes, w) = new_bytes(name)?;
        let pick = |rows: Vec<usize>| -> Vec<u8> { rows.iter().flat_map(|&r| fresh_bytes[r * w..(r + 1) * w].to_vec()).collect() };
        let by_frame = |col: &Option<Vec<i32>>| -> Vec<usize> {
            match (col, &kept) {
                (Some(c), Some(k)) => (0..c.len()).filter(|&r| k.binary_search(&(c[r] as usize)).is_ok()).collect(),
                (Some(c), None) => (0..c.len()).collect(),
                _ => Vec::new(),
            }
        };
        let expect = match (&kept, &state_rows) {
            (Some(k), Some(srows)) if !name.starts_with("static_") && !name.starts_with("traj_") => {
                if name.starts_with("state_") {
                    pick(srows.clone())
                } else if name.starts_with("frame_") || name.starts_with("qs_") {
                    pick(k.clone())
                } else if name.starts_with("punch_") {
                    pick(by_frame(&punch_f))
                } else if name.starts_with("contact_") {
                    pick(by_frame(&contact_f))
                } else if name.starts_with("pair_") {
                    pick(by_frame(&pair_f))
                } else if name.starts_with("break_") {
                    pick(by_frame(&break_f))
                } else if name.starts_with("peak_") {
                    pick(by_frame(&peak_f))
                } else if name.starts_with("mid_") {
                    pick(by_frame(&mid_f))
                } else {
                    fresh_bytes.clone()
                }
            }
            _ => fresh_bytes.clone(),
        };
        if expect != stored {
            return Err(format!("{name}: the regenerated rows differ from the stored ones"));
        }
        arrays_checked += 1;
        rows_checked += e.shape.first().copied().unwrap_or(0);
    }
    Ok(Regenerated { event: fresh, frames: summary.frames, kept_frames: kept.map_or(summary.frames, |k| k.len()), arrays_checked, rows_checked })
}
