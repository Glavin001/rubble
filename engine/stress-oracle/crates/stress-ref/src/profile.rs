//! Where a run spends its time and work: per-stage wall-clock timers and work counters
//! (`STRESS_PROFILE=1`, or `stress-ref run --profile`). Off by default; when off a timer
//! is one relaxed atomic load. Never part of an observation: profiling changes no result.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Stages of a frame and a substep. Timed from the thread that runs the step (the
/// parallel sections are timed as a whole), so the stages nest as listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    StepDt,
    Events,
    ScriptedLoads,
    Contacts,
    ContactBroadphase,
    PairSearch,
    ContactPairs,
    ContactImpactors,
    ContactGround,
    Wake,
    SolverSubstep,
    RigidAccel,
    BondResponses,
    ApplyBonds,
    ChunkLoads,
    Integrate,
    Splits,
    Handover,
    PairEval,
    PairApply,
    PairOverlap,
    PairCells,
    PairGeometry,
    PairLaw,
    ContactWork,
    Probes,
    FrameTail,
    Statics,
    Implicit,
    Settle,
    Refine,
}

const STAGES: [(Stage, &str, usize); 31] = [
    (Stage::StepDt, "step dt (stable_dt, contact dt)", 0),
    (Stage::Events, "events", 0),
    (Stage::ScriptedLoads, "scripted loads", 0),
    (Stage::Contacts, "contacts", 0),
    (Stage::ContactBroadphase, "broadphase", 1),
    (Stage::PairSearch, "pair search (chunk_pairs or neighbour list)", 2),
    (Stage::ContactPairs, "chunk pairs", 1),
    (Stage::ContactImpactors, "impactors", 1),
    (Stage::ContactGround, "ground", 1),
    (Stage::PairEval, "pair evaluation (all threads)", 2),
    (Stage::PairApply, "pair application", 2),
    (Stage::PairOverlap, "pair overlap (intersect; all threads)", 2),
    (Stage::PairCells, "pair field cells (all threads)", 2),
    (Stage::PairGeometry, "pair contact geometry (all threads)", 2),
    (Stage::PairLaw, "pair force law (all threads)", 2),
    (Stage::Wake, "adaptive wake", 0),
    (Stage::SolverSubstep, "solver substep", 0),
    (Stage::RigidAccel, "rigid accelerations", 1),
    (Stage::BondResponses, "bond responses", 1),
    (Stage::ApplyBonds, "apply bond responses", 1),
    (Stage::ChunkLoads, "chunk frame loads", 1),
    (Stage::Integrate, "integration", 1),
    (Stage::Splits, "splits", 1),
    (Stage::Handover, "split handover", 0),
    (Stage::ContactWork, "contact work, impactors", 0),
    (Stage::Probes, "probes", 0),
    (Stage::FrameTail, "frame tail", 0),
    (Stage::Statics, "statics", 1),
    (Stage::Implicit, "implicit steps", 1),
    (Stage::Settle, "settle, fatigue", 1),
    (Stage::Refine, "refinement", 1),
];

/// Work counted during a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    Frames,
    Substeps,
    ExplicitClusters,
    BondEvaluations,
    BroadphaseChunks,
    SphereTests,
    StaticSolves,
    StaticConverged,
    NewtonIterations,
    CgIterations,
    DirectSolves,
    StaticAtRoundoff,
    PairCandidates,
    PairOverlaps,
    PolytopeClips,
    FieldCells,
    Splits,
    SplitChunks,
    StepByStress,
    StepByContact,
    StepByFrame,
    StepByMaxSubstep,
    NeighbourBuilds,
    NeighbourPairs,
}

const COUNTERS: [(Counter, &str); 24] = [
    (Counter::Frames, "frames"),
    (Counter::Substeps, "substeps"),
    (Counter::ExplicitClusters, "explicit cluster-substeps"),
    (Counter::BondEvaluations, "bond evaluations"),
    (Counter::BroadphaseChunks, "broadphase chunks (per substep, summed)"),
    (Counter::SphereTests, "broadphase interval overlaps (cluster and sphere tests)"),
    (Counter::StaticSolves, "static solves (cluster equilibria)"),
    (Counter::StaticConverged, "static solves converged"),
    (Counter::NewtonIterations, "static Newton iterations"),
    (Counter::CgIterations, "static CG iterations"),
    (Counter::DirectSolves, "static direct (banded) solves"),
    (Counter::PairCandidates, "chunk-pair candidates"),
    (Counter::PairOverlaps, "chunk-pair overlaps"),
    (Counter::PolytopeClips, "polytope clips"),
    (Counter::FieldCells, "field cells"),
    (Counter::Splits, "cluster splits"),
    (Counter::SplitChunks, "chunks in split clusters"),
    (Counter::StaticAtRoundoff, "static solves accepted at round-off"),
    (Counter::StepByStress, "frames stepped by the stress bound"),
    (Counter::StepByContact, "frames stepped by the contact bound"),
    (Counter::StepByFrame, "frames stepped by the frame"),
    (Counter::StepByMaxSubstep, "frames stepped by max_substep"),
    (Counter::NeighbourBuilds, "contact neighbour list builds"),
    (Counter::NeighbourPairs, "contact neighbour list pairs (per substep, summed)"),
];

static ENABLED: AtomicBool = AtomicBool::new(false);
static FROM_ENV: OnceLock<()> = OnceLock::new();
static NANOS: [AtomicU64; 31] = [const { AtomicU64::new(0) }; 31];
static COUNTS: [AtomicU64; 24] = [const { AtomicU64::new(0) }; 24];

/// Turns profiling on (also on when `STRESS_PROFILE` is set).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    FROM_ENV.get_or_init(|| {
        if std::env::var_os("STRESS_PROFILE").is_some() {
            ENABLED.store(true, Ordering::Relaxed);
        }
    });
    ENABLED.load(Ordering::Relaxed)
}

fn stage_index(s: Stage) -> usize {
    STAGES.iter().position(|e| e.0 == s).expect("listed")
}

fn counter_index(c: Counter) -> usize {
    COUNTERS.iter().position(|e| e.0 == c).expect("listed")
}

/// Times `stage` until dropped.
pub struct Timer(Option<(usize, Instant)>);

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some((i, t)) = self.0 {
            NANOS[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

#[inline]
pub fn time(stage: Stage) -> Timer {
    Timer(enabled().then(|| (stage_index(stage), Instant::now())))
}

#[inline]
pub fn count(c: Counter, n: u64) {
    if enabled() {
        COUNTS[counter_index(c)].fetch_add(n, Ordering::Relaxed);
    }
}

/// Clears every timer and counter.
pub fn reset() {
    NANOS.iter().chain(COUNTS.iter()).for_each(|a| a.store(0, Ordering::Relaxed));
}

/// The timers (seconds, with their share of the total of the top-level stages) and
/// counters, as a table.
pub fn report() -> String {
    let secs: Vec<f64> = NANOS.iter().map(|a| a.load(Ordering::Relaxed) as f64 * 1e-9).collect();
    let total: f64 = STAGES.iter().zip(&secs).filter(|(e, _)| e.2 == 0).map(|(_, s)| s).sum();
    let mut out = format!("profile: {total:.3} s in timed stages\n");
    for (e, s) in STAGES.iter().zip(&secs) {
        if *s > 0.0 {
            out += &format!("  {}{:<34} {:9.3} s {:5.1}%\n", "  ".repeat(e.2), e.1, s, 100.0 * s / total.max(1e-300));
        }
    }
    for (e, a) in COUNTERS.iter().zip(&COUNTS) {
        let n = a.load(Ordering::Relaxed);
        if n > 0 {
            out += &format!("  {:<38} {n}\n", e.1);
        }
    }
    out
}
