//! Harness A relabelling (DAgger, P2): at a frame of a recorded event, the oracle's island
//! step from a given island state. The event's world is replayed natively to the frame (the
//! standalone oracle, as the recorder ran it: each frame's peak utilisation reset at its
//! start), the provided `IslandState` imposed on its island (a model's prediction, say), and
//! the oracle's island step taken (`stress_ref::world::oracle_island_step`). Returned: the
//! state the step started from, the oracle's frame-end state (the label), the island's
//! members, and the call in the calls format with the oracle's frame as its targets
//! (`outcome_calls::live_call_labelled`, the recorder's own frame capture).
//!
//! Deterministic (CPU f64). Batched per event: one replay serves every requested frame.
//! Island states are MessagePack files (`write_state` / `read_state`, field names kept):
//! exact, every f64 bit and infinity kept, and readable from Python (`msgpack`). (A model in Python produces its state in P1's full-state arrays; the
//! converter to `IslandState` follows that format.)

use std::path::Path;

use serde_json::Value;
use stress_ref::scene::Scene;
use stress_ref::world::{oracle_island_step, DecodedState, IslandMembers, IslandState, World};

use crate::outcome_calls::{live_call_full, live_call_labelled, CallEvent};

/// What a relabel job starts the oracle's frame from.
#[derive(Clone)]
pub enum Provided {
    /// The oracle's own state of the routing's island at the job's frame.
    Own,
    /// An island state imposed at the job's frame start (`IslandState`, exact).
    Start(IslandState),
    /// A model's decoded outcome of the job's frame (`learned-impact-decoded-state/1`): as
    /// harness A takes it (applied to the island on a copy of the frame-start world, the
    /// island's state taken, the rest the oracle's native frame), the world reaches the next
    /// frame, and the relabelled call is that frame's (`Relabelled::frame` = job frame + 1).
    Decoded(DecodedState),
}

pub struct Relabelled {
    /// The relabelled call's frame.
    pub frame: usize,
    pub members: IslandMembers,
    /// The island's state the oracle stepped from (the provided one, or the oracle's own).
    pub start: IslandState,
    /// The oracle's frame-end island state.
    pub end: IslandState,
    /// The call (`learned-impact-outcome/4`): inputs from the frame-start world as imposed,
    /// targets the oracle's frame (`outcome_calls::live_call_labelled`).
    pub call: Option<CallEvent>,
}

/// An island state as bytes (MessagePack with field names, `rmp_serde::to_vec_named`).
pub fn state_bytes(st: &IslandState) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec_named(st).map_err(|e| e.to_string())
}

pub fn write_state(path: &Path, st: &IslandState) -> Result<(), String> {
    std::fs::write(path, state_bytes(st)?).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_state(path: &Path) -> Result<IslandState, String> {
    rmp_serde::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| format!("{}: {e}", path.display()))
}

/// The event's scene (`scene.json` as recorded).
pub fn event_scene(dir: &Path) -> Result<Scene, String> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("scene.json")).map_err(|e| format!("{}: {e}", dir.display()))?).map_err(|e| e.to_string())?;
    serde_json::from_value(v).map_err(|e| format!("scene.json: {e}"))
}

/// The recorded event's context (`context.json`), the call manifest's provenance; empty
/// when the directory has none (a bare scene).
pub fn event_context(dir: &Path) -> Value {
    std::fs::read(dir.join("context.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_else(|| serde_json::json!({}))
}

/// The harness A frame with a decoded outcome (`World::step_frame_island_provider`, the
/// decoded state applied to the island on a copy of the frame-start world, as
/// `learned_p0 --host a` applies a model's).
fn decoded_frame(w: &mut World, d: &DecodedState) -> Result<(), String> {
    let mut provider = |start: &World, m: &IslandMembers| -> Result<IslandState, String> {
        let mut x = start.clone();
        x.solver.reset_peak_utilizations();
        x.apply_decoded(d)?;
        Ok(x.extract_island(m))
    };
    w.step_frame_island_provider(&mut provider).map(|_| ())
}

/// Relabel `jobs` (frame, what to start from) of the event in `dir`. One replay of the
/// event's native oracle serves every job.
pub fn relabel_event(dir: &Path, jobs: &[(usize, Provided)], with_calls: bool) -> Result<Vec<Relabelled>, String> {
    relabel_event_with(dir, jobs, with_calls, None)
}

/// `relabel_event`; with `full_state` the calls are `learned-impact-outcome/5`
/// (`outcome_calls::live_call_full`): `Some(columns)` a training manifest's
/// `full_state_columns`, `Some(Null)` the call's own split.
pub fn relabel_event_with(dir: &Path, jobs: &[(usize, Provided)], with_calls: bool, full_state: Option<&Value>) -> Result<Vec<Relabelled>, String> {
    relabel_event_jobs(dir, jobs, with_calls, full_state)?.into_iter().collect()
}

/// `relabel_event_with`, each job's result on its own: a job that fails (a state the oracle or
/// the call builder rejects, e.g. a blown-up model state) is its error, and the others go on.
pub fn relabel_event_jobs(dir: &Path, jobs: &[(usize, Provided)], with_calls: bool, full_state: Option<&Value>) -> Result<Vec<Result<Relabelled, String>>, String> {
    let scene = event_scene(dir)?;
    let ctx = event_context(dir);
    let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| scene.name.clone());
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&k| jobs[k].0);
    let mut out: Vec<Option<Result<Relabelled, String>>> = (0..jobs.len()).map(|_| None).collect();
    let mut w = World::new(&scene);
    let mut at = 0usize;
    for k in order {
        let (frame, provided) = &jobs[k];
        while at < *frame {
            w.solver.reset_peak_utilizations();
            w.step_frame();
            at += 1;
        }
        out[k] = Some(relabel_one(&w, *frame, provided, with_calls, full_state, &name, &ctx).map_err(|e| format!("{name} frame {frame}: {e}")));
    }
    Ok(out.into_iter().map(|r| r.expect("every job relabelled")).collect())
}

/// One job on the replayed world `w` at its frame.
fn relabel_one(w: &World, frame: usize, provided: &Provided, with_calls: bool, full_state: Option<&Value>, name: &str, ctx: &Value) -> Result<Relabelled, String> {
    let mut x = w.clone();
    let mut call_frame = frame;
    if let Provided::Decoded(d) = provided {
        decoded_frame(&mut x, d)?;
        call_frame += 1;
    }
    // (The frame-start peak utilisation is the previous frame's, as recorded: the call
    // reads it; the oracle's frame resets it at its start.)
    let members = match provided {
        Provided::Start(s) => s.members(),
        _ => x.island_members(),
    };
    if let Provided::Start(s) = provided {
        x = World::merge_island(&x, s, &members);
    }
    let start = x.extract_island(&members);
    let end = oracle_island_step(&x, &members)?;
    let call = if with_calls && !members.chunks.is_empty() {
        let mut ck: Vec<i64> = members.chunks.iter().map(|&(s, c)| stress_ref::world::stable_key(s, c)).collect();
        ck.sort_unstable();
        let mut ik: Vec<i64> = members.impactors.iter().map(|&i| stress_ref::keys::impactor_key(i)).collect();
        ik.sort_unstable();
        let mut stepped = x.clone();
        let ev = match full_state {
            None => live_call_labelled(&mut stepped, name, ctx, &ck, &ik, call_frame, true, true)?,
            Some(cols) => live_call_full(&mut stepped, name, ctx, &ck, &ik, call_frame, true, true, (!cols.is_null()).then_some(cols))?,
        };
        // The recorder's frame is the oracle's island step, bit for bit.
        if state_bytes(&stepped.extract_island(&members))? != state_bytes(&end)? {
            return Err("the recorded frame differs from the oracle's island step".into());
        }
        Some(ev)
    } else {
        None
    };
    Ok(Relabelled { frame: call_frame, members, start, end, call })
}
