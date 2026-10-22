//! Harness A relabelling (DAgger, P2): at a frame of a recorded event, the oracle's island
//! step from a given island state. The event's world is replayed natively to the frame (the
//! standalone oracle, as the recorder ran it: each frame's peak utilisation reset at its
//! start), the provided `IslandState` imposed on its island (a model's prediction, say), and
//! the oracle's island step taken (`stress_ref::world::oracle_island_step`). Returned: the
//! state the step started from, the oracle's frame-end state (the label), the island's
//! members, and the call's inputs in the calls format (`outcome_calls::live_call`).
//!
//! Deterministic (CPU f64). Batched per event: one replay serves every requested frame.
//! Island states are MessagePack files (`write_state` / `read_state`, field names kept):
//! exact, every f64 bit and infinity kept, and readable from Python (`msgpack`). (A model in Python produces its state in P1's full-state arrays; the
//! converter to `IslandState` follows that format.)

use std::path::Path;

use serde_json::Value;
use stress_ref::scene::Scene;
use stress_ref::world::{oracle_island_step, IslandMembers, IslandState, World};

use crate::outcome_calls::{live_call, CallEvent};

pub struct Relabelled {
    pub frame: usize,
    pub members: IslandMembers,
    /// The island's state the oracle stepped from (the provided one, or the oracle's own).
    pub start: IslandState,
    /// The oracle's frame-end island state.
    pub end: IslandState,
    /// The call's inputs (no targets) from the frame-start world as imposed.
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

/// Relabel `jobs` (frame, optional island state) of the event in `dir`. A job without a
/// state takes the routing's island at that frame and the oracle's own state of it.
pub fn relabel_event(dir: &Path, jobs: &[(usize, Option<IslandState>)], with_calls: bool) -> Result<Vec<Relabelled>, String> {
    let scene = event_scene(dir)?;
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&k| jobs[k].0);
    let mut out: Vec<Option<Relabelled>> = (0..jobs.len()).map(|_| None).collect();
    let mut w = World::new(&scene);
    let mut at = 0usize;
    for k in order {
        let (frame, st) = &jobs[k];
        while at < *frame {
            w.solver.reset_peak_utilizations();
            w.step_frame();
            at += 1;
        }
        let mut x = w.clone();
        x.solver.reset_peak_utilizations();
        let members = match st {
            Some(s) => s.members(),
            None => x.island_members(),
        };
        if let Some(s) = st {
            x = World::merge_island(&x, s, &members);
        }
        let start = x.extract_island(&members);
        let end = oracle_island_step(&x, &members)?;
        let call = if with_calls && !members.chunks.is_empty() {
            let mut ck: Vec<i64> = members.chunks.iter().map(|&(s, c)| stress_ref::world::stable_key(s, c)).collect();
            ck.sort_unstable();
            let mut ik: Vec<i64> = members.impactors.iter().map(|&i| stress_ref::keys::impactor_key(i)).collect();
            ik.sort_unstable();
            Some(live_call(&mut x.clone(), &scene.name, &serde_json::json!({}), &ck, &ik, *frame, true)?)
        } else {
            None
        };
        out[k] = Some(Relabelled { frame: *frame, members, start, end, call });
    }
    Ok(out.into_iter().map(|r| r.expect("every job relabelled")).collect())
}
