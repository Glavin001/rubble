//! Exact restart snapshots of a standalone World (learned-impact P1: the dataset keeps them
//! so any frame can be resumed without re-simulating from the start).
//!
//! A snapshot is every mutable field of the world and its solver, caches included,
//! serialized as MessagePack with named fields (self-describing, so tagged enums and fields
//! serde skips round-trip; every f64 bit kept). The static parts (the scene and the
//! structures built from it) are not stored: `restore_state` rebuilds them with `World::new` from
//! the same scene and takes everything else from the snapshot, so the restored world steps
//! bit-identically to the original (tested). Coupled worlds are refused.

use super::*;

/// The `scene` a deserialized world holds until `restore_state` sets the real one.
pub fn snapshot_placeholder_scene() -> Scene {
    crate::builders::new_scene("snapshot", "placeholder until World::restore_state sets the scene")
}

impl World {
    /// This world's state as exact bytes (bincode). Errors for a coupled world.
    pub fn save_state(&self) -> Result<Vec<u8>, String> {
        if self.coupling.is_some() {
            return Err("a coupled world (its engine's state) cannot be snapshot".into());
        }
        rmp_serde::to_vec_named(self).map_err(|e| format!("save_state: {e}"))
    }

    /// The world `snapshot` saved, for the same `scene` (the one it was built from).
    pub fn restore_state(scene: &Scene, bytes: &[u8]) -> Result<World, String> {
        let fresh = World::new(scene);
        let mut w: World = rmp_serde::from_slice(bytes).map_err(|e| format!("restore_state: {e}"))?;
        if w.solver.chunks.len() != fresh.solver.structures.len() {
            return Err("restore: the snapshot is of another scene".into());
        }
        w.scene = scene.clone();
        w.solver.structures = fresh.solver.structures.clone();
        Ok(w)
    }
}
