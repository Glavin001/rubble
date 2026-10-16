//! Path coverage: which code paths of the production solver ran, as a bitmask
//! (`GpuSolver::coverage`), so that a test can never pass without its path.
//!
//! The low 32 bits are the device's (`COV_*` in `shaders/common.slang`, same values):
//! kernels OR them into each island's `done[1]` word (kernels without an island: the
//! sentinel island's), which the host reads after every batch, sync and frame-end
//! dispatch. They are only written, never read by the physics. The high bits are the
//! host's own rare paths. A few device paths are proved by device-written data instead of
//! a word (cheaper in hot kernels): `CRUSH` (the impactor's crush energy grew on the
//! GPU) and `STATICS` (a static solve's result reports Newton iterations).

/// A chunk pair pushed (penalty force).
pub const PAIR_CONTACT: u64 = 1 << 0;
/// An impactor touched a chunk.
pub const IMPACTOR_CONTACT: u64 = 1 << 1;
/// A chunk touched the ground.
pub const GROUND_CONTACT: u64 = 1 << 2;
/// An impactor touched the ground.
pub const IMPACTOR_GROUND: u64 = 1 << 3;
/// An impactor's crush cap acted (its crush energy grew on the GPU).
pub const CRUSH: u64 = 1 << 4;
/// A bond's first damage (crack event).
pub const CRACKED: u64 = 1 << 5;
/// Static fatigue's creak event.
pub const CREAKED: u64 = 1 << 6;
/// A bond disconnected.
pub const DISCONNECTED: u64 = 1 << 7;
/// A wide island (many threadgroups) ran a substep.
pub const WIDE: u64 = 1 << 8;
/// A settled narrow island moved rigidly.
pub const SETTLED: u64 = 1 << 9;
/// A settled wide island checked its loads.
pub const SETTLED_WIDE: u64 = 1 << 10;
/// A settled island woke on the GPU.
pub const WOKE: u64 = 1 << 11;
/// A quiet island (settled, anchored, unloaded) kept count only.
pub const QUIET: u64 = 1 << 12;
/// A static solve ran on the GPU (its result reports Newton iterations).
pub const STATICS: u64 = 1 << 13;
/// Settled bonds' static fatigue ran on the GPU.
pub const SETTLED_FATIGUE: u64 = 1 << 14;
/// A scripted or engine load term was evaluated.
pub const LOAD_TERMS: u64 = 1 << 15;
/// A blast time function was evaluated.
pub const BLAST: u64 = 1 << 16;
/// An engine impact pulse was evaluated.
pub const PULSE: u64 = 1 << 17;
/// Probe items were recorded.
pub const PROBES: u64 = 1 << 18;
/// An island ran in the per-substep contact pipeline.
pub const CONTACT_PIPELINE: u64 = 1 << 19;
/// The travel check stopped a batch (a chunk left its contact budget).
pub const STOP_NOW: u64 = 1 << 20;
/// A disconnection left its island whole, found by the island's own connectivity check
/// on the GPU (no halt, no host round trip).
pub const RESUMED_ON_DEVICE: u64 = 1 << 21;

/// The host split a cluster (`ReferenceSolver::split_cluster` after a GPU halt).
pub const HOST_SPLIT: u64 = 1 << 32;
/// A halt whose disconnections left the cluster whole resumed on the GPU after the host's
/// check (islands too big for the device's check).
pub const RESUMED_HALT: u64 = 1 << 33;
/// The contact plan was rebuilt mid-step (travel check).
pub const REPLAN: u64 = 1 << 34;
/// A scripted structural event (removal) rebuilt the GPU state.
pub const REBUILD_EVENT: u64 = 1 << 35;
/// Adaptive mode: a quiet cluster settled (static equilibrium at the frame end).
pub const SETTLE: u64 = 1 << 36;
/// Quasi-static mode: the same-step cascade ran a second pass.
pub const QS_CASCADE: u64 = 1 << 37;
/// Coupled adaptive mode: settled clusters the engine can wake ran this frame.
pub const COUPLED_MAY_WAKE: u64 = 1 << 38;
/// `tangent_statics`: a static solve the GPU secant left unconverged was solved by the
/// oracle on the mirror (`GpuSolver::equilibrate`).
pub const STATICS_DELEGATED: u64 = 1 << 39;

/// The device's bits (the island words).
pub const DEVICE: u64 = (1 << 22) - 1;

/// Every bit with its name, in bit order.
pub const NAMES: [(u64, &str); 30] = [
    (PAIR_CONTACT, "PAIR_CONTACT"),
    (IMPACTOR_CONTACT, "IMPACTOR_CONTACT"),
    (GROUND_CONTACT, "GROUND_CONTACT"),
    (IMPACTOR_GROUND, "IMPACTOR_GROUND"),
    (CRUSH, "CRUSH"),
    (CRACKED, "CRACKED"),
    (CREAKED, "CREAKED"),
    (DISCONNECTED, "DISCONNECTED"),
    (WIDE, "WIDE"),
    (SETTLED, "SETTLED"),
    (SETTLED_WIDE, "SETTLED_WIDE"),
    (WOKE, "WOKE"),
    (QUIET, "QUIET"),
    (STATICS, "STATICS"),
    (SETTLED_FATIGUE, "SETTLED_FATIGUE"),
    (LOAD_TERMS, "LOAD_TERMS"),
    (BLAST, "BLAST"),
    (PULSE, "PULSE"),
    (PROBES, "PROBES"),
    (CONTACT_PIPELINE, "CONTACT_PIPELINE"),
    (STOP_NOW, "STOP_NOW"),
    (RESUMED_ON_DEVICE, "RESUMED_ON_DEVICE"),
    (HOST_SPLIT, "HOST_SPLIT"),
    (RESUMED_HALT, "RESUMED_HALT"),
    (REPLAN, "REPLAN"),
    (REBUILD_EVENT, "REBUILD_EVENT"),
    (SETTLE, "SETTLE"),
    (QS_CASCADE, "QS_CASCADE"),
    (COUPLED_MAY_WAKE, "COUPLED_MAY_WAKE"),
    (STATICS_DELEGATED, "STATICS_DELEGATED"),
];

/// Every defined bit.
pub fn all() -> u64 {
    NAMES.iter().fold(0, |a, (b, _)| a | b)
}

/// The names of the bits set in `bits`.
pub fn names(bits: u64) -> Vec<&'static str> {
    NAMES.iter().filter(|(b, _)| bits & b != 0).map(|(_, n)| *n).collect()
}

/// A bit by name.
pub fn bit(name: &str) -> Option<u64> {
    NAMES.iter().find(|(_, n)| *n == name).map(|(b, _)| *b)
}

#[cfg(test)]
mod tests {
    #[test]
    fn device_bits_match_the_shader() {
        // shaders/common.slang: `COV_<NAME> = 1u << k`.
        let src = include_str!("../shaders/common.slang");
        let mut device = 0u64;
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("public static const uint COV_") else { continue };
            let (name, value) = rest.split_once('=').expect("COV_ constant");
            let shift: u32 = value.split(';').next().unwrap().trim().trim_start_matches("1u << ").parse().expect("shift");
            let bit = super::bit(name.trim()).unwrap_or_else(|| panic!("COV_{} not in coverage.rs", name.trim()));
            assert_eq!(bit, 1 << shift, "COV_{}", name.trim());
            device |= bit;
        }
        assert_eq!(device, super::DEVICE);
    }
}
