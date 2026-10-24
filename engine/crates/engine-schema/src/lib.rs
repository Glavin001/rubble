//! Shared host/GPU ABI for the AVBD engine.
//!
//! Units: meters, kilograms, seconds, radians.
//! Frame: right-handed Y-up. Quaternions are `(x, y, z, w)`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

pub mod commands;
pub mod mass;
pub mod convex;
pub mod handles;
pub mod joint;
pub mod params;
pub mod scene;
pub mod snapshot;
pub mod vehicle;

pub use commands::*;
pub use handles::*;
pub use joint::*;
pub use params::*;
pub use snapshot::*;

/// ABI probe used by Gate 0 layout round-trips (must match `slang/kernels/abi_probe.slang`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AbiProbe {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub inertia_col0: [f32; 4],
    pub inertia_col1: [f32; 4],
    pub inertia_col2: [f32; 4],
    pub counter: u32,
    pub _pad1: [u32; 3],
}

impl AbiProbe {
    pub fn new(position: Vec3, inertia: Mat3, counter: u32) -> Self {
        let c0 = inertia.col(0);
        let c1 = inertia.col(1);
        let c2 = inertia.col(2);
        Self {
            position: position.to_array(),
            _pad0: 0.0,
            inertia_col0: [c0.x, c0.y, c0.z, 0.0],
            inertia_col1: [c1.x, c1.y, c1.z, 0.0],
            inertia_col2: [c2.x, c2.y, c2.z, 0.0],
            counter,
            _pad1: [0; 3],
        }
    }

    pub fn position(&self) -> Vec3 {
        Vec3::from_array(self.position)
    }

    pub fn inertia(&self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.inertia_col0[0], self.inertia_col0[1], self.inertia_col0[2]),
            Vec3::new(self.inertia_col1[0], self.inertia_col1[1], self.inertia_col1[2]),
            Vec3::new(self.inertia_col2[0], self.inertia_col2[1], self.inertia_col2[2]),
        )
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyFlags {
    None = 0,
    Dynamic = 1,
    Kinematic = 2,
    Sleeping = 4,
    Awake = 8,
}

/// Motion locks in `BodyState::flags` (world axes): bits 8..10 linear x, y,
/// z; 11..13 angular x, y, z.
pub const BODY_LOCK_SHIFT: u32 = 8;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    Sphere = 0,
    Capsule = 1,
    Box = 2,
    ConvexHull = 3,
    Plane = 4,
}

/// Slots in the joint-pair hash (`jointHashSize()` in world_step.slang): a power
/// of two, at least 65536 and twice the joint count. `gEvents` needs three
/// words per slot (joint index, then the body pair).
pub fn joint_hash_size(joints: u32) -> u32 {
    (joints.saturating_mul(2)).max(65536).next_power_of_two()
}

/// Hot body state (112 bytes). `initial_pos` / `initial_rot` are the step-start pose.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BodyState {
    pub position: [f32; 3],
    pub inv_mass: f32,
    pub orientation: [f32; 4],
    pub inv_inertia: [f32; 3],
    pub flags: u32,
    pub linear_vel: [f32; 3],
    pub sleep_timer: f32,
    pub angular_vel: [f32; 3],
    pub generation: u32,
    pub initial_pos: [f32; 3],
    pub initial_pad: f32,
    pub initial_rot: [f32; 4],
}

impl BodyState {
    pub fn static_plane_support() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0, 0.0, 0.0],
            flags: BodyFlags::None as u32,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn dynamic(position: Vec3, orientation: Quat, mass: f32, inertia: Vec3) -> Self {
        let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position: position.to_array(),
            inv_mass: inv_m,
            orientation: orientation.to_array(),
            inv_inertia: [
                if inertia.x > 0.0 { 1.0 / inertia.x } else { 0.0 },
                if inertia.y > 0.0 { 1.0 / inertia.y } else { 0.0 },
                if inertia.z > 0.0 { 1.0 / inertia.z } else { 0.0 },
            ],
            flags: BodyFlags::Dynamic as u32 | BodyFlags::Awake as u32,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: [0.0; 3],
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Moves under the solver or its set velocity (the kernels'
    /// `bodyIsDynamic`): dynamic with mass, or kinematic.
    pub fn is_dynamic(&self) -> bool {
        (self.flags & BodyFlags::Dynamic as u32) != 0 && (self.inv_mass > 0.0 || self.is_kinematic())
    }

    pub fn is_kinematic(&self) -> bool {
        self.flags & BodyFlags::Kinematic as u32 != 0
    }

    /// Moves at all (mass or kinematic flag): host-side broadphase sizing.
    pub fn moves(&self) -> bool {
        self.inv_mass > 0.0 || self.is_kinematic()
    }

    pub fn is_sleeping(&self) -> bool {
        (self.flags & BodyFlags::Sleeping as u32) != 0
    }

    pub fn quat(&self) -> Quat {
        Quat::from_array(self.orientation)
    }
}

/// Collider attached to a body (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Collider {
    pub body: u32,
    pub shape: u32,
    pub material: u32,
    pub generation: u32,
    pub local_pos: [f32; 3],
    pub density: f32,
    pub local_rot: [f32; 4],
    /// Sphere: radius; Capsule: radius, half_length; Box: half extents; Plane: normal.xyz + offset.w
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    pub friction: f32,
    pub restitution: f32,
    pub contact_offset: f32,
    pub _pad: f32,
}

/// Persistent contact row (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Contact {
    pub collider_a: u32,
    pub collider_b: u32,
    pub feature_id: u32,
    pub alive: u32,
    pub normal: [f32; 3],
    pub penetration: f32,
    pub anchor_a: [f32; 3],
    pub c0_n: f32,
    pub anchor_b: [f32; 3],
    pub friction: f32,
    pub tangent1: [f32; 3],
    pub lambda_n: f32,
    pub tangent2: [f32; 3],
    pub lambda_t1: f32,
    pub k_n: f32,
    pub k_t: f32,
    pub lambda_t2: f32,
    pub c0_t1: f32,
    pub c0_t2: f32,
    /// GPU contact metadata: stick flag, generation bits, second tangent penalty.
    pub _pad: [f32; 3],
}

/// Contact pair manifold (32 bytes) — three-avbd Stage 8b.
/// CSR / dual / primal adjacency key one edge per manifold, not per contact point.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Manifold {
    pub collider_a: u32,
    pub collider_b: u32,
    pub contact_start: u32,
    pub contact_count: u32,
    pub normal: [f32; 3],
    pub friction: f32,
}

/// Event ring in the scratch-atomics buffer (binding 21), after the 96 words
/// the hosts and kernels use: word `EVENT_RING_BASE` is a monotonically
/// increasing write cursor, then `EVENT_RING_CAPACITY` records of four words
/// `[type, id, a, b]` at `EVENT_RING_BASE + 4 + 4 * (n % EVENT_RING_CAPACITY)`.
/// (The events buffer belongs to the joint-pair hash, whose size follows the
/// joint count.)
pub const EVENT_RING_BASE: u32 = 96;
pub const EVENT_RING_CAPACITY: u32 = 1024;
/// Contact events (opt-in `contact_events` pass): after the large-static
/// list, a header word counting the records appended this step (the host
/// zeroes it), three spare words, then up to `CONTACT_EVENT_CAP` records of
/// `CONTACT_EVENT_WORDS`: `[colliderA, colliderB, approach speed, 0]`,
/// `[point, 0]`, `[normal, 0]`. Counts past the capacity are reported lost.
pub const CONTACT_EVENT_BASE: u32 = LARGE_STATIC_BASE + LARGE_STATIC_CAP;
pub const CONTACT_EVENT_CAP: u32 = 2048;
pub const CONTACT_EVENT_WORDS: u32 = 12;
/// `Collider.material` bits 12..15 (the index uses 0..11, the filter group
/// 16..31): the shape reports contact events, hit events, or is a sensor.
pub const COLLIDER_CONTACT_EVENTS: u32 = 1 << 12;
pub const COLLIDER_HIT_EVENTS: u32 = 1 << 13;
pub const COLLIDER_SENSOR: u32 = 1 << 14;
/// Total `u32` words of the scratch-atomics buffer (host words, lists, ring,
/// contact events).
pub const SCRATCH_WORDS: u32 = CONTACT_EVENT_BASE + 4 + CONTACT_EVENT_WORDS * CONTACT_EVENT_CAP;
/// Scratch word of the oversized-static count (uniform grid; see `LARGE_STATIC_CAP`).
pub const LARGE_STATIC_COUNT: u32 = 80;
/// First scratch word of the oversized-static collider ids, after the event ring.
pub const LARGE_STATIC_BASE: u32 = EVENT_RING_BASE + 4 + 4 * EVENT_RING_CAPACITY;
/// Event record types (`world_step.slang` EVENT_*).
pub const EVENT_BREAK: u32 = 3;

/// A joint broke this step: its id and the force / torque that exceeded the
/// threshold (read from the event ring).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointBreak {
    pub joint: u32,
    pub force: f32,
    pub torque: f32,
}

/// Decode new `EVENT_BREAK` records from a scratch-buffer ring readback.
/// `words` starts at `EVENT_RING_BASE`; `seen` is the caller's read cursor and is
/// advanced. Records overwritten before they were read are counted in the return.
pub fn decode_joint_breaks(words: &[u32], seen: &mut u32, out: &mut Vec<JointBreak>) -> u32 {
    let cursor = words[0];
    let lost = cursor.wrapping_sub(*seen).saturating_sub(EVENT_RING_CAPACITY);
    let mut n = seen.wrapping_add(lost);
    while n != cursor {
        let r = 4 + 4 * (n % EVENT_RING_CAPACITY) as usize;
        if words[r] == EVENT_BREAK {
            out.push(JointBreak {
                joint: words[r + 1],
                force: f32::from_bits(words[r + 2]),
                torque: f32::from_bits(words[r + 3]),
            });
        }
        n = n.wrapping_add(1);
    }
    *seen = cursor;
    lost
}

/// Event record type of a vehicle mount break (`vehicle_step.slang`
/// `EVENT_MOUNT_BREAK`): `[6, vehicle << 8 | is_wheel << 7 | index, force bits,
/// torque bits]`. (5 is the soft-body beam break.)
pub const EVENT_MOUNT_BREAK: u32 = 6;

/// A vehicle part or wheel whose mount broke: the vehicle, which part (or
/// wheel, when `wheel`) and the mount force / moment that exceeded its limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MountBreak {
    pub vehicle: u32,
    pub index: u32,
    pub wheel: bool,
    pub force: f32,
    pub torque: f32,
}

/// Decode new `EVENT_MOUNT_BREAK` records, as [`decode_joint_breaks`] does
/// (each event kind keeps its own read cursor over the shared ring).
pub fn decode_mount_breaks(words: &[u32], seen: &mut u32, out: &mut Vec<MountBreak>) -> u32 {
    let cursor = words[0];
    let lost = cursor.wrapping_sub(*seen).saturating_sub(EVENT_RING_CAPACITY);
    let mut n = seen.wrapping_add(lost);
    while n != cursor {
        let r = 4 + 4 * (n % EVENT_RING_CAPACITY) as usize;
        if words[r] == EVENT_MOUNT_BREAK {
            let id = words[r + 1];
            out.push(MountBreak {
                vehicle: id >> 8,
                index: id & 0x7F,
                wheel: id & 0x80 != 0,
                force: f32::from_bits(words[r + 2]),
                torque: f32::from_bits(words[r + 3]),
            });
        }
        n = n.wrapping_add(1);
    }
    *seen = cursor;
    lost
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct StepCounters {
    pub body_count: u32,
    pub collider_count: u32,
    pub contact_count: u32,
    pub joint_count: u32,
    pub candidate_count: u32,
    pub awake_count: u32,
    pub color_count: u32,
    pub overflow_flags: u32,
    pub pivot_floor_hits: u32,
    pub sleep_events: u32,
    pub wake_events: u32,
    /// Convex SAT pairs in the small bucket (GPU ABI byte 44).
    pub convex_sat_small: u32,
    pub manifold_count: u32,
    /// Number of bodies scheduled for incremental recoloring (GPU ABI byte 52).
    pub _pad0: u32,
    /// Two-pass narrowphase deferred pairs (GPU ABI byte 56).
    pub narrowphase_deferred: u32,
    /// Persist work-list length (GPU ABI byte 60).
    pub _pad2: u32,
    /// Convex SAT pairs in the large bucket (GPU ABI byte 64).
    pub convex_sat_large: u32,
    /// One past the last Jones-Plassmann round that left a body blocked
    /// (GPU ABI byte 68); hosts size the next step's colouring rounds from it.
    pub color_rounds: u32,
    /// Colouring rounds the host allows this step (GPU ABI byte 72), read by
    /// the single-workgroup tail.
    pub color_round_limit: u32,
    /// Pads the block to 80 bytes (GPU ABI byte 76).
    pub _pad3: u32,
}

impl Default for StepCounters {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CandidatePair {
    pub collider_a: u32,
    pub collider_b: u32,
    pub key: u32,
    pub _pad: u32,
}

pub const OVERFLOW_CONTACTS: u32 = 1;
pub const OVERFLOW_CANDIDATES: u32 = 2;
pub const OVERFLOW_JOINTS: u32 = 4;
pub const OVERFLOW_COLORS: u32 = 8;
pub const CCD_BUDGET_EXHAUSTED: u32 = 16;
pub const CCD_CANDIDATE_MISS: u32 = 32;
pub const OVERFLOW_MANIFOLDS: u32 = 64;
pub const OVERFLOW_BP_CHAIN: u32 = 128;
pub const OVERFLOW_WARM_HASH: u32 = 256;
/// More oversized statics than `LARGE_STATIC_CAP`: some are found only by the
/// endpoint probes.
pub const OVERFLOW_LARGE_STATICS: u32 = 512;
/// Oversized-static list (statics wider than half a grid cell, uniform grid):
/// this many collider ids in the scratch-atomics buffer at `LARGE_STATIC_BASE`.
pub const LARGE_STATIC_CAP: u32 = 1024;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn event_ring_decodes_new_breaks_and_counts_overwritten_ones() {
        let mut words = vec![0u32; (SCRATCH_WORDS - EVENT_RING_BASE) as usize];
        fn put(words: &mut [u32], n: u32, ty: u32, id: u32) {
            let r = 4 + 4 * (n % EVENT_RING_CAPACITY) as usize;
            words[r..r + 4].copy_from_slice(&[ty, id, 2.5f32.to_bits(), 0.5f32.to_bits()]);
            words[0] = n + 1;
        }
        put(&mut words, 0, EVENT_BREAK, 7);
        put(&mut words, 1, 1, 9); // not a break (sleep)
        put(&mut words, 2, EVENT_BREAK, 8);
        let (mut seen, mut out) = (0u32, Vec::new());
        assert_eq!(decode_joint_breaks(&words, &mut seen, &mut out), 0);
        assert_eq!(out, vec![JointBreak { joint: 7, force: 2.5, torque: 0.5 }, JointBreak { joint: 8, force: 2.5, torque: 0.5 }]);
        assert_eq!(seen, 3);
        // Nothing new: nothing returned.
        out.clear();
        decode_joint_breaks(&words, &mut seen, &mut out);
        assert!(out.is_empty());
        // The ring wraps past the reader: the oldest records are reported lost.
        for n in 3..3 + EVENT_RING_CAPACITY + 5 {
            put(&mut words, n, EVENT_BREAK, n);
        }
        let lost = decode_joint_breaks(&words, &mut seen, &mut out);
        assert_eq!(lost, 5);
        assert_eq!(out.len() as u32, EVENT_RING_CAPACITY);
        assert_eq!(out[0].joint, 8);
        assert_eq!(out.last().unwrap().joint, 3 + EVENT_RING_CAPACITY + 4);
    }

    #[test]
    fn abi_sizes() {
        assert_eq!(size_of::<AbiProbe>(), 80);
        assert_eq!(size_of::<BodyState>(), 112);
        assert_eq!(size_of::<Collider>(), 64);
        assert_eq!(size_of::<Contact>(), 128);
        assert_eq!(size_of::<Manifold>(), 32);
        assert_eq!(size_of::<StepCounters>(), 80);
        assert_eq!(size_of::<Joint>(), 288);
        assert_eq!(align_of::<BodyState>(), 4);
    }
}
