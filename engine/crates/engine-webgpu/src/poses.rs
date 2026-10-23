//! Render poses written by `pose_pack.slang` ([`crate::SimOptions::pose_output`]).

use crate::gpu::{MOVED_BINDING, POSES_BINDING, POSE_HEADER_BINDING};
use crate::WgpuSim;
use bytemuck::{Pod, Zeroable};

/// One body's pose (32 B; `pose_pack.slang` `PoseRecord`). In the pose buffer
/// `word` is the state last published; in the moved list it is the body index
/// and what changed ([`PoseRecord::body`], [`PoseRecord::slept`], ...).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PoseRecord {
    pub position: [f32; 3],
    pub word: u32,
    /// Quaternion (x, y, z, w).
    pub rotation: [f32; 4],
}

const MOVED_BODY_MASK: u32 = 0x0FFF_FFFF;
const MOVED_SLEPT: u32 = 1 << 28;
const MOVED_WOKE: u32 = 1 << 29;
const MOVED_FIRST: u32 = 1 << 30;

impl PoseRecord {
    /// The body slot of a moved-list record.
    pub fn body(&self) -> u32 {
        self.word & MOVED_BODY_MASK
    }

    /// The body fell asleep this step (its pose is final until it wakes).
    pub fn slept(&self) -> bool {
        self.word & MOVED_SLEPT != 0
    }

    /// The body woke this step.
    pub fn woke(&self) -> bool {
        self.word & MOVED_WOKE != 0
    }

    /// First report for this slot (new body, or a reused slot): there is no
    /// previous pose to interpolate from.
    pub fn first(&self) -> bool {
        self.word & MOVED_FIRST != 0
    }
}

impl WgpuSim {
    /// Whether `pose_pack` runs after each step.
    pub fn pose_output(&self) -> bool {
        self.pose_output_enabled()
    }

    /// Every body's last published pose ([`PoseRecord`] × `max_bodies`), for GPU
    /// renderers that read poses in place.
    pub fn pose_buffer(&self) -> &wgpu::Buffer {
        self.buffer(POSES_BINDING)
    }

    /// `[moved count, bodies scanned, 0, 0]` of the last step that ran.
    pub fn moved_header_buffer(&self) -> &wgpu::Buffer {
        self.buffer(POSE_HEADER_BINDING)
    }

    /// The bodies whose pose or sleep state changed in the last step that ran
    /// (the header's count of [`PoseRecord`]s, in nondeterministic order).
    pub fn moved_buffer(&self) -> &wgpu::Buffer {
        self.buffer(MOVED_BINDING)
    }

    /// The moved list of the last step (blocking; sorted by body). Empty when the
    /// last step was skipped because nothing was awake.
    pub fn read_moved_poses(&self) -> Vec<PoseRecord> {
        assert!(
            self.pose_output_enabled(),
            "pose output is off (SimOptions::pose_output)"
        );
        if !self.last_step_ran() {
            return Vec::new();
        }
        let header: Vec<u32> = self.read_binding(POSE_HEADER_BINDING, 0, 4);
        let mut moved: Vec<PoseRecord> = self.read_binding(MOVED_BINDING, 0, header[0] as usize);
        moved.sort_by_key(PoseRecord::body);
        moved
    }

    /// Forget body `id`'s published pose: its next report carries
    /// [`PoseRecord::first`] (a slot that now holds a different body).
    pub fn forget_pose(&mut self, id: u32) {
        if self.pose_output_enabled() {
            self.write_binding(POSES_BINDING, u64::from(id) * 32, &[PoseRecord::zeroed()]);
        }
    }
}
