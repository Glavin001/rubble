//! Directed debris ABI (`slang/kernels/guide_step.slang`): the per-step
//! parameters, the authored zones, the per-body guidance record and the plan
//! slots. Bindings follow the render-pose buffers (33-35).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

pub const GUIDE_PARAMS_BINDING: usize = 36;
pub const GUIDE_ZONES_BINDING: usize = 37;
pub const GUIDE_BODIES_BINDING: usize = 38;
pub const GUIDE_SCRATCH_BINDING: usize = 39;
pub const GUIDE_SLOTS_BINDING: usize = 40;

/// The guidance entries (`guide_step.slang`), in step order.
pub const GUIDE_ENTRIES: &[&str] = &[
    "guide_scan",
    "guide_select",
    "guide_rank",
    "guide_plan",
    "guide_commit",
];

/// Bodies per scan block (`GUIDE_BLOCK`); also the most plans per step.
pub const GUIDE_BLOCK: u32 = 256;

/// Scratch words before the per-zone claim counters (`SCRATCH_CLAIMS`).
pub const GUIDE_SCRATCH_CLAIMS: u32 = 16;

/// Fixed at world creation. Uploading more zones than `max_zones` is an
/// error; at most `plan_budget` bodies (up to [`GUIDE_BLOCK`]) are planned per
/// step, so a large burst is planned over several steps instead of in one
/// expensive one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidanceCapacity {
    pub max_zones: u32,
    pub plan_budget: u32,
}

impl Default for GuidanceCapacity {
    fn default() -> Self {
        Self {
            max_zones: 256,
            plan_budget: 256,
        }
    }
}

impl GuidanceCapacity {
    /// Scratch words: header, claims, then (count, first rank) per scan block.
    pub fn scratch_words(&self, max_bodies: u32) -> u64 {
        u64::from(GUIDE_SCRATCH_CLAIMS)
            + u64::from(self.max_zones)
            + 2 * u64::from(max_bodies.div_ceil(GUIDE_BLOCK).max(1))
    }
}

/// Guidance state of one body (`GuideBody::state`).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuideState {
    Off = 0,
    /// Guided, waiting for its first free flight.
    Armed = 1,
    /// Holding a planned kick and acceleration.
    Steering = 2,
    /// Its arc is over (contact, hit time, or nothing reachable).
    Done = 3,
}

impl GuideState {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Armed,
            2 => Self::Steering,
            3 => Self::Done,
            _ => Self::Off,
        }
    }
}

/// `GuideParams` (96 B), written by the host every guided step.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GuideParamsGpu {
    pub time: f32,
    pub dt: f32,
    pub gravity_y: f32,
    pub zone_count: u32,
    pub body_count: u32,
    pub plan_budget: u32,
    pub settle_frames: u32,
    pub replan_frames: u32,
    pub kick_frac: f32,
    pub kick_window: f32,
    pub accel_max: f32,
    pub max_jerk: f32,
    pub t_min: f32,
    pub t_max: f32,
    pub y_ground: f32,
    pub participation: f32,
    pub kick_frames: u32,
    pub seed: u32,
    pub min_speed: f32,
    pub block_count: u32,
    pub max_zones: u32,
    pub joint_count: u32,
    pub impact_dv: f32,
    pub _pad: u32,
}

pub const ZONE_POINT: u32 = 0;
pub const ZONE_SPHERE: u32 = 1;
pub const ZONE_BOX: u32 = 2;
pub const ZONE_RING: u32 = 3;
pub const ZONE_ARC: u32 = 4;
pub const ZONE_NO_ANCHOR: u32 = u32::MAX;
pub const ZONE_FRAME_WORLD: u32 = 0;
pub const ZONE_FRAME_ANCHOR: u32 = 1;

/// `GuideZone` (80 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GuideZoneGpu {
    pub center: [f32; 3],
    pub shape: u32,
    pub extent: [f32; 3],
    pub anchor: u32,
    pub offset: [f32; 3],
    pub weight: f32,
    pub t_start: f32,
    pub t_end: f32,
    pub quota: u32,
    pub tags: u32,
    pub yaw: f32,
    pub frame: u32,
    pub angle_from: f32,
    pub angle_to: f32,
}

/// `GuideBody` (64 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GuideBodyGpu {
    pub state: u32,
    pub zone: u32,
    pub generation: u32,
    pub free_frames: u32,
    pub accel: [f32; 3],
    pub hit_time: f32,
    pub kick: [f32; 3],
    pub kick_left: u32,
    pub rank: u32,
    pub since_plan: u32,
    pub tag_mask: u32,
    pub flags: u32,
}

impl GuideBodyGpu {
    /// A freshly guided body.
    pub fn armed(generation: u32, tag_mask: u32) -> Self {
        Self {
            state: GuideState::Armed as u32,
            zone: u32::MAX,
            generation,
            tag_mask,
            ..Self::default()
        }
    }
}

/// `GuideSlot` (64 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GuideSlotGpu {
    pub body: u32,
    pub zone: u32,
    pub replan: u32,
    pub _pad: u32,
    pub accel: [f32; 3],
    pub hit_time: f32,
    pub kick: [f32; 3],
    pub _pad1: f32,
    pub _pad2: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GuideParamsGpu>() == 96);
const _: () = assert!(std::mem::size_of::<GuideZoneGpu>() == 80);
const _: () = assert!(std::mem::size_of::<GuideBodyGpu>() == 64);
const _: () = assert!(std::mem::size_of::<GuideSlotGpu>() == 64);
