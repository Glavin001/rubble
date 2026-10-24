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

/// Why a body's guidance ended (`GuideBody::flags` bits 8..11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GuideEnd {
    /// Still armed or steering (or never guided).
    None,
    /// Pushed by a contact while steering.
    Pushed,
    /// Reached its planned hit time without a contact.
    Arrived,
    /// Landed (pushed near the ground, or left slow) before it was planned.
    Landed,
    /// Its flight was shorter than `min_flight`.
    Short,
    /// Slower than `min_speed` when it became free.
    Slow,
    /// Left out by `participation`.
    LeftOut,
    /// No zone was within its reach.
    NoZone,
}

impl GuideEnd {
    pub const ALL: [GuideEnd; 8] = [
        Self::None,
        Self::Pushed,
        Self::Arrived,
        Self::Landed,
        Self::Short,
        Self::Slow,
        Self::LeftOut,
        Self::NoZone,
    ];
    pub fn of(record: &GuideBodyGpu) -> Self {
        Self::ALL
            .get(((record.flags >> 8) & 15) as usize)
            .copied()
            .unwrap_or(Self::None)
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

/// Tuning of directed debris, in physical units (`World::set_guidance`, the
/// fracture lab). [`GuidanceSettings::to_params`] packs it for the GPU.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct GuidanceSettings {
    pub enabled: bool,
    /// Launch kick budget, as a fraction of the launch speed.
    pub launch_kick: f32,
    /// Largest change of launch direction the kick may make (radians).
    pub launch_cone: f32,
    /// Seconds after a body becomes free during which the kick is allowed;
    /// a body planned later gets the held acceleration only.
    pub launch_window: f32,
    /// Steps the kick is spread over.
    pub kick_steps: u32,
    /// Largest held acceleration, as a fraction of |gravity|.
    pub accel_max_g: f32,
    /// m/s³: how fast the held acceleration may change when a re-plan
    /// follows a moving anchor.
    pub max_jerk: f32,
    /// Flights shorter than this (s) are not steered.
    pub min_flight: f32,
    /// Flights are considered up to this long (s).
    pub max_flight: f32,
    /// Height of the ground plane flight times are measured to.
    pub ground_y: f32,
    /// Consecutive contact-free steps before a body counts as flying.
    pub settle_steps: u32,
    /// Seconds between re-plans toward a zone anchored to a body.
    pub replan_interval: f32,
    /// Fraction of eligible bodies that take part (chosen by a seeded hash).
    pub participation: f32,
    /// Bodies slower than this (m/s) when they become free are left alone.
    pub min_speed: f32,
    /// A step whose velocity change differs from gravity's (and the guidance's
    /// own) by more than this (m/s) was a contact push: it ends free flight.
    pub impact_dv: f32,
    /// Seeds every random choice (zone pick, aim point, participation).
    pub seed: u32,
}

impl Default for GuidanceSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            launch_kick: 0.06,
            launch_cone: 3.0f32.to_radians(),
            launch_window: 0.1,
            kick_steps: 3,
            accel_max_g: 0.03,
            max_jerk: 1.0,
            min_flight: 0.6,
            max_flight: 8.0,
            ground_y: 0.0,
            settle_steps: 2,
            replan_interval: 0.25,
            participation: 1.0,
            min_speed: 1.0,
            impact_dv: 0.02,
            seed: 0,
        }
    }
}

fn ensure(ok: bool, msg: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(msg.to_string())
    }
}

impl GuidanceSettings {
    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            self.launch_kick,
            self.launch_cone,
            self.launch_window,
            self.accel_max_g,
            self.max_jerk,
            self.min_flight,
            self.max_flight,
            self.ground_y,
            self.replan_interval,
            self.participation,
            self.min_speed,
            self.impact_dv,
        ];
        ensure(finite.iter().all(|v| v.is_finite()), "guidance settings must be finite")?;
        ensure(self.launch_kick >= 0.0, "launch_kick must be non-negative")?;
        ensure(
            (0.0..std::f32::consts::FRAC_PI_2).contains(&self.launch_cone),
            "launch_cone must be in [0, π/2)",
        )?;
        ensure(self.accel_max_g >= 0.0, "accel_max_g must be non-negative")?;
        ensure(self.max_jerk >= 0.0, "max_jerk must be non-negative")?;
        ensure(
            self.min_flight > 0.0 && self.max_flight >= self.min_flight,
            "flight window must satisfy 0 < min_flight <= max_flight",
        )?;
        ensure(
            (0.0..=1.0).contains(&self.participation),
            "participation must be in [0, 1]",
        )?;
        ensure(self.impact_dv > 0.0, "impact_dv must be positive")?;
        ensure(self.kick_steps > 0, "kick_steps must be at least 1")?;
        ensure(self.settle_steps > 0, "settle_steps must be at least 1")
    }

    /// The tuning half of the per-step parameters; hosts fill time, dt,
    /// gravity and the counts each step.
    pub fn to_params(&self, gravity: f32, dt: f32) -> Result<GuideParamsGpu, String> {
        self.validate()?;
        Ok(GuideParamsGpu {
            settle_frames: self.settle_steps,
            replan_frames: ((self.replan_interval / dt).round() as u32).max(1),
            kick_frac: self.launch_kick.min(self.launch_cone.tan()),
            kick_window: self.launch_window,
            accel_max: self.accel_max_g * gravity.abs(),
            max_jerk: self.max_jerk,
            t_min: self.min_flight,
            t_max: self.max_flight,
            y_ground: self.ground_y,
            participation: self.participation,
            kick_frames: self.kick_steps,
            seed: self.seed,
            min_speed: self.min_speed,
            impact_dv: self.impact_dv,
            ..GuideParamsGpu::default()
        })
    }
}

/// The region of a zone that debris is aimed into. Each body aims at its own
/// seeded point inside it, so a burst spreads over the region.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ZoneShape {
    Point,
    Sphere {
        radius: f32,
    },
    #[serde(rename_all = "camelCase")]
    Box {
        half_extents: [f32; 3],
    },
    /// Flat annulus on the zone's XZ plane, `height` thick upward: debris all
    /// around a subject without hitting it.
    Ring {
        inner: f32,
        outer: f32,
        height: f32,
    },
    /// Part of a ring between two angles (radians from the frame's +Z, toward
    /// +X): "behind the car", "to its left".
    Arc {
        inner: f32,
        outer: f32,
        height: f32,
        from: f32,
        to: f32,
    },
}

/// One zone with its anchor resolved to a body index (hosts map handles).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ZoneDesc {
    /// The body the zone follows, or `None` for a fixed zone at `position`.
    pub anchor_body: Option<u32>,
    pub position: [f32; 3],
    /// Turn the zone with its anchor body's heading (yaw only).
    pub anchor_relative: bool,
    /// Extra rotation about +Y (radians).
    pub yaw: f32,
    /// From the anchor, in the zone frame.
    pub offset: [f32; 3],
    pub shape: ZoneShape,
    /// Relative heat: reachable zones are picked in proportion to weight.
    pub weight: f32,
    /// Most bodies this zone takes (`None`: unlimited).
    pub quota: Option<u32>,
    /// World time window (s) in which the zone is open (`None`: always).
    pub active: Option<(f32, f32)>,
    /// Matched against a body's tag mask.
    pub tags: u32,
}

impl Default for ZoneDesc {
    fn default() -> Self {
        Self {
            anchor_body: None,
            position: [0.0; 3],
            anchor_relative: false,
            yaw: 0.0,
            offset: [0.0; 3],
            shape: ZoneShape::Point,
            weight: 1.0,
            quota: None,
            active: None,
            tags: 1,
        }
    }
}

impl ZoneDesc {
    pub fn to_gpu(&self) -> Result<GuideZoneGpu, String> {
        let (shape, extent, angle_from, angle_to) = match self.shape {
            ZoneShape::Point => (ZONE_POINT, [0.0; 3], 0.0, 0.0),
            ZoneShape::Sphere { radius } => (ZONE_SPHERE, [radius, 0.0, 0.0], 0.0, 0.0),
            ZoneShape::Box { half_extents } => (ZONE_BOX, half_extents, 0.0, 0.0),
            ZoneShape::Ring {
                inner,
                outer,
                height,
            } => (ZONE_RING, [inner, outer, height], 0.0, 0.0),
            ZoneShape::Arc {
                inner,
                outer,
                height,
                from,
                to,
            } => (ZONE_ARC, [inner, outer, height], from, to),
        };
        ensure(
            extent.iter().all(|v| v.is_finite() && *v >= 0.0),
            "zone sizes must be finite and non-negative",
        )?;
        if let ZoneShape::Ring { inner, outer, .. } | ZoneShape::Arc { inner, outer, .. } =
            self.shape
        {
            ensure(inner <= outer, "ring inner radius exceeds outer")?;
        }
        let (t_start, t_end) = self.active.unwrap_or((f32::MIN, f32::MAX));
        let finite = [self.yaw, self.weight, angle_from, angle_to];
        ensure(
            finite
                .iter()
                .chain(self.offset.iter())
                .chain(self.position.iter())
                .all(|v| v.is_finite()),
            "zone values must be finite",
        )?;
        ensure(self.weight >= 0.0, "zone weight must be non-negative")?;
        ensure(t_end >= t_start, "zone active window ends before it starts")?;
        ensure(self.quota != Some(0), "a zone quota of 0 takes nothing: remove the zone")?;
        Ok(GuideZoneGpu {
            center: if self.anchor_body.is_some() {
                [0.0; 3]
            } else {
                self.position
            },
            shape,
            extent,
            anchor: self.anchor_body.unwrap_or(ZONE_NO_ANCHOR),
            offset: self.offset,
            weight: self.weight,
            t_start,
            t_end,
            quota: self.quota.unwrap_or(0),
            tags: self.tags,
            yaw: self.yaw,
            frame: if self.anchor_relative {
                ZONE_FRAME_ANCHOR
            } else {
                ZONE_FRAME_WORLD
            },
            angle_from,
            angle_to,
        })
    }
}
