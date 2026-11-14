//! Vehicle sleep for explicit soft-body vehicles (`engine/docs/explicit-vehicles-in-world.md`
//! §10): an explicit, engine-wide setting (user decision 2026-11-06). A sleeping vehicle is
//! not stepped at all: its nodes, beams and every other state stay as they were frozen,
//! and its step counter does not advance, so on waking it resumes exactly there.
//!
//! **The rule**, evaluated at the end of every frame (the steps one dispatch runs) for
//! each awake vehicle, in the same arithmetic on the CPU (here) and on the GPU
//! (`slang/kernels/explicit_beams_sleep.slang`):
//!
//! 1. *moving*: at some step of the frame, some node had a velocity component larger than
//!    [`VehicleSleep::speed`] (`|v|∞ > speed`, compared exactly; a NaN is moving);
//! 2. otherwise *displaced*: the vehicle has no anchor (its node positions when its still
//!    stretch began), or some node is farther than [`VehicleSleep::reach`] from its anchor
//!    along an axis (`|x − anchor|∞ > reach`, one strict single-precision subtraction);
//! 3. moving: the still count is 0 and the anchor dropped; displaced: the anchor is the
//!    nodes' positions now and the still count 0; else the still count grows by the
//!    frame's steps;
//! 4. a solo vehicle falls asleep when its still count reaches [`VehicleSleep::dwell`]
//!    steps. The vehicles of a contact group (vehicle contact) fall asleep together, when
//!    every member's count has reached it.
//!
//! **Waking** (the whole vehicle; still count 0, anchor dropped): a contact group with an
//! awake member wakes its sleeping members (another vehicle's proximity: groups form from
//! overlapping frame bounds); the World's coupling plan finds a moving rigid body near it
//! (an awake dynamic body, or a body with velocity, among its proxy candidates); and any
//! host write to it: drive input changed, velocities written, the grab, a scripted force
//! ([`SleepState::wake`]).
//!
//! Off, nothing of this runs and nothing is allocated: results are bit for bit those of a
//! build without it.

use glam::Vec3;

/// The sleep setting (engine-wide: one per fleet or World).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleSleep {
    /// m/s: a node faster than this along any axis at any step keeps its vehicle awake.
    pub speed: f32,
    /// m: a node farther than this from its anchor along any axis keeps it awake.
    pub reach: f32,
    /// Steps (0.5 ms each) a vehicle must stay still before it sleeps.
    pub dwell: u32,
}

impl Default for VehicleSleep {
    /// From the solver's resting floor (`examples/sleep_probe.rs`, the repo sedan parked on
    /// its beam tyres: a tyre node chatters against the ground for good at 0.112 m/s with
    /// a 57 µm stroke, the RoR contact rule's two-step limit cycle, while every node's
    /// position stays within that stroke):
    /// * `speed` 0.25 m/s: 2.2 times that floor (kinetic energy 0.031 J/kg per node); a
    ///   vibration the frame-end anchor test could alias is caught by it at any step;
    /// * `reach` 0.1 mm: above the chatter's stroke; over the dwell it bounds a still
    ///   vehicle's mean speed by 0.2 mm/s, and it is the largest difference between a
    ///   frozen vehicle and one that never slept, plus the chatter;
    /// * `dwell` 1000 steps (0.5 s): Box3D's time to sleep (`b3_timeToSleep`), longer than
    ///   half a car body's bounce period (1 to 1.5 Hz), so the anchor test sees a bounce.
    fn default() -> Self {
        Self {
            speed: 0.25,
            reach: 1.0e-4,
            dwell: 1000,
        }
    }
}

/// A vehicle's sleep state (CPU; the GPU keeps the same words per vehicle).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SleepState {
    /// Steps it has stayed still (since its anchor).
    pub still: u32,
    pub asleep: bool,
    /// Node positions (relative to the vehicle origin) when the still stretch began;
    /// empty: no anchor.
    pub anchor: Vec<Vec3>,
    /// While a frame runs (set by the fleet): the speed bound to track, and whether a
    /// node exceeded it after some step's node pass.
    pub track: Option<f32>,
    pub moving: bool,
}

/// GPU record flag bits (`explicit_beams_sleep.slang`): asleep, anchor valid.
pub const SLEEP_ASLEEP: u32 = 1;
pub const SLEEP_ANCHORED: u32 = 2;

impl SleepState {
    /// Wake (or keep awake) and restart the still count.
    pub fn wake(&mut self) {
        self.still = 0;
        self.asleep = false;
        self.anchor.clear();
    }

    /// The GPU record's words: (still, flags, 0, 0).
    pub fn words(&self) -> [u32; 4] {
        let flags = if self.asleep { SLEEP_ASLEEP } else { 0 }
            | if self.anchor.is_empty() {
                0
            } else {
                SLEEP_ANCHORED
            };
        [self.still, flags, 0, 0]
    }
}

/// Whether a node's velocity is above the speed bound (NaN counts as moving).
pub fn moving(vel: Vec3, speed: f32) -> bool {
    !(vel.x.abs() <= speed && vel.y.abs() <= speed && vel.z.abs() <= speed)
}

/// Whether a node is farther than `reach` from its anchor along an axis.
pub fn displaced(pos: Vec3, anchor: Vec3, reach: f32) -> bool {
    let d = pos - anchor;
    !(d.x.abs() <= reach && d.y.abs() <= reach && d.z.abs() <= reach)
}

/// The frame-end rule (steps 1 to 3) for one vehicle whose frame of `steps` steps had
/// `was_moving` at some step: updates the still count and the anchor; returns whether
/// the count reached the dwell (step 4 decides on it).
pub fn frame_end(
    s: &mut SleepState,
    positions: &[Vec3],
    was_moving: bool,
    steps: u32,
    cfg: &VehicleSleep,
) -> bool {
    if was_moving {
        s.still = 0;
        s.anchor.clear();
    } else if s.anchor.len() != positions.len()
        || positions
            .iter()
            .zip(&s.anchor)
            .any(|(&p, &a)| displaced(p, a, cfg.reach))
    {
        s.anchor.clear();
        s.anchor.extend_from_slice(positions);
        s.still = 0;
    } else {
        s.still = s.still.saturating_add(steps).min(u32::MAX >> 1);
    }
    s.still >= cfg.dwell
}
