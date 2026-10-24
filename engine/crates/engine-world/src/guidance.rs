//! Directed debris: an opt-in gameplay force that nudges guided bodies, while
//! they fly free, toward authored zones (`slang/kernels/guide_step.slang`).
//!
//! A steered body gets a small velocity kick within `launch_window` of
//! becoming free (hidden in the chaos of the break) and then a constant extra
//! acceleration for the rest of its flight. Gravity plus a constant
//! acceleration is still an exact parabola, so the arc looks like plain
//! ballistics under a gravity a few percent off. Guidance ends for good at the
//! first contact that pushes the body: bounces and cascades are pure physics.
//!
//! This is not physics: it adds momentum. It is off unless the world is
//! created with a [`GuidanceCapacity`] and [`World::set_guidance`] enables it.

use crate::error::StaleHandle;
use crate::{BodyId, DestructibleId, World};
use engine_schema::guidance::{
    GuideBodyGpu, GuideParamsGpu, GuideZoneGpu, ZONE_ARC, ZONE_BOX, ZONE_FRAME_ANCHOR,
    ZONE_FRAME_WORLD, ZONE_NO_ANCHOR, ZONE_POINT, ZONE_RING, ZONE_SPHERE,
};
pub use engine_schema::guidance::{GuidanceCapacity, GuideState};

/// Tuning of directed debris ([`World::set_guidance`]).
#[derive(Clone, Debug, PartialEq)]
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

impl GuidanceSettings {
    fn validate(&self) -> anyhow::Result<()> {
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
        anyhow::ensure!(finite.iter().all(|v| v.is_finite()), "guidance settings must be finite");
        anyhow::ensure!(self.launch_kick >= 0.0, "launch_kick must be non-negative");
        anyhow::ensure!(
            (0.0..std::f32::consts::FRAC_PI_2).contains(&self.launch_cone),
            "launch_cone must be in [0, π/2)"
        );
        anyhow::ensure!(self.accel_max_g >= 0.0, "accel_max_g must be non-negative");
        anyhow::ensure!(self.max_jerk >= 0.0, "max_jerk must be non-negative");
        anyhow::ensure!(
            self.min_flight > 0.0 && self.max_flight >= self.min_flight,
            "flight window must satisfy 0 < min_flight <= max_flight"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.participation),
            "participation must be in [0, 1]"
        );
        anyhow::ensure!(self.impact_dv > 0.0, "impact_dv must be positive");
        anyhow::ensure!(self.kick_steps > 0, "kick_steps must be at least 1");
        anyhow::ensure!(self.settle_steps > 0, "settle_steps must be at least 1");
        Ok(())
    }
}

/// What a zone is attached to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoneAnchor {
    /// A fixed world position.
    World([f32; 3]),
    /// A body (an NPC, a car): the zone moves with it and is led by its
    /// velocity to where it will be when the debris arrives.
    Body(BodyId),
}

/// The region of a zone that debris is aimed into. Each body aims at its own
/// seeded point inside it, so a burst spreads over the region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoneShape {
    Point,
    Sphere {
        radius: f32,
    },
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

/// One authored target region ([`World::set_guidance_zones`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GuidanceZone {
    pub anchor: ZoneAnchor,
    /// Turn the zone with its anchor body's heading (yaw only), so arcs and
    /// offsets stay "behind" or "left of" it.
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
    /// Matched against a body's tag mask ([`World::body_set_guided_tags`]).
    pub tags: u32,
}

impl GuidanceZone {
    pub fn new(anchor: ZoneAnchor, shape: ZoneShape) -> Self {
        Self {
            anchor,
            anchor_relative: false,
            yaw: 0.0,
            offset: [0.0; 3],
            shape,
            weight: 1.0,
            quota: None,
            active: None,
            tags: 1,
        }
    }
}

/// A body's guidance, read from the GPU ([`World::body_guidance`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyGuidance {
    pub state: GuideState,
    /// The zone it is steering toward.
    pub zone: Option<u32>,
    /// The held extra acceleration (m/s²).
    pub acceleration: [f32; 3],
    /// World time (s) the plan reaches its aim point.
    pub hit_time: f32,
}

impl World {
    /// Enable, disable or retune directed debris. The world must have been
    /// created with [`crate::WorldDef::guidance`].
    pub fn set_guidance(&mut self, settings: &GuidanceSettings) -> anyhow::Result<()> {
        settings.validate()?;
        let g = self.sim.params().gravity_y.abs();
        let dt = self.sim.params().dt;
        let params = GuideParamsGpu {
            settle_frames: settings.settle_steps,
            replan_frames: ((settings.replan_interval / dt).round() as u32).max(1),
            kick_frac: settings.launch_kick.min(settings.launch_cone.tan()),
            kick_window: settings.launch_window,
            accel_max: settings.accel_max_g * g,
            max_jerk: settings.max_jerk,
            t_min: settings.min_flight,
            t_max: settings.max_flight,
            y_ground: settings.ground_y,
            participation: settings.participation,
            kick_frames: settings.kick_steps,
            seed: settings.seed,
            min_speed: settings.min_speed,
            impact_dv: settings.impact_dv,
            ..GuideParamsGpu::default()
        };
        self.sim.set_guidance(settings.enabled, params)
    }

    /// Replace the zone list (at most [`GuidanceCapacity::max_zones`]). Zones
    /// are indexed by position; claims toward quotas follow the index and
    /// survive a re-send (see [`World::reset_guidance_claims`]).
    pub fn set_guidance_zones(&mut self, zones: &[GuidanceZone]) -> anyhow::Result<()> {
        let mut gpu = Vec::with_capacity(zones.len());
        for (k, z) in zones.iter().enumerate() {
            gpu.push(self.zone_gpu(z).map_err(|e| e.context(format!("guidance zone {k}")))?);
        }
        self.sim.set_guide_zones(&gpu)
    }

    fn zone_gpu(&self, z: &GuidanceZone) -> anyhow::Result<GuideZoneGpu> {
        let (center, anchor) = match z.anchor {
            ZoneAnchor::World(p) => (p, ZONE_NO_ANCHOR),
            ZoneAnchor::Body(id) => {
                self.live_body(id)?;
                ([0.0; 3], id.index())
            }
        };
        let (shape, extent, angle_from, angle_to) = match z.shape {
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
        anyhow::ensure!(
            extent.iter().all(|v| v.is_finite() && *v >= 0.0),
            "zone sizes must be finite and non-negative"
        );
        if let ZoneShape::Ring { inner, outer, .. } | ZoneShape::Arc { inner, outer, .. } = z.shape
        {
            anyhow::ensure!(inner <= outer, "ring inner radius exceeds outer");
        }
        let (t_start, t_end) = z.active.unwrap_or((f32::MIN, f32::MAX));
        let finite = [z.yaw, z.weight, angle_from, angle_to, z.offset[0], z.offset[1], z.offset[2]];
        anyhow::ensure!(
            finite.iter().chain(center.iter()).all(|v| v.is_finite()),
            "zone values must be finite"
        );
        anyhow::ensure!(z.weight >= 0.0, "zone weight must be non-negative");
        anyhow::ensure!(!(t_end < t_start), "zone active window ends before it starts");
        anyhow::ensure!(z.quota != Some(0), "a zone quota of 0 takes nothing: remove the zone");
        Ok(GuideZoneGpu {
            center,
            shape,
            extent,
            anchor,
            offset: z.offset,
            weight: z.weight,
            t_start,
            t_end,
            quota: z.quota.unwrap_or(0),
            tags: z.tags,
            yaw: z.yaw,
            frame: if z.anchor_relative {
                ZONE_FRAME_ANCHOR
            } else {
                ZONE_FRAME_WORLD
            },
            angle_from,
            angle_to,
        })
    }

    /// Zero every zone's claim counter.
    pub fn reset_guidance_claims(&mut self) -> anyhow::Result<()> {
        self.sim.reset_guide_claims()
    }

    /// Guide a body (every zone tag) or stop guiding it. Arming resets its
    /// guidance: it waits for its next free flight.
    pub fn body_set_guided(&mut self, id: BodyId, guided: bool) -> anyhow::Result<()> {
        self.body_set_guided_tags(id, if guided { u32::MAX } else { 0 })
    }

    /// Guide a body toward zones whose tags intersect `tag_mask` (0: off).
    pub fn body_set_guided_tags(&mut self, id: BodyId, tag_mask: u32) -> anyhow::Result<()> {
        self.live_body(id)?;
        let record = if tag_mask == 0 {
            GuideBodyGpu::default()
        } else {
            GuideBodyGpu::armed(id.generation(), tag_mask)
        };
        self.sim.write_guide_bodies(id.index(), &[record])
    }

    /// Guide every chunk of a destructible (`tag_mask` 0: off). Set it before
    /// a strike to choose which zones that strike's debris may use.
    pub fn destructible_set_guided(
        &mut self,
        id: DestructibleId,
        tag_mask: u32,
    ) -> anyhow::Result<()> {
        for body in self.destructible_bodies(id)? {
            self.body_set_guided_tags(body, tag_mask)?;
        }
        Ok(())
    }

    /// A body's guidance (blocking GPU read; for tools and tests).
    pub fn body_guidance(&self, id: BodyId) -> anyhow::Result<BodyGuidance> {
        if !self.body_is_valid(id) {
            return Err(StaleHandle("body", id.to_bits()).into());
        }
        let r = self
            .sim
            .read_guide_bodies(id.index(), 1)
            .first()
            .copied()
            .unwrap_or_default();
        let state = GuideState::from_u32(r.state);
        Ok(BodyGuidance {
            state,
            zone: (state == GuideState::Steering || state == GuideState::Done)
                .then_some(r.zone)
                .filter(|&z| z != u32::MAX),
            acceleration: r.accel,
            hit_time: r.hit_time,
        })
    }

    /// Bodies each zone has taken so far (blocking GPU read).
    pub fn guidance_claims(&self) -> Vec<u32> {
        self.sim.read_guide_claims()
    }
}
