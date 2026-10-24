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
pub use engine_schema::guidance::{
    GuidanceCapacity, GuidanceSettings, GuideEnd, GuideState, ZoneShape,
};
use engine_schema::guidance::{GuideBodyGpu, GuideZoneGpu, ZoneDesc};

/// What a zone is attached to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZoneAnchor {
    /// A fixed world position.
    World([f32; 3]),
    /// A body (an NPC, a car): the zone moves with it and is led by its
    /// velocity to where it will be when the debris arrives.
    Body(BodyId),
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
    /// Why its guidance ended (`GuideEnd::None` while armed or steering).
    pub end: GuideEnd,
}

impl World {
    /// Enable, disable or retune directed debris. The world must have been
    /// created with [`crate::WorldDef::guidance`].
    pub fn set_guidance(&mut self, settings: &GuidanceSettings) -> anyhow::Result<()> {
        let p = self.sim.params();
        let params = settings
            .to_params(p.gravity_y, p.dt)
            .map_err(anyhow::Error::msg)?;
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
        let (position, anchor_body) = match z.anchor {
            ZoneAnchor::World(p) => (p, None),
            ZoneAnchor::Body(id) => {
                self.live_body(id)?;
                ([0.0; 3], Some(id.index()))
            }
        };
        ZoneDesc {
            anchor_body,
            position,
            anchor_relative: z.anchor_relative,
            yaw: z.yaw,
            offset: z.offset,
            shape: z.shape,
            weight: z.weight,
            quota: z.quota,
            active: z.active,
            tags: z.tags,
        }
        .to_gpu()
        .map_err(anyhow::Error::msg)
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
            end: GuideEnd::of(&r),
        })
    }

    /// Bodies each zone has taken so far (blocking GPU read).
    pub fn guidance_claims(&self) -> Vec<u32> {
        self.sim.read_guide_claims()
    }
}
