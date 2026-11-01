//! Directed debris (`World::set_guidance`): authored zones that free-flying
//! guided bodies are nudged toward, in the world's coordinates and units.
//! Needs `RbWorldDef::enable_guidance`.

use super::*;
use rubble::ZoneMotion;

const _: () = assert!(rubble::GUIDE_PATH_KEYS == 8);

/// Tuning (`rbDefaultGuidanceSettings`). Speeds and jerk in the world's
/// length unit; times in seconds; angles in radians.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbGuidanceSettings {
    pub enabled: bool,
    /// Launch kick budget, a fraction of the launch speed.
    pub launch_kick: f32,
    /// Largest launch direction change (radians).
    pub launch_cone: f32,
    /// Seconds after a body becomes free in which the kick is allowed.
    pub launch_window: f32,
    pub kick_steps: u32,
    /// Largest held acceleration, a fraction of |gravity|.
    pub accel_max_g: f32,
    /// Length/s³: re-plans toward moving anchors change the acceleration at most this fast.
    pub max_jerk: f32,
    pub min_flight: f32,
    pub max_flight: f32,
    /// Any point on the ground plane flight times are measured to.
    pub ground_point: RbVec3,
    pub settle_steps: u32,
    pub replan_interval: f32,
    pub participation: f32,
    pub min_speed: f32,
    /// A velocity change this far off the ballistic update ends free flight.
    pub impact_dv: f32,
    pub seed: u32,
    pub internal_value: u32,
}

/// `RbGuidanceZone::shape` (constructed by C callers).
#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbZoneShape {
    RB_ZONE_POINT = 0,
    /// `size.x` = radius.
    RB_ZONE_SPHERE = 1,
    /// `size` = half extents (zone frame).
    RB_ZONE_BOX = 2,
    /// `size` = (inner radius, outer radius, height up).
    RB_ZONE_RING = 3,
    /// As a ring, between `angle_from` and `angle_to`.
    RB_ZONE_ARC = 4,
}

/// A known position of a zone at a world time (`RbGuidanceZone::path`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbZonePathKey {
    pub position: RbVec3,
    /// World time (s).
    pub time: f32,
}

/// One zone (`rbDefaultGuidanceZone`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbGuidanceZone {
    /// 0: a fixed zone at `position`; otherwise the zone follows this body.
    pub anchor_body: RbBodyId,
    pub position: RbVec3,
    /// Turn the zone with the anchor body's heading.
    pub anchor_relative: bool,
    /// Rotation about up (radians, the host's sense).
    pub yaw: f32,
    /// From the anchor, in the zone frame.
    pub offset: RbVec3,
    pub shape: RbZoneShape,
    pub size: RbVec3,
    /// Arc angles about up from the zone frame's forward (radians, the host's
    /// sense; under `RB_COORDINATES_UNREAL` forward is +X and positive turns
    /// toward +Y).
    pub angle_from: f32,
    pub angle_to: f32,
    pub weight: f32,
    /// 0: unlimited.
    pub quota: u32,
    /// World time window (s); `active_from > active_to`: always open.
    pub active_from: f32,
    pub active_to: f32,
    pub tags: u32,
    /// Known acceleration of the zone (length/s²), on top of its anchor's
    /// velocity. Ignored when it has a path.
    pub accel: RbVec3,
    /// Keys in `path` (0: none; at most 8).
    pub path_count: u32,
    /// Where the zone will be (times increasing), joined by straight segments
    /// and extended past both ends: a route or patrol, so debris planned now
    /// meets it after a turn. An anchored zone moves from its body as the
    /// path moves from where it is now.
    pub path: [RbZonePathKey; 8],
    pub internal_value: u32,
}

#[no_mangle]
pub extern "C" fn rbDefaultGuidanceSettings() -> RbGuidanceSettings {
    let d = GuidanceSettings::default();
    RbGuidanceSettings {
        enabled: d.enabled,
        launch_kick: d.launch_kick,
        launch_cone: d.launch_cone,
        launch_window: d.launch_window,
        kick_steps: d.kick_steps,
        accel_max_g: d.accel_max_g,
        max_jerk: d.max_jerk,
        min_flight: d.min_flight,
        max_flight: d.max_flight,
        ground_point: RbVec3::default(),
        settle_steps: d.settle_steps,
        replan_interval: d.replan_interval,
        participation: d.participation,
        min_speed: d.min_speed,
        impact_dv: d.impact_dv,
        seed: d.seed,
        internal_value: COOKIE,
    }
}

#[no_mangle]
pub extern "C" fn rbDefaultGuidanceZone() -> RbGuidanceZone {
    RbGuidanceZone {
        anchor_body: 0,
        position: RbVec3::default(),
        anchor_relative: false,
        yaw: 0.0,
        offset: RbVec3::default(),
        shape: RbZoneShape::RB_ZONE_POINT,
        size: RbVec3::default(),
        angle_from: 0.0,
        angle_to: 0.0,
        weight: 1.0,
        quota: 0,
        active_from: 1.0,
        active_to: 0.0,
        tags: 1,
        accel: RbVec3::default(),
        path_count: 0,
        path: [RbZonePathKey::default(); 8],
        internal_value: COOKIE,
    }
}

/// Enable, disable or retune directed debris.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetGuidance(
    world: *mut RbWorld,
    settings: *const RbGuidanceSettings,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let s = read(settings, "null guidance settings")?;
        check_cookie(s.internal_value, "settings not from rbDefaultGuidanceSettings")?;
        let c = w.coords;
        w.world.set_guidance(&GuidanceSettings {
            enabled: s.enabled,
            launch_kick: s.launch_kick,
            launch_cone: s.launch_cone,
            launch_window: s.launch_window,
            kick_steps: s.kick_steps,
            accel_max_g: s.accel_max_g,
            max_jerk: c.length_in(s.max_jerk),
            min_flight: s.min_flight,
            max_flight: s.max_flight,
            ground_y: c.position_in(s.ground_point.a())[1],
            settle_steps: s.settle_steps,
            replan_interval: s.replan_interval,
            participation: s.participation,
            min_speed: c.length_in(s.min_speed),
            impact_dv: c.length_in(s.impact_dv),
            seed: s.seed,
        })
    })
}

/// An angle about up from the host's forward → Rubble's (from +Z toward +X).
/// Unreal forward (+X) is Rubble -Z and +Y is Rubble +X: θ = π - ψ.
fn arc_angle_in(c: &Coords, a: f32) -> f32 {
    if c.unreal() {
        std::f32::consts::PI - a
    } else {
        a
    }
}

/// Replace the zone list (`count` zones; at most the world's zone capacity).
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SetGuidanceZones(
    world: *mut RbWorld,
    zones: *const RbGuidanceZone,
    count: u32,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let raw: &[RbGuidanceZone] = if count == 0 {
            &[]
        } else if zones.is_null() {
            return Err(invalid("null zones"));
        } else {
            std::slice::from_raw_parts(zones, count as usize)
        };
        let c = w.coords;
        let mut out = Vec::with_capacity(raw.len());
        for z in raw {
            check_cookie(z.internal_value, "zone not from rbDefaultGuidanceZone")?;
            let size = z.size;
            let shape = match z.shape {
                RbZoneShape::RB_ZONE_POINT => ZoneShape::Point,
                RbZoneShape::RB_ZONE_SPHERE => ZoneShape::Sphere {
                    radius: c.length_in(size.x),
                },
                RbZoneShape::RB_ZONE_BOX => ZoneShape::Box {
                    half_extents: c.extents_in(size.a()),
                },
                RbZoneShape::RB_ZONE_RING => ZoneShape::Ring {
                    inner: c.length_in(size.x),
                    outer: c.length_in(size.y),
                    height: c.length_in(size.z),
                },
                RbZoneShape::RB_ZONE_ARC => ZoneShape::Arc {
                    inner: c.length_in(size.x),
                    outer: c.length_in(size.y),
                    height: c.length_in(size.z),
                    from: arc_angle_in(&c, z.angle_from),
                    to: arc_angle_in(&c, z.angle_to),
                },
            };
            let motion = if z.path_count > 0 {
                let n = (z.path_count as usize).min(9);
                if n > 8 {
                    return Err(invalid("a zone path takes at most 8 keys"));
                }
                ZoneMotion::Path {
                    keys: z.path[..n]
                        .iter()
                        .map(|k| (c.position_in(k.position.a()), k.time))
                        .collect(),
                }
            } else if z.accel != RbVec3::default() {
                ZoneMotion::Accelerating {
                    accel: c.linear_in(z.accel.a()),
                }
            } else {
                ZoneMotion::Velocity
            };
            out.push(GuidanceZone {
                anchor: if z.anchor_body == 0 {
                    ZoneAnchor::World(c.position_in(z.position.a()))
                } else {
                    ZoneAnchor::Body(BodyId::from_bits(z.anchor_body))
                },
                anchor_relative: z.anchor_relative,
                yaw: c.joint_angle_in(z.yaw),
                offset: c.position_in(z.offset.a()),
                shape,
                weight: z.weight,
                quota: (z.quota > 0).then_some(z.quota),
                active: (z.active_from <= z.active_to).then_some((z.active_from, z.active_to)),
                tags: z.tags,
                motion,
            });
        }
        w.world.set_guidance_zones(&out)
    })
}

/// Zero every zone's claim counter.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_ResetGuidanceClaims(world: *mut RbWorld) -> RbResult {
    guard_args(|| world_mut(world)?.world.reset_guidance_claims())
}

/// Guide a body toward zones whose tags intersect `tag_mask` (0: stop).
#[no_mangle]
pub unsafe extern "C" fn rbBody_SetGuided(
    world: *mut RbWorld,
    body: RbBodyId,
    tag_mask: u32,
) -> RbResult {
    body_command(world, body, |w, id| w.world.body_set_guided_tags(id, tag_mask))
}

/// Guide every chunk of a destructible (`tag_mask` 0: stop). Set it before a
/// strike to choose the zones that strike's debris may use.
#[no_mangle]
pub unsafe extern "C" fn rbDestructible_SetGuided(
    world: *mut RbWorld,
    id: RbDestructibleId,
    tag_mask: u32,
) -> RbResult {
    guard_args(|| {
        world_mut(world)?
            .world
            .destructible_set_guided(DestructibleId::from_bits(id), tag_mask)
    })
}
