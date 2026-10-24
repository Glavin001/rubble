//! Characters, casts and projectiles (`rubble::World`'s in-step motion
//! pool). Their state and events arrive in every frame (`RbFrame::characters`,
//! `casts`, `projectiles` and their events).

use super::*;

/// A character handle. 0 is null.
pub type RbCharacterId = u64;
/// A projectile handle. 0 is null.
pub type RbProjectileId = u64;

/// A capsule character controller (`rbDefaultCharacterDef`): an upright
/// capsule that walks, climbs steps, sticks to the floor, jumps and rides
/// moving support, run on the GPU inside the step. Its inner kinematic capsule
/// pushes dynamic bodies lighter than `mass * push_ratio`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbCharacterDef {
    /// Capsule centre.
    pub position: RbVec3,
    /// Capsule radius, and half the distance between its sphere centres.
    pub radius: f32,
    pub half_height: f32,
    /// Gap kept to every surface.
    pub skin: f32,
    /// Steepest walkable ground (degrees).
    pub max_slope_deg: f32,
    /// Highest step climbed while walking, and how far down it sticks to
    /// the floor.
    pub step_up: f32,
    pub step_down: f32,
    pub jump_speed: f32,
    /// How fast horizontal velocity follows the input in the air (1/s).
    pub air_control: f32,
    /// kg.
    pub mass: f32,
    pub push_ratio: f32,
    /// Most overlap removed per step, so a pinned character works free
    /// without being launched.
    pub max_recovery: f32,
    pub gravity_scale: f32,
    /// Shape categories it walks against.
    pub mask_bits: u16,
    /// Put its weight on dynamic bodies it stands on.
    pub push_weight: bool,
    pub user_data: u64,
    pub internal_value: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbGroundState {
    RB_ON_GROUND = 0,
    RB_ON_STEEP_GROUND = 1,
    RB_IN_AIR = 2,
}

/// A character as a step left it (`RbFrame::characters`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbCharacterOutput {
    pub character: RbCharacterId,
    /// Capsule centre.
    pub position: RbVec3,
    pub velocity: RbVec3,
    pub ground: RbGroundState,
    pub ground_normal: RbVec3,
    /// The body it stands on (0: none).
    pub ground_body: RbBodyId,
    pub ground_velocity: RbVec3,
    /// Weight it put on a dynamic support this step.
    pub push: RbVec3,
    pub user_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbCharacterEventType {
    RB_CHARACTER_LANDED = 0,
    RB_CHARACTER_SUPPORT_LOST = 1,
    /// Squeezed between surfaces it could not resolve.
    RB_CHARACTER_CRUSHED = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbCharacterEvent {
    pub character: RbCharacterId,
    pub type_: RbCharacterEventType,
    pub ground_body: RbBodyId,
    pub user_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbCastShape {
    RB_CAST_RAY = 0,
    RB_CAST_SPHERE = 1,
    /// Upright capsule: `radius`, `half_height`.
    RB_CAST_CAPSULE = 2,
}

/// A ray or shape cast answered inside the next step before it moves
/// anything (against the previous step's poses) and exact shapes, closest hit
/// (`rbDefaultCastDef`). The result arrives in that step's frame
/// (`RbFrame::casts`, matched by the id).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbCastDef {
    pub origin: RbVec3,
    /// Need not be normalised.
    pub direction: RbVec3,
    pub max_distance: f32,
    pub shape: RbCastShape,
    pub radius: f32,
    pub half_height: f32,
    /// Shapes whose categories meet these bits are hit.
    pub mask_bits: u16,
    /// Never hit (0: none), such as the caster's own body.
    pub ignore_body: RbBodyId,
    pub internal_value: u32,
}

/// A cast's closest hit (`hit.hit` false: none).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbCastResult {
    pub cast: u64,
    pub hit: RbRayHit,
}

/// A bullet-class projectile: its ballistic arc is cast inside every step
/// (no rigid body). Crossing a shape costs `penetration_cost` energy per unit
/// length of its thickness; with energy left it carries on beyond.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbProjectileDef {
    pub position: RbVec3,
    pub velocity: RbVec3,
    /// 0: a ray; > 0: a sphere.
    pub radius: f32,
    /// kg.
    pub mass: f32,
    /// Energy per unit length (J/m, or kg cm/s² — J per cm ×100 — in Unreal
    /// units: a force).
    pub penetration_cost: f32,
    /// Linear drag (1/s).
    pub drag: f32,
    pub max_hits_per_step: u32,
    pub mask_bits: u16,
    pub ignore_body: RbBodyId,
    /// Hits push dynamic bodies.
    pub push_bodies: bool,
    pub user_data: u64,
    pub internal_value: u32,
}

/// A projectile as a step left it: every one in flight, and each the frame
/// it stops (`RbFrame::projectiles`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbProjectileOutput {
    pub projectile: RbProjectileId,
    pub position: RbVec3,
    pub velocity: RbVec3,
    pub alive: bool,
    /// Kinetic energy left.
    pub energy: f32,
    pub hits: u32,
    pub user_data: u64,
}

/// A projectile went through a shape (`thickness` crossed) or stopped in it
/// (`stopped`, `thickness` 0).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbProjectileEvent {
    pub projectile: RbProjectileId,
    pub shape: RbShapeId,
    pub body: RbBodyId,
    pub stopped: bool,
    pub thickness: f32,
    pub user_data: u64,
}

fn mask_in(bits: u16) -> u32 {
    if bits == 0xFFFF {
        0
    } else {
        u32::from(bits)
    }
}

fn mask_out(mask: u32) -> u16 {
    if mask == 0 {
        0xFFFF
    } else {
        mask as u16
    }
}

fn filter(mask_bits: u16) -> QueryFilter {
    QueryFilter {
        categories: 0xFFFF,
        mask: mask_bits,
    }
}

/// Defaults in `coordinates`' units.
#[no_mangle]
pub extern "C" fn rbDefaultCharacterDef(coordinates: RbCoordinates) -> RbCharacterDef {
    let c = Coords::new(coordinates);
    let d = CharacterDesc::default();
    RbCharacterDef {
        position: RbVec3::default(),
        radius: c.length_out(d.radius),
        half_height: c.length_out(d.half_height),
        skin: c.length_out(d.skin),
        max_slope_deg: d.max_slope_deg,
        step_up: c.length_out(d.step_up),
        step_down: c.length_out(d.step_down),
        jump_speed: c.length_out(d.jump_speed),
        air_control: d.air_control,
        mass: d.mass,
        push_ratio: d.push_ratio,
        max_recovery: c.length_out(d.max_recovery),
        gravity_scale: d.gravity_scale,
        mask_bits: mask_out(d.query_mask),
        push_weight: d.push_weight,
        user_data: 0,
        internal_value: COOKIE,
    }
}

#[no_mangle]
pub unsafe extern "C" fn rbWorld_CreateCharacter(
    world: *mut RbWorld,
    def: *const RbCharacterDef,
    out: *mut RbCharacterId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null character def")?;
        check_cookie(
            d.internal_value,
            "character def not from rbDefaultCharacterDef",
        )?;
        if d.mask_bits == 0 {
            return Err(invalid("character mask_bits 0 walks on nothing"));
        }
        let c = w.coords;
        let def = CharacterDef {
            position: c.position_in(d.position.a()),
            desc: CharacterDesc {
                radius: c.length_in(d.radius),
                half_height: c.length_in(d.half_height),
                skin: c.length_in(d.skin),
                max_slope_deg: d.max_slope_deg,
                step_up: c.length_in(d.step_up),
                step_down: c.length_in(d.step_down),
                jump_speed: c.length_in(d.jump_speed),
                air_control: d.air_control,
                mass: d.mass,
                push_ratio: d.push_ratio,
                max_recovery: c.length_in(d.max_recovery),
                gravity_scale: d.gravity_scale,
                query_mask: mask_in(d.mask_bits),
                push_weight: d.push_weight,
            },
            user_data: d.user_data,
        };
        let id = w.world.create_character(&def)?;
        write(out, id.to_bits())
    })
}

/// Remove a character; its handle is refused from now on.
#[no_mangle]
pub unsafe extern "C" fn rbCharacter_Destroy(
    world: *mut RbWorld,
    character: RbCharacterId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        w.world.destroy_character(CharacterId::from_bits(character))
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbCharacter_IsValid(
    world: *const RbWorld,
    character: RbCharacterId,
) -> bool {
    world.as_ref().is_some_and(|w| {
        w.world
            .character_is_valid(CharacterId::from_bits(character))
    })
}

/// Desired horizontal velocity from the next step on, and whether to jump
/// once (when standing).
#[no_mangle]
pub unsafe extern "C" fn rbCharacter_SetInput(
    world: *mut RbWorld,
    character: RbCharacterId,
    velocity: RbVec3,
    jump: bool,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let v = w.coords.linear_in(velocity.a());
        w.world
            .character_set_input(CharacterId::from_bits(character), v, jump)
    })
}

/// Move a character's capsule centre before the next step.
#[no_mangle]
pub unsafe extern "C" fn rbCharacter_Teleport(
    world: *mut RbWorld,
    character: RbCharacterId,
    position: RbVec3,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let p = w.coords.position_in(position.a());
        w.world
            .character_teleport(CharacterId::from_bits(character), p)
    })
}

/// Its inner kinematic capsule (to ignore in casts).
#[no_mangle]
pub unsafe extern "C" fn rbCharacter_GetBody(
    world: *mut RbWorld,
    character: RbCharacterId,
    out: *mut RbBodyId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let b = w.world.character_body(CharacterId::from_bits(character))?;
        write(out, b.to_bits())
    })
}

/// The character as of the last polled frame; `*out_known` false before its
/// first frame.
#[no_mangle]
pub unsafe extern "C" fn rbCharacter_GetState(
    world: *mut RbWorld,
    character: RbCharacterId,
    out: *mut RbCharacterOutput,
    out_known: *mut bool,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let s = w.world.character_state(CharacterId::from_bits(character))?;
        write(out_known, s.is_some())?;
        if let Some(s) = s {
            write(out, character_out(w.coords, &s))?;
        }
        Ok(())
    })
}

/// Defaults in `coordinates`' units.
#[no_mangle]
pub extern "C" fn rbDefaultCastDef(coordinates: RbCoordinates) -> RbCastDef {
    let c = Coords::new(coordinates);
    let d = CastDef::default();
    RbCastDef {
        origin: RbVec3::default(),
        direction: RbVec3::of(c.direction_out(d.direction)),
        max_distance: c.length_out(d.max_distance),
        shape: RbCastShape::RB_CAST_RAY,
        radius: 0.0,
        half_height: 0.0,
        mask_bits: d.filter.mask,
        ignore_body: 0,
        internal_value: COOKIE,
    }
}

/// Queue a cast for the next step (closest hit). Up to 256 per step.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_Cast(
    world: *mut RbWorld,
    def: *const RbCastDef,
    out_cast: *mut u64,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null cast def")?;
        check_cookie(d.internal_value, "cast def not from rbDefaultCastDef")?;
        let c = w.coords;
        let (radius, half_height) = (c.length_in(d.radius), c.length_in(d.half_height));
        let shape = match d.shape {
            RbCastShape::RB_CAST_RAY => CastShape::Ray,
            RbCastShape::RB_CAST_SPHERE => CastShape::Sphere { radius },
            RbCastShape::RB_CAST_CAPSULE => CastShape::Capsule {
                radius,
                half_height,
            },
        };
        let id = w.world.cast(&CastDef {
            origin: c.position_in(d.origin.a()),
            direction: c.direction_in(d.direction.a()),
            max_distance: c.length_in(d.max_distance),
            shape,
            filter: filter(d.mask_bits),
            ignore_body: BodyId::from_bits(d.ignore_body),
        })?;
        write(out_cast, id.0)
    })
}

/// Defaults in `coordinates`' units.
#[no_mangle]
pub extern "C" fn rbDefaultProjectileDef(coordinates: RbCoordinates) -> RbProjectileDef {
    let c = Coords::new(coordinates);
    let d = ProjectileDef::default();
    RbProjectileDef {
        position: RbVec3::default(),
        velocity: RbVec3::default(),
        radius: c.length_out(d.radius),
        mass: d.mass,
        penetration_cost: c.force_out(d.penetration_cost),
        drag: d.drag,
        max_hits_per_step: d.max_hits_per_step,
        mask_bits: d.filter.mask,
        ignore_body: 0,
        push_bodies: d.push_bodies,
        user_data: 0,
        internal_value: COOKIE,
    }
}

/// Fire a projectile from the next step on (slots of stopped ones are
/// reused). Up to 256 in flight.
#[no_mangle]
pub unsafe extern "C" fn rbWorld_SpawnProjectile(
    world: *mut RbWorld,
    def: *const RbProjectileDef,
    out: *mut RbProjectileId,
) -> RbResult {
    guard_args(|| {
        let w = world_mut(world)?;
        let d = read(def, "null projectile def")?;
        check_cookie(
            d.internal_value,
            "projectile def not from rbDefaultProjectileDef",
        )?;
        let c = w.coords;
        let id = w.world.spawn_projectile(&ProjectileDef {
            position: c.position_in(d.position.a()),
            velocity: c.linear_in(d.velocity.a()),
            radius: c.length_in(d.radius),
            mass: d.mass,
            penetration_cost: c.force_in(d.penetration_cost),
            drag: d.drag,
            max_hits_per_step: d.max_hits_per_step,
            filter: filter(d.mask_bits),
            ignore_body: BodyId::from_bits(d.ignore_body),
            push_bodies: d.push_bodies,
            user_data: d.user_data,
        })?;
        write(out, id.to_bits())
    })
}

fn character_out(c: Coords, s: &CharacterOutput) -> RbCharacterOutput {
    RbCharacterOutput {
        character: s.character.to_bits(),
        position: RbVec3::of(c.position_out(s.position)),
        velocity: RbVec3::of(c.linear_out(s.velocity)),
        ground: match s.ground {
            GroundState::OnGround => RbGroundState::RB_ON_GROUND,
            GroundState::OnSteepGround => RbGroundState::RB_ON_STEEP_GROUND,
            GroundState::InAir => RbGroundState::RB_IN_AIR,
        },
        ground_normal: RbVec3::of(c.direction_out(s.ground_normal)),
        ground_body: s.ground_body.to_bits(),
        ground_velocity: RbVec3::of(c.linear_out(s.ground_velocity)),
        push: RbVec3::of(c.linear_out(s.push)),
        user_data: s.user_data,
    }
}

/// The motion part of a published frame.
pub(crate) struct MotionStore {
    pub characters: Vec<RbCharacterOutput>,
    pub character_events: Vec<RbCharacterEvent>,
    pub casts: Vec<RbCastResult>,
    pub projectiles: Vec<RbProjectileOutput>,
    pub projectile_events: Vec<RbProjectileEvent>,
}

impl MotionStore {
    pub fn new(c: Coords, f: &CoalescedFrame) -> Self {
        Self {
            characters: f.characters.iter().map(|s| character_out(c, s)).collect(),
            character_events: f
                .character_events
                .iter()
                .map(|e| RbCharacterEvent {
                    character: e.character.to_bits(),
                    type_: match e.kind {
                        CharacterEventKind::Landed => RbCharacterEventType::RB_CHARACTER_LANDED,
                        CharacterEventKind::SupportLost => {
                            RbCharacterEventType::RB_CHARACTER_SUPPORT_LOST
                        }
                        CharacterEventKind::Crushed => RbCharacterEventType::RB_CHARACTER_CRUSHED,
                    },
                    ground_body: e.ground_body.to_bits(),
                    user_data: e.user_data,
                })
                .collect(),
            casts: f
                .cast_results
                .iter()
                .map(|r| RbCastResult {
                    cast: r.cast.0,
                    hit: r.hit.map_or_else(RbRayHit::default, |h| RbRayHit {
                        hit: true,
                        body: h.body.to_bits(),
                        shape: h.shape.to_bits(),
                        point: RbVec3::of(c.position_out(h.point)),
                        normal: RbVec3::of(c.direction_out(h.normal)),
                        distance: c.length_out(h.distance),
                    }),
                })
                .collect(),
            projectiles: f
                .projectiles
                .iter()
                .map(|p| RbProjectileOutput {
                    projectile: p.projectile.to_bits(),
                    position: RbVec3::of(c.position_out(p.position)),
                    velocity: RbVec3::of(c.linear_out(p.velocity)),
                    alive: p.alive,
                    energy: c.energy_out(p.energy),
                    hits: p.hits,
                    user_data: p.user_data,
                })
                .collect(),
            projectile_events: f
                .projectile_events
                .iter()
                .map(|e| {
                    let (stopped, thickness) = match e.kind {
                        ProjectileEventKind::Passed { thickness } => {
                            (false, c.length_out(thickness))
                        }
                        ProjectileEventKind::Stopped => (true, 0.0),
                    };
                    RbProjectileEvent {
                        projectile: e.projectile.to_bits(),
                        shape: e.shape.to_bits(),
                        body: e.body.to_bits(),
                        stopped,
                        thickness,
                        user_data: e.user_data,
                    }
                })
                .collect(),
        }
    }
}
