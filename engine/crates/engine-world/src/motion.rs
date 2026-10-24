//! Characters, shape casts and projectiles: the step's own kernels
//! (`character_step.slang`, binding 33) move capsule characters, answer
//! queued casts and fly projectiles, and each frame reads their records back
//! with the rest of the step's results. Nothing here waits on the GPU.

use crate::handles::handle;
use crate::{readback, BodyId, Extra, Owner, QueryFilter, ShapeId, World};
use engine_schema::character::{
    CharacterState, ProjectileDesc, ProjectileState, QueryHit, SceneQuery, CHAR_STRIDE, NO_INDEX,
    PROJ_STRIDE, QUERY_STRIDE,
};
use engine_webgpu::{CharacterHandle, MotionEvent};

pub use engine_schema::character::{CharacterDesc, GroundState};

handle!(
    /// A capsule character controller.
    CharacterId
);
handle!(
    /// A projectile in flight (or one that has stopped, until its slot is
    /// reused).
    ProjectileId
);

/// A character (Box3D's mover, run on the GPU inside the step): an upright
/// capsule that walks, climbs steps, sticks to the floor, jumps and rides
/// moving support. Its inner kinematic capsule pushes dynamic bodies lighter
/// than `desc.mass * desc.push_ratio`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterDef {
    /// Capsule centre.
    pub position: [f32; 3],
    pub desc: CharacterDesc,
    pub user_data: u64,
}

impl Default for CharacterDef {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            desc: CharacterDesc::default(),
            user_data: 0,
        }
    }
}

/// A character as a step left it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterOutput {
    pub character: CharacterId,
    /// Capsule centre.
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub ground: GroundState,
    pub ground_normal: [f32; 3],
    /// The body it stands on (`NULL` in the air or on a body the API does
    /// not hand out).
    pub ground_body: BodyId,
    /// Velocity of the support point under it.
    pub ground_velocity: [f32; 3],
    /// Weight it put on a dynamic support this step (N).
    pub push: [f32; 3],
    pub user_data: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterEventKind {
    Landed,
    SupportLost,
    /// Squeezed between surfaces it could not resolve.
    Crushed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterEvent {
    pub character: CharacterId,
    pub kind: CharacterEventKind,
    pub ground_body: BodyId,
    pub user_data: u64,
}

/// What a cast sweeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CastShape {
    Ray,
    Sphere {
        radius: f32,
    },
    /// Upright (Y) capsule: half the distance between its sphere centres.
    Capsule {
        radius: f32,
        half_height: f32,
    },
}

/// A ray or shape cast answered inside the next step, before it moves
/// anything: against the poses it starts from (the previous step's results)
/// and exact shapes (Box3D `b3World_CastRayClosest` / `b3World_CastShape`,
/// closest hit). The result arrives with that step's frame
/// (`Frame::cast_results`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastDef {
    pub origin: [f32; 3],
    /// Need not be normalised.
    pub direction: [f32; 3],
    pub max_distance: f32,
    pub shape: CastShape,
    /// Shapes whose categories meet `filter.mask` are hit.
    pub filter: QueryFilter,
    /// A body the cast never hits (`NULL`: none), such as the caster's own.
    pub ignore_body: BodyId,
}

impl Default for CastDef {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            direction: [0.0, -1.0, 0.0],
            max_distance: 1.0,
            shape: CastShape::Ray,
            filter: QueryFilter::default(),
            ignore_body: BodyId::NULL,
        }
    }
}

/// A queued cast ([`World::cast`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CastId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastHit {
    /// `NULL` for a shape the API does not hand out (a vehicle part or a
    /// character capsule); `body` is then still set when known.
    pub shape: ShapeId,
    pub body: BodyId,
    /// A ray's surface point; a shape cast's centre at impact less its radius
    /// along the normal.
    pub point: [f32; 3],
    pub normal: [f32; 3],
    /// Distance travelled along the normalised direction.
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastResult {
    pub cast: CastId,
    pub hit: Option<CastHit>,
}

/// A bullet-class projectile: no rigid body, its ballistic arc is cast inside
/// every step. Energy decides penetration: crossing a shape costs
/// `penetration_cost` J per metre of its thickness along the path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileDef {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// 0: a ray; > 0: a sphere.
    pub radius: f32,
    pub mass: f32,
    pub penetration_cost: f32,
    /// Linear drag (1/s).
    pub drag: f32,
    pub max_hits_per_step: u32,
    pub filter: QueryFilter,
    pub ignore_body: BodyId,
    /// Hits push dynamic bodies.
    pub push_bodies: bool,
    pub user_data: u64,
}

impl Default for ProjectileDef {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            radius: 0.0,
            mass: 0.01,
            penetration_cost: 1e4,
            drag: 0.0,
            max_hits_per_step: 4,
            filter: QueryFilter::default(),
            ignore_body: BodyId::NULL,
            push_bodies: true,
            user_data: 0,
        }
    }
}

/// A projectile as a step left it (every one in flight, and each one the
/// frame it stops).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileOutput {
    pub projectile: ProjectileId,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub alive: bool,
    /// Kinetic energy left (J).
    pub energy: f32,
    pub hits: u32,
    pub user_data: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectileEventKind {
    /// Went through the shape, crossing `thickness` metres of it.
    Passed { thickness: f32 },
    /// Stopped in the shape.
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileEvent {
    pub projectile: ProjectileId,
    pub shape: ShapeId,
    pub body: BodyId,
    pub kind: ProjectileEventKind,
    pub user_data: u64,
}

#[derive(Clone, Debug)]
struct CharacterSlot {
    generation: u32,
    alive: bool,
    user_data: u64,
    jump_seq: u32,
    /// Results of steps before this one belong to an earlier occupant.
    since: u64,
    removed_at: Option<u64>,
    last: Option<CharacterOutput>,
}

#[derive(Clone, Debug)]
struct ProjectileSlot {
    generation: u32,
    /// Fired and not yet reported stopped.
    flying: bool,
    user_data: u64,
    since: u64,
}

#[derive(Default)]
pub(crate) struct MotionState {
    characters: Vec<CharacterSlot>,
    casts: Vec<SceneQuery>,
    cast_ids: Vec<CastId>,
    next_cast: u64,
    projectiles: Vec<ProjectileSlot>,
}

impl MotionState {
    pub(crate) fn has_characters(&self) -> bool {
        self.characters.iter().any(|c| c.alive)
    }
}

const QUARANTINE_STEPS: u64 = 2;

/// The GPU query mask (0: every collider). Scene queries test collider
/// categories against it only; `filter.categories` plays no part.
fn filter_mask(filter: QueryFilter) -> anyhow::Result<u32> {
    match filter.mask {
        0 => anyhow::bail!("a query filter mask of 0 hits nothing"),
        0xFFFF => Ok(0),
        m => Ok(u32::from(m)),
    }
}

impl World {
    // ----- characters ------------------------------------------------------

    /// Add a character standing with its capsule centre at `def.position`.
    /// Its state arrives in every frame (`Frame::characters`).
    pub fn create_character(&mut self, def: &CharacterDef) -> anyhow::Result<CharacterId> {
        let d = &def.desc;
        anyhow::ensure!(
            d.radius > 0.0 && d.half_height >= 0.0 && d.mass > 0.0,
            "character radius and mass must be positive and half height non-negative"
        );
        anyhow::ensure!(
            def.position.iter().all(|v| v.is_finite()),
            "character position must be finite"
        );
        self.dirty = true;
        let now = self.step;
        let reusable =
            self.motion.characters.iter().position(|c| {
                !c.alive && c.removed_at.is_some_and(|t| now >= t + QUARANTINE_STEPS)
            });
        let index = match reusable {
            Some(i) => {
                self.sim
                    .respawn_character(CharacterHandle(i as u32), d, def.position);
                let body = self.sim.character_body(CharacterHandle(i as u32)) as usize;
                self.bodies[body].alive = true;
                self.bodies[body].generation = self.bodies[body].generation.wrapping_add(1).max(1);
                i
            }
            None => {
                let cap = self.sim.character_capacity().max_characters;
                anyhow::ensure!(
                    self.sim.character_count() < cap,
                    "{cap} characters (CharacterCapacity::max_characters) reached"
                );
                let c = self.sim.capacity();
                anyhow::ensure!(
                    self.sim.body_count() < c.max_bodies
                        && self.sim.collider_count() < c.max_colliders,
                    "WorldCapacity::max_bodies or max_colliders reached"
                );
                let before = self.sim.body_count();
                let h = self.sim.add_character(d, def.position);
                self.adopt_owned(before);
                self.bodies[self.sim.character_body(h) as usize].owner = Owner::Character;
                self.motion.characters.push(CharacterSlot {
                    generation: 0,
                    alive: false,
                    user_data: 0,
                    jump_seq: 0,
                    since: 0,
                    removed_at: None,
                    last: None,
                });
                h.0 as usize
            }
        };
        let slot = &mut self.motion.characters[index];
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.alive = true;
        slot.user_data = def.user_data;
        slot.jump_seq = 0;
        slot.since = now;
        slot.removed_at = None;
        slot.last = None;
        Ok(CharacterId::new(index as u32, slot.generation))
    }

    /// Remove a character; its slot is reused after two steps.
    pub fn destroy_character(&mut self, id: CharacterId) -> anyhow::Result<()> {
        let i = self.live_character(id)?;
        self.dirty = true;
        let h = CharacterHandle(i as u32);
        let body = self.sim.character_body(h);
        let p = crate::graveyard(body);
        let mut parked = engine_schema::BodyState::static_plane_support();
        parked.position = p;
        parked.initial_pos = p;
        self.sim
            .disable_character(h, parked, crate::retired_collider(body));
        let b = &mut self.bodies[body as usize];
        b.alive = false;
        let slot = &mut self.motion.characters[i];
        slot.alive = false;
        slot.removed_at = Some(self.step);
        slot.last = None;
        Ok(())
    }

    pub fn character_is_valid(&self, id: CharacterId) -> bool {
        self.live_character(id).is_ok()
    }

    /// Desired horizontal velocity (world, m/s) from the next step on, and
    /// whether to jump once (when standing).
    pub fn character_set_input(
        &mut self,
        id: CharacterId,
        velocity: [f32; 3],
        jump: bool,
    ) -> anyhow::Result<()> {
        let i = self.live_character(id)?;
        anyhow::ensure!(
            velocity.iter().all(|v| v.is_finite()),
            "character velocity must be finite"
        );
        let slot = &mut self.motion.characters[i];
        if jump {
            slot.jump_seq = slot.jump_seq.wrapping_add(1);
        }
        let input = engine_schema::character::CharacterInput {
            velocity,
            jump_seq: slot.jump_seq,
        };
        self.sim
            .set_character_input(CharacterHandle(i as u32), input);
        self.dirty = true;
        Ok(())
    }

    /// Move a character's capsule centre to `position` before the next step.
    pub fn character_teleport(
        &mut self,
        id: CharacterId,
        position: [f32; 3],
    ) -> anyhow::Result<()> {
        let i = self.live_character(id)?;
        anyhow::ensure!(
            position.iter().all(|v| v.is_finite()),
            "character position must be finite"
        );
        self.sim
            .teleport_character(CharacterHandle(i as u32), position);
        self.dirty = true;
        Ok(())
    }

    /// The character as of the last polled frame (`None` before its first).
    pub fn character_state(&self, id: CharacterId) -> anyhow::Result<Option<CharacterOutput>> {
        Ok(self.motion.characters[self.live_character(id)?].last)
    }

    /// Its inner kinematic capsule, to leave out of casts (`CastDef::ignore_body`).
    pub fn character_body(&self, id: CharacterId) -> anyhow::Result<BodyId> {
        let i = self.live_character(id)?;
        let body = self.sim.character_body(CharacterHandle(i as u32));
        Ok(BodyId::new(body, self.bodies[body as usize].generation))
    }

    pub fn character_user_data(&self, id: CharacterId) -> anyhow::Result<u64> {
        Ok(self.motion.characters[self.live_character(id)?].user_data)
    }

    fn live_character(&self, id: CharacterId) -> anyhow::Result<usize> {
        let i = id.index() as usize;
        match self.motion.characters.get(i) {
            Some(c) if c.alive && c.generation == id.generation() => Ok(i),
            _ => Err(crate::StaleHandle("character", id.to_bits()).into()),
        }
    }

    // ----- casts -----------------------------------------------------------

    /// Queue a ray or shape cast for the next step (closest hit).
    pub fn cast(&mut self, def: &CastDef) -> anyhow::Result<CastId> {
        let cap = self.sim.character_capacity().max_queries;
        anyhow::ensure!(
            (self.motion.casts.len() as u32) < cap,
            "more than {cap} casts in one step (CharacterCapacity::max_queries)"
        );
        anyhow::ensure!(
            def.origin
                .iter()
                .chain(&def.direction)
                .all(|v| v.is_finite())
                && def.max_distance.is_finite()
                && def.max_distance >= 0.0,
            "cast origin, direction and distance must be finite"
        );
        anyhow::ensure!(
            def.direction.iter().any(|&v| v != 0.0),
            "cast direction must be non-zero"
        );
        let (radius, half_height) = match def.shape {
            CastShape::Ray => (0.0, 0.0),
            CastShape::Sphere { radius } => (radius, 0.0),
            CastShape::Capsule {
                radius,
                half_height,
            } => (radius, half_height),
        };
        anyhow::ensure!(
            radius >= 0.0 && half_height >= 0.0 && radius.is_finite() && half_height.is_finite(),
            "cast radius and half height must be non-negative"
        );
        let exclude = if def.ignore_body.is_null() {
            NO_INDEX
        } else {
            self.live_body(def.ignore_body)? as u32
        };
        let mask = filter_mask(def.filter)?;
        self.motion.next_cast += 1;
        let id = CastId(self.motion.next_cast);
        self.motion.casts.push(SceneQuery {
            origin: def.origin,
            dir: def.direction,
            max_dist: def.max_distance,
            radius,
            half_height,
            exclude_body: exclude,
            mask,
            full_scan: false,
        });
        self.motion.cast_ids.push(id);
        self.dirty = true;
        Ok(id)
    }

    // ----- projectiles -----------------------------------------------------

    /// Fire a projectile from the next step on. Slots of stopped projectiles
    /// are reused.
    pub fn spawn_projectile(&mut self, def: &ProjectileDef) -> anyhow::Result<ProjectileId> {
        anyhow::ensure!(
            def.position.iter().chain(&def.velocity).all(|v| v.is_finite())
                && def.radius >= 0.0
                && def.mass > 0.0
                && def.penetration_cost >= 0.0
                && def.drag >= 0.0,
            "projectile position and velocity must be finite, radius, cost and drag non-negative, mass positive"
        );
        let exclude = if def.ignore_body.is_null() {
            NO_INDEX
        } else {
            self.live_body(def.ignore_body)? as u32
        };
        let mask = filter_mask(def.filter)?;
        let desc = ProjectileDesc {
            position: def.position,
            velocity: def.velocity,
            radius: def.radius,
            mass: def.mass,
            penetration_cost: def.penetration_cost,
            drag: def.drag,
            max_hits_per_step: def.max_hits_per_step.max(1),
            exclude_body: exclude,
            mask,
            impulses: def.push_bodies,
        };
        let live = self.sim.projectile_count() as usize;
        let index = match self.motion.projectiles[..live]
            .iter()
            .position(|p| !p.flying)
        {
            Some(i) => i,
            None => {
                let cap = self.sim.character_capacity().max_projectiles;
                anyhow::ensure!(
                    (live as u32) < cap,
                    "{cap} projectiles in flight (CharacterCapacity::max_projectiles)"
                );
                if self.motion.projectiles.len() == live {
                    self.motion.projectiles.push(ProjectileSlot {
                        generation: 0,
                        flying: false,
                        user_data: 0,
                        since: 0,
                    });
                }
                live
            }
        };
        self.sim.respawn_projectile(index as u32, &desc);
        let slot = &mut self.motion.projectiles[index];
        slot.generation = slot.generation.wrapping_add(1).max(1);
        slot.flying = true;
        slot.user_data = def.user_data;
        slot.since = self.step;
        self.dirty = true;
        Ok(ProjectileId::new(index as u32, slot.generation))
    }

    // ----- per step --------------------------------------------------------

    /// Before encoding a step: write this step's casts (or stop answering the
    /// last ones), and release the projectile pool once every one stopped.
    pub(crate) fn motion_before_step(&mut self) -> Vec<CastId> {
        let casts = std::mem::take(&mut self.motion.casts);
        if !casts.is_empty() {
            self.sim.set_queries(&casts);
        } else if self.sim.query_count() > 0 {
            self.sim.clear_queries();
        }
        let live = self.sim.projectile_count() as usize;
        if live > 0 && self.motion.projectiles[..live].iter().all(|p| !p.flying) {
            self.sim.clear_projectiles();
        }
        std::mem::take(&mut self.motion.cast_ids)
    }
}

/// This step's readback copies of the motion records.
pub(crate) fn extras<'a>(
    sim: &'a engine_webgpu::WgpuSim,
    state: &MotionState,
    casts: Vec<CastId>,
) -> Vec<readback::Extra<'a, Extra>> {
    let mut out = Vec::new();
    if sim.character_count() > 0 && state.has_characters() {
        let (buffer, offset, size) = sim.character_record_range();
        out.push(readback::Extra {
            kind: Extra::Characters,
            buffer,
            offset,
            size,
        });
    }
    if !casts.is_empty() {
        let (buffer, offset, size) = sim.query_record_range();
        out.push(readback::Extra {
            kind: Extra::Casts(casts),
            buffer,
            offset,
            size,
        });
    }
    if sim.projectile_count() > 0 {
        let (buffer, offset, size) = sim.projectile_record_range();
        out.push(readback::Extra {
            kind: Extra::Projectiles,
            buffer,
            offset,
            size,
        });
    }
    out
}

impl World {
    fn body_id(&self, body: u32) -> BodyId {
        match self.bodies.get(body as usize) {
            Some(b) if b.alive && matches!(b.owner, Owner::User | Owner::Vehicle) => {
                BodyId::new(body, b.generation)
            }
            _ => BodyId::NULL,
        }
    }

    /// The shape and body a collider index belongs to.
    pub(crate) fn collider_ids(&self, collider: u32) -> (ShapeId, BodyId) {
        let shape = self.shape_id_of_collider(collider).unwrap_or(ShapeId::NULL);
        let body = self
            .sim
            .collider_body(collider)
            .map_or(BodyId::NULL, |b| self.body_id(b));
        (shape, body)
    }

    pub(crate) fn decode_characters(&mut self, step: u64, bytes: &[u8]) -> Vec<CharacterOutput> {
        let slots = crate::slots(bytes);
        let mut out = Vec::new();
        for (i, rec) in slots.chunks_exact(CHAR_STRIDE as usize).enumerate() {
            let Some(slot) = self.motion.characters.get(i) else {
                break;
            };
            if !slot.alive || step < slot.since {
                continue;
            }
            let s = CharacterState::decode(rec);
            let ground_body = s.ground_body.map_or(BodyId::NULL, |b| self.body_id(b));
            let o = CharacterOutput {
                character: CharacterId::new(i as u32, slot.generation),
                position: s.position,
                velocity: s.velocity,
                ground: s.ground,
                ground_normal: s.ground_normal,
                ground_body,
                ground_velocity: s.ground_velocity,
                push: s.push,
                user_data: slot.user_data,
            };
            self.motion.characters[i].last = Some(o);
            out.push(o);
        }
        out
    }

    pub(crate) fn decode_casts(&self, ids: &[CastId], bytes: &[u8]) -> Vec<CastResult> {
        let slots = crate::slots(bytes);
        ids.iter()
            .zip(slots.chunks_exact(QUERY_STRIDE as usize))
            .map(|(&cast, rec)| CastResult {
                cast,
                hit: QueryHit::decode(rec).map(|h| {
                    let (shape, body) = self.collider_ids(h.collider);
                    CastHit {
                        shape,
                        body,
                        point: h.point,
                        normal: h.normal,
                        distance: h.distance,
                    }
                }),
            })
            .collect()
    }

    pub(crate) fn decode_projectiles(&mut self, step: u64, bytes: &[u8]) -> Vec<ProjectileOutput> {
        let slots = crate::slots(bytes);
        let mut out = Vec::new();
        for (i, rec) in slots.chunks_exact(PROJ_STRIDE as usize).enumerate() {
            let Some(slot) = self.motion.projectiles.get_mut(i) else {
                break;
            };
            if !slot.flying || step < slot.since {
                continue;
            }
            let s = ProjectileState::decode(rec);
            if !s.alive {
                slot.flying = false;
            }
            out.push(ProjectileOutput {
                projectile: ProjectileId::new(i as u32, slot.generation),
                position: s.position,
                velocity: s.velocity,
                alive: s.alive,
                energy: s.energy,
                hits: s.total_hits,
                user_data: slot.user_data,
            });
        }
        out
    }

    /// Character and projectile events in this step's copy of the event ring.
    pub(crate) fn motion_events(
        &mut self,
        step: u64,
        ring: &[u32],
    ) -> (Vec<CharacterEvent>, Vec<ProjectileEvent>, u32) {
        let (events, lost) = self.sim.motion_events_from(ring);
        let mut characters = Vec::new();
        let mut projectiles = Vec::new();
        for e in events {
            match e {
                MotionEvent::Landed {
                    character,
                    ground_body,
                }
                | MotionEvent::SupportLost {
                    character,
                    ground_body,
                }
                | MotionEvent::Crushed {
                    character,
                    ground_body,
                } => {
                    let Some(slot) = self.motion.characters.get(character as usize) else {
                        continue;
                    };
                    if !slot.alive || step < slot.since {
                        continue;
                    }
                    let kind = match e {
                        MotionEvent::Landed { .. } => CharacterEventKind::Landed,
                        MotionEvent::SupportLost { .. } => CharacterEventKind::SupportLost,
                        _ => CharacterEventKind::Crushed,
                    };
                    characters.push(CharacterEvent {
                        character: CharacterId::new(character, slot.generation),
                        kind,
                        ground_body: if ground_body == NO_INDEX {
                            BodyId::NULL
                        } else {
                            self.body_id(ground_body)
                        },
                        user_data: slot.user_data,
                    });
                }
                MotionEvent::ProjectileHit {
                    projectile,
                    collider,
                    thickness,
                } => projectiles.extend(self.projectile_event(
                    step,
                    projectile,
                    collider,
                    if thickness < 0.0 {
                        ProjectileEventKind::Stopped
                    } else {
                        ProjectileEventKind::Passed { thickness }
                    },
                )),
                // Follows the hit that stopped it (reported as `Stopped`).
                MotionEvent::ProjectileStopped { .. } => {}
            }
        }
        (characters, projectiles, lost)
    }

    fn projectile_event(
        &self,
        step: u64,
        projectile: u32,
        collider: u32,
        kind: ProjectileEventKind,
    ) -> Option<ProjectileEvent> {
        let slot = self.motion.projectiles.get(projectile as usize)?;
        // A step before the slot's respawn reports its previous projectile.
        let generation = if step < slot.since {
            slot.generation.wrapping_sub(1)
        } else {
            slot.generation
        };
        if generation == 0 {
            return None;
        }
        let (shape, body) = self.collider_ids(collider);
        Some(ProjectileEvent {
            projectile: ProjectileId::new(projectile, generation),
            shape,
            body,
            kind,
            user_data: slot.user_data,
        })
    }
}
