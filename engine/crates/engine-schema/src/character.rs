//! Character controllers, in-step scene queries and projectiles: the pool in
//! `gCharData` (binding 32, float4 slots). Mirrors
//! `slang/physics/character_layout.slang`; `layout_matches_slang` keeps them
//! equal. The kernels are in `slang/kernels/character_step.slang`.

use serde::{Deserialize, Serialize};

pub const CH_COUNTS: u32 = 0; // uint4(maxCharacters, characterCount, maxQueries, queryCount)
pub const CH_BASES: u32 = 1; // uint4(characterBase, queryBase, projectileBase, maxProjectiles)
pub const CH_MISC: u32 = 2; // uint4(projectileCount, queryMode (0 grid walk, 1 full scan), pad, pad)
pub const CH_SLOTS: u32 = 4;
pub const CHAR_STRIDE: u32 = 16;
pub const C_CFG: u32 = 0; // uint4(innerBody, flags, queryMask, pad)
pub const C_SHAPE: u32 = 1; // radius, halfHeight, skin, cosMaxSlope
pub const C_MOVE: u32 = 2; // stepUp, stepDown, jumpSpeed, airControl
pub const C_MASS: u32 = 3; // mass, pushRatio, maxRecovery, gravityScale
pub const C_IN: u32 = 4; // desired velocity.xyz, asfloat(jumpSeq)
pub const C_POS: u32 = 5; // capsule centre.xyz, asfloat(lastJumpSeq)
pub const C_VEL: u32 = 6; // velocity.xyz, asfloat(groundState)
pub const C_GND: u32 = 7; // ground normal.xyz, asfloat(groundBody)
pub const C_GPT: u32 = 8; // ground point.xyz, asfloat(groundCollider)
pub const C_GVEL: u32 = 9; // ground point velocity.xyz, time in air
pub const C_OUT: u32 = 10; // uint4(eventSeq, lastEvent, planes, outFlags)
pub const C_PUSH: u32 = 11; // weight on a dynamic ground (N).xyz, pad
pub const C_DBG: u32 = 12; // verify-cast fraction, stepped height, blocking planes, pad
pub const CF_ENABLED: u32 = 1;
pub const CF_PUSH_WEIGHT: u32 = 2;
pub const GS_ON_GROUND: u32 = 0;
pub const GS_ON_STEEP: u32 = 1;
pub const GS_IN_AIR: u32 = 2;
pub const CO_STEPPED_UP: u32 = 1;
pub const CO_STUCK_TO_FLOOR: u32 = 2;
pub const CO_CRUSHED: u32 = 4;
pub const CO_JUMPED: u32 = 8;
pub const CO_CAST_CUT: u32 = 16;
pub const QUERY_STRIDE: u32 = 5;
pub const Q_IN0: u32 = 0;
pub const Q_IN1: u32 = 1;
pub const Q_IN2: u32 = 2;
pub const Q_OUT0: u32 = 3;
pub const Q_OUT1: u32 = 4;
pub const PROJ_STRIDE: u32 = 6;
pub const PJ_POS: u32 = 0;
pub const PJ_VEL: u32 = 1;
pub const PJ_CFG: u32 = 2;
pub const PJ_PEN: u32 = 3;
pub const PJ_OUT: u32 = 4;
pub const PJ_OUT2: u32 = 5;
pub const PF_ALIVE: u32 = 1;
pub const PF_IMPULSES: u32 = 2;
pub const EVENT_CHAR_LANDED: u32 = 5;
pub const EVENT_CHAR_SUPPORT_LOST: u32 = 6;
pub const EVENT_CHAR_CRUSHED: u32 = 7;
pub const EVENT_PROJECTILE_HIT: u32 = 8;
pub const EVENT_PROJECTILE_STOPPED: u32 = 9;
pub const CHAR_MAX_CANDIDATES: u32 = 192;
pub const PROJ_SEGMENTS: u32 = 4;
pub const PROJ_MAX_HITS: u32 = 8;

/// Every layout constant by name (hosts that cannot link this crate export it).
pub const LAYOUT: &[(&str, u32)] = &[
    ("CH_COUNTS", CH_COUNTS),
    ("CH_BASES", CH_BASES),
    ("CH_MISC", CH_MISC),
    ("CH_SLOTS", CH_SLOTS),
    ("CHAR_STRIDE", CHAR_STRIDE),
    ("C_CFG", C_CFG),
    ("C_SHAPE", C_SHAPE),
    ("C_MOVE", C_MOVE),
    ("C_MASS", C_MASS),
    ("C_IN", C_IN),
    ("C_POS", C_POS),
    ("C_VEL", C_VEL),
    ("C_GND", C_GND),
    ("C_GPT", C_GPT),
    ("C_GVEL", C_GVEL),
    ("C_OUT", C_OUT),
    ("C_PUSH", C_PUSH),
    ("C_DBG", C_DBG),
    ("CF_ENABLED", CF_ENABLED),
    ("CF_PUSH_WEIGHT", CF_PUSH_WEIGHT),
    ("GS_ON_GROUND", GS_ON_GROUND),
    ("GS_ON_STEEP", GS_ON_STEEP),
    ("GS_IN_AIR", GS_IN_AIR),
    ("CO_STEPPED_UP", CO_STEPPED_UP),
    ("CO_STUCK_TO_FLOOR", CO_STUCK_TO_FLOOR),
    ("CO_CRUSHED", CO_CRUSHED),
    ("CO_JUMPED", CO_JUMPED),
    ("CO_CAST_CUT", CO_CAST_CUT),
    ("QUERY_STRIDE", QUERY_STRIDE),
    ("Q_IN0", Q_IN0),
    ("Q_IN1", Q_IN1),
    ("Q_IN2", Q_IN2),
    ("Q_OUT0", Q_OUT0),
    ("Q_OUT1", Q_OUT1),
    ("PROJ_STRIDE", PROJ_STRIDE),
    ("PJ_POS", PJ_POS),
    ("PJ_VEL", PJ_VEL),
    ("PJ_CFG", PJ_CFG),
    ("PJ_PEN", PJ_PEN),
    ("PJ_OUT", PJ_OUT),
    ("PJ_OUT2", PJ_OUT2),
    ("PF_ALIVE", PF_ALIVE),
    ("PF_IMPULSES", PF_IMPULSES),
    ("EVENT_CHAR_LANDED", EVENT_CHAR_LANDED),
    ("EVENT_CHAR_SUPPORT_LOST", EVENT_CHAR_SUPPORT_LOST),
    ("EVENT_CHAR_CRUSHED", EVENT_CHAR_CRUSHED),
    ("EVENT_PROJECTILE_HIT", EVENT_PROJECTILE_HIT),
    ("EVENT_PROJECTILE_STOPPED", EVENT_PROJECTILE_STOPPED),
    ("CHAR_MAX_CANDIDATES", CHAR_MAX_CANDIDATES),
    ("PROJ_SEGMENTS", PROJ_SEGMENTS),
    ("PROJ_MAX_HITS", PROJ_MAX_HITS),
];

/// Slots `NO_INDEX` (FLT_MAX bits, as in the vehicle pool).
pub const NO_INDEX: u32 = crate::vehicle::NO_INDEX;

/// Pool sizes (binding 32). Explicit limits: adding past them fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CharacterCapacity {
    pub max_characters: u32,
    pub max_queries: u32,
    pub max_projectiles: u32,
}

impl Default for CharacterCapacity {
    fn default() -> Self {
        Self { max_characters: 64, max_queries: 256, max_projectiles: 256 }
    }
}

impl CharacterCapacity {
    pub fn character_base(&self) -> u32 {
        CH_SLOTS
    }
    pub fn query_base(&self) -> u32 {
        self.character_base() + self.max_characters * CHAR_STRIDE
    }
    pub fn projectile_base(&self) -> u32 {
        self.query_base() + self.max_queries * QUERY_STRIDE
    }
    pub fn slots(&self) -> u32 {
        self.projectile_base() + self.max_projectiles * PROJ_STRIDE
    }
    pub fn bytes(&self) -> u64 {
        u64::from(self.slots()) * 16
    }
    /// The header slots (counts zero).
    pub fn header(&self) -> [[u32; 4]; 4] {
        [
            [self.max_characters, 0, self.max_queries, 0],
            [self.character_base(), self.query_base(), self.projectile_base(), self.max_projectiles],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    }
}

/// A capsule character (kinematic mover). Units: m, kg, s.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CharacterDesc {
    /// Capsule radius and half the distance between its sphere centres.
    pub radius: f32,
    pub half_height: f32,
    /// Gap kept to every surface (padding), m.
    pub skin: f32,
    /// Steepest walkable ground, degrees.
    pub max_slope_deg: f32,
    /// Highest step climbed while walking, and how far down it sticks to the floor.
    pub step_up: f32,
    pub step_down: f32,
    pub jump_speed: f32,
    /// How fast horizontal velocity follows the input in the air (1/s).
    pub air_control: f32,
    pub mass: f32,
    /// Dynamic bodies lighter than `mass × push_ratio` do not block the
    /// character; its inner capsule pushes them in the solve.
    pub push_ratio: f32,
    /// Most overlap removed per step (m), so a character pinned by debris works
    /// free without being launched.
    pub max_recovery: f32,
    pub gravity_scale: f32,
    /// Collider categories it walks against (0 = all; scene filters on).
    pub query_mask: u32,
    /// Put the character's weight on dynamic bodies it stands on.
    pub push_weight: bool,
}

impl Default for CharacterDesc {
    fn default() -> Self {
        Self {
            radius: 0.3,
            half_height: 0.6,
            skin: 0.02,
            max_slope_deg: 45.0,
            step_up: 0.35,
            step_down: 0.35,
            jump_speed: 5.0,
            air_control: 4.0,
            mass: 80.0,
            push_ratio: 1.0,
            max_recovery: 0.05,
            gravity_scale: 1.0,
            query_mask: 0,
            push_weight: true,
        }
    }
}

impl CharacterDesc {
    /// The record's slots (16 float4 as u32 words) for a character whose inner
    /// kinematic capsule is `inner_body`, standing with its centre at `pos`.
    pub fn record(&self, inner_body: u32, pos: [f32; 3]) -> [[u32; 4]; CHAR_STRIDE as usize] {
        let f = |v: f32| v.to_bits();
        let mut s = [[0u32; 4]; CHAR_STRIDE as usize];
        let flags = CF_ENABLED | if self.push_weight { CF_PUSH_WEIGHT } else { 0 };
        s[C_CFG as usize] = [inner_body, flags, self.query_mask, 0];
        s[C_SHAPE as usize] = [f(self.radius), f(self.half_height), f(self.skin), f(self.max_slope_deg.to_radians().cos())];
        s[C_MOVE as usize] = [f(self.step_up), f(self.step_down), f(self.jump_speed), f(self.air_control)];
        s[C_MASS as usize] = [f(self.mass), f(self.push_ratio), f(self.max_recovery), f(self.gravity_scale)];
        s[C_IN as usize] = [0, 0, 0, 0];
        s[C_POS as usize] = [f(pos[0]), f(pos[1]), f(pos[2]), 0];
        s[C_VEL as usize] = [0, 0, 0, GS_IN_AIR];
        s[C_GND as usize] = [0, f(1.0), 0, NO_INDEX];
        s[C_GPT as usize] = [0, 0, 0, NO_INDEX];
        s
    }
}

/// What a character's input record holds (written by the host each frame).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CharacterInput {
    /// Desired horizontal velocity (world, m/s).
    pub velocity: [f32; 3],
    /// Bump to jump once (the GPU jumps when it differs from the last one used).
    pub jump_seq: u32,
}

impl CharacterInput {
    pub fn slot(&self) -> [u32; 4] {
        [self.velocity[0].to_bits(), self.velocity[1].to_bits(), self.velocity[2].to_bits(), self.jump_seq]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundState {
    OnGround,
    OnSteepGround,
    InAir,
}

/// A character as the last step left it (decoded from its record).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CharacterState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub ground: GroundState,
    pub ground_normal: [f32; 3],
    /// Body under the character (`None`: in the air).
    pub ground_body: Option<u32>,
    pub ground_velocity: [f32; 3],
    pub out_flags: u32,
    pub planes: u32,
    /// Weight put on a dynamic support this step (N).
    pub push: [f32; 3],
    pub stepped: f32,
}

impl CharacterState {
    pub fn decode(slots: &[[u32; 4]]) -> Self {
        let f = |s: u32, i: usize| f32::from_bits(slots[s as usize][i]);
        let v3 = |s: u32| [f(s, 0), f(s, 1), f(s, 2)];
        let gb = slots[C_GND as usize][3];
        Self {
            position: v3(C_POS),
            velocity: v3(C_VEL),
            ground: match slots[C_VEL as usize][3] {
                GS_ON_GROUND => GroundState::OnGround,
                GS_ON_STEEP => GroundState::OnSteepGround,
                _ => GroundState::InAir,
            },
            ground_normal: v3(C_GND),
            ground_body: (gb != NO_INDEX).then_some(gb),
            ground_velocity: v3(C_GVEL),
            out_flags: slots[C_OUT as usize][3],
            planes: slots[C_OUT as usize][2],
            push: v3(C_PUSH),
            stepped: f(C_DBG, 1),
        }
    }
}

/// A scene query answered inside the next step (`query_batch`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneQuery {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
    pub max_dist: f32,
    /// 0: a ray; > 0: a sphere (or capsule) radius.
    pub radius: f32,
    /// > 0: an upright capsule cast with this half height.
    pub half_height: f32,
    pub exclude_body: u32,
    pub mask: u32,
}

impl SceneQuery {
    pub fn ray(origin: [f32; 3], dir: [f32; 3], max_dist: f32) -> Self {
        Self { origin, dir, max_dist, radius: 0.0, half_height: 0.0, exclude_body: NO_INDEX, mask: 0 }
    }
    pub fn slots(&self) -> [[u32; 4]; QUERY_STRIDE as usize] {
        let f = |v: f32| v.to_bits();
        let kind = u32::from(self.half_height > 0.0);
        [
            [f(self.origin[0]), f(self.origin[1]), f(self.origin[2]), f(self.max_dist)],
            [f(self.dir[0]), f(self.dir[1]), f(self.dir[2]), f(self.radius)],
            [self.exclude_body, self.mask, kind, f(self.half_height)],
            [0, 0, 0, f(-1.0)],
            [0, 0, 0, NO_INDEX],
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryHit {
    pub point: [f32; 3],
    pub distance: f32,
    pub normal: [f32; 3],
    pub collider: u32,
}

impl QueryHit {
    pub fn decode(slots: &[[u32; 4]]) -> Option<Self> {
        let f = |s: u32, i: usize| f32::from_bits(slots[s as usize][i]);
        let d = f(Q_OUT0, 3);
        (d >= 0.0).then(|| Self {
            point: [f(Q_OUT0, 0), f(Q_OUT0, 1), f(Q_OUT0, 2)],
            distance: d,
            normal: [f(Q_OUT1, 0), f(Q_OUT1, 1), f(Q_OUT1, 2)],
            collider: slots[Q_OUT1 as usize][3],
        })
    }
}

/// A bullet-class projectile: segment casts along its ballistic arc before the
/// step (no rigid body). Energy decides penetration: crossing a collider costs
/// `penetration_cost` J per metre of its thickness along the path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileDesc {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// 0: a ray; > 0: a sphere cast.
    pub radius: f32,
    pub mass: f32,
    pub penetration_cost: f32,
    /// Linear drag (1/s).
    pub drag: f32,
    pub max_hits_per_step: u32,
    pub exclude_body: u32,
    pub mask: u32,
    /// Hits push dynamic bodies (impulse records).
    pub impulses: bool,
}

impl ProjectileDesc {
    pub fn slots(&self) -> [[u32; 4]; PROJ_STRIDE as usize] {
        let f = |v: f32| v.to_bits();
        let v = self.velocity;
        let e = 0.5 * self.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        let flags = PF_ALIVE | if self.impulses { PF_IMPULSES } else { 0 };
        [
            [f(self.position[0]), f(self.position[1]), f(self.position[2]), f(self.radius)],
            [f(v[0]), f(v[1]), f(v[2]), f(self.mass)],
            [flags, self.mask, self.exclude_body, 0],
            [f(e), f(self.penetration_cost), f(self.drag), f(self.max_hits_per_step as f32)],
            [0, 0, 0, NO_INDEX],
            [0, 0, 0, 0],
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectileState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub alive: bool,
    pub total_hits: u32,
    pub energy: f32,
    pub last_hit: Option<(u32, [f32; 3], [f32; 3])>,
}

impl ProjectileState {
    pub fn decode(slots: &[[u32; 4]]) -> Self {
        let f = |s: u32, i: usize| f32::from_bits(slots[s as usize][i]);
        let v3 = |s: u32| [f(s, 0), f(s, 1), f(s, 2)];
        let hit = slots[PJ_OUT as usize][3];
        Self {
            position: v3(PJ_POS),
            velocity: v3(PJ_VEL),
            alive: slots[PJ_CFG as usize][0] & PF_ALIVE != 0,
            total_hits: slots[PJ_CFG as usize][3],
            energy: f(PJ_PEN, 0),
            last_hit: (hit != NO_INDEX).then(|| (hit, v3(PJ_OUT), v3(PJ_OUT2))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_slang() {
        let src = include_str!("../../../slang/physics/character_layout.slang");
        let mut n = 0;
        for line in src.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("static const uint ") else { continue };
            let Some((name, rhs)) = rest.split_once('=') else { continue };
            let lit = rhs.trim().split('u').next().unwrap().trim();
            let v: u32 = lit.parse().unwrap();
            let rust = LAYOUT
                .iter()
                .find(|(k, _)| *k == name.trim())
                .unwrap_or_else(|| panic!("{} missing in character.rs", name.trim()))
                .1;
            assert_eq!(rust, v, "{}", name.trim());
            n += 1;
        }
        assert_eq!(n, LAYOUT.len(), "every Rust constant is in the Slang layout");
    }

    #[test]
    fn pool_regions_do_not_overlap() {
        let c = CharacterCapacity::default();
        assert!(c.query_base() >= c.character_base() + c.max_characters * CHAR_STRIDE);
        assert!(c.projectile_base() >= c.query_base() + c.max_queries * QUERY_STRIDE);
        assert_eq!(c.bytes(), u64::from(c.slots()) * 16);
    }
}
