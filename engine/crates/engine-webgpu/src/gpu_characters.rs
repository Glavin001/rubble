//! Characters, in-step scene queries and projectiles on the native wgpu host
//! (`slang/kernels/character_step.slang`, binding 32). Records are written
//! between steps; the step's own kernels move characters, answer queries and
//! cast projectiles, and the results are read back after it.

use super::WgpuSim;
use engine_schema::character::*;
use engine_schema::{BodyFlags, BodyState, Collider, ShapeType, EVENT_RING_BASE, EVENT_RING_CAPACITY};

/// A character controller (index into the pool).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CharacterHandle(pub u32);

/// A character or projectile event from the step's event ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotionEvent {
    Landed { character: u32, ground_body: u32 },
    SupportLost { character: u32, ground_body: u32 },
    Crushed { character: u32, ground_body: u32 },
    /// `thickness` < 0: the projectile stopped in this collider.
    ProjectileHit { projectile: u32, collider: u32, thickness: f32 },
    ProjectileStopped { projectile: u32, collider: u32 },
}

impl WgpuSim {
    fn write_char_slots(&self, slot: u32, words: &[[u32; 4]]) {
        self.queue.write_buffer(&self.bufs[32], u64::from(slot) * 16, bytemuck::cast_slice(words));
    }

    pub(crate) fn write_character_header(&self) {
        let mut h = self.char_cap.header();
        h[CH_COUNTS as usize][1] = self.characters.len() as u32;
        h[CH_COUNTS as usize][3] = self.query_count;
        h[CH_MISC as usize][0] = self.projectile_count;
        self.write_char_slots(0, &h);
    }

    fn read_char_slots(&self, slot: u32, n: u32) -> Vec<[u32; 4]> {
        let words = self.read_words(32, u64::from(slot) * 4, n as usize * 4);
        words.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }

    /// Add a capsule character standing with its centre at `pos`. Its inner
    /// kinematic capsule (a body and collider of its own) is what the world's
    /// dynamic bodies collide with; the character never collides with it.
    pub fn add_character(&mut self, desc: &CharacterDesc, pos: [f32; 3]) -> CharacterHandle {
        let index = self.characters.len() as u32;
        assert!(index < self.char_cap.max_characters, "more than {} characters", self.char_cap.max_characters);
        let body = self.add_body(BodyState {
            position: pos,
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: BodyFlags::Kinematic as u32 | BodyFlags::Awake as u32,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: pos,
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        });
        let collider = self.add_collider(Collider {
            body,
            shape: ShapeType::Capsule as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [desc.radius, desc.half_height, 0.0, desc.radius + desc.half_height],
        });
        self.characters.push((body, collider));
        let slot = self.char_cap.character_base() + index * CHAR_STRIDE;
        self.write_char_slots(slot, &desc.record(body, pos));
        self.write_character_header();
        self.wake_pending = true;
        CharacterHandle(index)
    }

    /// The character's input for the next steps (desired horizontal velocity,
    /// jump sequence).
    pub fn set_character_input(&mut self, h: CharacterHandle, input: CharacterInput) {
        let slot = self.char_cap.character_base() + h.0 * CHAR_STRIDE + C_IN;
        self.write_char_slots(slot, &[input.slot()]);
    }

    /// The character's inner kinematic capsule body.
    pub fn character_body(&self, h: CharacterHandle) -> u32 {
        self.characters[h.0 as usize].0
    }

    pub fn character_count(&self) -> u32 {
        self.characters.len() as u32
    }

    /// Every character as the last step left it (one blocking read).
    pub fn character_states(&self) -> Vec<CharacterState> {
        let n = self.characters.len() as u32;
        if n == 0 {
            return Vec::new();
        }
        let slots = self.read_char_slots(self.char_cap.character_base(), n * CHAR_STRIDE);
        slots.chunks(CHAR_STRIDE as usize).map(CharacterState::decode).collect()
    }

    pub fn character_state(&self, h: CharacterHandle) -> CharacterState {
        let slots = self.read_char_slots(self.char_cap.character_base() + h.0 * CHAR_STRIDE, CHAR_STRIDE);
        CharacterState::decode(&slots)
    }

    /// Scene queries the next steps answer inside the step (grid walk, exact
    /// hulls, filters); read with `query_results` after a step.
    pub fn set_queries(&mut self, queries: &[SceneQuery]) {
        assert!(queries.len() as u32 <= self.char_cap.max_queries, "more than {} queries", self.char_cap.max_queries);
        let base = self.char_cap.query_base();
        let words: Vec<[u32; 4]> = queries.iter().flat_map(|q| q.slots()).collect();
        if !words.is_empty() {
            self.write_char_slots(base, &words);
        }
        self.query_count = queries.len() as u32;
        self.write_character_header();
        // Answered by the next step, even in a world that is all asleep.
        self.wake_pending = true;
    }

    /// Results of the queries the last step answered (`None`: no hit).
    pub fn query_results(&self) -> Vec<Option<QueryHit>> {
        if self.query_count == 0 {
            return Vec::new();
        }
        let slots = self.read_char_slots(self.char_cap.query_base(), self.query_count * QUERY_STRIDE);
        slots.chunks(QUERY_STRIDE as usize).map(QueryHit::decode).collect()
    }

    /// Walk the whole collider list instead of the grid (the oracle the grid
    /// walk is checked against).
    pub fn set_query_full_scan(&mut self, on: bool) {
        let mut h = self.char_cap.header();
        h[CH_COUNTS as usize][1] = self.characters.len() as u32;
        h[CH_COUNTS as usize][3] = self.query_count;
        h[CH_MISC as usize][0] = self.projectile_count;
        h[CH_MISC as usize][1] = u32::from(on);
        self.write_char_slots(0, &h);
    }

    /// Fire a projectile: segment casts along its arc from the next step on.
    pub fn spawn_projectile(&mut self, desc: &ProjectileDesc) -> u32 {
        let index = self.projectile_count;
        assert!(index < self.char_cap.max_projectiles, "more than {} projectiles", self.char_cap.max_projectiles);
        self.write_char_slots(self.char_cap.projectile_base() + index * PROJ_STRIDE, &desc.slots());
        self.projectile_count += 1;
        self.write_character_header();
        self.wake_pending = true;
        index
    }

    pub fn clear_projectiles(&mut self) {
        self.projectile_count = 0;
        self.write_character_header();
    }

    pub fn projectile_states(&self) -> Vec<ProjectileState> {
        if self.projectile_count == 0 {
            return Vec::new();
        }
        let slots = self.read_char_slots(self.char_cap.projectile_base(), self.projectile_count * PROJ_STRIDE);
        slots.chunks(PROJ_STRIDE as usize).map(ProjectileState::decode).collect()
    }

    /// Character and projectile events since the last call (event ring).
    pub fn poll_motion_events(&mut self) -> Vec<MotionEvent> {
        let words = self.read_words(21, u64::from(EVENT_RING_BASE), (4 + 4 * EVENT_RING_CAPACITY) as usize);
        let cursor = words[0];
        let mut n = self.motion_events_seen.max(cursor.saturating_sub(EVENT_RING_CAPACITY));
        let mut out = Vec::new();
        while n < cursor {
            let r = 4 + 4 * (n % EVENT_RING_CAPACITY) as usize;
            let (ty, id, a, b) = (words[r], words[r + 1], words[r + 2], words[r + 3]);
            match ty {
                EVENT_CHAR_LANDED => out.push(MotionEvent::Landed { character: id, ground_body: a }),
                EVENT_CHAR_SUPPORT_LOST => out.push(MotionEvent::SupportLost { character: id, ground_body: a }),
                EVENT_CHAR_CRUSHED => out.push(MotionEvent::Crushed { character: id, ground_body: a }),
                EVENT_PROJECTILE_HIT => out.push(MotionEvent::ProjectileHit {
                    projectile: id,
                    collider: a,
                    thickness: f32::from_bits(b),
                }),
                EVENT_PROJECTILE_STOPPED => out.push(MotionEvent::ProjectileStopped { projectile: id, collider: a }),
                _ => {}
            }
            n += 1;
        }
        self.motion_events_seen = cursor;
        out
    }
}
