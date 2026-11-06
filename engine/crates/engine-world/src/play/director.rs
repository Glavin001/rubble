//! What happens in a placed scene after placement: the player's [`Action`]s (fire, cannon,
//! ram, crater…) and each scene's own script (a volley every 3 s, the moving platform, the
//! T-bone ram's launch, the crusher's lift…). One [`Director`] per placed scene, made from
//! its [`SceneDef`] and [`Placed`]; hosts call [`Director::before_step`] before every step
//! and [`Director::act`] for input. Pages, the C API and native hosts all run the same
//! script, so no host keeps its own copy.
//!
//! Directors read poses as of the last polled frame ([`World::body_transform`]), as the
//! pages read their last readback: frame-latent by design, never a hidden wait.

use super::scene::{Placed, SceneDef};
use super::vehicle_director::VehicleDirector;
use super::{hills, motion, softbody};
use crate::*;
use serde::{Deserialize, Serialize};

/// A camera ray: where the player looks from and along (unit length).
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct Aim {
    pub eye: [f64; 3],
    pub dir: [f64; 3],
}

/// Something the player does to a placed scene.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "action", rename_all = "camelCase")]
pub enum Action {
    /// Every body (and character) back to its start, at rest; the script starts over.
    Reset,
    /// The scene's fire key (F): the gallery's volley, the range's burst, the character's
    /// bullet along `aim`.
    Fire {
        #[serde(default)]
        aim: Option<Aim>,
    },
    /// The cannon (B): a pooled heavy ball, or the character's CCD rocket, along `aim`.
    Cannon { aim: Aim },
    /// The T-bone ram (T), now rather than at its scheduled step.
    Ram,
    /// A crater dug in the heightfield at `at` (x, z; none: the scene's next random
    /// spot), its earth heaped on a rim, bodies near it blasted outward.
    Crater {
        #[serde(default)]
        at: Option<[f64; 2]>,
    },
    /// A random tenth of the bodies lifted 10 m, at rest.
    Rain,
    /// Directed debris on or off, and its strength (0 subtle, 1 cinematic, 2 exaggerated).
    Guide {
        #[serde(default)]
        on: Option<bool>,
        #[serde(default)]
        strength: Option<u32>,
    },
    /// The meteor range's carts moving, and steering that knows their routes (the host
    /// resets the scene after).
    Targets {
        #[serde(default)]
        moving: Option<bool>,
        #[serde(default)]
        routes: Option<bool>,
    },
}

impl Action {
    pub fn name(&self) -> &'static str {
        match self {
            Action::Reset => "reset",
            Action::Fire { .. } => "fire",
            Action::Cannon { .. } => "cannon",
            Action::Ram => "ram",
            Action::Crater { .. } => "crater",
            Action::Rain => "rain",
            Action::Guide { .. } => "guide",
            Action::Targets { .. } => "targets",
        }
    }
}

/// Something a director did (the script's own volleys included), for the host's HUD.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Event {
    /// The step it happened before.
    pub step: u64,
    pub what: &'static str,
    /// What it was about (a touchdown's lane and score, a judged near miss…).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

enum Kind {
    /// Scenes with no script and no actions yet.
    None,
    Motion(MotionDirector),
    Soft(SoftDirector),
    Terrain(TerrainDirector),
    Vehicle(VehicleDirector),
}

/// A placed scene's script and the player's actions on it.
pub struct Director {
    kind: Kind,
    events: Vec<Event>,
}

impl Director {
    /// The director of `scene`, just placed (`placed`) in `world`.
    pub fn new(scene: &SceneDef, placed: &Placed, world: &mut World) -> anyhow::Result<Director> {
        let kind = match (scene, placed) {
            (SceneDef::Motion(d), Placed::Motion(p)) => {
                Kind::Motion(MotionDirector::new(d, p.clone()))
            }
            (SceneDef::Soft(d), Placed::Soft(p)) => {
                Kind::Soft(SoftDirector::new(d, p.clone(), world))
            }
            (SceneDef::Terrain(_), Placed::Terrain(p)) => {
                Kind::Terrain(TerrainDirector::new(p.clone(), world))
            }
            (SceneDef::Vehicle(d), Placed::Vehicle(p)) => {
                Kind::Vehicle(VehicleDirector::new(d, p.clone(), world)?)
            }
            _ => Kind::None,
        };
        Ok(Director {
            kind,
            events: Vec::new(),
        })
    }

    /// The scene's own script for the step about to be taken.
    pub fn before_step(&mut self, world: &mut World) -> anyhow::Result<()> {
        let ev = &mut self.events;
        match &mut self.kind {
            Kind::None => Ok(()),
            Kind::Motion(d) => d.before_step(world, ev),
            Kind::Soft(d) => d.before_step(world, ev),
            Kind::Terrain(_) => Ok(()),
            Kind::Vehicle(d) => d.before_step(world, ev),
        }
    }

    /// The frame's scripts before its steps (`dt`: the frame's time in s; `user_car`: the
    /// soft body the player steers now, which its driver leaves alone): the crusher's winch
    /// and the traffic drivers, from what [`Self::after_frame`] read.
    pub fn before_frame(
        &mut self,
        world: &mut World,
        dt: f64,
        user_car: Option<u32>,
    ) -> anyhow::Result<()> {
        let ev = &mut self.events;
        match &mut self.kind {
            Kind::Soft(d) => d.before_frame(world, dt, user_car),
            Kind::Vehicle(d) => d.before_frame(world, ev),
            _ => Ok(()),
        }
    }

    /// After the frame's steps: start the reads the next frame's scripts use (never waits).
    pub fn after_frame(&mut self, world: &World) -> anyhow::Result<()> {
        match &mut self.kind {
            Kind::Soft(d) => d.after_frame(world),
            Kind::Vehicle(d) => d.after_frame(world),
            _ => Ok(()),
        }
    }

    /// For frame-locked hosts (checks): start every read the next [`Self::before_frame`]
    /// needs; wait until [`Self::pending`] is false before calling it.
    pub fn prepare(&mut self, world: &World) -> anyhow::Result<()> {
        match &mut self.kind {
            Kind::Soft(d) => d.prepare(world),
            Kind::Vehicle(d) => d.after_frame(world),
            _ => Ok(()),
        }
    }

    /// Whether a read is in flight (taken by the next [`Self::before_frame`]).
    pub fn pending(&self, world: &World) -> bool {
        match &self.kind {
            Kind::Soft(d) => d.pending(world),
            Kind::Vehicle(d) => d.pending(world),
            _ => false,
        }
    }

    /// The soft body the player drives (the ram aims at it).
    pub fn set_player(&mut self, soft: u32) {
        if let Kind::Soft(d) = &mut self.kind {
            if d.player != soft {
                d.player = soft;
                d.player_read = None;
                d.last_player = None;
            }
        }
    }

    /// Do `action` now (before the next step). An action the scene has no part for fails
    /// naming why.
    pub fn act(&mut self, world: &mut World, action: &Action) -> anyhow::Result<()> {
        let ev = &mut self.events;
        match &mut self.kind {
            Kind::None => anyhow::bail!("this scene has no {} action", action.name()),
            Kind::Motion(d) => d.act(world, action, ev),
            Kind::Soft(d) => d.act(world, action, ev),
            Kind::Terrain(d) => d.act(world, action, ev),
            Kind::Vehicle(d) => d.act(world, action, ev),
        }
    }

    /// Whether the script fires on its own (the gallery's volleys, the range's bursts).
    /// Checks turn it off to fire at steps of their own.
    pub fn set_autofire(&mut self, on: bool) {
        match &mut self.kind {
            Kind::Motion(d) => d.autofire = on,
            Kind::Vehicle(d) => d.autofire = on,
            _ => {}
        }
    }

    /// The script's state for a host's HUD and checks (the vehicle scenes' guidance zone
    /// count; none for the others).
    pub fn state(&self) -> serde_json::Value {
        match &self.kind {
            Kind::Vehicle(d) => serde_json::json!({ "zones": d.zone_count }),
            Kind::Terrain(d) => serde_json::json!({ "craters": d.craters }),
            _ => serde_json::Value::Null,
        }
    }

    /// What happened since the last call, oldest first.
    pub fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }
}

pub(crate) fn note(ev: &mut Vec<Event>, world: &World, what: &'static str) {
    ev.push(Event {
        step: world.step_count(),
        what,
        data: None,
    });
}

pub(crate) fn note_with(
    ev: &mut Vec<Event>,
    world: &World,
    what: &'static str,
    data: serde_json::Value,
) {
    ev.push(Event {
        step: world.step_count(),
        what,
        data: Some(data),
    });
}

fn body(h: u64) -> BodyId {
    BodyId::from_bits(h)
}

/// A body moved to `p` at rest orientation with velocity `v`, awake.
fn launch(world: &mut World, h: u64, p: [f64; 3], v: [f64; 3]) -> anyhow::Result<()> {
    world.body_set_transform(
        body(h),
        Transform::new(p.map(|x| x as f32), [0.0, 0.0, 0.0, 1.0]),
    )?;
    world.body_set_velocity(body(h), v.map(|x| x as f32), [0.0; 3])?;
    world.body_set_awake(body(h), true)
}

// ------------------------------------------------------------------------------- motion

/// `motion.html`'s script: the gallery's volleys, the range's bursts, the playground's
/// walking NPCs, platform and lift, and the character's bullet and rocket.
pub struct MotionDirector {
    scene: String,
    placed: motion::Placed,
    speed: f64,
    pub autofire: bool,
    last_flip: i64,
    rocket_next: usize,
}

/// The playground's fixed step (60 Hz).
const MOTION_DT: f64 = 1.0 / 60.0;

impl MotionDirector {
    fn new(def: &motion::MotionSceneDef, placed: motion::Placed) -> Self {
        MotionDirector {
            scene: def.scene.clone(),
            placed,
            speed: def.speed,
            autofire: def.autofire,
            last_flip: 0,
            rocket_next: 0,
        }
    }

    fn before_step(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let steps = world.step_count() as i64;
        // Demo cadence: a volley every 3 s in the gallery, a burst every 2 s on the range.
        if self.autofire && self.scene == "ccd" && steps % 180 == 20 {
            self.volley(world, ev)?;
        }
        if self.autofire && self.scene == "bullets" && steps % 120 == 20 {
            self.burst(world, ev)?;
        }
        if self.scene != "character" {
            return Ok(());
        }
        // NPCs walk circles.
        let t = steps as f64 * MOTION_DT;
        for n in &self.placed.npcs {
            let ph = n.phase as f64;
            let v = [
                (2.0 * (t * 0.6 + ph).cos()) as f32,
                0.0,
                (2.0 * (t * 0.6 + ph).sin()) as f32,
            ];
            world.character_set_input(CharacterId::from_bits(n.h), v, false)?;
        }
        // The platform back and forth (3 s each way); the lift up and down.
        if steps - self.last_flip >= 180 || steps == 0 || self.last_flip < 0 {
            let dir = if (steps / 180) % 2 != 0 { -1.0 } else { 1.0 };
            if let Some(p) = self.placed.platform {
                world.body_set_velocity(body(p), [1.5 * dir, 0.0, 0.0], [0.0; 3])?;
            }
            if let Some(l) = self.placed.lift {
                world.body_set_velocity(body(l), [0.0, 1.0 * dir, 0.0], [0.0; 3])?;
            }
            self.last_flip = steps;
        }
        Ok(())
    }

    fn volley(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        note(ev, world, "volley");
        for l in &self.placed.lanes {
            let x = l.x as f64;
            launch(
                world,
                l.control,
                [x - 1.0, 1.4, 4.0],
                [0.0, 0.0, self.speed],
            )?;
            launch(world, l.ccd, [x + 1.0, 1.4, 4.0], [0.0, 0.0, self.speed])?;
        }
        if let Some(s) = &self.placed.shower {
            for c in s.control.iter().chain(&s.ccd) {
                launch(
                    world,
                    c.body,
                    [c.x as f64, c.y as f64, c.z as f64],
                    [0.0, -40.0, 0.0],
                )?;
            }
        }
        Ok(())
    }

    fn burst(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        note(ev, world, "burst");
        for k in 0..3 {
            world.spawn_projectile(&ProjectileDef {
                position: [((k as f64 - 1.0) * 1.2) as f32, 0.5, 0.0],
                velocity: [0.0, 0.6, 900.0],
                mass: 0.01,
                penetration_cost: 5000.0,
                max_hits_per_step: 8,
                ..ProjectileDef::default()
            })?;
        }
        Ok(())
    }

    /// The player character's position as of the last polled frame.
    fn player(&self, world: &World) -> anyhow::Result<Option<(CharacterId, [f64; 3])>> {
        let Some(h) = self.placed.player else {
            return Ok(None);
        };
        let id = CharacterId::from_bits(h);
        Ok(world
            .character_state(id)?
            .map(|c| (id, c.position.map(|x| x as f64))))
    }

    fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        match action {
            Action::Reset => {
                note(ev, world, "reset");
                self.reset(world)
            }
            Action::Fire { aim } => match self.scene.as_str() {
                "ccd" => self.volley(world, ev),
                "bullets" => self.burst(world, ev),
                _ => {
                    let aim =
                        aim.ok_or_else(|| anyhow::anyhow!("the character fires along an aim"))?;
                    // Before the character's first frame there is nothing to fire from.
                    let Some((id, c)) = self.player(world)? else {
                        return Ok(());
                    };
                    let v = aim.dir.map(|d| (d * 600.0) as f32);
                    world.spawn_projectile(&ProjectileDef {
                        position: [c[0] as f32, (c[1] + 0.6) as f32, c[2] as f32],
                        velocity: v,
                        mass: 0.012,
                        penetration_cost: 20000.0,
                        max_hits_per_step: 8,
                        ignore_body: world.character_body(id)?,
                        ..ProjectileDef::default()
                    })?;
                    note(ev, world, "bullet");
                    Ok(())
                }
            },
            Action::Cannon { aim } => {
                anyhow::ensure!(
                    self.scene == "character" && !self.placed.rockets.is_empty(),
                    "only the character playground has rockets"
                );
                let Some((_, c)) = self.player(world)? else {
                    return Ok(());
                };
                let d = aim.dir;
                let b = self.placed.rockets[self.rocket_next % self.placed.rockets.len()];
                self.rocket_next += 1;
                note(ev, world, "rocket");
                launch(
                    world,
                    b,
                    [
                        c[0] + d[0] * 1.2,
                        c[1] + 0.8 + d[1] * 1.2,
                        c[2] + d[2] * 1.2,
                    ],
                    d.map(|x| x * 150.0),
                )
            }
            a => anyhow::bail!("motion scenes have no {} action", a.name()),
        }
    }

    fn reset(&mut self, world: &mut World) -> anyhow::Result<()> {
        // Every body and character back to its start, at rest (parked bodies asleep again).
        for r in &self.placed.resets {
            world.body_set_transform(body(r.id), Transform::new(r.position, r.rotation))?;
            world.body_set_velocity(body(r.id), [0.0; 3], [0.0; 3])?;
            if r.asleep {
                world.body_set_awake(body(r.id), false)?;
            }
        }
        for c in &self.placed.characters {
            let id = CharacterId::from_bits(c.h);
            world.character_teleport(id, c.start)?;
            world.character_set_input(id, [0.0; 3], false)?;
        }
        self.last_flip = -1;
        Ok(())
    }
}

// ------------------------------------------------------------------------------ soft

/// `[x, y, z]` rotated by `q` (xyzw), as the pages compute it.
fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let d = 2.0 * (x * v[0] + y * v[1] + z * v[2]);
    let s = w * w - (x * x + y * y + z * z);
    let c = [
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0],
    ];
    [
        x * d + v[0] * s + c[0] * 2.0 * w,
        y * d + v[1] * s + c[1] * 2.0 * w,
        z * d + v[2] * s + c[2] * 2.0 * w,
    ]
}

fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = super::hills::js_hypot(&a);
    let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
    a.map(|x| x / l)
}

fn hypot2(a: f64, b: f64) -> f64 {
    super::hills::js_hypot(&[a, b])
}

/// A body record as the scripts read it (f64, as the pages do).
#[derive(Clone, Copy, Debug)]
struct Pose {
    p: [f64; 3],
    q: [f64; 4],
    v: [f64; 3],
}

impl From<BodyRecordRead> for Pose {
    fn from(r: BodyRecordRead) -> Pose {
        Pose {
            p: r.position.map(|x| x as f64),
            q: r.rotation.map(|x| x as f64),
            v: r.linear_velocity.map(|x| x as f64),
        }
    }
}

/// A traffic car's driver: wander to random points or chase the nearest car, brake for
/// cars ahead (`?ai=cruise`), reverse out when stuck.
struct AiCar {
    /// Its index among the scene's soft bodies.
    i: usize,
    vehicle: VehicleHandle,
    chassis: BodyId,
    chase: bool,
    cruise: bool,
    target: Option<[f64; 2]>,
    t: f64,
    slow: f64,
    reverse: f64,
    kmh: f64,
    seed: u32,
}

impl AiCar {
    /// The page's LCG, in f64 as the page computes it (the product rounds above 2^53).
    fn rand(&mut self) -> f64 {
        let x = self.seed as f64 * 1103515245.0 + 12345.0;
        self.seed = (x % 4294967296.0) as u64 as u32;
        (self.seed >> 8) as f64 / 16777216.0
    }
}

/// The crusher's winch: once its block has rested `lift_at` s, it is lifted (pose and
/// velocity rewritten each frame, at a near-infinite 1e7 kg) `lift_to` m above where it
/// rested, and held there.
struct Winch {
    c: softbody::Crusher,
    start: Option<Pose>,
    read: Option<BodyRead>,
    heavy: bool,
}

/// `softbody.html`'s script: the T-bone ram (launched at its step, or on T), the crusher's
/// lift, the cannon, and the traffic derby's drivers.
pub struct SoftDirector {
    placed: softbody::Placed,
    hz: f64,
    tbone: bool,
    car_speed: f64,
    runup: f64,
    pending_ram: i64,
    cannon_next: usize,
    winch: Option<Winch>,
    ai: Vec<AiCar>,
    ai_read: Option<BodyRead>,
    ai_poses: Option<Vec<Pose>>,
    /// The soft body the player drives (`set_player`), its pose as last read.
    player: u32,
    player_read: Option<BodyRead>,
    last_player: Option<Pose>,
}

/// The car's half width and the gap the ram parks at (m), as placed.
const CAR_HW: f64 = 0.85;

impl SoftDirector {
    fn new(def: &softbody::SoftSceneDef, placed: softbody::Placed, world: &World) -> Self {
        let mut ai = Vec::new();
        if def.scene == "traffic" && def.ai != "0" {
            let cruise = def.ai == "cruise";
            for (i, s) in placed.softs.iter().enumerate() {
                let Some((vehicle, chassis)) = soft_core(world, s.h) else {
                    continue;
                };
                ai.push(AiCar {
                    i,
                    vehicle,
                    chassis,
                    chase: !cruise && i % 3 == 2,
                    cruise,
                    target: None,
                    t: 0.0,
                    slow: 0.0,
                    reverse: 0.0,
                    kmh: 35.0 + ((i * 17) % 45) as f64,
                    seed: (i * 7919 + 13) as u32,
                });
            }
        }
        SoftDirector {
            hz: def.hz,
            tbone: def.scene == "tbone",
            car_speed: def.params.get("carSpeed").copied().unwrap_or(0.0),
            runup: def.runup,
            pending_ram: placed.pending_ram as i64,
            cannon_next: 0,
            winch: placed.crusher.map(|c| Winch {
                c,
                start: None,
                read: None,
                heavy: false,
            }),
            ai,
            ai_read: None,
            ai_poses: None,
            player: placed.softs.first().map(|s| s.h).unwrap_or(0),
            player_read: None,
            last_player: None,
            placed,
        }
    }

    fn before_step(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        if self.pending_ram >= 0 && world.step_count() as i64 >= self.pending_ram {
            self.pending_ram = -1;
            self.ram(world, ev)?;
        }
        Ok(())
    }

    /// The player's body: its core's chassis, or its first node body.
    fn player_body(&self, world: &World) -> Option<(BodyId, bool)> {
        if let Some((_, chassis)) = soft_core(world, self.player) {
            return Some((chassis, true));
        }
        let e = world.sim().soft_host().entries.get(self.player as usize)?;
        world.body_at(e.first_body).map(|b| (b, false))
    }

    /// Take the reads that have landed.
    fn take_reads(&mut self, world: &World) -> anyhow::Result<()> {
        if let Some(r) = &self.ai_read {
            if let Some(v) = r.try_take(world)? {
                self.ai_poses = Some(v.into_iter().map(Pose::from).collect());
                self.ai_read = None;
            }
        }
        if let Some(r) = &self.player_read {
            if let Some(v) = r.try_take(world)? {
                let mut p = Pose::from(v[0]);
                if !self.player_body(world).is_some_and(|b| b.1) {
                    p.q = [0.0, 0.0, 0.0, 1.0];
                }
                self.last_player = Some(p);
                self.player_read = None;
            }
        }
        if let Some(w) = &mut self.winch {
            if let Some(r) = &w.read {
                if let Some(v) = r.try_take(world)? {
                    w.start = Some(Pose::from(v[0]));
                    w.read = None;
                }
            }
        }
        Ok(())
    }

    /// Start the reads the script needs next (the drivers' cars, the player's car), none
    /// twice. Hosts call it after each frame's steps.
    fn after_frame(&mut self, world: &World) -> anyhow::Result<()> {
        if !self.ai.is_empty() && self.ai_read.is_none() {
            let ids: Vec<BodyId> = self.ai.iter().map(|c| c.chassis).collect();
            self.ai_read = Some(world.begin_body_read(&ids)?);
        }
        if self.placed.ram.is_some() && self.player_read.is_none() {
            if let Some((b, _)) = self.player_body(world) {
                self.player_read = Some(world.begin_body_read(&[b])?);
            }
        }
        Ok(())
    }

    /// Reads a frame-locked host waits for before [`Self::before_frame`]: the winch's
    /// rest pose once it is due, and the drivers' cars.
    fn prepare(&mut self, world: &World) -> anyhow::Result<()> {
        self.after_frame(world)?;
        self.winch_read(world)
    }

    /// Whether a read has not landed yet.
    fn pending(&self, world: &World) -> bool {
        let waiting = |r: &Option<BodyRead>| r.as_ref().is_some_and(|r| !r.ready(world));
        waiting(&self.ai_read)
            || waiting(&self.player_read)
            || self.winch.as_ref().is_some_and(|w| waiting(&w.read))
    }

    fn winch_read(&mut self, world: &World) -> anyhow::Result<()> {
        let t = world.step_count() as f64 / self.hz;
        if let Some(w) = &mut self.winch {
            if t >= w.c.lift_at && w.start.is_none() && w.read.is_none() {
                w.read = Some(world.begin_body_read(&[BodyId::from_bits(w.c.body)])?);
            }
        }
        Ok(())
    }

    /// The frame's scripts before its steps: the crusher's winch, then the drivers
    /// (`dt`: the frame's time, s; `user_car`: the soft body the player is steering, if any).
    fn before_frame(
        &mut self,
        world: &mut World,
        dt: f64,
        user_car: Option<u32>,
    ) -> anyhow::Result<()> {
        self.take_reads(world)?;
        self.winch(world)?;
        self.drive(world, dt, user_car)
    }

    fn winch(&mut self, world: &mut World) -> anyhow::Result<()> {
        let t = world.step_count() as f64 / self.hz;
        let Some(w) = &self.winch else {
            return Ok(());
        };
        if t < w.c.lift_at {
            return Ok(());
        }
        if w.start.is_none() {
            return self.winch_read(world);
        }
        let w = self.winch.as_mut().unwrap();
        let start = w.start.unwrap();
        let body = BodyId::from_bits(w.c.body);
        if !w.heavy {
            w.heavy = true;
            world.body_set_mass_data(
                body,
                MassData {
                    mass: 1e7,
                    center: [0.0; 3],
                    rotational_inertia: [[1e7, 0.0, 0.0], [0.0, 1e7, 0.0], [0.0, 0.0, 1e7]],
                },
            )?;
        }
        let rise = w.c.lift_to.min(w.c.speed * (t - w.c.lift_at));
        let moving = rise < w.c.lift_to;
        let p = [start.p[0], start.p[1] + rise, start.p[2]];
        world.body_set_type(
            body,
            BodyType::Dynamic,
            Transform::new(p.map(|x| x as f32), start.q.map(|x| x as f32)),
            [0.0, if moving { w.c.speed as f32 } else { 0.0 }, 0.0],
            [0.0; 3],
        )
    }

    fn drive(&mut self, world: &mut World, dt: f64, user_car: Option<u32>) -> anyhow::Result<()> {
        let Some(poses) = &self.ai_poses else {
            return Ok(());
        };
        let poses = poses.clone();
        for k in 0..self.ai.len() {
            let c = &mut self.ai[k];
            if user_car.is_some_and(|u| self.placed.softs.get(c.i).is_some_and(|s| s.h == u)) {
                continue;
            }
            let me = poses[k];
            let fwd = qrot(me.q, [0.0, 0.0, 1.0]);
            let right = qrot(me.q, [-1.0, 0.0, 0.0]);
            let up = qrot(me.q, [0.0, 1.0, 0.0]);
            let speed = hypot2(me.v[0], me.v[2]);
            c.t += dt;
            if up[1] < 0.3 {
                // On its side or roof.
                world.set_vehicle_input(c.vehicle, VehicleInput::default());
                continue;
            }
            if c.chase {
                let mut best: Option<Pose> = None;
                let mut bd = 1e9;
                for (j, o) in poses.iter().enumerate() {
                    if j != k {
                        let d = hypot2(o.p[0] - me.p[0], o.p[2] - me.p[2]);
                        if d < bd {
                            bd = d;
                            best = Some(*o);
                        }
                    }
                }
                if let Some(b) = best {
                    c.target = Some([b.p[0] + b.v[0] * 0.4, b.p[2] + b.v[2] * 0.4]);
                }
            } else if c.target.is_none()
                || c.t > 9.0
                || c.target
                    .is_some_and(|t| hypot2(t[0] - me.p[0], t[1] - me.p[2]) < 8.0)
            {
                let x = (c.rand() - 0.5) * 120.0;
                let z = (c.rand() - 0.5) * 120.0;
                c.target = Some([x, z]);
                c.t = 0.0;
            }
            let t = c.target.unwrap_or([me.p[0], me.p[2]]);
            let d = [t[0] - me.p[0], t[1] - me.p[2]];
            let ang = (d[0] * right[0] + d[1] * right[2]).atan2(d[0] * fwd[0] + d[1] * fwd[2]);
            let mut steer = (ang * 1.6).clamp(-1.0, 1.0);
            let mut want = c.kmh / 3.6 * if ang.abs() > 1.2 { 0.45 } else { 1.0 };
            if c.cruise {
                // Brake for a car ahead (within 14 m, ±30°), steer off to its far side.
                for (j, o) in poses.iter().enumerate() {
                    if j == k {
                        continue;
                    }
                    let (dx, dz) = (o.p[0] - me.p[0], o.p[2] - me.p[2]);
                    let ahead = dx * fwd[0] + dz * fwd[2];
                    let side = dx * right[0] + dz * right[2];
                    if ahead > 0.0 && ahead < 14.0 && side.abs() < ahead * 0.58 {
                        want = want.min((ahead - 6.0).max(0.0) * 0.6);
                        let sign = if side == 0.0 { 1.0 } else { side.signum() };
                        steer = (steer - sign * 0.6).clamp(-1.0, 1.0);
                    }
                }
            }
            let mut input = VehicleInput {
                forward: if speed < want { 0.85 } else { 0.0 },
                steer: steer as f32,
                brake: if speed > want * 1.3 { 0.4 } else { 0.0 },
                ..VehicleInput::default()
            };
            // Stuck against something: reverse out with the wheel turned the other way.
            c.slow = if speed < 0.8 { c.slow + dt } else { 0.0 };
            if c.slow > 1.6 && c.reverse <= 0.0 {
                c.reverse = 1.4;
                c.slow = 0.0;
            }
            if c.reverse > 0.0 {
                c.reverse -= dt;
                input = VehicleInput {
                    forward: -0.8,
                    steer: (-steer) as f32,
                    ..VehicleInput::default()
                };
            }
            world.set_vehicle_input(c.vehicle, input);
        }
        Ok(())
    }

    /// A 1,500 kg car-height block at the ram's speed into the player's left side.
    fn ram(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let Some(ram) = self.placed.ram else {
            return Ok(());
        };
        let pl = self.last_player.unwrap_or(Pose {
            p: [0.0, 0.6, 0.0],
            q: [0.0, 0.0, 0.0, 1.0],
            v: [0.0; 3],
        });
        let c = pl.p;
        let right = norm3(qrot(pl.q, [1.0, 0.0, 0.0]));
        let fwd = norm3(qrot(pl.q, [0.0, 0.0, 1.0]));
        let speed = self.placed.ram_speed;
        // Aim `ram_at` m forward of the centre, leading a moving car by its travel
        // until the ram arrives.
        let lead = if self.tbone {
            self.car_speed / 3.6
        } else {
            0.0
        };
        let along = self.placed.ram_at + lead * (2.4 + self.runup - CAR_HW - 0.6) / speed;
        let off = 2.4 + self.runup;
        let p = [
            c[0] + right[0] * off + fwd[0] * along,
            0.7,
            c[2] + right[2] * off + fwd[2] * along,
        ];
        note(ev, world, "ram");
        world.body_set_type(
            BodyId::from_bits(ram.body),
            BodyType::Dynamic,
            Transform::new(p.map(|x| x as f32), [0.0, 0.0, 0.0, 1.0]),
            right.map(|d| (-d * speed) as f32),
            [0.0; 3],
        )
    }

    fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        // Aim with the freshest pose that has landed.
        self.take_reads(world)?;
        match action {
            Action::Reset => {
                // The host restores its saved world first; the script starts over.
                note(ev, world, "reset");
                if let Some(w) = &mut self.winch {
                    w.start = None;
                    w.read = None;
                    w.heavy = false;
                }
                if self.tbone {
                    self.pending_ram = 60;
                }
                Ok(())
            }
            Action::Ram => {
                anyhow::ensure!(self.placed.ram.is_some(), "this scene has no ram");
                self.ram(world, ev)
            }
            Action::Cannon { aim } => {
                anyhow::ensure!(!self.placed.cannon.is_empty(), "this scene has no cannon");
                let b = self.placed.cannon[self.cannon_next % self.placed.cannon.len()];
                self.cannon_next += 1;
                let (e, d) = (aim.eye, aim.dir);
                let p = [e[0] + d[0] * 2.0, e[1] + d[1] * 2.0, e[2] + d[2] * 2.0];
                note(ev, world, "cannon");
                world.body_set_type(
                    BodyId::from_bits(b),
                    BodyType::Dynamic,
                    Transform::new(p.map(|x| x as f32), [0.0, 0.0, 0.0, 1.0]),
                    d.map(|x| (x * 30.0) as f32),
                    [0.0; 3],
                )
            }
            a => anyhow::bail!("soft scenes have no {} action", a.name()),
        }
    }
}

/// A soft body's rigid core: its vehicle and chassis body.
fn soft_core(world: &World, soft: u32) -> Option<(VehicleHandle, BodyId)> {
    let (v, chassis) = world.sim().soft_host().entries.get(soft as usize)?.core?;
    Some((v, world.body_at(chassis)?))
}

// ---------------------------------------------------------------------------- terrain

/// `terrain.html`'s actions: craters dug in the heightfield, bodies rained from 10 m, and
/// reset. Random spots continue the scene's sequence (placement's LCG) in order.
pub struct TerrainDirector {
    placed: hills::Placed,
    seed: u32,
    initial: Vec<f32>,
    heights: Vec<f32>,
    pub craters: u32,
}

impl TerrainDirector {
    fn new(placed: hills::Placed, world: &World) -> Self {
        let initial = match world.shape_geometry(ShapeId::from_bits(placed.terrain)) {
            Ok(Geometry::HeightField { field, .. }) => field.heights().to_vec(),
            _ => Vec::new(),
        };
        TerrainDirector {
            seed: placed.seed,
            heights: initial.clone(),
            initial,
            craters: 0,
            placed,
        }
    }

    /// The page's LCG, in f64 as the page computes it.
    fn rand(&mut self) -> f64 {
        let x = self.seed as f64 * 1664525.0 + 1013904223.0;
        self.seed = (x % 4294967296.0) as u64 as u32;
        self.seed as f64 / 4294967296.0
    }

    /// A body's position and rotation as of the last polled frame (its start before).
    fn pose(world: &World, b: &hills::PlacedBody) -> anyhow::Result<([f64; 3], [f32; 4])> {
        let t = world
            .body_transform(BodyId::from_bits(b.id))?
            .unwrap_or(Transform::new(b.p, b.q));
        Ok((t.position.map(|x| x as f64), t.rotation))
    }

    fn upload(&self, world: &mut World) -> anyhow::Result<()> {
        let p = &self.placed;
        world.set_height_field_region(
            ShapeId::from_bits(p.terrain),
            0,
            0,
            p.nx,
            p.nz,
            &self.heights,
        )
    }

    fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        match action {
            Action::Crater { at } => {
                let c = match at {
                    Some(c) => *c,
                    None => {
                        let x = (self.rand() - 0.5) * 24.0;
                        let z = (self.rand() - 0.5) * 24.0;
                        [x, z]
                    }
                };
                note(ev, world, "crater");
                self.crater(world, c)
            }
            Action::Rain => {
                note(ev, world, "rain");
                self.rain(world)
            }
            Action::Reset => {
                note(ev, world, "reset");
                // Every body back to its start, at rest; the ground as placed.
                for b in &self.placed.bodies {
                    let id = BodyId::from_bits(b.id);
                    world.body_set_transform(id, Transform::new(b.p, b.q))?;
                    world.body_set_velocity(id, [0.0; 3], [0.0; 3])?;
                }
                self.heights = self.initial.clone();
                self.upload(world)?;
                self.craters = 0;
                Ok(())
            }
            a => anyhow::bail!("the terrain scene has no {} action", a.name()),
        }
    }

    fn crater(&mut self, world: &mut World, c: [f64; 2]) -> anyhow::Result<()> {
        let p = &self.placed;
        let (nx, nz) = (p.nx as usize, p.nz as usize);
        let (dx, dz) = (p.dx as f64, p.dz as f64);
        let x0 = -0.5 * (nx as f64 - 1.0) * dx;
        let z0 = -0.5 * (nz as f64 - 1.0) * dz;
        let (r_, d_) = (3.0f64, 1.4f64);
        let at = |i: usize, j: usize| {
            hills::js_hypot(&[x0 + i as f64 * dx - c[0], z0 + j as f64 * dz - c[1]])
        };
        // A bowl of depth D inside R; the dug volume heaped onto a rim out to 1.6R.
        let mut dug = 0.0;
        for j in 0..nz {
            for i in 0..nx {
                let r = at(i, j);
                if r < r_ {
                    let d = d_ * (1.0 - (r / r_) * (r / r_));
                    let h = &mut self.heights[j * nx + i];
                    *h = (*h as f64 - d) as f32;
                    dug += d;
                }
            }
        }
        let rim = |r: f64| {
            if r >= r_ && r < 1.6 * r_ {
                (std::f64::consts::PI * (r - r_) / (0.6 * r_)).sin()
            } else {
                0.0
            }
        };
        let mut rim_w = 0.0;
        for j in 0..nz {
            for i in 0..nx {
                rim_w += rim(at(i, j));
            }
        }
        for j in 0..nz {
            for i in 0..nx {
                let w = rim(at(i, j));
                if w > 0.0 {
                    let h = &mut self.heights[j * nx + i];
                    *h = (*h as f64 + dug * w / rim_w) as f32;
                }
            }
        }
        self.upload(world)?;
        // Blast: an outward and upward impulse on bodies within 2R (wakes them).
        for b in &self.placed.bodies {
            let (pos, _) = Self::pose(world, b)?;
            let (ddx, ddz) = (pos[0] - c[0], pos[2] - c[1]);
            let r = hills::js_hypot(&[ddx, ddz]);
            if r > 2.0 * r_ {
                continue;
            }
            let k = 6.0 * (1.0 - r / (2.0 * r_));
            let j = [k * ddx / (r + 0.3), k * 1.2, k * ddz / (r + 0.3)];
            world.body_apply_linear_impulse(
                BodyId::from_bits(b.id),
                j.map(|x| x as f32),
                pos.map(|x| x as f32),
                true,
            )?;
        }
        self.craters += 1;
        Ok(())
    }

    fn rain(&mut self, world: &mut World) -> anyhow::Result<()> {
        let terrain = ShapeId::from_bits(self.placed.terrain);
        let bodies = self.placed.bodies.clone();
        for b in &bodies {
            if self.rand() > 0.1 {
                continue;
            }
            let (p, q) = Self::pose(world, b)?;
            let surface = match world.shape_geometry(terrain)? {
                Geometry::HeightField { field, .. } => field
                    .surface(p[0] as f32, p[2] as f32)
                    .map(|(h, _)| h as f64),
                _ => None,
            };
            let x = p[0] + (self.rand() - 0.5) * 2.0;
            let y = p[1].max(surface.unwrap_or(0.0)) + 10.0;
            let z = p[2] + (self.rand() - 0.5) * 2.0;
            let id = BodyId::from_bits(b.id);
            world.body_set_transform(id, Transform::new([x as f32, y as f32, z as f32], q))?;
            world.body_set_velocity(id, [0.0; 3], [0.0; 3])?;
        }
        Ok(())
    }
}
