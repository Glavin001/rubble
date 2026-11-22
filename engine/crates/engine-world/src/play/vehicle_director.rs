//! `vehicle.html`'s script ([`super::director`]): the gauntlet's shelling, the meteor
//! range's volleys, moving carts and touchdown scoring, the meteor run's director and
//! judge, directed debris (its zones and settings), and the cannon. Its arithmetic is the
//! page's: times in f64 from the solver's f32 step, the scripts' inputs as f32.

use super::director::{note, note_with, Action, Event};
use super::vehicles;
use crate::*;
use engine_vehicle::scripts::{gauntlet, meteors, run};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scene {
    Street,
    Range,
    Run,
    Plain,
}

/// A body record as the page reads it: position, rotation, velocity.
#[derive(Clone, Copy, Debug)]
struct Rec {
    p: [f32; 3],
    q: [f32; 4],
    v: [f32; 3],
}

impl From<BodyRecordRead> for Rec {
    fn from(r: BodyRecordRead) -> Rec {
        Rec {
            p: r.position,
            q: r.rotation,
            v: r.linear_velocity,
        }
    }
}

/// A meteor in the air on the range: its pool index, target, last read height.
struct Flying {
    m: usize,
    lane: u32,
    pad: u32,
    prev: Option<[f64; 3]>,
    prev_step: f64,
    t0: f64,
}

pub struct VehicleDirector {
    scene: Scene,
    placed: vehicles::Placed,
    street: Option<gauntlet::Gauntlet>,
    run: Option<run::MeteorRun>,
    /// The solver's step (s) and gravity, as the page reads them (f32, held as f64).
    dt: f64,
    gravity_y: f32,
    pub autofire: bool,
    guide_on: bool,
    strength: u32,
    moving: bool,
    routes: bool,
    chassis_rest: f32,
    last_shell: u64,
    cannon_next: usize,
    last_chassis: Option<Rec>,
    chassis_read: Option<BodyRead>,
    // The range.
    rest: f64,
    next: usize,
    next_marker: usize,
    volley_k: u32,
    flying: Vec<Flying>,
    landed: VecDeque<(usize, u64)>,
    score: [[u32; 2]; 2],
    // The range's touchdown reads and the run's judge reads share the page's throttle.
    guide_step: i64,
    track_read: Option<BodyRead>,
    judge_read: Option<(BodyRead, Vec<(run::Kind, u32)>, f64)>,
    /// Directed-debris zones as last set.
    pub zone_count: usize,
}

impl VehicleDirector {
    pub(crate) fn new(
        def: &vehicles::VehicleSceneDef,
        placed: vehicles::Placed,
        world: &mut World,
    ) -> anyhow::Result<Self> {
        let scene = match def.scene.as_str() {
            "gauntlet" => Scene::Street,
            "meteors" => Scene::Range,
            "run" => Scene::Run,
            _ => Scene::Plain,
        };
        let solver = world.solver();
        let range = vehicles::RangeLayout::meteors();
        let mut d = VehicleDirector {
            scene,
            street: (scene == Scene::Street).then(gauntlet::Gauntlet::new),
            run: (scene == Scene::Run).then(run::MeteorRun::new),
            dt: solver.dt as f64,
            gravity_y: solver.gravity_y,
            autofire: true,
            guide_on: def.guide,
            strength: def.guide_strength,
            moving: def.targets_moving,
            routes: def.routes,
            chassis_rest: 0.0,
            last_shell: 0,
            cannon_next: 0,
            last_chassis: None,
            chassis_read: None,
            rest: range.radius + 0.05,
            next: 0,
            next_marker: 0,
            volley_k: 0,
            flying: Vec::new(),
            landed: VecDeque::new(),
            score: [[0; 2]; 2],
            guide_step: -1,
            track_read: None,
            judge_read: None,
            zone_count: 0,
            placed,
        };
        // Directed debris armed at load.
        d.apply_guidance(world, true)?;
        Ok(d)
    }

    fn steps(world: &World) -> u64 {
        world.step_count()
    }

    fn chassis_slot(&self, world: &World) -> Option<u32> {
        let h = self.placed.vehicles.first()?.h;
        Some(world.sim().vehicle_host().chassis_body(VehicleHandle(h)))
    }

    fn settings(&self) -> GuidanceSettings {
        GuidanceSettings {
            enabled: self.guide_on,
            ..gauntlet::settings(self.strength)
        }
    }

    /// Directed debris: the zones (the range's pads; the street's car, kerbs and parked
    /// cars) and settings; `rearm`: the carts back to their start and the blocks tagged.
    fn apply_guidance(&mut self, world: &mut World, rearm: bool) -> anyhow::Result<()> {
        match self.scene {
            Scene::Range => {
                if rearm {
                    // Carts start where their motion has them at time 0.
                    for p in &self.placed.pads {
                        let x = meteors::pad_x(p.lane, p.pad, 0.0, self.moving);
                        world.body_set_type(
                            BodyId::from_bits(p.body),
                            BodyType::Kinematic,
                            Transform::new([x, p.y, p.z], [0.0, 0.0, 0.0, 1.0]),
                            [0.0; 3],
                            [0.0; 3],
                        )?;
                    }
                }
                let steered: Vec<u32> = self
                    .placed
                    .pads
                    .iter()
                    .filter(|p| p.lane == 0)
                    .map(|p| p.slot)
                    .collect();
                let t = (Self::steps(world) as f64 * self.dt) as f32;
                let zones = meteors::zones(&steered, self.moving, self.routes, t);
                world.set_guidance(&self.settings())?;
                self.zone_count = zones.len();
                world.set_guidance_zone_descs(&zones)
            }
            Scene::Street | Scene::Run => {
                let Some(player) = self.chassis_slot(world) else {
                    return Ok(());
                };
                let parked: Vec<u32> = self
                    .placed
                    .parked
                    .iter()
                    .map(|&h| world.sim().vehicle_host().chassis_body(VehicleHandle(h)))
                    .collect();
                let y = if self.chassis_rest != 0.0 {
                    self.chassis_rest
                } else {
                    0.8
                };
                let zones = if self.scene == Scene::Run {
                    run::zones(player, y, &parked)
                } else {
                    gauntlet::zones(player, y, &parked)
                };
                world.set_guidance(&self.settings())?;
                self.zone_count = zones.len();
                world.set_guidance_zone_descs(&zones)?;
                if rearm {
                    for b in &self.placed.blocks {
                        world.body_set_guided_tags(BodyId::from_bits(b.id), 1)?;
                    }
                }
                Ok(())
            }
            Scene::Plain => Ok(()),
        }
    }

    fn launch(
        world: &mut World,
        id: u64,
        p: [f32; 3],
        v: [f32; 3],
        w: [f32; 3],
    ) -> anyhow::Result<()> {
        world.body_set_type(
            BodyId::from_bits(id),
            BodyType::Dynamic,
            Transform::new(p, [0.0, 0.0, 0.0, 1.0]),
            v,
            w,
        )
    }

    fn park(world: &mut World, b: &vehicles::ParkedBody) -> anyhow::Result<()> {
        world.body_set_type(
            BodyId::from_bits(b.id),
            BodyType::Static,
            Transform::new(b.home, [0.0, 0.0, 0.0, 1.0]),
            [0.0; 3],
            [0.0; 3],
        )
    }

    fn launch_block(
        &self,
        world: &mut World,
        l: &gauntlet::BlockLaunch,
        street: &gauntlet::Layout,
    ) -> anyhow::Result<()> {
        let (i, v, w) = *l;
        let p = street.blocks[i as usize].p;
        Self::launch(world, self.placed.blocks[i as usize].id, p, v, w)
    }

    /// The range's next volley into both lanes.
    fn volley(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        self.last_shell = Self::steps(world);
        let t = Self::steps(world) as f64 * self.dt;
        // Known routes cover the next few seconds: refresh them for this volley.
        if self.moving && self.routes {
            self.apply_guidance(world, false)?;
        }
        let k = self.volley_k;
        self.volley_k += 1;
        note(ev, world, "volley");
        for l in meteors::volley(k, self.gravity_y, t as f32, self.moving) {
            let m = self.next % self.placed.meteors.len();
            self.next += 1;
            Self::launch(world, self.placed.meteors[m].id, l.p, l.v, [0.0; 3])?;
            world.body_set_guided_tags(BodyId::from_bits(self.placed.meteors[m].id), l.tags)?;
            self.flying.retain(|x| x.m != m);
            self.landed.retain(|x| x.0 != m);
            self.flying.push(Flying {
                m,
                lane: l.lane,
                pad: l.pad,
                prev: None,
                prev_step: 0.0,
                t0: t,
            });
        }
        Ok(())
    }

    /// Blast the facade ahead of the player's car.
    fn shell(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let Some(c) = self.last_chassis else {
            return Ok(());
        };
        self.last_shell = Self::steps(world);
        let street = self.street.as_mut().expect("street scene");
        let (centre, launches) = street.blast_at([c.p[0], c.p[2]], c.v[2]);
        let layout = street.street().clone();
        for l in &launches {
            self.launch_block(world, l, &layout)?;
        }
        let blocks: Vec<u32> = launches.iter().map(|l| l.0).collect();
        note_with(
            ev,
            world,
            "shell",
            serde_json::json!({ "centre": centre, "blocks": blocks }),
        );
        Ok(())
    }

    pub(crate) fn before_step(
        &mut self,
        world: &mut World,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        // The meteor run's director for this step: launch meteors, blast the facades
        // they are about to hit, park spent ones.
        if self.scene == Scene::Run && self.autofire {
            if let Some(c) = self.last_chassis {
                let t = Self::steps(world) as f64 * self.dt;
                let run = self.run.as_mut().expect("run scene");
                let (launches, blasts, parks) =
                    run.step(t as f32, self.dt as f32, c.p, c.v, self.gravity_y);
                let layout = run.street().street().clone();
                for s in &launches {
                    let rock = self.placed.rocks[s.slot as usize];
                    Self::launch(world, rock.id, s.p, s.v, [0.0; 3])?;
                    world.body_set_guided_tags(BodyId::from_bits(rock.id), s.tags)?;
                }
                for (centre, blocks) in &blasts {
                    for l in blocks {
                        self.launch_block(world, l, &layout)?;
                    }
                    let ids: Vec<u32> = blocks.iter().map(|l| l.0).collect();
                    note_with(
                        ev,
                        world,
                        "blast",
                        serde_json::json!({ "centre": centre, "blocks": ids }),
                    );
                }
                for i in parks {
                    Self::park(world, &self.placed.rocks[i as usize])?;
                }
                if !launches.is_empty() {
                    note(ev, world, "meteor");
                }
            }
        }
        // Moving targets: each cart's velocity carries it to where its motion has it a
        // step later (exact, so the carts never drift).
        if self.scene == Scene::Range && self.moving {
            let t = Self::steps(world) as f64 * self.dt;
            for p in &self.placed.pads {
                let a = meteors::pad_x(p.lane, p.pad, (t + self.dt) as f32, true) as f64;
                let b = meteors::pad_x(p.lane, p.pad, t as f32, true) as f64;
                let v = (a - b) / self.dt;
                world.body_set_velocity(
                    BodyId::from_bits(p.body),
                    [v as f32, 0.0, 0.0],
                    [0.0; 3],
                )?;
            }
        }
        Ok(())
    }

    /// Before the frame's steps: the reads that landed, then the volleys and shells due.
    pub(crate) fn before_frame(
        &mut self,
        world: &mut World,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        self.take_reads(world, ev)?;
        let steps = Self::steps(world);
        if self.scene == Scene::Range && self.autofire && steps - self.last_shell >= 120 {
            self.volley(world, ev)?;
        }
        if self.scene == Scene::Street && self.autofire {
            if let Some(c) = self.last_chassis {
                if c.p[2] > -12.0 && c.p[2] < 150.0 && steps - self.last_shell >= 70 {
                    self.shell(world, ev)?;
                }
            }
        }
        Ok(())
    }

    fn take_reads(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        if let Some(r) = &self.chassis_read {
            if let Some(v) = r.try_take(world)? {
                self.last_chassis = Some(v[0].into());
                self.chassis_read = None;
            }
        }
        if let Some(r) = &self.track_read {
            if let Some(v) = r.try_take(world)? {
                self.track_read = None;
                self.track(world, v, ev)?;
            }
        }
        if let Some((r, _, _)) = &self.judge_read {
            if let Some(v) = r.try_take(world)? {
                let (_, watched, t) = self.judge_read.take().unwrap();
                self.judge(world, v, &watched, t, ev)?;
            }
        }
        // The player chassis' height once it has settled: the zones lie on the ground.
        if matches!(self.scene, Scene::Street | Scene::Run)
            && self.chassis_rest == 0.0
            && Self::steps(world) > 90
        {
            if let Some(c) = self.last_chassis {
                self.chassis_rest = c.p[1];
                self.apply_guidance(world, false)?;
            }
        }
        Ok(())
    }

    /// Touchdowns since the last read (each meteor's first crossing of its resting height,
    /// interpolated between reads): marked on the ground and scored.
    fn track(
        &mut self,
        world: &mut World,
        recs: Vec<BodyRecordRead>,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        let now = Self::steps(world);
        let rest = self.rest;
        let moving = self.moving;
        let mut k = 0;
        while k < self.flying.len() {
            let x = &mut self.flying[k];
            let p = recs[x.m].position.map(|v| v as f64);
            match x.prev {
                Some(prev) if prev[1] > rest && p[1] <= rest => {
                    let s = (prev[1] - rest) / (prev[1] - p[1]);
                    let at = [0, 1, 2].map(|i| prev[i] + s * (p[i] - prev[i]));
                    let t = (x.prev_step + s * (now as f64 - x.prev_step)) * self.dt;
                    let (lane, pad, t0, m) = (x.lane, x.pad, x.t0, x.m);
                    let hit = meteors::on_pad(lane, pad, at.map(|v| v as f32), t as f32, moving);
                    self.score[lane as usize][0] += hit as u32;
                    self.score[lane as usize][1] += 1;
                    // Did the cart turn around while this meteor was in the air?
                    let dir = |u: f64| {
                        let d = meteors::pad_x(lane, pad, (u + 0.01) as f32, true) as f64
                            - meteors::pad_x(lane, pad, u as f32, true) as f64;
                        if d > 0.0 {
                            1
                        } else if d < 0.0 {
                            -1
                        } else {
                            0
                        }
                    };
                    let turned = moving && dir(t0) != dir(t);
                    let mk = self.placed.markers[self.next_marker % self.placed.markers.len()];
                    self.next_marker += 1;
                    world.body_set_type(
                        BodyId::from_bits(mk.id),
                        BodyType::Static,
                        Transform::new([at[0] as f32, 0.07, at[2] as f32], [0.0, 0.0, 0.0, 1.0]),
                        [0.0; 3],
                        [0.0; 3],
                    )?;
                    note_with(
                        ev,
                        world,
                        "touchdown",
                        serde_json::json!({ "lane": lane, "pad": pad, "at": at, "hit": hit, "turned": turned,
                            "body": self.placed.meteors[m].slot, "marker": mk.collider, "score": self.score }),
                    );
                    self.flying.remove(k);
                    self.landed.push_back((m, now));
                    continue;
                }
                _ => {
                    x.prev = Some(p);
                    x.prev_step = now as f64;
                }
            }
            k += 1;
        }
        // Park meteors 1.5 s after they come down, so the range stays clear.
        while let Some(&(m, step)) = self.landed.front() {
            if now - step <= 90 {
                break;
            }
            self.landed.pop_front();
            if self.flying.iter().any(|x| x.m == m) {
                continue; // relaunched
            }
            Self::park(world, &self.placed.meteors[m])?;
        }
        Ok(())
    }

    /// The run's judge on what it watches (and the car, from the same read).
    fn judge(
        &mut self,
        world: &World,
        recs: Vec<BodyRecordRead>,
        watched: &[(run::Kind, u32)],
        t: f64,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        let car = Rec::from(recs[0]);
        let bodies: Vec<_> = watched
            .iter()
            .zip(&recs[1..])
            .map(|(&(kind, i), r)| (kind, i, r.position, r.linear_velocity))
            .collect();
        let run = self.run.as_mut().expect("run scene");
        for (e, kind, i, at) in run.judge_at(t as f32, car.p, car.q, car.v, &bodies) {
            note_with(
                ev,
                world,
                "judge",
                serde_json::json!({ "event": e as u32, "meteor": kind == run::Kind::Meteor, "index": i, "at": at, "car": car.p, "t": t }),
            );
        }
        let status = run.status_at(t as f32, car.p[2]);
        note_with(
            ev,
            world,
            "status",
            serde_json::json!({ "t": t, "status": status }),
        );
        Ok(())
    }

    /// After the frame's steps: the chassis read, and the range's or the run's reads when
    /// due (one at a time, as the page throttles them).
    pub(crate) fn after_frame(&mut self, world: &World) -> anyhow::Result<()> {
        let steps = Self::steps(world) as i64;
        if matches!(self.scene, Scene::Street | Scene::Run) && self.chassis_read.is_none() {
            if let Some(s) = self.chassis_slot(world).and_then(|s| world.body_at(s)) {
                self.chassis_read = Some(world.begin_body_read(&[s])?);
            }
        }
        let busy = self.track_read.is_some() || self.judge_read.is_some();
        if self.scene == Scene::Range && !busy && steps - self.guide_step >= 6 {
            self.guide_step = steps;
            if !self.flying.is_empty() {
                let ids: Vec<BodyId> = self
                    .placed
                    .meteors
                    .iter()
                    .map(|m| BodyId::from_bits(m.id))
                    .collect();
                self.track_read = Some(world.begin_body_read(&ids)?);
            }
        }
        if self.scene == Scene::Run
            && !busy
            && self.last_chassis.is_some()
            && steps - self.guide_step >= 4
        {
            self.guide_step = steps;
            let run = self.run.as_ref().expect("run scene");
            let watched = run.watched_list();
            let chassis = self.chassis_slot(world).and_then(|s| world.body_at(s));
            if let Some(chassis) = chassis {
                let mut ids = vec![chassis];
                for &(kind, i) in &watched {
                    let id = match kind {
                        run::Kind::Meteor => self.placed.rocks[i as usize].id,
                        run::Kind::Block => self.placed.blocks[i as usize].id,
                    };
                    ids.push(BodyId::from_bits(id));
                }
                let t = steps as f64 * self.dt;
                self.judge_read = Some((world.begin_body_read(&ids)?, watched, t));
            }
        }
        Ok(())
    }

    pub(crate) fn pending(&self, world: &World) -> bool {
        let waiting = |r: Option<&BodyRead>| r.is_some_and(|r| !r.ready(world));
        waiting(self.chassis_read.as_ref())
            || waiting(self.track_read.as_ref())
            || waiting(self.judge_read.as_ref().map(|j| &j.0))
    }

    pub(crate) fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        // Aim with the freshest chassis read that has landed.
        if !matches!(action, Action::Reset) {
            self.take_reads(world, ev)?;
        }
        match action {
            Action::Fire { .. } => match self.scene {
                Scene::Range => self.volley(world, ev),
                Scene::Street => self.shell(world, ev),
                _ => anyhow::bail!("this vehicle scene has nothing to fire (the cannon is B)"),
            },
            Action::Cannon { aim } => {
                anyhow::ensure!(!self.placed.cannon.is_empty(), "this scene has no cannon");
                let b = self.placed.cannon[self.cannon_next % self.placed.cannon.len()];
                self.cannon_next += 1;
                let (e, d) = (aim.eye, aim.dir);
                let p = [e[0] + d[0] * 3.0, e[1] + d[1] * 3.0, e[2] + d[2] * 3.0];
                note(ev, world, "cannon");
                Self::launch(
                    world,
                    b.id,
                    p.map(|x| x as f32),
                    d.map(|x| (x * 45.0) as f32),
                    [0.0; 3],
                )
            }
            Action::Guide { on, strength } => {
                if let Some(on) = on {
                    self.guide_on = *on;
                }
                if let Some(s) = strength {
                    self.strength = *s;
                }
                self.apply_guidance(world, true)
            }
            Action::Targets { moving, routes } => {
                if let Some(m) = moving {
                    self.moving = *m;
                }
                if let Some(r) = routes {
                    self.routes = *r;
                }
                Ok(())
            }
            Action::Reset => {
                // The host restores its saved world first; the scripts start over.
                note(ev, world, "reset");
                if let Some(s) = &mut self.street {
                    s.reset();
                }
                if let Some(r) = &mut self.run {
                    r.reset();
                    let status = r.status_at(0.0, -25.0);
                    note_with(
                        ev,
                        world,
                        "status",
                        serde_json::json!({ "t": 0.0, "status": status }),
                    );
                }
                self.next = 0;
                self.next_marker = 0;
                self.volley_k = 0;
                self.flying.clear();
                self.landed.clear();
                self.score = [[0; 2]; 2];
                self.track_read = None;
                self.judge_read = None;
                self.apply_guidance(world, true)?;
                self.last_shell = 0;
                Ok(())
            }
            a => anyhow::bail!("vehicle scenes have no {} action", a.name()),
        }
    }
}
