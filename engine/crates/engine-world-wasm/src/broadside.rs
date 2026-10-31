//! Broadside: the Shardwright galleon stern (`fracture/assets/galleon-stern.glb`) as one
//! `World`, its hull a `Threshold` destructible (support loss, softwood crushing, blast
//! a captain (the GPU character controller) who walks down the quarterdeck stairs while
//! broadsides from both sides shoot the ship apart around him, and both batteries at once
//! once he stops. Everything is authored here; the page only draws
//! (`lib.rs` hands it the meshes and poses) and picks the playback rate.
//!
//! Frames. The world is the ship's own Shardwright frame (metres, Y up; x runs aft from
//! the cut-away end at 0 to the transom at 22, z across the beam, the main deck at 3.08,
//! the quarterdeck at 5.48): the destructible is placed at minus the scene's placement.
//! Fracture is forward only (AGENTS.md): every ball is a threshold striker whose breaks
//! are predicted at the end of the frame before it lands.

use rubble::*;

/// The galleon stern's leaf level: every member split into grain-stretched cells.
pub const LEVEL: u8 = 2;
/// A cannonball: 24 lb of iron (10.9 kg, 0.074 m radius).
pub const BALL_MASS: f32 = 10.9;
pub const BALL_RADIUS: f32 = 0.074;
/// Muzzle velocity: a 24-pounder's ~450 m/s, spent to this at close range.
pub const BALL_SPEED: f32 = 300.0;
/// Balls in flight or at rest the scene keeps (the oldest is removed past this).
pub const MAX_BALLS: usize = 128;
/// Guns firing each cue together (a battery's ripple): the first at the cue's target, the
/// others stepped `SPREAD` m further from the captain along the ship.
pub const GUNS_PER_CUE: usize = 3;
pub const SPREAD: f32 = 0.55;
/// The captain's capsule (radius, half height of its cylinder) and walking speed.
pub const CAPTAIN_RADIUS: f32 = 0.28;
pub const CAPTAIN_HALF: f32 = 0.62;
pub const WALK_SPEED: f32 = 1.1;

/// Ship dimensions the choreography aims at (GalleonStern::default in Shardwright).
pub mod ship {
    pub const MAIN_DECK: f32 = 3.08;
    pub const QUARTER_DECK: f32 = 5.48;
    /// The quarterdeck breast (fore face) and the stair flights' outer |z| and inner |z|.
    pub const BREAST: f32 = 11.92;
    pub const STAIR_INNER: f32 = 1.0;
    pub const STAIR_OUTER: f32 = 2.4;
    /// Inner half beam (inside the strakes).
    pub const HALF_BEAM: f32 = 4.38;
}

/// One authored shot: fired from out of frame on one side so that it reaches `target`
/// `delay` seconds after the captain's capsule passes `trigger_x` going forward (−x), or
/// at `time` seconds when `trigger_x` is None.
#[derive(Clone, Debug)]
pub struct Cue {
    pub trigger_x: Option<f32>,
    pub time: f32,
    pub delay: f32,
    pub target: [f32; 3],
    /// +1 starboard battery (fires towards −z), −1 port (towards +z).
    pub side: f32,
    /// Height the ball comes from above the target (a slight plunge reads better).
    pub drop: f32,
}

/// The take's choreography, in walking order (see the module notes for the frame).
pub fn cues() -> Vec<Cue> {
    use ship::*;
    let rail = QUARTER_DECK + 0.6;
    let at = |x: f32, y: f32, z: f32| [x, y, z];
    let mut c = Vec::new();
    let mut cue =
        |trigger_x: Option<f32>, time: f32, delay: f32, target: [f32; 3], side: f32, drop: f32| {
            c.push(Cue {
                trigger_x,
                time,
                delay,
                target,
                side,
                drop,
            })
        };
    // Opening: the quarterdeck bulwarks either side of him as he sets off.
    cue(
        None,
        0.6,
        0.0,
        at(17.6, QUARTER_DECK + 0.45, HALF_BEAM + 0.06),
        1.0,
        0.4,
    );
    cue(
        None,
        1.1,
        0.0,
        at(16.4, QUARTER_DECK + 0.45, -HALF_BEAM - 0.06),
        -1.0,
        0.4,
    );
    // The mizzen mast behind him, low (it topples on what is left).
    cue(
        Some(15.0),
        0.0,
        0.0,
        at(17.5, QUARTER_DECK + 0.7, 0.0),
        -1.0,
        0.4,
    );
    // The balustrade ahead: the port section, then the middle one beside his line.
    cue(
        Some(14.6),
        0.0,
        0.0,
        at(BREAST + 0.14, rail, -3.3),
        -1.0,
        0.3,
    );
    cue(
        Some(14.0),
        0.0,
        0.0,
        at(BREAST + 0.14, rail, -0.4),
        -1.0,
        0.25,
    );
    cue(
        Some(13.5),
        0.0,
        0.0,
        at(BREAST + 0.14, rail, 0.5),
        1.0,
        0.25,
    );
    // The port flight he does not take.
    cue(
        Some(13.0),
        0.0,
        0.0,
        at(BREAST - 0.8, MAIN_DECK + 1.6, -1.7),
        -1.0,
        0.5,
    );
    // The starboard section of the balustrade, outboard of his flight.
    cue(
        Some(12.6),
        0.0,
        0.0,
        at(BREAST + 0.14, rail, 3.4),
        1.0,
        0.25,
    );
    // On the stairs: the outer handrail beside him, then the treads he has just left.
    cue(
        Some(11.6),
        0.0,
        0.0,
        at(BREAST - 0.45, MAIN_DECK + 2.9, STAIR_OUTER - 0.05),
        1.0,
        0.2,
    );
    cue(
        Some(10.9),
        0.0,
        0.15,
        at(BREAST - 0.3, MAIN_DECK + 2.0, 1.7),
        1.0,
        0.6,
    );
    cue(
        Some(10.4),
        0.0,
        0.1,
        at(BREAST - 1.0, MAIN_DECK + 1.6, 1.7),
        -1.0,
        0.6,
    );
    // Off the stairs: the breast behind him, the main-deck bulwarks, the mainmast.
    cue(
        Some(9.4),
        0.0,
        0.0,
        at(BREAST - 0.04, MAIN_DECK + 1.2, -0.3),
        -1.0,
        0.3,
    );
    cue(
        Some(9.0),
        0.0,
        0.0,
        at(8.6, MAIN_DECK + 0.6, HALF_BEAM + 0.06),
        1.0,
        0.3,
    );
    cue(
        Some(8.6),
        0.0,
        0.0,
        at(5.25, MAIN_DECK + 0.8, 0.0),
        1.0,
        0.4,
    );
    cue(
        Some(8.2),
        0.0,
        0.0,
        at(7.6, MAIN_DECK + 0.6, -HALF_BEAM - 0.06),
        -1.0,
        0.3,
    );
    cue(
        Some(7.8),
        0.0,
        0.2,
        at(10.6, MAIN_DECK + 0.9, -1.7),
        -1.0,
        0.4,
    );
    c
}

/// The captain's walk: floor points (the capsule stands on each), quarterdeck to main deck
/// down the starboard flight.
pub fn walk() -> Vec<[f32; 3]> {
    use ship::*;
    let z = 0.5 * (STAIR_INNER + STAIR_OUTER);
    vec![
        [16.4, QUARTER_DECK, z],
        [12.35, QUARTER_DECK, z],
        [BREAST - 2.4, MAIN_DECK, z],
        [7.4, MAIN_DECK, z],
    ]
}

/// Seconds after the captain stops that both broadsides fire together.
pub const FINALE_DELAY: f32 = 1.0;

/// The closing volley: both batteries at once, raking the ship fore and aft around him
/// (never inside his corridor): bulwarks, the breast, both masts, the quarterdeck.
pub fn finale_targets() -> Vec<([f32; 3], f32)> {
    use ship::*;
    let mut t = Vec::new();
    for (i, x) in [1.4f32, 3.0, 4.4, 9.6, 11.0].into_iter().enumerate() {
        let y = MAIN_DECK + 0.25 + 0.3 * (i % 3) as f32;
        t.push(([x, y, HALF_BEAM + 0.06], 1.0));
        t.push(([x + 0.5, y, -HALF_BEAM - 0.06], -1.0));
    }
    t.push(([5.25, MAIN_DECK + 2.2, 0.0], 1.0));
    t.push(([5.25, MAIN_DECK + 5.0, 0.0], -1.0));
    t.push(([17.5, QUARTER_DECK + 1.2, 0.0], 1.0));
    t.push(([BREAST - 0.04, MAIN_DECK + 0.8, -2.8], -1.0));
    t.push(([BREAST - 0.04, MAIN_DECK + 1.9, -0.6], -1.0));
    t.push(([14.5, QUARTER_DECK + 0.4, HALF_BEAM + 0.06], 1.0));
    t.push(([19.5, QUARTER_DECK + 0.4, -HALF_BEAM - 0.06], -1.0));
    t
}

/// Splinter bodies crushed wood leaves (four per crushed chunk, its mass conserved),
/// reserved when the ship is placed; a take that crushes more fails with an error.
pub const CRUSH_BUDGET: u32 = 4096;

/// Crushing for the ship's wood: the stress import's `timber` (structural softwood, the
/// stress oracle's preset). The iron guns are ductile and never crush.
pub fn crush() -> Crush {
    let p = stress_ref::crush::presets::softwood();
    Crush {
        materials: vec![(
            "timber".into(),
            CrushMaterial {
                cap_pressure: p.cap_pressure,
                cohesion: p.cohesion,
                friction_slope: p.friction_slope,
                crush_energy: p.crush_energy,
                crush_viscosity: p.crush_viscosity,
            },
        )],
        mass: CrushMass::Conserve {
            fragments: 4,
            budget: CRUSH_BUDGET,
        },
        ..Crush::default()
    }
}

/// What a frame did (since the last `take_report`).
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub broken: usize,
    pub crushed: usize,
    pub splits: usize,
    pub shots: usize,
}

pub struct Broadside {
    pub world: World,
    pub ship: DestructibleId,
    pub captain: CharacterId,
    pub dt: f32,
    /// Steps taken.
    pub frame: u32,
    /// Each chunk's rest pose in the render scene (`engine_fracture::build_scene`), the
    /// frame its render vertices are in.
    pub rest: Vec<Transform>,
    pub chunks: u32,
    /// Each chunk's Shardwright fragment and its render colour (its mesh's mean).
    pub fragment: Vec<Option<u32>>,
    pub color: Vec<[f32; 3]>,
    cues: Vec<Cue>,
    /// Frame each cue fires at, once known (its trigger passed).
    due: Vec<Option<u32>>,
    fired: Vec<bool>,
    path: Vec<[f32; 3]>,
    leg: usize,
    /// The captain's last reported capsule centre.
    pub captain_at: [f32; 3],
    pub captain_ground: bool,
    stopped_at: Option<u32>,
    pub finale: Option<u32>,
    /// Balls in the world, oldest first, with the frame each was fired.
    pub balls: Vec<(BodyId, u32)>,
    /// The balls still in strike prediction's shot table, oldest first.
    striking: Vec<(BodyId, u32)>,
    /// Every shot fired: (frame, from, velocity), for the camera and the report.
    pub log: Vec<(u32, [f32; 3], [f32; 3])>,
    pub total: Report,
    /// Free play: shots asked for by the page, fired before the next step.
    pending: Vec<([f32; 3], [f32; 3])>,
    /// The take's script (cues, walk, finale) runs; off: free play only.
    pub scripted: bool,
    /// Chunks crushed so far (hidden: their mass is in `splinters`).
    pub crushed: Vec<bool>,
    /// Crush fragments: body and box half extents; `splinters_version` counts changes.
    pub splinters: Vec<(BodyId, [f32; 3])>,
    pub splinters_version: u32,
}

fn capacity(asset: &DestructibleAssetDesc) -> anyhow::Result<WorldCapacity> {
    let stress = asset
        .stress
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("asset has no stress import"))?;
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("asset has no threshold solver"))?;
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    // Every chunk may become a body, plus the crush splinters, the balls and the captain.
    let bodies = chunks as u32 + CRUSH_BUDGET + MAX_BALLS as u32 + 64;
    c.max_bodies = c.max_bodies.max(bodies.next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    Ok(c)
}

impl Broadside {
    /// The ship from its baked GLB, on `gpu`.
    pub fn new(gpu: GpuContext, glb: &[u8]) -> anyhow::Result<Self> {
        let asset = DestructibleAssetDesc::from_glb(glb, LEVEL)?;
        let parsed = engine_fracture::parse_fracture_glb(glb)?;
        let scene = engine_fracture::build_scene(&parsed, LEVEL, None)?;
        let solver = asset
            .solver_for(StrengthModel::Threshold)
            .ok_or_else(|| anyhow::anyhow!("asset has no threshold solver"))?;
        let def = WorldDef {
            capacity: capacity(&asset)?,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        };
        let mut world = World::new(gpu, &def)?;
        let a = world.create_destructible_asset(&asset)?;
        let placement = asset
            .stress
            .as_ref()
            .map(|s| s.placement())
            .unwrap_or([0.0; 3]);
        let ship = world.create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            transform: Transform::new(placement.map(|x| -x as f32), [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            support_loss: true,
            crush: Some(crush()),
            ..DestructibleDef::new(a)
        })?;
        let chunks = asset.chunks.len() as u32;
        // Render bodies: the ground (0), then one per chunk in order (from_glb's order).
        let f = |v: [f64; 3]| v.map(|x| x as f32);
        let rest: Vec<Transform> = scene
            .bodies
            .iter()
            .skip(1)
            .take(chunks as usize)
            .map(|b| Transform::new(f(b.p), b.q.map(|x| x as f32)))
            .collect();
        let fragment = asset.chunks.iter().map(|c| c.fragment).collect();
        let mut sums = vec![([0.0f32; 3], 0u32); chunks as usize];
        for v in scene.vertices.chunks_exact(10) {
            if let Some(k) = (v[9] as usize)
                .checked_sub(1)
                .filter(|&k| k < chunks as usize)
            {
                for i in 0..3 {
                    sums[k].0[i] += v[6 + i];
                }
                sums[k].1 += 1;
            }
        }
        let color = sums
            .iter()
            .map(|(c, n)| c.map(|x| x / (*n).max(1) as f32))
            .collect();
        let path = walk();
        let start = path[0];
        let captain = world.create_character(&CharacterDef {
            position: [
                start[0],
                start[1] + CAPTAIN_HALF + CAPTAIN_RADIUS + 0.02,
                start[2],
            ],
            desc: CharacterDesc {
                radius: CAPTAIN_RADIUS,
                half_height: CAPTAIN_HALF,
                step_up: 0.35,
                step_down: 0.35,
                ..CharacterDesc::default()
            },
            ..CharacterDef::default()
        })?;
        let cues = cues();
        let n = cues.len();
        Ok(Self {
            world,
            ship,
            captain,
            dt: solver.dt,
            frame: 0,
            rest,
            chunks,
            fragment,
            color,
            due: cues
                .iter()
                .map(|c| {
                    c.trigger_x
                        .is_none()
                        .then(|| (c.time / solver.dt).round() as u32)
                })
                .collect(),
            fired: vec![false; n],
            cues,
            captain_at: [start[0], start[1] + CAPTAIN_HALF + CAPTAIN_RADIUS, start[2]],
            captain_ground: false,
            path,
            leg: 1,
            stopped_at: None,
            finale: None,
            balls: Vec::new(),
            striking: Vec::new(),
            log: Vec::new(),
            total: Report::default(),
            pending: Vec::new(),
            scripted: true,
            crushed: vec![false; chunks as usize],
            splinters: Vec::new(),
            splinters_version: 0,
        })
    }

    pub fn seconds(&self) -> f32 {
        self.frame as f32 * self.dt
    }

    /// The captain's walk is over (he has stopped on the main deck).
    pub fn captain_stopped(&self) -> bool {
        self.stopped_at.is_some()
    }

    /// Fire a ball from `from` at `velocity` before the next step.
    pub fn shoot(&mut self, from: [f32; 3], velocity: [f32; 3]) {
        self.pending.push((from, velocity));
    }

    /// The launch that reaches `target` from `from` in `t` seconds under gravity.
    pub fn launch(&self, from: [f32; 3], target: [f32; 3], t: f32) -> [f32; 3] {
        let g = -9.81f32;
        [
            (target[0] - from[0]) / t,
            (target[1] - from[1]) / t - 0.5 * g * t,
            (target[2] - from[2]) / t,
        ]
    }

    /// A gun's muzzle for a cue: out of frame on its side, `drop` above the target.
    fn muzzle(cue: &Cue) -> [f32; 3] {
        [
            cue.target[0] - 0.6 * cue.side,
            cue.target[1] + cue.drop,
            cue.target[2] + cue.side * 24.0,
        ]
    }

    /// Balls a second old are spent: they leave strike prediction's shot table (64 at
    /// most) and roll on as plain rigid balls.
    fn retire_spent(&mut self) -> anyhow::Result<()> {
        let spent = (1.0 / self.dt).round() as u32;
        while self
            .striking
            .first()
            .is_some_and(|&(_, f)| self.frame >= f + spent)
        {
            let (ball, _) = self.striking.remove(0);
            if self.world.body_is_valid(ball) {
                self.world.body_set_projectile(ball, None)?;
            }
        }
        Ok(())
    }

    fn spawn_ball(&mut self, from: [f32; 3], v: [f32; 3]) -> anyhow::Result<()> {
        if self.balls.len() >= MAX_BALLS {
            let (old, _) = self.balls.remove(0);
            self.striking.retain(|(b, _)| *b != old);
            self.world.destroy_body(old)?;
        }
        let ball = self.world.create_body(&BodyDef {
            linear_velocity: v,
            is_bullet: true,
            ..BodyDef::dynamic(from)
        })?;
        self.world.create_sphere_shape(
            ball,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            BALL_RADIUS,
        )?;
        let i = 0.4 * BALL_MASS * BALL_RADIUS * BALL_RADIUS;
        self.world.body_set_mass_data(
            ball,
            MassData {
                mass: BALL_MASS,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )?;
        self.world
            .body_set_projectile(ball, Some(Projectile::default()))?;
        self.balls.push((ball, self.frame));
        self.striking.push((ball, self.frame));
        self.log.push((self.frame, from, v));
        self.total.shots += 1;
        Ok(())
    }

    /// The scripted part of a step: cues whose triggers passed, the captain's input along
    /// his walk, and the finale once he has stopped.
    fn direct(&mut self) -> anyhow::Result<()> {
        let x = self.captain_at[0];
        for (i, cue) in self.cues.iter().enumerate() {
            if self.due[i].is_none() && cue.trigger_x.is_some_and(|t| x <= t) {
                self.due[i] = Some(self.frame + (cue.delay / self.dt).round() as u32);
            }
        }
        for i in 0..self.cues.len() {
            let Some(impact) = self.due[i] else { continue };
            if self.fired[i] {
                continue;
            }
            let from = Self::muzzle(&self.cues[i]);
            let target = self.cues[i].target;
            let d = glam::Vec3::from(target) - glam::Vec3::from(from);
            let flight_frames = ((d.length() / BALL_SPEED) / self.dt).ceil().max(1.0) as u32;
            // Fire so the ball lands on its impact frame (now, if that is already late).
            if self.frame + flight_frames >= impact {
                let t = flight_frames as f32 * self.dt;
                self.fired[i] = true;
                // Step away from the captain, never towards him or along his way:
                // within 2 m of him the guns step up instead.
                let off = target[0] - self.captain_at[0];
                let step = if off.abs() < 2.0 {
                    [0.0, SPREAD, 0.0]
                } else {
                    [SPREAD * off.signum(), 0.0, 0.0]
                };
                for g in 0..GUNS_PER_CUE {
                    let k = g as f32;
                    let d = [step[0] * k, step[1] * k, 0.0];
                    let f = [from[0] + d[0], from[1] + d[1], from[2]];
                    let tg = [target[0] + d[0], target[1] + d[1], target[2]];
                    let v = self.launch(f, tg, t);
                    self.spawn_ball(f, v)?;
                }
            }
        }
        // Walk: head for the next floor point, horizontally; the controller climbs.
        let mut v = [0.0f32; 3];
        if self.leg < self.path.len() {
            let goal = self.path[self.leg];
            let d = glam::Vec2::new(goal[0] - self.captain_at[0], goal[2] - self.captain_at[2]);
            if d.length() < 0.12 {
                self.leg += 1;
            } else {
                let dir = d.normalize() * WALK_SPEED;
                v = [dir.x, 0.0, dir.y];
            }
        } else if self.stopped_at.is_none() {
            self.stopped_at = Some(self.frame);
        }
        self.world.character_set_input(self.captain, v, false)?;
        if let Some(stop) = self.stopped_at {
            if self.finale.is_none() && self.frame >= stop + (FINALE_DELAY / self.dt).round() as u32
            {
                self.final_volley()?;
            }
        }
        Ok(())
    }

    /// The finale: both broadsides at once ([`finale_targets`]), each ball landing within
    /// a few frames of the others.
    pub fn final_volley(&mut self) -> anyhow::Result<()> {
        self.finale = Some(self.frame);
        for (i, (target, side)) in finale_targets().into_iter().enumerate() {
            let cue = Cue {
                trigger_x: None,
                time: 0.0,
                delay: 0.0,
                target,
                side,
                drop: 0.3,
            };
            let from = Self::muzzle(&cue);
            let d = glam::Vec3::from(target) - glam::Vec3::from(from);
            // Ripple-fired: a frame apart in pairs.
            let t = d.length() / BALL_SPEED + (i / 2) as f32 * self.dt;
            let v = self.launch(from, target, t);
            self.spawn_ball(from, v)?;
        }
        Ok(())
    }

    /// One step; its events come back on a later `poll`.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.retire_spent()?;
        if self.scripted {
            self.direct()?;
        } else {
            self.world
                .character_set_input(self.captain, [0.0; 3], false)?;
        }
        for (from, v) in std::mem::take(&mut self.pending) {
            self.spawn_ball(from, v)?;
        }
        self.world.step()?;
        self.frame += 1;
        Ok(())
    }

    /// Collect the frames the GPU has finished: destruction counts and the captain.
    pub fn poll(&mut self) -> anyhow::Result<Report> {
        let mut r = Report::default();
        for f in self.world.poll()? {
            self.absorb(&f, &mut r);
        }
        Ok(r)
    }

    /// Wait for every step in flight.
    pub fn poll_blocking(&mut self) -> anyhow::Result<Report> {
        let mut r = Report::default();
        for f in self.world.poll_blocking()? {
            self.absorb(&f, &mut r);
        }
        Ok(r)
    }

    fn absorb(&mut self, f: &Frame, r: &mut Report) {
        for e in &f.destruction_events {
            match e {
                DestructionEvent::BondsBroken { bonds, .. } => r.broken += bonds.len(),
                DestructionEvent::ChunkCrushed {
                    chunk, fragments, ..
                } => {
                    r.crushed += 1;
                    if let Some(c) = self.crushed.get_mut(*chunk as usize) {
                        *c = true;
                    }
                    for &f in fragments {
                        let half = self
                            .world
                            .body_shapes(f)
                            .ok()
                            .and_then(|s| s.first().copied())
                            .and_then(|s| match self.world.shape_geometry(s) {
                                Ok(Geometry::Box { half, .. }) => Some(*half),
                                _ => None,
                            });
                        if let Some(half) = half {
                            self.splinters.push((f, half));
                        }
                    }
                    self.splinters_version += 1;
                }
                DestructionEvent::Split { .. } => r.splits += 1,
                _ => {}
            }
        }
        if let Some(c) = f.characters.iter().find(|c| c.character == self.captain) {
            self.captain_at = c.position;
            self.captain_ground = c.ground != GroundState::InAir;
        }
        self.total.broken += r.broken;
        self.total.crushed += r.crushed;
        self.total.splits += r.splits;
        r.broken = 0;
        r.crushed = 0;
        r.splits = 0;
    }

    /// Each chunk's world pose: its body's transform (every body origin is the instance
    /// frame) times its render rest pose. A chunk whose body has no pose yet is at rest.
    pub fn chunk_poses(&self) -> Vec<Transform> {
        (0..self.chunks)
            .map(|c| {
                let body = self.world.destructible_chunk_body(self.ship, c).ok();
                let t = body
                    .and_then(|b| self.world.body_transform(b).ok().flatten())
                    .unwrap_or(Transform::IDENTITY);
                compose(&t, &self.rest[c as usize])
            })
            .collect()
    }

    /// Every ball's pose, oldest first.
    pub fn ball_poses(&self) -> Vec<Transform> {
        self.balls
            .iter()
            .filter_map(|(b, _)| self.world.body_transform(*b).ok().flatten())
            .collect()
    }
}

pub fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (
        glam::Quat::from_array(a.rotation),
        glam::Vec3::from(a.position),
    );
    let (qb, pb) = (
        glam::Quat::from_array(b.rotation),
        glam::Vec3::from(b.position),
    );
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

/// The take's camera and playback rate, authored with the choreography: a low shot ahead
/// of the captain looking up at him as he crosses the quarterdeck and comes down the
/// stairs, closer on the main deck, then a wide shot for the closing volley. Render only.
#[derive(Clone, Debug)]
pub struct Director {
    eye: glam::Vec3,
    target: glam::Vec3,
    started: bool,
}

impl Default for Director {
    fn default() -> Self {
        Self {
            eye: glam::Vec3::ZERO,
            target: glam::Vec3::ZERO,
            started: false,
        }
    }
}

impl Director {
    /// Where the shot wants the camera now (eye, target), before smoothing.
    fn shot(b: &Broadside) -> (glam::Vec3, glam::Vec3) {
        use ship::*;
        let c = glam::Vec3::from(b.captain_at);
        let wide = (
            glam::Vec3::new(-7.5, 10.5, -15.0),
            glam::Vec3::new(9.5, 3.0, 0.0),
        );
        if let Some(f) = b.finale {
            let since = (b.frame - f) as f32 * b.dt;
            // Hold on the captain a beat as the volley lands, then pull out wide.
            return if since < 0.35 {
                (
                    glam::Vec3::new(c.x - 5.5, MAIN_DECK + 1.9, -1.9),
                    c + glam::Vec3::new(0.0, 0.3, 0.0),
                )
            } else {
                wide
            };
        }
        if b.captain_stopped() {
            return (
                glam::Vec3::new(c.x - 5.5, MAIN_DECK + 1.9, -1.9),
                c + glam::Vec3::new(0.0, 0.3, 0.0),
            );
        }
        if c.x > BREAST + 0.4 {
            // Quarterdeck: from the main deck ahead, below him, looking up past the rail.
            (
                glam::Vec3::new(c.x - 9.5, MAIN_DECK + 2.5, -0.9),
                c + glam::Vec3::new(0.0, 0.25, 0.0),
            )
        } else {
            // The stairs and on: low on the main deck, close, looking up at him.
            (
                glam::Vec3::new(c.x - 6.2, MAIN_DECK + 1.6, -1.4),
                c + glam::Vec3::new(0.0, 0.2, 0.0),
            )
        }
    }

    /// The camera for a frame `wall` seconds after the last one (eye, target): the shot,
    /// eased so cuts glide.
    pub fn camera(&mut self, b: &Broadside, wall: f32) -> ([f32; 3], [f32; 3]) {
        let (eye, target) = Self::shot(b);
        if !self.started {
            self.eye = eye;
            self.target = target;
            self.started = true;
        }
        let k = 1.0 - (-wall * 3.0).exp();
        self.eye = self.eye.lerp(eye, k);
        self.target = self.target.lerp(target, k);
        (self.eye.to_array(), self.target.to_array())
    }

    /// Simulated seconds per wall second: slow motion through the walk, slower as the
    /// closing volley lands, then a little faster as the wreck settles.
    pub fn time_scale(b: &Broadside) -> f32 {
        match b.finale {
            Some(f) if (b.frame - f) as f32 * b.dt < 2.0 => 0.18,
            Some(_) => 0.45,
            None if b.frame as f32 * b.dt < 0.4 => 1.0,
            None => 0.32,
        }
    }
}
