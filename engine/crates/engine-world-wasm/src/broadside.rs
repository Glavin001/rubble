//! Broadside: the Shardwright galleon stern (`fracture/assets/galleon-stern.glb`) as one
//! `World`, its hull a `Threshold` destructible (support loss, softwood crushing, blast
//! a captain (a scripted walker the camera follows) who walks down the quarterdeck stairs while
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
pub const MAX_BALLS: usize = 256;
/// The barrage: a gun fires every `FIRE_EVERY` steps, alternately at the nearest unhit
/// point of the aft just behind the captain (the chase) and at the rest of the aft.
pub const FIRE_EVERY: u32 = 1;
/// How far behind him (m along x, he walks towards −x) the chase may aim on his line.
pub const BEHIND: f32 = 0.6;
pub const BEHIND_ON_STAIRS: f32 = 0.6;
/// Nothing else is shot within this horizontal distance (m) of where he will be: the deck
/// he stands on stays whole and its debris does not carry him off.
pub const KEEP_OUT: f32 = 1.2;
/// The closest a ball's line may pass his capsule axis where he will be on impact (m).
pub const CLEARANCE: f32 = 1.0;
/// His line: the half width (m) about his walk where nothing ahead of him is shot.
pub const CORRIDOR: f32 = 1.3;
/// The captain's capsule (radius, half height of its cylinder) and walking speed.
pub const CAPTAIN_RADIUS: f32 = 0.28;
pub const CAPTAIN_HALF: f32 = 0.62;
pub const WALK_SPEED: f32 = 0.9;
/// Seconds the walker stands before setting off.
pub const WALK_START: f32 = 0.5;
/// Loose wreckage lighter than this times his 80 kg does not block the captain.
pub const HERO_PUSH: f32 = 40.0;

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

/// A point of the aft structure the barrage works through.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub at: [f32; 3],
    /// It carries the stairs he walks down (a stringer, the breast): shot only once he is
    /// off them.
    pub carries: bool,
    /// A quarterdeck plank: planks run fore and aft for metres and a ball goes on inside
    /// the deck, so any of them may be the one under his feet; shot once he is down.
    pub floor: bool,
}

/// Every point of the aft the barrage will hit, each once: both stair flights (treads,
/// balusters, handrails, newels, stringers), the balustrade, the breast, the quarterdeck's
/// planks and bulwarks, the mizzen and the wheel post.
pub fn targets() -> Vec<Target> {
    use ship::*;
    let mut t = Vec::new();
    let mut floors = Vec::new();
    let mut add = |x: f32, y: f32, z: f32, carries: bool| {
        t.push(Target {
            at: [x, y, z],
            carries,
            floor: false,
        })
    };
    let (r, g, n) = (0.3f32, 0.3f32, 8);
    for sgn in [1.0f32, -1.0] {
        let (zi, zo) = (sgn * (STAIR_INNER + 0.04), sgn * (STAIR_OUTER - 0.04));
        for k in 1..n {
            let x = BREAST - (n as f32 - 0.5 - k as f32) * g;
            let y = MAIN_DECK + k as f32 * r;
            for z in [0.25f32, 0.5, 0.75] {
                add(
                    x,
                    y - 0.03,
                    sgn * (STAIR_INNER + z * (STAIR_OUTER - STAIR_INNER)),
                    false,
                );
            }
            add(x, y + 0.45, zi, false);
            add(x, y + 0.45, zo, false);
            add(x, y + 0.9, zi, false);
            add(x, y + 0.9, zo, false);
            add(x, y - 0.2, sgn * (STAIR_INNER + 0.03), true);
            add(x, y - 0.2, sgn * (STAIR_OUTER - 0.03), true);
        }
    }
    // Balustrade along the quarterdeck front, rail and balusters.
    let mut z = -HALF_BEAM + 0.15;
    while z < HALF_BEAM {
        add(BREAST + 0.14, QUARTER_DECK + 0.3, z, false);
        add(BREAST + 0.14, QUARTER_DECK + 0.8, z, false);
        z += 0.3;
    }
    // The breast below it.
    let mut z = -HALF_BEAM + 0.2;
    while z < HALF_BEAM {
        // Its boards span the beam and carry the stair heads: once he is off the stairs.
        for y in [
            MAIN_DECK + 0.4,
            MAIN_DECK + 1.0,
            MAIN_DECK + 1.6,
            MAIN_DECK + 2.2,
        ] {
            add(BREAST - 0.04, y, z, true);
        }
        z += 0.5;
    }
    // The quarterdeck: planks, bulwarks, the mizzen, the wheel.
    let mut x = BREAST + 0.4;
    while x < 21.8 {
        let mut z = -HALF_BEAM + 0.3;
        while z < HALF_BEAM {
            floors.push([x, QUARTER_DECK - 0.04, z]);
            z += 0.5;
        }
        for sgn in [1.0f32, -1.0] {
            add(x, QUARTER_DECK + 0.25, sgn * (HALF_BEAM + 0.06), false);
            add(x, QUARTER_DECK + 0.7, sgn * (HALF_BEAM + 0.06), false);
        }
        x += 0.45;
    }
    for y in [0.5f32, 1.2, 2.2] {
        add(17.5, QUARTER_DECK + y, 0.0, false);
    }
    add(19.7, QUARTER_DECK + 0.8, 0.0, false);
    t.extend(floors.into_iter().map(|at| Target {
        at,
        carries: false,
        floor: true,
    }));
    t
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

/// Seconds after the captain stops before both broadsides fire together: at least
/// `FINALE_DELAY`, and once the barrage has shot every point (or `BARRAGE_AFTER`).
pub const FINALE_DELAY: f32 = 1.0;
pub const BARRAGE_AFTER: f32 = 4.0;

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
    /// Where the walk has got to (m along it); the walker is a scripted point the
    /// barrage keys off and the camera follows (no physical character).
    walked: f32,
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
    targets: Vec<Target>,
    /// Targets shot so far.
    hit: Vec<bool>,
    /// The barrage's next pick is the chase (true) or the sweep.
    chase_next: bool,
    path: Vec<[f32; 3]>,
    leg: usize,
    /// The captain's last reported capsule centre.
    pub captain_at: [f32; 3],
    pub captain_vel: [f32; 3],
    /// The body under him (`NULL` in the air) and its point's velocity.
    pub captain_on: BodyId,
    pub captain_ground_vel: [f32; 3],
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
    /// The take's script (barrage, walk, finale) runs; off: free play only.
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
            // One character; its candidate scratch is CHAR_MAX_CANDIDATES records.
            characters: CharacterCapacity {
                max_characters: 4,
                ..Default::default()
            },
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
        let targets = targets();
        let hit = vec![false; targets.len()];
        Ok(Self {
            world,
            ship,
            walked: 0.0,
            dt: solver.dt,
            frame: 0,
            rest,
            chunks,
            fragment,
            color,
            targets,
            hit,
            chase_next: true,
            captain_at: [start[0], start[1] + CAPTAIN_HALF + CAPTAIN_RADIUS, start[2]],
            captain_vel: [0.0; 3],
            captain_on: BodyId::NULL,
            captain_ground_vel: [0.0; 3],
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

    /// A gun's muzzle: out of frame on battery `side` (+1 starboard, −1 port), `drop`
    /// above the target.
    fn muzzle(target: [f32; 3], side: f32, drop: f32) -> [f32; 3] {
        [target[0], target[1] + drop, target[2] + side * 24.0]
    }

    /// Closest approach (m) of a ball's line, from `from` through `target` and on 8 m past
    /// it (balls go through), to the captain's capsule axis where he will be `t` s on.
    fn clearance(&self, from: [f32; 3], target: [f32; 3], t: f32) -> f32 {
        use glam::Vec3;
        let c = Vec3::from(self.captain_at) + Vec3::from(self.captain_vel) * t;
        let (p0, d) = (Vec3::from(from), Vec3::from(target) - Vec3::from(from));
        let p1 = Vec3::from(target) + d.normalize_or_zero() * 8.0;
        let (q0, q1) = (c - Vec3::Y * CAPTAIN_HALF, c + Vec3::Y * CAPTAIN_HALF);
        segment_distance(p0, p1, q0, q1)
    }

    /// Balls half a second old are spent (through the ship or stopped in it): they leave
    /// strike prediction's shot table (64 at most) and roll on as plain rigid balls.
    fn retire_spent(&mut self) -> anyhow::Result<()> {
        let spent = (0.5 / self.dt).round() as u32;
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

    /// The scripted part of a step: the barrage, the captain's input along his walk, and
    /// the closing volley once he has stopped and the aft is shot through.
    fn direct(&mut self) -> anyhow::Result<()> {
        if self.finale.is_none() && self.frame % FIRE_EVERY == 0 {
            self.barrage()?;
        }
        // Walk: along the floor points at walking pace, after a beat.
        if self.seconds() > WALK_START {
            self.walked += WALK_SPEED * self.dt;
        }
        let (at, dir, done) = along(&self.path, self.walked);
        let lift = CAPTAIN_HALF + CAPTAIN_RADIUS;
        self.captain_at = [at[0], at[1] + lift, at[2]];
        self.captain_vel = if done || self.seconds() <= WALK_START {
            [0.0; 3]
        } else {
            (glam::Vec3::from(dir) * WALK_SPEED).to_array()
        };
        self.captain_ground = true;
        if done && self.stopped_at.is_none() {
            self.stopped_at = Some(self.frame);
        }
        if let Some(stop) = self.stopped_at {
            let spent = self.hit.iter().all(|&h| h);
            if self.finale.is_none()
                && self.frame >= stop + (FINALE_DELAY / self.dt).round() as u32
                && (spent || self.frame >= stop + (BARRAGE_AFTER / self.dt).round() as u32)
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
            let from = Self::muzzle(target, side, 0.3);
            let d = glam::Vec3::from(target) - glam::Vec3::from(from);
            // Ripple-fired: a frame apart in pairs.
            let t = d.length() / BALL_SPEED + (i / 2) as f32 * self.dt;
            let c = self.captain_at;
            let d = ((target[0] - c[0]).powi(2) + (target[2] - c[2]).powi(2)).sqrt();
            if d < KEEP_OUT || self.clearance(from, target, t) < CLEARANCE {
                continue;
            }
            let v = self.launch(from, target, t);
            self.spawn_ball(from, v)?;
        }
        Ok(())
    }

    /// One gun of the barrage: the chase (the nearest unhit point behind him on his line)
    /// or the sweep (the next unhit point anywhere else, in a scattered order), from
    /// whichever battery keeps its line clear of him. Nothing ahead of him on his line,
    /// and no stringer while he is on the stairs.
    fn barrage(&mut self) -> anyhow::Result<()> {
        use ship::*;
        // Two guns of three chase him.
        let chase = self.frame % 3 != 2;
        let stopped = self.captain_stopped();
        let c = self.captain_at;
        let on_line = |p: [f32; 3]| (p[2] - c[2]).abs() < CORRIDOR;
        // Down: on the main deck (by height; he may stand anywhere on it).
        let down = c[1] < MAIN_DECK + CAPTAIN_HALF + CAPTAIN_RADIUS + 0.35;
        let lead = glam::Vec3::from(self.captain_vel) * 0.15;
        let at = [c[0] + lead.x, c[2] + lead.z];
        let allowed = |t: &Target| {
            let d = ((t.at[0] - at[0]).powi(2) + (t.at[2] - at[1]).powi(2)).sqrt();
            // On the stairs the chase lags further: what it breaks above him must not
            // free the flight he is still standing on.
            let lag = if !down && c[0] < BREAST + 0.3 {
                BEHIND_ON_STAIRS
            } else {
                BEHIND
            };
            let behind = on_line(t.at) && t.at[0] > c[0] + lag;
            // The deck and what carries the stairs wait until he is off the stairs: balls
            // go through what they hit and on inside the deck, under his feet.
            behind || (!on_line(t.at) && d >= KEEP_OUT)
        };
        let mut order: Vec<usize> = (0..self.targets.len())
            .filter(|&i| !self.hit[i] && allowed(&self.targets[i]))
            .filter(|&i| on_line(self.targets[i].at) == chase || stopped)
            .collect();
        if chase {
            order.sort_by(|&a, &b| self.targets[a].at[0].total_cmp(&self.targets[b].at[0]));
        } else {
            let h = |i: usize| (i as u32).wrapping_mul(2654435761);
            order.sort_by_key(|&i| h(i));
        }
        // What he will still walk on: his flight's width, from the foot of the stairs to
        // where the chase is allowed, from the main deck to just under his feet. Until
        // he is down no ball's line may cross it (they go on through what they hit).
        let feet = c[1] - CAPTAIN_HALF - CAPTAIN_RADIUS;
        let lag = if c[0] < BREAST + 0.3 {
            BEHIND_ON_STAIRS
        } else {
            BEHIND
        };
        let keep = (!down).then(|| {
            (
                glam::Vec3::new(BREAST - 2.5, MAIN_DECK - 0.1, STAIR_INNER - 0.15),
                glam::Vec3::new(c[0], feet + 0.1, STAIR_OUTER + 0.15),
            )
        });
        for i in order {
            let target = self.targets[i].at;
            let near = if target[2] >= 0.0 { 1.0 } else { -1.0 };
            for side in [near, -near] {
                let from = Self::muzzle(target, side, 0.3);
                let d = glam::Vec3::from(target) - glam::Vec3::from(from);
                let t = (((d.length() / BALL_SPEED) / self.dt).ceil().max(1.0)) * self.dt;
                let crosses = keep.is_some_and(|(lo, hi)| {
                    let (p0, d) = (
                        glam::Vec3::from(from),
                        glam::Vec3::from(target) - glam::Vec3::from(from),
                    );
                    let p1 = glam::Vec3::from(target) + d.normalize_or_zero() * 8.0;
                    segment_hits_box(p0, p1, lo, hi)
                });
                if !crosses && self.clearance(from, target, t) >= CLEARANCE {
                    self.hit[i] = true;
                    let v = self.launch(from, target, t);
                    return self.spawn_ball(from, v);
                }
            }
        }
        Ok(())
    }

    /// One step; its events come back on a later `poll`.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.retire_spent()?;
        if self.scripted {
            self.direct()?;
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

/// Closest distance between segments `p0`-`p1` and `q0`-`q1`.
fn segment_distance(p0: glam::Vec3, p1: glam::Vec3, q0: glam::Vec3, q1: glam::Vec3) -> f32 {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    let (s, t) = if a <= 1e-9 && e <= 1e-9 {
        (0.0, 0.0)
    } else if a <= 1e-9 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= 1e-9 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let den = a * e - b * b;
            let mut s = if den > 1e-9 {
                ((b * f - c * e) / den).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    ((p0 + d1 * s) - (q0 + d2 * t)).length()
}

/// Whether segment `p0`-`p1` passes through the box `lo`-`hi` (slabs).
fn segment_hits_box(p0: glam::Vec3, p1: glam::Vec3, lo: glam::Vec3, hi: glam::Vec3) -> bool {
    let d = p1 - p0;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for k in 0..3 {
        if d[k].abs() < 1e-9 {
            if p0[k] < lo[k] || p0[k] > hi[k] {
                return false;
            }
        } else {
            let (a, b) = ((lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

/// The point `s` m along the polyline `path` (floor points), its unit direction, and
/// whether `s` is past its end.
pub fn along(path: &[[f32; 3]], s: f32) -> ([f32; 3], [f32; 3], bool) {
    let mut left = s.max(0.0);
    for w in path.windows(2) {
        let (a, b) = (glam::Vec3::from(w[0]), glam::Vec3::from(w[1]));
        // Walking pace is measured on the ground (horizontal).
        let flat = glam::Vec2::new(b.x - a.x, b.z - a.z).length();
        if left <= flat {
            let f = left / flat.max(1e-6);
            return (
                a.lerp(b, f).to_array(),
                (b - a).normalize_or_zero().to_array(),
                false,
            );
        }
        left -= flat;
    }
    let n = path.len();
    let d = (glam::Vec3::from(path[n - 1]) - glam::Vec3::from(path[n - 2])).normalize_or_zero();
    (path[n - 1], d.to_array(), true)
}
