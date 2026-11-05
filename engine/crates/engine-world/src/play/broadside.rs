//! Broadside: the Shardwright galleon stern (`fracture/assets/galleon-stern.glb`) as one
//! `World`, its hull a `Threshold` destructible (support loss, softwood crushing, blast
//! a captain (a scripted walker the camera follows) who walks down the quarterdeck stairs while
//! broadsides from both sides shoot the ship apart around him, and both batteries at once
//! once he stops. Everything is authored here; the page only draws
//! (`lib.rs` hands it the meshes and poses) and picks the playback rate. The ship, its
//! balls and its debris are a [`Range`] (`range.rs`), shared with the timber yard.
//!
//! Frames. The world is the ship's own Shardwright frame (metres, Y up; x runs aft from
//! the cut-away end at 0 to the transom at 22, z across the beam, the main deck at 3.08,
//! the quarterdeck at 5.48): the destructible is placed at minus the scene's placement.
//! Fracture is forward only (AGENTS.md): every ball is a threshold striker whose breaks
//! are predicted at the end of the frame before it lands.

use crate::*;

pub use super::range::*;

/// The galleon stern's levels: 0 its 566 members whole (they part at their joints),
/// 1 each member in about 0.3 m pieces, 2 (the leaf, the default) in grain-stretched
/// cells. The take runs at any of them.
pub const LEVEL: u8 = 2;
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

/// The take's script.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Take {
    /// The captain walks down from the quarterdeck while the barrage destroys the aft
    /// behind him, then both batteries fire around him.
    #[default]
    Walk,
    /// The whole ship: the guns sweep it from the transom to the cut-away end and back,
    /// each ball at a chunk still part of a large piece, until almost none is, then both
    /// batteries rake what is left.
    Raze,
}

/// The galleon on a [`Range`] (its `structure`; `Deref`s to it), with the take's script.
pub struct Broadside {
    pub range: Range,
    pub take: Take,
    /// The ship (the range's structure).
    pub ship: DestructibleId,
    /// Where the walk has got to (m along it); the walker is a scripted point the
    /// barrage keys off and the camera follows (no physical character).
    walked: f32,
    targets: Vec<Target>,
    /// Targets shot so far.
    hit: Vec<bool>,
    path: Vec<[f32; 3]>,
    /// The captain's last reported capsule centre.
    pub captain_at: [f32; 3],
    pub captain_vel: [f32; 3],
    /// The body under him (`NULL` in the air) and its point's velocity.
    pub captain_on: BodyId,
    pub captain_ground_vel: [f32; 3],
    pub captain_ground: bool,
    stopped_at: Option<u32>,
    pub finale: Option<u32>,
    /// The take's script (barrage, walk, finale) runs; off: free play only.
    pub scripted: bool,
    /// Raze's aim and progress.
    pub razing: Razing,
}

impl std::ops::Deref for Broadside {
    type Target = Range;
    fn deref(&self) -> &Range {
        &self.range
    }
}

impl std::ops::DerefMut for Broadside {
    fn deref_mut(&mut self) -> &mut Range {
        &mut self.range
    }
}

impl Broadside {
    /// The ship from its baked GLB, on `gpu`, at the leaf level.
    pub fn new(gpu: GpuContext, glb: &[u8]) -> anyhow::Result<Self> {
        Self::at_level(gpu, glb, LEVEL)
    }

    /// The ship at fracture `level` (see [`LEVEL`]).
    pub fn at_level(gpu: GpuContext, glb: &[u8], level: u8) -> anyhow::Result<Self> {
        Self::with_take(gpu, glb, level, Take::Walk)
    }

    /// The ship at fracture `level`, running `take`'s script.
    pub fn with_take(gpu: GpuContext, glb: &[u8], level: u8, take: Take) -> anyhow::Result<Self> {
        // The ground under the ship's length (x 0..22).
        let range = Range::new(gpu, glb, level, [11.0, 0.0])?;
        let chunks = range.chunks as usize;
        let path = walk();
        let start = path[0];
        let targets = targets();
        let hit = vec![false; targets.len()];
        Ok(Self {
            ship: range.structure,
            range,
            take,
            walked: 0.0,
            targets,
            hit,
            captain_at: [start[0], start[1] + CAPTAIN_HALF + CAPTAIN_RADIUS, start[2]],
            captain_vel: [0.0; 3],
            captain_on: BodyId::NULL,
            captain_ground_vel: [0.0; 3],
            captain_ground: false,
            path,
            stopped_at: None,
            finale: None,
            scripted: true,
            razing: Razing::new(chunks),
        })
    }

    /// The take is over: its closing volley landed and the wreck has had time to fall.
    pub fn over(&self) -> bool {
        self.finale
            .is_some_and(|f| (self.frame - f) as f32 * self.dt > 3.5)
    }

    /// The captain's walk is over (he has stopped on the main deck).
    pub fn captain_stopped(&self) -> bool {
        self.stopped_at.is_some()
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

    /// The scripted part of a step: the barrage, the captain's input along his walk, and
    /// the closing volley once he has stopped and the aft is shot through.
    fn direct(&mut self) -> anyhow::Result<()> {
        if self.take == Take::Raze {
            return self.raze();
        }
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

    /// Raze's guns for one step: each aims at an intact chunk near the sweep (a plane
    /// running the ship's length and back), scattered across it, from the near battery,
    /// plunging into the decks from above. Once almost nothing is intact (or at
    /// [`RAZE_SECONDS`]), both batteries rake what is left.
    fn raze(&mut self) -> anyhow::Result<()> {
        use ship::*;
        if self.finale.is_some() {
            return Ok(());
        }
        if self.razing.finished(&self.range) {
            return self.raze_volley();
        }
        let t = self.seconds();
        let guns = Razing::guns(t);
        // The sweep: transom (x 22) to the cut-away end (x 0) and back.
        let sweep = Razing::sweep(t, 0.0, 22.0);
        let poses = self.chunk_poses();
        let frame = self.frame;
        for gun in 0..guns {
            let h = |c: u32| {
                (c ^ frame.wrapping_mul(97) ^ gun.wrapping_mul(7919)).wrapping_mul(2654435761)
            };
            let pick = self
                .razing
                .pick(&self.range, &poses, gun, |p| (p[0] - sweep).abs());
            let Some(c) = pick else { break };
            let target = poses[c as usize].position;
            // Decks are struck from above; everything else from the near battery, one ball
            // in four from the far one (raking through the hull).
            let deck = target[2].abs() < HALF_BEAM - 0.3
                && [MAIN_DECK, QUARTER_DECK].iter().any(|&y| (target[1] - y).abs() < 0.35);
            let near_side = if target[2] >= 0.0 { 1.0 } else { -1.0 };
            let side = if h(c) % 4 == 0 { -near_side } else { near_side };
            let from = if deck {
                [target[0], target[1] + 14.0, target[2] + side * 10.0]
            } else {
                Self::muzzle(target, side, 0.3 + 1.5 * ((h(c) >> 8) % 3) as f32)
            };
            let d = glam::Vec3::from(target) - glam::Vec3::from(from);
            let t = ((d.length() / BALL_SPEED) / self.dt).ceil().max(1.0) * self.dt;
            let v = self.launch(from, target, t);
            self.spawn_ball(from, v)?;
        }
        Ok(())
    }

    /// Raze's closing volley: both batteries at once into what is still standing (or
    /// across the wreck if nothing is), ripple-fired a frame apart in pairs.
    fn raze_volley(&mut self) -> anyhow::Result<()> {
        self.finale = Some(self.frame);
        self.razing.refresh(&self.range);
        let poses = self.chunk_poses();
        let mut aims: Vec<[f32; 3]> = self
            .razing
            .intact
            .iter()
            .step_by((self.razing.intact.len() / RAZE_VOLLEY).max(1))
            .take(RAZE_VOLLEY)
            .map(|&c| poses[c as usize].position)
            .collect();
        let mut k = 0;
        while aims.len() < RAZE_VOLLEY {
            let x = 1.0 + 20.0 * (k as f32 + 0.5) / RAZE_VOLLEY as f32;
            aims.push([x, ship::MAIN_DECK + 0.4 + 0.8 * (k % 3) as f32, 0.0]);
            k += 1;
        }
        for (i, target) in aims.into_iter().enumerate() {
            let side = if i % 2 == 0 { 1.0 } else { -1.0 };
            let from = Self::muzzle(target, side, 0.3);
            let d = glam::Vec3::from(target) - glam::Vec3::from(from);
            let t = d.length() / BALL_SPEED + (i / 2) as f32 * self.dt;
            let v = self.launch(from, target, t);
            self.spawn_ball(from, v)?;
        }
        Ok(())
    }

    /// One step; its events come back on a later `poll`.
    pub fn step(&mut self) -> anyhow::Result<()> {
        self.range.begin_step()?;
        if self.scripted {
            self.direct()?;
        }
        self.range.end_step()
    }
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
        if b.take == Take::Raze {
            return Self::raze_shot(b);
        }
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

    /// Raze: a slow orbit about the ship from the starboard bow quarter, low then rising,
    /// pulling out wide for the closing volley.
    fn raze_shot(b: &Broadside) -> (glam::Vec3, glam::Vec3) {
        let t = b.seconds();
        let centre = glam::Vec3::new(11.0, 2.5, 0.0);
        let (radius, height) = match b.finale {
            Some(_) => (27.0, 11.0),
            None => (19.0 + 0.1 * t, 5.0 + 0.12 * t),
        };
        let a = -2.3 + 0.07 * t;
        (
            centre + glam::Vec3::new(radius * a.cos(), height, radius * a.sin()),
            centre,
        )
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
        if b.take == Take::Raze {
            return match b.finale {
                Some(f) if (b.frame - f) as f32 * b.dt < 2.0 => 0.2,
                Some(_) => 0.5,
                None if b.frame as f32 * b.dt < 0.4 => 1.0,
                None => 0.45,
            };
        }
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
