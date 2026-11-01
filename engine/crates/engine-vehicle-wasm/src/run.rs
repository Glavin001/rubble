//! "Meteor gauntlet": the game. Drive the length of a city street under a
//! meteor shower. Most meteors slam into the facades ahead of you and blast
//! their masonry into the street; the rest come down on the road around you.
//! Directed debris (`engine_schema::guidance`) steers the thrown blocks to
//! land around the car, on the parked cars and behind it, and the falling
//! meteors to land close beside it: the street rains down all around you
//! instead of on you or nowhere near. Turn it off and the same shower throws
//! its debris where the blasts send it.
//!
//! The page (vehicle.html `?scene=run`) builds the street from `layout`,
//! carries out `tick`'s commands (launch a meteor, blast a facade, park a
//! spent meteor), reads back the bodies `watched` lists for `judge`, and
//! shows `status`.
//!
//! Scoring: a block or meteor that hits the car costs health (a meteor most
//! of it); one that lands close beside it scores a near miss. Reach the end of
//! the street to bank the score, plus bonuses for health and time.

use crate::gauntlet::{self, hash, BlockLaunch, Gauntlet, BLOCK_HALF, STREET_HALF};
use engine_schema::guidance::{ZoneDesc, ZoneShape};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Building spans along the street (z from, z to), both sides: 310 m.
pub const SPANS: [(f32, f32); 8] = [
    (0.0, 30.0),
    (40.0, 70.0),
    (80.0, 110.0),
    (120.0, 150.0),
    (160.0, 190.0),
    (200.0, 230.0),
    (240.0, 270.0),
    (280.0, 310.0),
];
/// Parked cars at the kerbs (x, z, yaw).
pub const PARKED: [(f32, f32, f32); 10] = [
    (-5.6, 22.0, 0.0),
    (5.6, 48.0, std::f32::consts::PI),
    (-5.6, 75.0, 0.0),
    (5.6, 96.0, std::f32::consts::PI),
    (-5.6, 128.0, 0.0),
    (5.6, 165.0, std::f32::consts::PI),
    (-5.6, 205.0, 0.0),
    (5.6, 238.0, std::f32::consts::PI),
    (-5.6, 262.0, 0.0),
    (5.6, 296.0, std::f32::consts::PI),
];
/// The finish line (z): the end of the street.
pub const FINISH: f32 = 315.0;
/// Meteor radius (m), mass (kg) and how many the page keeps parked for launch.
pub const METEOR_RADIUS: f32 = 0.8;
pub const METEOR_MASS: f32 = 2500.0;
pub const METEORS: u32 = 12;
/// A meteor's blast on a facade: radius (m) and the fastest block (m/s).
const BLAST_RADIUS: f32 = 2.0;
const BLAST_SPEED: f32 = 8.5;
/// Flight times (s): onto a facade, onto the road.
const FACADE_FLIGHT: f32 = 2.2;
const ROAD_FLIGHT: f32 = 2.6;
/// The car's hit box about its chassis centre, chassis frame (m): the monster
/// truck with its wheels.
const CAR_HALF: [f32; 3] = [1.6, 1.4, 2.9];
/// A landing this close beside the car (m, ground distance from its centre)
/// is a near miss: blocks, meteors.
const NEAR_BLOCK: f32 = 7.0;
const NEAR_METEOR: f32 = 10.0;
/// Health lost to a hit: a block (plus per m/s of closing speed, capped), a meteor.
const BLOCK_DAMAGE: (f32, f32, f32) = (2.5, 0.3, 8.0);
const METEOR_DAMAGE: f32 = 35.0;
/// Points: a block's near miss, a meteor's; at the finish, per health point
/// left and per second under PAR.
const NEAR_POINTS: u32 = 25;
const CLOSE_POINTS: u32 = 150;
const HEALTH_POINTS: u32 = 10;
const PAR: f32 = 60.0;
const TIME_POINTS: f32 = 20.0;
/// Tags: blocks are steered toward the street's zones (tag 1), road meteors
/// toward the ring beside the car (tag 2).
pub const BLOCK_TAGS: u32 = 1;
pub const METEOR_TAGS: u32 = 2;

/// The street.
pub fn layout() -> gauntlet::Layout {
    gauntlet::layout_of(&SPANS, &PARKED)
}

/// The zones: the Gauntlet's (around the car, behind it, the parked cars'
/// roofs; tag 1), the ring around the car set a little wider (a block steered
/// beside the car tumbles on), and, for the meteors (tag 2), an arc either
/// side of the car:
/// close enough to feel, and beside it rather than ahead, where it would
/// drive into the meteor (vehicle frame: angle 0 ahead, π behind).
pub fn zones(player: u32, chassis_y: f32, parked: &[u32]) -> Vec<ZoneDesc> {
    use std::f32::consts::PI;
    let mut z = gauntlet::zones(player, chassis_y, parked);
    if let ZoneShape::Ring { inner, outer, .. } = &mut z[0].shape {
        (*inner, *outer) = (3.6, 7.5);
    }
    for (from, to) in [(0.3 * PI, 0.7 * PI), (1.3 * PI, 1.7 * PI)] {
        z.push(ZoneDesc {
            anchor_body: Some(player),
            anchor_relative: true,
            offset: [0.0, METEOR_RADIUS - chassis_y, 0.0],
            shape: ZoneShape::Arc {
                inner: 4.0,
                outer: 8.0,
                height: 0.0,
                from,
                to,
            },
            tags: METEOR_TAGS,
            ..ZoneDesc::default()
        });
    }
    z
}

/// Seconds between meteors `elapsed` seconds into the run: faster as you last.
pub fn interval(elapsed: f32) -> f32 {
    (1.7 - 0.025 * elapsed).max(0.7)
}

/// One meteor launch: pool slot, launch state, the tags steering it (0: not
/// steered), where it is aimed and when it gets there.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Strike {
    pub slot: u32,
    pub p: [f32; 3],
    pub v: [f32; 3],
    pub tags: u32,
    pub facade: bool,
    pub target: [f32; 3],
    pub arrive: f32,
}

/// A step's commands: meteors to launch, facades to blast (centre and its
/// blocks' launches), meteor slots to park.
pub type Commands = (Vec<Strike>, Vec<([f32; 3], Vec<BlockLaunch>)>, Vec<u32>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Block,
    Meteor,
}

#[derive(Clone, Copy, Debug)]
struct Watch {
    kind: Kind,
    index: u32,
    since: f32,
    airborne: bool,
}

/// What `judge` saw happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    NearMiss = 1,
    Hit = 2,
    CloseCall = 3,
    MeteorHit = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum State {
    Ready,
    Running,
    Wrecked,
    Finished,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub state: State,
    pub health: f32,
    pub score: u32,
    pub near: u32,
    pub close: u32,
    pub hits: u32,
    pub meteor_hits: u32,
    pub meteors: u32,
    pub elapsed: f32,
    pub distance: f32,
    pub finish: f32,
    /// At the finish: the bonuses and the total.
    pub health_bonus: u32,
    pub time_bonus: u32,
    pub total: u32,
}

/// The run: the street's state, the meteor director and the score.
#[wasm_bindgen]
pub struct MeteorRun {
    street: Gauntlet,
    state: State,
    start: f32,
    end: f32,
    next: f32,
    strikes: u32,
    pending: Vec<Strike>,
    watches: Vec<Watch>,
    parks: Vec<(u32, f32)>,
    health: f32,
    score: u32,
    near: u32,
    close: u32,
    hits: u32,
    meteor_hits: u32,
}

impl Default for MeteorRun {
    fn default() -> Self {
        Self::new()
    }
}

/// `q`⁻¹ applied to `v` (unit quaternion x, y, z, w).
fn unrotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z, w] = [-q[0], -q[1], -q[2], q[3]];
    let d = 2.0 * (x * v[0] + y * v[1] + z * v[2]);
    let s = w * w - (x * x + y * y + z * z);
    let c = [
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0],
    ];
    [0, 1, 2].map(|k| [x, y, z][k] * d + v[k] * s + c[k] * 2.0 * w)
}

impl MeteorRun {
    /// Plan strike `k` at time `t` for a car at `car` (x, z) moving at `vz`.
    pub fn plan(&self, k: u32, t: f32, car: [f32; 2], vz: f32, gravity_y: f32) -> Strike {
        let lead = vz.max(0.0);
        let h = |n: u32| hash(k * 7 + n + 101);
        // Two in three onto a facade, the third onto the road.
        let facade = k % 3 != 2;
        let flight = if facade { FACADE_FLIGHT } else { ROAD_FLIGHT };
        let (target, p) = if facade {
            let side = if h(0) < 0.5 { 1.0 } else { -1.0 };
            let y = 3.0 + 6.0 * h(1);
            // Where the car will be when the blast's blocks come down (their
            // flight from that height at the blast's lift), give or take 4 m.
            let lift = 0.35 * BLAST_SPEED;
            let fall = (lift + (lift * lift + 2.0 * 9.81 * y).sqrt()) / 9.81;
            let want = car[1] + lead * (flight + fall) + 8.0 * (h(2) - 0.5);
            let z = SPANS
                .iter()
                .map(|&(a, b)| want.clamp(a + 2.5, b - 2.5))
                .min_by(|a, b| (a - want).abs().total_cmp(&(b - want).abs()))
                .unwrap();
            // The meteor's centre touches the facade's street face.
            let target = [side * (STREET_HALF - METEOR_RADIUS), y, z];
            // Down across the street from far ahead, low enough to cross the
            // chase camera's sky on the way in.
            (target, [-side * 30.0, 34.0, z + 48.0 + 8.0 * h(3)])
        } else {
            // On the road where the car is going, give or take a car length.
            let target = [
                (car[0] + 5.0 * (h(4) - 0.5)).clamp(-STREET_HALF + 2.0, STREET_HALF - 2.0),
                METEOR_RADIUS,
                car[1] + lead * flight + 6.0 * (h(5) - 0.4),
            ];
            // In from ahead, across the camera's sky.
            (
                target,
                [
                    target[0] + 14.0 * (h(6) - 0.5),
                    34.0,
                    target[2] + 48.0 + 8.0 * h(7),
                ],
            )
        };
        let v = [0, 1, 2].map(|a| {
            let g = if a == 1 { gravity_y } else { 0.0 };
            (target[a] - p[a] - 0.5 * g * flight * flight) / flight
        });
        Strike {
            slot: k % METEORS,
            p,
            v,
            tags: if facade { 0 } else { METEOR_TAGS },
            facade,
            target,
            arrive: t + flight,
        }
    }

    fn watch(&mut self, kind: Kind, index: u32, t: f32) {
        self.watches
            .retain(|w| !(w.kind == kind && w.index == index));
        self.watches.push(Watch {
            kind,
            index,
            since: t,
            airborne: false,
        });
    }

    fn running(&self) -> bool {
        self.state == State::Running
    }

    /// Advance the director to time `t` (s) for a car at `car` (x, y, z) moving
    /// at `car_v`: the meteors to launch, the facades to blast and the spent
    /// meteors to park now.
    pub fn step(
        &mut self,
        t: f32,
        dt: f32,
        car: [f32; 3],
        car_v: [f32; 3],
        gravity_y: f32,
    ) -> Commands {
        if self.state == State::Ready && car[2] > gauntlet::SPAWN[2] + 4.0 {
            self.state = State::Running;
            self.start = t;
            self.next = t + 0.6;
        }
        if self.running() && car[2] >= FINISH {
            self.state = State::Finished;
            self.end = t;
        }
        let mut launch = Vec::new();
        if self.running() && t >= self.next && car[2] < FINISH - 20.0 {
            let s = self.plan(self.strikes, t, [car[0], car[2]], car_v[2], gravity_y);
            self.strikes += 1;
            self.next = t + interval(t - self.start);
            // The slot's previous meteor is relaunched: stop watching it.
            self.parks.retain(|&(slot, _)| slot != s.slot);
            self.pending.retain(|p| p.slot != s.slot);
            self.watch(Kind::Meteor, s.slot, t);
            if s.facade {
                self.pending.push(s);
            }
            launch.push(s);
        }
        // Blast each facade a step before its meteor arrives, so the blocks are
        // already flying as it hits; the meteor ends there, buried in the
        // building (parked now), so the blast's masonry is what reaches the
        // street rather than a 2.5 t ball carrying on down it.
        let mut blasts = Vec::new();
        let due: Vec<Strike> = self
            .pending
            .iter()
            .copied()
            .filter(|s| s.arrive <= t + dt)
            .collect();
        self.pending.retain(|s| s.arrive > t + dt);
        for s in due {
            let centre = [s.target[0].signum() * STREET_HALF, s.target[1], s.target[2]];
            let k = self.strikes * 31 + s.slot;
            let list = self
                .street
                .blast_patch(centre, k, BLAST_RADIUS, BLAST_SPEED);
            for &(i, ..) in &list {
                self.watch(Kind::Block, i, t);
            }
            blasts.push((centre, list));
            self.watches
                .retain(|w| !(w.kind == Kind::Meteor && w.index == s.slot));
            self.parks.push((s.slot, t));
        }
        let mut park = Vec::new();
        self.parks.retain(|&(slot, at)| {
            if at <= t {
                park.push(slot);
                false
            } else {
                true
            }
        });
        (launch, blasts, park)
    }

    /// The bodies to read back for `judge`: `(kind, index)` with kind 0 a
    /// facade block, 1 a meteor's pool slot.
    pub fn watched_list(&self) -> Vec<(Kind, u32)> {
        self.watches.iter().map(|w| (w.kind, w.index)).collect()
    }

    /// Judge the watched bodies at time `t`: `bodies` is (kind, index,
    /// position, velocity) each; the car is its chassis centre, orientation and
    /// velocity. Returns what happened, where.
    pub fn judge_at(
        &mut self,
        t: f32,
        car_p: [f32; 3],
        car_q: [f32; 4],
        car_v: [f32; 3],
        bodies: &[(Kind, u32, [f32; 3], [f32; 3])],
    ) -> Vec<(Event, Kind, u32, [f32; 3])> {
        let mut out = Vec::new();
        let mut done = Vec::new();
        for &(kind, index, p, v) in bodies {
            let Some(w) = self
                .watches
                .iter_mut()
                .find(|w| w.kind == kind && w.index == index)
            else {
                continue;
            };
            let meteor = kind == Kind::Meteor;
            let r = if meteor { METEOR_RADIUS } else { BLOCK_HALF[1] };
            let d = unrotate(car_q, [0, 1, 2].map(|k| p[k] - car_p[k]));
            let closing = [0, 1, 2].map(|k| v[k] - car_v[k]);
            let closing =
                (closing[0] * closing[0] + closing[1] * closing[1] + closing[2] * closing[2])
                    .sqrt();
            let inside = (0..3).all(|k| d[k].abs() <= CAR_HALF[k] + r);
            let mut end = None;
            if inside && closing > 2.5 && self.state != State::Wrecked {
                if meteor {
                    self.health -= METEOR_DAMAGE;
                    self.meteor_hits += 1;
                    end = Some(Event::MeteorHit);
                } else {
                    let (base, per, cap) = BLOCK_DAMAGE;
                    self.health -= (base + per * closing).min(cap);
                    self.hits += 1;
                    end = Some(Event::Hit);
                }
            } else {
                if p[1] > if meteor { 2.0 } else { 1.4 } {
                    w.airborne = true;
                }
                let down = if meteor { METEOR_RADIUS + 0.25 } else { 0.8 };
                if w.airborne && p[1] < down {
                    let ground = ((p[0] - car_p[0]).powi(2) + (p[2] - car_p[2]).powi(2)).sqrt();
                    if meteor && ground < NEAR_METEOR {
                        self.close += 1;
                        self.score += CLOSE_POINTS;
                        end = Some(Event::CloseCall);
                    } else if !meteor && ground < NEAR_BLOCK {
                        self.near += 1;
                        self.score += NEAR_POINTS;
                        end = Some(Event::NearMiss);
                    } else {
                        done.push((kind, index));
                    }
                }
            }
            if let Some(e) = end {
                // Only a run in progress scores; a finished one keeps its score.
                if self.running() {
                    out.push((e, kind, index, p));
                }
                done.push((kind, index));
            } else if t - w.since > 6.0 {
                done.push((kind, index));
            }
        }
        if self.health <= 0.0 && self.running() {
            self.health = 0.0;
            self.state = State::Wrecked;
            self.end = t;
        }
        for (kind, index) in done {
            self.watches
                .retain(|w| !(w.kind == kind && w.index == index));
            if kind == Kind::Meteor {
                // Clear the road a while after it comes down.
                self.parks.push((index, t + 2.5));
            }
        }
        out
    }

    pub fn status_at(&self, t: f32, car_z: f32) -> Status {
        let elapsed = match self.state {
            State::Ready => 0.0,
            State::Running => t - self.start,
            _ => self.end - self.start,
        };
        let finished = self.state == State::Finished;
        let health_bonus = if finished {
            self.health.max(0.0) as u32 * HEALTH_POINTS
        } else {
            0
        };
        let time_bonus = if finished {
            ((PAR - elapsed).max(0.0) * TIME_POINTS) as u32
        } else {
            0
        };
        Status {
            state: self.state,
            health: self.health.max(0.0),
            score: self.score,
            near: self.near,
            close: self.close,
            hits: self.hits,
            meteor_hits: self.meteor_hits,
            meteors: self.strikes,
            elapsed,
            distance: (car_z - gauntlet::SPAWN[2]).max(0.0),
            finish: FINISH - gauntlet::SPAWN[2],
            health_bonus,
            time_bonus,
            total: self.score + health_bonus + time_bonus,
        }
    }
}

#[wasm_bindgen]
impl MeteorRun {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MeteorRun {
        MeteorRun {
            street: Gauntlet::of(&SPANS, &PARKED),
            state: State::Ready,
            start: 0.0,
            end: 0.0,
            next: 0.0,
            strikes: 0,
            pending: Vec::new(),
            watches: Vec::new(),
            parks: Vec::new(),
            health: 100.0,
            score: 0,
            near: 0,
            close: 0,
            hits: 0,
            meteor_hits: 0,
        }
    }

    /// The street (`Layout` JSON) with the meteors' `{radius, mass, count}`
    /// and the finish line.
    pub fn layout(&self) -> String {
        let mut v = serde_json::to_value(self.street.street()).unwrap();
        v["meteor"] =
            serde_json::json!({ "radius": METEOR_RADIUS, "mass": METEOR_MASS, "count": METEORS });
        v["finish"] = FINISH.into();
        v.to_string()
    }

    /// A fresh run on an unbroken street.
    pub fn reset(&mut self) {
        self.street.reset();
        self.state = State::Ready;
        (self.start, self.end, self.next) = (0.0, 0.0, 0.0);
        self.strikes = 0;
        self.pending.clear();
        self.watches.clear();
        self.parks.clear();
        self.health = 100.0;
        (
            self.score,
            self.near,
            self.close,
            self.hits,
            self.meteor_hits,
        ) = (0, 0, 0, 0, 0);
    }

    /// `step` at time `t` with the car's `[x, y, z, vx, vy, vz]`:
    /// `[launches, (slot, px, py, pz, vx, vy, vz, tags) × launches,
    ///   blasts, (cx, cy, cz, n, (block, vx, vy, vz, wx, wy, wz) × n) × blasts,
    ///   parks, slot × parks]`.
    pub fn tick(&mut self, t: f32, dt: f32, car: &[f32], gravity_y: f32) -> Vec<f32> {
        let (launch, blasts, park) = self.step(
            t,
            dt,
            [car[0], car[1], car[2]],
            [car[3], car[4], car[5]],
            gravity_y,
        );
        let mut out = vec![launch.len() as f32];
        for s in launch {
            out.push(s.slot as f32);
            out.extend_from_slice(&s.p);
            out.extend_from_slice(&s.v);
            out.push(s.tags as f32);
        }
        out.push(blasts.len() as f32);
        for (c, list) in blasts {
            out.extend_from_slice(&c);
            out.push(list.len() as f32);
            for (i, v, w) in list {
                out.push(i as f32);
                out.extend_from_slice(&v);
                out.extend_from_slice(&w);
            }
        }
        out.push(park.len() as f32);
        out.extend(park.iter().map(|&s| s as f32));
        out
    }

    /// `watched_list` flat: `(kind, index)` pairs.
    pub fn watched(&self) -> Vec<u32> {
        self.watched_list()
            .into_iter()
            .flat_map(|(k, i)| [if k == Kind::Meteor { 1 } else { 0 }, i])
            .collect()
    }

    /// `judge_at` with the car `[px, py, pz, qx, qy, qz, qw, vx, vy, vz]` and
    /// the bodies `(kind, index, px, py, pz, vx, vy, vz)` flat: what happened
    /// as `(event, kind, index, x, y, z)` each (`Event` codes, kind 1 a meteor).
    pub fn judge(&mut self, t: f32, car: &[f32], bodies: &[f32]) -> Vec<f32> {
        let list: Vec<_> = bodies
            .chunks_exact(8)
            .map(|b| {
                let kind = if b[0] == 1.0 {
                    Kind::Meteor
                } else {
                    Kind::Block
                };
                (kind, b[1] as u32, [b[2], b[3], b[4]], [b[5], b[6], b[7]])
            })
            .collect();
        self.judge_at(
            t,
            [car[0], car[1], car[2]],
            [car[3], car[4], car[5], car[6]],
            [car[7], car[8], car[9]],
            &list,
        )
        .into_iter()
        .flat_map(|(e, k, i, p)| {
            [
                e as u32 as f32,
                if k == Kind::Meteor { 1.0 } else { 0.0 },
                i as f32,
                p[0],
                p[1],
                p[2],
            ]
        })
        .collect()
    }

    /// `Status` JSON at time `t` with the car at `car_z`.
    pub fn status(&self, t: f32, car_z: f32) -> String {
        serde_json::to_string(&self.status_at(t, car_z)).unwrap()
    }

    /// `zones` as `ZoneDesc` JSON.
    pub fn zones(&self, player: u32, chassis_y: f32, parked: &[u32]) -> String {
        serde_json::to_string(&zones(player, chassis_y, parked)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f32 = -9.81;

    fn ballistic(s: &Strike, t: f32) -> [f32; 3] {
        [0, 1, 2].map(|a| s.p[a] + s.v[a] * t + if a == 1 { 0.5 * G * t * t } else { 0.0 })
    }

    #[test]
    fn the_street_is_long_and_clear() {
        let l = layout();
        assert!(l.blocks.len() > 10_000, "{}", l.blocks.len());
        for b in &l.blocks {
            assert!(b.p[0].abs() - b.half[0] >= STREET_HALF - 1e-4);
        }
        for p in &l.parked {
            assert!(p.p[0].abs() < STREET_HALF - 1.5);
        }
        for z in zones(1, 0.9, &[2, 3]) {
            z.to_gpu().unwrap();
        }
    }

    #[test]
    fn meteors_arrive_where_and_when_they_are_aimed() {
        let r = MeteorRun::new();
        for k in 0..60 {
            let s = r.plan(k, 3.0, [-2.0, 40.0], 11.0, G);
            let flight = s.arrive - 3.0;
            let at = ballistic(&s, flight);
            for a in 0..3 {
                assert!(
                    (at[a] - s.target[a]).abs() < 1e-2,
                    "{k}: {at:?} vs {:?}",
                    s.target
                );
            }
            if s.facade {
                assert_eq!(s.tags, 0);
                assert!((s.target[0].abs() + METEOR_RADIUS - STREET_HALF).abs() < 1e-4);
                // Over the street once it is below the roofs (12 m), so it
                // comes down onto this facade, not through the one opposite.
                let roofs = gauntlet::ROWS as f32 * 2.0 * BLOCK_HALF[1];
                for i in 0..=40 {
                    let p = ballistic(&s, flight * i as f32 / 40.0);
                    if p[1] - METEOR_RADIUS < roofs {
                        assert!(
                            p[0].abs() < STREET_HALF - METEOR_RADIUS + 1e-3,
                            "{k}: {p:?}"
                        );
                    }
                }
                assert!(SPANS
                    .iter()
                    .any(|&(a, b)| s.target[2] > a && s.target[2] < b));
            } else {
                assert_eq!(s.tags, METEOR_TAGS);
                assert!(s.target[2] > 40.0 + 11.0 * flight - 4.0);
            }
            assert!(
                s.p[1] >= 30.0 && s.p[2] > s.target[2] + 40.0,
                "from the sky ahead"
            );
        }
    }

    #[test]
    fn the_director_starts_on_the_move_speeds_up_and_stops_at_the_finish() {
        let mut r = MeteorRun::new();
        let dt = 1.0 / 60.0;
        let (mut launches, mut blasts, mut parks) = (vec![], 0, 0);
        let mut z = gauntlet::SPAWN[2];
        for step in 0..60 * 40 {
            let t = step as f32 * dt;
            // Wait two seconds, then drive at 10 m/s.
            if t > 2.0 {
                z += 10.0 * dt;
            }
            let (l, b, p) = r.step(t, dt, [-2.0, 1.0, z], [0.0, 0.0, 10.0], G);
            for _ in &l {
                assert!(t > 2.0 && z < FINISH - 20.0);
                launches.push(t);
            }
            blasts += b.len();
            parks += p.len();
            // Every meteor lands well away from the car (judged at once).
            let w: Vec<_> = r
                .watched_list()
                .into_iter()
                .map(|(k, i)| (k, i, [100.0, 0.1, z], [0.0; 3]))
                .collect();
            for x in &mut r.watches {
                x.airborne = true;
            }
            r.judge_at(
                t,
                [-2.0, 1.0, z],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 10.0],
                &w,
            );
        }
        assert_eq!(r.state, State::Finished);
        assert!(launches.len() > 20, "{}", launches.len());
        let gaps: Vec<f32> = launches.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps[0] > gaps[gaps.len() - 1], "{gaps:?}");
        assert!(blasts >= launches.len() / 2, "{blasts}");
        assert!(parks > 0);
        // Nothing launched after the finish.
        let s = r.status_at(40.0, FINISH);
        assert_eq!(s.state, State::Finished);
        assert_eq!(s.meteors as usize, launches.len());
    }

    #[test]
    fn hits_cost_health_and_close_landings_score() {
        let mut r = MeteorRun::new();
        let dt = 1.0 / 60.0;
        r.step(0.0, dt, [0.0, 1.0, 0.0], [0.0; 3], G);
        assert_eq!(r.state, State::Running);
        let car = [0.0, 1.0, 0.0];
        let q = [0.0, 0.0, 0.0, 1.0];
        // A block through the car's cab at 9 m/s.
        r.watch(Kind::Block, 5, 0.0);
        let e = r.judge_at(
            0.1,
            car,
            q,
            [0.0; 3],
            &[(Kind::Block, 5, [0.5, 1.5, 0.5], [0.0, -9.0, 0.0])],
        );
        assert_eq!(e[0].0, Event::Hit);
        let (base, per, _) = BLOCK_DAMAGE;
        assert!((r.health - (100.0 - (base + per * 9.0))).abs() < 1e-4);
        // A block coming down 4 m away: a near miss; 15 m away: nothing.
        for (i, x, want) in [(6, 4.0, Some(Event::NearMiss)), (7, 15.0, None)] {
            r.watch(Kind::Block, i, 0.0);
            r.judge_at(
                0.2,
                car,
                q,
                [0.0; 3],
                &[(Kind::Block, i, [x, 3.0, 0.0], [0.0, -5.0, 0.0])],
            );
            let e = r.judge_at(
                0.5,
                car,
                q,
                [0.0; 3],
                &[(Kind::Block, i, [x, 0.3, 0.0], [0.0, -5.0, 0.0])],
            );
            assert_eq!(e.first().map(|e| e.0), want);
            assert!(
                r.watched_list().iter().all(|&(_, j)| j != i),
                "done with it"
            );
        }
        assert_eq!((r.near, r.score), (1, NEAR_POINTS));
        // A meteor beside the car: a close call; three on it wreck the car.
        r.watch(Kind::Meteor, 0, 0.0);
        r.judge_at(
            0.2,
            car,
            q,
            [0.0; 3],
            &[(Kind::Meteor, 0, [6.0, 5.0, 2.0], [0.0, -20.0, 0.0])],
        );
        let e = r.judge_at(
            0.4,
            car,
            q,
            [0.0; 3],
            &[(Kind::Meteor, 0, [6.0, 0.9, 2.0], [0.0, -20.0, 0.0])],
        );
        assert_eq!(e[0].0, Event::CloseCall);
        for k in 1..4 {
            r.watch(Kind::Meteor, k, 0.0);
            r.judge_at(
                0.5,
                car,
                q,
                [0.0; 3],
                &[(Kind::Meteor, k, [0.0, 1.5, 1.0], [0.0, -25.0, 0.0])],
            );
        }
        assert_eq!(r.state, State::Wrecked);
        assert_eq!(r.health, 0.0);
        // Spent meteors are parked a while later.
        let (_, _, park) = r.step(10.0, dt, car, [0.0; 3], G);
        assert_eq!(park.len(), 4);
    }

    #[test]
    fn a_facade_meteor_ends_in_its_blast() {
        let mut r = MeteorRun::new();
        let dt = 1.0 / 60.0;
        let car = [-2.0, 1.0, 0.0];
        r.step(0.0, dt, car, [0.0; 3], G);
        let (mut slot, mut arrive) = (None, 0.0);
        for step in 0..600 {
            let t = step as f32 * dt;
            let (l, b, p) = r.step(t, dt, car, [0.0; 3], G);
            if let Some(s) = l.iter().find(|s| s.facade) {
                if slot.is_none() {
                    (slot, arrive) = (Some(s.slot), s.arrive);
                }
            }
            if !b.is_empty() && slot.is_some() {
                assert!(
                    arrive - t <= dt + 1e-4 && arrive - t > -1e-4,
                    "a step before it arrives"
                );
                assert!(p.contains(&slot.unwrap()), "parked as its wall bursts");
                assert!(r
                    .watched_list()
                    .iter()
                    .all(|&(k, i)| !(k == Kind::Meteor && Some(i) == slot)));
                return;
            }
        }
        panic!("no facade blast");
    }

    #[test]
    fn a_turned_car_is_hit_in_its_own_frame() {
        // Yawed 90°: its long axis is along x.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let q = [0.0, s, 0.0, s];
        let d = unrotate(q, [2.5, 0.0, 0.0]);
        assert!((d[2].abs() - 2.5).abs() < 1e-4 && d[0].abs() < 1e-4);
    }

    #[test]
    fn reset_clears_the_run_and_rebuilds_the_street() {
        let mut r = MeteorRun::new();
        let dt = 1.0 / 60.0;
        let n = r.street.street().blocks.len();
        for step in 0..600 {
            r.step(
                step as f32 * dt,
                dt,
                [0.0, 1.0, step as f32 * 0.2],
                [0.0, 0.0, 12.0],
                G,
            );
        }
        assert!(r.strikes > 0);
        r.reset();
        assert_eq!(r.state, State::Ready);
        assert_eq!(r.strikes, 0);
        assert_eq!(r.street.street().blocks.len(), n);
        assert!(
            r.street
                .blast_patch([STREET_HALF, 4.0, 20.0], 0, 2.0, 8.0)
                .len()
                > 6
        );
    }
}
