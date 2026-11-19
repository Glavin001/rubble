//! The vehicle lab's Derby mode (`vehicle-lab.html?mode=derby`): a demolition derby of
//! soft-body cars in a walled arena, one driven by the player and the rest by
//! `engine_softbody::drive_ai::DerbyAi`.
//!
//! Everything here is scene data and host logic, kept apart from the lab's GPU World so
//! that another host (a playground preset) can reuse it:
//! - the arena ([`arena_fixtures`]: eight walls round a 70 m floor and two ramps, static
//!   shapes in the cars' own solver and fixture bodies in the World; [`arena_props`]:
//!   brick walls, tyre stacks, barrels, crates and concrete blocks, rigid bodies that the
//!   cars knock about);
//! - where the cars start ([`starts`]: round the edge, facing the middle);
//! - the round ([`DerbyGame`]: the cars' poses from their wheels, who is out and why,
//!   the winner) and the AI's inputs.
//!
//! Nothing here changes the physics: the walls and ramps are the solver's ordinary static
//! shapes, the props ordinary rigid bodies, and the AI sets drive input as keys do.

use super::{LabGeom, Motion, ObjDef, GROUND};
use engine_softbody::drive_ai::{
    derby_drivers, derby_inputs, wheel_frame, AiArena, AiRamp, DerbyAi, DerbyCarView, Pose,
};
use engine_softbody::explicit_drive::{Drive, DriveInput};
#[cfg(test)]
use glam::DVec3;
use glam::{Quat, Vec3};
use serde_json::{json, Value};
use std::f32::consts::{FRAC_PI_4, PI, TAU};

/// The arena's floor radius to the walls' inner faces (m): 70 m across.
pub const RADIUS: f32 = 35.0;
/// The walls' height and half thickness (m). No ceiling: a part thrown up comes down.
pub const WALL_HEIGHT: f32 = 5.0;
const WALL_HALF_THICK: f32 = 1.5;
/// Where the cars start: this far from the middle (m), facing it.
pub const START_RADIUS: f32 = 27.0;
/// A round's longest time (s): then the least damaged car still in wins.
pub const ROUND_LIMIT: f64 = 180.0;
/// Knocked out: on its side or roof this long (s).
pub const FLIPPED_OUT: f32 = 3.0;
/// Knocked out: not moved [`STILL_REACH`] m in this long (s).
pub const STILL_OUT: f64 = 12.0;
pub const STILL_REACH: f32 = 2.0;
/// Knocked out: this many wheels off the main body.
pub const WHEELS_OUT: u32 = 2;
/// Where the obstacles go when they are off: far under the floor (the layout of the
/// cars' static shapes is the same either way, so the switch is a restart).
const AWAY: f32 = -1.0e4;

/// The ramps: (angle about the middle, the foot's distance from it, length, rise, width).
/// Each points at the middle, so a car off it lands among the others.
const RAMPS: [(f32, f32, f32, f32, f32); 2] = [
    (FRAC_PI_4, 21.0, 6.5, 1.3, 4.0),
    (PI + FRAC_PI_4, 21.0, 6.5, 1.3, 4.0),
];
/// How far before a ramp's foot the AI starts its run-up (m).
const RUN_UP: f32 = 8.0;
/// A ramp's half thickness (m): its underside reaches below the floor at its high end, so
/// nothing drives under it.
const RAMP_HALF_THICK: f32 = 1.0;

/// The derby's settings: the obstacles (ramps and props), whether car 0 is the player's,
/// the AI's aggression (0..1), and the AI's seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerbySpec {
    pub obstacles: bool,
    pub player: bool,
    pub aggression: f32,
    pub seed: u64,
}

impl Default for DerbySpec {
    fn default() -> Self {
        Self {
            obstacles: true,
            player: true,
            aggression: 0.6,
            seed: 1,
        }
    }
}

impl DerbySpec {
    /// `{ obstacles, player, aggression, seed }` over `d`.
    pub fn from_json(v: &Value, d: Self) -> Self {
        Self {
            obstacles: v
                .get("obstacles")
                .and_then(Value::as_bool)
                .unwrap_or(d.obstacles),
            player: v.get("player").and_then(Value::as_bool).unwrap_or(d.player),
            aggression: v
                .get("aggression")
                .and_then(Value::as_f64)
                .map_or(d.aggression, |x| x as f32)
                .clamp(0.0, 1.0),
            seed: v.get("seed").and_then(Value::as_u64).unwrap_or(d.seed),
        }
    }

    pub fn json(&self) -> Value {
        json!({ "obstacles": self.obstacles, "player": self.player, "aggression": self.aggression, "seed": self.seed })
    }
}

/// A point `r` m from the middle at angle `a` (from +z toward +x: a car's yaw there faces
/// outward).
fn at(a: f32, r: f32) -> Vec3 {
    Vec3::new(r * a.sin(), 0.0, r * a.cos())
}

/// A ramp's box: its top face the ramp's surface, from its foot (on the floor) to `rise`
/// m up `len` m on; centre, half extents, rotation.
fn ramp_box(foot: Vec3, dir: Vec3, len: f32, rise: f32, width: f32) -> (Vec3, Vec3, Quat) {
    let yaw = Quat::from_rotation_arc(Vec3::Z, dir);
    let tilt = Quat::from_rotation_x(-(rise / len).atan());
    let rot = yaw * tilt;
    let half = Vec3::new(
        0.5 * width,
        RAMP_HALF_THICK,
        0.5 * (len * len + rise * rise).sqrt(),
    );
    let top = foot + dir * (0.5 * len) + Vec3::Y * (0.5 * rise);
    (top - rot * Vec3::Y * RAMP_HALF_THICK, half, rot)
}

/// The arena's fixed shapes, in the cars' own solver (the raycast wheels see them, the
/// nodes get static contact) and as fixture bodies in the World (the props stand on and
/// bounce off them; the renderer draws them): eight walls round the floor, then the two
/// ramps (far under the floor when the obstacles are off). Always the same count.
pub fn arena_fixtures(obstacles: bool) -> Vec<(Vec3, Vec3, Quat)> {
    let mut out = Vec::new();
    // An octagon: each wall's inner face `RADIUS` from the middle, overlapping its
    // neighbours at the corners.
    let side = 2.0 * RADIUS * (PI / 8.0).tan();
    for k in 0..8 {
        let a = k as f32 * FRAC_PI_4;
        let c = at(a, RADIUS + WALL_HALF_THICK) + Vec3::Y * (GROUND + 0.5 * WALL_HEIGHT);
        out.push((
            c,
            Vec3::new(
                0.5 * side + WALL_HALF_THICK,
                0.5 * WALL_HEIGHT,
                WALL_HALF_THICK,
            ),
            Quat::from_rotation_y(a),
        ));
    }
    for (a, r, len, rise, width) in RAMPS {
        let foot = at(a, r) + Vec3::Y * GROUND;
        let (mut c, half, rot) = ramp_box(foot, -at(a, 1.0), len, rise, width);
        if !obstacles {
            c.y += AWAY;
        }
        out.push((c, half, rot));
    }
    out
}

/// The rigid props (only with the obstacles on): two brick walls that break apart when
/// rammed, tyre stacks, barrels, crates and concrete blocks. Laid out round the middle,
/// clear of the ramps' run-ups.
pub fn arena_props() -> Vec<ObjDef> {
    let mut out = Vec::new();
    let dynamic = |geom: LabGeom, pos: Vec3, rot: Quat, density: f32, friction: f32| ObjDef {
        motion: Motion::Dynamic,
        density,
        friction,
        geom,
        pos: pos.as_dvec3(),
        rot,
        vel: Vec3::ZERO,
        ang_vel: Vec3::ZERO,
        bullet: false,
    };
    // Brick walls (as the lab's brick stack: 1 x 0.5 x 0.5 m blocks, staggered rows):
    // 6 wide and 5 high, across the diagonals between the ramps.
    for a in [0.75 * PI, 1.75 * PI] {
        let centre = at(a, 15.0);
        let along = Vec3::new(a.cos(), 0.0, -a.sin()); // tangential
        let rot = Quat::from_rotation_y(a);
        let half = Vec3::new(0.5, 0.25, 0.25);
        let (cols, rows) = (6, 5);
        for r in 0..rows {
            for k in 0..cols {
                let off = if r % 2 == 1 { 0.5 } else { 0.0 };
                let x = (k as f32 - 0.5 * (cols as f32 - 1.0)) * 1.01 + off;
                let p = centre + along * x + Vec3::Y * (GROUND + 0.25 + 0.5 * r as f32);
                out.push(dynamic(LabGeom::Box { half }, p, rot, 160.0, 0.6));
            }
        }
    }
    // Tyre stacks (three tyres, about 10 kg each) round the middle.
    for k in 0..4 {
        let c = at(k as f32 * 0.5 * PI, 9.0);
        for h in 0..3 {
            let p = c + Vec3::Y * (GROUND + 0.13 + 0.27 * h as f32);
            out.push(dynamic(
                LabGeom::Cylinder {
                    radius: 0.4,
                    half_len: 0.13,
                },
                p,
                Quat::IDENTITY,
                75.0,
                0.9,
            ));
        }
    }
    // Barrels (about 30 kg) in threes near the walls.
    for a in [0.375 * PI, 1.375 * PI] {
        let c = at(a, 29.0);
        for (dx, dz) in [(0.0, 0.0), (0.65, 0.0), (0.32, 0.56)] {
            let p = c + Vec3::new(dx, GROUND + 0.45, dz);
            out.push(dynamic(
                LabGeom::Cylinder {
                    radius: 0.3,
                    half_len: 0.45,
                },
                p,
                Quat::IDENTITY,
                120.0,
                0.5,
            ));
        }
    }
    // Wooden crates (1 m, 150 kg), two by two.
    for a in [0.125 * PI, 1.125 * PI] {
        let c = at(a, 22.0);
        let rot = Quat::from_rotation_y(a);
        for (dx, dy) in [(-0.55, 0.0), (0.55, 0.0), (0.0, 1.0)] {
            let along = Vec3::new(a.cos(), 0.0, -a.sin());
            let p = c + along * dx + Vec3::Y * (GROUND + 0.5 + dy);
            out.push(dynamic(
                LabGeom::Box {
                    half: Vec3::splat(0.5),
                },
                p,
                rot,
                150.0,
                0.6,
            ));
        }
    }
    // Concrete blocks (2 x 0.8 x 0.6 m, about 2.3 t): moved only by a hard hit.
    for a in [0.625 * PI, 1.625 * PI] {
        let c = at(a, 24.0) + Vec3::Y * (GROUND + 0.4);
        out.push(dynamic(
            LabGeom::Box {
                half: Vec3::new(1.0, 0.4, 0.3),
            },
            c,
            Quat::from_rotation_y(a),
            2400.0,
            0.7,
        ));
    }
    out
}

/// What the AI knows of the arena: its radius, the ramps (none when the obstacles are
/// off), and the props' places to steer round (where they were laid out).
pub fn ai_arena(obstacles: bool) -> AiArena {
    if !obstacles {
        return AiArena {
            radius: RADIUS,
            ..AiArena::default()
        };
    }
    let ramps = RAMPS
        .iter()
        .map(|&(a, r, len, _, _)| {
            let foot = at(a, r);
            let dir = -at(a, 1.0);
            AiRamp {
                foot: [foot.x, foot.z],
                dir: [dir.x, dir.z],
                len,
                run_up: RUN_UP,
            }
        })
        .collect();
    let mut blockers: Vec<([f32; 2], f32)> = Vec::new();
    for a in [0.75 * PI, 1.75 * PI] {
        let c = at(a, 15.0);
        blockers.push(([c.x, c.z], 3.2));
    }
    for a in [0.625 * PI, 1.625 * PI] {
        let c = at(a, 24.0);
        blockers.push(([c.x, c.z], 1.2));
    }
    // (The ramps' sides, seen from across: a car not taking one goes round it.)
    for &(a, r, len, _, _) in &RAMPS {
        let c = at(a, r - 0.5 * len);
        blockers.push(([c.x, c.z], 3.0));
    }
    AiArena {
        radius: RADIUS,
        ramps,
        blockers,
    }
}

/// Where `n` cars start: round the edge, [`START_RADIUS`] from the middle, each facing
/// it; car 0 (the player's) at the south. `(x, z, yaw)`.
pub fn starts(n: usize) -> Vec<(f32, f32, f32)> {
    (0..n)
        .map(|k| {
            let a = PI + TAU * k as f32 / n.max(1) as f32;
            let p = at(a, START_RADIUS);
            (p.x, p.z, a + PI)
        })
        .collect()
}

/// One car's state in the round.
#[derive(Clone, Debug, Default)]
pub struct DerbyCar {
    pub pose: Option<Pose>,
    /// Out of the round: why, and when (round time, s).
    pub out: Option<(&'static str, f64)>,
    /// Seconds on its side or roof without a break.
    pub flipped_for: f32,
    /// Where it last was after moving [`STILL_REACH`] m, and when.
    anchor: Option<(Vec3, f64)>,
    /// Damage 0..1 (its beams bent and broken) and wheels off the main body.
    pub damage: f32,
    pub wheels_off: u32,
    /// Per wheel whether it is on the main body (the pose uses only those).
    pub attached: Vec<bool>,
}

/// A derby round: the cars, the AI, who is out, the winner.
pub struct DerbyGame {
    pub spec: DerbySpec,
    pub cars: Vec<DerbyCar>,
    drivers: Vec<DerbyAi>,
    arena: AiArena,
    /// The round's time (s).
    pub t: f64,
    pub over: bool,
    pub winner: Option<usize>,
    /// (round time, text), newest last.
    pub events: Vec<(f64, String)>,
    /// The last input sent to each car (only changes are sent).
    sent: Vec<Option<DriveInput>>,
    /// A drive state per car to decode its wheel records into.
    drive: Option<Drive>,
    /// Poses read so far.
    pub readings: u64,
}

impl DerbyGame {
    /// A new round of `n` cars like `car_drive` (the cars' raycast wheels).
    pub fn new(spec: DerbySpec, n: usize, car_drive: Option<&Drive>) -> Self {
        Self {
            spec,
            cars: vec![DerbyCar::default(); n],
            drivers: derby_drivers(n, spec.seed),
            arena: ai_arena(spec.obstacles),
            t: 0.0,
            over: false,
            winner: None,
            events: Vec::new(),
            sent: vec![None; n],
            drive: car_drive.cloned(),
            readings: 0,
        }
    }

    /// The player's car (car 0), when there is a player.
    pub fn player(&self) -> Option<usize> {
        self.spec.player.then_some(0)
    }

    /// The cars' poses from a reading: `origins` every car's origin record (4 words a car)
    /// and `drive` every car's drive region (`stride` records a car), as the solver's
    /// misc buffer holds them.
    pub fn ingest_drive(&mut self, origins: &[u32], drive: &[u32], stride: usize) {
        let Some(mut d) = self.drive.clone() else {
            return;
        };
        let f = |w: u32| f32::from_bits(w);
        let words = 4 * stride;
        for (v, car) in self.cars.iter_mut().enumerate() {
            let (Some(w), Some(o)) = (
                drive.get(v * words..(v + 1) * words),
                origins.get(4 * v..4 * v + 3),
            ) else {
                continue;
            };
            engine_exec::explicit_plan::drive::decode(w, &mut d);
            let mask = (car.attached.len() == d.wheels.len() && car.attached.iter().any(|a| *a))
                .then_some(car.attached.as_slice());
            let (c, fwd, up) = wheel_frame(&d, mask);
            let p = Vec3::new(f(o[0]), f(o[1]), f(o[2])) + c;
            if p.is_finite() && fwd.is_finite() && up.is_finite() && d.speed.is_finite() {
                car.pose = Some(Pose {
                    p,
                    fwd,
                    up,
                    speed: d.speed,
                });
            }
        }
        self.readings += 1;
    }

    /// Car `v`'s damage from a beam reading: beams bent and broken against its count,
    /// and its wheels' attachment (none: unknown).
    pub fn set_damage(
        &mut self,
        v: usize,
        yielded: u32,
        broken: u32,
        total: u32,
        attached: Option<Vec<bool>>,
    ) {
        let Some(car) = self.cars.get_mut(v) else {
            return;
        };
        // Full damage at half the beams bent, a broken beam counting as three (measured on
        // the generated sedan: one head-on hit at 50 km/h bends about 600 of its 4251 beams
        // and breaks about 60, which reads as a third).
        car.damage =
            ((yielded as f32 + 3.0 * broken as f32) / (0.5 * total.max(1) as f32)).min(1.0);
        if let Some(a) = attached {
            car.wheels_off = a.iter().filter(|x| !**x).count() as u32;
            car.attached = a;
        }
    }

    fn knock_out(&mut self, v: usize, why: &'static str) {
        if self.cars[v].out.is_some() || self.over {
            return;
        }
        self.cars[v].out = Some((why, self.t));
        let who = if Some(v) == self.player() {
            "You are".to_string()
        } else {
            format!("Car {} is", v + 1)
        };
        self.events.push((self.t, format!("{who} out: {why}")));
    }

    /// Advance the round by `dt` s: the knock-outs, the round's end, and each car's drive
    /// input to send now (only the changes; the player's car only once it is out).
    pub fn tick(&mut self, dt: f32) -> Vec<(usize, DriveInput)> {
        if !self.over {
            self.t += f64::from(dt);
        }
        let n = self.cars.len();
        if self.readings > 0 && !self.over {
            // This tick's knock-outs, then applied together.
            let mut outs: Vec<(usize, &'static str)> = Vec::new();
            for v in 0..n {
                let Some(p) = self.cars[v].pose else { continue };
                if self.cars[v].out.is_some() {
                    continue;
                }
                let t = self.t;
                let car = &mut self.cars[v];
                car.flipped_for = if p.up.y < 0.25 {
                    car.flipped_for + dt
                } else {
                    0.0
                };
                let moved = car
                    .anchor
                    .is_none_or(|(a, _)| (p.p - a).length() > STILL_REACH);
                if moved {
                    car.anchor = Some((p.p, t));
                }
                let still = car.anchor.map_or(0.0, |(_, at)| t - at);
                let outside =
                    Vec3::new(p.p.x, 0.0, p.p.z).length() > RADIUS + 4.0 || p.p.y < GROUND - 5.0;
                let why = if car.flipped_for > FLIPPED_OUT {
                    Some("rolled over")
                } else if car.wheels_off >= WHEELS_OUT {
                    Some("wheels torn off")
                } else if outside {
                    Some("thrown out of the arena")
                } else if still > STILL_OUT {
                    Some("stopped moving")
                } else {
                    None
                };
                if let Some(w) = why {
                    outs.push((v, w));
                }
            }
            // Every car still in out at once (a pile that stopped together): the least
            // damaged of them stays in and wins.
            let left = (0..n).filter(|&v| self.cars[v].out.is_none()).count();
            if left > 0 && outs.len() == left {
                let spare = outs.iter().map(|o| o.0).min_by(|a, b| {
                    self.cars[*a]
                        .damage
                        .total_cmp(&self.cars[*b].damage)
                        .then(a.cmp(b))
                });
                outs.retain(|o| Some(o.0) != spare);
            }
            for (v, why) in outs {
                self.knock_out(v, why);
            }
            let alive: Vec<usize> = (0..n).filter(|&v| self.cars[v].out.is_none()).collect();
            let timeout = self.t >= ROUND_LIMIT;
            if (n > 1 && alive.len() <= 1) || timeout {
                // The last car moving; at the time limit the least damaged still in.
                self.winner = alive.iter().copied().min_by(|a, b| {
                    self.cars[*a]
                        .damage
                        .total_cmp(&self.cars[*b].damage)
                        .then(a.cmp(b))
                });
                let text = match self.winner {
                    Some(w) if Some(w) == self.player() => "You win!".to_string(),
                    Some(w) => format!("Car {} wins", w + 1),
                    None => "No winner".to_string(),
                };
                self.events.push((
                    self.t,
                    if timeout {
                        format!("Time: {text}")
                    } else {
                        text
                    },
                ));
                self.over = true;
            }
        }
        let views: Vec<DerbyCarView> = self
            .cars
            .iter()
            .map(|c| DerbyCarView {
                pose: c.pose,
                alive: c.out.is_none(),
            })
            .collect();
        let skip = self.player().filter(|&p| self.cars[p].out.is_none());
        let player = self.player();
        let mut inputs = derby_inputs(
            &mut self.drivers,
            &views,
            &self.arena,
            skip,
            player,
            self.spec.aggression,
            dt,
        );
        if self.over {
            // The round is over: every car rolls to a stop (the player's too).
            inputs = vec![
                Some(DriveInput {
                    brake: 0.6,
                    ..DriveInput::default()
                });
                n
            ];
        }
        let mut out = Vec::new();
        for (v, i) in inputs.into_iter().enumerate() {
            if let Some(i) = i {
                if self.sent[v] != Some(i) {
                    self.sent[v] = Some(i);
                    out.push((v, i));
                }
            }
        }
        out
    }

    /// The player's car's input was set by the page (it is sent again when the AI takes
    /// the car over).
    pub fn player_input(&mut self) {
        if let Some(p) = self.player() {
            self.sent[p] = None;
        }
    }

    /// The live settings: the player on or off, the AI's aggression.
    pub fn set_spec(&mut self, spec: DerbySpec) {
        if spec.player != self.spec.player {
            self.sent = vec![None; self.cars.len()];
        }
        self.spec.player = spec.player;
        self.spec.aggression = spec.aggression;
    }

    /// The round for the page: `{ t, limit, over, winner, left, player, cars: [{ out,
    /// outAt, damage, wheelsOff, kmh, flipped, target, plan, pos }], events: [{ t, text }] }`.
    pub fn json(&self) -> Value {
        let left = self.cars.iter().filter(|c| c.out.is_none()).count();
        json!({
            "t": self.t, "limit": ROUND_LIMIT, "over": self.over, "winner": self.winner, "left": left,
            "player": self.player(), "spec": self.spec.json(), "readings": self.readings,
            "cars": self.cars.iter().enumerate().map(|(v, c)| json!({
                "out": c.out.map(|o| o.0), "outAt": c.out.map(|o| o.1), "damage": c.damage, "wheelsOff": c.wheels_off,
                "kmh": c.pose.map_or(0.0, |p| p.speed * 3.6), "flipped": c.flipped_for,
                "pos": c.pose.map(|p| p.p.to_array()), "up": c.pose.map(|p| p.up.y),
                "target": self.drivers.get(v).and_then(|d| d.target()),
                "plan": if Some(v) == self.player() { "player" } else { self.drivers.get(v).map_or("", |d| d.plan_name()) },
            })).collect::<Vec<_>>(),
            "events": self.events.iter().rev().take(12).map(|(t, s)| json!({ "t": t, "text": s })).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arena_keeps_its_layout_with_the_obstacles_off() {
        let on = arena_fixtures(true);
        let off = arena_fixtures(false);
        assert_eq!(on.len(), off.len());
        // The walls stay; the ramps go far under the floor.
        assert_eq!(on[..8], off[..8]);
        assert!(off[8..].iter().all(|f| f.0.y < -1000.0));
    }

    #[test]
    fn cars_start_inside_the_walls_facing_the_middle() {
        for n in [4, 6, 8, 12, 16, 32] {
            let s = starts(n);
            assert_eq!(s.len(), n);
            for (x, z, yaw) in &s {
                let p = Vec3::new(*x, 0.0, *z);
                assert!(p.length() + 2.6 < RADIUS);
                let fwd = Quat::from_rotation_y(*yaw) * Vec3::Z;
                assert!(fwd.dot(-p.normalize()) > 0.999, "{n}: {fwd} at {p}");
            }
            // Side by side with room between (the cars are under 2.1 m wide).
            let gap = 2.0 * PI * START_RADIUS / n as f32;
            assert!(gap > 4.0, "{n} cars: {gap} m apart");
        }
    }

    #[test]
    fn a_ramp_rises_from_its_foot_and_its_underside_meets_the_floor() {
        for (c, half, rot) in &arena_fixtures(true)[8..] {
            let top = |z: f32| *c + *rot * Vec3::new(0.0, half.y, z);
            // The surface's low end at the floor, its high end about the rise up.
            assert!(top(-half.z).y.abs() < 0.02, "{}", top(-half.z));
            assert!((top(half.z).y - 1.3).abs() < 0.05, "{}", top(half.z));
            // The underside at the high end below the floor: nothing drives under it.
            assert!((*c + *rot * Vec3::new(0.0, -half.y, half.z)).y < 0.0);
        }
    }

    #[test]
    fn the_props_stand_inside_the_arena_on_the_floor() {
        let props = arena_props();
        assert!(props.len() > 60 && props.len() < 200, "{}", props.len());
        for p in &props {
            let h = DVec3::new(p.pos.x, 0.0, p.pos.z).length() as f32;
            assert!(h < RADIUS - 1.0 && p.pos.y > 0.0, "{:?}", p.pos);
            assert_eq!(p.motion, Motion::Dynamic);
        }
    }

    fn pose(x: f32, z: f32, up: Vec3) -> Pose {
        Pose {
            p: Vec3::new(x, 0.6, z),
            fwd: Vec3::Z,
            up,
            speed: 0.0,
        }
    }

    #[test]
    fn a_car_on_its_roof_is_out_after_three_seconds_and_the_last_one_wins() {
        let mut g = DerbyGame::new(DerbySpec::default(), 2, None);
        g.cars[0].pose = Some(pose(0.0, 0.0, Vec3::Y));
        g.cars[1].pose = Some(pose(10.0, 0.0, -Vec3::Y));
        g.readings = 1;
        let dt = 1.0 / 60.0;
        for _ in 0..170 {
            g.tick(dt);
        }
        assert!(g.cars[1].out.is_none());
        for _ in 0..20 {
            g.tick(dt);
        }
        assert_eq!(g.cars[1].out.map(|o| o.0), Some("rolled over"));
        assert!(g.over);
        assert_eq!(g.winner, Some(0));
    }

    #[test]
    fn a_car_that_does_not_move_is_out_and_lost_wheels_knock_out() {
        let mut g = DerbyGame::new(DerbySpec::default(), 3, None);
        for v in 0..3 {
            g.cars[v].pose = Some(pose(5.0 * v as f32, 0.0, Vec3::Y));
        }
        g.readings = 1;
        g.set_damage(2, 0, 10, 1000, Some(vec![true, false, false, true]));
        g.tick(0.1);
        assert_eq!(g.cars[2].out.map(|o| o.0), Some("wheels torn off"));
        // Cars 0 and 1 never move: out after STILL_OUT, the round then over.
        for _ in 0..(STILL_OUT * 10.0) as usize + 5 {
            g.tick(0.1);
        }
        assert!(g.cars[0].out.is_some() || g.cars[1].out.is_some());
        assert!(g.over);
    }
}
