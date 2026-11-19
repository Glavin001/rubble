//! Self-driving cars on raycast wheels: host logic that reads the cars' poses and sets
//! their drive input ([`DriveInput`]), exactly as a player's keys do. Nothing here
//! touches the solver.
//!
//! - [`AiCar`]: the soft solver's traffic AI (`softbody.html?scene=traffic`): wander,
//!   chase the nearest car, or cruise and brake for cars ahead.
//! - [`DerbyAi`]: a demolition derby driver (the vehicle lab's Derby mode): it hunts a
//!   car (spread over the field so they do not all pile onto one), rams it at speed, backs
//!   off and rams again, reverses out when stuck, steers round the arena's walls and
//!   obstacles, sometimes takes a ramp, and does nothing once its car is flipped or out.
//!
//! Both are deterministic: the same poses, seed and time steps give the same inputs.

use crate::explicit_drive::{Drive, DriveInput, WheelState};
use glam::Vec3;

/// A vehicle's pose from its wheels: centre, forward, up and forward speed (m/s).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub p: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
    pub speed: f32,
}

/// A vehicle's centre, forward and up from its wheels (`use_wheel`: which wheels count;
/// none: every wheel): the centre their mean, up their ups' mean, forward the fixed
/// (unsteered) wheels' mean heading, else every counted wheel's.
pub fn wheel_frame(d: &Drive, use_wheel: Option<&[bool]>) -> (Vec3, Vec3, Vec3) {
    let on = |k: usize| use_wheel.is_none_or(|m| m.get(k).copied().unwrap_or(true));
    let counted = (0..d.wheels.len()).filter(|&k| on(k)).count();
    let n = counted.max(1) as f32;
    let centre = d
        .wheels
        .iter()
        .enumerate()
        .filter(|(k, _)| on(*k))
        .map(|(_, w)| w.centre)
        .sum::<Vec3>()
        / n;
    let up = d
        .wheels
        .iter()
        .enumerate()
        .filter(|(k, _)| on(*k))
        .map(|(_, w)| w.up)
        .sum::<Vec3>()
        .normalize_or_zero();
    let heading = |w: &WheelState| w.axle.cross(w.up);
    let fixed: Vec3 = d
        .wheels
        .iter()
        .zip(&d.config.wheels)
        .enumerate()
        .filter(|(k, (_, c))| on(*k) && c.max_steer == 0.0)
        .map(|(_, (w, _))| heading(w))
        .sum();
    let fwd = if fixed.length_squared() > 1e-8 {
        fixed.normalize()
    } else {
        d.wheels
            .iter()
            .enumerate()
            .filter(|(k, _)| on(*k))
            .map(|(_, w)| heading(w))
            .sum::<Vec3>()
            .normalize_or_zero()
    };
    (centre, fwd, up)
}

// ------------------------------------------------------------------------------------
// Traffic (softbody.html): unchanged from its first home in engine-world-wasm.

/// A self-driving car (the soft solver's traffic AI, softbody.html, in Rust): it
/// steers toward a target, a random point in the arena (wander) or the nearest
/// other car (chase); a stuck car reverses out; cruising cars brake for cars ahead.
#[derive(Clone, Debug)]
pub struct AiCar {
    chase: bool,
    cruise: bool,
    target: Option<[f32; 2]>,
    t: f32,
    slow: f32,
    reverse: f32,
    kmh: f32,
    seed: u32,
}

/// `n` cars' AI: `mode` 'derby' (two in three wander, one chases), 'cruise' or 'off'.
pub fn ai_cars(n: usize, mode: &str) -> Vec<AiCar> {
    if mode == "off" || mode == "0" {
        return Vec::new();
    }
    let cruise = mode == "cruise";
    (0..n)
        .map(|i| AiCar {
            chase: !cruise && i % 3 == 2,
            cruise,
            target: None,
            t: 0.0,
            slow: 0.0,
            reverse: 0.0,
            kmh: 35.0 + ((i * 17) % 45) as f32,
            seed: (i as u32).wrapping_mul(7919).wrapping_add(13),
        })
        .collect()
}

fn rand_u(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
    (*seed >> 8) as f32 / 16_777_216.0
}

fn rand(c: &mut AiCar) -> f32 {
    rand_u(&mut c.seed)
}

/// One AI car's input from the poses (`k` itself).
pub fn ai_input(c: &mut AiCar, k: usize, poses: &[Option<Pose>], dt: f32) -> DriveInput {
    let none = DriveInput::default();
    let Some(me) = poses[k] else { return none };
    let right = me.fwd.cross(me.up);
    let speed = me.speed.abs();
    c.t += dt;
    if me.up.y < 0.3 {
        return none; // on its side or roof
    }
    if c.chase {
        let mut best: Option<(f32, Pose)> = None;
        for (j, o) in poses.iter().enumerate() {
            if let (true, Some(o)) = (j != k, o) {
                let d = Vec3::new(o.p.x - me.p.x, 0.0, o.p.z - me.p.z).length();
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, *o));
                }
            }
        }
        if let Some((_, o)) = best {
            let v = o.fwd * o.speed;
            c.target = Some([o.p.x + v.x * 0.4, o.p.z + v.z * 0.4]);
        }
    } else if c
        .target
        .is_none_or(|t| c.t > 9.0 || Vec3::new(t[0] - me.p.x, 0.0, t[1] - me.p.z).length() < 8.0)
    {
        let x = (rand(c) - 0.5) * 120.0;
        let z = (rand(c) - 0.5) * 120.0;
        c.target = Some([x, z]);
        c.t = 0.0;
    }
    let t = c.target.unwrap_or([me.p.x, me.p.z]);
    let d = Vec3::new(t[0] - me.p.x, 0.0, t[1] - me.p.z);
    let ang = d.dot(right).atan2(d.dot(me.fwd));
    let mut steer = (ang * 1.6).clamp(-1.0, 1.0);
    let mut want = c.kmh / 3.6 * if ang.abs() > 1.2 { 0.45 } else { 1.0 };
    if c.cruise {
        for (j, o) in poses.iter().enumerate() {
            let (true, Some(o)) = (j != k, o) else {
                continue;
            };
            let dd = o.p - me.p;
            let ahead = dd.x * me.fwd.x + dd.z * me.fwd.z;
            let side = dd.x * right.x + dd.z * right.z;
            if ahead > 0.0 && ahead < 14.0 && side.abs() < ahead * 0.58 {
                want = want.min(((ahead - 6.0) * 0.6).max(0.0));
                let sg = if side > 0.0 {
                    1.0
                } else if side < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                steer = (steer - sg * 0.6).clamp(-1.0, 1.0);
            }
        }
    }
    let mut input = DriveInput {
        forward: if speed < want { 0.85 } else { 0.0 },
        steer,
        brake: if speed > want * 1.3 { 0.4 } else { 0.0 },
        handbrake: 0.0,
    };
    // Stuck against something: reverse out with the wheel turned the other way.
    c.slow = if speed < 0.8 { c.slow + dt } else { 0.0 };
    if c.slow > 1.6 && c.reverse <= 0.0 {
        c.reverse = 1.4;
        c.slow = 0.0;
    }
    if c.reverse > 0.0 {
        c.reverse -= dt;
        input = DriveInput {
            forward: -0.8,
            steer: -steer,
            brake: 0.0,
            handbrake: 0.0,
        };
    }
    input
}

// ------------------------------------------------------------------------------------
// Demolition derby.

/// A ramp the derby AI may take: its foot (x, z), its direction (unit, x z), its length
/// along that direction (m), and how far before the foot the run-up starts (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AiRamp {
    pub foot: [f32; 2],
    pub dir: [f32; 2],
    pub len: f32,
    pub run_up: f32,
}

/// What the derby AI knows of its arena: the walls' inner radius about the origin (the
/// arena is about round), the ramps, and the fixed things to steer round (centre x z,
/// radius).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AiArena {
    pub radius: f32,
    pub ramps: Vec<AiRamp>,
    pub blockers: Vec<([f32; 2], f32)>,
}

/// One derby car's view of the round, from the game layer: whether it is still in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DerbyCarView {
    pub pose: Option<Pose>,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Plan {
    /// Driving at its target car.
    Hunt,
    /// Reversing away (after a hit, before it rams again; or out of a wall or a car it
    /// is stuck against): for at least `min` s and until `dist` m from `from`, at most
    /// [`REVERSE_MOST`] s; `slow`: seconds it has hardly moved while reversing.
    Reverse {
        hit: bool,
        t: f32,
        min: f32,
        dist: f32,
        from: Vec3,
        slow: f32,
    },
    /// Wedged both ways: forward on full lock away from the target (s left).
    Wiggle(f32),
    /// Taking ramp `ramp`: to its run-up (`launched` false), then up it and off.
    Jump { ramp: usize, launched: bool, t: f32 },
}

/// A derby driver.
#[derive(Clone, Debug)]
pub struct DerbyAi {
    seed: u32,
    target: Option<usize>,
    /// Seconds until it picks its target again.
    retarget: f32,
    plan: Plan,
    /// Seconds slower than a walk with the throttle on (stuck).
    slow: f32,
    last_speed: f32,
    /// Its own temperament: 0.85 to 1.15 times the field's speed.
    temper: f32,
    /// Seconds it waits at the start (the field does not all leave at once and meet
    /// head-on in the middle).
    delay: f32,
    /// Which flank of its target it aims at (−1 or 1, m to the target's right): side
    /// hits, not nose to nose (two crushed noses ride up on each other, and a raycast
    /// wheel resting on another car finds no ground: such a pile cannot drive apart).
    flank: f32,
}

/// A derby field's drivers, deterministic from `seed`.
pub fn derby_drivers(n: usize, seed: u64) -> Vec<DerbyAi> {
    (0..n)
        .map(|i| {
            let mut s = (seed as u32 ^ (seed >> 32) as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add((i as u32).wrapping_mul(7919).wrapping_add(13));
            let temper = 0.85 + 0.3 * rand_u(&mut s);
            let delay = 2.0 * rand_u(&mut s);
            let flank = if rand_u(&mut s) < 0.5 { -1.0 } else { 1.0 };
            DerbyAi {
                seed: s,
                target: None,
                retarget: 0.0,
                plan: Plan::Hunt,
                slow: 0.0,
                last_speed: 0.0,
                temper,
                delay,
                flank,
            }
        })
        .collect()
}

/// The longest a derby driver reverses (s).
const REVERSE_MOST: f32 = 3.5;

fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

/// Steering toward horizontal direction `d` from pose `me` (positive right), and the
/// angle off the nose (rad).
fn steer_to(me: &Pose, d: Vec3) -> (f32, f32) {
    let right = me.fwd.cross(me.up);
    let ang = d.dot(right).atan2(d.dot(me.fwd));
    ((ang * 1.8).clamp(-1.0, 1.0), ang)
}

impl DerbyAi {
    /// Its target car now (none: none picked or none left).
    pub fn target(&self) -> Option<usize> {
        self.target
    }

    /// A short name of what it is doing, for the page's labels.
    pub fn plan_name(&self) -> &'static str {
        match self.plan {
            Plan::Hunt => "hunting",
            Plan::Reverse { hit: true, .. } => "backing off",
            Plan::Reverse { hit: false, .. } => "reversing out",
            Plan::Wiggle(_) => "wriggling free",
            Plan::Jump { .. } => "taking the ramp",
        }
    }
}

/// Every AI car's input for the next `dt` seconds. `cars`: every car's pose and whether
/// it is still in (the drivers are `ai[k]` for car `k`); `skip`: the player's car (no
/// input from here); `aggression` 0..1: the field's top speed and how much it prefers the
/// player; `player`: the player's car, if one. Out or flipped cars get no input (rolling
/// to a stop).
pub fn derby_inputs(
    ai: &mut [DerbyAi],
    cars: &[DerbyCarView],
    arena: &AiArena,
    skip: Option<usize>,
    player: Option<usize>,
    aggression: f32,
    dt: f32,
) -> Vec<Option<DriveInput>> {
    let aggression = aggression.clamp(0.0, 1.0);
    // How many drivers chase each car: the spread (each AI's own target counts once).
    let mut claims = vec![0u32; cars.len()];
    for (k, a) in ai.iter().enumerate() {
        if Some(k) == skip || !cars.get(k).is_some_and(|c| c.alive) {
            continue;
        }
        if let Some(t) = a.target.filter(|&t| t < cars.len()) {
            claims[t] += 1;
        }
    }
    (0..ai.len())
        .map(|k| {
            if Some(k) == skip || k >= cars.len() {
                return None;
            }
            Some(derby_input(
                &mut ai[k],
                k,
                cars,
                arena,
                &mut claims,
                player,
                aggression,
                dt,
            ))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn derby_input(
    c: &mut DerbyAi,
    k: usize,
    cars: &[DerbyCarView],
    arena: &AiArena,
    claims: &mut [u32],
    player: Option<usize>,
    aggression: f32,
    dt: f32,
) -> DriveInput {
    let none = DriveInput::default();
    let me_view = cars[k];
    let Some(me) = me_view.pose else { return none };
    if !me_view.alive || me.up.y < 0.3 {
        // Out, or on its side or roof: hands off.
        return none;
    }
    if c.delay > 0.0 {
        // Its start, held on the brake.
        c.delay -= dt;
        return DriveInput { brake: 1.0, ..none };
    }
    let speed = me.speed;
    let hit = c.last_speed - speed.abs() > 3.5 && c.last_speed > 4.0;
    c.last_speed = speed.abs();
    c.retarget -= dt;

    // The target: picked again every few seconds, or when it is out.
    let target_ok = |t: usize| t != k && cars.get(t).is_some_and(|o| o.alive && o.pose.is_some());
    if c.retarget <= 0.0 || !c.target.is_some_and(target_ok) {
        if let Some(t) = c.target.filter(|&t| t < claims.len()) {
            claims[t] = claims[t].saturating_sub(1);
        }
        let mut best: Option<(f32, usize)> = None;
        for (j, o) in cars.iter().enumerate() {
            let (true, Some(op)) = (target_ok(j), o.pose) else {
                continue;
            };
            let d = flat(op.p - me.p).length();
            let mut score = d * (1.0 + 0.7 * claims[j] as f32) + 10.0 * rand_u(&mut c.seed);
            if Some(j) == player {
                score -= 10.0 * aggression;
            }
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, j));
            }
        }
        c.target = best.map(|b| b.1);
        if let Some(t) = c.target {
            claims[t] += 1;
        }
        c.retarget = 4.0 + 4.0 * rand_u(&mut c.seed);
        c.flank = if rand_u(&mut c.seed) < 0.5 { -1.0 } else { 1.0 };
        // Now and then a ramp instead (more often the wilder the field).
        if matches!(c.plan, Plan::Hunt)
            && !arena.ramps.is_empty()
            && rand_u(&mut c.seed) < 0.12 + 0.18 * aggression
        {
            let r = (rand_u(&mut c.seed) * arena.ramps.len() as f32) as usize % arena.ramps.len();
            c.plan = Plan::Jump {
                ramp: r,
                launched: false,
                t: 0.0,
            };
        }
    }
    let top = (28.0 + 42.0 * aggression) / 3.6 * c.temper;

    // Where it wants to go and how fast.
    let (mut aim, mut want) = match c.target.and_then(|t| cars[t].pose) {
        Some(o) => {
            let d = flat(o.p - me.p).length();
            // Lead the target by its own motion, by about the time to reach it.
            let lead = (d / speed.abs().max(6.0)).min(1.2) * 0.8;
            // At one flank: a side hit (the aim comes onto the car itself in the last
            // few metres, so it still connects).
            let side = o.fwd.cross(o.up) * c.flank * (1.4 * (d / 12.0).min(1.0));
            let p = o.p + o.fwd * o.speed * lead + side;
            // Close in: flat out (a ram); further: the field's speed.
            (flat(p - me.p), if d < 14.0 { top * 1.25 } else { top })
        }
        None => (flat(-me.p), top * 0.5),
    };
    if let Plan::Jump {
        ramp,
        mut launched,
        mut t,
    } = c.plan
    {
        t += dt;
        let mut done = arena.ramps.is_empty() || t > 12.0;
        if let Some(r) = arena.ramps.get(ramp % arena.ramps.len().max(1)) {
            let dir = Vec3::new(r.dir[0], 0.0, r.dir[1]);
            let foot = Vec3::new(r.foot[0], 0.0, r.foot[1]);
            let run_up = foot - dir * r.run_up;
            if !launched {
                aim = flat(run_up - me.p);
                want = top * 0.7;
                launched = aim.length() < 4.5;
            } else {
                // Up the ramp and off its end, flat out, then back to the hunt.
                let past = (me.p - foot).dot(dir);
                aim = flat(foot + dir * (r.len + 20.0) - me.p);
                want = top * 1.4;
                done |= past > r.len + 8.0;
            }
        }
        c.plan = if done {
            Plan::Hunt
        } else {
            Plan::Jump { ramp, launched, t }
        };
    }
    let (mut steer, ang) = steer_to(&me, aim);
    // Turning hard: slower.
    if ang.abs() > 1.2 {
        want *= 0.5;
    }

    // The walls: heading for one with nothing to hit there, turn back in.
    let reach = (3.0 + speed.abs() * 0.9).min(14.0);
    let ahead = me.p + me.fwd * reach;
    let aim_far = aim.length() > reach + 4.0;
    if arena.radius > 0.0 && flat(ahead).length() > arena.radius - 2.5 && aim_far {
        let (s, _) = steer_to(&me, flat(-me.p));
        steer = (steer + s).clamp(-1.0, 1.0);
    }
    // Fixed obstacles across its path: round them (unless the target is just there).
    let right = me.fwd.cross(me.up);
    let on_ramp = matches!(c.plan, Plan::Jump { launched: true, .. });
    for (c0, r) in arena.blockers.iter().filter(|_| !on_ramp) {
        let d = Vec3::new(c0[0], 0.0, c0[1]) - me.p;
        let along = d.dot(flat(me.fwd).normalize_or_zero());
        let side = d.dot(flat(right).normalize_or_zero());
        if along > 0.0 && along < reach + r && side.abs() < r + 1.4 && aim.length() > along + r {
            let away = if side >= 0.0 { -1.0 } else { 1.0 };
            steer = (steer + away * 0.9).clamp(-1.0, 1.0);
        }
    }

    let mut input = DriveInput {
        forward: if speed < want { 1.0 } else { 0.0 },
        steer,
        brake: if speed > want * 1.3 + 1.0 { 0.5 } else { 0.0 },
        handbrake: 0.0,
    };

    // A hit (its speed gone at once) near its target: back off a run-up's length, then
    // ram again.
    let near_target = c
        .target
        .and_then(|t| cars[t].pose)
        .is_some_and(|o| flat(o.p - me.p).length() < 7.0);
    if hit && near_target && matches!(c.plan, Plan::Hunt) {
        c.plan = Plan::Reverse {
            hit: true,
            t: 0.0,
            min: 1.2,
            dist: 6.0 + 6.0 * rand_u(&mut c.seed),
            from: me.p,
            slow: 0.0,
        };
    }
    // Stuck (slower than a walk with the throttle on): reverse out.
    c.slow = if speed.abs() < 0.8
        && input.forward > 0.0
        && matches!(c.plan, Plan::Hunt | Plan::Jump { .. })
    {
        c.slow + dt
    } else {
        0.0
    };
    if c.slow > 1.2 {
        c.plan = Plan::Reverse {
            hit: false,
            t: 0.0,
            min: 1.5,
            dist: 4.0 + 4.0 * rand_u(&mut c.seed),
            from: me.p,
            slow: 0.0,
        };
        c.slow = 0.0;
    }
    match &mut c.plan {
        Plan::Reverse {
            t,
            min,
            dist,
            from,
            slow,
            ..
        } => {
            *t += dt;
            // The wheel turned the other way, so the nose swings toward the target.
            input = DriveInput {
                forward: -1.0,
                steer: -steer,
                brake: 0.0,
                handbrake: 0.0,
            };
            *slow = if speed.abs() < 0.6 { *slow + dt } else { 0.0 };
            let far = flat(me.p - *from).length() >= *dist;
            if *slow > 1.5 {
                // Wedged backwards too: forward on full lock, away from the target.
                c.plan = Plan::Wiggle(1.0 + 0.5 * rand_u(&mut c.seed));
            } else if (*t >= *min && far) || *t >= REVERSE_MOST {
                c.plan = Plan::Hunt;
            }
        }
        Plan::Wiggle(t) => {
            *t -= dt;
            input = DriveInput {
                forward: 1.0,
                steer: if steer > 0.0 { -1.0 } else { 1.0 },
                brake: 0.0,
                handbrake: 0.0,
            };
            if *t <= 0.0 {
                c.plan = Plan::Hunt;
            }
        }
        _ => {}
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Past their start delays.
    fn started(ai: &mut [DerbyAi]) {
        for a in ai {
            a.delay = 0.0;
        }
    }

    fn pose(x: f32, z: f32, fwd: Vec3) -> Option<Pose> {
        Some(Pose {
            p: Vec3::new(x, 0.5, z),
            fwd,
            up: Vec3::Y,
            speed: 0.0,
        })
    }

    #[test]
    fn derby_drivers_are_deterministic_from_their_seed() {
        let cars: Vec<DerbyCarView> = [(0.0, -20.0), (20.0, 0.0), (0.0, 20.0), (-20.0, 0.0)]
            .iter()
            .map(|&(x, z)| DerbyCarView {
                pose: pose(x, z, Vec3::new(-x, 0.0, -z).normalize()),
                alive: true,
            })
            .collect();
        let arena = AiArena {
            radius: 34.0,
            ..AiArena::default()
        };
        let run = |seed| {
            let mut ai = derby_drivers(4, seed);
            (0..200)
                .flat_map(|_| {
                    derby_inputs(&mut ai, &cars, &arena, Some(0), Some(0), 0.5, 1.0 / 60.0)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn a_driver_hunts_a_car_and_drives_at_it() {
        // Car 1 faces +z; car 0 is ahead and to its right (−x is right when facing +z
        // with +y up: right = fwd × up).
        let cars = vec![
            DerbyCarView {
                pose: pose(-10.0, 30.0, Vec3::Z),
                alive: true,
            },
            DerbyCarView {
                pose: pose(0.0, 0.0, Vec3::Z),
                alive: true,
            },
        ];
        let arena = AiArena {
            radius: 100.0,
            ..AiArena::default()
        };
        let mut ai = derby_drivers(2, 1);
        started(&mut ai);
        let i = derby_inputs(&mut ai, &cars, &arena, Some(0), Some(0), 1.0, 1.0 / 60.0)[1].unwrap();
        assert_eq!(ai[1].target(), Some(0));
        assert!(i.forward > 0.0, "{i:?}");
        let right = Vec3::Z.cross(Vec3::Y);
        assert_eq!(
            i.steer > 0.0,
            Vec3::new(-10.0, 0.0, 30.0).dot(right) > 0.0,
            "{i:?}"
        );
    }

    #[test]
    fn out_and_flipped_cars_get_no_input() {
        let mut flipped = pose(0.0, 0.0, Vec3::Z).unwrap();
        flipped.up = -Vec3::Y;
        let cars = vec![
            DerbyCarView {
                pose: pose(5.0, 5.0, Vec3::Z),
                alive: true,
            },
            DerbyCarView {
                pose: Some(flipped),
                alive: true,
            },
            DerbyCarView {
                pose: pose(-5.0, 5.0, Vec3::Z),
                alive: false,
            },
        ];
        let mut ai = derby_drivers(3, 3);
        started(&mut ai);
        let i = derby_inputs(&mut ai, &cars, &AiArena::default(), None, None, 0.5, 0.1);
        assert_eq!(i[1], Some(DriveInput::default()));
        assert_eq!(i[2], Some(DriveInput::default()));
        assert!(i[0].unwrap().forward > 0.0);
    }

    #[test]
    fn a_stuck_driver_reverses_out() {
        let cars = vec![
            DerbyCarView {
                pose: pose(0.0, 0.0, Vec3::Z),
                alive: true,
            },
            DerbyCarView {
                pose: pose(0.0, 30.0, -Vec3::Z),
                alive: true,
            },
        ];
        let mut ai = derby_drivers(2, 5);
        let arena = AiArena {
            radius: 100.0,
            ..AiArena::default()
        };
        let mut reversed = false;
        for _ in 0..240 {
            let i = derby_inputs(&mut ai, &cars, &arena, None, None, 0.5, 1.0 / 60.0);
            reversed |= i[0].unwrap().forward < 0.0;
        }
        assert!(
            reversed,
            "a car held still with the throttle on never reversed"
        );
    }
}
