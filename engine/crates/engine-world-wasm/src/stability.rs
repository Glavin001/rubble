//! The pages' stop on an unstable simulation ([`StabilityGuard`], [`VehicleHealth`]).
//!
//! **Detection and stop only. Not a fix.** The guard never clamps, damps, caps,
//! rescales or writes any simulation state, parameter or input: it only reads the
//! vehicles' node state back (a small readback once per displayed frame, never
//! waited for) and tells the page whether to go on submitting steps. A run with the
//! guard and a run without it are the same bit for bit (the test
//! `the_guard_does_not_change_a_run`). There is no speed limit: a valid simulation
//! at any speed never trips it.
//!
//! Why it exists: an explicit vehicle that has gone numerically unstable (energy
//! from nowhere, a body kilometres across, NaN) makes every later step slower on
//! the GPU, since its contact and self-collision lists grow with its box. A page
//! that keeps submitting steps then queues more GPU work than the GPU finishes and
//! the browser loses the device (engine/docs/gpu-overload.md, failure mode (c)).
//! Submitted work cannot be taken back, so the page must stop submitting as soon
//! as the state is seen to be impossible.
//!
//! It trips only on states correct physics cannot reach, and says which:
//! - a node's position, velocity or mass is not a finite number;
//! - the vehicles' kinetic energy is far past what the scene can have supplied
//!   (their energy at the start, what the page launched them with, the work of
//!   gravity over the height they fell, and what their tyres can have delivered at
//!   the speeds they drove), for several readbacks in a row;
//! - the vehicles' kinetic energy rises again after it has fallen (an impact, coming
//!   to rest) by more than a rebound and the scene's sources since then can explain,
//!   for several readbacks in a row;
//! - a vehicle's bounding box is far past its size while its kinetic energy rises.
//!
//! A trip is an **engine error**: the simulation has a bug (or content the solver
//! cannot integrate at its step), and the message says so, with the numbers, for a
//! report. The root cause still has to be fixed in the solver or the content.
//! A trip is final: the page pauses, keeps the last frame on screen and offers a
//! restart. Nothing resumes by itself.

use wasm_bindgen::prelude::*;

/// The energy budget: the vehicles' kinetic energy may reach `ENERGY_FACTOR` × what
/// the scene's inputs can have supplied, plus `ENERGY_FLOOR` joules. Real runs lose
/// energy (damping, friction, plastic work); 3 × leaves room for what the guard does
/// not see (energy stored in preloaded beams, rigid bodies that fall on a vehicle).
pub const ENERGY_FACTOR: f64 = 3.0;
pub const ENERGY_FLOOR: f64 = 5.0e4;
/// Past the budget for this many readbacks in a row before the guard trips (one
/// readback is a frame: 3 is 50 ms at 60 Hz).
pub const ENERGY_SUSTAIN: u32 = 3;
/// The drive allowance: the most work a vehicle's tyres can have done on it, a
/// traction force of `TRACTION_MAX` × its weight times the distance its centre of
/// mass moved (the integral of its speed over the readbacks). No tyre holds 2 g; a
/// car at rest gets no allowance, one at 100 km/h 790 kW per 1,450 kg. (A fixed power
/// allowance grew without bound: 1 MW over a 35 s drive was 315,000 kJ, and a blow-up
/// from 155 kJ to 7,569 kJ in the 2026-11-13 yard run was nowhere near it.)
pub const TRACTION_MAX: f64 = 2.0;
/// The rebound check, over the last `REBOUND_WINDOW` seconds of readbacks: from the
/// lowest kinetic energy in that window (and the peak before it there), it may
/// come back by at most `REBOUND_SHARE` of what was lost (energy stored in bent
/// beams and tyres: a restitution of 0.7, far springier than a car), plus
/// the sources since the low (gravity's work over the fall since then, the drive
/// allowance, launches) + `REBOUND_FLOOR`. Past that, and past `REBOUND_RISE` × the
/// low, for `REBOUND_SUSTAIN` readbacks in a row, it trips: energy multiplying from
/// nowhere, within its first two or three doublings and long before it reaches the
/// whole-run budget. (The sources are bounds already, so they are not multiplied
/// again here.)
pub const REBOUND_SHARE: f64 = 0.5;
pub const REBOUND_RISE: f64 = 2.0;
pub const REBOUND_FLOOR: f64 = 5.0e4;
pub const REBOUND_SUSTAIN: u32 = 2;
/// A rebound returns its energy within a fraction of a second; a window keeps the
/// sources counted to the ones that acted during the rise (not a whole drive's).
pub const REBOUND_WINDOW: f64 = 1.0;
/// Gravity (m/s²) for the work of a fall.
pub const GRAVITY: f64 = 9.81;
/// The box check: a vehicle's bounding box diagonal past `BOX_FACTOR` × its first
/// (spawn) diagonal + `BOX_MARGIN` metres (a car torn apart with debris thrown 50 m
/// is inside it) while its kinetic energy is above `ENERGY_FLOOR` and `BOX_ENERGY_RISE`
/// × the lowest it has been over the last `REBOUND_WINDOW` seconds (since the last
/// launch) + the work of its fall since that low (a car tumbling on down a mountain,
/// or driven on, may leave a torn-off part far behind).
pub const BOX_FACTOR: f32 = 10.0;
pub const BOX_MARGIN: f32 = 50.0;
pub const BOX_ENERGY_RISE: f64 = 1.5;
/// Vehicles a page's check reads. With more vehicles than this the energy and box
/// checks are off (an unread vehicle can hand a read one its energy) and only the
/// first ones' non-finite check runs.
pub const HEALTH_VEHICLES: usize = 8;

/// One vehicle's state at one readback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Every position, velocity and mass read is a finite number.
    pub finite: bool,
    /// Mass (kg).
    pub mass: f64,
    /// Height of the centre of mass (m, world).
    pub height: f64,
    /// Speed of the centre of mass (m/s): for the report only.
    pub speed: f32,
    /// The bounding box's diagonal (m).
    pub size: f32,
    /// Kinetic energy (J).
    pub kinetic: f64,
}

impl Sample {
    /// From nodes as (position, mass, velocity), positions relative to `origin_y`
    /// in height.
    pub fn from_points(origin_y: f32, points: impl Iterator<Item = ([f32; 3], f32, [f32; 3])>) -> Self {
        let mut finite = origin_y.is_finite();
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        let (mut mass, mut momentum, mut kinetic, mut my) = (0.0f64, [0.0f64; 3], 0.0f64, 0.0f64);
        for (p, m, v) in points {
            if !(p.iter().chain(&v).all(|x| x.is_finite()) && m.is_finite()) {
                finite = false;
                continue;
            }
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
                momentum[k] += f64::from(m) * f64::from(v[k]);
            }
            let v2: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            mass += f64::from(m);
            my += f64::from(m) * f64::from(p[1]);
            kinetic += 0.5 * f64::from(m) * v2;
        }
        let (speed, height) = if mass > 0.0 {
            (
                (momentum.iter().map(|x| x * x).sum::<f64>().sqrt() / mass) as f32,
                f64::from(origin_y) + my / mass,
            )
        } else {
            (0.0, f64::from(origin_y))
        };
        let size = if lo[0] <= hi[0] {
            (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f32>().sqrt()
        } else {
            0.0
        };
        Self {
            finite: finite && speed.is_finite() && size.is_finite() && kinetic.is_finite() && height.is_finite(),
            mass,
            height,
            speed,
            size,
            kinetic,
        }
    }

    /// From a vehicle's node records as the solver keeps them (12 words per node:
    /// position relative to the vehicle's origin and mass, velocity, force) and its
    /// origin's height.
    pub fn from_nodes(origin_y: f32, words: &[u32]) -> Self {
        let f = |w: u32| f32::from_bits(w);
        Self::from_points(
            origin_y,
            words
                .chunks_exact(12)
                .map(|n| ([f(n[0]), f(n[1]), f(n[2])], f(n[3]), [f(n[4]), f(n[5]), f(n[6])])),
        )
    }
}

/// Why and when the guard stopped the simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct Trip {
    /// Simulated seconds.
    pub t: f64,
    /// What was seen, in one sentence.
    pub reason: String,
    /// The numbers behind it, one line each.
    pub evidence: Vec<String>,
}

impl Trip {
    /// The page's panel text: the headline, then the evidence lines.
    pub fn message(&self) -> String {
        let mut out = format!(
            "ENGINE ERROR: instability detected at t = {:.3} s: {}. This is a simulation bug, not a limit: the simulation was stopped so the broken state does not overload the GPU. Please report it with the scene link and the numbers below.",
            self.t, self.reason
        );
        for line in &self.evidence {
            out.push('\n');
            out.push_str(line);
        }
        out
    }
}

fn kj(joules: f64) -> String {
    let n = (joules / 1000.0).round() as i64;
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{}{out} kJ", if n < 0 { "-" } else { "" })
}

#[derive(Clone, Debug, Default)]
struct Track {
    /// The first sample: the spawn box diagonal, height and kinetic energy.
    first: Option<Sample>,
    last: Option<Sample>,
    /// The lowest centre of mass so far (m): the fall's work is counted from the start.
    lowest: f64,
    /// The lowest kinetic energy over the last `REBOUND_WINDOW` seconds since the last
    /// launch (J). (Since the start, a car parked at first had a low of 0 J for good:
    /// driven on at 66 km/h with a torn-off part left 100 m behind it, its box tripped
    /// the check; 2026-11-24.)
    kinetic_low: f64,
    /// The centre of mass height at that low (m): the fall since it is a source.
    low_height: f64,
    /// At that low: the read vehicles' mechanical energy together (kinetic + m g h, J)
    /// and the inputs counted so far (launches, external work, J). What one
    /// read vehicle hands another in a collision is in neither's fall and is no
    /// energy from nowhere: the fleet's sum does not rise by it.
    fleet_low: f64,
    sources_low: f64,
    /// The readbacks of the last `REBOUND_WINDOW` seconds since the last launch: (t,
    /// kinetic energy, height, the fleet's mechanical energy, the inputs counted).
    recent: std::collections::VecDeque<(f64, f64, f64, f64, f64)>,
    damage: Option<(u32, u32, u32)>,
}

/// The detector: fed every vehicle's [`Sample`] of a readback, it trips once and
/// stays tripped. It holds no reference to the simulation.
#[derive(Clone, Debug, Default)]
pub struct StabilityGuard {
    vehicles: Vec<Track>,
    /// Vehicles in the scene (more than the tracked ones: energy and box checks off).
    fleet: usize,
    /// Kinetic energy the page's launches may have added (J).
    launched: f64,
    /// Energy the page said something outside the vehicles put in (J): shots, rigs.
    external: f64,
    /// Only the non-finite check runs (the scene has inputs the guard cannot bound).
    unbounded: bool,
    /// The drive allowance so far (J) and the time of the last readback.
    traction: f64,
    last_t: Option<f64>,
    /// Readbacks in a row past the budget, and the energy of the first of them.
    over: u32,
    over_first: (f64, f64),
    /// The rebound check: the readbacks of the last REBOUND_WINDOW seconds (t, kinetic
    /// energy, each vehicle's height, the launches' energy and the drive allowance
    /// then), and the readbacks in a row past the rebound budget.
    window: std::collections::VecDeque<(f64, f64, Vec<f64>, f64, f64)>,
    rebound_over: u32,
    /// Recent (t, kinetic energy, budget), newest last: the trip's evidence.
    history: std::collections::VecDeque<(f64, f64, f64)>,
    trip: Option<Trip>,
}

impl StabilityGuard {
    /// For a scene of `fleet` vehicles, the first [`HEALTH_VEHICLES`] of them read.
    pub fn new(fleet: usize) -> Self {
        Self {
            vehicles: vec![Track::default(); fleet.min(HEALTH_VEHICLES)],
            fleet,
            ..Default::default()
        }
    }

    /// Vehicles whose samples [`Self::observe`] takes.
    pub fn tracked(&self) -> usize {
        self.vehicles.len()
    }

    /// The trip, once tripped.
    pub fn trip(&self) -> Option<&Trip> {
        self.trip.as_ref()
    }

    /// The page launched the vehicles at `speed` (m/s): that kinetic energy is an
    /// input of the scene.
    pub fn launched(&mut self, speed: f32) {
        let speed = f64::from(speed);
        if !speed.is_finite() {
            return;
        }
        for v in &mut self.vehicles {
            if let Some(s) = v.last {
                // (Added to what it has: an upper bound of what the launch gives.)
                self.launched += 0.5 * s.mass * speed * speed;
                v.kinetic_low = f64::INFINITY;
                v.recent.clear();
            }
        }
        // The rebound check starts over from the next readback.
        self.window.clear();
        self.rebound_over = 0;
    }

    /// Something outside the vehicles put up to `joules` into them (a shot's kinetic
    /// energy, a plate's work): an input of the scene from now on.
    pub fn supplied(&mut self, joules: f64) {
        if joules.is_finite() && joules > 0.0 {
            self.external += joules;
        }
    }

    /// The scene has an input the guard cannot bound right now (a hand's grab, a
    /// pulling rig): the rebound check starts over from the next readback. Called
    /// every frame while the input acts, it keeps that check off; the others stay on.
    pub fn disturbed(&mut self) {
        self.window.clear();
        self.rebound_over = 0;
    }

    /// The scene has inputs the guard cannot bound for good (rigid bodies shot or
    /// pressed at the vehicles, a pulling rig): only the non-finite check runs.
    pub fn set_unbounded(&mut self, on: bool) {
        self.unbounded = on;
    }

    /// Vehicle `vehicle`'s damage as the page read it (beams broken, bent, in all):
    /// for the report only.
    pub fn damage(&mut self, vehicle: usize, broken: u32, bent: u32, beams: u32) {
        if let Some(v) = self.vehicles.get_mut(vehicle) {
            v.damage = Some((broken, bent, beams));
        }
    }

    fn describe(&self, samples: &[Sample]) -> Vec<String> {
        let mut out = Vec::new();
        for (i, (s, v)) in samples.iter().zip(&self.vehicles).enumerate() {
            let damage = v
                .damage
                .map_or(String::new(), |(broken, bent, beams)| format!(", {broken} broken and {bent} bent of {beams} beams (last read)"));
            let first = v.first.unwrap_or(*s);
            out.push(format!(
                "vehicle {i}: finite {}, speed {:.1} km/h, kinetic energy {} (at the start {}), box {:.2} m (at the start {:.2} m), centre of mass height {:.2} m (at the start {:.2} m), mass {:.1} kg{damage}",
                s.finite,
                s.speed * 3.6,
                kj(s.kinetic),
                kj(first.kinetic),
                s.size,
                first.size,
                s.height,
                first.height,
                s.mass
            ));
        }
        if !self.history.is_empty() {
            out.push(format!(
                "kinetic energy of the vehicles read (t s: energy / budget): {}",
                self.history
                    .iter()
                    .map(|(t, e, b)| format!("{t:.3}: {} / {}", kj(*e), kj(*b)))
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
        }
        out
    }

    fn stop(&mut self, t: f64, reason: String, samples: &[Sample]) -> Option<&Trip> {
        self.trip = Some(Trip {
            t,
            reason,
            evidence: self.describe(samples),
        });
        self.trip.as_ref()
    }

    /// The tracked vehicles' samples of one readback (`samples[i]`: vehicle `i`), at
    /// simulated time `t` (s). Returns the trip if the guard is (now or already)
    /// tripped.
    pub fn observe(&mut self, t: f64, samples: &[Sample]) -> Option<&Trip> {
        if self.trip.is_some() || samples.len() != self.vehicles.len() {
            return self.trip.as_ref();
        }
        if let Some(i) = samples.iter().position(|s| !s.finite) {
            return self.stop(
                t,
                format!("a node of vehicle {i} has a position, velocity or mass that is not a finite number"),
                samples,
            );
        }
        // The drive allowance: each vehicle's weight × TRACTION_MAX × the distance its
        // centre of mass moved since the last readback (the mean of the two speeds).
        if let Some(t0) = self.last_t {
            let dt = (t - t0).max(0.0);
            for (s, v) in samples.iter().zip(&self.vehicles) {
                let before = v.last.map_or(s.speed, |l| l.speed);
                self.traction += TRACTION_MAX * s.mass * GRAVITY * 0.5 * f64::from(before + s.speed) * dt;
            }
        }
        self.last_t = Some(t);
        let fleet_energy: f64 = samples.iter().map(|s| s.kinetic + s.mass * GRAVITY * s.height).sum();
        // (Launches and external work: the box check never counted traction, whose
        // allowance over a second at speed is larger than a blow-up's rise.)
        let sources_now = self.launched + self.external;
        for (s, v) in samples.iter().zip(&mut self.vehicles) {
            if v.first.is_none() {
                v.first = Some(*s);
                v.lowest = s.height;
                v.kinetic_low = s.kinetic;
                v.low_height = s.height;
                v.fleet_low = fleet_energy;
                v.sources_low = sources_now;
            }
            v.last = Some(*s);
            v.lowest = v.lowest.min(s.height);
            // The low over the last REBOUND_WINDOW seconds (a launch clears the window).
            v.recent.push_back((t, s.kinetic, s.height, fleet_energy, sources_now));
            while v.recent.front().is_some_and(|r| r.0 < t - REBOUND_WINDOW) {
                v.recent.pop_front();
            }
            if let Some(&(_, k, h, f, src)) = v.recent.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
                v.kinetic_low = k;
                v.low_height = h;
                v.fleet_low = f;
                v.sources_low = src;
            }
        }
        if self.fleet > self.vehicles.len() || self.unbounded {
            // Unread vehicles, or bodies and rigs the guard does not read, can hand
            // these their energy (and throw their parts far): no energy or box check.
            return None;
        }
        // What the scene can have supplied: the start's energy, the launches and other
        // reported inputs, the fall's work (from each vehicle's start to its lowest
        // point) and the drive allowance.
        let supplied: f64 = self.launched
            + self.external
            + self
                .vehicles
                .iter()
                .filter_map(|v| v.first.map(|f| f.kinetic + f.mass * GRAVITY * (f.height - v.lowest).max(0.0)))
                .sum::<f64>()
            + self.traction;
        let budget = ENERGY_FACTOR * supplied + ENERGY_FLOOR;
        let kinetic: f64 = samples.iter().map(|s| s.kinetic).sum();
        self.history.push_back((t, kinetic, budget));
        while self.history.len() > 8 {
            self.history.pop_front();
        }
        if kinetic > budget {
            if self.over == 0 {
                self.over_first = (t, kinetic);
            }
            self.over += 1;
            if self.over >= ENERGY_SUSTAIN {
                let before = self.history.iter().rev().find(|h| h.1 <= h.2).map_or(supplied, |h| h.1);
                return self.stop(
                    t,
                    format!(
                        "the vehicles' kinetic energy rose from {} to {}, past {} ({ENERGY_FACTOR} × the {} the scene can have supplied + {}), for {ENERGY_SUSTAIN} readbacks since t = {:.3} s",
                        kj(before),
                        kj(kinetic),
                        kj(budget),
                        kj(supplied),
                        kj(ENERGY_FLOOR),
                        self.over_first.0
                    ),
                    samples,
                );
            }
        } else {
            self.over = 0;
        }
        // The rebound check: from the lowest energy of the last REBOUND_WINDOW seconds.
        self.window
            .push_back((t, kinetic, samples.iter().map(|s| s.height).collect(), self.launched, self.traction));
        while self.window.front().is_some_and(|w| w.0 < t - REBOUND_WINDOW) {
            self.window.pop_front();
        }
        let at = (0..self.window.len())
            .min_by(|&a, &b| self.window[a].1.total_cmp(&self.window[b].1))
            .unwrap_or(0);
        if at + 1 == self.window.len() {
            self.rebound_over = 0;
        } else {
            let (t_low, low, heights, launched, traction) = &self.window[at];
            let peak = self.window.iter().take(at + 1).map(|w| w.1).fold(*low, f64::max);
            let fall: f64 = samples
                .iter()
                .zip(heights)
                .map(|(s, h)| s.mass * GRAVITY * (h - s.height).max(0.0))
                .sum();
            let sources = fall + (self.traction - traction) + (self.launched - launched) + self.external;
            let allowed = low + REBOUND_SHARE * (peak - low).max(0.0) + sources + REBOUND_FLOOR;
            if kinetic > allowed && kinetic > REBOUND_RISE * low {
                self.rebound_over += 1;
                if self.rebound_over >= REBOUND_SUSTAIN {
                    let reason = format!(
                        "the vehicles' kinetic energy rose from {} (its low at t = {t_low:.3} s, after a peak of {}) to {}, past the {} a rebound ({REBOUND_SHARE} × the energy lost) and the scene's sources since then ({}: the fall, what the tyres can have delivered, launches and reported inputs) can explain, for {REBOUND_SUSTAIN} readbacks",
                        kj(*low),
                        kj(peak),
                        kj(kinetic),
                        kj(allowed),
                        kj(sources)
                    );
                    return self.stop(t, reason, samples);
                }
            } else {
                self.rebound_over = 0;
            }
        }
        for (i, (s, v)) in samples.iter().zip(&self.vehicles).enumerate() {
            let rest = v.first.map_or(s.size, |f| f.size);
            let limit = BOX_FACTOR * rest + BOX_MARGIN;
            // The fall since the low is a source of the rise (a car tumbling on down a
            // mountain, a torn-off part left far behind it).
            let fall = s.mass * GRAVITY * (v.low_height - s.height).max(0.0);
            // A read vehicle struck by another gains kinetic energy its own fall does
            // not explain, and the other lost it: the read vehicles' mechanical energy
            // together has not risen. Only a rise of that sum past the inputs since
            // the low (and the floor) is energy from nowhere. (One vehicle: its own
            // rise less its fall, as before.)
            let fleet_rise = fleet_energy - v.fleet_low - (sources_now - v.sources_low);
            if s.size > limit && s.kinetic > ENERGY_FLOOR && s.kinetic > BOX_ENERGY_RISE * v.kinetic_low + fall && fleet_rise > ENERGY_FLOOR {
                return self.stop(
                    t,
                    format!(
                        "vehicle {i} is {:.0} m across (past {limit:.0} m: {BOX_FACTOR} × its {rest:.2} m at the start + {BOX_MARGIN} m) while its kinetic energy rose from {} to {}",
                        s.size,
                        kj(v.kinetic_low),
                        kj(s.kinetic)
                    ),
                    samples,
                );
            }
        }
        None
    }
}

/// A page's check of its explicit vehicles ([`StabilityGuard`] on a readback of
/// their nodes): `ExplicitSim.health()`, `WebWorld.explicitHealth()`. Read-only.
///
/// Each displayed frame the page calls [`Self::check`] (while one is in flight it
/// answers at once) and stops stepping for good when it or [`Self::stopped`] gives
/// a message. The page shows the message as an engine error with a restart
/// button; nothing resumes by itself.
#[wasm_bindgen]
pub struct VehicleHealth {
    device: wgpu::Device,
    queue: wgpu::Queue,
    nodes: wgpu::Buffer,
    misc: wgpu::Buffer,
    /// Per vehicle read: its first node and node count.
    spans: Vec<(u32, u32)>,
    guard: std::rc::Rc<std::cell::RefCell<StabilityGuard>>,
    busy: std::rc::Rc<std::cell::Cell<bool>>,
}

impl VehicleHealth {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &engine_webgpu::ExplicitBeamsGpu) -> Self {
        let all = &gpu.layout().spans;
        let guard = StabilityGuard::new(all.len());
        let spans: Vec<(u32, u32)> = all.iter().take(guard.tracked()).map(|s| (s.node_off, s.nodes)).collect();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            nodes: gpu.nodes_buffer().clone(),
            misc: gpu.misc_buffer().clone(),
            guard: std::rc::Rc::new(std::cell::RefCell::new(guard)),
            spans,
            busy: Default::default(),
        }
    }
}

#[wasm_bindgen]
impl VehicleHealth {
    /// Read the vehicles' nodes as the queue leaves them and judge them, `t` the
    /// simulated time (s) of the steps submitted so far: a promise of the stop
    /// message, or `undefined` while all is well. Never waits in place. While a
    /// check is in flight (or the device is lost) it resolves at once with what is
    /// known.
    pub fn check(&self, t: f64) -> js_sys::Promise {
        if self.busy.get() || crate::device_lost() || self.guard.borrow().trip().is_some() {
            return js_sys::Promise::resolve(&self.stopped().map_or(JsValue::UNDEFINED, JsValue::from));
        }
        self.busy.set(true);
        // Per vehicle its origin record (16 bytes), then its nodes (48 bytes each).
        let ranges = self
            .spans
            .iter()
            .enumerate()
            .flat_map(|(i, &(off, n))| {
                [
                    (self.misc.clone(), i as u64 * 16, 16),
                    (self.nodes.clone(), u64::from(off) * 48, u64::from(n) * 48),
                ]
            })
            .collect();
        let (spans, guard, busy) = (self.spans.clone(), self.guard.clone(), self.busy.clone());
        let read = crate::read::read_ranges_on(self.device.clone(), self.queue.clone(), ranges, move |w| {
            let mut at = 0;
            let samples: Vec<Sample> = spans
                .iter()
                .map(|&(_, n)| {
                    let origin_y = f32::from_bits(w[at + 1]);
                    let words = &w[at + 4..at + 4 + n as usize * 12];
                    at += 4 + n as usize * 12;
                    Sample::from_nodes(origin_y, words)
                })
                .collect();
            let mut guard = guard.borrow_mut();
            Ok(guard.observe(t, &samples).map_or(JsValue::UNDEFINED, |trip| trip.message().into()))
        });
        // The in-flight mark is cleared however the read ends.
        wasm_bindgen_futures::future_to_promise(async move {
            let r = wasm_bindgen_futures::JsFuture::from(read).await;
            busy.set(false);
            r
        })
    }

    /// A vehicle's damage as the page read it (`readDamage`): for the report only.
    pub fn damage(&self, vehicle: u32, broken: u32, bent: u32, beams: u32) {
        self.guard.borrow_mut().damage(vehicle as usize, broken, bent, beams);
    }

    /// The page launched the vehicles at `speed` (m/s): energy the scene supplied.
    pub fn launched(&self, speed: f32) {
        self.guard.borrow_mut().launched(speed);
    }

    /// Something outside the vehicles put up to `joules` into them (a shot's kinetic
    /// energy): an input of the scene from now on.
    pub fn supplied(&self, joules: f64) {
        self.guard.borrow_mut().supplied(joules);
    }

    /// An input the guard cannot bound acts this frame (a hand's grab): the rebound
    /// check starts over. Call it every frame while the input acts.
    pub fn disturbed(&self) {
        self.guard.borrow_mut().disturbed();
    }

    /// The scene has inputs the guard cannot bound (rigid bodies shot or pressed at the
    /// vehicles, a pulling rig): only the non-finite check runs while `on`.
    #[wasm_bindgen(js_name = setUnbounded)]
    pub fn set_unbounded(&self, on: bool) {
        self.guard.borrow_mut().set_unbounded(on);
    }

    /// The stop message once the guard has tripped or the device is lost
    /// (`undefined` otherwise). Final: a page that sees one never steps again.
    pub fn stopped(&self) -> Option<String> {
        if let Some(trip) = self.guard.borrow().trip() {
            return Some(trip.message());
        }
        crate::device_lost().then(|| {
            format!(
                "The browser lost the GPU device ({}). Nothing on this page is running; the simulation was not restarted.",
                crate::gpu_error()
            )
        })
    }
}

#[cfg(test)]
#[path = "stability_tests.rs"]
mod tests;
