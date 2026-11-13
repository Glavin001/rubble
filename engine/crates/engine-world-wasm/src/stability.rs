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
//!   gravity over the height they fell, and a drive allowance), for several
//!   readbacks in a row;
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
/// The drive allowance: work a vehicle's own drive may have done, watts × the
/// simulated time. 1 MW (1,341 hp) is past every vehicle the engine ships.
pub const DRIVE_POWER_MAX: f64 = 1.0e6;
/// Gravity (m/s²) for the work of a fall.
pub const GRAVITY: f64 = 9.81;
/// The box check: a vehicle's bounding box diagonal past `BOX_FACTOR` × its first
/// (spawn) diagonal + `BOX_MARGIN` metres (a car torn apart with debris thrown 50 m
/// is inside it) while its kinetic energy is above `ENERGY_FLOOR` and `BOX_ENERGY_RISE`
/// × the lowest it has been since the last launch.
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
    /// The lowest kinetic energy since the start or the last launch (J).
    kinetic_low: f64,
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
    /// Readbacks in a row past the budget, and the energy of the first of them.
    over: u32,
    over_first: (f64, f64),
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
            }
        }
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
        for (s, v) in samples.iter().zip(&mut self.vehicles) {
            if v.first.is_none() {
                v.first = Some(*s);
                v.lowest = s.height;
                v.kinetic_low = s.kinetic;
            }
            v.last = Some(*s);
            v.lowest = v.lowest.min(s.height);
            v.kinetic_low = v.kinetic_low.min(s.kinetic);
        }
        if self.fleet > self.vehicles.len() {
            // Unread vehicles can hand these their energy: no energy or box check.
            return None;
        }
        // What the scene can have supplied: the start's energy, the launches, the
        // fall's work (from each vehicle's start to its lowest point) and the drive
        // allowance.
        let supplied: f64 = self.launched
            + self
                .vehicles
                .iter()
                .filter_map(|v| v.first.map(|f| f.kinetic + f.mass * GRAVITY * (f.height - v.lowest).max(0.0)))
                .sum::<f64>()
            + DRIVE_POWER_MAX * self.vehicles.len() as f64 * t.max(0.0);
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
        for (i, (s, v)) in samples.iter().zip(&self.vehicles).enumerate() {
            let rest = v.first.map_or(s.size, |f| f.size);
            let limit = BOX_FACTOR * rest + BOX_MARGIN;
            if s.size > limit && s.kinetic > ENERGY_FLOOR && s.kinetic > BOX_ENERGY_RISE * v.kinetic_low {
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
mod tests {
    use super::*;

    const MASS: f64 = 1450.0;

    /// A 1,450 kg car: centre of mass at `speed` m/s and `height` m, `size` m across,
    /// with `extra` joules of kinetic energy besides (vibration, debris).
    fn car(speed: f32, height: f64, size: f32, extra: f64) -> Sample {
        Sample {
            finite: true,
            mass: MASS,
            height,
            speed,
            size,
            kinetic: 0.5 * MASS * f64::from(speed) * f64::from(speed) + extra,
        }
    }

    /// `n` nodes on a line `length` long, all at velocity `v`, 10 kg each.
    fn nodes(n: usize, length: f32, v: [f32; 3]) -> Vec<u32> {
        (0..n)
            .flat_map(|i| {
                let x = length * i as f32 / (n - 1) as f32;
                [x, 0.5, 0.0, 10.0, v[0], v[1], v[2], 0.0, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits)
            })
            .collect()
    }

    #[test]
    fn sample_from_nodes() {
        let s = Sample::from_nodes(2.0, &nodes(5, 4.0, [30.0, 0.0, 40.0]));
        assert!(s.finite);
        assert!((s.speed - 50.0).abs() < 1e-3);
        assert!((s.size - 4.0).abs() < 1e-5);
        assert!((s.kinetic - 0.5 * 50.0 * 2500.0).abs() < 1e-6);
        assert!((s.mass - 50.0).abs() < 1e-9 && (s.height - 2.5).abs() < 1e-6);
        for (word, bad) in [(12 + 5, f32::NAN), (24, f32::INFINITY), (3, f32::NAN)] {
            let mut w = nodes(5, 4.0, [0.0; 3]);
            w[word] = bad.to_bits();
            assert!(!Sample::from_nodes(0.0, &w).finite, "word {word}");
        }
        assert!(!Sample::from_nodes(f32::NAN, &nodes(5, 4.0, [0.0; 3])).finite);
    }

    /// A 200 km/h pole crash as the explicit solver's runs look: the car stops in
    /// 80 ms, parts fly 20 m, and the kinetic energy only falls. Then 10 s at rest.
    #[test]
    fn a_healthy_crash_at_200_kmh_does_not_trip() {
        let mut g = StabilityGuard::new(1);
        let v0 = 200.0 / 3.6;
        for frame in 0..660 {
            let t = frame as f64 / 60.0;
            let crash = ((t - 0.5) / 0.08).clamp(0.0, 1.0) as f32;
            let speed = v0 * (1.0 - crash);
            // A tenth of the lost energy in debris and vibration for a while.
            let extra = if crash > 0.0 && t < 1.5 { 0.1 * 0.5 * MASS * f64::from(v0 * v0) } else { 0.0 };
            let size = 5.0 + 20.0 * ((t - 0.5).max(0.0) as f32).min(1.0);
            assert_eq!(g.observe(t, &[car(speed, 0.6, size, extra)]), None, "t = {t}");
            g.damage(0, (600.0 * crash) as u32, (1900.0 * crash) as u32, 4228);
        }
        assert!(g.trip().is_none());
    }

    /// No speed limit: a valid run at any speed, a launch, driving and a long fall
    /// all stay inside the budget.
    #[test]
    fn fast_runs_launches_driving_and_falls_do_not_trip() {
        // Spawned at 3,000 km/h, coasting.
        let mut g = StabilityGuard::new(1);
        for k in 0..100 {
            assert_eq!(g.observe(k as f64 / 60.0, &[car(3000.0 / 3.6, 0.6, 5.0, 0.0)]), None);
        }
        // At rest, then the page launches it at 1,000 km/h.
        let mut g = StabilityGuard::new(1);
        assert_eq!(g.observe(0.0, &[car(0.0, 0.6, 5.0, 0.0)]), None);
        g.launched(1000.0 / 3.6);
        for k in 1..100 {
            assert_eq!(g.observe(k as f64 / 60.0, &[car(1000.0 / 3.6, 0.6, 5.0, 0.0)]), None);
        }
        // 0 to 300 km/h in 10 s under its own power (700 kW at the end).
        let mut g = StabilityGuard::new(1);
        for k in 0..=600 {
            let t = k as f64 / 60.0;
            assert_eq!(g.observe(t, &[car((300.0 / 3.6 * t / 10.0) as f32, 0.6, 5.0, 0.0)]), None, "t = {t}");
        }
        // Off a 500 m cliff from rest: free fall, then the landing throws parts 40 m.
        let mut g = StabilityGuard::new(1);
        for k in 0..=660 {
            let t = k as f64 / 60.0;
            let fall = (0.5 * GRAVITY * t * t).min(500.0);
            let speed = if fall < 500.0 { (GRAVITY * t) as f32 } else { 0.0 };
            let size = if fall < 500.0 { 5.0 } else { 45.0 };
            assert_eq!(g.observe(t, &[car(speed, 500.6 - fall, size, 0.0)]), None, "t = {t}");
        }
        // A ram: car 0 at 200 km/h hits car 1 at rest; car 1 leaves at 150 km/h.
        let mut g = StabilityGuard::new(2);
        for k in 0..120 {
            let hit = k >= 30;
            let cars = [
                car(if hit { 40.0 / 3.6 } else { 200.0 / 3.6 }, 0.6, 5.0, 0.0),
                car(if hit { 150.0 / 3.6 } else { 0.0 }, 0.6, if hit { 70.0 } else { 5.0 }, 0.0),
            ];
            assert_eq!(g.observe(k as f64 / 60.0, &cars), None, "frame {k}");
        }
    }

    #[test]
    fn a_nan_trips_at_once_and_for_good() {
        let mut g = StabilityGuard::new(1);
        assert_eq!(g.observe(0.0, &[Sample::from_nodes(0.0, &nodes(8, 4.0, [50.0, 0.0, 0.0]))]), None);
        let mut w = nodes(8, 4.0, [50.0, 0.0, 0.0]);
        w[3 * 12 + 1] = f32::NAN.to_bits();
        let trip = g.observe(1.05, &[Sample::from_nodes(0.0, &w)]).cloned().expect("tripped");
        assert!(trip.reason.contains("not a finite number"), "{}", trip.reason);
        let message = trip.message();
        assert!(message.starts_with("ENGINE ERROR: instability detected at t = 1.050 s"), "{message}");
        assert!(message.contains("This is a simulation bug") && message.contains("vehicle 0: finite false"), "{message}");
        // Final: a healthy sample afterwards changes nothing.
        assert_eq!(g.observe(1.1, &[car(50.0, 0.6, 4.0, 0.0)]), Some(&trip));
    }

    /// The 2026-11-13 side pole run: launched at 186 km/h; the HUD read 4,861 km/h
    /// with 373 broken and 2,017 bent of 4,228 beams when the device was lost.
    #[test]
    fn energy_from_nowhere_trips() {
        let mut g = StabilityGuard::new(1);
        let v0 = 186.0 / 3.6;
        for k in 0..60 {
            assert_eq!(g.observe(k as f64 / 60.0, &[car(v0, 0.6, 5.0, 0.0)]), None);
        }
        g.damage(0, 373, 2017, 4228);
        // The energy grows 1.6 × a frame from the impact on (4,861 km/h is 680 × the start's).
        let mut energy = 0.5 * MASS * f64::from(v0 * v0);
        let mut tripped = None;
        for k in 60..80 {
            energy *= 1.6;
            let t = k as f64 / 60.0;
            if let Some(trip) = g.observe(t, &[car(0.0, 0.6, 6.0, energy)]) {
                tripped = Some((k, energy, trip.clone()));
                break;
            }
        }
        let (k, energy, trip) = tripped.expect("tripped");
        // Past the budget (3 × 1,935 kJ + the drive allowance + the floor) at the 4th
        // frame of the growth, tripped 2 frames later: long before 680 ×.
        assert_eq!(k, 65, "{}", trip.message());
        assert!(energy < 20.0 * 0.5 * MASS * f64::from(v0 * v0));
        assert!(trip.reason.starts_with("the vehicles' kinetic energy rose from"), "{}", trip.reason);
        let message = trip.message();
        assert!(message.contains("373 broken and 2017 bent of 4228 beams"), "{message}");
        assert!(message.contains("kinetic energy of the vehicles read (t s: energy / budget)"), "{message}");
        // One readback past the budget (a reading mid-impact) is not a trip.
        let mut g = StabilityGuard::new(1);
        assert_eq!(g.observe(0.0, &[car(v0, 0.6, 5.0, 0.0)]), None);
        assert_eq!(g.observe(0.1, &[car(v0, 0.6, 5.0, 1.0e8)]), None);
        assert_eq!(g.observe(0.2, &[car(v0, 0.6, 5.0, 1.0e8)]), None);
        assert_eq!(g.observe(0.3, &[car(v0, 0.6, 5.0, 0.0)]), None);
        assert_eq!(g.observe(0.4, &[car(v0, 0.6, 5.0, 1.0e8)]), None);
    }

    #[test]
    fn an_exploding_box_with_rising_energy_trips() {
        let mut g = StabilityGuard::new(1);
        assert_eq!(g.observe(0.0, &[car(30.0, 0.6, 5.0, 0.0)]), None);
        assert_eq!(g.observe(0.5, &[car(5.0, 0.6, 6.0, 0.0)]), None);
        // Debris 240 m away with the energy falling: a torn car, not an instability.
        assert_eq!(g.observe(1.0, &[car(4.0, 0.6, 240.0, 0.0)]), None);
        // The same box with the energy up 20 × since its low (still inside the budget).
        let trip = g.observe(1.1, &[car(18.0, 0.6, 240.0, 0.0)]).cloned().expect("tripped");
        assert!(trip.reason.contains("240 m across (past 100 m"), "{}", trip.reason);
    }

    /// More vehicles than the guard reads: an unread one may hand a read one its
    /// energy, so only the non-finite check runs.
    #[test]
    fn a_large_fleet_checks_only_for_non_finite_values() {
        let mut g = StabilityGuard::new(64);
        assert_eq!(g.tracked(), HEALTH_VEHICLES);
        let mut cars = vec![car(0.0, 0.6, 5.0, 0.0); HEALTH_VEHICLES];
        assert_eq!(g.observe(0.0, &cars), None);
        cars[3] = car(100.0, 0.6, 500.0, 1.0e9);
        for k in 1..10 {
            assert_eq!(g.observe(k as f64 * 0.1, &cars), None);
        }
        cars[5].finite = false;
        assert!(g.observe(1.0, &cars).is_some());
    }

    /// The guard is read-only: a run sampled by it every step is, bit for bit, the
    /// run without it (the CPU reference; on the GPU the guard is a buffer copy and
    /// a map, which write nothing the solver reads).
    #[test]
    fn the_guard_does_not_change_a_run() {
        use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
        let truck = crate::explicit::explicit_sedan_truck(true);
        let spawn = || {
            ExplicitBeams::from_truck(
                &truck,
                glam::Vec3::new(0.0, 0.3, 0.0),
                glam::Vec3::new(0.0, 0.0, 25.0),
                ExplicitBeamsSettings {
                    fast_inv_sqrt: true,
                    ..Default::default()
                },
            )
            .expect("the sedan")
        };
        let bits = |sim: &ExplicitBeams| -> Vec<u32> {
            sim.nodes
                .iter()
                .flat_map(|n| n.pos.to_array().into_iter().chain(n.vel.to_array()).chain(n.force.to_array()))
                .map(f32::to_bits)
                .chain(sim.origin.to_array().map(f32::to_bits))
                .collect()
        };
        let (mut plain, mut guarded) = (spawn(), spawn());
        let mut guard = StabilityGuard::new(1);
        for step in 0..400 {
            plain.step();
            guarded.step();
            let sample = Sample::from_points(
                guarded.origin.y,
                guarded.nodes.iter().map(|n| (n.pos.to_array(), n.mass, n.vel.to_array())),
            );
            assert_eq!(guard.observe(f64::from(step) / 2000.0, &[sample]), None, "step {step}");
            guard.damage(0, 0, 0, guarded.beams.len() as u32);
        }
        assert!(guard.trip().is_none());
        assert_eq!(bits(&plain), bits(&guarded));
    }
}
