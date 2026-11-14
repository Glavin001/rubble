//! Real-time pacing of a fixed-step solver on a display ([`FramePacer`]): how many
//! fixed steps each displayed frame takes. The physics never changes (the same
//! fixed step, every step taken in order); only how much simulated time a frame
//! shows does.
//!
//! - **Accumulator.** Wall time × the time scale adds steps owed; a frame takes
//!   the whole steps and carries the remainder (at most one frame's share: past
//!   the cap the simulation runs slower than real time, never in catch-up bursts).
//! - **Real time first, down to a frame-rate floor** (`min_fps`, BeamNG's order:
//!   the physics keeps real time while the GPU can, the frame rate gives way).
//!   A frame takes at most twice the share of a target frame (the display's
//!   interval, or the floor's when the display is faster than the GPU can keep
//!   up with), and at most the steps whose GPU time (the solver's, measured:
//!   [`FramePacer::report_gpu`]) fits a budget share of that target frame. A
//!   frame later than the target (rendering and other GPU work count too) cuts
//!   the share below what its steps used; later frames grow it back quickly to
//!   just under that, hold it there a few seconds, then try for more (clocks and
//!   other work change; at most one late frame per try). A GPU that cannot keep
//!   real time at the floor then shows the simulation slower at an even frame
//!   rate, reported by [`FramePacer::speed`], instead of bursts of steps between
//!   dropped frames.
//! - **Shown time.** The steps owed but not yet taken (under one) are the
//!   frame's [`FramePacer::lead`]: a renderer carries the state forward by it
//!   (node velocity × lead) so whole steps do not show as uneven motion (in 20×
//!   slow motion a frame takes one or two steps). [`FramePacer::shown_dt`] is
//!   the wall time the frame's shown motion stands for (what a follow camera
//!   eases by): its wall time, less what a cap held back.
//! - **Time scale.** Slow-motion changes ease in log space (into slow motion
//!   within a few frames, out of it over about a second) instead of switching 20×
//!   in one frame.

use wasm_bindgen::prelude::*;

/// Wall seconds a frame counts at most (a stall, a hidden tab).
const MAX_DT: f64 = 0.1;
/// Display intervals the estimate stays within (360 Hz to 20 Hz); the target
/// frame without a frame-rate floor (the most one frame catches up).
const MIN_INTERVAL: f64 = 1.0 / 360.0;
const MAX_INTERVAL: f64 = 1.0 / 20.0;
/// A frame this many target frames long is late; one longer than `STALL` seconds
/// is a stall (loading, a hidden tab), not the GPU's.
const LATE: f64 = 1.2;
const STALL: f64 = 0.25;
/// The budget: the share of the display interval the solver's measured GPU time
/// may fill (at most, at least), its cut when the GPU falls behind, its growth per
/// wall second up to `NEAR` × the share that fell behind, and how long (wall
/// seconds) that share is remembered.
const SHARE_MAX: f64 = 0.9;
const SHARE_MIN: f64 = 0.4;
const SHARE_CUT: f64 = 0.85;
const GROW: f64 = 0.6;
const NEAR: f64 = 0.95;
const REMEMBER: f64 = 4.0;
/// GPU reports the estimate keeps (the page reports every 15 frames: 3 s at 60 Hz).
const REPORTS: usize = 12;
/// Time constants (wall seconds) of a time-scale change: slowing down (an impact's
/// slow motion), speeding up.
const EASE_IN: f64 = 0.04;
const EASE_OUT: f64 = 0.25;

/// The pacer's state (the page keeps one per solver clock).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct FramePacer {
    rate: f64,
    floor: f64,
    acc: f64,
    scale: f64,
    interval: f64,
    ms_per_step: f64,
    reports: Vec<f64>,
    share: f64,
    fell_behind: f64,
    fell_behind_age: f64,
    last_steps: u32,
    limited: bool,
    lead_steps: f64,
    shown_dt: f64,
    taken_avg: f64,
    wanted_avg: f64,
    speed: f64,
}

#[wasm_bindgen]
impl FramePacer {
    /// A pacer for a solver of `rate` steps per simulated second that keeps real
    /// time down to `min_fps` frames per second (0: no GPU budget; real time at any
    /// frame rate down to 20 fps, a GPU that cannot keep up queues frames and the
    /// page drops them).
    #[wasm_bindgen(constructor)]
    pub fn new(rate: f64, min_fps: f64) -> FramePacer {
        FramePacer {
            rate,
            floor: if min_fps > 0.0 {
                (1.0 / min_fps).clamp(MIN_INTERVAL, MAX_INTERVAL)
            } else {
                0.0
            },
            acc: 0.0,
            scale: 1.0,
            interval: 1.0 / 60.0,
            ms_per_step: 0.0,
            reports: Vec::new(),
            share: SHARE_MAX,
            fell_behind: f64::INFINITY,
            fell_behind_age: 0.0,
            last_steps: 0,
            limited: false,
            lead_steps: 0.0,
            shown_dt: 0.0,
            taken_avg: 0.0,
            wanted_avg: 0.0,
            speed: 1.0,
        }
    }

    /// Steps to take this displayed frame: `dt` wall seconds since the last one,
    /// shown `slow` times slower than real time, at most `max` steps.
    pub fn advance(&mut self, dt: f64, slow: f64, max: u32) -> u32 {
        let raw = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        let dt = raw.min(MAX_DT);
        if dt > 0.0 {
            // A frame shorter than the estimate lowers it at once; a longer one (a
            // missed frame is not the display's rate) raises it slowly.
            if dt < self.interval {
                self.interval = dt.max(MIN_INTERVAL);
            } else {
                self.interval = (self.interval + (dt - self.interval) * 0.02).min(MAX_INTERVAL);
            }
        }
        // The time scale eases toward 1 / slow in log space.
        let target = 1.0 / slow.max(1e-3);
        let tau = if target < self.scale {
            EASE_IN
        } else {
            EASE_OUT
        };
        let (ls, lt) = (self.scale.ln(), target.ln());
        let l = ls + (lt - ls) * (1.0 - (-dt / tau).exp());
        self.scale = if (l - lt).abs() < 0.01 {
            target
        } else {
            l.exp()
        };

        self.acc += dt * self.rate * self.scale;
        // The target frame: the display's, or the floor's when slower.
        let budgeted = self.floor > 0.0;
        let frame = self.interval.max(if budgeted {
            self.floor
        } else {
            MAX_INTERVAL / 2.0
        });
        let budget = budgeted && self.ms_per_step > 0.0;
        if budget {
            if raw > LATE * frame && raw < STALL && self.last_steps > 0 {
                // Late: below the share the last frame's steps used.
                let used = f64::from(self.last_steps) * self.ms_per_step / (frame * 1000.0);
                self.fell_behind = self.share.min(used);
                self.fell_behind_age = 0.0;
                self.share = (self.fell_behind * SHARE_CUT).max(SHARE_MIN);
            } else {
                self.fell_behind_age += dt;
                if self.fell_behind_age > REMEMBER {
                    self.fell_behind = f64::INFINITY;
                }
                let near = (NEAR * self.fell_behind).max(SHARE_MIN);
                if self.share < near {
                    self.share = (self.share + GROW * dt).min(near);
                }
                self.share = self.share.min(SHARE_MAX);
            }
        }
        // A target frame's share of steps; a frame takes at most twice it.
        let per = frame * self.rate * self.scale;
        let mut cap = (2.0 * per).ceil().clamp(1.0, f64::from(max.max(1)));
        if budget {
            cap = cap.min(
                (self.share * frame * 1000.0 / self.ms_per_step)
                    .floor()
                    .max(1.0),
            );
        }
        let steps = (self.acc.floor().max(0.0)).min(cap) as u32;
        self.limited = cap < per.floor();
        self.acc -= f64::from(steps);
        self.acc = self.acc.min(per.max(1.0));
        self.last_steps = steps;
        let lead = self.acc.clamp(0.0, 1.0);
        self.shown_dt = (f64::from(steps) + lead - self.lead_steps) / (self.rate * self.scale);
        self.lead_steps = lead;
        // The share of the asked-for simulated time taken (smoothed, a readout).
        let wanted = dt * self.rate * self.scale;
        if wanted > 0.0 {
            let k = 1.0 - (-dt / 0.5).exp();
            self.taken_avg += (f64::from(steps) - self.taken_avg) * k;
            self.wanted_avg += (wanted - self.wanted_avg) * k;
            self.speed = (self.taken_avg / self.wanted_avg).min(1.0);
        }
        steps
    }

    /// The GPU time of solver frames: `ms` milliseconds for a frame of `steps` steps
    /// on average (timestamps, or submit-to-done).
    #[wasm_bindgen(js_name = reportGpu)]
    pub fn report_gpu(&mut self, ms: f64, steps: f64) {
        if !(ms > 0.0 && steps >= 1.0 && ms.is_finite()) {
            return;
        }
        // The least of the recent reports: a GPU that lowers its clocks under a
        // lighter load reports more per step the fewer steps it gets, which would
        // otherwise spiral the cap down (missed frames still cut the budget).
        if self.reports.len() == REPORTS {
            self.reports.remove(0);
        }
        self.reports.push(ms / steps);
        self.ms_per_step = self.reports.iter().copied().fold(f64::INFINITY, f64::min);
    }

    /// Start over: no steps owed, the time scale at `slow` at once (a restart). The
    /// GPU estimate and budget stay (the same GPU).
    pub fn reset(&mut self, slow: f64) {
        self.acc = 0.0;
        self.lead_steps = 0.0;
        self.shown_dt = 0.0;
        self.scale = 1.0 / slow.max(1e-3);
        self.taken_avg = 0.0;
        self.wanted_avg = 0.0;
        self.speed = 1.0;
    }

    /// Simulated seconds past the last step the frame shows: the steps owed but
    /// not taken, at most one.
    pub fn lead(&self) -> f64 {
        self.lead_steps / self.rate
    }

    /// The wall seconds the frame's shown motion stands for (steps and the change
    /// of [`Self::lead`], at the time scale).
    #[wasm_bindgen(js_name = shownDt)]
    pub fn shown_dt(&self) -> f64 {
        self.shown_dt
    }

    /// The time scale in effect (simulated seconds per wall second).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The share of the asked-for simulated time the frames take (1: on pace; less:
    /// slowed by the step cap), smoothed over about half a second.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Whether the step cap held the last frame under its share.
    pub fn limited(&self) -> bool {
        self.limited
    }

    /// The display interval estimate (seconds).
    pub fn interval(&self) -> f64 {
        self.interval
    }

    /// GPU milliseconds per step estimate (0 until reported).
    #[wasm_bindgen(js_name = msPerStep)]
    pub fn ms_per_step(&self) -> f64 {
        self.ms_per_step
    }

    /// The budget's share of the display interval.
    pub fn share(&self) -> f64 {
        self.share
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(p: &mut FramePacer, frames: usize, dt: f64, slow: f64) -> Vec<u32> {
        (0..frames).map(|_| p.advance(dt, slow, 1000)).collect()
    }

    /// A display at `hz` driven by a GPU of `ms` per step (timed) plus `fixed` ms
    /// per frame (rendering: not timed): a frame is shown at the first vsync after
    /// its GPU work, the next one starts then. Steps and frame intervals.
    fn simulate(
        p: &mut FramePacer,
        hz: f64,
        ms: f64,
        fixed: f64,
        frames: usize,
    ) -> (Vec<u32>, Vec<f64>) {
        let (mut steps, mut dts) = (Vec::new(), Vec::new());
        let vsync = 1.0 / hz;
        let mut dt = vsync;
        for i in 0..frames {
            let s = p.advance(dt, 1.0, 1000);
            steps.push(s);
            dts.push(dt);
            let gpu = (fixed + ms * f64::from(s)) / 1000.0;
            dt = (gpu / vsync - 1e-9).ceil().max(1.0) * vsync;
            if i % 15 == 14 && s > 0 {
                p.report_gpu(ms * f64::from(s), f64::from(s));
            }
        }
        (steps, dts)
    }

    #[test]
    fn real_time_on_any_display_rate() {
        for hz in [60.0, 120.0, 144.0, 75.0] {
            let mut p = FramePacer::new(2000.0, 30.0);
            let steps = run(&mut p, (hz * 10.0) as usize, 1.0 / hz, 1.0);
            let total: u32 = steps.iter().sum();
            assert!(
                (f64::from(total) - 20000.0).abs() <= 2000.0 / hz + 1.0,
                "{hz} Hz: {total}"
            );
            // Whole steps per frame: floor or ceil of the share, never bursts.
            let per = 2000.0 / hz;
            assert!(
                steps[5..]
                    .iter()
                    .all(|&s| f64::from(s) >= per.floor() && f64::from(s) <= per.ceil()),
                "{hz} Hz: {steps:?}"
            );
        }
    }

    #[test]
    fn variable_frames_keep_pace() {
        let mut p = FramePacer::new(2000.0, 0.0);
        let dts = [0.016, 0.009, 0.021, 0.0083, 0.0167, 0.033];
        let mut wall = 0.0;
        let mut total = 0u32;
        for i in 0..1200 {
            let dt = dts[i % dts.len()];
            wall += dt;
            total += p.advance(dt, 1.0, 1000);
        }
        assert!(
            (f64::from(total) - wall * 2000.0).abs() < 70.0,
            "{total} vs {}",
            wall * 2000.0
        );
    }

    /// Simulated seconds per wall second over frames `from..`.
    fn pace(steps: &[u32], dts: &[f64], from: usize) -> f64 {
        let s: u32 = steps[from..].iter().sum();
        f64::from(s) / 2000.0 / dts[from..].iter().sum::<f64>()
    }

    #[test]
    fn a_gpu_that_keeps_real_time_at_a_lower_frame_rate_does() {
        // 0.3 ms per step plus 6 ms of rendering on a 120 Hz display: real time
        // fits 40 fps frames, not 120 Hz ones. The physics keeps real time.
        let mut p = FramePacer::new(2000.0, 30.0);
        let (steps, dts) = simulate(&mut p, 120.0, 0.3, 6.0, 1200);
        assert!(
            pace(&steps, &dts, 600) > 0.99,
            "{}",
            pace(&steps, &dts, 600)
        );
        assert!(!p.limited());
    }

    #[test]
    fn a_slow_gpu_slows_evenly_at_the_floor() {
        // 0.6 ms per step plus 4 ms of rendering on a 60 Hz display: real time
        // needs more than the GPU has at any frame rate. At the 30 fps floor the
        // frames are 33 ms (two vsyncs) and the physics runs at ~70 %.
        let mut p = FramePacer::new(2000.0, 30.0);
        simulate(&mut p, 60.0, 0.6, 4.0, 600);
        let (steps, dts) = simulate(&mut p, 60.0, 0.6, 4.0, 1200);
        let late = dts.iter().filter(|&&d| d > 2.5 / 60.0).count();
        // One late frame per try for more (every REMEMBER seconds).
        assert!(late <= 12, "{late} frames late in 40 s");
        let pc = pace(&steps, &dts, 0);
        assert!(pc > 0.6 && pc < 0.8, "{pc}");
        assert!(
            p.limited() && (p.speed() - pc).abs() < 0.1,
            "{} vs {pc}",
            p.speed()
        );
        // The 60 fps floor: every vsync, slower physics.
        let mut q = FramePacer::new(2000.0, 60.0);
        simulate(&mut q, 60.0, 0.6, 4.0, 600);
        let (steps, dts) = simulate(&mut q, 60.0, 0.6, 4.0, 1200);
        assert!(dts.iter().filter(|&&d| d > 1.5 / 60.0).count() <= 12);
        let pq = pace(&steps, &dts, 0);
        assert!(pq > 0.45 && pq < pc, "{pq}");
        // No budget: the frames stretch to 50 ms (the most one frame catches up)
        // and the physics still falls behind.
        let mut r = FramePacer::new(2000.0, 0.0);
        let (_, dts) = simulate(&mut r, 60.0, 0.6, 4.0, 600);
        assert!(dts[100..].iter().all(|&d| d > 2.5 / 60.0));
    }

    #[test]
    fn a_fast_gpu_is_never_capped() {
        let mut p = FramePacer::new(2000.0, 30.0);
        let (steps, dts) = simulate(&mut p, 120.0, 0.2, 1.0, 1200);
        assert!(dts.iter().all(|&d| (d - 1.0 / 120.0).abs() < 1e-9));
        assert!(steps[5..].iter().all(|&s| s == 16 || s == 17));
        assert!(!p.limited() && p.speed() > 0.99);
    }

    #[test]
    fn shown_time_moves_evenly_in_slow_motion() {
        // 20x slower on 60 Hz: 1.67 steps a frame, taken as 1 or 2; the shown
        // time (steps + lead) moves 1/1200 s every frame.
        let mut p = FramePacer::new(2000.0, 0.0);
        p.reset(20.0);
        let mut shown = 0.0;
        let mut taken = 0u32;
        for i in 0..600 {
            taken += p.advance(1.0 / 60.0, 20.0, 1000);
            let t = f64::from(taken) / 2000.0 + p.lead();
            if i > 0 {
                assert!(
                    (t - shown - 1.0 / 1200.0).abs() < 1e-9,
                    "{i}: {}",
                    t - shown
                );
                assert!((p.shown_dt() - 1.0 / 60.0).abs() < 1e-9);
            }
            shown = t;
        }
    }

    #[test]
    fn slow_motion_eases_in_and_out() {
        let mut p = FramePacer::new(2000.0, 0.0);
        run(&mut p, 60, 1.0 / 120.0, 1.0);
        let mut scales = Vec::new();
        for _ in 0..60 {
            p.advance(1.0 / 120.0, 20.0, 1000);
            scales.push(p.scale());
        }
        // Monotone, no 20x jump in one frame, at 1/20 within 0.25 s.
        assert!(scales.windows(2).all(|w| w[1] <= w[0]));
        assert!(scales[0] > 0.3, "{}", scales[0]);
        assert_eq!(scales[30], 1.0 / 20.0);
        for _ in 0..240 {
            p.advance(1.0 / 120.0, 1.0, 1000);
        }
        assert_eq!(p.scale(), 1.0);
        p.reset(20.0);
        assert_eq!(p.scale(), 1.0 / 20.0);
    }
}
