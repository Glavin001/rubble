//! Back-pressure on completed GPU work ([`StepGate`]): a host submits a World step
//! only when the GPU has finished the ones before it.
//!
//! A page that submits by wall time (so many steps a displayed frame) can queue more
//! GPU work a second than the GPU completes. On Apple Silicon a queued kernel cannot
//! be aborted, not even by closing the page, so a queue that grows ends with a lost
//! device and a GPU busy for as long as the queue is deep (the 2026-11-13 incident:
//! `engine/docs/vehicle-lab.md`, "GPU overload"). The gate makes that impossible by
//! construction:
//!
//! - **At most [`StepGate::MAX_IN_FLIGHT`] steps in flight.** A step is in flight from
//!   its submission until its completion has been *observed* (the queue's
//!   `onSubmittedWorkDone`). Completions are only observed between the host's tasks,
//!   so a displayed frame also takes at most that many steps: the hard cap per frame.
//! - **The simulation slows down; it never catches up.** A refused step is not owed.
//! - **Automatic pause.** A step's GPU service time (from the later of its submission
//!   and the completion before it, to its own completion) over the budget twice in a
//!   row pauses the gate with a message, as does a step still not finished after the
//!   stall time. Stepping resumes only when the host says so ([`StepGate::resume`]).
//! - **A lost device stops it for good** ([`StepGate::stop`]).
//!
//! The gate is plain state and takes the time from its caller, so it is tested on the
//! CPU against a simulated slow GPU.

use std::collections::VecDeque;

/// What the gate knows (a readout for the page's HUD).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pace {
    /// Steps submitted whose completion has not been observed.
    pub in_flight: u32,
    /// The last completed step's GPU service time (ms; 0 before the first).
    pub last_ms: f64,
    /// The largest service time since the last [`StepGate::resume`] (ms).
    pub worst_ms: f64,
    /// Steps refused since the gate was made (the simulation ran slower for each).
    pub refused: u64,
    pub paused: bool,
}

#[derive(Clone, Debug)]
pub struct StepGate {
    max_in_flight: u32,
    budget_ms: f64,
    stall_ms: f64,
    /// Submission times of the steps in flight, oldest first.
    flight: VecDeque<f64>,
    /// When the last completion was observed.
    last_done: f64,
    /// Completed steps over the budget in a row.
    over: u32,
    /// The service times of that run (the message).
    over_ms: [f64; 2],
    /// Completions the budget does not judge yet (a World's first steps compile and
    /// upload).
    grace: u32,
    last_ms: f64,
    worst_ms: f64,
    refused: u64,
    paused: Option<String>,
    stopped: bool,
}

impl Default for StepGate {
    fn default() -> Self {
        Self::new()
    }
}

impl StepGate {
    /// Steps in flight at most: one running and one queued behind it (a GPU that keeps
    /// pace never idles between steps), so the work that cannot be taken back is two
    /// steps deep at most.
    pub const MAX_IN_FLIGHT: u32 = 2;
    /// A step's GPU service time budget (ms): over it twice in a row pauses the gate.
    /// Three displayed frames at 60 Hz: far over any step that keeps a page usable.
    pub const BUDGET_MS: f64 = 50.0;
    /// A step in flight this long without completing pauses the gate (ms).
    pub const STALL_MS: f64 = 1000.0;
    /// Completions the budget skips after the gate is made or resumed.
    pub const GRACE: u32 = 2;

    pub fn new() -> Self {
        Self::with(Self::MAX_IN_FLIGHT, Self::BUDGET_MS, Self::STALL_MS)
    }

    /// A gate with its own limits (`max_in_flight` at least 1; tests, and a host that
    /// knows its GPU).
    pub fn with(max_in_flight: u32, budget_ms: f64, stall_ms: f64) -> Self {
        Self {
            max_in_flight: max_in_flight.max(1),
            budget_ms,
            stall_ms,
            flight: VecDeque::new(),
            last_done: f64::NEG_INFINITY,
            over: 0,
            over_ms: [0.0; 2],
            grace: Self::GRACE,
            last_ms: 0.0,
            worst_ms: 0.0,
            refused: 0,
            paused: None,
            stopped: false,
        }
    }

    /// The step budget (ms): a host's explicit setting.
    pub fn set_budget_ms(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.budget_ms = ms;
        }
    }

    pub fn budget_ms(&self) -> f64 {
        self.budget_ms
    }

    /// Whether a step may be submitted at `now` (ms, any monotonic clock): the gate is
    /// not paused or stopped and fewer than the most are in flight. Also where a
    /// stalled GPU is noticed (the oldest step in flight past the stall time).
    pub fn may_step(&mut self, now: f64) -> bool {
        if self.stopped || self.paused.is_some() {
            return false;
        }
        if let Some(&oldest) = self.flight.front() {
            let waited = now - oldest.max(self.last_done);
            if waited > self.stall_ms {
                self.paused = Some(format!(
                    "Paused: the GPU has not finished a World step for {waited:.0} ms ({} in flight; nothing more is submitted until it does). \
                     If the GPU stays busy, close the page and check it before resuming.",
                    self.flight.len()
                ));
                return false;
            }
        }
        if self.flight.len() as u32 >= self.max_in_flight {
            self.refused += 1;
            return false;
        }
        true
    }

    /// A step was submitted at `now`.
    pub fn submitted(&mut self, now: f64) {
        self.flight.push_back(now);
    }

    /// The oldest step in flight was observed complete at `now`.
    pub fn completed(&mut self, now: f64) {
        let Some(submitted) = self.flight.pop_front() else {
            return;
        };
        // The GPU ran this step from the later of its submission and the completion
        // before it (it queued behind that one).
        let ms = (now - submitted.max(self.last_done)).max(0.0);
        self.last_done = now;
        self.last_ms = ms;
        self.worst_ms = self.worst_ms.max(ms);
        if self.grace > 0 {
            self.grace -= 1;
            return;
        }
        if ms > self.budget_ms {
            self.over_ms[(self.over as usize).min(1)] = ms;
            self.over += 1;
            if self.over >= 2 && self.paused.is_none() && !self.stopped {
                self.paused = Some(format!(
                    "Paused: two World steps in a row took {:.0} ms and {:.0} ms of GPU time (budget {:.0} ms). \
                     The scene is more than this GPU runs in real time; nothing more is submitted until you resume.",
                    self.over_ms[0], self.over_ms[1], self.budget_ms
                ));
            }
        } else {
            self.over = 0;
        }
    }

    /// The pause's message (`None`: running).
    pub fn paused(&self) -> Option<&str> {
        self.paused.as_deref()
    }

    /// Step again after a pause (the host's decision: a button). The steps still in
    /// flight stay counted; the budget skips the next completions.
    pub fn resume(&mut self, now: f64) {
        if self.stopped {
            return;
        }
        self.paused = None;
        self.over = 0;
        self.grace = Self::GRACE;
        self.worst_ms = 0.0;
        // The stall clock starts again (a pause is not the GPU's time).
        self.last_done = now;
    }

    /// Stop for good (the device was lost): no step is ever allowed again.
    pub fn stop(&mut self, reason: &str) {
        self.stopped = true;
        self.paused = Some(format!("Stopped: {reason}. Nothing more is submitted to the GPU."));
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn pace(&self) -> Pace {
        Pace {
            in_flight: self.flight.len() as u32,
            last_ms: self.last_ms,
            worst_ms: self.worst_ms,
            refused: self.refused,
            paused: self.paused.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU that runs one step at a time, each taking `cost(k)` ms, and a page that
    /// asks for `want` steps every `frame_ms` (its displayed frame), observing
    /// completions only between frames (as a browser delivers them). Returns the most
    /// steps in flight, the most steps taken in one frame, the steps taken, and the
    /// deepest queue of GPU work (ms) at any submission.
    fn run(gate: &mut StepGate, frames: usize, frame_ms: f64, want: u32, cost: impl Fn(u64) -> f64) -> (u32, u32, u64, f64) {
        let mut gpu_free = 0.0f64;
        let mut done_at: VecDeque<f64> = VecDeque::new();
        let (mut most, mut most_frame, mut taken, mut deepest) = (0u32, 0u32, 0u64, 0.0f64);
        for f in 0..frames {
            let now = f as f64 * frame_ms;
            while done_at.front().is_some_and(|&t| t <= now) {
                // (Observed at the frame, not at the GPU's own time: the worst case for
                // the measured service time.)
                done_at.pop_front();
                gate.completed(now);
            }
            let mut this = 0;
            for _ in 0..want {
                if !gate.may_step(now) {
                    break;
                }
                gate.submitted(now);
                let c = cost(taken);
                gpu_free = gpu_free.max(now) + c;
                done_at.push_back(gpu_free);
                deepest = deepest.max(gpu_free - now);
                taken += 1;
                this += 1;
                most = most.max(gate.pace().in_flight);
            }
            most_frame = most_frame.max(this);
        }
        (most, most_frame, taken, deepest)
    }

    #[test]
    fn a_fast_gpu_keeps_real_time() {
        // 4 ms a step, one step a 60 Hz frame: every step taken, never paused.
        let mut g = StepGate::new();
        let (most, per_frame, taken, _) = run(&mut g, 600, 1000.0 / 60.0, 1, |_| 4.0);
        assert_eq!(taken, 600);
        assert!(most <= 1 && per_frame == 1);
        assert!(g.paused().is_none());
        assert_eq!(g.pace().refused, 0);
    }

    #[test]
    fn a_slow_gpu_never_has_more_than_two_steps_in_flight() {
        // The incident's page: up to 7 steps asked a frame (4 from the pacer, 3 owed),
        // 30 ms of GPU a step (a 60 Hz frame is 16.7 ms). Ungated that queue grows by
        // 7 × 30 − 16.7 ms a frame without bound.
        let mut g = StepGate::new();
        let (most, per_frame, taken, deepest) = run(&mut g, 600, 1000.0 / 60.0, 7, |_| 30.0);
        assert!(most <= StepGate::MAX_IN_FLIGHT, "{most} steps in flight");
        assert!(per_frame <= StepGate::MAX_IN_FLIGHT, "{per_frame} steps in one frame");
        // Never more GPU work queued than two steps' worth.
        assert!(deepest <= 2.0 * 30.0 + 1e-9, "{deepest} ms of GPU work queued");
        // The simulation ran slower instead (the GPU finishes 33 steps a second at most).
        assert!((300..=340).contains(&taken), "{taken} steps in 10 s");
        assert!(g.pace().refused > 0);
        assert!(g.paused().is_none(), "30 ms is within the budget: slow, not paused");
    }

    #[test]
    fn steps_over_the_budget_twice_in_a_row_pause() {
        // 200 ms a step: paused after the grace steps and two over the budget, and the
        // work queued never passed two steps.
        let mut g = StepGate::new();
        let (most, _, taken, deepest) = run(&mut g, 600, 1000.0 / 60.0, 7, |_| 200.0);
        assert!(most <= 2);
        assert!(g.paused().is_some_and(|m| m.contains("two World steps in a row")), "{:?}", g.paused());
        assert!(taken <= u64::from(StepGate::GRACE) + 3, "{taken} steps before the pause");
        assert!(deepest <= 400.0 + 1e-9);
        // Paused: nothing more is taken until the host resumes.
        let before = taken;
        let (_, _, more, _) = run(&mut g, 60, 1000.0 / 60.0, 7, |_| 200.0);
        assert_eq!(more, 0, "took {more} steps while paused (after {before})");
    }

    #[test]
    fn one_slow_step_alone_does_not_pause() {
        // A hitch (one step of 120 ms among 5 ms steps) is not an overload.
        let mut g = StepGate::new();
        let (_, _, taken, _) = run(&mut g, 600, 1000.0 / 60.0, 1, |k| if k % 50 == 10 { 120.0 } else { 5.0 });
        assert!(g.paused().is_none(), "{:?}", g.paused());
        assert!(taken > 500);
    }

    #[test]
    fn a_step_that_never_completes_pauses_by_the_stall_time() {
        let mut g = StepGate::new();
        assert!(g.may_step(0.0));
        g.submitted(0.0);
        assert!(g.may_step(1.0));
        g.submitted(1.0);
        assert!(!g.may_step(2.0), "a third step with two in flight");
        assert!(g.paused().is_none());
        assert!(!g.may_step(StepGate::STALL_MS + 2.0));
        assert!(g.paused().is_some_and(|m| m.contains("has not finished")));
        assert_eq!(g.pace().in_flight, 2);
    }

    #[test]
    fn resume_keeps_the_steps_in_flight_counted() {
        let mut g = StepGate::new();
        g.submitted(0.0);
        g.submitted(0.0);
        assert!(!g.may_step(5000.0));
        assert!(g.paused().is_some());
        g.resume(5000.0);
        // Still two in flight: resuming does not let a third in.
        assert!(!g.may_step(5001.0));
        assert!(g.paused().is_none(), "the stall clock started again");
        g.completed(5002.0);
        assert!(g.may_step(5003.0));
    }

    #[test]
    fn a_lost_device_stops_for_good() {
        let mut g = StepGate::new();
        g.submitted(0.0);
        g.stop("the GPU device was lost");
        g.completed(1.0);
        assert!(!g.may_step(2.0));
        g.resume(2.5);
        assert!(!g.may_step(3.0), "resume after a stop");
        assert!(g.paused().is_some_and(|m| m.contains("device was lost")));
    }

    #[test]
    fn the_budget_is_a_setting() {
        let mut g = StepGate::new();
        g.set_budget_ms(10.0);
        run(&mut g, 600, 1000.0 / 60.0, 1, |_| 14.0);
        assert!(g.paused().is_some());
        let mut g = StepGate::new();
        g.set_budget_ms(f64::NAN);
        assert_eq!(g.budget_ms(), StepGate::BUDGET_MS);
    }
}
