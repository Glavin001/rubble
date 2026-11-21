//! Physics first, drawing gives way ([`DrawPacer`]): which displayed frames a page
//! draws, and at what share of its pixels, so that drawing never holds back the World's
//! steps. The steps and their results never change; only how often the current state is
//! drawn does.
//!
//! The World's steps and the page's draws share one GPU queue: a draw submitted between
//! two steps runs before the second. A GPU that is behind refuses steps (the World does
//! not take one while its readbacks are still out, `WebWorld::can_step`), and a refused
//! step is simulated time lost. So, a frame at a time:
//!
//! - **No draw behind a draw.** While the last draw's GPU work is not done (`busy`),
//!   no frame is drawn (a second one would only queue in front of the next step, and
//!   the display shows one picture anyway), until the picture is older than the floor's
//!   interval.
//! - **Never a draw in a frame whose step was refused**, unless the picture is older
//!   than the floor's interval (the quality floor: `floor_fps`, 30 by default).
//! - **Levels.** Drawing gives way in levels: every display frame (frames between
//!   steps show the state carried forward by the pacer's lead), then at most 60
//!   pictures a second, then 30 (down to the floor), then, at the floor's rate, a
//!   lower resolution (75 %, 50 %; only when the page's render scale is automatic).
//!   Below the first level a frame is drawn only when a step was taken in it (a new
//!   state: the cars are skinned once a World step, not once a display frame).
//! - **Down when behind, back when not.** A half second with steps refused (two or
//!   more) moves a level down; [`QUIET`] seconds with none move a level back up. A
//!   level that was tried and lost again within [`QUIET`] seconds waits twice as long
//!   for its next try (up to [`QUIET_MAX`]), so drawing does not see-saw.
//!
//! With giving way off every frame is drawn at full scale (the page as it was).

use wasm_bindgen::prelude::*;

/// The window (wall seconds) over which refused steps are counted, and the refusals in
/// one that move a level down.
const WINDOW: f64 = 0.5;
const BEHIND: u32 = 2;
/// Quiet seconds (no steps refused) before a level back up; the most a lost try waits.
const QUIET: f64 = 2.0;
const QUIET_MAX: f64 = 16.0;
/// A frame is due at a level when this share of its interval has passed (display frames
/// jitter; at 120 Hz a 60-a-second level draws every second frame, not every third).
const DUE: f64 = 0.8;
/// Draw rates of the levels below every frame (pictures a second), then the scales.
const RATES: [f64; 4] = [60.0, 30.0, 20.0, 15.0];
const SCALES: [f64; 2] = [0.75, 0.5];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Level {
    /// Least seconds between draws (0: every display frame).
    gap: f64,
    scale: f64,
}

/// The page's draw scheduler (one per page; see the module).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DrawPacer {
    gives_way: bool,
    floor_fps: f64,
    auto_scale: bool,
    levels: Vec<Level>,
    level: usize,
    last_draw: f64,
    window_at: f64,
    window_refused: u32,
    /// The last time steps were refused (a window with any), or the level changed.
    calm_since: f64,
    /// Seconds of quiet before the next level up, and when the last level up was.
    quiet: f64,
    up_at: f64,
    drawn: u64,
}

impl Default for DrawPacer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl DrawPacer {
    /// Giving way on, a floor of 30 pictures a second, automatic scale on.
    #[wasm_bindgen(constructor)]
    pub fn new() -> DrawPacer {
        let mut p = DrawPacer {
            gives_way: true,
            floor_fps: 30.0,
            auto_scale: true,
            levels: Vec::new(),
            level: 0,
            last_draw: f64::NEG_INFINITY,
            window_at: f64::NAN,
            window_refused: 0,
            calm_since: f64::NAN,
            quiet: QUIET,
            up_at: f64::NEG_INFINITY,
            drawn: 0,
        };
        p.relevel();
        p
    }

    /// Drawing gives way to physics (the page's control; on by default).
    #[wasm_bindgen(js_name = setGivesWay)]
    pub fn set_gives_way(&mut self, on: bool) {
        if on != self.gives_way {
            self.gives_way = on;
            self.reset();
        }
    }

    /// The quality floor: the fewest pictures a second drawing gives way to (15 to 60).
    #[wasm_bindgen(js_name = setFloorFps)]
    pub fn set_floor_fps(&mut self, fps: f64) {
        let fps = if fps.is_finite() {
            fps.clamp(15.0, 60.0)
        } else {
            30.0
        };
        if fps != self.floor_fps {
            self.floor_fps = fps;
            self.relevel();
        }
    }

    /// Whether the levels may lower the picture's resolution (the page's render scale
    /// "automatic"); a fixed render scale keeps its pixels.
    #[wasm_bindgen(js_name = setAutoScale)]
    pub fn set_auto_scale(&mut self, on: bool) {
        if on != self.auto_scale {
            self.auto_scale = on;
            self.relevel();
        }
    }

    /// Back to every frame at full scale (a restart, a rebuild).
    pub fn reset(&mut self) {
        self.level = 0;
        self.window_at = f64::NAN;
        self.window_refused = 0;
        self.calm_since = f64::NAN;
        self.quiet = QUIET;
        self.up_at = f64::NEG_INFINITY;
        self.last_draw = f64::NEG_INFINITY;
    }

    /// Whether to draw this display frame: `now` wall seconds, `taken` World steps taken
    /// in it and `refused` steps owed that the World was not ready for, `stepping`
    /// (not paused), `busy` (the last draw's GPU work not done yet).
    pub fn frame(
        &mut self,
        now: f64,
        taken: u32,
        refused: u32,
        stepping: bool,
        busy: bool,
    ) -> bool {
        if !self.gives_way {
            self.mark(now);
            return true;
        }
        if stepping {
            self.adapt(now, refused);
        } else {
            // Paused: nothing to give way to; the level is kept for when it runs again.
            self.window_at = f64::NAN;
            self.window_refused = 0;
            self.calm_since = f64::NAN;
        }
        let since = now - self.last_draw;
        let floor_due = since >= DUE / self.floor_fps;
        // A picture still on the GPU holds the next one back, down to the floor: past it
        // the latest state is drawn anyway (the picture is never older than the floor).
        if busy && !(stepping && floor_due) {
            return false;
        }
        let draw = if !stepping {
            true
        } else if refused > 0 {
            floor_due
        } else {
            let lv = self.levels[self.level];
            lv.gap == 0.0 || (taken > 0 && since >= DUE * lv.gap) || floor_due
        };
        if draw {
            self.mark(now);
        }
        draw
    }

    /// The share of the display's pixels to draw at (1 unless a level lowered it).
    pub fn scale(&self) -> f64 {
        if self.gives_way {
            self.levels[self.level].scale
        } else {
            1.0
        }
    }

    /// Pictures a second the level allows (0: every display frame).
    #[wasm_bindgen(js_name = levelFps)]
    pub fn level_fps(&self) -> f64 {
        let g = if self.gives_way {
            self.levels[self.level].gap
        } else {
            0.0
        };
        if g > 0.0 {
            1.0 / g
        } else {
            0.0
        }
    }

    /// The level (0: every frame, full scale) and how many there are.
    pub fn level(&self) -> u32 {
        self.level as u32
    }
    pub fn levels(&self) -> u32 {
        self.levels.len() as u32
    }

    /// Seconds of quiet the next level up waits for.
    pub fn quiet(&self) -> f64 {
        self.quiet
    }

    /// Frames drawn so far.
    pub fn drawn(&self) -> f64 {
        self.drawn as f64
    }
}

impl DrawPacer {
    fn mark(&mut self, now: f64) {
        self.last_draw = now;
        self.drawn += 1;
    }

    fn relevel(&mut self) {
        let mut levels = vec![Level {
            gap: 0.0,
            scale: 1.0,
        }];
        for r in RATES {
            if r + 1e-9 >= self.floor_fps {
                levels.push(Level {
                    gap: 1.0 / r,
                    scale: 1.0,
                });
            }
        }
        let floor_gap = levels.last().map_or(0.0, |l| l.gap);
        if self.auto_scale {
            for s in SCALES {
                levels.push(Level {
                    gap: floor_gap,
                    scale: s,
                });
            }
        }
        self.levels = levels;
        self.level = self.level.min(self.levels.len() - 1);
    }

    fn adapt(&mut self, now: f64, refused: u32) {
        if self.window_at.is_nan() {
            self.window_at = now;
            self.calm_since = now;
        }
        self.window_refused += refused;
        if now - self.window_at < WINDOW {
            return;
        }
        let n = self.window_refused;
        self.window_at = now;
        self.window_refused = 0;
        if n >= BEHIND {
            if self.level + 1 < self.levels.len() {
                // A level up lost again before it had been quiet for QUIET: the next try
                // waits twice as long.
                if now - self.up_at < self.quiet + QUIET {
                    self.quiet = (self.quiet * 2.0).min(QUIET_MAX);
                }
                self.level += 1;
            }
            self.calm_since = now;
        } else if n > 0 {
            self.calm_since = now;
        } else if now - self.calm_since >= self.quiet {
            if self.level > 0 {
                self.level -= 1;
                self.up_at = now;
            } else {
                self.quiet = QUIET;
            }
            self.calm_since = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `secs` of a display at `hz`: each frame takes a step when one is owed (60 a
    /// second), and `refuse(t, level)` says whether the World refused it. Frames drawn.
    fn run(
        p: &mut DrawPacer,
        t0: f64,
        secs: f64,
        hz: f64,
        refuse: impl Fn(f64, u32) -> bool,
    ) -> (u32, f64) {
        let (mut drawn, mut acc) = (0, 0.0);
        let n = (secs * hz) as usize;
        let mut t = t0;
        for _ in 0..n {
            t += 1.0 / hz;
            acc += 60.0 / hz;
            let owed = acc.floor() as u32;
            acc -= f64::from(owed);
            let refused = if owed > 0 && refuse(t, p.level()) {
                owed
            } else {
                0
            };
            if p.frame(t, owed - refused, refused, true, false) {
                drawn += 1;
            }
        }
        (drawn, t)
    }

    #[test]
    fn headroom_draws_every_frame() {
        for hz in [60.0, 120.0, 144.0] {
            let mut p = DrawPacer::new();
            let (drawn, _) = run(&mut p, 0.0, 10.0, hz, |_, _| false);
            assert_eq!(drawn, (10.0 * hz) as u32, "{hz} Hz");
            assert_eq!(p.level(), 0);
            assert_eq!(p.scale(), 1.0);
        }
    }

    #[test]
    fn behind_gives_way_level_by_level_to_the_floor() {
        let mut p = DrawPacer::new();
        // Steps refused while drawing at more than 30 a second: down to 30.
        let (_, t) = run(&mut p, 0.0, 5.0, 120.0, |_, l| l < 2);
        assert_eq!(p.level(), 2);
        assert_eq!(p.level_fps(), 30.0);
        // At 30 it keeps up: about 30 pictures a second, each a frame with a step.
        let (drawn, _) = run(&mut p, t, 1.0, 120.0, |_, l| l < 2);
        assert!((28..=31).contains(&drawn), "{drawn}");
        assert_eq!(p.scale(), 1.0);
    }

    #[test]
    fn a_gpu_behind_at_any_rate_ends_at_the_floor_and_half_scale() {
        let mut p = DrawPacer::new();
        run(&mut p, 0.0, 10.0, 120.0, |_, _| true);
        assert_eq!(p.level() + 1, p.levels());
        assert_eq!(p.scale(), 0.5);
        assert_eq!(p.level_fps(), 30.0);
        // Never below the floor: the picture is at most a floor's interval old.
        let (drawn, _) = run(&mut p, 10.0, 2.0, 120.0, |_, _| true);
        assert!(drawn >= 2 * 30 - 2, "{drawn}");
    }

    #[test]
    fn comes_back_when_quiet_and_waits_longer_after_a_lost_try() {
        let mut p = DrawPacer::new();
        let (_, t) = run(&mut p, 0.0, 1.0, 120.0, |_, l| l < 1);
        assert_eq!(p.level(), 1);
        // Level 1 keeps up; QUIET seconds later it tries level 0, which loses again.
        let (_, t) = run(&mut p, t, QUIET + 1.2, 120.0, |_, l| l < 1);
        assert_eq!(p.level(), 1);
        assert_eq!(p.quiet(), 2.0 * QUIET);
        // Headroom for real (the GPU got faster): back to every frame.
        let (_, _) = run(&mut p, t, 2.0 * QUIET + 1.0, 120.0, |_, _| false);
        assert_eq!(p.level(), 0);
    }

    #[test]
    fn never_a_draw_behind_a_draw_and_off_draws_every_frame() {
        let mut p = DrawPacer::new();
        assert!(p.frame(0.0, 1, 0, true, false));
        assert!(!p.frame(1.0 / 120.0, 0, 0, true, true));
        // Past the floor (30 a second) the latest state is drawn even so.
        assert!(p.frame(1.0 / 30.0, 1, 0, true, true));
        p.set_gives_way(false);
        assert!(p.frame(2.0 / 120.0, 0, 3, true, true));
        assert_eq!(p.scale(), 1.0);
    }

    #[test]
    fn a_fixed_render_scale_and_a_floor_of_60() {
        let mut p = DrawPacer::new();
        p.set_auto_scale(false);
        p.set_floor_fps(60.0);
        assert_eq!(p.levels(), 2);
        run(&mut p, 0.0, 10.0, 120.0, |_, _| true);
        assert_eq!(p.level(), 1);
        assert_eq!(p.scale(), 1.0);
        assert_eq!(p.level_fps(), 60.0);
    }
}
