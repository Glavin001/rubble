//! Triggers, key-driven commands and slide nodes for the explicit solver
//! (`engine/docs/explicit-beams-sections-triggers-slidenodes-spec.md` §2–3, §6).
//!
//! * A trigger is a beam of [`crate::explicit_beams::ExplicitKind::Trigger`]: no
//!   stiffness or damping, so its force is zero. In the beam pass it compares its
//!   extension with its bounds and holds one of two command keys for the next step's
//!   command pass ([`Trigger`]).
//! * With triggers, every command line moves by its keys' values
//!   ([`KeyCommand::update`]): one step of latency after the trigger.
//! * A slide node is held to a rail of beams by an undamped spring toward the
//!   nearest point of its current segment, the reaction split between the segment's
//!   ends by the ratio along it ([`SlideNode`], [`segment_point`]). The pass runs
//!   after self-collision.

use crate::explicit_aero::normalise;
use glam::Vec3;

/// Key values per step: keys 0…84 (key 0 is never processed).
pub const KEYS: usize = crate::truck::COMMAND_KEYS + 1;

/// A trigger's keys; its bounds are its beam's short and long bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trigger {
    /// Its beam.
    pub beam: u32,
    /// Keys held on the short side (1…84) and on the long side (0…84).
    pub short_key: u32,
    pub long_key: u32,
    /// Disabled triggers do not test (every trigger starts enabled).
    pub enabled: bool,
}

impl Trigger {
    /// The key held at extension `e` of rest length `rest` with bounds `sb`, `lb`
    /// (spec §2.5): the long key when e > lb·L, else the short key when
    /// e < −sb·L, else none. The long side wins where both hold.
    pub fn held(&self, e: f32, rest: f32, sb: f32, lb: f32) -> Option<u32> {
        if !self.enabled {
            None
        } else if e > lb * rest {
            Some(self.long_key)
        } else if e < -sb * rest {
            Some(self.short_key)
        } else {
            None
        }
    }
}

/// A command line driven by its keys (spec §2.6), self-centering or not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyCommand {
    pub beam: u32,
    pub shorten_rate: f32,
    pub lengthen_rate: f32,
    pub max_contraction: f32,
    pub max_extension: f32,
    /// Keys 1…84 (0: a key the command pass never reaches).
    pub contract_key: u32,
    pub extend_key: u32,
    /// Option `c`: with neither key held, drives toward `centre` × its spawn length.
    pub centering: bool,
    pub centre: f32,
    /// The contraction entry runs first (its key is lower, or the keys are equal).
    pub contract_first: bool,
    /// −1 contracting, +1 extending, 0 holding (self-centering).
    pub mode: i32,
}

impl KeyCommand {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        beam: u32,
        shorten_rate: f32,
        lengthen_rate: f32,
        max_contraction: f32,
        max_extension: f32,
        contract_key: i32,
        extend_key: i32,
        centering: bool,
    ) -> Self {
        let key = |k: i32| {
            if (1..KEYS as i32).contains(&k) {
                k as u32
            } else {
                0
            }
        };
        Self {
            beam,
            shorten_rate,
            lengthen_rate,
            max_contraction,
            max_extension,
            contract_key: key(contract_key),
            extend_key: key(extend_key),
            centering,
            centre: (max_extension - max_contraction) / 2.0 + max_contraction,
            contract_first: contract_key <= extend_key,
            mode: 0,
        }
    }

    /// The values of its contraction and extension keys (key 0 is never held).
    pub fn values(&self, keys: &[f32; KEYS]) -> (f32, f32) {
        let v = |k: u32| if k == 0 { 0.0 } else { keys[k as usize] };
        (v(self.contract_key), v(self.extend_key))
    }

    /// One step of both entries in key order: the new rest length from `rest`,
    /// the spawn length `rest0`, step `h` and this step's key values.
    pub fn update(&mut self, mut rest: f32, rest0: f32, h: f32, keys: &[f32; KEYS]) -> f32 {
        let (vc, ve) = self.values(keys);
        // A held key locks a self-centering line and stops its centering.
        let locked = self.centering && (vc >= 0.5 || ve >= 0.5);
        if locked {
            self.mode = 0;
        }
        let order: [i32; 2] = if self.contract_first {
            [-1, 1]
        } else {
            [1, -1]
        };
        for sigma in order {
            if rest0 == 0.0 || rest == 0.0 {
                continue;
            }
            if self.centering && !locked {
                let r = rest / rest0;
                if (((r - self.centre).abs()) as f64) < 1e-4 {
                    self.mode = 0;
                } else {
                    let old = self.mode;
                    self.mode = if r > self.centre { -1 } else { 1 };
                    if old != 0 && old != self.mode {
                        rest = self.centre * rest0;
                        self.mode = 0;
                    }
                }
            }
            let r = rest / rest0;
            let may =
                (sigma == 1 && r < self.max_extension) || (sigma == -1 && r > self.max_contraction);
            if may {
                let mut v = if sigma > 0 { ve } else { vc };
                if self.centering && !locked && sigma * self.mode > 0 {
                    v = 1.0;
                }
                let rate = if sigma > 0 {
                    self.lengthen_rate
                } else {
                    self.shorten_rate
                };
                let cf = 1.0f32;
                let x = rate * v * cf * h / rest;
                let x = if sigma > 0 { x } else { -x };
                rest = ((1.0 + x as f64) * rest as f64) as f32;
            }
        }
        rest
    }
}

/// A rail segment: a beam (its ends a, b give the direction) and its neighbours in
/// the rail (indices into [`crate::explicit_beams::ExplicitBeams::segments`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RailSegment {
    pub beam: u32,
    pub prev: Option<u32>,
    pub next: Option<u32>,
}

/// A slide node: its spring to its current rail segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlideNode {
    pub node: u32,
    /// Spring rate (N/m), break force (N), current tolerance (m).
    pub spring: f32,
    pub break_force: f32,
    pub tolerance: f32,
    /// Its current segment; `None`: no rail (inert for ever).
    pub segment: Option<u32>,
    /// Broken away: no force, though it still follows its rail.
    pub broken_away: bool,
}

/// The point of segment p1→p2 nearest `x`, its distance, and the ratio along the
/// segment (spec §3.3).
pub fn segment_point(p1: Vec3, p2: Vec3, x: Vec3) -> (Vec3, f32, f32) {
    let mut d = p2 - p1;
    let beta = d.length();
    if beta > 0.0 {
        d *= 1.0 / beta;
    }
    let proj = (x - p1).dot(d);
    // min(proj, β), then max(0, ·), as the spec defines them.
    let m = if beta < proj { beta } else { proj };
    let tau = if 0.0 < m { m } else { 0.0 };
    let q = p1 + d * tau;
    let dist = (q - x).length();
    let ratio = if beta > 0.0 { tau / beta } else { 0.0 };
    (q, dist, ratio)
}

/// What one slide node does in a step: the forces on it and on its segment's ends
/// (`None` when it applies none).
pub struct SlideForce {
    pub node: Vec3,
    pub a: Vec3,
    pub b: Vec3,
}

impl SlideNode {
    /// The spring force toward `q` (ratio `ratio` along the segment) from the
    /// node's world position `x` (spec §3.4 steps 5–7); sets `broken_away` when it
    /// exceeds the break force (the force still applies this step).
    pub fn force(&mut self, q: Vec3, x: Vec3, ratio: f32) -> SlideForce {
        let w = q - x;
        let delta = w.length();
        let w = normalise(w);
        let stretch = delta - self.tolerance;
        let f_len = -self.spring * if 0.0 < stretch { stretch } else { 0.0 };
        let p = w * f_len;
        if p.length() > self.break_force {
            self.broken_away = true;
        }
        SlideForce {
            node: -p,
            a: p * (1.0 - ratio),
            b: p * ratio,
        }
    }
}
