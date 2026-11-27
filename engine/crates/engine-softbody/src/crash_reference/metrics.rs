//! What a crash run is measured by, computed on the host from sampled state, so every
//! executor is measured by the same code.
//!
//! Metric names: `<body>.<measure>` for a body, `all.<measure>` for the scene,
//! `pair.<a>_<b>.<measure>` for two bodies (in layout order). The body's own frame is its
//! spawn pose: forward its first heading, up +Y, left = up × forward (+X at yaw 0).
//! Orientation is the mass-weighted best-fit rotation of its nodes from the first sample
//! ([`fit::robust_fit`]: refitted without nodes that have flown off or crushed far).
//!
//! | measure | meaning |
//! |---|---|
//! | `mass` | kg (the spawned nodes, wheel masses on their hubs) |
//! | `speed_before` | the centre of mass's speed at the start (m/s) |
//! | `v_along_end` | its velocity along its first heading at the end (m/s) |
//! | `rebound_ratio` | −`v_along_end` / `speed_before` (positive: bounced back) |
//! | `speed_end`, `vx_end`, `vz_end` | the COM's speed and world velocity at the end (m/s) |
//! | `dv_along` | change of the COM's velocity along its first heading, end − start (m/s) |
//! | `yaw_deg`, `yaw_peak_deg`, `yaw_abs_peak_deg` | its turn at the end (signed, + from forward toward left; unwrapped: cumulative through ±180°), the sample's value of largest magnitude (signed, as `pitch_peak_deg`), and that magnitude |
//! | `pitch_peak_deg`, `pitch_end_deg` | pitch (+ nose down, rear up): the sample's value of largest magnitude, and at the end |
//! | `roll_peak_deg` | largest \|roll\| (roll + is left side up) |
//! | `lateral_drift` | COM displacement across its first heading at the end (m, + toward its left) |
//! | `decel_peak_g` | largest horizontal COM deceleration over a 10 ms window (in g; horizontal so landing on the suspension is not a pulse) |
//! | `pulse_ms`, `decel_mean_g` | from the first to the last 10 ms window above 2 g (each placed at its middle); \|Δv\| over that span / its length (0 when none) |
//! | `max_node_speed` | largest node speed in any sample (m/s) |
//! | `speed_peak` | largest COM speed in any sample (m/s) |
//! | `length0`, `crush_peak`, `crush_residual` | `gauge.length`'s rest; the most it shortened (rest − length, 0 if it never did); rest − its last value (m) |
//! | `bent_frac`, `broken_frac` | share of its length beams (not torsion, LBEAM, coupler, pressure, aero or trigger records) bent for good (> 0.1 % set, alive) and broken, at the end |
//! | `plastic_work_j` | plastic work of its beams' yield, with the thresholds hardening as the solver hardens them ([`beams::Plastic`]), and of its LBEAMs' ([`beams::LbeamPlastic`]) |
//! | `zone.<s>_<side>.work_j`, `.bent_frac`, `.broken_frac` | the same per zone: s in front_lower, front_upper, cabin, door, rear, wheel, other; side left, right (a centre-line beam is half on each) |
//! | `zone.front.work_j`, `zone.front_left.work_j`, `zone.front_right.work_j`, `zone.front_upper.work_j`, `zone.front_lower.work_j` | front = front_lower + front_upper |
//! | `zone.cabin.work_j`, `zone.rear.work_j`, `zone.left.work_j`, `zone.right.work_j` | cabin = cabin + door; left/right: every section |
//! | `zone.{front,cabin,rear}.bent_frac`, `.broken_frac` | over the beams of those sections |
//! | `zone.{front,cabin,rear,front_upper}.work_frac` | share of `plastic_work_j` (0 when none) |
//! | `zone.front_lr_ratio`, `zone.lr_ratio` | (left + 1 J) / (right + 1 J) work, of the front and of the whole body (1 when neither yields) |
//! | `gauge.<name>.rest`, `.change_peak`, `.change_residual`, `.shrink_frac` | a generator gauge's length at the start (m), largest \|change\|, end − start, −residual / rest |
//! | `gauge.length.*`, `gauge.length_left.*`, `gauge.length_right.*`, `gauge.height.*` | the same for robust extents in the body's fitted frame ([`robust_extent`]: end to end over the attached nodes, those in its heaviest part of unbroken bonds now, less any run of them cut off by a gap along the axis over max(0.25 m, 1.5 × the body's widest gap along it at the first sample)): along its fitted forward (the fit's rotation of its first heading) over all of them, over its left half and over its right half (by the node's side at rest, centre-line nodes in both), and along its fitted up without wheel hubs |
//! | `detached_parts` | connected parts of its unbroken bonds at the end minus at the start |
//! | `detached_mass` | kg outside its heaviest part at the end |
//! | `all.finite` | 1 when every sampled value was finite |
//! | `all.ke_start_j`, `all.ke_end_j`, `all.ke_lost_j` | kinetic energy of every node, and start − end |
//! | `all.momentum_{x,z}_{start,end}` | linear momentum of every body (kg m/s) |
//! | `com_end_past_box0` | (its COM at the end − the near face of the layout's box 0) · its first heading (m; only when the layout has a box) |
//! | `all.energy_gain_j` | largest rise of kinetic + gravitational + beam-stored + wheel-spin energy above its running minimum (J; created energy). beam-stored includes raycast wheels' suspension springs ([`wheels`]); spin is ½ I ω² of each raycast wheel (`Sample::wheel_spin`, none when the executor leaves it empty) |
//! | `all.com_speed_end` | \|Σ m v\| / Σ m over every node of every body at the end (m/s) |
//! | `all.box_depth_m` | deepest any node centre gets inside a static box of the layout (m; 0 when none) |
//! | `all.plastic_work_j` | every body's `plastic_work_j` |
//! | `pair.<a>_<b>.com_gap_min` | smallest distance between their COMs (m) |
//! | `pair.<a>_<b>.reach_max`, `.skin_reach_max` | furthest any node of a (any node of a's collision triangles) gets past b's COM at that sample along a's first heading (m; negative: none reached it; no skin reach for a body without collision triangles). Both orders |
//! | `pair.<a>_<b>.com_order_min` | least (b's COM − a's COM) · a's first heading over the run (m; negative: b's centre of mass got behind a's, through, under or over it). Both orders |
//! | `pair.<a>_<b>.crossings_max` | most edges of one's collision surface through the other's live triangles at any sample (`ExplicitBeamsWorld::mesh_crossings`, both ways; the hard rule is 0) |
//! | `pair.<a>_<b>.crush_peak_ratio`, `.plastic_work_ratio` | larger over smaller of the two `crush_peak` (each floored at 0, + 1 mm) and `plastic_work_j` (+ 1 J) |
//! | `pair.<a>_<b>.yaw_peak_diff_deg` | \|a's `yaw_peak_deg` − b's\| |
//! | `v_travel_end` | the COM's velocity at the end along its initial velocity's direction (m/s; 0 for a body that starts at rest) |
//! | `side_crush_{left,right}_{peak,residual}` | largest inward displacement of that side's skin nodes in the door band ([`SIDE_BAND`], [`SIDE_OUTER`]), in the body's fitted frame (the robust fit inverted), at any sample and at the last (m; NHTSA's side "maximum exterior crush", levels 1–4) |
//! | `all.momentum_along_start`, `all.momentum_along_end` | horizontal momentum of every body at the start, and at the end along its direction (kg m/s) |
//!
//! Time histories ([`series`]: SAE axes of the body's first frame, x forward, y right; t from
//! the layout's first contact; `ref` the reference location, the rear-seat crossmember's
//! place [`series::reference_nodes`], `com` the centre of mass):
//!
//! | measure | meaning |
//! |---|---|
//! | `ref.nodes` | nodes averaged for the reference location (0: it is the centre of mass) |
//! | `{ref,com}.peak_decel_g`, `.peak_ms` | largest −ax after contact (g; the 4 ms estimator: Δv over 8 ms) and when (ms) |
//! | `{ref,com}.stop_ms`, `.dyn_crush`, `.mean_decel_g` | when vx first reaches 0 (t_m, ms), dx then (m, Hermite between samples), vx at contact / t_m (g); only when it stops |
//! | `{ref,com}.rebound`, `.restitution` | −min vx from t_m to 200 ms after contact (m/s), over vx at contact; only when it stops |
//! | `{ref,com}.vy_peak`, `.vy_peak_ms`, `.ay_peak_g`, `.ay_peak_ms`, `.dy_peak` | largest-magnitude lateral velocity (signed, + right) and when, lateral acceleration (signed) and when, lateral displacement |
//! | `{ref,com}.dvy_peak`, `.dvy_peak_ms`, `.dvx_150ms`, `.dvy_150ms` | largest-magnitude change of vy from contact (signed) and when; vx and vy 150 ms after contact less at contact |
//! | `{ref,com}.cfc60_peak_decel_g`, `.cfc60_peak_ms`, `.cfc60_ay_peak_g`, `.cfc60_ay_peak_ms` | the same peaks on the 2 kHz trace through SAE J211 CFC 60, from an executor that traces every step ([`super::Runner::start_trace`], the CPU reference) |
//! | `all.box_force_peak_n`, `.box_force_peak_ms` | largest force on the static boxes after contact, −d(Σ m v)/dt along body 0's initial velocity (N), and when |
//! | `all.box_force_pre_n` | its mean magnitude over the 40 ms before contact: friction and drag alone, the method's error floor (N) |
//! | `all.box_impulse_ns` | momentum lost along that direction from contact to the end (N s): the boxes' impulse plus friction's |
//!
//! The executor may add `all.contact_removed_j` ([`super::run`]).

pub mod beams;
mod fit;
mod graph;
pub mod series;
pub mod signal;
mod wheels;
mod zones;
#[cfg(test)]
mod tests;

pub use fit::{robust_fit, weighted_fit, yaw_pitch_roll};
pub use series::{reference_nodes, BodySeries, Kinematics, Series};
pub use zones::{assign as assign_zones, Zone, SECTIONS};

use super::{Built, Sample, Section, Template, Templates};
use crate::explicit_beams::{ExplicitBeams, GRAVITY};
use crate::explicit_contact::ExplicitBeamsWorld;
use glam::{DVec3, Vec3};
use std::collections::BTreeMap;

/// Deceleration window (s) and the pulse threshold (g).
const WINDOW: f64 = 0.010;
const PULSE_G: f64 = 2.0;
const G: f64 = -GRAVITY as f64;
/// Added to both sides of a pair's ratios so that none divides by zero (1 mm of crush,
/// 1 J of work, as `zone.lr_ratio` adds 1 J).
const CRUSH_EPS: f64 = 0.001;
const WORK_EPS: f64 = 1.0;

/// A length followed through a run: its start, largest |change| and last value.
#[derive(Clone, Copy, Debug, Default)]
struct Track {
    rest: f64,
    peak: f64,
    /// Largest shortening, rest − value (0 when it never shortened).
    shrink: f64,
    last: f64,
    started: bool,
}

impl Track {
    fn push(&mut self, v: f64) {
        if !self.started {
            (self.rest, self.started) = (v, true);
        }
        self.peak = self.peak.max((v - self.rest).abs());
        self.shrink = self.shrink.max(self.rest - v);
        self.last = v;
    }
    fn put(&self, put: &mut dyn FnMut(&str, f64), name: &str) {
        let res = self.last - self.rest;
        put(&format!("gauge.{name}.rest"), self.rest);
        put(&format!("gauge.{name}.change_peak"), self.peak);
        put(&format!("gauge.{name}.change_residual"), res);
        put(&format!("gauge.{name}.shrink_frac"), if self.rest.abs() > 0.0 { -res / self.rest } else { 0.0 });
    }
}

struct BodyRec {
    name: &'static str,
    mass: Vec<f32>,
    heading: Vec3,
    left: Vec3,
    /// The vehicle as spawned (its beams' rates, its raycast wheels).
    fleet: ExplicitBeams,
    zones: Vec<Zone>,
    plastic: Vec<Option<beams::Plastic>>,
    /// LBEAMs' yield ([`beams::LbeamPlastic`]).
    lplastic: Vec<Option<beams::LbeamPlastic>>,
    wheel_node: Vec<bool>,
    /// Per node its side at rest: +1 left, −1 right, 0 on the centre line (in both halves).
    side: Vec<i8>,
    /// The nodes of its collision triangles (not anti-clip only).
    skin: Vec<bool>,
    /// Its raycast wheels' spin inertia (kg m², `drive.config.wheels` order).
    inertia: Vec<f64>,
    gauges: Vec<(String, usize, usize, Track)>,
    /// Robust extents ([`robust_extent`]): whole length, each side's, height.
    glength: ExtentTrack,
    gleft: ExtentTrack,
    gright: ExtentTrack,
    gheight: ExtentTrack,
    ref_pos: Vec<Vec3>,
    com0: DVec3,
    v0: DVec3,
    /// Yaw: the last raw angle, the unwrapped one, the unwrapped value of largest
    /// magnitude (signed) and that magnitude.
    yaw_raw: f64,
    yaw: f64,
    yaw_peak: f64,
    yaw_abs_peak: f64,
    pitch_peak: f64,
    pitch_end: f64,
    roll_peak: f64,
    /// (t, horizontal COM velocity) per sample.
    vh: Vec<(f64, DVec3)>,
    max_node_speed: f64,
    /// Largest COM speed (m/s).
    speed_peak: f64,
    parts0: usize,
    /// This sample's attached nodes (in its heaviest part) and COM.
    attached: Vec<bool>,
    com_now: DVec3,
    /// Its time histories ([`series`]).
    track: series::BodyTrack,
    /// Its initial velocity's direction (zero when it starts at rest).
    travel: DVec3,
    /// Skin nodes of each side's door band ([`SIDE_BAND`]): [left, right], and the largest
    /// inward displacement of any of them in the body's fitted frame, at any sample and at
    /// the last.
    side_nodes: [Vec<usize>; 2],
    side_peak: [f64; 2],
    side_last: [f64; 2],
}

/// The heights above the ground (m, at rest: above the wheels' lowest point) of the band of a
/// side's skin whose crush is measured: NHTSA's side crush levels 1–4 (the reports' "maximum
/// exterior crush" table: sill top about 0.30 m, occupant hip point 0.50–0.56, mid door 0.62,
/// window sill 0.92–0.94); level 5, the window top (glass), and the sill's underside are left
/// out.
pub const SIDE_BAND: (f32, f32) = (0.25, 0.95);
/// A side's skin: nodes at least this share of the half width out from the centre line.
pub const SIDE_OUTER: f32 = 0.75;

/// Two bodies (a < b in layout order) followed through a run.
struct PairRec {
    /// Smallest COM distance.
    gap: f64,
    /// [a past b, b past a]: furthest any node, and any collision-skin node, gets past the
    /// other's COM along its own first heading.
    reach: [f64; 2],
    skin_reach: [f64; 2],
    /// [a then b, b then a]: least (the other's COM − its own COM) along its own first
    /// heading over the run.
    order: [f64; 2],
    /// The two vehicles as spawned, moved to each sample's state for the crossings test.
    mesh: ExplicitBeamsWorld,
    crossings: usize,
}

pub struct Recorder {
    bodies: Vec<BodyRec>,
    boxes: Vec<(Vec3, Vec3)>,
    first: Option<Sample>,
    last: Option<Sample>,
    finite: bool,
    e_min: f64,
    e_gain: f64,
    box_depth: f64,
    pairs: BTreeMap<(usize, usize), PairRec>,
    /// The scene's time histories ([`series`]) and the layout's first contact (s).
    scene: series::SceneTrack,
    impact_s: f64,
    /// The executor's per-step trace ([`super::Runner::start_trace`]): per step, per location
    /// ([`Self::trace_sets`]); the first entry from the first sample.
    fine: Vec<Vec<DVec3>>,
}

/// The positions along `axis` of the nodes `keep` admits, sorted.
fn along(pos: &[Vec3], keep: impl Fn(usize) -> bool, axis: Vec3) -> Vec<f32> {
    let mut xs: Vec<f32> = pos.iter().enumerate().filter(|(i, _)| keep(*i)).map(|(_, p)| p.dot(axis)).collect();
    xs.sort_by(f32::total_cmp);
    xs
}

/// The largest gap between neighbours of sorted `xs` (0 with fewer than two).
fn widest_gap(xs: &[f32]) -> f32 {
    xs.windows(2).map(|w| w[1] - w[0]).fold(0.0, f32::max)
}

/// The run of sorted `xs` with the most entries and no gap over `gap`: its last − first.
fn run_extent(xs: &[f32], gap: f32) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let (mut best, mut start) = ((0usize, 0usize), 0usize);
    for i in 1..=xs.len() {
        if i == xs.len() || xs[i] - xs[i - 1] > gap {
            if i - start > best.1 - best.0 {
                best = (start, i);
            }
            start = i;
        }
    }
    f64::from(xs[best.1 - 1]) - f64::from(xs[best.0])
}

/// The extent along `axis` of the nodes `keep` admits (m), robust to nodes flung off but
/// still tied to the body (by a stretched beam, or a push-only support that holds nothing
/// in tension): their positions along the axis, sorted, are cut wherever two neighbours
/// are more than `gap` apart, and the extent is the run with the most nodes, last − first.
/// A body with no such gap is measured end to end, exactly (a percentile would trim its
/// ends and read an end flattened against a wall as crush). 0 with fewer than two nodes.
pub fn robust_extent(pos: &[Vec3], keep: impl Fn(usize) -> bool, axis: Vec3, gap: f32) -> f64 {
    run_extent(&along(pos, keep, axis), gap)
}

/// The least gap (m) that separates a flung node from the body in [`robust_extent`]:
/// several times the generated cars' node spacing. A track's own gap is the larger of
/// this and [`GAP_OVER_REST`] × the widest gap of the body at its first sample, so the
/// body as it starts is always measured end to end.
pub const EXTENT_GAP: f32 = 0.25;
pub const GAP_OVER_REST: f32 = 1.5;

/// A [`Track`] of a robust extent, its gap set at the first sample.
#[derive(Clone, Copy, Debug, Default)]
struct ExtentTrack {
    track: Track,
    gap: f32,
}

impl ExtentTrack {
    fn push(&mut self, pos: &[Vec3], keep: impl Fn(usize) -> bool, axis: Vec3) {
        let xs = along(pos, keep, axis);
        if !self.track.started {
            self.gap = EXTENT_GAP.max(GAP_OVER_REST * widest_gap(&xs));
        }
        self.track.push(run_extent(&xs, self.gap));
    }
}

fn com(x: &[Vec3], mass: &[f32]) -> DVec3 {
    let (s, m) = x.iter().zip(mass).fold((DVec3::ZERO, 0.0), |(s, m), (p, w)| (s + f64::from(*w) * p.as_dvec3(), m + f64::from(*w)));
    s / m.max(1e-12)
}

fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * ((a + std::f64::consts::PI) / t).floor()
}

/// The horizontal COM velocity at `t`, linear between samples.
fn v_at(vh: &[(f64, DVec3)], t: f64) -> DVec3 {
    let i = vh.partition_point(|x| x.0 <= t);
    match (i.checked_sub(1).map(|i| vh[i]), vh.get(i)) {
        (Some((t0, v0)), Some(&(t1, v1))) if t1 > t0 => v0 + (v1 - v0) * ((t - t0) / (t1 - t0)),
        (Some((_, v)), _) | (None, Some(&(_, v))) => v,
        _ => DVec3::ZERO,
    }
}

/// Largest windowed deceleration (g); the pulse: from the first to the last window above
/// 2 g, each window's deceleration placed at its middle; its length (ms) and |Δv| over it
/// / its length (g).
fn pulse(vh: &[(f64, DVec3)]) -> (f64, f64, f64) {
    let (mut peak, mut first, mut last) = (0.0f64, None, None);
    for i in 0..vh.len() {
        let Some(j) = (i + 1..vh.len()).find(|&j| vh[j].0 >= vh[i].0 + WINDOW - 1e-6) else { break };
        let a = (vh[j].1 - vh[i].1).length() / (vh[j].0 - vh[i].0) / G;
        peak = peak.max(a);
        if a > PULSE_G {
            let mid = 0.5 * (vh[i].0 + vh[j].0);
            first.get_or_insert(mid);
            last = Some(mid);
        }
    }
    match (first, last) {
        (Some(a), Some(b)) if b > a => (peak, (v_at(vh, b) - v_at(vh, a)).length() / (b - a) / G, 1000.0 * (b - a)),
        _ => (peak, 0.0, 0.0),
    }
}

/// Every node at rest in the body's own frame: the template's positions, any node the
/// spawn added un-turned from the fleet (`turned` in the parent module, inverted).
fn rest_frame(t: &Template, fleet_pos: &[Vec3], at: Vec3, yaw: f32) -> Vec<Vec3> {
    let (s, c) = yaw.sin_cos();
    fleet_pos
        .iter()
        .enumerate()
        .map(|(i, p)| match t.desc.nodes.get(i) {
            Some(n) => Vec3::from(n.position),
            None => {
                let d = *p - at;
                Vec3::new(d.x * c - d.z * s, d.y, d.x * s + d.z * c)
            }
        })
        .collect()
}

impl BodyRec {
    fn new(t: &Template, b: &super::Body, s: &ExplicitBeams) -> Self {
        let pos = s.positions();
        let rest = rest_frame(t, &pos, b.at, b.yaw);
        let zones = zones::assign(t, &s.beams, &rest);
        let (lo, hi) = rest.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.x), hi.max(p.x)));
        let (mid, tol) = (0.5 * (lo + hi), 1e-3 * (hi - lo).max(1e-3));
        let side = rest.iter().map(|p| if (p.x - mid).abs() <= tol { 0 } else if p.x > mid { 1 } else { -1 }).collect();
        let mut skin = vec![false; pos.len()];
        for (k, tri) in s.triangles.iter().enumerate() {
            if !s.tri_anti_only(k) {
                for &i in &tri.nodes {
                    if let Some(x) = skin.get_mut(i as usize) {
                        *x = true;
                    }
                }
            }
        }
        let inertia = s.drive.as_deref().map_or(Vec::new(), |d| d.config.wheels.iter().map(|w| f64::from(w.inertia)).collect());
        let d = &t.desc;
        let named = |i: usize| d.nodes.get(i).and_then(|n| t.zones.parts.iter().filter(|(p, _)| n.name.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, s)| *s));
        let mut wheel_node: Vec<bool> = (0..pos.len()).map(|i| (!d.nodes.is_empty() && i >= d.nodes.len()) || named(i) == Some(Section::Wheel)).collect();
        for w in &d.wheels {
            for n in [w.node1, w.node2] {
                if let Some(x) = wheel_node.get_mut(n as usize) {
                    *x = true;
                }
            }
        }
        let index = |name: &str| d.nodes.iter().position(|n| n.name == name);
        let gauges = t.zones.gauges.iter().filter_map(|g| Some((g.name.clone(), index(&g.a)?, index(&g.b)?, Track::default()))).collect();
        let heading = b.heading();
        let ref_nodes: Vec<usize> = series::reference_nodes(t).into_iter().filter(|&i| i < pos.len()).collect();
        let track = series::BodyTrack::new(ref_nodes.clone(), ref_nodes.iter().map(|&i| d.nodes[i].name.clone()).collect());
        let half = 0.5 * (hi - lo);
        let wheel_low = super::Extent::of(d).wheel_low.unwrap_or_else(|| rest.iter().map(|p| p.y).fold(f32::MAX, f32::min));
        let in_band = |p: &Vec3| p.y - wheel_low >= SIDE_BAND.0 && p.y - wheel_low <= SIDE_BAND.1;
        let side_nodes = [1.0f32, -1.0].map(|s| (0..pos.len()).filter(|&i| skin[i] && !wheel_node[i] && in_band(&rest[i]) && s * (rest[i].x - mid) >= SIDE_OUTER * half).collect());
        let v0 = b.velocity();
        Self {
            name: b.name,
            mass: s.nodes.iter().map(|n| n.mass).collect(),
            heading,
            left: Vec3::Y.cross(heading),
            plastic: s.beams.iter().map(|x| beams::can_yield(x).then(|| beams::Plastic::new(x))).collect(),
            lplastic: s.beams.iter().map(beams::LbeamPlastic::new).collect(),
            fleet: s.clone(),
            zones,
            wheel_node,
            side,
            skin,
            inertia,
            gauges,
            glength: ExtentTrack::default(),
            gleft: ExtentTrack::default(),
            gright: ExtentTrack::default(),
            gheight: ExtentTrack::default(),
            ref_pos: Vec::new(),
            com0: DVec3::ZERO,
            v0: DVec3::ZERO,
            yaw_raw: 0.0,
            yaw: 0.0,
            yaw_peak: 0.0,
            yaw_abs_peak: 0.0,
            pitch_peak: 0.0,
            pitch_end: 0.0,
            roll_peak: 0.0,
            vh: Vec::new(),
            max_node_speed: 0.0,
            speed_peak: 0.0,
            parts0: 0,
            attached: Vec::new(),
            com_now: DVec3::ZERO,
            track,
            travel: v0.as_dvec3().normalize_or_zero(),
            side_nodes,
            side_peak: [0.0; 2],
            side_last: [0.0; 2],
        }
    }

    fn parts(&self, beams: &[super::BeamState], nodes: usize) -> (Vec<usize>, usize) {
        let links = self.fleet.beams.iter().zip(beams).filter(|(b, s)| !s.broken && beams::is_link(b.kind)).map(|(b, _)| (b.a, b.b));
        graph::components(nodes, links)
    }

    /// This sample's body measures; returns the energy (J) stored in its beams and its
    /// raycast wheels' suspension, and the wheels' spin energy (`spin`: their ω).
    fn push(&mut self, t: f64, pos: &[Vec3], vel: &[Vec3], st: &[super::BeamState], spin: &[f32]) -> f64 {
        let first = self.ref_pos.is_empty();
        let (label, count) = self.parts(st, pos.len());
        if first {
            self.ref_pos = pos.to_vec();
            self.com0 = com(pos, &self.mass);
            self.v0 = com(vel, &self.mass);
            self.parts0 = count;
        }
        // Attached: the nodes of its heaviest part.
        let mut part_mass = vec![0.0f64; count];
        for (l, m) in label.iter().zip(&self.mass) {
            part_mass[*l] += f64::from(*m);
        }
        let main = (0..count).max_by(|a, b| part_mass[*a].total_cmp(&part_mass[*b])).unwrap_or(0);
        self.attached = label.iter().map(|l| *l == main).collect();
        self.com_now = com(pos, &self.mass);
        let (r, tr) = fit::robust_fit(&self.ref_pos, pos, &self.mass);
        // Each side's crush: its door band's nodes back in the first sample's frame (the fit
        // inverted), inward toward the centre line (left side: −left, right side: +left).
        let left = self.left.as_dvec3();
        for (k, s) in [1.0f64, -1.0].into_iter().enumerate() {
            let inward = self.side_nodes[k].iter().filter_map(|&i| Some((pos.get(i)?, self.ref_pos.get(i)?))).map(|(q, p)| -s * (r.transpose() * (q.as_dvec3() - tr) - p.as_dvec3()).dot(left)).fold(0.0f64, f64::max);
            self.side_peak[k] = self.side_peak[k].max(inward);
            self.side_last[k] = inward;
        }
        let (yaw, pitch, roll) = fit::yaw_pitch_roll(r, self.heading);
        self.yaw += wrap(yaw - self.yaw_raw);
        self.yaw_raw = yaw;
        if self.yaw.abs() > self.yaw_abs_peak {
            (self.yaw_peak, self.yaw_abs_peak) = (self.yaw, self.yaw.abs());
        }
        if pitch.abs() > self.pitch_peak.abs() {
            self.pitch_peak = pitch;
        }
        self.pitch_end = pitch;
        self.roll_peak = self.roll_peak.max(roll.abs());
        let v = com(vel, &self.mass);
        self.track.push(pos, vel, self.com_now, v);
        self.vh.push((t, DVec3::new(v.x, 0.0, v.z)));
        self.speed_peak = self.speed_peak.max(v.length());
        self.max_node_speed = vel.iter().map(|v| f64::from(v.length())).fold(self.max_node_speed, f64::max);
        for (p, s) in self.plastic.iter_mut().zip(st) {
            if let Some(p) = p {
                p.advance(s.rest);
            }
        }
        for (j, (p, s)) in self.lplastic.iter_mut().zip(st).enumerate() {
            if let Some(p) = p {
                p.advance(&self.fleet, j, s.rest, pos);
            }
        }
        for (_, a, b, tr) in &mut self.gauges {
            if let (Some(pa), Some(pb)) = (pos.get(*a), pos.get(*b)) {
                tr.push(f64::from(pa.distance(*pb)));
            }
        }
        // Extents in the body's fitted frame, over its attached nodes.
        let h = (r * self.heading.as_dvec3()).as_vec3();
        let up = (r * DVec3::Y).as_vec3();
        let att = &self.attached;
        let (side, wheel) = (&self.side, &self.wheel_node);
        self.glength.push(pos, |i| att[i], h);
        self.gleft.push(pos, |i| att[i] && side[i] >= 0, h);
        self.gright.push(pos, |i| att[i] && side[i] <= 0, h);
        self.gheight.push(pos, |i| att[i] && !wheel[i], up);
        let beams: f64 = (0..self.fleet.beams.len().min(st.len())).map(|j| beams::elastic(&self.fleet.beams, Some(&self.fleet), j, st[j].rest, st[j].broken, pos)).sum();
        let spin: f64 = self.inertia.iter().zip(spin).map(|(i, w)| 0.5 * i * f64::from(*w).powi(2)).sum();
        beams + wheels::suspension_energy(&self.fleet, pos) + spin
    }
}

impl Recorder {
    pub fn new(templates: &Templates, built: &Built) -> Self {
        let blank = Template { desc: Default::default(), settings: Default::default(), zones: Default::default() };
        let bodies: Vec<BodyRec> = built.layout.bodies.iter().zip(&built.fleet).map(|(b, s)| BodyRec::new(templates.get(b.design).unwrap_or(&blank), b, s)).collect();
        let mut pairs = BTreeMap::new();
        for a in 0..bodies.len() {
            for b in a + 1..bodies.len() {
                let mesh = ExplicitBeamsWorld::new(vec![built.fleet[a].clone(), built.fleet[b].clone()]);
                pairs.insert((a, b), PairRec { gap: f64::INFINITY, reach: [f64::NEG_INFINITY; 2], skin_reach: [f64::NEG_INFINITY; 2], order: [f64::INFINITY; 2], mesh, crossings: 0 });
            }
        }
        Self {
            bodies,
            boxes: built.layout.boxes.clone(),
            first: None,
            last: None,
            finite: true,
            e_min: f64::INFINITY,
            e_gain: 0.0,
            box_depth: 0.0,
            pairs,
            scene: series::SceneTrack::default(),
            impact_s: f64::from(built.layout.impact_s),
            fine: Vec::new(),
        }
    }

    /// The locations of a per-step trace: each body's reference location (its nodes; none:
    /// its centre of mass) and its centre of mass, in body order.
    pub fn trace_sets(&self) -> Vec<(usize, Vec<usize>)> {
        self.bodies.iter().enumerate().flat_map(|(k, b)| [(k, b.track.ref_nodes.clone()), (k, Vec::new())]).collect()
    }

    /// A frame's per-step trace (per step, per location of [`Self::trace_sets`]:
    /// [`super::Runner::take_trace`]). The first sample's velocities start it.
    pub fn push_trace(&mut self, steps: Vec<Vec<DVec3>>) {
        if self.fine.is_empty() {
            self.fine.push(self.bodies.iter().flat_map(|b| b.track.first_velocities()).collect());
        }
        self.fine.extend(steps);
    }

    pub fn push(&mut self, x: &Sample) {
        self.finite &= x.pos.iter().flatten().chain(x.vel.iter().flatten()).all(|p| p.is_finite());
        let t = f64::from(x.t);
        let momentum: DVec3 = self.bodies.iter().zip(&x.vel).map(|(b, v)| v.iter().zip(&b.mass).map(|(v, m)| f64::from(*m) * v.as_dvec3()).sum::<DVec3>()).sum();
        self.scene.t.push(t);
        self.scene.momentum.push(momentum);
        let mut energy = 0.0;
        for (k, b) in self.bodies.iter_mut().enumerate() {
            let (Some(pos), Some(vel)) = (x.pos.get(k), x.vel.get(k)) else { continue };
            let st = x.beams.get(k).map_or(&[][..], |v| &v[..]);
            let spin = x.wheel_spin.get(k).map_or(&[][..], |v| &v[..]);
            energy += b.push(t, pos, vel, st, spin);
            for ((p, v), m) in pos.iter().zip(vel).zip(&b.mass) {
                let m = f64::from(*m);
                energy += 0.5 * m * f64::from(v.length_squared()) + m * G * f64::from(p.y);
                for &(c, h) in &self.boxes {
                    let d = h - (*p - c).abs();
                    self.box_depth = self.box_depth.max(f64::from(d.min_element()));
                }
            }
        }
        if energy.is_finite() {
            self.e_gain = self.e_gain.max(energy - self.e_min);
            self.e_min = self.e_min.min(energy);
        }
        for (&(a, b), pr) in &mut self.pairs {
            let (Some(pa), Some(pb)) = (x.pos.get(a), x.pos.get(b)) else { continue };
            let (ba, bb) = (&self.bodies[a], &self.bodies[b]);
            pr.gap = pr.gap.min(ba.com_now.distance(bb.com_now));
            // How far the nodes of `p` (body `r`) get past body `o`'s COM along `r`'s heading.
            let past = |p: &[Vec3], r: &BodyRec, o: &BodyRec, skin: bool| -> f64 {
                let h = r.heading.as_dvec3();
                p.iter().enumerate().filter(|(i, _)| !skin || r.skin.get(*i).copied().unwrap_or(false)).map(|(_, q)| (q.as_dvec3() - o.com_now).dot(h)).fold(f64::NEG_INFINITY, f64::max)
            };
            pr.reach[0] = pr.reach[0].max(past(pa, ba, bb, false));
            pr.reach[1] = pr.reach[1].max(past(pb, bb, ba, false));
            pr.skin_reach[0] = pr.skin_reach[0].max(past(pa, ba, bb, true));
            pr.skin_reach[1] = pr.skin_reach[1].max(past(pb, bb, ba, true));
            let d = bb.com_now - ba.com_now;
            pr.order[0] = pr.order[0].min(d.dot(ba.heading.as_dvec3()));
            pr.order[1] = pr.order[1].min((-d).dot(bb.heading.as_dvec3()));
            // The collision meshes at this sample: positions, and beams broken (a torn
            // triangle is out of the surface).
            for (v, k) in pr.mesh.vehicles.iter_mut().zip([a, b]) {
                let o = v.origin;
                for (n, p) in v.nodes.iter_mut().zip(&x.pos[k]) {
                    n.pos = *p - o;
                }
                if let Some(st) = x.beams.get(k) {
                    v.torn.now = st.iter().map(|s| s.broken).collect();
                }
            }
            pr.crossings = pr.crossings.max(pr.mesh.mesh_crossings());
        }
        if self.first.is_none() {
            self.first = Some(x.clone());
        }
        self.last = Some(x.clone());
    }

    /// Every metric (as [`Self::finish_with_series`] without the time histories).
    pub fn finish(self) -> BTreeMap<String, f64> {
        self.finish_with_series().0
    }

    /// Every metric and the run's time histories ([`series`]).
    pub fn finish_with_series(self) -> (BTreeMap<String, f64>, Series) {
        let mut m = self.metrics();
        let tracks: Vec<&series::BodyTrack> = self.bodies.iter().map(|b| &b.track).collect();
        let input = series::SeriesInput {
            names: self.bodies.iter().map(|b| b.name).collect(),
            masses: self.bodies.iter().map(|b| b.mass.iter().map(|x| f64::from(*x)).sum()).collect(),
            headings: self.bodies.iter().map(|b| b.heading).collect(),
            travel0: self.bodies.first().map_or(DVec3::Z, |b| if b.travel == DVec3::ZERO { b.heading.as_dvec3() } else { b.travel }),
            tracks: &tracks,
            scene: &self.scene,
            impact_s: self.impact_s,
            has_boxes: !self.boxes.is_empty(),
            fine: &self.fine,
            step_s: f64::from(super::DT),
        };
        let s = series::finish(input, &mut m);
        (m, s)
    }

    fn metrics(&self) -> BTreeMap<String, f64> {
        let mut m = BTreeMap::new();
        let (Some(first), Some(last)) = (&self.first, &self.last) else { return m };
        let over = |x: &Sample, f: &dyn Fn(Vec3, f64) -> DVec3| -> DVec3 { self.bodies.iter().enumerate().map(|(k, b)| x.vel.get(k).map_or(DVec3::ZERO, |v| v.iter().zip(&b.mass).map(|(v, w)| f(*v, f64::from(*w))).sum())).sum() };
        let ke = |x: &Sample| over(x, &|v, w| DVec3::splat(0.5 * w * f64::from(v.length_squared()))).x;
        let p = |x: &Sample| over(x, &|v, w| w * v.as_dvec3());
        let mut total_work = 0.0;
        let near_face = self.boxes.first().map(|&(c, h)| (c, h));
        for (k, b) in self.bodies.iter().enumerate() {
            total_work += body_metrics(&mut m, b, last, k);
            if let (Some((c, h)), Some(pos)) = (near_face, last.pos.get(k)) {
                // Box 0's face toward the body's first heading: its support along −heading.
                let f = b.heading.as_dvec3();
                let face = c.as_dvec3().dot(f) - (h.as_dvec3() * f).abs().element_sum();
                m.insert(format!("{}.com_end_past_box0", b.name), com(pos, &b.mass).dot(f) - face);
            }
        }
        let (ke0, ke1, p0, p1) = (ke(first), ke(last), p(first), p(last));
        let mass: f64 = self.bodies.iter().flat_map(|b| b.mass.iter()).map(|m| f64::from(*m)).sum();
        for (n, v) in [
            ("finite", f64::from(u8::from(self.finite))),
            ("ke_start_j", ke0),
            ("ke_end_j", ke1),
            ("ke_lost_j", ke0 - ke1),
            ("momentum_x_start", p0.x),
            ("momentum_z_start", p0.z),
            ("momentum_x_end", p1.x),
            ("momentum_z_end", p1.z),
            ("energy_gain_j", self.e_gain),
            ("box_depth_m", self.box_depth),
            ("plastic_work_j", total_work),
            ("com_speed_end", if mass > 0.0 { p1.length() / mass } else { 0.0 }),
            ("momentum_along_start", DVec3::new(p0.x, 0.0, p0.z).length()),
            ("momentum_along_end", DVec3::new(p1.x, 0.0, p1.z).dot(DVec3::new(p0.x, 0.0, p0.z).normalize_or_zero())),
        ] {
            m.insert(format!("all.{n}"), v);
        }
        for (&(a, b), pr) in &self.pairs {
            let (na, nb) = (self.bodies[a].name, self.bodies[b].name);
            let mut put = |n: &str, v: f64| {
                m.insert(n.to_string(), v);
            };
            put(&format!("pair.{na}_{nb}.com_gap_min"), pr.gap);
            put(&format!("pair.{na}_{nb}.crossings_max"), pr.crossings as f64);
            for (i, (x, y)) in [(na, nb), (nb, na)].into_iter().enumerate() {
                put(&format!("pair.{x}_{y}.reach_max"), pr.reach[i]);
                put(&format!("pair.{x}_{y}.com_order_min"), pr.order[i]);
                // (None for a body without collision triangles.)
                if pr.skin_reach[i].is_finite() {
                    put(&format!("pair.{x}_{y}.skin_reach_max"), pr.skin_reach[i]);
                }
            }
        }
        // Symmetry of two bodies (in layout order): the larger over the smaller.
        for (&(a, b), _) in &self.pairs {
            let (na, nb) = (self.bodies[a].name, self.bodies[b].name);
            let g = |n: &str, q: &str| m.get(&format!("{n}.{q}")).copied().unwrap_or(0.0);
            let ratio = |x: f64, y: f64, eps: f64| {
                let (x, y) = (x.max(0.0) + eps, y.max(0.0) + eps);
                x.max(y) / x.min(y)
            };
            let crush = ratio(g(na, "crush_peak"), g(nb, "crush_peak"), CRUSH_EPS);
            let work = ratio(g(na, "plastic_work_j"), g(nb, "plastic_work_j"), WORK_EPS);
            let yaw = (g(na, "yaw_peak_deg") - g(nb, "yaw_peak_deg")).abs();
            m.insert(format!("pair.{na}_{nb}.crush_peak_ratio"), crush);
            m.insert(format!("pair.{na}_{nb}.plastic_work_ratio"), work);
            m.insert(format!("pair.{na}_{nb}.yaw_peak_diff_deg"), yaw);
        }
        m
    }
}

/// Per zone (section × side): weighted beam count, bent, broken, work.
type ZoneSums = [[[f64; 4]; 2]; 7];

/// One body's metrics into `m`; returns its plastic work.
fn body_metrics(m: &mut BTreeMap<String, f64>, b: &BodyRec, last: &Sample, k: usize) -> f64 {
    let mut put = |n: &str, v: f64| {
        m.insert(format!("{}.{n}", b.name), v);
    };
    let (pos, vel) = (&last.pos[k], &last.vel[k]);
    let st = last.beams.get(k).map_or(&[][..], |v| &v[..]);
    let h = b.heading.as_dvec3();
    let v1 = com(vel, &b.mass);
    let v0 = b.v0.length();
    put("mass", b.mass.iter().map(|x| f64::from(*x)).sum());
    put("speed_before", v0);
    put("v_along_end", v1.dot(h));
    put("rebound_ratio", if v0 > 0.0 { -v1.dot(h) / v0 } else { 0.0 });
    put("speed_end", v1.length());
    put("vx_end", v1.x);
    put("vz_end", v1.z);
    put("dv_along", (v1 - b.v0).dot(h));
    put("v_travel_end", v1.dot(b.travel));
    for (k, side) in ["left", "right"].into_iter().enumerate() {
        put(&format!("side_crush_{side}_peak"), b.side_peak[k]);
        put(&format!("side_crush_{side}_residual"), b.side_last[k]);
    }
    put("yaw_deg", b.yaw.to_degrees());
    put("yaw_peak_deg", b.yaw_peak.to_degrees());
    put("yaw_abs_peak_deg", b.yaw_abs_peak.to_degrees());
    put("pitch_peak_deg", b.pitch_peak.to_degrees());
    put("pitch_end_deg", b.pitch_end.to_degrees());
    put("roll_peak_deg", b.roll_peak.to_degrees());
    put("lateral_drift", (com(pos, &b.mass) - b.com0).dot(b.left.as_dvec3()));
    let (peak, mean, ms) = pulse(&b.vh);
    put("decel_peak_g", peak);
    put("decel_mean_g", mean);
    put("pulse_ms", ms);
    put("max_node_speed", b.max_node_speed);
    put("speed_peak", b.speed_peak);
    let l = &b.glength.track;
    put("length0", l.rest);
    put("crush_peak", l.shrink);
    put("crush_residual", l.rest - l.last);
    // Zones.
    let mut z: ZoneSums = [[[0.0; 4]; 2]; 7];
    for (j, (beam, s)) in b.fleet.beams.iter().zip(st).enumerate() {
        let zone = b.zones[j];
        let w = b.plastic[j].map_or(0.0, |p| p.work) + b.lplastic[j].map_or(0.0, |p| p.work);
        let length = beams::is_length(beam.kind);
        let bent = !s.broken && (s.rest - beam.rest0).abs() > 1e-3 * beam.rest0.abs();
        let row = &mut z[zones::section_index(zone.section)];
        for (side, f) in [(0, zone.left), (1, zone.right)] {
            let c = &mut row[side];
            c[0] += f * f64::from(u8::from(length));
            c[1] += f * f64::from(u8::from(length && bent));
            c[2] += f * f64::from(u8::from(length && s.broken));
            c[3] += f * w;
        }
    }
    let sum = |secs: &[usize], sides: &[usize], q: usize| -> f64 { secs.iter().flat_map(|&s| sides.iter().map(move |&d| (s, d))).map(|(s, d)| z[s][d][q]).sum() };
    let frac = |secs: &[usize], q: usize| -> f64 {
        let n = sum(secs, &[0, 1], 0);
        if n > 0.0 {
            sum(secs, &[0, 1], q) / n
        } else {
            0.0
        }
    };
    let all: Vec<usize> = (0..7).collect();
    let total = sum(&all, &[0, 1], 3);
    put("bent_frac", frac(&all, 1));
    put("broken_frac", frac(&all, 2));
    put("plastic_work_j", total);
    for (i, (_, sn)) in SECTIONS.iter().enumerate() {
        for (d, dn) in [(0, "left"), (1, "right")] {
            let c = z[i][d];
            put(&format!("zone.{sn}_{dn}.work_j"), c[3]);
            put(&format!("zone.{sn}_{dn}.bent_frac"), if c[0] > 0.0 { c[1] / c[0] } else { 0.0 });
            put(&format!("zone.{sn}_{dn}.broken_frac"), if c[0] > 0.0 { c[2] / c[0] } else { 0.0 });
        }
    }
    let (fl, fu, cab, door, rear) = (0usize, 1usize, 2usize, 3usize, 4usize);
    let groups: [(&str, Vec<usize>); 5] = [("front", vec![fl, fu]), ("front_upper", vec![fu]), ("front_lower", vec![fl]), ("cabin", vec![cab, door]), ("rear", vec![rear])];
    for (n, secs) in &groups {
        put(&format!("zone.{n}.work_j"), sum(secs, &[0, 1], 3));
        put(&format!("zone.{n}.work_frac"), if total > 0.0 { sum(secs, &[0, 1], 3) / total } else { 0.0 });
        put(&format!("zone.{n}.bent_frac"), frac(secs, 1));
        put(&format!("zone.{n}.broken_frac"), frac(secs, 2));
    }
    put("zone.front_left.work_j", sum(&[fl, fu], &[0], 3));
    put("zone.front_right.work_j", sum(&[fl, fu], &[1], 3));
    put("zone.left.work_j", sum(&all, &[0], 3));
    put("zone.right.work_j", sum(&all, &[1], 3));
    put("zone.front_lr_ratio", (sum(&[fl, fu], &[0], 3) + 1.0) / (sum(&[fl, fu], &[1], 3) + 1.0));
    put("zone.lr_ratio", (sum(&all, &[0], 3) + 1.0) / (sum(&all, &[1], 3) + 1.0));
    // Gauges.
    for (n, _, _, tr) in &b.gauges {
        tr.put(&mut put, n);
    }
    b.glength.track.put(&mut put, "length");
    b.gleft.track.put(&mut put, "length_left");
    b.gright.track.put(&mut put, "length_right");
    b.gheight.track.put(&mut put, "height");
    // Separation.
    let (label, count) = b.parts(st, pos.len());
    put("detached_parts", count as f64 - b.parts0 as f64);
    put("detached_mass", graph::detached_mass(&label, count, &b.mass));
    total
}
