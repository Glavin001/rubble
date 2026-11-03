//! Vehicle-count scaling of Rubble's explicit-beams GPU solver against Rigs of Rods'
//! own solver (the local reference harness, `inspiration/ror-harness`, when `ror=` or
//! `ROR_BIN` names its binary; Rubble alone otherwise), per scenario, with repeated
//! trials and confidence intervals, and the end state of sampled vehicles compared.
//!
//! ```bash
//! ROR_BIN=<path>/ror_ref cargo run -p engine-webgpu --release --bin explicit_beams_scaling -- \
//!     <file.truck> scenarios=idle,drive,impact counts=1,2,4,8,16,32,64,128,256 \
//!     rubble_counts=512,1024 trials=5 out=<dir>
//! ```
//!
//! Scenarios (each vehicle spawned with its lowest node 2 cm above flat ground, its
//! tyres rolling at the spawn velocity unless `roll=0`, with Rigs of Rods' per-node
//! drag unless `drag=0`, in a
//! grid far enough apart that no vehicle reaches another or another's wall):
//!   idle    at rest
//!   drive   rolling at `speed=` m/s (default 15) along the vehicle's forward axis
//!   impact  as drive, into its own static wall 1 + 0.5 v m ahead (hit about 0.57 s
//!           in at 15 m/s, inside the timed frames)
//!   mixed   a fleet doing what traffic does, vehicle k by k mod 4: standing still;
//!           driving; into a wall; into a narrow pole a quarter of its width off
//!           centre. Moving vehicles at 0.6 to 1.0 × `speed=`, and each wall and pole
//!           placed so its hit lands at a time spread evenly over the timed frames
//!           (from 0.1 s after the warm frames to 0.5 s before the end), so crashes
//!           keep happening through the run (a fixed sequence from k, the same on
//!           both engines). Vehicles never reach one another (both engines run
//!           vehicle–vehicle contact, but it never finds a partner).
//!   pileup  vehicle–vehicle crashes in pairs: vehicles 2g and 2g + 1 form group g,
//!           by g mod 3 a head-on (vehicle 2g + 1 turned 180° and driving at the
//!           same speed towards 2g), a T-bone (2g + 1 turned 90°, standing across
//!           2g's lane, centred on it) or a rear-end (2g + 1 ahead in the same lane
//!           at half 2g's speed). Vehicle 2g at 0.6 to 1.0 × `speed=`; each pair's
//!           gap set so its hit lands at a time spread over the timed frames as in
//!           `mixed` (a fixed sequence from g). Groups sit in a grid of lanes
//!           max(length, width) + 15 m apart and rows 2 × (length + `speed=` ×
//!           run time) + 30 m apart, so only a group's own pair meets. An odd last
//!           vehicle drives alone.
//!   headon, tbone  as `pileup` with every group of that kind (the accuracy
//!           ensembles run one group: two vehicles, the moving vehicle 2g nudged).
//! Every run: `warm=` untimed frames (default 0.5 s worth), then `frames=` timed
//! frames (default 2 s worth) at `fps=` (default 60: 33 steps of 0.5 ms a frame; 120:
//! 17 steps and an 8.33 ms budget). Both engines read one scene file.
//!
//! Timing. Rigs of Rods: wall time per frame with its own per-step thread pool policy.
//! Rubble: the GPU time of each frame (timestamps around the frame's pass), the
//! pipelined wall time (frames submitted back to back, as a game loop does, total
//! over frames), and in a second run the blocking time per frame (submit to
//! completion). Trials alternate which engine runs first; before each run the tool
//! waits (up to `quiet_wait=` s, default 600, then stops without measuring) for the 1-minute load average to drop
//! under `max_load=` (default 3) and, on macOS, the GPU's busy share under
//! `max_gpu=` % (default 10), and records both. Per count it reports the mean of
//! the trial means and its 95 % confidence interval, the p95 frame, and the
//! speed-up (Rigs of Rods' mean over Rubble's, a 95 % interval over the paired
//! trials' log ratios) with a verdict only when the interval excludes 1; and the
//! peak frames: p99, maximum and the share of frames over the frame budget.
//!
//! Accuracy. Crashes (and a vehicle that breaks beams under its own weight) are
//! chaotic: a float-level change to the spawn moves Rigs of Rods' own end state by
//! up to metres. So accuracy compares ensembles. Per scenario, `nudges=` K one-vehicle
//! runs on each engine (default 8; `pileup`, `headon`, `tbone`: one group, two
//! vehicles, node distances over both and broken / bent counts summed; RoR with
//! `threads=1`, its deterministic vehicle order), run k spawned 0.1 mm/s × k faster:
//! * RoR's spread: node RMS between every pair of RoR runs;
//! * Rubble against RoR: node RMS from every Rubble run to every RoR run;
//! * pass when the median Rubble–RoR distance is at most 1.5× RoR's median spread
//!   (or at most 1 mm), with a bootstrap 95 % interval of the ratio reported; and
//!   Rubble's median broken and bent counts lie within RoR's range over its runs,
//!   widened 10 %;
//! * a fleet's first and last vehicle (trial 0, moved to the one-vehicle spawn):
//!   median distance to the RoR runs within the same limit.
//! Every run also compares the sampled vehicles with Rigs of Rods' same vehicle
//! (end state: node positions, broken and bent beams, travel along the forward
//! axis, crush: loss of length along it), reported but not judged.
//!
//! Diagnostics: `engine=cpu` runs the ensembles' Rubble side on the CPU reference
//! (`ExplicitBeamsWorld`, the specification's order); `kernel=det` (the test-only
//! `EB_DET_HITS` build) or `kernel=probe` (the frozen baseline with its race probe)
//! picks the GPU kernel; `ror_reuse=1` keeps an earlier RoR result of the same scene
//! file in `out=`; `diff=1` steps nudge 0's scene on two GPU worlds and the CPU in
//! lockstep and reports, per scenario, the first frame each pair differs bit for bit
//! and the GPU–CPU distance over time (with `kernel=probe`, the frames each race hit).
//!
//! `config=<name>`: the section configuration on both engines (Rigs of Rods'
//! default when absent: the first `section` name in byte order).
//!
//! `trace=<file>` (with `trace_scene=<json>`, `trace_every=`, `cols=` for the grid's
//! columns) records the first scenario and count on Rubble's GPU for the crash
//! viewer instead of benchmarking.
//!
//! Bounds. `ceiling_ms=` (default 0: off) with `ceiling_frames=` (default 10): a
//! run, warm frames included, stops once that many consecutive frames each take
//! longer than the ceiling (RoR: its frame wall time; Rubble: per-frame wall time in
//! the warm and blocking frames, GPU time in the pipelined ones); that engine is too
//! slow at that count and runs no count from there on, in any trial, and the stopped
//! run is kept only as a record. `max_frame_ms=` (default: the ceiling, else 100): before each count, an engine's frame time
//! is predicted from its previous count (linearly in vehicles); past the limit that
//! engine runs no larger count, in any trial (Rubble's limit ends the count for both).
//! `deadline_s=` (default none; quiet-machine waits not counted): a run whose predicted wall time (the previous
//! count's, scaled linearly) would end past the deadline is not started; the trials
//! stop there and the summary covers what ran. `<out>/results.partial.json` holds
//! every finished run after each one.
//!
//! Writes `<out>/results.json` (every frame time and comparison) and
//! `<out>/summary.md`.

use engine_softbody::explicit_beams::{yaw_offset, ExplicitBeams, StaticBox};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::truck::parse_truck;
use engine_webgpu::{default_device, render_scene_json, ExplicitBeamsGpu, KernelVariant};
use glam::Vec3;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Rubble's frame budget lines in the summary.
const BUDGETS_MS: [f64; 3] = [2.0, 4.0, 8.33];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Scenario {
    Idle,
    Drive,
    Impact,
    Mixed,
    Pileup,
    Headon,
    Tbone,
}

impl Scenario {
    fn parse(s: &str) -> Self {
        match s {
            "idle" => Self::Idle,
            "drive" => Self::Drive,
            "impact" => Self::Impact,
            "mixed" => Self::Mixed,
            "pileup" => Self::Pileup,
            "headon" => Self::Headon,
            "tbone" => Self::Tbone,
            _ => panic!("scenario {s}: idle, drive, impact, mixed, pileup, headon or tbone"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Drive => "drive",
            Self::Impact => "impact",
            Self::Mixed => "mixed",
            Self::Pileup => "pileup",
            Self::Headon => "headon",
            Self::Tbone => "tbone",
        }
    }
    /// Vehicles in pairs that crash into each other.
    fn grouped(self) -> bool {
        matches!(self, Self::Pileup | Self::Headon | Self::Tbone)
    }
}

/// The vehicle's axes and extent, at spawn origin zero.
struct Geometry {
    forward: Vec3,
    lateral: Vec3,
    /// Node extent along forward and lateral.
    along: (f32, f32),
    across: (f32, f32),
    /// Node extent in x and z.
    lo: Vec3,
    hi: Vec3,
    /// Turned by 90° and 180° about the spawn point: (extent along forward, centre
    /// across).
    turned: [((f32, f32), f32); 2],
}

impl Geometry {
    fn of(text: &str) -> Self {
        let sim = spawn(text, Vec3::ZERO, Vec3::ZERO);
        let file = parse_truck(text);
        let p = |i: u32| sim.nodes[i as usize].pos;
        let forward = file
            .cameras
            .as_ref()
            .map(|c| {
                let d = p(c.center) - p(c.back);
                Vec3::new(d.x, 0.0, d.z).normalize()
            })
            .unwrap_or(-Vec3::X);
        let lateral = Vec3::Y.cross(forward).normalize();
        let span = |axis: Vec3| {
            sim.nodes.iter().fold((f32::MAX, f32::MIN), |(a, b), n| {
                (a.min(n.pos.dot(axis)), b.max(n.pos.dot(axis)))
            })
        };
        let (lo, hi) = sim.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        });
        let turned = [90.0, 180.0].map(|yaw| {
            let t: Vec<Vec3> = sim.nodes.iter().map(|n| yaw_offset(n.pos, yaw)).collect();
            let span = |axis: Vec3| {
                t.iter().fold((f32::MAX, f32::MIN), |(a, b), q| {
                    (a.min(q.dot(axis)), b.max(q.dot(axis)))
                })
            };
            let across = span(lateral);
            (span(forward), 0.5 * (across.0 + across.1))
        });
        Self {
            turned,
            forward,
            lateral,
            along: span(forward),
            across: span(lateral),
            lo,
            hi,
        }
    }
}

/// One vehicle: spawned at (x, 0, z), lifted so its lowest node is 2 cm up, its
/// tyres rolling at `vel` (still with `roll=0`).
fn spawn(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    spawn_yaw(text, at, 0.0, vel)
}

/// As [`spawn`], turned by `yaw` degrees about +y around the spawn point (Rigs of
/// Rods' `vehicle … yaw` scene field); 0 is exactly [`spawn`].
fn spawn_yaw(text: &str, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let drag = DRAG.load(std::sync::atomic::Ordering::Relaxed);
    let settings = engine_softbody::explicit_beams::ExplicitBeamsSettings {
        drag,
        turbulence: drag,
        ..Default::default()
    };
    let config = CONFIG.get().and_then(|c| c.as_deref());
    let at = Vec3::new(at.x, 0.0, at.z);
    let mut sim = if yaw == 0.0 {
        ExplicitBeams::from_truck_config(text, config, at, vel, settings)
    } else {
        ExplicitBeams::from_truck_config_yaw(text, config, at, yaw, vel, settings)
    }
    .expect("explicit beams spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    if ROLL.load(std::sync::atomic::Ordering::Relaxed) {
        sim.roll_tyres(vel);
    }
    sim
}

/// `wide=0|1`: force one workgroup per vehicle or the wide passes (default: the
/// host's choice, wide for large vehicles).
static WIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

fn apply_wide(gpu: &mut ExplicitBeamsGpu) {
    let on = match WIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => gpu.set_wide(false),
        1 => gpu.set_wide(true),
        _ => gpu.is_wide(),
    };
    if std::env::var("EB_SHOW_WIDE").is_ok() {
        eprintln!("wide passes: {on}");
    }
}

/// `config=<name>`: the section configuration (both engines).
static CONFIG: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// `roll=0`: spawn tyres still (every node at the spawn velocity) on both engines.
static ROLL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// `drag=0`: no per-node aerodynamic drag or turbulence on both engines (that model
/// gives every node the same force, so a vehicle of many light nodes brakes hard).
static DRAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

struct Scene {
    /// Spawn point (y unused) and velocity.
    vehicles: Vec<(Vec3, Vec3)>,
    /// Per vehicle its wall (centre, half extents), if any.
    walls: Vec<Option<(Vec3, Vec3)>>,
    /// Per vehicle its yaw (degrees about +y, around the spawn point).
    yaws: Vec<f32>,
}

/// `mixed`: vehicle k's role (k mod 4: 0 still, 1 driving, 2 into a wall, 3 into a
/// pole), its speed factor in [0.6, 1.0] and its hit time as a share of the timed
/// window, both from a fixed low-discrepancy sequence in k.
fn mixed_role(k: usize) -> (usize, f32, f32) {
    let golden = |i: usize, a: f64| ((i as f64 + 1.0) * a).fract() as f32;
    (
        k % 4,
        0.6 + 0.4 * golden(k, 0.618_033_988_75),
        golden(k, 0.754_877_666_25),
    )
}

/// Seconds of warm frames before the timed ones (for `mixed` hit times).
static WARM_S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();

/// `pileup`, `headon`, `tbone`: groups of two (see the module docs); `extra` m/s
/// added to each group's moving vehicle 2g.
fn group_scene(
    sc: Scenario,
    n: usize,
    g: &Geometry,
    speed: f32,
    seconds: f32,
    extra: f32,
    cols: Option<usize>,
) -> Scene {
    let groups = n.div_ceil(2);
    let cols = cols
        .unwrap_or((groups as f64).sqrt().ceil() as usize)
        .max(1);
    let width = g.across.1 - g.across.0;
    let length = g.along.1 - g.along.0;
    let lateral_step = width.max(length) + 15.0;
    let row_step = 2.0 * (length + speed * seconds) + 30.0;
    let warm_s = WARM_S.get().copied().unwrap_or(0.0);
    let mid = 0.5 * (g.across.0 + g.across.1);
    let mut vehicles = Vec::with_capacity(n);
    let mut yaws = Vec::with_capacity(n);
    for grp in 0..groups {
        let o = g.lateral * ((grp % cols) as f32 * lateral_step)
            - g.forward * ((grp / cols) as f32 * row_step);
        let (_, f, when) = mixed_role(grp);
        let kind = match sc {
            Scenario::Headon => 0,
            Scenario::Tbone => 1,
            _ => grp % 3,
        };
        let v0 = speed * f;
        let t_hit = (warm_s + 0.1) + when * (seconds - warm_s - 0.6).max(0.0);
        vehicles.push((o, g.forward * (v0 + extra)));
        yaws.push(0.0);
        if 2 * grp + 1 == n {
            break;
        }
        let (at, vel, yaw) = match kind {
            0 => {
                let ((lo, _), c) = g.turned[1];
                let gap = 2.0 * v0 * t_hit;
                (
                    o + g.forward * (g.along.1 + gap - lo) + g.lateral * (mid - c),
                    -g.forward * v0,
                    180.0,
                )
            }
            1 => {
                let ((lo, _), c) = g.turned[0];
                let gap = v0 * t_hit;
                (
                    o + g.forward * (g.along.1 + gap - lo) + g.lateral * (mid - c),
                    Vec3::ZERO,
                    90.0,
                )
            }
            _ => {
                let gap = 0.5 * v0 * t_hit;
                (
                    o + g.forward * (g.along.1 + gap - g.along.0),
                    g.forward * (0.5 * v0),
                    0.0,
                )
            }
        };
        vehicles.push((at, vel));
        yaws.push(yaw);
    }
    // Everything at x, z ≥ 100 (Rigs of Rods' collision hash): each group's whole
    // rectangle (its lane, its row) counted.
    let mut min = Vec3::MAX;
    for grp in 0..groups {
        let o = vehicles[2 * grp].0;
        for a in [g.along.0 - 5.0, g.along.0 + row_step] {
            for b in [-0.5 * lateral_step, 0.5 * lateral_step] {
                min = min.min(o + g.forward * a + g.lateral * (mid + b));
            }
        }
    }
    let shift = Vec3::new(100.0 - min.x, 0.0, 100.0 - min.z);
    Scene {
        vehicles: vehicles.into_iter().map(|(o, v)| (o + shift, v)).collect(),
        walls: vec![None; n],
        yaws,
    }
}

fn scene(
    sc: Scenario,
    n: usize,
    g: &Geometry,
    speed: f32,
    seconds: f32,
    extra: f32,
    cols: Option<usize>,
) -> Scene {
    if sc.grouped() {
        return group_scene(sc, n, g, speed, seconds, extra, cols);
    }
    let v = if sc == Scenario::Idle { 0.0 } else { speed };
    let cols = cols.unwrap_or((n as f64).sqrt().ceil() as usize).max(1);
    let width = g.across.1 - g.across.0;
    let length = g.along.1 - g.along.0;
    let travel = speed * seconds + 20.0;
    // Mixed: lanes 12 m apart, so a vehicle deflected by a pole cannot reach the next.
    let lateral_step = width + if sc == Scenario::Mixed { 12.0 } else { 3.0 };
    let row_step = length + travel;
    let mut origins: Vec<Vec3> = (0..n)
        .map(|k| {
            g.lateral * ((k % cols) as f32 * lateral_step)
                - g.forward * ((k / cols) as f32 * row_step)
        })
        .collect();
    // Everything (nodes, and where they can travel) at x, z ≥ 100: Rigs of Rods'
    // collision hash covers non-negative coordinates only.
    let reach = g.forward * travel;
    let mut min = Vec3::MAX;
    for o in &origins {
        min = min
            .min(*o + g.lo)
            .min(*o + g.lo + reach)
            .min(*o + g.hi + reach);
    }
    let shift = Vec3::new(100.0 - min.x, 0.0, 100.0 - min.z);
    for o in &mut origins {
        *o += shift;
    }
    let warm_s = WARM_S.get().copied().unwrap_or(0.0);
    // Per vehicle: speed and, for a wall or pole, (gap to the front, half width
    // across, lateral offset from the vehicle's centre line).
    let plan: Vec<(f32, Option<(f32, f32, f32)>)> = (0..n)
        .map(|k| match sc {
            Scenario::Idle => (0.0, None),
            Scenario::Drive => (v, None),
            Scenario::Impact => (v, Some((1.0 + 0.5 * speed, 0.5 * width + 1.0, 0.0))),
            Scenario::Mixed => {
                let (role, f, when) = mixed_role(k);
                let vk = speed * f;
                let t_hit = (warm_s + 0.1) + when * (seconds - warm_s - 0.6).max(0.0);
                let gap = 1.0 + vk * t_hit;
                match role {
                    0 => (0.0, None),
                    1 => (vk, None),
                    2 => (vk, Some((gap, 0.5 * width + 1.0, 0.0))),
                    _ => (vk, Some((gap, 0.3, 0.25 * width))),
                }
            }
            Scenario::Pileup | Scenario::Headon | Scenario::Tbone => unreachable!(),
        })
        .collect();
    let walls = origins
        .iter()
        .zip(&plan)
        .map(|(o, (_, wall))| {
            wall.map(|(gap, side, offset)| {
                let face = o.dot(g.forward) + g.along.1 + gap;
                let half_t = if side < 1.0 { 0.3 } else { 0.5 };
                let mid = o.dot(g.lateral) + 0.5 * (g.across.0 + g.across.1) + offset;
                let centre = g.forward * (face + half_t) + g.lateral * mid + Vec3::Y * 3.0;
                let half = if g.forward.x.abs() > 0.5 {
                    Vec3::new(half_t, 3.0, side)
                } else {
                    Vec3::new(side, 3.0, half_t)
                };
                (centre, half)
            })
        })
        .collect();
    Scene {
        vehicles: origins
            .into_iter()
            .zip(&plan)
            .map(|(o, (vk, _))| (o, g.forward * (vk + extra)))
            .collect(),
        walls,
        yaws: vec![0.0; n],
    }
}

fn write_scene(path: &Path, s: &Scene) {
    let mut t = String::new();
    for w in s.walls.iter().flatten() {
        let (c, h) = w;
        writeln!(t, "box {} {} {} {} {} {}", c.x, c.y, c.z, h.x, h.y, h.z).unwrap();
    }
    for ((o, v), yaw) in s.vehicles.iter().zip(&s.yaws) {
        if *yaw == 0.0 {
            writeln!(t, "vehicle {} {} {} {} {}", o.x, o.z, v.x, v.y, v.z).unwrap();
        } else {
            writeln!(t, "vehicle {} {} {} {} {} {}", o.x, o.z, v.x, v.y, v.z, yaw).unwrap();
        }
    }
    std::fs::write(path, t).unwrap();
}

/// End state of one vehicle.
#[derive(Clone)]
struct EndState {
    index: usize,
    pos: Vec<Vec3>,
    broken: usize,
    bent: usize,
}

impl EndState {
    fn json(&self, g: &Geometry, start: &[Vec3]) -> Value {
        json!({
            "index": self.index,
            "broken": self.broken,
            "bent": self.bent,
            "travel_m": self.travel(g, start),
            "crush_cm": self.crush(g, start) * 100.0,
        })
    }
    fn travel(&self, g: &Geometry, start: &[Vec3]) -> f32 {
        let c = |p: &[Vec3]| p.iter().copied().sum::<Vec3>() / p.len() as f32;
        (c(&self.pos) - c(start)).dot(g.forward)
    }
    /// Loss of length along the forward axis.
    fn crush(&self, g: &Geometry, start: &[Vec3]) -> f32 {
        let len = |p: &[Vec3]| {
            let (a, b) = p.iter().fold((f32::MAX, f32::MIN), |(a, b), q| {
                (a.min(q.dot(g.forward)), b.max(q.dot(g.forward)))
            });
            b - a
        };
        len(start) - len(&self.pos)
    }
}

struct Run {
    frame_ms: Vec<f64>,
    end: Vec<EndState>,
    /// Frame (warm frames counted) at which the frame ceiling stopped the run.
    stopped_at: Option<u32>,
}

struct RubbleRun {
    gpu_ms: Vec<f64>,
    pipelined_ms: f64,
    blocking_ms: Vec<f64>,
    end: Vec<EndState>,
    stopped_at: Option<u32>,
}

/// The frame ceiling: a run stops after `frames` consecutive frames slower than
/// `ms` (0: off).
#[derive(Clone, Copy)]
struct Ceiling {
    ms: f64,
    frames: u32,
}

impl Ceiling {
    /// Index of the frame ending the first run of `frames` consecutive slow ones.
    fn hit(self, frame_ms: impl IntoIterator<Item = f64>) -> Option<u32> {
        if self.ms <= 0.0 {
            return None;
        }
        let mut slow = 0;
        for (i, t) in frame_ms.into_iter().enumerate() {
            slow = if t > self.ms { slow + 1 } else { 0 };
            if slow >= self.frames {
                return Some(i as u32);
            }
        }
        None
    }
}

/// One scenario's accuracy ensemble: per nudged run its spawn positions and each
/// engine's end state.
#[derive(Default)]
struct Ensemble {
    start: Vec<Vec<Vec3>>,
    ror: Vec<EndState>,
    rubble: Vec<EndState>,
}

impl Ensemble {
    fn json(&self, g: &Geometry) -> Value {
        let side = |v: &[EndState]| -> Vec<Value> {
            v.iter()
                .zip(&self.start)
                .map(|(e, s)| e.json(g, s))
                .collect()
        };
        json!({"ror": side(&self.ror), "rubble": side(&self.rubble)})
    }

    /// The distances, their medians and ratio (with a bootstrap 95 % interval over
    /// runs), and the broken and bent counts.
    fn judge(&self) -> Value {
        let k = self.ror.len();
        let rms = |a: &EndState, b: &EndState| f64::from(rms_max(&a.pos, &b.pos).0) * 1e3;
        let rr: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..k).map(|j| rms(&self.ror[i], &self.ror[j])).collect())
            .collect();
        let xr: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..k).map(|j| rms(&self.rubble[i], &self.ror[j])).collect())
            .collect();
        let medians = |xi: &[usize], ri: &[usize]| {
            let mut a = Vec::new();
            for (p, &i) in ri.iter().enumerate() {
                for &j in &ri[p + 1..] {
                    if i != j {
                        a.push(rr[i][j]);
                    }
                }
            }
            let xr = &xr;
            let b: Vec<f64> = xi
                .iter()
                .flat_map(|&i| ri.iter().map(move |&j| xr[i][j]))
                .collect();
            (median(&b), median(&a))
        };
        let all: Vec<usize> = (0..k).collect();
        let (mx, mr) = medians(&all, &all);
        let floor = |m: f64| m.max(1.0 / 1.5);
        let ratio = mx / floor(mr);
        // Bootstrap: resample each engine's runs with replacement (fixed seed).
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut draw = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % k as u64) as usize
        };
        let mut ratios: Vec<f64> = (0..2000)
            .filter_map(|_| {
                let xi: Vec<usize> = (0..k).map(|_| draw()).collect();
                let ri: Vec<usize> = (0..k).map(|_| draw()).collect();
                let (x, r) = medians(&xi, &ri);
                (!r.is_nan()).then(|| x / floor(r))
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        let q = |p: f64| {
            ratios
                .get(((ratios.len() - 1) as f64 * p).round() as usize)
                .copied()
                .unwrap_or(f64::NAN)
        };
        let counts = |f: fn(&EndState) -> usize| {
            let r: Vec<f64> = self.ror.iter().map(|e| f(e) as f64).collect();
            let x: Vec<f64> = self.rubble.iter().map(|e| f(e) as f64).collect();
            json!({"rubble_median": median(&x), "ror_min": r.iter().copied().fold(f64::MAX, f64::min),
                   "ror_max": r.iter().copied().fold(f64::MIN, f64::max), "rubble": x, "ror": r})
        };
        json!({
            "runs": k, "median_rubble_ror_mm": mx, "median_ror_ror_mm": mr,
            "ratio": ratio, "ratio_ci": [q(0.025), q(0.975)],
            "ror_ror_mm": rr, "rubble_ror_mm": xr,
            "broken": counts(|e| e.broken), "bent": counts(|e| e.bent),
        })
    }

    /// Median node RMS (mm) from a fleet vehicle, moved by its spawn offset to the
    /// one-vehicle spawn, to every RoR run.
    fn median_to_ror(&self, end: &EndState, start: &[Vec3]) -> f64 {
        let shift = self.start[0][0] - start[0];
        let moved = EndState {
            pos: end.pos.iter().map(|p| *p + shift).collect(),
            ..end.clone()
        };
        let d: Vec<f64> = self
            .ror
            .iter()
            .map(|r| f64::from(rms_max(&moved.pos, &r.pos).0) * 1e3)
            .collect();
        median(&d)
    }
}

fn median(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// A fleet (one vehicle, or one crash group) on Rubble's GPU for `frames` frames:
/// its sampled vehicles' end states.
/// `engine=cpu`: the accuracy ensembles' Rubble runs on the CPU reference
/// (`ExplicitBeamsWorld`, the specification's order) instead of the GPU.
static CPU_ENGINE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `kernel=det`: the GPU runs the test-only `EB_DET_HITS` kernel variant (hits applied
/// on one lane in a fixed order); `kernel=current` (default) the shipped kernel.
static KERNEL: std::sync::OnceLock<Option<KernelVariant>> = std::sync::OnceLock::new();

fn apply_kernel(gpu: &mut ExplicitBeamsGpu) {
    if let Some(Some(v)) = KERNEL.get() {
        gpu.set_kernel_variant(*v);
    }
}

/// The CPU reference's end state of `fleet` after `frames` frames of `steps` steps.
fn run_cpu_end(fleet: &[ExplicitBeams], frames: u32, steps: u32) -> Vec<EndState> {
    let mut w = ExplicitBeamsWorld::new(fleet.to_vec());
    for _ in 0..frames {
        w.frame(steps);
    }
    samples(fleet.len())
        .into_iter()
        .map(|k| {
            let v = &w.vehicles[k];
            EndState {
                index: k,
                pos: v.positions(),
                broken: v.beams.iter().filter(|b| b.broken).count(),
                bent: v.beams.iter().filter(|b| b.bent()).count(),
            }
        })
        .collect()
}

fn run_rubble_end(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fleet: &[ExplicitBeams],
    frames: u32,
    steps: u32,
) -> Vec<EndState> {
    if CPU_ENGINE.load(std::sync::atomic::Ordering::Relaxed) {
        return run_cpu_end(fleet, frames, steps);
    }
    let mut gpu = ExplicitBeamsGpu::new(device, queue, fleet);
    apply_wide(&mut gpu);
    apply_kernel(&mut gpu);
    for _ in 0..frames {
        gpu.frame(steps);
    }
    gpu.wait();
    check_contact(&gpu);
    rubble_end(&gpu, fleet.len())
}

/// Vehicle contact's frame bounds held (a violation is an error, never a silently
/// missed contact).
fn check_contact(gpu: &ExplicitBeamsGpu) {
    if let Err(e) = gpu.contact_status() {
        panic!("Rubble vehicle contact: {e}");
    }
}

/// Grouped scenarios: at the end of a run no vehicle's box (padded 5 cm, the
/// contact partner test) overlaps a vehicle of another group, so only the intended
/// pairs met (a sort and sweep along x).
fn check_groups_apart(gpu: &ExplicitBeamsGpu) {
    if !GROUPED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let pos = gpu.positions();
    let mut boxes: Vec<(Vec3, Vec3, usize)> = pos
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let (lo, hi) = p
                .iter()
                .fold((Vec3::MAX, Vec3::MIN), |(a, b), q| (a.min(*q), b.max(*q)));
            (lo - Vec3::splat(0.05), hi + Vec3::splat(0.05), k)
        })
        .collect();
    boxes.sort_by(|a, b| a.0.x.total_cmp(&b.0.x));
    for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            let (a, b) = (&boxes[i], &boxes[j]);
            if b.0.x > a.1.x {
                break;
            }
            let overlap = a.0.cmple(b.1).all() && b.0.cmple(a.1).all();
            assert!(
                !overlap || a.2 / 2 == b.2 / 2,
                "vehicles {} and {} of different groups overlap at the end of the run",
                a.2,
                b.2
            );
        }
    }
}

/// Whether the scenario being run is a grouped one (`pileup`, `headon`, `tbone`):
/// its samples are the first and last group's vehicles.
static GROUPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn samples(n: usize) -> Vec<usize> {
    if GROUPED.load(std::sync::atomic::Ordering::Relaxed) {
        let mut v: Vec<usize> = [0, 1, n.saturating_sub(2), n - 1]
            .into_iter()
            .filter(|&k| k < n)
            .collect();
        v.dedup();
        return v;
    }
    if n > 1 {
        vec![0, n - 1]
    } else {
        vec![0]
    }
}

/// One end state of several vehicles (a crash group): positions concatenated,
/// counts summed.
fn combine(ends: &[EndState]) -> EndState {
    EndState {
        index: ends[0].index,
        pos: ends.iter().flat_map(|e| e.pos.iter().copied()).collect(),
        broken: ends.iter().map(|e| e.broken).sum(),
        bent: ends.iter().map(|e| e.bent).sum(),
    }
}

fn rubble_fleet(text: &str, s: &Scene) -> Vec<ExplicitBeams> {
    s.vehicles
        .iter()
        .zip(&s.walls)
        .zip(&s.yaws)
        .map(|(((o, v), w), yaw)| {
            let mut sim = spawn_yaw(text, *o, *yaw, *v);
            if let Some((c, h)) = w {
                sim.boxes.push(StaticBox::centered(*c, *h));
            }
            sim
        })
        .collect()
}

fn rubble_end(gpu: &ExplicitBeamsGpu, n: usize) -> Vec<EndState> {
    let pos = gpu.positions();
    let beams = gpu.beams();
    samples(n)
        .into_iter()
        .map(|k| EndState {
            index: k,
            pos: pos[k].clone(),
            broken: beams[k].iter().filter(|b| b.broken).count(),
            bent: beams[k].iter().filter(|b| b.bent()).count(),
        })
        .collect()
}

fn run_rubble(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fleet: &[ExplicitBeams],
    warm: u32,
    frames: u32,
    steps: u32,
    ceiling: Ceiling,
) -> RubbleRun {
    let n = fleet.len();
    // Pipelined, with the GPU time of each frame. The untimed warm frames each wait
    // for completion, so the ceiling can stop a run before its timed frames.
    let mut gpu = ExplicitBeamsGpu::new(device, queue, fleet);
    apply_wide(&mut gpu);
    apply_kernel(&mut gpu);
    let warm_ms: Vec<f64> = (0..warm)
        .map(|_| {
            let t = Instant::now();
            gpu.frame(steps);
            gpu.wait();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    if let Some(at) = ceiling.hit(warm_ms.iter().copied()) {
        return RubbleRun {
            gpu_ms: Vec::new(),
            pipelined_ms: f64::NAN,
            blocking_ms: Vec::new(),
            end: rubble_end(&gpu, n),
            stopped_at: Some(at),
        };
    }
    let timed = gpu.enable_frame_timing(frames);
    let t = Instant::now();
    for _ in 0..frames {
        gpu.frame(steps);
    }
    gpu.wait();
    let pipelined_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(frames);
    let gpu_ms = if timed {
        gpu.frame_times_ms()
    } else {
        Vec::new()
    };
    check_contact(&gpu);
    check_groups_apart(&gpu);
    let end = rubble_end(&gpu, n);
    drop(gpu);
    if let Some(at) = ceiling.hit(gpu_ms.iter().copied()) {
        return RubbleRun {
            gpu_ms,
            pipelined_ms,
            blocking_ms: Vec::new(),
            end,
            stopped_at: Some(warm + at),
        };
    }
    // Blocking: submit to completion, every frame.
    let mut gpu = ExplicitBeamsGpu::new(device, queue, fleet);
    apply_wide(&mut gpu);
    apply_kernel(&mut gpu);
    for _ in 0..warm {
        gpu.frame(steps);
    }
    gpu.wait();
    let mut blocking_ms = Vec::new();
    let mut stopped_at = None;
    for _ in 0..frames {
        let t = Instant::now();
        gpu.frame(steps);
        gpu.wait();
        blocking_ms.push(t.elapsed().as_secs_f64() * 1e3);
        if let Some(at) = ceiling.hit(
            blocking_ms
                .iter()
                .copied()
                .rev()
                .take(ceiling.frames as usize),
        ) {
            stopped_at = Some(warm + blocking_ms.len() as u32 - ceiling.frames + at);
            break;
        }
    }
    gpu.wait();
    check_contact(&gpu);
    RubbleRun {
        gpu_ms,
        pipelined_ms,
        blocking_ms,
        end,
        stopped_at,
    }
}

fn run_ror(
    bin: &Path,
    truck: &Path,
    scene_file: &Path,
    n: usize,
    warm: u32,
    frames: u32,
    steps: u32,
    ceiling: Ceiling,
    threads: Option<u32>,
) -> Run {
    let out = scene_file.with_extension("ror.json");
    let sample = samples(n)
        .iter()
        .map(|k| k.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let status = std::process::Command::new(bin)
        .arg(truck)
        .arg("scale")
        .arg(format!("scene={}", scene_file.display()))
        .arg(format!("warm={warm}"))
        .arg(format!("frames={frames}"))
        .arg(format!("steps={steps}"))
        .arg(format!(
            "roll={}",
            ROLL.load(std::sync::atomic::Ordering::Relaxed) as u8
        ))
        .arg(format!(
            "drag={}",
            DRAG.load(std::sync::atomic::Ordering::Relaxed) as u8
        ))
        .arg(format!("sample={sample}"))
        .arg(format!("ceiling_ms={}", ceiling.ms))
        .arg(format!("ceiling_frames={}", ceiling.frames))
        .args(threads.map(|t| format!("threads={t}")))
        .args(
            CONFIG
                .get()
                .and_then(|c| c.as_deref())
                .map(|c| format!("config={c}")),
        )
        .arg(format!("out={}", out.display()))
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .expect("ror_ref");
    assert!(status.success(), "ror_ref scale failed");
    read_ror(&out)
}

/// A `ror_ref scale` result file.
fn read_ror(out: &Path) -> Run {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let frame_ms = v["frame_ms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let end = v["vehicles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let p: Vec<f32> = e["pos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            EndState {
                index: e["index"].as_u64().unwrap() as usize,
                pos: p.chunks(3).map(Vec3::from_slice).collect(),
                broken: e["broken"].as_u64().unwrap() as usize,
                bent: e["bent"].as_u64().unwrap() as usize,
            }
        })
        .collect();
    let stopped_at = v["stopped_at"]
        .as_i64()
        .filter(|&f| f >= 0)
        .map(|f| f as u32);
    Run {
        frame_ms,
        end,
        stopped_at,
    }
}

fn load_average() -> f64 {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    // "{ 1.23 1.45 1.67 }"
    out.split_whitespace()
        .find_map(|t| t.parse().ok())
        .unwrap_or(f64::NAN)
}

/// Other processes' share of the GPU in percent over one second (macOS: the
/// accelerator's per-client accumulated GPU time, every client but this process);
/// NaN where it cannot be read. The device-wide utilisation counter is not used:
/// it reads high while no process is using the GPU.
fn gpu_busy() -> f64 {
    // Per client process: accumulated GPU nanoseconds.
    let sample = || -> Option<HashMap<String, u64>> {
        let xml = std::process::Command::new("ioreg")
            .args(["-r", "-c", "AGXDeviceUserClient", "-w", "0", "-a"])
            .output()
            .ok()?
            .stdout;
        let mut plutil = std::process::Command::new("plutil")
            .args(["-convert", "json", "-o", "-", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .ok()?;
        use std::io::Write;
        plutil.stdin.take()?.write_all(&xml).ok()?;
        let out = plutil.wait_with_output().ok()?;
        let v: Value = serde_json::from_slice(&out.stdout).ok()?;
        let me = format!("pid {},", std::process::id());
        let mut t = HashMap::new();
        for c in v.as_array()? {
            let who = c["IOUserClientCreator"].as_str().unwrap_or("?").to_string();
            if who.starts_with(&me) {
                continue;
            }
            let ns: u64 = c["AppUsage"]
                .as_array()
                .map(|u| {
                    u.iter()
                        .filter_map(|x| x["accumulatedGPUTime"].as_u64())
                        .sum()
                })
                .unwrap_or(0);
            *t.entry(who).or_insert(0) += ns;
        }
        Some(t)
    };
    let Some(a) = sample() else { return f64::NAN };
    let t0 = Instant::now();
    std::thread::sleep(std::time::Duration::from_secs(1));
    let Some(b) = sample() else { return f64::NAN };
    let dt = t0.elapsed().as_secs_f64();
    let used: u64 = b
        .iter()
        .map(|(k, v)| v.saturating_sub(*a.get(k).unwrap_or(v)))
        .sum();
    100.0 * used as f64 * 1e-9 / dt
}

/// Wait for a quiet machine: 1-minute load average under `max_load` and the GPU
/// busy under `max_gpu` % (another process on the GPU slows the GPU runs, one on the
/// CPU the CPU runs). The load and GPU busy share when the run starts.
fn wait_quiet(max_load: f64, max_gpu: f64, max_wait_s: f64) -> (f64, f64) {
    let t = Instant::now();
    loop {
        let (l, g) = (load_average(), gpu_busy());
        if !(l >= max_load) && !(g >= max_gpu) {
            return (l, g);
        }
        // A run on a busy machine would not measure either engine: stop instead.
        if t.elapsed().as_secs_f64() > max_wait_s {
            eprintln!("machine not quiet within {max_wait_s} s: stopping");
            std::process::exit(2);
        }
        eprintln!(
            "load average {l:.2} (max {max_load}), GPU busy {g:.0} % (max {max_gpu}): waiting"
        );
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn percentile(x: &[f64], q: f64) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Two-sided 95 % Student t quantile.
fn t95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if df == 0 {
        f64::INFINITY
    } else {
        T[(df - 1).min(29)]
    }
}

/// Mean and its 95 % interval.
fn ci(x: &[f64]) -> (f64, f64, f64) {
    let m = mean(x);
    if x.len() < 2 {
        return (m, f64::NAN, f64::NAN);
    }
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64;
    let h = t95(x.len() - 1) * (var / x.len() as f64).sqrt();
    (m, m - h, m + h)
}

fn rms_max(a: &[Vec3], b: &[Vec3]) -> (f32, f32) {
    let n = a.len().min(b.len()).max(1);
    let (mut s, mut m) = (0.0f64, 0.0f32);
    for (p, q) in a.iter().zip(b) {
        let d = p.distance(*q);
        s += f64::from(d) * f64::from(d);
        m = m.max(d);
    }
    ((s / n as f64).sqrt() as f32, m)
}

fn kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .find_map(|a| a.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().unwrap())
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let truck = PathBuf::from(
        args.first()
            .expect("usage: explicit_beams_scaling <file.truck> …"),
    );
    let text = std::fs::read_to_string(&truck)?;
    let scenarios: Vec<Scenario> = kv(&args, "scenarios")
        .unwrap_or("idle,drive,impact")
        .split(',')
        .map(Scenario::parse)
        .collect();
    let counts: Vec<usize> = list(kv(&args, "counts").unwrap_or("1,2,4,8,16,32,64,128,256"));
    let rubble_counts: Vec<usize> = list(kv(&args, "rubble_counts").unwrap_or("512,1024"));
    let trials: usize = kv(&args, "trials").map_or(5, |s| s.parse().unwrap());
    // Frame rate: steps per frame (whole 0.5 ms steps, rounded) and the budget.
    let fps: f32 = kv(&args, "fps").map_or(60.0, |s| s.parse().unwrap());
    let steps = (1.0 / (fps * engine_softbody::explicit_beams::STEP)).round() as u32;
    let budget_ms = 1000.0 / f64::from(fps);
    let warm: u32 = kv(&args, "warm").map_or((0.5 * fps).round() as u32, |s| s.parse().unwrap());
    let frames: u32 =
        kv(&args, "frames").map_or((2.0 * fps).round() as u32, |s| s.parse().unwrap());
    let speed: f32 = kv(&args, "speed").map_or(15.0, |s| s.parse().unwrap());
    let max_load: f64 = kv(&args, "max_load").map_or(3.0, |s| s.parse().unwrap());
    let max_gpu: f64 = kv(&args, "max_gpu").map_or(10.0, |s| s.parse().unwrap());
    let quiet_wait: f64 = kv(&args, "quiet_wait").map_or(600.0, |s| s.parse().unwrap());
    let ceiling = Ceiling {
        ms: kv(&args, "ceiling_ms").map_or(0.0, |s| s.parse().unwrap()),
        frames: kv(&args, "ceiling_frames").map_or(10, |s| s.parse().unwrap()),
    };
    let max_frame_ms: f64 = kv(&args, "max_frame_ms")
        .map_or(if ceiling.ms > 0.0 { ceiling.ms } else { 100.0 }, |s| {
            s.parse().unwrap()
        });
    let deadline_s: f64 = kv(&args, "deadline_s").map_or(f64::INFINITY, |s| s.parse().unwrap());
    let began = Instant::now();
    // Time spent waiting for a quiet machine: not counted against the deadline.
    let mut waited = 0.0f64;
    let out = PathBuf::from(kv(&args, "out").unwrap_or("explicit_beams_scaling"));
    std::fs::create_dir_all(&out)?;
    let ror_bin = kv(&args, "ror")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("ROR_BIN").map(PathBuf::from));
    let seconds = (warm + frames) as f32 / fps;
    let _ = WARM_S.set(warm as f32 / fps);
    let cols: Option<usize> = kv(&args, "cols").map(|s| s.parse().unwrap());
    let _ = CONFIG.set(kv(&args, "config").map(str::to_string));
    ROLL.store(
        kv(&args, "roll") != Some("0"),
        std::sync::atomic::Ordering::Relaxed,
    );
    if let Some(w) = kv(&args, "wide") {
        WIDE.store(u8::from(w == "1"), std::sync::atomic::Ordering::Relaxed);
    }
    DRAG.store(
        kv(&args, "drag") != Some("0"),
        std::sync::atomic::Ordering::Relaxed,
    );

    CPU_ENGINE.store(
        kv(&args, "engine") == Some("cpu"),
        std::sync::atomic::Ordering::Relaxed,
    );
    let _ = KERNEL.set(match kv(&args, "kernel") {
        None | Some("current") => None,
        Some("det") => Some(KernelVariant::CurrentDet),
        Some("probe") => Some(KernelVariant::BaselineProbe),
        Some(k) => panic!("kernel={k}: current or det"),
    });

    // Frame timestamps need the timestamp-query feature on the device.
    std::env::set_var("RUBBLE_GPU_TIMING", "1");
    let (info, device, queue) = default_device()?;
    let g = Geometry::of(&text);
    let one = spawn(&text, Vec3::ZERO, Vec3::ZERO);
    for w in &one.warnings {
        eprintln!("warning: {w} (set RUBBLE_AIRFOIL_DIR to the .afl directory)");
    }
    println!(
        "{}: {} nodes, {} beams; {} ({:?}); Rigs of Rods reference: {}",
        truck.display(),
        one.nodes.len(),
        one.beams.len(),
        info.name,
        info.backend,
        ror_bin
            .as_ref()
            .map_or("none".into(), |p| p.display().to_string())
    );

    // `diff=1`: per scenario, nudge 0's scene stepped on the GPU (twice, in
    // lockstep: run-to-run determinism) and on the CPU reference, frame by frame:
    // the first frame each pair differs bit for bit, and the GPU–CPU node distance
    // over time. Then stops.
    if kv(&args, "diff") == Some("1") {
        for &sc in &scenarios {
            let m = if sc.grouped() { 2 } else { 1 };
            let s = scene(sc, m, &g, speed, seconds, 0.0, None);
            let mut fleet = rubble_fleet(&text, &s);
            if kv(&args, "notri") == Some("1") {
                // Diagnostic: no self-collision on either engine.
                for v in &mut fleet {
                    v.candidates.clear();
                }
            }
            let mut cpu = ExplicitBeamsWorld::new(fleet.clone());
            let mut a = ExplicitBeamsGpu::new(&device, &queue, &fleet);
            let mut b = ExplicitBeamsGpu::new(&device, &queue, &fleet);
            for gpu in [&mut a, &mut b] {
                apply_wide(gpu);
                apply_kernel(gpu);
            }
            let (mut first_ab, mut first_cpu) = (None, None);
            // `kernel=probe`: frames whose race probe flagged a contact race (bit 0)
            // or a break-exception race (bit 2).
            let probe = matches!(KERNEL.get(), Some(Some(KernelVariant::BaselineProbe)));
            let (mut race_a, mut race_b, mut race_ab) = (0u32, 0u32, 0u32);
            if probe {
                a.take_race_flag();
            }
            let (mut hit_steps, mut hits, mut max_hits) = (0u64, 0u64, 0u32);
            let total = warm + frames;
            let mut line = String::new();
            for f in 0..total {
                a.frame(steps);
                b.frame(steps);
                if probe {
                    let r = a.take_race_flag();
                    assert_eq!(r & 2, 0, "the race probe gave up");
                    race_a += r & 1;
                    race_b += (r >> 2) & 1;
                    race_ab += u32::from(r & 5 == 5);
                }
                for v in &mut cpu.vehicles {
                    v.check_wings();
                    v.rebase_origin();
                }
                for _ in 0..steps {
                    cpu.step();
                    let h: u32 = cpu.vehicles.iter().map(|v| v.last_contacts).sum();
                    hits += u64::from(h);
                    hit_steps += u64::from(h > 0);
                    max_hits = max_hits.max(h);
                }
                let (pa, pb) = (a.positions(), b.positions());
                let flat =
                    |p: &Vec<Vec<Vec3>>| -> Vec<Vec3> { p.iter().flatten().copied().collect() };
                let (fa, fb) = (flat(&pa), flat(&pb));
                let fc: Vec<Vec3> = cpu.vehicles.iter().flat_map(|v| v.positions()).collect();
                let bits = |x: &[Vec3], y: &[Vec3]| {
                    x.iter()
                        .zip(y)
                        .filter(|(p, q)| {
                            p.to_array().map(f32::to_bits) != q.to_array().map(f32::to_bits)
                        })
                        .count()
                };
                let ab = bits(&fa, &fb);
                let ac = bits(&fa, &fc);
                if ab > 0 && first_ab.is_none() {
                    first_ab = Some(f);
                }
                if ac > 0 && first_cpu.is_none() {
                    first_cpu = Some(f);
                    let i = fa
                        .iter()
                        .zip(&fc)
                        .position(|(p, q)| {
                            p.to_array().map(f32::to_bits) != q.to_array().map(f32::to_bits)
                        })
                        .unwrap();
                    let gv: Vec<Vec3> = a.velocities().into_iter().flatten().collect();
                    let cv: Vec<Vec3> = cpu
                        .vehicles
                        .iter()
                        .flat_map(|v| v.nodes.iter().map(|n| n.vel))
                        .collect();
                    let n0 = cpu.vehicles[0].nodes.len();
                    let nd = &cpu.vehicles[i / n0].nodes[i % n0];
                    println!(
                        "  first differing node {i}: GPU pos {:?} vel {:?}; CPU pos {:?} vel {:?}; flags {:#x}, candidate {}",
                        fa[i], gv[i], fc[i], cv[i], nd.flags, cpu.vehicles[i / n0].candidates.contains(&((i % n0) as u32))
                    );
                    let gb = a.beams();
                    for (j, (x, y)) in gb[0].iter().zip(&cpu.vehicles[0].beams).enumerate() {
                        if x.rest.to_bits() != y.rest.to_bits() || x.broken != y.broken {
                            println!("  first differing beam {j}: GPU rest {} broken {}; CPU rest {} broken {} kind {:?} ends {} {}", x.rest, x.broken, y.rest, y.broken, y.kind, y.a, y.b);
                            break;
                        }
                    }
                }
                let (rms, max) = rms_max(&fa, &fc);
                let (rms_ab, _) = rms_max(&fa, &fb);
                if f % 30 == 29 || f + 1 == total || (ac > 0 && first_cpu == Some(f)) {
                    let _ = writeln!(
                        line,
                        "  frame {f:4}: GPU-CPU rms {:.4} mm max {:.4} mm ({ac} nodes differ); GPU-GPU rms {:.4} mm ({ab} nodes differ)",
                        rms * 1e3,
                        max * 1e3,
                        rms_ab * 1e3
                    );
                }
            }
            if probe {
                println!(
                    "race probe {}: {race_a} of {total} frames with a contact race (a), {race_b} with a break-exception race (b), {race_ab} with both",
                    sc.name()
                );
            }
            println!(
                "diff {}: GPU run-to-run first differs at frame {:?}; GPU vs CPU first differs at frame {:?} (of {total}); CPU self-collision: {hit_steps} steps with hits, {hits} hits, at most {max_hits} in a step\n{line}",
                sc.name(),
                first_ab,
                first_cpu
            );
        }
        return Ok(());
    }

    // `trace=<file>`: the first scenario and count on Rubble's GPU alone, every
    // vehicle's nodes concatenated, sampled every `trace_every=` frames (default 1)
    // through the warm and timed frames; `trace_scene=<json>` the matching render
    // scene (`render_scene_json`). Format: u32 nodes, u32 beams, then per sample
    // every node's position (f32 xyz) and every beam's broken flag (u8).
    if let Some(trace) = kv(&args, "trace") {
        use std::io::Write;
        let (sc, n) = (
            scenarios[0],
            counts
                .first()
                .or(rubble_counts.first())
                .copied()
                .unwrap_or(1),
        );
        let every: u32 = kv(&args, "trace_every").map_or(1, |s| s.parse().unwrap());
        let s = scene(sc, n, &g, speed, seconds, 0.0, cols);
        let fleet = rubble_fleet(&text, &s);
        if let Some(path) = kv(&args, "trace_scene") {
            let boxes: Vec<(Vec3, Vec3)> = s.walls.iter().flatten().copied().collect();
            std::fs::write(
                path,
                render_scene_json(&text, one.nodes.len(), n, &boxes, ""),
            )?;
        }
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
        apply_wide(&mut gpu);
        let mut w = std::io::BufWriter::new(std::fs::File::create(trace)?);
        w.write_all(&((one.nodes.len() * n) as u32).to_le_bytes())?;
        w.write_all(&((one.beams.len() * n) as u32).to_le_bytes())?;
        let mut sample = |gpu: &ExplicitBeamsGpu| -> std::io::Result<()> {
            for v in gpu.positions() {
                for p in v {
                    for c in p.to_array() {
                        w.write_all(&c.to_le_bytes())?;
                    }
                }
            }
            for v in gpu.beams() {
                w.write_all(&v.iter().map(|b| b.broken as u8).collect::<Vec<u8>>())?;
            }
            Ok(())
        };
        sample(&gpu)?;
        for f in 1..=warm + frames {
            gpu.frame(steps);
            if f % every == 0 {
                sample(&gpu)?;
            }
        }
        drop(sample);
        w.flush()?;
        let beams = gpu.beams();
        let broken: usize = beams
            .iter()
            .map(|v| v.iter().filter(|b| b.broken).count())
            .sum();
        let bent: usize = beams
            .iter()
            .map(|v| v.iter().filter(|b| b.bent()).count())
            .sum();
        println!(
            "trace: {} {n} vehicles, {} frames every {every}: {broken} beams broken, {bent} bent in all",
            sc.name(),
            (warm + frames) / every + 1
        );
        return Ok(());
    }

    let mut all: Vec<Value> = Vec::new();
    // Raw results: (scenario, count) -> per trial.
    let mut ror_means: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut ror_frames: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut gpu_means: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut gpu_frames: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut piped: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut blocking: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut blocking_frames: HashMap<(Scenario, usize), Vec<f64>> = HashMap::new();
    let mut accuracy: HashMap<(Scenario, usize), Vec<Value>> = HashMap::new();
    // Trial 0's sampled Rubble vehicles: end state and spawn positions.
    let mut fleet_ends: HashMap<(Scenario, usize), Vec<(EndState, Vec<Vec3>)>> = HashMap::new();
    // Runs that started on a busy machine (the wait gave up): none may count.
    let mut busy_runs = 0usize;
    let mut run_count = 0usize;
    // Bounds: per (scenario, engine) the last count's vehicles, mean frame ms and
    // wall seconds; the first count each engine skips (kept across trials).
    let mut last: HashMap<(Scenario, bool), (usize, f64, f64)> = HashMap::new();
    let mut cut: HashMap<(Scenario, bool), usize> = HashMap::new();
    let mut stopped: Option<String> = None;

    'trials: for trial in 0..trials {
        for &sc in &scenarios {
            GROUPED.store(sc.grouped(), std::sync::atomic::Ordering::Relaxed);
            for &n in counts.iter().chain(&rubble_counts) {
                // (frame ms, wall s) predicted for an engine (true: RoR) at n.
                let predict = |ror: bool| {
                    last.get(&(sc, ror)).map_or((0.0, 0.0), |&(m, f, w)| {
                        let k = n as f64 / m as f64;
                        (f * k, w * k)
                    })
                };
                for ror in [false, true] {
                    if !cut.contains_key(&(sc, ror)) && predict(ror).0 > max_frame_ms {
                        cut.insert((sc, ror), n);
                        eprintln!(
                            "{} {}: predicted {:.1} ms a frame at {n} vehicles > max_frame_ms {max_frame_ms}: no larger count",
                            sc.name(), if ror { "RoR" } else { "Rubble" }, predict(ror).0
                        );
                    }
                }
                let allowed = |ror: bool| cut.get(&(sc, ror)).map_or(true, |&c| n < c);
                if !allowed(false) {
                    continue;
                }
                let with_ror = ror_bin.is_some() && counts.contains(&n) && allowed(true);
                let need = predict(false).1 + if with_ror { predict(true).1 } else { 0.0 };
                let active = began.elapsed().as_secs_f64() - waited;
                if active + need > deadline_s {
                    let why = format!(
                        "deadline: trial {trial} {} {n} vehicles predicted {need:.0} s at {active:.0} s of {deadline_s:.0} s (quiet waits excluded); no further runs",
                        sc.name()
                    );
                    eprintln!("{why}");
                    stopped = Some(why);
                    break 'trials;
                }
                let s = scene(sc, n, &g, speed, seconds, 0.0, cols);
                let file = out.join(format!("scene_{}_{n}.txt", sc.name()));
                write_scene(&file, &s);
                let fleet = rubble_fleet(&text, &s);
                let start: Vec<Vec<Vec3>> =
                    samples(n).iter().map(|&k| fleet[k].positions()).collect();
                let mut ror_run = None;
                let mut rubble_run = None;
                let mut loads = Vec::new();
                // Alternate which engine runs first.
                for first in [trial % 2 == 0, trial % 2 != 0] {
                    if first && with_ror {
                        let w0 = Instant::now();
                        loads.push(wait_quiet(max_load, max_gpu, quiet_wait));
                        waited += w0.elapsed().as_secs_f64();
                        let t0 = Instant::now();
                        let run = run_ror(
                            ror_bin.as_ref().unwrap(),
                            &truck,
                            &file,
                            n,
                            warm,
                            frames,
                            steps,
                            ceiling,
                            None,
                        );
                        last.insert(
                            (sc, true),
                            (n, mean(&run.frame_ms), t0.elapsed().as_secs_f64()),
                        );
                        ror_run = Some(run);
                    } else if !first {
                        let w0 = Instant::now();
                        loads.push(wait_quiet(max_load, max_gpu, quiet_wait));
                        waited += w0.elapsed().as_secs_f64();
                        let t0 = Instant::now();
                        let run = run_rubble(&device, &queue, &fleet, warm, frames, steps, ceiling);
                        let frame = mean(&run.gpu_ms).max(run.pipelined_ms);
                        last.insert((sc, false), (n, frame, t0.elapsed().as_secs_f64()));
                        rubble_run = Some(run);
                    }
                }
                run_count += loads.len();
                busy_runs += loads
                    .iter()
                    .filter(|(l, g)| !(*l < max_load) || *g >= max_gpu)
                    .count();
                let r = rubble_run.unwrap();
                let key = (sc, n);
                // The frame ceiling: a stopped run counts as too slow at this count; that
                // engine runs no count from here on, and the run is kept only as a record.
                let mut ror_run = ror_run;
                if let Some(at) = ror_run.as_ref().and_then(|x| x.stopped_at) {
                    eprintln!("{} RoR {n}: {} consecutive frames over {} ms at frame {at}: no count from {n} on", sc.name(), ceiling.frames, ceiling.ms);
                    cut.insert((sc, true), n);
                    all.push(json!({"trial": trial, "scenario": sc.name(), "vehicles": n, "ror_stopped_at": at,
                        "ror": {"frame_ms": ror_run.take().unwrap().frame_ms}}));
                }
                if let Some(at) = r.stopped_at {
                    eprintln!("{} Rubble {n}: {} consecutive frames over {} ms at frame {at}: no count from {n} on", sc.name(), ceiling.frames, ceiling.ms);
                    cut.insert((sc, false), n);
                    all.push(json!({"trial": trial, "scenario": sc.name(), "vehicles": n, "rubble_stopped_at": at,
                        "rubble": {"gpu_ms": r.gpu_ms, "blocking_ms": r.blocking_ms}}));
                    std::fs::write(
                        out.join("results.partial.json"),
                        serde_json::to_string(&all)?,
                    )?;
                    continue;
                }
                if trial == 0 {
                    fleet_ends.insert(
                        key,
                        if sc.grouped() {
                            // The first group (the same crash at every count) as one.
                            let k = r.end.len().min(2);
                            vec![(combine(&r.end[..k]), start[..k].concat())]
                        } else {
                            r.end.iter().cloned().zip(start.iter().cloned()).collect()
                        },
                    );
                }
                gpu_means.entry(key).or_default().push(mean(&r.gpu_ms));
                gpu_frames.entry(key).or_default().extend(&r.gpu_ms);
                piped.entry(key).or_default().push(r.pipelined_ms);
                blocking.entry(key).or_default().push(mean(&r.blocking_ms));
                blocking_frames
                    .entry(key)
                    .or_default()
                    .extend(&r.blocking_ms);
                let mut run_json = json!({
                    "trial": trial, "scenario": sc.name(), "vehicles": n, "load_average_gpu_busy": loads,
                    "rubble": {
                        "gpu_ms": r.gpu_ms, "pipelined_ms": r.pipelined_ms,
                        "blocking_ms": r.blocking_ms,
                        "vehicles": r.end.iter().zip(&start).map(|(e, s0)| e.json(&g, s0)).collect::<Vec<_>>(),
                    },
                });
                if let Some(ror) = ror_run {
                    ror_means.entry(key).or_default().push(mean(&ror.frame_ms));
                    ror_frames.entry(key).or_default().extend(&ror.frame_ms);
                    let cmp: Vec<Value> = r
                        .end
                        .iter()
                        .zip(&ror.end)
                        .zip(&start)
                        .map(|((a, b), s0)| {
                            let (rms, max) = rms_max(&a.pos, &b.pos);
                            json!({
                                "index": a.index, "rms_mm": rms * 1e3, "max_mm": max * 1e3,
                                "rubble": a.json(&g, s0), "ror": b.json(&g, s0),
                            })
                        })
                        .collect();
                    if trial == 0 {
                        accuracy.insert(key, cmp.clone());
                    }
                    run_json["ror"] = json!({
                        "frame_ms": ror.frame_ms,
                        "vehicles": ror.end.iter().zip(&start).map(|(e, s0)| e.json(&g, s0)).collect::<Vec<_>>(),
                    });
                    run_json["comparison"] = Value::Array(cmp);
                }
                eprintln!(
                    "trial {trial} {} {n:5}: rubble gpu {:.3} ms, pipelined {:.3} ms{}",
                    sc.name(),
                    mean(&r.gpu_ms),
                    r.pipelined_ms,
                    ror_means
                        .get(&key)
                        .filter(|_| with_ror)
                        .map_or(String::new(), |v| format!(", ror {:.3} ms", v[v.len() - 1]))
                );
                all.push(run_json);
                std::fs::write(
                    out.join("results.partial.json"),
                    serde_json::to_string(&all)?,
                )?;
            }
        }
    }
    // The summary covers the counts that ran.
    let counts: Vec<usize> = counts
        .into_iter()
        .filter(|n| scenarios.iter().all(|s| gpu_means.contains_key(&(*s, *n))))
        .collect();
    let rubble_counts: Vec<usize> = rubble_counts
        .into_iter()
        .filter(|n| scenarios.iter().all(|s| gpu_means.contains_key(&(*s, *n))))
        .collect();

    // Accuracy ensembles: per scenario K one-vehicle runs per engine, run k spawned
    // 0.1 mm/s × k faster.
    let nudges: usize = kv(&args, "nudges").map_or(8, |s| s.parse().unwrap());
    let ror_reuse = kv(&args, "ror_reuse") == Some("1");
    let mut ensembles: HashMap<Scenario, Ensemble> = HashMap::new();
    if let Some(bin) = &ror_bin {
        for &sc in &scenarios {
            GROUPED.store(sc.grouped(), std::sync::atomic::Ordering::Relaxed);
            // One vehicle, or one crash group of two (RoR single-threaded: its
            // deterministic vehicle order).
            let (m, threads) = if sc.grouped() {
                (2, Some(1))
            } else {
                (1, None)
            };
            let mut e = Ensemble::default();
            for k in 0..nudges {
                let s = scene(sc, m, &g, speed, seconds, k as f32 * 1e-4, None);
                let file = out.join(format!("ens_{}_{k}.txt", sc.name()));
                // `ror_reuse=1`: keep an earlier RoR run of the same scene file (RoR
                // is deterministic for a scene: `threads=1` for groups).
                let before = std::fs::read_to_string(&file).ok();
                write_scene(&file, &s);
                let same = ror_reuse
                    && before.as_deref() == std::fs::read_to_string(&file).ok().as_deref()
                    && file.with_extension("ror.json").exists();
                let fleet = rubble_fleet(&text, &s);
                e.start
                    .push(fleet.iter().flat_map(|v| v.positions()).collect());
                e.ror.push(combine(
                    &if same {
                        read_ror(&file.with_extension("ror.json"))
                    } else {
                        run_ror(
                            bin,
                            &truck,
                            &file,
                            m,
                            warm,
                            frames,
                            steps,
                            Ceiling { ms: 0.0, frames: 1 },
                            threads,
                        )
                    }
                    .end,
                ));
                e.rubble.push(combine(&run_rubble_end(
                    &device,
                    &queue,
                    &fleet,
                    warm + frames,
                    steps,
                )));
                eprintln!(
                    "ensemble {} {k}: broken rubble {} / ror {}",
                    sc.name(),
                    e.rubble[k].broken,
                    e.ror[k].broken
                );
            }
            ensembles.insert(sc, e);
        }
    }
    let judged: HashMap<Scenario, Value> =
        ensembles.iter().map(|(sc, e)| (*sc, e.judge())).collect();

    // Summary.
    let mut md = String::new();
    writeln!(
        md,
        "# Explicit beams scaling: Rubble GPU vs Rigs of Rods\n\n`{}`: {} nodes, {} beams. {} ({:?}). {} trials × {} timed frames ({} untimed first) at {} fps: {} steps of 0.5 ms per frame, a {:.2} ms frame budget; drive and impact at {} m/s.\n",
        truck.file_name().unwrap().to_string_lossy(),
        one.nodes.len(),
        one.beams.len(),
        info.name,
        info.backend,
        trials,
        frames,
        warm,
        fps,
        steps,
        budget_ms,
        speed
    )
    .unwrap();
    let fmt_ci = |c: (f64, f64, f64)| {
        if c.1.is_nan() {
            format!("{:.3}", c.0)
        } else {
            format!("{:.3} [{:.3}, {:.3}]", c.0, c.1, c.2)
        }
    };
    let mut cuts: Vec<String> = cut
        .iter()
        .map(|((sc, ror), n)| {
            format!(
                "{} {} from {n} vehicles",
                sc.name(),
                if *ror { "RoR" } else { "Rubble" }
            )
        })
        .collect();
    cuts.sort();
    if !cuts.is_empty() || stopped.is_some() {
        writeln!(md, "Bounds (ceiling {} ms for {} consecutive frames, max_frame_ms {max_frame_ms} predicted, deadline_s {deadline_s} of running time): engines stopped or not run from: {}. {}\n", ceiling.ms, ceiling.frames,
            if cuts.is_empty() { "none".into() } else { cuts.join("; ") },
            stopped.as_deref().unwrap_or("All trials ran.")).unwrap();
    }
    for &sc in &scenarios {
        writeln!(md, "## {}\n", sc.name()).unwrap();
        writeln!(md, "ms per frame: mean of trial means [95 % CI]; p95 over every frame. Speed-up: Rigs of Rods' time over Rubble's GPU time, geometric mean of paired trials [95 % CI].\n").unwrap();
        writeln!(md, "| vehicles | RoR (thread pool) | RoR p95 | Rubble GPU | Rubble GPU p95 | Rubble pipelined wall | Rubble blocking wall | speed-up | verdict | µs per vehicle (Rubble GPU) |").unwrap();
        writeln!(md, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
        for &n in counts.iter().chain(&rubble_counts) {
            let key = (sc, n);
            let gm = &gpu_means[&key];
            let ror_cell = ror_means.get(&key).map(|v| fmt_ci(ci(v)));
            let ror_p95 = ror_frames
                .get(&key)
                .map(|v| format!("{:.3}", percentile(v, 0.95)));
            let (speed_cell, verdict) = match ror_means.get(&key) {
                Some(rm) => {
                    let logs: Vec<f64> = rm.iter().zip(gm).map(|(a, b)| (a / b).ln()).collect();
                    let c = ci(&logs);
                    let (m, lo, hi) = (c.0.exp(), c.1.exp(), c.2.exp());
                    let verdict = if lo > 1.0 {
                        "Rubble faster"
                    } else if hi < 1.0 {
                        "RoR faster"
                    } else {
                        "inconclusive"
                    };
                    (
                        if lo.is_nan() {
                            format!("{m:.2}×")
                        } else {
                            format!("{m:.2}× [{lo:.2}, {hi:.2}]")
                        },
                        verdict,
                    )
                }
                None => ("—".into(), "—"),
            };
            writeln!(
                md,
                "| {n} | {} | {} | {} | {:.3} | {} | {} | {speed_cell} | {verdict} | {:.2} |",
                ror_cell.unwrap_or("—".into()),
                ror_p95.unwrap_or("—".into()),
                fmt_ci(ci(gm)),
                percentile(&gpu_frames[&key], 0.95),
                fmt_ci(ci(&piped[&key])),
                fmt_ci(ci(&blocking[&key])),
                mean(gm) * 1e3 / n as f64,
            )
            .unwrap();
        }
        // The worst frames: what a fixed frame budget has to hold.
        writeln!(md, "\nPeak frames (every frame of every trial): p99 and maximum ms, and the share of frames over the {budget_ms:.2} ms budget. Worst-frame speed-up: Rigs of Rods' maximum over Rubble's GPU maximum.\n").unwrap();
        writeln!(md, "| vehicles | RoR p99 | RoR max | RoR over budget | Rubble GPU p99 | Rubble GPU max | Rubble GPU over budget | Rubble blocking p99 | Rubble blocking max | worst-frame speed-up |").unwrap();
        writeln!(md, "|---|---|---|---|---|---|---|---|---|---|").unwrap();
        let over = |v: &[f64]| {
            100.0 * v.iter().filter(|&&x| x > budget_ms).count() as f64 / v.len().max(1) as f64
        };
        let peak = |v: &[f64]| v.iter().copied().fold(f64::NAN, f64::max);
        for &n in counts.iter().chain(&rubble_counts) {
            let key = (sc, n);
            let gf = &gpu_frames[&key];
            let bf = &blocking_frames[&key];
            let (rp99, rmax, rover, worst) = match ror_frames.get(&key) {
                Some(rf) => (
                    format!("{:.3}", percentile(rf, 0.99)),
                    format!("{:.3}", peak(rf)),
                    format!("{:.1} %", over(rf)),
                    format!("{:.2}×", peak(rf) / peak(gf)),
                ),
                None => ("—".into(), "—".into(), "—".into(), "—".into()),
            };
            writeln!(
                md,
                "| {n} | {rp99} | {rmax} | {rover} | {:.3} | {:.3} | {:.1} % | {:.3} | {:.3} | {worst} |",
                percentile(gf, 0.99),
                peak(gf),
                over(gf),
                percentile(bf, 0.99),
                peak(bf),
            )
            .unwrap();
        }
        writeln!(md).unwrap();
        // Real-time capacity: the largest count (every smaller one passing too) whose
        // frames hold the budget at p99 and never pass the ceiling (30 fps unless
        // ceiling_ms= says otherwise); a count the ceiling stopped fails.
        let hard = if ceiling.ms > 0.0 {
            ceiling.ms
        } else {
            1000.0 / 30.0
        };
        let holds = |v: Option<&Vec<f64>>| {
            v.is_some_and(|v| !v.is_empty() && percentile(v, 0.99) <= budget_ms && peak(v) <= hard)
        };
        let capacity = |frames: &HashMap<(Scenario, usize), Vec<f64>>, of: &[usize]| {
            of.iter()
                .take_while(|&&n| holds(frames.get(&(sc, n))))
                .last()
                .copied()
        };
        let all_n: Vec<usize> = counts.iter().chain(&rubble_counts).copied().collect();
        let show = |c: Option<usize>| c.map_or("none".to_string(), |n| n.to_string());
        writeln!(
            md,
            "Real-time capacity (largest count with p99 ≤ {budget_ms:.2} ms and every frame ≤ {hard:.1} ms, all smaller counts too): Rubble GPU time {}, Rubble blocking wall {}{}.\n",
            show(capacity(&gpu_frames, &all_n)),
            show(capacity(&blocking_frames, &all_n)),
            if ror_bin.is_some() {
                format!(", Rigs of Rods {} (of counts up to {})", show(capacity(&ror_frames, &counts)), counts.last().copied().unwrap_or(0))
            } else {
                String::new()
            },
        )
        .unwrap();
        // Scaling law: least-squares fixed + per-vehicle cost.
        let fit = |pts: &[(f64, f64)]| {
            let n = pts.len() as f64;
            let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
            let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
            let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
            let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            ((sy - b * sx) / n, b)
        };
        let all_counts: Vec<usize> = counts.iter().chain(&rubble_counts).copied().collect();
        let rub: Vec<(f64, f64)> = all_counts
            .iter()
            .map(|&n| (n as f64, mean(&gpu_means[&(sc, n)])))
            .collect();
        let (a, b) = fit(&rub);
        write!(
            md,
            "\nScaling (least squares, time = fixed + per vehicle × N): Rubble GPU {a:.3} ms + {:.2} µs × N",
            b * 1e3
        )
        .unwrap();
        let mut ror_fit = None;
        // RoR's fit over the counts it ran (the frame ceiling may end it early).
        let ror: Vec<(f64, f64)> = counts
            .iter()
            .filter_map(|&n| ror_means.get(&(sc, n)).map(|v| (n as f64, mean(v))))
            .collect();
        if ror.len() >= 2 {
            let (ra, rb) = fit(&ror);
            ror_fit = Some((ra, rb));
            write!(md, "; Rigs of Rods {ra:.3} ms + {:.2} µs × N", rb * 1e3).unwrap();
        }
        writeln!(md, ".\n").unwrap();
        for budget in BUDGETS_MS {
            let fits = |f: (f64, f64)| ((budget - f.0) / f.1).floor().max(0.0);
            write!(
                md,
                "Vehicles in a {budget} ms physics budget (from the fit): Rubble {}",
                fits((a, b))
            )
            .unwrap();
            if let Some(f) = ror_fit {
                write!(md, ", Rigs of Rods {}", fits(f)).unwrap();
            }
            writeln!(md, ".  ").unwrap();
        }
        writeln!(md).unwrap();
        if ror_bin.is_some() {
            writeln!(md, "Accuracy, end of the run (trial 0), sampled vehicles: node distance Rubble–RoR; broken / bent beams, travel, crush, each Rubble / RoR.\n").unwrap();
            writeln!(md, "| vehicles | vehicle | node RMS (mm) | node max (mm) | broken | bent | travel (m) | crush (cm) |").unwrap();
            writeln!(md, "|---|---|---|---|---|---|---|---|").unwrap();
            for &n in &counts {
                for c in accuracy.get(&(sc, n)).into_iter().flatten() {
                    let f = |k: &str, side: &str| c[side][k].as_f64().unwrap_or(f64::NAN);
                    writeln!(
                        md,
                        "| {n} | {} | {:.2} | {:.2} | {} / {} | {} / {} | {:.3} / {:.3} | {:.2} / {:.2} |",
                        c["index"],
                        c["rms_mm"].as_f64().unwrap(),
                        c["max_mm"].as_f64().unwrap(),
                        c["rubble"]["broken"],
                        c["ror"]["broken"],
                        c["rubble"]["bent"],
                        c["ror"]["bent"],
                        f("travel_m", "rubble"),
                        f("travel_m", "ror"),
                        f("crush_cm", "rubble"),
                        f("crush_cm", "ror"),
                    )
                    .unwrap();
                }
            }
            if let Some(a) = judged.get(&sc).filter(|_| nudges > 0) {
                writeln!(
                    md,
                    "\nAccuracy ensembles ({} runs per engine, spawn speed + 0.1 mm/s × k): median node RMS Rubble–RoR {:.2} mm, RoR–RoR {:.2} mm, ratio {:.2} [95 % bootstrap {:.2}, {:.2}]; broken: Rubble median {}, RoR {}–{}; bent: Rubble median {}, RoR {}–{}.\n",
                    a["runs"], a["median_rubble_ror_mm"].as_f64().unwrap(), a["median_ror_ror_mm"].as_f64().unwrap(),
                    a["ratio"].as_f64().unwrap(), a["ratio_ci"][0].as_f64().unwrap(), a["ratio_ci"][1].as_f64().unwrap(),
                    a["broken"]["rubble_median"], a["broken"]["ror_min"], a["broken"]["ror_max"],
                    a["bent"]["rubble_median"], a["bent"]["ror_min"], a["bent"]["ror_max"],
                )
                .unwrap();
            }
        }
    }
    std::fs::write(
        out.join("results.json"),
        serde_json::to_string(&json!({
            "truck": truck.display().to_string(), "adapter": info.name,
            "trials": trials, "warm": warm, "frames": frames, "speed": speed,
            "stopped": stopped, "max_frame_ms": max_frame_ms, "ceiling_ms": ceiling.ms, "ceiling_frames": ceiling.frames,
            "cut": cut.iter().map(|((sc, ror), n)| json!({"scenario": sc.name(), "engine": if *ror { "ror" } else { "rubble" }, "from": n})).collect::<Vec<_>>(),
            "runs": all,
            "nudges": nudges,
            "ensembles": scenarios.iter().map(|s| (s.name(), ensembles.get(s).map(|e| e.json(&g)).unwrap_or(Value::Null))).collect::<HashMap<_, _>>(),
            "accuracy": scenarios.iter().map(|s| (s.name(), judged.get(s).cloned().unwrap_or(Value::Null))).collect::<HashMap<_, _>>(),
        }))?,
    )?;
    // Acceptance: every criterion with its measured values, PASS or FAIL.
    if ror_bin.is_some() {
        let mut fails = 0usize;
        let mut line = |md: &mut String, ok: bool, what: String| {
            fails += (!ok) as usize;
            writeln!(md, "| {} | {what} |", if ok { "PASS" } else { "FAIL" }).unwrap();
        };
        writeln!(
            md,
            "## Acceptance\n\nCommand: `{}`\n",
            std::env::args().collect::<Vec<_>>().join(" ")
        )
        .unwrap();
        writeln!(md, "| result | criterion |\n|---|---|").unwrap();
        line(
            &mut md,
            busy_runs == 0,
            format!("{run_count} runs started on a quiet machine (load < {max_load}, GPU < {max_gpu} %): {busy_runs} did not"),
        );
        for &sc in &scenarios {
            // Accuracy lines only with ensembles (`nudges=0`: speed alone).
            if nudges > 0 {
                let a = &judged[&sc];
                let e = &ensembles[&sc];
                let (rr, xr) = (
                    a["median_ror_ror_mm"].as_f64().unwrap(),
                    a["median_rubble_ror_mm"].as_f64().unwrap(),
                );
                let limit = (1.5 * rr).max(1.0);
                line(&mut md, xr <= limit, format!(
                    "{} N={} ({} runs per engine): median node RMS Rubble–RoR {xr:.2} mm <= {limit:.2} mm (1.5x RoR–RoR median {rr:.2} mm, at least 1 mm); ratio {:.2} [95 % {:.2}, {:.2}]",
                    sc.name(), if sc.grouped() { 2 } else { 1 }, a["runs"], a["ratio"].as_f64().unwrap(), a["ratio_ci"][0].as_f64().unwrap(), a["ratio_ci"][1].as_f64().unwrap()));
                for k in ["broken", "bent"] {
                    let (lo, hi) = (
                        a[k]["ror_min"].as_f64().unwrap(),
                        a[k]["ror_max"].as_f64().unwrap(),
                    );
                    let (lo, hi) = ((0.9 * lo).floor(), (1.1 * hi).ceil());
                    let v = a[k]["rubble_median"].as_f64().unwrap();
                    line(&mut md, v >= lo && v <= hi, format!(
                        "{} N={}: Rubble median {k} {v} within RoR's {}–{} widened 10% ({lo:.0}–{hi:.0})",
                        sc.name(), if sc.grouped() { 2 } else { 1 }, a[k]["ror_min"], a[k]["ror_max"]));
                }
                for &n in counts.iter().filter(|&&n| n > 1) {
                    for (end, start) in fleet_ends.get(&(sc, n)).into_iter().flatten() {
                        let d = e.median_to_ror(end, start);
                        line(&mut md, d <= limit, format!(
                            "{} N={n} vehicle {}: median node RMS to the RoR runs {d:.2} mm <= {limit:.2} mm",
                            sc.name(), end.index));
                    }
                }
            }
            for &n in &counts {
                let key = (sc, n);
                let (Some(rm), Some(rf)) = (ror_means.get(&key), ror_frames.get(&key)) else {
                    continue;
                };
                let gm = &gpu_means[&key];
                let logs: Vec<f64> = rm.iter().zip(gm).map(|(a, b)| (a / b).ln()).collect();
                let c = ci(&logs);
                line(&mut md, c.1.exp() > 1.0, format!(
                    "{} N={n}: speed-up {:.2}x, 95% CI [{:.2}, {:.2}] excludes 1 (RoR {:.3} ms, Rubble GPU {:.3} ms)",
                    sc.name(), c.0.exp(), c.1.exp(), c.2.exp(), mean(rm), mean(gm)));
                let pw = mean(&piped[&key]);
                line(
                    &mut md,
                    pw < mean(rm),
                    format!(
                        "{} N={n}: Rubble pipelined wall {pw:.3} ms < RoR mean {:.3} ms",
                        sc.name(),
                        mean(rm)
                    ),
                );
                let gf = &gpu_frames[&key];
                let (rp, gp) = (percentile(rf, 0.99), percentile(gf, 0.99));
                let (rx, gx) = (
                    rf.iter().copied().fold(f64::MIN, f64::max),
                    gf.iter().copied().fold(f64::MIN, f64::max),
                );
                line(
                    &mut md,
                    gp < rp && gx < rx,
                    format!(
                        "{} N={n}: Rubble p99 {gp:.3} < RoR {rp:.3} ms, max {gx:.3} < {rx:.3} ms",
                        sc.name()
                    ),
                );
            }
            if sc == Scenario::Impact {
                if let Some(gf) = gpu_frames.get(&(sc, 64)) {
                    let p = percentile(gf, 0.99);
                    line(
                        &mut md,
                        p <= 8.33,
                        format!("impact N=64: Rubble p99 GPU frame {p:.3} ms <= 8.33 ms"),
                    );
                }
            }
        }
        writeln!(
            md,
            "\n**{}** ({fails} failing)",
            if fails == 0 { "ALL PASS" } else { "NOT MET" }
        )
        .unwrap();
    }
    std::fs::write(out.join("summary.md"), &md)?;
    println!("{md}");
    Ok(())
}
