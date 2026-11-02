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
//! Accuracy. In every run the first and last vehicle's end state (node positions,
//! broken and bent beams, travel along the forward axis, and the crush: loss of
//! length along it) is compared with Rigs of Rods' same vehicle; and per scenario,
//! Rigs of Rods against itself with the spawn speed 0.1 mm/s higher, the
//! sensitivity a match cannot be expected to beat.
//!
//! `trace=<file>` (with `trace_scene=<json>`, `trace_every=`, `cols=` for the grid's
//! columns) records the first scenario and count on Rubble's GPU for the crash
//! viewer instead of benchmarking.
//!
//! Writes `<out>/results.json` (every frame time and comparison) and
//! `<out>/summary.md`.

use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox};
use engine_softbody::truck::parse_truck;
use engine_webgpu::{default_device, render_scene_json, ExplicitBeamsGpu};
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
}

impl Scenario {
    fn parse(s: &str) -> Self {
        match s {
            "idle" => Self::Idle,
            "drive" => Self::Drive,
            "impact" => Self::Impact,
            _ => panic!("scenario {s}: idle, drive or impact"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Drive => "drive",
            Self::Impact => "impact",
        }
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
        Self {
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
    let drag = DRAG.load(std::sync::atomic::Ordering::Relaxed);
    let settings = engine_softbody::explicit_beams::ExplicitBeamsSettings {
        drag,
        turbulence: drag,
        ..Default::default()
    };
    let mut sim = ExplicitBeams::from_truck(text, Vec3::new(at.x, 0.0, at.z), vel, settings)
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
    let v = if sc == Scenario::Idle { 0.0 } else { speed };
    let cols = cols.unwrap_or((n as f64).sqrt().ceil() as usize).max(1);
    let width = g.across.1 - g.across.0;
    let length = g.along.1 - g.along.0;
    let travel = speed * seconds + 20.0;
    let lateral_step = width + 3.0;
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
    let walls = origins
        .iter()
        .map(|o| {
            (sc == Scenario::Impact).then(|| {
                let face = o.dot(g.forward) + g.along.1 + 1.0 + 0.5 * speed;
                let half_t = 0.5;
                let side = 0.5 * width + 1.0;
                let mid = o.dot(g.lateral) + 0.5 * (g.across.0 + g.across.1);
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
            .map(|o| (o, g.forward * (v + extra)))
            .collect(),
        walls,
    }
}

fn write_scene(path: &Path, s: &Scene) {
    let mut t = String::new();
    for w in s.walls.iter().flatten() {
        let (c, h) = w;
        writeln!(t, "box {} {} {} {} {} {}", c.x, c.y, c.z, h.x, h.y, h.z).unwrap();
    }
    for (o, v) in &s.vehicles {
        writeln!(t, "vehicle {} {} {} {} {}", o.x, o.z, v.x, v.y, v.z).unwrap();
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
}

struct RubbleRun {
    gpu_ms: Vec<f64>,
    pipelined_ms: f64,
    blocking_ms: Vec<f64>,
    end: Vec<EndState>,
}

fn samples(n: usize) -> Vec<usize> {
    if n > 1 {
        vec![0, n - 1]
    } else {
        vec![0]
    }
}

fn rubble_fleet(text: &str, s: &Scene) -> Vec<ExplicitBeams> {
    s.vehicles
        .iter()
        .zip(&s.walls)
        .map(|((o, v), w)| {
            let mut sim = spawn(text, *o, *v);
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
) -> RubbleRun {
    let n = fleet.len();
    // Pipelined, with the GPU time of each frame.
    let mut gpu = ExplicitBeamsGpu::new(device, queue, fleet);
    apply_wide(&mut gpu);
    for _ in 0..warm {
        gpu.frame(steps);
    }
    gpu.wait();
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
    let end = rubble_end(&gpu, n);
    drop(gpu);
    // Blocking: submit to completion, every frame.
    let mut gpu = ExplicitBeamsGpu::new(device, queue, fleet);
    apply_wide(&mut gpu);
    for _ in 0..warm {
        gpu.frame(steps);
    }
    gpu.wait();
    let blocking_ms = (0..frames)
        .map(|_| {
            let t = Instant::now();
            gpu.frame(steps);
            gpu.wait();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    RubbleRun {
        gpu_ms,
        pipelined_ms,
        blocking_ms,
        end,
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
        .arg(format!("out={}", out.display()))
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .expect("ror_ref");
    assert!(status.success(), "ror_ref scale failed");
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
    Run { frame_ms, end }
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

/// The GPU's busy share in percent (macOS: the accelerator's "Device
/// Utilization %"), the highest of three samples a quarter second apart; NaN where
/// it cannot be read.
fn gpu_busy() -> f64 {
    let sample = || {
        std::process::Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|t| {
                let k = "\"Device Utilization %\"=";
                let i = t.find(k)? + k.len();
                t[i..]
                    .split(|c: char| !c.is_ascii_digit())
                    .next()?
                    .parse::<f64>()
                    .ok()
            })
            .unwrap_or(f64::NAN)
    };
    (0..3)
        .map(|i| {
            if i > 0 {
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            sample()
        })
        .fold(f64::NAN, f64::max)
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
    let out = PathBuf::from(kv(&args, "out").unwrap_or("explicit_beams_scaling"));
    std::fs::create_dir_all(&out)?;
    let ror_bin = kv(&args, "ror")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("ROR_BIN").map(PathBuf::from));
    let seconds = (warm + frames) as f32 / fps;
    let cols: Option<usize> = kv(&args, "cols").map(|s| s.parse().unwrap());
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

    // Frame timestamps need the timestamp-query feature on the device.
    std::env::set_var("RUBBLE_GPU_TIMING", "1");
    let (info, device, queue) = default_device()?;
    let g = Geometry::of(&text);
    let one = spawn(&text, Vec3::ZERO, Vec3::ZERO);
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
    // Runs that started on a busy machine (the wait gave up): none may count.
    let mut busy_runs = 0usize;
    let mut run_count = 0usize;

    for trial in 0..trials {
        for &sc in &scenarios {
            for &n in counts.iter().chain(&rubble_counts) {
                let with_ror = ror_bin.is_some() && counts.contains(&n);
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
                        loads.push(wait_quiet(max_load, max_gpu, quiet_wait));
                        ror_run = Some(run_ror(
                            ror_bin.as_ref().unwrap(),
                            &truck,
                            &file,
                            n,
                            warm,
                            frames,
                            steps,
                        ));
                    } else if !first {
                        loads.push(wait_quiet(max_load, max_gpu, quiet_wait));
                        rubble_run = Some(run_rubble(&device, &queue, &fleet, warm, frames, steps));
                    }
                }
                run_count += loads.len();
                busy_runs += loads
                    .iter()
                    .filter(|(l, g)| !(*l < max_load) || *g >= max_gpu)
                    .count();
                let r = rubble_run.unwrap();
                let key = (sc, n);
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
            }
        }
    }

    // Rigs of Rods against itself, 0.1 mm/s faster: the chaos floor.
    let mut sensitivity: HashMap<Scenario, Value> = HashMap::new();
    if let Some(bin) = &ror_bin {
        for &sc in &scenarios {
            let a = scene(sc, 1, &g, speed, seconds, 0.0, None);
            let b = scene(sc, 1, &g, speed, seconds, 1e-4, None);
            let fa = out.join(format!("sens_{}_a.txt", sc.name()));
            let fb = out.join(format!("sens_{}_b.txt", sc.name()));
            write_scene(&fa, &a);
            write_scene(&fb, &b);
            let ra = run_ror(bin, &truck, &fa, 1, warm, frames, steps);
            let rb = run_ror(bin, &truck, &fb, 1, warm, frames, steps);
            let (rms, max) = rms_max(&ra.end[0].pos, &rb.end[0].pos);
            sensitivity.insert(
                sc,
                json!({"rms_mm": rms * 1e3, "max_mm": max * 1e3,
                       "broken": [ra.end[0].broken, rb.end[0].broken],
                       "bent": [ra.end[0].bent, rb.end[0].bent]}),
            );
        }
    }

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
        if ror_bin.is_some() {
            let ror: Vec<(f64, f64)> = counts
                .iter()
                .map(|&n| (n as f64, mean(&ror_means[&(sc, n)])))
                .collect();
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
            if let Some(s) = sensitivity.get(&sc) {
                writeln!(
                    md,
                    "\nRigs of Rods against itself, spawn speed 0.1 mm/s higher: node RMS {:.2} mm, max {:.2} mm; broken {} / {}, bent {} / {}.\n",
                    s["rms_mm"].as_f64().unwrap(),
                    s["max_mm"].as_f64().unwrap(),
                    s["broken"][0],
                    s["broken"][1],
                    s["bent"][0],
                    s["bent"][1]
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
            "runs": all,
            "sensitivity": scenarios.iter().map(|s| (s.name(), sensitivity.get(s).cloned().unwrap_or(Value::Null))).collect::<HashMap<_, _>>(),
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
            let sens = &sensitivity[&sc];
            let s_rms = sens["rms_mm"].as_f64().unwrap();
            let range = |k: &str| {
                let a = sens[k][0].as_f64().unwrap();
                let b = sens[k][1].as_f64().unwrap();
                (0.9 * a.min(b), 1.1 * a.max(b))
            };
            for &n in &counts {
                for c in accuracy.get(&(sc, n)).into_iter().flatten() {
                    let rms = c["rms_mm"].as_f64().unwrap();
                    let limit = (1.5 * s_rms).max(1.0);
                    line(&mut md, rms <= limit, format!(
                        "{} N={n} vehicle {}: node RMS {rms:.2} mm <= {limit:.2} mm (1.5x RoR's own {s_rms:.2} mm, at least 1 mm)",
                        sc.name(), c["index"]));
                    for k in ["broken", "bent"] {
                        let (lo, hi) = range(k);
                        let v = c["rubble"][k].as_f64().unwrap();
                        line(&mut md, v >= lo.floor() && v <= hi.ceil(), format!(
                            "{} N={n} vehicle {}: {k} {v} within RoR's {}–{} widened 10% ({:.0}–{:.0})",
                            sc.name(), c["index"], sens[k][0], sens[k][1], lo.floor(), hi.ceil()));
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
