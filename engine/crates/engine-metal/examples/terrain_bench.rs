//! The browser demo's event natively (`impact.html`: an 8 m stony impactor at
//! 20 km/s into dry sand, margin 1.5 R, 12 cells per radius, the default
//! absorbing ground) at 1/120 s frames, on native Metal (`MetalSim`) and on
//! native wgpu-on-Metal (`WgpuSim`), the same `TerrainPlan` on both.
//!
//! Per frame: GPU time (Metal: the command buffer's GPUStart/EndTime; wgpu:
//! timestamp queries around the terrain pass), substeps, blocks. Repeated
//! `--runs` times per backend (interleaved, the GPU is shared) and reported as
//! the median per frame. `--profile` adds Metal's per-kernel breakdown (one
//! encoder per dispatch, timestamped at its boundaries).
//!
//! `cargo run --release -p engine-metal --example terrain_bench -- --secs 10 --runs 3 --profile`
//! Options: `--backend metal,metal-fast,wgpu` (any subset), `--cells 12`, `--margin 1.5`,
//! `--frame-hz 120`, `--json out.json`.

use engine_metal::{MetalSim, TerrainCaps};
use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterTarget, ImpactSource, RegionPlan, SandMaterial, TerrainConfig,
    TerrainStepReport,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};
use std::time::Instant;

const G: f32 = 9.81;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

#[derive(Clone, Debug)]
struct Opts {
    secs: f32,
    runs: usize,
    cells: f32,
    margin: f64,
    hz: f32,
    /// Backends, in the order each run takes them: `metal` (the plan's
    /// portable kernels), `metal-fast` (every `TerrainCaps` fast path),
    /// `metal+flag+...` (those fast paths), `wgpu`.
    backends: Vec<String>,
    profile: bool,
    json: Option<String>,
}

fn opts() -> Opts {
    let mut o = Opts {
        secs: 10.0,
        runs: 3,
        cells: 12.0,
        margin: 1.5,
        hz: 120.0,
        backends: vec!["metal".into(), "metal-fast".into(), "wgpu".into()],
        profile: false,
        json: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--secs" => o.secs = v.parse().unwrap(),
            "--runs" => o.runs = v.parse().unwrap(),
            "--cells" => o.cells = v.parse().unwrap(),
            "--margin" => o.margin = v.parse().unwrap(),
            "--frame-hz" => o.hz = v.parse().unwrap(),
            "--json" => o.json = Some(v),
            "--backend" => o.backends = v.split(',').map(str::to_string).collect(),
            "--profile" => {
                o.profile = true;
                i += 1;
                continue;
            }
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    o
}

/// The demo's event: config, region, field, source.
struct Event {
    cfg: TerrainConfig,
    sand: SandMaterial,
    plan: RegionPlan,
    field: HeightField,
    model: engine_terrain::ZModel,
}

fn event(o: &Opts) -> Event {
    let cfg = TerrainConfig {
        max_particles: 4_000_000,
        max_blocks: 16384,
        max_parcels: 1 << 19,
        max_columns: 1 << 17,
        max_field_samples: 1 << 19,
        max_substeps: 1024,
        cells_per_radius: o.cells,
        ..TerrainConfig::default()
    };
    let sand = SandMaterial::default();
    let src = ImpactSource {
        impactor_radius: 8.0,
        impactor_density: 3000.0,
        velocity: DVec3::new(0.0, -20000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let g = f64::from(G);
    let crater = src.crater(g).unwrap();
    let model = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
    let plan = crater.region(&cfg, Vec3::ZERO, o.margin, 0, 0).unwrap();
    let n = 2 * (3.0 * crater.radius as f32 / plan.dx).ceil() as u32 + 1;
    let field = HeightField::new(n, n, plan.dx, plan.dx, vec![0.0; (n * n) as usize]).unwrap();
    Event {
        cfg,
        sand,
        plan,
        field,
        model,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Frame {
    substeps: u32,
    gpu_ms: f64,
    wall_ms: f64,
    blocks: u32,
    max_speed: f32,
    particles: u32,
}

/// What both hosts do for the benchmark.
trait Bench {
    /// Terrain enabled (kernels built), material and field loaded.
    fn prepare(&mut self, e: &Event);
    /// Activation and source; their GPU time (ms; NaN without one).
    fn open(&mut self, e: &Event) -> f64;
    fn frame(&mut self, k: u32, dt: f32) -> (TerrainStepReport, f64);
    /// The region's state, for run-to-run determinism (`state_print`).
    fn fingerprint(&mut self) -> String;
}

struct MetalBench {
    sim: MetalSim,
    bed: u32,
    /// Print the activation's and the source's per-kernel times.
    profile_open: bool,
}

impl Bench for MetalBench {
    fn prepare(&mut self, e: &Event) {
        let s = &mut self.sim;
        s.terrain_enable(&e.cfg).unwrap();
        s.terrain_set_material(Some(&e.sand)).unwrap();
        let mut arena = HullArena::new();
        self.bed = s
            .add_height_field(&mut arena, &e.field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap()
            .1;
    }
    fn open(&mut self, e: &Event) -> f64 {
        let profile = self.profile_open;
        let s = &mut self.sim;
        if profile {
            s.terrain_profile_next();
        }
        s.terrain_activate(
            &e.plan,
            self.bed,
            &e.field,
            Vec3::ZERO,
            e.sand.density,
            e.sand.friction_angle.tan(),
        )
        .unwrap();
        let a = s.terrain_last_gpu_ms();
        let activation = s.terrain_profiled();
        if profile {
            s.terrain_profile_next();
        }
        s.terrain_apply_source(&e.model, Vec3::ZERO).unwrap();
        let source = s.terrain_profiled();
        for (label, t) in [("activation", activation), ("source", source)] {
            if let Some(t) = t {
                println!("  {label}: {:.3} ms GPU (profiled)", t.gpu_ms);
                for (name, ms, c) in t.kernels {
                    println!("    {name:<20} {ms:>8.4} ms ({c} dispatches)");
                }
            }
        }
        a + s.terrain_last_gpu_ms()
    }
    fn frame(&mut self, k: u32, dt: f32) -> (TerrainStepReport, f64) {
        self.sim.terrain_step_timed(k, dt, -G).unwrap()
    }
    fn fingerprint(&mut self) -> String {
        state_print(&self.sim.terrain_particles(), &self.sim.terrain_parcels())
    }
}

struct WgpuBench {
    sim: WgpuSim,
}

impl Bench for WgpuBench {
    fn prepare(&mut self, e: &Event) {
        let s = &mut self.sim;
        s.terrain_enable(&e.cfg).unwrap();
        s.terrain_set_material(Some(&e.sand)).unwrap();
        let mut arena = HullArena::new();
        s.add_height_field(&mut arena, &e.field, Vec3::ZERO, Quat::IDENTITY, 0)
            .unwrap();
        // The terrain pipelines compile on first use: one throwaway frame.
        let _ = s.read_words(engine_terrain::layout::MPM_COUNTERS_BINDING, 0, 1);
    }
    fn open(&mut self, e: &Event) -> f64 {
        let s = &mut self.sim;
        s.terrain_activate(
            &e.plan,
            0,
            &e.field,
            Vec3::ZERO,
            e.sand.density,
            e.sand.friction_angle.tan(),
        )
        .unwrap();
        s.terrain_apply_source(&e.model, Vec3::ZERO).unwrap();
        let _ = s.read_words(engine_terrain::layout::MPM_COUNTERS_BINDING, 0, 1);
        f64::NAN
    }
    /// wgpu-on-Metal has no reliable GPU time here: timestamps of passes
    /// around the terrain pass came back out of order (its encoders are not
    /// ordered by Metal's hazard tracking), so the frame is timed by wall
    /// clock, submission to counters read, as Metal's wall time is.
    fn frame(&mut self, k: u32, dt: f32) -> (TerrainStepReport, f64) {
        (self.sim.terrain_step(k, dt, -G).unwrap(), f64::NAN)
    }
    fn fingerprint(&mut self) -> String {
        state_print(&self.sim.terrain_particles(), &self.sim.terrain_parcels())
    }
}

/// One run of the timeline: the strike frame's activation and source, then
/// `secs` of frames.
/// FNV-1a of the particles (in slot order) and of the parcels as a set:
/// parcels take their slots by atomic claims in G2P, whose order the GPU
/// does not fix, so only their sorted list is comparable run to run.
fn state_print(
    p: &[engine_terrain::layout::MpmParticle],
    q: &[engine_terrain::layout::MpmParcel],
) -> String {
    let mut parcels: Vec<Vec<u8>> = q.iter().map(|x| bytemuck::bytes_of(x).to_vec()).collect();
    parcels.sort();
    format!(
        "particles {:016x}, {} parcels {:016x} (as a set)",
        fnv(bytemuck::cast_slice(p)),
        q.len(),
        fnv(&parcels.concat())
    )
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn run(b: &mut dyn Bench, e: &Event, o: &Opts) -> ([f64; 2], Vec<Frame>) {
    b.prepare(e);
    let t = Instant::now();
    let open_gpu = b.open(e);
    let open_ms = [open_gpu, t.elapsed().as_secs_f64() * 1e3];
    let frame = 1.0 / o.hz;
    let mut speed = engine_terrain::plan::source_peak_speed(&e.model);
    let n = (o.secs * o.hz).round() as usize;
    let mut frames = Vec::with_capacity(n);
    for _ in 0..n {
        let (k, dt) = substeps_for(
            &e.cfg,
            frame,
            e.plan.dx,
            e.sand.p_wave_speed(),
            1.5 * speed + 20.0,
        )
        .unwrap();
        let t = Instant::now();
        let (r, gpu) = b.frame(k, dt);
        frames.push(Frame {
            substeps: k,
            gpu_ms: gpu,
            wall_ms: t.elapsed().as_secs_f64() * 1e3,
            blocks: r.blocks,
            max_speed: r.max_speed,
            particles: r.particles,
        });
        speed = r.max_speed;
    }
    (open_ms, frames)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return 0.0;
    }
    v[v.len() / 2]
}

/// Frame-wise medians across runs.
fn medians(runs: &[([f64; 2], Vec<Frame>)]) -> ([f64; 2], Vec<Frame>) {
    let n = runs.iter().map(|r| r.1.len()).min().unwrap_or(0);
    let open = [0, 1].map(|k| median(&mut runs.iter().map(|r| r.0[k]).collect::<Vec<_>>()));
    let frames = (0..n)
        .map(|i| {
            let mut f = runs[0].1[i];
            f.gpu_ms = median(&mut runs.iter().map(|r| r.1[i].gpu_ms).collect::<Vec<_>>());
            f.wall_ms = median(&mut runs.iter().map(|r| r.1[i].wall_ms).collect::<Vec<_>>());
            f
        })
        .collect();
    (open, frames)
}

struct Summary {
    strike_ms: f64,
    worst_ms: f64,
    worst_frame: usize,
    first_mean: f64,
    first_max: f64,
    steady_mean: f64,
    steady_max: f64,
    steady_substeps: f64,
    first_substeps: (u32, u32),
}

/// Worst frame, the first second and the last second (steady state) of a
/// timeline, by the clock `t` picks.
fn summarize(open: f64, f: &[Frame], hz: f32, t: fn(&Frame) -> f64) -> Summary {
    let sec = hz.round() as usize;
    let first = &f[..sec.min(f.len())];
    let steady = &f[f.len().saturating_sub(sec)..];
    let (wi, w) = f
        .iter()
        .enumerate()
        .max_by(|a, b| t(a.1).total_cmp(&t(b.1)))
        .map(|(i, x)| (i, t(x)))
        .unwrap_or((0, 0.0));
    let mean = |s: &[Frame]| s.iter().map(t).sum::<f64>() / s.len().max(1) as f64;
    let max = |s: &[Frame]| s.iter().map(t).fold(0.0, f64::max);
    Summary {
        strike_ms: open + f.first().map_or(0.0, t),
        worst_ms: w,
        worst_frame: wi,
        first_mean: mean(first),
        first_max: max(first),
        steady_mean: mean(steady),
        steady_max: max(steady),
        steady_substeps: steady.iter().map(|x| f64::from(x.substeps)).sum::<f64>()
            / steady.len().max(1) as f64,
        first_substeps: (
            first.iter().map(|x| x.substeps).min().unwrap_or(0),
            first.iter().map(|x| x.substeps).max().unwrap_or(0),
        ),
    }
}

fn print_summary(label: &str, s: &Summary, f: &[Frame]) {
    println!(
        "{label}: strike frame (activate + source + frame 0) {:.2} ms; worst frame {:.2} ms (frame {}, {} substeps); \
         first second mean {:.2} ms, max {:.2} ms ({}-{} substeps); steady (last second) mean {:.2} ms, max {:.2} ms ({:.1} substeps)",
        s.strike_ms,
        s.worst_ms,
        s.worst_frame,
        f[s.worst_frame].substeps,
        s.first_mean,
        s.first_max,
        s.first_substeps.0,
        s.first_substeps.1,
        s.steady_mean,
        s.steady_max,
        s.steady_substeps
    );
}

fn summary_json(s: &Summary) -> serde_json::Value {
    serde_json::json!({
        "strike_ms": s.strike_ms,
        "worst_ms": s.worst_ms,
        "worst_frame": s.worst_frame,
        "first_second_mean_ms": s.first_mean,
        "first_second_max_ms": s.first_max,
        "steady_mean_ms": s.steady_mean,
        "steady_max_ms": s.steady_max,
        "steady_substeps": s.steady_substeps,
    })
}

/// `metal`: the plan's portable kernels; `metal-fast`: every fast path;
/// `metal+a+b`: those fast paths (`TerrainCaps::parse`).
fn caps_of(name: &str) -> TerrainCaps {
    match name {
        "metal" => TerrainCaps::default(),
        "metal-fast" => TerrainCaps::all(),
        n => TerrainCaps::parse(n.trim_start_matches("metal+")).unwrap(),
    }
}

fn main() {
    let o = opts();
    let e = event(&o);
    let cells = e.plan.blocks * 4;
    println!(
        "chip {}; region {:?} cells of {:.2} m ({} R margin, {} cells/R), field {}², frames 1/{} s, {} s, {} runs",
        MetalSim::new(capacity()).unwrap().chip(),
        cells,
        e.plan.dx,
        o.margin,
        o.cells,
        e.field.samples().0,
        o.hz,
        o.secs,
        o.runs
    );
    let make = |name: &str| -> Box<dyn Bench> {
        match name {
            n if n.starts_with("metal") => {
                let mut sim = MetalSim::new(capacity()).unwrap();
                sim.set_terrain_caps(caps_of(n));
                Box::new(MetalBench {
                    sim,
                    bed: 0,
                    profile_open: false,
                })
            }
            "wgpu" => Box::new(WgpuBench {
                sim: WgpuSim::new(capacity()).expect("wgpu"),
            }),
            other => panic!("unknown backend {other}"),
        }
    };
    let mut all_runs: Vec<(String, Vec<([f64; 2], Vec<Frame>)>)> =
        o.backends.iter().map(|b| (b.clone(), Vec::new())).collect();
    for r in 0..o.runs {
        for (name, runs) in all_runs.iter_mut() {
            let mut b = make(name);
            let x = run(b.as_mut(), &e, &o);
            let last = x.1.last().copied().unwrap_or_default();
            eprintln!(
                "run {r} {name}: activation and source {:.2} ms GPU, {:.2} ms wall, {} particles; \
                 at the end {} particle slots, {}",
                x.0[0],
                x.0[1],
                x.1.first().map_or(0, |f| f.particles),
                last.particles,
                b.fingerprint()
            );
            runs.push(x);
        }
    }
    let mut json = serde_json::Map::new();
    for (name, runs) in &all_runs {
        let name = name.as_str();
        if runs.is_empty() {
            continue;
        }
        let (open, f) = medians(runs);
        let gpu = name
            .starts_with("metal")
            .then(|| summarize(open[0], &f, o.hz, |x| x.gpu_ms));
        let wall = summarize(open[1], &f, o.hz, |x| x.wall_ms);
        if let Some(g) = &gpu {
            print_summary(&format!("{name} GPU"), g, &f);
        }
        print_summary(&format!("{name} wall"), &wall, &f);
        // A coarse timeline: per 0.5 s, the mean and max frame time and substeps.
        let step = (o.hz / 2.0).round() as usize;
        for (i, w) in f.chunks(step).enumerate() {
            let mean = |t: fn(&Frame) -> f64| w.iter().map(t).sum::<f64>() / w.len() as f64;
            let max = |t: fn(&Frame) -> f64| w.iter().map(t).fold(0.0, f64::max);
            println!(
                "  {name} {:>5.1}-{:<5.1} s: GPU mean {:>7.3} max {:>7.3} ms; wall mean {:>7.3} max {:>7.3} ms; substeps {}-{}, blocks {}, top speed {:.1} m/s",
                i as f32 * 0.5,
                (i + 1) as f32 * 0.5,
                mean(|x| x.gpu_ms),
                max(|x| x.gpu_ms),
                mean(|x| x.wall_ms),
                max(|x| x.wall_ms),
                w.iter().map(|x| x.substeps).min().unwrap(),
                w.iter().map(|x| x.substeps).max().unwrap(),
                w.last().unwrap().blocks,
                w.first().unwrap().max_speed,
            );
        }
        json.insert(
            name.into(),
            serde_json::json!({
                "gpu": gpu.as_ref().map(summary_json),
                "wall": summary_json(&wall),
                "open_ms": open,
                "frames_gpu_ms": f.iter().map(|x| x.gpu_ms).collect::<Vec<_>>(),
                "frames_wall_ms": f.iter().map(|x| x.wall_ms).collect::<Vec<_>>(),
                "frames_substeps": f.iter().map(|x| x.substeps).collect::<Vec<_>>(),
            }),
        );
    }
    let profiled: Vec<&String> = o
        .backends
        .iter()
        .filter(|b| b.starts_with("metal"))
        .collect();
    for variant in profiled.into_iter().filter(|_| o.profile) {
        // Per-kernel breakdown over three windows of a fresh Metal run.
        let mut sim = MetalSim::new(capacity()).unwrap();
        sim.set_terrain_caps(caps_of(variant));
        let mut b = MetalBench {
            sim,
            bed: 0,
            profile_open: true,
        };
        println!("{variant} strike:");
        b.prepare(&e);
        b.open(&e);
        let frame = 1.0 / o.hz;
        let mut speed = engine_terrain::plan::source_peak_speed(&e.model);
        let n = (o.secs * o.hz).round() as usize;
        let sec = o.hz.round() as usize;
        let windows = [
            ("frame 0", 0..1),
            ("first second", 0..sec.min(n)),
            ("last second", n.saturating_sub(sec)..n),
        ];
        let mut acc: Vec<Vec<(&'static str, f64, u32)>> = vec![Vec::new(); windows.len()];
        let mut totals = vec![0.0f64; windows.len()];
        let mut counts = vec![0usize; windows.len()];
        for i in 0..n {
            let (k, dt) = substeps_for(
                &e.cfg,
                frame,
                e.plan.dx,
                e.sand.p_wave_speed(),
                1.5 * speed + 20.0,
            )
            .unwrap();
            let profile = windows.iter().any(|(_, w)| w.contains(&i));
            let r = if profile {
                let (r, t) = b.sim.terrain_step_profiled(k, dt, -G).unwrap();
                for (wi, (_, w)) in windows.iter().enumerate() {
                    if !w.contains(&i) {
                        continue;
                    }
                    totals[wi] += t.gpu_ms;
                    counts[wi] += 1;
                    for &(name, ms, c) in &t.kernels {
                        match acc[wi].iter_mut().find(|x| x.0 == name) {
                            Some(x) => {
                                x.1 += ms;
                                x.2 += c;
                            }
                            None => acc[wi].push((name, ms, c)),
                        }
                    }
                }
                r
            } else {
                b.sim.terrain_step(k, dt, -G).unwrap()
            };
            speed = r.max_speed;
        }
        let mut prof = serde_json::Map::new();
        for (wi, (label, _)) in windows.iter().enumerate() {
            let frames = counts[wi].max(1) as f64;
            let mut ks = acc[wi].clone();
            ks.sort_by(|a, b| b.1.total_cmp(&a.1));
            let sum: f64 = ks.iter().map(|x| x.1).sum();
            println!(
                "{variant} kernels, {label} ({} frames; profiled frame {:.3} ms, kernel sum {:.3} ms per frame):",
                counts[wi],
                totals[wi] / frames,
                sum / frames
            );
            let mut rows = Vec::new();
            for (name, ms, c) in ks {
                println!(
                    "  {name:<20} {:>8.4} ms/frame {:>5.1}%  {:>6.1} dispatches/frame, {:>7.1} µs each",
                    ms / frames,
                    100.0 * ms / sum.max(1e-12),
                    f64::from(c) / frames,
                    1e3 * ms / f64::from(c.max(1))
                );
                rows.push(serde_json::json!([
                    name,
                    ms / frames,
                    f64::from(c) / frames
                ]));
            }
            prof.insert((*label).into(), serde_json::Value::Array(rows));
        }
        json.insert(
            format!("{variant}_kernels"),
            serde_json::Value::Object(prof),
        );
    }
    if let Some(path) = &o.json {
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}
