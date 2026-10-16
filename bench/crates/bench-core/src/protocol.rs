//! The timing protocol every adapter runs, and the files it writes.
//!
//! One run = one engine configuration on one scene:
//! 1. Build the scene (timed separately as `buildMs`), read the state back as checkpoint 0.
//! 2. Lead-in: `lead` untimed steps.
//! 3. Timed window, three-avbd's `compare` shape: `rounds` × (`warm_steps` untimed steps,
//!    drain, then `batches` × `per_batch` timed steps). A batch is timed from the first step's
//!    submission to the completion of the last (`drain`), divided by `per_batch`.
//!    - `batch` mode queues the whole batch before draining (a GPU engine may overlap CPU
//!      encoding with GPU work, as a game loop would);
//!    - `sync` mode drains after every step and also records each step's latency.
//!    CPU engines finish each step inside `step()`, so both modes measure the same thing.
//!    `wallMs` is the smaller of the round medians (three-avbd's statistic).
//! 4. Untimed steps on to `steps`, so the judge can check the motion at every checkpoint.
//! Checkpoints (pose dumps) never fall inside the timed window; reading state is not timed.

use crate::scene::Scene;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// One body's state, canonical frame, in scene order.
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BodySample {
    pub position: [f32; 3],
    /// (x, y, z, w)
    pub rotation: [f32; 4],
    pub linvel: [f32; 3],
    pub angvel: [f32; 3],
}

pub const SAMPLE_FLOATS: usize = 13;
const POSE_MAGIC: u32 = 0x5350_4252; // "RBPS"

#[derive(Clone, Debug, Default)]
pub struct EngineInfo {
    /// Engine family, e.g. "rapier", "physx", "rubble", "three-avbd".
    pub engine: String,
    /// Library version or source revision.
    pub version: String,
    /// Where it runs: "cpu", "wgpu/metal", "metal", "cuda", "dawn/metal", ...
    pub backend: String,
    /// GPU adapter or CPU model.
    pub device: String,
    /// CPU worker threads the engine may use (1 = single-threaded; GPU engines report 1).
    pub threads: usize,
    /// Whether `step()` can return before the work completes (GPU queue).
    pub pipelined: bool,
    /// Solver family, e.g. "AVBD", "TGS", "PGS soft-TGS (substeps)".
    pub solver: String,
    /// The engine's own iteration knob and what it means.
    pub iterations: u32,
    pub iterations_meaning: String,
    pub sleeping: bool,
    /// Every other setting that differs from the engine's defaults, and notable defaults.
    pub config: BTreeMap<String, String>,
}

pub trait Engine {
    fn info(&self) -> EngineInfo;
    /// Advance one step of the scene's dt. A GPU engine may return before the GPU finishes.
    fn step(&mut self) -> Result<()>;
    /// Wait until every submitted step has completed.
    fn drain(&mut self) -> Result<()> {
        Ok(())
    }
    /// Current state of every body, in scene order. Called after `drain`, never timed.
    fn read_state(&mut self, out: &mut [BodySample]) -> Result<()>;
    /// Engine counters worth reporting (contacts, colours, overflow, awake...). Never timed.
    fn counters(&mut self) -> Result<Value> {
        Ok(Value::Null)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Batch,
    Sync,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Batch => "batch",
            Mode::Sync => "sync",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub mode: Mode,
    pub poses: Option<PathBuf>,
    /// Stop the untimed tail after this many seconds (later gates then read "not reached").
    pub max_seconds: Option<f64>,
    pub label: Option<String>,
    pub quiet: bool,
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[(((s.len() - 1) as f64) * p).round() as usize]
}

struct PoseWriter {
    file: Option<std::io::BufWriter<std::fs::File>>,
}

impl PoseWriter {
    fn write(&mut self, step: u32, samples: &[BodySample]) -> Result<()> {
        if let Some(f) = &mut self.file {
            f.write_all(&POSE_MAGIC.to_le_bytes())?;
            f.write_all(&step.to_le_bytes())?;
            f.write_all(&(samples.len() as u32).to_le_bytes())?;
            f.write_all(&(SAMPLE_FLOATS as u32).to_le_bytes())?;
            f.write_all(bytemuck::cast_slice(samples))?;
        }
        Ok(())
    }
}

/// Run the protocol and return the result document (also what `write_result` saves).
pub fn run(engine: &mut dyn Engine, scene: &Scene, build_ms: f64, opts: &RunOptions) -> Result<Value> {
    let p = &scene.meta.protocol;
    let info = engine.info();
    let log = |s: String| {
        if !opts.quiet {
            eprintln!("[{} {}] {s}", info.engine, scene.meta.id);
        }
    };
    let checkpoints: Vec<u32> = p.checkpoints.clone();
    let window_end = p.lead + p.timed_span();
    ensure!(
        checkpoints.iter().all(|&c| c <= p.lead || c >= window_end),
        "scene checkpoints fall inside the timed window"
    );
    let mut poses = PoseWriter {
        file: match &opts.poses {
            Some(path) => Some(std::io::BufWriter::new(
                std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
            )),
            None => None,
        },
    };
    let mut samples = vec![BodySample::default(); scene.bodies.len()];
    let mut checkpoint_docs = Vec::new();
    let mut capture = |engine: &mut dyn Engine, step: u32| -> Result<()> {
        engine.drain()?;
        engine.read_state(&mut samples)?;
        poses.write(step, &samples)?;
        checkpoint_docs.push(json!({ "step": step, "counters": engine.counters()? }));
        Ok(())
    };
    // Untimed stepping, traced per chunk (each step in sync mode, ten in batch mode)
    let mut trace: Vec<(u32, f64)> = Vec::new();
    let chunk = if opts.mode == Mode::Sync { 1 } else { p.per_batch.max(1) };
    let advance = |engine: &mut dyn Engine, from: u32, to: u32, trace: &mut Vec<(u32, f64)>| -> Result<()> {
        let mut done = from;
        while done < to {
            let n = chunk.min(to - done);
            let t = Instant::now();
            for _ in 0..n {
                engine.step()?;
            }
            engine.drain()?;
            done += n;
            trace.push((done, t.elapsed().as_secs_f64() * 1e3 / n as f64));
        }
        Ok(())
    };

    let started = Instant::now();
    let mut done = 0u32;
    capture(engine, 0)?;
    for &c in checkpoints.iter().filter(|&&c| c > 0 && c <= p.lead) {
        advance(engine, done, c, &mut trace)?;
        done = c;
        capture(engine, c)?;
    }
    advance(engine, done, p.lead, &mut trace)?;
    done = p.lead;
    log(format!("lead-in {} steps done in {:.1} s; timing", p.lead, started.elapsed().as_secs_f64()));

    let mut rounds = Vec::new();
    let mut step_ms = Vec::new();
    for _ in 0..p.rounds {
        for _ in 0..p.warm_steps {
            engine.step()?;
        }
        engine.drain()?;
        done += p.warm_steps;
        let mut batch_ms = Vec::new();
        for _ in 0..p.batches {
            match opts.mode {
                Mode::Batch => {
                    let t = Instant::now();
                    for _ in 0..p.per_batch {
                        engine.step()?;
                    }
                    engine.drain()?;
                    batch_ms.push(t.elapsed().as_secs_f64() * 1e3 / p.per_batch as f64);
                }
                Mode::Sync => {
                    let mut sum = 0.0;
                    for _ in 0..p.per_batch {
                        let t = Instant::now();
                        engine.step()?;
                        engine.drain()?;
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        step_ms.push(ms);
                        sum += ms;
                    }
                    batch_ms.push(sum / p.per_batch as f64);
                }
            }
            done += p.per_batch;
        }
        rounds.push(json!({ "samplesMs": batch_ms, "medianMs": median(&batch_ms) }));
    }
    let wall_ms = rounds.iter().map(|r| r["medianMs"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
    let all_batches: Vec<f64> = rounds
        .iter()
        .flat_map(|r| r["samplesMs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect();
    log(format!("wall {wall_ms:.3} ms/step ({})", opts.mode.as_str()));

    let mut truncated = false;
    let tail_start = Instant::now();
    for &c in checkpoints.iter().filter(|&&c| c >= window_end) {
        if let Some(limit) = opts.max_seconds {
            if tail_start.elapsed().as_secs_f64() > limit {
                truncated = true;
                log(format!("time budget {limit} s spent at step {done}; later checkpoints not reached"));
                break;
            }
        }
        advance(engine, done, c, &mut trace)?;
        done = c;
        capture(engine, c)?;
    }
    if let Some(f) = &mut poses.file {
        f.flush()?;
    }
    let step_stats = if step_ms.is_empty() {
        Value::Null
    } else {
        json!({
            "meanMs": step_ms.iter().sum::<f64>() / step_ms.len() as f64,
            "p50Ms": percentile(&step_ms, 0.5),
            "p95Ms": percentile(&step_ms, 0.95),
            "maxMs": step_ms.iter().cloned().fold(0.0, f64::max),
            "samplesMs": step_ms,
        })
    };
    let untimed: Vec<f64> = trace.iter().map(|t| t.1).collect();
    Ok(json!({
        "format": "rubble-bench-result",
        "version": 1,
        "label": opts.label,
        "date": crate::machine::utc_now(),
        "machine": crate::machine::describe(),
        "scene": {
            "id": scene.meta.id,
            "tier": scene.meta.tier,
            "sha256": scene.meta.bin_sha256,
            "bodies": scene.bodies.len(),
            "dynamic": scene.dynamic_count(),
            "joints": scene.joints.len(),
            "dt": scene.meta.dt,
            "referenceIterations": scene.meta.iterations,
        },
        "engine": {
            "engine": info.engine,
            "version": info.version,
            "backend": info.backend,
            "device": info.device,
            "threads": info.threads,
            "pipelined": info.pipelined,
            "solver": info.solver,
            "iterations": info.iterations,
            "iterationsMeaning": info.iterations_meaning,
            "sleeping": info.sleeping,
            "config": info.config,
        },
        "protocol": {
            "mode": opts.mode.as_str(),
            "lead": p.lead,
            "rounds": p.rounds,
            "warmSteps": p.warm_steps,
            "batches": p.batches,
            "perBatch": p.per_batch,
            "steps": p.steps,
            "stepsRun": done,
            "truncated": truncated,
        },
        "buildMs": build_ms,
        "timing": {
            "wallMs": wall_ms,
            "batchMeanMs": all_batches.iter().sum::<f64>() / all_batches.len() as f64,
            "batchMaxMs": all_batches.iter().cloned().fold(0.0, f64::max),
            "rounds": rounds,
            "step": step_stats,
            "untimedMeanMs": if untimed.is_empty() { Value::Null } else { json!(untimed.iter().sum::<f64>() / untimed.len() as f64) },
            "trace": trace.iter().map(|(s, ms)| json!([s, ms])).collect::<Vec<_>>(),
        },
        "checkpoints": checkpoint_docs,
        "poses": opts.poses.as_ref().map(|p| p.file_name().unwrap().to_string_lossy().to_string()),
        "totalSeconds": started.elapsed().as_secs_f64(),
    }))
}

pub fn write_result(path: &std::path::Path, doc: &Value) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(doc)? + "\n").with_context(|| format!("writing {}", path.display()))
}

/// Read every checkpoint of a pose file: (step, samples).
pub fn read_poses(path: &std::path::Path) -> Result<Vec<(u32, Vec<BodySample>)>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    let mut at = 0usize;
    let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    while at + 16 <= bytes.len() {
        ensure!(word(at) == POSE_MAGIC, "{}: bad record at byte {at}", path.display());
        let (step, n, floats) = (word(at + 4), word(at + 8) as usize, word(at + 12) as usize);
        ensure!(floats == SAMPLE_FLOATS, "{}: {floats} floats per body", path.display());
        let len = n * SAMPLE_FLOATS * 4;
        ensure!(at + 16 + len <= bytes.len(), "{}: truncated record", path.display());
        let mut samples = vec![BodySample::default(); n];
        bytemuck::cast_slice_mut::<BodySample, u8>(&mut samples).copy_from_slice(&bytes[at + 16..at + 16 + len]);
        out.push((step, samples));
        at += 16 + len;
    }
    Ok(out)
}
