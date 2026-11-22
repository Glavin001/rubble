//! Per-kernel isolated GPU timing by replay (measurement only, never in the sim loop).
//!
//! [`WgpuSim::kernel_replay`] snapshots every buffer of the rigid step, builds the next
//! step's plan exactly as [`WgpuSim::step_encode`] would, and then, from that captured
//! state:
//!
//! 1. **whole step**: the plan's dispatches in one timestamped pass (the shipped shape),
//!    `reps` times, each from the snapshot;
//! 2. **pass per phase**: the same, a timestamped pass per phase (`GpuTiming::Phases`'
//!    shape), to measure what the split itself costs;
//! 3. **pass per dispatch, in sequence**: every dispatch in its own timestamped pass, in
//!    plan order (each kernel sees the caches the previous one left);
//! 4. **isolated replay**: for each dispatch `k`, the buffers it binds are restored to the
//!    state just before it and it runs alone in a timestamped pass, `reps` times. The
//!    state then moves past `k` (its last rep ran from the right input), so dispatch
//!    `k + 1` replays from exactly the input the real step gives it.
//! 5. **empty pass**: a timestamped pass with a one-workgroup no-op, the floor every
//!    single-pass timing above includes.
//!
//! Afterwards every buffer is restored to the state before the replay and no host
//! bookkeeping moved: the next real step runs as if the replay never happened. Nothing
//! here runs inside a step; a host calls it between frames, on demand.
//!
//! Every wait is a GPU-completion wait (native `device.poll`, the browser's
//! `engine_gpu_wait::until` under JSPI), and at most one replay chunk is in flight, so a
//! replay never queues more work than the GPU has finished.

use super::*;

/// What [`WgpuSim::kernel_replay`] measures.
#[derive(Clone, Copy, Debug)]
pub struct KernelReplayOptions {
    /// Timed repetitions per measurement (after one untimed warm-up).
    pub reps: u32,
    /// Skip the isolated per-dispatch replay (only the whole-step shapes).
    pub whole_step_only: bool,
}

impl Default for KernelReplayOptions {
    fn default() -> Self {
        Self { reps: 15, whole_step_only: false }
    }
}

/// One dispatch of the plan, replayed.
#[derive(Clone, Debug)]
pub struct ReplayedDispatch {
    /// Its index among the plan's dispatches.
    pub index: u32,
    pub phase: &'static str,
    pub kernel: &'static str,
    /// Workgroups (x, y, z); an indirect dispatch's are read from its arguments as they
    /// were just before it ran.
    pub workgroups: [u32; 3],
    pub indirect: bool,
    /// Isolated: ms of each rep (its own pass, from its own input).
    pub isolated_ms: Vec<f64>,
    /// In sequence: ms of each rep of the pass-per-dispatch run.
    pub in_sequence_ms: Vec<f64>,
}

/// [`WgpuSim::kernel_replay`]'s result. All times in ms.
#[derive(Clone, Debug, Default)]
pub struct KernelReplayReport {
    /// Steps the sim had run when the snapshot was taken.
    pub step: u32,
    pub reps: u32,
    pub timestamp_period_ns: f32,
    /// Bytes snapshotted (twice: the state to restore, and the replay's moving input).
    pub snapshot_bytes: u64,
    /// The plan in one pass, per rep.
    pub whole_step_ms: Vec<f64>,
    /// Pass per phase: per rep, the sum of the phase passes, and the span first begin ->
    /// last end (includes gaps between passes).
    pub phases_sum_ms: Vec<f64>,
    pub phases_span_ms: Vec<f64>,
    /// Each phase's median over the reps, in plan order.
    pub phases: Vec<(String, f64)>,
    /// Pass per phase: the GPU's idle time before each phase's pass (its begin after the
    /// previous pass's end; the first 0), medians, in plan order.
    pub phase_gaps_ms: Vec<f64>,
    /// Pass per dispatch in sequence: per rep, sum and span.
    pub dispatch_split_sum_ms: Vec<f64>,
    pub dispatch_split_span_ms: Vec<f64>,
    /// An empty pass (one-workgroup no-op), per rep.
    pub empty_pass_ms: Vec<f64>,
    /// Timed passes that wrote no begin or end timestamp (left out of every time).
    pub unwritten_passes: u32,
    pub dispatches: Vec<ReplayedDispatch>,
}

impl KernelReplayReport {
    /// Machine-readable form (`profile-webgpu` JSON): every sample, plus medians and p90s.
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        let stat = |v: &[f64]| json!({ "median": median(v), "p90": quantile(v, 0.9), "min": quantile(v, 0.0), "samples": v });
        let whole = median(&self.whole_step_ms);
        let isolated_sum: f64 = self.dispatches.iter().filter(|d| !d.empty()).map(|d| median(&d.isolated_ms)).sum();
        // Per kernel (phase/kernel): dispatches, workgroups, in-sequence and isolated ms.
        let mut by: Vec<((&str, &str), (u32, u64, f64, f64))> = Vec::new();
        for d in &self.dispatches {
            let k = (d.phase, d.kernel);
            let i = match by.iter().position(|(x, _)| *x == k) {
                Some(i) => i,
                None => {
                    by.push((k, (0, 0, 0.0, 0.0)));
                    by.len() - 1
                }
            };
            let a = &mut by[i].1;
            a.0 += 1;
            a.1 += u64::from(d.workgroups.iter().product::<u32>());
            a.2 += median(&d.in_sequence_ms);
            if !d.empty() {
                a.3 += median(&d.isolated_ms);
            }
        }
        by.sort_by(|a, b| b.1 .2.total_cmp(&a.1 .2));
        let in_sequence_sum: f64 = self.dispatches.iter().map(|d| median(&d.in_sequence_ms)).sum();
        json!({
            "step": self.step,
            "reps": self.reps,
            "timestamp_period_ns": self.timestamp_period_ns,
            "snapshot_bytes": self.snapshot_bytes,
            "whole_step_ms": stat(&self.whole_step_ms),
            "phases_sum_ms": stat(&self.phases_sum_ms),
            "phases_span_ms": stat(&self.phases_span_ms),
            "dispatch_split_sum_ms": stat(&self.dispatch_split_sum_ms),
            "dispatch_split_span_ms": stat(&self.dispatch_split_span_ms),
            "empty_pass_ms": stat(&self.empty_pass_ms),
            "unwritten_passes": self.unwritten_passes,
            "isolated_sum_ms": isolated_sum,
            "in_sequence_sum_ms": in_sequence_sum,
            "kernels": by.iter().map(|((p, k), (n, wg, seq, iso))| json!({
                "phase": p, "kernel": k, "dispatches": n, "workgroups": wg, "in_sequence_ms": seq, "isolated_ms": iso,
                "share_of_step": if whole > 0.0 { seq / whole } else { 0.0 },
            })).collect::<Vec<_>>(),
            "phases": self.phases.iter().zip(self.phase_gaps_ms.iter().chain(std::iter::repeat(&0.0))).map(|((p, ms), gap)| json!({ "phase": p, "ms": ms, "idle_before_ms": gap })).collect::<Vec<_>>(),
            "dispatches": self.dispatches.iter().map(|d| {
                let iso = median(&d.isolated_ms);
                json!({
                    "index": d.index,
                    "phase": d.phase,
                    "kernel": d.kernel,
                    "workgroups": d.workgroups,
                    "indirect": d.indirect,
                    "empty": d.empty(),
                    "isolated_ms": stat(&d.isolated_ms),
                    "in_sequence_ms": stat(&d.in_sequence_ms),
                    "share_of_step": if whole > 0.0 { iso / whole } else { 0.0 },
                })
            }).collect::<Vec<_>>(),
        })
    }

    /// A short text table: the step's shapes, then dispatches by isolated median, largest
    /// first (`top` rows; 0: all).
    pub fn table(&self, top: usize) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let whole = median(&self.whole_step_ms);
        let iso: f64 = self.dispatches.iter().filter(|d| !d.empty()).map(|d| median(&d.isolated_ms)).sum();
        let empties = self.dispatches.iter().filter(|d| d.empty()).count();
        let seq: f64 = self.dispatches.iter().map(|d| median(&d.in_sequence_ms)).sum();
        let empty = median(&self.empty_pass_ms);
        let _ = writeln!(out, "kernel replay at step {} ({} reps, {} dispatches, snapshot {:.1} MB)", self.step, self.reps, self.dispatches.len(), self.snapshot_bytes as f64 / 1e6);
        let _ = writeln!(out, "  timed passes that wrote no timestamp: {}", self.unwritten_passes);
        let _ = writeln!(out, "  whole step (one pass)        {whole:8.3} ms  p90 {:.3}", quantile(&self.whole_step_ms, 0.9));
        let _ = writeln!(out, "  pass per phase: sum / span   {:8.3} / {:.3} ms", median(&self.phases_sum_ms), median(&self.phases_span_ms));
        let gaps: f64 = self.phase_gaps_ms.iter().sum();
        let _ = writeln!(out, "    idle between phase passes {gaps:8.3} ms: {}", self.phases.iter().zip(&self.phase_gaps_ms).skip(1).map(|((p, _), g)| format!("{p} {g:.3}")).collect::<Vec<_>>().join(", "));
        let _ = writeln!(out, "  pass per dispatch: sum / span{:8.3} / {:.3} ms", median(&self.dispatch_split_sum_ms), median(&self.dispatch_split_span_ms));
        let _ = writeln!(out, "  isolated sum                 {iso:8.3} ms   ({empties} empty dispatches left out; empty pass {empty:.4} ms)");
        let _ = writeln!(out, "  in-sequence sum              {seq:8.3} ms");
        // Per kernel, by in-sequence time (completion deltas: they sum to the split span).
        let mut by: Vec<((&str, &str), (u32, u64, f64, f64))> = Vec::new();
        for d in &self.dispatches {
            let k = (d.phase, d.kernel);
            let i = by.iter().position(|(x, _)| *x == k).unwrap_or_else(|| {
                by.push((k, (0, 0, 0.0, 0.0)));
                by.len() - 1
            });
            let a = &mut by[i].1;
            a.0 += 1;
            a.1 += u64::from(d.workgroups.iter().product::<u32>());
            a.2 += median(&d.in_sequence_ms);
            if !d.empty() {
                a.3 += median(&d.isolated_ms);
            }
        }
        by.sort_by(|a, b| b.1 .2.total_cmp(&a.1 .2));
        let _ = writeln!(out, "  per kernel: {:<12} {:<34} {:>4} {:>10} {:>9} {:>9} {:>6}", "phase", "kernel", "n", "workgroups", "in-seq", "iso", "share");
        for ((p, k), (n, wg, seq, iso)) in by.iter().take(if top == 0 { usize::MAX } else { top }) {
            let _ = writeln!(out, "              {p:<12} {k:<34} {n:>4} {wg:>10} {seq:>9.4} {iso:>9.4} {:>5.1}%", if whole > 0.0 { 100.0 * seq / whole } else { 0.0 });
        }
        let mut rows: Vec<&ReplayedDispatch> = self.dispatches.iter().collect();
        rows.sort_by(|a, b| median(&b.in_sequence_ms).total_cmp(&median(&a.in_sequence_ms)));
        let _ = writeln!(out, "  {:>4} {:<14} {:<34} {:>16} {:>9} {:>9} {:>9} {:>6}", "#", "phase", "kernel", "workgroups", "iso ms", "p90", "in-seq", "share");
        for d in rows.iter().take(if top == 0 { usize::MAX } else { top }) {
            let m = median(&d.isolated_ms);
            let wg = format!("{}x{}x{}{}", d.workgroups[0], d.workgroups[1], d.workgroups[2], if d.indirect { "*" } else { "" });
            let seq = median(&d.in_sequence_ms);
            let iso = if d.empty() { "empty".to_string() } else { format!("{m:.4}") };
            let _ = writeln!(out, "  {:>4} {:<14} {:<34} {:>16} {:>9} {:>9.4} {:>9.4} {:>5.1}%", d.index, d.phase, d.kernel, wg, iso, quantile(&d.isolated_ms, 0.9), seq, if whole > 0.0 { 100.0 * seq / whole } else { 0.0 });
        }
        out
    }
}

/// For native probes and examples: when `RUBBLE_KERNEL_REPLAY_AT=N` is set, the first call
/// at or after step `N` runs [`WgpuSim::kernel_replay`] (`RUBBLE_KERNEL_REPLAY_REPS`,
/// default 15), prints its table and writes its JSON to `RUBBLE_KERNEL_REPLAY_OUT` (if
/// set). Call it between steps; it runs once per process. Returns whether it ran.
pub fn env_hook(sim: &mut WgpuSim) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    let Some(at) = std::env::var("RUBBLE_KERNEL_REPLAY_AT").ok().and_then(|v| v.parse::<u32>().ok()) else {
        return false;
    };
    if sim.steps_done < at || DONE.load(Ordering::Relaxed) {
        return false;
    }
    let reps = std::env::var("RUBBLE_KERNEL_REPLAY_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    match sim.kernel_replay(KernelReplayOptions { reps, whole_step_only: false }) {
        Ok(r) => {
            DONE.store(true, Ordering::Relaxed);
            println!("{}", r.table(25));
            if let Ok(path) = std::env::var("RUBBLE_KERNEL_REPLAY_OUT") {
                let mut j = r.to_json();
                j["table"] = serde_json::Value::String(r.table(0));
                // Host spans so far (`RUBBLE_HOST_PROFILE=1`): per name, the median ms an
                // occurrence and the count.
                let mut by: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
                for sp in crate::host_profile::drain() {
                    by.entry(sp.name).or_default().push(sp.dur_ms);
                }
                let host: serde_json::Map<String, serde_json::Value> = by
                    .iter()
                    .map(|(k, v)| (k.to_string(), serde_json::json!({ "median": median(v), "p90": quantile(v, 0.9), "n": v.len() })))
                    .collect();
                let doc = serde_json::json!({ "replay": j, "native": { "steps": sim.steps_done }, "host": host });
                if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap_or_default()) {
                    eprintln!("kernel replay: cannot write {path}: {e}");
                }
            }
            true
        }
        Err(e) => {
            // (Pending host commands, or every body asleep: try again at the next call;
            // said once.)
            static SAID: AtomicBool = AtomicBool::new(false);
            if !SAID.swap(true, Ordering::Relaxed) {
                eprintln!("kernel replay at step {}: {e:#} (retrying each step)", sim.steps_done);
            }
            false
        }
    }
}

impl ReplayedDispatch {
    /// No workgroups (an indirect dispatch whose count was zero): its isolated time is the
    /// cost of an empty pass on its own, not the kernel's (on wgpu-on-Metal that can be
    /// milliseconds where the same dispatch in sequence takes a microsecond).
    pub fn empty(&self) -> bool {
        self.workgroups.iter().product::<u32>() == 0
    }
}

/// A dispatch of the plan.
#[derive(Clone, Copy)]
struct Item {
    cmd: rigid_plan::Cmd,
    phase: &'static str,
    kernel: &'static str,
}

/// The median of `v` (0 when empty).
pub fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// The `q` quantile of `v` (nearest rank; 0 when empty).
pub fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let i = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[i.min(s.len() - 1)]
}

/// The indices of `set`'s buffers, and the buffers.
fn pairs(set: &[Option<wgpu::Buffer>]) -> (Vec<usize>, Vec<&wgpu::Buffer>) {
    let idx: Vec<usize> = (0..set.len()).filter(|&i| set[i].is_some()).collect();
    let bufs = idx.iter().map(|&i| set[i].as_ref().expect("copyable")).collect();
    (idx, bufs)
}

/// WebGPU's cap on a query set's size.
const MAX_QUERIES: u32 = 4096;

/// Timestamps for the replay's chunks. A slot is never written twice: a pass that writes
/// no timestamp (Metal, for some passes) would otherwise leave an older chunk's tick in
/// its slot and stretch every span computed over it. Unwritten slots read 0 and are
/// skipped; a full query set is replaced by a fresh one.
struct Timer {
    base: std::cell::Cell<u32>,
    /// Timed passes whose begin or end timestamp was never written (read 0).
    unwritten: std::rc::Rc<std::cell::Cell<u32>>,
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ms: f64,
}

impl Timer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let bytes = 8 * u64::from(MAX_QUERIES);
        Self {
            base: std::cell::Cell::new(0),
            unwritten: Default::default(),
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("kernel replay timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kernel replay resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kernel replay read"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ms: f64::from(queue.get_timestamp_period()) * 1e-6,
        }
    }

    /// Room for a chunk of `slots` pairs (a fresh query set when this one is full).
    fn room(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, slots: u32) {
        assert!(2 * slots <= MAX_QUERIES, "a chunk of {slots} timestamp pairs");
        if 2 * (self.base.get() + slots) > MAX_QUERIES {
            let unwritten = self.unwritten.clone();
            *self = Timer::new(device, queue);
            self.unwritten = unwritten;
        }
    }

    fn writes(&self, slot: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        let k = self.base.get() + slot;
        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(2 * k),
            end_of_pass_write_index: Some(2 * k + 1),
        }
    }

    /// Submit `enc`, resolve `slots` pairs in a submission of their own (Metal: a resolve in
    /// the same command buffer can read samples before they land), wait, read them as
    /// (begin, end) ticks.
    fn finish(&self, device: &wgpu::Device, queue: &wgpu::Queue, enc: wgpu::CommandEncoder, slots: u32) -> Vec<(u64, u64)> {
        queue.submit([enc.finish()]);
        wait_idle(device, queue);
        let mut r = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay resolve") });
        let n = 2 * slots;
        let first = 2 * self.base.get();
        r.resolve_query_set(&self.set, first..first + n, &self.resolve, 0);
        r.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, 8 * u64::from(n));
        queue.submit([r.finish()]);
        self.base.set(self.base.get() + slots);
        let t = read_mapped(device, &self.read, 8 * u64::from(n));
        let t: &[u64] = bytemuck::cast_slice(&t);
        let out: Vec<(u64, u64)> = (0..slots as usize).map(|i| (t[2 * i], t[2 * i + 1])).collect();
        let missing = out.iter().filter(|p| p.0 == 0 || p.1 == 0).count() as u32;
        self.unwritten.set(self.unwritten.get() + missing);
        out
    }

    /// ms from tick `a` to tick `b`; 0 when either was never written.
    fn ms(&self, (a, b): (u64, u64)) -> f64 {
        if a == 0 || b == 0 {
            return 0.0;
        }
        b.saturating_sub(a) as f64 * self.period_ms
    }
}

/// Wait until the GPU has finished everything submitted so far (native `poll`, the
/// browser's JSPI wait on `onSubmittedWorkDone`).
fn wait_idle(device: &wgpu::Device, queue: &wgpu::Queue) {
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = done.clone();
    queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    engine_gpu_wait::until(|| {
        let _ = device.poll(wgpu::PollType::Poll);
        done.load(std::sync::atomic::Ordering::Acquire)
    });
}

/// Map `buf` (MAP_READ) and copy its first `size` bytes out, waiting for the GPU.
fn read_mapped(device: &wgpu::Device, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = done.clone();
    let slice = buf.slice(0..size);
    slice.map_async(wgpu::MapMode::Read, move |_| flag.store(true, std::sync::atomic::Ordering::Release));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    engine_gpu_wait::until(|| {
        let _ = device.poll(wgpu::PollType::Poll);
        done.load(std::sync::atomic::Ordering::Acquire)
    });
    let out = slice.get_mapped_range().expect("mapped").to_vec();
    buf.unmap();
    out
}

impl WgpuSim {
    /// Per-kernel isolated GPU timing of the next step by replay (see the module docs).
    /// The sim is left exactly as it was. Errors when the next step would be skipped
    /// (every body asleep), when host commands or impulses are pending (they are consumed
    /// by a step; call it between steps), or without timestamp queries.
    pub fn kernel_replay(&mut self, opts: KernelReplayOptions) -> anyhow::Result<KernelReplayReport> {
        anyhow::ensure!(
            self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "kernel replay needs timestamp queries (Features::TIMESTAMP_QUERY)"
        );
        anyhow::ensure!(
            self.pending_commands.is_empty() && self.impulse_queue.is_empty() && !self.terrain_impulses_pending,
            "kernel replay runs between steps: host commands or impulses are pending"
        );
        anyhow::ensure!(opts.reps >= 1 && opts.reps <= 256, "reps must be 1..=256");
        self.build_all_bounds_if_dirty();
        let vehicles_dirty = self.vehicles.as_ref().is_some_and(|v| v.dirty);
        anyhow::ensure!(
            !(self.steps_done > 0
                && self.counted_steps == self.steps_done
                && self.last_counters.awake_count == 0
                && self.command_count == 0
                && self.body_count > 0
                && !self.wake_pending
                && !vehicles_dirty
                && !self.has_kinematic
                && self.projectile_count == 0),
            "the next step would be skipped (every body asleep): nothing to replay"
        );
        let device = self.device.clone();
        let queue = self.queue.clone();
        let mut timer = Timer::new(&device, &queue);
        let reps = opts.reps;

        // The state before anything of the replay, restored at the end (taken before the
        // step's host writes, which reach the GPU at the next submission).
        let n = self.bufs.len();
        let copyable = |b: &wgpu::Buffer| b.usage().contains(wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let mk = |b: &wgpu::Buffer, what: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(what),
                size: b.size(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let saved: Vec<Option<wgpu::Buffer>> = self.bufs.iter().map(|b| copyable(b).then(|| mk(b, "kernel replay saved"))).collect();
        let input: Vec<Option<wgpu::Buffer>> = self.bufs.iter().map(|b| copyable(b).then(|| mk(b, "kernel replay input"))).collect();
        let snapshot_bytes: u64 = saved.iter().flatten().map(|b| b.size()).sum();
        let copy_all = |enc: &mut wgpu::CommandEncoder, from: &[&wgpu::Buffer], to: &[&wgpu::Buffer]| {
            for (a, b) in from.iter().zip(to) {
                enc.copy_buffer_to_buffer(a, 0, b, 0, a.size());
            }
        };
        {
            let (idx, to) = pairs(&saved);
            let from: Vec<&wgpu::Buffer> = idx.iter().map(|&i| &self.bufs[i]).collect();
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay save") });
            copy_all(&mut enc, &from, &to);
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }

        // The next step's host writes and plan, as `step_encode` makes them.
        let step = self.steps_done;
        self.refresh_cell();
        self.ensure_rest_tables()?;
        self.write_host_uniforms_ex(true);
        self.write_guide_params();
        let modes = self.step_modes();
        let inputs = self.step_inputs();
        let cmds = rigid_plan::encode_step(&modes, &inputs, false, true);
        let phase_of = crate::gpu_labels::phase_of(&cmds);
        let skip = std::env::var("RUBBLE_SKIP_KERNELS").unwrap_or_default();
        let skipped = |k: &str| !skip.is_empty() && skip.split(',').any(|s| s == k);
        let items: Vec<Item> = cmds
            .iter()
            .enumerate()
            .filter_map(|(k, c)| match *c {
                rigid_plan::Cmd::Dispatch { kernel, .. } | rigid_plan::Cmd::Indirect { kernel, .. } if !skipped(kernel) => {
                    let allowed = if kernel.starts_with("solve_primal_color") { self.wg_allows("solve_primal_color") } else { self.wg_allows(kernel) };
                    allowed.then_some(Item { cmd: *c, phase: phase_of[k], kernel })
                }
                _ => None,
            })
            .collect();
        anyhow::ensure!(!items.is_empty(), "the plan has no dispatches");
        // The step's copies (before its pass), then the replay's input = the step's input.
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay input") });
            self.encode_plan_copies(&mut enc, &cmds);
            let (idx, to) = pairs(&input);
            let from: Vec<&wgpu::Buffer> = idx.iter().map(|&i| &self.bufs[i]).collect();
            copy_all(&mut enc, &from, &to);
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        let restore_all = |enc: &mut wgpu::CommandEncoder, sim: &WgpuSim, from_set: &[Option<wgpu::Buffer>]| {
            for i in 0..n {
                if let Some(src) = &from_set[i] {
                    enc.copy_buffer_to_buffer(src, 0, &sim.bufs[i], 0, src.size());
                }
            }
        };
        let mut report = KernelReplayReport {
            step,
            reps,
            timestamp_period_ns: queue.get_timestamp_period(),
            snapshot_bytes: 2 * snapshot_bytes,
            ..Default::default()
        };

        // 5. The empty pass.
        {
            let noop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("kernel replay noop"),
                layout: None,
                module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("kernel replay noop"),
                    source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
                }),
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            timer.room(&device, &queue, reps + 1);
            for r in 0..=reps {
                wait_idle(&device, &queue);
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay empty") });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("replay:empty"), timestamp_writes: Some(timer.writes(r)) });
                    pass.set_pipeline(&noop);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([enc.finish()]);
            }
            let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay empty") });
            let t = timer.finish(&device, &queue, enc, reps + 1);
            report.empty_pass_ms = t[1..].iter().map(|&x| timer.ms(x)).collect();
        }

        // 1-3. The whole step in its three shapes, each rep from the step's input.
        let n_items = items.len() as u32;
        anyhow::ensure!(n_items < MAX_QUERIES / 2, "{} dispatches exceed one query set", n_items);
        let phases: Vec<&'static str> = {
            let mut v: Vec<&'static str> = Vec::new();
            for it in &items {
                if v.last() != Some(&it.phase) {
                    v.push(it.phase);
                }
            }
            v
        };
        let mut phase_samples: Vec<Vec<f64>> = vec![Vec::new(); phases.len()];
        let mut gap_samples: Vec<Vec<f64>> = vec![Vec::new(); phases.len()];
        let mut in_seq: Vec<Vec<f64>> = vec![Vec::new(); items.len()];
        for shape in 0..3u32 {
            for r in 0..=reps {
                // The restore in a submission of its own, finished before the timed one: a
                // pass's begin timestamp is taken when Metal starts it, which can be before
                // the work encoded ahead of it has finished.
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay restore") });
                restore_all(&mut enc, self, &input);
                queue.submit([enc.finish()]);
                wait_idle(&device, &queue);
                timer.room(&device, &queue, n_items);
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay step") });
                let slots = match shape {
                    0 => {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("replay:step"), timestamp_writes: Some(timer.writes(0)) });
                        for it in &items {
                            self.replay_one(&mut pass, it);
                        }
                        1
                    }
                    1 => {
                        let mut slot = 0;
                        let mut k = 0;
                        while k < items.len() {
                            let phase = items[k].phase;
                            let label = format!("replay:{phase}");
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(&label), timestamp_writes: Some(timer.writes(slot)) });
                            while k < items.len() && items[k].phase == phase {
                                self.replay_one(&mut pass, &items[k]);
                                k += 1;
                            }
                            slot += 1;
                        }
                        slot
                    }
                    _ => {
                        for (k, it) in items.iter().enumerate() {
                            let label = format!("replay:{}/{}", it.phase, it.kernel);
                            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(&label), timestamp_writes: Some(timer.writes(k as u32)) });
                            self.replay_one(&mut pass, it);
                        }
                        n_items
                    }
                };
                let t = timer.finish(&device, &queue, enc, slots);
                if r == 0 {
                    continue; // warm-up
                }
                let sum: f64 = t.iter().map(|&x| timer.ms(x)).sum();
                // Written slots only (a pass may write none).
                let span = timer.ms((t.iter().map(|x| x.0).filter(|&x| x > 0).min().unwrap_or(0), t.iter().map(|x| x.1).filter(|&x| x > 0).max().unwrap_or(0)));
                match shape {
                    0 => report.whole_step_ms.push(sum),
                    1 => {
                        report.phases_sum_ms.push(sum);
                        report.phases_span_ms.push(span);
                        for (i, &x) in t.iter().enumerate() {
                            phase_samples[i].push(timer.ms(x));
                            gap_samples[i].push(if i == 0 || t[i - 1].1 > x.0 { 0.0 } else { timer.ms((t[i - 1].1, x.0)) });
                        }
                    }
                    _ => {
                        report.dispatch_split_sum_ms.push(sum);
                        report.dispatch_split_span_ms.push(span);
                        // Completion to completion: each pass's end after the previous one's
                        // (the first from its own begin). These sum to the span.
                        // (From the last pass that wrote its end: an unwritten pass is 0.)
                        let mut last = 0u64;
                        for (i, &x) in t.iter().enumerate() {
                            let from = if last == 0 { x.0 } else { last };
                            in_seq[i].push(timer.ms((from, x.1)));
                            if x.1 > 0 {
                                last = x.1;
                            }
                        }
                    }
                }
            }
        }
        report.phases = phases.iter().zip(&phase_samples).map(|(p, s)| (p.to_string(), median(s))).collect();
        report.phase_gaps_ms = gap_samples.iter().map(|s| median(s)).collect();

        // Indirect workgroup counts as each dispatch sees them, and 4. the isolated replay.
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kernel replay args"),
            size: 16 * u64::from(n_items),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            // Back to the step's input.
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay rewind input") });
            restore_all(&mut enc, self, &input);
            queue.submit([enc.finish()]);
        }
        let mut isolated: Vec<Vec<f64>> = vec![Vec::new(); items.len()];
        for (k, it) in items.iter().enumerate() {
            let bound: Vec<usize> = {
                let mut b: Vec<usize> = self.item_bindings(it).into_iter().filter(|&i| i < n && input[i].is_some() && i != 6 && i != 22).collect();
                if matches!(it.cmd, rigid_plan::Cmd::Indirect { .. }) && !b.contains(&23) {
                    b.push(23);
                }
                b
            };
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay dispatch") });
            // `input` holds the state before dispatch k for every buffer it binds.
            if let rigid_plan::Cmd::Indirect { slot, .. } = it.cmd {
                enc.copy_buffer_to_buffer(input[23].as_ref().expect("args"), u64::from(slot) * 12, &args, 16 * k as u64, 12);
            }
            let runs = if opts.whole_step_only { 0 } else { reps + 1 };
            timer.room(&device, &queue, runs.max(1));
            for r in 0..runs {
                // Restore, finish, then the timed pass alone in its submission (see above).
                for &i in &bound {
                    let src = input[i].as_ref().expect("copyable");
                    enc.copy_buffer_to_buffer(src, 0, &self.bufs[i], 0, src.size());
                }
                queue.submit([std::mem::replace(&mut enc, device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay dispatch") })).finish()]);
                wait_idle(&device, &queue);
                let label = format!("replay:{}/{}", it.phase, it.kernel);
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(&label), timestamp_writes: Some(timer.writes(r)) });
                    self.replay_one(&mut pass, it);
                }
                queue.submit([std::mem::replace(&mut enc, device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay dispatch") })).finish()]);
            }
            if opts.whole_step_only {
                // Advance without timing: restore, run once.
                for &i in &bound {
                    let src = input[i].as_ref().expect("copyable");
                    enc.copy_buffer_to_buffer(src, 0, &self.bufs[i], 0, src.size());
                }
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("replay:advance"), timestamp_writes: None });
                self.replay_one(&mut pass, it);
            }
            // The state after dispatch k becomes the next one's input.
            for &i in &bound {
                let dst = input[i].as_ref().expect("copyable");
                enc.copy_buffer_to_buffer(&self.bufs[i], 0, dst, 0, dst.size());
            }
            if runs > 0 {
                let t = timer.finish(&device, &queue, enc, runs);
                isolated[k] = t[1..].iter().map(|&x| timer.ms(x)).collect();
            } else {
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
        }
        let arg_words: Vec<u32> = bytemuck::cast_slice(&read_mapped(&device, &args, 16 * u64::from(n_items))).to_vec();
        report.dispatches = items
            .iter()
            .enumerate()
            .map(|(k, it)| {
                let (workgroups, indirect) = match it.cmd {
                    rigid_plan::Cmd::Dispatch { threads, .. } => {
                        let (x, y) = groups(threads);
                        ([x, y, 1], false)
                    }
                    _ => ([arg_words[4 * k], arg_words[4 * k + 1], arg_words[4 * k + 2]], true),
                };
                ReplayedDispatch {
                    index: k as u32,
                    phase: it.phase,
                    kernel: it.kernel,
                    workgroups,
                    indirect,
                    isolated_ms: std::mem::take(&mut isolated[k]),
                    in_sequence_ms: std::mem::take(&mut in_seq[k]),
                }
            })
            .collect();

        report.unwritten_passes = timer.unwritten.get();
        // Everything back as it was before the replay.
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("kernel replay restore") });
            restore_all(&mut enc, self, &saved);
            queue.submit([enc.finish()]);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        Ok(report)
    }

    /// A digest (FNV-1a 64) of every copyable buffer of the step, waiting for the GPU:
    /// `(binding, bytes, digest)`. A diagnostic to prove two states equal (for example
    /// before and after [`Self::kernel_replay`]).
    pub fn buffer_digests(&self) -> Vec<(usize, u64, u64)> {
        let mut out = Vec::new();
        for (i, b) in self.bufs.iter().enumerate() {
            if !b.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                continue;
            }
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("digest read"),
                size: b.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("digest") });
            enc.copy_buffer_to_buffer(b, 0, &staging, 0, b.size());
            self.queue.submit([enc.finish()]);
            let bytes = read_mapped(&self.device, &staging, b.size());
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &x in &bytes {
                h = (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3);
            }
            out.push((i, b.size(), h));
        }
        out
    }

    /// The plan's buffer copies (encoded before the step's pass).
    fn encode_plan_copies(&self, enc: &mut wgpu::CommandEncoder, cmds: &[rigid_plan::Cmd]) {
        for c in cmds {
            if let rigid_plan::Cmd::Copy { src, src_offset, dst, dst_offset, size } = *c {
                let from = if src == rigid_plan::COUNTER_UPLOAD { &self.counter_upload } else { &self.bufs[src as usize] };
                enc.copy_buffer_to_buffer(from, u64::from(src_offset), &self.bufs[dst as usize], u64::from(dst_offset), u64::from(size));
            }
        }
    }

    /// The bindings of the pipeline `it` dispatches.
    fn item_bindings(&self, it: &Item) -> Vec<usize> {
        let name = match it.cmd {
            rigid_plan::Cmd::Indirect { kernel, .. } if kernel.starts_with("solve_primal_color") => {
                if self.compact_contacts {
                    "solve_primal_color_compact"
                } else {
                    "solve_primal_color"
                }
            }
            _ => it.kernel,
        };
        self.pipes.get(name).map(|p| p.bindings.iter().map(|&b| b as usize).collect()).unwrap_or_default()
    }

    fn replay_one<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, it: &Item) {
        match it.cmd {
            rigid_plan::Cmd::Dispatch { kernel, threads, pass: p } => self.disp(pass, kernel, threads, Some(p)),
            rigid_plan::Cmd::Indirect { kernel, slot, pass: p } => {
                if kernel.starts_with("solve_primal_color") {
                    self.disp_indirect(pass, slot);
                } else {
                    self.disp_work(pass, kernel, slot, Some(p));
                }
            }
            _ => {}
        }
    }
}
