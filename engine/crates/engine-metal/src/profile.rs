//! Profiling of the native Metal hosts ([`crate::MetalSim`], [`crate::MetalExplicitBeams`]).
//! How to use each tool: `engine/docs/profiling-metal.md`.
//!
//! - **Labels** (`RUBBLE_METAL_LABELS=1`, on by itself while a capture is requested):
//!   every command buffer is named after its step, every plan phase is a debug group and
//!   every dispatch a group of its own named after the plan's kernel, so Xcode and
//!   Instruments read like the step plan. Off by default: hundreds of extra Objective-C
//!   calls a step (they cost host time, never GPU time).
//! - **One-step capture** (`RUBBLE_CAPTURE_STEP=N`, `RUBBLE_CAPTURE_OUT=path.gputrace`):
//!   `MTLCaptureManager` writes a `.gputrace` of exactly step (or frame) N.
//! - **Debug libraries** (`RUBBLE_METAL_DEBUG_LIB=1`): kernels are compiled offline by
//!   `xcrun metal -gline-tables-only -frecord-sources` from the same MSL (whose `#line`
//!   directives name the `.slang` files) instead of at run time, so the shader profiler
//!   puts each line's cost on Slang source. Same MSL, same math mode.
//! - **Kernel replay** ([`replay`]): measurement only, never in the sim loop. From a copy
//!   of a step's buffers, every dispatch of the step runs alone in its own command
//!   buffer, `reps` times from the same input, timed by `GPUStartTime`/`GPUEndTime`.
//! - **Clock and trace** ([`host_seconds`], [`Trace`]): host laps and GPU intervals on
//!   one timeline, written as Chrome/Perfetto trace JSON.

use metal::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ---------------------------------------------------------------- environment

/// A one-step GPU capture requested by `RUBBLE_CAPTURE_STEP=N` (and
/// `RUBBLE_CAPTURE_OUT`, default `out/metal-capture/step<N>.gputrace` under the
/// repository root).
#[derive(Clone, Debug)]
pub struct CaptureRequest {
    pub step: u32,
    pub out: PathBuf,
}

impl CaptureRequest {
    pub fn from_env() -> Option<Self> {
        let step: u32 = std::env::var("RUBBLE_CAPTURE_STEP")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let out = std::env::var_os("RUBBLE_CAPTURE_OUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                repo_out()
                    .join("metal-capture")
                    .join(format!("step{step}.gputrace"))
            });
        Some(Self { step, out })
    }
}

/// The repository's gitignored `out/` (long-lived outputs: traces, captures).
pub fn repo_out() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../out")
}

fn env_on(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

/// Debug groups and command-buffer labels are encoded (`RUBBLE_METAL_LABELS=1`, or a
/// capture is requested).
pub fn labels_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("RUBBLE_METAL_LABELS") || CaptureRequest::from_env().is_some())
}

/// Before the first Metal device: a requested capture needs `MTL_CAPTURE_ENABLED=1`
/// (a process not launched from Xcode cannot capture otherwise).
pub(crate) fn prepare_env() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if CaptureRequest::from_env().is_some() && std::env::var_os("MTL_CAPTURE_ENABLED").is_none()
        {
            // SAFETY: set once, before this process creates its Metal device; the
            // hosts' constructors call this first.
            unsafe { std::env::set_var("MTL_CAPTURE_ENABLED", "1") };
        }
    });
}

/// Start capturing `queue`'s work into a `.gputrace` at `out` (replaced if present).
pub(crate) fn begin_capture(queue: &CommandQueueRef, out: &Path) -> anyhow::Result<()> {
    let manager = CaptureManager::shared();
    anyhow::ensure!(
        manager.supports_destination(MTLCaptureDestination::GpuTraceDocument),
        "this process cannot write .gputrace documents (MTL_CAPTURE_ENABLED=1 must be set before the device exists)"
    );
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if out.exists() {
        std::fs::remove_dir_all(out).or_else(|_| std::fs::remove_file(out))?;
    }
    let desc = CaptureDescriptor::new();
    desc.set_capture_command_queue(queue);
    desc.set_destination(MTLCaptureDestination::GpuTraceDocument);
    desc.set_output_url(out);
    manager
        .start_capture(&desc)
        .map_err(|e| anyhow::anyhow!("start capture: {e}"))
}

pub(crate) fn end_capture(out: &Path) {
    CaptureManager::shared().stop_capture();
    eprintln!("rubble: wrote GPU capture {}", out.display());
}

// ---------------------------------------------------------------- libraries

/// A Metal library of `msl` (entry `entry`) in the given math mode: compiled at run time
/// by default; with `RUBBLE_METAL_DEBUG_LIB=1`, offline with line tables and the source
/// recorded (cached in `out/metal-debuglib/` by a hash of the source and flags).
pub(crate) fn new_library(
    device: &DeviceRef,
    msl: &str,
    entry: &str,
    fast_math: bool,
) -> anyhow::Result<Library> {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    if !*DEBUG.get_or_init(|| env_on("RUBBLE_METAL_DEBUG_LIB")) {
        let o = CompileOptions::new();
        o.set_fast_math_enabled(fast_math);
        return device
            .new_library_with_source(msl, &o)
            .map_err(|e| anyhow::anyhow!("compile {entry}: {e}"));
    }
    let lib = debug_metallib(msl, entry, fast_math)?;
    device
        .new_library_with_file(&lib)
        .map_err(|e| anyhow::anyhow!("load {}: {e}", lib.display()))
}

fn debug_metallib(msl: &str, entry: &str, fast_math: bool) -> anyhow::Result<PathBuf> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (msl, fast_math, "v1").hash(&mut h);
    let dir = repo_out().join("metal-debuglib");
    std::fs::create_dir_all(&dir)?;
    let stem = format!("{entry}-{:016x}", h.finish());
    let lib = dir.join(format!("{stem}.metallib"));
    if lib.is_file() {
        return Ok(lib);
    }
    let src = dir.join(format!("{stem}.metal"));
    let air = dir.join(format!("{stem}.air"));
    std::fs::write(&src, msl)?;
    let math = if fast_math {
        "-ffast-math"
    } else {
        "-fno-fast-math"
    };
    let st = std::process::Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            "-gline-tables-only",
            "-frecord-sources",
            math,
        ])
        .arg(&src)
        .arg("-o")
        .arg(&air)
        .status()?;
    anyhow::ensure!(st.success(), "xcrun metal failed for {entry}");
    let st = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .status()?;
    anyhow::ensure!(st.success(), "xcrun metallib failed for {entry}");
    let _ = std::fs::remove_file(&air);
    Ok(lib)
}

// ---------------------------------------------------------------- clock

#[repr(C)]
struct Timebase {
    numer: u32,
    denom: u32,
}

extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut Timebase) -> i32;
}

/// Host time in seconds on the clock of `GPUStartTime`/`GPUEndTime` (mach absolute
/// time, as `CACurrentMediaTime`), so host laps and GPU intervals share one timeline.
pub fn host_seconds() -> f64 {
    static TB: OnceLock<(u32, u32)> = OnceLock::new();
    let &(n, d) = TB.get_or_init(|| {
        let mut tb = Timebase { numer: 0, denom: 0 };
        // SAFETY: plain libSystem call writing the struct.
        unsafe { mach_timebase_info(&mut tb) };
        (tb.numer, tb.denom.max(1))
    });
    // SAFETY: plain libSystem call.
    let t = unsafe { mach_absolute_time() };
    t as f64 * f64::from(n) / f64::from(d) * 1e-9
}

/// (`GPUStartTime`, `GPUEndTime`) of a completed command buffer, seconds on
/// [`host_seconds`]'s clock.
#[allow(unexpected_cfgs)] // objc 0.2 msg_send! checks a cargo-clippy feature
pub fn gpu_interval(cb: &CommandBufferRef) -> (f64, f64) {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    let obj = cb.as_ptr() as *mut objc::runtime::Object;
    // SAFETY: MTLCommandBuffer responds to both selectors once completed.
    unsafe { (msg_send![obj, GPUStartTime], msg_send![obj, GPUEndTime]) }
}

/// A completed command buffer's interval, kept for the trace.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuSpan {
    pub label: String,
    pub start_s: f64,
    pub end_s: f64,
}

/// A plan phase's GPU interval in a split step (`MetalSim::step_profiled_spans`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct PhaseSpan {
    pub label: &'static str,
    pub start_s: f64,
    pub end_s: f64,
}

// ---------------------------------------------------------------- trace

/// Chrome/Perfetto trace events (`chrome://tracing`, `ui.perfetto.dev`): complete events
/// on named tracks, times in µs from the first event's clock origin.
#[derive(Default)]
pub struct Trace {
    events: Vec<serde_json::Value>,
    tracks: HashMap<(u32, String), u32>,
    origin: Option<f64>,
}

impl Trace {
    pub fn new() -> Self {
        Self::default()
    }

    fn tid(&mut self, pid: u32, track: &str) -> u32 {
        let n = self.tracks.len() as u32 + 1;
        let tid = *self.tracks.entry((pid, track.to_string())).or_insert(n);
        if tid == n {
            self.events.push(serde_json::json!({
                "ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": { "name": track }
            }));
        }
        tid
    }

    /// A span on `track` of process `pid` (1: host, 2: GPU), times in seconds on
    /// [`host_seconds`]'s clock.
    pub fn span(
        &mut self,
        pid: u32,
        track: &str,
        name: &str,
        start_s: f64,
        end_s: f64,
        args: serde_json::Value,
    ) {
        let origin = *self.origin.get_or_insert(start_s);
        let tid = self.tid(pid, track);
        self.events.push(serde_json::json!({
            "ph": "X", "name": name, "cat": track, "pid": pid, "tid": tid,
            "ts": (start_s - origin) * 1e6, "dur": ((end_s - start_s) * 1e6).max(0.0), "args": args,
        }));
    }

    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let names = serde_json::json!([
            { "ph": "M", "name": "process_name", "pid": 1, "args": { "name": "host (CPU)" } },
            { "ph": "M", "name": "process_name", "pid": 2, "args": { "name": "GPU" } },
        ]);
        let mut all = names.as_array().cloned().unwrap_or_default();
        all.extend(self.events.iter().cloned());
        let doc = serde_json::json!({ "traceEvents": all, "displayTimeUnit": "ms" });
        std::fs::write(path, serde_json::to_vec(&doc)?)?;
        Ok(())
    }
}

// ---------------------------------------------------------------- kernel replay

/// A dispatch's grid.
#[derive(Clone, Debug)]
pub enum Grid {
    Groups(u64, u64, u64),
    /// Workgroup counts at byte `offset` of buffer `buf` (x, y, z words).
    Indirect {
        buf: usize,
        offset: u64,
    },
}

/// One dispatch of a step, as its host encodes it.
#[derive(Clone)]
pub struct ReplayDispatch {
    /// The plan's (logical) kernel name.
    pub kernel: String,
    /// The compiled entry point.
    pub entry: String,
    pub phase: String,
    pub pso: ComputePipelineState,
    pub group: MTLSize,
    pub grid: Grid,
    /// (Metal `[[buffer(n)]]`, buffer and offset; None unbinds).
    pub binds: Vec<(u64, Option<(usize, u64)>)>,
    /// Metal buffer indices the kernel may write (not `const`/`constant` in its MSL).
    pub writable: Vec<u64>,
}

#[derive(Clone)]
pub enum ReplayOp {
    Dispatch(Box<ReplayDispatch>),
    Copy {
        src: usize,
        src_offset: u64,
        dst: usize,
        dst_offset: u64,
        size: u64,
        phase: String,
    },
}

/// Medians of one op, from `reps` runs alone.
#[derive(Clone, Debug, serde::Serialize)]
pub struct OpTiming {
    pub index: usize,
    pub kind: &'static str,
    pub kernel: String,
    pub entry: String,
    pub phase: String,
    pub groups: [u64; 3],
    pub group_size: [u64; 3],
    pub threads: u64,
    /// GPU ms of each run (its own command buffer).
    pub ms: Vec<f64>,
    pub median_ms: f64,
    pub p90_ms: f64,
    /// Bytes of the writable buffers bound (an upper bound of what it may write).
    pub writable_bytes: u64,
    /// Bytes of the read-only buffers bound (an upper bound of what it may read).
    pub readonly_bytes: u64,
    /// 128-byte lines whose content the op changed, ×128: a lower bound of its DRAM
    /// writes (a line written with the value it held does not count).
    pub changed_bytes: u64,
}

/// What [`replay`] measured.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ReplayReport {
    pub reps: u32,
    pub ops: Vec<OpTiming>,
    /// The whole step encoded as its host does (one command buffer, serial dispatch),
    /// GPU ms of each run from the same input.
    pub whole_step_ms: Vec<f64>,
    /// An empty kernel's command buffer: the floor every op's time includes.
    pub launch_floor_ms: Vec<f64>,
}

/// Percentile of unsorted samples (nearest rank).
pub fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[(((s.len() - 1) as f64) * p).round() as usize]
}

/// Buffer indices of `entry`'s parameters that are not `const`/`constant`.
pub fn writable_buffers(msl: &str, entry: &str) -> Vec<u64> {
    let Some(at) = msl.find(&format!("[[kernel]] void {entry}(")) else {
        return Vec::new();
    };
    // The signature runs to the body's `{` (parameters hold no braces).
    let sig = &msl[at..];
    let sig = &sig[..sig.find('{').unwrap_or(sig.len())];
    let mut out = Vec::new();
    for param in sig.split(',') {
        let Some(b) = param.find("[[buffer(") else {
            continue;
        };
        let n: u64 = param[b + 9..]
            .split(')')
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(u64::MAX);
        let head = &param[..b];
        let ro = head.contains("const ") || head.contains("constant ") || head.contains("const*");
        if !ro && n != u64::MAX {
            out.push(n);
        }
    }
    out
}

fn blit_copy(queue: &CommandQueueRef, pairs: &[(&Buffer, &Buffer)]) {
    if pairs.is_empty() {
        return;
    }
    let cb = queue.new_command_buffer();
    let b = cb.new_blit_command_encoder();
    for (src, dst) in pairs {
        b.copy_from_buffer(src, 0, dst, 0, src.length().min(dst.length()));
    }
    b.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn clone_all(device: &DeviceRef, queue: &CommandQueueRef, bufs: &[&Buffer]) -> Vec<Buffer> {
    let out: Vec<Buffer> = bufs
        .iter()
        .map(|b| device.new_buffer(b.length(), MTLResourceOptions::StorageModeShared))
        .collect();
    let pairs: Vec<(&Buffer, &Buffer)> = bufs.iter().copied().zip(out.iter()).collect();
    blit_copy(queue, &pairs);
    out
}

fn encode_op(cb: &CommandBufferRef, bufs: &[Buffer], op: &ReplayOp) {
    match op {
        ReplayOp::Copy {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
            ..
        } => {
            let b = cb.new_blit_command_encoder();
            b.copy_from_buffer(&bufs[*src], *src_offset, &bufs[*dst], *dst_offset, *size);
            b.end_encoding();
        }
        ReplayOp::Dispatch(d) => {
            let desc = ComputePassDescriptor::new();
            desc.set_dispatch_type(MTLDispatchType::Serial);
            let e = cb.compute_command_encoder_with_descriptor(desc);
            encode_dispatch(e, bufs, d);
            e.end_encoding();
        }
    }
}

fn encode_dispatch(e: &ComputeCommandEncoderRef, bufs: &[Buffer], d: &ReplayDispatch) {
    e.set_compute_pipeline_state(&d.pso);
    for &(index, b) in &d.binds {
        match b {
            Some((buf, offset)) => e.set_buffer(index, Some(&bufs[buf]), offset),
            None => e.set_buffer(index, None, 0),
        }
    }
    match d.grid {
        Grid::Groups(x, y, z) => e.dispatch_thread_groups(MTLSize::new(x, y, z), d.group),
        Grid::Indirect { buf, offset } => {
            e.dispatch_thread_groups_indirect(&bufs[buf], offset, d.group)
        }
    }
}

fn read_groups(bufs: &[Buffer], d: &ReplayDispatch) -> [u64; 3] {
    match d.grid {
        Grid::Groups(x, y, z) => [x, y, z],
        Grid::Indirect { buf, offset } => {
            let b = &bufs[buf];
            if offset + 12 > b.length() {
                return [0; 3];
            }
            // SAFETY: shared buffer, in bounds, GPU idle.
            let w = unsafe {
                std::slice::from_raw_parts(
                    (b.contents() as *const u8).add(offset as usize) as *const u32,
                    3,
                )
            };
            [u64::from(w[0]), u64::from(w[1]), u64::from(w[2])]
        }
    }
}

fn changed_lines(a: &Buffer, b: &Buffer) -> u64 {
    let n = a.length().min(b.length()) as usize;
    // SAFETY: shared buffers of at least n bytes, GPU idle.
    let (x, y) = unsafe {
        (
            std::slice::from_raw_parts(a.contents() as *const u8, n),
            std::slice::from_raw_parts(b.contents() as *const u8, n),
        )
    };
    x.chunks(128)
        .zip(y.chunks(128))
        .filter(|(p, q)| p != q)
        .count() as u64
        * 128
}

fn timed(queue: &CommandQueueRef, label: &str, encode: impl FnOnce(&CommandBufferRef)) -> f64 {
    let cb = queue.new_command_buffer();
    if labels_enabled() {
        cb.set_label(label);
    }
    encode(cb);
    cb.commit();
    cb.wait_until_completed();
    assert_ne!(
        cb.status(),
        MTLCommandBufferStatus::Error,
        "replay: {label} failed"
    );
    let (s, e) = gpu_interval(cb);
    (e - s) * 1e3
}

/// Replay a step's `ops` on copies of `bufs` (the live buffers are only read). Each op
/// runs alone `reps` times from the same input: before every run but the first, the
/// buffers it may write are restored from a copy taken before it. Then the whole step
/// runs `reps` times in one command buffer from the step's input. Measurement only.
pub fn replay(
    device: &DeviceRef,
    queue: &CommandQueueRef,
    bufs: &[&Buffer],
    ops: &[ReplayOp],
    reps: u32,
) -> ReplayReport {
    let reps = reps.max(1);
    let pristine = clone_all(device, queue, bufs);
    let work = clone_all(device, queue, bufs);
    let restore_all = || {
        let pairs: Vec<(&Buffer, &Buffer)> = pristine.iter().zip(work.iter()).collect();
        blit_copy(queue, &pairs);
    };

    // The launch floor: an empty kernel in its own command buffer.
    let noop = {
        let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void rubble_noop(device uint* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xffffffffu) b[0] = 1u; }\n";
        let lib = device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("noop kernel");
        let f = lib.get_function("rubble_noop", None).expect("noop");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("noop pipeline")
    };
    let scratch = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    let launch_floor_ms: Vec<f64> = (0..reps.max(5))
        .map(|_| {
            timed(queue, "noop", |cb| {
                let e = cb.new_compute_command_encoder();
                e.set_compute_pipeline_state(&noop);
                e.set_buffer(0, Some(&scratch), 0);
                e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(64, 1, 1));
                e.end_encoding();
            })
        })
        .collect();

    // The whole step in one command buffer, one compute encoder per run of dispatches
    // (serial, as the hosts encode it), copies in blit encoders between them.
    let mut whole_step_ms = Vec::new();
    for _ in 0..reps {
        restore_all();
        whole_step_ms.push(timed(queue, "whole step", |cb| {
            let mut enc: Option<&ComputeCommandEncoderRef> = None;
            for op in ops {
                match op {
                    ReplayOp::Dispatch(d) => {
                        let e = *enc.get_or_insert_with(|| {
                            let desc = ComputePassDescriptor::new();
                            desc.set_dispatch_type(MTLDispatchType::Serial);
                            cb.compute_command_encoder_with_descriptor(desc)
                        });
                        encode_dispatch(e, &work, d);
                    }
                    ReplayOp::Copy { .. } => {
                        if let Some(e) = enc.take() {
                            e.end_encoding();
                        }
                        encode_op(cb, &work, op);
                    }
                }
            }
            if let Some(e) = enc.take() {
                e.end_encoding();
            }
        }));
    }

    // Each op alone, in step order, from the state the ops before it left.
    restore_all();
    let mut saved: HashMap<usize, Buffer> = HashMap::new();
    let mut out = Vec::with_capacity(ops.len());
    for (index, op) in ops.iter().enumerate() {
        // The buffers this op may write: copied aside, restored before each repeat.
        let (written, readonly): (Vec<usize>, Vec<usize>) = match op {
            ReplayOp::Copy { dst, src, .. } => (vec![*dst], vec![*src]),
            ReplayOp::Dispatch(d) => {
                let mut w = Vec::new();
                let mut r = Vec::new();
                for &(i, b) in &d.binds {
                    if let Some((buf, _)) = b {
                        if d.writable.contains(&i) {
                            w.push(buf);
                        } else {
                            r.push(buf);
                        }
                    }
                }
                w.sort_unstable();
                w.dedup();
                r.sort_unstable();
                r.dedup();
                r.retain(|b| !w.contains(b));
                (w, r)
            }
        };
        for &b in &written {
            saved.entry(b).or_insert_with(|| {
                device.new_buffer(work[b].length(), MTLResourceOptions::StorageModeShared)
            });
        }
        let save: Vec<(&Buffer, &Buffer)> =
            written.iter().map(|&b| (&work[b], &saved[&b])).collect();
        blit_copy(queue, &save);
        let restore: Vec<(&Buffer, &Buffer)> =
            written.iter().map(|&b| (&saved[&b], &work[b])).collect();
        let (kind, kernel, entry, phase, groups, group_size) = match op {
            ReplayOp::Copy { size, phase, .. } => (
                "copy",
                "copy".to_string(),
                format!("{size} B"),
                phase.clone(),
                [0; 3],
                [0; 3],
            ),
            ReplayOp::Dispatch(d) => (
                if matches!(d.grid, Grid::Indirect { .. }) {
                    "indirect"
                } else {
                    "dispatch"
                },
                d.kernel.clone(),
                d.entry.clone(),
                d.phase.clone(),
                read_groups(&work, d),
                [d.group.width, d.group.height, d.group.depth],
            ),
        };
        let mut ms = Vec::with_capacity(reps as usize);
        for r in 0..reps {
            if r > 0 {
                blit_copy(queue, &restore);
            }
            ms.push(timed(queue, &kernel, |cb| encode_op(cb, &work, op)));
        }
        let changed_bytes = written
            .iter()
            .map(|&b| changed_lines(&work[b], &saved[&b]))
            .sum();
        let bytes = |v: &[usize]| v.iter().map(|&b| work[b].length()).sum::<u64>();
        out.push(OpTiming {
            index,
            kind,
            kernel,
            entry,
            phase,
            groups,
            group_size,
            threads: groups.iter().product::<u64>() * group_size.iter().product::<u64>(),
            median_ms: percentile(&ms, 0.5),
            p90_ms: percentile(&ms, 0.9),
            ms,
            writable_bytes: bytes(&written),
            readonly_bytes: bytes(&readonly),
            changed_bytes,
        });
    }
    ReplayReport {
        reps,
        ops: out,
        whole_step_ms,
        launch_floor_ms,
    }
}

/// Each dispatch's phase: the plan's `Phase` markers end a phase, so a command takes
/// the name of the next marker after it (`tail` past the last).
pub fn phases_of<C>(cmds: &[C], marker: impl Fn(&C) -> Option<&'static str>) -> Vec<&'static str> {
    let mut out = vec!["tail"; cmds.len()];
    let mut current = "tail";
    for (i, c) in cmds.iter().enumerate().rev() {
        if let Some(name) = marker(c) {
            current = name;
        }
        out[i] = current;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_buffers_reads_const() {
        let msl = "[[kernel]] void k(uint3 i [[thread_position_in_grid]], float device* a [[buffer(0)]], float const device* b [[buffer(1)]], constant P* p [[buffer(2)]], uint device* c [[buffer(3)]])\n{ }";
        assert_eq!(writable_buffers(msl, "k"), vec![0, 3]);
    }

    #[test]
    fn phases_from_end_markers() {
        let cmds = ["a", "b", "|x", "c", "|y", "d"];
        let p = phases_of(&cmds, |c| {
            c.strip_prefix('|')
                .map(|s| -> &'static str { Box::leak(s.to_string().into_boxed_str()) })
        });
        assert_eq!(p, vec!["x", "x", "x", "y", "y", "tail"]);
    }
}
