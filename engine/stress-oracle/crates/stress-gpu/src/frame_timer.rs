//! The GPU time of a span of one command encoder's work (a world's frame, a draw), read back
//! without waiting: a few slots in flight, a sample taken when its readback lands. `None`
//! where the device has no timestamp queries.
//!
//! A compute span is bracketed by marker passes, each one workgroup that binds `deps`, the
//! buffers the span's work writes last. A GPU may run passes that share no buffer at the same
//! time (Chrome's Metal backend does), so a marker without them can stamp before the work it
//! brackets has run; binding them orders the end marker after their writers. A render span
//! takes its timestamps on the render pass itself ([`FrameTimer::render_writes`]).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::gpu::Gpu;

const SLOTS: usize = 4;
const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;
/// Samples the reported median is taken over (the newest).
const WINDOW: usize = 32;

struct Slot {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    state: Arc<AtomicU8>,
    /// The span's number (spans begun before it).
    seq: u64,
}

pub struct FrameTimer {
    slots: Vec<Slot>,
    /// The slot the open span writes, if one was free.
    open: Option<usize>,
    /// Slots ended and submitted, to be mapped by [`Self::submitted`].
    ended: Vec<usize>,
    period_ms: f64,
    /// The newest samples, ms (a ring of `WINDOW`).
    recent: Vec<f64>,
    /// The marker kernels by dependency count (a browser drops a pass with no work, its
    /// timestamps with it); built on first use.
    markers: Vec<Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>>,
    label: String,
    /// The newest span read: its number and end timestamp.
    prev: Option<(u64, u64)>,
    /// Spans begun, and samples read back.
    pub begun: u64,
    pub samples: u64,
}

impl FrameTimer {
    pub fn new(gpu: &Gpu, label: &str) -> Option<Self> {
        if !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let slots = (0..SLOTS)
            .map(|_| Slot {
                queries: gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some(label), ty: wgpu::QueryType::Timestamp, count: 2 }),
                resolve: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
                read: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
                state: Arc::new(AtomicU8::new(IDLE)),
                seq: 0,
            })
            .collect();
        Some(Self { markers: Vec::new(), label: label.to_string(), slots, open: None, ended: Vec::new(), period_ms: gpu.queue.get_timestamp_period() as f64 * 1e-6, recent: Vec::new(), prev: None, begun: 0, samples: 0 })
    }

    /// Opens a span in a free slot (none while every slot is still in flight).
    fn open_span(&mut self) -> Option<usize> {
        self.open = self.slots.iter().position(|s| s.state.load(Ordering::Acquire) == IDLE);
        if let Some(i) = self.open {
            self.slots[i].seq = self.begun;
            self.begun += 1;
        }
        self.open
    }

    /// One marker pass over `deps`, writing timestamp `index` at its start (0) or end (1).
    fn marker(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, slot: usize, index: u32, deps: &[&wgpu::Buffer]) {
        let n = deps.len();
        if self.markers.len() <= n {
            self.markers.resize_with(n + 1, || None);
        }
        if self.markers[n].is_none() {
            let decls: String = (0..n).map(|k| format!("@group(0) @binding({k}) var<storage, read_write> b{k}: array<u32>;\n")).collect();
            // Every binding used (the layout keeps it), none written in practice.
            let used = if n == 0 { String::new() } else { format!("if ({} == 0u) {{ b0[0] = 0u; }}", (0..n).map(|k| format!("arrayLength(&b{k})")).collect::<Vec<_>>().join(" + ")) };
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(&self.label), source: wgpu::ShaderSource::Wgsl(format!("{decls}@compute @workgroup_size(1) fn main() {{ {used} }}").into()) });
            let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&self.label), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
            let layout = pipeline.get_bind_group_layout(0);
            self.markers[n] = Some((pipeline, layout));
        }
        let (pipeline, layout) = self.markers[n].as_ref().expect("built above");
        let bind = (n > 0).then(|| {
            let entries: Vec<_> = deps.iter().enumerate().map(|(k, b)| wgpu::BindGroupEntry { binding: k as u32, resource: b.as_entire_binding() }).collect();
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(&self.label), layout, entries: &entries })
        });
        let q = &self.slots[slot].queries;
        let at = if index == 0 {
            wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: None }
        } else {
            wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: None, end_of_pass_write_index: Some(1) }
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(&self.label), timestamp_writes: Some(at) });
        pass.set_pipeline(pipeline);
        if let Some(b) = &bind {
            pass.set_bind_group(0, b, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// The span's start, before the encoder's first work; `deps` as for [`Self::end`].
    pub fn begin(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, deps: &[&wgpu::Buffer]) {
        if let Some(i) = self.open_span() {
            self.marker(gpu, enc, i, 0, deps);
        }
    }

    /// The span's end, after the encoder's last work: a marker over `deps`, the buffers the
    /// span's work writes last (it runs once their writers are done). Call
    /// [`Self::submitted`] once the encoder is submitted.
    pub fn end(&mut self, gpu: &Gpu, enc: &mut wgpu::CommandEncoder, deps: &[&wgpu::Buffer]) {
        let Some(i) = self.open else { return };
        self.marker(gpu, enc, i, 1, deps);
        self.finish(enc);
    }

    /// Opens a span timed on one render pass: pass [`Self::render_writes`] as its
    /// `timestamp_writes`, then call [`Self::finish`] after it.
    pub fn begin_render(&mut self) {
        self.open_span();
    }

    /// The open span's timestamps for a render pass (its start and end), if one is open.
    pub fn render_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.open.map(|i| wgpu::RenderPassTimestampWrites { query_set: &self.slots[i].queries, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) })
    }

    /// Closes the open span: its timestamps resolved for readback.
    pub fn finish(&mut self, enc: &mut wgpu::CommandEncoder) {
        let Some(i) = self.open.take() else { return };
        let s = &self.slots[i];
        enc.resolve_query_set(&s.queries, 0..2, &s.resolve, 0);
        enc.copy_buffer_to_buffer(&s.resolve, 0, &s.read, 0, 16);
        s.state.store(PENDING, Ordering::Release);
        self.ended.push(i);
    }

    /// The ended spans' readbacks, once their encoder is submitted.
    pub fn submitted(&mut self) {
        for i in self.ended.drain(..) {
            let state = self.slots[i].state.clone();
            self.slots[i].read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                state.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release);
            });
        }
    }

    /// Takes the samples that have landed (after a `device.poll`) and returns the median of
    /// the newest [`WINDOW`] spans' GPU time, ms (robust to a slow first frame or a stall). Consecutive spans can overlap on the GPU (the next
    /// one's start marker runs before the last one's work ends): a span directly after the
    /// one read before it counts from the later of its start and that one's end, so no work
    /// is counted twice.
    pub fn poll(&mut self) -> Option<f64> {
        let mut landed = Vec::new();
        for s in &self.slots {
            if s.state.load(Ordering::Acquire) != READY {
                continue;
            }
            let t: [u64; 2] = match s.read.slice(..).get_mapped_range() {
                Ok(view) => bytemuck::pod_read_unaligned(&view),
                Err(_) => continue,
            };
            s.read.unmap();
            s.state.store(IDLE, Ordering::Release);
            landed.push((s.seq, t));
        }
        landed.sort_by_key(|l| l.0);
        for (seq, [start, end]) in landed {
            self.samples += 1;
            let from = match self.prev {
                Some((p, prev_end)) if p + 1 == seq => start.max(prev_end),
                _ => start,
            };
            self.prev = Some((seq, end));
            if end > from {
                let ms = (end - from) as f64 * self.period_ms;
                if self.recent.len() == WINDOW {
                    self.recent.remove(0);
                }
                self.recent.push(ms);
            }
        }
        if self.recent.is_empty() {
            return None;
        }
        let mut sorted = self.recent.clone();
        sorted.sort_by(f64::total_cmp);
        Some(sorted[sorted.len() / 2])
    }
}
