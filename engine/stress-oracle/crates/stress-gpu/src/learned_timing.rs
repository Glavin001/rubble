//! GPU time per stage of the learned impact path (`STRESS_LEARNED_TIMING=1`, a diagnostic:
//! timestamps around each of its compute passes, resolved and read after the frame's read,
//! an extra read per frame). Stages: select, gather, features, network, decode, apply,
//! deform. Printed when the batch is dropped (the run's end).

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use crate::gpu::Gpu;

const PASSES: u32 = 1024;

pub struct StageTimer {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    used: Cell<u32>,
    labels: RefCell<Vec<&'static str>>,
    /// Per stage: (sum of ms, frames it ran in, passes).
    sums: RefCell<BTreeMap<&'static str, (f64, u64, u64)>>,
    frames: Cell<u64>,
}

impl StageTimer {
    /// The timer, when asked for and the device has timestamps.
    pub fn new(gpu: &Gpu) -> Option<StageTimer> {
        if !std::env::var("STRESS_LEARNED_TIMING").is_ok_and(|v| v == "1") || !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let qs = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("learned stage timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * PASSES });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("learned stage timestamps"), size: 16 * PASSES as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        Some(StageTimer { qs, resolve, used: Cell::new(0), labels: RefCell::new(Vec::new()), sums: RefCell::new(BTreeMap::new()), frames: Cell::new(0) })
    }

    /// Timestamp writes for one compute pass of `stage` (none once the frame's slots are used).
    pub fn writes(&self, stage: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let i = self.used.get();
        if i >= PASSES {
            return None;
        }
        self.used.set(i + 1);
        self.labels.borrow_mut().push(stage);
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.qs, beginning_of_pass_write_index: Some(2 * i), end_of_pass_write_index: Some(2 * i + 1) })
    }

    /// After the frame's submission finished: its passes' times added to their stages.
    pub fn collect(&self, gpu: &Gpu) {
        let n = self.used.get();
        if n == 0 {
            return;
        }
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        enc.resolve_query_set(&self.qs, 0..2 * n, &self.resolve, 0);
        gpu.queue.submit([enc.finish()]);
        let t: Vec<u64> = gpu.read_range(&self.resolve, 0, 2 * n as usize);
        let period = gpu.queue.get_timestamp_period() as f64 * 1e-6;
        let labels = std::mem::take(&mut *self.labels.borrow_mut());
        let mut frame: BTreeMap<&'static str, (f64, u64)> = BTreeMap::new();
        for (i, l) in labels.iter().enumerate() {
            let ms = t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * period;
            let e = frame.entry(l).or_insert((0.0, 0));
            e.0 += ms;
            e.1 += 1;
        }
        let mut sums = self.sums.borrow_mut();
        for (l, (ms, passes)) in frame {
            let e = sums.entry(l).or_insert((0.0, 0, 0));
            e.0 += ms;
            e.1 += 1;
            e.2 += passes;
        }
        self.used.set(0);
        self.frames.set(self.frames.get() + 1);
    }

    /// Per stage: mean ms over the frames it ran in, frames, passes per frame.
    pub fn report(&self) -> String {
        let sums = self.sums.borrow();
        let parts: Vec<String> = sums.iter().map(|(l, (ms, f, p))| format!("{l} {:.3} ms ({f} frames, {:.0} passes)", ms / (*f).max(1) as f64, *p as f64 / (*f).max(1) as f64)).collect();
        format!("learned stage GPU time per frame it ran in: {}", parts.join(", "))
    }
}
