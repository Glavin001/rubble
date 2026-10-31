//! Diagnostic GPU timing across a frame's encoders (`RUBBLE_GPU_TIMING=1`): a mark is an
//! empty compute pass that writes a timestamp as it begins, so the time between two marks
//! is the GPU time of everything encoded between them, whoever encoded it (the step plan's
//! phases, a host's own passes after the step).

use std::sync::Mutex;

pub struct GpuMarks {
    queries: wgpu::QuerySet,
    /// A no-op kernel each mark's pass runs (a pass with no work writes no timestamp on
    /// some backends).
    noop: wgpu::ComputePipeline,
    names: Mutex<Vec<&'static str>>,
    cap: u32,
}

impl GpuMarks {
    /// Timing was asked for (`RUBBLE_GPU_TIMING=1`) and the device can write timestamps
    /// (the env var also makes `WgpuSim` request the feature).
    pub fn enabled(device: &wgpu::Device) -> bool {
        std::env::var("RUBBLE_GPU_TIMING").is_ok_and(|v| v == "1")
            && device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// Room for `cap` marks; more are ignored.
    pub fn new(device: &wgpu::Device, cap: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu marks noop"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let noop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu marks noop"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            noop,
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu marks"),
                ty: wgpu::QueryType::Timestamp,
                count: cap,
            }),
            names: Mutex::new(Vec::new()),
            cap,
        }
    }

    /// The end of segment `name` (and the start of the next).
    pub fn mark(&self, enc: &mut wgpu::CommandEncoder, name: &'static str) {
        let mut names = self.names.lock().expect("marks");
        let index = names.len() as u32;
        if index >= self.cap {
            return;
        }
        names.push(name);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        });
        pass.set_pipeline(&self.noop);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Resolve the marks into a readable buffer (encoded into `enc`; read it with
    /// [`Self::report`] once `enc` is submitted).
    pub fn resolve(&self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
        let n = self.names.lock().expect("marks").len().max(1) as u64;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu marks resolve"),
            size: 8 * n,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu marks read"),
            size: 8 * n,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.resolve_query_set(&self.queries, 0..n as u32, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 8 * n);
        staging
    }

    /// Each segment's GPU ms (from its previous mark to its own), waiting for the GPU.
    pub fn report(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: &wgpu::Buffer,
    ) -> Vec<(&'static str, f64)> {
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let t: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        let period = queue.get_timestamp_period() as f64 * 1e-6;
        let names = self.names.lock().expect("marks");
        (1..names.len())
            .map(|k| (names[k], t[k].saturating_sub(t[k - 1]) as f64 * period))
            .collect()
    }
}
