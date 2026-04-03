//! Multi-GPU support for distributing physics simulation across multiple wgpu devices.

use crate::buffer::GpuBuffer;
use crate::context::GpuContext;
use crate::kernel::{round_up_workgroups, ComputeKernel};
use crate::GpuError;
use std::marker::PhantomData;

/// A single GPU device with its associated queue and metadata.
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub device_index: usize,
}

impl GpuDevice {
    /// Create a [`GpuContext`] view of this device (borrows device and queue).
    /// This allows reusing existing `GpuBuffer` and `ComputeKernel` APIs that
    /// accept `&GpuContext`.
    pub fn as_context(&self) -> GpuContext {
        // We need GpuContext to work with references. Since GpuContext owns its
        // fields, we create a helper that wraps references. However, the existing
        // API uses owned GpuContext. We'll work directly with device/queue instead
        // for multi-GPU operations.
        //
        // For compatibility we provide this but the caller should be aware it
        // clones the Arc-backed wgpu handles (which is cheap).
        GpuContext::from_device_queue(self.device.clone(), self.queue.clone())
    }
}

/// Describes how work is distributed across GPUs.
#[derive(Debug, Clone)]
pub enum WorkDistribution {
    /// Split items evenly across all GPUs.
    EvenSplit,
    /// Assign explicit index ranges to specific devices.
    RangeBased { ranges: Vec<std::ops::Range<u32>> },
    /// Use a single device (fallback).
    SingleDevice(usize),
}

/// Multi-GPU context that manages multiple wgpu devices for parallel physics
/// simulation. Supports automatic work distribution across available GPUs.
pub struct MultiGpuContext {
    devices: Vec<GpuDevice>,
}

impl MultiGpuContext {
    /// Enumerate all available GPU adapters, create a device and queue for each.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });

        let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN).await;

        if adapters.is_empty() {
            return Err(GpuError::NoAdapter);
        }

        let mut devices = Vec::new();
        for (idx, adapter) in adapters.into_iter().enumerate() {
            let info = adapter.get_info();
            let result = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    ..Default::default()
                })
                .await;

            match result {
                Ok((device, queue)) => {
                    devices.push(GpuDevice {
                        device,
                        queue,
                        adapter_info: info,
                        device_index: idx,
                    });
                }
                Err(_) => {
                    // Skip adapters that fail device creation.
                    continue;
                }
            }
        }

        if devices.is_empty() {
            return Err(GpuError::NoAdapter);
        }

        Ok(Self { devices })
    }

    /// Number of GPU devices available.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get a reference to a specific device by index.
    pub fn device(&self, index: usize) -> &GpuDevice {
        &self.devices[index]
    }

    /// Returns the primary (first / highest performance) device.
    pub fn primary(&self) -> &GpuDevice {
        &self.devices[0]
    }

    /// Distribute `total_items` across devices according to the given strategy.
    /// Returns a list of `(device_index, offset, count)` triples.
    pub fn distribute_work(
        &self,
        total_items: u32,
        strategy: &WorkDistribution,
    ) -> Vec<(usize, u32, u32)> {
        match strategy {
            WorkDistribution::EvenSplit => {
                let n = self.devices.len() as u32;
                if n == 0 || total_items == 0 {
                    return Vec::new();
                }
                let base = total_items / n;
                let remainder = total_items % n;
                let mut result = Vec::with_capacity(n as usize);
                let mut offset = 0u32;
                for i in 0..n {
                    let count = base + if i < remainder { 1 } else { 0 };
                    if count > 0 {
                        result.push((i as usize, offset, count));
                    }
                    offset += count;
                }
                result
            }
            WorkDistribution::RangeBased { ranges } => ranges
                .iter()
                .enumerate()
                .filter(|(_, r)| r.start < r.end)
                .map(|(i, r)| (i, r.start, r.end - r.start))
                .collect(),
            WorkDistribution::SingleDevice(idx) => {
                vec![(*idx, 0, total_items)]
            }
        }
    }

    /// Upload `data` to all devices via the given multi-GPU buffer.
    pub fn sync_buffer_to_all<T: bytemuck::Pod>(
        &self,
        _src_device: usize,
        data: &[T],
        multi_buf: &mut MultiGpuBuffer<T>,
    ) {
        multi_buf.upload_to_all(self, data);
    }
}

/// Per-device buffer mirror for multi-GPU synchronization.
/// Holds one `GpuBuffer<T>` per device.
pub struct MultiGpuBuffer<T: bytemuck::Pod> {
    buffers: Vec<GpuBuffer<T>>,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> MultiGpuBuffer<T> {
    /// Allocate a buffer on every device in the context.
    pub fn new(ctx: &MultiGpuContext, capacity: usize) -> Self {
        let buffers = ctx
            .devices
            .iter()
            .map(|dev| {
                let gpu_ctx = dev.as_context();
                GpuBuffer::new(&gpu_ctx, capacity)
            })
            .collect();
        Self {
            buffers,
            _marker: PhantomData,
        }
    }

    /// Get the buffer associated with a specific device.
    pub fn buffer_on(&self, device_index: usize) -> &GpuBuffer<T> {
        &self.buffers[device_index]
    }

    /// Get a mutable reference to the buffer on a specific device.
    pub fn buffer_on_mut(&mut self, device_index: usize) -> &mut GpuBuffer<T> {
        &mut self.buffers[device_index]
    }

    /// Upload data to every device's buffer.
    pub fn upload_to_all(&mut self, ctx: &MultiGpuContext, data: &[T]) {
        for (i, buf) in self.buffers.iter_mut().enumerate() {
            let gpu_ctx = ctx.devices[i].as_context();
            buf.upload(&gpu_ctx, data);
        }
    }

    /// Download the buffer contents from a specific device.
    pub fn download_from(&self, ctx: &MultiGpuContext, device_index: usize) -> Vec<T> {
        let gpu_ctx = ctx.devices[device_index].as_context();
        self.buffers[device_index].download(&gpu_ctx)
    }

    /// Download results from each device according to the given ranges and
    /// merge them into a single contiguous vector.
    ///
    /// `ranges` contains `(device_index, offset, count)` triples. Each device's
    /// portion is placed at the correct offset in the output vector.
    pub fn gather_results(&self, ctx: &MultiGpuContext, ranges: &[(usize, u32, u32)]) -> Vec<T> {
        if ranges.is_empty() {
            return Vec::new();
        }

        // Determine total size from the ranges.
        let total = ranges
            .iter()
            .map(|(_, offset, count)| offset + count)
            .max()
            .unwrap_or(0) as usize;

        let mut result: Vec<T> = vec![T::zeroed(); total];

        for &(dev_idx, offset, count) in ranges {
            let gpu_ctx = ctx.devices[dev_idx].as_context();
            let data = self.buffers[dev_idx].download(&gpu_ctx);
            let src_slice = &data[..(count as usize)];
            result[offset as usize..(offset + count) as usize].copy_from_slice(src_slice);
        }

        result
    }
}

/// Pool-level orchestration for dispatching compute kernels across multiple GPUs.
pub struct GpuDevicePool;

impl GpuDevicePool {
    /// Compile and dispatch a compute kernel on each device in parallel according
    /// to the work distribution. Each device gets its own pipeline and buffer slice.
    ///
    /// The `input` multi-GPU buffer must already contain the data on all devices.
    /// The `output` multi-GPU buffer will receive the results, which can later be
    /// gathered with `MultiGpuBuffer::gather_results`.
    pub fn dispatch_parallel<T: bytemuck::Pod>(
        ctx: &MultiGpuContext,
        kernel_source: &str,
        entry_point: &str,
        input: &MultiGpuBuffer<T>,
        output: &mut MultiGpuBuffer<T>,
        work_dist: &[(usize, u32, u32)],
        workgroup_size: u32,
    ) {
        for &(dev_idx, _offset, count) in work_dist {
            let gpu_ctx = ctx.device(dev_idx).as_context();
            let kernel = ComputeKernel::from_wgsl(&gpu_ctx, kernel_source, entry_point);

            let in_buf = input.buffer_on(dev_idx);
            let out_buf = output.buffer_on(dev_idx);

            let bind_group = gpu_ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: in_buf.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: out_buf.buffer().as_entire_binding(),
                        },
                    ],
                });

            let mut encoder = gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(kernel.pipeline());
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(round_up_workgroups(count, workgroup_size), 1, 1);
            }
            gpu_ctx.queue.submit(Some(encoder.finish()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_multi_gpu() -> Option<MultiGpuContext> {
        pollster::block_on(MultiGpuContext::new()).ok()
    }

    #[test]
    fn test_enumerate_devices() {
        let Some(ctx) = try_multi_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        assert!(
            ctx.device_count() >= 1,
            "Should discover at least one GPU device"
        );
    }

    #[test]
    fn test_multi_gpu_context_creation() {
        let Some(ctx) = try_multi_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        for i in 0..ctx.device_count() {
            let dev = ctx.device(i);
            assert_eq!(dev.device_index, i);
            // Verify device is functional.
            let _ = dev.device.features();
        }
        // Primary should be the first device.
        assert_eq!(ctx.primary().device_index, 0);
    }

    #[test]
    fn test_multi_gpu_buffer_sync() {
        let Some(ctx) = try_multi_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut multi_buf = MultiGpuBuffer::<f32>::new(&ctx, 8);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        multi_buf.upload_to_all(&ctx, &data);

        // Verify every device has the same data.
        for i in 0..ctx.device_count() {
            let downloaded = multi_buf.download_from(&ctx, i);
            assert_eq!(downloaded, data, "Device {i} should have the uploaded data");
        }
    }

    #[test]
    fn test_work_distribution_even() {
        let Some(ctx) = try_multi_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let n = ctx.device_count() as u32;

        // Test even split with 100 items.
        let dist = ctx.distribute_work(100, &WorkDistribution::EvenSplit);
        let total: u32 = dist.iter().map(|(_, _, c)| *c).sum();
        assert_eq!(total, 100, "All items should be accounted for");

        // Check that offsets are contiguous.
        let mut expected_offset = 0u32;
        for &(_, offset, count) in &dist {
            assert_eq!(offset, expected_offset);
            expected_offset += count;
        }

        // Test with 0 items.
        let dist = ctx.distribute_work(0, &WorkDistribution::EvenSplit);
        assert!(dist.is_empty());

        // Test single device fallback.
        let dist = ctx.distribute_work(50, &WorkDistribution::SingleDevice(0));
        assert_eq!(dist.len(), 1);
        assert_eq!(dist[0], (0, 0, 50));

        // Test range-based.
        let ranges = vec![0..25, 25..50];
        let dist = ctx.distribute_work(50, &WorkDistribution::RangeBased { ranges });
        assert_eq!(dist.len(), 2);
        assert_eq!(dist[0], (0, 0, 25));
        assert_eq!(dist[1], (1, 25, 25));

        // Verify even split distributes remainder correctly.
        if n > 1 {
            let dist = ctx.distribute_work(n + 1, &WorkDistribution::EvenSplit);
            // First device should get one extra item.
            assert_eq!(dist[0].2, 2);
            for &(_, _, count) in &dist[1..] {
                assert_eq!(count, 1);
            }
        }
    }

    #[test]
    fn test_parallel_compute() {
        let Some(ctx) = try_multi_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };

        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&input) {
        output[id.x] = input[id.x] * 3.0;
    }
}
"#;

        let n: usize = 128;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let mut input_buf = MultiGpuBuffer::<f32>::new(&ctx, n);
        let mut output_buf = MultiGpuBuffer::<f32>::new(&ctx, n);

        // Upload input data to all devices.
        input_buf.upload_to_all(&ctx, &input_data);

        // Set output buffer lengths so download works correctly.
        for i in 0..ctx.device_count() {
            output_buf.buffer_on_mut(i).set_len(n as u32);
        }

        let work_dist = ctx.distribute_work(n as u32, &WorkDistribution::EvenSplit);

        GpuDevicePool::dispatch_parallel(
            &ctx,
            wgsl,
            "main",
            &input_buf,
            &mut output_buf,
            &work_dist,
            64,
        );

        let results = output_buf.gather_results(&ctx, &work_dist);

        let expected: Vec<f32> = (0..n).map(|i| i as f32 * 3.0).collect();
        assert_eq!(results, expected, "Parallel compute results should match");
    }
}
