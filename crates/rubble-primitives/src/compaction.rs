//! GPU-accelerated stream compaction using prefix scan.

use rubble_gpu::{ComputeKernel, GpuBuffer, GpuContext};

const WORKGROUP_SIZE: u32 = 256;

/// Scatter elements where predicate == 1 to their compacted position.
const SCATTER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> predicates: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>; // x = count

@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let global = gid.x;
    let n = params.x;
    if global < n {
        if predicates[global] == 1u {
            output[offsets[global]] = data[global];
        }
    }
}
"#;

/// GPU-accelerated stream compaction.
///
/// Given a data buffer and a predicate buffer (0/1 per element), produces a
/// compacted output containing only elements where predicate == 1.
pub struct GpuStreamCompaction {
    scan: crate::InternalPrefixScan,
    scatter_kernel: ComputeKernel,
    max_elements: usize,
}

impl GpuStreamCompaction {
    /// Create a new stream compaction instance supporting up to `max_elements`.
    pub fn new(ctx: &GpuContext, max_elements: usize) -> Self {
        let scan = crate::InternalPrefixScan::new(ctx);
        let scatter_kernel = ComputeKernel::from_wgsl(ctx, SCATTER_WGSL, "scatter");
        Self {
            scan,
            scatter_kernel,
            max_elements,
        }
    }

    /// Maximum number of elements this instance supports.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Compact `data` using `predicates` (0 or 1 per element).
    ///
    /// Returns `(output_buffer, count)` where `output_buffer` contains only the
    /// elements where predicate == 1, and `count` is the number of surviving elements.
    pub fn compact(
        &self,
        ctx: &GpuContext,
        data: &GpuBuffer<u32>,
        predicates: &GpuBuffer<u32>,
    ) -> (GpuBuffer<u32>, u32) {
        let n = data.len();
        if n == 0 {
            let empty = GpuBuffer::<u32>::new(ctx, 1);
            return (empty, 0);
        }

        // Copy predicates to an offsets buffer (scan will modify in-place)
        let pred_data = predicates.download(ctx);
        let mut offsets = GpuBuffer::<u32>::new(ctx, n as usize);
        offsets.upload(ctx, &pred_data);

        // Exclusive prefix scan on offsets
        self.scan.exclusive_scan(ctx, &offsets);

        // Compute total count: offsets[n-1] + predicates[n-1]
        let offsets_data = offsets.download(ctx);
        let count = offsets_data[n as usize - 1] + pred_data[n as usize - 1];

        if count == 0 {
            let empty = GpuBuffer::<u32>::new(ctx, 1);
            return (empty, 0);
        }

        // Create output buffer
        let mut output = GpuBuffer::<u32>::new(ctx, count as usize);
        output.upload(ctx, &vec![0u32; count as usize]);

        // Params uniform
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compact params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[n, 0u32, 0, 0]));

        // Scatter
        {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.scatter_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: predicates.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: offsets.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            let num_workgroups = rubble_gpu::round_up_workgroups(n, WORKGROUP_SIZE);
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.scatter_kernel.pipeline());
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }

        (output, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let compact = GpuStreamCompaction::new(&ctx, 256);

        let data_in = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let preds = [1u32, 0, 1, 0, 1, 1, 0, 1];

        let mut data = GpuBuffer::<u32>::new(&ctx, data_in.len());
        data.upload(&ctx, &data_in);
        let mut predicates = GpuBuffer::<u32>::new(&ctx, preds.len());
        predicates.upload(&ctx, &preds);

        let (output, count) = compact.compact(&ctx, &data, &predicates);

        assert_eq!(count, 5);
        let result = output.download(&ctx);
        assert_eq!(result, vec![10, 30, 50, 60, 80]);
    }

    #[test]
    fn test_compaction_all_zeros() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let compact = GpuStreamCompaction::new(&ctx, 256);

        let data_in = [10u32, 20, 30, 40];
        let preds = [0u32, 0, 0, 0];

        let mut data = GpuBuffer::<u32>::new(&ctx, data_in.len());
        data.upload(&ctx, &data_in);
        let mut predicates = GpuBuffer::<u32>::new(&ctx, preds.len());
        predicates.upload(&ctx, &preds);

        let (_output, count) = compact.compact(&ctx, &data, &predicates);
        assert_eq!(count, 0, "All-zero predicates should yield count=0");
    }

    #[test]
    fn test_compaction_all_ones() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let compact = GpuStreamCompaction::new(&ctx, 256);

        let data_in = [10u32, 20, 30, 40, 50];
        let preds = [1u32, 1, 1, 1, 1];

        let mut data = GpuBuffer::<u32>::new(&ctx, data_in.len());
        data.upload(&ctx, &data_in);
        let mut predicates = GpuBuffer::<u32>::new(&ctx, preds.len());
        predicates.upload(&ctx, &preds);

        let (output, count) = compact.compact(&ctx, &data, &predicates);
        assert_eq!(count, 5, "All-one predicates should yield count=N");
        let result = output.download(&ctx);
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }
}
