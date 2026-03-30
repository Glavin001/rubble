//! GPU parallel exclusive prefix scan.

use rubble_gpu::{ComputeKernel, GpuBuffer, GpuContext};

const WORKGROUP_SIZE: u32 = 256;

/// Blelloch exclusive scan within a single workgroup of WORKGROUP_SIZE elements.
/// Each workgroup writes its total sum to `block_sums[workgroup_id]`.
const SCAN_BLOCKS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = element count

var<workgroup> sdata: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local = lid.x;
    let global = wid.x * 256u + local;
    let n = params.x;

    // Load into sdata memory
    if global < n {
        sdata[local] = data[global];
    } else {
        sdata[local] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = 256u >> 1u; d > 0u; d >>= 1u) {
        if local < d {
            let ai = offset * (2u * local + 1u) - 1u;
            let bi = offset * (2u * local + 2u) - 1u;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1u;
        workgroupBarrier();
    }

    // Save block sum and clear last element
    if local == 0u {
        block_sums[wid.x] = sdata[255];
        sdata[255] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
    for (var d = 1u; d < 256u; d <<= 1u) {
        offset >>= 1u;
        if local < d {
            let ai = offset * (2u * local + 1u) - 1u;
            let bi = offset * (2u * local + 2u) - 1u;
            let t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
        workgroupBarrier();
    }

    // Write back
    if global < n {
        data[global] = sdata[local];
    }
}
"#;

/// Add block sums back: data[i] += block_sums[workgroup_id_of_i]
const ADD_BLOCK_SUMS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> block_sums: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = element count

@compute @workgroup_size(256)
fn add_block_sums(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let global = gid.x;
    let n = params.x;
    if global < n {
        data[global] += block_sums[wid.x];
    }
}
"#;

/// GPU-accelerated exclusive prefix scan.
pub struct PrefixScan {
    scan_blocks_kernel: ComputeKernel,
    add_block_sums_kernel: ComputeKernel,
}

impl PrefixScan {
    pub fn new(ctx: &GpuContext) -> Self {
        let scan_blocks_kernel = ComputeKernel::from_wgsl(ctx, SCAN_BLOCKS_WGSL, "scan_blocks");
        let add_block_sums_kernel =
            ComputeKernel::from_wgsl(ctx, ADD_BLOCK_SUMS_WGSL, "add_block_sums");
        Self {
            scan_blocks_kernel,
            add_block_sums_kernel,
        }
    }

    /// Exclusive prefix scan in-place. After this call, `data[i] = sum(data[0..i])`.
    ///
    /// Supports up to `WORKGROUP_SIZE * WORKGROUP_SIZE` (65536) elements via a
    /// three-pass approach. For larger inputs, falls back to CPU.
    pub fn exclusive_scan(&self, ctx: &GpuContext, data: &GpuBuffer<u32>) {
        let n = data.len();
        if n == 0 {
            return;
        }

        let max_gpu = WORKGROUP_SIZE * WORKGROUP_SIZE; // 65536
        if n > max_gpu {
            // Arrays > 65536 use a CPU round-trip. This is acceptable because
            // the broadphase pair count in practice stays well below this limit
            // (e.g. 256 bodies produce at most ~32k pairs).
            self.cpu_fallback(ctx, data);
            return;
        }

        let num_blocks = rubble_gpu::round_up_workgroups(n, WORKGROUP_SIZE);

        // Create block sums buffer (pad to WORKGROUP_SIZE for the single-block scan)
        let mut block_sums = GpuBuffer::<u32>::new(ctx, WORKGROUP_SIZE as usize);
        block_sums.upload(ctx, &vec![0u32; WORKGROUP_SIZE as usize]);

        // Params uniform
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[n, 0u32, 0, 0]));

        // Pass 1: scan each block, collect block sums
        {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.scan_blocks_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: block_sums.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.scan_blocks_kernel.pipeline());
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_blocks, 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }

        if num_blocks > 1 {
            // Pass 2: scan block sums (single workgroup since num_blocks <= 256)
            let block_sums_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block sums scan params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(
                &block_sums_params_buf,
                0,
                bytemuck::cast_slice(&[num_blocks, 0u32, 0, 0]),
            );

            // We need a dummy block_sums_of_block_sums buffer
            let mut dummy_block_sums = GpuBuffer::<u32>::new(ctx, WORKGROUP_SIZE as usize);
            dummy_block_sums.upload(ctx, &vec![0u32; WORKGROUP_SIZE as usize]);

            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.scan_blocks_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: block_sums.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dummy_block_sums.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: block_sums_params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(self.scan_blocks_kernel.pipeline());
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                ctx.queue.submit(Some(encoder.finish()));
            }

            // Pass 3: add scanned block sums back
            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.add_block_sums_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: data.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: block_sums.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(self.add_block_sums_kernel.pipeline());
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(num_blocks, 1, 1);
                }
                ctx.queue.submit(Some(encoder.finish()));
            }
        }
    }

    /// CPU fallback for arrays larger than 65536 elements.
    /// Used only when pair counts exceed the two-level GPU scan capacity.
    fn cpu_fallback(&self, ctx: &GpuContext, data: &GpuBuffer<u32>) {
        let mut host = data.download(ctx);
        let mut sum = 0u32;
        for v in host.iter_mut() {
            let old = *v;
            *v = sum;
            sum = sum.wrapping_add(old);
        }
        // Re-upload: we need mutable access but only have &GpuBuffer.
        // Use raw write_buffer to write back in-place.
        ctx.queue
            .write_buffer(data.buffer(), 0, bytemuck::cast_slice(&host));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_scan_ones() {
        let ctx = crate::test_gpu();
        let scan = PrefixScan::new(&ctx);

        let input: Vec<u32> = vec![1; 256];
        let mut buf = GpuBuffer::<u32>::new(&ctx, 256);
        buf.upload(&ctx, &input);

        scan.exclusive_scan(&ctx, &buf);

        let result = buf.download(&ctx);
        let expected: Vec<u32> = (0..256).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_prefix_scan_mixed() {
        let ctx = crate::test_gpu();
        let scan = PrefixScan::new(&ctx);

        let input = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut buf = GpuBuffer::<u32>::new(&ctx, input.len());
        buf.upload(&ctx, &input);

        scan.exclusive_scan(&ctx, &buf);

        let result = buf.download(&ctx);
        assert_eq!(result, vec![0, 3, 4, 8, 9, 14, 23, 25]);
    }
}
