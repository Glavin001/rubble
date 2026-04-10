//! GPU parallel exclusive and inclusive prefix scan.

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

    // Load into shared memory
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

/// Convert exclusive scan result to inclusive: inclusive[i] = exclusive[i] + original[i].
const EXCLUSIVE_TO_INCLUSIVE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> original: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = element count

@compute @workgroup_size(256)
fn exclusive_to_inclusive(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let global = gid.x;
    let n = params.x;
    if global < n {
        data[global] = data[global] + original[global];
    }
}
"#;

/// Internal prefix scan implementation shared by public API and other modules in this crate.
pub(crate) struct InternalPrefixScan {
    scan_blocks_kernel: ComputeKernel,
    add_block_sums_kernel: ComputeKernel,
    params_buf: wgpu::Buffer,
}

impl InternalPrefixScan {
    pub fn new(ctx: &GpuContext) -> Self {
        let scan_blocks_kernel = ComputeKernel::from_wgsl(ctx, SCAN_BLOCKS_WGSL, "scan_blocks");
        let add_block_sums_kernel =
            ComputeKernel::from_wgsl(ctx, ADD_BLOCK_SUMS_WGSL, "add_block_sums");
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scan params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            scan_blocks_kernel,
            add_block_sums_kernel,
            params_buf,
        }
    }

    /// Exclusive prefix scan in-place. After this call, `data[i] = sum(data[0..i])`.
    pub fn exclusive_scan(&self, ctx: &GpuContext, data: &GpuBuffer<u32>) {
        let n = data.len();
        if n == 0 {
            return;
        }

        let num_blocks = rubble_gpu::round_up_workgroups(n, WORKGROUP_SIZE);
        let block_sums_len = num_blocks.max(1) as usize;
        let mut block_sums = GpuBuffer::<u32>::new(ctx, block_sums_len);
        block_sums.set_len(num_blocks);

        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[n, 0u32, 0, 0]));

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
                        resource: self.params_buf.as_entire_binding()
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
            self.exclusive_scan(ctx, &block_sums);
            ctx.queue
                .write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&[n, 0u32, 0, 0]));

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
                            resource: self.params_buf.as_entire_binding()
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
}

/// GPU-accelerated exclusive and inclusive prefix scan.
pub struct GpuPrefixScan {
    internal: InternalPrefixScan,
    inclusive_kernel: ComputeKernel,
    inclusive_params_buf: wgpu::Buffer,
    max_elements: usize,
}

impl GpuPrefixScan {
    /// Create a new prefix scan instance supporting up to `max_elements`.
    pub fn new(ctx: &GpuContext, max_elements: usize) -> Self {
        let internal = InternalPrefixScan::new(ctx);
        let inclusive_kernel =
            ComputeKernel::from_wgsl(ctx, EXCLUSIVE_TO_INCLUSIVE_WGSL, "exclusive_to_inclusive");
        let inclusive_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inclusive scan params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            internal,
            inclusive_kernel,
            inclusive_params_buf,
            max_elements,
        }
    }

    /// Maximum number of elements this instance supports.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Exclusive prefix scan in-place. After this call, `data[i] = sum(data[0..i])`.
    pub fn exclusive_scan(&self, ctx: &GpuContext, data: &mut GpuBuffer<u32>) {
        self.internal.exclusive_scan(ctx, data);
    }

    /// Inclusive prefix scan in-place. After this call, `data[i] = sum(data[0..=i])`.
    pub fn inclusive_scan(&self, ctx: &GpuContext, data: &mut GpuBuffer<u32>) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // Save original data
        let original = data.download(ctx);
        let mut original_buf = GpuBuffer::<u32>::new(ctx, original.len());
        original_buf.upload(ctx, &original);

        // Run exclusive scan
        self.internal.exclusive_scan(ctx, data);

        // Convert: inclusive[i] = exclusive[i] + original[i]
        ctx.queue.write_buffer(
            &self.inclusive_params_buf,
            0,
            bytemuck::cast_slice(&[n, 0u32, 0, 0]),
        );

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.inclusive_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: original_buf.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.inclusive_params_buf.as_entire_binding()
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
            pass.set_pipeline(self.inclusive_kernel.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_scan_ones() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let scan = GpuPrefixScan::new(&ctx, 256);

        let input: Vec<u32> = vec![1; 256];
        let mut buf = GpuBuffer::<u32>::new(&ctx, 256);
        buf.upload(&ctx, &input);

        scan.exclusive_scan(&ctx, &mut buf);

        let result = buf.download(&ctx);
        let expected: Vec<u32> = (0..256).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_prefix_scan_mixed() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let scan = GpuPrefixScan::new(&ctx, 256);

        let input = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut buf = GpuBuffer::<u32>::new(&ctx, input.len());
        buf.upload(&ctx, &input);

        scan.exclusive_scan(&ctx, &mut buf);

        let result = buf.download(&ctx);
        assert_eq!(result, vec![0, 3, 4, 8, 9, 14, 23, 25]);
    }

    #[test]
    fn test_inclusive_scan_ones() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let scan = GpuPrefixScan::new(&ctx, 256);

        let input: Vec<u32> = vec![1; 256];
        let mut buf = GpuBuffer::<u32>::new(&ctx, 256);
        buf.upload(&ctx, &input);

        scan.inclusive_scan(&ctx, &mut buf);

        let result = buf.download(&ctx);
        let expected: Vec<u32> = (1..=256).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inclusive_scan_mixed() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let scan = GpuPrefixScan::new(&ctx, 256);

        let input = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let mut buf = GpuBuffer::<u32>::new(&ctx, input.len());
        buf.upload(&ctx, &input);

        scan.inclusive_scan(&ctx, &mut buf);

        let result = buf.download(&ctx);
        // inclusive: [3, 4, 8, 9, 14, 23, 25, 31]
        assert_eq!(result, vec![3, 4, 8, 9, 14, 23, 25, 31]);
    }

    #[test]
    fn test_exclusive_scan_large_n_stays_on_gpu() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let n = 70_000usize;
        let scan = GpuPrefixScan::new(&ctx, n);
        let input: Vec<u32> = vec![1; n];
        let mut buf = GpuBuffer::<u32>::new(&ctx, n);
        buf.upload(&ctx, &input);

        scan.exclusive_scan(&ctx, &mut buf);

        let result = buf.download(&ctx);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[n - 1], (n - 1) as u32);
    }
}
