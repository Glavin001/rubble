//! GPU-accelerated radix sort for u32 key-value pairs.

use rubble_gpu::{ComputeKernel, GpuBuffer, GpuContext};

const WORKGROUP_SIZE: u32 = 256;
const RADIX_BITS: u32 = 4;
const RADIX_BUCKETS: u32 = 1 << RADIX_BITS; // 16
const NUM_PASSES: u32 = 32 / RADIX_BITS; // 8

/// Per-block histogram: each workgroup counts occurrences of each radix digit
/// in its block, writing to histograms[bucket * num_blocks + workgroup_id].
const HISTOGRAM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x=count, y=shift

var<workgroup> local_hist: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let local = lid.x;
    let global = gid.x;
    let n = params.x;
    let shift = params.y;

    // Clear local histogram
    if local < 16u {
        atomicStore(&local_hist[local], 0u);
    }
    workgroupBarrier();

    // Count
    if global < n {
        let digit = (keys[global] >> shift) & 0xFu;
        atomicAdd(&local_hist[digit], 1u);
    }
    workgroupBarrier();

    // Write to global histograms: histograms[bucket * num_blocks + wid.x]
    let num_blocks = (n + 255u) / 256u;
    if local < 16u {
        histograms[local * num_blocks + wid.x] = atomicLoad(&local_hist[local]);
    }
}
"#;

/// Scatter keys and values to their sorted positions using scanned histograms.
const SCATTER_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x=count, y=shift

var<workgroup> local_offsets: array<atomic<u32>, 16>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let local = lid.x;
    let global = gid.x;
    let n = params.x;
    let shift = params.y;
    let num_blocks = (n + 255u) / 256u;

    // Load the prefix-scanned offset for this workgroup's bucket
    if local < 16u {
        atomicStore(&local_offsets[local], offsets[local * num_blocks + wid.x]);
    }
    workgroupBarrier();

    if global < n {
        let key = keys_in[global];
        let val = values_in[global];
        let digit = (key >> shift) & 0xFu;
        let pos = atomicAdd(&local_offsets[digit], 1u);
        keys_out[pos] = key;
        values_out[pos] = val;
    }
}
"#;

/// GPU-accelerated radix sort for u32 key-value pairs.
pub struct RadixSort {
    histogram_kernel: ComputeKernel,
    scatter_kernel: ComputeKernel,
    prefix_scan: crate::PrefixScan,
}

impl RadixSort {
    pub fn new(ctx: &GpuContext) -> Self {
        let histogram_kernel = ComputeKernel::from_wgsl(ctx, HISTOGRAM_WGSL, "histogram");
        let scatter_kernel = ComputeKernel::from_wgsl(ctx, SCATTER_WGSL, "scatter");
        let prefix_scan = crate::PrefixScan::new(ctx);
        Self {
            histogram_kernel,
            scatter_kernel,
            prefix_scan,
        }
    }

    /// Sort key-value pairs by key (ascending). Keys in `keys` buffer, values in `values` buffer.
    pub fn sort(&self, ctx: &GpuContext, keys: &mut GpuBuffer<u32>, values: &mut GpuBuffer<u32>) {
        let n = keys.len();
        if n <= 1 {
            return;
        }

        let num_blocks = rubble_gpu::round_up_workgroups(n, WORKGROUP_SIZE);
        let hist_size = (RADIX_BUCKETS * num_blocks) as usize;

        // Temp buffers for ping-pong
        let mut keys_tmp = GpuBuffer::<u32>::new(ctx, n as usize);
        keys_tmp.upload(ctx, &vec![0u32; n as usize]);
        let mut values_tmp = GpuBuffer::<u32>::new(ctx, n as usize);
        values_tmp.upload(ctx, &vec![0u32; n as usize]);

        // Histogram buffer (RADIX_BUCKETS * num_blocks)
        let mut histograms = GpuBuffer::<u32>::new(ctx, hist_size);

        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("radix sort params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for pass in 0..NUM_PASSES {
            let shift = pass * RADIX_BITS;

            // Determine which buffers are source and destination
            let (src_keys, src_values, dst_keys, dst_values) = if pass % 2 == 0 {
                (
                    keys as &GpuBuffer<u32>,
                    values as &GpuBuffer<u32>,
                    &mut keys_tmp,
                    &mut values_tmp,
                )
            } else {
                (
                    &keys_tmp as &GpuBuffer<u32>,
                    &values_tmp as &GpuBuffer<u32>,
                    keys as &mut GpuBuffer<u32>,
                    values as &mut GpuBuffer<u32>,
                )
            };

            // Clear histograms
            histograms.upload(ctx, &vec![0u32; hist_size]);

            // Update params
            ctx.queue
                .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[n, shift, 0u32, 0]));

            // Pass 1: Compute histograms
            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.histogram_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src_keys.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: histograms.buffer().as_entire_binding(),
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
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(self.histogram_kernel.pipeline());
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(num_blocks, 1, 1);
                }
                ctx.queue.submit(Some(encoder.finish()));
            }

            // Pass 2: Exclusive prefix scan on histograms (flattened)
            // The histograms layout is: [bucket0_block0, bucket0_block1, ..., bucket1_block0, ...]
            // A prefix scan over this gives the global write offset for each (bucket, block) pair.
            self.prefix_scan.exclusive_scan(ctx, &histograms);

            // Pass 3: Scatter
            {
                let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.scatter_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src_keys.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: src_values.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: dst_keys.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: dst_values.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: histograms.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params_buf.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(self.scatter_kernel.pipeline());
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(num_blocks, 1, 1);
                }
                ctx.queue.submit(Some(encoder.finish()));
            }
        }

        // After NUM_PASSES passes with ping-pong, the final result location depends
        // on the last pass index (NUM_PASSES - 1). Even passes write to tmp, odd
        // passes write to keys/values. NUM_PASSES=8, last pass=7 (odd), so result
        // is already in keys/values. If NUM_PASSES were odd, we'd need to copy back.
        if NUM_PASSES % 2 == 1 {
            // Last pass was even, result is in tmp -- copy back.
            let byte_len = (n as usize * std::mem::size_of::<u32>()) as u64;
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(keys_tmp.buffer(), 0, keys.buffer(), 0, byte_len);
            encoder.copy_buffer_to_buffer(values_tmp.buffer(), 0, values.buffer(), 0, byte_len);
            ctx.queue.submit(Some(encoder.finish()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_sort_simple() {
        let ctx = crate::test_gpu();
        let sort = RadixSort::new(&ctx);

        let input_keys = [5u32, 3, 8, 1, 4];
        let input_values = [50u32, 30, 80, 10, 40];

        let mut keys = GpuBuffer::<u32>::new(&ctx, input_keys.len());
        keys.upload(&ctx, &input_keys);
        let mut values = GpuBuffer::<u32>::new(&ctx, input_values.len());
        values.upload(&ctx, &input_values);

        sort.sort(&ctx, &mut keys, &mut values);

        let result_keys = keys.download(&ctx);
        let result_values = values.download(&ctx);

        assert_eq!(result_keys, vec![1, 3, 4, 5, 8]);
        assert_eq!(result_values, vec![10, 30, 40, 50, 80]);
    }

    #[test]
    fn test_radix_sort_random() {
        let ctx = crate::test_gpu();
        let sort = RadixSort::new(&ctx);

        // Simple LCG pseudo-random
        let mut rng_state = 12345u64;
        let mut rand_vals: Vec<u32> = Vec::with_capacity(1000);
        for _ in 0..1000 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rand_vals.push((rng_state >> 33) as u32);
        }

        let values_data: Vec<u32> = (0..1000).collect();

        let mut keys = GpuBuffer::<u32>::new(&ctx, 1000);
        keys.upload(&ctx, &rand_vals);
        let mut values = GpuBuffer::<u32>::new(&ctx, 1000);
        values.upload(&ctx, &values_data);

        sort.sort(&ctx, &mut keys, &mut values);

        let result_keys = keys.download(&ctx);

        // Verify sorted
        for i in 1..result_keys.len() {
            assert!(
                result_keys[i - 1] <= result_keys[i],
                "Not sorted at index {}: {} > {}",
                i,
                result_keys[i - 1],
                result_keys[i]
            );
        }
    }

    #[test]
    fn test_radix_sort_values_follow_keys() {
        let ctx = crate::test_gpu();
        let sort = RadixSort::new(&ctx);

        // Simple LCG pseudo-random
        let mut rng_state = 99999u64;
        let mut rand_vals: Vec<u32> = Vec::with_capacity(500);
        for _ in 0..500 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rand_vals.push((rng_state >> 33) as u32);
        }

        // Values encode the original index so we can verify pairing.
        let values_data: Vec<u32> = (0..500).collect();

        // Build a lookup from key -> original index (handle duplicate keys).
        let original_pairs: Vec<(u32, u32)> = rand_vals
            .iter()
            .copied()
            .zip(values_data.iter().copied())
            .collect();

        let mut keys = GpuBuffer::<u32>::new(&ctx, 500);
        keys.upload(&ctx, &rand_vals);
        let mut values = GpuBuffer::<u32>::new(&ctx, 500);
        values.upload(&ctx, &values_data);

        sort.sort(&ctx, &mut keys, &mut values);

        let result_keys = keys.download(&ctx);
        let result_values = values.download(&ctx);

        // Verify each (key, value) pair in the output matches an original pair.
        // The value encodes the original index, so result_keys[i] must equal
        // the original key at that original index.
        for i in 0..result_keys.len() {
            let sorted_key = result_keys[i];
            let original_index = result_values[i] as usize;
            assert!(
                original_index < original_pairs.len(),
                "Value {} at sorted position {} is out of range",
                original_index,
                i
            );
            assert_eq!(
                sorted_key, original_pairs[original_index].0,
                "At sorted position {}: key {} doesn't match original key {} for original index {}",
                i, sorted_key, original_pairs[original_index].0, original_index
            );
        }
    }
}
