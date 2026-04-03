//! GPU-accelerated radix sort for 32-bit key-value pairs.

use rubble_gpu::{ComputeKernel, GpuBuffer, GpuContext};

const WORKGROUP_SIZE: u32 = 256;
const RADIX_BITS: u32 = 4;
const RADIX_BUCKETS: u32 = 1 << RADIX_BITS; // 16
const NUM_PASSES: u32 = 32 / RADIX_BITS; // 8

/// A key-value pair for radix sort. Key is used for ordering, value is carried along.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RadixSortEntry {
    pub key: u32,
    pub value: u32,
}

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

var<workgroup> local_digits: array<u32, 256>;

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

    if global < n {
        let key = keys_in[global];
        local_digits[local] = (key >> shift) & 0xFu;
    } else {
        local_digits[local] = 0xFFFFFFFFu;
    }
    workgroupBarrier();

    if global < n {
        let key = keys_in[global];
        let val = values_in[global];
        let digit = local_digits[local];

        // Compute a stable rank for this key within the current workgroup by
        // counting earlier invocations with the same radix digit.
        var local_rank = 0u;
        for (var i = 0u; i < local; i = i + 1u) {
            if local_digits[i] == digit {
                local_rank = local_rank + 1u;
            }
        }

        let pos = offsets[digit * num_blocks + wid.x] + local_rank;
        keys_out[pos] = key;
        values_out[pos] = val;
    }
}
"#;

/// GPU-accelerated radix sort for 32-bit key-value pairs.
///
/// Sorts [`RadixSortEntry`] elements by key in ascending order using a 4-bit-per-pass
/// radix sort (8 passes total for 32-bit keys).
pub struct GpuRadixSort {
    histogram_kernel: ComputeKernel,
    scatter_kernel: ComputeKernel,
    prefix_scan: crate::InternalPrefixScan,
    max_elements: usize,
}

impl GpuRadixSort {
    /// Create a new radix sort instance supporting up to `max_elements` entries.
    pub fn new(ctx: &GpuContext, max_elements: usize) -> Self {
        let histogram_kernel = ComputeKernel::from_wgsl(ctx, HISTOGRAM_WGSL, "histogram");
        let scatter_kernel = ComputeKernel::from_wgsl(ctx, SCATTER_WGSL, "scatter");
        let prefix_scan = crate::InternalPrefixScan::new(ctx);
        Self {
            histogram_kernel,
            scatter_kernel,
            prefix_scan,
            max_elements,
        }
    }

    /// Maximum number of elements this instance supports.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Sort entries in-place by key (ascending).
    pub fn sort(&self, ctx: &GpuContext, entries: &mut GpuBuffer<RadixSortEntry>) {
        let n = entries.len();
        if n <= 1 {
            return;
        }

        // Download entries and split into separate key/value arrays for the shader.
        let host_entries = entries.download(ctx);
        let keys_data: Vec<u32> = host_entries.iter().map(|e| e.key).collect();
        let values_data: Vec<u32> = host_entries.iter().map(|e| e.value).collect();

        let mut keys = GpuBuffer::<u32>::new(ctx, n as usize);
        keys.upload(ctx, &keys_data);
        let mut values = GpuBuffer::<u32>::new(ctx, n as usize);
        values.upload(ctx, &values_data);

        self.sort_key_value_in_place(ctx, &mut keys, &mut values);

        // Recombine into entries buffer
        let sorted_keys = keys.download(ctx);
        let sorted_values = values.download(ctx);
        let sorted_entries: Vec<RadixSortEntry> = sorted_keys
            .iter()
            .zip(sorted_values.iter())
            .map(|(&k, &v)| RadixSortEntry { key: k, value: v })
            .collect();
        entries.upload(ctx, &sorted_entries);
    }

    /// Internal sort on separate key/value buffers.
    pub fn sort_key_value_in_place(
        &self,
        ctx: &GpuContext,
        keys: &mut GpuBuffer<u32>,
        values: &mut GpuBuffer<u32>,
    ) {
        let n = keys.len();
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

        // After NUM_PASSES (8) passes with ping-pong, last pass index = 7 (odd),
        // so the result is already in keys/values. If NUM_PASSES were odd, we'd copy back.
        if NUM_PASSES % 2 == 1 {
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
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let sort = GpuRadixSort::new(&ctx, 1024);

        let input = [
            RadixSortEntry { key: 5, value: 50 },
            RadixSortEntry { key: 3, value: 30 },
            RadixSortEntry { key: 8, value: 80 },
            RadixSortEntry { key: 1, value: 10 },
            RadixSortEntry { key: 4, value: 40 },
        ];

        let mut buf = GpuBuffer::<RadixSortEntry>::new(&ctx, input.len());
        buf.upload(&ctx, &input);

        sort.sort(&ctx, &mut buf);

        let result = buf.download(&ctx);
        let result_keys: Vec<u32> = result.iter().map(|e| e.key).collect();
        let result_values: Vec<u32> = result.iter().map(|e| e.value).collect();

        assert_eq!(result_keys, vec![1, 3, 4, 5, 8]);
        assert_eq!(result_values, vec![10, 30, 40, 50, 80]);
    }

    #[test]
    fn test_radix_sort_random() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let sort = GpuRadixSort::new(&ctx, 1024);

        // Simple LCG pseudo-random
        let mut rng_state = 12345u64;
        let mut entries: Vec<RadixSortEntry> = Vec::with_capacity(1000);
        for i in 0..1000u32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            entries.push(RadixSortEntry {
                key: (rng_state >> 33) as u32,
                value: i,
            });
        }

        let mut buf = GpuBuffer::<RadixSortEntry>::new(&ctx, entries.len());
        buf.upload(&ctx, &entries);

        sort.sort(&ctx, &mut buf);

        let result = buf.download(&ctx);

        // Verify sorted ascending by key
        for i in 1..result.len() {
            assert!(
                result[i - 1].key <= result[i].key,
                "Not sorted at index {}: {} > {}",
                i,
                result[i - 1].key,
                result[i].key
            );
        }
    }

    #[test]
    fn test_radix_sort_values_follow_keys() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let sort = GpuRadixSort::new(&ctx, 1024);

        // Simple LCG pseudo-random
        let mut rng_state = 99999u64;
        let mut entries: Vec<RadixSortEntry> = Vec::with_capacity(500);
        for i in 0..500u32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            entries.push(RadixSortEntry {
                key: (rng_state >> 33) as u32,
                value: i,
            });
        }

        let original_entries = entries.clone();

        let mut buf = GpuBuffer::<RadixSortEntry>::new(&ctx, entries.len());
        buf.upload(&ctx, &entries);

        sort.sort(&ctx, &mut buf);

        let result = buf.download(&ctx);

        // Verify each (key, value) pair: value encodes the original index,
        // so result[i].key must match original_entries[result[i].value].key
        for (i, entry) in result.iter().enumerate() {
            let original_index = entry.value as usize;
            assert!(
                original_index < original_entries.len(),
                "Value {} at sorted position {} is out of range",
                original_index,
                i
            );
            assert_eq!(
                entry.key, original_entries[original_index].key,
                "At sorted position {}: key {} doesn't match original key {} for original index {}",
                i, entry.key, original_entries[original_index].key, original_index
            );
        }
    }
}
