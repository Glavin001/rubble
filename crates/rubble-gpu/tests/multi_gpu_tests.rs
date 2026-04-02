//! Integration tests for multi-GPU support in rubble-gpu.
//!
//! These tests verify device enumeration, work distribution, buffer
//! synchronization, and parallel compute dispatch across multiple GPUs.

use rubble_gpu::{GpuBuffer, GpuContext, MultiGpuBuffer, MultiGpuContext, WorkDistribution};

fn try_multi_gpu() -> Option<MultiGpuContext> {
    pollster::block_on(MultiGpuContext::new()).ok()
}

fn _try_single_gpu() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
}

// ---------------------------------------------------------------------------
// Device enumeration
// ---------------------------------------------------------------------------

#[test]
fn enumerate_at_least_one_device() {
    let infos = pollster::block_on(GpuContext::enumerate_adapters());
    // In CI with lavapipe, there should be at least one adapter
    // This test succeeds if GPU is available
    if infos.is_empty() {
        eprintln!("SKIP: No GPU adapters found");
        return;
    }
    assert!(!infos.is_empty());
    for info in &infos {
        // Verify adapter info is accessible (doesn't panic)
        let _name = &info.name;
        let _backend = info.backend;
    }
}

#[test]
fn multi_gpu_context_creation() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU adapters for multi-GPU");
            return;
        }
    };

    assert!(ctx.device_count() >= 1, "Should have at least one device");

    let primary = ctx.primary();
    assert!(primary.device_index == 0, "Primary should be device 0");
}

// ---------------------------------------------------------------------------
// Work distribution
// ---------------------------------------------------------------------------

#[test]
fn work_distribution_even_split_single_device() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let dist = ctx.distribute_work(100, &WorkDistribution::EvenSplit);
    // With 1+ devices, all items should be accounted for
    let total: u32 = dist.iter().map(|&(_, _, count)| count).sum();
    assert_eq!(total, 100, "All items should be distributed: total={total}");
}

#[test]
fn work_distribution_even_split_covers_all() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    for total_items in [1u32, 7, 64, 100, 255, 1000] {
        let dist = ctx.distribute_work(total_items, &WorkDistribution::EvenSplit);
        let total: u32 = dist.iter().map(|&(_, _, count)| count).sum();
        assert_eq!(
            total, total_items,
            "EvenSplit should cover all {total_items} items, got {total}"
        );

        // Ranges should be contiguous
        let mut expected_offset = 0u32;
        for &(_, offset, count) in &dist {
            assert_eq!(
                offset, expected_offset,
                "Ranges should be contiguous: expected offset {expected_offset}, got {offset}"
            );
            expected_offset += count;
        }
    }
}

#[test]
fn work_distribution_single_device() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let dist = ctx.distribute_work(50, &WorkDistribution::SingleDevice(0));
    assert_eq!(dist.len(), 1, "SingleDevice should have one range");
    assert_eq!(dist[0].0, 0, "Should use device 0");
    assert_eq!(dist[0].1, 0, "Offset should be 0");
    assert_eq!(dist[0].2, 50, "Count should be 50");
}

#[test]
fn work_distribution_range_based() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let ranges = vec![0..30, 30..100];
    let dist = ctx.distribute_work(100, &WorkDistribution::RangeBased { ranges });
    // Should have up to device_count entries
    let total: u32 = dist.iter().map(|&(_, _, count)| count).sum();
    assert_eq!(total, 100, "RangeBased should cover all 100 items");
}

// ---------------------------------------------------------------------------
// Buffer operations
// ---------------------------------------------------------------------------

#[test]
fn multi_gpu_buffer_upload_download() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let mut buf = MultiGpuBuffer::<f32>::new(&ctx, 64);
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 1.5).collect();

    buf.upload_to_all(&ctx, &data);

    // Download from each device and verify
    for dev_idx in 0..ctx.device_count() {
        let result = buf.download_from(&ctx, dev_idx);
        assert_eq!(
            result.len(),
            data.len(),
            "Device {dev_idx}: length mismatch"
        );
        for (i, (&expected, &actual)) in data.iter().zip(result.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-6,
                "Device {dev_idx}: mismatch at index {i}: expected={expected}, got={actual}"
            );
        }
    }
}

#[test]
fn multi_gpu_buffer_gather_results() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let mut buf = MultiGpuBuffer::<f32>::new(&ctx, 64);
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    buf.upload_to_all(&ctx, &data);

    // Gather results as if we split work across devices
    let dist = ctx.distribute_work(64, &WorkDistribution::EvenSplit);
    let gathered = buf.gather_results(&ctx, &dist);

    assert_eq!(gathered.len(), 64, "Gathered should have 64 elements");
    for (i, &val) in gathered.iter().enumerate() {
        assert!(
            (val - i as f32).abs() < 1e-6,
            "Gathered mismatch at {i}: expected={}, got={val}",
            i as f32
        );
    }
}

// ---------------------------------------------------------------------------
// Parallel compute
// ---------------------------------------------------------------------------

#[test]
fn parallel_compute_multiply() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    // Simple kernel: multiply each element by 2
    let wgsl = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) {
        data[id.x] = data[id.x] * 2.0;
    }
}
"#;

    // Run on primary device
    let primary_ctx = ctx.primary().as_context();
    let mut buf = GpuBuffer::<f32>::new(&primary_ctx, 256);
    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    buf.upload(&primary_ctx, &input);

    let kernel = rubble_gpu::ComputeKernel::from_wgsl(&primary_ctx, wgsl, "main");
    let bg = primary_ctx
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: kernel.bind_group_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.buffer().as_entire_binding(),
            }],
        });

    let mut encoder = primary_ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(kernel.pipeline());
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(rubble_gpu::round_up_workgroups(256, 64), 1, 1);
    }
    primary_ctx.queue.submit(Some(encoder.finish()));

    let result = buf.download(&primary_ctx);
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32 * 2.0;
        assert!(
            (val - expected).abs() < 1e-4,
            "Compute mismatch at {i}: expected={expected}, got={val}"
        );
    }
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

#[test]
fn device_info_populated() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    for i in 0..ctx.device_count() {
        let dev = ctx.device(i);
        // Just verify it doesn't crash to access info
        let _name = &dev.adapter_info.name;
        let _backend = dev.adapter_info.backend;
        let _features = dev.device.features();
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn distribute_zero_items() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let dist = ctx.distribute_work(0, &WorkDistribution::EvenSplit);
    let total: u32 = dist.iter().map(|&(_, _, count)| count).sum();
    assert_eq!(total, 0, "Zero items should distribute zero work");
}

#[test]
fn distribute_one_item() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let dist = ctx.distribute_work(1, &WorkDistribution::EvenSplit);
    let total: u32 = dist.iter().map(|&(_, _, count)| count).sum();
    assert_eq!(total, 1, "One item should be assigned to one device");
}

#[test]
fn multi_gpu_buffer_empty() {
    let ctx = match try_multi_gpu() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: No GPU");
            return;
        }
    };

    let buf = MultiGpuBuffer::<f32>::new(&ctx, 0);
    let result = buf.download_from(&ctx, 0);
    assert!(result.is_empty(), "Empty buffer should download empty");
}
