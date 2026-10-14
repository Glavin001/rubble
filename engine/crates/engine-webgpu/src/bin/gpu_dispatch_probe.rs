//! Native counterpart of dispatch-diagnostic.html: no Slang or physics code.
//! Run separately from browser benchmarks. Verifies every element after timing.
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok_or_else(|| anyhow::anyhow!("Hardware GPU adapter required for dispatch diagnostic"))?;
    eprintln!("adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("dispatch diagnostic"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: Default::default(),
        },
        None,
    ))?;
    let group_size: u32 = std::env::var("GPU_PROBE_WORKGROUP")
        .unwrap_or_else(|_| "64".into())
        .parse()?;
    anyhow::ensure!(
        [64, 256].contains(&group_size),
        "GPU_PROBE_WORKGROUP must be 64 or 256"
    );
    eprintln!("workgroup size: {group_size}");
    let constants = [("groupSize".to_string(), group_size as f64)]
        .into_iter()
        .collect();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shared dispatch diagnostic"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../dispatch-diagnostic.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dispatch diagnostic"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });
    for (dispatches, split, elements) in [
        (0, false, 64u32),
        (1, false, 64),
        (32, false, 64),
        (128, false, 64),
        (128, true, 64),
        (1, false, 8388608),
        (4, false, 8388608),
        (4, true, 8388608),
    ] {
        let elements = elements.max(group_size);
        let bytes = u64::from(elements) * 4;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diagnostic output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diagnostic readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let mut submitted = 0u32;
        let mut step = || {
            let mut enc = device.create_command_encoder(&Default::default());
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for i in 0..dispatches {
                if split && i > 0 {
                    drop(pass);
                    pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                }
                pass.dispatch_workgroups(
                    (elements / group_size).min(32768),
                    (elements / group_size).div_ceil(32768),
                    1,
                );
            }
            drop(pass);
            queue.submit([enc.finish()]);
            submitted += 1;
        };
        for _ in 0..10 {
            step();
        }
        let _ = device.poll(wgpu::Maintain::Wait);
        let mut rounds = Vec::new();
        let mut medians = Vec::new();
        for _ in 0..2 {
            for _ in 0..3 {
                step();
            }
            let _ = device.poll(wgpu::Maintain::Wait);
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..10 {
                    step();
                }
                let _ = device.poll(wgpu::Maintain::Wait);
                samples.push(start.elapsed().as_secs_f64() * 100.0);
            }
            let mut sorted = samples.clone();
            sorted.sort_by(f64::total_cmp);
            medians.push(sorted[2]);
            rounds.push(samples);
        }
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&output, 0, &read, 0, bytes);
        queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let expected = submitted * dispatches;
        let mapped = read.slice(..).get_mapped_range();
        for (lane, &value) in bytemuck::cast_slice::<u8, u32>(&mapped).iter().enumerate() {
            anyhow::ensure!(value == expected, "Dispatch coverage failure: dispatches={dispatches}, split={split}, lane={lane}: got {value}, expected {expected}. Timing invalid if work is omitted.");
        }
        drop(mapped);
        read.unmap();
        let wall_ms = medians[0].min(medians[1]);
        println!("{{\"groupSize\":{group_size},\"dispatches\":{dispatches},\"splitPasses\":{split},\"bytes\":{bytes},\"verifiedLanes\":{elements},\"submittedSteps\":{submitted},\"expected\":{expected},\"wallMs\":{wall_ms},\"roundMediansMs\":{medians:?},\"rounds\":{rounds:?}}}");
        read.destroy();
        output.destroy();
    }
    Ok(())
}
