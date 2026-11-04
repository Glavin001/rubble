//! Measurement only: GPU time of N empty indirect dispatches (0 workgroups) in one
//! compute pass, against one dispatch.
use engine_webgpu::default_device_with;
fn main() -> anyhow::Result<()> {
    let (_, device, queue) = default_device_with(true)?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> b: array<u32>; @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) i: vec3<u32>) { if (i.x == 0u) { b[0] = b[0] + 1u; } }".into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let args = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
    let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    for (n, w, direct) in [(1u32, 0u32, false), (51, 0, false), (51, 1, false), (1, 1, true), (51, 0, true), (51, 1, true), (51, 40, true)] {
        queue.write_buffer(&args, 0, bytemuck::cast_slice(&[w, 1u32, 1u32, 0u32]));
        let mut times = Vec::new();
        for _ in 0..50 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }) });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                for _ in 0..n {
                    if direct { pass.dispatch_workgroups(w, 1, 1) } else { pass.dispatch_workgroups_indirect(&args, 0) }
                }
            }
            enc.resolve_query_set(&set, 0..2, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 16);
            queue.submit([enc.finish()]);
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let t: Vec<u64> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
            staging.unmap();
            times.push((t[1] - t[0]) as f64 * queue.get_timestamp_period() as f64 * 1e-3);
        }
        times.sort_by(f64::total_cmp);
        println!("{n} {} dispatches of {w} workgroups: median {:.1} µs", if direct { "direct" } else { "indirect" }, times[times.len() / 2]);
    }
    Ok(())
}
