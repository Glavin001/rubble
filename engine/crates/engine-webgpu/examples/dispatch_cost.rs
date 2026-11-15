//! Cost of a dependent compute dispatch: N dispatches of a trivial kernel that
//! reads and writes one storage buffer (so each depends on the last), in one pass,
//! timed with pass timestamps. Local measurement tool.
use wgpu::util::DeviceExt;

fn main() {
    std::env::set_var("RUBBLE_GPU_TIMING", "1");
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    // DISPATCH_WG=<words>: the kernel also uses a workgroup array of that many words
    // (zero-initialised by the shader translator at each workgroup's start).
    let wg: u32 = std::env::var("DISPATCH_WG")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let src = if wg == 0 {
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x < arrayLength(&b)) { b[g.x] = b[g.x] + 1u; } }"
            .to_string()
    } else {
        format!(
            "@group(0) @binding(0) var<storage, read_write> b: array<u32>;
var<workgroup> w: array<u32, {wg}>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_index) l: u32) {{
  w[l] = w[l] + 1u;
  workgroupBarrier();
  if (g.x < arrayLength(&b)) {{ b[g.x] = b[g.x] + w[(l + 1u) % 256u]; }} }}"
        )
    };
    println!("workgroup array: {wg} words");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    for elems in [256u32, 8192, 65536] {
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vec![0u32; elems as usize]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut wall = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipe);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..340 {
                        pass.dispatch_workgroups(elems.div_ceil(256), 1, 1);
                    }
                }
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                wall = wall.min(t.elapsed().as_secs_f64() * 1e3);
            }
            println!("{elems:6} threads: wall {wall:.3} ms for 340 dependent dispatches");
        }
        for n in [1u32, 51, 1000] {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &qs,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(&pipe);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..n {
                        pass.dispatch_workgroups(elems.div_ceil(256), 1, 1);
                    }
                }
                enc.resolve_query_set(&qs, 0..2, &resolve, 0);
                enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
                queue.submit([enc.finish()]);
                let slice = read.slice(..);
                slice.map_async(wgpu::MapMode::Read, |_| {});
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                let t: Vec<u64> =
                    bytemuck::cast_slice(&slice.get_mapped_range().unwrap().to_vec()).to_vec();
                read.unmap();
                let ms = (t[1] - t[0]) as f64 * f64::from(queue.get_timestamp_period()) * 1e-6;
                best = best.min(ms);
            }
            println!("{elems:6} threads, {n:5} dependent dispatches: {:.3} ms in the pass, {:.2} us per dispatch", best, best * 1e3 / f64::from(n));
        }
        // Every element must have been incremented once per dispatch: 5 x (1 + 51 + 1000).
        let check = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(elems) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&buf, 0, &check, 0, u64::from(elems) * 4);
        queue.submit([enc.finish()]);
        let slice = check.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let v: Vec<u32> =
            bytemuck::cast_slice(&slice.get_mapped_range().unwrap().to_vec()).to_vec();
        println!(
            "  values: min {} max {} (expected {})",
            v.iter().min().unwrap(),
            v.iter().max().unwrap(),
            5 * (1 + 51 + 1000)
        );
    }
}
