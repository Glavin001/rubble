//! Cost of one workgroup's dispatch of a shipped explicit-beams module that returns at
//! once (every buffer zero: no vehicle), N dependent dispatches in one pass, timed with
//! pass timestamps. Separates a pipeline's fixed cost per dispatch from its work.
//! Local measurement tool:
//!
//! ```text
//! cargo run --release -p engine-webgpu --example explicit_dispatch_cost [module …]
//! ```

fn main() {
    std::env::set_var("RUBBLE_GPU_TIMING", "1");
    let (_, device, queue) = engine_webgpu::default_device_with(true).unwrap();
    let shipped = engine_webgpu::explicit_beams_shipped_wgsl();
    let mut names: Vec<String> = std::env::args().skip(1).collect();
    if names.is_empty() {
        names = [
            "explicit_surface_find_pair",
            "explicit_surface_prep",
            "explicit_surface_find_tri",
            "explicit_beams_seg",
            "explicit_beams_solo",
        ]
        .map(String::from)
        .to_vec();
    }
    let n = 340u32;
    // DISPATCH_MB: each storage buffer's size (MB; default 1).
    let mb: usize = std::env::var("DISPATCH_MB")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(1);
    for name in &names {
        let Some((_, entry, wgsl)) = shipped.iter().find(|(m, _, _)| m == name) else {
            println!("{name}: not shipped");
            continue;
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl.as_str().into()),
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        // The module's bindings (group 0), each a zeroed buffer.
        let mut bufs = Vec::new();
        for line in wgsl.lines() {
            let Some(at) = line.find("@binding(") else {
                continue;
            };
            let rest = &line[at + 9..];
            let b: u32 = rest[..rest.find(')').unwrap()].parse().unwrap();
            let uniform = line.contains("var<uniform>");
            if bufs.iter().any(|(x, _)| *x == b) {
                continue;
            }
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: if uniform { 65536 } else { (mb as u64) << 20 },
                usage: if uniform {
                    wgpu::BufferUsages::UNIFORM
                } else {
                    wgpu::BufferUsages::STORAGE
                },
                mapped_at_creation: false,
            });
            bufs.push((b, buf));
        }
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
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
        let mut best = f64::MAX;
        let mut wall = [f64::MAX; 2];
        for (w, count) in [0u32, n].into_iter().enumerate() {
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&pipe);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..count {
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                }
                queue.submit([enc.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                wall[w] = wall[w].min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        println!(
            "{name:<32} wall: {:.3} ms with no dispatch, {:.3} ms with {n}",
            wall[0], wall[1]
        );
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
                    pass.dispatch_workgroups(1, 1, 1);
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
        println!(
            "{name:<32} {n} dispatches of one workgroup (no vehicle): {best:.3} ms, {:.2} us each",
            best * 1e3 / f64::from(n)
        );
    }
}
