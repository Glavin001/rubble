//! Dispatch and barrier costs for the contact-group design
//! (`engine/docs/measurements/contact-groups-design.md`), native wgpu: N dependent
//! dispatches of a 512-lane kernel bound as the explicit frame is (a uniform and ten
//! storage buffers), with a small or a vehicle-sized workgroup array, with and without
//! the shader translator's workgroup zeroing; and the cost of one `sync()` (storage then
//! workgroup barrier) inside a dispatch. The same WGSL runs in headless Chrome
//! (`engine/tools/contact-groups/dispatch.mjs`). Local measurement tool.
//!
//!   cargo run --release -p engine-webgpu-lab --example contact_dispatch_cost
use wgpu::util::DeviceExt;

const SRC: &str = include_str!("../../../tools/contact-groups/dispatch_cost.wgsl");

struct Rig {
    device: wgpu::Device,
    queue: wgpu::Queue,
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

impl Rig {
    /// (GPU, wall) ms of one pass holding `n` dispatches of `groups` workgroups: the
    /// medians of 9 (the wall time from submit to completion; differences of two such
    /// cancel the submission's own latency).
    fn time(&self, pipe: &wgpu::ComputePipeline, group: &wgpu::BindGroup, groups: u32, n: u32) -> (f64, f64) {
        let mut gpu = Vec::new();
        let mut wall = Vec::new();
        for _ in 0..9 {
            let t0 = std::time::Instant::now();
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &self.qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, group, &[]);
                for _ in 0..n {
                    pass.dispatch_workgroups(groups, 1, 1);
                }
            }
            enc.resolve_query_set(&self.qs, 0..2, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, 16);
            self.queue.submit([enc.finish()]);
            let slice = self.read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            wall.push(t0.elapsed().as_secs_f64() * 1e3);
            let t: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap().to_vec()).to_vec();
            self.read.unmap();
            gpu.push(t[1].wrapping_sub(t[0]) as f64 * f64::from(self.queue.get_timestamp_period()) * 1e-6);
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        (med(gpu), med(wall))
    }
}

fn main() {
    std::env::set_var("RUBBLE_GPU_TIMING", "1");
    engine_webgpu::hold_gpu_lock();
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
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
    let rig = Rig { device, queue, qs, resolve, read };
    let d = &rig.device;
    let storage: Vec<wgpu::Buffer> = (0..10)
        .map(|i| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&vec![0u32; if i == 0 { 512 * 128 } else { 64 }]),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    for words in [512u32, 7936] {
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SRC.replace("WG_WORDS", &words.to_string()).into()),
        });
        for zero in [true, false] {
            let pipe = |entry: &str| {
                d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: zero,
                        ..Default::default()
                    },
                    cache: None,
                })
            };
            let group = |p: &wgpu::ComputePipeline, k: u32| {
                let u = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[k, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
                entries.extend(storage.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: b.as_entire_binding(),
                }));
                d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
            };
            let disp = pipe("dispatch");
            let g = group(&disp, 0);
            for groups in [1u32, 40, 128] {
                let one = rig.time(&disp, &g, groups, 1);
                let many = rig.time(&disp, &g, groups, 341);
                println!(
                    "dispatch: {words:5} workgroup words, zeroing {zero:5}, {groups:3} workgroups: a dependent dispatch {:.2} us (pass timestamps) / {:.2} us (wall); one alone {:.1} us",
                    (many.0 - one.0) * 1e3 / 340.0,
                    (many.1 - one.1) * 1e3 / 340.0,
                    one.0 * 1e3,
                );
            }
            let sy = pipe("syncs");
            for groups in [1u32, 40] {
                let base = rig.time(&sy, &group(&sy, 0), groups, 1);
                let k = 4000;
                let loaded = rig.time(&sy, &group(&sy, k), groups, 1);
                println!(
                    "sync:     {words:5} workgroup words, zeroing {zero:5}, {groups:3} workgroups: {:.3} us a sync (pass timestamps) / {:.3} us (wall)",
                    (loaded.0 - base.0) * 1e3 / f64::from(k),
                    (loaded.1 - base.1) * 1e3 / f64::from(k),
                );
            }
        }
    }
}
