//! Diagnostic: the cost of N rounds of 3 dependent trivial dispatches in one compute pass
//! (the contact pipeline's shape), with 1 and 33 threadgroups of 256 and 8 000 threads.
use stress_gpu::gpu::Gpu;
use wgpu::util::DeviceExt;

fn main() {
    let gpu = Gpu::new().expect("gpu");
    let d = &gpu.device;
    let src = "@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256) fn k(@builtin(global_invocation_id) id: vec3<u32>) { let i = id.x % 1024u; b[i] = b[i] * 0.5 + 1.0; }";
    let m = d.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &vec![0u8; 4096], usage: wgpu::BufferUsages::STORAGE });
    let p: Vec<wgpu::ComputePipeline> = (0..3).map(|_| d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &m, entry_point: Some("k"), compilation_options: Default::default(), cache: None })).collect();
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p[0].get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });
    let bgs: Vec<wgpu::BindGroup> = p.iter().map(|pp| d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pp.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] })).collect();
    let _ = bg;
    for groups in [1u32, 33, 32] {
        for rounds in [450usize] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let mut enc = d.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for _ in 0..rounds {
                        for k in 0..3 {
                            pass.set_pipeline(&p[k]);
                            pass.set_bind_group(0, &bgs[k], &[]);
                            pass.dispatch_workgroups(groups, 1, 1);
                        }
                    }
                }
                let t = std::time::Instant::now();
                gpu.queue.submit([enc.finish()]);
                d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{groups} groups x 256: {:.1} us per round of 3 dispatches", best * 1e6 / rounds as f64);
        }
    }
}
