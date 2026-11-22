//! What an atomic costs a thread (vehicle-perf-first-principles.md §8): one workgroup of
//! 512 lanes, barrier phases; in a phase the first `active` lanes each make `n` operations
//! of a kind on a 1 MB buffer: `load` (a plain load of a vec4), `aload` (atomicLoad),
//! `aadd` (atomicAdd, every lane its own words), `aadd-shared` (atomicAdd, every lane the
//! same few words: contended), `store` (a plain store). us a phase =
//! (2041 phases - 1) / 2040.
//!
//!   cargo run --release -p engine-webgpu-lab --example gpu_atomic_cost
use wgpu::util::DeviceExt;

const SRC: &str = r#"
struct P { phases: u32, n: u32, act: u32, kind: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<i32>>;
@compute @workgroup_size(512)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  for (var s = 0u; s < p.phases; s++) {
    if (l.x < p.act) {
      var sum = 0.0;
      var isum = 0;
      let j = (l.x * 2654435761u) & 0xffffu;
      for (var e = 0u; e < p.n; e++) {
        let a = (j + e * 97u) & 0xffffu;
        if (p.kind == 0u) { sum += data[a].x; }
        else if (p.kind == 1u) { isum += atomicLoad(&acc[a]); }
        else if (p.kind == 2u) { isum += atomicAdd(&acc[a], 1); }
        else if (p.kind == 3u) { isum += atomicAdd(&acc[e & 7u], 1); }
        else { data[a] = vec4<f32>(f32(e)); }
      }
      data[l.x] = vec4<f32>(sum + f32(isum));
    }
    storageBarrier(); workgroupBarrier();
  }
}
"#;

fn main() {
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SRC.into()) });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &vec![0u8; 16 << 16], usage: wgpu::BufferUsages::STORAGE });
    let acc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &vec![0u8; 4 << 16], usage: wgpu::BufferUsages::STORAGE });
    let run = |phases: u32, n: u32, act: u32, kind: u32| {
        let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[phases, n, act, kind]), usage: wgpu::BufferUsages::UNIFORM });
        let g = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pu.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: acc.as_entire_binding() },
            ],
        });
        let mut ms = Vec::new();
        for k in 0..12 {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &g, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            let t = std::time::Instant::now();
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if k >= 3 {
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        ms.sort_by(f64::total_cmp);
        ms[ms.len() / 2]
    };
    for (kind, name) in ["load", "aload", "aadd", "aadd-shared", "store"].iter().enumerate() {
        for act in [4u32, 64, 512] {
            let mut line = format!("{name:>12} active {act:>3}:");
            for n in [0u32, 4, 16, 64] {
                line.push_str(&format!(" {:>6.2}", (run(2041, n, act, kind as u32) - run(1, n, act, kind as u32)) / 2040.0 * 1e3));
            }
            println!("{line}   (us a phase at 0 4 16 64 operations a lane)");
        }
    }
}
