//! The GPU's cost model for the explicit frame's design (vehicle-perf-first-principles.md
//! §8), time only: what a dependent dispatch, a barrier phase and a buffer load cost a
//! thread, and how they scale with threads, with synthetic kernels on a table the size of
//! a car's working set.
//!
//! WIDE: `dispatches` dependent dispatches in one compute pass, `threads` threads each
//! (workgroups of `wg`), every thread `loads` loads then one store into the buffer the next
//! dispatch reads. Load kinds: `direct` (addresses from the thread's index: a streamed
//! record), `gather` (an index load, then the data: an incidence list), `chain` (each
//! load's value addresses the next: a dependent chain).
//!
//! WORKGROUP: one dispatch of `groups` workgroups of `lanes` lanes, `phases` phases, a
//! barrier pair after each; in a phase the first `active` lanes take `items` items of
//! `loads` loads and a store each.
//!
//!   cargo run --release -p engine-webgpu-lab --example gpu_cost_model [-- quick]
use wgpu::util::DeviceExt;

const WIDE: &str = r#"
struct P { threads: u32, loads: u32, kind: u32, mask: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<vec4<f32>>;
@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= p.threads) { return; }
  var acc = vec3<f32>(0.0);
  var j = (i * 2654435761u) & p.mask;
  for (var e = 0u; e < p.loads; e++) {
    if (p.kind == 0u) {
      acc += data[(j + e) & p.mask].xyz;
    } else if (p.kind == 1u) {
      acc += data[idx[(i * 37u + e) & p.mask] & p.mask].xyz;
    } else {
      let v = data[j];
      acc += v.xyz;
      j = bitcast<u32>(v.w) & p.mask;
    }
  }
  let keep = data[i & p.mask].w;
  data[i & p.mask] = vec4<f32>(acc * 1e-9, keep);
}
"#;

const WORKGROUP: &str = r#"
struct P { phases: u32, loads: u32, act: u32, mask: u32, items: u32, kind: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<vec4<f32>>;
@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let base = g.x * 8191u;
  for (var s = 0u; s < p.phases; s++) {
    if (l.x < p.act) {
      for (var it = 0u; it < p.items; it++) {
        let i = base + l.x + it * LANES;
        var acc = vec3<f32>(0.0);
        var j = (i * 2654435761u) & p.mask;
        for (var e = 0u; e < p.loads; e++) {
          if (p.kind == 0u) {
            acc += data[(j + e) & p.mask].xyz;
          } else if (p.kind == 1u) {
            acc += data[idx[(i * 37u + e) & p.mask] & p.mask].xyz;
          } else {
            let v = data[j];
            acc += v.xyz;
            j = bitcast<u32>(v.w) & p.mask;
          }
        }
        let keep = data[i & p.mask].w;
        data[i & p.mask] = vec4<f32>(acc * 1e-9, keep);
      }
    }
    storageBarrier(); workgroupBarrier();
  }
}
"#;

fn main() {
    let quick = std::env::args().any(|a| a == "quick");
    let _lock = engine_webgpu::hold_gpu_lock();
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    let module = |src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
    let pipe = |m: &wgpu::ShaderModule| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    // Tables: 2^16 records (1 MB: a car's working set) and 2^22 (64 MB: past every cache).
    let table = |bits: u32| {
        let n = 1usize << bits;
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let idx: Vec<u32> = (0..n).map(|_| next()).collect();
        let data: Vec<[f32; 4]> = (0..n).map(|_| [1.0, 2.0, 3.0, f32::from_bits(next() & 0x007F_FFFF)]).collect();
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&idx), usage: wgpu::BufferUsages::STORAGE });
        let db = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE });
        (ib, db, (n - 1) as u32)
    };
    let median = |f: &mut dyn FnMut()| {
        for _ in 0..4 {
            f();
        }
        let mut ms = Vec::new();
        for _ in 0..(if quick { 9 } else { 15 }) {
            let t = std::time::Instant::now();
            f();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        ms[ms.len() / 2]
    };
    let bind = |p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect::<Vec<_>>(),
        })
    };
    let kinds = ["direct", "gather", "chain"];
    for bits in [16u32, 22] {
        let (ib, db, mask) = table(bits);
        println!("== table of 2^{bits} records ({} MB) ==", (16usize << bits) >> 20);
        println!("WIDE: us a dependent dispatch = (pass of 341 - pass of 1) / 340");
        let only_wg = std::env::args().any(|a| a == "workgroup");
        for wg in if only_wg { vec![] } else { vec![64u32, 256] } {
            let m = module(&WIDE.replace("WG", &wg.to_string()));
            let wp = pipe(&m);
            for threads in [64u32, 512, 4096, 32768, 262144] {
                if threads > mask + 1 {
                    continue;
                }
                for (kind, name) in kinds.iter().enumerate() {
                    let mut line = format!("  wg {wg:>3} threads {threads:>6} {name:>6}:");
                    for loads in [0u32, 1, 2, 4, 8, 16, 32] {
                        if kind > 0 && loads == 0 {
                            line.push_str("      -");
                            continue;
                        }
                        let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: bytemuck::cast_slice(&[threads, loads, kind as u32, mask]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                        let g = bind(&wp, &[&pu, &ib, &db]);
                        let mut run = |n: u32| {
                            let mut enc = device.create_command_encoder(&Default::default());
                            {
                                let mut pass = enc.begin_compute_pass(&Default::default());
                                pass.set_pipeline(&wp);
                                pass.set_bind_group(0, &g, &[]);
                                for _ in 0..n {
                                    pass.dispatch_workgroups(threads.div_ceil(wg), 1, 1);
                                }
                            }
                            queue.submit([enc.finish()]);
                            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                        };
                        let many = median(&mut || run(341));
                        let one = median(&mut || run(1));
                        line.push_str(&format!(" {:>6.2}", (many - one) / 340.0 * 1e3));
                    }
                    println!("{line}   (loads 0 1 2 4 8 16 32)");
                }
            }
        }
        println!("WORKGROUP: us a phase = (dispatch of 2041 phases - dispatch of 1) / 2040");
        for lanes in [256u32, 512] {
            let m = module(&WORKGROUP.replace("LANES", &format!("{lanes}u")).replace(&format!("workgroup_size({lanes}u)"), &format!("workgroup_size({lanes})")));
            let wp = pipe(&m);
            for groups in [1u32, 8, 32] {
                for (active, items) in [(0u32, 0u32), (4, 1), (64, 1), (lanes, 1), (lanes, 2), (lanes, 8)] {
                    for (kind, name) in kinds.iter().enumerate() {
                        if active == 0 && kind > 0 {
                            continue;
                        }
                        let mut line = format!("  lanes {lanes} groups {groups:>2} active {active:>3} items {items} {name:>6}:");
                        for loads in [0u32, 1, 4, 16] {
                            if kind > 0 && loads == 0 {
                                line.push_str("      -");
                                continue;
                            }
                            let mut run = |phases: u32| {
                                let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: None,
                                    contents: bytemuck::cast_slice(&[phases, loads, active, mask, items, kind as u32, 0, 0]),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                });
                                let g = bind(&wp, &[&pu, &ib, &db]);
                                let mut enc = device.create_command_encoder(&Default::default());
                                {
                                    let mut pass = enc.begin_compute_pass(&Default::default());
                                    pass.set_pipeline(&wp);
                                    pass.set_bind_group(0, &g, &[]);
                                    pass.dispatch_workgroups(groups, 1, 1);
                                }
                                queue.submit([enc.finish()]);
                                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                            };
                            let many = median(&mut || run(2041));
                            let one = median(&mut || run(1));
                            line.push_str(&format!(" {:>6.2}", (many - one) / 2040.0 * 1e3));
                        }
                        println!("{line}   (loads 0 1 4 16)");
                    }
                }
            }
        }
    }
}
