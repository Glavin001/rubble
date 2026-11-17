//! What one barrier phase of a single-workgroup kernel costs (vehicle-perf-first-principles.md
//! §8.35), time only: the list kernel's phases measured 3× what their dependent loads
//! predict. One dispatch of `groups` workgroups of 256 lanes runs `phases` phases; in each,
//! every lane takes `items` items of `k` dependent loads (each load's value addresses the
//! next: a chain), optionally a store, an atomic, or strict arithmetic, then a barrier.
//! µs a phase = (dispatch of 1025 phases − dispatch of 1) / 1024. One factor varies at a
//! time from the base: a read-write storage buffer of 2^16 records (1 MB), 2 items, both
//! barriers (storage and workgroup, as the kernels' sync()), no store, one workgroup.
//!
//!   cargo run --release -p engine-webgpu --example phase_cost
use wgpu::util::DeviceExt;

const SRC: &str = r#"
struct P { phases: u32, k: u32, items: u32, mask: u32, zero: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ro: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rw: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> ac: array<atomic<u32>>;
var<workgroup> wcount: atomic<u32>;
var<workgroup> wsum: array<u32, 256>;
fn nf(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ (p.zero >> 31u)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  var acc = 0.0;
  for (var s = 0u; s < p.phases; s++) {
    for (var it = 0u; it < p.items; it++) {
      var j = ((g.x * 8191u + l.x + it * 256u + s * 977u) * 2654435761u) & p.mask;
      for (var e = 0u; e < p.k; e++) {
        LOAD
      }
      EXTRA
    }
    BARRIER
  }
  if (acc == 12345.0) { rw[0] = ro[1]; atomicAdd(&ac[0], 1u); }
}
"#;

struct V {
    name: &'static str,
    load: &'static str,
    extra: &'static str,
    barrier: &'static str,
}

const LOAD_RW: &str = "let v = rw[j]; acc += v.x; j = bitcast<u32>(v.w) & p.mask;";
const LOAD_RO: &str = "let v = ro[j]; acc += v.x; j = bitcast<u32>(v.w) & p.mask;";
const LOAD_ATOM: &str = "let v = atomicLoad(&ac[j]); acc += f32(v & 1u); j = v & p.mask;";
// The strict arithmetic's opaque zero around each load's use (nf: an XOR with a uniform).
const LOAD_STRICT: &str = "let v = rw[j]; acc = nf(nf(acc + nf(v.x * v.y)) + nf(v.z * 0.5)); j = bitcast<u32>(v.w) & p.mask;";
// Each load's index also passes through a uniform read (gP's words read in the chain).
const LOAD_UNI: &str = "let v = rw[j ^ (p.zero >> 31u)]; acc += v.x; j = (bitcast<u32>(v.w) + p.pad0) & p.mask;";
const BOTH: &str = "storageBarrier(); workgroupBarrier();";
const WG: &str = "workgroupBarrier();";

// CODE: one workgroup of 256 lanes runs `n` dependent fma steps once, either written out
// (`n` distinct statements: n · ~8 bytes of machine code executed once) or as a loop (a few
// instructions executed n times): the same arithmetic chain, different code to fetch. Each
// target dispatch alternates with an evictor (a large kernel of other code) or with an
// empty one, so the target's code starts cold or hot in the instruction cache.
fn code_kernel(n: usize, unrolled: bool) -> String {
    let mut body = String::new();
    if unrolled {
        for i in 0..n {
            body.push_str(&format!("  acc = fma(acc, {:.6}, {:.6});\n", 1.0 + (i % 97) as f32 * 1e-6, (i % 89) as f32 * 1e-3));
        }
    } else {
        body.push_str(&format!("  for (var i = 0u; i < {n}u + (p.zero >> 31u); i++) {{ acc = fma(acc, 1.0 + f32(i % 97u) * 1e-6, f32(i % 89u) * 1e-3); }}\n"));
    }
    format!(
        "struct P {{ zero: u32, a: u32, b: u32, c: u32 }};\n@group(0) @binding(0) var<uniform> p: P;\n@group(0) @binding(1) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) l: vec3<u32>) {{\n  var acc = f32(l.x + p.zero);\n{body}  o[l.x] = acc;\n}}\n"
    )
}

fn code_mode(device: &wgpu::Device, queue: &wgpu::Queue) {
    let pipe = |src: String| {
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[0u32; 4]), usage: wgpu::BufferUsages::UNIFORM });
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    let group = |p: &wgpu::ComputePipeline| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: pu.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() }],
        })
    };
    let empty = pipe(code_kernel(0, false));
    let ge = group(&empty);
    // The evictor: 16384 distinct statements (≈128 KB of code), run between targets.
    let evict = pipe(code_kernel(16384, true));
    let gv = group(&evict);
    let time = |target: &wgpu::ComputePipeline, gt: &wgpu::BindGroup, other: &wgpu::ComputePipeline, go: &wgpu::BindGroup| {
        let mut run = |with_target: bool| {
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..200 {
                    if with_target {
                        pass.set_pipeline(target);
                        pass.set_bind_group(0, gt, &[]);
                    } else {
                        pass.set_pipeline(&empty);
                        pass.set_bind_group(0, &ge, &[]);
                    }
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(other);
                    pass.set_bind_group(0, go, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            queue.submit([enc.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        let mut med = |w: bool| {
            for _ in 0..2 {
                run(w);
            }
            let mut v: Vec<f64> = (0..7)
                .map(|_| {
                    let t = std::time::Instant::now();
                    run(w);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[3]
        };
        (med(true) - med(false)) / 200.0 * 1e3
    };
    println!("== CODE: µs a dispatch of one workgroup running n dependent fma steps, over an empty dispatch ==");
    for n in [256usize, 1024, 4096] {
        let mut line = format!("  n {n:>5}:");
        for unrolled in [false, true] {
            let t = pipe(code_kernel(n, unrolled));
            let gt = group(&t);
            let hot = time(&t, &gt, &empty, &ge);
            let cold = time(&t, &gt, &evict, &gv);
            line.push_str(&format!(" {} hot {hot:.2} cold {cold:.2};", if unrolled { "written out" } else { "loop" }));
        }
        println!("{line}");
    }
}

fn main() {
    let _lock = engine_webgpu::hold_gpu_lock();
    let (_, device, queue) = engine_webgpu::default_device().unwrap();
    if std::env::var("PHASE_MODE").as_deref() != Ok("loads") {
        code_mode(&device, &queue);
        if std::env::var("PHASE_MODE").as_deref() == Ok("code") {
            return;
        }
    }
    let variants = [
        V { name: "rw chain", load: LOAD_RW, extra: "", barrier: BOTH },
        V { name: "ro chain", load: LOAD_RO, extra: "", barrier: BOTH },
        V { name: "atomic-load chain", load: LOAD_ATOM, extra: "", barrier: BOTH },
        V { name: "rw chain, strict ops", load: LOAD_STRICT, extra: "", barrier: BOTH },
        V { name: "rw chain, uniform in index", load: LOAD_UNI, extra: "", barrier: BOTH },
        V { name: "rw chain, workgroup barrier only", load: LOAD_RW, extra: "", barrier: WG },
        V { name: "rw chain + store an item", load: LOAD_RW, extra: "rw[j] = vec4<f32>(acc, 0.0, 0.0, bitcast<f32>(j));", barrier: BOTH },
        V { name: "rw chain + store, wg barrier", load: LOAD_RW, extra: "rw[j] = vec4<f32>(acc, 0.0, 0.0, bitcast<f32>(j));", barrier: WG },
        V { name: "rw chain + device atomic add an item", load: LOAD_RW, extra: "atomicAdd(&ac[j], 1u);", barrier: BOTH },
        V { name: "rw chain + shared atomic add (one word)", load: LOAD_RW, extra: "let slot = atomicAdd(&wcount, 1u); wsum[slot & 255u] = j;", barrier: BOTH },
        V { name: "rw chain + device atomic max (8 words)", load: LOAD_RW, extra: "atomicMax(&ac[l.x & 7u], j);", barrier: BOTH },
    ];
    // Records whose .w addresses a random next record; the atomic table the same indices.
    let table = |bits: u32| {
        let n = 1usize << bits;
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let data: Vec<[f32; 4]> = (0..n).map(|_| [1.0, 2.0, 3.0, f32::from_bits(next() & ((n as u32) - 1))]).collect();
        let at: Vec<u32> = (0..n).map(|_| next() & ((n as u32) - 1)).collect();
        let usage = wgpu::BufferUsages::STORAGE;
        let ro = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage });
        let rw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage });
        let ac = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&at), usage });
        (ro, rw, ac, (n - 1) as u32)
    };
    let median = |f: &mut dyn FnMut()| {
        for _ in 0..3 {
            f();
        }
        let mut ms = Vec::new();
        for _ in 0..9 {
            let t = std::time::Instant::now();
            f();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(f64::total_cmp);
        ms[ms.len() / 2]
    };
    let only = std::env::var("PHASE_ONLY").ok();
    for bits in [16u32, 22] {
        let (ro, rw, ac, mask) = table(bits);
        println!("== table 2^{bits} records ({} MB); µs a phase (2 items a lane unless stated) ==", (16usize << bits) >> 20);
        for v in &variants {
            if only.as_deref().is_some_and(|o| !v.name.contains(o)) {
                continue;
            }
            let src = SRC.replace("LOAD", v.load).replace("EXTRA", v.extra).replace("BARRIER", v.barrier);
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let layout = pipe.get_bind_group_layout(0);
            let cases: Vec<(u32, u32, u32)> = if v.name == "rw chain" {
                // (groups, items, k): k, items and workgroups varied on the base.
                let mut c: Vec<(u32, u32, u32)> = [0u32, 1, 2, 4, 8, 16].iter().map(|&k| (1, 2, k)).collect();
                c.extend([(1, 1, 4), (1, 3, 4), (3, 2, 4), (16, 2, 4), (128, 2, 4)]);
                c
            } else {
                [0u32, 1, 4, 16].iter().map(|&k| (1, 2, k)).collect()
            };
            let mut line = format!("  {:<42}", v.name);
            for (groups, items, k) in cases {
                let mut run = |phases: u32| {
                    let pu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[phases, k, items, mask, 0, 0, 0, 0]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let entries: Vec<wgpu::BindGroupEntry> = [&pu, &ro, &rw, &ac]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                        .collect();
                    let g = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipe);
                        pass.set_bind_group(0, &g, &[]);
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                    queue.submit([enc.finish()]);
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                };
                let many = median(&mut || run(1025));
                let one = median(&mut || run(1));
                let us = (many - one) / 1024.0 * 1e3;
                let tag = if groups == 1 && items == 2 { format!("k{k}") } else { format!("g{groups}i{items}k{k}") };
                line.push_str(&format!(" {tag}={us:.2}"));
            }
            println!("{line}");
        }
    }
}
