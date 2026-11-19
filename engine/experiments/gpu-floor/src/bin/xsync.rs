//! Can threadgroups on different cores step in lockstep? One car on several cores needs a
//! barrier across threadgroups, which Metal does not have: here each group's lane 0 adds
//! one to a device counter at the end of a phase and spins (bounded) until every group
//! has; the other lanes wait at their threadgroup barrier. Reports µs a phase and how
//! often the bound was hit (a group not running beside the others), for 1 to 40 groups.
use gpu_floor::gpu::{read, write, Gpu};

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct P { uint rounds; uint groups; uint cap; uint work; };
kernel void xs(constant P& p [[buffer(0)]], device atomic_uint* cnt [[buffer(1)]], device float* out [[buffer(2)]],
               uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]])
{
    float acc = float(lid) * 0.001f + 1.0f;
    uint worst = 0u, late = 0u;
    for (uint r = 0; r < p.rounds; r++)
    {
        for (uint i = 0; i < p.work; i++) acc = fma(acc, 0.9993f, 0.001f);
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        if (lid == 0u && p.groups > 1u)
        {
            atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed);
            uint want = p.groups * (r + 1u), i = 0u;
            for (; i < p.cap; i++)
                if (atomic_load_explicit(&cnt[0], memory_order_relaxed) >= want) break;
            worst = max(worst, i);
            if (i >= p.cap) late++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    if (lid == 0u)
    {
        atomic_fetch_max_explicit(&cnt[1], worst, memory_order_relaxed);
        atomic_fetch_add_explicit(&cnt[2], late, memory_order_relaxed);
    }
    out[grp * 1024u + lid] = acc;
}
"#;

fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new()?;
    let pso = gpu.pipeline(SRC, "xs", true)?;
    let cnt = gpu.zeros(64);
    let out = gpu.zeros(64 * 1024 * 4);
    println!("groups | work (fma a lane a phase) | µs a phase | against one group | worst spin (loads) | phases where the bound (65536) was hit");
    for work in [0u32, 64] {
        let mut base = 0.0;
        for groups in [1u32, 2, 4, 8, 16, 40] {
            let run = |rounds: u32| {
                let mut ms = Vec::new();
                let mut info = (0u32, 0u32);
                for _ in 0..5 {
                    write(&cnt, &[0u32; 16]);
                    let p = gpu.buffer(&[rounds, groups, 65536u32, work]);
                    ms.push(gpu.run(&pso, &[&p, &cnt, &out], groups as u64, 1024));
                    let c: Vec<u32> = read(&cnt, 4);
                    info = (info.0.max(c[1]), info.1 + c[2]);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (ms[2], info)
            };
            let (a, info) = run(2000);
            let (b, _) = run(0);
            let us = (a - b) * 1000.0 / 2000.0;
            if groups == 1 {
                base = us;
            }
            println!("{groups:6} | {work:3} | {us:8.3} | {:+8.3} | {:8} | {}", us - base, info.0, info.1);
        }
    }
    Ok(())
}
