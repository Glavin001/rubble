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

/// Is memory one workgroup wrote seen by a workgroup on another core after the
/// cross-group barrier? Each round every group stamps its own 1,024 words (plain stores,
/// or atomic ones with `p.work` bit 0), all groups meet, then each reads its neighbour's
/// words (a plain load and an atomic load of each: the same addresses every round, so a
/// core that kept a private copy would read last round's) and adds to the neighbour's
/// sums (atomic); all groups meet again; next round each group reads its own sums back
/// (plain load) and clears them (plain store). Counts every wrong read by kind.
const DATA: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct P { uint rounds; uint groups; uint cap; uint work; };
#define BARR threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#define XB { BARR; if (lid == 0u) { xr++; atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed); uint i = 0u; for (; i < p.cap; i++) if (atomic_load_explicit(&cnt[0], memory_order_relaxed) >= p.groups * xr) break; if (i >= p.cap) atomic_fetch_add_explicit(&cnt[2], 1u, memory_order_relaxed); } BARR; }
kernel void xd(constant P& p [[buffer(0)]], device atomic_uint* cnt [[buffer(1)]], device uint* data [[buffer(2)]], device uint* sums [[buffer(3)]],
               uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]])
{
    uint xr = 0u;
    const uint nb = (grp + 1u) % p.groups;
    for (uint r = 0; r < p.rounds; r++)
    {
        uint got = sums[grp * 1024u + lid];
        uint want = r == 0u ? 0u : (r - 1u) * 7u + lid + 1u;
        if (got != want) atomic_fetch_add_explicit(&cnt[6], 1u, memory_order_relaxed);
        sums[grp * 1024u + lid] = 0u;
        const uint stamp = r << 16 | (grp * 1024u + lid);
        if ((p.work & 1u) != 0u) atomic_store_explicit((device atomic_uint*)(data + grp * 1024u + lid), stamp, memory_order_relaxed);
        else data[grp * 1024u + lid] = stamp;
        XB;
        const uint exp = r << 16 | (nb * 1024u + lid);
        if (data[nb * 1024u + lid] != exp) atomic_fetch_add_explicit(&cnt[4], 1u, memory_order_relaxed);
        if (atomic_load_explicit((device atomic_uint*)(data + nb * 1024u + lid), memory_order_relaxed) != exp) atomic_fetch_add_explicit(&cnt[5], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_uint*)(sums + nb * 1024u + lid), r * 7u + lid + 1u, memory_order_relaxed);
        XB;
    }
}
"#;

fn data_test(gpu: &Gpu) -> anyhow::Result<()> {
    let pso = gpu.pipeline(DATA, "xd", false)?;
    if std::env::args().any(|a| a == "--compile-only") {
        println!("compiled");
        return Ok(());
    }
    let cnt = gpu.zeros(64);
    let data = gpu.zeros(64 * 1024 * 4);
    let sums = gpu.zeros(64 * 1024 * 4);
    println!("memory across cores after the cross-group barrier (2,000 rounds, 1,024 words a group a round)");
    println!("groups | the writer's stores | wrong plain loads | wrong atomic loads | wrong sums read back by their owner | barriers missed");
    for groups in [2u32, 3, 8, 39, 40] {
        for atomic_store in [0u32, 1] {
            write(&cnt, &[0u32; 16]);
            write(&data, &vec![0u32; 64 * 1024]);
            write(&sums, &vec![0u32; 64 * 1024]);
            let p = gpu.buffer(&[2000u32, groups, 1 << 20, atomic_store]);
            gpu.run(&pso, &[&p, &cnt, &data, &sums], groups as u64, 1024);
            let c: Vec<u32> = read(&cnt, 8);
            println!("{groups:6} | {:6} | {:8} | {:8} | {:8} | {}", if atomic_store == 1 { "atomic" } else { "plain" }, c[4], c[5], c[6], c[2]);
        }
    }
    Ok(())
}

/// Does a workgroup waiting at the cross-group barrier (its lane 0 loading the counter in
/// a loop) slow the workgroup that is still working? Group 0's first warp does 40
/// scattered device atomic adds a lane a round (a hit pass's shape); group 1 has nothing to
/// do and waits. Against group 0 alone, and against a waiter that does some arithmetic
/// between its loads (`p.work` bit 1: 64 multiply-adds a poll).
const SPIN: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct P { uint rounds; uint groups; uint cap; uint work; };
#define BARR threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
kernel void xs(constant P& p [[buffer(0)]], device atomic_uint* cnt [[buffer(1)]], device atomic_uint* dat [[buffer(2)]], device float* out [[buffer(3)]],
               uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]])
{
    uint xr = 0u;
    float idle = float(lid) + 1.0f;
    for (uint r = 0; r < p.rounds; r++)
    {
        if (grp == 0u && lid < 32u && (p.work & 1u) != 0u)
            for (uint k = 0; k < 40u; k++) atomic_fetch_add_explicit(&dat[(lid * 977u + k * 131u + r * 7u) & 4095u], 1u, memory_order_relaxed);
        BARR;
        if (lid == 0u && p.groups > 1u)
        {
            xr++;
            atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed);
            uint i = 0u;
            for (; i < p.cap; i++)
            {
                if (atomic_load_explicit(&cnt[0], memory_order_relaxed) >= p.groups * xr) break;
                if ((p.work & 2u) != 0u) for (uint d = 0; d < 64u; d++) idle = idle * 1.000001f + 0.25f;
            }
            if (i >= p.cap) atomic_fetch_add_explicit(&cnt[2], 1u, memory_order_relaxed);
        }
        BARR;
    }
    out[grp * 1024u + lid] = idle;
}
"#;

fn spin_test(gpu: &Gpu) -> anyhow::Result<()> {
    let pso = gpu.pipeline(SPIN, "xs", true)?;
    if std::env::args().any(|a| a == "--compile-only") {
        println!("compiled");
        return Ok(());
    }
    let cnt = gpu.zeros(64);
    let dat = gpu.zeros(4096 * 4);
    let out = gpu.zeros(64 * 1024 * 4);
    println!("a working workgroup beside a waiting one (µs a round, medians of 5; 2,000 rounds)");
    println!("groups | group 0's work | the waiter | µs a round | barriers missed");
    for (groups, work, name, waiter) in [(1u32, 0u32, "none", "-"), (1, 1, "40 scattered atomic adds a lane, one warp", "-"), (2, 0, "none", "loads the counter"), (2, 1, "40 scattered atomic adds a lane, one warp", "loads the counter"), (2, 3, "40 scattered atomic adds a lane, one warp", "64 multiply-adds a load"), (2, 2, "none", "64 multiply-adds a load")] {
        let mut us = Vec::new();
        let mut missed = 0;
        for _ in 0..5 {
            write(&cnt, &[0u32; 16]);
            let p = gpu.buffer(&[2000u32, groups, 1 << 20, work]);
            us.push(gpu.run(&pso, &[&p, &cnt, &dat, &out], groups as u64, 1024) * 1000.0 / 2000.0);
            missed += read::<u32>(&cnt, 4)[2];
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{groups:6} | {name:42} | {waiter:24} | {:8.3} | {missed}", us[2]);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new()?;
    if std::env::args().any(|a| a == "--spin") {
        return spin_test(&gpu);
    }
    if std::env::args().any(|a| a == "--data") {
        return data_test(&gpu);
    }
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
