//! What one 1,024-lane threadgroup pays for each primitive the floor kernel is built of:
//! a barrier phase, dependent and independent arithmetic, threadgroup-memory and
//! device-memory loads and stores, atomics, and live registers. Each row: one dispatch
//! of ROUNDS phases against one of 0, µs a phase = the difference / ROUNDS (median of 5).
//! Every loop is bounded by a uniform count the host sets (no unbounded loops).
use gpu_floor::gpu::{read, Gpu};

fn kernel(body: &str, pre: &str, barrier: &str) -> String {
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;
struct P {{ uint rounds; uint warps; uint n; uint zero; }};
kernel void bench(constant P& p [[buffer(0)]], device float4* dev [[buffer(1)]], device atomic_uint* da [[buffer(2)]],
                  device float* out [[buffer(3)]], uint lid [[thread_position_in_threadgroup]], uint grp [[threadgroup_position_in_grid]])
{{
    threadgroup float tg[6144];
    threadgroup atomic_uint tga[256];
    dev += grp * 65536; da += grp * 4096; out += grp * 1024;
    const bool on = lid < p.warps * 32u;
    float acc = float(lid) * 0.001f + 1.0f;
    float4 v = float4(acc, acc + 1.0f, acc + 2.0f, acc + 3.0f);
    uint ix = (lid * 2654435761u) >> 16;
    if (lid < 256u) atomic_store_explicit(&tga[lid], 0u, memory_order_relaxed);
    for (uint i = lid; i < 6144u; i += 1024u) tg[i] = float(i & 1023u);
    {pre}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < p.rounds; r++)
    {{
        if (on)
        {{
            {body}
        }}
        {barrier}
    }}
    out[lid] = acc + v.x + v.y + v.z + v.w + float(ix) + tg[(lid * 6u) % 6144u] + float(atomic_load_explicit(&tga[lid & 255u], memory_order_relaxed));
}}
"#
    )
}

fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new()?;
    println!("device: max threadgroup memory {} B", gpu.device.max_threadgroup_memory_length());
    let rounds = 2000u32;
    let compile_only = std::env::args().any(|a| a == "--compile-only");
    let groups: u64 = std::env::args().nth(1).filter(|a| !a.starts_with("--")).map_or(1, |x| x.parse().unwrap());
    let dev = gpu.buffer(&vec![[1.0f32, 2.0, 3.0, 4.0]; 65536 * groups as usize]);
    {
        // The device table: each record's .x the index of another record (a chain).
        let mut t = vec![[0.0f32; 4]; 65536 * groups as usize];
        let mut s = 12345u32;
        for (i, r) in t.iter_mut().enumerate() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *r = [f32::from_bits(s >> 16), (i & 255) as f32 * 1e-3, 0.5, 0.25];
        }
        gpu_floor::gpu::write(&dev, &t);
    }
    let da = gpu.zeros(4096 * 4 * groups as usize);
    let out = gpu.zeros(1024 * 4 * groups as usize);
    let both = "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);";
    let tgb = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    let fma = |n: usize| (0..n).map(|i| format!("acc = fma(acc, 0.999{}f, 0.001f);", i % 7)).collect::<Vec<_>>().join("\n");
    // Four independent chains, written one after another, or interleaved.
    let seq4 = |n: usize| (0..4).map(|c| (0..n).map(|i| format!("v.{} = fma(v.{}, 0.999{}f, 0.001f);", ["x", "y", "z", "w"][c], ["x", "y", "z", "w"][c], i % 7)).collect::<Vec<_>>().join("\n")).collect::<Vec<_>>().join("\n");
    let int4 = |n: usize| (0..n).map(|i| format!("v = fma(v, float4(0.999{}f), float4(0.001f));", i % 7)).collect::<Vec<_>>().join("\n");
    let regs = |n: usize| -> (String, String) {
        let pre = (0..n).map(|i| format!("float q{i} = acc * {}.5f;", i + 1)).collect::<Vec<_>>().join("\n");
        let body = (0..n).map(|i| format!("q{i} = q{i} * 0.9999f + q{};", (i + 1) % n)).collect::<Vec<_>>().join("\n") + &format!("\nacc = q0;");
        (pre, body)
    };
    let mut rows: Vec<(String, String, String, &str, u32, u32)> = Vec::new(); // name, pre, body, barrier, warps, n
    for w in [32u32, 8, 1] {
        rows.push((format!("empty phase, both barriers, {w} warps"), String::new(), String::new(), both, w, 0));
    }
    rows.push(("empty phase, threadgroup barrier only, 32 warps".into(), String::new(), String::new(), tgb, 32, 0));
    for w in [32u32, 8, 1] {
        for n in [32usize, 128] {
            rows.push((format!("{n} dependent fma, {w} warps"), String::new(), fma(n), tgb, w, 0));
        }
    }
    rows.push(("4 chains of 32 fma one after another, 32 warps".into(), String::new(), seq4(32), tgb, 32, 0));
    rows.push(("4 chains of 32 fma interleaved (float4), 32 warps".into(), String::new(), int4(32), tgb, 32, 0));
    rows.push(("16 chains of 32 fma interleaved (4 float4), 32 warps".into(), "float4 v1 = v + 1.0f, v2 = v + 2.0f, v3 = v + 3.0f;".into(),
        (0..32).map(|i| format!("v = fma(v, float4(0.999{0}f), float4(0.001f)); v1 = fma(v1, float4(0.999{0}f), float4(0.001f)); v2 = fma(v2, float4(0.999{0}f), float4(0.001f)); v3 = fma(v3, float4(0.999{0}f), float4(0.001f));", i % 7)).collect::<Vec<_>>().join("\n") + "\nv = v + v1 + v2 + v3;", tgb, 32, 0));
    for w in [32u32, 8] {
        for n in [8usize, 32] {
            // Independent threadgroup loads (indices known up front), summed.
            let body = (0..n).map(|i| format!("acc += tg[(ix + {}u) % 6144u];", i * 191)).collect::<Vec<_>>().join("\n");
            rows.push((format!("{n} independent threadgroup loads, {w} warps"), String::new(), body, tgb, w, 0));
        }
    }
    {
        let body = (0..8).map(|_| "ix = uint(tg[ix % 6144u]) + ix * 7u + 1u;".to_string()).collect::<Vec<_>>().join("\n");
        rows.push(("8 dependent threadgroup loads, 32 warps".into(), String::new(), body, tgb, 32, 0));
        let body = (0..8).map(|i| format!("tg[(lid * 6u + {i}u) % 6144u] = acc + {i}.0f;")).collect::<Vec<_>>().join("\n");
        rows.push(("8 threadgroup stores, 32 warps".into(), String::new(), body, tgb, 32, 0));
        let body = (0..8).map(|i| format!("{{ float3 t = float3(tg[(ix + {0}u) % 6141u], tg[(ix + {0}u) % 6141u + 1u], tg[(ix + {0}u) % 6141u + 2u]); acc += t.x + t.y + t.z; }}", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push(("8 threadgroup vec3 loads (24 floats), 32 warps".into(), String::new(), body, tgb, 32, 0));
        rows.push(("1 threadgroup atomic add, 32 warps".into(), String::new(), "atomic_fetch_add_explicit(&tga[lid & 255u], 1u, memory_order_relaxed);".into(), tgb, 32, 0));
        rows.push(("1 threadgroup atomic add, one word, 32 warps".into(), String::new(), "atomic_fetch_add_explicit(&tga[0], 1u, memory_order_relaxed);".into(), tgb, 32, 0));
    }
    for w in [32u32, 8] {
        for n in [1usize, 4, 16] {
            let body = (0..n).map(|i| format!("acc += dev[(ix + {}u) & 65535u].y;", i * 4099)).collect::<Vec<_>>().join("\n");
            rows.push((format!("{n} independent device loads (1 MB table), {w} warps"), String::new(), body, both, w, 0));
        }
    }
    {
        let body = (0..4).map(|_| "ix = as_type<uint>(dev[ix & 65535u].x);".to_string()).collect::<Vec<_>>().join("\n");
        rows.push(("4 dependent device loads, 32 warps".into(), String::new(), body, both, 32, 0));
        let body = (0..4).map(|i| format!("acc += dev[lid * 4u + {i}u].y;")).collect::<Vec<_>>().join("\n");
        rows.push(("4 streamed device loads (own records), 32 warps".into(), String::new(), body, both, 32, 0));
        let body = (0..4).map(|i| format!("dev[lid * 4u + {i}u].z = acc;")).collect::<Vec<_>>().join("\n");
        rows.push(("4 device stores (own records), 32 warps".into(), String::new(), body, both, 32, 0));
        rows.push(("1 device atomic add (own word), 32 warps".into(), String::new(), "atomic_fetch_add_explicit(&da[lid], 1u, memory_order_relaxed);".into(), both, 32, 0));
        rows.push(("4 device atomic adds (scattered), 32 warps".into(), String::new(), (0..4).map(|i| format!("atomic_fetch_add_explicit(&da[(ix + {}u) & 4095u], 1u, memory_order_relaxed);", i * 977)).collect::<Vec<_>>().join("\n"), both, 32, 0));
    }
    for n in [16usize, 48, 80, 112, 160] {
        let (pre, body) = regs(n);
        rows.push((format!("{n} live registers, each updated a phase, 32 warps"), pre, body, tgb, 32, 0));
    }
    // Work in half the warps: does an idle warp cost?
    rows.push(("128 dependent fma in 16 of 32 warps".into(), String::new(), fma(128), tgb, 16, 0));
    // Threadgroup loads with no modulo in the index (a mask): what a load itself costs.
    for n in [8usize, 32] {
        let body = (0..n).map(|i| format!("acc += tg[(ix + {}u) & 4095u];", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push((format!("{n} threadgroup loads, masked index, 32 warps"), String::new(), body, tgb, 32, 0));
    }
    {
        let body = (0..8).map(|i| format!("{{ uint o = (ix + {}u) & 4088u; acc += tg[o] + tg[o + 1u] + tg[o + 2u]; }}", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push(("8 threadgroup loads of 3 adjacent floats, masked index, 32 warps".into(), String::new(), body, tgb, 32, 0));
        let body = (0..8).map(|i| format!("{{ uint o = (ix + {}u) & 4088u; acc += tg[o] + tg[o + 1u] + tg[o + 2u] + tg[o + 3u] + tg[o + 4u] + tg[o + 5u]; }}", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push(("8 threadgroup loads of 6 adjacent floats, masked index, 32 warps".into(), String::new(), body, tgb, 32, 0));
        let body = (0..8).map(|i| format!("acc += float((ix + {}u) & 4095u) * 1.0e-6f;", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push(("8 masked index computations alone (no load), 32 warps".into(), String::new(), body, tgb, 32, 0));
        let body = (0..8).map(|i| format!("acc += float((ix + {}u) % 6144u) * 1.0e-6f;", i * 191)).collect::<Vec<_>>().join("\n");
        rows.push(("8 modulo index computations alone (no load), 32 warps".into(), String::new(), body, tgb, 32, 0));
    }
    // A branch the lanes of a warp disagree on (odd and even lanes), against one they agree on.
    let two = |cond: &str| format!("if ({cond}) {{ {} }} else {{ {} }}", fma(64), fma(64).replace("0.001f", "0.002f"));
    rows.push(("64 fma in one of two branches, warps agree".into(), String::new(), two("(lid & 32u) != 0u"), tgb, 32, 0));
    rows.push(("64 fma in one of two branches, lanes of a warp disagree".into(), String::new(), two("(lid & 1u) != 0u"), tgb, 32, 0));
    // A loop shaped like a pair row: a point a pass (two vector loads at the point's own
    // address, two plane distances, a rare branch); the lanes of a warp agreeing on the
    // count (8) or not (4 to 11, so the warp runs 11), and with 48 more values kept alive.
    let row = |count: &str| format!(
        "uint n_ = {count}; float3 a0 = float3(acc, 0.5f, 0.25f), a1 = float3(0.3f, acc, 0.1f), a2 = float3(0.2f, 0.1f, acc), a3 = float3(acc, acc, 0.5f);\nfor (uint k = 0; k < 12u; k++) {{ if (k >= n_) break; uint h = (ix + k * 191u) & 1023u; float3 pp = float3(tg[h * 3u], tg[h * 3u + 1u], tg[h * 3u + 2u]); float3 qq = float3(tg[3072u + h * 3u], tg[3073u + h * 3u], tg[3074u + h * 3u]); float d1 = dot(pp - a0, a1); float d0 = dot((pp - qq * 0.0005f) - a2, a3); if (d1 * d0 < -1.0e30f) atomic_fetch_add_explicit(&tga[0], 1u, memory_order_relaxed); acc += (d1 + d0) * 1.0e-9f; }}"
    );
    rows.push(("row-like loop, 8 points a lane, lanes agree".into(), String::new(), row("8u"), tgb, 32, 0));
    rows.push(("row-like loop, 4 to 11 points a lane (a warp runs 11)".into(), String::new(), row("4u + (lid & 7u)"), tgb, 32, 0));
    {
        let (pre, body) = regs(48);
        rows.push(("row-like loop, 8 points, with 48 more live registers updated".into(), pre, format!("{body}\n{}", row("8u")), tgb, 32, 0));
    }
    // sqrt and divide.
    rows.push(("16 dependent sqrt, 32 warps".into(), String::new(), (0..16).map(|_| "acc = sqrt(acc + 1.5f);".to_string()).collect::<Vec<_>>().join("\n"), tgb, 32, 0));
    rows.push(("16 dependent divides, 32 warps".into(), String::new(), (0..16).map(|_| "acc = 1.5f / (acc + 0.25f);".to_string()).collect::<Vec<_>>().join("\n"), tgb, 32, 0));
    println!("{groups} threadgroup(s) of 1,024 lanes; µs a phase (fast math on / off)");
    let only = std::env::args().skip(1).find(|a| a.starts_with("only=")).map(|a| a[5..].to_string());
    for (name, pre, body, barrier, warps, n) in rows {
        if only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
            continue;
        }
        let src = kernel(&body, &pre, barrier);
        let mut cells = Vec::new();
        for fast in [true, false] {
            let pso = match gpu.pipeline(&src, "bench", fast) {
                Ok(p) => p,
                Err(e) => {
                    cells.push(format!("compile: {}", e.to_string().lines().next().unwrap_or("")));
                    continue;
                }
            };
            if pso.max_total_threads_per_threadgroup() < 1024 {
                cells.push(format!("max {} lanes", pso.max_total_threads_per_threadgroup()));
                continue;
            }
            if compile_only {
                cells.push(format!("ok ({} B)", pso.static_threadgroup_memory_length()));
                continue;
            }
            let run = |r: u32| {
                let p = gpu.buffer(&[r, warps, n, 0u32]);
                let mut ms: Vec<f64> = (0..5).map(|_| gpu.run(&pso, &[&p, &dev, &da, &out], groups, 1024)).collect();
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ms[2]
            };
            let (a, b) = (run(rounds), run(0));
            cells.push(format!("{:.3}", (a - b) * 1000.0 / rounds as f64));
        }
        println!("  {name:58} {}", cells.join(" / "));
    }
    let _ = read::<f32>(&out, 4);
    Ok(())
}
