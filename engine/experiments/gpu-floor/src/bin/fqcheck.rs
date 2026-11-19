//! The converted closest-feature queries against the engine's own
//! (`engine_schema::feature_query`), bit for bit, on random cases near contact: the
//! Metal source compiled as C++ (`emu/shim.h`), no GPU. Triangles, beams and points
//! against boxes and spheres.
//!   fqcheck [cases] [--gpu | --compile]
//! `--gpu` runs the cases through Metal instead (under the GPU lock), `--compile` only
//! compiles the kernel.
use engine_softbody::explicit_surface::{near, point_contact, seg_near, segment_contact, separated, triangle_contact, SurfaceShape};
use glam::{Quat, Vec3};

const KERNEL: &str = r#"
static void put3(device uint* o, float3 v) { o[0] = as_type<uint>(v.x); o[1] = as_type<uint>(v.y); o[2] = as_type<uint>(v.z); }
static void fqCase(device const float* c, device uint* res);
kernel void fqk(device const float* in [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    fqCase(in + gid * 28u, out + gid * 24u);
}
// One query's latency: every lane of one warp runs the same n cases in turn, REPS times.
kernel void fqlat(device const float* in [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]])
{
    for (uint rep = 0u; rep < 16u; rep++)
        for (uint i = 0u; i < n; i++) fqCase(in + i * 28u, out + (lid * n + i) * 24u);
}
static void fqCase(device const float* c, device uint* res)
{
    FqShape s;
    s.kind = (uint)c[1]; s.hull = 0u; s.nv = 0u; s.nf = 0u; s.ne = 0u; s.moved = 0u; s.off = float3(0.0f); s.b = float3(0.0f);
    const float r = c[2];
    s.c = float3(c[4], c[5], c[6]);
    s.ax0 = float3(c[7], c[8], c[9]); s.ax1 = float3(c[10], c[11], c[12]); s.ax2 = float3(c[13], c[14], c[15]);
    s.hext = float3(c[16], c[17], c[18]);
    const float3 x0 = float3(c[19], c[20], c[21]), x1 = float3(c[22], c[23], c[24]), x2 = float3(c[25], c[26], c[27]);
    float3 lo, hi, w, nn, q, w2, q2;
    float g = 0.0f;
    bool two = false;
    fqBounds(s, lo, hi);
    const uint kind = (uint)c[0];
    if (kind == 0u) {
        res[0] = fqNear(x0, x1, x2, r, lo, hi);
        res[1] = fqSeparated(s, x0, x1, x2, r);
        if (fqContact(s, x0, x1, x2, r, w, nn, g, q)) {
            res[2] = 1u; put3(res + 4, w); put3(res + 7, nn); res[10] = as_type<uint>(g); put3(res + 11, q);
        }
    } else if (kind == 1u) {
        res[0] = fqSegNear(x0, x1, r, lo, hi);
        res[1] = fqSeparated(s, x0, x1, x1, r);
        if (fqSegContact(s, x0, x1, r, w, nn, g, q, two, w2, q2)) {
            res[2] = 1u; put3(res + 4, w); put3(res + 7, nn); res[10] = as_type<uint>(g); put3(res + 11, q);
            if (two) { res[3] = 1u; put3(res + 14, w2); put3(res + 17, q2); }
        }
    } else {
        res[0] = fqSegNear(x0, x0, r, lo, hi);
        if (fqPointContact(s, x0, r, nn, g)) { res[2] = 1u; put3(res + 7, nn); res[10] = as_type<uint>(g); }
    }
}
"#;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = args.iter().any(|a| a == "--gpu");
    let compile = args.iter().any(|a| a == "--compile");
    let n: usize = args.iter().find(|a| !a.starts_with("--")).map_or(20000, |x| x.parse().unwrap());
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/gpu-floor/emu-fq");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("fq.inc"), gpu_floor::floor::fq_msl())?;
    let emu = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("emu");
    let exe = dir.join("fq");
    let out = std::process::Command::new("xcrun")
        .args(["clang++", "-std=c++20", "-O1", "-ffp-contract=off", "-w", "-I"])
        .arg(&dir)
        .arg("-I")
        .arg(&emu)
        .arg(emu.join("fq_main.cpp"))
        .arg("-o")
        .arg(&exe)
        .output()?;
    anyhow::ensure!(out.status.success(), "compile:\n{}", String::from_utf8_lossy(&out.stderr));
    let mut seed = 0x1234_5678u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0
    };
    // A case: kind (0 triangle, 1 beam, 2 point), shape kind (0 box, 4 sphere), 28 floats.
    let mut input: Vec<f32> = Vec::new();
    let mut cases = Vec::new();
    let r = 0.025f32;
    for k in 0..n {
        let c = Vec3::new(rnd() * 4.0 - 2.0, rnd() * 2.0, rnd() * 4.0 - 2.0);
        let sphere = k % 3 == 2;
        let q = if k % 5 == 0 { Quat::IDENTITY } else { Quat::from_euler(glam::EulerRot::XYZ, rnd() * 6.0, rnd() * 6.0, rnd() * 6.0).normalize() };
        let half = Vec3::new(0.05 + rnd() * 0.4, 0.05 + rnd() * 0.6, 0.05 + rnd() * 0.3);
        let axes = [q * Vec3::X, q * Vec3::Y, q * Vec3::Z];
        let shape = if sphere { SurfaceShape::Sphere { c, radius: half.x } } else { SurfaceShape::Box { c, axes, half } };
        // Element near the shape's surface.
        let reach = if sphere { half.x } else { half.length() };
        let mut p = || c + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * (2.0 * reach + 0.3);
        let base = p();
        let x = [base, base + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 0.6, base + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 0.6];
        let kind = (k % 7 % 3) as u32;
        input.extend([kind as f32, if sphere { 4.0 } else { 0.0 }, r, 0.0]);
        input.extend(c.to_array());
        for a in axes {
            input.extend(a.to_array());
        }
        input.extend(half.to_array());
        for v in x {
            input.extend(v.to_array());
        }
        cases.push((kind, shape, x));
    }
    std::fs::write(dir.join("in.bin"), bytemuck::cast_slice::<f32, u8>(&input))?;
    let got: Vec<u32> = if gpu || compile {
        // The same cases through Metal's compiler, strict (fast math off, no contraction).
        let g = gpu_floor::gpu::Gpu::new()?;
        let src = format!("#pragma clang fp contract(off)\n#include <metal_stdlib>\nusing namespace metal;\n{}\n{}", gpu_floor::floor::fq_msl(), KERNEL);
        let t = std::time::Instant::now();
        let pso = g.pipeline(&src, "fqk", false)?;
        println!("Metal compiled the queries in {:.1} s", t.elapsed().as_secs_f64());
        if compile {
            return Ok(());
        }
        let inb = g.buffer(&input);
        let outb = g.zeros(n * 24 * 4);
        let nb = g.buffer(&[n as u32]);
        let ms = g.run(&pso, &[&inb, &outb, &nb], n.div_ceil(64) as u64, 64);
        println!("GPU: {n} cases in {ms:.2} ms");
        gpu_floor::gpu::read::<u32>(&outb, n * 24)
    } else {
        let run = std::process::Command::new(&exe).arg(dir.join("in.bin")).arg(dir.join("out.bin")).output()?;
        anyhow::ensure!(run.status.success(), "run: {}", String::from_utf8_lossy(&run.stderr));
        std::fs::read(dir.join("out.bin"))?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    // Per case 24 words: near, separated, hit, two, w(3), n(3), gap, q(3), w2(3), q2(3), pad.
    let (mut bad, mut hits, mut seps, mut nears) = (0, 0, 0, 0);
    // (For --latency: the cases by element, shape and outcome.)
    let mut cats: std::collections::BTreeMap<(u32, bool, &str), Vec<usize>> = Default::default();
    for (i, (kind, shape, x)) in cases.iter().enumerate() {
        let g = &got[i * 24..i * 24 + 24];
        let mut want = [0u32; 24];
        let (lo, hi) = engine_schema::feature_query::bounds(shape);
        let bits = |v: Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
        match kind {
            0 => {
                want[0] = near(x, r, lo, hi) as u32;
                want[1] = separated(&x[..], shape, r) as u32;
                if let Some(h) = triangle_contact(x, shape, r) {
                    want[2] = 1;
                    want[4..7].copy_from_slice(&[h.w[0].to_bits(), h.w[1].to_bits(), h.w[2].to_bits()]);
                    want[7..10].copy_from_slice(&bits(h.n));
                    want[10] = h.gap.to_bits();
                    want[11..14].copy_from_slice(&bits(h.q));
                }
            }
            1 => {
                let s = [x[0], x[1]];
                want[0] = seg_near(&s, r, lo, hi) as u32;
                want[1] = separated(&s[..], shape, r) as u32;
                if let Some((h, second)) = segment_contact(&s, shape, r) {
                    want[2] = 1;
                    want[4..7].copy_from_slice(&[h.w[0].to_bits(), h.w[1].to_bits(), h.w[2].to_bits()]);
                    want[7..10].copy_from_slice(&bits(h.n));
                    want[10] = h.gap.to_bits();
                    want[11..14].copy_from_slice(&bits(h.q));
                    if let Some(h2) = second {
                        want[3] = 1;
                        want[14..17].copy_from_slice(&[h2.w[0].to_bits(), h2.w[1].to_bits(), h2.w[2].to_bits()]);
                        want[17..20].copy_from_slice(&bits(h2.q));
                    }
                }
            }
            _ => {
                want[0] = seg_near(&[x[0], x[0]], r, lo, hi) as u32;
                if let Some(h) = point_contact(x[0], shape, r) {
                    want[2] = 1;
                    want[7..10].copy_from_slice(&bits(h.n));
                    want[10] = h.gap.to_bits();
                }
            }
        }
        let outcome = if want[2] != 0 {
            if f32::from_bits(want[10]) > engine_softbody::explicit_surface::PIERCE { "hit, apart" } else { "hit, touching or crossing" }
        } else if want[0] == 0 {
            "not near"
        } else if want[1] != 0 {
            "separated (the cull)"
        } else {
            "near, no contact"
        };
        cats.entry((*kind, matches!(shape, SurfaceShape::Sphere { .. }), outcome)).or_default().push(i);
        nears += want[0];
        seps += want[1];
        hits += want[2];
        let same = if *kind == 2 { g[0] == want[0] && g[2] == want[2] && g[7..11] == want[7..11] } else { g[..20] == want[..20] };
        if !same {
            bad += 1;
            if bad <= 6 {
                println!("case {i} kind {kind} {shape:?}\n  x {x:?}\n  got  {:x?}\n  want {:x?}", &g[..20], &want[..20]);
            }
        }
    }
    println!("{n} cases ({nears} near, {seps} separated by the cull, {hits} hits): {bad} differ");
    anyhow::ensure!(bad == 0);
    if args.iter().any(|a| a == "--latency") {
        // One query's latency on the GPU by element, shape and outcome: one warp, every
        // lane the same 32 cases in turn 16 times (the surface pass's narrow stage is a lane
        // a pair, so a phase costs its slowest query, not their sum).
        let g = gpu_floor::gpu::Gpu::new()?;
        let src = format!("#pragma clang fp contract(off)\n#include <metal_stdlib>\nusing namespace metal;\n{}\n{}", gpu_floor::floor::fq_msl(), KERNEL);
        for (fast, name) in [(false, "strict"), (true, "fast math")] {
            let src = if fast { src.replace("#pragma clang fp contract(off)\n", "") } else { src.clone() };
            let pso = g.pipeline(&src, "fqlat", fast)?;
            println!("latency, {name} (microseconds a query; one warp, 32 cases x 16):");
            for ((kind, sphere, outcome), ids) in &cats {
                if ids.len() < 32 {
                    continue;
                }
                let mut inp: Vec<f32> = Vec::new();
                for &i in &ids[..32] {
                    inp.extend_from_slice(&input[i * 28..i * 28 + 28]);
                }
                let (inb, outb, nb) = (g.buffer(&inp), g.zeros(32 * 32 * 24 * 4), g.buffer(&[32u32]));
                let mut ms: Vec<f64> = (0..7).map(|_| g.run(&pso, &[&inb, &outb, &nb], 1, 32)).collect();
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("  {:8} vs {:6} {:28} {:7.2}", ["triangle", "beam", "point"][*kind as usize], if *sphere { "sphere" } else { "box" }, outcome, ms[3] * 1e3 / (32.0 * 16.0));
            }
        }
    }
    Ok(())
}
