//! Does Metal with fast math off give the CPU's IEEE single results? a*b+c (no
//! contraction), a/b and sqrt(a) correctly rounded, over random and edge inputs.
use gpu_floor::gpu::{read, Gpu};

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void probe(device const float4* in [[buffer(0)]], device float4* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    float4 x = in[i];
    out[i] = float4(x.x * x.y + x.z, x.x / x.y, sqrt(fabs(x.x)), (x.x * x.y) - (x.z * x.w));
}
"#;

fn main() -> anyhow::Result<()> {
    let gpu = Gpu::new()?;
    let n = 1 << 22;
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        let e = ((s >> 40) % 60) as i32 - 30;
        let m = ((s >> 8) & 0xFFFFFF) as f32 / 16777216.0 + 0.5;
        let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
        sign * m * 2f32.powi(e)
    };
    let inp: Vec<[f32; 4]> = (0..n).map(|_| [rnd(), rnd(), rnd(), rnd()]).collect();
    let b_in = gpu.buffer(&inp);
    let pragma = format!("#pragma clang fp contract(off)\n{SRC}");
    for (name, src, fast) in [("off", SRC, false), ("off+pragma", pragma.as_str(), false), ("on", SRC, true), ("on+pragma", pragma.as_str(), true)] {
        let pso = match gpu.pipeline(src, "probe", fast) { Ok(p) => p, Err(e) => { println!("{name}: {e}"); continue } };
        let b_out = gpu.zeros(n * 16);
        gpu.run(&pso, &[&b_in, &b_out], (n / 256) as u64, 256);
        let out: Vec<[f32; 4]> = read(&b_out, n);
        let mut bad = [0usize; 4];
        for (x, o) in inp.iter().zip(&out) {
            let want = [x[0] * x[1] + x[2], x[0] / x[1], x[0].abs().sqrt(), (x[0] * x[1]) - (x[2] * x[3])];
            for k in 0..4 {
                if want[k].to_bits() != o[k].to_bits() && !(want[k].is_nan() && o[k].is_nan()) {
                    bad[k] += 1;
                }
            }
        }
        println!("fast_math {name}: of {n}: a*b+c differs {}, a/b {}, sqrt {}, a*b-c*d {}", bad[0], bad[1], bad[2], bad[3]);
    }
    Ok(())
}
