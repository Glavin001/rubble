//! The crush law on the GPU (`crush.slang` `crush_step`) against its f32 mirror on the CPU
//! (stress-ref crush.rs `crush_step_f32`): the same damage, invariants and verdict over
//! compressive, confined, tensile and sheared states of the presets.

use stress_gpu::gpu::{Binding, Gpu};
use stress_ref::crush::{crush_step_f32, presets, sym_outer};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TestParams {
    count: u32,
    pad: [u32; 3],
}

#[test]
fn gpu_crush_step_matches_the_cpu_law() {
    let gpu = Gpu::new().expect("GPU");
    let mut cases: Vec<([f32; 6], f32, f32, [f32; 5], f32)> = Vec::new();
    for name in ["concrete", "brick_masonry", "glass", "softwood", "gypsum"] {
        let m = presets::preset(name).unwrap();
        let mf = [m.cap_pressure as f32, m.cohesion as f32, m.friction_slope as f32, m.crush_energy as f32, m.crush_viscosity as f32];
        for (i, s) in [-50e6, -1e6, 0.5e6, 3e6, 10e6, 30e6, 60e6, 200e6, 2e9].iter().enumerate() {
            // Uniaxial along a tilted axis, plus a shear and a confining part.
            let n = [0.3, -0.5, 0.81f64];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = n.map(|x| x / len);
            let a = sym_outer(n.map(|x| 0.5 * x), n.map(|x| -s * x));
            let b = sym_outer(n.map(|x| -0.5 * x), n.map(|x| s * x));
            let shear = sym_outer([0.5, 0.0, 0.0], [0.0, 0.2 * s, 0.0]);
            let confine = (i % 3) as f64 * 0.3 * s;
            let v: [f32; 6] = std::array::from_fn(|k| (a[k] + b[k] + shear[k] - if k < 3 { confine } else { 0.0 }) as f32);
            for (volume, dt, damage) in [(1.0f32, 1.0 / 60.0, 0.0), (0.064, 1e-3, 0.4), (0.5, 0.25, 0.99)] {
                cases.push((v, volume, dt, mf, damage));
            }
        }
    }
    let n = cases.len();
    let inputs: Vec<[f32; 4]> = cases
        .iter()
        .flat_map(|(v, vol, dt, m, d)| [[v[0], v[1], v[2], *vol], [v[3], v[4], v[5], *dt], [m[0], m[1], m[2], m[3]], [m[4], *d, 0.0, 0.0]])
        .collect();
    let buffers = [gpu.uniform("params", &TestParams { count: n as u32, pad: [0; 3] }), gpu.storage("inputs", &inputs), gpu.storage("outputs", &vec![[0f32; 4]; n])];
    let kernel = gpu.compute(&stress_gpu::shaders::CRUSH_TEST, "crush_step_test", &[Binding::Uniform, Binding::Storage, Binding::StorageRw]);
    let refs: Vec<&wgpu::Buffer> = buffers.iter().collect();
    let bind = gpu.bind(&kernel.group, &refs);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        kernel.dispatch(&mut pass, &bind, n);
    }
    gpu.queue.submit([encoder.finish()]);
    let out: Vec<[f32; 4]> = gpu.read(&buffers[2]);
    let mut crushed = 0;
    let mut grew = 0;
    for (i, (v, vol, dt, m, d)) in cases.iter().enumerate() {
        let (cd, cp, cq, cc) = crush_step_f32(*v, *vol, *dt, *m, *d);
        let g = out[i];
        let close = |a: f32, b: f32| (a - b).abs() <= 2e-5 * a.abs().max(b.abs()) + 1e-30;
        assert!(close(g[1], cp) && close(g[2], cq), "case {i}: p, q GPU {} {} CPU {cp} {cq}", g[1], g[2]);
        assert!(close(g[0], cd), "case {i}: damage GPU {} CPU {cd}", g[0]);
        assert_eq!(g[3] != 0.0, cc, "case {i}: crushed");
        crushed += cc as usize;
        grew += (cd > *d && !cc) as usize;
    }
    // The cases span all three outcomes.
    assert!(crushed > 0 && grew > 0 && crushed + grew < n, "{crushed} crushed, {grew} damaged of {n}");
}
