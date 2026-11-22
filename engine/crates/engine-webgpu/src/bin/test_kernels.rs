//! Compiles the test-only explicit kernels (`engine_webgpu::test_kernels_dir`: the frozen
//! baseline and reference, the deterministic-hits and ablation builds) and prints their
//! directory. Tests and measurements compile them on first use; this is for tools
//! (engine/tools/build-time/kernel_digest.py) and to warm them before a timed run.
fn main() {
    println!("{}", engine_webgpu::test_kernels_dir().display());
}
