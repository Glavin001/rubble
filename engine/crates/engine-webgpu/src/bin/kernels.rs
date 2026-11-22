//! The kernels of this build ready to run, and where they are: with the kernels compiled
//! at run time (the fast profile on native) this compiles the store for the current Slang
//! sources (what the first run after a kernel edit would do), so a timed or GPU-locked run
//! does not; with them embedded it prints the build's WGSL directory.
fn main() {
    println!("{}", engine_webgpu::explicit_beams_wgsl_dir().display());
}
