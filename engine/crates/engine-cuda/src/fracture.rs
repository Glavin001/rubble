//! The fracture world's kernels built for CUDA: the stress solver's shaders
//! (`engine/stress-oracle/crates/stress-gpu/shaders`) the World's destruction runs
//! (the contact ingest, the fracture frame and its split, crushing, the stress solve,
//! the change log), an entry a module, each with its `{entry}_gs` for the indirect
//! dispatches (engine/docs/world-on-cuda.md, C6).

/// One compute entry of a stress shader, built for CUDA (build.rs `fracture_kernels`).
pub struct FractureCudaKernel {
    /// The shader the entry is in (`fracture`, `coupling`, `blast_topology`, ...).
    pub module: &'static str,
    pub entry: &'static str,
    /// Bytes of the CUDA source Slang wrote.
    pub cu_bytes: u64,
    /// The fatbin (empty when the build had no nvcc).
    pub image: &'static [u8],
    /// The entry's workgroup size.
    pub threads: [u32; 3],
    /// Bytes of the module's `SLANG_globalParams`.
    pub params_bytes: u32,
    /// Each global: byte offset in `SLANG_globalParams`, field bytes (8: a uniform
    /// block's pointer; 16: a buffer's pointer and count), the module's binding.
    pub binds: &'static [(u32, u32, u32)],
}

include!(concat!(env!("OUT_DIR"), "/cuda_fracture_kernels.rs"));

/// The kernel `entry` of `module`.
pub fn kernel(module: &str, entry: &str) -> Option<&'static FractureCudaKernel> {
    CUDA_FRACTURE_KERNELS
        .iter()
        .find(|k| k.module == module && k.entry == entry)
}
