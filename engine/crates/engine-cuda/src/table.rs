//! The rigid plan's and the terrain plan's kernels as the build made them (build.rs).

pub struct CudaKernel {
    pub entry: &'static str,
    /// Bytes of Slang-emitted CUDA C++ (non-zero proves the source built).
    pub cu_bytes: u64,
    /// Fatbin: SASS for [`SASS_ARCHS`] plus PTX for [`PTX_ARCH`]; empty
    /// when the build host had no nvcc.
    pub image: &'static [u8],
    pub threads: [u32; 3],
    pub params_bytes: u32,
    /// The kernel body indexes `gIndirectArgs` (binding 23).
    pub writes_indirect: bool,
    /// (byte offset in `SLANG_globalParams`, field bytes 8|16, shared `BIND(slot)`)
    pub binds: &'static [(u32, u32, u32)],
}
include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));
include!(concat!(env!("OUT_DIR"), "/cuda_terrain_kernels.rs"));
