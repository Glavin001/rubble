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

/// Loads every fracture kernel on the device (a context of this thread's own on device
/// 0): each module's entry, its `{entry}_gs`, and its `SLANG_globalParams` no smaller
/// than the reflection says. The first check a build's device code gets before a host
/// runs it. Returns the modules loaded; an error names the first that does not load.
pub fn load_all() -> anyhow::Result<usize> {
    use cudarc::driver::result;
    use std::ffi::{c_void, CString};
    let map = |e: cudarc::driver::DriverError| anyhow::anyhow!("CUDA: {e:?}");
    result::init().map_err(map)?;
    let dev = result::device::get(0).map_err(map)?;
    let ctx = unsafe { result::primary_ctx::retain(dev) }.map_err(map)?;
    unsafe { result::ctx::set_current(ctx) }.map_err(map)?;
    let mut loaded = 0;
    for k in CUDA_FRACTURE_KERNELS {
        let name = format!("{}.{}", k.module, k.entry);
        anyhow::ensure!(
            !k.image.is_empty(),
            "{name}: no device code (the build had no nvcc)"
        );
        let module = unsafe { result::module::load_data(k.image.as_ptr() as *const c_void) }
            .map_err(|e| anyhow::anyhow!("{name}: load: {e:?}"))?;
        for function in [k.entry.to_string(), format!("{}_gs", k.entry)] {
            unsafe { result::module::get_function(module, CString::new(function.clone())?) }
                .map_err(|e| anyhow::anyhow!("{name}: function {function}: {e:?}"))?;
        }
        let (_, bytes) =
            unsafe { result::module::get_global(module, CString::new("SLANG_globalParams")?) }
                .map_err(|e| anyhow::anyhow!("{name}: SLANG_globalParams: {e:?}"))?;
        anyhow::ensure!(
            bytes >= k.params_bytes as usize,
            "{name}: SLANG_globalParams is {bytes} bytes, reflection says {}",
            k.params_bytes
        );
        unsafe { result::module::unload(module) }
            .map_err(|e| anyhow::anyhow!("{name}: unload: {e:?}"))?;
        loaded += 1;
    }
    Ok(loaded)
}
