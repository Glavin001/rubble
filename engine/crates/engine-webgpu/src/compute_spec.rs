//! A compute pipeline exactly as a host creates it (its WGSL, entry point and bind
//! group layout), so a browser host can compile it ahead with
//! `createComputePipelineAsync` before the synchronous creation.
//!
//! Why: in Chrome a synchronous `createComputePipeline` compiles on the GPU
//! process's main thread. A kernel whose Metal compile takes longer than Chrome's
//! GPU watchdog (the explicit solver's contact frame: about a minute cold on an
//! M3 Max) makes a windowed Chrome kill the GPU process, and every WebGPU device
//! is lost ("A valid external Instance reference no longer exists"). Headless
//! Chrome does not arm that watchdog. An async creation compiles on a worker
//! thread instead; once it finishes, the synchronous creation of the same
//! descriptor is a cache hit (Dawn's pipeline cache, then the Metal shader cache).
//! The pipelines are the same objects either way: nothing about the kernels changes.

/// One compute pipeline's descriptor, as plain data.
#[derive(Clone, Debug)]
pub struct ComputeSpec {
    pub label: &'static str,
    /// The WGSL exactly as passed to `create_shader_module`.
    pub wgsl: String,
    pub entry: &'static str,
    /// The single bind group's layout entries; `None`: the automatic layout.
    pub layout: Option<Vec<wgpu::BindGroupLayoutEntry>>,
}
