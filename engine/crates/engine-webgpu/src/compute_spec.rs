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
    /// The module's name (a loading card's progress line).
    pub module: &'static str,
    /// The WGSL exactly as passed to `create_shader_module`.
    pub wgsl: String,
    pub entry: &'static str,
    /// The single bind group's layout entries; `None`: the automatic layout.
    pub layout: Option<Vec<wgpu::BindGroupLayoutEntry>>,
    /// Not needed for the first frame (the host creates it at first use): a browser
    /// host compiles it after the others, without waiting for it.
    pub deferred: bool,
}

/// Pipelines compiled ahead (a browser host's `createComputePipelineAsync`): a host
/// that compiles deferred specs turns the gate on and marks each module when its
/// compile has finished; until then a path that only saves time (self-collision split,
/// [`crate::ExplicitBeamsGpu::set_self_split`]) keeps the path it replaces (the same
/// results), so the page never waits on a synchronous compile of a deferred kernel.
/// Without the gate (native hosts) every module counts as ready.
mod ready {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    static GATE: AtomicBool = AtomicBool::new(false);
    static DONE: Mutex<Vec<String>> = Mutex::new(Vec::new());

    /// Gate the deferred paths on their compiles ([`mark_pipeline_compiled`]).
    pub fn set_pipeline_gate(on: bool) {
        GATE.store(on, Ordering::Relaxed);
    }

    /// Module `module`'s async compile has finished.
    pub fn mark_pipeline_compiled(module: &str) {
        let mut d = DONE.lock().unwrap_or_else(|e| e.into_inner());
        if !d.iter().any(|m| m == module) {
            d.push(module.to_string());
        }
    }

    /// Whether a host compiles pipelines ahead ([`set_pipeline_gate`]): a browser's.
    pub fn pipeline_gated() -> bool {
        GATE.load(Ordering::Relaxed)
    }

    /// Whether `module` can be created without waiting on its compile.
    pub fn pipeline_ready(module: &str) -> bool {
        !GATE.load(Ordering::Relaxed) || DONE.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|m| m == module)
    }
}
pub use ready::{mark_pipeline_compiled, pipeline_gated, pipeline_ready, set_pipeline_gate};
