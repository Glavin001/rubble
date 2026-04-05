//! `rubble-gpu` — thin wgpu compute abstraction for the rubble physics engine.

mod buffer;
mod context;
mod kernel;
mod multi_gpu;

pub use buffer::{GpuAtomicCounter, GpuBuffer, PingPongBuffer};
#[cfg(test)]
pub use context::test_gpu;
pub use context::GpuContext;
pub use kernel::{round_up_workgroups, ComputeKernel};
pub use multi_gpu::{GpuDevice, GpuDevicePool, MultiGpuBuffer, MultiGpuContext, WorkDistribution};
pub use web_time;

/// Transitional breakdown of broadphase wall time.
///
/// These buckets are intentionally coarse so the viewer can distinguish
/// scene-bounds work, sorting, structure build/staging, traversal, and
/// host/device synchronization while rubble migrates to an all-GPU broadphase.
#[derive(Clone, Copy, Debug, Default)]
pub struct BroadphaseBreakdownMs {
    pub bounds_ms: f32,
    pub sort_ms: f32,
    pub build_ms: f32,
    pub traverse_ms: f32,
    pub readback_ms: f32,
}

impl BroadphaseBreakdownMs {
    pub fn as_array(&self) -> [f32; 5] {
        [
            self.bounds_ms,
            self.sort_ms,
            self.build_ms,
            self.traverse_ms,
            self.readback_ms,
        ]
    }

    pub fn total_ms(&self) -> f32 {
        self.bounds_ms + self.sort_ms + self.build_ms + self.traverse_ms + self.readback_ms
    }

    pub fn is_zero(&self) -> bool {
        self.total_ms() <= f32::EPSILON
    }

    /// Account for wall time spent in [`GpuLbvh::read_pair_count`] / `read_pair_count_async`.
    ///
    /// On **WebGPU**, `device.poll` does not wait, so the counter map often blocks until all
    /// prior GPU broadphase work completes. Split `elapsed_ms` so most of it lands in
    /// **`traverse_ms`** (GPU wait) and a small fixed allowance stays in **`readback_ms`**
    /// (host copy/map of 4 bytes). On native backends, the full elapsed time goes to
    /// **`readback_ms`** (after `wait_for_queue`, that is mostly the staging read).
    pub fn add_pair_counter_read_elapsed_ms(&mut self, elapsed_ms: f32) {
        #[cfg(target_arch = "wasm32")]
        {
            const HOST_COPY_MS: f32 = 0.15;
            let host = elapsed_ms.min(HOST_COPY_MS);
            let gpu_wait = (elapsed_ms - host).max(0.0);
            self.traverse_ms += gpu_wait;
            self.readback_ms += host;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.readback_ms += elapsed_ms;
        }
    }
}

/// Wall-clock timings (milliseconds) for each phase of a physics step.
///
/// Field order matches the array returned by `last_step_timings_ms()` in WASM:
///   [0] upload_ms, [1] predict_aabb_ms, [2] broadphase_ms,
///   [3] narrowphase_ms, [4] contact_fetch_ms, [5] solve_ms, [6] extract_ms
#[derive(Clone, Copy, Debug)]
pub struct StepTimingsMs {
    pub upload_ms: f32,
    pub predict_aabb_ms: f32,
    pub broadphase_ms: f32,
    pub broadphase_breakdown: BroadphaseBreakdownMs,
    pub narrowphase_ms: f32,
    pub contact_fetch_ms: f32,
    pub solve_ms: f32,
    pub extract_ms: f32,
}

impl Default for StepTimingsMs {
    fn default() -> Self {
        Self {
            upload_ms: 0.0,
            predict_aabb_ms: 0.0,
            broadphase_ms: 0.0,
            broadphase_breakdown: BroadphaseBreakdownMs::default(),
            narrowphase_ms: 0.0,
            contact_fetch_ms: 0.0,
            solve_ms: 0.0,
            extract_ms: 0.0,
        }
    }
}

impl StepTimingsMs {
    pub fn set_broadphase_breakdown(&mut self, breakdown: BroadphaseBreakdownMs) {
        self.broadphase_breakdown = breakdown;
        self.broadphase_ms = breakdown.total_ms();
    }

    pub fn as_array(&self) -> [f32; 7] {
        [
            self.upload_ms,
            self.predict_aabb_ms,
            self.broadphase_ms,
            self.narrowphase_ms,
            self.contact_fetch_ms,
            self.solve_ms,
            self.extract_ms,
        ]
    }
}

/// Errors that can occur when initialising GPU resources.
#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
}

/// Yield to the JavaScript event loop (WASM only).
/// Required for async GPU buffer mapping in WebGPU.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn yield_now() {
    use wasm_bindgen::JsCast;
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let global = js_sys::global();
        let set_timeout: js_sys::Function = js_sys::Reflect::get(&global, &"setTimeout".into())
            .unwrap()
            .dyn_into()
            .unwrap();
        set_timeout.call2(&global, &resolve, &0.into()).unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}
