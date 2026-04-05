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

    /// WebGPU only: the first buffer map after broadphase often waits for **all** prior GPU
    /// work, while `device.poll` does not block. That wall time is accumulated in
    /// `readback_ms` even though it is mostly GPU pair-finding / tree work.
    ///
    /// Call this **only** after timing a **4-byte** pair-counter read following the
    /// on-device LBVH path — not after large `GpuBuffer::download` readbacks (e.g. compound
    /// AABB staging).
    #[cfg(target_arch = "wasm32")]
    pub fn reattribute_webgpu_pair_counter_wait_to_traverse(&mut self) {
        /// Typical host-side cost to map/copy the atomic counter (ms); the rest is GPU wait.
        const COUNTER_READ_HOST_MS: f32 = 0.1;
        let rb = self.readback_ms;
        if rb <= COUNTER_READ_HOST_MS {
            return;
        }
        self.traverse_ms += rb - COUNTER_READ_HOST_MS;
        self.readback_ms = COUNTER_READ_HOST_MS;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn reattribute_webgpu_pair_counter_wait_to_traverse(&mut self) {}
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
