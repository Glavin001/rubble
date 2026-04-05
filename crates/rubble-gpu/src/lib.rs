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

/// Broadphase substage wall times (milliseconds), measured around GPU submits and host readbacks.
///
/// **WebGPU note:** `wgpu::Queue::submit` returns before work finishes. Buckets **Bounds** /
/// **Sort** / **Build** / **Traverse** here are mostly **CPU-side enqueue + recording** time,
/// not GPU execution time, unless the implementation explicitly waits (e.g. native
/// `GpuContext::wait_for_queue` after pair finding). **`readback_ms`** is **true host wall
/// time** for staging buffer mapping (e.g. pair counter, AABB download) and may include
/// waiting for prior GPU work when that map is the first synchronization point.
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
}

/// Labels for the seven top-level [`StepTimingsMs::as_array`] fields (name, lane).
pub const STEP_TIMING_LABELS: [(&str, &str); 7] = [
    ("Upload", "(CPU)"),
    ("Predict", "(GPU)"),
    ("Broadphase", "(GPU+CPU)"),
    ("Narrowphase", "(GPU)"),
    ("Contacts", "(GPU>CPU)"),
    ("Solve", "(GPU)"),
    ("Extract", "(GPU)"),
];

/// Labels for [`BroadphaseBreakdownMs::as_array`] (name, lane).
pub const BROADPHASE_SUB_LABELS: [(&str, &str); 5] = [
    ("Bounds", "(CPU)"),
    ("Sort", "(GPU)"),
    ("Build", "(CPU+GPU)"),
    ("Traverse", "(GPU)"),
    ("Readback", "(CPU)"),
];

/// Index of the broadphase row in [`STEP_TIMING_LABELS`] / [`StepTimingsMs::as_array`].
pub const STEP_INDEX_BROADPHASE: usize = 2;

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

    /// Monospace-friendly overlay: step total, per-stage ms and % of step, broadphase substages
    /// (always five lines, % of broadphase total), then render line. Single source of truth for
    /// dev overlays (web text, tests); native UI may use [`STEP_TIMING_LABELS`] for custom layout.
    pub fn format_text_overlay(&self, render_backend: &str, render_ms: f32) -> String {
        let arr = self.as_array();
        let total: f32 = arr.iter().sum();
        let mut lines = Vec::with_capacity(16);
        lines.push(format!("Step: {total:.2} ms"));

        for (i, &(name, lane)) in STEP_TIMING_LABELS.iter().enumerate() {
            let ms = arr[i];
            let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
            lines.push(format!(
                "  {name:<11} {lane:<8} {:>6.2} ms {:>5.1}%",
                ms, pct
            ));
            if i == STEP_INDEX_BROADPHASE {
                let bp = &self.broadphase_breakdown;
                let bp_arr = bp.as_array();
                let bp_total = bp.total_ms();
                for (j, &(sub_name, sub_lane)) in BROADPHASE_SUB_LABELS.iter().enumerate() {
                    let bms = bp_arr[j];
                    let bpct = if bp_total > 0.0 {
                        bms / bp_total * 100.0
                    } else {
                        0.0
                    };
                    lines.push(format!(
                        "    {sub_name:<9} {sub_lane:<10} {:>6.2} ms {:>5.1}%",
                        bms, bpct
                    ));
                }
            }
        }

        lines.push(format!(
            "Render      ({render_backend}) {:>6.2} ms",
            render_ms
        ));
        lines.join("\n")
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
