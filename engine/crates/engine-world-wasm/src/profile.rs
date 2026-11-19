//! The profilers' bindings (`tools/webgpu-profile`, the pages' profiler panel,
//! `engine-webgpu/profiler.mjs`): host spans as `performance.measure`s, debug labels on the
//! step's GPU work, and the kernel replay. All measurement: none changes the simulation.

use crate::WebWorld;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
    #[wasm_bindgen(js_namespace = performance, js_name = measure, catch)]
    fn performance_measure(name: &str, options: &JsValue) -> Result<JsValue, JsValue>;
}

fn now() -> f64 {
    performance_now()
}

/// A finished span as a `performance.measure` (Chrome's trace: User Timing, `blink.user_timing`).
fn measure(name: &'static str, start: f64, end: f64) {
    let o = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&o, &"start".into(), &start.into());
    let _ = js_sys::Reflect::set(&o, &"end".into(), &end.into());
    let _ = performance_measure(name, &o);
}

std::thread_local! {
    static REPLAY: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Record host spans (`engine_webgpu::host_profile`) on the page's clock; with `measures`
/// each also becomes a `performance.measure` (for traces; it costs a call per span).
#[wasm_bindgen(js_name = setHostProfile)]
pub fn set_host_profile(on: bool, measures: bool) {
    engine_webgpu::host_profile::set_clock(now);
    engine_webgpu::host_profile::set_sink(if on && measures { Some(measure) } else { None });
    engine_webgpu::host_profile::set_enabled(on);
}

/// The spans since the last call, as JSON `[{n, s, d, y}]` (name, start ms, ms, GPU-wait
/// suspensions inside it).
#[wasm_bindgen(js_name = hostSpans)]
pub fn host_spans() -> String {
    engine_webgpu::host_profile::drain_json()
}

#[wasm_bindgen]
impl WebWorld {
    /// Name the step's GPU work after the plan (passes `step:<phase>`, debug groups
    /// `phase/kernel`), for Chrome's GPU traces. The same work either way.
    #[wasm_bindgen(js_name = setDebugLabels)]
    pub fn set_debug_labels(&mut self, on: bool) {
        self.world.sim_mut().set_debug_labels(on);
    }

    /// The kernel replay of the next step (`WgpuSim::kernel_replay`): 1 when it ran (read
    /// [`Self::kernel_replay_result`]), 0 on an error (the result is `{"error": ...}`). It
    /// waits on the GPU, so call it through `WebAssembly.promising` (profiler.mjs
    /// `kernelReplay`), with the page's frame loop held: the world is borrowed throughout.
    #[wasm_bindgen(js_name = kernelReplayRun)]
    pub fn kernel_replay_run(&mut self, reps: u32) -> u32 {
        let opts = engine_webgpu::KernelReplayOptions {
            reps,
            whole_step_only: false,
        };
        let (ok, text) = match self.world.sim_mut().kernel_replay(opts) {
            Ok(r) => {
                let mut j = r.to_json();
                j["table"] = serde_json::Value::String(r.table(0));
                (1, j.to_string())
            }
            Err(e) => (0, serde_json::json!({ "error": format!("{e:#}") }).to_string()),
        };
        REPLAY.with(|r| *r.borrow_mut() = text);
        ok
    }

    /// The last kernel replay's report (JSON).
    #[wasm_bindgen(js_name = kernelReplayResult)]
    pub fn kernel_replay_result(&self) -> String {
        REPLAY.with(|r| r.borrow().clone())
    }
}
